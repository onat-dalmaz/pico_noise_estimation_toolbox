"""Numerically explicit primitives for the factor/direct covariance comparator.

This module is intentionally separate from ``mr_recon.gfactor`` so the audit can
use proper complex-Gaussian normalization and retain signed direct-diagonal
estimates without changing production behavior.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import torch

from mr_recon.covariance_apply import apply_regularized_covariance


def _real_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.complex64:
        return torch.float32
    if dtype == torch.complex128:
        return torch.float64
    raise TypeError(f"Expected a complex dtype, received {dtype}")


def proper_complex_gaussian(
    shape: Sequence[int],
    *,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Draw ``(g1 + i g2)/sqrt(2)`` so ``E|z_j|^2 = 1``."""
    real_dtype = _real_dtype(dtype)
    g1 = torch.randn(
        tuple(shape), generator=generator, device=device, dtype=real_dtype
    )
    g2 = torch.randn(
        tuple(shape), generator=generator, device=device, dtype=real_dtype
    )
    # torch.complex instead of ``g1 + 1j * g2``: the complex-scalar product
    # raises "CUDA driver error: invalid argument" on the A6000 node (amalfi).
    return (torch.complex(g1, g2) / math.sqrt(2.0)).to(dtype)


def random_phase(
    shape: Sequence[int],
    *,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Draw iid unit-modulus complex phases ``exp(i theta)``."""
    theta = 2.0 * math.pi * torch.rand(
        tuple(shape),
        generator=generator,
        device=device,
        dtype=_real_dtype(dtype),
    )
    # torch.polar instead of ``torch.exp(1j * theta)`` (same values); see above.
    return torch.polar(torch.ones_like(theta), theta).to(dtype)


def direct_real_contribution(
    probe: torch.Tensor, covariance_probe: torch.Tensor
) -> torch.Tensor:
    """Return signed ``Re{conj(v) * (RR^H v)}``, never its modulus."""
    return (probe.conj() * covariance_probe).real.to(torch.float64)


def negative_diagonal_stats(estimate: torch.Tensor) -> dict[str, float | int]:
    """Summarize finite negative entries without modifying the estimate."""
    values = estimate.detach().to(torch.float64)
    finite_values = values[torch.isfinite(values)]
    negative = finite_values[finite_values < 0.0]
    finite_count = int(finite_values.numel())
    negative_count = int(negative.numel())
    if negative_count:
        minimum = float(negative.min().item())
        abs_mean = float(negative.abs().mean().item())
        abs_max = float(negative.abs().max().item())
        abs_sum = float(negative.abs().sum().item())
    else:
        minimum = 0.0
        abs_mean = 0.0
        abs_max = 0.0
        abs_sum = 0.0
    return {
        "finite_count": finite_count,
        "negative_count": negative_count,
        "negative_fraction": (
            negative_count / finite_count if finite_count else float("nan")
        ),
        "negative_min": minimum,
        "negative_abs_mean": abs_mean,
        "negative_abs_max": abs_max,
        "negative_abs_sum": abs_sum,
    }


def masked_nrmse(
    estimate: torch.Tensor,
    reference: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> float:
    """NRMSE on raw finite values; no clipping or rescaling is applied."""
    if mask is not None:
        estimate = estimate[mask]
        reference = reference[mask]
    finite = torch.isfinite(estimate) & torch.isfinite(reference)
    estimate = estimate[finite].to(torch.float64)
    reference = reference[finite].to(torch.float64)
    if estimate.numel() == 0:
        return float("nan")
    denominator = torch.linalg.vector_norm(reference)
    if float(denominator.item()) == 0.0:
        return float("nan")
    return float(
        (torch.linalg.vector_norm(estimate - reference) / denominator).item()
    )


def masked_correlation(
    estimate: torch.Tensor,
    reference: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> float:
    """Pearson correlation on raw finite values."""
    if mask is not None:
        estimate = estimate[mask]
        reference = reference[mask]
    finite = torch.isfinite(estimate) & torch.isfinite(reference)
    estimate = estimate[finite].to(torch.float64)
    reference = reference[finite].to(torch.float64)
    if estimate.numel() < 2:
        return float("nan")
    estimate = estimate - estimate.mean()
    reference = reference - reference.mean()
    denominator = (
        torch.linalg.vector_norm(estimate)
        * torch.linalg.vector_norm(reference)
    )
    if float(denominator.item()) == 0.0:
        return float("nan")
    return float((torch.sum(estimate * reference) / denominator).item())


@dataclass
class PhysicalReconstruction:
    """Physical reconstruction ``R=(A^H A+lambda I)^-1 A^H``.

    ``solve_normalized`` must solve
    ``(A^H A / max_eigen + lambda_normalized I) x = rhs``.  Scaling each
    right-hand side by ``1/max_eigen`` therefore returns physical image units.
    A factor sample uses one solve.  The explicit ``RR^H`` path uses one
    unweighted ``A^H A`` application and two solves, evaluated as
    ``M^-2 (A^H A v)`` (see ``mr_recon.covariance_apply``); when the solver
    accepts ``x0`` and ``store_directions`` the second solve is warm-started
    from the first solve's Krylov space and certified as usual.  ``normal``
    optionally supplies ``A^H A`` directly (for example a Toeplitz normal
    operator); otherwise ``adjoint(forward(.))`` is used.
    """

    forward: Callable[[torch.Tensor], torch.Tensor]
    adjoint: Callable[[torch.Tensor], torch.Tensor]
    solve_normalized: Callable[..., tuple[torch.Tensor, dict]]
    max_eigen: float
    normal: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    direct_order: str = "reordered"

    def _solver_supports_recycling(self) -> bool:
        try:
            parameters = inspect.signature(self.solve_normalized).parameters
        except (TypeError, ValueError):
            return False
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
            return True
        return {"x0", "store_directions"} <= set(parameters)

    def apply_factor(
        self, input_probe: torch.Tensor
    ) -> tuple[torch.Tensor, list[dict]]:
        rhs = self.adjoint(input_probe) / float(self.max_eigen)
        image, info = self.solve_normalized(rhs)
        return image, [info]

    def apply_direct(
        self, output_probe: torch.Tensor
    ) -> tuple[torch.Tensor, list[dict]]:
        normal_apply = self.normal or (lambda x: self.adjoint(self.forward(x)))
        recycle = self._solver_supports_recycling()
        if recycle:
            solve = self.solve_normalized
        else:
            def solve(rhs, x0=None, store_directions=False):
                return self.solve_normalized(rhs)
        return apply_regularized_covariance(
            solve,
            normal_apply,
            output_probe,
            1.0 / float(self.max_eigen) ** 2,
            recycle=recycle,
            order=self.direct_order,
        )


def cartesian_sense_covariance_diagonal(
    csm: torch.Tensor,
    *,
    acceleration: int,
    lambda_physical: float = 0.0,
) -> torch.Tensor:
    """Closed-form diagonal of ``RR^H`` for regular Cartesian SENSE.

    With an orthonormal FFT and regular one-dimensional undersampling,
    each alias block of ``A^H A`` is ``C^H C / R``. This routine forms those
    blocks and evaluates ``H^-1 (A^H A) H^-1`` exactly.
    """
    if csm.ndim != 3:
        raise ValueError(f"Expected CxHxW sensitivity maps, got {tuple(csm.shape)}")
    width = int(csm.shape[-1])
    if acceleration < 1 or width % acceleration:
        raise ValueError(
            f"Acceleration {acceleration} must divide Cartesian width {width}"
        )
    sensitivities = csm.to(torch.complex128)
    aliases = torch.stack(
        [
            torch.roll(
                sensitivities,
                shifts=index * width // acceleration,
                dims=-1,
            )
            for index in range(acceleration)
        ],
        dim=1,
    )
    gram = torch.einsum(
        "crhw,cshw->hwrs", aliases.conj(), aliases
    )
    normal = gram / float(acceleration)
    eye = torch.eye(
        acceleration, dtype=normal.dtype, device=normal.device
    ).expand(*normal.shape[:-2], acceleration, acceleration)
    system = normal + float(lambda_physical) * eye
    inverse = torch.linalg.inv(system)
    covariance = inverse @ normal @ inverse
    return covariance[..., 0, 0].real.to(torch.float64)
