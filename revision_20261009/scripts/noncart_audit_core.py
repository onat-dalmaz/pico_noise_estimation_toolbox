"""Shared numerical primitives for the non-Cartesian validity audit."""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional, Tuple

import numpy as np
import torch


def normalized_to_physical_variance(variance, max_eigen: float):
    """Undo CG-SENSE's sqrt(max_eigen) output scaling for a variance map."""
    return variance / float(max_eigen)


def _squared_norm(x: torch.Tensor) -> torch.Tensor:
    return torch.real(torch.sum(x.conj() * x))


def solve_cg_true_residual(
    operator: Callable[[torch.Tensor], torch.Tensor],
    rhs: torch.Tensor,
    *,
    max_iter: int,
    rtol: float,
    true_residual_every: int = 10,
    snapshot_iters: Optional[Iterable[int]] = None,
    reliable_update: bool = True,
    reliable_update_threshold: float = 0.1,
    x0: Optional[torch.Tensor] = None,
    store_directions: bool = False,
) -> Tuple[torch.Tensor, Dict]:
    """Solve ``operator(x)=rhs`` with true-residual stopping.

    The recursive CG residual is never used as the stopping certificate.  The
    true residual ``rhs - operator(x)`` is recomputed periodically and at every
    requested snapshot.  If finite-precision drift becomes material, the
    recurrence is restarted from the true residual.

    ``x0`` is an optional warm start; the certificate is unchanged, because it
    is always evaluated on the actual iterate against ``rhs``.  With
    ``store_directions`` the CG search directions ``p_j`` and curvatures
    ``p_j^H H p_j`` are returned under ``info["directions"]`` so a later solve
    with the same operator can be warm-started from the Krylov space (see
    ``mr_recon.covariance_apply``).  The caller must pop that entry before
    serializing ``info``.
    """
    if max_iter < 0:
        raise ValueError("max_iter must be non-negative")
    if true_residual_every < 1:
        raise ValueError("true_residual_every must be at least one")

    snapshots_requested = set(int(i) for i in (snapshot_iters or []))
    snapshots = {}
    initialization = "zero" if x0 is None else "warm_start"
    x = torch.zeros_like(rhs) if x0 is None else x0.clone()
    rhs_norm = float(torch.linalg.norm(rhs).item())
    if rhs_norm == 0.0:
        return x, {
            "iterations": 0,
            "converged": True,
            "breakdown": None,
            "initialization": initialization,
            "initial_relative_residual": 0.0,
            "stopping_definition": "||rhs - Hx||_2 / ||rhs||_2",
            "true_relative_residual": 0.0,
            "history": [{"iteration": 0, "true_relative_residual": 0.0}],
            "snapshots": snapshots,
            "reliable_updates": 0,
        }

    if x0 is None:
        r = rhs.clone()
        initial_rel = 1.0
    else:
        r = rhs - operator(x)
        initial_rel = float(torch.linalg.norm(r).item()) / rhs_norm
    p = r.clone()
    rho = _squared_norm(r)
    history = [{"iteration": 0, "true_relative_residual": initial_rel}]
    true_rel = initial_rel
    converged = initial_rel <= rtol
    breakdown = None
    reliable_updates = 0
    used = 0
    directions: list = []
    curvatures: list = []

    for iteration in range(1, (0 if converged else max_iter) + 1):
        Ap = operator(p)
        pAp = torch.real(torch.sum(p.conj() * Ap))
        pAp_value = float(pAp.item())
        if not np.isfinite(pAp_value) or pAp_value <= 0.0:
            breakdown = f"nonpositive_or_nonfinite_pAp:{pAp_value}"
            break

        alpha = rho / pAp
        x = x + alpha * p
        r_recursive = r - alpha * Ap
        used = iteration
        if store_directions:
            directions.append(p.detach().clone())
            curvatures.append(pAp_value)

        needs_true = (
            iteration % true_residual_every == 0
            or iteration in snapshots_requested
            or iteration == max_iter
        )
        restarted = False
        if needs_true:
            r_true = rhs - operator(x)
            true_norm = float(torch.linalg.norm(r_true).item())
            true_rel = true_norm / rhs_norm
            recursive_rel = float(torch.linalg.norm(r_recursive).item()) / rhs_norm
            drift = float(torch.linalg.norm(r_true - r_recursive).item()) / max(true_norm, 1e-30)
            history.append(
                {
                    "iteration": iteration,
                    "true_relative_residual": true_rel,
                    "recursive_relative_residual": recursive_rel,
                    "residual_drift_relative": drift,
                }
            )
            if iteration in snapshots_requested:
                snapshots[iteration] = x.detach().clone()
            if true_rel <= rtol:
                r = r_true
                converged = True
                break
            if reliable_update and drift > reliable_update_threshold:
                r = r_true
                p = r_true.clone()
                rho = _squared_norm(r)
                reliable_updates += 1
                restarted = True

        if restarted:
            continue

        rho_new = _squared_norm(r_recursive)
        rho_value = float(rho.item())
        rho_new_value = float(rho_new.item())
        if (
            not np.isfinite(rho_value)
            or not np.isfinite(rho_new_value)
            or rho_value <= 0.0
        ):
            breakdown = f"nonpositive_or_nonfinite_rho:{rho_value}->{rho_new_value}"
            r = r_recursive
            break
        beta = rho_new / rho
        p = r_recursive + beta * p
        r = r_recursive
        rho = rho_new

    if used not in snapshots_requested and used == max_iter:
        snapshots[used] = x.detach().clone()
    if not converged:
        r_true = rhs - operator(x)
        true_rel = float(torch.linalg.norm(r_true).item()) / rhs_norm
        if not history or history[-1]["iteration"] != used:
            history.append(
                {
                    "iteration": used,
                    "true_relative_residual": true_rel,
                    "recursive_relative_residual": float(torch.linalg.norm(r).item()) / rhs_norm,
                }
            )

    info = {
        "iterations": used,
        "converged": bool(converged),
        "breakdown": breakdown,
        "initialization": initialization,
        "initial_relative_residual": initial_rel,
        "stopping_definition": "||rhs - Hx||_2 / ||rhs||_2",
        "true_relative_residual": true_rel,
        "history": history,
        "snapshots": snapshots,
        "reliable_updates": reliable_updates,
    }
    if store_directions:
        info["directions"] = (directions, curvatures)
    return x, info
