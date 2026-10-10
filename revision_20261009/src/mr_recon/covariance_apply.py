"""Applying the image-space noise covariance of a Tikhonov-regularized
linear reconstruction to a probe with two certified solves and one Krylov
recycle.

For ``R = (A^H A + lambda I)^-1 A^H`` the covariance operator is
``Sigma = R R^H = M^-1 (A^H A) M^-1`` with ``M = A^H A + lambda I``.  Because
``M`` is a function of ``A^H A`` the two commute, so

    Sigma v = M^-2 (A^H A v).

Evaluating the product in this order has two consequences that matter in
practice:

* the first right-hand side ``A^H A v`` lies in the range of ``A^H A`` and no
  longer carries the near-null content of a broadband probe, which removes the
  restart storms and the float32 cancellation error of the literal order
  ``M^-1 (A^H A (M^-1 v))``;
* the second solve ``M^-1 w1`` has the same operator as the first, so the
  Galerkin projection of ``M^-1 w1`` onto the first solve's Krylov space,
  formed from the stored CG search directions, is a warm start that typically
  halves the second solve.  The second solve is still run to the same
  true-residual certificate as before; only its initial guess changes.

The estimator itself is unchanged: ``diag(Sigma)_k ~ mean_v Re(conj(v_k)
(Sigma v)_k)`` over unit-magnitude random-phase probes.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import torch

SolveFn = Callable[..., Tuple[torch.Tensor, dict]]

COVARIANCE_APPLY_DESCRIPTION = (
    "Sigma v = M^-2 (A^H A v): first certified solve on the range(A^H A) "
    "right-hand side; second certified solve warm-started from the Galerkin "
    "projection onto the first solve's Krylov space (stored CG directions); "
    "both solves stop on the recomputed true residual"
)


def galerkin_recycled_guess(
    directions: Sequence[torch.Tensor],
    curvatures: Sequence[float],
    target: torch.Tensor,
) -> torch.Tensor:
    """M-orthogonal projection of ``M^-1 target`` onto ``span(directions)``.

    CG search directions are M-conjugate, so with ``c_j = p_j^H t / (p_j^H M p_j)``
    the vector ``sum_j c_j p_j`` is the Galerkin approximation of ``M^-1 t`` on
    the Krylov space spanned by the directions.  In floating point conjugacy
    decays with the iteration count, which only degrades the guess; correctness
    is restored by the certified solve that follows.
    """
    guess = torch.zeros_like(target)
    for p, curvature in zip(directions, curvatures):
        if curvature <= 0.0:
            continue
        coefficient = torch.sum(p.conj() * target) / curvature
        guess = guess + coefficient * p
    return guess


def apply_regularized_covariance(
    solve: SolveFn,
    normal_apply: Callable[[torch.Tensor], torch.Tensor],
    probe: torch.Tensor,
    scale: float,
    *,
    recycle: bool = True,
    order: str = "reordered",
) -> Tuple[torch.Tensor, list]:
    """Return ``scale * Sigma probe`` and the two solver info dicts.

    ``solve(rhs, x0=None, store_directions=False)`` must solve ``M x = rhs``
    for the normalized operator the caller uses inside CG and return
    ``(x, info)`` with ``info["directions"]`` when asked.  ``normal_apply``
    applies the unscaled ``A^H A``.  ``scale`` carries the operator
    normalization (for example ``1/max_eigen`` when ``M`` is
    ``A^H A / max_eigen + lambda_normalized I`` and the caller works in
    normalized units, or ``1/max_eigen**2`` for physical units).

    ``order="legacy"`` reproduces the literal product ``M^-1 A^H A M^-1 v``
    with two zero-initialized solves and is kept only for comparison.
    """
    if order == "legacy":
        half = float(scale) ** 0.5
        first, first_info = solve(probe * half)
        middle = normal_apply(first)
        second, second_info = solve(middle * half)
        second_info = dict(second_info)
        second_info["initialization"] = "zero"
        return second, [first_info, second_info]
    if order != "reordered":
        raise ValueError(f"unknown covariance evaluation order: {order!r}")

    rhs = normal_apply(probe) * float(scale)
    first, first_info = solve(rhs, store_directions=recycle)
    first_info = dict(first_info)
    guess: Optional[torch.Tensor] = None
    if recycle:
        directions, curvatures = first_info.pop("directions")
        guess = galerkin_recycled_guess(directions, curvatures, first)
        del directions, curvatures
    second, second_info = solve(first, x0=guess)
    second_info = dict(second_info)
    second_info["initialization"] = "krylov_recycled" if recycle else "zero"
    return second, [first_info, second_info]


def _squared_norm(x: torch.Tensor) -> torch.Tensor:
    return torch.real(torch.sum(x.conj() * x))


def cg_true_residual(
    operator: Callable[[torch.Tensor], torch.Tensor],
    rhs: torch.Tensor,
    *,
    max_iter: int,
    rtol: float,
    true_residual_every: int = 10,
    x0: Optional[torch.Tensor] = None,
    store_directions: bool = False,
    reliable_update_threshold: float = 0.1,
) -> Tuple[torch.Tensor, dict]:
    """Conjugate gradients on a Hermitian positive definite ``operator`` with
    the stopping certificate evaluated on the recomputed true residual
    ``||rhs - operator(x)|| / ||rhs||`` (never on the CG recurrence), an
    optional warm start ``x0``, and optional storage of the search directions
    and curvatures for Krylov recycling (see ``galerkin_recycled_guess``).

    This is the toolbox counterpart of the audit solver used for the
    manuscript's non-Cartesian experiments; it exists so that the covariance
    evaluation of this module can be used without the audit scripts.
    """
    rhs_norm = float(torch.linalg.norm(rhs))
    x = torch.zeros_like(rhs) if x0 is None else x0.clone()
    if rhs_norm == 0.0:
        return x, {"iterations": 0, "converged": True, "restarts": 0,
                   "initialization": "zero" if x0 is None else "warm_start",
                   "initial_relative_residual": 0.0, "true_relative_residual": 0.0}
    r = rhs.clone() if x0 is None else rhs - operator(x)
    initial_rel = float(torch.linalg.norm(r)) / rhs_norm
    converged = initial_rel <= rtol
    p = r.clone()
    rho = _squared_norm(r)
    directions: list = []
    curvatures: list = []
    restarts = 0
    true_rel = initial_rel
    used = 0
    for it in range(1, (0 if converged else max_iter) + 1):
        Ap = operator(p)
        pAp = torch.real(torch.sum(p.conj() * Ap))
        pAp_value = float(pAp)
        if not (pAp_value > 0.0) or pAp_value != pAp_value:
            break
        alpha = rho / pAp
        x = x + alpha * p
        r_rec = r - alpha * Ap
        used = it
        if store_directions:
            directions.append(p.detach().clone())
            curvatures.append(pAp_value)
        restarted = False
        if it % true_residual_every == 0 or it == max_iter:
            r_true = rhs - operator(x)
            tn = float(torch.linalg.norm(r_true))
            true_rel = tn / rhs_norm
            if true_rel <= rtol:
                converged = True
                break
            drift = float(torch.linalg.norm(r_true - r_rec)) / max(tn, 1e-30)
            if drift > reliable_update_threshold:
                r = r_true
                p = r_true.clone()
                rho = _squared_norm(r)
                restarts += 1
                restarted = True
        if restarted:
            continue
        rho_new = _squared_norm(r_rec)
        beta = rho_new / rho
        p = r_rec + beta * p
        r = r_rec
        rho = rho_new
    if not converged:
        true_rel = float(torch.linalg.norm(rhs - operator(x))) / rhs_norm
    info = {"iterations": used, "converged": bool(converged), "restarts": restarts,
            "initialization": "zero" if x0 is None else "warm_start",
            "initial_relative_residual": initial_rel, "true_relative_residual": true_rel}
    if store_directions:
        info["directions"] = (directions, curvatures)
    return x, info


def recycled_covariance_operator(
    A,
    lamda_l2: float,
    max_eigen: float,
    *,
    max_iter: int,
    rtol: float,
    true_residual_every: int = 10,
    recycle: bool = True,
):
    """Return ``apply(v) -> (Sigma v, infos)`` for the Tikhonov CG-SENSE
    reconstruction ``R = (A^H A + lamda I)^-1 A^H`` in the same normalized
    units as ``mr_recon.recons.doubleCG_inv_op_builder`` (``lamda_l2`` is the
    regularization of the eigenvalue-normalized operator ``A^H A / max_eigen``).

    The product is evaluated as ``M^-2 (A^H A v) / max_eigen`` with two
    true-residual-certified CG solves, the second warm-started from the first
    solve's Krylov space.  ``A`` must expose ``normal`` (the unweighted
    ``A^H A``).
    """
    mu = float(max_eigen)

    def normalized(x):
        return A.normal(x) / mu + float(lamda_l2) * x

    def solve(rhs, x0=None, store_directions=False):
        return cg_true_residual(normalized, rhs, max_iter=max_iter, rtol=rtol,
                                true_residual_every=true_residual_every, x0=x0,
                                store_directions=store_directions)

    def apply(v: torch.Tensor):
        return apply_regularized_covariance(solve, A.normal, v, 1.0 / mu, recycle=recycle)

    return apply
