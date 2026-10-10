"""
Early-stopping / truncated-CG noise propagation for PICO.

IMPORTANT — adaptive vs frozen CG (see revision point 7). Standard conjugate
gradient computes its step sizes ``alpha_k`` and ``beta_k`` from the current
residuals (``alpha_k = <r,r>/<p,Ap>``, ``beta_k = <r_new,r_new>/<r,r>``), which
depend on the right-hand side ``b``. A ``K``-step CG map ``R_K: b -> image`` is
therefore **homogeneous of degree 1** (scale-invariant: ``R_K(s b) = s R_K(b)``)
but **NOT additive** — i.e. it is a *nonlinear* map, not a fixed linear operator.

Consequences for the covariance estimate:

* ``jvp_covariance_diag`` with the default **adaptive** solver computes the
  **local Jacobian** ``J = dR_K/db`` evaluated at the operating point ``ksp0``.
  ``diag(J J^H)`` is then a *first-order / local* noise covariance about ``ksp0``
  — the correct linearization of the actual early-stopped recon that is used, and
  what PMR (which reconstructs ``ksp0 + noise``) matches to first order in the
  noise. It is **NOT** the covariance of a fixed linear operator applied to
  finite noise, so we do not claim exactness for adaptive finite-``K`` CG.
* Pass ``freeze_coeffs=True`` (or a precomputed ``coeffs`` list, e.g. via
  ``freeze_truncated_cg_recon``) to **freeze** ``alpha_k, beta_k`` at their
  values at ``ksp0``. The recursion then has constant coefficients and is an
  **exact fixed linear operator** ``R_K^{frozen}`` (additive: verified by the
  ``__main__`` self-test), whose Jacobian equals the operator itself and whose
  ``J J^H`` is an exact covariance. Use this if you want a genuinely linear
  truncated operator; note it differs from the adaptive recon at second order.

Hutchinson sketch (both modes): ``diag(J J^H)_i = E_v[ |(J v)_i|^2 ]`` with
k-space probes ``v``, ``E[v v^H] = I``.

The converged (``K -> inf``) operator ``R = (A^H A + lambda I)^{-1} A^H`` IS a
genuine fixed linear operator; its covariance
``(A^H A + lambda I)^{-1} A^H A (A^H A + lambda I)^{-1}`` is what
``mr_recon.gfactor.gfactor_sense`` / ``doubleCG_inv_op_builder`` provide as the
converged anchor (and is the regime used for the PRIMARY non-Cartesian result,
where the frozen-vs-adaptive distinction vanishes at convergence).
"""

import torch
from typing import Optional, Callable, Tuple
from tqdm import tqdm
from torch.func import jvp, vmap

from mr_recon.dtypes import complex_dtype
from mr_recon.algs import power_method_operator


def cg_solve_differentiable(AHA: Callable,
                            AHb: torch.Tensor,
                            num_iters: int,
                            lamda_l2: float = 0.0,
                            eps: float = 1e-20,
                            coeffs: Optional[list] = None) -> torch.Tensor:
    """A ``torch.func``-safe, fixed-iteration conjugate gradient solve.

    Mirrors ``mr_recon.algs.conjugate_gradient`` with an identity preconditioner
    and warm start ``x0 = AHb``, but contains no ``.item()`` calls, no
    data-dependent ``break``, no progress bar and no device moves, so it is
    differentiable under ``torch.func.jvp`` / ``vjp`` / ``vmap``.

    A fixed number of iterations is always run (no early exit).

    Two coefficient modes (see module docstring / revision point 7):

    * ``coeffs is None`` (default, **adaptive**): the step sizes ``alpha_k`` and
      ``beta_k`` are recomputed from the residuals every iteration. The resulting
      ``b -> x`` map is homogeneous but **not additive** (a nonlinear map); its
      ``jvp`` is the *local* Jacobian at the operating point.
    * ``coeffs`` provided (**frozen**): a list of ``(alpha_k, beta_k)`` scalar
      tensors (e.g. from :func:`freeze_cg_coeffs`) is used as constants. The
      recursion is then an **exact fixed linear operator** in ``b`` (additive),
      so its ``jvp`` is that operator and ``J J^H`` is an exact covariance.

    The ``eps`` guards protect against the 0/0 breakdown that would otherwise
    occur if the residual reaches machine zero before ``num_iters`` (adaptive
    mode only; large MR problems do not converge that fast).

    Parameters
    ----------
    AHA : callable
        Gram/normal operator ``x -> A^H A x`` (already eigen-normalized if the
        caller normalizes ``AHb`` consistently).
    AHb : torch.Tensor
        Right-hand side ``A^H b``.
    num_iters : int
        Number of CG iterations (the truncation level ``N_cg``).
    lamda_l2 : float
        Tikhonov weight; replaces ``AHA`` with ``AHA + lamda_l2 I``.
    eps : float
        Denominator guard for numerical breakdown (adaptive mode).
    coeffs : list of (alpha, beta), optional
        Frozen CG step coefficients. When supplied the solve becomes an exact
        fixed linear operator (see above). Must have length ``>= num_iters``.
    """
    AHA_w = lambda x: AHA(x) + lamda_l2 * x
    x = AHb
    if num_iters == 0:
        return x
    r = AHb - AHA_w(x)
    p = r.clone()
    frozen = coeffs is not None
    rs_old = None if frozen else torch.real(torch.sum(r.conj() * r))
    for k in range(num_iters):
        Ap = AHA_w(p)
        if frozen:
            alpha, beta = coeffs[k]
            x = x + alpha * p
            r = r - alpha * Ap
            p = r + beta * p
        else:
            pAp = torch.real(torch.sum(p.conj() * Ap))
            alpha = rs_old / (pAp + eps)
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = torch.real(torch.sum(r.conj() * r))
            beta = rs_new / (rs_old + eps)
            p = r + beta * p
            rs_old = rs_new
    return x


def freeze_cg_coeffs(AHA: Callable,
                     AHb0: torch.Tensor,
                     num_iters: int,
                     lamda_l2: float = 0.0,
                     eps: float = 1e-20) -> list:
    """Precompute (and detach) the adaptive CG step coefficients at ``AHb0``.

    Runs the standard adaptive CG recursion on the operating-point right-hand
    side ``AHb0`` in plain eager mode and returns ``[(alpha_0, beta_0), ...]`` as
    detached scalar tensors. Feeding these back into
    :func:`cg_solve_differentiable` (``coeffs=...``) turns the truncated solve
    into an exact fixed *linear* operator whose fixed coefficients are those the
    adaptive solver would have used at ``AHb0``.
    """
    AHA_w = lambda x: AHA(x) + lamda_l2 * x
    with torch.no_grad():
        x = AHb0
        r = AHb0 - AHA_w(x)
        p = r.clone()
        rs_old = torch.real(torch.sum(r.conj() * r))
        out = []
        for _ in range(num_iters):
            Ap = AHA_w(p)
            pAp = torch.real(torch.sum(p.conj() * Ap))
            alpha = rs_old / (pAp + eps)
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = torch.real(torch.sum(r.conj() * r))
            beta = rs_new / (rs_old + eps)
            p = r + beta * p
            rs_old = rs_new
            out.append((alpha.detach(), beta.detach()))
    return out


def build_truncated_cg_recon(normal_fn: Callable,
                             adjoint_fn: Callable,
                             max_iter: int,
                             lamda_l2: float = 0.0,
                             max_eigen: float = 1.0,
                             coeffs: Optional[list] = None) -> Callable:
    """Build a truncated-CG SENSE reconstruction ``R: ksp -> image``.

    Uses the exact same normalization convention as
    ``mr_recon.recons.CG_SENSE_recon`` (scale ``A^H b`` by ``max_eigen**-0.5``
    and the normal operator by ``max_eigen**-1``), so that a PMR reconstruction
    using ``CG_SENSE_recon(..., max_iter=N_cg, ...)`` and this operator describe
    the *same* map.

    ``coeffs`` (optional) freezes the CG step sizes → an exact fixed linear
    operator (see module docstring). Prefer :func:`freeze_truncated_cg_recon`,
    which precomputes them at a given operating point.

    Parameters
    ----------
    normal_fn : callable
        ``x -> A^H A x``.
    adjoint_fn : callable
        ``b -> A^H b``.
    max_iter : int
        CG truncation level ``N_cg``.
    lamda_l2 : float
        Tikhonov weight.
    max_eigen : float
        Eigenvalue normalization (as in ``CG_SENSE_recon``).
    coeffs : list, optional
        Frozen ``(alpha_k, beta_k)`` coefficients.
    """
    def R(ksp: torch.Tensor) -> torch.Tensor:
        AHb = adjoint_fn(ksp.type(complex_dtype)) / (max_eigen ** 0.5)
        AHA = lambda x: normal_fn(x) / max_eigen
        return cg_solve_differentiable(AHA, AHb, max_iter, lamda_l2=lamda_l2, coeffs=coeffs)
    return R


def freeze_truncated_cg_recon(normal_fn: Callable,
                              adjoint_fn: Callable,
                              ksp0: torch.Tensor,
                              max_iter: int,
                              lamda_l2: float = 0.0,
                              max_eigen: float = 1.0) -> Callable:
    """Build the **frozen** (exact fixed-linear) truncated-CG recon at ``ksp0``.

    Precomputes the adaptive CG coefficients at the operating point ``ksp0`` and
    returns a recon whose ``b -> image`` map is exactly linear. Its ``jvp`` is
    then the operator itself, so ``jvp_covariance_diag`` yields the exact
    covariance of that fixed linear operator (rather than a local Jacobian).
    """
    AHb0 = adjoint_fn(ksp0.type(complex_dtype)) / (max_eigen ** 0.5)
    AHA = lambda x: normal_fn(x) / max_eigen
    coeffs = freeze_cg_coeffs(AHA, AHb0, max_iter, lamda_l2=lamda_l2)
    return build_truncated_cg_recon(normal_fn, adjoint_fn, max_iter,
                                    lamda_l2=lamda_l2, max_eigen=max_eigen, coeffs=coeffs)


def _draw_probes(shape: Tuple[int, ...], probe: str, device, dtype) -> torch.Tensor:
    """Draw k-space probes ``v`` with E[v v^H] = I (unit total power per entry)."""
    if probe == 'phase':
        return torch.exp(1j * 2 * torch.pi * torch.rand(shape, device=device)).to(dtype)
    elif probe == 'rademacher':
        re = torch.randint(0, 2, shape, device=device).float() * 2 - 1
        im = torch.randint(0, 2, shape, device=device).float() * 2 - 1
        return ((re + 1j * im) / (2 ** 0.5)).to(dtype)
    elif probe == 'gauss':
        return torch.randn(shape, device=device, dtype=dtype)
    else:
        raise ValueError(f"Unknown probe type: {probe}")


def jvp_covariance_diag(recon_fn: Callable,
                        ksp0: torch.Tensor,
                        n_probes: int,
                        probe: str = 'phase',
                        batch_size: int = 50,
                        verbose: bool = True) -> torch.Tensor:
    """PICO variance map from the Jacobian of ``recon_fn`` at ``ksp0``.

    Estimates ``diag(J J^H)`` where ``J = d recon_fn / d ksp`` evaluated at the
    operating point ``ksp0``, via a Hutchinson sketch with forward-mode AD:
    ``diag(J J^H)_i = E_v[ |(J v)_i|^2 ]`` with k-space probes ``v``,
    ``E[v v^H] = I``.

    Interpretation depends on ``recon_fn`` (see module docstring):
    * adaptive truncated CG → ``J`` is the **local** Jacobian at ``ksp0`` and the
      result is a first-order / local covariance (matches PMR to first order).
    * frozen truncated CG (``freeze_truncated_cg_recon``) or the converged inverse
      → ``recon_fn`` is exactly linear and ``J J^H`` is the exact covariance.

    Returns a real image-space variance map (per unit k-space noise variance).
    Multiply by the k-space noise variance to obtain the reconstruction noise
    variance; the g-factor is a *ratio* of two such maps so the scaling cancels.
    """
    device = ksp0.device
    dtype = ksp0.dtype
    ksp_shape = tuple(ksp0.shape)

    def compute_sample(v):
        _, u = jvp(recon_fn, (ksp0,), (v,))
        return u.real.pow(2) + u.imag.pow(2)

    total = None
    done = 0
    pbar = tqdm(total=n_probes, disable=not verbose, desc=f'PICO jvp ({probe})')
    while done < n_probes:
        B = min(batch_size, n_probes - done)
        v_batch = _draw_probes((B,) + ksp_shape, probe, device, dtype)
        samples = vmap(compute_sample)(v_batch)
        total = samples.sum(0) if total is None else total + samples.sum(0)
        done += B
        pbar.update(B)
    pbar.close()
    return total / n_probes


if __name__ == '__main__':
    # Self-test (revision point 7): demonstrate that ADAPTIVE truncated CG is
    # NON-additive (a nonlinear map) while FROZEN truncated CG is an EXACT linear
    # operator. Prints the additivity residual ||R(b1+b2) - R(b1) - R(b2)|| / ||R(b1+b2)||.
    torch.manual_seed(0)
    n = 64
    M = torch.randn(n, n, dtype=torch.complex64)
    G = M.conj().T @ M + 0.05 * torch.eye(n, dtype=torch.complex64)  # SPD Gram
    AHA = lambda x: G @ x
    K = 8
    b1 = torch.randn(n, dtype=torch.complex64)
    b2 = torch.randn(n, dtype=torch.complex64)

    def relerr(a, ref):
        return (torch.linalg.norm(a - ref) / (torch.linalg.norm(ref) + 1e-20)).item()

    # Adaptive
    Ra = lambda b: cg_solve_differentiable(AHA, b, K, lamda_l2=0.0)
    add_res_adaptive = relerr(Ra(b1 + b2), Ra(b1) + Ra(b2))
    scale_res_adaptive = relerr(Ra(2.5 * b1), 2.5 * Ra(b1))  # homogeneity holds

    # Frozen at operating point b1+b2
    coeffs = freeze_cg_coeffs(AHA, b1 + b2, K, lamda_l2=0.0)
    Rf = lambda b: cg_solve_differentiable(AHA, b, K, lamda_l2=0.0, coeffs=coeffs)
    add_res_frozen = relerr(Rf(b1 + b2), Rf(b1) + Rf(b2))

    print(f"[frozen-CG self-test] K={K} SPD {n}x{n}")
    print(f"  adaptive additivity residual = {add_res_adaptive:.3e}  (nonzero => NONLINEAR)")
    print(f"  adaptive homogeneity residual= {scale_res_adaptive:.3e}  (~0 => scale-invariant)")
    print(f"  frozen   additivity residual = {add_res_frozen:.3e}  (~0 => EXACT LINEAR)")
    ok = (add_res_adaptive > 1e-3) and (add_res_frozen < 1e-5)
    print(f"  VERDICT: {'as expected (adaptive nonlinear, frozen linear)' if ok else 'UNEXPECTED'}")
