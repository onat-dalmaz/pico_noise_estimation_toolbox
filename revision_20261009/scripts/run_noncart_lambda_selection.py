"""Regularization selection for the non-Cartesian spiral operator.

Produces the evidence for Supporting Information S7:

* S7.1  spectral density of ``A^H A`` by stochastic Lanczos quadrature, and the
        fraction of a unit-magnitude random-phase probe's energy that lands in
        the near-null subspace compared with the fraction for ``A^H b``.
* S7.2  semiconvergence of the unregularized reconstruction: in-object signal
        against out-of-object streak-artifact energy versus CG iteration.
* S7.3  covariance-probe behaviour versus lambda, including the divergence of
        the probe solves at ``lambda = 0``.
* S7.4  L-curve and discrepancy-principle selection, computed from the signal
        reconstruction alone and independent of any PICO/PMR comparison.

Every solve uses the clean unweighted ``A^H A`` objective, deterministic
operator normalization, zero initialization, and stopping on a recomputed true
normal-equation residual, matching ``run_noncart_weekend_audit.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_noncart_weekend_audit import (  # noqa: E402
    SIGMA,
    build_problem,
    make_systems,
)
from noncart_audit_core import solve_cg_true_residual  # noqa: E402

from mr_recon.dtypes import complex_dtype  # noqa: E402


DEFAULT_OUT = ROOT / "experiments/noncartesian_phantom/results/lambda_selection"

# Log-spaced normalized regularization grid spanning the ill-posed regime
# through over-smoothing, plus the unregularized control.
DEFAULT_LAMBDAS = [
    0.0,
    1e-5,
    3e-5,
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
    3e-2,
    1e-1,
]

SEMICONVERGENCE_ITERS = [10, 25, 50, 100, 200, 400, 800, 1200]


# ---------------------------------------------------------------- utilities


def _erode(mask: np.ndarray, iterations: int) -> np.ndarray:
    """Binary erosion with a 4-connected structuring element."""
    out = mask.copy()
    for _ in range(iterations):
        shifted = np.ones_like(out)
        shifted[1:, :] &= out[:-1, :]
        shifted[:-1, :] &= out[1:, :]
        shifted[:, 1:] &= out[:, :-1]
        shifted[:, :-1] &= out[:, 1:]
        out = out & shifted
    return out


def _aligned_nrmse(x: torch.Tensor, ref: torch.Tensor, support: torch.Tensor) -> float:
    """Magnitude NRMSE on the support after optimal complex scale alignment."""
    a = x[support]
    b = ref[support]
    scale = torch.vdot(a.flatten(), b.flatten()) / torch.vdot(a.flatten(), a.flatten())
    return float(torch.linalg.norm(scale * a - b) / torch.linalg.norm(b))


def artifact_metrics(x_phys: torch.Tensor, background: np.ndarray, support: np.ndarray):
    """Out-of-object streak-artifact energy relative to in-object signal."""
    m = x_phys.abs().detach().cpu().numpy()
    sig = float(m[support].mean())
    bg = m[background]
    return {
        "in_object_mean": sig,
        "background_mean": float(bg.mean()),
        "background_std": float(bg.std()),
        "background_p99_9": float(np.percentile(bg, 99.9)),
        "artifact_to_signal_pct": float(100.0 * bg.mean() / max(sig, 1e-30)),
        "peak_background_over_signal": float(np.percentile(bg, 99.9) / max(sig, 1e-30)),
    }


# ------------------------------------------------- S7.1 spectrum (SLQ + probe)


def lanczos_weights(normal_op, start, *, steps):
    """Lanczos quadrature nodes and weights for one starting vector.

    For a Hermitian ``M`` and unit ``start``, the Ritz values ``theta_i`` and
    weights ``w_i = |e_1^T y_i|^2`` are a discrete approximation of the spectral
    measure of ``start`` in the eigenbasis of ``M``.  With a random start this
    estimates the spectral *density*; with a specific start it gives how that
    vector's energy is distributed over the spectrum.  Full reorthogonalization
    is used so the small Ritz values stay trustworthy.
    """
    v = start / torch.linalg.norm(start)
    alphas, betas, basis = [], [], [v]
    beta_prev, v_prev = 0.0, torch.zeros_like(v)
    for j in range(steps):
        w = normal_op(basis[-1])
        alpha = torch.real(torch.vdot(basis[-1].flatten(), w.flatten())).item()
        w = w - alpha * basis[-1] - beta_prev * v_prev
        for u in basis:
            w = w - torch.vdot(u.flatten(), w.flatten()) * u
        beta = float(torch.linalg.norm(w))
        alphas.append(alpha)
        if beta < 1e-12 or j == steps - 1:
            break
        betas.append(beta)
        v_prev = basis[-1]
        basis.append(w / beta)
        beta_prev = beta
    T = np.diag(alphas)
    if betas:
        T += np.diag(betas, 1) + np.diag(betas, -1)
    theta, vecs = np.linalg.eigh(T)
    return theta, vecs[0, :] ** 2


def energy_below(theta, weights, cutoff) -> float:
    return float(weights[theta < cutoff].sum() / max(weights.sum(), 1e-30))


def spectral_energy_comparison(problem, normal_op, *, steps, seed, cutoffs):
    """Where a random-phase probe puts its energy, versus where ``A^H b`` does.

    This is the quantitative form of the S7.1 argument: the data right-hand side
    lives in ``range(A^H)`` and carries almost no weight on the near-null
    spectrum, whereas a broadband unit-magnitude probe does.
    """
    g = torch.Generator(device=problem.device).manual_seed(seed)
    theta_p = 2 * math.pi * torch.rand(
        problem.A_acc.ishape, device=problem.device, generator=g
    )
    probe = torch.polar(torch.ones_like(theta_p), theta_p).type(complex_dtype)
    rhs = problem.A_acc.adjoint(problem.ksp_acc.type(complex_dtype))

    tp, wp = lanczos_weights(normal_op, probe, steps=steps)
    tr, wr = lanczos_weights(normal_op, rhs, steps=steps)
    out = {}
    for c in cutoffs:
        out[f"probe_energy_below_{c:g}"] = energy_below(tp, wp, c)
        out[f"data_rhs_energy_below_{c:g}"] = energy_below(tr, wr, c)
    out["probe_min_ritz"] = float(tp.min())
    out["data_rhs_min_ritz"] = float(tr.min())
    return out, (tp, wp), (tr, wr)


# ---------------------------------------------------------------- main sweep


def run(args) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out = Path(args.out_dir) / f"R{args.R}"
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    problem = build_problem(args.R, device)
    support_np = problem.support.detach().cpu().numpy()
    background_np = _erode(~support_np, args.background_erosion)
    ref_image = problem.image

    # ---- S7.1 spectrum of the normalized accelerated normal operator --------
    def normal_norm(x):
        return problem.A_acc.normal(x) / problem.max_eigen_acc

    g_spec = torch.Generator(device=device).manual_seed(args.seed + 11)
    all_t, all_w = [], []
    for _ in range(args.spectrum_probes):
        z = torch.randn(
            problem.A_acc.ishape,
            dtype=complex_dtype,
            device=device,
            generator=g_spec,
        )
        t, w = lanczos_weights(normal_norm, z, steps=args.lanczos_steps)
        all_t.append(t)
        all_w.append(w)
    theta = np.concatenate(all_t)
    weights = np.concatenate(all_w) / args.spectrum_probes
    order = np.argsort(theta)
    cutoffs = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    energy, probe_q, rhs_q = spectral_energy_comparison(
        problem,
        normal_norm,
        steps=args.lanczos_steps,
        seed=args.seed + 12,
        cutoffs=cutoffs,
    )
    np.savez(
        out / "spectrum_slq.npz",
        ritz_values=theta[order],
        weights=weights[order],
        probe_ritz=probe_q[0],
        probe_weights=probe_q[1],
        data_rhs_ritz=rhs_q[0],
        data_rhs_weights=rhs_q[1],
        max_eigen_acc=problem.max_eigen_acc,
        max_eigen_ref=problem.max_eigen_ref,
    )
    with open(out / "spectrum_summary.json", "w") as f:
        json.dump(
            {
                "slq_probes": args.spectrum_probes,
                "lanczos_steps": args.lanczos_steps,
                "ritz_min_normalized": float(theta.min()),
                "ritz_max_normalized": float(theta.max()),
                "effective_condition_number": float(
                    theta.max() / max(theta.min(), 1e-30)
                ),
                **{
                    f"spectral_weight_below_{c:g}": energy_below(theta, weights, c)
                    for c in cutoffs
                },
                **energy,
            },
            f,
            indent=2,
        )
    print(
        f"[spectrum] ritz {theta.min():.3e} .. {theta.max():.3e}  "
        f"probe energy<1e-4 = {energy['probe_energy_below_0.0001']:.4f}  "
        f"A^Hb energy<1e-4 = {energy['data_rhs_energy_below_0.0001']:.4f}",
        flush=True,
    )

    # ---- lambda sweep -------------------------------------------------------
    rows = []
    semiconv_rows = []
    for lam in args.lambdas:
        _, acc = make_systems(
            problem, lam, lam, args.max_iter, args.rtol, args.true_residual_every
        )
        # Fixed-budget pass for the semiconvergence trace (rtol=0, never stops
        # early), then a separate residual-controlled solve so that the reported
        # iteration count is the number actually needed to converge.
        _, trace_info = acc.reconstruct_data(
            problem.ksp_acc, snapshot_iters=SEMICONVERGENCE_ITERS
        )
        snapshots = trace_info["snapshots"]
        rhs_data = problem.A_acc.adjoint(
            problem.ksp_acc.type(complex_dtype)
        ) / math.sqrt(problem.max_eigen_acc)
        x_norm, info = acc.solve_image_rhs(rhs_data)

        # semiconvergence trace (all lambdas, so the contrast is explicit)
        for cap, xs in sorted(snapshots.items()):
            xp = xs / math.sqrt(problem.max_eigen_acc)
            am = artifact_metrics(xp, background_np, support_np)
            semiconv_rows.append(
                {
                    "lambda_normalized": lam,
                    "iteration": cap,
                    "solution_norm_physical": float(torch.linalg.norm(xp)),
                    "image_nrmse_aligned": _aligned_nrmse(
                        xp, ref_image, problem.support
                    ),
                    **am,
                }
            )

        x_phys = x_norm / math.sqrt(problem.max_eigen_acc)
        data_residual = problem.A_acc.forward(x_phys) - problem.ksp_acc
        res_norm = float(torch.linalg.norm(data_residual))
        sol_norm = float(torch.linalg.norm(x_phys))
        np.save(
            out / f"recon_lambda{lam:.0e}_converged_physical.npy",
            x_phys.detach().cpu().numpy(),
        )

        # covariance probes: this is where lambda=0 fails
        g = torch.Generator(device=device).manual_seed(args.seed + 7)
        probe_res, probe_iters = [], []
        diag_acc = torch.zeros(problem.A_acc.ishape, dtype=complex_dtype, device=device)
        for _ in range(args.cov_probes):
            th = 2 * math.pi * torch.rand(
                problem.A_acc.ishape, device=device, generator=g
            )
            v = torch.polar(torch.ones_like(th), th).type(complex_dtype)
            u, i1, i2 = acc.covariance_apply(v)
            diag_acc = diag_acc + v.conj() * u
            probe_res.append(
                max(i1["true_relative_residual"], i2["true_relative_residual"])
            )
            probe_iters.append(i1["iterations"] + i2["iterations"])
        diag_acc = (diag_acc / args.cov_probes).real
        d = diag_acc.detach().cpu().numpy()
        finite = np.isfinite(d)

        am = artifact_metrics(x_phys, background_np, support_np)
        rows.append(
            {
                "lambda_normalized": lam,
                "lambda_physical": acc.lambda_physical,
                "signal_iterations": info["iterations"],
                "signal_true_relative_residual": info["true_relative_residual"],
                "data_residual_norm": res_norm,
                "solution_norm_physical": sol_norm,
                "log_residual": math.log(max(res_norm, 1e-300)),
                "log_solution_norm": math.log(max(sol_norm, 1e-300)),
                "image_nrmse_aligned": _aligned_nrmse(
                    x_phys, ref_image, problem.support
                ),
                **am,
                "cov_probe_max_true_residual": float(max(probe_res)),
                "cov_probe_mean_iterations": float(np.mean(probe_iters)),
                "cov_diag_finite_fraction": float(finite.mean()),
                "cov_diag_median_support": float(
                    np.median(d[finite & support_np]) if (finite & support_np).any()
                    else np.nan
                ),
                "cov_diag_max": float(np.nanmax(np.where(finite, d, np.nan))),
                "cov_diag_negative_fraction": float((d[finite] < 0).mean()),
            }
        )
        print(
            f"[lambda {lam:.1e}] signal it={info['iterations']:4d} "
            f"res={info['true_relative_residual']:.2e} "
            f"probe_res={max(probe_res):.2e} "
            f"probe_it={np.mean(probe_iters):7.1f} "
            f"artifact={am['artifact_to_signal_pct']:6.2f}% "
            f"nrmse={rows[-1]['image_nrmse_aligned']*100:6.2f}%",
            flush=True,
        )

    _write_csv(out / "lambda_selection.csv", rows)
    _write_csv(out / "semiconvergence.csv", semiconv_rows)

    with open(out / "config.json", "w") as f:
        json.dump(
            {
                "R": args.R,
                "operator": "clean unweighted A^H A, no DCF weighting",
                "initialization": "zero",
                "stopping": "recomputed ||rhs-Hx||_2/||rhs||_2",
                "sigma": SIGMA,
                "max_eigen_acc": problem.max_eigen_acc,
                "max_eigen_ref": problem.max_eigen_ref,
                "support_definition": "|x_ref| > 0.05 max|x_ref|",
                "background_erosion_iterations": args.background_erosion,
                "arguments": vars(args),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            },
            f,
            indent=2,
            default=float,
        )
    print(f"[done] wrote {out}", flush=True)


def _write_csv(path: Path, rows) -> None:
    import csv

    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--R", type=int, default=2)
    p.add_argument("--lambdas", type=float, nargs="+", default=DEFAULT_LAMBDAS)
    p.add_argument("--max-iter", type=int, default=1200)
    p.add_argument("--rtol", type=float, default=1e-5)
    p.add_argument("--true-residual-every", type=int, default=10)
    p.add_argument("--cov-probes", type=int, default=8)
    p.add_argument("--spectrum-probes", type=int, default=8)
    p.add_argument("--lanczos-steps", type=int, default=120)
    p.add_argument("--nullspace-cutoff", type=float, default=1e-4)
    p.add_argument("--background-erosion", type=int, default=6)
    p.add_argument("--seed", type=int, default=20260830)
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT))
    run(p.parse_args())


if __name__ == "__main__":
    main()
