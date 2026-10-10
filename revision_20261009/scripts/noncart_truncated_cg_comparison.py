#!/usr/bin/env python
"""Unregularized (lambda = 0) PICO-versus-PMR comparison on truncated CG.

The reviewer asked for the non-Cartesian experiment without Tikhonov
regularization.  At lambda = 0 the converged inverse is not a usable target
(SI S7, Figure S4), but early-stopped conjugate gradients with a fixed
iteration count K is a perfectly well-defined reconstruction whose
regularization parameter is K (Qu et al. 2005).  This script compares the two
noise estimators on exactly that operator, on the same spiral problem, arms,
noise level, and analysis mask as the primary grid.

Operator.  Zero-initialized CG for K iterations on the normalized normal
equations (A^H A / mu) x = A^H b / sqrt(mu), with lambda = 0.  Two versions:

* frozen  -- the CG step coefficients (alpha_k, beta_k) recorded on the
  signal right-hand side are reused, so the map b -> x is the exact fixed
  linear operator R_K = p_K(H) A^H / sqrt(mu) (p_K the CG polynomial).  Both
  estimators target this operator: PMR by noise-only replicas
  (exact for a linear map) and PICO by the direct estimator
  diag(R_K R_K^H) = E Re{v* . p_K(H) H p_K(H) v}, two frozen K-step
  recurrences and one normal-operator apply per probe.
* adaptive -- ordinary CG whose coefficients depend on the data; its noise
  is measured by PMR replicas of (signal + noise) minus the signal
  reconstruction, which is what a user running early-stopped CG would see.
  The frozen/adaptive difference is reported.

Outputs mirror ``run_noncart_weekend_audit.py matched`` so that
``scripts/summarize_ratio_curve.py`` scores both the same way.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from mr_recon.dtypes import complex_dtype  # noqa: E402
from scripts.run_noncart_weekend_audit import (  # noqa: E402
    SIGMA,
    _append_or_replace_csv,
    _json_dump,
    _reference_maps,
    _save_npy,
    _save_reference_maps,
    _synced_time,
    build_problem,
)


def cg_fixed(H, rhs: torch.Tensor, K: int, coeffs=None, record: bool = False):
    """Zero-initialized CG for exactly K iterations.

    ``coeffs is None``: adaptive CG (coefficients from the residuals; a
    nonlinear map of ``rhs``).  ``coeffs`` given: the frozen recurrence, an
    exact linear map of ``rhs`` (the CG polynomial p_K(H) applied to rhs).
    """
    x = torch.zeros_like(rhs)
    r = rhs.clone()
    p = r.clone()
    rs = torch.real(torch.vdot(r.flatten(), r.flatten()))
    recorded = []
    for k in range(K):
        Hp = H(p)
        if coeffs is None:
            pHp = torch.real(torch.vdot(p.flatten(), Hp.flatten()))
            alpha = rs / pHp
            x = x + alpha * p
            r = r - alpha * Hp
            rs_new = torch.real(torch.vdot(r.flatten(), r.flatten()))
            beta = rs_new / rs
            rs = rs_new
            if record:
                recorded.append((float(alpha), float(beta)))
        else:
            alpha, beta = coeffs[k]
            x = x + alpha * p
            r = r - alpha * Hp
        p = r + beta * p
    return (x, recorded) if record else x


def aligned_nrmse(x: torch.Tensor, ref: torch.Tensor, support: torch.Tensor) -> float:
    xs, rs = x[support], ref[support]
    a = torch.vdot(xs, rs) / torch.vdot(xs, xs)
    return float(torch.linalg.norm(a * xs - rs) / torch.linalg.norm(rs))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--R", type=int, default=2)
    ap.add_argument("--K", type=int, required=True, help="fixed CG iteration count")
    ap.add_argument("--seed", type=int, default=20260830)
    ap.add_argument("--N-max", type=int, default=1000, help="replicas / probes for the frozen operator")
    ap.add_argument("--N-adaptive", type=int, default=500, help="signal+noise replicas through adaptive CG")
    ap.add_argument("--checkpoints", type=int, nargs="+", default=[50, 100, 200, 500, 1000, 2000])
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--pmr-common-noise", action="store_true",
                    help="coupled sampling: the accelerated arm takes the shot subset of the "
                         "reference arm's noise realization.")
    ap.add_argument("--pmr-sigma-scale", type=float, default=1.0,
                    help="multiply the injected PMR noise level (replica estimator only); the normalized map is level-independent for a linear reconstruction.")
    ap.add_argument("--pmr-estimator", choices=["noise_only", "replica"], default="noise_only",
                    help="noise_only: accumulate E|R(n)|^2. replica: Robson-style mean/variance over R(ksp + n).")
    ap.add_argument("--methods", nargs="+", choices=["pmr", "pico", "adaptive"], default=["pmr", "pico", "adaptive"],
                    help="which arms to run (a PMR-only run with a fresh seed and a large N is an independent reference)")
    ap.add_argument("--reference-seed", type=int, default=None,
                    help="also run an independent PMR reference (fresh noise seed, independent realizations for the "
                         "two reconstructions) on the SAME frozen operator as this run; written to <out-dir>/reference_seed<S>. "
                         "The late CG coefficients at lambda = 0 are not bit-reproducible across runs, so a reference from a "
                         "separate run would define a slightly different operator.")
    ap.add_argument("--reference-N-max", type=int, default=50000)
    ap.add_argument("--reference-checkpoints", type=int, nargs="+", default=[10000, 20000, 30000, 50000])
    ap.add_argument("--independent-seed", type=int, default=None,
                    help="also run a PMR series with independent noise realizations for the two reconstructions "
                         "(the ordinary pseudo-replica recipe) on the SAME frozen operator, with its own noise seed; "
                         "written to <out-dir>/independent_seed<S> (the shared-versus-independent comparison of SI S10).")
    ap.add_argument("--independent-N-max", type=int, default=5000)
    ap.add_argument("--independent-checkpoints", type=int, nargs="+", default=[50, 100, 200, 500, 1000, 2000, 5000])
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    problem = build_problem(args.R, device)
    out = Path(args.out_dir) if args.out_dir else (
        ROOT / f"experiments/noncartesian_phantom/results/truncated_cg_lam0_R{args.R}/K{args.K}/seed{args.seed}")
    out.mkdir(parents=True, exist_ok=True)

    arms = {}
    signal_metrics = {}
    for name, A, ksp, mu in (
        ("ref", problem.A_ref, problem.ksp_ref, problem.max_eigen_ref),
        ("acc", problem.A_acc, problem.ksp_acc, problem.max_eigen_acc),
    ):
        def H(x, A=A, mu=mu):
            return A.normal(x) / mu

        rhs0 = A.adjoint(ksp.type(complex_dtype)) / math.sqrt(mu)
        t0 = _synced_time()
        x0, coeffs = cg_fixed(H, rhs0, args.K, record=True)
        t_signal = _synced_time() - t0
        # true relative normal-equation residual at K
        resid = float(torch.linalg.norm(rhs0 - H(x0)) / torch.linalg.norm(rhs0))
        arms[name] = dict(A=A, ksp=ksp, mu=mu, H=H, rhs0=rhs0, x0=x0, coeffs=coeffs)
        _save_npy(out / f"signal_{name}_K{args.K}_normalized.npy", x0.cpu().numpy())
        signal_metrics[name] = {
            "true_relative_residual_at_K": resid,
            "nrmse_aligned_vs_reference_image_support": aligned_nrmse(x0 / math.sqrt(mu), problem.image, problem.support),
            "signal_solve_seconds": t_signal,
            "coefficients_first_last": [coeffs[0], coeffs[-1]],
        }
        print(f"[trunc] arm={name} K={args.K} residual={resid:.3e} nrmse={signal_metrics[name]['nrmse_aligned_vs_reference_image_support']:.4f}", flush=True)

    checkpoints = sorted({c for c in args.checkpoints if c <= args.N_max} | {args.N_max})
    ckpt_adaptive = sorted({c for c in args.checkpoints if c <= args.N_adaptive} | {args.N_adaptive})
    _json_dump(out / "config.json", {
        "R": args.R, "K": args.K, "seed": args.seed, "lambda": 0.0, "sigma": SIGMA,
        "operator": "zero-initialized truncated CG, K iterations, lambda = 0, unweighted A^H A (no DCF)",
        "frozen": "CG coefficients recorded on the signal right-hand side and reused: exact fixed linear operator",
        "pmr_frozen": "float64 noise-only replicas through the frozen recurrence",
        "pico_frozen": "direct estimator Re{v* p_K(H) H p_K(H) v}, random-phase probes",
        "pmr_adaptive": "replicas of (signal + noise) minus the signal reconstruction through adaptive CG",
        "max_eigen_ref": problem.max_eigen_ref, "max_eigen_acc": problem.max_eigen_acc,
        "N_max": args.N_max, "N_adaptive": args.N_adaptive, "checkpoints": checkpoints,
        "signal": signal_metrics, "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    })

    rows = []
    shape = problem.image.shape

    # ---------------- PMR on the frozen operator
    # replica: Robson-style mean/variance over reconstructions of ksp + n (Welford, float64).
    # noise_only: accumulate E|R(n)|^2, the same quantity for this linear operator.
    replica = getattr(args, "pmr_estimator", "noise_only") == "replica"
    sigma_scale = float(getattr(args, "pmr_sigma_scale", 1.0))
    g_ref = torch.Generator(device=device).manual_seed(args.seed + 401)
    g_acc = torch.Generator(device=device).manual_seed(args.seed + 509)
    sum_ref = torch.zeros(shape, dtype=torch.float64, device=device)
    sum_acc = torch.zeros_like(sum_ref)
    mean_ref = torch.zeros(shape, dtype=torch.complex128, device=device)
    mean_acc = torch.zeros_like(mean_ref)
    t0 = _synced_time()
    for i in range(1, (args.N_max if "pmr" in args.methods else 0) + 1):
        nr = (SIGMA * sigma_scale) * torch.randn(problem.ksp_ref.shape, dtype=problem.ksp_ref.dtype, device=device, generator=g_ref)
        na = (
            # coupled sampling: the accelerated arm is the shot subset of the
            # reference arm's realization
            nr[:, :, :: problem.R] if getattr(args, "pmr_common_noise", False)
            else (SIGMA * sigma_scale) * torch.randn(problem.ksp_acc.shape, dtype=problem.ksp_acc.dtype, device=device, generator=g_acc)
        )
        for name, n, acc, mean, ksp in (("ref", nr, sum_ref, mean_ref, problem.ksp_ref),
                                        ("acc", na, sum_acc, mean_acc, problem.ksp_acc)):
            a = arms[name]
            data = (ksp + n) if replica else n
            x = cg_fixed(a["H"], a["A"].adjoint(data.type(complex_dtype)) / math.sqrt(a["mu"]), args.K, coeffs=a["coeffs"])
            if replica:
                xd = x.to(torch.complex128)
                delta = xd - mean
                mean += delta / i
                acc += (delta.conj() * (xd - mean)).real / (sigma_scale ** 2)
            else:
                acc += x.abs().double().square()
        if i in checkpoints:
            _save_reference_maps(out, f"matched_pmr_N{i}", _reference_maps(problem, sum_ref, sum_acc, i))
            rows.append({"method": "PMR", "N": i, "cumulative_time_s": _synced_time() - t0})
            _append_or_replace_csv(out / "matched_progress.csv", rows)
            print(f"[trunc] K={args.K} PMR-frozen N={i}", flush=True)

    # ---------------- PICO direct on the frozen operator
    g_probe = torch.Generator(device=device).manual_seed(args.seed + 601)
    diag_ref = torch.zeros(shape, dtype=torch.float64, device=device)
    diag_acc = torch.zeros_like(diag_ref)
    t0 = _synced_time()
    for i in range(1, (args.N_max if "pico" in args.methods else 0) + 1):
        phase = 2 * torch.pi * torch.rand(shape, device=device, generator=g_probe)
        v = torch.polar(torch.ones_like(phase), phase).type(complex_dtype)
        for name, acc in (("ref", diag_ref), ("acc", diag_acc)):
            a = arms[name]
            u = cg_fixed(a["H"], v, args.K, coeffs=a["coeffs"])          # p_K(H) v
            y = cg_fixed(a["H"], a["H"](u), args.K, coeffs=a["coeffs"])  # p_K(H) H p_K(H) v
            acc += (v.conj() * y).real.double()
        if i in checkpoints:
            # sigma^2 * diag / i is the noise variance in normalized units, the same quantity PMR accumulates
            _save_reference_maps(out, f"matched_pico_N{i}", _reference_maps(problem, SIGMA ** 2 * diag_ref, SIGMA ** 2 * diag_acc, i))
            _save_npy(out / f"matched_pico_N{i}_diag_acc.npy", (diag_acc / i).cpu().numpy())
            _save_npy(out / f"matched_pico_N{i}_diag_ref.npy", (diag_ref / i).cpu().numpy())
            rows.append({"method": "PICO_random_phase", "N": i, "cumulative_time_s": _synced_time() - t0})
            _append_or_replace_csv(out / "matched_progress.csv", rows)
            print(f"[trunc] K={args.K} PICO-frozen N={i} neg_frac_acc={(diag_acc < 0).double().mean().item():.4f}", flush=True)

    # ---------------- PMR on the adaptive operator (signal + noise minus signal)
    g_ref = torch.Generator(device=device).manual_seed(args.seed + 701)
    g_acc = torch.Generator(device=device).manual_seed(args.seed + 809)
    sum_ref.zero_(); sum_acc.zero_()
    t0 = _synced_time()
    for i in range(1, (args.N_adaptive if "adaptive" in args.methods else 0) + 1):
        nr = SIGMA * torch.randn(problem.ksp_ref.shape, dtype=problem.ksp_ref.dtype, device=device, generator=g_ref)
        na = SIGMA * torch.randn(problem.ksp_acc.shape, dtype=problem.ksp_acc.dtype, device=device, generator=g_acc)
        for name, n, acc in (("ref", nr, sum_ref), ("acc", na, sum_acc)):
            a = arms[name]
            rhs = a["A"].adjoint((a["ksp"] + n).type(complex_dtype)) / math.sqrt(a["mu"])
            x = cg_fixed(a["H"], rhs, args.K) - a["x0"]
            acc += x.abs().double().square()
        if i in ckpt_adaptive:
            _save_reference_maps(out, f"matched_pmr_adaptive_N{i}", _reference_maps(problem, sum_ref, sum_acc, i))
            rows.append({"method": "PMR_adaptive", "N": i, "cumulative_time_s": _synced_time() - t0})
            _append_or_replace_csv(out / "matched_progress.csv", rows)
            print(f"[trunc] K={args.K} PMR-adaptive N={i}", flush=True)

    # ---------------- extra PMR streams with independent noise realizations on the SAME frozen operator:
    # a large independent reference (--reference-seed) and an ordinary independent-realization series
    # (--independent-seed) for the shared-versus-independent comparison of SI S10.
    def _independent_pmr_stream(seed: int, n_max: int, ckpts: list[int], subdir: str, tag: str) -> None:
        rd = out / subdir
        rd.mkdir(parents=True, exist_ok=True)
        cset = set(ckpts)
        g_r = torch.Generator(device=device).manual_seed(seed + 401)
        g_a = torch.Generator(device=device).manual_seed(seed + 509)
        s_ref = torch.zeros(shape, dtype=torch.float64, device=device)
        s_acc = torch.zeros_like(s_ref)
        rrows = []
        t_start = _synced_time()
        for i in range(1, n_max + 1):
            nr_ = SIGMA * torch.randn(problem.ksp_ref.shape, dtype=problem.ksp_ref.dtype, device=device, generator=g_r)
            na_ = SIGMA * torch.randn(problem.ksp_acc.shape, dtype=problem.ksp_acc.dtype, device=device, generator=g_a)
            for name, n, acc in (("ref", nr_, s_ref), ("acc", na_, s_acc)):
                a = arms[name]
                x = cg_fixed(a["H"], a["A"].adjoint(n.type(complex_dtype)) / math.sqrt(a["mu"]), args.K, coeffs=a["coeffs"])
                acc += x.abs().double().square()
            if i in cset:
                _save_reference_maps(rd, f"matched_pmr_N{i}", _reference_maps(problem, s_ref, s_acc, i))
                rrows.append({"method": "PMR", "N": i, "cumulative_time_s": _synced_time() - t_start})
                _append_or_replace_csv(rd / "matched_progress.csv", rrows)
                print(f"[trunc] K={args.K} PMR-{tag} N={i}", flush=True)
        _json_dump(rd / "config.json", {"R": args.R, "K": args.K, "seed": seed, "lambda": 0.0,
                                        "parent_seed": args.seed, "N_max": n_max, "checkpoints": sorted(cset), "sigma": SIGMA,
                                        "operator": "same frozen CG coefficients as the parent run (recorded on the signal right-hand side)",
                                        "sampling": "independent noise realizations for the fully sampled and accelerated reconstructions",
                                        "role": tag,
                                        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"})

    if args.reference_seed is not None and args.reference_N_max > 0:
        _independent_pmr_stream(args.reference_seed, args.reference_N_max, args.reference_checkpoints,
                                f"reference_seed{args.reference_seed}", "reference")
    if args.independent_seed is not None and args.independent_N_max > 0:
        _independent_pmr_stream(args.independent_seed, args.independent_N_max, args.independent_checkpoints,
                                f"independent_seed{args.independent_seed}", "independent")

    _json_dump(out / "status.json", {"status": "complete", "K": args.K, "N_max": args.N_max,
                                     "N_adaptive": args.N_adaptive,
                                     "pico_negative_diagonal_fraction_acc": float((diag_acc < 0).double().mean()),
                                     "pico_negative_diagonal_fraction_ref": float((diag_ref < 0).double().mean())})
    print("[trunc] COMPLETE", flush=True)


if __name__ == "__main__":
    main()
