#!/usr/bin/env python
"""Second non-Cartesian dataset: in vivo brain spiral (32 channels, 16
interleaves, 220 x 220; ``experiments/brain_data``), run through the same
pipeline as the phantom of the primary grid.

Stages (``--stage``):

quality   Reference image (all 16 interleaves, converged CG-SENSE at a small
          Tikhonov weight), then at R = 2 (every second interleaf): the
          lambda = 0 semiconvergence (fixed-iteration snapshots: NRMSE against
          the reference, solution change between checkpoints, true residual)
          and the converged reconstruction at every lambda_norm of the grid
          (iterations, true residual, data residual, NRMSE / PSNR / SSIM
          against the reference on the scale-aligned magnitude, out-of-object
          streak energy).  Writes ``quality.json``, ``reference_image.npy``,
          ``analysis_mask_B.npy`` (5 % support of the reference) and the
          reconstructions.
matched   PICO versus PMR at one common physical lambda with the primary
          grid's matched protocol (``run_noncart_weekend_audit.run_matched``,
          recycled two-solve probe, noise-only PMR); score with
          ``summarize_ratio_curve.py --mask <out>/analysis_mask_B.npy``.

Operator: unweighted pre-whitened A^H A (no DCF, no B0 correction), sigpy
NUFFT width 4, Toeplitz normal operator, zero initialization, true-residual
stopping, exactly as for the phantom.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from mr_recon.dtypes import complex_dtype, real_dtype  # noqa: E402
from mr_recon.fourier import sigpy_nufft  # noqa: E402
from mr_recon.linops import batching_params, sense_linop  # noqa: E402
from scripts import run_noncart_weekend_audit as wa  # noqa: E402
from scripts.noncart_audit_core import solve_cg_true_residual  # noqa: E402

DATA = ROOT / "experiments/brain_data"
OUT_DEFAULT = ROOT / "experiments/brain_spiral/results"
GRID = [0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
SNAPSHOTS = [25, 50, 100, 200, 400, 800, 1200]


def load_brain(device):
    def ld(name, dtype):
        return torch.load(DATA / f"{name}.pt", map_location="cpu").to(device=device, dtype=dtype)

    return ld("ksp", complex_dtype), ld("mps", complex_dtype), ld("trj", real_dtype), ld("dcf", real_dtype)


def build_operators(R: int, device):
    ksp, mps, trj, dcf = load_brain(device)
    im_size = tuple(mps.shape[-2:])
    nufft = sigpy_nufft(im_size, width=4)
    bparams = batching_params(mps.shape[0])
    A_ref = sense_linop(im_size, trj, mps, dcf, nufft, bparams=bparams, use_toeplitz=True, weight_data_consistency=False)
    A_acc = sense_linop(im_size, trj[:, ::R, :], mps, dcf[:, ::R], nufft, bparams=bparams, use_toeplitz=True, weight_data_consistency=False)
    mu_ref = wa.estimate_max_eigen(A_ref, seed=1701)
    mu_acc = wa.estimate_max_eigen(A_acc, seed=1701)
    return ksp, mps, A_ref, A_acc, mu_ref, mu_acc


def converged_solve(A, mu, lam_norm, ksp, max_iter=1200, rtol=1e-5):
    def normal(x):
        return A.normal(x) / mu + lam_norm * x

    rhs = A.adjoint(ksp.type(complex_dtype)) / math.sqrt(mu)
    x, info = solve_cg_true_residual(normal, rhs, max_iter=max_iter, rtol=rtol, true_residual_every=10)
    return x / math.sqrt(mu), info, rhs  # physical-scale image


def build_brain_problem(R: int, device, ref_lambda_norm: float = 1e-4, out: Path = OUT_DEFAULT,
                        reference: str = "cg"):
    """Drop-in replacement for ``run_noncart_weekend_audit.build_problem``.

    ``reference="cg"``: fully sampled (all interleaves) converged CG-SENSE at
    ``ref_lambda_norm``.  ``reference="gridding"``: density-compensated adjoint
    of all interleaves, coil-combined by the sensitivity maps, i.e. the
    regularization-free gridding reconstruction (the DCF enters only here).
    """
    ksp, mps, A_ref, A_acc, mu_ref, mu_acc = build_operators(R, device)
    ref_path = out / "reference_image.npy"
    if ref_path.exists():
        image = torch.tensor(np.load(ref_path), device=device, dtype=complex_dtype)
    else:
        out.mkdir(parents=True, exist_ok=True)
        if reference == "gridding":
            _, _, _, dcf = load_brain(device)
            image = A_ref.adjoint((ksp * dcf[None]).type(complex_dtype))
            meta = {"reference": "gridding (DCF-weighted adjoint, coil-combined)", "reference_lambda_norm": None,
                    "max_eigen_ref": mu_ref, "max_eigen_acc": mu_acc}
        else:
            image, info, _ = converged_solve(A_ref, mu_ref, ref_lambda_norm, ksp)
            meta = {"reference": "converged CG-SENSE, all interleaves", "reference_lambda_norm": ref_lambda_norm,
                    "iterations": info["iterations"],
                    "true_relative_residual": info["history"][-1]["true_relative_residual"],
                    "max_eigen_ref": mu_ref, "max_eigen_acc": mu_acc}
        np.save(ref_path, image.cpu().numpy())
        json.dump(meta, open(out / "reference_image.json", "w"), indent=2)
    support = image.abs() > 0.05 * image.abs().max()
    np.save(out / "analysis_mask_B.npy", support.cpu().numpy())
    return wa.Problem(R=R, device=device, image=image, support=support, ksp_ref=ksp, ksp_acc=ksp[:, :, ::R],
                      A_ref=A_ref, A_acc=A_acc, max_eigen_ref=mu_ref, max_eigen_acc=mu_acc)


def aligned_metrics(x: torch.Tensor, ref: torch.Tensor, support: torch.Tensor) -> dict:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    xs, rs = x[support], ref[support]
    a = torch.vdot(xs, rs) / torch.vdot(xs, xs)
    xa = (a * x)
    nrmse = float(torch.linalg.norm(xa[support] - rs) / torch.linalg.norm(rs))
    refm = ref.abs().cpu().numpy().astype(np.float64)
    xm = xa.abs().cpu().numpy().astype(np.float64)
    rng = float(refm.max())
    bg = ~support.cpu().numpy()
    return {"nrmse_aligned_support": nrmse,
            "psnr_db": float(peak_signal_noise_ratio(refm, np.clip(xm, 0, None), data_range=rng)),
            "ssim": float(structural_similarity(refm, xm, data_range=rng)),
            "streak_pct": 100 * float(xm[bg].mean() / xm[~bg].mean()),
            "background_p99_over_signal_mean": float(np.percentile(xm[bg], 99) / xm[~bg].mean())}


def stage_quality(args, device) -> None:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    problem = build_brain_problem(args.R, device, args.ref_lambda_norm, out, args.reference)
    ref = problem.image
    result = {"R": args.R, "max_eigen_ref": problem.max_eigen_ref, "max_eigen_acc": problem.max_eigen_acc,
              "reference": json.load(open(out / "reference_image.json")),
              "coils": int(problem.A_ref.ishape[0]) if hasattr(problem.A_ref, "ishape") and len(problem.A_ref.ishape) == 3 else None,
              "grid_lambda_norm_acc": GRID, "semiconvergence": [], "converged": []}
    rhs_acc = problem.A_acc.adjoint(problem.ksp_acc.type(complex_dtype)) / math.sqrt(problem.max_eigen_acc)

    # lambda = 0 semiconvergence: fixed-iteration snapshots
    def H0(x):
        return problem.A_acc.normal(x) / problem.max_eigen_acc
    _, info0 = solve_cg_true_residual(H0, rhs_acc, max_iter=max(SNAPSHOTS), rtol=0.0, true_residual_every=10, snapshot_iters=SNAPSHOTS)
    prev = None
    hist = {h["iteration"]: h["true_relative_residual"] for h in info0["history"]}
    for k in SNAPSHOTS:
        xk = info0["snapshots"][k] / math.sqrt(problem.max_eigen_acc)
        m = aligned_metrics(xk, ref, problem.support)
        m.update({"iteration": k, "true_relative_residual": hist.get(k),
                  "solution_change_pct": None if prev is None else 100 * float(torch.linalg.norm(xk - prev) / torch.linalg.norm(xk))})
        result["semiconvergence"].append(m)
        np.save(out / f"lambda0_iter{k}_acc_physical.npy", xk.cpu().numpy())
        prev = xk
        print(f"[brain] lambda=0 K={k}: NRMSE {100*m['nrmse_aligned_support']:.2f}% SSIM {m['ssim']:.3f} streak {m['streak_pct']:.2f}% resid {m['true_relative_residual']}", flush=True)

    # converged reconstruction at every positive lambda of the grid (common physical lambda, acc-arm normalized value)
    floor = None
    for lam in GRID:
        if lam == 0.0:
            continue
        lam_phys = lam * problem.max_eigen_acc
        x, info, _ = converged_solve(problem.A_acc, problem.max_eigen_acc, lam, problem.ksp_acc, rtol=args.rtol)
        data_res = float(torch.linalg.norm(problem.A_acc.forward(x) - problem.ksp_acc) / torch.linalg.norm(problem.ksp_acc))
        m = aligned_metrics(x, ref, problem.support)
        m.update({"lambda_norm": lam, "lambda_phys": lam_phys, "iterations": info["iterations"], "converged": info["converged"],
                  "true_relative_residual": info["history"][-1]["true_relative_residual"], "data_residual_rel": data_res})
        result["converged"].append(m)
        np.save(out / f"recon_lambda{lam:.0e}_acc_physical.npy", x.cpu().numpy())
        print(f"[brain] lambda_norm={lam:g}: it={info['iterations']} resid={m['true_relative_residual']:.2e} data={data_res:.5f} "
              f"NRMSE {100*m['nrmse_aligned_support']:.2f}% PSNR {m['psnr_db']:.2f} SSIM {m['ssim']:.3f} streak {m['streak_pct']:.2f}%", flush=True)
    floor = min(r["data_residual_rel"] for r in result["converged"])
    for r in result["converged"]:
        r["data_residual_excess_pct"] = 100 * (r["data_residual_rel"] / floor - 1)
    # simple selection by the phantom's rules: largest lambda within 0.25% of the residual floor; SSIM max; streak min
    ok = [r for r in result["converged"] if r["data_residual_excess_pct"] <= 0.25]
    result["selection"] = {
        "largest_lambda_within_0.25pct_of_floor": max(ok, key=lambda r: r["lambda_norm"])["lambda_norm"] if ok else None,
        "ssim_max_lambda": max(result["converged"], key=lambda r: r["ssim"])["lambda_norm"],
        "psnr_max_lambda": max(result["converged"], key=lambda r: r["psnr_db"])["lambda_norm"],
        "streak_min_lambda": min(result["converged"], key=lambda r: r["streak_pct"])["lambda_norm"],
        "nrmse_min_lambda": min(result["converged"], key=lambda r: r["nrmse_aligned_support"])["lambda_norm"],
    }
    result["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    json.dump(result, open(out / "quality.json", "w"), indent=2)
    print("[brain] selection:", json.dumps(result["selection"]), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["quality", "matched"], required=True)
    ap.add_argument("--R", type=int, default=2)
    ap.add_argument("--ref-lambda-norm", type=float, default=1e-4)
    ap.add_argument("--reference", choices=["cg", "gridding"], default="cg")
    ap.add_argument("--rtol", type=float, default=1e-5)
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    # matched-stage arguments (forwarded to run_noncart_weekend_audit.run_matched)
    ap.add_argument("--seed", type=int, default=20260830)
    ap.add_argument("--lambda-physical", type=float, default=None)
    ap.add_argument("--lambda-normalized", type=float, default=1e-3)
    ap.add_argument("--max-iter", type=int, default=1200)
    ap.add_argument("--true-residual-every", type=int, default=10)
    ap.add_argument("--N-max", type=int, default=2000)
    ap.add_argument("--checkpoints", type=int, nargs="+", default=[50, 100, 200, 500, 1000, 2000])
    ap.add_argument("--methods", nargs="+", choices=["pmr", "pico"], default=["pmr", "pico"])
    ap.add_argument("--pmr-sigma-scale", type=float, default=1.0,
                    help="multiply the injected PMR noise level (replica estimator only); the normalized map is level-independent for a linear reconstruction.")
    ap.add_argument("--pmr-common-noise", action="store_true",
                    help="coupled sampling: the accelerated arm takes the shot subset of the "
                         "reference arm's noise realization.")
    ap.add_argument("--pmr-estimator", choices=["noise_only", "replica"], default="noise_only",
                    help="noise_only: accumulate E|R(n)|^2. replica: Robson-style mean/variance over R(ksp + n).")
    args = ap.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.stage == "quality":
        stage_quality(args, device)
        return
    out = Path(args.out_dir)
    wa.build_problem = lambda R, dev: build_brain_problem(R, dev, args.ref_lambda_norm, out, args.reference)  # noqa: E731
    # attributes the phantom driver's config payload may read
    for name, value in (("mode", "matched"), ("resume", False), ("checkpoint_every", 500), ("N", args.N_max),
                        ("iterations", SNAPSHOTS), ("lambdas", GRID), ("n_probes", 16)):
        setattr(args, name, value)
    args.checkpoints = [c for c in args.checkpoints if c <= args.N_max]
    wa.run_matched(args)

if __name__ == "__main__":
    main()
