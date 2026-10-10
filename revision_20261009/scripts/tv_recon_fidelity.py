"""Reconstruction fidelity and FISTA convergence across the TV sweep (SI S4).

Reruns the compressed-sensing reconstruction of ``run_lambda_tv_sweep.py``
(slice 20, R = 2, 100 FISTA iterations, same mask, maps, and normalization)
at every lambda_TV under both TV proximal step rules, the fixed Chambolle step
tau = 0.25 and the stabilized tau = min(0.25, 1/(8 alpha lambda)), recording a
per-iteration trace (relative iterate change, data term, maximum magnitude)
and the final image.  ``plot``/metric evaluation against the fully sampled
reference is done separately on CPU.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from einops import rearrange
from sigpy.mri import poisson

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mr_recon.algs import power_method_operator  # noqa: E402
from mr_recon.fourier import gridded_nufft  # noqa: E402
from mr_recon.linops import batching_params, sense_linop  # noqa: E402
from mr_recon.prox import TV  # noqa: E402
from mr_recon.utils import gen_grd, np_to_torch  # noqa: E402


def fista_traced(AHA, AHb, proxg, num_iters, ptol_exit=0.5, A=None, ksp=None):
    """The toolbox FISTA loop (mr_recon.algs.FISTA) with a per-iteration trace."""
    x = AHb.clone()
    z = x.clone()
    trace = []
    for k in range(num_iters):
        x_old = x.clone()
        x = z.clone()
        gr = AHA(x) - AHb
        x = proxg(x - gr)
        z = x if k == 0 else x + (k / (k + 3)) * (x - x_old)
        ptol = 100 * torch.norm(x_old - x) / torch.norm(x)
        rec = {"iteration": k + 1, "relative_change_pct": float(ptol),
               "max_magnitude": float(x.abs().max()), "finite": bool(torch.isfinite(x).all())}
        if A is not None and ksp is not None and rec["finite"]:
            rec["data_term"] = float(torch.linalg.norm(A.forward(x) - ksp) ** 2)
        trace.append(rec)
        if not rec["finite"]:
            break
        if ptol < ptol_exit:
            break
    return x, trace


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slice-idx", type=int, default=20)
    ap.add_argument("--accel", type=int, default=2)
    ap.add_argument("--lamdas", type=float, nargs="+", default=[1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0])
    ap.add_argument("--num-iters", type=int, default=100)
    ap.add_argument("--data-file", default="experiments/fastmri_knee/data/file1000000.h5")
    ap.add_argument("--out-dir", default="experiments/fastmri_knee/results/lambda_tv_recon_fidelity")
    ap.add_argument("--seed", type=int, default=20260830)
    ap.add_argument("--tv-inner", type=int, default=50)
    ap.add_argument("--tv-tol", type=float, default=1e-4)
    ap.add_argument("--tv-prox", choices=["magnitude", "complex"], default="magnitude")
    ap.add_argument("--fixed-K", type=int, default=None, help="run exactly K FISTA iterations (no 0.5%% exit)")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with h5py.File(args.data_file, "r") as f:
        img = torch.tensor(f["reconstruction_rss"], device=dev)[args.slice_idx]
        mps = torch.tensor(f["jsense-12-cf=4"]["maps"], device=dev)[args.slice_idx, :, :, :, 0]
        ksp_full = torch.tensor(f["kspace"][:], device=dev)[args.slice_idx]
    im_size = (img.shape[0], img.shape[1])
    C = mps.shape[-1]
    mps = rearrange(mps, "h w c -> c h w")
    if ksp_full.ndim == 3 and ksp_full.shape[-1] == C:
        ksp_full = rearrange(ksp_full, "h w c -> c h w")
    mask = poisson(im_size, accel=args.accel, calib=(24, 24), tol=0.3, crop_corner=False)
    mask = np_to_torch(mask.real).type(torch.float32).to(dev)
    inds = torch.argwhere(mask > 0)
    trj = gen_grd(im_size, im_size).to(dev)[inds[:, 0], inds[:, 1]]
    ksp = (ksp_full * mask.unsqueeze(0).expand(C, -1, -1))[:, inds[:, 0], inds[:, 1]]
    A = sense_linop(im_size, trj, mps, torch.ones(trj.shape[0], device=dev),
                    nufft=gridded_nufft(im_size), bparams=batching_params(C))
    _, max_eigen = power_method_operator(A.normal, torch.randn_like(img), verbose=False)
    max_eigen = float(max_eigen) * 1.01
    AHb = A.adjoint(ksp) / (max_eigen ** 0.5)
    AHA = lambda x: A.normal(x) / max_eigen  # noqa: E731
    out = Path(args.out_dir)
    (out / "maps").mkdir(parents=True, exist_ok=True)
    np.save(out / "maps" / "reference_rss.npy", img.abs().detach().cpu().numpy())
    zero_filled = A.adjoint(ksp)
    np.save(out / "maps" / "zero_filled_mag.npy", zero_filled.abs().detach().cpu().numpy())
    summary = {"slice": args.slice_idx, "accel": args.accel, "num_iters": args.num_iters, "tv_inner": args.tv_inner, "tv_tol": args.tv_tol, "tv_prox": args.tv_prox, "fixed_K": args.fixed_K,
               "max_eigen": max_eigen, "seed": args.seed, "sampling_fraction": float(mask.mean()),
               "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu", "runs": []}
    for lam in args.lamdas:
        for step in (("complex",) if args.tv_prox == "complex" else ("fixed", "stable")):
            if step == "complex":
                from mr_recon.prox_complex_tv import ComplexTV
                proxg = ComplexTV(im_size, lam, max_iter=args.tv_inner, tol=0.0)
                tau = 0.125
            else:
                proxg = TV(im_size=im_size, lamda=lam, norm="l2", stable_step=(step == "stable"), max_iter=args.tv_inner, tol=args.tv_tol)
                tau = min(0.25, 1.0 / (8.0 * lam)) if step == "stable" else 0.25
            x, trace = fista_traced(AHA, AHb, proxg, args.fixed_K or args.num_iters, ptol_exit=(-1.0 if args.fixed_K else 0.5), A=A, ksp=ksp)
            mag = x.abs().detach().cpu().numpy()
            np.save(out / "maps" / f"recon_mag_L{lam:g}_{step}.npy", mag)
            summary["runs"].append({
                "lamda": lam, "step_rule": step, "tau": tau, "eight_tau_lambda": 8 * tau * lam,
                "iterations_run": len(trace), "finite": bool(trace[-1]["finite"]),
                "final_relative_change_pct": trace[-1]["relative_change_pct"],
                "max_magnitude_final": trace[-1]["max_magnitude"],
                "trace": trace,
            })
            print(f"lambda={lam:g} {step:6s} tau={tau:.4f} 8*tau*lambda={8*tau*lam:.2f} "
                  f"it={len(trace)} finite={trace[-1]['finite']} change={trace[-1]['relative_change_pct']:.3g}% "
                  f"max|x|={trace[-1]['max_magnitude']:.3g}", flush=True)
    json.dump(summary, open(out / "fidelity_runs.json", "w"), indent=2)
    print("wrote", out / "fidelity_runs.json")


if __name__ == "__main__":
    main()
