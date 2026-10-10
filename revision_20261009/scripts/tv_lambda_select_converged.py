"""Select lambda_TV for the corrected TV proximal operator (SI S4, R3.2).

With the dual exit tolerance forced to zero the Chambolle loop actually runs
(``tv_prox_one_step_check.py``), so every lambda_TV of the executed sweep
has to be re-selected.  This script reconstructs the representative slices
(R = 2, same mask, maps, normalization and FISTA rule as the sweep) at a
lambda_TV grid with the fixed-count proximal operator (tol = 0, ``--inner``
Chambolle iterations) and scores PSNR / SSIM / NRMSE against the fully sampled
RSS image.  It also records, for the reconstruction at each lambda, how far
the ``--inner``-iteration proximal output is from a 1000-iteration one (the
proximal operator's own convergence), the FISTA iteration count under the
0.5 % rule, and the wall time per reconstruction (cost planning for the
rerun).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from einops import rearrange
from sigpy.mri import poisson
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mr_recon.algs import power_method_operator  # noqa: E402
from mr_recon.fourier import gridded_nufft  # noqa: E402
from mr_recon.linops import batching_params, sense_linop  # noqa: E402
from mr_recon.prox import TV  # noqa: E402
from mr_recon.utils import gen_grd, np_to_torch  # noqa: E402
from tv_recon_fidelity import fista_traced  # noqa: E402


def fidelity(mag, ref):
    a = float(np.vdot(mag, ref) / np.vdot(mag, mag))
    m = np.clip(a * mag, 0, None)
    rng = float(ref.max())
    return {"psnr_db": float(peak_signal_noise_ratio(ref, m, data_range=rng)),
            "ssim": float(structural_similarity(ref, m, data_range=rng)),
            "nrmse": float(np.linalg.norm(m - ref) / np.linalg.norm(ref))}


def setup(data_file, slice_idx, accel, dev):
    with h5py.File(data_file, "r") as f:
        img = torch.tensor(f["reconstruction_rss"], device=dev)[slice_idx]
        mps = torch.tensor(f["jsense-12-cf=4"]["maps"], device=dev)[slice_idx, :, :, :, 0]
        ksp_full = torch.tensor(f["kspace"][:], device=dev)[slice_idx]
    im_size = (img.shape[0], img.shape[1])
    C = mps.shape[-1]
    mps = rearrange(mps, "h w c -> c h w")
    if ksp_full.ndim == 3 and ksp_full.shape[-1] == C:
        ksp_full = rearrange(ksp_full, "h w c -> c h w")
    mask = poisson(im_size, accel=accel, calib=(24, 24), tol=0.3, crop_corner=False)
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
    return img, im_size, A, ksp, AHb, AHA, max_eigen


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slices", type=int, nargs="+", default=[20, 12, 28])
    ap.add_argument("--accel", type=int, default=2)
    ap.add_argument("--lamdas", type=float, nargs="+", default=[3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0])
    ap.add_argument("--inner", type=int, default=50)
    ap.add_argument("--num-iters", type=int, default=200)
    ap.add_argument("--data-file", default="experiments/fastmri_knee/data/file1000000.h5")
    ap.add_argument("--out-dir", default="experiments/fastmri_knee/results/lambda_tv_select_converged")
    ap.add_argument("--seed", type=int, default=20260830)
    ap.add_argument("--tv-prox", choices=["magnitude", "complex"], default="magnitude")
    ap.add_argument("--fixed-K", type=int, default=None, help="run exactly K FISTA iterations instead of the 0.5%% rule")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out_dir)
    (out / "maps").mkdir(parents=True, exist_ok=True)
    summary = {"accel": args.accel, "inner": args.inner, "tol": 0.0, "num_iters_cap": args.num_iters, "seed": args.seed, "tv_prox": args.tv_prox, "fixed_K": args.fixed_K,
               "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu", "slices": {}}
    for sl in args.slices:
        img, im_size, A, ksp, AHb, AHA, max_eigen = setup(args.data_file, sl, args.accel, dev)
        ref = img.abs().detach().cpu().numpy().astype(np.float64)
        np.save(out / "maps" / f"reference_rss_slice{sl}.npy", ref)
        rows = []
        for lam in args.lamdas:
            if args.tv_prox == "complex":
                from mr_recon.prox_complex_tv import ComplexTV
                proxg = ComplexTV(im_size, lam, max_iter=args.inner, tol=0.0)
            else:
                proxg = TV(im_size=im_size, lamda=lam, norm="l2", max_iter=args.inner, tol=0.0, stable_step=False)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            x, trace = fista_traced(AHA, AHb, proxg, args.fixed_K or args.num_iters, ptol_exit=(-1.0 if args.fixed_K else 0.5), A=A, ksp=ksp)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            mag = x.abs().detach().cpu().numpy().astype(np.float64)
            np.save(out / "maps" / f"recon_mag_slice{sl}_L{lam:g}_inner{args.inner}.npy", mag)
            # proximal-operator convergence at this lambda: inner vs 1000 iterations on the final iterate's gradient step
            g = x - (AHA(x) - AHb)
            p_inner = proxg(g)
            if args.tv_prox == "complex":
                p_long = ComplexTV(im_size, lam, max_iter=1000, tol=0.0)(g)
            else:
                p_long = TV(im_size=im_size, lamda=lam, norm="l2", max_iter=1000, tol=0.0, stable_step=False)(g)
            prox_gap = float(torch.linalg.norm(p_inner - p_long) / torch.linalg.norm(p_long))
            rec = {"lamda": lam, "fista_iterations": len(trace), "finite": bool(trace[-1]["finite"]), "change_trace_pct": [t["relative_change_pct"] for t in trace],
                   "final_relative_change_pct": trace[-1]["relative_change_pct"], "seconds": dt,
                   "prox_inner_vs_1000_rel_diff": prox_gap, "max_magnitude": trace[-1]["max_magnitude"]}
            rec.update(fidelity(mag, ref))
            rows.append(rec)
            print(f"slice {sl} lambda={lam:g}: it={len(trace)} change={rec['final_relative_change_pct']:.3g}% PSNR={rec['psnr_db']:.2f} "
                  f"SSIM={rec['ssim']:.3f} NRMSE={100*rec['nrmse']:.2f}% prox_gap={prox_gap:.3e} t={dt:.1f}s", flush=True)
        best = {"psnr": max(rows, key=lambda r: r["psnr_db"])["lamda"], "ssim": max(rows, key=lambda r: r["ssim"])["lamda"],
                "nrmse": min(rows, key=lambda r: r["nrmse"])["lamda"]}
        summary["slices"][str(sl)] = {"rows": rows, "best": best, "max_eigen": max_eigen}
        print(f"slice {sl} best: {best}", flush=True)
    json.dump(summary, open(out / "lambda_select.json", "w"), indent=2)
    print("wrote", out / "lambda_select.json")


if __name__ == "__main__":
    main()
