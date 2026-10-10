# Authorized equal-budget cohort run; see dated manifest and job array.
# Equal-budget current complex-TV cohort benchmark, 2026-09-30.
"""Prepared equal-budget CS timing benchmark: one subject/slice per invocation.

Uses the current complex-TV pipeline: R=2, lambda=3e-7, fixed 30 FISTA
iterations, 50 complex-TV prox iterations, no data-dependent stopping,
noise SD 1e-7, N=2000 for each estimator, PICO batch size 25 and sequential
single-pass Welford PMR. GPU warmup is excluded; probe/noise generation and
variance accumulation are included in synchronized wall-clock timings.

Outputs must be written to a NEW dated run directory. The script rejects
an already completed output and nonfinite or negative variance; it
never silently changes the requested slice. Timing at equal N is distinct from an
independently validated time-to-accuracy target.
"""
import os
import sys
import json
import time
import platform
import hashlib
from pathlib import Path
import argparse

import numpy as np
import torch
import h5py
from einops import rearrange
from sigpy.mri import poisson

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mr_recon.fourier import gridded_nufft
from mr_recon.utils import np_to_torch, gen_grd
from mr_recon.algs import power_method_operator
from mr_recon.prox import TV
from mr_recon.linops import sense_linop, batching_params
from mr_recon.pmr_variance import WelfordComplexVariance

# EXACT manuscript CS primitives (identical recon / PICO estimator used by the main
# CS experiment and Exp C run_lambda_tv_sweep.py) -- imported read-only.
from nonlinear_cs_noise_sweep import fista_reconstruct, jvp_hutch_variance

# TV proximal operator inner loop: the toolbox default (50 Chambolle iterations, absolute exit tolerance 1e-4
# on the dual increment) exits after ONE iteration at the fastMRI data scale (see
# scripts/tv_prox_one_step_check.py); pass --tv-tol 0 --tv-inner N to run a converged proximal operator.
TV_INNER = 50
TV_PROX = 'magnitude'  # 'magnitude': mr_recon.prox.TV (magnitude/phase); 'complex': mr_recon.prox_complex_tv.ComplexTV
TV_TOL = 1e-4


def build_operator(img, mps, ksp_full, dev, R):
    """Retrospective R-fold Poisson-disc CS SENSE operator (manuscript recipe)."""
    im_size = tuple(img.shape[-2:])
    mask = poisson(im_size, accel=R, calib=(24, 24), tol=0.3, crop_corner=False)
    mask = np_to_torch(mask.real).type(torch.float32).to(dev)
    inds = torch.argwhere(mask > 0)
    trj = gen_grd(im_size, im_size).to(dev)[inds[:, 0], inds[:, 1]]
    C = mps.shape[0]
    ksp = (ksp_full * mask.unsqueeze(0).expand(C, -1, -1))[:, inds[:, 0], inds[:, 1]]
    nufft = gridded_nufft(im_size)
    bparams = batching_params(C)
    dcf = torch.ones(trj.shape[0], device=dev)
    A = sense_linop(im_size, trj, mps, dcf, nufft=nufft, bparams=bparams)
    return A, ksp, int(inds.shape[0])


def pmr_variance_map_f64(kspace, A, proxg, num_iters, max_eigen, sigma, n_replicas, seed=None):
    """PMR variance map (mean and variance across replicas), normalized like
    the manuscript ``pmr_variance_map``: complex noise (randn_re + 1j randn_im)*sigma,
    variance divided by sigma^2."""
    dev = kspace.device
    im_size = tuple(A.im_size)
    if seed is not None:
        torch.manual_seed(seed)
    w = WelfordComplexVariance(im_size, dev, torch.complex128, torch.float64)
    for _ in range(n_replicas):
        noise = (torch.randn_like(kspace.real) + 1j * torch.randn_like(kspace.imag)) * sigma
        r = fista_reconstruct(kspace + noise, A, proxg, num_iters, max_eigen)
        w.update(r)
    return (w.variance / (sigma ** 2)).to(torch.float64)


def _nrmse(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    d = np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / d) if d > 0 else float('nan')


def _corr(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    if a.std() < 1e-30 or b.std() < 1e-30:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def run(args):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with h5py.File(args.data_file, 'r') as f:
        nslices = f['reconstruction_rss'].shape[0]
        slice_idx = int(args.slice_idx)
        if not 0 <= slice_idx < nslices:
            raise ValueError(f"requested slice {slice_idx} outside 0..{nslices-1}; no silent substitution")
        img = torch.tensor(f['reconstruction_rss'], device=dev)[slice_idx]
        mps = torch.tensor(f['jsense-12-cf=4']['maps'], device=dev)[slice_idx, :, :, :, 0]
        ksp_full = torch.tensor(f['kspace'][:], device=dev)[slice_idx]
    im_size = (img.shape[0], img.shape[1])
    C = mps.shape[-1]
    mps = rearrange(mps, 'h w c -> c h w')
    if ksp_full.ndim == 3 and ksp_full.shape[-1] == C:
        ksp_full = rearrange(ksp_full, 'h w c -> c h w')

    subj = os.path.splitext(os.path.basename(args.data_file))[0]
    print(f"[cs-subject] {subj} slice={slice_idx}/{nslices} img={im_size} coils={C} "
          f"R={args.accel} lamda_TV={args.lamda:g} sigma={args.sigma:g} "
          f"iters={args.num_iters} N_pmr={args.n_pmr} N_pico={args.n_pico}", flush=True)

    A, ksp, n_samp = build_operator(img, mps, ksp_full, dev, args.accel)
    if TV_PROX == 'complex':
        from mr_recon.prox_complex_tv import ComplexTV
        proxg = ComplexTV(tuple(im_size), args.lamda, max_iter=TV_INNER, tol=TV_TOL)
    else:
        proxg = TV(im_size=im_size, lamda=args.lamda, norm='l2', max_iter=TV_INNER, tol=TV_TOL)
    x0 = torch.randn_like(img)
    _, max_eigen = power_method_operator(A.normal, x0, verbose=False)
    max_eigen = float(max_eigen) * 1.01

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    final_json = Path(out_dir, "cs_subject_metrics.json")
    if final_json.exists():
        try:
            existing = json.loads(final_json.read_text())
        except json.JSONDecodeError:
            final_json.rename(final_json.with_name(f"cs_subject_metrics.incomplete_{int(time.time())}.json"))
        else:
            raise FileExistsError("Completed timing output already exists; resume wrapper should validate and skip it")
    recon = fista_reconstruct(ksp, A, proxg, args.num_iters, max_eigen)
    recon_mag = recon.abs().detach().cpu().numpy().astype(np.float64)
    np.save(os.path.join(out_dir, "recon_mag.npy"), recon_mag)

    if dev.type != 'cuda':
        raise RuntimeError("This benchmark requires the specified GPU; no silent CPU fallback")
    if args.n_pico != args.n_pmr:
        raise ValueError("Equal-budget cohort benchmark requires n_pico == n_pmr")
    # Baseline reconstruction above initializes the common operator. Warm both paths;
    # reset the RNG before measured work so warmup does not change the saved samples.
    sync = lambda: torch.cuda.synchronize(dev)
    sync()
    torch.manual_seed(args.seed + 100)
    _ = jvp_hutch_variance(ksp, A, proxg, args.num_iters, max_eigen,
                          num_samples=min(args.batch_size, args.n_pico), batch_size=args.batch_size)
    sync()
    _ = pmr_variance_map_f64(ksp, A, proxg, args.num_iters, max_eigen,
                            sigma=args.sigma, n_replicas=1, seed=args.seed + 777)
    sync()
    def measure_pico():
        torch.manual_seed(args.seed + 100)
        sync(); t0 = time.perf_counter()
        v = jvp_hutch_variance(ksp, A, proxg, args.num_iters, max_eigen,
                              num_samples=args.n_pico, batch_size=args.batch_size)
        sync(); elapsed = time.perf_counter() - t0
        return v.detach().cpu().numpy().astype(np.float64), elapsed
    def measure_pmr():
        torch.manual_seed(args.seed + 777)
        sync(); t0 = time.perf_counter()
        v = pmr_variance_map_f64(ksp, A, proxg, args.num_iters, max_eigen,
                                sigma=args.sigma, n_replicas=args.n_pmr, seed=None)
        sync(); elapsed = time.perf_counter() - t0
        return v.detach().cpu().numpy().astype(np.float64), elapsed
    if args.method_order == 'pico-first':
        ours_np, t_pico = measure_pico(); pmr_np, t_pmr = measure_pmr()
    else:
        pmr_np, t_pmr = measure_pmr(); ours_np, t_pico = measure_pico()
    # Persist raw estimates; do not replace invalid variance by zero.
    if not np.isfinite(ours_np).all() or not np.isfinite(pmr_np).all():
        raise ValueError("Nonfinite variance estimate; checkpoint invalid")
    if np.any(ours_np < 0) or np.any(pmr_np < 0):
        raise ValueError("Negative factor/PMR variance estimate; checkpoint invalid")
    np.save(os.path.join(out_dir, "pico_var.npy"), ours_np)
    np.save(os.path.join(out_dir, "pmr_var_f64.npy"), pmr_np)

    # Manuscript agreement recipe: NRMSE / Pearson on the STD map over central ROI.
    H, W = im_size
    cy, cx = H // 2, W // 2
    hs = args.roi_size // 2
    roi = np.zeros((H, W), bool)
    roi[cy - hs:cy + hs, cx - hs:cx + hs] = True
    os_, ps_ = np.sqrt(ours_np), np.sqrt(pmr_np)

    row = dict(
        subject=subj, slice_idx=slice_idx, nslices=int(nslices), accel=int(args.accel),
        lamda=float(args.lamda), sigma=float(args.sigma), num_iters=int(args.num_iters),
        n_pmr=int(args.n_pmr), n_pico=int(args.n_pico), n_samples=int(n_samp),
        nrmse_std_roi_pct=_nrmse(os_[roi], ps_[roi]) * 100.0,
        corr_std_roi=_corr(os_[roi], ps_[roi]),
        roi_mean_var_pico=float(ours_np[roi].mean()),
        roi_mean_var_pmr=float(pmr_np[roi].mean()),
        roi_mean_var_ratio=float(ours_np[roi].mean() / pmr_np[roi].mean())
        if pmr_np[roi].mean() > 0 else float('nan'),
        roi_mean_std_pico=float(os_[roi].mean()),
        roi_mean_std_pmr=float(ps_[roi].mean()),
        t_pico_s=t_pico, t_pmr_s=t_pmr,
        pmr_over_pico_runtime=t_pmr/t_pico,
        batch_size=int(args.batch_size), method_order=args.method_order,
        device=torch.cuda.get_device_name(dev), hostname=platform.node(),
        pytorch_version=torch.__version__, cuda_version=torch.version.cuda,
        tv_prox=TV_PROX, tv_inner=int(TV_INNER), tv_tol=float(TV_TOL),
        fista_ptol_exit=FISTA_PTOL_EXIT, seed=int(args.seed),
        roi_size=int(args.roi_size), image_shape=list(im_size), coils=int(C),
        timing_clock="perf_counter; CUDA synchronize before and after each method",
        timed_scope="probe/noise generation, all reconstruction/JVP applications, variance accumulation; excludes common setup, warmup, device-to-host copy and disk I/O",
        warmup_pico_samples=min(args.batch_size,args.n_pico), warmup_pmr_replicas=1,
        initialization="common power-method and baseline solve; every timed FISTA reconstruction uses the same code-defined initial iterate",
        pmr_accumulation="single-pass Welford complex128/float64",
        comparison_type="equal sample budgets; no claim of equal error or time-to-target",
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    
    )
    import pandas as pd
    pd.DataFrame([row]).to_csv(os.path.join(out_dir, "cs_subject_metrics.csv"), index=False)
    temporary_json = Path(out_dir, "cs_subject_metrics.json.tmp")
    temporary_json.write_text(json.dumps(row, indent=2, default=float) + "\n")
    temporary_json.replace(Path(out_dir, "cs_subject_metrics.json"))

    print(f"  NRMSE(std,ROI)={row['nrmse_std_roi_pct']:.2f}%  corr={row['corr_std_roi']:.4f}  "
          f"ROI-mean-var ratio (PICO/PMR)={row['roi_mean_var_ratio']:.4f}  "
          f"t_pico={t_pico:.1f}s t_pmr={t_pmr:.1f}s", flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Per-subject CS PICO-vs-PMR.")
    p.add_argument('--data-file', type=str, required=True)
    p.add_argument('--out-dir', type=str, required=True)
    p.add_argument('--slice-idx', type=int, default=20)
    p.add_argument('--accel', type=int, default=2)
    p.add_argument('--lamda', type=float, default=3e-7)
    p.add_argument('--sigma', type=float, default=1e-7)
    p.add_argument('--num-iters', type=int, default=30)
    p.add_argument('--n-pmr', type=int, default=2000)
    p.add_argument('--n-pico', type=int, default=2000)
    p.add_argument('--batch-size', type=int, default=25)
    p.add_argument('--roi-size', type=int, default=128)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--tv-inner', type=int, default=50, help='Chambolle inner iterations of the TV prox (toolbox default 50)')
    p.add_argument('--tv-tol', type=float, default=0.0, help='absolute exit tolerance on the dual increment (toolbox default 1e-4; 0 forces --tv-inner iterations)')
    p.add_argument('--fista-ptol', type=float, default=-1.0, help='FISTA iterate-change exit rule in percent (toolbox default 0.5); a negative value runs exactly --num-iters iterations')
    p.add_argument('--tv-prox', choices=['magnitude', 'complex'], default='complex', help='TV proximal operator: magnitude/phase TV (toolbox) or complex-valued vector TV')
    p.add_argument('--method-order', choices=['pico-first','pmr-first'], default='pico-first')
    args = p.parse_args()
    _ptol = None if args.fista_ptol < 0 else float(args.fista_ptol)
    globals()['TV_PROX'] = args.tv_prox
    print(f'[tv-prox] operator={args.tv_prox}', flush=True)
    import nonlinear_cs_noise_sweep as _ncs, nonlinear_cs_N_comparison as _ncn
    _ncs.FISTA_PTOL_EXIT = _ptol; _ncn.FISTA_PTOL_EXIT = _ptol; globals()['FISTA_PTOL_EXIT'] = _ptol
    print(f'[fista] ptol_exit={_ptol} num_iters={args.num_iters}', flush=True)
    TV_INNER, TV_TOL = args.tv_inner, args.tv_tol
    print(f'[tv-prox] inner={TV_INNER} tol={TV_TOL:g}', flush=True)
    run(args)
