"""
CS PICO-vs-PMR comparison at the manuscript configuration.

Reuses the manuscript CS primitives (``fista_reconstruct``,
``jvp_hutch_variance``, ``roi_mask``) imported from
``scripts/nonlinear_cs_noise_sweep.py`` and computes the PMR map as the
voxelwise mean and variance across FISTA reconstructions of ``ksp + noise``.

For each (R, noise_scale) it computes the PICO and PMR variance maps at matched
N and evaluates their agreement under four metric recipes, so that the effect of
the metric itself is visible:
    * manuscript : NRMSE(std) + Pearson(std) over the central-128 ROI, and the
                   ROI-mean-variance ratio
    * expC       : NRMSE(var) + Pearson(var) over the full image
    * audit      : NRMSE(var) + Pearson(var) over the central-128 ROI
    * robust     : NRMSE(var) + Pearson(var) over the signal support, excluding
                   PICO > p99

Outputs -> experiments/pmr_audit/cs_recheck/
"""
TV_INNER = 50
TV_TOL = 1e-4
TV_PROX = 'magnitude'
import os
import sys
import json
import time
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

# EXACT manuscript CS primitives (identical recon / PICO / PMR two-pass used by
# the main CS experiment and Exp C).
from nonlinear_cs_noise_sweep import (
    fista_reconstruct, jvp_hutch_variance, pmr_variance_map, roi_mask,
)


# --------------------------------------------------------------------------- #
# PMR reference (mean and variance across replicas), normalized like pmr_variance_map
# (complex noise (randn_re + 1j randn_im)*sigma ; divide accumulated var by sigma^2).
# --------------------------------------------------------------------------- #
def pmr_variance_map_f64(kspace, A, proxg, num_iters, max_eigen, sigma, n_replicas,
                         seed=None):
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


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
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


def all_recipe_metrics(ours_var, pmr_var, recon_mag, roi_size=128):
    """Return every metric recipe from the SAME pair of variance maps."""
    H, W = recon_mag.shape
    cy, cx = H // 2, W // 2
    hs = roi_size // 2
    roi = np.zeros((H, W), bool)
    roi[cy - hs:cy + hs, cx - hs:cx + hs] = True
    supp = recon_mag > 0.05 * recon_mag.max()

    ov, pv = np.maximum(ours_var, 0.0), np.maximum(pmr_var, 0.0)
    os_, ps_ = np.sqrt(ov), np.sqrt(pv)

    # robust support: drop the top 1% PICO-variance voxels (TV-kink outliers)
    keep = supp.copy()
    if supp.sum() > 0:
        thr = np.percentile(ov[supp], 99)
        keep = supp & (ov <= thr)
    med = np.median(ov[supp]) if supp.sum() else 0.0
    n_out = int((ov[supp] > 10 * med).sum()) if supp.sum() else 0

    m = {}
    # manuscript recipe: std map, central ROI
    m['manuscript_nrmse_std_roi_pct'] = _nrmse(os_[roi], ps_[roi]) * 100
    m['manuscript_corr_std_roi'] = _corr(os_[roi], ps_[roi])
    m['roi_mean_var_ratio'] = float(ov[roi].mean() / pv[roi].mean()) if pv[roi].mean() else float('nan')
    m['roi_median_var_ratio'] = float(np.median(ov[roi]) / np.median(pv[roi])) if np.median(pv[roi]) else float('nan')
    # Exp C recipe: var map, full image
    m['expC_nrmse_var_full_pct'] = _nrmse(ov, pv) * 100
    m['expC_corr_var_full'] = _corr(ov, pv)
    # audit recipe: var map, central ROI (this is the 6793% metric)
    m['audit_nrmse_var_roi_pct'] = _nrmse(ov[roi], pv[roi]) * 100
    m['audit_corr_var_roi'] = _corr(ov[roi], pv[roi])
    # audit-on-support (reconcile "naive"): var map, support
    m['audit_nrmse_var_supp_pct'] = _nrmse(ov[supp], pv[supp]) * 100
    m['audit_corr_var_supp'] = _corr(ov[supp], pv[supp])
    # robust: var map, support minus top-1% PICO outliers
    m['robust_nrmse_var_pct'] = _nrmse(ov[keep], pv[keep]) * 100
    m['robust_corr_var'] = _corr(ov[keep], pv[keep])
    m['n_pico_outliers_gt10xmedian'] = n_out
    m['pico_var_roi_max'] = float(ov[roi].max())
    m['pmr_var_roi_max'] = float(pv[roi].max())
    return m


# --------------------------------------------------------------------------- #
# per (R, noise) combo
# --------------------------------------------------------------------------- #
def build_operator(img, mps, ksp_full, dev, R):
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


def run_combo(args, img, mps, ksp_full, dev, R, noise_scale, out_root):
    im_size = tuple(img.shape[-2:])
    sigma = args.sigma0 * noise_scale
    tag = f"R{R}_noise{noise_scale:g}"
    out = os.path.join(out_root, tag)
    os.makedirs(out, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    A, ksp, n_samp = build_operator(img, mps, ksp_full, dev, R)
    if args.save_zero_filled:
        np.save(os.path.join(out, "zero_filled_mag.npy"), A.adjoint(ksp).abs().detach().cpu().numpy())
    if TV_PROX == 'complex':
        from mr_recon.prox_complex_tv import ComplexTV
        proxg = ComplexTV(tuple(im_size), args.lamda, max_iter=args.tv_inner, tol=args.tv_tol)
    else:
        proxg = TV(im_size=im_size, lamda=args.lamda, norm='l2', max_iter=args.tv_inner, tol=args.tv_tol)
    x0 = torch.randn_like(img)
    _, max_eigen = power_method_operator(A.normal, x0, verbose=False)
    max_eigen = float(max_eigen) * 1.01

    recon = fista_reconstruct(ksp, A, proxg, args.num_iters, max_eigen)
    recon_mag = recon.abs().detach().cpu().numpy()
    np.save(os.path.join(out, "recon_mag.npy"), recon_mag)

    print(f"\n=== {tag}: R={R} sigma={sigma:.2e} n_samples={n_samp} "
          f"N_pico={args.n_pico} N_pmr={args.n_pmr} ===", flush=True)

    t0 = time.time()
    ours_var = jvp_hutch_variance(ksp, A, proxg, args.num_iters, max_eigen,
                                  num_samples=args.n_pico, batch_size=args.batch_size)
    t_pico = time.time() - t0
    ours_np = ours_var.detach().cpu().numpy().astype(np.float64)
    np.save(os.path.join(out, "ours_var_pico.npy"), ours_np)

    t0 = time.time()
    pmr_var = pmr_variance_map_f64(ksp, A, proxg, args.num_iters, max_eigen,
                                   sigma=sigma, n_replicas=args.n_pmr, seed=args.seed + 777)
    t_pmr = time.time() - t0
    pmr_np = pmr_var.detach().cpu().numpy().astype(np.float64)
    np.save(os.path.join(out, "pmr_var_f64.npy"), pmr_np)

    # optional: manuscript float32 two-pass PMR, to certify f32 ~ f64
    tp_np = None
    if args.check_f32:
        torch.manual_seed(args.seed + 777)
        tp = pmr_variance_map(ksp, A, proxg, args.num_iters, max_eigen,
                              sigma=sigma, n_replicas=args.n_pmr)
        tp_np = tp.detach().cpu().numpy().astype(np.float64)
        np.save(os.path.join(out, "pmr_var_twopass_f32.npy"), tp_np)

    metrics = all_recipe_metrics(ours_np, pmr_np, recon_mag, roi_size=args.roi_size)
    metrics.update(dict(R=R, noise_scale=noise_scale, sigma=sigma, n_samples=n_samp,
                        n_pico=args.n_pico, n_pmr=args.n_pmr, num_iters=args.num_iters,
                        lamda=args.lamda, t_pico_s=t_pico, t_pmr_s=t_pmr, slice_idx=args.slice_idx,
                        device=torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu'))
    if tp_np is not None:
        metrics['f32_twopass_vs_f64_nrmse_var_pct'] = _nrmse(
            np.maximum(tp_np, 0), np.maximum(pmr_np, 0)) * 100

    print(f"  [manuscript] NRMSE(std,ROI)={metrics['manuscript_nrmse_std_roi_pct']:.2f}%  "
          f"corr(std,ROI)={metrics['manuscript_corr_std_roi']:.4f}  "
          f"ROI-mean-var ratio={metrics['roi_mean_var_ratio']:.4f}", flush=True)
    print(f"  [Exp C]      NRMSE(var,full)={metrics['expC_nrmse_var_full_pct']:.2f}%  "
          f"corr(var,full)={metrics['expC_corr_var_full']:.4f}", flush=True)
    print(f"  [AUDIT]      NRMSE(var,ROI)={metrics['audit_nrmse_var_roi_pct']:.1f}%  "
          f"corr(var,ROI)={metrics['audit_corr_var_roi']:.4f}   "
          f"(n_outliers>10xmed={metrics['n_pico_outliers_gt10xmedian']})", flush=True)
    print(f"  [robust]     NRMSE(var,supp\\p99)={metrics['robust_nrmse_var_pct']:.2f}%  "
          f"corr={metrics['robust_corr_var']:.4f}", flush=True)
    if tp_np is not None:
        print(f"  [f32 check]  two-pass-f32 vs f64 NRMSE(var)={metrics['f32_twopass_vs_f64_nrmse_var_pct']:.4f}%",
              flush=True)
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    return metrics


def analyze_existing_audit(out_root, roi_size=128):
    """Load the existing R=4 audit maps and score them with the same recipes."""
    rows = []
    base = "experiments/pmr_audit/cs"
    for ns, scale in [("noise_1", 1.0), ("noise_100", 100.0)]:
        d = os.path.join(base, ns)
        try:
            ours = np.load(os.path.join(d, "var_jvp.npy")).astype(np.float64)
            pmr = np.load(os.path.join(d, "var_welford_f64_ref.npy")).astype(np.float64)
            rec = np.load(os.path.join(d, "recon_mag.npy")).astype(np.float64)
        except Exception as e:
            print(f"  [audit-existing {ns}] MISSING ({e})", flush=True)
            continue
        m = all_recipe_metrics(ours, pmr, rec, roi_size=roi_size)
        m.update(dict(source="audit_existing_R4", R=4, noise_scale=scale, n_pmr=3000, n_pico=3000))
        rows.append(m)
        print(f"\n=== audit_existing R4 {ns}: reproduce the reported number ===", flush=True)
        print(f"  [AUDIT]      NRMSE(var,ROI)={m['audit_nrmse_var_roi_pct']:.1f}%  "
              f"corr(var,ROI)={m['audit_corr_var_roi']:.4f}  <-- should match the 6793%/0.016", flush=True)
        print(f"  [manuscript] NRMSE(std,ROI)={m['manuscript_nrmse_std_roi_pct']:.2f}%  "
              f"corr(std,ROI)={m['manuscript_corr_std_roi']:.4f}  "
              f"ROI-mean-var ratio={m['roi_mean_var_ratio']:.4f}", flush=True)
        print(f"  [robust]     NRMSE(var,supp\\p99)={m['robust_nrmse_var_pct']:.2f}%  "
              f"corr={m['robust_corr_var']:.4f}  (n_outliers={m['n_pico_outliers_gt10xmedian']})", flush=True)
    return rows


def main():
    p = argparse.ArgumentParser(description="CS PICO-vs-PMR recheck under manuscript pipeline.")
    p.add_argument('--slice-idx', type=int, default=20)
    p.add_argument('--Rs', type=int, nargs='+', default=[2, 4])
    p.add_argument('--noise-scales', type=float, nargs='+', default=[1.0, 100.0])
    p.add_argument('--lamda', type=float, default=1e-2)
    p.add_argument('--sigma0', type=float, default=1e-7)
    p.add_argument('--num-iters', type=int, default=100)
    p.add_argument('--n-pico', type=int, default=2000)
    p.add_argument('--n-pmr', type=int, default=2000)
    p.add_argument('--batch-size', type=int, default=50)
    p.add_argument('--roi-size', type=int, default=128)
    p.add_argument('--check-f32', action='store_true',
                   help='also compute the manuscript float32 two-pass PMR to certify f32~f64')
    p.add_argument('--data-file', type=str, default='experiments/fastmri_knee/data/file1000000.h5')
    p.add_argument('--out-dir', type=str, default='experiments/pmr_audit/cs_recheck')
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--tv-inner', type=int, default=50, help='Chambolle inner iterations of the TV prox (toolbox default 50)')
    p.add_argument('--tv-tol', type=float, default=1e-4, help='absolute exit tolerance on the dual increment (toolbox default 1e-4; 0 forces --tv-inner iterations)')
    p.add_argument('--fista-ptol', type=float, default=0.5, help='FISTA iterate-change exit rule in percent (toolbox default 0.5); a negative value runs exactly --num-iters iterations')
    p.add_argument('--tv-prox', choices=['magnitude', 'complex'], default='magnitude', help='TV proximal operator: magnitude/phase TV (toolbox) or complex-valued vector TV')
    p.add_argument('--save-zero-filled', action='store_true',
                   help='also save the zero-filled magnitude A^H b (main Figure 6 top row)')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    _ptol = None if args.fista_ptol < 0 else float(args.fista_ptol)
    globals()['TV_PROX'] = args.tv_prox
    print(f'[tv-prox] operator={args.tv_prox}', flush=True)
    import nonlinear_cs_noise_sweep as _ncs, nonlinear_cs_N_comparison as _ncn
    _ncs.FISTA_PTOL_EXIT = _ptol; _ncn.FISTA_PTOL_EXIT = _ptol; globals()['FISTA_PTOL_EXIT'] = _ptol
    print(f'[fista] ptol_exit={_ptol} num_iters={args.num_iters}', flush=True)
    if args.smoke:
        args.Rs = [2, 4]
        args.noise_scales = [1.0, 100.0]
        args.n_pico = 128
        args.n_pmr = 256
        args.num_iters = 100
        args.batch_size = 32

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)
    with h5py.File(args.data_file, 'r') as f:  # read only the requested slice
        img = torch.tensor(f['reconstruction_rss'][args.slice_idx], device=dev)
        mps = torch.tensor(f['jsense-12-cf=4']['maps'][args.slice_idx, :, :, :, 0], device=dev)
        ksp_full = torch.tensor(f['kspace'][args.slice_idx], device=dev)
    C = mps.shape[-1]
    mps = rearrange(mps, 'h w c -> c h w')
    if ksp_full.ndim == 3 and ksp_full.shape[-1] == C:
        ksp_full = rearrange(ksp_full, 'h w c -> c h w')
    print(f"[cs_recheck] device={dev} img={tuple(img.shape)} coils={C} "
          f"Rs={args.Rs} noise_scales={args.noise_scales} "
          f"N_pico={args.n_pico} N_pmr={args.n_pmr} iters={args.num_iters}", flush=True)

    all_rows = []
    for R in args.Rs:
        for s in args.noise_scales:
            m = run_combo(args, img, mps, ksp_full, dev, R, s, args.out_dir)
            m['source'] = 'recheck_fresh'
            all_rows.append(m)

    # cross-check against the existing audit R=4 artifacts
    all_rows.extend(analyze_existing_audit(args.out_dir, roi_size=args.roi_size))

    import pandas as pd
    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(args.out_dir, "cs_recheck_decomposition.csv")
    df.to_csv(csv_path, index=False)
    with open(os.path.join(args.out_dir, "cs_recheck_all.json"), "w") as f:
        json.dump(all_rows, f, indent=2, default=float)
    print(f"\n[cs_recheck] wrote {csv_path}", flush=True)
    print("[cs_recheck] DONE", flush=True)


if __name__ == '__main__':
    main()
