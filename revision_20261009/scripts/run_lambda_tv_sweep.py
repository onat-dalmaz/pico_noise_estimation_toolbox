"""
Exp C - High-lambda_TV CS sweep (answers R3.2: "explore the limits: study high
regularization parameters, not only different noise levels").

Sweeps the TV regularization strength lambda over a grid (default
{1e-2, 3e-2, 1e-1, 3e-1, 1e0}) on the fastMRI-knee compressed-sensing pipeline and,
for each lambda, compares the PICO (jvp Hutchinson) noise-variance map against the
PMR ground truth at matched N.  It also reports a TV "active-set" diagnostic - the
fraction of voxels whose reconstructed spatial-gradient magnitude is essentially zero
(where the L1/L2-TV prox is on its non-smooth face) - which grows with lambda and
characterizes where the first-order linearization of the reconstruction (the basis of
PICO for nonlinear recon) is expected to hold vs. break down.

Reuses the exact CS reconstruction / PMR / PICO primitives from
``scripts/nonlinear_cs_noise_sweep.py`` so the recon is identical to the main CS
experiment; only lambda is varied here.
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt
from einops import rearrange
from sigpy.mri import poisson

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mr_recon.fourier import gridded_nufft
from mr_recon.utils import np_to_torch, gen_grd
from mr_recon.algs import power_method_operator
from mr_recon.prox import TV
from mr_recon.linops import sense_linop, batching_params

# Reuse the identical CS recon / PMR / PICO estimators from the main CS script
from nonlinear_cs_noise_sweep import (
    fista_reconstruct, pmr_variance_map, jvp_hutch_variance,
    roi_mask, pearsonr, nrmse_percent,
)

# TV proximal operator inner loop: the toolbox default (50 Chambolle iterations, absolute exit tolerance 1e-4
# on the dual increment) exits after ONE iteration at the fastMRI data scale (see
# scripts/tv_prox_one_step_check.py); pass --tv-tol 0 --tv-inner N to run a converged proximal operator.
TV_INNER = 50
TV_PROX = 'magnitude'  # 'magnitude': mr_recon.prox.TV (magnitude/phase); 'complex': mr_recon.prox_complex_tv.ComplexTV
TV_TOL = 1e-4


def tv_active_set_fraction(img_mag: np.ndarray, rel_thresh: float = 1e-3) -> float:
    """Fraction of voxels whose finite-difference gradient magnitude is below
    ``rel_thresh`` * (max gradient) - i.e. voxels made piecewise-constant by TV.
    A large active set indicates strong non-smoothness of the recon map."""
    gx = np.abs(np.diff(img_mag, axis=-1, prepend=img_mag[..., :1]))
    gy = np.abs(np.diff(img_mag, axis=-2, prepend=img_mag[..., :1, :]))
    gmag = np.sqrt(gx ** 2 + gy ** 2)
    thr = rel_thresh * (gmag.max() + 1e-12)
    return float(np.mean(gmag <= thr))


def run(args):
    _seed = int(getattr(args, 'seed', 20260830))
    torch.manual_seed(_seed)
    np.random.seed(_seed)
    torch_dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Exp C] device={torch_dev} slice={args.slice_idx} R={args.accel} "
          f"lambdas={args.lamdas} N_pmr={args.n_pmr} N_pico={args.n_pico}")

    with h5py.File(args.data_file, 'r') as f:
        img = torch.tensor(f['reconstruction_rss'][args.slice_idx], device=torch_dev)
        mps = torch.tensor(f['jsense-12-cf=4']['maps'][args.slice_idx, :, :, :, 0], device=torch_dev)
        ksp_full = torch.tensor(f['kspace'][args.slice_idx], device=torch_dev)
    im_size = (img.shape[0], img.shape[1])
    C = mps.shape[-1]
    mps = rearrange(mps, 'h w c -> c h w')
    if ksp_full.ndim == 3 and ksp_full.shape[-1] == C:
        ksp_full = rearrange(ksp_full, 'h w c -> c h w')

    R = args.accel
    # sigpy's poisson() defaults to seed=0, so the sampling mask is already
    # deterministic and is deliberately left at that default: its numba kernel
    # sizes the active list at nx*ny and overflows (double free / SIGABRT) for
    # some other seeds. Only the estimator RNG is reseeded, above.
    mask = poisson(im_size, accel=R, calib=(24, 24), tol=0.3, crop_corner=False)
    mask = np_to_torch(mask.real).type(torch.float32).to(torch_dev)
    inds = torch.argwhere(mask > 0)
    trj = gen_grd(im_size, im_size).to(torch_dev)[inds[:, 0], inds[:, 1]]
    mask_expanded = mask.unsqueeze(0).expand(C, -1, -1)
    ksp = (ksp_full * mask_expanded)[:, inds[:, 0], inds[:, 1]]
    nufft = gridded_nufft(im_size)
    bparams = batching_params(C)
    dcf = torch.ones(trj.shape[0], device=torch_dev)
    A = sense_linop(im_size, trj, mps, dcf, nufft=nufft, bparams=bparams)

    # max_eigen does not depend on lambda -> compute once
    x0 = torch.randn_like(img)
    _, max_eigen = power_method_operator(A.normal, x0, verbose=False)
    max_eigen = float(max_eigen) * 1.01

    roi = roi_mask(im_size, mode=args.roi, roi_size=args.roi_size)
    sigma = args.sigma

    out_dir = os.path.join(args.save_dir, args.run_tag)
    os.makedirs(out_dir, exist_ok=True)
    maps_dir = os.path.join(out_dir, 'maps')
    os.makedirs(maps_dir, exist_ok=True)

    rows = []
    for lamda in args.lamdas:
        print(f"\n==== lambda_TV = {lamda:g} ====")
        if TV_PROX == 'complex':
            from mr_recon.prox_complex_tv import ComplexTV
            proxg = ComplexTV(tuple(im_size), lamda, max_iter=TV_INNER, tol=TV_TOL)
        else:
            proxg = TV(im_size=im_size, lamda=lamda, norm='l2',
            stable_step=bool(getattr(args, 'tv_stable_step', False)), max_iter=TV_INNER, tol=TV_TOL)
        img_recon = fista_reconstruct(ksp, A, proxg, args.num_iters, max_eigen)
        act = tv_active_set_fraction(img_recon.abs().detach().cpu().numpy())

        t0 = time.time()
        pmr_var = pmr_variance_map(ksp, A, proxg, args.num_iters, max_eigen, sigma=sigma,
                                   n_replicas=args.n_pmr)
        t_pmr = time.time() - t0
        t0 = time.time()
        ours_var = jvp_hutch_variance(ksp, A, proxg, args.num_iters, max_eigen,
                                      num_samples=args.n_pico, batch_size=args.batch_size)
        t_pico = time.time() - t0

        pmr_np = pmr_var.detach().cpu().numpy()
        ours_np = ours_var.detach().cpu().numpy()
        e = nrmse_percent(ours_np, pmr_np)      # PICO vs PMR agreement (%)
        r = pearsonr(ours_np, pmr_np)
        roi_pmr = float(np.mean(pmr_np[roi]))
        roi_ours = float(np.mean(ours_np[roi]))
        row = dict(lamda=lamda, nrmse_pico_vs_pmr_pct=e, pearson_pico_vs_pmr=r,
                   roi_mean_var_pmr=roi_pmr, roi_mean_var_ours=roi_ours,
                   tv_active_set_frac=act, t_pmr=t_pmr, t_pico=t_pico)
        rows.append(row)
        print(f"  NRMSE(PICO vs PMR)={e:.2f}%  Pearson={r:.4f}  TV active-set frac={act:.3f}")
        print(f"  ROI mean var PMR={roi_pmr:.4e} ours={roi_ours:.4e}  t_pmr={t_pmr:.1f}s t_pico={t_pico:.1f}s")

        np.save(os.path.join(maps_dir, f"pmr_var_L{lamda:g}.npy"), pmr_np)
        np.save(os.path.join(maps_dir, f"ours_var_L{lamda:g}.npy"), ours_np)
        np.save(os.path.join(maps_dir, f"recon_mag_L{lamda:g}.npy"), img_recon.abs().detach().cpu().numpy())

    import pandas as pd
    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "lambda_tv_sweep_summary.csv")
    df.to_csv(csv_path, index=False)
    with open(os.path.join(out_dir, "run_manifest.json"), "w") as f:
        json.dump(dict(slice_idx=args.slice_idx, accel=R, lamdas=args.lamdas,
                       n_pmr=args.n_pmr, n_pico=args.n_pico, sigma=sigma,
                       num_iters=args.num_iters, roi=args.roi, roi_size=args.roi_size), f, indent=2)
    print(f"\n[Exp C] summary saved to {csv_path}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.plot(df['lamda'], df['nrmse_pico_vs_pmr_pct'], 'o-', label='NRMSE(PICO vs PMR) [%]')
    ax1b = ax1.twinx()
    ax1b.plot(df['lamda'], df['pearson_pico_vs_pmr'], 's--', color='C1', label='Pearson')
    ax1.set_xscale('log'); ax1.set_xlabel(r'$\lambda_{TV}$'); ax1.set_ylabel('NRMSE(PICO vs PMR) [%]')
    ax1b.set_ylabel('Pearson r')
    ax1.set_title('PICO vs PMR agreement across regularization strength')
    ax1.grid(True, alpha=0.3)

    ax2.plot(df['lamda'], df['tv_active_set_frac'], 'd-', color='C3')
    ax2.set_xscale('log'); ax2.set_xlabel(r'$\lambda_{TV}$'); ax2.set_ylabel('TV active-set fraction')
    ax2.set_title('TV active set grows with $\\lambda$ (where linearization is stressed)')
    ax2.grid(True, alpha=0.3)
    fig.suptitle(f'Exp C: high-$\\lambda_{{TV}}$ CS sweep (knee slice {args.slice_idx}, R={R})', fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.savefig(os.path.join(out_dir, "lambda_tv_sweep_plot.png"), dpi=200)
    plt.savefig(os.path.join(out_dir, "lambda_tv_sweep_plot.svg"))
    print(f"[Exp C] plot saved to {os.path.join(out_dir, 'lambda_tv_sweep_plot.png')}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Exp C: high-lambda_TV CS regularization sweep (PICO vs PMR).")
    p.add_argument('--slice-idx', type=int, default=20)
    p.add_argument('--accel', type=int, default=2)
    p.add_argument('--lamdas', type=float, nargs='+', default=[1e-2, 3e-2, 1e-1, 3e-1, 1e0])
    p.add_argument('--n-pmr', type=int, default=2000)
    p.add_argument('--n-pico', type=int, default=500)
    p.add_argument('--sigma', type=float, default=1e-7)
    p.add_argument('--num-iters', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=100)
    p.add_argument('--roi', type=str, default='central', choices=['central', 'all'])
    p.add_argument('--roi-size', type=int, default=128)
    p.add_argument('--data-file', type=str, default='experiments/fastmri_knee/data/file1000000.h5')
    p.add_argument('--save-dir', type=str, default='experiments/fastmri_knee/results')
    p.add_argument('--run-tag', type=str, default='lambda_tv_sweep')
    p.add_argument('--seed', type=int, default=20260830,
                   help='Seeds the estimator RNG (torch and numpy). The sampling mask '
                        'uses sigpy\'s default seed=0 and is unaffected.')
    p.add_argument('--tv-inner', type=int, default=50, help='Chambolle inner iterations of the TV prox (toolbox default 50)')
    p.add_argument('--tv-tol', type=float, default=1e-4, help='absolute exit tolerance on the dual increment (toolbox default 1e-4; 0 forces --tv-inner iterations)')
    p.add_argument('--fista-ptol', type=float, default=0.5, help='FISTA iterate-change exit rule in percent (toolbox default 0.5); a negative value runs exactly --num-iters iterations')
    p.add_argument('--tv-prox', choices=['magnitude', 'complex'], default='magnitude', help='TV proximal operator: magnitude/phase TV (toolbox) or complex-valued vector TV')
    p.add_argument('--tv-stable-step', action='store_true',
                   help='Use the Chambolle-stable dual step tau=min(0.25, 1/(8*alpha*lamda)) '
                        'instead of the fixed tau=0.25.')
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
