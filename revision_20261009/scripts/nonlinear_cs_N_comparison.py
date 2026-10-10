#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nonlinear CS: N-comparison script
Computes reference PMR at N_ref, then both ours and PMR at varying N values,
and plots them in a grid format similar to run_N_comparison.py
"""
import os, time, argparse, json
from pathlib import Path
import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt
from einops import rearrange
from sigpy.mri import poisson

from mr_recon.fourier import gridded_nufft
from mr_recon.utils import np_to_torch, gen_grd
from mr_recon.algs import FISTA, power_method_operator
from mr_recon.prox import TV
from mr_recon.linops import sense_linop, batching_params
from torch.func import vmap, jvp
from tqdm import tqdm
import pandas as pd

# TV proximal operator inner loop: the toolbox default (50 Chambolle iterations, absolute exit tolerance 1e-4
# on the dual increment) exits after ONE iteration at the fastMRI data scale (see
# scripts/tv_prox_one_step_check.py); pass --tv-tol 0 --tv-inner N to run a converged proximal operator.
TV_INNER = 50
TV_PROX = 'magnitude'  # 'magnitude': mr_recon.prox.TV (magnitude/phase); 'complex': mr_recon.prox_complex_tv.ComplexTV
TV_TOL = 1e-4

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)

def roi_mask(shape_hw, mode="central", roi_size=128):
    H, W = shape_hw
    if mode == "all":
        return np.ones((H, W), dtype=bool)
    cy, cx = H // 2, W // 2
    hs = roi_size // 2
    m = np.zeros((H, W), dtype=bool)
    m[cy - hs: cy + hs, cx - hs: cx + hs] = True
    return m

# FISTA exit rule: 0.5 = the toolbox iterate-change rule (data-dependent iteration count);
# None = run exactly num_iters iterations (a fixed map for replicas and probes alike).
FISTA_PTOL_EXIT = 0.5


def fista_reconstruct(kspace, A, proxg, num_iters, max_eigen):
    AHb = A.adjoint(kspace)
    AHb = AHb / (max_eigen ** 0.5)
    AHA = lambda x: A.normal(x) / max_eigen
    img_recon = FISTA(AHA, AHb, proxg, num_iters=num_iters,
                      ptol_exit=(-1.0 if FISTA_PTOL_EXIT is None else FISTA_PTOL_EXIT), verbose=False)  # negative: never exit early
    return img_recon

def pmr_variance_map(kspace, A, proxg, num_iters, max_eigen, sigma, n_replicas):
    """
    Compute variance map using PMR with incremental computation to avoid OOM.
    Uses Welford's online algorithm adapted for complex-valued images.
    """
    device = kspace.device
    im_size = A.im_size
    
    # Initialize mean and variance accumulators
    mean_img = torch.zeros(im_size, device=device, dtype=torch.complex64)
    
    # First pass: compute mean incrementally
    for n in tqdm(range(n_replicas), desc=f'PMR N={n_replicas} (mean)', leave=False):
        noise = (torch.randn_like(kspace.real) + 1j * torch.randn_like(kspace.imag)) * sigma
        ksp_noise = kspace + noise
        img = fista_reconstruct(ksp_noise, A, proxg, num_iters, max_eigen)
        # Online mean update: mean_new = mean_old + (x - mean_old) / n
        mean_img = mean_img + (img - mean_img) / (n + 1)
    
    # Second pass: compute variance incrementally
    # var = E[|X - E[X]|^2] = E[|X - μ|^2]
    var_accum = torch.zeros(im_size, device=device, dtype=torch.float32)
    for n in tqdm(range(n_replicas), desc=f'PMR N={n_replicas} (var)', leave=False):
        noise = (torch.randn_like(kspace.real) + 1j * torch.randn_like(kspace.imag)) * sigma
        ksp_noise = kspace + noise
        img = fista_reconstruct(ksp_noise, A, proxg, num_iters, max_eigen)
        # Accumulate squared magnitude of centered image
        centered = img - mean_img
        var_accum = var_accum + centered.abs().pow(2)
    
    # Normalize by N and sigma^2
    var_complex = (var_accum / n_replicas) / (sigma ** 2)
    return var_complex

def jvp_hutch_variance(kspace, A, proxg, num_iters, max_eigen, num_samples=100, batch_size=100):
    device = kspace.device
    dtype = kspace.dtype
    def model_func(kspace_input):
        return fista_reconstruct(kspace_input, A, proxg, num_iters, max_eigen)
    def compute_variance_sample(v):
        _, u = jvp(model_func, (kspace,), (v,))
        return (u.real.pow(2) + u.imag.pow(2))
    total = None
    total_batches = (num_samples + batch_size - 1) // batch_size
    for batch_idx in tqdm(range(total_batches), desc=f'Hutchinson N={num_samples}', leave=False):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        B = end_idx - start_idx
        v_shape = (B,) + tuple(kspace.shape)
        real_part = (torch.randint(0, 2, v_shape, device=device).float() * 2 - 1)
        imag_part = (torch.randint(0, 2, v_shape, device=device).float() * 2 - 1)
        v_batch = (real_part + 1j * imag_part).to(dtype)
        variance_samples = vmap(compute_variance_sample)(v_batch)
        batch_sum = variance_samples.sum(dim=0)
        total = batch_sum if total is None else (total + batch_sum)
    variance_est = total / num_samples
    return variance_est

def plot_N_comparison(img_recon, results, g_ref, N_values, N_ref, slice_idx, 
                      noise_scale, out_dir, accel):
    """
    Plot N-comparison grid following run_N_comparison.py style:
    Row 1: Our Method (N values) + Reference PMR (N_ref)
    Row 2: PMR (N values) + Reconstructed Image
    """
    num_comparisons = len(N_values)
    
    # Convert variance to standard deviation for display
    g_ref_std = torch.sqrt(torch.clamp(g_ref, min=0.0))
    results_std = {
        'ours': [torch.sqrt(torch.clamp(var, min=0.0)) for var in results['ours']],
        'pmr': [torch.sqrt(torch.clamp(var, min=0.0)) for var in results['pmr']]
    }
    
    # Determine display window (use reference as base)
    vmin = g_ref_std.min().item()
    vmax = g_ref_std.max().item()
    
    # Create figure: 2 rows × (num_comparisons + 1) columns
    fig, axes = plt.subplots(2, num_comparisons + 1, figsize=(5 * (num_comparisons + 1), 11))
    fig.patch.set_facecolor('black')
    
    # --- Row 1: Our Method + N values + Reference PMR ---
    # Columns 0 to num_comparisons-1: Our Method results
    for i, n in enumerate(N_values):
        ax = axes[0, i]
        var_std_np = results_std['ours'][i].cpu().numpy()
        im = ax.imshow(var_std_np, cmap='jet', vmin=vmin, vmax=vmax)
        ax.set_title(f"Our Method (N={n})", color='white', fontsize=12, fontweight='bold')
        ax.axis('off')
    
    # Last column: Reference PMR (N_ref)
    g_ref_std_np = g_ref_std.cpu().numpy()
    im = axes[0, num_comparisons].imshow(g_ref_std_np, cmap='jet', vmin=vmin, vmax=vmax)
    axes[0, num_comparisons].set_title(f"Reference PMR\n(N={N_ref})", color='white', fontsize=12, fontweight='bold')
    axes[0, num_comparisons].axis('off')
    
    # --- Row 2: PMR Method + N values + Reconstructed Image ---
    # Columns 0 to num_comparisons-1: PMR Method results
    for i, n in enumerate(N_values):
        ax = axes[1, i]
        var_std_np = results_std['pmr'][i].cpu().numpy()
        im = ax.imshow(var_std_np, cmap='jet', vmin=vmin, vmax=vmax)
        ax.set_title(f"PMR (N={n})", color='white', fontsize=12, fontweight='bold')
        ax.axis('off')
    
    # Last column: Reconstructed Image
    img_recon_np = img_recon.abs().detach().cpu().numpy()
    axes[1, num_comparisons].imshow(img_recon_np, cmap='gray')
    axes[1, num_comparisons].set_title(f"Reconstructed Image\n(R={accel})", color='white', fontsize=12, fontweight='bold')
    axes[1, num_comparisons].axis('off')
    
    # Adjust layout to make space for the colorbar and title
    fig.subplots_adjust(right=0.92, top=0.90)
    
    # Add a single colorbar for all variance maps on the right side
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Standard Deviation')
    
    # Set title
    N_str = "-".join(str(n) for n in N_values)
    fig.suptitle(f'Variance Map N-Comparison (Slice {slice_idx}, σ×{noise_scale:g}, N_ref={N_ref})', 
                 fontsize=16, color='white', fontweight='bold')
    
    # Save outputs
    output_base = out_dir / f'N_comparison_slice{slice_idx:04d}_Nref{N_ref}_N{N_str}'
    plt.savefig(f'{output_base}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_base}.pdf', bbox_inches='tight')
    plt.savefig(f'{output_base}.svg', bbox_inches='tight')
    plt.close()
    print(f"N-comparison plot saved to: {output_base}.{{png,pdf,svg}}")

def main():
    ap = argparse.ArgumentParser(description="Nonlinear CS: N-comparison script")
    ap.add_argument('--slice-idx', type=int, default=20)
    ap.add_argument('--N-ref', type=int, default=100, help='Reference N for PMR')
    ap.add_argument('--N-values', nargs='+', type=int, default=[10, 20, 50], help='N values for comparison')
    ap.add_argument('--noise-scale', type=float, default=1.0, help='Noise scale multiplier')
    ap.add_argument('--sigma0', type=float, default=1e-7)
    ap.add_argument('--accel', type=int, default=2)
    ap.add_argument('--lamda', type=float, default=1e-2)
    ap.add_argument('--num-iters', type=int, default=100)
    ap.add_argument('--batch-size', type=int, default=100)
    ap.add_argument('--roi', type=str, default='central', choices=['central','all'])
    ap.add_argument('--roi-size', type=int, default=128)
    ap.add_argument('--data-file', type=str, default='experiments/fastmri_knee/data/file1000000.h5')
    ap.add_argument('--save-dir', type=str, default='experiments/fastmri_knee/results')
    ap.add_argument('--run-tag', type=str, default='cs_N_comparison')
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--reuse-pmr-reference', action='store_true',
                    help='load an existing ref_pmr_var_N{N_ref}.npy instead of recomputing it')
    ap.add_argument('--ref-only', action='store_true',
                    help='compute and save the reference map(s), then stop')
    ap.add_argument('--pico-reference', action='store_true',
                    help='also compute the PICO gold map at N_ref, so each estimator is scored '
                         'against its own high-sample limit as in the original submission')
    ap.add_argument('--tv-inner', type=int, default=50, help='Chambolle inner iterations of the TV prox (toolbox default 50)')
    ap.add_argument('--tv-tol', type=float, default=1e-4, help='absolute exit tolerance on the dual increment (toolbox default 1e-4; 0 forces --tv-inner iterations)')
    ap.add_argument('--fista-ptol', type=float, default=0.5, help='FISTA iterate-change exit rule in percent (toolbox default 0.5); a negative value runs exactly --num-iters iterations')
    ap.add_argument('--tv-prox', choices=['magnitude', 'complex'], default='magnitude', help='TV proximal operator: magnitude/phase TV (toolbox) or complex-valued vector TV')
    args = ap.parse_args()
    _ptol = None if args.fista_ptol < 0 else float(args.fista_ptol)
    globals()['TV_PROX'] = args.tv_prox
    print(f'[tv-prox] operator={args.tv_prox}', flush=True)
    import nonlinear_cs_noise_sweep as _ncs, nonlinear_cs_N_comparison as _ncn
    _ncs.FISTA_PTOL_EXIT = _ptol; _ncn.FISTA_PTOL_EXIT = _ptol; globals()['FISTA_PTOL_EXIT'] = _ptol
    print(f'[fista] ptol_exit={_ptol} num_iters={args.num_iters}', flush=True)
    global TV_INNER, TV_TOL
    TV_INNER, TV_TOL = args.tv_inner, args.tv_tol
    print(f'[tv-prox] inner={TV_INNER} tol={TV_TOL:g}', flush=True)
    
    set_seed(args.seed)
    torch_dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    with h5py.File(args.data_file, 'r') as f:
        img = torch.tensor(f['reconstruction_rss'], device=torch_dev)[args.slice_idx]
        mps = torch.tensor(f['jsense-12-cf=4']['maps'], device=torch_dev)[args.slice_idx, :, :, :, 0]
        ksp_full = torch.tensor(f['kspace'][:], device=torch_dev)[args.slice_idx]
    
    C = mps.shape[-1]
    mps = rearrange(mps, 'h w c -> c h w')
    if ksp_full.ndim == 3:
        if ksp_full.shape[-1] == C:
            ksp_full = rearrange(ksp_full, 'h w c -> c h w')
        elif ksp_full.shape[0] == C:
            pass
        else:
            raise ValueError(f"Unexpected k-space shape: {ksp_full.shape}")
    else:
        raise ValueError(f"Unexpected k-space dim: {ksp_full.ndim}")
    
    # Setup output directory
    out_dir = Path(args.save_dir) / args.run_tag / f"slice{args.slice_idx:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup reconstruction
    R = args.accel
    lamda = args.lamda
    num_iters = args.num_iters
    sigma = args.sigma0 * args.noise_scale
    roi = roi_mask(img.shape[-2:], mode=args.roi, roi_size=args.roi_size)
    mask = poisson(img.shape[-2:], accel=R, calib=(24, 24), tol=0.3, crop_corner=False)
    mask = np_to_torch(mask.real).type(torch.float32).to(torch_dev)
    inds = torch.argwhere(mask > 0)
    trj = gen_grd(img.shape[-2:], img.shape[-2:]).to(torch_dev)[inds[:, 0], inds[:, 1]]
    C = mps.shape[0]
    mask_expanded = mask.unsqueeze(0).expand(C, -1, -1)
    ksp_undersampled = ksp_full * mask_expanded
    ksp = ksp_undersampled[:, inds[:, 0], inds[:, 1]]
    nufft = gridded_nufft(img.shape[-2:])
    bparams = batching_params(C)
    dcf = torch.ones(trj.shape[0], device=torch_dev)
    A = sense_linop(img.shape[-2:], trj, mps, dcf, nufft=nufft, bparams=bparams)
    if TV_PROX == 'complex':
        from mr_recon.prox_complex_tv import ComplexTV
        proxg = ComplexTV(tuple(img.shape[-2:]), lamda, max_iter=TV_INNER, tol=TV_TOL)
    else:
        proxg = TV(im_size=img.shape[-2:], lamda=lamda, norm='l2', max_iter=TV_INNER, tol=TV_TOL)
    x0 = torch.randn_like(img)
    _, max_eigen = power_method_operator(A.normal, x0)
    max_eigen *= 1.01
    if isinstance(max_eigen, torch.Tensor):
        print(f"Max Eigenvalue = {max_eigen.item()}")
    else:
        print(f"Max Eigenvalue = {max_eigen}")
    
    # Reconstruct image
    img_recon = fista_reconstruct(ksp, A, proxg, num_iters, max_eigen)
    np.save(out_dir / "recon_mag.npy", img_recon.abs().detach().cpu().numpy())
    
    # Calculate reference PMR at N_ref
    pmr_ref_path = out_dir / f"ref_pmr_var_N{args.N_ref}.npy"
    if getattr(args, "reuse_pmr_reference", False) and pmr_ref_path.exists():
        print(f"[ref] reusing {pmr_ref_path}", flush=True)
        g_ref = torch.as_tensor(np.load(pmr_ref_path), device=torch_dev)
        time_ref = float("nan")
    else:
        print(f"\n=== Calculating reference PMR at N={args.N_ref} ===")
        t0 = time.time()
        g_ref = pmr_variance_map(ksp, A, proxg, num_iters, max_eigen, sigma=sigma, n_replicas=args.N_ref)
        time_ref = time.time() - t0
        print(f"Reference PMR (N={args.N_ref}) completed in {time_ref:.2f}s")
        np.save(pmr_ref_path, g_ref.detach().cpu().numpy())

    # Method-specific gold map for PICO at the same budget, as in the original
    # submission: the two estimators converge to different limits here, so each
    # is scored against a high-sample map of its own kind.
    if getattr(args, "pico_reference", False):
        print(f"\n=== Calculating reference PICO (JVP) at N={args.N_ref} ===")
        t0 = time.time()
        pico_ref = jvp_hutch_variance(
            ksp, A, proxg, num_iters, max_eigen,
            num_samples=args.N_ref, batch_size=args.batch_size,
        )
        pico_ref_time = time.time() - t0
        print(f"Reference PICO (N={args.N_ref}) completed in {pico_ref_time:.2f}s")
        np.save(out_dir / f"ref_ours_var_N{args.N_ref}.npy", pico_ref.detach().cpu().numpy())
    if getattr(args, "ref_only", False):
        print("[ref] --ref-only: reference(s) saved; skipping the N grid", flush=True)
        return
    
    # Calculate both ours and PMR at varying N values
    results = {
        'N': [],
        'ours': [],
        'pmr': [],
        'time_ours': [],
        'time_pmr': []
    }
    
    print(f"\n=== Calculating for N values: {args.N_values} ===")
    for N in args.N_values:
        print(f"\n--- N={N} ---")
        results['N'].append(N)
        
        # Our method (Hutchinson)
        print(f"  Computing Our Method (Hutchinson)...")
        t1 = time.time()
        var_ours = jvp_hutch_variance(ksp, A, proxg, num_iters, max_eigen, 
                                      num_samples=N, batch_size=args.batch_size)
        t_ours = time.time() - t1
        results['ours'].append(var_ours)
        results['time_ours'].append(t_ours)
        print(f"  Our Method completed in {t_ours:.2f}s")
        np.save(out_dir / f"ours_var_N{N}.npy", var_ours.detach().cpu().numpy())
        
        # PMR method
        print(f"  Computing PMR...")
        t1 = time.time()
        var_pmr = pmr_variance_map(ksp, A, proxg, num_iters, max_eigen, 
                                   sigma=sigma, n_replicas=N)
        t_pmr = time.time() - t1
        results['pmr'].append(var_pmr)
        results['time_pmr'].append(t_pmr)
        print(f"  PMR completed in {t_pmr:.2f}s")
        np.save(out_dir / f"pmr_var_N{N}.npy", var_pmr.detach().cpu().numpy())
    
    # Save timing data
    df = pd.DataFrame({
        'N': results['N'],
        'time_ours_s': results['time_ours'],
        'time_pmr_s': results['time_pmr']
    })
    df.to_csv(out_dir / "timings.csv", index=False)
    
    # Save summary
    summary = {
        'slice_idx': args.slice_idx,
        'N_ref': args.N_ref,
        'N_values': args.N_values,
        'noise_scale': args.noise_scale,
        'sigma': float(sigma),
        'accel': args.accel,
        'lamda': args.lamda,
        'num_iters': args.num_iters,
        'time_ref_s': float(time_ref),
        'timings': df.to_dict('records')
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    
    # Generate plot
    print(f"\n=== Generating N-comparison plot ===")
    plot_N_comparison(
        img_recon=img_recon,
        results=results,
        g_ref=g_ref,
        N_values=args.N_values,
        N_ref=args.N_ref,
        slice_idx=args.slice_idx,
        noise_scale=args.noise_scale,
        out_dir=out_dir,
        accel=args.accel
    )
    
    print(f"\nDone! Results saved to: {out_dir}")

if __name__ == '__main__':
    main()

