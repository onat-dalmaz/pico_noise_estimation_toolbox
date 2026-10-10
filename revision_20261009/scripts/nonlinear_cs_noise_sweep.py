import json
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

def pearsonr(a, b):
    a = np.asarray(a).ravel().astype(np.float64)
    b = np.asarray(b).ravel().astype(np.float64)
    am, bm = a.mean(), b.mean()
    num = np.sum((a - am) * (b - bm))
    den = np.sqrt(np.sum((a - am) ** 2) * np.sum((b - bm) ** 2))
    return 0.0 if den == 0 else float(num / den)

def nrmse_percent(a, b):
    a = np.asarray(a).astype(np.float64)
    b = np.asarray(b).astype(np.float64)
    denom = np.linalg.norm(b.ravel())
    return 0.0 if denom == 0 else float(np.linalg.norm((a - b).ravel()) / denom * 100.0)

def rel_change_percent(curr, prev):
    if prev is None or prev == 0:
        return np.nan
    return abs(curr - prev) / abs(prev) * 100.0

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

def plot_final_comparison_per_noise(img_display, zero_filled_display, pmr_var_np, ours_var_np, 
                                     t_pmr, t_ours, targetN, slice_idx, noise_scale, 
                                     out_slice_dir, vmin_std_override=None, vmax_std_override=None):
    """
    Plot final comparison following PLOTTING_CONVENTIONS.md:
    3 rows × 2 columns: Zero-filled, Reconstructed, Our Method, PMR Reference, Difference maps
    """
    def to_numpy_magnitude(x):
        arr = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
        return np.abs(arr) if np.iscomplexobj(arr) else arr

    img_recon_np = to_numpy_magnitude(img_display)
    zero_filled_np = to_numpy_magnitude(zero_filled_display)
    
    # Ensure non-negative variance (should already be, but clamp to avoid numerical issues)
    pmr_var_np = np.maximum(pmr_var_np, 0.0)
    ours_var_np = np.maximum(ours_var_np, 0.0)
    
    # Convert variance to standard deviation
    pmr_std_np = np.sqrt(pmr_var_np)
    ours_std_np = np.sqrt(ours_var_np)
    
    # Create figure: 3 rows × 2 columns, size (12, 18)
    fig, axes = plt.subplots(3, 2, figsize=(12, 18))
    
    # Set display window: same scale for all std and diff maps; vmin=1
    vmin_std = vmin_std_override if vmin_std_override is not None else max(1.0, float(pmr_std_np.min()))
    vmax_std = vmax_std_override if vmax_std_override is not None else pmr_std_np.max()
    
    # Row 1: Reference Images
    vmax_img = max(zero_filled_np.max(), img_recon_np.max())
    axes[0, 0].imshow(zero_filled_np, cmap='gray', vmin=0, vmax=vmax_img)
    axes[0, 0].set_title('Zero-Filled', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img_recon_np, cmap='gray', vmin=0, vmax=vmax_img)
    axes[0, 1].set_title('Reconstructed Image', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Row 2: Noise Standard Deviation Maps (same display window)
    im_ours = axes[1, 0].imshow(ours_std_np, cmap='jet', vmin=vmin_std, vmax=vmax_std)
    axes[1, 0].set_title(f'Our Method (N={targetN})\nTime: {t_ours:.2f}s', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    im_pmr = axes[1, 1].imshow(pmr_std_np, cmap='jet', vmin=vmin_std, vmax=vmax_std)
    axes[1, 1].set_title(f'PMR Reference (N={targetN})\nTime: {t_pmr:.2f}s', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Row 3: Difference Maps (same color scale as std maps; vmin=1)
    diff_map = np.abs(ours_std_np - pmr_std_np)
    diff_map_10x = diff_map * 10.0
    axes[2, 0].imshow(diff_map, cmap='jet', vmin=vmin_std, vmax=vmax_std)
    axes[2, 0].set_title('Difference Map', fontsize=14, fontweight='bold')
    axes[2, 0].axis('off')

    axes[2, 1].imshow(diff_map_10x, cmap='jet', vmin=vmin_std, vmax=vmax_std)
    axes[2, 1].set_title('Difference Map (x10 Amplified)', fontsize=14, fontweight='bold')
    axes[2, 1].axis('off')
    
    # Add colorbar on the right
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.35, 0.02, 0.3])
    fig.colorbar(im_pmr, cax=cbar_ax, label='Standard Deviation')
    
    plt.suptitle(f'Non-linear Reconstruction Noise Standard Deviation Comparison (Slice {slice_idx}, σ×{noise_scale:g})', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    # Save figure
    output_path = out_slice_dir / f'comparison_slice{slice_idx:04d}_N{targetN}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.svg'), bbox_inches='tight')
    plt.close()

def run_one_noise(args, img, mps, ksp_full, torch_dev, noise_scale, out_slice_dir):
    out_slice_dir.mkdir(parents=True, exist_ok=True)
    R = args.accel
    base_lamda = args.lamda  # Fixed at 0.01 for all σ: same forward model → σ-independent variance maps.
    num_iters = args.num_iters
    sigma = args.sigma0 * noise_scale
    gridN = args.grid_N
    targetN = args.target_N
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
        proxg = ComplexTV(tuple(img.shape[-2:]), base_lamda, max_iter=TV_INNER, tol=TV_TOL)
    else:
        proxg = TV(im_size=img.shape[-2:], lamda=base_lamda, norm='l2', max_iter=TV_INNER, tol=TV_TOL)
    x0 = torch.randn_like(img)
    _, max_eigen = power_method_operator(A.normal, x0)
    max_eigen *= 1.01
    img_recon = fista_reconstruct(ksp, A, proxg, num_iters, max_eigen)
    zero_filled = A.adjoint(ksp)
    np.save(out_slice_dir / "recon_mag.npy", img_recon.abs().detach().cpu().numpy())
    np.save(out_slice_dir / "zero_filled_mag.npy", zero_filled.abs().detach().cpu().numpy())

    # Representative noisy sample for visualization
    sample_seed = args.seed + int(round(noise_scale * 1000))
    noise_gen = torch.Generator(device='cpu')
    noise_gen.manual_seed(sample_seed)
    noise_real = torch.randn(ksp.real.shape, generator=noise_gen).to(ksp.real.device)
    noise_imag = torch.randn(ksp.imag.shape, generator=noise_gen).to(ksp.imag.device)
    noise_sample = (noise_real + 1j * noise_imag).to(ksp.dtype) * sigma
    ksp_noisy_sample = ksp + noise_sample
    img_recon_noisy = fista_reconstruct(ksp_noisy_sample, A, proxg, num_iters, max_eigen)
    zero_filled_noisy = A.adjoint(ksp_noisy_sample)
    np.save(out_slice_dir / "kspace_noisy_sample.npy", ksp_noisy_sample.detach().cpu().numpy())
    np.save(out_slice_dir / "recon_noisy_mag.npy", img_recon_noisy.abs().detach().cpu().numpy())
    np.save(out_slice_dir / "zero_filled_noisy_mag.npy", zero_filled_noisy.abs().detach().cpu().numpy())
    N_ref = int(getattr(args, "N_ref", 10000))
    ref_path = out_slice_dir / f"ref_pmr_var_N{N_ref}.npy"
    ref_meta = out_slice_dir / "ref_meta.json"
    if getattr(args, "reuse_ref", False) and ref_path.exists():
        ref_np = np.load(ref_path)
        ref_time = float(json.load(open(ref_meta))["reference_time_s"]) if ref_meta.exists() else float("nan")
        print(f"[ref] reusing {ref_path} (N_ref={N_ref}); reference_time_s={ref_time}", flush=True)
    else:
        t0 = time.time()
        ref_var = pmr_variance_map(ksp, A, proxg, num_iters, max_eigen, sigma=sigma, n_replicas=N_ref)
        ref_time = time.time() - t0
        ref_np = ref_var.detach().cpu().numpy()
        np.save(ref_path, ref_np)
        json.dump({"N_ref": N_ref, "reference_time_s": ref_time,
                   "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"},
                  open(ref_meta, "w"), indent=2)
        print(f"[ref] computed {ref_path} in {ref_time:.1f} s", flush=True)
    # Method-specific gold maps, as in the original submission: PICO and PMR
    # converge to different limits in the nonlinear regime, so each estimator is
    # scored against a high-sample map of its own kind at the same budget.
    pico_ref_path = out_slice_dir / f"ref_ours_var_N{N_ref}.npy"
    pico_ref_np = None
    if getattr(args, "pico_reference", False):
        if getattr(args, "reuse_ref", False) and pico_ref_path.exists():
            pico_ref_np = np.load(pico_ref_path)
            print(f"[ref] reusing {pico_ref_path}", flush=True)
        else:
            t0 = time.time()
            pico_ref = jvp_hutch_variance(
                ksp, A, proxg, num_iters, max_eigen,
                num_samples=N_ref,
                batch_size=int(getattr(args, "batch_size", 100)),
            )
            pico_ref_time = time.time() - t0
            pico_ref_np = pico_ref.detach().cpu().numpy()
            np.save(pico_ref_path, pico_ref_np)
            meta = json.load(open(ref_meta)) if ref_meta.exists() else {}
            meta["pico_reference_N"] = N_ref
            meta["pico_reference_time_s"] = pico_ref_time
            json.dump(meta, open(ref_meta, "w"), indent=2)
            print(f"[ref] computed {pico_ref_path} in {pico_ref_time:.1f} s", flush=True)
    if getattr(args, "ref_only", False):
        print("[ref] --ref-only: reference(s) saved; skipping the PMR/PICO grid", flush=True)
        return
    def roi_mean(map_np): return float(np.mean(map_np[roi]))
    def metrics_vs_ref(map_np):
        e = nrmse_percent(map_np, ref_np)
        r = pearsonr(map_np, ref_np)
        return e, r, roi_mean(map_np)
    import pandas as pd
    pmr_rows = []
    last_roi_mean = None
    good_enough_N = None
    pmr_var_target_np = None
    t_pmr_target = None
    for N in gridN:
        t1 = time.time()
        varN = pmr_variance_map(ksp, A, proxg, num_iters, max_eigen, sigma=sigma, n_replicas=N)
        tN = time.time() - t1
        var_np = varN.detach().cpu().numpy()
        e, r, m = metrics_vs_ref(var_np)
        d_roi = rel_change_percent(m, last_roi_mean)
        pmr_rows.append([N, tN, e, r, m, d_roi])
        last_roi_mean = m
        if N == targetN:
            np.save(out_slice_dir / f"pmr_var_N{targetN}.npy", var_np)
            pmr_var_target_np = var_np.copy()
            t_pmr_target = tN
        if good_enough_N is None and len(pmr_rows) >= 2:
            prev_e = pmr_rows[-2][2]
            imp = prev_e - e
            if (not np.isnan(d_roi)) and (d_roi < 0.2) and (imp < 0.5):
                good_enough_N = N
    pmr_df = pd.DataFrame(pmr_rows, columns=["N", "time_s", "NRMSE_%", "Pearson_r", "ROI_mean_var", "ΔROIvar_%"])
    pmr_df.to_csv(out_slice_dir / "pmr_metrics.csv", index=False)
    ours_rows = []
    last_e = None
    conv_N_ours = None
    ours_var_target_np = None
    t_ours_target = None
    # When grid has only one point, the JVP Hutchinson estimate can be unstable (huge outliers/wrong scale)
    # because the RNG state differs from the multi-grid case. Use more samples for stability.
    for N in gridN:
        t1 = time.time()
        n_jvp = max(N, int(getattr(args, 'jvp_min_samples', 5000))) if len(gridN) == 1 else N  # single-grid: at least --jvp-min-samples (0 = exactly N)
        varN = jvp_hutch_variance(ksp, A, proxg, num_iters, max_eigen, num_samples=n_jvp, batch_size=args.batch_size)
        tN = time.time() - t1
        var_np = varN.detach().cpu().numpy()
        e, r, m = metrics_vs_ref(var_np)
        d_e = (e - last_e) if last_e is not None else np.nan
        ours_rows.append([N, tN, e, r, m, d_e])
        if N == targetN:
            np.save(out_slice_dir / f"ours_var_N{targetN}.npy", var_np)
            ours_var_target_np = var_np.copy()
            t_ours_target = tN
        if conv_N_ours is None and last_e is not None:
            if (N - ours_rows[-2][0]) >= 100 and (not np.isnan(d_e)) and d_e >= -0.1:
                conv_N_ours = N
        last_e = e
    ours_df = pd.DataFrame(ours_rows, columns=["N", "time_s", "NRMSE_%", "Pearson_r", "ROI_mean_var", "ΔNRMSE_abs%"])
    ours_df.to_csv(out_slice_dir / "ours_metrics.csv", index=False)
    
    # Generate final comparison plot for this noise level (following PLOTTING_CONVENTIONS.md)
    if pmr_var_target_np is not None and ours_var_target_np is not None and t_pmr_target is not None and t_ours_target is not None:
        try:
            plot_final_comparison_per_noise(
                img_display=img_recon_noisy,
                zero_filled_display=zero_filled_noisy,
                pmr_var_np=pmr_var_target_np,
                ours_var_np=ours_var_target_np,
                t_pmr=t_pmr_target,
                t_ours=t_ours_target,
                targetN=targetN,
                slice_idx=args.slice_idx,
                noise_scale=noise_scale,
                    out_slice_dir=out_slice_dir,
                    vmin_std_override=args.std_vmin, vmax_std_override=args.std_vmax
            )
        except Exception as e:
            print(f"Warning: Failed to generate comparison plot: {e}")
    
    def nearest_row(df, N):
        return df.iloc[(df["N"] - N).abs().argsort()].iloc[0]
    # Use stored t_pmr_target if available, otherwise use nearest_row
    if t_pmr_target is None:
        t_pmr_target = float(nearest_row(pmr_df, targetN)["time_s"])
    e_pmr_target = float(nearest_row(pmr_df, targetN)["NRMSE_%"])
    ours_eq = ours_df.iloc[(ours_df["time_s"] - t_pmr_target).abs().argsort()].iloc[0].to_dict()
    ours_acc = ours_df.iloc[(ours_df["NRMSE_%"] - e_pmr_target).abs().argsort()].iloc[0].to_dict()
    summary = dict(
        noise_scale=float(noise_scale),
        sigma=float(sigma),
        reference_time_s=float(ref_time),
        pmr_good_enough_N=good_enough_N,
        conv_N_ours=conv_N_ours,
        equal_runtime_probe=ours_eq,
        accuracy_matched_probe=ours_acc,
        target_N=targetN
    )
    (out_slice_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    try:
        plt.figure(figsize=(6,4))
        plt.xscale("log")
        plt.plot(pmr_df["N"], pmr_df["NRMSE_%"], marker="o", label="PMR")
        plt.plot(ours_df["N"], ours_df["NRMSE_%"], marker="o", label="Ours (JVP)")
        plt.axvline(targetN, linestyle="--", linewidth=1)
        plt.xlabel("N"); plt.ylabel("NRMSE (%) vs PMR@N=10000")
        plt.legend(); plt.tight_layout()
        plt.savefig(out_slice_dir / "fig_nrmse_vs_N.png", dpi=200)
        plt.savefig(out_slice_dir / "fig_nrmse_vs_N.pdf")
        plt.close()
        plt.figure(figsize=(6,4))
        plt.xscale("log")
        plt.plot(pmr_df["N"], pmr_df["ΔROIvar_%"], marker="o", label="PMR ΔROI-var")
        plt.axhline(0.2, linestyle="--", linewidth=1)
        plt.xlabel("N (replicas)"); plt.ylabel("Δ ROI-mean variance (%)")
        plt.legend(); plt.tight_layout()
        plt.savefig(out_slice_dir / "fig_deltaROI_vs_N.png", dpi=200)
        plt.savefig(out_slice_dir / "fig_deltaROI_vs_N.pdf")
        plt.close()
    except Exception:
        pass

def plot_zf_concatenated_svg(save_dir, run_tag, noise_scales, slice_idx, r4_suffix=""):
    """
    Save zero-filled images concatenated horizontally as a separate SVG file.
    Used for R=4 acceleration runs to show ZF artifacts separately.
    """
    save_root = Path(save_dir) / run_tag
    n_noise = len(noise_scales)
    zero_filled_list = []
    for s in noise_scales:
        sdir = save_root / f"noise_{s:g}" / f"slice{slice_idx:04d}"
        zero_pref = sdir / "zero_filled_noisy_mag.npy"
        zero_fallback = sdir / "zero_filled_mag.npy"
        if zero_pref.exists():
            zf = np.load(zero_pref)
        elif zero_fallback.exists():
            zf = np.load(zero_fallback)
        else:
            raise FileNotFoundError(f"Zero-filled image not found for noise={s} in {sdir}")
        zero_filled_list.append(zf)
    vmax = max(z.max() for z in zero_filled_list)
    fig, axes = plt.subplots(1, n_noise, figsize=(4 * n_noise, 4), squeeze=False)
    for idx, (s, zf) in enumerate(zip(noise_scales, zero_filled_list)):
        axes[0, idx].imshow(zf, cmap='gray', vmin=0, vmax=vmax)
        axes[0, idx].set_title(f'Zero-Filled\nσ×{s:g}', fontsize=12, fontweight='bold')
        axes[0, idx].axis('off')
    plt.suptitle(f'Zero-Filled Images R=4 (Slice {slice_idx})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_path = save_root / f'zero_filled_concatenated{r4_suffix}.svg'
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"Zero-filled concatenated SVG saved to: {out_path}")

def plot_concatenated_comparisons(save_dir, run_tag, noise_scales, slice_idx, targetN,
                                  vmin_std_override=None, vmax_std_override=None, r4_suffix="", output_suffix=""):
    """
    Create concatenated comparison plots for all noise levels following PLOTTING_CONVENTIONS.md.
    Layout: 3 rows × (2 columns × num_noise_levels)
    Each noise level occupies 2 columns: left and right comparison views.
    output_suffix: appended to filename before extension (e.g. "_vmin1" -> comparison_concatenated_..._vmin1.png).
    """
    save_root = Path(save_dir) / run_tag
    
    # Load data for all noise levels
    data_per_noise = []
    for s in noise_scales:
        sdir = save_root / f"noise_{s:g}" / f"slice{slice_idx:04d}"
        
        # Load variance maps
        pmr_p = sdir / f"pmr_var_N{targetN}.npy"
        ours_p = sdir / f"ours_var_N{targetN}.npy"
        recon_pref = sdir / "recon_noisy_mag.npy"
        recon_fallback = sdir / "recon_mag.npy"
        zero_pref = sdir / "zero_filled_noisy_mag.npy"
        zero_fallback = sdir / "zero_filled_mag.npy"
        
        if not (pmr_p.exists() and ours_p.exists()):
            raise FileNotFoundError(f"Missing variance files for noise={s} in {sdir}")
        
        pmr_var = np.load(pmr_p)
        ours_var = np.load(ours_p)
        recon_path = recon_pref if recon_pref.exists() else recon_fallback
        if not recon_path.exists():
            raise FileNotFoundError(f"Missing reconstruction magnitude for noise={s} in {sdir}")
        recon = np.load(recon_path)
        if recon_path == recon_fallback:
            print(f"Warning: recon_noisy_mag.npy not found for noise={s}, using clean reconstruction")
        if zero_pref.exists():
            zero_filled = np.load(zero_pref)
        elif zero_fallback.exists():
            zero_filled = np.load(zero_fallback)
            print(f"Warning: zero_filled_noisy_mag.npy not found for noise={s}, using clean zero-filled image")
        else:
            # Fallback: use reconstruction if zero-filled not available at all
            print(f"Warning: zero-filled image not found for noise={s}, using reconstruction as fallback")
            zero_filled = recon
        
        # Convert variance to std dev
        pmr_var = np.maximum(pmr_var, 0.0)
        ours_var = np.maximum(ours_var, 0.0)
        pmr_std = np.sqrt(pmr_var)
        ours_std = np.sqrt(ours_var)
        
        # Compute difference maps
        diff_map = np.abs(ours_std - pmr_std)
        diff_map_10x = diff_map * 10.0
        
        # Load timing info from summary if available
        summary_p = sdir / "summary.json"
        t_pmr = None
        t_ours = None
        if summary_p.exists():
            summary = json.loads(summary_p.read_text())
            # Try to get times from summary or metrics
            pmr_metrics_p = sdir / "pmr_metrics.csv"
            ours_metrics_p = sdir / "ours_metrics.csv"
            if pmr_metrics_p.exists() and ours_metrics_p.exists():
                import pandas as pd
                pmr_df = pd.read_csv(pmr_metrics_p)
                ours_df = pd.read_csv(ours_metrics_p)
                pmr_row = pmr_df[pmr_df["N"] == targetN]
                ours_row = ours_df[ours_df["N"] == targetN]
                if not pmr_row.empty:
                    t_pmr = float(pmr_row["time_s"].iloc[0])
                if not ours_row.empty:
                    t_ours = float(ours_row["time_s"].iloc[0])
        
        data_per_noise.append({
            'noise_scale': s,
            'zero_filled': zero_filled,
            'recon': recon,
            'ours_std': ours_std,
            'pmr_std': pmr_std,
            'diff_map': diff_map,
            'diff_map_10x': diff_map_10x,
            't_pmr': t_pmr,
            't_ours': t_ours
        })
    
    # Determine common display windows; same scale for all std and diff maps, vmin=1
    vmin_std = vmin_std_override if vmin_std_override is not None else \
        max(1.0, min(d['pmr_std'].min() for d in data_per_noise))
    vmax_std = vmax_std_override if vmax_std_override is not None else \
        max(d['pmr_std'].max() for d in data_per_noise)
    vmax_img = max(max(d['zero_filled'].max(), d['recon'].max()) for d in data_per_noise)
    
    # Create figure: 3 rows × (2 columns × num_noise_levels)
    n_noise = len(noise_scales)
    fig, axes = plt.subplots(3, 2 * n_noise, figsize=(6 * n_noise, 18), squeeze=False)
    
    # Plot each noise level
    for idx, data in enumerate(data_per_noise):
        col_start = idx * 2
        
        # Row 1: Zero-filled and Reconstructed Image
        axes[0, col_start].imshow(data['zero_filled'], cmap='gray', vmin=0, vmax=vmax_img)
        axes[0, col_start].set_title(f'Zero-Filled\nσ×{data["noise_scale"]:g}', fontsize=12, fontweight='bold')
        axes[0, col_start].axis('off')
        
        axes[0, col_start + 1].imshow(data['recon'], cmap='gray', vmin=0, vmax=vmax_img)
        axes[0, col_start + 1].set_title(f'Reconstructed Image\nσ×{data["noise_scale"]:g}', fontsize=12, fontweight='bold')
        axes[0, col_start + 1].axis('off')
        
        # Row 2: Noise Standard Deviation Maps
        im_ours = axes[1, col_start].imshow(data['ours_std'], cmap='jet', vmin=vmin_std, vmax=vmax_std)
        time_str = f'\nTime: {data["t_ours"]:.2f}s' if data['t_ours'] is not None else ''
        axes[1, col_start].set_title(f'Our Method (N={targetN}){time_str}\nσ×{data["noise_scale"]:g}', 
                                     fontsize=12, fontweight='bold')
        axes[1, col_start].axis('off')
        
        im_pmr = axes[1, col_start + 1].imshow(data['pmr_std'], cmap='jet', vmin=vmin_std, vmax=vmax_std)
        time_str = f'\nTime: {data["t_pmr"]:.2f}s' if data['t_pmr'] is not None else ''
        axes[1, col_start + 1].set_title(f'PMR Reference (N={targetN}){time_str}\nσ×{data["noise_scale"]:g}', 
                                         fontsize=12, fontweight='bold')
        axes[1, col_start + 1].axis('off')
        
        # Row 3: Difference Maps (same color scale as std maps; vmin=1)
        axes[2, col_start].imshow(data['diff_map'], cmap='jet', vmin=vmin_std, vmax=vmax_std)
        axes[2, col_start].set_title(f'Difference Map\nσ×{data["noise_scale"]:g}', fontsize=12, fontweight='bold')
        axes[2, col_start].axis('off')

        axes[2, col_start + 1].imshow(data['diff_map_10x'], cmap='jet', vmin=vmin_std, vmax=vmax_std)
        axes[2, col_start + 1].set_title(f'Difference Map (x10)\nσ×{data["noise_scale"]:g}', fontsize=12, fontweight='bold')
        axes[2, col_start + 1].axis('off')
        
        # Add y-labels only for first noise level
        if idx == 0:
            axes[0, col_start].set_ylabel('Reference Images', fontsize=14, fontweight='bold')
            axes[1, col_start].set_ylabel('Std Dev Maps', fontsize=14, fontweight='bold')
            axes[2, col_start].set_ylabel('Difference Maps', fontsize=14, fontweight='bold')
    
    # Add colorbar on the right
    fig.subplots_adjust(right=0.96)
    cbar_ax = fig.add_axes([0.97, 0.35, 0.01, 0.3])
    fig.colorbar(im_pmr, cax=cbar_ax, label='Standard Deviation')
    
    plt.suptitle(f'Non-linear Reconstruction Noise Standard Deviation Comparison - All Noise Levels (Slice {slice_idx}, N={targetN})', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Save figure in all formats (use bbox_inches='tight' which handles layout better than tight_layout with colorbar)
    output_base = save_root / f'comparison_concatenated_slice{slice_idx:04d}_N{targetN}{r4_suffix}{output_suffix}'
    plt.savefig(f'{output_base}.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(f'{output_base}.pdf', bbox_inches='tight', pad_inches=0.1)
    plt.savefig(f'{output_base}.svg', bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"Concatenated comparison plot saved to: {output_base}.{{png,pdf,svg}}")

def plot_multi_noise_grid(save_dir, run_tag, noise_scales, slice_idx, targetN,
                          cmap_var="jet", cmap_diff="hot", cmap_recon="gray",
                          vmin_var_override=None, vmax_var_override=None, r4_suffix=""):
    save_root = Path(save_dir) / run_tag
    pmr_list, ours_list, diff_list, recon_list = [], [], [], []
    for s in noise_scales:
        sdir = save_root / f"noise_{s:g}" / f"slice{slice_idx:04d}"
        pmr_p = sdir / f"pmr_var_N{targetN}.npy"
        ours_p = sdir / f"ours_var_N{targetN}.npy"
        recon_pref = sdir / "recon_noisy_mag.npy"
        recon_fallback = sdir / "recon_mag.npy"
        recon_path = recon_pref if recon_pref.exists() else recon_fallback
        if not (pmr_p.exists() and ours_p.exists()):
            raise FileNotFoundError(f"Missing variance files for noise={s} in {sdir}")
        pmr = np.load(pmr_p)
        ours = np.load(ours_p)
        diff = np.abs(ours - pmr)
        pmr_list.append(pmr); ours_list.append(ours); diff_list.append(diff)
        if recon_path.exists():
            recon_list.append(np.load(recon_path))
            if recon_path == recon_fallback:
                print(f"Warning: recon_noisy_mag.npy not found for noise={s}, using clean reconstruction in grid plot")
        else:
            raise FileNotFoundError(f"Missing reconstruction magnitude for noise={s} in {sdir}")
    have_recon = all([r is not None for r in recon_list])
    nrows = 4 if have_recon else 3
    vmax_recon = max(r.max() for r in recon_list) if have_recon else None
    vmin_var = vmin_var_override if vmin_var_override is not None else \
        min(m.min() for m in pmr_list + ours_list)
    vmax_var = vmax_var_override if vmax_var_override is not None else \
        max(m.max() for m in pmr_list + ours_list)
    vmax_diff = max(m.max() for m in diff_list)
    cols = len(noise_scales)
    fig, axes = plt.subplots(nrows, cols, figsize=(4.0*cols, 3.5*nrows), squeeze=False)
    for j, s in enumerate(noise_scales):
        title = f"σ×{s:g}"
        row0 = 0
        if have_recon:
            ax = axes[row0, j]
            ax.imshow(recon_list[j], cmap=cmap_recon, vmin=0, vmax=vmax_recon)
            if j == 0: ax.set_ylabel("Recon", fontsize=12, fontweight="bold")
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.axis("off")
            row0 = 1
        ax = axes[row0 + 0, j]
        ax.imshow(pmr_list[j], cmap=cmap_var, vmin=vmin_var, vmax=vmax_var)
        if j == 0: ax.set_ylabel(f"PMR @ N={targetN}", fontsize=12, fontweight="bold")
        ax.axis("off")
        ax = axes[row0 + 1, j]
        ax.imshow(ours_list[j], cmap=cmap_var, vmin=vmin_var, vmax=vmax_var)
        if j == 0: ax.set_ylabel("Ours", fontsize=12, fontweight="bold")
        ax.axis("off")
        ax = axes[row0 + 2, j]
        ax.imshow(diff_list[j], cmap=cmap_diff, vmin=0, vmax=vmax_diff)
        if j == 0: ax.set_ylabel("|Diff|", fontsize=12, fontweight="bold")
        ax.axis("off")
    plt.tight_layout()
    out_png = save_root / f"fig_noise_grid_slice{slice_idx:04d}_N{targetN}{r4_suffix}.png"
    out_pdf = save_root / f"fig_noise_grid_slice{slice_idx:04d}_N{targetN}{r4_suffix}.pdf"
    plt.savefig(out_png, dpi=200)
    plt.savefig(out_pdf)
    plt.close()

def main():
    ap = argparse.ArgumentParser(description="Nonlinear CS: PMR vs JVP (single script; multi-noise; shared grid).")
    ap.add_argument('--slice-idx', type=int, default=20)
    ap.add_argument('--grid-N', nargs='+', type=int, default=[3000],
                    help='N values for variance grid; use [3000] for single-N runs')
    ap.add_argument('--target-N', type=int, default=3000)
    ap.add_argument('--noise-scales', nargs='+', type=float, default=[1.0, 10.0, 50.0, 100.0, 200.0])
    ap.add_argument('--sigma0', type=float, default=1e-7)
    ap.add_argument('--accel', type=int, default=4, help='Acceleration factor R for undersampling (e.g. 4 for R=4)')
    ap.add_argument('--lamda', type=float, default=1e-2)
    ap.add_argument('--num-iters', type=int, default=100)
    ap.add_argument('--batch-size', type=int, default=100)
    ap.add_argument('--roi', type=str, default='central', choices=['central','all'])
    ap.add_argument('--roi-size', type=int, default=128)
    ap.add_argument('--data-file', type=str, default='experiments/fastmri_knee/data/file1000000.h5')
    ap.add_argument('--save-dir', type=str, default='experiments/fastmri_knee/results')
    ap.add_argument('--run-tag', type=str, default='cs_nonlinear_R4',
                    help='Run tag for output dir; use _R4 suffix when accel=4')
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--tv-inner', type=int, default=50, help='Chambolle inner iterations of the TV prox (toolbox default 50)')
    ap.add_argument('--tv-tol', type=float, default=1e-4, help='absolute exit tolerance on the dual increment (toolbox default 1e-4; 0 forces --tv-inner iterations)')
    ap.add_argument('--fista-ptol', type=float, default=0.5, help='FISTA iterate-change exit rule in percent (toolbox default 0.5); a negative value runs exactly --num-iters iterations')
    ap.add_argument('--tv-prox', choices=['magnitude', 'complex'], default='magnitude', help='TV proximal operator: magnitude/phase TV (toolbox) or complex-valued vector TV')
    ap.add_argument('--N-ref', type=int, default=10000, help='PMR reference replica count per noise scale (default 10000)')
    ap.add_argument('--reuse-ref', action='store_true', help='load an existing ref_pmr_var_N{N_ref}.npy instead of recomputing it')
    ap.add_argument('--ref-only', action='store_true', help='compute and save the reference(s), then stop')
    ap.add_argument('--pico-reference', action='store_true',
                    help='also compute the PICO gold map at N_ref, so each estimator is scored '
                         'against its own high-sample limit as in the original submission')
    ap.add_argument('--jvp-min-samples', type=int, default=5000, help='single-grid runs use max(N, this) JVP samples; pass 0 for exactly N')
    ap.add_argument('--std-vmin', type=float, default=None,
                    help='If set, overrides the lower bound of the variance/std colormap for all plots')
    ap.add_argument('--std-vmax', type=float, default=None,
                    help='If set, overrides the upper bound of the variance/std colormap for all plots')
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
    root = Path(args.save_dir) / args.run_tag
    root.mkdir(parents=True, exist_ok=True)
    (root / "run_manifest.json").write_text(json.dumps(dict(
        grid_N=args.grid_N,
        target_N=args.target_N,
        noise_scales=args.noise_scales,
        roi=args.roi, roi_size=args.roi_size,
        accel=args.accel,
        lamda=args.lamda,
        num_iters=args.num_iters,
        batch_size=args.batch_size,
        sigma0=args.sigma0,
        data_file=args.data_file,
        slice_idx=args.slice_idx,
        seed=args.seed,
        N_ref=args.N_ref, reuse_ref=args.reuse_ref, ref_only=args.ref_only, pico_reference=args.pico_reference, jvp_min_samples=args.jvp_min_samples,
        tv_prox=args.tv_prox, tv_inner=args.tv_inner, tv_tol=args.tv_tol, fista_ptol=args.fista_ptol
    ), indent=2))
    for s in args.noise_scales:
        out_dir = root / f"noise_{s:g}" / f"slice{args.slice_idx:04d}"
        print(f"\n=== Noise scale σ×{s:g} -> {out_dir} ===")
        run_one_noise(args, img, mps, ksp_full, torch_dev, s, out_dir)
    r4_suffix = "_R4" if args.accel == 4 else ""
    print("\nComposing multi-noise grid figure...")
    try:
        plot_multi_noise_grid(args.save_dir, args.run_tag, args.noise_scales, args.slice_idx,
                              args.target_N, vmin_var_override=args.std_vmin**2 if args.std_vmin is not None else None,
                              vmax_var_override=args.std_vmax**2 if args.std_vmax is not None else None,
                              r4_suffix=r4_suffix)
    except Exception as e:
        print(f"Warning: Failed to generate multi-noise grid: {e}")
    
    r4_suffix = "_R4" if args.accel == 4 else ""
    print("\nComposing concatenated comparison plots...")
    try:
        plot_concatenated_comparisons(args.save_dir, args.run_tag, args.noise_scales,
                                      args.slice_idx, args.target_N,
                                      vmin_std_override=args.std_vmin, vmax_std_override=args.std_vmax,
                                      r4_suffix=r4_suffix)
    except Exception as e:
        print(f"Warning: Failed to generate concatenated comparison plots: {e}")
        import traceback
        traceback.print_exc()

    if r4_suffix:
        print("\nSaving R=4 zero-filled concatenated SVG...")
        try:
            plot_zf_concatenated_svg(args.save_dir, args.run_tag, args.noise_scales,
                                     args.slice_idx, r4_suffix=r4_suffix)
        except Exception as e:
            print(f"Warning: Failed to generate ZF concatenated SVG: {e}")
    
    print("Done.")

if __name__ == '__main__':
    main()
