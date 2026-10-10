#!/usr/bin/env python3
"""
Reconstruction script for spiral brain imaging dataset.
Starts with basic reconstruction and plotting, g-factor analysis to be added later.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

from mr_recon.fourier import sigpy_nufft
from mr_recon.linops import sense_linop, batching_params
from mr_recon.recons import CG_SENSE_recon
from mr_recon.algs import density_compensation

def run_spiral_brain_recon(acquisition='nomb', slice_idx=11, lamda_l2=0.0, max_iter=100):
    """
    Reconstruct spiral brain imaging data.
    
    Parameters:
    -----------
    acquisition : str
        Acquisition type: 'nomb', 'caipi', 'wave', 'wave_nonlin', or 'fs'
    slice_idx : int
        Slice index to reconstruct (0-15 for SMS data, ignored for 'fs')
    lamda_l2 : float
        L2 regularization parameter
    max_iter : int
        Maximum iterations for CG-SENSE reconstruction
    """
    
    # --- Experiment Configuration ---
    max_eigen = 1  # Estimate max eigenvalue automatically
    tol = 1e-2
    output_dir = f"experiments/sim/results/{acquisition}"
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Data Loading ---
    data_dir = "experiments/sim"
    torch_dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    complex_dtype = torch.complex64
    print(f"Using device: {torch_dev}")
    
    # Load SMS simulation data
    print(f"Loading SMS simulation data...")
    sms_data = torch.load(os.path.join(data_dir, "sms_sim.pt"), map_location=torch_dev)
    
    # Extract data based on acquisition type
    if acquisition == 'fs':
        # Fully sampled - 3D data
        ksp = sms_data['ksps'][acquisition]  # [20, 316, 110, 16]
        trj = sms_data['trjs'][acquisition]   # [316, 110, 16, 3]
        mps = sms_data['mps']                 # [20, 110, 110, 16]
        img_gt = sms_data['img']               # [110, 110, 16]
        
        # For fully sampled, we can reconstruct all slices or a specific slice
        print(f"Fully sampled data: {ksp.shape}, trajectory: {trj.shape}")
        print(f"Reconstructing slice {slice_idx}...")
        
        # Extract single slice
        ksp_slice = ksp[:, :, :, slice_idx]  # [20, 316, 110]
        trj_slice = trj[:, :, slice_idx, :]  # [316, 110, 3]
        mps_slice = mps[:, :, :, slice_idx]  # [20, 110, 110]
        img_gt_slice = img_gt[:, :, slice_idx]  # [110, 110]
        
        # Reshape trajectory and k-space for 2D reconstruction
        # Trajectory: [316, 110, 3] -> [316*110, 2] (use only kx, ky)
        # K-space: [20, 316, 110] -> [20, 316*110]
        trj_2d = trj_slice[:, :, :2].reshape(-1, 2)  # [316*110, 2]
        ksp_2d = ksp_slice.reshape(ksp_slice.shape[0], -1)  # [20, 316*110]
        
        trj = trj_2d
        ksp = ksp_2d
        
    else:
        # Other acquisitions - 2D data
        ksp = sms_data['ksps'][acquisition]  # [20, shots, points]
        trj = sms_data['trjs'][acquisition]   # [shots, points, 3]
        mps = sms_data['mps']                 # [20, 110, 110, 16]
        img_gt = sms_data['img']               # [110, 110, 16]
        
        # Extract single slice for coil maps and ground truth
        mps_slice = mps[:, :, :, slice_idx]  # [20, 110, 110]
        img_gt_slice = img_gt[:, :, slice_idx]  # [110, 110]
        
        print(f"{acquisition} data: ksp {ksp.shape}, trajectory {trj.shape}")
        print(f"Reconstructing slice {slice_idx}...")
        
        # Reshape trajectory and k-space for 2D reconstruction
        # Trajectory: [shots, points, 3] -> [shots*points, 2] (use only kx, ky)
        # K-space: [20, shots, points] -> [20, shots*points]
        trj_2d = trj[:, :, :2].reshape(-1, 2)  # [shots*points, 2]
        ksp_2d = ksp.reshape(ksp.shape[0], -1)  # [20, shots*points]
        
        trj = trj_2d
        ksp = ksp_2d
    
    # Move to device and set dtype
    ksp = ksp.type(complex_dtype).to(torch_dev)
    trj = trj.type(torch.float32).to(torch_dev)
    mps_slice = mps_slice.type(complex_dtype).to(torch_dev)
    img_gt_slice = img_gt_slice.type(complex_dtype).to(torch_dev)
    
    im_size = img_gt_slice.shape  # (110, 110)
    C = mps_slice.shape[0]  # 20 coils
    
    print(f"Image size: {im_size}, Number of coils: {C}")
    print(f"K-space shape: {ksp.shape}, Trajectory shape: {trj.shape}")
    
    # --- Compute DCF if needed ---
    # Check if DCF needs to be computed (trajectory shape should match)
    print("Computing density compensation function...")
    dcf = density_compensation(trj, im_size, num_iters=30, method='sigpy')
    dcf = dcf.type(torch.float32).to(torch_dev)
    print(f"DCF shape: {dcf.shape}, max: {dcf.max():.4f}, min: {dcf.min():.4f}")
    
    # --- SENSE Operator Setup ---
    print("Setting up SENSE operator...")
    nufft = sigpy_nufft(im_size, width=4)
    bparams = batching_params(coil_batch_size=C)
    
    A = sense_linop(
        im_size=im_size,
        trj=trj,
        mps=mps_slice,
        dcf=dcf,
        nufft=nufft,
        bparams=bparams,
        use_toeplitz=True
    )
    
    # --- Reconstruct Image ---
    print(f"Reconstructing with CG-SENSE (max_iter={max_iter}, lamda_l2={lamda_l2})...")
    recon_image = CG_SENSE_recon(
        A, ksp, max_iter, lamda_l2, max_eigen, 
        verbose=True, tolerance=tol
    )
    
    # --- Compute Metrics ---
    recon_mag = recon_image.abs()
    gt_mag = img_gt_slice.abs()
    
    # Normalize both images to [0, 1] for better comparison
    recon_mag_norm = recon_mag / (recon_mag.max() + 1e-10)
    gt_mag_norm = gt_mag / (gt_mag.max() + 1e-10)
    
    # Normalized Root Mean Squared Error (using normalized images)
    mse = torch.mean((recon_mag_norm - gt_mag_norm) ** 2)
    nrmse = torch.sqrt(mse) * 100
    
    # Peak Signal-to-Noise Ratio
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    
    print(f"\nReconstruction Metrics:")
    print(f"  NRMSE: {nrmse.item():.2f}%")
    print(f"  PSNR: {psnr.item():.2f} dB")
    print(f"  Recon magnitude: max={recon_mag.max().item():.4f}, mean={recon_mag.mean().item():.4f}")
    print(f"  GT magnitude: max={gt_mag.max().item():.4f}, mean={gt_mag.mean().item():.4f}")
    
    # --- Plotting ---
    print("Generating plots...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Use consistent vmax for magnitude images
    vmax_mag = max(recon_mag.max().item(), gt_mag.max().item())
    
    # Ground truth magnitude
    im1 = axes[0, 0].imshow(gt_mag.cpu().numpy(), cmap='gray', vmin=0, vmax=vmax_mag)
    axes[0, 0].set_title(f'Ground Truth (Slice {slice_idx})', fontsize=14)
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Reconstructed magnitude
    im2 = axes[0, 1].imshow(recon_mag.cpu().numpy(), cmap='gray', vmin=0, vmax=vmax_mag)
    axes[0, 1].set_title(f'Reconstructed (Slice {slice_idx})', fontsize=14)
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Normalized comparison
    im3 = axes[0, 2].imshow(recon_mag_norm.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title(f'Reconstructed (Normalized)', fontsize=14)
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Difference (absolute)
    diff = (recon_mag_norm - gt_mag_norm).abs()
    im4 = axes[1, 0].imshow(diff.cpu().numpy(), cmap='hot', vmin=0, vmax=1)
    axes[1, 0].set_title(f'Absolute Difference (Normalized)', fontsize=14)
    axes[1, 0].axis('off')
    plt.colorbar(im4, ax=axes[1, 0])
    
    # Phase
    recon_phase = recon_image.angle()
    im5 = axes[1, 1].imshow(recon_phase.cpu().numpy(), cmap='hsv', vmin=-np.pi, vmax=np.pi)
    axes[1, 1].set_title(f'Reconstructed Phase', fontsize=14)
    axes[1, 1].axis('off')
    plt.colorbar(im5, ax=axes[1, 1])
    
    # K-space visualization (first coil, first 1000 points)
    ksp_vis = ksp.abs()[0, :min(1000, ksp.shape[1])].cpu().numpy()
    axes[1, 2].plot(ksp_vis)
    axes[1, 2].set_title(f'K-space Magnitude (Coil 0, first 1000 pts)', fontsize=14)
    axes[1, 2].set_xlabel('Sample')
    axes[1, 2].set_ylabel('Magnitude')
    axes[1, 2].grid(True, alpha=0.3)
    
    fig.suptitle(
        f'Spiral Brain Reconstruction: {acquisition.upper()}\n'
        f'NRMSE: {nrmse.item():.2f}%, PSNR: {psnr.item():.2f} dB, '
        f'λ={lamda_l2}, iter={max_iter}',
        fontsize=16
    )
    
    # --- Save Outputs ---
    plot_path = os.path.join(output_dir, f'recon_{acquisition}_slice{slice_idx:02d}_L{lamda_l2}_iter{max_iter}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plot_path}")
    
    # Save reconstructed image
    recon_path = os.path.join(output_dir, f'recon_{acquisition}_slice{slice_idx:02d}_L{lamda_l2}_iter{max_iter}.pt')
    torch.save(recon_image.cpu(), recon_path)
    print(f"Reconstructed image saved to {recon_path}")
    
    # Save metrics
    metrics_path = os.path.join(output_dir, f'metrics_{acquisition}_slice{slice_idx:02d}_L{lamda_l2}_iter{max_iter}.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"Acquisition: {acquisition}\n")
        f.write(f"Slice: {slice_idx}\n")
        f.write(f"L2 Regularization: {lamda_l2}\n")
        f.write(f"Max Iterations: {max_iter}\n")
        f.write(f"NRMSE: {nrmse.item():.4f}%\n")
        f.write(f"PSNR: {psnr.item():.4f} dB\n")
    print(f"Metrics saved to {metrics_path}")
    
    plt.close()
    
    return recon_image, img_gt_slice, nrmse, psnr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Reconstruct spiral brain imaging data.")
    parser.add_argument('--acquisition', type=str, default='nomb', 
                       choices=['nomb', 'caipi', 'wave', 'wave_nonlin', 'fs'],
                       help='Acquisition type to reconstruct.')
    parser.add_argument('--slice', type=int, default=11, 
                       help='Slice index to reconstruct (0-15).')
    parser.add_argument('--lamda_l2', type=float, default=0.0, 
                       help='L2 regularization parameter.')
    parser.add_argument('--max_iter', type=int, default=100, 
                       help='Maximum iterations for CG-SENSE.')
    
    args = parser.parse_args()
    
    run_spiral_brain_recon(
        acquisition=args.acquisition,
        slice_idx=args.slice,
        lamda_l2=args.lamda_l2,
        max_iter=args.max_iter
    )

