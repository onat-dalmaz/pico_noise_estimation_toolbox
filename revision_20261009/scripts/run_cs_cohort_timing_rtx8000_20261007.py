"""Equal-budget CS timing on Quadro RTX 8000 (2026-10-07): one subject/slice per invocation, both methods in one session.

Cases, settings, seeds and method order are those of scripts/run_cs_cohort_timing_20260930.py (manifest
scripts/cs_cohort_timing_manifest_20260930.csv): R=2, lambda_TV=3e-7, 30 FISTA iterations, complex TV with 50 Chambolle
iterations (tol 0), sigma 1e-7, N=2000 for each estimator, seed 123.
  PMR : unchanged September routine (pmr_variance_map_f64, replicas reconstructed one at a time, Welford accumulation,
        library in-place gridding).
  PICO: same estimator and probes, with the baseline reconstruction computed once
        (cs_pico_linearize_pilot_20261007.pico_linearized; batch 25; out-of-place gridding enabled only while PICO runs).
Timed scope (both): sample generation, all reconstructions / Jacobian-vector products (PICO's one-time baseline pass
included), variance accumulation; CUDA-synchronized wall clock; warm-up excluded."""
import argparse, hashlib, json, os, platform, sys, time
from contextlib import contextmanager
from pathlib import Path
import numpy as np, torch, h5py
from einops import rearrange

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nonlinear_cs_noise_sweep as ncs
import run_cs_cohort_timing_20260930 as sept
from cs_pico_linearize_pilot_20261007 import pico_linearized
from mr_recon.algs import power_method_operator
from mr_recon.prox_complex_tv import ComplexTV


@contextmanager
def functional_gridding():
    """Install the vmap-safe out-of-place gridding (identical maps) and restore the library code afterwards."""
    import cs_batched_timing as cbt, mr_recon.indexing, mr_recon.fourier, mr_recon.linops
    from mr_recon.fourier import gridded_nufft
    from mr_recon.linops import sense_linop
    saved = dict(FISTA=ncs.FISTA, nf=gridded_nufft.forward, na=gridded_nufft.adjoint, sf=sense_linop.forward, sa=sense_linop.adjoint,
                 mg={m: getattr(m, 'multi_grid') for m in (mr_recon.indexing, mr_recon.fourier, mr_recon.linops) if hasattr(m, 'multi_grid')})
    cbt.install_functional_gridding()
    try:
        yield
    finally:
        ncs.FISTA = saved['FISTA']; gridded_nufft.forward, gridded_nufft.adjoint = saved['nf'], saved['na']
        sense_linop.forward, sense_linop.adjoint = saved['sf'], saved['sa']
        for m, f in saved['mg'].items(): setattr(m, 'multi_grid', f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subject', required=True); ap.add_argument('--slice-idx', type=int, required=True)
    ap.add_argument('--method-order', choices=['pico-first', 'pmr-first'], required=True)
    ap.add_argument('--out-root', default='experiments/fastmri_knee/results/cs_cohort_timing_equalN2000_rtx8000_20261007')
    ap.add_argument('--n', type=int, default=2000); ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--batch-size', type=int, default=25); ap.add_argument('--roi-size', type=int, default=128)
    a = ap.parse_args()
    out_dir = Path(a.out_root) / a.subject / f"slice{a.slice_idx:04d}"
    if (out_dir / 'cs_subject_metrics.json').exists():
        print('[rtx8000-cohort] completed case exists; skip', out_dir); return
    ncs.FISTA_PTOL_EXIT = None                                    # exactly 30 FISTA iterations (September setting)
    dev = torch.device('cuda'); sync = lambda: torch.cuda.synchronize(dev)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    with h5py.File(f'experiments/fastmri_knee/data/{a.subject}.h5', 'r') as f:
        nslices = f['reconstruction_rss'].shape[0]
        img = torch.tensor(f['reconstruction_rss'], device=dev)[a.slice_idx]
        mps = torch.tensor(f['jsense-12-cf=4']['maps'], device=dev)[a.slice_idx, :, :, :, 0]
        ksp_full = torch.tensor(f['kspace'][:], device=dev)[a.slice_idx]
    C = mps.shape[-1]; mps = rearrange(mps, 'h w c -> c h w')
    if ksp_full.ndim == 3 and ksp_full.shape[-1] == C:
        ksp_full = rearrange(ksp_full, 'h w c -> c h w')
    im_size = (img.shape[0], img.shape[1])
    A, ksp, n_samp = sept.build_operator(img, mps, ksp_full, dev, 2)
    proxg = ComplexTV(tuple(im_size), 3e-7, max_iter=50, tol=0.0)
    _, max_eigen = power_method_operator(A.normal, torch.randn_like(img), verbose=False)
    max_eigen = float(max_eigen) * 1.01; K = 30
    recon = ncs.fista_reconstruct(ksp, A, proxg, K, max_eigen)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / 'recon_mag.npy', recon.abs().detach().cpu().numpy().astype(np.float64))
    sync()
    with functional_gridding():
        _ = pico_linearized(ksp, A, proxg, K, max_eigen, a.batch_size, a.batch_size); sync()     # warm-up (excluded)
    _ = sept.pmr_variance_map_f64(ksp, A, proxg, K, max_eigen, sigma=1e-7, n_replicas=1, seed=a.seed + 777); sync()
    def measure_pico():
        torch.manual_seed(a.seed + 100)
        with functional_gridding():
            sync(); t0 = time.perf_counter()
            v = pico_linearized(ksp, A, proxg, K, max_eigen, a.n, a.batch_size)
            sync(); el = time.perf_counter() - t0
        return v.detach().cpu().numpy().astype(np.float64), el
    def measure_pmr():
        torch.manual_seed(a.seed + 777)
        sync(); t0 = time.perf_counter()
        v = sept.pmr_variance_map_f64(ksp, A, proxg, K, max_eigen, sigma=1e-7, n_replicas=a.n, seed=None)
        sync(); el = time.perf_counter() - t0
        return v.detach().cpu().numpy().astype(np.float64), el
    if a.method_order == 'pico-first':
        ours, t_pico = measure_pico(); pmr, t_pmr = measure_pmr()
    else:
        pmr, t_pmr = measure_pmr(); ours, t_pico = measure_pico()
    if not (np.isfinite(ours).all() and np.isfinite(pmr).all()) or (ours < 0).any() or (pmr < 0).any():
        raise ValueError('invalid variance estimate')
    np.save(out_dir / 'pico_var.npy', ours); np.save(out_dir / 'pmr_var_f64.npy', pmr)
    H, W = im_size; cy, cx, hs = H // 2, W // 2, a.roi_size // 2
    roi = np.zeros((H, W), bool); roi[cy - hs:cy + hs, cx - hs:cx + hs] = True
    os_, ps_ = np.sqrt(ours), np.sqrt(pmr)
    row = dict(subject=a.subject, slice_idx=a.slice_idx, nslices=int(nslices), accel=2, lamda=3e-7, sigma=1e-7, num_iters=K,
               n_pmr=a.n, n_pico=a.n, n_samples=int(n_samp),
               nrmse_std_roi_pct=sept._nrmse(os_[roi], ps_[roi]) * 100.0, corr_std_roi=sept._corr(os_[roi], ps_[roi]),
               nrmse_std_full_image_pct=sept._nrmse(os_, ps_) * 100.0,
               t_pico_s=t_pico, t_pmr_s=t_pmr, pmr_over_pico_runtime=t_pmr / t_pico,
               batch_size=a.batch_size, method_order=a.method_order,
               device=torch.cuda.get_device_name(dev), hostname=platform.node(), pytorch_version=torch.__version__,
               cuda_version=torch.version.cuda, tv_prox='complex', tv_inner=50, tv_tol=0.0, seed=a.seed, roi_size=a.roi_size,
               image_shape=list(im_size), coils=int(C),
               pico_implementation='baseline reconstruction computed once (stored FISTA/Chambolle trajectory + tangent recursion), batch 25',
               pmr_implementation='September routine: replicas reconstructed one at a time, single-pass Welford complex128/float64',
               timing_clock='perf_counter; CUDA synchronize before and after each method',
               timed_scope='sample generation, all reconstructions / Jacobian-vector products (incl. PICO one-time baseline pass), variance accumulation; excludes setup, warm-up, device-to-host copy and disk I/O',
               comparison_type='equal sample budgets; no claim of equal error or time-to-target',
               script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    tmp = out_dir / 'cs_subject_metrics.json.tmp'; tmp.write_text(json.dumps(row, indent=2, default=float) + '\n')
    tmp.replace(out_dir / 'cs_subject_metrics.json')
    print(f"[rtx8000-cohort] {a.subject} {a.slice_idx} t_pico={t_pico:.1f}s t_pmr={t_pmr:.1f}s ratio={t_pmr/t_pico:.3f} "
          f"NRMSE(std,ROI)={row['nrmse_std_roi_pct']:.2f}%", flush=True)


if __name__ == '__main__':
    main()
