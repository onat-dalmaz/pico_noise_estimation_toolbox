"""Pilot (2026-10-07): PICO for the CS pipeline with the baseline reconstruction computed once.

The estimator is unchanged: the same data-domain Rademacher probes (same RNG draws), the Jacobian of the same
reconstruction (fista_reconstruct: 30 FISTA iterations, ComplexTV prox with 50 Chambolle iterations, lambda_TV 3e-7),
and the same accumulation of squared Jacobian responses.  Only the evaluation of the Jacobian-vector products
changes.  The current code (ncs.jvp_hutch_variance) re-runs the baseline reconstruction inside every batched jvp
call.  Here the baseline FISTA/Chambolle trajectory is computed once and the quantities its derivative needs (q, |q|
of every Chambolle step) are stored; each probe batch is then propagated through the tangent recursion (the
linear FISTA updates, A^H A, and the derivative of each Chambolle step, with PyTorch's clamp-derivative convention).
torch.func.linearize cannot trace this pipeline under torch 2.0.1 (forward AD x Python scalars inside make_fx).
The one-time baseline pass is counted in PICO's time.  Out-of-place gridding (cs_batched_timing.py; identical maps)
lets the operator run under vmap.

Pilot checks on one cohort slice: equivalence with the current implementation on identical probes, and
synchronized wall time of the current implementation (batch 25, in-place gridding, as in the cohort) and of the
new one at several batch sizes, with peak GPU memory.
"""
import argparse, json, os, platform, sys, time, hashlib
from pathlib import Path
import numpy as np, torch, h5py
from einops import rearrange
from torch.func import vmap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nonlinear_cs_noise_sweep as ncs
from run_cs_cohort_timing_20260930 import build_operator
from mr_recon.algs import power_method_operator
from mr_recon.prox_complex_tv import ComplexTV, _grad2, _div2


def draw_probes(B, shape, device, dtype):
    # identical draws to ncs.jvp_hutch_variance
    real_part = (torch.randint(0, 2, (B,) + shape, device=device).float() * 2 - 1)
    imag_part = (torch.randint(0, 2, (B,) + shape, device=device).float() * 2 - 1)
    return (real_part + 1j * imag_part).to(dtype)


def baseline_trajectory(kspace, A, proxg, num_iters, max_eigen):
    """fista_reconstruct + FISTA + ComplexTV.forward, op for op, storing (q, |q|) of every Chambolle step."""
    w = 1.0 * proxg.lamda
    tau = 0.125 if proxg.tau is None else float(proxg.tau)
    AHb = A.adjoint(kspace) / (max_eigen ** 0.5)
    x = AHb.clone(); z = x.clone(); store = []
    for k in range(num_iters):
        x_old = x.clone(); x = z.clone()
        y = x - (A.normal(x) / max_eigen - AHb)
        X = torch.stack((y.real, y.imag), dim=0)
        p = torch.zeros((2,) + tuple(X.shape), dtype=X.dtype, device=X.device)
        inner = []
        for _ in range(proxg.max_iter):
            q = p + tau * _grad2(_div2(p) - X / w)
            nrm = torch.sqrt((q * q).sum(dim=(0, 1), keepdim=True))
            p = q / torch.clamp(nrm, min=1.0)
            inner.append((q, nrm))
        u = X - w * _div2(p)
        x = torch.complex(u[0], u[1])
        z = x if k == 0 else x + (k / (k + 3)) * (x - x_old)
        store.append(inner)
    return x, store


def make_tangent(A, proxg, num_iters, max_eigen, store):
    w = 1.0 * proxg.lamda
    tau = 0.125 if proxg.tau is None else float(proxg.tau)
    def tangent(v):
        dAHb = A.adjoint(v) / (max_eigen ** 0.5)
        dx = dAHb.clone(); dz = dx.clone()
        for k in range(num_iters):
            dx_old = dx.clone(); dx = dz.clone()
            dy = dx - (A.normal(dx) / max_eigen - dAHb)
            dX = torch.stack((dy.real, dy.imag), dim=0)
            dp = None
            for q, nrm in store[k]:
                g = (-dX / w) if dp is None else (_div2(dp) - dX / w)
                dq = tau * _grad2(g) if dp is None else dp + tau * _grad2(g)
                m = torch.clamp(nrm, min=1.0)
                dm = torch.where(nrm >= 1.0, (q * dq).sum(dim=(0, 1), keepdim=True) / nrm, torch.zeros_like(nrm))
                dp = dq / m - q * dm / (m * m)
            du = dX - w * _div2(dp)
            dx = torch.complex(du[0], du[1])
            dz = dx if k == 0 else dx + (k / (k + 3)) * (dx - dx_old)
        return dx
    return tangent


def pico_linearized(kspace, A, proxg, num_iters, max_eigen, num_samples, batch_size):
    """Same estimator as ncs.jvp_hutch_variance; the baseline reconstruction is computed once."""
    with torch.no_grad():
        _, store = baseline_trajectory(kspace, A, proxg, num_iters, max_eigen)
        tangent = make_tangent(A, proxg, num_iters, max_eigen, store)
        def sample(v):
            u = tangent(v)
            return u.real.pow(2) + u.imag.pow(2)
        total = None
        for start in range(0, num_samples, batch_size):
            B = min(batch_size, num_samples - start)
            s = vmap(sample)(draw_probes(B, tuple(kspace.shape), kspace.device, kspace.dtype)).sum(0)
            total = s if total is None else total + s
        return total / num_samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-file', default='experiments/fastmri_knee/data/file1000000.h5')
    ap.add_argument('--slice-idx', type=int, default=13)
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--N', type=int, default=200)
    ap.add_argument('--N-check', type=int, default=50)
    ap.add_argument('--batch-sizes', type=int, nargs='+', default=[25, 50, 100])
    ap.add_argument('--functional-gridding', action='store_true', help='ignored (always installed for the new path)')
    ap.add_argument('--out-root', default='experiments/fastmri_knee/results/cs_pico_linearize_pilot_20261007')
    a = ap.parse_args()
    ncs.FISTA_PTOL_EXIT = None                       # cohort setting: exactly 30 FISTA iterations
    dev = torch.device('cuda'); sync = lambda: torch.cuda.synchronize(dev)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    with h5py.File(a.data_file, 'r') as f:
        img = torch.tensor(f['reconstruction_rss'], device=dev)[a.slice_idx]
        mps = torch.tensor(f['jsense-12-cf=4']['maps'], device=dev)[a.slice_idx, :, :, :, 0]
        ksp_full = torch.tensor(f['kspace'][:], device=dev)[a.slice_idx]
    C = mps.shape[-1]; mps = rearrange(mps, 'h w c -> c h w')
    if ksp_full.ndim == 3 and ksp_full.shape[-1] == C:
        ksp_full = rearrange(ksp_full, 'h w c -> c h w')
    im_size = (img.shape[0], img.shape[1])
    A, ksp, n_samp = build_operator(img, mps, ksp_full, dev, 2)
    proxg = ComplexTV(tuple(im_size), 3e-7, max_iter=50, tol=0.0)
    _, max_eigen = power_method_operator(A.normal, torch.randn_like(img), verbose=False)
    max_eigen = float(max_eigen) * 1.01
    K = 30
    x_ref = ncs.fista_reconstruct(ksp, A, proxg, K, max_eigen); sync()
    out = dict(case=f"{Path(a.data_file).stem}/slice{a.slice_idx:04d}", device=torch.cuda.get_device_name(dev),
               host=platform.node(), torch=torch.__version__, N=a.N, image_shape=list(im_size), coils=int(C),
               script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    def timed(fn):
        torch.cuda.reset_peak_memory_stats(dev); sync(); t0 = time.perf_counter(); r = fn(); sync()
        return r, time.perf_counter() - t0, torch.cuda.max_memory_allocated(dev) / 2**30
    # current implementation, exactly as in the cohort (in-place gridding, batch 25)
    torch.manual_seed(a.seed + 100); v_cur = ncs.jvp_hutch_variance(ksp, A, proxg, K, max_eigen, num_samples=a.N_check, batch_size=25)
    _ = ncs.jvp_hutch_variance(ksp, A, proxg, K, max_eigen, num_samples=25, batch_size=25)
    torch.manual_seed(a.seed + 100)
    _, t, m = timed(lambda: ncs.jvp_hutch_variance(ksp, A, proxg, K, max_eigen, num_samples=a.N, batch_size=25))
    out['current_batch25'] = dict(seconds=t, per_probe=t / a.N, peak_gib=m); print('[pilot] current b25', out['current_batch25'], flush=True)
    # new path
    import cs_batched_timing as cbt
    cbt.install_functional_gridding()
    (x_new, store), t, m = timed(lambda: baseline_trajectory(ksp, A, proxg, K, max_eigen))
    out['baseline_pass'] = dict(seconds=t, peak_gib=m, rel_l2_vs_fista=float(torch.linalg.norm(x_new - x_ref) / torch.linalg.norm(x_ref)))
    print('[pilot] baseline pass', out['baseline_pass'], flush=True); del store, x_new
    torch.manual_seed(a.seed + 100); v_lin = pico_linearized(ksp, A, proxg, K, max_eigen, a.N_check, 25)
    out['equivalence_rel_l2'] = float(torch.linalg.norm(v_lin - v_cur) / torch.linalg.norm(v_cur))
    out['equivalence_max_rel'] = float(((v_lin - v_cur).abs() / v_cur.abs().clamp_min(1e-30)).max())
    print('[pilot] equivalence', out['equivalence_rel_l2'], out['equivalence_max_rel'], flush=True)
    out['linearized'] = {}
    for B in a.batch_sizes:
        try:
            _ = pico_linearized(ksp, A, proxg, K, max_eigen, B, B)
            torch.manual_seed(a.seed + 100)
            _, t, m = timed(lambda: pico_linearized(ksp, A, proxg, K, max_eigen, a.N, B))
            out['linearized'][str(B)] = dict(seconds=t, per_probe=t / a.N, peak_gib=m)
        except torch.cuda.OutOfMemoryError:
            out['linearized'][str(B)] = dict(error='OOM'); torch.cuda.empty_cache()
        print('[pilot] linearized', B, out['linearized'][str(B)], flush=True)
    d = Path(a.out_root) / out['case']; d.mkdir(parents=True, exist_ok=True)
    (d / 'pilot.json').write_text(json.dumps(out, indent=1) + '\n'); print('[pilot] wrote', d / 'pilot.json')


if __name__ == '__main__':
    main()
