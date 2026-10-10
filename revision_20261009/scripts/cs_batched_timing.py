"""Like-for-like per-sample cost of PICO and PMR for the nonlinear CS pipeline.

The CS experiments evaluate PICO's Jacobian-vector products in vmap batches
(``jvp_hutch_variance``, batch_size 25) but reconstruct PMR replicas one at a
time.  That compares a batched estimator with an unbatched one.  This script
times both estimators in one session, on one GPU, at the manuscript
configuration (R = 2, slice 20, complex TV with 50 fixed dual iterations,
30 FISTA iterations, lambda_TV = 3e-7, sigma0 = 1e-7), at the SAME batch size:

    * PICO   : vmap over a batch of JVP probes (the manuscript's estimator)
    * PMR    : vmap over a batch of FISTA reconstructions of ksp + noise
    * PMR-seq: the replica loop the experiments used (batch size 1, no vmap)

and checks that batched and sequential PMR give the same map for the same
noise realizations, so the batched replica estimator is the same estimator.

Output: experiments/fastmri_knee/results/cs_batched_timing/timing.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import h5py
import numpy as np
import torch
from einops import rearrange
from torch.func import vmap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nonlinear_cs_noise_sweep as ncs  # noqa: E402
from cs_recheck_pico_pmr import build_operator  # noqa: E402
from mr_recon.algs import power_method_operator  # noqa: E402
from mr_recon.prox_complex_tv import ComplexTV  # noqa: E402


def _multi_grid_functional(x, idx, final_size, raveled=False):
    """Out-of-place copy of mr_recon.indexing.multi_grid.

    The library version accumulates with an in-place index_add_ into a buffer it
    creates itself; under vmap that buffer carries no batch dimension, so a
    vmapped reconstruction cannot be formed.  index_add (out of place) gives
    the same values and is vmap-safe.  It is installed for every configuration
    timed after the as-run baseline, so all compared timings use the same code.
    """
    from mr_recon.indexing import ravel
    if not raveled:
        idx = ravel(idx, final_size, dim=-1)
    ndims = len(idx.shape)
    x_flat = torch.flatten(x, start_dim=-ndims, end_dim=-1)
    idx_flat = torch.flatten(idx)
    batch_dims = x_flat.shape[:-1]
    y = torch.zeros((*batch_dims, int(np.prod(final_size))), dtype=x_flat.dtype, device=x_flat.device)
    y = y.index_add(-1, idx_flat, x_flat)
    return y.reshape(*batch_dims, *final_size)


def _fista_no_exit(AHA, AHb, proxg, num_iters=20, ptol_exit=0.5, verbose=True):
    """mr_recon.algs.FISTA with the iterate-change exit test removed.  The
    manuscript runs FISTA with ptol_exit = -1, so the test never fires; under
    vmap the comparison itself cannot be traced, hence this copy."""
    assert ptol_exit is None or ptol_exit < 0, "only the fixed-count configuration is supported"
    x = AHb.clone()
    z = x.clone()
    for k in range(num_iters):
        x_old = x
        x = z
        gr = AHA(x) - AHb
        x = proxg(x - gr)
        if k == 0:
            z = x
        else:
            z = x + (k / (k + 3)) * (x - x_old)
    return x


ORIGINAL_FISTA = ncs.FISTA


def install_functional_gridding():
    ncs.FISTA = _fista_no_exit
    import mr_recon.indexing, mr_recon.fourier, mr_recon.linops
    from mr_recon.fourier import gridded_nufft, fft, ifft, np_to_torch
    from mr_recon.indexing import multi_index
    for mod in (mr_recon.indexing, mr_recon.fourier, mr_recon.linops):
        if hasattr(mod, "multi_grid"):
            mod.multi_grid = _multi_grid_functional

    # The gridded NUFFT writes each batch entry into a preallocated buffer; the
    # same values assembled with torch.stack are vmap-safe.
    def forward(self, img, trj):
        img_torch, trj_torch = np_to_torch(img, trj)
        d, N = trj.shape[-1], trj.shape[0]
        ksp_os = fft(self.grog_padder.forward(img_torch), dim=tuple(range(-d, 0)))
        ksp = torch.stack([multi_index(ksp_os[i], d, trj_torch[i].type(torch.int32)) for i in range(N)])
        return ksp * self.grid_oversamp

    def adjoint(self, ksp, trj):
        ksp_torch, trj_torch = np_to_torch(ksp, trj)
        d, N = trj.shape[-1], trj.shape[0]
        grid_os_size = self.grog_padder.pad_im_size
        ksp_os = torch.stack([_multi_grid_functional(ksp_torch[i], trj_torch[i].type(torch.int32), grid_os_size)
                              for i in range(N)])
        img = self.grog_padder.adjoint(ifft(ksp_os, dim=tuple(range(-d, 0))))
        return img * self.grid_oversamp

    gridded_nufft.forward = forward
    gridded_nufft.adjoint = adjoint

    # sense_linop accumulates coil batches into preallocated buffers; the same
    # sums formed out of place are vmap-safe.  Only the imperfection-free model
    # used by the CS experiments is covered.
    from mr_recon.linops import sense_linop
    from mr_recon.utils import batch_iterator
    from einops import einsum

    def sl_forward(self, img):
        assert self.imperf_model is None
        nc = self.mps.shape[0]
        parts = []
        for c, d in batch_iterator(nc, self.bparams.coil_batch_size):
            Sx = (self.mps[c:d] * img)[:, None, ...]
            parts.append(self.nufft.forward(Sx[None, ], self.trj[None, ...])[0][:, 0, ...])
        return torch.cat(parts, dim=0)

    def sl_adjoint(self, ksp):
        assert self.imperf_model is None
        nc = self.mps.shape[0]
        img = None
        for c, d in batch_iterator(nc, self.bparams.coil_batch_size):
            HWy = (ksp[c:d, ...] * self.dcf[None, ...])[:, None, ...]
            FHWy = self.nufft.adjoint(HWy[None, ...], self.trj[None, ...])[0]
            SFHWy = einsum(FHWy, self.mps[c:d].conj(), 'nc nseg ..., nc ... -> nseg ...')[0]
            img = SFHWy if img is None else img + SFHWy
        return img

    sense_linop.forward = sl_forward
    sense_linop.adjoint = sl_adjoint


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_pico(ksp, A, proxg, K, max_eigen, N, B):
    ncs.jvp_hutch_variance(ksp, A, proxg, K, max_eigen, num_samples=B, batch_size=B)  # warm-up
    sync(); t0 = time.perf_counter()
    ncs.jvp_hutch_variance(ksp, A, proxg, K, max_eigen, num_samples=N, batch_size=B)
    sync()
    return (time.perf_counter() - t0) / N


def pmr_batched(ksp, A, proxg, K, max_eigen, noise):
    """Replica variance over a stack of noise realizations, reconstructed under vmap."""
    recon = vmap(lambda n: ncs.fista_reconstruct(ksp + n, A, proxg, K, max_eigen))
    s1 = s2 = None
    for start in range(0, noise.shape[0], pmr_batched.B):
        r = recon(noise[start:start + pmr_batched.B]).to(torch.complex128)
        b1, b2 = r.sum(0), (r.abs() ** 2).sum(0)
        s1 = b1 if s1 is None else s1 + b1
        s2 = b2 if s2 is None else s2 + b2
    n = noise.shape[0]
    mean = s1 / n
    return ((s2 / n - mean.abs() ** 2) * n / (n - 1)).real


def pmr_sequential(ksp, A, proxg, K, max_eigen, noise):
    """The replica loop as run in the experiments: library FISTA, one at a time."""
    s1 = s2 = None
    for i in range(noise.shape[0]):
        AHb = A.adjoint(ksp + noise[i]) / (max_eigen ** 0.5)
        r = ORIGINAL_FISTA(lambda x: A.normal(x) / max_eigen, AHb, proxg, num_iters=K,
                           ptol_exit=-1.0, verbose=False).to(torch.complex128)
        s1 = r if s1 is None else s1 + r
        s2 = r.abs() ** 2 if s2 is None else s2 + r.abs() ** 2
    n = noise.shape[0]
    mean = s1 / n
    return ((s2 / n - mean.abs() ** 2) * n / (n - 1)).real


def draw_noise(ksp, n, sigma, seed):
    g = torch.Generator(device=ksp.device).manual_seed(seed)
    shape = (n,) + tuple(ksp.shape)
    re = torch.randn(shape, generator=g, device=ksp.device)
    im = torch.randn(shape, generator=g, device=ksp.device)
    return ((re + 1j * im) * sigma).to(ksp.dtype)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-file", default="experiments/fastmri_knee/data/file1000000.h5")
    ap.add_argument("--slice-idx", type=int, default=20)
    ap.add_argument("--R", type=int, default=2)
    ap.add_argument("--lamda", type=float, default=3e-7)
    ap.add_argument("--sigma0", type=float, default=1e-7)
    ap.add_argument("--num-iters", type=int, default=30)
    ap.add_argument("--tv-inner", type=int, default=50)
    ap.add_argument("--N", type=int, default=200, help="samples timed per configuration")
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 25, 50, 100, 200])
    ap.add_argument("--check-N", type=int, default=50, help="replicas for the batched-vs-sequential map check")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", default="experiments/fastmri_knee/results/cs_batched_timing/timing.json")
    args = ap.parse_args()

    ncs.FISTA_PTOL_EXIT = None   # fixed iteration count, as in the manuscript (--fista-ptol -1)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    with h5py.File(args.data_file, "r") as f:
        img = torch.tensor(f["reconstruction_rss"], device=dev)[args.slice_idx]
        mps = torch.tensor(f["jsense-12-cf=4"]["maps"], device=dev)[args.slice_idx, :, :, :, 0]
        ksp_full = torch.tensor(f["kspace"][:], device=dev)[args.slice_idx]
    C = mps.shape[-1]
    mps = rearrange(mps, "h w c -> c h w")
    if ksp_full.ndim == 3 and ksp_full.shape[-1] == C:
        ksp_full = rearrange(ksp_full, "h w c -> c h w")
    A, ksp, n_samp = build_operator(img, mps, ksp_full, dev, args.R)
    proxg = ComplexTV(tuple(img.shape[-2:]), args.lamda, max_iter=args.tv_inner, tol=0.0)
    _, max_eigen = power_method_operator(A.normal, torch.randn_like(img), verbose=False)
    max_eigen = float(max_eigen) * 1.01
    K, sigma = args.num_iters, args.sigma0
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"[cs-timing] device={gpu} R={args.R} K={K} lambda_TV={args.lamda} N={args.N} "
          f"batch sizes={args.batch_sizes}", flush=True)

    # 0. as-run baseline: PICO with the library's in-place gridding, and the
    #    same after swapping in the functional gridding, to show the swap does
    #    not change PICO's cost or map
    B0 = max(args.batch_sizes)
    t_pico_inplace = time_pico(ksp, A, proxg, K, max_eigen, args.N, B0)
    torch.manual_seed(args.seed + 9)
    v_inplace = ncs.jvp_hutch_variance(ksp, A, proxg, K, max_eigen, num_samples=B0, batch_size=B0)
    install_functional_gridding()
    t_pico_functional = time_pico(ksp, A, proxg, K, max_eigen, args.N, B0)
    torch.manual_seed(args.seed + 9)
    v_functional = ncs.jvp_hutch_variance(ksp, A, proxg, K, max_eigen, num_samples=B0, batch_size=B0)
    gridding_map_diff = float(torch.linalg.norm(v_functional - v_inplace) / torch.linalg.norm(v_inplace))
    print(f"[cs-timing] PICO batch {B0}: in-place gridding {t_pico_inplace:.4f} s, functional "
          f"{t_pico_functional:.4f} s; map difference {gridding_map_diff:.2e}", flush=True)

    # 1. the batched replica estimator is the same estimator as the sequential one
    batched_error = None
    agreement = None
    try:
        noise = draw_noise(ksp, args.check_N, sigma, args.seed + 1)
        pmr_batched.B = max(args.batch_sizes)
        v_b = pmr_batched(ksp, A, proxg, K, max_eigen, noise)
        v_s = pmr_sequential(ksp, A, proxg, K, max_eigen, noise)
        agreement = float(torch.linalg.norm(v_b - v_s) / torch.linalg.norm(v_s))
        print(f"[cs-timing] batched vs sequential PMR map, same {args.check_N} realizations: "
              f"relative difference {agreement:.2e}", flush=True)
    except RuntimeError as exc:
        batched_error = str(exc).splitlines()[0][:300]
        print(f"[cs-timing] batched PMR not available under vmap: {batched_error}", flush=True)
        torch.cuda.empty_cache()

    # 2. per-sample cost at matched batch size
    rows = {}
    seq_noise = draw_noise(ksp, args.N, sigma, args.seed + 2)
    pmr_sequential(ksp, A, proxg, K, max_eigen, seq_noise[:1])   # warm-up
    sync(); t0 = time.perf_counter()
    pmr_sequential(ksp, A, proxg, K, max_eigen, seq_noise)
    sync(); rows["PMR sequential (as run in the experiments)"] = (time.perf_counter() - t0) / args.N
    oom = {}
    for B in args.batch_sizes:
        N = max(args.N, B)
        try:
            rows[f"PICO batch {B}"] = time_pico(ksp, A, proxg, K, max_eigen, N, B)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            oom[f"PICO batch {B}"] = True; torch.cuda.empty_cache()
            print(f"[cs-timing] PICO batch {B}: out of memory", flush=True)
            continue
        if batched_error is None:
            try:
                pmr_batched.B = B
                bn = draw_noise(ksp, N, sigma, args.seed + 3 + B)
                pmr_batched(ksp, A, proxg, K, max_eigen, bn[:B])          # warm-up
                sync(); t0 = time.perf_counter()
                pmr_batched(ksp, A, proxg, K, max_eigen, bn)
                sync(); rows[f"PMR batch {B}"] = (time.perf_counter() - t0) / N
                del bn
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                oom[f"PMR batch {B}"] = True
                print(f"[cs-timing] PMR batch {B}: out of memory", flush=True)
        elif B == 1:
            rows["PMR batch 1"] = None   # filled from the sequential timing below
        torch.cuda.empty_cache()
        print(f"[cs-timing] B={B}: PICO {rows[f'PICO batch {B}']:.4f} s"
              + (f"  PMR {rows[f'PMR batch {B}']:.4f} s  ratio {rows[f'PICO batch {B}']/rows[f'PMR batch {B}']:.3f}"
                 if rows.get(f"PMR batch {B}") else ""), flush=True)
    seq = rows["PMR sequential (as run in the experiments)"]
    if rows.get("PMR batch 1") is None:
        rows["PMR batch 1"] = seq     # an unbatched replica loop is batch size 1
    print(f"[cs-timing] PMR sequential {seq:.4f} s", flush=True)

    out = {
        "device": gpu, "R": args.R, "slice": args.slice_idx, "lambda_tv": args.lamda,
        "fista_iterations": K, "tv_inner": args.tv_inner, "sigma0": sigma, "N_timed": args.N,
        "per_sample_seconds": rows,
        "cost_ratio_PICO_over_PMR_matched_batch": {
            str(B): rows[f"PICO batch {B}"] / rows[f"PMR batch {B}"]
            for B in args.batch_sizes if rows.get(f"PMR batch {B}")},
        "batched_pmr_error": batched_error,
        "out_of_memory": sorted(oom),
        "largest_batch_both_fit": max((B for B in args.batch_sizes
                                       if rows.get(f"PICO batch {B}") and rows.get(f"PMR batch {B}")), default=None),
        "cost_ratio_as_run_PICO_batched_over_PMR_sequential": {
            str(B): rows[f"PICO batch {B}"] / rows["PMR sequential (as run in the experiments)"]
            for B in args.batch_sizes},
        "batched_vs_sequential_pmr_map_relative_difference": agreement,
        "pico_seconds_inplace_gridding_as_run": t_pico_inplace,
        "pico_seconds_functional_gridding": t_pico_functional,
        "functional_vs_inplace_gridding_pico_map_relative_difference": gridding_map_diff,
        "note": ("all timings after the baseline use out-of-place gridding (index_add) so that "
                 "reconstructions can be vmapped; values are identical to the in-place version"),
        "batched_vs_sequential_pmr_check_N": args.check_N,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"[cs-timing] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
