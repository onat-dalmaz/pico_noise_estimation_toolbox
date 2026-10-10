"""
Pseudo-Multiple-Replica (PMR) variance accumulation helpers.

Modes
-----
  * ``welford``     -- voxelwise mean and variance across reconstructions of
                       ``ksp + noise`` (Robson et al.), accumulated online.
  * ``two_pass``    -- the same quantity in two passes (mean, then centered
                       squared magnitude).
  * ``noise_only``  -- ``E[|R(noise)|^2]``, which equals the above for a linear
                       reconstruction, where ``R(ksp + n) - R(ksp) = R(n)``.

All modes accumulate in float64 over the reconstruction dtype and return the
variance map divided by the k-space noise variance.
"""
from typing import Callable, Iterable, Optional, List, Dict

import torch


# ---------------------------------------------------------------------------
# Core accumulators (float64)
# ---------------------------------------------------------------------------
class WelfordComplexVariance:
    """Online (one-pass) Welford mean/variance for complex tensors, in float64.

    For a stream of complex samples x_n, tracks the running mean and the sum of
    squared magnitude deviations M2 = sum_n |x_n - mean_n|^2. The population
    variance E[|x - E x|^2] = M2 / n (this is Var(Re) + Var(Im), the total
    power convention used by ``torch.var`` on complex input).
    """

    def __init__(self, shape, device, complex_dtype=torch.complex128,
                 real_dtype=torch.float64):
        self.n = 0
        self._cdt = complex_dtype
        self._rdt = real_dtype
        self.mean = torch.zeros(shape, dtype=complex_dtype, device=device)
        self.M2 = torch.zeros(shape, dtype=real_dtype, device=device)

    def update(self, x: torch.Tensor):
        x = x.to(self._cdt)
        self.n += 1
        delta = x - self.mean
        self.mean = self.mean + delta / self.n
        delta2 = x - self.mean
        # |delta| * |delta2| with the correct sign via real(conj(delta)*delta2)
        self.M2 = self.M2 + (delta.conj() * delta2).real

    @property
    def variance(self) -> torch.Tensor:
        """Population variance E[|x-mean|^2] (float64). Returns zeros if n==0."""
        if self.n == 0:
            return self.M2.clone()
        return self.M2 / self.n

    def variance_at(self, unbiased: bool = False) -> torch.Tensor:
        if self.n <= 1:
            return self.M2 * 0.0
        denom = (self.n - 1) if unbiased else self.n
        return self.M2 / denom


def _draw_ksp_noise(ksp: torch.Tensor, noise_std: float) -> torch.Tensor:
    """i.i.d. complex/real Gaussian noise matching ksp dtype, std = noise_std.

    Matches the legacy convention ``(noise_var**0.5) * torch.randn_like(ksp)``:
    for a complex ksp, ``torch.randn_like`` draws complex noise whose real and
    imaginary parts are each N(0, 0.5) so that E|z|^2 = 1 (times noise_var).
    """
    return noise_std * torch.randn_like(ksp)


# ---------------------------------------------------------------------------
# Incremental stable PMR variance (yields at each N in N_values)
# ---------------------------------------------------------------------------
def incremental_variance_PMR_stable(R: Callable,
                                    ksp: torch.Tensor,
                                    noise_var: float,
                                    N_values: Iterable[int],
                                    mode: str = 'noise_only',
                                    generator: Optional[torch.Generator] = None,
                                    verbose: bool = False):
    """Yield the stable PMR variance map at each N in ``N_values`` (sorted asc).

    Reuses replicas across N (like ``incremental_calc_variance_PMR``) so the
    curve at N=[100,500,...,30000] costs one pass to the largest N.

    Parameters
    ----------
    R : callable   k-space -> image reconstruction operator.
    ksp : Tensor   the (real) k-space; only its shape/dtype/device are used for
                   ``noise_only`` (noise is injected on zeros).
    noise_var : float   pseudo-noise variance (per complex k-space sample).
    N_values : iterable[int]
    mode : {'noise_only','baseline_subtracted','welford','two_pass','naive_f32'}
    generator : optional torch.Generator for reproducibility (same device).

    Yields
    ------
    (N, var_map_float64) for each N in sorted(N_values).
    """
    from tqdm import tqdm

    mode = mode.lower()
    noise_std = float(noise_var) ** 0.5
    sorted_N = sorted(int(n) for n in N_values)
    device = ksp.device

    if mode in ('noise_only', 'baseline_subtracted'):
        # Accumulate S = sum |y_n|^2 in float64, where y_n = R(noise) (noise_only)
        # or y_n = R(ksp+noise) - R(ksp) (baseline_subtracted). var = S/(N*noise_var).
        ksp_zero = torch.zeros_like(ksp)
        base = None
        if mode == 'baseline_subtracted':
            base = R(ksp).to(torch.complex128)
        running = None
        cum = 0
        total = sorted_N[-1]
        pbar = tqdm(total=total, desc=f'PMR[{mode}]', disable=not verbose)
        for n_target in sorted_N:
            for _ in range(n_target - cum):
                noise = noise_std * torch.randn(ksp.shape, dtype=ksp.dtype,
                                                device=device, generator=generator)
                if mode == 'noise_only':
                    y = R(ksp_zero + noise).to(torch.complex128)
                else:
                    y = R(ksp + noise).to(torch.complex128) - base
                sq = (y.conj() * y).real  # |y|^2, float64
                running = sq if running is None else running + sq
                pbar.update(1)
            cum = n_target
            yield n_target, running / (cum * noise_var)
        pbar.close()

    elif mode in ('welford', 'two_pass'):
        # Replica-based variance of x_n = R(ksp+noise). For nonlinear recon.
        if mode == 'welford':
            acc = None
            cum = 0
            total = sorted_N[-1]
            pbar = tqdm(total=total, desc='PMR[welford]', disable=not verbose)
            for n_target in sorted_N:
                for _ in range(n_target - cum):
                    noise = noise_std * torch.randn(ksp.shape, dtype=ksp.dtype,
                                                    device=device, generator=generator)
                    x = R(ksp + noise)
                    if acc is None:
                        acc = WelfordComplexVariance(tuple(x.shape), device)
                    acc.update(x)
                    pbar.update(1)
                cum = n_target
                yield n_target, acc.variance / noise_var
            pbar.close()
        else:  # two_pass: store nothing large; do two passes per checkpoint is
            # expensive, so we buffer replicas in float32 and recompute in f64.
            # For memory safety we cap by streaming: first pass mean, second var.
            # Here we implement an exact two-pass over a growing replica buffer.
            raise NotImplementedError(
                "two_pass is provided via calc_variance_PMR_stable(mode='two_pass'); "
                "use 'welford' for the incremental (multi-N) path.")

    elif mode == 'naive_f32':
        # Legacy buggy estimator (float32 torch.var of R(ksp+noise)); reproduces
        # the cancellation. Requires storing replicas -> only for small N / toy.
        recons = []
        cum = 0
        for n_target in sorted_N:
            for _ in range(n_target - cum):
                noise = noise_std * torch.randn(ksp.shape, dtype=ksp.dtype,
                                                device=device, generator=generator)
                recons.append(R(ksp + noise))
            cum = n_target
            stacked = torch.stack(recons, dim=0)
            var = torch.var(stacked, dim=0, unbiased=False) / noise_var
            yield n_target, var
    else:
        raise ValueError(f"Unknown mode: {mode}")


def calc_variance_PMR_stable(R: Callable,
                             ksp: torch.Tensor,
                             noise_var: float = 1e-4,
                             n_replicas: int = 100,
                             mode: str = 'noise_only',
                             generator: Optional[torch.Generator] = None,
                             verbose: bool = False) -> torch.Tensor:
    """Numerically stable PMR variance map at a single N (=n_replicas).

    Convention: returns ``Var[R(ksp+noise)] / noise_var`` (same as the legacy
    ``calc_variance_PMR``), computed in float64.

    mode:
      'noise_only'          - E[|R(noise)|^2]/noise_var (LINEAR recon; exact).
      'baseline_subtracted' - E[|R(ksp+noise)-R(ksp)|^2]/noise_var (linear check).
      'welford'             - Welford f64 variance of replicas (NONLINEAR).
      'two_pass'            - two-pass f64 variance of replicas (NONLINEAR).
      'naive_f32'           - legacy float32 torch.var (bug reproduction only).

    Returns a float64 tensor (cast as needed by the caller).
    """
    mode = mode.lower()
    noise_std = float(noise_var) ** 0.5
    device = ksp.device

    if mode == 'two_pass':
        # Exact two-pass in float64 (nonlinear). Pass 1: mean; pass 2: centered SS.
        from tqdm import tqdm
        g = generator
        mean = None
        # Pass 1
        for _ in tqdm(range(n_replicas), desc='PMR two-pass (mean)', disable=not verbose):
            noise = noise_std * torch.randn(ksp.shape, dtype=ksp.dtype, device=device, generator=g)
            x = R(ksp + noise).to(torch.complex128)
            mean = x if mean is None else mean + x
        mean = mean / n_replicas
        # Pass 2 (fresh noise -> unbiased population variance estimate)
        var = torch.zeros(mean.shape, dtype=torch.float64, device=device)
        for _ in tqdm(range(n_replicas), desc='PMR two-pass (var)', disable=not verbose):
            noise = noise_std * torch.randn(ksp.shape, dtype=ksp.dtype, device=device, generator=g)
            x = R(ksp + noise).to(torch.complex128)
            c = x - mean
            var = var + (c.conj() * c).real
        return var / n_replicas / noise_var

    # All other modes go through the incremental generator with a single N.
    last = None
    for _, v in incremental_variance_PMR_stable(R, ksp, noise_var, [n_replicas],
                                                mode=mode, generator=generator,
                                                verbose=verbose):
        last = v
    return last


def gfactor_PMR_stable(R_ref: Callable,
                       R_acc: Callable,
                       ksp_ref: torch.Tensor,
                       ksp_acc: torch.Tensor,
                       noise_var: float = 1e-4,
                       n_replicas: int = 100,
                       mode: str = 'noise_only',
                       eps: float = 1e-12,
                       verbose: bool = False) -> torch.Tensor:
    """g-factor (unnormalized by sqrt(R)) from stable PMR variances.

    g = sqrt( Var_acc / Var_ref ). Caller divides by sqrt(R) as usual.
    """
    var_ref = calc_variance_PMR_stable(R_ref, ksp_ref, noise_var, n_replicas, mode, verbose=verbose)
    var_acc = calc_variance_PMR_stable(R_acc, ksp_acc, noise_var, n_replicas, mode, verbose=verbose)
    ratio = var_acc / (var_ref + eps)
    return torch.sqrt(torch.clamp(ratio, min=0.0))
