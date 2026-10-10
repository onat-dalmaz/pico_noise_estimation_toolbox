import torch
import time

from tqdm import tqdm
from typing import Optional
from mr_recon.dtypes import complex_dtype
from mr_recon.linops import linop
from mr_recon.utils import np_to_torch, torch_to_np
from mr_recon.algs import (
    density_compensation, 
    conjugate_gradient, 
    power_method_operator, 
    gradient_descent,
    FISTA
)

def min_norm_recon(A: linop,
                   ksp: torch.Tensor,
                   max_iter: int = 15,
                   lamda_l2: Optional[float] = 0.0,
                   max_eigen: Optional[float] = None,
                   verbose: Optional[bool] = True) -> torch.Tensor:
    """
    Run min norm recon:
    recon = A^H(AA^H)^-1b
    
    Parameters:
    -----------
    A : linop
        The linear operator (see linop)
    ksp : torch.Tensor
        k-space data with shape (nc, ...)
    max_iter : int
        max number of iterations for recon algorithm
    lamda_l2 : float
        l2 lamda regularization
    max_eigen : float
        maximum eigenvalue of AHA
    verbose : bool 
        Toggles print statements

    Returns:
    --------
    recon : torch.Tensor
        the reconstructed image/volume
    """

    # Consts
    device = ksp.device

    # Estimate largest eigenvalue so that lambda max of AHA is 1
    if max_eigen is None:
        x0 = torch.randn(A.ishape, dtype=complex_dtype, device=device)
        _, max_eigen = power_method_operator(A.normal, x0, verbose=verbose)
        max_eigen *= 1.01

    # Wrap normal with max eigen
    AAH = lambda x : A.forward(A.adjoint(x)) / max_eigen

    # Run CG
    y = conjugate_gradient(AHA=AAH, 
                           AHb=ksp.type(complex_dtype),
                           lamda_l2=lamda_l2,
                           num_iters=max_iter,
                           verbose=verbose)
    
    # Apply adjoint 
    start = time.perf_counter()
    recon = A.adjoint(y) / (max_eigen ** 0.5)
    end = time.perf_counter()
    if verbose:
        print(f'AHy took {end-start:.3f}(s)')
    
    return recon

def CG_SENSE_recon(A: linop,
                   ksp: torch.Tensor,
                   max_iter: Optional[int] = 15,
                   lamda_l2: Optional[float] = 0.0,
                   max_eigen: Optional[float] = None,
                   tolerance: Optional[float] = 1e-8,
                   tolerance_mode: Optional[str] = 'abs',
                   P: Optional[callable] = None,
                   dcf_gridding_init: Optional[bool] = False,
                   verbose: Optional[bool] = True) -> torch.Tensor:
    """
    Run CG SENSE recon:
    recon = (AHA + lamda_l2I)^-1 AHb
    
    Parameters:
    -----------
    A : linop
        The linear operator (see linop)
    ksp : torch.Tensor
        k-space data with shape (nc, ...)
    max_iter : int
        max number of iterations for recon algorithm
    lamda_l2 : float
        l2 lamda regularization for SENSE: ||Ax - b||_2^2 + lamda_l2||x||_2^2
    max_eigen : float
        maximum eigenvalue of AHA
    tolerance : float
        tolerance for CG algorithm
    P : callable, optional
        SPD preconditioner passed through to conjugate_gradient. The converged
        solution is invariant to P; it only affects the convergence rate.
    dcf_gridding_init : bool
        If True and A exposes ``dcf_gridding_adjoint`` (a sense_linop with the physical
        DCF retained), initialise CG from the density-compensated gridding image
        ``A^H D b`` instead of ``A^H b``. This accelerates convergence of the clean
        (unweighted) normal equations WITHOUT changing the converged solution/noise,
        i.e. it uses the DCF purely as an initialiser. No-op if A has no DCF.
    verbose : bool 
        Toggles print statements

    Returns:
    --------
    recon : torch.Tensor
        the reconstructed image/volume
    """

    # Consts
    device = ksp.device

    # Estimate largest eigenvalue so that lambda max of AHA is 1
    if max_eigen is None:
        x0 = torch.randn(A.ishape, dtype=complex_dtype, device=device)
        _, max_eigen = power_method_operator(A.normal, x0, verbose=verbose)
        max_eigen *= 1.01
    
    # Starting with AHb
    start = time.perf_counter()
    y = ksp.type(complex_dtype)
    AHb = A.adjoint(y) / (max_eigen ** 0.5)

    # Optional DCF gridding initial guess (A^H D b). Uses the physical DCF only to seed
    # CG; the fixed point of (A^H A + lamda I) is init-independent so noise is unchanged.
    x0 = None
    if dcf_gridding_init and hasattr(A, 'dcf_gridding_adjoint'):
        x0 = A.dcf_gridding_adjoint(y) * (max_eigen ** 0.5)
    end = time.perf_counter()
    if verbose:
        print(f'AHb took {end-start:.3f}(s)')
    if max_iter == 0:
        return AHb if x0 is None else x0

    # Clear data (we dont need it anymore)
    y = y.cpu()
    with device:
        torch.cuda.empty_cache()

    # Wrap normal with max eigen
    AHA = lambda x : A.normal(x) / max_eigen

    # Run CG
    recon = conjugate_gradient(AHA=AHA, 
                               AHb=AHb,
                               P=P,
                               x0=x0,
                               num_iters=max_iter,
                               lamda_l2=lamda_l2,
                               tolerance=tolerance,
                               tolerance_mode=tolerance_mode,
                               verbose=verbose)
    
    return recon

def CG_SENSE_recon_adjoint(A: linop,
                           img_vec: torch.Tensor,
                           max_iter: Optional[int] = 15,
                           lamda_l2: Optional[float] = 0.0,
                           max_eigen: Optional[float] = None,
                           tolerance: Optional[float] = 1e-8,
                           verbose: Optional[bool] = True) -> torch.Tensor:
    """
    Apply the Hermitian conjugate of the CG-SENSE reconstruction operator R^H:
    output = R^H @ img_vec = A @ (AHA + lamda_l2*I)^-1 @ img_vec
    
    Parameters:
    -----------
    A : linop
        The linear operator (see linop)
    img_vec : torch.Tensor
        Input image-space vector with shape A.ishape
    max_iter : int
        Max number of iterations for CG algorithm
    lamda_l2 : float
        L2 lambda regularization for SENSE
    max_eigen : float
        Maximum eigenvalue of AHA. If None, will be estimated.
    tolerance : float
        Tolerance for CG algorithm
    verbose : bool 
        Toggles print statements

    Returns:
    --------
    ksp_vec : torch.Tensor
        Resulting k-space like vector with shape A.oshape
    """

    # Consts
    device = img_vec.device

    # Ensure input vector has correct dtype
    v = img_vec.type(complex_dtype)

    # Estimate largest eigenvalue if not provided
    if max_eigen is None:
        if verbose:
            print(f"Estimating maximum eigenvalue with power method...")
        x0 = torch.randn(A.ishape, dtype=complex_dtype, device=device)
        _, max_eigen = power_method_operator(A.normal, x0, verbose=verbose)
        max_eigen *= 1.01
        if verbose:
            print(f"Maximum eigenvalue: {max_eigen:.4e}")

    # Wrap normal operator with max eigenvalue for CG
    AHA_normalized = lambda x : A.normal(x) / max_eigen

    # Scale the input vector for the CG solve
    scaled_v = v / (max_eigen ** 0.5)

    # --- Step 1: Solve (AHA + lambda*I)z = v --- 
    if verbose:
        print(f"Running CG solve for R.H step 1...")
    z = conjugate_gradient(AHA=AHA_normalized,
                           AHb=scaled_v,
                           num_iters=max_iter,
                           lamda_l2=lamda_l2, # Matches CG_SENSE_recon
                           tolerance=tolerance,
                           verbose=verbose)
    
    # --- Step 2: Apply A (Forward operator) --- 
    if verbose:
        print(f"Applying forward operator A for R.H step 2...")
    ksp_vec = A.forward(z)
    
    return ksp_vec

def coil_combine(multi_chan_img: torch.Tensor,
                 mps: Optional[torch.Tensor] = None,
                 walsh_kernel_size: Optional[int] = None) -> torch.Tensor:
    """
    Combine multi-channel images using SENSE, walsh, or SoS

    Parameters:
    -----------
    multi_chan_img : torch.Tensor
        multi-channel image with shape (nc, ...)
    mps : torch.Tensor
        coil sensitivity maps with shape (nc, ...)
    walsh_kernel_size : int
        size of walsh kernel for walsh coil combination
    
    Returns:
    --------
    img_comb : torch.Tensor
        the combined image/volume with shape (...)
    """

    if mps is not None:
        img_comb = (multi_chan_img * mps.conj()).sum(0) / (mps.abs().square().sum(0) + 1e-5)
    elif walsh_kernel_size is not None:
        # Reshape image into blocks 
        raise NotImplementedError
    else:
        img_comb = multi_chan_img.abs().square().sum(0).sqrt()
    
    return img_comb

def SPIRIT_recon(A: linop,
                 ksp: torch.Tensor,
                 ksp_cal: torch.Tensor,
                 max_iter: Optional[int] = 15,
                 lamda_l2: Optional[float] = 0.0,
                 max_eigen: Optional[float] = None,
                 verbose: Optional[bool] = True) -> torch.Tensor:
    """
    Run SPIRIT recon:
    recon = (AHA + lamda_l2I)^-1 AHb


    
    """
    return None

def FISTA_recon(A: linop,
                ksp: torch.Tensor,
                proxg: callable,
                max_iter: int = 40,
                max_eigen: Optional[float] = None,
                verbose: Optional[bool] = True) -> torch.Tensor:
    """
    Run FISTA recon
    recon = min_x ||Ax - b||_2^2 + g(x)
    
    Parameters:
    -----------
    A : linop
        The linear operator (see linop)
    ksp : torch.Tensor
        k-space data with shape (nc, nro, npe, ntr)
    proxg : callable
        proximal operator for g(x)
    max_iter : int
        max number of iterations for recon algorithm
    max_eigen : float
        maximum eigenvalue of AHA
    verbose : bool 
        Toggles print statements

    Returns:
    --------
    recon : torch.Tensor
        the reconstructed image/volume
    """

    # Consts
    device = ksp.device

    # Estimate largest eigenvalue so that lambda max of AHA is 1
    if max_eigen is None:
        x0 = torch.randn(A.ishape, dtype=complex_dtype, device=device)
        _, max_eigen = power_method_operator(A.normal, x0, verbose=verbose)
        max_eigen *= 1.01
    
    # Starting with AHb
    start = time.perf_counter()
    y = ksp.type(complex_dtype)
    AHb = A.adjoint(y) / (max_eigen ** 0.5)
    end = time.perf_counter()
    if verbose:
        print(f'AHb took {end-start:.3f}(s)')

    # Clear data (we dont need it anymore)
    y = y.cpu()
    with device:
        torch.cuda.empty_cache()

    # Wrap normal with max eigen
    AHA = lambda x : A.normal(x) / max_eigen

    # Run FISTA
    recon = FISTA(AHA, AHb, proxg, max_iter)

    return recon

def doubleCG_inv_op_builder(A: linop,
                             dcf: Optional[torch.Tensor] = None,
                             max_iter: Optional[int] = 15,
                             lamda_l2: Optional[float] = 0.0,
                             max_eigen: Optional[float] = None,
                             tolerance: Optional[float] = 1e-8,
                             tolerance_mode: Optional[str] = 'abs',
                             P: Optional[callable] = None,
                             dcf_precond: Optional[bool] = False,
                             verbose: Optional[bool] = True) -> callable:
    """
    Build the reconstruction noise-covariance operator that PICO probes, using
    double conjugate gradient solves with max eigenvalue normalization.

    For a linear recon x_hat = (A^H W A + lamda I)^-1 A^H W b with white k-space noise
    (cov = sigma^2 I), the covariance is the sandwich
        Sigma = (A^H W A + lamda I)^-1 (A^H W^2 A) (A^H W A + lamda I)^-1,
    where W is the data-consistency weighting baked into ``A`` (``A.dcf``). The two CG
    solves apply the outer inverses and the middle step applies A^H W^2 A.

    IMPORTANT (clean vs DCF-weighted, R2.6a): pass ``dcf`` equal to the linop's
    data-consistency weighting so the middle step is correct:
      * CLEAN operator: build A with ``weight_data_consistency=False`` (so ``A.dcf`` is
        ones and ``A.normal`` = A^H A) and pass ``dcf=None`` -> middle = A^H A, giving the
        OLS/Tikhonov covariance (A^H A + lamda I)^-1 (A^H A) (A^H A + lamda I)^-1 (g >= 1).
      * DCF-WEIGHTED operator (legacy / SI diagnostic): build A with the physical DCF and
        pass ``dcf=W`` -> middle = A^H W^2 A, giving Sigma_W (reproduces sub-unity g).

    Parameters:
    -----------
    A : linop
        The linear operator (see linop)
    dcf : torch.Tensor, optional
        The data-consistency weighting to insert in the middle A^H (dcf) A step. Should
        match ``A.dcf``. If None, the middle step is the unweighted A^H A.
    max_iter : int
        Max number of iterations for each CG solve
    lamda_l2 : float
        L2 lambda regularization
    max_eigen : float
        Maximum eigenvalue of AHA. If None, will be estimated using power method
    tolerance : float
        Tolerance for CG algorithm
    P : callable, optional
        SPD preconditioner passed through to the inner CG solves. Does not change the
        converged operator, only convergence speed. If given, it overrides ``dcf_precond``.
    dcf_precond : bool
        If True and ``P`` is None, build a DCF-based SPD preconditioner from the linop's
        physical DCF and pass it to BOTH inner CG solves. This is the covariance-solve
        analogue of the recon path's DCF gridding initialiser (``dcf_gridding_adjoint``):
        it uses the SAME density-compensation principle to speed convergence of the clean
        (unweighted, ill-conditioned) non-Cartesian normal equations WITHOUT changing the
        converged solution / probed covariance (SPD preconditioners preserve the fixed
        point). No-op if A lacks the DCF hooks (``dcf_true`` / ``_dcf_gridding_scale``).
    verbose : bool
        Toggle print statements
    
    Returns:
    --------
    AHA_inv : callable
        A function that applies (AHA)^-1 to its input
    """
    # Constants
    device = A.torch_dev

    # Estimate largest eigenvalue if not provided
    if max_eigen is None:
        if verbose:
            print(f"Estimating maximum eigenvalue with power method...")
        x0 = torch.randn(A.ishape, dtype=complex_dtype, device=device)
        _, max_eigen = power_method_operator(A.normal, x0, verbose=verbose)
        max_eigen *= 1.01  # Add 1% for safety, like in CG_SENSE_recon
        if verbose:
            print(f"Maximum eigenvalue: {max_eigen:.4e}")

    # Optional DCF-based preconditioner for the inner CG solves (the covariance-solve
    # analogue of the recon path's DCF gridding initialiser). For the CLEAN non-Cartesian
    # normal equations the ill-conditioning comes from the non-uniform k-space sampling
    # density, which the physical DCF D compensates. The gridding operator G = A^H (D/c)
    # (c from A._dcf_gridding_scale, so A^H (D/c) A ~ I) is an approximate left-inverse of
    # A, hence
    #     P = max_eigen * G G^H = max_eigen * A^H (D/c)^2 A  ~  (A^H A / max_eigen)^-1,
    # i.e. the inverse of the exact operator each inner CG runs on. P is SPD (it equals
    # (D A / c)^H (D A / c) times max_eigen > 0), so by PCG theory the converged solution
    # -- and therefore the probed covariance / g-factor -- is UNCHANGED; only the spectrum
    # is flattened, cutting the iteration count. Using the SAME DCF principle as the recon
    # path keeps the PICO-vs-PMR comparison fair.
    if dcf_precond and P is None:
        if hasattr(A, 'dcf_true') and hasattr(A, 'dcf_gridding_adjoint') and hasattr(A, '_dcf_gridding_scale'):
            c = A._dcf_gridding_scale()
            precond_scale = max_eigen / (c ** 2)
            d_true = A.dcf_true
            def _dcf_preconditioner(r: torch.Tensor) -> torch.Tensor:
                # P r = (max_eigen / c^2) * A^H D^2 A r  (D = physical DCF). The physical DCF
                # is applied twice (once explicitly, once inside the unweighted gridding
                # adjoint) so this is independent of A's data-consistency weighting.
                weighted = A.forward(r) * d_true[None, ...]
                return precond_scale * A.dcf_gridding_adjoint(weighted, normalize=False)
            P = _dcf_preconditioner
            if verbose:
                print(f"doubleCG: DCF preconditioner enabled (c={c:.4e}, scale={precond_scale:.4e})")
        elif verbose:
            print("doubleCG: dcf_precond=True but A lacks DCF hooks; using identity preconditioner")

    # Create normalized operator for CG
    AHA_normalized = lambda x: A.normal(x) / max_eigen
    
    # Define the inverse gram operator function
    def AHA_inv(x: torch.Tensor) -> torch.Tensor:
        """
        Apply (AHA)^-1 to input x using double CG with eigenvalue normalization
        """
        # First CG solve
        scaled_x = x / (max_eigen ** 0.5)
        ret = conjugate_gradient(
            AHA=AHA_normalized,
            AHb=scaled_x,
            P=P,
            num_iters=max_iter,
            lamda_l2=lamda_l2,
            tolerance=tolerance,
            tolerance_mode=tolerance_mode,
            verbose=verbose
        )
        
        
        # Apply A^H DCF A in the middle
        if dcf is not None:
            # Apply DCF weighting
            ret = A.adjoint(A.forward(ret) * dcf)
        else:
            # No DCF, just apply A^H A
            ret = A.adjoint(A.forward(ret))
        
        # Second CG solve
        scaled_ret = ret / (max_eigen ** 0.5)
        ret = conjugate_gradient(
            AHA=AHA_normalized,
            AHb=scaled_ret,
            P=P,
            num_iters=max_iter,
            lamda_l2=lamda_l2,
            tolerance=tolerance,
            tolerance_mode=tolerance_mode,
            verbose=verbose
        )
        
        return ret
    
    return AHA_inv
