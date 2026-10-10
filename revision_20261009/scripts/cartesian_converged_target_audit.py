#!/usr/bin/env python
"""Residual-converged Cartesian Figure-2 target audit.

This script owns audit-only outputs under
``experiments/cartesian_knee/results/converged_target_audit``.  It never writes
manuscript text or final figure assets.

The primary mode uses the exact Figure-2 data, slice, SENSE operators, R=2,
lambda=0, sigma=0.05, and max-eigen normalization constant 2.  PMR and PICO
both call the same CG implementation, start from zero, and stop on the *true*
normal-equation relative residual.  ``current25`` mode reproduces the two
finite-25 solver paths used by the current Figure-2 generator for a paired
target-sensitivity diagnostic.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Callable, Iterable

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mr_recon.dtypes import complex_dtype, real_dtype
from mr_recon.fourier import gridded_nufft
from mr_recon.gfactor import gfactor_sense
from mr_recon.linops import batching_params, sense_linop
from mr_recon.utils import gen_grd


DEFAULT_DATA = (
    REPO_ROOT
    / "experiments"
    / "cartesian_knee"
    / "data"
    / "efa383b6-9446-438a-9901-1fe951653dbd.h5"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "experiments"
    / "cartesian_knee"
    / "results"
    / "converged_target_audit"
)
DISPLAY_N = (10, 20, 40, 80, 200)


def cg_solve(
    operator: Callable[[torch.Tensor], torch.Tensor],
    rhs: torch.Tensor,
    *,
    tolerance: float,
    max_iterations: int,
    initialization: str = "zero",
    stop_mode: str = "true_residual",
) -> tuple[torch.Tensor, dict[str, object]]:
    """Solve an HPD system and report the true normal-equation residual.

    ``stop_mode='true_residual'`` checks ``||b-Hx||/||b||`` after every update.
    ``stop_mode='fixed'`` runs exactly ``max_iterations`` updates and reports the
    same true residual without using it to stop.
    """

    if initialization not in {"zero", "rhs"}:
        raise ValueError(f"Unsupported initialization: {initialization}")
    if stop_mode not in {"true_residual", "fixed"}:
        raise ValueError(f"Unsupported stop mode: {stop_mode}")
    if max_iterations < 0:
        raise ValueError("max_iterations must be nonnegative")

    x = torch.zeros_like(rhs) if initialization == "zero" else rhs.clone()
    rhs_norm = float(torch.linalg.vector_norm(rhs).item())
    scale = rhs_norm if rhs_norm > 0.0 else 1.0
    true_r = rhs - operator(x)
    true_rel = float(torch.linalg.vector_norm(true_r).item() / scale)
    if rhs_norm == 0.0 or (
        stop_mode == "true_residual" and true_rel <= tolerance
    ):
        return x, {
            "iterations": 0,
            "converged": True,
            "true_relative_residual": true_rel,
            "true_residual_norm": true_rel * scale,
            "rhs_norm": rhs_norm,
            "initialization": initialization,
            "stop_mode": stop_mode,
            "breakdown": False,
        }

    r = true_r
    p = r.clone()
    rr = torch.real(torch.sum(r.conj() * r))
    converged = False
    breakdown = False
    iterations = 0

    for index in range(max_iterations):
        hp = operator(p)
        denominator = torch.real(torch.sum(p.conj() * hp))
        denominator_value = float(denominator.item())
        rr_value = float(rr.item())
        if (
            not math.isfinite(denominator_value)
            or not math.isfinite(rr_value)
            or denominator_value <= 0.0
            or rr_value < 0.0
        ):
            breakdown = True
            break

        alpha = rr / denominator
        x = x + alpha * p
        r = r - alpha * hp
        iterations = index + 1

        # This is deliberately an extra operator application: stopping is based
        # on the true residual, not the recursively updated residual.
        true_r = rhs - operator(x)
        true_rel = float(torch.linalg.vector_norm(true_r).item() / scale)
        if stop_mode == "true_residual" and true_rel <= tolerance:
            converged = True
            break

        rr_new = torch.real(torch.sum(r.conj() * r))
        rr_new_value = float(rr_new.item())
        if not math.isfinite(rr_new_value) or rr_new_value < 0.0:
            breakdown = True
            break
        if rr_value == 0.0:
            breakdown = True
            break
        beta = rr_new / rr
        p = r + beta * p
        rr = rr_new

    true_r = rhs - operator(x)
    true_norm = float(torch.linalg.vector_norm(true_r).item())
    true_rel = true_norm / scale
    if stop_mode == "fixed":
        converged = true_rel <= tolerance

    return x, {
        "iterations": iterations,
        "converged": bool(converged),
        "true_relative_residual": true_rel,
        "true_residual_norm": true_norm,
        "rhs_norm": rhs_norm,
        "initialization": initialization,
        "stop_mode": stop_mode,
        "breakdown": bool(breakdown),
    }


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
    os.replace(temporary, path)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _nrmse(
    estimate: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
) -> float:
    if mask is not None:
        estimate = estimate[mask]
        target = target[mask]
    finite = torch.isfinite(estimate) & torch.isfinite(target)
    estimate = estimate[finite].to(torch.float64)
    target = target[finite].to(torch.float64)
    return float(
        (
            torch.linalg.vector_norm(estimate - target)
            / (torch.linalg.vector_norm(target) + 1e-30)
        ).item()
    )


def _corr(
    estimate: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
) -> float:
    if mask is not None:
        estimate = estimate[mask]
        target = target[mask]
    finite = torch.isfinite(estimate) & torch.isfinite(target)
    estimate = estimate[finite].to(torch.float64)
    target = target[finite].to(torch.float64)
    estimate = estimate - estimate.mean()
    target = target - target.mean()
    return float(
        (
            torch.sum(estimate * target)
            / (
                torch.linalg.vector_norm(estimate)
                * torch.linalg.vector_norm(target)
                + 1e-30
            )
        ).item()
    )


def _g_from_diagonals(
    accelerated: torch.Tensor, reference: torch.Tensor, acceleration: int
) -> torch.Tensor:
    ratio = torch.clamp(accelerated, min=0.0) / torch.clamp(
        reference, min=1e-30
    )
    return torch.sqrt(torch.clamp(ratio, min=0.0)) / math.sqrt(acceleration)


def _record_solver(
    rows: list[dict[str, object]],
    *,
    method: str,
    arm: str,
    sample: int,
    info: dict[str, object],
) -> None:
    rows.append({"method": method, "arm": arm, "sample": sample, **info})


def _plot_candidate(
    arrays: dict[str, np.ndarray], output_dir: Path, solver_label: str
) -> None:
    required = [
        *(f"pico_g_N{count}" for count in DISPLAY_N),
        *(f"pmr_g_N{count}" for count in DISPLAY_N),
        "g_analytical",
        "magnitude",
    ]
    if any(key not in arrays for key in required):
        return

    cmap = matplotlib.colormaps["turbo"].copy()
    cmap.set_bad("black")
    figure = plt.figure(figsize=(17.8, 7.1), facecolor="black")
    grid = figure.add_gridspec(
        2,
        7,
        width_ratios=[1, 1, 1, 1, 1, 0.055, 1],
        left=0.045,
        right=0.985,
        bottom=0.055,
        top=0.855,
        wspace=0.045,
        hspace=0.10,
    )
    image_artist = None

    def reciprocal(values: np.ndarray) -> np.ndarray:
        result = np.full(values.shape, np.nan, dtype=np.float64)
        np.divide(1.0, values, out=result, where=np.isfinite(values) & (values > 0))
        return result

    for row, method in enumerate(("pico", "pmr")):
        for column, count in enumerate(DISPLAY_N):
            axis = figure.add_subplot(grid[row, column])
            image_artist = axis.imshow(
                reciprocal(arrays[f"{method}_g_N{count}"]),
                cmap=cmap,
                vmin=0.30,
                vmax=1.00,
            )
            axis.set_title(f"N={count}", color="white", fontsize=10)
            axis.axis("off")
            if column == 0:
                axis.text(
                    -0.08,
                    0.5,
                    "PICO" if method == "pico" else "PMR",
                    color="white",
                    rotation=90,
                    va="center",
                    ha="right",
                    transform=axis.transAxes,
                    fontsize=12,
                    weight="bold",
                )

    color_axis = figure.add_subplot(grid[:, 5])
    if image_artist is not None:
        colorbar = figure.colorbar(image_artist, cax=color_axis)
        colorbar.set_label("Reciprocal g-factor (1/g)", color="white")
        colorbar.ax.tick_params(colors="white")

    reference_axis = figure.add_subplot(grid[0, 6])
    reference_axis.imshow(
        reciprocal(arrays["g_analytical"]), cmap=cmap, vmin=0.30, vmax=1.00
    )
    reference_axis.set_title("Analytical SENSE", color="white", fontsize=10)
    reference_axis.axis("off")
    magnitude_axis = figure.add_subplot(grid[1, 6])
    magnitude_axis.imshow(arrays["magnitude"], cmap="gray")
    magnitude_axis.set_title("Magnitude", color="white", fontsize=10)
    magnitude_axis.axis("off")
    figure.suptitle(
        f"Audit candidate only — Cartesian slice 120, R=2 ({solver_label})",
        color="white",
        fontsize=14,
    )
    for suffix in ("png", "pdf"):
        figure.savefig(
            output_dir / f"figure2_{solver_label}_candidate.{suffix}",
            dpi=220 if suffix == "png" else None,
            facecolor=figure.get_facecolor(),
        )
    plt.close(figure)


def _compare_with_converged(
    arrays: dict[str, np.ndarray], comparison_dir: Path, output_dir: Path
) -> None:
    comparison_path = comparison_dir / "maps_checkpoint.npz"
    if not comparison_path.is_file():
        raise FileNotFoundError(f"Missing converged comparison bundle: {comparison_path}")
    with np.load(comparison_path) as bundle:
        converged = {key: bundle[key] for key in bundle.files}
    rows: list[dict[str, object]] = []
    for method in ("pmr", "pico"):
        for count in DISPLAY_N:
            key = f"{method}_g_N{count}"
            if key not in arrays or key not in converged:
                continue
            finite_map = torch.from_numpy(arrays[key])
            converged_map = torch.from_numpy(converged[key])
            rows.append(
                {
                    "method": method.upper(),
                    "N": count,
                    "finite_solver": "current Figure-2 adaptive CG, up to 25 iterations",
                    "converged_solver": str(comparison_dir),
                    "finite_vs_converged_nrmse": _nrmse(
                        finite_map, converged_map
                    ),
                    "finite_vs_converged_corr": _corr(finite_map, converged_map),
                    "finite_mean": float(torch.nanmean(finite_map)),
                    "converged_mean": float(torch.nanmean(converged_map)),
                }
            )
    _atomic_csv(pd.DataFrame(rows), output_dir / "solver_target_comparison.csv")


def _build_operators(
    data_path: Path, slice_number: int, acceleration: int, device: torch.device,
    slice_axis: int = 2,
):
    """Load one slice.  ``slice_axis=2`` is the sagittal cut used by the
    revision's fixed-iteration scripts; ``slice_axis=0`` is the readout-resolved
    axial cut of the original submission's Figure 2 (``target[slice]``,
    ``maps[slice]``), i.e. the phase-encode plane of the 3D acquisition."""
    with h5py.File(data_path, "r") as handle:
        if slice_axis == 2:
            csm_np = np.asarray(handle["maps"][:, :, slice_number])
            img_np = np.asarray(handle["target"][:, :, slice_number, 0])
        elif slice_axis == 0:
            csm_np = np.asarray(handle["maps"][slice_number])
            img_np = np.asarray(handle["target"][slice_number, :, :, 0])
        else:
            raise ValueError("slice_axis must be 0 or 2")
        csm = (
            torch.tensor(csm_np, device=device)
            .squeeze()
            .permute(2, 0, 1)
            .to(complex_dtype)
        )
        image = torch.tensor(img_np, device=device).to(complex_dtype)

    image_size = tuple(image.shape)
    coil_count = int(csm.shape[0])
    trajectory_full = gen_grd(image_size, image_size).to(device)
    trajectory_acc = trajectory_full[:, ::acceleration]
    nufft = gridded_nufft(image_size)
    dcf_full = torch.ones(
        trajectory_full.shape[:-1], dtype=real_dtype, device=device
    )
    dcf_acc = torch.ones(
        trajectory_acc.shape[:-1], dtype=real_dtype, device=device
    )
    batch = batching_params(coil_count)
    operator_full = sense_linop(
        image_size,
        trajectory_full,
        csm,
        dcf=dcf_full,
        nufft=nufft,
        bparams=batch,
    )
    operator_acc = sense_linop(
        image_size,
        trajectory_acc,
        csm,
        dcf=dcf_acc,
        nufft=nufft,
        bparams=batch,
    )
    kspace_full = operator_full(image)
    kspace_acc = kspace_full[:, :, ::acceleration]
    return csm, image, operator_full, operator_acc, kspace_full, kspace_acc


def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = sorted(set(int(value) for value in args.n_values))
    if not checkpoints or checkpoints[0] <= 0:
        raise ValueError("n_values must contain positive integers")
    max_samples = checkpoints[-1]
    methods = set(args.methods)
    if not methods.issubset({"pmr", "pico"}):
        raise ValueError(f"Unknown methods: {methods}")

    configuration = {
        "status": "running",
        "started_unix": time.time(),
        "data_path": str(Path(args.data_path).resolve()),
        "slice": args.slice,
        "slice_axis": args.slice_axis,
        "R": args.acceleration,
        "lambda": args.regularization,
        "sigma": args.sigma,
        "max_eigenvalue": args.max_eigenvalue,
        "solver_mode": args.solver_mode,
        "true_residual_tolerance": args.tolerance,
        "max_iterations": args.max_iterations,
        "initialization": "zero"
        if args.solver_mode == "converged"
        else "current Figure-2 paths",
        "N_values": checkpoints,
        "methods": sorted(methods),
        "seed": args.seed,
        "device": str(device),
        "torch_version": torch.__version__,
        "python": platform.python_version(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
    _atomic_json(configuration, output_dir / "status.json")
    print(json.dumps(configuration, indent=2), flush=True)

    (
        csm,
        image,
        operator_full,
        operator_acc,
        kspace_full,
        kspace_acc,
    ) = _build_operators(
        Path(args.data_path), args.slice, args.acceleration, device,
        slice_axis=args.slice_axis,
    )
    analytical = torch.nan_to_num(
        gfactor_sense(
            csm,
            1,
            args.acceleration,
            l2_reg=args.regularization,
        ),
        nan=1.0,
    ).to(device)
    foreground = image.abs() > args.foreground_fraction * image.abs().max()
    normalization = float(args.max_eigenvalue)

    def normal_full_normalized(value: torch.Tensor) -> torch.Tensor:
        return (
            operator_full.normal(value) / normalization
            + args.regularization * value
        )

    def normal_acc_normalized(value: torch.Tensor) -> torch.Tensor:
        return (
            operator_acc.normal(value) / normalization
            + args.regularization * value
        )

    def normal_full(value: torch.Tensor) -> torch.Tensor:
        return operator_full.normal(value) + args.regularization * value

    def normal_acc(value: torch.Tensor) -> torch.Tensor:
        return operator_acc.normal(value) + args.regularization * value

    pmr_full_sum = torch.zeros(image.shape, dtype=torch.float64, device=device)
    pmr_acc_sum = torch.zeros_like(pmr_full_sum)
    # "replica" is PMR as defined by Robson et al.: reconstruct the noisy data
    # k + n and take the voxelwise mean and variance across replicas (Welford in
    # float64 over complex64 reconstructions).  "noise_only" accumulates
    # E|R(n)|^2, the same quantity for a linear R without subtracting the signal.
    pmr_replica = getattr(args, "pmr_estimator", "noise_only") == "replica"
    pmr_full_mean = torch.zeros(image.shape, dtype=torch.complex128, device=device)
    pmr_acc_mean = torch.zeros_like(pmr_full_mean)
    pico_full_sum = torch.zeros_like(pmr_full_sum)
    pico_acc_sum = torch.zeros_like(pmr_full_sum)
    half_sums: dict[str, torch.Tensor] = {}
    solver_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    arrays: dict[str, np.ndarray] = {
        "g_analytical": analytical.detach().cpu().numpy(),
        "magnitude": image.abs().detach().cpu().numpy(),
        "foreground_mask": foreground.detach().cpu().numpy(),
        "n_values": np.asarray(checkpoints, dtype=np.int32),
        "slice_number": np.asarray(args.slice, dtype=np.int32),
        "seed": np.asarray(args.seed, dtype=np.int64),
        "tolerance": np.asarray(args.tolerance, dtype=np.float64),
    }
    generators = {
        "pmr_full": torch.Generator(device=device).manual_seed(args.seed + 11),
        "pmr_acc": torch.Generator(device=device).manual_seed(args.seed + 13),
        "pico_full": torch.Generator(device=device).manual_seed(args.seed + 17),
        "pico_acc": torch.Generator(device=device).manual_seed(args.seed + 19),
    }
    cumulative_runtime = {"pmr": 0.0, "pico": 0.0}
    noise_variance = args.sigma**2
    half_index = max_samples // 2

    try:
        for sample_index in range(1, max_samples + 1):
            if "pmr" in methods:
                started = time.perf_counter()
                noise_full = args.sigma * torch.randn(
                    kspace_full.shape,
                    dtype=kspace_full.dtype,
                    device=device,
                    generator=generators["pmr_full"],
                )
                noise_acc = (
                    # coupled sampling: the accelerated arm is the sampled subset of the
                    # reference arm's realization
                    noise_full[:, :, :: args.acceleration]
                    if getattr(args, "coupled_arms", False)
                    else args.sigma * torch.randn(
                        kspace_acc.shape,
                        dtype=kspace_acc.dtype,
                        device=device,
                        generator=generators["pmr_acc"],
                    )
                )
                data_full = kspace_full + noise_full if pmr_replica else noise_full
                data_acc = kspace_acc + noise_acc if pmr_replica else noise_acc
                if args.solver_mode == "converged":
                    rhs_full = operator_full.adjoint(data_full) / normalization
                    rhs_acc = operator_acc.adjoint(data_acc) / normalization
                    pmr_full, info_full = cg_solve(
                        normal_full_normalized,
                        rhs_full,
                        tolerance=args.tolerance,
                        max_iterations=args.max_iterations,
                        initialization="zero",
                        stop_mode="true_residual",
                    )
                    pmr_acc, info_acc = cg_solve(
                        normal_acc_normalized,
                        rhs_acc,
                        tolerance=args.tolerance,
                        max_iterations=args.max_iterations,
                        initialization="zero",
                        stop_mode="true_residual",
                    )
                else:
                    # Exact current Figure-2 reconstruction scaling/path:
                    # H/2, A^H b/sqrt(2), x0=rhs, up to 25 adaptive-CG updates.
                    rhs_full = operator_full.adjoint(data_full) / math.sqrt(
                        normalization
                    )
                    rhs_acc = operator_acc.adjoint(data_acc) / math.sqrt(
                        normalization
                    )
                    pmr_full, info_full = cg_solve(
                        normal_full_normalized,
                        rhs_full,
                        tolerance=args.tolerance,
                        max_iterations=args.max_iterations,
                        initialization="rhs",
                        stop_mode="fixed",
                    )
                    pmr_acc, info_acc = cg_solve(
                        normal_acc_normalized,
                        rhs_acc,
                        tolerance=args.tolerance,
                        max_iterations=args.max_iterations,
                        initialization="rhs",
                        stop_mode="fixed",
                    )
                if pmr_replica:
                    for x, mean, m2 in (
                        (pmr_full, pmr_full_mean, pmr_full_sum),
                        (pmr_acc, pmr_acc_mean, pmr_acc_sum),
                    ):
                        xd = x.to(torch.complex128)
                        delta = xd - mean
                        mean += delta / sample_index
                        m2 += (delta.conj() * (xd - mean)).real / noise_variance
                else:
                    pmr_full_sum += (
                        pmr_full.to(torch.complex128).abs().square() / noise_variance
                    )
                    pmr_acc_sum += (
                        pmr_acc.to(torch.complex128).abs().square() / noise_variance
                    )
                _synchronize(device)
                cumulative_runtime["pmr"] += time.perf_counter() - started
                _record_solver(
                    solver_rows,
                    method="PMR",
                    arm="full",
                    sample=sample_index,
                    info=info_full,
                )
                _record_solver(
                    solver_rows,
                    method="PMR",
                    arm="accelerated",
                    sample=sample_index,
                    info=info_acc,
                )

            if "pico" in methods:
                started = time.perf_counter()
                phase_full = 2 * math.pi * torch.rand(
                    image.shape,
                    dtype=torch.float32,
                    device=device,
                    generator=generators["pico_full"],
                )
                phase_acc = 2 * math.pi * torch.rand(
                    image.shape,
                    dtype=torch.float32,
                    device=device,
                    generator=generators["pico_acc"],
                )
                probe_full = args.sigma * torch.exp(1j * phase_full).to(
                    complex_dtype
                )
                probe_acc = args.sigma * torch.exp(
                    1j * (phase_full if getattr(args, "coupled_arms", False) else phase_acc)
                ).to(
                    complex_dtype
                )
                if args.solver_mode == "converged":
                    pico_full, info_full = cg_solve(
                        normal_full_normalized,
                        probe_full / normalization,
                        tolerance=args.tolerance,
                        max_iterations=args.max_iterations,
                        initialization="zero",
                        stop_mode="true_residual",
                    )
                    pico_acc, info_acc = cg_solve(
                        normal_acc_normalized,
                        probe_acc / normalization,
                        tolerance=args.tolerance,
                        max_iterations=args.max_iterations,
                        initialization="zero",
                        stop_mode="true_residual",
                    )
                else:
                    # Exact current Figure-2 PICO path: unnormalized H, x0=rhs,
                    # up to 25 adaptive-CG updates.
                    pico_full, info_full = cg_solve(
                        normal_full,
                        probe_full,
                        tolerance=args.tolerance,
                        max_iterations=args.max_iterations,
                        initialization="rhs",
                        stop_mode="fixed",
                    )
                    pico_acc, info_acc = cg_solve(
                        normal_acc,
                        probe_acc,
                        tolerance=args.tolerance,
                        max_iterations=args.max_iterations,
                        initialization="rhs",
                        stop_mode="fixed",
                    )
                pico_full_sum += (
                    (probe_full.conj() * pico_full).real.to(torch.float64)
                    / noise_variance
                )
                pico_acc_sum += (
                    (probe_acc.conj() * pico_acc).real.to(torch.float64)
                    / noise_variance
                )
                _synchronize(device)
                cumulative_runtime["pico"] += time.perf_counter() - started
                _record_solver(
                    solver_rows,
                    method="PICO",
                    arm="full",
                    sample=sample_index,
                    info=info_full,
                )
                _record_solver(
                    solver_rows,
                    method="PICO",
                    arm="accelerated",
                    sample=sample_index,
                    info=info_acc,
                )

            if sample_index == half_index:
                if "pmr" in methods:
                    half_sums["pmr_full"] = pmr_full_sum.clone()
                    half_sums["pmr_acc"] = pmr_acc_sum.clone()
                if "pico" in methods:
                    half_sums["pico_full"] = pico_full_sum.clone()
                    half_sums["pico_acc"] = pico_acc_sum.clone()

            if sample_index not in checkpoints:
                continue

            row: dict[str, object] = {
                "N": sample_index,
                "solver_mode": args.solver_mode,
                "tolerance": args.tolerance,
                "max_iterations": args.max_iterations,
                "initialization": configuration["initialization"],
                "seed": args.seed,
            }
            maps: dict[str, torch.Tensor] = {}
            if "pmr" in methods:
                pmr_g = _g_from_diagonals(
                    pmr_acc_sum / sample_index,
                    pmr_full_sum / sample_index,
                    args.acceleration,
                )
                maps["pmr"] = pmr_g
                arrays[f"pmr_g_N{sample_index}"] = (
                    pmr_g.detach().cpu().numpy()
                )
                row.update(
                    {
                        "pmr_nrmse_full": _nrmse(pmr_g, analytical),
                        "pmr_corr_full": _corr(pmr_g, analytical),
                        "pmr_nrmse_foreground": _nrmse(
                            pmr_g, analytical, foreground
                        ),
                        "pmr_corr_foreground": _corr(
                            pmr_g, analytical, foreground
                        ),
                        "pmr_mean_full": float(torch.nanmean(pmr_g)),
                        "pmr_runtime_cumulative_s": cumulative_runtime["pmr"],
                    }
                )
            if "pico" in methods:
                pico_g = _g_from_diagonals(
                    pico_acc_sum / sample_index,
                    pico_full_sum / sample_index,
                    args.acceleration,
                )
                maps["pico"] = pico_g
                arrays[f"pico_g_N{sample_index}"] = (
                    pico_g.detach().cpu().numpy()
                )
                row.update(
                    {
                        "pico_nrmse_full": _nrmse(pico_g, analytical),
                        "pico_corr_full": _corr(pico_g, analytical),
                        "pico_nrmse_foreground": _nrmse(
                            pico_g, analytical, foreground
                        ),
                        "pico_corr_foreground": _corr(
                            pico_g, analytical, foreground
                        ),
                        "pico_mean_full": float(torch.nanmean(pico_g)),
                        "pico_negative_diagonal_fraction": float(
                            (
                                (pico_acc_sum < 0).float().mean()
                                + (pico_full_sum < 0).float().mean()
                            )
                            / 2
                        ),
                        "pico_runtime_cumulative_s": cumulative_runtime["pico"],
                    }
                )
            if {"pmr", "pico"}.issubset(maps):
                row.update(
                    {
                        "pico_vs_pmr_nrmse_full": _nrmse(
                            maps["pico"], maps["pmr"]
                        ),
                        "pico_vs_pmr_corr_full": _corr(
                            maps["pico"], maps["pmr"]
                        ),
                    }
                )

            current_solver = pd.DataFrame(solver_rows)
            for method in methods:
                method_label = method.upper()
                subset = current_solver[current_solver["method"] == method_label]
                if subset.empty:
                    continue
                row[f"{method}_iterations_median"] = float(
                    subset["iterations"].median()
                )
                row[f"{method}_iterations_p95"] = float(
                    subset["iterations"].quantile(0.95)
                )
                row[f"{method}_true_residual_p95"] = float(
                    subset["true_relative_residual"].quantile(0.95)
                )
                row[f"{method}_converged_fraction"] = float(
                    subset["converged"].mean()
                )

            metric_rows.append(row)
            _atomic_csv(pd.DataFrame(metric_rows), output_dir / "metrics.csv")
            _atomic_csv(current_solver, output_dir / "normal_residuals.csv")
            _atomic_npz(output_dir / "maps_checkpoint.npz", arrays)
            checkpoint_status = {
                **configuration,
                "status": "running",
                "last_checkpoint_N": sample_index,
                "updated_unix": time.time(),
            }
            _atomic_json(checkpoint_status, output_dir / "status.json")
            _plot_candidate(
                arrays,
                output_dir,
                "converged"
                if args.solver_mode == "converged"
                else "current25",
            )
            print(pd.DataFrame([row]).to_string(index=False), flush=True)

        solver_frame = pd.DataFrame(solver_rows)
        residual_summary = (
            solver_frame.groupby(["method", "arm"], as_index=False)
            .agg(
                samples=("sample", "count"),
                converged_fraction=("converged", "mean"),
                breakdown_fraction=("breakdown", "mean"),
                iterations_median=("iterations", "median"),
                iterations_p95=("iterations", lambda values: values.quantile(0.95)),
                iterations_max=("iterations", "max"),
                true_relative_residual_median=(
                    "true_relative_residual",
                    "median",
                ),
                true_relative_residual_p95=(
                    "true_relative_residual",
                    lambda values: values.quantile(0.95),
                ),
                true_relative_residual_max=("true_relative_residual", "max"),
            )
            if not solver_frame.empty
            else pd.DataFrame()
        )
        _atomic_csv(residual_summary, output_dir / "residual_summary.csv")

        split_rows: list[dict[str, object]] = []
        second_count = max_samples - half_index
        if half_index > 0 and second_count > 0:
            for method in sorted(methods):
                full_key = f"{method}_full"
                acc_key = f"{method}_acc"
                if full_key not in half_sums or acc_key not in half_sums:
                    continue
                total_full = pmr_full_sum if method == "pmr" else pico_full_sum
                total_acc = pmr_acc_sum if method == "pmr" else pico_acc_sum
                first = _g_from_diagonals(
                    half_sums[acc_key] / half_index,
                    half_sums[full_key] / half_index,
                    args.acceleration,
                )
                second = _g_from_diagonals(
                    (total_acc - half_sums[acc_key]) / second_count,
                    (total_full - half_sums[full_key]) / second_count,
                    args.acceleration,
                )
                split_rows.append(
                    {
                        "method": method.upper(),
                        "first_half_N": half_index,
                        "second_half_N": second_count,
                        "split_half_nrmse": _nrmse(first, second),
                        "split_half_corr": _corr(first, second),
                        "first_half_vs_analytical_nrmse": _nrmse(
                            first, analytical
                        ),
                        "second_half_vs_analytical_nrmse": _nrmse(
                            second, analytical
                        ),
                    }
                )
        _atomic_csv(pd.DataFrame(split_rows), output_dir / "split_half_qc.csv")

        if args.comparison_dir:
            _compare_with_converged(
                arrays, Path(args.comparison_dir).resolve(), output_dir
            )

        configuration.update(
            {
                "status": "complete",
                "completed_unix": time.time(),
                "last_checkpoint_N": max_samples,
                "metrics_path": str(output_dir / "metrics.csv"),
                "residuals_path": str(output_dir / "normal_residuals.csv"),
                "maps_path": str(output_dir / "maps_checkpoint.npz"),
                "split_half_path": str(output_dir / "split_half_qc.csv"),
            }
        )
        _atomic_json(configuration, output_dir / "status.json")
        print(f"COMPLETE {output_dir}", flush=True)
    except Exception as error:
        configuration.update(
            {
                "status": "failed",
                "failed_unix": time.time(),
                "error": repr(error),
            }
        )
        _atomic_json(configuration, output_dir / "status.json")
        raise


def _parse_ints(values: Iterable[str]) -> list[int]:
    return [int(value) for value in values]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default=str(DEFAULT_DATA))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--coupled-arms",
        action="store_true",
        help="drive both arms with the same randomness: one probe for both arms for "
             "PICO, and the sampled subset of one noise realization for PMR.",
    )
    parser.add_argument("--slice", type=int, default=120)
    parser.add_argument("--slice-axis", type=int, default=2, choices=[0, 2],
                        help="2: sagittal cut maps[:, :, slice] (revision default); "
                             "0: readout-resolved axial cut maps[slice] (original Figure 2)")
    parser.add_argument("--acceleration", type=int, default=2)
    parser.add_argument("--regularization", type=float, default=0.0)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--max-eigenvalue", type=float, default=2.0)
    parser.add_argument(
        "--solver-mode", choices=["converged", "current25"], default="converged"
    )
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument(
        "--n-values",
        nargs="+",
        default=["10", "20", "40", "80", "200", "500", "1000", "2000"],
    )
    parser.add_argument(
        "--methods", nargs="+", default=["pmr", "pico"], choices=["pmr", "pico"]
    )
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--foreground-fraction", type=float, default=0.05)
    parser.add_argument("--comparison-dir", default=None)
    parser.add_argument(
        "--pmr-estimator",
        choices=["noise_only", "replica"],
        default="noise_only",
        help="noise_only: accumulate E|R(n)|^2 (exact for a linear R). "
             "replica: Robson-style mean/variance across reconstructions of k + n.",
    )
    args = parser.parse_args()
    args.n_values = _parse_ints(args.n_values)
    if args.solver_mode == "current25" and args.max_iterations != 25:
        raise ValueError("current25 mode requires --max-iterations 25")
    run(args)


if __name__ == "__main__":
    main()
