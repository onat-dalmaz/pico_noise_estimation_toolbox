#!/usr/bin/env python
"""Run the final corrected, checkpointed non-Cartesian experiment grid."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from mr_recon.covariance_apply import COVARIANCE_APPLY_DESCRIPTION
from mr_recon.dtypes import complex_dtype
from scripts.audit_noncart_saved_maps import build_masks
from scripts.factor_comparator_core import (
    PhysicalReconstruction,
    direct_real_contribution,
    masked_nrmse,
    negative_diagonal_stats,
    proper_complex_gaussian,
    random_phase,
)
from scripts.noncart_audit_core import solve_cg_true_residual
from scripts.noncart_final_grid_core import (
    COMMON_LAMBDA_PHYSICAL,
    DENSE_CHECKPOINTS,
    MAX_ITERATIONS,
    REFERENCE_CHECKPOINTS,
    TRUE_RESIDUAL_EVERY,
    TRUE_RESIDUAL_RTOL,
    build_operator_manifest,
    evaluate_stability,
    normalized_lambda,
    physical_variance_products,
    real_rademacher,
)
from scripts.run_noncart_weekend_audit import System, build_problem


OUT_ROOT = Path(
    os.environ.get(
        "NONCART_FINAL_OUT_ROOT",
        str(ROOT / "experiments/noncartesian_phantom/results/noncart_final_grid"),
    )
)
MANIFEST_PATH = OUT_ROOT / "operator_manifest.json"
NOISE_SIGMA_FOR_DISPLAY = 1e-2
STABILITY_ITERATIONS = (25, 50, 100, 200, 400, 800, 1200)
def _welford_increment(image, means, arm, sample):
    """One Welford update for the replica PMR estimator, returned as the float64
    increment of the sum of squared deviations (so that sum / N is the variance)."""
    x = image.to(torch.complex128)
    delta = x - means[arm]
    means[arm] = means[arm] + delta / sample
    return (delta.conj() * (x - means[arm])).real


ABLATION_METHODS = (
    "gaussian_pmr",
    "random_phase_factor",
    "random_phase_direct",
    "real_rademacher_direct",
    "proper_gaussian_direct",
)
_STOP_REQUESTED = False


def _request_stop(signum, _frame) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print(f"[final-grid] received signal {signum}; checkpointing", flush=True)


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, default=float)
    os.replace(temporary, path)


def _atomic_npy(path: Path, array) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "wb") as stream:
        np.save(stream, np.asarray(array))
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _metadata(args: argparse.Namespace) -> dict:
    return {
        "arguments": vars(args),
        "audit_sources": [
            "reports/NONCART_VALIDITY_AUDIT.md",
            "reports/NONCART_FINAL_DOC_NUMBERS.md",
        ],
        "operator": "unweighted pre-whitened A^H A",
        "dcf_data_weighting": False,
        "dcf_warm_start": False,
        "preconditioner": None,
        "initialization": (
            "zero initialization for every PMR/factor solve and for the first "
            "direct-covariance solve; the second direct-covariance solve is "
            "warm-started from the first solve's Krylov space and certified on "
            "the same true residual"
        ),
        "covariance_apply": COVARIANCE_APPLY_DESCRIPTION,
        "stopping": "true ||rhs-Hx||_2/||rhs||_2",
        "true_residual_tolerance": TRUE_RESIDUAL_RTOL,
        "true_residual_recompute_every": TRUE_RESIDUAL_EVERY,
        "max_iterations": MAX_ITERATIONS,
        "lambda_physical": COMMON_LAMBDA_PHYSICAL,
        "pmr_moment_accumulation": "float64",
        "pmr_estimator": getattr(args, "pmr_estimator", "noise_only"),
        "arm_sampling": "coupled" if getattr(args, "coupled_arms", True) else "independent",
        "pmr_sigma_scale": getattr(args, "pmr_sigma_scale", 1.0),
        "linear_operator_input": "noise only; no signal added",
        "probe_second_moment": "E|z_j|^2=1",
        "primary_outputs": "absolute variance and SD in physical image units",
        "secondary_output": "regularized relative-noise ratio",
        "maps": "full resolution and unmasked",
        "mask_B": "|x_ref| > 0.05 max(|x_ref|)",
        "mask_C": "largest 8-connected B component; fill holes; erode 3x3 once",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "host": os.environ.get("SLURMD_NODENAME"),
    }


def _load_manifest(path: Path) -> dict:
    with open(path, encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not manifest.get("accepted"):
        raise RuntimeError(f"operator manifest was not accepted: {path}")
    if not math.isclose(
        float(manifest["lambda_physical"]),
        COMMON_LAMBDA_PHYSICAL,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise RuntimeError("operator manifest has the wrong physical lambda")
    return manifest


def _build_final_problem(
    R: int, device: torch.device, manifest_path: Path
) -> tuple[object, System, System, dict]:
    manifest = _load_manifest(manifest_path)
    problem = build_problem(R, device)
    estimated_ref = problem.max_eigen_ref
    estimated_acc = problem.max_eigen_acc
    canonical_ref = float(manifest["arms"]["reference"]["max_eigen"])
    canonical_acc = float(manifest["arms"][f"R{R}"]["max_eigen"])
    for name, estimated, canonical in (
        ("reference", estimated_ref, canonical_ref),
        ("accelerated", estimated_acc, canonical_acc),
    ):
        relative = abs(estimated - canonical) / max(abs(canonical), 1e-30)
        if relative > 1e-6:
            raise RuntimeError(
                f"{name} eigenvalue changed: estimated={estimated}, "
                f"manifest={canonical}, relative={relative}"
            )
    problem.max_eigen_ref = canonical_ref
    problem.max_eigen_acc = canonical_acc
    ref_system = System(
        problem.A_ref,
        canonical_ref,
        normalized_lambda(COMMON_LAMBDA_PHYSICAL, canonical_ref),
        MAX_ITERATIONS,
        TRUE_RESIDUAL_RTOL,
        TRUE_RESIDUAL_EVERY,
    )
    acc_system = System(
        problem.A_acc,
        canonical_acc,
        normalized_lambda(COMMON_LAMBDA_PHYSICAL, canonical_acc),
        MAX_ITERATIONS,
        TRUE_RESIDUAL_RTOL,
        TRUE_RESIDUAL_EVERY,
    )
    return problem, ref_system, acc_system, manifest


def _physical_reconstruction(system: System) -> PhysicalReconstruction:
    return PhysicalReconstruction(
        forward=system.A.forward,
        adjoint=system.A.adjoint,
        solve_normalized=system.solve_image_rhs,
        max_eigen=system.max_eigen,
        normal=system.A.normal,
    )


def _masks(problem) -> tuple[dict[str, np.ndarray], dict]:
    masks, metadata = build_masks(problem.image.abs().cpu().numpy())
    selected = {
        "B": masks["B_support_5pct"],
        "C": masks["C_lcc_fill_erode1"],
    }
    counts = {name: int(mask.sum()) for name, mask in selected.items()}
    if counts != {"B": 18040, "C": 16449}:
        raise RuntimeError(f"final masks disagree with audit: {counts}")
    return selected, metadata


def _save_masks_and_magnitude(out: Path, problem, masks: dict[str, np.ndarray]) -> None:
    _atomic_npy(out / "analysis_mask_B.npy", masks["B"])
    _atomic_npy(out / "analysis_mask_C.npy", masks["C"])
    _atomic_npy(
        out / "reference_magnitude_for_masks.npy",
        problem.image.abs().cpu().numpy(),
    )


def _signal_reconstruction(system: System, kspace: torch.Tensor):
    rhs = system.A.adjoint(kspace.type(complex_dtype)) / float(system.max_eigen)
    return system.solve_image_rhs(rhs)


def _save_signal_reconstructions(
    out: Path,
    problem,
    ref_system: System,
    acc_system: System,
) -> dict:
    ref, ref_info = _signal_reconstruction(
        ref_system, problem.ksp_ref
    )
    acc, acc_info = _signal_reconstruction(
        acc_system, problem.ksp_acc
    )
    _atomic_npy(out / "reconstruction_ref_physical.npy", ref.cpu().numpy())
    _atomic_npy(out / "reconstruction_acc_physical.npy", acc.cpu().numpy())
    _atomic_npy(
        out / "reconstructed_magnitude_ref_physical.npy",
        ref.abs().cpu().numpy(),
    )
    _atomic_npy(
        out / "reconstructed_magnitude_acc_physical.npy",
        acc.abs().cpu().numpy(),
    )
    return {"reference": ref_info, "accelerated": acc_info}


def _info_values(info: dict) -> tuple[int, float, bool, int]:
    return (
        int(info["iterations"]),
        float(info["true_relative_residual"]),
        bool(info["converged"]),
        int(info.get("reliable_updates", 0)),
    )


def _validate_infos(infos: list[dict], context: str) -> None:
    failures = [
        info
        for info in infos
        if not info["converged"]
        or not np.isfinite(info["true_relative_residual"])
        or info["true_relative_residual"] > TRUE_RESIDUAL_RTOL
    ]
    if failures:
        raise RuntimeError(f"{context} produced unconverged solve: {failures[0]}")


def _save_state(
    path: Path,
    *,
    count: int,
    elapsed: float,
    sums: dict[str, torch.Tensor],
    generators: dict[str, torch.Generator],
    diagnostics: dict[str, list[tuple[int, float, bool, int]]],
    sample_times: list[float],
    extras: dict[str, list[float]] | None = None,
) -> None:
    arrays = {
        "count": np.asarray(count, dtype=np.int64),
        "elapsed": np.asarray(elapsed, dtype=np.float64),
        "sample_times": np.asarray(sample_times, dtype=np.float64),
    }
    arrays.update(
        {f"sum_{name}": value.detach().cpu().numpy() for name, value in sums.items()}
    )
    arrays.update(
        {
            f"rng_{name}": generator.get_state().detach().cpu().numpy()
            for name, generator in generators.items()
        }
    )
    for name, values in diagnostics.items():
        if values:
            arrays[f"diag_{name}"] = np.asarray(values, dtype=np.float64)
        else:
            arrays[f"diag_{name}"] = np.empty((0, 4), dtype=np.float64)
    for name, values in (extras or {}).items():
        arrays[f"extra_{name}"] = np.asarray(values, dtype=np.float64)
    _atomic_npz(path, **arrays)


def _load_state(
    path: Path,
    *,
    device: torch.device,
    sums: dict[str, torch.Tensor],
    generators: dict[str, torch.Generator],
    diagnostics: dict[str, list[tuple[int, float, bool, int]]],
    extras: dict[str, list[float]] | None = None,
) -> tuple[int, float, list[float]]:
    if not path.is_file():
        return 0, 0.0, []
    state = np.load(path, allow_pickle=False)
    for name in sums:
        sums[name] = torch.tensor(
            state[f"sum_{name}"], dtype=torch.float64, device=device
        )
    for name, generator in generators.items():
        generator.set_state(
            torch.tensor(state[f"rng_{name}"], dtype=torch.uint8, device="cpu")
        )
    for name in diagnostics:
        diagnostics[name] = [
            (int(row[0]), float(row[1]), bool(row[2]), int(row[3]))
            for row in state[f"diag_{name}"]
        ]
    for name in extras or {}:
        extras[name] = state[f"extra_{name}"].astype(float).tolist()
    return (
        int(state["count"]),
        float(state["elapsed"]),
        state["sample_times"].astype(float).tolist(),
    )


def _diagnostic_rows(
    diagnostics: dict[str, list[tuple[int, float, bool, int]]],
    sample_times: list[float],
    *,
    solves_per_sample: int,
    method: str,
) -> list[dict]:
    rows = []
    for arm, values in diagnostics.items():
        for offset, (iterations, residual, converged, reliable_updates) in enumerate(
            values
        ):
            sample = offset // solves_per_sample + 1
            rows.append(
                {
                    "method": method,
                    "arm": arm,
                    "sample": sample,
                    "solve_index": offset % solves_per_sample + 1,
                    "sample_wall_time_s": sample_times[sample - 1],
                    "iterations": iterations,
                    "true_relative_residual": residual,
                    "converged": converged,
                    "reliable_updates": reliable_updates,
                }
            )
    return rows


def _save_variance_products(
    out: Path,
    prefix: str,
    products: dict[str, np.ndarray],
) -> None:
    for name, array in products.items():
        _atomic_npy(out / f"{prefix}_{name}.npy", array)
    for arm in ("ref", "acc"):
        variance = products[f"variance_{arm}_physical"]
        _atomic_npy(
            out / f"{prefix}_variance_{arm}_physical_sigma0p01.npy",
            variance * NOISE_SIGMA_FOR_DISPLAY**2,
        )
        _atomic_npy(
            out / f"{prefix}_sd_{arm}_physical_sigma0p01.npy",
            np.sqrt(np.maximum(variance, 0.0)) * NOISE_SIGMA_FOR_DISPLAY,
        )


def _bootstrap_roi(
    ref: np.ndarray,
    acc: np.ndarray,
    R: int,
    seed: int,
    replicates: int = 500,
) -> dict:
    generator = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled_ref = ref[
            generator.integers(0, len(ref), size=len(ref))
        ].mean()
        sampled_acc = acc[
            generator.integers(0, len(acc), size=len(acc))
        ].mean()
        values[index] = math.sqrt(
            max(sampled_acc, 0.0) / max(sampled_ref, 1e-30)
        ) / math.sqrt(R)
    return {
        "estimate": math.sqrt(max(acc.mean(), 0.0) / max(ref.mean(), 1e-30))
        / math.sqrt(R),
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
    }


def run_reference(args: argparse.Namespace) -> None:
    expected = 50000 if args.R == 2 else 30000
    if args.N_max != expected:
        raise ValueError(f"R={args.R} final reference requires N={expected}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    problem, ref_system, acc_system, manifest = _build_final_problem(
        args.R, device, Path(args.manifest)
    )
    out = Path(args.out_root) / "reference" / f"R{args.R}" / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    masks, mask_metadata = _masks(problem)
    _save_masks_and_magnitude(out, problem, masks)
    signal_info = _save_signal_reconstructions(
        out, problem, ref_system, acc_system
    )
    config = {
        **_metadata(args),
        "R": args.R,
        "seed": args.seed,
        "N_max": args.N_max,
        "checkpoints": [
            value for value in REFERENCE_CHECKPOINTS if value <= args.N_max
        ],
        "checkpoint_state_every": args.checkpoint_every,
        "max_eigen_ref": ref_system.max_eigen,
        "max_eigen_acc": acc_system.max_eigen,
        "lambda_normalized_ref": ref_system.lambda_normalized,
        "lambda_normalized_acc": acc_system.lambda_normalized,
        "manifest": manifest,
        "mask_metadata": mask_metadata,
        "signal_solver_diagnostics": signal_info,
    }
    _atomic_json(out / "config.json", config)

    ref_reconstruction = _physical_reconstruction(ref_system)
    acc_reconstruction = _physical_reconstruction(acc_system)
    shape = tuple(problem.image.shape)
    sums = {
        name: torch.zeros(shape, dtype=torch.float64, device=device)
        for name in ("ref", "acc", "ref_even", "ref_odd", "acc_even", "acc_odd")
    }
    generators = {
        "ref": torch.Generator(device=device).manual_seed(args.seed * 1000 + 11),
        "acc": torch.Generator(device=device).manual_seed(args.seed * 1000 + 29),
    }
    diagnostics: dict[str, list[tuple[int, float, bool, int]]] = {
        "reference": [],
        "accelerated": [],
    }
    extras = {
        f"roi_{arm}_{mask}": []
        for arm in ("ref", "acc")
        for mask in ("B", "C")
    }
    state_path = out / "resume_state.npz"
    count, elapsed, sample_times = _load_state(
        state_path,
        device=device,
        sums=sums,
        generators=generators,
        diagnostics=diagnostics,
        extras=extras,
    )
    progress = _read_csv(out / "checkpoint_summary.csv")
    checkpoints = {
        value for value in REFERENCE_CHECKPOINTS if value <= args.N_max
    }
    print(
        f"[reference] R={args.R} seed={args.seed} resume_N={count} "
        f"target={args.N_max}",
        flush=True,
    )

    for sample in range(count + 1, args.N_max + 1):
        _sync(device)
        started = time.perf_counter()
        ref_probe = proper_complex_gaussian(
            problem.ksp_ref.shape,
            generator=generators["ref"],
            device=device,
            dtype=problem.ksp_ref.dtype,
        )
        acc_probe = proper_complex_gaussian(
            problem.ksp_acc.shape,
            generator=generators["acc"],
            device=device,
            dtype=problem.ksp_acc.dtype,
        )
        ref_image, ref_infos = ref_reconstruction.apply_factor(ref_probe)
        acc_image, acc_infos = acc_reconstruction.apply_factor(acc_probe)
        _validate_infos(ref_infos, "reference PMR reference arm")
        _validate_infos(acc_infos, "reference PMR accelerated arm")
        ref_power = ref_image.abs().double().square()
        acc_power = acc_image.abs().double().square()
        sums["ref"] += ref_power
        sums["acc"] += acc_power
        parity = "even" if sample % 2 == 0 else "odd"
        sums[f"ref_{parity}"] += ref_power
        sums[f"acc_{parity}"] += acc_power
        for mask_name, mask in masks.items():
            torch_mask = torch.tensor(mask, device=device)
            extras[f"roi_ref_{mask_name}"].append(
                float(ref_power[torch_mask].mean().item())
            )
            extras[f"roi_acc_{mask_name}"].append(
                float(acc_power[torch_mask].mean().item())
            )
        diagnostics["reference"].append(_info_values(ref_infos[0]))
        diagnostics["accelerated"].append(_info_values(acc_infos[0]))
        _sync(device)
        sample_elapsed = time.perf_counter() - started
        sample_times.append(sample_elapsed)
        elapsed += sample_elapsed
        count = sample

        save_state = (
            sample % args.checkpoint_every == 0
            or sample in checkpoints
            or sample == args.N_max
            or _STOP_REQUESTED
        )
        if save_state:
            _save_state(
                state_path,
                count=count,
                elapsed=elapsed,
                sums=sums,
                generators=generators,
                diagnostics=diagnostics,
                sample_times=sample_times,
                extras=extras,
            )
        if sample in checkpoints:
            products = physical_variance_products(
                sum_ref=sums["ref"].cpu().numpy(),
                sum_acc=sums["acc"].cpu().numpy(),
                count=sample,
                acceleration=args.R,
            )
            prefix = f"pmr_N{sample}"
            _save_variance_products(out, prefix, products)
            split_products = {}
            for parity, parity_count in (
                ("even", sample // 2),
                ("odd", sample - sample // 2),
            ):
                split_products[parity] = physical_variance_products(
                    sum_ref=sums[f"ref_{parity}"].cpu().numpy(),
                    sum_acc=sums[f"acc_{parity}"].cpu().numpy(),
                    count=parity_count,
                    acceleration=args.R,
                )
                _save_variance_products(
                    out, f"{prefix}_split_{parity}", split_products[parity]
                )
            row = {
                "N": sample,
                "cumulative_wall_time_s": elapsed,
                "mean_sample_wall_time_s": float(np.mean(sample_times)),
            }
            for mask_name, mask in masks.items():
                row[f"split_half_ratio_nrmse_mask_{mask_name}"] = masked_nrmse(
                    torch.tensor(
                        split_products["even"][
                            "regularized_relative_noise_ratio"
                        ]
                    ),
                    torch.tensor(
                        split_products["odd"][
                            "regularized_relative_noise_ratio"
                        ]
                    ),
                    torch.tensor(mask),
                )
                bootstrap = _bootstrap_roi(
                    np.asarray(extras[f"roi_ref_{mask_name}"]),
                    np.asarray(extras[f"roi_acc_{mask_name}"]),
                    args.R,
                    seed=args.seed * 100000 + sample + ord(mask_name),
                )
                row[f"roi_ratio_mask_{mask_name}"] = bootstrap["estimate"]
                row[f"roi_ratio_ci95_low_mask_{mask_name}"] = bootstrap[
                    "ci95_low"
                ]
                row[f"roi_ratio_ci95_high_mask_{mask_name}"] = bootstrap[
                    "ci95_high"
                ]
            progress = [
                existing
                for existing in progress
                if int(existing["N"]) != sample
            ] + [row]
            progress.sort(key=lambda value: int(value["N"]))
            _atomic_csv(out / "checkpoint_summary.csv", progress)
            _atomic_csv(
                out / "solver_diagnostics.csv",
                _diagnostic_rows(
                    diagnostics,
                    sample_times,
                    solves_per_sample=1,
                    method="gaussian_pmr",
                ),
            )
            print(
                f"[reference] R={args.R} seed={args.seed} N={sample} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
        if _STOP_REQUESTED:
            raise RuntimeError("checkpointed after scheduler signal")

    _atomic_json(
        out / "status.json",
        {
            "status": "complete",
            "N": count,
            "completed_unix": time.time(),
            "output": str(out),
        },
    )


def _run_pair_estimator(
    *,
    args: argparse.Namespace,
    method: str,
    out: Path,
    problem,
    ref_system: System,
    acc_system: System,
    masks: dict[str, np.ndarray],
) -> None:
    device = problem.device
    ref_reconstruction = _physical_reconstruction(ref_system)
    acc_reconstruction = _physical_reconstruction(acc_system)
    sums = {
        "ref": torch.zeros(problem.image.shape, dtype=torch.float64, device=device),
        "acc": torch.zeros(problem.image.shape, dtype=torch.float64, device=device),
    }
    if method == "pmr_gaussian":
        generators = {
            "ref": torch.Generator(device=device).manual_seed(args.seed + 401),
            "acc": torch.Generator(device=device).manual_seed(args.seed + 509),
        }
        solves_per_sample = 1
    elif method == "pico_random_phase_direct":
        generators = {
            "shared": torch.Generator(device=device).manual_seed(args.seed + 601),
            "acc_probe": torch.Generator(device=device).manual_seed(args.seed + 607),
        }
        solves_per_sample = 2
    else:
        raise ValueError(method)
    diagnostics: dict[str, list[tuple[int, float, bool, int]]] = {
        "reference": [],
        "accelerated": [],
    }
    state_path = out / "resume_state.npz"
    count, elapsed, sample_times = _load_state(
        state_path,
        device=device,
        sums=sums,
        generators=generators,
        diagnostics=diagnostics,
    )
    pmr_replica = (
        method == "pmr_gaussian"
        and getattr(args, "pmr_estimator", "noise_only") == "replica"
    )
    if pmr_replica and count:
        raise SystemExit(
            "replica PMR cannot resume: the running mean is not part of the saved state. "
            f"Remove {state_path} and run the stage in one job."
        )
    pmr_sigma_scale = float(getattr(args, "pmr_sigma_scale", 1.0))
    replica_means = {
        "ref": torch.zeros(problem.image.shape, dtype=torch.complex128, device=device),
        "acc": torch.zeros(problem.image.shape, dtype=torch.complex128, device=device),
    }
    progress = _read_csv(out / "checkpoint_summary.csv")
    checkpoints = set(DENSE_CHECKPOINTS)
    print(
        f"[matched] R={args.R} method={method} resume_N={count} "
        f"target={args.N_max}",
        flush=True,
    )
    for sample in range(count + 1, args.N_max + 1):
        _sync(device)
        started = time.perf_counter()
        if method == "pmr_gaussian":
            ref_probe = proper_complex_gaussian(
                problem.ksp_ref.shape,
                generator=generators["ref"],
                device=device,
                dtype=problem.ksp_ref.dtype,
            )
            acc_probe = (
                # coupled sampling: the accelerated arm is the shot subset of the
                # reference arm's realization, so the shared part of the error
                # cancels in the ratio
                ref_probe[:, :, :: problem.R]
                if getattr(args, "coupled_arms", False)
                else proper_complex_gaussian(
                    problem.ksp_acc.shape,
                    generator=generators["acc"],
                    device=device,
                    dtype=problem.ksp_acc.dtype,
                )
            )
            if pmr_replica:
                ref_image, ref_infos = ref_reconstruction.apply_factor(
                    problem.ksp_ref + pmr_sigma_scale * ref_probe
                )
                acc_image, acc_infos = acc_reconstruction.apply_factor(
                    problem.ksp_acc + pmr_sigma_scale * acc_probe
                )
                ref_contribution = _welford_increment(
                    ref_image, replica_means, "ref", sample
                ) / (pmr_sigma_scale ** 2)
                acc_contribution = _welford_increment(
                    acc_image, replica_means, "acc", sample
                ) / (pmr_sigma_scale ** 2)
            else:
                ref_image, ref_infos = ref_reconstruction.apply_factor(ref_probe)
                acc_image, acc_infos = acc_reconstruction.apply_factor(acc_probe)
                ref_contribution = ref_image.abs().double().square()
                acc_contribution = acc_image.abs().double().square()
        else:
            probe = random_phase(
                problem.image.shape,
                generator=generators["shared"],
                device=device,
                dtype=problem.image.dtype,
            )
            acc_probe = (
                probe
                if getattr(args, "coupled_arms", True)
                else random_phase(
                    problem.image.shape,
                    generator=generators["acc_probe"],
                    device=device,
                    dtype=problem.image.dtype,
                )
            )
            ref_image, ref_infos = ref_reconstruction.apply_direct(probe)
            acc_image, acc_infos = acc_reconstruction.apply_direct(acc_probe)
            ref_contribution = direct_real_contribution(probe, ref_image)
            acc_contribution = direct_real_contribution(acc_probe, acc_image)
        _validate_infos(ref_infos, f"{method} reference arm")
        _validate_infos(acc_infos, f"{method} accelerated arm")
        sums["ref"] += ref_contribution
        sums["acc"] += acc_contribution
        diagnostics["reference"].extend(_info_values(info) for info in ref_infos)
        diagnostics["accelerated"].extend(_info_values(info) for info in acc_infos)
        _sync(device)
        sample_elapsed = time.perf_counter() - started
        sample_times.append(sample_elapsed)
        elapsed += sample_elapsed
        count = sample
        save_state = (
            sample % args.checkpoint_every == 0
            or sample in checkpoints
            or sample == args.N_max
            or _STOP_REQUESTED
        )
        if save_state:
            _save_state(
                state_path,
                count=count,
                elapsed=elapsed,
                sums=sums,
                generators=generators,
                diagnostics=diagnostics,
                sample_times=sample_times,
            )
        if sample in checkpoints:
            products = physical_variance_products(
                sum_ref=sums["ref"].cpu().numpy(),
                sum_acc=sums["acc"].cpu().numpy(),
                count=sample,
                acceleration=args.R,
            )
            _save_variance_products(out, f"N{sample}", products)
            row = {
                "method": method,
                "N": sample,
                "cumulative_wall_time_s": elapsed,
                "mean_sample_wall_time_s": float(np.mean(sample_times)),
                "total_cg_iterations": int(
                    sum(value[0] for values in diagnostics.values() for value in values)
                ),
                "maximum_true_relative_residual": float(
                    max(value[1] for values in diagnostics.values() for value in values)
                ),
            }
            for arm in ("ref", "acc"):
                estimate = products[f"variance_{arm}_physical"]
                for mask_name, mask in masks.items():
                    stats = negative_diagonal_stats(
                        torch.tensor(estimate[mask])
                    )
                    row[
                        f"negative_fraction_{arm}_mask_{mask_name}"
                    ] = stats["negative_fraction"]
                    row[f"negative_min_{arm}_mask_{mask_name}"] = stats[
                        "negative_min"
                    ]
            progress = [
                existing for existing in progress if int(existing["N"]) != sample
            ] + [row]
            progress.sort(key=lambda value: int(value["N"]))
            _atomic_csv(out / "checkpoint_summary.csv", progress)
            _atomic_csv(
                out / "solver_diagnostics.csv",
                _diagnostic_rows(
                    diagnostics,
                    sample_times,
                    solves_per_sample=solves_per_sample,
                    method=method,
                ),
            )
            print(
                f"[matched] R={args.R} method={method} N={sample} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
        if _STOP_REQUESTED:
            raise RuntimeError("checkpointed after scheduler signal")
    _atomic_json(
        out / "status.json",
        {"status": "complete", "N": count, "method": method},
    )


def run_matched(args: argparse.Namespace) -> None:
    if args.N_max != 10000:
        raise ValueError("matched final run requires N=10000")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    problem, ref_system, acc_system, manifest = _build_final_problem(
        args.R, device, Path(args.manifest)
    )
    parent = Path(args.out_root) / "matched" / f"R{args.R}" / f"seed{args.seed}"
    parent.mkdir(parents=True, exist_ok=True)
    masks, mask_metadata = _masks(problem)
    _save_masks_and_magnitude(parent, problem, masks)
    signal_info = _save_signal_reconstructions(
        parent, problem, ref_system, acc_system
    )
    _atomic_json(
        parent / "config.json",
        {
            **_metadata(args),
            "R": args.R,
            "seed": args.seed,
            "N_max": args.N_max,
            "checkpoints": DENSE_CHECKPOINTS,
            "methods": ["pmr_gaussian", "pico_random_phase_direct"],
            "max_eigen_ref": ref_system.max_eigen,
            "max_eigen_acc": acc_system.max_eigen,
            "lambda_normalized_ref": ref_system.lambda_normalized,
            "lambda_normalized_acc": acc_system.lambda_normalized,
            "manifest": manifest,
            "mask_metadata": mask_metadata,
            "signal_solver_diagnostics": signal_info,
        },
    )
    selected = getattr(args, "matched_methods", None) or ["pmr_gaussian", "pico_random_phase_direct"]
    for method in selected:
        # a run that changes the arm-sampling protocol writes beside the others
        method_out = parent / (method + (getattr(args, "arm_tag", "") or ""))
        method_out.mkdir(parents=True, exist_ok=True)
        _run_pair_estimator(
            args=args,
            method=method,
            out=method_out,
            problem=problem,
            ref_system=ref_system,
            acc_system=acc_system,
            masks=masks,
        )
    _atomic_json(
        parent / "status.json",
        {"status": "complete", "methods": ["pmr_gaussian", "pico_random_phase_direct"]},
    )


def _ablation_sample(
    method: str,
    reconstruction: PhysicalReconstruction,
    problem,
    generator: torch.Generator,
    probe_shape: str = "acc",
    probe: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[dict], torch.Tensor]:
    """One ablation sample for one arm.

    Data-space probes (``gaussian_pmr``, ``random_phase_factor``) have the shape
    of that arm's k-space and are drawn per arm.  Image-space probes are shared
    between the two arms, as in the matched stage, so ``probe`` can be passed in
    to reuse the reference arm's draw.
    """
    ksp = problem.ksp_ref if probe_shape == "ref" else problem.ksp_acc
    if method == "gaussian_pmr":
        probe = proper_complex_gaussian(
            ksp.shape,
            generator=generator,
            device=problem.device,
            dtype=ksp.dtype,
        )
        image, infos = reconstruction.apply_factor(probe)
        return image.abs().double().square(), infos, probe
    if method == "random_phase_factor":
        probe = random_phase(
            ksp.shape,
            generator=generator,
            device=problem.device,
            dtype=ksp.dtype,
        )
        image, infos = reconstruction.apply_factor(probe)
        return image.abs().double().square(), infos, probe
    if probe is None:
        if method == "random_phase_direct":
            probe = random_phase(
                problem.image.shape,
                generator=generator,
                device=problem.device,
                dtype=problem.image.dtype,
            )
        elif method == "real_rademacher_direct":
            probe = real_rademacher(
                problem.image.shape,
                generator=generator,
                device=problem.device,
                dtype=problem.image.dtype,
            )
        elif method == "proper_gaussian_direct":
            probe = proper_complex_gaussian(
                problem.image.shape,
                generator=generator,
                device=problem.device,
                dtype=problem.image.dtype,
            )
        else:
            raise ValueError(method)
    image, infos = reconstruction.apply_direct(probe)
    return direct_real_contribution(probe, image), infos, probe


def run_ablation(args: argparse.Namespace) -> None:
    if args.R != 2 or args.N_max != 10000:
        raise ValueError("final ablation requires R=2 and N=10000")
    if args.method not in ABLATION_METHODS:
        raise ValueError(args.method)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    problem, ref_system, acc_system, manifest = _build_final_problem(
        2, device, Path(args.manifest)
    )
    out = (
        Path(args.out_root)
        / "ablation"
        / "R2"
        / f"seed{args.seed}"
        / args.method
    )
    out.mkdir(parents=True, exist_ok=True)
    masks, mask_metadata = _masks(problem)
    _save_masks_and_magnitude(out, problem, masks)
    _, signal_info = _signal_reconstruction(acc_system, problem.ksp_acc)
    reconstruction = _physical_reconstruction(acc_system)
    ref_reconstruction = _physical_reconstruction(ref_system)
    sums = {
        arm: torch.zeros(problem.image.shape, dtype=torch.float64, device=device)
        for arm in ("ref", "acc")
    }
    # Data-space probes are drawn per arm because the two arms' k-spaces differ
    # in shape; image-space probes are shared between arms, as in the matched
    # stage, so that the ablation reports the same two-arm relative-noise ratio.
    data_space = args.method in ("gaussian_pmr", "random_phase_factor")
    generators = (
        {
            "ref": torch.Generator(device=device).manual_seed(args.seed + 401),
            "acc": torch.Generator(device=device).manual_seed(args.seed + 509),
        }
        if data_space
        else {"method": torch.Generator(device=device).manual_seed(args.seed + 601)}
    )
    diagnostics: dict[str, list[tuple[int, float, bool, int]]] = {
        "reference": [],
        "accelerated": [],
    }
    state_path = out / "resume_state.npz"
    count, elapsed, sample_times = _load_state(
        state_path,
        device=device,
        sums=sums,
        generators=generators,
        diagnostics=diagnostics,
    )
    solves_per_sample = 2 if data_space else 4   # both arms
    progress = _read_csv(out / "checkpoint_summary.csv")
    _atomic_json(
        out / "config.json",
        {
            **_metadata(args),
            "R": 2,
            "seed": args.seed,
            "method": args.method,
            "estimator": {
                "gaussian_pmr": "mean |Rz|^2; proper complex Gaussian data probes",
                "random_phase_factor": "mean |Rz|^2; unit-modulus data probes",
                "random_phase_direct": "mean Re(conj(v) * RR^H v); unit phases",
                "real_rademacher_direct": "mean Re(conj(v) * RR^H v); real +/-1",
                "proper_gaussian_direct": "mean Re(conj(v) * RR^H v); proper Gaussian",
            }[args.method],
            "raw_real_direct_estimator": True,
            "negative_entry_diagnostics": True,
            "N_max": args.N_max,
            "checkpoints": DENSE_CHECKPOINTS,
            "max_eigen_acc": acc_system.max_eigen,
            "lambda_normalized_acc": acc_system.lambda_normalized,
            "max_eigen_ref": ref_system.max_eigen,
            "lambda_normalized_ref": ref_system.lambda_normalized,
            "arms": "reference and accelerated; secondary output is the regularized relative-noise ratio",
            "manifest": manifest,
            "mask_metadata": mask_metadata,
            "signal_solver_diagnostics": signal_info,
        },
    )
    checkpoints = set(DENSE_CHECKPOINTS)
    print(
        f"[ablation] method={args.method} resume_N={count} target={args.N_max}",
        flush=True,
    )
    for sample in range(count + 1, args.N_max + 1):
        _sync(device)
        started = time.perf_counter()
        if data_space:
            ref_contribution, ref_infos, _ = _ablation_sample(
                args.method, ref_reconstruction, problem, generators["ref"],
                probe_shape="ref",
            )
            acc_contribution, acc_infos, _ = _ablation_sample(
                args.method, reconstruction, problem, generators["acc"],
                probe_shape="acc",
            )
        else:
            ref_contribution, ref_infos, probe = _ablation_sample(
                args.method, ref_reconstruction, problem, generators["method"],
            )
            acc_contribution, acc_infos, _ = _ablation_sample(
                args.method, reconstruction, problem, generators["method"],
                probe=probe,
            )
        _validate_infos(ref_infos, args.method)
        _validate_infos(acc_infos, args.method)
        sums["ref"] += ref_contribution
        sums["acc"] += acc_contribution
        diagnostics["reference"].extend(_info_values(info) for info in ref_infos)
        diagnostics["accelerated"].extend(_info_values(info) for info in acc_infos)
        _sync(device)
        sample_elapsed = time.perf_counter() - started
        sample_times.append(sample_elapsed)
        elapsed += sample_elapsed
        count = sample
        save_state = (
            sample % args.checkpoint_every == 0
            or sample in checkpoints
            or sample == args.N_max
            or _STOP_REQUESTED
        )
        if save_state:
            _save_state(
                state_path,
                count=count,
                elapsed=elapsed,
                sums=sums,
                generators=generators,
                diagnostics=diagnostics,
                sample_times=sample_times,
            )
        if sample in checkpoints:
            products = physical_variance_products(
                sum_ref=sums["ref"].cpu().numpy(),
                sum_acc=sums["acc"].cpu().numpy(),
                count=sample,
                acceleration=2,
            )
            _save_variance_products(out, f"N{sample}", products)
            variance = (sums["acc"] / sample).cpu().numpy()
            _atomic_npy(out / f"N{sample}_variance_acc_physical_raw.npy", variance)
            _atomic_npy(
                out / f"N{sample}_sd_acc_physical_clipped.npy",
                np.sqrt(np.maximum(variance, 0.0)),
            )
            _atomic_npy(
                out / f"N{sample}_variance_acc_physical_sigma0p01_raw.npy",
                variance * NOISE_SIGMA_FOR_DISPLAY**2,
            )
            _atomic_npy(
                out / f"N{sample}_sd_acc_physical_sigma0p01_clipped.npy",
                np.sqrt(np.maximum(variance, 0.0)) * NOISE_SIGMA_FOR_DISPLAY,
            )
            row = {
                "method": args.method,
                "N": sample,
                "cumulative_wall_time_s": elapsed,
                "mean_sample_wall_time_s": float(np.mean(sample_times)),
                "total_cg_iterations": int(
                    sum(value[0] for values in diagnostics.values() for value in values)
                ),
                "maximum_true_relative_residual": float(
                    max(value[1] for values in diagnostics.values() for value in values)
                ),
            }
            for scope, mask in (
                ("full", np.ones(variance.shape, dtype=bool)),
                ("mask_B", masks["B"]),
                ("mask_C", masks["C"]),
            ):
                stats = negative_diagonal_stats(torch.tensor(variance[mask]))
                for key, value in stats.items():
                    row[f"{scope}_{key}"] = value
            progress = [
                existing for existing in progress if int(existing["N"]) != sample
            ] + [row]
            progress.sort(key=lambda value: int(value["N"]))
            _atomic_csv(out / "checkpoint_summary.csv", progress)
            _atomic_csv(
                out / "solver_diagnostics.csv",
                _diagnostic_rows(
                    diagnostics,
                    sample_times,
                    solves_per_sample=solves_per_sample,
                    method=args.method,
                ),
            )
            print(
                f"[ablation] method={args.method} N={sample} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
        if _STOP_REQUESTED:
            raise RuntimeError("checkpointed after scheduler signal")
    _atomic_json(
        out / "status.json",
        {"status": "complete", "N": count, "method": args.method},
    )


def _signal_stability_rows(
    *,
    arm: str,
    system: System,
    kspace: torch.Tensor,
    reference_image: torch.Tensor,
    mask_B: torch.Tensor,
) -> tuple[list[dict], dict[int, torch.Tensor]]:
    rhs = system.A.adjoint(kspace.type(complex_dtype)) / float(system.max_eigen)
    _, info = solve_cg_true_residual(
        system.normal,
        rhs,
        max_iter=max(STABILITY_ITERATIONS),
        rtol=0.0,
        true_residual_every=TRUE_RESIDUAL_EVERY,
        snapshot_iters=STABILITY_ITERATIONS,
    )
    snapshots = info["snapshots"]
    rows = []
    previous = None
    previous_objective = None
    for iteration in STABILITY_ITERATIONS:
        image = snapshots[iteration]
        residual = float(
            torch.linalg.norm(rhs - system.normal(image))
            / torch.linalg.norm(rhs)
        )
        data_residual = system.A.forward(image) - kspace
        objective = float(
            0.5 * data_residual.abs().double().square().sum()
            + 0.5
            * COMMON_LAMBDA_PHYSICAL
            * image.abs().double().square().sum()
        )
        solution_change = (
            float(torch.linalg.norm(image - previous) / torch.linalg.norm(image))
            if previous is not None
            else float("nan")
        )
        objective_change = (
            abs(objective - previous_objective) / max(abs(objective), 1e-30)
            if previous_objective is not None
            else float("nan")
        )
        aligned_denominator = torch.sum(
            image[mask_B].conj() * image[mask_B]
        ).real.clamp(min=1e-30)
        alpha = (
            torch.sum(image[mask_B].conj() * reference_image[mask_B])
            / aligned_denominator
        )
        image_nrmse = float(
            torch.linalg.norm(
                alpha * image[mask_B] - reference_image[mask_B]
            )
            / torch.linalg.norm(reference_image[mask_B])
        )
        rows.append(
            {
                "arm": arm,
                "iteration": iteration,
                "true_relative_normal_residual": residual,
                "solution_relative_change": solution_change,
                "objective_physical": objective,
                "objective_relative_change": objective_change,
                "scale_aligned_image_nrmse_mask_B": image_nrmse,
            }
        )
        previous = image
        previous_objective = objective
    return rows, snapshots


def run_stability(args: argparse.Namespace) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    problem = build_problem(args.R, device)
    ref_system = System(
        problem.A_ref,
        problem.max_eigen_ref,
        normalized_lambda(COMMON_LAMBDA_PHYSICAL, problem.max_eigen_ref),
        MAX_ITERATIONS,
        0.0,
        TRUE_RESIDUAL_EVERY,
    )
    acc_system = System(
        problem.A_acc,
        problem.max_eigen_acc,
        normalized_lambda(COMMON_LAMBDA_PHYSICAL, problem.max_eigen_acc),
        MAX_ITERATIONS,
        0.0,
        TRUE_RESIDUAL_EVERY,
    )
    out = Path(args.out_root) / "stability" / f"R{args.R}"
    out.mkdir(parents=True, exist_ok=True)
    masks, mask_metadata = _masks(problem)
    _save_masks_and_magnitude(out, problem, masks)
    mask_B = torch.tensor(masks["B"], device=device)
    signal_rows = []
    ref_rows, ref_snapshots = _signal_stability_rows(
        arm="reference",
        system=ref_system,
        kspace=problem.ksp_ref,
        reference_image=problem.image,
        mask_B=mask_B,
    )
    acc_rows, acc_snapshots = _signal_stability_rows(
        arm="accelerated",
        system=acc_system,
        kspace=problem.ksp_acc,
        reference_image=problem.image,
        mask_B=mask_B,
    )
    signal_rows.extend(ref_rows)
    signal_rows.extend(acc_rows)
    _atomic_csv(out / "signal_stability.csv", signal_rows)
    for arm, snapshots in (
        ("reference", ref_snapshots),
        ("accelerated", acc_snapshots),
    ):
        for iteration, image in snapshots.items():
            _atomic_npy(
                out / f"reconstruction_{arm}_iter{iteration}_physical.npy",
                image.cpu().numpy(),
            )
            _atomic_npy(
                out / f"magnitude_{arm}_iter{iteration}_physical.npy",
                image.abs().cpu().numpy(),
            )

    previous_ratio = None
    map_rows = []
    for cap in STABILITY_ITERATIONS:
        ref_cap = System(
            problem.A_ref,
            problem.max_eigen_ref,
            ref_system.lambda_normalized,
            cap,
            0.0,
            TRUE_RESIDUAL_EVERY,
        )
        acc_cap = System(
            problem.A_acc,
            problem.max_eigen_acc,
            acc_system.lambda_normalized,
            cap,
            0.0,
            TRUE_RESIDUAL_EVERY,
        )
        ref_reconstruction = _physical_reconstruction(ref_cap)
        acc_reconstruction = _physical_reconstruction(acc_cap)
        generator = torch.Generator(device=device).manual_seed(args.seed)
        sum_ref = torch.zeros(
            problem.image.shape, dtype=torch.float64, device=device
        )
        sum_acc = torch.zeros_like(sum_ref)
        ref_residuals = []
        acc_residuals = []
        for _ in range(args.stability_probes):
            probe = random_phase(
                problem.image.shape,
                generator=generator,
                device=device,
                dtype=problem.image.dtype,
            )
            ref_image, ref_infos = ref_reconstruction.apply_direct(probe)
            acc_image, acc_infos = acc_reconstruction.apply_direct(probe)
            sum_ref += direct_real_contribution(probe, ref_image)
            sum_acc += direct_real_contribution(probe, acc_image)
            ref_residuals.extend(
                float(info["true_relative_residual"]) for info in ref_infos
            )
            acc_residuals.extend(
                float(info["true_relative_residual"]) for info in acc_infos
            )
        products = physical_variance_products(
            sum_ref=sum_ref.cpu().numpy(),
            sum_acc=sum_acc.cpu().numpy(),
            count=args.stability_probes,
            acceleration=args.R,
        )
        _save_variance_products(out, f"pico_iter{cap}", products)
        ratio = products["regularized_relative_noise_ratio"]
        changes = {}
        for name, mask in masks.items():
            if previous_ratio is None:
                changes[name] = float("nan")
            else:
                changes[name] = float(
                    np.linalg.norm((ratio - previous_ratio)[mask])
                    / max(np.linalg.norm(ratio[mask]), 1e-30)
                )
        map_rows.append(
            {
                "iteration": cap,
                "n_probes": args.stability_probes,
                "regularized_ratio_relative_change_mask_B": changes["B"],
                "regularized_ratio_relative_change_mask_C": changes["C"],
                "reference_max_probe_true_relative_residual": max(ref_residuals),
                "accelerated_max_probe_true_relative_residual": max(acc_residuals),
            }
        )
        previous_ratio = ratio
        _atomic_csv(out / "variance_map_stability.csv", map_rows)
        print(
            f"[stability] R={args.R} cap={cap} "
            f"ratio_change_B={changes['B']:.4g}",
            flush=True,
        )
    decision = evaluate_stability(signal_rows, map_rows)
    record = {
        "R": args.R,
        "max_eigen_ref": problem.max_eigen_ref,
        "max_eigen_acc": problem.max_eigen_acc,
        "lambda_physical": COMMON_LAMBDA_PHYSICAL,
        "lambda_normalized_ref": ref_system.lambda_normalized,
        "lambda_normalized_acc": acc_system.lambda_normalized,
        "reference_passed": decision["passed"],
        "accelerated_passed": decision["passed"],
        "decision": decision,
        "mask_metadata": mask_metadata,
        **_metadata(args),
    }
    _atomic_json(out / "stability_record.json", record)
    _atomic_json(
        out / "status.json",
        {"status": "complete" if decision["passed"] else "failed_gate", **decision},
    )
    if not decision["passed"]:
        raise RuntimeError(f"R={args.R} failed common-lambda stability gate")


def run_gate(args: argparse.Namespace) -> None:
    records = []
    for R in (2, 3, 4):
        path = Path(args.out_root) / "stability" / f"R{R}" / "stability_record.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        with open(path, encoding="utf-8") as stream:
            records.append(json.load(stream))
    manifest = build_operator_manifest(records, COMMON_LAMBDA_PHYSICAL)
    manifest.update(
        {
            "created_unix": time.time(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "protocol": _metadata(args),
        }
    )
    _atomic_json(Path(args.manifest), manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    if not manifest["accepted"]:
        raise RuntimeError("common physical lambda failed at least one operator")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=["stability", "gate", "reference", "matched", "ablation"]
    )
    parser.add_argument("--R", type=int, choices=(2, 3, 4), default=2)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--N-max", type=int, default=10000)
    parser.add_argument("--method", choices=ABLATION_METHODS)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--stability-probes", type=int, default=16)
    parser.add_argument(
        "--matched-methods",
        nargs="+",
        choices=["pmr_gaussian", "pico_random_phase_direct"],
        help="run only these arms of the matched stage (default: both).",
    )
    parser.add_argument(
        "--arm-tag",
        default="",
        help="suffix appended to the per-method output directory, for runs that "
             "vary the arm-sampling protocol (e.g. _independent).",
    )
    parser.add_argument(
        "--coupled-arms",
        dest="coupled_arms",
        action="store_true",
        default=True,
        help="drive both arms with the same randomness (default): one probe for both "
             "arms for PICO, and the shot subset of one noise realization for PMR.",
    )
    parser.add_argument(
        "--independent-arms",
        dest="coupled_arms",
        action="store_false",
        help="draw each arm's randomness independently.",
    )
    parser.add_argument(
        "--pmr-sigma-scale",
        type=float,
        default=1.0,
        help="multiply the injected PMR noise level by this factor (replica estimator only). "
             "The normalized map is independent of the level for a linear reconstruction; the "
             "level only sets whether replica statistics are resolvable in single precision.",
    )
    parser.add_argument(
        "--pmr-estimator",
        choices=["noise_only", "replica"],
        default="noise_only",
        help=(
            "PMR arm of the matched stage. noise_only: accumulate E|R(n)|^2 (exact for a "
            "linear R). replica: PMR as defined by Robson et al., the voxelwise mean and "
            "variance across reconstructions of ksp + n (Welford, float64). The reference "
            "stage is unaffected."
        ),
    )
    parser.add_argument("--out-root", default=str(OUT_ROOT))
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    args = parser.parse_args()
    if args.checkpoint_every not in (500, 1000):
        raise ValueError("checkpoint-every must be 500 or 1000")
    return args


def main() -> None:
    signal.signal(signal.SIGUSR1, _request_stop)
    args = parse_args()
    print(
        f"[final-grid] mode={args.mode} R={args.R} seed={args.seed} "
        f"job={os.environ.get('SLURM_JOB_ID')} cuda={torch.cuda.is_available()}",
        flush=True,
    )
    if args.mode == "stability":
        run_stability(args)
    elif args.mode == "gate":
        run_gate(args)
    elif args.mode == "reference":
        run_reference(args)
    elif args.mode == "matched":
        run_matched(args)
    elif args.mode == "ablation":
        run_ablation(args)


if __name__ == "__main__":
    main()
