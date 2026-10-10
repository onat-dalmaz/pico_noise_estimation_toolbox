"""Pure helpers for the final corrected non-Cartesian run grid."""

from __future__ import annotations

import math
import os
from collections.abc import Sequence

import numpy as np
import torch


# The published grid uses 0.049014.  A regeneration at another common physical
# Tikhonov strength sets NONCART_FINAL_LAMBDA_PHYSICAL (and
# NONCART_FINAL_OUT_ROOT / NONCART_FINAL_REPORT_DIR) in the environment of every
# stage, so the original grid, its manifest, and its reports are never touched.
COMMON_LAMBDA_PHYSICAL = float(os.environ.get("NONCART_FINAL_LAMBDA_PHYSICAL", "0.049014"))
DENSE_CHECKPOINTS = (
    10,
    20,
    30,
    40,
    50,
    100,
    200,
    500,
    1000,
    2000,
    5000,
    10000,
)
REFERENCE_CHECKPOINTS = (10000, 20000, 30000, 50000)
MAX_ITERATIONS = 1200
TRUE_RESIDUAL_RTOL = 1e-5
TRUE_RESIDUAL_EVERY = 10


def final_grid_submission_plan() -> list[dict]:
    """Return the dependency-ordered final grid, including resume passes."""
    return [
        {
            "name": "stability",
            "runner": "gpu",
            "mode": "stability",
            "array": "0-2%3",
            "dependency": None,
        },
        {
            "name": "gate",
            "runner": "post",
            "mode": "gate",
            "array": None,
            "dependency": ("afterany", ["stability"]),
        },
        {
            "name": "reference_primary",
            "runner": "gpu",
            "mode": "reference",
            "array": "0-5%3",
            "dependency": ("afterok", ["gate"]),
        },
        {
            "name": "reference_resume",
            "runner": "gpu",
            "mode": "reference",
            "array": "0-5%3",
            "dependency": ("afterany", ["reference_primary"]),
        },
        {
            "name": "matched_primary",
            "runner": "gpu",
            "mode": "matched",
            "array": "0-2%3",
            "dependency": ("afterok", ["gate"]),
        },
        {
            "name": "matched_resume",
            "runner": "gpu",
            "mode": "matched",
            "array": "0-2%3",
            "dependency": ("afterany", ["matched_primary"]),
        },
        {
            "name": "ablation_primary",
            "runner": "gpu",
            "mode": "ablation",
            "array": "0-4%3",
            "dependency": ("afterok", ["gate"]),
        },
        {
            "name": "ablation_resume",
            "runner": "gpu",
            "mode": "ablation",
            "array": "0-4%3",
            "dependency": ("afterany", ["ablation_primary"]),
        },
        {
            "name": "candidate_analysis",
            "runner": "post",
            "mode": "analysis",
            "array": None,
            "dependency": (
                "afterok",
                ["reference_resume", "matched_resume", "ablation_resume"],
            ),
        },
    ]


def conservative_reference_floor(
    independent_map_nrmse: float,
    worst_split_half_nrmse: float,
) -> float:
    """Estimate the pooled two-stream floor using the audited conservative rule."""
    return max(
        float(independent_map_nrmse) / 2.0,
        float(worst_split_half_nrmse) / (2.0 * math.sqrt(2.0)),
    )


def normalized_lambda(lambda_physical: float, max_eigen: float) -> float:
    """Convert a physical Tikhonov strength through one operator scaling."""
    lambda_physical = float(lambda_physical)
    max_eigen = float(max_eigen)
    if not math.isfinite(lambda_physical) or lambda_physical < 0.0:
        raise ValueError("lambda_physical must be finite and nonnegative")
    if not math.isfinite(max_eigen) or max_eigen <= 0.0:
        raise ValueError("max_eigen must be finite and positive")
    return lambda_physical / max_eigen


def physical_variance_products(
    *,
    sum_ref,
    sum_acc,
    count: int,
    acceleration: int,
) -> dict[str, np.ndarray]:
    """Pool physical variances before deriving SD and the secondary ratio."""
    if count <= 0:
        raise ValueError("count must be positive")
    if acceleration <= 0:
        raise ValueError("acceleration must be positive")
    variance_ref = np.asarray(sum_ref, dtype=np.float64) / float(count)
    variance_acc = np.asarray(sum_acc, dtype=np.float64) / float(count)
    sd_ref = np.sqrt(np.maximum(variance_ref, 0.0))
    sd_acc = np.sqrt(np.maximum(variance_acc, 0.0))
    ratio = np.divide(
        sd_acc,
        np.sqrt(float(acceleration)) * sd_ref,
        out=np.full(sd_acc.shape, np.nan, dtype=np.float64),
        where=sd_ref > 0.0,
    )
    return {
        "variance_ref_physical": variance_ref,
        "variance_acc_physical": variance_acc,
        "sd_ref_physical": sd_ref,
        "sd_acc_physical": sd_acc,
        "regularized_relative_noise_ratio": ratio,
    }


def _real_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.complex64:
        return torch.float32
    if dtype == torch.complex128:
        return torch.float64
    raise TypeError(f"Expected complex dtype, received {dtype}")


def real_rademacher(
    shape: Sequence[int],
    *,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Draw a real Rademacher probe represented in the requested complex dtype."""
    bits = torch.randint(
        0,
        2,
        tuple(shape),
        generator=generator,
        device=device,
        dtype=torch.int8,
    )
    return (2 * bits.to(_real_dtype(dtype)) - 1).to(dtype)


def build_operator_manifest(
    records: Sequence[dict],
    lambda_physical: float,
    *,
    reference_relative_tolerance: float = 1e-6,
) -> dict:
    """Combine R-specific stability records into one canonical operator manifest."""
    by_r = {int(record["R"]): record for record in records}
    if set(by_r) != {2, 3, 4} or len(records) != 3:
        raise ValueError("stability records must contain exactly R=2,3,4")

    reference_eigenvalues = np.asarray(
        [float(by_r[R]["max_eigen_ref"]) for R in (2, 3, 4)],
        dtype=np.float64,
    )
    canonical_reference = float(reference_eigenvalues.mean())
    spread = float(
        np.max(np.abs(reference_eigenvalues - canonical_reference))
        / max(abs(canonical_reference), 1e-30)
    )
    if spread > reference_relative_tolerance:
        raise ValueError(
            "reference eigenvalue scalings disagree beyond tolerance: "
            f"relative spread={spread:.6g}"
        )

    arms = {
        "reference": {
            "max_eigen": canonical_reference,
            "lambda_normalized": normalized_lambda(
                lambda_physical, canonical_reference
            ),
            "stability_passed": bool(
                all(by_r[R]["reference_passed"] for R in (2, 3, 4))
            ),
        }
    }
    for R in (2, 3, 4):
        max_eigen = float(by_r[R]["max_eigen_acc"])
        arms[f"R{R}"] = {
            "max_eigen": max_eigen,
            "lambda_normalized": normalized_lambda(
                lambda_physical, max_eigen
            ),
            "stability_passed": bool(by_r[R]["accelerated_passed"]),
        }

    accepted = bool(all(arm["stability_passed"] for arm in arms.values()))
    return {
        "accepted": accepted,
        "lambda_physical": float(lambda_physical),
        "operator": "unweighted pre-whitened A^H A",
        "normalization": "lambda_normalized=lambda_physical/max_eigen",
        "reference_eigen_relative_spread": spread,
        "arms": arms,
        "source_records": list(records),
    }


def evaluate_stability(
    signal_rows: Sequence[dict],
    map_rows: Sequence[dict],
) -> dict:
    """Apply the audit's residual, iterate, objective, and map stability gates."""
    signal_by_key = {
        (str(row["arm"]), int(row["iteration"])): row for row in signal_rows
    }
    checked = []
    first_stable = None
    for map_row in sorted(map_rows, key=lambda row: int(row["iteration"])):
        iteration = int(map_row["iteration"])
        reference = signal_by_key.get(("reference", iteration))
        accelerated = signal_by_key.get(("accelerated", iteration))
        stable = bool(reference is not None and accelerated is not None)
        for row in (reference, accelerated):
            if row is None:
                continue
            stable &= bool(
                np.isfinite(row["true_relative_normal_residual"])
                and row["true_relative_normal_residual"] <= 1e-5
                and np.isfinite(row["solution_relative_change"])
                and row["solution_relative_change"] <= 1e-3
                and np.isfinite(row["objective_relative_change"])
                and row["objective_relative_change"] <= 1e-5
            )
        stable &= bool(
            np.isfinite(
                map_row["regularized_ratio_relative_change_mask_B"]
            )
            and map_row["regularized_ratio_relative_change_mask_B"] <= 2e-2
            and np.isfinite(
                map_row["reference_max_probe_true_relative_residual"]
            )
            and map_row["reference_max_probe_true_relative_residual"] <= 1e-5
            and np.isfinite(
                map_row["accelerated_max_probe_true_relative_residual"]
            )
            and map_row["accelerated_max_probe_true_relative_residual"] <= 1e-5
        )
        checked.append({"iteration": iteration, "passed": stable})
        if stable and first_stable is None:
            first_stable = iteration
    return {
        "passed": first_stable is not None,
        "first_stable_iteration": first_stable,
        "checkpoint_decisions": checked,
        "gates": {
            "true_relative_normal_residual": "<=1e-5",
            "solution_relative_change": "<=1e-3",
            "objective_relative_change": "<=1e-5",
            "regularized_ratio_relative_change_mask_B": "<=2e-2",
        },
    }
