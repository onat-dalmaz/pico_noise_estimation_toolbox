"""Recompute non-Cartesian accuracy metrics from saved full-resolution maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
import numpy as np
import pandas as pd
from scipy import ndimage

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = (
    ROOT
    / "experiments/noncartesian_phantom/results/"
    "quantitative_accuracy_incremental/inv_g"
)
DEFAULT_OUT = ROOT / "reports/noncart_validity_audit"


def build_masks(magnitude: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, dict]]:
    """Build the three prespecified audit masks from a reference magnitude image."""
    mag = np.asarray(magnitude, dtype=np.float64)
    positive = mag[mag > 0]
    if positive.size == 0:
        raise ValueError("reference magnitude contains no positive pixels")

    percentile_threshold = float(np.percentile(positive, 97.5))
    current = mag > percentile_threshold
    support = mag > 0.05 * float(mag.max())

    labels, n_components = ndimage.label(
        support, structure=np.ones((3, 3), dtype=np.uint8)
    )
    component_sizes = np.bincount(labels.ravel())
    component_sizes[0] = 0
    largest = labels == int(component_sizes.argmax())
    filled = ndimage.binary_fill_holes(largest)
    manual = ndimage.binary_erosion(
        filled, structure=np.ones((3, 3), dtype=bool), iterations=1
    )

    masks = {
        "A_current_percentile": current,
        "B_support_5pct": support,
        "C_lcc_fill_erode1": manual,
    }
    metadata = {
        "A_current_percentile": {
            "rule": "|x_ref| > percentile_97.5(|x_ref| over positive pixels)",
            "threshold": percentile_threshold,
            "pixel_count": int(current.sum()),
        },
        "B_support_5pct": {
            "rule": "|x_ref| > 0.05 * max(|x_ref|)",
            "threshold": 0.05 * float(mag.max()),
            "pixel_count": int(support.sum()),
        },
        "C_lcc_fill_erode1": {
            "rule": (
                "largest 8-connected component of B; fill holes; "
                "one 3x3 binary erosion"
            ),
            "connectivity": 8,
            "components_before_selection": int(n_components),
            "erosion_iterations": 1,
            "pixel_count": int(manual.sum()),
        },
    }
    return masks, metadata


def masked_nrmse(
    estimate: np.ndarray, reference: np.ndarray, mask: np.ndarray
) -> float:
    estimate = np.asarray(estimate)
    reference = np.asarray(reference)
    valid = np.asarray(mask, dtype=bool) & np.isfinite(estimate) & np.isfinite(reference)
    if not np.any(valid):
        return float("nan")
    numerator = np.linalg.norm(
        (estimate[valid] - reference[valid]).astype(np.float64)
    )
    denominator = np.linalg.norm(reference[valid].astype(np.float64))
    return float(numerator / max(denominator, 1e-30))


def first_crossing(n_values, errors, batch_times, threshold: float):
    cumulative = np.cumsum(np.asarray(batch_times, dtype=np.float64))
    for n_value, error, elapsed in zip(n_values, errors, cumulative):
        if np.isfinite(error) and error < threshold:
            return {
                "N": int(n_value),
                "nrmse": float(error),
                "cumulative_time_s": float(elapsed),
            }
    return None


def masked_correlation(
    estimate: np.ndarray, reference: np.ndarray, mask: np.ndarray
) -> float:
    valid = np.asarray(mask, dtype=bool) & np.isfinite(estimate) & np.isfinite(reference)
    if valid.sum() < 2:
        return float("nan")
    a = np.asarray(estimate[valid], dtype=np.float64)
    b = np.asarray(reference[valid], dtype=np.float64)
    a -= a.mean()
    b -= b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / max(denominator, 1e-30))


def _safe_inverse(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    return np.divide(
        1.0,
        array,
        out=np.full(array.shape, np.nan, dtype=np.float64),
        where=np.isfinite(array) & (array > 1e-12),
    )


def _save_masks_visual(
    magnitude: np.ndarray, masks: Dict[str, np.ndarray], path: Path
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(magnitude, cmap="gray")
    axes[0].set_title("|x_ref|")
    for axis, (name, mask) in zip(axes[1:], masks.items()):
        axis.imshow(mask, cmap="gray", vmin=0, vmax=1)
        axis.set_title(f"{name}\n{int(mask.sum()):,} pixels")
    for axis in axes:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _main_curve_specs(R: int):
    return {
        "PICO_random_phase": {
            "map_stem": "Diag",
            "nrmse_column": "NRMSE_Diag",
            "time_column": "Time_Diag",
            "provenance": "exact method=all saved realization",
        },
        "PMR": {
            "map_stem": "PMR",
            "nrmse_column": "NRMSE_PMR",
            "time_column": "Time_PMR",
            "provenance": (
                "probe-run overwrite; not method=all Figure-4 realization"
                if R == 2
                else "exact method=all saved realization"
            ),
        },
    }


def _probe_curve_specs():
    return {
        "PICO_random_phase": {
            "map_stem": "Complex_Rademacher",
            "nrmse_column": "NRMSE_Complex_Rademacher",
            "time_column": "Time_Complex_Rademacher",
        },
        "PICO_real_rademacher": {
            "map_stem": "Real_Rademacher",
            "nrmse_column": "NRMSE_Real_Rademacher",
            "time_column": "Time_Real_Rademacher",
        },
        "PICO_complex_gaussian": {
            "map_stem": "Complex_Gaussian",
            "nrmse_column": "NRMSE_Complex_Gaussian",
            "time_column": "Time_Complex_Gaussian",
        },
        "PMR": {
            "map_stem": "PMR",
            "nrmse_column": "NRMSE_PMR",
            "time_column": "Time_PMR",
        },
    }


def _curve_rows(
    *,
    curve: str,
    R: int,
    lambda_normalized: float,
    base: Path,
    csv_path: Path,
    specs: dict,
    masks: Dict[str, np.ndarray],
):
    frame = pd.read_csv(csv_path)
    animation = base / "animation_data_incremental"
    reference_stored = np.load(animation / "g_ref.npy")
    rows = []
    provenance_rows = []
    cumulative_by_method = {
        method: np.cumsum(frame[spec["time_column"]].to_numpy(dtype=float))
        for method, spec in specs.items()
    }
    for method, spec in specs.items():
        for row_index, csv_row in frame.iterrows():
            N = int(csv_row["N"])
            map_path = animation / f"g_{spec['map_stem']}_N{N}.npy"
            estimate_stored = np.load(map_path)
            for domain, estimate, reference in (
                ("stored_inv_g", estimate_stored, reference_stored),
                (
                    "direct_relative_noise",
                    _safe_inverse(estimate_stored),
                    _safe_inverse(reference_stored),
                ),
            ):
                for mask_name, mask in masks.items():
                    error = masked_nrmse(estimate, reference, mask)
                    rows.append(
                        {
                            "curve": curve,
                            "R": R,
                            "lambda_normalized": lambda_normalized,
                            "method": method,
                            "N": N,
                            "domain": domain,
                            "mask": mask_name,
                            "mask_pixels": int(mask.sum()),
                            "finite_pixels": int(
                                (
                                    mask
                                    & np.isfinite(estimate)
                                    & np.isfinite(reference)
                                ).sum()
                            ),
                            "nrmse": error,
                            "correlation": masked_correlation(
                                estimate, reference, mask
                            ),
                            "batch_time_s": float(csv_row[spec["time_column"]]),
                            "cumulative_time_s": float(
                                cumulative_by_method[method][row_index]
                            ),
                            "map_path": str(map_path.relative_to(ROOT)),
                            "map_provenance": spec.get(
                                "provenance", "exact probe-run saved realization"
                            ),
                        }
                    )
            recomputed_current = masked_nrmse(
                estimate_stored, reference_stored, masks["A_current_percentile"]
            )
            provenance_rows.append(
                {
                    "curve": curve,
                    "R": R,
                    "method": method,
                    "N": N,
                    "csv_nrmse": float(csv_row[spec["nrmse_column"]]),
                    "saved_map_current_mask_nrmse": recomputed_current,
                    "difference": recomputed_current
                    - float(csv_row[spec["nrmse_column"]]),
                    "exact_match_1e-6": abs(
                        recomputed_current - float(csv_row[spec["nrmse_column"]])
                    )
                    <= 1e-6,
                    "provenance": spec.get(
                        "provenance", "exact probe-run saved realization"
                    ),
                }
            )
    return rows, provenance_rows


def _crossing_rows(metrics: pd.DataFrame):
    output = []
    group_columns = ["curve", "R", "domain", "mask", "method"]
    for keys, group in metrics.groupby(group_columns, sort=True):
        group = group.sort_values("N")
        for threshold in (0.01, 0.02, 0.05):
            crossing = first_crossing(
                group["N"].to_numpy(),
                group["nrmse"].to_numpy(),
                group["batch_time_s"].to_numpy(),
                threshold,
            )
            output.append(
                {
                    **dict(zip(group_columns, keys)),
                    "threshold": threshold,
                    "reached": crossing is not None,
                    "crossing_N": None if crossing is None else crossing["N"],
                    "crossing_nrmse": None
                    if crossing is None
                    else crossing["nrmse"],
                    "crossing_cumulative_time_s": None
                    if crossing is None
                    else crossing["cumulative_time_s"],
                    "endpoint_N": int(group["N"].iloc[-1]),
                    "endpoint_nrmse": float(group["nrmse"].iloc[-1]),
                    "total_cumulative_time_s": float(
                        group["batch_time_s"].sum()
                    ),
                }
            )
    return pd.DataFrame(output)


def _timing_ratio_rows(crossings: pd.DataFrame):
    output = []
    main = crossings[crossings["curve"] == "main"]
    keys = ["R", "domain", "mask", "threshold"]
    for values, group in main.groupby(keys, sort=True):
        by_method = group.set_index("method")
        pico = by_method.loc["PICO_random_phase"]
        pmr = by_method.loc["PMR"]
        both = bool(pico["reached"] and pmr["reached"])
        output.append(
            {
                **dict(zip(keys, values)),
                "PICO_reached": bool(pico["reached"]),
                "PMR_reached": bool(pmr["reached"]),
                "PICO_N": pico["crossing_N"],
                "PMR_N": pmr["crossing_N"],
                "PICO_time_s": pico["crossing_cumulative_time_s"],
                "PMR_time_s": pmr["crossing_cumulative_time_s"],
                "PMR_over_PICO_time_ratio": (
                    float(
                        pmr["crossing_cumulative_time_s"]
                        / pico["crossing_cumulative_time_s"]
                    )
                    if both
                    else np.nan
                ),
                "PMR_over_PICO_sample_ratio": (
                    float(pmr["crossing_N"] / pico["crossing_N"])
                    if both
                    else np.nan
                ),
                "R2_PMR_map_provenance_warning": bool(values[0] == 2),
            }
        )
    return pd.DataFrame(output)


def _ablation_ranking_rows(metrics: pd.DataFrame):
    selected = metrics[
        (metrics["curve"] == "probe_ablation")
        & metrics["N"].isin([50, 200, 1000, 5000, 10000])
    ].copy()
    selected["rank_nrmse"] = selected.groupby(
        ["R", "domain", "mask", "N"]
    )["nrmse"].rank(method="min")
    return selected[
        [
            "R",
            "domain",
            "mask",
            "N",
            "method",
            "nrmse",
            "rank_nrmse",
            "cumulative_time_s",
        ]
    ].sort_values(["domain", "mask", "N", "rank_nrmse"])


def _endpoint_rows(metrics: pd.DataFrame):
    last_indices = metrics.groupby(
        ["curve", "R", "domain", "mask", "method"]
    )["N"].idxmax()
    return metrics.loc[last_indices].sort_values(
        ["curve", "R", "domain", "mask", "nrmse"]
    )


def _reference_floor_fit_rows(metrics: pd.DataFrame):
    rows = []
    main = metrics[
        (metrics["curve"] == "main")
        & (metrics["domain"] == "stored_inv_g")
        & (metrics["N"] >= 1000)
    ]
    for (R, mask, method), group in main.groupby(
        ["R", "mask", "method"], sort=True
    ):
        group = group.sort_values("N")
        n_values = group["N"].to_numpy(dtype=np.float64)
        errors = group["nrmse"].to_numpy(dtype=np.float64)
        design = np.column_stack((1.0 / n_values, np.ones_like(n_values)))
        coefficient = np.linalg.lstsq(design, errors**2, rcond=None)[0]
        coefficient = np.maximum(coefficient, 0.0)
        prediction = np.sqrt(design @ coefficient)
        rows.append(
            {
                "R": R,
                "mask": mask,
                "method": method,
                "fit_model": "NRMSE(N)^2 = a/N + floor^2",
                "fit_N_min": int(n_values.min()),
                "fit_N_max": int(n_values.max()),
                "a": float(coefficient[0]),
                "floor_nrmse": float(np.sqrt(coefficient[1])),
                "fit_rmse": float(
                    np.sqrt(np.mean((prediction - errors) ** 2))
                ),
                "interpretation": (
                    "descriptive extrapolation; not a certified reference error"
                ),
            }
        )
    return pd.DataFrame(rows)


def _plot_mask_sensitivity(metrics: pd.DataFrame, out: Path) -> None:
    main = metrics[
        (metrics["curve"] == "main")
        & (metrics["domain"] == "stored_inv_g")
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    colors = {
        "A_current_percentile": "tab:red",
        "B_support_5pct": "tab:blue",
        "C_lcc_fill_erode1": "tab:green",
    }
    for axis, R in zip(axes, (2, 3, 4)):
        panel = main[main["R"] == R]
        for method, linestyle in (("PICO_random_phase", "-"), ("PMR", "--")):
            for mask, color in colors.items():
                group = panel[
                    (panel["method"] == method) & (panel["mask"] == mask)
                ].sort_values("N")
                axis.loglog(
                    group["N"],
                    group["nrmse"] * 100,
                    linestyle=linestyle,
                    color=color,
                    linewidth=1.2,
                    label=f"{method}; {mask}" if R == 2 else None,
                )
        for threshold in (1, 2, 5):
            axis.axhline(threshold, color="0.7", linewidth=0.6)
        axis.set_title(f"R={R}")
        axis.set_xlabel("N")
        axis.grid(True, which="both", alpha=0.2)
    axes[0].set_ylabel("NRMSE (%)")
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "mask_sensitivity_nrmse.png", dpi=200)
    plt.close(fig)


def run(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    image = np.load(ROOT / "experiments/noncartesian_phantom/data/img.npy")
    magnitude = np.abs(image)
    masks, metadata = build_masks(magnitude)
    for name, mask in masks.items():
        _save = out / f"mask_{name}.npy"
        with open(_save, "wb") as f:
            np.save(f, mask)
    _save_masks_visual(magnitude, masks, out / "masks_visual.png")
    pd.DataFrame(
        [{"mask": name, **details} for name, details in metadata.items()]
    ).to_csv(out / "mask_summary.csv", index=False)
    with open(out / "mask_summary.json", "w") as f:
        json.dump(metadata, f, indent=2)

    all_rows = []
    provenance_rows = []
    for R in (2, 3, 4):
        base = RESULT_ROOT / f"R{R}_L0.001_clean_norm_it200"
        rows, provenance = _curve_rows(
            curve="main",
            R=R,
            lambda_normalized=0.001,
            base=base,
            csv_path=base / "N_comparison_metrics_incremental.csv",
            specs=_main_curve_specs(R),
            masks=masks,
        )
        all_rows.extend(rows)
        provenance_rows.extend(provenance)
    base = RESULT_ROOT / "R2_L0.001_clean_norm_it200"
    rows, provenance = _curve_rows(
        curve="probe_ablation",
        R=2,
        lambda_normalized=0.001,
        base=base,
        csv_path=base / "N_comparison_metrics_incremental_probes.csv",
        specs=_probe_curve_specs(),
        masks=masks,
    )
    all_rows.extend(rows)
    provenance_rows.extend(provenance)

    for R in (2, 3, 4):
        base_l0 = RESULT_ROOT / f"R{R}_L0.0_clean_norm_it200"
        rows, provenance = _curve_rows(
            curve="lambda0_main",
            R=R,
            lambda_normalized=0.0,
            base=base_l0,
            csv_path=base_l0 / "N_comparison_metrics_incremental.csv",
            specs=_main_curve_specs(R),
            masks=masks,
        )
        all_rows.extend(rows)
        provenance_rows.extend(provenance)
    base_l0 = RESULT_ROOT / "R2_L0.0_clean_norm_it200"
    rows, provenance = _curve_rows(
        curve="lambda0_probe_ablation",
        R=2,
        lambda_normalized=0.0,
        base=base_l0,
        csv_path=base_l0 / "N_comparison_metrics_incremental_probes.csv",
        specs=_probe_curve_specs(),
        masks=masks,
    )
    all_rows.extend(rows)
    provenance_rows.extend(provenance)

    metrics = pd.DataFrame(all_rows)
    metrics.to_csv(out / "mask_sensitivity.csv", index=False)
    provenance = pd.DataFrame(provenance_rows)
    provenance.to_csv(out / "saved_map_provenance.csv", index=False)
    crossings = _crossing_rows(metrics)
    crossings.to_csv(out / "threshold_crossings.csv", index=False)
    timing = _timing_ratio_rows(crossings)
    timing.to_csv(out / "timing_ratios.csv", index=False)
    rankings = _ablation_ranking_rows(metrics)
    rankings.to_csv(out / "ablation_ranking.csv", index=False)
    endpoints = _endpoint_rows(metrics)
    endpoints.to_csv(out / "endpoint_sensitivity.csv", index=False)
    floor_fits = _reference_floor_fit_rows(metrics)
    floor_fits.to_csv(out / "reference_floor_fit.csv", index=False)

    ratio_rows = []
    for R in (2, 3, 4):
        base = RESULT_ROOT / f"R{R}_L0.001_clean_norm_it200"
        direct = _safe_inverse(
            np.load(base / "animation_data_incremental/g_ref.npy")
        )
        for name, mask in masks.items():
            values = direct[mask & np.isfinite(direct)]
            ratio_rows.append(
                {
                    "R": R,
                    "mask": name,
                    "pixels": int(len(values)),
                    "mean": float(values.mean()),
                    "median": float(np.median(values)),
                    "p01": float(np.quantile(values, 0.01)),
                    "p05": float(np.quantile(values, 0.05)),
                    "p95": float(np.quantile(values, 0.95)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "fraction_below_1": float(np.mean(values < 1)),
                }
            )
    pd.DataFrame(ratio_rows).to_csv(
        out / "reference_ratio_mask_stats.csv", index=False
    )

    current_matches = provenance[
        (provenance["curve"] == "main")
        & (provenance["R"] == 2)
        & (provenance["method"] == "PMR")
    ]["exact_match_1e-6"]
    endpoint_rankings = rankings[rankings["N"] == 10000]
    summary = {
        "mask_counts": {
            name: int(mask.sum()) for name, mask in masks.items()
        },
        "r2_main_pmr_saved_maps_match_figure4_csv": bool(current_matches.all()),
        "r2_main_pmr_saved_maps_match_probe_csv": bool(
            provenance[
                (provenance["curve"] == "probe_ablation")
                & (provenance["method"] == "PMR")
            ]["exact_match_1e-6"].all()
        ),
        "endpoint_ablation_order": {
            f"{domain}|{mask}": group.sort_values("rank_nrmse")["method"].tolist()
            for (domain, mask), group in endpoint_rankings.groupby(
                ["domain", "mask"]
            )
        },
        "outputs": {
            "metrics": "mask_sensitivity.csv",
            "crossings": "threshold_crossings.csv",
            "timing_ratios": "timing_ratios.csv",
            "ablation_ranking": "ablation_ranking.csv",
            "endpoints": "endpoint_sensitivity.csv",
            "provenance": "saved_map_provenance.csv",
            "reference_floor_fit": "reference_floor_fit.csv",
        },
    }
    with open(out / "mask_sensitivity_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    _plot_mask_sensitivity(metrics, out)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    run(Path(args.out))


if __name__ == "__main__":
    main()
