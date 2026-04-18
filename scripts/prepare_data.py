"""Prepare single-slice data bundles for the three PICO demonstration notebooks.

This script extracts representative slices from the raw ``.h5`` files used in
Dalmaz et al., "Fast Voxelwise SNR Estimation for Iterative MRI
Reconstructions" (§3.1, §4), bundles them as small ``.npz`` artifacts under
``experiments/*/data/``, and copies pre-computed high-replica reference maps
into the bundles so the companion notebooks can run in minutes rather than
hours.

Three bundles are produced:

1. ``experiments/cartesian_knee/data/slice120_R2.npz``
   - Stanford knee subject (mridata.org, §3.1), hybrid-space slice 120.
   - Keys: ``img_ref`` (fully-sampled reference image),
     ``mps`` (coil sensitivities), ``R`` (= 2),
     ``gfactor_analytical`` (closed-form SENSE g-factor from
     ``mr_recon.gfactor.gfactor_sense``).

2. ``experiments/noncartesian_phantom/data/slice_R2_bundle.npz``
   - Physical brain phantom (§3.1), spiral trajectory, single slice.
   - Keys: ``img_ref``, ``mps``, ``trj``, ``dcf``, ``ksp`` (fully-sampled),
     ``gfactor_pmr_ref`` (PMR surrogate reference at ``N_ref = 30000``,
     §4.2, Appendix D), ``nrmse_csv`` (pre-computed NRMSE-vs-N curves for
     PICO and PMR; basis for the quantitative figure).

3. ``experiments/compressed_sensing/data/slice017_R2.npz``
   - fastMRI knee corpus (§3.1, §4.3), slice 17, R = 2 Poisson-disc with a
     24 x 24 calibration region.
   - Keys: ``ksp_full`` (fully-sampled k-space -- undersampling is
     re-applied at load time so the mask is reproducible),
     ``mps``, ``mask`` (Poisson-disc sampling mask, seed locked),
     ``img_rss`` (root-sum-of-squares reference image),
     ``pico_var_gold`` (PICO variance gold reference, N = 10000,
     from ``experiments/fastmri_knee/results/.../ref_ours_var_N10000.npy``),
     ``pmr_var_gold`` (PMR variance gold reference, N = 10000, likewise).

The script is idempotent: if a bundle already exists and is non-empty, it is
left in place unless ``--force`` is passed. Run from the repo root.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _log(msg: str) -> None:
    print(f"[prepare_data] {msg}", flush=True)


def _save_or_skip(out_path: Path, force: bool, payload: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not force:
        _log(f"{out_path} already exists; skipping (use --force to overwrite).")
        return
    np.savez_compressed(out_path, **payload)
    _log(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")


# ---------------------------------------------------------------------------
# 1. Cartesian knee (Stanford, §4.1)
# ---------------------------------------------------------------------------


def prepare_cartesian_knee(force: bool = False) -> Path:
    """Extract slice 120 of the Stanford knee subject and compute the
    closed-form SENSE g-factor as the analytical reference.

    Following the manuscript (§4.1): R = 2 uniform undersampling along the
    phase-encoding direction (Rx = 1, Ry = 2). The raw ``.h5`` stores
    fully-sampled images and ESPIRiT coil sensitivities per slice; we slice
    both at index 120 and compute the closed-form g-factor locally.
    """
    import torch

    from mr_recon.gfactor import gfactor_sense

    raw_h5 = REPO_ROOT / "experiments/cartesian_knee/data/efa383b6-9446-438a-9901-1fe951653dbd.h5"
    out_path = REPO_ROOT / "experiments/cartesian_knee/data/slice120_R2.npz"
    if out_path.exists() and not force:
        _log(f"{out_path} already exists; skipping (use --force to overwrite).")
        return out_path
    if not raw_h5.exists():
        raise FileNotFoundError(
            f"Missing raw Stanford knee file: {raw_h5}\n"
            "Place the mridata.org subject HDF5 there before running this script."
        )

    slice_num = 120
    Rx, Ry = 1, 2

    _log(f"reading {raw_h5} ...")
    with h5py.File(raw_h5, "r") as f:
        img_ref = np.asarray(f["target"][slice_num, :, :, 0])            # (H, W)
        mps_np = np.asarray(f["maps"][slice_num, :, :, :, 0])            # (H, W, C)

    mps_np = np.transpose(mps_np, (2, 0, 1))                             # (C, H, W)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mps_t = torch.as_tensor(mps_np, dtype=torch.complex64, device=dev)
    gmap = gfactor_sense(mps_t, Rx, Ry, l2_reg=0.0).cpu().numpy().astype(np.float32)

    payload = {
        "img_ref": img_ref.astype(np.complex64),
        "mps": mps_np.astype(np.complex64),
        "gfactor_analytical": gmap,
        "R": np.int32(Rx * Ry),
        "Rx": np.int32(Rx),
        "Ry": np.int32(Ry),
        "slice_index": np.int32(slice_num),
        "source": np.str_("mridata.org Stanford knee (Epperson 2013); subject efa383b6"),
    }
    _save_or_skip(out_path, force=True, payload=payload)
    return out_path


# ---------------------------------------------------------------------------
# 2. Non-Cartesian spiral phantom (§4.2)
# ---------------------------------------------------------------------------


def prepare_noncartesian_phantom(force: bool = False) -> Path:
    """The phantom bundle is already laid out as separate ``.npy`` files in
    ``experiments/noncartesian_phantom/data/``; we wrap them into a single
    ``.npz`` plus a pre-computed PMR reference drawn from the existing
    high-N sweep in ``results/quantitative_accuracy_incremental/`` so the
    companion notebook does not need to re-run the 30 000-replica reference.
    """
    data_dir = REPO_ROOT / "experiments/noncartesian_phantom/data"
    out_path = data_dir / "slice_R2_bundle.npz"
    if out_path.exists() and not force:
        _log(f"{out_path} already exists; skipping.")
        return out_path

    img = np.load(data_dir / "img.npy")
    mps = np.load(data_dir / "mps.npy")
    ksp = np.load(data_dir / "ksp.npy")
    trj = np.load(data_dir / "trj.npy")
    dcf = np.load(data_dir / "dcf.npy")
    evals = np.load(data_dir / "evals.npy")

    ref_dir = REPO_ROOT / "experiments/noncartesian_phantom/results/quantitative_accuracy_incremental/inv_g/R2_L0.1"
    ref_anim = ref_dir / "animation_data_incremental"

    gref_path = ref_anim / "g_ref.npy"
    csv_path = ref_dir / "N_comparison_metrics_incremental.csv"

    if not gref_path.exists():
        raise FileNotFoundError(
            f"Missing high-N surrogate reference: {gref_path}\n"
            "Re-run the §4.2 R=2 sweep (see scripts/run_N_comparison.py) or provide it."
        )
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing NRMSE-vs-N CSV: {csv_path}\nExpected from the §4.2 R=2 sweep."
        )

    gfactor_pmr_ref = np.load(gref_path).astype(np.float32)
    with open(csv_path, "r") as fh:
        nrmse_csv_text = fh.read()

    canonical_N = [50, 200, 1000, 5000, 10000]
    map_payload = {}
    for n in canonical_N:
        pico_map = ref_anim / f"g_Diag_N{n}.npy"
        pmr_map = ref_anim / f"g_PMR_N{n}.npy"
        if not (pico_map.exists() and pmr_map.exists()):
            raise FileNotFoundError(
                f"Missing canonical PICO/PMR map for N={n} at {pico_map} / {pmr_map}"
            )
        map_payload[f"pico_map_N{n}"] = np.load(pico_map).astype(np.float32)
        map_payload[f"pmr_map_N{n}"] = np.load(pmr_map).astype(np.float32)

    payload = {
        "img_ref": img.astype(np.complex64),
        "mps": mps.astype(np.complex64),
        "ksp": ksp.astype(np.complex64),
        "trj": trj.astype(np.float32),
        "dcf": dcf.astype(np.float32),
        "evals": evals.astype(np.float32),
        "gfactor_pmr_ref": gfactor_pmr_ref,
        "nrmse_csv": np.str_(nrmse_csv_text),
        "canonical_N": np.asarray(canonical_N, dtype=np.int32),
        "R": np.int32(2),
        "lamda_l2": np.float32(0.1),
        "N_ref": np.int32(30000),
        "source": np.str_("GE Ultra High Performance 3T, spiral GRE brain phantom (§3.1)"),
        **map_payload,
    }
    _save_or_skip(out_path, force=True, payload=payload)
    return out_path


# ---------------------------------------------------------------------------
# 3. Compressed-sensing fastMRI knee (§4.3)
# ---------------------------------------------------------------------------


def prepare_compressed_sensing(force: bool = False) -> Path:
    """Extract slice 17 of the fastMRI knee volume, regenerate the
    Poisson-disc mask with the same seed the manuscript experiments used,
    and ship the N = 10 000 PICO and PMR gold variance references from
    the existing §4.3 run ``final_comparison_accmatch_vmin1_vmax3``.
    """
    from sigpy.mri import poisson

    raw_h5 = REPO_ROOT / "experiments/fastmri_knee/data/file1000000.h5"
    out_path = REPO_ROOT / "experiments/compressed_sensing/data/slice017_R2.npz"
    if out_path.exists() and not force:
        _log(f"{out_path} already exists; skipping.")
        return out_path
    if not raw_h5.exists():
        raise FileNotFoundError(
            f"Missing raw fastMRI knee file: {raw_h5}\n"
            "Place the fastMRI ``.h5`` subject (file1000000.h5) there before running this script."
        )

    slice_idx = 17
    R = 2
    calib = (24, 24)
    mask_seed = 1234

    _log(f"reading slice {slice_idx} from {raw_h5} ...")
    with h5py.File(raw_h5, "r") as f:
        img_rss = np.asarray(f["reconstruction_rss"][slice_idx])                    # (H, W)
        mps_np = np.asarray(f["jsense-12-cf=4"]["maps"][slice_idx, :, :, :, 0])      # (H, W, C)
        ksp_full = np.asarray(f["kspace"][slice_idx])                                # (H, W, C)

    mps_np = np.transpose(mps_np, (2, 0, 1))                                          # (C, H, W)
    ksp_full = np.transpose(ksp_full, (2, 0, 1))                                      # (C, H, W)

    im_size = img_rss.shape
    np.random.seed(mask_seed)
    mask = poisson(im_size, accel=R, calib=calib, tol=0.3, crop_corner=False)
    mask = np.asarray(mask.real > 0, dtype=bool)                                      # (H, W)

    ref_dir = REPO_ROOT / "experiments/fastmri_knee/results/final_comparison_accmatch_vmin1_vmax3/slice0017"
    pico_gold_path = ref_dir / "ref_ours_var_N10000.npy"
    pmr_gold_path = ref_dir / "ref_pmr_var_N10000.npy"
    summary_path = ref_dir / "summary.json"

    for p in [pico_gold_path, pmr_gold_path, summary_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing CS reference artifact: {p}")

    pico_var_gold = np.load(pico_gold_path).astype(np.float32)
    pmr_var_gold = np.load(pmr_gold_path).astype(np.float32)
    with open(summary_path, "r") as fh:
        summary = json.load(fh)

    payload = {
        "ksp_full": ksp_full.astype(np.complex64),
        "mps": mps_np.astype(np.complex64),
        "mask": mask,
        "img_rss": img_rss.astype(np.float32),
        "pico_var_gold": pico_var_gold,
        "pmr_var_gold": pmr_var_gold,
        "mask_seed": np.int32(mask_seed),
        "R": np.int32(R),
        "calib": np.asarray(calib, dtype=np.int32),
        "sigma_k": np.float32(1e-7),
        "lamda_tv": np.float32(1e-2),
        "fista_iters": np.int32(100),
        "N_gold": np.int32(10000),
        "ref_summary": np.str_(json.dumps(summary, indent=2)),
        "source": np.str_("fastMRI knee corpus (Zbontar 2018), subject file1000000"),
    }
    _save_or_skip(out_path, force=True, payload=payload)
    return out_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=["cartesian", "noncartesian", "cs", "all"],
        default="all",
        help="Which bundle(s) to prepare.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing bundles.")
    args = parser.parse_args(argv)

    targets = []
    if args.only in ("cartesian", "all"):
        targets.append(("cartesian knee", prepare_cartesian_knee))
    if args.only in ("noncartesian", "all"):
        targets.append(("non-cartesian phantom", prepare_noncartesian_phantom))
    if args.only in ("cs", "all"):
        targets.append(("compressed sensing", prepare_compressed_sensing))

    for label, fn in targets:
        _log(f"=== preparing {label} ===")
        path = fn(force=args.force)
        _log(f"    -> {path}")

    _log("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
