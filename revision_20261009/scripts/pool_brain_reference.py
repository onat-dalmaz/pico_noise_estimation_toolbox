"""Pool the independent PMR reference streams of the in vivo brain spiral
(matched stage, PMR arm only, seeds 20260901-05, 10,000 replicas each) into
one 50,000-replica surrogate reference: average the physical variance maps of
the accelerated and fully sampled arms and recompute the relative-noise ratio
sigma_acc / (sqrt(R) sigma_ref), exactly as run_noncart_weekend_audit does for
a single stream.  Also records the between-stream NRMSE of the ratio map on
mask B (the reference's own precision).
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "experiments/brain_spiral/results"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[20260901, 20260902, 20260903, 20260904, 20260905])
    ap.add_argument("--N", type=int, default=10000)
    ap.add_argument("--R", type=int, default=2)
    ap.add_argument("--out", default=str(RES / "reference_pooled_R2"))
    args = ap.parse_args()
    mask = np.load(RES / "analysis_mask_B.npy").astype(bool)
    acc, ref, ratios = [], [], []
    for s in args.seeds:
        d = RES / f"matched_incremental/R{args.R}/seed{s}"
        acc.append(np.load(d / f"matched_pmr_N{args.N}_var_acc_physical_abs.npy").astype(np.float64))
        ref.append(np.load(d / f"matched_pmr_N{args.N}_var_ref_physical_abs.npy").astype(np.float64))
        ratios.append(np.load(d / f"matched_pmr_N{args.N}_ratio_common_units.npy").astype(np.float64))
    va, vr = np.mean(acc, axis=0), np.mean(ref, axis=0)
    ratio = np.sqrt(np.maximum(va, 0) / np.maximum(vr, 1e-30)) / math.sqrt(args.R)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    N_total = args.N * len(args.seeds)
    np.save(out / f"pmr_N{N_total}_ratio_common_units.npy", ratio)
    np.save(out / f"pmr_N{N_total}_var_acc_physical_abs.npy", va)
    np.save(out / f"pmr_N{N_total}_var_ref_physical_abs.npy", vr)
    pair = []
    for i in range(len(ratios)):
        for j in range(i + 1, len(ratios)):
            pair.append(float(np.linalg.norm((ratios[i] - ratios[j])[mask]) / np.linalg.norm(ratio[mask])))
    stream_vs_pooled = [float(np.linalg.norm((r - ratio)[mask]) / np.linalg.norm(ratio[mask])) for r in ratios]
    meta = {"seeds": args.seeds, "N_per_stream": args.N, "N_total": N_total, "R": args.R, "mask": "B",
            "between_stream_ratio_nrmse": pair, "stream_vs_pooled_ratio_nrmse": stream_vs_pooled,
            "pooled_floor_estimate": float(np.mean(stream_vs_pooled) / math.sqrt(len(ratios) - 1)) if len(ratios) > 1 else None,
            "note": "pooled_floor_estimate: mean stream-vs-pooled NRMSE divided by sqrt(n_streams - 1), the same convention as the phantom reference QC"}
    json.dump(meta, open(out / "pooled_reference.json", "w"), indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
