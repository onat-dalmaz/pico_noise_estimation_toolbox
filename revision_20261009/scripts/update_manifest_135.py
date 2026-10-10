#!/usr/bin/env python
"""Bump reproducibility_manifest.json to 1.3.5 (2026-09-27): main Figures 2-7 are drawn on the
original submission's Inkscape documents (MRM-AMA-MainManuscript/Archive.zip) so that they have
the original figures' style, by scripts/original_style/build_fig{2..7}.py with the shared toolkit
scripts/original_style/svgkit.py (Inkscape 1.3.2, sans-serif rendered as Verdana).  Each script
regenerates its figure deterministically from the same data as before into
MRM-AMA-MainManuscript/figs/original_style/fig<k>/ (with NOTES.md); the paper assets are copies of
those PNGs, the PDFs of Figures 6 and 7 are exported from the same SVGs, and the previous
matplotlib renderings are in figs/archive_pre_20260927_origstyle/.  Figures 6 and 7 share one
noise-map window (scripts/original_style/cs_noise_window.py), as the original's did.  Figure 6 shows
the original submission's slices 13, 17, 21 with their measured runtimes (the manuscript-configuration
CS recheck, slurm_scripts/submit_cs_fig6_orig_slices.slurm), and Figure 7 shows the five levels
sigma_0 ... 100 sigma_0 in the original's five columns.  Idempotent.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "reproducibility_manifest.json"
FIGS = "MRM-AMA-MainManuscript/figs"
ARCHIVE = f"{FIGS}/archive_pre_20260927_origstyle"
KIT = "scripts/original_style/svgkit.py"
WINDOW = "scripts/original_style/cs_noise_window.py"

# figure key -> (asset name, template inside MRM-AMA-MainManuscript/Archive.zip, extra sources)
FIGURES = {
    "figure_2": ("knee_ours_vs_pmr", "figures/figure_2.svg", []),
    "figure_3": ("phantom_ours_vs_pmr_paper", "figures/figure_3.svg", []),
    "figure_4": ("quantitative_R", "figures/quant_R.svg", ["reports/measured_time_to_target.json"]),
    "figure_5": ("ablation", "figures/ablation/NRMSE_vs_N_probe_comparison.svg", []),
    "figure_6": ("cs_ours_vs_pmr_ctv", "figures/ours_vs_pmr_cs.svg", [WINDOW]),
    "figure_7": ("cs_linearity_ctv", "results/cs_results/noise linearity.svg",
                 [WINDOW, f"{FIGS}/cs_linearity_ctv_metrics.json"]),
}

STYLE_NOTE = (
    "Figure style of the 2026-09-27 build: main Figures 2-7 are drawn on the original submission's "
    "Inkscape documents (MRM-AMA-MainManuscript/Archive.zip) by scripts/original_style/build_fig2.py ... "
    "build_fig7.py, which replace the embedded panels, data-dependent labels and plots of each "
    "template with the revised data and export the original's element at the original's dpi with "
    "Inkscape 1.3.2 and sans-serif rendered as Verdana (scripts/original_style/svgkit.py). Each script "
    "regenerates its figure deterministically from the data files listed under the figure into "
    "MRM-AMA-MainManuscript/figs/original_style/fig<k>/ (SVG, PNG, NOTES.md with the data, windows, "
    "labels and every deviation from the original layout); the paper assets figs/<name>.png are "
    "copies of those PNGs and figs/cs_ours_vs_pmr_ctv.pdf and figs/cs_linearity_ctv.pdf are exported "
    "from the same SVGs. Figures 6 and 7 share one noise-map window, chosen from the pooled maps of "
    "both figures (scripts/original_style/cs_noise_window.py: 1.0 to 2.5), as the original's shared "
    "1.0 to 3.0. The previous matplotlib renderings are in figs/archive_pre_20260927_origstyle/; "
    "scripts/generate_final_revision_figures.py, scripts/make_cs_figure.py and "
    "scripts/compose_cs_noise_figure_ctv.py still write those renderings under the same names."
)


def cs_figures(m) -> None:
    """Figure 6: the original's slices 13, 17, 21 with runtimes; Figure 7: the five levels shown."""
    fig6 = json.load(open(ROOT / FIGS / "cs_ours_vs_pmr_ctv_metrics.json"))
    rows = fig6["rows"]
    f6 = m["main_figures"]["figure_6"]
    f6["sources"] = (["scripts/cs_recheck_pico_pmr.py", "slurm_scripts/submit_cs_fig6_orig_slices.slurm",
                      "src/mr_recon/prox_complex_tv.py"]
                     + [f"{r['source']}/metrics.json" for r in rows]
                     + ["experiments/fastmri_knee/data/file1000000.h5", "scripts/original_style/build_fig6.py", KIT,
                        WINDOW, f"{FIGS}/cs_ours_vs_pmr_ctv_metrics.json"])
    f6["slices"] = [r["slice"] for r in rows]
    f6["note"] = (
        "The original submission's slices " + ", ".join(str(r["slice"]) for r in rows) + " (Slice 1-3). Complex-valued "
        "TV proximal operator (Chambolle tau = 1/8, 50 iterations, no exit test), FISTA fixed at 30 iterations, "
        "lambda_TV = 3e-7 (SSIM maximum on slices 12/20/28); N = " + f"{rows[0]['n_pico']}" + " per estimator; "
        "std-map NRMSE on the support " + "/".join(f"{r['std_map_nrmse_support_pct']:.2f}" for r in rows) + "%; "
        "wall time per slice on one " + f"{rows[0]['device']}" + " (shown in the figure) PICO "
        + "/".join(f"{r['t_pico_s']:.1f}" for r in rows) + " s, PMR " + "/".join(f"{r['t_pmr_s']:.1f}" for r in rows)
        + " s, from the manuscript-configuration recheck (scripts/cs_recheck_pico_pmr.py) that gives the Results' "
        "N = 2000 wall times on slice 20 (experiments/pmr_audit/cs_recheck_ctv/R2_noise1/metrics.json).")
    f7 = m["main_figures"]["figure_7"]
    f7["note"] = ("Five noise levels shown, sigma_0, 5, 10, 50 and 100 sigma_0, in the original's five columns; the "
                  "sweep's 200 sigma_0 level (experiments/fastmri_knee/results/cs_nonlinear_R2_ctv/noise_200) is not "
                  "shown.")


def main() -> None:
    m = json.load(open(PATH))
    cs_figures(m)
    m["manifest_version"] = "1.3.5"
    m["last_audited"] = date.today().isoformat()
    m["notes"] = [n for n in m["notes"] if "Figure style of the 2026-09-27 build" not in n] + [STYLE_NOTE]
    for key, (name, template, extra) in FIGURES.items():
        f = m["main_figures"][key]
        f["style"] = {
            "build": f"scripts/original_style/build_fig{key[-1]}.py",
            "template": f"MRM-AMA-MainManuscript/Archive.zip:{template}",
            "outputs": f"{FIGS}/original_style/fig{key[-1]}/",
            "previous_rendering": f"{ARCHIVE}/{name}.png",
        }
        for src in [f"scripts/original_style/build_fig{key[-1]}.py", KIT] + extra:
            if src not in f["sources"]:
                f["sources"].append(src)
    op = m["main_figures"]["figure_4"].get("operator", "")
    clause = "; both estimators use shared realizations for the two reconstructions (Section S10)"
    if clause in op:
        m["main_figures"]["figure_4"]["operator"] = op.split(clause)[0] + clause
    archived = m["release_readiness"].setdefault("superseded_assets_archived_at", [])
    for name, _, _ in FIGURES.values():
        p = f"{ARCHIVE}/{name}.png"
        if p not in archived:
            archived.append(p)
    json.dump(m, open(PATH, "w"), indent=2)
    print(f"manifest {m['manifest_version']} audited {m['last_audited']}; notes {len(m['notes'])}")


if __name__ == "__main__":
    main()
