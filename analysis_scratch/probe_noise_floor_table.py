"""
Reader-facing probe noise-floor table — h272/high (initial) vs h100/low (final).
==================================================================================

Companion table for CH04 §1. Strips the intermediate configs (h136/high,
h100/high) to give a simple before/after comparison: the initial setup
(h=272 mm, highrange) vs the final canon setup (h=100 mm, lowrange).

Columns:
    Sonde
    h272/high (innledende) — Deteksjonsterskel (3σ) [mm]
    h100/low (endelig)     — Deteksjonsterskel (3σ) [mm]
    Forbedring             — ratio (innledende / endelig)

Method note: "Deteksjonsterskel (3σ)" = 3 × σ_RMS of the stillwater signal,
mean across accepted runs in each config. This is the smallest single-sample
amplitude change reliably distinguishable from noise at 3σ Gaussian
confidence. The pipeline column is `detection_threshold_mm = max(3σ, 2q)`,
which collapses to 3σ here because the ULS digital quantum is q = 0.01 mm
in both modes (verified against the histogram of raw values across 8
run × probe instances), so 2q = 0.02 mm is always smaller than 3σ.

The earlier table version used (P99.5 − P0.5)/2 ("99 % half-range"). 3σ is
the conventional physics noise-threshold metric and is what later thesis
claims about "smallest reliable amplitude" rest on.

Outputs:
    output/TABLES/ch04_probe_noise_floor_table.tex
    analysis_scratch/probe_noise_floor_table.csv
"""

from __future__ import annotations

import os
import sys
import warnings
from datetime import datetime as _dt
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
os.chdir(BASE)

from wavescripts.improved_data_loader import load_analysis_data
from wavescripts.plotter import plot_probe_noise_floor
from wavescripts.plot_utils import _lookup_central_caption

import glob

THESIS_NAME = "ch04_probe_noise_floor_table"
CHAPTER = "04"
OUT_TEX = BASE / "output" / "TABLES" / f"{THESIS_NAME}.tex"
OUT_CSV = BASE / "analysis_scratch" / f"{THESIS_NAME}.csv"

# Probes in row order — display label tracks the existing thesis vocabulary.
PROBE_ORDER = [
    ("8804/250",  "8804/250 (oppstrøms)"),
    ("9373/170",  "9373/170 (IN, vegg)"),
    ("9373/340",  "9373/340 (IN, fjern)"),
    ("12400/250", "12400/250 (OUT)"),
]

GROUP_INITIAL = "h272 / high"   # innledende
GROUP_FINAL   = "h100 / low"    # endelig

ANALYSIS_PROBES = [p for p, _ in PROBE_ORDER]

# ── Load data ──────────────────────────────────────────────────────────────
dirs = sorted(glob.glob("waveprocessed/PROCESSED-*"))
print(f"Loading {len(dirs)} folders for stillwater pool …")
combined_meta, _, _, _ = load_analysis_data(*dirs, load_processed=False)
print(f"  {len(combined_meta)} runs total")

pv = {
    "filters": {},
    "plotting": {
        "show_plot": False,
        "save_plot": False,
        "draft": True,
        "figure_name": "noise_floor_table_data",
        "force_stub": False,
    },
}
print("Computing per-config noise summary …")
_figs, summary = plot_probe_noise_floor(
    combined_meta, ANALYSIS_PROBES, pv,
    group_by=["probe_height_mm", "probe_range_mode"],
    processed_dfs=None,    # quantization step skipped — we don't need it for this table
)

# Restrict + pivot — using 3σ detection threshold instead of 99 % half-range
keep = summary[summary["group"].isin([GROUP_INITIAL, GROUP_FINAL])]
print(f"\nRows in scope:")
print(keep[["group", "probe", "noise_rms_mm",
             "detection_threshold_mm", "n_runs"]].round(4).to_string(index=False))
print()

# detection_threshold_mm = max(3·σ_RMS, 2q) — pipeline-computed.
# In both modes 3σ ≥ 0.09 mm and q = 0.01 mm, so the 3σ term always dominates.
det = keep.pivot(index="probe", columns="group", values="detection_threshold_mm")
sig = keep.pivot(index="probe", columns="group", values="noise_rms_mm")
nr  = keep.pivot(index="probe", columns="group", values="n_runs")

records = []
for pos, label in PROBE_ORDER:
    d_init = float(det.loc[pos, GROUP_INITIAL]) if pos in det.index else np.nan
    d_fin  = float(det.loc[pos, GROUP_FINAL])   if pos in det.index else np.nan
    s_init = float(sig.loc[pos, GROUP_INITIAL]) if pos in sig.index else np.nan
    s_fin  = float(sig.loc[pos, GROUP_FINAL])   if pos in sig.index else np.nan
    n_init = int(nr.loc[pos, GROUP_INITIAL])    if pos in nr.index and pd.notna(nr.loc[pos, GROUP_INITIAL]) else 0
    n_fin  = int(nr.loc[pos, GROUP_FINAL])      if pos in nr.index and pd.notna(nr.loc[pos, GROUP_FINAL]) else 0
    ratio  = (d_init / d_fin) if (d_fin and np.isfinite(d_fin) and np.isfinite(d_init)) else np.nan
    records.append({
        "probe":          pos,
        "probe_label":    label,
        "thr3sigma_initial_mm": d_init,
        "thr3sigma_final_mm":   d_fin,
        "sigma_initial_mm":     s_init,
        "sigma_final_mm":       s_fin,
        "n_initial":            n_init,
        "n_final":              n_fin,
        "ratio_init_over_final": ratio,
    })
table = pd.DataFrame(records)
print("\nFinal table:")
print(table.round(3).to_string(index=False))
print()

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
table.to_csv(OUT_CSV, index=False)
print(f"CSV → {OUT_CSV.relative_to(BASE)}")

# ── LaTeX rendering ────────────────────────────────────────────────────────
def fmt_mm(v: float) -> str:
    if not np.isfinite(v):
        return "—"
    return rf"$\num{{{v:.2f}}}$"

def fmt_ratio(v: float) -> str:
    if not np.isfinite(v):
        return "—"
    return rf"$\num{{{v:.1f}}}\times$"

body = []
for r in records:
    body.append(
        "    " + " & ".join([
            r["probe_label"],
            fmt_mm(r["thr3sigma_initial_mm"]),
            fmt_mm(r["thr3sigma_final_mm"]),
            fmt_ratio(r["ratio_init_over_final"]),
        ]) + r" \\"
    )

caption_full  = _lookup_central_caption(THESIS_NAME, kind="full")
caption_short = _lookup_central_caption(THESIS_NAME, kind="short")
if caption_full and caption_short:
    caption_block = (
        f"  \\caption[{caption_short}]{{\n"
        f"    {caption_full}\n"
        f"  }}\n"
    )
elif caption_full:
    caption_block = f"  \\caption{{\n    {caption_full}\n  }}\n"
else:
    caption_block = "  \\caption{\n    % TODO: write caption\n  }\n"

n_initial = records[0]["n_initial"]
n_final   = records[0]["n_final"]

immutable = "\n".join([
    "%! TEX root = ../main.tex",
    "% ==============================================================",
    "% IMMUTABLE — generated automatically, do not edit this block",
    "%",
    "% — Provenance ───────────────────────────────────────────────────",
    "%   script            : analysis_scratch/probe_noise_floor_table.py",
    "%   plot_type         : probe_noise_floor_table",
    f"%   chapter           : {CHAPTER}",
    f"%   generated_at      : {_dt.now().isoformat(timespec='seconds')}",
    f"%   caption_label     : tab:{THESIS_NAME}",
    f"%   caption_short     : {caption_short}",
    "%",
    "% — Inputs ────────────────────────────────────────────────────",
    "%   source            : combined_meta stillwater rows",
    "%                       (WindCondition=='no', WaveFrequencyInput is NaN)",
    "%   group_by          : probe_height_mm × probe_range_mode",
    "%   selected groups   : h272/high (innledende), h100/low (endelig)",
    "%   intermediate configs h136/high and h100/high are NOT shown here.",
    "%",
    "% — Counts ────────────────────────────────────────────────────",
    f"%   n stillwater (h272/high): {n_initial}",
    f"%   n stillwater (h100/low) : {n_final}",
    "%",
    "% — Method ────────────────────────────────────────────────────",
    "%   detection threshold = max(3 · σ_RMS, 2 · q)  [pipeline column",
    "%     'detection_threshold_mm'].",
    "%   σ_RMS         = std of stillwater signal per run, then averaged",
    "%                    across accepted stillwater runs in the group.",
    "%   q             = ULS digital quantum = 0.01 mm in BOTH modes",
    "%                    (verified against histogram of raw values across",
    "%                    8 run × probe instances; values are exact",
    "%                    multiples of 0.01 mm and not of any coarser grid).",
    "%   In both modes 3σ ≥ 0.09 mm and 2q = 0.02 mm, so 3σ always dominates.",
    "%   ratio column  = thr3sigma_initial / thr3sigma_final.",
    "%",
    "% — Caveats ───────────────────────────────────────────────────",
    "%   * The four 'sonde' rows are PROBE POSITIONS, not the same physical",
    "%     sensors across configs. Per the march2026_better_rearranging",
    "%     layout (CLAUDE.md §8), each position is held constant; only the",
    "%     hardware height and range mode change between groups.",
    "%   * h272/high pool is the broader pre-2026-03-23 stillwater set;",
    "%     h100/low pool is from the canon Mar 26-27 lowrange dates only.",
    "%   * Numbers in mm; all are well below typical wave amplitudes (5-20 mm).",
    "%",
    "% ── end immutable block ─────────────────────────────────────────",
])

table_body = (
    "\\begin{table}[hbt]\n"
    "  \\centering\n"
    + caption_block
    + f"  \\label{{tab:{THESIS_NAME}}}\n"
    "  \\begin{tabular}{lccc}\n"
    "    \\toprule\n"
    "    Sonde &\n"
    "      h272/high (innledende) &\n"
    "      h100/low (endelig) &\n"
    "      Forbedring \\\\\n"
    "          &\n"
    "      $3\\sigma$ [\\unit{\\milli\\metre}] &\n"
    "      $3\\sigma$ [\\unit{\\milli\\metre}] &\n"
    "          \\\\\n"
    "    \\midrule\n"
    + "\n".join(body) + "\n"
    "    \\bottomrule\n"
    "  \\end{tabular}\n"
    "\\end{table}\n"
)

OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
OUT_TEX.write_text(immutable + "\n" + table_body, encoding="utf-8")
print(f"TEX → {OUT_TEX.relative_to(BASE)}")

print("\nDone.")
