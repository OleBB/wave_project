"""
All-data damping scatter (CH05 supplementary)
==============================================

The thesis headline OUT/IN figures use only `meta_results` (two
validated lowrange folders, cond4, 2026-03-26/27) and are scoped to
1.3–1.6 Hz where both wind conditions exist. That's the right cut for
the primary claim.

This supplementary figure shows OUT/IN (FFT) for **every** quality-ok,
full-panel wave run ever recorded — across all four hardware conditions
(cond1 h272/high, cond2 h136/high, cond3 h100/high WRONG, cond4 h100/low)
and the legacy November-2025 probe configuration (in=9373/250,
out=12400/170). Purpose: cross-condition pattern check. Does the
OUT/IN curve shape depend on the hardware configuration? If cond1 and
cond4 overlap, that's additional evidence that the cond4 headline
result generalises.

Run from repo root:
    conda run -n draumkvedet python analysis_scratch/all_data_damping_scatter.py

Outputs:
    analysis_scratch/all_data_damping_scatter.pdf     (scratch quick-view)
    output/FIGURES/ch05_damping_all_data_scatter.pdf   (thesis supplementary)
    output/TEXFIGU/ch05_damping_all_data_scatter.tex   (tex stub, write-once)
    analysis_scratch/all_data_damping_scatter_summary.csv (per-condition counts)
"""

import sys
import warnings
from pathlib import Path
from datetime import datetime
import glob

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import os
os.chdir(BASE)

from wavescripts.improved_data_loader import load_analysis_data
from wavescripts.constants import PROBE_HEIGHT_DEFAULT_MM
from wavescripts.plot_utils import freq_to_k, add_freq_axis

# ── I/O ────────────────────────────────────────────────────────────────────────
SCRATCH_PDF = Path(__file__).parent / "all_data_damping_scatter.pdf"
SCRATCH_CSV = Path(__file__).parent / "all_data_damping_scatter_summary.csv"
THESIS_NAME = "ch05_damping_all_data_scatter"
OUT_PDF = BASE / "output" / "FIGURES" / f"{THESIS_NAME}.pdf"
OUT_STUB = BASE / "output" / "TEXFIGU" / f"{THESIS_NAME}.tex"
CHAPTER = "05"

# ── 1. Load everything ─────────────────────────────────────────────────────────
print("1. Loading all processed folders …")
all_dirs = sorted(glob.glob(str(BASE / "waveprocessed" / "PROCESSED-*")))
print(f"   {len(all_dirs)} folders")
meta, _, _, _ = load_analysis_data(*all_dirs, load_processed=False)
print(f"   {len(meta)} total rows")

# ── 2. Classify and filter ─────────────────────────────────────────────────────
def assign_condition(row):
    # Legacy Nov-2025 probe config uses in=9373/250 instead of 9373/170 —
    # detect via in_position column (or via file_date if needed).
    in_pos = row.get("in_position", None)
    if in_pos == "9373/250":
        return "legacy_nov2025"
    h = row.get("probe_height_mm", PROBE_HEIGHT_DEFAULT_MM)
    r = row.get("probe_range_mode", "high")
    if pd.isna(h):
        h = PROBE_HEIGHT_DEFAULT_MM
    h = int(h)
    if h == 272 and r == "high":
        return "cond1_h272_high"
    if h == 136 and r == "high":
        return "cond2_h136_high"
    if h == 100 and r == "high":
        return "cond3_h100_high_WRONG"
    if h == 100 and r == "low":
        return "cond4_h100_low"
    return "other"

meta["condition"] = meta.apply(assign_condition, axis=1)

# Require: wave run, full panel, quality_flag==ok, OUT/IN(FFT) populated
wave = meta[
    meta["WaveFrequencyInput [Hz]"].notna()
    & (meta["WaveFrequencyInput [Hz]"] > 0)
    & (meta["PanelCondition"] == "full")
    & (meta["quality_flag"] == "ok")
    & meta["OUT/IN (FFT)"].notna()
].copy()

# Clamp extreme outliers (> 2.0 or < 0.1) for plot readability — these are
# almost certainly wind-contamination or low-SNR artefacts. Flag in summary.
n_extreme = ((wave["OUT/IN (FFT)"] > 2.0) | (wave["OUT/IN (FFT)"] < 0.1)).sum()
wave_clip = wave[(wave["OUT/IN (FFT)"] <= 2.0) & (wave["OUT/IN (FFT)"] >= 0.1)].copy()
print(f"   {len(wave)} wave/fullpanel/quality=ok runs  ({n_extreme} extreme outliers clipped)")

# Restrict wind conditions to {no, full} — lowest/other is rare and muddies
wave_clip = wave_clip[wave_clip["WindCondition"].isin(["no", "full"])].copy()
print(f"   {len(wave_clip)} after restricting to wind ∈ {{no, full}}")

# Compute k per row (same tank depth as plot_utils)
wave_clip["k"] = freq_to_k(wave_clip["WaveFrequencyInput [Hz]"].values)

print("\n2. Counts per condition × wind:")
pivot = wave_clip.groupby(["condition", "WindCondition"]).size().unstack(fill_value=0)
print(pivot.to_string())

# ── 3. Save summary CSV ────────────────────────────────────────────────────────
summary = (wave_clip.groupby(["condition", "WindCondition", "PanelCondition"])
                     .agg(n=("path", "count"),
                          freq_min=("WaveFrequencyInput [Hz]", "min"),
                          freq_max=("WaveFrequencyInput [Hz]", "max"),
                          out_in_mean=("OUT/IN (FFT)", "mean"),
                          out_in_std=("OUT/IN (FFT)", "std"))
                     .reset_index())
summary.to_csv(SCRATCH_CSV, index=False)
print(f"   Summary → {SCRATCH_CSV.relative_to(BASE)}")

# ── 4. Plot ───────────────────────────────────────────────────────────────────
# Colors per condition (reuse the §3b palette + a neutral gray for legacy)
COND_ORDER = [
    "legacy_nov2025",
    "cond1_h272_high",
    "cond2_h136_high",
    "cond3_h100_high_WRONG",
    "cond4_h100_low",
]
COND_COLOR = {
    "legacy_nov2025":        "#7F8C8D",  # gray
    "cond1_h272_high":       "#2ECC71",  # green
    "cond2_h136_high":       "#F1C40F",  # yellow
    "cond3_h100_high_WRONG": "#E74C3C",  # red
    "cond4_h100_low":        "#3498DB",  # blue (thesis scope)
}
COND_LABEL = {
    "legacy_nov2025":        "Nov-2025 innledende oppsett",
    "cond1_h272_high":       "Høyde 272mm (high)",
    "cond2_h136_high":       "Høyde 136mm (high)",
    "cond3_h100_high_WRONG": "Høyde 100mm (high) (ustabil)",
    "cond4_h100_low":        "Høyde 100mm (low) ← resultatene",
}
# Wind condition → marker
WIND_MARKER = {"no": "o", "full": "^"}
WIND_LABEL  = {"no": "no wind (circles)", "full": "full wind (triangles)"}

# Amplitude → size
AMP_SIZE = {0.1: 18, 0.2: 45, 0.3: 90, 0.6: 140}

fig, ax = plt.subplots(figsize=(11, 6))

for cond in COND_ORDER:
    sub = wave_clip[wave_clip["condition"] == cond]
    if sub.empty:
        continue
    for wind, marker in WIND_MARKER.items():
        s = sub[sub["WindCondition"] == wind]
        if s.empty:
            continue
        sizes = s["WaveAmplitudeInput [Volt]"].map(
            lambda a: AMP_SIZE.get(round(float(a), 1), 30)
        )
        ax.scatter(
            s["k"], s["OUT/IN (FFT)"],
            c=COND_COLOR[cond],
            marker=marker,
            s=sizes,
            alpha=0.60,
            edgecolors="black",
            linewidths=0.3,
            zorder=3,
        )

# Highlight the thesis scope band (1.3–1.6 Hz) with a light axvspan — convert
# to k using the same dispersion relation.
thesis_k_lo = float(freq_to_k(np.array([1.3]))[0])
thesis_k_hi = float(freq_to_k(np.array([1.6]))[0])
ax.axvspan(thesis_k_lo, thesis_k_hi,
           color="#3498DB", alpha=0.08, lw=0, zorder=1,
           label=None)
ax.text(thesis_k_hi - 0.1, 1.95,
        "Hovedfokus \n1,3–1,6 Hz", ha="right", va="top",
        fontsize=8, color="#1F618D", alpha=0.8,
        bbox=dict(boxstyle="round,pad=0.2",
                  facecolor="white", alpha=0.75, edgecolor="none"))

ax.axhline(1.0, color="black", lw=0.6, ls="--", alpha=0.5)

ax.set_xlabel("$k$ (rad/m)", fontsize=11)
ax.set_ylabel("Ut/Inn (FFT)", fontsize=11)
# ax.set_title(
#     "All bølgekjøringer (full panel, quality=ok) — supplementary cross-condition view",
#     fontsize=11, fontweight="bold",
# )
ax.grid(True, alpha=0.25, lw=0.5)
ax.set_ylim(0.1, 2.0)
add_freq_axis(ax)

# Legend — two sections: condition (colour) + wind (marker). Amplitude size
# is documented in the caption rather than legended (legend would get huge).
cond_handles = [
    mlines.Line2D([], [], color=COND_COLOR[c],
                  marker="o", linestyle="None", markersize=7,
                  label=COND_LABEL[c])
    for c in COND_ORDER
    if (wave_clip["condition"] == c).any()
]
wind_handles = [
    mlines.Line2D([], [], color="gray",
                  marker=m, linestyle="None", markersize=7,
                  label=WIND_LABEL[w])
    for w, m in WIND_MARKER.items()
]
first = ax.legend(handles=cond_handles, loc="upper left",
                  fontsize=8, framealpha=0.92, title="Probehøyde og innstilling")
ax.add_artist(first)
ax.legend(handles=wind_handles, loc="upper right",
          fontsize=8, framealpha=0.92, title="vind")

fig.text(0.5, 0.01,
         f"n = {len(wave_clip)} kjøringer (2025-10 til 2026-03). "#f"Blue band highlights the tight thesis scope (cond4 only, nowind+fullwind present at every freq).", #f"Marker size ∝ input amplitude (0.1/0.2/0.3 V). "
         ha="center", fontsize=8, color="#444", style="italic")

fig.subplots_adjust(left=0.08, right=0.98, top=0.85, bottom=0.14)

# ── 5. Save ────────────────────────────────────────────────────────────────────
print("\n3. Saving figure …")
SCRATCH_PDF.parent.mkdir(parents=True, exist_ok=True)
OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
OUT_STUB.parent.mkdir(parents=True, exist_ok=True)

fig.savefig(SCRATCH_PDF, bbox_inches="tight")
print(f"   Saved → {SCRATCH_PDF.relative_to(BASE)}")
fig.savefig(OUT_PDF, bbox_inches="tight")
print(f"   Saved → {OUT_PDF.relative_to(BASE)}")

# Stub: write once. Re-running preserves user edits to caption/label.
if not OUT_STUB.exists():
    n_total = len(wave_clip)
    n_by_cond = {c: int((wave_clip["condition"] == c).sum()) for c in COND_ORDER}
    cond_count_str = ", ".join(
        f"{c.replace('_', r' ')}: {n}"
        for c, n in n_by_cond.items() if n > 0
    ).replace(r"WRONG", r"WRONG")

    _caption = (
        f"OUT/IN (FFT) versus $k$ (rad/m) for all {n_total} quality-ok full-panel "
        "wave runs across the full experimental record. Colour encodes the "
        "hardware configuration (probe height $\\times$ range mode); marker "
        "encodes wind condition (circles = no wind, triangles = full wind); "
        "size encodes input amplitude ($0.1$/$0.2$/$0.3$\\,V). The blue-"
        "shaded band marks the thesis scope (cond4, $1.3$--$1.6$\\,Hz) where "
        "both wind conditions are present at every frequency. Data outside "
        "that band is informative for cross-condition pattern-checking but "
        "is not used for the primary results. Extreme outliers (OUT/IN $>2$ "
        "or $<0.1$) from low-SNR wind-contamination are clipped from the "
        "y-axis for readability; see CLAUDE.md \\S16 on the two competing "
        "biases in FFT OUT/IN under fullwind."
    )
    _stub = (
        "%! TEX root = ../main.tex\n"
        "% =============================================================\n"
        "% IMMUTABLE — generated automatically, do not edit this block\n"
        "%   script          : analysis_scratch/all_data_damping_scatter.py\n"
        "%   plot_type       : damping_all_data_scatter\n"
        f"%   chapter         : {CHAPTER}\n"
        f"%   n_runs_total    : {n_total}\n"
        f"%   conditions      : {', '.join(c for c, n in n_by_cond.items() if n > 0)}\n"
        "% =============================================================\n"
        "\\begin{figure}[htbp]\n"
        "  \\centering\n"
        f"  \\includegraphics[width=0.95\\linewidth]{{FIGURES/{THESIS_NAME}.pdf}}\n"
        "  \\caption[All-data OUT/IN scatter — cross-condition view]{%\n"
        f"    {_caption}\n"
        "  }\n"
        f"  \\label{{fig:{THESIS_NAME}}}\n"
        "\\end{figure}\n"
    )
    OUT_STUB.write_text(_stub)
    print(f"   Wrote stub → {OUT_STUB.relative_to(BASE)}")
else:
    print(f"   Stub exists (not overwritten): {OUT_STUB.relative_to(BASE)}")

print("\nDone.")
