"""
All-data damping scatter — v2 (recoloured)
==========================================

Variant of all_data_damping_scatter.py with the colour/marker scheme
aligned to the rest of the thesis:

  - Wind condition  → colour (WIND_COLOR_MAP: blue=no, red=full)
  - Amplitude tier  → marker shape (○=A1=0.1V, □=A2=0.2V, △=A3=0.3V)
  - Hardware condition → marker fill (filled = cond4 final, hollow = earlier)

The hardware-condition dimension was the primary colour axis in v1 (5
distinct colours). Here it's downgraded to a fill modifier — most runs
are cond4 (the final hardware), and the few from earlier conditions are
flagged with hollow markers so the figure reads in the same visual
language as ch05_damping_freq, ch05_damping_ka, etc.

Run from repo root:
    conda run -n draumkvedet python analysis_scratch/all_data_damping_scatter_v2.py

Outputs (scratch only — does NOT overwrite the v1 thesis figure):
    analysis_scratch/all_data_damping_scatter_v2.pdf
    analysis_scratch/all_data_damping_scatter_v2_summary.csv

Compare with analysis_scratch/all_data_damping_scatter.pdf and pick one.
The losing file goes into ignore_this_archive/ via the user's normal
archive workflow.
"""

import sys
import warnings
from pathlib import Path
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
from wavescripts.plot_utils import (
    freq_to_k, add_freq_axis, WIND_COLOR_MAP, amp_to_label, amp_to_tag,
)

# ── I/O ────────────────────────────────────────────────────────────────────────
SCRATCH_PDF = Path(__file__).parent / "all_data_damping_scatter_v2.pdf"
SCRATCH_CSV = Path(__file__).parent / "all_data_damping_scatter_v2_summary.csv"

# ── 1. Load everything ─────────────────────────────────────────────────────────
print("1. Loading all processed folders …")
all_dirs = sorted(glob.glob(str(BASE / "waveprocessed" / "PROCESSED-*")))
print(f"   {len(all_dirs)} folders")
meta, _, _, _ = load_analysis_data(*all_dirs, load_processed=False)
print(f"   {len(meta)} total rows")

# ── 2. Classify and filter ─────────────────────────────────────────────────────
def assign_condition(row):
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

wave = meta[
    meta["WaveFrequencyInput [Hz]"].notna()
    & (meta["WaveFrequencyInput [Hz]"] > 0)
    & (meta["PanelCondition"] == "full")
    & (meta["quality_flag"] == "ok")
    & meta["OUT/IN (FFT)"].notna()
].copy()

n_extreme = ((wave["OUT/IN (FFT)"] > 2.0) | (wave["OUT/IN (FFT)"] < 0.1)).sum()
wave_clip = wave[(wave["OUT/IN (FFT)"] <= 2.0) & (wave["OUT/IN (FFT)"] >= 0.1)].copy()
print(f"   {len(wave)} wave/fullpanel/quality=ok runs  ({n_extreme} extreme outliers clipped)")

wave_clip = wave_clip[wave_clip["WindCondition"].isin(["no", "full"])].copy()
print(f"   {len(wave_clip)} after restricting to wind ∈ {{no, full}}")

wave_clip["k"] = freq_to_k(wave_clip["WaveFrequencyInput [Hz]"].values)

print("\n2. Counts per condition × wind:")
pivot = wave_clip.groupby(["condition", "WindCondition"]).size().unstack(fill_value=0)
print(pivot.to_string())

# Mark the "is final hardware" flag — used to choose filled vs hollow marker.
FINAL_CONDITION = "cond4_h100_low"
wave_clip["is_final"] = wave_clip["condition"] == FINAL_CONDITION

print("\n   final-vs-earlier hardware split:")
print(wave_clip.groupby(["is_final", "WindCondition"]).size()
                .unstack(fill_value=0).to_string())

# ── 3. Save summary CSV (same fields as v1, plus is_final) ─────────────────────
summary = (wave_clip.groupby(["condition", "is_final", "WindCondition", "PanelCondition"])
                     .agg(n=("path", "count"),
                          freq_min=("WaveFrequencyInput [Hz]", "min"),
                          freq_max=("WaveFrequencyInput [Hz]", "max"),
                          out_in_mean=("OUT/IN (FFT)", "mean"),
                          out_in_std=("OUT/IN (FFT)", "std"))
                     .reset_index())
summary.to_csv(SCRATCH_CSV, index=False)
print(f"\n   Summary → {SCRATCH_CSV.relative_to(BASE)}")

# ── 4. Plot ───────────────────────────────────────────────────────────────────
# Wind condition → colour (thesis-wide WIND_COLOR_MAP).
WIND_LABEL = {"no": "uten vind", "full": "med vind"}

# Amplitude → marker (paddle-voltage → tier shape).
AMP_MARKER = {0.10: "o", 0.20: "s", 0.30: "^"}   # circle / square / triangle
AMP_MARKER_DEFAULT = "X"                          # any unexpected V (e.g. 0.6)
MARKER_SIZE = 55                                  # constant — shape carries the info

# Final-vs-earlier hardware → fill style.
#   final (cond4):  fully filled with the wind colour (edge slightly darker)
#   earlier:        hollow (facecolor='none'), edge in the wind colour
ALPHA_FILLED  = 0.65
ALPHA_HOLLOW  = 0.85
EDGE_LW_FILLED = 0.3
EDGE_LW_HOLLOW = 1.4


def _round_amp(v):
    return round(float(v), 2)


fig, ax = plt.subplots(figsize=(11, 6))

# Plot order: earlier hardware first (so the more-numerous final cond4
# markers paint over them — keeps the canonical data on top).
for is_final in [False, True]:
    sub_h = wave_clip[wave_clip["is_final"] == is_final]
    if sub_h.empty:
        continue
    for wind, color in [("no", WIND_COLOR_MAP["no"]),
                        ("full", WIND_COLOR_MAP["full"])]:
        for amp_v, marker in AMP_MARKER.items():
            s = sub_h[(sub_h["WindCondition"] == wind)
                      & (sub_h["WaveAmplitudeInput [Volt]"].apply(_round_amp) == amp_v)]
            if s.empty:
                continue
            if is_final:
                fc = color
                ec = "black"
                lw = EDGE_LW_FILLED
                a  = ALPHA_FILLED
            else:
                fc = "none"
                ec = color
                lw = EDGE_LW_HOLLOW
                a  = ALPHA_HOLLOW
            ax.scatter(
                s["k"], s["OUT/IN (FFT)"],
                facecolors=fc,
                edgecolors=ec,
                marker=marker,
                s=MARKER_SIZE,
                linewidths=lw,
                alpha=a,
                zorder=3 if is_final else 2,
            )

# Catch-all for amplitudes outside {0.1, 0.2, 0.3} (rare — 0.6 V if any).
_recognised_amps = set(AMP_MARKER.keys())
unknown = wave_clip[~wave_clip["WaveAmplitudeInput [Volt]"]
                    .apply(_round_amp).isin(_recognised_amps)]
if not unknown.empty:
    print(f"   note: {len(unknown)} runs with amp ∉ {{0.1, 0.2, 0.3}} V "
          f"plotted as marker '{AMP_MARKER_DEFAULT}'")
    for is_final in [False, True]:
        u = unknown[unknown["is_final"] == is_final]
        if u.empty: continue
        for wind, color in [("no", WIND_COLOR_MAP["no"]),
                            ("full", WIND_COLOR_MAP["full"])]:
            uw = u[u["WindCondition"] == wind]
            if uw.empty: continue
            ax.scatter(uw["k"], uw["OUT/IN (FFT)"],
                       facecolors=(color if is_final else "none"),
                       edgecolors=("black" if is_final else color),
                       marker=AMP_MARKER_DEFAULT,
                       s=MARKER_SIZE,
                       linewidths=(EDGE_LW_FILLED if is_final else EDGE_LW_HOLLOW),
                       alpha=(ALPHA_FILLED if is_final else ALPHA_HOLLOW),
                       zorder=3 if is_final else 2)

# Thesis-scope band (1.3–1.6 Hz) → light blue axvspan in k-space.
thesis_k_lo = float(freq_to_k(np.array([1.3]))[0])
thesis_k_hi = float(freq_to_k(np.array([1.6]))[0])
ax.axvspan(thesis_k_lo, thesis_k_hi,
           color=WIND_COLOR_MAP["no"], alpha=0.07, lw=0, zorder=1)
ax.text(thesis_k_hi - 0.1, 1.27,
        "Hovedfokus\n1,3–1,6 Hz", ha="right", va="top",
        fontsize=8, color="#1F618D", alpha=0.85,
        bbox=dict(boxstyle="round,pad=0.2",
                  facecolor="white", alpha=0.75, edgecolor="none"))

ax.axhline(1.0, color="black", lw=0.6, ls="--", alpha=0.5)
ax.set_xlabel("$k$ (rad/m)", fontsize=11)
ax.set_ylabel("Ut/Inn (FFT)", fontsize=11)
ax.grid(True, alpha=0.25, lw=0.5)
# Cap at 1.3 — points above 1.3 are wind-contamination / low-SNR artefacts
# (CLAUDE.md §16); not informative for the cross-condition pattern check.
Y_TOP = 1.3
ax.set_ylim(0.1, Y_TOP)
n_hidden = int((wave_clip["OUT/IN (FFT)"] > Y_TOP).sum())
if n_hidden:
    ax.text(0.99, 0.99,
            f"+{n_hidden} kjøringer over y={Y_TOP:.1f} (vind-kontaminert, ikke vist)",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=7, color="#666", style="italic",
            bbox=dict(boxstyle="round,pad=0.25",
                      facecolor="white", alpha=0.85, edgecolor="#ccc"))
add_freq_axis(ax)

# ── Legend ────────────────────────────────────────────────────────────────────
# Three small legends, one per dimension.
wind_handles = [
    mlines.Line2D([], [], color=WIND_COLOR_MAP[w],
                  marker="s", linestyle="None", markersize=8,
                  markerfacecolor=WIND_COLOR_MAP[w],
                  markeredgecolor="black", markeredgewidth=0.3,
                  label=WIND_LABEL[w])
    for w in ["no", "full"]
]
amp_handles = [
    mlines.Line2D([], [], color="black",
                  marker=AMP_MARKER[v], linestyle="None", markersize=8,
                  markerfacecolor="lightgray", markeredgecolor="black",
                  markeredgewidth=0.3,
                  label=amp_to_label(v))
    for v in (0.10, 0.20, 0.30)
]
hardware_handles = [
    mlines.Line2D([], [], color="black",
                  marker="o", linestyle="None", markersize=8,
                  markerfacecolor="black", markeredgecolor="black",
                  markeredgewidth=0.3,
                  label="endelig oppsett (h100/low)"),
    mlines.Line2D([], [], color="black",
                  marker="o", linestyle="None", markersize=8,
                  markerfacecolor="none", markeredgecolor="black",
                  markeredgewidth=1.4,
                  label="tidligere oppsett"),
]

leg1 = ax.legend(handles=wind_handles, loc="upper left",
                 fontsize=8, framealpha=0.92, title="Vind", title_fontsize=8)
ax.add_artist(leg1)
leg2 = ax.legend(handles=amp_handles, loc="upper center",
                 fontsize=8, framealpha=0.92, title="Amplitude",
                 title_fontsize=8, bbox_to_anchor=(0.42, 0.99))
ax.add_artist(leg2)
ax.legend(handles=hardware_handles, loc="upper right",
          fontsize=8, framealpha=0.92, title="Maskinvare", title_fontsize=8)

n_total = len(wave_clip)
n_final = int(wave_clip["is_final"].sum())
fig.text(0.5, 0.01,
         f"n = {n_total} kjøringer  ({n_final} fra endelig oppsett, "
         f"{n_total - n_final} fra tidligere oppsett).",
         ha="center", fontsize=8, color="#444", style="italic")

fig.subplots_adjust(left=0.08, right=0.98, top=0.85, bottom=0.14)

# ── 5. Save ────────────────────────────────────────────────────────────────────
print("\n3. Saving v2 figure (scratch only — does NOT touch output/) …")
SCRATCH_PDF.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(SCRATCH_PDF, bbox_inches="tight")
print(f"   Saved → {SCRATCH_PDF.relative_to(BASE)}")
print("\nDone.  Compare with all_data_damping_scatter.pdf and pick one.")
