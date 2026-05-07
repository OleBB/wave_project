"""
All-data damping scatter (CH05 supplementary) — ka variant
==========================================================

Companion to `all_data_damping_scatter.py`. Same data, same encoding;
the only change is the x-axis: paddle-only ka per run instead of
input-frequency-derived k.

x-axis = `IN Wavenumber (FFT)` × `IN Amplitude (FFT)` [m]. Both factors
are FFT-measured, paddle-tone-only quantities (the pipeline column
`IN ka (FFT)` mixes FFT wavenumber with the time-domain percentile
amplitude, which inflates ka under wind by the wind-wave energy on
top of the paddle wave — misleading for steepness, see CLAUDE.md §16).

Visual language identical to `ch05_damping_all_data_scatter`:
  - Wind condition  → colour (WIND_COLOR_MAP: blue=no, red=full)
  - Amplitude tier  → marker shape (○ = A1, □ = A2, △ = A3)
  - Hardware        → marker fill (filled = cond4 final, hollow = earlier)

Run from repo root:
    conda run -n draumkvedet python analysis_scratch/all_data_damping_scatter_ka.py

Outputs:
    analysis_scratch/all_data_damping_scatter_ka.pdf       (scratch quick-view)
    analysis_scratch/all_data_damping_scatter_ka_summary.csv
    output/FIGURES/ch05_damping_all_data_scatter_ka.pdf    (thesis supplementary)
    output/TEXFIGU/ch05_damping_all_data_scatter_ka.tex    (stub; caption from
                                                            FIGURE_CAPTIONS in
                                                            main_save_figures.py)
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
from matplotlib.ticker import MultipleLocator

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import os
os.chdir(BASE)

from wavescripts.improved_data_loader import load_analysis_data
from wavescripts.constants import PROBE_HEIGHT_DEFAULT_MM
from wavescripts.plot_utils import (
    WIND_COLOR_MAP, amp_to_label, apply_thesis_style,
)

apply_thesis_style()

# ── I/O ────────────────────────────────────────────────────────────────────────
SCRATCH_PDF = Path(__file__).parent / "all_data_damping_scatter_ka.pdf"
SCRATCH_CSV = Path(__file__).parent / "all_data_damping_scatter_ka_summary.csv"

THESIS_NAME = "ch05_damping_all_data_scatter_ka"
OUT_PDF  = BASE / "output" / "FIGURES" / f"{THESIS_NAME}.pdf"
OUT_STUB = BASE / "output" / "TEXFIGU" / f"{THESIS_NAME}.tex"
CHAPTER  = "05"

K_COL = "IN Wavenumber (FFT)"
A_COL = "IN Amplitude (FFT)"

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
    & meta[K_COL].notna()
    & meta[A_COL].notna()
].copy()

n_extreme = ((wave["OUT/IN (FFT)"] > 2.0) | (wave["OUT/IN (FFT)"] < 0.1)).sum()
wave_clip = wave[(wave["OUT/IN (FFT)"] <= 2.0) & (wave["OUT/IN (FFT)"] >= 0.1)].copy()
print(f"   {len(wave)} wave/fullpanel/quality=ok runs  ({n_extreme} extreme outliers clipped)")

wave_clip = wave_clip[wave_clip["WindCondition"].isin(["no", "full"])].copy()
print(f"   {len(wave_clip)} after restricting to wind ∈ {{no, full}}")

n_drop_2hz = int((wave_clip["WaveFrequencyInput [Hz]"] >= 2.0).sum())
wave_clip = wave_clip[wave_clip["WaveFrequencyInput [Hz]"] < 2.0].copy()
print(f"   {len(wave_clip)} after dropping {n_drop_2hz} run(s) at f >= 2.0 Hz")

# Paddle-only ka: FFT wavenumber × FFT amplitude (mm → m).
wave_clip["ka"] = (wave_clip[K_COL].astype(float)
                   * wave_clip[A_COL].astype(float) / 1000.0)

print(f"\n   ka range: [{wave_clip['ka'].min():.3f}, {wave_clip['ka'].max():.3f}]")
print(f"   freq range: [{wave_clip['WaveFrequencyInput [Hz]'].min():.2f}, "
      f"{wave_clip['WaveFrequencyInput [Hz]'].max():.2f}] Hz")

print("\n2. Counts per condition × wind:")
pivot = wave_clip.groupby(["condition", "WindCondition"]).size().unstack(fill_value=0)
print(pivot.to_string())

FINAL_CONDITION = "cond4_h100_low"
wave_clip["is_final"] = wave_clip["condition"] == FINAL_CONDITION

print("\n   final-vs-earlier hardware split:")
print(wave_clip.groupby(["is_final", "WindCondition"]).size()
                .unstack(fill_value=0).to_string())

# ── 3. Save summary CSV ───────────────────────────────────────────────────────
summary = (wave_clip.groupby(["condition", "is_final", "WindCondition", "PanelCondition"])
                     .agg(n=("path", "count"),
                          freq_min=("WaveFrequencyInput [Hz]", "min"),
                          freq_max=("WaveFrequencyInput [Hz]", "max"),
                          ka_min=("ka", "min"),
                          ka_max=("ka", "max"),
                          out_in_mean=("OUT/IN (FFT)", "mean"),
                          out_in_std=("OUT/IN (FFT)", "std"))
                     .reset_index())
summary.to_csv(SCRATCH_CSV, index=False)
print(f"\n   Summary → {SCRATCH_CSV.relative_to(BASE)}")

# ── 4. Plot ───────────────────────────────────────────────────────────────────
WIND_LABEL = {"no": "uten vind", "full": "med vind"}

AMP_MARKER = {0.10: "o", 0.20: "s", 0.30: "^"}
AMP_MARKER_DEFAULT = "X"
MARKER_SIZE = 55

ALPHA_FILLED  = 0.65
ALPHA_HOLLOW  = 0.85
EDGE_LW_FILLED = 0.3
EDGE_LW_HOLLOW = 1.4


def _round_amp(v):
    return round(float(v), 2)


fig, ax = plt.subplots(figsize=(6.27, 9.5))

# Plot order: earlier hardware first (so the canonical cond4 markers paint over).
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
                fc, ec = color, "black"
                lw, a  = EDGE_LW_FILLED, ALPHA_FILLED
            else:
                fc, ec = "none", color
                lw, a  = EDGE_LW_HOLLOW, ALPHA_HOLLOW
            ax.scatter(
                s["ka"], s["OUT/IN (FFT)"],
                facecolors=fc, edgecolors=ec, marker=marker,
                s=MARKER_SIZE, linewidths=lw, alpha=a,
                zorder=3 if is_final else 2,
            )

# Catch-all for unexpected amplitudes.
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
            ax.scatter(uw["ka"], uw["OUT/IN (FFT)"],
                       facecolors=(color if is_final else "none"),
                       edgecolors=("black" if is_final else color),
                       marker=AMP_MARKER_DEFAULT,
                       s=MARKER_SIZE,
                       linewidths=(EDGE_LW_FILLED if is_final else EDGE_LW_HOLLOW),
                       alpha=(ALPHA_FILLED if is_final else ALPHA_HOLLOW),
                       zorder=3 if is_final else 2)

ax.axhline(1.0, color="black", lw=0.6, ls="--", alpha=0.5)
ax.set_xlabel(r"$ka$  (Inn, målt)", fontsize=11)
ax.set_ylabel(r"$K_t$", fontsize=12,
              rotation=0, ha="left", va="bottom")

ax.yaxis.set_major_locator(MultipleLocator(0.1))
ax.yaxis.set_minor_locator(MultipleLocator(0.05))
ax.xaxis.set_major_locator(MultipleLocator(0.1))
ax.xaxis.set_minor_locator(MultipleLocator(0.05))
ax.grid(which="major", alpha=0.30, lw=0.6)
ax.grid(which="minor", alpha=0.15, lw=0.4)

ax.set_ylim(0.1, 1.18)

# Legend stack: hardware (top) → wind → amplitude.
wind_handles = [
    mlines.Line2D([], [], color=WIND_COLOR_MAP[w],
                  linestyle="-", linewidth=5,
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

leg_hw = ax.legend(handles=hardware_handles, loc="upper right",
                   bbox_to_anchor=(0.995, 0.995),
                   fontsize=8, framealpha=0.92,
                   title="Eksperiment", title_fontsize=8)
ax.add_artist(leg_hw)

leg_w = ax.legend(handles=wind_handles, loc="upper right",
                  bbox_to_anchor=(0.995, 0.86),
                  fontsize=8, framealpha=0.92,
                  title="Vind", title_fontsize=8)
ax.add_artist(leg_w)

ax.legend(handles=amp_handles, loc="upper right",
          bbox_to_anchor=(0.995, 0.74),
          fontsize=8, framealpha=0.92,
          title="Amplitude", title_fontsize=8)

n_total = len(wave_clip)
n_final = int(wave_clip["is_final"].sum())
print(f"\n   For caption use:  n = {n_total} kjøringer  "
      f"({n_final} fra endelig oppsett, "
      f"{n_total - n_final} fra tidligere oppsett).")

fig.subplots_adjust(left=0.10, right=0.98, top=0.95, bottom=0.06)

# Horizontal y-axis label aligned with leftmost edge of y-tick labels.
fig.canvas.draw()
_renderer = fig.canvas.get_renderer()
_ticks = [t for t in ax.yaxis.get_ticklabels()
          if t.get_visible() and t.get_text().strip()]
if _ticks:
    _left_disp = min(t.get_window_extent(renderer=_renderer).x0 for t in _ticks)
    _x_axes = ax.transAxes.inverted().transform((_left_disp, 0))[0]
    ax.yaxis.set_label_coords(_x_axes, 1.02)

# ── 5. Save ───────────────────────────────────────────────────────────────────
print("\n3. Saving figure …")
SCRATCH_PDF.parent.mkdir(parents=True, exist_ok=True)
OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
OUT_STUB.parent.mkdir(parents=True, exist_ok=True)

fig.savefig(SCRATCH_PDF, bbox_inches="tight", pad_inches=0.02)
print(f"   Saved → {SCRATCH_PDF.relative_to(BASE)}")
fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.02)
print(f"   Saved → {OUT_PDF.relative_to(BASE)}")

# ── 6. TEXFIGU stub ───────────────────────────────────────────────────────────
import wavescripts.plot_utils as pu
pu.ACTIVE_DATASETS = [Path(d).name for d in all_dirs]

_meta_stub = pu.build_fig_meta(
    {
        "filters": {
            "PanelCondition":   "full",
            "WindCondition":    "no, full",
            "quality_flag":     "ok",
        },
        "plotting": {"figure_name": THESIS_NAME},
    },
    chapter=CHAPTER,
    extra={"script": "analysis_scratch/all_data_damping_scatter_ka.py"},
    computed_in=("analysis_scratch/all_data_damping_scatter_ka.py "
                 "(cross-condition supplementary scatter, ka on x)"),
    data_class="DELEG",
    findings_doc=None,
    fft_window_hz=0.1,
    extra_params=(
        f"all PROCESSED-* folders ({len(all_dirs)}). Same filter as the k-axis "
        f"sibling ch05_damping_all_data_scatter; x-axis swapped from k to "
        f"paddle-only ka = `IN Wavenumber (FFT)` × `IN Amplitude (FFT)` [m]. "
        f"Both factors are FFT-measured, paddle-tone-only — explicitly NOT "
        f"the pipeline `IN ka (FFT)` column, which mixes FFT wavenumber with "
        f"time-domain percentile amplitude and is wind-contaminated. "
        f"Encoding: wind → colour, amplitude → marker shape, hardware → fill. "
        f"Y-axis capped at 1.18; ka range from {wave_clip['ka'].min():.3f} "
        f"to {wave_clip['ka'].max():.3f}."
    ),
    extra_stats={
        "n_total":           len(wave_clip),
        "n_final_cond4":     n_final,
        "n_earlier_hw":      len(wave_clip) - n_final,
        "n_extreme_clipped": int(n_extreme),
        "n_drop_2hz":        int(n_drop_2hz),
        "freq_min":          float(wave_clip["WaveFrequencyInput [Hz]"].min()),
        "freq_max":          float(wave_clip["WaveFrequencyInput [Hz]"].max()),
        "ka_min":            float(wave_clip["ka"].min()),
        "ka_max":            float(wave_clip["ka"].max()),
    },
)
pu.write_figure_stub(_meta_stub, plot_type="damping_all_data_scatter_ka",
                     subfig_filenames=[THESIS_NAME],
                     thispagestyle="empty")
print(f"   Stub → {OUT_STUB.relative_to(BASE)}")

print("\nDone.")
