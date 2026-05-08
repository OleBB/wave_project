"""
All-data damping scatter (CH05 supplementary)
==============================================

Cross-condition pattern check: OUT/IN(FFT) for every quality-ok,
full-panel wave run across the full experimental record (cond1 h272/high,
cond2 h136/high, cond3 h100/high WRONG, cond4 h100/low — final, plus the
legacy Nov-2025 probe config). The thesis-headline result uses cond4 only
at 1.3–1.6 Hz; this figure shows the broader band so the reader can see
the cond4 cluster in context.

Visual language (matches ch05_damping_freq, ch05_damping_ka):
  - Wind condition  → colour (WIND_COLOR_MAP: blue=no, red=full)
  - Amplitude tier  → marker shape (○ = A1 = 0.1V, □ = A2 = 0.2V, △ = A3 = 0.3V)
  - Hardware        → marker fill (filled = cond4 final, hollow = earlier)

The hardware dimension was the primary colour axis in v1 (5 distinct
colours). v2 (this file, 2026-04-28) downgrades it to a fill modifier —
the canonical cond4 cluster reads as solid colour; earlier hardware reads
as hollow markers in the same wind/amp encoding. The v1 is archived at
    ignore_this_archive/analysis_scratch/all_data_damping_scatter_v1_2026-04-28.{py,pdf,_summary.csv}

Run from repo root:
    conda run -n draumkvedet python analysis_scratch/all_data_damping_scatter.py

Outputs:
    analysis_scratch/all_data_damping_scatter.pdf       (scratch quick-view)
    analysis_scratch/all_data_damping_scatter_summary.csv
    output/FIGURES/ch05_damping_all_data_scatter.pdf    (thesis supplementary)
    output/TEXFIGU/ch05_damping_all_data_scatter.tex    (stub; caption from
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

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import os
os.chdir(BASE)

from wavescripts.improved_data_loader import load_analysis_data
from wavescripts.constants import PROBE_HEIGHT_DEFAULT_MM
from wavescripts.plot_utils import (
    freq_to_k, add_freq_axis, WIND_COLOR_MAP, amp_to_label, amp_to_tag,
    apply_thesis_style,
)

# Thesis body font (NewComputerModern OTFs registered via FontManager).
# usetex=False keeps the script fast; mathtext renders math in a CM-compatible
# font that visually matches NCM body text. If true siunitx \unit{\hertz} is
# required later, flip to apply_thesis_style(usetex=True) and add
# r"\usepackage{siunitx}" to text.latex.preamble — slower but literal.
apply_thesis_style()

# ── I/O ────────────────────────────────────────────────────────────────────────
SCRATCH_PDF = Path(__file__).parent / "all_data_damping_scatter.pdf"
SCRATCH_CSV = Path(__file__).parent / "all_data_damping_scatter_summary.csv"

# Thesis outputs — figure_name (== .tex stem == .pdf stem == \label suffix).
THESIS_NAME = "ch05_damping_all_data_scatter"
OUT_PDF  = BASE / "output" / "FIGURES" / f"{THESIS_NAME}.pdf"
OUT_STUB = BASE / "output" / "TEXFIGU" / f"{THESIS_NAME}.tex"
CHAPTER  = "05"

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

# Drop the lone 2.0 Hz run (1 fullwind/0.3V) — way out at k≈16, wastes
# horizontal space and not a real cluster.
n_drop_2hz = int((wave_clip["WaveFrequencyInput [Hz]"] >= 2.0).sum())
wave_clip = wave_clip[wave_clip["WaveFrequencyInput [Hz]"] < 2.0].copy()
print(f"   {len(wave_clip)} after dropping {n_drop_2hz} run(s) at f >= 2.0 Hz")

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


# A4 portrait, 1-inch margins → text width 6.27 in, text height 9.69 in.
# 6.27 × 9.5 fills the page with room for caption (~0.2 in residual).
fig, ax = plt.subplots(figsize=(6.27, 9.5))

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
ax.text(thesis_k_hi - 0.1, 1.16,
        "Hovedfokus\n1,3–1,6 Hz", ha="right", va="top",
        fontsize=8, color="#1F618D", alpha=0.85,
        bbox=dict(boxstyle="round,pad=0.2",
                  facecolor="white", alpha=0.75, edgecolor="none"))

ax.axhline(1.0, color="black", lw=0.6, ls="--", alpha=0.5)
ax.set_xlabel("$k$", fontsize=11)
ax.set_ylabel(r"$K_t$", fontsize=12,
              rotation=0, ha="left", va="bottom")

# Grid: majors + minors (denser y-grid since the figure is now tall and the
# story is mostly along y).
from matplotlib.ticker import MultipleLocator
ax.yaxis.set_major_locator(MultipleLocator(0.1))
ax.yaxis.set_minor_locator(MultipleLocator(0.05))
ax.xaxis.set_minor_locator(MultipleLocator(1.0))   # majors stay default (~5)
ax.grid(which="major", alpha=0.30, lw=0.6)
ax.grid(which="minor", alpha=0.15, lw=0.4)

# Cap at 1.18 — squeezes the dense middle band into more vertical space.
# Points above 1.18 are wind-contamination / low-SNR artefacts (CLAUDE.md §16),
# not informative for the cross-condition pattern check.
ax.set_ylim(0.1, 1.18)

# Top axis: every frequency that actually appears in the dataset, as ticks
# (instead of the default sparse "every 0.5 Hz" labels).
secax = add_freq_axis(ax)
secax.set_xlabel(r"Frekvens (Hz)", fontsize=9)   # Norwegian; siunitx-style
                                                 # \unit{\hertz} would need
                                                 # text.usetex=True.
_used_freqs = sorted(wave_clip["WaveFrequencyInput [Hz]"].unique())
secax.set_xticks(_used_freqs)
secax.set_xticklabels([f"{f:.1f}" for f in _used_freqs])
secax.tick_params(labelsize=7)

# ── Legend ────────────────────────────────────────────────────────────────────
# Three small legends, one per dimension.
# ka range per wind condition for the legend — same data subset the scatter
# plots. Prefer "IN ka (FFT)" from meta (post-2026-05-07 fix); fallback
# computes it from k × IN-amplitude with mm→m conversion.
_ka_ranges = {}
for _w in ["no", "full"]:
    _sub = wave_clip[wave_clip["WindCondition"] == _w]
    if "IN ka (FFT)" in _sub.columns:
        _ka_vals = _sub["IN ka (FFT)"].dropna()
    elif "IN Amplitude (FFT)" in _sub.columns:
        _ka_vals = (_sub["k"] * _sub["IN Amplitude (FFT)"] * 1e-3).dropna()
    else:
        _ka_vals = pd.Series(dtype=float)
    _ka_ranges[_w] = (float(_ka_vals.min()), float(_ka_vals.max())) if len(_ka_vals) else None

def _wind_label_with_ka(w):
    rng = _ka_ranges.get(w)
    return WIND_LABEL[w] if rng is None else \
        f"{WIND_LABEL[w]}  ($ka$: {rng[0]:.3f}–{rng[1]:.3f})"

wind_handles = [
    mlines.Line2D([], [], color=WIND_COLOR_MAP[w],
                  linestyle="-", linewidth=5,
                  label=_wind_label_with_ka(w))
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

# Layout: all three legends stacked along the right edge, top → bottom:
#   Probeinnstillinger  (hardware fill, 2 entries)
#   Vind                (wind colour, 2 entries)
#   Amplitude           (marker shape, 3 entries)
# y-anchors are in axes-fraction; tweak if boxes overlap or there's a gap.
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

# In-figure subtitle removed by request — counts (n_total, n_final, etc.)
# go into the caption manually. Print them here so they're easy to copy.
n_total = len(wave_clip)
n_final = int(wave_clip["is_final"].sum())
print(f"\n   For caption use:  n = {n_total} kjøringer  "
      f"({n_final} fra endelig oppsett, "
      f"{n_total - n_final} fra tidligere oppsett).")

fig.subplots_adjust(left=0.10, right=0.98, top=0.95, bottom=0.06)

# Horizontal y-axis label aligned with the leftmost edge of the y-tick labels —
# mirrors ch04_plateau_overview / ch04_inspirational convention. Maximises the
# horizontal plotting area by killing the rotated side label.
fig.canvas.draw()
_renderer = fig.canvas.get_renderer()
_ticks = [t for t in ax.yaxis.get_ticklabels()
          if t.get_visible() and t.get_text().strip()]
if _ticks:
    _left_disp = min(t.get_window_extent(renderer=_renderer).x0 for t in _ticks)
    _x_axes = ax.transAxes.inverted().transform((_left_disp, 0))[0]
    ax.yaxis.set_label_coords(_x_axes, 1.02)

# ── 5. Save ────────────────────────────────────────────────────────────────────
print("\n3. Saving figure …")
SCRATCH_PDF.parent.mkdir(parents=True, exist_ok=True)
OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
OUT_STUB.parent.mkdir(parents=True, exist_ok=True)

fig.savefig(SCRATCH_PDF, bbox_inches="tight", pad_inches=0.02)
print(f"   Saved → {SCRATCH_PDF.relative_to(BASE)}")
fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.02)
print(f"   Saved → {OUT_PDF.relative_to(BASE)}")

# ── 6. TEXFIGU stub ───────────────────────────────────────────────────────────
# Caption text is sourced from FIGURE_CAPTIONS / FIGURE_CAPTIONS_SHORT in
# main_save_figures.py via output/.figure_captions.json — no caption is
# passed in meta here. force=True (default in pu.write_figure_stub) rewrites
# the stub body on every regen so the latest authored caption lands.
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
    extra={"script": "analysis_scratch/all_data_damping_scatter.py"},
    computed_in=("analysis_scratch/all_data_damping_scatter.py "
                 "(cross-condition supplementary scatter)"),
    data_class="DELEG",
    findings_doc=None,
    fft_window_hz=0.1,
    extra_params=(
        f"all PROCESSED-* folders ({len(all_dirs)}). "
        f"Filter: wave runs (WaveFrequencyInput > 0), full panel, quality_flag=ok, "
        f"OUT/IN(FFT) ∈ [0.1, 2.0], wind ∈ {{no, full}}, freq < 2.0 Hz. "
        f"Encoding: wind → colour (WIND_COLOR_MAP), amplitude → marker shape "
        f"(○=A1, □=A2, △=A3), hardware → fill (filled = cond4 final, "
        f"hollow = earlier). Y-axis capped at 1.18 — points above are "
        f"wind-contamination / low-SNR artefacts (CLAUDE.md §16). "
        f"Top axis: every used frequency (15 ticks, 0.5–1.9 Hz). "
        f"Body font: NewComputerModern10 via apply_thesis_style()."
    ),
    extra_stats={
        "n_total":           len(wave_clip),
        "n_final_cond4":     n_final,
        "n_earlier_hw":      len(wave_clip) - n_final,
        "n_extreme_clipped": int(n_extreme),
        "n_drop_2hz":        int(n_drop_2hz),
        "freq_min":          float(wave_clip["WaveFrequencyInput [Hz]"].min()),
        "freq_max":          float(wave_clip["WaveFrequencyInput [Hz]"].max()),
        "k_min":             float(wave_clip["k"].min()),
        "k_max":             float(wave_clip["k"].max()),
    },
)
pu.write_figure_stub(_meta_stub, plot_type="damping_all_data_scatter",
                     subfig_filenames=[THESIS_NAME],
                     thispagestyle="empty")
print(f"   Stub → {OUT_STUB.relative_to(BASE)}")

print("\nDone.")
