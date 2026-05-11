"""
All-data damping scatter (CH05 supplementary, k-axis)
======================================================

Cross-condition pattern check: K_t (OUT/IN with per-row LS+PSD override)
for every quality-ok, full+reverse-panel wave run across the broader
experimental record. The thesis-headline §1 result uses canon (cond4
h100/lowrange) loose300 only at 1.3–1.6 Hz; this figure shows the wider
band and the other moorings so the reader can see the canonical cluster
in context.

Visual encoding (mooring × panel — mirrors the ka sibling
`all_data_damping_scatter_ka.py` and the k-axis under/over script
`under_and_over_mooring_scatter_k.py`):

  - 4 categories: below_loose300 full | below_loose230 full
                  above_50 full     | above_50 reverse
  - Mooring × wind → colour:
      below_loose300:  canonical red / blue   (WIND_COLOR_MAP)
      below_loose230:  light salmon / light blue
      above_50:        firebrick / steel blue
  - Mooring family → marker family:
      below moorings:  ○ A1, □ A2, △ A3
      above_50:        D  A1, P  A2, h  A3
  - above_50 reverse-panel: hollow markers (face='none', colour to edge)

History: this script was rewritten 2026-05-11 from the earlier
hardware-based encoding (cond4 final vs earlier hardware, filled vs
hollow) to the mooring × panel encoding used by the ka sibling. The
old hardware-encoded v2 is recoverable from git history (search for
`assign_condition` and `FINAL_CONDITION = "cond4_h100_low"`). A v1
with hardware as the primary colour axis is archived at
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

# ── K_t SOURCE: FFT canonical, with three-way LS+PSD fallback ─────────────────
# IMPORTANT — read before changing anything that touches y-values.
#
# K_t plotted on this figure is NOT a flat `OUT/IN (FFT)` read. Per row we
# compute the canonical FFT K_t and ALSO the LS and PSD K_t (using each
# row's `in_probes_used` / `out_probes_used` to know which probes to
# average), then apply a per-row override:
#
#     if |Kt_FFT − Kt_LS| > 0.05  AND  |Kt_FFT − Kt_PSD| > 0.05
#                                AND  |Kt_LS  − Kt_PSD| < 0.05:
#         use Kt_LS         # FFT is the odd one out; LS+PSD agree → trust them
#     else:
#         use Kt_FFT        # canonical
#
# Why: a handful of runs have non-zero `cut_samples_*` (e.g. mar13 1.4 Hz
# A3 nowind above_50, mar19 1.5 Hz A1 fullwind loose230) — those gaps
# break integer-cycle coherence in the H&G-snapped FFT window and leak
# the paddle-bin amplitude out, inflating K_t. The LS sinusoid fit at
# f_paddle and the PSD variance integration are bin-grid-independent
# and robust to such gaps; when they corroborate each other against
# FFT, we trust them.
#
# Asymmetric in FFT's favour: ONE substitute method disagreeing isn't
# enough — we require both LS and PSD to disagree with FFT AND to agree
# with each other. Threshold 0.05 sits well above measurement noise:
# next-nearest non-trigger has |Kt_FFT − Kt_LS| ≤ 0.03 across the
# in-scope dataset, clean separation from the few triggered runs.
#
# Sibling scripts that implement the identical rule:
#   - analysis_scratch/under_and_over_mooring_scatter_k.py    (k-axis under/over)
#   - analysis_scratch/all_data_damping_scatter_ka.py         (ka-axis all/under/over)
# Keep all three in sync if you change anything here.
#
# Kill-switch: set OVERRIDE_THRESHOLD = float("inf") to revert to pure FFT.
OVERRIDE_THRESHOLD = 0.05

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
# Mooring × panel-orientation category — mirrors the ka sibling (see top-of-file
# IMMUTABLE block sibling cross-reference). Replaces the earlier hardware-based
# encoding (cond1/2/3/4 + legacy) — that lens lives in
# `ignore_this_archive/analysis_scratch/all_data_damping_scatter_hardware_2026-04-28/`
# via git history if anyone wants to recover it.
def _category(row):
    m = row.get("Mooring", None)
    p = row.get("PanelCondition", None)
    if m == "below_90_loose300" and p == "full":     return "below_loose300_full"
    if m == "below_90_loose230" and p == "full":     return "below_loose230_full"
    if m == "above_50"          and p == "full":     return "above_50_full"
    if m == "above_50"          and p == "reverse":  return "above_50_reverse"
    return "other"

wave = meta[
    meta["WaveFrequencyInput [Hz]"].notna()
    & (meta["WaveFrequencyInput [Hz]"] > 0)
    # 2026-05-11: switched from `== "full"` to `isin(["full", "reverse"])`
    # so the above_50 reverse-panel runs join the figure (panel-pooled
    # for above_50; below moorings only have full panel in canon).
    # Encoding mirrors the ka sibling (mooring × panel, not hardware).
    & meta["PanelCondition"].isin(["full", "reverse"])
    & (meta["quality_flag"] == "ok")
    & meta["OUT/IN (FFT)"].notna()
    # in/out_probes_used drive the per-method K_t recomputation below.
    # If they're missing on a row, we can't apply the three-way override
    # rule, so drop the row from this figure's scope.
    & meta["in_probes_used"].notna()
    & meta["out_probes_used"].notna()
].copy()

n_extreme = ((wave["OUT/IN (FFT)"] > 2.0) | (wave["OUT/IN (FFT)"] < 0.1)).sum()
wave_clip = wave[(wave["OUT/IN (FFT)"] <= 2.0) & (wave["OUT/IN (FFT)"] >= 0.1)].copy()
print(f"   {len(wave)} wave/quality=ok runs  ({n_extreme} extreme outliers clipped)")

wave_clip = wave_clip[wave_clip["WindCondition"].isin(["no", "full"])].copy()
print(f"   {len(wave_clip)} after restricting to wind ∈ {{no, full}}")

# Drop the lone 2.0 Hz run (1 fullwind/0.3V) — way out at k≈16, wastes
# horizontal space and not a real cluster.
n_drop_2hz = int((wave_clip["WaveFrequencyInput [Hz]"] >= 2.0).sum())
wave_clip = wave_clip[wave_clip["WaveFrequencyInput [Hz]"] < 2.0].copy()
print(f"   {len(wave_clip)} after dropping {n_drop_2hz} run(s) at f >= 2.0 Hz")

# Drop the 3 oddball above_200 mooring runs (same as ka sibling).
n_above200 = int((wave_clip["Mooring"] == "above_200").sum())
wave_clip = wave_clip[wave_clip["Mooring"] != "above_200"].copy()
if n_above200:
    print(f"   dropped {n_above200} runs at Mooring=above_200")

wave_clip["k"] = freq_to_k(wave_clip["WaveFrequencyInput [Hz]"].values)

# ── K_t override rule — see top-of-file IMMUTABLE block for full reasoning ───
# Per row: recompute K_t under each of the three amplitude methods (FFT, LS,
# PSD) using `in_probes_used` / `out_probes_used` to know which probes
# contribute. Then keep canonical FFT unless FFT contradicts BOTH LS and
# PSD, AND those two agree with each other — in which case use LS.
def _kt_method(row, method_suffix):
    inp = [p.strip() for p in str(row["in_probes_used"]).split("+")]
    out = [p.strip() for p in str(row["out_probes_used"]).split("+")]
    try:
        a_in  = float(np.nanmean([row[f"Probe {p} Amplitude{method_suffix}"] for p in inp]))
        a_out = float(np.nanmean([row[f"Probe {p} Amplitude{method_suffix}"] for p in out]))
        if a_in <= 0 or not np.isfinite(a_in) or not np.isfinite(a_out):
            return np.nan
        return a_out / a_in
    except KeyError:
        return np.nan

wave_clip["Kt_LS"]    = wave_clip.apply(lambda r: _kt_method(r, " (LS)"),  axis=1)
wave_clip["Kt_PSD"]   = wave_clip.apply(lambda r: _kt_method(r, " (PSD)"), axis=1)
wave_clip["Kt_canon"] = wave_clip["OUT/IN (FFT)"].astype(float)

_dF  = (wave_clip["Kt_canon"] - wave_clip["Kt_LS"]).abs()
_dP  = (wave_clip["Kt_canon"] - wave_clip["Kt_PSD"]).abs()
_dLP = (wave_clip["Kt_LS"]    - wave_clip["Kt_PSD"]).abs()
_override = (_dF > OVERRIDE_THRESHOLD) & (_dP > OVERRIDE_THRESHOLD) & (_dLP < OVERRIDE_THRESHOLD)
wave_clip["Kt_override"] = _override
wave_clip["Kt_eff"] = np.where(_override, wave_clip["Kt_LS"], wave_clip["Kt_canon"])

print(f"\n   K_t override (FFT → LS) fires on "
      f"{int(_override.sum())} of {len(wave_clip)} runs "
      f"(threshold |ΔKt| > {OVERRIDE_THRESHOLD}, LS+PSD agreement required)")
if _override.any():
    _cols = ["WaveFrequencyInput [Hz]", "WaveAmplitudeInput [Volt]",
             "WindCondition", "PanelCondition",
             "Kt_canon", "Kt_LS", "Kt_PSD"]
    if "Mooring" in wave_clip.columns:
        _cols.insert(3, "Mooring")
    print(wave_clip[_override][_cols].to_string())
    print("   Override paths:")
    for _p in wave_clip[_override]["path"]:
        print(f"     {_p.split('/wavedata/', 1)[1]}")

# ── Categorize by (mooring × panel) and prune "other" ─────────────────────────
wave_clip["category"] = wave_clip.apply(_category, axis=1)
n_other = int((wave_clip["category"] == "other").sum())
if n_other:
    print(f"   {n_other} runs in 'other' category dropped")
    wave_clip = wave_clip[wave_clip["category"] != "other"].copy()

print("\n2. Counts per category × wind:")
print(wave_clip.groupby(["category", "WindCondition"]).size()
                .unstack(fill_value=0).to_string())
print(f"   total: {len(wave_clip)} runs")

# ── 3. Save summary CSV grouped by mooring×panel category ────────────────────
summary = (wave_clip.groupby(["category", "WindCondition", "PanelCondition"])
                     .agg(n=("path", "count"),
                          freq_min=("WaveFrequencyInput [Hz]", "min"),
                          freq_max=("WaveFrequencyInput [Hz]", "max"),
                          out_in_mean=("OUT/IN (FFT)", "mean"),
                          out_in_std=("OUT/IN (FFT)", "std"))
                     .reset_index())
summary.to_csv(SCRATCH_CSV, index=False)
print(f"\n   Summary → {SCRATCH_CSV.relative_to(BASE)}")

# ── 4. Plot — mooring × panel encoding (mirrors ka all-data + k-axis under/over) ──
# Replaces the earlier hardware-based encoding (cond4 vs earlier, filled vs
# hollow). Sibling scripts that use the identical scheme:
#   - analysis_scratch/all_data_damping_scatter_ka.py (all-data, under, over)
#   - analysis_scratch/under_and_over_mooring_scatter_k.py (under, over)
# Above-water (above_50) gets a distinct marker family (D/P/h) and a muted
# colour palette (firebrick / steel blue) so the over-vs-under K_t gap reads
# at a glance.
ABOVE_FULLWIND_COLOR = "#B22222"   # firebrick (muted dark red)
ABOVE_NOWIND_COLOR   = "#4682B4"   # steel blue (muted blue)

COLORS = {
    ("below_loose300_full", "no"):   WIND_COLOR_MAP["no"],   # canonical blue
    ("below_loose300_full", "full"): WIND_COLOR_MAP["full"], # canonical red
    ("below_loose230_full", "no"):   "#9ECAE1",              # light blue
    ("below_loose230_full", "full"): "#F4815A",              # orange salmon
    ("above_50_full",       "no"):   ABOVE_NOWIND_COLOR,
    ("above_50_full",       "full"): ABOVE_FULLWIND_COLOR,
    ("above_50_reverse",    "no"):   ABOVE_NOWIND_COLOR,
    ("above_50_reverse",    "full"): ABOVE_FULLWIND_COLOR,
}
HOLLOW_CATEGORIES = {"above_50_reverse"}
CATEGORY_LABELS = {
    "below_loose300_full": "Under, 30 cm",
    "below_loose230_full": "Under, 23 cm",
    "above_50_full":       "Over",
    "above_50_reverse":    "Over, revers",
}
# Paint order: above_50 first (background), below moorings on top — canonical
# loose300 stays most visible. Mirrors the ka sibling.
CATEGORY_ORDER = [
    "above_50_full", "above_50_reverse",
    "below_loose230_full", "below_loose300_full",
]
MARKERS = {
    "below_loose300_full": {0.10: "o",  0.20: "s",  0.30: "^"},
    "below_loose230_full": {0.10: "o",  0.20: "s",  0.30: "^"},
    "above_50_full":       {0.10: "D",  0.20: "P",  0.30: "h"},
    "above_50_reverse":    {0.10: "D",  0.20: "P",  0.30: "h"},
}
WIND_LABEL = {"no": "uten vind", "full": "full vind"}
MARKER_SIZE = 55
ALPHA = 0.75
EDGE_LW = 0.4

def _round_amp(v): return round(float(v), 2)


# A4 portrait, 1-inch margins → text width 6.27 in. Same aspect as before.
fig, ax = plt.subplots(figsize=(6.27, 9.5))

for cat in CATEGORY_ORDER:
    sub_cat = wave_clip[wave_clip["category"] == cat]
    if sub_cat.empty:
        continue
    for wind in ("no", "full"):
        for amp_v in (0.10, 0.20, 0.30):
            s = sub_cat[(sub_cat["WindCondition"] == wind)
                        & (sub_cat["WaveAmplitudeInput [Volt]"].apply(_round_amp) == amp_v)]
            if s.empty:
                continue
            color  = COLORS.get((cat, wind), "gray")
            marker = MARKERS[cat][amp_v]
            sz = MARKER_SIZE * (1.6 if cat.startswith("above_50") else 1.0)
            hollow = cat in HOLLOW_CATEGORIES
            face_color = "none" if hollow else color
            edge_color = color  if hollow else "black"
            edge_lw    = 1.1    if hollow else EDGE_LW
            # y-source: Kt_eff = per-row FFT→LS override (see top-of-file block).
            ax.scatter(
                s["k"], s["Kt_eff"],
                facecolors=face_color, edgecolors=edge_color,
                marker=marker, s=sz, linewidths=edge_lw, alpha=ALPHA,
                zorder=3 if cat == "below_loose300_full" else 2,
            )

# Catch-all for amplitudes outside {0.1, 0.2, 0.3} — uncommon (mostly 0.6V).
# Print a note; do NOT plot (no marker mapping for them under this scheme).
_recognised_amps = {0.10, 0.20, 0.30}
unknown = wave_clip[~wave_clip["WaveAmplitudeInput [Volt]"]
                    .apply(_round_amp).isin(_recognised_amps)]
if not unknown.empty:
    print(f"   note: {len(unknown)} runs with amp ∉ {{0.1, 0.2, 0.3}} V "
          f"NOT plotted (mooring×amp marker scheme has no slot for them).")

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

# LS-override count annotation (lower-left corner). See top-of-file IMMUTABLE
# block for the override rule. Always rendered, even when n=0, so the reader
# knows the check was applied.
_n_ls_swap = int(wave_clip["Kt_override"].sum())
_xlim_lo, _xlim_hi = ax.get_xlim()
ax.text(
    _xlim_lo + 0.02 * (_xlim_hi - _xlim_lo),
    0.1 + 0.025,
    f"n = {_n_ls_swap} data estimert med LS",
    ha="left", va="bottom",
    fontsize=8, color="#555555", alpha=0.90,
    bbox=dict(boxstyle="round,pad=0.25",
              facecolor="white", alpha=0.80, edgecolor="none"),
    zorder=5,
)

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

# ── Legend — two stacked legends (mirrors ka sibling _make_view) ─────────────
#   Konfigurasjon  (cat × wind, 8 entries — 4 categories × 2 winds, 2-col layout)
#   Amplitude      (3 amps × 2 marker families = 6 entries, 2-col layout)
config_handles = []
for cat in CATEGORY_ORDER[::-1]:
    is_above = cat.startswith("above_50")
    cat_marker_exemplar = "D" if is_above else "o"
    cat_msize = 11 if is_above else 9
    hollow = cat in HOLLOW_CATEGORIES
    for wind, wlabel in (("full", "full vind"), ("no", "uten vind")):
        wind_color = COLORS[(cat, wind)]
        mfc = "none"      if hollow else wind_color
        mec = wind_color  if hollow else "black"
        mew = 1.1         if hollow else 0.4
        config_handles.append(
            mlines.Line2D([], [],
                          marker=cat_marker_exemplar, linestyle="None",
                          markerfacecolor=mfc, markeredgecolor=mec,
                          markeredgewidth=mew, markersize=cat_msize,
                          label=f"{CATEGORY_LABELS[cat]}, {wlabel}")
        )
amp_handles = []
for v in (0.10, 0.20, 0.30):
    amp_handles.append(
        mlines.Line2D([], [], color="black",
                      marker=MARKERS["below_loose300_full"][v], linestyle="None",
                      markersize=8, markerfacecolor="lightgray",
                      markeredgecolor="black", markeredgewidth=0.3,
                      label=f"{amp_to_label(v)}  (under)"))
    amp_handles.append(
        mlines.Line2D([], [], color="black",
                      marker=MARKERS["above_50_full"][v], linestyle="None",
                      markersize=10, markerfacecolor="lightgray",
                      markeredgecolor="black", markeredgewidth=0.3,
                      label=f"{amp_to_label(v)}  (over)"))

leg_cfg = ax.legend(handles=config_handles, loc="upper right",
                    bbox_to_anchor=(0.995, 0.995),
                    fontsize=7.5, framealpha=0.92,
                    title="Konfigurasjon", title_fontsize=8,
                    ncol=2)
ax.add_artist(leg_cfg)
fig.canvas.draw()
leg_cfg_bbox = leg_cfg.get_window_extent().transformed(ax.transAxes.inverted())
amp_anchor_y = leg_cfg_bbox.y0 - 0.010
ax.legend(handles=amp_handles, loc="upper right",
          bbox_to_anchor=(0.995, amp_anchor_y),
          fontsize=7, framealpha=0.92,
          title="Amplitude", title_fontsize=8, ncol=2)

# Diagnostic prints — for caption authoring.
n_total = len(wave_clip)
n_over  = int((wave_clip["category"].str.startswith("above_50")).sum())
n_under = int((wave_clip["category"].str.startswith("below_")).sum())
print(f"\n   For caption use:  n = {n_total} kjøringer  "
      f"(over: {n_over}, under: {n_under}).")

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
            "PanelCondition":   "full + reverse",
            "WindCondition":    "no, full",
            "quality_flag":     "ok",
            "categories":       ", ".join(CATEGORY_ORDER),
        },
        "plotting": {"figure_name": THESIS_NAME},
    },
    chapter=CHAPTER,
    extra={"script": "analysis_scratch/all_data_damping_scatter.py"},
    computed_in=("analysis_scratch/all_data_damping_scatter.py "
                 "(cross-condition supplementary scatter, mooring × panel encoding)"),
    data_class="DELEG",
    findings_doc=None,
    fft_window_hz=0.1,
    extra_params=(
        f"all PROCESSED-* folders ({len(all_dirs)}). "
        f"Filter: wave runs (WaveFrequencyInput > 0), PanelCondition ∈ "
        f"{{full, reverse}}, quality_flag=ok, OUT/IN(FFT) ∈ [0.1, 2.0], "
        f"wind ∈ {{no, full}}, freq < 2.0 Hz, Mooring != 'above_200'. "
        f"Encoding: mooring × panel category — colour shade encodes mooring "
        f"(canonical red/blue for loose300, light salmon/light blue for "
        f"loose230, firebrick/steel blue for above_50). Marker family encodes "
        f"mooring family (○/□/△ for below; D/P/h for above_50, hollow on "
        f"reverse panel). Y-axis capped at 1.18 — points above are "
        f"wind-contamination / low-SNR artefacts (CLAUDE.md §16). "
        f"Top axis: every used frequency. Thesis-scope band (1.3–1.6 Hz) "
        f"shaded as a light-blue axvspan. Body font: NewComputerModern10. "
        f"K_t SOURCE: canonical OUT/IN (FFT) with per-row LS+PSD override "
        f"— if |Kt_FFT−Kt_LS|>{OVERRIDE_THRESHOLD} AND "
        f"|Kt_FFT−Kt_PSD|>{OVERRIDE_THRESHOLD} AND "
        f"|Kt_LS−Kt_PSD|<{OVERRIDE_THRESHOLD}, use Kt_LS (FFT broken on "
        f"that run — typically by cut_samples breaking window cycle "
        f"coherence). See script's top-of-file IMMUTABLE block."
    ),
    extra_stats={
        "n_total":           len(wave_clip),
        **{f"n_{c}": int((wave_clip["category"] == c).sum()) for c in CATEGORY_ORDER},
        "n_above_50":        int(wave_clip["category"].str.startswith("above_50").sum()),
        "n_below_90":        int(wave_clip["category"].str.startswith("below_").sum()),
        "n_extreme_clipped": int(n_extreme),
        "n_drop_2hz":        int(n_drop_2hz),
        "freq_min":          float(wave_clip["WaveFrequencyInput [Hz]"].min()),
        "freq_max":          float(wave_clip["WaveFrequencyInput [Hz]"].max()),
        "k_min":             float(wave_clip["k"].min()),
        "k_max":             float(wave_clip["k"].max()),
        # n_kt_override = runs where the FFT-vs-LS+PSD rule swapped
        # Kt_FFT for Kt_LS. See top-of-script IMMUTABLE.
        "n_kt_override":     int(wave_clip["Kt_override"].sum()),
        "kt_override_threshold": OVERRIDE_THRESHOLD,
    },
)
pu.write_figure_stub(_meta_stub, plot_type="damping_all_data_scatter",
                     subfig_filenames=[THESIS_NAME],
                     thispagestyle="empty")
print(f"   Stub → {OUT_STUB.relative_to(BASE)}")

print("\nDone.")
