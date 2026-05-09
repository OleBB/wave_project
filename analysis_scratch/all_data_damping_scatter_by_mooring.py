"""All-data damping scatter, split by (mooring × panel) configuration.

Sister to analysis_scratch/all_data_damping_scatter.py, but instead of
splitting by hardware-cluster (cond1/cond4 fill), splits by physical
configuration: which mooring + panel layout produced each run.

Question being investigated: how much of the ch05_damping_freq errorbar
spread is mooring-dependent? We saw a +0.04 mean delta between
loose230 and loose300 in canon (panel=full only); this plot extends the
view to all four (mooring × panel) configurations actually run.

Visual language (final convention 2026-05-09):
  - Wind condition  → colour hue (red family = full wind, blue family = no wind)
  - (Mooring) → colour shade AND marker family
  - Amplitude tier  → marker shape within family

THREE mooring categories (above_50 fullpanel + reversepanel pooled, since
their |Δ| ≈ 0.03 is comparable to within-panel noise — see
analysis_scratch/_above50_panel_diff.py):
  - below_loose300 × fullpanel  : standard red / blue        | ○ A1, □ A2, △ A3
  - below_loose230 × fullpanel  : light salmon / light blue  | ○ A1, □ A2, △ A3
  - above_50      (panel pooled): pink / turquoise            | ✶ A1 (6-pt star),
                                                                ⋆ A2 (5-pt star),
                                                                ✦ A3 (4-pt star)

Output: scratch only (analysis_scratch/), not a thesis figure (yet).
"""
from __future__ import annotations

import sys
import warnings
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.ticker import MultipleLocator

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from wavescripts.improved_data_loader import load_analysis_data
from wavescripts.plot_utils import (
    freq_to_k, add_freq_axis, amp_to_label, apply_thesis_style,
)
apply_thesis_style()

# ── Load everything ────────────────────────────────────────────────────────────
print("Loading all PROCESSED-* folders …")
all_dirs = sorted(glob.glob(str(BASE / "waveprocessed" / "PROCESSED-*")))
meta, _, _, _ = load_analysis_data(*all_dirs, load_processed=False)
print(f"   {len(meta)} total rows")

# ── Filter & categorize ────────────────────────────────────────────────────────
wave = meta[
    meta["WaveFrequencyInput [Hz]"].notna()
    & (meta["WaveFrequencyInput [Hz]"] > 0)
    & meta["PanelCondition"].isin(["full", "reverse"])
    & meta["WindCondition"].isin(["no", "full"])
    & (meta["quality_flag"] == "ok")
    & meta["OUT/IN (FFT)"].notna()
    & (meta["OUT/IN (FFT)"] >= 0.1) & (meta["OUT/IN (FFT)"] <= 2.0)
    & (meta["WaveFrequencyInput [Hz]"] < 2.0)
].copy()

# Drop the 3 oddball above_200 runs (not in the user's reference list).
n_above200 = int((wave["Mooring"] == "above_200").sum())
wave = wave[wave["Mooring"] != "above_200"].copy()
if n_above200:
    print(f"   dropped {n_above200} runs at Mooring=above_200")

# Compose the (mooring, panel) category tag. Above-50 fullpanel and
# reversepanel are pooled — their |Δ| is comparable to within-panel noise
# (see analysis_scratch/_above50_panel_diff.py).
def _category(row):
    m = row["Mooring"]; p = row["PanelCondition"]
    if m == "below_90_loose300" and p == "full":           return "below_loose300_full"
    if m == "below_90_loose230" and p == "full":           return "below_loose230_full"
    if m == "above_50"          and p in ("full", "reverse"): return "above_50"
    return "other"

wave["category"] = wave.apply(_category, axis=1)
n_other = int((wave["category"] == "other").sum())
if n_other:
    print(f"   {n_other} runs in category 'other' (unmapped mooring/panel combo)")
    print(wave[wave["category"] == "other"]
          .groupby(["Mooring", "PanelCondition"]).size().to_string())

wave["k"] = freq_to_k(wave["WaveFrequencyInput [Hz]"].values)

print("\nRuns per category × wind:")
print(wave.groupby(["category", "WindCondition"]).size().unstack(fill_value=0).to_string())
print(f"   total: {len(wave)} runs")

# ── Visual encoding ────────────────────────────────────────────────────────────
# (category, wind) → colour. Reds for fullwind, blues for nowind.
# Reference (below_loose300, fullpanel): standard project red/blue.
# below_loose230: darker / warmer shades.
# above_50 fullpanel: lighter shades.
# above_50 reverse: pink/turquoise (snappy contrasting hue).
COLORS = {
    # canon-loose300 = reference (standard project red/blue)
    ("below_loose300_full", "no"):   "#1F77B4",  # standard blue
    ("below_loose300_full", "full"): "#D62728",  # standard red
    # canon-loose230 = lighter red/blue (sibling shade — clearly related
    # but distinguishable; was vivid orange/royal blue earlier, swapped
    # 2026-05-09 once above_50 adopted pink/turquoise).
    ("below_loose230_full", "no"):   "#9ECAE1",  # light blue
    ("below_loose230_full", "full"): "#F4815A",  # orange salmon
                                                 # (was #FCAE91 light salmon
                                                 # — shifted 2026-05-09 to
                                                 # widen gap to magenta below)
    # above_50 pooled (full + reverse) — magenta/turquoise + star markers,
    # giving a clear visual separation from the below mooring family.
    ("above_50",            "no"):   "#17BECF",  # turquoise
    ("above_50",            "full"): "#D81B7A",  # magenta (was #E377C2 pink)
}

CATEGORY_LABELS = {
    "below_loose300_full": "Under, loose300, full panel",
    "below_loose230_full": "Under, loose230, full panel",
    "above_50":            "Over (50 mm), pooled paneler",
}

# Plot order: bottom → top (later categories paint on top).
# Put loose300 last so the canonical reference points sit on top of the
# above50 cloud where they overlap.
CATEGORY_ORDER = [
    "above_50",             # bottom
    "below_loose230_full",
    "below_loose300_full",  # top — reference data is most visible
]

# Marker shape per (category, amp). Below-mooring categories use the
# project amp-tier convention (○ A1, □ A2, △ A3). Above_50 uses stars
# (6/5/4-pointed) so it's distinguishable at a glance from the below
# moorings even when the colour is muted by overlap.
MARKERS = {
    "below_loose300_full": {0.10: "o",        0.20: "s",        0.30: "^"},
    "below_loose230_full": {0.10: "o",        0.20: "s",        0.30: "^"},
    "above_50":            {0.10: (6, 1, 0),  0.20: (5, 1, 0),  0.30: (4, 1, 0)},
}
MARKER_SIZE = 50
ALPHA = 0.75
EDGE_LW = 0.4

def _round_amp(v): return round(float(v), 2)

# ── Plot — three separate tall full-page PDFs, one per amplitude ──────────────
PANES = [(0.10, r"$A_1$ (0.10 V)", "A1"),
         (0.20, r"$A_2$ (0.20 V)", "A2"),
         (0.30, r"$A_3$ (0.30 V)", "A3")]

thesis_k_lo = float(freq_to_k(np.array([1.3]))[0])
thesis_k_hi = float(freq_to_k(np.array([1.6]))[0])
_used_freqs = sorted(wave["WaveFrequencyInput [Hz]"].unique())

OUT_PDFS = []
for amp_v, amp_label, amp_tag in PANES:
    sub_amp = wave[wave["WaveAmplitudeInput [Volt]"].apply(_round_amp) == amp_v]

    fig, ax = plt.subplots(figsize=(6.27, 9.5))
    for cat in CATEGORY_ORDER:
        sub = sub_amp[sub_amp["category"] == cat]
        if sub.empty:
            continue
        marker = MARKERS[cat][amp_v]
        # Stars need a slightly larger size to read as similarly-massive
        # to circles/squares/triangles.
        sz = MARKER_SIZE * (1.6 if cat == "above_50" else 1.0)
        for wind in ("no", "full"):
            s = sub[sub["WindCondition"] == wind]
            if s.empty:
                continue
            color = COLORS.get((cat, wind), "gray")
            ax.scatter(
                s["k"], s["OUT/IN (FFT)"],
                facecolors=color, edgecolors="black",
                marker=marker, s=sz,
                linewidths=EDGE_LW, alpha=ALPHA,
                zorder=3 if cat == "below_loose300_full" else 2,
            )

    ax.axhline(1.0, color="black", lw=0.6, ls="--", alpha=0.5)
    ax.axvspan(thesis_k_lo, thesis_k_hi, color="#1F77B4",
               alpha=0.06, lw=0, zorder=1)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.xaxis.set_minor_locator(MultipleLocator(1.0))
    ax.grid(which="major", alpha=0.30, lw=0.6)
    ax.grid(which="minor", alpha=0.15, lw=0.4)
    ax.set_ylim(0.1, 1.18)
    ax.set_xlabel("$k$", fontsize=11)
    ax.set_ylabel(r"$K_t$", fontsize=12, rotation=0, ha="left", va="bottom")
    ax.text(0.99, 0.97, amp_label, transform=ax.transAxes,
            ha="right", va="top", fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="white", alpha=0.92, edgecolor="black", lw=0.5))

    secax = add_freq_axis(ax)
    secax.set_xlabel("Frekvens (Hz)", fontsize=9)
    secax.set_xticks(_used_freqs)
    secax.set_xticklabels([f"{f:.1f}" for f in _used_freqs])
    secax.tick_params(labelsize=7)
    ax.text(thesis_k_hi - 0.1, 1.16, "Hovedfokus\n1,3–1,6 Hz",
            ha="right", va="top", fontsize=8, color="#1F618D", alpha=0.85,
            bbox=dict(boxstyle="round,pad=0.2",
                      facecolor="white", alpha=0.75, edgecolor="none"))

    # Config legend — markers reflect the per-amp shape for this PDF, so
    # the reader sees both colour and shape mapping at once.
    config_handles = []
    for cat in CATEGORY_ORDER[::-1]:
        sub_cat_amp = sub_amp[sub_amp["category"] == cat]
        n_no_amp   = int((sub_cat_amp["WindCondition"] == "no").sum())
        n_full_amp = int((sub_cat_amp["WindCondition"] == "full").sum())
        cat_marker = MARKERS[cat][amp_v]
        cat_msize = 11 if cat == "above_50" else 9
        for wind, wlabel, n in (("full", "full vind", n_full_amp),
                                ("no",   "uten vind", n_no_amp)):
            config_handles.append(
                mlines.Line2D([], [],
                              marker=cat_marker, linestyle="None",
                              markerfacecolor=COLORS[(cat, wind)],
                              markeredgecolor="black",
                              markeredgewidth=0.4,
                              markersize=cat_msize,
                              label=f"{CATEGORY_LABELS[cat]}, {wlabel}  (n={n})")
            )
    ax.legend(handles=config_handles, loc="lower left",
              bbox_to_anchor=(0.005, 0.005),
              fontsize=8, framealpha=0.92,
              title=f"Konfigurasjon ({amp_label}-data)", title_fontsize=8,
              ncol=2)

    fig.subplots_adjust(left=0.10, right=0.98, top=0.95, bottom=0.06)
    fig.canvas.draw()
    _renderer = fig.canvas.get_renderer()
    _ticks = [t for t in ax.yaxis.get_ticklabels()
              if t.get_visible() and t.get_text().strip()]
    if _ticks:
        _left_disp = min(t.get_window_extent(renderer=_renderer).x0 for t in _ticks)
        _x_axes = ax.transAxes.inverted().transform((_left_disp, 0))[0]
        ax.yaxis.set_label_coords(_x_axes, 1.02)

    out_pdf = Path(__file__).parent / f"all_data_damping_scatter_by_mooring_{amp_tag}.pdf"
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    OUT_PDFS.append(out_pdf)

print()
for p in OUT_PDFS:
    print(f"Saved → {p.relative_to(BASE)}")
