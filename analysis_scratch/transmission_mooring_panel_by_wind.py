"""
Transmission K_t across mooring and panel — split by wind condition
====================================================================

Two figures, one per wind condition (uten / med vind). Same 2 × 3 layout
as `transmission_mooring_panel_compare.py` (mooring row | panel row;
amp tier columns) but with wind held constant in each figure so the
mooring (above_50 vs below_90_*) and panel comparisons can be read
without colour competing for two dimensions.

Visual encoding inside each figure:
  Row A (mooring, full panel)
    above_50          → blue,       marker D
    below_90_loose230 → orange,     marker v
    below_90_loose300 → dark red,   marker P
    above/below contrast = warm-vs-cool hue (primary distinction).
    230 vs 300 = shade within the warm family (secondary).

  Row B (panel, above_50 mooring)
    full    → green,  marker s  (PANEL_MARKERS["full"])
    reverse → purple, marker ^  (PANEL_MARKERS["reverse"])
    no      → grey,   marker o  (PANEL_MARKERS["no"])

Figure title colour = WIND_COLOR_MAP (red for med vind, blue for uten),
so the wind=colour thesis convention is preserved at figure level.

Outputs:
    analysis_scratch/transmission_mooring_panel_nowind.pdf
    analysis_scratch/transmission_mooring_panel_fullwind.pdf
    output/FIGURES/ch05_transmission_mooring_panel_nowind.pdf
    output/FIGURES/ch05_transmission_mooring_panel_fullwind.pdf
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
from wavescripts.plot_utils import (
    apply_thesis_style, WIND_COLOR_MAP, PANEL_MARKERS,
)

apply_thesis_style()

# ── 1. Load & filter ───────────────────────────────────────────────────────────
print("1. Loading processed folders …")
all_dirs = sorted(glob.glob(str(BASE / "waveprocessed" / "PROCESSED-*")))
meta, _, _, _ = load_analysis_data(*all_dirs, load_processed=False)
print(f"   {len(meta)} total rows")

base = meta[
    meta["WaveFrequencyInput [Hz]"].notna()
    & (meta["WaveFrequencyInput [Hz]"] > 0)
    & (meta["quality_flag"] == "ok")
    & meta["OUT/IN (FFT)"].notna()
    & meta["WindCondition"].isin(["no", "full"])
    & (meta["WaveFrequencyInput [Hz]"] < 2.0)
    & (meta["OUT/IN (FFT)"].between(0.1, 2.0))
].copy()
print(f"   {len(base)} rows after base filter")

def _round_amp(v): return round(float(v), 2)
base["amp_v"] = base["WaveAmplitudeInput [Volt]"].apply(_round_amp)

# Drop sparse moorings.
base = base[~base["Mooring"].isin(["above_200"])]

AMP_TIERS = [0.10, 0.20, 0.30]
AMP_LABEL = {0.10: r"$A_1$ (0.1 V)", 0.20: r"$A_2$ (0.2 V)",
             0.30: r"$A_3$ (0.3 V)"}

# Mooring: above_50 = blue (cool), the two below_90 = warm family.
MOORING_GROUPS = ["above_50", "below_90_loose230", "below_90_loose300"]
MOORING_COLOR  = {
    "above_50":           "#1F4E79",   # dark navy
    "below_90_loose230":  "#E47200",   # orange
    "below_90_loose300":  "#C00000",   # dark red
}
MOORING_MARKER = {
    "above_50":           "D",   # diamond
    "below_90_loose230":  "v",   # down-triangle
    "below_90_loose300":  "P",   # plus
}
# Physically meaningful labels. Probe-config composition kept in PROBE_NOTE
# below — printed as a footer so reader knows where residual scatter comes
# from. Probe config is a measurement-uncertainty axis, not a physics axis.
MOORING_LABEL = {
    "above_50":          "above 50 (h272/high)",
    "below_90_loose230": "below 90, 23 cm slakk (4 cfgs)",
    "below_90_loose300": "below 90, 30 cm slakk (h100/low)",
}
PROBE_NOTE = (
    "Probe-cfg per linje: above_50 = h272/high (1 cfg, 308 runs); "
    "below 23 cm = h100/high 137 + h100/low 46 + h136/high 7 + h272/high 18 "
    "(4 cfgs, 208 runs); below 30 cm = h100/low (1 cfg, 86 runs). "
    "Probe-cfg legges som måleusikkerhet, ikke fysikk."
)

# Panel: green / purple / grey so palette doesn't echo Row A.
PANEL_GROUPS = ["full", "reverse", "no"]
PANEL_COLOR  = {
    "full":    "#2CA02C",   # green
    "reverse": "#9467BD",   # purple
    "no":      "#7F7F7F",   # grey
}
PANEL_LABEL = {"full": "full", "reverse": "revers", "no": "ingen panel"}

WIND_TITLE  = {"no": "Uten vind", "full": "Med vind"}
WIND_TAG    = {"no": "nowind", "full": "fullwind"}

# ── 2. Aggregator (per (group, freq, amp), wind already filtered upstream) ─────
def aggregate(df, group_col):
    return (df.groupby([group_col, "WaveFrequencyInput [Hz]", "amp_v"])
              ["OUT/IN (FFT)"]
              .agg(["mean", "std", "count"])
              .reset_index()
              .rename(columns={"WaveFrequencyInput [Hz]": "freq"}))

# ── 3. Plot one wind condition ─────────────────────────────────────────────────
def make_figure(wind: str):
    df_wind = base[base["WindCondition"] == wind]
    rowA = aggregate(df_wind[df_wind["PanelCondition"] == "full"], "Mooring")
    rowB = aggregate(df_wind[df_wind["Mooring"] == "above_50"], "PanelCondition")

    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.5),
                              sharex=True, sharey="row")

    def _plot_cell(ax, df, group_col, group_order, color_map, marker_map):
        for grp in group_order:
            sub = df[df[group_col] == grp]
            if sub.empty:
                continue
            sub = sub.sort_values("freq")
            ax.errorbar(
                sub["freq"], sub["mean"],
                yerr=sub["std"].fillna(0),
                marker=marker_map[grp], ms=7, mew=0.5, mec="black",
                color=color_map[grp], lw=1.4, ls="-",
                capsize=2.5, capthick=0.8, elinewidth=0.7,
                alpha=0.95,
            )

    for j, amp_v in enumerate(AMP_TIERS):
        _plot_cell(axes[0, j], rowA[rowA["amp_v"] == amp_v], "Mooring",
                    MOORING_GROUPS, MOORING_COLOR, MOORING_MARKER)
        axes[0, j].set_title(f"{AMP_LABEL[amp_v]}", fontsize=10)
        _plot_cell(axes[1, j], rowB[rowB["amp_v"] == amp_v], "PanelCondition",
                    PANEL_GROUPS, PANEL_COLOR, PANEL_MARKERS)

    axes[0, 0].set_ylabel(
        "Mooring-sammenligning\n(full panel)\n$K_t$ = OUT/IN (FFT)",
        fontsize=9.5,
    )
    axes[1, 0].set_ylabel(
        "Panel-sammenligning\n(above_50 mooring)\n$K_t$ = OUT/IN (FFT)",
        fontsize=9.5,
    )
    for ax in axes[1, :]:
        ax.set_xlabel("Frekvens (Hz)", fontsize=10)
    for ax in axes.flat:
        ax.axhline(1.0, color="black", lw=0.6, ls=":", alpha=0.5)
        ax.grid(which="major", alpha=0.30, lw=0.6)
        ax.grid(which="minor", alpha=0.15, lw=0.4)
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.xaxis.set_minor_locator(MultipleLocator(0.1))
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.05))

    mooring_handles = [
        mlines.Line2D([], [], color=MOORING_COLOR[m], marker=MOORING_MARKER[m],
                      ms=8, mec="black", mew=0.4, lw=1.4,
                      label=MOORING_LABEL[m]) for m in MOORING_GROUPS
    ]
    panel_handles = [
        mlines.Line2D([], [], color=PANEL_COLOR[p], marker=PANEL_MARKERS[p],
                      ms=8, mec="black", mew=0.4, lw=1.4,
                      label=PANEL_LABEL[p]) for p in PANEL_GROUPS
    ]
    axes[0, 2].legend(handles=mooring_handles, loc="best",
                       fontsize=7.5, framealpha=0.92,
                       title="Mooring", title_fontsize=8, ncol=1)
    axes[1, 2].legend(handles=panel_handles, loc="best",
                       fontsize=7.5, framealpha=0.92,
                       title="Panel", title_fontsize=8, ncol=1)

    fig.suptitle(f"{WIND_TITLE[wind]} — Transmission $K_t$ "
                  f"på tvers av mooring og panel",
                  fontsize=12, color=WIND_COLOR_MAP[wind], y=0.997)
    fig.text(0.005, 0.005, PROBE_NOTE,
             fontsize=6.5, color="#555", ha="left", va="bottom",
             wrap=True)
    fig.tight_layout(rect=[0, 0.03, 1, 0.985])

    scratch = (Path(__file__).parent
                / f"transmission_mooring_panel_{WIND_TAG[wind]}.pdf")
    out_pdf = (BASE / "output" / "FIGURES"
                / f"ch05_transmission_mooring_panel_{WIND_TAG[wind]}.pdf")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(scratch, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    print(f"   Saved → {scratch.relative_to(BASE)}")
    print(f"   Saved → {out_pdf.relative_to(BASE)}")
    plt.close(fig)

print()
for w in ["no", "full"]:
    print(f"── {WIND_TITLE[w]} ─────────────────────────")
    make_figure(w)

print("\nDone.")
