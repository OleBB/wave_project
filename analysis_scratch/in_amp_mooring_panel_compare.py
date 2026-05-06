"""
Transmission (OUT/IN FFT) across mooring and panel conditions
==============================================================

Two complementary comparisons in one figure (the "extended" version of the
earlier mooring-only quick-look):

  Row A — Mooring effect (PanelCondition = full):
      moorings: above_50, below_90_loose230, below_90_loose300
      (above_200 dropped — only 2 rows in the dataset)

  Row B — Panel effect (Mooring = above_50):
      panels: full, no (no panel installed), reverse

Cols: amplitude tier A1 = 0.1 V, A2 = 0.2 V, A3 = 0.3 V.

Y-axis: mean K_t = OUT/IN (FFT), pooled across hardware runs.
Lines: linestyle = wind (solid = no, dashed = full); colour = group
       (mooring on row A, panel on row B). Errorbars = std (omitted when n=1).
Reference line at K_t = 1 (no panel attenuation).

Outputs:
    analysis_scratch/transmission_mooring_panel_compare.pdf
    output/FIGURES/ch05_transmission_mooring_panel_compare.pdf
    analysis_scratch/transmission_mooring_panel_compare_summary.csv
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
from wavescripts.plot_utils import apply_thesis_style

apply_thesis_style()

SCRATCH_PDF = Path(__file__).parent / "transmission_mooring_panel_compare.pdf"
SCRATCH_CSV = Path(__file__).parent / "transmission_mooring_panel_compare_summary.csv"
OUT_PDF     = BASE / "output" / "FIGURES" / "ch05_transmission_mooring_panel_compare.pdf"
OUT_PDF.parent.mkdir(parents=True, exist_ok=True)

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

# Quantize amplitude.
def _round_amp(v):
    return round(float(v), 2)
base["amp_v"] = base["WaveAmplitudeInput [Volt]"].apply(_round_amp)

# Drop the lone above_200 row(s) — too sparse.
DROP_MOORINGS = ["above_200"]
base = base[~base["Mooring"].isin(DROP_MOORINGS)]

AMP_TIERS = [0.10, 0.20, 0.30]
AMP_LABEL = {0.10: r"$A_1$ (0.1 V)", 0.20: r"$A_2$ (0.2 V)",
             0.30: r"$A_3$ (0.3 V)"}

# Mooring colours (Row A).
MOORING_GROUPS = ["above_50", "below_90_loose230", "below_90_loose300"]
MOORING_COLOR  = {
    "above_50":           "#1f77b4",   # blue
    "below_90_loose230":  "#2ca02c",   # green
    "below_90_loose300":  "#9467bd",   # purple
}
MOORING_LABEL  = {
    "above_50":          "above_50",
    "below_90_loose230": "below_90_loose230",
    "below_90_loose300": "below_90_loose300",
}

# Panel colours (Row B).
PANEL_GROUPS = ["full", "reverse", "no"]
PANEL_COLOR  = {
    "full":    "#d62728",   # red
    "reverse": "#ff7f0e",   # orange
    "no":      "#7f7f7f",   # grey
}
PANEL_LABEL  = {"full": "full", "reverse": "revers", "no": "ingen panel"}

WIND_LS    = {"no": "-", "full": "--"}
WIND_LABEL = {"no": "uten vind", "full": "med vind"}
WIND_MARKER = {"no": "o", "full": "s"}

# ── 2. Aggregator ──────────────────────────────────────────────────────────────
def aggregate(df, group_col):
    """Mean / std / n per (group_col, freq, wind, amp)."""
    return (df.groupby([group_col, "WaveFrequencyInput [Hz]",
                         "WindCondition", "amp_v"])
              ["OUT/IN (FFT)"]
              .agg(["mean", "std", "count"])
              .reset_index()
              .rename(columns={"WaveFrequencyInput [Hz]": "freq",
                                "WindCondition": "wind"}))

# Row A data: full panel, varying mooring.
rowA_src = base[base["PanelCondition"] == "full"]
rowA = aggregate(rowA_src, "Mooring")

# Row B data: above_50 mooring, varying panel.
rowB_src = base[base["Mooring"] == "above_50"]
rowB = aggregate(rowB_src, "PanelCondition")

# ── 3. Plot ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.5),
                          sharex=True, sharey="row")

def _plot_cell(ax, df, group_col, group_order, color_map):
    for grp in group_order:
        for wind in ["no", "full"]:
            sub = df[(df[group_col] == grp) & (df["wind"] == wind)]
            if sub.empty:
                continue
            sub = sub.sort_values("freq")
            color = color_map[grp]
            ax.errorbar(
                sub["freq"], sub["mean"],
                yerr=sub["std"].fillna(0),
                marker=WIND_MARKER[wind], ms=5, mew=0.4, mec="black",
                color=color, lw=1.2, ls=WIND_LS[wind],
                capsize=2.5, capthick=0.8, elinewidth=0.7,
                alpha=0.9,
            )

# Row A — mooring comparison (full panel).
for j, amp_v in enumerate(AMP_TIERS):
    cell_df = rowA[rowA["amp_v"] == amp_v]
    _plot_cell(axes[0, j], cell_df, "Mooring",
                MOORING_GROUPS, MOORING_COLOR)
    axes[0, j].set_title(f"{AMP_LABEL[amp_v]}", fontsize=10)

# Row B — panel comparison (above_50 mooring).
for j, amp_v in enumerate(AMP_TIERS):
    cell_df = rowB[rowB["amp_v"] == amp_v]
    _plot_cell(axes[1, j], cell_df, "PanelCondition",
                PANEL_GROUPS, PANEL_COLOR)

# Row labels via leftmost-cell ylabels.
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

# Per-row legends.
mooring_handles = [
    mlines.Line2D([], [], color=MOORING_COLOR[m], marker="o",
                  ms=6, mec="black", mew=0.4,
                  label=MOORING_LABEL[m]) for m in MOORING_GROUPS
]
panel_handles = [
    mlines.Line2D([], [], color=PANEL_COLOR[p], marker="o",
                  ms=6, mec="black", mew=0.4,
                  label=PANEL_LABEL[p]) for p in PANEL_GROUPS
]
wind_handles = [
    mlines.Line2D([], [], color="black", marker=WIND_MARKER[w],
                  ms=5, mec="black", mew=0.4, lw=1.2, ls=WIND_LS[w],
                  label=WIND_LABEL[w]) for w in ["no", "full"]
]

# Place row-A legend on top-right cell, row-B on bottom-right cell.
leg_a = axes[0, 2].legend(handles=mooring_handles + wind_handles,
                            loc="best", fontsize=7.5, framealpha=0.92,
                            title="Mooring + vind", title_fontsize=8, ncol=1)
leg_b = axes[1, 2].legend(handles=panel_handles + wind_handles,
                            loc="best", fontsize=7.5, framealpha=0.92,
                            title="Panel + vind", title_fontsize=8, ncol=1)

fig.suptitle("Transmission $K_t$ på tvers av mooring og panel",
              fontsize=12, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.985])

# ── 4. Save ────────────────────────────────────────────────────────────────────
SCRATCH_PDF.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(SCRATCH_PDF, bbox_inches="tight", pad_inches=0.02)
print(f"\n   Saved → {SCRATCH_PDF.relative_to(BASE)}")
fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.02)
print(f"   Saved → {OUT_PDF.relative_to(BASE)}")
plt.close(fig)

# ── 5. Summary CSV + headline numbers ──────────────────────────────────────────
summary = pd.concat([
    rowA.assign(_compare="mooring (full panel)").rename(columns={"Mooring": "group"}),
    rowB.assign(_compare="panel (above_50)").rename(columns={"PanelCondition": "group"}),
], ignore_index=True, sort=False)
summary = summary[["_compare", "group", "freq", "wind", "amp_v",
                    "mean", "std", "count"]]
summary.to_csv(SCRATCH_CSV, index=False)
print(f"   Summary → {SCRATCH_CSV.relative_to(BASE)}")

# Quick eyeball: mean K_t across moorings at A2 nowind.
print("\n   Row A snapshot (A2 = 0.2 V, nowind, full panel) — mean K_t:")
snap = (rowA[(rowA["amp_v"] == 0.20) & (rowA["wind"] == "no")]
          .pivot_table(index="freq", columns="Mooring", values="mean"))
print(snap.round(3).to_string())

print("\n   Row A snapshot (A2 = 0.2 V, fullwind, full panel) — mean K_t:")
snap = (rowA[(rowA["amp_v"] == 0.20) & (rowA["wind"] == "full")]
          .pivot_table(index="freq", columns="Mooring", values="mean"))
print(snap.round(3).to_string())

print("\n   Row B snapshot (A2 = 0.2 V, nowind, above_50 mooring) — mean K_t:")
snap = (rowB[(rowB["amp_v"] == 0.20) & (rowB["wind"] == "no")]
          .pivot_table(index="freq", columns="PanelCondition", values="mean"))
print(snap.round(3).to_string())

print("\n   Row B snapshot (A2 = 0.2 V, fullwind, above_50 mooring) — mean K_t:")
snap = (rowB[(rowB["amp_v"] == 0.20) & (rowB["wind"] == "full")]
          .pivot_table(index="freq", columns="PanelCondition", values="mean"))
print(snap.round(3).to_string())

print("\nDone.")
