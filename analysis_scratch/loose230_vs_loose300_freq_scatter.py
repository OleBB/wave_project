"""
loose230 vs loose300 — K_t vs frequency scatter, one figure per amp tier
==========================================================================

Companion to `loose230_vs_loose300_table.py`. Same canon scope; this
script visualises the table on a frequency axis. Three figures (A1 / A2 /
A3) so the reader can scan amp-by-amp without colour competing for
amplitude. Within each figure: K_t vs f, with mooring × wind = 4 groups.

Encoding:
  x       : frequency (Hz)
  y       : K_t (mean ± std, errorbars where n>1)
  hue     : wind condition  — blue uten / red med (WIND_COLOR_MAP)
  marker  : mooring × amp tier (consistent with mooring_focus_at_1_3hz_ka.pdf)
              loose300 (loose, looks "normal"): ○ A1 / □ A2 / △ A3
              loose230 (tight): 6/5/4-point star (matches the
              "revers"-style family from earlier ka plots — re-used as
              "the secondary mooring")
  fill    : all hollow so overlapping points stay visible
  line    : connect the per-(mooring, wind) means across frequencies

Outputs:
    analysis_scratch/loose230_vs_loose300_freq_A{1,2,3}.pdf
    output/FIGURES/ch05_loose230_vs_loose300_freq_A{1,2,3}.pdf
    analysis_scratch/loose230_vs_loose300_freq_scatter_summary.csv
"""

import sys
import warnings
from pathlib import Path

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
    apply_thesis_style, WIND_COLOR_MAP, apply_horizontal_ylabel,
)

apply_thesis_style()

RESULTS_DIRS = [
    BASE / "waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange",
    BASE / "waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange",
]

THESIS_FREQS = [1.3, 1.4, 1.5, 1.6]
THESIS_AMPS  = [(0.10, "A1", "0.1V"),
                (0.20, "A2", "0.2V"),
                (0.30, "A3", "0.3V")]

# ── 1. Load & filter ───────────────────────────────────────────────────────────
print("1. Loading canon results folders …")
meta, _, _, _ = load_analysis_data(*[str(d) for d in RESULTS_DIRS],
                                    load_processed=False)

m = meta.copy()
m = m[m["PanelCondition"] == "full"]
m = m[m["WaveFrequencyInput [Hz]"].isin(THESIS_FREQS)]
m = m[m["WaveAmplitudeInput [Volt]"].apply(
        lambda v: any(abs(float(v) - a) < 1e-3 for a, _, _ in THESIS_AMPS))]
if "quality_flag" in m.columns:
    m = m[m["quality_flag"].isna() | (m["quality_flag"] == "ok")]
m = m.dropna(subset=["OUT/IN (FFT)", "WindCondition", "Mooring"])
m = m[m["WindCondition"].isin(["no", "full"])]
m = m[m["Mooring"].isin(["below_90_loose230", "below_90_loose300"])]
m["amp_v"] = m["WaveAmplitudeInput [Volt]"].apply(lambda v: round(float(v), 2))
print(f"   {len(m)} rows in scope")

# ── 2. Aggregate per (amp, mooring, wind, freq) ────────────────────────────────
agg = (m.groupby(["amp_v", "Mooring", "WindCondition", "WaveFrequencyInput [Hz]"])
        ["OUT/IN (FFT)"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "Kt_mean", "std": "Kt_std", "count": "n",
                          "WaveFrequencyInput [Hz]": "freq",
                          "WindCondition": "wind", "Mooring": "mooring"}))

# ── 3. Visual constants ────────────────────────────────────────────────────────
MOORING_LBL = {"below_90_loose230": "loose230 (23 cm)",
               "below_90_loose300": "loose300 (30 cm)"}
WIND_LBL    = {"no": "uten vind", "full": "med vind"}

# Marker family per (mooring, amp). loose300 = "normal" shapes; loose230 =
# star family (re-using the "secondary panel" convention from earlier
# CH05 plots — here it tags the secondary mooring).
MOORING_AMP_MARKER = {
    ("below_90_loose300", 0.10): "o",
    ("below_90_loose300", 0.20): "s",
    ("below_90_loose300", 0.30): "^",
    ("below_90_loose230", 0.10): (6, 1, 0),
    ("below_90_loose230", 0.20): (5, 1, 0),
    ("below_90_loose230", 0.30): (4, 1, 0),
}

# Mooring → linestyle (for the connecting line between freqs).
MOORING_LS = {"below_90_loose300": "-", "below_90_loose230": "--"}

MARKER_SIZE = 95
EDGE_LW     = 1.6
ALPHA       = 0.90
LINE_LW     = 1.4
LINE_ALPHA  = 0.85

# Shared y-range across the 3 figures so they stack visually.
y_lo = m["OUT/IN (FFT)"].min() - 0.03
y_hi = m["OUT/IN (FFT)"].max() + 0.03

# ── 4. One figure per amp tier ─────────────────────────────────────────────────
summary_rows = []

for amp_v, amp_tag, amp_v_lbl in THESIS_AMPS:
    sub = agg[np.isclose(agg["amp_v"], amp_v)]
    if sub.empty:
        print(f"   [{amp_tag}] no rows — skipping")
        continue

    fig, ax = plt.subplots(figsize=(7.0, 4.6))

    for mooring in ["below_90_loose300", "below_90_loose230"]:
        for wind in ["no", "full"]:
            cell = sub[(sub["mooring"] == mooring) & (sub["wind"] == wind)]
            if cell.empty:
                continue
            cell = cell.sort_values("freq")
            color  = WIND_COLOR_MAP[wind]
            marker = MOORING_AMP_MARKER[(mooring, amp_v)]

            # Connecting line (sorted by freq).
            if len(cell) >= 2:
                ax.plot(cell["freq"], cell["Kt_mean"],
                        color=color, ls=MOORING_LS[mooring],
                        lw=LINE_LW, alpha=LINE_ALPHA, zorder=2)

            # Errorbars + scatter markers.
            ax.errorbar(
                cell["freq"], cell["Kt_mean"],
                yerr=cell["Kt_std"].fillna(0),
                fmt="None", ecolor=color,
                elinewidth=0.9, capsize=3, alpha=0.7, zorder=3,
            )
            ax.scatter(
                cell["freq"], cell["Kt_mean"],
                facecolors="none", edgecolors=color,
                marker=marker, s=MARKER_SIZE,
                linewidths=EDGE_LW, alpha=ALPHA, zorder=4,
            )
            for _, r in cell.iterrows():
                summary_rows.append(dict(
                    amp_tag=amp_tag, amp_v=amp_v,
                    mooring=mooring, wind=wind,
                    freq=float(r["freq"]),
                    n=int(r["n"]),
                    Kt_mean=float(r["Kt_mean"]),
                    Kt_std=float(r["Kt_std"]) if pd.notna(r["Kt_std"]) else None,
                ))

    ax.axhline(1.0, color="black", lw=0.6, ls="--", alpha=0.5)
    ax.set_xlim(1.25, 1.65)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("Frekvens (Hz)", fontsize=11)
    ax.set_xticks(THESIS_FREQS)
    ax.grid(which="major", alpha=0.30, lw=0.6)
    ax.grid(which="minor", alpha=0.15, lw=0.4)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(MultipleLocator(0.025))
    apply_horizontal_ylabel(ax, r"$K_t$", fontsize=12)

    # Two-block legend.
    wind_handles = [
        mlines.Line2D([], [], color=WIND_COLOR_MAP[w], marker="o",
                      ms=8, lw=LINE_LW, mfc="none",
                      mec=WIND_COLOR_MAP[w], mew=EDGE_LW,
                      label=WIND_LBL[w])
        for w in ["no", "full"]
    ]
    moor_handles = [
        mlines.Line2D([], [], color="black",
                      marker=MOORING_AMP_MARKER[(mn, amp_v)],
                      ms=8, lw=LINE_LW, mfc="none", mec="black",
                      ls=MOORING_LS[mn], mew=EDGE_LW,
                      label=MOORING_LBL[mn])
        for mn in ["below_90_loose300", "below_90_loose230"]
    ]
    leg1 = ax.legend(handles=wind_handles, loc="upper right",
                      fontsize=8, framealpha=0.92,
                      title="Vind", title_fontsize=8,
                      bbox_to_anchor=(0.995, 0.995))
    ax.add_artist(leg1)
    ax.legend(handles=moor_handles, loc="lower right",
               fontsize=8, framealpha=0.92,
               title="Mooring (slakk)", title_fontsize=8,
               bbox_to_anchor=(0.995, 0.005))

    fig.tight_layout()
    scratch = (Path(__file__).parent
                / f"loose230_vs_loose300_freq_{amp_tag}.pdf")
    out_pdf = (BASE / "output" / "FIGURES"
                / f"ch05_loose230_vs_loose300_freq_{amp_tag}.pdf")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(scratch, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    print(f"   [{amp_tag}] Saved → {scratch.relative_to(BASE)}")
    print(f"   [{amp_tag}] Saved → {out_pdf.relative_to(BASE)}")
    plt.close(fig)

csv_path = (Path(__file__).parent
             / "loose230_vs_loose300_freq_scatter_summary.csv")
pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
print(f"\n   Summary → {csv_path.relative_to(BASE)}")

print("\nDone.")
