"""
σ ratio (fullwind / nowind) of K_t vs frequency
================================================

For each (frequency, panel) cell, compute the standard deviation of K_t
under nowind and fullwind, then plot the ratio σ_full / σ_no vs frequency.
A ratio > 1 means wind inflates measurement spread; ratio < 1 means it
tightens it (the panel is locking the response).

Cells with n_no < 3 or n_full < 3 are dropped — std isn't meaningful with
n=1 or n=2.

Top panel: absolute σ_no and σ_full vs freq, faceted by panel.
Bottom panel: ratio σ_full / σ_no vs freq (log-y, ref line at 1).

Outputs:
    analysis_scratch/std_ratio_vs_freq.pdf
    output/FIGURES/ch05_std_ratio_vs_freq.pdf
    analysis_scratch/std_ratio_vs_freq_summary.csv
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
from wavescripts.plot_utils import WIND_COLOR_MAP, apply_thesis_style

apply_thesis_style()

SCRATCH_PDF = Path(__file__).parent / "std_ratio_vs_freq.pdf"
SCRATCH_CSV = Path(__file__).parent / "std_ratio_vs_freq_summary.csv"
OUT_PDF     = BASE / "output" / "FIGURES" / "ch05_std_ratio_vs_freq.pdf"
OUT_PDF.parent.mkdir(parents=True, exist_ok=True)

MIN_N = 3   # drop cells with fewer

# ── 1. Load & filter ───────────────────────────────────────────────────────────
print("1. Loading processed folders …")
all_dirs = sorted(glob.glob(str(BASE / "waveprocessed" / "PROCESSED-*")))
meta, _, _, _ = load_analysis_data(*all_dirs, load_processed=False)
print(f"   {len(meta)} total rows")

sel = meta[
    meta["WaveFrequencyInput [Hz]"].notna()
    & (meta["WaveFrequencyInput [Hz]"] > 0)
    & meta["PanelCondition"].isin(["full", "reverse"])
    & (meta["quality_flag"] == "ok")
    & meta["OUT/IN (FFT)"].notna()
    & meta["WindCondition"].isin(["no", "full"])
    & (meta["OUT/IN (FFT)"].between(0.1, 2.0))
    & (meta["WaveFrequencyInput [Hz]"] < 2.0)
].copy()
print(f"   {len(sel)} rows after filter")

# ── 2. Per (freq, panel, wind) std + n ─────────────────────────────────────────
g = (sel.groupby(["WaveFrequencyInput [Hz]", "PanelCondition", "WindCondition"])
        ["OUT/IN (FFT)"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"WaveFrequencyInput [Hz]": "freq",
                          "PanelCondition": "panel",
                          "WindCondition":  "wind"}))

# Pivot so each (freq, panel) row has both wind columns side-by-side.
piv = g.pivot_table(index=["freq", "panel"], columns="wind",
                     values=["mean", "std", "count"])
piv.columns = [f"{a}_{b}" for a, b in piv.columns]
piv = piv.reset_index()

# Keep only cells with both nowind AND fullwind data, both n >= MIN_N.
keep = (
    piv["count_no"].ge(MIN_N).fillna(False)
    & piv["count_full"].ge(MIN_N).fillna(False)
)
piv = piv[keep].copy()
piv["std_ratio"] = piv["std_full"] / piv["std_no"]
piv["mean_diff_full_minus_no"] = piv["mean_full"] - piv["mean_no"]

piv.to_csv(SCRATCH_CSV, index=False)
print(f"\n   Summary → {SCRATCH_CSV.relative_to(BASE)}")
print(piv.to_string(index=False))

# ── 3. Plot ────────────────────────────────────────────────────────────────────
PANEL_MARKER = {"full": "o", "reverse": "D"}    # circle / diamond
PANEL_LABEL  = {"full": "full",  "reverse": "revers"}

fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, figsize=(7.0, 6.5), sharex=True,
    gridspec_kw={"height_ratios": [1.2, 1.0], "hspace": 0.10},
)

# Top: absolute σ_no, σ_full vs freq.
for panel, marker in PANEL_MARKER.items():
    sub = piv[piv["panel"] == panel].sort_values("freq")
    if sub.empty:
        continue
    ax_top.plot(sub["freq"], sub["std_no"],
                marker=marker, color=WIND_COLOR_MAP["no"],
                lw=1.0, ms=8, mec="black", mew=0.4,
                label=f"σ uten vind, {PANEL_LABEL[panel]}")
    ax_top.plot(sub["freq"], sub["std_full"],
                marker=marker, color=WIND_COLOR_MAP["full"],
                lw=1.0, ms=8, mec="black", mew=0.4,
                label=f"σ med vind, {PANEL_LABEL[panel]}")

ax_top.set_ylabel(r"$\sigma(K_t)$", fontsize=11,
                   rotation=0, ha="right", va="center")
ax_top.set_yscale("log")
ax_top.grid(which="major", alpha=0.30, lw=0.6)
ax_top.grid(which="minor", alpha=0.15, lw=0.4)
ax_top.legend(fontsize=8, loc="best", framealpha=0.92, ncol=2)
ax_top.set_title("Spredning i $K_t$ per frekvens — uten/med vind", fontsize=11)

# Bottom: ratio.
for panel, marker in PANEL_MARKER.items():
    sub = piv[piv["panel"] == panel].sort_values("freq")
    if sub.empty:
        continue
    ax_bot.plot(sub["freq"], sub["std_ratio"],
                marker=marker, color="black",
                lw=1.0, ms=9, mfc=("#444" if panel == "full" else "white"),
                mec="black", mew=0.8,
                label=f"{PANEL_LABEL[panel]} panel")
    # n labels next to each point.
    for _, row in sub.iterrows():
        ax_bot.annotate(
            f"{int(row['count_no'])}/{int(row['count_full'])}",
            xy=(row["freq"], row["std_ratio"]),
            xytext=(0, -12 if panel == "full" else 9),
            textcoords="offset points",
            fontsize=7, color="#666", ha="center")

ax_bot.axhline(1.0, color="gray", lw=0.8, ls="--", alpha=0.7)
ax_bot.text(0.5, 1.02, "vind = ingen endring i spredning", transform=None,
             fontsize=8, color="gray", ha="left", va="bottom",
             clip_on=False, zorder=5,
             bbox=dict(facecolor="white", alpha=0.8, edgecolor="none",
                       pad=1.2),
             # x in data coords; annotate near a sensible spot.
             )
# Above call has weird coord mix — replace with clean axes-coords annotation.
ax_bot.lines  # noop; the previous text used wrong transform — fix below.

# Replace the misplaced annotation cleanly.
# (The above line was mistakenly added; we'll remove via clearing texts that
# match its content. Easier: just don't add the annotation — the dashed line
# at 1 + the y-axis label is enough.)
for txt in list(ax_bot.texts):
    if "vind = ingen endring" in txt.get_text():
        txt.remove()

ax_bot.set_yscale("log")
ax_bot.set_ylabel(r"$\sigma_{\mathrm{med\;vind}}\,/\,\sigma_{\mathrm{uten\;vind}}$",
                   fontsize=10)
ax_bot.set_xlabel("Frekvens (Hz)", fontsize=11)
ax_bot.grid(which="major", alpha=0.30, lw=0.6)
ax_bot.grid(which="minor", alpha=0.15, lw=0.4)
ax_bot.xaxis.set_major_locator(MultipleLocator(0.1))
ax_bot.legend(fontsize=8, loc="best", framealpha=0.92,
               title="(annoteringer: $n_\\mathrm{uten}/n_\\mathrm{med}$)",
               title_fontsize=7)

# Cosmetic y-limit so the ref line at 1 isn't at the edge.
ymin = max(0.1, piv["std_ratio"].min() * 0.7)
ymax = min(10.0, piv["std_ratio"].max() * 1.4)
ax_bot.set_ylim(ymin, ymax)

# Annotate the y=1 line on the bottom axis with a clean axes-coord label.
ax_bot.text(0.985, 1.0, " ingen endring",
             transform=ax_bot.get_yaxis_transform(),
             fontsize=7, color="gray", ha="right", va="bottom")

fig.tight_layout()

SCRATCH_PDF.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(SCRATCH_PDF, bbox_inches="tight", pad_inches=0.02)
print(f"\n   Saved → {SCRATCH_PDF.relative_to(BASE)}")
fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.02)
print(f"   Saved → {OUT_PDF.relative_to(BASE)}")
plt.close(fig)

# ── 4. Cross-frequency print ───────────────────────────────────────────────────
print("\n   σ ratio (med vind / uten vind) by freq, panel:")
print(piv[["freq", "panel", "count_no", "count_full",
            "std_no", "std_full", "std_ratio"]]
       .to_string(index=False))

# Bracketed summary: mean of std_ratio in low-freq vs high-freq bands.
LOW_HI_SPLIT = 1.0
low = piv[piv["freq"] < LOW_HI_SPLIT]
high = piv[piv["freq"] >= LOW_HI_SPLIT]
print(f"\n   Median σ ratio: f<{LOW_HI_SPLIT} Hz → "
      f"{low['std_ratio'].median():.2f}  (n_cells={len(low)});  "
      f"f≥{LOW_HI_SPLIT} Hz → "
      f"{high['std_ratio'].median():.2f}  (n_cells={len(high)})")

print("\nDone.")
