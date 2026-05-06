"""
Probe-config bias within below_90_loose230 mooring
====================================================

`below_90_loose230` is the only mooring with substantial probe-config
variation (h100/high 137, h100/low 46, h136/high 7, h272/high 18). If the
user's claim — "probe config = measurement uncertainty, not physics" — is
right, then K_t at a given (freq, panel, wind, amp) cell should be
statistically the same across configs.

This diagnostic plots K_t vs frequency, full panel only, with one line
per probe config inside `below_90_loose230`. Visual overlap = no bias.

Outputs:
    analysis_scratch/probe_config_bias_within_mooring.pdf
    analysis_scratch/probe_config_bias_within_mooring_summary.csv
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
from wavescripts.plot_utils import apply_thesis_style, WIND_COLOR_MAP

apply_thesis_style()

SCRATCH_PDF = Path(__file__).parent / "probe_config_bias_within_mooring.pdf"
SCRATCH_CSV = Path(__file__).parent / "probe_config_bias_within_mooring_summary.csv"

# ── 1. Load & filter ───────────────────────────────────────────────────────────
print("1. Loading processed folders …")
all_dirs = sorted(glob.glob(str(BASE / "waveprocessed" / "PROCESSED-*")))
meta, _, _, _ = load_analysis_data(*all_dirs, load_processed=False)

sel = meta[
    meta["WaveFrequencyInput [Hz]"].notna()
    & (meta["WaveFrequencyInput [Hz]"] > 0)
    & (meta["quality_flag"] == "ok")
    & meta["OUT/IN (FFT)"].notna()
    & meta["WindCondition"].isin(["no", "full"])
    & (meta["WaveFrequencyInput [Hz]"] < 2.0)
    & (meta["OUT/IN (FFT)"].between(0.1, 2.0))
    & (meta["Mooring"] == "below_90_loose230")
    & (meta["PanelCondition"] == "full")
].copy()

sel["cfg"] = (sel["probe_height_mm"].astype("Int64").astype(str)
              + "/" + sel["probe_range_mode"].astype(str))
print(f"   {len(sel)} rows in below_90_loose230 / full panel")
print("   counts per probe config:")
print(sel["cfg"].value_counts().to_string())

# Quantize amplitude.
sel["amp_v"] = sel["WaveAmplitudeInput [Volt]"].apply(lambda v: round(float(v), 2))

# ── 2. Aggregate per (cfg, freq, wind) — pool A1+A2+A3 since cells are tiny ──
agg = (sel.groupby(["cfg", "WaveFrequencyInput [Hz]", "WindCondition"])
          ["OUT/IN (FFT)"]
          .agg(["mean", "std", "count"])
          .reset_index()
          .rename(columns={"WaveFrequencyInput [Hz]": "freq",
                            "WindCondition": "wind"}))
agg.to_csv(SCRATCH_CSV, index=False)

# ── 3. Plot ────────────────────────────────────────────────────────────────────
CFG_ORDER  = ["100/high", "100/low", "136/high", "272/high"]
CFG_COLOR  = {
    "100/high": "#1f77b4",
    "100/low":  "#ff7f0e",
    "136/high": "#2ca02c",
    "272/high": "#d62728",
}
CFG_MARKER = {
    "100/high": "o",
    "100/low":  "s",
    "136/high": "^",
    "272/high": "D",
}

fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8),
                          sharex=True, sharey=True)

for ax, wind in zip(axes, ["no", "full"]):
    for cfg in CFG_ORDER:
        sub = agg[(agg["cfg"] == cfg) & (agg["wind"] == wind)]
        if sub.empty:
            continue
        sub = sub.sort_values("freq")
        ax.errorbar(
            sub["freq"], sub["mean"],
            yerr=sub["std"].fillna(0),
            marker=CFG_MARKER[cfg], ms=6, mew=0.4, mec="black",
            color=CFG_COLOR[cfg], lw=1.2, ls="-",
            capsize=2.5, capthick=0.8, elinewidth=0.7,
            alpha=0.9, label=cfg,
        )
    title_color = WIND_COLOR_MAP[wind]
    ax.set_title(f"{'Uten' if wind=='no' else 'Med'} vind",
                  fontsize=11, color=title_color)
    ax.set_xlabel("Frekvens (Hz)", fontsize=10)
    ax.axhline(1.0, color="black", lw=0.6, ls=":", alpha=0.5)
    ax.grid(which="major", alpha=0.30, lw=0.6)
    ax.grid(which="minor", alpha=0.15, lw=0.4)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))

axes[0].set_ylabel("$K_t$ = OUT/IN (FFT)", fontsize=11)
axes[1].legend(loc="best", fontsize=8, framealpha=0.92,
                title="Probe cfg (height_mm/range)", title_fontsize=8)

fig.suptitle("Probe-cfg innenfor below_90_loose230 / full panel "
              "— sjekk at cfg ikke biaser $K_t$", fontsize=11, y=1.0)
fig.tight_layout()

SCRATCH_PDF.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(SCRATCH_PDF, bbox_inches="tight", pad_inches=0.02)
print(f"\n   Saved → {SCRATCH_PDF.relative_to(BASE)}")
plt.close(fig)

# ── 4. Quantify spread between configs ─────────────────────────────────────────
# At each (freq, wind), compute (max − min) of cfg means; report median.
cell_disagree = (agg.groupby(["freq", "wind"])["mean"]
                     .agg(lambda s: s.max() - s.min() if len(s) > 1 else np.nan)
                     .dropna())
print(f"\n   Inter-cfg spread (max−min of K_t means) per (freq, wind), "
      f"n_cells = {len(cell_disagree)}:")
print(f"     median: {cell_disagree.median():.4f}")
print(f"     90th %: {cell_disagree.quantile(0.9):.4f}")
print(f"     max:    {cell_disagree.max():.4f}")
print("   For reference: typical within-cell σ at 1.3-1.6 Hz is ~0.07.")

print("\nDone.")
