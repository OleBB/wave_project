"""
Probe-config bias within below_90_loose230 — like-for-like
============================================================

Restrict the comparison to cells (freq, wind, amp) where ≥2 probe
configs each have ≥3 runs. Then compute the residual disagreement
between configs at the same physical condition. This isolates
"config bias" from "I sampled different cells with different configs".

Outputs:
    analysis_scratch/probe_config_bias_likeforlike.pdf
    analysis_scratch/probe_config_bias_likeforlike_summary.csv
    analysis_scratch/probe_config_bias_likeforlike_cells.csv
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
from wavescripts.plot_utils import apply_thesis_style, WIND_COLOR_MAP

apply_thesis_style()

MIN_RUNS_PER_CFG = 1     # was 2 — relaxed to "all data" per user request
MIN_CFGS_PER_CELL = 2

_TAG = "all" if MIN_RUNS_PER_CFG <= 1 else f"n{MIN_RUNS_PER_CFG}plus"
SCRATCH_PDF      = Path(__file__).parent / f"probe_config_bias_likeforlike_{_TAG}.pdf"
SCRATCH_CSV_AGG  = Path(__file__).parent / f"probe_config_bias_likeforlike_{_TAG}_summary.csv"
SCRATCH_CSV_CELL = Path(__file__).parent / f"probe_config_bias_likeforlike_{_TAG}_cells.csv"

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
sel["amp_v"] = sel["WaveAmplitudeInput [Volt]"].apply(lambda v: round(float(v), 2))
sel = sel.rename(columns={"WaveFrequencyInput [Hz]": "freq",
                           "WindCondition": "wind"})
print(f"   {len(sel)} rows in below_90_loose230 / full panel")

# ── 2. Build per-(cell, cfg) aggregate, then keep only like-for-like cells ─────
per_cell_cfg = (sel.groupby(["freq", "wind", "amp_v", "cfg"])
                    ["OUT/IN (FFT)"]
                    .agg(["mean", "std", "count"])
                    .reset_index())

# Keep only cell-cfg rows with at least MIN_RUNS_PER_CFG runs.
per_cell_cfg = per_cell_cfg[per_cell_cfg["count"] >= MIN_RUNS_PER_CFG].copy()

# In each cell, count surviving configs.
cfgs_per_cell = (per_cell_cfg.groupby(["freq", "wind", "amp_v"])
                              .size().rename("n_cfgs").reset_index())
keep_cells = cfgs_per_cell[cfgs_per_cell["n_cfgs"] >= MIN_CFGS_PER_CELL]
keep_keys = set(map(tuple, keep_cells[["freq", "wind", "amp_v"]].values))

per_cell_cfg["cell_key"] = list(zip(per_cell_cfg["freq"],
                                      per_cell_cfg["wind"],
                                      per_cell_cfg["amp_v"]))
ll = per_cell_cfg[per_cell_cfg["cell_key"].isin(keep_keys)].copy()
print(f"\n2. Like-for-like cells (≥{MIN_CFGS_PER_CELL} cfgs × ≥"
      f"{MIN_RUNS_PER_CFG} runs each):")
print(f"   {len(keep_cells)} cells survive (out of "
      f"{len(cfgs_per_cell)} total).")
print(f"   {len(ll)} (cell, cfg) rows in the comparison.")

if ll.empty:
    print("\n   No cells survive — nothing to compare. Done.")
    sys.exit(0)

ll.to_csv(SCRATCH_CSV_CELL, index=False)
print(f"   Per-(cell, cfg) means → {SCRATCH_CSV_CELL.relative_to(BASE)}")

# ── 3. Residual disagreement per surviving cell ────────────────────────────────
def _spread(g):
    return pd.Series({
        "mean_grand":  g["mean"].mean(),
        "spread_max_min":  g["mean"].max() - g["mean"].min(),
        "spread_std":  g["mean"].std(ddof=0),
        "n_cfgs":      len(g),
        "cfgs":        ",".join(sorted(g["cfg"].unique())),
        "n_runs_total": int(g["count"].sum()),
    })

cell_spread = (ll.groupby(["freq", "wind", "amp_v"])
                  .apply(_spread).reset_index())
cell_spread.to_csv(SCRATCH_CSV_AGG, index=False)
print(f"   Per-cell spread → {SCRATCH_CSV_AGG.relative_to(BASE)}\n")
print(cell_spread.to_string(index=False))

print("\n   Headline:")
print(f"     median (max−min) across cells: "
      f"{cell_spread['spread_max_min'].median():.4f}")
print(f"     90th %:                         "
      f"{cell_spread['spread_max_min'].quantile(0.9):.4f}")
print(f"     max:                            "
      f"{cell_spread['spread_max_min'].max():.4f}")
print("     Reference: typical within-cell σ at 1.3-1.6 Hz ≈ 0.07.")

# Reference: within-config σ_K_t inside surviving cells (the "pure measurement
# noise" scale) — average σ from per_cell_cfg["std"] across the same cells.
ref_sigma = ll["std"].dropna()
if len(ref_sigma):
    print(f"     Median within-(cell, cfg) σ: {ref_sigma.median():.4f}  "
          f"(n={len(ref_sigma)} cells×cfgs)")

# ── 4. Plot ────────────────────────────────────────────────────────────────────
CFG_ORDER  = ["100/high", "100/low", "136/high", "272/high"]
CFG_COLOR  = {
    "100/high": "#1f77b4",
    "100/low":  "#ff7f0e",
    "136/high": "#2ca02c",
    "272/high": "#d62728",
}
CFG_MARKER = {
    "100/high": "o", "100/low":  "s",
    "136/high": "^", "272/high": "D",
}

# Build a stable cell ordering for the x-axis: sort by freq, wind, amp.
cell_order = (cell_spread.sort_values(["freq", "wind", "amp_v"])
                          [["freq", "wind", "amp_v"]]
                          .apply(tuple, axis=1).tolist())
cell_xpos = {key: i for i, key in enumerate(cell_order)}
def _cell_label(k):
    f, w, a = k
    wind_short = "no" if w == "no" else "fw"
    return f"{f:.1f}\n{wind_short}\nA{int(round(a*10))}"

fig, ax = plt.subplots(figsize=(max(7.0, 0.6 * len(cell_order) + 3),
                                  5.0))

# For each (cell, cfg), plot a point. Add jitter only for visual separation
# of overlapping configs in same cell (very small).
for _, row in ll.iterrows():
    key = (row["freq"], row["wind"], row["amp_v"])
    x = cell_xpos[key]
    cfg = row["cfg"]
    yerr = row["std"] if pd.notna(row["std"]) else 0.0
    ax.errorbar(
        x, row["mean"], yerr=yerr,
        marker=CFG_MARKER.get(cfg, "x"),
        ms=8, mew=0.5, mec="black",
        color=CFG_COLOR.get(cfg, "gray"),
        capsize=3, capthick=0.8, elinewidth=0.7,
        ls="None", zorder=3, alpha=0.95,
    )

# Connect configs within each cell with a thin grey vertical line for clarity.
for key in cell_order:
    grp = ll[ll["cell_key"] == key]
    if len(grp) < 2:
        continue
    x = cell_xpos[key]
    ymin, ymax = grp["mean"].min(), grp["mean"].max()
    ax.plot([x, x], [ymin, ymax], color="gray", lw=0.8, alpha=0.45,
             zorder=1)

# x-axis labels.
ax.set_xticks(list(cell_xpos.values()))
ax.set_xticklabels([_cell_label(k) for k in cell_order],
                    fontsize=8.5)
ax.set_xlabel("Cell (freq Hz / wind / amp tier)", fontsize=10)
ax.set_ylabel("$K_t$ = OUT/IN (FFT)", fontsize=11)
ax.axhline(1.0, color="black", lw=0.6, ls=":", alpha=0.5)
ax.grid(which="major", axis="y", alpha=0.30, lw=0.6)
ax.grid(which="minor", axis="y", alpha=0.15, lw=0.4)

cfg_handles = [
    mlines.Line2D([], [], color=CFG_COLOR[c], marker=CFG_MARKER[c],
                  ms=8, mec="black", mew=0.4, ls="None",
                  label=c) for c in CFG_ORDER
]
ax.legend(handles=cfg_handles, loc="best", fontsize=8,
           title="Probe cfg (height_mm/range)", title_fontsize=8,
           framealpha=0.92)

ax.set_title(
    f"Like-for-like: probe-cfg innenfor below_90_loose230 / full panel\n"
    f"(celler m. ≥{MIN_CFGS_PER_CELL} cfgs × ≥{MIN_RUNS_PER_CFG} runs hver)",
    fontsize=10,
)

fig.tight_layout()
fig.savefig(SCRATCH_PDF, bbox_inches="tight", pad_inches=0.02)
print(f"\n   Saved → {SCRATCH_PDF.relative_to(BASE)}")
plt.close(fig)

print("\nDone.")
