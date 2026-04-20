#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Surface-elevation statistics under wind-only (no-wave) runs.

Wave-probe analogue of windprofile_combined.py fig11. For each wind-only
no-wave run and each of the four analysis-probe positions, compute from the
zeroed eta_{pos} time series:

    sigma_eta       = std(eta)                 [mm]        — surface roughness
    skew(eta)                                             — PDF asymmetry
    excess_kurt     = kurtosis(eta, fisher=True)          — heavy-tails
    mm_gap_norm     = (mean(eta) - median(eta)) / sigma    — same symmetry check
                                                             used for the pitot

The script mirrors fig11: 3 horizontal panels, fullwind in tab:red, lowestwind
in tab:green, scatter-no-lines. Gaussian-reference shaded bands at |skew|<0.3
and |excess_kurt|<1. Y-axis is categorical probe position (not log height).

Run from any cwd: the script uses absolute paths for waveprocessed/.
"""
import os
import sys
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                # headless — avoid plt.show() hang
import matplotlib.pyplot as plt      # noqa: E402
from scipy.stats import skew as scipy_skew, kurtosis as scipy_kurtosis

# ── make sure wavescripts package is importable ─────────────────────────────
# Script lives in <repo>/analysis_scratch; add <repo> to path.
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from wavescripts.improved_data_loader import (           # noqa: E402
    load_analysis_data,
    load_processed_dfs,
)
from wavescripts.constants import MEASUREMENT            # noqa: E402

# ── config ──────────────────────────────────────────────────────────────────
# waveprocessed/ lives in the main repo; this worktree does not have its own
# copy. Use the canonical absolute path so the script runs from any cwd.
PROCESSED_ROOT = Path("/Users/ole/Kodevik/wave_project/waveprocessed")

# Probes covered in CLAUDE.md §8 across configs. We only plot positions that
# actually carry data for a given run (eta_{pos} columns present).
PROBE_POSITIONS = [
    "8804/250",     # upstream, center, wind-exposed
    "9373/170",     # IN, wall-side, wind-exposed
    "9373/340",     # parallel to 9373/170, far-side, wind-exposed
    "12400/250",    # OUT, panel-sheltered
]

# Plot order top→bottom (fig11 uses y=height; we use y=position)
# Plot from upstream (top) to downstream (bottom) so the reader sees the
# evolution along the fetch.
PROBE_Y_ORDER = ["12400/250", "9373/340", "9373/170", "8804/250"]

# Seconds to discard at the start of each run (fan/wind spin-up transient).
# CLAUDE.md §9 notes the fromZeroToMaxWind runs are a *different* class used
# to characterise setup; we exclude them. Steady-fan wind-only runs still
# have a small transient we crop conservatively.
TRIM_START_S = 5.0

FS = float(MEASUREMENT.SAMPLING_RATE)  # 250 Hz

FW_COLOR = "tab:red"
LW_COLOR = "tab:green"

SAVE = True
OUT_PDF = _HERE / "windwave_eta_statistics.pdf"
OUT_MD_CSV = _HERE / "windwave_eta_statistics_summary.csv"


# ── discover datasets ───────────────────────────────────────────────────────
def _discover_processed_dirs(root: Path) -> list[Path]:
    """Return every PROCESSED-* directory under ``root`` that contains a
    meta.json (the subset that actually loads)."""
    dirs = sorted(p for p in root.glob("PROCESSED-*") if (p / "meta.json").exists())
    if not dirs:
        raise FileNotFoundError(f"no PROCESSED-* dirs with meta.json under {root}")
    return dirs


PROCESSED_DIRS = _discover_processed_dirs(PROCESSED_ROOT)
print(f"Using {len(PROCESSED_DIRS)} PROCESSED-* directories from {PROCESSED_ROOT}")

# ── load meta (no time-series yet) ───────────────────────────────────────────
combined_meta, _, _, _ = load_analysis_data(*PROCESSED_DIRS, load_processed=False)
print(f"Loaded combined_meta: {len(combined_meta)} total rows")

# ── filter: wind-only, no-wave, exclude ramp/experimental runs ──────────────
# Rules:
#   - WaveFrequencyInput [Hz] is NaN         → no paddle wave
#   - WindCondition in {"full", "lowest"}    → wind actually on
#   - run_category == "nowave_control"       → excludes:
#         "wind_decay"  (fromMax* ramp-down)
#         "experimental" (fromZeroToMaxWind and other non-standard)
#         "partial"/"diagnostic"
#     (nowave_control itself matches filenames containing "nowave",
#      "stillwater", "ulsonly", "nopaddle". That's our target set.)
#   - Belt-and-suspenders: filename must not contain "fromzero" or "frommax".
is_nowave = combined_meta["WaveFrequencyInput [Hz]"].isna()
is_wind   = combined_meta["WindCondition"].astype(str).str.lower().isin(["full", "lowest"])
is_nowcat = combined_meta["run_category"].astype(str).str.strip() == "nowave_control"

_lower_name = combined_meta["path"].astype(str).apply(lambda p: Path(p).name.lower())
is_not_ramp = ~_lower_name.str.contains(r"fromzero|frommax", regex=True)

filt_mask = is_nowave & is_wind & is_nowcat & is_not_ramp
meta_wind = combined_meta[filt_mask].copy()

print(f"Wind-only no-wave runs after filter: {len(meta_wind)}")
print(f"  fullwind runs:    {(meta_wind['WindCondition']=='full').sum()}")
print(f"  lowestwind runs:  {(meta_wind['WindCondition']=='lowest').sum()}")

# Sanity report: anything we filtered out that might deserve a mention
n_dropped_windramp = int(is_nowave.sum() & is_wind.sum()) - len(meta_wind)
_excluded = combined_meta[is_nowave & is_wind & ~(is_nowcat & is_not_ramp)]
if len(_excluded):
    print(f"  excluded (ramp/experimental): {len(_excluded)}")
    for nm, cat in _excluded[["path", "run_category"]].itertuples(index=False):
        print(f"    [{cat:15s}] {Path(nm).name}")

# ── load only the filtered time series ─────────────────────────────────────
paths_needed = meta_wind["path"].tolist()
print(f"\nLoading processed_dfs for {len(paths_needed)} runs (subset)...")
dfs = load_processed_dfs(*PROCESSED_DIRS, paths=paths_needed)
print(f"  loaded {len(dfs)} DataFrames")


# ── compute stats per run × probe ───────────────────────────────────────────
def _row_stats(sig: np.ndarray) -> dict:
    """sigma_eta [mm], skew, excess kurtosis (Fisher), (mean-median)/sigma."""
    sig = sig[np.isfinite(sig)]
    n = sig.size
    if n < 50:    # need enough samples for moments to mean anything
        return {"n": n, "sigma": np.nan, "skew": np.nan,
                "excess_kurt": np.nan, "mm_gap_norm": np.nan}
    sigma = float(np.std(sig, ddof=1))
    mean  = float(np.mean(sig))
    median = float(np.median(sig))
    # scipy kurtosis with fisher=True returns excess kurtosis (Gaussian=0)
    return {
        "n":           n,
        "sigma":       sigma,
        "skew":        float(scipy_skew(sig, bias=False)),
        "excess_kurt": float(scipy_kurtosis(sig, fisher=True, bias=False)),
        "mm_gap_norm": (mean - median) / sigma if sigma > 1e-9 else 0.0,
    }


rows = []
trim_n = int(round(TRIM_START_S * FS))
for _, r in meta_wind.iterrows():
    path = r["path"]
    df = dfs.get(path)
    if df is None:
        print(f"  skip (no df loaded): {Path(path).name}")
        continue
    # Drop leading transient: fan must reach steady state. For ~1–5 min runs,
    # trimming 5 s has negligible impact on the PDF statistics.
    if len(df) <= trim_n + 50:
        print(f"  skip (too short after trim): {Path(path).name}")
        continue
    for pos in PROBE_POSITIONS:
        col = f"eta_{pos}"
        if col not in df.columns:
            continue
        sig = df[col].to_numpy()[trim_n:]
        st = _row_stats(sig)
        rows.append({
            "path":          path,
            "filename":      Path(path).name,
            "WindCondition": r["WindCondition"],
            "file_date":     r.get("file_date"),
            "probe":         pos,
            **st,
        })

stats_df = pd.DataFrame(rows)
print(f"\nStats dataframe: {len(stats_df)} (run × probe) rows")
if len(stats_df) == 0:
    sys.exit("No stats computed — nothing to plot. Check filters / waveprocessed/.")

# Save tidy CSV alongside the PDF
stats_df.sort_values(["WindCondition", "probe", "filename"]).to_csv(
    OUT_MD_CSV, index=False, float_format="%.4f"
)
print(f"Saved CSV: {OUT_MD_CSV}")

# ── compact text report (medians per probe × condition) ─────────────────────
summary = (
    stats_df
    .groupby(["WindCondition", "probe"])
    .agg(n_runs=("n", "size"),
         sigma_med=("sigma",       "median"),
         sigma_min=("sigma",       "min"),
         sigma_max=("sigma",       "max"),
         skew_med=("skew",         "median"),
         ek_med=("excess_kurt",    "median"),
         ek_min=("excess_kurt",    "min"),
         ek_max=("excess_kurt",    "max"),
         gap_med=("mm_gap_norm",   "median"))
    .reset_index()
)
print("\nMedian stats per (WindCondition, probe):")
print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


# ── figure (mirrors fig11) ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(11.5, 5.2), sharey=True)
axS, axK, axSk = axes    # sigma, excess kurtosis, skew

# Y-axis: categorical probe positions
y_map = {p: i for i, p in enumerate(PROBE_Y_ORDER)}

def _plot_panel(ax, xcol, xlabel, xlim=None, ref_band=None, ref_line=None):
    """Scatter; points outside xlim are clipped to the edge and marked with '×'."""
    for cond, color, marker in [("full", FW_COLOR, "D"),
                                ("lowest", LW_COLOR, "o")]:
        sub = stats_df[stats_df["WindCondition"] == cond]
        if len(sub) == 0:
            continue
        xs = sub[xcol].to_numpy()
        ys = sub["probe"].map(y_map).to_numpy()
        # Small y-jitter so overlapping runs are visible
        ys_j = ys + np.random.uniform(-0.12, 0.12, size=len(ys))
        if xlim is not None:
            clipped = (xs < xlim[0]) | (xs > xlim[1])
        else:
            clipped = np.zeros_like(xs, dtype=bool)
        # In-range points
        ax.scatter(xs[~clipped], ys_j[~clipped], marker=marker, color=color,
                   s=32, edgecolor="white", linewidth=0.5, alpha=0.8,
                   zorder=4, label=f"{cond}wind (n={len(sub)})")
        # Out-of-range points: clip to nearest edge, draw as '×' in same colour
        if clipped.any():
            xs_edge = np.where(xs > xlim[1], xlim[1] - 0.02 * (xlim[1]-xlim[0]),
                               xlim[0] + 0.02 * (xlim[1]-xlim[0]))
            ax.scatter(xs_edge[clipped], ys_j[clipped], marker="x",
                       color=color, s=55, linewidth=1.6, zorder=5,
                       label=f"{cond}wind clipped (n={int(clipped.sum())})")
    if ref_band is not None:
        ax.axvspan(*ref_band, color="green", alpha=0.07, zorder=1)
    if ref_line is not None:
        ax.axvline(ref_line, color="black", linestyle=":",
                   linewidth=1, alpha=0.6, zorder=2)
    ax.set_xlabel(xlabel)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(fontsize=8, loc="best")

np.random.seed(0)  # reproducible jitter

# σ_η panel — log-ish range is not needed; use linear with gentle padding.
sigma_max = float(np.nanmax(stats_df["sigma"]))
_plot_panel(axS, "sigma", r"$\sigma_\eta$ [mm]",
            xlim=(0, max(1.0, sigma_max * 1.1)))

# Excess kurtosis panel (fig11-style: shaded |exc.kurt|<1, dotted at 0).
# x-axis capped at [-1.5, 3] — a few pre-2026-03-21 runs reach ek ≈ 6–15
# (probe too close to water → dropouts, known issue, see MEMORY.md). They are
# drawn at the right edge as '×' so they remain visible without compressing the
# in-population range.
_plot_panel(axK, "excess_kurt", r"Excess kurtosis ($\kappa - 3$)",
            xlim=(-1.5, 3.0), ref_band=(-1, 1), ref_line=0.0)

# Skew panel — same cap logic; one run reaches skew = -1.29, clipped.
_plot_panel(axSk, "skew", "Skewness",
            xlim=(-1.2, 1.2), ref_band=(-0.3, 0.3), ref_line=0.0)

# Y-axis labels (categorical)
axS.set_yticks(list(y_map.values()))
axS.set_yticklabels(list(y_map.keys()))
axS.set_ylabel("Probe position  [mm from paddle / lateral mm]")
axS.set_ylim(-0.6, len(y_map) - 0.4)

_n_full = stats_df.loc[stats_df["WindCondition"] == "full", "path"].nunique()
_n_lw   = stats_df.loc[stats_df["WindCondition"] == "lowest", "path"].nunique()
_lw_note = (f"{_n_lw} lowestwind runs" if _n_lw
            else "no lowestwind+nowave runs in this dataset")
fig.suptitle(
    "Surface-elevation statistics under wind-only, no-wave runs\n"
    f"{_n_full} fullwind runs  •  {_lw_note}  •  "
    f"trim {TRIM_START_S:.0f}s, fs={int(FS)} Hz",
    fontsize=11,
)
fig.tight_layout()

if SAVE:
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"Saved PDF: {OUT_PDF}")

plt.close(fig)
print("Done.")
