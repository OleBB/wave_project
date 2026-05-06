"""
loose230 vs loose300 — apples-to-apples sanity check
=====================================================

The mooring × probe-range matrix has only one apples-to-apples comparison
available: lowrange-to-lowrange (canon scope). loose300 has no highrange
data, so highrange-to-highrange is not possible.

This script verifies that the broad-scope "Δ ≈ 0" finding is consistent
with the canon "Δ = +0.036" once you correct for the asymmetric
probe-mode pooling.

Steps:
  1. Compute loose230 mean K_t separately per probe-mode (lowrange-only,
     highrange-only).
  2. Compute the within-loose230 mode bias = mean(loose230_low) − mean(loose230_high).
  3. Predict broad-scope loose230 mean = weighted mix of mode-specific means
     by their broad-scope counts, and verify it matches the broad observed
     loose230 mean.
  4. Predict "apples-to-apples-broad" = what loose230 broad mean WOULD read
     if all rows had been in lowrange (= broad loose230 + bias × highrange_share),
     and compare to loose300 (always lowrange).
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
os.chdir(BASE)

from wavescripts.improved_data_loader import load_analysis_data

import glob

dirs = sorted(glob.glob("waveprocessed/PROCESSED-*"))
m, _, _, _ = load_analysis_data(*dirs, load_processed=False)

m = m[m["PanelCondition"] == "full"]
m = m[m["Mooring"].isin(["below_90_loose230", "below_90_loose300"])]
m = m[m["WaveFrequencyInput [Hz]"].isin([1.3, 1.4, 1.5, 1.6])]
m = m[m["WaveAmplitudeInput [Volt]"].apply(
    lambda v: any(abs(float(v) - a) < 1e-3 for a in [0.10, 0.20, 0.30]))]
m = m[m["WindCondition"].isin(["no", "full"])]
m = m.dropna(subset=["OUT/IN (FFT)"])
m["amp_v"] = m["WaveAmplitudeInput [Volt]"].apply(lambda v: round(float(v), 2))
m["mooring_short"] = m["Mooring"].str.replace("below_90_", "")
m["wind"] = m["WindCondition"]
m["mode"] = m["probe_range_mode"].astype(str)
m["probe_config"] = m["probe_height_mm"].astype("Int64").astype(str) + "/" + m["mode"]
m["probe_config"] = m["probe_config"].str.replace("<NA>", "?")

# We do this per (amp, freq, wind) cell so the comparison is paired.
# Then average across cells to get a global delta.

# 1. Per-cell K_t per (mooring, mode, amp, freq, wind)
g = (m.groupby(["mooring_short", "mode", "amp_v", "WaveFrequencyInput [Hz]", "wind"])
        ["OUT/IN (FFT)"].agg(["mean", "count"]).reset_index()
        .rename(columns={"mean": "Kt", "count": "n",
                          "WaveFrequencyInput [Hz]": "freq"}))

# 2. Pivot to (cell, mode) with mooring sub-key
def pv(moor, mode):
    sub = g[(g["mooring_short"] == moor) & (g["mode"] == mode)]
    return (sub.set_index(["amp_v", "freq", "wind"])
              [["Kt", "n"]]
              .rename(columns={"Kt": f"Kt_{moor}_{mode}", "n": f"n_{moor}_{mode}"}))

l230_low  = pv("loose230", "low")
l230_high = pv("loose230", "high")
l300_low  = pv("loose300", "low")

print("=" * 70)
print("Cell counts per slice (in-scope rows only)")
print("=" * 70)
print(f"  loose230 low : {l230_low['n_loose230_low'].sum()} runs across {len(l230_low)} cells")
print(f"  loose230 high: {l230_high['n_loose230_high'].sum()} runs across {len(l230_high)} cells")
print(f"  loose300 low : {l300_low['n_loose300_low'].sum()} runs across {len(l300_low)} cells")
print()

joined = l230_low.join(l230_high, how="outer").join(l300_low, how="outer").reset_index()

# ── (a) APPLES-TO-APPLES delta: lowrange-to-lowrange ────────────────────
ap = joined.dropna(subset=["Kt_loose230_low", "Kt_loose300_low"])
print("=" * 70)
print("(a) APPLES-TO-APPLES (canon = lowrange-to-lowrange) — Δ_Kt per cell")
print("=" * 70)
ap["dKt"] = ap["Kt_loose230_low"] - ap["Kt_loose300_low"]
print(ap[["amp_v", "freq", "wind", "Kt_loose230_low", "n_loose230_low",
          "Kt_loose300_low", "n_loose300_low", "dKt"]].round(3).to_string(index=False))
print()
print(f"  median Δ (apples) = {ap['dKt'].median():+.4f}  across {len(ap)} cells")
print(f"  mean   Δ (apples) = {ap['dKt'].mean():+.4f}")
print()

# ── (b) Within-loose230 probe-mode bias ─────────────────────────────────
bias = joined.dropna(subset=["Kt_loose230_low", "Kt_loose230_high"])
print("=" * 70)
print("(b) Within-loose230: probe-mode bias (low − high) per cell")
print("=" * 70)
bias["dmode"] = bias["Kt_loose230_low"] - bias["Kt_loose230_high"]
print(bias[["amp_v", "freq", "wind", "Kt_loose230_low", "n_loose230_low",
            "Kt_loose230_high", "n_loose230_high", "dmode"]].round(3).to_string(index=False))
print()
print(f"  median bias (low − high) = {bias['dmode'].median():+.4f}  across {len(bias)} cells")
print(f"  mean   bias (low − high) = {bias['dmode'].mean():+.4f}")
print()

# ── (c) Broad-scope mooring delta with mode-corrected loose230 ──────────
print("=" * 70)
print("(c) Broad-scope loose230 vs loose300, corrected for probe-mode bias")
print("=" * 70)

# loose230 broad mean per cell = weighted mean of low and high K_t with their n.
def _broad_mean(row, moor):
    klow  = row.get(f"Kt_{moor}_low")
    nlow_raw  = row.get(f"n_{moor}_low",  0)
    khigh = row.get(f"Kt_{moor}_high")
    nhigh_raw = row.get(f"n_{moor}_high", 0)
    nlow  = int(nlow_raw) if pd.notna(nlow_raw) else 0
    nhigh = int(nhigh_raw) if pd.notna(nhigh_raw) else 0
    parts, weights = [], []
    if pd.notna(klow) and nlow > 0:
        parts.append(klow);  weights.append(nlow)
    if pd.notna(khigh) and nhigh > 0:
        parts.append(khigh); weights.append(nhigh)
    if not parts:
        return np.nan, nlow, nhigh
    return float(np.average(parts, weights=weights)), nlow, nhigh

broad_rows = []
for _, r in joined.iterrows():
    bmean, nl, nh = _broad_mean(r, "loose230")
    l300 = r.get("Kt_loose300_low")
    n300 = r.get("n_loose300_low", 0) or 0
    if pd.isna(bmean) or pd.isna(l300):
        continue
    # Predict "what would loose230 broad read if all rows were lowrange?"
    # = broad_mean + bias × (highrange share)
    # bias is the loose230 low − high bias for this cell (cell-specific) when known,
    # else fall back to global median.
    if pd.notna(r.get("Kt_loose230_low")) and pd.notna(r.get("Kt_loose230_high")):
        bias_local = r["Kt_loose230_low"] - r["Kt_loose230_high"]
    else:
        bias_local = bias["dmode"].median()
    high_share = nh / (nl + nh) if (nl + nh) else 0.0
    bmean_corrected = bmean + bias_local * high_share
    broad_rows.append({
        "amp_v": r["amp_v"], "freq": r["freq"], "wind": r["wind"],
        "loose230_broad": bmean,
        "n230_low": nl, "n230_high": nh,
        "loose300_low": l300, "n300_low": int(n300),
        "high_share_in_230": high_share,
        "bias_used": bias_local,
        "loose230_broad_corrected": bmean_corrected,
        "dKt_uncorrected": bmean - l300,
        "dKt_corrected":   bmean_corrected - l300,
    })
broad_df = pd.DataFrame(broad_rows)
print(broad_df.round(3).to_string(index=False))
print()
print(f"  median Δ (broad uncorrected) = {broad_df['dKt_uncorrected'].median():+.4f}")
print(f"  median Δ (broad MODE-corrected) = {broad_df['dKt_corrected'].median():+.4f}")
print(f"  median Δ (apples / canon)     = {ap['dKt'].median():+.4f}  ← reference")
print()
print("Reading: if mode-corrected broad Δ ≈ apples/canon Δ, then the broad-vs-canon")
print("gap is fully explained by asymmetric probe-mode pooling, and both scopes")
print("agree on the underlying mooring effect.")

print("\nDone.")
