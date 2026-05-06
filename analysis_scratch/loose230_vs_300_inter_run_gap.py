"""
loose230 vs loose300 — settling time via inter_run_gap_s
=========================================================

Correction to round 3: prev_run_nperiods is the previous run's wave duration,
NOT the wait time between runs. The actual wait is in `inter_run_gap_s`.

Re-do the settling-time confounder check with the right column.

  V. Distribution of inter_run_gap_s per day × mooring (canon scope).
  W. Inside loose300 (Mar 27, the rushed day): does K_t depend on
     inter_run_gap_s and on pre-paddle σ_η_OUT (residual sloshing)?
  X. Mar 26 vs Mar 27 head-to-head, controlling for gap quartile.
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

THESIS_FREQS = [1.3, 1.4, 1.5, 1.6]
THESIS_AMPS = [0.10, 0.20, 0.30]

dirs = sorted(glob.glob("waveprocessed/PROCESSED-*"))
m, _, _, _ = load_analysis_data(*dirs, load_processed=False)

m = m[m["PanelCondition"] == "full"]
m = m[m["Mooring"].isin(["below_90_loose230", "below_90_loose300"])]
m = m[m["WindCondition"].isin(["no", "full"])]
m = m[m["WaveFrequencyInput [Hz]"].isin(THESIS_FREQS)]
m = m[m["WaveAmplitudeInput [Volt]"].apply(
    lambda v: any(abs(float(v) - a) < 1e-3 for a in THESIS_AMPS))]
m = m.dropna(subset=["OUT/IN (FFT)"])
m["amp_v"] = m["WaveAmplitudeInput [Volt]"].apply(lambda v: round(float(v), 2))
m["mooring_short"] = m["Mooring"].str.replace("below_90_", "")
m["wind"] = m["WindCondition"]
m["probe_config"] = m["probe_height_mm"].astype("Int64").astype(str) + "/" + m["probe_range_mode"].astype(str)
m["probe_config"] = m["probe_config"].str.replace("<NA>", "?")
m["date"] = pd.to_datetime(m["file_date"]).dt.strftime("%Y-%m-%d")

# Restrict to canon (h100/low) for the head-to-head questions, but keep broad
# for the loose230 distribution.
print("=" * 80)
print("V. Inter-run gap distribution per day × mooring (BROAD scope)")
print("=" * 80)
g = (m.groupby(["date", "mooring_short", "probe_config"])
       .agg(n=("inter_run_gap_s", "count"),
            gap_mean=("inter_run_gap_s", "mean"),
            gap_med=("inter_run_gap_s", "median"),
            gap_std=("inter_run_gap_s", "std"),
            gap_min=("inter_run_gap_s", "min"),
            gap_max=("inter_run_gap_s", "max")))
print(g.round(1).to_string())
print()

# CANON head-to-head
canon = m[m["probe_config"] == "100/low"]
mar26 = canon[canon["date"] == "2026-03-26"]
mar27 = canon[canon["date"] == "2026-03-27"]
print("=" * 80)
print("Canon scope (h100/low) — Mar 26 vs Mar 27 inter_run_gap_s")
print("=" * 80)
for d, lbl in [(mar26, "Mar 26 loose230"), (mar27, "Mar 27 loose300")]:
    s = d["inter_run_gap_s"].dropna()
    print(f"  {lbl}: n={len(s)}, mean={s.mean():.0f} s, median={s.median():.0f} s, "
          f"min={s.min():.0f} s, max={s.max():.0f} s")
    print(f"     quartiles: {s.quantile(0.25):.0f} / {s.median():.0f} / {s.quantile(0.75):.0f}")
print()

# Same split by wind
print("Split by wind:")
for d, lbl in [(mar26, "Mar 26 loose230"), (mar27, "Mar 27 loose300")]:
    for w in ["no", "full"]:
        s = d[d["wind"] == w]["inter_run_gap_s"].dropna()
        if len(s):
            print(f"  {lbl}, wind={w:4s}: n={len(s)}, median={s.median():.0f} s, "
                  f"q25={s.quantile(0.25):.0f}, q75={s.quantile(0.75):.0f}")
print()

# ── W. Within loose300 canon: does K_t depend on inter_run_gap_s? ────────
print("=" * 80)
print("W. Within loose300 canon: K_t × inter_run_gap_s")
print("=" * 80)
for w in ["no", "full"]:
    sub = mar27[mar27["wind"] == w].dropna(subset=["inter_run_gap_s", "OUT/IN (FFT)"])
    if len(sub) >= 4:
        r = sub[["OUT/IN (FFT)", "inter_run_gap_s"]].corr().iloc[0, 1]
        print(f"  wind={w:4s} n={len(sub)}: r(K_t, inter_run_gap_s) = {r:+.3f}")
        # quartile binning
        q = pd.qcut(sub["inter_run_gap_s"], q=min(4, len(sub) // 4 if len(sub) >= 8 else 2),
                     duplicates="drop")
        gq = sub.groupby(q, observed=True)["OUT/IN (FFT)"].agg(["count", "mean", "std"])
        print(f"    K_t by inter_run_gap quartile:")
        print(gq.round(3).to_string())
print()

# ── Inside loose300, also check vs A_in / A_out ─────────────────────────
print("Same vs A_in_FFT and A_out_FFT (for context — sloshing should bias more A_in):")
for w in ["no", "full"]:
    sub = mar27[mar27["wind"] == w].dropna(subset=["inter_run_gap_s", "IN Amplitude (FFT)", "OUT Amplitude (FFT)"])
    if len(sub) >= 4:
        r_in  = sub[["IN Amplitude (FFT)",  "inter_run_gap_s"]].corr().iloc[0, 1]
        r_out = sub[["OUT Amplitude (FFT)", "inter_run_gap_s"]].corr().iloc[0, 1]
        print(f"  wind={w:4s} n={len(sub)}: r(A_in, gap)={r_in:+.3f}, r(A_out, gap)={r_out:+.3f}")
print()

# ── X. Cross-day: cell-by-cell delta with gap stratification ────────────
print("=" * 80)
print("X. Mar 26 vs Mar 27 head-to-head, gap stratification on Mar 27")
print("=" * 80)
def _agg(df):
    return (df.groupby(["amp_v", "WaveFrequencyInput [Hz]", "wind"])
              .agg(Kt=("OUT/IN (FFT)", "mean"),
                   gap=("inter_run_gap_s", "mean"),
                   n=("OUT/IN (FFT)", "count"))
              .reset_index()
              .rename(columns={"WaveFrequencyInput [Hz]": "freq"}))

a26 = _agg(mar26).rename(columns={c: f"{c}_26" for c in ["Kt", "gap", "n"]})
a27 = _agg(mar27).rename(columns={c: f"{c}_27" for c in ["Kt", "gap", "n"]})

# Separately split Mar 27 into "long-gap" (≥ median) and "short-gap" (< median)
gap_med27 = mar27["inter_run_gap_s"].median()
mar27_long  = mar27[mar27["inter_run_gap_s"] >= gap_med27]
mar27_short = mar27[mar27["inter_run_gap_s"] <  gap_med27]
print(f"Mar 27 inter_run_gap median = {gap_med27:.0f} s")
print(f"Mar 27 long-gap subset:  n={len(mar27_long)},  median K_t fullwind = "
      f"{mar27_long[mar27_long['wind']=='full']['OUT/IN (FFT)'].median():.3f}, "
      f"nowind = {mar27_long[mar27_long['wind']=='no']['OUT/IN (FFT)'].median():.3f}")
print(f"Mar 27 short-gap subset: n={len(mar27_short)}, median K_t fullwind = "
      f"{mar27_short[mar27_short['wind']=='full']['OUT/IN (FFT)'].median():.3f}, "
      f"nowind = {mar27_short[mar27_short['wind']=='no']['OUT/IN (FFT)'].median():.3f}")
print()

a27_long  = _agg(mar27_long).rename(columns={c: f"{c}_27long" for c in ["Kt", "gap", "n"]})
a27_short = _agg(mar27_short).rename(columns={c: f"{c}_27short" for c in ["Kt", "gap", "n"]})
joined = a26.merge(a27, on=["amp_v", "freq", "wind"], how="inner")
joined = joined.merge(a27_long, on=["amp_v", "freq", "wind"], how="left")
joined = joined.merge(a27_short, on=["amp_v", "freq", "wind"], how="left")
joined["dKt_full"]  = joined["Kt_26"] - joined["Kt_27"]
joined["dKt_long"]  = joined["Kt_26"] - joined["Kt_27long"]
joined["dKt_short"] = joined["Kt_26"] - joined["Kt_27short"]
print(joined.round(3).to_string(index=False))
print()
print("Δ_Kt (Mar 26 minus Mar 27) using full Mar 27, vs only-long-gap Mar 27, vs only-short-gap Mar 27:")
print(f"  full Mar 27:        median = {joined['dKt_full'].median():+.4f}, n = {joined['dKt_full'].notna().sum()}")
print(f"  long-gap Mar 27:    median = {joined['dKt_long'].median():+.4f}, n = {joined['dKt_long'].notna().sum()}")
print(f"  short-gap Mar 27:   median = {joined['dKt_short'].median():+.4f}, n = {joined['dKt_short'].notna().sum()}")

print("\nDone.")
