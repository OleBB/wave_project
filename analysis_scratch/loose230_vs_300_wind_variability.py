"""
loose230 vs loose300 — day-to-day wind variability via pre-paddle σ_η
======================================================================

Correction to round 3: Windspeed in meta is just the SETPOINT (5.8 m/s flat
for fullwind). Day-to-day actual-wind variability is captured INSTEAD by
σ_η in the 3 s pre-paddle window, already computed by
analysis_scratch/wind_qc_3s.py and saved to wind_qc_3s_per_run.csv.

This script revisits the canon-only loose230 vs loose300 question with
that QC data:

  N. Per-day pre-paddle σ_η at IN and OUT, fullwind cells only.
     Mar 26 = loose230 canon, Mar 27 = loose300 canon.
  O. Per-run correlation: does K_t track σ_η (the actual local wind
     energy) inside each mooring? If wind energy is higher on Mar 27,
     and higher wind ⇒ lower K_t (via IN-probe contamination boosting
     A_in), the +0.036 canon delta could be explained by the wind QC
     gap, not by the mooring change.
  P. Pre-paddle σ_η at the OUT probe under NO-WIND (= sloshing proxy).
     Mar 26 had no fullwind→nowind transitions in canon (only 2 nowind
     runs), so this is mostly a Mar 27 self-check.
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

QC_CSV = BASE / "analysis_scratch" / "wind_qc_3s_per_run.csv"
qc = pd.read_csv(QC_CSV)
print(f"Loaded {len(qc)} rows from {QC_CSV.relative_to(BASE)}")
print(f"Columns: {list(qc.columns)}")
print()

# Load meta to merge K_t and Mooring on path
dirs = sorted(glob.glob("waveprocessed/PROCESSED-*"))
m_full, _, _, _ = load_analysis_data(*dirs, load_processed=False)
meta_keep = m_full[[
    "path", "Mooring", "PanelCondition", "WindCondition",
    "WaveFrequencyInput [Hz]", "WaveAmplitudeInput [Volt]",
    "OUT/IN (FFT)", "IN Amplitude (FFT)", "OUT Amplitude (FFT)",
    "IN ka (FFT)", "probe_height_mm", "probe_range_mode",
    "file_date",
]].copy()
meta_keep["amp_v"] = meta_keep["WaveAmplitudeInput [Volt]"].apply(
    lambda v: round(float(v), 2) if pd.notna(v) else np.nan)
meta_keep["mooring_short"] = meta_keep["Mooring"].fillna("").str.replace("below_90_", "")
meta_keep["probe_config"] = (
    meta_keep["probe_height_mm"].astype("Int64").astype(str) + "/" +
    meta_keep["probe_range_mode"].astype(str)
).str.replace("<NA>", "?")
meta_keep["date"] = pd.to_datetime(meta_keep["file_date"]).dt.strftime("%Y-%m-%d")

joined = qc.merge(meta_keep, on="path", how="left", suffixes=("", "_meta"))
print(f"Joined rows: {len(joined)}")
print()

# Restrict to canon thesis-frequency wave runs (the ones we care about for the
# loose230 vs loose300 comparison) — the QC CSV was already restricted to canon.
sel = (
    (joined["PanelCondition"] == "full")
    & joined["Mooring"].isin(["below_90_loose230", "below_90_loose300"])
    & joined["WaveFrequencyInput [Hz]"].isin([1.3, 1.4, 1.5, 1.6])
    & joined["WindCondition"].isin(["full", "no"])
    & joined["amp_v"].isin([0.10, 0.20, 0.30])
    & joined["OUT/IN (FFT)"].notna()
)
J = joined[sel].copy().reset_index(drop=True)
print(f"After in-scope filter: {len(J)} rows  "
      f"({(J['mooring_short']=='loose230').sum()} loose230, "
      f"{(J['mooring_short']=='loose300').sum()} loose300)")
print()

# ── N. Per-day pre-paddle σ_η statistics ──────────────────────────────────
print("=" * 80)
print("N. Pre-paddle σ_η per day × wind  (mm)")
print("=" * 80)
g = (J.groupby(["date", "mooring_short", "WindCondition"])
       .agg(n=("path", "count"),
            sIN_mean=("sigma_9373/170", "mean"),
            sIN_med=("sigma_9373/170", "median"),
            sIN_std=("sigma_9373/170", "std"),
            sIN_min=("sigma_9373/170", "min"),
            sIN_max=("sigma_9373/170", "max"),
            sOUT_mean=("sigma_12400/250", "mean"),
            sOUT_med=("sigma_12400/250", "median"),
            sOUT_std=("sigma_12400/250", "std"),
            sOUT_min=("sigma_12400/250", "min"),
            sOUT_max=("sigma_12400/250", "max"))
       .round(3))
print(g.to_string())
print()

# Long-run reference σ_η for context (from wind_pre_paddle_table.py)
print("Long-run reference σ_η (5 fullwind+nowave runs, durations 31-381 s):")
print("  IN  9373/170: ~4.28 mm")
print("  OUT 12400/250: ~0.36 mm")
print()

# Headline summary: fullwind only, mooring × σ_η (pre-paddle wind energy)
print("Headline — fullwind cells only, σ_η at each probe (mean across runs):")
fw = J[J["WindCondition"] == "full"]
g2 = fw.groupby("mooring_short").agg(
    n=("path", "count"),
    sIN_mean=("sigma_9373/170", "mean"),
    sIN_std=("sigma_9373/170", "std"),
    sOUT_mean=("sigma_12400/250", "mean"),
    sOUT_std=("sigma_12400/250", "std"),
).round(3)
print(g2.to_string())
print()
if len(g2) == 2:
    delta_in  = g2.loc["loose230", "sIN_mean"]  - g2.loc["loose300", "sIN_mean"]
    delta_out = g2.loc["loose230", "sOUT_mean"] - g2.loc["loose300", "sOUT_mean"]
    print(f"  Δ σ_η_IN  (Mar 26 − Mar 27)  = {delta_in:+.3f} mm")
    print(f"  Δ σ_η_OUT (Mar 26 − Mar 27)  = {delta_out:+.3f} mm")
print()

# ── O. Per-run K_t vs σ_η — does wind QC explain K_t variance? ────────────
print("=" * 80)
print("O. Per-run K_t vs σ_η (pre-paddle wind energy at IN/OUT)")
print("=" * 80)
for moor in ["loose230", "loose300"]:
    sub = fw[fw["mooring_short"] == moor].dropna(subset=["sigma_9373/170", "OUT/IN (FFT)"])
    if len(sub) >= 4:
        r_in  = sub[["OUT/IN (FFT)", "sigma_9373/170"]].corr().iloc[0, 1]
        r_out = sub[["OUT/IN (FFT)", "sigma_12400/250"]].corr().iloc[0, 1]
        r_inAfft = sub[["IN Amplitude (FFT)", "sigma_9373/170"]].corr().iloc[0, 1]
        print(f"  {moor:9s} fullwind n={len(sub):3d}")
        print(f"    Pearson(K_t,        σ_η_IN)  = {r_in:+.3f}")
        print(f"    Pearson(K_t,        σ_η_OUT) = {r_out:+.3f}")
        print(f"    Pearson(A_in_paddle,σ_η_IN)  = {r_inAfft:+.3f}   "
              f"(positive ⇒ pre-paddle wind energy carries into FFT bin)")
print()

# Cross-day pooled correlation
sub = fw.dropna(subset=["sigma_9373/170", "OUT/IN (FFT)"])
if len(sub) >= 4:
    r = sub[["OUT/IN (FFT)", "sigma_9373/170"]].corr().iloc[0, 1]
    r_a = sub[["IN Amplitude (FFT)", "sigma_9373/170"]].corr().iloc[0, 1]
    print(f"Pooled (both moorings) fullwind n={len(sub)}:")
    print(f"  Pearson(K_t, σ_η_IN)        = {r:+.3f}")
    print(f"  Pearson(A_in_FFT, σ_η_IN)   = {r_a:+.3f}")
print()

# ── P. Pre-paddle σ_η at OUT under nowind = sloshing proxy ────────────────
print("=" * 80)
print("P. Pre-paddle σ_η at OUT under NOWIND (= sloshing proxy)")
print("   loose230 has only 2 canon nowind cells; loose300 has 32.")
print("=" * 80)
nw = J[J["WindCondition"] == "no"]
if len(nw):
    g3 = nw.groupby("mooring_short").agg(
        n=("path", "count"),
        sOUT_mean=("sigma_12400/250", "mean"),
        sOUT_med=("sigma_12400/250", "median"),
        sOUT_std=("sigma_12400/250", "std"),
        sOUT_max=("sigma_12400/250", "max"),
        sIN_mean=("sigma_9373/170", "mean"),
        sIN_med=("sigma_9373/170", "median"),
        sIN_std=("sigma_9373/170", "std"),
    ).round(4)
    print(g3.to_string())
    print()
    print("Note: probe noise floor at 12400/250 ~0.14 mm (gold std), 0.14-0.36 mm")
    print("range across settled stillwater runs (CLAUDE.md §16).")
    print()

    # Within loose300 nowind, does K_t track σ_η at OUT (residual sloshing)?
    sub = nw[nw["mooring_short"] == "loose300"].dropna(subset=["sigma_12400/250", "OUT/IN (FFT)"])
    if len(sub) >= 4:
        r_out = sub[["OUT/IN (FFT)", "sigma_12400/250"]].corr().iloc[0, 1]
        r_a   = sub[["OUT Amplitude (FFT)", "sigma_12400/250"]].corr().iloc[0, 1]
        print(f"loose300 nowind n={len(sub)}:")
        print(f"  Pearson(K_t,         σ_η_OUT_pre) = {r_out:+.3f}")
        print(f"  Pearson(A_out_FFT,   σ_η_OUT_pre) = {r_a:+.3f}    "
              f"(positive ⇒ residual sloshing inflates OUT amplitude)")
print()

# ── Q. Decompose: how much of A_in's day-to-day wobble comes from σ_η_IN? ─
print("=" * 80)
print("Q. A_in_FFT vs σ_η_IN_pre, fullwind only — quantify wind-leak coupling")
print("=" * 80)
fw_clean = fw.dropna(subset=["sigma_9373/170", "IN Amplitude (FFT)"])
print(f"n={len(fw_clean)} fullwind runs")
print()
# Per (amp, freq), report mean σ_η_IN_pre and mean A_in_FFT, and the run-to-run
# scatter as a fraction of the mean
g4 = (fw_clean
      .groupby(["amp_v", "WaveFrequencyInput [Hz]"])
      .agg(n=("path", "count"),
           sIN_mean=("sigma_9373/170", "mean"),
           sIN_std=("sigma_9373/170", "std"),
           Ain_mean=("IN Amplitude (FFT)", "mean"),
           Ain_std=("IN Amplitude (FFT)", "std")))
g4["sIN_cv"] = g4["sIN_std"] / g4["sIN_mean"]
g4["Ain_cv"] = g4["Ain_std"] / g4["Ain_mean"]
print(g4.round(3).to_string())
print()
print("Interpretation: σ_η_IN_pre directly captures broadband wind energy at IN.")
print("Run-to-run CV of σ_η_IN_pre is the unavoidable 'wind weather' noise floor")
print("for any K_t comparison under fullwind.")

print("\nDone.")
