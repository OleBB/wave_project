"""
loose230 vs loose300 — confounder diagnostic (round 2)
========================================================

Round 1 surfaced one big pattern (h100/low alone reads ~+0.05 higher than
loose230's other configs) and one bug (out_stillwater_std all-NaN).
This script:

  E. Fixes the sloshing-proxy extraction.
  F. Compares Mar 26 loose230 (canon) vs Mar 27 loose300 (canon) head-to-head:
     same probe config, same Windspeed setpoint, only thing that changed is
     mooring + 1-day setup-drift.
  G. Inside Mar 26 loose230 alone: does K_t depend on out_stillwater_std,
     prev_run_nperiods, or prev_run_wind?
  H. Inside Mar 27 loose300 alone: same checks.
  I. Run-order sequence on Mar 26 vs Mar 27: does first-run-of-day differ
     from later-run-of-day? (sloshing build-up over a session)
"""

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

# Fixed sloshing extraction: build per-row column-name and look it up safely.
def _stillstd(row, side):
    pos_col = "out_position" if side == "out" else "in_position"
    pos = row.get(pos_col)
    if not isinstance(pos, str) or "/" not in pos:
        return np.nan
    col = f"Probe {pos} Stillwater Std"
    if col not in row.index:
        return np.nan
    val = row[col]
    try:
        return float(val) if pd.notna(val) else np.nan
    except (ValueError, TypeError):
        return np.nan

m["out_sw_std"] = m.apply(_stillstd, axis=1, side="out")
m["in_sw_std"]  = m.apply(_stillstd, axis=1, side="in")

print(f"Total rows: {len(m)}, out_sw_std non-NaN: {m['out_sw_std'].notna().sum()}, "
      f"in_sw_std non-NaN: {m['in_sw_std'].notna().sum()}")
print()

# ── E. Sloshing-proxy distributions per mooring ──────────────────────────────
print("=" * 80)
print("E. Pre-wave Stillwater Std at IN/OUT probes per mooring × wind")
print("=" * 80)
g = m.groupby(["mooring_short", "wind"])[["in_sw_std", "out_sw_std"]].agg(["count", "mean", "median", "std", "min", "max"])
print(g.round(3).to_string())
print()

# ── F. Mar 26 (loose230 canon) vs Mar 27 (loose300 canon) head-to-head ───────
print("=" * 80)
print("F. Mar 26 loose230/h100low  vs  Mar 27 loose300/h100low — head-to-head")
print("   (canon scope: same probe config, same Windspeed setpoint, mostly")
print("    1 day apart; the cleanest mooring comparison the data allows.)")
print("=" * 80)
canon = m[m["probe_config"] == "100/low"]
print(f"Canon scope rows: {len(canon)}")
print(f"  by date × mooring × wind:")
g = canon.groupby(["date", "mooring_short", "wind"]).size()
print(g.to_string())
print()

# Mar 26 loose230 = the only canon loose230 day. Mar 27 = loose300.
mar26 = canon[canon["date"] == "2026-03-26"]
mar27 = canon[canon["date"] == "2026-03-27"]
print(f"Mar 26 loose230: {len(mar26)} rows")
print(f"Mar 27 loose300: {len(mar27)} rows")
print()

# Per (amp, freq, wind), Mar26 K_t vs Mar27 K_t
def _agg(df):
    return (df.groupby(["amp_v", "WaveFrequencyInput [Hz]", "wind"])
              .agg(Kt=("OUT/IN (FFT)", "mean"),
                   Kt_std=("OUT/IN (FFT)", "std"),
                   n=("OUT/IN (FFT)", "count"),
                   sw_out=("out_sw_std", "mean"),
                   sw_in=("in_sw_std", "mean"),
                   prev_per=("prev_run_nperiods", "mean"),
                   ka_in=("IN ka (FFT)", "mean"))
              .reset_index()
              .rename(columns={"WaveFrequencyInput [Hz]": "freq"}))

a26 = _agg(mar26).rename(columns={c: f"{c}_26" for c in ["Kt", "Kt_std", "n", "sw_out", "sw_in", "prev_per", "ka_in"]})
a27 = _agg(mar27).rename(columns={c: f"{c}_27" for c in ["Kt", "Kt_std", "n", "sw_out", "sw_in", "prev_per", "ka_in"]})
joined = a26.merge(a27, on=["amp_v", "freq", "wind"], how="outer")
joined["delta_Kt"] = joined["Kt_26"] - joined["Kt_27"]
joined["delta_swout"] = joined["sw_out_26"] - joined["sw_out_27"]
joined["delta_swin"] = joined["sw_in_26"] - joined["sw_in_27"]
joined["delta_kain"] = joined["ka_in_26"] - joined["ka_in_27"]
joined = joined.sort_values(["wind", "amp_v", "freq"]).reset_index(drop=True)

cols = ["amp_v", "freq", "wind", "Kt_26", "Kt_27", "delta_Kt", "n_26", "n_27",
         "ka_in_26", "ka_in_27", "delta_kain",
         "sw_out_26", "sw_out_27", "delta_swout",
         "sw_in_26", "sw_in_27", "delta_swin",
         "prev_per_26", "prev_per_27"]
print(joined[cols].round(3).to_string(index=False))
print()

paired = joined.dropna(subset=["delta_Kt"])
print(f"Paired cells (both Mar 26 and Mar 27): {len(paired)}")
if len(paired):
    print(f"  delta_Kt   median = {paired['delta_Kt'].median():+.4f}, mean = {paired['delta_Kt'].mean():+.4f}, "
          f"range = [{paired['delta_Kt'].min():+.4f}, {paired['delta_Kt'].max():+.4f}]")
    if paired["delta_kain"].notna().sum():
        print(f"  delta_ka_in median = {paired['delta_kain'].median():+.4f}  "
              f"(positive = Mar26 IN-amplitude higher; possible incident-wave drift)")
    if paired["delta_swout"].notna().sum():
        print(f"  delta_swout median= {paired['delta_swout'].median():+.4f} mm   "
              f"(positive = Mar26 had more residual OUT-probe motion)")
    if paired["delta_swin"].notna().sum():
        print(f"  delta_swin  median= {paired['delta_swin'].median():+.4f} mm   "
              f"(positive = Mar26 had more residual IN-probe motion)")
print()

# ── G. Inside Mar 26 loose230: K_t correlations with confounders ─────────────
print("=" * 80)
print("G. Within Mar 26 loose230 (canon): K_t vs sloshing/setting confounders")
print("=" * 80)
for w in ["no", "full"]:
    sub = mar26[mar26["wind"] == w].dropna(subset=["OUT/IN (FFT)"])
    print(f"  wind={w}, n={len(sub)}")
    for col in ["out_sw_std", "in_sw_std", "prev_run_nperiods", "IN ka (FFT)"]:
        s = sub[[col, "OUT/IN (FFT)"]].dropna()
        if len(s) >= 4:
            r = s.corr().iloc[0, 1]
            print(f"    Pearson(K_t, {col}) = {r:+.3f}  n={len(s)}")
print()
print("=" * 80)
print("H. Within Mar 27 loose300 (canon): K_t vs sloshing/setting confounders")
print("=" * 80)
for w in ["no", "full"]:
    sub = mar27[mar27["wind"] == w].dropna(subset=["OUT/IN (FFT)"])
    print(f"  wind={w}, n={len(sub)}")
    for col in ["out_sw_std", "in_sw_std", "prev_run_nperiods", "IN ka (FFT)"]:
        s = sub[[col, "OUT/IN (FFT)"]].dropna()
        if len(s) >= 4:
            r = s.corr().iloc[0, 1]
            print(f"    Pearson(K_t, {col}) = {r:+.3f}  n={len(s)}")
print()

# ── I. Run-order sequence: first runs of the day vs later ────────────────────
print("=" * 80)
print("I. Run-order on Mar 26 vs Mar 27 (sloshing build-up over session)")
print("=" * 80)
def _seq(df):
    df = df.copy().sort_values("file_date").reset_index(drop=True)
    df["seq"] = df.index + 1
    return df
mar26_seq = _seq(mar26)
mar27_seq = _seq(mar27)
print(f"Mar 26 loose230 run sequence: {len(mar26_seq)} runs")
print(mar26_seq[["seq", "wind", "amp_v", "WaveFrequencyInput [Hz]",
                  "OUT/IN (FFT)", "out_sw_std", "prev_run_nperiods", "prev_run_wind"]]
        .rename(columns={"WaveFrequencyInput [Hz]": "freq"}).round(3).to_string(index=False))
print()
print(f"Mar 27 loose300 run sequence: {len(mar27_seq)} runs")
print(mar27_seq[["seq", "wind", "amp_v", "WaveFrequencyInput [Hz]",
                  "OUT/IN (FFT)", "out_sw_std", "prev_run_nperiods", "prev_run_wind"]]
        .rename(columns={"WaveFrequencyInput [Hz]": "freq"}).round(3).to_string(index=False))
print()

print("Done.")
