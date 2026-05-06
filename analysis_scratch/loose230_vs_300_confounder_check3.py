"""
loose230 vs loose300 — confounder diagnostic round 3
=====================================================

Round 2 found:
  - h100/low alone reads +0.05 higher K_t than other loose230 configs
  - Mar 26 (loose230 canon) had prev_run_nperiods=240 mostly; Mar 27
    (loose300 canon) had 40-140
  - Stillwater Std columns came back all NaN (extraction bug or sparse?)

Round 3:
  J. Verify Stillwater Std availability per dataset.
  K. Decompose +0.036 K_t delta: is it A_in changing or A_out changing?
  L. Settling-time pattern across loose230 broad-scope days (does the
     "lazy/no-settling" description match Mar 19/23/24 reality?).
  M. Mooring × prev_run_wind contingency: does the previous run's wind
     condition correlate with current K_t?
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
m_full, _, _, _ = load_analysis_data(*dirs, load_processed=False)

# Filter as before
m = m_full.copy()
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

# ── J. Stillwater Std column availability ───────────────────────────────────
print("=" * 80)
print("J. Stillwater Std column availability — debug")
print("=" * 80)
sw_cols = [c for c in m.columns if "Stillwater Std" in c]
print(f"Stillwater Std columns found: {sw_cols}")
for c in sw_cols:
    n_nonnull = m[c].notna().sum()
    print(f"  {c}: {n_nonnull}/{len(m)} non-null")
print()

# Reference probe per row, then check that probe's column
def _check_pos(row, side):
    pos = row.get("out_position" if side == "out" else "in_position")
    if not isinstance(pos, str) or "/" not in pos:
        return ("NO_POS", np.nan)
    col = f"Probe {pos} Stillwater Std"
    if col not in row.index:
        return (f"COL_MISSING:{col}", np.nan)
    val = row[col]
    if pd.isna(val):
        return (f"NaN", np.nan)
    return ("OK", float(val))

print("Sample of (out_position, Probe {pos} Stillwater Std) per mooring:")
for moor in ["loose230", "loose300"]:
    print(f"\n{moor}:")
    sub = m[m["mooring_short"] == moor].head(3)
    for _, r in sub.iterrows():
        pos = r.get("out_position", "?")
        col = f"Probe {pos} Stillwater Std"
        val = r.get(col, "MISSING")
        print(f"  out_pos={pos}, col_exists={col in r.index}, value={val}")
print()

# Try the alternative — the per-row out probe might be set differently. Check
# what out_position values look like.
print("out_position values seen in scope:")
print(m["out_position"].value_counts(dropna=False).to_string())
print()
print("in_position values seen in scope:")
print(m["in_position"].value_counts(dropna=False).to_string())
print()

# Does any specific Stillwater Std column have data?
print("Per-column non-null counts inside the scope:")
for c in sw_cols:
    print(f"  {c}: non-null = {m[c].notna().sum()}")
print()
# Now check across the FULL meta (no filter)
print("Per-column non-null counts in the FULL meta (no filter):")
for c in sw_cols:
    print(f"  {c}: non-null = {m_full[c].notna().sum()} of {len(m_full)}")
print()

# ── K. Decompose K_t delta: A_in vs A_out ────────────────────────────────────
print("=" * 80)
print("K. Decompose K_t delta into A_in vs A_out contributions")
print("=" * 80)
canon = m[m["probe_config"] == "100/low"]
mar26 = canon[canon["date"] == "2026-03-26"]
mar27 = canon[canon["date"] == "2026-03-27"]

def _agg_amp(df):
    return (df.groupby(["amp_v", "WaveFrequencyInput [Hz]", "wind"])
              .agg(Kt=("OUT/IN (FFT)", "mean"),
                   A_in=("IN Amplitude (FFT)", "mean"),
                   A_out=("OUT Amplitude (FFT)", "mean"),
                   ka_in=("IN ka (FFT)", "mean"),
                   n=("OUT/IN (FFT)", "count"))
              .reset_index()
              .rename(columns={"WaveFrequencyInput [Hz]": "freq"}))

a26 = _agg_amp(mar26).rename(columns={c: f"{c}_26" for c in ["Kt", "A_in", "A_out", "ka_in", "n"]})
a27 = _agg_amp(mar27).rename(columns={c: f"{c}_27" for c in ["Kt", "A_in", "A_out", "ka_in", "n"]})
joined = a26.merge(a27, on=["amp_v", "freq", "wind"], how="inner")
joined["dKt"]    = joined["Kt_26"]    - joined["Kt_27"]
joined["dA_in"]  = joined["A_in_26"]  - joined["A_in_27"]
joined["dA_out"] = joined["A_out_26"] - joined["A_out_27"]
joined["pct_A_in"]  = 100 * joined["dA_in"]  / joined["A_in_27"]
joined["pct_A_out"] = 100 * joined["dA_out"] / joined["A_out_27"]
joined = joined.sort_values(["wind", "amp_v", "freq"]).reset_index(drop=True)
print(joined.round(3).to_string(index=False))
print()
print(f"Across {len(joined)} paired cells:")
print(f"  median dKt   = {joined['dKt'].median():+.4f}")
print(f"  median dA_in = {joined['dA_in'].median():+.4f} mm   ({joined['pct_A_in'].median():+.2f}%)")
print(f"  median dA_out= {joined['dA_out'].median():+.4f} mm   ({joined['pct_A_out'].median():+.2f}%)")
print(f"  ⇒ if dA_in ≈ 0 and dA_out > 0, the K_t lift is real in A_out (transmission)")
print(f"  ⇒ if both A_in and A_out shift up, IN-side energy boost is the driver")
print()

# Pearson per cell: does dKt track dA_in or dA_out better?
fw = joined[joined["wind"] == "full"].dropna(subset=["dKt", "dA_in", "dA_out"])
nw = joined[joined["wind"] == "no"].dropna(subset=["dKt", "dA_in", "dA_out"])
if len(fw) >= 4:
    r_in  = fw[["dKt", "dA_in"]].corr().iloc[0, 1]
    r_out = fw[["dKt", "dA_out"]].corr().iloc[0, 1]
    print(f"  fullwind: r(dKt, dA_in)={r_in:+.3f}, r(dKt, dA_out)={r_out:+.3f}  n={len(fw)}")
if len(nw) >= 4:
    r_in  = nw[["dKt", "dA_in"]].corr().iloc[0, 1]
    r_out = nw[["dKt", "dA_out"]].corr().iloc[0, 1]
    print(f"  nowind:   r(dKt, dA_in)={r_in:+.3f}, r(dKt, dA_out)={r_out:+.3f}  n={len(nw)}")
print()

# ── L. Settling time across loose230 broad-scope days ───────────────────────
print("=" * 80)
print("L. prev_run_nperiods across loose230 broad-scope days")
print("   The user said earlier days had less settling between runs.")
print("=" * 80)
g = (m.groupby(["date", "mooring_short", "probe_config"])
       ["prev_run_nperiods"]
       .agg(["count", "mean", "median", "min", "max"]))
print(g.round(1).to_string())
print()
print("nperiods value-count per date (loose230 broad scope):")
mar_loose230 = m[m["mooring_short"] == "loose230"]
g2 = mar_loose230.groupby(["date", "prev_run_nperiods"]).size().unstack(fill_value=0)
print(g2.to_string())
print()

# ── M. prev_run_wind × current K_t ───────────────────────────────────────────
print("=" * 80)
print("M. Effect of prev_run_wind on current-run K_t")
print("=" * 80)
print("Hypothesis: if previous run was fullwind, residual wind setup or")
print("sloshing could bias current OUT-probe baseline.")
print()
for moor in ["loose230", "loose300"]:
    print(f"\n{moor}:")
    sub = m[m["mooring_short"] == moor]
    g = (sub.groupby(["wind", "prev_run_wind"])
            ["OUT/IN (FFT)"]
            .agg(["count", "mean", "std", "median"]))
    print(g.round(3).to_string())
print()

print("Done.")
