"""
loose230 vs loose300 — confounder diagnostic
=============================================

After session_2026-05-06b: canon scope says loose230 transmits +0.036 more
than loose300; broad scope says ≈0. The user flagged three known confounders
to dig into before drawing any conclusion:

  1. Probe height (probe_height_mm) and range setting (probe_range_mode)
  2. Day-to-day wind-speed variability (Windspeed, file_date)
  3. Between-run sloshing (prev_run_*, Stillwater Std)

This script writes 4 plain-text tables to repl/loose230_vs_300_confounders.txt:

  A. Per-folder roster: dates, probe_height/range, Windspeed mean,
     n loose230 vs loose300 in scope.
  B. Probe-config bias inside loose230 alone: K_t per (amp, freq, wind)
     × probe_config — does h100/low alone agree with the canon-scope number
     across all configs?
  C. Day-to-day Windspeed inside fullwind cells, separated by mooring.
     If loose300's Mar 27 had a different wind speed than loose230's
     Mar 16-26 average, the +0.036 effect can't cleanly be assigned to
     mooring slack.
  D. Between-run sloshing proxy: pre-wave Stillwater Std at 12400/250
     (the OUT probe — quiet, sheltered) per run. Compare loose230 vs
     loose300 distributions, then check if K_t correlates with stillwater
     residual motion within each mooring.

No causal claims here — observations only, hypotheses flagged.
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

OUT = BASE / "repl" / "loose230_vs_300_confounders.txt"
OUT.parent.mkdir(parents=True, exist_ok=True)

THESIS_FREQS = [1.3, 1.4, 1.5, 1.6]
THESIS_AMPS = [0.10, 0.20, 0.30]

dirs = sorted(glob.glob("waveprocessed/PROCESSED-*"))
print(f"Loading {len(dirs)} folders …")
m, _, _, _ = load_analysis_data(*dirs, load_processed=False)

m = m[m["PanelCondition"] == "full"]
m = m[m["Mooring"].isin(["below_90_loose230", "below_90_loose300"])]
m = m[m["WindCondition"].isin(["no", "full"])]
m = m[m["WaveFrequencyInput [Hz]"].isin(THESIS_FREQS)]
m = m[m["WaveAmplitudeInput [Volt]"].apply(
    lambda v: any(abs(float(v) - a) < 1e-3 for a in THESIS_AMPS))]
m = m.dropna(subset=["OUT/IN (FFT)"])

m["amp_v"] = m["WaveAmplitudeInput [Volt]"].apply(lambda v: round(float(v), 2))
m["freq"] = m["WaveFrequencyInput [Hz]"]
m["mooring_short"] = m["Mooring"].str.replace("below_90_", "")
m["wind"] = m["WindCondition"]

# probe_config tag = "h{height}/{mode}"
m["probe_config"] = m["probe_height_mm"].astype("Int64").astype(str) + "/" + m["probe_range_mode"].astype(str)
m["probe_config"] = m["probe_config"].str.replace("<NA>", "?")

# file_date as string for grouping
m["date"] = pd.to_datetime(m["file_date"]).dt.strftime("%Y-%m-%d")

print(f"Rows in scope: {len(m)}")
print()

# ── A. Per-folder roster ─────────────────────────────────────────────────────
print("=" * 80)
print("A. Per-folder roster (date × mooring × probe-config × Windspeed)")
print("=" * 80)
grp_a = (m.groupby(["date", "mooring_short", "probe_config"])
          .agg(n_runs=("OUT/IN (FFT)", "count"),
               wind_mean=("Windspeed", "mean"),
               wind_std=("Windspeed", "std"),
               wind_min=("Windspeed", "min"),
               wind_max=("Windspeed", "max"))
          .reset_index())
print(grp_a.round(3).to_string(index=False))
print()

# ── B. Probe-config bias inside loose230 alone ──────────────────────────────
print("=" * 80)
print("B. Probe-config bias INSIDE loose230 (broad scope)")
print("   Does h100/low (canon) alone differ from the other configs?")
print("=" * 80)
m230 = m[m["mooring_short"] == "loose230"]
print(f"loose230 total rows: {len(m230)}")
print(f"loose230 probe configs: {sorted(m230['probe_config'].unique())}")
print()
b = (m230.groupby(["amp_v", "freq", "wind", "probe_config"])
       ["OUT/IN (FFT)"]
       .agg(["mean", "std", "count"])
       .reset_index()
       .rename(columns={"mean": "Kt", "std": "Kt_std", "count": "n"}))
piv_b = b.pivot_table(index=["amp_v", "freq", "wind"],
                       columns="probe_config", values="Kt").reset_index()
piv_bn = b.pivot_table(index=["amp_v", "freq", "wind"],
                        columns="probe_config", values="n").reset_index()
piv_bn.columns = [f"n_{c}" if c not in ("amp_v", "freq", "wind") else c for c in piv_bn.columns]
joined = piv_b.merge(piv_bn, on=["amp_v", "freq", "wind"])
print(joined.round(3).to_string(index=False))
print()

# Does h100/low (canon) systematically differ from the mean of the other configs?
config_cols = [c for c in piv_b.columns if c not in ("amp_v", "freq", "wind")]
print(f"Probe-config columns: {config_cols}")
if "100/low" in config_cols and len(config_cols) > 1:
    others = [c for c in config_cols if c != "100/low"]
    diffs = []
    for _, r in piv_b.iterrows():
        canon = r.get("100/low", np.nan)
        if pd.isna(canon):
            continue
        for o in others:
            v = r.get(o, np.nan)
            if pd.notna(v):
                diffs.append({"amp_v": r["amp_v"], "freq": r["freq"],
                               "wind": r["wind"], "other_config": o,
                               "delta": canon - v})
    if diffs:
        d_df = pd.DataFrame(diffs)
        print()
        print("Per-cell K_t(100/low canon) − K_t(other) inside loose230:")
        print(d_df.round(3).to_string(index=False))
        print(f"\n  Across {len(d_df)} pairings:")
        print(f"    median delta = {d_df['delta'].median():+.4f}")
        print(f"    mean delta   = {d_df['delta'].mean():+.4f}")
        print(f"    range        = [{d_df['delta'].min():+.4f}, {d_df['delta'].max():+.4f}]")
print()

# ── C. Wind-speed by date × mooring (fullwind only) ──────────────────────────
print("=" * 80)
print("C. Windspeed by date × mooring (fullwind cells only)")
print("=" * 80)
mfw = m[m["wind"] == "full"]
ws = (mfw.groupby(["date", "mooring_short"])
        .agg(n=("Windspeed", "count"),
             ws_mean=("Windspeed", "mean"),
             ws_std=("Windspeed", "std"),
             ws_min=("Windspeed", "min"),
             ws_max=("Windspeed", "max"))
        .reset_index())
print(ws.round(3).to_string(index=False))
print()
mby = mfw.groupby("mooring_short")["Windspeed"].agg(["count", "mean", "std", "min", "max"])
print("Pooled fullwind Windspeed per mooring:")
print(mby.round(3).to_string())
print()

# Is there a per-day Windspeed × K_t correlation inside loose230 fullwind?
m230fw = mfw[mfw["mooring_short"] == "loose230"]
if len(m230fw) > 5 and m230fw["Windspeed"].notna().sum() > 5:
    corr = m230fw[["Windspeed", "OUT/IN (FFT)"]].corr().iloc[0, 1]
    print(f"loose230 fullwind: Pearson(Windspeed, K_t) = {corr:+.3f}  (n={len(m230fw)})")
m300fw = mfw[mfw["mooring_short"] == "loose300"]
if len(m300fw) > 5 and m300fw["Windspeed"].notna().sum() > 5:
    corr = m300fw[["Windspeed", "OUT/IN (FFT)"]].corr().iloc[0, 1]
    print(f"loose300 fullwind: Pearson(Windspeed, K_t) = {corr:+.3f}  (n={len(m300fw)})")
print()

# ── D. Between-run sloshing proxy ────────────────────────────────────────────
print("=" * 80)
print("D. Between-run sloshing proxy")
print("=" * 80)
print("   Pre-wave Stillwater Std at OUT probe = residual motion before paddle.")
print("   Higher values = water hadn't settled (or wind-pumped) before this run.")
print()

# OUT probe varies by config. For each row, fetch its `out_position` and the
# matching `Probe {pos} Stillwater Std` value.
def _outstd(row):
    pos = row.get("out_position")
    if not isinstance(pos, str):
        return np.nan
    col = f"Probe {pos} Stillwater Std"
    return row.get(col, np.nan)

m["out_stillwater_std"] = m.apply(_outstd, axis=1)
m["in_stillwater_std"] = m.apply(
    lambda r: r.get(f"Probe {r.get('in_position', '')} Stillwater Std", np.nan)
    if isinstance(r.get("in_position"), str) else np.nan, axis=1)

# Distribution per mooring
print("Pre-wave Stillwater Std at OUT probe (mm) per mooring:")
g = m.groupby("mooring_short")["out_stillwater_std"].agg(["count", "mean", "std", "min", "max", "median"])
print(g.round(3).to_string())
print()
print("Pre-wave Stillwater Std at IN probe (mm) per mooring:")
g = m.groupby("mooring_short")["in_stillwater_std"].agg(["count", "mean", "std", "min", "max", "median"])
print(g.round(3).to_string())
print()

# Split by wind condition
print("Same, split by wind:")
g = m.groupby(["mooring_short", "wind"])["out_stillwater_std"].agg(["count", "mean", "median", "std"])
print(g.round(3).to_string())
print()

# Correlation: does K_t depend on out_stillwater_std inside each mooring?
print("Correlation K_t × out_stillwater_std (within mooring × wind):")
for moor in ["loose230", "loose300"]:
    for w in ["no", "full"]:
        sub = m[(m["mooring_short"] == moor) & (m["wind"] == w)]
        sub = sub.dropna(subset=["OUT/IN (FFT)", "out_stillwater_std"])
        if len(sub) >= 4:
            r = sub[["OUT/IN (FFT)", "out_stillwater_std"]].corr().iloc[0, 1]
            print(f"  {moor:9s} {w:4s} n={len(sub):3d}  r = {r:+.3f}  "
                  f"sw_std mean={sub['out_stillwater_std'].mean():.3f} mm")
print()

# Previous-run effect: prev_run_freq_hz, prev_run_wind, prev_run_nperiods
print("Previous-run context per mooring (mooring × wind cell):")
print()
prev_cols = ["prev_run_freq_hz", "prev_run_nperiods", "prev_run_wind"]
have_prev = [c for c in prev_cols if c in m.columns]
print(f"prev-run columns available: {have_prev}")
print()
if have_prev:
    g = m.groupby("mooring_short")[have_prev[0]].agg(["count", "mean", "min", "max"])
    print(f"Prev-run frequency (Hz) per mooring:")
    print(g.round(2).to_string())
    print()
    if "prev_run_wind" in have_prev:
        g = m.groupby(["mooring_short", "prev_run_wind"]).size().unstack(fill_value=0)
        print("Prev-run wind condition counts per mooring:")
        print(g.to_string())
        print()
    if "prev_run_nperiods" in have_prev:
        g = m.groupby("mooring_short")["prev_run_nperiods"].agg(["count", "mean", "median", "min", "max"])
        print(f"Prev-run n_periods (settling time proxy — higher means more wait):")
        print(g.round(1).to_string())
        print()

# ── Save dump to repl/ ───────────────────────────────────────────────────────
print("=" * 80)
print(f"DONE. Tables shown above.")
print("=" * 80)
