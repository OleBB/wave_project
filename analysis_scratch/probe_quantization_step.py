"""
Quantization step analysis — h272/high vs h100/low
====================================================

For each stillwater run in the two configs, compute on the eta_{probe} or
eta_{probe}_interp time series:

  HOW BIG:
    quant_step_5pct_mm   = 5th  percentile of |Δη| over nonzero diffs
    quant_step_50pct_mm  = 50th percentile (median) of |Δη| over nonzero diffs

  HOW OFTEN (at 250 Hz sampling):
    update_rate_hz       = nonzero-diff samples / total samples × 250 Hz
                          = "how many quantization steps per second"
    samples_per_update   = 250 / update_rate_hz
                          = "samples between consecutive updates" (idle samples)
    frac_zero_diffs      = fraction of consecutive samples that are identical

Only stillwater rows. Average per (config, probe) across stillwater runs.
Reported as one row per (config, probe).
"""

from __future__ import annotations

import os
import sys
import warnings
from datetime import datetime as _dt
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
os.chdir(BASE)

from wavescripts.improved_data_loader import load_analysis_data, load_processed_dfs

import glob

FS = 250.0

PROBE_ORDER = [
    ("8804/250",  "8804/250 (oppstrøms)"),
    ("9373/170",  "9373/170 (IN, vegg)"),
    ("9373/340",  "9373/340 (IN, fjern)"),
    ("12400/250", "12400/250 (OUT)"),
]
ANALYSIS_PROBES = [p for p, _ in PROBE_ORDER]

dirs = sorted(glob.glob("waveprocessed/PROCESSED-*"))
print(f"Loading meta from {len(dirs)} folders …")
meta, _, _, _ = load_analysis_data(*dirs, load_processed=False)

sw = meta[(meta["WindCondition"] == "no") & meta["WaveFrequencyInput [Hz]"].isna()].copy()
sw["top_folder"] = sw["path"].apply(lambda p: Path(p).parts[-2])
sw["top_folder_full"] = sw["top_folder"].apply(lambda x: f"PROCESSED-{x}")
sw["range"] = sw["probe_range_mode"]
sw["height"] = sw["probe_height_mm"].astype("Int64")

# Restrict to h272/high and h100/low only
sw = sw[((sw["height"] == 272) & (sw["range"] == "high"))
         | ((sw["height"] == 100) & (sw["range"] == "low"))]
print(f"Stillwater rows in scope (h272/high + h100/low): {len(sw)}")
print()

# Folders to load processed_dfs for
needed_folders = sw["top_folder_full"].dropna().unique()
print(f"Loading processed_dfs for {len(needed_folders)} folders containing stillwater …")
processed_dfs = {}
for folder in sorted(needed_folders):
    p = BASE / "waveprocessed" / folder
    if p.exists():
        processed_dfs.update(load_processed_dfs(str(p)))
print(f"  {len(processed_dfs)} time-series cached")
print()

# ── Per-run, per-probe quantization metrics ────────────────────────────────
def _eta_signal(df, probe):
    col = f"eta_{probe}_interp" if f"eta_{probe}_interp" in df.columns else f"eta_{probe}"
    if col not in df.columns:
        return None
    return df[col].to_numpy(dtype=float)

records = []
for _, row in sw.iterrows():
    path = row["path"]
    df = processed_dfs.get(path)
    if df is None:
        continue
    config = f"h{int(row['height'])}/{row['range']}"
    for probe in ANALYSIS_PROBES:
        sig = _eta_signal(df, probe)
        if sig is None:
            continue
        sig = sig[np.isfinite(sig)]
        if len(sig) < 100:
            continue
        diffs = np.abs(np.diff(sig))
        n_total = len(diffs)
        nz = diffs[diffs > 0]
        n_nz = len(nz)
        if n_nz == 0:
            continue
        records.append({
            "config":            config,
            "probe":             probe,
            "n_samples":         len(sig),
            "n_diffs_nonzero":   n_nz,
            "n_diffs_zero":      n_total - n_nz,
            "frac_zero":         (n_total - n_nz) / n_total,
            "step_5pct_mm":      float(np.percentile(nz, 5)),
            "step_50pct_mm":     float(np.percentile(nz, 50)),
            "step_min_mm":       float(np.min(nz)),
            "update_rate_hz":    n_nz / n_total * FS,
            "samples_per_update": n_total / n_nz,
        })

per_run = pd.DataFrame(records)
print(f"Computed metrics for {len(per_run)} (run × probe) entries")
print()

# ── Aggregate per (config, probe) ──────────────────────────────────────────
agg = (per_run
       .groupby(["config", "probe"])
       .agg(n_runs=("n_samples", "count"),
            step_5pct_mm=("step_5pct_mm", "mean"),
            step_5pct_std=("step_5pct_mm", "std"),
            step_50pct_mm=("step_50pct_mm", "mean"),
            step_min_mm=("step_min_mm", "min"),
            update_rate_hz=("update_rate_hz", "mean"),
            samples_per_update=("samples_per_update", "mean"),
            frac_zero=("frac_zero", "mean"))
       .reset_index())

# Sort to put config rows in the desired order
agg["_co"] = agg["config"].map({"h272/high": 0, "h100/low": 1})
agg["_pr"] = agg["probe"].map({p: i for i, (p, _) in enumerate(PROBE_ORDER)})
agg = agg.sort_values(["_co", "_pr"]).drop(columns=["_co", "_pr"]).reset_index(drop=True)

print("=" * 90)
print("Quantization step + update-rate per (config, probe), averaged across stillwater runs")
print("=" * 90)
cols = ["config", "probe", "n_runs",
         "step_5pct_mm", "step_50pct_mm", "step_min_mm",
         "update_rate_hz", "samples_per_update", "frac_zero"]
print(agg[cols].round(4).to_string(index=False))
print()

# Also dump a side-by-side compact table h272 vs h100 per probe
print("=" * 90)
print("Side-by-side h272/high vs h100/low — focus on OUT probe and IN probe")
print("=" * 90)
for pos, label in PROBE_ORDER:
    sub = agg[agg["probe"] == pos].set_index("config")
    if "h272/high" in sub.index and "h100/low" in sub.index:
        h = sub.loc["h272/high"]; l = sub.loc["h100/low"]
        print(f"\n{label}:")
        print(f"  step (5th pct):  h272/high = {h['step_5pct_mm']:.4f} mm   "
              f"h100/low = {l['step_5pct_mm']:.4f} mm   "
              f"ratio = {h['step_5pct_mm']/l['step_5pct_mm']:.2f}×")
        print(f"  step (median):   h272/high = {h['step_50pct_mm']:.4f} mm   "
              f"h100/low = {l['step_50pct_mm']:.4f} mm")
        print(f"  update rate:     h272/high = {h['update_rate_hz']:.1f} Hz    "
              f"h100/low = {l['update_rate_hz']:.1f} Hz   "
              f"({h['samples_per_update']:.1f} vs {l['samples_per_update']:.1f} samples/update)")
        print(f"  frac samples idle: h272/high = {h['frac_zero']:.2%}   "
              f"h100/low = {l['frac_zero']:.2%}")

# Save CSVs
out_csv = BASE / "analysis_scratch" / "probe_quantization_step.csv"
agg.to_csv(out_csv, index=False)
print(f"\nCSV → {out_csv.relative_to(BASE)}")

per_run_csv = BASE / "analysis_scratch" / "probe_quantization_step_per_run.csv"
per_run.to_csv(per_run_csv, index=False)
print(f"per-run CSV → {per_run_csv.relative_to(BASE)}")

print("\nDone.")
