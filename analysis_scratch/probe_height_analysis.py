"""
Probe height & range-mode analysis
====================================
Investigates how probe height and hardware range mode affect:
  1. Stillwater noise floor (nowave + nowind)
  2. Wind-wave background amplitude (nowave + fullwind)
  3. Signal-to-noise ratio for paddle-wave detection

Four conditions under study:
  cond1  height272  highrange  (standard, all pre-20260323 folders)
  cond2  height136  highrange  (borderline — still-water already near window min)
  cond3  height100  highrange  (WRONG MODE — 100mm < 130mm window minimum)
  cond4  height100  lowrange   (correct — 100mm well inside 30-250mm window)

Run with:
  conda run -n draumkvedet python analysis_scratch/probe_height_analysis.py
"""

import sys
import os
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
os.chdir(repo_root)

import numpy as np
import pandas as pd

from wavescripts.improved_data_loader import load_analysis_data, ANALYSIS_PROBES
from wavescripts.constants import PROBE_RANGE_MODES, PROBE_HEIGHT_DEFAULT_MM

# ── 1. Load metadata ──────────────────────────────────────────────────────────

PROCESSED_DIRS = [
    Path("waveprocessed/PROCESSED-20260307-ProbPos4_31_FPV_2-tett6roof"),
    Path("waveprocessed/PROCESSED-20260312-ProbPos4_31_FPV_2-tett6roof"),
    Path("waveprocessed/PROCESSED-20260313-ProbePos4_31_FPV_2-tett6roof"),
    Path("waveprocessed/PROCESSED-20260314-ProbePos4_31_FPV_2-tett6roof"),
    Path("waveprocessed/PROCESSED-20260316-ProbePos4_31_FPV_2-tett6roof"),
    Path("waveprocessed/PROCESSED-20260316-ProbePos4_31_FPV_2-tett6roof-under9Mooring"),
    Path("waveprocessed/PROCESSED-20260319-ProbePos4_31_FPV_2-tett6roof-under9Mooring"),
    Path("waveprocessed/PROCESSED-20260321-ProbePos4_31_FPV_2-tett6roof-under9Mooring"),
    Path("waveprocessed/PROCESSED-20260323-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height136"),
    Path("waveprocessed/PROCESSED-20260323-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100"),
    Path("waveprocessed/PROCESSED-20260324-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100"),
    Path("waveprocessed/PROCESSED-20260325-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100"),
    Path("waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100"),
    Path("waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange"),
    Path("waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"),
]

print("Loading metadata …")
combined_meta, _, _, _ = load_analysis_data(*PROCESSED_DIRS, load_processed=False)
print(f"  {len(combined_meta)} total runs loaded\n")

# ── 2. Assign condition label ─────────────────────────────────────────────────

def assign_condition(row):
    h = row.get("probe_height_mm", PROBE_HEIGHT_DEFAULT_MM)
    r = row.get("probe_range_mode", "high")
    if pd.isna(h):
        h = PROBE_HEIGHT_DEFAULT_MM
    h = int(h)
    if h == 272 and r == "high":
        return "cond1_h272_high"
    elif h == 136 and r == "high":
        return "cond2_h136_high"
    elif h == 100 and r == "high":
        return "cond3_h100_high_WRONG"
    elif h == 100 and r == "low":
        return "cond4_h100_low"
    else:
        return f"other_h{h}_{r}"

combined_meta["condition"] = combined_meta.apply(assign_condition, axis=1)
print("Condition counts (all runs):")
print(combined_meta["condition"].value_counts().to_string())
print()

# ── 3. Probe window specs ─────────────────────────────────────────────────────

def window_headroom(height_mm, range_mode):
    """
    Crest headroom (mm): how far the wave surface can RISE before clipping.
    Trough headroom (mm): how far it can DROP before clipping.
    Water surface is at distance=height_mm from sensor.
    Crest arrives → surface rises → distance DECREASES toward min_mm.
    Trough arrives → surface drops → distance INCREASES toward max_mm.
    """
    spec = PROBE_RANGE_MODES.get(range_mode, {"min_mm": 0, "max_mm": 9999})
    crest = height_mm - spec["min_mm"]   # how much above still-water before hitting min
    trough = spec["max_mm"] - height_mm  # how much below still-water before hitting max
    return crest, trough

print("Probe window headroom per condition:")
print(f"  cond1 h272 high:  crest={window_headroom(272,'high')[0]:+.0f}mm  trough={window_headroom(272,'high')[1]:+.0f}mm")
print(f"  cond2 h136 high:  crest={window_headroom(136,'high')[0]:+.0f}mm  trough={window_headroom(136,'high')[1]:+.0f}mm")
print(f"  cond3 h100 high:  crest={window_headroom(100,'high')[0]:+.0f}mm  trough={window_headroom(100,'high')[1]:+.0f}mm  ← 100mm < 130mm MIN")
print(f"  cond4 h100 low:   crest={window_headroom(100,'low')[0]:+.0f}mm  trough={window_headroom(100,'low')[1]:+.0f}mm")
print()

# ── 4. Identify probe amplitude columns ──────────────────────────────────────

amp_cols = [c for c in combined_meta.columns
            if c.startswith("Probe ") and c.endswith(" Amplitude")
            and "FFT" not in c and "PSD" not in c]
print(f"Amplitude columns present: {amp_cols}\n")

# ── 5. Stillwater noise floor (nowave + nowind) ───────────────────────────────

stillwater = combined_meta[
    combined_meta["WaveFrequencyInput [Hz]"].isna() &
    (combined_meta["WindCondition"] == "no") &
    (~combined_meta["run_category"].isin(["diagnostic", "partial"]))
].copy()

print(f"Stillwater runs (nowave+nowind, non-diagnostic): {len(stillwater)}")
print()

# Per-condition, per-probe: noise floor statistics
print("=" * 70)
print("STILLWATER NOISE FLOOR PER CONDITION (mm, P97.5-P2.5 amplitude)")
print("=" * 70)

sw_rows = []
for cond, grp in stillwater.groupby("condition"):
    row = {"condition": cond, "n_runs": len(grp), "folders": grp["path"].apply(lambda p: Path(p).parent.name).nunique()}
    for col in amp_cols:
        vals = grp[col].dropna()
        if len(vals) > 0:
            row[f"{col}__mean"] = vals.mean()
            row[f"{col}__std"]  = vals.std()
            row[f"{col}__min"]  = vals.min()
            row[f"{col}__max"]  = vals.max()
            row[f"{col}__n"]    = len(vals)
    sw_rows.append(row)

sw_summary = pd.DataFrame(sw_rows).set_index("condition")

# Print per-probe summary per condition
for col in amp_cols:
    pos = col.replace("Probe ", "").replace(" Amplitude", "")
    mean_col = f"{col}__mean"
    std_col  = f"{col}__std"
    n_col    = f"{col}__n"
    if mean_col not in sw_summary.columns:
        continue
    print(f"\nProbe {pos}:")
    for cond, row2 in sw_summary.iterrows():
        n = row2.get(n_col, 0)
        if n > 0:
            print(f"  {cond:<30s}  mean={row2[mean_col]:.3f}mm  std={row2[std_col]:.3f}mm  "
                  f"n={int(n)}  runs={int(row2['n_runs'])}  folders={int(row2['folders'])}")

print()

# ── 6. Per-folder noise floor (all stillwater runs) ───────────────────────────

print("=" * 70)
print("STILLWATER NOISE FLOOR PER FOLDER")
print("=" * 70)

stillwater["folder"] = stillwater["path"].apply(lambda p: Path(p).parent.name)
folder_sw = stillwater.groupby(["condition", "folder"])[amp_cols].agg(["mean", "std", "count"])
folder_sw.columns = ["_".join(c) for c in folder_sw.columns]

for col in amp_cols:
    pos = col.replace("Probe ", "").replace(" Amplitude", "")
    mean_col = f"{col}_mean"
    n_col    = f"{col}_count"
    if mean_col not in folder_sw.columns:
        continue
    print(f"\nProbe {pos}:")
    for (cond, folder), row2 in folder_sw.iterrows():
        n = row2.get(n_col, 0)
        if n > 0:
            print(f"  [{cond}] {folder[-30:]:30s}  mean={row2[mean_col]:.3f}mm  n={int(n)}")

print()

# ── 7. Wind-wave background (nowave + fullwind) ───────────────────────────────

wind_bg = combined_meta[
    combined_meta["WaveFrequencyInput [Hz]"].isna() &
    (combined_meta["WindCondition"] == "full") &
    (~combined_meta["run_category"].isin(["diagnostic", "partial"]))
].copy()

print(f"Wind-background runs (nowave+fullwind, non-diagnostic): {len(wind_bg)}")
print()

print("=" * 70)
print("WIND BACKGROUND AMPLITUDE PER CONDITION (mm)")
print("=" * 70)

wb_rows = []
for cond, grp in wind_bg.groupby("condition"):
    row = {"condition": cond, "n_runs": len(grp), "folders": grp["path"].apply(lambda p: Path(p).parent.name).nunique()}
    for col in amp_cols:
        vals = grp[col].dropna()
        if len(vals) > 0:
            row[f"{col}__mean"] = vals.mean()
            row[f"{col}__std"]  = vals.std()
            row[f"{col}__min"]  = vals.min()
            row[f"{col}__max"]  = vals.max()
            row[f"{col}__n"]    = len(vals)
    wb_rows.append(row)

wb_summary = pd.DataFrame(wb_rows).set_index("condition")

for col in amp_cols:
    pos = col.replace("Probe ", "").replace(" Amplitude", "")
    mean_col = f"{col}__mean"
    std_col  = f"{col}__std"
    n_col    = f"{col}__n"
    if mean_col not in wb_summary.columns:
        continue
    print(f"\nProbe {pos}:")
    for cond, row2 in wb_summary.iterrows():
        n = row2.get(n_col, 0)
        if n > 0:
            print(f"  {cond:<30s}  mean={row2[mean_col]:.3f}mm  std={row2[std_col]:.3f}mm  n={int(n)}")

print()

# ── 8. Per-folder wind background ────────────────────────────────────────────

print("=" * 70)
print("WIND BACKGROUND PER FOLDER (detail)")
print("=" * 70)

wind_bg["folder"] = wind_bg["path"].apply(lambda p: Path(p).parent.name)
wind_bg["mooring"] = wind_bg["Mooring"]

for col in amp_cols:
    pos = col.replace("Probe ", "").replace(" Amplitude", "")
    print(f"\nProbe {pos}:")
    for (cond, folder), grp2 in wind_bg.groupby(["condition", "folder"]):
        vals = grp2[col].dropna()
        if len(vals) > 0:
            moor = grp2["mooring"].iloc[0] if "mooring" in grp2.columns else "?"
            print(f"  [{cond}] {folder[-35:]:35s}  mooring={moor:<20s}  "
                  f"mean={vals.mean():.3f}mm  std={vals.std():.3f}mm  n={len(vals)}")

print()

# ── 9. SNR: wind background vs stillwater noise floor ─────────────────────────

print("=" * 70)
print("WIND/STILLWATER RATIO PER CONDITION (how much louder is full wind?)")
print("=" * 70)

for col in amp_cols:
    pos = col.replace("Probe ", "").replace(" Amplitude", "")
    print(f"\nProbe {pos}:")
    sw_means = sw_summary[f"{col}__mean"].dropna() if f"{col}__mean" in sw_summary.columns else pd.Series(dtype=float)
    wb_means = wb_summary[f"{col}__mean"].dropna() if f"{col}__mean" in wb_summary.columns else pd.Series(dtype=float)
    for cond in sorted(set(sw_means.index) | set(wb_means.index)):
        sw_val = sw_means.get(cond, np.nan)
        wb_val = wb_means.get(cond, np.nan)
        if not np.isnan(sw_val) and not np.isnan(wb_val) and sw_val > 0:
            ratio = wb_val / sw_val
            print(f"  {cond:<30s}  SW={sw_val:.3f}mm  Wind={wb_val:.3f}mm  Wind/SW={ratio:.2f}x")
        elif not np.isnan(sw_val):
            print(f"  {cond:<30s}  SW={sw_val:.3f}mm  Wind=N/A")

print()

# ── 10. Condition 3 anomaly check ─────────────────────────────────────────────

print("=" * 70)
print("CONDITION 3 ANOMALY CHECK (height100 highrange = WRONG MODE)")
print("=" * 70)

cond3 = combined_meta[combined_meta["condition"] == "cond3_h100_high_WRONG"].copy()
cond3["folder"] = cond3["path"].apply(lambda p: Path(p).parent.name)
cond3["run_type"] = cond3.apply(
    lambda r: "wave" if pd.notna(r.get("WaveFrequencyInput [Hz]")) else
              f"nowave+{r.get('WindCondition','?')}wind",
    axis=1
)

print(f"\nTotal runs in cond3: {len(cond3)}")
print(cond3.groupby(["folder", "run_type"]).size().to_string())
print()

# Compare cond3 amplitude vs cond4 for same probe
print("Stillwater noise floor: cond3 vs cond4 (same probe height, different mode)")
sw3 = stillwater[stillwater["condition"] == "cond3_h100_high_WRONG"]
sw4 = stillwater[stillwater["condition"] == "cond4_h100_low"]

for col in amp_cols:
    pos = col.replace("Probe ", "").replace(" Amplitude", "")
    v3 = sw3[col].dropna()
    v4 = sw4[col].dropna()
    if len(v3) > 0 and len(v4) > 0:
        print(f"  Probe {pos}:  cond3(wrong)={v3.mean():.3f}±{v3.std():.3f}mm (n={len(v3)})  "
              f"cond4(correct)={v4.mean():.3f}±{v4.std():.3f}mm (n={len(v4)})  "
              f"ratio={v3.mean()/v4.mean():.2f}x")

print()

# ── 11. Mooring effect within same probe height ───────────────────────────────

print("=" * 70)
print("MOORING EFFECT ON WIND BACKGROUND (cond1 only — height272)")
print("=" * 70)

cond1_wind = wind_bg[wind_bg["condition"] == "cond1_h272_high"].copy()
if len(cond1_wind) > 0:
    for col in amp_cols:
        pos = col.replace("Probe ", "").replace(" Amplitude", "")
        print(f"\nProbe {pos}:")
        for moor, grp2 in cond1_wind.groupby("Mooring"):
            vals = grp2[col].dropna()
            if len(vals) > 0:
                print(f"  Mooring={moor:<25s}  mean={vals.mean():.3f}mm  std={vals.std():.3f}mm  n={len(vals)}")

# ── 12. Summary table for markdown ───────────────────────────────────────────

print()
print("=" * 70)
print("SUMMARY: NOISE FLOOR AND WIND BACKGROUND BY CONDITION")
print("(averaged across all probes listed, for key probes only)")
print("=" * 70)

key_probes = ["9373/170", "12400/250"]  # IN and OUT probes
key_amp_cols = [f"Probe {p} Amplitude" for p in key_probes]
key_amp_cols = [c for c in key_amp_cols if c in combined_meta.columns]

conditions = [
    ("cond1_h272_high",      "height272, high-range (standard)"),
    ("cond2_h136_high",      "height136, high-range (borderline)"),
    ("cond3_h100_high_WRONG","height100, high-range (wrong mode)"),
    ("cond4_h100_low",       "height100, low-range (correct)"),
]

header = f"{'Condition':<35s}  {'Probe':<12s}  {'SW noise (mm)':<14s}  {'Wind bg (mm)':<13s}  {'Wind/SW':<7s}  n_sw  n_wind"
print(header)
print("-" * len(header))
for cond_key, cond_label in conditions:
    for col in key_amp_cols:
        pos = col.replace("Probe ", "").replace(" Amplitude", "")
        sw_vals = stillwater[stillwater["condition"] == cond_key][col].dropna()
        wb_vals = wind_bg[wind_bg["condition"] == cond_key][col].dropna()
        sw_str  = f"{sw_vals.mean():.3f}±{sw_vals.std():.3f}" if len(sw_vals) > 0 else "N/A"
        wb_str  = f"{wb_vals.mean():.3f}±{wb_vals.std():.3f}" if len(wb_vals) > 0 else "N/A"
        if len(sw_vals) > 0 and len(wb_vals) > 0 and sw_vals.mean() > 0:
            ratio_str = f"{wb_vals.mean()/sw_vals.mean():.2f}x"
        else:
            ratio_str = "N/A"
        print(f"{cond_label:<35s}  {pos:<12s}  {sw_str:<14s}  {wb_str:<13s}  {ratio_str:<7s}  {len(sw_vals):<5}  {len(wb_vals)}")
    print()

print()
print("Done.")
