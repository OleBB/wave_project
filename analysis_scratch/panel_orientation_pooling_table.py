"""Data sidecar for the appendix table `app_panel_pooling` — quantitative
justification for pooling above_50 fullpanel + reversepanel in CH05
figures.

Writes:
  output/TABLES/data/app_panel_pooling.csv
  output/TABLES/data/app_panel_pooling.meta.json

Render cell lives in main_save_tables.py.

Source analysis: analysis_scratch/_above50_panel_diff.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = (Path(__file__).resolve().parent.parent
        if "__file__" in globals() else Path.cwd())
sys.path.insert(0, str(ROOT))

from wavescripts.improved_data_loader import load_analysis_data
from wavescripts.filters import apply_experimental_filters

# ── Load and filter ───────────────────────────────────────────────────────────
import glob
all_dirs = sorted(glob.glob(str(ROOT / "waveprocessed" / "PROCESSED-*")))
meta, _, _, _ = load_analysis_data(*all_dirs, load_processed=False)

m = apply_experimental_filters(meta, {
    "filters": {
        "Mooring":         "above_50",
        "PanelCondition":  ["full", "reverse"],
        "WindCondition":   ["no", "full"],
    },
}).copy()
m = m[m["WaveFrequencyInput [Hz]"].notna()
      & (m["WaveFrequencyInput [Hz]"] > 0)
      & m["OUT/IN (FFT)"].notna()]

# ── Per-cell aggregation ──────────────────────────────────────────────────────
cell_keys = ["WaveAmplitudeInput [Volt]", "WaveFrequencyInput [Hz]", "WindCondition"]
agg = (m.groupby(cell_keys + ["PanelCondition"])
        .agg(K=("OUT/IN (FFT)", "mean"),
             K_std=("OUT/IN (FFT)", "std"),
             n=("path", "count"))
        .reset_index())
piv = agg.pivot_table(index=cell_keys, columns="PanelCondition",
                      values=["K", "K_std", "n"]).reset_index()
piv.columns = ["_".join([str(c) for c in col if c]).strip("_") for col in piv.columns]

# Keep only cells with BOTH panels present
piv = piv.dropna(subset=["K_full", "K_reverse"]).copy()
piv["delta_rev_minus_full"] = piv["K_reverse"] - piv["K_full"]
piv["delta_pct_of_full"]    = 100.0 * piv["delta_rev_minus_full"] / piv["K_full"]
piv = piv.sort_values(cell_keys).reset_index(drop=True)

# ── Render-shape CSV ──────────────────────────────────────────────────────────
out = pd.DataFrame({
    "amp_v":     piv["WaveAmplitudeInput [Volt]"].astype(float),
    "freq_hz":   piv["WaveFrequencyInput [Hz]"].astype(float),
    "wind":      piv["WindCondition"].astype(str),
    "n_full":    piv["n_full"].astype(int),
    "K_full":    piv["K_full"].astype(float),
    "n_rev":     piv["n_reverse"].astype(int),
    "K_rev":     piv["K_reverse"].astype(float),
    "delta":     piv["delta_rev_minus_full"].astype(float),
})

CSV  = ROOT / "output" / "TABLES" / "data" / "app_panel_pooling.csv"
META = ROOT / "output" / "TABLES" / "data" / "app_panel_pooling.meta.json"
CSV.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(CSV, index=False)
print(f"  CSV  → {CSV.relative_to(ROOT)}  ({len(out)} rows)")

# ── Headline summary (for caption + footer; written to meta.json) ────────────
d = piv["delta_rev_minus_full"]
summary = {
    "n_cells":             int(len(piv)),
    "median_delta":        float(d.median()),
    "mean_delta":          float(d.mean()),
    "std_delta":           float(d.std()),
    "abs_delta_max":       float(d.abs().max()),
    "frac_abs_le_0p05":    float((d.abs() <= 0.05).mean()),
    "frac_abs_le_0p10":    float((d.abs() <= 0.10).mean()),
}
print("\nHeadline:")
for k, v in summary.items():
    print(f"  {k:>20s} = {v}")

# ── meta.json ─────────────────────────────────────────────────────────────────
meta_dict = {
    "name":           "app_panel_pooling",
    "script":         "analysis_scratch/panel_orientation_pooling_table.py",
    "plot_type":      "table",
    "chapter":        "appendix",
    "caption_label":  "tab:app_panel_pooling",
    "caption_short":  "",   # owned by main_save_tables.py
    "n_rows":         int(len(out)),
    "summary":        summary,
    "filter_scope":   {
        "Mooring":        "above_50",
        "PanelCondition": ["full", "reverse"],
        "WindCondition":  ["no", "full"],
        "freq_range":     "all available (1.3 Hz only — reverse not tested elsewhere)",
        "quality_flag":   "ok (default in apply_experimental_filters)",
    },
    "delta_convention": "Δ = K_rev − K_full  (negative ⇒ reverse panel transmits less)",
    "generated_at":     datetime.now().isoformat(timespec="seconds"),
}
META.write_text(json.dumps(meta_dict, indent=2, ensure_ascii=False),
                encoding="utf-8")
print(f"  META → {META.relative_to(ROOT)}")
