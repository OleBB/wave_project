"""
How does probe noise floor relate to the +0.048 K_t probe-mode bias?
=====================================================================

The within-loose230 probe-mode bias finding (O1):
  K_t(h100/low) - K_t(h100/high) = +0.048  (median, fullwind canon-amp/freq cells)

This script:
  1. Calls plot_probe_noise_floor with the same group_by used in the thesis
     (probe_height_mm × probe_range_mode) to get noise floor numbers per
     probe per config.
  2. For the OUT probe (12400/250) — the one that drives the K_t lift —
     extracts noise_rms_mm and detection_threshold_mm in h100/high vs h100/low.
  3. Compares against the typical OUT-probe FFT amplitude under fullwind
     paddle waves (~5–17 mm depending on amp tier), to compute "noise as
     fraction of signal" per mode.
  4. Estimates the K_t bias that quantization-noise floor change ALONE
     could plausibly explain — and asks whether it accounts for +0.048
     or whether something else is going on.

No causal claim. Result is reported as "the noise-floor change is
consistent with / inconsistent with / smaller than the observed bias".
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
os.chdir(BASE)

from wavescripts.improved_data_loader import load_analysis_data, load_processed_dfs
from wavescripts.plotter import plot_probe_noise_floor

import glob

# Load all processed folders to keep the noise-floor pool large enough that
# every config has multiple stillwater rows.
dirs = sorted(glob.glob("waveprocessed/PROCESSED-*"))
print(f"Loading {len(dirs)} folders for stillwater pool …")
combined_meta, _, _, _ = load_analysis_data(*dirs, load_processed=False)
print(f"  {len(combined_meta)} total runs")

# We need processed_dfs for quantization step. Loading 25 folders is heavy,
# so just use the ones that contain stillwater rows in the relevant configs.
sw_mask = (
    (combined_meta["WindCondition"] == "no")
    & combined_meta["WaveFrequencyInput [Hz]"].isna()
)
sw_paths = combined_meta.loc[sw_mask, "PROCESSED_folder"].dropna().unique()
sw_dirs = [str(BASE / "waveprocessed" / p) for p in sw_paths]
print(f"Loading processed_dfs for {len(sw_dirs)} stillwater-bearing folders …")
processed_dfs = {}
for d in sw_dirs:
    if Path(d).exists():
        processed_dfs.update(load_processed_dfs(d))
print(f"  {len(processed_dfs)} time-series cached")

# Probes to analyse: same as ANALYSIS_PROBES in main_save_figures.py
ANALYSIS_PROBES = ["8804/250", "9373/170", "9373/340", "12400/250"]

pv = {
    "filters": {},
    "plotting": {
        "show_plot": False,
        "save_plot": False,
        "draft": True,
        "figure_name": "probe_noise_floor_vs_mode_bias",
        "force_stub": False,
    },
}

print("\nCalling plot_probe_noise_floor …")
_figs, summary = plot_probe_noise_floor(
    combined_meta, ANALYSIS_PROBES, pv,
    group_by=["probe_height_mm", "probe_range_mode"],
    processed_dfs=processed_dfs,
)
print(f"\nSummary table ({len(summary)} rows):")
print(summary.round(4).to_string())
print()

# ── Now answer the apples question ─────────────────────────────────────────
# For each config, find the OUT-probe noise floor at 12400/250.
out_pos = "12400/250"
in_pos  = "9373/170"

print("=" * 78)
print(f"Noise floor at IN/OUT probes per config (mm)")
print("=" * 78)
cols = ["group", "probe", "noise_rms_mm", "noise_95pct_amp_mm",
         "quantization_step_mm", "detection_threshold_mm", "n_runs"]
sub = summary[summary["probe"].isin([in_pos, out_pos])][cols]
print(sub.round(4).to_string(index=False))
print()

# ── K_t bias arithmetic ───────────────────────────────────────────────────
# Typical OUT amplitude (canon scope, h100/low) per amp tier from earlier work:
#   A1 (0.1 V): ~5.5 mm OUT, ~7.5 mm IN
#   A2 (0.2 V): ~12 mm OUT, ~15 mm IN
#   A3 (0.3 V): ~17 mm OUT, ~23 mm IN
typical = pd.DataFrame([
    {"amp": "A1 (0.1 V)", "A_in_mm":  7.6, "A_out_mm":  5.6},
    {"amp": "A2 (0.2 V)", "A_in_mm": 15.3, "A_out_mm": 11.7},
    {"amp": "A3 (0.3 V)", "A_in_mm": 23.0, "A_out_mm": 17.0},
])

# Pull noise floor for h100/high and h100/low at OUT and IN.
def _get(grp_label, pos, col):
    row = summary[(summary["group"] == grp_label) & (summary["probe"] == pos)]
    if not len(row):
        return np.nan
    return float(row.iloc[0][col])

# Group labels in summary are "h100 / high", "h100 / low", etc.
candidates = sorted(summary["group"].unique())
print(f"Group labels seen in summary: {candidates}")
print()

GH = next((g for g in candidates if "100" in g and "high" in g.lower()), None)
GL = next((g for g in candidates if "100" in g and "low"  in g.lower()), None)
print(f"highrange config: {GH}")
print(f"lowrange config:  {GL}")
print()

if GH and GL:
    rms_out_high = _get(GH, out_pos, "noise_rms_mm")
    rms_out_low  = _get(GL, out_pos, "noise_rms_mm")
    rms_in_high  = _get(GH, in_pos,  "noise_rms_mm")
    rms_in_low   = _get(GL, in_pos,  "noise_rms_mm")
    q_out_high   = _get(GH, out_pos, "quantization_step_mm")
    q_out_low    = _get(GL, out_pos, "quantization_step_mm")

    print(f"OUT probe ({out_pos}):  σ_noise(high) = {rms_out_high:.3f} mm,  σ_noise(low) = {rms_out_low:.3f} mm")
    print(f"OUT probe quantization: q(high) = {q_out_high:.3f} mm,         q(low) = {q_out_low:.3f} mm")
    print(f"IN probe ({in_pos}):   σ_noise(high) = {rms_in_high:.3f} mm,  σ_noise(low) = {rms_in_low:.3f} mm")
    print()

    print("=" * 78)
    print("Per-amp tier: noise as fraction of signal, and K_t-bias if noise added")
    print("=" * 78)
    print()
    rows = []
    for _, t in typical.iterrows():
        Ain  = t["A_in_mm"]
        Aout = t["A_out_mm"]
        Kt_clean = Aout / Ain

        # Add quadratic noise budget: A_measured ≈ sqrt(A_signal^2 + sigma_noise^2)
        # — this is the rough magnitude an FFT-bin amplitude picks up.
        A_in_h  = float(np.sqrt(Ain**2  + rms_in_high**2))  if np.isfinite(rms_in_high)  else np.nan
        A_in_l  = float(np.sqrt(Ain**2  + rms_in_low**2))   if np.isfinite(rms_in_low)   else np.nan
        A_out_h = float(np.sqrt(Aout**2 + rms_out_high**2)) if np.isfinite(rms_out_high) else np.nan
        A_out_l = float(np.sqrt(Aout**2 + rms_out_low**2))  if np.isfinite(rms_out_low)  else np.nan
        Kt_h = A_out_h / A_in_h if A_in_h else np.nan
        Kt_l = A_out_l / A_in_l if A_in_l else np.nan

        rows.append({
            "amp": t["amp"],
            "A_in_mm":  Ain,
            "A_out_mm": Aout,
            "noise/Aout_high": rms_out_high / Aout if np.isfinite(rms_out_high) else np.nan,
            "noise/Aout_low":  rms_out_low  / Aout if np.isfinite(rms_out_low)  else np.nan,
            "K_t_clean":      Kt_clean,
            "K_t_high(noisy)": Kt_h,
            "K_t_low(noisy)":  Kt_l,
            "Δ_Kt(low-high)_predicted": (Kt_l - Kt_h) if (np.isfinite(Kt_h) and np.isfinite(Kt_l)) else np.nan,
        })
    bias_pred = pd.DataFrame(rows)
    print(bias_pred.round(4).to_string(index=False))
    print()
    print("Reading: predicted Δ_Kt(low − high) from random Gaussian noise added in")
    print("quadrature to the clean amplitude. If predicted ≈ 0, random noise floor")
    print("difference does NOT explain the +0.048 K_t bias — the bias must come")
    print("from a different mechanism (e.g. systematic, not random).")
    print()
    print(f"Observed within-loose230 mode bias (median): +0.048")

    # ── Alternative hypothesis: WAVE-RUN noise (with paddle running),
    # not stillwater noise. The pre-paddle 3 s window σ_η at OUT was already
    # computed by wind_qc_3s. That's a better measure of OUT-probe noise
    # under operating conditions. Check it.
    qc_path = BASE / "analysis_scratch" / "wind_qc_3s_per_run.csv"
    if qc_path.exists():
        qc = pd.read_csv(qc_path)
        # Mar 26 = h100/low, Mar 27 = h100/low, both canon. So this CSV
        # cannot give us a high-vs-low split (both days were lowrange).
        # We need the older highrange days.
        print()
        print("Note: wind_qc_3s only covers Mar 26 + Mar 27 (both h100/low). The")
        print("highrange σ at OUT under operating conditions is NOT in that CSV.")
        print("Above check uses STILLWATER noise floor — which is fine for")
        print("characterising the noise floor itself.")
print("\nDone.")
