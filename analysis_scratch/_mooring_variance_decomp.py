"""Variance decomposition for the damping_freq plot's std bars.

Question: are the visible std-bars in ch05_damping_freq dominated by
(a) lateral probe asymmetry within a run, (b) run-to-run noise within a
mooring, or (c) systematic mooring/date difference between loose230 and
loose300?

Approach: pull per-probe FFT amplitudes directly from meta.json, compute
three K_t values per run (wall-IN-only, far-IN-only, both-IN-mean), and
report per-cell spreads.

Scope: canon lowrange, panel=full, freq 1.3-1.6 Hz, wind in [no, full],
amp 0.1-0.3 V (matches the live ch05_damping_freq filter).
"""
from pathlib import Path
import sys
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wavescripts.improved_data_loader import load_analysis_data
from wavescripts.filters import apply_experimental_filters

PROCESSED_DIRS = [
    Path("waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange"),
    Path("waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"),
]

meta, _, _, _ = load_analysis_data(*PROCESSED_DIRS, load_processed=False)

# Apply the same quality_flag + scope filters that ch05_damping_freq uses,
# so this decomposition reports on the data the published plot actually sees.
# Earlier version bypassed apply_experimental_filters and included a
# dropout_critical row at A3×1.6Hz×nowind×loose300 which gave a misleading
# Kt_wall=1.082 outlier — that row is filtered here by quality_flag default.
m = apply_experimental_filters(meta, {
    "filters": {
        "PanelCondition":             "full",
        "WaveFrequencyInput [Hz]":    (1.3, 1.6),
        "WaveAmplitudeInput [Volt]":  (0.10, 0.30),
        "WindCondition":              ["no", "full"],
    },
}).copy()

# Re-derive mooring tag from file_date
m["Mooring"] = m["file_date"].astype(str).map({
    "2026-03-26": "loose230",
    "2026-03-27": "loose300",
})

# Extract per-probe amplitudes
A_in_wall = m["Probe 9373/170 Amplitude (FFT)"].astype(float)
A_in_far  = m["Probe 9373/340 Amplitude (FFT)"].astype(float)
A_out_ctr = m["Probe 12400/250 Amplitude (FFT)"].astype(float)

m["Kt_wall"] = A_out_ctr / A_in_wall
m["Kt_far"]  = A_out_ctr / A_in_far
m["Kt_mean"] = A_out_ctr / ((A_in_wall + A_in_far) / 2.0)
m["probe_disagree_frac"] = (m["Kt_wall"] - m["Kt_far"]).abs() / m["Kt_mean"]

# Tag per (period count) for visibility
m["per_tag"] = m["path"].apply(
    lambda p: next((tok for tok in str(p).split("-") if tok.startswith("per")), "")
)
m["run_tag"] = m["path"].apply(
    lambda p: next((tok.replace(".csv", "") for tok in str(p).split("-")
                    if tok.startswith("run")), "")
)

cols_show = ["WaveAmplitudeInput [Volt]", "WaveFrequencyInput [Hz]",
             "WindCondition", "Mooring", "per_tag", "run_tag",
             "Kt_wall", "Kt_far", "Kt_mean", "probe_disagree_frac"]
m_view = m[cols_show].sort_values(cols_show[:6]).reset_index(drop=True)

print("\n" + "=" * 100)
print("PER-RUN K_t VALUES (sorted by amp/freq/wind/mooring)")
print("=" * 100)
print(m_view.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ─── Cell-level summary ─────────────────────────────────────────────────────
print("\n" + "=" * 100)
print("CELL-LEVEL SUMMARY: spread within mooring vs between moorings")
print("=" * 100)
print("Columns: n=run count, K_mean=avg of Kt_mean, sd_within=std within mooring,")
print("         abs(Δ_mooring) = |Kt(loose230) - Kt(loose300)| (cell-level means)")
print()

cell_keys = ["WaveAmplitudeInput [Volt]", "WaveFrequencyInput [Hz]", "WindCondition"]
rows = []
for (amp, freq, wind), g in m.groupby(cell_keys):
    m230 = g[g["Mooring"] == "loose230"]["Kt_mean"].dropna()
    m300 = g[g["Mooring"] == "loose300"]["Kt_mean"].dropna()
    rows.append({
        "amp":       amp,
        "freq":      freq,
        "wind":      wind,
        "n_230":     len(m230),
        "n_300":     len(m300),
        "K_230":     m230.mean() if len(m230) else np.nan,
        "K_300":     m300.mean() if len(m300) else np.nan,
        "sd_w_230":  m230.std()  if len(m230) > 1 else np.nan,
        "sd_w_300":  m300.std()  if len(m300) > 1 else np.nan,
        "Δ_moor":    (m230.mean() - m300.mean()) if (len(m230) and len(m300)) else np.nan,
        "probe_dis_max": g["probe_disagree_frac"].max(),
    })
summary = pd.DataFrame(rows)
print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("\n" + "=" * 100)
print("HEADLINE NUMBERS")
print("=" * 100)
both = summary.dropna(subset=["Δ_moor"])
print(f"Cells with both moorings present: {len(both)}/{len(summary)}")
print(f"Δ_mooring (signed K_230 − K_300):")
print(f"   mean: {both['Δ_moor'].mean():+.4f}")
print(f"   std:  {both['Δ_moor'].std():.4f}")
print(f"   |Δ| range: [{both['Δ_moor'].abs().min():.4f}, {both['Δ_moor'].abs().max():.4f}]")
print()
sd_w_pool = pd.concat([summary["sd_w_230"], summary["sd_w_300"]]).dropna()
print(f"Within-mooring run-to-run std (pooled both moorings, only n>1 cells):")
print(f"   n_cells: {len(sd_w_pool)}")
print(f"   median:  {sd_w_pool.median():.4f}")
print(f"   mean:    {sd_w_pool.mean():.4f}")
print(f"   max:     {sd_w_pool.max():.4f}")
print()
print(f"Probe disagreement fraction |Kt_wall - Kt_far| / Kt_mean (across all runs):")
print(f"   median:  {m['probe_disagree_frac'].median():.4f}")
print(f"   mean:    {m['probe_disagree_frac'].mean():.4f}")
print(f"   max:     {m['probe_disagree_frac'].max():.4f}")

# Save the per-run table for further inspection
out_csv = ROOT / "analysis_scratch" / "_mooring_variance_decomp.csv"
m_view.to_csv(out_csv, index=False)
print(f"\nPer-run table saved to: {out_csv.name}")
