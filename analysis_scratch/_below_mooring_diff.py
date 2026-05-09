"""Quantify K_t difference between below_90_loose230 and below_90_loose300
at fullpanel — same shape as _above50_panel_diff.py but for the within-below
mooring axis.

Run-level + pairwise Δ = K_t(loose230) - K_t(loose300).
"""
from pathlib import Path
import sys
import glob
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wavescripts.improved_data_loader import load_analysis_data
from wavescripts.filters import apply_experimental_filters

all_dirs = sorted(glob.glob(str(ROOT / "waveprocessed" / "PROCESSED-*")))
meta, _, _, _ = load_analysis_data(*all_dirs, load_processed=False)

m = apply_experimental_filters(meta, {
    "filters": {
        "Mooring":                    ["below_90_loose230", "below_90_loose300"],
        "PanelCondition":             "full",
        "WindCondition":              ["no", "full"],
        "WaveFrequencyInput [Hz]":    (1.3, 1.6),  # thesis band only — broader scope hid amp×wind signal
    },
}).copy()
m = m[m["WaveFrequencyInput [Hz]"].notna() & (m["WaveFrequencyInput [Hz]"] > 0)]
m = m[m["OUT/IN (FFT)"].notna()]
print(f"\nFiltered to {len(m)} runs (thesis band 1.3-1.6 Hz only)")

cell_keys = ["WaveAmplitudeInput [Volt]", "WaveFrequencyInput [Hz]", "WindCondition"]

# Find cells where BOTH moorings exist
counts = (m.groupby(cell_keys + ["Mooring"])["path"].count()
            .unstack("Mooring", fill_value=0))
cells_both = counts[(counts.get("below_90_loose230", 0) > 0)
                    & (counts.get("below_90_loose300", 0) > 0)].index.tolist()
print(f"\nCells with BOTH moorings present: {len(cells_both)}/{len(counts)}")

# ─── PER-RUN view ─────────────────────────────────────────────────────────────
print("\n" + "=" * 100)
print("PER-RUN K_t — every individual run in cells with both moorings")
print("=" * 100)
m_view = m.set_index(cell_keys).loc[cells_both].reset_index()
m_view = m_view.sort_values(cell_keys + ["Mooring", "path"])
m_view["per_tag"] = m_view["path"].apply(
    lambda p: next((tok for tok in str(p).split("-") if tok.startswith("per")), ""))
m_view["run_tag"] = m_view["path"].apply(
    lambda p: next((tok.replace(".csv", "") for tok in str(p).split("-")
                    if tok.startswith("run")), ""))
m_view["mooring_short"] = m_view["Mooring"].str.replace("below_90_", "")
print(m_view[cell_keys + ["mooring_short", "per_tag", "run_tag", "OUT/IN (FFT)"]]
      .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ─── PAIRWISE DELTA  (Δ = K_loose230 - K_loose300) ───────────────────────────
print("\n" + "=" * 100)
print("PAIRWISE DELTAS — every (loose230 run, loose300 run) pair within each cell")
print("=" * 100)
pair_rows = []
for (amp, freq, wind) in cells_both:
    cell = m_view[(m_view["WaveAmplitudeInput [Volt]"] == amp)
                  & (m_view["WaveFrequencyInput [Hz]"] == freq)
                  & (m_view["WindCondition"] == wind)]
    l230 = cell[cell["Mooring"] == "below_90_loose230"]["OUT/IN (FFT)"].tolist()
    l300 = cell[cell["Mooring"] == "below_90_loose300"]["OUT/IN (FFT)"].tolist()
    for k230 in l230:
        for k300 in l300:
            pair_rows.append({
                "amp": amp, "freq": freq, "wind": wind,
                "K_230": k230, "K_300": k300, "delta": k230 - k300,
            })
pairs = pd.DataFrame(pair_rows)
print(f"Total pairwise comparisons: {len(pairs)}")
print(pairs.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("\nPairwise Δ distribution (K_230 - K_300, per-run):")
print(f"  median:  {pairs['delta'].median():+.4f}")
print(f"  mean:    {pairs['delta'].mean():+.4f}")
print(f"  std:     {pairs['delta'].std():.4f}")
print(f"  range:   [{pairs['delta'].min():+.4f}, {pairs['delta'].max():+.4f}]")
print(f"  |Δ| ≤ 0.05: {(pairs['delta'].abs() <= 0.05).mean()*100:.0f}% of pairs")
print(f"  |Δ| ≤ 0.10: {(pairs['delta'].abs() <= 0.10).mean()*100:.0f}% of pairs")

print("\nΔ by wind condition:")
for w in ["no", "full"]:
    sub = pairs[pairs["wind"] == w]["delta"]
    if len(sub):
        print(f"  wind={w}: n_pairs={len(sub)}, mean Δ={sub.mean():+.4f}, "
              f"median={sub.median():+.4f}, range=[{sub.min():+.4f}, {sub.max():+.4f}]")
print("\nΔ by amplitude:")
for a in sorted(pairs["amp"].unique()):
    sub = pairs[pairs["amp"] == a]["delta"]
    if len(sub):
        print(f"  amp={a:.2f}V: n_pairs={len(sub)}, mean Δ={sub.mean():+.4f}, "
              f"median={sub.median():+.4f}")
print("\nΔ by frequency:")
for f in sorted(pairs["freq"].unique()):
    sub = pairs[pairs["freq"] == f]["delta"]
    if len(sub):
        print(f"  freq={f:.2f} Hz: n_pairs={len(sub)}, mean Δ={sub.mean():+.4f}, "
              f"median={sub.median():+.4f}")

# ─── Crossed breakdown: (amp × wind) — the user's hypothesis was that the gap
# is amplitude-dependent under fullwind. Show it explicitly.
print("\n" + "=" * 100)
print("Δ by (amp × wind) — crossed breakdown")
print("=" * 100)
print(f"{'amp':>8s}  {'wind':>5s}  {'n_pairs':>8s}  {'mean Δ':>10s}  {'median Δ':>10s}  {'range':>20s}")
for a in sorted(pairs["amp"].unique()):
    for w in ["no", "full"]:
        sub = pairs[(pairs["amp"] == a) & (pairs["wind"] == w)]["delta"]
        if len(sub):
            print(f"{a:8.2f}V  {w:>5s}  {len(sub):8d}  "
                  f"{sub.mean():+10.4f}  {sub.median():+10.4f}  "
                  f"[{sub.min():+.4f}, {sub.max():+.4f}]")
        else:
            print(f"{a:8.2f}V  {w:>5s}  {'(no data)':>8s}")

# ─── Cell-aggregated view (mean K_t per mooring per cell) for the thesis band
print("\n" + "=" * 100)
print("CELL-AGGREGATED Δ = mean(K_t,loose230) - mean(K_t,loose300) per cell")
print("(this is what the damping_freq plot's per-mooring lines show)")
print("=" * 100)
agg = (m.groupby(cell_keys + ["Mooring"])
        .agg(K=("OUT/IN (FFT)", "mean"),
             n=("path", "count"))
        .reset_index())
piv = agg.pivot_table(index=cell_keys, columns="Mooring",
                      values=["K", "n"]).reset_index()
piv.columns = ["_".join([str(c) for c in col if c]).strip("_") for col in piv.columns]
piv["delta_230_minus_300"] = piv.get("K_below_90_loose230", np.nan) - piv.get("K_below_90_loose300", np.nan)
piv = piv.dropna(subset=["delta_230_minus_300"])
piv = piv.sort_values(["WindCondition", "WaveAmplitudeInput [Volt]", "WaveFrequencyInput [Hz]"])
show_cols = cell_keys + ["n_below_90_loose230", "K_below_90_loose230",
                          "n_below_90_loose300", "K_below_90_loose300",
                          "delta_230_minus_300"]
print(piv[show_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
