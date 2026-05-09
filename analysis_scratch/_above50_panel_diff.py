"""Quantify K_t difference between above-50 fullpanel and above-50 reversepanel.

If the difference is small, we can pool them in the all-data scatter for
easier reading (4 categories → 3).

Per (amp, freq, wind) cell, compute mean K_t for each panel, then
delta = K(reverse) - K(full).
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

# Filter: above_50 mooring, panel ∈ {full, reverse}, wind ∈ {no, full}
m = apply_experimental_filters(meta, {
    "filters": {
        "Mooring":                    "above_50",
        "PanelCondition":             ["full", "reverse"],
        "WindCondition":              ["no", "full"],
    },
}).copy()
m = m[m["WaveFrequencyInput [Hz]"].notna() & (m["WaveFrequencyInput [Hz]"] > 0)]
m = m[m["OUT/IN (FFT)"].notna()]
print(f"\nFiltered to {len(m)} runs")

# ─── PER-RUN view: list every individual K_t in cells where BOTH panels exist ─
# So the reader can see run-to-run spread directly, not just mean-per-panel.
cell_keys = ["WaveAmplitudeInput [Volt]", "WaveFrequencyInput [Hz]", "WindCondition"]
cells_both = (m.groupby(cell_keys + ["PanelCondition"])["path"].count()
                .unstack("PanelCondition", fill_value=0))
cells_both = cells_both[(cells_both.get("full", 0) > 0)
                        & (cells_both.get("reverse", 0) > 0)].index.tolist()

print("\n" + "=" * 100)
print("PER-RUN K_t — every individual run in the 6 cells with both panels")
print("=" * 100)
m_view = m.set_index(cell_keys).loc[cells_both].reset_index()
m_view = m_view.sort_values(cell_keys + ["PanelCondition", "path"])
m_view["per_tag"] = m_view["path"].apply(
    lambda p: next((tok for tok in str(p).split("-") if tok.startswith("per")), ""))
m_view["run_tag"] = m_view["path"].apply(
    lambda p: next((tok.replace(".csv", "") for tok in str(p).split("-")
                    if tok.startswith("run")), ""))
m_view["date"] = m_view["path"].apply(
    lambda p: str(p).split("/wavedata/")[-1][:8])
print(m_view[cell_keys + ["PanelCondition", "date", "per_tag", "run_tag",
                          "OUT/IN (FFT)"]]
      .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ─── PAIRWISE DELTA distribution (every full × reverse pair in same cell) ────
# For each (amp, freq, wind) cell with both panels: enumerate the cartesian
# product of (full runs) × (reverse runs), compute K_rev - K_full per pair.
# This avoids the "mean-of-means" smoothing and shows the full spread of
# observed deltas at the run level.
print("\n" + "=" * 100)
print("PAIRWISE DELTAS — every (full run, reverse run) pair within each cell")
print("=" * 100)
pair_rows = []
for (amp, freq, wind) in cells_both:
    cell = m_view[(m_view["WaveAmplitudeInput [Volt]"] == amp)
                  & (m_view["WaveFrequencyInput [Hz]"] == freq)
                  & (m_view["WindCondition"] == wind)]
    fulls = cell[cell["PanelCondition"] == "full"]["OUT/IN (FFT)"].tolist()
    revs  = cell[cell["PanelCondition"] == "reverse"]["OUT/IN (FFT)"].tolist()
    for kf in fulls:
        for kr in revs:
            pair_rows.append({
                "amp": amp, "freq": freq, "wind": wind,
                "K_full": kf, "K_rev": kr, "delta": kr - kf,
            })
pairs = pd.DataFrame(pair_rows)
print(f"Total pairwise comparisons: {len(pairs)} (cartesian {{full}} × {{reverse}} per cell)")
print(pairs.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("\nPairwise Δ distribution (per-run, no mean smoothing):")
print(f"  median:  {pairs['delta'].median():+.4f}")
print(f"  mean:    {pairs['delta'].mean():+.4f}")
print(f"  std:     {pairs['delta'].std():.4f}")
print(f"  range:   [{pairs['delta'].min():+.4f}, {pairs['delta'].max():+.4f}]")
print(f"  |Δ| ≤ 0.05: {(pairs['delta'].abs() <= 0.05).mean()*100:.0f}% of pairs")
print(f"  |Δ| ≤ 0.10: {(pairs['delta'].abs() <= 0.10).mean()*100:.0f}% of pairs")


# Per-cell mean K_t per panel
cell_keys = ["WaveAmplitudeInput [Volt]", "WaveFrequencyInput [Hz]", "WindCondition"]
agg = (m.groupby(cell_keys + ["PanelCondition"])
        .agg(K=("OUT/IN (FFT)", "mean"),
             K_std=("OUT/IN (FFT)", "std"),
             n=("path", "count"))
        .reset_index())

# Pivot so each row is one (amp, freq, wind) cell with K_full, K_reverse columns
pivot = agg.pivot_table(
    index=cell_keys,
    columns="PanelCondition",
    values=["K", "K_std", "n"],
).reset_index()
pivot.columns = [
    "_".join([str(c) for c in col if c]).strip("_") for col in pivot.columns
]
pivot = pivot.rename(columns={
    "K_full": "K_full", "K_reverse": "K_rev",
    "K_std_full": "sd_full", "K_std_reverse": "sd_rev",
    "n_full": "n_full", "n_reverse": "n_rev",
})
pivot["delta_rev_minus_full"] = pivot["K_rev"] - pivot["K_full"]

both = pivot.dropna(subset=["K_full", "K_rev"]).copy()

print("\n" + "=" * 100)
print("PER-CELL K_t: full panel vs reverse panel  (both at above_50 mooring)")
print("=" * 100)
print(both[cell_keys + ["n_full", "K_full", "sd_full",
                         "n_rev",  "K_rev",  "sd_rev",
                         "delta_rev_minus_full"]]
      .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("\n" + "=" * 100)
print("HEADLINE NUMBERS  Δ = K_t(reverse) - K_t(full)")
print("=" * 100)
d = both["delta_rev_minus_full"]
print(f"Cells with both panels present: {len(both)}/{len(pivot)}")
print(f"Mean Δ: {d.mean():+.4f}")
print(f"Std Δ:   {d.std():.4f}")
print(f"Median:  {d.median():+.4f}")
print(f"Range:   [{d.min():+.4f}, {d.max():+.4f}]")
print(f"|Δ| median: {d.abs().median():.4f}")
print(f"|Δ| max:    {d.abs().max():.4f}")
print()

# Compare to within-panel run-to-run noise floor for the same data
within_panel_sd = pd.concat([both["sd_full"], both["sd_rev"]]).dropna()
print(f"Within-panel run-to-run std (pooled across cells, n>1 only):")
print(f"  median: {within_panel_sd.median():.4f}")
print(f"  mean:   {within_panel_sd.mean():.4f}")
print(f"  max:    {within_panel_sd.max():.4f}")

print()
print("INTERPRETATION:")
ratio = d.abs().median() / within_panel_sd.median() if within_panel_sd.median() else float("inf")
print(f"  |Δ_panel|_median / sd_within_panel_median = {ratio:.2f}")
if ratio < 1.0:
    print(f"  → panel difference is SMALLER than within-panel noise: SAFE TO POOL")
elif ratio < 2.0:
    print(f"  → panel difference is COMPARABLE to noise: maybe pool, maybe not")
else:
    print(f"  → panel difference EXCEEDS noise by {ratio:.1f}×: KEEP SEPARATE")

# Also show by wind condition — maybe the panel effect depends on wind
print("\nΔ by wind condition:")
for w in ["no", "full"]:
    sub = both[both["WindCondition"] == w]["delta_rev_minus_full"]
    if len(sub):
        print(f"  wind={w}: n={len(sub)}, mean Δ={sub.mean():+.4f}, "
              f"|Δ| median={sub.abs().median():.4f}, range=[{sub.min():+.4f}, {sub.max():+.4f}]")
