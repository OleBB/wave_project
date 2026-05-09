"""Compare canon-lowrange loose230 (cond4) against the OTHER valid-range
loose230 data (cond1 h272/high + cond2 h136/high). Exclude cond3
(h100/high) because that's the known dropout-prone configuration.

Question: at the wave-amplitude scale (FFT magnitude at paddle freq), is
the canon-lowrange measurement systematically different from the
high-range measurement at higher panel mounts? If not, the pool of
usable loose230 data nearly doubles.
"""
from pathlib import Path
import sys
import glob
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wavescripts.improved_data_loader import load_analysis_data
from wavescripts.constants import PROBE_HEIGHT_DEFAULT_MM

all_dirs = sorted(glob.glob(str(ROOT / "waveprocessed" / "PROCESSED-*")))
meta, _, _, _ = load_analysis_data(*all_dirs, load_processed=False)


def _tag_condition(row):
    h = row.get("probe_height_mm", PROBE_HEIGHT_DEFAULT_MM)
    r = row.get("probe_range_mode", "high")
    if pd.isna(h):
        h = PROBE_HEIGHT_DEFAULT_MM
    h = int(h)
    if h == 272 and r == "high": return "cond1_h272_high"
    if h == 136 and r == "high": return "cond2_h136_high"
    if h == 100 and r == "high": return "cond3_h100_high_WRONG"
    if h == 100 and r == "low":  return "cond4_h100_low_canon"
    return "other"


m = meta[(meta["Mooring"] == "below_90_loose230")
         & (meta["PanelCondition"] == "full")
         & (meta["quality_flag"] == "ok")
         & meta["WaveFrequencyInput [Hz]"].notna()
         & (meta["WaveFrequencyInput [Hz]"] > 0)
         & meta["WindCondition"].isin(["no", "full"])
         & meta["OUT/IN (FFT)"].notna()
         & (meta["WaveFrequencyInput [Hz]"].between(1.3, 1.6))].copy()
m["cond"] = m.apply(_tag_condition, axis=1)

# Exclude cond3 (dropout-prone) and cond2 (intermediate height — too small
# a sample to add anything; user wants the cleanest comparison: extreme
# height delta h272 vs h100, each in its valid range mode).
m = m[m["cond"].isin(["cond1_h272_high", "cond4_h100_low_canon"])].copy()
m["era"] = m["cond"].map({
    "cond4_h100_low_canon": "canon",
    "cond1_h272_high":      "h272_high",
})

print("loose230, thesis band 1.3-1.6 Hz, panel=full, valid-range only:")
print(m.groupby(["era", "WindCondition"]).size().unstack(fill_value=0))
print()

cell_keys = ["WaveAmplitudeInput [Volt]", "WaveFrequencyInput [Hz]", "WindCondition"]
counts = (m.groupby(cell_keys + ["era"])["path"].count()
            .unstack("era", fill_value=0))
cells_both = counts[(counts.get("canon", 0) > 0)
                    & (counts.get("h272_high", 0) > 0)].index.tolist()
print(f"Cells with BOTH canon AND h272_high loose230 data: {len(cells_both)}")
print(counts.to_string())
print()

# ─── Per-run table ────────────────────────────────────────────────────────────
print("=" * 100)
print("PER-RUN K_t — canon (cond4 h100/low) vs cond1 (h272/high) loose230")
print("=" * 100)
m_view = m.sort_values(cell_keys + ["era", "path"]).copy()
m_view["per_tag"] = m_view["path"].apply(
    lambda p: next((tok for tok in str(p).split("-") if tok.startswith("per")), ""))
m_view["date"] = m_view["path"].apply(
    lambda p: str(p).split("/wavedata/")[-1][:8])
print(m_view[cell_keys + ["era", "cond", "date", "per_tag", "OUT/IN (FFT)"]]
      .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ─── Pairwise Δ in cells with both eras ───────────────────────────────────────
print("\n" + "=" * 100)
print("PAIRWISE Δ = K_canon − K_h272_high")
print("=" * 100)
pair_rows = []
for (amp, freq, wind) in cells_both:
    cell = m_view[(m_view["WaveAmplitudeInput [Volt]"] == amp)
                  & (m_view["WaveFrequencyInput [Hz]"] == freq)
                  & (m_view["WindCondition"] == wind)]
    canon_runs = cell[cell["era"] == "canon"]["OUT/IN (FFT)"].tolist()
    other_runs = cell[cell["era"] == "h272_high"]["OUT/IN (FFT)"].tolist()
    for kc in canon_runs:
        for ko in other_runs:
            pair_rows.append({
                "amp": amp, "freq": freq, "wind": wind,
                "K_canon": kc, "K_other": ko, "delta": kc - ko,
            })
pairs = pd.DataFrame(pair_rows)
print(f"Total pairs: {len(pairs)}")
if len(pairs):
    print("\nPairwise Δ distribution:")
    d = pairs["delta"]
    print(f"  median:  {d.median():+.4f}")
    print(f"  mean:    {d.mean():+.4f}")
    print(f"  std:     {d.std():.4f}")
    print(f"  range:   [{d.min():+.4f}, {d.max():+.4f}]")
    print(f"  |Δ| ≤ 0.05: {(d.abs() <= 0.05).mean()*100:.0f}% of pairs")
    print(f"  |Δ| ≤ 0.10: {(d.abs() <= 0.10).mean()*100:.0f}% of pairs")

    # Crossed (amp × wind)
    print("\nΔ by (amp × wind):")
    print(f"{'amp':>8s}  {'wind':>5s}  {'n_pairs':>8s}  {'mean Δ':>10s}  {'median':>10s}")
    for a in sorted(pairs["amp"].unique()):
        for w in ["no", "full"]:
            sub = pairs[(pairs["amp"] == a) & (pairs["wind"] == w)]["delta"]
            if len(sub):
                print(f"{a:8.2f}V  {w:>5s}  {len(sub):8d}  "
                      f"{sub.mean():+10.4f}  {sub.median():+10.4f}")
            else:
                print(f"{a:8.2f}V  {w:>5s}  {'(no data)':>8s}")

# ─── Cell-aggregated view ─────────────────────────────────────────────────────
print("\n" + "=" * 100)
print("CELL-AGGREGATED Δ = mean(K_t,canon) - mean(K_t,other_valid) per cell")
print("=" * 100)
agg = (m.groupby(cell_keys + ["era"])
        .agg(K=("OUT/IN (FFT)", "mean"),
             K_std=("OUT/IN (FFT)", "std"),
             n=("path", "count"))
        .reset_index())
piv = agg.pivot_table(index=cell_keys, columns="era",
                      values=["K", "K_std", "n"]).reset_index()
piv.columns = ["_".join([str(c) for c in col if c]).strip("_") for col in piv.columns]
piv["delta_canon_minus_other"] = piv.get("K_canon") - piv.get("K_h272_high")
piv = piv.dropna(subset=["delta_canon_minus_other"])
piv = piv.sort_values(["WindCondition", "WaveAmplitudeInput [Volt]", "WaveFrequencyInput [Hz]"])
show = cell_keys + ["n_canon", "K_canon", "K_std_canon",
                     "n_h272_high", "K_h272_high", "K_std_h272_high",
                     "delta_canon_minus_other"]
print(piv[show].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
