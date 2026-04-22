"""
Mooring-type vs SW bias: do above_50 and below_90 differ in OUT/IN
by the amount predicted by the IN probe's position in the standing wave?

Context
-------
above_50  mooring line ≈ 7–9 cm → panel stays ≈ 5 cm from 9373/170 probe
below_90  mooring line ≈ 23–30 cm → panel stays ≈ 25 cm from 9373/170 probe

With R ≈ 0.07 the SW_factor at those positions differs by 5–12 % across
0.9–1.5 Hz.  If this bias explains the observed OUT/IN difference, the two
mooring types have identical true transmission — the difference is a pure
measurement artefact from the IN probe sitting at different standing-wave
phase offsets.

Method
------
1. Load combined_meta; filter nowind, 0.2V, fullpanel, quality_flag==ok.
2. Group by (freq, mooring); compute mean OUT/IN(FFT) and n.
3. Pivot → above_50 vs below_90 side by side; compute observed ratio.
4. Compute theoretical SW_factor per mooring per frequency (R=0.07, best-
   estimate panel positions).
5. Print comparison table.

Run from repo root:
    conda run -n draumkvedet python analysis_scratch/mooring_sw_bias.py
"""

import sys, warnings, glob
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from wavescripts.improved_data_loader import load_analysis_data

# ── constants ─────────────────────────────────────────────────────────────────
G, DEPTH   = 9.81, 0.580
X_IN       = 9.373          # IN probe longitudinal position (m)

R          = 0.07           # reflection coefficient (MF median)

# Best-estimate panel positions (leading-edge x, metres)
# above_50:  mooring ≈ 7–9 cm → panel ≈ 5 cm from probe  → x_panel ≈ 9.42 m
# below_90:  mooring ≈ 23–30 cm → panel ≈ 25 cm from probe → x_panel ≈ 9.62 m
PANEL_ABOVE = 9.42   # m
PANEL_BELOW = 9.62   # m

BASE = Path(__file__).parent.parent

# ── helpers ───────────────────────────────────────────────────────────────────
def solve_k(f):
    omega = 2 * np.pi * f
    return brentq(lambda k: omega**2 - G * k * np.tanh(k * DEPTH), 1e-4, 500.0)

def sw_factor(f, R, x_panel):
    k     = solve_k(f)
    delta = x_panel - X_IN
    return np.sqrt(1 + R**2 + 2 * R * np.cos(2 * k * delta))

# ── 1. load metadata ──────────────────────────────────────────────────────────
print("Loading metadata...")
dirs = sorted(glob.glob(str(BASE / "waveprocessed" / "PROCESSED-*")))
meta, _, _, _ = load_analysis_data(*dirs, load_processed=False)
print(f"  {len(meta)} total rows")

# ── 2. filter ─────────────────────────────────────────────────────────────────
sel = meta[
    meta["WaveFrequencyInput [Hz]"].notna() &
    (meta["WaveFrequencyInput [Hz]"] > 0) &
    (meta["WaveFrequencyInput [Hz]"] >= 0.80) &   # sub-1 Hz out of scope
    (meta["WaveAmplitudeInput [Volt]"] == 0.2) &
    (meta["WindCondition"] == "no") &
    (meta["PanelCondition"] == "full") &
    (meta["quality_flag"] == "ok") &
    meta["Mooring"].isin(["above_50", "below_90_loose230", "below_90_loose300"])
].copy()

print(f"  {len(sel)} nowind 0.2V fullpanel ok runs ≥ 0.8 Hz")

# Merge the two below_90 variants
sel["mooring_grp"] = sel["Mooring"].replace({
    "below_90_loose230": "below_90",
    "below_90_loose300": "below_90",
})
print(f"  Mooring groups: {sel['mooring_grp'].value_counts().to_dict()}")

# ── 3. compute OUT/IN(FFT) per run ────────────────────────────────────────────
# Use the FFT amplitude columns already in meta
pos_in  = "9373/170"
pos_out = "12400/250"
col_in  = f"Probe {pos_in} Amplitude (FFT)"
col_out = f"Probe {pos_out} Amplitude (FFT)"

if col_in not in sel.columns or col_out not in sel.columns:
    raise RuntimeError(f"Missing amplitude columns. Available: {[c for c in sel.columns if 'Amplitude' in c]}")

sel = sel[sel[col_in].notna() & sel[col_out].notna() & (sel[col_in] > 0)].copy()
sel["outIN_fft"] = sel[col_out] / sel[col_in]
print(f"  {len(sel)} runs with valid FFT amplitudes")

# ── 4. group by (freq, mooring_grp) ──────────────────────────────────────────
grp = (
    sel.groupby(["WaveFrequencyInput [Hz]", "mooring_grp"])["outIN_fft"]
    .agg(mean="mean", std="std", n="count")
    .reset_index()
)

pivot = grp.pivot(index="WaveFrequencyInput [Hz]", columns="mooring_grp", values="mean")
pivot_n = grp.pivot(index="WaveFrequencyInput [Hz]", columns="mooring_grp", values="n")

freqs = sorted(pivot.index)

# ── 5. compare to theory ──────────────────────────────────────────────────────
print()
print("=" * 90)
print(f"{'freq':>5}  {'OUT/IN above':>13}  {'OUT/IN below':>13}  {'n above':>7}  {'n below':>7}  "
      f"{'obs ratio':>10}  {'SW theory':>10}  {'residual':>9}")
print(f"{'Hz':>5}  {'(above_50)':>13}  {'(below_90)':>13}  {'':>7}  {'':>7}  "
      f"{'A/B obs':>10}  {'A/B pred':>10}  {'obs-pred':>9}")
print("-" * 90)

rows = []
for f in freqs:
    if "above_50" not in pivot.columns or "below_90" not in pivot.columns:
        continue
    a = pivot.loc[f, "above_50"]   if f in pivot.index else np.nan
    b = pivot.loc[f, "below_90"]   if f in pivot.index else np.nan
    na = int(pivot_n.loc[f, "above_50"]) if (f in pivot_n.index and not np.isnan(pivot_n.loc[f, "above_50"])) else 0
    nb = int(pivot_n.loc[f, "below_90"]) if (f in pivot_n.index and not np.isnan(pivot_n.loc[f, "below_90"])) else 0

    if np.isnan(a) or np.isnan(b) or b == 0:
        continue

    obs_ratio  = a / b

    # Theoretical SW_factor ratio: above / below
    # SW_factor accounts for IN probe reading differently between the two moorings.
    # If true T is the same, then:
    #   OUT/IN_above = T / SW_above  →  observed is deflated by SW_above
    #   OUT/IN_below = T / SW_below
    #   Predicted obs ratio = SW_below / SW_above
    sw_a = sw_factor(f, R, PANEL_ABOVE)
    sw_b = sw_factor(f, R, PANEL_BELOW)
    pred_ratio = sw_b / sw_a  # predicted OUT/IN_above / OUT/IN_below

    residual = obs_ratio - pred_ratio

    rows.append(dict(freq=f, above=a, below=b, na=na, nb=nb,
                     obs=obs_ratio, pred=pred_ratio, resid=residual))

    print(f"  {f:4.2f}  {a:13.4f}  {b:13.4f}  {na:7d}  {nb:7d}  "
          f"  {obs_ratio:9.4f}  {pred_ratio:9.4f}  {residual:+9.4f}")

print()
df_out = pd.DataFrame(rows)
if not df_out.empty:
    print(f"Mean |residual|  = {df_out['resid'].abs().mean():.4f}")
    print(f"Mean residual    = {df_out['resid'].mean():+.4f}  (positive = above > predicted)")
    print(f"Std residual     = {df_out['resid'].std():.4f}")
    print()
    print("Interpretation:")
    print("  If residual ≈ 0 at all freqs → observed difference IS the SW bias (same true T)")
    print("  If residual is systematic → real mooring-type effect on transmission")

# ── 6. also print the raw above/below difference ─────────────────────────────
print()
print("=" * 90)
print("Raw difference: OUT/IN_above − OUT/IN_below and SW bias prediction")
print(f"{'freq':>5}  {'above':>8}  {'below':>8}  {'obs diff':>10}  {'SW bias':>10}  {'unexplained':>12}")
print(f"{'Hz':>5}  {'':>8}  {'':>8}  {'A−B obs':>10}  {'predicted':>10}  {'obs−pred diff':>12}")
print("-" * 70)
for r in rows:
    f, a, b = r["freq"], r["above"], r["below"]
    obs_diff  = a - b
    sw_a = sw_factor(f, R, PANEL_ABOVE)
    sw_b = sw_factor(f, R, PANEL_BELOW)
    # If T_above = T_below = T:
    #   A = T/sw_a, B = T/sw_b  (OUT/IN = T / SW_factor because IN is amplified by SW)
    #   pred_diff = A − B = T*(1/sw_a − 1/sw_b)
    # Use mean T as proxy: T ≈ (a*sw_a + b*sw_b) / 2
    T_est = (a * sw_a + b * sw_b) / 2.0
    pred_diff = T_est * (1/sw_a - 1/sw_b)
    unexplained = obs_diff - pred_diff
    print(f"  {f:4.2f}  {a:8.4f}  {b:8.4f}  {obs_diff:+10.4f}  {pred_diff:+10.4f}  {unexplained:+12.4f}")

print()
print("Done.")
