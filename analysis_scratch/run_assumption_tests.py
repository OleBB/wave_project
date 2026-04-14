"""
Assumption audit test runner
============================
Tests 6a, 6b, 6d, 1a, 2a from analysis_scratch/assumption_audit.md

Run with:
  conda run -n draumkvedet python analysis_scratch/run_assumption_tests.py 2>&1 | tee analysis_scratch/run_assumption_tests_output.txt
"""

import sys, os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import brentq

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
os.chdir(repo_root)

from wavescripts.improved_data_loader import load_analysis_data

# ── Constants ─────────────────────────────────────────────────────────────────
g   = 9.81    # m/s²
d   = 0.580   # tank depth, m
R_ASSUMED = 0.20  # assumed reflection coefficient for phase-sensitivity illustration

# Panel centre is approximate. Distance from IN probe to panel front/centre:
# IN probe at 9.373 m, panel centroid noted as ~11.0 m in reflection cells.
# Panel length from project context is the "L" in kL — need to confirm, use 2.0 m guess.
X_IN   = 9.373   # m
X_PANEL_CENTRE = 11.0   # m  — approximate, verify from lab drawings
DELTA  = X_PANEL_CENTRE - X_IN   # m, distance IN probe → panel centroid

PROBE_SEPARATION = 0.569   # m between 8804/250 and 9373/170 (for Mansard-Funke)

SEPARATOR = "\n" + "═" * 72 + "\n"

# ── Data load: standard dirs (no 20260307) ────────────────────────────────────
PROCESSED_DIRS_MAIN = [
    Path("waveprocessed/PROCESSED-20260312-ProbPos4_31_FPV_2-tett6roof"),
    Path("waveprocessed/PROCESSED-20260313-ProbePos4_31_FPV_2-tett6roof"),
    Path("waveprocessed/PROCESSED-20260314-ProbePos4_31_FPV_2-tett6roof"),
    Path("waveprocessed/PROCESSED-20260316-ProbePos4_31_FPV_2-tett6roof"),
    Path("waveprocessed/PROCESSED-20260316-ProbePos4_31_FPV_2-tett6roof-under9Mooring"),
    Path("waveprocessed/PROCESSED-20260319-ProbePos4_31_FPV_2-tett6roof-under9Mooring"),
    Path("waveprocessed/PROCESSED-20260321-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-RENAMED"),
    Path("waveprocessed/PROCESSED-20260323-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height136"),
    Path("waveprocessed/PROCESSED-20260323-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100"),
    Path("waveprocessed/PROCESSED-20260324-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100"),
    Path("waveprocessed/PROCESSED-20260325-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100"),
    Path("waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100"),
    Path("waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange"),
    Path("waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"),
]

# 20260307 adds no-panel (and reverse-panel) runs for the current config
PROCESSED_DIRS_WITH_20260307 = [
    Path("waveprocessed/PROCESSED-20260307-ProbPos4_31_FPV_2-tett6roof"),
] + PROCESSED_DIRS_MAIN

# ── Dispersion helper ─────────────────────────────────────────────────────────
def solve_k(f):
    """Wavenumber from full dispersion relation at depth d."""
    omega = 2 * np.pi * f
    return brentq(lambda k: omega**2 - g * k * np.tanh(k * d), 1e-4, 200.0)

def c_group(f):
    """Group velocity."""
    k = solve_k(f)
    omega = 2 * np.pi * f
    return (omega / k) * 0.5 * (1 + 2*k*d / np.sinh(2*k*d))

# ─────────────────────────────────────────────────────────────────────────────
print("Loading data (main dirs, no 20260307)...")
meta, _, _, _ = load_analysis_data(*PROCESSED_DIRS_MAIN, load_processed=False)
print(f"  {len(meta)} total runs loaded")

# ═════════════════════════════════════════════════════════════════════════════
print(SEPARATOR + "TEST 6a — Verify in_position / out_position for all standard runs")
# ═════════════════════════════════════════════════════════════════════════════
standard = meta[meta["run_category"] == "standard"].copy()
print(f"Standard runs: {len(standard)}")

pos_counts = standard.groupby(
    ["in_position", "out_position"], dropna=False
).size().reset_index(name="n")
print("\nAll in_position / out_position combinations found:")
print(pos_counts.to_string(index=False))

unexpected = standard[
    (standard["in_position"] != "9373/170") | (standard["out_position"] != "12400/250")
]
print(f"\nRuns with unexpected positions: {len(unexpected)}")
if len(unexpected):
    print(unexpected[["file_date", "path", "in_position", "out_position"]].to_string())
else:
    print("  → All standard runs have in=9373/170, out=12400/250  ✓")

nan_in  = standard["in_position"].isna().sum()
nan_out = standard["out_position"].isna().sum()
print(f"\nNaN in_position: {nan_in},  NaN out_position: {nan_out}")

# ═════════════════════════════════════════════════════════════════════════════
print(SEPARATOR + "TEST 6b — Load 20260307 and check no-panel inventory")
# ═════════════════════════════════════════════════════════════════════════════
print("Loading data WITH 20260307...")
meta_w, _, _, _ = load_analysis_data(*PROCESSED_DIRS_WITH_20260307, load_processed=False)
print(f"  {len(meta_w)} total runs (was {len(meta)})")

nopanel = meta_w[
    (meta_w["PanelCondition"] == "no") &
    (meta_w["WaveFrequencyInput [Hz]"].notna()) &
    (meta_w["WaveFrequencyInput [Hz]"] > 0)
].copy()
print(f"\nNo-panel wave runs in extended load: {len(nopanel)}")
if len(nopanel):
    print(f"in_position values: {nopanel['in_position'].unique()}")
    print(f"out_position values: {nopanel['out_position'].unique()}")
    print(f"\nFrequencies: {sorted(nopanel['WaveFrequencyInput [Hz]'].dropna().unique())}")
    print(f"WindConditions: {sorted(nopanel['WindCondition'].dropna().unique())}")
    print(f"WaveAmplitude values: {sorted(nopanel['WaveAmplitudeInput [Volt]'].dropna().unique())}")
    nopanel_dates = nopanel["file_date"].unique()
    print(f"Dates: {sorted(nopanel_dates)}")
else:
    print("  ⚠  No no-panel runs found even after adding 20260307!")

# ═════════════════════════════════════════════════════════════════════════════
print(SEPARATOR + "TEST 6d — No-panel OUT/IN (current config, new baseline)")
# ═════════════════════════════════════════════════════════════════════════════
nopanel_curr = nopanel[
    (nopanel["in_position"] == "9373/170") &
    (nopanel["WindCondition"] == "no")
].copy()
print(f"No-panel, no-wind, current config: {len(nopanel_curr)} runs")

if len(nopanel_curr):
    grp = nopanel_curr.groupby("WaveFrequencyInput [Hz]")["OUT/IN (FFT)"].agg(
        ["mean", "std", "count"]
    )
    grp.columns = ["mean_OUT_IN", "std", "n"]
    print("\nNo-panel OUT/IN by frequency (no-wind, current probe config):")
    print(grp.round(4).to_string())

    # Compare with full-panel no-wind
    fullpanel = meta_w[
        (meta_w["PanelCondition"] == "full") &
        (meta_w["WindCondition"] == "no") &
        (meta_w["in_position"] == "9373/170") &
        (meta_w["quality_flag"].isin(["ok", "probe_malfunction_secondary"]) if "quality_flag" in meta_w.columns else True)
    ].copy()
    fp_grp = fullpanel.groupby("WaveFrequencyInput [Hz]")["OUT/IN (FFT)"].mean()

    print("\nFull-panel / no-panel ratio (corrected transmission T):")
    ratio = fp_grp / grp["mean_OUT_IN"]
    print(ratio.dropna().round(4).to_string())
    print("\n(ratio < 1 = panel damps; ratio > 1 = apparent amplification)")

    # Check if no-panel OUT/IN is flat across frequency
    if len(grp) >= 3:
        freq_arr = grp.index.values.astype(float)
        outin_arr = grp["mean_OUT_IN"].values
        slope, intercept = np.polyfit(freq_arr, outin_arr, 1)
        print(f"\nNo-panel OUT/IN linear trend vs frequency: slope = {slope:.4f} per Hz")
        print("  (near-zero slope = flat = purely geometric offset; significant slope = frequency-dependent)")
else:
    print("  ⚠  No usable no-panel current-config data — cannot compute baseline.")

# ═════════════════════════════════════════════════════════════════════════════
print(SEPARATOR + "TEST 1a — Standing wave phase at IN probe vs OUT/IN dips/peaks")
# ═════════════════════════════════════════════════════════════════════════════
print(f"Parameters: IN probe at {X_IN} m, panel centroid at {X_PANEL_CENTRE} m")
print(f"            Δ = {DELTA:.3f} m,  d = {d} m,  R_assumed = {R_ASSUMED}")

freqs_test = sorted(meta["WaveFrequencyInput [Hz]"].dropna().unique())
freqs_test = [f for f in freqs_test if 0.6 <= f <= 1.95]

print(f"\n{'Freq':>6}  {'k':>7}  {'kd':>6}  {'λ(m)':>7}  {'2kΔ':>8}  "
      f"{'phase/π':>8}  {'SW factor':>10}  {'SW effect on OUT/IN':>20}")
print("-" * 90)

phase_data = []
for f in freqs_test:
    k  = solve_k(f)
    kd = k * d
    lam = 2 * np.pi / k
    two_k_delta = 2 * k * DELTA
    phase_norm = (two_k_delta % (2 * np.pi)) / np.pi  # in units of π, range [0, 2)
    # Standing wave amplitude factor at IN probe
    sw_factor = np.sqrt(1 + R_ASSUMED**2 + 2*R_ASSUMED*np.cos(two_k_delta))
    # OUT/IN_meas = T / sw_factor  →  effect: OUT/IN inflated if sw_factor < 1 (near node)
    #                                         OUT/IN deflated if sw_factor > 1 (near antinode)
    node_or_anti = "NODE (↑ OUT/IN)" if np.cos(two_k_delta) < 0 else "antinode (↓ OUT/IN)"
    phase_data.append({
        "freq": f, "k": k, "kd": kd, "lambda_m": lam,
        "two_k_delta": two_k_delta, "phase_norm": phase_norm,
        "sw_factor": sw_factor, "node_or_anti": node_or_anti
    })
    print(f"{f:>6.2f}  {k:>7.3f}  {kd:>6.3f}  {lam:>7.3f}  "
          f"{two_k_delta:>8.3f}  {phase_norm:>8.3f}π  "
          f"{sw_factor:>10.4f}  {node_or_anti}")

# Now pull observed OUT/IN and compare
print(f"\n{'Freq':>6}  {'phase/π':>8}  {'SW factor':>10}  {'obs no-wind':>12}  "
      f"{'obs full-wind':>14}  {'SW-corrected nw':>16}  {'SW-corrected fw':>16}")
print("-" * 100)

wave_nw = meta[
    (meta["run_category"] == "standard") &
    (meta["WindCondition"] == "no") &
    (meta["PanelCondition"] == "full") &
    (meta["in_position"] == "9373/170")
]
wave_fw = meta[
    (meta["run_category"] == "standard") &
    (meta["WindCondition"] == "full") &
    (meta["PanelCondition"] == "full") &
    (meta["in_position"] == "9373/170")
]
nw_grp = wave_nw.groupby("WaveFrequencyInput [Hz]")["OUT/IN (FFT)"].mean()
fw_grp = wave_fw.groupby("WaveFrequencyInput [Hz]")["OUT/IN (FFT)"].mean()

for pd_ in phase_data:
    f = pd_["freq"]
    sw = pd_["sw_factor"]
    pn = pd_["phase_norm"]
    obs_nw = nw_grp.get(f, np.nan)
    obs_fw = fw_grp.get(f, np.nan)
    # Corrected T = obs_OUT_IN * sw_factor  (if IN probe at node sw<1, obs is too high, correction < 1)
    corr_nw = obs_nw * sw if not np.isnan(obs_nw) else np.nan
    corr_fw = obs_fw * sw if not np.isnan(obs_fw) else np.nan
    print(f"{f:>6.2f}  {pn:>8.3f}π  {sw:>10.4f}  "
          f"{obs_nw:>12.4f}  {obs_fw:>14.4f}  "
          f"{corr_nw:>16.4f}  {corr_fw:>16.4f}")

print(f"\nNote: SW factor < 1 → IN probe near node → observed OUT/IN is INFLATED")
print(f"      SW factor > 1 → IN probe near antinode → observed OUT/IN is DEFLATED")
print(f"      Correction: T_true ≈ OUT/IN_obs × SW_factor  (approximate; ignores panel kL effects)")
print(f"\n⚠  Panel centroid at {X_PANEL_CENTRE} m is approximate.")
print(f"   If actual position differs by ±0.5 m, phase shifts by ±{2*solve_k(1.3)*0.5/np.pi:.2f}π at 1.3 Hz.")

# ═════════════════════════════════════════════════════════════════════════════
print(SEPARATOR + "TEST 2a — No-panel OUT/IN vs frequency (frequency-dependence of geometric offset)")
# ═════════════════════════════════════════════════════════════════════════════
if len(nopanel_curr):
    print("No-panel OUT/IN per frequency (no-wind, current config) — already printed in Test 6d")
    print("Key question: is the no-panel OUT/IN flat (pure geometry) or sloped (frequency-dependent)?")
    if len(grp) >= 2:
        freq_arr2 = grp.index.values.astype(float)
        outin_arr2 = grp["mean_OUT_IN"].values
        mask = ~np.isnan(outin_arr2)
        if mask.sum() >= 2:
            slope2, intercept2 = np.polyfit(freq_arr2[mask], outin_arr2[mask], 1)
            span = outin_arr2[mask].max() - outin_arr2[mask].min()
            print(f"\nNo-panel OUT/IN: mean={outin_arr2[mask].mean():.4f}, "
                  f"range=[{outin_arr2[mask].min():.4f}, {outin_arr2[mask].max():.4f}], "
                  f"peak-to-peak={span:.4f}")
            print(f"Linear trend slope: {slope2:.4f} per Hz  (R²={np.corrcoef(freq_arr2[mask], outin_arr2[mask])[0,1]**2:.3f})")
            if span > 0.05:
                print("  → SIGNIFICANT frequency variation in no-panel OUT/IN.")
                print("  → Cannot use a single constant geometric correction factor.")
            else:
                print("  → No-panel OUT/IN is approximately flat — single correction factor is acceptable.")
else:
    print("No no-panel data available. Cannot assess frequency dependence of geometric offset.")

# ═════════════════════════════════════════════════════════════════════════════
print(SEPARATOR + "TEST 4a — Far-end reflection arrival time vs analysis window")
# ═════════════════════════════════════════════════════════════════════════════
L_TANK = 25.0   # m — APPROXIMATE; verify from lab drawings
print(f"Assumed tank length: {L_TANK} m  (verify from lab notes)")
print(f"\n{'Freq':>6}  {'c_g':>7}  {'t_reflect':>11}  {'per40 window?':>14}  {'per240 window?':>15}")
print("-" * 60)

# Estimate stable window start from SNARVEI calibration (approximate)
SNARVEI_START_S = {0.65: 16.3, 0.70: 15.0, 0.80: 18.0, 0.90: 19.5,
                   1.00: 19.5, 1.10: 20.0, 1.20: 20.5, 1.30: 19.2,
                   1.40: 20.5, 1.50: 22.0, 1.60: 25.0, 1.70: 22.0,
                   1.80: 24.0, 1.90: 26.5}  # sample / 250 Hz, rough

for f in sorted(SNARVEI_START_S.keys()):
    cg   = c_group(f)
    t_r  = 2 * L_TANK / cg
    t_start = SNARVEI_START_S.get(f, 20.0)
    # per40: mstop at ~40s from recording start (approx; exact depends on run)
    # per240: mstop at ~240s
    in_per40  = "YES ⚠" if t_r < 40.0 else "no"
    in_per240 = "YES ⚠" if t_r < 240.0 else "no"
    print(f"{f:>6.2f}  {cg:>7.3f}  {t_r:>11.1f} s  {in_per40:>14}  {in_per240:>15}")

print("\nNote: far-end reflection arrives within the recording window for per40 at low freq.")
print("Whether it contaminates the ANALYSIS window depends on mstop timing vs t_reflect.")
print("For per40 runs, mstop ~ 10–30 s before end of recording; check combined_meta['mstop_sec'].")

# ═════════════════════════════════════════════════════════════════════════════
print(SEPARATOR + "SUMMARY: key numbers for assumption_audit.md")
# ═════════════════════════════════════════════════════════════════════════════
print(f"Test 6a: unexpected probe positions = {len(unexpected)}")
print(f"Test 6b: no-panel runs after adding 20260307 = {len(nopanel)}")
if len(nopanel_curr):
    outin_np = grp["mean_OUT_IN"]
    print(f"Test 6d: no-panel OUT/IN range = [{outin_np.min():.4f}, {outin_np.max():.4f}]")
else:
    print("Test 6d: no no-panel data available")

# Critical phase values for Attack 1
print("\nTest 1a: standing wave factor (R=0.20) at key frequencies:")
for pd_ in phase_data:
    f = pd_["freq"]
    if f in [0.70, 0.80, 1.00, 1.10, 1.30, 1.50, 1.70]:
        print(f"  {f:.2f} Hz: phase={pd_['phase_norm']:.3f}π, SW_factor={pd_['sw_factor']:.4f} "
              f"({pd_['node_or_anti']})")
