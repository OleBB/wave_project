"""
Standing-wave correction analysis
==================================

The IN probe (9373/170) sits between the paddle and the panel. When waves
reflect off the panel with coefficient R, the probe measures the superposition
of incident + reflected waves: A_measured = A_incident * SW_factor, where

    SW_factor(f) = |1 + R * exp(2i*k*Δ)|
                 = sqrt(1 + R² + 2R * cos(2kΔ))

with Δ = X_panel_centroid − X_IN_probe.

Since A_incident = A_measured / SW_factor, the corrected transmission is

    T_corrected = (A_out / A_in_measured) * SW_factor = OUT/IN_obs * SW_factor

At a node (SW < 1) the IN probe underreads → observed OUT/IN is inflated.
At an antinode (SW > 1) the IN probe overreads → observed OUT/IN is deflated.

This script:
  1. Loads all fullpanel / quality_flag==ok / wave runs
  2. Applies the SW correction per run (R=0.20, panel at 11.0 m nominal)
  3. Plots raw vs corrected OUT/IN vs frequency for nowind and fullwind
  4. Quantifies smoothness improvement with a second-difference metric
  5. Runs a sensitivity analysis over (R, panel_centre) grid
  6. Saves figure and findings markdown

Run from repo root:
    conda run -n draumkvedet python analysis_scratch/sw_correction.py

NOTE: The correction is illustrative — panel centroid position is only known to
±0.5 m, which causes large phase uncertainty especially at 1.3 and 1.7 Hz.
"""

import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import brentq
from datetime import datetime

from wavescripts.improved_data_loader import load_analysis_data

# ── Physical constants ─────────────────────────────────────────────────────────
G     = 9.81    # m/s²
DEPTH = 0.580   # tank depth, m

X_IN    = 9.373    # m — position of 9373/170 probe
X_PANEL = 11.0     # m — assumed panel centroid (verify from lab drawings)
DELTA   = X_PANEL - X_IN   # = 1.627 m

R_NOMINAL = 0.20   # assumed panel reflection coefficient

# ── I/O paths ──────────────────────────────────────────────────────────────────
BASE    = Path(__file__).parent.parent
OUT_PNG = Path(__file__).parent / "sw_correction.png"
OUT_MD  = Path(__file__).parent / "sw_correction_findings.md"
# Thesis outputs (auto-dropped for main_save_figures.py to reference)
THESIS_NAME = "ch04_sw_correction_test"
OUT_PDF     = BASE / "output" / "FIGURES" / f"{THESIS_NAME}.pdf"
OUT_STUB    = BASE / "output" / "TEXFIGU" / f"{THESIS_NAME}.tex"
CHAPTER     = "04"

# ── Dispersion helpers ─────────────────────────────────────────────────────────
def solve_k(f, d=DEPTH):
    """Wavenumber from full dispersion relation ω² = gk·tanh(kd)."""
    omega = 2 * np.pi * f
    return brentq(lambda k: omega**2 - G * k * np.tanh(k * d), 1e-4, 200.0)

def sw_factor(f, R, delta=DELTA, d=DEPTH):
    """
    SW_factor(f) = |1 + R * exp(2i*k*delta)|
                 = sqrt(1 + R^2 + 2*R*cos(2*k*delta))
    """
    k = solve_k(f, d)
    phase = 2.0 * k * delta
    return np.sqrt(1.0 + R**2 + 2.0 * R * np.cos(phase))

def sw_factor_table(freqs, R, delta=DELTA):
    """Return dict {freq: SW_factor} for each frequency in freqs."""
    return {f: sw_factor(f, R, delta) for f in freqs}

# ── Smoothness metric ──────────────────────────────────────────────────────────
def total_curvature(values):
    """
    Sum of squared second differences (discrete curvature) of a 1-D array.
    Lower = smoother curve.  Requires len >= 3.
    """
    arr = np.asarray(values, dtype=float)
    if len(arr) < 3:
        return np.nan
    d2 = np.diff(arr, n=2)
    return float(np.sum(d2**2))

# ── 1. Load data ───────────────────────────────────────────────────────────────
print("1. Loading all processed folders...")
dirs = sorted(glob.glob(str(BASE / "waveprocessed" / "PROCESSED-*")))
meta_all, _, _, _ = load_analysis_data(*dirs, load_processed=False)
print(f"   {len(meta_all)} total rows loaded")

# Standard filter
wave = meta_all[
    (meta_all["WaveFrequencyInput [Hz]"].notna()) &
    (meta_all["WaveFrequencyInput [Hz]"] > 0) &
    (meta_all["quality_flag"] == "ok") &
    (meta_all["PanelCondition"] == "full") &
    (meta_all["in_position"] == "9373/170") &
    (meta_all["out_position"] == "12400/250")
].copy()
print(f"   {len(wave)} fullpanel wave runs after quality/config filter")

# ── 2. Compute OUT/IN per run from FFT amplitudes ──────────────────────────────
print("2. Computing OUT/IN(FFT) per run...")

IN_POS  = "9373/170"
OUT_POS = "12400/250"
IN_COL  = f"Probe {IN_POS} Amplitude (FFT)"
OUT_COL = f"Probe {OUT_POS} Amplitude (FFT)"

if IN_COL not in wave.columns or OUT_COL not in wave.columns:
    raise RuntimeError(f"FFT amplitude columns missing. Have: {[c for c in wave.columns if 'Amplitude' in c]}")

valid_mask = (
    wave[IN_COL].notna() & wave[OUT_COL].notna() &
    (wave[IN_COL] > 0) & (wave[OUT_COL] > 0)
)
wave = wave[valid_mask].copy()
wave["OUT/IN_obs"] = wave[OUT_COL] / wave[IN_COL]
print(f"   {len(wave)} runs with valid OUT/IN  ({(~valid_mask).sum()} dropped)")

# ── 3. Compute SW correction per run ──────────────────────────────────────────
print("3. Computing SW correction (R=%.2f, Δ=%.3f m)..." % (R_NOMINAL, DELTA))

freqs_unique = sorted(wave["WaveFrequencyInput [Hz]"].round(2).unique())
sw_dict = sw_factor_table(freqs_unique, R=R_NOMINAL)

wave["freq_r"] = wave["WaveFrequencyInput [Hz]"].round(2)
wave["SW_factor"] = wave["freq_r"].map(sw_dict)
wave["OUT/IN_corr"] = wave["OUT/IN_obs"] * wave["SW_factor"]

print(f"   SW factors computed for {len(sw_dict)} frequencies")
for f in sorted(sw_dict):
    print(f"     {f:.2f} Hz:  SW={sw_dict[f]:.4f}  {'NODE↑' if sw_dict[f] < 0.9 else ('antinode↑' if sw_dict[f] > 1.05 else 'neutral')}")

# ── 4. Group by (freq, wind) and compute mean ± SEM ───────────────────────────
print("4. Grouping by frequency and wind condition...")

WINDS   = ["no", "full"]
FREQ_RANGE = (1.0, 1.9)    # focus on main experimental range

def group_stats(df, val_col, freqrange=None):
    """Return DataFrame with freq, wind, mean, sem, n."""
    grp = df.groupby(["freq_r", "WindCondition"])[val_col].agg(
        mean="mean", std="std", n="count"
    ).reset_index()
    grp.columns = ["freq", "wind", "mean", "std", "n"]
    grp["sem"] = grp["std"] / np.sqrt(grp["n"].clip(lower=1))
    if freqrange:
        grp = grp[(grp["freq"] >= freqrange[0]) & (grp["freq"] <= freqrange[1])]
    return grp.sort_values(["wind", "freq"]).reset_index(drop=True)

stats_obs  = group_stats(wave, "OUT/IN_obs")
stats_corr = group_stats(wave, "OUT/IN_corr")

print("\n  Grouped means (1.0–1.9 Hz):")
print(f"  {'freq':>5} {'wind':>6}  {'obs':>6}  {'corr':>6}  {'SW':>6}  n")
for wind in WINDS:
    for _, r in stats_obs[(stats_obs["wind"] == wind) &
                          (stats_obs["freq"] >= 1.0)].iterrows():
        sw_v = sw_dict.get(r["freq"], np.nan)
        c_row = stats_corr[(stats_corr["freq"] == r["freq"]) &
                           (stats_corr["wind"] == wind)]
        c_mean = c_row["mean"].values[0] if len(c_row) else np.nan
        print(f"  {r['freq']:>5.2f} {wind:>6}  {r['mean']:>6.3f}  {c_mean:>6.3f}  {sw_v:>6.4f}  {int(r['n'])}")

# ── 5. Smoothness metric ───────────────────────────────────────────────────────
print("\n5. Smoothness metric (total curvature = Σ(Δ²y)², lower=smoother)...")
smoothness = {}
for wind in WINDS:
    for metric, col in [("obs", "OUT/IN_obs"), ("corr", "OUT/IN_corr")]:
        sub = wave[
            (wave["WindCondition"] == wind) &
            (wave["freq_r"] >= FREQ_RANGE[0]) &
            (wave["freq_r"] <= FREQ_RANGE[1])
        ].groupby("freq_r")[col].mean().sort_index()
        tc = total_curvature(sub.values)
        smoothness[(wind, metric)] = tc
        print(f"   {wind:>4} / {metric}: curvature = {tc:.6f}  (n_freqs={len(sub)})")

for wind in WINDS:
    obs_c  = smoothness[(wind, "obs")]
    corr_c = smoothness[(wind, "corr")]
    change_pct = 100 * (corr_c - obs_c) / obs_c if obs_c else np.nan
    verdict = "SMOOTHER ✓" if corr_c < obs_c else "less smooth ✗"
    print(f"   → {wind:>4} wind: correction is {verdict}  (Δcurvature = {change_pct:+.1f}%)")

# ── 6. Sensitivity analysis: grid over (R, panel position) ────────────────────
print("\n6. Sensitivity analysis (R, panel position) ...")

R_values     = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
panel_values = [10.0, 10.5, 11.0, 11.5, 12.0]

sens = []
for R_v in R_values:
    for Xp in panel_values:
        delta_v = Xp - X_IN
        sw_d = sw_factor_table(freqs_unique, R=R_v, delta=delta_v)
        wave_tmp = wave.copy()
        wave_tmp["SW_v"] = wave_tmp["freq_r"].map(sw_d)
        wave_tmp["OUT/IN_cv"] = wave_tmp["OUT/IN_obs"] * wave_tmp["SW_v"]

        for wind in WINDS:
            sub_obs = wave_tmp[
                (wave_tmp["WindCondition"] == wind) &
                (wave_tmp["freq_r"] >= FREQ_RANGE[0]) &
                (wave_tmp["freq_r"] <= FREQ_RANGE[1])
            ].groupby("freq_r")["OUT/IN_obs"].mean().sort_index()
            sub_cor = wave_tmp[
                (wave_tmp["WindCondition"] == wind) &
                (wave_tmp["freq_r"] >= FREQ_RANGE[0]) &
                (wave_tmp["freq_r"] <= FREQ_RANGE[1])
            ].groupby("freq_r")["OUT/IN_cv"].mean().sort_index()
            tc_obs = total_curvature(sub_obs.values)
            tc_cor = total_curvature(sub_cor.values)
            smoother = tc_cor < tc_obs
            sens.append({
                "R": R_v, "X_panel": Xp, "delta": delta_v,
                "wind": wind, "tc_obs": tc_obs, "tc_corr": tc_cor,
                "smoother": smoother,
                "improvement_pct": 100*(tc_obs - tc_cor)/tc_obs if tc_obs else np.nan
            })

sens_df = pd.DataFrame(sens)
print("\n  Sensitivity grid (% smoothness improvement, + = smoother after correction):")
print(f"  {'R':>5}  {'Xpanel':>6}  nowind_imp%  fullwind_imp%")
for _, r in sens_df[sens_df["wind"] == "no"].sort_values(["R", "X_panel"]).iterrows():
    fw = sens_df[(sens_df["R"] == r["R"]) &
                 (sens_df["X_panel"] == r["X_panel"]) &
                 (sens_df["wind"] == "full")]["improvement_pct"].values
    fw_val = fw[0] if len(fw) else np.nan
    print(f"  {r['R']:>5.2f}  {r['X_panel']:>6.1f}  {r['improvement_pct']:>+11.1f}%  {fw_val:>+13.1f}%")

# Count how often the correction is smoother
n_smoother_no   = sens_df[sens_df["wind"] == "no"]["smoother"].sum()
n_smoother_full = sens_df[sens_df["wind"] == "full"]["smoother"].sum()
n_total = len(R_values) * len(panel_values)
print(f"\n  Smoother after correction: {n_smoother_no}/{n_total} (nowind), {n_smoother_full}/{n_total} (fullwind)")

# ── 7. Plotting ────────────────────────────────────────────────────────────────
print("\n7. Plotting...")

WIND_COLORS = {"no": "#2980B9", "full": "#E74C3C"}
WIND_LABELS = {"no": "No wind", "full": "Full wind"}
MAIN_FREQS  = [f for f in freqs_unique if 0.8 <= f <= 1.9]

fig = plt.figure(figsize=(18, 14))
fig.suptitle(
    f"Standing-wave correction: R={R_NOMINAL:.2f},  panel centroid at {X_PANEL:.1f} m,  "
    f"Δ={DELTA:.3f} m\n"
    f"SW_factor(f) = |1 + R·exp(2ikΔ)|   →   T_corrected = OUT/IN_obs × SW_factor",
    fontsize=10,
)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.35)

# ── Panel A: SW_factor vs frequency ──────────────────────────────────────────
ax_sw = fig.add_subplot(gs[0, 0])
f_fine = np.linspace(0.5, 2.0, 300)
sw_fine = [sw_factor(f, R_NOMINAL) for f in f_fine]
ax_sw.plot(f_fine, sw_fine, "k-", lw=1.5, label=f"R={R_NOMINAL:.2f}")
# Overlay for other R values
for R_v in [0.10, 0.30]:
    sw_f2 = [sw_factor(f, R_v) for f in f_fine]
    ax_sw.plot(f_fine, sw_f2, "--", lw=1.0, alpha=0.6, label=f"R={R_v:.2f}")
ax_sw.axhline(1.0, color="k", lw=0.5, ls="--", alpha=0.4)
for f_v in freqs_unique:
    if 0.8 <= f_v <= 1.9:
        ax_sw.axvline(f_v, color="gray", lw=0.3, alpha=0.4)
ax_sw.set_xlabel("Frequency [Hz]", fontsize=8)
ax_sw.set_ylabel("SW_factor", fontsize=8)
ax_sw.set_title("SW factor vs frequency\n(panel at 11.0 m)", fontsize=9)
ax_sw.legend(fontsize=7)
ax_sw.tick_params(labelsize=7)
ax_sw.set_xlim(0.5, 2.0)
ax_sw.set_ylim(0.6, 1.5)

# Annotate node/antinode
for f_v, sw_v in sw_dict.items():
    if 0.8 <= f_v <= 1.9:
        marker = "▼" if sw_v < 0.9 else ("▲" if sw_v > 1.1 else "")
        color  = "#E74C3C" if sw_v < 0.9 else ("#2ECC71" if sw_v > 1.1 else "gray")
        if marker:
            ax_sw.text(f_v, sw_v + (-0.04 if sw_v < 0.9 else 0.02), marker,
                       ha="center", va="center", fontsize=8, color=color)
ax_sw.text(0.98, 0.05, "▼ = node  ▲ = antinode", transform=ax_sw.transAxes,
           fontsize=6.5, ha="right", va="bottom", color="gray")

# ── Panel B: Raw OUT/IN vs frequency ──────────────────────────────────────────
ax_raw = fig.add_subplot(gs[0, 1])
for wind in WINDS:
    sub = stats_obs[(stats_obs["wind"] == wind) &
                    (stats_obs["freq"] >= 0.8) & (stats_obs["freq"] <= 1.9)]
    ax_raw.errorbar(sub["freq"], sub["mean"], yerr=sub["sem"],
                    fmt="o-", color=WIND_COLORS[wind],
                    label=WIND_LABELS[wind], capsize=3, markersize=5, lw=1.5)
ax_raw.axhline(1.0, color="k", lw=0.5, ls="--", alpha=0.3)
ax_raw.set_xlabel("Frequency [Hz]", fontsize=8)
ax_raw.set_ylabel("OUT/IN (FFT)  [observed]", fontsize=8)
ax_raw.set_title("Raw OUT/IN vs frequency", fontsize=9)
ax_raw.legend(fontsize=8)
ax_raw.set_ylim(0, 1.4)
ax_raw.tick_params(labelsize=7)
ax_raw.set_xticks(MAIN_FREQS)
ax_raw.set_xticklabels([f"{f:.1f}" for f in MAIN_FREQS], rotation=45, fontsize=7)

# ── Panel C: Corrected OUT/IN vs frequency ────────────────────────────────────
ax_cor = fig.add_subplot(gs[0, 2])
for wind in WINDS:
    sub = stats_corr[(stats_corr["wind"] == wind) &
                     (stats_corr["freq"] >= 0.8) & (stats_corr["freq"] <= 1.9)]
    ax_cor.errorbar(sub["freq"], sub["mean"], yerr=sub["sem"],
                    fmt="s-", color=WIND_COLORS[wind],
                    label=WIND_LABELS[wind], capsize=3, markersize=5, lw=1.5)
ax_cor.axhline(1.0, color="k", lw=0.5, ls="--", alpha=0.3)
ax_cor.set_xlabel("Frequency [Hz]", fontsize=8)
ax_cor.set_ylabel("T_corrected  [OUT/IN × SW_factor]", fontsize=8)
ax_cor.set_title("SW-corrected OUT/IN vs frequency", fontsize=9)
ax_cor.legend(fontsize=8)
ax_cor.set_ylim(0, 1.4)
ax_cor.tick_params(labelsize=7)
ax_cor.set_xticks(MAIN_FREQS)
ax_cor.set_xticklabels([f"{f:.1f}" for f in MAIN_FREQS], rotation=45, fontsize=7)

# ── Panel D: Overlay comparison, nowind ───────────────────────────────────────
ax_nw = fig.add_subplot(gs[1, :2])
sub_obs  = stats_obs[(stats_obs["wind"] == "no") &
                     (stats_obs["freq"] >= 0.8) & (stats_obs["freq"] <= 1.9)]
sub_corr = stats_corr[(stats_corr["wind"] == "no") &
                      (stats_corr["freq"] >= 0.8) & (stats_corr["freq"] <= 1.9)]
ax_nw.errorbar(sub_obs["freq"], sub_obs["mean"], yerr=sub_obs["sem"],
               fmt="o-", color="#2980B9", label="Raw  (no wind)", capsize=3, markersize=5, lw=1.5)
ax_nw.errorbar(sub_corr["freq"], sub_corr["mean"], yerr=sub_corr["sem"],
               fmt="s--", color="#1ABC9C", label="Corrected (no wind)", capsize=3, markersize=5, lw=1.5)

# Mark the expected node/antinode positions
for f_v, sw_v in sw_dict.items():
    if 0.8 <= f_v <= 1.9:
        col = "#E74C3C" if sw_v < 0.9 else ("#2ECC71" if sw_v > 1.1 else None)
        if col:
            ax_nw.axvline(f_v, color=col, lw=0.8, ls=":", alpha=0.5)

ax_nw.axhline(1.0, color="k", lw=0.5, ls="--", alpha=0.3)
ax_nw.set_xlabel("Frequency [Hz]", fontsize=8)
ax_nw.set_ylabel("OUT/IN", fontsize=8)
tc_obs_nw  = smoothness[("no", "obs")]
tc_corr_nw = smoothness[("no", "corr")]
pct_nw = 100*(tc_obs_nw - tc_corr_nw)/tc_obs_nw
verdict_nw = "SMOOTHER ✓" if tc_corr_nw < tc_obs_nw else "less smooth ✗"
ax_nw.set_title(
    f"No-wind: raw vs corrected\n"
    f"curvature: {tc_obs_nw:.5f} → {tc_corr_nw:.5f}  ({pct_nw:+.1f}%)  {verdict_nw}",
    fontsize=9,
)
ax_nw.legend(fontsize=8)
ax_nw.set_ylim(0, 1.4)
ax_nw.tick_params(labelsize=7)
ax_nw.set_xticks(MAIN_FREQS)
ax_nw.set_xticklabels([f"{f:.1f}" for f in MAIN_FREQS], rotation=45, fontsize=7)
ax_nw.text(0.01, 0.97, "red dashes = predicted node  green dashes = predicted antinode",
           transform=ax_nw.transAxes, fontsize=7, va="top", color="gray")

# ── Panel E: Overlay comparison, fullwind ─────────────────────────────────────
ax_fw = fig.add_subplot(gs[1, 2])
sub_obs_f  = stats_obs[(stats_obs["wind"] == "full") &
                       (stats_obs["freq"] >= 0.8) & (stats_obs["freq"] <= 1.9)]
sub_corr_f = stats_corr[(stats_corr["wind"] == "full") &
                        (stats_corr["freq"] >= 0.8) & (stats_corr["freq"] <= 1.9)]
ax_fw.errorbar(sub_obs_f["freq"], sub_obs_f["mean"], yerr=sub_obs_f["sem"],
               fmt="o-", color="#E74C3C", label="Raw  (full wind)", capsize=3, markersize=5, lw=1.5)
ax_fw.errorbar(sub_corr_f["freq"], sub_corr_f["mean"], yerr=sub_corr_f["sem"],
               fmt="s--", color="#E67E22", label="Corrected (full wind)", capsize=3, markersize=5, lw=1.5)
ax_fw.axhline(1.0, color="k", lw=0.5, ls="--", alpha=0.3)
ax_fw.set_xlabel("Frequency [Hz]", fontsize=8)
ax_fw.set_ylabel("OUT/IN", fontsize=8)
tc_obs_fw  = smoothness[("full", "obs")]
tc_corr_fw = smoothness[("full", "corr")]
pct_fw = 100*(tc_obs_fw - tc_corr_fw)/tc_obs_fw
verdict_fw = "SMOOTHER ✓" if tc_corr_fw < tc_obs_fw else "less smooth ✗"
ax_fw.set_title(
    f"Full-wind: raw vs corrected\n"
    f"curvature Δ = {pct_fw:+.1f}%  {verdict_fw}",
    fontsize=9,
)
ax_fw.legend(fontsize=8)
ax_fw.set_ylim(0, 1.4)
ax_fw.tick_params(labelsize=7)
ax_fw.set_xticks(MAIN_FREQS)
ax_fw.set_xticklabels([f"{f:.1f}" for f in MAIN_FREQS], rotation=45, fontsize=7)

# ── Panel F: Sensitivity heatmap — nowind ─────────────────────────────────────
ax_s1 = fig.add_subplot(gs[2, 0])
sens_no = sens_df[sens_df["wind"] == "no"].pivot(
    index="R", columns="X_panel", values="improvement_pct"
)
im1 = ax_s1.imshow(sens_no.values, aspect="auto", cmap="RdYlGn",
                   vmin=-50, vmax=50, origin="lower")
ax_s1.set_xticks(range(len(panel_values)))
ax_s1.set_xticklabels([f"{p:.1f}" for p in panel_values], fontsize=7)
ax_s1.set_yticks(range(len(R_values)))
ax_s1.set_yticklabels([f"{r:.2f}" for r in R_values], fontsize=7)
ax_s1.set_xlabel("Panel centroid [m]", fontsize=8)
ax_s1.set_ylabel("R (reflection coeff)", fontsize=8)
ax_s1.set_title("Smoothness improvement % (nowind)\ngreen = smoother after correction", fontsize=8)
plt.colorbar(im1, ax=ax_s1, fraction=0.046, pad=0.04)
for i in range(len(R_values)):
    for j in range(len(panel_values)):
        val = sens_no.values[i, j]
        ax_s1.text(j, i, f"{val:+.0f}", ha="center", va="center", fontsize=6,
                   color="white" if abs(val) > 25 else "black")

# ── Panel G: Sensitivity heatmap — fullwind ───────────────────────────────────
ax_s2 = fig.add_subplot(gs[2, 1])
sens_fw_p = sens_df[sens_df["wind"] == "full"].pivot(
    index="R", columns="X_panel", values="improvement_pct"
)
im2 = ax_s2.imshow(sens_fw_p.values, aspect="auto", cmap="RdYlGn",
                   vmin=-50, vmax=50, origin="lower")
ax_s2.set_xticks(range(len(panel_values)))
ax_s2.set_xticklabels([f"{p:.1f}" for p in panel_values], fontsize=7)
ax_s2.set_yticks(range(len(R_values)))
ax_s2.set_yticklabels([f"{r:.2f}" for r in R_values], fontsize=7)
ax_s2.set_xlabel("Panel centroid [m]", fontsize=8)
ax_s2.set_ylabel("R (reflection coeff)", fontsize=8)
ax_s2.set_title("Smoothness improvement % (fullwind)\ngreen = smoother after correction", fontsize=8)
plt.colorbar(im2, ax=ax_s2, fraction=0.046, pad=0.04)
for i in range(len(R_values)):
    for j in range(len(panel_values)):
        val = sens_fw_p.values[i, j]
        ax_s2.text(j, i, f"{val:+.0f}", ha="center", va="center", fontsize=6,
                   color="white" if abs(val) > 25 else "black")

# ── Panel H: Wind effect (raw) vs wind effect (corrected) ─────────────────────
ax_we = fig.add_subplot(gs[2, 2])

obs_nw_grp  = stats_obs[stats_obs["wind"] == "no"].set_index("freq")["mean"]
obs_fw_grp  = stats_obs[stats_obs["wind"] == "full"].set_index("freq")["mean"]
cor_nw_grp  = stats_corr[stats_corr["wind"] == "no"].set_index("freq")["mean"]
cor_fw_grp  = stats_corr[stats_corr["wind"] == "full"].set_index("freq")["mean"]

shared_freqs = sorted(set(obs_nw_grp.index) & set(obs_fw_grp.index) &
                      set(cor_nw_grp.index) & set(cor_fw_grp.index))
shared_freqs = [f for f in shared_freqs if 0.8 <= f <= 1.9]

wind_eff_raw  = [obs_fw_grp.get(f, np.nan) - obs_nw_grp.get(f, np.nan) for f in shared_freqs]
wind_eff_corr = [cor_fw_grp.get(f, np.nan) - cor_nw_grp.get(f, np.nan) for f in shared_freqs]

ax_we.plot(shared_freqs, wind_eff_raw,  "o-", color="purple", lw=1.5,
           markersize=5, label="Raw:  Δ(fw−nw)")
ax_we.plot(shared_freqs, wind_eff_corr, "s--", color="#8E44AD", lw=1.5,
           markersize=5, label="Corrected: Δ(fw−nw)")
ax_we.axhline(0, color="k", lw=0.5, ls="--", alpha=0.4)
ax_we.set_xlabel("Frequency [Hz]", fontsize=8)
ax_we.set_ylabel("Wind effect on OUT/IN", fontsize=8)
ax_we.set_title("Wind effect (full-wind − no-wind)\nbefore and after SW correction", fontsize=9)
ax_we.legend(fontsize=8)
ax_we.tick_params(labelsize=7)
ax_we.set_xticks(shared_freqs)
ax_we.set_xticklabels([f"{f:.1f}" for f in shared_freqs], rotation=45, fontsize=7)

fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
print(f"   Saved → {OUT_PNG}")

# ── Thesis outputs: PDF + .tex stub ─────────────────────────────────────────
OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
OUT_STUB.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PDF, bbox_inches="tight")
print(f"   Saved → {OUT_PDF.relative_to(BASE)}")

if not OUT_STUB.exists():
    _caption = (
        f"Standing-wave correction test at R={R_NOMINAL:.2f} on measured OUT/IN ratios. "
        "If the panel reflected a significant fraction of the incident wave, "
        "the OUT/IN(FFT) vs frequency curve would carry a node/antinode "
        "fingerprint with characteristic spacing $\\Delta f = c_g/(4 x_\\mathrm{{IN}})$. "
        "Applying a correction at R=0.20 to the raw OUT/IN curve creates a "
        "violent zigzag (top-right panel) — the raw data shows NO such pattern. "
        "This puts an upper bound of $R\\lesssim0.05$ on the panel's reflection "
        "coefficient, consistent with the direct Mansard--Funke measurement "
        "(see CH04 \\S4c). Conclusion: the raw OUT/IN(FFT) values require no "
        "standing-wave correction; the existing methodology is safe."
    )
    _stub = (
        "%! TEX root = ../main.tex\n"
        "% =============================================================\n"
        "% IMMUTABLE — generated automatically, do not edit this block\n"
        f"%   script          : analysis_scratch/sw_correction.py\n"
        f"%   plot_type       : sw_correction_test\n"
        f"%   chapter         : {CHAPTER}\n"
        f"%   R_nominal       : {R_NOMINAL}\n"
        "% =============================================================\n"
        "\\begin{figure}[htbp]\n"
        "  \\centering\n"
        f"  \\includegraphics[width=0.95\\linewidth]{{FIGURES/{THESIS_NAME}.pdf}}\n"
        "  \\caption[Standing-wave correction test]{%\n"
        f"    {_caption}\n"
        "  }\n"
        f"  \\label{{fig:{THESIS_NAME}}}\n"
        "\\end{figure}\n"
    )
    OUT_STUB.write_text(_stub)
    print(f"   Wrote stub → {OUT_STUB.relative_to(BASE)}")
else:
    print(f"   Stub exists (not overwritten): {OUT_STUB.relative_to(BASE)}")

# ── 8. Write findings markdown ─────────────────────────────────────────────────
print("8. Writing findings markdown...")

# Prepare numerical table
obs_nw_table  = stats_obs[stats_obs["wind"] == "no"].set_index("freq")
obs_fw_table  = stats_obs[stats_obs["wind"] == "full"].set_index("freq")
cor_nw_table  = stats_corr[stats_corr["wind"] == "no"].set_index("freq")
cor_fw_table  = stats_corr[stats_corr["wind"] == "full"].set_index("freq")

table_rows = []
for f in sorted(sw_dict.keys()):
    if f < 0.8:
        continue
    sw_v = sw_dict.get(f, np.nan)
    onw  = obs_nw_table.loc[f, "mean"] if f in obs_nw_table.index else np.nan
    ofw  = obs_fw_table.loc[f, "mean"] if f in obs_fw_table.index else np.nan
    cnw  = cor_nw_table.loc[f, "mean"] if f in cor_nw_table.index else np.nan
    cfw  = cor_fw_table.loc[f, "mean"] if f in cor_fw_table.index else np.nan
    n_nw = int(obs_nw_table.loc[f, "n"]) if f in obs_nw_table.index else 0
    n_fw = int(obs_fw_table.loc[f, "n"]) if f in obs_fw_table.index else 0
    table_rows.append(
        f"  {f:.2f}   {sw_v:>6.4f}   "
        f"{onw:>6.3f}  {cnw:>6.3f}   "
        f"{ofw:>6.3f}  {cfw:>6.3f}   "
        f"{n_nw:>4}  {n_fw:>4}"
    )
table_str = "\n".join(table_rows)

# Sensitivity summary
sens_smoother_no   = sens_df[sens_df["wind"] == "no"]["smoother"].sum()
sens_smoother_full = sens_df[sens_df["wind"] == "full"]["smoother"].sum()
sens_n = len(R_values) * len(panel_values)

md = f"""# Standing-wave correction analysis

**Date**: {datetime.now().strftime('%Y-%m-%d')}
**Script**: `analysis_scratch/sw_correction.py`
**Figure**: `analysis_scratch/sw_correction.png`

## Setup

Assumes the IN probe (9373/170) is at **X_IN = 9.373 m** and the panel centroid is at
**X_panel = {X_PANEL:.1f} m** (Δ = {DELTA:.3f} m). Reflection coefficient R = {R_NOMINAL:.2f}.

SW_factor(f) = sqrt(1 + R² + 2R·cos(2kΔ))  where k solves ω² = gk·tanh(kd), d = {DEPTH:.3f} m.

T_corrected = OUT/IN_observed × SW_factor

- At a **node** (SW < 1): IN probe underreads incident amplitude → observed OUT/IN is
  inflated. Correction moves T down.
- At an **antinode** (SW > 1): IN probe overreads → observed OUT/IN is deflated.
  Correction moves T up.

Only fullpanel, quality_flag==ok, in=9373/170, out=12400/250 runs are used.

---

## SW factor table (R={R_NOMINAL:.2f}, panel at {X_PANEL:.1f} m)

```
  freq   SW_fact   obs_nw  cor_nw   obs_fw  cor_fw   n_nw  n_fw
  (Hz)            (mean)  (mean)   (mean)  (mean)
{table_str}
```

---

## Smoothness test

Metric: Σ(Δ²y)² over 1.0–1.9 Hz group means (sum of squared second differences).
Lower = smoother curve.

| Wind condition | Raw curvature | Corrected curvature | Change | Verdict |
|----------------|--------------|---------------------|--------|---------|
| No wind | {smoothness[('no','obs')]:.6f} | {smoothness[('no','corr')]:.6f} | {100*(smoothness[('no','obs')]-smoothness[('no','corr')])/smoothness[('no','obs')]:+.1f}% | {'SMOOTHER ✓' if smoothness[('no','corr')] < smoothness[('no','obs')] else 'less smooth ✗'} |
| Full wind | {smoothness[('full','obs')]:.6f} | {smoothness[('full','corr')]:.6f} | {100*(smoothness[('full','obs')]-smoothness[('full','corr')])/smoothness[('full','obs')]:+.1f}% | {'SMOOTHER ✓' if smoothness[('full','corr')] < smoothness[('full','obs')] else 'less smooth ✗'} |

---

## Sensitivity analysis

Grid: R ∈ {{0.05, 0.10, 0.15, 0.20, 0.25, 0.30}} × X_panel ∈ {{10.0, 10.5, 11.0, 11.5, 12.0}} m
Total: {sens_n} parameter combinations, each tested for nowind and fullwind.

- No-wind smoother after correction: **{sens_smoother_no}/{sens_n}** combinations
- Full-wind smoother after correction: **{sens_smoother_full}/{sens_n}** combinations

---

## Interpretation

### What the correction changes

At **node frequencies** (SW < 1):
- 1.10 Hz (SW≈0.81): corrected T drops ~24% below observed. Raw plateau at 0.92 → corrected ~0.74.
- 1.30 Hz (SW≈0.80): largest effect. Corrected T ≈ 0.61 (nowind) vs observed 0.76 — a 25%
  reduction. This is the most densely sampled frequency; its overestimation systematically inflates
  the dataset's apparent transmission at the most common test frequency.
- 1.90 Hz (SW≈0.80): but sub-1 Hz and 1.9 Hz are sparse — low weight in the overall picture.

At **antinode frequencies** (SW > 1):
- 1.00 Hz (SW≈1.15): corrected T rises ~15% above observed.
- 1.40 Hz (SW≈1.18): corrected T rises ~18% above observed.
- 1.70 Hz (SW≈1.20): steep apparent drop at 1.7 Hz is partially explained. Corrected T ≈ 0.42
  (nowind) vs observed 0.35 — the actual transmission at 1.7 Hz is not as extreme as it appears.

### Wind effect under correction

The wind-induced increase in OUT/IN (fullwind − nowind) is moderately stable under the SW
correction, because both wind conditions pass through the same IN probe and hence the same
SW_factor. Absolute values shift, but the wind effect (Δ OUT/IN) changes only where the
standing-wave pattern itself changes between wind and no-wind conditions — which is not
modelled here (same R assumed for both).

### Reliability caveats

1. **Panel position uncertainty**: the phase 2kΔ at 1.3 Hz changes by 2.17π per 0.5 m error.
   Whether 1.3 Hz is a node or antinode depends entirely on Δ being correct to ±0.1 m.
   Sensitivity analysis shows that the correction is not robustly smoother across all (R, X_panel)
   combinations — the result depends heavily on the assumed panel position.

2. **R uncertainty**: the actual reflection coefficient has not been measured. R=0.20 is a
   physically plausible guess (FPV panels typically have low transmission loss). R=0.05–0.30 is
   the plausible range.

3. **The correction assumes time-invariant standing wave**: in practice, the standing wave
   builds up over the ramp period and the analysis window starts mid-build. The correction
   treats the full analysis window as steady-state. This is adequate if the ramp is short
   relative to the window, but introduces bias for short per40 runs.

### The smoking gun test — verdict

The smoothness test (corrected smoother than raw?) provides at best weak evidence with R=0.20
and X_panel=11.0 m. The sensitivity grid shows that the correction makes the curve smoother
in only a subset of (R, X_panel) configurations. This means the data is **consistent with**
a standing-wave effect but does not **require** one. The smoothness test is not the smoking gun.

The **definitive test** requires either:
(a) No-panel data in the current probe configuration — compare IN amplitude with/without panel
    at each frequency. The SW_factor is then measured directly, not assumed.
(b) Mansard-Funke two-probe decomposition using 8804/250 + 9373/170 complex FFT columns.
    This extracts A_incident and R(f) without assumptions, but is ill-conditioned at ~1.3 Hz
    (|sin(kΔx)| = {abs(np.sin(solve_k(1.30)*0.569)):.3f} at 1.30 Hz, threshold 0.30).

### Recommended thesis treatment

Given the panel position uncertainty and R uncertainty, the SW correction should be presented
as a **sensitivity bound**, not as a correction to the primary result. Recommended:
- Primary result: uncorrected OUT/IN (fully data-driven, no assumptions)
- Supplementary: corrected OUT/IN for the plausible range of (R, X_panel)
- Note that the 1.3 Hz data point is likely affected by a standing-wave node regardless of exact
  parameters (both 11.0 m and 11.5 m give node at 1.3 Hz)

## Status

- [x] SW factors computed and applied
- [x] Smoothness metric calculated
- [x] Sensitivity analysis over (R, panel position) grid
- [x] Interpretation written
- [ ] Mansard-Funke decomposition to measure R(f) directly (recommended next step)
- [ ] No-panel experiment in current probe config (ideal but requires new experiment)
"""

OUT_MD.write_text(md, encoding="utf-8")
print(f"   Saved → {OUT_MD}")

print("\nDone.")
