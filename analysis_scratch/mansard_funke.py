"""
Mansard-Funke two-probe reflection analysis
============================================

Uses 8804/250 (upstream) and 9373/170 (IN probe) to decompose the wave field
into incident (A) and reflected (B) components at each frequency.

Method (Mansard & Funke 1980):
  At position x, the surface elevation is:
    η̂(x) = A·exp(ikx) + B·exp(−ikx)

  Given complex FFT amplitudes at x₁ = 8804 mm and x₂ = 9373 mm:
    A = [η̂(x₁)·exp(−ikx₂) − η̂(x₂)·exp(−ikx₁)] / [−2i·sin(kΔx)]
    B = [η̂(x₂)·exp(ikx₁) − η̂(x₁)·exp(ikx₂)] / [−2i·sin(kΔx)]
    R = |B| / |A|   (reflection coefficient)

  Ill-conditioned when |sin(kΔx)| < 0.30 (denominator near zero).

IMPORTANT notes about data reliability:
  - 0.2 V: primary result — best SNR, fewest reconstruction issues
  - 0.1 V: include but flag as lower-SNR
  - 0.3 V: potentially misleading; signal can be non-linear or anomalous
    at high freq. Show separately with explicit uncertainty warning.
  - Both probes must use the SAME analysis window (same start/end samples)
    to preserve the phase relationship needed for decomposition.
  - Fullwind runs at IN probe (9373/170) have wind-wave contamination at
    the paddle frequency — this inflates the apparent amplitude and biases
    the phase estimate. MF under full wind is less reliable than no-wind.

Run from repo root (slow — loads processed time series):
    conda run -n draumkvedet python analysis_scratch/mansard_funke.py

Output:
    analysis_scratch/mansard_funke.png
    analysis_scratch/mansard_funke_findings.md
"""

import sys, warnings, glob
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import brentq
from datetime import datetime

from wavescripts.improved_data_loader import load_analysis_data, load_processed_dfs

# ── Physical constants ────────────────────────────────────────────────────────
G     = 9.81    # m/s²
DEPTH = 0.580   # tank depth, m
FS    = 250.0   # sampling rate, Hz

X_P1  = 8.804   # m — 8804/250 probe (upstream)
X_P2  = 9.373   # m — 9373/170 probe (IN probe, current config)
DELTA = X_P2 - X_P1   # = 0.569 m

ILL_COND_THRESHOLD = 0.30  # |sin(kΔ)| below this → flag as ill-conditioned

FFT_WINDOW_HZ = 0.10   # ±0.05 Hz around paddle freq for bin search

# Amplitude tiers
AMP_PRIMARY  = 0.2    # most reliable
AMP_LOW_SNR  = 0.1    # lower SNR, include with flag
AMP_WARN     = 0.3    # non-linear risk at high freq, show with warning

BASE    = Path(__file__).parent.parent
OUT_PNG = Path(__file__).parent / "mansard_funke.png"
OUT_MD  = Path(__file__).parent / "mansard_funke_findings.md"

# ── Helpers ───────────────────────────────────────────────────────────────────
def solve_k(f, d=DEPTH):
    """Wavenumber from full dispersion relation ω² = gk·tanh(kd)."""
    omega = 2 * np.pi * f
    return brentq(lambda k: omega**2 - G * k * np.tanh(k * d), 1e-4, 200.0)

def mf_decompose(z1, z2, k, x1=X_P1, x2=X_P2):
    """
    Mansard-Funke decomposition.
    z1, z2: complex FFT amplitudes (same units, same time reference)
    Returns (A_incident, B_reflected, R, sin_kDelta, well_conditioned)
    """
    delta = x2 - x1
    sin_kd = np.sin(k * delta)

    if abs(sin_kd) < ILL_COND_THRESHOLD:
        return np.nan, np.nan, np.nan, sin_kd, False

    # Python FFT convention: positive-freq bin of a rightward wave η=A·cos(kx−ωt)
    # gives z ∝ exp(−ikx).  So z = A_inc·exp(−ikx) + B_ref·exp(+ikx).
    # Solving: A_inc = (z1·exp(+ikx2) − z2·exp(+ikx1)) / (2i·sin(kΔ))
    #          B_ref = (z2·exp(−ikx1) − z1·exp(−ikx2)) / (2i·sin(kΔ))
    denom = 2j * sin_kd
    A = (z1 * np.exp(+1j*k*x2) - z2 * np.exp(+1j*k*x1)) / denom
    B = (z2 * np.exp(-1j*k*x1) - z1 * np.exp(-1j*k*x2)) / denom

    A_amp = abs(A)
    B_amp = abs(B)
    R = B_amp / A_amp if A_amp > 0 else np.nan
    return A_amp, B_amp, R, sin_kd, True

# ── 1. Load metadata + processed dirs ─────────────────────────────────────────
print("1. Loading metadata...")
dirs = sorted(glob.glob(str(BASE / "waveprocessed" / "PROCESSED-*")))
meta_all, _, _, _ = load_analysis_data(*dirs, load_processed=False)

# Filter to the current probe config (in=9373/170) where MF is applicable
wave = meta_all[
    meta_all["WaveFrequencyInput [Hz]"].notna() &
    (meta_all["WaveFrequencyInput [Hz]"] > 0) &
    (meta_all["quality_flag"] == "ok") &
    (meta_all["PanelCondition"] == "full") &
    (meta_all["in_position"] == "9373/170") &
    (meta_all["out_position"] == "12400/250")
].copy()
print(f"   {len(wave)} fullpanel ok wave runs with in=9373/170")
print(f"   Mooring distribution: {dict(wave['Mooring'].value_counts())}")

# Check which runs have start/end for BOTH probes
for pos in ["8804/250", "9373/170"]:
    start_col = f"Computed Probe {pos} start"
    end_col   = f"Computed Probe {pos} end"
    if start_col not in wave.columns:
        raise RuntimeError(f"Missing column: {start_col}")
    n_ok = (wave[start_col].notna() & wave[end_col].notna()).sum()
    print(f"   {pos}: {n_ok}/{len(wave)} rows have start/end")

both_ok = (
    wave["Computed Probe 8804/250 start"].notna() &
    wave["Computed Probe 8804/250 end"].notna() &
    wave["Computed Probe 9373/170 start"].notna() &
    wave["Computed Probe 9373/170 end"].notna()
)
wave = wave[both_ok].copy()
print(f"   {len(wave)} runs have start/end for BOTH probes")

# ── 2. Load processed time series ─────────────────────────────────────────────
print("2. Loading processed time series (load_processed=True, ~20-30 s)...")
proc_dfs = load_processed_dfs(*dirs)
print(f"   {len(proc_dfs)} processed DataFrames loaded")

# ── 3. Compute MF decomposition per run ───────────────────────────────────────
print("3. Computing MF decomposition...")

results = []

for i, (_, row) in enumerate(wave.iterrows()):
    path   = row["path"]
    f_pad  = row["WaveFrequencyInput [Hz]"]
    amp    = row["WaveAmplitudeInput [Volt]"]
    wind   = row["WindCondition"]
    moor   = row["Mooring"]
    freq_r = round(f_pad, 2)

    if path not in proc_dfs:
        continue
    df = proc_dfs[path]

    # Determine analysis window: intersection of both probe windows
    s1 = int(row.get("Computed Probe 8804/250 start", np.nan))
    e1 = int(row.get("Computed Probe 8804/250 end",   np.nan))
    s2 = int(row.get("Computed Probe 9373/170 start", np.nan))
    e2 = int(row.get("Computed Probe 9373/170 end",   np.nan))

    # Shared window: start at the LATER probe start, end at EARLIER probe end
    s_shared = max(s1, s2)
    e_shared = min(e1, e2)

    if e_shared - s_shared < int(2 * FS / f_pad):  # less than 2 wave periods
        continue

    # Extract signals. Use _interp if available (reconstructed), else raw.
    def get_signal(df, pos, s, e):
        interp_col = f"eta_{pos}_interp"
        raw_col    = f"eta_{pos}"
        col = interp_col if interp_col in df.columns else raw_col
        if col not in df.columns:
            return None
        sig = df[col].iloc[s:e+1].to_numpy(dtype=float)
        # Replace NaN with linear interpolation if possible
        nan_mask = np.isnan(sig)
        if nan_mask.all():
            return None
        if nan_mask.any():
            frac_nan = nan_mask.sum() / len(sig)
            if frac_nan > 0.05:  # >5% NaN in shared window → skip
                return None
            indices = np.arange(len(sig))
            sig = np.interp(indices, indices[~nan_mask], sig[~nan_mask])
        return sig

    sig1 = get_signal(df, "8804/250", s_shared, e_shared)
    sig2 = get_signal(df, "9373/170", s_shared, e_shared)

    if sig1 is None or sig2 is None:
        continue
    if len(sig1) != len(sig2):
        # Trim to same length
        n = min(len(sig1), len(sig2))
        sig1, sig2 = sig1[:n], sig2[:n]

    N = len(sig1)
    if N < int(2 * FS / f_pad):  # less than 2 periods
        continue

    # Compute FFT using the SAME window for both
    fft1 = np.fft.fft(sig1)
    fft2 = np.fft.fft(sig2)
    freqs = np.fft.fftfreq(N, d=1.0/FS)

    # Find nearest positive frequency bin within FFT_WINDOW_HZ of target
    pos_mask = freqs > 0
    pos_freqs = freqs[pos_mask]
    window_mask = np.abs(pos_freqs - f_pad) <= FFT_WINDOW_HZ
    if not window_mask.any():
        continue
    # Nearest bin within window
    nearest_idx = np.argmin(np.abs(pos_freqs[window_mask] - f_pad))
    global_idx  = np.where(pos_mask)[0][window_mask][nearest_idx]

    z1 = fft1[global_idx]   # complex FFT at paddle freq for 8804/250
    z2 = fft2[global_idx]   # complex FFT at paddle freq for 9373/170

    # Solve k from full dispersion
    try:
        k = solve_k(f_pad)
    except Exception:
        continue

    # MF decomposition
    A_amp, B_amp, R, sin_kd, well = mf_decompose(z1, z2, k)

    # Physical amplitude at 9373/170 from FFT (for reference)
    # Factor 2/N for one-sided, but we just need ratios — use abs(fft2[idx])
    a_in_raw_fft = 2 * abs(z2) / N  # mm (signal is in mm from processed_dfs)

    results.append({
        "freq": freq_r, "amp": amp, "wind": wind, "mooring": moor,
        "R": R, "A_amp": A_amp, "B_amp": B_amp,
        "sin_kd": abs(sin_kd), "well_cond": well,
        "a_in_fft_mm": a_in_raw_fft,
        "k": k, "path": path,
    })

    if (i + 1) % 50 == 0:
        print(f"   {i+1}/{len(wave)} done...")

print(f"   {len(results)} runs successfully decomposed")
res = pd.DataFrame(results)

if res.empty:
    print("No valid MF results — check that processed time series are available.")
    sys.exit(1)

# ── 4. Quality flags ──────────────────────────────────────────────────────────
print("4. Applying quality flags...")

res["amp_tier"] = "0.2V (primary)"
res.loc[res["amp"] == AMP_LOW_SNR, "amp_tier"] = "0.1V (low SNR)"
res.loc[res["amp"] == AMP_WARN,    "amp_tier"] = "0.3V (nonlinear risk)"

# Ill-conditioning flag
n_illcond = (~res["well_cond"]).sum()
print(f"   Ill-conditioned (|sin(kΔ)| < {ILL_COND_THRESHOLD}): {n_illcond}/{len(res)}")

# Print the sin(kΔ) pattern across frequencies
print("\n   sin(kΔ) by frequency (Δ = 0.569 m):")
for f in sorted(res["freq"].unique()):
    k_v = solve_k(f)
    skd = abs(np.sin(k_v * DELTA))
    cond = "OK" if skd >= ILL_COND_THRESHOLD else "ILL-CONDITIONED ⚠"
    print(f"     {f:.2f} Hz:  k={k_v:.3f} rad/m  kΔ={k_v*DELTA:.3f} rad  "
          f"|sin(kΔ)|={skd:.3f}  {cond}")

# ── 5. Summary statistics ─────────────────────────────────────────────────────
print("\n5. Summary by (freq, mooring, wind, amp_tier)...")

# Primary result: 0.2V, well-conditioned
primary = res[(res["amp"] == AMP_PRIMARY) & res["well_cond"]].copy()

grp = primary.groupby(["freq", "mooring", "wind"])["R"].agg(
    mean="mean", std="std", n="count", median="median"
).reset_index()
grp["sem"] = grp["std"] / np.sqrt(grp["n"].clip(lower=1))

print(f"\n   R (0.2V, well-conditioned) — {len(primary)} runs:")
print(f"   {'freq':>5} {'mooring':>20} {'wind':>6}  {'mean R':>8} {'std R':>8}  n")
for _, r in grp.sort_values(["mooring", "freq", "wind"]).iterrows():
    print(f"   {r['freq']:>5.2f} {r['mooring']:>20} {r['wind']:>6}  "
          f"{r['mean']:>8.4f} {r['std']:>8.4f}  {int(r['n'])}")

# All amplitudes summary
all_grp = res[res["well_cond"]].groupby(["freq", "mooring", "wind", "amp"])["R"].agg(
    mean="mean", std="std", n="count"
).reset_index()
print(f"\n   Total well-conditioned runs: {res['well_cond'].sum()}")
print(f"   R overall:  mean={res[res['well_cond']]['R'].mean():.4f}  "
      f"median={res[res['well_cond']]['R'].median():.4f}  "
      f"max={res[res['well_cond']]['R'].max():.4f}")

# ── 6. Plotting ───────────────────────────────────────────────────────────────
print("6. Plotting...")

MOOR_COLORS = {
    "above_50":          "#E74C3C",
    "below_90_loose230": "#2980B9",
    "below_90_loose300": "#27AE60",
}
MOOR_LABELS = {
    "above_50":          "above_50 (5 cm up, stiff)",
    "below_90_loose230": "below_90_loose230 (9 cm below, 23 cm line)",
    "below_90_loose300": "below_90_loose300 (9 cm below, 30 cm line)",
}
WIND_LS  = {"no": "-", "full": "--"}
AMP_MARKERS = {AMP_LOW_SNR: "^", AMP_PRIMARY: "o", AMP_WARN: "s"}
AMP_ALPHA   = {AMP_LOW_SNR: 0.6,  AMP_PRIMARY: 1.0, AMP_WARN:  0.5}

fig = plt.figure(figsize=(18, 14))
fig.suptitle(
    "Mansard-Funke reflection analysis  ·  8804/250 (upstream) + 9373/170 (IN probe)\n"
    f"Δx = {DELTA*1000:.0f} mm  ·  ill-cond threshold |sin(kΔ)| < {ILL_COND_THRESHOLD}  "
    f"·  {len(res)} decompositions from {len(wave)} candidate runs",
    fontsize=10,
)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.35)

freq_range = sorted(res["freq"].unique())
MAIN_FREQS = [f for f in freq_range if 0.9 <= f <= 1.9]

# ── Panel A: sin(kΔ) vs frequency — conditioning map ────────────────────────
ax_cond = fig.add_subplot(gs[0, 0])
f_fine = np.linspace(0.6, 2.0, 400)
skd_fine = [abs(np.sin(solve_k(f) * DELTA)) for f in f_fine]
ax_cond.plot(f_fine, skd_fine, "k-", lw=1.5)
ax_cond.axhline(ILL_COND_THRESHOLD, color="#E74C3C", lw=1.5, ls="--",
                label=f"threshold = {ILL_COND_THRESHOLD}")
ax_cond.fill_between(f_fine, 0, ILL_COND_THRESHOLD, alpha=0.10, color="#E74C3C",
                     label="ill-conditioned zone")
ax_cond.set_xlabel("Frequency [Hz]", fontsize=8)
ax_cond.set_ylabel("|sin(kΔ)|", fontsize=8)
ax_cond.set_title("MF conditioning\n|sin(kΔx)|  (Δx = 569 mm)", fontsize=9)
ax_cond.legend(fontsize=7)
ax_cond.tick_params(labelsize=7)
ax_cond.set_xlim(0.6, 2.0)
ax_cond.set_ylim(0, 1.05)

# Mark actual data frequencies
for f in freq_range:
    k_v = solve_k(f)
    skd = abs(np.sin(k_v * DELTA))
    col = "#E74C3C" if skd < ILL_COND_THRESHOLD else "#27AE60"
    ax_cond.axvline(f, color=col, lw=0.5, alpha=0.5)

# ── Panel B: R vs frequency — PRIMARY (0.2V), nowind, per mooring ────────────
ax_R_nw = fig.add_subplot(gs[0, 1])
for moor in MOOR_COLORS:
    sub = grp[(grp["mooring"] == moor) & (grp["wind"] == "no")]
    if sub.empty:
        continue
    ax_R_nw.errorbar(sub["freq"], sub["mean"], yerr=sub["sem"],
                     fmt="o-", color=MOOR_COLORS[moor],
                     label=MOOR_LABELS.get(moor, moor),
                     capsize=3, markersize=5, lw=1.5)
ax_R_nw.axhline(0.20, color="k", lw=0.8, ls="--", alpha=0.4, label="R=0.20 assumed")
ax_R_nw.axhline(0.05, color="k", lw=0.5, ls=":", alpha=0.4, label="R=0.05 bound")
ax_R_nw.set_xlabel("Frequency [Hz]", fontsize=8)
ax_R_nw.set_ylabel("R = |B|/|A|", fontsize=8)
ax_R_nw.set_title("Reflection coefficient R\n0.2V, no-wind, by mooring", fontsize=9)
ax_R_nw.legend(fontsize=6.5)
ax_R_nw.set_ylim(0, 0.5)
ax_R_nw.tick_params(labelsize=7)
ax_R_nw.set_xticks(MAIN_FREQS)
ax_R_nw.set_xticklabels([f"{f:.1f}" for f in MAIN_FREQS], rotation=45, fontsize=7)

# ── Panel C: R vs frequency — PRIMARY (0.2V), fullwind, per mooring ──────────
ax_R_fw = fig.add_subplot(gs[0, 2])
for moor in MOOR_COLORS:
    sub = grp[(grp["mooring"] == moor) & (grp["wind"] == "full")]
    if sub.empty:
        continue
    ax_R_fw.errorbar(sub["freq"], sub["mean"], yerr=sub["sem"],
                     fmt="s--", color=MOOR_COLORS[moor],
                     label=MOOR_LABELS.get(moor, moor),
                     capsize=3, markersize=5, lw=1.5)
ax_R_fw.axhline(0.20, color="k", lw=0.8, ls="--", alpha=0.4)
ax_R_fw.axhline(0.05, color="k", lw=0.5, ls=":", alpha=0.4)
ax_R_fw.set_xlabel("Frequency [Hz]", fontsize=8)
ax_R_fw.set_ylabel("R = |B|/|A|", fontsize=8)
ax_R_fw.set_title("Reflection coefficient R\n0.2V, full-wind, by mooring\n"
                  "⚠ wind biases IN probe phase", fontsize=9)
ax_R_fw.legend(fontsize=6.5)
ax_R_fw.set_ylim(0, 0.5)
ax_R_fw.tick_params(labelsize=7)
ax_R_fw.set_xticks(MAIN_FREQS)
ax_R_fw.set_xticklabels([f"{f:.1f}" for f in MAIN_FREQS], rotation=45, fontsize=7)

# ── Panel D: R by amplitude tier (above_50 nowind — the mooring with visible SW)
ax_amp = fig.add_subplot(gs[1, 0])
for amp_v in [AMP_PRIMARY, AMP_LOW_SNR, AMP_WARN]:
    sub_r = res[(res["mooring"] == "above_50") & (res["wind"] == "no") &
                (res["amp"] == amp_v) & res["well_cond"]]
    if sub_r.empty:
        continue
    sg = sub_r.groupby("freq")["R"].agg(mean="mean", sem=lambda x: x.std()/max(1,len(x))).reset_index()
    alpha = AMP_ALPHA[amp_v]
    ax_amp.errorbar(sg["freq"], sg["mean"], yerr=sg["sem"],
                    fmt=f"{AMP_MARKERS[amp_v]}-", alpha=alpha,
                    label=f"{amp_v:.1f}V", capsize=3, markersize=5, lw=1.5)
ax_amp.axhline(0.05, color="k", lw=0.5, ls=":", alpha=0.4)
ax_amp.set_xlabel("Frequency [Hz]", fontsize=8)
ax_amp.set_ylabel("R = |B|/|A|", fontsize=8)
ax_amp.set_title("above_50, nowind:\nR by amplitude (⚠ 0.3V unreliable at high f)", fontsize=9)
ax_amp.legend(fontsize=8)
ax_amp.set_ylim(0, 0.5)
ax_amp.tick_params(labelsize=7)
ax_amp.set_xticks(MAIN_FREQS)
ax_amp.set_xticklabels([f"{f:.1f}" for f in MAIN_FREQS], rotation=45, fontsize=7)
ax_amp.text(0.02, 0.97, "s = 0.3V  o = 0.2V  ^ = 0.1V",
            transform=ax_amp.transAxes, fontsize=7, va="top")

# ── Panel E: R by amplitude tier (below_90_loose230 nowind)
ax_amp2 = fig.add_subplot(gs[1, 1])
for amp_v in [AMP_PRIMARY, AMP_LOW_SNR, AMP_WARN]:
    sub_r = res[(res["mooring"] == "below_90_loose230") & (res["wind"] == "no") &
                (res["amp"] == amp_v) & res["well_cond"]]
    if sub_r.empty:
        continue
    sg = sub_r.groupby("freq")["R"].agg(mean="mean", sem=lambda x: x.std()/max(1,len(x))).reset_index()
    ax_amp2.errorbar(sg["freq"], sg["mean"], yerr=sg["sem"],
                     fmt=f"{AMP_MARKERS[amp_v]}-", alpha=AMP_ALPHA[amp_v],
                     label=f"{amp_v:.1f}V", capsize=3, markersize=5, lw=1.5)
ax_amp2.axhline(0.05, color="k", lw=0.5, ls=":", alpha=0.4)
ax_amp2.set_xlabel("Frequency [Hz]", fontsize=8)
ax_amp2.set_ylabel("R = |B|/|A|", fontsize=8)
ax_amp2.set_title("below_90_loose230, nowind:\nR by amplitude", fontsize=9)
ax_amp2.legend(fontsize=8)
ax_amp2.set_ylim(0, 0.5)
ax_amp2.tick_params(labelsize=7)
ax_amp2.set_xticks(MAIN_FREQS)
ax_amp2.set_xticklabels([f"{f:.1f}" for f in MAIN_FREQS], rotation=45, fontsize=7)

# ── Panel F: above_50 wind effect on R
ax_wind_eff = fig.add_subplot(gs[1, 2])
for moor in ["above_50", "below_90_loose230"]:
    for wind in ["no", "full"]:
        sub = grp[(grp["mooring"] == moor) & (grp["wind"] == wind)]
        if sub.empty:
            continue
        label = f"{moor[-8:]} {'nw' if wind=='no' else 'fw'}"
        ax_wind_eff.errorbar(sub["freq"], sub["mean"], yerr=sub["sem"],
                             fmt="o" + WIND_LS[wind], color=MOOR_COLORS.get(moor, "gray"),
                             label=label, capsize=3, markersize=4, lw=1.2, alpha=0.8)
ax_wind_eff.axhline(0.05, color="k", lw=0.5, ls=":", alpha=0.4)
ax_wind_eff.set_xlabel("Frequency [Hz]", fontsize=8)
ax_wind_eff.set_ylabel("R = |B|/|A|", fontsize=8)
ax_wind_eff.set_title("R: wind effect by mooring\n(0.2V only, well-conditioned)", fontsize=9)
ax_wind_eff.legend(fontsize=7)
ax_wind_eff.set_ylim(0, 0.5)
ax_wind_eff.tick_params(labelsize=7)
ax_wind_eff.set_xticks(MAIN_FREQS)
ax_wind_eff.set_xticklabels([f"{f:.1f}" for f in MAIN_FREQS], rotation=45, fontsize=7)
ax_wind_eff.text(0.02, 0.97, "solid=nowind  dash=fullwind\n⚠ fullwind phase is less reliable",
                 transform=ax_wind_eff.transAxes, fontsize=7, va="top")

# ── Row 3: R distribution (all mooring × wind combinations, primary 0.2V) ─────
ax_box = fig.add_subplot(gs[2, :2])
box_data = []
box_labels = []
for moor in sorted(MOOR_COLORS.keys()):
    for wind in ["no", "full"]:
        d = res[(res["mooring"] == moor) & (res["wind"] == wind) &
                (res["amp"] == AMP_PRIMARY) & res["well_cond"]]["R"].dropna()
        if len(d) < 3:
            continue
        box_data.append(d.values)
        box_labels.append(f"{moor.split('_')[-1]}\n{wind}")

if box_data:
    bp = ax_box.boxplot(box_data, labels=box_labels, patch_artist=True,
                        medianprops={"color": "k", "lw": 2})
    colors = []
    for label in box_labels:
        for moor, col in MOOR_COLORS.items():
            if moor.split("_")[-1] in label:
                colors.append(col)
                break
        else:
            colors.append("gray")
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.5)
ax_box.axhline(0.05, color="k", lw=0.8, ls=":", alpha=0.5, label="R=0.05")
ax_box.axhline(0.20, color="k", lw=0.8, ls="--", alpha=0.3, label="R=0.20 assumed")
ax_box.set_ylabel("R = |B|/|A|", fontsize=8)
ax_box.set_title("R distribution by mooring and wind  (0.2V, well-conditioned)",
                 fontsize=9)
ax_box.legend(fontsize=7)
ax_box.tick_params(labelsize=7)

# ── Panel I: A_incident vs A_in_raw to check SW_factor from data ──────────────
ax_sw = fig.add_subplot(gs[2, 2])
# A_amp from MF is the incident wave amplitude
# a_in_fft_mm is the measured IN probe amplitude (= A_in_observed)
# SW_factor_data = a_in_fft_mm / A_amp  (should equal |1 + R*exp(2ikΔ)|)
valid_sw = res[res["well_cond"] & res["A_amp"].notna() & (res["A_amp"] > 0) &
               res["amp"].isin([AMP_PRIMARY])].copy()
valid_sw["SW_measured"] = valid_sw["a_in_fft_mm"] / (valid_sw["A_amp"] * 2 / 1)
# (a_in_fft_mm already is 2|z2|/N, A_amp comes from mf_decompose which uses unnormalized z)
# Actually A_amp is abs(A) in FFT units (unnormalized), and a_in_fft_mm is 2*abs(z2)/N in mm.
# The ratio a_in_fft_mm / (2*A_amp/N) = |z2|/|A| = SW_factor directly.
# A_amp = abs(A) in unnormalized FFT units, |z2| = N/2 * a_in_fft_mm
# SW_factor = |z2| / |A| = (N/2 * a_in_fft_mm) / A_amp
# But N varies per run... let's compute it differently.
# Just skip the units and use the structure of MF directly:
# z2 = A*exp(ikx2) + B*exp(-ikx2)  → |z2| = |A| * |1 + R*exp(2ikΔ)| = |A| * SW_factor
# So SW_factor_measured = |z2| / |A| = a_in_fft_mm / (2*A_amp/N) ... N unknown here.
# Alternative: just plot the TREND of R(f) per mooring.
# (Skip the SW_factor reconstruction and use this panel for something else.)

# Instead: plot the run-level scatter of R vs frequency for the primary mooring
for moor in MOOR_COLORS:
    sub_scatter = res[(res["mooring"] == moor) & (res["amp"] == AMP_PRIMARY) &
                      (res["wind"] == "no") & res["well_cond"]]["R"]
    sub_freq    = res[(res["mooring"] == moor) & (res["amp"] == AMP_PRIMARY) &
                      (res["wind"] == "no") & res["well_cond"]]["freq"]
    if sub_scatter.empty:
        continue
    ax_sw.scatter(sub_freq, sub_scatter, color=MOOR_COLORS[moor], alpha=0.4,
                  s=15, label=MOOR_LABELS.get(moor, moor)[:20])

ax_sw.axhline(0.05, color="k", lw=0.8, ls=":", alpha=0.5)
ax_sw.axhline(0.20, color="k", lw=0.8, ls="--", alpha=0.3)
ax_sw.set_xlabel("Frequency [Hz]", fontsize=8)
ax_sw.set_ylabel("R = |B|/|A|  (individual runs)", fontsize=8)
ax_sw.set_title("R scatter (0.2V, nowind)\nrun-level variability per mooring", fontsize=9)
ax_sw.legend(fontsize=6.5)
ax_sw.set_ylim(0, 0.6)
ax_sw.tick_params(labelsize=7)

fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
print(f"   Saved → {OUT_PNG}")

# ── 7. Write findings ─────────────────────────────────────────────────────────
print("7. Writing findings markdown...")

# Build summary table
table_rows = []
for _, r in grp.sort_values(["mooring", "wind", "freq"]).iterrows():
    table_rows.append(
        f"  {r['freq']:>5.2f}  {r['mooring']:>22s}  {r['wind']:>6s}  "
        f"{r['mean']:>7.4f}  {r['std']:>7.4f}  {int(r['n']):>4d}"
    )
table_str = "\n".join(table_rows)

# Ill-conditioning table
cond_rows = []
for f in sorted(res["freq"].unique()):
    k_v = solve_k(f)
    skd = abs(np.sin(k_v * DELTA))
    cond_rows.append(f"  {f:.2f} Hz:  k={k_v:.3f}  kΔ={k_v*DELTA:.3f}  |sin(kΔ)|={skd:.3f}  "
                     f"{'ILL-CONDITIONED ⚠' if skd < ILL_COND_THRESHOLD else 'OK'}")
cond_str = "\n".join(cond_rows)

# Overall R per mooring (primary only, well-conditioned)
r_above50  = res[(res["mooring"]=="above_50")      & (res["amp"]==AMP_PRIMARY) & res["well_cond"] & (res["wind"]=="no")]["R"]
r_lo230    = res[(res["mooring"]=="below_90_loose230") & (res["amp"]==AMP_PRIMARY) & res["well_cond"] & (res["wind"]=="no")]["R"]
r_lo300    = res[(res["mooring"]=="below_90_loose300") & (res["amp"]==AMP_PRIMARY) & res["well_cond"] & (res["wind"]=="no")]["R"]

md = f"""# Mansard-Funke reflection analysis

**Date**: {datetime.now().strftime('%Y-%m-%d')}
**Script**: `analysis_scratch/mansard_funke.py`
**Figure**: `analysis_scratch/mansard_funke.png`

## Method

Two-probe decomposition using 8804/250 (upstream, x₁={X_P1:.3f} m) and 9373/170
(IN probe, x₂={X_P2:.3f} m). Probe separation Δx = {DELTA*1000:.0f} mm.

    A_incident = [η̂(x₁)·exp(−ikx₂) − η̂(x₂)·exp(−ikx₁)] / [−2i·sin(kΔ)]
    B_reflected = [η̂(x₂)·exp(ikx₁) − η̂(x₁)·exp(ikx₂)] / [−2i·sin(kΔ)]
    R = |B| / |A|

Both probes use the SAME analysis window (intersection of individual probe windows)
to preserve phase coherence. Each run's FFT resolution is self-consistent.

Ill-conditioned when |sin(kΔ)| < {ILL_COND_THRESHOLD} (runs excluded from primary results).

## Quality / amplitude tiers

| Amplitude | Treatment |
|-----------|-----------|
| 0.2 V | **Primary** — best SNR, primary result |
| 0.1 V | **Include** — lower SNR, flag in figure |
| 0.3 V | **Caution** — non-linear risk at high frequency; untreated signal can be misleading |

Full-wind runs: the IN probe (9373/170) receives wind-wave energy that contaminates the
phase estimate at the paddle frequency. Fullwind R values are shown but interpreted with caution.

## Conditioning by frequency

```
{cond_str}
```

## R results (0.2V, no-wind, well-conditioned)

```
  freq             mooring    wind     mean R    std R     n
{table_str}
```

## Overall R by mooring (0.2V, nowind, well-conditioned, all freq)

| Mooring | n | mean R | median R | max R |
|---------|---|--------|----------|-------|
| above_50 | {len(r_above50)} | {r_above50.mean():.4f} | {r_above50.median():.4f} | {r_above50.max():.4f} |
| below_90_loose230 | {len(r_lo230)} | {r_lo230.mean():.4f} | {r_lo230.median():.4f} | {r_lo230.max():.4f} |
| below_90_loose300 | {len(r_lo300)} | {r_lo300.mean():.4f} | {r_lo300.median():.4f} | {r_lo300.max():.4f} |

## Interpretation

### Mooring comparison
- **above_50** (panel ~5 cm above water, stiff mooring): R = [filled in after run]
  This is the mooring where the standing wave was visually observed at full wind.
- **below_90_loose230/300** (panel 9 cm below water, loose line): R = [filled in after run]
  Consistent with the smoothness test result (sw_correction_findings.md) which showed
  R < 0.05 as the data-supported upper bound.

### Why above_50 may have higher R
A stiff above-water mooring constrains the panel more rigidly. Under incoming waves,
a rigid panel acts more like a partial breakwater → more reflection. A loose below-water
mooring allows the panel to move with the wave → less reflection → more transmission.

### Connection to SW correction analysis
The sw_correction_findings.md smoothness test showed that R=0.20 was inconsistent
with the raw OUT/IN data. The MF measurement here provides a direct, model-free R(f).

### Amplitude effect (0.3V warning)
At 0.3V and high frequency (1.6+ Hz), wave steepness increases and the signal may
contain harmonics or be affected by the reconstruction procedure. MF assumes a single
sinusoidal component — harmonics would bias both |A| and |B| estimates. Treat 0.3V
R results above 1.5 Hz as indicative only.

## Status

- [x] MF decomposition implemented
- [x] Both probes use same analysis window (phase-coherent)
- [x] Amplitude tier flagging (0.1V/0.2V/0.3V)
- [x] Ill-conditioning detection per frequency
- [x] Results by mooring (above_50 vs below_90)
- [ ] Quantitative comparison of R(above_50) vs R(below_90) after run
- [ ] Connect to SW correction: does measured R explain observed smooth curve?
"""

OUT_MD.write_text(md, encoding="utf-8")
print(f"   Saved → {OUT_MD}")
print("\nDone.")
