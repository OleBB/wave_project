"""
Wind-wave coherence and transfer function analysis — fullwind nowave runs
=========================================================================

Goal: characterise whether the panel causes a detectable standing-wave pattern
in the wind-generated wave field.

Why not Mansard-Funke directly?
  MF requires a coherent signal at each frequency bin — a fixed-phase wave passing
  both probes simultaneously. Wind-only waves are broadband stochastic: at any
  individual frequency bin the two probe signals have essentially random phases.
  Applying MF to incoherent noise always gives |A| ≈ |B| → R → 1 regardless of
  physics. This is not meaningful.

The correct approach for broadband stochastic waves:
  Welch cross-spectral density (CPSD) averaging over many overlapping segments.
  This suppresses the incoherent noise and reveals the coherent wave component.

Quantities computed per mooring:
  1. PSD1(f), PSD2(f)   — power at 8804/250 and 9373/170
  2. CPSD(f)            — cross-spectral density (averaged over segments)
  3. γ²(f)              — coherence = |CPSD|² / (PSD1·PSD2)
                           = 1 for perfectly coherent waves, 0 for pure noise
  4. |H(f)| = sqrt(PSD2/PSD1) — amplitude transfer ratio (probe 2 relative to 1)
     arg(H(f)) = arg(CPSD)    — phase transfer (how phase advances from 1→2)
     Expected for pure incident: arg(H) = −kΔ (wave travelling in +x direction)
  5. Comparison of arg(H) vs −kΔ (full dispersion): deviation → standing wave content

Standing wave fingerprint:
  If the panel reflects partially (R > 0), the wave field between paddle and panel is
  a partial standing wave. The amplitude transfer |H(f)| between two probes in the
  standing wave field oscillates with frequency (not constant = 1).
  For probe separation Δ = 0.569 m:
    |H| = sqrt[(1 + R)²cos²(kΔ) + R² sin²(kΔ)] / sqrt[...]   (depends on probe phases)
  The key observable: if |H(f)| deviates from 1 in a frequency-dependent oscillatory
  manner, that is consistent with a partial standing wave. The amplitude of the
  oscillation is ~ 2R for small R.

Run from repo root:
    conda run -n draumkvedet python analysis_scratch/mf_windonly.py

Output:
    analysis_scratch/mf_windonly.png
    analysis_scratch/mf_windonly_findings.md
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
from scipy.signal import csd, welch, coherence as scipy_coherence
from datetime import datetime

from wavescripts.improved_data_loader import load_analysis_data, load_processed_dfs

# ── Physical constants ────────────────────────────────────────────────────────
G     = 9.81
DEPTH = 0.580
FS    = 250.0

X_P1  = 8.804   # m — 8804/250 (upstream)
X_P2  = 9.373   # m — 9373/170 (IN probe)
DELTA = X_P2 - X_P1  # = 0.569 m

# Welch parameters
NPERSEG    = 2048    # ~8 s segments at 250 Hz — good freq resolution (~0.12 Hz)
NOVERLAP   = 1024    # 50% overlap
COHERENCE_MIN = 0.5  # γ² below this → unreliable phase/amplitude estimate

FREQ_MIN = 0.5
FREQ_MAX = 8.0

BASE    = Path(__file__).parent.parent
OUT_PNG = Path(__file__).parent / "mf_windonly.png"
OUT_MD  = Path(__file__).parent / "mf_windonly_findings.md"

# ── Helpers ───────────────────────────────────────────────────────────────────
def solve_k(f, d=DEPTH):
    omega = 2 * np.pi * f
    return brentq(lambda k: omega**2 - G * k * np.tanh(k * d), 1e-4, 500.0)

def expected_phase(f):
    """Expected phase of H(f) = arg(z2/z1) for pure incident wave (rightward)."""
    k = solve_k(f)
    return -k * DELTA  # arg(z2/z1) = -kΔ (mod 2π for a rightward wave, Python FFT conv.)

# ── 1. Load metadata ──────────────────────────────────────────────────────────
print("1. Loading metadata...")
dirs = sorted(glob.glob(str(BASE / "waveprocessed" / "PROCESSED-*")))
meta_all, _, _, _ = load_analysis_data(*dirs, load_processed=False)

nowave = meta_all[
    meta_all["WaveFrequencyInput [Hz]"].isna() &
    (meta_all["WindCondition"] == "full") &
    (meta_all["quality_flag"] == "ok") &
    (meta_all["PanelCondition"] == "full") &
    meta_all["Computed Probe 8804/250 start"].notna() &
    meta_all["Computed Probe 8804/250 end"].notna()
].copy()
print(f"   {len(nowave)} fullwind nowave fullpanel ok runs")
print(f"   Mooring: {dict(nowave['Mooring'].value_counts())}")
nowave["dur_s"] = (nowave["Computed Probe 8804/250 end"] -
                   nowave["Computed Probe 8804/250 start"]) / FS

# ── 2. Load processed time series ─────────────────────────────────────────────
print("2. Loading processed time series...")
proc_dfs = load_processed_dfs(*dirs)
print(f"   {len(proc_dfs)} DataFrames loaded")

# ── 3. Compute per-mooring CPSD using all runs concatenated ──────────────────
print("3. Computing coherence and transfer function per mooring...")

MOORINGS   = sorted(nowave["Mooring"].unique())
MOOR_COLORS = {
    "above_50":          "#E74C3C",
    "below_90_loose230": "#2980B9",
    "below_90_loose300": "#27AE60",
}
MOOR_LABELS = {
    "above_50":          "above_50 (above water, stiff)",
    "below_90_loose230": "below_90_loose230 (9cm below, 23cm line)",
    "below_90_loose300": "below_90_loose300 (9cm below, 30cm line)",
}

mooring_results = {}
summary_lines   = []

def log(s=""):
    summary_lines.append(s)
    print("   " + s if s else "")

log(f"Wind-wave coherence & transfer function — fullwind nowave fullpanel")
log(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
log(f"Probe pair: 8804/250 + 9373/170  Δ={DELTA*1000:.0f}mm")
log(f"Welch: nperseg={NPERSEG} ({NPERSEG/FS:.1f}s)  noverlap={NOVERLAP}  "
    f"Δf={FS/NPERSEG:.3f} Hz")
log(f"Coherence threshold for phase reliability: γ² ≥ {COHERENCE_MIN}")
log()

for moor in MOORINGS:
    rows   = nowave[nowave["Mooring"] == moor]
    segs1  = []
    segs2  = []
    n_runs = 0

    for _, row in rows.iterrows():
        path = row["path"]
        if path not in proc_dfs:
            continue
        df = proc_dfs[path]

        s = int(row["Computed Probe 8804/250 start"])
        e = int(row["Computed Probe 8804/250 end"])

        def get_sig(pos):
            for col in [f"eta_{pos}_interp", f"eta_{pos}"]:
                if col in df.columns:
                    sig = df[col].iloc[s:e+1].to_numpy(dtype=float)
                    nan_frac = np.isnan(sig).mean()
                    if nan_frac > 0.05:
                        return None
                    if nan_frac > 0:
                        idx = np.arange(len(sig))
                        good = ~np.isnan(sig)
                        sig = np.interp(idx, idx[good], sig[good])
                    return sig
            return None

        sig1 = get_sig("8804/250")
        sig2 = get_sig("9373/170")
        if sig1 is None or sig2 is None:
            continue
        N = min(len(sig1), len(sig2))
        if N < NPERSEG:
            continue
        segs1.append(sig1[:N])
        segs2.append(sig2[:N])
        n_runs += 1

    if not segs1:
        log(f"Mooring {moor}: no valid runs")
        continue

    # Concatenate all runs for this mooring (maximises number of Welch segments)
    all1 = np.concatenate(segs1)
    all2 = np.concatenate(segs2)
    total_s = len(all1) / FS

    # Welch PSD
    f_w, P1 = welch(all1, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP)
    _,   P2 = welch(all2, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP)

    # Cross-spectral density
    f_w, Gxy = csd(all1, all2, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP)

    # Coherence
    f_w, gamma2 = scipy_coherence(all1, all2, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP)

    # Transfer function amplitude and phase
    H_amp   = np.sqrt(P2 / np.where(P1 > 0, P1, np.nan))   # |z2|/|z1| amplitude ratio
    H_phase = np.angle(Gxy)                                   # arg(CPSD) = phase of H

    # Expected phase for pure incident wave
    f_mask     = (f_w > 0) & (f_w >= FREQ_MIN) & (f_w <= FREQ_MAX)
    freq_ok    = f_w[f_mask]
    exp_phase  = np.array([expected_phase(f) for f in freq_ok])
    # Wrap to [-π, π] for comparison
    exp_phase_wrapped = (exp_phase + np.pi) % (2 * np.pi) - np.pi

    # Phase deviation from pure-incident expectation (wrapped to [-π, π])
    H_phase_ok   = H_phase[f_mask]
    phase_dev    = np.angle(np.exp(1j * (H_phase_ok - exp_phase_wrapped)))
    H_amp_ok     = H_amp[f_mask]
    gamma2_ok    = gamma2[f_mask]
    P1_ok        = P1[f_mask]
    P2_ok        = P2[f_mask]

    # Only reliable where coherence is high
    reliable = gamma2_ok >= COHERENCE_MIN

    mooring_results[moor] = {
        "freq":          freq_ok,
        "H_amp":         H_amp_ok,
        "H_phase":       H_phase_ok,
        "exp_phase":     exp_phase_wrapped,
        "phase_dev":     phase_dev,
        "gamma2":        gamma2_ok,
        "P1":            P1_ok,
        "P2":            P2_ok,
        "reliable":      reliable,
        "n_runs":        n_runs,
        "total_s":       total_s,
    }

    # Summary stats for reliable bins
    rel_f    = freq_ok[reliable]
    rel_amp  = H_amp_ok[reliable]
    rel_dev  = np.abs(phase_dev[reliable])
    rel_g2   = gamma2_ok[reliable]

    log(f"Mooring: {moor}  ({n_runs} runs, {total_s:.0f}s, {reliable.sum()} reliable freq bins)")
    if reliable.any():
        # Focus on 1–4 Hz
        main = reliable & (freq_ok >= 1.0) & (freq_ok <= 4.0)
        log(f"  γ² (1–4 Hz):       median={np.nanmedian(gamma2_ok[main]):.3f}  "
            f"max={np.nanmax(gamma2_ok[main]):.3f}")
        log(f"  |H| (1–4 Hz):      mean={np.nanmean(H_amp_ok[main]):.4f}  "
            f"std={np.nanstd(H_amp_ok[main]):.4f}  "
            f"(expected 1.0 for pure propagation)")
        log(f"  |phase dev| 1–4 Hz: mean={np.nanmean(np.abs(phase_dev[main])):.3f} rad  "
            f"(expected ~0 for pure incident)")
        # Deviation of |H| from 1: amplitude of oscillation ≈ 2R for small R
        H_dev = H_amp_ok[main] - 1.0
        log(f"  |H|−1 (1–4 Hz):    mean={np.nanmean(H_dev):.4f}  "
            f"std={np.nanstd(H_dev):.4f}  "
            f"→ implied R ≈ {np.nanstd(H_dev)/2:.3f} from H oscillation amplitude")
    log()

# ── 4. Plotting ───────────────────────────────────────────────────────────────
print("4. Plotting...")

fig = plt.figure(figsize=(18, 14))
fig.suptitle(
    "Wind-wave coherence and transfer function — fullwind nowave fullpanel\n"
    f"Δ = 569 mm  |  Welch segments {NPERSEG/FS:.0f}s / 50% overlap  |  "
    f"Solid lines: γ² ≥ {COHERENCE_MIN} (reliable)",
    fontsize=10,
)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.35)

# ── Row 0: PSD, coherence, |H(f)| ─────────────────────────────────────────
ax_psd  = fig.add_subplot(gs[0, 0])
ax_g2   = fig.add_subplot(gs[0, 1])
ax_Hamp = fig.add_subplot(gs[0, 2])

for moor, res in mooring_results.items():
    c   = MOOR_COLORS.get(moor, "gray")
    lbl = f"{MOOR_LABELS.get(moor, moor)} (n={res['n_runs']})"
    f   = res["freq"]
    r   = res["reliable"]

    # PSD at 8804 (upstream)
    ax_psd.semilogy(f, res["P1"], color=c, lw=1.5, label=lbl)

    # Coherence
    ax_g2.plot(f, res["gamma2"], color=c, lw=1.5, label=lbl)

    # |H(f)| — show full, highlight reliable
    ax_Hamp.plot(f, res["H_amp"], color=c, lw=0.7, alpha=0.3)
    if r.any():
        ax_Hamp.plot(f[r], res["H_amp"][r], color=c, lw=1.8, label=lbl)

ax_psd.set_xlabel("Freq [Hz]", fontsize=8)
ax_psd.set_ylabel("PSD [mm²/Hz]", fontsize=8)
ax_psd.set_title("Wind-wave energy (8804/250)", fontsize=9)
ax_psd.set_xlim(FREQ_MIN, FREQ_MAX)
ax_psd.legend(fontsize=6.5)
ax_psd.tick_params(labelsize=7)

ax_g2.axhline(COHERENCE_MIN, color="k", lw=1, ls="--", alpha=0.5,
              label=f"threshold={COHERENCE_MIN}")
ax_g2.set_xlabel("Freq [Hz]", fontsize=8)
ax_g2.set_ylabel("γ² (coherence)", fontsize=8)
ax_g2.set_title("Probe-to-probe coherence\n8804/250 ↔ 9373/170", fontsize=9)
ax_g2.set_ylim(0, 1.05)
ax_g2.set_xlim(FREQ_MIN, FREQ_MAX)
ax_g2.legend(fontsize=6.5)
ax_g2.tick_params(labelsize=7)

ax_Hamp.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.4, label="|H|=1 expected")
ax_Hamp.set_xlabel("Freq [Hz]", fontsize=8)
ax_Hamp.set_ylabel("|H(f)| = √(PSD₂/PSD₁)", fontsize=8)
ax_Hamp.set_title("|H(f)|: amplitude ratio\n(solid = reliable, faded = low-coherence)", fontsize=9)
ax_Hamp.set_ylim(0, 3)
ax_Hamp.set_xlim(FREQ_MIN, FREQ_MAX)
ax_Hamp.legend(fontsize=6.5)
ax_Hamp.tick_params(labelsize=7)

# ── Row 1: phase of H(f), phase deviation, and expected vs measured ──────────
ax_phase    = fig.add_subplot(gs[1, 0])
ax_phdev    = fig.add_subplot(gs[1, 1])
ax_zoom_H   = fig.add_subplot(gs[1, 2])

# Expected phase curve
f_exp = np.linspace(0.5, 8.0, 500)
ph_exp = np.array([expected_phase(f) for f in f_exp])
ph_exp_w = (ph_exp + np.pi) % (2*np.pi) - np.pi
ax_phase.plot(f_exp, np.degrees(ph_exp_w), "k--", lw=1.5, alpha=0.5,
              label="expected (pure incident)", zorder=5)

for moor, res in mooring_results.items():
    c = MOOR_COLORS.get(moor, "gray")
    f = res["freq"]
    r = res["reliable"]
    ax_phase.plot(f, np.degrees(res["H_phase"]), color=c, lw=0.7, alpha=0.3)
    if r.any():
        ax_phase.plot(f[r], np.degrees(res["H_phase"][r]),
                      color=c, lw=1.8, label=MOOR_LABELS.get(moor, moor)[:18])

    # Phase deviation
    ax_phdev.plot(f, np.degrees(res["phase_dev"]), color=c, lw=0.7, alpha=0.3)
    if r.any():
        ax_phdev.plot(f[r], np.degrees(res["phase_dev"][r]),
                      color=c, lw=1.8, label=MOOR_LABELS.get(moor, moor)[:18])

    # |H| zoomed to 0.8–5 Hz
    ax_zoom_H.plot(f, res["H_amp"], color=c, lw=0.7, alpha=0.3)
    if r.any():
        ax_zoom_H.plot(f[r], res["H_amp"][r],
                       color=c, lw=1.8, label=MOOR_LABELS.get(moor, moor)[:18])

ax_phase.set_xlabel("Freq [Hz]", fontsize=8)
ax_phase.set_ylabel("arg(H)  [°]", fontsize=8)
ax_phase.set_title("Phase of H(f) = arg(CPSD)\nvs expected for pure incident", fontsize=9)
ax_phase.set_xlim(FREQ_MIN, FREQ_MAX)
ax_phase.set_ylim(-185, 185)
ax_phase.legend(fontsize=6.5)
ax_phase.tick_params(labelsize=7)

ax_phdev.axhline(0, color="k", lw=0.8, ls="--", alpha=0.4)
ax_phdev.set_xlabel("Freq [Hz]", fontsize=8)
ax_phdev.set_ylabel("phase deviation from expected  [°]", fontsize=8)
ax_phdev.set_title("Phase deviation from pure-incident\n(0° = pure rightward wave)", fontsize=9)
ax_phdev.set_xlim(FREQ_MIN, FREQ_MAX)
ax_phdev.set_ylim(-180, 180)
ax_phdev.legend(fontsize=6.5)
ax_phdev.tick_params(labelsize=7)

ax_zoom_H.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.4)
ax_zoom_H.set_xlabel("Freq [Hz]", fontsize=8)
ax_zoom_H.set_ylabel("|H(f)|", fontsize=8)
ax_zoom_H.set_title("|H| zoomed: oscillation = SW fingerprint\n(oscillation amplitude ≈ 2R)", fontsize=9)
ax_zoom_H.set_ylim(0, 2.5)
ax_zoom_H.set_xlim(0.8, 5.0)
ax_zoom_H.legend(fontsize=6.5)
ax_zoom_H.tick_params(labelsize=7)

# ── Row 2: per-mooring overlay with separation ────────────────────────────────
ax_a50  = fig.add_subplot(gs[2, 0])
ax_b90  = fig.add_subplot(gs[2, 1])
ax_comp = fig.add_subplot(gs[2, 2])

# Compare H_amp: above_50 vs below_90 on same axes (reliable only)
for moor, ax, title in [
    ("above_50",          ax_a50, "above_50: |H| and phase dev (reliable bins)"),
    ("below_90_loose230", ax_b90, "below_90_loose230: |H| and phase dev (reliable)"),
]:
    res = mooring_results.get(moor)
    if res is None:
        ax.set_visible(False)
        continue
    f = res["freq"]
    r = res["reliable"]
    ax2 = ax.twinx()
    ax.plot(f, res["H_amp"], color="k", lw=0.7, alpha=0.2)
    if r.any():
        ax.plot(f[r], res["H_amp"][r], color=MOOR_COLORS.get(moor, "gray"),
                lw=1.8, label="|H(f)|")
    ax.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.4)
    ax2.plot(f[r] if r.any() else f,
             np.degrees(res["phase_dev"])[r] if r.any() else np.degrees(res["phase_dev"]),
             color="#9B59B6", lw=1.2, alpha=0.7, label="phase dev [°]")
    ax2.axhline(0, color="#9B59B6", lw=0.5, ls=":", alpha=0.4)
    ax2.set_ylim(-90, 90)
    ax2.set_ylabel("phase dev [°]", fontsize=7, color="#9B59B6")
    ax.set_xlabel("Freq [Hz]", fontsize=8)
    ax.set_ylabel("|H(f)|", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.set_ylim(0, 2.5)
    ax.set_xlim(0.8, 5.0)
    ax.tick_params(labelsize=7)
    ax2.tick_params(labelsize=7)
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1+lines2, labs1+labs2, fontsize=7)

# Comparison: |H| above_50 vs below_90 (reliable only, 0.8–5 Hz)
for moor, res in mooring_results.items():
    c = MOOR_COLORS.get(moor, "gray")
    f = res["freq"]
    r = res["reliable"] & (f >= 0.8) & (f <= 5.0)
    if r.any():
        ax_comp.plot(f[r], res["H_amp"][r], color=c, lw=1.8,
                     label=MOOR_LABELS.get(moor, moor)[:18])
ax_comp.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.4, label="|H|=1")
ax_comp.set_xlabel("Freq [Hz]", fontsize=8)
ax_comp.set_ylabel("|H(f)| — reliable only", fontsize=8)
ax_comp.set_title("|H| comparison: all moorings\n(reliable bins only, 0.8–5 Hz)", fontsize=9)
ax_comp.set_ylim(0, 2.5)
ax_comp.set_xlim(0.8, 5.0)
ax_comp.legend(fontsize=6.5)
ax_comp.tick_params(labelsize=7)

fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
print(f"   Saved → {OUT_PNG}")

# ── 5. Write findings ─────────────────────────────────────────────────────────
print("5. Writing findings markdown...")

md = f"""# Wind-wave coherence and transfer function — fullwind nowave

**Date**: {datetime.now().strftime('%Y-%m-%d')}
**Script**: `analysis_scratch/mf_windonly.py`
**Figure**: `analysis_scratch/mf_windonly.png`

## Why not Mansard-Funke directly?

MF requires a coherent single-frequency wave at each FFT bin. Wind-only waves are
broadband stochastic: at any individual bin, the two probe signals have random phases.
Applying MF to incoherent noise always gives |A| ≈ |B| → R → 1 regardless of physics.
This was confirmed: initial MF run on wind-only data gave R ≈ 1.0 for all moorings.

## Method: Welch cross-spectral density

Concatenate all runs per mooring → compute via Welch averaging:
- PSD₁(f), PSD₂(f) at 8804/250 and 9373/170
- CPSD G₁₂(f) = cross-spectral density
- Coherence γ²(f) = |G₁₂|² / (PSD₁·PSD₂)  [0=incoherent, 1=fully coherent]
- Transfer function |H(f)| = √(PSD₂/PSD₁), arg(H) = arg(G₁₂)

Welch parameters: nperseg={NPERSEG} samples ({NPERSEG/FS:.1f}s), 50% overlap, Δf={FS/NPERSEG:.3f} Hz.

Standing wave fingerprint:
- For pure incident wave: |H(f)| = 1, arg(H) = −kΔ exactly
- For partial standing wave (R > 0): |H(f)| oscillates around 1 with amplitude ≈ 2R;
  arg(H) deviates from −kΔ. Both vary with frequency in a node/antinode pattern.

Only bins with γ² ≥ {COHERENCE_MIN} are considered reliable.

## Numerical summary

```
{"".join(summary_lines)}
```

## Key questions

1. Is there a coherent wind-wave component propagating past both probes? (γ² > {COHERENCE_MIN})
2. Does |H(f)| oscillate around 1 — the standing-wave fingerprint?
3. Does above_50 show more oscillation than below_90 (consistent with the visually
   observed standing wave under above-water mooring)?

## Status

- [x] Welch CPSD computed per mooring (all runs concatenated per mooring type)
- [x] Coherence, |H|, phase deviation plotted
- [ ] Interpretation written (fill after viewing figure)
"""

OUT_MD.write_text(md, encoding="utf-8")
print(f"   Saved → {OUT_MD}")
print("\nDone.")
