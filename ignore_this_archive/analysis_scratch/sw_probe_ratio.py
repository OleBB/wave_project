"""
Standing-wave fingerprint via 8804/9373 amplitude ratio
========================================================

Uses 8804/250 as a clean "far-field incident" reference and 9373/170 as the probe
that may be inside the standing-wave zone close to the panel.

The amplitude ratio  |z₂/z₁|  at each paddle frequency measures the standing-wave
factor at the IN probe:

    |z₂/z₁| ≈ SW_factor(9373, f) = sqrt(1 + R² + 2R·cos(2k·Δ_panel))

where Δ_panel = x_panel − 9.373 m (distance from IN probe to panel).

If a standing wave is present at 9373:
  - The ratio oscillates with frequency (node → below baseline, antinode → above)
  - Oscillation amplitude ≈ 2R
  - Oscillation period in k-space: π / Δ_panel

The baseline (flat offset from probe calibration + ~0.6m fetch growth) is removed by
normalising the ratio to its mean across frequencies, or by dividing by a smooth fit.

Also done for wind-only runs: the PSD ratio PSD(9373)/PSD(8804) is the incoherent
equivalent — no phase, but the oscillation still shows in the PSD amplitude if R > 0.

Run from repo root:
    conda run -n draumkvedet python analysis_scratch/sw_probe_ratio.py
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
from scipy.optimize import brentq, minimize_scalar
from scipy.signal import welch
from datetime import datetime

from wavescripts.improved_data_loader import load_analysis_data, load_processed_dfs

# ── Constants ─────────────────────────────────────────────────────────────────
G, DEPTH, FS = 9.81, 0.580, 250.0
X_P1 = 8.804   # 8804/250 — upstream reference
X_P2 = 9.373   # 9373/170 — IN probe (standing-wave zone)

BASE    = Path(__file__).parent.parent
OUT_PNG = Path(__file__).parent / "sw_probe_ratio.png"
OUT_MD  = Path(__file__).parent / "sw_probe_ratio_findings.md"

def solve_k(f):
    omega = 2 * np.pi * f
    return brentq(lambda k: omega**2 - G * k * np.tanh(k * DEPTH), 1e-4, 500.0)

def sw_factor(f, R, x_panel):
    """SW_factor at x_P2 for panel at x_panel."""
    k = solve_k(f)
    delta = x_panel - X_P2   # distance from IN probe to panel
    return np.sqrt(1 + R**2 + 2*R*np.cos(2*k*delta))

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("1. Loading metadata...")
dirs = sorted(glob.glob(str(BASE / "waveprocessed" / "PROCESSED-*")))
meta_all, _, _, _ = load_analysis_data(*dirs, load_processed=False)

# Paddle-wave runs with both probe windows available
wave = meta_all[
    meta_all["WaveFrequencyInput [Hz]"].notna() &
    (meta_all["WaveFrequencyInput [Hz]"] > 0) &
    (meta_all["quality_flag"] == "ok") &
    (meta_all["PanelCondition"] == "full") &
    (meta_all["in_position"] == "9373/170") &
    meta_all["Computed Probe 8804/250 start"].notna() &
    meta_all["Computed Probe 9373/170 start"].notna()
].copy()
print(f"   {len(wave)} paddle-wave runs (fullpanel, ok, in=9373/170)")

# Wind-only runs
nowave = meta_all[
    meta_all["WaveFrequencyInput [Hz]"].isna() &
    (meta_all["WindCondition"] == "full") &
    (meta_all["quality_flag"] == "ok") &
    (meta_all["PanelCondition"] == "full") &
    meta_all["Computed Probe 8804/250 start"].notna()
].copy()
print(f"   {len(nowave)} fullwind nowave runs")

# ── 2. Load processed time series ─────────────────────────────────────────────
print("2. Loading processed time series...")
proc_dfs = load_processed_dfs(*dirs)

def get_sig(df, pos, s, e):
    for col in [f"eta_{pos}_interp", f"eta_{pos}"]:
        if col in df.columns:
            sig = df[col].iloc[s:e+1].to_numpy(dtype=float)
            if np.isnan(sig).mean() > 0.05:
                return None
            if np.isnan(sig).any():
                idx = np.arange(len(sig)); good = ~np.isnan(sig)
                sig = np.interp(idx, idx[good], sig[good])
            return sig
    return None

# ── 3. PADDLE-WAVE: |z₂/z₁| at paddle frequency per run ─────────────────────
print("3. Computing |z₂/z₁| for paddle-wave runs...")

FFT_WIN = 0.10   # Hz — bin search window

paddle_rows = []
for _, row in wave.iterrows():
    path   = row["path"]
    f_pad  = row["WaveFrequencyInput [Hz]"]
    amp    = row["WaveAmplitudeInput [Volt]"]
    wind   = row["WindCondition"]
    moor   = row["Mooring"]

    if path not in proc_dfs:
        continue
    df = proc_dfs[path]

    s1 = int(row["Computed Probe 8804/250 start"])
    e1 = int(row["Computed Probe 8804/250 end"])
    s2 = int(row["Computed Probe 9373/170 start"])
    e2 = int(row["Computed Probe 9373/170 end"])
    s_sh, e_sh = max(s1, s2), min(e1, e2)
    if e_sh - s_sh < int(2 * FS / f_pad):
        continue

    sig1 = get_sig(df, "8804/250", s_sh, e_sh)
    sig2 = get_sig(df, "9373/170", s_sh, e_sh)
    if sig1 is None or sig2 is None:
        continue
    N = min(len(sig1), len(sig2))
    sig1, sig2 = sig1[:N], sig2[:N]

    fft1   = np.fft.fft(sig1)
    fft2   = np.fft.fft(sig2)
    freqs  = np.fft.fftfreq(N, 1.0/FS)
    pos_m  = freqs > 0
    pfreqs = freqs[pos_m]
    win_m  = np.abs(pfreqs - f_pad) <= FFT_WIN
    if not win_m.any():
        continue
    nidx   = np.argmin(np.abs(pfreqs[win_m] - f_pad))
    gidx   = np.where(pos_m)[0][win_m][nidx]

    z1 = fft1[gidx]
    z2 = fft2[gidx]
    if abs(z1) == 0:
        continue

    ratio_amp   = abs(z2) / abs(z1)
    ratio_phase = np.angle(z2 / z1)

    paddle_rows.append({
        "freq": round(f_pad, 2),
        "amp": amp, "wind": wind, "mooring": moor,
        "ratio_amp": ratio_amp,
        "ratio_phase": ratio_phase,
        "k": solve_k(f_pad),
    })

pdf = pd.DataFrame(paddle_rows)
print(f"   {len(pdf)} runs with valid z₂/z₁")

# ── 4. WIND-ONLY: PSD ratio per mooring ──────────────────────────────────────
print("4. Computing PSD ratio for wind-only runs...")

NPERSEG = 4096   # ~16s segments, Δf=0.061 Hz

MOORINGS = ["above_50", "below_90_loose230", "below_90_loose300"]
psd_profiles = {}   # moor → {freq, ratio, psd1, psd2}

for moor in MOORINGS:
    rows = nowave[nowave["Mooring"] == moor]
    segs1, segs2 = [], []
    for _, row in rows.iterrows():
        path = row["path"]
        if path not in proc_dfs: continue
        df = proc_dfs[path]
        s = int(row["Computed Probe 8804/250 start"])
        e = int(row["Computed Probe 8804/250 end"])
        if e - s < NPERSEG: continue
        s1 = get_sig(df, "8804/250", s, e)
        s2 = get_sig(df, "9373/170", s, e)
        if s1 is not None and s2 is not None:
            N = min(len(s1), len(s2))
            segs1.append(s1[:N]); segs2.append(s2[:N])

    if not segs1: continue
    all1 = np.concatenate(segs1)
    all2 = np.concatenate(segs2)
    f_w, P1 = welch(all1, fs=FS, nperseg=NPERSEG, noverlap=NPERSEG//2)
    _,   P2 = welch(all2, fs=FS, nperseg=NPERSEG, noverlap=NPERSEG//2)
    ratio = np.sqrt(P2 / np.where(P1 > 0, P1, np.nan))   # amplitude ratio = sqrt(PSD ratio)
    psd_profiles[moor] = {"freq": f_w, "ratio": ratio, "P1": P1, "P2": P2,
                           "n_runs": len(segs1), "total_s": len(all1)/FS}

# ── 5. Theoretical SW_factor curves ──────────────────────────────────────────
print("5. Building theoretical curves...")

f_fine = np.linspace(0.7, 1.9, 500)
k_fine = np.array([solve_k(f) for f in f_fine])

# Expected phase of z₂/z₁ for pure incident wave: -kΔ₁₂ (probe separation)
delta12 = X_P2 - X_P1  # = 0.569 m
exp_phase_fine = -k_fine * delta12   # expected in (-inf, inf); wrap to [-π,π] for plot

# Theoretical |z₂/z₁| for a standing wave at 9373 but NOT at 8804:
# |z₂/z₁| = SW_factor(9373, f) / 1  = sqrt(1 + R² + 2R·cos(2k·Δ_panel))
# We try multiple (R, x_panel) combos
COMBOS = [
    (0.05, 10.5), (0.05, 11.0), (0.05, 11.5),
    (0.10, 10.5), (0.10, 11.0), (0.10, 11.5),
    (0.15, 11.0),
]

# ── 6. Print summary ──────────────────────────────────────────────────────────
print("\n6. Summary:")

# Group paddle-wave ratio by (freq, mooring, wind)
primary = pdf[(pdf["amp"] == 0.2) & pdf["mooring"].isin(MOORINGS)]
grp = primary.groupby(["freq", "mooring", "wind"])["ratio_amp"].agg(
    mean="mean", std="std", n="count", median="median"
).reset_index()

print(f"\n   |z₂/z₁| at paddle freq — 0.2V, all moorings:")
print(f"   {'freq':>5} {'mooring':>22} {'wind':>6}  {'mean':>6} {'std':>6}  n")
for _, r in grp.sort_values(["mooring","wind","freq"]).iterrows():
    print(f"   {r['freq']:>5.2f} {r['mooring']:>22} {r['wind']:>6}  "
          f"{r['mean']:>6.4f} {r['std']:>6.4f}  {int(r['n'])}")

# Overall baseline per mooring (nowind, 0.2V)
print(f"\n   Baseline |z₂/z₁| per mooring (nowind, 0.2V):")
for moor in MOORINGS:
    sub = primary[(primary["mooring"] == moor) & (primary["wind"] == "no")]
    if sub.empty: continue
    overall = sub["ratio_amp"].mean()
    overall_std = sub["ratio_amp"].std()
    print(f"   {moor}: mean={overall:.4f} std={overall_std:.4f} n={len(sub)}")

# ── 7. Plotting ───────────────────────────────────────────────────────────────
print("7. Plotting...")

MOOR_COLORS = {
    "above_50":          "#E74C3C",
    "below_90_loose230": "#2980B9",
    "below_90_loose300": "#27AE60",
}
WIND_LS = {"no": "-", "full": "--"}

fig = plt.figure(figsize=(18, 14))
fig.suptitle(
    "Standing-wave fingerprint via 8804/250 → 9373/170 amplitude ratio\n"
    f"8804 = far-field reference  ·  9373 = IN probe (~1.6m from panel)  ·  "
    f"oscillation amplitude ≈ 2R if SW present",
    fontsize=10,
)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.35)

# ── Panel A: |z₂/z₁| vs freq — nowind, 0.2V, per mooring ───────────────────
ax_ratio_nw = fig.add_subplot(gs[0, :2])

for moor in MOORINGS:
    c = MOOR_COLORS.get(moor, "gray")
    for wind in ["no", "full"]:
        sub = grp[(grp["mooring"] == moor) & (grp["wind"] == wind)]
        if sub.empty: continue
        ls = WIND_LS[wind]
        lbl = f"{moor[-10:]} {'nw' if wind=='no' else 'fw'}"
        ax_ratio_nw.errorbar(
            sub["freq"], sub["mean"], yerr=sub["std"] / np.sqrt(sub["n"].clip(1)),
            fmt=f"o{ls}", color=c, capsize=3, markersize=5, lw=1.5,
            label=lbl, alpha=0.85 if wind == "no" else 0.5,
        )

ax_ratio_nw.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.4, label="|H|=1")
ax_ratio_nw.set_xlabel("Paddle frequency [Hz]", fontsize=8)
ax_ratio_nw.set_ylabel("|z₂(9373) / z₁(8804)|", fontsize=8)
ax_ratio_nw.set_title(
    "|z₂/z₁| at paddle frequency — 0.2V\n"
    "flat → no SW  ·  oscillating → SW at 9373\n"
    "solid=nowind  dashed=fullwind", fontsize=9)
ax_ratio_nw.legend(fontsize=6.5, ncol=2)
ax_ratio_nw.set_ylim(0.5, 1.5)
freqs_shown = sorted(grp["freq"].unique())
ax_ratio_nw.set_xticks(freqs_shown)
ax_ratio_nw.set_xticklabels([f"{f:.2f}" for f in freqs_shown], rotation=45, fontsize=7)
ax_ratio_nw.tick_params(labelsize=7)

# ── Panel B: theoretical SW_factor overlay — nowind only ────────────────────
ax_theory = fig.add_subplot(gs[0, 2])

for moor in ["above_50", "below_90_loose230"]:
    c = MOOR_COLORS.get(moor, "gray")
    sub = grp[(grp["mooring"] == moor) & (grp["wind"] == "no")]
    if sub.empty: continue
    # Normalise to mean to remove calibration offset
    mean_val = sub["mean"].mean()
    ax_theory.errorbar(
        sub["freq"], sub["mean"] / mean_val,
        yerr=sub["std"] / np.sqrt(sub["n"].clip(1)) / mean_val,
        fmt="o-", color=c, capsize=3, markersize=5, lw=1.5,
        label=moor[-10:], alpha=0.9,
    )

# Overlay theoretical curves (normalised to mean=1)
line_styles = ["-", "--", ":"]
colors_t    = ["#2C3E50", "#7F8C8D", "#95A5A6"]
for i, (R_t, xp_t) in enumerate([(0.07, 11.0), (0.07, 10.5), (0.15, 11.0)]):
    sw_vals = np.array([sw_factor(f, R_t, xp_t) for f in f_fine])
    mean_sw = sw_vals[(f_fine >= 0.8) & (f_fine <= 1.9)].mean()
    ax_theory.plot(f_fine, sw_vals / mean_sw,
                   color=colors_t[i % len(colors_t)], lw=1.2,
                   ls=line_styles[i % len(line_styles)],
                   label=f"theory R={R_t}, panel={xp_t}m", alpha=0.7)

ax_theory.axhline(1.0, color="k", lw=0.5, ls="--", alpha=0.3)
ax_theory.set_xlabel("Freq [Hz]", fontsize=8)
ax_theory.set_ylabel("|z₂/z₁| / mean  (normalised)", fontsize=8)
ax_theory.set_title("Data normalised to mean\nvs theoretical SW_factor shapes", fontsize=9)
ax_theory.legend(fontsize=6.5)
ax_theory.set_xlim(0.75, 1.95)
ax_theory.set_ylim(0.75, 1.25)
ax_theory.tick_params(labelsize=7)

# ── Row 1: per-mooring breakdown (nowind 0.2V) ───────────────────────────────
for col_i, moor in enumerate(["above_50", "below_90_loose230", "below_90_loose300"]):
    ax = fig.add_subplot(gs[1, col_i])
    c  = MOOR_COLORS.get(moor, "gray")

    # All amplitudes, nowind
    for amp_v, marker in [(0.1, "^"), (0.2, "o"), (0.3, "s")]:
        sub = pdf[(pdf["mooring"] == moor) & (pdf["wind"] == "no") & (pdf["amp"] == amp_v)]
        if sub.empty: continue
        sg = sub.groupby("freq")["ratio_amp"].agg(mean="mean", sem=lambda x: x.std()/max(1,len(x))).reset_index()
        alpha = 1.0 if amp_v == 0.2 else 0.55
        ax.errorbar(sg["freq"], sg["mean"], yerr=sg["sem"],
                    fmt=f"{marker}-", color=c, alpha=alpha,
                    capsize=3, markersize=5, lw=1.5, label=f"{amp_v:.1f}V")

    # Theoretical SW_factor overlay for R=0.07, panel=11.0
    sw7 = np.array([sw_factor(f, 0.07, 11.0) for f in f_fine])
    mean_sw7 = sw7[(f_fine >= 0.8) & (f_fine <= 1.9)].mean()
    # Scale to match data mean
    sub02 = pdf[(pdf["mooring"] == moor) & (pdf["wind"] == "no") & (pdf["amp"] == 0.2)]
    data_mean = sub02["ratio_amp"].mean() if len(sub02) > 0 else 1.0
    ax.plot(f_fine, sw7 / mean_sw7 * data_mean, "k--", lw=1.2, alpha=0.5,
            label="theory R=0.07\npanel=11.0m")

    ax.axhline(data_mean, color="k", lw=0.5, ls=":", alpha=0.3)
    ax.set_xlabel("Freq [Hz]", fontsize=8)
    ax.set_ylabel("|z₂/z₁|", fontsize=8)
    ax.set_title(f"{moor}\nnowind, by amplitude", fontsize=9)
    ax.set_ylim(0.5, 1.5)
    ax.legend(fontsize=6.5)
    ax.tick_params(labelsize=7)
    ax.set_xticks(sorted(pdf["freq"].unique()))
    ax.set_xticklabels([f"{f:.1f}" for f in sorted(pdf["freq"].unique())],
                       rotation=45, fontsize=7)

# ── Row 2: wind-only PSD ratio per mooring ───────────────────────────────────
ax_psd_rat   = fig.add_subplot(gs[2, :2])
ax_psd_zoom  = fig.add_subplot(gs[2, 2])

for moor, prof in psd_profiles.items():
    c   = MOOR_COLORS.get(moor, "gray")
    f   = prof["freq"]
    rat = prof["ratio"]  # sqrt(PSD2/PSD1) — amplitude ratio
    mask = (f >= 0.5) & (f <= 6.0)
    ax_psd_rat.plot(f[mask], rat[mask], color=c, lw=1.5, alpha=0.85,
                    label=f"{moor[-10:]} (n={prof['n_runs']}, {prof['total_s']:.0f}s)")
    # Zoomed: 0.8–3 Hz
    zoom = (f >= 0.8) & (f <= 3.0)
    ax_psd_zoom.plot(f[zoom], rat[zoom], color=c, lw=1.5, alpha=0.85,
                     label=f"{moor[-10:]}")

ax_psd_rat.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.4, label="ratio=1")
ax_psd_rat.set_xlabel("Frequency [Hz]", fontsize=8)
ax_psd_rat.set_ylabel("sqrt(PSD₂/PSD₁) = |H(f)|", fontsize=8)
ax_psd_rat.set_title(
    "Wind-only: PSD amplitude ratio 9373/8804 (incoherent avg)\n"
    "Flat → fetch growth only  ·  oscillation → SW modulation",
    fontsize=9)
ax_psd_rat.legend(fontsize=7)
ax_psd_rat.set_xlim(0.5, 6.0)
ax_psd_rat.set_ylim(0, 3)
ax_psd_rat.tick_params(labelsize=7)
ax_psd_rat.text(0.01, 0.97,
    "Note: R≈0.07 → oscillation amplitude ≈0.14 on top of baseline",
    transform=ax_psd_rat.transAxes, fontsize=7, va="top")

ax_psd_zoom.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.4)
ax_psd_zoom.set_xlabel("Frequency [Hz]", fontsize=8)
ax_psd_zoom.set_ylabel("sqrt(PSD₂/PSD₁)", fontsize=8)
ax_psd_zoom.set_title("Wind-only ratio — zoomed 0.8–3 Hz", fontsize=9)
ax_psd_zoom.legend(fontsize=7)
ax_psd_zoom.set_xlim(0.8, 3.0)
ax_psd_zoom.set_ylim(0.7, 2.0)
ax_psd_zoom.tick_params(labelsize=7)

fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
print(f"   Saved → {OUT_PNG}")

# ── 8. Write findings ─────────────────────────────────────────────────────────
print("8. Writing findings markdown...")

grp_summary = grp.sort_values(["mooring","wind","freq"]).to_string(index=False)  # already 0.2V (primary filtered)

md = f"""# Standing-wave fingerprint: 8804/9373 amplitude ratio

**Date**: {datetime.now().strftime('%Y-%m-%d')}
**Script**: `analysis_scratch/sw_probe_ratio.py`
**Figure**: `analysis_scratch/sw_probe_ratio.png`

## Method

8804/250 (x={X_P1} m) = far-field reference — assumed pure incident wave.
9373/170 (x={X_P2} m) = IN probe — potentially inside standing-wave zone (~1.6 m from panel).

For paddle-wave runs:
    |z₂/z₁| = |FFT(9373) / FFT(8804)| at paddle frequency (±0.10 Hz window, same window)

For pure incident wave at both probes: |z₂/z₁| = 1 (constant).
For SW at 9373 but not at 8804:       |z₂/z₁| = SW_factor(9373, f)
    = sqrt(1 + R² + 2R·cos(2k·Δ_panel))  where Δ_panel = x_panel − {X_P2} m

Oscillation amplitude ≈ 2R. Oscillation period in k: π / Δ_panel.
The baseline offset (probe calibration difference, fetch growth) is a flat factor —
divide by mean or fit a constant to remove it.

For wind-only runs: PSD ratio sqrt(PSD₂/PSD₁) — incoherent equivalent.

## Theoretical node/antinode frequencies (R=0.07, panel at 11.0 m, Δ_panel=1.627 m)

Nodes   (|z₂/z₁| < baseline): when 2k·1.627 = π + 2nπ
Antinodes (|z₂/z₁| > baseline): when 2k·1.627 = 2nπ

## Results (0.2V)

```
{grp_summary}
```

## Key observations

[Fill after viewing figure]

## Status

- [x] |z₂/z₁| computed for all paddle-wave runs
- [x] Wind-only PSD ratio computed per mooring
- [x] Theoretical SW_factor curves overlaid for several (R, panel position) combos
- [ ] Interpretation written (fill after viewing figure)
"""

OUT_MD.write_text(md, encoding="utf-8")
print(f"   Saved → {OUT_MD}")
print("\nDone.")
