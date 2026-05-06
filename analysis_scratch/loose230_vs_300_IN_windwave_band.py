"""
loose230 vs loose300 — IN-probe wind-wave (3-5 Hz) energy
==========================================================

The IN probe (9373/170) sits ~50 cm in front of the panel mooring. Wind
generates choppy 3-5 Hz waves that the panel can interact with — looser
mooring (more free heave/surge) might damp wind-wave energy differently
than tighter mooring.

Round 4 of the loose230 vs loose300 confounder check:

  R. Pre-paddle PSD at IN (9373/170), 3 s window, integrated in the
     wind-wave band 3-5 Hz, per run. Compared loose230 vs loose300,
     fullwind only.
  S. Same for the swell band 0.5-2 Hz (paddle band) — should be near
     noise floor in pre-paddle (no paddle yet) but residual sloshing
     would show up here.
  T. Same for OUT probe (12400/250) — already sheltered, but useful
     as control: any Δ in 3-5 Hz at OUT must be wind passing UNDER
     or AROUND the panel, not bobbing.
  U. Quick overlay PNG of pre-paddle PSD at IN, mean per mooring.

Same data source as wind_qc_3s.py (pre-paddle 3 s window from
processed_dfs) but computes PSD instead of total σ.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
os.chdir(BASE)

from wavescripts.improved_data_loader import load_analysis_data, load_processed_dfs

import glob

FS = 250.0
SNIPPET_S = 3.0
SNIPPET_N = int(SNIPPET_S * FS)

WIND_WAVE_BAND = (3.0, 5.0)
SWELL_BAND     = (0.5, 2.0)

CANON_DIRS = [
    BASE / "waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange",
    BASE / "waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange",
]

# ── Load canon ─────────────────────────────────────────────────────────────
print(f"Loading {len(CANON_DIRS)} canon datasets …")
meta, _, _, _ = load_analysis_data(*[str(d) for d in CANON_DIRS], load_processed=False)
proc = {}
for d in CANON_DIRS:
    proc.update(load_processed_dfs(str(d)))
print(f"  → {len(meta)} runs, {len(proc)} time-series")

# Restrict scope
meta["amp_v"] = meta["WaveAmplitudeInput [Volt]"].apply(
    lambda v: round(float(v), 2) if pd.notna(v) else np.nan)
meta["mooring_short"] = meta["Mooring"].fillna("").str.replace("below_90_", "")
meta["date"] = pd.to_datetime(meta["file_date"]).dt.strftime("%Y-%m-%d")

sel = (
    (meta["PanelCondition"] == "full")
    & meta["Mooring"].isin(["below_90_loose230", "below_90_loose300"])
    & meta["WaveFrequencyInput [Hz]"].isin([1.3, 1.4, 1.5, 1.6])
    & meta["WindCondition"].isin(["full", "no"])
    & meta["amp_v"].isin([0.10, 0.20, 0.30])
    & meta["OUT/IN (FFT)"].notna()
)
W = meta[sel].copy().reset_index(drop=True)
print(f"In-scope rows: {len(W)}  (loose230={(W['mooring_short']=='loose230').sum()}, loose300={(W['mooring_short']=='loose300').sum()})")
print()

# ── PSD per run on pre-paddle 3 s ──────────────────────────────────────────
def _psd_eta(df, probe):
    col = f"eta_{probe}_interp" if f"eta_{probe}_interp" in df.columns else f"eta_{probe}"
    if col not in df.columns:
        return None, None
    eta = df[col].to_numpy(dtype=float)
    if len(eta) < SNIPPET_N:
        return None, None
    seg = eta[:SNIPPET_N]
    if not np.all(np.isfinite(seg)):
        return None, None
    seg = seg - np.mean(seg)
    f, Pxx = welch(seg, fs=FS, nperseg=min(512, len(seg)), detrend="constant")
    return f, Pxx

def _band_var(f, Pxx, band):
    mask = (f >= band[0]) & (f <= band[1])
    if not mask.any():
        return np.nan
    return float(np.trapezoid(Pxx[mask], f[mask]))

PROBES = ["9373/170", "12400/250"]

rows = []
psd_store_in  = {"loose230": [], "loose300": []}
psd_store_out = {"loose230": [], "loose300": []}
freqs_ref = None
for _, r in W.iterrows():
    path = r["path"]
    if path not in proc:
        continue
    df = proc[path]
    rec = {
        "path": path,
        "mooring_short": r["mooring_short"],
        "date": r["date"],
        "WindCondition": r["WindCondition"],
        "amp_v": r["amp_v"],
        "freq": r["WaveFrequencyInput [Hz]"],
        "Kt": r["OUT/IN (FFT)"],
    }
    for probe in PROBES:
        f, Pxx = _psd_eta(df, probe)
        if f is None:
            continue
        if freqs_ref is None:
            freqs_ref = f
        rec[f"var_wind_{probe}"]  = _band_var(f, Pxx, WIND_WAVE_BAND)
        rec[f"var_swell_{probe}"] = _band_var(f, Pxx, SWELL_BAND)
        rec[f"sigma_wind_{probe}"]  = np.sqrt(rec[f"var_wind_{probe}"])  if pd.notna(rec[f"var_wind_{probe}"])  else np.nan
        rec[f"sigma_swell_{probe}"] = np.sqrt(rec[f"var_swell_{probe}"]) if pd.notna(rec[f"var_swell_{probe}"]) else np.nan
        # Stash PSD curves per (mooring, wind) for IN probe and OUT probe
        if r["WindCondition"] == "full":
            if probe == "9373/170":
                psd_store_in[r["mooring_short"]].append(Pxx)
            elif probe == "12400/250":
                psd_store_out[r["mooring_short"]].append(Pxx)
    rows.append(rec)

D = pd.DataFrame(rows)
print(f"Computed PSD for {len(D)} runs")
print()

# ── R. IN probe wind-wave-band σ ──────────────────────────────────────────
print("=" * 80)
print(f"R. IN probe (9373/170) σ_η in 3-5 Hz wind-wave band, fullwind only")
print("=" * 80)
fw = D[D["WindCondition"] == "full"]
g = (fw.groupby("mooring_short")
       .agg(n=("Kt", "count"),
            sIN_wind_mean=("sigma_wind_9373/170", "mean"),
            sIN_wind_med=("sigma_wind_9373/170", "median"),
            sIN_wind_std=("sigma_wind_9373/170", "std"),
            sIN_wind_min=("sigma_wind_9373/170", "min"),
            sIN_wind_max=("sigma_wind_9373/170", "max")))
print(g.round(3).to_string())
print()
if {"loose230", "loose300"}.issubset(g.index):
    delta = g.loc["loose230", "sIN_wind_mean"] - g.loc["loose300", "sIN_wind_mean"]
    rel = 100 * delta / g.loc["loose300", "sIN_wind_mean"]
    print(f"  Δ σ_η_IN (3-5 Hz, Mar 26 − Mar 27) = {delta:+.3f} mm  ({rel:+.1f} %)")
print()

# Per-frequency breakdown
print("Per (amp, freq) breakdown — does the wind-wave-band signal vary by paddle setting?")
g2 = (fw.groupby(["amp_v", "freq", "mooring_short"])
        .agg(n=("Kt", "count"),
             sIN=("sigma_wind_9373/170", "mean"),
             Kt=("Kt", "mean")).round(3))
print(g2.to_string())
print()

# ── S. IN probe SWELL band (control — should be tiny pre-paddle) ──────────
print("=" * 80)
print(f"S. IN probe σ_η in 0.5-2 Hz swell band, fullwind only (control)")
print("=" * 80)
g = (fw.groupby("mooring_short")
       .agg(n=("Kt", "count"),
            sIN_swell_mean=("sigma_swell_9373/170", "mean"),
            sIN_swell_med=("sigma_swell_9373/170", "median"),
            sIN_swell_std=("sigma_swell_9373/170", "std")))
print(g.round(3).to_string())
print()

# ── T. OUT probe wind-wave-band ──────────────────────────────────────────
print("=" * 80)
print(f"T. OUT probe (12400/250) σ_η in 3-5 Hz wind-wave band, fullwind")
print("=" * 80)
g = (fw.groupby("mooring_short")
       .agg(n=("Kt", "count"),
            sOUT_wind_mean=("sigma_wind_12400/250", "mean"),
            sOUT_wind_std=("sigma_wind_12400/250", "std"),
            sOUT_wind_min=("sigma_wind_12400/250", "min"),
            sOUT_wind_max=("sigma_wind_12400/250", "max")))
print(g.round(4).to_string())
print()
if {"loose230", "loose300"}.issubset(g.index):
    delta = g.loc["loose230", "sOUT_wind_mean"] - g.loc["loose300", "sOUT_wind_mean"]
    rel = 100 * delta / g.loc["loose300", "sOUT_wind_mean"]
    print(f"  Δ σ_η_OUT (3-5 Hz, Mar 26 − Mar 27) = {delta:+.4f} mm  ({rel:+.1f} %)")
print()

# ── Per-run K_t ↔ σ_η_IN_3-5Hz correlation ────────────────────────────────
print("=" * 80)
print("U. Per-run correlation: K_t vs IN-probe wind-wave-band σ")
print("=" * 80)
for moor in ["loose230", "loose300"]:
    sub = fw[fw["mooring_short"] == moor].dropna(subset=["sigma_wind_9373/170", "Kt"])
    if len(sub) >= 4:
        r = sub[["Kt", "sigma_wind_9373/170"]].corr().iloc[0, 1]
        print(f"  {moor:9s} fullwind n={len(sub)}: r(K_t, σ_η_IN_3-5Hz) = {r:+.3f}")
print()

# Now save CSV
csv_out = BASE / "analysis_scratch" / "loose230_vs_300_IN_windwave_band.csv"
D.to_csv(csv_out, index=False)
print(f"CSV → {csv_out.relative_to(BASE)}")

# ── PSD overlay plot ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
COLOR = {"loose230": "#1f77b4", "loose300": "#d62728"}
for ax, (probe, store) in zip(axes, [("9373/170 (IN)", psd_store_in),
                                       ("12400/250 (OUT)", psd_store_out)]):
    for moor in ["loose230", "loose300"]:
        psds = store.get(moor, [])
        if not psds:
            continue
        arr = np.vstack(psds)
        mean = arr.mean(axis=0)
        med = np.median(arr, axis=0)
        ax.semilogy(freqs_ref, mean, color=COLOR[moor], lw=1.6,
                     label=f"{moor} mean (n={len(psds)})")
        ax.semilogy(freqs_ref, med,  color=COLOR[moor], lw=1.0, ls="--",
                     label=f"{moor} median")
    ax.axvspan(*WIND_WAVE_BAND, color="orange", alpha=0.10, label="3-5 Hz")
    ax.axvspan(*SWELL_BAND,     color="green",  alpha=0.08, label="0.5-2 Hz")
    ax.set_xlim(0, 20)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(r"$P_{\eta\eta}$  [mm$^2$/Hz]")
    ax.set_title(f"{probe} — pre-paddle 3 s, fullwind", fontsize=11)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="upper right")

fig.suptitle("Pre-paddle PSD — loose230 (Mar 26) vs loose300 (Mar 27), canon", fontsize=12)
fig.tight_layout()
png_out = BASE / "analysis_scratch" / "loose230_vs_300_IN_windwave_psd.png"
fig.savefig(png_out, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"PNG → {png_out.relative_to(BASE)}")

print("\nDone.")
