"""
Travel-time-shifted H&G window vs same-window — quick sanity diagnostic
========================================================================

The previous scripts used [50T, 60T] at BOTH probes. Physically, the
same absolute window measures different portions of the paddle output
at two probes separated by 3 m in our tank. This script tests whether
that simplification matters in practice for per240 thesis-scope.

Two H&G implementations compared at OUT probe (12400/250):

  AFFT_OUT_same    = AFFT in window [50T, 60T]   (same as H&G at IN)
  AFFT_OUT_shifted = AFFT in window [50T + ΔT, 60T + ΔT],
                     ΔT = (r_OUT − r_IN) / c_group × f_paddle

At 1.3–1.6 Hz, c_group = g/(4πf) (deep water — tanh(kh) ≈ 1 at h=0.58 m).
ΔT ranges from 6.6 T at 1.3 Hz to 9.9 T at 1.6 Hz.

Verdict:
  median |ΔOUT/IN| < 2 %   → same-window H&G is defensible; keep (A).
  median |ΔOUT/IN| ≥ 2 %   → probe-shifted is more principled; use (B).

Run from repo root:
    /opt/anaconda3/envs/draumkvedet/bin/python analysis_scratch/hg_travel_time_shift_check.py
"""

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
os.chdir(BASE)

from wavescripts.improved_data_loader import load_analysis_data, load_processed_dfs

# ── Config ────────────────────────────────────────────────────────────────────
FS              = 250.0
FFT_BAND_HZ     = 0.05
G               = 9.81
HG_START_T      = 50
HG_END_T        = 60
HG_LENGTH_T     = HG_END_T - HG_START_T

R_IN_M          = 9.373
R_OUT_M         = 12.40
DELTA_R_M       = R_OUT_M - R_IN_M     # 3.027 m

IN_PROBES = ["9373/170", "9373/340"]
OUT_PROBE = "12400/250"

FREQS = [1.3, 1.4, 1.5, 1.6]
AMPS  = [0.1, 0.2, 0.3]
WINDS = ["no", "full"]

# Per240-rich datasets suffice — we only need long runs for this check.
PROCESSED_DIRS = [
    Path("waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange"),
    Path("waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"),
]

OUT_CSV = Path(__file__).parent / "hg_travel_time_shift_check_summary.csv"


# ── Helpers (copied from huseby_grue_window.py) ──────────────────────────────
def fft_amp_at_freq(segment: np.ndarray, target_hz: float,
                    fs: float = FS, band_hz: float = FFT_BAND_HZ) -> float:
    N = len(segment)
    if N < 4:
        return np.nan
    seg = np.asarray(segment, dtype=float).copy()
    if np.isnan(seg).any():
        idx  = np.arange(N)
        good = ~np.isnan(seg)
        if good.sum() < N * 0.9:
            return np.nan
        seg = np.interp(idx, idx[good], seg[good])
    freqs  = np.fft.fftfreq(N, d=1.0 / fs)
    pos    = freqs > 0
    pos_f  = freqs[pos]
    mask   = (pos_f >= target_hz - band_hz) & (pos_f <= target_hz + band_hz)
    if not mask.any():
        j = int(np.argmin(np.abs(pos_f - target_hz)))
    else:
        local_idx = int(np.argmin(np.abs(pos_f[mask] - target_hz)))
        j = np.where(mask)[0][local_idx]
    fft_vals = np.fft.fft(seg)
    amps_pos = 2.0 * np.abs(fft_vals[pos]) / N
    return float(amps_pos[j])


def get_eta(df_run, pos):
    for col in (f"eta_{pos}_interp", f"eta_{pos}"):
        if col in df_run.columns:
            return df_run[col].to_numpy(dtype=float)
    return None


def canonical_in(df_run):
    sigs = [get_eta(df_run, p) for p in IN_PROBES]
    sigs = [s for s in sigs if s is not None]
    if not sigs:
        return None
    nmin = min(len(s) for s in sigs)
    return np.nanmean(np.vstack([s[:nmin] for s in sigs]), axis=0)


def delta_T_periods(f_hz: float) -> float:
    """Travel-time difference IN->OUT in paddle periods, deep water."""
    c_group = G / (4 * np.pi * f_hz)
    delta_s = DELTA_R_M / c_group
    return delta_s * f_hz


# ── Load + compute ───────────────────────────────────────────────────────────
print("1. Loading data …")
meta, _, _, _ = load_analysis_data(*PROCESSED_DIRS, load_processed=False)
meta["Mooring"] = meta["Mooring"].replace({"below_90_loose230": "below_90_loose",
                                            "below_90_loose300": "below_90_loose"})
wave = meta[
    meta["WaveFrequencyInput [Hz]"].notna()
    & (meta["PanelCondition"] == "full")
    & (meta["Mooring"] == "below_90_loose")
    & (meta["quality_flag"] == "ok")
    & (meta["WavePeriodInput"].astype(float) >= 50)          # per240 only
    & meta["WaveFrequencyInput [Hz]"].round(2).isin(FREQS)
    & meta["WaveAmplitudeInput [Volt]"].round(2).isin(AMPS)
    & meta["WindCondition"].isin(WINDS)
].copy()
print(f"   {len(wave)} per240 thesis-scope runs")

print("\n2. Loading processed_dfs …")
pdfs = load_processed_dfs(*PROCESSED_DIRS)

print("\n3. Travel-time deltas (deep water, ΔR = 3.027 m):")
for f in FREQS:
    print(f"   {f:.1f} Hz: c_group = {G/(4*np.pi*f):.3f} m/s  →  ΔT = {delta_T_periods(f):.2f} T")

print("\n4. Computing AFFT per run, both OUT windows …")
rows = []
for _, r in wave.iterrows():
    df_run = pdfs.get(r["path"])
    if df_run is None:
        continue
    f = float(r["WaveFrequencyInput [Hz]"])
    spp = int(round(FS / f))
    sig_in  = canonical_in(df_run)
    sig_out = get_eta(df_run, OUT_PROBE)
    if sig_in is None or sig_out is None:
        continue

    dT = delta_T_periods(f)
    # IN H&G window [50T, 60T] (reference, unchanged)
    i0_in, i1_in = HG_START_T * spp, HG_END_T * spp
    if i1_in > len(sig_in):
        continue
    a_in = fft_amp_at_freq(sig_in[i0_in:i1_in], f)

    # OUT same window [50T, 60T]
    i0_os, i1_os = HG_START_T * spp, HG_END_T * spp
    if i1_os > len(sig_out):
        continue
    a_out_same = fft_amp_at_freq(sig_out[i0_os:i1_os], f)

    # OUT shifted window [50T + ΔT, 60T + ΔT]
    shift_samples = int(round(dT * spp))
    i0_osh = HG_START_T * spp + shift_samples
    i1_osh = HG_END_T   * spp + shift_samples
    if i1_osh > len(sig_out):
        a_out_shifted = np.nan
    else:
        a_out_shifted = fft_amp_at_freq(sig_out[i0_osh:i1_osh], f)

    rows.append({
        "path": r["path"],
        "freq_hz": round(f, 2),
        "amp_V": round(float(r["WaveAmplitudeInput [Volt]"]), 2),
        "wind": r["WindCondition"],
        "delta_T_periods": round(dT, 3),
        "A_in_hg": a_in,
        "A_out_same": a_out_same,
        "A_out_shifted": a_out_shifted,
        "OUT_IN_same": a_out_same / a_in if a_in > 0 else np.nan,
        "OUT_IN_shifted": a_out_shifted / a_in if (a_in > 0 and np.isfinite(a_out_shifted)) else np.nan,
    })
df = pd.DataFrame(rows)
df["delta_OUTIN"] = df["OUT_IN_shifted"] - df["OUT_IN_same"]
df["delta_OUTIN_rel"] = df["delta_OUTIN"] / df["OUT_IN_same"]
df.to_csv(OUT_CSV, index=False)
print(f"   per-run table → {OUT_CSV.relative_to(BASE)}")

print("\n5. Aggregate per (freq, amp, wind):")
print(df.groupby(["freq_hz", "amp_V", "wind"]).agg(
    n=("path", "count"),
    OUTIN_same_mean=("OUT_IN_same", "mean"),
    OUTIN_shifted_mean=("OUT_IN_shifted", "mean"),
    delta_abs_mean=("delta_OUTIN", lambda s: s.abs().mean()),
    delta_rel_mean_pct=("delta_OUTIN_rel", lambda s: (s.abs().mean() * 100)),
).round(4).to_string())

print("\n6. Verdict across all runs (n = {}):".format(df["delta_OUTIN"].dropna().size))
abs_med = float(df["delta_OUTIN"].abs().median())
abs_max = float(df["delta_OUTIN"].abs().max())
rel_med = float(df["delta_OUTIN_rel"].abs().median())
rel_max = float(df["delta_OUTIN_rel"].abs().max())
print(f"   median |Δ OUT/IN|        = {abs_med:.4f}")
print(f"   max    |Δ OUT/IN|        = {abs_max:.4f}")
print(f"   median |Δ OUT/IN| / same = {rel_med*100:.2f} %")
print(f"   max    |Δ OUT/IN| / same = {rel_max*100:.2f} %")
print()
if rel_med < 0.02:
    print("   ✓ Same-window H&G is defensible (< 2 % median). Stick with option A.")
else:
    print("   ⚠ Travel-time-shifted H&G is more principled. Switch to option B.")

print("\nDone.")
