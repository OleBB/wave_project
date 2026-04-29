"""
A_IN settle time vs frequency — H&G mechanism vs panel-equilibrium test.
==========================================================================

For each thesis frequency, compute the sliding A_IN(t) (10·T window stepped
at 0.5 s) on the canonical March-2026 cond4 lowrange runs, then find the
EARLIEST window-start time at which A_IN settles to within ±1.5% of its
plateau median, computed across [t_arr_in + 15·T,  t_arr_in + 30·T].

If settle time (in PERIODS past t_arr) is roughly constant ≈ 10 across
freqs → consistent with H&G wavetrain dispersion (constant in T).
If it grows from ~4 T at 1.3 Hz to ~7 T at 1.6 Hz → consistent with
panel equilibration (L_panel/c_g · f scales with f).
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import os
os.chdir(BASE)

from wavescripts.improved_data_loader import load_analysis_data, load_processed_dfs
from wavescripts.constants import c_group, HG

FS                 = 250.0
FFT_BAND_HZ        = 0.05
THESIS_FREQS       = [1.3, 1.4, 1.5, 1.6]
TARGET_AMP         = 0.2
PER240_THRESHOLD_T = 50

IN_PROBES = ["9373/170", "9373/340"]
PROBE_R_M = {"9373/170": 9.373, "9373/340": 9.373}
TANK_DEPTH_M = HG.TANK_DEPTH_M

WINDOW_PERIODS = 10        # AFFT integration window length
SLIDING_STEP_S = 0.25      # stepped AFFT
SETTLE_TOL_PCT = 1.5       # tolerance for "settled"
PLATEAU_MIN_T  = 15        # plateau definition window
PLATEAU_MAX_T  = 30

PROCESSED_DIRS = [
    Path("waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange"),
    Path("waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"),
]


def fft_amp(seg, target_hz, fs=FS, band_hz=FFT_BAND_HZ):
    N = len(seg)
    if N < 4:
        return np.nan
    s = np.asarray(seg, dtype=float).copy()
    if np.isnan(s).any():
        idx = np.arange(N); good = ~np.isnan(s)
        if good.sum() < N * 0.9:
            return np.nan
        s = np.interp(idx, idx[good], s[good])
    freqs = np.fft.fftfreq(N, d=1.0 / fs)
    pos = freqs > 0
    pos_f = freqs[pos]
    mask = (pos_f >= target_hz - band_hz) & (pos_f <= target_hz + band_hz)
    if not mask.any():
        j = int(np.argmin(np.abs(pos_f - target_hz)))
    else:
        local = int(np.argmin(np.abs(pos_f[mask] - target_hz)))
        j = np.where(mask)[0][local]
    return float(2.0 * np.abs(np.fft.fft(s)[pos])[j] / N)


def get_eta(df, pos):
    for col in (f"eta_{pos}_interp", f"eta_{pos}"):
        if col in df.columns:
            return df[col].to_numpy(dtype=float)
    return None


def can_in(df):
    sigs = [get_eta(df, p) for p in IN_PROBES]
    sigs = [s for s in sigs if s is not None]
    if not sigs:
        return None
    nmin = min(len(s) for s in sigs)
    return np.nanmean(np.vstack([s[:nmin] for s in sigs]), axis=0)


def sliding_afft(signal, target_hz, window_s, step_s=SLIDING_STEP_S):
    N_win = int(round(window_s * FS))
    step = int(round(step_s * FS))
    if N_win >= len(signal):
        return np.array([]), np.array([])
    starts = np.arange(0, len(signal) - N_win + 1, step)
    ts = starts / FS
    A = np.array([fft_amp(signal[s:s + N_win], target_hz) for s in starts])
    return ts, A


print("Loading meta + processed_dfs (canon March-2026 lowrange) …")
combined_meta, _, _, _ = load_analysis_data(*PROCESSED_DIRS, load_processed=False)
processed_dfs = load_processed_dfs(*PROCESSED_DIRS)


def select_canon(target_freq):
    f_col = pd.to_numeric(combined_meta["WaveFrequencyInput [Hz]"], errors="coerce")
    a_col = pd.to_numeric(combined_meta["WaveAmplitudeInput [Volt]"], errors="coerce")
    mask = (
        np.isclose(f_col, target_freq, atol=0.02)
        & np.isclose(a_col, TARGET_AMP, atol=0.01)
        & (combined_meta["PanelCondition"] == "full")
        & (combined_meta["quality_flag"] == "ok")
    )
    sub = combined_meta[mask].copy()
    sub["N_input_periods"] = pd.to_numeric(sub["WavePeriodInput"], errors="coerce")
    sub["run_type"] = np.where(sub["N_input_periods"] >= PER240_THRESHOLD_T, "per240", "per40")
    return sub


# Compute settle time per (run, f, wind, run_type)
records = []
for f in THESIS_FREQS:
    sel = select_canon(f)
    t_arr_in = PROBE_R_M[IN_PROBES[0]] / c_group(f, TANK_DEPTH_M)
    T = 1.0 / f
    window_s = WINDOW_PERIODS / f

    # Plateau region in absolute time
    plat_lo_s = t_arr_in + PLATEAU_MIN_T / f
    plat_hi_s = t_arr_in + PLATEAU_MAX_T / f

    for _, r in sel.iterrows():
        df = processed_dfs.get(r["path"])
        if df is None:
            continue
        sig_in = can_in(df)
        if sig_in is None:
            continue
        ts, A = sliding_afft(sig_in, f, window_s)
        if len(A) == 0:
            continue

        # Plateau median
        plat_mask = (ts >= plat_lo_s) & (ts <= plat_hi_s)
        if not plat_mask.any():
            continue
        plat_median = float(np.nanmedian(A[plat_mask]))
        if not np.isfinite(plat_median) or plat_median == 0:
            continue

        # Find earliest t at which A is within tolerance and STAYS within
        # tolerance for at least the next 5 s of sliding window starts.
        rel = np.abs(A - plat_median) / plat_median * 100.0
        ok = rel < SETTLE_TOL_PCT
        n_lookahead = int(round(5.0 / SLIDING_STEP_S))   # 5 s lookahead

        settle_idx = None
        # Only consider points before paddle stop (per40) or before t = 50 s
        max_idx = int(np.searchsorted(ts, 50.0))
        for i in range(min(len(ok), max_idx) - n_lookahead):
            if ok[i:i + n_lookahead].all():
                settle_idx = i
                break

        if settle_idx is None:
            settle_t_s = np.nan
            settle_t_T = np.nan
        else:
            settle_t_s = float(ts[settle_idx])
            settle_t_T = (settle_t_s - t_arr_in) * f

        records.append({
            "freq_hz":     f,
            "wind":        r["WindCondition"],
            "run_type":    r["run_type"],
            "path":        Path(r["path"]).name,
            "t_arr_in_s":  t_arr_in,
            "settle_t_s":  settle_t_s,
            "settle_t_T_past_arr": settle_t_T,
            "plateau_med_mm": plat_median,
        })

per_run = pd.DataFrame(records)
print(f"\nPer-run results ({len(per_run)} rows):")
print(per_run.round(2).to_string(index=False))

# Aggregate per (f, wind, run_type) — median across runs
agg = per_run.groupby(["freq_hz", "wind", "run_type"]).agg(
    settle_T_med=("settle_t_T_past_arr", "median"),
    settle_T_min=("settle_t_T_past_arr", "min"),
    settle_T_max=("settle_t_T_past_arr", "max"),
    n=("settle_t_T_past_arr", "size"),
).reset_index()

print("\nMedian settle time (in periods past wave-front arrival), per (f, wind, run_type):")
print(agg.round(2).to_string(index=False))

# Headline: cross-frequency comparison, marginalised across wind+run_type
hl = per_run.groupby("freq_hz")["settle_t_T_past_arr"].agg(["median", "min", "max", "count"])
print("\nHeadline — settle time in T past arrival, across wind+run_type:")
print(hl.round(2).to_string())

# Save
out_csv = Path(__file__).parent / "in_settle_time_vs_f.csv"
per_run.to_csv(out_csv, index=False)
print(f"\nSaved → {out_csv.relative_to(BASE)}")
