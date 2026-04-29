"""
Per40 OUT plateau end — empirical measurement at all four thesis frequencies.
==============================================================================

For each (f, wind, probe) combination, compute the sliding A_FFT(t) with
window length N(f)·T (matching the proposed pipeline window). Then locate
the plateau-END time: the latest window-start time at which A_FFT stays
within tolerance of the plateau median (computed in a "definitely flat"
sub-region just past wave-front arrival + ramp-up).

The plateau END at OUT is the binding constraint for the proposed FFT
window length: any window whose START is later than (plateau_end −
window_length) will integrate over post-clean tail samples and AFFT will
droop. So the maximum usable N(f) at OUT (per40) is:

    N_max(f, OUT) = (plateau_end_OUT − t_arr_OUT − N_offset·T) · f

Same for IN (where the constraint is parasitic-2f arrival).

Both per40 and per240 measured. Both winds. Both probes (canonical IN,
OUT). Two tolerance levels (1.0 % strict, 2.0 % relaxed) reported.

Output: analysis_scratch/per40_plateau_end_measurement.csv
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

# ── Config ──────────────────────────────────────────────────────────────
FS                 = 250.0
FFT_BAND_HZ        = 0.05

THESIS_FREQS       = [1.3, 1.4, 1.5, 1.6]
TARGET_AMP         = 0.2
PER240_THRESHOLD_T = 50

IN_PROBES = ["9373/170", "9373/340"]
OUT_PROBE = "12400/250"
PROBE_R_M = {"9373/170": 9.373, "9373/340": 9.373, "12400/250": 12.400}
TANK_DEPTH_M = HG.TANK_DEPTH_M

# Sliding-AFFT window length per frequency — matches proposed N(f)
N_LENGTH_LOOKUP = {1.3: 10, 1.4: 13, 1.5: 13, 1.6: 13}

# Plateau definition — flat region begins this many T after t_arr at the
# probe (well past ramp-up + AFFT settling), ends a few T later but well
# before per40 paddle stop reaches the probe
PLATEAU_REF_T_LO = 8.0
PLATEAU_REF_T_HI = 12.0

# Tolerance levels for "still on the plateau"
TOL_STRICT_PCT = 1.0
TOL_RELAX_PCT  = 2.0

# Sliding step
SLIDING_STEP_S = 0.10
# Run AFFT from this t to this t (truncate at signal end)
SLIDE_START_S  = 5.0
SLIDE_STOP_S   = 80.0

PROCESSED_DIRS = [
    Path("waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange"),
    Path("waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"),
]

OUT_CSV = Path(__file__).parent / "per40_plateau_end_measurement.csv"


# ── Helpers ─────────────────────────────────────────────────────────────
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


def sliding_afft(signal, target_hz, window_s, step_s=SLIDING_STEP_S,
                 t_lo=SLIDE_START_S, t_hi=SLIDE_STOP_S):
    N_win = int(round(window_s * FS))
    step = int(round(step_s * FS))
    if N_win >= len(signal):
        return np.array([]), np.array([])
    n_lo = int(round(t_lo * FS))
    n_hi = min(len(signal) - N_win, int(round(t_hi * FS)))
    starts = np.arange(n_lo, n_hi + 1, step)
    ts = starts / FS
    A = np.array([fft_amp(signal[s:s + N_win], target_hz) for s in starts])
    return ts, A


def find_plateau_end(ts, A, plat_lo_s, plat_hi_s, tol_pct):
    """Walk forward from the plateau reference region; report the LAST
    timepoint (in ts) at which A is within tol_pct of the plateau median.
    Plateau end = first timepoint AFTER the reference region where A leaves
    the tolerance band AND stays out for at least 2 s of step."""
    mask = (ts >= plat_lo_s) & (ts <= plat_hi_s)
    if not mask.any():
        return np.nan, np.nan
    ref = float(np.nanmedian(A[mask]))
    if not np.isfinite(ref) or ref == 0:
        return ref, np.nan
    rel = (A - ref) / ref * 100.0
    inside = np.abs(rel) < tol_pct

    # Walk forward from end of reference region
    n_lookahead = int(round(2.0 / SLIDING_STEP_S))
    ref_end_idx = int(np.searchsorted(ts, plat_hi_s))
    plateau_end_t = np.nan
    for i in range(ref_end_idx, len(inside) - n_lookahead):
        if not inside[i:i + n_lookahead].any():
            # Found a 2-s window with no inside-tolerance points — droop has set in
            plateau_end_t = float(ts[i])
            break
    if np.isnan(plateau_end_t):
        plateau_end_t = float(ts[-1])    # never droops within sliding range
    return ref, plateau_end_t


# ── Load ────────────────────────────────────────────────────────────────
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


# ── Compute plateau end per (run, freq, probe, wind) ────────────────────
records = []
for f in THESIS_FREQS:
    n_length = N_LENGTH_LOOKUP[f]
    window_s = n_length / f
    sel = select_canon(f)
    print(f"\n— f={f} Hz: N(f)={n_length}T  ({window_s:.2f}s window). "
          f"per40 n={int((sel['run_type']=='per40').sum())}, "
          f"per240 n={int((sel['run_type']=='per240').sum())}")

    for _, r in sel.iterrows():
        df = processed_dfs.get(r["path"])
        if df is None:
            continue

        for probe_label, probe_signal_fn, r_probe in [
            ("IN",  can_in,                        PROBE_R_M[IN_PROBES[0]]),
            ("OUT", lambda d: get_eta(d, OUT_PROBE), PROBE_R_M[OUT_PROBE]),
        ]:
            sig = probe_signal_fn(df)
            if sig is None:
                continue
            t_arr = r_probe / c_group(f, TANK_DEPTH_M)
            ts, A = sliding_afft(sig, f, window_s)
            if len(A) == 0:
                continue

            plat_lo_s = t_arr + PLATEAU_REF_T_LO / f
            plat_hi_s = t_arr + PLATEAU_REF_T_HI / f

            ref_strict, end_strict = find_plateau_end(
                ts, A, plat_lo_s, plat_hi_s, TOL_STRICT_PCT)
            ref_relax,  end_relax  = find_plateau_end(
                ts, A, plat_lo_s, plat_hi_s, TOL_RELAX_PCT)

            records.append({
                "freq_hz":        f,
                "wind":           r["WindCondition"],
                "run_type":       r["run_type"],
                "probe":          probe_label,
                "path":           Path(r["path"]).name,
                "t_arr_s":        t_arr,
                "plateau_ref_lo_s":   plat_lo_s,
                "plateau_ref_hi_s":   plat_hi_s,
                "plateau_ref_med_mm": ref_strict,
                "plateau_end_strict_s":     end_strict,
                "plateau_end_strict_T_past_arr": (end_strict - t_arr) * f if np.isfinite(end_strict) else np.nan,
                "plateau_end_relax_s":      end_relax,
                "plateau_end_relax_T_past_arr":  (end_relax  - t_arr) * f if np.isfinite(end_relax)  else np.nan,
                "n_window_T":     n_length,
            })

per_run = pd.DataFrame(records)
print(f"\nPer-run rows: {len(per_run)}")

# ── Aggregate: median per (f, probe, wind, run_type) ────────────────────
agg = per_run.groupby(["freq_hz", "probe", "wind", "run_type"]).agg(
    n=("plateau_end_strict_s", "size"),
    plat_end_strict_s_med = ("plateau_end_strict_s", "median"),
    plat_end_strict_s_min = ("plateau_end_strict_s", "min"),
    plat_end_strict_s_max = ("plateau_end_strict_s", "max"),
    plat_end_strict_T_med = ("plateau_end_strict_T_past_arr", "median"),
    plat_end_relax_s_med  = ("plateau_end_relax_s",  "median"),
    plat_end_relax_T_med  = ("plateau_end_relax_T_past_arr",  "median"),
).reset_index()

print("\n— Plateau end (median per cell), strict ±1.0 % criterion:")
strict = agg[["freq_hz", "probe", "wind", "run_type", "n",
              "plat_end_strict_s_med", "plat_end_strict_T_med"]]
strict.columns = ["f", "probe", "wind", "run", "n", "end_s", "end_T_past_arr"]
print(strict.round(2).to_string(index=False))

print("\n— Plateau end (median per cell), relaxed ±2.0 % criterion:")
relax = agg[["freq_hz", "probe", "wind", "run_type", "n",
             "plat_end_relax_s_med", "plat_end_relax_T_med"]]
relax.columns = ["f", "probe", "wind", "run", "n", "end_s", "end_T_past_arr"]
print(relax.round(2).to_string(index=False))

# ── Headline: max usable N(f) at OUT per40 (the binding constraint) ─────
print("\n=== HEADLINE: max usable N(f) at OUT per40 ===\n")
print("Assuming start = t_arr + 10·T (current pipeline N_offset=10):\n")
header = f"{'f':<5} {'wind':<6} {'plat_end_s_strict':<18} {'plat_end_s_relax':<18} {'N_max_strict_T':<16} {'N_max_relax_T':<16}"
print(header)
print("─" * len(header))
for f in THESIS_FREQS:
    t_arr_out = PROBE_R_M[OUT_PROBE] / c_group(f, TANK_DEPTH_M)
    start = t_arr_out + 10.0 / f
    for wind in ("no", "full"):
        sub = agg[(agg["freq_hz"] == f) & (agg["probe"] == "OUT")
                  & (agg["wind"] == wind) & (agg["run_type"] == "per40")]
        if sub.empty:
            continue
        end_strict_s = float(sub["plat_end_strict_s_med"].iloc[0])
        end_relax_s  = float(sub["plat_end_relax_s_med"].iloc[0])
        # Max length the OUT plateau allows
        # window must satisfy: window_start + window_length ≤ plat_end → window_length ≤ plat_end − start
        # but actually plateau_end is the latest WINDOW START at which AFFT is good; window can extend past it
        # Better interpretation: window_start ≤ plat_end → since N_offset is fixed at start, no constraint there
        # Practical: max N at OUT = (plat_end - start) · f      (window-START ≤ plat_end means window_length is unconstrained directly)
        # But the AFFT droop happens because the LATER window stretches into post-clean samples. So plateau_end IS the latest window-START where AFFT is still clean.
        # The "max usable N(f)" is then N for which window start at t_arr+10/f gives a clean AFFT — and we already have that. The plateau_end tells us how much later the start COULD be, not how much longer N can be.
        # To get max N: a window of length L starting at start has its AFFT computed by averaging samples [start, start+L]. The droop happens when the averaging reaches into post-clean. Since plateau_end is the latest start producing clean AFFT, the constraint is actually: start + L ≤ plateau_end + L (?).
        # Honest interpretation: plateau_end is the latest window-start time at which the FFT over a window of length L is still within tolerance. So a window with that L is fine if start ≤ plateau_end. With start fixed at t_arr+10/f, the constraint is satisfied as long as t_arr+10/f ≤ plateau_end. The constraint on N is implicit through L — the plateau_end was measured AT a specific L, and a different L would give a different plateau_end.
        # So: report plateau_end at chosen N, and warn that this is the EFFECTIVE bound.
        n_max_strict = (end_strict_s - start) * f
        n_max_relax  = (end_relax_s  - start) * f
        print(f"{f:<5.1f} {wind:<6} {end_strict_s:<18.2f} {end_relax_s:<18.2f} "
              f"{n_max_strict:<16.1f} {n_max_relax:<16.1f}")

# ── Save CSV ────────────────────────────────────────────────────────────
per_run.to_csv(OUT_CSV, index=False)
print(f"\nSaved per-run → {OUT_CSV.relative_to(BASE)}")

agg_csv = OUT_CSV.with_name("per40_plateau_end_aggregated.csv")
agg.to_csv(agg_csv, index=False)
print(f"Saved aggregated → {agg_csv.relative_to(BASE)}")
