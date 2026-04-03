#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 09:44:49 2026

@author: ole
"""

#wave_detection.py



from pathlib import Path
import pandas as pd
import numpy as np
from wavescripts.improved_data_loader import update_processed_metadata, PROBE_CONFIGS
from scipy.signal import find_peaks
from scipy import signal
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt

from wavescripts.constants import SIGNAL, RAMP, MEASUREMENT, get_smoothing_window
from wavescripts.constants import (
    ProbeColumns as PC,
    GlobalColumns as GC,
    ColumnGroups as CG,
    CalculationResultColumns as RC
)


def find_wave_range(
    df: pd.DataFrame,
    meta_row: pd.DataFrame,  # metadata for selected files
    data_col: str,
    probe_num: int,           # physical probe number (1-4) for stillwater lookup
    detect_win: int,
    range_plot: bool = False,
    debug: bool = False,
) -> Tuple[int, int, dict[str, Any]] :
    """

    Finner waverange.

    Args:
        utvalgt signal, tilhørende metadatarad, Probe {i}, detect-vindu

    Toggle:x
        Smoothing Window, Range-plot, Debug

    Returns:
        good_start_idx, good_end_idx, debug_info

    Raises:
        ?Error: legg til
    """

    wind_condition = meta_row["WindCondition"]
    detect_win = detect_win if detect_win is not None else get_smoothing_window(wind_condition)
    # ==========================================================
    # 1. smoothe signalet med moving average vindu: detect_win
    # ==========================================================
    signal_smooth = (
        df[data_col]
        .rolling(window=detect_win, center=True, min_periods=1)
        .mean()
        .bfill().ffill()
        .values
    )

    debug_info = {
        "baseline_mean": None,
        "baseline_std": None,
        "first_motion_idx": None,
        "samples_per_period": None,
        "detected_peaks": None,
        "keep_periods_used": None,
    }

    # ─────── finne tidsstegene ───────
    dt = (df["Date"].iloc[1] - df["Date"].iloc[0]).total_seconds()
    Fs = 1.0 / dt

    # ─────── hente ut input-frekvens ───────
    input_freq = meta_row["WaveFrequencyInput [Hz]"] if isinstance(meta_row, pd.Series) else meta_row["WaveFrequencyInput [Hz]"].iloc[0]
    importertfrekvens = float(input_freq)
    if pd.isna(input_freq):
        print("no freq found, assuming no wave")
        good_start_idx = 0
        good_end_idx = len(df)
        debug_info = None
        return good_start_idx, good_end_idx, debug_info

    samples_per_period = int(round(Fs / importertfrekvens))
    probe_num_int = probe_num  # physical probe number for stillwater lookup

    # ─────── velge antall perioder ───────
    input_periods = (meta_row["WavePeriodInput"])
    keep_periods= round((input_periods-13)*0.9) # empirical: (input_periods - 13) * 0.9; per15→2, per40→24
    keep_seconds= keep_periods/input_freq
    keep_idx = keep_seconds*250 # 1 sek = 250 målinger
    good_range = keep_idx

    # ==========================================================
    #  1.b  Snarvei: calibrated per-probe anchors, interpolated across frequency
    # ==========================================================
    # Calibration points: eyeballed good-start sample indices at specific frequencies.
    # All laterals at the same longitudinal distance share the same arrival timing.
    # Add more points by eyeballing the RampDetectionBrowser — more points = better fit.
    #
    # Format: list of (freq_hz, start_sample) sorted by frequency.
    # Interpolation: linear between calibrated points; linear extrapolation beyond range.
    # Samples at 250 Hz (ms / 4).
    #
    # TODO: re-eyeball and add more calibration points, especially for intermediate freqs.
    _SNARVEI_CALIB = {
        # ~8800 mm from paddle
        "8804":  [(0.65, 3975), (1.30, 4700), (1.80, 6000)],
        # ~9373 mm from paddle
        "9373":  [(0.65, 4075), (0.70, 3750), (1.30, 4800), (1.60, 5500)],
        # ~11800 mm from paddle (march2026_rearranging config, 4–6 Mar 2026 only)
        # Values interpolated from 9373 and 12400 at distance fraction 0.802 — eyeball-refine
        # in RampDetectionBrowser once confirmed.
        # 1.70/1.80: estimated, needs eyeballing
        "11800": [(0.65, 4030), (0.70, 4150), (1.30, 6160), (1.60, 6700), (1.70, 6700), (1.80, 6650)],
        # ~12400 mm from paddle
        # 1.60: reduced from 7000 → 6700 (~2 periods earlier, artifact at tail was misleading)
        # 1.70: estimated, needs eyeballing in RampDetectionBrowser
        # 1.80: extrapolated ~7350 minus 4 periods (4×139=556) → 6800 — verify in browser
        "12400": [(0.65, 4020), (0.70, 4250), (1.30, 6500), (1.60, 6700), (1.70, 6750), (1.80, 6800)],
    }

    # Map every probe column name to a distance group — auto-generated from PROBE_CONFIGS
    # so this never goes stale when distances are corrected in improved_data_loader.py.
    # Keys are "Probe dist/lat" strings; values are the distance prefix used as
    # the _SNARVEI_CALIB key (e.g. "Probe 12400/250" → "12400").
    _PROBE_GROUP = {
        f"Probe {pos}": pos.split("/")[0]
        for cfg in PROBE_CONFIGS
        for pos in cfg.probe_col_names().values()
    }

    def _snarvei_start(freq: float, calib: list[tuple[float, int]]) -> int:
        """Polynomial interp (deg 2) of start sample; linear extrap beyond calibrated range."""
        fs = np.array([p[0] for p in calib])
        ss = np.array([p[1] for p in calib], dtype=float)
        deg = min(2, len(calib) - 1)
        poly  = np.poly1d(np.polyfit(fs, ss, deg))
        dpoly = poly.deriv()
        if freq <= fs[0]:
            return int(round(float(poly(fs[0])) + float(dpoly(fs[0])) * (freq - fs[0])))
        if freq >= fs[-1]:
            return int(round(float(poly(fs[-1])) + float(dpoly(fs[-1])) * (freq - fs[-1])))
        return int(round(float(poly(freq))))

    good_start_idx   = None
    good_end_idx     = None
    wave_upcrossings = None

    # How many periods to trim from each end of the snarvei window.
    # Start trim: removes the ramp-exit transition period(s) still building to full amplitude.
    # End trim:   removes the wavemaker deceleration / mstop onset period(s).
    # These are applied to the snarvei reference BEFORE the upcrossing snap,
    # so wave_upcrossings, debug_info, and the range_plot all see the same trimmed window.
    # High-frequency runs (≥1.6 Hz) need more trimming at both ends:
    # — start: ramp transition is slower, more buildup bleeds in (+2 extra)
    # — end:   ramp-down begins earlier and swell tail is longer (+3 extra)
    _TRIM_START_PERIODS = (RAMP.TRIM_START_PERIODS_HIGH_FREQ if importertfrekvens >= RAMP.HIGH_FREQ_TRIM_HZ
                           else RAMP.TRIM_START_PERIODS_DEFAULT)
    _TRIM_END_PERIODS   = (RAMP.TRIM_END_PERIODS_HIGH_FREQ   if importertfrekvens >= RAMP.HIGH_FREQ_TRIM_HZ
                           else RAMP.TRIM_END_PERIODS_DEFAULT)

    _group = _PROBE_GROUP.get(data_col)
    if _group is not None and _group in _SNARVEI_CALIB:
        good_start_idx  = _snarvei_start(importertfrekvens, _SNARVEI_CALIB[_group])
        good_start_idx += _TRIM_START_PERIODS * samples_per_period
        good_end_idx    = good_start_idx + int(keep_idx) - _TRIM_END_PERIODS * samples_per_period
        if debug:
            print(f"[snarvei] {data_col} (group={_group}): f={importertfrekvens:.3f} Hz → "
                  f"good_start={good_start_idx} (trim_start={_TRIM_START_PERIODS}p, "
                  f"trim_end={_TRIM_END_PERIODS}p)")


    # ==========================================================
    # 1.c  Snap start and end independently to nearest zero-upcrossing
    # ==========================================================
    # Foundation: snarvei gives the approximate start; WavePeriodInput gives the duration.
    # Both endpoints are snapped to the nearest stillwater upcrossing independently.
    # This is robust for windy signals – no chain-walking required.

    # Use per-run DC mean of the first 2 s as the upcrossing threshold.
    # The global stillwater can differ from the run-local DC level by 0.1–0.2 mm,
    # which is enough to delay the first upcrossing by several seconds when waves are
    # small (e.g. ramp-up phase of nowind runs). The local baseline is always centred
    # on the actual signal, so the first upcrossing is found reliably.
    _baseline_n    = int(2 * Fs)
    upcross_level  = float(np.mean(signal_smooth[:_baseline_n]))

    # All zero-upcrossings in the full smoothed signal
    above_still     = signal_smooth > upcross_level
    all_upcrossings = np.where((~above_still[:-1]) & above_still[1:])[0] + 1

    n_periods_target = max(5, int(keep_periods))
    n_found          = 0

    if len(all_upcrossings) == 0:
        if debug:
            print(f"[find_wave_range] {data_col}: no upcrossings found in signal")
    else:
        ref_start = good_start_idx if good_start_idx is not None else int(2 * samples_per_period)

        # 1. Snap start: nearest upcrossing to snarvei guess
        refined_start = int(all_upcrossings[np.argmin(np.abs(all_upcrossings - ref_start))])

        # 2. Expected end from WavePeriodInput: start + n_periods_target full periods
        expected_end  = min(refined_start + int(n_periods_target * samples_per_period),
                            len(signal_smooth) - 1)

        # 3. Snap end: nearest upcrossing to expected end
        refined_end   = int(all_upcrossings[np.argmin(np.abs(all_upcrossings - expected_end))])

        n_found = int(round((refined_end - refined_start) / samples_per_period))

        good_start_idx   = refined_start
        good_end_idx     = refined_end
        good_range       = good_end_idx - good_start_idx
        wave_upcrossings = all_upcrossings[
            (all_upcrossings >= refined_start) & (all_upcrossings <= refined_end)
        ]

        if debug:
            print(f"[find_wave_range] {data_col}: snarvei→{ref_start}, "
                  f"start={refined_start}, end={refined_end}, "
                  f"periods≈{n_found}/{n_periods_target}")


    # ==========================================================
    # 1.d  Mstop warning: check if good_end_idx falls inside the post-stop window
    # ==========================================================
    # "Extra seconds" (mstop) = recording time after wavemaker stops.
    # A far probe may not receive the full wave train within this window.
    _mstop_raw = (
        meta_row.get("Extra seconds", None) if isinstance(meta_row, pd.Series)
        else (meta_row["Extra seconds"].iloc[0] if "Extra seconds" in meta_row.columns else None)
    )
    _mstop_float  = float(_mstop_raw) if _mstop_raw is not None else 0.0
    mstop_sec     = 0.0 if (np.isnan(_mstop_float) or np.isinf(_mstop_float)) else _mstop_float
    mstop_samples = int(mstop_sec * Fs)
    signal_length = len(signal_smooth)

    if mstop_samples > 0 and good_end_idx is not None:
        # Only warn when periods are actually missing — sitting inside the mstop
        # tail is normal for short runs and is not itself a problem.
        if n_found < n_periods_target:
            print(f"  WARNING [{data_col}]: only {n_found}/{n_periods_target} periods found – "
                  f"signal may be cut short (mstop={mstop_sec:.0f} s, "
                  f"probe at {meta_row[PC.MM_FROM_PADDLE.format(i=probe_num_int)]:.0f} mm from paddle).")

    # Safety fallback if nothing was set (snarvei miss + no upcrossing found)
    if good_start_idx is None:
        good_start_idx = int(2 * samples_per_period)
        good_end_idx   = good_start_idx + int(keep_idx)
        good_range     = good_end_idx - good_start_idx

    #fullpanel-fullwind-amp02-freq13- correct @5780
    # no panel, amp03, freq0650: 2300? probe=??
    #fullpanel-fullwind-amp01-freq0650-per15-probe3: 4000 korrekt


    baseline_seconds = 2
    sigma_factor = 1.0
    skip_periods = None

    # ==========================================================
    # 2. Baseline & first motion (still useful for rough start)
    # ==========================================================
    baseline_samples = int(baseline_seconds * Fs)
    baseline = signal_smooth[:baseline_samples]
    baseline_mean = np.mean(baseline)
    baseline_std = np.std(baseline)
    threshold = baseline_mean + sigma_factor*baseline_std

    if debug:
        print('baselines:')
        print(f'_samples: {baseline_samples}, _mean: {baseline_mean}, _seconds {baseline_seconds}, _std {baseline_std}')
    above_noise = signal_smooth > threshold

    first_motion_idx = np.argmax(above_noise) if np.any(above_noise) else 0

    # ==========================================================
    # 3. Peak detection on absolute signal (handles both positive/negative swings)
    # ==========================================================
    # Use prominence and distance tuned to your frequency
    min_distance = max(3, input_periods *0.9 )  # at least 0.9 period apart
    peaks, properties = find_peaks(
        np.abs(signal_smooth),
        distance=min_distance,
        prominence=3 * baseline_std,  # ignore noise peaks
        height=threshold
    )


    # ==========================================================
    # 5.b) Plotting – safe version that works with your current plot_ramp_detection
    # ==========================================================
    if range_plot:
            from wavescripts.plotter import plot_ramp_detection

            # Build kwargs only with arguments your current function actually accepts
            plot_kwargs = {
                "df": df,
                "meta_sel": meta_row,
                "data_col": data_col,
                "signal": signal_smooth,
                "baseline_mean": baseline_mean,
                "threshold": threshold,
                "first_motion_idx": first_motion_idx,
                "good_start_idx": good_start_idx,
                "good_range": good_range,
                "good_end_idx": good_end_idx,
                "title": f"Smart Ramp Detection – {data_col}"
            }

            # Only add new arguments if we have them and ramp was found
            """
            if 'peaks' in locals() and ramp_result is not None:
                plot_kwargs["peaks"] = peaks
                plot_kwargs["peak_amplitudes"] = peak_amplitudes
                ramp_peak_samples = peaks[ramp_result[0]:ramp_result[1]+1]
                plot_kwargs["ramp_peak_indices"] = ramp_peak_samples
            """
            try:
                fig, ax = plot_ramp_detection(**plot_kwargs)
                plt.show()
            except Exception as e:
                import traceback
                print("plot failed fordi:", e)
                traceback.print_exc()
                print(f"Plot failed (will work after you update plotter): {e}")

    debug_info = {
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "first_motion_idx": first_motion_idx,
        "samples_per_period": samples_per_period,
        "detected_peaks": len(peaks),
        "keep_periods_used": keep_periods,
        "n_periods_target": n_periods_target,
        "n_periods_found": n_found,
        "wave_upcrossings": wave_upcrossings,   # array of period-start indices; last = end of final period
    }

    return good_start_idx, good_end_idx, debug_info


def find_first_arrival(
    signal: np.ndarray,
    noise_floor_mm: float,
    fs: float = 250.0,
    threshold_factor: float = 2.0,
    window_s: float = 0.5,
) -> tuple[int | None, float | None]:
    """Detect the first sample where wave energy exceeds the stillwater noise floor.

    Uses a rolling (P97.5 - P2.5) / 2 amplitude in a short sliding window —
    the same definition as the pipeline amplitude — and finds the first window
    whose amplitude exceeds threshold_factor * noise_floor_mm.

    Args:
        signal:            1-D array of probe elevation [mm], already zeroed.
        noise_floor_mm:    Stillwater noise amplitude for this probe [mm]
                           (mean of 'Probe {pos} Amplitude' across stillwater runs).
        fs:                Sampling rate [Hz]. Default 250.
        threshold_factor:  Detection threshold = threshold_factor × noise_floor.
                           2.0 means "twice the stillwater noise". Default 2.0.
        window_s:          Rolling window length [s]. Default 0.5 s (125 samples).

    Returns:
        (arrival_idx, arrival_s): sample index and time [s] of first detection,
        or (None, None) if signal never exceeds the threshold.
    """
    threshold = threshold_factor * noise_floor_mm
    win = max(1, int(round(window_s * fs)))
    n = len(signal)

    for i in range(0, n - win + 1):
        chunk = signal[i : i + win]
        amp = (np.nanpercentile(chunk, 97.5) - np.nanpercentile(chunk, 2.5)) / 2.0
        if amp >= threshold:
            return i, i / fs

    return None, None
