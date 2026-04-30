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
from wavescripts.improved_data_loader import update_processed_metadata
from scipy.signal import find_peaks
from scipy import signal
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt

from wavescripts.constants import SIGNAL, RAMP, MEASUREMENT, HG, hg_window_for_probe, get_smoothing_window
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

    # Per<40 runs are too short for the new H&G [arrival + 7T, +17T] window.
    # Decision (2026-04-30): per40 is the analysis target; per80 and per240
    # are kept; anything shorter (per15, per20, per30) is skipped → AFFT NaN.
    if pd.notna(input_periods) and float(input_periods) < 40:
        if debug:
            print(f"[H&G] {data_col}: per{int(input_periods)} < 40 — skip "
                  f"(window does not fit short runs).")
        return None, None, None

    # ==========================================================
    # 1.b  Probe arrival-anchored Huseby–Grue window (pipeline standard 2026-04-30)
    # ==========================================================
    # Per-probe window starting N_OFFSET periods past wave arrival at the
    # probe, spanning N_LENGTH periods:
    #
    #     T_start = r_probe·f / c_group(f, depth) + N_OFFSET   [periods]
    #     T_end   = T_start + N_LENGTH
    #
    # N_OFFSET = 7, N_LENGTH = 10 uniform across all thesis frequencies.
    # Math + constants live in wavescripts/constants.py :: HG + c_group() +
    # hg_window_for_probe().  Old eyeballed SNARVEI calibration archived as
    # SNARVEI_ARCHIVE_START / _END in the same file.

    # r_probe in metres — parse from the probe column name "Probe DIST/LAT".
    try:
        _r_probe_m = int(data_col.split(" ", 1)[1].split("/")[0]) / 1000.0
    except (IndexError, ValueError):
        _r_probe_m = None

    good_start_idx   = None
    good_end_idx     = None
    wave_upcrossings = None
    n_found          = 0
    n_periods_target = HG.N_LENGTH

    if _r_probe_m is not None:
        _start_T, _end_T = hg_window_for_probe(_r_probe_m, importertfrekvens)
        _start_sample = int(round(_start_T * samples_per_period))
        _end_sample   = int(round(_end_T   * samples_per_period))

        if _start_sample < 0 or _end_sample > len(signal_smooth):
            if debug:
                print(f"[H&G] {data_col}: window [{_start_T:.1f}T, {_end_T:.1f}T] "
                      f"= samples [{_start_sample}, {_end_sample}] "
                      f"does not fit signal length {len(signal_smooth)} — skip.")
        else:
            good_start_idx = _start_sample
            good_end_idx   = _end_sample
            n_found        = int(round((_end_sample - _start_sample) / samples_per_period))
            if debug:
                print(f"[H&G] {data_col} (r={_r_probe_m:.3f} m, f={importertfrekvens:.3f} Hz) "
                      f"→ window [{_start_T:.2f}T, {_end_T:.2f}T] "
                      f"= samples [{good_start_idx}, {good_end_idx}]")


    # ==========================================================
    # 1.c  Upcrossings on signal_smooth (raw ULS signal)
    # ==========================================================
    # Detect first, so we can both snap the H&G window (section 1.c-snap) and
    # restrict to the in-window subset for quality metrics (section 1.c-filter).
    #
    # Use per-run DC mean of the first 2 s as the upcrossing threshold. The
    # global stillwater can differ from the run-local DC level by 0.1–0.2 mm;
    # the local baseline is always centred on the actual signal so the first
    # upcrossing is found reliably.
    _baseline_n   = int(2 * Fs)
    upcross_level = float(np.mean(signal_smooth[:_baseline_n]))
    above_still     = signal_smooth > upcross_level
    all_upcrossings = np.where((~above_still[:-1]) & above_still[1:])[0] + 1

    # ── 1.c-snap  Snap H&G window to zero-upcrossings at both endpoints ──
    # Physics first: every measurement window is a signal-complete integer
    # number of wave cycles, bounded by two detected zero-upcrossings. This
    # eliminates the int(round(Fs/f)) window-length quantization (0.024 T at
    # 1.4 Hz) and delivers perfectly coherent FFT sampling (sinc leakage = 0).
    #
    # Under fullwind the upcrossings jitter (±4 samples per cycle), so the
    # 10th-upcrossing end has some uncertainty too — but the jitter is zero-
    # mean, so 10-cycle-averaged window length matches the true period mean.
    # See memory/methodology_detector_jitter_and_noise_floor.md.
    #
    # The snap is on the RAW ULS signal, so both endpoints are raw-upcrossings
    # = eta-downcrossings (raw distance ↓ ⇔ elevation ↑). Either orientation
    # of crossing delimits integer cycles, which is what FFT/LS care about.
    #
    # Diagnostics retained:
    #   hg_expected_start / hg_expected_end  : pre-snap theoretical window
    #   hg_snap_shift_samples                : signed shift applied to start
    # End-snap amount is derivable from
    #   (good_end_idx − good_start_idx) − n_periods_target·samples_per_period
    hg_expected_start = good_start_idx
    hg_expected_end   = good_end_idx
    hg_snap_shift_samples: int | None = None

    if (
        len(all_upcrossings) > 0
        and good_start_idx is not None
        and good_end_idx is not None
    ):
        _search_halfwidth = samples_per_period      # ±1 full period
        _lo = good_start_idx - _search_halfwidth
        _hi = good_start_idx + _search_halfwidth
        _candidates = all_upcrossings[(all_upcrossings >= _lo) & (all_upcrossings <= _hi)]

        if len(_candidates) > 0:
            _snap_to = int(_candidates[np.argmin(np.abs(_candidates - good_start_idx))])
            hg_snap_shift_samples = int(_snap_to - good_start_idx)
            good_start_idx = _snap_to

            # End-snap: find the n_periods_target-th upcrossing after start.
            # `n_periods_target` = HG.N_LENGTH = 10 cycles.
            # `_snap_to` is itself an upcrossing, so the i-th cycle ends at
            # the upcrossing with offset i in all_upcrossings from _snap_to.
            _start_uc_idx = np.where(all_upcrossings == _snap_to)[0][0]
            _end_uc_idx   = _start_uc_idx + n_periods_target

            # Sanity guard: the N-th UC should land within ±0.5 period of
            # the theoretical N·T position. Under legitimate detector jitter
            # (±4 samples per cycle, documented) the accumulated end offset
            # is √10·4 ≈ 13 samples ≈ 0.07 T — well inside 0.5 T. Anything
            # beyond 0.5 T means the detector found spurious upcrossings
            # (most commonly at 1.3 Hz × 0.1 V × fullwind where wind chop
            # creates extra near-zero crossings). In that case fall back to
            # fixed length — the amplitude metrics will still be reasonable
            # on a 10 T window even with the 0.024 T quantization residual.
            _fixed_end = good_end_idx + hg_snap_shift_samples
            _max_end_shift = samples_per_period // 2

            if _end_uc_idx < len(all_upcrossings):
                _uc_end = int(all_upcrossings[_end_uc_idx])
                if abs(_uc_end - _fixed_end) <= _max_end_shift:
                    good_end_idx = _uc_end
                else:
                    if debug:
                        print(f"[H&G snap-end] {data_col}: 10th UC at "
                              f"{_uc_end} is {(_uc_end - _fixed_end)/samples_per_period:+.2f} T "
                              f"from theoretical end — likely spurious "
                              f"wind-chop crossings. Falling back to fixed length.")
                    good_end_idx = _fixed_end
            else:
                # Fewer than n_periods_target upcrossings after start —
                # fall back to fixed length so the window still exists.
                if debug:
                    print(f"[H&G snap-end] {data_col}: only "
                          f"{len(all_upcrossings) - _start_uc_idx - 1} upcrossings "
                          f"past start, expected {n_periods_target}; "
                          f"falling back to fixed-length end.")
                good_end_idx = _fixed_end

            # Re-check fit in signal after shift
            if good_start_idx < 0 or good_end_idx > len(signal_smooth):
                if debug:
                    print(f"[H&G snap] {data_col}: shifted window falls off signal "
                          f"(start_shift={hg_snap_shift_samples}); reverting to expected.")
                good_start_idx = hg_expected_start
                good_end_idx   = hg_expected_end
                hg_snap_shift_samples = None
            elif debug:
                _win_len = good_end_idx - good_start_idx
                print(f"[H&G snap] {data_col}: "
                      f"start {hg_snap_shift_samples:+d} samples "
                      f"({hg_snap_shift_samples/samples_per_period:+.3f} T); "
                      f"window length = {_win_len} samples "
                      f"({_win_len/samples_per_period:.3f} T)")

    # ── 1.c-filter  Upcrossings within the final (post-snap) window ──────
    # Used by processor.py for per-run quality metrics (wave_stability,
    # period_amplitude_cv, (cycles)/(phase) amplitudes).
    if len(all_upcrossings) > 0 and good_start_idx is not None and good_end_idx is not None:
        wave_upcrossings = all_upcrossings[
            (all_upcrossings >= good_start_idx) & (all_upcrossings <= good_end_idx)
        ]


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
    mstop_sec_tag = 0.0 if (np.isnan(_mstop_float) or np.isinf(_mstop_float)) else _mstop_float
    signal_length = len(signal_smooth)

    # mstop from filename is manually typed and may exceed the actual recording.
    # Clamp to the signal actually available after good_end so warnings are honest.
    if mstop_sec_tag > 0 and good_end_idx is not None:
        actual_post_end_sec = max(0.0, (signal_length - good_end_idx) / Fs)
        if mstop_sec_tag > actual_post_end_sec + 1.0:  # +1 s tolerance
            print(f"  NOTE [{data_col}]: filename mstop={mstop_sec_tag:.0f} s but only "
                  f"{actual_post_end_sec:.0f} s available after analysis window — "
                  f"recording was cut short.")
        mstop_sec = min(mstop_sec_tag, actual_post_end_sec)
    else:
        mstop_sec = mstop_sec_tag
    mstop_samples = int(mstop_sec * Fs)

    if mstop_samples > 0 and good_end_idx is not None:
        # Only warn when periods are actually missing — sitting inside the mstop
        # tail is normal for short runs and is not itself a problem.
        if n_found < n_periods_target:
            print(f"  WARNING [{data_col}]: only {n_found}/{n_periods_target} periods found – "
                  f"signal may be cut short (mstop={mstop_sec:.0f} s actual, "
                  f"probe at {meta_row[PC.MM_FROM_PADDLE.format(i=probe_num_int)]:.0f} mm from paddle).")

    # No fallback: if the H&G window does not fit the signal (too-short run, or
    # out-of-pipeline probe at unusual distance), good_start_idx / good_end_idx
    # remain None. Downstream code (_extract_probe_signal) returns None for that
    # probe; its AFFT / time-domain amplitude are NaN in meta.json for that run.
    good_range = (good_end_idx - good_start_idx) if (good_end_idx is not None and good_start_idx is not None) else 0

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
        # H&G snap diagnostics (section 1.c-snap). `hg_expected_*` is the
        # theoretical probe-shifted H&G window (before upcrossing snap);
        # the returned good_start_idx / good_end_idx reflect the snapped
        # position. `hg_snap_shift_samples` is their signed difference
        # (None if no upcrossing found in ±1T or the snap fell off signal).
        "hg_expected_start":           hg_expected_start,
        "hg_expected_end":             hg_expected_end,
        "hg_snap_shift_samples":       hg_snap_shift_samples,
        # End-snap amount is derivable from
        #   (good_end_idx − good_start_idx) − n_periods_target·samples_per_period
        # so no separate sample-level diagnostic is returned.
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
