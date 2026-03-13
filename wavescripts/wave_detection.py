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
from scipy.signal import welch
from scipy.optimize import brentq
from scipy import signal
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt

from wavescripts.constants import get_smoothing_window
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
        "ramp_found": None,
        "ramp_length_peaks": None,
        "keep_periods_used": None,
    }

    # ─────── finne tidsstegene ─────── 
    dt = (df["Date"].iloc[1] - df["Date"].iloc[0]).total_seconds()
    Fs = 1.0 / dt

    # ─────── hente ut input-frekvens ─────── TK bytte ut med ekte frekven senere?
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
    keep_periods= round((input_periods-13)*0.9) #trekke fra perioder, -per15- er det bare 4 gode, mens på -per40- per er ish 30 gode. TK todo velge en bedre skalering
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
        "11800": [(0.65, 4030), (0.70, 4150), (1.30, 6160), (1.60, 6700)],
        # ~12400 mm from paddle
        "12400": [(0.65, 4020), (0.70, 4250), (1.30, 6500), (1.60, 7000)],
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
        """Linear interp/extrap of start sample from calibration points."""
        fs = [p[0] for p in calib]
        ss = [p[1] for p in calib]
        if freq <= fs[0]:
            slope = (ss[1] - ss[0]) / (fs[1] - fs[0])
            return int(round(ss[0] + slope * (freq - fs[0])))
        if freq >= fs[-1]:
            slope = (ss[-1] - ss[-2]) / (fs[-1] - fs[-2])
            return int(round(ss[-1] + slope * (freq - fs[-1])))
        return int(round(float(np.interp(freq, fs, ss))))

    good_start_idx   = None
    good_end_idx     = None
    wave_upcrossings = None

    _group = _PROBE_GROUP.get(data_col)
    if _group is not None and _group in _SNARVEI_CALIB:
        good_start_idx = _snarvei_start(importertfrekvens, _SNARVEI_CALIB[_group])
        good_end_idx   = good_start_idx + int(keep_idx)
        if debug:
            print(f"[snarvei] {data_col} (group={_group}): f={importertfrekvens:.3f} Hz → "
                  f"good_start={good_start_idx}")

    """
    # PHYSICS-BASED SNARVEI (too early in practice – kept for reference / future use)
    # Deep-water dispersion: ω² = g·k  →  c = g/ω
    # omega   = 2 * np.pi * importertfrekvens
    # c_phase = PHYSICS.GRAVITY / omega           # phase velocity [m/s]  (import PHYSICS to use)
    # P1_ANCHORS = {1.3: 4700, 0.65: 3950}
    # anchor = P1_ANCHORS.get(round(importertfrekvens, 2))
    # if anchor is not None:
    #     pos_current = float(
    #         meta_row[PC.MM_FROM_PADDLE.format(i=probe_num_int)] if isinstance(meta_row, pd.Series)
    #         else meta_row[PC.MM_FROM_PADDLE.format(i=probe_num_int)].iloc[0]
    #     )
    #     pos_p1 = float(
    #         meta_row[PC.MM_FROM_PADDLE.format(i=1)] if isinstance(meta_row, pd.Series)
    #         else meta_row[PC.MM_FROM_PADDLE.format(i=1)].iloc[0]
    #     )
    #     delta_d   = (pos_current - pos_p1) / 1000       # mm → m
    #     delta_idx = int(round(delta_d / c_phase * Fs))  # travel time in samples
    #     good_start_idx = anchor + delta_idx
    #     good_end_idx   = good_start_idx + int(keep_idx)
    """
    #import sys; print('exit'); sys.exit()

    # TODO stability_skip: some 1.3 Hz runs not fully stable at snarvei start.
    # Add _STABILITY_SKIP = {1.3: 2, 0.65: 0} (periods) applied before section 1.c.
    # Future: replace with autocorrelation-based stability detection.
    # good_start_idx += _STABILITY_SKIP.get(round(importertfrekvens,2), 0) * samples_per_period
    # good_end_idx    = good_start_idx + int(keep_idx)

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

    """
    # WALK-FORWARD APPROACH (failed on windy signals – chain breaks after 1 period)
    # Kept for reference; may be useful for individual period boundary detection later.
    # tol = 0.10
    # min_period_samp = int((1 - tol) * samples_per_period)
    # max_period_samp = int((1 + tol) * samples_per_period)
    # valid = [first_uc]
    # prev  = first_uc
    # for uc in subsequent:
    #     gap = int(uc) - prev
    #     if min_period_samp <= gap <= max_period_samp:
    #         valid.append(int(uc))
    #         prev = int(uc)
    #         if len(valid) - 1 >= n_periods_target:
    #             break
    # good_start_idx = valid[0]; good_end_idx = valid[-1]
    """

    # ==========================================================
    # 1.d  Mstop warning: check if good_end_idx falls inside the post-stop window
    # ==========================================================
    # "Extra seconds" (mstop) = recording time after wavemaker stops.
    # A far probe may not receive the full wave train within this window.
    _mstop_raw = (
        meta_row.get("Extra seconds", None) if isinstance(meta_row, pd.Series)
        else (meta_row["Extra seconds"].iloc[0] if "Extra seconds" in meta_row.columns else None)
    )
    mstop_sec     = float(_mstop_raw) if _mstop_raw is not None else 0.0
    mstop_samples = int(mstop_sec * Fs)
    signal_length = len(signal_smooth)

    if mstop_samples > 0 and good_end_idx is not None:
        cutoff_idx = signal_length - mstop_samples   # sample where wavemaker stopped
        if good_end_idx > cutoff_idx:
            overlap = good_end_idx - cutoff_idx
            print(f"  WARNING [{data_col}]: good_end_idx ({good_end_idx}) is {overlap} samples "
                  f"({overlap/Fs:.1f} s) into the mstop window. "
                  f"Probe may be missing the tail of the wave group.")
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
    
    """
    Først: sjekke paneltilstand:
        hvis ingen panel
    
    Neste: sjekke vindforhold: 
        hvis sterk vind
        
    Hvis lav frekvens: da er det kortere (nesten ingen) ramp.
    
    Ramp må tape for høyeste peaks, i hvertfall når panel
        
    Så, enkelt basere probe 2 på 1 , og 34 på 2?
    """
    
    """elif (meta_row["WindCondition"]) == "lowest" and input_freq == 1.3:
        print('lowestwind og 1.3')
        if data_col == "Probe 1": 
                good_start_idx = P1amp01frwq13eyeball 
                good_end_idx = good_start_idx+keep_idx
                #return good_start_idx, good_end_idx, debug_info
        elif data_col == "Probe 2" : 
                good_start_idx = P2handcalc
                good_end_idx = P2handcalc + keep_idx
                #return good_start_idx, good_end_idx, debug_info
        elif data_col == "Probe 3" : 
                good_start_idx = P3handcalc
                good_end_idx = P3handcalc + keep_idx
                #return good_start_idx, good_end_idx, debug_info
        elif data_col == "Probe 4" : 
                good_start_idx = P3handcalc
                good_end_idx = P3handcalc + keep_idx
                #return good_start_idx, good_end_idx, debug_info"""

#TODO: claude sitt forslag til bruk av konstanter
    # def detect_baseline_AFTER(signal_smooth):
    #     """NEW VERSION."""
    #     baseline_samples = int(SIGNAL.BASELINE_DURATION_SEC * MEASUREMENT.SAMPLING_RATE)
    #     baseline = signal_smooth[:baseline_samples]
    #     baseline_mean = np.mean(baseline)
    #     baseline_std = np.std(baseline)
    #     threshold = baseline_mean + SIGNAL.BASELINE_SIGMA_FACTOR * baseline_std
    #     return threshold
    baseline_seconds = 2# testkommentar
    sigma_factor=1.0
    skip_periods=None

    min_ramp_peaks=5
    max_ramp_peaks=15
    max_dips_allowed=2
    min_growth_factor = 1.015
    #import sys; print('exit'); sys.exit()

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
    #import sys; print('exit'); sys.exit()
    
    """ærbe
    #print('threshold verdi: ', threshold)
    #print('eexit')
    #print('='*99)
    
    #voltgrense  = meta_row["WaveAmplitudeInput [Volt]"]
    #grense = baseline_mean + (voltgrense*100)/3
    #input volt på 0.1 gir omtrentelig <10mm amplitude.
    #import sys; sys.exit()
    """
    
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
    
    #TODO
    # ==========================================================
    # ikke i bruk, TK , endret sigmafaktor i staden.
    # ==========================================================
    """
    TODO: LAGE noe som fanger opp de med 15 perioder.
    for der er signalet veldig kort.
    lettest: FANGE OPP DE 10 største bølgene. starte fra den første.

    """
    """
    # ta 10 største, så ta den første, så ta 10 perioder
    numbaofpeaks = len(peaks)
    if meta_row["WavePeriodInput"] <16 and numbaofpeaks >3 :
        
        largest_ampl = np.abs(signal_smooth[peaks])
        kth = 3
        largest_peaks = np.argpartition(peaks, kth)
        good_start_idx = largest_peaks[0]
        good_end_idx = largest_peaks[-1]
        debug_info = None
        print("="*99)
        print(f'LESS THAN 16 periods, choosing largest peaks')
        print("="*99)
        return good_start_idx, good_end_idx, debug_info
    
    """
    #
    # ==========================================================
    #  
    # ==========================================================
    """
    if len(peaks) < min_ramp_peaks + 3:
        print("Not enough peaks detected – falling back to legacy method")
        skip_periods = skip_periods or (5 + 12)
        keep_periods = keep_periods or 5
        good_start_idx = first_motion_idx + int(skip_periods * samples_per_period)
        good_range = int(keep_periods * samples_per_period)
        good_range = min(good_range, len(df) - good_start_idx)
        good_end_idx = good_start_idx + good_range
        return good_start_idx, good_end_idx, {}

    peak_amplitudes = np.abs(signal_smooth[peaks])

    # ==========================================================
    # 4. Ramp-up detection: nearly monotonic increase with dips
    # ==========================================================
    def find_best_ramp(seq, min_len=5, max_len=15, max_dips=2, min_growth=2.0):
        n = len(seq)
        best_start = best_end = -1
        best_len = 0
        best_dips = 99

        for length in range(min_len, min(max_len + 1, n)):
            for start in range(n - length + 1):
                end = start + length - 1
                sub = seq[start:end + 1]

                dips = sum(1 for i in range(1, len(sub)) if sub[i] <= sub[i - 1])
                growth_ok = sub[-1] >= sub[0] * min_growth

                if dips <= max_dips and growth_ok:
                    if length > best_len or (length == best_len and dips < best_dips):
                        best_start, best_end = start, end
                        best_len = length
                        best_dips = dips

        if best_start == -1:
            return None
        return best_start, best_end, seq[best_start:best_end + 1]
    #TODO
    CLAUDE
    def find_best_ramp_AFTER(seq):
        # All parameters come from constants
        return _find_ramp_core(
            seq,
            min_len=RAMP.MIN_RAMP_PEAKS,
            max_len=RAMP.MAX_RAMP_PEAKS,
            max_dips=RAMP.MAX_DIPS_ALLOWED,
            min_growth=RAMP.MIN_GROWTH_FACTOR
    # ==========================================================
    # Kjører "find_best_ramp(...)"
    # ==========================================================
    ramp_result = find_best_ramp(
        peak_amplitudes,
        min_len=min_ramp_peaks,
        max_len=max_ramp_peaks,
        max_dips=max_dips_allowed,
        min_growth=min_growth_factor
    )
    
    
    if ramp_result is None:
        print("No clear ramp-up found – using legacy timing")
        skip_periods = skip_periods or (5 + 12)
        keep_periods = keep_periods or 8
        good_start_idx = first_motion_idx + int(skip_periods * samples_per_period)
    else:
        ramp_start_peak_idx, ramp_end_peak_idx, ramp_seq = ramp_result
        print(f"RAMP-UP DETECTED: {len(ramp_seq)} peaks, "
              f"from {ramp_seq[0]:.2f} → {ramp_seq[-1]:.2f} (x{ramp_seq[-1]/ramp_seq[0]:.1f})")
        # Convert last ramp peak → sample index
        last_ramp_sample_idx = peaks[ramp_end_peak_idx]
        # Stable phase starts right after ramp-up
        good_start_idx = last_ramp_sample_idx + samples_per_period // 4  # small safety margin
        keep_periods = keep_periods or 10

    # Final stable window - nå tar den start+range=end
    good_range     = int(keep_periods * samples_per_period)
    good_start_idx = min(good_start_idx, len(df) - good_range - 1)
    good_range     = min(good_range, len(df) - good_start_idx)
    good_end_idx   = good_start_idx + good_range
    """
    # ==========================================================
    # 5.a Få hjelp av grok
    # ==========================================================
    # for å printe verdiene slik at grok kunne forstå signalet
    # if 'dumped' not in locals():   # only once
        # print("\n=== SIGNAL FOR GROK (downsampled 200:1) ===")
        # print("value")
        # print("\n".join(f"{x:.5f}" for x in df[data_col].values[::200]))  # every 200th point
        # print("=== FULL STATS ===")
        # print(f"total_points: {len(df)}")
        # print(f"dt_sec: {dt:.6f}")
        # print(f"frequency_hz: {importertfrekvens}")
        # print(f"samples_per_period: {samples_per_period}")
        # print(f"baseline_mean: {baseline_mean:.4f}")
        # print(f"baseline_std: {baseline_std:.4f}")
        # print(f"first_motion_idx: {first_motion_idx}")
        # print("=== PEAKS (index, value) ===")
        # peaks_abs = np.abs(signal_smooth[peaks]) if 'peaks' in locals() and len(peaks)>0 else []
        # for i, (pidx, pval) in enumerate(zip(peaks[:30], peaks_abs[:30])):  # max 30 peaks
            # print(f"{pidx:5d} -> {pval:.5f}")
        # if len(peaks) > 30:
            # print("... (more peaks exist)")
        # print("=== END – COPY ALL ABOVE AND SEND TO GROK ===")
        # dumped = True
        #import sys; sys.exit(0)
        
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
        #"ramp_found": ramp_result is not None,
        #"ramp_length_peaks": len(ramp_result[2]) if ramp_result else None,
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
