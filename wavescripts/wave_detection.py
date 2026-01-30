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
from wavescripts.data_loader import update_processed_metadata
from scipy.signal import find_peaks
from scipy.signal import welch
from scipy.optimize import brentq
from typing import Dict, List, Tuple


def find_wave_range(
    df,
    meta_row,  # metadata for selected files
    data_col,
    detect_win,
    range_plot: bool,
    debug: bool,
):
    
    if (meta_row["WindCondition"]) == "full":
        detect_win = 15
    elif (meta_row["WindCondition"]) == "low":
        detect_win = 10
    elif (meta_row["WindCondition"]) == "no":
        detect_win = 1
    else:
        detect_win = 1
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
    
    # ─────── velge antall perioder ─────── 
    input_periods = (meta_row["WavePeriodInput"])
    keep_periods= round((input_periods-13)*1.0) #trekke fra perioder, -per15- er det bare 4 gode, mens på -per40- per er ish 30 gode. TK todo velge en bedre skalering
    keep_seconds= keep_periods/input_freq
    keep_idx = keep_seconds*250 # 1 sek = 250 målinger
    good_range = keep_idx

    # MANUELL CALCULERING for 1.3 Hz
    P1amp01frwq13eyeball = 4500
    P2handcalc = P1amp01frwq13eyeball+100 #tidligere: 62.5 fra 250målinger på ett sekund, ganget et kvart sekund, estimert reisetid for bølgen på 1.3hz  
    P3handcalc = P2handcalc+1700 #tidligere: en 1.3hz gir periode på 700idx? 250 målinger per sek
    
    P1amp01f065eyeball = 3950
    P2_f065_handcalc = P1amp01f065eyeball+50 #
    P3_f065_handcalc = P2_f065_handcalc+500 #



    # ==========================================================
    #  1.b Start og slutt på signalet tilpasset innkommende bølge og vindforhold
    # ==========================================================
    """ELIF RETURN SNARVEI"""
    if input_freq == 1.3:
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
            good_end_idx = good_start_idx + keep_idx
        elif data_col == "Probe 4" : 
            good_start_idx = P3handcalc
            good_end_idx = good_start_idx + keep_idx
                # return good_start_idx, good_end_idx, debug_info
    if input_freq == 0.65:
        if data_col == "Probe 1": 
            good_start_idx = P1amp01f065eyeball 
            good_end_idx = good_start_idx+keep_idx
                #return good_start_idx, good_end_idx, debug_info
        elif data_col == "Probe 2" : 
             good_start_idx = P2_f065_handcalc
             good_end_idx = good_start_idx + keep_idx
                #return good_start_idx, good_end_idx, debug_info
        elif data_col == "Probe 3" : 
            good_start_idx = P3_f065_handcalc
            good_end_idx = good_start_idx + keep_idx
        elif data_col == "Probe 4" : 
            good_start_idx = P3_f065_handcalc
            good_end_idx = good_start_idx + keep_idx
                # return good_start_idx, good_end_idx, debug_info
    #import sys; print('exit'); sys.exit()
     
    
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

    
    baseline_seconds = 2
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
        try:
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
            plot_ramp_detection(**plot_kwargs)

        except Exception as e:
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
    }

    return good_start_idx, good_end_idx, debug_info
