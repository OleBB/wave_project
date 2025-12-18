#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 17:18:03 2025

@author: ole
"""

from pathlib import Path
from typing import Iterator, Dict, Tuple
import json
import re
import pandas as pd
import os
from datetime import datetime
#from wavescripts.data_loader import load_or_update #blir vel motsatt.. 
import numpy as np
import matplotlib.pyplot as plt
from wavescripts.data_loader import update_processed_metadata
from scipy.signal import find_peaks


PROBES = ["Probe 1", "Probe 2", "Probe 3", "Probe 4"]
def find_wave_range(
    df,
    meta_row,  # metadata for selected files
    data_col,
    detect_win=1,
    range_plot: bool = False,
):
    """
    detection of stable oscillation phase using peak amplitude ramp-up.
    """
    if (meta_row["WindCondition"]) == "full":
        detect_win = 15
    if (meta_row["WindCondition"]) == "low":
        detect_win = 10

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

    # ───────  FREQUENCY EXTRACTION ───────
    input_freq = meta_row["WaveFrequencyInput [Hz]"] if isinstance(meta_row, pd.Series) else meta_row["WaveFrequencyInput [Hz]"].iloc[0]
    if pd.isna(input_freq) or str(input_freq).strip() in ["", "nan"]:
        importertfrekvens = 1.3
        print(f"Warning: No valid frequency found → using fallback {importertfrekvens} Hz")
    else:
        importertfrekvens = float(input_freq)

    samples_per_period = int(round(Fs / importertfrekvens))
    
    input_period = (meta_row["WavePeriodInput"])
    keep_periods= round((input_period-13)*1.0) #trekke fra perioder, 15 per er det bare 4 gode, mens på 40 per er ish 30 gode. TK todo velge en bedre skalering
    keep_seconds= keep_periods/input_freq
    keep_idx = keep_seconds*250 
    good_range = keep_idx
    
    
    baseline_seconds = 2

    sigma_factor=1.0
    skip_periods=None

    min_ramp_peaks=5
    max_ramp_peaks=15
    max_dips_allowed=2
    min_growth_factor = 1.015
    
    # MANUELL CALCULERING
    P1amp01frwq13eyeball = 4500
    P2handcalc = P1amp01frwq13eyeball+62.5 #250målinger på ett sekund, ganget et kvart sekund, estimert reisetid for bølgen på 1.3hz  
    P3handcalc = P2handcalc+7*250 #en 1.3hz gir periode på 700idx? 250 målinger per sek
    
    #print(keep_periods, keep_seconds, keep_idx)
    #print('og:', P1amp01frwq13eyeball, P2handcalc)
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
    
    #nesten så jeg vil lage en teoretisk utregnign... 
    #må jo få tildet. en predikering! 
    #hva sier teorien!
    
    #TODO forstå phase-speed og 
    
    """Setter basis idx basert på probe """
    if data_col == "Probe 1": 
            good_start_idx = P1amp01frwq13eyeball 
            good_end_idx = good_start_idx+keep_idx
            #return good_start_idx, good_end_idx, debug_info
    elif data_col == "Probe 2" : 
            good_start_idx = P2handcalc
            good_end_idx = P2handcalc + keep_idx
            #return good_start_idx, good_end_idx, debug_info
    elif data_col == "Probe 3" or "Probe 4": 
            good_start_idx = P3handcalc
            good_end_idx = P3handcalc + keep_idx
 
    
    #todo tilpasses input
    # ==========================================================
    #  1.b tilpasses innkommende bølge og vindforhold
    # ==========================================================
    """ELIF RETURN SNARVEI"""
    if input_freq == 1.3:
        print('fullwind og 1.3')
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
        elif data_col == "Probe 4" : 
                good_start_idx = P3handcalc
                good_end_idx = P3handcalc + keep_idx
                # return good_start_idx, good_end_idx, debug_info
   
    elif (meta_row["WaveFrequencyInput [Hz]"]) == 0.65:
        #print('vellyket ELIF 0.65')
        baseline_seconds=1.0
        sigma_factor=4.0
        skip_periods=None

        min_ramp_peaks=1
        max_ramp_peaks=15
        max_dips_allowed=2
        min_growth_factor=1
        good_start_idx = 4000
        good_end_idx = 5000
    else:
        print("ingen if-statments traff, ELSE basics kjøres:")
        baseline_seconds=2.0
        sigma_factor=1.0
        skip_periods=None

        min_ramp_peaks=1
        max_ramp_peaks=15
        max_dips_allowed=2
        min_growth_factor = 1.015
        
        good_start_idx = 4000
        good_end_idx = 5000
    
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

        
    #import sys; print('exit'); sys.exit()
    samples_per_period = int(round(Fs / importertfrekvens))

    # ==========================================================
    # 2. Baseline & first motion (still useful for rough start)
    # ==========================================================
    baseline_samples = int(baseline_seconds * Fs)
    baseline = signal_smooth[:baseline_samples]
    baseline_mean = np.mean(baseline)
    baseline_std = np.std(baseline)
    threshold = baseline_mean + sigma_factor*baseline_std
    
    print('baselines:')
    print(baseline_samples, baseline_mean, baseline_seconds, baseline_std)
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
    min_distance = max(3, input_period *0.9 )  # at least 0.9 period apart
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


# ========================================================== #
# === Make sure stillwater levels are computed and valid === #
# ========================================================== #
PROBES = ["Probe 1", "Probe 2", "Probe 3", "Probe 4"]
def ensure_stillwater_columns(
    dfs: dict[str, pd.DataFrame],
    meta: pd.DataFrame,
) -> pd.DataFrame:
    """
    Computes the true still-water level for each probe using ALL "no wind" runs,
    then copies that value into EVERY row of the metadata (including windy runs).
    Safe to call multiple times.
    """
    probe_cols = [f"Stillwater Probe {i}" for i in range(1, 5)]

    # If all columns exist AND all values are real numbers → we're done
    if all(col in meta.columns for col in probe_cols):
        if meta[probe_cols].notna().all().all():  # every cell has a real number
            print("Stillwater levels already computed and valid → skipping")
            return meta

    print("Computing still-water levels from all 'WindCondition == no' runs...")

    # Find all no-wind runs
    mask = meta["WindCondition"].astype(str).str.strip().str.lower() == "no"
    nowind_paths = meta.loc[mask, "path"].tolist()

    if not nowind_paths:
        raise ValueError("No runs with WindCondition == 'no' found! Cannot compute still water.")

    # Compute median from ALL calm data combined
    stillwater_values = {}
    for i in range(1, 5):
        probe_name = f"probe {i}"  # adjust if your columns are named differently
        all_values = []

        for path in nowind_paths:
            if path in dfs:
                df = dfs[path]
                if probe_name in df.columns:
                    clean = pd.to_numeric(df[probe_name], errors='coerce').dropna()
                    all_values.extend(clean.tolist())
                # Also try other common names just in case
                elif f"Probe {i}" in df.columns:
                    clean = pd.to_numeric(df[f"Probe {i}"], errors='coerce').dropna()
                    all_values.extend(clean.tolist())

        if len(all_values) == 0:
            raise ValueError(f"No valid data found for {probe_name} in any no-wind run!")

        level = np.median(all_values)
        stillwater_values[f"Stillwater Probe {i}"] = float(level)
        print(f"  Stillwater Probe {i}: {level:.3f} mm  (from {len(all_values):,} samples)")

    # Write the same value into EVERY row (this is correct!)
    for col, value in stillwater_values.items():
        meta[col] = value

    # Make sure we can save correctly
    if "PROCESSED_folder" not in meta.columns:
        if "experiment_folder" in meta.columns:
            meta["PROCESSED_folder"] = "PROCESSED-" + meta["experiment_folder"].iloc[0]
        else:
            raw_folder = Path(meta["path"].iloc[0]).parent.name
            meta["PROCESSED_folder"] = f"PROCESSED-{raw_folder}"

    # Save to disk
    update_processed_metadata(meta)
    print("Stillwater levels successfully saved to meta.json for ALL runs")

    return meta



# ------------------------------------------------------------
# enkel utregning av amplituder
# ------------------------------------------------------------
def compute_simple_amplitudes(processed_dfs: dict, meta_row: pd.DataFrame) -> pd.DataFrame:
    records = []
    for path, df in processed_dfs.items():
        subset_meta = meta_row[meta_row["path"] == path]
        for _, row in subset_meta.iterrows():
            row_out = {"path": path}
            for i in range(1, 5):
                col = f"eta_{i}"
                
                start_val = row.get(f"Computed Probe {i} start")
                end_val   = row.get(f"Computed Probe {i} end")
                
                # skip if missing/NaN
                if pd.isna(start_val) or pd.isna(end_val):
                    continue
                
                try:
                    s_idx = int(start_val)
                    e_idx = int(end_val)
                except (TypeError, ValueError):
                    continue
                
                # require start strictly before end
                if s_idx >= e_idx:
                    continue
                
                col = f"eta_{i}"
                if col not in df.columns:
                    continue
                
                n = len(df)
                # clamp indices to valid range
                s_idx = max(0, s_idx)
                e_idx = min(n - 1, e_idx)
                if s_idx > e_idx:
                    continue
                
                s = df[col].iloc[s_idx:e_idx+1].dropna().to_numpy()
                if s.size == 0:
                    continue
                
                amp = (np.percentile(s, 99.5) - np.percentile(s, 0.5)) / 2.0
                row_out[f"Probe {i} Amplitude"] = float(amp)

            records.append(row_out)
            #print(f"appended records: {records}")
    return pd.DataFrame.from_records(records)

        
from scipy.optimize import brentq
# def calculate_simple_wavenumber(meta_row):
#     """Tar inn metadata 
#     bruker BRENTQ fra scipy
#     """
#     g = 9.81
#     freq = meta_row["WaveFrequencyInput [Hz]"]
#     H = meta_row["WaterDepth [mm]"]
#     print('frq og H : ', freq, H)

#     period = 1/freq
#     omega = 2*np.pi/period
#     f = lambda k: g*k*np.tanh(k*H) - omega**2
#     k0 = omega**2/g #deep water guess
#     k1 = omega/np.sqrt(g*H) #shallow water guess
#     a, b = min(k0, k1)*0.1, max(k0, k1)*10
#     while f(a)*f(b) >0:
#         a, b = a/2, b*2
    
#     return brentq(f, a, b)

def calculate_wavenumbers(frequencies, heights):
    """Tar inn frekvens og høyde
    bruker BRENTQ fra scipy
    """
    freq = np.asarray(frequencies)
    H = np.broadcast_to(np.asarray(heights), freq.shape)
    k = np.zeros_like(freq, dtype=float)
    
    valid = freq>0
    i_valid = np.flatnonzero(valid)
    if i_valid.size ==0:
        return k
    g = 9.81
    for idx in i_valid:
        fr = freq.flat[idx]
        h = H.flat[idx]/1000 #konverter til millimeter
        omega = 2 * np.pi * fr
        
        def disp(k_wave):
            return g*k_wave* np.tanh(k_wave * h) - omega**2
        
        k_deep = omega**2 / g
        k_shallow = omega / np.sqrt(g * h) if h >0 else k_deep
        
        a = min(k_deep, k_shallow) *0.1
        b = max(k_deep, k_shallow) *10
        
        fa = disp(a)
        fb = disp(b)
        while fa * fb > 0:
            a /= 2
            b *= 2
            fa = disp(a)
            fb = disp(b)
        k.flat[idx] = brentq(disp, a, b)
    return k

def calculate_celerity(wavenumbers,heights):
    k = np.asarray(wavenumbers)
    H = np.broadcast_to(np.asarray(heights), k.shape)
    c = np.zeros_like(k,dtype=float)
    
    g = 9.81
    sigma = 0.074 #ved 20celcius
    rho = 1000 #10^3 kg/m^3
    
    c = np.sqrt( g/ k * np.tanh(k*H))
    
    
    return c
 #   for idx 



def remove_outliers():
    #lag noe basert på steepness, kanskje tilogmed ak. Hvis ak er for bratt
    # og datapunktet for høyt, så må den markeres, og så fjernes.
    #se Karens script
    return




# ================================================== #
# === Take in a filtered subset then process     === #
# === using functions: ensure_stillwater_columns === #
# === using functions: find_wave_range           === #
# === using functions: compute_simple_amplitudes === #
# === using functions: update_processed_metadata === #
# ================================================== #
def process_selected_data(
    dfs: dict[str, pd.DataFrame],
    meta_sel: pd.DataFrame,
    meta_full: pd.DataFrame,
    debug: bool = True,
    win: int = 10,
    find_range: bool = True,
    range_plot: bool = True
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Zeroes all selected runs using the shared stillwater levels.
    Adds eta_1..eta_4 (zeroed signal) and moving average.
    """
    # 1. Make sure stillwater levels are computed and valid
    meta_full = ensure_stillwater_columns(dfs, meta_full)

    # Extract the four stillwater values (same for whole experiment)
    stillwater = {}
    for i in range(1, 5):
        val = meta_full[f"Stillwater Probe {i}"].iloc[0]
        if pd.isna(val):
            raise ValueError(f"Stillwater Probe {i} is NaN! Run ensure_stillwater_columns first.")
        stillwater[i] = float(val)
        if debug:
            print(f"  Stillwater Probe {i} = {val:.3f} mm")

    if debug:
        print(f"Using stillwater levels: {stillwater}")

    # 2.a) Ta utvalgte kjøringer og sett null ved "stillwater"
    processed_dfs = {}
    for _, row in meta_sel.iterrows():
        path = row["path"]
        if path not in dfs:
            print(f"Warning: File not loaded: {path}")
            continue

        df = dfs[path].copy()

        # Zero each probe
        for i in range(1, 5):
            probe_col = f"Probe {i}"           
            if probe_col not in df.columns:
                print(f"  Missing column {probe_col} in {Path(path).name}")
                continue

            sw = stillwater[i]
            eta_col = f"eta_{i}"

            # subtract stillwater → zero mean
            df[eta_col] = -(df[probe_col] - sw) #bruk  MINUS for å snu signalet!
            #print(df[eta_col].iloc[0:10]) sjekk om den flipper
            # Optional: moving average of the zeroed signal
            df[f"{probe_col}_ma"] = df[eta_col].rolling(window=win, center=False).mean()
            
            if debug:
                print(f"  {Path(path).name:35} → eta_{i} mean = {df[eta_col].mean():.4f} mm")
        processed_dfs[path] = df
    
    # ==========================================================
    # 2. b) Optional- kjører FIND_WAVE_RANGE(...)
    # ==========================================================
    if find_range:
        for idx, row in meta_sel.iterrows():
            path = row["path"]
            df = processed_dfs[path].copy()
          
            for i in range(1,5):
                probe = f"Probe {i}"
                #print('nu kjøres FIND_WAVE_RANGE, i indre loop i 2.b) i process_selected_data')
                start, end, debug_info = find_wave_range(df, 
                                                         row,#pass single row
                                                         data_col=probe, 
                                                         detect_win=win, 
                                                         range_plot=range_plot
                                                         )
                probestartcolumn  = f'Computed Probe {i} start'
                meta_sel.loc[idx, probestartcolumn] = start
                probeendcolumn = f'Computed Probe {i} end'
                meta_sel.loc[idx, probeendcolumn] = end
                #print(f'meta_sel sin Computed probe {i} start: {meta_sel[probestartcolumn]} og end: {meta_sel[probeendcolumn]}')

        print(f'start: {start}, end: {end} og debug_range_info: {debug_info}')
    
    # ==========================================================
    # 3.a Kjøre compute_simple_amplitudes, basert på computed range i meta_sel
    # Oppdaterer meta_sel
    # ==========================================================
    #DataFrame.update aligns on index and columns and then in-place replaces values 
    #in meta_sel with the corresponding non-NA values from the other frame. 
    #It uses index+column labels for alignment.
    amplituder = compute_simple_amplitudes(processed_dfs, meta_sel)
    cols = [f"Probe {i} Amplitude" for i in range(1, 5)]
    meta_sel_indexed = meta_sel.set_index("path")
    meta_sel_indexed.update(amplituder.set_index("path")[cols])
    meta_sel = meta_sel_indexed.reset_index()
    
    
    # ==========================================================
    # 3.b Kjøre calculate_simple_wavenember, basert på inputfrekvens i meta_sel
    #     Oppdaterer meta_sel
    # ==========================================================
    columnz = ["path", "WaveFrequencyInput [Hz]", "WaterDepth [mm]"]
    sub_df = meta_sel[columnz].copy()
    sub_df["Wavenumber"] = calculate_wavenumbers(sub_df["WaveFrequencyInput [Hz]"], sub_df["WaterDepth [mm]"])
    m_s_indexed = meta_sel.set_index("path")
    w_s = sub_df.set_index("path")["Wavenumber"]
    m_s_indexed["Wavenumber"] = w_s
    meta_sel = m_s_indexed.reset_index()
    
    # ==========================================================
    # 3.c Kjøre calculate_celerity, basert på wavenumber i meta_sel
    #     Oppdaterer meta_sel
    # ==========================================================
    columnz3 = ["path", "WaterDepth [mm]", "Wavenumber"]
    sub_df3 = meta_sel[columnz3].copy()
    sub_df3["Celerity"] = calculate_celerity(sub_df3["Wavenumber"], sub_df3["WaterDepth [mm]"])
    m_s_indexed3 = meta_sel.set_index("path")
    w_s3 = sub_df3.set_index("path")["Celerity"]
    m_s_indexed3["Celerity"] = w_s3
    meta_sel = m_s_indexed3.reset_index()

    
    # ==========================================================
    # 4. Hvis StillwaterKollonen ikke... 
    # så fylles HELE stillwater-kolonnen med samme verdi
    # ==========================================================
    for i in range(1, 5):
        col = f"Stillwater Probe {i}"
        if col not in meta_sel.columns:
            print(f"stillwater av i er {stillwater[i]}")
            meta_sel[col] = stillwater[i]
            
    # 5.a Make sure meta_sel knows where to save
    if "PROCESSED_folder" not in meta_sel.columns:
        if "PROCESSED_folder" in meta_full.columns:
            folder = meta_full["PROCESSED_folder"].iloc[0]
        # elif "experiment_folder" in meta_full.columns:
            folder = "PROCESSED-" + meta_full["experiment_folder"].iloc[0]
        else:
            raw_folder = Path(meta_full["path"].iloc[0]).parent.name
            folder = f"PROCESSED-{raw_folder}"
        meta_sel["PROCESSED_folder"] = folder
        if debug:
            print(f"Set PROCESSED_folder = {folder}")

    # 5.b Save updated metadata (now with stillwater columns)
    update_processed_metadata(meta_sel)
    # after all probes processed, then we drop the Raw Probe data
    cols_to_drop = ["Probe 1", "Probe 2", "Probe 3", "Probe 4", "Mach"]
    processed_dfs = {
        path: df.drop(columns=cols_to_drop, errors="ignore").copy()
        for path, df in processed_dfs.items()
    }

    print(f"\nProcessing complete! {len(processed_dfs)} files zeroed and ready.")
    return processed_dfs, meta_sel

