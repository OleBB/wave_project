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
    df_sel,  # metadata for selected files
    data_col,
    detect_win=1,
    baseline_seconds=2.0,
    sigma_factor=5.0,
    skip_periods=None,
    keep_periods=None,
    range_plot: bool = False,
    min_ramp_peaks=5,
    max_ramp_peaks=15,
    max_dips_allowed=2,
    min_growth_factor=2.0,   # final amp must be at least 2x first
):
    """
    Intelligent detection of stable oscillation phase using peak amplitude ramp-up.
    Replaces crude "skip 5s + 12 periods" with actual detection of monotonic increase.
    """
    # ==========================================================
    # 1. Basic preprocessing
    # ==========================================================
    signal_smooth = (
        df[data_col]
        .rolling(window=detect_win, center=True, min_periods=1)
        .mean()
        .bfill().ffill()
        .values
    )
    
   
    dt = (df["Date"].iloc[1] - df["Date"].iloc[0]).total_seconds()
    Fs = 1.0 / dt
    
    # ───────  FREQUENCY EXTRACTION ───────
    freq_raw = df_sel["WaveFrequencyInput [Hz]"] if isinstance(df_sel, pd.Series) else df_sel["WaveFrequencyInput [Hz]"].iloc[0]
    if pd.isna(freq_raw) or str(freq_raw).strip() in ["", "nan"]:
        importertfrekvens = 1.3
        print(f"Warning: No valid frequency found → using fallback {importertfrekvens} Hz")
    else:
        importertfrekvens = float(freq_raw)

    samples_per_period = int(round(Fs / importertfrekvens))
  
# ─────────────────────────────────────────────────────────────
    # Make ramp detection work on your real, gentle ramp-ups
    min_ramp_peaks = 5
    max_ramp_peaks = 20
    max_dips_allowed = 2
    min_growth_factor = 1.015   # 1.5% total growth is enough (your ramp is slow!)
# ───────────────────────────────────────────────
    samples_per_period = int(round(Fs / importertfrekvens))

    # ==========================================================
    # 2. Baseline & first motion (still useful for rough start)
    # ==========================================================
    baseline_samples = int(baseline_seconds * Fs)
    baseline = signal_smooth[:baseline_samples]
    baseline_mean = np.mean(baseline)
    baseline_std = np.std(baseline)
    threshold = baseline_mean + sigma_factor * baseline_std

    above_noise = signal_smooth > threshold
    first_motion_idx = np.argmax(above_noise) if np.any(above_noise) else 0

    # ==========================================================
    # 3. Peak detection on absolute signal (handles both positive/negative swings)
    # ==========================================================
    # Use prominence and distance tuned to your frequency
    min_distance = max(3, samples_per_period // 3)  # at least 1/3 period apart
    peaks, properties = find_peaks(
        np.abs(signal_smooth),
        distance=min_distance,
        prominence=0.5 * baseline_std,  # ignore tiny noise peaks
        height=threshold
    )

    if len(peaks) < min_ramp_peaks + 3:
        print("Not enough peaks detected – falling back to legacy method")
        # Legacy fallback
        skip_periods = skip_periods or (5 + 12)
        keep_periods = keep_periods or 8
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
                "df_sel": df_sel,
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
            if 'peaks' in locals() and ramp_result is not None:
                plot_kwargs["peaks"] = peaks
                plot_kwargs["peak_amplitudes"] = peak_amplitudes
                ramp_peak_samples = peaks[ramp_result[0]:ramp_result[1]+1]
                plot_kwargs["ramp_peak_indices"] = ramp_peak_samples

            plot_ramp_detection(**plot_kwargs)

        except Exception as e:
            print(f"Plot failed (will work after you update plotter): {e}")

    debug_info = {
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "first_motion_idx": first_motion_idx,
        "samples_per_period": samples_per_period,
        "detected_peaks": len(peaks),
        "ramp_found": ramp_result is not None,
        "ramp_length_peaks": len(ramp_result[2]) if ramp_result else None,
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
def compute_simple_amplitudes(processed_dfs: dict, meta_sel: pd.DataFrame) -> pd.DataFrame:
    records = []
    for path, df in processed_dfs.items():
        subset_meta = meta_sel[meta_sel["path"] == path]
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



def remove_outliers():
    #lag noe basert på steepness, kanskje tilogmed ak. Hvis ak er for bratt
    # og datapunktet for høyt, så må den markeres, og så fjernes.
    #se Karens script
    return


# ================================================reco== #
# === Take in a filtered subset then process     === #
# === using functions: ensure_stillwater_columns === #
# === using functions: find_wave_range           === #
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

    # 2.a) Process only the selected runs
    processed_dfs = {}

    for _, row in meta_sel.iterrows():
        path = row["path"]
        if path not in dfs:
            print(f"Warning: File not loaded: {path}")
            continue

        df = dfs[path].copy()

        # Zero each probe
        for i in range(1, 5):
            probe_col = f"Probe {i}"           # ← your actual column name
            if probe_col not in df.columns:
                print(f"  Missing column {probe_col} in {Path(path).name}")
                continue

            sw = stillwater[i]
            eta_col = f"eta_{i}"

            # This is the key line: subtract stillwater → zero mean
            df[eta_col] = df[probe_col] - sw

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
                print('nu kjøres FIND_WAVE_RANGE, i indre loop i 2.b) i process_selected_data')
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
    # 3. Kjøre compute_simple_amplitudes, basert på computed range i meta_sel
    # ==========================================================
    #DataFrame.update aligns on index and columns and then in-place replaces values 
    #in meta_sel with the corresponding non-NA values from the other frame. 
    #It uses index+column labels for alignment.
    amplituder = compute_simple_amplitudes(processed_dfs, meta_sel)
    cols = [f"Probe {i} Amplitude" for i in range(1, 5)]
    meta_sel = meta_sel.set_index("path")
    meta_sel.update(amplituder.set_index("path")[cols])
    meta_sel = meta_sel.reset_index()

    # 3. Make sure meta_sel has the stillwater columns too (for plotting later)
    #The stillwater assignment uses scalar broadcasting to create or 
    #fill an entire column with the same value for every row.
    for i in range(1, 5):
        col = f"Stillwater Probe {i}"
        if col not in meta_sel.columns:
            print(f"stillwater av i er {stillwater[i]}")
            meta_sel[col] = stillwater[i]
            
    # 4. Make sure meta_sel knows where to save
    if "PROCESSED_folder" not in meta_sel.columns:
        if "PROCESSED_folder" in meta_full.columns:
            folder = meta_full["PROCESSED_folder"].iloc[0]
        elif "experiment_folder" in meta_full.columns:
            folder = "PROCESSED-" + meta_full["experiment_folder"].iloc[0]
        else:
            raw_folder = Path(meta_full["path"].iloc[0]).parent.name
            folder = f"PROCESSED-{raw_folder}"
        meta_sel["PROCESSED_folder"] = folder
        if debug:
            print(f"Set PROCESSED_folder = {folder}")

    # 5. Save updated metadata (now with stillwater columns)
    update_processed_metadata(meta_sel)
    # after all probes processed, then we drop the Raw Probe data
    cols_to_drop = ["Probe 1", "Probe 2", "Probe 3", "Probe 4", "Mach"]
    processed_dfs = {
        path: df.drop(columns=cols_to_drop, errors="ignore").copy()
        for path, df in processed_dfs.items()
    }

    print(f"\nProcessing complete! {len(processed_dfs)} files zeroed and ready.")
    return processed_dfs, meta_sel

