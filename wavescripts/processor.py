#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 17:18:03 2025

@author: ole
"""
from pathlib import Path
import pandas as pd
import numpy as np
from wavescripts.data_loader import update_processed_metadata
from scipy.signal import find_peaks
from scipy.signal import welch
from scipy.optimize import brentq
from typing import Dict, List, Tuple


# In your main script:
from wavescripts.wave_detection import find_wave_range
from wavescripts.signal_processing import compute_psd_with_amplitudes, compute_fft_with_amplitudes
from wavescripts.wave_physics import calculate_wavenumbers, calculate_wavedimensions

PROBES = ["Probe 1", "Probe 2", "Probe 3", "Probe 4"]


# ========================================================== #
# === Make sure stillwater levels are computed and valid === #
# ========================================================== #
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
    update_processed_metadata(meta, force_recompute=False)
    print("Stillwater levels successfully saved to meta.json for ALL runs")

    return meta



def compute_amplitudes(processed_dfs: dict, meta_row: pd.DataFrame) -> pd.DataFrame:
    """Compute wave amplitudes from np.percentile"""
    records = []
    for path, df in processed_dfs.items():
        subset_meta = meta_row[meta_row["path"] == path]
        for _, row in subset_meta.iterrows():
            row_out = {"path": path}
            for i in range(1, 5):
                amplitude = _extract_probe_amplitude(df, row, i)
                if amplitude is not None:
                    row_out[f"Probe {i} Amplitude"] = amplitude
            records.append(row_out)
    return pd.DataFrame.from_records(records)

def compute_amplitudes_from_psd(f, pxx, target_freq, window=0.5):
    """Hent amplituden fra PSD ved gitt frekvens"""
    mask = (f >= target_freq - window) & (f <= target_freq + window)
    # psd_at_freq = pxx[mask].max()
    deltaf = f[1]-f[0] #frequency resolution
    # amplitude = np.sqrt(2 * psd_at_freq * deltaf)
    var = pxx[mask].sum() * deltaf
    sigma = np.sqrt(var)
    amplitude = np.sqrt(2) * sigma
    return amplitude

def compute_amplitudes_from_fft(fft_freqs, fft_magnitude, target_freq, window=0.5):
    """
    Extract amplitude from FFT at a given frequency.
    
    Args:
        fft_freqs: Frequency array from FFT
        fft_magnitude: Magnitude of FFT (already normalized to amplitude)
        target_freq: Target frequency (Hz)
        window: Frequency window around target (Hz). Default 0.5 Hz.
    
    Returns:
        Amplitude at target frequency
    """
    mask = (fft_freqs >= target_freq - window) & (fft_freqs <= target_freq + window)
    
    if not mask.any():
        # No frequencies in range - fallback to closest
        closest_idx = np.argmin(np.abs(fft_freqs - target_freq))
        return fft_magnitude[closest_idx]
    
    # Return the maximum amplitude in the window (peak)
    amplitude = fft_magnitude[mask].max()
    
    return amplitude


def compute_psd_with_amplitudes(processed_dfs: dict, meta_row: pd.DataFrame, fs: float = 250, debug:bool=False) -> Tuple[dict, list]:
    """Compute Power Spectral Density for each probe."""
    psd_dict = {}
    amplitude_records = []
    for path, df in processed_dfs.items():
        subset_meta = meta_row[meta_row["path"] == path ]
        for _, row in subset_meta.iterrows():
            row_out = {"path": path}
            
            freq = row["WaveFrequencyInput [Hz]"]
            # Validate frequency
            if pd.isna(freq) or freq <= 0:
                if debug:
                    print(f"Advarsel: Ingen Input frequency {freq} for {path}, skipping")
                continue
            
            psd_df = None
            desired_resolution = 0.125 #Hz per steg i Psd'en.
            npersegment = int(fs/desired_resolution)
            for i in range(1, 5):
                signal = _extract_probe_signal(df, row, i)
                nperseg = max(1, min(npersegment, len(signal)))
                    
                if signal is not None:

                    # nperseg = npersegment #har noen mye kortere signaler
                    noverlap = nperseg // 2  # or int(0.75 * nperseg)
                    
                    f, pxx = welch(
                        signal, fs=fs,
                        window='hann',
                        nperseg=nperseg,
                        noverlap=noverlap,
                        detrend='constant',
                        scaling='density'
                    )
                    if psd_df is None:
                        psd_df = pd.DataFrame(index=f)
                        psd_df.index.name = "Frequencies"  
                    psd_df[f"Pxx {i}"] = pxx
                    # - - - amplitude
                    amplitude = compute_amplitudes_from_psd(f, pxx, freq)
                    if debug:
                        print("amplitden inni PSD loopen er ", amplitude)
                    row_out[f"Probe {i} Amplitude (PSD)"] = amplitude
            if psd_df is not None:
                psd_dict[path] = psd_df
            amplitude_records.append(row_out)

    if debug:
        print(f"=== PSD Complete: {len(amplitude_records)} records ===\n")
    return psd_dict, amplitude_records

def compute_fft_with_amplitudes(processed_dfs: dict, meta_row: pd.DataFrame, fs: float = 250, debug:bool=False) -> Tuple[dict, list]:
    """Compute FFT for each probe. and calculate amplitude"""
    fft_dict = {}
    amplitude_records = []
    for path, df in processed_dfs.items():
        subset_meta = meta_row[meta_row["path"] == path]
        for _, row in subset_meta.iterrows():
            row_out = {"path": path}
            
            freq = row["WaveFrequencyInput [Hz]"]
            # Validate frequency
            if pd.isna(freq) or freq <= 0:
                if debug:
                    print(f"advarsel: No input frequency {freq} for {path}, skipping")
                continue
            
            fft_df = None
            series_list = []
            for i in range(1, 5):
                
                if (signal := _extract_probe_signal(df, row, i) )is not None: #Walrus operator := sjekker om den ikke er None
                    N = len(signal)
                    fft_vals = np.fft.rfft(signal)  # rfft only returns positive frequencies
                    fft_freqs = np.fft.rfftfreq(N, d=1/fs)
                    
                    amplitudar = np.abs(fft_vals) * 2 / N
                    amplitudar[0] = amplitudar[0] / 2 #0hz should not be doubled
                    
                    if N % 2 == 0:
                        amplitudar[-1] = amplitudar[-1] / 2 #if N is even, Nyquist freq should not be doubled
                    
                    series_list.append(pd.Series(amplitudar, index=fft_freqs, name=f"FFT {i}"))
                    # amplitude entall
                    amplitude = compute_amplitudes_from_fft(fft_freqs, amplitudar, freq)
                    row_out[f"Probe {i} Amplitude (FFT)"] = amplitude
                    
                    if debug:
                        print(f"  Probe {i}: Amplitude = {amplitude:.3f} mm at {freq:.3f} Hz")
            
            if series_list:  # Only create DataFrame if we have data
                fft_df = pd.concat(series_list, axis=1)
                fft_df = fft_df.sort_index()  # sorterer
                fft_df.index.name = "Frequencies"
                fft_dict[path] = fft_df
            amplitude_records.append(row_out)
    if debug:
        print(f"=== FFT Complete: {len(amplitude_records)} records ===\n")
    return fft_dict, amplitude_records


    
    
def _extract_probe_signal(df: pd.DataFrame, row: pd.Series, probe_num: int) -> np.ndarray | None:
    """Extract and validate signal data for a specific probe."""
    col = f"eta_{probe_num}"
    start_val = row.get(f"Computed Probe {probe_num} start")
    end_val = row.get(f"Computed Probe {probe_num} end")
    
    if pd.isna(start_val) or pd.isna(end_val) or col not in df.columns:
        return None
    
    try:
        s_idx = max(0, int(start_val))
        e_idx = min(len(df) - 1, int(end_val))
    except (TypeError, ValueError):
        return None
    
    if s_idx >= e_idx:
        return None
    
    signal = df[col].iloc[s_idx:e_idx+1].dropna().to_numpy()
    return signal if signal.size > 0 else None

def _extract_probe_amplitude(df: pd.DataFrame, row: pd.Series, probe_num: int) -> float | None:
    """Extract amplitude for a specific probe."""
    signal = _extract_probe_signal(df, row, probe_num)
    if signal is None:
        return None
    return float((np.percentile(signal, 99.5) - np.percentile(signal, 0.5)) / 2.0)



def remove_outliers():
    #lag noe basert på steepness, kanskje tilogmed ak. Hvis ak er for bratt
    # og datapunktet for høyt, så må den markeres, og så fjernes.
    #se Karens script
    return


"claude=============="
def _extract_stillwater_levels(meta_full: pd.DataFrame, debug: bool) -> dict:
    """Extract stillwater levels from metadata."""
    stillwater = {}
    for i in range(1, 5):
        val = meta_full[f"Stillwater Probe {i}"].iloc[0]
        if pd.isna(val):
            raise ValueError(f"Stillwater Probe {i} is NaN!")
        stillwater[i] = float(val)
        if debug:
            print(f"  Stillwater Probe {i} = {val:.3f} mm")
    return stillwater


def _zero_and_smooth_signals(
    dfs: dict, 
    meta_sel: pd.DataFrame, 
    stillwater: dict, 
    win: int,
    debug: bool
) -> dict[str, pd.DataFrame]:
    """Zero signals using stillwater and add moving averages."""
    processed_dfs = {}
    for _, row in meta_sel.iterrows():
        path = row["path"]
        if path not in dfs:
            print(f"Warning: File not loaded: {path}")
            continue

        df = dfs[path].copy()
        for i in range(1, 5):
            probe_col = f"Probe {i}"
            if probe_col not in df.columns:
                print(f"  Missing column {probe_col} in {Path(path).name}")
                continue

            eta_col = f"eta_{i}"
            df[eta_col] = -(df[probe_col] - stillwater[i])
            df[f"{probe_col}_ma"] = df[eta_col].rolling(window=win, center=False).mean()
            
            if debug:
                print(f"  {Path(path).name:35} → eta_{i} mean = {df[eta_col].mean():.4f} mm")
        
        processed_dfs[path] = df
    
    return processed_dfs


def run_find_wave_ranges(
    processed_dfs: dict,
    meta_sel: pd.DataFrame,
    win: int,
    range_plot: bool,
    debug: bool
) -> pd.DataFrame:
    """Find wave ranges for all probes."""
    for idx, row in meta_sel.iterrows():
        path = row["path"]
        df = processed_dfs[path]
      
        for i in range(1, 5):
            probe = f"Probe {i}"
            start, end, debug_info = find_wave_range(
                df, row, data_col=probe, detect_win=win, range_plot=range_plot, debug=debug
            )
            meta_sel.loc[idx, f'Computed Probe {i} start'] = start
            meta_sel.loc[idx, f'Computed Probe {i} end'] = end
        
        if debug and start:
            print(f'start: {start}, end: {end}, debug: {debug_info}')
    
    return meta_sel


def _update_all_metrics(
    processed_dfs: dict,
    meta_sel: pd.DataFrame,
    stillwater: dict,
    amplitudes_psd_df: pd.DataFrame,
    amplitudes_fft_df: pd.DataFrame,
) -> pd.DataFrame:
    """Kalkuler og oppdater all computed metrics in metadata."""
    meta_indexed = meta_sel.set_index("path")
    
    # Amplitudes from np.percentile
    amplitudes = compute_amplitudes(processed_dfs, meta_sel)
    amp_cols = [f"Probe {i} Amplitude" for i in range(1, 5)]
    meta_indexed.update(amplitudes.set_index("path")[amp_cols])
    
    #Amplitudes from psd
    amp_psd_cols = [f"Probe {i} Amplitude (PSD)" for i in range(1, 5)]
    meta_indexed[amp_psd_cols] = amplitudes_psd_df.set_index("path")[amp_psd_cols]  # Direct assignment
    
    #Amplitudes from FFT
    amp_fft_cols = [f"Probe {i} Amplitude (FFT)" for i in range(1, 5)]
    meta_indexed[amp_fft_cols] = amplitudes_fft_df.set_index("path")[amp_fft_cols]  # Direct assignment

    
    # Wavenumbers
    meta_indexed["Wavenumber"] = calculate_wavenumbers(
        meta_indexed["WaveFrequencyInput [Hz]"], 
        meta_indexed["WaterDepth [mm]"]
    )
    
    # Wave dimensions
    wave_dims = calculate_wavedimensions(
        k=meta_indexed["Wavenumber"],
        H=meta_indexed["WaterDepth [mm]"],
        PC=meta_indexed["PanelCondition"],
        P2A=meta_indexed["Probe 2 Amplitude"],
    )
    meta_indexed[["Wavelength", "kL", "ak", "kH", "tanh(kH)", "Celerity"]] = wave_dims
    
    # Windspeed
    meta_indexed["Windspeed"] = calculate_windspeed(meta_indexed["WindCondition"])
    
    # Add stillwater columns if missing
    for i in range(1, 5):
        col = f"Stillwater Probe {i}"
        if col not in meta_indexed.columns:
            meta_indexed[col] = stillwater[i]
    
    return meta_indexed.reset_index()


def _set_output_folder(
    meta_sel: pd.DataFrame,
    meta_full: pd.DataFrame,
    debug: bool
) -> pd.DataFrame:
    """Velg output folder for processed data."""
    if "PROCESSED_folder" not in meta_sel.columns:
        if "PROCESSED_folder" in meta_full.columns:
            folder = meta_full["PROCESSED_folder"].iloc[0]
        elif "experiment_folder" in meta_full.columns:
            folder = f"PROCESSED-{meta_full['experiment_folder'].iloc[0]}"
        else:
            raw_folder = Path(meta_full["path"].iloc[0]).parent.name
            folder = f"PROCESSED-{raw_folder}"
        
        meta_sel["PROCESSED_folder"] = folder
        if debug:
            print(f"Set PROCESSED_folder = {folder}")
    
    return meta_sel


def process_selected_data(
    dfs: dict[str, pd.DataFrame],
    meta_sel: pd.DataFrame,
    meta_full: pd.DataFrame,
    processvariables: dict, 
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict]:
    """
    1. Zeroes all selected runs using the shared stillwater levels.
    2. Adds eta_1..eta_4 (zeroed signal) and moving average.
    3. Find wave range (optional)
    4. Regner PSD og FFT med tilhørende 
    5. Oppdaterer meta
    """
    fs = 250 #samplerate -foreløpig hardkodet mange steder.
    # 0. unpack processvariables
    # overordnet = processvariables.get("overordnet", {})
    prosessering = processvariables.get("prosessering", {})
    
    debug = prosessering.get("debug", False)
    win = prosessering.get("smoothing_window", 1)
    find_range =prosessering.get("find_range", False)
    range_plot =prosessering.get("range_plot", False)
    force_recompute =prosessering.get("force_recompute", False)
    
    # 1. Ensure stillwater levels are computed
    meta_full = ensure_stillwater_columns(dfs, meta_full)
    stillwater = _extract_stillwater_levels(meta_full, debug)

    # 2. Process dataframes: zero and add moving averages
    processed_dfs = _zero_and_smooth_signals(dfs, meta_sel, stillwater, win, debug)
    
    # 3. Optional: find wave ranges
    if find_range:
        meta_sel = run_find_wave_ranges(processed_dfs, meta_sel, win, range_plot, debug)
    
    # 4. a - Compute PSDs and amplitudes from PSD
    psd_dict, amplitudes_from_psd  = compute_psd_with_amplitudes(processed_dfs, meta_sel, fs=fs,debug=debug)
    amplitudes_psd_df = pd.DataFrame(amplitudes_from_psd)
   
    # 4. b - compute FFT and amplitudes from FFT
    fft_dict, amplitudes_from_fft = compute_fft_with_amplitudes(processed_dfs, meta_sel, fs=fs, debug=debug)
    amplitudes_fft_df = pd.DataFrame(amplitudes_from_fft)
    # print("FFT columns:", amplitudes_fft_df.columns.tolist())
    # print("FFT shape:", amplitudes_fft_df.shape)
    # print("FFT head:", amplitudes_fft_df.head())

    # 5. Compute and update all metrics (amplitudes, wavenumbers, dimensions, windspeed)
    meta_sel = _update_all_metrics(processed_dfs, meta_sel, stillwater, amplitudes_psd_df, amplitudes_fft_df)

    # 6. Set output folder and save metadata
    meta_sel = _set_output_folder(meta_sel, meta_full, debug)

    update_processed_metadata(meta_sel, force_recompute=force_recompute)

    if debug:
        print(f"\nProcessing complete! {len(processed_dfs)} files zeroed and ready.")
    
    return processed_dfs, meta_sel, psd_dict, fft_dict

