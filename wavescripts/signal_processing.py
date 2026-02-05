#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 09:47:15 2026

@author: ole
"""


import pandas as pd
import numpy as np
from scipy.signal import welch

from typing import Dict, List, Tuple, Optional

from wavescripts.constants import AMPLITUDE, MEASUREMENT, SIGNAL

# %% - Fysisk amplitude

def _extract_probe_signal(
    df: pd.DataFrame, 
    row: pd.Series, 
    probe_num: int
) -> Optional[np.ndarray]:
    """Your original function - unchanged."""
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
    

def _extract_probe_amplitude(
    df: pd.DataFrame, 
    row: pd.Series, 
    probe_num: int
) -> Optional[float]:
    """Finner høyeste percentil a=h/2 for én og én probe."""
    signal = _extract_probe_signal(df, row, probe_num)
    if signal is None:
        return None
    
    upper_p = np.percentile(signal, AMPLITUDE.UPPER_PERCENTILE)
    lower_p = np.percentile(signal, AMPLITUDE.LOWER_PERCENTILE)
    
    return float((upper_p - lower_p) / AMPLITUDE.AMPLITUDE_DIVISOR)


def _extract_probe_matrix(
    df: pd.DataFrame,
    row: pd.Series
) -> tuple[Optional[np.ndarray], list[int]]:
    """
    Extract probe signals as a 2D matrix where possible.
    
    Returns:
        (matrix, valid_probe_numbers) or (None, []) if extraction fails
        Matrix shape: (n_samples, n_valid_probes)
    """
    # Find which probes have same start/end indices
    probe_ranges = {}
    for i in range(1, 5):
        start = row.get(f"Computed Probe {i} start")
        end = row.get(f"Computed Probe {i} end")
        
        if pd.notna(start) and pd.notna(end):
            try:
                s_idx = max(0, int(start))
                e_idx = min(len(df) - 1, int(end))
                if s_idx < e_idx:
                    probe_ranges[i] = (s_idx, e_idx)
            except (TypeError, ValueError):
                pass
    
    if not probe_ranges:
        return None, []
    
    # Check if all valid probes have the same range
    ranges = list(probe_ranges.values())
    if len(set(ranges)) == 1:
        # All probes have identical range - can extract as matrix!
        s_idx, e_idx = ranges[0]
        valid_probes = sorted(probe_ranges.keys())
        
        cols = [f"eta_{i}" for i in valid_probes]
        if all(col in df.columns for col in cols):
            matrix = df[cols].iloc[s_idx:e_idx+1].values
            return matrix, valid_probes
    
    # Fall back to individual extraction
    return None, []


def _compute_matrix_amplitudes(matrix: np.ndarray) -> list[float]:
    """
    Compute amplitudes for all columns at once.
    
    Args:
        matrix: Shape (n_samples, n_probes)
    
    Returns:
        List of amplitudes (one per probe)
    """
    # Vectorized percentile calculation across axis=0 (for each column/probe)
    upper = np.percentile(matrix, AMPLITUDE.UPPER_PERCENTILE, axis=0)
    lower = np.percentile(matrix, AMPLITUDE.LOWER_PERCENTILE, axis=0)
    
    amplitudes = (upper - lower) / AMPLITUDE.AMPLITUDE_DIVISOR
    
    return amplitudes.tolist()

def compute_amplitudes(
    processed_dfs: dict,
    meta_row: pd.DataFrame
) -> pd.DataFrame:
    """
    Use matrix approach when possible, fall back otherwise.
    
    """
    records = []
    
    for path, df in processed_dfs.items():
        subset_meta = meta_row[meta_row["path"] == path]
        
        for _, row in subset_meta.iterrows():
            row_out = {"path": path}
            
            # Try fast matrix extraction first
            probe_data, valid_probes = _extract_probe_matrix(df, row)
            
            if probe_data is not None and len(valid_probes) == 4:
                # Fast path: all 4 probes with same range
                amplitudes = _compute_matrix_amplitudes(probe_data)
                for probe_idx, amp in zip(valid_probes, amplitudes):
                    row_out[f"Probe {probe_idx} Amplitude"] = amp
            
            else:
                # Slow path: extract individually (handles mismatched ranges)
                for i in range(1, 5):
                    amplitude = _extract_probe_amplitude(df, row, i)
                    if amplitude is not None:
                        row_out[f"Probe {i} Amplitude"] = amplitude
            
            records.append(row_out)
    
    return pd.DataFrame.from_records(records)

# %% - PSD og FFT
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
    Extract amplitude and corresponding frequency from FFT at a given target frequency.
    
    Args:
        fft_freqs: Frequency array from FFT
        fft_magnitude: Magnitude of FFT (already normalized to amplitude)
        target_freq: Target frequency (Hz)
        window: Frequency window around target (Hz). Default 0.5 Hz.
    
    Returns:
        tuple: (amplitude, frequency) - amplitude at peak and its corresponding frequency
    """
    mask = (fft_freqs >= target_freq - window) & (fft_freqs <= target_freq + window)
    
    if not mask.any():
        # No frequencies in range - fallback to closest
        closest_idx = np.argmin(np.abs(fft_freqs - target_freq))
        return fft_magnitude[closest_idx], fft_freqs[closest_idx]
    
    # Find the index of maximum amplitude in the window
    masked_magnitudes = fft_magnitude[mask]
    masked_freqs = fft_freqs[mask]
    
    max_idx = np.argmax(masked_magnitudes)
    
    amplitude = masked_magnitudes[max_idx]
    frequency = masked_freqs[max_idx]
    
    return amplitude, frequency


def compute_psd_with_amplitudes(processed_dfs: dict, meta_row: pd.DataFrame, fs, debug:bool=False) -> Tuple[dict, list]:
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
            
            series_list = []
            npersegment = int(MEASUREMENT.SAMPLING_RATE / SIGNAL.PSD_FREQUENCY_RESOLUTION)
            for i in range(1, 5):
                signal = _extract_probe_signal(df, row, i)
                if signal is None or len(signal) < 2:
                    print('for this probe - no signal')
                    continue #skip very short signals
                nperseg = max(2, min(npersegment, len(signal))) #ensure 2 for spectrum
                overlap_fraction = SIGNAL.PSD_OVERLAP_FRACTION
                noverlap = int(overlap_fraction * nperseg) if nperseg > 1 else 0
                noverlap = min(noverlap, nperseg - 1)  # Ensure < nperseg
                   
                f, pxx = welch(
                    signal, fs=fs,
                    window='hann',
                    nperseg=nperseg,
                    noverlap=noverlap,
                    detrend='constant',
                    scaling='density'
                )
                series_list.append(pd.Series(pxx, index=f, name = f"Pxx {i}"))
                # - - - amplitude
                amplitude = compute_amplitudes_from_psd(f, pxx, freq)
                if debug:
                    print("amplitden inni PSD loopen er ", amplitude)
                row_out[f"Probe {i} Amplitude (PSD)"] = amplitude
            if series_list:
                psd_df = pd.concat(series_list, axis=1).sort_index()
                psd_df.index.name = "Frequencies"
                psd_dict[path] = psd_df
            amplitude_records.append(row_out)

    if debug:
        print(f"=== PSD Complete: {len(amplitude_records)} records ===\n")
    return psd_dict, amplitude_records

def compute_fft_with_amplitudes(processed_dfs: dict, meta_row: pd.DataFrame, fs, debug:bool=False) -> Tuple[dict, list]:
    """Compute FFT for each probe and calculate amplitude, frequency, and period"""
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
                
                if (signal := _extract_probe_signal(df, row, i)) is not None:
                    N = len(signal)
                    fft_vals = np.fft.rfft(signal)
                    fft_freqs = np.fft.rfftfreq(N, d=1/fs)
                    
                    amplitudar = np.abs(fft_vals) * 2 / N
                    amplitudar[0] = amplitudar[0] / 2  # 0hz should not be doubled
                    
                    if N % 2 == 0:
                        amplitudar[-1] = amplitudar[-1] / 2  # if N is even, Nyquist freq should not be doubled
                    
                    series_list.append(pd.Series(amplitudar, index=fft_freqs, name=f"FFT {i}"))
                    
                    # Extract amplitude and frequency from FFT
                    amplitude, frequency = compute_amplitudes_from_fft(fft_freqs, amplitudar, freq)
                    
                    # Store all FFT-derived metrics
                    row_out[f"Probe {i} Amplitude (FFT)"] = amplitude
                    row_out[f"Probe {i} Frequency (FFT)"] = frequency
                    row_out[f"Probe {i} WavePeriod (FFT)"] = 1.0 / frequency if frequency > 0 else np.nan
                    
                    if debug:
                        print(f"  Probe {i}: Amplitude = {amplitude:.3f} mm at {frequency:.3f} Hz (T = {1.0/frequency:.3f} s)")
            
            if series_list:  # Only create DataFrame if we have data
                fft_df = pd.concat(series_list, axis=1)
                fft_df = fft_df.sort_index()
                fft_df.index.name = "Frequencies"
                fft_dict[path] = fft_df
            amplitude_records.append(row_out)
    
    if debug:
        print(f"=== FFT Complete: {len(amplitude_records)} records ===\n")
    return fft_dict, amplitude_records





