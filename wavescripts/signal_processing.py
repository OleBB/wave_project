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
from wavescripts.constants import SIGNAL, RAMP, MEASUREMENT, get_smoothing_window
from wavescripts.constants import (
    ProbeColumns as PC, 
    GlobalColumns as GC, 
    ColumnGroups as CG,
    CalculationResultColumns as RC
)



# %% - Fysisk amplitude
def _extract_probe_signal(
    df: pd.DataFrame,
    row: pd.Series,
    pos: str,
) -> Optional[np.ndarray]:
    """Extract signal for a probe identified by position string (e.g. '9373/170')."""
    col = f"eta_{pos}"
    start_val = row.get(f"Computed Probe {pos} start")
    end_val = row.get(f"Computed Probe {pos} end")
    
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
    pos: str,
) -> Optional[float]:
    """Finner høyeste percentil a=h/2 for én og én probe."""
    signal = _extract_probe_signal(df, row, pos)
    if signal is None:
        return None
    
    upper_p = np.percentile(signal, AMPLITUDE.UPPER_PERCENTILE)
    lower_p = np.percentile(signal, AMPLITUDE.LOWER_PERCENTILE)
    
    return float((upper_p - lower_p) / AMPLITUDE.AMPLITUDE_DIVISOR)


def _extract_probe_matrix(
    df: pd.DataFrame,
    row: pd.Series,
    col_names: dict,
) -> tuple[Optional[np.ndarray], list[str]]:
    """
    Extract probe signals as a 2D matrix where possible.

    Args:
        col_names: {probe_num: pos_str} e.g. {1: "9373/170", 2: "12545", ...}

    Returns:
        (matrix, valid_pos_list) or (None, []) if extraction fails
        Matrix shape: (n_samples, n_valid_probes)
    """
    probe_ranges = {}
    for pos in col_names.values():
        start = row.get(f"Computed Probe {pos} start")
        end = row.get(f"Computed Probe {pos} end")

        if pd.notna(start) and pd.notna(end):
            try:
                s_idx = max(0, int(start))
                e_idx = min(len(df) - 1, int(end))
                if s_idx < e_idx:
                    probe_ranges[pos] = (s_idx, e_idx)
            except (TypeError, ValueError):
                pass

    if not probe_ranges:
        return None, []

    ranges = list(probe_ranges.values())
    if len(set(ranges)) == 1:
        s_idx, e_idx = ranges[0]
        valid_pos = sorted(probe_ranges.keys())
        cols = [f"eta_{pos}" for pos in valid_pos]
        if all(col in df.columns for col in cols):
            matrix = df[cols].iloc[s_idx:e_idx+1].values
            return matrix, valid_pos

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
    # nanpercentile ignores NaN samples (common in nowave runs) instead of propagating them
    upper = np.nanpercentile(matrix, AMPLITUDE.UPPER_PERCENTILE, axis=0)
    lower = np.nanpercentile(matrix, AMPLITUDE.LOWER_PERCENTILE, axis=0)
    
    amplitudes = (upper - lower) / AMPLITUDE.AMPLITUDE_DIVISOR
    
    return amplitudes.tolist()

def compute_amplitudes(
    processed_dfs: dict,
    meta_row: pd.DataFrame,
    cfg,
) -> pd.DataFrame:
    """Use matrix approach when possible, fall back otherwise."""
    col_names = cfg.probe_col_names()  # {1: "9373/170", ...}
    records = []

    for path, df in processed_dfs.items():
        subset_meta = meta_row[meta_row["path"] == path]

        for _, row in subset_meta.iterrows():
            row_out = {"path": path}

            probe_data, valid_pos = _extract_probe_matrix(df, row, col_names)

            if probe_data is not None and len(valid_pos) == 4:
                amplitudes = _compute_matrix_amplitudes(probe_data)
                for pos, amp in zip(valid_pos, amplitudes):
                    row_out[f"Probe {pos} Amplitude"] = amp
            else:
                for pos in col_names.values():
                    amplitude = _extract_probe_amplitude(df, row, pos)
                    if amplitude is not None:
                        row_out[f"Probe {pos} Amplitude"] = amplitude

            records.append(row_out)

    return pd.DataFrame.from_records(records)

# %% - PSD og FFT
def get_positive_spectrum(fft_df):
    """Extract only positive frequencies from full FFT DataFrame"""
    return fft_df[fft_df.index >= 0]

def get_complex_spectrum(fft_df):
    """Extract only complex-valued columns"""
    complex_cols = [c for c in fft_df.columns if 'complex' in c]
    return fft_df[complex_cols]


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

def compute_amplitudes_from_fft(fft_freqs, fft_magnitude, target_freq, window=0.1):
    """
    Extract amplitude at a given target frequency from FFT.

    Uses the nearest bin to target_freq within ±window Hz.
    The window exists only to guard against empty-mask edge cases; it is
    intentionally narrow (default 0.1 Hz) so that wind-generated waves at
    adjacent frequencies do not contaminate the amplitude reading.
    With typical signal lengths (10–60 s) and fs=250 Hz the FFT bin spacing
    is 0.017–0.1 Hz, so ±0.1 Hz always catches at least one bin.

    Args:
        fft_freqs: Frequency array from FFT (positive half)
        fft_magnitude: Magnitude spectrum (already normalised to amplitude)
        target_freq: Target frequency in Hz (WaveFrequencyInput)
        window: Half-width of search window in Hz. Default 0.1 Hz.

    Returns:
        (amplitude, frequency) at the bin nearest target_freq
    """
    mask = (fft_freqs >= target_freq - window) & (fft_freqs <= target_freq + window)

    if not mask.any():
        # Fallback: absolute nearest bin (should be rare with window=0.1)
        closest_idx = np.argmin(np.abs(fft_freqs - target_freq))
        return fft_magnitude[closest_idx], fft_freqs[closest_idx]

    masked_magnitudes = fft_magnitude[mask]
    masked_freqs = fft_freqs[mask]

    # Nearest bin to target — NOT argmax, to avoid grabbing wind-wave peaks
    nearest_idx = np.argmin(np.abs(masked_freqs - target_freq))

    return masked_magnitudes[nearest_idx], masked_freqs[nearest_idx]


def compute_psd_with_amplitudes(processed_dfs: dict, meta_row: pd.DataFrame, cfg, fs, debug: bool = False) -> Tuple[dict, pd.DataFrame]:
    """Compute Power Spectral Density for each probe."""
    col_names = cfg.probe_col_names()  # {1: "9373/170", ...}
    psd_dict = {}
    amplitude_records = []
    for path, df in processed_dfs.items():
        subset_meta = meta_row[meta_row["path"] == path]
        for _, row in subset_meta.iterrows():
            row_out = {"path": path}

            freq = row["WaveFrequencyInput [Hz]"]
            if pd.isna(freq) or freq <= 0:
                if debug:
                    print(f"Advarsel: Ingen Input frequency {freq} for {path}, skipping")
                continue

            series_list = []
            npersegment = int(MEASUREMENT.SAMPLING_RATE / SIGNAL.PSD_FREQUENCY_RESOLUTION)
            for i, pos in col_names.items():
                signal = _extract_probe_signal(df, row, pos)
                if signal is None or len(signal) < 2:
                    continue
                nperseg = max(2, min(npersegment, len(signal)))
                noverlap = min(int(SIGNAL.PSD_OVERLAP_FRACTION * nperseg), nperseg - 1)

                f, pxx = welch(
                    signal, fs=fs,
                    window='hann',
                    nperseg=nperseg,
                    noverlap=noverlap,
                    detrend='constant',
                    scaling='density'
                )
                series_list.append(pd.Series(pxx, index=f, name=f"Pxx {pos}"))
                amplitude = compute_amplitudes_from_psd(f, pxx, freq)
                row_out[f"Probe {pos} Amplitude (PSD)"] = amplitude

            if series_list:
                psd_df = pd.concat(series_list, axis=1).sort_index()
                psd_df.index.name = "Frequencies"
                psd_dict[path] = psd_df
            amplitude_records.append(row_out)

    if debug:
        print(f"=== PSD Complete: {len(amplitude_records)} records ===\n")
    return psd_dict, pd.DataFrame(amplitude_records)

def compute_nowave_psd(
    processed_dfs: dict,
    meta_nowave: pd.DataFrame,
    cfg,
    fs: float,
    nperseg: int = 4096,
) -> dict:
    """Compute broadband Welch PSD for nowave runs (wind-only + stillwater).

    Uses the full zeroed signal (eta_) — no wave-range window — with a large
    nperseg suitable for resolving wind-wave frequencies (2–10 Hz).
    Returns a dict in the same format as psd_dict: {path: DataFrame(index=Frequencies)}.
    """
    col_names = cfg.probe_col_names()
    nowave_paths = set(meta_nowave["path"].values)
    result = {}

    for path, df in processed_dfs.items():
        if path not in nowave_paths:
            continue
        series_list = []
        for pos in col_names.values():
            eta_col = f"eta_{pos}"
            sig = df[eta_col].dropna().values if eta_col in df.columns else None
            if sig is None or len(sig) < nperseg:
                continue
            _nperseg = min(nperseg, len(sig))
            f, pxx = welch(
                sig, fs=fs,
                window="hann",
                nperseg=_nperseg,
                noverlap=_nperseg // 2,
                detrend="constant",
                scaling="density",
            )
            series_list.append(pd.Series(pxx, index=f, name=f"Pxx {pos}"))
        if series_list:
            psd_df = pd.concat(series_list, axis=1).sort_index()
            psd_df.index.name = "Frequencies"
            result[path] = psd_df

    return result


def compute_fft_with_amplitudes(processed_dfs: dict, meta_row: pd.DataFrame, cfg, fs, debug: bool = False) -> Tuple[dict, pd.DataFrame]:
    """Compute FFT for each probe and calculate amplitude, frequency, and period."""
    col_names = cfg.probe_col_names()  # {1: "9373/170", ...}
    fft_dict = {}
    amplitude_records = []

    for path, df in processed_dfs.items():
        subset_meta = meta_row[meta_row["path"] == path]

        for _, row in subset_meta.iterrows():
            row_out = {"path": path}

            freq = row["WaveFrequencyInput [Hz]"]
            if pd.isna(freq) or freq <= 0:
                if debug:
                    print(f"advarsel: No input frequency {freq} for {path}, skipping")
                continue

            series_list = []

            for i, pos in col_names.items():
                signal = _extract_probe_signal(df, row, pos)
                if signal is None:
                    continue
                N = len(signal)

                fft_vals = np.fft.fft(signal)
                fft_freqs = np.fft.fftfreq(N, d=1/fs)
                amplitudes = np.abs(fft_vals) / N

                series_list.append(pd.Series(amplitudes, index=fft_freqs, name=f"FFT {pos}"))
                series_list.append(pd.Series(fft_vals, index=fft_freqs, name=f"FFT {pos} complex"))

                pos_mask = fft_freqs > 0
                amplitudes_pos = 2 * np.abs(fft_vals[pos_mask]) / N
                amplitude, frequency = compute_amplitudes_from_fft(fft_freqs[pos_mask], amplitudes_pos, freq)

                row_out[f"Probe {pos} Amplitude (FFT)"] = amplitude
                row_out[f"Probe {pos} Frequency (FFT)"] = frequency
                row_out[f"Probe {pos} WavePeriod (FFT)"] = 1.0 / frequency if frequency > 0 else np.nan

            if series_list:
                fft_df = pd.concat(series_list, axis=1).sort_index()
                fft_df.index.name = "Frequencies"
                fft_dict[path] = fft_df

            amplitude_records.append(row_out)
    
    if debug:
        print(f"=== FFT Complete: {len(amplitude_records)} records ===\n")
    
    return fft_dict, pd.DataFrame(amplitude_records)





