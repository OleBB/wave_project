#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example: How to refactor your existing code to use constants.py

This shows before/after comparisons for key functions.
"""

# =============================================================================
# IMPORTS
# =============================================================================

from constants import (
    PHYSICS, MEASUREMENT, SIGNAL, RAMP, AMPLITUDE, MANUAL, WAVENUMBER,
    get_smoothing_window, get_wind_speed, get_panel_length
)
import numpy as np
import pandas as pd


# =============================================================================
# EXAMPLE 1: find_wave_range() - smoothing window selection
# =============================================================================

def find_wave_range_BEFORE(df, meta_row, data_col, detect_win, range_plot, debug):
    """OLD VERSION - hardcoded values scattered throughout."""
    if (meta_row["WindCondition"]) == "full":
        detect_win = 15
    elif (meta_row["WindCondition"]) == "low":
        detect_win = 10
    elif (meta_row["WindCondition"]) == "no":
        detect_win = 1
    else:
        detect_win = 1
    # ... rest of function


def find_wave_range_AFTER(df, meta_row, data_col, range_plot, debug):
    """NEW VERSION - uses constants."""
    wind_condition = meta_row["WindCondition"]
    detect_win = get_smoothing_window(wind_condition)
    
    signal_smooth = (
        df[data_col]
        .rolling(window=detect_win, center=True, min_periods=1)
        .mean()
        .bfill().ffill()
        .values
    )
    # ... rest of function


# =============================================================================
# EXAMPLE 2: Baseline detection
# =============================================================================

def detect_baseline_BEFORE(df, Fs):
    """OLD VERSION."""
    baseline_seconds = 2
    sigma_factor = 1.0
    
    baseline_samples = int(baseline_seconds * Fs)
    baseline = signal_smooth[:baseline_samples]
    baseline_mean = np.mean(baseline)
    baseline_std = np.std(baseline)
    threshold = baseline_mean + sigma_factor * baseline_std
    return threshold


def detect_baseline_AFTER(signal_smooth):
    """NEW VERSION."""
    baseline_samples = int(SIGNAL.BASELINE_DURATION_SEC * MEASUREMENT.SAMPLING_RATE)
    baseline = signal_smooth[:baseline_samples]
    baseline_mean = np.mean(baseline)
    baseline_std = np.std(baseline)
    threshold = baseline_mean + SIGNAL.BASELINE_SIGMA_FACTOR * baseline_std
    return threshold


# =============================================================================
# EXAMPLE 3: Ramp detection parameters
# =============================================================================

def find_best_ramp_BEFORE(seq):
    """OLD VERSION."""
    min_ramp_peaks = 5
    max_ramp_peaks = 15
    max_dips_allowed = 2
    min_growth_factor = 1.015
    # ... rest of function


def find_best_ramp_AFTER(seq):
    """NEW VERSION."""
    # All parameters come from constants
    return _find_ramp_core(
        seq,
        min_len=RAMP.MIN_RAMP_PEAKS,
        max_len=RAMP.MAX_RAMP_PEAKS,
        max_dips=RAMP.MAX_DIPS_ALLOWED,
        min_growth=RAMP.MIN_GROWTH_FACTOR
    )


# =============================================================================
# EXAMPLE 4: Manual detection points
# =============================================================================

def get_manual_start_BEFORE(input_freq, data_col):
    """OLD VERSION - hardcoded everywhere."""
    P1amp01frwq13eyeball = 4500
    P2handcalc = P1amp01frwq13eyeball + 100
    P3handcalc = P2handcalc + 1700
    
    if input_freq == 1.3:
        if data_col == "Probe 1":
            return P1amp01frwq13eyeball
        elif data_col == "Probe 2":
            return P2handcalc
        elif data_col == "Probe 3":
            return P3handcalc
    # ... etc


def get_manual_start_AFTER(input_freq, data_col):
    """NEW VERSION - clean and maintainable."""
    if input_freq == 1.3:
        if data_col == "Probe 1":
            return MANUAL.FREQ_1_3_HZ_PROBE1_START
        elif data_col == "Probe 2":
            return MANUAL.FREQ_1_3_HZ_PROBE1_START + MANUAL.FREQ_1_3_HZ_PROBE2_OFFSET
        elif data_col == "Probe 3":
            p2_start = MANUAL.FREQ_1_3_HZ_PROBE1_START + MANUAL.FREQ_1_3_HZ_PROBE2_OFFSET
            return p2_start + MANUAL.FREQ_1_3_HZ_PROBE3_OFFSET
        elif data_col == "Probe 4":
            # Probe 4 uses same as Probe 3
            p2_start = MANUAL.FREQ_1_3_HZ_PROBE1_START + MANUAL.FREQ_1_3_HZ_PROBE2_OFFSET
            return p2_start + MANUAL.FREQ_1_3_HZ_PROBE3_OFFSET
    
    elif input_freq == 0.65:
        if data_col == "Probe 1":
            return MANUAL.FREQ_0_65_HZ_PROBE1_START
        elif data_col == "Probe 2":
            return MANUAL.FREQ_0_65_HZ_PROBE1_START + MANUAL.FREQ_0_65_HZ_PROBE2_OFFSET
        elif data_col in ["Probe 3", "Probe 4"]:
            p2_start = MANUAL.FREQ_0_65_HZ_PROBE1_START + MANUAL.FREQ_0_65_HZ_PROBE2_OFFSET
            return p2_start + MANUAL.FREQ_0_65_HZ_PROBE3_OFFSET
    
    return None  # No manual override


# =============================================================================
# EXAMPLE 5: Physical calculations
# =============================================================================

def calculate_wavenumbers_BEFORE(frequencies, heights):
    """OLD VERSION."""
    g = 9.81  # hardcoded
    # ... calculation


def calculate_wavenumbers_AFTER(frequencies, heights):
    """NEW VERSION."""
    freq = np.asarray(frequencies)
    H = np.broadcast_to(np.asarray(heights), freq.shape)
    k = np.zeros_like(freq, dtype=float)
    
    valid = freq > 0
    i_valid = np.flatnonzero(valid)
    if i_valid.size == 0:
        return k
    
    for idx in i_valid:
        fr = freq.flat[idx]
        h = H.flat[idx] * MEASUREMENT.MM_TO_M  # Use constant for conversion
        omega = 2 * np.pi * fr
        
        def disp(k_wave):
            return PHYSICS.GRAVITY * k_wave * np.tanh(k_wave * h) - omega**2
        
        k_deep = omega**2 / PHYSICS.GRAVITY
        k_shallow = omega / np.sqrt(PHYSICS.GRAVITY * h) if h > 0 else k_deep
        
        a = min(k_deep, k_shallow) * WAVENUMBER.DEEP_WATER_BRACKET_FACTOR
        b = max(k_deep, k_shallow) * WAVENUMBER.SHALLOW_WATER_BRACKET_FACTOR
        
        # ... rest of Brent's method
        
    return k


# =============================================================================
# EXAMPLE 6: PSD calculation
# =============================================================================

def compute_psd_BEFORE(signal, fs=250):
    """OLD VERSION."""
    desired_resolution = 0.125  # hardcoded
    npersegment = int(fs / desired_resolution)
    noverlap = nperseg // 2
    # ... rest


def compute_psd_AFTER(signal):
    """NEW VERSION."""
    nperseg = int(MEASUREMENT.SAMPLING_RATE / SIGNAL.PSD_FREQUENCY_RESOLUTION)
    noverlap = int(nperseg * SIGNAL.PSD_OVERLAP_FRACTION)
    # ... rest


# =============================================================================
# EXAMPLE 7: Amplitude calculation
# =============================================================================

def calculate_amplitude_BEFORE(signal):
    """OLD VERSION."""
    return (np.percentile(signal, 99.5) - np.percentile(signal, 0.5)) / 2.0


def calculate_amplitude_AFTER(signal):
    """NEW VERSION."""
    upper = np.percentile(signal, AMPLITUDE.UPPER_PERCENTILE)
    lower = np.percentile(signal, AMPLITUDE.LOWER_PERCENTILE)
    return (upper - lower) / AMPLITUDE.AMPLITUDE_DIVISOR


# =============================================================================
# BENEFITS OF THIS APPROACH
# =============================================================================

"""
WHY THIS IS BETTER:

1. **All tuning parameters in one place**
   - No hunting through 955 lines to find magic numbers
   - Easy to see what values you've tried
   
2. **Documented decisions**
   - Comments explain why values were chosen
   - Future you will thank present you
   
3. **Easy experimentation**
   - Change SIGNAL.BASELINE_SIGMA_FACTOR from 1.0 to 1.5 in ONE place
   - Rerun entire analysis with new parameters
   
4. **Version control friendly**
   - Git diff shows exactly which parameters changed
   - Easy to revert to previous values
   
5. **Can create parameter sets**
   - "Conservative" profile for noisy data
   - "Aggressive" profile for clean data
   - Switch between them easily
   
6. **Validation**
   - Check parameters make sense at startup
   - Catch errors before running expensive analysis
   
7. **Self-documenting code**
   - `RAMP.MIN_RAMP_PEAKS` is clearer than `5`
   - Reader knows what the number means
"""


if __name__ == "__main__":
    print("This file shows refactoring examples.")
    print("Copy the patterns you like into your actual code!")
