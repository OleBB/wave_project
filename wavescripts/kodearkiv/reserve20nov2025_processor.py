#copy of processor thursday 20 nov 14:50
# før store endringer i find_wave ranges

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

def find_wave_range(df, data_col,
                    input_volt,
                    input_freq,
                    input_per,
                    detect_win=10,
                    baseline_seconds=2.0,
                    sigma_factor=5.0,
                    skip_periods=None,
                    keep_periods=None,
                    debug=False):
    
    """
    Find a 'good' wave interval for your signal:
    - flat noise first (several seconds)
    - gradual ramp-up of 5–12 periods
    - then stable waves

    Returns
    -------
    (int, int): (good_start_idx, good_end_idx)
    """
    # --- Auto-compute periods if not given ---
    if skip_periods is None:
        # Skip some early transient cycles
        skip_periods = int(max(1, input_freq * 0.3 + input_volt * 0.1))

    if keep_periods is None:
        # Keep enough cycles to capture a full stable wave envelope
        keep_periods = int(max(3, input_freq * 5 + input_volt * 0.5))
    
    # ------
    #print(df.head())
    # 1) Smooth signal for detection
    signal = (df[data_col]
              .rolling(window=detect_win, min_periods=1)
              .mean()
              .values)
    signal = np.nan_to_num(signal)

    # 2) Compute sample rate from Date column
    dt = (df["Date"].loc[1] - df["Date"].loc[0]).total_seconds()
    Fs = 1.0 / dt
    print(f'Fs = {Fs}, and dt = {dt}')

    # 3) Baseline window length in samples
    baseline_samples = int(baseline_seconds * Fs)
    baseline = signal[:baseline_samples]
    baseline_mean = np.mean(baseline)
    baseline_std  = np.std(baseline)
    print(f'from find_wave_range:baseline-MEAN is {baseline_mean}')
    # 4) Movement threshold
    threshold = baseline_std * sigma_factor
    movement = np.abs(signal - baseline_mean)

    # 5) First time we see real motion
    first_motion_idx = np.argmax(movement > threshold)

    # 6) Period-based skipping and windowing
    T = 1.0 / float(input_freq)
    samples_per_period = int(Fs * T)

    good_start_idx = first_motion_idx + skip_periods * samples_per_period
    good_end_idx   = good_start_idx  + keep_periods * samples_per_period
    
    # Clamp to valid range
    good_start_idx = min(good_start_idx, len(signal) - 1)
    good_end_idx   = min(good_end_idx,   len(signal) - 1)
    print(f'from find_wave_range:goodstartindex = {good_start_idx}')
    print("input_freq used:", input_freq)
    print("samples_per_period =", samples_per_period)
    print("first_motion_idx =", first_motion_idx)
    print("skip_periods =", skip_periods)
    print("good_start_idx =", good_start_idx)
    
    if debug:
        debug_plot_ramp_detection(
            df=df,
            data_col=data_col,
            signal=signal,
            baseline_mean=baseline_mean,
            threshold=threshold,
            first_motion_idx=first_motion_idx,
            good_start_idx=good_start_idx,
            good_end_idx=good_end_idx,
            title=f"Ramp Detection Debug – {data_col}"
        )
        
    debug_info = {
        "signal": signal,
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "threshold": threshold,
        "first_motion_idx": first_motion_idx
    }
    return good_start_idx, good_end_idx, debug_info




# === Take in a filtered subset then process === #
def process_selected_data(dfs, df_sel, plotvariables):
    processed = {}
    auto_ranges={}
    debug_data ={}
    
    data_col = plotvariables["processing"]["data_cols"][0] # one column only
    win      = plotvariables["processing"]["win"]
    input_volt      = float(plotvariables["filters"]["amp"])/1000
    input_freq     = float(plotvariables["filters"]["freq"])/1000 #1300 blir til 1.3
    input_per      = float(plotvariables["filters"]["per"])
    
    print(f'amp {input_volt}, input_freq {input_freq}, input_per {input_per}')
    
    # FUTURE: If I should change my Json-entry to the proper 1.3 insted of the 1300
    #Auto-correct: if input_freq is too large, assume it's in mHz
    input_freq = input_freq / 1000 if input_freq > 50 else input_freq
    print('datacol is =',data_col)

    for _, row in df_sel.iterrows():
        path = row["path"]
        df_raw = dfs[path]
        #print('df_raw process_selected_data : ',df_raw.head())
        
        # Step 1: Smooth this probe only
        df_ma = apply_moving_average(df_raw, [data_col], win) 
        #print('df_ma inside process_selected_data where, \n the first number of samples will become Nan because thats how the function works \n',df_ma.head())
        
        # Step 2: Determine where the wave begins and ends
        detect_window = 10
        start, end, debug_info = find_wave_range(df_raw, 
                                     data_col, 
                                     input_volt,
                                     input_freq,
                                     input_per,
                                     detect_win=detect_window, 
                                     debug=True) #her skrur man på debug

        df_ma["wave_start"] = start
        df_ma["wave_end"] = end
        
        processed[path] = df_ma
        auto_ranges[path] = (start, end)
        debug_data[path] = debug_info


    return processed, auto_ranges, debug_data

"""minner om at auto_ranges ikke er fullstendig korrigert
 for hva som er automatisk, hva som er manuelt input
 og hva som er 'final' """

def debug_plot_ramp_detection(df, data_col,
                              signal,
                              baseline_mean,
                              threshold,
                              first_motion_idx,
                              good_start_idx,
                              good_end_idx,
                              title="Ramp Detection Debug"):

    time = df["Date"]

    plt.figure(figsize=(14, 6))

    # Raw probe signal
    plt.plot(time, df[data_col], label="Raw signal", alpha=0.4)

    # Smoothed detection signal
    plt.plot(time, signal, label="Smoothed (detect)", linewidth=2)

    # Baseline mean
    plt.axhline(baseline_mean, color="blue", linestyle="--",
                label=f"Baseline mean = {baseline_mean:.3f}")

    # Threshold region
    plt.axhline(baseline_mean + threshold, color="red", linestyle="--",
                label=f"+ Threshold ({threshold:.3f})")
    plt.axhline(baseline_mean - threshold, color="red", linestyle="--")

    # First motion
    plt.axvline(time.iloc[first_motion_idx],
                color="orange", linestyle="--", linewidth=2,
                label=f"First motion @ {first_motion_idx}")

    # Good interval start / end
    plt.axvline(time.iloc[good_start_idx],
                color="green", linestyle="--", linewidth=2,
                label=f"Good start @ {good_start_idx}")

    plt.axvline(time.iloc[good_end_idx],
                color="purple", linestyle="--", linewidth=2,
                label=f"Good end @ {good_end_idx}")

    # Shaded good region
    plt.axvspan(time.iloc[good_start_idx],
                time.iloc[good_end_idx],
                color="green", alpha=0.15)

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel(data_col)
    plt.legend()
    plt.tight_layout()
    plt.show()



def find_wave_range_text(df, data_col, input_freq, duration_factor=1.5):
    """
    Find the time-window range where the wave starts and define
    an interval based on wave frequency.

    Parameters
    ----------
    df : pandas.DataFrame
        Single raw or processed dataframe.
    data_col : str
        Single probe column to analyze (e.g. "Probe 3").
    freq : float
        Wave frequency in Hz.
    duration_factor : float
        How many wave periods the returned interval should cover.

    Returns
    -------
    (int, int)
        start index, end index of the window
    """

def find_wave_range_originali(df, data_cols):
    #tar inn valgt dataframe
    #tar inn utvalgte kolonner
    
    #finner første skikkelige topp, ved å se etter ramp-up?
    #   eller, ved å se etter aller største...?
    #finner siste skikkelige topp. 
    #beregner avstand basert på innkommende bølgeparametere
    #velger de bølgene mellom
    
    return

def find_resting_levels():
    resting_files = [f for f in CSVFILES if 'nowind' in f.lower()]
    if not resting_files:
        raise ValueError("No valid nowind-files found to compute resting level")
    
    
    for f in resting_files:
        df99 = df99.copy()
    return


def remove_outliers():
    #lag noe basert på steepness, kanskje tilogmed ak. Hvis ak er for bratt
    # og datapunktet for høyt, så må den markeres, og så fjernes.
    return
    
# ------------------------------------------------------------
# Moving average helper
# ------------------------------------------------------------
def apply_moving_average(df, data_cols, win=1):
    df_ma = df.copy()
    #print('inside moving average: ', df_ma.head())
    df_ma[data_cols] = df[data_cols].rolling(window=win, min_periods=win).mean()
    return df_ma

# ------------------------------------------------------------
# Ny funksjon
# ------------------------------------------------------------
def compute_simple_amplitudes(df_ma, chosenprobe, n):
    top_n = df_ma[chosenprobe].nlargest(n)
    bottom_n = df_ma[chosenprobe].nsmallest(n)
    average = (top_n.sum()-bottom_n.sum())/(len(top_n))
    return average #top_n, bottom_n

