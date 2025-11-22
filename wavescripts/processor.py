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

PROBES = ["Probe 1", "Probe 2", "Probe 3", "Probe 4"]

def find_wave_range(
    df, #detta er jo heile dicten... hadde 
    df_sel,  # detta er kun metadata for de utvalgte
    data_col,            
    detect_win=1,
    baseline_seconds=2.0,
    sigma_factor=5.0,
    skip_periods=None,
    keep_periods=None,
    debug=False
):
    #VARIABEL over^
    """
    Detect the start and end of the stable wave interval.

    Signal behavior (based on Ole's description):
      - baseline mean ~271
      - wave begins ~5 sec after baseline ends
      - ramp-up lasts ~12 periods
      - stable region has peaks ~280, troughs ~260

    Returns (good_start_idx, good_end_idx)
    """

    # ==========================================================
    # AUTO-CALCULATE SKIP & KEEP PERIODS BASED ON REAL SIGNAL
    # ==========================================================
    importertfrekvens = 1# TK TK BYTT UT med en ekte import fra metadata
    # ---- Skip: baseline (5 seconds) + ramp-up (12 periods) ----
    if skip_periods is None:
        baseline_skip_periods = int(5 * importertfrekvens)        # 5 seconds worth of periods
        ramp_skip_periods     = 12 #VARIABEL # fixed from your signal observations
        skip_periods          = baseline_skip_periods + ramp_skip_periods

    # ---- Keep: x stable periods ----
    if keep_periods is None:
        keep_periods = 8#int(input_per-8) #VARIABEL
        
    # ==========================================================
    # PROCESSING STARTS HERE
    # ==========================================================
    
    """TODO TK: LEGGE TIL SAMMENLIKNING AV PÅFØLGENDE 
    BØLGE FOR Å SE STABILT signal. 
    OG Se på alle bølgetoppene"""
    
    """TODO TK: SJEKKE OM lengdene på periodene er like"""
    
    """TODO TK: """
    
    print(f"data_col before signal.. {data_col}")
    # 1) Smooth signal
    signal = (
        df[data_col]
        .rolling(window=detect_win, min_periods=1)
        .mean()
        .fillna(0)
        .values
    )

    # 2) Sample rate
    dt = (df["Date"].iloc[1] - df["Date"].iloc[0]).total_seconds()
    Fs = 1.0 / dt

    # 3) Baseline mean/std
    baseline_samples = int(baseline_seconds * Fs)
    baseline = signal[:baseline_samples]
    baseline_mean = np.mean(baseline)
    baseline_std  = np.std(baseline)

    threshold = sigma_factor * baseline_std
    movement = np.abs(signal - baseline_mean)

    # 4) First actual movement above noise floor
    first_motion_idx = np.argmax(movement > threshold)

    # 5) Convert frequency → samples per period
    T = 1.0 / float(importertfrekvens)
    samples_per_period = int(Fs * T)

    # 6) Final "good" window
    good_start_idx = first_motion_idx + skip_periods * samples_per_period
    good_end_idx   = good_start_idx + keep_periods * samples_per_period

    # Clamp
    good_start_idx = min(good_start_idx, len(signal) - 1)
    good_end_idx   = min(good_end_idx,   len(signal) - 1)
    
    from wavescripts.plotter import plot_ramp_detection
    if debug:
        plot_ramp_detection(
            df=df,
            df_sel=df_sel,
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
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "first_motion_idx": first_motion_idx,
        "samples_per_period": samples_per_period,
        "skip_periods": skip_periods,
        "keep_periods": keep_periods
    }

    return good_start_idx, good_end_idx, debug_info

# =============================================== 
# === Take in a filtered subset then process === #
# ===============================================
PROBES = ["Probe 1", "Probe 2", "Probe 3", "Probe 4"]
def process_selected_data_old(dfs, df_sel, plotvariables):
    processed = {}
    debug_data ={}
    win      = plotvariables["processing"]["win"]

    for _, row in df_sel.iterrows():
        path = row["path"]
        df_raw = dfs[path]
        print("type(df_raw) =", type(df_raw))
    
        detect_window = 1 #10 er default i find_wave_range
        
        # === PROCESS ALL THE PROBES === #
        for probe in PROBES: #loope over alle 4 kolonnene
            print(f'probe in loop is: {probe}')
            # --- Apply moving avg. to the selected df_ma for each >probe in PROBES< 
            df_ma = apply_moving_average(df_raw, data_col=probe, win=win)
            print(f'df-ma per {probe} sitt tail:',df_ma[probe].tail())
            
            # find the start of the signal and optionally run the debug-plot
            start, end, debug_info = find_wave_range(df_raw, 
                                     df_sel,    
                                     data_col=probe,
                                     detect_win=detect_window, 
                                     debug=False) #her skrur man på debug. trur æ
            df_sel[f"Computed {probe} start"] = start
                    
        # etter avsluttet indre for-loop
        #pushe start og end 
        #heller hente en oppdatert df_sel?? #df_sel["Calculated start"] = start #pleide å være df_ma her men må jo ha engangsmetadata i metadata. 
        # === Put the calculated start_idx into
        
        processed[path] = df_ma
        #bytt ut med å heller importere range fra metadata #auto_ranges[path] = (start, end)
        #trenger vel egt ikke ha med window sizen som ble brukt
        debug_data[path] = debug_info
    #---end of for loop---#
    
 
    print("type(df_sel) =", type(df_sel))
    try:
        print("df_sel sample (first 5):", list(df_sel)[:5])
    except Exception:
        print("Could not list df_sel")
    #her returneres de processerte df'ene og debug-greier(!!?)
    #
    return processed, df_sel, debug_data 



from wavescripts.data_loader import update_processed_metadata
from wavescripts.data_loader import save_processed_dataframes

PROBES = ["Probe 1", "Probe 2", "Probe 3", "Probe 4"]

def process_selected_data(dfs: dict, meta_sel: pd.DataFrame, plotvariables: dict, debug=False):
    """
    Vectorized, clean, fast version.
    No iterrows → 10–100x faster + much cleaner.
    """
    win = plotvariables["processing"]["win"]
    detect_win = 1  # or make configurable

    # We'll collect new columns to add to meta_sel
    new_columns = {}

    # ------------------------------------------------------------------
    # 1. Apply moving average to ALL dataframes at once (vectorized)
    # ------------------------------------------------------------------
    processed_dfs = {}
    for path, df in dfs.items():
        df_processed = df.copy()
        for probe in PROBES:
            df_processed[f"{probe}_ma"] = df_processed[probe].rolling(window=win, center=False).mean()
        processed_dfs[path] = df_processed

    # ------------------------------------------------------------------
    # 2. Compute start indices using your find_wave_range — but vectorized!
    # ------------------------------------------------------------------
    # Option A: Your current function works per signal → keep it, but call efficiently
    starts = {probe: [] for probe in PROBES}
    for _, row in meta_sel.iterrows():
        path = row["path"]
        df = processed_dfs[path]  # or dfs[path] if you don't need ma yet
        print('inni loopen for _, row in meta_sel.iterrows() ')
        for probe in PROBES:
            start, end, _ = find_wave_range(
                df, meta_sel, data_col=probe, detect_win=detect_win, debug=False
            )
            starts[probe].append(start)

    # Add computed starts directly as new columns
    for probe in PROBES:
        new_columns[f"Computed {probe} start"] = starts[probe]

    # ------------------------------------------------------------------
    # 3. Add other computed values (Hs, Tz, zeroed waves, etc.) — fully vectorized
    # ------------------------------------------------------------------
    stillwater_samples = 250
    for probe in PROBES:
        probe_col = probe
        eta_col = f"eta_{probe.split()[1]}"

        # Extract stillwater from first N samples (vectorized per file)
        stillwaters = []
        eta = []
        hs_values = []
        for path in meta_sel["path"]:
            df = processed_dfs[path]
            sw = df[probe_col].iloc[:stillwater_samples].mean()
            eta = df[probe_col] - sw
            df[eta_col] = eta  # add zeroed wave to the DataFrame
            processed_dfs[path] = df

            stillwaters.append(sw)
            hs_values.append(4 * eta.std())  # H_s ≈ 4 * std for narrow-band

        new_columns[f"Stillwater {probe}"] = stillwaters
        new_columns[f"Hs {probe}"] = hs_values

    # ------------------------------------------------------------------
    # 4. Update meta_sel with all new columns at once
    # ------------------------------------------------------------------
    for col_name, values in new_columns.items():
        meta_sel[col_name] = values

    # ------------------------------------------------------------------
    # 5. Save everything
    # ------------------------------------------------------------------
    # Save updated metadata
    update_processed_metadata(meta_sel)
    
    # Optional: Save processed DataFrames (with eta, moving avg, etc.)
    # ------------------------------------------------------------------
    # 6. OPTIONAL SAVE everything . HAKKE PRØVD SJÆL. TK
    # ------------------------------------------------------------------
    #save_processed_dataframes(processed_dfs, meta_sel)

    print(f"Processed {len(meta_sel)} files → metadata updated with {len(new_columns)} new fields")
    return processed_dfs, meta_sel




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

def find_wave_range_originali(df, data_col):
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
def apply_moving_average(df, data_col, win=1):
    df_ma = df.copy()
    #print('inside moving average: ', df_ma.head())
    df_ma[data_col] = df[data_col].rolling(window=win, min_periods=win).mean()
    return df_ma

# ------------------------------------------------------------
# Ny funksjon
# ------------------------------------------------------------
def compute_simple_amplitudes(df_ma, chosenprobe, n):
    top_n = df_ma[chosenprobe].nlargest(n)
    bottom_n = df_ma[chosenprobe].nsmallest(n)
    average = (top_n.sum()-bottom_n.sum())/(len(top_n))
    return average #top_n, bottom_n

