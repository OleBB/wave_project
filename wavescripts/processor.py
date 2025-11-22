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
from wavescripts.data_loader import update_processed_metadata
from wavescripts.data_loader import save_processed_dataframes
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

    # Robust median

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

def process_selected_data(
    dfs: dict[str, pd.DataFrame],
    meta_sel: pd.DataFrame,
    meta_full: pd.DataFrame,
    debug: bool = True,
    win: int = 10,
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

    # 2. Process only the selected runs
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

    # 3. Make sure meta_sel has the stillwater columns too (for plotting later)
    for i in range(1, 5):
        col = f"Stillwater Probe {i}"
        if col not in meta_sel.columns:
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

    print(f"\nProcessing complete! {len(processed_dfs)} files zeroed and ready.")
    return processed_dfs, meta_sel

def process_selected_data_old(
    dfs: dict[str, pd.DataFrame],
    meta_sel: pd.DataFrame,
    meta_full: pd.DataFrame,   # full meta of the experiment
    debug: bool = True,
    win: int = 10,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:

    # 1. Ensure stillwater levels exist (idempotent — safe to call anytime)
    meta_full = ensure_stillwater_columns(dfs, meta_full)

    # Extract the values (they are the same in every row!)
    stillwater = {f"Stillwater Probe {i}": meta_full[f"Stillwater Probe {i}"].iloc[0] for i in range(1,5)}

    for i in range(1,5):
        val = stillwater[f"Stillwater Probe {i}"]
        print(f"  Stillwater Probe {i} → {val!r}  (type: {type(val).__name__})")
    print(f'stillwater = hva , jo: {stillwater}')
    # 2. Process only selected runs
    processed_dfs = {}
    for _, row in meta_sel.iterrows():
        path = row["path"]
        df = dfs[path].copy()
        # Clean columns once and for all
        for i in range(1, 5):
            col = f"Probe {i}"                    
        for i in range(1, 5):
            probe = f"Probe {2}"
            sw = stillwater[f"Stillwater Probe {i}"] 
            df[f"eta_{i}"] = df[probe] - sw
            df[f"{probe}_ma"] = df[f"eta_{i}"].rolling(win, center=False).mean()
    
        processed_dfs[path] = df
    
    # Add stillwater columns to meta_sel too (in case they were missing)
    for col, val in stillwater.items():
        if col not in meta_sel.columns:
            meta_sel[col] = val
    #for kolonne, value in dfs[riktig path]:
    
        # === DEBUG === #
    #find_wave_range(df_raw, df_sel,data_col=probe, detect_win=detect_window, debug=False)   
    
    # Ensure meta_sel knows which folder to save into
    if "PROCESSED_folder" not in meta_sel.columns:
        if "PROCESSED_folder" in meta_full.columns:
            folder = meta_full["PROCESSED_folder"].iloc[0]
        elif "experiment_folder" in meta_full.columns:
            folder = "PROCESSED-" + meta_full["experiment_folder"].iloc[0]
        else:
            raw_folder = Path(meta_full["path"].iloc[0]).parent.name
            folder = f"PROCESSED-{raw_folder}"
        
        meta_sel["PROCESSED_folder"] = folder
        print(f"Set PROCESSED_folder = {folder}")

    update_processed_metadata(meta_sel)

    return processed_dfs, meta_sel

#######################
# =============================================== 
# === OLD OLD OLD Take in a filtered subset then process === #
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
                                     debug=False) 
            df_sel[f"Computed {probe} start"] = start
        processed[path] = df_ma
        debug_data[path] = debug_info
    return processed, df_sel, debug_data 


######################




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

