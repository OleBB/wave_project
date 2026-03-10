#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 17:18:03 2025

@author: ole
"""
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

from typing import Dict, List, Tuple

from wavescripts.improved_data_loader import update_processed_metadata, get_configuration_for_date
from wavescripts.wave_detection import find_wave_range
from wavescripts.signal_processing import compute_psd_with_amplitudes, compute_fft_with_amplitudes, compute_amplitudes
from wavescripts.wave_physics import calculate_wavenumbers_vectorized, calculate_wavedimensions, calculate_windspeed

from wavescripts.constants import SIGNAL, RAMP, MEASUREMENT, get_smoothing_window
from wavescripts.constants import (
    ProbeColumns as PC, 
    GlobalColumns as GC, 
    ColumnGroups as CG,
    CalculationResultColumns as RC
)


# ========================================================== #
# === Make sure stillwater levels are computed and valid === #
# ========================================================== #
def ensure_stillwater_columns(
    dfs: dict[str, pd.DataFrame],
    meta: pd.DataFrame,
    cfg,
) -> pd.DataFrame:
    """
    Computes the true still-water level for each probe using ALL "no wind" runs,
    then copies that value into EVERY row of the metadata (including windy runs).
    Safe to call multiple times.
    """
    probe_cols = [f"Stillwater Probe {i}" for i in range(1, 5)]

    if all(col in meta.columns for col in probe_cols):
        if meta[probe_cols].notna().all().all():
            print("Stillwater levels already computed and valid → skipping")
            return meta

    print("Computing still-water levels from all 'WindCondition == no' runs...")

    mask = meta["WindCondition"].astype(str).str.strip().str.lower() == "no"
    nowind_paths = meta.loc[mask, "path"].tolist()

    if not nowind_paths:
        raise ValueError("No runs with WindCondition == 'no' found! Cannot compute still water.")

    col_names = cfg.probe_col_names()  # {1: "9373/170", ...}
    stillwater_values = {}
    for i, pos in col_names.items():
        probe_col = f"Probe {pos}"
        all_values = []

        for path in nowind_paths:
            if path in dfs:
                df = dfs[path]
                if probe_col in df.columns:
                    clean = pd.to_numeric(df[probe_col], errors='coerce').dropna()
                    all_values.extend(clean.tolist())

        if len(all_values) == 0:
            raise ValueError(f"No valid data found for {probe_col} in any no-wind run!")

        level = np.median(all_values)
        stillwater_values[f"Stillwater Probe {i}"] = float(level)
        print(f"  Stillwater Probe {i} ({probe_col}): {level:.3f} mm  (from {len(all_values):,} samples)")

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


def remove_outliers():
    #lag noe basert på steepness, kanskje tilogmed ak. Hvis ak er for bratt
    # og datapunktet for høyt, så må den markeres, og så fjernes.
    #se Karens script
    return

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
    cfg,
    win: int,
    debug: bool
) -> dict[str, pd.DataFrame]:
    """Zero signals using stillwater and add moving averages."""
    col_names = cfg.probe_col_names()  # {1: "9373/170", 2: "12545", ...}
    processed_dfs = {}
    for _, row in meta_sel.iterrows():
        path = row["path"]
        if path not in dfs:
            print(f"Warning: File not loaded: {path}")
            continue

        df = dfs[path].copy()
        for i, pos in col_names.items():
            probe_col = f"Probe {pos}"
            if probe_col not in df.columns:
                print(f"  Missing column {probe_col} in {Path(path).name}")
                continue

            eta_col = f"eta_{pos}"
            df[eta_col] = -(df[probe_col] - stillwater[i])
            df[f"{probe_col}_ma"] = df[eta_col].rolling(window=win, center=False).mean()

            if debug:
                print(f"  {Path(path).name:35} → {eta_col} mean = {df[eta_col].mean():.4f} mm")

        processed_dfs[path] = df

    return processed_dfs


def run_find_wave_ranges(
    processed_dfs: dict,
    meta_sel: pd.DataFrame,
    cfg,
    win: int,
    range_plot: bool,
    debug: bool
) -> pd.DataFrame:
    """Find wave ranges for all probes."""
    col_names = cfg.probe_col_names()  # {1: "9373/170", ...}
    for idx, row in meta_sel.iterrows():
        path = row["path"]
        df = processed_dfs[path]

        for i, pos in col_names.items():
            probe_col = f"Probe {pos}"
            start, end, debug_info = find_wave_range(
                df, row, data_col=probe_col, probe_num=i, detect_win=win, range_plot=range_plot, debug=debug
            )
            meta_sel.loc[idx, f"Computed Probe {pos} start"] = start
            meta_sel.loc[idx, f"Computed Probe {pos} end"] = end

        if debug and start:
            print(f'start: {start}, end: {end}, debug: {debug_info}')

    return meta_sel


# def _update_all_metrics(
#     processed_dfs: dict,
#     meta_sel: pd.DataFrame,
#     stillwater: dict,
#     amplitudes_psd_df: pd.DataFrame,
#     amplitudes_fft_df: pd.DataFrame,
# ) -> pd.DataFrame:
#     """
#     Calculate and update all computed metrics in metadata.
    
#     This function handles TWO types of updates:
#     1. DIRECT ASSIGNMENT: Pre-computed amplitudes from PSD/FFT analysis
#     2. DERIVED CALCULATIONS: Wavenumbers, wavelengths, etc. based on the amplitudes
#     """
#     meta_indexed = meta_sel.set_index("path").copy()
    
#     # ============================================================================
#     # SECTION 1: DIRECT ASSIGNMENT of pre-computed values
#     # ============================================================================
    
#     # Amplitudes from np.percentile
#     amplitudes = compute_amplitudes(processed_dfs, meta_sel)
#     amp_cols = [f"Probe {i} Amplitude" for i in range(1, 5)]
#     meta_indexed.update(amplitudes.set_index("path")[amp_cols])
    
#     # Amplitudes from PSD
#     psd_cols = [f"Probe {i} Amplitude (PSD)" for i in range(1, 5)]
#     meta_indexed[psd_cols] = amplitudes_psd_df.set_index("path")[psd_cols]
    
#     # Amplitudes AND periods from FFT (both needed for downstream calculations)
#     fft_amplitude_cols = [f"Probe {i} Amplitude (FFT)" for i in range(1, 5)]
#     fft_period_cols = [f"Probe {i} WavePeriod (FFT)" for i in range(1, 5)]
#     fft_freq_cols = [f"Probe {i} Frequency (FFT)" for i in range(1, 5)]
    
#     fft_df_indexed = amplitudes_fft_df.set_index("path")
#     meta_indexed[fft_amplitude_cols] = fft_df_indexed[fft_amplitude_cols]
#     meta_indexed[fft_period_cols] = fft_df_indexed[fft_period_cols]
#     meta_indexed[fft_freq_cols] = fft_df_indexed[fft_freq_cols]
    
#     # ============================================================================
#     # SECTION 2: DERIVED CALCULATIONS using the assigned values
#     # ============================================================================
    
#     # Define probe configurations
#     probe_config = {
#         1: ("Probe 1 Frequency (FFT)", "Probe 1 Wavenumber (FFT)"),
#         2: ("Probe 2 Frequency (FFT)", "Probe 2 Wavenumber (FFT)"),
#         3: ("Probe 3 Frequency (FFT)", "Probe 3 Wavenumber (FFT)"),
#         4: ("Probe 4 Frequency (FFT)", "Probe 4 Wavenumber (FFT)"),
#     }

#     # Process all probes - vectorized for speed
#     for i, (freq_col, k_col) in probe_config.items():
#         # Convert Period to Frequency: f = 1/T
#         freq_data = meta_indexed[freq_col]
        
#         # Vectorized Wavenumber Calculation
#         meta_indexed[k_col] = calculate_wavenumbers_vectorized(
#             frequencies=freq_data,
#             heights=meta_indexed["WaterDepth [mm]"]
#         )
        
#         # Vectorized Dimension Calculation
#         res = calculate_wavedimensions(
#             k=meta_indexed[k_col],
#             H=meta_indexed["WaterDepth [mm]"],
#             PC=meta_indexed["PanelCondition"],
#             amp=meta_indexed[f"Probe {i} Amplitude"]
#         )
        
#         # Bulk assign results
#         target_cols = [f"Probe {i} Wavelength (FFT)", f"Probe {i} kL (FFT)", 
#                        f"Probe {i} ak (FFT)", f"Probe {i} tanh(kH) (FFT)", 
#                        f"Probe {i} Celerity (FFT)"]
#         source_cols = ["Wavelength", "kL", "ak", "tanh(kH)", "Celerity"]
#         meta_indexed[target_cols] = res[source_cols]

#     # Process the 'Given' / 'Global' columns
#     meta_indexed["Wavenumber"] = calculate_wavenumbers_vectorized(
#         frequencies=meta_indexed["WaveFrequencyInput [Hz]"],
#         heights=meta_indexed["WaterDepth [mm]"]
#     )
    
#     global_res = calculate_wavedimensions(
#         k=meta_indexed["Wavenumber"],
#         H=meta_indexed["WaterDepth [mm]"],
#         PC=meta_indexed["PanelCondition"],
#         amp=meta_indexed["Probe 2 Amplitude"]
#     )

#     final_cols = ["Wavelength", "kL", "ak", "kH", "tanh(kH)", "Celerity"]
#     meta_indexed[final_cols] = global_res[final_cols]
    
#     # Windspeed
#     meta_indexed["Windspeed"] = calculate_windspeed(meta_indexed["WindCondition"])
    
#     # Add stillwater columns if missing
#     for i in range(1, 5):
#         col = f"Stillwater Probe {i}"
#         if col not in meta_indexed.columns:
#             meta_indexed[col] = stillwater[i]
    
#     return meta_indexed.reset_index()

def _update_all_metrics(
    processed_dfs: dict,
    meta_sel: pd.DataFrame,
    stillwater: dict,
    amplitudes_psd_df: pd.DataFrame,
    amplitudes_fft_df: pd.DataFrame,
    cfg,
) -> pd.DataFrame:
    """Update metadata with all computed metrics, using position-based probe names."""
    meta_indexed = meta_sel.set_index(GC.PATH).copy()
    col_names = cfg.probe_col_names()  # {1: "9373/170", 2: "12545", ...}

    # ============================================================================
    # SECTION 1: DIRECT ASSIGNMENT of pre-computed values
    # ============================================================================

    # Amplitudes from np.percentile — columns already position-based
    amplitudes = compute_amplitudes(processed_dfs, meta_sel, cfg)
    amp_cols = [c for c in amplitudes.columns if c != GC.PATH]
    meta_indexed[amp_cols] = amplitudes.set_index(GC.PATH)[amp_cols]

    # Amplitudes from PSD
    psd_cols = [c for c in amplitudes_psd_df.columns if c != GC.PATH]
    meta_indexed[psd_cols] = amplitudes_psd_df.set_index(GC.PATH)[psd_cols]

    # FFT amplitudes, frequencies, periods
    fft_df_indexed = amplitudes_fft_df.set_index(GC.PATH)
    fft_cols = [c for c in amplitudes_fft_df.columns if c != GC.PATH]
    meta_indexed[fft_cols] = fft_df_indexed[fft_cols]

    # ============================================================================
    # SECTION 2: DERIVED CALCULATIONS using the assigned values
    # ============================================================================

    for i, pos in col_names.items():
        freq_col = f"Probe {pos} Frequency (FFT)"
        k_col    = f"Probe {pos} Wavenumber (FFT)"
        amp_col  = f"Probe {pos} Amplitude"

        meta_indexed[k_col] = calculate_wavenumbers_vectorized(
            frequencies=meta_indexed[freq_col],
            heights=meta_indexed[GC.WATER_DEPTH]
        )

        res = calculate_wavedimensions(
            k=meta_indexed[k_col],
            H=meta_indexed[GC.WATER_DEPTH],
            PC=meta_indexed[GC.PANEL_CONDITION],
            amp=meta_indexed[amp_col]
        )

        wave_dim_cols = [
            f"Probe {pos} Wavelength (FFT)",
            f"Probe {pos} kL (FFT)",
            f"Probe {pos} ak (FFT)",
            f"Probe {pos} tanh(kH) (FFT)",
            f"Probe {pos} Celerity (FFT)",
        ]
        meta_indexed[wave_dim_cols] = res[RC.WAVE_DIMENSION_COLS]

    # Global wave dimensions (use IN probe amplitude as representative)
    in_pos = col_names[cfg.in_probe]
    meta_indexed[GC.WAVENUMBER] = calculate_wavenumbers_vectorized(
        frequencies=meta_indexed[GC.WAVE_FREQUENCY_INPUT],
        heights=meta_indexed[GC.WATER_DEPTH]
    )
    global_res = calculate_wavedimensions(
        k=meta_indexed[GC.WAVENUMBER],
        H=meta_indexed[GC.WATER_DEPTH],
        PC=meta_indexed[GC.PANEL_CONDITION],
        amp=meta_indexed[f"Probe {in_pos} Amplitude"]
    )
    meta_indexed[CG.GLOBAL_WAVE_DIMENSION_COLS] = global_res[RC.WAVE_DIMENSION_COLS_WITH_KH]

    # Windspeed
    meta_indexed[GC.WINDSPEED] = calculate_windspeed(meta_indexed[GC.WIND_CONDITION])

    # Stillwater (keyed by probe number — physical measurement, not position)
    for i in range(1, 5):
        col = PC.STILLWATER.format(i=i)
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
    fs = MEASUREMENT.SAMPLING_RATE
    # 0. unpack processvariables
    prosessering = processvariables.get("prosessering", {})

    force_recompute = prosessering.get("force_recompute", False)
    debug = prosessering.get("debug", False)
    win = prosessering.get("smoothing_window", 1)
    find_range = prosessering.get("find_range", False)
    range_plot = prosessering.get("range_plot", False)

    # Derive cfg once for the whole folder (all files share the same configuration)
    file_date = datetime.fromisoformat(str(meta_sel["file_date"].iloc[0]))
    cfg = get_configuration_for_date(file_date)

    # 1. Ensure stillwater levels are computed
    meta_full = ensure_stillwater_columns(dfs, meta_full, cfg)
    stillwater = _extract_stillwater_levels(meta_full, debug)

    # 2. Process dataframes: zero and add moving averages
    processed_dfs = _zero_and_smooth_signals(dfs, meta_sel, stillwater, cfg, win, debug)
    
    # 3. Optional: find wave ranges
    if find_range:
        meta_sel = run_find_wave_ranges(processed_dfs, meta_sel, cfg, win, range_plot, debug)
    
    # 4. a - Compute PSDs and amplitudes from PSD
    psd_dict, amplitudes_psd_df = compute_psd_with_amplitudes(processed_dfs, meta_sel, cfg, fs=fs, debug=debug)

    # 4. b - compute FFT and amplitudes from FFT
    fft_dict, amplitudes_fft_df = compute_fft_with_amplitudes(processed_dfs, meta_sel, cfg, fs=fs, debug=debug)

    # 5. Compute and update all metrics (amplitudes, wavenumbers, dimensions, windspeed)
    meta_sel = _update_all_metrics(processed_dfs, meta_sel, stillwater, amplitudes_psd_df, amplitudes_fft_df, cfg)

    # 6. Set output folder and save metadata
    meta_sel = _set_output_folder(meta_sel, meta_full, debug)

    update_processed_metadata(meta_sel, force_recompute=force_recompute)

    if debug:
        print(f"\nProcessing complete! {len(processed_dfs)} files zeroed and ready.")
    
    return processed_dfs, meta_sel, psd_dict, fft_dict
