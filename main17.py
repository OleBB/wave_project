#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 10:21:49 2025

@author: ole
"""

import os
from pathlib import Path
import pandas as pd
file_dir = Path(__file__).resolve().parent
os.chdir(file_dir) # Make the script always run from the folder where THIS file lives
from wavescripts.data_loader import load_or_update
from wavescripts.filters import filter_chosen_files
from wavescripts.processor import process_selected_data
from wavescripts.processor2nd import process_processed_data



# List of dataset paths you want to process
dataset_paths = [
    Path("/Users/ole/Kodevik/wave_project/wavedata/20251110-tett6roof-lowMooring"),
    Path("/Users/ole/Kodevik/wave_project/wavedata/20251110-tett6roof-lowM-ekte580"),  # per15
    Path("/Users/ole/Kodevik/wave_project/wavedata/20251112-tett6roof"),
    Path("/Users/ole/Kodevik/wave_project/wavedata/20251113-tett6roof-loosepaneltaped")
]

#%%
# Initialize containers for all results
all_meta_sel = []
all_processed_dfs = []

# === Config ===
chooseAll = False
chooseFirst = False
# range debug and plot
debug = True
win = 10
find_range = True
range_plot = True

processvariables = {
    "filters": {
        "amp": 0.1,  # 0.1, 0.2, 0.3 
        "freq": 1.3,  # bruk et tall  
        "per": None,  # bruk et tall #brukes foreløpig kun til find_wave_range, ennå ikke knyttet til filtrering
        "wind": None,  # full, no, lowest
        "tunnel": None,
        "mooring": "low",
        "panel": ["full", "reverse"],  # no, full, reverse, 
    },
    "processing": {
        "chosenprobe": "Probe 3",  # ikkje i bruk
        "rangestart": None,  # ikkje i bruk
        "rangeend": None,  # ikkje i bruk
        "data_cols": ["Probe 2"],  # ikkje i bruk
        "win": 11  # ikkje i bruk
    },
    "plotting": {
        "figsize": None,
        "separate": True,
        "overlay": False   
    }
}

# Loop through each dataset
for i, data_path in enumerate(dataset_paths):
    print(f"\n{'='*50}")
    print(f"Processing dataset {i+1}/{len(dataset_paths)}: {data_path.name}")
    print(f"{'='*50}")
    
    try:
        # Load data
        dfs, meta = load_or_update(data_path)
        
        # === Filter ===
        print('# === Filter === #')
        meta_sel = filter_chosen_files(meta, processvariables, chooseAll, chooseFirst)
        
        print('# === Single probe process === #')
        processed_dfs, meta_sel = process_selected_data(
            dfs, meta_sel, meta, debug, win, find_range, range_plot
        )
    
        print('# === Probe comparison processing === #')
        processed_dfs, meta_sel = process_processed_data(
            dfs, meta_sel
        )
        
        
        
        # Store results
        all_meta_sel.append(meta_sel)
        all_processed_dfs.append(processed_dfs)
        
        print(f"Successfully processed {len(meta_sel)} selections from {data_path.name}")
        
    except Exception as e:
        print(f"Error processing {data_path.name}: {str(e)}")
        continue

# After the loop, you can analyze all meta_sel together
print(f"\n{'='*50}")
print("Processing complete!")
print(f"Total datasets processed: {len(all_meta_sel)}")
print(f"Total selections across all datasets: {sum(len(sel) for sel in all_meta_sel)}")
print(f"{'='*50}")

# Example: Combine all meta_sel into a single DataFrame for analysis
if all_meta_sel:
    combined_meta_sel = pd.concat(all_meta_sel, ignore_index=True)
    print("\nCombined meta_selections ready for analysis:")
    print(combined_meta_sel.head())
    
    # You can now analyze the combined data
    # For example:
    # - Count selections by dataset
    # - Analyze properties across all selections
    # - Compare results between different datasets

# %%


from wavescripts.wavestudyer import compare_probe_amplitudes_and_lag, amplitude_overview, full_tank_diagnostics, wind_damping_analysis 
summary_df = wind_damping_analysis(combined_meta_sel)

#%%
from wavescripts.wavestudyer import wind_damping_analysis
damping_analysis_results = wind_damping_analysis(combined_meta_sel)



























