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
    #Path("/Users/ole/Kodevik/wave_project/wavedata/20251110-tett6roof-lowM-ekte580"),  # per15
    Path("/Users/ole/Kodevik/wave_project/wavedata/20251112-tett6roof"),
    Path("/Users/ole/Kodevik/wave_project/wavedata/20251113-tett6roof-loosepaneltaped")
]

#%%
# Initialize containers for all results
all_meta_sel = []
all_processed_dfs = []

# === Config ===
chooseAll = True
chooseFirst = False
debug = False
win = 10
find_range = True
range_plot = False

processvariables = {
    "filters": {
        "amp": 0.1,  # 0.1, 0.2, 0.3 
        "freq": 1.3,  # bruk et tall  
        "per": None,  # bruk et tall #brukes foreløpig kun til find_wave_range, ennå ikke knyttet til filtrering
        "wind": None,  # full, no, lowest
        "tunnel": None,
        "mooring": "low",
        "panel": ["full", "reverse"],  # no, full, reverse, 
    }
}

# Loop through each dataset
for i, data_path in enumerate(dataset_paths):
    print(f"\n{'='*50}")
    print(f"Processing dataset {i+1}/{len(dataset_paths)}: {data_path.name}")
    print(f"{'='*50}")
    try:
        dfs, meta = load_or_update(data_path)
        
        print('# === Filter === #')
        meta_sel = filter_chosen_files(meta, processvariables, chooseAll, chooseFirst)
        
        print('# === Single probe process === #')
        processed_dfs, meta_sel = process_selected_data( dfs, meta_sel, meta, debug, win, find_range, range_plot)
    
        print('# === Probe comparison processing === #')
        meta_sel = process_processed_data(dfs, meta_sel)
        all_meta_sel.append(meta_sel)
        all_processed_dfs.append(processed_dfs)
        print(f"Successfully processed {len(meta_sel)} selections from {data_path.name}")
        
    except Exception as e:
        print(f"Error processing {data_path.name}: {str(e)}")
        continue
print(f"\n{'='*50}")
print(f"PROCESSING COMPLETE - Total datasets processed: {len(all_meta_sel)}")
print(f"Total selections across all datasets: {sum(len(sel) for sel in all_meta_sel)}")
print(f"{'='*50}")

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
# from wavescripts.wavestudyer import compare_probe_amplitudes_and_lag, amplitude_overview, full_tank_diagnostics, wind_damping_analysis 
# summary_df = wind_damping_analysis(combined_meta_sel)

#%%
from wavescripts.wavestudyer import wind_damping_analysis
damping_analysis_results = wind_damping_analysis(combined_meta_sel)

# %%
from wavescripts.wavestudyer import damping
damping_comparison_df = damping(combined_meta_sel)

# %%


import matplotlib.pyplot as plt
# Extract mean values and reset index
mean_p3p2 = damping_comparison_df['mean'].reset_index()

# Simple plot
plt.figure(figsize=(10, 6))
for condition in ['no', 'lowest', 'full']:
    subset = mean_p3p2[mean_p3p2['WindCondition'] == condition]
    plt.scatter(subset['kL'], subset['mean'], label=condition)

plt.xlabel('kL (wavenumber x geometry length')
plt.ylabel('Mean P3/P2')
plt.legend()
plt.grid()
plt.minorticks_on() 
plt.show()

# %%
chooseAll = False
amplitudeplotvariables = {
    "filters": {
        "WaveAmplitudeInput [Volt]": 0.1, #0.1, 0.2, 0.3 
        "WaveFrequencyInput [Hz]": None, #bruk et tall  
        "WavePeriodInput": 40, #bruk et tall #brukes foreløpig kun til find_wave_range, ennå ikke knyttet til filtrering
        "WindCondition": ["no", "lowest", "full"], #full, no, lowest, all
        "TunnelCondition": None,
        "Mooring": "low",
        "PanelCondition": ["full", "reverse"], # no, full, reverse, 
        
    },
    "processing": {
        "chosenprobe": "Probe 2",
        "rangestart": None,
        "rangeend": None,
        "data_cols": ["Probe 2"],#her kan jeg velge fler, må huske [listeformat]
        "win": 11
    },
    "plotting": {
        "figsize": None,
        "separate":True,
        "overlay": False,
        "annotate": True
        
    }
    
}

# %%
from wavescripts.filters import filter_for_amplitude_plot
m_filtrert = filter_for_amplitude_plot(combined_meta_sel, amplitudeplotvariables, chooseAll)


# %%
"""Plot_all_probes plotter alt den tar inn"""
from wavescripts.plotter import plot_all_probes
plot_all_probes(m_filtrert, amplitudeplotvariables)

# %%
from wavescripts.plotter import plot_damping
# plot_damping(combined_meta_sel, amplitudeplotvariables)
# %%
from wavescripts.wavestudyer import damping
damping_df = damping(combined_meta_sel)

from wavescripts.filters import filter_for_damping
m_damping_filtrert = filter_for_damping(
    damping_df,
    amplitudeplotvariables["filters"]
)

from wavescripts.plotter import amplitude_plot
amplitude_plot(
    m_damping_filtrert,
    filters=amplitudeplotvariables["filters"],   # optional bookkeeping
    plotting=amplitudeplotvariables["plotting"]
)













