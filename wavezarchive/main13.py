#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 16:12:15 2025

@author: ole
"""
from pathlib import Path
from wavescripts.data_loader import load_or_update
dfs, meta = load_or_update(Path(""))

print(meta.tail())
print("Loaded:", len(dfs), "dataframes")

from wavescripts.processor import find_resting_levels, remove_outliers, apply_moving_average, compute_simple_amplitudes
#%%
# === Config ===
plotvariables = {
    "filters": {
        "amp": "0100", #bruker variabelen input_volt
        "freq": "1300", #bruker variabelen input_freq
        "per": "15", #brukes foreløpig kun til find_wave_range, ennå ikke knyttet til filtrering
        "wind": "no", #full, no, lowest
        "tunnel": None,
        "mooring": "low"
    },
    "processing": {
        "chosenprobe": "Probe 2",
        "rangestart": 0,
        "rangeend": 10000,
        "data_cols": ["Probe 2"],#her kan jeg velge fler, må huske [listeformat]
        "win": 11
    },
    "plotting": {
        "figsize": None,
        "separate":True,
        "overlay": False
        
    }
}
#
#import json
#with open("plotsettings.json") as f:
#      plotvariables = json.load(f)

print('# === Filter ===')
from wavescripts.filters import filter_chosen_files
df_sel = filter_chosen_files(meta,plotvariables)
#nå har vi de utvalgte: df_sel altså dataframes_selected
#så da kan vi processere dataframesene slik vi ønsker

print('# === Process ===')
from wavescripts.processor import process_selected_data#, plot_ramp_debug
# - and optional check (debug) range 
processed_dfs, auto_ranges, debug_data = process_selected_data(dfs, df_sel, plotvariables)
#%% -  Med ferdig processerte dataframes, kan vi plotte dem,
# === Plot selection separately and/or overlaid ===
from wavescripts.plotter import plotter_selection
plotter_selection(processed_dfs, df_sel, auto_ranges, plotvariables)

