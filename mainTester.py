#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 08:41:03 2025
@author: ole
"""
import os
from pathlib import Path
# ------------------------------------------------------------------
# Make the script always run from the folder where THIS file lives
# ------------------------------------------------------------------
file_dir = Path(__file__).resolve().parent
os.chdir(file_dir)
# ------------------------------------------------------------------
#%%

from wavescripts.data_loader import load_or_update
dfs, meta = load_or_update(Path("/Users/ole/Kodevik/wave_project/wavedata/20251110-tett6roof-lowMooring"))

print(meta.tail())
print("Loaded:", len(dfs), "dataframes")

from wavescripts.processor import find_resting_levels, remove_outliers, apply_moving_average, compute_simple_amplitudes
#%%
# === Config ===
plotvariables = {
    "filters": {
        "amp": 0.1, #bruk et tall 
        "freq": 1.3, #bruk et tall  
        "per": None, #bruk et tall #brukes foreløpig kun til find_wave_range, ennå ikke knyttet til filtrering
        "wind": "no", #full, no, lowest
        "tunnel": None,
        "mooring": "low"
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
        "overlay": False
        
    }
}
#
#import json
#with open("plotsettings.json") as f:_
#      plotvariables = json.load(f)

print('# === Filter ===')
from wavescripts.filters import filter_chosen_files
df_sel = filter_chosen_files(meta,plotvariables)
#nå har vi de utvalgte: df_sel altså dataframes_selected
#så da kan vi processere dataframesene slik vi ønsker

print('# === Process ===')
from wavescripts.processor import process_selected_data#, plot_ramp_debug
# - and optional check (or "debug") range (turn on/off in processor.py)
processed_dfs, df_sel, debug_data = process_selected_data(dfs, df_sel, plotvariables)
#from wavescripts.dataloader import update_metadata
#update_metadata(df_sel)
#%% -  Med ferdig processerte dataframes, kan vi plotte dem,
# === Plot selection separately and/or overlaid ===
from wavescripts.plotter import plotter_selection
plotter_selection(processed_dfs, df_sel, plotvariables)
#%%

# Step 1: Load raw data + basic metadata
from wavescripts.data_loader import load_or_update, update_processed_metadata
dfs, meta = load_or_update("wavedata/20251110-tett6roof-lowM-ekte580")
import pandas as pd
# Step 2: YOUR ANALYSIS — you modify meta and/or dfs as much as you want
for key, df in dfs.items():
    path = Path(key)
    row = meta[meta["path"] == key].iloc[0]

    # Example: compute zeroed waves and significant height
    stillwater = df["Probe 1"].iloc[:250].mean()
    eta = df["Probe 1"] - stillwater

    # Update the metadata row (in-place)
    meta.loc[meta["path"] == key, "Computed Probe 1 start"] = float(stillwater)
    meta.loc[meta["path"] == key, "Hs"] = float(4 * eta.std())
    meta.loc[meta["path"] == key, "T_z"] = float(0.71)  # or proper zero-crossing
    meta.loc[meta["path"] == key, "Processed at"] = pd.Timestamp("now")

    # Optionally save updated DataFrame back (with eta, filtered, etc.)
    df["eta_1"] = eta
    dfs[key] = df  # will be saved next time you call load_or_update or manually
update_processed_metadata(meta)

#%%
#average_simple_amplitude = compute_simple_amplitudes(df_ma, chosenprobe, n_amplitudes) 
#print('avg simp  amp  = ', average_simple_amplitude)
pæf = df_sel["path"]
df_raw = dfs[pæf]
from wavescripts.processor import find_wave_range
PROBES = ["Probe 1", "Probe 2", "Probe 3", "Probe 4"]
for probe in PROBES: #loope over alle 4 kolonnene
     #smooth the probe
     print(f'probe in loop is: {probe}')
     df_ma = apply_moving_average(df_raw, data_cols=probe, win=10)
     #find the start of the signal
     start, end, debug_info = find_wave_range(df_raw, 
                              df_sel,    
                              data_cols=probe,
                              detect_win=10, 
                              debug=True) #her skrur man på debug
 
 #heller hente en oppdatert df_sel?? #df_sel["Calculated start"] = start #pleide å være df_ma her men må jo ha engangsmetadata i metadata. 
 # === Put the calculated start_idx into




#%%
"""... KANSKJE DISPLAYE JSON ENTRYEN SÅ
#JEG VET HVA SOM BLE KJØRT. ELLER DISPLAY MOORING SOM TILLEGGSINFO I PLOTT?
# altså, nedenfor printes mappe og fil... tanken var å fange opp  
#dersom jeg kjører ulik mooring- og tunnelcondition
#sånn den er nå så kjører den bare dersom alle parameterrne stemmer"""


