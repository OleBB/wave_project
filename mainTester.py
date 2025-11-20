#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 08:41:03 2025
@author: ole
"""
from pathlib import Path
from wavescripts.data_loader import load_or_update
dfs, meta = load_or_update(Path("wavezarchive/testingfolder"))

print(meta.tail())
print("Loaded:", len(dfs), "dataframes")

from wavescripts.processor import find_resting_levels, remove_outliers, apply_moving_average, compute_simple_amplitudes

#%%
# === Config ===
plotvariables = {
    "filters": {
        "amp": "0100",
        "freq": "1300",
        "wind": "no", #full, no, lowest
        "tunnel": None,
        "mooring": "low"
    },
    "processing": {
        "chosenprobe": "Probe 1",
        "rangestart": None,
        "rangeend": None,
        "data_cols": ["Probe 1"],#her kan jeg velge fler, må huske [listeformat]
        "win": 11
    },
    "plotting": {
        "figsize": None
    }
}
#
#import json
#with open("plotsettings.json") as f:
#      plotvariables = json.load(f)

# === Filter ===
from wavescripts.filters import filter_chosen_files
df_sel = filter_chosen_files(meta,plotvariables)
#nå har vi de utvalgte: df_sel altså dataframes_selected
#så da kan vi processere dataframesene slik vi ønsker

# === Process ===
from wavescripts.processor import process_selected_data
processed_dfs = process_selected_data(dfs, df_sel, plotvariables)

# === Plot ===
from wavescripts.plotter import plot_filtered
#%%
processed_dfs, auto_ranges = process_selected_data(dfs, df_sel, plotvariables)

manual_start = plotvariables["processing"]["rangestart"]
manual_end   = plotvariables["processing"]["rangeend"]

for path, df_ma in processed_dfs.items():

    auto_start, auto_end = auto_ranges[path]

    # Use manual if provided, otherwise automatic
    final_start = manual_start if manual_start is not None else auto_start
    final_end   = manual_end   if manual_end   is not None else auto_end

    runtime_vars = {
        **plotvariables["filters"],
        **plotvariables["processing"],
        **plotvariables["plotting"],
        "rangestart": final_start,
        "rangeend": final_end,
    }

    plot_filtered(
        processed_dfs={path: df_ma},
        df_sel=df_sel[df_sel["path"] == path],
        **runtime_vars
    )

#%% - Med ferdig processerte dataframes, kan vi plotte dem
from wavescripts.plotter import plot_filtered
plot_filtered(
    processed_dfs=processed_dfs,
    df_sel=df_sel,
    **plotvariables["filters"],     # amp, freq, wind, tunnel, mooring
    **plotvariables["processing"],  # chosenprobe, rangestart, rangeend, data_cols, win
    **plotvariables["plotting"],    # figsize
)#**herEkspandererPlotvariables[med nested dictionary]

#%% - TESTE RAMPUP

# 1. Smooth signal using SAME moving average function you already use
df_smoothed = apply_moving_average(df, [data_col], win)
signal = df_smoothed[data_col].values

# 2. Estimate still-water noise from early part of signal
baseline_std = np.std(signal[:200])
threshold = baseline_std * 3.0

# 3. Detect ramp-up point: where smoothed signal exceeds threshold
start_idx = np.argmax(np.abs(signal) > threshold)


#%%
average_simple_amplitude = compute_simple_amplitudes(df_ma, chosenprobe, n_amplitudes) 
print('avg simp  amp  = ', average_simple_amplitude)


#%%
"""... KANSKJE DISPLAYE JSON ENTRYEN SÅ
#JEG VET HVA SOM BLE KJØRT. ELLER DISPLAY MOORING SOM TILLEGGSINFO I PLOTT?
# altså, nedenfor printes mappe og fil... tanken var å fange opp  
#dersom jeg kjører ulik mooring- og tunnelcondition
#sånn den er nå så kjører den bare dersom alle parameterrne stemmer"""


#%%
firstkey = meta["path"].iloc[0] #take first path value
mydf = dfs[firstkey]

#dfcopy = mydf.copy()
df99 = mydf["Probe 1"].iloc[0:99].mean(skipna=True)
df250 = mydf["Probe 1"].iloc[0:250].mean(skipna=True)
df1000 = mydf["Probe 1"].iloc[0:1000].mean(skipna=True)

import matplotlib.pyplot as plt
import os
from wavescripts.processor import compute_simple_amplitudes

top, bott = compute_simple_amplitudes(mydf, "Probe 1", 10)
total_top = top.sum()
total_bott = bott.sum()
avg = (total_top-total_bott)/(len(top))

vindu = 4000
fra = 1000
til = fra+vindu

fra2 = 1000
til2 = fra2+vindu

fra3 = 1000
til3  = fra3+vindu

#x1 = mydf["Probe 1"].iloc[fra:til]
#x2 = mydf["Probe 2"].iloc[fra2:til2]
#x3 = mydf["Probe 3"].iloc[fra3:til3]
#x4 = mydf["Probe 4"].iloc[fra3:til3]

#plt.title(firstkey[58:])

mydf[["Probe 1", "Probe 2", "Probe 3", "Probe 4"]].iloc[fra:til].plot()
plt.legend()

#%%
x1 = mydf["Probe 1"].iloc[fra:til]
x1.name = "Probe 1"
x2 = mydf["Probe 2"].iloc[fra2:til2] 
x2.name = "Probe 2"
x3 = mydf["Probe 3"].iloc[fra3:til3] 
x3.name = "Probe 3"
x4 = mydf["Probe 4"].iloc[fra3:til3] 
x4.name = "Probe 4"

plt.plot(x1)
plt.plot(x2)
plt.plot(x3)
plt.plot(x4)

plt.legend()


