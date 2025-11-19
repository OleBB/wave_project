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


from wavescripts.plotter import plot_filtered
from wavescripts.processor import find_resting_levels, remove_outliers, apply_moving_average, compute_simple_amplitudes

#%% - kan gjøres om senere til å passe med en JSON
# de første variablene brukes til filtrering av fil
# de 
plotvariables = {
    "amp":"0100", #0100, 0200, 0300
    "freq":"1300", #1300, 0650
    "wind":"full", #full, no, lowest
    "tunnel":None, #ingen foreløpig?
    "mooring" : "low", #low, har jeg lagt inn noen med high?
    "chosenprobe":"Probe 3", 
    "rangestart":2100,
    "rangeend":9800,
    "data_cols":["Probe 3"], #None = ["Probe 1","Probe 2","Probe 3","Probe 4"]
    "win": 10,
    "figsize":None,
    }
#%%
#For å mappe dictionary
column_map = {
    "amp": "WaveAmplitudeInput [Volt]",
    "freq": "WaveFrequencyInput [Hz]",
    "wind": "WindCondition",
    "tunnel": "TunnelCondition",
    "mooring": "Mooring",
}

plotvariables = {
    "filters": {
        "amp": "0200",
        "freq": "1300",
        "wind": "lowest",
        "tunnel": None,
        "mooring": "low"
    },

    "processing": {
        "chosenprobe": "Probe 3",
        "rangestart": 2100,
        "rangeend": 9800,
        "data_cols": ["Probe 3"],
        "win": 11
    },

    "plotting": {
        "figsize": None
    }
}
#
#import json
#with open("plotsettings.json") as f:
#   plotvariables = json.load(f)
def filter_chosen_files(plotvariables):
    df_sel = meta.copy()

    filter_values = plotvariables["filters"]

    for var_key, col_name in column_map.items():
        value = filter_values[var_key]
        if value is not None:
            df_sel = df_sel[df_sel[col_name] == value]

    return df_sel


df_sel = filter_chosen_files(plotvariables)
#nå har vi de utvalgte: df_sel eller, dataframes_selected
#så da kan vi processere dataframesene slik vi ønsker
processed_dfs = {}
for idx, row in df_sel.iterrows():
    
    path_key = row["path"]
    df_raw   = dfs[path_key]
    
    print("Columns for", row["path"], df_raw.columns.tolist())
    # proccess to apply moving average 
    df_ma = apply_moving_average(df_raw, 
                                 plotvariables["processing"]["data_cols"], 
                                 plotvariables["processing"]["win"])
    processed_dfs[path_key] = df_ma  
#%% - Med ferdig processerte dataframes, kan vi plotte dem
plot_filtered(
    processed_dfs=processed_dfs,
    df_sel=df_sel,
    **plotvariables["filters"],     # amp, freq, wind, tunnel, mooring
    **plotvariables["processing"],  # chosenprobe, rangestart, rangeend, data_cols, win
    **plotvariables["plotting"],    # figsize
)#**herekspandererPlotvariables[med nested dictionary]

#%%
#Basert på utvalgte plottevariabler så kan vi nå velge alle datasett som samsvarer
def filter_chosen_files_igjen(plotvariables):
    df_sel = meta.copy()
    mapping = {
        "amp": ("WaveAmplitudeInput [Volt]"),
        "freq": ("WaveFrequencyInput [Hz]"),
        "wind": ("WindCondition"),
        "tunnel": ("TunnelCondition"),
        "mooring": ("Mooring"),
    }
    for key, col in mapping.items():
        if plotvariables[key] is not None:
            df_sel = df_sel[df_sel[col] == plotvariables[key]]
    if df_sel.empty:
        print("No matching datasets found.")
    return df_sel

df_sel = filter_chosen_files(plotvariables)
#nå har vi de utvalgte: df_sel eller, dataframes_selected
#så da kan vi processere dataframesene slik vi ønsker
processed_dfs = {}
for idx, row in df_sel.iterrows():
    
    path_key = row["path"]
    df_raw   = dfs[path_key]
    
    print("Columns for", row["path"], df_raw.columns.tolist())
    # proccess to apply moving average
    df_ma = apply_moving_average(df_raw, plotvariables["data_cols"], plotvariables["win"])
    processed_dfs[path_key] = df_ma  
#%% - Med ferdig processerte dataframes, kan vi plotte dem
plot_filtered(
    processed_dfs = processed_dfs,
    df_sel = df_sel,
    **plotvariables   # this expands the dict into arguments!
) #"""This works only if plot_filtered() has matching parameter names:"""


#%% #Equivalent JSON file would look like:
"""
{
  "filters": {
    "amp": "0100",
    "freq": "1300",
    "wind": "lowest",
    "tunnel": null,
    "mooring": "low"
  },
  "processing": {
    "chosenprobe": "Probe 3",
    "rangestart": 2100,
    "rangeend": 9800,
    "data_cols": ["Probe 3"],
    "win": 10
  },
  "plotting": {
    "figsize": null
  }
}

#load with
import json

with open("plotsettings.json") as f:
    plotvariables = json.load(f)

"""

#%%
#average_simple_amplitude = compute_simple_amplitudes(df_ma, chosenprobe, n_amplitudes) 
#print('avg simp  amp  = ', average_simple_amplitude)


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


