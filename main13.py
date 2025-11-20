#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 16:12:15 2025

@author: ole
"""
from pathlib import Path
from wavescripts.data_loader import load_or_update
dfs, meta = load_or_update(Path("wavezarchive/testingfolder"))

print(meta.tail())
print("Loaded:", len(dfs), "dataframes")

#%%
# === Config ===

plotvariables = {
    "filters": {
        "amp": "0100",
        "freq": "1300",
        "wind": "full",
        "tunnel": None,
        "mooring": "low"
    },

    "processing": {
        "chosenprobe": "Probe 3",
        "rangestart": 2100,
        "rangeend": 9800,
        "data_cols": ["Probe 3"],#her kan jeg velge fler, uten effekt
        "win": 50
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
#%% - Med ferdig processerte dataframes, kan vi plotte dem
from wavescripts.plotter import plot_filtered
plot_filtered(
    processed_dfs=processed_dfs,
    df_sel=df_sel,
    **plotvariables["filters"],     # amp, freq, wind, tunnel, mooring
    **plotvariables["processing"],  # chosenprobe, rangestart, rangeend, data_cols, win
    **plotvariables["plotting"],    # figsize
)#**herekspandererPlotvariables[med nested dictionary]

