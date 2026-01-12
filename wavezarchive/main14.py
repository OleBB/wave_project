#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 14:48:52 2025

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

from wavescripts.data_loader import load_or_update
dfs, meta = load_or_update(Path("/Users/ole/Kodevik/wave_project/wavedata/20251110-tett6roof-lowMooring"))

print(meta.tail())
print("Loaded:", len(dfs), "dataframes")

from wavescripts.processor import remove_outliers, compute_simple_amplitudes
#%%
# === Config ===
chooseAll = False
plotvariables = {
    "filters": {
        "amp": 0.1, #0.1, 0.2, 0.3 
        "freq": 1.3, #bruk et tall  
        "per": None, #bruk et tall #brukes foreløpig kun til find_wave_range, ennå ikke knyttet til filtrering
        "wind": "lowest", #full, no, lowest
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
# ==== ELLER IMPORTER EGEN CONFIG ==== #
#import json
#with open("plotsettings.json") as f:_
#      plotvariables = json.load(f)

#%%
print('# === Filter ===')
from wavescripts.filters import filter_chosen_files
meta_sel = filter_chosen_files(meta,
                             plotvariables,
                             chooseAll=False)
#nå har vi de utvalgte: df_sel altså dataframes_selected
#så da kan vi processere dataframesene slik vi ønsker

print('# === Process ===')
from wavescripts.processor import process_selected_data#, plot_ramp_debug
# - and optional check: DEBUG gir noen ekstra printa linjer
processed_dfs, meta_sel = process_selected_data(dfs, 
                                                meta_sel, 
                                                meta, 
                                                debug=True, 
                                                win=10, 
                                                find_range=True,
                                                range_plot=True)

#%%
from wavescripts.wavestudyer import compare_probe_amplitudes_and_lag, amplitude_overview, full_tank_diagnostics, wind_damping_analysis

summary_df = wind_damping_analysis(processed_dfs, meta_sel, window_ms=(6000, 14000))

summary = full_tank_diagnostics(processed_dfs, window_ms=(8000, 8100))


overview = amplitude_overview(processed_dfs, window_ms=(5000, 15000))
# Pick any file
df = list(processed_dfs.values())[0]
print('name:' ,df.head())

print("Raw amplitudes (before any fix):")
for i in range(1,5):
    col = f"eta_{i}"
    if col in df.columns:
        amp = (df[col].quantile(0.99) - df[col].quantile(0.01)) / 2
        print(f"  Probe {i}: {amp:.1f} mm  →  {'PROBABLY BAD' if amp > 50 else 'OK'}")

result = compare_probe_amplitudes_and_lag(df, start_ms=6000, end_ms=7000)
res = compare_probe_amplitudes_and_lag(df, start_ms=5000, end_ms=15000)

window = df.loc[5000:15000]
t = (window.index - window.index[0]) / 1000  # seconds

import matplotlib.pyplot as plt
plt.figure(figsize=(10,4))
plt.plot(t, window["eta_2"], label="Probe 2", alpha=0.8)
plt.plot(t, window["eta_3"] - res["lag_ms"], label=f"Probe 3 (shifted -{res['lag_ms']:.0f}ms)", alpha=0.8)
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("Elevation [mm]")
plt.title("Perfect alignment after time-shift correction")
plt.grid(alpha=0.3)
plt.show()
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



