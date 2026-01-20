#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 08:41:03 2025
@author: ole
"""

#TODO: 
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
#dfs, meta = load_or_update(Path("/Users/ole/Kodevik/wave_project/wavedata/20251110-tett6roof-lowMooring"))
dfs, meta = load_or_update(Path("/Users/ole/Kodevik/wave_project/wavedata/20251110-tett6roof-lowM-ekte580")) #per15
# (Path("/Users/ole/Kodevik/wave_project/wavedata/20251110-tett6roof-lowMooring-2")) #per15

#dfs, meta = load_or_update(Path("/Users/ole/Kodevik/wave_project/wavedata/20251112-tett6roof"))
#dfs, meta = load_or_update(Path("/Users/ole/Kodevik/wave_project/wavedata/20251113-tett6roof-loosepaneltaped"))
#%%
# === Config ===
chooseAll = False
chooseFirst = False
# range debug and plot
debug=True
win=10
find_range = True
range_plot = False

processvariables = {
    "filters": {
        "amp": 0.1, #0.1, 0.2, 0.3 
        "freq": 1.3, #bruk et tall  
        "per": None, #bruk et tall #brukes foreløpig kun til find_wave_range, ennå ikke knyttet til filtrering
        "wind": "full", #full, no, lowest
        "tunnel": None,
        "mooring": "low",
        "panel": ["full", "reverse"], # no, full, reverse, 
    },
    "processing": {
        "chosenprobe": "Probe 3", #ikkje i bruk
        "rangestart": None, #ikkje i bruk
        "rangeend": None, #ikkje i bruk
        "data_cols": ["Probe 2"],#ikkje i bruk
        "win": 11 #ikkje i bruk
    },
    "plotting": {
        "figsize": None,
        "separate":True,
        "overlay": False   
    }
}
# alternativt importere plotvariabler
#import json
#with open("plotsettings.json") as f:_
#      plotvariables = json.load(f)
print('# === Filter === #')
from wavescripts.filters import filter_chosen_files
meta_sel = filter_chosen_files(meta,
                             processvariables,
                             chooseAll,chooseFirst)
#nå har vi de utvalgte: meta_sel altså metadataframes_selected
#%%
print('# === Process === #')
from wavescripts.processor import process_selected_data
# - and optional check: DEBUG gir noen ekstra printa linjer
processed_dfs, meta_sel, psd_dictionary, fft_dictionary = process_selected_data(dfs, 
                                                meta_sel, 
                                                meta, 
                                                debug, 
                                                win, 
                                                find_range,
                                                range_plot)
#TODO fiks slik at find_wave_range starter ved null eller ved en topp?
# nå tar den first_motion_idx+ gitt antall bølger.

# %%

#dagens mål: implementere likningen fra John. 
from datetime import datetime

#fpdf - first
#ts_df timeseries_df

fpdf = next(iter(processed_dfs.values()))
ts_df = fpdf[["Date", "eta_1"]]
print(ts_df)
# %%

dt = (fpdf["Date"].iloc[1] - fpdf["Date"].iloc[0]).total_seconds()
print(dt)

#regne coeffisienter
# ck =
# første coeffisient
# %%
N = 10

for m  in range (1,N)
# c0
c1 = 




# %%





import matplotlib.ticker as mticker

first_df = next(iter(psd_dictionary.values()))
# python
ax = first_df[["Pxx 1", "Pxx 2", "Pxx 3", "Pxx 4"]].plot()
ax.set_xlim(0, 6)
ax.set_ylim(1e-6, 40)
ax.minorticks_on()
ax.xaxis.set_major_locator(mticker.MultipleLocator(0.5))   # major every 0.5 (adjust)

ax.grid(True, which="major")
#ax.grid(True, which="minor", linestyle="--")

# python
# ax.set_xscale("log")
# ax.set_yscale("log")   # or "symlog" if values span zero

# %%
#
# python
import pandas as pd
import matplotlib.pyplot as plt

# df_plot has columns from your psd_dictionary (as in your example)
first_cols = {k: d.iloc[:, 0] for k, d in psd_dictionary.items()}
df_plot = pd.concat(first_cols, axis=1)

fig, ax = plt.subplots(figsize=(7, 4))

# Iterate columns for full control
for name in df_plot.columns:
    ax.plot(df_plot.index, df_plot[name], label=str(name), linewidth=1.5, marker=None)

ax.set_xlabel("freq (Hz)")
# ax.set_ylabel("PSD")
ax.set_xlim(0, 10)
ax.grid(True, which="both", ls="--", alpha=0.3)
#ax.legend(title="Series", ncol=2)  # or remove if not needed
plt.tight_layout()
plt.show()










# %% - OBS OBS koden over lagrer greier. nå vil jeg jo sammenlikne flere datasett uten å lagre.

np.correlate()



#%% - 
from wavescripts.wavestudyer import compare_probe_amplitudes_and_lag, amplitude_overview, full_tank_diagnostics, wind_damping_analysis 
summary_df = wind_damping_analysis(meta_sel)

# %% 
"""skriver et sammendrag av dempningen i """
from wavescripts.wavestudyer import wind_damping_analysis
damping_analysis_results = wind_damping_analysis(meta_sel)


#%%
from wavescripts.wavestudyer import damping
oppgradert_meta_sel = damping(meta_sel)




#%% - Her plotter man en enkeltkjøring oppå en annen


chooseAll = True

amplitudeplotvariables = {
    "filters": {
        "amp": 0.1, #0.1, 0.2, 0.3 
        "freq": 1.3, #bruk et tall  
        "per": None, #bruk et tall #brukes foreløpig kun til find_wave_range, ennå ikke knyttet til filtrering
        "wind": ["no", "lowest", "full"], #full, no, lowest, all
        "tunnel": None,
        "mooring": "low",
        "panel": ["full", "no"], # no, full, reverse, 
        
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

from wavescripts.filters import filter_for_amplitude_plot
m_filtrert = filter_for_amplitude_plot(oppgradert_meta_sel, amplitudeplotvariables, chooseAll)

"""Plot amplitude summary plotter alt den tar inn"""
from wavescripts.plotter import plot_all_probes
plot_all_probes(m_filtrert, amplitudeplotvariables)




#%%æ
summary = full_tank_diagnostics(processed_dfs)
#TODO få resultater fra disse over i run_and_save_report.py
#%%
overview = amplitude_overview(processed_dfs, window_ms)
#to måter å få ut første verdien fra dictionary
#next(iter(processed_dfs.values())) #uten .values får man key, eller så kan man skrive .keys, og .item gir tuple med key og value.
# Pick any file
df = list(processed_dfs.values())[0]
print('name:' ,df.head())

print("Raw amplitudes (before any fix):")
for i in range(1,5):
    col = f"eta_{i}"
    if col in df.columns:
        amp = (df[col].quantile(0.99) - df[col].quantile(0.01)) / 2
        print(f"  Probe {i}: {amp:.1f} mm  →  {'PROBABLY BAD' if amp > 50 else 'OK'}")
#%%
col1 = "eta_2"
col2 = "eta_3"
start_ms = 6000
end_ms = 7000
result = compare_probe_amplitudes_and_lag(df,col1, col2, start_ms, end_ms)

res = compare_probe_amplitudes_and_lag(df, col1, col2, start_ms, end_ms)

window = df.loc[start_ms:end_ms]
t = (window.index - window.index[0]) #/ 1000  #seconds

import matplotlib.pyplot as plt
plt.figure(figsize=(10,4))
plt.plot(t, window["eta_2"], label="Probe 2", alpha=0.8)
plt.plot(t - res["lag_ms"], window["eta_3"], label=f"Probe 3 (shifted -{res['lag_ms']:.0f}ms)", alpha=0.8)
plt.legend()
plt.xlabel("milliseconds")
plt.ylabel("Elevation [mm]")
plt.title("Perfect alignment after time-shift correction")
plt.grid(alpha=0.3)
plt.show()

#%% - Create a "report"  - but currently its really just a table with df data. 
from wavescripts.create_report import markdown_report_from_df
markdown_report_from_df(df,
                            title="Wave Tank Analysis Report",
                            subtitle="a preliminary draft",
                            out_path="report.md",
                            plots_folder="reportplots",
                            save_plots=True,
                            max_rows=50)

#%%

# Step 1: Load raw data + basic metadata
from wavescripts.data_loader import load_or_, _processed_metadata
dfs, meta = load_or_("wavedata/20251110-tett6roof-lowM-ekte580")
import pandas as pd
# Step 2: YOUR ANALYSIS — you modify meta and/or dfs as much as you want
for key, df in dfs.items():
    path = Path(key)
    row = meta[meta["path"] == key].iloc[0]

    # Example: compute zeroed waves and significant height
    stillwater = df["Probe 1"].iloc[:250].mean()
    eta = df["Probe 1"] - stillwater

    #  the metadata row (in-place)
    meta.loc[meta["path"] == key, "Computed Probe 1 start"] = float(stillwater)
    meta.loc[meta["path"] == key, "Hs"] = float(4 * eta.std())
    meta.loc[meta["path"] == key, "T_z"] = float(0.71)  # or proper zero-crossing
    meta.loc[meta["path"] == key, "Processed at"] = pd.Timestamp("now")

    # Optionally save d DataFrame back (with eta, filtered, etc.)
    df["eta_1"] = eta
    dfs[key] = df  # will be saved next time you call load_or_ or manually
_processed_metadata(meta)


#%%

# processing.py — clean and professional
from wavescripts.data_loader import load_or_, _processed_metadata
from wavescripts.processor import process_selected_data


from wavescripts.data_loader import load_meta_from_processed
meta = load_meta_from_processed("PROCESSED-20251110-tett6roof-lowM-ekte580")


#%%
"""... KANSKJE DISPLAYE JSON ENTRYEN SÅ
#JEG VET HVA SOM BLE KJØRT. ELLER DISPLAY MOORING SOM TILLEGGSINFO I PLOTT?
# altså, nedenfor printes mappe og fil... tanken var å fange opp  
#dersom jeg kjører ulik mooring- og tunnelcondition
#sånn den er nå så kjører den bare dersom alle parameterrne stemmer"""


