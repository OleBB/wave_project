#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
from wavescripts.data_loader import load_or_update      # import from root
from wavescripts.plotter import plot_column # import from root


# Step 1: Load your cached data
dfs, meta = load_or_update(Path("wavedata/20251110-tett6roof-lowM-ekte580"))

print(meta.head())
print("Loaded:", len(dfs), "dataframes")


#%%



#%%

import os
import matplotlib.pyplot as plt

first_key = meta["path"].loc[0] #take first path value
df2 = dfs[first_key]

data_cols = ["Probe 1","Probe 2","Probe 3","Probe 4"]
win = 50  # example window

df_ma = df2.copy()
df_ma[data_cols] = df2[data_cols].rolling(window=win, min_periods=win).mean()



keys = list(dfs.keys())
df_first = dfs[keys[0]]
df_third = dfs[keys[1]]
df_last = dfs[keys[2]]
# Step 2: Smoothing
win = 50
data_cols = ["Probe 1","Probe 2","Probe 3","Probe 4"]

df_1 = df_first.copy()
df_1[data_cols] = df_first[data_cols].rolling(win, min_periods=win).mean()

df_2 = df_third.copy()
df_2[data_cols] = df_third[data_cols].rolling(win, min_periods=win).mean()

df_3 = df_last.copy()
df_3[data_cols]  = df_last[data_cols].rolling(win, min_periods=win).mean()

rangestart = 3100
rangeend   = 7200
chosencolumn = 3
""" ---  --- --- ---"""
fig, ax = plt.subplots(figsize=(10,5))

plot_column(df_1, rangestart, rangeend, chosencolumn,title=os.path.basename(keys[0]), ax=ax)

plot_column(df_2, rangestart, rangeend, chosencolumn,title=os.path.basename(keys[2]), ax=ax)

plot_column(df_3, rangestart, rangeend, chosencolumn, title=os.path.basename(keys[-1]), ax=ax)

ax.legend([
    os.path.basename(keys[0]),
    os.path.basename(keys[2]),
    os.path.basename(keys[-1])
])

plt.show()



#%%
amp_val = "0100"
freq_val = "1300"

rangestart = 0
rangeend   = 10000

data_cols = ["Probe 1","Probe 2","Probe 3","Probe 4"]
win = 1  # moving average window

# --- Color map for wind condition ---
wind_colors = {
    "full": "red",
    "no": "blue",
    "lowest": "green"
}

# ---- FILTER ----
selected = meta[
    (meta["WaveAmplitudeInput [Volt]"] == amp_val) &
    (meta["WaveFrequencyInput [Hz]"] == freq_val)
]

selected_keys = selected["path"].tolist()
#print("Selected keys:", selected_keys)

# ---- PLOT ----
import matplotlib.pyplot as plt
import os

chosenprobe = "Probe 3"

plt.figure(figsize=(10, 6))

for idx, row in selected.iterrows():

    path_key = row["path"]
    wind = row["WindCondition"]        # full / nowind / lowest
    panel = row["PanelCondition"]      # nopanel / fullpanel / etc.
    
    df_raw = dfs[path_key]

    # ---- MOVING AVERAGE ----
    df_ma = df_raw.copy()
    df_ma[data_cols] = df_raw[data_cols].rolling(
        window=win,
        min_periods=win
    ).mean()

    # ---- LABEL ----
    #label = os.path.basename(path_key)
    filename = os.path.basename(path_key)
    
    # Extract short components
    panel = row["PanelCondition"]
    wind  = row["WindCondition"]
    amp   = row["WaveAmplitudeInput [Volt]"]
    freq  = row["WaveFrequencyInput [Hz]"]
    
    # Build short label
    label = f"{panel}panel-{wind}wind-a{amp}-f{freq}"

    # ---- PICK COLOR BASED ON WIND ----
    color = wind_colors.get(wind, "black")   # fallback: black

    # ---- LINE STYLE BASED ON PANEL ----
    linestyle = "--" if panel == "full" else "-"

    # ---- PLOT ----
    plt.plot(
        df_ma[chosenprobe].loc[rangestart:rangeend],
        label=label,
        color=color,
        linestyle=linestyle
    )

plt.title(f" {chosenprobe} (amp={amp_val}, freq={freq_val}) — Moving Avg. window={win}")
plt.xlabel("milliseconds")
plt.ylabel("Probe 2")
plt.legend()
plt.grid(True)
plt.show()

#%%  

#-----------------#

amp_val = "0200"
freq_val = "1300"

rangestart = 4800
rangeend   = 5800

data_cols = ["Probe 1","Probe 2","Probe 3","Probe 4"]
win = 1   # moving average window

# --- Color map for wind condition ---
wind_colors = {
    "full": "red",
    "no": "blue",
    "lowest": "green"
}

# ---- FILTER ----
selected = meta[
    (meta["WaveAmplitudeInput [Volt]"] == amp_val) &
    (meta["WaveFrequencyInput [Hz]"] == freq_val)
]

selected_keys = selected["path"].tolist()

print("Selected keys:", selected_keys)

# ---- PLOT ----
import matplotlib.pyplot as plt
import os

plt.figure(figsize=(10, 6))

for idx, row in selected.iterrows():

    path_key = row["path"]
    wind = row["WindCondition"]        # full / nowind / lowest
    panel = row["PanelCondition"]      # nopanel / fullpanel / etc.
    
    df_raw = dfs[path_key]

    # ---- MOVING AVERAGE ----
    df_ma = df_raw.copy()
    df_ma[data_cols] = df_raw[data_cols].rolling(
        window=win,
        min_periods=win
    ).mean()

    # ---- LABEL ----
    #label = os.path.basename(path_key)
    filename = os.path.basename(path_key)
    
    # Extract short components
    panel = row["PanelCondition"]
    wind  = row["WindCondition"]
    amp   = row["WaveAmplitudeInput [Volt]"]
    freq  = row["WaveFrequencyInput [Hz]"]
    
    # Build short label
    label = f"{panel}panel-{wind}wind-a{amp}-f{freq}"

    # ---- PICK COLOR BASED ON WIND ----
    color = wind_colors.get(wind, "black")   # fallback: black

    # ---- LINE STYLE BASED ON PANEL ----
    linestyle = "--" if panel == "full" else "-"

    # ---- PLOT ----
    plt.plot(
        df_ma["Probe 2"].loc[rangestart:rangeend],
        label=label,
        color=color,
        linestyle=linestyle
    )

plt.title(f"Probe 2 (amp={amp_val}, freq={freq_val}) — Moving Average {win}")
plt.xlabel("milliseconds")
plt.ylabel("Probe 2")
plt.legend()
plt.grid(True)
plt.show()

#%%















#%%

data_cols = ["Probe 1","Probe 2","Probe 3","Probe 4"]
win = 50  # example window

df_ma = df2.copy()
df_ma[data_cols] = df2[data_cols].rolling(window=win, min_periods=win).mean()

rangestart3 = 3000
rangeend3 = 7000
chosencolumn = 1
plot_column(df_ma,rangestart3, rangeend3, chosencolumn, f"movingAvg window size {win}")

import pandas as pd
extremes = pd.concat([
    df_ma.nsmallest(10, "Probe 1"),
    df_ma.nlargest(10, "Probe 1")
]).sort_values("Probe 1").reset_index(drop=True)

#%%

amp_val = "0100"
freq_val = "1300"

rangestart = 3100
rangeend   = 7200

# --- FILTER ---
selected = meta[
    (meta["WaveAmplitudeInput [Volt]"] == amp_val) &
    (meta["WaveFrequencyInput [Hz]"] == freq_val)
]

selected_keys = selected["path"].tolist()

print("Selected:", selected_keys)

# --- PLOT ---
import matplotlib.pyplot as plt
import os

plt.figure(figsize=(10, 6))

for key in selected_keys:
    df_plot = dfs[key]
    label = os.path.basename(key)
    
    # plot only Probe 2 in the chosen range
    plt.plot(
        df_plot["Probe 2"].loc[rangestart:rangeend],
        label=label
    )

plt.title(f"Probe 2 (amp={amp_val}, freq={freq_val})")
plt.xlabel("Index")
plt.ylabel("Probe 2")
plt.legend()
plt.show()

#%%
import pandas as pd
import matplotlib.pyplot as plt

time = df_ma["Date"]
signal = df_ma["Probe 1"]

top10 = signal.nlargest(10)
bottom10 = signal.nsmallest(10)

#plt.figure(figsize=(12,6))

# FAST LINE PLOT
#plt.plot(time, signal, '-', linewidth=0.8, label="Signal")
# EXTREME MARKERS (only 10 points each)
#plt.plot(time.loc[top10.index], top10.values, 'ro', markersize=6, label="Top 10")
#plt.plot(time.loc[bottom10.index], bottom10.values, 'bo', markersize=6, label="Bottom 10")

#plt.title("Waveform with Extremes")
#plt.xlabel("Time")
#plt.ylabel("Value")
#plt.legend()
#plt.grid(True)

#plt.tight_layout()
#plt.show()

