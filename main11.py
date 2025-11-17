#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
from wavescripts.data_loader import load_or_update      # import from root
from wavescripts.plotter import plot_second_column # import from root
from wavescripts.plotter import plot_column # import from root


# Step 1: Load your cached data
dfs, meta = load_or_update(Path("wavedata/20251110-tett6roof-lowM-ekte580"))

print(meta.head())
print("Loaded:", len(dfs), "dataframes")

#%%
# Step 2: Pick a dataframe (first one)
df = next(iter(dfs.values()))

rangestart = 0
rangeend = len(df.values)
# Step 3: Plot the 2nd column
plot_second_column(df, rangestart,rangeend ,title="Second Column Plot")


rangestart2 = 3000
rangeend2 = 7000
chosencolumn = 2
plot_column(df,rangestart2, rangeend2, chosencolumn, "samme")


#%%

data_cols = ["Probe 1","Probe 2","Probe 3","Probe 4"]
win = 50  # example window

df_ma = df.copy()
df_ma[data_cols] = df[data_cols].rolling(window=win, min_periods=win).mean()

rangestart3 = 3000
rangeend3 = 7000
chosencolumn = 1
plot_column(df_ma,rangestart2, rangeend2, chosencolumn, f"movingAvg window size {win}")

import pandas as pd
extremes = pd.concat([
    df.nsmallest(10, "Probe 1"),
    df.nlargest(10, "Probe 1")
]).sort_values("Probe 1").reset_index(drop=True)


#%%
import pandas as pd
import matplotlib.pyplot as plt

time = df_ma["Date"]
signal = df_ma["Probe 1"]

top10 = signal.nlargest(10)
bottom10 = signal.nsmallest(10)

plt.figure(figsize=(12,6))

# FAST LINE PLOT
plt.plot(time, signal, '-', linewidth=0.8, label="Signal")

# EXTREME MARKERS (only 10 points each)
plt.plot(time.loc[top10.index], top10.values, 'ro', markersize=6, label="Top 10")
plt.plot(time.loc[bottom10.index], bottom10.values, 'bo', markersize=6, label="Bottom 10")

plt.title("Waveform with Extremes")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

