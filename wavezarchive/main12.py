#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 13:55:25 2025


@author: ole
"""


from pathlib import Path
from wavescripts.data_loader import load_or_update
from wavescripts.plotter import plot_column 
from wavescripts.plotter import plot_filtered


# Step 1: Load your cached data
dfs, meta = load_or_update(Path("wavedata/20251110-tett6roof-lowM-ekte580"))

print(meta.head())
print("Loaded:", len(dfs), "dataframes")

#%%
plot_filtered(
    meta,
    dfs,
    amp="0200", 
    freq="1300",
    wind="full",
    chosenprobe="Probe 1",
    rangestart=3100,
    rangeend=5800,
    win = 100,
    figsize=None

)
#%%









