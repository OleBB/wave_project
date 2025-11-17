#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 13:55:25 2025

@author: ole
"""


from pathlib import Path
from wavescripts.data_loader import load_or_update      # import from root
from wavescripts.plotter import plot_column # import from root
from wavescripts.plotter import plot_filtered


# Step 1: Load your cached data
dfs, meta = load_or_update(Path("wavedata/20251110-tett6roof-lowM-ekte580"))

print(meta.head())
print("Loaded:", len(dfs), "dataframes")

plot_filtered(
    meta,
    dfs,
    amp="0100",
    freq="1300",
    wind="full",
    chosenprobe="Probe 2",
    rangestart=3100,
    rangeend=7200,
    win = 1,

)
