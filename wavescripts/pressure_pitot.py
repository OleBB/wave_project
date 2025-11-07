#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 09:37:54 2025

@author: ole
"""

import matplotlib.pyplot as plt
import glob
import re
import pandas as pd
import os

def load_data(folder):
    files = glob.glob(f"{folder}/*.txt")
    data = {}
    for filename in files:
        match = re.search(r"moh(\d+)", filename)
        if not match:
            continue
        key = int(match.group(1))
        with open(filename, "r", encoding="utf-8") as f:
            lines = [line.strip().strip('"').replace(',', '.') for line in f if line.strip()]
            values = [float(x) for x in lines]
        data[key] = values

    # ✅ FIX: allow unequal-length columns
    df = pd.DataFrame({k: pd.Series(v) for k, v in sorted(data.items())})
    return df

    return pd.DataFrame(dict(sorted(data.items())))

# --- load both folders ---
#path1 = "../pressuredata/20251105-lowestwindUtenProbe2-fullpanel"
#path2 = "../pressuredata/20251105-fullwindUtenProbe2-fullpanel"

#path1 = "../pressuredata/20251104-lowestwind/"
#path2 = "../pressuredata/20251104-fullwind/"

#path1 = "../pressuredata/20251105-lowestwindUtenProbe2-fullpanel"
#path2 = "../pressuredata/20251106-lowestwindUtenProbe2-fullpanel-amp0100-freq1300"

path1 = "../pressuredata/20251106-fullwindUtenProbe2-fullpanel-amp0100-freq1300"
path2 = "../pressuredata/20251106-fullwindUtenProbe2-fullpanel"

df_1 = load_data(path1)
df_2 = load_data(path2)
pathname1 = os.path.basename(path1)
pathname2 = os.path.basename(path2)

# --- compute means ---
mean_1 = df_1.mean()
mean_2 = df_2.mean()


#%%
# --- plot both on same figure ---

plt.figure(figsize=(7,5))
plt.plot(mean_1.values, mean_1.index, '-o', label=f"{pathname1}")
plt.plot(mean_2.values, mean_2.index, '-s', label=f"{pathname2}")
plt.title('Windspeed')
plt.ylabel("Height (mm)")
plt.xlabel("Average Windspeed")
plt.legend()
plt.ylim(bottom=0) 
plt.grid(True)
plt.tight_layout()
plt.show()