#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 15:34:16 2025

@author: ole
"""
import pandas as pd
from pathlib import Path

def read_lvm(filename):
    """Read a LabVIEW .lvm file with European characters"""
    # Find the line where data starts
    with open(filename, 'r', encoding='latin-1') as f:
        lines = f.readlines()

    start_index = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('***End_of_Header***'):
            start_index = i
    data_start = start_index + 1

    # Read using pandas with encoding='latin-1'
    df = pd.read_csv(
        filename,
        sep='\t',
        skiprows=data_start,
        decimal=',',
        engine='python',
        usecols=[0, 1],
        names=['Time_s', 'Current_A'],
        encoding='latin-1'  # important!
    )

    df['Source_File'] = Path(filename).name
    return df


# ---- MAIN ----
# Folder containing your .lvm files
data_folder = Path("pressuredata")  # <-- change to your folder name or full path

# Find all .lvm files in that folder
files = sorted(data_folder.glob("*.lvm"))

print(f"Found {len(files)} .lvm files")

# Read all files into one big DataFrame
all_data = pd.concat([read_lvm(f) for f in files[:1]], ignore_index=True)

# Optional: convert to mA
all_data['Current_mA'] = all_data['Current_A'] * 1000

# Save to a single CSV
all_data.to_csv("all_measurements.csv", index=False)

# Plot example
import matplotlib.pyplot as plt
for name, group in all_data.groupby("Source_File"):
    plt.plot(group['Time_s'], group['Current_mA'], label=name)
plt.xlabel("Time [s]")
plt.ylabel("Current [mA]")
plt.title("All LVM Files")
plt.legend()
plt.show()
