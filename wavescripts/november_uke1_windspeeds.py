#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 20:51:51 2025

@author: gpt
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt

# --- USER SETTINGS ---
# Root directory containing your folders
root_dir = r"/Users/ole/Kodevik/wave_project/pressuredata/"

# List of folders you want to plot (only these will be processed)
selected_folders = [
    "20251106-lowestwindUtenProbe2-fullpanel-amp0100-freq1300",
    #"20251106-fullwindUtenProbe2-fullpanel-amp0100-freq1300",
    #"20251106-fullwindUtenProbe2-fullpanel",
    "20251105-lowestwindUtenProbe2-fullpanel",
    "20251105-lowestwind-fullpanel",
    "20251105-lowestwind",
    #"20251105-fullwindUtenProbe2-fullpanel",
    "20251104-lowestwind",
    #"20251104-fullwind"
    
]

# Pattern to extract height (e.g., moh000, moh010, ...)
height_pattern = re.compile(r"moh(\d+)", re.IGNORECASE)


def read_mean_from_file(file_path):
    """Reads a text file and returns the mean of all numeric values."""
    values = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().replace('"', '').replace(',', '.')
            if line:
                try:
                    values.append(float(line))
                except ValueError:
                    pass
    return np.mean(values) if values else np.nan


def process_folder(folder_path):
    """Processes one folder, returning sorted (heights, mean_values)."""
    heights = []
    means = []

    for fname in os.listdir(folder_path):
        if fname.lower().endswith(".txt"):
            match = height_pattern.search(fname)
            if match:
                height = int(match.group(1))
                file_path = os.path.join(folder_path, fname)
                mean_val = read_mean_from_file(file_path)
                heights.append(height)
                means.append(mean_val)

    # Sort by height
    heights, means = zip(*sorted(zip(heights, means)))
    return np.array(heights), np.array(means)

#%%
# --- MAIN PLOTTING ---
fig, ax = plt.subplots(figsize=(8, 6))  
markers = ['o', 's', '^', 'D', '*', 'x', '+', 'P', '|', '<', '>']
marker_index = 0

for folder in selected_folders:
    folder_path = os.path.join(root_dir, folder)
    if not os.path.isdir(folder_path):
        print(f"Skipping (not found): {folder_path}")
        continue

    heights, means = process_folder(folder_path)
    
    # Use a marker from the list and update index
    marker = markers[marker_index % len(markers)]  # Cycle through markers
    ax.plot(means, heights, marker=marker, label=folder)
    
    # Incrementing the marker index for the next folder
    marker_index += 1

# Set labels and titles
ax.set_ylabel("Height [mm]")
ax.set_xlabel("Mean value")
ax.set_title("Windspeeds")
ax.set_ylim(0, 380)
ax.yaxis.set_major_locator(plt.MultipleLocator(20))
ax.yaxis.set_minor_locator(plt.MultipleLocator(5))
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.legend()

plt.tight_layout()
#plt.savefig("../windresults/nov04-06-lowestwind.pdf", bbox_inches="tight")
plt.show()