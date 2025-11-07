#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 11:44:59 2025

@author: ole
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
    "20251107-lowestwindUP2-allpanel-angleTest"
]

# Pattern to extract height (e.g., moh000, moh010, ...)
height_pattern = re.compile(r"ang(\d+)", re.IGNORECASE)


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


# --- MAIN PLOTTING ---

markers = ['o', 's', '^', 'D', '*', 'x', '+', 'P', '|', '<', '>']
marker_index = 0
plt.figure(figsize=(8, 6))

for folder in selected_folders:
    folder_path = os.path.join(root_dir, folder)
    if not os.path.isdir(folder_path):
        print(f"Skipping (not found): {folder_path}")
        continue

    heights, means = process_folder(folder_path)
    marker = markers[marker_index % len(markers)]  # Cycle through markers
    plt.plot(means, heights, marker=marker, label=folder)    
    marker_index += 1

plt.ylabel("Height [mm]")
plt.xlabel("Mean value")

plt.title("Mean Value vs Height per Folder")
plt.legend(title="Folder")
plt.grid(True)
plt.tight_layout()
plt.show()
