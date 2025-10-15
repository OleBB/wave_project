#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 15:34:16 2025
@author: ole
"""
import pandas as pd
from pathlib import Path
from io import StringIO

def read_lvm(filename, skip_lines_after_header=0):
    """Read a LabVIEW .lvm file with European characters, skipping lines after header"""
    # Read file once
    with open(filename, 'r', encoding='latin-1') as f:
        lines = f.readlines()
        print('first line: ', lines[0])
        for line in lines[:10]: print(line.strip())

        # Find header end
        for i, line in enumerate(lines):
            if line.strip().startswith('***End_of_Header***'):
                start_index = i + 1
                break
        else:
            start_index = 0  # Fallback if no header found

    # Skip additional lines after header
    data_start = start_index + skip_lines_after_header
    if data_start >= len(lines):
        raise ValueError(f"Cannot skip {skip_lines_after_header} lines after header; file too short.")

    # Convert remaining lines to string for pandas
    data_lines = '\n'.join(lines[data_start:])
    df = pd.read_csv(
        StringIO(data_lines),
        sep='\t',
        decimal=',',
        usecols=[0, 1],
        names=['Time_s', 'Current_A'],
        encoding='latin-1'
    )

    df['Source_File'] = Path(filename).name
    return df

# ---- MAIN ----
data_folder = Path("../pressuredata")  # Adjust path as needed
files = sorted(data_folder.glob("*.lvm"))
print(f"Found {len(files)} .lvm files")

# Process files (set skip_lines_after_header to desired number, e.g., 2)
all_data = pd.concat(
    [read_lvm(f, skip_lines_after_header=2) for f in files[:1]],  # Change files[:1] to files for all
    ignore_index=True
)

# Convert to mA
all_data['Current_mA'] = all_data['Current_A'] * 1000

# Save to CSV
all_data.to_csv("all_measurements.csv", index=False)

# Plot example
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
for name, group in all_data.groupby("Source_File"):
    plt.plot(group['Time_s'], group['Current_mA'], label=name, alpha=0.7)
plt.xlabel("Time [s]")
plt.ylabel("Current [mA]")
plt.title("All LVM Files")
plt.legend()
plt.tight_layout()
plt.show()