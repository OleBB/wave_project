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
        #print('kommentar: ', lines[23:24])
        #for line in lines[:30]: print(line.strip())

        # Find header end
        for i, line in enumerate(lines):
            if line.strip().startswith('X_Value'): #X_value er navnet på første kolonne
                start_index = i + 2
                print('startindex er ', start_index)
                print('Kommentar=', lines[start_index-1])
                break

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
        names=['Time', 'mA'],
        encoding='latin-1'
    )

    #df['Source_File'] = Path(filename).name
    return df

# ---- MAIN ----
data_folder = Path("../pressuredata/fyrste_skikkelige_pitot")  # Adjust path as needed
files = sorted(data_folder.glob("*.lvm"))
print(f"Found {len(files)} .lvm files")
#%%
# Process files (set skip_lines_after_header to desired number, e.g., 2)
#df = pd.concat(
#    [read_lvm(f, skip_lines_after_header=2) for f in files[:1]],  # Change files[:1] to files for all
#    ignore_index=True
#)

amps = df['mA']
amps = amps[(amps >= 0.002) & (amps <= 0.02)] *1000
 
"""Use .where() when you want to preserve time/index continuity — useful for time series, 
so you can see where the dropouts occurred."""
# Convert to mA
#df['mA'] =  df['mA'].where(df['mA'] != 0) * 1000
#df = df[amps.values != 0]
#amps.where(amps !=0).mean(trim(0.0))
#sales.where(sales != 0).mean(trim=0.1) * 1000


"""If you want to cap the values instead of removing them:
amps = amps.clip(lower=-0.01, upper=0.01)"""
# drop rows with all zeros
#df = df.loc[(df!=0).any(axis=1)] #trenger ikke den any trur eg.


snittet = amps.mean()
print('snittet er ', snittet)

# Save to CSV
#df.to_csv("all_measurements.csv", index=False)

#%%
# Plot exmAle
import matplotlib.pyplot as plt
#df['mA'].clip(lower=5, upper=95)
amps.iloc[0:2000].plot()  # Line plot of A only
#df.iloc[0:3].plot()


"""
If you’re working with large time-series data, you can downsample before plotting:
df["mA"].iloc[::100].plot()  # plot every 100th point"""

#plt.figure(figsize=(10, 6))
#plt.
plt.xlabel("Time [ms]")
plt.ylabel("Current [mA]")
plt.show()
#plt.title("All LVM Files")
#plt.legend()
#plt.tight_layout()
#plt.show()