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
#amps = df['mA']
#amps = amps[(amps >= 0.002) & (amps <= 0.02)] *1000

# files = list of your LVM files
dfs = [read_lvm(f, skip_lines_after_header=2)[['mA']] for f in files]  # select only mA column
# Optionally rename columns so each file has a unique name
for i, f in enumerate(files):
    dfs[i].columns = [f"mA_{i+1}"]
# Concatenate horizontally
df = pd.concat(dfs, axis=1)
df = df[(df >= 0.002) & (df <= 0.02)] *1000


snittet = df.mean()
print('snittet er ', snittet)



#%%
import matplotlib.pyplot as plt
df.iloc[0:2000].plot()


"""
If you’re working with large time-series data, you can downsample before plotting:
df["mA"].iloc[::100].plot()  # plot every 100th point"""

#plt.figure(figsize=(10, 6))
#plt.
plt.xlabel("Time [ms]")
plt.ylabel("Current [mA]")
plt.show()
#plt.title("All LVM Files")
plt.legend(snittet.iloc[0], snittet.iloc[1], snittet.iloc[2])
#plt.tight_layout()
plt.show()