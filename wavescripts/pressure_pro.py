#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 15:34:16 2025
@author: ole
"""
import pandas as pd
from pathlib import Path
from io import StringIO
import numpy as np
import matplotlib.pyplot as plt

def read_lvm(filename, skip_lines_after_header=0):
    """Read a LabVIEW .lvm file with European characters, skipping lines after header"""
    # Read file once
    with open(filename, 'r', encoding='latin-1') as f:
        lines = f.readlines()
        #print('kommentar: ', lines[23:24])
        for line in lines[:30]: print(line.strip())

        # Find header end
        for i, line in enumerate(lines):
            if line.strip().startswith('X_Value'): #X_value er navnet på første kolonne
                start_index = i + 2
                #print('startindex er ', start_index)
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
data_folder = Path("../pressuredata/20251017-pitot-fullwind")  # Adjust path as needed
files = sorted(data_folder.glob("*.lvm"))
print(f"Found {len(files)} .lvm files")
#%%
dfs = [read_lvm(f, skip_lines_after_header=2)[['mA']] for f in files]  # select only mA column
# Optionally rename columns so each file has a unique name
for i, f in enumerate(files):
    dfs[i].columns = [f"mA_{i+1}"]
# Concatenate horizontally
df = pd.concat(dfs, axis=1)
df = df[(df >= 0.002) & (df <= 0.02)] *1000


"""vent nå litt.. jeg ønsker å ta snittet av df i hver rad."""
snittet = df.mean()
print('snittet er ', snittet)
#%%

#%% - Convert to pressure
#df.iloc[0:2000].plot()
sni = df.mean(axis=0)
#sni = pd.DataFrame([df.mean(axis=0)], columns=df.columns) for å 
vv = np.sqrt((sni-3.8)*18.5)
#print(vv)

#%% 
fig, ax = plt.subplots(figsize=(6, 4))  # Create figure and axes
ax.grid(True, linestyle='--', alpha=0.7)
ax.plot(vv, range(len(vv),0,-1), marker='o', linewidth=1) #kan byttes med semilogy
# Extend y-axis by 10 cm (or whatever units your axis uses)
y_min, y_max = ax.get_ylim()          # get current limits
ax.set_ylim(y_min, y_max+10)       # extend the upper limit
ax.invert_yaxis() #denne og range(len, 0,-1) for å telle nedover
ax.set_xlabel("m/s??")
ax.set_ylabel("cm fra taket")
ax.set_title("m/s??")
fig.tight_layout()
# Add fine major + minor grids for higher accuracy
ax.grid(True, which='major', linestyle='-', alpha=0.6)
ax.grid(True, which='minor', linestyle='--', alpha=0.3)
ax.minorticks_on()



# Show or save
#plt.savefig("sni_plot.pdf", bbox_inches="tight")   # vector for LaTeX
# plt.savefig("sni_plot.png", dpi=300, bbox_inches="tight")  # raster version
plt.show()


