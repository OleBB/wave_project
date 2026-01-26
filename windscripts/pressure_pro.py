#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 15:34:16 2025
@author: ole


dette skriptet plotter to vindprofiler i ett plott

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
        #for line in lines[:30]: print(line.strip())

        # Find header end
        for i, line in enumerate(lines):
            if line.strip().startswith('X_Value'): #X_value er navnet på første kolonne
                start_index = i-1
                #print('startindex er ', start_index)
                print('Kommentar=', lines[start_index-1])
                break

    # Skip additional lines after header
    data_start = start_index + skip_lines_after_header
    if data_start >= len(lines):
        raise ValueError(f"Cannot skip {skip_lines_after_header} lines after header; file too short.")

    # Convert remaining lines to string for pandas
    data_lines = '\n'.join(lines[data_start:])
    dataframeFromPandasLibrary_package = pd.read_csv(
        StringIO(data_lines),
        sep='\t',
        decimal=',',
        usecols=[0, 1],
        names=['Time', 'mA'],
        encoding='latin-1'
    )

    #df['Source_File'] = Path(filename).name
    return dataframeFromPandasLibrary_package

#%%
# ---- MAIN ----
data_folder = Path("../pressuredata/20251024-pitot-mean-fullwind")  # Adjust path as needed
files = sorted(data_folder.glob("*.lvm"))
print(f"Found {len(files)} .lvm files")
datafoldername1 = data_folder.name

files = files #her kan man iterere over et utvalg av alle filene files eller files[:1] eller files[3:5]
dataRamme = [read_lvm(f, skip_lines_after_header=2)[['mA']] for f in files]  # select only mA column
# Optionally rename columns so each file has a unique name
for i, f in enumerate(files):
    dataRamme[i].columns = [f"mA_{i+1}"]
# Concatenate horizontally
df = pd.concat(dataRamme, axis=1)
#df = df[(df >= 0.002) & (df <= 0.02)] *1000 #denne var vel for å fjerne en outlier

# ---- Main2 ----
data_folder2 = Path("../pressuredata/20251025-pitot-mean-lowestwind")  # Adjust path as needed
files2 = sorted(data_folder2.glob("*.lvm"))
print(f"Found {len(files)} .lvm files")
datafoldername2 = data_folder2.name

files2 = files2 #her kan man iterere over et utvalg av alle filene files eller files2[:1] eller files2[3:5]
dataRamme2 = [read_lvm(f, skip_lines_after_header=2)[['mA']] for f in files2]  # select only mA column
# Optionally rename columns so each file has a unique name
for i, f in enumerate(files2):
    dataRamme2[i].columns = [f"mA_{i+1}"]
#alternativ navnendinrg: sni2.columns = [f"Windspeed_{i+1}" for i in range(len(sni2.columns))]

# Concatenate horizontally
df2= pd.concat(dataRamme2, axis=1)

#%%
"""Når der er flere rader, og man ønsker å ta snittet av df i hver rad."""
#snittet = df.mean()
#print('snittet er ', snittet)


#%% - Convert to pressure
#sni = df.mean(axis=0)
sni = df.iloc[0]
sni.index = [name.replace('mA_', 'høyde_') for name in sni.index]
windspeed1 = np.sqrt((sni-3.8)*18.51) #til de målingene fra 24og25okt var 3.8 base-verdi.

#%% ---- Convert to pressure2 ----
#sni = df.mean(axis=0)
sni2 = df2.iloc[0]
sni2.index = [name.replace('mA_', 'høyde_') for name in sni2.index]
windspeed2 = np.sqrt((sni2-3.8)*18.51) #2*(150Pa/16mArange) Bruk 18.51 når 3,8 mA 
#print(windspeed2) #(sni2-4)*18.75 er for nullstilt ved 4.0mA

#%% - string manipulation. Beholder alt før første dash og alt etter siste dash
import re
datafoldername1 = re.sub(r'-.*-', '-', datafoldername1)
datafoldername2 = re.sub(r'-.*-', '-', datafoldername2)

#%% 
ylengde = len(windspeed1)
yhøgd = range(0,ylengde)
ylengde2 = len(windspeed2)
yhøgd2 = range(0,ylengde2)

fig, ax = plt.subplots(figsize=(6, 4))  # Create figure and axes
ax.grid(True, linestyle='--', alpha=0.7)
ax.semilogy(-windspeed1, yhøgd, marker='o', linewidth=1,label=datafoldername1) #kan byttes med semilogy
ax.semilogy(-windspeed2, yhøgd2, marker='x', linewidth=1,label=datafoldername2) #kan byttes med semilogy

# Extend y-axis by 10 cm (or whatever units your axis uses)
y_min, y_max = ax.get_ylim()          # get current limits
ax.set_ylim(y_min, y_max+10)       # extend the upper limit
ax.invert_yaxis() #denne og range(len, 0,-1) for å telle nedover
ax.set_xlabel("m/s")
ax.set_ylabel("cm fra taket")
ax.set_title("Vindhastighet")
fig.tight_layout()
# Add fine major + minor grids for higher accuracy
ax.grid(True, which='major', linestyle='-', alpha=0.6)
ax.grid(True, which='minor', linestyle='--', alpha=0.3)
import matplotlib.ticker as ticker
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))  # major ticks every 0.5 m/s
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))  # minor ticks every 0.1 m/s
#ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
#ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))  # major ticks every 0.5 m/s
#ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))  # minor ticks every 0.1 m/s
from matplotlib.ticker import LogFormatterExponent
#ax.yaxis.set_major_formatter(LogFormatterExponent(base=10.0))

ax.minorticks_on()

ax.legend()

# Show or save
#plt.savefig("sni_plot.pdf", bbox_inches="tight")   # vector for LaTeX
# plt.savefig("sni_plot.png", dpi=300, bbox_inches="tight")  # raster version
plt.show()
#%%

import numpy as np
import matplotlib.pyplot as plt

# Example data
x = np.linspace(-6, -2, 10)
y = 10**(x+2)  # arbitrary function just to have something that fits
print(x)
print()
print(y)

fig, ax = plt.subplots()

# Plot with semilog-y
ax.semilogy(x, y, label="Example curve")

# Axis limits as requested
#ax.set_xlim(-6, -2)
#ax.set_ylim(1e-3, 1e0)

# ✅ invert the y-axis so that 10^0 is at the top and 10^-3 at the bottom
#ax.invert_yaxis()

# Labels and title
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis (log scale)")
ax.set_title("Semilog Y-axis with inverted orientation")

# Grid and legend
ax.grid(True, which='both', linestyle='--', alpha=0.6)
ax.legend()
plt.tight_layout()
plt.show()



