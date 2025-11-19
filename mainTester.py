#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 08:41:03 2025

@author: ole
"""
from pathlib import Path
from wavescripts.data_loader import load_or_update
dfs, meta = load_or_update(Path("wavezarchive/testingfolder"))

print(meta.tail())
print("Loaded:", len(dfs), "dataframes")

#%%

from wavescripts.plotter import plot_filtered

#lage input til plot_filtered()

meta
dfs


amp="0100"
freq="1300"
wind="full"
chosenprobe="Probe 3"
rangestart=3100
rangeend=5800
data_cols=["Probe 3"] #None = ["Probe 1","Probe 2","Probe 3","Probe 4"]
win = 1
figsize=None



#Plot_filtered må ta inn ferdig utvalgte prober, med spesifisert amp og freq
plot_filtered(
    meta,
    dfs,
    amp, 
    freq,
    wind,
    chosenprobe,
    rangestart=3100,
    rangeend=5800,
    data_cols=["Probe 3"], #None = ["Probe 1","Probe 2","Probe 3","Probe 4"]
    win = 1,
    figsize=None

)

#average_simple_amplitude = compute_simple_amplitudes(df_ma, chosenprobe, n_amplitudes) 
#print('avg simp  amp  = ', average_simple_amplitude)







#%%
firstkey = meta["path"].iloc[0] #take first path value
mydf = dfs[firstkey]

#dfcopy = mydf.copy()
df99 = mydf["Probe 1"].iloc[0:99].mean(skipna=True)
df250 = mydf["Probe 1"].iloc[0:250].mean(skipna=True)
df1000 = mydf["Probe 1"].iloc[0:1000].mean(skipna=True)

import matplotlib.pyplot as plt
import os
from wavescripts.processor import compute_simple_amplitudes

top, bott = compute_simple_amplitudes(mydf, "Probe 1", 10)
total_top = top.sum()
total_bott = bott.sum()
avg = (total_top-total_bott)/(len(top))

vindu = 4000
fra = 1000
til = fra+vindu

fra2 = 1000
til2 = fra2+vindu

fra3 = 1000
til3  = fra3+vindu

#x1 = mydf["Probe 1"].iloc[fra:til]
#x2 = mydf["Probe 2"].iloc[fra2:til2]
#x3 = mydf["Probe 3"].iloc[fra3:til3]
#x4 = mydf["Probe 4"].iloc[fra3:til3]

#plt.title(firstkey[58:])

mydf[["Probe 1", "Probe 2", "Probe 3", "Probe 4"]].iloc[fra:til].plot()
plt.legend()

#%%
x1 = mydf["Probe 1"].iloc[fra:til]
x1.name = "Probe 1"
x2 = mydf["Probe 2"].iloc[fra2:til2] 
x2.name = "Probe 2"
x3 = mydf["Probe 3"].iloc[fra3:til3] 
x3.name = "Probe 3"
x4 = mydf["Probe 4"].iloc[fra3:til3] 
x4.name = "Probe 4"

plt.plot(x1)
plt.plot(x2)
plt.plot(x3)
plt.plot(x4)

plt.legend()


