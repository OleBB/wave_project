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


firstkey = meta["path"].iloc[0] #take first path value
mydf = dfs[firstkey]

#dfcopy = mydf.copy()
df99 = mydf["Probe 1"].iloc[0:99].mean(skipna=True)
df250 = mydf["Probe 1"].iloc[0:250].mean(skipna=True)
df1000 = mydf["Probe 1"].iloc[0:1000].mean(skipna=True)

import matplotlib.pyplot as plt
import os
pr = 3000
x1 = mydf["Probe 1"].iloc[0:pr]
x2 = mydf["Probe 2"].iloc[0:pr]
x3 = mydf["Probe 3"].iloc[0:pr]
x4 = mydf["Probe 4"].iloc[0:pr]

plt.title(firstkey[58:])
plt.legend()
plt.plot(x1)
plt.plot(x2)
plt.plot(x3)
plt.plot(x4)






