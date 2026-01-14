#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 12:36:58 2026

@author: ole
"""


import numpy as np
import pandas as pd
from scipy.signal import welch


n_signals, n_samples = 400, 3000
fs = 30.0

# Example DataFrame: rows = signals, cols = time points
# df = pd.DataFrame(data, index=signal_ids, columns=time_labels)
# For example:
signal_ids = [f'sig_{i}' for i in range(n_signals)]
time_cols = [f't{t}' for t in range(n_samples)]
df = pd.DataFrame(np.random.randn(n_signals, n_samples), index=signal_ids, columns=time_cols)

data = df.values  # shape (n_signals, n_samples)
X = np.fft.rfft(data, axis=1)          # shape (n_signals, n_freqs)
freqs = np.fft.rfftfreq(n_samples, d=1.0/fs)
n_freqs = X.shape[1]

col_names = [f'{f:.3f}Hz' for f in freqs]
df_fft_complex = pd.DataFrame(X, index=df.index, columns=col_names)
# Each cell is a complex number


def compute_psd(df,m_df):
    """
    Regner ut PSD for en DF
    """
    
    for i in range (1,5):
        column  = f"eta_{i}"
        
        psd_loopvalue = welch(column)
        psd_df.append(psd_loopvalue)
    
    return psd_df


def processor_psd(processed_dfs_dict:dict[str,pd.DataFrame], m_df) -> pd.DataFrame:
    
    #vi tar inn en dict med df-er. 
    pddc = processed_dfs_dict.copy()
    
    psd_dict = {}
    for key, df in pddc.items():
        psd = compute_psd(df)
        psd_dict[key] = psd
    
    return psd_dict


# ==========================================================
# 2.a Ta utvalgte kjøringer, lag en ny dataframe, og sett null ved "stillwater"
# ==========================================================
processed_dfs = {}
for _, row in meta_sel.iterrows():
    path = row["path"]
    if path not in dfs:
        print(f"Warning: File not loaded: {path}")
        continue

    df = dfs[path].copy()

    # Zero each probe
    for i in range(1, 5):
        probe_col = f"Probe {i}"           
        if probe_col not in df.columns:
            print(f"  Missing column {probe_col} in {Path(path).name}")
            continue

        sw = stillwater[i]
        eta_col = f"eta_{i}"

        # subtract stillwater → zero mean
        df[eta_col] = -(df[probe_col] - sw) #bruker  MINUS for å snu signalet!
        #print(df[eta_col].iloc[0:10]) sjekk om den flipper
        # Optional: moving average of the zeroed signal
        df[f"{probe_col}_ma"] = df[eta_col].rolling(window=win, center=False).mean()
        
        if debug:
            print(f"  {Path(path).name:35} → eta_{i} mean = {df[eta_col].mean():.4f} mm")
    processed_dfs[path] = df