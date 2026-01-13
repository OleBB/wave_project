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


def compute_psd(df):
    """
    Regner ut PSD for en DF
    """
    
    for i in range (1,5):
        column  = f"eta {i}"
        
        psd_loopvalue = welch(column)
        psd_df.append(psd_loopvalue)
    
    return psd_df


def processor_psd(processed_dfs_dict:dict[str,pd.DataFrame]) -> pd.DataFrame:
    
    #vi tar inn en dict med df-er. 
    pddc = processed_dfs_dict.copy()
    
    psd_dict = {}
    for key, df in pddc.items():
        psd = compute_psd(df)
        psd_dict[key] = psd
    
    return psd_dict