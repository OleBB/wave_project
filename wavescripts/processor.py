#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 17:18:03 2025

@author: ole
"""

from pathlib import Path
from typing import Iterator, Dict, Tuple
import json
import re
import pandas as pd
import os
from datetime import datetime
#from wavescripts.data_loader import load_or_update #blir vel motsatt.. 

CSVFILES = []

def find_resting_levels():
    resting_files = [f for f in CSVFILES if 'nowind' in f.lower()]
    if not rest_files:
        raise ValueError("No valid nowind-files found to compute resting level")
    
    
    for f in resting_files:
        df99 = df99.copy()


def remove_outliers():
    
    
# ------------------------------------------------------------
# Moving average helper
# ------------------------------------------------------------
def apply_moving_average(df, data_cols, win=1):
    df_ma = df.copy()
    df_ma[data_cols] = df[data_cols].rolling(window=win, min_periods=win).mean()
    return df_ma


# ------------------------------------------------------------
# Ny funksjon
# ------------------------------------------------------------
def compute_simple_amplitudes(df_ma, data_cols, n):
    top_n = df_ma[data_cols].nlargest(n)
    bottom_n = df_ma[data_cols].nsmallest(n)
    return topn




def comp_amplitudes(peaks, troughs, signal, time_axis, max_amplitudes=10):
    amplitudes = []
    time_pairs = []
    
    for i in range(min(len(peaks), len(troughs), max_amplitudes)):
        if i + 1 < len(troughs) and peaks[i] < troughs[i + 1]:
            if peaks[i] < len(signal) and troughs[i] < len(signal):
                amp = abs(signal[peaks[i]] - signal[troughs[i]]) / 2
                amplitudes.append(amp)
                time_pairs.append((time_axis[peaks[i]], time_axis[troughs[i]]))
        elif peaks[i] < troughs[i]:
            if peaks[i] < len(signal) and troughs[i] < len(signal):
                amp = abs(signal[peaks[i]] - signal[troughs[i]]) / 2
                amplitudes.append(amp)
                time_pairs.append((time_axis[peaks[i]], time_axis[troughs[i]]))
    return amplitudes[:max_amplitudes], time_pairs[:max_amplitudes]
