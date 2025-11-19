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

def find_wave_range(df, data_cols):
    #tar inn valgt dataframe
    #tar inn utvalgte kolonner
    
    #finner første skikkelige topp, ved å se etter ramp-up?
    #   eller, ved å se etter aller største...?
    #finner siste skikkelige topp. 
    #beregner avstand basert på innkommende bølgeparametere
    #velger de bølgene mellom
    
    return

def find_resting_levels():
    resting_files = [f for f in CSVFILES if 'nowind' in f.lower()]
    if not resting_files:
        raise ValueError("No valid nowind-files found to compute resting level")
    
    
    for f in resting_files:
        df99 = df99.copy()
    return


def remove_outliers():
    #lag noe basert på steepness, kanskje tilogmed ak. Hvis ak er for bratt
    # og datapunktet for høyt, så må den markeres, og så fjernes.
    return
    
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
def compute_simple_amplitudes(df_ma, chosenprobe, n):
    top_n = df_ma[chosenprobe].nlargest(n)
    bottom_n = df_ma[chosenprobe].nsmallest(n)
    average = (top_n.sum()-bottom_n.sum())/(len(top_n))
    return average #top_n, bottom_n

