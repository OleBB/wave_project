#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 09:42:29 2026

@author: ole
"""
from pathlib import Path
import pandas as pd
import numpy as np
from wavescripts.data_loader import update_processed_metadata
from scipy.signal import find_peaks
from scipy.signal import welch
from scipy.optimize import brentq
from typing import Dict, List, Tuple

from wavescripts.constants import PHYSICS, WAVENUMBER, MEASUREMENT
#wave_physics.py


def calculate_wavenumbers(frequencies, heights):
    """Tar inn frekvens og høyde
    bruker BRENTQ fra scipy
    """
    freq = np.asarray(frequencies)
    H = np.broadcast_to(np.asarray(heights), freq.shape)
    k = np.zeros_like(freq, dtype=float)
    
    valid = freq>0
    i_valid = np.flatnonzero(valid)
    if i_valid.size ==0:
        return k
    g = PHYSICS.GRAVITY
    w_dwbf = WAVENUMBER.DEEP_WATER_BRACKET_FACTOR
    w_swbf = WAVENUMBER.SHALLOW_WATER_BRACKET_FACTOR
    for idx in i_valid:
        fr = freq.flat[idx]
        h = H.flat[idx] * MEASUREMENT.MM_TO_M #konverter fra millimeter
        omega = 2 * np.pi * fr
        
        def disp(k_wave):
            return g*k_wave* np.tanh(k_wave * h) - omega**2
        
        k_deep = omega**2 / g
        k_shallow = omega / np.sqrt(g * h) if h >0 else k_deep
        
        a = min(k_deep, k_shallow) *w_dwbf
        b = max(k_deep, k_shallow) *w_swbf
        
        fa = disp(a)
        fb = disp(b)
        while fa * fb > 0:
            a /= 2
            b *= 2
            fa = disp(a)
            fb = disp(b)
        k.flat[idx] = brentq(disp, a, b)
    return k

def calculate_celerity(wavenumbers,heights):
    k = np.asarray(wavenumbers)
    H = np.broadcast_to(np.asarray(heights), k.shape)
    c = np.zeros_like(k,dtype=float)
    
    g = PHYSICS.GRAVITY
    sigma = PHYSICS.WATER_SURFACE_TENSION #ved 20celcius
    rho = PHYSICS.WATER_DENSITY #10^3 kg/m^3
    
    c = np.sqrt( g/ k * np.tanh(k*H))
    return c


def calculate_windspeed(windcond: pd.Series) -> pd.Series:
    # normalize labels to avoid casing/whitespace issues
    wc = windcond.astype(str).str.lower().str.strip()
    speed_map = {
        "no": 0.0,
        "lowest": 3.8,
        "low": 3.8,   # if you also use "low"
        "full": 5.8,
    }
    return wc.map(speed_map)


def calculate_wavedimensions(k: pd.Series, 
                             H: pd.Series, 
                             PC: pd.Series,
                             P2A: pd.Series,
                             ) -> pd.DataFrame:
    panel_length_map = {
            "purple": 1.048,
            "yellow": 1.572,
            "full": 2.62,
            "reverse": 2.62,
        }
    
    # Align (inner) on indices to keep only rows present in both k and H
    k_aligned, H_aligned = k.align(H, join="inner")
    idx = k_aligned.index
    P2A_aligned = None if P2A is None else P2A.reindex(idx)
    PC_aligned = None if PC is None else PC.reindex(idx)

    k_arr = k_aligned.to_numpy(dtype=float)
    Hm = H_aligned.to_numpy(dtype=float)* MEASUREMENT.MM_TO_M  #fra millimeter
     
    valid_k = k_arr > 0.0 # Mask for valid k values
    g = PHYSICS.GRAVITY
    
    kH = np.full_like(k_arr, np.nan, dtype=float)
    kH[valid_k] = k_arr[valid_k] * Hm[valid_k]
     
    tanhkh = np.full_like(k_arr, np.nan, dtype=float)
    tanhkh[valid_k] = np.tanh(kH[valid_k])
    
    wavelength = np.full_like(k_arr, np.nan, dtype=float)
    wavelength[valid_k] = 2.0 * np.pi / k_arr[valid_k]
    
    L_arr = np.full_like(k_arr, np.nan, dtype=float)
    if PC_aligned is not None:
        pc_norm = PC_aligned.astype(str).str.strip().str.lower()
        L_char = pc_norm.map(panel_length_map)
        L_arr = L_char.to_numpy(dtype=float)
    else:
        print('PanelCondition missing - no kL to calculate')
    kL = np.full_like(k_arr, np.nan, dtype=float)
    mask_kL = valid_k & np.isfinite(L_arr)
    kL[mask_kL] = k_arr[mask_kL] * L_arr[mask_kL]
    
    ak = np.full_like(k_arr, np.nan, dtype=float)
    if P2A_aligned is not None:
        a_arr = P2A_aligned.to_numpy(dtype=float) * MEASUREMENT.MM_TO_M   #fra millimeter
        ak[valid_k] = a_arr[valid_k] * k_arr[valid_k]
    else:
        print('No probe 2 amplitude - no ak to calculate')
    
    c = np.full_like(k_arr, np.nan, dtype=float)
    c[valid_k] = np.sqrt((g / k_arr[valid_k]) * tanhkh[valid_k])
    
    out = pd.DataFrame(
        {"Wavelength": wavelength, 
         "kL": kL, 
         "ak": ak, 
         "kH": kH, 
         "tanh(kH)": tanhkh, 
         "Celerity": c,
             }, 
            index=idx
        )
    return out
