#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 15:25:36 2025

@author: ole
"""

for path, (start, end) in auto_ranges.items():
    mask = meta["path"] == path
    meta.loc[mask, "auto_start"] = good_start_idx
    meta.loc[mask, "auto_end"]   = good_end_idx
    """OOOPS MÅ VÆRE TILPASSET PROBE OGSÅ"""
    

meta_volt = row["WaveAmplitudeInput [Volt]"]
meta_freq = row["WaveFrequencyInput [Hz]"]
meta_per = row["WavePeriodInput"]

df_sel["Calculated end"] = end #