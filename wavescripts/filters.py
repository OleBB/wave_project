#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 16:18:36 2025

@author: ole
"""
import pandas as pd
#For å mappe dictionary fra input json(eller bare tilsvarende input rett i main)
#denne mappingen sørger for at dictionaryen kan sjekkes mot metadataene
#målet er å filtrere slik at jeg bare prosesserer og plotter utvalgte filer.
column_map = {
    "amp": "WaveAmplitudeInput [Volt]",
    "freq": "WaveFrequencyInput [Hz]",
    "wind": "WindCondition",
    "tunnel": "TunnelCondition",
    "mooring": "Mooring",
}

def filter_chosen_files(meta,plotvariables):
    df_sel = meta.copy()

    filter_values = plotvariables["filters"] #husk at det er nested dictionary

    for var_key, col_name in column_map.items():
        value = filter_values[var_key]
        if value is not None:
            df_sel = df_sel[df_sel[col_name] == value]
    antall = len(df_sel)
    print(f'Found {antall} files:')
    pd.set_option("display.max_colwidth", 200)
    print(df_sel["path"])#.apply(lambda p: p[-90:]))
    return df_sel