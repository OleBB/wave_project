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

def filter_chosen_files(meta, plotvariables,chooseAll=True):
    """
    meta: pd.DataFrame with columns referenced in column_map
    plotvariables: dict with nested "filters" mapping short keys -> value or list-of-values
    Behavior:
      - If a filter value is None or empty string -> skip that filter (no restriction)
      - If a filter value is a list/tuple/set -> match any of those values (.isin)
      - Otherwise -> equality match
    Returns a filtered DataFrame (index preserved).
    """
    # === Førstemann til mølla! This one overrides === #
    if chooseAll:
        return meta
    # === === === #
    df_sel = meta.copy()
    filter_values = plotvariables.get("filters", {})
    """Use .get(..., default) when the key may be absent and you want a
    safe fallback (common for config-like dicts).
    Use [] when the key must exist and its absence 
    should be treated as a bug (so you want an immediate KeyError).
    """

    applied = []  # collect applied filters for debug
    
    for var_key, col_name in column_map.items():
        if var_key not in filter_values:
            continue
        value = filter_values[var_key]

        # skip "no filter" values
        if value is None or (isinstance(value, str) and value.strip() == ""):
            continue

        if isinstance(value, (list, tuple, set, pd.Series)):
            df_sel = df_sel[df_sel[col_name].isin(value)]
            applied.append((col_name, "in", value))
        else:
            df_sel = df_sel[df_sel[col_name] == value]
            applied.append((col_name, "==", value))

    number_of = len(df_sel)
    print(f'Applied filters: {applied}')
    print(f'Found {number_of} files:')
    pd.set_option("display.max_colwidth", 200)
    print(df_sel["path"])
    
    return df_sel


def filter_ampl_plot(meta_df, plotvar,chooseAll=True):
    
    
    #unpack filters:
    amp = plotvar["filters"]["amp"]
    freq = plotvar["filters"]["freq"]
    per = plotvar["filters"]["per"]
        
    
    if chooseAll:
        return meta_df
    
    df = meta_df.copy()
    
    for idx, row in df.iterrows():
        
        return
    
    return