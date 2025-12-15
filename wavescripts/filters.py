#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 16:18:36 2025

@author: ole
"""
import pandas as pd
import numpy as np
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





def filter_for_amplitude_plot(meta_df :pd.DataFrame, amplotvars: dict, chooseAll: bool = False) -> pd.DataFrame:
    if chooseAll:
        return meta_df
    
    df = meta_df.copy()

    filters = amplotvars.get("filters", {})
    
    mask = pd.Series(True, index=df.index)
    
    col_map = {
       "amp":   "WaveAmplitudeInput [Volt]",
       "freq":  "WaveFrequencyInput [Hz]",
       "per":   "WavePeriodInput",               # example – adjust if needed
       "wind":  "WindCondition",                 # you must have such a column
       "tunnel":"TunnelCondition",               # placeholder name
       "mooring":"Mooring",
       "panel": "PanelCondition" 
   }
    
    for key, value in filters.items():
        if value is None:
            continue
        col_name = col_map.get(key, key)
        
        if isinstance(value, (list, tuple, set, np.ndarray)):
            mask &= df[col_name].isin(value)

        elif callable(value):
            # The callable must return a boolean Series of the same length
            mask &= value(df[col_name])

        else:   # scalar → exact match
            # Special handling for the “wind” and “mooring” strings you mentioned
            if key == "wind":
                # Example: you store the wind condition as a string column.
                # Allowed values: "full", "no", "lowest", "all"
                # If the user passes "all" we *do not filter* on this column.
                if value == "all":
                    continue          # skip – keep current mask unchanged
                mask &= df[col_name] == value

            elif key == "mooring":
                # Same idea as wind – you can add more complex logic here.
                if value == "all":
                    continue
                mask &= df[col_name] == value

            else:
                mask &= df[col_name] == value
            
            
    return df[mask].copy()


















