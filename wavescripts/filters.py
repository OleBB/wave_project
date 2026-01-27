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
    "amp":   "WaveAmplitudeInput [Volt]",
    "freq":  "WaveFrequencyInput [Hz]",
    "per":   "WavePeriodInput",               # 
    "wind":  "WindCondition",                 # 
    "tunnel":"TunnelCondition",               # 
    "mooring":"Mooring",
    "panel": "PanelCondition" 
}

def filter_chosen_files(meta, plotvariables,chooseAll,chooseFirst):
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
        print("Alle valgt, fordi chooseAll=True")
        return meta
    elif chooseFirst:
        return meta.iloc[0:1]
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


def filter_for_amplitude_plot(meta_df:pd.DataFrame, amplotvars: dict, chooseAll: bool = False) -> pd.DataFrame:
    if chooseAll:
        return meta_df
    
    df = meta_df.copy()

    filters = amplotvars.get("filters", {})
    
    mask = pd.Series(True, index=df.index)
        
    for key, value in filters.items():
        if value is None:
            continue
        col_name = column_map.get(key, key)
        
        if isinstance(value, (list, tuple, set, np.ndarray)):
            mask &= df[col_name].isin(value)

        elif callable(value):
            # The callable must return a boolean Series of the same length
            mask &= value(df[col_name])

        else:   
            if key == "wind":
                # Allowed values: "full", "no", "lowest", "all"
                # "all" = no windfilter on this column.
                if value == "all":
                    continue          # skip – keep current mask unchanged
                mask &= df[col_name] == value

            elif key == "mooring":
                if value == "all":
                    continue
                mask &= df[col_name] == value
            elif key == "panel":
                if value == "all":
                    continue
                mask &= df[col_name] == value
            #TK - denne logikken klarer ikke listen ["all"] 
            else:
                mask &= df[col_name] == value
            
            
    return df[mask].copy()


from typing import Mapping, Any, Sequence, Callable, Union

def filter_for_damping(
    df: pd.DataFrame,
    criteria: Mapping[str, Any],
) -> pd.DataFrame:
    """
    Return a *view* of ``df`` that respects the key/value pairs in ``criteria``.
    The function is deliberately simple – it only implements the operations you
    need for the amplitude‑plot workflow.  Anything more complex can be added
    later as a ``custom_filter`` callback.

    Parameters
    ----------
    df : pd.DataFrame
        The original (unfiltered) data.
    criteria : dict
        Keys are column names, values are either:
            • a single scalar → keep rows where column == scalar
            • a list/tuple   → keep rows where column is in that list
            • a tuple (low, high) → keep rows where low ≤ column ≤ high
            • ``None``        → *ignore* this column (no filtering)
    Eg fjerna: custom_filter : callable, optional
        If you need a bespoke filter that cannot be expressed with the simple
        rules above, pass a function that receives the intermediate DataFrame
        and returns a new one.

    Returns
    -------
    pd.DataFrame
        A filtered copy (``df.copy()``) so the original data stays untouched.
    """
    out = df.copy()

    for col, val in criteria.items():
        if val is None:
            continue                # nothing to do for this column
        if isinstance(val, (list, tuple, set)):
            # treat a 2‑tuple specially: low‑high range
            if len(val) == 2 and not isinstance(val, list):
                low, high = val
                out = out[(out[col] >= low) & (out[col] <= high)]
            else:
                out = out[out[col].isin(val)]
        else:
            # scalar equality
            out = out[out[col] == val]

    return out



def filter_for_frequencyspectrum(
    df: pd.DataFrame,
    criteria: Mapping[str, Any],
) -> pd.DataFrame:
    """
    Return a *view* of ``df`` that respects the key/value pairs in ``criteria``.
    
    Parameters
    ----------
    df : pd.DataFrame
        The original (unfiltered) data.
    criteria : dict
        Can be either:
        - A simple dict with column filters
        - A nested dict with "overordnet" and "filters" keys
        
        If nested:
        - "overordnet": {"chooseAll": bool} - if True, skip all filtering
        - "filters": dict of column filters
        
    Returns
    -------
    pd.DataFrame
        A filtered copy (``df.copy()``) so the original data stays untouched.
    """
    
    # Check if criteria has nested structure with "filters"
    if "filters" in criteria:
        # Check for override in "overordnet"
        if "overordnet" in criteria:
            overordnet = criteria["overordnet"]
            if overordnet.get("chooseAll", False):
                # Override: return all data without filtering
                return df.copy()
            elif overordnet.get("chooseFirst", False):
                return df.iloc[0]
        
        # Use the "filters" sub-dictionary
        actual_criteria = criteria["filters"]
    else:
        # Direct criteria dictionary
        actual_criteria = criteria
    
    out = df.copy()
    for col, val in actual_criteria.items():
        if val is None:
            continue
        if isinstance(val, (list, tuple, set)):
            # treat a 2‑tuple specially: low‑high range
            if len(val) == 2 and not isinstance(val, list):
                low, high = val
                out = out[(out[col] >= low) & (out[col] <= high)]
            else:
                out = out[out[col].isin(val)]
        else:
            # scalar equality
            out = out[out[col] == val]
    return out
















