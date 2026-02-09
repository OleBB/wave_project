#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 16:18:36 2025

@author: ole
"""
import pandas as pd
import numpy as np
from typing import Mapping, Any, Sequence, Callable, Union
from wavescripts.constants import (
    ProbeColumns as PC,
    GlobalColumns as GC,
    ColumnGroups as CG,
)

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

def filter_chosen_files(meta, processvariables):
    """
    meta: pd.DataFrame with columns referenced in column_map
    plotvariables: dict with nested "filters" mapping short keys -> value or list-of-values
    Behavior:
      - If a filter value is None or empty string -> skip that filter (no restriction)
      - If a filter value is a list/tuple/set -> match any of those values (.isin)
      - Otherwise -> equality match
    Returns a filtered DataFrame (index preserved).
    """
    # 0. unpack
    overordnet = processvariables.get("overordnet", {})
    chooseAll = overordnet.get("chooseAll", False)
    chooseFirst = overordnet.get("chooseFirst", False)

    # === Førstemann til mølla! This one overrides === #
    if chooseAll:
        print("Alle valgt, fordi chooseAll=True")
        return meta
    elif chooseFirst:
        return meta.iloc[0:1]
    # === === === #
    df_sel = meta.copy()
    filter_values = processvariables.get("filters", {})
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
# %%


def filter_for_amplitude_plot(meta_df: pd.DataFrame, amplotvars: dict) -> pd.DataFrame:
    overordnet = amplotvars.get("overordnet", {})
    chooseAll = overordnet.get("chooseAll", False)
    chooseFirst = overordnet.get("chooseFirst", False)

    df = meta_df.copy()
    n_original = len(df)
    
    if chooseAll:
        print("No filtering — chooseAll = True")
        return df.copy()
    
    if chooseFirst:
        print("Selected only first row — chooseFirst = True")
        return df.iloc[[0]].copy()

    filters = amplotvars.get("filters", {})
    mask = pd.Series(True, index=df.index)
    print("Starting with full dataset:", n_original, "rows")
    
    for key, value in filters.items():
        if value is None:
            continue
            
        col_name = column_map.get(key, key)
        if col_name not in df.columns:
            print(f"  ✗ Column '{col_name}' not found → skipping {key}")
            continue

        before = mask.sum()
        
        if isinstance(value, (list, tuple, set, np.ndarray)):
            if "all" in [v.lower() if isinstance(v,str) else v for v in value]:
                print(f"  ✓ {key}: 'all' in list → no filter applied")
                continue
            mask &= df[col_name].isin(value)
            applied = f"isin({value})"
        elif callable(value):
            mask &= value(df[col_name])
            applied = "custom function"
        else:
            if value == "all":
                print(f"  ✓ {key}: 'all' → no filter")
                continue
            mask &= df[col_name] == value
            applied = f"== {value!r}"
        
        after = mask.sum()
        removed = before - after
        print(f"  ✓ {key}: {applied}  → kept {after} rows (removed {removed})")

    filtered_df = df[mask].copy()
    
    print(f"Final result: {len(filtered_df)} rows (removed {n_original - len(filtered_df)})")
    return filtered_df


# %%



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
                return df.iloc[[0]].copy()
        
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
        
    if "overordnet" in criteria:
        overordnet = criteria["overordnet"]
        if overordnet.get("chooseFirstUnique", False):
            #find those cols to compare for uniqueness
            # use only those filtered on
            cols_to_compare = [
                col for col in actual_criteria.keys()
                if col in out.columns and actual_criteria[col] is not None
                ]
            if cols_to_compare:
                out = out.drop_duplicates(subset=cols_to_compare, keep="first")            
    return out

# %% gpts forsøk på å forbedre filterne

import re
import pandas as pd
from typing import Any, Mapping, Iterable, Callable, Union

CriteriaVal = Union[
    Any,                    # scalar equality
    Iterable[Any],          # membership (isin)
    dict                    # structured ops: {'between': (...)} | {'regex': '...'} | {'callable': fn} | {'not': {...}}
]

def _is_iterable_but_not_str(x) -> bool:
    return isinstance(x, (list, tuple, set)) and not isinstance(x, (str, bytes))

def _normalize_criteria(criteria: Mapping[str, Any], *, allow_nested: bool = True):
    """
    Return (mode, filters) where:
      - mode in {'all', 'first', 'filter'}
      - filters is a flat dict of column -> criterion
    """
    if allow_nested and "filters" in criteria:
        over = criteria.get("overordnet", {}) or {}
        if over.get("chooseAll", False):
            return "all", {}
        if over.get("chooseFirst", False):
            return "first", {}
        return "filter", criteria["filters"]
    return "filter", criteria

def _apply_one(series: pd.Series, spec: CriteriaVal) -> pd.Series:
    """
    Turn a single column criterion into a boolean mask aligned to series.
    Supported:
      - scalar            -> equality
      - iterable          -> membership (isin)
      - dict ops:
          {'between': (low, high)}        inclusive bounds
          {'regex': pattern}              str contains regex (NaN -> False)
          {'callable': fn}                fn(series) -> boolean Series
          {'op': 'between'|'regex'|'callable', 'value': ...}  # alternative form
          {'not': <any of the above>}     negation
    """
    # Negation wrapper
    if isinstance(spec, dict) and "not" in spec:
        inner = spec["not"]
        mask = _apply_one(series, inner)
        return ~mask

    # Structured ops
    if isinstance(spec, dict):
        # Unified op/value form
        op = spec.get("op")
        if "between" in spec or op == "between":
            bounds = spec.get("between", spec.get("value"))
            if not (_is_iterable_but_not_str(bounds) and len(bounds) == 2):
                raise ValueError(f"between expects (low, high); got {bounds}")
            low, high = bounds
            return series.between(low, high)
        if "regex" in spec or op == "regex":
            pattern = spec.get("regex", spec.get("value"))
            return series.astype(str).str.contains(pattern, regex=True, na=False)
        if "callable" in spec or op == "callable":
            fn = spec.get("callable", spec.get("value"))
            mask = fn(series)
            if not isinstance(mask, pd.Series) or mask.dtype != bool or mask.shape != series.shape:
                raise ValueError("callable must return a boolean Series aligned with the column")
            return mask
        raise ValueError(f"Unsupported filter dict for column '{series.name}': {spec}")

    # Membership (explicit iterable)
    if _is_iterable_but_not_str(spec):
        return series.isin(list(spec))

    # Scalar equality
    return series == spec

def filter_dataframe(
    df: pd.DataFrame,
    criteria: Mapping[str, Any],
    *,
    ignore_missing_columns: bool = False,
    allow_nested: bool = True,
    return_first_if_requested: bool = True,
) -> pd.DataFrame:
    """
    Unified DataFrame filter.

    Parameters
    - df: DataFrame to filter.
    - criteria:
        Flat form: {'Col': value_or_iterable_or_struct, ...}
        Nested form: {'overordnet': {'chooseAll': bool, 'chooseFirst': bool}, 'filters': {...}}
    - ignore_missing_columns: skip criteria for columns not in df (instead of raising).
    - allow_nested: enable nested form parsing.
    - return_first_if_requested: if chooseFirst=True, return df.iloc[[0]].

    Returns
    - A filtered copy (df.loc[mask].copy()).
    """
    mode, filters = _normalize_criteria(criteria, allow_nested=allow_nested)

    if mode == "all":
        return df.copy()
    if mode == "first" and return_first_if_requested:
        return df.iloc[[0]].copy()

    if not isinstance(filters, Mapping):
        raise ValueError("filters must be a dict mapping column -> criterion")

    mask = pd.Series(True, index=df.index)
    for col, spec in filters.items():
        if spec is None:
            continue
        if col not in df.columns:
            if ignore_missing_columns:
                continue
            raise KeyError(f"Column '{col}' not in DataFrame")
        col_mask = _apply_one(df[col], spec)
        if not (isinstance(col_mask, pd.Series) and col_mask.dtype == bool):
            raise ValueError(f"Filter for column '{col}' must yield a boolean Series")
        mask &= col_mask

    return df.loc[mask].copy()



# %% GROUPERS

# def damping_grouper(combined_meta_df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Aggregates P3/P2 (and optional probe amplitudes) by:
#       - WaveAmplitudeInput [Volt]
#       - Frekvens, men byttes med kL senere(?) 
#       - PanelConditionGrouped (full|reverse -> all; no stays no)
#       - WindCondition

#     Returns mean/std for P3/P2 (extendable with more metrics).
#     """
#     cmdf = combined_meta_df.copy()
    
#     columns = ["path", "WindCondition", "PanelCondition",
#         "WaveAmplitudeInput [Volt]", "WaveFrequencyInput [Hz]",
#         "Probe 1 Amplitude (FFT)", "Probe 2 Amplitude (FFT)", "Probe 3 Amplitude (FFT)", "Probe 4 Amplitude (FFT)",
#         "kL", "P2/P1", "P3/P2", "P4/P3"
#     ]
#     rmdf = cmdf[columns].copy()

#     # Group PanelCondition: "full" and "reverse" -> "all"; keep "no" as "no"
#     rmdf["PanelConditionGrouped"] = rmdf["PanelCondition"].replace({"full": "all", "reverse": "all"})

#     grouping_keys = [
#         "WaveAmplitudeInput [Volt]",
#         "WaveFrequencyInput [Hz]",
#         "PanelConditionGrouped",
#         "WindCondition",
#     ]

#     stats = (
#         rmdf.groupby(grouping_keys)
#             .agg(
#                 mean_P3P2=("P3/P2", "mean"),
#                 std_P3P2=("P3/P2", "std"),
#                 n_runs=("path", "nunique"),
#                 paths=("path", lambda s: pd.unique(s).tolist()),  # unique paths, order preserved
#                 # Uncomment if you want probe amplitude means as well:
#                 mean_A_Probe1=("Probe 1 Amplitude (FFT)", "mean"),
#                 mean_A_Probe2=("Probe 2 Amplitude (FFT)", "mean"),
#                 mean_A_Probe3=("Probe 3 Amplitude (FFT)", "mean"),
#                 mean_A_Probe4=("Probe 4 Amplitude (FFT)", "mean"),
#                 mean_kL=("kL", "mean")
#             )
#             .reset_index()
#     )
    
#     wide = stats.pivot_table(index=["WaveAmplitudeInput [Volt]", "PanelConditionGrouped"],
#                              columns ="WindCondition", 
#                              values = ["mean_P3P2", "std_P3P2"])
#     #optionally flatten the columns
#     wide.columns = ["_".join(map(str,col)).strip() for col in wide.columns]
#     wide = wide.reset_index()
    
# #     wide = stats.explode("paths").rename(columns={"paths": "path"})
# # # Optional: set a MultiIndex including path
# # wide = wide.set_index(["WaveAmplitudeInput [Volt]", "kL", "PanelConditionGrouped", "WindCondition", "path"])
#     return stats, wide
# def damping_grouper(combined_meta_df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Aggregates P3/P2 (and optional probe amplitudes) by:
#       - WaveAmplitudeInput [Volt]
#       - Frekvens, men byttes med kL senere(?)
#       - PanelConditionGrouped (full|reverse -> all; no stays no)
#       - WindCondition
#     Returns mean/std for P3/P2 (extendable with more metrics).
#     """
#     cmdf = combined_meta_df.copy()
    
#     print(f"Input dataframe shape: {cmdf.shape}")
#     print(f"Unique PanelCondition values: {cmdf['PanelCondition'].unique().tolist()}")
#     print(f"Unique WindCondition values: {cmdf['WindCondition'].unique().tolist()}")
    
#     columns = [
#         "path", "WindCondition", "PanelCondition",
#         "WaveAmplitudeInput [Volt]", "WaveFrequencyInput [Hz]",
#         "Probe 1 Amplitude (FFT)", "Probe 2 Amplitude (FFT)", 
#         "Probe 3 Amplitude (FFT)", "Probe 4 Amplitude (FFT)",
#         "kL", "P2/P1", "P3/P2", "P4/P3"
#     ]
    
#     # Quick safety check
#     missing_cols = [c for c in columns if c not in cmdf.columns]
#     if missing_cols:
#         print(f"WARNING: Missing columns: {missing_cols}")
    
#     rmdf = cmdf[columns].copy()
    
#     # ─── Panel grouping step ───
#     rmdf["PanelConditionGrouped"] = rmdf["PanelCondition"].replace({"full": "all", "reverse": "all"})
    
#     print("\nAfter PanelCondition grouping:")
#     print(rmdf["PanelConditionGrouped"].value_counts().to_string())
#     # or more detailed:
#     # print(rmdf.groupby("PanelCondition")["PanelConditionGrouped"].value_counts())
    
#     grouping_keys = [
#         "WaveAmplitudeInput [Volt]",
#         "WaveFrequencyInput [Hz]",
#         "PanelConditionGrouped",
#         "WindCondition",
#     ]
    
#     # Very useful: see how many unique combinations exist before aggregation
#     n_combinations = rmdf[grouping_keys].drop_duplicates().shape[0]
#     print(f"\nNumber of unique grouping combinations: {n_combinations}")
    
#     # Optional: see distribution of groups
#     group_sizes = rmdf.groupby(grouping_keys).size()
#     print("\nGroup sizes (number of rows per group):")
#     print(group_sizes.describe())
#     # if you want extremes:
#     # print("Largest groups:\n", group_sizes.nlargest(5))
#     # print("Smallest groups:\n", group_sizes.nsmallest(5))
    
#     # ─── The aggregation ───
#     stats = (
#         rmdf.groupby(grouping_keys)
#             .agg(
#                 mean_P3P2=("P3/P2", "mean"),
#                 std_P3P2=("P3/P2", "std"),
#                 n_runs=("path", "nunique"),
#                 paths=("path", lambda s: pd.unique(s).tolist()),
#                 mean_A_Probe1=("Probe 1 Amplitude (FFT)", "mean"),
#                 mean_A_Probe2=("Probe 2 Amplitude (FFT)", "mean"),
#                 mean_A_Probe3=("Probe 3 Amplitude (FFT)", "mean"),
#                 mean_A_Probe4=("Probe 4 Amplitude (FFT)", "mean"),
#                 mean_kL=("kL", "mean")
#             )
#             .reset_index()
#     )
    
#     print(f"\nAfter aggregation — stats dataframe shape: {stats.shape}")
#     print(f"Number of groups after aggregation: {len(stats)}")
#     print("Columns in stats:", stats.columns.tolist())
    
#     # Very informative: how many groups have low number of runs
#     low_n = stats[stats["n_runs"] <= 2]
#     if not low_n.empty:
#         print(f"\nWARNING: {len(low_n)} groups have ≤ 2 runs:")
#         print(low_n[["WaveAmplitudeInput [Volt]", "WaveFrequencyInput [Hz]", 
#                      "PanelConditionGrouped", "WindCondition", "n_runs"]])
    
#     # ─── Pivot ───
#     wide = stats.pivot_table(
#         index=["WaveAmplitudeInput [Volt]", "PanelConditionGrouped"],
#         columns="WindCondition",
#         values=["mean_P3P2", "std_P3P2"]
#     )
    
#     wide.columns = ["_".join(map(str, col)).strip() for col in wide.columns]
#     wide = wide.reset_index()
    
#     print(f"\nWide (pivoted) shape: {wide.shape}")
#     print("Wide columns:", wide.columns.tolist())
    
#     # Optional: quick look at the result
#     print("\nFirst few rows of wide format:")
#     print(wide.head().to_string(index=False))
    
#     return stats, wide



def damping_grouper(combined_meta_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregates P3/P2 (and optional probe amplitudes) by:
      - WaveAmplitudeInput [Volt]
      - Frequency (to be replaced with kL later?)
      - PanelConditionGrouped (full|reverse -> all; no stays no)
      - WindCondition
      
    Returns:
        tuple: (stats, wide) where stats is the aggregated DataFrame
               and wide is the pivoted version
    """
    cmdf = combined_meta_df.copy()
    
    print(f"Input dataframe shape: {cmdf.shape}")
    print(f"Unique PanelCondition values: {cmdf[GC.PANEL_CONDITION].unique().tolist()}")
    print(f"Unique WindCondition values: {cmdf[GC.WIND_CONDITION].unique().tolist()}")
    
    # Define required columns using constants
    columns = [
        GC.PATH,
        GC.WIND_CONDITION,
        GC.PANEL_CONDITION,
        GC.WAVE_AMPLITUDE_INPUT,
        GC.WAVE_FREQUENCY_INPUT,
        *CG.FFT_AMPLITUDE_COLS,  # Probe 1-4 Amplitude (FFT)
        GC.KL,
        GC.P2_P1_FFT,
        GC.P3_P2_FFT,
        GC.P4_P3_FFT,
    ]
    
    # Quick safety check
    missing_cols = [c for c in columns if c not in cmdf.columns]
    if missing_cols:
        print(f"WARNING: Missing columns: {missing_cols}")
    
    rmdf = cmdf[columns].copy()
    
    # ─── Panel grouping step ───
    PANEL_CONDITION_GROUPED = "PanelConditionGrouped"  # Temporary column name
    rmdf[PANEL_CONDITION_GROUPED] = rmdf[GC.PANEL_CONDITION].replace({
        "full": "all",
        "reverse": "all"
    })
    
    print("\nAfter PanelCondition grouping:")
    print(rmdf[PANEL_CONDITION_GROUPED].value_counts().to_string())
    
    # Define grouping keys using constants
    grouping_keys = [
        GC.WAVE_AMPLITUDE_INPUT,
        GC.WAVE_FREQUENCY_INPUT,
        PANEL_CONDITION_GROUPED,
        GC.WIND_CONDITION,
    ]
    
    # Very useful: see how many unique combinations exist before aggregation
    n_combinations = rmdf[grouping_keys].drop_duplicates().shape[0]
    print(f"\nNumber of unique grouping combinations: {n_combinations}")
    
    # Optional: see distribution of groups
    group_sizes = rmdf.groupby(grouping_keys).size()
    print("\nGroup sizes (number of rows per group):")
    print(group_sizes.describe())
    
    # ─── The aggregation ───
    stats = (
        rmdf.groupby(grouping_keys)
            .agg(
                mean_P3P2=(GC.P3_P2_FFT, "mean"),
                std_P3P2=(GC.P3_P2_FFT, "std"),
                n_runs=(GC.PATH, "nunique"),
                paths=(GC.PATH, lambda s: pd.unique(s).tolist()),
                mean_A_Probe1=(PC.AMPLITUDE_FFT.format(i=1), "mean"),
                mean_A_Probe2=(PC.AMPLITUDE_FFT.format(i=2), "mean"),
                mean_A_Probe3=(PC.AMPLITUDE_FFT.format(i=3), "mean"),
                mean_A_Probe4=(PC.AMPLITUDE_FFT.format(i=4), "mean"),
                mean_kL=(GC.KL, "mean"),
            )
            .reset_index()
    )
    
    print(f"\nAfter aggregation — stats dataframe shape: {stats.shape}")
    print(f"Number of groups after aggregation: {len(stats)}")
    print("Columns in stats:", stats.columns.tolist())
    
    # Very informative: how many groups have low number of runs
    low_n = stats[stats["n_runs"] <= 2]
    if not low_n.empty:
        print(f"\nWARNING: {len(low_n)} groups have ≤ 2 runs:")
        print(low_n[[
            GC.WAVE_AMPLITUDE_INPUT,
            GC.WAVE_FREQUENCY_INPUT,
            PANEL_CONDITION_GROUPED,
            GC.WIND_CONDITION,
            "n_runs"
        ]])
    
    # ─── Pivot ───
    wide = stats.pivot_table(
        index=[GC.WAVE_AMPLITUDE_INPUT, PANEL_CONDITION_GROUPED],
        columns=GC.WIND_CONDITION,
        values=["mean_P3P2", "std_P3P2"]
    )
    
    wide.columns = ["_".join(map(str, col)).strip() for col in wide.columns]
    wide = wide.reset_index()
    
    print(f"\nWide (pivoted) shape: {wide.shape}")
    print("Wide columns:", wide.columns.tolist())
    
    # Optional: quick look at the result
    print("\nFirst few rows of wide format:")
    print(wide.head().to_string(index=False))
    
    return stats, wide


# def damping_all_amplitude_grouper(combined_meta_df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Aggregates P3/P2 (and optional probe amplitudes) by:
#       - WaveAmplitudeInput [Volt]
#       - Frekvens, men byttes med kL senere(?) 
#       - PanelConditionGrouped (full|reverse -> all; no stays no)
#       - WindCondition

#     Returns mean/std for P3/P2 (extendable with more metrics).
#     """
#     cmdf = combined_meta_df.copy()
    
#     columns = ["path", "WindCondition", "PanelCondition",
#         "WaveAmplitudeInput [Volt]", "WaveFrequencyInput [Hz]",
#         "Probe 1 Amplitude", "Probe 2 Amplitude", "Probe 3 Amplitude", "Probe 4 Amplitude",
#         "Wavenumber", "kL", GC.P2_P1_FFT,  GC.P3_P2_FFT,  GC.P4_P3_FFT
#     ]
#     rmdf = cmdf[columns].copy()

#     # Group PanelCondition: "full" and "reverse" -> "all"; keep "no" as "no"
#     rmdf["PanelConditionGrouped"] = rmdf["PanelCondition"].replace({"full": "all", "reverse": "all"})

#     grouping_keys = [
#         "WaveFrequencyInput [Hz]",
#         "Wavenumber",
#         "PanelConditionGrouped",
#         "WindCondition",
#     ]

#     stats = (
#         rmdf.groupby(grouping_keys)
#             .agg(
#                 mean_P3P2=("P3/P2 (FFT)", "mean"),
#                 std_P3P2=("P3/P2 (FFT)", "std"),
#                 n_runs=("path", "nunique"),
#                 paths=("path", lambda s: pd.unique(s).tolist()),  # unique paths, order preserved
#                 mean_A_Probe1=("Probe 1 Amplitude", "mean"),
#                 mean_A_Probe2=("Probe 2 Amplitude", "mean"),
#                 mean_A_Probe3=("Probe 3 Amplitude", "mean"),
#                 mean_A_Probe4=("Probe 4 Amplitude", "mean"),
#                 mean_Wavenumber=("Wavenumber", "mean"),
#                 mean_kL=("kL", "mean")
                
#             )
#             .reset_index()
#     )
    
#     return stats


# Assuming these are already imported/defined:
# from your_constants import GLOBALCONSTANTS as GC
# from your_probe_constants import PC
# CG.FFT_AMPLITUDE_COLS = [PC.AMPLITUDE_FFT.format(i=i) for i in 1..4]

def damping_all_amplitude_grouper(combined_meta_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates damping-related metrics (mainly P3/P2 ratio + probe amplitudes)
    grouped by:
      - WaveAmplitudeInput [Volt]
      - WaveFrequencyInput [Hz]  (can later switch to kL)
      - PanelConditionGrouped ("full" & "reverse" → "all"; "no" stays "no")
      - WindCondition

    Returns:
        pd.DataFrame: aggregated statistics (means, stds, run counts, etc.)
    """
    cmdf = combined_meta_df.copy()

    print(f"Input dataframe shape: {cmdf.shape}")
    print(f"Unique {GC.PANEL_CONDITION}: {cmdf[GC.PANEL_CONDITION].unique().tolist()}")
    print(f"Unique {GC.WIND_CONDITION}: {cmdf[GC.WIND_CONDITION].unique().tolist()}")

    # ─── Select relevant columns using constants ───
    columns = [
        GC.PATH,
        GC.WIND_CONDITION,
        GC.PANEL_CONDITION,
        GC.WAVE_AMPLITUDE_INPUT,
        GC.WAVE_FREQUENCY_INPUT,
        *CG.FFT_AMPLITUDE_COLS,           # Probe 1–4 FFT amplitudes
        GC.KL,
        GC.P2_P1_FFT,
        GC.P3_P2_FFT,
        GC.P4_P3_FFT,
    ]

    missing_cols = [c for c in columns if c not in cmdf.columns]
    if missing_cols:
        print(f"WARNING: Missing columns: {missing_cols}")

    rmdf = cmdf[columns].copy()

    # ─── Panel grouping (same logic as damping_grouper) ───
    PANEL_CONDITION_GROUPED = "PanelConditionGrouped"
    rmdf[PANEL_CONDITION_GROUPED] = rmdf[GC.PANEL_CONDITION].replace({
        "full": "all",
        "reverse": "all"
    })

    print(f"\nAfter {PANEL_CONDITION_GROUPED} mapping:")
    print(rmdf[PANEL_CONDITION_GROUPED].value_counts().to_string())

    # ─── Grouping keys ───
    grouping_keys = [
        GC.WAVE_AMPLITUDE_INPUT,
        GC.WAVE_FREQUENCY_INPUT,
        PANEL_CONDITION_GROUPED,
        GC.WIND_CONDITION,
    ]

    # Diagnostic: data coverage
    n_combinations = rmdf[grouping_keys].drop_duplicates().shape[0]
    print(f"\nNumber of unique grouping combinations: {n_combinations}")

    group_sizes = rmdf.groupby(grouping_keys).size()
    print("\nGroup sizes (rows per group):")
    print(group_sizes.describe())

    # ─── Aggregation ───
    agg_dict = {
        "mean_P3P2":    (GC.P3_P2_FFT, "mean"),
        "std_P3P2":     (GC.P3_P2_FFT, "std"),
        "n_runs":       (GC.PATH, "nunique"),
        "paths":        (GC.PATH, lambda s: pd.unique(s).tolist()),
        "mean_kL":      (GC.KL, "mean"),
        # Probe amplitudes
        "mean_A_Probe1": (PC.AMPLITUDE_FFT.format(i=1), "mean"),
        "mean_A_Probe2": (PC.AMPLITUDE_FFT.format(i=2), "mean"),
        "mean_A_Probe3": (PC.AMPLITUDE_FFT.format(i=3), "mean"),
        "mean_A_Probe4": (PC.AMPLITUDE_FFT.format(i=4), "mean"),
    }

    # Optional additions (uncomment if needed):
    # "mean_P2P1":    (GC.P2_P1_FFT, "mean"),
    # "mean_P4P3":    (GC.P4_P3_FFT, "mean"),
    # "median_P3P2":  (GC.P3_P2_FFT, "median"),
    # "min_P3P2":     (GC.P3_P2_FFT, "min"),
    # "max_P3P2":     (GC.P3_P2_FFT, "max"),

    stats = (
        rmdf.groupby(grouping_keys)
            .agg(**agg_dict)
            .reset_index()
    )

    print(f"\nAfter aggregation — stats shape: {stats.shape}")
    print("Columns in stats:", stats.columns.tolist())

    # Warn about groups with very few runs
    low_n = stats[stats["n_runs"] <= 2]
    if not low_n.empty:
        print(f"\nWARNING: {len(low_n)} groups have ≤ 2 runs:")
        print(low_n[[
            GC.WAVE_AMPLITUDE_INPUT,
            GC.WAVE_FREQUENCY_INPUT,
            PANEL_CONDITION_GROUPED,
            GC.WIND_CONDITION,
            "n_runs"
        ]].to_string(index=False))

    return stats
    













