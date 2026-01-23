#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 09:50:06 2025

@author: ole
"""
import os
from pathlib import Path
import pandas as pd
import copy

from wavescripts.data_loader import load_or_update
from wavescripts.filters import filter_chosen_files
from wavescripts.processor import process_selected_data

file_dir = Path(__file__).resolve().parent
os.chdir(file_dir) # Make the script always run from the folder where THIS file lives
# ------------------------------------------------
#%%# Python
# Python
from pathlib import Path
import copy
import pandas as pd

base_processvars = {
    "filters": {
        "amp": 0.1,
        "freq": 0.65,
        "per": None,
        "wind": "no",
        "tunnel": None,
        "mooring": "low",
        "panel": ["full", "reverse"],
    },
    "processing": {
        "chosenprobe": "Probe 3",
        "rangestart": None,
        "rangeend": None,
        "data_cols": ["Probe 2"],
        "win": 11
    },
    "plotting": {
        "figsize": None,
        "separate": True,
        "overlay": False
    }
}

datasets = [
    {
        "name": "20251112-tett6roof",
        "path": Path("/Users/ole/Kodevik/wave_project/wavedata/20251112-tett6roof"),
        "overrides": {}
    },
    {
        "name": "20251113-tett6roof-loosepaneltaped",
        "path": Path("/Users/ole/Kodevik/wave_project/wavedata/20251113-tett6roof-loosepaneltaped"),
        "overrides": {
            "filters": {"amp": 0.2, "panel": ["full"]},
            "processing": {"win": 13},
        }
    }
]

chooseAll = False
chooseFirst = False
debug = True
win = 10
find_range = True
range_plot = True

try:
    dataset_registry
except NameError:
    dataset_registry = {}

def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            deep_update(d[k], v)
        else:
            d[k] = v
    return d

for ds in datasets:
    name = ds["name"]
    if name in dataset_registry:
        # Skip if already processed in this session
        continue

    processvars = copy.deepcopy(base_processvars)
    deep_update(processvars, ds.get("overrides", {}))

    dfs, meta = load_or_update(ds["path"])

    # Run your selection
    meta_sel = filter_chosen_files(meta, processvars, chooseAll, chooseFirst)

    # Ensure meta_sel is a proper row-subset that includes 'path'.
    # If the filter returned a reduced column set, rebuild from meta by index once.
    if isinstance(meta_sel, pd.DataFrame):
        if "path" not in meta_sel.columns:
            # Reconstitute using the index (assumes index alignment is preserved)
            if meta_sel.index.isin(meta.index).all():
                meta_sel = meta.loc[meta_sel.index].copy()
            else:
                raise KeyError(
                    "meta_sel is missing 'path' and does not share index with meta. "
                    "Please make filter_chosen_files return a row-subset (all columns), "
                    "or return an id so we can re-join."
                )
    else:
        # If filter returns mask/index, make a row-subset with all columns
        meta_sel = meta.loc[meta_sel].copy()

    # Final guard
    if "path" not in meta_sel.columns:
        raise KeyError("Selected metadata (meta_sel) lacks required 'path' column.")

    # Process: pass meta_sel as BOTH the selection and the meta argument
    # so the original meta is ignored by the processor.
    processed_dfs, meta_sel = process_selected_data(
        dfs,
        meta_sel,   # selection
        meta_sel,   # meta (use the same subset so full meta is ignored)
        debug,
        win,
        find_range,
        range_plot
    )

    dataset_registry[name] = {
        "path": ds["path"],
        "dfs": dfs,
        "meta_sel": meta_sel,            # canonical metadata
        "processed_dfs": processed_dfs,
        "processvars": processvars
    }

# Optional: quick combined view of selections (for inspection only)
def combined_meta_sel(registry: dict) -> pd.DataFrame:
    frames = []
    for k, r in registry.items():
        df = r.get("meta_sel")
        if isinstance(df, pd.DataFrame) and not df.empty:
            tmp = df.copy()
            tmp["dataset"] = k
            frames.append(tmp)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

meta_sel_catalog = combined_meta_sel(dataset_registry)











