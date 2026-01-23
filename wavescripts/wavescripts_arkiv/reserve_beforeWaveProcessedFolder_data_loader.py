#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 14:29:18 2025

@author: gpt fixa grok
"""

from pathlib import Path
from typing import Iterator, Dict, Tuple
import json
import re
import pandas as pd
import os
from datetime import datetime


# -------------------------------------------------
# File discovery
# -------------------------------------------------
def get_data_files(folder: Path) -> Iterator[Path]:
    """
    Recursively yield data files from `folder`.

    - Supports: csv, CSV, parquet, h5, feather
    - Ignores: *.stats.csv
    """
    folder = Path(folder)

    if not folder.exists() or not folder.is_dir():
        print(f"  Folder not found or not a directory: {folder}")
        return iter([])

    patterns = ["*.csv", "*.CSV", "*.parquet", "*.h5", "*.feather"]
    total = 0

    for pat in patterns:
        # All matches for this pattern
        matches = list(folder.rglob(pat))

        # Filter out stats files
        matches = [m for m in matches if not m.name.endswith(".stats.csv")]

        if matches:
            print(f"  Found {len(matches)} files with pattern {pat}")
            total += len(matches)
            for m in matches:
                yield m

    if total == 0:
        print(f"  No data files found in {folder}")


# -------------------------------------------------
# Cache loader / updater
# -------------------------------------------------
def load_or_update(
    *folders: Path,
    cache_dir: Path = Path("data_cache"),
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Load cached DataFrames and metadata from `cache_dir`,
    then scan `folders` for new files and load those.

    - DataFrames are cached in dfs.pkl (pickle of dict[str, DataFrame])
    - Metadata is cached in meta.json (list of dicts)
    - Returns: (dfs_dict, meta_dataframe)
    """

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    dfs_path = cache_dir / "dfs.pkl"
    meta_path = cache_dir / "meta.json"

    # ---- Load existing cache (if present) ----
    dfs: Dict[str, pd.DataFrame] = {}
    meta_list: list[dict] = []

    if dfs_path.exists() and meta_path.exists():
        print("Loading cached data...")
        try:
            dfs = pd.read_pickle(dfs_path)
            meta_list = json.loads(meta_path.read_text())
            print(f" → {len(dfs)} files already cached")
        except Exception as e:
            print(f"Cache corrupted ({e}), rebuilding from scratch...")
    else:
        print("No cache found → starting fresh")

    # Keys in dfs are absolute paths as strings
    seen = set(dfs.keys())
    new_files: list[Path] = []

    # ---- Discover new files ----
    for folder in folders:
        folder = Path(folder)
        if not folder.is_dir():
            print(f"Warning: folder not found or not a directory: {folder}")
            continue

        for path in get_data_files(folder):
            key = str(path.resolve())
            if key not in seen:
                new_files.append(path)

    if not new_files:
        print("No new files found.")
        # Return dfs + DataFrame view of meta_list
        return dfs, pd.DataFrame(meta_list)

    print(f"Loading {len(new_files)} new files...")
    
    # ---- Load new files ----
    for i, path in enumerate(new_files, 1):
        key = str(path.resolve())
        try:
            suffix = path.suffix.lower()

            if suffix == ".csv":
                df = pd.read_csv(path,names=["Date", "Probe 1", "Probe 2", "Probe 3", "Probe 4", "Mach"])
            elif suffix == ".parquet":
                # Will raise if pyarrow/fastparquet not installed
                df = pd.read_parquet(path)
            elif suffix in (".h5", ".hdf5"):
                df = pd.read_hdf(path)
            elif suffix == ".feather":
                df = pd.read_feather(path)
            else:
                print(f"  Skipping unsupported type: {path.name}")
                continue
            
            #--------------------------#
            #------ FORMATTING --------#
            #--------------------------#
            for i in range(1, 5):
                df[f"Probe {i}"] *= 1000 #gange med millimeter

            df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y %H:%M:%S.%f")
            
            #--------------------------#
            #- Her puttes df i en dict-#
            #--------------------------#
            dfs[key] = df 
            
            filename = path.name
            metadata = {
                "path": key,
                "WindCondition": "",
                "TunnelCondition": "",       # Default value
                "PanelCondition": "",    # Default value
                "Mooring": "",          # high was default BEFORE 20251106-fullwind.
                "WaveAmplitudeInput [Volt]": "",       
                "WaveFrequencyInput [Hz]": "",          
                "WavePeriodInput": "",
                "WaterDepth [mm]": "",         
                "Extra seconds": "",         
                "Run number": "",
                "Stillwater Probe 1": "",
                "Stillwater Probe 2": "",
                "Stillwater Probe 3": "",
                "Stillwater Probe 4": "",
                "Computed Probe 1 start" : "",
                "Computed Probe 2 start" : "",
                "Computed Probe 3 start" : "",
                "Computed Probe 4 start" : "",
                "Computed range" : ""   
            }
            #########################
            #Big note to self! viktig beskjed
            #dersom noe gjøres med metadata-teksten så må
            #man være påpasselig med å endre navn der inputen taes inn
            #... som er...(?)?)?)
            #########################
            stillwater_samples = 250
            ## VARIABEL
            
            wind_match = re.search(r'-([A-Za-z]+)wind-', filename)
            if wind_match:
                metadata["WindCondition"] = wind_match.group(1)
                if wind_match.group(1) == "no":
                    metadata["Stillwater Probe 1"]  = df["Probe 1"].loc[0:stillwater_samples].mean(skipna=True)
                    metadata["Stillwater Probe 2"] = df["Probe 2"].loc[0:stillwater_samples].mean(skipna=True)
                    metadata["Stillwater Probe 3"] = df["Probe 3"].loc[0:stillwater_samples].mean(skipna=True)
                    metadata["Stillwater Probe 4"] = df["Probe 4"].loc[0:stillwater_samples].mean(skipna=True)
                
            tunnel_match = re.search(r'([0-9])roof', filename)
            if tunnel_match:
                metadata["TunnelCondition"] = tunnel_match.group(1) + " roof plates"
            
            panel_match = re.search(r'([A-Za-z]+)panel', filename)
            if panel_match:
                metadata["PanelCondition"] = panel_match.group(1)
            
            modtime = os.path.getmtime(path)
            date = datetime.fromtimestamp(modtime)
            date_match = re.search(r'(\d{8})', filename)
            if date_match:
                    print('first if')
                    metadata["Date"] = date_match.group(1)
                    if metadata["Date"] < "20251106":
                        metadata["Mooring"] = "high"
                    else:
                        metadata["Mooring"] = "low"
            mooringdate = '2025-11-6'
            mooringdatetime = datetime.strptime(mooringdate, '%Y-%m-%d')
            if date < mooringdatetime:
                metadata["Mooring"] = "high"
            else:
                metadata["Mooring"] = "low"
                    
            amplitude_match = re.search(r'-amp([A-Za-z0-9]+)-', filename)
            if amplitude_match:
                raw_amp = amplitude_match.group(1)
                metadata["WaveAmplitudeInput [Volt]"] = int(raw_amp)/1000.0

            freq_match = re.search(r'-freq(\d+)-', filename)
            if freq_match:
                raw_freq = freq_match.group(1)
                metadata["WaveFrequencyInput [Hz]"] = int(raw_freq) / 1000.0

            per_match = re.search(r'-per(\d+)-', filename)
            if per_match:
                metadata["WavePeriodInput"] = int(per_match.group(1))

            depth_match = re.search(r'-depth([A-Za-z0-9]+)', filename)
            if depth_match:
                metadata["WaterDepth [mm]"] =  int(depth_match.group(1))
                
            mstop_match = re.search(r'-mstop([A-Za-z0-9]+)', filename)
            if mstop_match:
                metadata["Extra seconds"] =  int(mstop_match.group(1))
            
            run_match = re.search(r'-run([0-9])', filename, re.IGNORECASE)  # case-insensitive match
            if run_match:
                metadata["Run number"] = run_match.group(1).lower()  # Store as lower case for consistency
            meta_list.append(metadata)
            
            # - - - - - - - - - - - #
            
            
            
            print(f"  [{i}/{len(new_files)}] Loaded {path.name} → {len(df):,} rows")
            
        except Exception as e:
            print(f"  Failed to load {path.name}: {e}")

    # ---- Save updated cache ----
    pd.to_pickle(dfs, dfs_path)
    meta_path.write_text(json.dumps(meta_list, indent=2))
    print(f"Cache updated → {len(dfs)} files now cached in {cache_dir}")

    # Return DataFrame view of metadata
    meta_df = pd.DataFrame(meta_list)
    print(meta_df)
    return dfs, meta_df


def update_metadata(df_sel, skip_empty_strings=True):
    #nå har vi tatt inn en df der rett row må svare til rett json-entry 
    #we now have a small dataframe "df_sel" that contains only the few 
    #paths we want to update. 
    """df_sel.dtypes
    Out[9]: 
    path                         object
    WindCondition                object
    TunnelCondition              object
    PanelCondition               object
    Mooring                      object
    WaveAmplitudeInput [Volt]    object
    WaveFrequencyInput [Hz]      object
    WavePeriodInput              object
    WaterDepth [mm]              object
    Extra seconds                object
    Run number                   object
    Stillwater Probe 1           object
    Stillwater Probe 2           object
    Stillwater Probe 3           object
    Stillwater Probe 4           object
    Computed Probe 1 start        int64
    Computed Probe 2 start        int64
    Computed Probe 3 start        int64
    Computed Probe 4 start        int64
    Computed range               object
    dtype: object"""
    #JSON file style:
    """{
    "path": "/Users/ole/Kodevik/wave_project/wavedata/20251110-tett6roof-lowMooring/fullpanel-fullwind-amp0100-freq1300-per30-depth580-mstop10-run1.csv",
    "WindCondition": "full",
    "TunnelCondition": "",
    "PanelCondition": "full",
    "Mooring": "low",
    "WaveAmplitudeInput [Volt]": 0.1,
    "WaveFrequencyInput [Hz]": 1.3,
    "WavePeriodInput": 30,
    "WaterDepth [mm]": "580",
    "Extra seconds": "10",
    "Run number": "1",
    "Stillwater Probe 1": "",
    "Stillwater Probe 2": "",
    "Stillwater Probe 3": "",
    "Stillwater Probe 4": "",
    "Computed Probe 1 start": "",
    "Computed Probe 2 start": "",
    "Computed Probe 3 start": "",
    "Computed Probe 4 start": "",
    "Computed range": ""
  },"""
    all_updates = {}
    for _, row in df_sel.iterrows():
        paf = row["path"]
        update = {}
        for column, value in row.items():
            if column == paf:
                continue
            if pd.isna(value):
                continue
            if skip_empty_strings==True and isinstance(value, str) and value == "":
                continue
            update[column] = value
        if update:
            all_updates[paf] = update
    return all_updates


def apply_updates_to_metadata(metadata_list, updates):
    # metadata_list is a list[dict] like in your JSON
    for obj in metadata_list:
        p = 1
        if p in updates:
            obj.update(updates[p])
    return metadata_list

#example usage
#build updates from dataframes

    
#######    

FOLDER1 = Path("/Users/ole/Kodevik/wave_project/wavedata/20251110-tett6roof-lowM-ekte580")

if __name__ == "__main__":
    print(f"CALLING load_or_update with: {FOLDER1}") 
    dfs, meta = load_or_update(FOLDER1)
    
    print(f"\nSUCCESS: {len(dfs)} DataFrames loaded and cached!") 
    print(f"Metadata shape: {meta.shape}")





