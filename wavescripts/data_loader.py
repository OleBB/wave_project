#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 14:29:18 2025

@author: gpt fixa grok
"""

from pathlib import Path
from typing import Iterator, Dict, Tuple
import json

import pandas as pd


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
    
    meinHeaders = ["Date", "Probe 1", "Probe 2", "Probe 3", "Probe 4", "Mach 1"]
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

            dfs[key] = df
            
            #print(path.name.strip())
            """SE FILEN PATHFINDER_MORO.py for ferdig klipp og lim kode til å hente ut fra filnavn."""
            row = {
                "filename": path.name,
                "path": key,
                "rows": int(len(df)),
                "cols": int(len(df.columns)),
                "size_mb": path.stat().st_size / 1e6,
                "Panelcondition": 999,
                "Windcondition": 999, 
                "input amplitude(Volt)": 999,
                "input Frequency (Hz)": 999, 
                "input periods": 999, 
                "depth": 999,
                "Extra time, before measurement stops": 999, 
                "run number": 999
            }
            meta_list.append(row)

            print(f"  [{i}/{len(new_files)}] Loaded {path.name} → {len(df):,} rows")

        except Exception as e:
            print(f"  Failed to load {path.name}: {e}")

    # ---- Save updated cache ----
    pd.to_pickle(dfs, dfs_path)
    meta_path.write_text(json.dumps(meta_list, indent=2))
    print(f"Cache updated → {len(dfs)} files now cached in {cache_dir}")

    # Return DataFrame view of metadata
    meta_df = pd.DataFrame(meta_list)
    return dfs, meta_df



FOLDER1 = Path("/Users/ole/Kodevik/wave_project/wavedata/20251110-tett6roof-lowM-ekte580")

if __name__ == "__main__":
    print(f"CALLING load_or_update with: {FOLDER1}") 
    dfs, meta = load_or_update(FOLDER1)
    
    print(f"\nSUCCESS: {len(dfs)} DataFrames loaded and cached!") 
    print(f"Metadata shape: {meta.shape}")





