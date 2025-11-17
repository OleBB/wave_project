#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 16:06:42 2025

@author: ole
"""

#from wavescripts.wave_processor import WaveProcessor
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import os
import re

"""1 . Lese inn signalet"""

the_project_root = Path(__file__).parent
the_data_folder = the_project_root / "wavedata/20251110-tett6roof-lowM-ekte580"
the_results_folder = the_project_root / "waveresults"

the_results_folder.mkdir(exist_ok=True) # Ensure results folder exists

#print('root is =  ', the_project_root)
#print('datafolder = ', the_data_folder)
#print('resultatene ligger i =', the_results_folder)


#grok

FILENAME_PATTERN = re.compile(
    r"(?P<panel>\w+)-(?P<wind>\w+)-amp(?P<amp>\d{4})-freq(?P<freq>\d{4})-"
    r"per(?P<periods>\d+)-depth(?P<depth>\d+)-mstop(?P<mstop>\d+)-run(?P<run>\d+)\.csv$"
)

def parse_filename(fname: str) -> dict:
    m = FILENAME_PATTERN.match(fname)
    if not m:
        raise ValueError(f"Invalid filename: {fname}")
    d = m.groupdict()
    d["amp"] = float(d["amp"]) / 10000
    d["freq"] = float(d["freq"]) / 1000
    d["periods"], d["depth"], d["mstop"], d["run"] = map(int, [
        d["periods"], d["depth"], d["mstop"], d["run"]
    ])
    return d

def get_data_files(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.glob("*.csv")
        if p.is_file() and p.suffix == ".csv" and not p.name.endswith(".stats.csv")
    )



def load_or_update(
    cache_dir: Path = Path("data_cache"),
    *folders: Path
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Load cached data + add new folders.
    Creates cache_dir if missing.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    dfs_path = cache_dir / "dfs.pkl"
    meta_path = cache_dir / "meta.parquet"

    # --- Load existing ---
    dfs = {}
    meta = pd.DataFrame()
    if dfs_path.exists() or meta_path.exists():
        print("Loading cached data...")
        print(f"Cache files exist: dfs.pkl={dfs_path.exists()}, meta.parquet={meta_path.exists()}")
        dfs = pd.read_pickle(dfs_path)
        meta = pd.read_parquet(meta_path)
        print(f"  → {len(dfs)} files already loaded")

    # --- Find new files ---
    seen = set(dfs.keys())
    new_files = []
    for folder in folders:
        for path in get_data_files(folder):
            if path.name not in seen:
                new_files.append(path)

    if not new_files:
        print("No new files found.")
        return dfs, meta

    print(f"Adding {len(new_files)} new files...")
    new_dfs = {}
    new_meta = []

    for path in new_files:
        params = parse_filename(path.name)
        params["filename"] = path.name

        df = pd.read_csv(
            path,
            header=None,
            names=["datetime", "p1", "p2", "p3", "p4", "ref"],
            parse_dates=["datetime"],
            dtype="float32",
        )
        t0 = df["datetime"].iloc[0]
        df["t_sec"] = (df["datetime"] - t0).dt.total_seconds().astype("float32")
        df = df.drop(columns=["datetime"]).set_index("t_sec")

        new_dfs[path.name] = df
        new_meta.append(params)

    # --- Merge ---
    dfs.update(new_dfs)
    if new_meta:
        new_meta_df = pd.DataFrame(new_meta).set_index("filename")
        meta = pd.concat([meta, new_meta_df])

    # --- Save cache ---
    pd.to_pickle(dfs, dfs_path)
    meta.to_parquet(meta_path)
    print(f"Updated cache: {len(dfs)} total files")

    return dfs, meta

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
FOLDER1 = PROJECT_ROOT / "wavedata" / "20251110-tett6roof-lowM-ekte580"

# First run
dfs, meta = load_or_update(FOLDER1)


#%%
#GROK
def get_data_files(folder: Path, pattern: str = "*.csv") -> List[Path]:
    #"""Return a sorted list of Path objects for CSV files in `folder` that: are real files, end with .csv, do NOT end with .stats.csv"""
    return sorted(
        p
        for p in folder.glob(pattern)
        if p.is_file()
        and p.suffix == ".csv"
        and not p.name.endswith(".stats.csv")
    )

the_data_files = get_data_files(the_data_folder)
print(f"Found {len(the_data_files)} data files (excluding .stats.csv)")

#%%
# Then read lazily
def read_files(file_list):
    for path in file_list:
        df = pd.read_csv(path, ...)
        df = df.assign(file=path.name)
        yield path.name, df

# ------------------------------------------------------------------
# 1. Lazy generator – yields a DataFrame + filename for *selected* CSVs
# ------------------------------------------------------------------
def csv_generator(folder: str, pattern: str = "*.csv"):
    for path in sorted(Path(folder).glob(pattern)):
        # ---- parsing tricks ------------------------------------------------
        df = pd.read_csv(
            path,
            header=None,                     # no header line
            names=["datetime", "p1", "p2", "p3", "p4", "ref"],
            parse_dates=["datetime"],        # proper Timestamp
            dtype={                          # save memory
                "p1": "float32",
                "p2": "float32",
                "p3": "float32",
                "p4": "float32",
                "ref": "float32",
            },
        )
        yield path.name, df
# Dictionary:  file_name  →  DataFrame
dfs = {fname: df for fname, df in csv_generator("data_folder")}

# Compare p3 of two specific files
df_a = dfs["run_01.csv"]
df_b = dfs["run_02.csv"]

# numeric
corr = df_a["p3"].corr(df_b["p3"])

# visual
import matplotlib.pyplot as plt
plt.plot(df_a["datetime"], df_a["p3"], label="run_01 p3")
plt.plot(df_b["datetime"], df_b["p3"], label="run_02 p3")
plt.legend(); plt.show()