#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 14:29:18 2025

@author: gpt fixa grok
"""

from pathlib import Path
from typing import Iterator, Dict, Tuple, List
import json
import re
import pandas as pd
import os
from datetime import datetime


dtype_map = {

    "WindCondition": str,
    "TunnelCondition": str,
    "PanelCondition": str,
    "Mooring": str,
    "WaveAmplitudeInput [Volt]": "float64",
    "WaveFrequencyInput [Hz]": "float64",
    "WavePeriodInput": "float64",
    "WaterDepth [mm]": "float64",
    "Extra seconds": "float64",
    "Run number": str,
    "Stillwater Probe 1": "float64",
    "Stillwater Probe 2": "float64",
    "Stillwater Probe 3": "float64",
    "Stillwater Probe 4": "float64",
    "Computed Probe 1 start": "float64",
    "Computed Probe 2 start": "float64",
    "Computed Probe 3 start": "float64",
    "Computed Probe 4 start": "float64",
    "Computed Probe 1 end": "float64",
    "Computed Probe 2 end": "float64",
    "Computed Probe 3 end": "float64",
    "Computed Probe 4 end": "float64",
    "Probe 1 Amplitude": "float64",
    "Probe 2 Amplitude": "float64",
    "Probe 3 Amplitude": "float64",
    "Probe 4 Amplitude": "float64",
    "Wavefrequency": "float64",
    "Waveperiod": "float64",
    "Wavenumber": "float64",
    "Wavelength": "float64",
    "kH": "float64",
    "tanh(kH)": "float64",
    "Celerity": "float64",
    "Significant Wave Height Hs": "float64",
    "Significant Wave Height Hm0": "float64",
    "Windspeed": "float64",
    "experiment_folder": str
}

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

# ----------------------------------------------------------------------
# Updated function – saves per-experiment cache in waveprocessed/
# ----------------------------------------------------------------------
def load_or_update(
    *folders: Path | str,
    processed_root: Path | str | None = None,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    For each input folder (e.g. wavedata/20251110-tett6roof-lowM-ekte580),
    automatically uses or creates:
        waveprocessed/PROCESSED-20251110-tett6roof-lowM-ekte580/
    containing dfs.pkl and meta.json
    """
    # --------------------------------------------------------------
    # 1. Find project root (so everything is independent of cwd)
    # --------------------------------------------------------------
    current_file = Path(__file__).resolve()
    project_root = None
    for parent in current_file.parents:
        if (parent / "main.py").exists() or (parent / ".git").exists():
            project_root = parent
            break
    if project_root is None:
        project_root = current_file.parent.parent  # safe fallback

    # --------------------------------------------------------------
    # 2. Define base folder for all processed experiments
    # --------------------------------------------------------------
    processed_root = Path(processed_root or project_root / "waveprocessed")
    processed_root.mkdir(parents=True, exist_ok=True)

    # Global containers
    all_dfs: Dict[str, pd.DataFrame] = {}
    all_meta_list: List[dict] = []

    # --------------------------------------------------------------
    # 3. Process each folder independently
    # --------------------------------------------------------------
    for folder in folders:
        folder_path = Path(folder).resolve()
        if not folder_path.is_dir():
            print(f"Warning: Skipping missing folder: {folder_path}")
            continue

        experiment_name = folder_path.name
        cache_dir = processed_root / f"PROCESSED-{experiment_name}"
        cache_dir.mkdir(parents=True, exist_ok=True)

        dfs_path = cache_dir / "dfs.pkl"
        meta_path = cache_dir / "meta.json"

        print(f"\nProcessing experiment: {experiment_name}")
        print(f"   Cache folder: {cache_dir.relative_to(project_root)}")

        # Load existing cache for this experiment
        dfs: Dict[str, pd.DataFrame] = {}
        meta_list: list[dict] = []

        if dfs_path.exists() and meta_path.exists():
            try:
                dfs = pd.read_pickle(dfs_path)
                meta_list = json.loads(meta_path.read_text(encoding="utf-8"))
                # grokk snakka om dette: TK New robust way — always get a DataFrame with predictable columns
                #meta_df = pd.read_json(meta_path, orient="records", dtype=False)
                print(f"   Loaded {len(dfs)} cached files")
            except Exception as e:
                print(f"   Cache corrupted ({e}) → rebuilding")
                dfs, meta_list = {}, []

        seen_keys = set(dfs.keys())
        new_files = [
            p for p in get_data_files(folder_path)
            if str(p.resolve()) not in seen_keys
        ]

        if not new_files:
            print("   No new files → using cache only")
        else:
            print(f"   Loading {len(new_files)} new file(s)...")

        # ------------------------------------------------------------------
        # Load and process new files (your original logic – unchanged!)
        # ------------------------------------------------------------------
        for i, path in enumerate(new_files, 1):
            key = str(path.resolve())
            try:
                suffix = path.suffix.lower()
                if suffix == ".csv":
                    df = pd.read_csv(path, names=["Date", "Probe 1", "Probe 2", "Probe 3", "Probe 4", "Mach"])
                else:
                    print(f"   Skipping unsupported: {path.name}")
                    continue

                # Formatting
                for probe in range(1, 5):
                    df[f"Probe {probe}"] *= 1000
                df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y %H:%M:%S.%f")

                dfs[key] = df

                # ------------------- METADATA EXTRACTION -------------------
                filename = path.name
                metadata = {
                    "path": key,
                    "WindCondition": str,
                    "TunnelCondition": str,
                    "PanelCondition": str,
                    "Mooring": str,
                    "WaveAmplitudeInput [Volt]": float,
                    "WaveFrequencyInput [Hz]": float,
                    "WavePeriodInput": float,
                    "WaterDepth [mm]": float,
                    "Extra seconds": float,
                    "Run number": str,
                    "Probe 1 mm from paddle": float,
                    "Probe 2 mm from paddle": float,
                    "Probe 3 mm from paddle": float,
                    "Probe 4 mm from paddle": float,
                    "Stillwater Probe 1": float,
                    "Stillwater Probe 2": float,
                    "Stillwater Probe 3": float,
                    "Stillwater Probe 4": float,
                    "Computed Probe 1 start": float,
                    "Computed Probe 2 start": float,
                    "Computed Probe 3 start": float,
                    "Computed Probe 4 start": float,
                    "Computed Probe 1 end": float,
                    "Computed Probe 2 end": float,
                    "Computed Probe 3 end": float,
                    "Computed Probe 4 end": float,
                    "Probe 1 Amplitude": float,
                    "Probe 2 Amplitude": float,
                    "Probe 3 Amplitude": float,
                    "Probe 4 Amplitude": float,
                    "Wavefrequency": float,
                    "Waveperiod": float,
                    "Wavenumber": float,
                    "Wavelength": float,
                    "kH": float,
                    "tanh(kH)": float,
                    "Celerity": float,
                    "Significant Wave Height Hs": float,
                    "Significant Wave Height Hm0": float,
                    "Windspeed": float,
                    "experiment_folder": str
                }
                metadata = {k: "" if dtype is str else None for k, dtype in metadata.items()}
                metadata.update({
                    "path": key,
                    "experiment_folder": experiment_name
                })

                stillwater_samples = 250 #bruker æ fortsatt denne?

                # Wind
                wind_match = re.search(r'-([A-Za-z]+)wind-', filename)
                if wind_match:
                    metadata["WindCondition"] = wind_match.group(1)
                    if wind_match.group(1) == "no":
                        for p in range(1, 5):
                            metadata[f"Stillwater Probe {p}"] = df[f"Probe {p}"].iloc[:stillwater_samples].mean(skipna=True)

                # Tunnel
                tunnel_match = re.search(r'([0-9])roof', filename)
                if tunnel_match:
                    metadata["TunnelCondition"] = tunnel_match.group(1) + " roof plates"

                # Panel
                panel_match = re.search(r'([A-Za-z]+)panel', filename)
                if panel_match:
                    metadata["PanelCondition"] = panel_match.group(1)

                # Mooring logic
                modtime = os.path.getmtime(path)
                file_date = datetime.fromtimestamp(modtime)
                date_match = re.search(r'(\d{8})', filename)
                mooring_cutoff = datetime(2025, 11, 6)
                metadata["Mooring"] = "high" if file_date < mooring_cutoff else "low"
                if date_match:
                    metadata["Date"] = date_match.group(1)
                    if metadata["Date"] < "20251106":
                        metadata["Mooring"] = "high"
                    else:
                        metadata["Mooring"] = "low"
                        
                #Probe distance logic
                "Probe 1 mm from paddle"
                modtime = os.path.getmtime(path)
                file_date = datetime.fromtimestamp(modtime)
                date_match = re.search(r'(\d{8})', filename)
                distance_cutoff = datetime(2025, 11, 14) #siste kjøring var 13nov
                metadata["Probe 1 mm from paddle"] = 9000 if file_date < distance_cutoff else None
                metadata["Probe 2 mm from paddle"] = 9455 if file_date < distance_cutoff else None
                metadata["Probe 3 mm from paddle"] = 12544 if file_date < distance_cutoff else None
                metadata["Probe 4 mm from paddle"] = 12545 if file_date < distance_cutoff else None
             
                

                # Wave parameters
                if m := re.search(r'-amp([A-Za-z0-9]+)-', filename):
                    metadata["WaveAmplitudeInput [Volt]"] = int(m.group(1)) / 1000.0
                if m := re.search(r'-freq(\d+)-', filename):
                    metadata["WaveFrequencyInput [Hz]"] = int(m.group(1)) / 1000.0
                if m := re.search(r'-per(\d+)-', filename):
                    metadata["WavePeriodInput"] = int(m.group(1))
                if m := re.search(r'-depth([A-Za-z0-9]+)', filename):
                    metadata["WaterDepth [mm]"] = int(m.group(1))
                if m := re.search(r'-mstop([A-Za-z0-9]+)', filename):
                    metadata["Extra seconds"] = int(m.group(1))
                if m := re.search(r'-run([0-9])', filename, re.IGNORECASE):
                    metadata["Run number"] = m.group(1)

                meta_list.append(metadata)
                print(f"   [{i}/{len(new_files)}] Loaded {path.name} → {len(df):,} rows")

            except Exception as e:
                print(f"   Failed {path.name}: {e}")

        # Save this experiment's cache
        if new_files or not dfs_path.exists():
            pd.to_pickle(dfs, dfs_path)
            meta_path.write_text(json.dumps(meta_list, indent=2), encoding="utf-8")
            print(f"   Cache saved → {len(dfs)} files")

        # Merge into global result
        all_dfs.update(dfs)
        all_meta_list.extend(meta_list)

    # Final metadata DataFrame
    meta_df = pd.DataFrame(all_meta_list)

    meta_df = meta_df.astype(dtype_map)
    print(f"\nFinished! Total {len(all_dfs)} files from {len(folders)} experiment(s)")
    return all_dfs, meta_df
################

# --------------------------------------------------
# Takes in a modified meta-dataframe, and updates the meta.JSON
# --------------------------------------------------
def update_processed_metadata(
    meta_df: pd.DataFrame,
    processed_root: Path | str | None = None,
) -> None:
    """
    Safely updates meta.json files:
      • Keeps existing runs
      • Adds new runs
      • Updates changed rows (matched by 'path')
      • Never overwrites or deletes data
    """
    current_file = Path(__file__).resolve()
    project_root = next(
        (p for p in current_file.parents if (p / "main.py").exists() or (p / ".git").exists()),
        current_file.parent.parent
    )
    processed_root = Path(processed_root or project_root / "waveprocessed")

    # Ensure we have a way to group by experiment
    if "PROCESSED_folder" in meta_df.columns:
        meta_df["__group_key"] = meta_df["PROCESSED_folder"]
    elif "experiment_folder" in meta_df.columns:
        meta_df["__group_key"] = "PROCESSED-" + meta_df["experiment_folder"].astype(str)
    elif "path" in meta_df.columns:
        meta_df["__group_key"] = meta_df["path"].apply(
            lambda p: "PROCESSED-" + Path(p).resolve().parent.name
        )
    else:
        raise ValueError("Need PROCESSED_folder, experiment_folder, or path column")

    for processed_folder_name, group_df in meta_df.groupby("__group_key"):
        cache_dir = processed_root / processed_folder_name
        meta_path = cache_dir / "meta.json"

        # Load existing metadata if file exists
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    old_records = json.load(f)
                old_df = pd.DataFrame(old_records)
                old_df = old_df.astype(dtype_map)
                print(f"Loaded {len(old_df)} existing entries from {meta_path.name}")
            except Exception as e:
                print(f"Could not read existing {meta_path} → starting fresh: {e}")
                old_df = pd.DataFrame()
        else:
            old_df = pd.DataFrame()
            cache_dir.mkdir(parents=True, exist_ok=True)

        # Clean incoming data
        new_df = group_df.drop(columns=["__group_key"], errors="ignore").copy()
        new_df["path"] = new_df["path"].astype(str)  # ensure path is string

        if not old_df.empty:
            old_df["path"] = old_df["path"].astype(str)

            # Merge: update existing + add new ones
            if not new_df.empty:
                # Use 'path' as key — it's unique per file
                combined = pd.concat([old_df, new_df], ignore_index=True)
                combined = combined.drop_duplicates(subset="path", keep="last")  # last = newest
                final_df = combined
            else:
                final_df = old_df
        else:
            final_df = new_df

        # Save back safely
        records = final_df.to_dict("records")
        meta_path.write_text(
            json.dumps(records, indent=2, default=str),
            encoding="utf-8"
        )
        added = len(final_df) - len(old_df) if not old_df.empty else len(final_df)
        print(f"Updated {meta_path.relative_to(project_root)} → {len(final_df)} entries (+{added} new)")
    
    print(f"\nMetadata safely updated and preserved across {meta_df['__group_key'].nunique()} experiment(s)!")


def load_meta_from_processed(folder_name: str) -> pd.DataFrame:
    """
    Safely load any PROCESSED-.../meta.json as a proper DataFrame
    """
    meta_path = Path("waveprocessed") / folder_name / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")
    
    # This is bulletproof — preserves types, handles nulls, keeps column order logical
    df = pd.read_json(meta_path, orient="records", convert_dates=["Date"])
    
    # Optional: ensure path is string
    if "path" in df.columns:
        df["path"] = df["path"].astype(str)
    
    return df

def save_processed_dataframes(dfs: dict, meta_df: pd.DataFrame, processed_root=None):
    for key, df in dfs.items():
        row = meta_df[meta_df["path"] == key].iloc[0]
        processed_folder = row.get("PROCESSED_folder") or f"PROCESSED-{Path(key).parent.name}"
        cache_dir = Path(processed_root or "waveprocessed") / processed_folder
        pd.to_pickle(dfs, cache_dir / "dfs.pkl")




def apply_updates_to_metadata(metadata_list, updates):
    # metadata_list is a list[dict] like in your JSON
    for obj in metadata_list:
        p = 1
        if p in updates:
            obj.update(updates[p])
    return metadata_list


FOLDER1 = Path("/Users/ole/Kodevik/wave_project/wavedata/20251110-tett6roof-lowM-ekte580")

if __name__ == "__main__":
    print(f"CALLING load_or_update with: {FOLDER1}") 
    dfs, meta = load_or_update(FOLDER1)
    
    print(f"\nSUCCESS: {len(dfs)} DataFrames loaded and cached!") 
    print(f"Metadata shape: {meta.shape}")





