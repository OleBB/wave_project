#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved data_loader.py with better probe position management.

KEY IMPROVEMENTS:
1. Centralized probe position configuration (no more hardcoded dates in logic)
2. Easy to add new configurations as experiments evolve
3. Clear documentation of when changes were made
4. Automatic validation of probe positions
"""

from pathlib import Path
from typing import Iterator, Dict, Tuple, List, Optional
import json
import re
import pandas as pd
import os
from datetime import datetime
from dataclasses import dataclass
# import pyarrow


from wavescripts.constants import MEASUREMENT

# Only list the exceptions (non-floats)
NON_FLOAT_COLUMNS = {
    "WindCondition": str,
    "TunnelCondition": str,
    "PanelCondition": str,
    "Mooring": str,
    "Run number": str,
    "experiment_folder": str,
    "path": str,
    "file_date": str,
}

def apply_dtypes(meta_df: pd.DataFrame) -> pd.DataFrame:
    """Everything is float64 except NON_FLOAT_COLUMNS."""
    meta_df = meta_df.copy()
    for col in meta_df.columns:
        if col in NON_FLOAT_COLUMNS:
            meta_df[col] = meta_df[col].astype(NON_FLOAT_COLUMNS[col])
        else:
            meta_df[col] = pd.to_numeric(meta_df[col], errors='coerce')
    return meta_df

# =============================================================================
# PROBE POSITION CONFIGURATION
# =============================================================================

@dataclass
class ProbeConfiguration:
    """A set of probe positions valid for a date range."""
    name: str
    valid_from: datetime  # inclusive
    valid_until: Optional[datetime]  # exclusive, None = forever
    distances_mm: Dict[int, float]  # probe_num -> mm from paddle
    notes: str = ""

# Define all probe configurations chronologically
PROBE_CONFIGS = [
    ProbeConfiguration(
        name="initial_setup",
        valid_from=datetime(2025, 1, 1),  # start of experiment
        valid_until=datetime(2025, 11, 14),
        distances_mm={
            1: 18000, #langt bak en plass
            2: 9455.0,
            3: 12544.0,
            4: 12545.0,
        },
        notes="Utprøvet configuration used until Nov 13, 2025"
    ),
    ProbeConfiguration(
        name="nov14_normalt_oppsett",
        valid_from=datetime(2025, 11, 14),
        valid_until=None,  # current configuration
        distances_mm={
            1: 8855.0,  # TODO: Update with actual new positions
            2: 9455.0,
            3: 12544.0,
            4: 12545.0,
        },
        notes="Configuration after Nov 14 adjustment"
    ),
    
    # ADD NEW CONFIGURATIONS HERE:
    # ProbeConfiguration(
    #     name="feb2026_recalibration",
    #     valid_from=datetime(2026, 2, 1),
    #     valid_until=None,
    #     distances_mm={
    #         1: 8900.0,  # example new values
    #         2: 9500.0,
    #         3: 12600.0,
    #         4: 12601.0,
    #     },
    #     notes="Recalibrated efter TK TODO fYLL UT"
    # ),
]


def get_probe_positions(file_date: datetime) -> Dict[int, float]:
    """
    Get probe positions for a given date.
    
    Args:
        file_date: Date of the measurement file
    
    Returns:
        Dictionary mapping probe number (1-4) to distance from paddle in mm
    
    Raises:
        ValueError: If no valid configuration found for date
    """
    for config in PROBE_CONFIGS:
        if config.valid_from <= file_date:
            if config.valid_until is None or file_date < config.valid_until:
                return config.distances_mm.copy()
    
    raise ValueError(
        f"No probe configuration found for date {file_date}. "
        f"Check PROBE_CONFIGS in data_loader.py"
    )


def validate_probe_configs():
    """
    Validate that probe configurations don't have gaps or overlaps.
    Call this at module load time.
    """
    if not PROBE_CONFIGS:
        raise ValueError("PROBE_CONFIGS is empty!")
    
    # Sort by valid_from
    sorted_configs = sorted(PROBE_CONFIGS, key=lambda c: c.valid_from)
    
    # Check for gaps
    for i in range(len(sorted_configs) - 1):
        current = sorted_configs[i]
        next_config = sorted_configs[i + 1]
        
        if current.valid_until is None:
            raise ValueError(
                f"Config '{current.name}' has no end date but is not the last config"
            )
        
        if current.valid_until != next_config.valid_from:
            raise ValueError(
                f"Gap or overlap between '{current.name}' and '{next_config.name}': "
                f"{current.valid_until} != {next_config.valid_from}"
            )
    
    # Check last config
    if sorted_configs[-1].valid_until is not None:
        print(
            f"Warning: Last probe config '{sorted_configs[-1].name}' has an end date. "
            f"Consider setting valid_until=None for the current configuration."
        )
    
    print(f"✓ Validated {len(PROBE_CONFIGS)} probe configurations")


# Validate on import
validate_probe_configs()


# =============================================================================
# MOORING CONFIGURATION (similar pattern)
# =============================================================================

@dataclass
class MooringConfiguration:
    """Mooring settings for a date range."""
    name: str
    valid_from: datetime
    valid_until: Optional[datetime]
    mooring_type: str  # "high" or "low"
    notes: str = ""

MOORING_CONFIGS = [
    MooringConfiguration(
        name="initial_high_mooring",
        valid_from=datetime(2025, 1, 1),
        valid_until=datetime(2025, 11, 6),
        mooring_type="high",
        notes="tidlig forsøk high mooring setup"
    ),
    MooringConfiguration(
        name="low_mooring",
        valid_from=datetime(2025, 11, 6),
        valid_until=None,
        mooring_type="low",
        notes="Switched to low mooring on Nov 6 - høyde omtrent = x millimeter over vannet mmov "
    ),
]


def get_mooring_type(file_date: datetime) -> str:
    """Get mooring type for a given date."""
    for config in MOORING_CONFIGS:
        if config.valid_from <= file_date:
            if config.valid_until is None or file_date < config.valid_until:
                return config.mooring_type
    
    return "unknown"


# =============================================================================
# IMPROVED METADATA EXTRACTION
# =============================================================================

def extract_metadata_from_filename(
    filename: str,
    file_path: Path,
    df: pd.DataFrame,
    experiment_name: str
) -> dict:
    """
    Extract metadata from filename with improved date handling.
    
    This version:
    - Uses configuration system for probe positions
    - Cleaner logic for date-dependent values
    - Better error handling
    - More maintainable
    """
    # Initialize metadata structure
    metadata = _initialize_metadata_dict(str(file_path), experiment_name)
    
    # Get file date (prefer filename date over modification time)
    file_date = _extract_file_date(filename, file_path)
    metadata["file_date"] = file_date.isoformat() if file_date else None
    
    # ─────── Extract basic info from filename ───────
    _extract_wind_condition(metadata, filename, df)
    _extract_tunnel_condition(metadata, filename)
    _extract_panel_condition(metadata, filename)
    _extract_wave_parameters(metadata, filename)
    
    # ─────── Date-dependent configurations ───────
    if file_date:
        try:
            # Probe positions (uses configuration system)
            probe_positions = get_probe_positions(file_date)
            for probe_num, distance in probe_positions.items():
                metadata[f"Probe {probe_num} mm from paddle"] = distance
        except ValueError as e:
            print(f"   Warning: {e}")
            # Set to None if no config found
            for probe_num in range(1, 5):
                metadata[f"Probe {probe_num} mm from paddle"] = None
        
        # Mooring type (uses configuration system)
        metadata["Mooring"] = get_mooring_type(file_date)
    else:
        # No date available - set to None
        for probe_num in range(1, 5):
            metadata[f"Probe {probe_num} mm from paddle"] = None
        metadata["Mooring"] = "unknown"
    
    return metadata


def _initialize_metadata_dict(file_path: str, experiment_name: str) -> dict:
    """Initialize metadata dictionary with all expected fields."""
    metadata = {
        "path": file_path,
        "experiment_folder": experiment_name,
        "file_date": None,
        "WindCondition": "",
        "TunnelCondition": "",
        "PanelCondition": "",
        "Mooring": "",
        "WaveAmplitudeInput [Volt]": None,
        "WaveFrequencyInput [Hz]": None,
        "WavePeriodInput": None,
        "WaterDepth [mm]": None,
        "Extra seconds": None,
        "Run number": "",
    }
    
    # Add probe-related fields
    for i in range(1, 5):
        metadata[f"Probe {i} mm from paddle"] = None
        metadata[f"Stillwater Probe {i}"] = None
        metadata[f"Computed Probe {i} start"] = None
        metadata[f"Computed Probe {i} end"] = None
        metadata[f"Probe {i} Amplitude"] = None
        metadata[f"Probe {i} Amplitude (PSD)"] = None
        metadata[f"Probe {i} Amplitude (FFT)"] = None
        metadata[f"Probe {i} Swell Amplitude (FFT)"] = None
        metadata[f"Probe {i} Wind Amplitude (FFT)"] = None
        metadata[f"Probe {i} Total Amplitude (FFT)"] = None
        
        metadata[f"Probe {i} WavePeriod (FFT)"] = None #new
        metadata[f"Probe {i} Wavenumber (FFT)"] = None #new
        metadata[f"Probe {i} Wavelength (FFT)"] = None #new
        metadata[f"Probe {i} Significant Wave Height Hm0 (FFT)"] = None #new
        
    
    # Add computed fields
    for field in ["Wavefrequency", "Waveperiod", "Wavenumber", "Wavelength",
                  "kL", "ak", "kH", "tanh(kH)", "Celerity",
                  "Significant Wave Height Hs", "Significant Wave Height Hm0",
                  "Windspeed", "P2/P1", "P3/P2", "P4/P3"]:
        metadata[field] = None
    
    return metadata


def _extract_file_date(filename: str, file_path: Path) -> Optional[datetime]:
    """
    Extract date from filename, fall back to file modification time.
    
    Priority:
    1. Date in filename (YYYYMMDD format)
    2. File modification time
    """
    # Try to extract from filename first
    date_match = re.search(r'(\d{8})', filename)
    if date_match:
        date_str = date_match.group(1)
        try:
            return datetime.strptime(date_str, "%Y%m%d")
        except ValueError:
            pass
    
    # Fall back to file modification time
    try:
        modtime = os.path.getmtime(file_path)
        return datetime.fromtimestamp(modtime)
    except Exception:
        return None


def _extract_wind_condition(metadata: dict, filename: str, df: pd.DataFrame):
    """Extract wind condition and compute stillwater if no wind."""
    wind_match = re.search(r'-([A-Za-z]+)wind-', filename)
    if wind_match:
        metadata["WindCondition"] = wind_match.group(1)
        
        # Compute stillwater for no-wind runs
        if wind_match.group(1).lower() == "no":
            stillwater_samples = MEASUREMENT.STILLWATER_SAMPLES
            for p in range(1, 5):
                probe_col = f"Probe {p}"
                if probe_col in df.columns:
                    metadata[f"Stillwater Probe {p}"] = df[probe_col].iloc[:stillwater_samples].mean(skipna=True)


def _extract_tunnel_condition(metadata: dict, filename: str):
    """Extract tunnel condition from filename."""
    tunnel_match = re.search(r'([0-9])roof', filename)
    if tunnel_match:
        metadata["TunnelCondition"] = tunnel_match.group(1) + " roof plates"


def _extract_panel_condition(metadata: dict, filename: str):
    """Extract panel condition from filename."""
    panel_match = re.search(r'([A-Za-z]+)panel', filename)
    if panel_match:
        metadata["PanelCondition"] = panel_match.group(1)


def _extract_wave_parameters(metadata: dict, filename: str):
    """Extract wave parameters from filename."""
    if m := re.search(r'-amp([A-Za-z0-9]+)-', filename):
        metadata["WaveAmplitudeInput [Volt]"] = int(m.group(1)) * MEASUREMENT.MM_TO_M
    
    if m := re.search(r'-freq(\d+)-', filename):
        metadata["WaveFrequencyInput [Hz]"] = int(m.group(1)) * MEASUREMENT.MM_TO_M
    
    if m := re.search(r'-per(\d+)-', filename):
        metadata["WavePeriodInput"] = int(m.group(1))
    
    if m := re.search(r'-depth([A-Za-z0-9]+)', filename):
        metadata["WaterDepth [mm]"] = int(m.group(1))
    
    if m := re.search(r'-mstop([A-Za-z0-9]+)', filename):
        metadata["Extra seconds"] = int(m.group(1))
    
    if m := re.search(r'-run([0-9])', filename, re.IGNORECASE):
        metadata["Run number"] = m.group(1)


# =============================================================================
# CONFIGURATION MANAGEMENT UTILITIES
# =============================================================================

def print_probe_configuration_history():
    """Print a timeline of all probe configurations."""
    print("\n" + "="*70)
    print("PROBE CONFIGURATION HISTORY")
    print("="*70)
    
    for i, config in enumerate(sorted(PROBE_CONFIGS, key=lambda c: c.valid_from), 1):
        print(f"\n{i}. {config.name.upper()}")
        print(f"   Valid: {config.valid_from.date()} → ", end="")
        print("present" if config.valid_until is None else config.valid_until.date())
        print(f"   Positions (mm from paddle):")
        for probe_num in range(1, 5):
            print(f"      Probe {probe_num}: {config.distances_mm[probe_num]:,.1f} mm")
        if config.notes:
            print(f"   Notes: {config.notes}")
    
    print("\n" + "="*70)


def get_configuration_for_date(target_date: datetime) -> ProbeConfiguration:
    """Get the full configuration object for a specific date."""
    for config in PROBE_CONFIGS:
        if config.valid_from <= target_date:
            if config.valid_until is None or target_date < config.valid_until:
                return config
    
    raise ValueError(f"No configuration found for {target_date}")


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

def load_or_update(
    *folders: Path | str,
    force_recompute: bool = False,
    total_reset: bool = False
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    For each input folder (e.g. wavedata/20251110-tett6roof-lowM-ekte580),
    automatically uses or creates:
        waveprocessed/PROCESSED-20251110-tett6roof-lowM-ekte580/
    containing dfs.parquet (cached DataFrames) and meta.json (metadata)
    
    Args:
        folders: One or more paths to raw data folders
        force_recompute: If True, recompute metadata from cached DataFrames (skip loading CSVs)
        total_reset: If True, ignore all cache and reload everything from CSV files
    """
    # Find project root
    current_file = Path(__file__).resolve()
    project_root = next(
        (p for p in current_file.parents if (p / "main.py").exists() or (p / ".git").exists()),
        current_file.parent.parent
    )

    # Define base folder for all processed experiments
    processed_root = Path(project_root / "waveprocessed")
    processed_root.mkdir(parents=True, exist_ok=True)

    # Global containers
    all_dfs: Dict[str, pd.DataFrame] = {}
    all_meta_list: List[dict] = []

    # Process each folder independently
    for folder in folders:
        folder_path = Path(folder).resolve()
        if not folder_path.is_dir():
            print(f"Warning: Skipping missing folder: {folder_path}")
            continue

        experiment_name = folder_path.name
        cache_dir = processed_root / f"PROCESSED-{experiment_name}"
        cache_dir.mkdir(parents=True, exist_ok=True)

        dfs_cache_path = cache_dir / "dfs.parquet"
        meta_path = cache_dir / "meta.json"

        print(f"\nProcessing experiment: {experiment_name}")
        print(f"   Cache folder: {cache_dir.relative_to(project_root)}")

        # Initialize containers
        dfs: Dict[str, pd.DataFrame] = {}
        meta_list: list[dict] = []

        # ============================================================
        # Step 1: Load or rebuild DataFrame cache (dfs.parquet)
        # ============================================================
        if total_reset:
            print(f"   Total reset: Reloading all CSVs from scratch")
            # Load all CSVs fresh
            csv_files = list(get_data_files(folder_path))
            dfs = _load_csv_files(csv_files, experiment_name)
            
        elif dfs_cache_path.exists():
            print(f"   Loading cached DataFrames from {dfs_cache_path.name}")
            try:
                # Load from parquet - reconstruct the dict
                cached_df = pd.read_parquet(dfs_cache_path)
                # Parquet stores as single DF with 'path' column, need to split back
                for path in cached_df['_path'].unique():
                    dfs[path] = cached_df[cached_df['_path'] == path].drop(columns=['_path'])
                print(f"   Loaded {len(dfs)} cached DataFrames")
                
                # Check for new CSV files not in cache
                seen_keys = set(dfs.keys())
                new_csv_files = [
                    p for p in get_data_files(folder_path)
                    if str(p.resolve()) not in seen_keys
                ]
                
                if new_csv_files:
                    print(f"   Found {len(new_csv_files)} new CSV file(s), loading...")
                    new_dfs = _load_csv_files(new_csv_files, experiment_name)
                    dfs.update(new_dfs)
                    
            except Exception as e:
                print(f"   Cache corrupted ({e}) → rebuilding from CSVs")
                csv_files = list(get_data_files(folder_path))
                dfs = _load_csv_files(csv_files, experiment_name)
        else:
            print(f"   No cache found, loading all CSVs")
            csv_files = list(get_data_files(folder_path))
            dfs = _load_csv_files(csv_files, experiment_name)

        # ============================================================
        # Step 2: Compute or recompute metadata (meta.json)
        # ============================================================
        if force_recompute or total_reset:
            mode = "total reset" if total_reset else "force recompute"
            print(f"   {mode.capitalize()}: Extracting metadata from {len(dfs)} DataFrames")
            # Extract metadata from all loaded DataFrames
            for path, df in dfs.items():
                filename = Path(path).name
                metadata = extract_metadata_from_filename(filename, Path(path), df, experiment_name)
                meta_list.append(metadata)
                
        elif meta_path.exists():
            print(f"   Loading existing metadata from {meta_path.name}")
            try:
                meta_list = json.loads(meta_path.read_text(encoding="utf-8"))
                print(f"   Loaded metadata for {len(meta_list)} files")
                
                # Check if we have new DataFrames that need metadata
                existing_paths = {m['path'] for m in meta_list}
                new_paths = set(dfs.keys()) - existing_paths
                
                if new_paths:
                    print(f"   Extracting metadata for {len(new_paths)} new file(s)")
                    for path in new_paths:
                        filename = Path(path).name
                        metadata = extract_metadata_from_filename(
                            filename, Path(path), dfs[path], experiment_name
                        )
                        meta_list.append(metadata)
                        
            except Exception as e:
                print(f"   Metadata corrupted ({e}) → rebuilding")
                for path, df in dfs.items():
                    filename = Path(path).name
                    metadata = extract_metadata_from_filename(filename, Path(path), df, experiment_name)
                    meta_list.append(metadata)
        else:
            print(f"   No metadata found, extracting from {len(dfs)} DataFrames")
            for path, df in dfs.items():
                filename = Path(path).name
                metadata = extract_metadata_from_filename(filename, Path(path), df, experiment_name)
                meta_list.append(metadata)

        # ============================================================
        # Step 3: Save cache
        # ============================================================
        # Save DataFrame cache to parquet (if new/updated)
        if dfs and (total_reset or not dfs_cache_path.exists() or len(dfs) != len(meta_list)):
            print(f"   Saving DataFrame cache to {dfs_cache_path.name}")
            # Combine all dfs into single DataFrame for parquet
            combined_list = []
            for path, df in dfs.items():
                df_copy = df.copy()
                df_copy['_path'] = path  # Add path identifier
                combined_list.append(df_copy)
            combined_df = pd.concat(combined_list, ignore_index=True)
            combined_df.to_parquet(dfs_cache_path, index=False, engine='pyarrow')
        
        # Save metadata to JSON
        if meta_list:
            meta_path.write_text(json.dumps(meta_list, indent=2, default=str), encoding="utf-8")
            print(f"   Saved metadata for {len(meta_list)} files")

        # Merge into global result
        all_dfs.update(dfs)
        all_meta_list.extend(meta_list)

    # Final metadata DataFrame
    meta_df = pd.DataFrame(all_meta_list)
    meta_df = apply_dtypes(meta_df)
    
    mode = "reset" if total_reset else ("recomputed metadata" if force_recompute else "loaded")
    print(f"\nFinished! Total {len(all_dfs)} files ({mode}) from {len(folders)} experiment(s)")
    return all_dfs, meta_df


def _load_csv_files(
    csv_files: List[Path], 
    experiment_name: str
) -> Dict[str, pd.DataFrame]:
    """Helper function to load CSV files and return dict of DataFrames."""
    dfs = {}
    
    for i, path in enumerate(csv_files, 1):
        key = str(path.resolve())
        try:
            suffix = path.suffix.lower()
            if suffix == ".csv":
                df = pd.read_csv(
                    path, 
                    engine='pyarrow',
                    names=["Date", "Probe 1", "Probe 2", "Probe 3", "Probe 4", "Mach"]
                )
                
                # Formatting
                for probe in range(1, 5):
                    df[f"Probe {probe}"] *= MEASUREMENT.M_TO_MM
                df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y %H:%M:%S.%f")
                
                dfs[key] = df
                print(f"   [{i}/{len(csv_files)}] Loaded {path.name} → {len(df):,} rows")
            else:
                print(f"   Skipping unsupported: {path.name}")
                
        except Exception as e:
            print(f"   Failed {path.name}: {e}")
    
    return dfs

# --------------------------------------------------
# Takes in a modified meta-dataframe, and updates the meta.JSON and meta excel
# --------------------------------------------------
def update_processed_metadata(
    meta_df: pd.DataFrame,
    force_recompute: bool = False, 
) -> None:
    """
    Safely updates meta.json files:
      Keeps existing runs
      Adds new runs
      Updates changed rows (matched by 'path')
      Never overwrites or deletes data unless forced
      Lagrer til meta json og meta xlsx
    """
    current_file = Path(__file__).resolve()
    project_root = next(
        (p for p in current_file.parents if (p / "main.py").exists() or (p / ".git").exists()),
        current_file.parent.parent
    )
    # def ...     processed_root: Path | str | None = None, fjernet
    processed_root = Path( project_root / "waveprocessed") #processed_root or
    
    # Ensure we have a way to group by experiment
    meta_df = meta_df.copy()
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
        excel_path = cache_dir / "meta.xlsx"

        # Load existing metadata if file exists
        if meta_path.exists() and not force_recompute:
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    old_records = json.load(f)
                old_df = pd.DataFrame(old_records)
                old_df = apply_dtypes(old_df)
                
                print(f"Loaded {len(old_df)} existing entries from {meta_path.name}")
            except Exception as e:
                print(f"Could not read existing {meta_path} → starting fresh: {e}")
                old_df = pd.DataFrame()
        else:
            old_df = pd.DataFrame()
            cache_dir.mkdir(parents=True, exist_ok=True)

        # Clean incoming data
        new_df = group_df.drop(columns=["__group_key"], errors="ignore").copy()
        new_df["path"] = new_df["path"].astype(str)
        
        # Ensure old_df path is also string
        if not old_df.empty:
            old_df["path"] = old_df["path"].astype(str)
        
        # Merge logic
        if old_df.empty:
            final_df = new_df
        elif new_df.empty:
            final_df = old_df
        else:
            combined = pd.concat([old_df, new_df], ignore_index=True)
            final_df = combined.drop_duplicates(subset="path", keep="last")

        # Save back safely
        records = final_df.to_dict("records")
        temp_path = meta_path.with_suffix(".json.tmp")
        temp_path.write_text(
            json.dumps(records, indent=2, default=str),
            encoding="utf-8")
        temp_path.replace(meta_path) #atomic
        
        final_df.to_excel(excel_path, index=False)
        
        added = len(final_df) - len(old_df) if not old_df.empty else len(final_df)
        print(f"Updated {meta_path.relative_to(project_root)} → {len(final_df)} entries (+{added} new)")
    
    print(f"\nMetadata safely updated and preserved across {meta_df['__group_key'].nunique()} experiment(s)!")




# =============================================================================
# TESTING
# =============================================================================
if __name__ == "__main__":
    # Test with sample data
    sample_data = [
        {
            "path": "/path/to/file.csv",
            "WindCondition": "full",
            "PanelCondition": "yellow",
            "WaveFrequencyInput [Hz]": "1.3",  # String that should become float
            "Probe 1 Amplitude": "10.5",
            "Probe 2 Amplitude": None,  # Should become NaN
            "kL": "2.5",
            "experiment_folder": "20251110-test",
        }
    ]
    
    df = pd.DataFrame(sample_data)
    
    print("BEFORE apply_dtypes:")
    print(df.dtypes)
    print()
    
    df = apply_dtypes(df)
    
    print("AFTER apply_dtypes:")
    print(df.dtypes)
    print()
    print("Values:")
    print(df)
    
    # Verify types
    assert df["WindCondition"].dtype == object  # string
    assert df["WaveFrequencyInput [Hz]"].dtype == "float64"
    assert df["Probe 1 Amplitude"].dtype == "float64"
    assert pd.isna(df["Probe 2 Amplitude"].iloc[0])  # None → NaN
    
    print("\n✓ All type conversions working correctly!")


# if __name__ == "__main__":
#     print("Testing probe position system...")
    
#     # Print configuration history
#     print_probe_configuration_history()
    
#     # Test getting positions for different dates
#     test_dates = [
#         datetime(2025, 11, 1),   # Before Nov 14
#         datetime(2025, 11, 14),  # On Nov 14
#         datetime(2025, 11, 20),  # After Nov 14
#         datetime(2026, 1, 15),   # Future date
#     ]
    
#     print("\n" + "="*70)
#     print("TESTING DATE LOOKUPS")
#     print("="*70)
    
#     for test_date in test_dates:
#         print(f"\nDate: {test_date.date()}")
#         try:
#             positions = get_probe_positions(test_date)
#             config = get_configuration_for_date(test_date)
#             print(f"  Config: {config.name}")
#             print(f"  Mooring: {get_mooring_type(test_date)}")
#             print(f"  Probe 1: {positions[1]:,.1f} mm")
#         except ValueError as e:
#             print(f"  ERROR: {e}")
            
            
