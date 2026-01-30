"""
CLEANUP GUIDE for your data_loader.py

Shows what to DELETE and how to fix datatypes
"""

# =============================================================================
# ISSUE 1: Duplicate/redundant metadata initialization
# =============================================================================

# ────────────────────────────────────────────────────────────
# CURRENT CODE (lines ~480-550) - DELETE THIS ENTIRE BLOCK:
# ────────────────────────────────────────────────────────────

"""
                # ------------------- METADATA EXTRACTION -------------------
                filename = path.name
                metadata = {
                    "path": key,
                    "WindCondition": str,
                    "TunnelCondition": str,
                    "PanelCondition": str,
                    # ... 40 more lines of field: type mappings ...
                }
                metadata = {k: "" if dtype is str else None for k, dtype in metadata.items()}
                metadata.update({
                    "path": key,
                    "experiment_folder": experiment_name
                })
"""

# ────────────────────────────────────────────────────────────
# REPLACE WITH (use your helper function):
# ────────────────────────────────────────────────────────────

"""
                # ------------------- METADATA EXTRACTION -------------------
                filename = path.name
                metadata = extract_metadata_from_filename(filename, path, df, experiment_name)
"""

# That's it! The function already does everything properly.


# =============================================================================
# ISSUE 2: Missing import at top of file
# =============================================================================

# ────────────────────────────────────────────────────────────
# CURRENT (line ~17):
# ────────────────────────────────────────────────────────────
"""
import dataclass  # WRONG - this doesn't exist!
"""

# ────────────────────────────────────────────────────────────
# REPLACE WITH:
# ────────────────────────────────────────────────────────────
"""
from dataclasses import dataclass
"""


# =============================================================================
# ISSUE 3: dtype_map needs updating for new fields
# =============================================================================

# Add these new fields to your dtype_map at the top of the file:

dtype_map_additions = {
    # New FFT-based fields (add these)
    "Probe 1 WavePeriod (FFT)": "float64",
    "Probe 1 Wavenumber (FFT)": "float64",
    "Probe 1 Wavelength (FFT)": "float64",
    "Probe 1 Significant Wave Height Hm0 (FFT)": "float64",
    
    "Probe 2 WavePeriod (FFT)": "float64",
    "Probe 2 Wavenumber (FFT)": "float64",
    "Probe 2 Wavelength (FFT)": "float64",
    "Probe 2 Significant Wave Height Hm0 (FFT)": "float64",
    
    "Probe 3 WavePeriod (FFT)": "float64",
    "Probe 3 Wavenumber (FFT)": "float64",
    "Probe 3 Wavelength (FFT)": "float64",
    "Probe 3 Significant Wave Height Hm0 (FFT)": "float64",
    
    "Probe 4 WavePeriod (FFT)": "float64",
    "Probe 4 Wavenumber (FFT)": "float64",
    "Probe 4 Wavelength (FFT)": "float64",
    "Probe 4 Significant Wave Height Hm0 (FFT)": "float64",
    
    # New metadata fields
    "file_date": str,  # ISO format string
    "Probe 1 mm from paddle": "float64",
    "Probe 2 mm from paddle": "float64",
    "Probe 3 mm from paddle": "float64",
    "Probe 4 mm from paddle": "float64",
}


# =============================================================================
# ISSUE 4: Redundant code in load_or_update
# =============================================================================

# ────────────────────────────────────────────────────────────
# DELETE THESE LINES (they're now in extract_metadata_from_filename):
# ────────────────────────────────────────────────────────────

"""
                stillwater_samples = MEASUREMENT.STILLWATER_SAMPLES
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

                # Mooring logic (using configuration system)
                file_date = _extract_file_date(filename, path)
                if file_date:
                    metadata["Mooring"] = get_mooring_type(file_date)
                else:
                    metadata["Mooring"] = "unknown"
                
                # Probe distance logic (using configuration system)
                file_date = _extract_file_date(filename, path)
                if file_date:
                    try:
                        probe_positions = get_probe_positions(file_date)
                        for probe_num, distance in probe_positions.items():
                            metadata[f"Probe {probe_num} mm from paddle"] = distance
                    except ValueError as e:
                        print(f"   Warning: No probe config for {path.name}: {e}")
                        for probe_num in range(1, 5):
                            metadata[f"Probe {probe_num} mm from paddle"] = None
                else:
                    for probe_num in range(1, 5):
                        metadata[f"Probe {probe_num} mm from paddle"] = None
                    

                # Wave parameters
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
"""

# All this is already handled by extract_metadata_from_filename()!


# =============================================================================
# ISSUE 5: Missing STILLWATER_SAMPLES in constants.py
# =============================================================================

# Add this to your wavescripts/constants.py:

"""
@dataclass(frozen=True)
class MeasurementConstants:
    SAMPLING_RATE: float = 250.0
    NUM_PROBES: int = 4
    PROBE_NAMES: tuple = ("Probe 1", "Probe 2", "Probe 3", "Probe 4")
    MM_TO_M: float = 0.001
    M_TO_MM: float = 1000.0
    
    # NEW: Add this
    STILLWATER_SAMPLES: int = 500  # 2 seconds at 250 Hz
"""


# =============================================================================
# CLEAN VERSION: What load_or_update should look like
# =============================================================================

def load_or_update_CLEANED(
    *folders: Path | str,
    force_recompute: bool = False,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """Clean version with no redundant code."""
    
    # ... (project root finding code - keep as is)
    
    for folder in folders:
        # ... (setup code - keep as is)
        
        for i, path in enumerate(new_files, 1):
            key = str(path.resolve())
            try:
                # Load CSV
                suffix = path.suffix.lower()
                if suffix == ".csv":
                    df = pd.read_csv(path, names=["Date", "Probe 1", "Probe 2", "Probe 3", "Probe 4", "Mach"])
                else:
                    print(f"   Skipping unsupported: {path.name}")
                    continue

                # Format data
                for probe in range(1, 5):
                    df[f"Probe {probe}"] *= MEASUREMENT.M_TO_MM
                df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y %H:%M:%S.%f")

                dfs[key] = df

                # Extract metadata - THIS IS THE ONLY LINE YOU NEED!
                filename = path.name
                metadata = extract_metadata_from_filename(filename, path, df, experiment_name)
                
                meta_list.append(metadata)
                print(f"   [{i}/{len(new_files)}] Loaded {path.name} → {len(df):,} rows")

            except Exception as e:
                print(f"   Failed {path.name}: {e}")
        
        # ... (rest of function - keep as is)


# =============================================================================
# ENSURING CORRECT DATATYPES
# =============================================================================

"""
PANDAS DATATYPE ISSUES AND SOLUTIONS:

PROBLEM 1: Numbers stored as strings
---------------------------------
Symptom: meta_df["WaveFrequencyInput [Hz]"] contains "0.13" instead of 0.13
Cause: Wrong conversion in regex extraction

FIX in _extract_wave_parameters():
"""

def _extract_wave_parameters_FIXED(metadata: dict, filename: str):
    """Fixed version with correct type conversions."""
    
    if m := re.search(r'-amp([A-Za-z0-9]+)-', filename):
        # Convert to int first, then multiply
        metadata["WaveAmplitudeInput [Volt]"] = float(int(m.group(1)) * MEASUREMENT.MM_TO_M)
    
    if m := re.search(r'-freq(\d+)-', filename):
        # This looks wrong: int * MM_TO_M (0.001)?
        # Should probably be: float(int(m.group(1)) / 100)  if freq is like "130" for 1.3 Hz
        # OR just: float(int(m.group(1)))  if freq is already correct
        metadata["WaveFrequencyInput [Hz]"] = float(int(m.group(1)) / 100)  # Assuming "130" → 1.3
    
    if m := re.search(r'-per(\d+)-', filename):
        metadata["WavePeriodInput"] = int(m.group(1))  # Keep as int
    
    if m := re.search(r'-depth([A-Za-z0-9]+)', filename):
        metadata["WaterDepth [mm]"] = float(int(m.group(1)))
    
    if m := re.search(r'-mstop([A-Za-z0-9]+)', filename):
        metadata["Extra seconds"] = float(int(m.group(1)))
    
    if m := re.search(r'-run([0-9])', filename, re.IGNORECASE):
        metadata["Run number"] = str(m.group(1))  # Keep as string


"""
PROBLEM 2: Inconsistent None vs NaN vs ""
---------------------------------------
Your current code initializes with None, but pandas prefers:
- np.nan for numeric columns
- "" or None for string columns

BETTER: Let pandas handle it with dtype_map
"""

# At the END of load_or_update, before returning:
def load_or_update_final_step():
    # ... (all previous code)
    
    # Create DataFrame
    meta_df = pd.DataFrame(all_meta_list)
    
    # IMPORTANT: Apply dtype_map to enforce types
    # This converts None → NaN for floats, handles strings correctly
    for col, dtype in dtype_map.items():
        if col in meta_df.columns:
            try:
                meta_df[col] = meta_df[col].astype(dtype)
            except (ValueError, TypeError) as e:
                print(f"Warning: Could not convert {col} to {dtype}: {e}")
    
    return all_dfs, meta_df


"""
PROBLEM 3: dtype_map not comprehensive
------------------------------------
You're missing some columns. Here's a helper to find them:
"""

def validate_dtype_map(meta_df: pd.DataFrame):
    """Check which columns are missing from dtype_map."""
    missing = set(meta_df.columns) - set(dtype_map.keys())
    if missing:
        print(f"⚠ Warning: These columns not in dtype_map: {missing}")
        print("Add them to dtype_map for consistent types!")


# =============================================================================
# COMPLETE UPDATED dtype_map
# =============================================================================

dtype_map_COMPLETE = {
    # Experimental conditions
    "WindCondition": str,
    "TunnelCondition": str,
    "PanelCondition": str,
    "Mooring": str,
    
    # Input parameters
    "WaveAmplitudeInput [Volt]": "float64",
    "WaveFrequencyInput [Hz]": "float64",
    "WavePeriodInput": "Int64",  # Use Int64 (nullable int) or "float64"
    "WaterDepth [mm]": "float64",
    "Extra seconds": "float64",
    "Run number": str,
    
    # Probe positions
    "Probe 1 mm from paddle": "float64",
    "Probe 2 mm from paddle": "float64",
    "Probe 3 mm from paddle": "float64",
    "Probe 4 mm from paddle": "float64",
    
    # Stillwater levels
    "Stillwater Probe 1": "float64",
    "Stillwater Probe 2": "float64",
    "Stillwater Probe 3": "float64",
    "Stillwater Probe 4": "float64",
    
    # Computed wave ranges
    "Computed Probe 1 start": "float64",
    "Computed Probe 2 start": "float64",
    "Computed Probe 3 start": "float64",
    "Computed Probe 4 start": "float64",
    "Computed Probe 1 end": "float64",
    "Computed Probe 2 end": "float64",
    "Computed Probe 3 end": "float64",
    "Computed Probe 4 end": "float64",
    
    # Amplitudes (percentile-based)
    "Probe 1 Amplitude": "float64",
    "Probe 2 Amplitude": "float64",
    "Probe 3 Amplitude": "float64",
    "Probe 4 Amplitude": "float64",
    
    # Amplitudes (PSD-based)
    "Probe 1 Amplitude (PSD)": "float64",
    "Probe 2 Amplitude (PSD)": "float64",
    "Probe 3 Amplitude (PSD)": "float64",
    "Probe 4 Amplitude (PSD)": "float64",
    
    # Amplitudes (FFT-based)
    "Probe 1 Amplitude (FFT)": "float64",
    "Probe 2 Amplitude (FFT)": "float64",
    "Probe 3 Amplitude (FFT)": "float64",
    "Probe 4 Amplitude (FFT)": "float64",
    
    # NEW: FFT-derived metrics
    "Probe 1 WavePeriod (FFT)": "float64",
    "Probe 1 Wavenumber (FFT)": "float64",
    "Probe 1 Wavelength (FFT)": "float64",
    "Probe 1 Significant Wave Height Hm0 (FFT)": "float64",
    
    "Probe 2 WavePeriod (FFT)": "float64",
    "Probe 2 Wavenumber (FFT)": "float64",
    "Probe 2 Wavelength (FFT)": "float64",
    "Probe 2 Significant Wave Height Hm0 (FFT)": "float64",
    
    "Probe 3 WavePeriod (FFT)": "float64",
    "Probe 3 Wavenumber (FFT)": "float64",
    "Probe 3 Wavelength (FFT)": "float64",
    "Probe 3 Significant Wave Height Hm0 (FFT)": "float64",
    
    "Probe 4 WavePeriod (FFT)": "float64",
    "Probe 4 Wavenumber (FFT)": "float64",
    "Probe 4 Wavelength (FFT)": "float64",
    "Probe 4 Significant Wave Height Hm0 (FFT)": "float64",
    
    # Wave properties
    "Wavefrequency": "float64",
    "Waveperiod": "float64",
    "Wavenumber": "float64",
    "Wavelength": "float64",
    "kL": "float64",
    "ak": "float64",
    "kH": "float64",
    "tanh(kH)": "float64",
    "Celerity": "float64",
    
    # Wave statistics
    "Significant Wave Height Hs": "float64",
    "Significant Wave Height Hm0": "float64",
    
    # Environment
    "Windspeed": "float64",
    
    # Probe ratios
    "P2/P1": "float64",
    "P3/P2": "float64",
    "P4/P3": "float64",
    
    # Metadata
    "experiment_folder": str,
    "path": str,
    "file_date": str,  # ISO format datetime string
}


# =============================================================================
# SUMMARY OF CHANGES
# =============================================================================

"""
DELETE:
1. Lines ~480-550: The entire manual metadata initialization block
2. Lines ~555-610: All the regex extraction (Wind, Tunnel, Panel, Mooring, Probes, Wave params)
3. Just keep: metadata = extract_metadata_from_filename(filename, path, df, experiment_name)

FIX:
1. Line 17: Change "import dataclass" → "from dataclasses import dataclass"
2. _extract_wave_parameters(): Fix type conversions (especially frequency!)
3. Add STILLWATER_SAMPLES to constants.py
4. Update dtype_map with all new fields

ADD:
1. Call validate_dtype_map(meta_df) before returning to catch missing columns

RESULT:
- ~100 lines deleted
- Cleaner code
- Correct datatypes
- No duplication
- Easy to maintain
"""
