"""
Clean dtype_map with default float64 and explicit string overrides.

This makes it much easier to maintain - just list the exceptions!
"""

from typing import Dict

import pandas as pd

# =============================================================================
# APPROACH 1: Simple dict with defaults (RECOMMENDED)
# =============================================================================

# Only specify the NON-float columns
NON_FLOAT_COLUMNS = {
    "WindCondition": str,
    "TunnelCondition": str,
    "PanelCondition": str,
    "Mooring": str,
    "Run number": str,
    "experiment_folder": str,
    "path": str,
    "file_date": str,  # ISO datetime string
}


def apply_dtypes(meta_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply datatypes with default float64 for everything except strings.

    Args:
        meta_df: Metadata DataFrame with any columns

    Returns:
        DataFrame with correct types applied
    """
    meta_df = meta_df.copy()

    for col in meta_df.columns:
        if col in NON_FLOAT_COLUMNS:
            # Explicit string columns
            try:
                meta_df[col] = meta_df[col].astype(NON_FLOAT_COLUMNS[col])
            except Exception as e:
                print(
                    f"Warning: Could not convert {col} to {NON_FLOAT_COLUMNS[col]}: {e}"
                )
        else:
            # Everything else defaults to float64
            try:
                meta_df[col] = pd.to_numeric(meta_df[col], errors="coerce")
            except Exception as e:
                print(f"Warning: Could not convert {col} to float64: {e}")

    return meta_df


# =============================================================================
# APPROACH 2: Build dtype_map programmatically (ALTERNATIVE)
# =============================================================================


def build_dtype_map(meta_df: pd.DataFrame) -> Dict[str, str]:
    """
    Build dtype_map from DataFrame columns.
    Everything is float64 except explicit overrides.

    Use this to generate the dtype_map from an actual DataFrame.
    """
    dtype_map = {}

    for col in meta_df.columns:
        if col in NON_FLOAT_COLUMNS:
            dtype_map[col] = NON_FLOAT_COLUMNS[col]
        else:
            dtype_map[col] = "float64"

    return dtype_map


# =============================================================================
# APPROACH 3: Use defaultdict (CLEANEST for your use case)
# =============================================================================

from collections import defaultdict


class FloatDefaultDict(defaultdict):
    """
    A dict that returns 'float64' for any missing key.

    Usage:
        dtype_map = FloatDefaultDict()
        dtype_map.update(NON_FLOAT_COLUMNS)

        # Now any column not in NON_FLOAT_COLUMNS will return 'float64'
        dtype_map["Probe 1 Amplitude"]  # → 'float64'
        dtype_map["WindCondition"]      # → <class 'str'>
    """

    def __init__(self):
        super().__init__(lambda: "float64")

    def __missing__(self, key):
        return "float64"


# Create the dtype_map
dtype_map = FloatDefaultDict()
dtype_map.update(NON_FLOAT_COLUMNS)


# =============================================================================
# HOW TO USE IN YOUR CODE
# =============================================================================

"""
IN data_loader.py, at the end of load_or_update():
"""


def load_or_update_final_step():
    # ... (all your existing code)

    # Create DataFrame
    meta_df = pd.DataFrame(all_meta_list)

    # OPTION 1: Use the function (safest, most explicit)
    meta_df = apply_dtypes(meta_df)

    # OPTION 2: Use defaultdict approach
    # for col, dtype in dtype_map.items():
    #     if col in meta_df.columns:
    #         try:
    #             meta_df[col] = meta_df[col].astype(dtype)
    #         except Exception as e:
    #             print(f"Warning: {col} → {dtype} failed: {e}")

    return all_dfs, meta_df


"""
IN update_processed_metadata(), when loading existing data:
"""


def update_processed_metadata_dtype_handling():
    # ... (existing code)

    # After loading from JSON
    old_df = pd.DataFrame(old_records)
    old_df = apply_dtypes(old_df)  # Apply correct types

    # ... (rest of function)


# =============================================================================
# COMPLETE CODE FOR YOUR data_loader.py
# =============================================================================

# At top of file (after imports):

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
    """Apply datatypes: strings for NON_FLOAT_COLUMNS, float64 for everything else."""
    meta_df = meta_df.copy()

    for col in meta_df.columns:
        if col in NON_FLOAT_COLUMNS:
            try:
                meta_df[col] = meta_df[col].astype(NON_FLOAT_COLUMNS[col])
            except Exception as e:
                print(f"Warning: Could not convert {col} to string: {e}")
        else:
            try:
                meta_df[col] = pd.to_numeric(meta_df[col], errors="coerce")
            except Exception as e:
                print(f"Warning: Could not convert {col} to float64: {e}")

    return meta_df


# In load_or_update(), replace the line:
#     meta_df = meta_df.astype(dtype_map)
# With:
#     meta_df = apply_dtypes(meta_df)


# In update_processed_metadata(), replace:
#     old_df = old_df.astype(dtype_map)
# With:
#     old_df = apply_dtypes(old_df)


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


# =============================================================================
# MIGRATION GUIDE
# =============================================================================

"""
TO UPDATE YOUR EXISTING data_loader.py:

1. DELETE the entire dtype_map dict (all ~60 lines)

2. ADD these at top of file (after imports):

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
       meta_df = meta_df.copy()
       for col in meta_df.columns:
           if col in NON_FLOAT_COLUMNS:
               meta_df[col] = meta_df[col].astype(NON_FLOAT_COLUMNS[col])
           else:
               meta_df[col] = pd.to_numeric(meta_df[col], errors='coerce')
       return meta_df

3. In load_or_update(), replace:
   meta_df = meta_df.astype(dtype_map)

   With:
   meta_df = apply_dtypes(meta_df)

4. In update_processed_metadata(), replace:
   old_df = old_df.astype(dtype_map)

   With:
   old_df = apply_dtypes(old_df)

5. Done! Now you can add new float columns without updating dtype_map.

BENEFITS:
- Add new probe metrics? Automatically float64
- Add new FFT columns? Automatically float64
- Only need to update NON_FLOAT_COLUMNS when adding string fields
- Much less maintenance
- Clearer code (explicit about exceptions, not the rule)
"""
