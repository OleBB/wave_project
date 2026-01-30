"""
Improvements for _extract_probe_amplitude()

Shows current version, suggested improvements, and alternatives.
"""

import numpy as np
import pandas as pd
from typing import Optional
from constants import AMPLITUDE


# =============================================================================
# YOUR CURRENT VERSION (already pretty good!)
# =============================================================================

def _extract_probe_amplitude_CURRENT(
    df: pd.DataFrame,
    row: pd.Series,
    probe_num: int,
) -> float | None:
    """Extract peak-to-peak amplitude for a specific probe (percentile-based)."""
    signal = _extract_probe_signal(df, row, probe_num)
    if signal is None or len(signal) == 0:
        return None
    
    signal = np.asarray(signal)           
    if signal.dtype.kind in "SUO":       
        return None
    
    upper_p = AMPLITUDE.UPPER_PERCENTILE
    lower_p = AMPLITUDE.LOWER_PERCENTILE
    divisor  = AMPLITUDE.AMPLITUDE_DIVISOR
    
    upper, lower = np.nanpercentile(
        signal,
        [upper_p, lower_p],
        method="linear"
    )
    amplitude = (upper - lower) / divisor
    return float(amplitude)


# =============================================================================
# IMPROVEMENT 1: Add validation and edge case handling
# =============================================================================

def _extract_probe_amplitude_IMPROVED(
    df: pd.DataFrame,
    row: pd.Series,
    probe_num: int,
    min_samples: int = 10,  # need minimum data for valid statistics
) -> Optional[float]:
    """
    Extract peak-to-peak amplitude for a specific probe (percentile-based).
    
    Args:
        df: DataFrame with probe data
        row: Metadata row for this measurement
        probe_num: Probe number (1-4)
        min_samples: Minimum samples required for valid amplitude (default: 10)
    
    Returns:
        Amplitude in mm, or None if insufficient/invalid data
        
    Notes:
        - Uses nanpercentile to handle NaN values gracefully
        - Returns None if >50% of signal is NaN (too unreliable)
        - Returns None if signal has zero variance (flat line)
    """
    signal = _extract_probe_signal(df, row, probe_num)
    if signal is None:
        return None
    
    # Convert to array and validate
    signal = np.asarray(signal, dtype=float)
    
    # Check for non-numeric data
    if signal.dtype.kind in "SUO":
        return None
    
    # Remove any +/- inf values (replace with NaN)
    signal = np.where(np.isinf(signal), np.nan, signal)
    
    # Count valid samples
    valid_mask = np.isfinite(signal)
    n_valid = np.sum(valid_mask)
    
    # Need minimum samples for statistics
    if n_valid < min_samples:
        return None
    
    # If >50% NaN, signal is too corrupted
    if n_valid < len(signal) * 0.5:
        return None
    
    # Check for zero variance (flat line = broken sensor?)
    if n_valid > 1 and np.nanstd(signal) < 1e-10:  # essentially zero
        return None
    
    # Calculate amplitude using percentiles
    upper_p = AMPLITUDE.UPPER_PERCENTILE
    lower_p = AMPLITUDE.LOWER_PERCENTILE
    
    upper, lower = np.nanpercentile(
        signal,
        [upper_p, lower_p],
        method="linear"
    )
    
    # Sanity check: amplitude should be positive
    amplitude = (upper - lower) / AMPLITUDE.AMPLITUDE_DIVISOR
    
    if amplitude < 0:  # should never happen, but defensive
        return None
    
    return float(amplitude)


# =============================================================================
# IMPROVEMENT 2: Add debug mode for troubleshooting
# =============================================================================

def _extract_probe_amplitude_WITH_DEBUG(
    df: pd.DataFrame,
    row: pd.Series,
    probe_num: int,
    min_samples: int = 10,
    debug: bool = False,
) -> Optional[float]:
    """
    Extract amplitude with optional debug output.
    
    Set debug=True to print diagnostic info when amplitude extraction fails.
    """
    signal = _extract_probe_signal(df, row, probe_num)
    
    if signal is None:
        if debug:
            print(f"⚠ Probe {probe_num}: _extract_probe_signal returned None")
        return None
    
    signal = np.asarray(signal, dtype=float)
    
    if signal.dtype.kind in "SUO":
        if debug:
            print(f"⚠ Probe {probe_num}: Signal has non-numeric dtype {signal.dtype}")
        return None
    
    # Remove infinities
    signal = np.where(np.isinf(signal), np.nan, signal)
    
    n_valid = np.sum(np.isfinite(signal))
    n_total = len(signal)
    
    if n_valid < min_samples:
        if debug:
            print(f"⚠ Probe {probe_num}: Only {n_valid} valid samples (need {min_samples})")
        return None
    
    if n_valid < n_total * 0.5:
        if debug:
            print(f"⚠ Probe {probe_num}: {n_valid}/{n_total} valid ({n_valid/n_total*100:.1f}%) - too many NaNs")
        return None
    
    std = np.nanstd(signal)
    if n_valid > 1 and std < 1e-10:
        if debug:
            print(f"⚠ Probe {probe_num}: Zero variance (std={std:.2e}) - flat line?")
        return None
    
    upper, lower = np.nanpercentile(
        signal,
        [AMPLITUDE.UPPER_PERCENTILE, AMPLITUDE.LOWER_PERCENTILE],
        method="linear"
    )
    
    amplitude = (upper - lower) / AMPLITUDE.AMPLITUDE_DIVISOR
    
    if amplitude < 0:
        if debug:
            print(f"⚠ Probe {probe_num}: Negative amplitude {amplitude:.3f} - impossible!")
        return None
    
    if debug:
        print(f"✓ Probe {probe_num}: amp={amplitude:.3f} mm "
              f"(n={n_valid}, std={std:.3f}, range=[{lower:.2f}, {upper:.2f}])")
    
    return float(amplitude)


# =============================================================================
# ALTERNATIVE: Return dict with metadata for quality control
# =============================================================================

def extract_probe_amplitude_with_metadata(
    df: pd.DataFrame,
    row: pd.Series,
    probe_num: int,
    min_samples: int = 10,
) -> dict:
    """
    Extract amplitude AND quality metrics for downstream filtering.
    
    Returns:
        dict with keys:
            - 'amplitude': float or None
            - 'n_samples': total samples
            - 'n_valid': valid (non-NaN) samples
            - 'std': standard deviation
            - 'percentiles': (lower, upper) values used
            - 'quality': 'good' | 'warning' | 'bad'
    
    Use case: You can filter out low-quality measurements later:
        results = [extract_probe_amplitude_with_metadata(...) for ...]
        good_results = [r for r in results if r['quality'] == 'good']
    """
    result = {
        'amplitude': None,
        'n_samples': 0,
        'n_valid': 0,
        'std': np.nan,
        'percentiles': (np.nan, np.nan),
        'quality': 'bad',
    }
    
    signal = _extract_probe_signal(df, row, probe_num)
    if signal is None:
        return result
    
    signal = np.asarray(signal, dtype=float)
    signal = np.where(np.isinf(signal), np.nan, signal)
    
    n_total = len(signal)
    n_valid = np.sum(np.isfinite(signal))
    
    result['n_samples'] = n_total
    result['n_valid'] = n_valid
    
    if n_valid < min_samples:
        return result
    
    std = np.nanstd(signal)
    result['std'] = float(std)
    
    if std < 1e-10:
        return result
    
    upper, lower = np.nanpercentile(
        signal,
        [AMPLITUDE.UPPER_PERCENTILE, AMPLITUDE.LOWER_PERCENTILE],
        method="linear"
    )
    
    result['percentiles'] = (float(lower), float(upper))
    
    amplitude = (upper - lower) / AMPLITUDE.AMPLITUDE_DIVISOR
    result['amplitude'] = float(amplitude)
    
    # Assign quality based on data completeness and variance
    if n_valid == n_total and std > 0.1:
        result['quality'] = 'good'
    elif n_valid >= n_total * 0.8 and std > 0.01:
        result['quality'] = 'warning'  # usable but not perfect
    else:
        result['quality'] = 'bad'
    
    return result


# =============================================================================
# IMPROVEMENT 3: Cache repeated calculations
# =============================================================================

from functools import lru_cache

def _extract_probe_amplitude_CACHED(
    df: pd.DataFrame,
    row: pd.Series,
    probe_num: int,
) -> Optional[float]:
    """
    Cached version - useful if you call this multiple times with same inputs.
    
    NOTE: Be careful with caching DataFrames - only use if:
    1. Your df doesn't change between calls
    2. You're processing many probes from same df
    
    For typical use (process once, move on), caching adds overhead for no benefit.
    """
    # Convert row to hashable tuple for caching
    path = row.get('path', '')
    start_key = f"Computed Probe {probe_num} start"
    end_key = f"Computed Probe {probe_num} end"
    
    cache_key = (
        id(df),  # DataFrame identity
        path,
        row.get(start_key),
        row.get(end_key),
        probe_num
    )
    
    return _cached_amplitude_calc(cache_key, df, row, probe_num)


@lru_cache(maxsize=128)
def _cached_amplitude_calc(cache_key, df, row, probe_num):
    """Actual cached calculation - separated for @lru_cache to work."""
    # ... same logic as above
    pass  # (implement as needed)


# =============================================================================
# TESTING HELPERS
# =============================================================================

def test_amplitude_extraction():
    """Quick tests to verify behavior."""
    # Test 1: Normal signal
    signal = np.sin(np.linspace(0, 4*np.pi, 1000)) * 10
    # Expected amplitude ≈ 10 mm (peak-to-peak / 2)
    
    # Test 2: Signal with NaNs
    signal_with_nans = signal.copy()
    signal_with_nans[::10] = np.nan  # 10% NaN
    # Should still work
    
    # Test 3: Flat line
    flat = np.ones(1000) * 5.0
    # Should return None (zero variance)
    
    # Test 4: Too few samples
    tiny = signal[:5]
    # Should return None (< min_samples)
    
    # Test 5: All NaN
    all_nan = np.full(1000, np.nan)
    # Should return None
    
    print("Run these tests with actual data to validate!")


# =============================================================================
# RECOMMENDATIONS
# =============================================================================

"""
WHAT TO USE:

1. **For production (recommended):** _extract_probe_amplitude_IMPROVED
   - Adds important edge case handling
   - Prevents silent errors from bad data
   - Minimal performance cost
   
2. **For debugging:** _extract_probe_amplitude_WITH_DEBUG
   - Use when amplitudes look wrong
   - Helps diagnose data quality issues
   - Can disable debug output in production
   
3. **For quality control:** extract_probe_amplitude_with_metadata
   - Use if you need to filter results by quality
   - Good for automated QC pipelines
   - More complex return type
   
4. **Avoid caching** unless you profile and confirm it helps
   - Most likely not needed for your use case
   - Adds complexity
   
YOUR CURRENT CODE IS ALREADY GOOD! These are just polish.

The main additions are:
✓ Validate minimum samples
✓ Check for excessive NaN percentage
✓ Detect zero variance (flat line)
✓ Handle +/- infinity values
✓ Optional debug output

MINOR TWEAKS TO YOUR CURRENT CODE:

1. Add `min_samples` parameter (default 10)
2. Check `n_valid < len(signal) * 0.5` for NaN threshold
3. Check `np.nanstd(signal) < 1e-10` for flat line
4. Add `np.where(np.isinf(...), np.nan, ...)` before percentile
5. Consider changing return type hint to `Optional[float]` (clearer than `float | None`)

That's it! Your function is already well-structured.
"""


if __name__ == "__main__":
    print(__doc__)
    print("\nYour current function is already quite good!")
    print("Main suggestions:")
    print("  1. Add min_samples validation")
    print("  2. Check for excessive NaN percentage")
    print("  3. Detect flat lines (zero variance)")
    print("  4. Handle infinity values")
    print("  5. Optional: add debug mode")
