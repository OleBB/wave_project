"""
QUICK REFERENCE: Using constants.py
====================================

STEP 1: Import what you need
-----------------------------
from constants import PHYSICS, MEASUREMENT, SIGNAL, RAMP, AMPLITUDE

# Or import specific helper functions
from constants import get_smoothing_window, get_wind_speed, get_panel_length


STEP 2: Replace magic numbers
------------------------------

BAD:
    baseline_samples = int(2 * Fs)  # What's 2? Seconds? Samples?
    
GOOD:
    baseline_samples = int(SIGNAL.BASELINE_DURATION_SEC * MEASUREMENT.SAMPLING_RATE)


STEP 3: Use helper functions for lookups
-----------------------------------------

BAD:
    if wind_condition == "full":
        window = 15
    elif wind_condition == "low":
        window = 10
    else:
        window = 1

GOOD:
    window = get_smoothing_window(wind_condition)


COMMON PATTERNS
===============

Physical calculations
---------------------
g = PHYSICS.GRAVITY
sigma = PHYSICS.WATER_SURFACE_TENSION
rho = PHYSICS.WATER_DENSITY


Unit conversions
----------------
height_m = height_mm * MEASUREMENT.MM_TO_M
depth_mm = depth_m * MEASUREMENT.M_TO_MM


Sampling math
-------------
Fs = MEASUREMENT.SAMPLING_RATE
dt = 1.0 / Fs
samples_per_period = int(Fs / frequency)


Signal processing
-----------------
# Baseline
baseline_samples = int(SIGNAL.BASELINE_DURATION_SEC * MEASUREMENT.SAMPLING_RATE)
threshold = baseline_mean + SIGNAL.BASELINE_SIGMA_FACTOR * baseline_std

# PSD
nperseg = int(MEASUREMENT.SAMPLING_RATE / SIGNAL.PSD_FREQUENCY_RESOLUTION)
noverlap = int(nperseg * SIGNAL.PSD_OVERLAP_FRACTION)

# Smoothing
window = get_smoothing_window(meta_row["WindCondition"])


Ramp detection
--------------
peaks, _ = find_peaks(
    signal,
    distance=int(RAMP.MIN_PEAK_DISTANCE_FACTOR * samples_per_period),
    prominence=RAMP.PEAK_PROMINENCE_SIGMA * baseline_std
)

ramp = find_best_ramp(
    peak_amplitudes,
    min_len=RAMP.MIN_RAMP_PEAKS,
    max_len=RAMP.MAX_RAMP_PEAKS,
    max_dips=RAMP.MAX_DIPS_ALLOWED,
    min_growth=RAMP.MIN_GROWTH_FACTOR
)


Amplitude calculation
---------------------
upper = np.percentile(signal, AMPLITUDE.UPPER_PERCENTILE)
lower = np.percentile(signal, AMPLITUDE.LOWER_PERCENTILE)
amplitude = (upper - lower) / AMPLITUDE.AMPLITUDE_DIVISOR


Geometry lookups
----------------
panel_length_m = get_panel_length(meta_row["PanelCondition"])
wind_speed_ms = get_wind_speed(meta_row["WindCondition"])


WHEN TO UPDATE constants.py
============================

Add new constant when:
✓ You type the same number twice
✓ You have a comment explaining what a number means
✓ The number might need tuning later
✓ The number has physical meaning

Don't add constant when:
✗ It's obviously 0, 1, or 2 in simple math (like /2 for amplitude)
✗ It's a temporary debugging value
✗ It's specific to one test case and won't be reused


TESTING NEW VALUES
==================

1. Edit constants.py
2. Add entry to CONSTANTS_CHANGELOG.md
3. Run your analysis
4. Check results
5. Update changelog with results
6. Keep or revert the change


ADVANCED: Parameter profiles (optional)
=======================================

If you need multiple parameter sets, uncomment the ConfigProfile
section in constants.py:

# Use conservative parameters
from constants import CONSERVATIVE
SIGNAL = CONSERVATIVE.signal
RAMP = CONSERVATIVE.ramp

# Use aggressive parameters  
from constants import AGGRESSIVE
SIGNAL = AGGRESSIVE.signal


DEBUGGING TIPS
==============

Print all constants:
    python constants.py

Validate at startup:
    from constants import validate_constants
    validate_constants()

Check what value you're using:
    print(f"Using baseline sigma = {SIGNAL.BASELINE_SIGMA_FACTOR}")
"""

if __name__ == "__main__":
    print(__doc__)
