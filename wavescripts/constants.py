#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Central configuration for wave analysis constants.

All magic numbers and tuning parameters should be defined here.
This makes it easy to:
- Track what values you've tried
- Document why values were chosen
- Adjust parameters without hunting through code
- Switch between different "profiles" for different experiments
"""

from dataclasses import dataclass
from typing import Dict

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

@dataclass(frozen=True)
class PhysicalConstants:
    """Universal physical constants - these never change."""
    GRAVITY: float = 9.80665  # m/s²
    WATER_SURFACE_TENSION: float = 0.074  # N/m at 20°C
    WATER_DENSITY: float = 1000.0  # kg/m³

PHYSICS = PhysicalConstants()


# =============================================================================
# MEASUREMENT SYSTEM
# =============================================================================

@dataclass(frozen=True)
class MeasurementConstants:
    """Hardware and sampling parameters."""
    SAMPLING_RATE: float = 250.0  # Hz (samples per second)
    NUM_PROBES: int = 4
    PROBE_NAMES: tuple = ("Probe 1", "Probe 2", "Probe 3", "Probe 4")
    
    # Unit conversions
    MM_TO_M: float = 0.001
    M_TO_MM: float = 1000.0

MEASUREMENT = MeasurementConstants()


# =============================================================================
# SIGNAL PROCESSING
# =============================================================================

@dataclass
class SignalProcessingParams:
    """Parameters for smoothing, filtering, and baseline detection."""
    
    # Baseline detection
    BASELINE_DURATION_SEC: float = 2.0  # seconds of data to use for baseline
    BASELINE_SIGMA_FACTOR: float = 1.0  # how many std devs above baseline = signal start
    
    # Smoothing windows (samples)
    DEFAULT_SMOOTHING_WINDOW: int = 1
    SMOOTHING_FULL_WIND: int = 15
    SMOOTHING_LOW_WIND: int = 10
    SMOOTHING_NO_WIND: int = 1
    
    # FFT/PSD parameters
    PSD_FREQUENCY_RESOLUTION: float = 0.125  # Hz per step in PSD
    FFT_FREQUENCY_WINDOW: float = 0.5  # Hz window around target frequency
    PSD_OVERLAP_FRACTION: float = 0.5  # fraction of segment overlap in Welch method

SIGNAL = SignalProcessingParams()


# =============================================================================
# WAVE DETECTION (RAMP-UP ALGORITHM)
# =============================================================================

@dataclass
class RampDetectionParams:
    """Parameters for detecting wave ramp-up phase."""
    
    # Peak detection
    MIN_PEAK_DISTANCE_FACTOR: float = 0.9  # minimum distance = this * samples_per_period
    PEAK_PROMINENCE_SIGMA: float = 3.0  # peak must be this many std devs above baseline
    
    # Ramp-up detection
    MIN_RAMP_PEAKS: int = 5  # minimum peaks in ramp-up
    MAX_RAMP_PEAKS: int = 15  # maximum peaks to search
    MAX_DIPS_ALLOWED: int = 2  # allow this many decreases in amplitude during ramp
    MIN_GROWTH_FACTOR: float = 1.015  # ramp end amplitude must be this * start amplitude
    
    # Period trimming (for selecting stable region)
    SKIP_PERIODS_FALLBACK: int = 17  # if ramp detection fails, skip this many periods
    KEEP_PERIODS_DEFAULT: int = 5  # default number of periods to analyze
    KEEP_PERIODS_SCALING: float = 1.0  # scaling factor: (input_periods - 13) * this

RAMP = RampDetectionParams()


# =============================================================================
# MANUAL WAVE DETECTION OVERRIDES
# =============================================================================
# These are hardcoded values for specific frequencies where automatic detection fails.
# TODO: Replace with robust automatic detection

@dataclass
class ManualDetectionPoints:
    """Manually calibrated detection points for specific test cases."""
    
    # 1.3 Hz wave parameters
    FREQ_1_3_HZ_PROBE1_START: int = 4500
    FREQ_1_3_HZ_PROBE2_OFFSET: int = 100  # from probe 1
    FREQ_1_3_HZ_PROBE3_OFFSET: int = 1700  # from probe 2
    
    # 0.65 Hz wave parameters
    FREQ_0_65_HZ_PROBE1_START: int = 3950
    FREQ_0_65_HZ_PROBE2_OFFSET: int = 50  # from probe 1
    FREQ_0_65_HZ_PROBE3_OFFSET: int = 500  # from probe 2

MANUAL = ManualDetectionPoints()


# =============================================================================
# PHYSICAL GEOMETRY
# =============================================================================

# Panel lengths in meters
PANEL_LENGTHS: Dict[str, float] = {
    "purple": 1.048,
    "yellow": 1.572,
    "full": 2.62,
    "reverse": 2.62,
}

# Wind speeds by condition (m/s)
WIND_SPEEDS: Dict[str, float] = {
    "no": 0.0,
    "lowest": 3.8,
    "low": 3.8,
    "full": 5.8,
}


# =============================================================================
# AMPLITUDE CALCULATION
# =============================================================================

@dataclass
class AmplitudeParams:
    """Parameters for extracting wave amplitude from signals."""
    
    # Percentile-based method
    UPPER_PERCENTILE: float = 99.5
    LOWER_PERCENTILE: float = 0.5
    AMPLITUDE_DIVISOR: float = 2.0  # (max - min) / 2

AMPLITUDE = AmplitudeParams()


# =============================================================================
# WAVENUMBER CALCULATION (DISPERSION RELATION)
# =============================================================================

@dataclass
class WavenumberParams:
    """Parameters for solving dispersion relation via Brent's method."""
    
    # Initial bracket multipliers for root finding
    DEEP_WATER_BRACKET_FACTOR: float = 0.1  # k_initial * this for lower bound
    SHALLOW_WATER_BRACKET_FACTOR: float = 10.0  # k_initial * this for upper bound
    
    # Bracket expansion factors if initial bracket fails
    BRACKET_EXPAND_LOWER: float = 0.5  # divide lower bound by this
    BRACKET_EXPAND_UPPER: float = 2.0  # multiply upper bound by this

WAVENUMBER = WavenumberParams()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_smoothing_window(wind_condition: str) -> int:
    """Get smoothing window size based on wind condition."""
    wind_map = {
        "full": SIGNAL.SMOOTHING_FULL_WIND,
        "low": SIGNAL.SMOOTHING_LOW_WIND,
        "no": SIGNAL.SMOOTHING_NO_WIND,
    }
    return wind_map.get(wind_condition.lower().strip(), SIGNAL.DEFAULT_SMOOTHING_WINDOW)


def get_wind_speed(wind_condition: str) -> float:
    """Get wind speed in m/s for given condition."""
    return WIND_SPEEDS.get(wind_condition.lower().strip(), 0.0)


def get_panel_length(panel_condition: str) -> float:
    """Get panel length in meters for given condition."""
    return PANEL_LENGTHS.get(panel_condition.lower().strip(), 0.0)


# =============================================================================
# CONFIGURATION PROFILES (OPTIONAL)
# =============================================================================
# Uncomment and use if you want to switch between different parameter sets

# @dataclass
# class ConfigProfile:
#     """A complete set of analysis parameters."""
#     signal: SignalProcessingParams
#     ramp: RampDetectionParams
#     amplitude: AmplitudeParams
# 
# # Conservative profile (stricter detection)
# CONSERVATIVE = ConfigProfile(
#     signal=SignalProcessingParams(BASELINE_SIGMA_FACTOR=2.0),
#     ramp=RampDetectionParams(MIN_RAMP_PEAKS=7, MIN_GROWTH_FACTOR=1.05),
#     amplitude=AmplitudeParams(UPPER_PERCENTILE=98.0, LOWER_PERCENTILE=2.0),
# )
# 
# # Aggressive profile (more permissive)
# AGGRESSIVE = ConfigProfile(
#     signal=SignalProcessingParams(BASELINE_SIGMA_FACTOR=0.5),
#     ramp=RampDetectionParams(MIN_RAMP_PEAKS=3, MIN_GROWTH_FACTOR=1.01),
#     amplitude=AmplitudeParams(UPPER_PERCENTILE=99.9, LOWER_PERCENTILE=0.1),
# )


# =============================================================================
# VALIDATION
# =============================================================================

def validate_constants():
    """Sanity check all constants. Call this at startup."""
    assert MEASUREMENT.SAMPLING_RATE > 0, "Sampling rate must be positive"
    assert 0 < SIGNAL.PSD_OVERLAP_FRACTION < 1, "Overlap must be between 0 and 1"
    assert RAMP.MIN_RAMP_PEAKS < RAMP.MAX_RAMP_PEAKS, "Min ramp peaks must be < max"
    assert AMPLITUDE.AMPLITUDE_DIVISOR != 0, "Amplitude divisor cannot be zero"
    print("✓ All constants validated")


if __name__ == "__main__":
    # Print all constants for review
    print("=== PHYSICAL CONSTANTS ===")
    print(f"Gravity: {PHYSICS.GRAVITY} m/s²")
    print(f"Surface tension: {PHYSICS.WATER_SURFACE_TENSION} N/m")
    
    print("\n=== MEASUREMENT SYSTEM ===")
    print(f"Sampling rate: {MEASUREMENT.SAMPLING_RATE} Hz")
    print(f"Number of probes: {MEASUREMENT.NUM_PROBES}")
    
    print("\n=== SIGNAL PROCESSING ===")
    print(f"Baseline duration: {SIGNAL.BASELINE_DURATION_SEC} sec")
    print(f"Baseline threshold: {SIGNAL.BASELINE_SIGMA_FACTOR} σ")
    print(f"PSD resolution: {SIGNAL.PSD_FREQUENCY_RESOLUTION} Hz")
    
    print("\n=== RAMP DETECTION ===")
    print(f"Ramp peaks: {RAMP.MIN_RAMP_PEAKS}-{RAMP.MAX_RAMP_PEAKS}")
    print(f"Min growth factor: {RAMP.MIN_GROWTH_FACTOR}")
    
    print("\n=== WIND SPEEDS ===")
    for condition, speed in WIND_SPEEDS.items():
        print(f"{condition:10s}: {speed} m/s")
    
    print("\n=== PANEL LENGTHS ===")
    for condition, length in PANEL_LENGTHS.items():
        print(f"{condition:10s}: {length} m")
    
    validate_constants()
