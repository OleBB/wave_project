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
    """Universal physical constants - these seldom change."""
    GRAVITY: float = 9.80665          # [m/s^2] standard gravity
    # --- Water properties at ~21.5 °C ---
    WATER_SURFACE_TENSION: float = 0.0725      # [N/m] at ~21.5 °C
    WATER_DENSITY: float = 997.8               # [kg/m^3] at ~21.5 °C
    KINEMATIC_VISCOSITY_WATER: float = 0.96e-6 # [m^2/s] at ~21.5 °C
    # --- Air properties at 21.5 °C, ~100 kPa, 50% RH (Blindern) ---
    AIR_DENSITY: float = 1.18                  # [kg/m^3] at 21.5 °C, 50% RH
    KINEMATIC_VISCOSITY_AIR: float = 1.52e-5   # [m^2/s] at ~21.5 °C

#vanntemperaturen er ikke nødvendigvis like varm som luften, pga påfyll av nytt vann.

PHYSICS = PhysicalConstants()


# =============================================================================
# MEASUREMENT SYSTEM
# =============================================================================

@dataclass(frozen=True)
class MeasurementConstants:
    """Hardware and sampling parameters."""
    SAMPLING_RATE: float = 250.0  # Hz (samples per second)
    STILLWATER_SAMPLES: int = 250 # Dette brukes av data_loader til å hente stilltilstand fra en no-wind-run.
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


@dataclass
class ClipParams:
    """Outlier clipping thresholds applied to probe signals.

    The cleaning pipeline runs in layers, in this order:

      Layer 0a — Ceiling detection (raw signal, before zeroing)
        Probe hardware outputs ~198.70 mm when the tip is beyond its measuring
        range. Three observed variants:
          1. Exact freeze:   raw ≈ 198.70 mm, rolling std ≈ 0
          2. Near-ceiling:   raw ≈ 198.69 mm ± tiny noise (std < STUCK_STD_MM)
          3. DC shift:       raw base ≈ 190 mm, but wave signal still visible on
                             top — NOT caught by ceiling detection alone; caught by
                             Layer 0b.
        Any sample within CEILING_BAND_MM of PROBE_CEILING_MM is set to NaN
        before zeroing. Diagnostic print: CEILCLIP.

      Layer 0b — DC step detection (raw signal, before zeroing)
        A probe repositioned or hardware-faulted mid-run shifts the DC level by
        ~80–90 mm (e.g. from ~100 mm to ~190 mm). The stillwater reference
        (captured in the first 2 s) is no longer valid after the step. All samples
        from the step onset to end-of-run are set to NaN.
        The step location is stored as step_at_s_{pos} in combined_meta.
        Diagnostic print: STEPCLIP.

      Layer 0c — Stuck probe detection (raw signal, before zeroing)
        Extended flat segment at the WRONG level (catches variants 1 and 2 above
        as a redundant safety net, and any other stuck-at-wrong-level faults).
        Diagnostic print: STUCK.

      ── zeroing: eta_ = -(raw − stillwater_ref) ──────────────────────────────

      Layer 1 — Dynamic hard cap (eta_ signal, wave/wind runs only)
        Backstop for any fault that slipped through Layers 0a–0c.
        Stillwater/nowave runs have no hard cap — detection layers handle faults.

      Layer 2 — Velocity filter + shoulder removal
        Single-sample spikes: up AND back within 2 samples, opposite sign.
        VEL_BUFFER samples removed on each side of each spike core.
        Diagnostic print: VELCLIP.

      Layer 3 — Isolated sample check
        A valid sample surrounded by NaN on both sides — set to NaN.
        Diagnostic print: ISOCLIP.
    """
    # ── Layer 1: hard cap ──────────────────────────────────────────────────────
    WAVE_MM:   float = 200.0  # fallback hard cap when no voltage info available
    WAVE_CLIP_FACTOR: float = 270.0  # hard cap = factor × amp_volts; 0.1V→27mm, 0.2V→54mm, 0.3V→81mm
    WIND_BASE_VOLT: float = 0.05     # extra effective voltage for wind runs: adds ~13.5mm to cap

    # ── Layer 2: velocity filter ───────────────────────────────────────────────
    DIFF_MM: float = 10.0   # velocity threshold: 10 mm/sample = 2500 mm/s (~4× max physical wave velocity)
    VEL_BUFFER: int = 2     # samples removed on each side of a velocity-detected spike (shoulder contamination)

    # ── Layer 0c: stuck probe ──────────────────────────────────────────────────
    STUCK_STD_MM: float = 0.05   # rolling std below this → probe is stuck (well below ~0.13 mm noise floor)
    STUCK_MIN_S: float = 1.0     # minimum duration of flat segment to flag as malfunction [s]

    # ── Layer 0a: ceiling detection ────────────────────────────────────────────
    PROBE_CEILING_MM: float = 198.70  # hardware output when probe tip is beyond max range [mm]
    CEILING_BAND_MM:  float = 2.0     # tolerance: |raw − PROBE_CEILING_MM| < this → NaN [mm]
                                       # Covers exact freeze (198.70) and near-ceiling noise (198.69 ± δ).

    # ── Layer 0b: DC step detection ───────────────────────────────────────────
    DC_STEP_MM:      float = 50.0  # minimum level change to flag as a DC step [mm]
                                    # Observed fault steps are ~80–90 mm; max wave-induced
                                    # drift in a 1 s rolling median is <25 mm even at 0.65 Hz
                                    # high-amplitude, so 50 mm is safely above the noise floor.
    DC_STEP_WINDOW_S: float = 1.0  # rolling median window for DC level estimation [s]
                                    # 1 s covers ~0.65 wave periods at the lowest frequency (0.65 Hz),
                                    # which is enough to suppress most wave oscillation in the median.
    DC_STEP_BUFFER_S: float = 0.5  # extra NaN margin added before the detected step onset [s]
                                    # Guards against the step being detected slightly late due to
                                    # the rolling-window lag.

    # ── mstop tail measurement ─────────────────────────────────────────────────
    TAIL_WINDOW_S: float = 5.0  # seconds of post-wave signal used for mstop_tail_mm_{pos}
                                  # Captures residual energy immediately after the wave train ends.
                                  # Should cover at least 3 periods at the lowest frequency (0.65 Hz
                                  # → T=1.54 s → 3 periods = 4.6 s), so 5 s is a safe minimum.

    # ── tail period diagnostics (seiche emergence tracking) ────────────────────
    # After the wave train ends the tank rings down. Higher frequencies (paddle freq)
    # dissipate faster; longer-period seiche modes persist. These constants control
    # how the per-cycle period series is summarised in the pipeline.
    # Physical seiche periods at d=580 mm, L≈25 m:  T1≈20.9 s, T2≈10.4 s, T3≈7.0 s
    #
    # Method: zero-upcrossing period tracking.
    #   eta_ is already zeroed to stillwater, so zero-upcrossings of eta_ define
    #   wave cycle boundaries. Period of cycle i = t_upcross[i+1] - t_upcross[i].
    #   One value per cycle — no windowing, no averaging over the evolving period.
    TAIL_PERIOD_SAMPLE_OFFSETS_S: tuple = (5.0, 15.0, 25.0)  # seconds from good_end at which to read nearest upcrossing period
    TAIL_PERIOD_MIN_S:  float = 0.15   # reject upcrossing-to-upcrossing gaps shorter than this [s]
                                        # Filters noise crossings; must be < 1/max_freq = 1/1.9 ≈ 0.53 s at highest paddle freq
    TAIL_PERIOD_MAX_S:  float = 30.0   # reject gaps longer than this [s]
                                        # Filters slow DC drift masquerading as a crossing; safely above T1 ≈ 20.9 s
    TAIL_CLEAR_FACTOR:  float = 2.0    # tail is "cleared" at the first upcrossing cycle whose period > FACTOR × paddle_T

    # ── Misc ───────────────────────────────────────────────────────────────────
    MAX_NAN_FRACTION: float = 0.05  # if >5% of a signal window is clipped, skip that probe/run for FFT
    INTERP_MAX_GAP: int = 10        # fallback max interpolation gap (nowave/stillwater runs) [samples]
                                     # Wave runs use int(fs / (4 × freq)) = 1/4 wavelength instead.

CLIP = ClipParams()


# =============================================================================
# STILLWATER LEVEL FITTING
# =============================================================================

@dataclass(frozen=True)
class StillwaterParams:
    """Tuning parameters for per-run stillwater self-calibration.

    The fit uses all no-wind runs in a folder to estimate how the still-water
    level changes over a session (evaporation, wind setup).
    """
    PRE_WAVE_S:  float = 2.0   # seconds of pre-wave signal used from wave runs [s]
    # 2 s is safe for 1.0+ Hz (wave front arrives later); may clip for 0.65 Hz
    # but those runs are low priority. Increase if diagnostics show wave contamination.
    OUTLIER_MM:  float = 1.0   # kept for backward compat (no longer used in fitting)

STILLWATER = StillwaterParams()

# Filenames containing any of these substrings are excluded from the stillwater
# drift fit — the file is kept on disk for reference but not used as a calibration
# point.  Add entries here rather than deleting files.
#
# NOTE: sloshing runs (e.g. "nestenstille") are NOT excluded — the median of a
# sloshing signal is the mid-level, which is the true water surface regardless
# of oscillation amplitude.  Only exclude runs with a genuine DC bias (e.g.
# probe out of water, equipment fault).
STILLWATER_EXCLUDE: tuple[str, ...] = ()


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
# VISUALIZATION CONSTANTS
# =============================================================================

@dataclass(frozen=True)
class PlottPent:
    """Standardized color palette for different experimental conditions."""
    WIND_FULL: str = "#D62728"   # Red
    WIND_LOW: str = "#2ca02c"    # grøn
    WIND_NO: str = "#3F51B5"     # Blue indigo
    DEFAULT: str = "#7F7F7F"    # Grey

# TIPS: Kjør bunnen av plotter.py for å se på fargene
#  for plotter (Plotly/Matplotlib)
WIND_COLOR_MAP: Dict[str, str] = {
    "full": PlottPent.WIND_FULL,
    "low": PlottPent.WIND_LOW,
    "lowest": PlottPent.WIND_LOW,
    "no": PlottPent.WIND_NO,
}

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
# COLUMN NAMES
# =============================================================================

class ProbeColumns:
    """Column name templates for probe-specific data.

    Use .format(i=probe_number) to get the actual column name.
    Example: ProbeColumns.AMPLITUDE.format(i=1) → "Probe 1 Amplitude"
    """

    # Physical setup
    MM_FROM_PADDLE = "Probe {i} mm from paddle"
    STILLWATER = "Stillwater Probe {i}"

    # Computed analysis ranges
    START = "Computed Probe {i} start"
    END = "Computed Probe {i} end"

    # Basic amplitude (from percentile method)
    # TODO - rename - give descriptive name
    AMPLITUDE = "Probe {i} Amplitude"

    # PSD-derived metrics
    AMPLITUDE_PSD = "Probe {i} Amplitude (PSD)"
    SWELL_AMPLITUDE_PSD = "Probe {i} Swell Amplitude (PSD)"
    WIND_AMPLITUDE_PSD = "Probe {i} Wind Amplitude (PSD)"
    TOTAL_AMPLITUDE_PSD = "Probe {i} Total Amplitude (PSD)"

    # Signal quality metrics
    WAVE_STABILITY = "Probe {i} wave_stability"
    PERIOD_CV      = "Probe {i} period_amplitude_cv"

    # FFT-derived metrics
    AMPLITUDE_FFT = "Probe {i} Amplitude (FFT)"
    FREQUENCY_FFT = "Probe {i} Frequency (FFT)"
    PERIOD_FFT = "Probe {i} WavePeriod (FFT)"
    WAVENUMBER_FFT = "Probe {i} Wavenumber (FFT)"
    WAVELENGTH_FFT = "Probe {i} Wavelength (FFT)"
    KL_FFT = "Probe {i} kL (FFT)"
    KA_FFT = "Probe {i} ka (FFT)"
    TANH_KH_FFT = "Probe {i} tanh(kH) (FFT)"
    CELERITY_FFT = "Probe {i} Celerity (FFT)"
    HS  = "Probe {i} Significant Wave Height Hs"
    HM0 = "Probe {i} Significant Wave Height Hm0"
    FROUDE_FFT      = "Probe {i} Froude (FFT)"
    WIND_CELERITY_FFT = "Probe {i} Wind/Celerity (FFT)"
    F_PM_RATIO_FFT  = "Probe {i} f/f_PM (FFT)"
    URSELL_FFT      = "Probe {i} Ursell (FFT)"


class GlobalColumns:
    """Non-probe-specific column names."""

    # Identifiers
    PATH = "path"
    EXPERIMENT_FOLDER = "experiment_folder"
    FILE_DATE = "file_date"
    RUN_NUMBER = "Run number"
    PROCESSED_FOLDER = "PROCESSED_folder"

    # Experimental conditions
    WIND_CONDITION = "WindCondition"
    TUNNEL_CONDITION = "TunnelCondition"
    PANEL_CONDITION = "PanelCondition"
    MOORING = "Mooring"

    #Grouping conditions
    PANEL_CONDITION_GROUPED = "PanelConditionGrouped"

    # Input parameters
    WAVE_AMPLITUDE_INPUT = "WaveAmplitudeInput [Volt]"
    WAVE_FREQUENCY_INPUT = "WaveFrequencyInput [Hz]"
    WAVE_PERIOD_INPUT = "WavePeriodInput"
    WATER_DEPTH = "WaterDepth [mm]"
    EXTRA_SECONDS = "Extra seconds"

    # Computed global metrics — "Expected" = derived from WaveFrequencyInput (wavemaker setting),
    # not from measured FFT frequency. Use "Probe {pos} ka (FFT)" etc. for actual measured values.
    WAVENUMBER = "Expected Wavenumber"
    WAVELENGTH = "Expected Wavelength"
    KL = "Expected kL"
    KA = "Expected ka"
    KH = "Expected kH"
    TANH_KH = "Expected tanh(kH)"
    CELERITY = "Expected Celerity"
    FROUDE          = "Expected Froude"
    WIND_CELERITY   = "Expected Wind/Celerity"
    F_PM_RATIO      = "Expected f/f_PM"
    URSELL          = "Expected Ursell"
    WINDSPEED = "Windspeed"

    # "Given" metrics (legacy columns - consider deprecating)
    #DE umerka over (de gamle) er jo basert på input-frekvens.
    WAVE_FREQUENCY_GIVEN = "Wavefrequency (given)"
    WAVE_PERIOD_GIVEN = "Waveperiod (given)"
    WAVENUMBER_GIVEN = "Wavenumber (given)"
    WAVELENGTH_GIVEN = "Wavelength (given)"
    KL_GIVEN = "kL (given)"
    KA_GIVEN = "ka (given)"
    KH_GIVEN = "kH (given)"
    TANH_KH_GIVEN = "tanh(kH) (given)"
    CELERITY_GIVEN = "Celerity (given)"
    HS_GIVEN = "Significant Wave Height Hs (given)"
    HM0_GIVEN = "Significant Wave Height Hm0 (given)"

    # Probe ratios (adjacent pairs — always computed)
    P2_P1_FFT = "P2/P1 (FFT)"
    P3_P2_FFT = "P3/P2 (FFT)"
    P4_P3_FFT = "P4/P3 (FFT)"
    # Semantic ratio — outgoing / incoming wave (config-dependent probe numbers)
    OUT_IN_FFT = "OUT/IN (FFT)"


class CalculationResultColumns:
    """Column names returned by calculation functions.

    These are the keys in DataFrames returned by functions like
    calculate_wavedimensions() and need to be mapped to metadata columns.
    """

    # Individual column names
    WAVELENGTH = "Wavelength"
    KL = "kL"
    KA = "ka"
    KH = "kH"
    TANH_KH = "tanh(kH)"
    CELERITY = "Celerity"
    FROUDE = "Froude"
    WIND_CELERITY = "Wind/Celerity"
    F_PM_RATIO = "f/f_PM"
    URSELL = "Ursell"

    # Pre-built column lists for bulk operations
    WAVE_DIMENSION_COLS = ["Wavelength", "kL", "ka", "tanh(kH)", "Celerity"]
    WAVE_DIMENSION_COLS_WITH_KH = ["Wavelength", "kL", "ka", "kH", "tanh(kH)", "Celerity"]
    PHYSICS_COLS = ["Froude", "Wind/Celerity", "f/f_PM", "Ursell"]


class ColumnGroups:
    """Pre-computed column name lists for bulk operations.

    These are computed once at module load time for performance.
    Use these instead of calling helper functions repeatedly.
    """

    # Position-independent columns (these don't depend on probe arrangement)
    # "Expected *" = computed from WaveFrequencyInput; per-probe "(FFT)" columns use measured frequency
    GLOBAL_WAVE_DIMENSION_COLS = ["Expected Wavelength", "Expected kL", "Expected ka", "Expected kH", "Expected tanh(kH)", "Expected Celerity"]
    PROBE_RATIO_COLS = ["P2/P1 (FFT)", "P3/P2 (FFT)", "P4/P3 (FFT)", "OUT/IN (FFT)"]

    @staticmethod
    def fft_wave_dimension_cols(pos: str) -> list[str]:
        """FFT-derived wave dimension columns for a probe identified by position string."""
        return [
            f"Probe {pos} Wavelength (FFT)",
            f"Probe {pos} kL (FFT)",
            f"Probe {pos} ka (FFT)",
            f"Probe {pos} tanh(kH) (FFT)",
            f"Probe {pos} Celerity (FFT)",
        ]

    @staticmethod
    def all_probe_cols_for_category(category: str, cfg) -> list[str]:
        """Get all metadata columns for a category, keyed by physical probe position.

        Args:
            category: One of 'amplitude', 'fft', 'psd', 'setup'
            cfg: ProbeConfiguration for the dataset
        Returns:
            Flat list of column names using position strings (e.g. 'Probe 9373/170 Amplitude')
        """
        positions = list(cfg.probe_col_names().values())
        category_map = {
            'amplitude': [f"Probe {p} Amplitude" for p in positions],
            'fft': (
                [f"Probe {p} Amplitude (FFT)" for p in positions] +
                [f"Probe {p} Frequency (FFT)" for p in positions] +
                [f"Probe {p} WavePeriod (FFT)" for p in positions]
            ),
            'psd': (
                [f"Probe {p} Amplitude (PSD)" for p in positions] +
                [f"Probe {p} Swell Amplitude (PSD)" for p in positions] +
                [f"Probe {p} Wind Amplitude (PSD)" for p in positions]
            ),
            'setup': (
                [f"Stillwater Probe {p}" for p in positions] +
                [f"Probe {p} mm from paddle" for p in positions]
            ),
        }
        return category_map.get(category.lower(), [])


# NOTE: RawDataColumns and ProcessedDataColumns have been removed.
# Probes are named by physical position at load time (e.g. 'Probe 9373/170'),
# and eta columns follow the same convention (e.g. 'eta_9373/170').
# Use ProbeConfiguration.probe_col_names() to get position strings for the active setup.


# =============================================================================
# COLUMN NAME HELPERS
# =============================================================================

def get_probe_column(probe_num: int, column_template: str) -> str:
    """Get a specific probe's column name from a template.

    Args:
        probe_num: Probe number (1-4)
        column_template: Template string with {i} placeholder

    Returns:
        Formatted column name

    Example:
        >>> get_probe_column(2, ProbeColumns.AMPLITUDE_FFT)
        'Probe 2 Amplitude (FFT)'
    """
    if not 1 <= probe_num <= MEASUREMENT.NUM_PROBES:
        raise ValueError(f"Probe number must be 1-{MEASUREMENT.NUM_PROBES}, got {probe_num}")
    return column_template.format(i=probe_num)


def validate_columns_exist(df, required_columns: list[str], context: str = "") -> None:
    """Validate that all required columns exist in a DataFrame.

    Args:
        df: DataFrame to check
        required_columns: List of column names that must exist
        context: Description of where this check is happening (for error message)

    Raises:
        ValueError: If any required columns are missing
    """
    missing = set(required_columns) - set(df.columns)
    if missing:
        ctx = f" in {context}" if context else ""
        raise ValueError(f"Missing required columns{ctx}: {sorted(missing)}")


# =============================================================================
# VALIDATION
# =============================================================================

def validate_column_constants():
    """Sanity check column name constants."""

    # Check that ProbeColumns templates format correctly with a position string
    test_col = ProbeColumns.AMPLITUDE.format(i="9373/170")
    assert test_col == "Probe 9373/170 Amplitude", f"Unexpected: {test_col}"

    # GLOBAL_WAVE_DIMENSION_COLS and PROBE_RATIO_COLS are position-independent
    assert len(ColumnGroups.GLOBAL_WAVE_DIMENSION_COLS) > 0
    assert len(ColumnGroups.PROBE_RATIO_COLS) > 0

    print("✓ All column constants validated")


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

    print("\n=== COLUMN NAMES ===")
    print(f"Number of probes: {len(ColumnGroups.BASIC_AMPLITUDE_COLS)}")
    print(f"Sample amplitude column: {ProbeColumns.AMPLITUDE.format(i=1)}")
    print(f"FFT amplitude columns: {ColumnGroups.FFT_AMPLITUDE_COLS}")
    print(f"Global columns sample: {GlobalColumns.PATH}, {GlobalColumns.WATER_DEPTH}")

    validate_constants()
