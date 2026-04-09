#!/usr/bin/env python3
"""
Save publication-quality figures and tables for the thesis.

Run from terminal:
    conda activate draumkvedet
    cd /Users/ole/Kodevik/wave_project
    python main_save_figures.py

Each section corresponds to a thesis chapter and figure/table number.
Set save_plot=True (or call save_and_stub) when a figure is ready to export.
Requires processed cache. Run main.py first if stale.

INPUT KEYS  (experimental conditions):
    WaveAmplitudeInput [Volt]   — paddle drive voltage (0.1 V / 0.2 V)
    WaveFrequencyInput [Hz]     — paddle frequency (0.65–1.9 Hz)
    PanelCondition              — full / reverse / no
    WindCondition               — full / lowest / no

OUTPUT KEYS (measured results):
    OUT/IN (FFT)                — damping ratio, FFT amplitude at paddle freq only
                                  (wind waves excluded — they are characterised separately)
    ka                          — wavenumber × amplitude, found per probe per run
                                  (not pre-calculated; measured from the actual wave)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FIGURE INDEX
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Status legend:  ✓ ready   ~ draft (DRAFT stamp)   ✗ placeholder (blank fig)

CHAPTER 04 — METHODOLOGY
  §1   ch04_probe_noise_floor      ~  Stillwater noise floor per probe / hw config
  §2   ch04_stillwater_timing      ✗  Swell decay time vs wait time  [TODO]
  §3   ch04_parallel_ratio         ~  Wall/far-side amplitude ratio vs frequency
  §3b  ch04_probe_height           ✗  Probe height validity range    [TODO]
  §3c  ch04_sound_speed            ~  Speed-of-sound / lab temperature drift
  §4-1 ch04_wind_psd               ~  Wind PSD per probe (nowave runs)
  §4-2 ch04_wind_reflection        ✗  Wind reflection from panel     [TODO]
  §4-3 ch04_fft_wave               ~  FFT spectrum at paddle freq (1.3 Hz example)
  §4-4 ch04_wind_snr               ~  Spectral SNR: paddle / wind noise per probe
  §4-5 ch04_td_vs_fft              ~  A_td vs A_FFT: why FFT is required
  §5   ch04_timeseries_overview    ~  Full time-series with stable-window band
  §6   ch04_first_arrival          ~  First wave arrival vs probe distance
  §7   ch04_wave_stability         ~  Wave stability and period_cv vs frequency
  §8   ch04_lateral_nowind         ~  Lateral equality (parallel ratio, no-wind)
  §9   ch04_amplitude_profile      ~  Amplitude at every probe, all runs

CHAPTER 05 — RESULTS
  §1   ch05_damping_freq           ✓  OUT/IN (FFT) vs frequency  ← primary result
  §2   ch05_damping_scatter        ✓  OUT/IN scatter vs amplitude
  §3   ch05_damping_wind_delta     ✗  Wind effect on damping (delta plot)  [TODO]
  §4   ch05_damping_ka             ✗  Damping vs ka (wavenumber × amplitude) [TODO]
  §5   ch05_swell_scatter          ~  IN vs OUT by swell/wind/total band
  §6   ch05_reconstructed          ~  FFT-reconstructed paddle signal

DIAGNOSTICS
  D1   diag_13hz_consistency        ✗  1.3 Hz cross-session consistency check [TODO]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# %% ── dev: reload modules (run this cell after editing wavescripts/) ─────────
import importlib, wavescripts.plotter as _pm, wavescripts.filters as _fm

importlib.reload(_pm); importlib.reload(_fm);

from wavescripts.plotter import (plot_probe_noise_floor, plot_parallel_ratio,
                                  plot_frequency_spectrum, plot_wave_stability,
                                  plot_timeseries_overview,
                                  plot_damping_freq, plot_damping_scatter,
                                  plot_damping_wind_delta)
from wavescripts.filters import apply_experimental_filters as _aef


# %% ----------- Velkommen ----------------------------
import os
from datetime import datetime as _dt
from pathlib import Path

import time

import matplotlib
matplotlib.use("Agg")   # non-interactive — plt.show() is a no-op; avoids hanging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from wavescripts.constants import MEASUREMENT
from wavescripts.filters import (
    apply_experimental_filters,
    damping_all_amplitude_grouper,
    filter_for_frequencyspectrum,
)
from wavescripts.improved_data_loader import load_analysis_data, load_processed_dfs, ANALYSIS_PROBES, get_configuration_for_date
from wavescripts.plot_utils import (WIND_COLOR_MAP, apply_thesis_style,
                                    add_draft_stamp, build_fig_meta, save_and_stub)
from wavescripts.plotter import (
    plot_all_probes,
    plot_damping_freq,
    plot_damping_scatter,
    plot_damping_wind_delta,
    plot_first_arrival,
    plot_frequency_spectrum,
    plot_parallel_ratio,
    plot_probe_noise_floor,
    plot_reconstructed,
    plot_sound_speed,
    plot_swell_scatter,
    plot_td_vs_fft,
    plot_timeseries_overview,
    plot_wave_stability,
    plot_wind_snr,
)
from wavescripts.wave_detection import find_first_arrival
# %%
FS = MEASUREMENT.SAMPLING_RATE

try:
    file_dir = Path(__file__).resolve().parent
except NameError:
    file_dir = Path.cwd()
os.chdir(file_dir)

# ── Dataset(s) ────────────────────────────────────────────────────────────────
# Only the two most recent folders active — most reliable data (lowrange, h100, mooring30).
# Earlier folders have interpolation artefacts at high frequencies (≥1.6 Hz).
PROCESSED_DIRS = [
    # ── Nov 2025: probe 1 at 18000 mm, roof not fully sealed ──────────────────
    # Path("waveprocessed/PROCESSED-20251005-sixttry6roof-highMooring"),        # probe 1 på 18000 mm; tak ikke tetta helt
    # ── Nov 2025: probe 1 moved to 8804 mm, lowMooring ────────────────────────
    # Path("waveprocessed/PROCESSED-20251110-tett6roof-lowM-ekte580"),          # mange kjøringer med -per15
    # Path("waveprocessed/PROCESSED-20251110-tett6roof-lowMooring"),            # noen kjøringer med -per30
    # Path("waveprocessed/PROCESSED-20251110-tett6roof-lowMooring-2"),          # et par kjøringer med -per15
    # Path("waveprocessed/PROCESSED-20251112-tett6roof"),
    # Path("waveprocessed/PROCESSED-20251113-tett6roof"),
    # Path("waveprocessed/PROCESSED-20251113-tett6roof-loosepaneltaped"),
    # Path("waveprocessed/PROCESSED-20251113-tett6roof-probeadjusted"),
    # ── Mar 2026: new probe positions (march2026_rearranging config) ───────────
    # Path("waveprocessed/PROCESSED-20260305-newProbePos-tett6roof"),           # in=9373/170, out=11800/250 — transitional
    # Path("waveprocessed/PROCESSED-20260306-newProbePos-tett6roof"),           # in=9373/170, out=11800/250 — transitional
    # ── Mar 2026: final probe positions (march2026_better_rearranging) ─────────
    # Path("waveprocessed/PROCESSED-20260307-ProbPos4_31_FPV_2-tett6roof"),    # disse bør være greie bortsett fra de steile
    # Path("waveprocessed/PROCESSED-20260312-ProbPos4_31_FPV_2-tett6roof"),    # noen filer med FALSEDATE (riktig dato fra mappe)
    # Path("waveprocessed/PROCESSED-20260313-ProbePos4_31_FPV_2-tett6roof"),   # noen filer med FALSEDATE?
    # Path("waveprocessed/PROCESSED-20260314-ProbePos4_31_FPV_2-tett6roof"),   # noen filer med FALSEDATE?
    # Path("waveprocessed/PROCESSED-20260316-ProbePos4_31_FPV_2-tett6roof"),   # noen filer med FALSEDATE?
    # Path("waveprocessed/PROCESSED-20260316-ProbePos4_31_FPV_2-tett6roof-under9Mooring"),
    # Path("waveprocessed/PROCESSED-20260319-ProbePos4_31_FPV_2-tett6roof-under9Mooring"),
    # Path("waveprocessed/PROCESSED-20260321-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-RENAMED"),
    # ── Mar 2026: probe lowered — height136 (transitional, 1 dag) ─────────────
    # Path("waveprocessed/PROCESSED-20260323-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height136"),  # h136/high, 1 dag
    # ── Mar 2026: probe lowered to height100 ──────────────────────────────────
    # Path("waveprocessed/PROCESSED-20260323-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100"),
    # Path("waveprocessed/PROCESSED-20260324-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100"),
    # Path("waveprocessed/PROCESSED-20260325-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100"),
    # Path("waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100"),
    # ── Mar 2026: lowrange switch enabled ─────────────────────────────────────
    Path("waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange"),
    Path("waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"),
]

# Register dataset names globally so every stub's immutable block records them.
import wavescripts.plot_utils as _pu
_pu.ACTIVE_DATASETS = [p.name for p in PROCESSED_DIRS]

# ── Load from cache ───────────────────────────────────────────────────────────
print("Loading analysis data...")
combined_meta, _, combined_fft_dict, combined_psd_dict = load_analysis_data(
    *PROCESSED_DIRS
)

# %%
# processed_dfs is heavy (~75 MB). Loaded when needed by sections below.
processed_dfs = load_processed_dfs(*PROCESSED_DIRS)


# ── Placeholder helper ────────────────────────────────────────────────────────
def _save_placeholder(figure_name: str, section_label: str, chapter: str) -> None:
    """Save a red-stamped DRAFT placeholder for a not-yet-implemented figure."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.text(0.5, 0.55, section_label, ha="center", va="center",
            transform=ax.transAxes, fontsize=13, fontweight="bold", color="#444")
    ax.text(0.5, 0.38, "Not yet implemented", ha="center", va="center",
            transform=ax.transAxes, fontsize=10, color="#888", style="italic")
    ax.set_axis_off()
    meta = build_fig_meta(
        {"filters": {}, "plotting": {
            "figure_name": figure_name,
            "draft": True,
            "caption": f"PLACEHOLDER — {section_label}. Not yet implemented.",
        }},
        chapter=chapter,
    )
    save_and_stub(fig, meta, plot_type=figure_name, force_stub=True)
    plt.close(fig)


# =============================================================================
# CHAPTER 04 — METHODOLOGY
# =============================================================================
# %%

# %%
"""
── CH04 § 1 — Probe uncertainty / noise floor ───────────────────────────────
Goal: show the stillwater noise amplitude per probe and hardware configuration,
and derive the minimum detectable wave amplitude (detection threshold).

Three questions answered per (probe, config):
  1. Precision  — how much does the reading fluctuate in still water?
  2. Bias       — do probes agree on the mean water level within a config?
  3. Threshold  — what is the smallest detectable wave amplitude?

Data: combined_meta stillwater rows (WindCondition=="no", WaveFrequencyInput NaN).
      processed_dfs needed for quantization_step_mm (optional but recommended).

Groups: probe_height_mm × probe_range_mode — 4 hardware configurations:
  h272/high  (default pre-2026-03-23)
  h136/high
  h100/high
  h100/low

Metrics (all from combined_meta, shift-invariant — valid at any probe height):
  noise_95pct_amp_mm   (P97.5−P2.5)/2   mean across accepted runs in group
  noise_rms_mm         std(raw signal)   mean across accepted runs in group
  mean_level_mm        median level      mean across accepted runs in group
  bias_vs_ref_mm       mean_level − cross-probe mean (within group)
  quantization_step_mm P5 of nonzero |diff(η)|  from processed_dfs
  detection_threshold_mm  max(k_sigma·σ,  k_q·q)   default max(3σ, 2q)
"""

from datetime import datetime as _dt
from wavescripts.improved_data_loader import get_configuration_for_date
# Probe numbers derived from current config (hardware IDs, fixed across configs):
_active_cfg = get_configuration_for_date(_dt(2026, 3, 15))
_PROBE_NUM_MAP = {pos: num for num, pos in _active_cfg.probe_col_names().items()}

_pv_noise_floor = {
    "filters": {},
    "plotting": {
        "show_plot": True,
        "save_plot": True,            # DRAFT — noise floor plot not yet polished
        "draft":     True,
        "figure_name": "ch04_probe_noise_floor",
        "force_stub": True,
        "caption": (
            "Stillwater 95\\% noise amplitude $(P_{{97.5}} - P_{{2.5}})/2$ per "
            "ultrasound wave gauge, with no waves and no wind. "
            "Each panel shows one hardware configuration "
            "(probe height above still water / range mode). "
            "Blue bars: mean across accepted stillwater runs within each configuration "
            "(error bars: \\pm 1\\,std). "
            "White dots: individual run values. "
            "Quantization step $q = 0.03$--$0.05$\\,mm per probe "
            "(P5 of nonzero sample-to-sample differences); "
            "Dashed red line: detection threshold $\\max(3\\,\\sigma,\\; 2\\,q)$ "
            "per probe, where $\\sigma$ is the rms noise; "
            "wave amplitudes below this line are indistinguishable from stillwater noise."
        ),
    },
}

start = time.perf_counter()
_figs_nf, _noise_summary = plot_probe_noise_floor(
    combined_meta, ANALYSIS_PROBES, _pv_noise_floor,
    group_by=["probe_height_mm", "probe_range_mode"],
    processed_dfs = processed_dfs,
    highlight_keyword="wavemakeroff-1hour",  # visual star only, no effect on metrics
    probe_number_map=_PROBE_NUM_MAP,
)
print("\n=== Probe noise floor summary [mm] ===")
print(_noise_summary.round(4).to_string())
end = time.perf_counter()
print(f"probe uncertainty-plot took {end - start:.4f} s")






# %%
"""
── CH04 § 2 — Stillwater timing (how long to wait between runs) ─────────────
Goal: show that long-wave swell from previous runs decays over time, and that
wind dramatically shortens the required waiting time.

Data: repeated stillwater runs at different times after wave runs; look at
low-frequency PSD content in eta_* columns over time.

Figures:
  - Plot:  PSD of eta at the OUT probe vs time-after-wave (semi-log, low freqs)
  - Note:  wind-only runs show near-immediate settling(return to wind-wave spectrum) — physical explanation
           (wind chops suppress long-wave coherence in the tank).
"""
# TODO: implement stillwater timing figure
_save_placeholder("ch04_stillwater_timing", "CH04 §2 — Stillwater timing", chapter="04")


# %%
"""
── CH04 § 3 — Probe placement: longitudinal and lateral effects ─────────────
Goal: show what parallel probes tell us — lateral uniformity without wind,
lateral asymmetry with wind. Also: why the longitudinal positions were chosen.

Data: combined_meta, parallel_ratio column, no-wind wave runs. But,
the probes placed downstream are to be trusted more, because no interference from mooring and panel.

Figures:
  - Plot:  parallel_ratio vs frequency, coloured by WindCondition
  - Plot:  parallel_ratio vs frequency, coloured by PanelCondition (reflection)
  - Table: parallel_ratio summary (mean, std) by wind/panel group
"""

""" PRINTOUT
Ratio of wall-side to far-side probe amplitude at the same longitudinal distance, for 154 wave runs across 1 panel configurations. A ratio of 1 indicates lateral symmetry. Deviations indicate wall reflections or wind-driven lateral asymmetry. Error bars: standard deviation across runs at the same frequency. Dashed line: ratio = 1.
"""

_pv_parallel_ratio = {
    "filters": {},
    "plotting": {
        "show_plot": True,
        "save_plot": True,            # DRAFT — parallel ratio not yet polished
        "draft":     True,
        "figure_name": "ch04_parallel_ratio",
        "force_stub": True,
        "caption": (
            "Ratio of wall-side to far-side probe amplitude at the same longitudinal "
            "distance, for {n_runs} wave runs across {n_panels} panel condition(s) "
            "({panel_conditions}). "
            "A ratio of 1 indicates lateral symmetry. "
            "Deviations indicate wall reflections or wind-driven lateral asymmetry. "
            r"Error bars: standard deviation across runs at the same frequency. "
            "Dashed line: ratio = 1."
        ),
    },
}
start = time.perf_counter()
_fig_pr = plot_parallel_ratio(combined_meta, _pv_parallel_ratio)
end = time.perf_counter()
print(f"Lateral symmetry of plot_parallel_ratio {end-start:.4f} seconds")

# %%
"""
── CH04 § 3b — Probe height: validity range and the 100 mm lowering ─────────
Goal: explain and quantify the effect of probe height above the water surface.

Background:
  - All runs up to 2026-03-23: probe tips ~272 mm above water surface.
    This was fine for moderate wave amplitudes, but for steep/high waves the
    probe tips risked leaving the water surface on the wave troughs.
  - 2026-03-23: probes lowered to ~100 mm above water.

What to show:
  1. At what wave amplitude / frequency does a 272 mm height become problematic?
     Compute: if trough drops below -272 mm from stillwater → probe tip exits water.
     Compare against measured wave amplitudes to find the steepness limit.
  2. Plot: OUT/IN ratio vs amplitude, coloured by probe height configuration.
     If the 272 mm runs are consistent with the 100 mm runs at moderate amplitude,
     the earlier data is valid. If they diverge at high amplitude → flag those runs.
  3. Data: probe_height column in combined_meta (from mooring/config — check if this
     needs to be added as a new config field, or derive from run date).

TODO: needs data from both height configurations to make the comparison.
      100 mm data was collected 2026-03-23 onward.
      Earliest comparison runs need to be identified.
"""
# TODO: implement plot_probe_height_comparison() in plotter.py when data is ready
_save_placeholder("ch04_probe_height", "CH04 §3b — Probe height validity range", chapter="04")

# %%
"""
── CH04 § 3c — Speed-of-sound / lab temperature ─────────────────────────────
Goal: show that lab temperature variation introduces < 0.4 % amplitude scale
error, and that this cancels exactly for OUT/IN ratios.
Data: sound_speed_mean_ms / sound_speed_std_ms in combined_meta (pipeline).
"""

_pv_sound_speed = {
    "filters": {},
    "plotting": {
        "show_plot":   True,
        "save_plot":   True,            # DRAFT — not yet polished
        "draft":       True,
        "figure_name": "ch04_sound_speed",
        "force_stub":  True,
        "figsize":     (10, 3),
    },
}

plot_sound_speed(combined_meta, _pv_sound_speed, chapter="04")

# _pv_probe_height = {
#     "filters": {"run_category": "standard"},
#     "plotting": {
#         "show_plot": False,
#         "save_plot": False,
#         "figure_name": "ch04_probe_height",
#         "caption": "TODO",
#     },
# }

# %%
"""
── CH04 § 4-1 — Wind characterisation ─────────────────────────────────────────
Goal: characterise what the wind does to the water surface — spectrum, spatial
extent, interaction with the panel.

Subtopics:
  4a. Wind-wave PSD at each probe (broadband, 2–10 Hz dominant)
  4b. Wind-only amplitude vs probe position (SNR context)
  4c. Wind-only amplitude: IN probe (~10 mm) vs OUT probe (~0.9 mm) —
      panel attenuates wind waves almost completely at 12400 mm
  4d. Lateral coherence: cross-correlate /170 and /340 at same distance
      (coherent = tank-wide fetch; incoherent = local turbulence)

Data: combined_psd_dict (nowave entries), nowave+fullwind rows of combined_meta.

Figures:
  - Plot:  wind PSD per probe, fullwind vs stillwater overlay (log y-axis)
  - Plot:  wind-only amplitude vs longitudinal distance, bar per probe
  - Plot:  cross-correlation coefficient /170 vs /340 for fullwind runs
"""
from wavescripts.filters import apply_experimental_filters as _aef

_pv_wind_psd = {
    "filters": {
        "WaveFrequencyInput [Hz]": None,
        "WindCondition":           None,
        "PanelCondition":          None,
        # exclude diagnostic/experimental runs by filename keyword
        "exclude_run_keywords": ["nestenstille", "mstop"],
    },
    "plotting": {
        "show_plot":     True,
        "save_plot":     True,          # set True when ready
        "figure_name":   "ch04_wind_psd",
        "force_stub":    True,
        "figsize":       (11, 4 * 4),
        "linewidth":     1.0,
        "facet_by":      "probe",
        "probes":        ANALYSIS_PROBES,
        "xlim":          (0, 5),
        "logaritmic":    False,
        "peaks":         0,
        "max_points":    500,
        "grid":          True,
        "legend":        "inside",

        "caption": (
            "POWER spectral density (PSD) of the free surface at each wave gauge "
            "during wind-only runs (no paddle waves). "
            "All {n_runs} nowave runs overlaid; colour encodes wind condition. "
            "Stillwater runs (no wind) shown as baseline. "
            "Wind energy is concentrated above 2\\,Hz — "
            "the paddle frequency range (0.65--1.9\\,Hz) is unaffected."
        ),
    },
}

_meta_nowave_all = combined_meta[combined_meta["WaveFrequencyInput [Hz]"].isna()].copy()
_meta_nowave     = _aef(_meta_nowave_all, _pv_wind_psd)
_nowave_paths    = set(_meta_nowave["path"])
_wind_psd_dict   = {k: v for k, v in combined_psd_dict.items() if k in _nowave_paths}

start = time.perf_counter()
_fig_wind_psd, _ = plot_frequency_spectrum(
    _wind_psd_dict, _meta_nowave, _pv_wind_psd, data_type="psd", chapter="04"
)
end = time.perf_counter()
print(f"Wind PSD plot took {end - start:.4f} s")



# %%
"""
── CH04 § 4-2 — Wind wave-reflection from panel ─────────────────────────────────────────
Goal: find out the reflection — spectrum, spatial
extent, interaction with the panel.

Data: combined_psd_dict (nowave entries), nowave+fullwind rows of combined_meta.

Figures:
  - Plot:
"""
# TODO: implement wind reflection figure
_save_placeholder("ch04_wind_reflection", "CH04 §4-2 — Wind reflection from panel", chapter="04")

# %%
"""
── CH04 § 4-3 — FFT spectrum: paddle frequency peak ────────────────────────
Goal: show what the FFT looks like for a wave run — narrow peak at the paddle
frequency, wind condition overlaid. Motivates using FFT amplitude (not
time-domain) for OUT/IN. One representative frequency (e.g. 1.3 Hz).

Data: combined_fft_dict, wave runs.

Figures:
  - plot_frequency_spectrum with data_type="fft", facet_by="probe"
"""

_pv_fft_wave = {
    "filters": {
        "WaveAmplitudeInput [Volt]": 0.2,
        "WaveFrequencyInput [Hz]":   1.3,
        "WindCondition":             None,
        "PanelCondition":            "full",
        "run_category":              "standard",
    },
    "plotting": {
        "show_plot":   True,
        "save_plot":   True,           # DRAFT — FFT wave example not yet polished
        "draft":       True,
        "figure_name": "ch04_fft_wave",
        "force_stub":  True,
        "figsize":     (11, 4 * 4),
        "linewidth":   0.8,
        "facet_by":    "probe",
        "probes":      ANALYSIS_PROBES,
        "xlim":        (0, 5),
        "logaritmic":  False,
        "peaks":       3,
        "max_points":  500,
        "grid":        True,
        "legend":      "inside",
        "caption": (
            "FFT amplitude spectrum of the free surface during wave runs "
            "(paddle frequency 1.3\\,Hz, amplitude 0.2\\,V, full panel). "
            "Each panel shows one probe; colour encodes wind condition. "
            "The narrow paddle-frequency peak is the target signal used "
            "for OUT/IN ratio computation."
        ),
    },
}

_fft_wave_meta = _aef(combined_meta, _pv_fft_wave)
_fft_wave_paths = set(_fft_wave_meta["path"])
_fft_wave_dict  = {k: v for k, v in combined_fft_dict.items() if k in _fft_wave_paths}

_fig_fft_wave, _ = plot_frequency_spectrum(
    _fft_wave_dict, _fft_wave_meta, _pv_fft_wave, data_type="fft", chapter="04"
)

# %%
"""
── CH04 § 4-4 — Spectral SNR: paddle signal vs wind noise ───────────────────
Goal: quantify how much of the FFT amplitude at paddle frequencies is
wind noise. SNR < 5 = unreliable; SNR < 3 = dominated by wind.
Data: combined_meta (wave runs + FFT amplitudes) + combined_psd_dict (nowave PSDs).
TODO: ... the 1.3 hz wave is noticably different.. but why, this is the run i have waay more data on .somehing is wrong.
"""

_pv_wind_snr = {
    "filters": {
        "WaveAmplitudeInput [Volt]": None,
        "WaveFrequencyInput [Hz]":   None,
        "WindCondition":             None,
        "PanelCondition":            None,
    },
    "plotting": {
        "show_plot":      True,
        "save_plot":      True,         # DRAFT — not yet polished
        "draft":          True,
        "figure_name":    "ch04_wind_snr",
        "force_stub":     True,
        "probes":         ANALYSIS_PROBES,
        "fft_window_hz":  0.1,
    },
}

plot_wind_snr(combined_meta, combined_psd_dict, _pv_wind_snr, chapter="04")

# %%
"""
── CH04 § 4-5 — Time-domain vs FFT amplitude: why A_FFT is required ─────────
Goal: demonstrate that time-domain amplitude is wind-dominated at the IN probe
under full wind, making OUT/IN from A_td meaningless. FFT amplitude isolates
the paddle frequency and is unaffected by broadband wind energy.
Data: combined_meta wave rows (Probe {pos} Amplitude and Probe {pos} Amplitude (FFT)).
"""

_pv_td_vs_fft = {
    "filters": {
        "WaveAmplitudeInput [Volt]": None,
        "WaveFrequencyInput [Hz]":   None,
        "WindCondition":             None,
        "PanelCondition":            None,
    },
    "plotting": {
        "show_plot":   True,
        "save_plot":   True,            # DRAFT — not yet polished
        "draft":       True,
        "figure_name": "ch04_td_vs_fft",
        "force_stub":  True,
        "probes":      ANALYSIS_PROBES,
    },
}

plot_td_vs_fft(combined_meta, _pv_td_vs_fft, chapter="04")

# %%
"""
── CH04 § 5 — What does a full signal look like? ────────────────────────────
Goal: show the full signal for a select few runs — stillwater baseline,
wavemaker ramp, stable wavetrain, decay. Wind-wave noise visible at IN probe
vs clean signal at OUT probe.

Layout: rows = probes, columns = runs selected by filters.
Grey band = detected stable-window used for all amplitude/FFT analysis.
"""

_pv_timeseries = {
    "filters": {
        # Pick a representative condition — adjust as needed:
        "WaveFrequencyInput [Hz]": 1.3,
        "WaveAmplitudeInput [Volt]": 0.2,
        "WindCondition": None,      # None = all wind conditions
        "PanelCondition": "full",
        # "run_category": "standard",
    },
    "plotting": {
        "show_plot":   True,
        "save_plot":   True,           # DRAFT — timeseries overview not yet polished
        "draft":       True,
        "figure_name": "ch04_timeseries_overview",
        "force_stub":  True,
        "probes":      ["9373/170", "12400/250"],   # IN and OUT only
        "max_runs":    4,           # cap columns; reduce if too crowded
        "xlim":        None,        # e.g. (0, 60) to zoom; None = full run
        "ylim":        None,        # e.g. (-30, 30); None = auto per row
        # caption printed on first run — paste the one-liner here:
        # "caption": "...",
    },
}

_fig_ts = plot_timeseries_overview(combined_meta, processed_dfs, _pv_timeseries)

# %%
"""
── CH04 § 6 — Wave-range detection ──────────────────────────────────────────
Goal: explain and validate _SNARVEI_CALIB. Show how the stable wavetrain
window is detected: (1) threshold crossing, (2) ramp-up skip, (3) n periods.

Data: processed_dfs, Computed Probe {pos} start/end columns.

Figures:
  - Plot:  single run with detected start/end marked, one probe panel per row
  - Plot:  start sample vs frequency (all probes) — show _SNARVEI_CALIB points
"""

_pv_first_arrival = {
    "filters": {},
    "plotting": {
        "show_plot":        True,
        "save_plot":        True,       # DRAFT — threshold not yet calibrated
        "draft":            True,
        "figure_name":      "ch04_first_arrival",
        "force_stub":       True,
        "probes":           ANALYSIS_PROBES,
        "threshold_factor": 5.0,        # TODO: calibrate per-probe after noise floor analysis
        "window_s":         2.5,
        "min_arrival_s":    0.5,
        "figsize":          (9, 5),
    },
}

plot_first_arrival(combined_meta, processed_dfs, _pv_first_arrival, chapter="04")

# %%
"""
── CH04 § 7 — Autocorrelation A: wavetrain stability ────────────────────────
Goal: show wave_stability and period_cv as quality metrics. Demonstrate that
fullwind + low amplitude (0.1 V) degrades IN probe stability, while OUT probe
stays clean.

Data: combined_meta, wave_stability {pos} and period_cv {pos} columns.

Figures:
  - Plot:  wave_stability vs frequency, faceted by probe, coloured by wind
  - Plot:  period_cv vs frequency, same layout
  - Note:  this motivates use of FFT amplitude (not time-domain) for OUT/IN
"""

_pv_wave_stability = {
    "filters": {
        "WaveAmplitudeInput [Volt]": None,
        "WaveFrequencyInput [Hz]":   (0.9,1.6),
        "WindCondition":             None,
        "PanelCondition":            "full",
        # "run_category": "standard",   # re-enable after --force-recompute
    },
    "plotting": {
        "show_plot":   True,
        "save_plot":   True,          # DRAFT — wave stability not yet polished
        "draft":       True,
        "figure_name": "ch04_wave_stability",
        "force_stub":  True,
        "figsize":     (10, 3.5),
        "probes":      ANALYSIS_PROBES,
        # caption printed to terminal on first run — paste the one-liner here:
        # "caption": "...",
    },
}

_fig_stab = plot_wave_stability(combined_meta, ANALYSIS_PROBES, _pv_wave_stability)

# %%
"""
── CH04 § 8 — Autocorrelation B: lateral wave equality ──────────────────────
Goal: show that the paddle wave is laterally uniform (parallel probes agree)
under no-wind conditions, and that full wind introduces lateral asymmetry.

Data: combined_meta, parallel_ratio column, wave_stability columns.

Figures:
  - Plot:  parallel_ratio vs frequency, no-wind runs (should be ~1.0)
  - Plot:  parallel_ratio vs frequency, fullwind runs (asymmetry visible?)
  - Table: mean parallel_ratio ± std by (WindCondition, frequency)
"""

# Lateral equality uses the same plot_parallel_ratio function (already defined in §3),
# but filtered to a single wind condition at a time for the per-wind breakdown.
_pv_lateral_nowind = {
    "filters": {"WindCondition": "no", "run_category": "standard"},
    "plotting": {
        "show_plot":   True,
        "save_plot":   True,          # DRAFT — lateral equality not yet polished
        "draft":       True,
        "figure_name": "ch04_lateral_nowind",
        "force_stub":  True,
        "caption": (
            "Wall-side to far-side amplitude ratio at matched longitudinal distance, "
            "no-wind runs only. A ratio of 1 indicates the paddle wave is laterally "
            r"uniform. Dashed line: ratio = 1."
        ),
    },
}
_fig_lat_nw = plot_parallel_ratio(combined_meta, _pv_lateral_nowind)

# %% - perhaps skip this one. its the physical plot.
# """
# ── CH04 § 9 — Amplitude profile across all probes ───────────────────────────
# Goal: show measured amplitude at each probe position for all runs, giving a
# # physical overview of how wave energy is distributed along the tank.
# Colour = wind condition, linestyle = panel condition.
# Data: combined_meta wave rows, all Probe {pos} Amplitude columns.
# """

# _pv_all_probes = {
#     "filters": {
#         "WaveAmplitudeInput [Volt]": None,
#         "WaveFrequencyInput [Hz]":   None,
#         "WindCondition":             None,
#         "PanelCondition":            None,
#     },
#     "plotting": {
#         "show_plot":   True,
#         "save_plot":   False,            # this one is mostly
#         "draft":       True,
#         "figure_name": "ch04_amplitude_profile",
#         "force_stub":  True,
#         "figsize":     (10, 6),
#         "annotate":    False,
#     },
# }

# _ap_meta = apply_experimental_filters(
#     combined_meta[combined_meta["WaveFrequencyInput [Hz]"].notna()], _pv_all_probes
# )
# plot_all_probes(_ap_meta, _pv_all_probes, chapter="04")

# =============================================================================
# CHAPTER 05 — RESULTS
# =============================================================================

# %%
"""
── CH05 § 1 — Damping overview: OUT/IN vs frequency ─────────────────────────
THE central result. "How much is left of the paddle-frequency wave after
travelling through the panel geometry?"

x-axis:  WaveFrequencyInput [Hz]  (or ka — wavenumber × amplitude)
y-axis:  OUT/IN (FFT) — always FFT-based, never time-domain
colour:  WindCondition  (no / lowest / full)
facets:  PanelCondition  ×  WaveAmplitudeInput [Volt]

Key question: does wind change the damping? If yes: how much, and at which
frequencies?

Data: combined_meta → damping_all_amplitude_grouper → plot_damping_freq.

Figures:
  - plot_damping_freq: OUT/IN vs freq, errorbars (std or ±10% for n=1)
  - Same plot with ka on x-axis (requires wavenumber column)
"""

_pv_damping_freq = {
    "filters": {
        "WaveAmplitudeInput [Volt]": (0.1,0.3),
        "WaveFrequencyInput [Hz]":   (0.8,1.7),
        "WindCondition":             None,
        "PanelCondition":            None,
    },
    "plotting": {
        "show_plot":  True,
        "save_plot":  True,          # set True when figure is ready for thesis
        "force_stub": True,
        "figure_name": "ch05_damping_freq",
        "figsize":    (7, 3),
        "annotate":   True,
        "legend":     "outside_right",
        "probes":     ANALYSIS_PROBES,
        "caption": (
            "Damping ratio OUT/IN (FFT amplitude at paddle frequency) versus wave frequency. "
            "Colour encodes wind condition ({wind_conds}); "
            "each line shows one amplitude ({amps}). "
            "Errorbars: standard deviation across repeated runs. "
            "Dashed line: ratio = 1 (no damping)."
        ),
    },
}

_damping_meta   = _aef(combined_meta, _pv_damping_freq)
_damping_grouped = damping_all_amplitude_grouper(_damping_meta)
plot_damping_freq(_damping_grouped, _pv_damping_freq)

# %%
"""
── CH05 § 2 — Damping vs amplitude ──────────────────────────────────────────
Secondary result. Is there an amplitude dependence? (Expected: small effect
at these steepnesses, but worth showing explicitly.)

x-axis:  WaveAmplitudeInput [Volt]
y-axis:  OUT/IN (FFT)
colour:  WaveFrequencyInput [Hz]
facets:  PanelCondition  ×  WindCondition

Figures:
  - plot_damping_scatter or similar
"""

_pv_damping_scatter = {
    "filters": {
        "WaveAmplitudeInput [Volt]": None,
        "WaveFrequencyInput [Hz]":   (0.1, 1.8),
        "WindCondition":             None,
        "PanelCondition":            None,
        # "run_category":            "standard",   # re-enable after --force-recompute
    },
    "plotting": {
        "show_plot":   True,
        "save_plot":   True,         # set True when figure is ready for thesis
        "figure_name": "ch05_damping_scatter",
        "force_stub":  True,
        "figsize":     (5, 4),
        "caption": "UT/INN damping ratio versus wave frequency, all amplitudes combined. all panel condition(s); colour = wind condition (full, no); marker size = wave amplitude (0.10\,V, 0.20\,V, 0.30\,V, 0.60\,V). Errorbars: standard deviation across runs."
}}

_scatter_meta   = _aef(combined_meta, _pv_damping_scatter)
_scatter_grouped = damping_all_amplitude_grouper(_scatter_meta)
plot_damping_scatter(_scatter_grouped, _pv_damping_scatter)

# %%
"""
── CH05 § 3 — Wind effect on damping ────────────────────────────────────────
The single key question of the thesis, isolated:
"Does adding wind increase or decrease damping by the panel?"

Show: OUT/IN (no wind) vs OUT/IN (full wind) at matched frequency/amplitude/panel.
Expected: wind may add energy at IN → apparent increase in damping if using
time-domain; FFT-based OUT/IN removes this artefact and shows the true effect.

Figures:
  - Plot:  delta-OUT/IN (full wind minus no wind) vs frequency
  - Table: OUT/IN summary — (frequency × panel) with wind as columns
"""

# _damping_grouped already computed in §1.
# TODO: add plot_damping_wind_delta() to plotter.py (pivot no/full wind, plot difference)
#
# Interim: the §1 plot already shows all three wind conditions on the same axes —
# the "wind effect" is readable directly from that figure. The delta plot is a
# cleaner standalone for the thesis.
_pv_damping_wind_delta = {
    "filters": {
        "WaveAmplitudeInput [Volt]": None,
        "WindCondition":             ["no", "full"],
        "PanelCondition":            None,
    },
    "plotting": {
        "show_plot":   True,
        "save_plot":   True,
        "draft":       True,
        "figure_name": "ch05_damping_wind_delta",
        "force_stub":  True,
        "figsize":     (6, 5),
        "ref_wind":    "no",
        "target_wind": "full",
    },
}

_wind_delta_meta    = _aef(combined_meta, _pv_damping_wind_delta)
_wind_delta_grouped = damping_all_amplitude_grouper(_wind_delta_meta)
plot_damping_wind_delta(_wind_delta_grouped, _pv_damping_wind_delta, chapter="05")

# %%
"""
── CH05 § 4 — Wave steepness: ka as axis variable ───────────────────────────
All damping plots should optionally show ka on the x-axis instead of Hz.

ka is not pre-calculated from dispersion — it is found per probe per run from
the measured wavenumber k (from FFT phase or zero-crossing period) and the
measured amplitude a at that probe position. This reflects the actual wave
seen by each probe, not the theoretical incident wave.

Note: the panel changes both amplitude AND ka between IN and OUT. The
frequency changes little, but amplitude can drop up to ~95%. Report both
IN-side ka and OUT-side ka.

Requires: wavenumber column in combined_meta (computed in processor2nd).
"""
# TODO: verify wavenumber column is populated for all runs, then replace
#       Hz x-axis with ka where appropriate in CH05 §1-3
_save_placeholder("ch05_damping_ka", "CH05 §4 — Damping vs ka (wave steepness axis)", chapter="05")

# %%
"""
── CH05 § 5 — Swell / band amplitude scatter (IN vs OUT) ────────────────────
Goal: show IN vs OUT amplitude for swell, wind-wave, and total bands.
Characterises what energy bands the panel attenuates and passes.
Data: combined_meta band amplitude columns (Probe {pos} Swell Amplitude (PSD) etc.)
"""

_pv_swell_scatter = {
    "filters": {
        "WaveAmplitudeInput [Volt]": [0.1, 0.2, 0.3],
        "WaveFrequencyInput [Hz]":   None,
        "WindCondition":             None,
        "PanelCondition":            None,
    },
    "plotting": {
        "show_plot":   True,
        "save_plot":   True,            # DRAFT — not yet polished
        "draft":       True,
        "figure_name": "ch05_swell_scatter",
        "force_stub":  True,
    },
}

plot_swell_scatter(combined_meta, _pv_swell_scatter, chapter="05")

# %%
"""
── CH05 § 6 — Reconstructed wave signal ─────────────────────────────────────
Goal: show the FFT-reconstructed paddle-frequency signal alongside the raw
time-series. Illustrates what A_FFT actually isolates from the full signal.
Data: combined_fft_dict, one representative run (1.3 Hz, 0.2 V, full panel).
"""

_pv_reconstructed = {
    "filters": {
        "WaveAmplitudeInput [Volt]": 0.2,
        "WaveFrequencyInput [Hz]":   1.3,
        "WindCondition":             None,
        "PanelCondition":            "full",
    },
    "plotting": {
        "show_plot":    True,
        "save_plot":    True,           # DRAFT — not yet polished
        "draft":        True,
        "figure_name":  "ch05_reconstructed",
        "force_stub":   True,
        "facet_by":     "probe",
        "probes":       ["9373/170", "12400/250"],
        "linewidth":    0.8,
        "grid":         True,
        "legend":       "inside",
        "xlim":         None,
        "max_points":   500,
    },
}

_recon_meta  = apply_experimental_filters(combined_meta, _pv_reconstructed)
_recon_paths = {p: combined_fft_dict[p]
                for p in _recon_meta["path"] if p in combined_fft_dict}
if _recon_paths:
    plot_reconstructed(_recon_paths, _recon_meta, _pv_reconstructed,
                       data_type="fft", chapter="05")
else:
    print("ch05_reconstructed: no matching runs found — check filters.")


# =============================================================================
# WAVE DETECTION (diagnostic, possibly CH04 § 6)
# =============================================================================

# %%
"""
── First wave arrival ────────────────────────────────────────────────────────
Detection of first wave energy arriving at each probe. Useful for validating
_SNARVEI_CALIB start-sample estimates and understanding wave group velocity.
"""
# Detection parameters
THRESHOLD_FACTOR = 2.0    # 2× stillwater noise floor
WINDOW_S         = 0.5    # rolling window length [s]
MIN_ARRIVAL_S    = 0.5    # arrivals below this are wind-wave artefacts

PROBE_POSITIONS = ANALYSIS_PROBES

_sw_mask = (
    combined_meta["WaveFrequencyInput [Hz]"].isna()
    & (combined_meta["WindCondition"] == "no")
)
_noise_floor = {
    col.replace("Probe ", "").replace(" Amplitude", ""):
        combined_meta.loc[_sw_mask, col].mean()
    for col in combined_meta.columns
    if col.startswith("Probe ") and col.endswith(" Amplitude")
    and "FFT" not in col and "PSD" not in col
    and combined_meta.loc[_sw_mask, col].notna().any()
}

# Arrival detection — needs processed_dfs
# processed_dfs = load_processed_dfs(*PROCESSED_DIRS)   # uncomment if needed
_arrival_rows = []
# for _, row in combined_meta[combined_meta["WaveFrequencyInput [Hz]"].notna()].iterrows():
#     df = processed_dfs.get(row["path"])
#     if df is None:
#         continue
#     for pos in PROBE_POSITIONS:
#         eta_col = f"eta_{pos}"
#         if eta_col not in df.columns:
#             continue
#         noise = _noise_floor.get(pos)
#         if not noise or noise <= 0:
#             continue
#         sig = df[eta_col].dropna().values
#         idx, t_s = find_first_arrival(sig, noise, fs=FS,
#                                       threshold_factor=THRESHOLD_FACTOR,
#                                       window_s=WINDOW_S)
#         _arrival_rows.append({
#             "run": Path(row["path"]).name,
#             "freq_hz":    row["WaveFrequencyInput [Hz]"],
#             "amp_volt":   row.get("WaveAmplitudeInput [Volt]"),
#             "wind":       row.get("WindCondition"),
#             "panel":      row.get("PanelCondition"),
#             "probe":      pos,
#             "dist_mm":    int(pos.split("/")[0]),
#             "arrival_idx": idx,
#             "arrival_s":  t_s,
#         })

arrival_df = pd.DataFrame(_arrival_rows)
if not arrival_df.empty:
    print(f"Arrival detections: {arrival_df['arrival_s'].notna().sum()} / {len(arrival_df)}")

# ── Plot A: faceted by frequency ──────────────────────────────────────────────
# if not arrival_df.empty:
#     _plot_facet = arrival_df.dropna(subset=["arrival_s"])
#     sns.set_style("ticks", {"axes.grid": True})
#     g = sns.relplot(
#         data=_plot_facet, x="dist_mm", y="arrival_s",
#         hue="wind", col="freq_hz",
#         kind="line", marker="o", dashes=False,
#         errorbar=None, markersize=10, linewidth=0.8,
#         height=3.5, aspect=1.3,
#         palette=WIND_COLOR_MAP,
#         facet_kws={"sharey": True},
#     )
#     for ax in g.axes.flat:
#         ax.set_xlabel("Probe distance from paddle [mm]")
#         ax.set_ylabel("First arrival [s]")
#     g.figure.suptitle(
#         f"First wave arrival  (threshold = {THRESHOLD_FACTOR}× noise floor,"
#         f"  window = {WINDOW_S} s)",
#         y=1.02, fontsize=9,
#     )
#     plt.tight_layout()
#     plt.show()

# ── Plot B: all frequencies, single axes ──────────────────────────────────────
# if not arrival_df.empty:
#     WIND_LS = {"no": "-", "lowest": "--", "full": ":"}
#     _plot_single = arrival_df[arrival_df["arrival_s"] > MIN_ARRIVAL_S].copy()
#     _freqs_sorted = sorted(_plot_single["freq_hz"].dropna().unique())
#     _freq_colors  = {f: c for f, c in zip(
#         _freqs_sorted, plt.cm.rainbow(np.linspace(0, 1, len(_freqs_sorted)))
#     )}
#     _agg = (
#         _plot_single
#         .groupby(["wind", "freq_hz", "dist_mm"])["arrival_s"]
#         .agg(mean="mean", err=lambda x: (x.max() - x.min()) / 2)
#         .reset_index()
#     )
#     apply_thesis_style()
#     fig, ax = plt.subplots(figsize=(9, 5))
#     for (wind, freq), grp in _agg.groupby(["wind", "freq_hz"]):
#         grp_s = grp.sort_values("dist_mm")
#         ax.errorbar(grp_s["dist_mm"], grp_s["mean"], yerr=grp_s["err"],
#                     marker="o", markersize=8, linewidth=1.2, capsize=4,
#                     color=_freq_colors[freq],
#                     linestyle=WIND_LS.get(wind, "-"),
#                     label=f"{freq} Hz / {wind}")
#     ax.set_xlabel("Probe distance from paddle [mm]")
#     ax.set_ylabel("First arrival [s]")
#     ax.set_title(
#         f"First wave arrival — all frequencies  "
#         f"(threshold {THRESHOLD_FACTOR}× noise,  arrivals > {MIN_ARRIVAL_S} s shown)"
#     )
#     handles, labels = ax.get_legend_handles_labels()
#     ax.legend(handles[::-1], labels[::-1], fontsize=8, title="freq / wind")
#     plt.tight_layout()
#     _meta = {
#         "chapter": "04", "panel": None, "wind": None,
#         "amplitude": None, "frequency": None,
#         "probes": PROBE_POSITIONS, "script": "main_save_figures.py::first_arrival",
#     }
#     save_and_stub(fig, _meta, "first_arrival")
#     plt.show()

print("main_save_figures.py — all figure sections complete.")

# TODO: check the phase on the sine vs signal comparison.
# BIG TODO: change all plots with freq on x-axis to kL.
