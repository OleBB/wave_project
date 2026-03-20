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
"""

# %%
import os
from pathlib import Path

import time

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
from wavescripts.improved_data_loader import load_analysis_data, load_processed_dfs, ANALYSIS_PROBES
from wavescripts.plot_utils import WIND_COLOR_MAP, apply_thesis_style, save_and_stub
from wavescripts.plotter import (
    plot_all_probes,
    plot_damping_freq,
    plot_damping_scatter,
    plot_frequency_spectrum,
    plot_parallel_ratio,
    plot_probe_noise_floor,
    plot_swell_scatter,
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
PROCESSED_DIRS = [
    # Path("waveprocessed/PROCESSED-20251005-sixttry6roof-highMooring"), #denne har probe 1 på 18000. Men husk at taket ikke var tetta helt.
    # Path("waveprocessed/PROCESSED-20251110-tett6roof-lowM-ekte580"), # mange kjøringer med -per15
    # Path("waveprocessed/PROCESSED-20251110-tett6roof-lowMooring"), # noen kjøringer med  -per30-
    # Path("waveprocessed/PROCESSED-20251110-tett6roof-lowMooring-2"), #et par kjøringer med -per15-
    # Path("waveprocessed/PROCESSED-20251112-tett6roof"),
    # Path("waveprocessed/PROCESSED-20251113-tett6roof"),
    # Path("waveprocessed/PROCESSED-20251113-tett6roof-loosepaneltaped"),
    # Path("waveprocessed/PROCESSED-20251113-tett6roof-probeadjusted"),
    # Path("waveprocessed/PROCESSED-20260305-newProbePos-tett6roof"),
    # Path("waveprocessed/PROCESSED-20260306-newProbePos-tett6roof"),
    Path("waveprocessed/PROCESSED-20260307-ProbPos4_31_FPV_2-tett6roof"),
    Path("waveprocessed/PROCESSED-20260312-ProbPos4_31_FPV_2-tett6roof"),
    Path("waveprocessed/PROCESSED-20260313-ProbePos4_31_FPV_2-tett6roof"),
    Path("waveprocessed/PROCESSED-20260314-ProbePos4_31_FPV_2-tett6roof"),
    Path("waveprocessed/PROCESSED-20260316-ProbePos4_31_FPV_2-tett6roof"),
    Path("waveprocessed/PROCESSED-20260316-ProbePos4_31_FPV_2-tett6roof-under9Mooring"),
]

# ── Load from cache ───────────────────────────────────────────────────────────
print("Loading analysis data...")
combined_meta, _, combined_fft_dict, combined_psd_dict = load_analysis_data(
    *PROCESSED_DIRS
)

# %%
# processed_dfs is heavy (~75 MB). Loaded when needed by sections below.
processed_dfs = load_processed_dfs(*PROCESSED_DIRS)


# =============================================================================
# CHAPTER 04 — METHODOLOGY
# =============================================================================
# %%

# %%
"""
── CH04 § 1 — Probe uncertainty / noise floor ───────────────────────────────
Goal: show that each probe has a measurable noise floor and that it is stable
across stillwater runs. Defines the detection threshold ((2?)× noise floor).

Data: stillwater runs (WindCondition == "no", WaveFrequencyInput NaN or regex:nowwave).
Already in combined_meta as "Probe {pos} Amplitude" for those rows.

Figures/tables:
  - Table: noise floor per probe (mean ± std across stillwater runs) [mm]
  - Plot:  stillwater amplitude bar chart or box plot, one bar per probe

Mine notater:
"""


"""
PRINTOUT:
    [plot_probe_noise_floor] caption slots: {'window_ms': 200.0, 'n_runs': 14, 'n_flagged': 1, 'amp_cap_mm': 0.5}
    [plot_probe_noise_floor] formatted caption:
      "Probe noise floor estimated as the minimum windowed (P$_{97.5}$--P$_{2.5}$)/2 amplitude over 200\,ms sliding windows of 14 st
    illwater recordings (1 run(s) excluded — name keyword or windowed minimum above 0.5\,mm). Short windows suppress slow tank slosh
    ing so only electronic jitter and capillary ripples remain. Error bars: standard deviation across runs. White dots: individual r
    un values."
    === Probe noise floor summary [mm] ===
                 mean     std  min     max
    probe
    9373/170   0.0285  0.0159  0.0  0.0400
    12400/250  0.0314  0.0174  0.0  0.0489
    9373/340   0.0251  0.0195  0.0  0.0400
    8804/250   0.0258  0.0174  0.0  0.0450

"""

# import importlib
# import wavescripts.plotter as _plotter_mod
# importlib.reload(_plotter_mod)
# from wavescripts.plotter import plot_probe_noise_floor

_pv_noise_floor = {
    "filters": {},
    "plotting": {
        "show_plot": False,
        "save_plot": False,           # set True when figure is ready for thesis
        "figure_name": "ch04_probe_noise_floor",
        "force_stub": True,
    },
    "caption": {
        "PROBE noise floor estimated as the minimum windowed (P$_{97.5}$--P$_{2.5}$)/2 amplitude over 200\,ms sliding windows of 14 stillwater recordings (1 run(s) excluded — name keyword or windowed minimum above 0.5\,mm). Short windows suppress slow tank sloshing so only electronic jitter and capillary ripples remain. Error bars: standard deviation across runs. White dots: individual run values."
    }
}

start = time.perf_counter()
_fig_nf, _noise_summary = plot_probe_noise_floor(
    combined_meta,
    processed_dfs,
    probe_positions=ANALYSIS_PROBES,
    plotvariables=_pv_noise_floor,
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
# TODO

# %%
import time
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
        "save_plot": True,           # set True when figure is ready for thesis
        "figure_name": "ch04_parallel_ratio",
        "force_stub": True,
    },
    "caption": {
        "Ratio of wall-side to far-side probe amplitude at the same longitudinal distance, for 154 wave runs across 1 panel configurations. A ratio of 1 indicates lateral symmetry. Deviations indicate wall reflections or wind-driven lateral asymmetry. Error bars: standard deviation across runs at the same frequency. Dashed line: ratio = 1."
    }
}
start = time.perf_counter()
_fig_pr = plot_parallel_ratio(combined_meta, _pv_parallel_ratio)
end = time.perf_counter()
print(f"Lateral symmetry of plot_parallel_ratio {end-start:.4f} seconds")

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
        "force_stub":    False,
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
            "Power spectral density of the free surface at each wave gauge "
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

# %%
"""
── CH04 § 5 — What does a full signal look like? ────────────────────────────
Goal: annotated time-domain plot of one complete run showing:
  - stillwater baseline → wavemaker ramp → stable wavetrain → decay
  - wind-wave riding on top of the paddle wave (IN probe vs OUT probe)

Data: processed_dfs, one representative wave+fullwind run.

Figures:
  - Plot:  eta_* vs time for IN and OUT probe, one run, annotated regions
"""
# TODO — needs processed_dfs

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
# TODO — needs processed_dfs

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
# TODO

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
# TODO


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

Data: combined_meta → damping_grouper → plot_damping_freq.

Figures:
  - plot_damping_freq: OUT/IN vs freq, errorbars (std or ±10% for n=1)
  - Same plot with ka on x-axis (requires wavenumber column)
"""
# TODO: uncomment and configure when ready
# from wavescripts.filters import damping_grouper
# grouped, wide = damping_grouper(combined_meta)
# plot_damping_freq(grouped, ...)

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
# TODO

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
# TODO

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

print("main_save_figures.py loaded — all figure sections are stubs. Uncomment to run.")
