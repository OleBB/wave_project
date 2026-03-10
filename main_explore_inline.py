#!/usr/bin/env python3
"""
Inline cell-by-cell wave data analysis.

Open in Zed and run cells individually (Shift+Enter or your keybinding).
Plots appear inline in the Zed output panel.

Requires processed data in waveprocessed/. Run main.py first if missing or stale.

For interactive Qt browsers → use main_explore_browser.py (run from terminal).
For saving publication figures → use main_save_figures.py.
"""

# %% ── imports ────────────────────────────────────────────────────────────────
%matplotlib inline
import os
from pathlib import Path

import numpy as np
import pandas as pd

from wavescripts.constants import ColumnGroups as CG
from wavescripts.constants import GlobalColumns as GC
from wavescripts.constants import ProbeColumns as PC
from wavescripts.filters import (
    apply_experimental_filters,
    damping_all_amplitude_grouper,
    damping_grouper,
    filter_for_amplitude_plot,
    filter_for_damping,
    filter_for_frequencyspectrum,
)
from wavescripts.improved_data_loader import load_analysis_data
from wavescripts.plot_quicklook import (
    explore_damping_vs_amp,
    explore_damping_vs_freq,
    save_interactive_plot,
)
from wavescripts.plotter import (
    plot_all_probes,
    plot_damping_freq,
    plot_damping_scatter,
    plot_frequency_spectrum,
    plot_reconstructed,
    plot_swell_scatter,
)

try:
    file_dir = Path(__file__).resolve().parent
except NameError:
    file_dir = Path.cwd()
os.chdir(file_dir)

# %% ── load from cache (fast — no reprocessing) ───────────────────────────────
PROCESSED_DIRS = [
    Path("waveprocessed/PROCESSED-20260307-ProbPos4_31_FPV_2-tett6roof"),
    # Path("waveprocessed/PROCESSED-20251112-tett6roof"),
]

combined_meta, processed_dfs, combined_fft_dict, combined_psd_dict = load_analysis_data(
    *PROCESSED_DIRS
)

# Paths present in both metadata and FFT dict
matching_paths = set(combined_fft_dict.keys()) & set(combined_meta["path"].unique())
filtered_fft_dict = {p: combined_fft_dict[p] for p in matching_paths}
filtered_meta = combined_meta[combined_meta["path"].isin(matching_paths)].copy()

print(f"Loaded: {len(combined_meta)} total rows, {len(filtered_fft_dict)} wave experiments")
print(f"PanelCondition values: {sorted(combined_meta['PanelCondition'].dropna().unique())}")
print(f"WindCondition values:  {sorted(combined_meta['WindCondition'].dropna().unique())}")
print(f"Frequencies [Hz]:      {sorted(combined_meta['WaveFrequencyInput [Hz]'].dropna().unique())}")

# %% ── amplitude — all probes physical layout ─────────────────────────────────
amplitudeplotvariables = {
    "overordnet": {
        "chooseAll": True,
        "chooseFirst": False,
        "chooseFirstUnique": False,
    },
    "filters": {
        "WaveAmplitudeInput [Volt]": None,
        "WaveFrequencyInput [Hz]":   None,
        "WavePeriodInput":           None,
        "WindCondition":             ["no", "lowest", "full"],
        "TunnelCondition":           None,
        "Mooring":                   "low",
        "PanelCondition":            None,
    },
    "plotting": {
        "figsize":   [7, 4],
        "separate":  True,
        "overlay":   False,
        "annotate":  True,
    },
}

m_filtrert = apply_experimental_filters(combined_meta, amplitudeplotvariables)
plot_all_probes(m_filtrert, amplitudeplotvariables)

# %% ── damping grouper + interactive HTML ─────────────────────────────────────
damping_groupedruns_df, damping_pivot_wide = damping_grouper(combined_meta)
save_interactive_plot(damping_groupedruns_df)

# %% ── damping vs frequency ───────────────────────────────────────────────────
dampingplotvariables = {
    "overordnet": {"chooseAll": True, "chooseFirst": False, "chooseFirstUnique": False},
    "filters": {
        "WaveAmplitudeInput [Volt]": None,
        "WaveFrequencyInput [Hz]":   None,
        "WavePeriodInput":           None,
        "WindCondition":             None,
        "TunnelCondition":           None,
        "PanelCondition":            None,
    },
    "plotting": {"figsize": None, "separate": True, "overlay": False, "annotate": True},
}

damping_filtrert = filter_for_damping(damping_groupedruns_df, dampingplotvariables["filters"])
explore_damping_vs_freq(damping_filtrert, dampingplotvariables)

# %% ── damping vs amplitude ───────────────────────────────────────────────────
explore_damping_vs_amp(damping_filtrert, dampingplotvariables)

# %% ── damping all amplitudes grouped ────────────────────────────────────────
dampingplotvariables_all = {
    "overordnet": {"chooseAll": True, "chooseFirst": False},
    "filters": {
        "WaveAmplitudeInput [Volt]": None,
        "WaveFrequencyInput [Hz]":   None,
        "WavePeriodInput":           None,
        "WindCondition":             ["no", "lowest", "full"],
        "TunnelCondition":           None,
        "PanelCondition":            None,
    },
    "plotting": {
        "show_plot":  True,
        "save_plot":  False,
        "figsize":    None,
        "separate":   False,
        "facet_by":   None,
        "overlay":    False,
        "annotate":   True,
        "legend":     "outside_right",
        "logaritmic": False,
        "peaks":      7,
        "probes":     ["9373/170", "12545", "9373/340", "8804"],
    },
}

damping_groupedallruns_df = damping_all_amplitude_grouper(combined_meta)
plot_damping_freq(damping_groupedallruns_df, dampingplotvariables_all)

# %% ── damping scatter ────────────────────────────────────────────────────────
plot_damping_scatter(damping_groupedallruns_df, dampingplotvariables_all)

# %% ── FFT spectrum — filter config ──────────────────────────────────────────
# Adjust filters to match the PanelCondition values in your dataset.
# Run the load cell first and check the printed PanelCondition values.
freqplotvariables = {
    "overordnet": {
        "chooseAll": False,
        "chooseFirst": False,
        "chooseFirstUnique": True,
    },
    "filters": {
        "WaveAmplitudeInput [Volt]": [0.1],
        "WaveFrequencyInput [Hz]":   [1.3],
        "WavePeriodInput":           None,
        "WindCondition":             ["no", "lowest", "full"],
        "TunnelCondition":           None,
        "Mooring":                   None,
        "PanelCondition":            None,  # set to match your data, e.g. "full"
    },
    "plotting": {
        "show_plot":   True,
        "save_plot":   False,
        "figsize":     (5, 5),
        "linewidth":   0.7,
        "facet_by":    "probe",
        "max_points":  120,
        "xlim":        (0, 5.2),
        "legend":      "inside",
        "logaritmic":  False,
        "peaks":       3,
        "probes":      ["12545", "9373/340"],
    },
}

filtrert_frequencies = filter_for_frequencyspectrum(combined_meta, freqplotvariables)
print(f"Frequency filter: {len(filtrert_frequencies)} runs matched")

# %% ── FFT spectrum plot ───────────────────────────────────────────────────────
fig, axes = plot_frequency_spectrum(
    combined_fft_dict, filtrert_frequencies, freqplotvariables, data_type="fft"
)

# %% ── PSD spectrum plot ──────────────────────────────────────────────────────
fig, axes = plot_frequency_spectrum(
    combined_psd_dict, filtrert_frequencies, freqplotvariables, data_type="psd"
)

# %% ── swell scatter ──────────────────────────────────────────────────────────
swellplotvariables = {
    "overordnet": {
        "chooseAll": True,
        "chooseFirst": False,
        "chooseFirstUnique": True,
    },
    "filters": {
        "WaveAmplitudeInput [Volt]": [0.1, 0.2, 0.3],
        "WaveFrequencyInput [Hz]":   [1.3],
        "WavePeriodInput":           None,
        "WindCondition":             ["no", "lowest", "full"],
        "TunnelCondition":           None,
        "Mooring":                   None,
        "PanelCondition":            None,  # set to match your data
    },
    "plotting": {
        "show_plot":  True,
        "save_plot":  False,
        "figsize":    (5, 5),
        "linewidth":  0.7,
        "facet_by":   "probe",
        "max_points": 120,
        "xlim":       (0, 5.2),
        "legend":     "inside",
        "logaritmic": False,
        "peaks":      3,
        "probes":     ["12545", "9373/340"],
    },
}

plot_swell_scatter(combined_meta, swellplotvariables)

# %% ── wavenumber study ───────────────────────────────────────────────────────
_probe_positions = ["9373/170", "12545", "9373/340", "8804"]
wavenumber_cols = [f"Probe {pos} Wavenumber (FFT)" for pos in _probe_positions]
fft_dimension_cols = [CG.fft_wave_dimension_cols(pos) for pos in _probe_positions]
meta_wavenumber = combined_meta[["path"] + [c for c in wavenumber_cols if c in combined_meta.columns]].copy()
print(meta_wavenumber.describe())

# %% ── reconstructed signal — single experiment ───────────────────────────────
# Pick one experiment to inspect its reconstructed time-domain signal.
single_path = list(filtered_fft_dict.keys())[0]
single_meta = filtered_meta[filtered_meta["path"] == single_path]

fig, ax = plot_reconstructed(
    {single_path: filtered_fft_dict[single_path]}, single_meta, freqplotvariables
)

# %% ── reconstructed signal — all filtered experiments ───────────────────────
fig, axes = plot_reconstructed(
    filtered_fft_dict, filtered_meta, freqplotvariables, data_type="fft"
)
