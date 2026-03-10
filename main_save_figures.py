#!/usr/bin/env python3
"""
Save publication-quality figures for the thesis.

Run from terminal when you want to regenerate all output figures:
    conda activate draumkvedet
    cd /Users/ole/Kodevik/wave_project
    python main_save_figures.py

Saves to figures/ directory (created if it doesn't exist).
Set save_plot=True in each section when the figure is ready to export.

Requires processed data in waveprocessed/. Run main.py first if missing or stale.
"""

import os
from pathlib import Path

from wavescripts.filters import (
    apply_experimental_filters,
    damping_all_amplitude_grouper,
    filter_for_frequencyspectrum,
)
from wavescripts.improved_data_loader import load_analysis_data
from wavescripts.plotter import (
    plot_all_probes,
    plot_damping_freq,
    plot_damping_scatter,
    plot_frequency_spectrum,
    plot_swell_scatter,
)

try:
    file_dir = Path(__file__).resolve().parent
except NameError:
    file_dir = Path.cwd()
os.chdir(file_dir)

figures_dir = file_dir / "figures"
figures_dir.mkdir(exist_ok=True)

# ── Dataset(s) ────────────────────────────────────────────────────────────────
PROCESSED_DIRS = [
    Path("waveprocessed/PROCESSED-20260307-ProbPos4_31_FPV_2-tett6roof"),
]

# ── Load from cache ───────────────────────────────────────────────────────────
print("Loading analysis data...")
combined_meta, processed_dfs, combined_fft_dict, combined_psd_dict = load_analysis_data(
    *PROCESSED_DIRS
)

# ── Figure: amplitude all probes ──────────────────────────────────────────────
# TODO: set save_plot=True and configure save path when ready
amplitudeplotvariables = {
    "overordnet": {"chooseAll": True, "chooseFirst": False, "chooseFirstUnique": False},
    "filters": {
        "WaveAmplitudeInput [Volt]": None,
        "WaveFrequencyInput [Hz]":   None,
        "WindCondition":             ["no", "lowest", "full"],
        "TunnelCondition":           None,
        "Mooring":                   "low",
        "PanelCondition":            None,
    },
    "plotting": {
        "figsize":    [7, 4],
        "separate":   True,
        "overlay":    False,
        "annotate":   True,
        "save_plot":  False,  # ← set True when ready
    },
}
# m_filtrert = apply_experimental_filters(combined_meta, amplitudeplotvariables)
# plot_all_probes(m_filtrert, amplitudeplotvariables)

# ── Figure: damping vs frequency ──────────────────────────────────────────────
# TODO
# damping_groupedallruns_df = damping_all_amplitude_grouper(combined_meta)
# plot_damping_freq(damping_groupedallruns_df, ...)
# plot_damping_scatter(damping_groupedallruns_df, ...)

# ── Figure: FFT spectrum ──────────────────────────────────────────────────────
# TODO
# filtrert = filter_for_frequencyspectrum(combined_meta, freqplotvariables)
# plot_frequency_spectrum(combined_fft_dict, filtrert, freqplotvariables, data_type="fft")

# ── Figure: swell scatter ─────────────────────────────────────────────────────
# TODO
# plot_swell_scatter(combined_meta, swellplotvariables)

print("main_save_figures: no figures saved yet (all sections are stubs).")
print(f"Output directory: {figures_dir}")
