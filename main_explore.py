#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 10:21:49 2025

@author: ole
"""

import copy
import os
import sys

# %%
# %% load results from disk — no reprocessing needed
from pathlib import Path

import pandas as pd
from PyQt5.QtWidgets import QApplication

from wavescripts.constants import (
    MEASUREMENT,
    RAMP,
    SIGNAL,
    WIND_COLOR_MAP,
    get_smoothing_window,
)
from wavescripts.constants import CalculationResultColumns as RC
from wavescripts.constants import ColumnGroups as CG
from wavescripts.constants import GlobalColumns as GC
from wavescripts.constants import PlottPent as PP
from wavescripts.constants import ProbeColumns as PC

# ── Filters ───────────────────────────────────────────────────────────────────
from wavescripts.filters import (
    apply_experimental_filters,
    damping_all_amplitude_grouper,
    damping_grouper,
    filter_chosen_files,
    filter_dataframe,
    filter_for_amplitude_plot,
    filter_for_damping,
    filter_for_frequencyspectrum,
)
from wavescripts.improved_data_loader import (
    load_or_update,
    load_processed_dfs,
    save_processed_dfs,
)

# ── Quicklook / exploration ───────────────────────────────────────────────────
from wavescripts.plot_quicklook import (
    RampDetectionBrowser,
    SignalBrowserFiltered,
    explore_damping_vs_amp,
    explore_damping_vs_freq,
    save_interactive_plot,
)

# ── Thesis plots ──────────────────────────────────────────────────────────────
from wavescripts.plotter import (
    gather_ramp_data,
    plot_all_probes,
    plot_damping_freq,
    plot_damping_scatter,
    plot_frequency_spectrum,
    plot_ramp_detection,
    plot_reconstructed,
    plot_reconstructed_rms,
    plot_swell_scatter,
)
from wavescripts.processor import process_selected_data
from wavescripts.processor2nd import process_processed_data

file_dir = Path("/Users/ole/Kodevik/wave_project")
processed_dirs = [
    file_dir / "waveprocessed" / f"PROCESSED-{p.name}" for p in dataset_paths
]
combined_processed_dfs = load_processed_dfs(*processed_dirs)
# ... now explore freely


print("hello")

# %%

print("world")
