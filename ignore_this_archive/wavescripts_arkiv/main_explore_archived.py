#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 10:21:49 2025

@author: ole
"""

# %%
import copy
import os
import sys

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

file_dir = Path(__file__).resolve().parent

# dataset_paths = [
#     Path("/Users/ole/Kodevik/wave_project/wavedata/20251112-tett6roof"),
#     Path("/Users/ole/Kodevik/wave_project/wavedata/20251112-tett6roof-lowM-579komma8"),
#     Path("/Users/ole/Kodevik/wave_project/wavedata/20251113-tett6roof"),
#     Path("/Users/ole/Kodevik/wave_project/wavedata/20251113-tett6roof-loosepaneltaped"),
#     Path("/Users/ole/Kodevik/wave_project/wavedata/20251113-tett6roof-probeadjusted"),
# ]
dataset_paths = [
    "/Users/ole/Kodevik/wave_project/wavedata/20260307-ProbPos4_31_FPV_2-tett6roof"
]

processvariables = {
    "overordnet": {"chooseAll": True, "chooseFirst": False},
    "filters": {
        "WaveAmplitudeInput [Volt]": [0.1],
        "WaveFrequencyInput [Hz]": 1.3,
        "WavePeriodInput": None,
        "WindCondition": None,
        "TunnelCondition": None,
        "Mooring": "low",
        "PanelCondition": None,
    },
    "prosessering": {
        "total_reset": False,
        "force_recompute": False,
        "debug": False,
        "smoothing_window": 10,
        "find_range": False,
        "range_plot": False,
    },
}

# %% load / process
all_meta_sel = []
processed_dirs = []
all_fft_dicts = []
all_psd_dicts = []

prosessering = processvariables.get("prosessering", {})

for i, data_path in enumerate(dataset_paths):
    print(f"Processing {i + 1}/{len(dataset_paths)}: {data_path.name}")
    try:
        dfs, meta = load_or_update(
            data_path,
            force_recompute=prosessering.get("force_recompute", False),
            total_reset=False,
        )
        meta_sel = filter_chosen_files(meta, processvariables)
        processed_dfs, meta_sel, psd_dictionary, fft_dictionary = process_selected_data(
            dfs, meta_sel, meta, processvariables
        )
        del dfs

        _cache_dir = file_dir / "waveprocessed" / f"PROCESSED-{data_path.name}"
        save_processed_dfs(processed_dfs, _cache_dir)
        processed_dirs.append(_cache_dir)
        del processed_dfs

        meta_sel = process_processed_data(
            psd_dictionary, fft_dictionary, meta_sel, meta, processvariables
        )
        del meta

        all_meta_sel.append(meta_sel)
        all_fft_dicts.append(fft_dictionary)
        all_psd_dicts.append(psd_dictionary)
    except Exception as e:
        print(f"Error: {e}")
        continue

combined_meta_sel = pd.concat(all_meta_sel, ignore_index=True)
combined_fft_dict = {k: v for d in all_fft_dicts for k, v in d.items()}
combined_psd_dict = {k: v for d in all_psd_dicts for k, v in d.items()}
combined_processed_dfs = load_processed_dfs(*processed_dirs)
print(f"Loaded {len(combined_meta_sel)} rows")

# %% dtale — explore combined_meta_sel
import dtale

d = dtale.show(combined_meta_sel, host="localhost")
d.open_browser()
input("dtale running — press Enter to stop")
