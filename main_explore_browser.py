#!/usr/bin/env python3
# %%
"""
Interactive Qt browsers for wave data exploration.

Run from terminal — NOT from Zed REPL (Qt event loop conflicts with Jupyter kernel):
    conda activate draumkvedet
    cd /Users/ole/Kodevik/wave_project
    python main_explore_browser.py

Requires processed data in waveprocessed/. Run main.py first if missing or stale.

Browsers launched:
    SignalBrowserFiltered  — step through FFT signal reconstruction run by run,
                             filter by wind / panel / freq / amplitude, select probes
    RampDetectionBrowser   — inspect ramp detection results for all runs
"""

# %%
# Must be set before any matplotlib / wavescripts imports
import matplotlib
matplotlib.use("Qt5Agg")

import sys
from pathlib import Path
import dtale

from PyQt5.QtWidgets import QApplication

from wavescripts.improved_data_loader import load_analysis_data, load_processed_dfs
from wavescripts.plot_browsers import RampDetectionBrowser, SignalBrowserFiltered
from wavescripts.plotter import gather_ramp_data
# %%
# ── Dataset(s) ────────────────────────────────────────────────────────────────
# List all PROCESSED-* folders you want to load. Add more for multi-dataset sessions.
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
]

# ── Load from cache (fast — no reprocessing) ──────────────────────────────────
print("Loading analysis data from cache...")
combined_meta, processed_dfs, fft_dict, psd_dict = load_analysis_data(*PROCESSED_DIRS, load_processed=True)

# Filter to paths that have both FFT data and metadata
matching_paths = set(fft_dict.keys()) & set(combined_meta["path"].unique())
filtered_fft_dict = {p: fft_dict[p] for p in matching_paths}
filtered_meta = combined_meta[combined_meta["path"].isin(matching_paths)].copy()
print(f"  {len(filtered_fft_dict)} experiments ready for browsing")

# ── Plot variables — initial state of browser controls ───────────────────────
freqplotvariables = {
    "filters": {
        "WaveFrequencyInput [Hz]": None,
        "WaveAmplitudeInput [Volt]": None,
        "WindCondition": None,
        "PanelCondition": None,
    },
    "plotting": {
        "probes": ["12400/250", "9373/340"],
        "facet_by": "probe",
        "dual_yaxis": False,
        "show_full_signal": False,
        "linewidth": 1.0,
        "show_amplitude_stats": True,
        "grid": True,
    },
}

# ── Launch ────────────────────────────────────────────────────────────────────
app = QApplication.instance() or QApplication(sys.argv)



# %% ------------RAMP---------------------
ramp_df = gather_ramp_data(processed_dfs, combined_meta)
browser_ramp = RampDetectionBrowser(ramp_df)
browser_ramp.setWindowTitle("Ramp Detection Browser")
browser_ramp.show()
app.exec_()

# %% ----- Signal Browser
browser_signal = SignalBrowserFiltered(
    filtered_fft_dict, filtered_meta, freqplotvariables
)
browser_signal.setWindowTitle("Signal Browser — FFT reconstruction")
browser_signal.show()
print("Browsers open. Close all windows to exit.")
app.exec_()

# %%
# %% ----------dtale----------- explore combined_meta

dtale.show(combined_meta, host="localhost").open_browser()
