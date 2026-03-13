#!/usr/bin/env python3
"""
Wave data processing pipeline.

Run this when you have new data or need to reprocess:
    conda activate draumkvedet && python main.py

Outputs per dataset in waveprocessed/PROCESSED-<folder>/:
    meta.json             — metadata for all runs (stillwater, FFT amplitudes, etc.)
    processed_dfs.parquet — zeroed + smoothed time series
    dfs.parquet           — raw time series cache

After processing, use:
    main_explore_browser.py  — Qt interactive signal/ramp browsers (terminal)
    main_explore_inline.py   — cell-by-cell inline plots (Zed REPL)
"""

# %%
import gc
import os
import sys
from pathlib import Path

import pandas as pd
from numpy._core.numeric import True_

from wavescripts.filters import filter_chosen_files
from wavescripts.improved_data_loader import (
    load_or_update,
    save_processed_dfs,
    save_spectra_dicts,
)
from wavescripts.processor import process_selected_data
from wavescripts.processor2nd import process_processed_data

# ── Working directory ─────────────────────────────────────────────────────────
try:
    file_dir = Path(__file__).resolve().parent
except NameError:
    file_dir = Path.cwd()
os.chdir(file_dir)

# ── Dataset paths ─────────────────────────────────────────────────────────────
# Add / uncomment folders here as new data arrives.
dataset_paths = [
    Path("/Users/ole/Kodevik/wave_project/wavedata/20251005-sixttry6roof-highMooring"), #denne har probe 1 på 18000
    Path("/Users/ole/Kodevik/wave_project/wavedata/20251110-tett6roof-lowM-ekte580"),
    Path("/Users/ole/Kodevik/wave_project/wavedata/20251110-tett6roof-lowMooring"),
    Path("/Users/ole/Kodevik/wave_project/wavedata/20251110-tett6roof-lowMooring-2"),
    Path("/Users/ole/Kodevik/wave_project/wavedata/20251112-tett6roof"),
    Path("/Users/ole/Kodevik/wave_project/wavedata/20251113-tett6roof"),
    Path("/Users/ole/Kodevik/wave_project/wavedata/20251113-tett6roof-loosepaneltaped"),
    Path("/Users/ole/Kodevik/wave_project/wavedata/20251113-tett6roof-probeadjusted"),
    Path("/Users/ole/Kodevik/wave_project/wavedata/20260305-newProbePos-tett6roof"),
    Path("/Users/ole/Kodevik/wave_project/wavedata/20260306-newProbePos-tett6roof"),
    Path("/Users/ole/Kodevik/wave_project/wavedata/20260307-ProbPos4_31_FPV_2-tett6roof"),
    Path("/Users/ole/Kodevik/wave_project/wavedata/20260312-ProbPos4_31_FPV_2-tett6roof"),
]

# ── Processing options ────────────────────────────────────────────────────────
processvariables = {
    "overordnet": {
        "chooseAll": True,
        "chooseFirst": False,
    },
    "filters": {
        "WaveAmplitudeInput [Volt]": None,
        "WaveFrequencyInput [Hz]": None,
        "WavePeriodInput": None,
        "WindCondition": None,
        "TunnelCondition": None,
        "Mooring": "low",
        "PanelCondition": None,
    },
    "prosessering": {
        "total_reset": True,  # True → wipe CSV cache and reload everything
        "force_recompute": True,  # True → ignore cached meta.json, recompute all
        "debug": True,
        "smoothing_window": 10,
        "find_range": True,
        "range_plot": False,
    },
}

# ── Processing loop ───────────────────────────────────────────────────────────
prosessering = processvariables.get("prosessering", {})
total_reset = prosessering.get("total_reset", False)

if total_reset:
    input(
        "TOTAL RESET — all CSV caches will be wiped. Press Enter to continue, Ctrl-C to abort."
    )

processed_dirs = []
all_meta_sel = []

for i, data_path in enumerate(dataset_paths):
    print(f"\n{'=' * 50}")
    print(f"Processing dataset {i + 1}/{len(dataset_paths)}: {data_path.name}")
    print(f"{'=' * 50}")
    try:
        dfs, meta = load_or_update(
            data_path,
            force_recompute=prosessering.get("force_recompute", False),
            total_reset=total_reset,
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
        save_spectra_dicts(fft_dictionary, psd_dictionary, _cache_dir)
        del meta, psd_dictionary, fft_dictionary

        all_meta_sel.append(meta_sel)
        print(f"Done: {len(meta_sel)} runs processed from {data_path.name}")

    except Exception as e:
        import traceback

        print(f"Error processing {data_path.name}: {e}")
        traceback.print_exc()
        continue

print(f"\n{'=' * 50}")
print(f"PROCESSING COMPLETE — {len(all_meta_sel)} dataset(s)")
print(f"Total runs: {sum(len(s) for s in all_meta_sel)}")
print(f"Cache dirs:")
for d in processed_dirs:
    print(f"  {d}")
print(f"{'=' * 50}")

gc.collect()

print("Main is complete - you can close this")

"""
One note on mainTester:** it contained exploration of time-lag
corrections between probes (measuring wave propagation speed between
probe pairs). Worth knowing that's in the archive if you need it later
for celerity validation.
"""
