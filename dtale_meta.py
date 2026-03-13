#!/usr/bin/env python3
"""
dtale_meta.py — open combined_meta in dtale, nothing else.

    conda activate draumkvedet && python dtale_meta.py

Loads only combined_meta (~2 s). No FFT, no PSD, no processed_dfs.
Keeps the dtale server alive until you press Enter or Ctrl-C.
"""

import os
import time
from pathlib import Path

import dtale

from wavescripts.improved_data_loader import load_analysis_data

# ── working dir ───────────────────────────────────────────────────────────────
file_dir = Path(__file__).resolve().parent
os.chdir(file_dir)

# ── all processed datasets ────────────────────────────────────────────────────
PROCESSED_DIRS = [
    Path("waveprocessed/PROCESSED-20251005-sixttry6roof-highMooring"),
    Path("waveprocessed/PROCESSED-20251110-tett6roof-lowM-ekte580"),
    Path("waveprocessed/PROCESSED-20251110-tett6roof-lowMooring"),
    Path("waveprocessed/PROCESSED-20251110-tett6roof-lowMooring-2"),
    Path("waveprocessed/PROCESSED-20251112-tett6roof"),
    Path("waveprocessed/PROCESSED-20251113-tett6roof"),
    Path("waveprocessed/PROCESSED-20251113-tett6roof-loosepaneltaped"),
    Path("waveprocessed/PROCESSED-20251113-tett6roof-probeadjusted"),
    Path("waveprocessed/PROCESSED-20260305-newProbePos-tett6roof"),
    Path("waveprocessed/PROCESSED-20260306-newProbePos-tett6roof"),
    Path("waveprocessed/PROCESSED-20260307-ProbPos4_31_FPV_2-tett6roof"),
    Path("waveprocessed/PROCESSED-20260312-ProbPos4_31_FPV_2-tett6roof"),
    Path("waveprocessed/PROCESSED-20260313-ProbePos4_31_FPV_2-tett6roof"),
]

# ── load ──────────────────────────────────────────────────────────────────────
print("Loading combined_meta …")
t0 = time.perf_counter()

combined_meta, _, _, _ = load_analysis_data(*PROCESSED_DIRS, load_processed=False)

dt = time.perf_counter() - t0
print(f"  {len(combined_meta)} rows · {len(combined_meta.columns)} columns · {dt:.1f} s")

# ── open dtale ────────────────────────────────────────────────────────────────
d = dtale.show(combined_meta, host="localhost")
d.open_browser()
print(f"\ndtale running → {d._url}")
print("Press Enter (or Ctrl-C) to quit.\n")

try:
    input()
except KeyboardInterrupt:
    pass
finally:
    d.kill()
    print("dtale closed.")