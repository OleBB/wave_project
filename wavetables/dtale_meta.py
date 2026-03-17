#!/usr/bin/env python3
"""
dtale_meta.py — open combined_meta in dtale, nothing else.

    conda activate draumkvedet && python dtale_meta.py

Loads only combined_meta (~2 s). No FFT, no PSD, no processed_dfs.
Keeps the dtale server alive until you press Enter or Ctrl-C.
"""

import os
import sys
import time
from pathlib import Path

# ── project root (one level up from wavetables/) ──────────────────────────────
project_root = Path(__file__).resolve().parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

import dtale
from wavescripts.improved_data_loader import load_analysis_data

# ── auto-discover all processed datasets ──────────────────────────────────────
PROCESSED_DIRS = sorted(Path("waveprocessed").glob("PROCESSED-*"))

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