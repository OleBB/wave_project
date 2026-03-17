#!/usr/bin/env python3
"""
dtale_clean.py — combined_meta without per-probe columns, for quick overview.

    conda activate draumkvedet && python wavetables/dtale_clean.py

Drops all columns starting with "Probe " — keeps experiment metadata,
IN/OUT generic columns, wave parameters, and quality metrics.
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

combined_meta, _, _, _ = load_analysis_data(*PROCESSED_DIRS, load_processed=False, load_spectra=False)

dt = time.perf_counter() - t0
print(f"  {len(combined_meta)} rows · {len(combined_meta.columns)} columns · {dt:.1f} s")

# ── drop unwanted columns ─────────────────────────────────────────────────────
import re
_rules = [
    (re.compile(r"\d{4,5}/\d{3}"),        "probe-position (9373/170 etc.)"),
    (re.compile(r"^Probe \d+ "),           "old probe-number (Probe 1/2/3/4)"),
    (re.compile(r"mm from (paddle|wall)"), "probe distances"),
    (re.compile(r"^Extra seconds$"),       "extra seconds"),
]
drop_cols = [
    c for c in combined_meta.columns
    if any(pat.search(c) for pat, _ in _rules)
]
clean = combined_meta.drop(columns=drop_cols)
print(f"  dropped {len(drop_cols)} columns → {len(clean.columns)} remaining")

# ── open dtale ────────────────────────────────────────────────────────────────
d = dtale.show(clean, host="localhost")
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
