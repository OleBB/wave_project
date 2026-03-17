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

combined_meta, _, _, _ = load_analysis_data(*PROCESSED_DIRS, load_processed=False)

dt = time.perf_counter() - t0
print(f"  {len(combined_meta)} rows · {len(combined_meta.columns)} columns · {dt:.1f} s")

# ── drop all probe-position-specific columns ──────────────────────────────────
# Matches any column containing a position string like "9373/170" or "12400/250"
import re
_pos_pattern = re.compile(r"\d{4,5}/\d{3}")
drop_cols = [c for c in combined_meta.columns if _pos_pattern.search(c)]
clean = combined_meta.drop(columns=drop_cols)
print(f"  dropped {len(drop_cols)} probe-position columns → {len(clean.columns)} remaining")

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
