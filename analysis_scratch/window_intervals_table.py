"""
H&G window intervals table — CH04.
==================================

Per thesis frequency (1.3, 1.4, 1.5, 1.6 Hz), this table records:

  - Innkommende [s]  : IN-probe window [t_start, t_end] (theoretical, pre-snap)
  - Utgående  [s]    : OUT-probe window [t_start, t_end] (theoretical, pre-snap)
  - Antall samples per periode : Fs / f  (sampling rate ÷ paddle frequency)

The window times come from the proposed CH04 §4 formula:

    t_start(probe) = r_probe / c_g(f, h) + N_offset / f       (seconds)
    t_end(probe)   = t_start + 10 / f                         (10T length)

with N_offset = 15 (5 wavemaker-ramp + 10 H&G "10 periods after arrival"),
r_IN = 9.373 m, r_OUT = 12.400 m, h = 0.58 m. Group velocity c_g uses
the full dispersion relation ω² = g·k·tanh(k·h) via wavescripts.constants.c_group.

The table reports the THEORETICAL (pre-snap) window — the deterministic
output of the formula. Per-run windows differ from these by the ±T
upcrossing snap (typically a few samples) which is a per-run thing and
not table-friendly. See ch04_hg_per40_window_fitness_f{13,14,15,16}.pdf
for the snapped windows visualised against η(t).

Outputs:
    output/TABLES/data/ch04_window_intervals.csv       (render-shape data, 3 rows × per-freq cols)
    output/TABLES/data/ch04_window_intervals.meta.json (provenance)
    analysis_scratch/window_intervals_table.csv        (audit-trail companion, long form)

Caption text is owned by main_save_tables.py (TABLE_CAPTIONS).
"""

import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

BASE = (Path(__file__).resolve().parent.parent
        if "__file__" in globals() else Path.cwd())
sys.path.insert(0, str(BASE))
os.chdir(BASE)

from wavescripts.constants import c_group, HG, MEASUREMENT

# ── I/O ────────────────────────────────────────────────────────────────────
THESIS_NAME = "ch04_window_intervals"
CHAPTER     = "04"
SCRIPT_REL  = "analysis_scratch/window_intervals_table.py"

DATA_DIR    = BASE / "output" / "TABLES" / "data"
RENDER_CSV  = DATA_DIR / f"{THESIS_NAME}.csv"
META_JSON   = DATA_DIR / f"{THESIS_NAME}.meta.json"
SCRATCH_CSV = Path(__file__).parent / "window_intervals_table.csv" if "__file__" in globals() else BASE / "analysis_scratch" / "window_intervals_table.csv"

# ── Formula parameters (must match analysis_scratch/hg_per40_window_fitness.py
#    and any future pipeline update). ─────────────────────────────────────
THESIS_FREQS     = [1.3, 1.4, 1.5, 1.6]
N_OFFSET_PERIODS = 15.0    # 5 wavemaker-ramp + 10 H&G safety
WINDOW_PERIODS   = 10.0    # H&G's 10T window length
TANK_DEPTH_M     = HG.TANK_DEPTH_M    # 0.58 m
R_IN_M           = 9.373              # IN reference probe (9373/170, 9373/340)
R_OUT_M          = 12.400             # OUT probe (12400/250)
FS               = float(MEASUREMENT.SAMPLING_RATE)


def proposed_window(r_m: float, f_hz: float):
    """(t_start, t_end) seconds — theoretical pre-snap window."""
    cg = c_group(f_hz, TANK_DEPTH_M)
    t_start = r_m / cg + N_OFFSET_PERIODS / f_hz
    t_end   = t_start + WINDOW_PERIODS / f_hz
    return t_start, t_end


# ── 1. Compute the per-frequency values ────────────────────────────────────
print("1. Computing theoretical H&G window intervals across thesis freqs …")
long_rows = []
for f in THESIS_FREQS:
    in_s,  in_e  = proposed_window(R_IN_M,  f)
    out_s, out_e = proposed_window(R_OUT_M, f)
    samples_per_period = FS / f
    long_rows.append({
        "freq_hz":             f,
        "in_start_s":          in_s,
        "in_end_s":            in_e,
        "out_start_s":         out_s,
        "out_end_s":           out_e,
        "samples_per_period":  samples_per_period,
    })

long_df = pd.DataFrame(long_rows)
print(long_df.round(3).to_string(index=False))

# Audit-trail CSV (long form).
SCRATCH_CSV.parent.mkdir(parents=True, exist_ok=True)
long_df.to_csv(SCRATCH_CSV, index=False)
print(f"\n   audit CSV → {SCRATCH_CSV.relative_to(BASE)}")


# ── 2. Reshape to render shape: 3 rows × per-freq columns ──────────────────
# Each output row in the rendered table is one (label, kind) pair. The
# per-freq value is encoded as a string like "27.1|34.8" for ranges, and a
# float for samples_per_period. The render-side cell formatter parses it.
def _freq_col(f: float) -> str:
    return f"f_{f:.1f}"


def _row_for_kind(kind: str, label: str) -> dict:
    rec: dict = {"row_label": label, "kind": kind}
    for r in long_rows:
        f = r["freq_hz"]
        col = _freq_col(f) + "_value"
        if kind == "in_range":
            rec[col] = f"{r['in_start_s']:.6f}|{r['in_end_s']:.6f}"
        elif kind == "out_range":
            rec[col] = f"{r['out_start_s']:.6f}|{r['out_end_s']:.6f}"
        elif kind == "spp":
            rec[col] = f"{r['samples_per_period']:.10f}"
    return rec


render_rows = [
    _row_for_kind("in_range",  r"Innkommende  [\unit{\second}]"),
    _row_for_kind("out_range", r"Utgående  [\si{\second}]"),
    _row_for_kind("spp",       r"Antall samples per periode [\textendash]"),
]
render_df = pd.DataFrame(render_rows)

DATA_DIR.mkdir(parents=True, exist_ok=True)
render_df.to_csv(RENDER_CSV, index=False)
print(f"   render CSV → {RENDER_CSV.relative_to(BASE)}")


# ── 3. Build provenance meta.json ──────────────────────────────────────────
meta_payload = {
    "script":          SCRIPT_REL,
    "plot_type":       "window_intervals_table",
    "chapter":         CHAPTER,
    "caption_label":   f"tab:{THESIS_NAME}",
    "caption_short":   "",
    "sections": [
        {
            "title": "Method",
            "lines": [
                "formula           : t_start = r/c_g(f, h) + N_offset/f, t_end = t_start + 10/f",
                f"N_offset          : {N_OFFSET_PERIODS} periods (5 wavemaker-ramp + 10 H&G safety)",
                f"window length     : {WINDOW_PERIODS} T",
                f"r_IN              : {R_IN_M} m  (probes 9373/170, 9373/340)",
                f"r_OUT             : {R_OUT_M} m  (probe 12400/250)",
                f"tank depth h      : {TANK_DEPTH_M} m",
                "c_g dispersion    : full ω²=gk·tanh(kh) via wavescripts.constants.c_group",
                f"sampling rate     : {FS} Hz",
                "note              : table values are the THEORETICAL pre-snap window;"
                " per-run windows snap to ±T upcrossings.",
            ],
        },
        {
            "title": "Inputs",
            "lines": [
                f"frequencies [Hz]  : {', '.join(f'{f:.1f}' for f in THESIS_FREQS)}",
            ],
        },
    ],
}

META_JSON.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
print(f"   meta JSON  → {META_JSON.relative_to(BASE)}")

print("\nDone.")
