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
    output/TABLES/ch04_window_intervals.tex  (thesis include)
    analysis_scratch/window_intervals_table.csv  (human-readable companion)

Caption text is read from FIGURE_CAPTIONS["ch04_window_intervals"] in
main_save_figures.py via output/.figure_captions.json.
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import os
os.chdir(BASE)

from wavescripts.constants import c_group, HG, MEASUREMENT
from wavescripts.plot_utils import _lookup_central_caption

# ── I/O ────────────────────────────────────────────────────────────────────
SCRATCH_CSV = Path(__file__).parent / "window_intervals_table.csv"
THESIS_NAME = "ch04_window_intervals"
OUT_TEX     = BASE / "output" / "TABLES" / f"{THESIS_NAME}.tex"
CHAPTER     = "04"

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


# ── 1. Compute the table ───────────────────────────────────────────────────
print("1. Computing theoretical H&G window intervals across thesis freqs …")
rows = []
for f in THESIS_FREQS:
    in_s,  in_e  = proposed_window(R_IN_M,  f)
    out_s, out_e = proposed_window(R_OUT_M, f)
    samples_per_period = FS / f
    rows.append({
        "freq_hz":             f,
        "in_start_s":          in_s,
        "in_end_s":            in_e,
        "out_start_s":         out_s,
        "out_end_s":           out_e,
        "samples_per_period":  samples_per_period,
    })

table = pd.DataFrame(rows)
print(table.round(3).to_string(index=False))


# ── 2. Save companion CSV ──────────────────────────────────────────────────
SCRATCH_CSV.parent.mkdir(parents=True, exist_ok=True)
table.to_csv(SCRATCH_CSV, index=False)
print(f"\n   CSV → {SCRATCH_CSV.relative_to(BASE)}")


# ── 3. Render LaTeX table ──────────────────────────────────────────────────
def _fmt_range(a: float, b: float, decimals: int = 1) -> str:
    """Render an interval as `\tabnumrange{a}{b}`."""
    return f"\\tabnumrange{{{a:.{decimals}f}}}{{{b:.{decimals}f}}}"


# Header columns: one per frequency
freq_cells = " &\n      ".join(
    f"$\\num{{{f}}}$" for f in THESIS_FREQS
)
in_cells = " &\n      ".join(
    _fmt_range(r["in_start_s"], r["in_end_s"], 1)
    for _, r in table.iterrows()
)
out_cells = " &\n      ".join(
    _fmt_range(r["out_start_s"], r["out_end_s"], 1)
    for _, r in table.iterrows()
)
spp_cells = " &\n      ".join(
    f"$\\num{{{r['samples_per_period']:.10f}}}$"
    for _, r in table.iterrows()
)

# Caption from central FIGURE_CAPTIONS dict (via JSON cache).
caption_full  = _lookup_central_caption(THESIS_NAME, kind="full")
caption_short = _lookup_central_caption(THESIS_NAME, kind="short")

if caption_full:
    if caption_short:
        caption_block = (
            f"  \\caption[{caption_short}]{{\n"
            f"    {caption_full}\n"
            f"  }}\n"
        )
    else:
        caption_block = (
            f"  \\caption{{\n"
            f"    {caption_full}\n"
            f"  }}\n"
        )
else:
    caption_block = (
        "  \\caption{\n"
        "    % TODO: write caption\n"
        "  }\n"
    )

# IMMUTABLE provenance block.
from datetime import datetime as _dt
immutable = "\n".join([
    "%! TEX root = ../main.tex",
    "% ==============================================================",
    "% IMMUTABLE — generated automatically, do not edit this block",
    "%",
    "% — Provenance ───────────────────────────────────────────────────",
    "%   script            : analysis_scratch/window_intervals_table.py",
    "%   plot_type         : window_intervals_table",
    f"%   chapter           : {CHAPTER}",
    f"%   generated_at      : {_dt.now().isoformat(timespec='seconds')}",
    f"%   caption_label     : tab:{THESIS_NAME}",
    f"%   caption_short     : {caption_short}",
    "%",
    "% — Method ────────────────────────────────────────────────────",
    "%   formula           : t_start = r/c_g(f, h) + N_offset/f, t_end = t_start + 10/f",
    f"%   N_offset          : {N_OFFSET_PERIODS} periods (5 wavemaker-ramp + 10 H&G safety)",
    f"%   window length     : {WINDOW_PERIODS} T",
    f"%   r_IN              : {R_IN_M} m  (probes 9373/170, 9373/340)",
    f"%   r_OUT             : {R_OUT_M} m  (probe 12400/250)",
    f"%   tank depth h      : {TANK_DEPTH_M} m",
    f"%   c_g dispersion    : full ω²=gk·tanh(kh) via wavescripts.constants.c_group",
    f"%   sampling rate     : {FS} Hz",
    "%   note              : table values are the THEORETICAL pre-snap window;"
    " per-run windows snap to ±T upcrossings.",
    "%",
    "% — Inputs ────────────────────────────────────────────────────",
    f"%   frequencies [Hz]  : {', '.join(f'{f:.1f}' for f in THESIS_FREQS)}",
    "%",
    "% ── end immutable block ─────────────────────────────────────────",
])

table_body = (
    "\\begin{table}[hbt]\n"
    "  \\centering\n"
    + caption_block
    + f"  \\label{{tab:{THESIS_NAME}}}\n"
    "  \\begin{tabular}{lcccc}\n"
    "    \\toprule\n"
    "    Frekvens [\\unit{\\hertz}] &\n"
    f"      {freq_cells} \\\\\n"
    "    \\midrule\n"
    "    Innkommende  [\\unit{\\second}] &\n"
    f"      {in_cells} \\\\\n"
    "    Utgående  [\\si{\\second}] &\n"
    f"      {out_cells} \\\\\n"
    "    Antall samples per periode [\\textendash] &\n"
    f"      {spp_cells} \\\\\n"
    "    \\bottomrule\n"
    "  \\end{tabular}\n"
    "\\end{table}\n"
)

OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
OUT_TEX.write_text(immutable + "\n" + table_body, encoding="utf-8")
print(f"   TEX → {OUT_TEX.relative_to(BASE)}")

print("\nDone.")
