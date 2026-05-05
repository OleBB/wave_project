"""
Damping-vs-frequency table — CH05 §1 companion to ch05_damping_freq.
====================================================================

Same data path as analysis_scratch/wind_effect_table.py and the figure
plot_damping_freq → ch05_damping_freq_full_{A1,A2,A3}: canon March-2026
lowrange folders, full panel, 1.3–1.6 Hz, quality_flag=ok.

Layout (per amplitude tier A1 / A2 / A3):

                       1.3 Hz  1.4 Hz  1.5 Hz  1.6 Hz
    K_t (uten vind)   ...     ...     ...     ...
    K_t (full vind)   ...     ...     ...     ...
    ΔK_t              ...     ...     ...     ...

Three blocks are stacked into one tabular, separated by \\midrule, so the
reader's eye maps row-by-row onto the three stacked subfigures of
ch05_damping_freq.

Aggregation note (2026-05-05): the input filt is stripped of its `Mooring`
column before being passed to `damping_all_amplitude_grouper`, so each
(freq, amp, wind) cell pools across ALL canon moorings — true n-weighted
mean / true across-canon std / true total n_runs. Earlier versions used
pivot_table(aggfunc="first"), which silently kept exactly one mooring per
cell. See memory/finding_wind_effect_table_aggregation_bias.md.

Outputs:
    output/TABLES/ch05_damping_freq_table.tex   (thesis include)
    analysis_scratch/damping_freq_table.csv     (human-readable companion)

Caption text is read from FIGURE_CAPTIONS["ch05_damping_freq_table"] in
main_save_figures.py via output/.figure_captions.json.
"""

import os
import sys
import warnings
from pathlib import Path
from datetime import datetime as _dt

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
os.chdir(BASE)

from wavescripts.improved_data_loader import load_analysis_data
from wavescripts.filters import (apply_experimental_filters,
                                 damping_all_amplitude_grouper)
from wavescripts.plot_utils import _lookup_central_caption, amp_to_label

# ── I/O ────────────────────────────────────────────────────────────────────
SCRATCH_CSV = Path(__file__).parent / "damping_freq_table.csv"
THESIS_NAME = "ch05_damping_freq_table"
OUT_TEX     = BASE / "output" / "TABLES" / f"{THESIS_NAME}.tex"
CHAPTER     = "05"

# Same canon scope as ch05_damping_freq + ch05_wind_effect_table.
RESULTS_DIRS = [
    BASE / "waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange",
    BASE / "waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange",
]

THESIS_FREQS = [1.3, 1.4, 1.5, 1.6]
THESIS_AMPS  = [0.10, 0.20, 0.30]

# ── 1. Load + filter ───────────────────────────────────────────────────────
print("1. Loading canon results folders …")
meta, _, _, _ = load_analysis_data(*[str(d) for d in RESULTS_DIRS],
                                   load_processed=False)
print(f"   {len(meta)} rows total")

_pv = {
    "filters": {
        "WaveAmplitudeInput [Volt]": (0.1, 0.3),
        "WaveFrequencyInput [Hz]":   (1.3, 1.6),
        "WindCondition":             ["no", "full"],
        "PanelCondition":            "full",
    },
    "plotting": {},
}
filt = apply_experimental_filters(meta, _pv)
print(f"   {len(filt)} rows after thesis-scope filter")

# Pool across moorings: see wind_effect_table.py for the rationale. Dropping
# the Mooring column makes damping_all_amplitude_grouper skip Mooring as a
# grouping key, so each (freq, amp, panel, wind) cell pools across all canon
# moorings (n-weighted mean / true std / total n_runs). Replaces the older
# pivot_table(aggfunc="first") pattern that silently kept one mooring per
# cell. See memory/finding_wind_effect_table_aggregation_bias.md.
filt = filt.drop(columns=["Mooring"], errors="ignore")

stats = damping_all_amplitude_grouper(filt)
print(f"   {len(stats)} grouped rows from damping_all_amplitude_grouper")


# ── 2. Pivot → (freq × amp) cells with one column per wind ─────────────────
# After the Mooring drop above, each (freq, amp, wind) cell has at most one
# row in `stats`, so the pivot's aggfunc is a no-op for the K_t pivot;
# n_runs uses sum so the count remains correct if a future input ever
# reintroduces multiple rows per cell.
def _pivot(values: str, aggfunc: str = "mean") -> pd.DataFrame:
    return stats.pivot_table(
        index=["WaveFrequencyInput [Hz]", "WaveAmplitudeInput [Volt]"],
        columns="WindCondition",
        values=values,
        aggfunc=aggfunc,
    ).reset_index()


pivot   = _pivot("mean_out_in").rename(columns={"no": "Kt_nw", "full": "Kt_fw"})
pivot_n = _pivot("n_runs", aggfunc="sum")

table = pivot.copy()
table["Delta_Kt"] = table["Kt_fw"] - table["Kt_nw"]
table["n_nw"]     = pivot_n["no"].astype("Int64")
table["n_fw"]     = pivot_n["full"].astype("Int64")

table = table[
    table["WaveFrequencyInput [Hz]"].isin(THESIS_FREQS)
    & table["WaveAmplitudeInput [Volt]"].isin(THESIS_AMPS)
].copy()
table = table.sort_values(
    ["WaveAmplitudeInput [Volt]", "WaveFrequencyInput [Hz]"]
).reset_index(drop=True)

n_before = len(table)
table = table.dropna(subset=["Kt_nw", "Kt_fw"]).reset_index(drop=True)
n_dropped = n_before - len(table)
if n_dropped:
    print(f"   {n_dropped} (freq, amp) cell(s) dropped — at least one wind missing")
print(f"   {len(table)} cells kept\n")
print(table.round(3).to_string(index=False))

SCRATCH_CSV.parent.mkdir(parents=True, exist_ok=True)
table.to_csv(SCRATCH_CSV, index=False)
print(f"\n   CSV → {SCRATCH_CSV.relative_to(BASE)}")


# ── 3. Render LaTeX table ──────────────────────────────────────────────────
def _fmt_signed(x: float, decimals: int = 3) -> str:
    if pd.isna(x):
        return "—"
    return f"{x:+.{decimals}f}"


def _fmt_unsigned(x: float, decimals: int = 3) -> str:
    if pd.isna(x):
        return "—"
    return f"{x:.{decimals}f}"


def _row_for(amp: float, kind: str) -> str:
    """One LaTeX row for a (amp, kind) cell across THESIS_FREQS columns.

    kind ∈ {"nw", "fw", "delta"} selects which value to render.
    """
    sub = table[np.isclose(table["WaveAmplitudeInput [Volt]"], amp)]
    cells = []
    for f in THESIS_FREQS:
        row = sub[np.isclose(sub["WaveFrequencyInput [Hz]"], f)]
        if row.empty:
            cells.append("—")
            continue
        r = row.iloc[0]
        if kind == "nw":
            cells.append(_fmt_unsigned(r["Kt_nw"], 3))
        elif kind == "fw":
            cells.append(_fmt_unsigned(r["Kt_fw"], 3))
        elif kind == "delta":
            cells.append(_fmt_signed(r["Delta_Kt"], 3))
        else:
            cells.append("—")
    return " & ".join(cells)


# Three row-blocks (A1/A2/A3), each three rows. The amp tier label sits
# in the first column of the first row of each block; the cell stays
# blank for the other two rows so the reader's eye groups them visually
# without needing the multirow package.
body_lines: list[str] = []
for i, amp in enumerate(THESIS_AMPS):
    label = amp_to_label(amp)        # e.g. "$A_1$"
    if i > 0:
        body_lines.append("    \\midrule")
    body_lines.append(
        f"    {label} & $K_t$ (uten vind) & {_row_for(amp, 'nw')} \\\\"
    )
    body_lines.append(
        f"          & $K_t$ (full vind) & {_row_for(amp, 'fw')} \\\\"
    )
    body_lines.append(
        f"          & $\\Delta K_t$       & {_row_for(amp, 'delta')} \\\\"
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


# IMMUTABLE provenance block — same pattern as wind_effect_table.py.
n_total_runs = int(table["n_nw"].fillna(0).sum() + table["n_fw"].fillna(0).sum())
immutable = "\n".join([
    "%! TEX root = ../main.tex",
    "% ==============================================================",
    "% IMMUTABLE — generated automatically, do not edit this block",
    "%",
    "% — Provenance ───────────────────────────────────────────────────",
    "%   script            : analysis_scratch/damping_freq_table.py",
    "%   plot_type         : damping_freq_table",
    f"%   chapter           : {CHAPTER}",
    f"%   generated_at      : {_dt.now().isoformat(timespec='seconds')}",
    f"%   caption_label     : tab:{THESIS_NAME}",
    f"%   caption_short     : {caption_short}",
    "%",
    "% — Filters ────────────────────────────────────────────────────",
    "%   panel             : full",
    "%   wind              : no, full",
    f"%   amplitude [V]     : {', '.join(f'{a:.1f}' for a in THESIS_AMPS)}",
    f"%   frequency [Hz]    : {', '.join(f'{f:.1f}' for f in THESIS_FREQS)}",
    "%   quality_flag      : ok",
    "%",
    "% — Data provenance ────────────────────────────────────────────",
    f"%   n_cells           : {len(table)}",
    f"%   n_runs (nw + fw)  : {n_total_runs}",
    "%   datasets        :",
    *[f"%     {p.name}" for p in RESULTS_DIRS],
    "%",
    "% — Method ────────────────────────────────────────────────────",
    "%   grouper           : damping_all_amplitude_grouper",
    "%   mooring pooling   : Mooring column dropped pre-grouper, so each",
    "%                       (freq, amp, wind) cell pools across all canon",
    "%                       moorings (n-weighted mean / true std / total n).",
    "%   metric_definitions:",
    "%     K_t (uten vind)   : mean OUT/IN(FFT) at no-wind",
    "%     K_t (full vind)   : mean OUT/IN(FFT) at full-wind",
    "%     Delta K_t         : Kt_fw - Kt_nw  (signed, raw ratio units)",
    "%",
    "% ── end immutable block ─────────────────────────────────────────",
])

# Three row-groups visually mirror the three stacked subfigures of
# ch05_damping_freq (A1 → A2 → A3 from top to bottom).
table_body = (
    "\\begin{table}[htbp]\n"
    "  \\centering\n"
    "  \\small\n"
    "  \\begin{tabular}{ll cccc}\n"
    "    \\toprule\n"
    "     &  & 1.3\\,Hz & 1.4\\,Hz & 1.5\\,Hz & 1.6\\,Hz \\\\\n"
    "    \\midrule\n"
    + "\n".join(body_lines) + "\n"
    "    \\bottomrule\n"
    "  \\end{tabular}\n"
    + caption_block
    + f"  \\label{{tab:{THESIS_NAME}}}\n"
    "\\end{table}\n"
)

OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
OUT_TEX.write_text(immutable + "\n" + table_body, encoding="utf-8")
print(f"   TEX → {OUT_TEX.relative_to(BASE)}")

print("\nDone.")
