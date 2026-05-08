"""
Wind-effect table — CH05 §3.
============================

Replaces the archived `ch05_damping_wind_delta` 2-row scatter (see
ignore_this_archive/wavescripts/plot_damping_wind_delta_2026-04-28.py).
Same data path: canon March-2026 lowrange folders, full panel, 1.3–1.6 Hz,
quality_flag=ok, both wind conditions.

Per (freq, amp) cell, computes:

    K_t,uten          = mean OUT/IN(FFT) at no-wind
    K_t,vind          = mean OUT/IN(FFT) at full-wind
    ΔK_t              = K_t,vind − K_t,uten             (signed decimal fraction)
    K_t,vind / K_t,uten = transmission ratio   (>1 ↔ wind passes more wave)
    D_vind / D_uten   = damping ratio,  D = 1 − K_t
                        (<1 ↔ wind erodes the panel's damping)

Reading convention: the K_t-ratio and D-ratio cross 1.00 in opposite
directions for the same physical effect — wind making the panel "leak"
more drives K_t-ratio above 1 and D-ratio below 1.

Aggregation note (2026-05-05): the input filt is stripped of its `Mooring`
column before being passed to `damping_all_amplitude_grouper`, so each
(freq, amp, panel, wind) cell pools across ALL canon moorings at once —
true n-weighted mean / true across-canon std / true total n_runs. Earlier
versions used pivot_table(aggfunc="first") which silently kept exactly one
mooring per cell; see memory/finding_wind_effect_table_aggregation_bias.md.

Outputs:
    output/TABLES/ch05_wind_effect_table.tex   (thesis include)
    analysis_scratch/wind_effect_table.csv     (human-readable companion)

Caption text is read from FIGURE_CAPTIONS["ch05_wind_effect_table"] in
main_save_figures.py via output/.figure_captions.json.
"""

import sys
import warnings
from pathlib import Path
import glob

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import os
os.chdir(BASE)

from wavescripts.improved_data_loader import load_analysis_data
from wavescripts.filters import (apply_experimental_filters,
                                 damping_all_amplitude_grouper)
from wavescripts.plot_utils import _lookup_central_caption, amp_to_label

# ── I/O ────────────────────────────────────────────────────────────────────
SCRATCH_CSV = Path(__file__).parent / "wind_effect_table.csv"
THESIS_NAME = "ch05_wind_effect_table"
OUT_TEX     = BASE / "output" / "TABLES" / f"{THESIS_NAME}.tex"
CHAPTER     = "05"

# Same canon scope as ch05_damping_freq + ch05_damping_scatter.
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

# Pool across moorings: dropping the Mooring column makes
# damping_all_amplitude_grouper skip Mooring as a grouping key (filters.py:984
# tests `if "Mooring" in rmdf.columns and rmdf["Mooring"].notna().any():`),
# so each (amp, freq, panel, wind) cell collapses to ONE row across all canon
# moorings. mean_out_in / std_out_in / n_runs are then computed honestly over
# every contributing CSV. Replaces the older pivot_table(aggfunc="first")
# pattern that silently kept one mooring per cell — see
# memory/finding_wind_effect_table_aggregation_bias.md.
filt = filt.drop(columns=["Mooring"], errors="ignore")

stats = damping_all_amplitude_grouper(filt)
print(f"   {len(stats)} grouped rows from damping_all_amplitude_grouper")

# ── 2. Pivot per (freq, amp) → wind columns ────────────────────────────────
# After the Mooring drop above, each (freq, amp, wind) cell has at most one
# row in `stats`, so the pivot's aggfunc is a no-op (any choice would do).
pivot = stats.pivot_table(
    index=["WaveFrequencyInput [Hz]", "WaveAmplitudeInput [Volt]"],
    columns="WindCondition",
    values="mean_out_in",
    aggfunc="mean",
).reset_index()

# Same pivot for std + n.
pivot_std = stats.pivot_table(
    index=["WaveFrequencyInput [Hz]", "WaveAmplitudeInput [Volt]"],
    columns="WindCondition",
    values="std_out_in",
    aggfunc="mean",
).reset_index()

pivot_n = stats.pivot_table(
    index=["WaveFrequencyInput [Hz]", "WaveAmplitudeInput [Volt]"],
    columns="WindCondition",
    values="n_runs",
    aggfunc="sum",
).reset_index()

# Compute the four wind-effect metrics.
table = pivot.rename(columns={"no": "Kt_nw", "full": "Kt_fw"})
table["Delta_Kt"]    = table["Kt_fw"] - table["Kt_nw"]   # absolute change in K_t
table["ratio_Kt"]    = table["Kt_fw"] / table["Kt_nw"]   # K_t,vind / K_t,uten
_D_nw = 1.0 - table["Kt_nw"]
_D_fw = 1.0 - table["Kt_fw"]
table["ratio_D"]     = _D_fw / _D_nw                      # D_vind / D_uten,  D = 1 - K_t
table["std_nw"]      = pivot_std["no"]
table["std_fw"]      = pivot_std["full"]
table["n_nw"]        = pivot_n["no"].astype("Int64")
table["n_fw"]        = pivot_n["full"].astype("Int64")

# Keep only thesis-scope (freq, amp) cells, in the canonical order.
table = table[table["WaveFrequencyInput [Hz]"].isin(THESIS_FREQS)
              & table["WaveAmplitudeInput [Volt]"].isin(THESIS_AMPS)].copy()
table = table.sort_values(["WaveFrequencyInput [Hz]",
                           "WaveAmplitudeInput [Volt]"]).reset_index(drop=True)

# Drop cells where either wind condition is missing (no Δ to compute).
n_before = len(table)
table = table.dropna(subset=["Kt_nw", "Kt_fw"]).reset_index(drop=True)
n_dropped = n_before - len(table)
if n_dropped:
    print(f"   {n_dropped} (freq, amp) cell(s) dropped — at least one wind missing")
print(f"   {len(table)} cells in final table\n")
print(table.round(3).to_string(index=False))

# ── 3. Save companion CSV ──────────────────────────────────────────────────
SCRATCH_CSV.parent.mkdir(parents=True, exist_ok=True)
table.to_csv(SCRATCH_CSV, index=False)
print(f"\n   CSV → {SCRATCH_CSV.relative_to(BASE)}")


# ── 4. Render LaTeX table ──────────────────────────────────────────────────
def _fmt_signed(x: float, decimals: int = 1) -> str:
    """Render a signed number with a leading + on positives."""
    if pd.isna(x):
        return "—"
    return f"{x:+.{decimals}f}"

def _fmt_unsigned(x: float, decimals: int = 3) -> str:
    if pd.isna(x):
        return "—"
    return f"{x:.{decimals}f}"


def _fmt_ratio(x: float, decimals: int = 3) -> str:
    """Unsigned ratio (e.g. 1.181, 0.653). NaN → em-dash."""
    if pd.isna(x):
        return "—"
    return f"{x:.{decimals}f}"


# Body rows. Insert \midrule between frequency blocks for visual grouping.
body_rows = []
last_freq = None
for _, r in table.iterrows():
    f = float(r["WaveFrequencyInput [Hz]"])
    a = float(r["WaveAmplitudeInput [Volt]"])
    if last_freq is not None and f != last_freq:
        body_rows.append(r"\midrule")
    body_rows.append(
        f"  {f:.1f} & {amp_to_label(a)} & "
        f"{_fmt_unsigned(r['Kt_nw'], 3)} & {_fmt_unsigned(r['Kt_fw'], 3)} & "
        f"{_fmt_signed(r['Delta_Kt'], 3)} & "        # ΔK_t as decimal fraction
        f"{_fmt_ratio(r['ratio_Kt'], 3)} & "         # K_t,vind / K_t,uten
        f"{_fmt_ratio(r['ratio_D'],  3)} \\\\"        # D_vind / D_uten
    )
    last_freq = f

# Caption is read from output/.table_captions.json (written by main_save_tables.py).
# Run main_save_tables.py once before this script to populate the cache.
_CAPTIONS_JSON = BASE / "output" / ".table_captions.json"
caption_full  = _lookup_central_caption(THESIS_NAME, kind="full",  json_path=_CAPTIONS_JSON)
caption_short = _lookup_central_caption(THESIS_NAME, kind="short", json_path=_CAPTIONS_JSON)

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

# IMMUTABLE provenance block — same idea as figure stubs but for a table.
from datetime import datetime as _dt
n_total_runs = int(table["n_nw"].fillna(0).sum() + table["n_fw"].fillna(0).sum())
immutable = "\n".join([
    "%! TEX root = ../main.tex",
    "% ==============================================================",
    "% IMMUTABLE — generated automatically, do not edit this block",
    "%",
    "% — Provenance ───────────────────────────────────────────────────",
    "%   script            : analysis_scratch/wind_effect_table.py",
    "%   plot_type         : wind_effect_table",
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
    "%     Kt_nw / Kt_fw      : mean OUT/IN(FFT) at no- / full-wind",
    "%     Delta_Kt           : Kt_fw - Kt_nw  (signed decimal fraction)",
    "%     ratio_Kt           : Kt_fw / Kt_nw  (>1 = wind passes more wave)",
    "%     ratio_D            : D_fw / D_nw    (D = 1 - K_t)",
    "%                          <1 = wind erodes panel damping",
    "%",
    "% ── end immutable block ─────────────────────────────────────────",
])

# Note: the column headers below are intentionally shorter than the verbal
# names in the body of the thesis. The IMMUTABLE block above has the full
# definitions; the table caption + thesis prose can spell them out.
table_body = (
    "\\begin{table}[htbp]\n"
    "  \\centering\n"
    "  \\small\n"
    "  \\begin{tabular}{cc cc r r r}\n"
    "    \\toprule\n"
    "      $f$ [Hz] & Amp & $K_{t,\\text{uten}}$ & $K_{t,\\text{vind}}$ "
    "& $\\Delta K_t$ & "
    "$K_{t,\\text{vind}}/K_{t,\\text{uten}}$ & "
    "$D_{\\text{vind}}/D_{\\text{uten}}$ \\\\\n"
    "    \\midrule\n"
    + "\n".join(body_rows) + "\n"
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
