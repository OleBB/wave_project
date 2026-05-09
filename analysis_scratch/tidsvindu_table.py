"""
Tidsvindu — main-text companion table to ch04_plateau_overview.
================================================================

Per thesis frequency, tabulate:

  - c_g [m/s]                 — group velocity (full dispersion ω²=gk·tanh(kh))
  - Innkommende vindu [s]     — IN-probe window [t_start, t_end], read from meta
  - Utgående vindu [s]        — OUT-probe window, same
  - Δt mellom probene [s]     — propagation delay (= IN→OUT window shift)
  - Vindusbredde [s]          — window length (t_end − t_start)

Window times come from the live pipeline via the meta columns
`Computed Probe {pos} start/end` (in samples, converted to seconds).
Median across canon runs (cond4 March-2026 lowrange, full panel,
quality_ok). The table therefore reflects whatever window the pipeline
is currently producing — when the pipeline is later updated to a new
N_offset / N_length, this table auto-tracks on next regen.

Inferred N_offset and N_length (in periods, derived per f from
median windows) are printed at runtime and recorded in the immutable
provenance block, so it's always clear which window the table
documents.

Outputs:
    output/TABLES/ch04_tidsvindu.tex
    analysis_scratch/tidsvindu.csv
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

from wavescripts.improved_data_loader import load_analysis_data
from wavescripts.constants import c_group, HG, MEASUREMENT
from wavescripts.plot_utils import _lookup_central_caption


# ── Config ──────────────────────────────────────────────────────────────
FS = float(MEASUREMENT.SAMPLING_RATE)

THESIS_FREQS = [1.3, 1.4, 1.5, 1.6]

IN_PROBES  = ["9373/170", "9373/340"]
OUT_PROBE  = "12400/250"
PROBE_R_M  = {"9373/170": 9.373, "9373/340": 9.373, "12400/250": 12.400}
TANK_DEPTH_M = HG.TANK_DEPTH_M

PROCESSED_DIRS = [
    Path("waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange"),
    Path("waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"),
]

THESIS_NAME = "ch04_tidsvindu"
OUT_TEX = BASE / "output" / "TABLES" / f"{THESIS_NAME}.tex"
OUT_CSV = Path(__file__).parent / "tidsvindu.csv"
CHAPTER = "04"


# ── Load meta (canon only) ──────────────────────────────────────────────
print("Loading meta (canon March-2026 lowrange) …")
combined_meta, _, _, _ = load_analysis_data(*PROCESSED_DIRS, load_processed=False)

f_col = pd.to_numeric(combined_meta["WaveFrequencyInput [Hz]"], errors="coerce")
mask = (
    f_col.between(THESIS_FREQS[0] - 0.02, THESIS_FREQS[-1] + 0.02)
    & (combined_meta["PanelCondition"] == "full")
    & (combined_meta["quality_flag"] == "ok")
)
sel = combined_meta[mask].copy()
sel["f_match"] = sel["WaveFrequencyInput [Hz]"].apply(
    lambda v: next((f for f in THESIS_FREQS if abs(f - v) < 0.02), np.nan)
)
sel = sel.dropna(subset=["f_match"])
print(f"   {len(sel)} canon runs at {THESIS_FREQS} Hz, full panel, quality ok.")


def med_window_seconds(sub: pd.DataFrame, probe: str) -> tuple[float, float]:
    """Median (start, end) in seconds across runs, for a given probe."""
    s_col = f"Computed Probe {probe} start"
    e_col = f"Computed Probe {probe} end"
    if s_col not in sub.columns or e_col not in sub.columns:
        return (np.nan, np.nan)
    s_samples = pd.to_numeric(sub[s_col], errors="coerce").dropna()
    e_samples = pd.to_numeric(sub[e_col], errors="coerce").dropna()
    if s_samples.empty or e_samples.empty:
        return (np.nan, np.nan)
    return (float(s_samples.median()) / FS, float(e_samples.median()) / FS)


# ── Per-frequency table ────────────────────────────────────────────────
records = []
for f in THESIS_FREQS:
    sub = sel[sel["f_match"] == f]

    # IN window: pool both 9373/170 and 9373/340 (they share r and the H&G
    # formula → near-identical Computed values).
    in_starts, in_ends = [], []
    for p in IN_PROBES:
        s, e = med_window_seconds(sub, p)
        if np.isfinite(s) and np.isfinite(e):
            in_starts.append(s); in_ends.append(e)
    in_start = float(np.median(in_starts)) if in_starts else np.nan
    in_end   = float(np.median(in_ends))   if in_ends   else np.nan

    out_start, out_end = med_window_seconds(sub, OUT_PROBE)

    cg     = c_group(f, TANK_DEPTH_M)
    dt_in_out = (PROBE_R_M[OUT_PROBE] - PROBE_R_M[IN_PROBES[0]]) / cg
    width  = (in_end - in_start) if np.isfinite(in_end - in_start) else np.nan

    # Inferred N_offset / N_length (periods past arrival)
    t_arr_in = PROBE_R_M[IN_PROBES[0]] / cg
    n_off_inferred = (in_start - t_arr_in) * f if np.isfinite(in_start) else np.nan
    n_len_inferred = width * f if np.isfinite(width) else np.nan

    records.append({
        "freq_hz":      f,
        "n_runs":       len(sub),
        "c_g_m_per_s":  cg,
        "in_start_s":   in_start,
        "in_end_s":     in_end,
        "out_start_s":  out_start,
        "out_end_s":    out_end,
        "delta_t_s":    dt_in_out,
        "width_s":      width,
        "N_off_T":      n_off_inferred,
        "N_len_T":      n_len_inferred,
    })

table = pd.DataFrame(records)
print("\nPer-frequency summary (read from meta):")
print(table.round(3).to_string(index=False))

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
table.to_csv(OUT_CSV, index=False)
print(f"\n   CSV → {OUT_CSV.relative_to(BASE)}")


# ── Inferred window parameters (for provenance block) ──────────────────
n_off_med = float(np.nanmedian(table["N_off_T"]))
n_len_med = float(np.nanmedian(table["N_len_T"]))
print(f"\n   Inferred N_offset ≈ {n_off_med:.2f} T, N_length ≈ {n_len_med:.2f} T")


# ── LaTeX renderer ──────────────────────────────────────────────────────
def fmt_num(v, decimals=2):
    if pd.isna(v):
        return r"\textendash"
    return rf"$\num{{{v:.{decimals}f}}}$"

def fmt_range(a, b, decimals=1):
    if pd.isna(a) or pd.isna(b):
        return r"\textendash"
    return rf"\tabnumrange{{{a:.{decimals}f}}}{{{b:.{decimals}f}}}"


# Column cells per frequency
def cells_for(key, fmt=fmt_num, decimals=2):
    return " &\n      ".join(
        fmt(table[table["freq_hz"] == f][key].iloc[0], decimals)
        for f in THESIS_FREQS
    )

freq_header = " &\n      ".join(
    rf"$\num{{{f}}}$" for f in THESIS_FREQS
)
in_window_cells = " &\n      ".join(
    fmt_range(
        table[table["freq_hz"] == f]["in_start_s"].iloc[0],
        table[table["freq_hz"] == f]["in_end_s"].iloc[0],
        decimals=1,
    )
    for f in THESIS_FREQS
)
out_window_cells = " &\n      ".join(
    fmt_range(
        table[table["freq_hz"] == f]["out_start_s"].iloc[0],
        table[table["freq_hz"] == f]["out_end_s"].iloc[0],
        decimals=1,
    )
    for f in THESIS_FREQS
)
cg_cells       = cells_for("c_g_m_per_s",  decimals=3)
delta_t_cells  = cells_for("delta_t_s",    decimals=2)
width_cells    = cells_for("width_s",      decimals=2)


# Caption is read from output/.table_captions.json (written by main_save_tables.py).
# Run main_save_tables.py once before this script to populate the cache.
_CAPTIONS_JSON = BASE / "output" / ".table_captions.json"
caption_full  = _lookup_central_caption(THESIS_NAME, kind="full",  json_path=_CAPTIONS_JSON)
caption_short = _lookup_central_caption(THESIS_NAME, kind="short", json_path=_CAPTIONS_JSON)
# Wrap the caption in CAPTION-SYNC sentinels (Option D, 2026-05-09) so
# analysis_scratch/sync_captions.py can rewrite the caption later
# without re-running this script.
from wavescripts.plot_utils import wrap_caption_with_sentinels
caption_block = wrap_caption_with_sentinels(caption_full, caption_short)


from datetime import datetime as _dt
immutable = "\n".join([
    "%! TEX root = ../main.tex",
    "% ==============================================================",
    "% IMMUTABLE — generated automatically, do not edit this block",
    "%",
    "% — Provenance ───────────────────────────────────────────────────",
    "%   script            : analysis_scratch/tidsvindu_table.py",
    "%   plot_type         : tidsvindu_table",
    f"%   chapter           : {CHAPTER}",
    f"%   generated_at      : {_dt.now().isoformat(timespec='seconds')}",
    f"%   caption_label     : tab:{THESIS_NAME}",
    f"%   caption_short     : {caption_short}",
    "%",
    "% — Method ────────────────────────────────────────────────────",
    "%   Window times read from meta columns:",
    "%     'Computed Probe {pos} start / end'    (sample indices, ÷ Fs for seconds)",
    "%   Aggregation: median across canon runs at each thesis frequency.",
    "%   IN row: median across both 9373/170 and 9373/340 (probe-shifted formula",
    "%   gives them identical theoretical windows).",
    "%   c_g: full dispersion ω²=gk·tanh(kh) via wavescripts.constants.c_group.",
    "%   Δt = (r_OUT − r_IN) / c_g(f, h)",
    "%",
    "% — Inferred window (from meta data, snap-rounded median) ───────",
    f"%   N_offset (median, in T)   : {n_off_med:.2f}",
    f"%   N_length (median, in T)   : {n_len_med:.2f}",
    "%   These are derived from the cached 'Computed Probe' values; if the",
    "%   pipeline is later updated to a new N_offset / N_length, regenerate",
    "%   this table to track it.",
    "%",
    "% — Geometry ──────────────────────────────────────────────────",
    f"%   r_IN              : {PROBE_R_M[IN_PROBES[0]]} m  ({', '.join(IN_PROBES)})",
    f"%   r_OUT             : {PROBE_R_M[OUT_PROBE]} m  ({OUT_PROBE})",
    f"%   tank depth h      : {TANK_DEPTH_M} m",
    f"%   sampling rate     : {FS} Hz",
    "%",
    "% — Inputs ────────────────────────────────────────────────────",
    f"%   frequencies [Hz]  : {', '.join(f'{f:.1f}' for f in THESIS_FREQS)}",
    "%   filter            : PanelCondition=full, quality_flag=ok, canon March-2026 cond4 lowrange",
    "%   n_runs per freq   : "
    + ", ".join(f"{f:.1f}={int(table[table['freq_hz']==f]['n_runs'].iloc[0])}"
                for f in THESIS_FREQS),
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
    f"      {freq_header} \\\\\n"
    "    \\midrule\n"
    "    $c_g$ [\\unit{\\meter\\per\\second}] &\n"
    f"      {cg_cells} \\\\\n"
    "    $\\Delta t$, reisetid for $c_g$ [\\unit{\\second}] &\n"
    f"      {delta_t_cells} \\\\\n"
    "    Innkommende vindu [\\unit{\\second}] &\n"
    f"      {in_window_cells} \\\\\n"
    "    Utgående vindu [\\unit{\\second}] &\n"
    f"      {out_window_cells} \\\\\n"
    "    Tidsvinduets lengde [\\unit{\\second}] &\n"
    f"      {width_cells} \\\\\n"
    "    \\bottomrule\n"
    "  \\end{tabular}\n"
    "\\end{table}\n"
)

OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
OUT_TEX.write_text(immutable + "\n" + table_body, encoding="utf-8")
print(f"   TEX → {OUT_TEX.relative_to(BASE)}")
