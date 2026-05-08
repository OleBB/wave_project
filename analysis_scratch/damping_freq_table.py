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
    output/TABLES/data/ch05_damping_freq_table.csv       (render-shape data)
    output/TABLES/data/ch05_damping_freq_table.meta.json (provenance)
    output/TABLES/ch05_damping_freq_table.tex            (thesis include)
    analysis_scratch/damping_freq_table.csv              (audit-trail companion)

Caption text is read from FIGURE_CAPTIONS["ch05_damping_freq_table"] in
main_save_figures.py via output/.figure_captions.json.
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

from wavescripts.improved_data_loader import load_analysis_data
from wavescripts.filters import (apply_experimental_filters,
                                 damping_all_amplitude_grouper)
from wavescripts.plot_utils import _lookup_central_caption, amp_to_label
from wavescripts.table_render import render_table

# ── I/O ────────────────────────────────────────────────────────────────────
SCRATCH_CSV   = Path(__file__).parent / "damping_freq_table.csv"
THESIS_NAME   = "ch05_damping_freq_table"
DATA_DIR      = BASE / "output" / "TABLES" / "data"
RENDER_CSV    = DATA_DIR / f"{THESIS_NAME}.csv"
META_JSON     = DATA_DIR / f"{THESIS_NAME}.meta.json"
OUT_TEX       = BASE / "output" / "TABLES" / f"{THESIS_NAME}.tex"
CHAPTER       = "05"
SCRIPT_REL    = "analysis_scratch/damping_freq_table.py"

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
print(f"\n   audit CSV → {SCRATCH_CSV.relative_to(BASE)}")

# ── 3. Reshape into render-shape (one row per output table line) ───────────
# Three logical metric rows per amplitude tier.
KIND_ORDER  = ["nw", "fw", "delta"]
KIND_LABELS = {
    "nw":    r"$K_t$ (uten vind)",
    "fw":    r"$K_t$ (full vind)",
    "delta": r"$\Delta K_t$      ",
}
FREQ_COLS = [f"f_{f:.1f}" for f in THESIS_FREQS]


def _value(row: pd.Series, kind: str, freq: float) -> float:
    if kind == "nw":
        return float(row["Kt_nw"])
    if kind == "fw":
        return float(row["Kt_fw"])
    return float(row["Delta_Kt"])


render_rows: list[dict] = []
for i, amp in enumerate(THESIS_AMPS):
    sub = table[np.isclose(table["WaveAmplitudeInput [Volt]"], amp)]
    for j, kind in enumerate(KIND_ORDER):
        rec: dict = {
            "amp_volt":          amp,
            "amp_label_display": amp_to_label(amp) if j == 0 else "     ",
            "kind":              kind,
            "kind_label":        KIND_LABELS[kind],
        }
        for f, col in zip(THESIS_FREQS, FREQ_COLS):
            r = sub[np.isclose(sub["WaveFrequencyInput [Hz]"], f)]
            rec[col] = float("nan") if r.empty else _value(r.iloc[0], kind, f)
        render_rows.append(rec)

render_df = pd.DataFrame(render_rows)

DATA_DIR.mkdir(parents=True, exist_ok=True)
render_df.to_csv(RENDER_CSV, index=False)
print(f"   render CSV → {RENDER_CSV.relative_to(BASE)}")


# ── 4. Build provenance meta.json ──────────────────────────────────────────
caption_full  = _lookup_central_caption(THESIS_NAME, kind="full") or None
caption_short = _lookup_central_caption(THESIS_NAME, kind="short") or None

n_total_runs = int(table["n_nw"].fillna(0).sum() + table["n_fw"].fillna(0).sum())

meta_payload = {
    "script":          SCRIPT_REL,
    "plot_type":       "damping_freq_table",
    "chapter":         CHAPTER,
    "caption_label":   f"tab:{THESIS_NAME}",
    "caption_short":   caption_short or "",
    "sections": [
        {
            "title": "Filters",
            "lines": [
                "panel             : full",
                "wind              : no, full",
                f"amplitude [V]     : {', '.join(f'{a:.1f}' for a in THESIS_AMPS)}",
                f"frequency [Hz]    : {', '.join(f'{f:.1f}' for f in THESIS_FREQS)}",
                "quality_flag      : ok",
            ],
        },
        {
            "title": "Data provenance",
            "lines": [
                f"n_cells           : {len(table)}",
                f"n_runs (nw + fw)  : {n_total_runs}",
                "datasets        :",
                *[f"  {p.name}" for p in RESULTS_DIRS],
            ],
        },
        {
            "title": "Method",
            "lines": [
                "grouper           : damping_all_amplitude_grouper",
                "mooring pooling   : Mooring column dropped pre-grouper, so each",
                "                    (freq, amp, wind) cell pools across all canon",
                "                    moorings (n-weighted mean / true std / total n).",
                "metric_definitions:",
                "  K_t (uten vind)   : mean OUT/IN(FFT) at no-wind",
                "  K_t (full vind)   : mean OUT/IN(FFT) at full-wind",
                "  Delta K_t         : Kt_fw - Kt_nw  (signed, raw ratio units)",
            ],
        },
    ],
}

META_JSON.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
print(f"   meta JSON  → {META_JSON.relative_to(BASE)}")


# ── 5. Cell formatters ─────────────────────────────────────────────────────
def _fmt_signed(x: float, decimals: int = 3) -> str:
    if pd.isna(x):
        return "—"
    return f"{x:+.{decimals}f}"


def _fmt_unsigned(x: float, decimals: int = 3) -> str:
    if pd.isna(x):
        return "—"
    return f"{x:.{decimals}f}"


def _fmt_value_cell(freq_col: str):
    def _impl(row: pd.Series) -> str:
        v = row[freq_col]
        if row["kind"] == "delta":
            return _fmt_signed(v, 3)
        return _fmt_unsigned(v, 3)
    return _impl


cell_format = {
    "amp_label_display": lambda r: str(r["amp_label_display"]),
    "kind_label":        lambda r: str(r["kind_label"]),
}
for fc in FREQ_COLS:
    cell_format[fc] = _fmt_value_cell(fc)

columns        = ["amp_label_display", "kind_label"] + FREQ_COLS
column_headers = [
    "", "",
    *[f"{f:.1f}\\,Hz" for f in THESIS_FREQS],
]

row_groups: list[tuple[str | None, callable]] = []
for amp in THESIS_AMPS:
    row_groups.append(
        (None, (lambda a: lambda df: df[np.isclose(df["amp_volt"], a)])(amp))
    )


# ── 6. Render ──────────────────────────────────────────────────────────────
render_table(
    csv_path=RENDER_CSV,
    meta_path=META_JSON,
    out_tex_path=OUT_TEX,
    columns=columns,
    column_headers=column_headers,
    column_spec="ll cccc",
    cell_format=cell_format,
    row_groups=row_groups,
    label=f"tab:{THESIS_NAME}",
    caption=caption_full,
    short_caption=caption_short,
)

print(f"   TEX → {OUT_TEX.relative_to(BASE)}")
print("\nDone.")
