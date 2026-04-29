"""
Parasitic-wave timing budget — CH04.
====================================

Companion to ch04_window_intervals (formula values) and to the squeeze
study figures (empirical stability). This table provides the analytical
half of H&G's window justification:

    The window must end BEFORE the parasitic free second-harmonic
    waves reach the probe.

Free 2f waves travel at c_g(2f), which in deep water (kh ≫ 1) is exactly
c_g(f)/2 — half the speed of the main paddle wave. They arrive at a
probe at radius r at:

    t_paras(f, r) = r / c_g(2·f, h)

The proposed pipeline window ends at:

    t_end(f, r) = r / c_g(f, h) + (N_offset + N_length) / f
                = t_arr + 25/f         (with N_offset=10, N_length=15)

For the proposed squeeze (N_offset=10, N_length=15) to be safe, we need
t_end < t_paras at every frequency × probe combination. This table
tabulates the margin and shows it is positive at every cell — the
binding case is 1.3 Hz at IN+OUT (smallest margins).

Outputs:
    output/TABLES/ch04_parasitic_budget.tex  (thesis include)
    analysis_scratch/parasitic_budget_table.csv  (companion CSV)

Caption read from FIGURE_CAPTIONS["ch04_parasitic_budget"] in
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

from wavescripts.constants import c_group, HG
from wavescripts.plot_utils import _lookup_central_caption

# ── I/O ────────────────────────────────────────────────────────────────────
SCRATCH_CSV = Path(__file__).parent / "parasitic_budget_table.csv"
THESIS_NAME = "ch04_parasitic_budget"
OUT_TEX     = BASE / "output" / "TABLES" / f"{THESIS_NAME}.tex"
CHAPTER     = "04"

# ── Formula parameters (must match window_intervals_table.py and the
#    proposed pipeline formula). ─────────────────────────────────────────
THESIS_FREQS    = [1.3, 1.4, 1.5, 1.6]
N_OFFSET        = 10.0     # PROPOSED post-squeeze
N_LENGTH        = 15.0     # PROPOSED post-squeeze
TANK_DEPTH_M    = HG.TANK_DEPTH_M    # 0.58 m
PROBES          = [
    ("IN",  9.373,  "9373/170, 9373/340"),
    ("OUT", 12.400, "12400/250"),
]


# ── 1. Compute the table ───────────────────────────────────────────────────
print("1. Computing parasitic-wave timing budget …")
records = []
for label, r_m, probe_names in PROBES:
    for f in THESIS_FREQS:
        cg_f   = c_group(f,        TANK_DEPTH_M)
        cg_2f  = c_group(2.0 * f,  TANK_DEPTH_M)
        t_arr  = r_m / cg_f
        t_paras = r_m / cg_2f
        t_end  = t_arr + (N_OFFSET + N_LENGTH) / f
        margin_s = t_paras - t_end
        margin_T = margin_s * f
        records.append({
            "probe":         label,
            "r_m":           r_m,
            "freq_hz":       f,
            "t_arr_s":       t_arr,
            "t_end_s":       t_end,
            "t_paras_s":     t_paras,
            "margin_s":      margin_s,
            "margin_T":      margin_T,
            "c_g_f_m_s":     cg_f,
            "c_g_2f_m_s":    cg_2f,
        })

table = pd.DataFrame(records)
print(table.round(3).to_string(index=False))


# ── 2. Save companion CSV ──────────────────────────────────────────────────
SCRATCH_CSV.parent.mkdir(parents=True, exist_ok=True)
table.to_csv(SCRATCH_CSV, index=False)
print(f"\n   CSV → {SCRATCH_CSV.relative_to(BASE)}")


# ── 3. Render LaTeX table ──────────────────────────────────────────────────
def _row(metric_label_tex: str, key: str, fmt: str = "{:.2f}") -> str:
    """One LaTeX row: metric label + 4 frequency cells."""
    cells = []
    for f in THESIS_FREQS:
        v = table.loc[(table["probe"] == _CURRENT_PROBE) & (table["freq_hz"] == f), key].iloc[0]
        cells.append(f"$\\num{{{fmt.format(v).strip()}}}$")
    return (
        f"    {metric_label_tex} &\n"
        "      " + " &\n      ".join(cells) + " \\\\"
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
    "%   script            : analysis_scratch/parasitic_budget_table.py",
    "%   plot_type         : parasitic_budget_table",
    f"%   chapter           : {CHAPTER}",
    f"%   generated_at      : {_dt.now().isoformat(timespec='seconds')}",
    f"%   caption_label     : tab:{THESIS_NAME}",
    f"%   caption_short     : {caption_short}",
    "%",
    "% — Method ────────────────────────────────────────────────────",
    "%   t_arr(f, r)       = r / c_g(f, h)            (main wave arrival)",
    "%   t_paras(f, r)     = r / c_g(2f, h)           (free 2nd-harmonic arrival)",
    "%   t_end(f, r)       = t_arr + (N_offset + N_length) / f",
    f"%   N_offset          : {N_OFFSET} periods (post-squeeze)",
    f"%   N_length          : {N_LENGTH} periods (post-squeeze)",
    f"%   tank depth h      : {TANK_DEPTH_M} m",
    "%   c_g dispersion    : full ω²=gk·tanh(kh) via wavescripts.constants.c_group",
    "%   reference         : Huseby & Grue (2000), J. Fluid Mech. — window must end",
    "%                       before free-2nd-harmonic waves reach the probe.",
    "%",
    "% — Inputs ────────────────────────────────────────────────────",
    f"%   frequencies [Hz]  : {', '.join(f'{f:.1f}' for f in THESIS_FREQS)}",
    f"%   probes            : IN at r={PROBES[0][1]} m (9373/170, 9373/340),",
    f"%                       OUT at r={PROBES[1][1]} m (12400/250)",
    "%",
    "% — Headline ──────────────────────────────────────────────────",
    f"%   binding case      : 1.3 Hz, IN probe (smallest margin)",
    f"%   margin at 1.3 Hz IN: "
    f"{table.loc[(table['probe']=='IN') & (table['freq_hz']==1.3), 'margin_s'].iloc[0]:.2f} s "
    f"({table.loc[(table['probe']=='IN') & (table['freq_hz']==1.3), 'margin_T'].iloc[0]:.1f} T)",
    f"%   margin at 1.3 Hz OUT: "
    f"{table.loc[(table['probe']=='OUT') & (table['freq_hz']==1.3), 'margin_s'].iloc[0]:.2f} s "
    f"({table.loc[(table['probe']=='OUT') & (table['freq_hz']==1.3), 'margin_T'].iloc[0]:.1f} T)",
    "%",
    "% ── end immutable block ─────────────────────────────────────────",
])

# Build body — one (multicolumn header + 4 metric rows) block per probe.
body_lines: list[str] = []
for label, r_m, probe_names in PROBES:
    global _CURRENT_PROBE
    _CURRENT_PROBE = label
    body_lines.append(
        f"    \\multicolumn{{5}}{{l}}{{\\textbf{{{label}-probe "
        f"($r = \\qty{{{r_m}}}{{\\meter}}$, {probe_names})}}}} \\\\"
    )
    body_lines.append(
        _row(r"Bølge ankommer $t_\mathrm{arr}$ [\unit{\second}]",          "t_arr_s")
    )
    body_lines.append(
        _row(r"Vinduslutt $t_\mathrm{end}$ [\unit{\second}]",              "t_end_s")
    )
    body_lines.append(
        _row(r"Andreharmonisk ankommer $t_\mathrm{2f}$ [\unit{\second}]", "t_paras_s")
    )
    body_lines.append(
        _row(r"Margin $t_\mathrm{2f}-t_\mathrm{end}$ [\unit{\second}]",   "margin_s")
    )
    if label != PROBES[-1][0]:
        body_lines.append("    \\midrule")

table_body = (
    "\\begin{table}[hbt]\n"
    "  \\centering\n"
    + caption_block
    + f"  \\label{{tab:{THESIS_NAME}}}\n"
    "  \\begin{tabular}{lcccc}\n"
    "    \\toprule\n"
    "    Frekvens [\\unit{\\hertz}] &\n"
    "      " + " &\n      ".join(f"$\\num{{{f}}}$" for f in THESIS_FREQS) + " \\\\\n"
    "    \\midrule\n"
    + "\n".join(body_lines) + "\n"
    "    \\bottomrule\n"
    "  \\end{tabular}\n"
    "\\end{table}\n"
)

OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
OUT_TEX.write_text(immutable + "\n" + table_body, encoding="utf-8")
print(f"   TEX → {OUT_TEX.relative_to(BASE)}")

print("\nDone.")
