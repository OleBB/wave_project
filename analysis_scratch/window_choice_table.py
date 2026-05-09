"""
Window-choice tables — CH04 §4o, nowind + fullwind side by side.
==================================================================

Per (f, probe), tabulate:

  - t_arr [s]                  main wave arrival, r / c_g(f, h)
  - window [s]                 chosen FFT window [t_start, t_end] under Option B:
                                 t_start = r/c_g(f,h) + 10/f
                                 t_end   = t_start + N(f)/f
                                 N(f)    = {1.3: 10, 1.4: 13, 1.5: 13, 1.6: 13}
  - Δ [s]                      window length (t_end - t_start)
  - t_paras [s]                free 2nd-harmonic arrival, r / c_g(2f, h)
  - plateau window [s]         empirical plateau bounds:
                                 nowind   →  eyeball  from snarvei_eyeballing.md (Day 1, 0.2 V)
                                 fullwind →  per40 sliding A_FFT, ±2 % relaxed
                                              criterion, from per40_plateau_end_aggregated.csv
  - N [periods]                window length in periods

Two tables emitted:
    output/TABLES/ch04_window_choice_nowind.tex     (eyeball plateau)
    output/TABLES/ch04_window_choice_fullwind.tex   (empirical plateau)

Companion CSVs:
    analysis_scratch/window_choice_nowind.csv
    analysis_scratch/window_choice_fullwind.csv
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


# ── Option B parameters (CH04 §4o post-squeeze choice) ──────────────────
THESIS_FREQS     = [1.3, 1.4, 1.5, 1.6]
N_OFFSET_T       = 10.0
N_LENGTH_LOOKUP  = {1.3: 10, 1.4: 13, 1.5: 13, 1.6: 13}
TANK_DEPTH_M     = HG.TANK_DEPTH_M
PROBES = [
    ("IN",  9.373,  "9373/170, 9373/340"),
    ("OUT", 12.400, "12400/250"),
]
CHAPTER = "04"

# ── Eyeball plateau from analysis_scratch/snarvei_eyeballing.md (Day 1, 0.2 V)
#    Read off RampDetectionBrowser, NOWIND ONLY. ─────────────────────────
EYEBALL_NOWIND = {
    "IN":  {1.3: (22.0, 39.0), 1.4: (22.0, 39.0), 1.5: (24.0, 37.0), 1.6: (26.0, 37.0)},
    "OUT": {1.3: (27.0, 44.0), 1.4: (27.0, 43.0), 1.5: (28.0, 41.0), 1.6: (29.0, 41.0)},
}

# ── Empirical plateau end (fullwind, per40, ±2 % relaxed criterion).
#    Read from per40_plateau_end_aggregated.csv produced by
#    analysis_scratch/per40_plateau_end_measurement.py.
#    Start = t_arr + 8/f (definitionally-safe — the script's plateau-reference
#    region begins there). End = plat_end_relax_s_med. ──────────────────
PER40_AGG_CSV = Path("analysis_scratch/per40_plateau_end_aggregated.csv")


# ── Window geometry helpers ─────────────────────────────────────────────
def t_arr_at(r_m: float, f: float) -> float:
    return r_m / c_group(f, TANK_DEPTH_M)

def t_paras_at(r_m: float, f: float) -> float:
    return r_m / c_group(2.0 * f, TANK_DEPTH_M)

def proposed_window(r_m: float, f: float):
    t_start = t_arr_at(r_m, f) + N_OFFSET_T / f
    t_end   = t_start + N_LENGTH_LOOKUP[f] / f
    return t_start, t_end


# ── Build per-row records ───────────────────────────────────────────────
agg = pd.read_csv(PER40_AGG_CSV)


def empirical_fullwind_plateau(probe_label: str, f: float):
    """[start, end] from per40 fullwind sliding-AFFT plateau measurement.

    Start  = t_arr + 8/f  (script's plateau-reference window low edge)
    End    = plat_end_relax_s_med  (median across the per40 cohort, n=1 per
                                    (f, probe, fullwind, per40) cell on
                                    canon March-2026 lowrange)
    Returns (start, end) in seconds; returns (np.nan, np.nan) if missing.
    """
    sub = agg[(agg["freq_hz"] == f) & (agg["probe"] == probe_label)
              & (agg["wind"] == "full") & (agg["run_type"] == "per40")]
    if sub.empty:
        return (np.nan, np.nan)
    r_m = next(r for lbl, r, _ in PROBES if lbl == probe_label)
    plat_start = t_arr_at(r_m, f) + 8.0 / f
    plat_end   = float(sub["plat_end_relax_s_med"].iloc[0])
    return (plat_start, plat_end)


records = []
for f in THESIS_FREQS:
    N = N_LENGTH_LOOKUP[f]
    for probe_label, r_m, _ in PROBES:
        t_arr   = t_arr_at(r_m, f)
        t_paras = t_paras_at(r_m, f)
        win_s, win_e = proposed_window(r_m, f)
        delta = win_e - win_s

        eb_s, eb_e = EYEBALL_NOWIND[probe_label][f]
        em_s, em_e = empirical_fullwind_plateau(probe_label, f)

        records.append({
            "freq_hz":          f,
            "probe":            probe_label,
            "t_arr_s":          t_arr,
            "win_start_s":      win_s,
            "win_end_s":        win_e,
            "delta_s":          delta,
            "t_paras_s":        t_paras,
            "plateau_no_start": eb_s,
            "plateau_no_end":   eb_e,
            "plateau_full_start": em_s,
            "plateau_full_end":   em_e,
            "N_periods":        N,
        })

table = pd.DataFrame(records)
print(table.round(2).to_string(index=False))


# ── Save companion CSVs (one per wind, only the relevant plateau cols) ──
nowind_csv  = Path("analysis_scratch/window_choice_nowind.csv")
fullwind_csv = Path("analysis_scratch/window_choice_fullwind.csv")

cols_no = ["freq_hz", "probe", "t_arr_s", "win_start_s", "win_end_s",
           "delta_s", "t_paras_s", "plateau_no_start", "plateau_no_end",
           "N_periods"]
cols_fw = ["freq_hz", "probe", "t_arr_s", "win_start_s", "win_end_s",
           "delta_s", "t_paras_s", "plateau_full_start", "plateau_full_end",
           "N_periods"]
table[cols_no].to_csv(nowind_csv, index=False)
table[cols_fw].to_csv(fullwind_csv, index=False)
print(f"\n   CSV nowind   → {nowind_csv}")
print(f"   CSV fullwind → {fullwind_csv}")


# ── LaTeX table renderer ────────────────────────────────────────────────
def fmt_num(v, decimals=1):
    if pd.isna(v):
        return r"\textendash"
    return rf"$\num{{{v:.{decimals}f}}}$"

def fmt_range(a, b, decimals=1):
    if pd.isna(a) or pd.isna(b):
        return r"\textendash"
    return rf"\tabnumrange{{{a:.{decimals}f}}}{{{b:.{decimals}f}}}"


def render_table(wind_label: str, plat_start_col: str, plat_end_col: str,
                 plat_header_tex: str, plat_caption_tex: str,
                 thesis_name: str) -> str:
    """Render one LaTeX table (8 data rows, one per (f, probe))."""
    body_lines = []
    for _, r in table.iterrows():
        f       = r["freq_hz"]
        probe   = r["probe"]
        cells = [
            rf"$\num{{{f}}}$",                                   # f
            probe,                                               # probe
            fmt_num(r["t_arr_s"], 1),                            # t_arr
            fmt_range(r["win_start_s"], r["win_end_s"], 1),      # window
            fmt_num(r["delta_s"], 2),                            # Δ
            fmt_num(r["t_paras_s"], 1),                          # t_paras
            fmt_range(r[plat_start_col], r[plat_end_col], 1),    # plateau
            rf"$\num{{{int(r['N_periods'])}}}$",                 # N
        ]
        body_lines.append("    " + " & ".join(cells) + r" \\")

    # Caption is read from output/.table_captions.json (written by main_save_tables.py).
    # Run main_save_tables.py once before this script to populate the cache.
    captions_json = BASE / "output" / ".table_captions.json"
    caption_full  = _lookup_central_caption(thesis_name, kind="full",  json_path=captions_json)
    caption_short = _lookup_central_caption(thesis_name, kind="short", json_path=captions_json)
    # Wrap the caption in CAPTION-SYNC sentinels (Option D, 2026-05-09)
    # so analysis_scratch/sync_captions.py can rewrite the caption later
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
        "%   script            : analysis_scratch/window_choice_table.py",
        "%   plot_type         : window_choice_table",
        f"%   chapter           : {CHAPTER}",
        f"%   wind condition    : {wind_label}",
        f"%   generated_at      : {_dt.now().isoformat(timespec='seconds')}",
        f"%   caption_label     : tab:{thesis_name}",
        f"%   caption_short     : {caption_short}",
        "%",
        "% — Method ────────────────────────────────────────────────────",
        "%   Option B post-squeeze window:",
        "%     t_start(probe, f) = r_probe / c_g(f, h) + N_offset / f",
        "%     t_end(probe, f)   = t_start + N(f) / f",
        f"%   N_offset          : {int(N_OFFSET_T)} periods",
        f"%   N(f)              : {{1.3: 10, 1.4: 13, 1.5: 13, 1.6: 13}}",
        f"%   r_IN              : {PROBES[0][1]} m  ({PROBES[0][2]})",
        f"%   r_OUT             : {PROBES[1][1]} m  ({PROBES[1][2]})",
        f"%   tank depth h      : {TANK_DEPTH_M} m",
        "%   c_g dispersion    : full ω²=gk·tanh(kh) via wavescripts.constants.c_group",
        "%",
        "% — Plateau column ───────────────────────────────────────────────",
        f"%   {plat_caption_tex}",
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
        + f"  \\label{{tab:{thesis_name}}}\n"
        "  \\begin{tabular}{cccccccc}\n"
        "    \\toprule\n"
        "    $f$ [\\unit{\\hertz}] &\n"
        "      sonde &\n"
        "      $t_\\mathrm{arr}$ [\\unit{\\second}] &\n"
        "      vindu [\\unit{\\second}] &\n"
        "      $\\Delta$ [\\unit{\\second}] &\n"
        "      $t_\\mathrm{2f}$ [\\unit{\\second}] &\n"
        f"      {plat_header_tex} &\n"
        "      $N$ [perioder]\\\\\n"
        "    \\midrule\n"
        + "\n".join(body_lines) + "\n"
        "    \\bottomrule\n"
        "  \\end{tabular}\n"
        "\\end{table}\n"
    )

    return immutable + "\n" + table_body


# ── Render and save both tables ─────────────────────────────────────────
out_nowind   = BASE / "output" / "TABLES" / "ch04_window_choice_nowind.tex"
out_fullwind = BASE / "output" / "TABLES" / "ch04_window_choice_fullwind.tex"
out_nowind.parent.mkdir(parents=True, exist_ok=True)

txt_nowind = render_table(
    wind_label      = "nowind",
    plat_start_col  = "plateau_no_start",
    plat_end_col    = "plateau_no_end",
    plat_header_tex = "platå (eyeball) [\\unit{\\second}]",
    plat_caption_tex = ("plateau column = eyeball plateau bounds from "
                        "RampDetectionBrowser, snarvei_eyeballing.md Day 1 "
                        "(canon nowind 0.2 V runs)."),
    thesis_name     = "ch04_window_choice_nowind",
)
out_nowind.write_text(txt_nowind, encoding="utf-8")
print(f"\n   TEX nowind   → {out_nowind.relative_to(BASE)}")

txt_fullwind = render_table(
    wind_label      = "fullwind",
    plat_start_col  = "plateau_full_start",
    plat_end_col    = "plateau_full_end",
    plat_header_tex = "platå (empirisk) [\\unit{\\second}]",
    plat_caption_tex = ("plateau column = per40 fullwind sliding-A_FFT plateau, "
                        "±2 % relaxed criterion. Start = t_arr + 8/f (script's "
                        "reference-region low edge). End = plat_end_relax_s_med "
                        "from per40_plateau_end_aggregated.csv (canon March-2026 "
                        "lowrange, n=1 per cell)."),
    thesis_name     = "ch04_window_choice_fullwind",
)
out_fullwind.write_text(txt_fullwind, encoding="utf-8")
print(f"   TEX fullwind → {out_fullwind.relative_to(BASE)}")

print("\nDone.")
