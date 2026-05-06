"""
Mooring + panelretning at 1.30 Hz — wind-effect table
======================================================

Companion to `analysis_scratch/mooring_focus_at_1_3hz_ka.py`.
Same data, same scope (1.30 Hz only, panels ∈ {full, reverse}, moorings ∈
{below_90, above_50}). Where the figure shows the points, this table
gives the hard numbers.

Sibling structure to `wind_effect_table_by_amp.py`:
  outer sort = amplitude  (A1 / A2 / A3, one row block each)
  inner sort = (panel, mooring), so each block has up to 3 rows:
    normal · below_90
    normal · above_50
    revers · above_50
  (revers · below_90 doesn't exist — reverse panel was never run on the
   below_90 mooring.)

Columns mirror the wind_effect tables: K_t per wind, ΔK_t in pp, K_t-økn %,
D-red %. Per-cell n shown alongside K_t since several cells have n ≤ 4.

Outputs:
    output/TABLES/ch05_mooring_focus_at_1_3hz_table.tex   (thesis include)
    analysis_scratch/mooring_focus_at_1_3hz_table.csv     (companion CSV)

Caption text is read from FIGURE_CAPTIONS["ch05_mooring_focus_at_1_3hz_table"]
in main_save_figures.py via output/.figure_captions.json.
"""

import os
import sys
import warnings
from pathlib import Path
from datetime import datetime as _dt
import glob

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
os.chdir(BASE)

from wavescripts.improved_data_loader import load_analysis_data
from wavescripts.plot_utils import _lookup_central_caption, amp_to_label

# ── I/O ────────────────────────────────────────────────────────────────────
SCRATCH_CSV = Path(__file__).parent / "mooring_focus_at_1_3hz_table.csv"
THESIS_NAME = "ch05_mooring_focus_at_1_3hz_table"
OUT_TEX     = BASE / "output" / "TABLES" / f"{THESIS_NAME}.tex"
CHAPTER     = "05"

TARGET_FREQ  = 1.30
THESIS_AMPS  = [0.10, 0.20, 0.30]

# ── 1. Load + filter ───────────────────────────────────────────────────────
print("1. Loading all processed folders …")
all_dirs = sorted(glob.glob(str(BASE / "waveprocessed" / "PROCESSED-*")))
meta, _, _, _ = load_analysis_data(*all_dirs, load_processed=False)
print(f"   {len(meta)} rows total")

def mooring_group(m):
    if m in ("below_90_loose230", "below_90_loose300"):
        return "below_90"
    if m == "above_50":
        return "above_50"
    return "other"
meta["moor_grp"] = meta["Mooring"].apply(mooring_group)

sel = meta[
    (meta["WaveFrequencyInput [Hz]"] == TARGET_FREQ)
    & meta["PanelCondition"].isin(["full", "reverse"])
    & (meta["quality_flag"] == "ok")
    & meta["OUT/IN (FFT)"].notna()
    & meta["WindCondition"].isin(["no", "full"])
    & (meta["OUT/IN (FFT)"].between(0.1, 2.0))
    & meta["moor_grp"].isin(["below_90", "above_50"])
].copy()
sel["amp_v"] = sel["WaveAmplitudeInput [Volt]"].apply(lambda v: round(float(v), 2))
print(f"   {len(sel)} rows at 1.30 Hz, panels ∈ {{full, reverse}}, "
      f"moor ∈ {{below_90, above_50}}")

# ── 2. Aggregate per (amp, panel, mooring, wind) — pool hardware ───────────
agg = (sel.groupby(["amp_v", "PanelCondition", "moor_grp", "WindCondition"])
          ["OUT/IN (FFT)"]
          .agg(["mean", "std", "count"])
          .reset_index()
          .rename(columns={"mean": "Kt", "std": "Kt_std", "count": "n",
                            "PanelCondition": "panel",
                            "WindCondition": "wind",
                            "moor_grp": "mooring"}))
print(f"\n   {len(agg)} (amp, panel, mooring, wind) cells")

# Pivot to one row per (amp, panel, mooring) with wind columns side-by-side.
pivot_K  = agg.pivot_table(index=["amp_v", "panel", "mooring"],
                            columns="wind", values="Kt", aggfunc="mean").reset_index()
pivot_n  = agg.pivot_table(index=["amp_v", "panel", "mooring"],
                            columns="wind", values="n",  aggfunc="sum").reset_index()
pivot_sd = agg.pivot_table(index=["amp_v", "panel", "mooring"],
                            columns="wind", values="Kt_std", aggfunc="mean").reset_index()

table = pivot_K.rename(columns={"no": "Kt_nw", "full": "Kt_fw"})
table["std_nw"] = pivot_sd["no"]
table["std_fw"] = pivot_sd["full"]
table["n_nw"]   = pivot_n["no"].astype("Int64")
table["n_fw"]   = pivot_n["full"].astype("Int64")
table["Delta_Kt"]  = table["Kt_fw"] - table["Kt_nw"]    # absolute change in K_t
table["ratio_Kt"]  = table["Kt_fw"] / table["Kt_nw"]    # K_t,vind / K_t,uten
_D_nw = 1.0 - table["Kt_nw"]
_D_fw = 1.0 - table["Kt_fw"]
table["ratio_D"]   = _D_fw / _D_nw                       # D_vind / D_uten,  D = 1 - K_t

# Drop cells where either wind condition is missing (no Δ to compute).
n_before = len(table)
table = table.dropna(subset=["Kt_nw", "Kt_fw"]).reset_index(drop=True)
n_dropped = n_before - len(table)
if n_dropped:
    print(f"   {n_dropped} cell(s) dropped — at least one wind missing")

# Sort: amplitude OUTER, then panel (normal first), then mooring
# (below_90 first → "canon" sits at top of each block).
PANEL_ORDER   = {"full": 0, "reverse": 1}
MOORING_ORDER = {"below_90": 0, "above_50": 1}
table["_p_ord"] = table["panel"].map(PANEL_ORDER)
table["_m_ord"] = table["mooring"].map(MOORING_ORDER)
table = table.sort_values(["amp_v", "_p_ord", "_m_ord"]).reset_index(drop=True)
table = table.drop(columns=["_p_ord", "_m_ord"])

print(f"   {len(table)} cells in final table\n")
print(table.round(3).to_string(index=False))

# ── 3. Save companion CSV ──────────────────────────────────────────────────
SCRATCH_CSV.parent.mkdir(parents=True, exist_ok=True)
table.to_csv(SCRATCH_CSV, index=False)
print(f"\n   CSV → {SCRATCH_CSV.relative_to(BASE)}")

# ── 4. Render LaTeX table ──────────────────────────────────────────────────
PANEL_LBL   = {"full": "normal", "reverse": "revers"}
MOORING_LBL = {"below_90": "below\\_90", "above_50": "above\\_50"}

def _fmt_signed(x: float, decimals: int = 1) -> str:
    if pd.isna(x):
        return "—"
    return f"{x:+.{decimals}f}"

def _fmt_kt_n(k: float, n) -> str:
    """K_t value with sample size: '0.661 (13)'."""
    if pd.isna(k):
        return "—"
    n_str = "—" if pd.isna(n) else f"{int(n)}"
    return f"{k:.3f}\\,({n_str})"

def _fmt_ratio(x: float, decimals: int = 3) -> str:
    """Unsigned ratio (e.g. 1.181, 0.653). NaN → em-dash."""
    if pd.isna(x):
        return "—"
    return f"{x:.{decimals}f}"

# Body rows. \midrule between AMPLITUDE blocks.
body_rows = []
last_amp = None
for _, r in table.iterrows():
    a = float(r["amp_v"])
    if last_amp is not None and a != last_amp:
        body_rows.append(r"\midrule")
    body_rows.append(
        f"  {amp_to_label(a)} & {PANEL_LBL[r['panel']]} & "
        f"{MOORING_LBL[r['mooring']]} & "
        f"{_fmt_kt_n(r['Kt_nw'], r['n_nw'])} & "
        f"{_fmt_kt_n(r['Kt_fw'], r['n_fw'])} & "
        f"{_fmt_signed(r['Delta_Kt'], 3)} & "        # ΔK_t as decimal fraction
        f"{_fmt_ratio(r['ratio_Kt'], 3)} & "         # K_t,vind / K_t,uten
        f"{_fmt_ratio(r['ratio_D'],  3)} \\\\"        # D_vind / D_uten
    )
    last_amp = a

# Caption from central FIGURE_CAPTIONS dict.
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
n_total_runs = int(table["n_nw"].fillna(0).sum() + table["n_fw"].fillna(0).sum())
immutable = "\n".join([
    "%! TEX root = ../main.tex",
    "% ==============================================================",
    "% IMMUTABLE — generated automatically, do not edit this block",
    "%",
    "% — Provenance ───────────────────────────────────────────────────",
    "%   script            : analysis_scratch/mooring_focus_at_1_3hz_table.py",
    "%   plot_type         : mooring_focus_at_1_3hz_table",
    f"%   chapter           : {CHAPTER}",
    f"%   generated_at      : {_dt.now().isoformat(timespec='seconds')}",
    f"%   caption_label     : tab:{THESIS_NAME}",
    f"%   caption_short     : {caption_short}",
    "%",
    "% — Filters ────────────────────────────────────────────────────",
    "%   panel             : full, reverse",
    "%   wind              : no, full",
    "%   mooring           : below_90 (canon: loose230 + loose300),",
    "%                       above_50",
    f"%   amplitude [V]     : {', '.join(f'{a:.1f}' for a in THESIS_AMPS)}",
    f"%   frequency [Hz]    : {TARGET_FREQ}",
    "%   quality_flag      : ok",
    "%",
    "% — Data provenance ────────────────────────────────────────────",
    f"%   n_cells           : {len(table)}",
    f"%   n_runs (nw + fw)  : {n_total_runs}",
    "%   datasets        :",
    *[f"%     {Path(d).name}" for d in all_dirs],
    "%",
    "% — Method ────────────────────────────────────────────────────",
    "%   grouper           : groupby(amp, panel, mooring, wind) → mean K_t",
    "%   sort_order        : amplitude outer, then panel (normal first),",
    "%                       then mooring (below_90 first within each panel).",
    "%   mooring pooling   : below_90 lumps loose230 + loose300; both panels",
    "%                       pool hardware (cond4 + earlier) — see",
    "%                       analysis_scratch/probe_config_bias_likeforlike.pdf",
    "%                       for the bias check.",
    "%   metric_definitions:",
    "%     K_t                 : OUT/IN(FFT), narrow 0.1 Hz window at f_paddle",
    "%     Kt_nw / Kt_fw       : mean K_t at no- / full-wind",
    "%     n_nw  / n_fw        : run count per (amp, panel, mooring, wind)",
    "%     Delta_Kt            : Kt_fw - Kt_nw  (signed decimal fraction)",
    "%     ratio_Kt            : Kt_fw / Kt_nw  (>1 = wind passes more wave)",
    "%     ratio_D             : D_fw / D_nw    (D = 1 - K_t)",
    "%                           <1 = wind erodes panel damping",
    "%",
    "% ── end immutable block ─────────────────────────────────────────",
])

table_body = (
    "\\begin{table}[htbp]\n"
    "  \\centering\n"
    "  \\small\n"
    "  \\begin{tabular}{ccl rr r r r}\n"
    "    \\toprule\n"
    "      Amp & Panel & Mooring & $K_{t,\\text{uten}}\\,(n)$ & "
    "$K_{t,\\text{vind}}\\,(n)$ & $\\Delta K_t$ & "
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
