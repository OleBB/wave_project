"""
loose230 vs loose300 mooring — within-canon comparison table
=============================================================

Sibling to `wind_effect_table_by_amp.py`. Same canon scope (the two
lowrange folders), but instead of pooling over Mooring (the 2026-05-05 fix)
this table KEEPS Mooring as a row dimension — one row per
(amp, freq, wind) cell, with K_t for each mooring and the difference.

Motivated by the A1 / 1.4 Hz outlier visible in
`output/FIGURES/ch05_damping_ka_fit_A1.pdf`: two near-identical-ka points
differ by 0.17 in K_t. Inspection showed the difference is the mooring
(loose230 vs loose300, the only thing that changed between Mar 26 and
Mar 27 setups). This table surfaces the within-canon mooring effect
that the headline pooled tables hide.

Methodology caveats baked into the data:
  - Mooring is aliased with date: loose230 = Mar 26 only, loose300 = Mar 27.
    Daily setup drift cannot be separated from mooring effect.
  - loose230 has almost no nowind runs at A2/A3; nowind comparison only
    exists at A1 / 1.3 Hz. Most fullwind cells have n=1 for loose230.
  - Cells with only one mooring sampled show "—" for the missing column.

Outputs:
    output/TABLES/ch05_loose230_vs_loose300_table.tex   (thesis include)
    analysis_scratch/loose230_vs_loose300_table.csv     (companion CSV)

Caption from FIGURE_CAPTIONS["ch05_loose230_vs_loose300_table"] in
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
from wavescripts.plot_utils import _lookup_central_caption, amp_to_label

SCRATCH_CSV = Path(__file__).parent / "loose230_vs_loose300_table.csv"
THESIS_NAME = "ch05_loose230_vs_loose300_table"
OUT_TEX     = BASE / "output" / "TABLES" / f"{THESIS_NAME}.tex"
CHAPTER     = "05"

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

m = meta.copy()
m = m[m["PanelCondition"] == "full"]
m = m[m["WaveFrequencyInput [Hz]"].isin(THESIS_FREQS)]
m = m[m["WaveAmplitudeInput [Volt]"].apply(
        lambda v: any(abs(float(v) - a) < 1e-3 for a in THESIS_AMPS))]
if "quality_flag" in m.columns:
    m = m[m["quality_flag"].isna() | (m["quality_flag"] == "ok")]
m = m.dropna(subset=["OUT/IN (FFT)", "WindCondition", "Mooring"])
m = m[m["WindCondition"].isin(["no", "full"])]
m = m[m["Mooring"].isin(["below_90_loose230", "below_90_loose300"])]
m["amp_v"] = m["WaveAmplitudeInput [Volt]"].apply(lambda v: round(float(v), 2))
print(f"   {len(m)} rows in scope")

# ── 2. Aggregate per (amp, freq, wind, mooring) → K_t mean / std / n ───────
agg = (m.groupby(["amp_v", "WaveFrequencyInput [Hz]", "WindCondition", "Mooring"])
        ["OUT/IN (FFT)"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "Kt", "std": "Kt_std", "count": "n",
                          "WaveFrequencyInput [Hz]": "freq",
                          "WindCondition": "wind", "Mooring": "mooring"}))

# Pivot: one row per (amp, freq, wind), columns per mooring.
piv_K  = agg.pivot_table(index=["amp_v", "freq", "wind"],
                          columns="mooring", values="Kt", aggfunc="mean").reset_index()
piv_n  = agg.pivot_table(index=["amp_v", "freq", "wind"],
                          columns="mooring", values="n",  aggfunc="sum").reset_index()
piv_sd = agg.pivot_table(index=["amp_v", "freq", "wind"],
                          columns="mooring", values="Kt_std", aggfunc="mean").reset_index()

table = piv_K.rename(columns={"below_90_loose230": "Kt_230",
                               "below_90_loose300": "Kt_300"})
table["std_230"] = piv_sd["below_90_loose230"]
table["std_300"] = piv_sd["below_90_loose300"]
table["n_230"]   = piv_n["below_90_loose230"].astype("Int64")
table["n_300"]   = piv_n["below_90_loose300"].astype("Int64")
table["Delta"]   = table["Kt_230"] - table["Kt_300"]
table["ratio"]   = table["Kt_230"] / table["Kt_300"]

# Sort: amp outer, freq inner, wind (no first then full).
WIND_ORDER = {"no": 0, "full": 1}
table["_w"] = table["wind"].map(WIND_ORDER)
table = table.sort_values(["amp_v", "freq", "_w"]).reset_index(drop=True)
table = table.drop(columns=["_w"])

print(f"   {len(table)} cells in final table\n")
print(table.round(3).to_string(index=False))

SCRATCH_CSV.parent.mkdir(parents=True, exist_ok=True)
table.to_csv(SCRATCH_CSV, index=False)
print(f"\n   CSV → {SCRATCH_CSV.relative_to(BASE)}")

# ── 3. LaTeX table ─────────────────────────────────────────────────────────
def _fmt_kt_n(k, n):
    if pd.isna(k):
        return "—"
    n_str = "—" if pd.isna(n) else f"{int(n)}"
    return f"{k:.3f}\\,({n_str})"

def _fmt_signed(x, decimals=3):
    if pd.isna(x):
        return "—"
    return f"{x:+.{decimals}f}"

def _fmt_ratio(x, decimals=3):
    if pd.isna(x):
        return "—"
    return f"{x:.{decimals}f}"

WIND_LBL = {"no": "uten", "full": "med"}

body_rows = []
last_amp = None
for _, r in table.iterrows():
    a = float(r["amp_v"])
    if last_amp is not None and a != last_amp:
        body_rows.append(r"\midrule")
    body_rows.append(
        f"  {amp_to_label(a)} & {float(r['freq']):.1f} & {WIND_LBL[r['wind']]} & "
        f"{_fmt_kt_n(r['Kt_230'], r['n_230'])} & "
        f"{_fmt_kt_n(r['Kt_300'], r['n_300'])} & "
        f"{_fmt_signed(r['Delta'], 3)} & "
        f"{_fmt_ratio(r['ratio'], 3)} \\\\"
    )
    last_amp = a

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

n_total = int(table["n_230"].fillna(0).sum() + table["n_300"].fillna(0).sum())
both_present = int(table[["Kt_230", "Kt_300"]].dropna().shape[0])

immutable = "\n".join([
    "%! TEX root = ../main.tex",
    "% ==============================================================",
    "% IMMUTABLE — generated automatically, do not edit this block",
    "%",
    "% — Provenance ───────────────────────────────────────────────────",
    "%   script            : analysis_scratch/loose230_vs_loose300_table.py",
    "%   plot_type         : loose230_vs_loose300_table",
    f"%   chapter           : {CHAPTER}",
    f"%   generated_at      : {_dt.now().isoformat(timespec='seconds')}",
    f"%   caption_label     : tab:{THESIS_NAME}",
    f"%   caption_short     : {caption_short}",
    "%",
    "% — Filters ────────────────────────────────────────────────────",
    "%   panel             : full",
    "%   wind              : no, full",
    "%   mooring           : below_90_loose230 (Mar 26 only)",
    "%                       below_90_loose300 (Mar 27 only)",
    f"%   amplitude [V]     : {', '.join(f'{a:.1f}' for a in THESIS_AMPS)}",
    f"%   frequency [Hz]    : {', '.join(f'{f:.1f}' for f in THESIS_FREQS)}",
    "%   quality_flag      : ok",
    "%",
    "% — Data provenance ────────────────────────────────────────────",
    f"%   n_cells           : {len(table)}",
    f"%   n_cells_both_moor : {both_present}",
    f"%   n_runs (230 + 300): {n_total}",
    "%   datasets        :",
    *[f"%     {p.name}" for p in RESULTS_DIRS],
    "%",
    "% — Method ────────────────────────────────────────────────────",
    "%   grouper           : groupby(amp, freq, wind, mooring) → mean K_t",
    "%   sort_order        : amplitude outer, frequency middle, wind inner.",
    "%   metric_definitions:",
    "%     K_t                : OUT/IN(FFT), narrow 0.1 Hz window at f_paddle",
    "%     Kt_230 / Kt_300    : mean K_t per mooring",
    "%     n_230  / n_300     : run count per mooring",
    "%     Delta              : Kt_230 - Kt_300  (signed decimal fraction)",
    "%     ratio              : Kt_230 / Kt_300  (>1 = loose230 transmits more)",
    "%",
    "% — Caveats ───────────────────────────────────────────────────",
    "%   * Mooring aliased with date (loose230=Mar 26, loose300=Mar 27).",
    "%   * loose230 nowind data exists only at A1 1.3 Hz.",
    "%   * Per-cell n is small (often n=1 for loose230 fullwind).",
    "%",
    "% ── end immutable block ─────────────────────────────────────────",
])

table_body = (
    "\\begin{table}[htbp]\n"
    "  \\centering\n"
    "  \\small\n"
    "  \\begin{tabular}{cccc rr r r}\n"
    "    \\toprule\n"
    "      Amp & $f$ [Hz] & vind & "
    "$K_t$ loose230 $(n)$ & $K_t$ loose300 $(n)$ & "
    "$\\Delta = 230-300$ & $K_{t,230}/K_{t,300}$ \\\\\n"
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

# Headlines
both = table.dropna(subset=["Kt_230", "Kt_300"])
print(f"\n   Headlines (across {len(both)} cells with both moorings):")
print(f"     median Δ = {both['Delta'].median():+.4f}")
print(f"     mean Δ   = {both['Delta'].mean():+.4f}")
print(f"     range    = [{both['Delta'].min():+.4f}, {both['Delta'].max():+.4f}]")

print("\nDone.")
