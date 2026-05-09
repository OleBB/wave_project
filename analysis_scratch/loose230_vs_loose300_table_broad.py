"""
loose230 vs loose300 mooring — BROAD-SCOPE comparison table
============================================================

Sibling of `loose230_vs_loose300_table.py` (canon scope). Same analysis,
loaded across **all** PROCESSED-* folders so loose230 picks up runs from
Mar 16-26 (multiple probe configs: h272/high, h100/high, h136/high,
h100/low) instead of only the Mar 26 lowrange canon folder. loose300 is
unchanged — only the Mar 27 lowrange folder has it.

Why both versions:
  - canon: matches the headline thesis tables exactly (h100/low only,
    n=4 typical for loose230 fullwind). Strict but small-sample.
  - broad (this file): more statistical power for the mooring question
    (loose230 fullwind n typically 5-9 here), at the cost of mixing
    probe configs. Probe-config bias was previously checked as
    consistent with measurement noise (see
    analysis_scratch/probe_config_bias_likeforlike_all.pdf), so the
    pool is defensible.

Side-by-side reading: load both .tex tables. Cells where canon and broad
agree closely → mooring effect is robust. Cells where they diverge →
either probe-config does matter at that cell, or small-sample noise
flipped the canon-only direction.

Methodology caveats baked into the data:
  - Mooring is aliased with date: loose230 spans Mar 16-26,
    loose300 = Mar 27 only. Daily setup drift cannot be cleanly separated.
  - loose230 nowind comparison still mostly limited to A1.
  - Cells with only one mooring sampled show "—" for the missing column.

Outputs:
    output/TABLES/ch05_loose230_vs_loose300_table_broad.tex   (thesis include)
    analysis_scratch/loose230_vs_loose300_table_broad.csv     (companion CSV)

Caption from FIGURE_CAPTIONS["ch05_loose230_vs_loose300_table_broad"] in
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

import glob

SCRATCH_CSV = Path(__file__).parent / "loose230_vs_loose300_table_broad.csv"
THESIS_NAME = "ch05_loose230_vs_loose300_table_broad"
OUT_TEX     = BASE / "output" / "TABLES" / f"{THESIS_NAME}.tex"
CHAPTER     = "05"

# Broad scope: every PROCESSED-* folder. Filter by Mooring tag in the loop —
# any folder that contributes to below_90_loose230 / loose300 is in. Picks up
# Mar 16-26 (loose230, 4 probe configs) + Mar 27 (loose300, h100/low only).
RESULTS_DIRS = sorted(Path(p) for p in
                       glob.glob(str(BASE / "waveprocessed" / "PROCESSED-*")))

THESIS_FREQS = [1.3, 1.4, 1.5, 1.6]
THESIS_AMPS  = [0.10, 0.20, 0.30]

# ── 1. Load + filter ───────────────────────────────────────────────────────
print(f"1. Loading {len(RESULTS_DIRS)} processed folders …")
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

# Wrap the caption in CAPTION-SYNC sentinels (Option D, 2026-05-09) so
# analysis_scratch/sync_captions.py can rewrite the caption later
# without re-running this script.
from wavescripts.plot_utils import wrap_caption_with_sentinels
caption_block = wrap_caption_with_sentinels(caption_full, caption_short)

n_total = int(table["n_230"].fillna(0).sum() + table["n_300"].fillna(0).sum())
both_present = int(table[["Kt_230", "Kt_300"]].dropna().shape[0])

immutable = "\n".join([
    "%! TEX root = ../main.tex",
    "% ==============================================================",
    "% IMMUTABLE — generated automatically, do not edit this block",
    "%",
    "% — Provenance ───────────────────────────────────────────────────",
    "%   script            : analysis_scratch/loose230_vs_loose300_table_broad.py",
    "%   plot_type         : loose230_vs_loose300_table_broad",
    f"%   chapter           : {CHAPTER}",
    f"%   generated_at      : {_dt.now().isoformat(timespec='seconds')}",
    f"%   caption_label     : tab:{THESIS_NAME}",
    f"%   caption_short     : {caption_short}",
    "%",
    "% — Filters ────────────────────────────────────────────────────",
    "%   panel             : full",
    "%   wind              : no, full",
    "%   mooring           : below_90_loose230 (Mar 16-26, multiple probe cfgs)",
    "%                       below_90_loose300 (Mar 27 only, h100/low)",
    f"%   amplitude [V]     : {', '.join(f'{a:.1f}' for a in THESIS_AMPS)}",
    f"%   frequency [Hz]    : {', '.join(f'{f:.1f}' for f in THESIS_FREQS)}",
    "%   quality_flag      : ok",
    "%   probe_config      : pooled across all (h272/high, h100/high,",
    "%                       h136/high, h100/low) — see canon-scope sibling",
    "%                       for h100/low only.",
    "%",
    "% — Data provenance ────────────────────────────────────────────",
    f"%   n_cells           : {len(table)}",
    f"%   n_cells_both_moor : {both_present}",
    f"%   n_runs (230 + 300): {n_total}",
    f"%   n_folders_loaded  : {len(RESULTS_DIRS)}  (filtered by Mooring tag)",
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
    "%   * Mooring aliased with date (loose230=Mar 16-26, loose300=Mar 27).",
    "%   * Probe configurations vary within loose230 (canon = h100/low only;",
    "%     broad pulls in h272/high, h100/high, h136/high too).",
    "%   * Compare cell-by-cell to canon sibling table to spot config-driven drift.",
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
