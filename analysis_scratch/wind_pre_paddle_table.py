"""
Pre-paddle wind summary table — CH04 §4q companion.
====================================================

Renders the four-probe summary that accompanies ch04_wind_pre_paddle_psd:

    probe          | long-run sigma | 3 s mean sigma | delta (%) | 3 s 1-sigma scatter

Source CSV (must exist; produced by analysis_scratch/wind_2s_vs_360s.py):
    analysis_scratch/wind_2s_vs_360s_stats_3s.csv

Output:
    output/TABLES/ch04_wind_pre_paddle_table.tex

Caption is blank — populated centrally in main_save_figures.py
(FIGURE_CAPTIONS / FIGURE_CAPTIONS_SHORT) and resolved at write time
by pu._lookup_central_caption.

Immutable block records:
  - which CSV was read
  - long-run reference set: 5 fullwind+nowave runs, durations 31, 33, 63,
    360, 381 s (only 2 of 5 are >= 360 s; phrase carefully in caption)
  - OUT-probe (12400/250) noise-floor caveat: long-run sigma 0.36 mm
    sits in the probe's measured stillwater-noise range (0.14-0.36 mm
    per CLAUDE.md §16); the -8.6 % delta vs the 3 s ensemble is
    consistent with sampling noise, NOT a window-length bias.
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime as _dt

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import os
os.chdir(BASE)

from wavescripts.plot_utils import _lookup_central_caption


SRC_CSV   = Path("analysis_scratch/wind_2s_vs_360s_stats_3s.csv")
LONG_CSV  = Path("analysis_scratch/wind_2s_vs_360s_per_long_run_3s.csv")
OUT_TEX   = Path("output/TABLES/ch04_wind_pre_paddle_table.tex")
THESIS_NAME = "ch04_wind_pre_paddle_table"
CHAPTER     = "04"

# Probes in row order (label as it appears in the CSV → display label).
PROBE_ORDER = [
    ("8804/250 (upstream)",   "8804/250 (upstream)"),
    ("9373/170 (IN, wall)",   "9373/170 (IN, wall)"),
    ("9373/340 (IN, far)",    "9373/340 (IN, far)"),
    ("12400/250 (OUT)",       "12400/250 (OUT)"),
]

# ── Read inputs ──────────────────────────────────────────────────────────
if not SRC_CSV.exists():
    raise SystemExit(
        f"Missing {SRC_CSV}. Run analysis_scratch/wind_2s_vs_360s.py first."
    )
stats = pd.read_csv(SRC_CSV)

if not LONG_CSV.exists():
    raise SystemExit(
        f"Missing {LONG_CSV}. Run analysis_scratch/wind_2s_vs_360s.py first."
    )
long_runs = pd.read_csv(LONG_CSV)
n_long_runs   = long_runs["long_run"].nunique()
long_durations = sorted(long_runs.drop_duplicates("long_run")["duration_s"].tolist())

# ── Pivot stats CSV into one row per probe with both segments ────────────
def _row(probe_label: str) -> dict:
    sub = stats[stats["probe"] == probe_label]
    long_row = sub[sub["segment"].str.startswith("long")].iloc[0]
    short_row = sub[sub["segment"].str.startswith("3 s")].iloc[0]
    sigma_long = float(long_row["sigma_mm"])
    sigma_3s   = float(short_row["sigma_mm"])
    sigma_3s_scatter = float(short_row["sigma_std"])
    delta_pct = 100.0 * (sigma_3s - sigma_long) / sigma_long
    return {
        "probe":               probe_label,
        "sigma_long_mm":       sigma_long,
        "sigma_3s_mm":         sigma_3s,
        "delta_pct":           delta_pct,
        "sigma_3s_scatter_mm": sigma_3s_scatter,
        "n_long":              int(long_row["n"]),
        "n_3s":                int(short_row["n"]),
    }

records = [_row(label) for label, _ in PROBE_ORDER]
n_long_seg = records[0]["n_long"]
n_3s       = records[0]["n_3s"]


# ── LaTeX rendering ──────────────────────────────────────────────────────
def fmt_mm(v: float, decimals: int = 2) -> str:
    return rf"$\num{{{v:.{decimals}f}}}$"

def fmt_pct(v: float) -> str:
    sign = "+" if v >= 0 else "-"
    return rf"${sign}\num{{{abs(v):.1f}}}$"

body_lines = []
for r in records:
    cells = [
        r["probe"],
        fmt_mm(r["sigma_long_mm"], 2),
        fmt_mm(r["sigma_3s_mm"], 2),
        fmt_pct(r["delta_pct"]),
        fmt_mm(r["sigma_3s_scatter_mm"], 2),
    ]
    body_lines.append("    " + " & ".join(cells) + r" \\")


# Caption is read from output/.table_captions.json (written by main_save_tables.py).
# Run main_save_tables.py once before this script to populate the cache.
_CAPTIONS_JSON = BASE / "output" / ".table_captions.json"
caption_full  = _lookup_central_caption(THESIS_NAME, kind="full",  json_path=_CAPTIONS_JSON)
caption_short = _lookup_central_caption(THESIS_NAME, kind="short", json_path=_CAPTIONS_JSON)
if caption_full and caption_short:
    caption_block = (
        f"  \\caption[{caption_short}]{{\n"
        f"    {caption_full}\n"
        f"  }}\n"
    )
elif caption_full:
    caption_block = f"  \\caption{{\n    {caption_full}\n  }}\n"
else:
    caption_block = "  \\caption{\n    % TODO: write caption\n  }\n"


durations_str = ", ".join(f"{int(d)}" for d in long_durations)

immutable = "\n".join([
    "%! TEX root = ../main.tex",
    "% ==============================================================",
    "% IMMUTABLE — generated automatically, do not edit this block",
    "%",
    "% — Provenance ───────────────────────────────────────────────────",
    "%   script            : analysis_scratch/wind_pre_paddle_table.py",
    "%   plot_type         : wind_pre_paddle_table",
    f"%   chapter           : {CHAPTER}",
    f"%   generated_at      : {_dt.now().isoformat(timespec='seconds')}",
    f"%   caption_label     : tab:{THESIS_NAME}",
    f"%   caption_short     : {caption_short}",
    "%",
    "% — Inputs ────────────────────────────────────────────────────",
    f"%   stats CSV         : {SRC_CSV}",
    f"%   per-long-run CSV  : {LONG_CSV}",
    f"%   datasets          : PROCESSED-20260326-*-lowrange,",
    f"%                       PROCESSED-20260327-*-lowrange (canon)",
    f"%   long-run set      : {n_long_runs} fullwind+nowave runs",
    f"%   long-run durations: {durations_str} s  (only 2 of {n_long_runs} are ≥ 360 s)",
    f"%   3 s ensemble      : {n_3s} fullwind+wave runs, one 3 s pre-paddle",
    f"%                       snippet each (paddle-free: √(gh)=2.39 m/s,",
    f"%                       closest probe at 8804 mm → safe to 3.68 s)",
    "%",
    "% — Method ────────────────────────────────────────────────────",
    "%   sigma_eta = std(eta), Hs = 4 sigma_eta, mean = mean(eta).",
    "%   long-run column     : ensemble mean of per-run sigma over long runs",
    "%   3 s mean column     : ensemble mean of per-snippet sigma over 70 runs",
    "%   delta column        : (sigma_3s − sigma_long) / sigma_long, percent",
    "%   3 s 1σ scatter      : std (ddof=1) of per-snippet sigma across 70 runs",
    "%",
    "% — Caveats ───────────────────────────────────────────────────",
    "%   OUT probe (12400/250):",
    "%     long-run sigma = 0.36 mm; 3 s sigma = 0.33 mm; delta = -8.6 %.",
    "%     The probe's measured stillwater noise floor is ~0.14 mm",
    "%     (gold standard, CLAUDE.md §16) and ranges 0.14-0.36 mm",
    "%     across settled stillwater runs. The -8.6 % delta sits inside",
    "%     this noise-floor envelope, so it is consistent with sampling",
    "%     noise, not a window-length bias. Caption should reflect this",
    "%     when interpreting the OUT row.",
    "%   Long-run reference set: 5 runs total, durations vary 31-381 s",
    "%     (median 63 s; only 2 of 5 are ≥ 360 s). Phrase as e.g. ",
    '%     "five fullwind+nowave runs, durations 31-381 s" — not "5 × 360 s".',
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
    "    sonde &\n"
    "      $\\sigma_\\eta$ (lang) [\\unit{\\milli\\metre}] &\n"
    "      $\\sigma_\\eta$ (\\qty{3}{\\second}) [\\unit{\\milli\\metre}] &\n"
    "      $\\Delta$ [\\%] &\n"
    "      $\\sigma_\\eta$ (\\qty{3}{\\second}, $1\\sigma$) [\\unit{\\milli\\metre}] \\\\\n"
    "    \\midrule\n"
    + "\n".join(body_lines) + "\n"
    "    \\bottomrule\n"
    "  \\end{tabular}\n"
    "\\end{table}\n"
)


# ── Write ────────────────────────────────────────────────────────────────
OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
OUT_TEX.write_text(immutable + "\n" + table_body, encoding="utf-8")
print(f"   TEX  → {OUT_TEX}")

# Also dump a CSV for quick inspection.
OUT_CSV = Path("analysis_scratch/wind_pre_paddle_table.csv")
pd.DataFrame(records).to_csv(OUT_CSV, index=False)
print(f"   CSV  → {OUT_CSV}")

print("Done.")
