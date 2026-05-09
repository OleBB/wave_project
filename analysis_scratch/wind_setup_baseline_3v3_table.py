"""
3v3 wind-setup baseline table — CH04 appendix.
==============================================

Renders the per-transition table that documents the bracketing-stillwater
3-nowind-vs-3-fullwind absolute-baseline comparison at OUT (12400/250),
across all March 2026 datasets that have both nowind and fullwind runs.

Source CSV (must exist; produced by analysis_scratch/wind_setup_baseline_3v3.py):
    analysis_scratch/wind_setup_baseline_3v3_results.csv

Output:
    output/TABLES/ch04_wind_setup_baseline_table.tex

Caption is blank — populated centrally in main_save_figures.py
(FIGURE_CAPTIONS / FIGURE_CAPTIONS_SHORT) and resolved at write time
by pu._lookup_central_caption.

Layout: one row per detected nowind↔fullwind transition, midrule between
datasets. Per-dataset summary block (mean magnitude across all that day's
transitions) appears at the bottom of each dataset block.

Method recap recorded in immutable block:
- Read `Stillwater Probe 12400/250` from each run's meta (absolute ULS
  baseline, mm; ULS reads distance DOWN to water — lower number = higher
  water level).
- Sort runs chronologically by file mtime, restrict to canonical
  WindCondition ∈ {no, full} and OUT baseline within (80, 120) mm.
- For each consecutive nowind↔fullwind transition, take last n_pre nowind
  runs before and first n_post fullwind runs after (each up to 3 runs).
- Δη_OUT (water rise under wind) = mean(nowind) − mean(fullwind).

Caveats noted in immutable block:
- Some transitions on dense-transition days have n_pre < 3 or n_post < 3
  (small-sample); flagged.
- The OFF > ON direction asymmetry (~0.1–0.4 mm in 3 of 4 datasets) is
  flagged; candidate explanations recorded but untested.
- Cross-dataset reproducibility under strict 3+3 sampling is excellent
  (~0.01 mm spread between dataset means at ~1.30 mm).
"""

from __future__ import annotations

import os
import sys
from datetime import datetime as _dt
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
os.chdir(BASE)

from wavescripts.plot_utils import _lookup_central_caption


SRC_CSV     = Path("analysis_scratch/wind_setup_baseline_3v3_results.csv")
OUT_TEX     = Path("output/TABLES/ch04_wind_setup_baseline_table.tex")
THESIS_NAME = "ch04_wind_setup_baseline_table"
CHAPTER     = "04"

# Direction labels for the table — arrow notation makes the BEFORE→AFTER
# direction explicit. "wind ON" = transition from no-wind to fullwind, i.e.
# the day was in 'av' (off) state and we turned wind 'på' (on) → 'av → på'.
# "wind OFF" = the inverse → 'på → av'.
DIR_LABEL = {"wind ON": r"av $\rightarrow$ på", "wind OFF": r"på $\rightarrow$ av"}


# ── Read input ───────────────────────────────────────────────────────────
if not SRC_CSV.exists():
    raise SystemExit(
        f"Missing {SRC_CSV}. Run analysis_scratch/wind_setup_baseline_3v3.py first."
    )
df = pd.read_csv(SRC_CSV)
df["date"] = df["dataset"].str.extract(r"PROCESSED-(\d{8})-")[0]
# Chronological within each dataset: transition_row_idx is the row index in
# the chronologically-sorted, cleaned df, so smaller = earlier in the day.
sort_cols = ["date", "dataset"]
if "transition_row_idx" in df.columns:
    sort_cols.append("transition_row_idx")
else:
    # Older CSV without transition_row_idx — fall back to the original
    # ON-then-OFF order (label sort) so the table still renders.
    sort_cols.append("label")
df = df.sort_values(sort_cols).reset_index(drop=True)

# Strict-3+3 mask helps identify the headline-worthy transitions.
df["strict_3v3"] = (df["n_pre"] == 3) & (df["n_post"] == 3)


def _short_dataset_tag(name: str) -> str:
    """Compress dataset folder name into a thesis-friendly tag.
    e.g. PROCESSED-20260326-...-under9Mooring-height100-lowrange → M9 / lowrange."""
    mooring = "Moring 9-30" if "under9Mooring30" in name else (
              "Moring 9-23"    if "under9Mooring"   in name else "—")
    rng     = "lowrange" if "lowrange" in name else "highrange"
    return f"{mooring} / {rng}"


# ── LaTeX rendering helpers ──────────────────────────────────────────────
def fmt_mm(v: float, decimals: int = 3) -> str:
    if pd.isna(v):
        return "—"
    return rf"\num{{{v:.{decimals}f}}}"


def fmt_signed_mm(v: float, decimals: int = 3) -> str:
    if pd.isna(v):
        return "—"
    sign = "+" if v >= 0 else "-"
    return rf"${sign}\num{{{abs(v):.{decimals}f}}}$"


def fmt_n(n_pre: int, n_post: int) -> str:
    return rf"{int(n_pre)}/{int(n_post)}"


# ── Build body rows: per-transition, midrule between datasets ────────────
body_lines: list[str] = []
last_date = None

for date, sub in df.groupby("date", sort=True):
    if last_date is not None:
        body_lines.append(r"    \midrule")

    if "transition_row_idx" in sub.columns:
        sub_sorted = sub.sort_values("transition_row_idx").reset_index(drop=True)
    else:
        sub_sorted = sub.sort_values("label").reset_index(drop=True)

    # Tag row at top of each block: date + setup tag.
    setup_tag = _short_dataset_tag(sub_sorted["dataset"].iloc[0])
    block_header = (
        rf"    \multicolumn{{6}}{{l}}{{\itshape "
        rf"{date}  ({setup_tag})}} \\"
    )
    body_lines.append(block_header)

    for _, r in sub_sorted.iterrows():
        retning = DIR_LABEL.get(r["label"], r["label"])
        # Convention: mean_uten_vind → nowind side; mean_full_vind → fullwind side.
        # In the CSV, before/after are state-based: wind ON has before=no, after=full.
        if r["before"] == "no":
            mu_uten = r["pre_mean"]
            mu_full = r["post_mean"]
            n_uten  = r["n_pre"]
            n_full  = r["n_post"]
        else:
            mu_uten = r["post_mean"]
            mu_full = r["pre_mean"]
            n_uten  = r["n_post"]
            n_full  = r["n_pre"]

        # Magnitude is always positive (water rise under wind, by convention).
        magnitude = r["magnitude_mm"]
        flag = r"\,\textsuperscript{*}" if not r["strict_3v3"] else ""

        # 6 cells across the 6-column layout: leading date column stays
        # blank (date is shown in the block-header multicolumn row above).
        cells = [
            "",
            retning,
            fmt_mm(mu_uten, 3),
            fmt_mm(mu_full, 3),
            fmt_mm(magnitude, 3) + flag,
            fmt_n(n_uten, n_full),
        ]
        body_lines.append("    " + " & ".join(cells) + r" \\")

    # Per-dataset summary row (mean of |Δη| across all transitions in the block).
    mean_mag = sub_sorted["magnitude_mm"].mean()
    summary_label = rf"\textit{{snitt (n=}}{len(sub_sorted)}\textit{{)}}"
    body_lines.append(r"    \cmidrule(l){2-6}")
    body_lines.append(
        "    & "
        + summary_label
        + " & — & — & "
        + fmt_mm(mean_mag, 3)
        + r" & — \\"
    )
    last_date = date


# ── Caption + immutable block ────────────────────────────────────────────
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

n_total = len(df)
n_strict_total = int(df["strict_3v3"].sum())
mean_strict_all = df.loc[df["strict_3v3"], "magnitude_mm"].mean()
datasets_used = sorted(df["dataset"].unique())

immutable = "\n".join([
    "%! TEX root = ../main.tex",
    "% ==============================================================",
    "% IMMUTABLE — generated automatically, do not edit this block",
    "%",
    "% — Provenance ───────────────────────────────────────────────────",
    "%   script            : analysis_scratch/wind_setup_baseline_3v3_table.py",
    "%   plot_type         : wind_setup_baseline_table",
    f"%   chapter           : {CHAPTER}",
    f"%   generated_at      : {_dt.now().isoformat(timespec='seconds')}",
    f"%   caption_label     : tab:{THESIS_NAME}",
    f"%   caption_short     : {caption_short}",
    "%",
    "% — Inputs ────────────────────────────────────────────────────",
    f"%   per-transition CSV: {SRC_CSV}",
    f"%   datasets contributing :",
    *[f"%     {p}" for p in datasets_used],
    "%",
    "% — Method ────────────────────────────────────────────────────",
    "%   Each row = one nowind↔fullwind transition detected by chronological",
    "%   sorting of canon WindCondition ∈ {no, full} runs. For each transition,",
    "%   take last n_pre ≤ 3 nowind runs and first n_post ≤ 3 fullwind runs.",
    "%   Means computed on `Stillwater Probe 12400/250` (absolute ULS baseline",
    "%   at OUT, in mm). |Δη_OUT| = water rise at OUT under wind =",
    "%   mean(nowind) − mean(fullwind). ULS reads distance DOWN to water:",
    "%   lower ULS number = higher water level.",
    "%",
    f"%   * superscript on |Δη| flags transitions where n_pre < 3 OR n_post < 3",
    f"%     (small-sample). {n_total - n_strict_total} of {n_total} transitions are flagged.",
    "%",
    "% — Headline ────────────────────────────────────────────────",
    f"%   Strict 3+3 mean across all datasets: {mean_strict_all:.3f} mm "
    f"(n={n_strict_total} transitions).",
    "%   Cross-dataset spread under strict 3+3 sampling is sub-millimetre",
    "%   (~0.01 mm between dataset means in current data).",
    "%",
    "% — Caveats ───────────────────────────────────────────────────",
    "%   (1) Direction asymmetry: |OFF| magnitude exceeds |ON| magnitude by",
    "%       0.1–0.4 mm in 3 of 4 datasets (20260327 is tied). Candidate",
    "%       explanations (untested):",
    "%         • wind-on stress build-up biases ON measurement low",
    "%         • residual relaxation biases OFF measurement high",
    "%       See analysis_scratch/wind_setup_baseline_3v3_investigation.md.",
    "%   (2) Small-sample artefact: 20260326 had high transition density,",
    "%       leaving 0 strict 3+3 transitions on that day. Its lower mean",
    "%       (0.88 mm vs ~1.30 mm elsewhere) is a sampling artefact, not",
    "%       a real day-to-day setup difference. Last transition of that",
    "%       day (1.330 mm) lands on par with other datasets.",
    "%   (3) Seiche-bias on short-gap ON transitions. The OUT probe shows",
    "%       a clean 8.2 s seiche under wind (amplitude ~1 mm right after",
    "%       wind-on, decaying to ~0.5 mm by ~5 min — see ch04_wind_rampup).",
    "%       The pipeline anchor is the first ~1 s of each run, capturing",
    "%       1/8 of one seiche period → each individual fullwind reading is",
    "%       phase-biased by ±A. Mean of 3 random-phase samples reduces this",
    "%       to ≈ A/√6 ≈ 0.4·A → up to 0.4 mm bias if seiche still at 1 mm.",
    "%       Worst-case transition: 20260326 av→på #2 (6 min gap, post-block",
    "%       reads spread 0.44 mm). Long-gap ON transitions (≥ 15 min) and",
    "%       all OFF transitions are below the 0.2 mm bias threshold.",
    "%   (4) Pipeline `Stillwater Probe 12400/250` for fullwind runs uses a",
    "%       per-run anchor that may BE wind-on (first 1 s of THAT run); the",
    "%       absolute baseline reported here is the raw ULS reading, NOT a",
    "%       deviation from a settled-tank reference. Setup magnitude is",
    "%       still recovered correctly because both nowind and fullwind",
    "%       sides report raw ULS values from the same probe.",
    "%",
    "% ── end immutable block ─────────────────────────────────────────",
])


# ── Final table ──────────────────────────────────────────────────────────
table_body = (
    "\\begin{table}[hbt]\n"
    "  \\centering\n"
    + caption_block
    + f"  \\label{{tab:{THESIS_NAME}}}\n"
    "  \\small\n"
    "  \\begin{tabular}{l l c c c c}\n"
    "    \\toprule\n"
    "    Dato & Retning &\n"
    "      $\\mu_\\text{uten vind}$ [\\unit{\\milli\\metre}] &\n"
    "      $\\mu_\\text{full vind}$ [\\unit{\\milli\\metre}] &\n"
    "      $|\\Delta\\eta_\\text{OUT}|$ [\\unit{\\milli\\metre}] &\n"
    "      $n_\\text{uten}/n_\\text{full}$ \\\\\n"
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
print(f"   Strict 3+3 mean across {n_strict_total} transitions: {mean_strict_all:.3f} mm")
print("Done.")
