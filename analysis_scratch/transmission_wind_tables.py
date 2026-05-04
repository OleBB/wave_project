"""
Wind-effect summary tables — IN, OUT, T per (frequency, amplitude).
====================================================================

Reads the per-run rows that drove the R_IN / R_OUT plot, applies symmetric
NaN filtering (drop a run if either A_IN or A_OUT is missing), and emits:

  1. analysis_scratch/transmission_wind_ratios_summary.csv
     One row per (f, A): R_IN, R_OUT, R_T mean ± std + n_runs per wind state.

  2. analysis_scratch/transmission_wind_amplitudes_per_condition.csv
     One row per (f, A, wind): A_IN, A_OUT, T mean ± std + n_runs.

  3. output/TABLES/ch05_transmission_wind_ratios.tex
     LaTeX render of (1) — main thesis table.

  4. output/TABLES/ch05_transmission_wind_amplitudes.tex
     LaTeX render of (2) — supporting / appendix table.

  5. console / markdown snippet for the A1/A2/A3 tiers (terminal copy/paste).

Caption text for the two LaTeX tables is read from FIGURE_CAPTIONS in
main_save_figures.py via output/.figure_captions.json — empty value
renders a TODO placeholder in the .tex body.

Convention for ratio columns
----------------------------
**Ratio-of-means** is used (matches the plot in
analysis_scratch/wind_effect_per_condition.py):

    R_IN  = mean(A_IN, wind)  / mean(A_IN, no)
    R_OUT = mean(A_OUT, wind) / mean(A_OUT, no)
    R_T   = T_wind / T_no    with T_xxx = mean(A_OUT,xxx) / mean(A_IN,xxx)

Std on the ratios is propagated as if the means' std/√n were independent
Gaussian errors:

    σ(R) / R ≈ √( (σ_A_w / A_w)² / n_w + (σ_A_n / A_n)² / n_n )

This is one-sigma run-to-run variability of the ratio estimate, *not* a
confidence interval. With n = 2–4 per cell it should be read as a rough
sense of scatter, not a hypothesis test.

Per-run T values (T_run = A_OUT_run / A_IN_run) are also reported in the
supporting table — that gives a more direct std for T than propagation.

Source data
-----------
analysis_scratch/wind_effect_per_condition_per_run.csv (72 rows produced
by wind_effect_per_condition.py, March panel canon, fullpanel,
quality_flag=ok, 1.3-1.6 Hz × A1/A2/A3, A3@1.6 Hz excluded). A_IN per
run is the mean of probes 9373/170 + 9373/340. A_OUT per run is at
12400/250.
"""

from __future__ import annotations

import sys
from datetime import datetime as _dt
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import os
os.chdir(BASE)

from wavescripts.plot_utils import _lookup_central_caption


SRC = Path("analysis_scratch/wind_effect_per_condition_per_run.csv")
OUT_RATIOS  = Path("analysis_scratch/transmission_wind_ratios_summary.csv")
OUT_AMPS    = Path("analysis_scratch/transmission_wind_amplitudes_per_condition.csv")

OUT_TEX_RATIOS = Path("output/TABLES/ch05_transmission_wind_ratios.tex")
OUT_TEX_AMPS   = Path("output/TABLES/ch05_transmission_wind_amplitudes.tex")
THESIS_NAME_RATIOS = "ch05_transmission_wind_ratios"
THESIS_NAME_AMPS   = "ch05_transmission_wind_amplitudes"
CHAPTER = "05"

AMP_LABEL = {0.1: "A1", 0.2: "A2", 0.3: "A3"}
AMP_FROM_TAG = {v: k for k, v in AMP_LABEL.items()}

# ── Load + symmetric filter ──────────────────────────────────────────────
if not SRC.exists():
    raise SystemExit(
        f"Missing {SRC}. Run analysis_scratch/wind_effect_per_condition.py first."
    )
per_run = pd.read_csv(SRC)
print(f"Loaded {len(per_run)} per-run rows from {SRC.name}")

# Symmetric filter — keep only runs with BOTH A_IN_mean and A_OUT finite
mask = per_run["A_IN_mean"].notna() & per_run["A_OUT"].notna()
n_dropped = (~mask).sum()
per_run_clean = per_run[mask].copy()
print(f"Symmetric filter: kept {len(per_run_clean)} runs, dropped {n_dropped} "
      "(missing A_IN_mean or A_OUT)")

# Per-run transmission
per_run_clean["T_run"] = per_run_clean["A_OUT"] / per_run_clean["A_IN_mean"]


# ── Supporting table: per (f, A, wind) amplitudes + T ────────────────────
agg_rows = []
for (f, a, wind), grp in per_run_clean.groupby(["freq_hz", "amp_v", "wind"], sort=True):
    A_in  = grp["A_IN_mean"].to_numpy()
    A_out = grp["A_OUT"].to_numpy()
    T_per = grp["T_run"].to_numpy()
    n = int(len(grp))
    A_in_mean,  A_in_std  = float(A_in.mean()),  (float(A_in.std(ddof=1))  if n > 1 else np.nan)
    A_out_mean, A_out_std = float(A_out.mean()), (float(A_out.std(ddof=1)) if n > 1 else np.nan)
    # Primary T = ratio of means (matches the R_T plot convention).
    T_ratio_of_means = A_out_mean / A_in_mean if A_in_mean else np.nan
    # Std of T from per-run T values (run-to-run variability).
    T_per_run_std = float(T_per.std(ddof=1)) if n > 1 else np.nan
    agg_rows.append({
        "freq_hz":       round(f, 2),
        "amp_v":         round(a, 2),
        "amp_tag":       AMP_LABEL.get(round(a, 2), "?"),
        "wind_state":    "nowind" if wind == "no" else "wind",
        "A_IN_mean_mm":  round(A_in_mean,  4),
        "A_IN_std_mm":   round(A_in_std,   4) if np.isfinite(A_in_std)  else "",
        "A_OUT_mean_mm": round(A_out_mean, 4),
        "A_OUT_std_mm":  round(A_out_std,  4) if np.isfinite(A_out_std) else "",
        "T_mean":        round(T_ratio_of_means, 4) if np.isfinite(T_ratio_of_means) else "",
        "T_std":         round(T_per_run_std,    4) if np.isfinite(T_per_run_std)    else "",
        "n_runs":        n,
    })

amps_df = pd.DataFrame(agg_rows).sort_values(["amp_v", "freq_hz", "wind_state"]).reset_index(drop=True)
amps_df.to_csv(OUT_AMPS, index=False)
print(f"\n   CSV → {OUT_AMPS}  ({len(amps_df)} rows)")


# ── Main table: R_IN, R_OUT, R_T per (f, A) ──────────────────────────────
def _ratio_with_propagated_std(mean_w: float, std_w: float, n_w: int,
                               mean_n: float, std_n: float, n_n: int) -> tuple[float, float]:
    """R = mean_w / mean_n; σ(R)/R = √( (σ_w/mean_w)²/n_w + (σ_n/mean_n)²/n_n )."""
    if not (np.isfinite(mean_w) and np.isfinite(mean_n) and mean_n != 0):
        return np.nan, np.nan
    R = mean_w / mean_n
    var = 0.0
    if n_w > 1 and np.isfinite(std_w) and mean_w != 0:
        var += (std_w / mean_w) ** 2 / n_w
    if n_n > 1 and np.isfinite(std_n) and mean_n != 0:
        var += (std_n / mean_n) ** 2 / n_n
    return float(R), float(R * np.sqrt(var)) if var > 0 else np.nan


# Pivot the supporting table to wide form keyed by (f, A): one column block per wind state.
def _row_for(condition_df: pd.DataFrame, wind_state: str, col: str):
    sub = condition_df[condition_df["wind_state"] == wind_state]
    if sub.empty:
        return np.nan
    v = sub.iloc[0][col]
    try:
        return float(v) if v != "" else np.nan
    except (TypeError, ValueError):
        return np.nan


ratio_rows = []
for (f, a), grp in amps_df.groupby(["freq_hz", "amp_v"], sort=True):
    A_in_n   = _row_for(grp, "nowind", "A_IN_mean_mm");  A_in_n_std  = _row_for(grp, "nowind", "A_IN_std_mm")
    A_out_n  = _row_for(grp, "nowind", "A_OUT_mean_mm"); A_out_n_std = _row_for(grp, "nowind", "A_OUT_std_mm")
    A_in_w   = _row_for(grp, "wind",   "A_IN_mean_mm");  A_in_w_std  = _row_for(grp, "wind",   "A_IN_std_mm")
    A_out_w  = _row_for(grp, "wind",   "A_OUT_mean_mm"); A_out_w_std = _row_for(grp, "wind",   "A_OUT_std_mm")
    T_n      = _row_for(grp, "nowind", "T_mean")
    T_w      = _row_for(grp, "wind",   "T_mean")
    n_n      = int(_row_for(grp, "nowind", "n_runs") or 0)
    n_w      = int(_row_for(grp, "wind",   "n_runs") or 0)

    R_IN,  R_IN_std  = _ratio_with_propagated_std(A_in_w,  A_in_w_std,  n_w, A_in_n,  A_in_n_std,  n_n)
    R_OUT, R_OUT_std = _ratio_with_propagated_std(A_out_w, A_out_w_std, n_w, A_out_n, A_out_n_std, n_n)

    # R_T propagation — four-amplitude relative variance.
    if all(np.isfinite([T_n, T_w, A_in_n, A_in_w, A_out_n, A_out_w])):
        var = 0.0
        if n_w > 1 and np.isfinite(A_in_w_std)  and A_in_w  != 0: var += (A_in_w_std  / A_in_w ) ** 2 / n_w
        if n_w > 1 and np.isfinite(A_out_w_std) and A_out_w != 0: var += (A_out_w_std / A_out_w) ** 2 / n_w
        if n_n > 1 and np.isfinite(A_in_n_std)  and A_in_n  != 0: var += (A_in_n_std  / A_in_n ) ** 2 / n_n
        if n_n > 1 and np.isfinite(A_out_n_std) and A_out_n != 0: var += (A_out_n_std / A_out_n) ** 2 / n_n
        R_T = T_w / T_n
        R_T_std = float(R_T * np.sqrt(var)) if var > 0 else np.nan
    else:
        R_T = np.nan; R_T_std = np.nan

    ratio_rows.append({
        "freq_hz":        round(f, 2),
        "amp_tag":        AMP_LABEL.get(round(a, 2), "?"),
        "R_IN_mean":      round(R_IN,      4) if np.isfinite(R_IN)      else "",
        "R_IN_std":       round(R_IN_std,  4) if np.isfinite(R_IN_std)  else "",
        "R_OUT_mean":     round(R_OUT,     4) if np.isfinite(R_OUT)     else "",
        "R_OUT_std":      round(R_OUT_std, 4) if np.isfinite(R_OUT_std) else "",
        "R_T_mean":       round(R_T,       4) if np.isfinite(R_T)       else "",
        "R_T_std":        round(R_T_std,   4) if np.isfinite(R_T_std)   else "",
        "n_runs_nowind":  n_n,
        "n_runs_wind":    n_w,
    })

ratios_df = pd.DataFrame(ratio_rows).sort_values(
    ["amp_tag", "freq_hz"]
).reset_index(drop=True)
ratios_df.to_csv(OUT_RATIOS, index=False)
print(f"   CSV → {OUT_RATIOS}  ({len(ratios_df)} rows)")


# ── Console preview ──────────────────────────────────────────────────────
print("\n=== Main table (transmission_wind_ratios_summary.csv) ===")
print(ratios_df.to_string(index=False))

print("\n=== Supporting table (transmission_wind_amplitudes_per_condition.csv) ===")
print(amps_df.to_string(index=False))


# ── Markdown snippet for A1 tier (priority for thesis copy/paste) ────────
def _fmt(v, dp=3, with_pm=None):
    """v with dp decimals; if with_pm given, render '\\num{v} ± \\num{pm}'."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(v):
        return "—"
    if with_pm is None:
        return f"{v:.{dp}f}"
    try:
        pm = float(with_pm)
        if not np.isfinite(pm):
            return f"{v:.{dp}f}"
    except (TypeError, ValueError):
        return f"{v:.{dp}f}"
    return f"{v:.{dp}f} ± {pm:.{dp}f}"


def _md_block(tier: str) -> list[str]:
    """One markdown table for a single amplitude tier — A1, A2, or A3."""
    sub = ratios_df[ratios_df["amp_tag"] == tier].sort_values("freq_hz")
    out = ["", f"=== Markdown snippet — {tier} tier (copy-pasteable) ===", ""]
    out.append("| f [Hz] | R_IN          | R_OUT         | R_T           | n_now | n_wind |")
    out.append("|-------:|:--------------|:--------------|:--------------|------:|-------:|")
    for _, r in sub.iterrows():
        out.append(
            f"| {float(r['freq_hz']):.1f}    | "
            f"{_fmt(r['R_IN_mean'],  3, r['R_IN_std']):<13} | "
            f"{_fmt(r['R_OUT_mean'], 3, r['R_OUT_std']):<13} | "
            f"{_fmt(r['R_T_mean'],   3, r['R_T_std']):<13} | "
            f"{int(r['n_runs_nowind']):>5} | {int(r['n_runs_wind']):>6} |"
        )
    return out


for tier in ("A1", "A2", "A3"):
    print("\n".join(_md_block(tier)))
    print()


# ── LaTeX rendering helpers ──────────────────────────────────────────────
def _fmt_num(v, decimals: int = 3) -> str:
    """\\num{v}, or \\textendash for missing/NaN."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return r"\textendash"
    if not np.isfinite(v):
        return r"\textendash"
    return rf"$\num{{{v:.{decimals}f}}}$"


def _fmt_num_pm(mean, std, decimals: int = 3) -> str:
    """``$\\num{mean} \\pm \\num{std}$``; fall back to plain mean if std missing."""
    try:
        m = float(mean)
    except (TypeError, ValueError):
        return r"\textendash"
    if not np.isfinite(m):
        return r"\textendash"
    try:
        s = float(std)
    except (TypeError, ValueError):
        s = np.nan
    if not np.isfinite(s):
        return rf"$\num{{{m:.{decimals}f}}}$"
    return rf"$\num{{{m:.{decimals}f}}} \pm \num{{{s:.{decimals}f}}}$"


def _caption_block(thesis_name: str) -> str:
    """Render \\caption[short]{full} from central FIGURE_CAPTIONS lookup,
    or a TODO placeholder when both are empty."""
    caption_full  = _lookup_central_caption(thesis_name, kind="full")
    caption_short = _lookup_central_caption(thesis_name, kind="short")
    if caption_full and caption_short:
        return (f"  \\caption[{caption_short}]{{\n"
                f"    {caption_full}\n"
                f"  }}\n")
    if caption_full:
        return f"  \\caption{{\n    {caption_full}\n  }}\n"
    return "  \\caption{\n    % TODO: write caption\n  }\n"


# ── Common immutable-block factory ───────────────────────────────────────
def _immutable_block(thesis_name: str, *, table_kind: str, n_rows: int,
                     extra_lines: list[str]) -> str:
    base = [
        "%! TEX root = ../main.tex",
        "% ==============================================================",
        "% IMMUTABLE — generated automatically, do not edit this block",
        "%",
        "% — Provenance ───────────────────────────────────────────────────",
        "%   script            : analysis_scratch/transmission_wind_tables.py",
        "%   plot_type         : transmission_wind_table",
        f"%   table_kind        : {table_kind}",
        f"%   chapter           : {CHAPTER}",
        f"%   generated_at      : {_dt.now().isoformat(timespec='seconds')}",
        f"%   caption_label     : tab:{thesis_name}",
        f"%   caption_short     : {_lookup_central_caption(thesis_name, kind='short')}",
        f"%   n_rows            : {n_rows}",
        "%",
        "% — Inputs ────────────────────────────────────────────────────",
        f"%   per-run CSV       : {SRC}",
        f"%                       (produced by analysis_scratch/wind_effect_per_condition.py)",
        "%   datasets          : PROCESSED-20260326-*-lowrange,",
        "%                       PROCESSED-20260327-*-lowrange (March panel canon)",
        "%   filters           : PanelCondition=full, quality_flag=ok,",
        "%                       WindCondition in {no, full}, 1.3-1.6 Hz x A1/A2/A3",
        "%                       (A3 @ 1.6 Hz excluded — high-amp dropout caveat)",
        "%   IN canonicalisation: A_IN per run = mean(9373/170, 9373/340)",
        "%   OUT probe         : 12400/250",
        "%",
        "% — Method ────────────────────────────────────────────────────",
        "%   Per (f, A) cell amplitudes are the mean of finite per-run values",
        "%   across all canon runs at that condition (symmetric NaN filter:",
        "%   drop a run if either A_IN or A_OUT is missing).",
        "%   Ratios are ratio-of-means at the (f, A) level (matches the",
        "%   R_IN / R_OUT plot in wind_effect_per_condition.py):",
        "%     R_IN  = mean(A_IN, wind)  / mean(A_IN, no)",
        "%     R_OUT = mean(A_OUT, wind) / mean(A_OUT, no)",
        "%     T_xxx = mean(A_OUT, xxx)  / mean(A_IN, xxx)",
        "%     R_T   = T_wind / T_no",
        "%   sigma propagation (independent-Gaussian on the run-mean estimators):",
        "%     sigma(R) / R ~ sqrt( (sigma_w/mean_w)^2 / n_w",
        "%                         + (sigma_n/mean_n)^2 / n_n )",
        "%   With n = 2-4 per cell (except A1 @ 1.3 Hz: n_no=9, n_wind=10),",
        "%   read sigma as a rough run-to-run scatter, not a confidence interval.",
        "%",
    ]
    base.extend(extra_lines)
    base.append("% ── end immutable block ─────────────────────────────────────────")
    return "\n".join(base)


# ── Table 1 — main: R_IN, R_OUT, R_T per (f, A) ──────────────────────────
def _render_ratios_tex() -> str:
    body_lines = []
    for _, r in ratios_df.iterrows():
        cells = [
            rf"$\num{{{float(r['freq_hz']):.1f}}}$",
            rf"$\mathrm{{{r['amp_tag']}}}$",
            _fmt_num_pm(r["R_IN_mean"],  r["R_IN_std"],  3),
            _fmt_num_pm(r["R_OUT_mean"], r["R_OUT_std"], 3),
            _fmt_num_pm(r["R_T_mean"],   r["R_T_std"],   3),
            rf"$\num{{{int(r['n_runs_nowind'])}}}$",
            rf"$\num{{{int(r['n_runs_wind'])}}}$",
        ]
        body_lines.append("    " + " & ".join(cells) + r" \\")

    immutable = _immutable_block(
        THESIS_NAME_RATIOS,
        table_kind="ratios (main)",
        n_rows=len(ratios_df),
        extra_lines=[
            "% — Columns ───────────────────────────────────────────────────",
            "%   f       : paddle frequency [Hz]",
            "%   amp     : amplitude tier (A1=0.10 V, A2=0.20 V, A3=0.30 V)",
            "%   R_IN    : mean(A_IN, wind)  / mean(A_IN, no)   ± propagated sigma",
            "%   R_OUT   : mean(A_OUT, wind) / mean(A_OUT, no)  ± propagated sigma",
            "%   R_T     : T_wind / T_no                         ± propagated sigma",
            "%   n_no    : run count at no-wind for this (f, A)",
            "%   n_wind  : run count at full-wind for this (f, A)",
            "%",
        ],
    )

    table_body = (
        "\\begin{table}[hbt]\n"
        "  \\centering\n"
        + _caption_block(THESIS_NAME_RATIOS)
        + f"  \\label{{tab:{THESIS_NAME_RATIOS}}}\n"
        "  \\begin{tabular}{ccccccc}\n"
        "    \\toprule\n"
        "    $f$ [\\unit{\\hertz}] &\n"
        "      amp &\n"
        "      $R_\\mathrm{IN}$ &\n"
        "      $R_\\mathrm{OUT}$ &\n"
        "      $R_{K_t}$ &\n"
        "      $n_\\mathrm{no}$ &\n"
        "      $n_\\mathrm{vind}$ \\\\\n"
        "    \\midrule\n"
        + "\n".join(body_lines) + "\n"
        "    \\bottomrule\n"
        "  \\end{tabular}\n"
        "\\end{table}\n"
    )
    return immutable + "\n" + table_body


# ── Table 2 — supporting: A_IN, A_OUT, T per (f, A, wind) ────────────────
def _render_amps_tex() -> str:
    body_lines = []
    for _, r in amps_df.iterrows():
        wind_cell = "no" if r["wind_state"] == "nowind" else "full"
        cells = [
            rf"$\num{{{float(r['freq_hz']):.1f}}}$",
            rf"$\mathrm{{{r['amp_tag']}}}$",
            wind_cell,
            _fmt_num_pm(r["A_IN_mean_mm"],  r["A_IN_std_mm"],  3),
            _fmt_num_pm(r["A_OUT_mean_mm"], r["A_OUT_std_mm"], 3),
            _fmt_num_pm(r["T_mean"],        r["T_std"],        4),
            rf"$\num{{{int(r['n_runs'])}}}$",
        ]
        body_lines.append("    " + " & ".join(cells) + r" \\")

    immutable = _immutable_block(
        THESIS_NAME_AMPS,
        table_kind="amplitudes per condition (supporting)",
        n_rows=len(amps_df),
        extra_lines=[
            "% — Columns ───────────────────────────────────────────────────",
            "%   f             : paddle frequency [Hz]",
            "%   amp           : amplitude tier (A1=0.10 V, A2=0.20 V, A3=0.30 V)",
            "%   wind          : wind state (no, full)",
            "%   A_IN  [mm]    : mean ± std of per-run A_IN_mean over the cell",
            "%                   (A_IN_mean = mean of probes 9373/170 + 9373/340)",
            "%   A_OUT [mm]    : mean ± std of per-run A_OUT over the cell",
            "%                   (A_OUT measured at probe 12400/250)",
            "%   T             : T_mean = mean(A_OUT) / mean(A_IN) (ratio-of-means);",
            "%                   T_std  = std (ddof=1) of per-run T = A_OUT/A_IN",
            "%   n_runs        : run count contributing to this (f, A, wind) cell",
            "%   '\\textendash' : value undefined (n=1 → no std, or missing cell)",
            "%",
        ],
    )

    table_body = (
        "\\begin{table}[hbt]\n"
        "  \\centering\n"
        + _caption_block(THESIS_NAME_AMPS)
        + f"  \\label{{tab:{THESIS_NAME_AMPS}}}\n"
        "  \\begin{tabular}{cccccccr}\n"
        "    \\toprule\n"
        "    $f$ [\\unit{\\hertz}] &\n"
        "      amp &\n"
        "      vind &\n"
        "      $A_\\mathrm{IN}$ [\\unit{\\milli\\metre}] &\n"
        "      $A_\\mathrm{OUT}$ [\\unit{\\milli\\metre}] &\n"
        "      $K_t$ &\n"
        "      $n_\\mathrm{runs}$ \\\\\n"
        "    \\midrule\n"
        + "\n".join(body_lines) + "\n"
        "    \\bottomrule\n"
        "  \\end{tabular}\n"
        "\\end{table}\n"
    )
    return immutable + "\n" + table_body


# ── Write LaTeX outputs ──────────────────────────────────────────────────
OUT_TEX_RATIOS.parent.mkdir(parents=True, exist_ok=True)
OUT_TEX_RATIOS.write_text(_render_ratios_tex(), encoding="utf-8")
print(f"   TEX → {OUT_TEX_RATIOS}")

OUT_TEX_AMPS.write_text(_render_amps_tex(), encoding="utf-8")
print(f"   TEX → {OUT_TEX_AMPS}")


print("\nDone.")
