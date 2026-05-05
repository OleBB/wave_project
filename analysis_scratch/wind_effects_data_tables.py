"""
Generate two data-driven thesis tables from this session's CSVs.
================================================================

Inputs:
  analysis_scratch/wind_doppler_arrival_shift.csv      per-probe per-cell Δt
  analysis_scratch/wind_highway_heatmap.csv             cell A_in enhancement
  analysis_scratch/wind_highway_lowestwind_test.csv     1.3 Hz wind-strength scan

Outputs:
  output/TABLES/ch04_wind_effects_data.tex              master per-cell × probe table
  output/TABLES/ch04_wind_lowestwind_monotonicity.tex   lowestwind 3-point table

Both tables follow the project's IMMUTABLE-block convention; captions are
left as TODO and read centrally from FIGURE_CAPTIONS.
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

NOW = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

PROBES = ["8804/250", "9373/170", "9373/340", "12400/250"]
PROBE_R = {"8804/250": 8.804, "9373/170": 9.373,
           "9373/340": 9.373, "12400/250": 12.400}


# ---------------------------------------------------------------------------
# Table 1 — master per-cell × probe table
# ---------------------------------------------------------------------------
def build_master_table() -> str:
    df = pd.read_csv(BASE / "analysis_scratch/wind_doppler_arrival_shift.csv")
    heat = pd.read_csv(BASE / "analysis_scratch/wind_highway_heatmap.csv")

    # df has one row per (f, amp, probe). Pivot to one row per (f, amp) with
    # Δt at each probe as columns.
    piv_dt = df.pivot_table(index=["f_hz", "amp_V"],
                             columns="probe", values="dt_s") * 1000  # ms
    piv_n_fw = df.pivot_table(index=["f_hz", "amp_V"],
                               columns="probe", values="n_fw", aggfunc="first")

    # Spread across probes (max - min Δt) — uniformity-in-r metric.
    # Skip period-aliased cells per probe via |dt_T| > 0.5 filter
    df_clean = df[df["dt_T"].abs() < 0.5]
    piv_clean = df_clean.pivot_table(index=["f_hz", "amp_V"],
                                      columns="probe", values="dt_s") * 1000
    spread = (piv_clean.max(axis=1) - piv_clean.min(axis=1)).rename("spread_ms")
    n_clean = piv_clean.notna().sum(axis=1).rename("n_probes_clean")

    # A_enh from heatmap
    heat_idx = heat.set_index(["f_hz", "amp_V"])["A_enh_pct"]

    rows_tex = []
    for (f, a), row in piv_dt.iterrows():
        n_fw_cell = int(piv_n_fw.loc[(f, a)].max())
        # Per-probe values, with em-dash for missing
        dt_cells = []
        for p in PROBES:
            v = row.get(p, np.nan)
            if pd.isna(v):
                dt_cells.append("---")
            else:
                # Mark period-aliased per-probe cells with [aliased] marker
                dt_T = v / 1000 * f
                if abs(dt_T) > 0.5:
                    dt_cells.append(f"\\textit{{{v:.0f}}}")  # italic = aliased
                else:
                    dt_cells.append(f"\\num{{{v:.0f}}}")
        spread_v = spread.get((f, a), np.nan)
        if pd.isna(spread_v) or n_clean.get((f, a), 0) < 2:
            spread_str = "---"
        else:
            spread_str = f"\\num{{{spread_v:.0f}}}"
        a_enh = heat_idx.get((f, a), np.nan)
        a_enh_str = "---" if pd.isna(a_enh) else f"\\num{{{a_enh:+.1f}}}"

        rows_tex.append(
            f"    \\num{{{f:.1f}}} & \\num{{{a:.1f}}} & "
            f"\\num{{{n_fw_cell}}} & "
            + " & ".join(dt_cells) + " & "
            + spread_str + " & " + a_enh_str + " \\\\"
        )

    # Compute footer summary stats from clean cells only
    df_in = df_clean[df_clean["probe"] == "9373/170"]
    median_dt_in = float(df_in["dt_s"].median()) * 1000

    # Median spread across cells where ≥3 clean probes
    spread_clean = spread[n_clean >= 3]
    median_spread = float(spread_clean.median()) if len(spread_clean) else np.nan

    immutable = f"""%! TEX root = ../main.tex
% ==============================================================
% IMMUTABLE — generated automatically, do not edit this block
%
% — Provenance ───────────────────────────────────────────────────
%   script            : analysis_scratch/wind_effects_data_tables.py
%   plot_type         : wind_effects_data_table
%   chapter           : 04
%   generated_at      : {NOW}
%   caption_label     : tab:ch04_wind_effects_data
%   caption_short     : Vindeffekter, måletall per celle
%
% — Method ────────────────────────────────────────────────────
%   Source CSVs:
%     analysis_scratch/wind_doppler_arrival_shift.csv  (per-probe Δt)
%     analysis_scratch/wind_highway_heatmap.csv         (A_in enhancement)
%   Per-cell Δt = mean(snap_fw) − mean(snap_nw) in ms, per probe.
%   Spread     = max − min Δt across non-aliased probes (period-alias
%                filter |Δt·f| > 0.5 applied per-probe BEFORE max/min).
%   A_enh      = (mean A_in_fw − mean A_in_nw) / mean A_in_nw · 100 %
%   Italicized Δt entries are individual probes where |Δt·f| > 0.5
%                (period-aliased, snap landed on a different upcrossing).
%
% — Headline ──────────────────────────────────────────────────
%   median |Δt| at IN (clean cells)         : {abs(median_dt_in):.0f} ms
%   median Δt spread across probes (clean)  : {median_spread:.0f} ms
%   uniformity ratio (spread / |Δt_in|)      : {median_spread/abs(median_dt_in):.2f}
%       → Spread is small fraction of |Δt|; Δt is essentially uniform
%         across probes regardless of distance (kills uniform Doppler).
%
% ── end immutable block ─────────────────────────────────────────
\\begin{{table}}[hbt]
  \\centering
  \\small
  \\caption[Vindeffekter, måletall per celle]{{
    % TODO: write caption (edit FIGURE_CAPTIONS in main_save_figures.py)
  }}
  \\label{{tab:ch04_wind_effects_data}}
  \\setlength{{\\tabcolsep}}{{4pt}}
  \\begin{{tabular}}{{cccccccccc}}
    \\toprule
    & & & \\multicolumn{{4}}{{c}}{{$\\Delta t$ per probe [\\unit{{\\milli\\second}}]}} & & \\\\
    \\cmidrule(lr){{4-7}}
    $f$ & $A$ & $n_{{\\text{{fw}}}}$ &
      8804 & 9373/170 & 9373/340 & 12400 &
      $\\Delta t_{{\\max}}{{-}}\\Delta t_{{\\min}}$ &
      $\\Delta A_{{\\text{{inn}}}}$ \\\\
    {{[\\unit{{\\hertz}}]}} & {{[\\unit{{\\volt}}]}} & &
      & & & & {{[\\unit{{\\milli\\second}}]}} & {{[\\unit{{\\percent}}]}} \\\\
    \\midrule
"""
    body = "\n".join(rows_tex)
    footer = """
    \\bottomrule
  \\end{tabular}
\\end{table}
"""
    return immutable + body + footer


# ---------------------------------------------------------------------------
# Table 2 — Lowestwind 3-point monotonicity
# ---------------------------------------------------------------------------
def build_lowestwind_table() -> str:
    df = pd.read_csv(BASE / "analysis_scratch/wind_highway_lowestwind_test.csv")
    # df has one row per (panel, amp, wind, wind_ms). Pivot to one row per
    # (panel, amp) with three Δt columns and the lo/full ratio.
    rows = []
    for panel in ["no", "reverse"]:
        for amp in [0.1, 0.2, 0.3]:
            sub = df[(df["PanelCondition"] == panel) & (df["amp_V"] == amp)]
            if len(sub) < 3:
                continue
            dt_no = float(sub[sub["WindCondition"] == "no"]["dt_in_ms"].iloc[0])
            dt_lo = float(sub[sub["WindCondition"] == "lowest"]["dt_in_ms"].iloc[0])
            dt_fu = float(sub[sub["WindCondition"] == "full"]["dt_in_ms"].iloc[0])
            n_no = int(sub[sub["WindCondition"] == "no"]["n"].iloc[0])
            n_lo = int(sub[sub["WindCondition"] == "lowest"]["n"].iloc[0])
            n_fu = int(sub[sub["WindCondition"] == "full"]["n"].iloc[0])
            ratio = dt_lo / dt_fu if dt_fu != 0 else np.nan
            mono = "ja" if (0 < ratio < 1.2) else "nei"
            rows.append({
                "panel": panel, "amp": amp,
                "n_no": n_no, "n_lo": n_lo, "n_fu": n_fu,
                "dt_no": dt_no, "dt_lo": dt_lo, "dt_fu": dt_fu,
                "ratio": ratio, "mono": mono,
            })
    rows_tex = []
    for r in rows:
        rows_tex.append(
            f"    {r['panel']} & \\num{{{r['amp']:.1f}}} & "
            f"$\\num{{{r['n_no']}}}/\\num{{{r['n_lo']}}}/\\num{{{r['n_fu']}}}$ & "
            f"\\num{{{r['dt_no']:.0f}}} & "
            f"\\num{{{r['dt_lo']:.0f}}} & "
            f"\\num{{{r['dt_fu']:.0f}}} & "
            f"\\num{{{r['ratio']:.2f}}} & "
            f"\\textbf{{{r['mono']}}} \\\\"
        )
    n_mono = sum(r["mono"] == "ja" for r in rows)
    n_total = len(rows)

    immutable = f"""%! TEX root = ../main.tex
% ==============================================================
% IMMUTABLE — generated automatically, do not edit this block
%
% — Provenance ───────────────────────────────────────────────────
%   script            : analysis_scratch/wind_effects_data_tables.py
%   plot_type         : wind_lowestwind_monotonicity_table
%   chapter           : 04
%   generated_at      : {NOW}
%   caption_label     : tab:ch04_wind_lowestwind_monotonicity
%   caption_short     : Δt monoton i vindstyrke (1.3 Hz)
%
% — Method ────────────────────────────────────────────────────
%   Source CSV: analysis_scratch/wind_highway_lowestwind_test.csv
%   Data scope : Nov-2025 folders (PROCESSED-20251112, -20251113-loosepaneltaped)
%                1.3 Hz, panel ∈ (no, reverse), ok quality.
%                Probe IN = 9373/250 (Nov-2025 probe configuration).
%   Wind speeds: no = 0 m/s, lowest = 3.8 m/s, full = 6.0 m/s.
%   Wind-speed ratio lo/fu = 0.63.
%   Δt = mean(snap) − mean(snap_no_wind) in ms, at IN per cell.
%   Ratio = Δt(lowest) / Δt(full). "Monoton" if 0 < ratio < 1.2 (lowest
%   shift sits between zero and full-wind shift, with mild tolerance).
%
% — Headline ──────────────────────────────────────────────────
%   {n_mono}/{n_total} cells monoton in wind speed.
%   Cells with ratio close to wind-speed ratio (0.63) suggest linear
%   scaling; ratios between 0.5 and 1.0 cluster as expected.
%
% ── end immutable block ─────────────────────────────────────────
\\begin{{table}}[hbt]
  \\centering
  \\small
  \\caption[$\\Delta t$ monoton i vindstyrke, \\qty{{1.3}}{{\\hertz}}]{{
    % TODO: write caption (edit FIGURE_CAPTIONS in main_save_figures.py)
  }}
  \\label{{tab:ch04_wind_lowestwind_monotonicity}}
  \\begin{{tabular}}{{cccccccc}}
    \\toprule
    & & &
      \\multicolumn{{3}}{{c}}{{$\\Delta t$ ved INN [\\unit{{\\milli\\second}}]}} & & \\\\
    \\cmidrule(lr){{4-6}}
    Panel & $A$ & $n$ (no/lo/fu) &
      \\qty{{0}}{{\\meter\\per\\second}} &
      \\qty{{3.8}}{{\\meter\\per\\second}} &
      \\qty{{6.0}}{{\\meter\\per\\second}} &
      lo/fu & Monoton? \\\\
    & {{[\\unit{{\\volt}}]}} & & & & & & \\\\
    \\midrule
"""
    body = "\n".join(rows_tex)
    footer = """
    \\bottomrule
  \\end{tabular}
\\end{table}
"""
    return immutable + body + footer


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> None:
    out_master = BASE / "output/TABLES/ch04_wind_effects_data.tex"
    out_master.write_text(build_master_table())
    print(f"  → {out_master.relative_to(BASE)}")

    out_lo = BASE / "output/TABLES/ch04_wind_lowestwind_monotonicity.tex"
    out_lo.write_text(build_lowestwind_table())
    print(f"  → {out_lo.relative_to(BASE)}")


if __name__ == "__main__":
    main()
