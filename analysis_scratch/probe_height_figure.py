"""
Probe height & range-mode figure (CH04 §3b)
=============================================

Promotes the analysis in `probe_height_analysis.py` (text-only) and the
review document `probe_height_wind_findings.md` into a thesis-grade
figure. Two-panel layout:

  (A) Stillwater noise floor (mm) — per probe, per condition. Linear
      y-axis; bars show mean (P97.5−P2.5)/2 across nowave+nowind runs;
      error bars = ±1 std across runs.
  (B) Wind background amplitude (mm) — per probe, per condition. Log
      y-axis (OUT probe is ~10× smaller than wind-exposed probes).
      Bars show mean amplitude across nowave+fullwind runs; error bars
      = ±1 std.

Conditions (combinations of probe height above still water × hardware
range mode):

  cond1: h272 / high   — standard, pre-2026-03-23 (longest air path)
  cond2: h136 / high   — transitional, 1 day, n=1 stillwater run
  cond3: h100 / high   — WRONG mode (100 mm < 130 mm window minimum);
                          source of the P2-malfunction runs
  cond4: h100 / low    — correct lowrange; used for all CH05 results

Probes:
  9373/170  IN
  12400/250 OUT (sheltered by panel)
  9373/340  parallel to IN (other lateral side)
  8804/250  upstream (closer to paddle)

Run from repo root:
    conda run -n draumkvedet python analysis_scratch/probe_height_figure.py

Outputs:
    analysis_scratch/probe_height_figure.pdf       (scratch quick-view)
    output/FIGURES/ch04_probe_height.pdf            (thesis figure)
    output/TEXFIGU/ch04_probe_height.tex            (thesis stub, write-once)
    analysis_scratch/probe_height_figure_summary.csv (numeric table)
"""

import sys
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import os
os.chdir(BASE)  # waveprocessed/ paths below are repo-relative

from wavescripts.improved_data_loader import load_analysis_data
from wavescripts.constants import PROBE_RANGE_MODES, PROBE_HEIGHT_DEFAULT_MM
from wavescripts.plot_utils import apply_thesis_style

apply_thesis_style()

# ── Output paths ──────────────────────────────────────────────────────────────
SCRATCH_PDF = Path(__file__).parent / "probe_height_figure.pdf"
SCRATCH_CSV = Path(__file__).parent / "probe_height_figure_summary.csv"
THESIS_NAME = "ch04_probe_height"
OUT_PDF = BASE / "output" / "FIGURES" / f"{THESIS_NAME}.pdf"
OUT_STUB = BASE / "output" / "TEXFIGU" / f"{THESIS_NAME}.tex"
CHAPTER = "04"

# ── Datasets ──────────────────────────────────────────────────────────────────
# Same set as probe_height_analysis.py — every march2026 folder so all 4
# conditions are represented. Older Nov-2025 folders use a different probe
# config and are excluded.
PROCESSED_DIRS = [
    Path("waveprocessed/PROCESSED-20260307-ProbPos4_31_FPV_2-tett6roof"),
    Path("waveprocessed/PROCESSED-20260312-ProbPos4_31_FPV_2-tett6roof"),
    Path("waveprocessed/PROCESSED-20260313-ProbePos4_31_FPV_2-tett6roof"),
    Path("waveprocessed/PROCESSED-20260314-ProbePos4_31_FPV_2-tett6roof"),
    Path("waveprocessed/PROCESSED-20260316-ProbePos4_31_FPV_2-tett6roof"),
    Path("waveprocessed/PROCESSED-20260316-ProbePos4_31_FPV_2-tett6roof-under9Mooring"),
    Path("waveprocessed/PROCESSED-20260319-ProbePos4_31_FPV_2-tett6roof-under9Mooring"),
    Path("waveprocessed/PROCESSED-20260321-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-RENAMED"),
    Path("waveprocessed/PROCESSED-20260323-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height136"),
    Path("waveprocessed/PROCESSED-20260323-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100"),
    Path("waveprocessed/PROCESSED-20260324-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100"),
    Path("waveprocessed/PROCESSED-20260325-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100"),
    Path("waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100"),
    Path("waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange"),
    Path("waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"),
]

# ── Probe + condition definitions ─────────────────────────────────────────────
PROBES = ["9373/170", "12400/250", "9373/340", "8804/250"]
PROBE_LABELS = {
    "9373/170": "9373/170\nIN",
    "12400/250": "12400/250\nOUT",
    "9373/340": "9373/340\nparallel",
    "8804/250": "8804/250\nupstream",
}

# Conditions are listed in the order they were used in the experiment.
# cond3 is flagged as "WRONG" because h=100 mm is below the high-range
# 130 mm window minimum, which produces out-of-spec hardware behaviour.
CONDITIONS = [
    {"key": "cond1_h272_high", "label": "cond1 h272/high",     "color": "#2ECC71"},  # green — standard
    {"key": "cond2_h136_high", "label": "cond2 h136/high",     "color": "#F1C40F"},  # yellow — borderline
    {"key": "cond3_h100_high_WRONG", "label": "cond3 h100/high (WRONG)", "color": "#E74C3C"},  # red — wrong-mode
    {"key": "cond4_h100_low",  "label": "cond4 h100/low",      "color": "#3498DB"},  # blue — correct lowrange
]

# ── 1. Load and label ─────────────────────────────────────────────────────────
print("1. Loading metadata …")
combined_meta, _, _, _ = load_analysis_data(*PROCESSED_DIRS, load_processed=False)
print(f"   {len(combined_meta)} total rows loaded")


def assign_condition(row):
    h = row.get("probe_height_mm", PROBE_HEIGHT_DEFAULT_MM)
    r = row.get("probe_range_mode", "high")
    if pd.isna(h):
        h = PROBE_HEIGHT_DEFAULT_MM
    h = int(h)
    if h == 272 and r == "high":
        return "cond1_h272_high"
    if h == 136 and r == "high":
        return "cond2_h136_high"
    if h == 100 and r == "high":
        return "cond3_h100_high_WRONG"
    if h == 100 and r == "low":
        return "cond4_h100_low"
    return f"other_h{h}_{r}"


combined_meta["condition"] = combined_meta.apply(assign_condition, axis=1)
print("   condition counts:")
print(combined_meta["condition"].value_counts().to_string())

amp_cols = {p: f"Probe {p} Amplitude" for p in PROBES}
missing = [p for p, c in amp_cols.items() if c not in combined_meta.columns]
if missing:
    raise RuntimeError(
        f"Missing amplitude columns for probes {missing}. "
        f"Have: {[c for c in combined_meta.columns if c.startswith('Probe ') and c.endswith(' Amplitude')]}"
    )

# ── 2. Stillwater (nowave + nowind) and wind-background (nowave + fullwind) ─
diag_excl = ["diagnostic", "partial"]

stillwater = combined_meta[
    combined_meta["WaveFrequencyInput [Hz]"].isna()
    & (combined_meta["WindCondition"] == "no")
    & (~combined_meta.get("run_category", pd.Series(["standard"] * len(combined_meta))).isin(diag_excl))
].copy()
print(f"\n2a. Stillwater runs (nowave+nowind, non-diagnostic): {len(stillwater)}")

wind_bg = combined_meta[
    combined_meta["WaveFrequencyInput [Hz]"].isna()
    & (combined_meta["WindCondition"] == "full")
    & (~combined_meta.get("run_category", pd.Series(["standard"] * len(combined_meta))).isin(diag_excl))
].copy()
print(f"2b. Wind-background runs (nowave+fullwind, non-diagnostic): {len(wind_bg)}")


def per_condition_stats(df: pd.DataFrame, probe: str):
    """Return dict {cond_key: (mean, std, n)} for the given probe column."""
    col = amp_cols[probe]
    out = {}
    for c in CONDITIONS:
        sub = df[df["condition"] == c["key"]][col].dropna()
        if len(sub) > 0:
            out[c["key"]] = (float(sub.mean()), float(sub.std()) if len(sub) > 1 else 0.0, int(len(sub)))
        else:
            out[c["key"]] = (np.nan, np.nan, 0)
    return out


sw_stats = {p: per_condition_stats(stillwater, p) for p in PROBES}
wb_stats = {p: per_condition_stats(wind_bg, p) for p in PROBES}

# ── 3. Detection threshold (per probe, max across conditions) ─────────────────
# Used as a horizontal annotation on the noise-floor panel.
# Standard rule from CLAUDE.md §16: ≈ 2× noise floor.
DETECTION_FACTOR = 2.0
detection_threshold = {
    p: DETECTION_FACTOR * np.nanmax([sw_stats[p][c["key"]][0] for c in CONDITIONS])
    for p in PROBES
}

# ── 4. Save numeric summary table ────────────────────────────────────────────
print("\n3. Writing numeric summary CSV …")
rows = []
for p in PROBES:
    for c in CONDITIONS:
        sw_m, sw_s, sw_n = sw_stats[p][c["key"]]
        wb_m, wb_s, wb_n = wb_stats[p][c["key"]]
        rows.append({
            "probe": p,
            "condition": c["key"],
            "sw_mean_mm": sw_m,
            "sw_std_mm": sw_s,
            "sw_n_runs": sw_n,
            "wind_mean_mm": wb_m,
            "wind_std_mm": wb_s,
            "wind_n_runs": wb_n,
            "wind_over_sw": wb_m / sw_m if (sw_m and sw_m > 0) else np.nan,
        })
summary_df = pd.DataFrame(rows)
summary_df.to_csv(SCRATCH_CSV, index=False)
print(f"   Saved → {SCRATCH_CSV.relative_to(BASE)}")
print()
print(summary_df.round(3).to_string(index=False))

# ── 5. Plot ───────────────────────────────────────────────────────────────────
print("\n4. Building figure …")

n_probes = len(PROBES)
n_cond = len(CONDITIONS)
bar_w = 0.18
group_x = np.arange(n_probes)

fig = plt.figure(figsize=(13, 5.6))
gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.22, left=0.07, right=0.985,
                       top=0.86, bottom=0.16)
ax_sw = fig.add_subplot(gs[0, 0])
ax_wb = fig.add_subplot(gs[0, 1])

# Panel A: stillwater noise floor (linear) ─────────────────────────────────────
for j, c in enumerate(CONDITIONS):
    means = [sw_stats[p][c["key"]][0] for p in PROBES]
    stds  = [sw_stats[p][c["key"]][1] for p in PROBES]
    ns    = [sw_stats[p][c["key"]][2] for p in PROBES]
    xs = group_x + (j - (n_cond - 1) / 2) * bar_w
    bars = ax_sw.bar(xs, means, bar_w, label=c["label"], color=c["color"],
                     edgecolor="black", linewidth=0.4, zorder=2)
    # Error bars only where n > 1; otherwise draw a small marker for the
    # single point
    for x, m, s, n in zip(xs, means, stds, ns):
        if n == 0 or np.isnan(m):
            continue
        if n > 1:
            ax_sw.errorbar(x, m, yerr=s, fmt="none", ecolor="black", capsize=2,
                           elinewidth=0.6, alpha=0.65, zorder=3)
        # Annotate n-count under each bar
        ax_sw.text(x, max(m, 0.005) * 0.04, f"n={n}",
                   ha="center", va="bottom", fontsize=5.5, color="white", zorder=5)

# Detection threshold lines (dashed, per probe)
for i, p in enumerate(PROBES):
    th = detection_threshold[p]
    if not np.isnan(th):
        ax_sw.hlines(th, group_x[i] - 0.45, group_x[i] + 0.45,
                     colors="#444", linestyles=(0, (4, 2)), linewidth=1.0, zorder=4)

ax_sw.set_xticks(group_x)
ax_sw.set_xticklabels([PROBE_LABELS[p] for p in PROBES], fontsize=8)
ax_sw.set_ylabel("Stillwater noise (P97.5−P2.5)/2 [mm]", fontsize=9)
ax_sw.set_title("", fontsize=10, fontweight="bold")
ax_sw.grid(True, axis="y", alpha=0.25, lw=0.5)
ax_sw.set_axisbelow(True)
ax_sw.tick_params(axis="y", labelsize=7)
ax_sw.tick_params(axis="x", labelsize=8)
ax_sw.legend(loc="upper right", fontsize=6.5, framealpha=0.92, ncol=1)
ax_sw.text(0.02, 0.97, "dashed = 2× max-cond noise floor (detection threshold)",
           transform=ax_sw.transAxes, fontsize=6.5, va="top", color="#444")

# Panel B: wind background (log) ──────────────────────────────────────────────
for j, c in enumerate(CONDITIONS):
    means = [wb_stats[p][c["key"]][0] for p in PROBES]
    stds  = [wb_stats[p][c["key"]][1] for p in PROBES]
    ns    = [wb_stats[p][c["key"]][2] for p in PROBES]
    xs = group_x + (j - (n_cond - 1) / 2) * bar_w
    bars = ax_wb.bar(xs, means, bar_w, label=c["label"], color=c["color"],
                     edgecolor="black", linewidth=0.4, zorder=2)
    for x, m, s, n in zip(xs, means, stds, ns):
        if n == 0 or np.isnan(m):
            continue
        if n > 1:
            ax_wb.errorbar(x, m, yerr=s, fmt="none", ecolor="black", capsize=2,
                           elinewidth=0.6, alpha=0.65, zorder=3)

ax_wb.set_yscale("log")
ax_wb.set_xticks(group_x)
ax_wb.set_xticklabels([PROBE_LABELS[p] for p in PROBES], fontsize=8)
ax_wb.set_ylabel("Wind-background amplitude [mm, log]", fontsize=9)
ax_wb.set_title("", fontsize=10, fontweight="bold")
ax_wb.grid(True, which="both", axis="y", alpha=0.25, lw=0.5)
ax_wb.set_axisbelow(True)
ax_wb.tick_params(axis="y", labelsize=7)
ax_wb.tick_params(axis="x", labelsize=8)
ax_wb.legend(loc="lower right", fontsize=6.5, framealpha=0.92, ncol=1)

# Annotate the OUT probe specifically — it sits ~10× lower than the others
# because the panel shelters it from the wind fetch.
ax_wb.annotate("OUT probe is sheltered\nby the panel geometry",
               xy=(group_x[1], wb_stats["12400/250"]["cond1_h272_high"][0] or 1.0),
               xytext=(group_x[1] + 0.45, 3.0),
               fontsize=6.5, ha="left", va="center",
               arrowprops=dict(arrowstyle="->", lw=0.7, color="#444"))

# Suptitle with key context
fig.suptitle("", fontsize=11, fontweight="bold", y=0.97)
fig.text(0.5, 0.022,
         "cond3 (h100 highrange) operates 30 mm below the high-range window minimum (130 mm); "
         "noise floor is degraded and the same date is the source of the P2-malfunction runs.",
         ha="center", fontsize=7.5, color="#444", style="italic")

# ── 6. Save ──────────────────────────────────────────────────────────────────
print("5. Saving figure …")
SCRATCH_PDF.parent.mkdir(parents=True, exist_ok=True)
OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
OUT_STUB.parent.mkdir(parents=True, exist_ok=True)

fig.savefig(SCRATCH_PDF, bbox_inches="tight")
print(f"   Saved → {SCRATCH_PDF.relative_to(BASE)}")
fig.savefig(OUT_PDF, bbox_inches="tight")
print(f"   Saved → {OUT_PDF.relative_to(BASE)}")

# Stub: write once. After the user has hand-edited the caption / label,
# never overwrite. Re-running this script will preserve the user's edits.
if not OUT_STUB.exists():
    # Headline numbers for the caption
    in_p = "9373/170"
    out_p = "12400/250"
    sw_c1_in = sw_stats[in_p]["cond1_h272_high"][0]
    sw_c4_in = sw_stats[in_p]["cond4_h100_low"][0]
    sw_c1_out = sw_stats[out_p]["cond1_h272_high"][0]
    sw_c4_out = sw_stats[out_p]["cond4_h100_low"][0]
    wb_c1_out = wb_stats[out_p]["cond1_h272_high"][0]
    wb_c1_in = wb_stats[in_p]["cond1_h272_high"][0]

    _caption = (
        "Stillwater noise floor (a) and wind-background amplitude (b) per probe, "
        "for the four hardware configurations encountered during the experiment: "
        "cond1 ($h\\!=\\!272$\\,mm, high-range, standard), "
        "cond2 ($h\\!=\\!136$\\,mm, high-range, $n\\!=\\!1$ stillwater), "
        "cond3 ($h\\!=\\!100$\\,mm, high-range — wrong mode: $h$ is below the "
        "$130$\\,mm window minimum), and "
        "cond4 ($h\\!=\\!100$\\,mm, low-range — correct, used for all CH05 results). "
        "Bars: mean across nowave runs in each group (errorbars: $\\pm 1\\,\\sigma$ "
        f"where $n\\!>\\!1$). At the IN probe, lowering the probe drops the "
        f"stillwater noise from $\\sim\\!{sw_c1_in*1000:.0f}$\\,$\\mu$m (cond1) to "
        f"$\\sim\\!{sw_c4_in*1000:.0f}$\\,$\\mu$m (cond4) — the longer acoustic "
        "path of cond1 contributes more drift and attenuation. The OUT probe is "
        f"sheltered by the panel geometry, with wind-background $\\sim\\!{wb_c1_out:.2f}$\\,mm "
        f"vs $\\sim\\!{wb_c1_in:.1f}$\\,mm at the IN probe — a $\\sim\\!10\\times$ "
        "reduction. Dashed segments in (a): detection threshold $2\\times$ "
        "max-condition noise floor per probe. Note: cond3 is the source of the "
        "P2-probe-malfunction runs flagged by the pipeline quality system."
    )
    _stub = (
        "%! TEX root = ../main.tex\n"
        "% =============================================================\n"
        "% IMMUTABLE — generated automatically, do not edit this block\n"
        "%   script          : analysis_scratch/probe_height_figure.py\n"
        "%   plot_type       : probe_height\n"
        f"%   chapter         : {CHAPTER}\n"
        "%   conditions      : cond1 h272/high, cond2 h136/high, cond3 h100/high (WRONG), cond4 h100/low\n"
        f"%   probes          : {', '.join(PROBES)}\n"
        f"%   sw_n_total      : {len(stillwater)}\n"
        f"%   wind_n_total    : {len(wind_bg)}\n"
        "% =============================================================\n"
        "\\begin{figure}[htbp]\n"
        "  \\centering\n"
        f"  \\includegraphics[width=0.98\\linewidth]{{FIGURES/{THESIS_NAME}.pdf}}\n"
        "  \\caption[Probe height and range-mode validity]{%\n"
        f"    {_caption}\n"
        "  }\n"
        f"  \\label{{fig:{THESIS_NAME}}}\n"
        "\\end{figure}\n"
    )
    OUT_STUB.write_text(_stub)
    print(f"   Wrote stub → {OUT_STUB.relative_to(BASE)}")
else:
    print(f"   Stub exists (not overwritten): {OUT_STUB.relative_to(BASE)}")

print("\nDone.")
