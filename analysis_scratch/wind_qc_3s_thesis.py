"""
QC plots — pre-paddle wind variability across canon campaign.
==============================================================

Promotes the two diagnostic figures from analysis_scratch/wind_qc_3s.py
to thesis-grade outputs:

    output/FIGURES/ch04_wind_qc_control_chart.pdf
    output/FIGURES/ch04_wind_qc_boxplot.pdf

with TEXFIGU stubs in output/TEXFIGU/.

Inputs (must exist; produced by upstream scratch scripts):
    analysis_scratch/wind_qc_3s_per_run.csv              — per-run sigma at IN+OUT
    analysis_scratch/wind_2s_vs_360s_per_long_run_3s.csv — 5 long-run reference sigmas

Captions are looked up centrally (FIGURE_CAPTIONS in main_save_figures.py).
Body of stub is rewritten on every run (force=True); hand-edit captions
in the central dict, not in the .tex.

Caveats recorded in each stub's immutable block (per user request 2026-05-01):
  (1) QC plots are grouped only by WindCondition × date so far. Other
      run-level factors (Mooring, PanelCondition, run-type per40/per240,
      time-of-day) are NOT yet decomposed in these figures. Refine
      before publication.
  (2) OUT probe (12400/250) sigma sits inside the probe's measured
      stillwater noise floor (0.14-0.36 mm, gold std ~0.14 mm per
      CLAUDE.md §16). Apparent OUT scatter is partly noise floor, not
      genuine wind variability.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import os
os.chdir(BASE)

import wavescripts.plot_utils as pu
from wavescripts.plot_utils import (
    apply_thesis_style, build_fig_meta, save_and_stub, WIND_COLOR_MAP,
)

# ── Inputs / outputs ─────────────────────────────────────────────────────
PER_RUN_CSV  = Path("analysis_scratch/wind_qc_3s_per_run.csv")
LONG_RUN_CSV = Path("analysis_scratch/wind_2s_vs_360s_per_long_run_3s.csv")

CONTROL_NAME = "ch04_wind_qc_control_chart"
BOX_NAME     = "ch04_wind_qc_boxplot"
CHAPTER      = "04"

PROBES = [
    ("9373/170",  "9373/170 (IN, wall)"),
    ("12400/250", "12400/250 (OUT)"),
]
SNIPPET_S = 3.0

WIND_MARKER = {"full": "o", "no": "s", "lowest": "^"}

# Panel A wave / nowave shape distinction (within a wind condition).
WAVE_MARKER  = "o"
NOWAVE_MARKER = "x"

CAVEATS_TEXT = (
    "QC caveats. "
    "(1) Stratification: snippets are grouped here only by WindCondition × date; "
    "other run-level factors (Mooring, PanelCondition, run-type per40/per240, "
    "time-of-day, instrument-cable touches) are NOT yet decomposed and may "
    "explain part of the within-group scatter — refine before publication. "
    "(2) Probe noise floor: OUT (12400/250) σ sits inside the probe's measured "
    "stillwater noise envelope (0.14-0.36 mm; gold standard ≈0.14 mm, "
    "CLAUDE.md §16). Apparent OUT-side scatter is therefore partly the noise "
    "floor of the probe itself, not genuine variability of the wind-wave field. "
    "(3) Reference band: long-run σ range drawn from 5 fullwind+nowave runs of "
    "31, 33, 63, 360, 381 s duration (only 2 of 5 are ≥ 360 s); ensemble "
    "mean is dashed. "
)


# ── Load inputs ──────────────────────────────────────────────────────────
if not PER_RUN_CSV.exists():
    raise SystemExit(
        f"Missing {PER_RUN_CSV}. "
        "Run analysis_scratch/wind_qc_3s.py first."
    )
if not LONG_RUN_CSV.exists():
    raise SystemExit(
        f"Missing {LONG_RUN_CSV}. "
        "Run analysis_scratch/wind_2s_vs_360s.py first."
    )

per_run = pd.read_csv(PER_RUN_CSV)
long_runs_df = pd.read_csv(LONG_RUN_CSV)

# Normalise probe label in long_runs_df (csv has "9373/170 (IN, wall)" full label).
def _short_probe(label: str) -> str:
    return label.split(" ")[0]
long_runs_df["probe_short"] = long_runs_df["probe"].apply(_short_probe)


# ── Long-run reference per probe ─────────────────────────────────────────
long_ref: dict[str, tuple[float, float, float]] = {}
for probe, _ in PROBES:
    sigmas = long_runs_df.loc[long_runs_df["probe_short"] == probe, "sigma_mm"].to_numpy()
    if sigmas.size:
        long_ref[probe] = (float(np.nanmin(sigmas)),
                           float(np.nanmean(sigmas)),
                           float(np.nanmax(sigmas)))
    else:
        long_ref[probe] = (np.nan, np.nan, np.nan)
    print(f"  long-run σ at {probe}: "
          f"min={long_ref[probe][0]:.3f}  mean={long_ref[probe][1]:.3f}  "
          f"max={long_ref[probe][2]:.3f}  mm")


# ── Apply thesis style ───────────────────────────────────────────────────
apply_thesis_style(usetex=False)
plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3})


# ── Figure A: control chart ──────────────────────────────────────────────
sub_data = per_run.copy().sort_values("run_mtime").reset_index(drop=True)

fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.4), sharex=True)
for ax, (probe, label) in zip(axes, PROBES):
    sub = sub_data.dropna(subset=[f"sigma_{probe}"]).reset_index(drop=True)

    smin, smean, smax = long_ref[probe]
    if np.isfinite(smin) and np.isfinite(smax):
        ax.axhspan(smin, smax, color="#444", alpha=0.10,
                   label=rf"langtidsmål $\sigma\in[{smin:.2f},\ {smax:.2f}]$ mm")
    if np.isfinite(smean):
        ax.axhline(smean, color="#444", lw=0.8, ls="--",
                   label=rf"langtidsmål $\overline{{\sigma}}={smean:.2f}$ mm")

    for wc in ("full", "lowest", "no"):
        for is_wave in (True, False):
            mask = (sub["WindCondition"] == wc) & (sub["is_wave"] == is_wave)
            if not mask.any():
                continue
            x = np.arange(len(sub))[mask]
            y = sub.loc[mask, f"sigma_{probe}"].to_numpy(dtype=float)
            ax.scatter(
                x, y,
                color=WIND_COLOR_MAP.get(wc, "#666"),
                marker=WAVE_MARKER if is_wave else NOWAVE_MARKER,
                s=22 if is_wave else 42,
                linewidths=1.2,
                alpha=0.85,
                label=f"{wc} · {'wave' if is_wave else 'nowave'} (n={int(mask.sum())})",
            )

    if len(sub):
        for d in sub["file_date"].unique():
            first = (sub["file_date"] == d).idxmax()
            ax.axvline(first - 0.5, color="#bbb", lw=0.5, alpha=0.6)
            ax.text(first, ax.get_ylim()[1] * 0.97, f" {d}", fontsize=7,
                    va="top", ha="left", color="#666",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white",
                              ec="none", alpha=0.85))

    ax.set_ylabel(r"$\sigma_\eta$ [mm]")
    ax.set_title(label, fontsize=10)
    ax.legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.92)

axes[1].set_xlabel("Kjøring (kronologisk)")
fig.suptitle(
    f"Kontrolldiagram — første {SNIPPET_S:g} s av hver kjøring (canon, kvalitet=ok)",
    fontsize=11,
)
fig.tight_layout()

control_meta = build_fig_meta(
    {
        "filters": {
            "PanelCondition":  "all",
            "WindCondition":   "full+no+lowest",
            "quality_flag":    "ok",
        },
        "plotting": {
            "figure_name": CONTROL_NAME,
        },
    },
    chapter=CHAPTER,
    extra={
        "script": "analysis_scratch/wind_qc_3s_thesis.py",
        "snippet_s": SNIPPET_S,
    },
    computed_in=("analysis_scratch/wind_qc_3s.py (per-run σ over first 3 s) "
                 "→ analysis_scratch/wind_qc_3s_thesis.py (thesis render)"),
    data_class="DELEG",
    findings_doc="(none — diagnostic figure, see immutable-block caveats)",
    fft_window_hz=0.0,
    extra_params=(
        f"sigma_eta = std of η over first {SNIPPET_S:g} s of each ok run "
        f"(pre-paddle for wave runs, opening 3 s for nowave runs — both "
        f"are pre-disturbance: √(gh)=2.39 m/s, closest probe at 8804 mm "
        f"safe to 3.68 s). "
        f"Datasets: PROCESSED-20260326-*-lowrange + PROCESSED-20260327-*-"
        f"lowrange (canon March-2026, h100 + low-range ULS). "
        f"Long-run reference band: 5 fullwind+nowave runs (durations 31, "
        f"33, 63, 360, 381 s; only 2 of 5 are ≥ 360 s) — phrase carefully "
        f'in caption ("31-381 s", not "5 × 360 s"). '
        f"Wind colours follow CLAUDE.md WIND_COLOR_MAP (red=full, "
        f"blue=no, green=lowest). Marker shape: circle = wave run, "
        f"× = nowave run. Date-change lines are vertical grey ticks. "
        f"Thesis style via pu.apply_thesis_style(); NewComputerModern10. "
        + CAVEATS_TEXT
    ),
    extra_stats={
        f"long_sigma_min_mm[{probe}]":  f"{long_ref[probe][0]:.3f}"
        for probe, _ in PROBES
    } | {
        f"long_sigma_mean_mm[{probe}]": f"{long_ref[probe][1]:.3f}"
        for probe, _ in PROBES
    } | {
        f"long_sigma_max_mm[{probe}]":  f"{long_ref[probe][2]:.3f}"
        for probe, _ in PROBES
    } | {
        "n_runs_total":  str(int(sub_data.shape[0])),
        "n_long_runs":   str(int(long_runs_df["long_run"].nunique())),
    },
)
save_and_stub(fig, control_meta, plot_type="wind_qc_control_chart")
plt.close(fig)


# ── Figure B: boxplot WindCondition × date ───────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(7.5, 4.0), sharey=False)
for ax, (probe, label) in zip(axes, PROBES):
    sub = per_run.dropna(subset=[f"sigma_{probe}"]).copy()
    sub["_grp"] = sub["WindCondition"] + "\n" + sub["file_date"]
    groups = sorted(sub["_grp"].unique())
    data = [sub.loc[sub["_grp"] == g, f"sigma_{probe}"].to_numpy(dtype=float)
            for g in groups]
    counts = [len(d) for d in data]

    bp = ax.boxplot(
        data,
        labels=[f"{g}\n(n={n})" for g, n in zip(groups, counts)],
        patch_artist=True, showfliers=True, whis=(5, 95),
    )
    for patch, g in zip(bp["boxes"], groups):
        wc = g.split("\n")[0]
        patch.set_facecolor(WIND_COLOR_MAP.get(wc, "#999"))
        patch.set_alpha(0.42)
    for med in bp["medians"]:
        med.set_color("black"); med.set_linewidth(1.4)

    smin, smean, smax = long_ref[probe]
    if np.isfinite(smin) and np.isfinite(smax):
        ax.axhspan(smin, smax, color="#444", alpha=0.10)
    if np.isfinite(smean):
        ax.axhline(smean, color="#444", lw=0.8, ls="--", label="langtidsmål $\\overline{\\sigma}$")

    ax.set_ylabel(r"$\sigma_\eta$ [mm]")
    ax.set_title(label, fontsize=10)
    ax.tick_params(axis="x", rotation=15, labelsize=7)
    ax.legend(loc="upper right", fontsize=7)

fig.suptitle(
    "Fordeling av $\\sigma_\\eta$ etter (vind × dato) — første 3 s, canon",
    fontsize=11,
)
fig.tight_layout()

box_meta = build_fig_meta(
    {
        "filters": {
            "PanelCondition":  "all",
            "WindCondition":   "full+no+lowest",
            "quality_flag":    "ok",
        },
        "plotting": {
            "figure_name": BOX_NAME,
        },
    },
    chapter=CHAPTER,
    extra={
        "script": "analysis_scratch/wind_qc_3s_thesis.py",
        "snippet_s": SNIPPET_S,
    },
    computed_in=("analysis_scratch/wind_qc_3s.py (per-run σ over first 3 s) "
                 "→ analysis_scratch/wind_qc_3s_thesis.py (thesis render)"),
    data_class="DELEG",
    findings_doc="(none — diagnostic figure, see immutable-block caveats)",
    fft_window_hz=0.0,
    extra_params=(
        f"Each box pools the first-3 s σ_η across all ok runs at one "
        f"(WindCondition × date) cell. Whiskers: 5-95 percentile; "
        f"median = solid black line; box = IQR. Long-run reference "
        f"(grey band + dashed mean) overlaid. "
        f"Box face colour = wind condition (CLAUDE.md WIND_COLOR_MAP, "
        f"alpha=0.42). Datasets: canon March-2026 -lowrange. "
        + CAVEATS_TEXT
    ),
    extra_stats={
        f"long_sigma_mean_mm[{probe}]": f"{long_ref[probe][1]:.3f}"
        for probe, _ in PROBES
    } | {
        "n_runs_total":  str(int(per_run.shape[0])),
    },
)
save_and_stub(fig, box_meta, plot_type="wind_qc_boxplot")
plt.close(fig)

print("Done.")
