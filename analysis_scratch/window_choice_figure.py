"""
Window-choice figure — CH04 §4o, nowind + fullwind plateau bounds visualised.
==============================================================================

Builds on `window_proof_figure.py`. Same 4-rows-per-frequency layout, two
probe lifelines per row (IN above, OUT below), but each lifeline now carries
TWO horizontal hash bands stacked vertically:

    upper hash (grey  ////) — nowind  eyeball plateau (RampDetectionBrowser,
                              snarvei_eyeballing.md Day 1, 0.2 V)
    lower hash (red   \\\\) — fullwind empirical plateau (per40 sliding A_FFT,
                              ±2 % relaxed criterion, per40_plateau_end_*.csv)

Markers (same as window_proof_figure):

    ▼ filled blue   — t_arr   (main wave arrival, r / c_g(f, h))
    ▽ open red      — t_paras (free 2f arrival, r / c_g(2f, h))
    ╳ purple        — paddle stop on per40 (40 / f)

Green block — proposed Option B FFT window:
    t_start = r/c_g(f, h) + 10/f
    t_end   = t_start + N(f)/f, with N(f) = {1.3: 10, 1.4: 13, 1.5: 13, 1.6: 13}

Outputs:
    output/FIGURES/ch04_window_choice.{pdf,pgf}
    output/TEXFIGU/ch04_window_choice.tex
    analysis_scratch/window_choice_figure.{pdf,png}

Run from repo root:
    /opt/anaconda3/envs/draumkvedet/bin/python analysis_scratch/window_choice_figure.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import os
os.chdir(BASE)

from wavescripts.constants import c_group, HG
from wavescripts.plot_utils import apply_thesis_style
import wavescripts.plot_utils as pu

apply_thesis_style()


# ── Config ──────────────────────────────────────────────────────────────
THESIS_FREQS    = [1.3, 1.4, 1.5, 1.6]
N_OFFSET        = 10
N_LENGTH_LOOKUP = {1.3: 10, 1.4: 13, 1.5: 13, 1.6: 13}

PROBES = [
    ("IN",  9.373),
    ("OUT", 12.400),
]
TANK_DEPTH_M  = HG.TANK_DEPTH_M
PER40_PERIODS = 40

# Eyeball plateau (nowind, 0.2 V) from snarvei_eyeballing.md (Day 1).
EYEBALL_NOWIND = {
    "IN":  {1.3: (22.0, 39.0), 1.4: (22.0, 39.0), 1.5: (24.0, 37.0), 1.6: (26.0, 37.0)},
    "OUT": {1.3: (27.0, 44.0), 1.4: (27.0, 43.0), 1.5: (28.0, 41.0), 1.6: (29.0, 41.0)},
}

# Empirical plateau (fullwind, per40, ±2 % relaxed) from
# per40_plateau_end_aggregated.csv. Start = t_arr + 8/f. End = plat_end_relax.
AGG_CSV = Path("analysis_scratch/per40_plateau_end_aggregated.csv")

X_MAX_S = 50.0

THESIS_NAME = "ch04_window_choice"
THESIS_FIGS  = BASE / "output" / "FIGURES"
THESIS_STUBS = BASE / "output" / "TEXFIGU"
SCRATCH_PNG  = Path(__file__).parent / "window_choice_figure.png"
SCRATCH_PDF  = Path(__file__).parent / "window_choice_figure.pdf"
THESIS_FIGS.mkdir(parents=True, exist_ok=True)
THESIS_STUBS.mkdir(parents=True, exist_ok=True)

# Colours
COL_WIN     = "#2ECC71"
COL_ARRIVAL = "#1F77B4"
COL_PARAS   = "#D62728"
COL_PSTOP   = "#7F3FBF"
COL_EYE_NW  = "#666666"   # grey  — nowind  eyeball
COL_EYE_FW  = "#C0392B"   # red   — fullwind empirical


# ── Geometry helpers ────────────────────────────────────────────────────
def t_arr_at(r_m: float, f: float) -> float:
    return r_m / c_group(f, TANK_DEPTH_M)

def t_paras_at(r_m: float, f: float) -> float:
    return r_m / c_group(2.0 * f, TANK_DEPTH_M)


# ── Empirical fullwind plateau ──────────────────────────────────────────
agg = pd.read_csv(AGG_CSV)


def fullwind_plateau(probe_label: str, f: float):
    sub = agg[(agg["freq_hz"] == f) & (agg["probe"] == probe_label)
              & (agg["wind"] == "full") & (agg["run_type"] == "per40")]
    if sub.empty:
        return (np.nan, np.nan)
    r_m = next(r for lbl, r in PROBES if lbl == probe_label)
    plat_start = t_arr_at(r_m, f) + 8.0 / f
    plat_end   = float(sub["plat_end_relax_s_med"].iloc[0])
    return (plat_start, plat_end)


# ── Figure ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(len(THESIS_FREQS), 1,
                         figsize=(11, 9.4), sharex=True)

ROW_Y_IN  = 1.0
ROW_Y_OUT = 0.0

# Vertical offsets for the two plateau hash bands (relative to row baseline)
H_HALF      = 0.18      # half-height of each band
GAP_BETWEEN = 0.05      # gap between nowind (upper) and fullwind (lower) bands

for ax, f in zip(axes, THESIS_FREQS):
    N = N_LENGTH_LOOKUP[f]
    pad_stop_s = PER40_PERIODS / f

    for row_y, (label, r_m) in zip((ROW_Y_IN, ROW_Y_OUT), PROBES):
        t_arr   = t_arr_at(r_m, f)
        t_paras = t_paras_at(r_m, f)
        win_start = t_arr + N_OFFSET / f
        win_end   = t_arr + (N_OFFSET + N) / f

        # ── Plateau bands: upper = nowind eyeball, lower = fullwind empirical
        eb_no_s, eb_no_e = EYEBALL_NOWIND[label][f]
        eb_fw_s, eb_fw_e = fullwind_plateau(label, f)

        upper_lo = row_y + GAP_BETWEEN / 2
        upper_hi = row_y + H_HALF
        lower_lo = row_y - H_HALF
        lower_hi = row_y - GAP_BETWEEN / 2

        ax.fill_between([eb_no_s, eb_no_e], upper_lo, upper_hi,
                        facecolor="none", edgecolor=COL_EYE_NW,
                        hatch="////", lw=0.0, alpha=0.65, zorder=1)
        if np.isfinite(eb_fw_s) and np.isfinite(eb_fw_e):
            ax.fill_between([eb_fw_s, eb_fw_e], lower_lo, lower_hi,
                            facecolor="none", edgecolor=COL_EYE_FW,
                            hatch=r"\\\\", lw=0.0, alpha=0.55, zorder=1)

        # ── Proposed FFT window (green filled, full row height)
        ax.fill_between([win_start, win_end], row_y - H_HALF, row_y + H_HALF,
                        color=COL_WIN, alpha=0.45, lw=0, zorder=3)
        ax.plot([win_start, win_end], [row_y, row_y],
                color="#1A6E2A", lw=1.0, alpha=0.9, zorder=4)

        # ── Markers
        ax.scatter([t_arr], [row_y], marker="v", s=72, color=COL_ARRIVAL,
                   edgecolor="black", lw=0.4, zorder=5)
        ax.scatter([t_paras], [row_y], marker="v", s=72,
                   facecolor="white", edgecolor=COL_PARAS, lw=1.4, zorder=5)
        if pad_stop_s <= X_MAX_S:
            ax.scatter([pad_stop_s], [row_y], marker="x", s=80,
                       color=COL_PSTOP, lw=1.6, zorder=5)

        # ── Annotations
        ax.annotate(f"{t_arr:.1f} s", xy=(t_arr, row_y),
                    xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=7, color=COL_ARRIVAL,
                    fontweight="bold")
        ax.annotate(f"{t_paras:.1f} s", xy=(t_paras, row_y),
                    xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=7, color=COL_PARAS,
                    fontweight="bold")
        ax.annotate(f"[{win_start:.1f}, {win_end:.1f}] s\n($N{{=}}{N}$ T)",
                    xy=(0.5*(win_start + win_end), row_y - H_HALF - 0.02),
                    xytext=(0, -22), textcoords="offset points",
                    ha="center", fontsize=7, color="#1A6E2A")

        # Probe label on the left
        ax.text(-0.5, row_y, label, fontsize=10, fontweight="bold",
                ha="right", va="center", color="#222")

    # Per-panel header
    ax.text(0.005, 0.985, f"$f = {f}$ Hz   ($N = {N}$ T)",
            transform=ax.transAxes,
            ha="left", va="top", fontsize=9.5, color="#222",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#bbb",
                      alpha=0.85, lw=0.4))

    ax.set_ylim(-0.65, 1.65)
    ax.set_xlim(-2.0, X_MAX_S)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.grid(True, axis="x", alpha=0.25, lw=0.4)

axes[-1].set_xlabel("time from wavemaker start [s]", fontsize=10)

# Single legend at the bottom
legend_handles = [
    Line2D([], [], marker="v", linestyle="",
           markersize=8, markerfacecolor=COL_ARRIVAL, markeredgecolor="black",
           label=r"$t_\mathrm{arr}$  (main-wave arrival, $r/c_g(f)$)"),
    Patch(facecolor=COL_WIN, alpha=0.45,
          label=r"proposed window  $[t_\mathrm{arr}+10/f,\;t_\mathrm{arr}+(10+N)/f]$"),
    Line2D([], [], marker="v", linestyle="",
           markersize=8, markerfacecolor="white",
           markeredgecolor=COL_PARAS, markeredgewidth=1.4,
           label=r"$t_\mathrm{2f}$  (free 2$f$ arrival, $r/c_g(2f)$)"),
    Line2D([], [], marker="x", linestyle="", markersize=10,
           color=COL_PSTOP, markeredgewidth=1.6,
           label=r"paddle stop on per40  ($40/f$)"),
    Patch(facecolor="white", edgecolor=COL_EYE_NW, hatch="////",
          label="plateau (nowind, eyeball)"),
    Patch(facecolor="white", edgecolor=COL_EYE_FW, hatch=r"\\\\",
          label=r"plateau (fullwind, empirical $\pm 2\%$)"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=3,
           fontsize=8, bbox_to_anchor=(0.5, -0.02), frameon=True)

fig.tight_layout(rect=[0, 0.07, 1, 1])

# ── Save scratch + thesis PDFs ──────────────────────────────────────────
fig.savefig(SCRATCH_PDF, bbox_inches="tight")
fig.savefig(SCRATCH_PNG, dpi=130, bbox_inches="tight")
thesis_pdf = THESIS_FIGS / f"{THESIS_NAME}.pdf"
# thesis_pgf = THESIS_FIGS / f"{THESIS_NAME}.pgf"
fig.savefig(thesis_pdf, bbox_inches="tight")
plt.close(fig)
print(f"saved → {SCRATCH_PDF.relative_to(BASE)}")
print(f"        {SCRATCH_PNG.relative_to(BASE)}")
print(f"        {thesis_pdf.relative_to(BASE)}")


# ── TEXFIGU stub via shared helper ──────────────────────────────────────
pu.TEXFIGU_DIR = THESIS_STUBS
pu.FIGURES_DIR = THESIS_FIGS

_meta_stub = pu.build_fig_meta(
    {
        "filters": {
            "WaveAmplitudeInput [Volt]": 0.2,
            "PanelCondition":            "full",
            "WindCondition":             ["no", "full"],
            "probes":                    "IN=9373/170+9373/340, OUT=12400/250",
        },
        "plotting": {"figure_name": THESIS_NAME},
    },
    chapter="04",
    extra={"script": "analysis_scratch/window_choice_figure.py"},
    computed_in=(
        "analysis_scratch/window_choice_figure.py "
        "(Option B post-squeeze: t_start = r/c_g(f,h) + 10/f, "
        "length N(f) where N = {1.3:10, 1.4:13, 1.5:13, 1.6:13})"
    ),
    data_class="GEOM",
    grouper="one (f, probe) row per cell, no per-run aggregation",
    collapse_panels=False,
    extra_params=(
        f"N_offset = {N_OFFSET} periods; "
        f"N(f)     = {N_LENGTH_LOOKUP}; "
        f"r_IN     = {PROBES[0][1]} m, r_OUT = {PROBES[1][1]} m; "
        f"depth h  = {TANK_DEPTH_M} m"
    ),
    extra_stats={
        "plateau_nowind":   "eyeball, snarvei_eyeballing.md (Day 1, 0.2 V)",
        "plateau_fullwind": "per40 sliding-A_FFT, ±2% relaxed criterion",
    },
)
pu.write_figure_stub(_meta_stub, plot_type="window_choice")
print(f"        output/TEXFIGU/{THESIS_NAME}.tex")
