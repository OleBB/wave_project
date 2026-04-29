"""
Window-choice proof figure — analytical, no data, all geometry.
================================================================

Single figure: 4 rows (one per thesis frequency: 1.3, 1.4, 1.5, 1.6 Hz),
each row a horizontal time-axis (0 → 50 s) showing two probe lifelines
(IN above, OUT below) with the four physics events marked:

    ▼ t_arr     — main-wave front arrival at probe (r / c_g(f, h))
    █ window    — proposed FFT window [t_arr + 10/f, t_arr + (10+N)/f]
                  N = {10, 13, 17, 15} for f = {1.3, 1.4, 1.5, 1.6} Hz
    ▽ t_paras   — free 2nd-harmonic arrival (r / c_g(2f, h)). Window
                  must end before this for clean spectrum at the probe.
    ╳ paddle    — wavemaker stop time on per40 runs (40/f)

The figure makes three claims visible at a glance:
    1. Window starts ≥ 10T after t_arr           (H&G stable region)
    2. Window ends ≤ t_paras                      (no parasitic 2f)
    3. Window length N(f) is the maximum allowed by the binding pair —
       parasitic-at-IN at low f, per40-ringdown-at-OUT at high f.

The eyeball-measured plateau end (from analysis_scratch/snarvei_eyeballing.md)
is shown as a soft hash band, confirming the window also lives inside the
empirical plateau.

Outputs (scratch only):
    analysis_scratch/window_proof_figure.pdf
    analysis_scratch/window_proof_figure.png

Run from repo root:
    /opt/anaconda3/envs/draumkvedet/bin/python analysis_scratch/window_proof_figure.py
"""

import sys
from pathlib import Path
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

apply_thesis_style()


# ── Config ──────────────────────────────────────────────────────────────
THESIS_FREQS    = [1.3, 1.4, 1.5, 1.6]
N_OFFSET        = 10
N_LENGTH_LOOKUP = {1.3: 10, 1.4: 13, 1.5: 17, 1.6: 15}

PROBES = [
    ("IN",  9.373),
    ("OUT", 12.400),
]
TANK_DEPTH_M = HG.TANK_DEPTH_M
PER40_PERIODS = 40   # paddle-stop = 40/f for per40 runs

# Eyeball plateau ends from analysis_scratch/snarvei_eyeballing.md (Day 1, 0.2 V):
EYEBALL_END_S = {
    "IN":  {1.3: 39.0, 1.4: 39.0, 1.5: 37.0, 1.6: 37.0},
    "OUT": {1.3: 44.0, 1.4: 43.0, 1.5: 41.0, 1.6: 41.0},
}
# Eyeball plateau STARTS (the "first part of train" arrival the user calibrated)
EYEBALL_START_S = {
    "IN":  {1.3: 22.0, 1.4: 22.0, 1.5: 24.0, 1.6: 26.0},
    "OUT": {1.3: 27.0, 1.4: 27.0, 1.5: 28.0, 1.6: 29.0},
}

X_MAX_S = 50.0

# Colours (consistent with rest of CH04)
COL_WIN     = "#2ECC71"   # green  — proposed FFT window
COL_ARRIVAL = "#1F77B4"   # blue   — main-wave arrival
COL_PARAS   = "#D62728"   # red    — parasitic 2f arrival
COL_PSTOP   = "#7F3FBF"   # purple — per40 paddle stop
COL_EYE     = "#888888"   # grey   — eyeball plateau band

OUT_PDF = Path(__file__).parent / "window_proof_figure.pdf"
OUT_PNG = Path(__file__).parent / "window_proof_figure.png"


# ── Geometry helpers ────────────────────────────────────────────────────
def t_arr_at(r_m: float, f: float) -> float:
    return r_m / c_group(f, TANK_DEPTH_M)

def t_paras_at(r_m: float, f: float) -> float:
    return r_m / c_group(2.0 * f, TANK_DEPTH_M)


# ── Figure ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(len(THESIS_FREQS), 1,
                         figsize=(11, 8.8), sharex=True)

ROW_Y_IN  = 1.0
ROW_Y_OUT = 0.0

for ax, f in zip(axes, THESIS_FREQS):
    N = N_LENGTH_LOOKUP[f]
    pad_stop_s = PER40_PERIODS / f

    for row_y, (label, r_m) in zip((ROW_Y_IN, ROW_Y_OUT), PROBES):
        t_arr   = t_arr_at(r_m, f)
        t_paras = t_paras_at(r_m, f)
        win_start = t_arr + N_OFFSET / f
        win_end   = t_arr + (N_OFFSET + N) / f

        # Eyeball plateau band (soft grey hash)
        eb_start = EYEBALL_START_S[label][f]
        eb_end   = EYEBALL_END_S[label][f]
        ax.fill_between([eb_start, eb_end], row_y - 0.18, row_y + 0.18,
                        facecolor="none", edgecolor=COL_EYE,
                        hatch="////", lw=0.0, alpha=0.55, zorder=1)

        # Proposed FFT window (green filled)
        ax.fill_between([win_start, win_end], row_y - 0.18, row_y + 0.18,
                        color=COL_WIN, alpha=0.55, lw=0, zorder=3)
        ax.plot([win_start, win_end], [row_y, row_y],
                color="#1A6E2A", lw=1.0, alpha=0.9, zorder=4)

        # t_arr marker (▼ filled blue)
        ax.scatter([t_arr], [row_y], marker="v", s=72, color=COL_ARRIVAL,
                   edgecolor="black", lw=0.4, zorder=5)
        # t_paras marker (▽ open red)
        ax.scatter([t_paras], [row_y], marker="v", s=72,
                   facecolor="white", edgecolor=COL_PARAS, lw=1.4, zorder=5)
        # paddle stop (×, purple) — only inside x-range
        if pad_stop_s <= X_MAX_S:
            ax.scatter([pad_stop_s], [row_y], marker="x", s=80,
                       color=COL_PSTOP, lw=1.6, zorder=5)

        # Annotations on the timeline (above the row)
        ax.annotate(f"{t_arr:.1f} s", xy=(t_arr, row_y),
                    xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=7, color=COL_ARRIVAL,
                    fontweight="bold")
        ax.annotate(f"{t_paras:.1f} s", xy=(t_paras, row_y),
                    xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=7, color=COL_PARAS,
                    fontweight="bold")
        ax.annotate(f"[{win_start:.1f}, {win_end:.1f}] s\n(N={N} T)",
                    xy=(0.5*(win_start + win_end), row_y - 0.20),
                    xytext=(0, -20), textcoords="offset points",
                    ha="center", fontsize=7, color="#1A6E2A")

        # Probe label on the left
        ax.text(-0.5, row_y, label, fontsize=10, fontweight="bold",
                ha="right", va="center", color="#222")

    # Title-like text in upper-left for the panel (kept empty per project rule;
    # replace by the user later)
    ax.text(0.005, 0.98, f"$f = {f}$ Hz   ($N = {N}$ T)",
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
fig.suptitle("", fontsize=11)

# Single legend at the bottom
legend_handles = [
    Line2D([], [], marker="v", linestyle="",
           markersize=8, markerfacecolor=COL_ARRIVAL,
           markeredgecolor="black", label=r"$t_\mathrm{arr}$  (main wave arrival, $r/c_g(f)$)"),
    Patch(facecolor=COL_WIN, alpha=0.55,
          label=r"proposed window  $[t_\mathrm{arr}+10/f,\;t_\mathrm{arr}+(10+N)/f]$"),
    Line2D([], [], marker="v", linestyle="",
           markersize=8, markerfacecolor="white",
           markeredgecolor=COL_PARAS,
           label=r"$t_\mathrm{paras}$  (free 2$f$ arrival, $r/c_g(2f)$)"),
    Line2D([], [], marker="x", linestyle="", markersize=10,
           color=COL_PSTOP, markeredgewidth=1.6,
           label=r"paddle stop on per40  ($40/f$)"),
    Patch(facecolor="white", edgecolor=COL_EYE, hatch="////",
          label="eyeball plateau (snarvei calib, 0.2 V)"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=3,
           fontsize=8, bbox_to_anchor=(0.5, -0.02), frameon=True)

fig.tight_layout(rect=[0, 0.07, 1, 1])
fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"saved → {OUT_PDF.relative_to(BASE)}")
print(f"        {OUT_PNG.relative_to(BASE)}")
