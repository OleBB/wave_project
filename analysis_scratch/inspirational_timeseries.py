"""
Inspirational time-series figures for the opening of Ch. 4 §5
("reading a time series").

Canon run: nowind, 1.4 Hz, 0.2 V, per240, fullpanel — clean ramp-up,
long stable plateau, paddle stop, slow decay. Enough material to carry
a full page without clutter.

Variants (all IN on top / OUT on bottom, shared axes):
  A — bare stacked time series, nothing but the wave
  B — annotated narrative: H&G analysis window shaded, paddle-on band,
      one period marked as a ruler in the plateau
  C — macro + micro: full 200 s run above, 5-period zoom below per probe

Output: analysis_scratch/inspirational_timeseries_{A,B,C}.pdf
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

from wavescripts.improved_data_loader import load_analysis_data, load_processed_dfs
from wavescripts.plot_utils import apply_thesis_style, WIND_COLOR_MAP

FS = 250.0
BASE = Path("/Users/ole/Kodevik/wave_project")
SCRATCH = Path(__file__).parent

TARGET_DIR = BASE / "waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"
CHOSEN_PATH = str(BASE / "wavedata/20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange/fullpanel-nowind-amp0200-freq1400-per240-depth580-mstop30-run1.csv")

IN_PROBE  = "9373/170"
OUT_PROBE = "12400/250"
FREQ = 1.4
AMP_V = 0.2

# Thesis-wide convention: colour encodes wind condition, not probe identity.
# Both IN and OUT share the wind colour; the stack layout and labels already
# separate them visually.
WIND = "no"                           # "no" → blue, "full" → red, "lowest" → green
WIND_COLOR = WIND_COLOR_MAP[WIND]
IN_COLOR  = WIND_COLOR
OUT_COLOR = WIND_COLOR
WIN_COLOR = "#F1B24A"                 # soft amber (analysis-window marker, neutral)
PADDLE_BAND_COLOR = "#E8EEF7"         # very pale blue-grey (backdrop, neutral)

apply_thesis_style()


# ─── Load ────────────────────────────────────────────────────────────────
print("Loading …")
meta, _, _, _ = load_analysis_data(str(TARGET_DIR), load_processed=False)
proc = load_processed_dfs(str(TARGET_DIR))
row = meta[meta["path"] == CHOSEN_PATH].iloc[0]
df = proc[CHOSEN_PATH]
t = np.arange(len(df)) / FS

def get_eta(probe):
    col = f"eta_{probe}_interp"
    if col not in df.columns:
        col = f"eta_{probe}"
    return df[col].to_numpy(dtype=float)

eta_in  = get_eta(IN_PROBE)
eta_out = get_eta(OUT_PROBE)

def window_s(probe):
    s = int(row[f"Computed Probe {probe} start"])
    e = int(row[f"Computed Probe {probe} end"])
    return s / FS, e / FS

in_ws,  in_we  = window_s(IN_PROBE)
out_ws, out_we = window_s(OUT_PROBE)

# shared symmetric y-range for clean stacking
def sym_ylim(*sigs, pad=0.15):
    y = np.concatenate(sigs)
    lo, hi = np.nanpercentile(y, [0.5, 99.5])
    m = max(abs(lo), abs(hi))
    return -m * (1 + pad), m * (1 + pad)

ylim = sym_ylim(eta_in, eta_out)


def apply_ticks(ax, *, x_major=5.0, x_minor=1.0, y_major=2.0, y_minor=1.0):
    """Major + minor grid ticks. Call after xlim/ylim are set."""
    ax.xaxis.set_major_locator(MultipleLocator(x_major))
    ax.xaxis.set_minor_locator(MultipleLocator(x_minor))
    ax.yaxis.set_major_locator(MultipleLocator(y_major))
    ax.yaxis.set_minor_locator(MultipleLocator(y_minor))
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(True, which="minor", alpha=0.12)

# per240 = 240 periods at target freq
paddle_on_end_approx_s = 240.0 / FREQ  # ≈ 171.4 s
# wavemaker soft-start before sample 0 is small; treat paddle-on band as [0, 240/f]
paddle_on_start_s = 0.0

# Cut the x-axis ~10 s after the later of the two H&G window ends — we want
# ramp-up, the analysis window, and a short post-window breather, nothing more.
X_CUTOFF_S = max(in_we, out_we) + 10.0


# ─── Figure A: bare stacked ──────────────────────────────────────────────
print("Rendering A …")
figA, (axA_in, axA_out) = plt.subplots(
    2, 1, figsize=(10, 7.8), sharex=True, sharey=True
)

axA_in.plot(t, eta_in, color=IN_COLOR, lw=0.55)
axA_out.plot(t, eta_out, color=OUT_COLOR, lw=0.55)

for ax, probe, color, label in [
    (axA_in,  IN_PROBE,  IN_COLOR,  "IN"),
    (axA_out, OUT_PROBE, OUT_COLOR, "OUT"),
]:
    ax.set_ylabel(r"$\eta$ (mm)")
    ax.text(
        0.005, 0.92, f"{label}  ·  {probe}",
        transform=ax.transAxes, va="top", ha="left",
        fontsize=10, color=color, weight="bold",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, lw=0.6),
    )
    ax.axhline(0, color="#888", lw=0.5, alpha=0.6)

axA_out.set_xlabel("time from recording start (s)")
axA_in.set_xlim(0, t[-1])
axA_in.set_ylim(ylim)
for ax in (axA_in, axA_out):
    apply_ticks(ax)
figA.suptitle(
    f"Wave-tank recording · nowind · $f$ = {FREQ} Hz · $A$ = {AMP_V} V · per240",
    fontsize=11,
)
figA.tight_layout(rect=[0, 0, 1, 0.96])
outA = SCRATCH / "inspirational_timeseries_A_stacked.pdf"
figA.savefig(outA)
figA.savefig(outA.with_suffix(".png"), dpi=180)
# cutoff variant — window + ~10 s tail
axA_in.set_xlim(0, X_CUTOFF_S)
outA_cut = SCRATCH / "inspirational_timeseries_A_stacked_cut.pdf"
figA.savefig(outA_cut)
figA.savefig(outA_cut.with_suffix(".png"), dpi=180)
plt.close(figA)
print(f"  → {outA.name} + _cut (each with .png)")


# ─── Figure B: annotated narrative ───────────────────────────────────────
print("Rendering B …")
figB, (axB_in, axB_out) = plt.subplots(
    2, 1, figsize=(10, 6.2), sharex=True, sharey=True
)

for ax, sig, probe, color, ws, we, label in [
    (axB_in,  eta_in,  IN_PROBE,  IN_COLOR,  in_ws,  in_we,  "IN"),
    (axB_out, eta_out, OUT_PROBE, OUT_COLOR, out_ws, out_we, "OUT"),
]:
    # pale paddle-on band
    ax.axvspan(
        paddle_on_start_s, paddle_on_end_approx_s,
        color=PADDLE_BAND_COLOR, alpha=1.0, zorder=0,
        label=f"paddle on (≈{int(paddle_on_end_approx_s)} s, 240 T)",
    )
    # H&G 10-period analysis window
    ax.axvspan(
        ws, we, color=WIN_COLOR, alpha=0.55, zorder=1,
        label=f"FFT window  [{int(round(ws*FREQ))}T, {int(round(we*FREQ))}T]",
    )
    ax.plot(t, sig, color=color, lw=0.55, zorder=2)
    ax.axhline(0, color="#666", lw=0.5, alpha=0.6, zorder=1)
    ax.set_ylabel(r"$\eta$ (mm)")
    ax.text(
        0.005, 0.92, f"{label}  ·  {probe}",
        transform=ax.transAxes, va="top", ha="left",
        fontsize=10, color=color, weight="bold",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, lw=0.6),
    )
    ax.legend(loc="lower left", frameon=True, framealpha=0.93, fontsize=8)

# one-period ruler (visible only in IN, placed mid-window)
T = 1.0 / FREQ
ruler_center = (in_ws + in_we) / 2
ruler_x0 = ruler_center - T / 2
ruler_x1 = ruler_center + T / 2
ruler_y = ylim[1] * 0.72
axB_in.annotate(
    "", xy=(ruler_x1, ruler_y), xytext=(ruler_x0, ruler_y),
    arrowprops=dict(arrowstyle="|-|", lw=1.0, color="#333"),
)
axB_in.text(
    ruler_center, ruler_y + 0.05 * (ylim[1] - ylim[0]),
    f"$T$ = {1000*T:.1f} ms", ha="center", va="bottom",
    fontsize=9, color="#333",
)

# end-of-paddle vertical marker (dashed, both axes)
_paddle_off_artists = []
for ax in (axB_in, axB_out):
    _paddle_off_artists.append(
        ax.axvline(paddle_on_end_approx_s, color="#555", lw=0.7, ls="--", alpha=0.7)
    )
_paddle_off_artists.append(
    axB_out.text(
        paddle_on_end_approx_s + 1.5, ylim[0] * 0.85,
        "wavemaker off", fontsize=9, color="#444", va="bottom", ha="left",
    )
)

axB_out.set_xlabel("time from recording start (s)")
axB_in.set_xlim(0, t[-1])
axB_in.set_ylim(ylim)
for ax in (axB_in, axB_out):
    apply_ticks(ax)
figB.suptitle(
    f"Reading a time series · nowind · $f$ = {FREQ} Hz · $A$ = {AMP_V} V",
    fontsize=11,
)
figB.tight_layout(rect=[0, 0, 1, 0.96])
outB = SCRATCH / "inspirational_timeseries_B_annotated.pdf"
figB.savefig(outB)
figB.savefig(outB.with_suffix(".png"), dpi=180)
# cutoff variant — hide the paddle-off artists (they'd sit far outside the axis
# and stretch the tight-bbox figure sideways)
for art in _paddle_off_artists:
    art.set_visible(False)
axB_in.set_xlim(0, X_CUTOFF_S)
outB_cut = SCRATCH / "inspirational_timeseries_B_annotated_cut.pdf"
figB.savefig(outB_cut)
figB.savefig(outB_cut.with_suffix(".png"), dpi=180)
plt.close(figB)
print(f"  → {outB.name} + _cut (each with .png)")


# ─── Figure C: macro (full) + micro (zoom) ───────────────────────────────
print("Rendering C …")
figC = plt.figure(figsize=(10, 7.2))
gs = figC.add_gridspec(
    4, 1,
    height_ratios=[2.2, 1, 2.2, 1],
    hspace=0.42,
)
axC_in_full  = figC.add_subplot(gs[0, 0])
axC_in_zoom  = figC.add_subplot(gs[1, 0])
axC_out_full = figC.add_subplot(gs[2, 0], sharex=axC_in_full)
axC_out_zoom = figC.add_subplot(gs[3, 0], sharex=axC_in_zoom)

# Zoom: 5 periods centred on the H&G window
zoom_center = (in_ws + in_we) / 2
zoom_half   = 2.5 / FREQ
zx0, zx1 = zoom_center - zoom_half, zoom_center + zoom_half

for ax_full, ax_zoom, sig, color, probe, label in [
    (axC_in_full,  axC_in_zoom,  eta_in,  IN_COLOR,  IN_PROBE,  "IN"),
    (axC_out_full, axC_out_zoom, eta_out, OUT_COLOR, OUT_PROBE, "OUT"),
]:
    # macro
    ax_full.plot(t, sig, color=color, lw=0.5)
    ax_full.axvspan(zx0, zx1, color=WIN_COLOR, alpha=0.55, zorder=0)
    ax_full.axhline(0, color="#888", lw=0.5, alpha=0.6)
    ax_full.set_xlim(0, t[-1])
    ax_full.set_ylim(ylim)
    ax_full.set_ylabel(r"$\eta$ (mm)")
    ax_full.text(
        0.005, 0.92, f"{label}  ·  {probe}",
        transform=ax_full.transAxes, va="top", ha="left",
        fontsize=10, color=color, weight="bold",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, lw=0.6),
    )

    # micro
    m = (t >= zx0) & (t <= zx1)
    ax_zoom.plot(t[m], sig[m], color=color, lw=1.3)
    ax_zoom.scatter(t[m][::6], sig[m][::6], s=6, color=color, alpha=0.55,
                    edgecolor="none")
    ax_zoom.axhline(0, color="#888", lw=0.5, alpha=0.6)
    ax_zoom.set_xlim(zx0, zx1)
    ax_zoom.set_ylim(ylim)
    ax_zoom.set_ylabel(r"$\eta$ (mm)")
    ax_zoom.text(
        0.005, 0.92, "zoom · 5 periods",
        transform=ax_zoom.transAxes, va="top", ha="left",
        fontsize=9, color="#444",
    )

    # connector lines from macro shaded band → zoom subplot
    for xx in (zx0, zx1):
        con = ConnectionPatch(
            xyA=(xx, ylim[0]), coordsA=ax_full.transData,
            xyB=(xx, ylim[1]), coordsB=ax_zoom.transData,
            color=WIN_COLOR, lw=0.7, alpha=0.9, zorder=0,
        )
        figC.add_artist(con)

axC_out_zoom.set_xlabel("time (s)")
axC_in_zoom.set_xlabel("time (s)")
for ax in (axC_in_full, axC_out_full):
    apply_ticks(ax)
for ax in (axC_in_zoom, axC_out_zoom):
    apply_ticks(ax, x_major=0.5, x_minor=0.1)
figC.suptitle(
    f"Macro and micro · nowind · $f$ = {FREQ} Hz · $A$ = {AMP_V} V",
    fontsize=11,
)
outC = SCRATCH / "inspirational_timeseries_C_zoom.pdf"
figC.savefig(outC, bbox_inches="tight")
figC.savefig(outC.with_suffix(".png"), dpi=180, bbox_inches="tight")
# cutoff variant — shrink macro panels; the zoom axes are untouched
axC_in_full.set_xlim(0, X_CUTOFF_S)
axC_out_full.set_xlim(0, X_CUTOFF_S)
outC_cut = SCRATCH / "inspirational_timeseries_C_zoom_cut.pdf"
figC.savefig(outC_cut, bbox_inches="tight")
figC.savefig(outC_cut.with_suffix(".png"), dpi=180, bbox_inches="tight")
plt.close(figC)
print(f"  → {outC.name} + _cut (each with .png)")

print("Done.")
