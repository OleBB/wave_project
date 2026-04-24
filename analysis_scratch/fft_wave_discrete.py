"""
Exploration: discrete FFT spectrum for a canonical wave run (1.4 Hz, 0.2 V,
fullpanel, per240). Shows the paddle-frequency peak as stems / bars at the
actual FFT bin frequencies — a beginner-friendly view of "the FFT is discrete
bins, not a continuous curve".

Outputs PNG variants to output/fft_wave_discrete_exploration/ for user to
pick from. No PDFs, no TEXFIGU stubs yet — promote the chosen variant later.

Variants:
    V1_2x2_by_condition_stem.png   rows=side (IN/OUT), cols=wind
    V2_overlay_2panels_stem.png    panels=side, stems overlaid per wind
    V3_2x2_by_side_stem.png        rows=wind, cols=side
    V4_2x2_by_condition_bar.png    same layout as V1 but bars instead of stems
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wavescripts.improved_data_loader import load_analysis_data
from wavescripts.plot_utils import apply_thesis_style, WIND_COLOR_MAP
from wavescripts.signal_processing import get_positive_spectrum

FS = 250.0
BASE = Path("/Users/ole/Kodevik/wave_project")
EXPLORE_DIR = BASE / "output" / "fft_wave_discrete_exploration"
EXPLORE_DIR.mkdir(parents=True, exist_ok=True)

TARGET_DIR = BASE / "waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"
DATADIR    = BASE / "wavedata/20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"

RUNS = {
    "nowind":   DATADIR / "fullpanel-nowind-amp0200-freq1400-per240-depth580-mstop30-run1.csv",
    "fullwind": DATADIR / "fullpanel-fullwind-amp0200-freq1400-per240-depth580-mstop30-run1.csv",
}

IN_PROBE  = "9373/170"
OUT_PROBE = "12400/250"
FREQ  = 1.4
AMP_V = 0.2
XMAX  = 5.0

WIND_KEY   = {"nowind": "no", "fullwind": "full"}
WIND_COLOR = {w: WIND_COLOR_MAP[WIND_KEY[w]] for w in ("nowind", "fullwind")}
WIND_LABEL = {"nowind": "uten vind", "fullwind": "med vind"}

apply_thesis_style()


# ─── Load (just the one canon folder) ──────────────────────────────────────
print("Loading …")
meta, _, fft_dict, _ = load_analysis_data(str(TARGET_DIR), load_processed=False)


def get_spectrum(run_csv: Path, probe: str):
    """Return (freq_hz, one_sided_amplitude_mm) in [0, XMAX]."""
    csv = str(run_csv)
    df  = fft_dict[csv]
    df_pos = get_positive_spectrum(df)
    col = f"FFT {probe}"
    y   = df_pos[col].dropna()
    f   = y.index.values
    mask = (f >= 0) & (f <= XMAX)
    # `FFT {pos}` stores |fft|/N (two-sided). Multiply by 2 for the
    # one-sided amplitude, which matches `Probe {pos} Amplitude (FFT)`
    # in combined_meta.
    return f[mask], 2.0 * y.values[mask]


# Collect spectra + diagnostic cross-check against meta
specs = {}
print("\nSpectra (peak vs meta.json):")
for wind, csv in RUNS.items():
    for side, probe in (("IN", IN_PROBE), ("OUT", OUT_PROBE)):
        f, A = get_spectrum(csv, probe)
        specs[(wind, side)] = (f, A)
        peak_i = int(np.argmax(A))
        meta_row = meta[meta["path"] == str(csv)].iloc[0]
        meta_A = float(meta_row[f"Probe {probe} Amplitude (FFT)"])
        print(f"  {wind:8s} {side:3s}  peak={A[peak_i]:.4f} @ {f[peak_i]:.3f} Hz   "
              f"meta_A(FFT)={meta_A:.4f}")

# Shared y per row (by probe side) — IN and OUT differ by ~×10, so one global
# scale would squash OUT. Shared within a row makes nowind vs fullwind
# directly comparable for the same probe.
YMAX_IN  = max(specs[("nowind", "IN")][1].max(),  specs[("fullwind", "IN")][1].max())  * 1.15
YMAX_OUT = max(specs[("nowind", "OUT")][1].max(), specs[("fullwind", "OUT")][1].max()) * 1.15


# ─── Panel drawers ────────────────────────────────────────────────────────
def stem_panel(ax, f, A, color, title, *, ymax=None):
    ml, sl, bl = ax.stem(f, A, linefmt="-", markerfmt="o", basefmt=" ")
    plt.setp(sl, color=color, lw=1.3, alpha=0.9)
    plt.setp(ml, color=color, ms=3.0, mec=color, mfc=color)
    plt.setp(bl, visible=False)
    ax.set_xlim(0, XMAX)
    if ymax is not None:
        ax.set_ylim(0, ymax)
    ax.set_title(title, fontsize=10)
    ax.grid(True, which="major", alpha=0.30)
    ax.axvline(FREQ, color="#888", lw=0.5, ls=":", alpha=0.7, zorder=0)


def bar_panel(ax, f, A, color, title, *, ymax=None):
    # Bin width = actual FFT bin spacing (uniform) minus a tiny gap
    df = f[1] - f[0]
    ax.bar(f, A, width=df * 0.9, color=color, edgecolor=color, lw=0.4,
           alpha=0.85, align="center")
    ax.set_xlim(0, XMAX)
    if ymax is not None:
        ax.set_ylim(0, ymax)
    ax.set_title(title, fontsize=10)
    ax.grid(True, which="major", alpha=0.30)
    ax.axvline(FREQ, color="#888", lw=0.5, ls=":", alpha=0.7, zorder=0)


def overlay_stem_panel(ax, side):
    for wind in ("nowind", "fullwind"):
        f, A = specs[(wind, side)]
        ml, sl, bl = ax.stem(f, A, linefmt="-", markerfmt="o", basefmt=" ",
                             label=WIND_LABEL[wind])
        plt.setp(sl, color=WIND_COLOR[wind], lw=1.2, alpha=0.75)
        plt.setp(ml, color=WIND_COLOR[wind], ms=3.0,
                 mec=WIND_COLOR[wind], mfc=WIND_COLOR[wind])
        plt.setp(bl, visible=False)
    ax.set_xlim(0, XMAX)
    ax.set_title(f"{side}", fontsize=10)
    ax.grid(True, which="major", alpha=0.30)
    ax.axvline(FREQ, color="#888", lw=0.5, ls=":", alpha=0.7, zorder=0)
    ax.legend(fontsize=9, loc="upper right")


# ─── V1: 2×2 by condition — rows=side, cols=wind, sharey per row ─────────
fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
for i, (side, ymax) in enumerate((("IN", YMAX_IN), ("OUT", YMAX_OUT))):
    for j, wind in enumerate(("nowind", "fullwind")):
        f, A = specs[(wind, side)]
        stem_panel(axes[i, j], f, A, WIND_COLOR[wind],
                   f"{side} · {WIND_LABEL[wind]}", ymax=ymax)
        if j == 0:
            axes[i, j].set_ylabel("Amplitude [mm]")
        if i == 1:
            axes[i, j].set_xlabel("Frekvens [Hz]")
fig.suptitle(f"FFT-spektrum · $f = {FREQ}$\u00a0Hz · $V = {AMP_V}$\u00a0V · "
             "fullpanel", fontsize=12)
fig.tight_layout()
fig.savefig(EXPLORE_DIR / "V1_2x2_by_condition_stem.png", dpi=140, bbox_inches="tight")
plt.close(fig)


# ─── V2: 2 panels, stems overlaid per wind, sharey=True ───────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
for ax, side in zip(axes, ("IN", "OUT")):
    overlay_stem_panel(ax, side)
    ax.set_xlabel("Frekvens [Hz]")
axes[0].set_ylabel("Amplitude [mm]")
fig.suptitle(f"FFT-spektrum · $f = {FREQ}$\u00a0Hz · $V = {AMP_V}$\u00a0V · "
             "fullpanel — vindsammenligning", fontsize=12)
fig.tight_layout()
fig.savefig(EXPLORE_DIR / "V2_overlay_2panels_stem.png", dpi=140, bbox_inches="tight")
plt.close(fig)


# ─── V3: 2×2 by side — rows=wind, cols=side, sharey per column ───────────
fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
for i, wind in enumerate(("nowind", "fullwind")):
    for j, (side, ymax) in enumerate((("IN", YMAX_IN), ("OUT", YMAX_OUT))):
        f, A = specs[(wind, side)]
        stem_panel(axes[i, j], f, A, WIND_COLOR[wind],
                   f"{WIND_LABEL[wind]} · {side}", ymax=ymax)
        if j == 0:
            axes[i, j].set_ylabel("Amplitude [mm]")
        if i == 1:
            axes[i, j].set_xlabel("Frekvens [Hz]")
fig.suptitle(f"FFT-spektrum · $f = {FREQ}$\u00a0Hz · $V = {AMP_V}$\u00a0V · "
             "fullpanel", fontsize=12)
fig.tight_layout()
fig.savefig(EXPLORE_DIR / "V3_2x2_by_side_stem.png", dpi=140, bbox_inches="tight")
plt.close(fig)


# ─── V4: same layout as V1 but bars (column plot) ──────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
for i, (side, ymax) in enumerate((("IN", YMAX_IN), ("OUT", YMAX_OUT))):
    for j, wind in enumerate(("nowind", "fullwind")):
        f, A = specs[(wind, side)]
        bar_panel(axes[i, j], f, A, WIND_COLOR[wind],
                  f"{side} · {WIND_LABEL[wind]}", ymax=ymax)
        if j == 0:
            axes[i, j].set_ylabel("Amplitude [mm]")
        if i == 1:
            axes[i, j].set_xlabel("Frekvens [Hz]")
fig.suptitle(f"FFT-spektrum · $f = {FREQ}$\u00a0Hz · $V = {AMP_V}$\u00a0V · "
             "fullpanel — søyler", fontsize=12)
fig.tight_layout()
fig.savefig(EXPLORE_DIR / "V4_2x2_by_condition_bar.png", dpi=140, bbox_inches="tight")
plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# TALLER VARIANTS — physically taller PDFs, same layouts
# ══════════════════════════════════════════════════════════════════════════
# The landscape V1–V4 are short and wide; a tall layout makes the paddle-
# peak stems visually dominant, and fits a \textwidth figure in the thesis
# with room for margin overshoot.
#
# V5  = V1 recipe at (8, 12)          2×2, sides × wind
# V6  = V2 recipe stacked (8, 10)     1 col × 2 rows, overlay per side
# V7  = V3 recipe at (8, 12)          2×2, wind × sides
# V8  = V4 recipe at (8, 12)          2×2 bars, sides × wind
# V9  = V5 recipe even taller (8, 15) for "margin overshoot" feel

# ─── V5: 2×2 by condition, tall ───────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(8, 12), sharex=True)
for i, (side, ymax) in enumerate((("IN", YMAX_IN), ("OUT", YMAX_OUT))):
    for j, wind in enumerate(("nowind", "fullwind")):
        f, A = specs[(wind, side)]
        stem_panel(axes[i, j], f, A, WIND_COLOR[wind],
                   f"{side} · {WIND_LABEL[wind]}", ymax=ymax)
        if j == 0:
            axes[i, j].set_ylabel("Amplitude [mm]")
        if i == 1:
            axes[i, j].set_xlabel("Frekvens [Hz]")
fig.suptitle(f"FFT-spektrum · $f = {FREQ}$\u00a0Hz · $V = {AMP_V}$\u00a0V · "
             "fullpanel", fontsize=12)
fig.tight_layout()
fig.savefig(EXPLORE_DIR / "V5_2x2_by_condition_stem_tall.png", dpi=140, bbox_inches="tight")
plt.close(fig)


# ─── V6: overlay stacked (1 col, 2 rows), tall ────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
for ax, side in zip(axes, ("IN", "OUT")):
    overlay_stem_panel(ax, side)
    ax.set_ylabel("Amplitude [mm]")
axes[-1].set_xlabel("Frekvens [Hz]")
fig.suptitle(f"FFT-spektrum · $f = {FREQ}$\u00a0Hz · $V = {AMP_V}$\u00a0V · "
             "fullpanel — vindsammenligning", fontsize=12)
fig.tight_layout()
fig.savefig(EXPLORE_DIR / "V6_overlay_stacked_stem_tall.png", dpi=140, bbox_inches="tight")
plt.close(fig)


# ─── V7: 2×2 by side, tall ────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(8, 12), sharex=True)
for i, wind in enumerate(("nowind", "fullwind")):
    for j, (side, ymax) in enumerate((("IN", YMAX_IN), ("OUT", YMAX_OUT))):
        f, A = specs[(wind, side)]
        stem_panel(axes[i, j], f, A, WIND_COLOR[wind],
                   f"{WIND_LABEL[wind]} · {side}", ymax=ymax)
        if j == 0:
            axes[i, j].set_ylabel("Amplitude [mm]")
        if i == 1:
            axes[i, j].set_xlabel("Frekvens [Hz]")
fig.suptitle(f"FFT-spektrum · $f = {FREQ}$\u00a0Hz · $V = {AMP_V}$\u00a0V · "
             "fullpanel", fontsize=12)
fig.tight_layout()
fig.savefig(EXPLORE_DIR / "V7_2x2_by_side_stem_tall.png", dpi=140, bbox_inches="tight")
plt.close(fig)


# ─── V8: 2×2 bars, tall ───────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(8, 12), sharex=True)
for i, (side, ymax) in enumerate((("IN", YMAX_IN), ("OUT", YMAX_OUT))):
    for j, wind in enumerate(("nowind", "fullwind")):
        f, A = specs[(wind, side)]
        bar_panel(axes[i, j], f, A, WIND_COLOR[wind],
                  f"{side} · {WIND_LABEL[wind]}", ymax=ymax)
        if j == 0:
            axes[i, j].set_ylabel("Amplitude [mm]")
        if i == 1:
            axes[i, j].set_xlabel("Frekvens [Hz]")
fig.suptitle(f"FFT-spektrum · $f = {FREQ}$\u00a0Hz · $V = {AMP_V}$\u00a0V · "
             "fullpanel — søyler", fontsize=12)
fig.tight_layout()
fig.savefig(EXPLORE_DIR / "V8_2x2_by_condition_bar_tall.png", dpi=140, bbox_inches="tight")
plt.close(fig)


# ─── V9: V5 recipe, extra tall — "margin overshoot OK" ────────────────────
fig, axes = plt.subplots(2, 2, figsize=(8, 15), sharex=True)
for i, (side, ymax) in enumerate((("IN", YMAX_IN), ("OUT", YMAX_OUT))):
    for j, wind in enumerate(("nowind", "fullwind")):
        f, A = specs[(wind, side)]
        stem_panel(axes[i, j], f, A, WIND_COLOR[wind],
                   f"{side} · {WIND_LABEL[wind]}", ymax=ymax)
        if j == 0:
            axes[i, j].set_ylabel("Amplitude [mm]")
        if i == 1:
            axes[i, j].set_xlabel("Frekvens [Hz]")
fig.suptitle(f"FFT-spektrum · $f = {FREQ}$\u00a0Hz · $V = {AMP_V}$\u00a0V · "
             "fullpanel", fontsize=12)
fig.tight_layout()
fig.savefig(EXPLORE_DIR / "V9_2x2_by_condition_stem_extratall.png", dpi=140, bbox_inches="tight")
plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# V10 / V11 — unified y-axis across all 4 panels, bar/column plots
# ══════════════════════════════════════════════════════════════════════════
# All four panels share one y-axis (tiny padding above the global tallest
# peak) so the paddle peak's dominance is visible without wasting whitespace
# on whatever the row scale would have demanded.

YMAX_ALL = max(YMAX_IN, YMAX_OUT)  # already had 1.15 padding baked in
# Shrink the padding: just enough breathing room over the tallest peak.
_peak_all = max(specs[(w, s)][1].max() for w in ("nowind", "fullwind") for s in ("IN", "OUT"))
YMAX_ALL = _peak_all * 1.04

# ─── V10: rows=side, cols=wind. IN|IN / OUT|OUT, unified y ────────────────
fig, axes = plt.subplots(2, 2, figsize=(8, 12), sharex=True, sharey=True)
for i, side in enumerate(("IN", "OUT")):
    for j, wind in enumerate(("nowind", "fullwind")):
        f, A = specs[(wind, side)]
        bar_panel(axes[i, j], f, A, WIND_COLOR[wind],
                  f"{side} · {WIND_LABEL[wind]}", ymax=YMAX_ALL)
        if j == 0:
            axes[i, j].set_ylabel("Amplitude [mm]")
        if i == 1:
            axes[i, j].set_xlabel("Frekvens [Hz]")
fig.suptitle(f"FFT-spektrum · $f = {FREQ}$\u00a0Hz · $V = {AMP_V}$\u00a0V · "
             "fullpanel", fontsize=12)
fig.tight_layout()
fig.savefig(EXPLORE_DIR / "V10_bar_sides_rows_unified_y.png", dpi=140, bbox_inches="tight")
plt.close(fig)


# ─── V11: rows=wind, cols=side. IN|OUT / IN|OUT, unified y ────────────────
fig, axes = plt.subplots(2, 2, figsize=(8, 12), sharex=True, sharey=True)
for i, wind in enumerate(("nowind", "fullwind")):
    for j, side in enumerate(("IN", "OUT")):
        f, A = specs[(wind, side)]
        bar_panel(axes[i, j], f, A, WIND_COLOR[wind],
                  f"{WIND_LABEL[wind]} · {side}", ymax=YMAX_ALL)
        if j == 0:
            axes[i, j].set_ylabel("Amplitude [mm]")
        if i == 1:
            axes[i, j].set_xlabel("Frekvens [Hz]")
fig.suptitle(f"FFT-spektrum · $f = {FREQ}$\u00a0Hz · $V = {AMP_V}$\u00a0V · "
             "fullpanel", fontsize=12)
fig.tight_layout()
fig.savefig(EXPLORE_DIR / "V11_bar_winds_rows_unified_y.png", dpi=140, bbox_inches="tight")
plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# V12 / V13 — V10/V11 recipe + reference horizontal lines + harmonic markers
# ══════════════════════════════════════════════════════════════════════════
# Extensions over V10/V11:
#   - Horizontal dotted lines at the tallest IN and tallest OUT paddle peaks
#     (visible across all 4 panels thanks to sharey=True)
#   - Vertical dotted markers at f, 2f, 3f, 4f (= 1.4, 2.8, 4.2, 5.6 Hz)
#   - xlim extended to (0, 6) so 4f is in view
#   - Denser x-ticks: major every 1 Hz, minor every 0.2 Hz

from matplotlib.ticker import MultipleLocator

XMAX_HARMONIC = 6.0
HARMONICS = [(k, k * FREQ) for k in (1, 2, 3, 4)]
HARMONIC_LABELS = {1: r"$f$", 2: r"$2f$", 3: r"$3f$", 4: r"$4f$"}

# Tallest paddle peak per probe side (across both wind conditions)
IN_MAX_A  = max(specs[("nowind",  "IN")][1].max(),  specs[("fullwind", "IN")][1].max())
OUT_MAX_A = max(specs[("nowind", "OUT")][1].max(),  specs[("fullwind", "OUT")][1].max())
print(f"\nReference peaks: IN_max={IN_MAX_A:.3f} mm  OUT_max={OUT_MAX_A:.3f} mm")

REF_LINES = [
    (IN_MAX_A,  "#222", "IN"),   # tallest IN  peak (fullwind IN)
    (OUT_MAX_A, "#888", "OUT"),  # tallest OUT peak (fullwind OUT)
]


def bar_panel_v2(ax, f, A, color, title, *, ymax, show_delta=False):
    """Bar plot + harmonic vertical markers + IN/OUT reference horizontal lines.

    show_delta=True draws a two-headed arrow between the IN and OUT reference
    lines with Δ (mm) and the OUT/IN ratio annotated. Set on one panel only
    to avoid cluttering the grid.
    """
    df = f[1] - f[0]
    ax.bar(f, A, width=df * 0.9, color=color, edgecolor=color, lw=0.4,
           alpha=0.85, align="center", zorder=3)

    # Horizontal reference lines at tallest IN / tallest OUT peaks
    for y_ref, c_ref, lbl in REF_LINES:
        ax.axhline(y_ref, color=c_ref, lw=0.7, ls=":", alpha=0.8, zorder=2)
        ax.text(XMAX_HARMONIC * 0.985, y_ref, f" {lbl} peak",
                color=c_ref, fontsize=8, va="bottom", ha="right", zorder=4,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))

    # Harmonic vertical markers
    for k, fk in HARMONICS:
        if fk <= XMAX_HARMONIC:
            ax.axvline(fk, color="#888", lw=0.5, ls=":", alpha=0.7, zorder=1)
            ax.text(fk, ymax * 0.97, HARMONIC_LABELS[k],
                    color="#444", fontsize=8, ha="center", va="top",
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.75))

    # Optional Δ annotation between the two reference lines
    if show_delta:
        delta_mm = IN_MAX_A - OUT_MAX_A
        ratio    = OUT_MAX_A / IN_MAX_A
        x_arr = 5.6  # between 4f marker and right edge
        ax.annotate("", xy=(x_arr, IN_MAX_A), xytext=(x_arr, OUT_MAX_A),
                    arrowprops=dict(arrowstyle="<->", color="#222", lw=1.1,
                                    shrinkA=0, shrinkB=0),
                    zorder=5)
        mid_y = (IN_MAX_A + OUT_MAX_A) / 2
        ax.text(x_arr - 0.12, mid_y,
                rf"$\Delta = {delta_mm:.2f}$" + "\u00a0mm\n"
                rf"$A_\mathrm{{OUT}}/A_\mathrm{{IN}} = {ratio:.2f}$",
                ha="right", va="center", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec="#444", lw=0.6, alpha=0.95),
                zorder=6)

    ax.set_xlim(0, XMAX_HARMONIC)
    ax.set_ylim(0, ymax)
    ax.set_title(title, fontsize=10)
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(MultipleLocator(0.2))
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(True, which="minor", alpha=0.12)


# ─── V12: V10 layout (IN|IN / OUT|OUT) + refs + harmonics ─────────────────
fig, axes = plt.subplots(2, 2, figsize=(8, 12), sharex=True, sharey=True)
for i, side in enumerate(("IN", "OUT")):
    for j, wind in enumerate(("nowind", "fullwind")):
        f, A = specs[(wind, side)]
        bar_panel_v2(axes[i, j], f, A, WIND_COLOR[wind],
                     f"{side} · {WIND_LABEL[wind]}", ymax=YMAX_ALL)
        if j == 0:
            axes[i, j].set_ylabel("Amplitude [mm]")
        if i == 1:
            axes[i, j].set_xlabel("Frekvens [Hz]")
fig.suptitle(f"FFT-spektrum · $f = {FREQ}$\u00a0Hz · $V = {AMP_V}$\u00a0V · "
             "fullpanel", fontsize=12)
fig.tight_layout()
fig.savefig(EXPLORE_DIR / "V12_bar_sides_rows_refs_harmonics.png", dpi=140, bbox_inches="tight")
plt.close(fig)


# ─── V13: V11 layout (IN|OUT / IN|OUT) + refs + harmonics ─────────────────
fig, axes = plt.subplots(2, 2, figsize=(8, 12), sharex=True, sharey=True)
for i, wind in enumerate(("nowind", "fullwind")):
    for j, side in enumerate(("IN", "OUT")):
        f, A = specs[(wind, side)]
        bar_panel_v2(axes[i, j], f, A, WIND_COLOR[wind],
                     f"{WIND_LABEL[wind]} · {side}", ymax=YMAX_ALL)
        if j == 0:
            axes[i, j].set_ylabel("Amplitude [mm]")
        if i == 1:
            axes[i, j].set_xlabel("Frekvens [Hz]")
fig.suptitle(f"FFT-spektrum · $f = {FREQ}$\u00a0Hz · $V = {AMP_V}$\u00a0V · "
             "fullpanel", fontsize=12)
fig.tight_layout()
fig.savefig(EXPLORE_DIR / "V13_bar_winds_rows_refs_harmonics.png", dpi=140, bbox_inches="tight")
plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# V14 / V15 — V12/V13 + Δ annotation between IN and OUT peak reference lines
# ══════════════════════════════════════════════════════════════════════════
# Same as V12/V13 but the top-right panel carries a two-headed arrow between
# the horizontal reference lines, labeled with:
#     Δ = 4.34 mm          (absolute difference)
#     A_OUT / A_IN = 0.73  (thesis OUT/IN ratio, the central metric)

# ─── V14: V12 layout (IN|IN / OUT|OUT) + Δ on top-right panel ────────────
fig, axes = plt.subplots(2, 2, figsize=(8, 12), sharex=True, sharey=True)
for i, side in enumerate(("IN", "OUT")):
    for j, wind in enumerate(("nowind", "fullwind")):
        f, A = specs[(wind, side)]
        bar_panel_v2(axes[i, j], f, A, WIND_COLOR[wind],
                     f"{side} · {WIND_LABEL[wind]}", ymax=YMAX_ALL,
                     show_delta=(i == 0 and j == 1))  # fullwind · IN
        if j == 0:
            axes[i, j].set_ylabel("Amplitude [mm]")
        if i == 1:
            axes[i, j].set_xlabel("Frekvens [Hz]")
fig.suptitle(f"FFT-spektrum · $f = {FREQ}$\u00a0Hz · $V = {AMP_V}$\u00a0V · "
             "fullpanel", fontsize=12)
fig.tight_layout()
fig.savefig(EXPLORE_DIR / "V14_bar_sides_rows_refs_delta.png", dpi=140, bbox_inches="tight")
plt.close(fig)


# ─── V15: V13 layout (IN|OUT / IN|OUT) + Δ on bottom-left panel ──────────
fig, axes = plt.subplots(2, 2, figsize=(8, 12), sharex=True, sharey=True)
for i, wind in enumerate(("nowind", "fullwind")):
    for j, side in enumerate(("IN", "OUT")):
        f, A = specs[(wind, side)]
        bar_panel_v2(axes[i, j], f, A, WIND_COLOR[wind],
                     f"{WIND_LABEL[wind]} · {side}", ymax=YMAX_ALL,
                     show_delta=(i == 1 and j == 0))  # fullwind · IN
        if j == 0:
            axes[i, j].set_ylabel("Amplitude [mm]")
        if i == 1:
            axes[i, j].set_xlabel("Frekvens [Hz]")
fig.suptitle(f"FFT-spektrum · $f = {FREQ}$\u00a0Hz · $V = {AMP_V}$\u00a0V · "
             "fullpanel", fontsize=12)
fig.tight_layout()
fig.savefig(EXPLORE_DIR / "V15_bar_winds_rows_refs_delta.png", dpi=140, bbox_inches="tight")
plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# V16 — winner track: V15 layout, no title, harmonic labels on TOP x-axis,
#                     Δ marker shown on BOTH fullwind panels
# ══════════════════════════════════════════════════════════════════════════
# Refinements over V15:
#   - No suptitle (context belongs in \caption{} / IMMUTABLE stub block)
#   - f, 2f, 3f, 4f labels move from inside each panel (blocked the paddle
#     peak) to a secondary x-axis on top of the top-row panels
#   - Vertical dotted markers at those same harmonics remain in every panel
#   - Δ annotation now on both fullwind panels (fullwind IN + fullwind OUT)

def bar_panel_v3(ax, f, A, color, title, *, ymax, show_delta=False):
    """Bar plot + harmonic vertical markers + IN/OUT reference horizontal lines.

    Harmonic labels are NOT placed inside the panel — see secondary_xaxis
    setup outside this helper for the top-axis labeling.
    """
    df = f[1] - f[0]
    ax.bar(f, A, width=df * 0.9, color=color, edgecolor=color, lw=0.4,
           alpha=0.85, align="center", zorder=3)

    for y_ref, c_ref, lbl in REF_LINES:
        ax.axhline(y_ref, color=c_ref, lw=0.7, ls=":", alpha=0.8, zorder=2)
        ax.text(XMAX_HARMONIC * 0.985, y_ref, f" {lbl} peak",
                color=c_ref, fontsize=8, va="bottom", ha="right", zorder=4,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))

    # Vertical dotted markers at f, 2f, 3f, 4f — labels live on the top axis
    for k, fk in HARMONICS:
        if fk <= XMAX_HARMONIC:
            ax.axvline(fk, color="#888", lw=0.5, ls=":", alpha=0.7, zorder=1)

    if show_delta:
        delta_mm = IN_MAX_A - OUT_MAX_A
        ratio    = OUT_MAX_A / IN_MAX_A
        x_arr = 5.6
        ax.annotate("", xy=(x_arr, IN_MAX_A), xytext=(x_arr, OUT_MAX_A),
                    arrowprops=dict(arrowstyle="<->", color="#222", lw=1.1,
                                    shrinkA=0, shrinkB=0),
                    zorder=5)
        mid_y = (IN_MAX_A + OUT_MAX_A) / 2
        ax.text(x_arr - 0.12, mid_y,
                rf"$\Delta = {delta_mm:.2f}$" + "\u00a0mm\n"
                rf"$A_\mathrm{{OUT}}/A_\mathrm{{IN}} = {ratio:.2f}$",
                ha="right", va="center", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec="#444", lw=0.6, alpha=0.95),
                zorder=6)

    ax.set_xlim(0, XMAX_HARMONIC)
    ax.set_ylim(0, ymax)
    ax.set_title(title, fontsize=10)
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(MultipleLocator(0.2))
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(True, which="minor", alpha=0.12)


fig, axes = plt.subplots(2, 2, figsize=(8, 12), sharex=True, sharey=True)
for i, wind in enumerate(("nowind", "fullwind")):
    for j, side in enumerate(("IN", "OUT")):
        f, A = specs[(wind, side)]
        bar_panel_v3(axes[i, j], f, A, WIND_COLOR[wind],
                     f"{WIND_LABEL[wind]} · {side}", ymax=YMAX_ALL,
                     show_delta=(i == 1))  # both fullwind panels
        if j == 0:
            axes[i, j].set_ylabel("Amplitude [mm]")
        if i == 1:
            axes[i, j].set_xlabel("Frekvens [Hz]")

# Harmonic labels on the top x-axis of the top-row panels only
_harmonic_ticks  = [fk for _, fk in HARMONICS if fk <= XMAX_HARMONIC]
_harmonic_ticklab = [HARMONIC_LABELS[k] for k, fk in HARMONICS if fk <= XMAX_HARMONIC]
for ax_top in axes[0, :]:
    sec = ax_top.secondary_xaxis("top")
    sec.set_xticks(_harmonic_ticks)
    sec.set_xticklabels(_harmonic_ticklab)
    sec.tick_params(axis="x", which="major", labelsize=9, pad=2,
                    length=4, color="#444", labelcolor="#222")

fig.tight_layout()
fig.savefig(EXPLORE_DIR / "V16_bar_winds_rows_top_harmonics_delta2.png",
            dpi=140, bbox_inches="tight")
plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# V17 — same as V16 with Norwegian panel titles
#     IN  → "Innkommende, {med/uten} vind"
#     OUT → "Utgående,   {med/uten} vind"
# ══════════════════════════════════════════════════════════════════════════
SIDE_LABEL_NO = {"IN": "Innkommende", "OUT": "Utgående"}

fig, axes = plt.subplots(2, 2, figsize=(8, 12), sharex=True, sharey=True)
for i, wind in enumerate(("nowind", "fullwind")):
    for j, side in enumerate(("IN", "OUT")):
        f, A = specs[(wind, side)]
        bar_panel_v3(axes[i, j], f, A, WIND_COLOR[wind],
                     f"{SIDE_LABEL_NO[side]}, {WIND_LABEL[wind]}",
                     ymax=YMAX_ALL,
                     show_delta=(i == 1))
        if j == 0:
            axes[i, j].set_ylabel("Amplitude [mm]")
        if i == 1:
            axes[i, j].set_xlabel("Frekvens [Hz]")

for ax_top in axes[0, :]:
    sec = ax_top.secondary_xaxis("top")
    sec.set_xticks(_harmonic_ticks)
    sec.set_xticklabels(_harmonic_ticklab)
    sec.tick_params(axis="x", which="major", labelsize=9, pad=2,
                    length=4, color="#444", labelcolor="#222")

fig.tight_layout()
fig.savefig(EXPLORE_DIR / "V17_bar_winds_rows_norwegian_titles.png",
            dpi=140, bbox_inches="tight")
plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# V18 — V17 refinements:
#   - Δ box on BOTH Utgående (OUT) panels, for both wind conditions
#   - Ratio uses Norwegian notation: A_{Ut} / A_{inn}
#   - Reference-line "IN peak" / "OUT peak" labels removed
# ══════════════════════════════════════════════════════════════════════════

def bar_panel_v4(ax, f, A, color, title, *, ymax, show_delta=False):
    """V3 recipe without reference-line labels; Norwegian Δ notation."""
    df = f[1] - f[0]
    ax.bar(f, A, width=df * 0.9, color=color, edgecolor=color, lw=0.4,
           alpha=0.85, align="center", zorder=3)

    # Horizontal reference lines (no text labels on the right edge)
    for y_ref, c_ref, _ in REF_LINES:
        ax.axhline(y_ref, color=c_ref, lw=0.7, ls=":", alpha=0.8, zorder=2)

    for k, fk in HARMONICS:
        if fk <= XMAX_HARMONIC:
            ax.axvline(fk, color="#888", lw=0.5, ls=":", alpha=0.7, zorder=1)

    if show_delta:
        delta_mm = IN_MAX_A - OUT_MAX_A
        ratio    = OUT_MAX_A / IN_MAX_A
        x_arr = 5.6
        ax.annotate("", xy=(x_arr, IN_MAX_A), xytext=(x_arr, OUT_MAX_A),
                    arrowprops=dict(arrowstyle="<->", color="#222", lw=1.1,
                                    shrinkA=0, shrinkB=0),
                    zorder=5)
        mid_y = (IN_MAX_A + OUT_MAX_A) / 2
        ax.text(x_arr - 0.12, mid_y,
                rf"$\Delta = {delta_mm:.2f}$" + "\u00a0mm\n"
                rf"$A_\mathrm{{Ut}}/A_\mathrm{{inn}} = {ratio:.2f}$",
                ha="right", va="center", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec="#444", lw=0.6, alpha=0.95),
                zorder=6)

    ax.set_xlim(0, XMAX_HARMONIC)
    ax.set_ylim(0, ymax)
    ax.set_title(title, fontsize=10)
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(MultipleLocator(0.2))
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(True, which="minor", alpha=0.12)


fig, axes = plt.subplots(2, 2, figsize=(8, 12), sharex=True, sharey=True)
for i, wind in enumerate(("nowind", "fullwind")):
    for j, side in enumerate(("IN", "OUT")):
        f, A = specs[(wind, side)]
        bar_panel_v4(axes[i, j], f, A, WIND_COLOR[wind],
                     f"{SIDE_LABEL_NO[side]}, {WIND_LABEL[wind]}",
                     ymax=YMAX_ALL,
                     show_delta=(j == 1))  # both Utgående (OUT) panels
        if j == 0:
            axes[i, j].set_ylabel("Amplitude [mm]")
        if i == 1:
            axes[i, j].set_xlabel("Frekvens [Hz]")

for ax_top in axes[0, :]:
    sec = ax_top.secondary_xaxis("top")
    sec.set_xticks(_harmonic_ticks)
    sec.set_xticklabels(_harmonic_ticklab)
    sec.tick_params(axis="x", which="major", labelsize=9, pad=2,
                    length=4, color="#444", labelcolor="#222")

fig.tight_layout()
fig.savefig(EXPLORE_DIR / "V18_bar_deltaOUT_norwegian_ratio.png",
            dpi=140, bbox_inches="tight")
plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# V19 — V18 + numeric value label on top of each panel's paddle peak
# ══════════════════════════════════════════════════════════════════════════

def bar_panel_v5(ax, f, A, color, title, *, ymax, show_delta=False):
    """V4 recipe + peak-value label on top of each panel's tallest bar."""
    df = f[1] - f[0]
    ax.bar(f, A, width=df * 0.9, color=color, edgecolor=color, lw=0.4,
           alpha=0.85, align="center", zorder=3)

    for y_ref, c_ref, _ in REF_LINES:
        ax.axhline(y_ref, color=c_ref, lw=0.7, ls=":", alpha=0.8, zorder=2)

    for k, fk in HARMONICS:
        if fk <= XMAX_HARMONIC:
            ax.axvline(fk, color="#888", lw=0.5, ls=":", alpha=0.7, zorder=1)

    # Value label directly above the paddle peak (tallest bar)
    peak_i = int(np.argmax(A))
    f_peak, A_peak = f[peak_i], A[peak_i]
    ax.text(f_peak, A_peak + ymax * 0.012,
            f"{A_peak:.2f}\u00a0mm",
            ha="center", va="bottom", fontsize=9, color=color, weight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color,
                      lw=0.6, alpha=0.95),
            zorder=7)

    if show_delta:
        delta_mm = IN_MAX_A - OUT_MAX_A
        ratio    = OUT_MAX_A / IN_MAX_A
        x_arr = 5.6
        ax.annotate("", xy=(x_arr, IN_MAX_A), xytext=(x_arr, OUT_MAX_A),
                    arrowprops=dict(arrowstyle="<->", color="#222", lw=1.1,
                                    shrinkA=0, shrinkB=0),
                    zorder=5)
        mid_y = (IN_MAX_A + OUT_MAX_A) / 2
        ax.text(x_arr - 0.12, mid_y,
                rf"$\Delta = {delta_mm:.2f}$" + "\u00a0mm\n"
                rf"$A_\mathrm{{Ut}}/A_\mathrm{{inn}} = {ratio:.2f}$",
                ha="right", va="center", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec="#444", lw=0.6, alpha=0.95),
                zorder=6)

    ax.set_xlim(0, XMAX_HARMONIC)
    ax.set_ylim(0, ymax)
    ax.set_title(title, fontsize=10)
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(MultipleLocator(0.2))
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(True, which="minor", alpha=0.12)


# Slightly more head-room so the peak-value labels don't clip into the top
# axis' harmonic ticks on the top row.
YMAX_V19 = _peak_all * 1.10

fig, axes = plt.subplots(2, 2, figsize=(8, 12), sharex=True, sharey=True)
for i, wind in enumerate(("nowind", "fullwind")):
    for j, side in enumerate(("IN", "OUT")):
        f, A = specs[(wind, side)]
        bar_panel_v5(axes[i, j], f, A, WIND_COLOR[wind],
                     f"{SIDE_LABEL_NO[side]}, {WIND_LABEL[wind]}",
                     ymax=YMAX_V19,
                     show_delta=(j == 1))
        if j == 0:
            axes[i, j].set_ylabel("Amplitude [mm]")
        if i == 1:
            axes[i, j].set_xlabel("Frekvens [Hz]")

for ax_top in axes[0, :]:
    sec = ax_top.secondary_xaxis("top")
    sec.set_xticks(_harmonic_ticks)
    sec.set_xticklabels(_harmonic_ticklab)
    sec.tick_params(axis="x", which="major", labelsize=9, pad=2,
                    length=4, color="#444", labelcolor="#222")

fig.tight_layout()
fig.savefig(EXPLORE_DIR / "V19_bar_with_peak_values.png",
            dpi=140, bbox_inches="tight")
plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# V20 — V19 refinements:
#   - Drop the extra y-headroom (back to tight YMAX_ALL)
#   - Peak-value label placed to the RIGHT of the paddle peak, not on top
#   - Black text, arrow-shaped (larrow) box pointing back at the peak
# ══════════════════════════════════════════════════════════════════════════

def bar_panel_v6(ax, f, A, color, title, *, ymax, show_delta=False):
    """V4 recipe + side-placed peak-value label in a left-arrow box."""
    df = f[1] - f[0]
    ax.bar(f, A, width=df * 0.9, color=color, edgecolor=color, lw=0.4,
           alpha=0.85, align="center", zorder=3)

    for y_ref, c_ref, _ in REF_LINES:
        ax.axhline(y_ref, color=c_ref, lw=0.7, ls=":", alpha=0.8, zorder=2)

    for k, fk in HARMONICS:
        if fk <= XMAX_HARMONIC:
            ax.axvline(fk, color="#888", lw=0.5, ls=":", alpha=0.7, zorder=1)

    # Peak-value label — to the right of the peak bar, black text,
    # larrow box pointing back toward the peak
    peak_i = int(np.argmax(A))
    f_peak, A_peak = f[peak_i], A[peak_i]
    ax.text(f_peak + 0.18, A_peak,
            f"{A_peak:.2f}\u00a0mm",
            ha="left", va="center", fontsize=9, color="black",
            bbox=dict(boxstyle="larrow,pad=0.22", fc="white", ec="#333",
                      lw=0.7, alpha=0.97),
            zorder=7)

    if show_delta:
        delta_mm = IN_MAX_A - OUT_MAX_A
        ratio    = OUT_MAX_A / IN_MAX_A
        x_arr = 5.6
        ax.annotate("", xy=(x_arr, IN_MAX_A), xytext=(x_arr, OUT_MAX_A),
                    arrowprops=dict(arrowstyle="<->", color="#222", lw=1.1,
                                    shrinkA=0, shrinkB=0),
                    zorder=5)
        mid_y = (IN_MAX_A + OUT_MAX_A) / 2
        ax.text(x_arr - 0.12, mid_y,
                rf"$\Delta = {delta_mm:.2f}$" + "\u00a0mm\n"
                rf"$A_\mathrm{{Ut}}/A_\mathrm{{inn}} = {ratio:.2f}$",
                ha="right", va="center", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec="#444", lw=0.6, alpha=0.95),
                zorder=6)

    ax.set_xlim(0, XMAX_HARMONIC)
    ax.set_ylim(0, ymax)
    ax.set_title(title, fontsize=10)
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(MultipleLocator(0.2))
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(True, which="minor", alpha=0.12)


fig, axes = plt.subplots(2, 2, figsize=(8, 12), sharex=True, sharey=True)
for i, wind in enumerate(("nowind", "fullwind")):
    for j, side in enumerate(("IN", "OUT")):
        f, A = specs[(wind, side)]
        bar_panel_v6(axes[i, j], f, A, WIND_COLOR[wind],
                     f"{SIDE_LABEL_NO[side]}, {WIND_LABEL[wind]}",
                     ymax=YMAX_ALL,                    # tight headroom
                     show_delta=(j == 1))
        if j == 0:
            axes[i, j].set_ylabel("Amplitude [mm]")
        if i == 1:
            axes[i, j].set_xlabel("Frekvens [Hz]")

for ax_top in axes[0, :]:
    sec = ax_top.secondary_xaxis("top")
    sec.set_xticks(_harmonic_ticks)
    sec.set_xticklabels(_harmonic_ticklab)
    sec.tick_params(axis="x", which="major", labelsize=9, pad=2,
                    length=4, color="#444", labelcolor="#222")

fig.tight_layout()
fig.savefig(EXPLORE_DIR / "V20_bar_peak_side_larrow.png",
            dpi=140, bbox_inches="tight")
plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# V21 — V20 refinements:
#   - Plain square box for peak-value label (black text)
#   - Label placed further down the bar (65 % of peak height)
#   - Headroom reduced to ×1.015 of the tallest peak
#   - Nowind (top) row gets its own x-axis tick labels
# ══════════════════════════════════════════════════════════════════════════

def bar_panel_v7(ax, f, A, color, title, *, ymax, show_delta=False):
    """Plain-box peak-value label, placed lower along the peak bar."""
    df = f[1] - f[0]
    ax.bar(f, A, width=df * 0.9, color=color, edgecolor=color, lw=0.4,
           alpha=0.85, align="center", zorder=3)

    for y_ref, c_ref, _ in REF_LINES:
        ax.axhline(y_ref, color=c_ref, lw=0.7, ls=":", alpha=0.8, zorder=2)

    for k, fk in HARMONICS:
        if fk <= XMAX_HARMONIC:
            ax.axvline(fk, color="#888", lw=0.5, ls=":", alpha=0.7, zorder=1)

    peak_i = int(np.argmax(A))
    f_peak, A_peak = f[peak_i], A[peak_i]
    # Placed lower on the peak (65 %) — obvious what it refers to even
    # without any pointer shape.
    ax.text(f_peak + 0.18, A_peak * 0.65,
            f"{A_peak:.2f}\u00a0mm",
            ha="left", va="center", fontsize=9, color="black",
            bbox=dict(boxstyle="square,pad=0.22", fc="white", ec="#888",
                      lw=0.5, alpha=0.97),
            zorder=7)

    if show_delta:
        delta_mm = IN_MAX_A - OUT_MAX_A
        ratio    = OUT_MAX_A / IN_MAX_A
        x_arr = 5.6
        ax.annotate("", xy=(x_arr, IN_MAX_A), xytext=(x_arr, OUT_MAX_A),
                    arrowprops=dict(arrowstyle="<->", color="#222", lw=1.1,
                                    shrinkA=0, shrinkB=0),
                    zorder=5)
        mid_y = (IN_MAX_A + OUT_MAX_A) / 2
        ax.text(x_arr - 0.12, mid_y,
                rf"$\Delta = {delta_mm:.2f}$" + "\u00a0mm\n"
                rf"$A_\mathrm{{Ut}}/A_\mathrm{{inn}} = {ratio:.2f}$",
                ha="right", va="center", fontsize=8.5,
                bbox=dict(boxstyle="square,pad=0.25", fc="white",
                          ec="#444", lw=0.6, alpha=0.95),
                zorder=6)

    ax.set_xlim(0, XMAX_HARMONIC)
    ax.set_ylim(0, ymax)
    ax.set_title(title, fontsize=10)
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(MultipleLocator(0.2))
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(True, which="minor", alpha=0.12)


YMAX_V21 = _peak_all * 1.015

fig, axes = plt.subplots(2, 2, figsize=(8, 12), sharex=True, sharey=True)
for i, wind in enumerate(("nowind", "fullwind")):
    for j, side in enumerate(("IN", "OUT")):
        f, A = specs[(wind, side)]
        bar_panel_v7(axes[i, j], f, A, WIND_COLOR[wind],
                     f"{SIDE_LABEL_NO[side]}, {WIND_LABEL[wind]}",
                     ymax=YMAX_V21,
                     show_delta=(j == 1))
        if j == 0:
            axes[i, j].set_ylabel("Amplitude [mm]")
        # Bottom x-axis label on BOTH rows (nowind row gets its own too)
        axes[i, j].set_xlabel("Frekvens [Hz]")
        axes[i, j].tick_params(axis="x", labelbottom=True)

# Harmonic labels on the very top (only above nowind row)
for ax_top in axes[0, :]:
    sec = ax_top.secondary_xaxis("top")
    sec.set_xticks(_harmonic_ticks)
    sec.set_xticklabels(_harmonic_ticklab)
    sec.tick_params(axis="x", which="major", labelsize=9, pad=2,
                    length=4, color="#444", labelcolor="#222")

fig.tight_layout()
fig.savefig(EXPLORE_DIR / "V21_bar_plainbox_lowerheadroom_nowindticks.png",
            dpi=140, bbox_inches="tight")
plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# V22 — V21 with the peak-value label moved closer to the peak (~88 %)
# ══════════════════════════════════════════════════════════════════════════

def bar_panel_v8(ax, f, A, color, title, *, ymax, show_delta=False,
                 label_frac: float = 0.88):
    df = f[1] - f[0]
    ax.bar(f, A, width=df * 0.9, color=color, edgecolor=color, lw=0.4,
           alpha=0.85, align="center", zorder=3)

    for y_ref, c_ref, _ in REF_LINES:
        ax.axhline(y_ref, color=c_ref, lw=0.7, ls=":", alpha=0.8, zorder=2)

    for k, fk in HARMONICS:
        if fk <= XMAX_HARMONIC:
            ax.axvline(fk, color="#888", lw=0.5, ls=":", alpha=0.7, zorder=1)

    peak_i = int(np.argmax(A))
    f_peak, A_peak = f[peak_i], A[peak_i]
    ax.text(f_peak + 0.18, A_peak * label_frac,
            f"{A_peak:.2f}\u00a0mm",
            ha="left", va="center", fontsize=9, color="black",
            bbox=dict(boxstyle="square,pad=0.22", fc="white", ec="#888",
                      lw=0.5, alpha=0.97),
            zorder=7)

    if show_delta:
        delta_mm = IN_MAX_A - OUT_MAX_A
        ratio    = OUT_MAX_A / IN_MAX_A
        x_arr = 5.6
        ax.annotate("", xy=(x_arr, IN_MAX_A), xytext=(x_arr, OUT_MAX_A),
                    arrowprops=dict(arrowstyle="<->", color="#222", lw=1.1,
                                    shrinkA=0, shrinkB=0),
                    zorder=5)
        mid_y = (IN_MAX_A + OUT_MAX_A) / 2
        ax.text(x_arr - 0.12, mid_y,
                rf"$\Delta = {delta_mm:.2f}$" + "\u00a0mm\n"
                rf"$A_\mathrm{{Ut}}/A_\mathrm{{inn}} = {ratio:.2f}$",
                ha="right", va="center", fontsize=8.5,
                bbox=dict(boxstyle="square,pad=0.25", fc="white",
                          ec="#444", lw=0.6, alpha=0.95),
                zorder=6)

    ax.set_xlim(0, XMAX_HARMONIC)
    ax.set_ylim(0, ymax)
    ax.set_title(title, fontsize=10)
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(MultipleLocator(0.2))
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(True, which="minor", alpha=0.12)


fig, axes = plt.subplots(2, 2, figsize=(8, 12), sharex=True, sharey=True)
for i, wind in enumerate(("nowind", "fullwind")):
    for j, side in enumerate(("IN", "OUT")):
        f, A = specs[(wind, side)]
        bar_panel_v8(axes[i, j], f, A, WIND_COLOR[wind],
                     f"{SIDE_LABEL_NO[side]}, {WIND_LABEL[wind]}",
                     ymax=YMAX_V21,
                     show_delta=(j == 1))
        if j == 0:
            axes[i, j].set_ylabel("Amplitude [mm]")
        axes[i, j].set_xlabel("Frekvens [Hz]")
        axes[i, j].tick_params(axis="x", labelbottom=True)

for ax_top in axes[0, :]:
    sec = ax_top.secondary_xaxis("top")
    sec.set_xticks(_harmonic_ticks)
    sec.set_xticklabels(_harmonic_ticklab)
    sec.tick_params(axis="x", which="major", labelsize=9, pad=2,
                    length=4, color="#444", labelcolor="#222")

fig.tight_layout()
fig.savefig(EXPLORE_DIR / "V22_bar_label_near_peak.png",
            dpi=140, bbox_inches="tight")
plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# V23 — per-wind Δ and ratio (recomputed from each row's own IN/OUT peaks)
# ══════════════════════════════════════════════════════════════════════════
# Previously both OUT panels showed the same Δ (fullwind IN − fullwind OUT)
# because the reference lines were global IN_MAX / OUT_MAX. Each row now
# has its own reference pair — and the two Δ boxes report the wind-specific
# numbers from meta.
#
#   nowind   : Δ = 4.67 mm,  A_Ut/A_inn = 0.70
#   fullwind : Δ = 4.34 mm,  A_Ut/A_inn = 0.73   ← less damping with wind
#
# The difference between the two rows' ratios is the wind-enhancement
# signal recorded in memory/methodology_wind_enhances_A_in.md.

# Per-wind reference values, looked up from meta.json via the existing specs.
WIND_PEAKS = {
    wind: {
        "IN":  float(meta[meta["path"] == str(RUNS[wind])].iloc[0]
                     [f"Probe {IN_PROBE} Amplitude (FFT)"]),
        "OUT": float(meta[meta["path"] == str(RUNS[wind])].iloc[0]
                     [f"Probe {OUT_PROBE} Amplitude (FFT)"]),
    }
    for wind in ("nowind", "fullwind")
}
print("\nPer-wind reference peaks (from meta):")
for w, d in WIND_PEAKS.items():
    dlt = d["IN"] - d["OUT"]
    rat = d["OUT"] / d["IN"]
    print(f"  {w:8s}  IN={d['IN']:.3f}  OUT={d['OUT']:.3f}  "
          f"Δ={dlt:.3f} mm  ratio={rat:.3f}")


def bar_panel_v9(ax, f, A, color, title, *,
                 ymax, in_peak, out_peak,
                 show_delta=False, label_frac=0.88):
    """Row-specific reference lines + Δ; peak-value label near the top."""
    df = f[1] - f[0]
    ax.bar(f, A, width=df * 0.9, color=color, edgecolor=color, lw=0.4,
           alpha=0.85, align="center", zorder=3)

    # Row-specific horizontal reference lines
    ax.axhline(in_peak,  color="#222", lw=0.7, ls=":", alpha=0.8, zorder=2)
    ax.axhline(out_peak, color="#888", lw=0.7, ls=":", alpha=0.8, zorder=2)

    for k, fk in HARMONICS:
        if fk <= XMAX_HARMONIC:
            ax.axvline(fk, color="#888", lw=0.5, ls=":", alpha=0.7, zorder=1)

    peak_i = int(np.argmax(A))
    f_peak, A_peak = f[peak_i], A[peak_i]
    ax.text(f_peak + 0.18, A_peak * label_frac,
            f"{A_peak:.2f}\u00a0mm",
            ha="left", va="center", fontsize=9, color="black",
            bbox=dict(boxstyle="square,pad=0.22", fc="white", ec="#888",
                      lw=0.5, alpha=0.97),
            zorder=7)

    if show_delta:
        delta_mm = in_peak - out_peak
        ratio    = out_peak / in_peak
        x_arr = 5.6
        ax.annotate("", xy=(x_arr, in_peak), xytext=(x_arr, out_peak),
                    arrowprops=dict(arrowstyle="<->", color="#222", lw=1.1,
                                    shrinkA=0, shrinkB=0),
                    zorder=5)
        mid_y = (in_peak + out_peak) / 2
        ax.text(x_arr - 0.12, mid_y,
                rf"$\Delta = {delta_mm:.2f}$" + "\u00a0mm\n"
                rf"$A_\mathrm{{Ut}}/A_\mathrm{{inn}} = {ratio:.2f}$",
                ha="right", va="center", fontsize=8.5,
                bbox=dict(boxstyle="square,pad=0.25", fc="white",
                          ec="#444", lw=0.6, alpha=0.95),
                zorder=6)

    ax.set_xlim(0, XMAX_HARMONIC)
    ax.set_ylim(0, ymax)
    ax.set_title(title, fontsize=10)
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(MultipleLocator(0.2))
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(True, which="minor", alpha=0.12)


fig, axes = plt.subplots(2, 2, figsize=(8, 12), sharex=True, sharey=True)
for i, wind in enumerate(("nowind", "fullwind")):
    for j, side in enumerate(("IN", "OUT")):
        f, A = specs[(wind, side)]
        bar_panel_v9(axes[i, j], f, A, WIND_COLOR[wind],
                     f"{SIDE_LABEL_NO[side]}, {WIND_LABEL[wind]}",
                     ymax=YMAX_V21,
                     in_peak=WIND_PEAKS[wind]["IN"],
                     out_peak=WIND_PEAKS[wind]["OUT"],
                     show_delta=(j == 1))
        if j == 0:
            axes[i, j].set_ylabel("Amplitude [mm]")
        axes[i, j].set_xlabel("Frekvens [Hz]")
        axes[i, j].tick_params(axis="x", labelbottom=True)

for ax_top in axes[0, :]:
    sec = ax_top.secondary_xaxis("top")
    sec.set_xticks(_harmonic_ticks)
    sec.set_xticklabels(_harmonic_ticklab)
    sec.tick_params(axis="x", which="major", labelsize=9, pad=2,
                    length=4, color="#444", labelcolor="#222")

fig.tight_layout()
fig.savefig(EXPLORE_DIR / "V23_bar_perwind_delta.png",
            dpi=140, bbox_inches="tight")
plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# V24 — fullwind IN as the global baseline
# ══════════════════════════════════════════════════════════════════════════
# Reference lines are the same in every panel now — fullwind IN (16.07 mm)
# and fullwind OUT (11.73 mm) — so all 4 panels share identical x, y, and
# horizontal reference guides. Δ is recomputed per Utgående panel using
# fullwind IN as the common baseline:
#
#   nowind OUT  : Δ = 16.07 − 10.66 = 5.41 mm,  A_Ut/A_inn = 0.66
#   fullwind OUT: Δ = 16.07 − 11.73 = 4.34 mm,  A_Ut/A_inn = 0.73
#
# The gap between these two ratios is the wind-enhancement signal.

BASELINE_IN  = WIND_PEAKS["fullwind"]["IN"]   # 16.07 mm — global IN baseline
BASELINE_OUT = WIND_PEAKS["fullwind"]["OUT"]  # 11.73 mm — global OUT guide
print(f"\nBaseline (fullwind IN): {BASELINE_IN:.3f} mm")


def bar_panel_v10(ax, f, A, color, title, *,
                  ymax, baseline_in, baseline_out,
                  out_peak=None, show_delta=False, label_frac=0.88):
    """Global reference lines at fullwind IN/OUT; Δ uses baseline_in."""
    df = f[1] - f[0]
    ax.bar(f, A, width=df * 0.9, color=color, edgecolor=color, lw=0.4,
           alpha=0.85, align="center", zorder=3)

    ax.axhline(baseline_in,  color="#222", lw=0.7, ls=":", alpha=0.8, zorder=2)
    ax.axhline(baseline_out, color="#888", lw=0.7, ls=":", alpha=0.8, zorder=2)

    for k, fk in HARMONICS:
        if fk <= XMAX_HARMONIC:
            ax.axvline(fk, color="#888", lw=0.5, ls=":", alpha=0.7, zorder=1)

    peak_i = int(np.argmax(A))
    f_peak, A_peak = f[peak_i], A[peak_i]
    ax.text(f_peak + 0.18, A_peak * label_frac,
            f"{A_peak:.2f}\u00a0mm",
            ha="left", va="center", fontsize=9, color="black",
            bbox=dict(boxstyle="square,pad=0.22", fc="white", ec="#888",
                      lw=0.5, alpha=0.97),
            zorder=7)

    if show_delta and out_peak is not None:
        delta_mm = baseline_in - out_peak
        ratio    = out_peak / baseline_in
        x_arr = 5.6
        ax.annotate("", xy=(x_arr, baseline_in), xytext=(x_arr, out_peak),
                    arrowprops=dict(arrowstyle="<->", color="#222", lw=1.1,
                                    shrinkA=0, shrinkB=0),
                    zorder=5)
        mid_y = (baseline_in + out_peak) / 2
        ax.text(x_arr - 0.12, mid_y,
                rf"$\Delta = {delta_mm:.2f}$" + "\u00a0mm\n"
                rf"$A_\mathrm{{Ut}}/A_\mathrm{{inn}} = {ratio:.2f}$",
                ha="right", va="center", fontsize=8.5,
                bbox=dict(boxstyle="square,pad=0.25", fc="white",
                          ec="#444", lw=0.6, alpha=0.95),
                zorder=6)

    ax.set_xlim(0, XMAX_HARMONIC)
    ax.set_ylim(0, ymax)
    ax.set_title(title, fontsize=10)
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(MultipleLocator(0.2))
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(True, which="minor", alpha=0.12)


fig, axes = plt.subplots(2, 2, figsize=(8, 12), sharex=True, sharey=True)
for i, wind in enumerate(("nowind", "fullwind")):
    for j, side in enumerate(("IN", "OUT")):
        f, A = specs[(wind, side)]
        bar_panel_v10(axes[i, j], f, A, WIND_COLOR[wind],
                      f"{SIDE_LABEL_NO[side]}, {WIND_LABEL[wind]}",
                      ymax=YMAX_V21,
                      baseline_in=BASELINE_IN,
                      baseline_out=BASELINE_OUT,
                      out_peak=WIND_PEAKS[wind]["OUT"],
                      show_delta=(j == 1))
        if j == 0:
            axes[i, j].set_ylabel("Amplitude [mm]")
        axes[i, j].set_xlabel("Frekvens [Hz]")
        axes[i, j].tick_params(axis="x", labelbottom=True)

for ax_top in axes[0, :]:
    sec = ax_top.secondary_xaxis("top")
    sec.set_xticks(_harmonic_ticks)
    sec.set_xticklabels(_harmonic_ticklab)
    sec.tick_params(axis="x", which="major", labelsize=9, pad=2,
                    length=4, color="#444", labelcolor="#222")

fig.tight_layout()
fig.savefig(EXPLORE_DIR / "V24_bar_fullwind_in_baseline.png",
            dpi=140, bbox_inches="tight")
plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# V25 — IN baseline global, OUT line per-row so the arrow always fits
# ══════════════════════════════════════════════════════════════════════════
# V24 drew the OUT horizontal line at the fullwind OUT peak (11.73 mm) on
# every panel, so the nowind Δ arrow overshot it down to 10.66 — ugly.
# Now the IN line stays global (fullwind IN = 16.07 mm baseline), but the
# OUT line is per-row (matches each row's own OUT peak). Result:
#
#   nowind   row: lines at 16.07 & 10.66 — arrow spans those   (Δ 5.41, r 0.66)
#   fullwind row: lines at 16.07 & 11.73 — arrow spans those   (Δ 4.34, r 0.73)
#
# Arrow endpoints always touch the two reference lines of that row.

def bar_panel_v11(ax, f, A, color, title, *,
                  ymax, baseline_in, row_out_peak,
                  show_delta=False, label_frac=0.88):
    df = f[1] - f[0]
    ax.bar(f, A, width=df * 0.9, color=color, edgecolor=color, lw=0.4,
           alpha=0.85, align="center", zorder=3)

    ax.axhline(baseline_in,  color="#222", lw=0.7, ls=":", alpha=0.8, zorder=2)
    ax.axhline(row_out_peak, color="#888", lw=0.7, ls=":", alpha=0.8, zorder=2)

    for k, fk in HARMONICS:
        if fk <= XMAX_HARMONIC:
            ax.axvline(fk, color="#888", lw=0.5, ls=":", alpha=0.7, zorder=1)

    peak_i = int(np.argmax(A))
    f_peak, A_peak = f[peak_i], A[peak_i]
    ax.text(f_peak + 0.18, A_peak * label_frac,
            f"{A_peak:.2f}\u00a0mm",
            ha="left", va="center", fontsize=9, color="black",
            bbox=dict(boxstyle="square,pad=0.22", fc="white", ec="#888",
                      lw=0.5, alpha=0.97),
            zorder=7)

    if show_delta:
        delta_mm = baseline_in - row_out_peak
        ratio    = row_out_peak / baseline_in
        x_arr = 5.6
        ax.annotate("", xy=(x_arr, baseline_in),
                    xytext=(x_arr, row_out_peak),
                    arrowprops=dict(arrowstyle="<->", color="#222", lw=1.1,
                                    shrinkA=0, shrinkB=0),
                    zorder=5)
        mid_y = (baseline_in + row_out_peak) / 2
        ax.text(x_arr - 0.12, mid_y,
                rf"$\Delta = {delta_mm:.2f}$" + "\u00a0mm\n"
                rf"$A_\mathrm{{Ut}}/A_\mathrm{{inn}} = {ratio:.2f}$",
                ha="right", va="center", fontsize=8.5,
                bbox=dict(boxstyle="square,pad=0.25", fc="white",
                          ec="#444", lw=0.6, alpha=0.95),
                zorder=6)

    ax.set_xlim(0, XMAX_HARMONIC)
    ax.set_ylim(0, ymax)
    ax.set_title(title, fontsize=10)
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(MultipleLocator(0.2))
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(True, which="minor", alpha=0.12)


fig, axes = plt.subplots(2, 2, figsize=(8, 12), sharex=True, sharey=True)
for i, wind in enumerate(("nowind", "fullwind")):
    row_out = WIND_PEAKS[wind]["OUT"]
    for j, side in enumerate(("IN", "OUT")):
        f, A = specs[(wind, side)]
        bar_panel_v11(axes[i, j], f, A, WIND_COLOR[wind],
                      f"{SIDE_LABEL_NO[side]}, {WIND_LABEL[wind]}",
                      ymax=YMAX_V21,
                      baseline_in=BASELINE_IN,
                      row_out_peak=row_out,
                      show_delta=(j == 1))
        if j == 0:
            axes[i, j].set_ylabel("Amplitude [mm]")
        axes[i, j].set_xlabel("Frekvens [Hz]")
        axes[i, j].tick_params(axis="x", labelbottom=True)

for ax_top in axes[0, :]:
    sec = ax_top.secondary_xaxis("top")
    sec.set_xticks(_harmonic_ticks)
    sec.set_xticklabels(_harmonic_ticklab)
    sec.tick_params(axis="x", which="major", labelsize=9, pad=2,
                    length=4, color="#444", labelcolor="#222")

fig.tight_layout()
fig.savefig(EXPLORE_DIR / "V25_bar_perrow_out_line.png",
            dpi=140, bbox_inches="tight")
plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# V26 — fully per-row reference lines (both IN and OUT), per-wind Δ
# ══════════════════════════════════════════════════════════════════════════
# V25's IN line was fixed at fullwind IN (16.07 mm) on both rows, so the
# nowind row showed an IN line ABOVE its actual nowind IN peak (15.33 mm).
# Now the IN line is per-row too, both arrow endpoints sit on the two
# reference lines of that row, and Δ/ratio are recomputed per wind:
#
#   nowind   row: lines at 15.33 & 10.66,  Δ 4.67 mm, A_Ut/A_inn 0.70
#   fullwind row: lines at 16.07 & 11.73,  Δ 4.34 mm, A_Ut/A_inn 0.73

def bar_panel_v12(ax, f, A, color, title, *,
                  ymax, row_in_peak, row_out_peak,
                  show_delta=False, label_frac=0.88):
    df = f[1] - f[0]
    ax.bar(f, A, width=df * 0.9, color=color, edgecolor=color, lw=0.4,
           alpha=0.85, align="center", zorder=3)

    ax.axhline(row_in_peak,  color="#222", lw=0.7, ls=":", alpha=0.8, zorder=2)
    ax.axhline(row_out_peak, color="#888", lw=0.7, ls=":", alpha=0.8, zorder=2)

    for k, fk in HARMONICS:
        if fk <= XMAX_HARMONIC:
            ax.axvline(fk, color="#888", lw=0.5, ls=":", alpha=0.7, zorder=1)

    peak_i = int(np.argmax(A))
    f_peak, A_peak = f[peak_i], A[peak_i]
    ax.text(f_peak + 0.18, A_peak * label_frac,
            f"{A_peak:.2f}\u00a0mm",
            ha="left", va="center", fontsize=9, color="black",
            bbox=dict(boxstyle="square,pad=0.22", fc="white", ec="#888",
                      lw=0.5, alpha=0.97),
            zorder=7)

    if show_delta:
        delta_mm = row_in_peak - row_out_peak
        ratio    = row_out_peak / row_in_peak
        x_arr = 5.6
        ax.annotate("", xy=(x_arr, row_in_peak),
                    xytext=(x_arr, row_out_peak),
                    arrowprops=dict(arrowstyle="<->", color="#222", lw=1.1,
                                    shrinkA=0, shrinkB=0),
                    zorder=5)
        mid_y = (row_in_peak + row_out_peak) / 2
        ax.text(x_arr - 0.12, mid_y,
                rf"$\Delta = {delta_mm:.2f}$" + "\u00a0mm\n"
                rf"$A_\mathrm{{Ut}}/A_\mathrm{{inn}} = {ratio:.2f}$",
                ha="right", va="center", fontsize=8.5,
                bbox=dict(boxstyle="square,pad=0.25", fc="white",
                          ec="#444", lw=0.6, alpha=0.95),
                zorder=6)

    ax.set_xlim(0, XMAX_HARMONIC)
    ax.set_ylim(0, ymax)
    ax.set_title(title, fontsize=10)
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(MultipleLocator(0.2))
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(True, which="minor", alpha=0.12)


fig, axes = plt.subplots(2, 2, figsize=(8, 12), sharex=True, sharey=True)
for i, wind in enumerate(("nowind", "fullwind")):
    row_in  = WIND_PEAKS[wind]["IN"]
    row_out = WIND_PEAKS[wind]["OUT"]
    for j, side in enumerate(("IN", "OUT")):
        f, A = specs[(wind, side)]
        bar_panel_v12(axes[i, j], f, A, WIND_COLOR[wind],
                      f"{SIDE_LABEL_NO[side]}, {WIND_LABEL[wind]}",
                      ymax=YMAX_V21,
                      row_in_peak=row_in,
                      row_out_peak=row_out,
                      show_delta=(j == 1))
        if j == 0:
            axes[i, j].set_ylabel("Amplitude [mm]")
        axes[i, j].set_xlabel("Frekvens [Hz]")
        axes[i, j].tick_params(axis="x", labelbottom=True)

for ax_top in axes[0, :]:
    sec = ax_top.secondary_xaxis("top")
    sec.set_xticks(_harmonic_ticks)
    sec.set_xticklabels(_harmonic_ticklab)
    sec.tick_params(axis="x", which="major", labelsize=9, pad=2,
                    length=4, color="#444", labelcolor="#222")

fig.tight_layout()
fig.savefig(EXPLORE_DIR / "V26_bar_fully_perrow.png",
            dpi=140, bbox_inches="tight")
plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# V27 — same as V26 rendered in New Computer Modern
# ══════════════════════════════════════════════════════════════════════════
# Register the NCM OpenType files shipped with TeX Live directly with
# matplotlib — no LaTeX round-trip needed. Math uses the built-in Computer
# Modern mathtext set (visually identical to NCM for the symbols used).
# newcomputermodern's LaTeX package itself requires XeTeX/LuaTeX, so going
# the OTF route is the cleanest way to match the thesis body text in PNG
# output.

from matplotlib import font_manager as _fm

_NCM_DIR = "/usr/local/texlive/2025/texmf-dist/fonts/opentype/public/newcomputermodern"
for _fname in ("NewCM10-Regular.otf", "NewCM10-Bold.otf",
               "NewCM10-Italic.otf", "NewCM10-BoldItalic.otf",
               "NewCMMath-Regular.otf"):
    _fp = f"{_NCM_DIR}/{_fname}"
    try:
        _fm.fontManager.addfont(_fp)
    except Exception as _e:
        print(f"   warn: could not register {_fname}: {_e}")

_prev_rc = {k: plt.rcParams[k] for k in (
    "text.usetex", "font.family", "font.serif", "mathtext.fontset",
    "mathtext.rm", "mathtext.it", "mathtext.bf",
)}

plt.rcParams.update({
    "text.usetex":        False,
    "font.family":        "serif",
    "font.serif":         ["NewComputerModern10", "DejaVu Serif"],
    "mathtext.fontset":   "cm",  # CM mathtext (NCM is built on CM shapes)
})


def _nbsp(s: str) -> str:
    return s  # NBSP fine in non-LaTeX mode


def bar_panel_v13(ax, f, A, color, title, *,
                  ymax, row_in_peak, row_out_peak,
                  show_delta=False, label_frac=0.88):
    df = f[1] - f[0]
    ax.bar(f, A, width=df * 0.9, color=color, edgecolor=color, lw=0.4,
           alpha=0.85, align="center", zorder=3)

    ax.axhline(row_in_peak,  color="#222", lw=0.7, ls=":", alpha=0.8, zorder=2)
    ax.axhline(row_out_peak, color="#888", lw=0.7, ls=":", alpha=0.8, zorder=2)

    for k, fk in HARMONICS:
        if fk <= XMAX_HARMONIC:
            ax.axvline(fk, color="#888", lw=0.5, ls=":", alpha=0.7, zorder=1)

    peak_i = int(np.argmax(A))
    f_peak, A_peak = f[peak_i], A[peak_i]
    ax.text(f_peak + 0.18, A_peak * label_frac,
            _nbsp(f"{A_peak:.2f}\u00a0mm"),
            ha="left", va="center", fontsize=9, color="black",
            bbox=dict(boxstyle="square,pad=0.22", fc="white", ec="#888",
                      lw=0.5, alpha=0.97),
            zorder=7)

    if show_delta:
        delta_mm = row_in_peak - row_out_peak
        ratio    = row_out_peak / row_in_peak
        x_arr = 5.6
        ax.annotate("", xy=(x_arr, row_in_peak),
                    xytext=(x_arr, row_out_peak),
                    arrowprops=dict(arrowstyle="<->", color="#222", lw=1.1,
                                    shrinkA=0, shrinkB=0),
                    zorder=5)
        mid_y = (row_in_peak + row_out_peak) / 2
        ax.text(x_arr - 0.12, mid_y,
                rf"$\Delta = {delta_mm:.2f}$" + "\u00a0mm\n"
                rf"$A_\mathrm{{Ut}}/A_\mathrm{{inn}} = {ratio:.2f}$",
                ha="right", va="center", fontsize=8.5,
                bbox=dict(boxstyle="square,pad=0.25", fc="white",
                          ec="#444", lw=0.6, alpha=0.95),
                zorder=6)

    ax.set_xlim(0, XMAX_HARMONIC)
    ax.set_ylim(0, ymax)
    ax.set_title(title, fontsize=10)
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(MultipleLocator(0.2))
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(True, which="minor", alpha=0.12)


fig, axes = plt.subplots(2, 2, figsize=(8, 12), sharex=True, sharey=True)
for i, wind in enumerate(("nowind", "fullwind")):
    row_in  = WIND_PEAKS[wind]["IN"]
    row_out = WIND_PEAKS[wind]["OUT"]
    for j, side in enumerate(("IN", "OUT")):
        f, A = specs[(wind, side)]
        bar_panel_v13(axes[i, j], f, A, WIND_COLOR[wind],
                      f"{SIDE_LABEL_NO[side]}, {WIND_LABEL[wind]}",
                      ymax=YMAX_V21,
                      row_in_peak=row_in,
                      row_out_peak=row_out,
                      show_delta=(j == 1))
        if j == 0:
            axes[i, j].set_ylabel("Amplitude [mm]")
        axes[i, j].set_xlabel("Frekvens [Hz]")
        axes[i, j].tick_params(axis="x", labelbottom=True)

for ax_top in axes[0, :]:
    sec = ax_top.secondary_xaxis("top")
    sec.set_xticks(_harmonic_ticks)
    sec.set_xticklabels([r"$f$", r"$2f$", r"$3f$", r"$4f$"][:len(_harmonic_ticks)])
    sec.tick_params(axis="x", which="major", labelsize=9, pad=2,
                    length=4, color="#444", labelcolor="#222")

fig.tight_layout()
fig.savefig(EXPLORE_DIR / "V27_bar_newcomputermodern.png",
            dpi=160, bbox_inches="tight")
plt.close(fig)

# Restore rc so subsequent figures in this script (if any) don't inherit.
plt.rcParams.update(_prev_rc)


print("\nDone. Output PNGs:")
for p in sorted(EXPLORE_DIR.glob("V*.png")):
    print(f"  {p.relative_to(BASE)}")
