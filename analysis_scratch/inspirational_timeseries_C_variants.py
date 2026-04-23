"""
Variants of Figure C (macro+micro, _cut) exploring:
  - figure height (taller than the 7.2 in draft)
  - y-tick spacing (current y_major=2 gives 17 labels — overcrowded)
  - zoom panel height share

Output: output/timeseries_exploration/insp_C_<tag>.pdf + .png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from matplotlib.ticker import MultipleLocator

from wavescripts.improved_data_loader import load_analysis_data, load_processed_dfs
from wavescripts.plot_utils import apply_thesis_style, WIND_COLOR_MAP

FS = 250.0
BASE = Path("/Users/ole/Kodevik/wave_project")
OUTDIR = BASE / "output/timeseries_exploration"
OUTDIR.mkdir(parents=True, exist_ok=True)

TARGET_DIR = BASE / "waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"
CHOSEN_PATH = str(BASE / "wavedata/20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange/fullpanel-nowind-amp0200-freq1400-per240-depth580-mstop30-run1.csv")

IN_PROBE  = "9373/170"
OUT_PROBE = "12400/250"
FREQ = 1.4
AMP_V = 0.2

WIND_COLOR = WIND_COLOR_MAP["no"]
IN_COLOR  = WIND_COLOR
OUT_COLOR = WIND_COLOR
WIN_COLOR = "#F1B24A"

apply_thesis_style()

# ─── Load once ───────────────────────────────────────────────────────────
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

# ka values for the title (reader-facing wave descriptor — CLAUDE.md §19).
# Show IN and OUT separately: the panel attenuates, so the two differ.
KA_IN  = float(row["IN ka (FFT)"])
KA_OUT = float(row["OUT ka (FFT)"])


def sym_ylim(*sigs, pad=0.15):
    y = np.concatenate(sigs)
    lo, hi = np.nanpercentile(y, [0.5, 99.5])
    m = max(abs(lo), abs(hi))
    return -m * (1 + pad), m * (1 + pad)


ylim_macro = sym_ylim(eta_in, eta_out)

# Per-probe zoom windows: centre on each probe's own H&G window so the OUT
# zoom reflects the later wave arrival at the OUT probe. Zoom spans 5 periods.
zoom_half = 2.5 / FREQ
in_zcenter  = (in_ws + in_we) / 2
out_zcenter = (out_ws + out_we) / 2
in_zx0,  in_zx1  = in_zcenter  - zoom_half, in_zcenter  + zoom_half
out_zx0, out_zx1 = out_zcenter - zoom_half, out_zcenter + zoom_half

# Tighter ylim for zoom: use plateau amplitude, not the ramp-up full range.
m_in  = (t >= in_zx0)  & (t <= in_zx1)
m_out = (t >= out_zx0) & (t <= out_zx1)
ylim_zoom = sym_ylim(eta_in[m_in], eta_out[m_out], pad=0.25)

X_CUTOFF_S = max(in_we, out_we) + 10.0


def apply_ticks(ax, *, x_major, x_minor, y_major, y_minor):
    ax.xaxis.set_major_locator(MultipleLocator(x_major))
    ax.xaxis.set_minor_locator(MultipleLocator(x_minor))
    ax.yaxis.set_major_locator(MultipleLocator(y_major))
    ax.yaxis.set_minor_locator(MultipleLocator(y_minor))
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(True, which="minor", alpha=0.12)


def make_figure_C(
    tag: str,
    *,
    figsize: tuple[float, float],
    height_ratios: tuple[float, float, float, float],
    y_major_macro: float,
    y_minor_macro: float,
    y_major_zoom: float,
    y_minor_zoom: float,
    tight_zoom_ylim: bool = False,
    hspace: float = 0.42,
):
    """Build one variant and save as .pdf + .png."""
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(4, 1, height_ratios=list(height_ratios), hspace=hspace)
    ax_in_full  = fig.add_subplot(gs[0, 0])
    ax_in_zoom  = fig.add_subplot(gs[1, 0])
    # Macros share x (0 … X_CUTOFF_S). Zooms do NOT share x — each follows its
    # own probe's H&G window (OUT arrives later, so its zoom is shifted right).
    ax_out_full = fig.add_subplot(gs[2, 0], sharex=ax_in_full)
    ax_out_zoom = fig.add_subplot(gs[3, 0])

    yl_zoom = ylim_zoom if tight_zoom_ylim else ylim_macro

    for ax_full, ax_zoom, sig, color, probe, label, zx0, zx1 in [
        (ax_in_full,  ax_in_zoom,  eta_in,  IN_COLOR,  IN_PROBE,  "Innkommende bølge",
         in_zx0,  in_zx1),
        (ax_out_full, ax_out_zoom, eta_out, OUT_COLOR, OUT_PROBE, "Utgående bølge",
         out_zx0, out_zx1),
    ]:
        ax_full.plot(t, sig, color=color, lw=0.5)
        ax_full.axvspan(zx0, zx1, color=WIN_COLOR, alpha=0.55, zorder=0)
        ax_full.axhline(0, color="#888", lw=0.5, alpha=0.6)
        ax_full.set_xlim(0, X_CUTOFF_S)
        ax_full.set_ylim(ylim_macro)
        ax_full.set_ylabel(r"$\eta$ (mm)")
        ax_full.text(
            0.005, 0.92, label,
            transform=ax_full.transAxes, va="top", ha="left",
            fontsize=10, color=color, weight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, lw=0.6),
        )

        m = (t >= zx0) & (t <= zx1)
        ax_zoom.plot(t[m], sig[m], color=color, lw=1.3)
        ax_zoom.scatter(t[m][::6], sig[m][::6], s=6, color=color, alpha=0.55,
                        edgecolor="none")
        ax_zoom.axhline(0, color="#888", lw=0.5, alpha=0.6)
        ax_zoom.set_xlim(zx0, zx1)
        ax_zoom.set_ylim(yl_zoom)
        ax_zoom.set_ylabel(r"$\eta$ (mm)")
        ax_zoom.text(
            0.005, 0.92, "zoom · 5 perioder",
            transform=ax_zoom.transAxes, va="top", ha="left",
            fontsize=9, color="#444",
        )

        for xx in (zx0, zx1):
            con = ConnectionPatch(
                xyA=(xx, ylim_macro[0]), coordsA=ax_full.transData,
                xyB=(xx, yl_zoom[1]),   coordsB=ax_zoom.transData,
                color=WIN_COLOR, lw=0.7, alpha=0.9, zorder=0,
            )
            fig.add_artist(con)

    ax_out_zoom.set_xlabel("Tid [s]")
    ax_in_zoom.set_xlabel("Tid [s]")
    for ax in (ax_in_full, ax_out_full):
        apply_ticks(ax, x_major=5.0, x_minor=1.0,
                    y_major=y_major_macro, y_minor=y_minor_macro)
    for ax in (ax_in_zoom, ax_out_zoom):
        apply_ticks(ax, x_major=0.5, x_minor=0.1,
                    y_major=y_major_zoom, y_minor=y_minor_zoom)

    # Norwegian decimal convention: comma separator for numeric literals in
    # the title. ka stays with dots per user's "leave as is for now" note.
    freq_no = f"{FREQ:g}".replace(".", ",")
    amp_no  = f"{AMP_V:g}".replace(".", ",")
    fig.suptitle(
        f"Tidsserier, uten vind · Frekvensvalg = {freq_no} Hz · "
        f"Amplitudevalg = {amp_no} V · "
        f"$ka_{{\\mathrm{{IN}}}}$ = {KA_IN:.3f} · $ka_{{\\mathrm{{OUT}}}}$ = {KA_OUT:.3f}   [{tag}]",
        fontsize=11,
    )

    out_pdf = OUTDIR / f"insp_C_{tag}.pdf"
    out_png = OUTDIR / f"insp_C_{tag}.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_pdf.name}  (+ .png)")


# ─── Variants ────────────────────────────────────────────────────────────
# baseline: current draft geometry for comparison
make_figure_C(
    "v00_baseline",
    figsize=(10, 7.2),
    height_ratios=(2.2, 1, 2.2, 1),
    y_major_macro=2.0, y_minor_macro=1.0,
    y_major_zoom=2.0,  y_minor_zoom=1.0,
)

# taller, coarser y-ticks
make_figure_C(
    "v01_tall_y5",
    figsize=(10, 10.0),
    height_ratios=(2.2, 1, 2.2, 1),
    y_major_macro=5.0, y_minor_macro=1.0,
    y_major_zoom=5.0,  y_minor_zoom=1.0,
)

# extra-tall, coarser still
make_figure_C(
    "v02_xtall_y5",
    figsize=(10, 12.0),
    height_ratios=(2.2, 1, 2.2, 1),
    y_major_macro=5.0, y_minor_macro=1.0,
    y_major_zoom=5.0,  y_minor_zoom=1.0,
)

# tall + zoom panels larger share
make_figure_C(
    "v03_tall_bigzoom_y5",
    figsize=(10, 10.5),
    height_ratios=(2.0, 1.3, 2.0, 1.3),
    y_major_macro=5.0, y_minor_macro=1.0,
    y_major_zoom=5.0,  y_minor_zoom=1.0,
)

# tall + tight zoom ylim (zoom gets its own scale, ~plateau amplitude)
make_figure_C(
    "v04_tall_tightzoom",
    figsize=(10, 10.0),
    height_ratios=(2.2, 1, 2.2, 1),
    y_major_macro=5.0, y_minor_macro=1.0,
    y_major_zoom=2.0,  y_minor_zoom=1.0,
    tight_zoom_ylim=True,
)

# same but every 4 mm major (no labelled "10" lookalike)
make_figure_C(
    "v05_tall_y4",
    figsize=(10, 10.0),
    height_ratios=(2.2, 1, 2.2, 1),
    y_major_macro=4.0, y_minor_macro=2.0,
    y_major_zoom=4.0,  y_minor_zoom=2.0,
)

# tall + very coarse (only -10, 0, 10 and edge ends on macro)
make_figure_C(
    "v06_tall_y10",
    figsize=(10, 10.0),
    height_ratios=(2.2, 1, 2.2, 1),
    y_major_macro=10.0, y_minor_macro=2.0,
    y_major_zoom=5.0,   y_minor_zoom=1.0,
)

# extra-tall + tight zoom + coarse zoom y-ticks
make_figure_C(
    "v07_xtall_tightzoom_y4",
    figsize=(10, 12.0),
    height_ratios=(2.0, 1.2, 2.0, 1.2),
    y_major_macro=5.0, y_minor_macro=1.0,
    y_major_zoom=4.0,  y_minor_zoom=2.0,
    tight_zoom_ylim=True,
)

print(f"\nAll variants in: {OUTDIR}")
