"""
Exploration: compare two subpanel-height layouts for the CH04 §5
inspirational time-series figures.

Two layouts, two wind conditions → 4 PNGs in the exploration folder.

Variant A — all four subpanels equal height
    height_ratios = [1, 1, 1, 1]
Variant B — zooms taller than macros
    height_ratios = [1, 2, 1, 2]

Output:
    output/timeseries_height_exploration/{nowind,fullwind}_{A,B}.png

Does NOT touch output/FIGURES/ or output/TEXFIGU/. Once a layout is
picked, we promote by editing analysis_scratch/inspirational_timeseries.py.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm
from matplotlib.patches import ConnectionPatch
from matplotlib.ticker import MultipleLocator

from wavescripts.improved_data_loader import load_analysis_data, load_processed_dfs
from wavescripts.plot_utils import apply_thesis_style, WIND_COLOR_MAP
import wavescripts.plot_utils as pu


FS = 250.0
BASE = Path("/Users/ole/Kodevik/wave_project")
OUT_DIR = BASE / "output" / "timeseries_height_exploration"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_DIR = BASE / "waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"
DATADIR    = BASE / "wavedata/20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"

RUNS = {
    "nowind":   DATADIR / "fullpanel-nowind-amp0200-freq1400-per240-depth580-mstop30-run1.csv",
    "fullwind": DATADIR / "fullpanel-fullwind-amp0200-freq1400-per240-depth580-mstop30-run1.csv",
}

IN_PROBE  = "9373/170"
OUT_PROBE = "12400/250"
FREQ      = 1.4
WIND_KEY  = {"nowind": "no", "fullwind": "full"}
WIN_COLOR = "#F1B24A"


# Font: match the FFT figure and the thesis body (NewComputerModern)
apply_thesis_style()
_NCM_DIR = "/usr/local/texlive/2025/texmf-dist/fonts/opentype/public/newcomputermodern"
for _fname in ("NewCM10-Regular.otf", "NewCM10-Bold.otf",
               "NewCM10-Italic.otf", "NewCM10-BoldItalic.otf",
               "NewCMMath-Regular.otf"):
    try:
        _fm.fontManager.addfont(f"{_NCM_DIR}/{_fname}")
    except Exception as _e:
        print(f"   warn: could not register {_fname}: {_e}")
plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["NewComputerModern10", "DejaVu Serif"],
    "mathtext.fontset": "cm",
})


print("Loading …")
meta, _, _, _ = load_analysis_data(str(TARGET_DIR), load_processed=False)
proc = load_processed_dfs(str(TARGET_DIR))


def _load_run(wind_tag: str):
    csv = str(RUNS[wind_tag])
    row = meta[meta["path"] == csv].iloc[0]
    df  = proc[csv]
    t   = np.arange(len(df)) / FS

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

    return dict(
        row=row, t=t, eta_in=eta_in, eta_out=eta_out,
        in_ws=in_ws, in_we=in_we, out_ws=out_ws, out_we=out_we,
        ka_in=float(row["IN ka (FFT)"]),
        ka_out=float(row["OUT ka (FFT)"]),
    )


def _sym_ylim(*sigs, pad=0.15):
    y = np.concatenate(sigs)
    lo, hi = np.nanpercentile(y, [0.5, 99.5])
    m = max(abs(lo), abs(hi))
    return -m * (1 + pad), m * (1 + pad)


def _apply_ticks(ax, *, x_major, x_minor, y_major, y_minor):
    ax.xaxis.set_major_locator(MultipleLocator(x_major))
    ax.xaxis.set_minor_locator(MultipleLocator(x_minor))
    ax.yaxis.set_major_locator(MultipleLocator(y_major))
    ax.yaxis.set_minor_locator(MultipleLocator(y_minor))
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(True, which="minor", alpha=0.12)


def make_figure(wind_tag: str, data: dict, out_png: Path,
                height_ratios=(1, 1, 1, 1), figsize=(10, 12)):
    color = WIND_COLOR_MAP[WIND_KEY[wind_tag]]

    eta_in, eta_out = data["eta_in"], data["eta_out"]
    t = data["t"]
    in_ws, in_we   = data["in_ws"],  data["in_we"]
    out_ws, out_we = data["out_ws"], data["out_we"]
    ylim = _sym_ylim(eta_in, eta_out)
    x_cutoff = max(in_we, out_we) + 10.0

    zoom_half = 2.5 / FREQ
    in_zx0,  in_zx1  = (in_ws + in_we) / 2 - zoom_half,  (in_ws + in_we) / 2 + zoom_half
    out_zx0, out_zx1 = (out_ws + out_we) / 2 - zoom_half, (out_ws + out_we) / 2 + zoom_half

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(4, 1, height_ratios=list(height_ratios), hspace=0.42)
    ax_in_full  = fig.add_subplot(gs[0, 0])
    ax_in_zoom  = fig.add_subplot(gs[1, 0])
    ax_out_full = fig.add_subplot(gs[2, 0], sharex=ax_in_full)
    ax_out_zoom = fig.add_subplot(gs[3, 0])

    for ax_full, ax_zoom, sig, probe, label, zx0, zx1 in [
        (ax_in_full,  ax_in_zoom,  eta_in,  IN_PROBE,  "Innkommende bølge",
         in_zx0,  in_zx1),
        (ax_out_full, ax_out_zoom, eta_out, OUT_PROBE, "Utgående bølge",
         out_zx0, out_zx1),
    ]:
        ax_full.plot(t, sig, color=color, lw=0.5)
        ax_full.axvspan(zx0, zx1, color=WIN_COLOR, alpha=0.55, zorder=0)
        ax_full.axhline(0, color="#888", lw=0.5, alpha=0.6)
        ax_full.set_xlim(0, x_cutoff)
        ax_full.set_ylim(ylim)
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
        ax_zoom.set_ylim(ylim)
        ax_zoom.set_ylabel(r"$\eta$ (mm)")
        ax_zoom.text(
            0.005, 0.92, "zoom · 5 perioder",
            transform=ax_zoom.transAxes, va="top", ha="left",
            fontsize=9, color="#444",
        )

        for xx in (zx0, zx1):
            con = ConnectionPatch(
                xyA=(xx, ylim[0]), coordsA=ax_full.transData,
                xyB=(xx, ylim[1]), coordsB=ax_zoom.transData,
                color=WIN_COLOR, lw=0.7, alpha=0.9, zorder=0,
            )
            fig.add_artist(con)

    ax_in_zoom.set_xlabel("Tid [s]")
    ax_out_zoom.set_xlabel("Tid [s]")
    for ax in (ax_in_full, ax_out_full):
        _apply_ticks(ax, x_major=5.0, x_minor=1.0, y_major=5.0, y_minor=1.0)
    for ax in (ax_in_zoom, ax_out_zoom):
        _apply_ticks(ax, x_major=0.5, x_minor=0.1, y_major=5.0, y_minor=1.0)

    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"   → {out_png.relative_to(BASE)}")


# ─── Build both variants for both wind conditions ─────────────────────────
VARIANTS = {
    "A_equal":      dict(height_ratios=(1, 1, 1, 1), figsize=(10, 12)),
    "B_zoom_tall":  dict(height_ratios=(1, 2, 1, 2), figsize=(10, 13)),
}

for wind_tag in ("nowind", "fullwind"):
    print(f"\n{wind_tag} …")
    data = _load_run(wind_tag)
    for tag, cfg in VARIANTS.items():
        out_png = OUT_DIR / f"{wind_tag}_{tag}.png"
        make_figure(wind_tag, data, out_png, **cfg)

print("\nDone. Outputs:")
for p in sorted(OUT_DIR.glob("*.png")):
    print(f"  {p.relative_to(BASE)}")
