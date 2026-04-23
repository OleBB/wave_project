"""
Inspirational time-series figures for Ch. 4 §5 opening.

Two figures, identical layout — nowind vs fullwind canon run at 1.4 Hz, 0.2 V,
per240, fullpanel. Macro (full recording, ~52 s) + micro (5-period zoom) for
both IN and OUT probes. The zoom window follows each probe's own H&G window,
so the OUT zoom sits later in time than IN (wave arrival is ~5 s later at OUT).

Outputs (written directly; no stub regeneration once captions have been
hand-edited):
    output/FIGURES/ch04_inspirational_nowind.pdf
    output/FIGURES/ch04_inspirational_fullwind.pdf
    output/TEXFIGU/ch04_inspirational_nowind.tex
    output/TEXFIGU/ch04_inspirational_fullwind.tex

See also the exploratory variants in
    analysis_scratch/inspirational_timeseries_C_variants.py
which writes to output/timeseries_exploration/ for side-by-side comparison
of figure height / y-tick density / zoom-ylim choices.
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
import wavescripts.plot_utils as pu

FS = 250.0
BASE = Path("/Users/ole/Kodevik/wave_project")
FIGURES_DIR = BASE / "output" / "FIGURES"
TEXFIGU_DIR = BASE / "output" / "TEXFIGU"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TEXFIGU_DIR.mkdir(parents=True, exist_ok=True)

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

# Title wording per wind condition (Norwegian, thesis-facing).
TITLE_WIND = {
    "nowind":   "Tidsserier, uten vind",
    "fullwind": "Tidsserier, med full vind",
}

# Thesis-wide colour convention: wind condition → colour (CLAUDE.md).
# The badge border + signal colour both pick this up.
WIND_KEY = {"nowind": "no", "fullwind": "full"}
WIN_COLOR = "#F1B24A"  # amber — shaded H&G window + zoom connectors


apply_thesis_style()
pu.ACTIVE_DATASETS = [p.name for p in
                     sorted(BASE.glob("waveprocessed/PROCESSED-*"))]
pu.TEXFIGU_DIR = TEXFIGU_DIR
pu.FIGURES_DIR = FIGURES_DIR


# ─── Load once ───────────────────────────────────────────────────────────
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

    return {
        "row":     row,
        "t":       t,
        "eta_in":  eta_in,
        "eta_out": eta_out,
        "in_ws":   in_ws,  "in_we":  in_we,
        "out_ws":  out_ws, "out_we": out_we,
        "ka_in":   float(row["IN ka (FFT)"]),
        "ka_out":  float(row["OUT ka (FFT)"]),
    }


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


def make_figure(wind_tag: str, data: dict, out_pdf: Path) -> None:
    """Geometry: v02_xtall_y5 — (10×12) figsize, y_major=5, default height ratios."""
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

    fig = plt.figure(figsize=(10, 12.0))
    gs = fig.add_gridspec(4, 1, height_ratios=[2.2, 1, 2.2, 1], hspace=0.42)
    ax_in_full  = fig.add_subplot(gs[0, 0])
    ax_in_zoom  = fig.add_subplot(gs[1, 0])
    # Macros share x (0 … x_cutoff). Zooms do NOT share x — each follows its
    # own probe's H&G window, so OUT zoom shifts right (later wave arrival).
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

    # No in-figure title: LaTeX \caption{} handles identification in the
    # thesis. ka_IN and ka_OUT are already in the immutable stub block
    # (extra_stats → stat:ka_in / stat:ka_out) and in the caption text.
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"   → {out_pdf.relative_to(BASE)}  (+ .png)")


def _write_stub(wind_tag: str, data: dict, figure_name: str) -> None:
    """Canonical 6-section TEXFIGU stub via pu.write_figure_stub (write-once)."""
    stub_path = TEXFIGU_DIR / f"{figure_name}.tex"

    caption = (
        f"Inspirational time-series overview ({TITLE_WIND[wind_tag].lower()}) "
        f"of a canonical run at $f = {FREQ}$\\,Hz, paddle drive $V = {AMP_V}$\\,V, "
        "full panel, per240. Top: incoming wave (probe at 9373/170, IN side of "
        "the panel). Bottom: outgoing wave (probe at 12400/250, OUT side). Each "
        "probe panel is accompanied by a 5-period zoom centred on that probe's "
        "own H\\&G analysis window; the OUT zoom sits later in time because the "
        "wave group arrives at the OUT probe $\\sim\\!5$\\,s after the IN probe. "
        f"Measured $ka_{{\\mathrm{{IN}}}}$ = {data['ka_in']:.3f}, "
        f"$ka_{{\\mathrm{{OUT}}}}$ = {data['ka_out']:.3f}. "
        "Amber shading on the macro panels marks the zoom interval; the 10-period "
        "analysis window is a subset of that interval (see H\\&G methodology)."
    )

    _meta = pu.build_fig_meta(
        {
            "filters": {
                "PanelCondition":            "full",
                "WaveFrequencyInput [Hz]":   FREQ,
                "WaveAmplitudeInput [Volt]": AMP_V,
                "WindCondition":             WIND_KEY[wind_tag],
                "quality_flag":              "ok",
            },
            "plotting": {
                "figure_name": figure_name,
                "caption":     caption,
            },
        },
        chapter="04",
        extra={"script": "analysis_scratch/inspirational_timeseries.py"},
        computed_in="analysis_scratch/inspirational_timeseries.py (single-run time-series render)",
        data_class="DELEG",
        fft_window_hz=0.1,
        extra_params=(
            f"run_path={RUNS[wind_tag].relative_to(BASE)}, "
            f"probes=IN:{IN_PROBE}+OUT:{OUT_PROBE}, "
            "zoom=5 periods centred on each probe's H&G window"
        ),
        extra_stats={
            "ka_in":        f"{data['ka_in']:.3f}",
            "ka_out":       f"{data['ka_out']:.3f}",
            "in_window_s":  f"[{data['in_ws']:.2f}, {data['in_we']:.2f}]",
            "out_window_s": f"[{data['out_ws']:.2f}, {data['out_we']:.2f}]",
        },
    )

    pu.write_figure_stub(_meta, plot_type="inspirational_timeseries",
                         subfig_filenames=[figure_name])
    print(f"   stub → {stub_path.relative_to(BASE)}")


# ─── Build both figures ──────────────────────────────────────────────────
for wind_tag in ("nowind", "fullwind"):
    print(f"\n{wind_tag} …")
    data = _load_run(wind_tag)
    figure_name = f"ch04_inspirational_{wind_tag}"
    make_figure(wind_tag, data, FIGURES_DIR / f"{figure_name}.pdf")
    _write_stub(wind_tag, data, figure_name)

print("\nDone.")
