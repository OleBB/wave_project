"""
Snap-shift diagnostic visualization — single-run, IN vs OUT, both winds.

Purpose: visually demonstrate why the user's trough-count between IN and OUT
macro panels differs by ~1 trough before the zoom band starts, and whether
that asymmetry changes under fullwind.

Overlay on the canon runs (1.4 Hz, 0.2 V, per240, fullpanel):
  - Dashed vertical lines : theoretical H&G window (pre-snap, c_g-anchored)
  - Solid amber band      : snapped H&G window (actually used for FFT)
  - Zoom panel            : 10 periods = exact snapped-window contents
                            → user can count exactly 10 troughs inside
  - Annotation            : snap shift in samples + wave periods per probe

Side-by-side nowind vs fullwind shows that the ~1-trough visual asymmetry is
a snap-shift artifact (per-probe phase offset), not a wind-dependent physics
effect — the phase offsets are similar magnitude in both cases but bias
changes depending on which probe ends up nearer the flip boundary (±0.5 T).

Companion to analysis_scratch/hg_snap_shift_diagnostic.py (population stats).

Output: output/timeseries_exploration/insp_snap_diagnostic_{nowind,fullwind}.pdf + .png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

from wavescripts.improved_data_loader import load_analysis_data, load_processed_dfs
from wavescripts.plot_utils import apply_thesis_style, WIND_COLOR_MAP

FS = 250.0
BASE = Path("/Users/ole/Kodevik/wave_project")
OUTDIR = BASE / "output/timeseries_exploration"
OUTDIR.mkdir(parents=True, exist_ok=True)

TARGET_DIR  = BASE / "waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"
DATADIR     = BASE / "wavedata/20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"

RUNS = {
    "nowind":   str(DATADIR / "fullpanel-nowind-amp0200-freq1400-per240-depth580-mstop30-run1.csv"),
    "fullwind": str(DATADIR / "fullpanel-fullwind-amp0200-freq1400-per240-depth580-mstop30-run1.csv"),
}
WIND_KEY = {"nowind": "no", "fullwind": "full"}
TITLE_WIND = {"nowind": "uten vind", "fullwind": "med full vind"}

IN_PROBE  = "9373/170"
OUT_PROBE = "12400/250"
FREQ  = 1.4
AMP_V = 0.2

SNAPPED_COLOR   = "#F1B24A"   # amber — actually-used window
THEORY_COLOR    = "#555555"   # grey  — pre-snap theoretical window

apply_thesis_style()

# ─── Load once ───────────────────────────────────────────────────────────
print("Loading …")
meta, _, _, _ = load_analysis_data(str(TARGET_DIR), load_processed=False)
proc = load_processed_dfs(str(TARGET_DIR))

T = 1.0 / FREQ
SAMPLES_PER_T = FS * T


def load_run(wind_tag: str) -> dict:
    csv = RUNS[wind_tag]
    row = meta[meta["path"] == csv].iloc[0]
    df  = proc[csv]
    t   = np.arange(len(df)) / FS

    def get_eta(probe):
        col = f"eta_{probe}_interp"
        if col not in df.columns:
            col = f"eta_{probe}"
        return df[col].to_numpy(dtype=float)

    def win(probe):
        return dict(
            snap_s  = int(row[f"Computed Probe {probe} start"]) / FS,
            snap_e  = int(row[f"Computed Probe {probe} end"])   / FS,
            theor_s = int(row[f"Probe {probe} hg_expected_start"]) / FS,
            theor_e = int(row[f"Probe {probe} hg_expected_end"])   / FS,
            shift_T = int(row[f"Probe {probe} hg_snap_shift"]) / SAMPLES_PER_T,
            shift_samples = int(row[f"Probe {probe} hg_snap_shift"]),
        )

    return {
        "row":     row,
        "t":       t,
        "eta_in":  get_eta(IN_PROBE),
        "eta_out": get_eta(OUT_PROBE),
        "in_win":  win(IN_PROBE),
        "out_win": win(OUT_PROBE),
        "ka_in":   float(row["IN ka (FFT)"]),
        "ka_out":  float(row["OUT ka (FFT)"]),
    }


def sym_ylim(*sigs, pad=0.15):
    y = np.concatenate(sigs)
    lo, hi = np.nanpercentile(y, [0.5, 99.5])
    m = max(abs(lo), abs(hi))
    return -m * (1 + pad), m * (1 + pad)


def apply_ticks(ax, *, x_major, x_minor, y_major, y_minor):
    ax.xaxis.set_major_locator(MultipleLocator(x_major))
    ax.xaxis.set_minor_locator(MultipleLocator(x_minor))
    ax.yaxis.set_major_locator(MultipleLocator(y_major))
    ax.yaxis.set_minor_locator(MultipleLocator(y_minor))
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(True, which="minor", alpha=0.12)


def make_figure(wind_tag: str, data: dict, out_pdf: Path) -> None:
    color = WIND_COLOR_MAP[WIND_KEY[wind_tag]]

    eta_in, eta_out = data["eta_in"], data["eta_out"]
    t = data["t"]
    in_w  = data["in_win"]
    out_w = data["out_win"]
    ylim  = sym_ylim(eta_in, eta_out)
    x_cutoff = max(in_w["snap_e"], out_w["snap_e"]) + 10.0

    net_shift_T = in_w["shift_T"] - out_w["shift_T"]

    fig = plt.figure(figsize=(10, 12.0))
    gs = fig.add_gridspec(4, 1, height_ratios=[2.2, 1, 2.2, 1], hspace=0.42)
    ax_in_full  = fig.add_subplot(gs[0, 0])
    ax_in_zoom  = fig.add_subplot(gs[1, 0])
    ax_out_full = fig.add_subplot(gs[2, 0], sharex=ax_in_full)
    ax_out_zoom = fig.add_subplot(gs[3, 0])

    for ax_full, ax_zoom, sig, probe, label, w in [
        (ax_in_full,  ax_in_zoom,  eta_in,  IN_PROBE,  "Innkommende bølge",  in_w),
        (ax_out_full, ax_out_zoom, eta_out, OUT_PROBE, "Utgående bølge",     out_w),
    ]:
        ax_full.plot(t, sig, color=color, lw=0.5)
        ax_full.axvspan(w["snap_s"], w["snap_e"], color=SNAPPED_COLOR,
                        alpha=0.55, zorder=0)
        for xx in (w["theor_s"], w["theor_e"]):
            ax_full.axvline(xx, color=THEORY_COLOR, ls="--", lw=1.0,
                            alpha=0.85, zorder=1)
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
        ax_full.text(
            0.995, 0.92,
            f"snap: {w['shift_T']:+.2f} T  ({w['shift_samples']:+d} samples)",
            transform=ax_full.transAxes, va="top", ha="right",
            fontsize=9, color="#333",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#999", lw=0.5),
        )

        # Zoom = exactly the 10-period snapped H&G window contents.
        m = (t >= w["snap_s"]) & (t <= w["snap_e"])
        ax_zoom.plot(t[m], sig[m], color=color, lw=1.1)
        ax_zoom.axhline(0, color="#888", lw=0.5, alpha=0.6)
        ax_zoom.set_xlim(w["snap_s"], w["snap_e"])
        ax_zoom.set_ylim(ylim)
        ax_zoom.set_ylabel(r"$\eta$ (mm)")
        ax_zoom.text(
            0.005, 0.92, "zoom · 10 perioder (H&G-vindu)",
            transform=ax_zoom.transAxes, va="top", ha="left",
            fontsize=9, color="#444",
        )

        for xx in (w["snap_s"], w["snap_e"]):
            con = ConnectionPatch(
                xyA=(xx, ylim[0]), coordsA=ax_full.transData,
                xyB=(xx, ylim[1]), coordsB=ax_zoom.transData,
                color=SNAPPED_COLOR, lw=0.7, alpha=0.9, zorder=0,
            )
            fig.add_artist(con)

    ax_in_zoom.set_xlabel("Tid [s]")
    ax_out_zoom.set_xlabel("Tid [s]")
    for ax in (ax_in_full, ax_out_full):
        apply_ticks(ax, x_major=5.0, x_minor=1.0, y_major=5.0, y_minor=1.0)
    for ax in (ax_in_zoom, ax_out_zoom):
        apply_ticks(ax, x_major=1.0, x_minor=0.5, y_major=5.0, y_minor=1.0)

    _legend_handles = [
        Patch(facecolor=SNAPPED_COLOR, alpha=0.55, label="snapped H&G (used, 10 T)"),
        Line2D([0], [0], color=THEORY_COLOR, ls="--", lw=1.0,
               label="theoretical H&G (pre-snap, c_g-anchored)"),
    ]
    ax_in_full.legend(handles=_legend_handles, loc="lower right",
                      fontsize=8, frameon=True, framealpha=0.9)

    freq_no = f"{FREQ:g}".replace(".", ",")
    amp_no  = f"{AMP_V:g}".replace(".", ",")
    fig.suptitle(
        f"H&G snap-shift diagnostic — {TITLE_WIND[wind_tag]} · "
        f"f = {freq_no} Hz · A = {amp_no} V · "
        f"IN snap {in_w['shift_T']:+.2f} T, OUT snap {out_w['shift_T']:+.2f} T, "
        f"net {net_shift_T:+.2f} T ≈ {abs(net_shift_T):.1f} trough",
        fontsize=10,
    )

    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"   → {out_pdf.relative_to(BASE)}  (+ .png)")


# ─── Build both figures ──────────────────────────────────────────────────
for wind_tag in ("nowind", "fullwind"):
    print(f"\n{wind_tag} …")
    data = load_run(wind_tag)
    print(f"  IN  snap: {data['in_win']['shift_samples']:+d} samples = {data['in_win']['shift_T']:+.3f} T")
    print(f"  OUT snap: {data['out_win']['shift_samples']:+d} samples = {data['out_win']['shift_T']:+.3f} T")
    net = data['in_win']['shift_T'] - data['out_win']['shift_T']
    print(f"  Net (IN − OUT): {net:+.3f} T ≈ {abs(net):.2f} wave periods")
    make_figure(wind_tag, data, OUTDIR / f"insp_snap_diagnostic_{wind_tag}.pdf")

print("\nDone.")
