"""Timeseries exploration — five new visual styles (v2).

Companion/sibling to analysis_scratch/thesis_timeseries_showcase.py — that
script already covers nine "obvious" framings (anatomy / probe stack /
wind-vs-nowind / envelope / frequency ladder / spectrogram / overlay).
This script experiments with five less-obvious framings that answer
different questions.

    1. eye_diagram              — fold the H&G stable window by period T and
                                  overlay every cycle.  Tight = stable tone;
                                  fuzzy = wind jitter or envelope drift.
                                  Answers: how period-to-period stable is
                                  the wave inside the H&G window?
    2. cycle_amplitude_barcode  — each cycle's (max−min)/2 as a bar along x;
                                  nowind vs fullwind on one axis.  Answers:
                                  is the wind enhancement of A_IN steady
                                  through the window or does it drift?
    3. upcrossing_ladder        — zero-upcrossing time vs index.  Straight
                                  line = perfectly periodic; residuals after
                                  linear fit show the jitter budget.
    4. spatiotemporal_heatmap   — 2-D imshow with time on x, probe distance
                                  on y, η as colour.  Wave propagation seen
                                  as a diagonal band.
    5. phase_portrait           — η(t) vs η(t+T/4) quadrature plot.  Pure
                                  sinusoid = round loop; wind = fuzzy loop;
                                  Stokes nonlinearity = flattened top.

Outputs → output/FIGURES/timeseries_exploration/*.pdf (+ .pgf).
Nothing is deleted; existing showcase stays in output/FIGURES/showcase_timeseries/.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, "/Users/ole/Kodevik/wave_project")
os.chdir("/Users/ole/Kodevik/wave_project")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

from wavescripts.constants import MEASUREMENT
from wavescripts.improved_data_loader import load_analysis_data, load_processed_dfs
from wavescripts.plot_utils import WIND_COLOR_MAP, apply_thesis_style

# ── Constants ────────────────────────────────────────────────────────────────
FS = MEASUREMENT.SAMPLING_RATE  # 250 Hz
OUT_DIR = Path("output/FIGURES/timeseries_exploration")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load canon March-2026 data ───────────────────────────────────────────────
PROCESSED_DIRS = sorted(Path("waveprocessed").glob("PROCESSED-202603*"))[-6:]
meta, _, _, _ = load_analysis_data(*PROCESSED_DIRS, load_processed=False)
dfs = load_processed_dfs(*PROCESSED_DIRS)


def _find_run(fname_tag: str) -> pd.Series:
    matches = meta[meta["path"].str.endswith(fname_tag)]
    if matches.empty:
        raise SystemExit(f"no match for {fname_tag!r}")
    return matches.iloc[0]


# A matched pair at 1.5 Hz, 0.2 V, full panel — same conditions except wind
WIND_OFF = _find_run("fullpanel-nowind-amp0200-freq1500-per40-depth580-mstop30-run1.csv")
WIND_ON  = _find_run("fullpanel-fullwind-amp0200-freq1500-per40-depth580-mstop30-run1.csv")
# Long record for the spatiotemporal heatmap (per240 = more settled train)
LONG_RUN = _find_run("fullpanel-nowind-amp0200-freq1400-per240-depth580-run1.csv")

apply_thesis_style()


def _probe_signal(row: pd.Series, pos: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (t_seconds, eta_mm) for probe *pos* in *row*'s run."""
    df = dfs[row["path"]]
    col = f"eta_{pos}_interp" if f"eta_{pos}_interp" in df.columns else f"eta_{pos}"
    eta = df[col].to_numpy()
    t = np.arange(len(eta)) / FS
    return t, eta


def _hg_samples(row: pd.Series, pos: str) -> tuple[int | None, int | None]:
    s = row.get(f"Computed Probe {pos} start")
    e = row.get(f"Computed Probe {pos} end")
    if pd.isna(s) or pd.isna(e):
        return None, None
    return int(s), int(e)


def _zero_upcrossings(eta: np.ndarray) -> np.ndarray:
    """Return integer sample indices where eta crosses 0 going up."""
    s = np.signbit(eta).astype(np.int8)  # 1 where eta<0, 0 where eta>=0
    # upcrossing: prev negative (s=1), next non-negative (s=0) → diff = -1
    return np.where(np.diff(s) == -1)[0]


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — Eye diagram
# ─────────────────────────────────────────────────────────────────────────────
def fig_eye_diagram() -> None:
    """2×2 grid: fold the H&G window by period T and overlay every cycle."""
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.4),
                             sharex=True, sharey="row")
    f_hz = WIND_OFF["WaveFrequencyInput [Hz]"]
    T_per = 1.0 / f_hz
    T_samp = int(round(T_per * FS))

    col_defs = [
        ("no",   "nowind",   WIND_OFF),
        ("full", "fullwind", WIND_ON),
    ]
    row_defs = [
        ("IN",  WIND_OFF["in_position"]),   # 9373/170
        ("OUT", WIND_OFF["out_position"]),  # 12400/250
    ]

    for ri, (side, pos) in enumerate(row_defs):
        for ci, (wkey, wname, run_row) in enumerate(col_defs):
            ax = axes[ri, ci]
            t, eta = _probe_signal(run_row, pos)
            hg_s, hg_e = _hg_samples(run_row, pos)
            if hg_s is None:
                ax.text(0.5, 0.5, "no H&G window", transform=ax.transAxes,
                        ha="center", va="center", fontsize=9, color="grey")
                continue
            # Align to first upcrossing inside the H&G window
            upc = _zero_upcrossings(eta[hg_s:hg_e])
            if len(upc) < 3:
                ax.text(0.5, 0.5, "no upcrossings", transform=ax.transAxes,
                        ha="center", va="center", fontsize=9, color="grey")
                continue
            start = hg_s + upc[0]
            n_cycles = min(len(upc) - 1, (hg_e - start) // T_samp)

            color = WIND_COLOR_MAP[wkey]
            cycles = []
            for k in range(n_cycles):
                i0, i1 = start + k * T_samp, start + (k + 1) * T_samp
                if i1 > len(eta):
                    break
                seg = eta[i0:i1]
                if len(seg) != T_samp:
                    continue
                cycles.append(seg)
                ax.plot(np.arange(T_samp) / FS * 1000, seg,
                        color=color, lw=0.35, alpha=0.28)
            if cycles:
                stack = np.asarray(cycles)
                median = np.median(stack, axis=0)
                ax.plot(np.arange(T_samp) / FS * 1000, median,
                        color=color, lw=1.8, label=f"median of {len(cycles)} cycles")

            ax.axhline(0, color="black", lw=0.5, alpha=0.5)
            ax.grid(alpha=0.25)
            ax.set_title(f"{side} probe ({pos}) — {wname}",
                         fontsize=9, color=color)
            if ri == 1:
                ax.set_xlabel("phase within period [ms]")
            if ci == 0:
                ax.set_ylabel(r"$\eta$ [mm]")
            ax.legend(loc="lower right", fontsize=7, frameon=False)

    fig.suptitle(
        f"Eye diagram — cycles folded by $T = 1/f$ inside the H\\&G window "
        f"(f = {f_hz:.2f} Hz, 0.2 V, fullpanel)",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in ("pdf", "pgf"):
        fig.savefig(OUT_DIR / f"eye_diagram.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ eye_diagram")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — Cycle-by-cycle amplitude barcode
# ─────────────────────────────────────────────────────────────────────────────
def fig_cycle_amplitude_barcode() -> None:
    """Sequential (max-min)/2 per cycle within H&G window; nowind vs fullwind
    overlaid on same axis for IN and OUT."""
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 4.4), sharex=True)
    f_hz = WIND_OFF["WaveFrequencyInput [Hz]"]
    T_samp = int(round((1.0 / f_hz) * FS))

    probes = [("IN",  WIND_OFF["in_position"]),
              ("OUT", WIND_OFF["out_position"])]

    for ax, (side, pos) in zip(axes, probes):
        for wkey, wname, run_row in (("no",   "nowind",   WIND_OFF),
                                     ("full", "fullwind", WIND_ON)):
            t, eta = _probe_signal(run_row, pos)
            hg_s, hg_e = _hg_samples(run_row, pos)
            if hg_s is None:
                continue
            upc = _zero_upcrossings(eta[hg_s:hg_e])
            if len(upc) < 3:
                continue
            starts = hg_s + upc
            amps = []
            for i0, i1 in zip(starts[:-1], starts[1:]):
                seg = eta[i0:i1]
                if len(seg) < T_samp * 0.8:
                    continue
                amps.append(0.5 * (seg.max() - seg.min()))
            amps = np.asarray(amps)
            color = WIND_COLOR_MAP[wkey]
            ax.plot(np.arange(len(amps)), amps,
                    color=color, marker="o", ms=3.5, lw=0.9,
                    label=f"{wname}  (mean {amps.mean():.2f} mm)")
            ax.axhline(amps.mean(), color=color, lw=0.7, ls="--", alpha=0.5)

        ax.set_ylabel(r"$(max-min)/2$  [mm]")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=8, frameon=False)
        ax.text(0.015, 0.92, f"{side} probe ({pos})",
                transform=ax.transAxes, fontsize=9, ha="left", va="top",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5))

    axes[-1].set_xlabel("cycle index within H\\&G window")
    fig.suptitle(
        f"Per-cycle amplitude across the stable window "
        f"(f = {f_hz:.2f} Hz, 0.2 V, fullpanel)",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("pdf", "pgf"):
        fig.savefig(OUT_DIR / f"cycle_amplitude_barcode.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ cycle_amplitude_barcode")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — Zero-upcrossing ladder
# ─────────────────────────────────────────────────────────────────────────────
def fig_upcrossing_ladder() -> None:
    """Upper: upcrossing time vs index + linear fit. Lower: residuals.
    Nowind and fullwind overlaid."""
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(7.6, 4.6), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )
    f_hz = WIND_OFF["WaveFrequencyInput [Hz]"]
    pos = WIND_OFF["in_position"]

    for wkey, wname, run_row in (("no",   "nowind",   WIND_OFF),
                                 ("full", "fullwind", WIND_ON)):
        t, eta = _probe_signal(run_row, pos)
        hg_s, hg_e = _hg_samples(run_row, pos)
        if hg_s is None:
            continue
        upc_local = _zero_upcrossings(eta[hg_s:hg_e])
        upc_t = (hg_s + upc_local) / FS
        idx = np.arange(len(upc_t))
        if len(upc_t) < 3:
            continue
        slope, intercept = np.polyfit(idx, upc_t, 1)
        resid_ms = (upc_t - (slope * idx + intercept)) * 1000

        color = WIND_COLOR_MAP[wkey]
        ax_top.plot(idx, upc_t, color=color, marker="o", ms=3.5, lw=0.7,
                    label=f"{wname}  (T = {slope*1000:.2f} ms)")
        ax_bot.plot(idx, resid_ms, color=color, marker="o", ms=3.5, lw=0.7,
                    label=f"{wname}  (σ = {resid_ms.std():.2f} ms)")

    ax_top.set_ylabel("upcrossing time [s]")
    ax_top.grid(alpha=0.25)
    ax_top.legend(loc="upper left", fontsize=8, frameon=False)
    ax_top.set_title(
        f"Zero-upcrossing ladder — IN probe ({pos}), "
        f"f = {f_hz:.2f} Hz, 0.2 V",
        fontsize=10,
    )
    ax_bot.axhline(0, color="black", lw=0.5, alpha=0.7)
    ax_bot.set_ylabel("residual [ms]")
    ax_bot.set_xlabel("upcrossing index within H\\&G window")
    ax_bot.grid(alpha=0.25)
    ax_bot.legend(loc="upper right", fontsize=8, frameon=False)

    fig.tight_layout()
    for ext in ("pdf", "pgf"):
        fig.savefig(OUT_DIR / f"upcrossing_ladder.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ upcrossing_ladder")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — Spatiotemporal heatmap
# ─────────────────────────────────────────────────────────────────────────────
def fig_spatiotemporal_heatmap() -> None:
    """Time × probe-distance heatmap of η. Wave arrival seen as a diagonal."""
    positions_ordered = [
        ("8804/250",  8.804),
        ("9373/170",  9.373),
        ("9373/340",  9.373),
        ("12400/250", 12.400),
    ]
    df = dfs[LONG_RUN["path"]]
    series, y_vals, labels = [], [], []
    for pos, dist in positions_ordered:
        col = f"eta_{pos}_interp" if f"eta_{pos}_interp" in df.columns else f"eta_{pos}"
        if col not in df.columns:
            continue
        eta = df[col].to_numpy()
        series.append(eta)
        y_vals.append(dist)
        labels.append(pos)
    if len(series) < 2:
        print("  ✗ spatiotemporal_heatmap (not enough probes)")
        return
    n = min(len(s) for s in series)
    M = np.vstack([s[:n] for s in series])
    t = np.arange(n) / FS

    # Symmetric clim
    vmax = float(np.nanpercentile(np.abs(M), 99))
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    im = ax.imshow(
        M, aspect="auto", origin="lower",
        extent=[t[0], t[-1], -0.5, M.shape[0] - 0.5],
        cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        interpolation="nearest",
    )
    ax.set_yticks(np.arange(M.shape[0]))
    ax.set_yticklabels([f"{lab}\n({d:.2f} m)" for lab, d in zip(labels, y_vals)],
                       fontsize=8)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("probe (distance from paddle)")
    cb = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.04)
    cb.set_label(r"$\eta$ [mm]")

    hg_s, hg_e = _hg_samples(LONG_RUN, positions_ordered[0][0])
    if hg_s is not None:
        ax.axvline(hg_s / FS, color="black", lw=0.7, ls=":", alpha=0.6)
        ax.axvline(hg_e / FS, color="black", lw=0.7, ls=":", alpha=0.6)
        ax.text(hg_s / FS, M.shape[0] - 0.2, "  H\\&G",
                fontsize=8, color="black", va="top")

    f_hz = LONG_RUN["WaveFrequencyInput [Hz]"]
    ax.set_title(
        f"Spatiotemporal surface elevation (f = {f_hz:.2f} Hz, per240, nowind, fullpanel)",
        fontsize=10, pad=6,
    )
    fig.tight_layout()
    for ext in ("pdf", "pgf"):
        fig.savefig(OUT_DIR / f"spatiotemporal_heatmap.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ spatiotemporal_heatmap")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5 — Phase portrait η(t) vs η(t+T/4)
# ─────────────────────────────────────────────────────────────────────────────
def fig_phase_portrait() -> None:
    """2×2: IN/OUT × nowind/fullwind. Quadrature embedding colored by time."""
    f_hz = WIND_OFF["WaveFrequencyInput [Hz]"]
    tau = int(round(0.25 * FS / f_hz))  # T/4 in samples

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.0),
                             sharex="row", sharey="row")

    col_defs = [("no",   "nowind",   WIND_OFF),
                ("full", "fullwind", WIND_ON)]
    row_defs = [("IN",  WIND_OFF["in_position"]),
                ("OUT", WIND_OFF["out_position"])]

    for ri, (side, pos) in enumerate(row_defs):
        for ci, (wkey, wname, run_row) in enumerate(col_defs):
            ax = axes[ri, ci]
            t, eta = _probe_signal(run_row, pos)
            hg_s, hg_e = _hg_samples(run_row, pos)
            if hg_s is None or (hg_e - hg_s) <= tau:
                ax.text(0.5, 0.5, "no H&G window", transform=ax.transAxes,
                        ha="center", va="center", fontsize=9, color="grey")
                continue
            x = eta[hg_s:hg_e - tau]
            y = eta[hg_s + tau:hg_e]
            # Colour trajectory by normalised progress
            pts = np.column_stack([x, y]).reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            lc = LineCollection(segs, cmap="viridis",
                                norm=plt.Normalize(0, 1), linewidth=0.45,
                                alpha=0.75)
            lc.set_array(np.linspace(0, 1, len(segs)))
            ax.add_collection(lc)
            lim = 1.15 * max(abs(x).max(), abs(y).max())
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect("equal")
            ax.axhline(0, color="black", lw=0.4, alpha=0.5)
            ax.axvline(0, color="black", lw=0.4, alpha=0.5)
            ax.grid(alpha=0.25)
            ax.set_title(f"{side} ({pos}) — {wname}",
                         fontsize=9, color=WIND_COLOR_MAP[wkey])
            if ri == 1:
                ax.set_xlabel(r"$\eta(t)$  [mm]")
            if ci == 0:
                ax.set_ylabel(r"$\eta(t + T/4)$  [mm]")

    # Shared colorbar for time progression
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label("normalised progress through H\\&G window", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    fig.suptitle(
        f"Phase portrait — $\\eta(t)$ vs $\\eta(t+T/4)$  "
        f"(f = {f_hz:.2f} Hz, 0.2 V, fullpanel)",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 0.9, 0.97])
    for ext in ("pdf", "pgf"):
        fig.savefig(OUT_DIR / f"phase_portrait.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ phase_portrait")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"writing to {OUT_DIR}")
    fig_eye_diagram()
    fig_cycle_amplitude_barcode()
    fig_upcrossing_ladder()
    fig_spatiotemporal_heatmap()
    fig_phase_portrait()
    print("done.")
