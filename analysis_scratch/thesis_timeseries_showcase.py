"""Thesis-ready time-series showcase: four different visual styles.

Each figure introduces the reader to a different aspect of what a wave-tank
record looks like, using the recent (post-H&G) march2026 data.

    1. anatomy_of_a_run   — single-probe long record with annotated regions
                            (soft-start, stable train, post-stop decay) and a
                            few-period zoom inset.
    2. probe_travel_stack — same run, four probes stacked vertically ordered
                            by distance from the paddle; shows the wavetrain
                            arriving at each probe.
    3. wind_vs_nowind     — two-panel comparison at 1.5 Hz of the IN probe
                            with/without wind; motivates the thesis question.
    4. signal_and_envelope— IN vs OUT probe with Hilbert envelope overlay;
                            shows damping as an envelope ratio and the
                            group-velocity arrival plateau.

Outputs → output/FIGURES/showcase_timeseries/*.pdf  (+ .pgf).
"""
from __future__ import annotations
import os, sys
from pathlib import Path

sys.path.insert(0, "/Users/ole/Kodevik/wave_project")
os.chdir("/Users/ole/Kodevik/wave_project")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.offsetbox import AnchoredText
from scipy.signal import hilbert

from wavescripts.improved_data_loader import load_analysis_data, load_processed_dfs
from wavescripts.plot_utils import apply_thesis_style, WIND_COLOR_MAP
from wavescripts.constants import MEASUREMENT

# ── Constants ────────────────────────────────────────────────────────────────
FS = MEASUREMENT.SAMPLING_RATE  # 250 Hz
OUT_DIR = Path("output/FIGURES/showcase_timeseries")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load ─────────────────────────────────────────────────────────────────────
PROCESSED_DIRS = sorted(Path("waveprocessed").glob("PROCESSED-202603*"))[-6:]
meta, _, _, _ = load_analysis_data(*PROCESSED_DIRS, load_processed=False)
dfs = load_processed_dfs(*PROCESSED_DIRS)

def find_run(fname_tag: str) -> pd.Series:
    matches = meta[meta["path"].str.endswith(fname_tag)]
    if len(matches) == 0:
        raise SystemExit(f"no match for {fname_tag!r}")
    return matches.iloc[0]

# Target runs
ANATOMY   = find_run("fullpanel-nowind-amp0200-freq1400-per240-depth580-run1.csv")
WIND_OFF  = find_run("fullpanel-nowind-amp0200-freq1500-per40-depth580-mstop30-run1.csv")
WIND_ON   = find_run("fullpanel-fullwind-amp0200-freq1500-per40-depth580-mstop30-run1.csv")

apply_thesis_style()


def _probe_signal(row: pd.Series, pos: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (t_seconds, eta_mm) for probe *pos* in *row*'s run."""
    df = dfs[row["path"]]
    col = f"eta_{pos}_interp" if f"eta_{pos}_interp" in df.columns else f"eta_{pos}"
    eta = df[col].to_numpy()
    t = np.arange(len(eta)) / FS
    return t, eta


def _hg_window(row: pd.Series, pos: str) -> tuple[float | None, float | None]:
    s = row.get(f"Computed Probe {pos} start")
    e = row.get(f"Computed Probe {pos} end")
    if pd.isna(s) or pd.isna(e):
        return None, None
    return float(s) / FS, float(e) / FS


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — Anatomy of a wave run
# ─────────────────────────────────────────────────────────────────────────────
def fig_anatomy() -> None:
    pos = ANATOMY["in_position"]  # 9373/170
    t, eta = _probe_signal(ANATOMY, pos)
    hg_s, hg_e = _hg_window(ANATOMY, pos)
    f_hz = ANATOMY["WaveFrequencyInput [Hz]"]

    fig = plt.figure(figsize=(7.2, 4.2))
    ax = fig.add_subplot(111)

    ax.plot(t, eta, color="#17375E", lw=0.6)

    # Region shading (soft start → H&G stable window → wavemaker stop)
    if hg_s is not None:
        ax.axvspan(0, hg_s, color="#FFE8B3", alpha=0.55, zorder=0)             # ramp
        ax.axvspan(hg_s, hg_e, color="#C9E4C5", alpha=0.55, zorder=0)          # stable
        ax.axvspan(hg_e, t[-1], color="#F2C8C8", alpha=0.35, zorder=0)         # tail

    y_top = float(np.nanpercentile(eta, 99.5)) * 1.2
    y_bot = float(np.nanpercentile(eta,  0.5)) * 1.2
    ax.set_ylim(y_bot, y_top + 6)

    # Region labels
    label_y = y_top + 1.2
    if hg_s is not None:
        ax.text(hg_s/2,            label_y, "soft-start\n(wavemaker ramp)",
                ha="center", va="center", fontsize=9, color="#805600")
        ax.text((hg_s+hg_e)/2,     label_y, "stable wavetrain\n(H\\&G window)",
                ha="center", va="center", fontsize=9, color="#0E5614")
        ax.text((hg_e + t[-1])/2,  label_y, "decay\n(paddle off)",
                ha="center", va="center", fontsize=9, color="#802020")

    # Inset: a few periods of pure sinusoid
    if hg_s is not None:
        n_periods = 4
        t0_ins = hg_s + 1.0
        t1_ins = t0_ins + n_periods / f_hz
        mask = (t >= t0_ins) & (t <= t1_ins)
        ax_in = ax.inset_axes([0.55, 0.09, 0.42, 0.32])
        ax_in.plot(t[mask], eta[mask], color="#17375E", lw=0.9)
        ax_in.set_xlabel("time [s]", fontsize=8)
        ax_in.set_ylabel(r"$\eta$ [mm]", fontsize=8)
        ax_in.tick_params(labelsize=7)
        ax_in.grid(alpha=0.3)
        ax_in.set_title(f"zoom: {n_periods} periods @ {f_hz:.2f} Hz",
                        fontsize=8, pad=2)
        # Draw rectangle on main plot showing inset window
        y_ins_span = (ax_in.get_ylim()[0], ax_in.get_ylim()[1])
        ax.plot([t0_ins, t1_ins, t1_ins, t0_ins, t0_ins],
                [y_ins_span[0], y_ins_span[0], y_ins_span[1], y_ins_span[1], y_ins_span[0]],
                color="black", lw=0.6, alpha=0.5)

    ax.set_xlabel("time [s]")
    ax.set_ylabel(r"free-surface elevation $\eta$ [mm]")
    ax.set_title(
        f"Anatomy of a wave-tank run — probe {pos} "
        f"(f = {f_hz:.2f} Hz, per240, nowind, fullpanel)",
        pad=12,
    )
    ax.grid(alpha=0.3)

    for ext in ("pdf", "pgf"):
        fig.savefig(OUT_DIR / f"anatomy_of_a_run.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ anatomy_of_a_run")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — Four-probe travel stack
# ─────────────────────────────────────────────────────────────────────────────
def fig_travel_stack() -> None:
    # Order probes by longitudinal distance (paddle → end)
    positions_ordered = [
        ("8804/250",  "upstream (8.80 m)"),
        ("9373/170",  "IN wall (9.37 m)"),
        ("9373/340",  "IN far  (9.37 m)"),
        ("12400/250", "OUT center (12.40 m)"),
    ]
    f_hz = ANATOMY["WaveFrequencyInput [Hz]"]

    fig, axes = plt.subplots(
        len(positions_ordered), 1,
        figsize=(7.2, 5.8),
        sharex=True, sharey=True,
    )
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(positions_ordered)))

    # Shared xlim / ylim derived from all probes
    global_ylim = [0, 0]
    series_cache = []
    for pos, _ in positions_ordered:
        if f"eta_{pos}" not in dfs[ANATOMY["path"]].columns and \
           f"eta_{pos}_interp" not in dfs[ANATOMY["path"]].columns:
            series_cache.append(None)
            continue
        t, eta = _probe_signal(ANATOMY, pos)
        series_cache.append((t, eta))
        global_ylim[0] = min(global_ylim[0], float(np.nanpercentile(eta, 0.2)))
        global_ylim[1] = max(global_ylim[1], float(np.nanpercentile(eta, 99.8)))
    pad = 0.1 * (global_ylim[1] - global_ylim[0])
    global_ylim = (global_ylim[0] - pad, global_ylim[1] + pad)

    for ax, (pos, label), data, color in zip(axes, positions_ordered, series_cache, colors):
        if data is None:
            ax.text(0.5, 0.5, f"no data for {pos}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="grey")
            continue
        t, eta = data
        hg_s, hg_e = _hg_window(ANATOMY, pos)
        if hg_s is not None:
            ax.axvspan(hg_s, hg_e, color="#C9E4C5", alpha=0.35, zorder=0)
        ax.plot(t, eta, color=color, lw=0.5)
        ax.set_ylabel(r"$\eta$ [mm]")
        ax.set_ylim(global_ylim)
        ax.grid(alpha=0.25)
        ax.text(0.015, 0.86, label, transform=ax.transAxes,
                fontsize=9, ha="left", va="top",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))

    axes[-1].set_xlabel("time [s]")
    fig.suptitle(
        f"Wavetrain travel along the tank — {f_hz:.2f} Hz, per240, nowind, fullpanel",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])

    for ext in ("pdf", "pgf"):
        fig.savefig(OUT_DIR / f"probe_travel_stack.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ probe_travel_stack")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — Wind vs no wind (side-by-side)
# ─────────────────────────────────────────────────────────────────────────────
def fig_wind_vs_nowind() -> None:
    pos = WIND_OFF["in_position"]  # 9373/170 — fully exposed to wind
    f_hz = WIND_OFF["WaveFrequencyInput [Hz]"]

    t_off, eta_off = _probe_signal(WIND_OFF, pos)
    t_on,  eta_on  = _probe_signal(WIND_ON,  pos)
    hg_s_off, hg_e_off = _hg_window(WIND_OFF, pos)
    hg_s_on,  hg_e_on  = _hg_window(WIND_ON,  pos)

    t_max = min(t_off[-1], t_on[-1])
    y_lo = min(float(np.nanpercentile(eta_off, 0.5)),
               float(np.nanpercentile(eta_on,  0.5)))
    y_hi = max(float(np.nanpercentile(eta_off, 99.5)),
               float(np.nanpercentile(eta_on,  99.5)))
    pad  = 0.12 * (y_hi - y_lo)

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True, sharey=True)

    for ax, t, eta, label, wind_key, hg in zip(
        axes,
        (t_off, t_on),
        (eta_off, eta_on),
        ("no wind", "full wind"),
        ("no", "full"),
        ((hg_s_off, hg_e_off), (hg_s_on, hg_e_on)),
    ):
        color = WIND_COLOR_MAP[wind_key]
        if hg[0] is not None:
            ax.axvspan(hg[0], hg[1], color=color, alpha=0.08, zorder=0)
        ax.plot(t, eta, color=color, lw=0.5)
        ax.axhline(0, color="grey", lw=0.4, alpha=0.6)
        ax.set_ylabel(r"$\eta$ [mm]")
        ax.grid(alpha=0.25)
        ax.set_xlim(0, t_max)
        ax.set_ylim(y_lo - pad, y_hi + pad)
        ax.text(0.99, 0.93, label, transform=ax.transAxes,
                fontsize=10, ha="right", va="top", weight="bold",
                color=color,
                bbox=dict(facecolor="white", edgecolor=color, alpha=0.85, pad=3))

    axes[-1].set_xlabel("time [s]")
    fig.suptitle(
        rf"Wind riding on a {f_hz:.2f}\,Hz paddle wave — IN probe {pos}, per40, 0.2\,V",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])

    for ext in ("pdf", "pgf"):
        fig.savefig(OUT_DIR / f"wind_vs_nowind.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ wind_vs_nowind")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — Signal and Hilbert envelope (IN vs OUT)
# ─────────────────────────────────────────────────────────────────────────────
def fig_signal_envelope() -> None:
    in_pos  = ANATOMY["in_position"]      # 9373/170
    out_pos = ANATOMY["out_position"]     # 12400/250
    f_hz    = ANATOMY["WaveFrequencyInput [Hz]"]

    t_in,  eta_in  = _probe_signal(ANATOMY, in_pos)
    t_out, eta_out = _probe_signal(ANATOMY, out_pos)
    hg_s_in,  hg_e_in  = _hg_window(ANATOMY, in_pos)
    hg_s_out, hg_e_out = _hg_window(ANATOMY, out_pos)

    # Envelope via Hilbert transform on interpolated (NaN-free) signal.
    # Short residual NaNs get linearly filled so hilbert() does not propagate.
    def _envelope(x: np.ndarray) -> np.ndarray:
        m = ~np.isnan(x)
        if not m.all():
            idx = np.arange(len(x))
            x = np.interp(idx, idx[m], x[m])
        return np.abs(hilbert(x))

    env_in  = _envelope(eta_in)
    env_out = _envelope(eta_out)

    fig, (ax_in, ax_out) = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True)

    for ax, t, eta, env, hg, title, color in [
        (ax_in,  t_in,  eta_in,  env_in,  (hg_s_in,  hg_e_in),
         f"IN probe {in_pos}",      "#1F77B4"),
        (ax_out, t_out, eta_out, env_out, (hg_s_out, hg_e_out),
         f"OUT probe {out_pos}",    "#D62728"),
    ]:
        if hg[0] is not None:
            ax.axvspan(hg[0], hg[1], color=color, alpha=0.08, zorder=0)
        ax.plot(t, eta, color=color, lw=0.3, alpha=0.55, label="signal")
        ax.plot(t,  env, color=color, lw=1.3, label="envelope")
        ax.plot(t, -env, color=color, lw=1.3)
        ax.axhline(0, color="grey", lw=0.4, alpha=0.6)
        ax.set_ylabel(r"$\eta$ [mm]")
        ax.grid(alpha=0.25)
        ax.text(0.015, 0.92, title, transform=ax.transAxes,
                fontsize=10, ha="left", va="top", weight="bold", color=color,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2))

    # Share y so envelope shrinkage is visually obvious
    ylim = (min(ax_in.get_ylim()[0], ax_out.get_ylim()[0]),
            max(ax_in.get_ylim()[1], ax_out.get_ylim()[1]))
    ax_in.set_ylim(ylim); ax_out.set_ylim(ylim)

    # Annotate the envelope ratio at mid-H&G window
    if hg_s_in is not None and hg_s_out is not None:
        mid_t = 0.5 * (hg_s_out + hg_e_out)
        mid_idx = int(round(mid_t * FS))
        a_in  = float(np.nanmedian(env_in [int(hg_s_out*FS):int(hg_e_out*FS)]))
        a_out = float(np.nanmedian(env_out[int(hg_s_out*FS):int(hg_e_out*FS)]))
        ratio = a_out / a_in if a_in > 0 else float("nan")
        ax_out.annotate(
            rf"envelope ratio $A_{{\mathrm{{out}}}} / A_{{\mathrm{{in}}}} \approx {ratio:.2f}$",
            xy=(mid_t, a_out), xytext=(mid_t + 20, a_out + 0.35 * a_out),
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="black", lw=0.6),
        )

    ax_out.set_xlabel("time [s]")
    fig.suptitle(
        rf"Carrier + Hilbert envelope — {f_hz:.2f}\,Hz, per240, nowind, fullpanel",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])

    for ext in ("pdf", "pgf"):
        fig.savefig(OUT_DIR / f"signal_and_envelope.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ signal_and_envelope")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5 — Anatomy, per40 variant (cleaner proportions, labels above axes)
# ─────────────────────────────────────────────────────────────────────────────
def fig_anatomy_per40() -> None:
    """Same idea as fig_anatomy() but on a per40 run where the H&G [50T, 60T]
    analysis window covers most of the stable wavetrain — so shading and
    labels actually match the physics, and nothing overlaps."""
    row = find_run("fullpanel-nowind-amp0200-freq1400-per40-depth580-mstop30-run1.csv")
    pos = row["in_position"]
    t, eta = _probe_signal(row, pos)
    hg_s, hg_e = _hg_window(row, pos)
    f_hz = row["WaveFrequencyInput [Hz]"]

    fig = plt.figure(figsize=(7.4, 3.8))
    ax = fig.add_subplot(111)
    ax.plot(t, eta, color="#17375E", lw=0.55)

    y_top = float(np.nanpercentile(eta, 99.7)) * 1.15
    y_bot = float(np.nanpercentile(eta,  0.3)) * 1.15
    ax.set_ylim(y_bot, y_top)

    # Shade ramp / stable / decay. H&G window highlighted as a darker stripe.
    if hg_s is not None:
        ax.axvspan(0,    hg_s,   color="#FFE8B3", alpha=0.55, zorder=0)
        ax.axvspan(hg_s, hg_e,   color="#7FCB8A", alpha=0.40, zorder=0)
        ax.axvspan(hg_e, t[-1],  color="#F2C8C8", alpha=0.45, zorder=0)

    # Region labels ABOVE the axis so they never touch the signal.
    def _region_label(x: float, txt: str, color: str) -> None:
        ax.annotate(txt, xy=(x, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(0, 8), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, color=color,
                    annotation_clip=False)

    if hg_s is not None:
        _region_label(hg_s/2,           "ramp-up",            "#805600")
        _region_label((hg_s+hg_e)/2,    "H-and-G window",     "#0E5614")
        _region_label((hg_e+t[-1])/2,   "paddle off \u2192 decay",    "#802020")

    # Connector ticks under the region labels
    for x in [0, hg_s, hg_e, t[-1]]:
        ax.axvline(x, ymin=1.0, ymax=1.03, clip_on=False,
                   color="grey", lw=0.6)

    ax.set_xlabel("time [s]")
    ax.set_ylabel(r"$\eta$ [mm]")
    ax.set_title(
        f"Three phases of a run — probe {pos}, f = {f_hz:.2f} Hz, per40, nowind, fullpanel",
        pad=26,
    )
    ax.grid(alpha=0.25)

    for ext in ("pdf", "pgf"):
        fig.savefig(OUT_DIR / f"anatomy_of_a_run_per40.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ anatomy_of_a_run_per40")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 6 — Frequency ladder (stacked sweep)
# ─────────────────────────────────────────────────────────────────────────────
def fig_frequency_ladder() -> None:
    """Stack one OUT-probe record per frequency with a constant vertical
    offset — wavelength shrinks, amplitude drops visibly as f rises."""
    target_freqs = [0.9, 1.1, 1.3, 1.5, 1.7]
    runs = []
    for f in target_freqs:
        q = meta[(meta["WaveFrequencyInput [Hz]"] == f)
                 & (meta["WindCondition"] == "no")
                 & (meta["PanelCondition"] == "full")
                 & (meta["WaveAmplitudeInput [Volt]"] == 0.2)
                 & (meta["WavePeriodInput"] == 40)
                 & (meta.get("quality_flag", "ok") == "ok")]
        if len(q) == 0:
            continue
        runs.append((f, q.iloc[0]))
    if not runs:
        print("  (no frequency-ladder runs found — skipped)")
        return

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(runs)))

    # Pick a common xlim so every trace starts at ramp end and shows ~20 periods.
    # We align each run to its H&G window start so the visual rhythm syncs.
    n_show_periods = 20
    offset_step = 35.0  # mm between traces (headroom for amp ~15 mm)
    for i, ((f, row), color) in enumerate(zip(runs, colors)):
        pos = row["out_position"]
        t, eta = _probe_signal(row, pos)
        hg_s, _ = _hg_window(row, pos)
        if hg_s is None:
            continue
        # Align: use t' = t - hg_s so every trace starts at 0 at its analysis window
        t_rel = t - hg_s
        mask = (t_rel >= -2.0) & (t_rel <= n_show_periods / f)
        y = eta[mask] + i * offset_step
        ax.plot(t_rel[mask], y, color=color, lw=0.6)
        ax.text(n_show_periods / f + 0.2, i * offset_step,
                f"{f:.1f} Hz", color=color, fontsize=9, va="center",
                weight="bold")
        ax.axhline(i * offset_step, color="grey", lw=0.3, alpha=0.25)

    ax.set_xlabel(r"time relative to H-and-G window start [s]")
    ax.set_ylabel(r"$\eta$ [mm] (offset per frequency)")
    ax.set_title(
        "Frequency sweep at the OUT probe — 0.9 \u2192 1.7 Hz, "
        "nowind, 0.2 V, fullpanel, per40",
        pad=8,
    )
    ax.set_yticks([])
    ax.grid(axis="x", alpha=0.25)
    ax.set_xlim(-2.0, n_show_periods / min(f for f, _ in runs) + 2.5)

    for ext in ("pdf", "pgf"):
        fig.savefig(OUT_DIR / f"frequency_ladder.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ frequency_ladder")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 7 — Wind ON vs OFF, overlaid on one axis
# ─────────────────────────────────────────────────────────────────────────────
def fig_wind_overlay() -> None:
    pos = WIND_OFF["in_position"]
    f_hz = WIND_OFF["WaveFrequencyInput [Hz]"]
    t_off, eta_off = _probe_signal(WIND_OFF, pos)
    t_on,  eta_on  = _probe_signal(WIND_ON,  pos)

    t_max = min(t_off[-1], t_on[-1])

    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ax.plot(t_on,  eta_on,  color=WIND_COLOR_MAP["full"], lw=0.45, alpha=0.75,
            label="full wind")
    ax.plot(t_off, eta_off, color=WIND_COLOR_MAP["no"],   lw=0.45, alpha=0.85,
            label="no wind")
    ax.axhline(0, color="grey", lw=0.4, alpha=0.6)
    ax.set_xlim(0, t_max)
    ax.set_xlabel("time [s]")
    ax.set_ylabel(r"$\eta$ [mm]")
    ax.set_title(
        rf"Wind rides on top of the paddle wave — IN probe {pos}, {f_hz:.2f}\,Hz, 0.2\,V, per40",
        pad=6,
    )
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", framealpha=0.9)

    # Zoom inset: 3 periods of each, mid-analysis-window
    hg_s, hg_e = _hg_window(WIND_OFF, pos)
    if hg_s is not None:
        t0 = hg_s + 1.0
        t1 = t0 + 3 / f_hz
        ax_in = ax.inset_axes([0.55, 0.58, 0.42, 0.38])
        for t, eta, color in [
            (t_off, eta_off, WIND_COLOR_MAP["no"]),
            (t_on,  eta_on,  WIND_COLOR_MAP["full"]),
        ]:
            m = (t >= t0) & (t <= t1)
            ax_in.plot(t[m], eta[m], color=color, lw=0.9)
        ax_in.set_title(f"3 periods @ {f_hz:.2f} Hz", fontsize=8, pad=2)
        ax_in.tick_params(labelsize=7)
        ax_in.set_ylabel(r"$\eta$ [mm]", fontsize=8)
        ax_in.grid(alpha=0.3)

    for ext in ("pdf", "pgf"):
        fig.savefig(OUT_DIR / f"wind_overlay.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ wind_overlay")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 8 — IN and OUT overlaid (damping in one glance)
# ─────────────────────────────────────────────────────────────────────────────
def fig_in_out_overlay() -> None:
    in_pos  = ANATOMY["in_position"]
    out_pos = ANATOMY["out_position"]
    f_hz    = ANATOMY["WaveFrequencyInput [Hz]"]
    t_in,  eta_in  = _probe_signal(ANATOMY, in_pos)
    t_out, eta_out = _probe_signal(ANATOMY, out_pos)
    hg_s, hg_e     = _hg_window(ANATOMY, in_pos)

    # Zoom inside the stable train
    if hg_s is None:
        print("  (no H&G window on anatomy run — skipped)")
        return
    t0 = hg_s + 1.0
    t1 = t0 + 6 / f_hz

    fig, (ax_all, ax_zoom) = plt.subplots(
        2, 1, figsize=(7.4, 5.0),
        gridspec_kw={"height_ratios": [1.6, 1.0]}, sharey=False,
    )

    for ax, a, b in [(ax_all, 0, t_in[-1]), (ax_zoom, t0, t1)]:
        m_in  = (t_in  >= a) & (t_in  <= b)
        m_out = (t_out >= a) & (t_out <= b)
        ax.plot(t_in[m_in],   eta_in[m_in],   color="#1F77B4",
                lw=0.6, label=f"IN {in_pos}",  alpha=0.85)
        ax.plot(t_out[m_out], eta_out[m_out], color="#D62728",
                lw=0.6, label=f"OUT {out_pos}", alpha=0.85)
        ax.axhline(0, color="grey", lw=0.4, alpha=0.6)
        ax.grid(alpha=0.25)
        ax.set_xlim(a, b)
        ax.set_ylabel(r"$\eta$ [mm]")

    ax_all.axvspan(t0, t1, color="orange", alpha=0.18, zorder=0)
    ax_all.set_title(
        f"Same wave, before and after the panel — "
        f"{f_hz:.2f} Hz, per240, nowind, fullpanel",
        pad=6,
    )
    ax_all.legend(loc="upper right", framealpha=0.9)
    ax_zoom.set_xlabel("time [s]")
    ax_zoom.set_title("Zoom: 6 periods inside the stable train",
                      fontsize=9, pad=4)

    for ext in ("pdf", "pgf"):
        fig.savefig(OUT_DIR / f"in_out_overlay.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ in_out_overlay")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 9 — Spectrogram (wind waves riding on paddle wave)
# ─────────────────────────────────────────────────────────────────────────────
def fig_spectrogram() -> None:
    """Short-time FFT showing the paddle-freq band steady in time while wind
    adds broadband 2–6 Hz energy only when the fan is on."""
    from scipy.signal import spectrogram
    pos = WIND_ON["in_position"]
    f_hz = WIND_ON["WaveFrequencyInput [Hz]"]
    t_off, eta_off = _probe_signal(WIND_OFF, pos)
    t_on,  eta_on  = _probe_signal(WIND_ON,  pos)

    def _fill_nan(x: np.ndarray) -> np.ndarray:
        m = ~np.isnan(x)
        if m.all():
            return x
        idx = np.arange(len(x))
        return np.interp(idx, idx[m], x[m])

    nperseg = int(2.0 * FS)   # 2-second windows
    noverlap = nperseg // 2

    fig, axes = plt.subplots(2, 1, figsize=(7.4, 5.2), sharex=True, sharey=True)
    import matplotlib.colors as mcolors

    for ax, t, eta, title in [
        (axes[0], t_off, eta_off, "no wind"),
        (axes[1], t_on,  eta_on,  "full wind"),
    ]:
        f_arr, t_arr, Sxx = spectrogram(
            _fill_nan(eta), fs=FS,
            window="hann", nperseg=nperseg, noverlap=noverlap,
            scaling="density",
        )
        # Clip to 0–8 Hz (paddle + wind-wave band)
        mask = f_arr <= 8.0
        im = ax.pcolormesh(
            t_arr, f_arr[mask], 10 * np.log10(Sxx[mask] + 1e-12),
            cmap="magma", norm=mcolors.Normalize(vmin=-40, vmax=20),
            shading="auto",
        )
        ax.axhline(f_hz, color="cyan", lw=0.8, alpha=0.7, linestyle="--",
                   label=f"paddle {f_hz:.2f} Hz")
        ax.text(0.015, 0.93, title, transform=ax.transAxes,
                fontsize=10, ha="left", va="top", weight="bold", color="white",
                bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=2))
        ax.set_ylabel("frequency [Hz]")
        ax.legend(loc="upper right", framealpha=0.9, fontsize=8)

    axes[-1].set_xlabel("time [s]")
    fig.suptitle(
        rf"Spectrogram — IN probe {pos}, {f_hz:.2f}\,Hz, 0.2\,V, per40",
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.975])

    # Shared colorbar
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(),
                        orientation="vertical", pad=0.02, shrink=0.9)
    cbar.set_label("PSD [dB re mm$^2$/Hz]")

    for ext in ("pdf", "pgf"):
        fig.savefig(OUT_DIR / f"spectrogram_wind_vs_nowind.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ spectrogram_wind_vs_nowind")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*",
                        help="Run only named figures (anatomy, travel, wind, env, "
                             "anatomy_per40, ladder, overlay, inout, spectrogram).")
    args = parser.parse_args()

    ALL = {
        "anatomy":          fig_anatomy,
        "travel":           fig_travel_stack,
        "wind":             fig_wind_vs_nowind,
        "env":              fig_signal_envelope,
        "anatomy_per40":    fig_anatomy_per40,
        "ladder":           fig_frequency_ladder,
        "overlay":          fig_wind_overlay,
        "inout":            fig_in_out_overlay,
        "spectrogram":      fig_spectrogram,
    }

    print(f"Writing figures to {OUT_DIR}/")
    names = args.only or list(ALL)
    for name in names:
        if name not in ALL:
            print(f"  (unknown figure {name!r})")
            continue
        ALL[name]()
    print(f"done ({len(names)} figure(s)).")
