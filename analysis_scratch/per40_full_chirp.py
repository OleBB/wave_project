"""
Per40 full-train chirp characterization (2026-05-05)
====================================================

Companion to pre_7T_wave_chirp.py. Per40 runs (40 periods of paddle
motion + ramp-down) let us see BOTH the ramp-up chirp at the start AND
the ramp-down at the end. The pre-paddle chirp study only covered
ramp-up; per40 closes the loop.

Method (identical detector to pre_7T_wave_chirp.py):
  - 25-sample (100 ms) rolling-mean smooth on η at IN (9373/170)
  - Zero upcrossings with time-lockout = T_paddle/4 — accepts low-amp
    real cycles, rejects ripple-on-zero artefacts
  - Per-cycle period T_i and per-cycle amplitude (max−min)/2
  - Plot full window 0 to ~60 s

Comparison: fullwind vs nowind, overlaid for one canonical run each.

Cells: 1.3-1.6 Hz × 0.1-0.3 V (canon March-2026 lowrange, full panel)

Outputs:
  analysis_scratch/per40_full_chirp_detail_<freq>Hz_<amp>V.png  one per cell
  analysis_scratch/per40_full_chirp_grid_period.png             4×3 T_i grid
  analysis_scratch/per40_full_chirp_settle_decay.csv            settle + decay times
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from wavescripts.plot_utils import apply_thesis_style  # noqa: E402
apply_thesis_style()

OUT_DIR = BASE / "analysis_scratch"
FS = 250.0

CANON = [
    BASE / "waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange",
    BASE / "waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange",
]

FREQS = [1.3, 1.4, 1.5, 1.6]
AMPS  = [0.1, 0.2, 0.3]
WINDS = ["no", "full"]
PROBE = "9373/170"
WIND_COLOR = {"no": "#1f77b4", "full": "#d62728"}
WIND_LABEL = {"no": "Uten vind", "full": "Full vind"}

PROBE_LABEL_NO = {
    "8804/250":  "Foran",
    "9373/170":  "Innkommende",
    "9373/340":  "Innkommende",
    "12400/250": "Utgående",
}
AMP_LABEL = {0.1: r"$A_1$", 0.2: r"$A_2$", 0.3: r"$A_3$"}

# A4 width minus 1 inch margin on each side → matches \linewidth in thesis.
A4_W_IN = 8.27
FIG_W   = A4_W_IN - 2.0  # 6.27"


def _format_info_box(chosen, f_hz, amp, probe):
    """Info-box string: probe label, amp/freq, Δt in 7T–17T window.

    Δt = (start_fw − start_nw)/FS, read from `Computed Probe {pos} start`
    in the per-run meta — this is the snap-aligned anchor of the 10-period
    analysis window. Negative Δt = fullwind window starts earlier.
    """
    start_col = f"Computed Probe {probe} start"
    fw = chosen.get((f_hz, amp, "full"))
    nw = chosen.get((f_hz, amp, "no"))
    if fw is None or nw is None or pd.isna(fw.get(start_col)) or pd.isna(nw.get(start_col)):
        dt_str = r"$\Delta t$ = n/a"
    else:
        dt_ms = (float(fw[start_col]) - float(nw[start_col])) / FS * 1000.0
        tag = "fullvind tidligere" if dt_ms < 0 else "fullvind senere"
        dt_str = rf"$\Delta t$ = {dt_ms:+.0f} ms ({tag})"
    return (
        f"{PROBE_LABEL_NO[probe]} ({probe})\n"
        f"{AMP_LABEL[amp]}, {f_hz:.1f} Hz\n"
        f"{dt_str}"
    )

SETTLE_TOL = 0.10        # ±10% of T_paddle
SETTLE_RUN = 3           # 3 consecutive cycles in band
DECAY_RUN  = 3           # 3 consecutive cycles OUT of band (ramp-down)
WIN_END_S  = 60.0        # analysis window end


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_canon_meta() -> pd.DataFrame:
    rows = []
    for d in CANON:
        for r in json.load(open(d / "meta.json")):
            r["_folder"] = d.name
            rows.append(r)
    df = pd.DataFrame(rows)
    df["fname"] = df["path"].str.split("/").str[-1]
    df = df[df["fname"].str.contains("per40", case=False, regex=False)]
    df = df[df["WaveFrequencyInput [Hz]"].isin(FREQS)]
    df = df[df["WaveAmplitudeInput [Volt]"].isin(AMPS)]
    df = df[df["PanelCondition"] == "full"]
    df = df[df["quality_flag"] == "ok"]
    df = df[df["WindCondition"].isin(WINDS)]
    return df.copy()


def load_processed():
    print("Loading processed_dfs from canon (~45s)...")
    dfs = [pd.read_parquet(d / "processed_dfs.parquet") for d in CANON]
    big = pd.concat(dfs, ignore_index=True)
    print(f"  {len(big):,} rows across {big['_path'].nunique()} runs")
    return big


# ---------------------------------------------------------------------------
# Detector — same as pre_7T_wave_chirp.py
# ---------------------------------------------------------------------------
def _light_smooth(eta, win_samples=25):
    if win_samples <= 1:
        return eta
    kern = np.ones(win_samples) / win_samples
    return np.convolve(eta, kern, mode="same")


def detect_upcrossings(eta, lockout_samples=0):
    s = eta - np.nanmean(eta[:int(3 * FS)])
    ucs, last = [], -lockout_samples - 1
    for i in range(1, len(s)):
        if s[i - 1] <= 0 < s[i] and (i - last) > lockout_samples:
            ucs.append(i)
            last = i
    return np.array(ucs, dtype=int)


def lockout_for_paddle(t_paddle, fraction=0.25):
    return int(round(fraction * t_paddle * FS))


def per_cycle_metrics(eta, ucs):
    rows = []
    for i in range(len(ucs) - 1):
        i0, i1 = ucs[i], ucs[i + 1]
        cycle = eta[i0:i1]
        T_i = (i1 - i0) / FS
        amp_i = (np.nanmax(cycle) - np.nanmin(cycle)) / 2 if len(cycle) >= 2 else np.nan
        rows.append({"uc_idx": i0, "uc_time": i0 / FS,
                     "T_i": T_i, "amp_i": amp_i})
    return pd.DataFrame(rows)


def time_to_settle(metrics, T_paddle, tol=SETTLE_TOL, run=SETTLE_RUN):
    """First uc where T_i ∈ ±tol of T_pad for `run` consecutive cycles."""
    rel = (metrics["T_i"] - T_paddle).abs() / T_paddle
    in_band = rel < tol
    for i in range(len(in_band) - run + 1):
        if in_band.iloc[i:i + run].all():
            return float(metrics["uc_time"].iloc[i])
    return np.nan


def time_to_decay(metrics, T_paddle, t_settle, tol=SETTLE_TOL, run=DECAY_RUN):
    """First uc AFTER t_settle where T_i exits ±tol band for `run` consecutive
    cycles (ramp-down). Returns NaN if no decay detected."""
    if not np.isfinite(t_settle):
        return np.nan
    after = metrics[metrics["uc_time"] > t_settle].reset_index(drop=True)
    rel = (after["T_i"] - T_paddle).abs() / T_paddle
    out_band = rel >= tol
    for i in range(len(out_band) - run + 1):
        if out_band.iloc[i:i + run].all():
            return float(after["uc_time"].iloc[i])
    return np.nan


# ---------------------------------------------------------------------------
# Pick one canonical run per cell × wind
# ---------------------------------------------------------------------------
def pick_runs(meta):
    chosen = {}
    for f in FREQS:
        for a in AMPS:
            for w in WINDS:
                cell = meta[(meta["WaveFrequencyInput [Hz]"] == f)
                            & (meta["WaveAmplitudeInput [Volt]"] == a)
                            & (meta["WindCondition"] == w)]
                if not cell.empty:
                    chosen[(f, a, w)] = cell.iloc[0]
    return chosen


# ---------------------------------------------------------------------------
# Detail plot — one cell, fullwind vs nowind, 3 rows × 2 cols
# ---------------------------------------------------------------------------
def plot_detail(meta, big, chosen, out_path, f_hz, amp):
    eta_col = f"eta_{PROBE}"
    T_paddle = 1.0 / f_hz
    end_idx = int(WIN_END_S * FS)

    fig, axes = plt.subplots(3, 2, figsize=(16, 9), sharex="col", sharey="row")
    for col, w in enumerate(WINDS):
        key = (f_hz, amp, w)
        if key not in chosen:
            continue
        row = chosen[key]
        eta = big[big["_path"] == row["path"]][eta_col].values
        eta = eta[:end_idx]
        t = np.arange(len(eta)) / FS
        eta_sm = _light_smooth(eta)
        lock = lockout_for_paddle(T_paddle)
        ucs = detect_upcrossings(eta_sm, lockout_samples=lock)
        metrics = per_cycle_metrics(eta, ucs)
        t_settle = time_to_settle(metrics, T_paddle)
        t_decay  = time_to_decay(metrics, T_paddle, t_settle)

        clr = WIND_COLOR[w]

        # Row 0: η(t)
        ax = axes[0, col]
        ax.plot(t, eta, color=clr, lw=0.5, alpha=0.4, label="raw")
        ax.plot(t, eta_sm, color=clr, lw=0.7, label="smoothed (25-sample)")
        ax.axhline(0, color="k", lw=0.3)
        for uc in ucs:
            ax.axvline(uc / FS, color="grey", lw=0.2, alpha=0.3)
        if np.isfinite(t_settle):
            ax.axvline(t_settle, color="orange", ls="--", lw=1.0,
                       label=f"settle ({t_settle:.1f}s)")
        if np.isfinite(t_decay):
            ax.axvline(t_decay, color="purple", ls="--", lw=1.0,
                       label=f"decay ({t_decay:.1f}s)")
        steady_periods = (t_decay - t_settle) * f_hz if (
            np.isfinite(t_settle) and np.isfinite(t_decay)) else np.nan
        ax.set_title(f"{w} — {f_hz} Hz, {amp} V, per40  "
                     f"(steady ≈ {steady_periods:.1f} T)" if np.isfinite(steady_periods)
                     else f"{w} — {f_hz} Hz, {amp} V, per40", fontsize=11)
        ax.set_ylabel("η [mm]")
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(alpha=0.3)

        # Row 1: T_i(t)
        ax = axes[1, col]
        ax.plot(metrics["uc_time"], metrics["T_i"], "o-", color=clr,
                lw=0.6, ms=3)
        ax.axhline(T_paddle, color="k", lw=0.5,
                   label=f"T_pad = {T_paddle*1000:.0f} ms")
        ax.axhline(T_paddle * (1 + SETTLE_TOL), color="grey", lw=0.3, ls=":")
        ax.axhline(T_paddle * (1 - SETTLE_TOL), color="grey", lw=0.3, ls=":")
        if np.isfinite(t_settle):
            ax.axvline(t_settle, color="orange", ls="--", lw=1.0)
        if np.isfinite(t_decay):
            ax.axvline(t_decay, color="purple", ls="--", lw=1.0)
        ax.set_ylabel("T_i [s]")
        ax.set_ylim(0, 5)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Row 2: amp_i(t)
        ax = axes[2, col]
        ax.plot(metrics["uc_time"], metrics["amp_i"], "o-", color=clr,
                lw=0.6, ms=3)
        if np.isfinite(t_settle):
            ax.axvline(t_settle, color="orange", ls="--", lw=1.0)
        if np.isfinite(t_decay):
            ax.axvline(t_decay, color="purple", ls="--", lw=1.0)
        ax.set_ylabel("per-cycle (max−min)/2 [mm]")
        ax.set_xlabel("time [s]")
        ax.set_xlim(0, WIN_END_S)
        ax.grid(alpha=0.3)

    fig.suptitle(f"Per40 full chirp at IN ({PROBE})  —  f={f_hz} Hz, A={amp} V\n"
                 f"orange = settle (chirp → T_pad), purple = decay (T_pad → ramp-down)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Overlay zoom — fw vs nw on same axes, two time windows
# ---------------------------------------------------------------------------
def plot_overlay_zoom(meta, big, chosen, out_path,
                      f_hz, amp, t_start, t_end, label,
                      thesis_pdf=None, thesis_name=None,
                      info_loc="upper left"):
    eta_col = f"eta_{PROBE}"
    T_paddle = 1.0 / f_hz

    fig, ax = plt.subplots(figsize=(FIG_W, 3.2))
    fig.subplots_adjust(left=0.055, right=0.995, top=0.93, bottom=0.16)

    for w in WINDS:
        key = (f_hz, amp, w)
        if key not in chosen:
            continue
        row = chosen[key]
        eta = big[big["_path"] == row["path"]][eta_col].values
        eta_sm = _light_smooth(eta)
        t = np.arange(len(eta)) / FS

        i0 = max(0, int(t_start * FS))
        i1 = min(len(eta), int(t_end * FS))

        # raw (faint) — no legend entry
        ax.plot(t[i0:i1], eta[i0:i1], color=WIND_COLOR[w], lw=0.5,
                alpha=0.35)
        # smoothed — legend entry per wind condition
        ax.plot(t[i0:i1], eta_sm[i0:i1], color=WIND_COLOR[w], lw=1.0,
                label=WIND_LABEL[w])

        # Paddle-period grid: vertical lines every T_pad anchored to the
        # smoothed signal's first upcrossing inside the visible window.
        eta_win = eta_sm[i0:i1]
        signs = np.sign(eta_win - np.nanmean(eta_win))
        diff = np.diff(signs)
        first_uc = np.where(diff > 0)[0]
        if len(first_uc):
            t0_uc = (i0 + first_uc[0]) / FS
            for k in range(int((t_end - t0_uc) / T_paddle) + 2):
                ax.axvline(t0_uc + k * T_paddle, color=WIND_COLOR[w],
                           lw=0.4, alpha=0.15)

    ax.axhline(0, color="k", lw=0.3)
    ax.set_xlim(t_start, t_end)
    ax.set_xlabel("Tid [s]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")

    # Info box — probe / amp / freq / Δt over 7T–17T window
    info_xy = {"upper left":   (0.012, 0.96, "left"),
               "upper center": (0.5,   0.96, "center"),
               "upper right":  (0.988, 0.96, "right")}[info_loc]
    ax.text(info_xy[0], info_xy[1], _format_info_box(chosen, f_hz, amp, PROBE),
            transform=ax.transAxes, fontsize=8, va="top", ha=info_xy[2],
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="grey", alpha=0.9))

    # y-axis label, lifted to figure top-left corner
    fig.text(0.006, 0.985, r"$\eta$ [mm]", fontsize=10,
             va="top", ha="left")

    # Force the saved PDF to match figsize exactly (apply_thesis_style sets
    # savefig.bbox='tight', which would otherwise crop and shrink the page).
    fig.savefig(out_path, dpi=140, bbox_inches=fig.bbox_inches)
    if thesis_pdf is not None:
        thesis_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(thesis_pdf, bbox_inches=fig.bbox_inches)
    plt.close(fig)

    if thesis_name is not None:
        _write_overlay_stub(meta, chosen, f_hz, amp, t_start, t_end,
                            label, thesis_name)


def _write_overlay_stub(meta, chosen, f_hz, amp, t_start, t_end,
                        label, figure_name):
    import wavescripts.plot_utils as pu
    pu.TEXFIGU_DIR = BASE / "output" / "TEXFIGU"
    pu.FIGURES_DIR = BASE / "output" / "FIGURES"

    # Pull contributing run paths so stub records them
    nw_run = chosen.get((f_hz, amp, "no"))
    fw_run = chosen.get((f_hz, amp, "full"))
    paths = []
    if nw_run is not None:
        paths.append(nw_run["path"])
    if fw_run is not None:
        paths.append(fw_run["path"])

    _meta = pu.build_fig_meta(
        {
            "filters": {
                "PanelCondition":            "full",
                "WaveFrequencyInput [Hz]":   f_hz,
                "WaveAmplitudeInput [Volt]": amp,
                "WindCondition":             ["no", "full"],
                "quality_flag":              "ok",
                "run_type":                  "per40",
                "probes":                    PROBE,
            },
            "plotting": {
                "figure_name": figure_name,
            },
        },
        chapter="04",
        extra={"script": "analysis_scratch/per40_full_chirp.py"},
        computed_in=("analysis_scratch/per40_full_chirp.py "
                     "(eta_{IN} overlay; raw + 25-sample rolling-mean smoothed; "
                     "paddle-period grid anchored at first uc per condition)"),
        data_class="DELEG",
        findings_doc="memory/methodology_wind_enhances_A_in.md",
        extra_params=(
            f"zoom_window_s=({t_start}, {t_end}). label={label}. "
            f"smoother=25 samples (100 ms). "
            f"runs: nw={nw_run['path'] if nw_run is not None else None}, "
            f"fw={fw_run['path'] if fw_run is not None else None}."
        ),
        max_run_paths=4,
    )
    pu.write_figure_stub(_meta, plot_type="per40_overlay",
                         subfig_filenames=[figure_name], force=True)
    print(f"   stub → output/TEXFIGU/{figure_name}.tex")


# ---------------------------------------------------------------------------
# Period grid — all 12 cells, fw vs nw overlaid, T_i panel only
# ---------------------------------------------------------------------------
def plot_period_grid(meta, big, chosen, out_path, settle_table):
    eta_col = f"eta_{PROBE}"
    end_idx = int(WIN_END_S * FS)

    fig, axes = plt.subplots(len(FREQS), len(AMPS), figsize=(16, 12),
                             sharex=True)
    for i, f_hz in enumerate(FREQS):
        T_paddle = 1.0 / f_hz
        for j, amp in enumerate(AMPS):
            ax = axes[i, j]
            for w in WINDS:
                key = (f_hz, amp, w)
                if key not in chosen:
                    continue
                row = chosen[key]
                eta = big[big["_path"] == row["path"]][eta_col].values
                eta = eta[:end_idx]
                eta_sm = _light_smooth(eta)
                lock = lockout_for_paddle(T_paddle)
                ucs = detect_upcrossings(eta_sm, lockout_samples=lock)
                metrics = per_cycle_metrics(eta, ucs)
                t_settle = time_to_settle(metrics, T_paddle)
                t_decay  = time_to_decay(metrics, T_paddle, t_settle)
                settle_table.append({
                    "f_hz": f_hz, "amp_V": amp, "wind": w,
                    "t_settle": t_settle, "t_decay": t_decay,
                    "steady_periods": (t_decay - t_settle) * f_hz
                                       if np.isfinite(t_settle) and np.isfinite(t_decay)
                                       else np.nan,
                })

                ax.plot(metrics["uc_time"], metrics["T_i"], "o-",
                        color=WIND_COLOR[w], lw=0.6, ms=3, alpha=0.85,
                        label=w)
                if np.isfinite(t_settle):
                    ax.axvline(t_settle, color=WIND_COLOR[w], ls="--",
                               lw=0.6, alpha=0.5)
                if np.isfinite(t_decay):
                    ax.axvline(t_decay, color=WIND_COLOR[w], ls=":",
                               lw=0.6, alpha=0.5)

            ax.axhline(T_paddle, color="k", lw=0.5)
            ax.axhline(T_paddle * (1 + SETTLE_TOL), color="grey", lw=0.3, ls=":")
            ax.axhline(T_paddle * (1 - SETTLE_TOL), color="grey", lw=0.3, ls=":")
            ax.set_title(f"{f_hz} Hz, {amp} V (T_pad={T_paddle*1000:.0f} ms)",
                         fontsize=9)
            ax.set_ylim(0, 5)
            ax.grid(alpha=0.3)
            if j == 0:
                ax.set_ylabel("T_i [s]")
            if i == len(FREQS) - 1:
                ax.set_xlabel("time of upcrossing [s]")
            if i == 0 and j == 0:
                ax.legend(fontsize=8)

    fig.suptitle(f"Per40 instantaneous period T_i(t) at IN ({PROBE})\n"
                 f"dashed = settle (chirp ends), dotted = decay (ramp-down begins)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    print("Loading per40 canon meta...")
    meta = load_canon_meta()
    print(f"  {len(meta)} per40 ok runs after filters")

    chosen = pick_runs(meta)
    print(f"  picked {len(chosen)} canonical runs")

    big = load_processed()

    # Single canonical detail: 1.3 Hz, 0.2 V (matches pre_7T study)
    print("\nDetail plot for 1.3 Hz, 0.2 V...")
    plot_detail(meta, big, chosen,
                OUT_DIR / "per40_full_chirp_detail_1.3Hz_0.2V.png",
                f_hz=1.3, amp=0.2)
    print("  → analysis_scratch/per40_full_chirp_detail_1.3Hz_0.2V.png")

    print("\nOverlay zooms (fw vs nw, raw + smoothed, paddle-period grid)...")
    plot_overlay_zoom(meta, big, chosen,
                      OUT_DIR / "per40_full_chirp_overlay_t10-21.png",
                      f_hz=1.3, amp=0.2, t_start=10, t_end=21,
                      label="ramp-up + settle",
                      thesis_pdf=BASE / "output/FIGURES/ch04_per40_overlay_t10-21.pdf",
                      thesis_name="ch04_per40_overlay_t10-21")
    print("  → analysis_scratch/per40_full_chirp_overlay_t10-21.png "
          "+ output/FIGURES/ch04_per40_overlay_t10-21.pdf")
    plot_overlay_zoom(meta, big, chosen,
                      OUT_DIR / "per40_full_chirp_overlay_t40-51.png",
                      f_hz=1.3, amp=0.2, t_start=40, t_end=51,
                      label="ramp-down + decay",
                      thesis_pdf=BASE / "output/FIGURES/ch04_per40_overlay_t40-51.pdf",
                      thesis_name="ch04_per40_overlay_t40-51",
                      info_loc="upper center")
    print("  → analysis_scratch/per40_full_chirp_overlay_t40-51.png "
          "+ output/FIGURES/ch04_per40_overlay_t40-51.pdf")

    # Grid plot of all cells
    print("\nGrid plot of all cells T_i(t)...")
    settle_rows = []
    plot_period_grid(meta, big, chosen,
                     OUT_DIR / "per40_full_chirp_grid_period.png",
                     settle_rows)
    print("  → analysis_scratch/per40_full_chirp_grid_period.png")

    settle_df = pd.DataFrame(settle_rows)
    csv = OUT_DIR / "per40_full_chirp_settle_decay.csv"
    settle_df.to_csv(csv, index=False)
    print(f"  → {csv.relative_to(BASE)}")

    print("\n=== Settle and decay times per cell ===")
    piv = settle_df.pivot_table(
        index=["f_hz", "amp_V"], columns="wind",
        values=["t_settle", "t_decay", "steady_periods"]
    )
    print(piv.round(1).to_string())


if __name__ == "__main__":
    main()
