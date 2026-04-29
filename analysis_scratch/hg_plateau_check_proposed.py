"""
H&G plateau check — PROPOSED FORMULA prototype (not pipeline yet)
==================================================================

Side-by-side companion to `hg_plateau_check.py`. Same layout (2 rows
× 3 cols per frequency, sliding AFFT + snapped H&G window overlay),
but the window is computed from a NEW proposed formula instead of
read from meta.json:

    t_start = r / c_g(f, h) + N_offset / f       (seconds)
    t_end   = t_start + 10 / f                   (10T length)

with N_OFFSET = 15 periods (= ~5 wavemaker-ramp + 10 H&G safety).
Start is then snapped to nearest zero-upcrossing within ±T of the
theoretical start (same as live pipeline). End = start + 10·T_samples
(no separate end-snap in this prototype — keep it simple).

Visual question: does the proposed formula land the H&G window inside
the user's eyeballed plateau better than the current [50T, 60T]+probe-
shift rule? Especially at 1.3 Hz, where the current rule overshoots
by ~2 T.

Run from repo root:
    /opt/anaconda3/envs/draumkvedet/bin/python analysis_scratch/hg_plateau_check_proposed.py

Outputs (scratch, prototype only — NOT a thesis figure, NOT pipeline):
    analysis_scratch/hg_plateau_check_proposed_f13.{pdf,png}
    analysis_scratch/hg_plateau_check_proposed_f14.{pdf,png}
    analysis_scratch/hg_plateau_check_proposed_f15.{pdf,png}
    analysis_scratch/hg_plateau_check_proposed_f16.{pdf,png}
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import os
os.chdir(BASE)

from wavescripts.improved_data_loader import load_analysis_data, load_processed_dfs
from wavescripts.plot_utils import apply_thesis_style, WIND_COLOR_MAP
from wavescripts.constants import c_group, HG

apply_thesis_style()

# ── Config ──────────────────────────────────────────────────────────────
FS                 = 250.0
FFT_BAND_HZ        = 0.05
TARGET_FREQS       = [1.3, 1.4, 1.5, 1.6]
TARGET_AMP         = 0.2
PER40_PERIODS      = 40
PER240_THRESHOLD_T = 50

SLIDING_STEP_S     = 0.5
SLIDING_T_LO       = 5.0
SLIDING_T_HI       = 80.0

IN_PROBES = ["9373/170", "9373/340"]
OUT_PROBE = "12400/250"

# Probe radial distances [m] — parsed from "DIST/LAT" string convention.
# Canonical IN distance = 9.373 m (both 9373/170 and 9373/340 share it).
PROBE_R_M = {
    "9373/170": 9.373,
    "9373/340": 9.373,
    "12400/250": 12.400,
}

# === The proposed formula ===============================================
N_OFFSET_PERIODS = 15.0    # 5 (wavemaker ramp) + 10 (H&G "10 periods after arrival")
WINDOW_PERIODS   = 10.0    # H&G's 10T window length (preserved)
SNAP_HALFWIDTH_T = 1.0     # ±1 wave period UC search window for snap-to-start
TANK_DEPTH_M     = HG.TANK_DEPTH_M   # 0.58 m (project canonical)

# Canon — march-2026 cond4 lowrange
PROCESSED_DIRS = [
    Path("waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange"),
    Path("waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"),
]

SCRATCH_DIR = Path(__file__).parent

COL_HG     = "#2ECC71"
COL_PSTOP  = "#7F3FBF"


# ── Helpers ─────────────────────────────────────────────────────────────
def fft_amp_at_freq(segment: np.ndarray, target_hz: float,
                    fs: float = FS, band_hz: float = FFT_BAND_HZ) -> float:
    N = len(segment)
    if N < 4:
        return np.nan
    seg = np.asarray(segment, dtype=float).copy()
    if np.isnan(seg).any():
        idx  = np.arange(N)
        good = ~np.isnan(seg)
        if good.sum() < N * 0.9:
            return np.nan
        seg = np.interp(idx, idx[good], seg[good])
    freqs = np.fft.fftfreq(N, d=1.0 / fs)
    pos   = freqs > 0
    pos_f = freqs[pos]
    mask  = (pos_f >= target_hz - band_hz) & (pos_f <= target_hz + band_hz)
    if not mask.any():
        j = int(np.argmin(np.abs(pos_f - target_hz)))
    else:
        local_idx = int(np.argmin(np.abs(pos_f[mask] - target_hz)))
        j = np.where(mask)[0][local_idx]
    fft_vals = np.fft.fft(seg)
    amps_pos = 2.0 * np.abs(fft_vals[pos]) / N
    return float(amps_pos[j])


def get_eta(df_run: pd.DataFrame, pos: str):
    for col in (f"eta_{pos}_interp", f"eta_{pos}"):
        if col in df_run.columns:
            return df_run[col].to_numpy(dtype=float)
    return None


def sliding_afft(signal: np.ndarray, target_hz: float,
                 window_s: float, step_s: float = SLIDING_STEP_S,
                 fs: float = FS):
    N_win = int(round(window_s * fs))
    step  = int(round(step_s * fs))
    if N_win >= len(signal):
        return np.array([]), np.array([])
    starts = np.arange(0, len(signal) - N_win + 1, step)
    t_start = starts / fs
    A = np.full(len(starts), np.nan)
    for i, s in enumerate(starts):
        A[i] = fft_amp_at_freq(signal[s:s + N_win], target_hz, fs)
    return t_start, A


def canonical_in_signal(df_run: pd.DataFrame):
    sigs = []
    for p in IN_PROBES:
        s = get_eta(df_run, p)
        if s is not None:
            sigs.append(s)
    if not sigs:
        return None
    nmin = min(len(s) for s in sigs)
    stack = np.vstack([s[:nmin] for s in sigs])
    return np.nanmean(stack, axis=0)


# === The proposed-formula window ========================================
def proposed_window_theoretical(probe: str, f_paddle: float):
    """t_start, t_end (seconds) per the proposed formula, pre-snap."""
    r_m = PROBE_R_M[probe]
    c_g = c_group(f_paddle, TANK_DEPTH_M)
    t_start = r_m / c_g + N_OFFSET_PERIODS / f_paddle
    t_end   = t_start + WINDOW_PERIODS / f_paddle
    return t_start, t_end


def snap_to_upcrossing(signal: np.ndarray, target_idx: int,
                       f_paddle: float, fs: float = FS) -> int:
    """Snap target sample idx to nearest zero-upcrossing of `signal` within
    ±SNAP_HALFWIDTH_T periods. Same logic as the live pipeline:
    upcrossing threshold = local DC mean of first 2 s of signal.
    Returns target_idx unchanged if no UC found in window.
    """
    if target_idx < 0 or target_idx >= len(signal):
        return target_idx
    samples_per_period = int(round(fs / f_paddle))
    halfwidth = int(round(SNAP_HALFWIDTH_T * samples_per_period))

    baseline_n = int(2 * fs)
    upcross_level = float(np.nanmean(signal[:baseline_n]))
    above = signal > upcross_level
    all_uc = np.where((~above[:-1]) & above[1:])[0] + 1

    lo = max(0, target_idx - halfwidth)
    hi = min(len(signal) - 1, target_idx + halfwidth)
    cands = all_uc[(all_uc >= lo) & (all_uc <= hi)]
    if len(cands) == 0:
        return target_idx
    return int(cands[np.argmin(np.abs(cands - target_idx))])


def proposed_window_snapped(signal: np.ndarray, probe: str,
                            f_paddle: float, fs: float = FS):
    """(t_start_s, t_end_s) using the proposed formula, with start snapped
    to the nearest UC within ±T. End = start + 10·samples_per_period."""
    t_th_start, _ = proposed_window_theoretical(probe, f_paddle)
    target_start_idx = int(round(t_th_start * fs))
    snap_start_idx = snap_to_upcrossing(signal, target_start_idx, f_paddle, fs)
    samples_per_period = int(round(fs / f_paddle))
    snap_end_idx = snap_start_idx + int(round(WINDOW_PERIODS * samples_per_period))
    return snap_start_idx / fs, snap_end_idx / fs


# ── Load ────────────────────────────────────────────────────────────────
print("1. Loading meta + processed_dfs (canon March-2026 lowrange) …")
combined_meta, _, _, _ = load_analysis_data(*PROCESSED_DIRS, load_processed=False)
processed_dfs = load_processed_dfs(*PROCESSED_DIRS)
print(f"   meta: {len(combined_meta)} rows · processed_dfs: {len(processed_dfs)} runs")


def select_canon(target_freq: float) -> pd.DataFrame:
    mask = (
        np.isclose(pd.to_numeric(combined_meta["WaveFrequencyInput [Hz]"], errors="coerce"),
                   target_freq, atol=0.02)
        & np.isclose(pd.to_numeric(combined_meta["WaveAmplitudeInput [Volt]"], errors="coerce"),
                     TARGET_AMP, atol=0.01)
        & (combined_meta["PanelCondition"] == "full")
        & (combined_meta["quality_flag"] == "ok")
    )
    sub = combined_meta[mask].copy()
    sub["N_input_periods"] = pd.to_numeric(sub["WavePeriodInput"], errors="coerce")
    sub["run_type"] = np.where(sub["N_input_periods"] >= PER240_THRESHOLD_T,
                               "per240", "per40")
    return sub


def median_proposed_window(rs: pd.DataFrame, probe: str, f_paddle: float):
    """Median (start, end) seconds across rs after applying proposed formula
    + snap on each run's IN/OUT signal."""
    starts, ends = [], []
    for _, r in rs.iterrows():
        df = processed_dfs.get(r["path"])
        if df is None:
            continue
        sig = canonical_in_signal(df) if probe.startswith("9373") else get_eta(df, probe)
        if sig is None:
            continue
        s, e = proposed_window_snapped(sig, probe, f_paddle, FS)
        starts.append(s)
        ends.append(e)
    if not starts:
        return None
    return float(np.median(starts)), float(np.median(ends))


# ── Per-frequency figure builder ────────────────────────────────────────
def build_for(target_freq: float):
    sel = select_canon(target_freq)
    print(f"\n— f={target_freq:.2f} Hz: {len(sel)} canonical runs "
          f"(per40 n={int((sel['run_type']=='per40').sum())}, "
          f"per240 n={int((sel['run_type']=='per240').sum())})")

    # Print out the formula's theoretical positions for reference.
    th_in_s,  th_in_e  = proposed_window_theoretical("9373/170",  target_freq)
    th_out_s, th_out_e = proposed_window_theoretical("12400/250", target_freq)
    print(f"   PROPOSED  IN  (theoretical, pre-snap): [{th_in_s:5.2f}, {th_in_e:5.2f}] s")
    print(f"   PROPOSED  OUT (theoretical, pre-snap): [{th_out_s:5.2f}, {th_out_e:5.2f}] s")

    window_s_10T = WINDOW_PERIODS / target_freq
    paddle_stop_s = PER40_PERIODS / target_freq

    sliding = {}
    outin   = {}
    for _, r in sel.iterrows():
        path = r["path"]
        df = processed_dfs.get(path)
        if df is None:
            continue
        sig_in  = canonical_in_signal(df)
        sig_out = get_eta(df, OUT_PROBE)
        if sig_in is None or sig_out is None:
            continue
        ts_in,  A_in  = sliding_afft(sig_in,  target_freq, window_s_10T)
        ts_out, A_out = sliding_afft(sig_out, target_freq, window_s_10T)
        sliding[path] = {"IN": (ts_in, A_in), "OUT": (ts_out, A_out)}
        if len(ts_in) == len(ts_out) and len(ts_in) > 0:
            outin[path] = (ts_in, A_out / A_in)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)

    for row_i, wind in enumerate(("no", "full")):
        rs_240 = sel[(sel["run_type"] == "per240") & (sel["WindCondition"] == wind)]
        rs_40  = sel[(sel["run_type"] == "per40")  & (sel["WindCondition"] == wind)]
        color  = WIND_COLOR_MAP[wind]

        # Median PROPOSED-formula snapped windows across the cohort
        all_for_med = pd.concat([rs_240, rs_40], ignore_index=True)
        sn_in  = median_proposed_window(all_for_med, "9373/170",  target_freq)
        sn_out = median_proposed_window(all_for_med, "12400/250", target_freq)

        def _plot_curves(ax, probe_tag: str, sn_band):
            for _, r in rs_240.iterrows():
                path = r["path"]
                if path not in sliding:
                    continue
                ts, A = sliding[path][probe_tag]
                ax.plot(ts, A, color=color, lw=1.2, alpha=0.85)
            for _, r in rs_40.iterrows():
                path = r["path"]
                if path not in sliding:
                    continue
                ts, A = sliding[path][probe_tag]
                ax.plot(ts, A, color=color, lw=1.0, alpha=0.85, ls="--")
            if sn_band is not None:
                ax.axvspan(sn_band[0], sn_band[1], color=COL_HG, alpha=0.22, lw=0)
                ax.axvline(sn_band[0], color=COL_HG, lw=0.6, ls="--", alpha=0.7)
                ax.axvline(sn_band[1], color=COL_HG, lw=0.6, ls="--", alpha=0.7)
                ax.text(0.5 * (sn_band[0] + sn_band[1]), 0.92,
                        f"[{sn_band[0]:.1f}, {sn_band[1]:.1f}] s",
                        transform=ax.get_xaxis_transform(),
                        ha="center", va="top", fontsize=7,
                        color="#1A6E2A", alpha=0.85)
            ax.axvline(paddle_stop_s, color=COL_PSTOP, ls=":", lw=1.0, alpha=0.85)
            ax.set_xlim(SLIDING_T_LO, SLIDING_T_HI)
            ax.grid(True, alpha=0.25, lw=0.4)
            ax.tick_params(labelsize=8)

        ax = axes[row_i, 0]
        _plot_curves(ax, "IN", sn_in)
        ax.set_ylabel(f"{wind} wind\n$A_\\mathrm{{IN}}$  [mm]", fontsize=9)

        ax = axes[row_i, 1]
        _plot_curves(ax, "OUT", sn_out)
        ax.tick_params(labelleft=False)
        if row_i == 0:
            axes[row_i, 1].set_ylabel("$A_\\mathrm{OUT}$  [mm]", fontsize=9, labelpad=-5)

        ax = axes[row_i, 2]
        for _, r in rs_240.iterrows():
            path = r["path"]
            if path not in outin:
                continue
            ts, R = outin[path]
            ax.plot(ts, R, color=color, lw=1.2, alpha=0.85)
        for _, r in rs_40.iterrows():
            path = r["path"]
            if path not in outin:
                continue
            ts, R = outin[path]
            ax.plot(ts, R, color=color, lw=1.0, alpha=0.85, ls="--")
        if sn_out is not None:
            ax.axvspan(sn_out[0], sn_out[1], color=COL_HG, alpha=0.22, lw=0)
            ax.axvline(sn_out[0], color=COL_HG, lw=0.6, ls="--", alpha=0.7)
            ax.axvline(sn_out[1], color=COL_HG, lw=0.6, ls="--", alpha=0.7)
        ax.axvline(paddle_stop_s, color=COL_PSTOP, ls=":", lw=1.0, alpha=0.85)
        ax.axhline(1.0, color="black", lw=0.5, ls="--", alpha=0.4)
        ax.set_ylim(0, 1.2)
        ax.set_xlim(SLIDING_T_LO, SLIDING_T_HI)
        ax.grid(True, alpha=0.25, lw=0.4)
        ax.tick_params(labelsize=8)
        ax.set_ylabel("OUT/IN", fontsize=9)

        axes[row_i, 0].text(
            0.02, 0.95, f"n_per240={len(rs_240)}  n_per40={len(rs_40)}",
            transform=axes[row_i, 0].transAxes,
            fontsize=7, color="#444", va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#bbb",
                      alpha=0.85, lw=0.4),
        )

    for ax in axes[-1, :]:
        ax.set_xlabel("window start [s from wavemaker start]", fontsize=8)

    fig.suptitle("", fontsize=11)

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([], [], color=WIND_COLOR_MAP["no"],   lw=1.4, label="per240 (uten vind)"),
        Line2D([], [], color=WIND_COLOR_MAP["no"],   lw=1.2, ls="--", label="per40 (uten vind)"),
        Line2D([], [], color=WIND_COLOR_MAP["full"], lw=1.4, label="per240 (med vind)"),
        Line2D([], [], color=WIND_COLOR_MAP["full"], lw=1.2, ls="--", label="per40 (med vind)"),
        Patch(facecolor=COL_HG, alpha=0.22,
              label=f"PROPOSED window (N_offset={int(N_OFFSET_PERIODS)} + UC-snap), median over cohort"),
        Line2D([], [], color=COL_PSTOP, ls=":", lw=1.2,
               label=f"per40 paddle stop (40/f = {paddle_stop_s:.1f} s)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               fontsize=7.5, bbox_to_anchor=(0.5, -0.01), frameon=True)

    fig.tight_layout(rect=[0, 0.05, 1, 0.99])

    f_tag = f"f{int(round(target_freq * 10))}"
    out_pdf = SCRATCH_DIR / f"hg_plateau_check_proposed_{f_tag}.pdf"
    out_png = SCRATCH_DIR / f"hg_plateau_check_proposed_{f_tag}.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"   saved → {out_pdf.relative_to(BASE)}")
    print(f"          {out_png.relative_to(BASE)}")


for _f in TARGET_FREQS:
    build_for(_f)
