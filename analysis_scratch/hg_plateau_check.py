"""
H&G plateau check — sliding AFFT for per40 vs per240, all 4 thesis freqs
=========================================================================

Successor to the archived `hg_window_stability_with_per40.py`. That earlier
figure used a fixed [50T, 60T] H&G window (eyeballed, anchored at the OUT
probe) and a SNARVEI band for per40. The current pipeline uses:

  * **Probe-shifted** H&G window — anchored at r = 12.4 m (= our OUT probe);
    closer probes have the same 10T window shifted earlier by group-velocity
    travel time ΔT(f) = (12.4 − r_probe) / c_g(f) · f periods.
  * **UC-snapped** start AND end — both endpoints snap to the nearest
    zero-upcrossing of the raw signal within ±T (start) / ±0.5T (end) of
    the theoretical H&G location, guaranteeing integer-cycle windows.

So this script reads each run's actual snapped window from
`Computed Probe {pos} start/end` in meta.json, and overlays it on the
sliding-AFFT curve. The visual question:

    Does the snapped H&G window land on a flat plateau of the sliding
    AFFT — same plateau that per240 sits on?

If per40's sliding curve plateaus where per240's does, AND the snapped
H&G band sits inside that plateau → pooling per40 + per240 under H&G is
honest. (Confirmed quantitatively at 1–2 % median agreement in
ch04_per40_and_per240_HG_shifted; this script is the visual companion.)

Layout: one PDF per thesis frequency. 2 rows (nowind, fullwind) × 3 cols
(IN sliding, OUT sliding, OUT/IN sliding). Per40 dashed, per240 solid;
nowind colour-keyed to WIND_COLOR_MAP["no"], fullwind to ["full"].

Run from repo root:
    /opt/anaconda3/envs/draumkvedet/bin/python analysis_scratch/hg_plateau_check.py

Outputs (scratch, diagnostic only — not a thesis figure):
    analysis_scratch/hg_plateau_check_f13.{pdf,png}
    analysis_scratch/hg_plateau_check_f14.{pdf,png}
    analysis_scratch/hg_plateau_check_f15.{pdf,png}
    analysis_scratch/hg_plateau_check_f16.{pdf,png}
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
SLIDING_T_HI       = 80.0    # both run-types fit inside this (per40 ends ~30s)

IN_PROBES = ["9373/170", "9373/340"]   # mean for canonical IN
OUT_PROBE = "12400/250"

# Canon — march-2026 cond4 lowrange
PROCESSED_DIRS = [
    Path("waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange"),
    Path("waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"),
]

SCRATCH_DIR = Path(__file__).parent

# Colours
COL_HG     = "#2ECC71"   # green H&G band
COL_PSTOP  = "#7F3FBF"   # purple paddle stop


# ── Helpers ─────────────────────────────────────────────────────────────
def fft_amp_at_freq(segment: np.ndarray, target_hz: float,
                    fs: float = FS, band_hz: float = FFT_BAND_HZ) -> float:
    """Peak-bin AFFT, pipeline convention (2·|FFT|/N)."""
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


def median_snap_window(rs: pd.DataFrame, probe: str):
    """Median snapped (start, end) seconds across rs for one probe."""
    st = pd.to_numeric(rs[f"Computed Probe {probe} start"], errors="coerce").dropna()
    et = pd.to_numeric(rs[f"Computed Probe {probe} end"],   errors="coerce").dropna()
    if st.empty or et.empty:
        return None
    return float(st.median()) / FS, float(et.median()) / FS


# ── Per-frequency figure builder ────────────────────────────────────────
def build_for(target_freq: float):
    sel = select_canon(target_freq)
    print(f"\n— f={target_freq:.2f} Hz: {len(sel)} canonical runs "
          f"(per40 n={int((sel['run_type']=='per40').sum())}, "
          f"per240 n={int((sel['run_type']=='per240').sum())})")

    window_s_10T = 10.0 / target_freq
    paddle_stop_s = PER40_PERIODS / target_freq

    # Compute sliding curves per run, indexed by (run_type, wind, path).
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

        # Median snapped windows (median over the per240 cohort for stability;
        # per40 windows are within 1-sample of per240 by construction since they
        # use the same probe-shifted H&G + UC-snap rule).
        sn_in  = median_snap_window(rs_240, IN_PROBES[0]) or median_snap_window(rs_40, IN_PROBES[0])
        sn_out = median_snap_window(rs_240, OUT_PROBE)    or median_snap_window(rs_40, OUT_PROBE)

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
            ax.axvline(paddle_stop_s, color=COL_PSTOP, ls=":", lw=1.0, alpha=0.85)
            ax.set_xlim(SLIDING_T_LO, SLIDING_T_HI)
            ax.grid(True, alpha=0.25, lw=0.4)
            ax.tick_params(labelsize=8)

        # Col 0 — IN sliding AFFT
        ax = axes[row_i, 0]
        _plot_curves(ax, "IN", sn_in)
        ax.set_ylabel(f"{wind} wind\n$A_\\mathrm{{IN}}$  [mm]", fontsize=9)

        # Col 1 — OUT sliding AFFT
        ax = axes[row_i, 1]
        _plot_curves(ax, "OUT", sn_out)
        if row_i == 0:
            pass
        ax.tick_params(labelleft=False)
        if row_i == 0:
            axes[row_i, 1].set_ylabel("$A_\\mathrm{OUT}$  [mm]", fontsize=9, labelpad=-5)

        # Col 2 — OUT/IN sliding (the money column)
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
        # OUT/IN: shade the OUT-probe window since that's where the ratio's
        # numerator is integrated; light overlay of IN window for context.
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

        # Annotate cohort sizes on column 0
        axes[row_i, 0].text(
            0.02, 0.95, f"n_per240={len(rs_240)}  n_per40={len(rs_40)}",
            transform=axes[row_i, 0].transAxes,
            fontsize=7, color="#444", va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#bbb",
                      alpha=0.85, lw=0.4),
        )

    # Bottom-row x labels
    for ax in axes[-1, :]:
        ax.set_xlabel("window start [s from wavemaker start]", fontsize=8)

    # Per-figure caption-equivalent (kept empty per the project rule;
    # paddle-stop and H&G band are self-explanatory once labelled).
    fig.suptitle("", fontsize=11)

    # Custom legend (single, bottom-centre)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([], [], color=WIND_COLOR_MAP["no"],   lw=1.4, label="per240 (uten vind)"),
        Line2D([], [], color=WIND_COLOR_MAP["no"],   lw=1.2, ls="--", label="per40 (uten vind)"),
        Line2D([], [], color=WIND_COLOR_MAP["full"], lw=1.4, label="per240 (med vind)"),
        Line2D([], [], color=WIND_COLOR_MAP["full"], lw=1.2, ls="--", label="per40 (med vind)"),
        Patch(facecolor=COL_HG, alpha=0.22, label="snapped H&G window (median over cohort)"),
        Line2D([], [], color=COL_PSTOP, ls=":", lw=1.2, label=f"per40 paddle stop (40/f = {paddle_stop_s:.1f} s)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               fontsize=7.5, bbox_to_anchor=(0.5, -0.01), frameon=True)

    fig.tight_layout(rect=[0, 0.05, 1, 0.99])

    f_tag = f"f{int(round(target_freq * 10))}"
    out_pdf = SCRATCH_DIR / f"hg_plateau_check_{f_tag}.pdf"
    out_png = SCRATCH_DIR / f"hg_plateau_check_{f_tag}.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"   saved → {out_pdf.relative_to(BASE)}")
    print(f"          {out_png.relative_to(BASE)}")


for _f in TARGET_FREQS:
    build_for(_f)
