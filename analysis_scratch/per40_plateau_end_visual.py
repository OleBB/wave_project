"""
Per40 OUT plateau-end visualization — companion to per40_plateau_end_measurement.

For each thesis frequency: one panel showing the sliding A_FFT(t) at OUT for
per40 nowind + fullwind, overlaid with:

    - light grey hatched band : plateau reference region (where the ref median
                                 is computed)
    - solid line              : empirical plateau-end at strict ±1 % criterion
    - dashed line             : empirical plateau-end at relaxed ±2 % criterion
    - vertical green line     : right edge of the proposed pipeline window
                                 (start = t_arr_OUT + 10·T, length = N(f)·T)
    - vertical purple line    : per40 paddle-stop time (40/f at the wavemaker;
                                 wave train at OUT continues for r_OUT/c_g
                                 longer than this)

The reader sees: where exactly the AFFT droops, and how the chosen window's
right edge sits relative to the plateau bounds. If the green line is to the
LEFT of the dashed line, the window is within the relaxed plateau.

Output:
    analysis_scratch/per40_plateau_end_visual.{pdf,png}
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

# ── Config (mirror per40_plateau_end_measurement.py) ────────────────────
FS                 = 250.0
FFT_BAND_HZ        = 0.05
THESIS_FREQS       = [1.3, 1.4, 1.5, 1.6]
TARGET_AMP         = 0.2
PER240_THRESHOLD_T = 50

IN_PROBES = ["9373/170", "9373/340"]
OUT_PROBE = "12400/250"
PROBE_R_M = {"9373/170": 9.373, "9373/340": 9.373, "12400/250": 12.400}
TANK_DEPTH_M = HG.TANK_DEPTH_M

N_LENGTH_LOOKUP = {1.3: 10, 1.4: 13, 1.5: 13, 1.6: 13}
N_OFFSET_T      = 10
PER40_PERIODS   = 40

PLATEAU_REF_T_LO = 8.0
PLATEAU_REF_T_HI = 12.0
TOL_STRICT_PCT   = 1.0
TOL_RELAX_PCT    = 2.0

# OUT-probe eyeballed plateau bounds (from window_proof_figure.py / snarvei
# calibration, 0.2 V). These are the "we started on the downhill and will
# never come back" markers — what actually bounds the usable window.
EYEBALL_OUT_END_S   = {1.3: 44.0, 1.4: 43.0, 1.5: 41.0, 1.6: 41.0}
EYEBALL_OUT_START_S = {1.3: 27.0, 1.4: 27.0, 1.5: 28.0, 1.6: 29.0}

# H&G classic [50T, 60T] window at the reference probe (r = 12.4 m = OUT).
# Anchored to the wavemaker start. Reference: Huseby & Grue (2000), J. Fluid Mech.
HG_START_T = 50
HG_END_T   = 60

SLIDING_STEP_S = 0.10
SLIDE_START_S  = 5.0
SLIDE_STOP_S   = 60.0

PROCESSED_DIRS = [
    Path("waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange"),
    Path("waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"),
]

OUT_PDF = Path(__file__).parent / "per40_plateau_end_visual.pdf"
OUT_PNG = Path(__file__).parent / "per40_plateau_end_visual.png"

COL_WIN_END = "#2ECC71"   # green  — proposed window's right edge
COL_PSTOP   = "#7F3FBF"   # purple — per40 paddle stop
COL_EYE     = "#F39C12"   # orange — eyeballed plateau bounds
COL_HG      = "#1A6E2A"   # dark green — H&G [50T, 60T] classic window


# ── Helpers (copy from measurement script) ──────────────────────────────
def fft_amp(seg, target_hz, fs=FS, band_hz=FFT_BAND_HZ):
    N = len(seg)
    if N < 4:
        return np.nan
    s = np.asarray(seg, dtype=float).copy()
    if np.isnan(s).any():
        idx = np.arange(N); good = ~np.isnan(s)
        if good.sum() < N * 0.9:
            return np.nan
        s = np.interp(idx, idx[good], s[good])
    freqs = np.fft.fftfreq(N, d=1.0 / fs)
    pos = freqs > 0
    pos_f = freqs[pos]
    mask = (pos_f >= target_hz - band_hz) & (pos_f <= target_hz + band_hz)
    if not mask.any():
        j = int(np.argmin(np.abs(pos_f - target_hz)))
    else:
        local = int(np.argmin(np.abs(pos_f[mask] - target_hz)))
        j = np.where(mask)[0][local]
    return float(2.0 * np.abs(np.fft.fft(s)[pos])[j] / N)


def get_eta(df, pos):
    for col in (f"eta_{pos}_interp", f"eta_{pos}"):
        if col in df.columns:
            return df[col].to_numpy(dtype=float)
    return None


def sliding_afft(signal, target_hz, window_s, step_s=SLIDING_STEP_S,
                 t_lo=SLIDE_START_S, t_hi=SLIDE_STOP_S):
    N_win = int(round(window_s * FS))
    step = int(round(step_s * FS))
    if N_win >= len(signal):
        return np.array([]), np.array([])
    n_lo = int(round(t_lo * FS))
    n_hi = min(len(signal) - N_win, int(round(t_hi * FS)))
    starts = np.arange(n_lo, n_hi + 1, step)
    ts = starts / FS
    A = np.array([fft_amp(signal[s:s + N_win], target_hz) for s in starts])
    return ts, A


def find_plateau_end(ts, A, plat_lo_s, plat_hi_s, tol_pct):
    mask = (ts >= plat_lo_s) & (ts <= plat_hi_s)
    if not mask.any():
        return np.nan, np.nan
    ref = float(np.nanmedian(A[mask]))
    if not np.isfinite(ref) or ref == 0:
        return ref, np.nan
    rel = (A - ref) / ref * 100.0
    inside = np.abs(rel) < tol_pct
    n_lookahead = int(round(2.0 / SLIDING_STEP_S))
    ref_end_idx = int(np.searchsorted(ts, plat_hi_s))
    plateau_end_t = np.nan
    for i in range(ref_end_idx, len(inside) - n_lookahead):
        if not inside[i:i + n_lookahead].any():
            plateau_end_t = float(ts[i])
            break
    if np.isnan(plateau_end_t):
        plateau_end_t = float(ts[-1])
    return ref, plateau_end_t


# ── Load ────────────────────────────────────────────────────────────────
print("Loading meta + processed_dfs (canon March-2026 lowrange) …")
combined_meta, _, _, _ = load_analysis_data(*PROCESSED_DIRS, load_processed=False)
processed_dfs = load_processed_dfs(*PROCESSED_DIRS)


def select_canon(target_freq):
    f_col = pd.to_numeric(combined_meta["WaveFrequencyInput [Hz]"], errors="coerce")
    a_col = pd.to_numeric(combined_meta["WaveAmplitudeInput [Volt]"], errors="coerce")
    mask = (
        np.isclose(f_col, target_freq, atol=0.02)
        & np.isclose(a_col, TARGET_AMP, atol=0.01)
        & (combined_meta["PanelCondition"] == "full")
        & (combined_meta["quality_flag"] == "ok")
    )
    sub = combined_meta[mask].copy()
    sub["N_input_periods"] = pd.to_numeric(sub["WavePeriodInput"], errors="coerce")
    sub["run_type"] = np.where(sub["N_input_periods"] >= PER240_THRESHOLD_T,
                               "per240", "per40")
    return sub


# ── Build figure ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, len(THESIS_FREQS), figsize=(15, 4.8),
                         sharey=False)

for ax, f in zip(axes, THESIS_FREQS):
    n_length     = N_LENGTH_LOOKUP[f]
    window_s     = n_length / f
    sel          = select_canon(f)
    per40        = sel[sel["run_type"] == "per40"]
    t_arr_out    = PROBE_R_M[OUT_PROBE] / c_group(f, TANK_DEPTH_M)
    plat_lo_s    = t_arr_out + PLATEAU_REF_T_LO / f
    plat_hi_s    = t_arr_out + PLATEAU_REF_T_HI / f
    win_start_s  = t_arr_out + N_OFFSET_T / f
    win_end_s    = win_start_s + n_length / f
    paddle_stop  = PER40_PERIODS / f

    # Per40 sliding AFFT — one curve per wind
    for _, r in per40.iterrows():
        df = processed_dfs.get(r["path"])
        if df is None:
            continue
        sig = get_eta(df, OUT_PROBE)
        if sig is None:
            continue
        wind = r["WindCondition"]
        color = WIND_COLOR_MAP[wind]
        ts, A = sliding_afft(sig, f, window_s)
        if len(A) == 0:
            continue
        ax.plot(ts, A, color=color, lw=1.4, alpha=0.90,
                label=f"per40 {wind}")

    # ── Reference markers (per-frequency, not per-run) ──────────────────
    # Eyeball plateau START + END at OUT (orange band)
    eb_start = EYEBALL_OUT_START_S[f]
    eb_end   = EYEBALL_OUT_END_S[f]
    ax.axvspan(eb_start, eb_end, color=COL_EYE, alpha=0.10, lw=0)
    ax.axvline(eb_start, color=COL_EYE, lw=1.2, ls="--", alpha=0.7)
    ax.axvline(eb_end,   color=COL_EYE, lw=1.6, ls="-",  alpha=0.85)

    # H&G classic [50T, 60T] window at the reference probe (=OUT, r=12.4m)
    hg_start_s = HG_START_T / f
    hg_end_s   = HG_END_T   / f
    ax.axvline(hg_start_s, color=COL_HG, lw=1.0, ls=":", alpha=0.7)
    ax.axvline(hg_end_s,   color=COL_HG, lw=1.4, ls=":", alpha=0.85)

    # Pipeline window right edge (green) and paddle stop (purple)
    ax.axvline(win_end_s, color=COL_WIN_END, lw=2.0, ls="-", alpha=0.85)
    ax.axvline(paddle_stop, color=COL_PSTOP, lw=1.0, ls=":", alpha=0.85)

    ax.set_xlim(SLIDE_START_S, SLIDE_STOP_S)
    ax.set_ylim(bottom=4)
    ax.set_xlabel("window start [s from wavemaker start]", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.96, f"$f={f}$ Hz · OUT, per40\n"
            f"window [t_arr+10T, +{n_length+10}T] = "
            f"[{win_start_s:.1f}, {win_end_s:.1f}] s\n"
            f"H&G [50T, 60T] = [{hg_start_s:.1f}, {hg_end_s:.1f}] s\n"
            f"eyeball plateau = [{eb_start:.1f}, {eb_end:.1f}] s",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=7.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="#bbb", alpha=0.85, lw=0.4))

axes[0].set_ylabel("$A_\\mathrm{OUT}$ (FFT)  [mm]", fontsize=10)

# Suptitle replaces the legend — colour/style key inline.
fig.suptitle(
    r"Per40 OUT sliding $A_\mathrm{FFT}$ at $r{=}12.4$ m, $0.2$ V, full panel.   "
    r"Curves: $\mathbf{blue}$ = nowind,  $\mathbf{red}$ = fullwind.   "
    r"Vertical refs: $\mathbf{orange}$ band/lines = eyeball plateau START–END (snarvei calib);   "
    r"$\mathbf{dark\ green}$ dotted = H&G [50T, 60T] at $r{=}12.4$ m;   "
    r"$\mathbf{bright\ green}$ = proposed-window right edge;   "
    r"$\mathbf{purple}$ dotted = per40 paddle stop $40/f$.",
    fontsize=8.5, y=0.99, wrap=True,
)
fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"saved → {OUT_PDF.relative_to(BASE)}")
print(f"        {OUT_PNG.relative_to(BASE)}")
