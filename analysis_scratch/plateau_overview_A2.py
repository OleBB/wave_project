"""
Plateau overview at A_2 (0.20 V) — sliding A_FFT(t) per (f, probe).
=====================================================================

Reader-facing figure: 4 rows (f = 1.3, 1.4, 1.5, 1.6 Hz) × 2 cols
(IN, OUT). Each panel overlays both wind conditions (blue = nowind,
red = fullwind, per project WIND_COLOR_MAP). One thin line per canon
run (cond4 March-2026 lowrange, full panel, 0.20 V, quality_ok). No
per40/per240 split — pooled per cell.

Sliding A_FFT(t) at the paddle frequency, window length matches
Option B: N(f) = {1.3:10, 1.4:13, 1.5:13, 1.6:13} periods. Step 0.1 s.

Visual claim: in every (f, probe, wind) cell, the chosen Option B
window (green band) sits inside a flat A_FFT plateau.

Layout decisions:
  * x-axis in seconds, [0, 80] uniform across panels — covers the
    whole per40 record length.
  * y-axis fixed [0, 20] mm across all 8 panels (same amp tier).
  * each individual run plotted as its own thin line (low alpha) —
    small N cells are honest; no median.
  * green band = Option B window applied at THE PROBE OF THE PANEL
    (so the green window sits where the FFT actually evaluates that
    probe's slice).
  * vertical purple dotted = per40 paddle stop (40/f).
  * vertical red dotted    = parasitic 2f arrival at THE PROBE OF THE PANEL.

Outputs (scratch only — promote to main_save_figures once approved):
    analysis_scratch/plateau_overview_A2.{pdf,png}

Run from repo root:
    /opt/anaconda3/envs/draumkvedet/bin/python analysis_scratch/plateau_overview_A2.py
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
from matplotlib.lines import Line2D

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import os
os.chdir(BASE)

from wavescripts.improved_data_loader import load_analysis_data, load_processed_dfs
from wavescripts.constants import c_group, HG
from wavescripts.plot_utils import apply_thesis_style, WIND_COLOR_MAP

apply_thesis_style()


# ── Config ──────────────────────────────────────────────────────────────
FS               = 250.0
FFT_BAND_HZ      = 0.05

THESIS_FREQS     = [1.3, 1.4, 1.5, 1.6]
TARGET_AMP       = 0.2
N_OFFSET         = 7
N_LENGTH         = 10                     # uniform across all thesis freqs

TANK_LENGTH_M    = 25.0   # wavemaker → back wall distance (round trip = 2L = 50 m)

# Seiche speed = √(g·h) — shallow-water (long-wave) speed in the tank.
# Hard-coded per user instruction; mirrors PHYSICS.GRAVITY and HG.TANK_DEPTH_M
# in wavescripts/constants.py.
SEICHE_SPEED_M_S = (9.81 * 0.58) ** 0.5     # ≈ 2.385 m/s

IN_PROBES        = ["9373/170", "9373/340"]
OUT_PROBE        = "12400/250"
PROBE_R_M        = {"9373/170": 9.373, "9373/340": 9.373, "12400/250": 12.400}
TANK_DEPTH_M     = HG.TANK_DEPTH_M
PER40_PERIODS    = 40

SLIDING_STEP_S   = 0.10
SLIDING_T_LO_S   = 0.0
SLIDING_T_HI_S   = 55.0     # crop after the post-window region of interest

Y_LO, Y_HI       = 0.0, 20.0    # mm — fits A_2 IN (~16) and OUT (~12) with margin

WINDS            = ["no", "full"]

# Canon — march-2026 cond4 lowrange
PROCESSED_DIRS = [
    Path("waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange"),
    Path("waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"),
]

OUT_PDF = Path(__file__).parent / "plateau_overview_A2.pdf"
OUT_PNG = Path(__file__).parent / "plateau_overview_A2.png"


# ── Helpers ─────────────────────────────────────────────────────────────
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


def can_in(df):
    sigs = [get_eta(df, p) for p in IN_PROBES]
    sigs = [s for s in sigs if s is not None]
    if not sigs:
        return None
    nmin = min(len(s) for s in sigs)
    return np.nanmean(np.vstack([s[:nmin] for s in sigs]), axis=0)


def sliding_afft(signal, target_hz, window_s):
    N_win = int(round(window_s * FS))
    step  = int(round(SLIDING_STEP_S * FS))
    n_lo  = int(round(SLIDING_T_LO_S * FS))
    n_hi  = min(len(signal) - N_win, int(round(SLIDING_T_HI_S * FS)))
    if N_win >= len(signal) or n_hi <= n_lo:
        return np.array([]), np.array([])
    starts = np.arange(n_lo, n_hi + 1, step)
    ts = starts / FS
    A = np.array([fft_amp(signal[s:s + N_win], target_hz) for s in starts])
    return ts, A


# ── Load ────────────────────────────────────────────────────────────────
print("Loading meta + processed_dfs (canon March-2026 lowrange) …")
combined_meta, _, _, _ = load_analysis_data(*PROCESSED_DIRS, load_processed=False)
processed_dfs = load_processed_dfs(*PROCESSED_DIRS)

f_col = pd.to_numeric(combined_meta["WaveFrequencyInput [Hz]"], errors="coerce")
a_col = pd.to_numeric(combined_meta["WaveAmplitudeInput [Volt]"], errors="coerce")
mask = (
    np.isclose(a_col, TARGET_AMP, atol=0.01)
    & f_col.between(THESIS_FREQS[0] - 0.02, THESIS_FREQS[-1] + 0.02)
    & (combined_meta["PanelCondition"] == "full")
    & combined_meta["WindCondition"].isin(WINDS)
    & (combined_meta["quality_flag"] == "ok")
)
sel = combined_meta[mask].copy()
print(f"\n{len(sel)} canon runs at A_2 (0.20 V), full panel, quality ok.")


def runs_for(f, wind):
    return sel[np.isclose(pd.to_numeric(sel["WaveFrequencyInput [Hz]"], errors="coerce"),
                           f, atol=0.02)
               & (sel["WindCondition"] == wind)]


# ── Compute sliding A_FFT per (run, probe) ──────────────────────────────
print("\nComputing sliding A_FFT at IN and OUT per run …")
curves = {}    # (f, wind, probe_tag, path) -> (ts, A) in mm

for f in THESIS_FREQS:
    L_T = N_LENGTH
    window_s = L_T / f
    for wind in WINDS:
        rs = runs_for(f, wind)
        for _, r in rs.iterrows():
            df = processed_dfs.get(r["path"])
            if df is None:
                continue
            sig_in  = can_in(df)
            sig_out = get_eta(df, OUT_PROBE)
            if sig_in is not None:
                ts, A = sliding_afft(sig_in, f, window_s)
                curves[(f, wind, "IN", r["path"])] = (ts, A)
            if sig_out is not None:
                ts, A = sliding_afft(sig_out, f, window_s)
                curves[(f, wind, "OUT", r["path"])] = (ts, A)
        print(f"  f={f} Hz, wind={wind}: {len(rs)} runs")


# ── Geometry ────────────────────────────────────────────────────────────
def t_arr(r_m, f):     return r_m / c_group(f, TANK_DEPTH_M)
def t_paras(r_m, f):   return r_m / c_group(2.0 * f, TANK_DEPTH_M)


def c_phase(f_hz, h_m=TANK_DEPTH_M, g=9.81):
    """Phase velocity (celerity) under full dispersion ω² = gk·tanh(kh)."""
    from scipy.optimize import brentq
    omega = 2.0 * np.pi * f_hz
    k_deep = omega ** 2 / g
    if k_deep * h_m > 10.0:
        return g / (2.0 * np.pi * f_hz)
    def _disp(k): return omega ** 2 - g * k * np.tanh(k * h_m)
    k = brentq(_disp, 1e-4, 200.0)
    return float(np.sqrt(g / k * np.tanh(k * h_m)))


def t_refl(r_m, f):
    """Time at which the reflection off the back wall returns to a probe at r,
    measured from wavemaker start. Forward leg r→L plus return leg (L−r)→r:
        t_refl = (L + (L − r)) / c_phase = (2L − r) / c_phase
    """
    return (2.0 * TANK_LENGTH_M - r_m) / c_phase(f)


def t_seiche(r_m):
    """First-motion arrival at the probe — direct forward propagation of the
    long-wave (seiche-mode) disturbance launched at wavemaker start, at the
    shallow-water speed √(g·h). Frequency-independent.
        t_seiche = r / √(g·h)
    """
    return r_m / SEICHE_SPEED_M_S


# ── Figure: 4 rows × 2 cols ─────────────────────────────────────────────
fig, axes = plt.subplots(len(THESIS_FREQS), 2, figsize=(11, 13),
                         sharex=True, sharey=True)

PROBE_LABEL = {"IN": "IN  (canonical mean of 9373/170, 9373/340)",
               "OUT": "OUT  (12400/250)"}
PROBE_R_PANEL = {"IN": PROBE_R_M[IN_PROBES[0]], "OUT": PROBE_R_M[OUT_PROBE]}

for row_i, f in enumerate(THESIS_FREQS):
    L_T = N_LENGTH
    paddle_stop = PER40_PERIODS / f

    for col_i, probe_tag in enumerate(["IN", "OUT"]):
        ax = axes[row_i, col_i]
        r_m = PROBE_R_PANEL[probe_tag]
        t_arr_p   = t_arr(r_m, f)
        t_paras_p = t_paras(r_m, f)
        win_s     = t_arr_p + N_OFFSET / f
        win_e     = win_s + L_T / f

        for (ff, ww, pp, path), (ts, A) in curves.items():
            if ff != f or pp != probe_tag:
                continue
            color = WIND_COLOR_MAP[ww]
            ax.plot(ts, A, color=color, lw=0.9, alpha=0.7)

        # Option B window (green band)
        ax.axvspan(win_s, win_e, color="#2ECC71", alpha=0.20, lw=0)
        ax.axvline(win_s, color="#1A6E2A", ls="--", lw=0.7, alpha=0.7)
        ax.axvline(win_e, color="#1A6E2A", ls="--", lw=0.7, alpha=0.7)

        # Geometric reference times (subtle)
        t_refl_p   = t_refl(r_m, f)
        t_seiche_p = t_seiche(r_m)
        ax.axvline(paddle_stop, color="#7F3FBF", ls=":",  lw=0.9, alpha=0.7)
        ax.axvline(t_paras_p,   color="#D62728", ls=":",  lw=0.9, alpha=0.7)
        ax.axvline(t_refl_p,    color="#E67E22", ls="-.", lw=0.9, alpha=0.7)
        ax.axvline(t_seiche_p,  color="#17A2B8", ls="-.", lw=0.9, alpha=0.7)

        # Header in upper-left corner — frequency (Norwegian comma) + window length
        f_label = f"{f:.1f}".replace(".", ",")
        ax.text(0.012, 0.97,
                f"$f$ = {f_label} Hz\n{L_T} perioder",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=8, color="#222",
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec="#bbb", alpha=0.85, lw=0.4))

        ax.set_ylim(Y_LO, Y_HI)
        ax.set_xlim(SLIDING_T_LO_S, SLIDING_T_HI_S)
        ax.grid(True, alpha=0.25, lw=0.4)
        if row_i == len(THESIS_FREQS) - 1:
            ax.set_xlabel("vindusstart [s fra bølgemaker-start]", fontsize=9)
        if col_i == 0 and row_i == 0:
            ax.set_ylabel(r"$A_\mathrm{FFT}$  [mm]", fontsize=9,
                          rotation=0, ha="left", va="bottom")

# Single legend at the bottom
legend_handles = [
    Line2D([], [], color=WIND_COLOR_MAP["no"],   lw=1.2, label="uten vind"),
    Line2D([], [], color=WIND_COLOR_MAP["full"], lw=1.2, label="full vind"),
    Line2D([], [], color="#2ECC71", lw=8, alpha=0.4,
           label=fr"Tidsvindu [{N_OFFSET}T, {N_OFFSET + N_LENGTH}T]"),
    Line2D([], [], color="#7F3FBF", ls=":", lw=1.0,
           label=r"Bølgeskyver stoppet"),
    Line2D([], [], color="#D62728", ls=":", lw=1.0,
           label=r"andre harmoniske, $2f$"),
    Line2D([], [], color="#E67E22", ls="-.", lw=1.0,
           label=r"Refleksjon, $(2L-r)/c_\phi$"),
    Line2D([], [], color="#17A2B8", ls="-.", lw=1.0,
           label=r"Første bevegelse, $r/\sqrt{gh}$"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=3,
           fontsize=8, bbox_to_anchor=(0.5, -0.005), frameon=True)

fig.suptitle(
    f"$A_\\mathrm{{FFT}}(t)$ sliding plateau · $A_2$ ({TARGET_AMP:.2f} V) · "
    f"canon March-2026 cond4 lowrange",
    fontsize=10, y=0.995,
)
fig.tight_layout(rect=[0, 0.025, 1, 0.97])

# Align the horizontal y-axis label's left edge with the leftmost edge of the
# top-row y-tick labels (matplotlib API: get_window_extent → axes-fraction
# transform). Only the top-left panel carries the label since y-axis is shared.
fig.canvas.draw()
_renderer = fig.canvas.get_renderer()
_ax = axes[0, 0]
_ticks = [t for t in _ax.yaxis.get_ticklabels()
          if t.get_visible() and t.get_text().strip()]
if _ticks:
    _left_disp = min(t.get_window_extent(renderer=_renderer).x0 for t in _ticks)
    _x_axes = _ax.transAxes.inverted().transform((_left_disp, 0))[0]
    _ax.yaxis.set_label_coords(_x_axes, 1.02)

fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"\nsaved → {OUT_PDF.relative_to(BASE)}")
print(f"        {OUT_PNG.relative_to(BASE)}")
