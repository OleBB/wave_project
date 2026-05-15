"""
Plateau overview — sliding A_FFT(t) per (f, probe) at A_1 / A_2 / A_3.
=======================================================================

Reader-facing CH04 §4o figure. One figure per amplitude tier (A_1, A_2,
A_3 = 0.10, 0.20, 0.30 V), each laid out as 4 rows (f = 1.3, 1.4, 1.5,
1.6 Hz) × 2 cols (IN, OUT). Each panel overlays both wind conditions
(blue=nowind, red=fullwind, project WIND_COLOR_MAP). One thin line per
canon run (cond4 March-2026 lowrange, full panel, quality_ok). No
per40/per240 split — pooled per cell.

Sliding A_FFT(t) at the paddle frequency, window length = N_LENGTH·T,
step 0.1 s. Window placement matches the post-squeeze choice
(`N_OFFSET = 7`, `N_LENGTH = 10` — uniform across all four thesis freqs).

Visual claim: in every (f, probe, wind, amp) cell, the chosen window
(green band) sits inside a flat A_FFT plateau.

Outputs (per amp tier):
    output/FIGURES/ch04_plateau_overview_A{1,2,3}.pdf
    output/TEXFIGU/ch04_plateau_overview_A{1,2,3}.tex
    analysis_scratch/plateau_overview_A{1,2,3}.{pdf,png}

Run from repo root:
    /opt/anaconda3/envs/draumkvedet/bin/python analysis_scratch/plateau_overview.py
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
import wavescripts.plot_utils as pu

apply_thesis_style()


# ── Config ──────────────────────────────────────────────────────────────
FS               = 250.0
FFT_BAND_HZ      = 0.05

THESIS_FREQS     = [1.3, 1.4, 1.5, 1.6]
N_OFFSET         = 7
N_LENGTH         = 10                     # uniform across all thesis freqs. WINDOW SECONDS IS N/F

TANK_LENGTH_M    = 25.0   # wavemaker → back wall (round trip = 2L = 50 m)
SEICHE_SPEED_M_S = (9.81 * 0.58) ** 0.5   # √(g·h) ≈ 2.385 m/s

IN_PROBES        = ["9373/170", "9373/340"]
OUT_PROBE        = "12400/250"
PROBE_R_M        = {"9373/170": 9.373, "9373/340": 9.373, "12400/250": 12.400}
TANK_DEPTH_M     = HG.TANK_DEPTH_M
PER40_PERIODS    = 40

SLIDING_STEP_S   = 0.2 #changed from 0.1 to 0.2 testing, and tried 0.01. 02 was good enough it seems
SLIDING_T_LO_S   = 0.0
SLIDING_T_HI_S   = 55.0

WINDS            = ["no", "full"]

# Amplitude tiers — (target voltage, label, file tag, y-axis upper limit)
AMP_TIERS = [
    (0.10, r"A_1", "A1", 10.0),
    (0.20, r"A_2", "A2", 18.0),
    (0.30, r"A_3", "A3", 28.0),
]

PROCESSED_DIRS = [
    Path("waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange"),
    Path("waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"),
]

SCRATCH_DIR  = Path(__file__).parent
THESIS_FIGS  = BASE / "output" / "FIGURES"
THESIS_STUBS = BASE / "output" / "TEXFIGU"
THESIS_FIGS.mkdir(parents=True, exist_ok=True)
THESIS_STUBS.mkdir(parents=True, exist_ok=True)


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
    step  = int(round(SLIDING_STEP_S * FS)) #x2
    n_lo  = int(round(SLIDING_T_LO_S * FS))
    n_hi  = min(len(signal) - N_win, int(round(SLIDING_T_HI_S * FS)))
    if N_win >= len(signal) or n_hi <= n_lo:
        return np.array([]), np.array([])
    starts = np.arange(n_lo, n_hi + 1, step)
    ts = starts / FS
    A = np.array([fft_amp(signal[s:s + N_win], target_hz) for s in starts])
    return ts, A


def t_arr_at(r_m, f):     return r_m / c_group(f, TANK_DEPTH_M)
def t_paras_at(r_m, f):   return r_m / c_group(2.0 * f, TANK_DEPTH_M)


def c_phase(f_hz, h_m=TANK_DEPTH_M, g=9.81):
    """Phase speed (celerity) under full dispersion ω² = gk·tanh(kh)."""
    from scipy.optimize import brentq
    omega = 2.0 * np.pi * f_hz
    k_deep = omega ** 2 / g
    if k_deep * h_m > 10.0:
        return g / (2.0 * np.pi * f_hz)
    def _disp(k): return omega ** 2 - g * k * np.tanh(k * h_m)
    k = brentq(_disp, 1e-4, 200.0)
    return float(np.sqrt(g / k * np.tanh(k * h_m)))


def t_refl_at(r_m, f): return (2.0 * TANK_LENGTH_M - r_m) / c_phase(f)
def t_seiche_at(r_m):  return r_m / SEICHE_SPEED_M_S


# ── Load (once, all amp tiers share the same combined_meta) ─────────────
print("Loading meta + processed_dfs (canon March-2026 lowrange) …")
combined_meta, _, _, _ = load_analysis_data(*PROCESSED_DIRS, load_processed=False)
processed_dfs = load_processed_dfs(*PROCESSED_DIRS)


# ── Per-tier figure builder ─────────────────────────────────────────────
def build_figure_for(target_amp: float, amp_label_tex: str,
                     amp_tag: str, y_hi: float):
    figure_name = f"ch04_plateau_overview_{amp_tag}"

    # Filter canon runs at this amplitude tier
    f_col = pd.to_numeric(combined_meta["WaveFrequencyInput [Hz]"], errors="coerce")
    a_col = pd.to_numeric(combined_meta["WaveAmplitudeInput [Volt]"], errors="coerce")
    mask = (
        np.isclose(a_col, target_amp, atol=0.01)
        & f_col.between(THESIS_FREQS[0] - 0.02, THESIS_FREQS[-1] + 0.02)
        & (combined_meta["PanelCondition"] == "full")
        & combined_meta["WindCondition"].isin(WINDS)
        # Match ch05_damping_freq's quality gate (apply_experimental_filters
        # default): keep "ok" plus "probe_malfunction_secondary".
        & combined_meta["quality_flag"].isin(["ok", "probe_malfunction_secondary"])
    )
    sel = combined_meta[mask].copy()
    # Single-probe dropout filter — match ch05_damping_freq
    # (main_save_figures.py ~line 2207). K_t from a single IN probe can't
    # exceed 1: a transmissive panel always damps. K_t,probe > 1 means that
    # probe's amplitude registered below the OUT probe — a single-probe
    # dropout (the 9373/170 probe is known to drop out at higher frequencies).
    _kt_wall = sel[f"Probe {OUT_PROBE} Amplitude (FFT)"] / sel[f"Probe {IN_PROBES[0]} Amplitude (FFT)"]
    _kt_far  = sel[f"Probe {OUT_PROBE} Amplitude (FFT)"] / sel[f"Probe {IN_PROBES[1]} Amplitude (FFT)"]
    _dropout = (_kt_wall > 1.0) | (_kt_far > 1.0)
    if _dropout.any():
        print(f"   Dropping {_dropout.sum()} runs with K_t,probe > 1 (single-probe dropout):")
        for _, _r in sel[_dropout].iterrows():
            print(f"     Kt_wall={_kt_wall[_r.name]:.3f} Kt_far={_kt_far[_r.name]:.3f}  {_r['path'].split('/')[-1]}")
        sel = sel[~_dropout].copy()
    print(f"\n— {amp_tag} ({target_amp:.2f} V): {len(sel)} canon runs.")

    # Sliding A_FFT per (f, wind, probe, run)
    curves = {}
    for f in THESIS_FREQS:
        window_s = N_LENGTH / f
        for wind in WINDS:
            rs = sel[
                np.isclose(pd.to_numeric(sel["WaveFrequencyInput [Hz]"],
                                         errors="coerce"), f, atol=0.02)
                & (sel["WindCondition"] == wind)
            ]
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
            print(f"   f={f} Hz, wind={wind}: {len(rs)} runs")

    # ── Build figure ────────────────────────────────────────────────────
    # Sized for an A4 page with 1-inch margins (text block 6.27 x 9.69 in).
    # At \includegraphics[width=\linewidth] only the aspect ratio matters:
    # h/w = 15.3/11 = 1.39 -> rendered height 6.27 * 1.39 = 8.72 in, leaving
    # ~0.97 in of text height below the figure for a 2-line 12 pt caption.
    fig, axes = plt.subplots(len(THESIS_FREQS), 2, figsize=(11, 15.3),
                             sharex=True, sharey=True)
    PROBE_R_PANEL = {"IN": PROBE_R_M[IN_PROBES[0]], "OUT": PROBE_R_M[OUT_PROBE]}

    for row_i, f in enumerate(THESIS_FREQS):
        paddle_stop = PER40_PERIODS / f
        for col_i, probe_tag in enumerate(["IN", "OUT"]):
            ax = axes[row_i, col_i]
            r_m = PROBE_R_PANEL[probe_tag]
            t_arr_p   = t_arr_at(r_m, f)
            t_paras_p = t_paras_at(r_m, f)
            win_s     = t_arr_p + N_OFFSET / f
            win_e     = win_s + N_LENGTH / f

            for (ff, ww, pp, _), (ts, A) in curves.items():
                if ff != f or pp != probe_tag:
                    continue
                ax.plot(ts, A, color=WIND_COLOR_MAP[ww], lw=0.9, alpha=0.7)

            # Window band
            ax.axvspan(win_s, win_e, color="#2ECC71", alpha=0.22, lw=0)
            ax.axvline(win_s, color="#1A6E2A", ls="--", lw=1.4, alpha=0.85)
            ax.axvline(win_e, color="#1A6E2A", ls="--", lw=1.4, alpha=0.85)

            # Geometric reference times
            ax.axvline(paddle_stop,    color="#7F3FBF", ls=":",  lw=1.7, alpha=0.85)
            ax.axvline(t_paras_p,      color="#D62728", ls=":",  lw=1.7, alpha=0.85)
            ax.axvline(t_refl_at(r_m, f), color="#E67E22", ls="-.", lw=1.7, alpha=0.85)
            ax.axvline(t_seiche_at(r_m),  color="#17A2B8", ls="-.", lw=1.7, alpha=0.85)

            # In-pane corner label (Norwegian comma)
            f_label = f"{f:.1f}".replace(".", ",")
            ax.text(0.012, 0.97,
                    f"$f$ = {f_label} Hz\n{N_LENGTH} perioder",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=10, color="#222",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white",
                              ec="#bbb", alpha=0.85, lw=0.4))

            ax.set_ylim(0.0, y_hi)
            ax.set_xlim(SLIDING_T_LO_S, SLIDING_T_HI_S)
            ax.grid(True, alpha=0.25, lw=0.4)
            ax.tick_params(labelsize=10)
            if row_i == len(THESIS_FREQS) - 1:
                ax.set_xlabel("[s]", fontsize=12)
            if col_i == 0 and row_i == 0:
                ax.set_ylabel(r"$A$ [mm]", fontsize=12,
                              rotation=0, ha="left", va="bottom")

    legend_handles = [
        Line2D([], [], color=WIND_COLOR_MAP["no"],   lw=1.2, label="uten vind"),
        Line2D([], [], color=WIND_COLOR_MAP["full"], lw=1.2, label="full vind"),
        Line2D([], [], color="#2ECC71", lw=8, alpha=0.45,
               label=fr"Tidsvindu [{N_OFFSET}T, {N_OFFSET + N_LENGTH}T]"),
        Line2D([], [], color="#7F3FBF", ls=":", lw=1.7,
               label=r"Bølgeskyver stoppet"),
        Line2D([], [], color="#D62728", ls=":", lw=1.7,
               label=r"Andre harmoniske, $2f$"),
        Line2D([], [], color="#E67E22", ls="-.", lw=1.7,
               label=r"Refleksjon, $(2L-r)/c$"),
        Line2D([], [], color="#17A2B8", ls="-.", lw=1.7,
               label=r"Første bevegelse, $r/\sqrt{gh}$"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               fontsize=12, bbox_to_anchor=(0.5, 0.008), frameon=True)
    # Reserve the bottom 6 % of the figure for the legend so it sits clear
    # below the bottom-row x-axis instead of overlapping the "[s]" labels.
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])

    # Align horizontal y-label's left edge with leftmost tick label edge
    fig.canvas.draw()
    _renderer = fig.canvas.get_renderer()
    _ax0 = axes[0, 0]
    _ticks = [t for t in _ax0.yaxis.get_ticklabels()
              if t.get_visible() and t.get_text().strip()]
    if _ticks:
        _left_disp = min(t.get_window_extent(renderer=_renderer).x0 for t in _ticks)
        _x_axes = _ax0.transAxes.inverted().transform((_left_disp, 0))[0]
        _ax0.yaxis.set_label_coords(_x_axes, 1.02)

    scratch_pdf = SCRATCH_DIR / f"plateau_overview_{amp_tag}.pdf"
    scratch_png = SCRATCH_DIR / f"plateau_overview_{amp_tag}.png"
    thesis_pdf  = THESIS_FIGS / f"{figure_name}.pdf"

    fig.savefig(scratch_pdf, bbox_inches="tight")
    fig.savefig(scratch_png, dpi=130, bbox_inches="tight")
    fig.savefig(thesis_pdf,  bbox_inches="tight")
    plt.close(fig)
    print(f"   scratch → {scratch_pdf.relative_to(BASE)}")
    print(f"   thesis  → {thesis_pdf.relative_to(BASE)}")

    # TEXFIGU stub
    pu.TEXFIGU_DIR = THESIS_STUBS
    pu.FIGURES_DIR = THESIS_FIGS
    _meta_stub = pu.build_fig_meta(
        {
            "filters": {
                "WaveAmplitudeInput [Volt]": target_amp,
                "PanelCondition":            "full",
                "WindCondition":             ["no", "full"],
                "quality_flag":              "ok",
                "probes":                    "IN=9373/170+9373/340, OUT=12400/250",
            },
            "plotting": {"figure_name": figure_name},
        },
        chapter="04",
        data_df=sel,
        extra={"script": "analysis_scratch/plateau_overview.py"},
        computed_in=(
            "analysis_scratch/plateau_overview.py "
            f"(sliding A_FFT, window N_off={N_OFFSET}T, N_len={N_LENGTH}T, "
            f"step {SLIDING_STEP_S}s, FFT band ±{FFT_BAND_HZ} Hz)"
        ),
        data_class="DFS",
        grouper="all canon runs at this (amp) — pooled per (f, wind, probe)",
        collapse_panels=False,
        extra_params=(
            f"target_amp = {target_amp} V ({amp_tag}); "
            f"freqs = {THESIS_FREQS}; "
            f"r_IN = {PROBE_R_M[IN_PROBES[0]]} m, r_OUT = {PROBE_R_M[OUT_PROBE]} m; "
            f"L_tank = {TANK_LENGTH_M} m; depth = {TANK_DEPTH_M} m; "
            f"seiche speed = {SEICHE_SPEED_M_S:.3f} m/s"
        ),
    )
    pu.write_figure_stub(_meta_stub, plot_type="plateau_overview")
    print(f"   stub    → output/TEXFIGU/{figure_name}.tex")


# ── Loop over the three amp tiers ───────────────────────────────────────
for _amp_v, _amp_tex, _amp_tag, _y_hi in AMP_TIERS:
    build_figure_for(_amp_v, _amp_tex, _amp_tag, _y_hi)
