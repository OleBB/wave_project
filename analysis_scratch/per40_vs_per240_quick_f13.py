"""
Quick per40 vs per240 overlay at 1.3 Hz, 0.2 V, fullwind.
==========================================================

Just plots η(t) for the canon March-2026 cond4 lowrange runs at this one
condition. No pooling, no fancy metric — just raw signals overlaid, with
geometric reference times marked. Lets the user eyeball where per40 and
per240 actually diverge.

For IN, the canonical IN signal (mean of 9373/170 + 9373/340) is used.

Reference times annotated on each panel:
    t_arr        — main-wave arrival, r / c_g(f)
    paddle_stop  — wavemaker turns off (per40 only), 40/f
    t_back       — last per40 wave reaches probe, paddle_stop + r/c_g
    t_paras      — free 2nd-harmonic arrival, r / c_g(2f)
    Option B     — proposed FFT window [t_arr+10/f, t_arr+(10+N)/f]; N=10 here

Outputs (scratch only):
    analysis_scratch/per40_vs_per240_quick_f13.{pdf,png}

Run from repo root:
    /opt/anaconda3/envs/draumkvedet/bin/python analysis_scratch/per40_vs_per240_quick_f13.py
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
from wavescripts.constants import c_group, HG
from wavescripts.plot_utils import apply_thesis_style, WIND_COLOR_MAP

apply_thesis_style()

# ── Config ──────────────────────────────────────────────────────────────
FS               = 250.0
TARGET_FREQ      = 1.3
TARGET_AMP       = 0.2
TARGET_WIND      = "full"
PER240_THRESHOLD_T = 50

IN_PROBES = ["9373/170", "9373/340"]
OUT_PROBE = "12400/250"
PROBE_R_M = {"9373/170": 9.373, "9373/340": 9.373, "12400/250": 12.400}
TANK_DEPTH_M = HG.TANK_DEPTH_M

# Option B window (for visualisation only)
N_OFFSET = 10
N_LEN_AT_F = {1.3: 10, 1.4: 13, 1.5: 13, 1.6: 13}[TARGET_FREQ]

PER40_PERIODS = 40

# Canon — march-2026 cond4 lowrange
PROCESSED_DIRS = [
    Path("waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange"),
    Path("waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"),
]

X_LO_S = 10.0
X_HI_S = 55.0


# ── Helpers ─────────────────────────────────────────────────────────────
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


# ── Load ────────────────────────────────────────────────────────────────
print("Loading meta + processed_dfs (canon March-2026 lowrange) …")
combined_meta, _, _, _ = load_analysis_data(*PROCESSED_DIRS, load_processed=False)
processed_dfs = load_processed_dfs(*PROCESSED_DIRS)

f_col = pd.to_numeric(combined_meta["WaveFrequencyInput [Hz]"], errors="coerce")
a_col = pd.to_numeric(combined_meta["WaveAmplitudeInput [Volt]"], errors="coerce")
mask = (
    np.isclose(f_col, TARGET_FREQ, atol=0.02)
    & np.isclose(a_col, TARGET_AMP, atol=0.01)
    & (combined_meta["WindCondition"] == TARGET_WIND)
    & (combined_meta["PanelCondition"] == "full")
    & (combined_meta["quality_flag"] == "ok")
)
sel = combined_meta[mask].copy()
sel["N_input_periods"] = pd.to_numeric(sel["WavePeriodInput"], errors="coerce")
sel["run_type"] = np.where(sel["N_input_periods"] >= PER240_THRESHOLD_T, "per240", "per40")
print(f"\n{len(sel)} canon runs at {TARGET_FREQ} Hz, {TARGET_AMP} V, {TARGET_WIND} wind:")
for _, r in sel.iterrows():
    print(f"  [{r['run_type']:>6}] {Path(str(r['path'])).name}")


# ── Geometry ────────────────────────────────────────────────────────────
t_arr_in  = PROBE_R_M[IN_PROBES[0]] / c_group(TARGET_FREQ, TANK_DEPTH_M)
t_arr_out = PROBE_R_M[OUT_PROBE]   / c_group(TARGET_FREQ, TANK_DEPTH_M)
t_paras_in  = PROBE_R_M[IN_PROBES[0]] / c_group(2 * TARGET_FREQ, TANK_DEPTH_M)
t_paras_out = PROBE_R_M[OUT_PROBE]   / c_group(2 * TARGET_FREQ, TANK_DEPTH_M)
t_paddle_stop = PER40_PERIODS / TARGET_FREQ
t_back_in   = t_paddle_stop + t_arr_in
t_back_out  = t_paddle_stop + t_arr_out

win_start_in  = t_arr_in  + N_OFFSET / TARGET_FREQ
win_end_in    = win_start_in  + N_LEN_AT_F / TARGET_FREQ
win_start_out = t_arr_out + N_OFFSET / TARGET_FREQ
win_end_out   = win_start_out + N_LEN_AT_F / TARGET_FREQ

print(f"\nGeometry at {TARGET_FREQ} Hz:")
print(f"  IN  (r={PROBE_R_M[IN_PROBES[0]]} m): t_arr={t_arr_in:.2f}, t_paras={t_paras_in:.2f}, "
      f"t_back={t_back_in:.2f}, win=[{win_start_in:.2f}, {win_end_in:.2f}]")
print(f"  OUT (r={PROBE_R_M[OUT_PROBE]} m): t_arr={t_arr_out:.2f}, t_paras={t_paras_out:.2f}, "
      f"t_back={t_back_out:.2f}, win=[{win_start_out:.2f}, {win_end_out:.2f}]")


# ── Plot ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

COL_PER40  = "#D6691E"   # warm orange — short runs
COL_PER240 = "#1F4E79"   # deep blue   — long reference

PROBE_LABEL = {"IN": "IN  (canonical mean of 9373/170, 9373/340)",
               "OUT": "OUT  (12400/250)"}

for ax, probe_tag, t_arr, t_paras, t_back, win_start, win_end in [
    (axes[0], "IN",  t_arr_in,  t_paras_in,  t_back_in,  win_start_in,  win_end_in),
    (axes[1], "OUT", t_arr_out, t_paras_out, t_back_out, win_start_out, win_end_out),
]:
    for _, r in sel.iterrows():
        df = processed_dfs.get(r["path"])
        if df is None:
            continue
        sig = can_in(df) if probe_tag == "IN" else get_eta(df, OUT_PROBE)
        if sig is None:
            continue
        t = np.arange(len(sig)) / FS
        m = (t >= X_LO_S) & (t <= X_HI_S)
        col = COL_PER40 if r["run_type"] == "per40" else COL_PER240
        ls  = "--" if r["run_type"] == "per40" else "-"
        lw  = 0.9
        alpha = 0.85
        label = f"{r['run_type']}  {Path(str(r['path'])).parent.name[:8]}"
        ax.plot(t[m], sig[m], color=col, ls=ls, lw=lw, alpha=alpha, label=label)

    # Geometric reference times
    ax.axvline(t_arr,         color="#1F77B4", ls="-",  lw=1.0, alpha=0.65)
    ax.axvline(t_paddle_stop, color="#7F3FBF", ls=":",  lw=1.0, alpha=0.7)
    ax.axvline(t_back,        color="#7F3FBF", ls="-",  lw=1.0, alpha=0.5)
    ax.axvline(t_paras,       color="#D62728", ls=":",  lw=1.0, alpha=0.65)
    ax.axvspan(win_start, win_end, color="#2ECC71", alpha=0.15, lw=0)
    ax.axvline(win_start, color="#1A6E2A", ls="--", lw=0.9, alpha=0.7)
    ax.axvline(win_end,   color="#1A6E2A", ls="--", lw=0.9, alpha=0.7)

    # Annotations along the top
    yhi = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1
    ax.text(t_arr,         0.98, f"$t_\\mathrm{{arr}}$ {t_arr:.1f}",
            transform=ax.get_xaxis_transform(), fontsize=7, color="#1F77B4",
            ha="left", va="top", rotation=90)
    ax.text(t_paddle_stop, 0.98, f"paddle stop {t_paddle_stop:.1f}",
            transform=ax.get_xaxis_transform(), fontsize=7, color="#7F3FBF",
            ha="left", va="top", rotation=90)
    ax.text(t_back,        0.98, f"$t_\\mathrm{{back}}$ {t_back:.1f}",
            transform=ax.get_xaxis_transform(), fontsize=7, color="#7F3FBF",
            ha="left", va="top", rotation=90)
    ax.text(t_paras,       0.98, f"$t_\\mathrm{{2f}}$ {t_paras:.1f}",
            transform=ax.get_xaxis_transform(), fontsize=7, color="#D62728",
            ha="left", va="top", rotation=90)
    ax.text(0.5*(win_start+win_end), 0.04, f"Option B window\n[{win_start:.1f}, {win_end:.1f}] s",
            transform=ax.get_xaxis_transform(), fontsize=7, color="#1A6E2A",
            ha="center", va="bottom")

    ax.set_xlim(X_LO_S, X_HI_S)
    ax.grid(True, alpha=0.25, lw=0.4)
    ax.set_ylabel(f"{PROBE_LABEL[probe_tag]}\n$\\eta$ [mm]", fontsize=9)
    ax.legend(fontsize=7, loc="lower right", ncol=2)

axes[1].set_xlabel("time from wavemaker start [s]", fontsize=10)
fig.suptitle(f"per40 vs per240 — $f$ = {TARGET_FREQ} Hz, $A_2$ (0.2 V), fullwind, full panel  "
             "(canon March-2026 cond4 lowrange)",
             fontsize=10, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.97])

OUT_PDF = Path(__file__).parent / "per40_vs_per240_quick_f13.pdf"
OUT_PNG = Path(__file__).parent / "per40_vs_per240_quick_f13.png"
fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
plt.close(fig)

print(f"\nsaved → {OUT_PDF.relative_to(BASE)}")
print(f"        {OUT_PNG.relative_to(BASE)}")
