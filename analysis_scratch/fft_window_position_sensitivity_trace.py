"""
Visual companion to `fft_window_position_sensitivity_lsfit.py`.
Show the probe time series with the sweep's 10T windows overlaid at
representative T_ref positions, so the reader can SEE where each window
sits in the signal and why the drift pattern emerges.

Layout: 2 rows (nowind, fullwind) × 2 cols (IN probe 9373/170, OUT probe
12400/250). Same (f, A): 1.4 Hz / 0.2 V per240. Windows coloured by
T_ref, pipeline default (T_ref=50) highlighted.

Output: analysis_scratch/fft_window_position_sensitivity_trace.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from wavescripts.improved_data_loader import load_analysis_data, load_processed_dfs
from wavescripts.constants import HG, c_group
from wavescripts.plot_utils import apply_thesis_style

apply_thesis_style()

FS = 250.0
BASE = Path(__file__).parent.parent
SCRATCH = Path(__file__).parent
OUT_PNG = SCRATCH / "fft_window_position_sensitivity_trace.png"

TARGET_DIRS = [
    BASE / "waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange",
    BASE / "waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange",
]

TARGET_FREQ = 1.4
TARGET_AMP  = 0.2

IN_PROBE  = "9373/170"
OUT_PROBE = "12400/250"

PROBE_R_M = {IN_PROBE: 9.373, OUT_PROBE: 12.400}

# Positions to overlay (every 5 periods across the sweep range)
T_REF_OVERLAYS = [40, 45, 50, 55, 60, 65, 70, 75, 80]
T_REF_CANON = HG.START_T_REF  # 50
N_PERIODS = 10

# Time range for the plot
T_PLOT_START_S = 10.0            # fixed axis start
# T_plot_end computed dynamically (last window end + buffer)
T_PLOT_BUFFER_AFTER_LAST_WINDOW_S = 10.0


def pick_run(meta, wind):
    mask = (
        (meta["PanelCondition"] == "full")
        & np.isclose(meta["WaveFrequencyInput [Hz]"], TARGET_FREQ, atol=0.01)
        & np.isclose(meta["WaveAmplitudeInput [Volt]"], TARGET_AMP, atol=0.01)
        & (meta["WindCondition"] == wind)
        & (meta["WavePeriodInput"] >= 100)
        & (meta["quality_flag"] == "ok")
    )
    sub = meta[mask]
    if sub.empty:
        return None
    return sub.iloc[0]


print("Loading …")
dirs_str = [str(d) for d in TARGET_DIRS]
meta, _, _, _ = load_analysis_data(*dirs_str, load_processed=False)
proc = load_processed_dfs(*dirs_str)

runs = {}
for wind in ["no", "full"]:
    row = pick_run(meta, wind)
    if row is None:
        raise RuntimeError(f"no {wind} run at f={TARGET_FREQ} Hz, A={TARGET_AMP} V")
    runs[wind] = row
    print(f"  {wind}: {Path(row['path']).name}")


fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex="col", sharey="row")

cmap = plt.get_cmap("viridis")
t_ref_norm = (np.array(T_REF_OVERLAYS) - min(T_REF_OVERLAYS)) / (max(T_REF_OVERLAYS) - min(T_REF_OVERLAYS))

samples_per_period = int(round(FS / TARGET_FREQ))

for col_idx, (probe, r_m) in enumerate([(IN_PROBE, PROBE_R_M[IN_PROBE]),
                                         (OUT_PROBE, PROBE_R_M[OUT_PROBE])]):
    # Probe-local ΔT
    dT_periods = (HG.REF_R_M - r_m) / c_group(TARGET_FREQ, HG.TANK_DEPTH_M) * TARGET_FREQ
    dT_seconds = dT_periods / TARGET_FREQ

    # last window end in probe-local seconds:  (T_ref_max - ΔT) / f + N_PERIODS / f
    t_last_win_end_s = ((max(T_REF_OVERLAYS) - dT_periods) + N_PERIODS) / TARGET_FREQ
    t_plot_end_s = t_last_win_end_s + T_PLOT_BUFFER_AFTER_LAST_WINDOW_S

    for row_idx, wind in enumerate(["no", "full"]):
        ax = axes[row_idx, col_idx]
        row = runs[wind]
        df = proc[row["path"]]

        eta_col = f"eta_{probe}_interp" if f"eta_{probe}_interp" in df.columns else f"eta_{probe}"
        sig = df[eta_col].to_numpy(dtype=float)   # already in mm
        t = np.arange(len(sig)) / FS

        # Restrict display range
        mask = (t >= T_PLOT_START_S) & (t <= t_plot_end_s)
        ax.plot(t[mask], sig[mask], "-", color="#333", lw=0.55, alpha=0.9)

        # Y-axis: use percentiles of the PLATEAU region only (exclude pre-arrival spikes).
        # Plateau region = from first overlay window start to last overlay window end.
        t_plateau_start = (min(T_REF_OVERLAYS) - dT_periods) / TARGET_FREQ
        t_plateau_end   = t_last_win_end_s
        plateau_mask = (t >= t_plateau_start) & (t <= t_plateau_end)
        sig_plateau = sig[plateau_mask]
        y_lo, y_hi = np.nanpercentile(sig_plateau, [0.5, 99.5])
        y_pad = 0.25 * (y_hi - y_lo)
        y_lo -= y_pad
        y_hi += y_pad

        for T_ref, cnorm in zip(T_REF_OVERLAYS, t_ref_norm):
            start_T_probe = T_ref - dT_periods
            start_s = start_T_probe / TARGET_FREQ
            end_s = start_s + N_PERIODS / TARGET_FREQ
            is_canon = (T_ref == T_REF_CANON)
            color = cmap(cnorm)
            rect = Rectangle(
                (start_s, y_lo), end_s - start_s, y_hi - y_lo,
                fill=True,
                facecolor=color, alpha=0.09 if not is_canon else 0.30,
                edgecolor="black" if is_canon else color,
                lw=2.2 if is_canon else 0.5,
                zorder=1,
            )
            ax.add_patch(rect)
            # Alternate vertical label height so they don't collide
            y_label_high = y_hi - 0.04 * (y_hi - y_lo)
            y_label_low  = y_lo + 0.04 * (y_hi - y_lo)
            y_text = y_label_high if (T_ref // 5) % 2 == 0 else y_label_low
            label = f"{T_ref}T" + ("★" if is_canon else "")
            ax.text(start_s + (end_s - start_s) / 2, y_text,
                    label, ha="center", va="center", fontsize=8,
                    color="black" if is_canon else "#222",
                    weight="bold" if is_canon else "normal",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color, lw=0.8, alpha=0.9))

        ax.set_xlim(T_PLOT_START_S, t_plot_end_s)
        ax.set_ylim(y_lo, y_hi)
        ax.grid(True, alpha=0.3)

        ax.set_title("", fontsize=10)
        if col_idx == 0:
            ax.set_ylabel("η (mm)")
        if row_idx == 1:
            ax.set_xlabel("time from wavemaker start (s)")

# Add colorbar
sm = plt.cm.ScalarMappable(cmap=cmap,
                            norm=plt.Normalize(vmin=min(T_REF_OVERLAYS), vmax=max(T_REF_OVERLAYS)))
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.85, pad=0.02, aspect=30)
cbar.set_label("T_ref (periods from wavemaker start, OUT-probe equivalent)")

fig.suptitle(
    "",
    fontsize=11,
)

fig.savefig(OUT_PNG, dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"Wrote {OUT_PNG.relative_to(BASE)}")
