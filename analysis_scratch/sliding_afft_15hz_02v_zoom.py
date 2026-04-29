"""
Zoom on the 1.5 Hz / 0.2 V fullwind per240 anomaly at 9373/170
==============================================================

The parent sweep (sliding_afft_fullwind_sweep.py) found that 7/8 conditions
have pipeline_AFFT ≈ matched_mid_AFFT (no within-run transient) — but
1.5 Hz / 0.2 V fullwind is a genuine transient: pipeline reads 14.5 mm
at the start, mid-slice reads 12.8 mm, plateau drops to 7.3 mm.

This script zooms on all fullwind per240 runs at 1.5 Hz / 0.2 V in meta
(4 runs across 20260314, 20260323, 20260326, 20260327), plus the nowind
reference at the same frequency/amplitude, and plots for each:
  - top: η time series (eta_9373/170)
  - middle: sliding AFFT at 1.5 Hz (15 s window)
  - bottom: sliding Atd (time-domain (P97.5 − P2.5)/2, 15 s window)

If the anomaly is consistent across runs, the wavemaker produces a real
early burst at this specific frequency/amplitude under fullwind. If only
one run shows it, it is a noise event or data quality issue.

Output:
    analysis_scratch/sliding_afft_15hz_02v_zoom.png
    analysis_scratch/sliding_afft_15hz_02v_zoom_findings.md
"""

import sys, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wavescripts.improved_data_loader import load_analysis_data, load_processed_dfs
from wavescripts.plot_utils import apply_thesis_style

apply_thesis_style()

FS                   = 250.0
PROBE                = "9373/170"
FREQ                 = 1.5
AMP                  = 0.2
FFT_WINDOW_HZ        = 0.10
SLIDING_WINDOW_S     = 15.0
SLIDING_STEP_S       = 1.0

BASE    = Path(__file__).parent.parent
OUT_PNG = Path(__file__).parent / "sliding_afft_15hz_02v_zoom.png"
OUT_MD  = Path(__file__).parent / "sliding_afft_15hz_02v_zoom_findings.md"


def sliding_afft(signal: np.ndarray, target_freq: float = FREQ,
                 fs: float = FS, window_s: float = SLIDING_WINDOW_S,
                 step_s: float = SLIDING_STEP_S,
                 search_window_hz: float = FFT_WINDOW_HZ):
    N_win = int(round(window_s * fs))
    step  = int(round(step_s   * fs))
    if N_win >= len(signal):
        return np.array([]), np.array([])
    freqs = np.fft.fftfreq(N_win, d=1.0 / fs)
    pos_mask = freqs > 0
    pos_freqs = freqs[pos_mask]
    bin_mask = (pos_freqs >= target_freq - search_window_hz) & \
               (pos_freqs <= target_freq + search_window_hz)
    masked_freqs = pos_freqs[bin_mask]
    nearest_idx = int(np.argmin(np.abs(masked_freqs - target_freq))) if bin_mask.any() else 0

    starts = np.arange(0, len(signal) - N_win + 1, step)
    t_centers = (starts + N_win / 2) / fs
    amps = np.full_like(t_centers, np.nan, dtype=float)
    for i, s in enumerate(starts):
        seg = signal[s:s + N_win]
        if np.isnan(seg).any():
            if np.isnan(seg).mean() > 0.10:
                continue
            idx = np.arange(len(seg))
            seg = np.interp(idx, idx[~np.isnan(seg)], seg[~np.isnan(seg)])
        fft_vals = np.fft.fft(seg)
        amps_pos = 2.0 * np.abs(fft_vals[pos_mask]) / N_win
        amps[i] = amps_pos[bin_mask][nearest_idx]
    return t_centers, amps


def sliding_atd(signal: np.ndarray, fs: float = FS,
                window_s: float = SLIDING_WINDOW_S,
                step_s: float = SLIDING_STEP_S):
    """Time-domain amplitude = (P97.5 − P2.5)/2 over a sliding window."""
    N_win = int(round(window_s * fs))
    step  = int(round(step_s   * fs))
    if N_win >= len(signal):
        return np.array([]), np.array([])
    starts = np.arange(0, len(signal) - N_win + 1, step)
    t_centers = (starts + N_win / 2) / fs
    amps = np.full_like(t_centers, np.nan, dtype=float)
    for i, s in enumerate(starts):
        seg = signal[s:s + N_win]
        if np.isnan(seg).mean() > 0.10:
            continue
        amps[i] = (np.nanpercentile(seg, 97.5) - np.nanpercentile(seg, 2.5)) / 2
    return t_centers, amps


# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading metadata…")
dirs = sorted(glob.glob(str(BASE / "waveprocessed" / "PROCESSED-*")))
meta_all, _, _, _ = load_analysis_data(*dirs, load_processed=False)

mask_common = (
    (meta_all["PanelCondition"] == "full")
    & (meta_all["in_position"] == PROBE)
    & (meta_all["WaveFrequencyInput [Hz]"] == FREQ)
    & (meta_all["WaveAmplitudeInput [Volt]"] == AMP)
    & meta_all["path"].str.contains("per240", na=False)
    & ~meta_all["path"].str.contains("mstop330", na=False)
    & (meta_all["quality_flag"] == "ok")
)
fullwind = meta_all[mask_common & (meta_all["WindCondition"] == "full")].copy()
nowind   = meta_all[mask_common & (meta_all["WindCondition"] == "no")].copy()

print(f"  {len(fullwind)} fullwind runs, {len(nowind)} nowind runs")
print("  Fullwind paths:")
for _, r in fullwind.iterrows():
    print(f"    {Path(r['path']).parent.name}/{Path(r['path']).name}")

print("Loading processed time series…")
proc_dfs = load_processed_dfs(*dirs)


# ── Plot ──────────────────────────────────────────────────────────────────────
rows_to_plot = list(fullwind.iterrows())[:4]
nwind_row = nowind.iloc[0] if not nowind.empty else None

N_fw = len(rows_to_plot)
N_cols = 3  # eta / sliding AFFT / sliding Atd
fig, axes = plt.subplots(N_fw + 1, N_cols, figsize=(4 * N_cols, 2.2 * (N_fw + 1)),
                         sharex=False)

def plot_one_run(axes_row, row, label_prefix, color="tab:red"):
    path = row["path"]
    df = proc_dfs.get(path)
    if df is None:
        for ax in axes_row:
            ax.text(0.5, 0.5, "missing", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
        return

    col = f"eta_{PROBE}_interp" if f"eta_{PROBE}_interp" in df.columns else f"eta_{PROBE}"
    sig = df[col].to_numpy(dtype=float)
    t = np.arange(len(sig)) / FS

    # Pipeline window
    s_col = f"Computed Probe {PROBE} start"
    e_col = f"Computed Probe {PROBE} end"
    pip_start, pip_end = row.get(s_col), row.get(e_col)
    pip_afft = row.get(f"Probe {PROBE} Amplitude (FFT)", np.nan)
    pip_atd  = row.get(f"Probe {PROBE} Amplitude", np.nan)

    ax_eta, ax_afft, ax_atd = axes_row

    # eta signal
    ax_eta.plot(t, sig, color=color, lw=0.3, alpha=0.7)
    ax_eta.set_ylabel(f"η (mm)\n{label_prefix}", fontsize=8)
    ax_eta.grid(True, alpha=0.3)
    if pd.notna(pip_start):
        ax_eta.axvspan(pip_start / FS, pip_end / FS, color="tab:green", alpha=0.15)

    # sliding AFFT
    t_afft, a_afft = sliding_afft(sig)
    ax_afft.plot(t_afft, a_afft, color=color, lw=1.2, label="sliding AFFT")
    ax_afft.set_ylabel("AFFT (mm)", fontsize=8)
    ax_afft.grid(True, alpha=0.3)
    if pd.notna(pip_start):
        ax_afft.axvspan(pip_start / FS, pip_end / FS, color="tab:green", alpha=0.15)
    if pd.notna(pip_afft):
        ax_afft.axhline(pip_afft, color="tab:red", ls="--", lw=0.9,
                        label=f"pipeline {pip_afft:.1f} mm")
    ax_afft.legend(fontsize=6, loc="lower right")

    # sliding Atd
    t_atd, a_atd = sliding_atd(sig)
    ax_atd.plot(t_atd, a_atd, color=color, lw=1.2, label="sliding Atd")
    ax_atd.set_ylabel("Atd (mm)", fontsize=8)
    ax_atd.grid(True, alpha=0.3)
    if pd.notna(pip_start):
        ax_atd.axvspan(pip_start / FS, pip_end / FS, color="tab:green", alpha=0.15)
    if pd.notna(pip_atd):
        ax_atd.axhline(pip_atd, color="tab:red", ls="--", lw=0.9,
                       label=f"pipeline {pip_atd:.1f} mm")
    ax_atd.legend(fontsize=6, loc="lower right")

for i, (_, fw_row) in enumerate(rows_to_plot):
    folder = Path(fw_row["path"]).parent.name[:20]
    plot_one_run(axes[i], fw_row, f"fullwind ({folder})")

if nwind_row is not None:
    folder = Path(nwind_row["path"]).parent.name[:20]
    plot_one_run(axes[-1], nwind_row, f"nowind ({folder})", color="tab:blue")

for ax in axes[-1]:
    ax.set_xlabel("time (s)", fontsize=8)

fig.suptitle("", fontsize=10)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"PNG → {OUT_PNG.relative_to(BASE)}")

# ── Findings ──────────────────────────────────────────────────────────────────
lines = []
lines.append("# Zoom on 1.5 Hz / 0.2 V fullwind per240 anomaly at 9373/170")
lines.append("")
lines.append(f"Generated: {pd.Timestamp.utcnow().isoformat()[:19]}Z")
lines.append("")
lines.append(f"**Runs analyzed**: {len(fullwind)} fullwind + {len(nowind)} nowind")
lines.append("")
for _, r in fullwind.iterrows():
    lines.append(f"- **fullwind** `{Path(r['path']).parent.name}/{Path(r['path']).name}`")
    lines.append(f"  pipeline_AFFT={r.get(f'Probe {PROBE} Amplitude (FFT)', float('nan')):.2f} mm, "
                 f"pipeline_Atd={r.get(f'Probe {PROBE} Amplitude', float('nan')):.2f} mm")
for _, r in nowind.iterrows():
    lines.append(f"- **nowind** `{Path(r['path']).parent.name}/{Path(r['path']).name}`")
    lines.append(f"  pipeline_AFFT={r.get(f'Probe {PROBE} Amplitude (FFT)', float('nan')):.2f} mm, "
                 f"pipeline_Atd={r.get(f'Probe {PROBE} Amplitude', float('nan')):.2f} mm")
lines.append("")
lines.append("## See the figure")
lines.append("")
lines.append(f"`analysis_scratch/sliding_afft_15hz_02v_zoom.png` — 3 columns per run: ")
lines.append("  1. η raw time series (sampling rate 250 Hz)")
lines.append("  2. sliding AFFT at 1.5 Hz")
lines.append("  3. sliding Atd (time-domain (P97.5−P2.5)/2)")
lines.append("")
lines.append("If the anomaly is **consistent across all 4 fullwind runs** (early burst, later decay),")
lines.append("the wavemaker is producing a real transient at this freq/amp under fullwind.")
lines.append("")
lines.append("If only one run shows it, it is a noise event or one-off data quality issue.")
OUT_MD.write_text("\n".join(lines) + "\n")
print(f"MD  → {OUT_MD.relative_to(BASE)}")
