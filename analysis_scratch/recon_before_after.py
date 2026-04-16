"""
Before / after reconstruction — pipeline version.

Shows the TRUE pipeline-cleaned raw signal (Probe {pos}) alongside the
reconstructed signal (eta_{pos}_interp) that now comes out of the pipeline.

Layout: 3 rows (run2 / run3 / run4), 2 columns (IN probe / OUT probe).
Each panel: gray = raw (false troughs visible), blue = reconstructed (clean),
            red dots = samples that were changed (raw → recon differs).

Run from repo root:
    conda run -n draumkvedet python analysis_scratch/recon_before_after.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from wavescripts.improved_data_loader import load_analysis_data, load_processed_dfs

BASE     = Path(__file__).parent.parent
FS       = 250.0
IN_POS   = "9373/170"
OUT_POS  = "12400/250"
FREQ     = 1.6
PROC_DIR = (
    BASE / "waveprocessed"
    / "PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"
)

RUNS = [
    "fullpanel-nowind-amp0300-freq1600-per40-depth580-mstop30-run2.csv",
    "fullpanel-nowind-amp0300-freq1600-per40-depth580-mstop30-run3.csv",
    "fullpanel-nowind-amp0300-freq1600-per40-depth580-mstop30-run4.csv",
]

print("Loading cache...")
combined_meta, _, _, _ = load_analysis_data(PROC_DIR, load_processed=False)
processed_dfs = load_processed_dfs(PROC_DIR)
print(f"  {len(combined_meta)} runs, {len(processed_dfs)} time series")


def fft_amp(sig, fs=FS, f0=FREQ, window_hz=0.05):
    n = len(sig)
    s = np.nan_to_num(sig - np.nanmean(sig))
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    spec  = np.abs(np.fft.rfft(s))
    mask  = np.abs(freqs - f0) <= window_hz
    return spec[mask][np.argmin(np.abs(freqs[mask] - f0))] * 2 / n if mask.any() else np.nan


PROBES = [IN_POS, "9373/340", OUT_POS]
PROBE_LABELS = ["IN  (9373/170)", "IN parallel  (9373/340)", "OUT  (12400/250)"]

fig, axes = plt.subplots(3, 3, figsize=(22, 12))
fig.suptitle(
    "Pipeline reconstruction — before vs after  (1.6 Hz nowind 0.3V, analysis window only)\n"
    "Gray: zeroed signal before reconstruction  (eta_{pos}, false troughs intact as real values)\n"
    "Blue: post-reconstruction  (eta_{pos}_interp, derivative-guided fill applied)\n"
    "Red dots: changed samples  (|before − after| > 0.5 mm)",
    fontsize=10, y=1.01,
)

for row_i, fname in enumerate(RUNS):
    key = next((k for k in processed_dfs if fname in k), None)
    if key is None:
        print(f"  WARNING: {fname} not found in cache")
        continue

    df  = processed_dfs[key]
    row = combined_meta[combined_meta["path"] == key].iloc[0]

    for col_i, pos in enumerate(PROBES):
        ax = axes[row_i, col_i]

        # eta_{pos}       = zeroed signal, NaN where pipeline clipped,
        #                   false troughs still present as genuine values
        # eta_{pos}_interp = zeroed + PCHIP + reconstruction applied in-pipeline
        before_col = f"eta_{pos}"
        recon_col  = f"eta_{pos}_interp"

        if before_col not in df.columns or recon_col not in df.columns:
            ax.set_visible(False)
            continue

        gs = int(row.get(f"Computed Probe {pos} start", 0))
        ge = int(row.get(f"Computed Probe {pos} end",   len(df)))

        # Slice to analysis window
        raw_win   = df[before_col].values[gs:ge]
        recon_win = df[recon_col].values[gs:ge]
        t         = np.arange(gs, ge) / FS

        # Where reconstruction changed the signal materially (>0.5 mm difference)
        # NaN in raw_win means the pipeline already zeroed it — recon filled those too
        changed = np.abs(recon_win - raw_win) > 0.5

        # FFT amplitudes
        fft_raw   = fft_amp(raw_win)
        fft_recon = fft_amp(recon_win)
        n_changed = int(changed.sum())

        # ── Plot ──────────────────────────────────────────────────────────────
        ax.plot(t, raw_win,   color="#AAAAAA", lw=0.8, zorder=1, label="raw (pipeline cleaned)")
        ax.plot(t, recon_win, color="#1f77b4", lw=0.9, zorder=2, label="reconstructed")

        # Highlight changed samples as red dots
        if changed.any():
            ax.scatter(t[changed], recon_win[changed],
                       color="red", s=3, zorder=3, label=f"changed ({n_changed} samples)")

        # Zero line
        ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.4)

        # Title with FFT info
        label = fname.replace("fullpanel-nowind-amp0300-freq1600-per40-depth580-mstop30-", "")
        run_label = f"{label}  •  {PROBE_LABELS[col_i]}"
        fft_delta = fft_recon - fft_raw
        ax.set_title(
            f"{run_label}\n"
            f"FFT amplitude: {fft_raw:.2f} mm  →  {fft_recon:.2f} mm  (Δ={fft_delta:+.2f} mm)   "
            f"{n_changed} samples changed",
            fontsize=8, pad=4,
        )
        ax.set_xlabel("Time [s]", fontsize=8)
        ax.set_ylabel("η [mm]", fontsize=8)
        ax.tick_params(labelsize=7)

        # y-axis: show both the false troughs (raw) and clean signal (recon)
        all_vals = np.concatenate([raw_win[~np.isnan(raw_win)], recon_win])
        ylo = min(np.nanmin(all_vals) * 1.08, -5)
        yhi = max(np.nanmax(all_vals) * 1.08,  5)
        ax.set_ylim(ylo, yhi)

        if row_i == 0:
            ax.legend(fontsize=7, loc="lower right", markerscale=3)

for ax, lbl in zip(axes[0], PROBE_LABELS):
    ax.set_title(ax.get_title())
for ax in axes[:, 0]:
    ax.set_ylabel("η [mm]  (IN 9373/170)", fontsize=8)
for ax in axes[:, 1]:
    ax.set_ylabel("η [mm]  (IN 9373/340)", fontsize=8)
for ax in axes[:, 2]:
    ax.set_ylabel("η [mm]  (OUT 12400/250)", fontsize=8)

fig.tight_layout()
out = BASE / "analysis_scratch" / "recon_before_after.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
