"""
Reconstruction check plot — shows:
  Top row:    mstop330-run3 (ABORTED — over-convergence guard fired, signal left untouched)
  Bottom row: freq1600 run2/run3/run4 (RECONSTRUCTED — fault artifacts removed)

Run from repo root:
    conda run -n draumkvedet python analysis_scratch/recon_check.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from wavescripts.improved_data_loader import load_analysis_data

PROCESSED_DIRS = [
    "waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange",
]

print("Loading data...")
combined_meta, processed_dfs, _, _ = load_analysis_data(
    *PROCESSED_DIRS, load_processed=True
)
print(f"  {len(combined_meta)} rows, {len(processed_dfs)} time series loaded")

IN_POS  = "9373/170"
IN_COL  = f"eta_{IN_POS}_interp"

# ── Identify runs ──────────────────────────────────────────────────────────────
def pick(freq_hz, amp_v, keyword):
    mask = (
        (combined_meta["WaveFrequencyInput [Hz]"].round(2) == round(freq_hz, 2)) &
        (combined_meta["WaveAmplitudeInput [Volt]"].round(2) == round(amp_v, 2)) &
        (combined_meta["WindCondition"] == "no") &
        (combined_meta["path"].str.contains(keyword))
    )
    return combined_meta[mask]

mstop330_rows = combined_meta[
    combined_meta["path"].str.contains("mstop330-run3") &
    (combined_meta["WaveFrequencyInput [Hz]"].round(2) == 1.30)
]
run2_rows = combined_meta[
    combined_meta["path"].str.contains("freq1600") &
    combined_meta["path"].str.contains("amp0300") &
    combined_meta["path"].str.contains("run2") &
    (combined_meta["WindCondition"] == "no")
]
run3_rows = combined_meta[
    combined_meta["path"].str.contains("freq1600") &
    combined_meta["path"].str.contains("amp0300") &
    combined_meta["path"].str.contains("run3") &
    (combined_meta["WindCondition"] == "no")
]
run4_rows = combined_meta[
    combined_meta["path"].str.contains("freq1600") &
    combined_meta["path"].str.contains("amp0300") &
    combined_meta["path"].str.contains("run4") &
    (combined_meta["WindCondition"] == "no")
]

print(f"mstop330 rows: {len(mstop330_rows)}")
print(f"run2 rows:     {len(run2_rows)}")
print(f"run3 rows:     {len(run3_rows)}")
print(f"run4 rows:     {len(run4_rows)}")

FS = 250.0


def get_ts(rows, pos):
    """Return (t, eta_interp, good_start, good_end) for the first matching row."""
    if rows.empty:
        return None
    row  = rows.iloc[0]
    path = row["path"]
    df   = processed_dfs.get(path)
    if df is None:
        return None
    col = f"eta_{pos}_interp"
    if col not in df.columns:
        return None
    eta = df[col].values
    t   = np.arange(len(eta)) / FS
    gs  = row.get(f"Computed Probe {pos} start", np.nan)
    ge  = row.get(f"Computed Probe {pos} end",   np.nan)
    return t, eta, gs, ge, Path(path).name


# ── Build figure ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
fig.subplots_adjust(hspace=0.45, wspace=0.25)

GRAY   = "#888888"
BLUE   = "#3F51B5"
RED    = "#D62728"
GREEN  = "#2ca02c"

def plot_panel(ax, rows, pos, title, color, zoom_s=None):
    res = get_ts(rows, pos)
    if res is None:
        ax.set_visible(False)
        return
    t, eta, gs, ge, fname = res
    ax.plot(t, eta, color=color, lw=0.7, alpha=0.85)
    if not (np.isnan(gs) or np.isnan(ge)):
        ax.axvspan(gs / FS, ge / FS, color="gold", alpha=0.18, label="analysis window")
    if zoom_s is not None:
        ax.set_xlim(zoom_s)
    ax.axhline(0, color="k", lw=0.4, ls="--")
    ax.set_title(title, fontsize=9, pad=3)
    ax.set_xlabel("Time [s]", fontsize=8)
    ax.set_ylabel("η [mm]", fontsize=8)
    ax.tick_params(labelsize=7)
    # annotate signal_confidence
    sc_col = f"signal_confidence_{pos}"
    sc = rows.iloc[0].get(sc_col, "?") if not rows.empty else "?"
    color_sc = {"high": GREEN, "reconstructed": BLUE, "reconstruction_failed": RED}.get(sc, GRAY)
    ax.text(0.99, 0.97, f"signal_confidence: {sc}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=7.5, color=color_sc,
            bbox=dict(fc="white", ec=color_sc, alpha=0.8, pad=2))

# ── Top row: mstop330-run3 — full signal + zoom ────────────────────────────────
res330 = get_ts(mstop330_rows, IN_POS)
if res330:
    t330, eta330, gs330, ge330, fname330 = res330
    axes[0, 0].plot(t330, eta330, color=GRAY, lw=0.6)
    if not (np.isnan(gs330) or np.isnan(ge330)):
        axes[0, 0].axvspan(gs330 / FS, ge330 / FS, color="gold", alpha=0.18)
    axes[0, 0].axhline(0, color="k", lw=0.4, ls="--")
    axes[0, 0].set_title("mstop330-run3  •  1.3 Hz nowind 0.1V\n(ABORTED — over-convergence guard)", fontsize=8.5, pad=3)
    axes[0, 0].set_xlabel("Time [s]", fontsize=8)
    axes[0, 0].set_ylabel("η [mm]", fontsize=8)
    axes[0, 0].tick_params(labelsize=7)
    sc = mstop330_rows.iloc[0].get(f"signal_confidence_{IN_POS}", "?")
    axes[0, 0].text(0.99, 0.97, f"signal_confidence: {sc}",
                    transform=axes[0, 0].transAxes, ha="right", va="top",
                    fontsize=7.5, color=RED,
                    bbox=dict(fc="white", ec=RED, alpha=0.8, pad=2))

    # zoom into middle of analysis window
    mid = (gs330 + ge330) / 2 / FS
    zoom_lo, zoom_hi = mid - 5, mid + 5
    axes[0, 1].plot(t330, eta330, color=GRAY, lw=0.8)
    axes[0, 1].axvspan(gs330 / FS, ge330 / FS, color="gold", alpha=0.18)
    axes[0, 1].axhline(0, color="k", lw=0.4, ls="--")
    axes[0, 1].set_xlim(zoom_lo, zoom_hi)
    axes[0, 1].set_title("mstop330-run3  •  zoom (10 s window)\nSignal is clean — no artifacts to fix", fontsize=8.5, pad=3)
    axes[0, 1].set_xlabel("Time [s]", fontsize=8)
    axes[0, 1].set_ylabel("η [mm]", fontsize=8)
    axes[0, 1].tick_params(labelsize=7)

# Top right: mstop330 σ_res annotation
ax_txt = axes[0, 2]
ax_txt.axis("off")
sc_col  = f"signal_confidence_{IN_POS}"
nflag   = mstop330_rows.iloc[0].get(f"recon_n_flagged_{IN_POS}", "?") if not mstop330_rows.empty else "?"
nclean  = mstop330_rows.iloc[0].get(f"recon_n_clean_{IN_POS}",   "?") if not mstop330_rows.empty else "?"
sres    = mstop330_rows.iloc[0].get(f"recon_sigma_{IN_POS}",     "?") if not mstop330_rows.empty else "?"
samp    = mstop330_rows.iloc[0].get(f"recon_amp_sigma_{IN_POS}", "?") if not mstop330_rows.empty else "?"
win_len = (ge330 - gs330) if res330 and not (np.isnan(gs330) or np.isnan(ge330)) else np.nan
frac    = float(nflag) / float(win_len) * 100 if (isinstance(nflag, (int, float)) and not np.isnan(win_len)) else "?"
txt = (
    "RECON ABORTED\n"
    f"probe:  {IN_POS}\n\n"
    f"n_newly_flagged: {nflag}\n"
    f"window length:   {int(win_len) if not np.isnan(win_len) else '?'} samples\n"
    f"fraction flagged: {frac:.1f}%\n\n"
    f"σ_res  = {float(sres):.4f} mm  ← sub-noise!\n"
    f"σ_amp  = {float(samp):.4f} mm\n\n"
    "Threshold: 15%\n"
    "→ Signal left untouched\n"
    "→ signal_confidence = 'reconstruction_failed'"
)
ax_txt.text(0.05, 0.95, txt, transform=ax_txt.transAxes,
            va="top", ha="left", fontsize=8.5, family="monospace",
            bbox=dict(fc="#fff3f3", ec=RED, alpha=0.9, pad=8))
ax_txt.set_title("Diagnostic summary", fontsize=9, pad=3)

# ── Bottom row: 1.6 Hz run2 / run3 / run4 ─────────────────────────────────────
for ax, rows, label in zip(axes[1], [run2_rows, run3_rows, run4_rows],
                            ["run2", "run3", "run4"]):
    res = get_ts(rows, IN_POS)
    if res is None:
        ax.set_visible(False)
        continue
    t, eta, gs, ge, fname = res
    ax.plot(t, eta, color=BLUE, lw=0.8)
    if not (np.isnan(gs) or np.isnan(ge)):
        ax.axvspan(gs / FS, ge / FS, color="gold", alpha=0.18, label="analysis window")
    ax.axhline(0, color="k", lw=0.4, ls="--")
    # annotate
    sc   = rows.iloc[0].get(f"signal_confidence_{IN_POS}", "?")
    nf   = rows.iloc[0].get(f"recon_n_flagged_{IN_POS}", "?")
    sr   = rows.iloc[0].get(f"recon_sigma_{IN_POS}", "?")
    sa   = rows.iloc[0].get(f"recon_amp_sigma_{IN_POS}", "?")
    color_sc = {"high": GREEN, "reconstructed": BLUE}.get(sc, GRAY)
    ax.set_title(f"1.6 Hz nowind 0.3V  •  {label}\n"
                 f"n_flagged={nf}  σ_res={float(sr):.2f} mm  σ_amp={float(sa):.2f} mm",
                 fontsize=8.5, pad=3)
    ax.set_xlabel("Time [s]", fontsize=8)
    ax.set_ylabel("η [mm]", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.text(0.99, 0.97, f"signal_confidence: {sc}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=7.5, color=color_sc,
            bbox=dict(fc="white", ec=color_sc, alpha=0.8, pad=2))

# ── Overall title ─────────────────────────────────────────────────────────────
fig.suptitle(
    "Reconstruction safety cap check  —  20260327 folder\n"
    "Top: mstop330 long clean run (ABORTED — over-convergence)  |  "
    "Bottom: 1.6 Hz fault runs (RECONSTRUCTED correctly)",
    fontsize=10, y=1.01
)

out = Path("analysis_scratch/recon_check.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
