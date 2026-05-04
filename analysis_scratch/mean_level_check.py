"""
Quick mean-level diagnostic for the two inspirational-timeseries runs.

Same 2 CSV runs that feed ch04_inspirational_{nowind,fullwind}.pdf:
  - fullpanel-nowind-amp0200-freq1400-per240-depth580-mstop30-run1.csv
  - fullpanel-fullwind-amp0200-freq1400-per240-depth580-mstop30-run1.csv

For each of the 4 (run × probe) combinations (IN=9373/170, OUT=12400/250),
plot the η(t) time series with TWO horizontal reference lines:

  1. y = 0  → the "Stillwater Probe {pos}" zero, computed by
              processor.ensure_stillwater_columns from the first 2 s of
              the same run (per-run pre-wave anchor). The raw mm value
              is annotated as text so the reader can see the actual
              raw probe distance the pipeline used as the zero-ref.
  2. y = median(η) inside the H&G analysis window — the "offset median"
              the wave train actually sits at relative to the stillwater
              anchor. Non-zero values flag (a) wind setup, (b) Stokes-2
              broad-trough lift of the median, or (c) anchor drift.

Quick sanity-check, not a thesis figure. Outputs:
    output/FIGURES/ch04_mean_level_check.pdf  (+ .png)
    stdout: numerical summary table per (run, probe)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from wavescripts.improved_data_loader import load_analysis_data, load_processed_dfs
from wavescripts.plot_utils import apply_thesis_style, WIND_COLOR_MAP

FS = 250.0
BASE = Path("/Users/ole/Kodevik/wave_project")
FIGURES_DIR = BASE / "output" / "FIGURES"

TARGET_DIR = BASE / "waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"
DATADIR    = BASE / "wavedata/20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"

RUNS = {
    "nowind":   DATADIR / "fullpanel-nowind-amp0200-freq1400-per240-depth580-mstop30-run1.csv",
    "fullwind": DATADIR / "fullpanel-fullwind-amp0200-freq1400-per240-depth580-mstop30-run1.csv",
}
PROBES = [("IN", "9373/170"), ("OUT", "12400/250")]
WIND_KEY  = {"nowind": "no", "fullwind": "full"}
WIND_TITLE = {"nowind": "uten vind", "fullwind": "full vind"}
WIN_COLOR = "#F1B24A"  # amber — H&G window vspan, matches inspirational figure


apply_thesis_style()

print("Loading …")
meta, _, _, _ = load_analysis_data(str(TARGET_DIR), load_processed=False)
proc          = load_processed_dfs(str(TARGET_DIR))


def _load(wind_tag: str, probe: str) -> dict:
    """Pull η(t), the H&G window indices, and the stillwater anchor for one
    (run, probe). Mirrors the load logic in inspirational_timeseries.py
    so the plotted η values are bit-identical to the thesis figure."""
    csv = str(RUNS[wind_tag])
    row = meta[meta["path"] == csv].iloc[0]
    df  = proc[csv]

    # Pipeline writes both eta_{pos} and eta_{pos}_interp. _interp is the
    # gap-filled version used by all downstream FFT/LS code; prefer it.
    col = f"eta_{probe}_interp" if f"eta_{probe}_interp" in df.columns \
          else f"eta_{probe}"
    eta = df[col].to_numpy(dtype=float)
    t   = np.arange(len(eta)) / FS

    # H&G analysis window — start UC-snapped within ±T of the theoretical
    # arrival-anchored window opening, end UC-snapped to the 10th detected
    # upcrossing past start. These two columns are the canonical "where
    # the pipeline actually measured" indices used by FFT/LS amplitude.
    s = int(row[f"Computed Probe {probe} start"])
    e = int(row[f"Computed Probe {probe} end"])

    # Stillwater raw mm value — the zero reference subtracted from raw to
    # form η.  Computed by processor.ensure_stillwater_columns:
    #   - nowave runs: full-run median of the raw probe distance
    #   - wave runs (this script): mean of the first STILLWATER.PRE_WAVE_S
    #     = 2.0 s of raw signal, BEFORE the wave front arrives.
    # By construction η = -(raw - stillwater), so y=0 in the plots IS
    # this stillwater value.  Annotated on each panel for transparency.
    sw_raw = float(row[f"Stillwater Probe {probe}"])

    win = eta[s:e]
    win = win[~np.isnan(win)]
    return {
        "row":      row,
        "t":        t,
        "eta":      eta,
        "s":        s,
        "e":        e,
        "sw_raw":   sw_raw,
        "win_med":  float(np.median(win)) if len(win) else np.nan,
        "win_mean": float(np.mean(win))   if len(win) else np.nan,
        "pre_mean": float(np.nanmean(eta[:int(2.0 * FS)])),  # should be ≈0
    }


# ── Layout: 2 rows (nowind, fullwind) × 2 cols (IN, OUT) ────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True, sharey=True)

print(f"\n{'(run, probe)':25s}  {'sw_raw':>8s}  {'pre_mean η':>10s}  {'win mean':>9s}  {'win med':>9s}")
print("-" * 70)

for r_idx, wind_tag in enumerate(("nowind", "fullwind")):
    for c_idx, (label, probe) in enumerate(PROBES):
        ax = axes[r_idx, c_idx]
        d  = _load(wind_tag, probe)

        signal_color = WIND_COLOR_MAP[WIND_KEY[wind_tag]]

        # η(t) — the same series the thesis figure plots.
        ax.plot(d["t"], d["eta"], color=signal_color, lw=0.45,
                label=r"$\eta(t)$")

        # H&G window highlight.
        ax.axvspan(d["s"]/FS, d["e"]/FS,
                   color=WIN_COLOR, alpha=0.30, zorder=0,
                   label="H&G vindu")

        # Reference 1 — y = 0 IS the stillwater anchor in η-space. Drawn
        # solid black so it reads as the pipeline's zero. The raw mm value
        # is annotated to show what the pipeline subtracted to get here.
        ax.axhline(0, color="black", lw=0.8, alpha=0.7,
                   label=f"Stillwater = 0  (raw {d['sw_raw']:.2f} mm)")

        # Reference 2 — median of η inside the H&G window. If the anchor is
        # correct AND the wave is symmetric, this should sit near 0.
        # Stokes-2 (broad-trough) pushes it slightly negative; wind setup
        # at IN/OUT would push it negative/positive respectively.
        ax.axhline(d["win_med"], color="#D946EF", lw=1.2, ls="--", alpha=0.95,
                   label=f"window-median η = {d['win_med']:+.3f} mm")

        ax.set_title(f"{wind_tag} ({WIND_TITLE[wind_tag]}) · {label} probe {probe}",
                     fontsize=10)
        ax.grid(True, which="major", alpha=0.30)
        ax.grid(True, which="minor", alpha=0.10)
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.xaxis.set_minor_locator(MultipleLocator(2))
        ax.yaxis.set_major_locator(MultipleLocator(5))
        ax.yaxis.set_minor_locator(MultipleLocator(1))
        if c_idx == 0:
            ax.set_ylabel(r"$\eta$ [mm]")
        if r_idx == 1:
            ax.set_xlabel("Tid [s]")
        ax.legend(loc="lower right", fontsize=7, framealpha=0.92)

        print(f"({wind_tag:8s}, {probe:9s})  "
              f"{d['sw_raw']:8.3f}  {d['pre_mean']:+10.4f}  "
              f"{d['win_mean']:+9.4f}  {d['win_med']:+9.4f}")

fig.suptitle("Mean-level check — η(t) with stillwater anchor (y=0) and H&G-window median",
             fontsize=11, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.985))

OUT = FIGURES_DIR / "ch04_mean_level_check.pdf"
fig.savefig(OUT, bbox_inches="tight")
fig.savefig(OUT.with_suffix(".png"), dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"\n→ {OUT.relative_to(BASE)}  (+ .png)")
print("Done.")
