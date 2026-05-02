"""
Parallel-probe agreement, faceted by frequency  (CH04 §3e sibling)
==================================================================

Per-thesis-frequency variant of `parallel_probe_agreement.py` panel (a):
A(9373/170) vs A(9373/340) with identity + ±5/±10 % bands, one panel per
paddle frequency in the thesis band (1.3, 1.4, 1.5, 1.6 Hz).

Layout: 2×2 grid (one panel per frequency).
  Marker color  = WindCondition  (no = blue, full = red, lowest = green)
  Marker size   = paddle amplitude tier (0.1 / 0.2 / 0.3 V)
  Identity dash + ±5 % (green band) + ±10 % (yellow band)
  Per-panel title shows the frequency.

Scope: full-panel quality-ok wave runs in the canon March-2026 lowrange
folders.

Outputs
-------
    analysis_scratch/parallel_probe_agreement_by_freq.pdf      (scratch)
    output/FIGURES/ch04_parallel_probe_agreement_by_freq.pdf   (thesis)
    output/TEXFIGU/ch04_parallel_probe_agreement_by_freq.tex   (stub)

Caption text is sourced from the central FIGURE_CAPTIONS dict in
main_save_figures.py — edit there, not here.
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import os
os.chdir(BASE)

from wavescripts.improved_data_loader import load_analysis_data
from wavescripts.plot_utils import apply_thesis_style
import wavescripts.plot_utils as pu

apply_thesis_style()

THESIS_NAME  = "ch04_parallel_probe_agreement_by_freq"
SCRATCH_PDF  = Path(__file__).parent / "parallel_probe_agreement_by_freq.pdf"
OUT_PDF      = BASE / "output" / "FIGURES" / f"{THESIS_NAME}.pdf"

RESULTS_DIRS = [
    Path("waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange"),
    Path("waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"),
]

FREQS      = [1.3, 1.4, 1.5, 1.6]
WIND_COLOR = {"no": "#2980B9", "full": "#E74C3C", "lowest": "#27AE60"}
AMP_SIZE   = {0.1: 22, 0.2: 55, 0.3: 100}


def main():
    print("Loading meta …")
    meta, _, _, _ = load_analysis_data(*RESULTS_DIRS, load_processed=False)
    scope = meta[
        meta["WaveFrequencyInput [Hz]"].notna()
        & meta["WaveFrequencyInput [Hz]"].round(2).between(1.3, 1.6)
        & (meta["PanelCondition"] == "full")
        & (meta["quality_flag"] == "ok")
        & meta["Probe 9373/170 Amplitude (FFT)"].notna()
        & meta["Probe 9373/340 Amplitude (FFT)"].notna()
    ].copy()
    scope["freq_r"] = scope["WaveFrequencyInput [Hz]"].round(2)
    print(f"  {len(scope)} runs in scope")

    fig, axes = plt.subplots(2, 2, figsize=(11, 10), sharex=True, sharey=True)

    a170_all = scope["Probe 9373/170 Amplitude (FFT)"].to_numpy()
    a340_all = scope["Probe 9373/340 Amplitude (FFT)"].to_numpy()
    lim = max(float(np.nanmax(a170_all)), float(np.nanmax(a340_all))) * 1.08

    for ax, fh in zip(axes.flat, FREQS):
        sub = scope[np.isclose(scope["freq_r"], fh, atol=0.01)]
        n_panel = len(sub)

        xs = np.linspace(0, lim, 50)
        ax.plot(xs, xs, "k--", lw=0.8, alpha=0.6, zorder=1, label="identity")
        ax.fill_between(xs, xs * 0.95, xs * 1.05, color="#2ECC71", alpha=0.10,
                        zorder=0, label="±5%")
        ax.fill_between(xs, xs * 0.90, xs * 1.10, color="#F1C40F", alpha=0.08,
                        zorder=0, label="±10%")

        for wind, grp in sub.groupby("WindCondition"):
            sizes = grp["WaveAmplitudeInput [Volt]"].map(
                lambda a: AMP_SIZE.get(round(float(a), 1), 30)
            )
            ax.scatter(grp["Probe 9373/170 Amplitude (FFT)"],
                       grp["Probe 9373/340 Amplitude (FFT)"],
                       s=sizes, color=WIND_COLOR.get(wind, "gray"),
                       edgecolor="black", linewidth=0.4,
                       alpha=0.78, label=f"{wind} wind", zorder=3)

        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"$f = {fh:.1f}$ Hz   (n = {n_panel})", fontsize=11)

    for ax in axes[-1, :]:
        ax.set_xlabel("A(9373/170)  [mm, FFT]", fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel("A(9373/340)  [mm, FFT]", fontsize=10)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 5),
               fontsize=9, bbox_to_anchor=(0.5, -0.01))

    fig.tight_layout(rect=(0, 0.04, 1, 1))

    SCRATCH_PDF.parent.mkdir(parents=True, exist_ok=True)
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SCRATCH_PDF, bbox_inches="tight")
    print(f"  Saved -> {SCRATCH_PDF.relative_to(BASE)}")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"  Saved -> {OUT_PDF.relative_to(BASE)}")
    plt.close(fig)

    # --- TEXFIGU stub: caption resolved from central FIGURE_CAPTIONS dict
    _meta = pu.build_fig_meta(
        {
            "filters": {
                "PanelCondition":   "full",
                "quality_flag":     "ok",
                "WaveFrequencyInput [Hz]": "1.3–1.6 Hz",
            },
            "plotting": {"figure_name": THESIS_NAME},
        },
        chapter="04",
        extra={"script": "analysis_scratch/parallel_probe_agreement_by_freq.py"},
        computed_in=("analysis_scratch/parallel_probe_agreement_by_freq.py "
                     "(per-frequency facet of parallel_probe_agreement panel a)"),
        data_class="DELEG",
        findings_doc="memory/methodology_*.md (parallel-probe section)",
        data_df=scope,
        extra_params=(
            "2x2 grid of A(9373/170) vs A(9373/340) FFT amplitudes, one "
            "panel per thesis paddle frequency (1.3, 1.4, 1.5, 1.6 Hz). "
            "Identity dashed line, ±5% (green) and ±10% (yellow) bands. "
            "Marker colour = WindCondition (no=blue, full=red, lowest=green). "
            "Marker size = paddle amplitude tier (0.1 / 0.2 / 0.3 V). "
            "Shared axes across panels for direct visual comparison."
        ),
        extra_stats={
            "n_runs_total": str(len(scope)),
            **{f"n_runs_at_f{int(fh*10):02d}":
                str(int(np.isclose(scope["freq_r"], fh, atol=0.01).sum()))
                for fh in FREQS},
        },
    )
    pu.write_figure_stub(_meta, plot_type="parallel_probe_agreement_by_freq",
                         force=True)
    print(f"  Wrote stub -> output/TEXFIGU/{THESIS_NAME}.tex")


if __name__ == "__main__":
    main()
