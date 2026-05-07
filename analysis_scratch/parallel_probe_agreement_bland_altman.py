"""
Parallel-probe agreement (Bland-Altman style) — single-figure variant
=====================================================================

Companion to `parallel_probe_agreement_by_freq.py` (the 2×2 facet).
Same data, same scope, same n=80 runs — replotted as a single
Bland-Altman-style agreement panel:

  x : mean amplitude   A_mean = (A_nær + A_fjern) / 2   [mm]
  y : signed disagreement   (A_fjern - A_nær) / A_mean  [%]
  colour : WindCondition  (no = blue, full = red)
  marker : amplitude tier × paddle frequency, via
           wavescripts.plotter._freq_marker (same shape system as
           ch05_damping_ka):
             A1 (0.10 V, circles):    ○ 1.3 / 3/4 wedge 1.4 / right-half 1.5 / upper-quarter 1.6
             A2 (0.20 V, rectangles): 0° / 45° / 90° / 135° rotated rectangles
             A3 (0.30 V, triangles):  ^ / < / v / > pointing
  bands  : 0 (perfect agreement), ±5%, ±10%

Where the 2×2 facet shows "scatter around identity" intuition per
frequency, this panel shows the agreement metric directly on the y-axis,
making the wind / amplitude trend across all frequencies legible at
once.

Probe naming reminder: 9373/170 is the **wall-side** probe (170 mm from
the centre line, closer to the tank wall) — labelled `A_nær` here as
"near" the wall. 9373/340 is the **far-side** probe — labelled `A_fjern`.

Outputs:
    analysis_scratch/parallel_probe_agreement_bland_altman.pdf       (scratch)
    output/FIGURES/ch04_parallel_probe_agreement_bland_altman.pdf    (thesis)
    output/TEXFIGU/ch04_parallel_probe_agreement_bland_altman.tex    (stub)
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
import matplotlib.lines as mlines
from matplotlib.ticker import MultipleLocator

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import os
os.chdir(BASE)

from wavescripts.improved_data_loader import load_analysis_data
from wavescripts.plot_utils import apply_thesis_style, apply_horizontal_ylabel, amp_to_label
from wavescripts.plotter import _freq_marker
import wavescripts.plot_utils as pu

apply_thesis_style()

THESIS_NAME = "ch04_parallel_probe_agreement_bland_altman"
SCRATCH_PDF = Path(__file__).parent / "parallel_probe_agreement_bland_altman.pdf"
OUT_PDF     = BASE / "output" / "FIGURES" / f"{THESIS_NAME}.pdf"
OUT_STUB    = BASE / "output" / "TEXFIGU" / f"{THESIS_NAME}.tex"

# Same 2-folder canon scope as the facet sibling.
RESULTS_DIRS = [
    Path("waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange"),
    Path("waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"),
]

FREQS       = [1.3, 1.4, 1.5, 1.6]
FREQ_IDX    = {f: i for i, f in enumerate(FREQS)}   # 0..3 → _freq_marker
AMPS        = [0.10, 0.20, 0.30]
WIND_COLOR  = {"no": "#2980B9", "full": "#E74C3C", "lowest": "#27AE60"}
WIND_LABEL  = {"no": "uten vind", "full": "med vind", "lowest": "laveste vind"}


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

    # 9373/170 = wall-side ("nær"), 9373/340 = far-side ("fjern").
    a_naer  = scope["Probe 9373/170 Amplitude (FFT)"].astype(float)
    a_fjern = scope["Probe 9373/340 Amplitude (FFT)"].astype(float)
    scope["a_mean"]      = (a_naer + a_fjern) / 2.0
    scope["disagree_pp"] = (a_fjern - a_naer) / scope["a_mean"] * 100.0   # signed %
    scope["amp_v"]       = scope["WaveAmplitudeInput [Volt]"].apply(
        lambda v: round(float(v), 2))
    print(f"  {len(scope)} runs in scope")
    print(f"  disagreement: median {scope['disagree_pp'].median():+.2f} %, "
          f"|max| {scope['disagree_pp'].abs().max():.2f} %, "
          f"std {scope['disagree_pp'].std():.2f} %")

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8.5, 5.4))

    # Reference bands first (behind data).
    x_lo, x_hi = 5, 27
    for pct, color, label in [(5, "#2ECC71", "±5 %"), (10, "#F1C40F", "±10 %")]:
        ax.fill_between([x_lo, x_hi], -pct, pct, color=color, alpha=0.10,
                         zorder=0, label=label)
    ax.axhline(0.0, color="black", lw=0.8, ls="--", alpha=0.6,
               zorder=1, label="perfekt enighet")

    # Scatter all 80 points: colour = wind, marker = (amp tier × freq) via
    # _freq_marker (same convention as ch05_damping_ka). Outline shape carries
    # amplitude (○ A1, □ A2, △ A3); orientation/fill carries frequency.
    for (wind, amp_v, fh), grp in scope.groupby(["WindCondition", "amp_v", "freq_r"]):
        if grp.empty or fh not in FREQ_IDX:
            continue
        ax.scatter(
            grp["a_mean"], grp["disagree_pp"],
            s=70, color=WIND_COLOR.get(wind, "gray"),
            marker=_freq_marker(amp_v, FREQ_IDX[fh]),
            edgecolor="black", linewidth=0.4,
            alpha=0.82, zorder=3,
        )

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(-30, 30)
    ax.set_xlabel(r"$\bar{A} = (A_\mathrm{nær} + A_\mathrm{fjern}) / 2$  [mm, FFT]",
                   fontsize=10)
    apply_horizontal_ylabel(ax,
                             r"$(A_\mathrm{fjern} - A_\mathrm{nær}) / \bar{A}$  [%]",
                             fontsize=11)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.grid(which="major", alpha=0.30, lw=0.6)
    ax.grid(which="minor", alpha=0.15, lw=0.4)

    # Vind (colour) → upper-right corner.
    wind_handles = [
        mlines.Line2D([], [], color=WIND_COLOR[w], lw=0,
                      marker="o", markersize=8,
                      markeredgecolor="black", markeredgewidth=0.4,
                      label=WIND_LABEL[w])
        for w in ("no", "full")
        if w in scope["WindCondition"].unique()
    ]
    ax.legend(handles=wind_handles, loc="upper right",
               fontsize=9, framealpha=0.92,
               title="Vind", title_fontsize=9,
               bbox_to_anchor=(0.998, 0.998))

    # Marker matrix inset — explicit 3×4 grid of every (amp × freq) marker.
    # Replaces the prior two-legend "Frekvens (one amp exemplar) + Amplitude
    # (one freq exemplar)" pair so the reader can read the actual marker for
    # any (amp, freq) cell directly. Placed at the top-centre of the plot
    # area where the data is sparse (no points above y ≈ 14 %).
    ax_legend = ax.inset_axes([0.27, 0.74, 0.46, 0.22])
    ax_legend.set_facecolor("white")
    for i, amp_v in enumerate(AMPS):
        for j, fh in enumerate(FREQS):
            ax_legend.scatter(
                j, len(AMPS) - 1 - i,
                marker=_freq_marker(amp_v, FREQ_IDX[fh]),
                color="gray", s=80,
                edgecolor="black", linewidth=0.4,
            )
    ax_legend.set_xlim(-0.5, len(FREQS) - 0.5)
    ax_legend.set_ylim(-0.7, len(AMPS) - 0.3)
    ax_legend.set_xticks(range(len(FREQS)))
    ax_legend.set_xticklabels([f"{f:.1f} Hz" for f in FREQS], fontsize=8)
    ax_legend.set_yticks(range(len(AMPS)))
    ax_legend.set_yticklabels(
        [amp_to_label(v) for v in reversed(AMPS)], fontsize=9
    )
    ax_legend.tick_params(length=0, pad=2)
    for spine in ax_legend.spines.values():
        spine.set_edgecolor("#999")
        spine.set_linewidth(0.6)
    ax_legend.set_title("Amplitude · Frekvens", fontsize=9, pad=4)
    ax_legend.set_axisbelow(True)

    fig.tight_layout()

    SCRATCH_PDF.parent.mkdir(parents=True, exist_ok=True)
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    OUT_STUB.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SCRATCH_PDF, bbox_inches="tight", pad_inches=0.02)
    print(f"  Saved -> {SCRATCH_PDF.relative_to(BASE)}")
    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.02)
    print(f"  Saved -> {OUT_PDF.relative_to(BASE)}")
    plt.close(fig)

    # ── TEXFIGU stub ──────────────────────────────────────────────────────
    n_per_freq = {int(fh * 10): int(np.isclose(scope["freq_r"], fh, atol=0.01).sum())
                  for fh in FREQS}
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
        extra={"script": "analysis_scratch/parallel_probe_agreement_bland_altman.py"},
        computed_in=("analysis_scratch/parallel_probe_agreement_bland_altman.py "
                     "(single-panel Bland-Altman variant of the 2x2 facet)"),
        data_class="DELEG",
        findings_doc="memory/methodology_*.md (parallel-probe section)",
        data_df=scope,
        extra_params=(
            "Single-panel Bland-Altman-style agreement plot pooling all 4 "
            "thesis frequencies (1.3, 1.4, 1.5, 1.6 Hz). "
            "x = (A_nær + A_fjern) / 2 [mm, FFT] where A_nær = "
            "Probe 9373/170 (wall-side) and A_fjern = Probe 9373/340 "
            "(far-side). y = (A_fjern - A_nær) / mean [%]. "
            "Same data scope as ch04_parallel_probe_agreement_by_freq "
            "(2 canon lowrange folders, panel=full, quality_flag=ok). "
            "Colour = WindCondition (no=blue, full=red). "
            "Marker = (amp tier × paddle freq) via "
            "wavescripts.plotter._freq_marker — outline shape is amp "
            "(○ A1, □ A2, △ A3), orientation/fill is freq (1.3 → 1.6 Hz). "
            "Reference bands: 0 % (perfect agreement, dashed), ±5 % "
            "(green fill), ±10 % (yellow fill)."
        ),
        extra_stats={
            "n_runs_total":    str(len(scope)),
            "median_signed":   f"{scope['disagree_pp'].median():+.3f} %",
            "max_abs_signed":  f"{scope['disagree_pp'].abs().max():.3f} %",
            "std_signed":      f"{scope['disagree_pp'].std():.3f} %",
            **{f"n_runs_at_f{k:02d}": str(v) for k, v in n_per_freq.items()},
        },
    )
    pu.write_figure_stub(_meta, plot_type="parallel_probe_agreement_bland_altman",
                         force=True)
    print(f"  Wrote stub -> output/TEXFIGU/{THESIS_NAME}.tex")


if __name__ == "__main__":
    main()
