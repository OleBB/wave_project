"""Experiment: ch05_damping_scatter_full with truly even k-spacing.

Standalone — does NOT touch wavescripts/plotter.py or main_save_figures.py.
Produces output/FIGURES/ch05_damping_scatter_full_evenk.pdf alongside the
existing _full.pdf so they can be compared side by side.

The current plot (wavescripts/plotter.py::_make_damping_scatter_fig) plots
at actual k values 6.8, 7.9, 9.1, 10.3 — Δk = 1.1, 1.2, 1.2. Visually almost
even but not exactly. This variant plots at index positions 1, 2, 3, 4 with
k labels on the bottom axis, so the four columns are visibly equidistant.
"""
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

BASE = (Path(__file__).resolve().parent.parent
        if "__file__" in globals() else Path.cwd())
os.chdir(BASE)
sys.path.insert(0, str(BASE))

from wavescripts.improved_data_loader import load_analysis_data
from wavescripts.filters import (apply_experimental_filters,
                                 damping_all_amplitude_grouper)
from wavescripts.plot_utils import (apply_thesis_style, WIND_COLOR_MAP,
                                    apply_horizontal_ylabel,
                                    amp_to_label, freq_to_k)
from wavescripts.constants import GlobalColumns as GC

# ── 1. Load same data the live cell uses ─────────────────────────────────
# Live setup (main_save_figures.py:763–779): meta_results is the canon
# 2-folder subset of combined_meta, with the loose230/loose300 mooring
# variants merged into "below_90_loose".
print("Loading canon-only meta_results …")
RESULTS_PROCESSED_DIRS = [
    BASE / "waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange",
    BASE / "waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange",
]
meta_results, _, _, _ = load_analysis_data(*RESULTS_PROCESSED_DIRS,
                                           load_processed=False)
meta_results["Mooring"] = meta_results["Mooring"].replace({
    "below_90_loose230": "below_90_loose",
    "below_90_loose300": "below_90_loose",
})
print(f"  {len(meta_results)} rows total")

# Same filter as main_save_figures.py § 2 (_pv_damping_scatter)
_pv = {
    "filters": {
        "WaveAmplitudeInput [Volt]": None,
        "WaveFrequencyInput [Hz]":   (1.3, 1.6),
        "WindCondition":             None,
        "PanelCondition":            None,
    },
    "plotting": {},
}
filt = apply_experimental_filters(meta_results, _pv)
stats_df = damping_all_amplitude_grouper(filt)
print(f"  {len(stats_df)} grouped rows after filter+grouper")

# ── 2. Plot — full panel only, even-k variant ────────────────────────────
apply_thesis_style()

PANEL = "full"
KEY_FREQS = np.array([1.3, 1.4, 1.5, 1.6])
KEY_KS    = freq_to_k(KEY_FREQS)
AMP_MARKER = {0.10: "o", 0.20: "s", 0.30: "^"}

# Map freq → integer index 1..4 for even visual spacing
_freq_to_idx = {f: i + 1 for i, f in enumerate(KEY_FREQS)}

subset = stats_df[stats_df[GC.PANEL_CONDITION] == PANEL].copy()
subset["k_idx"] = subset[GC.WAVE_FREQUENCY_INPUT].round(2).map(_freq_to_idx)

fig, ax = plt.subplots(figsize=(5, 7))

for (wind, amp), grp in subset.groupby([GC.WIND_CONDITION,
                                        GC.WAVE_AMPLITUDE_INPUT]):
    marker = AMP_MARKER.get(round(float(amp), 2), "o")
    color  = WIND_COLOR_MAP.get(wind, "gray")
    grp = grp.sort_values("k_idx")
    ax.scatter(
        grp["k_idx"], grp["mean_out_in"],
        color=color, marker=marker, s=70, alpha=0.85,
        edgecolors="black", linewidths=0.3, zorder=3,
    )

ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.4)
ax.set_ylim(0.33, 0.93)
ax.grid(True, alpha=0.3)

# Bottom axis: even index positions, labelled with k values.
ax.set_xticks([1, 2, 3, 4])
ax.set_xticklabels([f"{k:.1f}" for k in KEY_KS])
ax.set_xlim(0.6, 4.4)
ax.set_xlabel("$k$", fontsize=11)

# Top axis: same even positions, labelled with frequencies.
secax = ax.twiny()
secax.set_xlim(ax.get_xlim())
secax.set_xticks([1, 2, 3, 4])
secax.set_xticklabels([f"{f:.1f}" for f in KEY_FREQS])
secax.set_xlabel("Frekvens [Hz]", fontsize=9)

# ── Legend (matches the live figure exactly) ─────────────────────────────
wind_handles = [
    mlines.Line2D([], [], color=WIND_COLOR_MAP[w], lw=4, label=label)
    for w, label in (("no", "uten vind"), ("full", "med vind"))
]
amp_handles = [
    mlines.Line2D([], [], color="black", marker=AMP_MARKER[v], lw=0,
                  markersize=8, markerfacecolor="lightgray",
                  markeredgecolor="black", markeredgewidth=0.3,
                  label=amp_to_label(v))
    for v in (0.10, 0.20, 0.30)
]
leg_w = ax.legend(handles=wind_handles, title="Vind",
                  loc="upper right", bbox_to_anchor=(0.99, 0.99),
                  fontsize=7, title_fontsize=7, framealpha=0.92)
ax.add_artist(leg_w)
ax.legend(handles=amp_handles, title="Amplitude",
          loc="upper right", bbox_to_anchor=(0.99, 0.83),
          fontsize=7, title_fontsize=7, framealpha=0.92)

fig.subplots_adjust(left=0.07, right=0.97, top=0.80, bottom=0.13)
apply_horizontal_ylabel(ax, r"$K_t$", fontsize=12)

# ── 3. Save next to the live figure for visual comparison ────────────────
OUT_PDF = BASE / "output" / "FIGURES" / "ch05_damping_scatter_full_evenk.pdf"
fig.savefig(OUT_PDF, bbox_inches="tight")
print(f"  → {OUT_PDF.relative_to(BASE)}")
plt.close(fig)
