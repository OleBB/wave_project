"""
Full vs reverse panel — head-to-head K_t comparison
====================================================

Reverse-panel runs only exist at two frequencies in the full record:
0.65 Hz (19 runs) and 1.30 Hz (23 runs). This script plots K_t at those
two frequencies for both panels side-by-side so the panel-geometry effect
can be read directly.

Layout: 1 row × 2 cols (freq = 0.65 Hz | 1.30 Hz).
  x-axis: panel condition (full | reverse), small horizontal jitter
  y-axis: K_t (= OUT/IN (FFT))
  colour: wind (no = blue, full = red)
  marker: amplitude tier (○ A1, □ A2, △ A3)
  fill:   canon (filled) vs earlier hardware (hollow)
  overlay: per (panel, wind) mean ± std

Outputs:
    analysis_scratch/full_vs_reverse_comparison.pdf
    output/FIGURES/ch05_full_vs_reverse_comparison.pdf
    analysis_scratch/full_vs_reverse_comparison_summary.csv

Run:
    conda run -n draumkvedet python analysis_scratch/full_vs_reverse_comparison.py
"""

import sys
import warnings
from pathlib import Path
import glob

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
from wavescripts.constants import PROBE_HEIGHT_DEFAULT_MM
from wavescripts.plot_utils import (
    WIND_COLOR_MAP, amp_to_label, apply_thesis_style,
)

apply_thesis_style()

SCRATCH_PDF = Path(__file__).parent / "full_vs_reverse_comparison.pdf"
SCRATCH_CSV = Path(__file__).parent / "full_vs_reverse_comparison_summary.csv"
OUT_PDF     = BASE / "output" / "FIGURES" / "ch05_full_vs_reverse_comparison.pdf"
OUT_PDF.parent.mkdir(parents=True, exist_ok=True)

# ── 1. Load & filter ───────────────────────────────────────────────────────────
print("1. Loading processed folders …")
all_dirs = sorted(glob.glob(str(BASE / "waveprocessed" / "PROCESSED-*")))
meta, _, _, _ = load_analysis_data(*all_dirs, load_processed=False)
print(f"   {len(meta)} total rows")


def assign_condition(row):
    in_pos = row.get("in_position", None)
    if in_pos == "9373/250":
        return "legacy_nov2025"
    h = row.get("probe_height_mm", PROBE_HEIGHT_DEFAULT_MM)
    r = row.get("probe_range_mode", "high")
    if pd.isna(h):
        h = PROBE_HEIGHT_DEFAULT_MM
    h = int(h)
    if h == 100 and r == "low":
        return "cond4_h100_low"
    return "earlier"


meta["condition"] = meta.apply(assign_condition, axis=1)
meta["is_final"] = meta["condition"] == "cond4_h100_low"

sel = meta[
    meta["WaveFrequencyInput [Hz]"].notna()
    & (meta["WaveFrequencyInput [Hz]"] > 0)
    & meta["PanelCondition"].isin(["full", "reverse"])
    & (meta["quality_flag"] == "ok")
    & meta["OUT/IN (FFT)"].notna()
    & meta["WindCondition"].isin(["no", "full"])
    & (meta["OUT/IN (FFT)"].between(0.1, 2.0))
].copy()

# Frequencies where reverse panel exists.
rev_freqs = sorted(sel.loc[sel["PanelCondition"] == "reverse",
                            "WaveFrequencyInput [Hz]"].unique())
print(f"   reverse-panel freqs: {rev_freqs}")
sel = sel[sel["WaveFrequencyInput [Hz]"].isin(rev_freqs)].copy()
print(f"   {len(sel)} rows at those freqs (full + reverse pooled)")

# ── 2. Plot ────────────────────────────────────────────────────────────────────
WIND_LABEL  = {"no": "uten vind", "full": "med vind"}
PANEL_X     = {"full": 0, "reverse": 1}
PANEL_LABEL = {"full": "full", "reverse": "revers"}
AMP_MARKER  = {0.10: "o", 0.20: "s", 0.30: "^"}
MARKER_SIZE = 60
JITTER_HALF_WIDTH = 0.10  # x-jitter so points within the same panel/wind aren't on top of each other
ALPHA_FILLED   = 0.65
ALPHA_HOLLOW   = 0.85
EDGE_LW_FILLED = 0.3
EDGE_LW_HOLLOW = 1.4

# Wind colour gets a small x-offset so the no/full clouds don't overlap inside one panel column.
WIND_X_OFFSET = {"no": -0.18, "full": +0.18}


def _round_amp(v):
    return round(float(v), 2)


def _jitter(n, half=JITTER_HALF_WIDTH, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-half, half, size=n)


fig, axes = plt.subplots(
    1, len(rev_freqs),
    figsize=(3.6 * len(rev_freqs), 5.6),
    sharey=True,
)
if len(rev_freqs) == 1:
    axes = [axes]

summary_rows = []

for ax, f in zip(axes, rev_freqs):
    sub = sel[sel["WaveFrequencyInput [Hz]"] == f].copy()

    # Scatter individual runs with horizontal jitter.
    seed = 0
    for panel, x_base in PANEL_X.items():
        sp = sub[sub["PanelCondition"] == panel]
        if sp.empty:
            continue
        for wind in ["no", "full"]:
            sw = sp[sp["WindCondition"] == wind]
            if sw.empty:
                continue
            color = WIND_COLOR_MAP[wind]
            for amp_v, marker in AMP_MARKER.items():
                sa = sw[sw["WaveAmplitudeInput [Volt]"]
                        .apply(_round_amp) == amp_v]
                if sa.empty:
                    continue
                # Filled vs hollow per is_final.
                for is_final, sub_h in sa.groupby("is_final"):
                    n = len(sub_h)
                    if n == 0:
                        continue
                    seed += 1
                    x = (x_base + WIND_X_OFFSET[wind]
                         + _jitter(n, seed=seed))
                    if is_final:
                        fc, ec = color, "black"
                        lw, a = EDGE_LW_FILLED, ALPHA_FILLED
                    else:
                        fc, ec = "none", color
                        lw, a = EDGE_LW_HOLLOW, ALPHA_HOLLOW
                    ax.scatter(
                        x, sub_h["OUT/IN (FFT)"].values,
                        facecolors=fc, edgecolors=ec, marker=marker,
                        s=MARKER_SIZE, linewidths=lw, alpha=a,
                        zorder=3 if is_final else 2,
                    )

    # Overlay mean ± std per (panel, wind) — pooled across amp & hardware.
    for panel, x_base in PANEL_X.items():
        sp = sub[sub["PanelCondition"] == panel]
        if sp.empty:
            continue
        for wind in ["no", "full"]:
            sw = sp[sp["WindCondition"] == wind]
            n = len(sw)
            if n == 0:
                continue
            mean = float(sw["OUT/IN (FFT)"].mean())
            std  = float(sw["OUT/IN (FFT)"].std()) if n > 1 else float("nan")
            color = WIND_COLOR_MAP[wind]
            x_pos = x_base + WIND_X_OFFSET[wind]
            # Mean line — short horizontal segment.
            ax.plot([x_pos - 0.13, x_pos + 0.13], [mean, mean],
                    color=color, lw=2.0, zorder=4, solid_capstyle="round")
            if n > 1:
                ax.errorbar([x_pos], [mean], yerr=[std],
                            fmt="none", ecolor=color, elinewidth=1.4,
                            capsize=4, capthick=1.2, zorder=4, alpha=0.85)
            summary_rows.append(dict(
                freq_hz=f, panel=panel, wind=wind,
                n=n, mean_kt=mean, std_kt=std,
                n_canon=int(sw["is_final"].sum()),
                n_earlier=int((~sw["is_final"]).sum()),
            ))

    ax.set_title(f"f = {f} Hz", fontsize=11)
    ax.set_xticks(list(PANEL_X.values()))
    ax.set_xticklabels([PANEL_LABEL[p] for p in PANEL_X.keys()])
    ax.set_xlim(-0.6, 1.6)
    ax.axhline(1.0, color="black", lw=0.6, ls="--", alpha=0.5)
    ax.grid(which="major", axis="y", alpha=0.30, lw=0.6)
    ax.grid(which="minor", axis="y", alpha=0.15, lw=0.4)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(MultipleLocator(0.025))

# Y-axis label, only on left-most panel.
axes[0].set_ylabel(r"$K_t$", fontsize=12, rotation=0, ha="right", va="center")
for ax in axes:
    ax.set_xlabel("Panelgeometri", fontsize=10)

# Tight y-range based on actual data.
y_lo = sel["OUT/IN (FFT)"].min() - 0.02
y_hi = sel["OUT/IN (FFT)"].max() + 0.02
for ax in axes:
    ax.set_ylim(y_lo, y_hi)

# Legend (single, on rightmost panel).
wind_handles = [
    mlines.Line2D([], [], color=WIND_COLOR_MAP[w], lw=4,
                  label=WIND_LABEL[w])
    for w in ["no", "full"]
]
amp_handles = [
    mlines.Line2D([], [], color="black",
                  marker=AMP_MARKER[v], linestyle="None", markersize=8,
                  markerfacecolor="lightgray", markeredgecolor="black",
                  markeredgewidth=0.3, label=amp_to_label(v))
    for v in (0.10, 0.20, 0.30)
]
hw_handles = [
    mlines.Line2D([], [], color="black",
                  marker="o", linestyle="None", markersize=8,
                  markerfacecolor="black", markeredgecolor="black",
                  markeredgewidth=0.3, label="endelig (cond4)"),
    mlines.Line2D([], [], color="black",
                  marker="o", linestyle="None", markersize=8,
                  markerfacecolor="none", markeredgecolor="black",
                  markeredgewidth=1.4, label="tidligere"),
]
mean_handle = [
    mlines.Line2D([], [], color="gray", lw=2.0,
                  label="middel (m. errorbar = std)"),
]

leg1 = axes[-1].legend(handles=wind_handles + mean_handle, loc="lower right",
                        fontsize=8, framealpha=0.92,
                        title="Vind", title_fontsize=8,
                        bbox_to_anchor=(0.99, 0.02))
axes[-1].add_artist(leg1)
leg2 = axes[-1].legend(handles=amp_handles + hw_handles, loc="upper right",
                        fontsize=7.5, framealpha=0.92,
                        title="Amp / oppsett", title_fontsize=8,
                        bbox_to_anchor=(0.99, 0.99), ncol=1)

fig.tight_layout()

# ── 3. Save ────────────────────────────────────────────────────────────────────
SCRATCH_PDF.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(SCRATCH_PDF, bbox_inches="tight", pad_inches=0.02)
print(f"\n   Saved → {SCRATCH_PDF.relative_to(BASE)}")
fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.02)
print(f"   Saved → {OUT_PDF.relative_to(BASE)}")
plt.close(fig)

# ── 4. Summary CSV + console table ─────────────────────────────────────────────
summary = pd.DataFrame(summary_rows)
summary.to_csv(SCRATCH_CSV, index=False)
print(f"   Summary → {SCRATCH_CSV.relative_to(BASE)}\n")
print(summary.to_string(index=False))

# Quick delta print: full − reverse per (freq, wind).
print("\n   Δ = mean(K_t)_reverse − mean(K_t)_full per (freq, wind):")
piv_mean = (summary.pivot_table(index=["freq_hz", "wind"],
                                 columns="panel", values="mean_kt")
                    .reset_index())
if "full" in piv_mean.columns and "reverse" in piv_mean.columns:
    piv_mean["delta_rev_minus_full"] = piv_mean["reverse"] - piv_mean["full"]
    print(piv_mean.to_string(index=False))

print("\nDone.")
