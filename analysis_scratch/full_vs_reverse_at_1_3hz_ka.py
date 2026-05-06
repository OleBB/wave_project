"""
Mooring og panelretning ved 1.30 Hz — K_t vs ka
=================================================

Originally framed as "full vs reverse panel" (script and filename still
carry that name for traceability). After running it the dominant visual
story turned out to be the **mooring effect** (canon = below_90 vs
above_50): the canon/above_50 K_t gap is +0.043 nowind / +0.121 fullwind,
while the panel-direction gap on above_50 is < 0.01 either wind. Panel
direction is a *secondary* observation visible only on above_50 — canon
mooring was never run with revers panelretning.

Read this plot as: mooring is primary, panelretning is the within-mooring
sub-comparison.

Same data as `full_vs_reverse_at_1_3hz.py`, restyled with ka on x. ka
clusters the points into 3 amp groups along x; a connecting line through
per-amp means added more clutter than it resolved (4 whole + 2 dashed
lines, all clamped to the same 3 ka values). Removed.

Panel orientation moves into a parallel marker family. The thesis-wide
"normal" panel keeps the standard ○ □ △ amp shapes; "revers panelretning"
gets a distinct star / X family that doesn't echo the standard set.

Naming note: the dataset's `PanelCondition` is `full` / `reverse`; the
reader-facing labels are "normal" (since the panel sits in its physically
designed orientation) and "revers panelretning" (panel rotated). Internal
filter keeps the data-column names.

Encoding:
  x      : ka (per-run, k(1.30 Hz) × IN Amplitude (FFT) [mm] / 1000)
  y      : K_t
  marker : amp tier × panel
             normal panel  → ○ A1, □ A2, △ A3
             revers panel  → ✡ A1, ★ A2, ✕ A3
  hue    : wind (red med, blue uten)
  hue    : mooring × wind (below_90 → blue/red; above_50 → cyan/bright pink)
           — palette synced with `mooring_focus_at_1_3hz_ka.py` so the
           reader can compare the combined and per-amp variants directly
  fill   : hardware (filled = canon cond4_h100_low, hollow = earlier)

Outputs:
    analysis_scratch/full_vs_reverse_at_1_3hz_ka.pdf
    output/FIGURES/ch05_full_vs_reverse_at_1_3hz_ka.pdf
    analysis_scratch/full_vs_reverse_at_1_3hz_ka_summary.csv
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
    WIND_COLOR_MAP, amp_to_label, apply_thesis_style, freq_to_k,
)

# Per-(panel, amp) markers. Reader sees panel via shape family
# (round/square/triangle vs star/star/X), not via linestyle.
# (6, 1, 0) = matplotlib "6-pointed star" — Star of David approximation.
PANEL_AMP_MARKER = {
    ("full",    0.10): "o",
    ("full",    0.20): "s",
    ("full",    0.30): "^",
    ("reverse", 0.10): (6, 1, 0),
    ("reverse", 0.20): "*",
    ("reverse", 0.30): "X",
}
PANEL_LABEL = {"full": "normal", "reverse": "revers panelretning"}

apply_thesis_style()

SCRATCH_PDF = Path(__file__).parent / "full_vs_reverse_at_1_3hz_ka.pdf"
SCRATCH_CSV = Path(__file__).parent / "full_vs_reverse_at_1_3hz_ka_summary.csv"
OUT_PDF     = BASE / "output" / "FIGURES" / "ch05_full_vs_reverse_at_1_3hz_ka.pdf"
OUT_PDF.parent.mkdir(parents=True, exist_ok=True)

TARGET_FREQ = 1.30

# ── 1. Load & filter ───────────────────────────────────────────────────────────
print("1. Loading processed folders …")
all_dirs = sorted(glob.glob(str(BASE / "waveprocessed" / "PROCESSED-*")))
meta, _, _, _ = load_analysis_data(*all_dirs, load_processed=False)

def assign_condition(row):
    h = row.get("probe_height_mm", PROBE_HEIGHT_DEFAULT_MM)
    r = row.get("probe_range_mode", "high")
    if pd.isna(h):
        h = PROBE_HEIGHT_DEFAULT_MM
    h = int(h)
    return "cond4_h100_low" if (h == 100 and r == "low") else "earlier"

meta["condition"] = meta.apply(assign_condition, axis=1)
meta["is_final"] = meta["condition"] == "cond4_h100_low"

def mooring_group(m):
    if m in ("below_90_loose230", "below_90_loose300"):
        return "below_90"
    if m == "above_50":
        return "above_50"
    return "other"
meta["moor_grp"] = meta["Mooring"].apply(mooring_group)

sel = meta[
    (meta["WaveFrequencyInput [Hz]"] == TARGET_FREQ)
    & meta["PanelCondition"].isin(["full", "reverse"])
    & (meta["quality_flag"] == "ok")
    & meta["OUT/IN (FFT)"].notna()
    & meta["IN Amplitude (FFT)"].notna()
    & meta["WindCondition"].isin(["no", "full"])
    & (meta["OUT/IN (FFT)"].between(0.1, 2.0))
    & meta["moor_grp"].isin(["below_90", "above_50"])
].copy()

# Per-run ka.
k_const = float(freq_to_k(np.array([TARGET_FREQ]))[0])
sel["a_m"] = sel["IN Amplitude (FFT)"].astype(float) / 1000.0   # mm → m
sel["ka"]  = k_const * sel["a_m"]
sel["amp_v"] = sel["WaveAmplitudeInput [Volt]"].apply(lambda v: round(float(v), 2))
print(f"   {len(sel)} rows; k({TARGET_FREQ} Hz) = {k_const:.3f} rad/m")
print(f"   ka range: [{sel['ka'].min():.3f}, {sel['ka'].max():.3f}]")

# ── 2. Visual constants ────────────────────────────────────────────────────────
WIND_LABEL  = {"no": "uten vind", "full": "med vind"}
MARKER_SIZE = 70
ALPHA_FILLED   = 0.70
ALPHA_HOLLOW   = 0.85
EDGE_LW_FILLED = 0.3
EDGE_LW_HOLLOW = 1.4

COLOR = {
    ("below_90", "no"):   WIND_COLOR_MAP["no"],     # #1F77B4 — blue
    ("below_90", "full"): WIND_COLOR_MAP["full"],   # #D62728 — red
    ("above_50", "no"):   "#00CED1",                # cyan (DarkTurquoise)
    ("above_50", "full"): "#FF1493",                # bright pink (DeepPink)
}
MOORING_LABEL = {"below_90": "below_90 (canon)", "above_50": "above_50"}

# ── 3. Plot ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.5, 5.4))

# Scatter all individual runs. Marker = (panel, amp); colour = (mooring, wind);
# fill = hardware. No connecting lines / mean overlays — the ka grouping
# already places points where the eye expects them.
mean_rows = []
for (panel, moor, wind, amp_v, is_final), grp in sel.groupby(
    ["PanelCondition", "moor_grp", "WindCondition", "amp_v", "is_final"]
):
    if grp.empty:
        continue
    color = COLOR[(moor, wind)]
    marker = PANEL_AMP_MARKER.get((panel, amp_v), "X")
    if is_final:
        fc, ec = color, "black"
        lw, a = EDGE_LW_FILLED, ALPHA_FILLED
    else:
        fc, ec = "none", color
        lw, a = EDGE_LW_HOLLOW, ALPHA_HOLLOW
    ax.scatter(
        grp["ka"], grp["OUT/IN (FFT)"],
        facecolors=fc, edgecolors=ec, marker=marker,
        s=MARKER_SIZE, linewidths=lw, alpha=a,
        zorder=3 if is_final else 2,
    )
    # Bookkeeping (mean stats kept for the CSV, not plotted).
    mean_rows.append(dict(
        panel=panel, moor_grp=moor, wind=wind, amp_v=amp_v,
        is_final=bool(is_final), n=int(len(grp)),
        ka_mean=float(grp["ka"].mean()),
        Kt_mean=float(grp["OUT/IN (FFT)"].mean()),
        Kt_std=float(grp["OUT/IN (FFT)"].std()) if len(grp) > 1 else None,
    ))

ax.axhline(1.0, color="black", lw=0.6, ls="--", alpha=0.5)
ax.set_xlabel(r"$ka$  (per kjøring; $k(1.30\,\mathrm{Hz})\cdot a_\mathrm{IN}$)",
               fontsize=10)
ax.set_ylabel(r"$K_t$", fontsize=12, rotation=0, ha="right", va="center")
ax.set_title(
    f"Mooring og panelretning ved {TARGET_FREQ} Hz — $K_t$ vs $ka$\n"
    "(mooring dominerer; panelretning er sekundær, kun målbar på above_50)",
    fontsize=10.5,
)
ax.grid(which="major", alpha=0.30, lw=0.6)
ax.grid(which="minor", alpha=0.15, lw=0.4)
ax.yaxis.set_major_locator(MultipleLocator(0.05))
ax.yaxis.set_minor_locator(MultipleLocator(0.025))

# Y-range fitted with small pad.
y_lo = sel["OUT/IN (FFT)"].min() - 0.02
y_hi = sel["OUT/IN (FFT)"].max() + 0.02
ax.set_ylim(y_lo, y_hi)

# Legend stack — split the visual dimensions into clear mini-legends.
moor_wind_handles = [
    mlines.Line2D([], [], color=COLOR[("below_90", "no")], lw=4,
                  label=f"{MOORING_LABEL['below_90']} · uten vind"),
    mlines.Line2D([], [], color=COLOR[("below_90", "full")], lw=4,
                  label=f"{MOORING_LABEL['below_90']} · med vind"),
    mlines.Line2D([], [], color=COLOR[("above_50", "no")], lw=4,
                  label=f"{MOORING_LABEL['above_50']} · uten vind"),
    mlines.Line2D([], [], color=COLOR[("above_50", "full")], lw=4,
                  label=f"{MOORING_LABEL['above_50']} · med vind"),
]
# Amp × panel — 6 marker entries, grouped by panel for legibility.
amp_panel_handles = []
for panel in ("full", "reverse"):
    for v in (0.10, 0.20, 0.30):
        amp_panel_handles.append(
            mlines.Line2D([], [], color="black",
                          marker=PANEL_AMP_MARKER[(panel, v)],
                          linestyle="None", markersize=8,
                          markerfacecolor="lightgray", markeredgecolor="black",
                          markeredgewidth=0.4,
                          label=f"{PANEL_LABEL[panel]} · {amp_to_label(v)}")
        )
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

leg1 = ax.legend(handles=moor_wind_handles, loc="upper left",
                  fontsize=8, framealpha=0.92,
                  title="Mooring · vind", title_fontsize=8,
                  bbox_to_anchor=(0.005, 0.995))
ax.add_artist(leg1)
leg2 = ax.legend(handles=amp_panel_handles, loc="lower left",
                  fontsize=7.5, framealpha=0.92,
                  title="Panel · amplitude", title_fontsize=8,
                  ncol=2, bbox_to_anchor=(0.005, 0.005))
ax.add_artist(leg2)
ax.legend(handles=hw_handles, loc="lower right",
           fontsize=8, framealpha=0.92,
           title="Oppsett (fyll)", title_fontsize=8,
           bbox_to_anchor=(0.995, 0.005))

fig.tight_layout()

SCRATCH_PDF.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(SCRATCH_PDF, bbox_inches="tight", pad_inches=0.02)
print(f"\n   Saved → {SCRATCH_PDF.relative_to(BASE)}")
fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.02)
print(f"   Saved → {OUT_PDF.relative_to(BASE)}")
plt.close(fig)

mean_df = pd.DataFrame(mean_rows)
mean_df.to_csv(SCRATCH_CSV, index=False)
print(f"   Summary → {SCRATCH_CSV.relative_to(BASE)}\n")
print(mean_df.round(4).to_string(index=False))

print("\nDone.")
