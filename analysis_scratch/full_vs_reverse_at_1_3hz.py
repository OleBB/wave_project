"""
Full vs reverse at 1.30 Hz — mooring split by colour shade
============================================================

Iteration on `full_vs_reverse_comparison_above50.py`. Only 1.30 Hz (the
freq with cleanest reverse-vs-full head-to-head). All moorings included
so the mooring confound is visible *as part of the figure*, not hidden
in the filter.

Encoding:
  canon mooring (below_90_loose230 + below_90_loose300) → standard
    WIND_COLOR_MAP — blue (uten vind) / red (med vind)
  above_50 mooring                                       → light blue / pink

Reverse panel only exists on above_50, so its column shows only the
light/pink shades. Full panel column shows all four shades. The vertical
gap between canon (dark) and above_50 (light) markers within each
(panel, wind) cell is exactly the mooring effect from Thread 4.

Outputs:
    analysis_scratch/full_vs_reverse_at_1_3hz.pdf
    output/FIGURES/ch05_full_vs_reverse_at_1_3hz.pdf
    analysis_scratch/full_vs_reverse_at_1_3hz_summary.csv
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

SCRATCH_PDF = Path(__file__).parent / "full_vs_reverse_at_1_3hz.pdf"
SCRATCH_CSV = Path(__file__).parent / "full_vs_reverse_at_1_3hz_summary.csv"
OUT_PDF     = BASE / "output" / "FIGURES" / "ch05_full_vs_reverse_at_1_3hz.pdf"
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
    if h == 100 and r == "low":
        return "cond4_h100_low"
    return "earlier"

meta["condition"] = meta.apply(assign_condition, axis=1)
meta["is_final"] = meta["condition"] == "cond4_h100_low"

# Mooring grouping: canon = below_90_*; above = above_50.
def mooring_group(m):
    if m in ("below_90_loose230", "below_90_loose300"):
        return "canon"
    if m == "above_50":
        return "above_50"
    return "other"

meta["moor_grp"] = meta["Mooring"].apply(mooring_group)

sel = meta[
    (meta["WaveFrequencyInput [Hz]"] == TARGET_FREQ)
    & meta["PanelCondition"].isin(["full", "reverse"])
    & (meta["quality_flag"] == "ok")
    & meta["OUT/IN (FFT)"].notna()
    & meta["WindCondition"].isin(["no", "full"])
    & (meta["OUT/IN (FFT)"].between(0.1, 2.0))
    & meta["moor_grp"].isin(["canon", "above_50"])
].copy()
print(f"   {len(sel)} rows at {TARGET_FREQ} Hz, panel ∈ {{full, reverse}}, "
      f"moor_grp ∈ {{canon, above_50}}")

# Counts per (panel, moor_grp, wind) for sanity.
print("\n   counts per (panel, moor_grp, wind):")
print(sel.groupby(["PanelCondition", "moor_grp", "WindCondition"])
         .size().unstack(fill_value=0).to_string())

# ── 2. Visual constants ────────────────────────────────────────────────────────
WIND_LABEL  = {"no": "uten vind", "full": "med vind"}
PANEL_X     = {"full": 0, "reverse": 1}
PANEL_LABEL = {"full": "full", "reverse": "revers"}
AMP_MARKER  = {0.10: "o", 0.20: "s", 0.30: "^"}
MARKER_SIZE = 65
JITTER_HALF_WIDTH = 0.07

# Color scheme: canon = full-saturation thesis WIND_COLOR_MAP;
# above_50 = washed-out shades of the same hues so the reader sees
# "different mooring, same wind" via lightness.
COLOR = {
    ("canon",    "no"):   WIND_COLOR_MAP["no"],     # #1F77B4 — blue
    ("canon",    "full"): WIND_COLOR_MAP["full"],   # #D62728 — red
    ("above_50", "no"):   "#9EC9E2",                # light blue
    ("above_50", "full"): "#F4A6A6",                # pink
}

# Group offsets along x: canon left of centre, above_50 right of centre.
# Combined with wind offsets within each mooring half so all 4 (mooring, wind)
# clouds stay visually separated within each panel column.
MOOR_X_OFFSET = {"canon": -0.20, "above_50": +0.20}
WIND_X_OFFSET = {"no":    -0.08, "full":     +0.08}

ALPHA_FILLED   = 0.70
ALPHA_HOLLOW   = 0.85
EDGE_LW_FILLED = 0.3
EDGE_LW_HOLLOW = 1.4
MEAN_BAR_HALF_W = 0.06

def _round_amp(v): return round(float(v), 2)
def _jitter(n, half=JITTER_HALF_WIDTH, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-half, half, size=n)

# ── 3. Plot ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 5.4))

summary_rows = []
seed = 0

for panel, x_base in PANEL_X.items():
    sp = sel[sel["PanelCondition"] == panel]
    if sp.empty:
        continue
    for moor in ["canon", "above_50"]:
        sm = sp[sp["moor_grp"] == moor]
        if sm.empty:
            continue
        for wind in ["no", "full"]:
            sw = sm[sm["WindCondition"] == wind]
            if sw.empty:
                continue
            color = COLOR[(moor, wind)]
            x_pos = (x_base + MOOR_X_OFFSET[moor] + WIND_X_OFFSET[wind])
            for amp_v, marker in AMP_MARKER.items():
                sa = sw[sw["WaveAmplitudeInput [Volt]"]
                        .apply(_round_amp) == amp_v]
                if sa.empty:
                    continue
                for is_final, sub_h in sa.groupby("is_final"):
                    n = len(sub_h)
                    if n == 0:
                        continue
                    seed += 1
                    x = x_pos + _jitter(n, seed=seed)
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

            # Mean ± std overlay per (panel, moor, wind), pooled across amp.
            n = len(sw)
            mean = float(sw["OUT/IN (FFT)"].mean())
            std  = float(sw["OUT/IN (FFT)"].std()) if n > 1 else float("nan")
            ax.plot([x_pos - MEAN_BAR_HALF_W, x_pos + MEAN_BAR_HALF_W],
                     [mean, mean], color=color, lw=2.2, zorder=4,
                     solid_capstyle="round")
            if n > 1:
                ax.errorbar([x_pos], [mean], yerr=[std],
                            fmt="none", ecolor=color, elinewidth=1.6,
                            capsize=4, capthick=1.2, zorder=4, alpha=0.9)
            summary_rows.append(dict(
                panel=panel, moor_grp=moor, wind=wind,
                n=n, mean_kt=mean, std_kt=std,
            ))

ax.set_xticks(list(PANEL_X.values()))
ax.set_xticklabels([PANEL_LABEL[p] for p in PANEL_X.keys()], fontsize=11)
ax.set_xlim(-0.7, 1.7)
ax.set_xlabel("Panelgeometri", fontsize=11)
ax.set_ylabel(r"$K_t$", fontsize=12, rotation=0, ha="right", va="center")
ax.set_title(f"Full vs reverse ved {TARGET_FREQ} Hz "
              "— canon vs above_50 mooring",
              fontsize=11)

# Y-range fitted with small pad.
y_lo = sel["OUT/IN (FFT)"].min() - 0.02
y_hi = sel["OUT/IN (FFT)"].max() + 0.02
ax.set_ylim(y_lo, y_hi)

ax.axhline(1.0, color="black", lw=0.6, ls="--", alpha=0.5)
ax.grid(which="major", axis="y", alpha=0.30, lw=0.6)
ax.grid(which="minor", axis="y", alpha=0.15, lw=0.4)
ax.yaxis.set_major_locator(MultipleLocator(0.05))
ax.yaxis.set_minor_locator(MultipleLocator(0.025))

# Legend — split into mooring×wind colour key, amp-shape key, hardware-fill
# key. Compact stacking on the right.
moor_wind_handles = [
    mlines.Line2D([], [], color=COLOR[("canon", "no")], lw=4,
                  label="canon · uten vind"),
    mlines.Line2D([], [], color=COLOR[("canon", "full")], lw=4,
                  label="canon · med vind"),
    mlines.Line2D([], [], color=COLOR[("above_50", "no")], lw=4,
                  label="above_50 · uten vind"),
    mlines.Line2D([], [], color=COLOR[("above_50", "full")], lw=4,
                  label="above_50 · med vind"),
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

leg1 = ax.legend(handles=moor_wind_handles, loc="lower left",
                  fontsize=8, framealpha=0.92,
                  title="Mooring · vind", title_fontsize=8,
                  bbox_to_anchor=(0.01, 0.01))
ax.add_artist(leg1)
leg2 = ax.legend(handles=amp_handles + hw_handles, loc="upper right",
                  fontsize=7.5, framealpha=0.92,
                  title="Amp / oppsett", title_fontsize=8,
                  bbox_to_anchor=(0.99, 0.99))

fig.tight_layout()

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

# Cross-tab: mean K_t by (panel, moor) × wind, plus Δ_wind per cell.
print("\n   Cross-tab mean K_t — rows = (panel, moor), cols = wind:")
piv = summary.pivot_table(index=["panel", "moor_grp"], columns="wind",
                            values="mean_kt")
piv["Δ_wind"] = piv.get("full", np.nan) - piv.get("no", np.nan)
print(piv.round(4).to_string())

# Also Δ_mooring = canon − above_50 per (panel, wind) — quantifies the
# mooring confound that motivated this iteration.
print("\n   Δ_mooring = canon mean − above_50 mean per (panel, wind):")
for panel in ["full", "reverse"]:
    for wind in ["no", "full"]:
        try:
            c = summary[(summary["panel"]==panel) &
                        (summary["moor_grp"]=="canon") &
                        (summary["wind"]==wind)]["mean_kt"].iloc[0]
            a = summary[(summary["panel"]==panel) &
                        (summary["moor_grp"]=="above_50") &
                        (summary["wind"]==wind)]["mean_kt"].iloc[0]
            print(f"     {panel:7} / {wind:4}: canon = {c:.4f}, "
                  f"above_50 = {a:.4f}, Δ = {c-a:+.4f}")
        except IndexError:
            print(f"     {panel:7} / {wind:4}: incomplete (one side missing)")

print("\nDone.")
