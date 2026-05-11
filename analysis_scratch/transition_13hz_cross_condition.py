"""
1.3 Hz cross-condition figure (scratch) — amplitude × mooring × wind
=====================================================================

The 1.3 Hz frequency cell is the densest in the dataset (108 runs across
4 moorings, both winds, all three amps). This figure asks: how does
wind effect ΔK_t depend on amplitude, separately for each mooring?

Encoding:
  - x-axis: WaveAmplitudeInput (A1=0.1 V, A2=0.2 V, A3=0.3 V) with small
            jitter inside each amp slot so overlapping runs are visible.
  - y-axis: K_t (with FFT→LS override).
  - Colour: (mooring × wind), 8 vivid shades matching the 4-way mooring
            scatter (canon_loose300 stock red/blue, canon_loose230 light
            red/blue, canon_above_loose pink/turquoise, nov_above_stiff
            yellow/purple).
  - Marker: amplitude tier (○ A1, □ A2, △ A3).
  - Solid line: per-mooring per-wind median across the 3 amps, drawn so
            the amplitude-dependence is easy to read.

K_t source: FFT canonical with three-way LS+PSD override (same rule as
all sibling scripts).

Run from repo root:
    conda run -n draumkvedet python analysis_scratch/transition_13hz_cross_condition.py
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

from wavescripts.improved_data_loader import load_analysis_data, get_configuration_for_date
from wavescripts.plot_utils import (
    amp_to_label, apply_thesis_style, apply_horizontal_ylabel,
    WIND_COLOR_MAP,
)

apply_thesis_style()

OVERRIDE_THRESHOLD = 0.05
OUT_PDF = Path(__file__).parent / "transition_13hz_cross_condition.pdf"
OUT_CSV = Path(__file__).parent / "transition_13hz_cross_condition_summary.csv"

# ── 1. Load & filter to 1.3 Hz ────────────────────────────────────────────────
print("1. Loading …")
all_dirs = sorted(glob.glob(str(BASE / "waveprocessed" / "PROCESSED-*")))
meta, _, _, _ = load_analysis_data(*all_dirs, load_processed=False)

m = meta[
    (meta["WaveFrequencyInput [Hz]"].round(2) == 1.30)
    & meta["PanelCondition"].isin(["full", "reverse"])
    & meta["WindCondition"].isin(["no", "full"])
    & (meta["quality_flag"] == "ok")
    & meta["OUT/IN (FFT)"].notna()
    & (meta["OUT/IN (FFT)"].between(0.1, 2.0))
    & (meta["Mooring"] != "above_200")
    & meta["in_probes_used"].notna()
    & meta["out_probes_used"].notna()
].copy()
m["file_date_dt"] = pd.to_datetime(m["file_date"]).dt.tz_localize(None)
m["cfg"] = m["file_date_dt"].apply(lambda d: get_configuration_for_date(d).name if pd.notnull(d) else "N/A")

def _category(row):
    mo, c = row["Mooring"], row["cfg"]
    if mo == "below_90_loose300" and c == "march2026_better_rearranging": return "canon_loose300"
    if mo == "below_90_loose230" and c == "march2026_better_rearranging": return "canon_loose230"
    if mo == "above_50"          and c == "march2026_better_rearranging": return "canon_above_loose"
    if mo == "above_50"          and c == "nov_normalt_oppsett":          return "nov_above_stiff"
    return "other"
m["category"] = m.apply(_category, axis=1)
m = m[m["category"] != "other"].copy()

# LS override
def kt(row, suffix):
    inp = [p.strip() for p in str(row["in_probes_used"]).split("+")]
    out = [p.strip() for p in str(row["out_probes_used"]).split("+")]
    try:
        ai = np.nanmean([row[f"Probe {p} Amplitude{suffix}"] for p in inp])
        ao = np.nanmean([row[f"Probe {p} Amplitude{suffix}"] for p in out])
        return ao/ai if ai > 0 else np.nan
    except KeyError: return np.nan

m["Kt_FFT"] = m["OUT/IN (FFT)"]
m["Kt_LS"]  = m.apply(lambda r: kt(r, " (LS)"),  axis=1)
m["Kt_PSD"] = m.apply(lambda r: kt(r, " (PSD)"), axis=1)
_ovr = ((m["Kt_FFT"]-m["Kt_LS"]).abs()>OVERRIDE_THRESHOLD) \
       & ((m["Kt_FFT"]-m["Kt_PSD"]).abs()>OVERRIDE_THRESHOLD) \
       & ((m["Kt_LS"]-m["Kt_PSD"]).abs()<OVERRIDE_THRESHOLD)
m["Kt_override"] = _ovr
m["Kt_eff"] = np.where(_ovr, m["Kt_LS"], m["Kt_FFT"])
m["amp"] = m["WaveAmplitudeInput [Volt]"].round(2)

print(f"   {len(m)} runs at 1.3 Hz across 4 moorings")
print(f"   LS override fires on {int(_ovr.sum())} of {len(m)} runs")

# Summary
agg = (m.groupby(["category", "WindCondition", "amp"])
        .agg(n=("Kt_eff", "size"),
             mean=("Kt_eff", "mean"),
             std=("Kt_eff", "std"))
        .round(3).reset_index())
agg.to_csv(OUT_CSV, index=False)
print(f"   Summary → {OUT_CSV.relative_to(BASE)}")
print("\n=== Cell means (K_t_eff) ===")
print(agg.to_string(index=False))

# ── 2. Plot ───────────────────────────────────────────────────────────────────
COLORS = {
    ("canon_loose300",    "no"):   WIND_COLOR_MAP["no"],
    ("canon_loose300",    "full"): WIND_COLOR_MAP["full"],
    ("canon_loose230",    "no"):   "#9ECAE1",
    ("canon_loose230",    "full"): "#FCAE91",
    ("canon_above_loose", "no"):   "#1ABC9C",
    ("canon_above_loose", "full"): "#E91E63",
    ("nov_above_stiff",   "no"):   "#9B59B6",
    ("nov_above_stiff",   "full"): "#F1C40F",
}
CATEGORY_LABEL = {
    "canon_loose300":    "Under, 30 cm strikk (canon)",
    "canon_loose230":    "Under, 23 cm strikk (canon)",
    "canon_above_loose": "Over, 16 cm strikk (canon)",
    "nov_above_stiff":   "Over, 6 cm strikk + revers (Nov 2025)",
}
MARKERS = {0.10: "o", 0.20: "s", 0.30: "^"}
CATEGORY_ORDER = ["nov_above_stiff", "canon_above_loose", "canon_loose230", "canon_loose300"]

fig, ax = plt.subplots(figsize=(7.5, 6.5))

# Horizontal x-positions: A1 at 0.1, A2 at 0.2, A3 at 0.3. We jitter
# per (category × wind) so all 8 series at each amp are side-by-side
# rather than overlapping. Within a series, individual runs jitter
# vertically near 0 to spread overlapping K_t values.
N_CATS = len(CATEGORY_ORDER)
WIDTH = 0.018   # total x-span per amp tier
rng = np.random.default_rng(seed=42)

# Compute x-offset per (cat, wind): 8 slots evenly spaced inside ±WIDTH around amp.
slot_idx = {}
slots = [(c, w) for c in CATEGORY_ORDER for w in ("no", "full")]
n_slots = len(slots)
for i, (c, w) in enumerate(slots):
    # offsets symmetric about 0
    slot_idx[(c, w)] = (i - (n_slots - 1) / 2.0) * (WIDTH * 2 / n_slots)

# Plot runs as dots and connect cell means with thin segments
for cat in CATEGORY_ORDER:
    for wind in ("no", "full"):
        s = m[(m["category"] == cat) & (m["WindCondition"] == wind)]
        if s.empty: continue
        color = COLORS[(cat, wind)]
        # Individual run dots
        xs, ys = [], []
        for amp_v in (0.10, 0.20, 0.30):
            ss = s[s["amp"] == amp_v]
            if ss.empty: continue
            n = len(ss)
            base_x = amp_v + slot_idx[(cat, wind)]
            jitter_x = rng.uniform(-0.0035, 0.0035, size=n)
            ax.scatter(
                base_x + jitter_x, ss["Kt_eff"],
                facecolors=color, edgecolors="black",
                marker=MARKERS[amp_v], s=42,
                linewidths=0.4, alpha=0.85,
                zorder={"canon_loose300":4, "canon_loose230":3,
                        "canon_above_loose":2, "nov_above_stiff":2}.get(cat, 2),
            )
            xs.append(base_x); ys.append(ss["Kt_eff"].mean())
        # Connect means
        if len(xs) >= 2:
            ax.plot(xs, ys, color=color, lw=1.2, alpha=0.55,
                    zorder={"canon_loose300":4, "canon_loose230":3,
                            "canon_above_loose":2, "nov_above_stiff":2}.get(cat, 2))

ax.axhline(1.0, color="black", lw=0.6, ls="--", alpha=0.5)
ax.set_xticks([0.10, 0.20, 0.30])
ax.set_xticklabels([r"$A_1$ (0,1 V)", r"$A_2$ (0,2 V)", r"$A_3$ (0,3 V)"])
ax.set_xlabel("Amplitude (paddleinput)", fontsize=11)
ax.yaxis.set_major_locator(MultipleLocator(0.05))
ax.yaxis.set_minor_locator(MultipleLocator(0.025))
ax.set_ylim(0.50, 0.92)
ax.grid(which="major", alpha=0.30, lw=0.6)
ax.grid(which="minor", alpha=0.15, lw=0.4)
ax.set_title(r"Tverrsnitt ved 1,3 Hz: $K_t$ mot amplitude per forankring & vind",
             fontsize=11, loc="left")

apply_horizontal_ylabel(ax, r"$K_t$", fontsize=12)

# Legend: same scheme as 4-way scatter
config_handles = []
for cat in CATEGORY_ORDER[::-1]:
    for wind, wlabel in (("full", "full vind"), ("no", "uten vind")):
        if not ((m["category"] == cat) & (m["WindCondition"] == wind)).any(): continue
        wc = COLORS[(cat, wind)]
        config_handles.append(
            mlines.Line2D([], [], marker="o", linestyle="None",
                          markerfacecolor=wc, markeredgecolor="black",
                          markeredgewidth=0.4, markersize=8,
                          label=f"{CATEGORY_LABEL[cat]}, {wlabel}")
        )

leg = ax.legend(handles=config_handles, loc="lower right",
                bbox_to_anchor=(0.998, 0.002),
                fontsize=7.5, framealpha=0.92,
                title="Forankring × vind", title_fontsize=8,
                ncol=1)

fig.subplots_adjust(left=0.10, right=0.98, top=0.93, bottom=0.10)

OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.02)
print(f"\n   Saved → {OUT_PDF.relative_to(BASE)}")
plt.close(fig)

print("\nDone.")
