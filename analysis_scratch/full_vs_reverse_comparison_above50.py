"""
Full vs reverse — corrected to above_50 mooring only
=====================================================

Thread 2a's `full_vs_reverse_comparison.py` did NOT restrict by mooring.
Full-panel data at 1.30 Hz pooled across all moorings (above_50 +
below_90_loose230 + below_90_loose300, ~83 rows). Reverse-panel data is
above_50-only by physical existence (only mooring where reverse was run).
That made full mooring-mixed and reverse above_50-only — not like-for-like.

Below_90 moorings transmit ~0.05–0.10 more K_t than above_50 (Thread 4
Row A). Mixing them inflates the full-panel mean. This script corrects
the comparison: both panels restricted to `above_50`, so the only variable
is panel geometry.

Same visual layout as the original (1 × N facets per reverse-panel freq).

Outputs:
    analysis_scratch/full_vs_reverse_comparison_above50.pdf
    output/FIGURES/ch05_full_vs_reverse_comparison_above50.pdf
    analysis_scratch/full_vs_reverse_comparison_above50_summary.csv
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

SCRATCH_PDF = Path(__file__).parent / "full_vs_reverse_comparison_above50.pdf"
SCRATCH_CSV = Path(__file__).parent / "full_vs_reverse_comparison_above50_summary.csv"
OUT_PDF     = BASE / "output" / "FIGURES" / "ch05_full_vs_reverse_comparison_above50.pdf"
OUT_PDF.parent.mkdir(parents=True, exist_ok=True)

# ── 1. Load & filter ───────────────────────────────────────────────────────────
print("1. Loading processed folders …")
all_dirs = sorted(glob.glob(str(BASE / "waveprocessed" / "PROCESSED-*")))
meta, _, _, _ = load_analysis_data(*all_dirs, load_processed=False)

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
    & (meta["Mooring"] == "above_50")        # ← the key restriction
].copy()
print(f"   {len(sel)} rows after above_50 restriction")

# Frequencies where reverse panel exists.
rev_freqs = sorted(sel.loc[sel["PanelCondition"] == "reverse",
                            "WaveFrequencyInput [Hz]"].unique())
print(f"   reverse-panel freqs (above_50 only): {rev_freqs}")
sel = sel[sel["WaveFrequencyInput [Hz]"].isin(rev_freqs)].copy()
print(f"   {len(sel)} rows at those freqs (full + reverse, above_50 only)")

# Compare to the mooring-mixed counts for context.
mixed_full = meta[
    meta["WaveFrequencyInput [Hz]"].isin(rev_freqs)
    & (meta["PanelCondition"] == "full")
    & (meta["quality_flag"] == "ok")
    & meta["OUT/IN (FFT)"].notna()
    & meta["WindCondition"].isin(["no", "full"])
    & (meta["OUT/IN (FFT)"].between(0.1, 2.0))
]
print(f"   (for context: mooring-mixed full-panel n at same freqs = "
      f"{len(mixed_full)})")

# ── 2. Plot ────────────────────────────────────────────────────────────────────
WIND_LABEL  = {"no": "uten vind", "full": "med vind"}
PANEL_X     = {"full": 0, "reverse": 1}
PANEL_LABEL = {"full": "full", "reverse": "revers"}
AMP_MARKER  = {0.10: "o", 0.20: "s", 0.30: "^"}
MARKER_SIZE = 60
JITTER_HALF_WIDTH = 0.10
ALPHA_FILLED   = 0.65
ALPHA_HOLLOW   = 0.85
EDGE_LW_FILLED = 0.3
EDGE_LW_HOLLOW = 1.4
WIND_X_OFFSET = {"no": -0.18, "full": +0.18}

def _round_amp(v): return round(float(v), 2)
def _jitter(n, half=JITTER_HALF_WIDTH, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-half, half, size=n)

fig, axes = plt.subplots(1, len(rev_freqs),
                          figsize=(3.6 * len(rev_freqs), 5.6),
                          sharey=True)
if len(rev_freqs) == 1:
    axes = [axes]

summary_rows = []
seed = 0
for ax, f in zip(axes, rev_freqs):
    sub = sel[sel["WaveFrequencyInput [Hz]"] == f].copy()

    for panel, x_base in PANEL_X.items():
        sp = sub[sub["PanelCondition"] == panel]
        if sp.empty:
            ax.text(x_base, 0.5, "n = 0",
                    ha="center", va="center",
                    transform=ax.get_xaxis_transform(),
                    fontsize=9, color="#888",
                    bbox=dict(boxstyle="round,pad=0.25",
                              facecolor="white", alpha=0.85,
                              edgecolor="#bbb"))
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
    ax.set_xlabel("Panelgeometri", fontsize=10)

axes[0].set_ylabel(r"$K_t$", fontsize=12, rotation=0, ha="right", va="center")

y_lo = sel["OUT/IN (FFT)"].min() - 0.02
y_hi = sel["OUT/IN (FFT)"].max() + 0.02
for ax in axes:
    ax.set_ylim(y_lo, y_hi)

# Indicate the above_50 restriction at the top of the figure.
fig.suptitle("Full vs reverse — restricted to Mooring = above_50",
              fontsize=11, y=1.00)

wind_handles = [
    mlines.Line2D([], [], color=WIND_COLOR_MAP[w], lw=4, label=WIND_LABEL[w])
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
                  label="middel (errorbar = std)"),
]
leg1 = axes[-1].legend(handles=wind_handles + mean_handle, loc="lower right",
                        fontsize=8, framealpha=0.92,
                        title="Vind", title_fontsize=8,
                        bbox_to_anchor=(0.99, 0.02))
axes[-1].add_artist(leg1)
axes[-1].legend(handles=amp_handles + hw_handles, loc="upper right",
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

# ── 3. Summary CSV + before/after comparison ───────────────────────────────────
summary = pd.DataFrame(summary_rows)
summary.to_csv(SCRATCH_CSV, index=False)
print(f"   Summary → {SCRATCH_CSV.relative_to(BASE)}\n")
print(summary.to_string(index=False))

# Compute Δ_wind per (freq, panel) for the like-for-like report.
print("\n   Wind effect Δ = mean_kt(fullwind) − mean_kt(nowind):")
for (f, p), grp in summary.groupby(["freq_hz", "panel"]):
    g = grp.set_index("wind")
    if "no" in g.index and "full" in g.index:
        d = g.loc["full", "mean_kt"] - g.loc["no", "mean_kt"]
        print(f"     f={f} Hz, panel={p:7}: Δ = {d:+.4f}  "
              f"(n_no={int(g.loc['no','n'])}, n_fw={int(g.loc['full','n'])})")
    else:
        print(f"     f={f} Hz, panel={p:7}: incomplete (no head-to-head wind)")

# Print before/after table — Thread 2a's mooring-mixed numbers vs above_50 only.
print("\n   Before (mooring-mixed) vs after (above_50 only) — 1.30 Hz:")
print("     OLD  full nowind  ≈ 0.688   |  NEW (above_50 only) "
      f"= {summary[(summary['freq_hz']==1.3)&(summary['panel']=='full')&(summary['wind']=='no')]['mean_kt'].iloc[0]:.4f}")
print("     OLD  full fullwd  ≈ 0.770   |  NEW (above_50 only) "
      f"= {summary[(summary['freq_hz']==1.3)&(summary['panel']=='full')&(summary['wind']=='full')]['mean_kt'].iloc[0]:.4f}")

print("\nDone.")
