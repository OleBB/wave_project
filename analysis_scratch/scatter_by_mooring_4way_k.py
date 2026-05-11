"""
4-way mooring split — k-axis scatter, vivid palette (scratch)
==============================================================

Splits the over+under-mooring data into the four physically distinct
mooring configurations identified from the lab notebook (2026-05-11
context dump). Each gets a wind-color pair, and the four pairs are
visually well-separated so the four moorings can be inspected as
independent populations.

The four mooring conditions in scope:

  - canon_loose300   = canon era, below_90, 9 cm under water, 30 cm strikk,
                       full panel  (stock red / stock blue)
  - canon_loose230   = canon era, below_90, 9 cm under water, 23 cm strikk,
                       full panel  (lighter red / lighter blue)
  - canon_above_loose= canon era, 5 cm above water, 16 cm strikk,
                       full panel  (pink / turquoise)
  - nov_above_stiff  = Nov 2025 era, 5 cm above water, 6 cm strikk,
                       REVERSE panel  (yellow / purple)

The "above_50" Mooring tag in meta.json conflates canon_above_loose and
nov_above_stiff — they share the same height-above-water but differ in
strikk stiffness (16 cm vs 6 cm stretched), panel orientation (full vs
reverse), AND probe averaging scheme (IN paired vs OUT paired). Here we
disambiguate them via the probe-geometry config (file_date → config).

K_t source: per-row FFT canonical with three-way LS+PSD override
(threshold 0.05, |LS−PSD| < 0.05 for swap to trigger). Same rule as the
sibling scripts under_and_over_mooring_scatter_k.py / all_data_ka.

This is a SCRATCH figure (analysis_scratch/ output only). No thesis
stub. The thesis-track scatters are untouched.

Run from repo root:
    conda run -n draumkvedet python analysis_scratch/scatter_by_mooring_4way_k.py
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
    freq_to_k, add_freq_axis, WIND_COLOR_MAP,
)

apply_thesis_style()

OVERRIDE_THRESHOLD = 0.05
OUT_PDF = Path(__file__).parent / "scatter_by_mooring_4way.pdf"
OUT_CSV = Path(__file__).parent / "scatter_by_mooring_4way_summary.csv"

# ── 1. Load ────────────────────────────────────────────────────────────────────
print("1. Loading all processed folders …")
all_dirs = sorted(glob.glob(str(BASE / "waveprocessed" / "PROCESSED-*")))
print(f"   {len(all_dirs)} folders")
meta, _, _, _ = load_analysis_data(*all_dirs, load_processed=False)
print(f"   {len(meta)} total rows")

# ── 2. Filter (same as the over-mooring scatter scope) ───────────────────────
wave = meta[
    meta["WaveFrequencyInput [Hz]"].notna()
    & (meta["WaveFrequencyInput [Hz]"] > 0)
    & meta["PanelCondition"].isin(["full", "reverse"])
    & meta["WindCondition"].isin(["no", "full"])
    & (meta["quality_flag"] == "ok")
    & meta["OUT/IN (FFT)"].notna()
    & (meta["OUT/IN (FFT)"].between(0.1, 2.0))
    & (meta["WaveFrequencyInput [Hz]"] < 2.0)
    & (meta["Mooring"] != "above_200")
    & meta["in_probes_used"].notna()
    & meta["out_probes_used"].notna()
].copy()
print(f"   {len(wave)} runs after standard filter")

# Probe-geometry config per row (from file_date).
wave["file_date_dt"] = pd.to_datetime(wave["file_date"]).dt.tz_localize(None)
wave["cfg"] = wave["file_date_dt"].apply(
    lambda d: get_configuration_for_date(d).name if pd.notnull(d) else "N/A"
)

# 4-way categorisation.
def _category(row):
    m, c = row["Mooring"], row["cfg"]
    if m == "below_90_loose300" and c == "march2026_better_rearranging": return "canon_loose300"
    if m == "below_90_loose230" and c == "march2026_better_rearranging": return "canon_loose230"
    if m == "above_50"          and c == "march2026_better_rearranging": return "canon_above_loose"
    if m == "above_50"          and c == "nov_normalt_oppsett":          return "nov_above_stiff"
    return "other"

wave["category"] = wave.apply(_category, axis=1)
n_other = int((wave["category"] == "other").sum())
if n_other:
    print(f"   {n_other} runs in 'other' category dropped")
    wave = wave[wave["category"] != "other"].copy()

print("\n2. Counts per (category × wind):")
print(wave.groupby(["category", "WindCondition"]).size().unstack(fill_value=0).to_string())
print(f"   total: {len(wave)} runs")

# ── 3. K_t override rule (same as siblings) ──────────────────────────────────
def _kt_method(row, method_suffix):
    inp = [p.strip() for p in str(row["in_probes_used"]).split("+")]
    out = [p.strip() for p in str(row["out_probes_used"]).split("+")]
    try:
        a_in  = float(np.nanmean([row[f"Probe {p} Amplitude{method_suffix}"] for p in inp]))
        a_out = float(np.nanmean([row[f"Probe {p} Amplitude{method_suffix}"] for p in out]))
        if a_in <= 0 or not np.isfinite(a_in) or not np.isfinite(a_out):
            return np.nan
        return a_out / a_in
    except KeyError:
        return np.nan

wave["Kt_LS"]    = wave.apply(lambda r: _kt_method(r, " (LS)"),  axis=1)
wave["Kt_PSD"]   = wave.apply(lambda r: _kt_method(r, " (PSD)"), axis=1)
wave["Kt_canon"] = wave["OUT/IN (FFT)"].astype(float)

_dF  = (wave["Kt_canon"] - wave["Kt_LS"]).abs()
_dP  = (wave["Kt_canon"] - wave["Kt_PSD"]).abs()
_dLP = (wave["Kt_LS"]    - wave["Kt_PSD"]).abs()
_override = (_dF > OVERRIDE_THRESHOLD) & (_dP > OVERRIDE_THRESHOLD) & (_dLP < OVERRIDE_THRESHOLD)
wave["Kt_override"] = _override
wave["Kt_eff"] = np.where(_override, wave["Kt_LS"], wave["Kt_canon"])

print(f"\n   K_t override (FFT → LS) fires on "
      f"{int(_override.sum())} of {len(wave)} runs "
      f"(threshold |ΔKt| > {OVERRIDE_THRESHOLD}, LS+PSD agreement required)")

wave["k"] = freq_to_k(wave["WaveFrequencyInput [Hz]"].values)

# Per-cell CSV for review
summary = (wave.groupby(["category", "WindCondition", "WaveFrequencyInput [Hz]"])
                .agg(n=("Kt_eff", "size"),
                     Kt_mean=("Kt_eff", "mean"),
                     Kt_std=("Kt_eff", "std"))
                .round(3).reset_index())
summary.to_csv(OUT_CSV, index=False)
print(f"   Summary → {OUT_CSV.relative_to(BASE)}")

# ── 4. Vivid 4-way palette (8 colours total = 4 moorings × 2 winds) ──────────
COLORS = {
    # below_90, 30 cm strikk (canon §1 reference) — stock matplotlib pair
    ("canon_loose300",    "no"):   WIND_COLOR_MAP["no"],     # stock blue   ~ #1F77B4
    ("canon_loose300",    "full"): WIND_COLOR_MAP["full"],   # stock red    ~ #D62728

    # below_90, 23 cm strikk — lighter shades (tints)
    ("canon_loose230",    "no"):   "#9ECAE1",                # light blue
    ("canon_loose230",    "full"): "#FCAE91",                # light red / salmon

    # above_50 (canon era, 16 cm loose strikk, full panel) — pink/turquoise
    ("canon_above_loose", "no"):   "#1ABC9C",                # turquoise
    ("canon_above_loose", "full"): "#E91E63",                # pink

    # above_50 (Nov 2025, 6 cm stiff strikk, REVERSE panel) — yellow/purple
    ("nov_above_stiff",   "no"):   "#9B59B6",                # purple
    ("nov_above_stiff",   "full"): "#F1C40F",                # vivid yellow
}

CATEGORY_LABEL = {
    "canon_loose300":    "Under, 30 cm strikk (canon)",
    "canon_loose230":    "Under, 23 cm strikk (canon)",
    "canon_above_loose": "Over, 16 cm strikk (canon)",
    "nov_above_stiff":   "Over, 6 cm strikk + revers (Nov 2025)",
}
# Plot order — paint canon_loose300 (the thesis-canonical reference) last so
# its dots sit on top. Outer-noisy categories paint first.
CATEGORY_ORDER = [
    "nov_above_stiff",
    "canon_above_loose",
    "canon_loose230",
    "canon_loose300",
]
MARKERS = {0.10: "o", 0.20: "s", 0.30: "^"}
MARKER_SIZE = 55
ALPHA = 0.80
EDGE_LW = 0.4

def _round_amp(v): return round(float(v), 2)

# Shared axis envelope
_k_min = float(wave["k"].min())
_k_max = float(wave["k"].max())
_x_pad = max(0.15, 0.02 * (_k_max - _k_min))
XLIM = (_k_min - _x_pad, _k_max * 1.02)
YLIM = (0.1, 1.18)

THESIS_K_LO = float(freq_to_k(np.array([1.3]))[0])
THESIS_K_HI = float(freq_to_k(np.array([1.6]))[0])

# ── 5. Plot ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.27, 9.5))

for cat in CATEGORY_ORDER:
    sub_cat = wave[wave["category"] == cat]
    if sub_cat.empty:
        continue
    for wind in ("no", "full"):
        for amp_v in (0.10, 0.20, 0.30):
            s = sub_cat[(sub_cat["WindCondition"] == wind)
                        & (sub_cat["WaveAmplitudeInput [Volt]"].apply(_round_amp) == amp_v)]
            if s.empty: continue
            color  = COLORS.get((cat, wind), "gray")
            marker = MARKERS[amp_v]
            ax.scatter(
                s["k"], s["Kt_eff"],
                facecolors=color, edgecolors="black",
                marker=marker, s=MARKER_SIZE,
                linewidths=EDGE_LW, alpha=ALPHA,
                zorder={"canon_loose300":4, "canon_loose230":3,
                        "canon_above_loose":2, "nov_above_stiff":2}.get(cat, 2),
            )

# Thesis-scope band
ax.axvspan(THESIS_K_LO, THESIS_K_HI,
           color=WIND_COLOR_MAP["no"], alpha=0.07, lw=0, zorder=1)
ax.text(THESIS_K_HI - 0.05, YLIM[1] - 0.04,
        "Hovedfokus\n1,3–1,6 Hz", ha="right", va="top",
        fontsize=8, color="#1F618D", alpha=0.85,
        bbox=dict(boxstyle="round,pad=0.2",
                  facecolor="white", alpha=0.75, edgecolor="none"))

ax.axhline(1.0, color="black", lw=0.6, ls="--", alpha=0.5)
ax.set_xlabel("$k$", fontsize=11)
ax.set_ylabel("")
ax.yaxis.set_major_locator(MultipleLocator(0.1))
ax.yaxis.set_minor_locator(MultipleLocator(0.05))
ax.xaxis.set_major_locator(MultipleLocator(1.0))
ax.xaxis.set_minor_locator(MultipleLocator(0.5))
ax.grid(which="major", alpha=0.30, lw=0.6)
ax.grid(which="minor", alpha=0.15, lw=0.4)
ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)

# LS-override annotation (lower-left)
_n_ls_swap = int(wave["Kt_override"].sum())
ax.text(
    XLIM[0] + 0.02 * (XLIM[1] - XLIM[0]),
    YLIM[0] + 0.025,
    f"n = {_n_ls_swap} data estimert med LS",
    ha="left", va="bottom",
    fontsize=8, color="#555555", alpha=0.90,
    bbox=dict(boxstyle="round,pad=0.25",
              facecolor="white", alpha=0.80, edgecolor="none"),
    zorder=5,
)

# Top frequency axis
secax = add_freq_axis(ax)
secax.set_xlabel(r"Frekvens (Hz)", fontsize=9)
_used_freqs = sorted(wave["WaveFrequencyInput [Hz]"].unique())
secax.set_xticks(_used_freqs)
secax.set_xticklabels([f"{f:.1f}" for f in _used_freqs])
secax.tick_params(labelsize=7)

# ── Legends — Konfigurasjon × Vind, Amplitude ──
config_handles = []
for cat in CATEGORY_ORDER[::-1]:  # legend top→bottom: canon_loose300, 230, above_loose, nov_stiff
    sub_cat = wave[wave["category"] == cat]
    if sub_cat.empty: continue
    for wind, wlabel in (("full", "full vind"), ("no", "uten vind")):
        if not (sub_cat["WindCondition"] == wind).any(): continue
        wind_color = COLORS[(cat, wind)]
        config_handles.append(
            mlines.Line2D([], [], marker="o", linestyle="None",
                          markerfacecolor=wind_color, markeredgecolor="black",
                          markeredgewidth=0.4, markersize=9,
                          label=f"{CATEGORY_LABEL[cat]}, {wlabel}")
        )
amp_handles = [
    mlines.Line2D([], [], color="black",
                  marker=MARKERS[v], linestyle="None", markersize=8,
                  markerfacecolor="lightgray", markeredgecolor="black",
                  markeredgewidth=0.3, label=amp_to_label(v))
    for v in (0.10, 0.20, 0.30)
]

leg_cfg = ax.legend(handles=config_handles, loc="upper right",
                    bbox_to_anchor=(0.995, 0.995),
                    fontsize=8, framealpha=0.92,
                    title="Forankring × vind", title_fontsize=8,
                    ncol=1)
ax.add_artist(leg_cfg)
fig.canvas.draw()
leg_cfg_bbox = leg_cfg.get_window_extent().transformed(ax.transAxes.inverted())
amp_anchor_y = leg_cfg_bbox.y0 - 0.010
ax.legend(handles=amp_handles, loc="upper right",
          bbox_to_anchor=(0.995, amp_anchor_y),
          fontsize=8, framealpha=0.92,
          title="Amplitude", title_fontsize=8)

fig.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.06)
apply_horizontal_ylabel(ax, r"$K_t$", fontsize=12)

OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.02)
print(f"\n   Saved → {OUT_PDF.relative_to(BASE)}")
plt.close(fig)

print("\nDone.")
