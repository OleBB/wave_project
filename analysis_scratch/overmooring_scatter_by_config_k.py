"""
Over-mooring K_t scatter SPLIT BY PROBE-GEOMETRY CONFIG (scratch, k-axis)
=========================================================================

Investigates the transition zone (~1.5 Hz sign-flip) at above_50 mooring
by splitting the scatter into the two probe-geometry configs that
contribute to it. The two configs have inverted IN/OUT averaging
schemes, so any sign-flip that appears in both is a real physical
effect; one that appears only in one is a probe-geometry artefact.

Configs in scope (after the standard filter set):
  - march2026_better_rearranging  (the canon era):
      IN  = mean(9373/170, 9373/340)  (parallel pair)
      OUT = 12400/250                  (single)
      Panel: full only
  - nov_normalt_oppsett  (Nov 2025):
      IN  = 9373/250                   (single)
      OUT = mean(12400/170, 12400/340) (parallel pair)
      Panel: reverse only (after quality + K_t filters)

So in this scope, panel orientation collinear with config:
    canon ↔ full panel
    nov   ↔ reverse panel

Encoding:
  - Color shade  = (config, wind) — 4 distinct shades:
      canon × no    : standard blue   (#1F77B4)
      canon × full  : standard red    (#D62728)
      nov   × no    : steel blue      (#4682B4)
      nov   × full  : firebrick       (#B22222)
  - Marker shape = amplitude tier (○ A1, □ A2, △ A3)
  - Marker fill  = solid (no hollow — config dimension is in colour)
  - Marker edge  = black thin (0.4 px)

K_t source: per-row FFT canonical with three-way LS+PSD override
(mirrors the sibling scripts; see top-of-file IMMUTABLE blocks there).
Override fires on the small handful of runs where FFT is broken by
cut_samples / window-coherence issues.

This is a SCRATCH figure (analysis_scratch/ output only). The
thesis-track ch05_damping_overmooring_scatter.pdf is unaffected.

Run from repo root:
    conda run -n draumkvedet python analysis_scratch/overmooring_scatter_by_config_k.py
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

# Per-row FFT→LS override threshold (same rule + magic number as
# under_and_over_mooring_scatter_k.py — see that script's top-of-file
# IMMUTABLE block for full motivation).
OVERRIDE_THRESHOLD = 0.05

# Scratch outputs
OUT_PDF = Path(__file__).parent / "overmooring_scatter_by_config.pdf"
OUT_CSV = Path(__file__).parent / "overmooring_scatter_by_config_summary.csv"

# ── 1. Load ────────────────────────────────────────────────────────────────────
print("1. Loading all processed folders …")
all_dirs = sorted(glob.glob(str(BASE / "waveprocessed" / "PROCESSED-*")))
print(f"   {len(all_dirs)} folders")
meta, _, _, _ = load_analysis_data(*all_dirs, load_processed=False)
print(f"   {len(meta)} total rows")

# ── 2. Filter to over-mooring scatter scope ───────────────────────────────────
wave = meta[
    meta["WaveFrequencyInput [Hz]"].notna()
    & (meta["WaveFrequencyInput [Hz]"] > 0)
    & (meta["Mooring"] == "above_50")
    & meta["PanelCondition"].isin(["full", "reverse"])
    & meta["WindCondition"].isin(["no", "full"])
    & (meta["quality_flag"] == "ok")
    & meta["OUT/IN (FFT)"].notna()
    & (meta["OUT/IN (FFT)"].between(0.1, 2.0))
    & (meta["WaveFrequencyInput [Hz]"] < 2.0)
    & meta["in_probes_used"].notna()
    & meta["out_probes_used"].notna()
].copy()
print(f"   {len(wave)} runs after over-mooring scatter filter")

# Probe-geometry config per row (from file_date).
wave["file_date_dt"] = pd.to_datetime(wave["file_date"]).dt.tz_localize(None)
wave["cfg"] = wave["file_date_dt"].apply(
    lambda d: get_configuration_for_date(d).name if pd.notnull(d) else "N/A"
)

# Category = (cfg, panel). Drop the 0-row combinations.
def _category(row):
    c, p = row["cfg"], row["PanelCondition"]
    if c == "march2026_better_rearranging" and p == "full":    return "canon_full"
    if c == "nov_normalt_oppsett"          and p == "reverse": return "nov_reverse"
    if c == "nov_normalt_oppsett"          and p == "full":    return "nov_full"   # rare (n=0 after filter usually)
    if c == "march2026_better_rearranging" and p == "reverse": return "canon_reverse"  # doesn't exist
    return "other"

wave["category"] = wave.apply(_category, axis=1)
n_other = int((wave["category"] == "other").sum())
if n_other:
    print(f"   {n_other} runs in 'other' category dropped")
    wave = wave[wave["category"] != "other"].copy()

print("\n2. Counts per (category × wind):")
print(wave.groupby(["category", "WindCondition"]).size().unstack(fill_value=0).to_string())
print(f"   total: {len(wave)} runs")

# ── 3. Per-row K_t recomputation under FFT, LS, PSD, then override ───────────
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
if _override.any():
    print(wave[_override][["WaveFrequencyInput [Hz]", "WaveAmplitudeInput [Volt]",
                           "WindCondition", "category", "Kt_canon", "Kt_LS", "Kt_PSD"]].to_string())

# x-axis: k via dispersion
wave["k"] = freq_to_k(wave["WaveFrequencyInput [Hz]"].values)

# Per-cell summary CSV
summary = (wave.groupby(["category", "WindCondition", "WaveFrequencyInput [Hz]"])
                .agg(n=("Kt_eff", "size"),
                     Kt_mean=("Kt_eff", "mean"),
                     Kt_std=("Kt_eff", "std"))
                .round(3).reset_index())
summary.to_csv(OUT_CSV, index=False)
print(f"\n   Summary → {OUT_CSV.relative_to(BASE)}")

# ── 4. Visual encoding (4 colour shades) ──────────────────────────────────────
# canon × no   : standard blue,         WIND_COLOR_MAP["no"]   ~ #1F77B4
# canon × full : standard red,          WIND_COLOR_MAP["full"] ~ #D62728
# nov   × no   : steel blue (muted)     #4682B4
# nov   × full : firebrick (muted dark) #B22222
# (Nov full-panel runs, if any survive, get a lighter pair distinct
# from both — light blue / orange salmon.)
COLORS = {
    ("canon_full",     "no"):   WIND_COLOR_MAP["no"],
    ("canon_full",     "full"): WIND_COLOR_MAP["full"],
    ("nov_reverse",    "no"):   "#4682B4",   # steel blue
    ("nov_reverse",    "full"): "#B22222",   # firebrick
    ("nov_full",       "no"):   "#9ECAE1",   # light blue (rare)
    ("nov_full",       "full"): "#F4815A",   # orange salmon (rare)
}
CATEGORY_LABEL = {
    "canon_full":  "Canon (mar2026), panel: full",
    "nov_reverse": "Nov 2025, panel: revers",
    "nov_full":    "Nov 2025, panel: full",
}
# Plot order: canon last so its dots paint over Nov where they overlap.
CATEGORY_ORDER = ["nov_full", "nov_reverse", "canon_full"]
MARKERS = {0.10: "o", 0.20: "s", 0.30: "^"}
WIND_LABEL = {"no": "uten vind", "full": "full vind"}
MARKER_SIZE = 55
ALPHA = 0.75
EDGE_LW = 0.4

def _round_amp(v): return round(float(v), 2)

# Shared axis envelope (same family as the existing over-mooring scatter)
_k_min = float(wave["k"].min())
_k_max = float(wave["k"].max())
_x_pad = max(0.15, 0.02 * (_k_max - _k_min))
XLIM = (_k_min - _x_pad, _k_max * 1.02)
YLIM = (0.1, 1.18)

THESIS_K_LO = float(freq_to_k(np.array([1.3]))[0])
THESIS_K_HI = float(freq_to_k(np.array([1.6]))[0])

# ── 5. Plot ────────────────────────────────────────────────────────────────────
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
                zorder=3 if cat == "canon_full" else 2,
            )
            # Per-cell linear fit (one line per cat × wind × amp where ≥2 unique k).
            k_cell = s["k"].to_numpy(float)
            kt_cell = s["Kt_eff"].to_numpy(float)
            if len(k_cell) >= 2 and k_cell.std() > 1e-9:
                p = np.polyfit(k_cell, kt_cell, deg=1)
                xl = np.array([k_cell.min(), k_cell.max()])
                ax.plot(xl, np.polyval(p, xl),
                        color=color, lw=0.6, alpha=0.45,
                        zorder=3 if cat == "canon_full" else 2,
                        solid_capstyle="round")

# Thesis-scope band + annotation
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

# ── Legends (Konfigurasjon × wind + Amplitude) ──
config_handles = []
for cat in CATEGORY_ORDER[::-1]:  # legend top → bottom: canon, nov_reverse, nov_full
    sub_cat = wave[wave["category"] == cat]
    if sub_cat.empty: continue
    for wind, wlabel in (("full", "full vind"), ("no", "uten vind")):
        if not ((sub_cat["WindCondition"] == wind).any()): continue
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
                    title="Konfigurasjon × vind", title_fontsize=8)
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

# ── 6. Diagnostic: per (cfg × wind), what's the per-freq median K_t? ─────────
print("\n=== Per-cfg per-wind median K_t per frequency ===\n")
mt = wave.groupby(["category", "WindCondition", "WaveFrequencyInput [Hz]"])["Kt_eff"].agg(["count", "median", "mean"]).round(3)
print(mt.to_string())

print("\nDone.")
