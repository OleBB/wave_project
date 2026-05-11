"""
ε(f) = ΔK_t / (1 − K_t_nowind) — wind-enhancement coefficient per mooring
==========================================================================

Tests the "panel-compliance ↔ wind-enhancement" hypothesis:

  ε(f) is the fraction of remaining panel blockage that wind removes:
        ε(f) = (K_t,full(f) − K_t,no(f)) / (1 − K_t,no(f))
        ε ≈ 0  → wind has no effect on transmission
        ε ≈ 1  → wind removes 100% of remaining blockage (panel fully lifts)

  - Below-water moorings (panel submerged, rigid): ε constant ≈ 0.32 across f.
  - Above-water canon (loose 16 cm strikk + full panel): ε ≈ 0 below 1.4 Hz,
    jumps to ε ≈ 0.32 above 1.5 Hz. The transition lines up with the panel's
    natural frequency (mass 4.55 kg + strikk stiffness ~400 N/m → f_n ≈ 1.5 Hz).
  - Above-water Nov 2025 stiff (6 cm strikk + reverse panel): only 2 freqs
    in scope (0.65, 1.3 Hz); both at low ε. Predicted f_n is higher (~2.4 Hz)
    so transition (if any) is above our measurement range.

K_t values use the per-row FFT canonical with three-way LS+PSD override.

Run from repo root:
    conda run -n draumkvedet python analysis_scratch/transition_lift_fraction_eps.py
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
    apply_thesis_style, apply_horizontal_ylabel, WIND_COLOR_MAP,
)

apply_thesis_style()

OVERRIDE_THRESHOLD = 0.05
OUT_PDF = Path(__file__).parent / "transition_lift_fraction_eps.pdf"
OUT_CSV = Path(__file__).parent / "transition_lift_fraction_eps_summary.csv"

# ── 1. Load + filter ──────────────────────────────────────────────────────────
print("1. Loading …")
all_dirs = sorted(glob.glob(str(BASE / "waveprocessed" / "PROCESSED-*")))
meta, _, _, _ = load_analysis_data(*all_dirs, load_processed=False)

m = meta[
    meta["WaveFrequencyInput [Hz]"].notna()
    & (meta["WaveFrequencyInput [Hz]"] > 0)
    # Narrowed to 1.2–1.6 Hz: this is the thesis band ± one freq for
    # low-side context. The low-freq mask in eps_valid throws out small-
    # blockage outliers anyway; this filter keeps the figure focused on
    # the transition zone with maximum data density.
    & (meta["WaveFrequencyInput [Hz]"].between(1.2, 1.6))
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

def _cat(r):
    mo, c = r["Mooring"], r["cfg"]
    if mo == "below_90_loose300" and c == "march2026_better_rearranging": return "canon_loose300"
    if mo == "below_90_loose230" and c == "march2026_better_rearranging": return "canon_loose230"
    if mo == "above_50"          and c == "march2026_better_rearranging": return "canon_above"
    if mo == "above_50"          and c == "nov_normalt_oppsett":          return "nov_above_stiff"
    return "other"
m["cat"] = m.apply(_cat, axis=1)
m = m[m["cat"] != "other"].copy()

# LS override (consistent with sibling scripts)
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
m["Kt"] = np.where(_ovr, m["Kt_LS"], m["Kt_FFT"])

# ── 2. Cell-level ε with errorbars ────────────────────────────────────────────
agg = (m.groupby(["cat", "WaveFrequencyInput [Hz]", "WindCondition"])
        .agg(n=("Kt", "size"), Kt_mean=("Kt", "mean"), Kt_std=("Kt", "std"))
        .reset_index())
piv = agg.pivot_table(index=["cat", "WaveFrequencyInput [Hz]"],
                      columns="WindCondition",
                      values=["n", "Kt_mean", "Kt_std"])
piv.columns = [f"{a}_{b}" for a, b in piv.columns]
piv = piv.reset_index()
# Keep only cells where both winds present
piv = piv.dropna(subset=["Kt_mean_full", "Kt_mean_no"]).copy()

piv["blockage_no"] = 1.0 - piv["Kt_mean_no"]
piv["delta_Kt"]    = piv["Kt_mean_full"] - piv["Kt_mean_no"]
piv["eps"]         = piv["delta_Kt"] / piv["blockage_no"]
# Mask ε where blockage is too small to compute reliably (1 − K_t < 0.10).
# At those freqs the panel barely blocks anything, so ε amplifies noise to
# unreadable magnitudes (±7 at 0.5 Hz). Below this threshold the wind-effect
# is itself small and the figure ignores it.
piv["eps_valid"] = piv["blockage_no"] >= 0.10
piv.loc[~piv["eps_valid"], "eps"]   = np.nan
piv.loc[~piv["eps_valid"], "se_eps_to_be"] = np.nan

# Standard error of the mean K_t per cell (NaN std → ±10% fallback)
def _se(std, n, mean):
    if pd.isna(std):
        return 0.10 * abs(mean)
    return std / np.sqrt(max(n, 1))
piv["se_Kt_full"] = piv.apply(lambda r: _se(r["Kt_std_full"], r["n_full"], r["Kt_mean_full"]), axis=1)
piv["se_Kt_no"]   = piv.apply(lambda r: _se(r["Kt_std_no"],   r["n_no"],   r["Kt_mean_no"]),   axis=1)
# Propagate to ε: ε = (Kt_f − Kt_n) / (1 − Kt_n)
# ∂ε/∂Kt_f = 1/(1 − Kt_n);  ∂ε/∂Kt_n = (Kt_f − 1)/(1 − Kt_n)^2
piv["se_eps"] = np.sqrt(
    (piv["se_Kt_full"] / piv["blockage_no"])**2
    + (piv["se_Kt_no"]  * (piv["Kt_mean_full"] - 1) / piv["blockage_no"]**2)**2
)

# CSV
piv.to_csv(OUT_CSV, index=False)
print(f"   Summary → {OUT_CSV.relative_to(BASE)}")

# ── 3. Predicted natural frequencies ──────────────────────────────────────────
# Mass m = 4.55 kg (panel array). Stiffness unknown.
# Solve k from f_n = 1.5 Hz observed for canon_above (16 cm strikk):
M_PANEL_KG = 4.55
F_N_CANON_ABOVE = 1.5  # observed transition
k_canon_above = (2*np.pi*F_N_CANON_ABOVE)**2 * M_PANEL_KG  # N/m
# Scale to Nov stiff (6 cm strikk): rubber bands ≈ 1/L stiffness scaling
k_nov_stiff = k_canon_above * (16.0 / 6.0)
f_n_nov_stiff = (1/(2*np.pi)) * np.sqrt(k_nov_stiff / M_PANEL_KG)
print(f"\n   Estimated panel-mooring natural frequencies (panel m = {M_PANEL_KG} kg):")
print(f"     canon_above (16 cm strikk):   f_n ≈ {F_N_CANON_ABOVE:.2f} Hz  → k ≈ {k_canon_above:.0f} N/m")
print(f"     nov_above_stiff (6 cm strikk): f_n ≈ {f_n_nov_stiff:.2f} Hz  → k ≈ {k_nov_stiff:.0f} N/m  (out of measured range)")

# ── 4. Figure ─────────────────────────────────────────────────────────────────
# Vivid 4-way palette (one solid colour per mooring — ε already integrates wind).
PALETTE = {
    "canon_loose300":  WIND_COLOR_MAP["full"],   # stock red (canon thesis ref)
    "canon_loose230":  "#FCAE91",                # light salmon
    "canon_above":     "#E91E63",                # pink
    "nov_above_stiff": "#F1C40F",                # vivid yellow (with edge for visibility)
}
LABEL = {
    "canon_loose300":  "Under, 30 cm strikk (canon)",
    "canon_loose230":  "Under, 23 cm strikk (canon)",
    "canon_above":     "Over, 16 cm strikk (canon)",
    "nov_above_stiff": "Over, 6 cm strikk + revers (Nov 2025)",
}
MARKERS = {
    "canon_loose300":  "o",
    "canon_loose230":  "s",
    "canon_above":     "D",
    "nov_above_stiff": "P",
}
ORDER = ["canon_above", "nov_above_stiff", "canon_loose230", "canon_loose300"]

fig, ax = plt.subplots(figsize=(7.5, 5.5))

# Reference horizontal line at ε = 0.32 (under-water universal value)
EPS_REF = float(piv[piv["cat"] == "canon_loose300"]["eps"].mean())
ax.axhline(EPS_REF, color="black", lw=0.8, ls=":", alpha=0.55,
           zorder=1, label=None)
ax.text(1.64, EPS_REF + 0.01, rf"$\langle\varepsilon\rangle_{{under}} = {EPS_REF:.2f}$",
        ha="right", va="bottom", fontsize=8, color="#444444", alpha=0.85)

# Zero line
ax.axhline(0.0, color="black", lw=0.5, ls="-", alpha=0.4, zorder=1)

# Vertical band marking the canon_above transition (1.4 ↔ 1.5 Hz)
ax.axvspan(1.40, 1.55, color=PALETTE["canon_above"], alpha=0.08, lw=0, zorder=1)
ax.text(1.475, 0.52, "overgang\ncanon_above",
        ha="center", va="top", fontsize=8, color="#A11860", alpha=0.85,
        bbox=dict(boxstyle="round,pad=0.2",
                  facecolor="white", alpha=0.7, edgecolor="none"))

# Plot each mooring's ε(f) with errorbars + connecting line
for cat in ORDER:
    sub = piv[piv["cat"] == cat].sort_values("WaveFrequencyInput [Hz]")
    if sub.empty: continue
    color = PALETTE[cat]
    edge  = "black" if cat == "nov_above_stiff" else color
    ax.errorbar(
        sub["WaveFrequencyInput [Hz]"], sub["eps"],
        yerr=sub["se_eps"],
        marker=MARKERS[cat], markersize=8,
        markerfacecolor=color, markeredgecolor=edge, markeredgewidth=0.5,
        linestyle="-", linewidth=1.5, color=color, alpha=0.85,
        ecolor=color, elinewidth=1.0, capsize=3, capthick=1.0,
        label=LABEL[cat], zorder=3,
    )

ax.set_xlabel("Bølgefrekvens (Hz)", fontsize=11)
ax.xaxis.set_major_locator(MultipleLocator(0.1))
ax.xaxis.set_minor_locator(MultipleLocator(0.025))
ax.set_xlim(1.15, 1.65)
ax.yaxis.set_major_locator(MultipleLocator(0.1))
ax.yaxis.set_minor_locator(MultipleLocator(0.05))
ax.set_ylim(-0.3, 0.55)
ax.grid(which="major", alpha=0.30, lw=0.6)
ax.grid(which="minor", alpha=0.15, lw=0.4)

ax.set_title(
    r"$\varepsilon(f) = \Delta K_t / (1-K_{t,\mathrm{nowind}})$ — andel av panelblokkering som vinden fjerner",
    fontsize=10.5, loc="left")
apply_horizontal_ylabel(ax, r"$\varepsilon$", fontsize=14)

leg = ax.legend(loc="lower right", fontsize=8, framealpha=0.92,
                title="Forankring", title_fontsize=8.5)

# Note about masking
ax.text(1.16, -0.27,
        r"$\varepsilon$ vises kun for $(1-K_{t,\mathrm{nowind}}) \geq 0.10$",
        ha="left", va="bottom", fontsize=7,
        color="#666666", alpha=0.85,
        bbox=dict(boxstyle="round,pad=0.25",
                  facecolor="white", alpha=0.75, edgecolor="none"))

# Predicted f_n annotation
ax.axvline(F_N_CANON_ABOVE, color=PALETTE["canon_above"], lw=0.8,
           ls="--", alpha=0.5)
ax.text(F_N_CANON_ABOVE + 0.005, -0.18,
        rf"$f_n \approx {F_N_CANON_ABOVE:.2f}$ Hz" + "\n"
        r"($k \approx 400$ N/m, $m=4.55$ kg)",
        ha="left", va="bottom", fontsize=7.5,
        color="#A11860", alpha=0.85,
        bbox=dict(boxstyle="round,pad=0.25",
                  facecolor="white", alpha=0.75, edgecolor="none"))

fig.subplots_adjust(left=0.10, right=0.97, top=0.92, bottom=0.10)
fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.02)
print(f"\n   Saved → {OUT_PDF.relative_to(BASE)}")
plt.close(fig)

# Also dump the ε(f) table to console for quick review
print("\n=== ε(f) per mooring (rounded) ===")
for cat in ORDER:
    sub = piv[piv["cat"] == cat].sort_values("WaveFrequencyInput [Hz]")
    print(f"\n--- {LABEL[cat]} ---")
    print(sub[["WaveFrequencyInput [Hz]", "n_no", "n_full", "Kt_mean_no",
               "Kt_mean_full", "delta_Kt", "blockage_no", "eps", "se_eps"]]
          .round(3).to_string(index=False))

print("\nDone.")
