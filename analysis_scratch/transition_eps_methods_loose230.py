"""
ε(f) for canon_loose230 — three K_t methods overlaid (scratch)
=================================================================

Same ε = ΔK_t / (1 − K_t_nowind) metric as transition_lift_fraction_eps.py,
restricted to canon_loose230 (most data, n ≥ 22+15 at 1.3 Hz). Three lines:

  - solid FFT (paddle-bin amplitude) — robust, filters wind chop
  - dashed percentile (P99.5−P0.5)/2 — captures full time-domain envelope
                                       including wind chop
  - dotted phase-locked (T/4, 3T/4 samples per cycle) — breaks under wind
                                                       due to upcrossing
                                                       contamination on IN

At low freq (1.2–1.4 Hz) the methods strongly diverge: FFT says wind
slightly enhances K_t, percentile says wind reduces it (wind adds noise
envelope to IN), phase-locked produces K_t > 1 artefacts (cycle
detection fails on wind-contaminated IN).

At high freq (1.5+ Hz) all three methods agree on sign.

Use this for the methodology section to motivate FFT-based K_t.

Run from repo root:
    conda run -n draumkvedet python analysis_scratch/transition_eps_methods_loose230.py
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
from wavescripts.plot_utils import apply_thesis_style, apply_horizontal_ylabel

apply_thesis_style()

OUT_PDF = Path(__file__).parent / "transition_eps_methods_loose230.pdf"

# ── 1. Load ───────────────────────────────────────────────────────────────────
print("1. Loading …")
all_dirs = sorted(glob.glob(str(BASE / "waveprocessed" / "PROCESSED-*")))
meta, _, _, _ = load_analysis_data(*all_dirs, load_processed=False)

m = meta[
    meta["WaveFrequencyInput [Hz]"].notna()
    & (meta["WaveFrequencyInput [Hz]"] > 0)
    & (meta["WaveFrequencyInput [Hz]"].between(1.2, 1.6))
    & meta["PanelCondition"].isin(["full", "reverse"])
    & meta["WindCondition"].isin(["no", "full"])
    & (meta["quality_flag"] == "ok")
    & meta["OUT/IN (FFT)"].notna()
    & (meta["OUT/IN (FFT)"].between(0.1, 2.0))
    & (meta["Mooring"] == "below_90_loose230")
    & meta["in_probes_used"].notna()
    & meta["out_probes_used"].notna()
].copy()
m["file_date_dt"] = pd.to_datetime(m["file_date"]).dt.tz_localize(None)
m["cfg"] = m["file_date_dt"].apply(lambda d: get_configuration_for_date(d).name if pd.notnull(d) else "")
m = m[m["cfg"] == "march2026_better_rearranging"].copy()
print(f"   {len(m)} canon_loose230 wave runs in 1.2–1.6 Hz band")

# ── 2. K_t under three methods ────────────────────────────────────────────────
def a_method(row, suffix):
    inp = [p.strip() for p in str(row["in_probes_used"]).split("+")]
    out = [p.strip() for p in str(row["out_probes_used"]).split("+")]
    try:
        ai = np.nanmean([row[f"Probe {p} Amplitude{suffix}"] for p in inp])
        ao = np.nanmean([row[f"Probe {p} Amplitude{suffix}"] for p in out])
        return ai, ao
    except KeyError:
        return np.nan, np.nan

for label, suffix in [("tot", ""), ("fft", " (FFT)"), ("phase", " (phase) mean")]:
    cols = m.apply(lambda r: pd.Series(a_method(r, suffix),
                                       index=[f"A_IN_{label}", f"A_OUT_{label}"]), axis=1)
    m = pd.concat([m, cols], axis=1)
    m[f"Kt_{label}"] = m[f"A_OUT_{label}"] / m[f"A_IN_{label}"]

# Cell aggregation: (freq, wind)
agg = (m.groupby(["WaveFrequencyInput [Hz]", "WindCondition"])
        .agg(n=("Kt_fft", "size"),
             **{f"Kt_{lbl}_mean": (f"Kt_{lbl}", "mean") for lbl in ["tot", "fft", "phase"]},
             **{f"Kt_{lbl}_std":  (f"Kt_{lbl}", "std")  for lbl in ["tot", "fft", "phase"]})
        .reset_index())
piv = agg.pivot_table(index="WaveFrequencyInput [Hz]",
                      columns="WindCondition",
                      values=["n"] +
                             [f"Kt_{lbl}_mean" for lbl in ["tot", "fft", "phase"]] +
                             [f"Kt_{lbl}_std"  for lbl in ["tot", "fft", "phase"]])
piv.columns = [f"{a}_{b}" for a, b in piv.columns]
piv = piv.reset_index().dropna(subset=[f"Kt_{lbl}_mean_full" for lbl in ["fft"]])

# ε per method
for lbl in ["tot", "fft", "phase"]:
    piv[f"d_{lbl}"]    = piv[f"Kt_{lbl}_mean_full"] - piv[f"Kt_{lbl}_mean_no"]
    piv[f"blk_{lbl}"]  = 1.0 - piv[f"Kt_{lbl}_mean_no"]
    piv[f"eps_{lbl}"]  = piv[f"d_{lbl}"] / piv[f"blk_{lbl}"]
    # SE
    def _se(s, n):
        return np.where(np.isfinite(s), s / np.sqrt(np.maximum(n, 1)), np.nan)
    se_full = _se(piv[f"Kt_{lbl}_std_full"], piv["n_full"])
    se_no   = _se(piv[f"Kt_{lbl}_std_no"],   piv["n_no"])
    piv[f"se_eps_{lbl}"] = np.sqrt(
        (se_full / piv[f"blk_{lbl}"])**2
        + (se_no  * (piv[f"Kt_{lbl}_mean_full"] - 1) / piv[f"blk_{lbl}"]**2)**2
    )
    # Mask very-small-blockage cells (only matters for tot since others stay well-blocked)
    piv.loc[piv[f"blk_{lbl}"] < 0.10, f"eps_{lbl}"] = np.nan

print("\nε per method, canon_loose230 1.2–1.6 Hz:")
cols_show = ["WaveFrequencyInput [Hz]", "n_no", "n_full",
             "Kt_fft_mean_no", "Kt_fft_mean_full", "eps_fft",
             "Kt_tot_mean_no", "Kt_tot_mean_full", "eps_tot",
             "Kt_phase_mean_no", "Kt_phase_mean_full", "eps_phase"]
print(piv[cols_show].round(3).to_string(index=False))

# ── 3. Figure ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.5, 5.5))

# Reference horizontal lines
ax.axhline(0.32, color="black", lw=0.8, ls=":", alpha=0.55, zorder=1)
ax.text(1.205, 0.34, r"$\varepsilon \approx 0.32$ (under-FFT universal)",
        ha="left", va="bottom", fontsize=7.5, color="#444444", alpha=0.85)
ax.axhline(0.0, color="black", lw=0.5, ls="-", alpha=0.4, zorder=1)
ax.axhline(1.0, color="black", lw=0.5, ls="-", alpha=0.25, zorder=1)
ax.text(1.205, 1.02, r"$\varepsilon = 1$ — K_t,full = 1 (full unblock)",
        ha="left", va="bottom", fontsize=7.5, color="#666666", alpha=0.85)

# Transition band (panel-resonance shading)
ax.axvspan(1.40, 1.55, color="#E91E63", alpha=0.07, lw=0, zorder=1)

# Three method lines (canon_loose230's salmon as base, vary linestyle)
BASE_COLOR = "#FCAE91"   # canon_loose230 salmon (matches sibling figures)
STYLES = [
    ("eps_fft",   "se_eps_fft",   "FFT (paddle-bin)",       "-",  "o",  9,  2.0),
    ("eps_tot",   "se_eps_tot",   "Percentile (P99.5−P0.5)/2","--", "s",  9,  2.0),
    ("eps_phase", "se_eps_phase", "Phase-locked (T/4, 3T/4)", ":", "D",  9,  2.0),
]
METHOD_COLOR = {
    "eps_fft":   "#C0392B",   # darker red — robust paddle method
    "eps_tot":   "#2980B9",   # blue — total envelope (different physical Q)
    "eps_phase": "#7F8C8D",   # grey — diagnostic / artefact under wind
}

for col, se_col, label, ls, marker, ms, lw in STYLES:
    sub = piv.sort_values("WaveFrequencyInput [Hz]")
    color = METHOD_COLOR[col]
    ax.errorbar(sub["WaveFrequencyInput [Hz]"], sub[col],
                yerr=sub[se_col],
                marker=marker, markersize=ms, markerfacecolor=color,
                markeredgecolor="black", markeredgewidth=0.5,
                linestyle=ls, linewidth=lw, color=color, alpha=0.90,
                ecolor=color, elinewidth=1.0, capsize=3, capthick=1.0,
                label=label, zorder=3)

# Panel-resonance annotation
ax.axvline(1.50, color="#A11860", lw=0.8, ls="--", alpha=0.5)
ax.text(1.51, -0.55, r"$f_n \approx 1.5$ Hz",
        ha="left", va="bottom", fontsize=7.5,
        color="#A11860", alpha=0.85,
        bbox=dict(boxstyle="round,pad=0.25",
                  facecolor="white", alpha=0.75, edgecolor="none"))

# Annotations: highlight K_t_phase > 1 artefact
ax.annotate(r"$K_t^{phase}\!>\!1$ artefakt" + "\n(sykluser i IN ødelegges av vind)",
            xy=(1.30, 1.369), xytext=(1.34, 1.7),
            fontsize=8, color="#555555",
            arrowprops=dict(arrowstyle="->", color="#888888", lw=0.7),
            bbox=dict(boxstyle="round,pad=0.25",
                      facecolor="white", alpha=0.85, edgecolor="#bbbbbb"))
ax.annotate(r"percentil $K_t$ synker:" + "\nvinden legger støy" + "\npå IN, panelet filtrerer ut",
            xy=(1.30, -0.388), xytext=(1.34, -0.95),
            fontsize=8, color="#555555",
            arrowprops=dict(arrowstyle="->", color="#888888", lw=0.7),
            bbox=dict(boxstyle="round,pad=0.25",
                      facecolor="white", alpha=0.85, edgecolor="#bbbbbb"))

ax.set_xlabel("Bølgefrekvens (Hz)", fontsize=11)
ax.xaxis.set_major_locator(MultipleLocator(0.1))
ax.xaxis.set_minor_locator(MultipleLocator(0.025))
ax.set_xlim(1.15, 1.65)
ax.yaxis.set_major_locator(MultipleLocator(0.5))
ax.yaxis.set_minor_locator(MultipleLocator(0.1))
ax.set_ylim(-1.3, 2.8)
ax.grid(which="major", alpha=0.30, lw=0.6)
ax.grid(which="minor", alpha=0.15, lw=0.4)

ax.set_title(r"canon\_loose230: $\varepsilon(f)$ etter målemetode "
             r"— FFT vs persentil vs faselåst",
             fontsize=10.5, loc="left")
apply_horizontal_ylabel(ax, r"$\varepsilon$", fontsize=14)

leg = ax.legend(loc="upper right", fontsize=8, framealpha=0.92,
                title="K$_t$-metode", title_fontsize=8.5)

fig.subplots_adjust(left=0.10, right=0.97, top=0.92, bottom=0.10)
fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.02)
print(f"\nSaved → {OUT_PDF.relative_to(BASE)}")
plt.close(fig)
print("Done.")
