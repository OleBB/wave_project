"""
Parallel-probe agreement diagnostic (CH04 §3b follow-up)
========================================================

Documents why using mean(9373/170, 9373/340) as the IN reference is
trustworthy: the two probes, sitting at the same longitudinal distance
from the paddle, agree to within a few percent across the thesis band
in the conditions that matter for the headline result.

Inspired by the Huseby & Grue window check (2026-04-18) that exposed
a single-probe transient dip on 9373/170 which 9373/340 did not see
(see `analysis_scratch/huseby_grue_window.pdf`). The user's decision
was to make the mean the canonical IN reference everywhere (option
3 in `huseby_grue_window_findings.md`). This figure is the supporting
methodology diagnostic for that decision.

Three panels:

  (a) A(9373/170) vs A(9373/340) scatter — identity line, ±5% and ±10%
      bands. Colour = WindCondition, marker size = amplitude. A clean
      dataset sits tightly on the identity; outliers are flagged.
  (b) Disagreement fraction |A₁ − A₂| / mean(A₁,A₂) vs input voltage,
      split by wind condition. Shows where the disagreement lives.
  (c) Per-(freq, amp, wind) mean disagreement fraction as a heatmap,
      marking cells that exceed the 10% consistency threshold.

Scope: full-panel quality-ok wave runs in meta_results (cond4
lowrange, 2026-03-26/27). Only the thesis frequency scope 1.3–1.6 Hz
is plotted.

Run from repo root:
    conda run -n draumkvedet python analysis_scratch/parallel_probe_agreement.py

Outputs:
    analysis_scratch/parallel_probe_agreement.pdf      (scratch quick-view)
    output/FIGURES/ch04_parallel_probe_agreement.pdf    (thesis figure)
    output/TEXFIGU/ch04_parallel_probe_agreement.tex    (stub, write-once)
    analysis_scratch/parallel_probe_agreement_summary.csv
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import os
os.chdir(BASE)

from wavescripts.improved_data_loader import load_analysis_data
from wavescripts.plot_utils import apply_thesis_style

apply_thesis_style()

# ── I/O ────────────────────────────────────────────────────────────────────────
SCRATCH_PDF = Path(__file__).parent / "parallel_probe_agreement.pdf"
SCRATCH_CSV = Path(__file__).parent / "parallel_probe_agreement_summary.csv"
THESIS_NAME = "ch04_parallel_probe_agreement"
OUT_PDF = BASE / "output" / "FIGURES" / f"{THESIS_NAME}.pdf"
OUT_STUB = BASE / "output" / "TEXFIGU" / f"{THESIS_NAME}.tex"
CHAPTER = "04"

RESULTS_DIRS = [
    Path("waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange"),
    Path("waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"),
]

THRESHOLD = 0.10

# ── 1. Load ─────────────────────────────────────────────────────────────────
# Canonical IN/OUT amplitudes and ain_disagree_frac / aout_disagree_frac
# now live in meta.json directly (pipeline-level as of 2026-04-18). The
# old mean_in_probe hook has been archived.
print("1. Loading …")
meta, _, _, _ = load_analysis_data(*RESULTS_DIRS, load_processed=False)
meta["Mooring"] = meta["Mooring"].replace({
    "below_90_loose230": "below_90_loose",
    "below_90_loose300": "below_90_loose",
})
meta["ain_probe_consistent"] = meta["ain_disagree_frac"] < THRESHOLD

# Scope: thesis band only (1.3–1.6 Hz), full-panel quality-ok wave runs
scope = meta[
    meta["WaveFrequencyInput [Hz]"].notna()
    & (meta["WaveFrequencyInput [Hz]"].round(2).between(1.3, 1.6))
    & (meta["PanelCondition"] == "full")
    & (meta["quality_flag"] == "ok")
    & meta["Probe 9373/170 Amplitude (FFT)"].notna()
    & meta["Probe 9373/340 Amplitude (FFT)"].notna()
].copy()
print(f"   {len(scope)} quality-ok fullpanel wave runs in thesis scope")

# ── 2. Per-run summary ─────────────────────────────────────────────────────────
wave_all = meta[meta["WaveFrequencyInput [Hz]"].notna()].copy()
wave_all["freq_r"] = wave_all["WaveFrequencyInput [Hz]"].round(2)
wave_all["amp_r"]  = wave_all["WaveAmplitudeInput [Volt]"].round(2)
summary = (wave_all.groupby(["amp_r", "freq_r", "WindCondition"])
                   .agg(n=("ain_disagree_frac", "size"),
                        disagree_frac_mean=("ain_disagree_frac", "mean"),
                        disagree_frac_max=("ain_disagree_frac", "max"),
                        n_inconsistent=("ain_probe_consistent",
                                        lambda s: int((~s).sum())))
                   .reset_index())
summary.to_csv(SCRATCH_CSV, index=False)
print(f"   Summary → {SCRATCH_CSV.relative_to(BASE)}")

n_total = len(scope)
n_consistent = int(scope["ain_probe_consistent"].sum())
print(f"\n   Thesis scope: {n_consistent}/{n_total} runs consistent at {THRESHOLD*100:.0f}%")
print(f"   median disagree: {scope['ain_disagree_frac'].median():.3f}  "
      f"max: {scope['ain_disagree_frac'].max():.3f}")

# ── 3. Plot ───────────────────────────────────────────────────────────────────
print("\n2. Plotting …")

WIND_COLOR = {"no": "#2980B9", "full": "#E74C3C"}
AMP_SIZE   = {0.1: 22, 0.2: 55, 0.3: 100}

fig = plt.figure(figsize=(16, 5.4))
gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 1],
                       wspace=0.30, left=0.06, right=0.99,
                       top=0.85, bottom=0.18)

# ── Panel (a): Scatter A_170 vs A_340 ─────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
a170 = scope["Probe 9373/170 Amplitude (FFT)"].to_numpy()
a340 = scope["Probe 9373/340 Amplitude (FFT)"].to_numpy()
lim = max(float(np.nanmax(a170)), float(np.nanmax(a340))) * 1.08
# Identity + ±5/10% bands
xs = np.linspace(0, lim, 50)
ax_a.plot(xs, xs, "k--", lw=0.8, alpha=0.6, zorder=1, label="identity")
ax_a.fill_between(xs, xs * 0.95, xs * 1.05, color="#2ECC71", alpha=0.10,
                  zorder=0, label="±5%")
ax_a.fill_between(xs, xs * 0.90, xs * 1.10, color="#F1C40F", alpha=0.08,
                  zorder=0, label="±10%")
for wind, grp in scope.groupby("WindCondition"):
    sizes = grp["WaveAmplitudeInput [Volt]"].map(
        lambda a: AMP_SIZE.get(round(float(a), 1), 30)
    )
    ax_a.scatter(grp["Probe 9373/170 Amplitude (FFT)"],
                 grp["Probe 9373/340 Amplitude (FFT)"],
                 s=sizes, color=WIND_COLOR.get(wind, "gray"),
                 edgecolor="black", linewidth=0.4,
                 alpha=0.75, label=f"{wind} wind", zorder=3)
ax_a.set_xlabel("A(9373/170)  [mm, FFT]", fontsize=9)
ax_a.set_ylabel("A(9373/340)  [mm, FFT]", fontsize=9)
ax_a.set_title("", fontsize=10, fontweight="bold")
ax_a.set_xlim(0, lim); ax_a.set_ylim(0, lim)
ax_a.set_aspect("equal", adjustable="box")
ax_a.grid(True, alpha=0.3)
ax_a.legend(fontsize=7, loc="lower right", framealpha=0.9)

# ── Panel (b): Disagreement vs input amplitude ────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])
for wind, grp in scope.groupby("WindCondition"):
    x = grp["WaveAmplitudeInput [Volt]"].to_numpy(dtype=float)
    y = grp["ain_disagree_frac"].to_numpy(dtype=float) * 100.0
    jx = x + (0.008 if wind == "no" else -0.008)   # slight offset for clarity
    ax_b.scatter(jx, y, color=WIND_COLOR.get(wind, "gray"),
                 s=40, alpha=0.75, edgecolor="black", linewidth=0.3,
                 label=f"{wind} wind")
ax_b.axhline(THRESHOLD * 100, color="#F1C40F", lw=1.0, ls="--",
             label=f"{THRESHOLD*100:.0f}% consistency threshold")
ax_b.set_xlabel("input amplitude [V]", fontsize=9)
ax_b.set_ylabel("disagreement  |A₁−A₂| / mean(A)  [%]", fontsize=9)
ax_b.set_title("", fontsize=10, fontweight="bold")
ax_b.set_xticks([0.1, 0.2, 0.3])
ax_b.grid(True, alpha=0.3)
ax_b.legend(fontsize=7, loc="upper right", framealpha=0.9)

# ── Panel (c): Per-(freq, amp, wind) mean disagreement heatmap ────────────────
ax_c = fig.add_subplot(gs[0, 2])
scope["freq_r"] = scope["WaveFrequencyInput [Hz]"].round(2)
scope["amp_r"] = scope["WaveAmplitudeInput [Volt]"].round(2)
# Build a (freq, amp×wind) grid — mean disagreement
freqs = [1.3, 1.4, 1.5, 1.6]
amps  = [0.1, 0.2, 0.3]
winds = ["no", "full"]
mat = np.full((len(freqs), len(amps) * len(winds)), np.nan)
col_labels = []
for j, amp in enumerate(amps):
    for k, wind in enumerate(winds):
        col_labels.append(f"{amp:.1f}V\n{wind}")
        col_idx = j * len(winds) + k
        for i, f in enumerate(freqs):
            sub = scope[(scope["freq_r"] == f) & (scope["amp_r"] == amp)
                        & (scope["WindCondition"] == wind)]
            if len(sub) > 0:
                mat[i, col_idx] = sub["ain_disagree_frac"].mean() * 100.0
im = ax_c.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=20,
                 origin="upper")
cb = plt.colorbar(im, ax=ax_c, fraction=0.05, pad=0.04)
cb.set_label("disagreement [%]", fontsize=8)
ax_c.set_xticks(range(len(col_labels)))
ax_c.set_xticklabels(col_labels, fontsize=7)
ax_c.set_yticks(range(len(freqs)))
ax_c.set_yticklabels([f"{f:.1f} Hz" for f in freqs], fontsize=8)
ax_c.set_title("", fontsize=10, fontweight="bold")
for i in range(len(freqs)):
    for j in range(len(col_labels)):
        v = mat[i, j]
        if np.isfinite(v):
            color = "black" if v < 10 else "white"
            marker = "⚠" if v > THRESHOLD * 100 else ""
            ax_c.text(j, i, f"{v:.1f}{marker}", ha="center", va="center",
                      fontsize=7, color=color, fontweight="bold" if marker else "normal")

fig.suptitle("", fontsize=11, fontweight="bold", y=0.95)
fig.text(0.5, 0.02,
         f"Scope: {n_total} quality-ok full-panel wave runs, 1.3–1.6 Hz, cond4 lowrange. "
         f"{n_consistent}/{n_total} ({100*n_consistent/n_total:.0f}%) consistent at "
         f"{THRESHOLD*100:.0f}% threshold. "
         "Disagreement is concentrated in 0.1 V fullwind — the low-SNR regime where "
         "wind contamination affects each probe differently.",
         ha="center", fontsize=8, color="#444", style="italic")

# ── Save ──────────────────────────────────────────────────────────────────────
SCRATCH_PDF.parent.mkdir(parents=True, exist_ok=True)
OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
OUT_STUB.parent.mkdir(parents=True, exist_ok=True)

fig.savefig(SCRATCH_PDF, bbox_inches="tight")
print(f"   Saved → {SCRATCH_PDF.relative_to(BASE)}")
fig.savefig(OUT_PDF, bbox_inches="tight")
print(f"   Saved → {OUT_PDF.relative_to(BASE)}")

# Stub
if not OUT_STUB.exists():
    _caption = (
        f"Parallel-probe cross-check for the canonical IN reference. "
        f"9373/170 and 9373/340 sit at the same longitudinal distance "
        f"from the paddle; in a 1D-wave approximation they should see "
        f"the same incident amplitude. "
        f"(a) A(9373/170) vs A(9373/340), identity line and $\\pm 5$/$\\pm 10$\\% "
        f"bands. Colour: wind; marker size: input amplitude. "
        f"(b) Fractional disagreement versus input voltage, split by wind. "
        f"The $10\\%$ consistency threshold is drawn; disagreement is "
        f"concentrated at $0.1$\\,V fullwind (low SNR, wind contamination "
        f"dominates the IN probe FFT). "
        f"(c) Heatmap of mean disagreement per (frequency, amplitude, wind) "
        f"across the thesis band. Across all {n_total} quality-ok full-panel "
        f"wave runs in the thesis scope (1.3--1.6\\,Hz, cond4 lowrange), "
        f"${n_consistent}/{n_total}$ "
        f"($\\sim{100*n_consistent/n_total:.0f}\\%$) are within the $10\\%$ "
        f"threshold. This validates replacing the single-probe IN reference "
        f"with the mean of both 9373 probes as the canonical IN amplitude "
        f"in all CH05 figures."
    )
    _stub = (
        "%! TEX root = ../main.tex\n"
        "% =============================================================\n"
        "% IMMUTABLE — generated automatically, do not edit this block\n"
        "%   script          : analysis_scratch/parallel_probe_agreement.py\n"
        "%   plot_type       : parallel_probe_agreement\n"
        f"%   chapter         : {CHAPTER}\n"
        f"%   threshold       : {THRESHOLD*100:.0f}%\n"
        f"%   n_runs          : {n_total}\n"
        f"%   n_consistent    : {n_consistent}\n"
        "% =============================================================\n"
        "\\begin{figure}[htbp]\n"
        "  \\centering\n"
        f"  \\includegraphics[width=0.98\\linewidth]{{FIGURES/{THESIS_NAME}.pdf}}\n"
        "  \\caption[Parallel-probe agreement]{%\n"
        f"    {_caption}\n"
        "  }\n"
        f"  \\label{{fig:{THESIS_NAME}}}\n"
        "\\end{figure}\n"
    )
    OUT_STUB.write_text(_stub)
    print(f"   Wrote stub → {OUT_STUB.relative_to(BASE)}")
else:
    print(f"   Stub exists (not overwritten): {OUT_STUB.relative_to(BASE)}")

print("\nDone.")
