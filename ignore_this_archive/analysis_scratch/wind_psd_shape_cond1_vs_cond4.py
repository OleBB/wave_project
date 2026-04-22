"""
Wind PSD shape: cond1 (h272/high) vs cond4 (h100/low)
=======================================================

Open question from `probe_height_wind_findings.md` (Finding 2):

  > The physical wind-wave field is approximately constant across
  > sessions … Amplitude differences across conditions at the IN probe
  > reflect probe-measurement properties, not wind-field differences.

This script tests that claim spectrally. If the wind-wave field is truly
constant and conditions only differ in how the probe reports its
amplitude, then the PSD *shape* should be the same across cond1 and
cond4 — just scaled. If the shape differs, the probe reports the wind
differently (frequency-dependent attenuation, range-mode rolloff, etc.)
and the conclusion needs nuance.

Method:
  - Load all nowave+fullwind PSDs from a curated cond1 set (h272/high,
    pre-2026-03-23) and the cond4 set (h100/low, 2026-03-26/27 lowrange).
  - For each probe, compute the per-condition mean PSD (Welch already
    applied during pipeline processing).
  - Compare:
      (i) overlaid mean PSDs (log y),
      (ii) ratio cond1/cond4 vs frequency.
  - Pure amplitude scaling → ratio is flat across f. Shape difference →
    ratio has frequency-dependent structure.

Run from repo root:
    conda run -n draumkvedet python analysis_scratch/wind_psd_shape_cond1_vs_cond4.py

Outputs:
    analysis_scratch/wind_psd_shape_cond1_vs_cond4.pdf
    analysis_scratch/wind_psd_shape_cond1_vs_cond4_findings.md
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

# ── I/O ────────────────────────────────────────────────────────────────────────
OUT_PDF = Path(__file__).parent / "wind_psd_shape_cond1_vs_cond4.pdf"
OUT_MD = Path(__file__).parent / "wind_psd_shape_cond1_vs_cond4_findings.md"

# ── Datasets ──────────────────────────────────────────────────────────────────
# cond1 (h272/high, pre-2026-03-23): same probe positions as cond4, just a
# different probe-tip height + range mode. Pick folders with the most
# nowave+fullwind runs (so the mean PSD has support).
COND1_DIRS = [
    Path("waveprocessed/PROCESSED-20260307-ProbPos4_31_FPV_2-tett6roof"),
    Path("waveprocessed/PROCESSED-20260312-ProbPos4_31_FPV_2-tett6roof"),
    Path("waveprocessed/PROCESSED-20260319-ProbePos4_31_FPV_2-tett6roof-under9Mooring"),
]
# cond4 (h100/low): the two thesis-result folders
COND4_DIRS = [
    Path("waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange"),
    Path("waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"),
]

PROBES = ["9373/170", "12400/250", "9373/340", "8804/250"]
PROBE_LABELS = {
    "9373/170": "9373/170 (IN)",
    "12400/250": "12400/250 (OUT)",
    "9373/340": "9373/340 (parallel)",
    "8804/250": "8804/250 (upstream)",
}

# Frequency range to display. Wind waves dominate ~2–6 Hz; below 1 Hz the
# PSD tail interacts with the paddle band. 10 Hz upper cap drops the noise
# floor + alias regime.
F_MIN_DISPLAY = 0.2
F_MAX_DISPLAY = 10.0
# Ratio plot: focus where there's signal, suppress tail noise
F_MIN_RATIO = 0.5
F_MAX_RATIO = 8.0

# ── 1. Load ───────────────────────────────────────────────────────────────────
print("1. Loading cond1 + cond4 metadata + PSDs …")
all_dirs = COND1_DIRS + COND4_DIRS
meta, _, _, psd_dict = load_analysis_data(*all_dirs, load_processed=False)
print(f"   {len(meta)} rows, {len(psd_dict)} PSD spectra")

# Mark which condition each row belongs to (by folder substring match).
def assign_set(path):
    p = str(path)
    if any(d.name.removeprefix("PROCESSED-") in p for d in COND1_DIRS):
        return "cond1"
    if any(d.name.removeprefix("PROCESSED-") in p for d in COND4_DIRS):
        return "cond4"
    return "other"

meta["cond_set"] = meta["path"].apply(assign_set)

nw = meta[
    meta["WaveFrequencyInput [Hz]"].isna()
    & (meta["WindCondition"] == "full")
    & (meta["cond_set"].isin(["cond1", "cond4"]))
].copy()
print(f"   nowave+fullwind: {len(nw)} rows")
print(nw.groupby("cond_set").size().to_string())

# ── 2. Mean PSDs per probe per cond_set ───────────────────────────────────────
print("\n2. Computing per-condition mean PSDs per probe …")

def collect_psds(rows: pd.DataFrame, probe: str):
    """Stack PSDs into a (n_runs, n_freqs) array. Drop runs missing the column."""
    arrs = []
    freqs_ref = None
    for _, r in rows.iterrows():
        df = psd_dict.get(r["path"])
        if df is None:
            continue
        col = f"Pxx {probe}"
        if col not in df.columns:
            continue
        s = df[col]
        if freqs_ref is None:
            freqs_ref = s.index.values
        elif len(s.index) != len(freqs_ref) or not np.allclose(s.index.values, freqs_ref):
            # Reindex if frequency grids differ (rare — pipeline usually fixed grid)
            s = s.reindex(freqs_ref)
        arrs.append(s.values)
    if not arrs:
        return None, None
    return freqs_ref, np.array(arrs)

# Store {probe: {cond: (freqs, psd_matrix)}}
data = {}
for probe in PROBES:
    data[probe] = {}
    for cs in ("cond1", "cond4"):
        rows = nw[nw["cond_set"] == cs]
        freqs, mat = collect_psds(rows, probe)
        data[probe][cs] = (freqs, mat)
        n = mat.shape[0] if mat is not None else 0
        print(f"   {probe} | {cs} | n_runs = {n}")

# ── 3. Build summary table ────────────────────────────────────────────────────
print("\n3. Summary statistics (per probe × condition)")

def integrate_psd(freqs, mat, fmin, fmax):
    """Trapezoidal integral over [fmin, fmax]; returns one value per row."""
    mask = (freqs >= fmin) & (freqs <= fmax)
    if mask.sum() < 2:
        return np.full(mat.shape[0], np.nan)
    return np.trapezoid(mat[:, mask], freqs[mask], axis=1)

# Bands: paddle (0.5–1.8 Hz), wind (2–6 Hz), tail (6–10 Hz)
band_defs = {
    "paddle (0.5-1.8)": (0.5, 1.8),
    "wind (2-6)":       (2.0, 6.0),
    "tail (6-10)":      (6.0, 10.0),
}
summary_rows = []
for probe in PROBES:
    for cs in ("cond1", "cond4"):
        freqs, mat = data[probe][cs]
        if mat is None:
            continue
        row = {"probe": probe, "condition": cs, "n_runs": mat.shape[0]}
        for band_name, (fmin, fmax) in band_defs.items():
            band_int = integrate_psd(freqs, mat, fmin, fmax)
            row[f"{band_name}_mean"] = float(np.nanmean(band_int))
            row[f"{band_name}_std"]  = float(np.nanstd(band_int))
        # Peak frequency and value (mean across runs)
        mean_psd = np.nanmean(mat, axis=0)
        wind_mask = (freqs >= 1.0) & (freqs <= 8.0)
        peak_idx = np.argmax(mean_psd[wind_mask])
        row["peak_freq_hz"]  = float(freqs[wind_mask][peak_idx])
        row["peak_psd_mm2_per_hz"] = float(mean_psd[wind_mask][peak_idx])
        summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
print(summary_df.round(3).to_string(index=False))

# ── 4. Plot ───────────────────────────────────────────────────────────────────
print("\n4. Plotting …")

COND_COLORS = {"cond1": "#2ECC71", "cond4": "#3498DB"}
COND_LABELS = {
    "cond1": "cond1 h272/high",
    "cond4": "cond4 h100/low",
}

n_probes = len(PROBES)
fig = plt.figure(figsize=(16, 7.2))
gs = gridspec.GridSpec(2, n_probes, figure=fig, hspace=0.35, wspace=0.25,
                       left=0.05, right=0.99, top=0.92, bottom=0.10,
                       height_ratios=[2.2, 1.0])

for j, probe in enumerate(PROBES):
    ax_psd = fig.add_subplot(gs[0, j])
    ax_rat = fig.add_subplot(gs[1, j], sharex=ax_psd)

    # Top: mean PSD per condition (log y)
    for cs in ("cond1", "cond4"):
        freqs, mat = data[probe][cs]
        if mat is None:
            ax_psd.text(0.5, 0.5, f"no {cs}", transform=ax_psd.transAxes, ha="center")
            continue
        mean_psd = np.nanmean(mat, axis=0)
        std_psd = np.nanstd(mat, axis=0)
        mask = (freqs >= F_MIN_DISPLAY) & (freqs <= F_MAX_DISPLAY)
        f, m, s = freqs[mask], mean_psd[mask], std_psd[mask]
        ax_psd.plot(f, m, color=COND_COLORS[cs], lw=1.4, label=f"{COND_LABELS[cs]} (n={mat.shape[0]})")
        ax_psd.fill_between(f, np.maximum(m - s, 1e-12), m + s,
                            color=COND_COLORS[cs], alpha=0.18, lw=0)
    ax_psd.set_yscale("log")
    ax_psd.set_xlim(F_MIN_DISPLAY, F_MAX_DISPLAY)
    ax_psd.set_xlabel("Frequency [Hz]", fontsize=8)
    ax_psd.set_ylabel("PSD [mm²/Hz]", fontsize=8)
    ax_psd.set_title(PROBE_LABELS[probe], fontsize=10, fontweight="bold")
    ax_psd.tick_params(labelsize=7)
    ax_psd.grid(True, which="both", alpha=0.25, lw=0.4)
    ax_psd.legend(fontsize=7, loc="upper right", framealpha=0.92)
    # Annotate the wind-wave band
    ax_psd.axvspan(2.0, 6.0, color="orange", alpha=0.07, lw=0, zorder=0)

    # Bottom: ratio cond1/cond4, linear
    f1, m1 = data[probe]["cond1"]
    f4, m4 = data[probe]["cond4"]
    if m1 is not None and m4 is not None:
        # Use cond1's frequency grid (should match cond4 — pipeline standard)
        if not np.allclose(f1, f4):
            print(f"   [WARN] {probe}: freq grids differ between cond1/cond4")
        mean1 = np.nanmean(m1, axis=0)
        mean4 = np.nanmean(m4, axis=0)
        # Mask to the ratio range and where both have appreciable signal
        mask = (f1 >= F_MIN_RATIO) & (f1 <= F_MAX_RATIO)
        # Suppress points where cond4 PSD is below 1% of its peak (noise)
        cond4_norm = mean4 / (np.nanmax(mean4) + 1e-30)
        mask = mask & (cond4_norm >= 0.01)
        if mask.sum() > 1:
            ratio = mean1[mask] / (mean4[mask] + 1e-30)
            ax_rat.plot(f1[mask], ratio, color="#444", lw=1.0)
            ax_rat.axhline(1.0, color="black", lw=0.5, ls="--", alpha=0.5)
            # Mean ratio over the wind band as the headline number
            wb = (f1[mask] >= 2.0) & (f1[mask] <= 6.0)
            if wb.sum() > 0:
                ratio_mean = float(np.nanmean(ratio[wb]))
                ax_rat.axhspan(ratio_mean * 0.9, ratio_mean * 1.1, color="orange", alpha=0.15, lw=0)
                ax_rat.text(0.98, 0.92,
                            f"⟨c1/c4⟩₂₋₆Hz = {ratio_mean:.2f}",
                            transform=ax_rat.transAxes, ha="right", va="top",
                            fontsize=7.5, color="#222",
                            bbox=dict(boxstyle="round,pad=0.2",
                                      facecolor="white", alpha=0.8, edgecolor="none"))
    ax_rat.set_xlim(F_MIN_DISPLAY, F_MAX_DISPLAY)
    ax_rat.set_xlabel("Frequency [Hz]", fontsize=8)
    ax_rat.set_ylabel("PSD ratio\ncond1/cond4", fontsize=7.5)
    ax_rat.tick_params(labelsize=7)
    ax_rat.grid(True, alpha=0.25, lw=0.4)
    ax_rat.set_ylim(0, 4)

fig.suptitle(
    "Wind PSD shape: cond1 (h272/high) vs cond4 (h100/low) — same probes, "
    "same wind, different hardware configuration",
    fontsize=11, fontweight="bold", y=0.98,
)
fig.text(0.5, 0.005,
         "Top: mean PSD per condition (band: 2–6 Hz wind-wave region shaded). "
         "Bottom: ratio cond1/cond4 — flat ⟹ pure amplitude scaling, "
         "structured ⟹ shape difference (probe-frequency-dependent response).",
         ha="center", fontsize=7.5, color="#444", style="italic")

OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PDF, bbox_inches="tight")
print(f"   Saved → {OUT_PDF.relative_to(BASE)}")

# ── 5. Findings markdown ──────────────────────────────────────────────────────
print("\n5. Writing findings markdown …")

# For each probe, compute the headline ratio
def headline_ratio(probe):
    f1, m1 = data[probe]["cond1"]
    f4, m4 = data[probe]["cond4"]
    if m1 is None or m4 is None:
        return np.nan, np.nan
    mean1 = np.nanmean(m1, axis=0)
    mean4 = np.nanmean(m4, axis=0)
    wb = (f1 >= 2.0) & (f1 <= 6.0)
    ratio = mean1[wb] / (mean4[wb] + 1e-30)
    return float(np.nanmean(ratio)), float(np.nanstd(ratio))

table_rows = []
for probe in PROBES:
    rmean, rstd = headline_ratio(probe)
    summary_c1 = summary_df[(summary_df["probe"] == probe) & (summary_df["condition"] == "cond1")]
    summary_c4 = summary_df[(summary_df["probe"] == probe) & (summary_df["condition"] == "cond4")]
    p1 = float(summary_c1["peak_freq_hz"].iloc[0]) if len(summary_c1) else np.nan
    p4 = float(summary_c4["peak_freq_hz"].iloc[0]) if len(summary_c4) else np.nan
    table_rows.append((probe, rmean, rstd, p1, p4))

ratio_table = "\n".join(
    f"| {p} | {rm:.2f} ± {rs:.2f} | {p1:.2f} | {p4:.2f} |"
    for p, rm, rs, p1, p4 in table_rows
)

interpretation = []
for p, rm, rs, p1, p4 in table_rows:
    interp = []
    if rm > 1.15:
        interp.append(f"cond1 reports **higher** wind PSD ({rm:.2f}× cond4)")
    elif rm < 0.85:
        interp.append(f"cond1 reports **lower** wind PSD ({rm:.2f}× cond4)")
    else:
        interp.append(f"PSDs match in amplitude (ratio {rm:.2f})")
    if rs / max(rm, 1e-9) > 0.30:
        interp.append("with substantial frequency-dependent structure (ratio std/mean > 30%)")
    else:
        interp.append("approximately flat across the wind band")
    df_peak = abs(p1 - p4)
    if df_peak > 0.3:
        interp.append(f"and peak frequency differs by {df_peak:.2f} Hz")
    interpretation.append(f"- **{p}**: " + ", ".join(interp) + ".")

n_c1 = len(nw[nw["cond_set"] == "cond1"])
n_c4 = len(nw[nw["cond_set"] == "cond4"])

md = f"""# Wind PSD shape: cond1 (h272/high) vs cond4 (h100/low)

**Date**: 2026-04-18 (free-time exploration follow-up to today's ch04 §3b figure)
**Script**: `analysis_scratch/wind_psd_shape_cond1_vs_cond4.py`
**Figure**: `analysis_scratch/wind_psd_shape_cond1_vs_cond4.pdf`

## Question

`probe_height_wind_findings.md` Finding 2 claims:

> The physical wind-wave field is approximately constant across sessions
> … Amplitude differences across conditions at the IN probe reflect
> probe-measurement properties, not wind-field differences.

If this is true, the wind PSD *shape* should match between cond1 and
cond4 — only the amplitude scale differs. If the shape differs, the
probe at h272/high reports the wind differently from the probe at
h100/low (frequency-dependent response, range-mode rolloff, …) and the
"probe-measurement effect" framing is correct but more nuanced than just
a single scaling factor.

## Data

- cond1 nowave+fullwind runs: **{n_c1}** (folders: 20260307, 20260312, 20260319)
- cond4 nowave+fullwind runs: **{n_c4}** (folders: 20260326-lowrange, 20260327)
- Frequency grid: 0.0–125.0 Hz, n=2049 (Welch from pipeline)
- Same physical probes throughout — same probe positions in cond1 and cond4

## Headline result — ratio of mean PSDs over 2–6 Hz wind band

| Probe | ⟨cond1/cond4⟩ ± std | cond1 peak (Hz) | cond4 peak (Hz) |
|-------|---------------------|-----------------|-----------------|
{ratio_table}

A flat ratio across the wind band would mean "pure amplitude scaling"
(probe height changes the gain, not the spectral response). A structured
ratio means "frequency-dependent probe response" (the probe at h272 sees
the wind PSD with a different transfer function than at h100).

## Interpretation per probe

{chr(10).join(interpretation)}

## Caveats

- **Sample size is small** ({n_c1} cond1 runs, {n_c4} cond4 runs). Ratio
  uncertainty is dominated by between-run variability of the wind PSD
  itself, not by within-run noise. Bigger samples would tighten the ratio
  estimates.
- **Mooring length differs** between the cond1 and cond4 sets (cond1
  uses early-Mar mooring, cond4 uses under9Mooring(30) loose230/loose300).
  The OUT probe wind background depends on mooring length via the
  post-panel fetch (see `physics_wavetank_mooring_fetch.md`), so OUT-probe
  ratios mix probe-height effects and mooring-fetch effects.
- **Tank temperature / ambient conditions** vary day-to-day; not
  controlled for here.
- **No paddle-frequency contribution** since these are nowave runs. The
  PSD is purely wind-wave + noise floor.

## Bottom line

See per-probe interpretation above. For the IN-side probes (9373/170,
9373/340, 8804/250), if the ratio is approximately flat ≈ 1.0–1.2, the
"wind field constant, probe height changes amplitude" framing holds. A
ratio significantly different from 1 with frequency-dependent structure
would mean the cond1 vs cond4 difference at the IN probe is not just an
amplitude scaling — the probe geometry alters spectral sensitivity too.

The OUT probe ratio is harder to interpret because of the confounding
mooring-fetch mechanism — different mooring lengths between the cond1
and cond4 datasets change the post-panel fetch even when wind and probe
are identical.

## Next steps (if the result motivates a deeper look)

1. Subset cond1 and cond4 to matched mooring (e.g. both `above_50` or
   both `loose230`) to isolate probe-height from mooring-fetch effects.
2. Compute coherence between cond1 and cond4 spectra at the IN probe to
   test the *shape* claim more rigorously than the simple ratio.
3. Check whether cond3 (h100/high WRONG) shows the same shape as cond4
   (same height, different range mode) — would isolate the range-mode
   contribution.
"""
OUT_MD.write_text(md, encoding="utf-8")
print(f"   Saved → {OUT_MD.relative_to(BASE)}")

print("\nDone.")
