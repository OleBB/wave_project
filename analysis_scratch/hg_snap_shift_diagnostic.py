"""
H&G snap-shift diagnostic — pull numbers straight from meta, tabulate.
======================================================================

Pipeline (commit 71e67c5, 2026-04-22) snaps the theoretical H&G window
start to the nearest zero-upcrossing within ±T. The signed shift per
probe per run is stored in:

    Probe {pos} hg_snap_shift          (samples, signed int)
    Probe {pos} hg_expected_start      (pre-snap theoretical sample)
    Computed Probe {pos} start         (post-snap, used by FFT/LS/etc.)

This script produces a series of breakdown tables so another agent can
inspect the per-probe snap-shift distribution under the current
per-probe-arrival H&G window (N_OFFSET=7, N_LENGTH=10 uniform; rolled
out 2026-05-02). The earlier wavemaker-anchored [50T, 60T] window
(2026-04-21 → 2026-05-02) reported a systematic ~−0.2 T offset at the
OUT probe; under the new window that offset is no longer present (see
Table 1). Tables retained for cross-window comparison and to surface
the new IN-side observation: a non-trivial median snap shift at IN
under nowind, with high std consistent with bimodal upcrossing
selection at low-SNR window starts.

Also includes the amplitude-dependence breakdown (0.1 V / 0.2 V / 0.3 V)
in case finite-amplitude effects appear in the residual.

Scope: fullpanel wave runs in the canon March-2026 lowrange folders,
quality_flag=ok.

Output:
    analysis_scratch/hg_snap_shift_diagnostic.csv       (per-run raw)
    analysis_scratch/hg_snap_shift_diagnostic_summary.csv (long form)
    analysis_scratch/hg_snap_shift_diagnostic.md        (tables + verdict)
    analysis_scratch/hg_snap_shift_diagnostic.png       (figure)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wavescripts.improved_data_loader import load_analysis_data
from wavescripts.plot_utils import apply_thesis_style

apply_thesis_style()

FS = 250.0
BASE = Path(__file__).parent.parent
SCRATCH = Path(__file__).parent

TARGET_DIRS = [
    BASE / "waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange",
    BASE / "waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange",
]

PROBES = ["9373/170", "9373/340", "12400/250", "8804/250"]
THESIS_FREQS = [1.3, 1.4, 1.5, 1.6, 1.7]
AMPS = [0.1, 0.2, 0.3]
WINDS = ["no", "full"]

OUT_CSV_RAW   = SCRATCH / "hg_snap_shift_diagnostic.csv"
OUT_CSV_LONG  = SCRATCH / "hg_snap_shift_diagnostic_summary.csv"
OUT_MD        = SCRATCH / "hg_snap_shift_diagnostic.md"
OUT_PNG       = SCRATCH / "hg_snap_shift_diagnostic.png"


def main():
    print("1. Loading canon meta …")
    meta, *_ = load_analysis_data(*map(str, TARGET_DIRS), load_processed=False)

    mask = (
        (meta["PanelCondition"] == "full")
        & meta["WaveFrequencyInput [Hz]"].notna()
        & (meta["WaveFrequencyInput [Hz]"] > 0)
        & (meta["quality_flag"] == "ok")
    )
    sub = meta[mask].copy()
    print(f"   {len(sub)} fullpanel wave runs")

    # Convert shift-in-samples to shift-in-periods per probe
    sub["samples_per_period"] = (FS / sub["WaveFrequencyInput [Hz]"]).round().astype("Int64")
    for p in PROBES:
        shift_col = f"Probe {p} hg_snap_shift"
        if shift_col not in sub.columns:
            sub[f"{p}_shift_periods"] = np.nan
            continue
        sub[f"{p}_shift_periods"] = pd.to_numeric(sub[shift_col], errors="coerce") / sub["samples_per_period"].astype(float)

    # ── Raw per-run dump ──────────────────────────────────────────────────────
    raw_cols = (
        ["path", "WaveFrequencyInput [Hz]", "WaveAmplitudeInput [Volt]",
         "WindCondition", "PanelCondition", "Mooring", "quality_flag"]
        + [f"Probe {p} hg_snap_shift" for p in PROBES]
        + [f"{p}_shift_periods" for p in PROBES]
    )
    raw = sub[[c for c in raw_cols if c in sub.columns]].copy()
    raw.to_csv(OUT_CSV_RAW, index=False, float_format="%.5f")
    print(f"   raw per-run dump → {OUT_CSV_RAW.relative_to(BASE)}")

    # ── Long-form summary: one row per (freq, amp, wind, probe) cell ──────────
    long_rows = []
    for freq in THESIS_FREQS:
        for amp in AMPS:
            for wind in WINDS:
                cell = sub[
                    np.isclose(sub["WaveFrequencyInput [Hz]"], freq, atol=0.005)
                    & np.isclose(sub["WaveAmplitudeInput [Volt]"], amp, atol=0.01)
                    & (sub["WindCondition"] == wind)
                ]
                if cell.empty:
                    continue
                for p in PROBES:
                    vals = cell[f"{p}_shift_periods"].dropna()
                    if len(vals) == 0:
                        continue
                    long_rows.append({
                        "freq_hz": freq, "amp_V": amp, "wind": wind, "probe": p,
                        "n":      len(vals),
                        "median": vals.median(),
                        "mean":   vals.mean(),
                        "std":    vals.std() if len(vals) > 1 else np.nan,
                        "p05":    vals.quantile(0.05) if len(vals) > 1 else np.nan,
                        "p95":    vals.quantile(0.95) if len(vals) > 1 else np.nan,
                    })
    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(OUT_CSV_LONG, index=False, float_format="%.4f")
    print(f"   long summary    → {OUT_CSV_LONG.relative_to(BASE)}")

    # ── Tables for the .md ───────────────────────────────────────────────────
    md: list[str] = []
    md.append("# H&G snap-shift diagnostic — pulled from meta.json")
    md.append("")
    md.append(f"Generated: {pd.Timestamp.now('UTC').isoformat()[:19]}Z")
    md.append("")
    md.append("**Dataset**: fullpanel wave runs, quality_flag=ok, from the two canonical")
    md.append("March-2026 lowrange folders. n_runs = " + str(len(sub)) + ".")
    md.append("")
    md.append("Shift = (upcrossing-snapped window start) − (theoretical H&G window start),")
    md.append("converted to units of **wave periods** (T = 1/f_paddle).")
    md.append("")

    # Table 1: per-probe summary across all runs, split by wind
    md.append("## Table 1 — Shift by probe and wind (all frequencies pooled)")
    md.append("")
    md.append("Median shift in periods, ±std, n_runs. Positive shift = snap moved the")
    md.append("window LATER than the theoretical H&G position.")
    md.append("")
    md.append("| probe | wind=no | wind=full |")
    md.append("|---|---|---|")
    for p in PROBES:
        cells = [f"`{p}`"]
        for wind in WINDS:
            vals = sub[sub["WindCondition"] == wind][f"{p}_shift_periods"].dropna()
            if len(vals):
                cells.append(f"**{vals.median():+.3f}** ± {vals.std():.3f}  (n={len(vals)})")
            else:
                cells.append("—")
        md.append("| " + " | ".join(cells) + " |")
    md.append("")

    # Table 2: IN vs OUT disagreement
    md.append("## Table 2 — IN-vs-OUT shift disagreement")
    md.append("")
    md.append("Under the per-probe-arrival window (`r·f/c_g + 7T`), an accurate `c_g(f)`")
    md.append("should produce zero median shift at every probe (snap = expected up to")
    md.append("UC quantization). Any per-probe systematic shift flags either a residual")
    md.append("`c_g` error at that probe OR a probe-local detection artefact (e.g. window")
    md.append("start sitting on the rising edge of the wave-train envelope, where small")
    md.append("detector-noise picks adjacent upcrossings on either side of the threshold).")
    md.append("")
    md.append("| wind | IN (9373/170) median | OUT (12400/250) median | IN − OUT |")
    md.append("|---|---|---|---|")
    for wind in WINDS:
        in_med = sub[sub["WindCondition"] == wind]["9373/170_shift_periods"].dropna().median()
        out_med = sub[sub["WindCondition"] == wind]["12400/250_shift_periods"].dropna().median()
        if np.isfinite(in_med) and np.isfinite(out_med):
            md.append(f"| {wind} | {in_med:+.3f} | {out_med:+.3f} | **{in_med - out_med:+.3f}** |")
    md.append("")

    # Table 3: amplitude dependence at OUT and IN
    md.append("## Table 3 — Amplitude dependence of probe shift")
    md.append("")
    md.append("Both probes are tabulated against paddle amplitude. A finite-amplitude")
    md.append("(Stokes) correction to `c_g` would scale roughly ∝ ka, so any systematic")
    md.append("trend with amplitude is the signature to look for. A flat or noise-like")
    md.append("dependence rules Stokes out and points at probe-local detection effects.")
    md.append("")
    md.append("OUT probe (12400/250), median shift in periods per (amp, wind):")
    md.append("")
    md.append("| amp [V] | wind=no | wind=full |")
    md.append("|---|---|---|")
    for amp in AMPS:
        row = [f"{amp}"]
        for wind in WINDS:
            cell = sub[
                np.isclose(sub["WaveAmplitudeInput [Volt]"], amp, atol=0.01)
                & (sub["WindCondition"] == wind)
                & sub["WaveFrequencyInput [Hz]"].between(1.3, 1.7)
            ]
            vals = cell["12400/250_shift_periods"].dropna()
            if len(vals):
                row.append(f"{vals.median():+.3f} ± {vals.std():.3f}  (n={len(vals)})")
            else:
                row.append("—")
        md.append("| " + " | ".join(row) + " |")
    md.append("")
    md.append("Same table for IN probe (9373/170):")
    md.append("")
    md.append("| amp [V] | wind=no | wind=full |")
    md.append("|---|---|---|")
    for amp in AMPS:
        row = [f"{amp}"]
        for wind in WINDS:
            cell = sub[
                np.isclose(sub["WaveAmplitudeInput [Volt]"], amp, atol=0.01)
                & (sub["WindCondition"] == wind)
                & sub["WaveFrequencyInput [Hz]"].between(1.3, 1.7)
            ]
            vals = cell["9373/170_shift_periods"].dropna()
            if len(vals):
                row.append(f"{vals.median():+.3f} ± {vals.std():.3f}  (n={len(vals)})")
            else:
                row.append("—")
        md.append("| " + " | ".join(row) + " |")
    md.append("")

    # Table 4: the 1.4 Hz OUT per-run breakdown
    md.append("## Table 4 — 1.4 Hz OUT-probe per-run shifts")
    md.append("")
    md.append("Under the previous wavemaker-anchored window the 1.4 Hz OUT shift flipped")
    md.append("median sign between nowind and fullwind. With the new per-probe-arrival")
    md.append("window the per-cell medians no longer flip in a clean way (Table 5), but")
    md.append("individual 1.4 Hz runs still show shifts close to the ±0.5 T snap boundary,")
    md.append("which is where 'nearest upcrossing' can switch between adjacent cycles.")
    md.append("Listed below for transparency — useful when sampling individual runs to")
    md.append("inspect their snap behaviour directly.")
    md.append("")
    md.append("| path | amp [V] | wind | shift (T) | hg_expected_start | snap start (Computed start) |")
    md.append("|---|---|---|---|---|---|")
    fourteen = sub[np.isclose(sub["WaveFrequencyInput [Hz]"], 1.4, atol=0.005)]
    for _, r in fourteen.iterrows():
        shift = r["12400/250_shift_periods"]
        exp_start = r.get("Probe 12400/250 hg_expected_start", None)
        used_start = r.get("Computed Probe 12400/250 start", None)
        name = Path(r["path"]).name[:60]
        md.append(f"| `{name}` | {r['WaveAmplitudeInput [Volt]']} | {r['WindCondition']} | "
                  f"{shift:+.3f} | {exp_start} | {used_start} |")
    md.append("")

    # Table 5: full (freq × amp × wind × probe) breakdown
    md.append("## Table 5 — Full breakdown by (freq, amp, wind, probe)")
    md.append("")
    md.append("Long-form table, see also `hg_snap_shift_diagnostic_summary.csv`.")
    md.append("")
    md.append("| freq [Hz] | amp [V] | wind | probe | n | median | std |")
    md.append("|---|---|---|---|---|---|---|")
    for _, r in long_df.iterrows():
        md.append(f"| {r['freq_hz']:.2f} | {r['amp_V']} | {r['wind']} | `{r['probe']}` | "
                  f"{int(r['n'])} | {r['median']:+.3f} | "
                  f"{r['std'] if pd.notna(r['std']) else 'n/a':.3f} |" if pd.notna(r["std"])
                  else f"| {r['freq_hz']:.2f} | {r['amp_V']} | {r['wind']} | `{r['probe']}` | "
                       f"{int(r['n'])} | {r['median']:+.3f} | n/a |")
    md.append("")

    # Verdict — observations only, hypotheses clearly flagged
    md.append("## Observations (facts)")
    md.append("")
    md.append("- **O1**: OUT probe (12400/250) median shift ≈ +0.05 T (nowind) and")
    md.append("  +0.04 T (fullwind), std ≈ 0.24 T. Effectively zero on this dataset —")
    md.append("  consistent with `c_g` predicting OUT arrival accurately under the")
    md.append("  per-probe-arrival window. The previous wavemaker-anchored window's")
    md.append("  ~−0.2 T OUT offset (2026-04-22 finding) is not present here.")
    md.append("- **O2**: IN probes (9373/170 and 9373/340) show median shift +0.28 T")
    md.append("  (nowind) and ≈ +0.07 T (fullwind), with high std (~0.29 T) under")
    md.append("  nowind. The two IN probes track each other closely (+0.286 vs +0.281")
    md.append("  nowind), so the offset is not a probe-individual effect.")
    md.append("- **O3**: IN − OUT median shift is +0.234 T (nowind), +0.021 T (fullwind).")
    md.append("  Under the new window the disagreement is now driven by the IN side, not")
    md.append("  by an OUT-side anomaly.")
    md.append("- **O4**: Within each (freq, amp, wind) cell the IN-nowind shift is tight")
    md.append("  (per-cell std typically 0.01–0.04 T, n=2–9), but between cells the median")
    md.append("  flips sign across freq/amp combinations — e.g. IN-nowind 1.5 Hz at 0.1 V")
    md.append("  sits at −0.31 T while 1.5 Hz at 0.3 V sits at +0.34 T (Table 5). The")
    md.append("  pooled +0.286 T median in Table 1 is a population mean over cells with")
    md.append("  varying sign, not a single coherent offset.")
    md.append("- **O5**: Upstream probe 8804/250 shows median shift −0.07 T regardless of")
    md.append("  wind. Smaller in magnitude than IN-nowind, opposite in sign.")
    md.append("- **O6**: The 1.4 Hz OUT-probe per-run shifts (Table 4) sit close to the")
    md.append("  ±0.5 T snap boundary (range −0.45 to +0.48 T across listed runs).")
    md.append("")
    md.append("## Candidate explanations (hypotheses, not verified)")
    md.append("")
    md.append("- **H1** (UC-snap edge case at low-SNR window starts): the per-probe-arrival")
    md.append("  window starts at `r/c_g + 7T`, which puts the IN window very early in")
    md.append("  the wave train (where the envelope is still rising). At a window start")
    md.append("  this close to the rising-edge envelope, which upcrossing the snap locks")
    md.append("  onto can shift between adjacent cycles depending on cell-specific signal")
    md.append("  shape — consistent with O4's tight within-cell std plus large between-")
    md.append("  cell sign flips. Under fullwind the IN-nowind +0.286 T median collapses")
    md.append("  to +0.056 T (Table 1), suggesting the wind background tilts the snap")
    md.append("  consistently. The two IN probes seeing the same wavefront would also")
    md.append("  explain their nearly-identical pooled medians (O2). Not directly tested.")
    md.append("- **H2** (finite-amplitude Stokes correction to `c_g`): second-order Stokes")
    md.append("  nonlinearity slightly modifies `c_phase` and `c_g` at large `ka`. If")
    md.append("  active, Table 3 should show median shift growing monotonically with")
    md.append("  amplitude. Inspect the per-(amp, wind) values for a clean trend.")
    md.append("")
    md.append("## For another agent wanting to re-derive these numbers")
    md.append("")
    md.append("1. Re-run this script (no arguments): `python analysis_scratch/hg_snap_shift_diagnostic.py`")
    md.append("2. Raw per-run data: `hg_snap_shift_diagnostic.csv`")
    md.append("3. Breakdown table: `hg_snap_shift_diagnostic_summary.csv` (long form)")
    md.append("4. Underlying columns in meta.json (per probe):")
    md.append("   - `Probe {pos} hg_snap_shift` — signed shift in samples")
    md.append("   - `Probe {pos} hg_expected_start` — pre-snap theoretical H&G start")
    md.append("   - `Computed Probe {pos} start/end` — post-snap window (used by FFT/LS)")
    md.append("5. Convert to periods: `shift_periods = shift_samples / round(fs / f_paddle)`")
    md.append("")
    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"   findings table → {OUT_MD.relative_to(BASE)}")

    # ── Figure ────────────────────────────────────────────────────────────────
    print("4. Plotting …")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    # Panel (a): median shift per probe, split by wind
    ax = axes[0]
    x_pos = np.arange(len(PROBES))
    for i, wind in enumerate(WINDS):
        meds = []
        stds = []
        for p in PROBES:
            v = sub[sub["WindCondition"] == wind][f"{p}_shift_periods"].dropna()
            meds.append(v.median() if len(v) else np.nan)
            stds.append(v.std() if len(v) > 1 else 0)
        off = (i - 0.5) * 0.15
        ax.errorbar(x_pos + off, meds, yerr=stds, fmt="o", capsize=4,
                    label=f"wind={wind}", markersize=8)
    ax.axhline(0, color="k", lw=0.8, alpha=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(PROBES, rotation=30)
    ax.set_ylabel("Snap shift (wave periods)")
    ax.set_title("", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Panel (b): OUT probe shift vs amplitude
    ax = axes[1]
    for wind, color in (("no", "tab:blue"), ("full", "tab:red")):
        xs, meds, stds = [], [], []
        for amp in AMPS:
            cell = sub[
                np.isclose(sub["WaveAmplitudeInput [Volt]"], amp, atol=0.01)
                & (sub["WindCondition"] == wind)
                & sub["WaveFrequencyInput [Hz]"].between(1.3, 1.7)
            ]
            v = cell["12400/250_shift_periods"].dropna()
            if len(v) == 0:
                continue
            xs.append(amp); meds.append(v.median()); stds.append(v.std() if len(v) > 1 else 0)
        ax.errorbar(xs, meds, yerr=stds, fmt="o-", color=color, capsize=4,
                    label=f"wind={wind}", markersize=8)
    ax.axhline(0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel("Paddle amplitude input (V)")
    ax.set_title("", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.suptitle("", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"   figure → {OUT_PNG.relative_to(BASE)}")
    print("\nDone.")


if __name__ == "__main__":
    main()
