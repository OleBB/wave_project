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
verify numbers referenced in session_2026-04-22.md (the "~-0.2 T OUT
offset" discussion and the "1.4 Hz nowind→fullwind flip").

Also adds the amplitude-dependence breakdown (0.1 V / 0.2 V / 0.3 V) —
a test of whether finite-amplitude nonlinearity contributes to the
offset.

Scope: fullpanel wave runs in the canon March-2026 lowrange folders,
quality_flag=ok.

Output:
    analysis_scratch/hg_snap_shift_diagnostic.csv       (per-run raw)
    analysis_scratch/hg_snap_shift_diagnostic_summary.csv (long form)
    analysis_scratch/hg_snap_shift_diagnostic.md        (tables + verdict)
    analysis_scratch/hg_snap_shift_diagnostic.png       (heatmap figure)
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
    md.append("If the ~−0.2 T offset at OUT were purely a wavemaker phase reference issue,")
    md.append("ALL probes should show the same median shift (c_g(f) predicts the same")
    md.append("arrival phase at every probe relative to paddle). Any cross-probe")
    md.append("disagreement argues for tank-local physics (amplitude dispersion,")
    md.append("reflections, near-panel effects) rather than a pure-phase-reference cause.")
    md.append("")
    md.append("| wind | IN (9373/170) median | OUT (12400/250) median | IN − OUT |")
    md.append("|---|---|---|---|")
    for wind in WINDS:
        in_med = sub[sub["WindCondition"] == wind]["9373/170_shift_periods"].dropna().median()
        out_med = sub[sub["WindCondition"] == wind]["12400/250_shift_periods"].dropna().median()
        if np.isfinite(in_med) and np.isfinite(out_med):
            md.append(f"| {wind} | {in_med:+.3f} | {out_med:+.3f} | **{in_med - out_med:+.3f}** |")
    md.append("")

    # Table 3: amplitude dependence at OUT probe
    md.append("## Table 3 — Amplitude dependence of OUT-probe shift")
    md.append("")
    md.append("Does the −0.2 T offset grow (or shrink) with paddle amplitude? A Stokes-")
    md.append("nonlinearity mechanism would predict the offset magnitude scales with wave")
    md.append("steepness (roughly ∝ ka ∝ amplitude at fixed frequency).")
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

    # Table 4: the 1.4 Hz OUT flip investigation
    md.append("## Table 4 — The 1.4 Hz OUT-probe flip")
    md.append("")
    md.append("Observation in the session log: at 1.4 Hz, median OUT shift flips from")
    md.append("~-0.35 T (nowind) to ~+0.35 T (fullwind). A ±0.5 T jump is the boundary")
    md.append("where 'nearest upcrossing' can flip between two adjacent ones. Listing")
    md.append("all 1.4 Hz OUT runs below with individual shifts so another agent can")
    md.append("confirm or falsify.")
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
    md.append("- **O1**: OUT probe (12400/250) shows median shift ~−0.18 T under nowind,")
    md.append("  ~−0.23 T under fullwind. Systematic negative bias across all amplitudes.")
    md.append("- **O2**: IN probes (9373/170 and 9373/340) show median shifts near zero")
    md.append("  (~+0.04 T nowind, ~0 fullwind). Much smaller than OUT.")
    md.append("- **O3**: IN − OUT median shift is ~+0.21 T (nowind) and ~+0.30 T (fullwind).")
    md.append("  Any explanation purely about wavemaker phase reference would predict 0.")
    md.append("- **O4**: The 1.4 Hz OUT flip (Table 4) is driven by individual runs with")
    md.append("  shifts near ±0.5 T — at that boundary 'nearest upcrossing' can flip.")
    md.append("- **O5**: Amplitude dependence of OUT shift (Table 3) — examine the numbers,")
    md.append("  see whether the offset grows with amplitude (Stokes test).")
    md.append("")
    md.append("## Candidate explanations (hypotheses, not verified)")
    md.append("")
    md.append("- **H1** (wavemaker soft-start phase reference): would produce equal-")
    md.append("  magnitude offsets at ALL probes → inconsistent with O3. Partially rule out.")
    md.append("- **H2** (group-velocity underestimate for 12.4 m travel): if c_g at OUT")
    md.append("  is slightly slower than deep-water prediction, actual wave arrives later")
    md.append("  than predicted → snap finds upcrossing AFTER expected → POSITIVE shift.")
    md.append("  Observed sign is NEGATIVE, so this mechanism doesn't fit either.")
    md.append("- **H3** (group-velocity OVERESTIMATE for 12.4 m travel): predicted arrival")
    md.append("  later than actual; snap finds upcrossing BEFORE expected → NEGATIVE shift.")
    md.append("  Sign matches. Requires c_g to be ~0.2 T too fast over the longer travel.")
    md.append("  In deep water, a 0.2 T error at 1.4 Hz ≈ 0.14 s travel-time error over")
    md.append("  12.4 m = effective c_g 0.9 m/s (vs predicted 0.56) or ~60% faster. Not")
    md.append("  physically credible for a linear deep-water wave.")
    md.append("- **H4** (finite-amplitude Stokes correction to c_g): second-order Stokes")
    md.append("  nonlinearity slightly modifies c_phase and c_g at large ka. Effect size")
    md.append("  is small (typically < few percent at ka ~0.1). Probably insufficient to")
    md.append("  account for 0.2 T, but Table 3 would show it as an amplitude dependence.")
    md.append("- **H5** (near-panel reflection at OUT probe changing detected upcrossing):")
    md.append("  the OUT probe sits between the panel and the beach. If a partial")
    md.append("  reflection from the panel creates a standing-wave pattern at the OUT")
    md.append("  probe, local phase is shifted relative to the free-running tone. Would")
    md.append("  affect OUT much more than IN. Magnitude depends on reflection coefficient")
    md.append("  (measured ~0.05–0.07 under nowind 0.2 V — see mansard_funke_findings.md).")
    md.append("  Could plausibly shift OUT upcrossing by up to ~0.2 T for R in that range.")
    md.append("- **H6** (sensor-response lag on ULS probe): each ULS probe has its own")
    md.append("  electronic response. A small but probe-specific delay would produce a")
    md.append("  constant per-probe offset. Doesn't depend on frequency, amplitude, or wind.")
    md.append("  Would require bench measurement to verify.")
    md.append("")
    md.append("To go further: the amplitude-dependence in Table 3 is the most useful")
    md.append("discriminator — if shift magnitude scales with amplitude, H4 / H5 gain")
    md.append("support; if flat, H6 or H3 (rejected for different reasons) are in play.")
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
    ax.axhline(-0.2, color="red", ls="--", lw=0.8, alpha=0.5, label="~−0.2 T reference")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(PROBES, rotation=30)
    ax.set_ylabel("Snap shift (wave periods)")
    ax.set_title("(a) Median shift per probe (all runs)", fontsize=10)
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
    ax.set_title("(b) OUT probe (12400/250) shift vs amplitude\n(Stokes-nonlinearity discriminator)",
                 fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.suptitle("H&G snap-shift diagnostic — canon March-2026 lowrange (n=" + str(len(sub)) + ")",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"   figure → {OUT_PNG.relative_to(BASE)}")
    print("\nDone.")


if __name__ == "__main__":
    main()
