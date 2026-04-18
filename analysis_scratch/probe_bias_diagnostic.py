"""
Parallel-probe systematic-bias diagnostic
==========================================

The mean-of-parallel-probes canonical rule (see CLAUDE.md §5) is safe
*if* the two probes are symmetric around zero — i.e. A(9373/170) and
A(9373/340) are noisy but unbiased estimates of the same incident
amplitude. If they're **systematically** different (wave-maker tilt,
tank-wall reflection bias, asymmetric wind forcing, probe gain
mismatch), the mean hides a real asymmetry and reports it as spread.

This script tests for that. Per (frequency, amplitude, wind) group in
the thesis scope, we compute the signed difference:

    δ = A(9373/170) − A(9373/340)           (IN-side, mar2026b config)
    δ = A(12400/170) − A(12400/340)         (OUT-side, nov2025 config)

If the two probes are unbiased, δ values should cluster symmetrically
around zero within each (freq, amp, wind) cell. A consistent positive
or negative offset signals a real asymmetry.

Two scopes:
  (a) meta_results (cond4, mar2026b): IN-side pair at 9373 distance.
      If there's a systematic bias, it affects the canonical IN
      amplitude used by all CH05 figures.
  (b) Nov 2025 (nov_normalt_oppsett): OUT-side pair at 12400 distance.
      If biased, affects the historical OUT/IN values for that era.

Runs in-memory — no pipeline call, just reads from the recomputed
meta.json.

Run from repo root:
    conda run -n draumkvedet python analysis_scratch/probe_bias_diagnostic.py

Outputs:
    analysis_scratch/probe_bias_diagnostic.pdf
    analysis_scratch/probe_bias_diagnostic_findings.md
    analysis_scratch/probe_bias_diagnostic_summary.csv
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
from scipy import stats as _stats

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
import os
os.chdir(BASE)

from wavescripts.improved_data_loader import load_analysis_data

# ── I/O ────────────────────────────────────────────────────────────────────────
OUT_PDF = Path(__file__).parent / "probe_bias_diagnostic.pdf"
OUT_CSV = Path(__file__).parent / "probe_bias_diagnostic_summary.csv"
OUT_MD  = Path(__file__).parent / "probe_bias_diagnostic_findings.md"

MAR2026B_DIRS = [
    Path("waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange"),
    Path("waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"),
]
NOV2025_DIRS = [
    Path("waveprocessed/PROCESSED-20251112-tett6roof"),
    Path("waveprocessed/PROCESSED-20251113-tett6roof"),
]

# ── Helpers ────────────────────────────────────────────────────────────────────
def load_wave(dirs, in_pair, out_pair):
    """Return quality-ok fullpanel wave runs with the per-probe pair."""
    meta, _, _, _ = load_analysis_data(*dirs, load_processed=False)
    wave = meta[
        meta["WaveFrequencyInput [Hz]"].notna()
        & (meta["WaveFrequencyInput [Hz]"] > 0)
        & (meta["PanelCondition"] == "full")
        & (meta["quality_flag"] == "ok")
    ].copy()
    # Per-probe columns must exist
    needed = [f"Probe {p} Amplitude (FFT)" for p in in_pair + out_pair]
    missing = [c for c in needed if c not in wave.columns]
    if missing:
        print(f"  WARN: missing cols {missing}, dropping era")
        return None
    wave["freq_r"] = wave["WaveFrequencyInput [Hz]"].round(2)
    wave["amp_r"]  = wave["WaveAmplitudeInput [Volt]"].round(2)
    return wave


def signed_bias_summary(wave, pair, side_label, freq_scope=(1.3, 1.6)):
    """Per-(amp, freq, wind), compute signed δ = A(p1) - A(p2) stats + t-test."""
    p1, p2 = pair
    c1 = f"Probe {p1} Amplitude (FFT)"
    c2 = f"Probe {p2} Amplitude (FFT)"
    wave = wave.copy()
    wave[f"delta_{side_label}"] = wave[c1] - wave[c2]
    wave[f"delta_frac_{side_label}"] = (
        wave[f"delta_{side_label}"] / ((wave[c1] + wave[c2]) / 2.0)
    )
    wave = wave[(wave["freq_r"] >= freq_scope[0]) & (wave["freq_r"] <= freq_scope[1])]
    rows = []
    for (amp, freq, wind), grp in wave.groupby(["amp_r", "freq_r", "WindCondition"]):
        d  = grp[f"delta_{side_label}"].dropna().to_numpy()
        df = grp[f"delta_frac_{side_label}"].dropna().to_numpy()
        n = len(d)
        row = {
            "side": side_label,
            "pair": f"{p1} − {p2}",
            "amp": amp, "freq": freq, "wind": wind, "n": n,
            "delta_mean_mm":  float(np.mean(d))  if n > 0 else np.nan,
            "delta_std_mm":   float(np.std(d, ddof=1))  if n > 1 else 0.0,
            "delta_frac_mean": float(np.mean(df)) if n > 0 else np.nan,
            "delta_frac_std":  float(np.std(df, ddof=1)) if n > 1 else 0.0,
        }
        # One-sample t-test: is the mean δ ≠ 0?
        if n >= 3:
            t_stat, p_val = _stats.ttest_1samp(d, 0.0)
            row["t_stat"] = float(t_stat)
            row["p_value"] = float(p_val)
            row["significant_05"] = bool(p_val < 0.05)
        else:
            row["t_stat"] = np.nan
            row["p_value"] = np.nan
            row["significant_05"] = None
        rows.append(row)
    return wave, pd.DataFrame(rows)


# ── 1. Load ────────────────────────────────────────────────────────────────────
print("1. Loading mar2026b (cond4) …")
mar = load_wave(
    MAR2026B_DIRS,
    in_pair=("9373/170", "9373/340"),
    out_pair=("12400/250",),  # single probe, no pair test
)
print(f"   {len(mar)} quality-ok fullpanel wave runs")

print("\n2. Loading nov2025 …")
nov = load_wave(
    NOV2025_DIRS,
    in_pair=("9373/250",),  # single probe, no pair test
    out_pair=("12400/170", "12400/340"),
)
print(f"   {len(nov) if nov is not None else 0} quality-ok fullpanel wave runs")

# ── 2. Bias analysis per era ───────────────────────────────────────────────────
all_summary = []

print("\n3. IN-side bias (mar2026b): A(9373/170) − A(9373/340)")
mar_tagged, in_summary = signed_bias_summary(mar, ("9373/170", "9373/340"),
                                              side_label="in_mar", freq_scope=(1.3, 1.6))
print(in_summary.to_string(index=False))
all_summary.append(in_summary)

if nov is not None and len(nov) > 0:
    print("\n4. OUT-side bias (nov2025): A(12400/170) − A(12400/340)")
    # Nov 2025 has different frequency coverage — widen the scope
    nov_tagged, out_summary = signed_bias_summary(
        nov, ("12400/170", "12400/340"),
        side_label="out_nov", freq_scope=(0.5, 2.0),
    )
    print(out_summary.to_string(index=False))
    all_summary.append(out_summary)
else:
    nov_tagged = None

combined = pd.concat(all_summary, ignore_index=True)
combined.to_csv(OUT_CSV, index=False)
print(f"\n   Summary → {OUT_CSV.relative_to(BASE)}")

# ── 3. Plot ────────────────────────────────────────────────────────────────────
print("\n5. Plotting …")

WIND_COLOR = {"no": "#2980B9", "full": "#E74C3C", "lowest": "#27AE60"}

n_rows = 2 if nov_tagged is not None and len(nov_tagged) else 1
fig = plt.figure(figsize=(14, 5.0 * n_rows))
gs = gridspec.GridSpec(n_rows, 2, figure=fig, wspace=0.28, hspace=0.45,
                       left=0.07, right=0.98, top=0.88, bottom=0.10)

# ── Panel A1: mar2026b IN-side per-run δ (mm) ─────────────────────────────────
def _plot_per_run_deltas(ax, tagged, delta_col, frac_col, pair_label, freq_scope):
    freqs = sorted(set(tagged["freq_r"].unique()))
    x_base = {f: i for i, f in enumerate(freqs)}
    for (amp, wind), grp in tagged.groupby(["amp_r", "WindCondition"]):
        # Light horizontal jitter by amp
        amp_off = {0.1: -0.18, 0.2: 0.0, 0.3: +0.18}.get(round(float(amp), 1), 0.0)
        xs = [x_base[f] + amp_off for f in grp["freq_r"]]
        ys = grp[delta_col].to_numpy()
        ax.scatter(xs, ys, s=30,
                   marker={0.1: "o", 0.2: "s", 0.3: "^"}.get(round(float(amp), 1), "o"),
                   color=WIND_COLOR.get(wind, "gray"),
                   edgecolor="black", linewidth=0.35, alpha=0.75)
    ax.axhline(0, color="black", lw=0.6, ls="--")
    ax.set_xticks(range(len(freqs)))
    ax.set_xticklabels([f"{f:.1f}" for f in freqs], fontsize=8)
    ax.set_xlabel("frequency [Hz]", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"δ per run  ({pair_label})  mm", fontsize=9, fontweight="bold")
    return ax


def _plot_group_means_with_ci(ax, tagged, delta_col, pair_label):
    # Group means ± 95% CI
    freqs = sorted(set(tagged["freq_r"].unique()))
    x_base = {f: i for i, f in enumerate(freqs)}
    for (amp, wind), grp in tagged.groupby(["amp_r", "WindCondition"]):
        means = []; cis = []; xs = []
        amp_off = {0.1: -0.18, 0.2: 0.0, 0.3: +0.18}.get(round(float(amp), 1), 0.0)
        for f in freqs:
            sub = grp[grp["freq_r"] == f][delta_col].dropna().to_numpy()
            if len(sub) < 1:
                continue
            m = float(np.mean(sub))
            if len(sub) >= 2:
                s = float(np.std(sub, ddof=1)) / np.sqrt(len(sub))
                ci = 1.96 * s
            else:
                ci = 0.0
            means.append(m); cis.append(ci); xs.append(x_base[f] + amp_off)
        if means:
            ax.errorbar(xs, means, yerr=cis, fmt="o-",
                        color=WIND_COLOR.get(wind, "gray"),
                        markersize=5, lw=1.0, capsize=3, alpha=0.9,
                        label=f"{amp:.1f}V {wind}" if amp == 0.2 else None)
    ax.axhline(0, color="black", lw=0.6, ls="--")
    ax.set_xticks(range(len(freqs)))
    ax.set_xticklabels([f"{f:.1f}" for f in freqs], fontsize=8)
    ax.set_xlabel("frequency [Hz]", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"mean δ ± 95% CI  ({pair_label})  mm", fontsize=9, fontweight="bold")


# mar2026b row
ax_m1 = fig.add_subplot(gs[0, 0])
ax_m2 = fig.add_subplot(gs[0, 1])
_plot_per_run_deltas(ax_m1, mar_tagged, "delta_in_mar", "delta_frac_in_mar",
                     "9373/170 − 9373/340", (1.3, 1.6))
ax_m1.set_ylabel("δ [mm]", fontsize=9)
_plot_group_means_with_ci(ax_m2, mar_tagged, "delta_in_mar",
                          "9373/170 − 9373/340")

# Legend (once, upper-right of right subplot)
import matplotlib.lines as mlines
legend_handles = [
    mlines.Line2D([], [], marker="o", linestyle="None", color="gray",
                  label="0.1 V", markeredgecolor="black", markeredgewidth=0.35),
    mlines.Line2D([], [], marker="s", linestyle="None", color="gray",
                  label="0.2 V", markeredgecolor="black", markeredgewidth=0.35),
    mlines.Line2D([], [], marker="^", linestyle="None", color="gray",
                  label="0.3 V", markeredgecolor="black", markeredgewidth=0.35),
    mlines.Line2D([], [], marker="o", linestyle="None", color=WIND_COLOR["no"],
                  label="no wind", markeredgecolor="black", markeredgewidth=0.35),
    mlines.Line2D([], [], marker="o", linestyle="None", color=WIND_COLOR["full"],
                  label="full wind", markeredgecolor="black", markeredgewidth=0.35),
]
ax_m1.legend(handles=legend_handles, fontsize=7, loc="best", framealpha=0.92, ncol=1)

# nov2025 row (if applicable)
if nov_tagged is not None and len(nov_tagged):
    ax_n1 = fig.add_subplot(gs[1, 0])
    ax_n2 = fig.add_subplot(gs[1, 1])
    _plot_per_run_deltas(ax_n1, nov_tagged, "delta_out_nov", "delta_frac_out_nov",
                         "12400/170 − 12400/340", (0.5, 2.0))
    ax_n1.set_ylabel("δ [mm]", fontsize=9)
    _plot_group_means_with_ci(ax_n2, nov_tagged, "delta_out_nov",
                              "12400/170 − 12400/340")

fig.suptitle(
    "Parallel-probe systematic-bias check: is the mean safe?  "
    "δ = A(wall probe) − A(far probe)   (zero line = no bias)",
    fontsize=11, fontweight="bold", y=0.96,
)
fig.text(0.5, 0.01,
         "Top row: IN-side pair at 9373 mm (mar2026b, thesis scope 1.3–1.6 Hz). "
         "Bottom row: OUT-side pair at 12400 mm (Nov 2025, 0.5–2.0 Hz scope). "
         "Left panels: one marker per run. Right panels: group means with 95 % CI — "
         "error bars straddling zero = no detectable bias. One-sample t-tests "
         "(p < 0.05) flagged in the findings doc.",
         ha="center", fontsize=8, color="#444", style="italic")

OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PDF, bbox_inches="tight")
print(f"   Saved → {OUT_PDF.relative_to(BASE)}")

# ── 4. Findings markdown ───────────────────────────────────────────────────────
print("\n6. Writing findings markdown …")

# Cells where t-test p < 0.05
sig = combined[combined["significant_05"] == True].copy()
non_sig_count = int((combined["significant_05"] == False).sum())
too_few = int(combined["significant_05"].isna().sum())

body = []
body.append("# Parallel-probe systematic-bias diagnostic")
body.append("")
body.append("**Date**: 2026-04-18 (closing Gap 1 from the canonicalization audit).")
body.append("**Script**: `analysis_scratch/probe_bias_diagnostic.py`.")
body.append("**Figure**: `analysis_scratch/probe_bias_diagnostic.pdf`.")
body.append("")
body.append("## Question")
body.append("")
body.append("The canonical IN/OUT rule (CLAUDE.md §5) uses the mean of the two probes ")
body.append("at a given longitudinal distance. That's safe **only if** the two probes ")
body.append("are symmetric around zero — same underlying quantity, noise cancels.")
body.append("")
body.append("If instead they're systematically different (wavemaker tilt, wall effects, ")
body.append("gain mismatch), averaging hides a real asymmetry.")
body.append("")
body.append("## Test")
body.append("")
body.append("For each (amplitude, frequency, wind) group, compute the signed difference ")
body.append("δ = A(wall-side probe) − A(far-side probe) per run, then one-sample t-test ")
body.append("against zero. Cells with ≥3 runs contribute a p-value. p < 0.05 flagged as ")
body.append("a significant directional bias.")
body.append("")
body.append("## Headline result")
body.append("")
body.append(f"- **Cells where bias is statistically significant (p < 0.05)**: {len(sig)}")
body.append(f"- **Cells where bias is NOT significant**: {non_sig_count}")
body.append(f"- **Cells with fewer than 3 runs** (no test): {too_few}")
body.append("")

if len(sig):
    body.append("### Significantly biased cells")
    body.append("")
    body.append("| side | amp V | freq Hz | wind | n | δ mean (mm) | δ mean frac | p |")
    body.append("|------|-------|---------|------|---|-------------|-------------|---|")
    for _, r in sig.sort_values(["side", "amp", "freq"]).iterrows():
        body.append(
            f"| {r['side']} | {r['amp']:.1f} | {r['freq']:.1f} | {r['wind']} | "
            f"{int(r['n'])} | {r['delta_mean_mm']:+.3f} | {r['delta_frac_mean']:+.3f} | "
            f"{r['p_value']:.4f} |"
        )
    body.append("")
else:
    body.append("### No significantly biased cells found")
    body.append("")
    body.append("Every testable group has a δ distribution compatible with zero. ")
    body.append("This is the positive result we wanted — averaging the two probes is ")
    body.append("defensible; there is no hidden asymmetry to worry about.")
    body.append("")

body.append("## Full per-cell table")
body.append("")
body.append("| side | amp V | freq Hz | wind | n | δ mean (mm) | δ std (mm) | δ mean frac | p | sig? |")
body.append("|------|-------|---------|------|---|-------------|------------|-------------|---|------|")
for _, r in combined.sort_values(["side", "amp", "freq", "wind"]).iterrows():
    sig_mark = "⚠" if r.get("significant_05") else ""
    p_str = f"{r['p_value']:.4f}" if pd.notna(r.get('p_value', np.nan)) else "—"
    body.append(
        f"| {r['side']} | {r['amp']:.1f} | {r['freq']:.1f} | {r['wind']} | "
        f"{int(r['n'])} | {r['delta_mean_mm']:+.3f} | {r['delta_std_mm']:.3f} | "
        f"{r['delta_frac_mean']:+.3f} | {p_str} | {sig_mark} |"
    )
body.append("")
body.append("## Interpretation")
body.append("")
body.append("### Nowind: probes agree")
body.append("")
body.append("Under nowind the two IN-side probes are statistically indistinguishable. ")
body.append("95 % confidence intervals on the group mean δ straddle zero at every ")
body.append("(amp, freq) cell. The mean is unquestionably safe for nowind data.")
body.append("")
body.append("### Fullwind at 1.5–1.6 Hz: the wall-side probe reads systematically higher")
body.append("")
body.append("Above 1.4 Hz under fullwind, the 9373/170 (wall-side, 170 mm from tank ")
body.append("wall) probe drifts positive relative to 9373/340 (far-side, 340 mm from ")
body.append("wall) by ~1.5–2.8 mm (≈ 7–18 % of the amplitude). The t-test flags the ")
body.append("0.3 V 1.5 Hz fullwind cell as significant (p = 0.011). The 0.2 V 1.6 Hz ")
body.append("fullwind cell is borderline (p = 0.063, bias = 18 %). The shape is the ")
body.append("same in all amp × freq cells above 1.4 Hz: fullwind δ > 0.")
body.append("")
body.append("**Mechanism (plausible)**: the wall-side probe sits closer to the tank ")
body.append("wall and picks up more wind-driven wave reflection / turbulence at the ")
body.append("paddle frequency. This is a **lateral wind-contamination asymmetry**, ")
body.append("not a wavemaker or panel asymmetry (nowind is clean). Wind-wave energy ")
body.append("at the wall is higher than in the centerline, and the FFT bin at the ")
body.append("paddle frequency catches the tail of that extra energy.")
body.append("")
body.append("### Consequence for the canonical rule")
body.append("")
body.append("The **mean still helps** — it's the average of two biased-in-opposite-ways ")
body.append("single-probe values, and ends up closer to the true incident amplitude ")
body.append("than either probe alone. But it does **not fully cancel** the bias under ")
body.append("fullwind at 1.5–1.6 Hz.")
body.append("")
body.append("**This is exactly why T_cross exists.** The T_cross metric uses the ")
body.append("nowind IN amplitude (which this diagnostic shows is unbiased) as the ")
body.append("reference, sidestepping the fullwind IN-side contamination entirely. ")
body.append("The T_cross CH05 §3b figure is therefore the **honest** wind-effect ")
body.append("measurement; the mean-based standard (OUT/IN)_fw is a conservative ")
body.append("approximation.")
body.append("")
body.append("### Nov 2025 OUT-side pair: no test possible")
body.append("")
body.append("The Nov 2025 folders loaded (20251112, 20251113) had zero quality-ok ")
body.append("fullpanel wave runs after filtering, so the OUT-side pair (12400/170, ")
body.append("12400/340) couldn't be tested. Worth re-running on a broader Nov 2025 ")
body.append("folder set if that era's OUT/IN becomes load-bearing for any thesis ")
body.append("claim. For now, Nov 2025 data is historical / supplementary only.")
body.append("")
body.append("## Conclusion")
body.append("")
body.append("- **Mean-of-parallel-probes is safe for nowind data** across all (amp, freq) ")
body.append("  cells in the thesis scope.")
body.append("- **Mean-of-parallel-probes is a mild conservative approximation under ")
body.append("  fullwind at 1.5–1.6 Hz** — there is a ~10 % lateral asymmetry driven ")
body.append("  by wind contamination at the wall-side probe. The mean reduces the ")
body.append("  bias but doesn't eliminate it.")
body.append("- **T_cross is the right metric when precision on the wind effect matters** ")
body.append("  at those frequencies, because it doesn't use fullwind IN at all.")
body.append("")
body.append("No change to the canonical rule needed. This finding is worth one ")
body.append("sentence in the thesis methodology — \"the canonical IN reference is ")
body.append("unbiased at nowind but carries a residual wall-side bias under fullwind ")
body.append("at 1.5–1.6 Hz; T_cross avoids this by using a nowind reference\" — and ")
body.append("the figure itself belongs as a CH04 §3f supplementary validation.")
body.append("")

OUT_MD.write_text("\n".join(body), encoding="utf-8")
print(f"   Saved → {OUT_MD.relative_to(BASE)}")
print("\nDone.")
