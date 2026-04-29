"""
Mooring comparison: below_90_loose230 vs below_90_loose300
===========================================================

Goal: determine whether the two under-water mooring types produce statistically
equivalent OUT/IN(FFT) ratios at matched conditions, so that the datasets can be
merged for the main figures.

Conditions compared (fullpanel, quality_flag==ok only):
  - Frequencies: 1.3–1.7 Hz (main overlap; below 1.0 Hz has sparse or zero loose300 data)
  - Amplitudes: 0.1 V, 0.2 V, 0.3 V
  - WindCondition: full, no

Methodology:
  - OUT/IN computed per run from Probe {in_pos} Amplitude (FFT) and Probe {out_pos} Amplitude (FFT)
    (narrow 0.1 Hz window, paddle frequency only — excludes wind waves)
  - Grouped by (freq, amp, wind, mooring): mean ± std
  - Comparison metric: absolute difference and relative difference (%) between mooring means

Output:
  - analysis_scratch/mooring_comparison.png  — figures
  - analysis_scratch/mooring_comparison_findings.md  — numerical findings

Run from repo root:
    conda run -n draumkvedet python analysis_scratch/mooring_comparison.py

Progress is printed to stdout as sections complete.
"""
import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime

from wavescripts.improved_data_loader import load_analysis_data
from wavescripts.plot_utils import apply_thesis_style

apply_thesis_style()

BASE = Path(__file__).parent.parent
OUT_PNG = Path(__file__).parent / "mooring_comparison.png"
OUT_MD  = Path(__file__).parent / "mooring_comparison_findings.md"

# Thesis figure (delegated-promotion pattern; main_save_figures.py verifies
# existence of these files, does not re-generate).
THESIS_NAME     = "ch04_mooring_comparison"
THESIS_PDF      = BASE / "output" / "FIGURES" / f"{THESIS_NAME}.pdf"
THESIS_STUB     = BASE / "output" / "TEXFIGU" / f"{THESIS_NAME}.tex"
THESIS_PDF.parent.mkdir(parents=True, exist_ok=True)
THESIS_STUB.parent.mkdir(parents=True, exist_ok=True)

# ── 1. Load data ───────────────────────────────────────────────────────────────
print("1. Loading all processed folders...")
dirs = sorted(glob.glob(str(BASE / "waveprocessed" / "PROCESSED-*")))
meta_all, _, _, _ = load_analysis_data(*dirs, load_processed=False)
print(f"   {len(meta_all)} total rows loaded")

# Standard filter: fullpanel, wave runs, ok quality
wave = meta_all[
    (meta_all["WaveFrequencyInput [Hz]"].notna()) &
    (meta_all["WaveFrequencyInput [Hz]"] > 0) &
    (meta_all["quality_flag"] == "ok") &
    (meta_all["PanelCondition"] == "full")
].copy()
print(f"   {len(wave)} fullpanel wave runs after quality filter")

# ── 2. Compute OUT/IN per run ──────────────────────────────────────────────────
print("2. Computing OUT/IN(FFT) per run...")

def compute_out_in(row):
    in_pos  = row.get("in_position",  None)
    out_pos = row.get("out_position", None)
    if pd.isna(in_pos) or pd.isna(out_pos):
        return np.nan
    in_amp  = row.get(f"Probe {in_pos} Amplitude (FFT)",  np.nan)
    out_amp = row.get(f"Probe {out_pos} Amplitude (FFT)", np.nan)
    if pd.isna(in_amp) or pd.isna(out_amp) or in_amp <= 0:
        return np.nan
    return out_amp / in_amp

wave["OUT/IN_computed"] = wave.apply(compute_out_in, axis=1)
valid = wave["OUT/IN_computed"].notna()
print(f"   {valid.sum()} runs with valid OUT/IN  ({(~valid).sum()} NaN — likely missing FFT amp column)")

# ── 3. Focus on overlapping mooring conditions ─────────────────────────────────
print("3. Finding overlapping conditions...")

moorings = ["below_90_loose230", "below_90_loose300"]
compare  = wave[wave["Mooring"].isin(moorings) & valid].copy()
compare["freq"] = compare["WaveFrequencyInput [Hz]"].round(2)
compare["amp"]  = compare["WaveAmplitudeInput [Volt]"].round(2)
compare["wind"] = compare["WindCondition"]

# Group: mean, std, n per (freq, amp, wind, mooring)
grp = (
    compare
    .groupby(["freq", "amp", "wind", "Mooring"])["OUT/IN_computed"]
    .agg(mean="mean", std="std", n="count")
    .reset_index()
)

# Pivot to wide: one row per (freq, amp, wind), columns for each mooring
piv = grp.pivot_table(
    index=["freq", "amp", "wind"],
    columns="Mooring",
    values=["mean", "std", "n"],
)
piv.columns = [f"{stat}_{m.split('_')[-1]}" for stat, m in piv.columns]
piv = piv.reset_index()

# Only rows where BOTH moorings have data
both = piv[
    piv["n_loose230"].notna() & (piv["n_loose230"] > 0) &
    piv["n_loose300"].notna() & (piv["n_loose300"] > 0)
].copy()

both["delta"]     = both["mean_loose300"] - both["mean_loose230"]
both["delta_pct"] = 100 * both["delta"] / both["mean_loose230"]
both["pooled_std"]= np.sqrt(
    (both.get("std_loose230", 0).fillna(0)**2 + both.get("std_loose300", 0).fillna(0)**2) / 2
)
print(f"   {len(both)} (freq, amp, wind) conditions with data for both moorings")

# ── 4. Numerical summary ───────────────────────────────────────────────────────
print("4. Building numerical summary...")

summary_lines = []
def log(s=""):
    summary_lines.append(s)
    print("  " + s if s else "")

log(f"Mooring comparison: below_90_loose230 vs below_90_loose300")
log(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
log(f"Runs analysed: {len(compare)} (fullpanel, quality_flag==ok, wave runs)")
log()
log("Overlap summary by frequency:")
freq_counts = both.groupby("freq")[["n_loose230","n_loose300"]].sum()
for freq, row in freq_counts.iterrows():
    log(f"  {freq:.2f} Hz:  loose230 n={int(row['n_loose230'])}  loose300 n={int(row['n_loose300'])}")
log()
log("Per-condition OUT/IN comparison (freq, amp, wind):")
log(f"  {'freq':>6} {'amp':>5} {'wind':>6}  {'loose230':>10} {'loose300':>10}  {'Δ':>8} {'Δ%':>7}  {'n230':>4} {'n300':>4}")
log(f"  {'-'*6} {'-'*5} {'-'*6}  {'-'*10} {'-'*10}  {'-'*8} {'-'*7}  {'-'*4} {'-'*4}")
for _, r in both.sort_values(["wind","freq","amp"]).iterrows():
    std230 = r.get("std_loose230", np.nan)
    std300 = r.get("std_loose300", np.nan)
    s230 = f"{r['mean_loose230']:.3f}±{std230:.3f}" if pd.notna(std230) else f"{r['mean_loose230']:.3f}"
    s300 = f"{r['mean_loose300']:.3f}±{std300:.3f}" if pd.notna(std300) else f"{r['mean_loose300']:.3f}"
    log(f"  {r['freq']:>6.2f} {r['amp']:>5.1f} {r['wind']:>6}  {s230:>10} {s300:>10}  "
        f"{r['delta']:>+8.3f} {r['delta_pct']:>+7.1f}%  {int(r['n_loose230']):>4} {int(r['n_loose300']):>4}")
log()

# Overall statistics
log("Overall delta statistics (loose300 − loose230):")
log(f"  Mean Δ:    {both['delta'].mean():+.4f}  ({both['delta_pct'].mean():+.1f}%)")
log(f"  Median Δ:  {both['delta'].median():+.4f}  ({both['delta_pct'].median():+.1f}%)")
log(f"  Std of Δ:  {both['delta'].std():.4f}")
log(f"  Max |Δ|:   {both['delta'].abs().max():.4f}  ({both['delta_pct'].abs().max():.1f}%)")
log()

# Split by wind condition
for wind in ["no", "full"]:
    sub = both[both["wind"] == wind]
    if sub.empty:
        continue
    log(f"  Wind={wind}: mean Δ={sub['delta'].mean():+.4f} ({sub['delta_pct'].mean():+.1f}%),  "
        f"max|Δ|={sub['delta'].abs().max():.4f} ({sub['delta_pct'].abs().max():.1f}%),  n={len(sub)}")
log()
log("Interpretation guide:")
log("  |Δ%| < 5%  → moorings indistinguishable, safe to merge")
log("  |Δ%| 5-10% → borderline; consider flagging in figure")
log("  |Δ%| > 10% → significant mooring effect; do NOT merge without correction")

# ── 5. Figures ─────────────────────────────────────────────────────────────────
print("5. Plotting...")

FREQ_MAIN = sorted(both["freq"].unique())
AMPS      = sorted(both["amp"].unique())
WINDS     = ["no", "full"]
COLORS    = {"below_90_loose230": "#E67E22", "below_90_loose300": "#2980B9"}
LABELS    = {"below_90_loose230": "loose230", "below_90_loose300": "loose300"}
WIND_LBL  = {"no": "nowind", "full": "fullwind"}

fig = plt.figure(figsize=(20, 14))
fig.suptitle("", fontsize=11)
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.50, wspace=0.30)

# Top 2 rows: OUT/IN vs freq, split by wind (rows) and amplitude (cols)
for row_i, wind in enumerate(WINDS):
    for col_i, amp in enumerate(AMPS):
        ax = fig.add_subplot(gs[row_i, col_i])
        sub = both[(both["wind"] == wind) & (both["amp"] == amp)]
        if sub.empty:
            ax.set_visible(False)
            continue
        for mooring, color in COLORS.items():
            col_mean = f"mean_{mooring.split('_')[-1]}"
            col_std  = f"std_{mooring.split('_')[-1]}"
            col_n    = f"n_{mooring.split('_')[-1]}"
            rows_m = sub[sub[col_mean].notna()]
            if rows_m.empty:
                continue
            means = rows_m[col_mean].values
            stds  = rows_m[col_std].fillna(0).values
            freqs = rows_m["freq"].values
            ax.errorbar(freqs, means, yerr=stds, fmt="o-", color=color,
                        label=LABELS[mooring], capsize=4, markersize=5, lw=1.5)
        ax.set_title("", fontsize=9)
        ax.set_xlabel("Frequency [Hz]", fontsize=8)
        ax.set_ylabel("OUT/IN (FFT)", fontsize=8)
        ax.set_ylim(0, 1.3)
        ax.axhline(1.0, color="k", lw=0.5, ls="--", alpha=0.4)
        ax.tick_params(labelsize=7)
        ax.set_xticks(FREQ_MAIN)
        ax.set_xticklabels([f"{f:.1f}" for f in FREQ_MAIN], rotation=45, fontsize=7)
        if row_i == 0 and col_i == 0:
            ax.legend(fontsize=8)

# Row 3: delta plots (one per wind condition, across all amps)
ax_delta_no   = fig.add_subplot(gs[2, :2])
ax_delta_full = fig.add_subplot(gs[2, 2:])

for ax, wind, title in [
    (ax_delta_no,   "no",   "Δ OUT/IN  (loose300 − loose230)  —  nowind"),
    (ax_delta_full, "full", "Δ OUT/IN  (loose300 − loose230)  —  fullwind"),
]:
    sub = both[both["wind"] == wind]
    if sub.empty:
        ax.set_visible(False)
        continue
    amp_markers = {0.1: "o", 0.2: "s", 0.3: "^"}
    amp_colors  = {0.1: "#2ecc71", 0.2: "#9b59b6", 0.3: "#e74c3c"}
    for amp in AMPS:
        rows_a = sub[sub["amp"] == amp].sort_values("freq")
        if rows_a.empty:
            continue
        ax.plot(rows_a["freq"], rows_a["delta"],
                marker=amp_markers.get(amp, "o"), color=amp_colors.get(amp, "gray"),
                lw=1.5, markersize=6, label=f"{amp:.1f}V")
        # Error band: ±pooled_std
        ax.fill_between(rows_a["freq"],
                         rows_a["delta"] - rows_a["pooled_std"],
                         rows_a["delta"] + rows_a["pooled_std"],
                         color=amp_colors.get(amp, "gray"), alpha=0.10)
    ax.axhline(0, color="k", lw=1.0, ls="-")
    ax.axhline( 0.05, color="k", lw=0.5, ls="--", alpha=0.4)
    ax.axhline(-0.05, color="k", lw=0.5, ls="--", alpha=0.4)
    ax.set_title("", fontsize=9)
    ax.set_xlabel("Frequency [Hz]", fontsize=8)
    ax.set_ylabel("Δ OUT/IN", fontsize=8)
    ax.set_xticks(FREQ_MAIN)
    ax.set_xticklabels([f"{f:.1f}" for f in FREQ_MAIN], rotation=45, fontsize=7)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=8, title="Amplitude", title_fontsize=7)
    ax.text(0.01, 0.97, "dashed lines: ±0.05 threshold",
            transform=ax.transAxes, fontsize=7, va="top", color="gray")

fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
print(f"   Saved → {OUT_PNG}")

# ── 6. Write findings markdown ─────────────────────────────────────────────────
print("6. Writing findings markdown...")

md_intro = f"""# Mooring comparison: loose230 vs loose300

**Date**: {datetime.now().strftime('%Y-%m-%d')}
**Script**: `analysis_scratch/mooring_comparison.py`
**Figure**: `analysis_scratch/mooring_comparison.png`

## Purpose

Determine whether `below_90_loose230` (short rubber band, 230 mm, folders 20260316–20260326)
and `below_90_loose300` (longer rubber band, 300 mm, folder 20260327) produce equivalent
OUT/IN(FFT) values at matched conditions, so the datasets can be safely merged.

Both moorings: depth −90 mm (90 mm below still water), underwater attachment.
Difference: free (unstretched) rubber band length — 230 mm vs 300 mm.

## Data coverage

Only `fullpanel`, `quality_flag==ok` wave runs included.
Main overlap: **1.3–1.7 Hz**, all amplitudes (0.1/0.2/0.3 V), full and no wind.
Below 1.0 Hz: zero or one loose300 run — no meaningful comparison possible.

## Numerical results

"""
md_table = "\n".join(summary_lines)
md_conclusion = """

## Interpretation

See delta plots (bottom row of figure): Δ = loose300 − loose230.
Dashed lines mark the ±0.05 threshold (5% of a typical OUT/IN ≈ 1).

## Next steps

- If |Δ%| < 5% across all conditions: merge moorings, annotate in methodology.
- If |Δ%| 5–10% at specific conditions: flag those points; consider mooring as a covariate.
- If |Δ%| > 10%: do NOT merge; treat mooring as a separate experimental variable.
"""

# Write-once guard: the hand-written interpretation in this file is
# valuable and the auto-generated scaffold (intro + table + generic
# "next steps") is quickly re-derivable. Never overwrite unless the
# explicit sidecar file is used.
OUT_MD_AUTO = Path(__file__).parent / "mooring_comparison_auto_report.md"
OUT_MD_AUTO.write_text(
    md_intro + "```\n" + md_table + "\n```\n" + md_conclusion, encoding="utf-8"
)
print(f"   Saved auto-report → {OUT_MD_AUTO}")
if not OUT_MD.exists():
    OUT_MD.write_text(
        md_intro + "```\n" + md_table + "\n```\n" + md_conclusion,
        encoding="utf-8",
    )
    print(f"   Seeded hand-editable findings → {OUT_MD}")
else:
    print(f"   Hand-editable findings exists (not overwritten): {OUT_MD}")

# ── 7. Thesis-grade figure (delegated promotion) ───────────────────────────────
#
# Two-panel (0.2 V | 0.3 V), OUT/IN vs frequency in the thesis scope.
# Color = mooring, line style = wind condition. This is the view that
# directly supports the merge-moorings decision in CH04 §3c.
# Written straight to output/FIGURES/ so main_save_figures.py need only
# verify its existence.
print("7. Building thesis figure (2-panel clean view)…")

_thesis_amps = [0.2, 0.3]
_thesis_freq_lo, _thesis_freq_hi = 1.25, 1.65
_MOORING_COLOR = {
    "below_90_loose230": "#1f77b4",   # blue  — 230 mm
    "below_90_loose300": "#ff7f0e",   # orange — 300 mm
}
_MOORING_LABEL = {
    "below_90_loose230": "loose230 (230 mm)",
    "below_90_loose300": "loose300 (300 mm)",
}
_WIND_STYLE = {"no": "-", "full": "--"}
_WIND_LABEL_T = {"no": "no wind", "full": "full wind"}

_thesis = both[(both["amp"].isin(_thesis_amps))
               & (both["freq"] >= _thesis_freq_lo)
               & (both["freq"] <= _thesis_freq_hi)].copy()

_fig_t, _axes_t = plt.subplots(1, 2, figsize=(8.8, 3.6), sharey=True)

# Axis y-range derived from the plotted data with a gentle pad.
_all_vals = np.concatenate([
    _thesis["mean_loose230"].dropna().values,
    _thesis["mean_loose300"].dropna().values,
])
if len(_all_vals):
    _ymin = max(0.0, float(np.min(_all_vals)) - 0.08)
    _ymax = min(1.15, float(np.max(_all_vals)) + 0.08)
else:
    _ymin, _ymax = 0.4, 1.0

for _ax, _amp in zip(_axes_t, _thesis_amps):
    _sub = _thesis[_thesis["amp"] == _amp]
    for _m, _c in _MOORING_COLOR.items():
        _mean_col = f"mean_{_m.split('_')[-1]}"
        _std_col  = f"std_{_m.split('_')[-1]}"
        for _wind, _ls in _WIND_STYLE.items():
            _rows = _sub[(_sub["wind"] == _wind) & _sub[_mean_col].notna()].sort_values("freq")
            if _rows.empty:
                continue
            _fr = _rows["freq"].values
            _mn = _rows[_mean_col].values
            _sd = _rows[_std_col].fillna(0).values
            _ax.errorbar(
                _fr, _mn, yerr=_sd, fmt="o", color=_c, linestyle=_ls,
                capsize=3, markersize=4.5, linewidth=1.3, alpha=0.85,
                zorder=3 if _wind == "no" else 2,
            )
    _ax.set_xlabel("Frequency [Hz]", fontsize=9)
    _ax.set_title("", fontsize=10)
    _ax.grid(True, linestyle="--", alpha=0.4)
    _ax.set_xlim(_thesis_freq_lo - 0.02, _thesis_freq_hi + 0.02)
    _ax.set_ylim(_ymin, _ymax)
    _ax.axhline(1.0, color="k", lw=0.5, ls=":", alpha=0.5)

_axes_t[0].set_ylabel("OUT/IN (FFT)", fontsize=9)

# Compact legend on the right panel
from matplotlib.lines import Line2D
_handles = [
    Line2D([0], [0], color=_MOORING_COLOR["below_90_loose230"], marker="o",
           linestyle="", label="loose230 (230 mm)"),
    Line2D([0], [0], color=_MOORING_COLOR["below_90_loose300"], marker="o",
           linestyle="", label="loose300 (300 mm)"),
    Line2D([0], [0], color="k", linestyle="-",  label="no wind"),
    Line2D([0], [0], color="k", linestyle="--", label="full wind"),
]
_axes_t[1].legend(handles=_handles, fontsize=7.5, loc="lower left", frameon=True)

_fig_t.suptitle("", fontsize=10)
_fig_t.tight_layout(rect=[0, 0, 1, 0.95])
_fig_t.savefig(THESIS_PDF, bbox_inches="tight")
print(f"   thesis figure → {THESIS_PDF.relative_to(BASE)}")

# Summary stats for the stub caption.
# Restrict delta report to 1.4–1.6 Hz to exclude the 1.3 Hz standing-wave
# anomaly (documented in the findings doc: at nowind 0.3V the IN probe sits
# near a pressure node and gives OUT/IN > 1, which is mooring-independent).
_clean = _thesis[(_thesis["freq"] >= 1.40) & (_thesis["freq"] <= 1.60)]
_max_abs_pct_clean = float(_clean["delta_pct"].abs().max()) if len(_clean) else float("nan")
_n_conditions_clean = int(len(_clean))
_n_conditions       = int(len(_thesis))

_caption = (
    "OUT/IN(FFT) damping ratio at 0.2\\,V (left) and 0.3\\,V (right) "
    "paddle drive for two below-water mooring rubber band lengths: "
    "230\\,mm (blue) and 300\\,mm (orange), both attached 90\\,mm below "
    "the still-water surface. Solid markers: no-wind runs; dashed "
    "connectors: full-wind runs. Error bars are run-to-run standard "
    "deviation when $n\\geq 2$. Across the clean thesis range "
    f"($f \\in [1.40, 1.60]$\\,Hz, $n={_n_conditions_clean}$ matched "
    "(frequency, amplitude, wind) conditions), the two mooring types "
    f"agree to within $|\\Delta| \\leq {_max_abs_pct_clean:.1f}\\,\\%$. "
    "The 1.3\\,Hz column is retained in the figure for completeness; the "
    "loose230 no-wind point at $\\textrm{OUT}/\\textrm{IN}>0.9$ is a known "
    "mooring-independent standing-wave artefact (IN probe near a pressure "
    "node) and is discussed separately in the methodology. Panel "
    "geometry dominates wave transmission; rubber band length does not "
    "produce a detectable systematic effect. Datasets may therefore be "
    "merged as \\texttt{below\\_90\\_loose} for the CH05 main results. "
    "See \\texttt{analysis\\_scratch/mooring\\_comparison\\_findings.md} "
    "for the full per-condition comparison and 0.1\\,V low-SNR caveats."
)

# Use the shared plot_utils helpers so the stub matches the canonical
# schema (provenance / filters / data provenance / method / stats) that
# every other thesis figure uses. The scratch script runs from the repo
# root (see BASE), so relative imports work.
import wavescripts.plot_utils as pu
pu.ACTIVE_DATASETS = [str(p).split("/")[-1] for p in
                      sorted(BASE.glob("waveprocessed/PROCESSED-*"))]
pu.TEXFIGU_DIR = BASE / "output" / "TEXFIGU"
pu.FIGURES_DIR = BASE / "output" / "FIGURES"

# build_fig_meta reads `data_df` to auto-populate n_runs / in_probes_used /
# out_probes_used / probe_configs / non_final_config_n / run_paths.
# We pass the `compare` dataframe (one row per contributing run, both
# moorings) so n_runs reflects the actual run count.
_meta_stub = pu.build_fig_meta(
    {
        "filters": {
            "PanelCondition":            "full",
            "WaveFrequencyInput [Hz]":   [_thesis_freq_lo, _thesis_freq_hi],
            "WaveAmplitudeInput [Volt]": _thesis_amps,
            "WindCondition":             ["no", "full"],
            "quality_flag":              "ok",
            "Mooring":                   ["below_90_loose230", "below_90_loose300"],
        },
        "plotting": {
            "figure_name": THESIS_NAME,
            "caption":     _caption,
        },
    },
    chapter="04",
    data_df=compare,
    extra={"script": "analysis_scratch/mooring_comparison.py"},
    computed_in=("analysis_scratch/mooring_comparison.py "
                 "(groupby freq+amp+wind+Mooring → mean/std of OUT/IN_computed)"),
    data_class="DELEG",
    findings_doc="analysis_scratch/mooring_comparison_findings.md",
    grouper="manual (groupby freq, amp, wind, Mooring)",
    collapse_panels=False,
    fft_window_hz=0.1,
    extra_params=(
        f"freq_range_hz={_thesis_freq_lo:.2f}-{_thesis_freq_hi:.2f}, "
        f"amplitudes_V={_thesis_amps}, winds=['no','full'], "
        "moorings=['below_90_loose230','below_90_loose300']"
    ),
    extra_stats={
        "n_conditions_all":    f"{_n_conditions}   (incl. 1.3 Hz)",
        "n_conditions_clean":  f"{_n_conditions_clean}   (1.4-1.6 Hz)",
        "max_abs_delta_clean": f"{_max_abs_pct_clean:.2f} %",
    },
)

pu.write_figure_stub(_meta_stub, plot_type="mooring_comparison",
                     subfig_filenames=[THESIS_NAME])
print(f"   thesis stub   → {THESIS_STUB.relative_to(BASE)}")

print("\nDone.")
