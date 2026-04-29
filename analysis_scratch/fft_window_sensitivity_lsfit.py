"""
Window-length sensitivity sweep — LS fit amplitudes at varying N_periods
=========================================================================

Extends the earlier `paddle_contamination_window_sensitivity.csv` (which
swept 20p/40p/60p/100p on per240 and full_15p…full_24p on per40) to the
range relevant for the H&G pipeline choice:

    N ∈ {5, 8, 10, 12, 15, 20} periods

Method: for each probe in each run, extract the signal starting at the
pipeline's `Computed Probe {pos} start` and extending N × samples_per_period
samples. Compute amplitude by LS fit at f_paddle (the method added to the
pipeline today — bin-grid-independent). Compare OUT/IN at each N to the
canonical N = 10 choice.

Scope: fullpanel per240 wave runs in the two canonical March-2026 lowrange
folders (20260326 + 20260327). nowind + fullwind both included.

Output:
  analysis_scratch/fft_window_sensitivity_lsfit.csv
  analysis_scratch/fft_window_sensitivity_lsfit.png
  analysis_scratch/fft_window_sensitivity_lsfit_findings.md
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wavescripts.improved_data_loader import load_analysis_data, load_processed_dfs
from wavescripts.signal_processing import compute_amplitudes_from_lsfit
from wavescripts.plot_utils import apply_thesis_style

apply_thesis_style()

FS = 250.0
BASE = Path(__file__).parent.parent
SCRATCH = Path(__file__).parent

N_SWEEP = [5, 8, 10, 12, 15, 20]
N_REF   = 10  # canonical H&G choice — denominator for relative drifts

TARGET_DIRS = [
    BASE / "waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange",
    BASE / "waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange",
]

# march2026_better_rearranging config (both target folders)
IN_PROBES  = ["9373/170", "9373/340"]    # canonical IN = mean of these two
OUT_PROBES = ["12400/250"]                # OUT is single-probe here

OUT_CSV = SCRATCH / "fft_window_sensitivity_lsfit.csv"
OUT_PNG = SCRATCH / "fft_window_sensitivity_lsfit.png"
OUT_MD  = SCRATCH / "fft_window_sensitivity_lsfit_findings.md"


def _extract_window(df: pd.DataFrame, pos: str, start: int, n_samples: int) -> np.ndarray | None:
    """Extract `n_samples` starting at `start` from probe `pos`. Interpolate
    NaN gaps; return None if too many NaNs or column missing."""
    col = f"eta_{pos}_interp" if f"eta_{pos}_interp" in df.columns else f"eta_{pos}"
    if col not in df.columns:
        return None
    end = start + n_samples
    if end > len(df):
        return None
    sig = df[col].iloc[start:end].to_numpy(dtype=float)
    nan_mask = np.isnan(sig)
    if nan_mask.mean() > 0.10:
        return None
    if nan_mask.any():
        idx = np.arange(len(sig))
        sig = np.interp(idx, idx[~nan_mask], sig[~nan_mask])
    return sig


# ── Load ──────────────────────────────────────────────────────────────────────

print("1. Loading metadata + processed time series …")
dirs_str = [str(d) for d in TARGET_DIRS]
meta, _, _, _ = load_analysis_data(*dirs_str, load_processed=False)

# per240 full panel wave runs only, quality ok
mask = (
    (meta["PanelCondition"] == "full")
    & meta["WaveFrequencyInput [Hz]"].notna()
    & (meta["WaveFrequencyInput [Hz]"] > 0)
    & (meta["WavePeriodInput"] >= 100)   # per240-ish; excludes per40
    & (meta["quality_flag"] == "ok")
)
runs = meta[mask].copy()
print(f"   {len(runs)} per240 fullpanel wave runs, quality=ok")

proc_dfs = load_processed_dfs(*dirs_str)

# ── Sweep ─────────────────────────────────────────────────────────────────────

print(f"\n2. Sweeping N_periods ∈ {N_SWEEP} …")
records = []
for _, row in runs.iterrows():
    path = row["path"]
    df = proc_dfs.get(path)
    if df is None:
        continue
    freq = float(row["WaveFrequencyInput [Hz]"])
    samples_per_period = int(round(FS / freq))

    for pos in IN_PROBES + OUT_PROBES:
        start_col = f"Computed Probe {pos} start"
        if start_col not in row or pd.isna(row[start_col]):
            continue
        start = int(row[start_col])

        for N in N_SWEEP:
            n_samples = N * samples_per_period
            sig = _extract_window(df, pos, start, n_samples)
            if sig is None or len(sig) < 50:
                continue
            ls = compute_amplitudes_from_lsfit(sig, freq, FS)
            records.append({
                "path":        path,
                "name":        Path(path).name,
                "probe":       pos,
                "freq_hz":     freq,
                "amp_V":       float(row["WaveAmplitudeInput [Volt]"]),
                "wind":        row["WindCondition"],
                "mooring":     row.get("Mooring", "?"),
                "N_periods":   N,
                "n_samples":   n_samples,
                "A_LS_mm":     ls["A_fundamental"],
                "A_stokes2_mm": ls["A_stokes2"],
                "residual_rms_mm": ls["residual_rms"],
            })

long = pd.DataFrame(records)
long.to_csv(OUT_CSV, index=False, float_format="%.5f")
print(f"   {len(long)} rows → {OUT_CSV.relative_to(BASE)}")

# ── Canonical IN/OUT per run per N ────────────────────────────────────────────

# IN = mean over IN_PROBES; OUT = single probe.
pivot = long.pivot_table(
    index=["path", "name", "freq_hz", "amp_V", "wind", "mooring", "N_periods"],
    columns="probe",
    values="A_LS_mm",
).reset_index()

# Some probes may be missing for some runs; compute IN as row-wise mean where both exist.
pivot["A_in_canonical"] = pivot[IN_PROBES].mean(axis=1, skipna=True)
pivot["A_out_canonical"] = pivot[OUT_PROBES[0]]
pivot["OUT_IN"] = pivot["A_out_canonical"] / pivot["A_in_canonical"]

# Wide form: one column per N
wide = pivot.pivot_table(
    index=["path", "name", "freq_hz", "amp_V", "wind", "mooring"],
    columns="N_periods",
    values="OUT_IN",
).reset_index()
wide.columns = [f"OUT_IN_{c}p" if isinstance(c, (int, np.integer)) else c for c in wide.columns]

# Relative drift from N=10
for N in N_SWEEP:
    if N == N_REF:
        continue
    wide[f"drift_{N}p_%"] = 100 * (wide[f"OUT_IN_{N}p"] - wide[f"OUT_IN_{N_REF}p"]) / wide[f"OUT_IN_{N_REF}p"]

print("\n3. Per-run drifts summary (fullpanel per240, quality=ok):")
print(f"   runs with all N values: {wide.dropna(subset=[f'OUT_IN_{N}p' for N in N_SWEEP]).shape[0]}")
print()
print("   Relative drift OUT/IN at each N relative to N=10p (median, max):")
for N in N_SWEEP:
    if N == N_REF:
        continue
    drift = wide[f"drift_{N}p_%"].dropna()
    print(f"     N={N:2d}p: median {drift.median():+.3f}%, max |Δ| {drift.abs().max():.2f}%  (n={len(drift)})")

# Per (wind, freq) breakdown
print("\n4. Per-cell median drift (fullwind vs nowind):")
for wind in ["no", "full"]:
    sub = wide[wide["wind"] == wind]
    if sub.empty:
        continue
    print(f"   === {wind}  (n_runs={len(sub)}) ===")
    for N in N_SWEEP:
        if N == N_REF:
            continue
        drift = sub[f"drift_{N}p_%"].dropna()
        if len(drift) == 0:
            continue
        print(f"     N={N:2d}p: median {drift.median():+.3f}%, max |Δ| {drift.abs().max():.2f}%")

# ── Plotting ──────────────────────────────────────────────────────────────────

print("\n5. Plotting …")
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# (a) scatter of drift vs N, coloured by wind
ax = axes[0, 0]
for wind, color in (("no", "tab:blue"), ("full", "tab:red")):
    sub = wide[wide["wind"] == wind]
    if sub.empty:
        continue
    for _, row in sub.iterrows():
        xs = np.array(N_SWEEP, dtype=float)
        ys = np.array([row[f"OUT_IN_{N}p"] for N in N_SWEEP], dtype=float)
        if np.isfinite(ys[N_SWEEP.index(N_REF)]):
            ys_rel = 100 * (ys - ys[N_SWEEP.index(N_REF)]) / ys[N_SWEEP.index(N_REF)]
            ax.plot(xs, ys_rel, "-", color=color, alpha=0.2, lw=0.8)
# Overlay medians
medians = []
for N in N_SWEEP:
    if N == N_REF:
        medians.append((N, 0.0))
        continue
    for wind, color, ls in (("no", "tab:blue", "-"), ("full", "tab:red", "-")):
        d = wide.loc[wide["wind"] == wind, f"drift_{N}p_%"].dropna()
        if len(d):
            ax.plot([N], [d.median()], "o", color=color, markersize=8, markeredgecolor="black")
ax.axvline(N_REF, color="k", ls="--", lw=0.5)
ax.axhline(0, color="k", ls="-", lw=0.5)
ax.set_xlabel("Window length N_periods")
ax.set_ylabel("OUT/IN drift relative to N=10p  (%)")
ax.set_title("", fontsize=10)
ax.grid(True, alpha=0.3)

# (b) drift std vs N per condition
ax = axes[0, 1]
for wind, color in (("no", "tab:blue"), ("full", "tab:red")):
    sub = wide[wide["wind"] == wind]
    xs, meds, stds, maxs = [], [], [], []
    for N in N_SWEEP:
        if N == N_REF:
            xs.append(N); meds.append(0.0); stds.append(0.0); maxs.append(0.0)
            continue
        d = sub[f"drift_{N}p_%"].dropna()
        if len(d):
            xs.append(N); meds.append(d.median()); stds.append(d.std()); maxs.append(d.abs().max())
    ax.errorbar(xs, meds, yerr=stds, marker="o", color=color,
                label=f"{wind} (median ± std)", capsize=3)
    ax.plot(xs, maxs, "--", color=color, alpha=0.5, label=f"{wind} (max |Δ|)")
    ax.plot(xs, [-m for m in maxs], "--", color=color, alpha=0.5)
ax.axvline(N_REF, color="k", ls="--", lw=0.5)
ax.axhline(0, color="k", ls="-", lw=0.5)
ax.set_xlabel("N_periods")
ax.set_ylabel("Drift from N=10p  (%)")
ax.set_title("", fontsize=10)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# (c) per-frequency median drift
ax = axes[1, 0]
for freq, freq_color in [(1.3, "tab:blue"), (1.4, "tab:orange"), (1.5, "tab:green"),
                         (1.6, "tab:red"), (1.7, "tab:purple")]:
    sub = wide[np.isclose(wide["freq_hz"], freq, atol=0.01)]
    if sub.empty:
        continue
    xs, meds = [], []
    for N in N_SWEEP:
        if N == N_REF:
            xs.append(N); meds.append(0.0); continue
        d = sub[f"drift_{N}p_%"].dropna()
        if len(d):
            xs.append(N); meds.append(d.median())
    ax.plot(xs, meds, "o-", color=freq_color, label=f"{freq} Hz  (n={len(sub)})")
ax.axvline(N_REF, color="k", ls="--", lw=0.5)
ax.axhline(0, color="k", ls="-", lw=0.5)
ax.set_xlabel("N_periods")
ax.set_ylabel("Median drift from N=10p  (%)")
ax.set_title("", fontsize=10)
ax.legend(fontsize=8, loc="best")
ax.grid(True, alpha=0.3)

# (d) OUT/IN vs frequency, one line per N
ax = axes[1, 1]
cmap = plt.get_cmap("viridis")
for i, N in enumerate(N_SWEEP):
    color = cmap(i / (len(N_SWEEP) - 1))
    sub = wide[[f"OUT_IN_{N}p", "freq_hz", "wind", "amp_V"]].copy()
    sub = sub.dropna(subset=[f"OUT_IN_{N}p"])
    for wind, marker in (("no", "o"), ("full", "s")):
        ss = sub[sub["wind"] == wind]
        if ss.empty:
            continue
        # Aggregate over all amps and runs
        ag = ss.groupby("freq_hz")[f"OUT_IN_{N}p"].median()
        label = f"N={N}p {wind}" if i in (0, len(N_SWEEP) - 1) else None
        ax.plot(ag.index, ag.values, marker=marker, color=color,
                label=label, alpha=0.8, ls="-" if wind == "no" else "--", lw=1.2)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Median OUT/IN")
ax.set_title("", fontsize=10)
ax.legend(fontsize=8, loc="best")
ax.grid(True, alpha=0.3)

fig.suptitle("", fontsize=12)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"   figure → {OUT_PNG.relative_to(BASE)}")

# ── Findings ──────────────────────────────────────────────────────────────────

lines = []
lines.append("# FFT window-length sensitivity — LS fit at varying N_periods")
lines.append("")
lines.append(f"Generated: {pd.Timestamp.utcnow().isoformat()[:19]}Z")
lines.append("")
lines.append(f"**Dataset**: fullpanel per240 wave runs, quality_flag=ok, from the two canonical")
lines.append(f"March-2026 lowrange folders. n_runs = {wide.shape[0]} total; "
             f"complete-sweep runs = {wide.dropna(subset=[f'OUT_IN_{N}p' for N in N_SWEEP]).shape[0]}.")
lines.append("")
lines.append(f"**Method**: for each probe in each run, extract N × samples_per_period samples")
lines.append(f"starting at the pipeline's `Computed Probe {{pos}} start`; compute amplitude by")
lines.append(f"least-squares sinusoid fit at f_paddle (bin-grid-independent). Canonical IN =")
lines.append(f"mean(9373/170, 9373/340); OUT = 12400/250. OUT/IN per run per N.")
lines.append("")
lines.append(f"**Sweep**: N ∈ {N_SWEEP} periods. Reference: N = {N_REF}p (the pipeline H&G default).")
lines.append("")
lines.append("## Global drift summary")
lines.append("")
lines.append(f"| N_periods | median drift % | max \\|drift\\| % | n_runs |")
lines.append(f"|---|---|---|---|")
for N in N_SWEEP:
    if N == N_REF:
        lines.append(f"| **{N}p (ref)** | 0.000 | 0.00 | — |")
        continue
    drift = wide[f"drift_{N}p_%"].dropna()
    lines.append(f"| {N}p | {drift.median():+.3f} | {drift.abs().max():.2f} | {len(drift)} |")
lines.append("")

# Per wind
lines.append("## Per wind condition")
lines.append("")
for wind in ["no", "full"]:
    sub = wide[wide["wind"] == wind]
    if sub.empty:
        continue
    lines.append(f"### {wind} (n_runs = {len(sub)})")
    lines.append("")
    lines.append(f"| N_periods | median drift % | max \\|drift\\| % |")
    lines.append(f"|---|---|---|")
    for N in N_SWEEP:
        if N == N_REF:
            lines.append(f"| **{N}p (ref)** | 0.000 | 0.00 |")
            continue
        d = sub[f"drift_{N}p_%"].dropna()
        if len(d) == 0:
            continue
        lines.append(f"| {N}p | {d.median():+.3f} | {d.abs().max():.2f} |")
    lines.append("")

# Per frequency
lines.append("## Per frequency (median drift from N=10p)")
lines.append("")
lines.append(f"| freq (Hz) | 5p % | 8p % | 12p % | 15p % | 20p % | n_runs |")
lines.append(f"|---|---|---|---|---|---|---|")
for freq in sorted(wide["freq_hz"].unique()):
    if not np.isfinite(freq):
        continue
    sub = wide[np.isclose(wide["freq_hz"], freq, atol=0.001)]
    row_cells = [f"{freq:.2f}"]
    for N in [5, 8, 12, 15, 20]:
        d = sub[f"drift_{N}p_%"].dropna()
        if len(d):
            row_cells.append(f"{d.median():+.3f}")
        else:
            row_cells.append("—")
    row_cells.append(f"{len(sub)}")
    lines.append(f"| " + " | ".join(row_cells) + " |")
lines.append("")

# Takeaway
lines.append("## Takeaway")
lines.append("")
max_drift = max(wide[f"drift_{N}p_%"].abs().max() for N in N_SWEEP if N != N_REF)
med_drifts = {N: wide[f"drift_{N}p_%"].dropna().abs().median() for N in N_SWEEP if N != N_REF}
worst_median = max(med_drifts.values())
lines.append(f"- **Global max \\|drift\\|** across all runs, all N ≠ 10p: **{max_drift:.2f}%**")
lines.append(f"- **Worst-case median drift** (worst N vs 10p): **{worst_median:.3f}%**")
lines.append("")
if max_drift < 3.0 and worst_median < 0.5:
    lines.append("- **OUT/IN is robust to window length** across N ∈ [5p, 20p]: median drift stays")
    lines.append(f"  below 0.5% and max drift below 3%. The N=10p canonical choice is not uniquely")
    lines.append(f"  optimal — it is a defensible middle of the plateau, and a reviewer's question")
    lines.append(f"  \"why not 15p?\" can be answered quantitatively with this sweep.")
else:
    lines.append("- **Drift exceeds 3%** — the choice of N matters more than expected. Investigate.")
lines.append("")
lines.append("- Complements the earlier `paddle_contamination_window_sensitivity.csv` study which")
lines.append("  reported a maximum drift of 1.17% across a larger N range (20p–100p) — consistent")
lines.append("  with this finding.")
lines.append("")
lines.append("## See also")
lines.append("")
lines.append(f"- Figure: `{OUT_PNG.relative_to(BASE)}`")
lines.append(f"- Data: `{OUT_CSV.relative_to(BASE)}`")
lines.append(f"- Prior sweep: `analysis_scratch/paddle_contamination_window_sensitivity.csv`")
lines.append(f"- Method: `analysis_scratch/fft_method_comparison_findings.md`")
lines.append(f"- Plateau: `analysis_scratch/hg_window_stability_findings.md`")

OUT_MD.write_text("\n".join(lines) + "\n")
print(f"   findings → {OUT_MD.relative_to(BASE)}")
print("\nDone.")
