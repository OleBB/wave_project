"""
Window-position sensitivity — slide a 10T window across the per240 plateau
===========================================================================

Companion to `fft_window_sensitivity_lsfit.py` (which varies window LENGTH
at a fixed start). Here we vary POSITION at a fixed length (10 periods =
pipeline H&G default).

Sweep: reference start position T_ref ∈ [40T, 80T] in 1T steps.
For each probe the local start is `T_ref − ΔT_probe` where ΔT_probe is
the group-velocity travel-time shift from HG.REF_R_M (the probe-shifted
H&G convention — i.e. moving T_ref by 1T shifts every probe's window by
exactly 1T in probe-local time).

At T_ref = 50 the window equals the current pipeline default (H&G [50T, 60T]
at OUT, probe-shifted for IN). So drift is measured relative to T_ref = 50.

Scope: fullpanel per240 wave runs, quality=ok, in the two canonical
March-2026 lowrange folders.

Output:
  analysis_scratch/fft_window_position_sensitivity_lsfit.csv
  analysis_scratch/fft_window_position_sensitivity_lsfit.png
  analysis_scratch/fft_window_position_sensitivity_lsfit_findings.md
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
from wavescripts.constants import HG, c_group
from wavescripts.plot_utils import apply_thesis_style

apply_thesis_style()

FS = 250.0
BASE = Path(__file__).parent.parent
SCRATCH = Path(__file__).parent

N_PERIODS  = 10          # fixed window length (pipeline default)
T_REF_MIN  = 40
T_REF_MAX  = 80
T_REF_STEP = 1
T_REF_SWEEP = np.arange(T_REF_MIN, T_REF_MAX + 1, T_REF_STEP)

T_REF_CANON = HG.START_T_REF   # = 50 — the pipeline default

TARGET_DIRS = [
    BASE / "waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange",
    BASE / "waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange",
]

# Probe positions in metres from paddle (march2026_better_rearranging)
PROBE_R_M = {
    "9373/170":  9.373,
    "9373/340":  9.373,
    "12400/250": 12.400,
    "8804/250":   8.804,
}
IN_PROBES  = ["9373/170", "9373/340"]
OUT_PROBES = ["12400/250"]

OUT_CSV = SCRATCH / "fft_window_position_sensitivity_lsfit.csv"
OUT_PNG = SCRATCH / "fft_window_position_sensitivity_lsfit.png"
OUT_MD  = SCRATCH / "fft_window_position_sensitivity_lsfit_findings.md"


def _probe_local_start_T(T_ref: float, r_probe_m: float, f_hz: float) -> float:
    """T_ref is in OUT-probe coordinates (r=12.4 m). Shift to probe-local time
    by subtracting group-velocity travel time ΔT = (R_ref − r)/c_group · f."""
    dT = (HG.REF_R_M - r_probe_m) / c_group(f_hz, HG.TANK_DEPTH_M) * f_hz
    return T_ref - dT


def _extract_window(df, pos, start, n_samples):
    col = f"eta_{pos}_interp" if f"eta_{pos}_interp" in df.columns else f"eta_{pos}"
    if col not in df.columns:
        return None
    end = start + n_samples
    if start < 0 or end > len(df):
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

mask = (
    (meta["PanelCondition"] == "full")
    & meta["WaveFrequencyInput [Hz]"].notna()
    & (meta["WaveFrequencyInput [Hz]"] > 0)
    & (meta["WavePeriodInput"] >= 100)   # per240-ish
    & (meta["quality_flag"] == "ok")
)
runs = meta[mask].copy()
print(f"   {len(runs)} per240 fullpanel wave runs, quality=ok")

proc_dfs = load_processed_dfs(*dirs_str)

# ── Sweep ─────────────────────────────────────────────────────────────────────

print(f"\n2. Sweeping T_ref ∈ [{T_REF_MIN}, {T_REF_MAX}] periods "
      f"(step {T_REF_STEP}, {len(T_REF_SWEEP)} positions) at N = {N_PERIODS}p …")
records = []
for _, row in runs.iterrows():
    path = row["path"]
    df = proc_dfs.get(path)
    if df is None:
        continue
    freq = float(row["WaveFrequencyInput [Hz]"])
    samples_per_period = int(round(FS / freq))
    n_samples = N_PERIODS * samples_per_period

    for pos in IN_PROBES + OUT_PROBES:
        r_m = PROBE_R_M[pos]
        for T_ref in T_REF_SWEEP:
            start_T_probe = _probe_local_start_T(T_ref, r_m, freq)
            start_sample = int(round(start_T_probe * samples_per_period))
            sig = _extract_window(df, pos, start_sample, n_samples)
            if sig is None:
                continue
            ls = compute_amplitudes_from_lsfit(sig, freq, FS)
            records.append({
                "path":           path,
                "name":           Path(path).name,
                "probe":          pos,
                "freq_hz":        freq,
                "amp_V":          float(row["WaveAmplitudeInput [Volt]"]),
                "wind":           row["WindCondition"],
                "mooring":        row.get("Mooring", "?"),
                "T_ref":          int(T_ref),
                "start_sample":   start_sample,
                "A_LS_mm":        ls["A_fundamental"],
                "A_stokes2_mm":   ls["A_stokes2"],
                "residual_rms_mm": ls["residual_rms"],
            })

long = pd.DataFrame(records)
long.to_csv(OUT_CSV, index=False, float_format="%.5f")
print(f"   {len(long)} rows → {OUT_CSV.relative_to(BASE)}")

# ── Canonical IN/OUT per run per T_ref ────────────────────────────────────────

pivot = long.pivot_table(
    index=["path", "name", "freq_hz", "amp_V", "wind", "mooring", "T_ref"],
    columns="probe",
    values="A_LS_mm",
).reset_index()
pivot["A_in_canonical"] = pivot[IN_PROBES].mean(axis=1, skipna=True)
pivot["A_out_canonical"] = pivot[OUT_PROBES[0]]
pivot["OUT_IN"] = pivot["A_out_canonical"] / pivot["A_in_canonical"]

# Per-run CV across positions
cv = pivot.groupby(["name", "freq_hz", "amp_V", "wind"])["OUT_IN"].agg(
    ["median", "std", "min", "max", "count"]
).reset_index()
cv["range_pct"] = 100 * (cv["max"] - cv["min"]) / cv["median"]

print("\n3. Per-run OUT/IN variability across T_ref ∈ [40T, 80T]:")
print(f"   median range: {cv['range_pct'].median():.2f}%  |  max range: {cv['range_pct'].max():.2f}%")
print(f"   median std/median: {(100*cv['std']/cv['median']).median():.2f}%")

# Drift from T_ref = 50 (pipeline default)
wide = pivot.pivot_table(
    index=["path", "name", "freq_hz", "amp_V", "wind", "mooring"],
    columns="T_ref",
    values="OUT_IN",
).reset_index()
if T_REF_CANON in wide.columns:
    for T_ref in T_REF_SWEEP:
        if T_ref == T_REF_CANON:
            continue
        wide[f"drift_{int(T_ref)}T_%"] = 100 * (wide[int(T_ref)] - wide[T_REF_CANON]) / wide[T_REF_CANON]

print(f"\n4. Drift vs T_ref={T_REF_CANON} (pipeline default):")
for wind in ["no", "full"]:
    sub = wide[wide["wind"] == wind]
    if sub.empty:
        continue
    print(f"   === {wind}  (n_runs={len(sub)}) ===")
    for T_ref in [40, 45, 50, 55, 60, 65, 70, 75, 80]:
        if T_ref == T_REF_CANON:
            print(f"     T_ref={T_ref}T (ref): 0.00%")
            continue
        col = f"drift_{T_ref}T_%"
        if col not in sub.columns:
            continue
        d = sub[col].dropna()
        if len(d):
            print(f"     T_ref={T_ref}T: median {d.median():+.3f}%, max |Δ| {d.abs().max():.2f}%  (n={len(d)})")

# ── Plotting ──────────────────────────────────────────────────────────────────

print("\n5. Plotting …")
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# (a) per-run OUT/IN traces vs T_ref (relative to T_ref=50), coloured by wind
ax = axes[0, 0]
for wind, color in (("no", "tab:blue"), ("full", "tab:red")):
    sub = wide[wide["wind"] == wind]
    for _, r in sub.iterrows():
        xs = np.array(T_REF_SWEEP, dtype=float)
        ys = np.array([r.get(int(t), np.nan) for t in T_REF_SWEEP], dtype=float)
        ref = r.get(T_REF_CANON, np.nan)
        if not np.isfinite(ref) or ref == 0:
            continue
        ax.plot(xs, 100 * (ys - ref) / ref, "-", color=color, alpha=0.25, lw=0.7)
# medians
for wind, color in (("no", "tab:blue"), ("full", "tab:red")):
    sub = wide[wide["wind"] == wind]
    med = []
    for t in T_REF_SWEEP:
        col = t if t in sub.columns else None
        if col is None:
            med.append(np.nan); continue
        ref = sub[T_REF_CANON]
        drift = 100 * (sub[col] - ref) / ref
        med.append(drift.median())
    ax.plot(T_REF_SWEEP, med, "-", color=color, lw=2.5,
            label=f"{wind} median (n={len(sub)})")
ax.axvline(T_REF_CANON, color="k", ls="--", lw=0.8, label=f"T_ref={T_REF_CANON}T (pipeline)")
ax.axhline(0, color="k", ls="-", lw=0.5)
ax.set_xlabel("Window start T_ref (periods from wavemaker start, OUT-probe equivalent)")
ax.set_ylabel("OUT/IN drift from T_ref=50T  (%)")
ax.set_title("", fontsize=10)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# (b) per-run absolute OUT/IN traces
ax = axes[0, 1]
for wind, color in (("no", "tab:blue"), ("full", "tab:red")):
    sub = wide[wide["wind"] == wind]
    for _, r in sub.iterrows():
        xs = np.array(T_REF_SWEEP, dtype=float)
        ys = np.array([r.get(int(t), np.nan) for t in T_REF_SWEEP], dtype=float)
        ax.plot(xs, ys, "-", color=color, alpha=0.2, lw=0.7)
ax.axvline(T_REF_CANON, color="k", ls="--", lw=0.8)
ax.set_xlabel("T_ref (periods)")
ax.set_ylabel("Absolute OUT/IN")
ax.set_title("", fontsize=10)
ax.grid(True, alpha=0.3)

# (c) range (max - min) / median per run, vs wind and freq
ax = axes[1, 0]
cv["color"] = cv["wind"].map({"no": "tab:blue", "full": "tab:red", "lowest": "tab:orange"})
for wind, color in (("no", "tab:blue"), ("full", "tab:red")):
    sub = cv[cv["wind"] == wind]
    ax.scatter(sub["freq_hz"], sub["range_pct"], s=25, alpha=0.6,
               color=color, label=f"{wind} (n={len(sub)})")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("OUT/IN range across T_ref  (max − min)/median, %")
ax.set_title("", fontsize=10)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# (d) median A_in_canonical vs T_ref, per wind+freq — does the plateau exist?
ax = axes[1, 1]
for wind, color in (("no", "tab:blue"), ("full", "tab:red")):
    sub = pivot[pivot["wind"] == wind]
    med = sub.groupby("T_ref")["A_in_canonical"].median()
    ax.plot(med.index, med.values, "-", color=color, lw=1.5, label=f"IN — {wind}")
    med_out = sub.groupby("T_ref")["A_out_canonical"].median()
    ax.plot(med_out.index, med_out.values, "--", color=color, lw=1.5, label=f"OUT — {wind}")
ax.axvline(T_REF_CANON, color="k", ls="--", lw=0.8)
ax.set_xlabel("T_ref (periods)")
ax.set_ylabel("median A_canonical (mm)")
ax.set_title("", fontsize=10)
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)

fig.suptitle("", fontsize=11)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"   figure → {OUT_PNG.relative_to(BASE)}")

# ── Findings ──────────────────────────────────────────────────────────────────

lines = []
lines.append("# FFT window-position sensitivity — 10T window slid across the per240 plateau")
lines.append("")
lines.append(f"Generated: {pd.Timestamp.now('UTC').isoformat()[:19]}Z")
lines.append("")
lines.append(f"**Dataset**: fullpanel per240 wave runs, quality_flag=ok, from the two canonical")
lines.append(f"March-2026 lowrange folders. n_runs = {wide.shape[0]}.")
lines.append("")
lines.append(f"**Method**: hold window length at N = {N_PERIODS} periods (pipeline default);")
lines.append(f"sweep the start position T_ref from {T_REF_MIN}T to {T_REF_MAX}T in {T_REF_STEP}T steps.")
lines.append(f"T_ref is in OUT-probe coordinates; each probe's local start is")
lines.append(f"`T_ref − ΔT_probe` via the probe-shifted H&G convention. Reference:")
lines.append(f"**T_ref = {T_REF_CANON}T** = current pipeline default.")
lines.append("")

lines.append("## Per-run OUT/IN variability across T_ref")
lines.append("")
lines.append(f"- Median range (max − min)/median across positions: **{cv['range_pct'].median():.2f} %**")
lines.append(f"- Max range across positions (worst run): **{cv['range_pct'].max():.2f} %**")
lines.append(f"- Median std(OUT/IN) / median(OUT/IN): **{(100*cv['std']/cv['median']).median():.2f} %**")
lines.append("")

lines.append("## Median drift from T_ref=50T, per wind condition")
lines.append("")
lines.append("| T_ref | nowind median % | nowind max \\|Δ\\| % | fullwind median % | fullwind max \\|Δ\\| % |")
lines.append("|---|---|---|---|---|")
for T_ref in [40, 42, 45, 48, 50, 52, 55, 58, 60, 65, 70, 75, 80]:
    cells = [f"**{T_ref}T**" if T_ref == T_REF_CANON else f"{T_ref}T"]
    for wind in ["no", "full"]:
        sub = wide[wide["wind"] == wind]
        if T_ref == T_REF_CANON:
            cells.extend(["0.000", "0.00"])
            continue
        col = f"drift_{T_ref}T_%"
        if col not in sub.columns:
            cells.extend(["—", "—"])
            continue
        d = sub[col].dropna()
        if len(d) == 0:
            cells.extend(["—", "—"])
        else:
            cells.extend([f"{d.median():+.3f}", f"{d.abs().max():.2f}"])
    lines.append("| " + " | ".join(cells) + " |")
lines.append("")

# Flatness region
lines.append("## Where is the plateau?")
lines.append("")
cv_global = pivot.groupby("T_ref")["OUT_IN"].agg(["median", "std"]).reset_index()
cv_global["cv_pct"] = 100 * cv_global["std"] / cv_global["median"]
# Rolling std of per-position median to find flat region
med_trace = cv_global.set_index("T_ref")["median"]
lines.append("Median OUT/IN vs T_ref (all runs pooled):")
lines.append("")
lines.append("| T_ref | median OUT/IN | drift from 50T % |")
lines.append("|---|---|---|")
ref_val = med_trace.get(T_REF_CANON, np.nan)
for T_ref in [40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 65, 68, 70, 75, 80]:
    v = med_trace.get(T_ref, np.nan)
    if np.isfinite(v) and np.isfinite(ref_val) and ref_val != 0:
        d = 100 * (v - ref_val) / ref_val
        bold = "**" if T_ref == T_REF_CANON else ""
        lines.append(f"| {bold}{T_ref}T{bold} | {bold}{v:.4f}{bold} | {bold}{d:+.3f}{bold} |")
lines.append("")

lines.append("## Takeaway (interpret manually before publishing)")
lines.append("")
lines.append(f"- Per-run OUT/IN across T_ref ∈ [40, 80]T varies by median **{cv['range_pct'].median():.2f} %**")
lines.append(f"  (max − min). This is the \"how sensitive is OUT/IN to where we start the window\" answer.")
lines.append("")
lines.append("- See panel (a) of the figure for per-run drift traces vs T_ref.")
lines.append(f"  Panel (d) shows median A_in and A_out independently — the plateau regions are visible.")
lines.append("")
lines.append("- Cross-reference with `hg_window_stability_findings.md`:")
lines.append("  - per240 10T sliding AFFT CV within ±5T of H&G start was ~0.5 % (IN and OUT).")
lines.append("  - This sweep extends the ±5T to ±30T — confirms / refutes plateau beyond pipeline default.")

lines.append("")
lines.append("## See also")
lines.append("")
lines.append(f"- Figure: `{OUT_PNG.relative_to(BASE)}`")
lines.append(f"- Data: `{OUT_CSV.relative_to(BASE)}`")
lines.append(f"- Length sensitivity (companion): `analysis_scratch/fft_window_sensitivity_lsfit_findings.md`")
lines.append(f"- Plateau (prior): `analysis_scratch/hg_window_stability_findings.md`")

OUT_MD.write_text("\n".join(lines) + "\n")
print(f"   findings → {OUT_MD.relative_to(BASE)}")
print("\nDone.")
