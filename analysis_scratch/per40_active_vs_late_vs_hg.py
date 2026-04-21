"""
Per40 active- vs late-window AFFT, compared against per240 H&G gold
=====================================================================

Follow-up to `hg_window_stability_with_per40.py`, which revealed that
the pipeline SNARVEI window sits on a per-run-noisy transient for per40
OUT (CV ≈ 2–11 % vs < 1 % for per240). This script tests which parts
of the per40 signal, if any, are as trustworthy as the H&G gold
standard.

Three 6T windows, all integer-T (no leakage), all from wavemaker start:

    per40_active   = [32T, 38T]   — 2T before paddle stops at 40T.
                                    Deep inside the paddle-active plateau.
    per40_late     = [42T, 48T]   — 2T after paddle-stop. Late arrivals +
                                    early decay at both probes.
    per240_HG_6T   = [52T, 58T]   — H&G gold-standard subset at equal
                                    length for fair comparison.
    per240_HG_10T  = [50T, 60T]   — H&G canonical (reference only).

All three 6T windows are verified inside the user's SNARVEI eyeballing
at every thesis frequency (see snarvei_eyeballing.md).

Scope:
    freq  in {1.3, 1.4, 1.5} Hz     (1.6 excluded — noisier data)
    amp   in {0.1, 0.2, 0.3} V
    wind  in {no, full}
    panel  = full, mooring = below_90_loose (230+300 merged)
    quality_flag = "ok" only (probe_malfunction_secondary excluded to
                   isolate the window effect from data-quality effects).

Run from repo root:
    /opt/anaconda3/envs/draumkvedet/bin/python analysis_scratch/per40_active_vs_late_vs_hg.py

Outputs (scratch only):
    analysis_scratch/per40_active_vs_late_vs_hg.pdf
    analysis_scratch/per40_active_vs_late_vs_hg_summary.csv
    analysis_scratch/per40_active_vs_late_vs_hg_findings.md
"""

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
os.chdir(BASE)

from wavescripts.improved_data_loader import load_analysis_data, load_processed_dfs

# ── Config ────────────────────────────────────────────────────────────────────
FS              = 250.0
FFT_BAND_HZ     = 0.05
L_PERIODS       = 6              # window length in paddle periods (common)
ACTIVE_START_T  = 32             # per40_active = [32T, 38T]
LATE_START_T    = 42             # per40_late   = [42T, 48T]
HG_START_6T_T   = 52             # per240_HG_6T = [52T, 58T]
HG_START_10T_T  = 50             # per240_HG_10T canonical = [50T, 60T]
HG_END_10T_T    = 60

PER240_THRESHOLD_T = 50           # WavePeriodInput >= 50 → per240 pool

IN_PROBES = ["9373/170", "9373/340"]   # canonical IN mean
OUT_PROBE = "12400/250"

FREQS = [1.3, 1.4, 1.5]           # thesis scope minus 1.6 (noisy data)
AMPS  = [0.1, 0.2, 0.3]
WINDS = ["no", "full"]

PROCESSED_DIRS = [
    Path("waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange"),
    Path("waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"),
    Path("waveprocessed/PROCESSED-20260307-ProbPos4_31_FPV_2-tett6roof"),
    Path("waveprocessed/PROCESSED-20260312-ProbPos4_31_FPV_2-tett6roof"),
    Path("waveprocessed/PROCESSED-20260313-ProbePos4_31_FPV_2-tett6roof"),
    Path("waveprocessed/PROCESSED-20260314-ProbePos4_31_FPV_2-tett6roof"),
    Path("waveprocessed/PROCESSED-20260316-ProbePos4_31_FPV_2-tett6roof"),
    Path("waveprocessed/PROCESSED-20260316-ProbePos4_31_FPV_2-tett6roof-under9Mooring"),
    Path("waveprocessed/PROCESSED-20260319-ProbePos4_31_FPV_2-tett6roof-under9Mooring"),
    Path("waveprocessed/PROCESSED-20260321-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-RENAMED"),
    Path("waveprocessed/PROCESSED-20260323-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100"),
    Path("waveprocessed/PROCESSED-20260324-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100"),
    Path("waveprocessed/PROCESSED-20260325-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100"),
    Path("waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100"),
]

SCRATCH_PDF = Path(__file__).parent / "per40_active_vs_late_vs_hg.pdf"
SCRATCH_CSV = Path(__file__).parent / "per40_active_vs_late_vs_hg_summary.csv"
SCRATCH_MD  = Path(__file__).parent / "per40_active_vs_late_vs_hg_findings.md"


# ── FFT helpers (lifted from huseby_grue_window.py) ──────────────────────────
def fft_amp_at_freq(segment: np.ndarray, target_hz: float,
                    fs: float = FS, band_hz: float = FFT_BAND_HZ) -> float:
    """Peak-bin AFFT, pipeline convention. 2·|FFT|/N."""
    N = len(segment)
    if N < 4:
        return np.nan
    seg = np.asarray(segment, dtype=float).copy()
    if np.isnan(seg).any():
        idx  = np.arange(N)
        good = ~np.isnan(seg)
        if good.sum() < N * 0.9:
            return np.nan
        seg = np.interp(idx, idx[good], seg[good])
    freqs  = np.fft.fftfreq(N, d=1.0 / fs)
    pos    = freqs > 0
    pos_f  = freqs[pos]
    mask   = (pos_f >= target_hz - band_hz) & (pos_f <= target_hz + band_hz)
    if not mask.any():
        j = int(np.argmin(np.abs(pos_f - target_hz)))
    else:
        local_idx = int(np.argmin(np.abs(pos_f[mask] - target_hz)))
        j = np.where(mask)[0][local_idx]
    fft_vals = np.fft.fft(seg)
    amps_pos = 2.0 * np.abs(fft_vals[pos]) / N
    return float(amps_pos[j])


def get_eta(df_run: pd.DataFrame, pos: str) -> np.ndarray | None:
    for col in (f"eta_{pos}_interp", f"eta_{pos}"):
        if col in df_run.columns:
            return df_run[col].to_numpy(dtype=float)
    return None


def canonical_in_signal(df_run: pd.DataFrame) -> np.ndarray | None:
    sigs = []
    for p in IN_PROBES:
        s = get_eta(df_run, p)
        if s is not None:
            sigs.append(s)
    if not sigs:
        return None
    nmin = min(len(s) for s in sigs)
    stack = np.vstack([s[:nmin] for s in sigs])
    return np.nanmean(stack, axis=0)


def window_afft(signal: np.ndarray, start_T: int, length_T: int,
                samples_per_period: int, f_paddle: float) -> float:
    """AFFT in [start_T · samples_per_period : (start_T + length_T) · samples_per_period]."""
    i0 = start_T * samples_per_period
    i1 = (start_T + length_T) * samples_per_period
    if i1 > len(signal):
        return np.nan
    return fft_amp_at_freq(signal[i0:i1], f_paddle)


# ── Load data ────────────────────────────────────────────────────────────────
print("1. Loading meta + processed_dfs (14 datasets)…")
combined_meta, _, _, _ = load_analysis_data(*PROCESSED_DIRS, load_processed=False)
combined_meta["Mooring"] = combined_meta["Mooring"].replace({
    "below_90_loose230": "below_90_loose",
    "below_90_loose300": "below_90_loose",
})
wave = combined_meta[
    combined_meta["WaveFrequencyInput [Hz]"].notna()
    & (combined_meta["WaveFrequencyInput [Hz]"] > 0)
    & (combined_meta["PanelCondition"] == "full")
    & (combined_meta["Mooring"] == "below_90_loose")
    & (combined_meta["quality_flag"] == "ok")
].copy()
wave["N_input_periods"] = wave["WavePeriodInput"].astype(float)
wave = wave[wave["WaveFrequencyInput [Hz]"].round(2).isin(FREQS)
            & wave["WaveAmplitudeInput [Volt]"].round(2).isin(AMPS)
            & wave["WindCondition"].isin(WINDS)]
wave["run_type"] = np.where(wave["N_input_periods"] >= PER240_THRESHOLD_T, "per240", "per40")
print(f"   scope rows: {len(wave)} "
      f"(per240 n={int((wave['run_type']=='per240').sum())}, "
      f"per40 n={int((wave['run_type']=='per40').sum())})")

print("\n2. Loading processed_dfs …")
processed_dfs = load_processed_dfs(*PROCESSED_DIRS)
print(f"   {len(processed_dfs)} time-series cached")


# ── Compute per-run AFFT in each window ──────────────────────────────────────
print("\n3. Computing per-run AFFTs in 4 windows …")

rows = []
for _, r in wave.iterrows():
    path = r["path"]
    df_run = processed_dfs.get(path)
    if df_run is None:
        continue
    f_paddle = float(r["WaveFrequencyInput [Hz]"])
    samples_per_period = int(round(FS / f_paddle))

    sig_in  = canonical_in_signal(df_run)
    sig_out = get_eta(df_run, OUT_PROBE)
    if sig_in is None or sig_out is None:
        continue

    row = {
        "path":     path,
        "run_type": r["run_type"],
        "freq_hz":  round(f_paddle, 2),
        "amp_V":    round(float(r["WaveAmplitudeInput [Volt]"]), 2),
        "wind":     r["WindCondition"],
        "f_hz":     f_paddle,
    }

    # per40_active = [32T, 38T]
    row["A_in_active"]  = window_afft(sig_in,  ACTIVE_START_T, L_PERIODS, samples_per_period, f_paddle)
    row["A_out_active"] = window_afft(sig_out, ACTIVE_START_T, L_PERIODS, samples_per_period, f_paddle)

    # per40_late   = [42T, 48T]
    row["A_in_late"]    = window_afft(sig_in,  LATE_START_T,   L_PERIODS, samples_per_period, f_paddle)
    row["A_out_late"]   = window_afft(sig_out, LATE_START_T,   L_PERIODS, samples_per_period, f_paddle)

    # per240_HG_6T = [52T, 58T]
    row["A_in_hg6"]     = window_afft(sig_in,  HG_START_6T_T,  L_PERIODS, samples_per_period, f_paddle)
    row["A_out_hg6"]    = window_afft(sig_out, HG_START_6T_T,  L_PERIODS, samples_per_period, f_paddle)

    # per240_HG_10T canonical
    row["A_in_hg10"]    = window_afft(sig_in,  HG_START_10T_T, HG_END_10T_T - HG_START_10T_T, samples_per_period, f_paddle)
    row["A_out_hg10"]   = window_afft(sig_out, HG_START_10T_T, HG_END_10T_T - HG_START_10T_T, samples_per_period, f_paddle)

    # Per-window OUT/IN
    for tag in ("active", "late", "hg6", "hg10"):
        a_in  = row[f"A_in_{tag}"]
        a_out = row[f"A_out_{tag}"]
        row[f"OUT_IN_{tag}"] = a_out / a_in if (a_in and np.isfinite(a_in) and a_in > 0) else np.nan

    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(SCRATCH_CSV, index=False)
print(f"   per-run table → {SCRATCH_CSV.relative_to(BASE)}")


# ── Aggregate per (freq, amp, wind, run_type) ────────────────────────────────
agg_funcs = {}
for quantity in ("A_in", "A_out", "OUT_IN"):
    for tag in ("active", "late", "hg6", "hg10"):
        agg_funcs[f"{quantity}_{tag}"] = ["mean", "std", "count"]

grouped = df.groupby(["freq_hz", "amp_V", "wind", "run_type"])
agg = grouped.agg(agg_funcs)
agg.columns = [f"{col[0]}__{col[1]}" for col in agg.columns]
agg = agg.reset_index()

# For plotting: per (freq, amp, wind) take:
#   per40 runs → active + late AFFTs
#   per240 runs → hg6 + hg10 AFFTs
# Build a tidy table: one row per (freq, amp, wind, zone).
zone_rows = []
for (freq, amp, wind, rt), grp in grouped:
    if rt == "per40":
        for zone in ("active", "late"):
            zone_rows.append({
                "freq_hz": freq, "amp_V": amp, "wind": wind, "zone": zone, "n_runs": len(grp),
                "A_in_mean":  float(grp[f"A_in_{zone}" ].mean()),
                "A_in_std":   float(grp[f"A_in_{zone}" ].std(ddof=0)) if len(grp) > 1 else 0.0,
                "A_out_mean": float(grp[f"A_out_{zone}"].mean()),
                "A_out_std":  float(grp[f"A_out_{zone}"].std(ddof=0))  if len(grp) > 1 else 0.0,
                "OUT_IN_mean": float(grp[f"OUT_IN_{zone}"].mean()),
                "OUT_IN_std":  float(grp[f"OUT_IN_{zone}"].std(ddof=0)) if len(grp) > 1 else 0.0,
            })
    else:  # per240
        for zone in ("hg6", "hg10"):
            zone_rows.append({
                "freq_hz": freq, "amp_V": amp, "wind": wind, "zone": zone, "n_runs": len(grp),
                "A_in_mean":  float(grp[f"A_in_{zone}" ].mean()),
                "A_in_std":   float(grp[f"A_in_{zone}" ].std(ddof=0)) if len(grp) > 1 else 0.0,
                "A_out_mean": float(grp[f"A_out_{zone}"].mean()),
                "A_out_std":  float(grp[f"A_out_{zone}"].std(ddof=0))  if len(grp) > 1 else 0.0,
                "OUT_IN_mean": float(grp[f"OUT_IN_{zone}"].mean()),
                "OUT_IN_std":  float(grp[f"OUT_IN_{zone}"].std(ddof=0)) if len(grp) > 1 else 0.0,
            })
zone_df = pd.DataFrame(zone_rows)


# ── Compute per40_active_vs_HG and per40_late_vs_HG ratios ───────────────────
ref = (zone_df[zone_df["zone"] == "hg6"]
        .set_index(["freq_hz", "amp_V", "wind"])[["A_in_mean", "A_out_mean", "OUT_IN_mean"]])

ratios = []
for _, r in zone_df.iterrows():
    if r["zone"] not in ("active", "late"):
        continue
    key = (r["freq_hz"], r["amp_V"], r["wind"])
    if key not in ref.index:
        continue
    ref_row = ref.loc[key]
    ratios.append({
        "freq_hz": r["freq_hz"], "amp_V": r["amp_V"], "wind": r["wind"], "zone": r["zone"],
        "ratio_A_in":  r["A_in_mean"]  / ref_row["A_in_mean"]  if ref_row["A_in_mean"]  > 0 else np.nan,
        "ratio_A_out": r["A_out_mean"] / ref_row["A_out_mean"] if ref_row["A_out_mean"] > 0 else np.nan,
        "ratio_OUTIN": r["OUT_IN_mean"] / ref_row["OUT_IN_mean"] if ref_row["OUT_IN_mean"] > 0 else np.nan,
    })
ratio_df = pd.DataFrame(ratios)

# Headline numbers
active_rows = ratio_df[ratio_df["zone"] == "active"]
late_rows   = ratio_df[ratio_df["zone"] == "late"]

def _summary(rdf, col):
    v = rdf[col].dropna()
    if v.empty:
        return np.nan, np.nan, np.nan
    return float(v.median()), float((v - 1).abs().median()), float((v - 1).abs().max())

print("\n4. Ratios (per40_zone_AFFT / per240_HG6T_AFFT): median, median |Δ|, max |Δ|")
print("    (zone × quantity)")
for zone_name, sub in [("active", active_rows), ("late", late_rows)]:
    for col in ("ratio_A_in", "ratio_A_out", "ratio_OUTIN"):
        med, abs_med, abs_max = _summary(sub, col)
        print(f"    {zone_name:6s}  {col:13s}  median={med:.4f}  |med|={abs_med:.4f}  |max|={abs_max:.4f}")


# ── Plot ─────────────────────────────────────────────────────────────────────
print("\n5. Plotting …")
fig, axes = plt.subplots(2, 3, figsize=(14, 8), dpi=120, sharey="row")

WIND_COLOR = {"no": "#2E86AB", "full": "#E74C3C"}
AMP_MARKER = {0.1: "o", 0.2: "s", 0.3: "^"}

TITLES = {
    "ratio_A_in":  "IN AFFT ratio  (per40 / per240 H&G 6T)",
    "ratio_A_out": "OUT AFFT ratio (per40 / per240 H&G 6T)",
    "ratio_OUTIN": "OUT/IN ratio   (per40 / per240 H&G 6T)",
}

for row_idx, zone_name in enumerate(["active", "late"]):
    sub = ratio_df[ratio_df["zone"] == zone_name]
    for col_idx, q in enumerate(["ratio_A_in", "ratio_A_out", "ratio_OUTIN"]):
        ax = axes[row_idx, col_idx]
        for amp in AMPS:
            for wind in WINDS:
                cell = sub[np.isclose(sub["amp_V"], amp) & (sub["wind"] == wind)].sort_values("freq_hz")
                if cell.empty:
                    continue
                ax.plot(
                    cell["freq_hz"], cell[q],
                    marker=AMP_MARKER[amp], color=WIND_COLOR[wind],
                    mfc=WIND_COLOR[wind] if wind == "no" else "white",
                    mec=WIND_COLOR[wind], markersize=9, markeredgewidth=1.4,
                    linestyle="-", linewidth=1.0, alpha=0.92,
                    label=f"{amp:.2f} V · {wind}" if (row_idx == 0 and col_idx == 0) else None,
                )
        ax.axhline(1.0, color="black", lw=0.7, ls="--", alpha=0.5)
        ax.axhline(0.95, color="gray",  lw=0.5, ls=":", alpha=0.4)
        ax.axhline(1.05, color="gray",  lw=0.5, ls=":", alpha=0.4)
        ax.set_xticks(FREQS)
        ax.set_xlim(min(FREQS) - 0.05, max(FREQS) + 0.05)
        ax.grid(True, alpha=0.3)
        if row_idx == 1:
            ax.set_xlabel("frequency [Hz]", fontsize=9)
        if col_idx == 0:
            ax.set_ylabel(f"{'ACTIVE' if zone_name == 'active' else 'LATE'}\nper40 / H&G", fontsize=10)
        if row_idx == 0:
            ax.set_title(TITLES[q], fontsize=10)

# Legend only on (0, 0)
axes[0, 0].legend(fontsize=7, loc="best", framealpha=0.9, ncol=2)

fig.suptitle(
    "Per40 active [32T, 38T]  vs  per40 late [42T, 48T]  "
    "vs  per240 H&G 6T [52T, 58T]  "
    "(dashed = perfect agreement; grey dots = ±5 %)",
    fontsize=11, fontweight="bold", y=0.99,
)
fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.09, hspace=0.22, wspace=0.12)
fig.savefig(SCRATCH_PDF, bbox_inches="tight")
print(f"   → {SCRATCH_PDF.relative_to(BASE)}")
plt.close(fig)


# ── Findings markdown (run log only) ────────────────────────────────────────
print("\n6. Writing findings markdown …")
lines = [
    "# Per40 zones vs per240 H&G — run log",
    "",
    "Generated by `analysis_scratch/per40_active_vs_late_vs_hg.py`.  ",
    f"Scope: full panel, below_90_loose, quality_flag='ok' only, thesis freqs "
    f"{FREQS} Hz, amps {AMPS} V, winds {WINDS}.",
    "",
    "## Windows",
    "",
    "| zone         | periods      | no-leakage | mode |",
    "|--------------|--------------|------------|------|",
    f"| per40_active | [32T, 38T] = 6T | ✓ | paddle-active plateau |",
    f"| per40_late   | [42T, 48T] = 6T | ✓ | post-paddle-stop (late arrivals + early decay) |",
    f"| per240_HG_6T | [52T, 58T] = 6T | ✓ | H&G gold-standard subset, length-matched |",
    f"| per240_HG_10T| [50T, 60T] = 10T | ✓ | H&G canonical, reference only |",
    "",
    "## Cohort",
    "",
    f"- per240 runs: {int((wave['run_type']=='per240').sum())}",
    f"- per40  runs: {int((wave['run_type']=='per40').sum())}",
    f"- quality_flag strictly 'ok' (probe_malfunction_secondary excluded).",
    "",
    "## Ratios (per40_zone / per240_HG_6T) — median across all (freq, amp, wind) cells",
    "",
    "| zone   | A_in ratio | A_out ratio | OUT/IN ratio |",
    "|--------|-----------:|------------:|-------------:|",
]
for zone_name, sub in [("active", active_rows), ("late", late_rows)]:
    mins = [_summary(sub, c)[0] for c in ("ratio_A_in", "ratio_A_out", "ratio_OUTIN")]
    lines.append(f"| {zone_name:6} | {mins[0]:.4f} | {mins[1]:.4f} | {mins[2]:.4f} |")
lines += [
    "",
    "## |Δ| (absolute deviation from 1.0) — median, max",
    "",
    "| zone   | |Δ A_in| med / max | |Δ A_out| med / max | |Δ OUT/IN| med / max |",
    "|--------|--------------------|---------------------|----------------------|",
]
for zone_name, sub in [("active", active_rows), ("late", late_rows)]:
    parts = []
    for c in ("ratio_A_in", "ratio_A_out", "ratio_OUTIN"):
        _m, absmed, absmax = _summary(sub, c)
        parts.append(f"{absmed:.4f} / {absmax:.4f}")
    lines.append(f"| {zone_name:6} | {parts[0]} | {parts[1]} | {parts[2]} |")
lines += [
    "",
    "## Per-cell detail (per40 / H&G6T ratios)",
    "",
    "```",
    ratio_df.round(4).to_string(index=False),
    "```",
    "",
    "## Files",
    "",
    f"  {SCRATCH_PDF.name}  — ratio scatter, 2 rows (active / late) × 3 cols (IN / OUT / OUT-IN)",
    f"  {SCRATCH_CSV.name}  — per-run AFFTs in all four windows",
    "",
    "Interpretation intentionally omitted — user-eyeball of the PDF "
    "decides whether per40_active is trustworthy as an H&G-equivalent.",
]
SCRATCH_MD.write_text("\n".join(lines))
print(f"   → {SCRATCH_MD.relative_to(BASE)}")

print("\nDone.")
