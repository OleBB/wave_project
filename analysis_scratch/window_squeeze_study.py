"""
Window squeeze study at f = 1.6 Hz (worst-case)
================================================

Two-step calibration to set the H&G window for the proposed formula:

  Step 1 — per240 (long signal, gold standard).
           Sweep N_offset (= periods after wave-front arrival at probe)
           ∈ [5, 30] T at 1 T step, with fixed window length 10 T.
           Find the EARLIEST N_offset where OUT/IN(FFT) flattens — i.e.
           the wave train has settled into a stable plateau.

  Step 2 — per40 (short signal, the metric we care about pooling).
           Hold start at N_offset = (Step 1 result), sweep window LENGTH
           N ∈ [5, max-fitting] T at 1 T step. Find the LARGEST N where
           OUT/IN stays stable before per40 ringdown contaminates.

Both steps at f = 1.6 Hz, 0.2 V, full panel, quality ok — the highest
thesis frequency, where the per40 plateau is shortest and any window
slop hurts most. If we can squeeze it here, we can squeeze it everywhere.

Method: FFT amplitude at the paddle bin (peak-bin, ±0.05 Hz),
    A_in  = FFT[t_arr_IN + N_offset/f , length L/f] over canonical
            IN (mean of 9373/170 and 9373/340)
    A_out = same start time at IN's r, but applied at OUT — i.e. each
            probe gets its own arrival time t_arr = r / c_g(f, h).
    OUT/IN = A_out / A_in.

Group velocity from full dispersion ω² = g·k·tanh(k·h), via
wavescripts.constants.c_group.

Outputs (scratch only — not yet wired into main_save_figures):
    analysis_scratch/window_squeeze_step1_per240.{pdf,png}
    analysis_scratch/window_squeeze_step2_per40.{pdf,png}
    analysis_scratch/window_squeeze_summary.csv
    analysis_scratch/window_squeeze_findings.md
    stdout: numerical summary

Run from repo root:
    /opt/anaconda3/envs/draumkvedet/bin/python analysis_scratch/window_squeeze_study.py
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

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import os
os.chdir(BASE)

from wavescripts.improved_data_loader import load_analysis_data, load_processed_dfs
from wavescripts.plot_utils import apply_thesis_style, WIND_COLOR_MAP
from wavescripts.constants import c_group, HG

apply_thesis_style()

# ── Config ──────────────────────────────────────────────────────────────
FS               = 250.0
FFT_BAND_HZ      = 0.05

TARGET_FREQ      = 1.6      # worst-case thesis frequency
TARGET_AMP       = 0.2

IN_PROBES        = ["9373/170", "9373/340"]    # canonical IN = mean
OUT_PROBE        = "12400/250"
PROBE_R_M        = {"9373/170": 9.373, "9373/340": 9.373, "12400/250": 12.400}
TANK_DEPTH_M     = HG.TANK_DEPTH_M

WINDOW_PERIODS_FIXED = 10   # Step 1: keep window length = 10 T while sweeping start

# Step 1 — sweep start offset relative to wave-front arrival at each probe
N_OFFSET_MIN_T   = 5
N_OFFSET_MAX_T   = 30        # well past pipeline default of 15
N_OFFSET_STEP_T  = 1

# Step 2 — sweep window length, keep start fixed at Step-1 winner
N_LENGTH_MIN_T   = 5
N_LENGTH_MAX_T   = 30        # script will truncate at per40 signal end
N_LENGTH_STEP_T  = 1

# Stability criterion — ±1 % drift from the curve's flat-region median
STABILITY_PCT    = 1.0

# Canon — march-2026 cond4 lowrange
PROCESSED_DIRS = [
    Path("waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange"),
    Path("waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"),
]

SCRATCH_DIR = Path(__file__).parent
OUT_PNG_S1  = SCRATCH_DIR / "window_squeeze_step1_per240.png"
OUT_PDF_S1  = SCRATCH_DIR / "window_squeeze_step1_per240.pdf"
OUT_PNG_S2  = SCRATCH_DIR / "window_squeeze_step2_per40.png"
OUT_PDF_S2  = SCRATCH_DIR / "window_squeeze_step2_per40.pdf"
OUT_CSV     = SCRATCH_DIR / "window_squeeze_summary.csv"
OUT_MD      = SCRATCH_DIR / "window_squeeze_findings.md"


# ── Helpers ─────────────────────────────────────────────────────────────
def fft_amp_at_freq(segment: np.ndarray, target_hz: float = TARGET_FREQ,
                    fs: float = FS, band_hz: float = FFT_BAND_HZ) -> float:
    """Peak-bin AFFT, pipeline convention (2·|FFT|/N)."""
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
    freqs = np.fft.fftfreq(N, d=1.0 / fs)
    pos   = freqs > 0
    pos_f = freqs[pos]
    mask  = (pos_f >= target_hz - band_hz) & (pos_f <= target_hz + band_hz)
    if not mask.any():
        j = int(np.argmin(np.abs(pos_f - target_hz)))
    else:
        local_idx = int(np.argmin(np.abs(pos_f[mask] - target_hz)))
        j = np.where(mask)[0][local_idx]
    fft_vals = np.fft.fft(seg)
    amps_pos = 2.0 * np.abs(fft_vals[pos]) / N
    return float(amps_pos[j])


def get_eta(df_run: pd.DataFrame, pos: str):
    for col in (f"eta_{pos}_interp", f"eta_{pos}"):
        if col in df_run.columns:
            return df_run[col].to_numpy(dtype=float)
    return None


def canonical_in_signal(df_run: pd.DataFrame):
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


def t_arrive_at_probe(probe: str, f_paddle: float = TARGET_FREQ) -> float:
    """Wave-front arrival time at a probe in seconds."""
    r_m = PROBE_R_M[probe]
    cg  = c_group(f_paddle, TANK_DEPTH_M)
    return r_m / cg


def amp_in_window(signal: np.ndarray, start_s: float, length_s: float,
                  fs: float = FS) -> float:
    """FFT amplitude at TARGET_FREQ over [start_s, start_s + length_s]."""
    s_idx = int(round(start_s * fs))
    e_idx = s_idx + int(round(length_s * fs))
    if s_idx < 0 or e_idx > len(signal):
        return np.nan
    return fft_amp_at_freq(signal[s_idx:e_idx])


def stable_threshold_first_index(values: np.ndarray, ref_value: float,
                                  pct: float = STABILITY_PCT) -> int | None:
    """Return the first index `i` such that for all j >= i, abs drift from
    ref_value is < pct%. Returns None if no such i exists."""
    if not np.isfinite(ref_value) or ref_value == 0:
        return None
    rel = (values - ref_value) / ref_value * 100.0
    ok = np.abs(rel) < pct
    if not ok.any():
        return None
    # Walk back from the end: longest tail of all-True
    for i in range(len(ok) - 1, -1, -1):
        if not ok[i]:
            return i + 1 if (i + 1) < len(ok) else None
    return 0


# ── Load ────────────────────────────────────────────────────────────────
print("1. Loading meta + processed_dfs (canon March-2026 lowrange) …")
combined_meta, _, _, _ = load_analysis_data(*PROCESSED_DIRS, load_processed=False)
processed_dfs = load_processed_dfs(*PROCESSED_DIRS)
print(f"   meta: {len(combined_meta)} rows · processed_dfs: {len(processed_dfs)} runs")


def select_canon() -> pd.DataFrame:
    mask = (
        np.isclose(pd.to_numeric(combined_meta["WaveFrequencyInput [Hz]"], errors="coerce"),
                   TARGET_FREQ, atol=0.02)
        & np.isclose(pd.to_numeric(combined_meta["WaveAmplitudeInput [Volt]"], errors="coerce"),
                     TARGET_AMP, atol=0.01)
        & (combined_meta["PanelCondition"] == "full")
        & (combined_meta["quality_flag"] == "ok")
    )
    sub = combined_meta[mask].copy()
    sub["N_input_periods"] = pd.to_numeric(sub["WavePeriodInput"], errors="coerce")
    sub["run_type"] = np.where(sub["N_input_periods"] >= 50, "per240", "per40")
    return sub


sel = select_canon()
print(f"\n2. Canonical scope at f={TARGET_FREQ} Hz, {TARGET_AMP} V, full, quality=ok: "
      f"{len(sel)} runs (per40 n={int((sel['run_type']=='per40').sum())}, "
      f"per240 n={int((sel['run_type']=='per240').sum())})")

t_arr_in  = t_arrive_at_probe(IN_PROBES[0],  TARGET_FREQ)
t_arr_out = t_arrive_at_probe(OUT_PROBE,     TARGET_FREQ)
T_period  = 1.0 / TARGET_FREQ
print(f"   t_arrive(IN  r={PROBE_R_M[IN_PROBES[0]]} m): {t_arr_in:.2f} s "
      f"(= {t_arr_in*TARGET_FREQ:.1f} T)")
print(f"   t_arrive(OUT r={PROBE_R_M[OUT_PROBE]} m): {t_arr_out:.2f} s "
      f"(= {t_arr_out*TARGET_FREQ:.1f} T)")
print(f"   per40 paddle stop: {40/TARGET_FREQ:.2f} s (= 40 T from wavemaker)")


# ── Step 1: per240 N_offset sweep ───────────────────────────────────────
print("\n3. Step 1 — per240 N_offset sweep (window length = 10 T) …")

per240 = sel[sel["run_type"] == "per240"].copy()
N_OFFSETS = list(range(N_OFFSET_MIN_T, N_OFFSET_MAX_T + 1, N_OFFSET_STEP_T))

step1_records = []
for _, r in per240.iterrows():
    df = processed_dfs.get(r["path"])
    if df is None:
        continue
    sig_in  = canonical_in_signal(df)
    sig_out = get_eta(df, OUT_PROBE)
    if sig_in is None or sig_out is None:
        continue
    for N_off in N_OFFSETS:
        start_in  = t_arr_in  + N_off / TARGET_FREQ
        start_out = t_arr_out + N_off / TARGET_FREQ
        length_s  = WINDOW_PERIODS_FIXED / TARGET_FREQ
        a_in  = amp_in_window(sig_in,  start_in,  length_s)
        a_out = amp_in_window(sig_out, start_out, length_s)
        outin = a_out / a_in if (np.isfinite(a_in) and a_in > 0) else np.nan
        step1_records.append({
            "path":       r["path"],
            "wind":       r["WindCondition"],
            "N_offset_T": N_off,
            "A_in":       a_in,
            "A_out":      a_out,
            "OUT_IN":     outin,
        })

step1 = pd.DataFrame(step1_records)
print(f"   {len(step1)} (run × N_offset) rows from {step1['path'].nunique()} per240 runs")


# ── Step 2: per40 N_length sweep at fixed N_offset = (Step 1 winner) ────
print("\n4. Pre-flight: choose N_offset for Step 2.")
# We don't yet know the winner — the analysis of step1 happens after we
# plot. Use a candidate set that brackets where we expect the winner.
# Final N_offset_min for Step 2 will be picked AFTER plotting Step 1 + a
# numerical check. For the script to be self-contained, we run Step 2 at
# multiple N_offset candidates and the user can read off the right one
# from the per40 figure.
N_OFFSET_CANDIDATES = [10, 12, 15, 18]   # bracket H&G's "10–15 + 5 ramp"
print(f"   Step 2 will be done at N_offset ∈ {N_OFFSET_CANDIDATES} for comparison.")

print("\n5. Step 2 — per40 N_length sweep …")
per40 = sel[sel["run_type"] == "per40"].copy()
N_LENGTHS = list(range(N_LENGTH_MIN_T, N_LENGTH_MAX_T + 1, N_LENGTH_STEP_T))

step2_records = []
for _, r in per40.iterrows():
    df = processed_dfs.get(r["path"])
    if df is None:
        continue
    sig_in  = canonical_in_signal(df)
    sig_out = get_eta(df, OUT_PROBE)
    if sig_in is None or sig_out is None:
        continue
    for N_off in N_OFFSET_CANDIDATES:
        start_in  = t_arr_in  + N_off / TARGET_FREQ
        start_out = t_arr_out + N_off / TARGET_FREQ
        for N_len in N_LENGTHS:
            length_s  = N_len / TARGET_FREQ
            a_in  = amp_in_window(sig_in,  start_in,  length_s)
            a_out = amp_in_window(sig_out, start_out, length_s)
            outin = a_out / a_in if (np.isfinite(a_in) and a_in > 0) else np.nan
            step2_records.append({
                "path":       r["path"],
                "wind":       r["WindCondition"],
                "N_offset_T": N_off,
                "N_length_T": N_len,
                "A_in":       a_in,
                "A_out":      a_out,
                "OUT_IN":     outin,
            })

step2 = pd.DataFrame(step2_records)
print(f"   {len(step2)} (run × N_offset × N_length) rows from {step2['path'].nunique()} per40 runs")


# ── Numerical analysis: where do step1 curves flatten? ──────────────────
print("\n6. Step 1 stability analysis (per240) …")
# Per (path, wind), fit a stable median across the upper half of the
# sweep [15..30]T then walk back to find the smallest N_offset where the
# curve is within ±STABILITY_PCT % of that median.
step1_summary = []
for path, sub in step1.groupby("path"):
    sub = sub.sort_values("N_offset_T")
    wind = sub["wind"].iloc[0]
    if sub["OUT_IN"].dropna().empty:
        continue
    upper = sub[sub["N_offset_T"] >= 15]
    ref   = float(np.nanmedian(upper["OUT_IN"]))
    rel   = (sub["OUT_IN"].values - ref) / ref * 100.0
    ok    = np.abs(rel) < STABILITY_PCT
    # earliest N_offset where curve is within tolerance and STAYS within
    earliest_stable = None
    for i in range(len(ok)):
        if ok[i:].all():
            earliest_stable = int(sub["N_offset_T"].values[i])
            break
    step1_summary.append({
        "path":             Path(str(path)).name,
        "wind":             wind,
        "ref_OUTIN":        ref,
        "earliest_stable_N_offset": earliest_stable,
    })
step1_summary_df = pd.DataFrame(step1_summary)
print(step1_summary_df.to_string(index=False))


# ── Plot Step 1 ────────────────────────────────────────────────────────
print("\n7. Plotting Step 1 …")
fig, ax = plt.subplots(figsize=(10, 5.5))
for path, sub in step1.groupby("path"):
    sub = sub.sort_values("N_offset_T")
    wind = sub["wind"].iloc[0]
    color = WIND_COLOR_MAP[wind]
    ax.plot(sub["N_offset_T"], sub["OUT_IN"], color=color, lw=1.2, alpha=0.85,
            marker="o", markersize=3)
ax.axvspan(10, 15, color="#888", alpha=0.10, lw=0,
           label="H&G '10–15 T after arrival'")
ax.axvline(15, color="#888", lw=0.5, ls=":", alpha=0.5)
ax.set_xlabel("N_offset  [periods after wave arrival at each probe]", fontsize=10)
ax.set_ylabel("OUT/IN (FFT)", fontsize=10)
ax.grid(True, alpha=0.3)

# Legend
from matplotlib.lines import Line2D
handles = [
    Line2D([], [], color=WIND_COLOR_MAP["no"],   lw=1.4, label="per240 nowind"),
    Line2D([], [], color=WIND_COLOR_MAP["full"], lw=1.4, label="per240 fullwind"),
    Line2D([], [], color="#888", lw=4, alpha=0.4, label="H&G 10–15 T zone"),
]
ax.legend(handles=handles, fontsize=8, loc="best")
ax.set_title("", fontsize=11)
fig.tight_layout()
fig.savefig(OUT_PDF_S1, bbox_inches="tight")
fig.savefig(OUT_PNG_S1, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"   saved → {OUT_PDF_S1.relative_to(BASE)}")
print(f"          {OUT_PNG_S1.relative_to(BASE)}")


# ── Plot Step 2 ────────────────────────────────────────────────────────
print("\n8. Plotting Step 2 …")
n_cands = len(N_OFFSET_CANDIDATES)
fig, axes = plt.subplots(1, n_cands, figsize=(4 * n_cands, 5.0),
                         sharey=True, sharex=True)
if n_cands == 1:
    axes = [axes]

# Compute a per40 reference for Step 2 stability — use the median OUT/IN
# at N_length=10 across all runs, per N_offset, as the "stable benchmark".
step2_ref = (step2.groupby(["N_offset_T", "wind"])
                  .apply(lambda d: d.loc[d["N_length_T"] == 10, "OUT_IN"].median())
                  .reset_index(name="ref_OUTIN"))

step2_summary = []
for ax, N_off in zip(axes, N_OFFSET_CANDIDATES):
    sub_all = step2[step2["N_offset_T"] == N_off]
    for path, sub in sub_all.groupby("path"):
        sub = sub.sort_values("N_length_T")
        wind = sub["wind"].iloc[0]
        color = WIND_COLOR_MAP[wind]
        ax.plot(sub["N_length_T"], sub["OUT_IN"], color=color, lw=1.2, alpha=0.85,
                marker="o", markersize=3)
    ax.axvline(10, color="#2ECC71", lw=1.0, ls=":", alpha=0.7,
               label="N=10 (H&G default)")
    ax.set_xlabel("N_length  [periods]", fontsize=9)
    ax.set_title(f"start = N_offset = {N_off} T after arrival", fontsize=10)
    ax.grid(True, alpha=0.3)
    # Per (path, wind), find the largest N_length where curve stays
    # within ±STABILITY_PCT % of N_length=10 reference.
    for path, sub in sub_all.groupby("path"):
        sub = sub.sort_values("N_length_T")
        wind = sub["wind"].iloc[0]
        ref_val = sub.loc[sub["N_length_T"] == 10, "OUT_IN"].iloc[0] \
                  if (sub["N_length_T"] == 10).any() else np.nan
        if not np.isfinite(ref_val) or ref_val == 0:
            continue
        rel = (sub["OUT_IN"].values - ref_val) / ref_val * 100.0
        ok  = np.abs(rel) < STABILITY_PCT
        # Find largest N where sub[0..N] all OK
        max_stable = None
        for i in range(len(ok)):
            if not ok[i]:
                max_stable = int(sub["N_length_T"].values[i - 1]) if i > 0 else None
                break
        if max_stable is None and ok.all():
            max_stable = int(sub["N_length_T"].values[-1])
        step2_summary.append({
            "path":          Path(str(path)).name,
            "wind":          wind,
            "N_offset_T":    N_off,
            "ref_OUTIN":     ref_val,
            "max_stable_N_length_T": max_stable,
        })

axes[0].set_ylabel("OUT/IN (FFT)", fontsize=10)
axes[0].legend(fontsize=8, loc="best")
fig.suptitle("", fontsize=11)
fig.tight_layout()
fig.savefig(OUT_PDF_S2, bbox_inches="tight")
fig.savefig(OUT_PNG_S2, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"   saved → {OUT_PDF_S2.relative_to(BASE)}")
print(f"          {OUT_PNG_S2.relative_to(BASE)}")

step2_summary_df = pd.DataFrame(step2_summary)
print("\n9. Step 2 stability summary (per40, max stable N_length per N_offset):")
print(step2_summary_df.to_string(index=False))

# Aggregate across runs for the headline numbers
hl_step1 = step1_summary_df.groupby("wind")["earliest_stable_N_offset"].median()
hl_step2 = step2_summary_df.groupby(["wind", "N_offset_T"])["max_stable_N_length_T"].median()
print("\n10. Headline numbers:")
print("   Step 1 — earliest stable N_offset (median across per240 runs):")
print(hl_step1.to_string())
print("\n   Step 2 — max stable N_length (median across per40 runs):")
print(hl_step2.to_string())


# ── Save CSV + findings markdown ────────────────────────────────────────
combined = pd.concat([
    step1.assign(step=1),
    step2.assign(step=2),
], ignore_index=True)
combined.to_csv(OUT_CSV, index=False)
print(f"\n   CSV → {OUT_CSV.relative_to(BASE)}")

md_lines = [
    f"# Window squeeze study at f = {TARGET_FREQ} Hz",
    "",
    f"Generated by `analysis_scratch/window_squeeze_study.py`.",
    "",
    f"Scope: full panel, {TARGET_AMP} V, quality_ok, march-2026 cond4 lowrange. "
    f"Goal: find earliest stable start (per240) and longest stable length (per40) "
    f"at the worst-case thesis frequency.",
    "",
    "## Step 1 — per240 N_offset sweep (length fixed at 10T)",
    "",
    "Earliest N_offset at which OUT/IN(FFT) settles to within "
    f"±{STABILITY_PCT:.1f} % of its median across N_offset ∈ [15, 30]T:",
    "",
    "| run | wind | ref OUT/IN | earliest stable N_offset [T] |",
    "|---|---|---:|---:|",
]
for _, r in step1_summary_df.iterrows():
    md_lines.append(
        f"| {r['path']} | {r['wind']} | {r['ref_OUTIN']:.4f} | "
        f"{r['earliest_stable_N_offset']} |"
    )
md_lines += [
    "",
    "Median per wind:",
    "",
    "| wind | earliest stable N_offset [T] |",
    "|---|---:|",
]
for w, v in hl_step1.items():
    md_lines.append(f"| {w} | {v:.0f} |")

md_lines += [
    "",
    "## Step 2 — per40 N_length sweep at fixed N_offset",
    "",
    f"Largest N_length at which OUT/IN stays within ±{STABILITY_PCT:.1f} % of "
    f"N_length = 10T (the H&G default), per N_offset candidate "
    f"∈ {N_OFFSET_CANDIDATES}T:",
    "",
    "| run | wind | N_offset [T] | ref OUT/IN | max stable N_length [T] |",
    "|---|---|---:|---:|---:|",
]
for _, r in step2_summary_df.iterrows():
    md_lines.append(
        f"| {r['path']} | {r['wind']} | {r['N_offset_T']} | "
        f"{r['ref_OUTIN']:.4f} | {r['max_stable_N_length_T']} |"
    )
md_lines += [
    "",
    "Median per wind × N_offset:",
    "",
    "| wind | N_offset [T] | max stable N_length [T] |",
    "|---|---:|---:|",
]
for (w, n_off), v in hl_step2.items():
    md_lines.append(f"| {w} | {n_off} | {v:.0f} |")

OUT_MD.write_text("\n".join(md_lines), encoding="utf-8")
print(f"   findings → {OUT_MD.relative_to(BASE)}")

print("\nDone.")
