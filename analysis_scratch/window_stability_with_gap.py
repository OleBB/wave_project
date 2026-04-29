"""
Window stability proof — gap-filtered version.
================================================

Companion to `window_stability_proof.py`. That figure showed drift at
chosen N(f) per (f, wind, run_type) aggregated across all canonical
runs. At 1.5 and 1.6 Hz the per240 nowind cell drifted by ~3 %, which
the user hypothesized was driven by insufficient inter-run wait time
(`inter_run_gap_s`).

This script tests that hypothesis directly:

  Top row    — drift-vs-gap scatter for all canonical runs, all 4 freqs.
               One point per run; x = inter_run_gap_s; y = drift at
               chosen N(f) relative to N=5–10 flat-region median.
               Colour by frequency, marker by (wind, run_type).

  Bottom row — same 4 stability panels as window_stability_proof.py,
               but with runs filtered to inter_run_gap_s >= GAP_MIN_S.
               Drift annotations recomputed on the filtered cohort.

If the user's hypothesis holds, the scatter shows |drift| dropping as
gap grows, and the filtered stability panels show drift collapsing
below ~1 % at every cell.

Outputs:
    analysis_scratch/window_stability_with_gap.pdf
    analysis_scratch/window_stability_with_gap.png
    analysis_scratch/window_stability_with_gap.csv

Run from repo root:
    /opt/anaconda3/envs/draumkvedet/bin/python analysis_scratch/window_stability_with_gap.py
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
FS                 = 250.0
FFT_BAND_HZ        = 0.05
THESIS_FREQS       = [1.3, 1.4, 1.5, 1.6]
TARGET_AMP         = 0.2
PER240_THRESHOLD_T = 50

IN_PROBES          = ["9373/170", "9373/340"]
OUT_PROBE          = "12400/250"
PROBE_R_M          = {"9373/170": 9.373, "9373/340": 9.373, "12400/250": 12.400}
TANK_DEPTH_M       = HG.TANK_DEPTH_M

N_OFFSET_FIXED     = 10
N_LENGTH_LOOKUP    = {1.3: 10, 1.4: 13, 1.5: 17, 1.6: 15}
N_LENGTH_RANGE     = list(range(5, 31))
FLAT_REGION_NS     = (5, 10)

# Gap filter — minimum inter-run gap in seconds for "well-rested"
GAP_MIN_S          = 100.0

# Frequency colours for the scatter
FREQ_COLOR = {1.3: "#1F77B4", 1.4: "#2CA02C", 1.5: "#FF7F0E", 1.6: "#D62728"}

PROCESSED_DIRS = [
    Path("waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange"),
    Path("waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"),
]

SCRATCH_DIR = Path(__file__).parent
OUT_PDF = SCRATCH_DIR / "window_stability_with_gap.pdf"
OUT_PNG = SCRATCH_DIR / "window_stability_with_gap.png"
OUT_CSV = SCRATCH_DIR / "window_stability_with_gap.csv"


# ── Helpers ─────────────────────────────────────────────────────────────
def fft_amp(seg, target_hz, fs=FS, band_hz=FFT_BAND_HZ):
    N = len(seg)
    if N < 4:
        return np.nan
    s = np.asarray(seg, dtype=float).copy()
    if np.isnan(s).any():
        idx = np.arange(N); good = ~np.isnan(s)
        if good.sum() < N * 0.9:
            return np.nan
        s = np.interp(idx, idx[good], s[good])
    freqs = np.fft.fftfreq(N, d=1.0 / fs)
    pos   = freqs > 0
    pos_f = freqs[pos]
    mask  = (pos_f >= target_hz - band_hz) & (pos_f <= target_hz + band_hz)
    if not mask.any():
        j = int(np.argmin(np.abs(pos_f - target_hz)))
    else:
        local = int(np.argmin(np.abs(pos_f[mask] - target_hz)))
        j = np.where(mask)[0][local]
    return float(2.0 * np.abs(np.fft.fft(s)[pos])[j] / N)


def get_eta(df, pos):
    for col in (f"eta_{pos}_interp", f"eta_{pos}"):
        if col in df.columns:
            return df[col].to_numpy(dtype=float)
    return None


def can_in(df):
    sigs = [get_eta(df, p) for p in IN_PROBES]
    sigs = [s for s in sigs if s is not None]
    if not sigs:
        return None
    nmin = min(len(s) for s in sigs)
    return np.nanmean(np.vstack([s[:nmin] for s in sigs]), axis=0)


def amp_window(sig, start_s, length_s, target_hz, fs=FS):
    s_idx = int(round(start_s * fs))
    e_idx = s_idx + int(round(length_s * fs))
    if s_idx < 0 or e_idx > len(sig):
        return np.nan
    return fft_amp(sig[s_idx:e_idx], target_hz)


# ── Load ────────────────────────────────────────────────────────────────
print("1. Loading meta + processed_dfs (canon March-2026 lowrange) …")
combined_meta, _, _, _ = load_analysis_data(*PROCESSED_DIRS, load_processed=False)
processed_dfs = load_processed_dfs(*PROCESSED_DIRS)
print(f"   meta: {len(combined_meta)} rows · processed_dfs: {len(processed_dfs)} runs")


def select_canon(target_freq):
    f_col = pd.to_numeric(combined_meta["WaveFrequencyInput [Hz]"], errors="coerce")
    a_col = pd.to_numeric(combined_meta["WaveAmplitudeInput [Volt]"], errors="coerce")
    mask = (
        np.isclose(f_col, target_freq, atol=0.02)
        & np.isclose(a_col, TARGET_AMP, atol=0.01)
        & (combined_meta["PanelCondition"] == "full")
        & (combined_meta["quality_flag"] == "ok")
    )
    sub = combined_meta[mask].copy()
    sub["N_input_periods"] = pd.to_numeric(sub["WavePeriodInput"], errors="coerce")
    sub["run_type"] = np.where(sub["N_input_periods"] >= PER240_THRESHOLD_T, "per240", "per40")
    return sub


# ── Sweep N_length per run, compute per-run drift at chosen N(f) ────────
print("\n2. Sweeping N_length per run + computing per-run drift …")
records_curve = []     # (run × N) rows for plot
records_drift = []     # one row per run for scatter

for f in THESIS_FREQS:
    sel = select_canon(f)
    Nc = N_LENGTH_LOOKUP[f]
    t_arr_in  = PROBE_R_M[IN_PROBES[0]]  / c_group(f, TANK_DEPTH_M)
    t_arr_out = PROBE_R_M[OUT_PROBE]     / c_group(f, TANK_DEPTH_M)
    start_in  = t_arr_in  + N_OFFSET_FIXED / f
    start_out = t_arr_out + N_OFFSET_FIXED / f

    for _, r in sel.iterrows():
        df = processed_dfs.get(r["path"])
        if df is None:
            continue
        sig_in  = can_in(df)
        sig_out = get_eta(df, OUT_PROBE)
        if sig_in is None or sig_out is None:
            continue

        # OUT/IN(N) curve for this run
        outin = {}
        for N in N_LENGTH_RANGE:
            length_s = N / f
            ai = amp_window(sig_in,  start_in,  length_s, f)
            ao = amp_window(sig_out, start_out, length_s, f)
            ratio = ao / ai if (np.isfinite(ai) and ai > 0) else np.nan
            outin[N] = ratio
            records_curve.append({
                "freq_hz":   f,
                "wind":      r["WindCondition"],
                "run_type":  r["run_type"],
                "path":      r["path"],
                "gap_s":     float(r.get("inter_run_gap_s", np.nan)),
                "N_length":  N,
                "OUT_IN":    ratio,
            })

        # Per-run drift at chosen N vs flat-region median
        flat_vals = [outin[k] for k in range(FLAT_REGION_NS[0], FLAT_REGION_NS[1] + 1)
                     if np.isfinite(outin.get(k, np.nan))]
        if not flat_vals:
            continue
        ref = float(np.median(flat_vals))
        chosen = outin.get(Nc, np.nan)
        if not np.isfinite(chosen) or ref == 0:
            continue
        drift_pct = (chosen - ref) / ref * 100.0
        records_drift.append({
            "freq_hz":   f,
            "wind":      r["WindCondition"],
            "run_type":  r["run_type"],
            "path":      r["path"],
            "gap_s":     float(r.get("inter_run_gap_s", np.nan)),
            "N_chosen":  Nc,
            "ref":       ref,
            "val":       float(chosen),
            "drift_pct": drift_pct,
        })

curve_df = pd.DataFrame(records_curve)
drift_df = pd.DataFrame(records_drift)
print(f"   {len(curve_df)} (run × N) sweep points; {len(drift_df)} per-run drift values")


# ── Aggregate per (f, wind, run_type, N) — ALL runs vs FILTERED ─────────
def agg_per_cond(df_in):
    return df_in.groupby(["freq_hz", "wind", "run_type", "N_length"]).agg(
        OUT_IN_med=("OUT_IN", "median"),
        OUT_IN_std=("OUT_IN", "std"),
        n=("OUT_IN", "size"),
    ).reset_index()


agg_all      = agg_per_cond(curve_df)
filtered     = curve_df[curve_df["gap_s"] >= GAP_MIN_S]
agg_filtered = agg_per_cond(filtered)

print(f"\n3. Filtering at gap >= {GAP_MIN_S} s …")
print(f"   ALL runs:      {curve_df['path'].nunique()} runs")
print(f"   FILTERED runs: {filtered['path'].nunique()} runs")
dropped_runs = set(curve_df["path"].unique()) - set(filtered["path"].unique())
print(f"   Dropped {len(dropped_runs)} runs with gap < {GAP_MIN_S} s")


# ── Drift recompute on FILTERED cohort ─────────────────────────────────
print("\n4. Drift on filtered cohort:")
print(f"  {'f':<5} {'wind':<5} {'run_type':<8} {'N':<4} {'all':<10} {'filtered':<10} {'n_filt':<6}")
print("  " + "─" * 60)

drift_summary = []
for f in THESIS_FREQS:
    Nc = N_LENGTH_LOOKUP[f]
    for wind in ("no", "full"):
        for rt in ("per240", "per40"):
            for label, agg in [("all", agg_all), ("filt", agg_filtered)]:
                sub = agg[(agg["freq_hz"] == f)
                          & (agg["wind"] == wind)
                          & (agg["run_type"] == rt)]
                if sub.empty:
                    continue
                flat = sub[(sub["N_length"] >= FLAT_REGION_NS[0])
                           & (sub["N_length"] <= FLAT_REGION_NS[1])]
                if flat.empty:
                    continue
                ref = float(flat["OUT_IN_med"].median())
                row = sub[sub["N_length"] == Nc]
                if row.empty:
                    continue
                val = float(row["OUT_IN_med"].iloc[0])
                if not np.isfinite(ref) or ref == 0:
                    continue
                drift_pct = (val - ref) / ref * 100.0
                n_filt = int(row["n"].iloc[0])
                drift_summary.append({
                    "f": f, "wind": wind, "rt": rt, "N": Nc,
                    "label": label, "drift_pct": drift_pct, "n": n_filt,
                })
            # Print one combined row per (f, wind, rt)
            row_all  = next((d for d in drift_summary
                             if d["f"] == f and d["wind"] == wind
                             and d["rt"] == rt and d["label"] == "all"), None)
            row_filt = next((d for d in drift_summary
                             if d["f"] == f and d["wind"] == wind
                             and d["rt"] == rt and d["label"] == "filt"), None)
            if row_all is None:
                continue
            d_all  = row_all["drift_pct"]
            d_filt = row_filt["drift_pct"] if row_filt else np.nan
            n_filt = row_filt["n"] if row_filt else 0
            print(f"  {f:<5.1f} {wind:<5} {rt:<8} N={Nc:<4} "
                  f"{d_all:+7.3f}%   {d_filt:+7.3f}%   {n_filt:<6}")


# ── Plot — 2 rows × 4 cols ──────────────────────────────────────────────
print("\n5. Plotting …")
fig, axes = plt.subplots(2, 4, figsize=(15, 8.5))

# === Row 0: drift vs gap scatter, one panel per frequency ===
for ax, f in zip(axes[0], THESIS_FREQS):
    sub = drift_df[drift_df["freq_hz"] == f]
    Nc = N_LENGTH_LOOKUP[f]
    for (wind, rt), sg in sub.groupby(["wind", "run_type"]):
        marker = {"per240": "o", "per40": "s"}[rt]
        color  = WIND_COLOR_MAP[wind]
        ax.scatter(sg["gap_s"], sg["drift_pct"],
                   marker=marker, s=80, color=color, edgecolor="black",
                   lw=0.4, alpha=0.85,
                   label=f"{wind} {rt}")
    ax.axhline(0, color="black", lw=0.5, alpha=0.5)
    ax.axhline(+1.0, color="#aaa", lw=0.5, ls=":", alpha=0.7)
    ax.axhline(-1.0, color="#aaa", lw=0.5, ls=":", alpha=0.7)
    ax.axvline(GAP_MIN_S, color="#1A6E2A", lw=0.8, ls="--", alpha=0.7,
               label=f"gap = {GAP_MIN_S:.0f} s")
    ax.set_xlabel("inter_run_gap_s [s]", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.96, f"$f = {f}$ Hz   ($N = {Nc}$ T)",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec="#bbb", alpha=0.85, lw=0.4))
    if ax is axes[0][0]:
        ax.set_ylabel("drift at N(f) [%]", fontsize=10)

axes[0][0].legend(fontsize=7, loc="lower right", framealpha=0.9, ncol=1)

# === Row 1: stability panels (filtered cohort) ===
for ax, f in zip(axes[1], THESIS_FREQS):
    Nc = N_LENGTH_LOOKUP[f]
    sub = agg_filtered[agg_filtered["freq_hz"] == f]

    for wind in ("no", "full"):
        for rt, ls, marker in [("per240", "-", "o"), ("per40", "--", "s")]:
            sw = sub[(sub["wind"] == wind) & (sub["run_type"] == rt)].sort_values("N_length")
            if sw.empty:
                continue
            color = WIND_COLOR_MAP[wind]
            ax.plot(sw["N_length"], sw["OUT_IN_med"],
                    color=color, ls=ls, marker=marker, ms=3.5,
                    lw=1.2, alpha=0.85,
                    label=f"{rt} {wind}")

    ax.axvspan(Nc - 0.5, Nc + 0.5, color="#2ECC71", alpha=0.30, lw=0)
    ax.axvline(Nc, color="#1A6E2A", lw=1.0, ls="-", alpha=0.85)
    ax.axvspan(FLAT_REGION_NS[0], FLAT_REGION_NS[1],
               color="#888", alpha=0.10, lw=0)

    # Drift annotation — biggest |drift| across filtered cohort
    drifts_here = [d["drift_pct"] for d in drift_summary
                   if d["f"] == f and d["label"] == "filt"]
    if drifts_here:
        max_drift = max(abs(d) for d in drifts_here)
        ax.text(0.97, 0.04,
                f"max |drift| at N={Nc}:\n{max_drift:.2f}%  (filtered)",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, color="#1A6E2A",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="#1A6E2A", alpha=0.9, lw=0.5))

    ax.set_xlim(min(N_LENGTH_RANGE), max(N_LENGTH_RANGE))
    ax.set_xlabel("N_length [periods]", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.96, f"$f = {f}$ Hz   filtered (gap≥{GAP_MIN_S:.0f}s)",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec="#bbb", alpha=0.85, lw=0.4))
    if ax is axes[1][0]:
        ax.set_ylabel("OUT/IN (FFT)", fontsize=10)

# Single legend at the bottom for row 1
hlist, llist = axes[1][-1].get_legend_handles_labels()
seen, h_uniq, l_uniq = set(), [], []
for h, l in zip(hlist, llist):
    if l in seen:
        continue
    seen.add(l)
    h_uniq.append(h); l_uniq.append(l)
fig.legend(h_uniq, l_uniq, loc="lower center", ncol=4,
           fontsize=8, bbox_to_anchor=(0.5, -0.02), frameon=True)

fig.suptitle("", fontsize=11)
fig.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"   saved → {OUT_PDF.relative_to(BASE)}")
print(f"          {OUT_PNG.relative_to(BASE)}")


# ── CSV ────────────────────────────────────────────────────────────────
drift_df.to_csv(OUT_CSV, index=False)
print(f"   CSV → {OUT_CSV.relative_to(BASE)}")

# Headlines
print("\n6. Headlines:")
print("   ALL runs:")
all_drifts = [d["drift_pct"] for d in drift_summary if d["label"] == "all"]
print(f"     median |drift| = {np.median(np.abs(all_drifts)):.3f}%")
print(f"     max    |drift| = {np.max(np.abs(all_drifts)):.3f}%")
print(f"   FILTERED (gap >= {GAP_MIN_S}s):")
filt_drifts = [d["drift_pct"] for d in drift_summary if d["label"] == "filt"]
print(f"     median |drift| = {np.median(np.abs(filt_drifts)):.3f}%")
print(f"     max    |drift| = {np.max(np.abs(filt_drifts)):.3f}%")

print("\nDone.")
