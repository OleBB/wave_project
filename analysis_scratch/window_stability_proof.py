"""
Window stability proof — all 4 thesis frequencies, with numbers.
==================================================================

Companion to window_proof_figure.py. That figure shows WHERE the window
sits relative to physics constraints. This figure shows that within
that window, OUT/IN(FFT) is stable.

For each thesis frequency (1.3, 1.4, 1.5, 1.6 Hz at 0.2 V, full panel,
quality ok), one panel showing OUT/IN(FFT) vs window length N, with
N_offset fixed at 10 T after wave-front arrival. Two run types overlaid:

    per240 (gold standard, long signal) — solid lines
    per40  (the metric we care about pooling) — dashed lines

A vertical green band marks the chosen N(f) ∈ {10, 13, 17, 15}.
Numerical annotations: per-curve drift at the chosen N relative to the
flat-region median (computed over N ∈ [5, 10]).

If the chosen N is inside the flat region of all curves, the window
choice is empirically stable — not just analytically safe.

Outputs (scratch only):
    analysis_scratch/window_stability_proof.pdf
    analysis_scratch/window_stability_proof.png
    analysis_scratch/window_stability_proof.csv

Run from repo root:
    /opt/anaconda3/envs/draumkvedet/bin/python analysis_scratch/window_stability_proof.py
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

N_OFFSET_FIXED     = 10               # H&G start, fixed for this study
N_LENGTH_LOOKUP    = {1.3: 10, 1.4: 13, 1.5: 17, 1.6: 15}
N_LENGTH_RANGE     = list(range(5, 31))     # sweep range
FLAT_REGION_NS     = (5, 10)                # baseline window for "stable median"

# Canon — march-2026 cond4 lowrange
PROCESSED_DIRS = [
    Path("waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange"),
    Path("waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"),
]

SCRATCH_DIR = Path(__file__).parent
OUT_PDF = SCRATCH_DIR / "window_stability_proof.pdf"
OUT_PNG = SCRATCH_DIR / "window_stability_proof.png"
OUT_CSV = SCRATCH_DIR / "window_stability_proof.csv"


# ── Helpers ─────────────────────────────────────────────────────────────
def fft_amp(segment: np.ndarray, target_hz: float,
            fs: float = FS, band_hz: float = FFT_BAND_HZ) -> float:
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


def t_arr_at(r_m: float, f: float) -> float:
    return r_m / c_group(f, TANK_DEPTH_M)


def amp_in_window(signal: np.ndarray, start_s: float, length_s: float,
                  fs: float = FS) -> float:
    s_idx = int(round(start_s * fs))
    e_idx = s_idx + int(round(length_s * fs))
    if s_idx < 0 or e_idx > len(signal):
        return np.nan
    return fft_amp(signal[s_idx:e_idx], _CURRENT_FREQ)


# ── Load ────────────────────────────────────────────────────────────────
print("1. Loading meta + processed_dfs (canon March-2026 lowrange) …")
combined_meta, _, _, _ = load_analysis_data(*PROCESSED_DIRS, load_processed=False)
processed_dfs = load_processed_dfs(*PROCESSED_DIRS)
print(f"   meta: {len(combined_meta)} rows · processed_dfs: {len(processed_dfs)} runs")


def select_canon(target_freq: float) -> pd.DataFrame:
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
    sub["run_type"] = np.where(sub["N_input_periods"] >= PER240_THRESHOLD_T,
                               "per240", "per40")
    return sub


# ── Compute OUT/IN(N) per (run, N_length) at fixed N_offset=10 ──────────
print("\n2. Sweeping N_length at fixed N_offset=10, all 4 freqs …")
records = []
for f in THESIS_FREQS:
    global _CURRENT_FREQ
    _CURRENT_FREQ = f
    sel = select_canon(f)
    print(f"   f={f} Hz: per240 n={int((sel['run_type']=='per240').sum())}, "
          f"per40 n={int((sel['run_type']=='per40').sum())}")
    t_arr_in  = t_arr_at(PROBE_R_M[IN_PROBES[0]], f)
    t_arr_out = t_arr_at(PROBE_R_M[OUT_PROBE], f)
    start_in  = t_arr_in  + N_OFFSET_FIXED / f
    start_out = t_arr_out + N_OFFSET_FIXED / f

    for _, r in sel.iterrows():
        df = processed_dfs.get(r["path"])
        if df is None:
            continue
        sig_in  = canonical_in_signal(df)
        sig_out = get_eta(df, OUT_PROBE)
        if sig_in is None or sig_out is None:
            continue
        for N in N_LENGTH_RANGE:
            length_s = N / f
            a_in  = amp_in_window(sig_in,  start_in,  length_s)
            a_out = amp_in_window(sig_out, start_out, length_s)
            outin = a_out / a_in if (np.isfinite(a_in) and a_in > 0) else np.nan
            records.append({
                "freq_hz":    f,
                "wind":       r["WindCondition"],
                "run_type":   r["run_type"],
                "path":       r["path"],
                "N_length_T": N,
                "OUT_IN":     outin,
            })

per_run = pd.DataFrame(records)
print(f"   {len(per_run)} (run × N) rows total")


# ── Aggregate per (f, wind, run_type, N): median across runs ────────────
agg = per_run.groupby(["freq_hz", "wind", "run_type", "N_length_T"]).agg(
    OUT_IN_med = ("OUT_IN", "median"),
    OUT_IN_std = ("OUT_IN", "std"),
    n          = ("OUT_IN", "size"),
).reset_index()


# ── Stability — drift from flat-region median at chosen N(f) ────────────
def stability_at(fhz: float, wind: str, run_type: str, N_chosen: int):
    sub = agg[(agg["freq_hz"] == fhz)
              & (agg["wind"] == wind)
              & (agg["run_type"] == run_type)]
    if sub.empty:
        return None
    flat = sub[(sub["N_length_T"] >= FLAT_REGION_NS[0])
               & (sub["N_length_T"] <= FLAT_REGION_NS[1])]
    if flat.empty:
        return None
    ref = flat["OUT_IN_med"].median()
    chosen_row = sub[sub["N_length_T"] == N_chosen]
    if chosen_row.empty:
        return None
    val = float(chosen_row["OUT_IN_med"].iloc[0])
    if not np.isfinite(ref) or ref == 0:
        return None
    drift_pct = (val - ref) / ref * 100.0
    return {"ref": ref, "val": val, "drift_pct": drift_pct}


print("\n3. Drift at chosen N(f) vs flat-region median (N=5..10):")
print(f"  {'f':<5} {'wind':<5} {'run_type':<8} {'N_chosen':<10} {'ref':<8} "
      f"{'val':<8} {'drift %':<10}")
print("  " + "─" * 60)
stab_rows = []
for f in THESIS_FREQS:
    Nc = N_LENGTH_LOOKUP[f]
    for wind in ("no", "full"):
        for run_type in ("per240", "per40"):
            s = stability_at(f, wind, run_type, Nc)
            if s is None:
                continue
            print(f"  {f:<5.1f} {wind:<5} {run_type:<8} N={Nc:<8} "
                  f"{s['ref']:<8.4f} {s['val']:<8.4f} {s['drift_pct']:+.3f}%")
            stab_rows.append({
                "freq_hz":   f, "wind": wind, "run_type": run_type,
                "N_chosen":  Nc, "ref": s["ref"], "val": s["val"],
                "drift_pct": s["drift_pct"],
            })

stability_df = pd.DataFrame(stab_rows)


# ── Plot ────────────────────────────────────────────────────────────────
print("\n4. Plotting …")
fig, axes = plt.subplots(1, 4, figsize=(15, 4.8), sharey=False)

for ax, f in zip(axes, THESIS_FREQS):
    Nc = N_LENGTH_LOOKUP[f]
    sub = agg[agg["freq_hz"] == f]

    # Plot 4 curves: 2 winds × 2 run_types
    for wind in ("no", "full"):
        for rt, ls, marker in [("per240", "-",  "o"), ("per40", "--", "s")]:
            sw = sub[(sub["wind"] == wind) & (sub["run_type"] == rt)].sort_values("N_length_T")
            if sw.empty:
                continue
            color = WIND_COLOR_MAP[wind]
            ax.plot(sw["N_length_T"], sw["OUT_IN_med"],
                    color=color, ls=ls, marker=marker, ms=3.5,
                    lw=1.2, alpha=0.85,
                    label=f"{rt} {wind}")

    # Chosen-N green band
    ax.axvspan(Nc - 0.5, Nc + 0.5, color="#2ECC71", alpha=0.30, lw=0,
               label=f"chosen N = {Nc} T")
    ax.axvline(Nc, color="#1A6E2A", lw=1.0, ls="-", alpha=0.85)

    # Flat region band (the reference for drift computation)
    ax.axvspan(FLAT_REGION_NS[0], FLAT_REGION_NS[1],
               color="#888", alpha=0.10, lw=0)

    # Drift annotation — biggest |drift| across the four curves at chosen N
    sub_stab = stability_df[(stability_df["freq_hz"] == f)
                            & (stability_df["N_chosen"] == Nc)]
    if not sub_stab.empty:
        max_drift = sub_stab["drift_pct"].abs().max()
        ax.text(0.97, 0.04,
                f"max |drift| at N={Nc}:\n{max_drift:.2f}%",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, color="#1A6E2A",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="#1A6E2A", alpha=0.9, lw=0.5))

    ax.set_xlim(min(N_LENGTH_RANGE), max(N_LENGTH_RANGE))
    ax.set_xlabel("N_length [periods]", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.96, f"$f = {f}$ Hz",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec="#bbb", alpha=0.85, lw=0.4))
    if ax is axes[0]:
        ax.set_ylabel("OUT/IN (FFT)", fontsize=10)

# Single legend at bottom
handles, labels = axes[-1].get_legend_handles_labels()
# Dedupe
seen, h_uniq, l_uniq = set(), [], []
for h, l in zip(handles, labels):
    if l in seen:
        continue
    seen.add(l)
    h_uniq.append(h); l_uniq.append(l)
fig.legend(h_uniq, l_uniq, loc="lower center", ncol=5,
           fontsize=8, bbox_to_anchor=(0.5, -0.02), frameon=True)

fig.suptitle("", fontsize=11)
fig.tight_layout(rect=[0, 0.06, 1, 1])
fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"   saved → {OUT_PDF.relative_to(BASE)}")
print(f"          {OUT_PNG.relative_to(BASE)}")


# ── CSV summary ─────────────────────────────────────────────────────────
agg.to_csv(OUT_CSV, index=False)
print(f"   CSV → {OUT_CSV.relative_to(BASE)}")

# Headlines
print("\n5. Headline numbers:")
print(f"   Median |drift| at chosen N(f):  "
      f"{stability_df['drift_pct'].abs().median():.3f}%")
print(f"   Max    |drift| at chosen N(f):  "
      f"{stability_df['drift_pct'].abs().max():.3f}%")
worst = stability_df.loc[stability_df['drift_pct'].abs().idxmax()]
print(f"   Worst case: f={worst['freq_hz']} Hz, "
      f"{worst['wind']} {worst['run_type']}, drift={worst['drift_pct']:+.3f}%")

print("\nDone.")
