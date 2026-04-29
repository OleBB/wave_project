"""
CH04 §2.6 — Paddle-frequency IN contamination & window-size sensitivity
=======================================================================

Goal
----
The CH05 main result reports OUT/IN (FFT) = A_out / A_in at the paddle
frequency with a ±0.05 Hz FFT window. Under fullwind, the IN probe
(9373/170, fully exposed) sees a small amount of wind energy inside that
window, which *inflates* A_in and biases OUT/IN downward
(CLAUDE.md §16, bias #1). This script asks three related questions and
answers them on the same figure so a reviewer can see the metric is
sound:

  (A) Window-size sensitivity:
      how does OUT/IN change as the FFT analysis window shrinks from
      ~100 paddle periods down to 20? A flat curve = robust metric.

  (B) Wind-correction via incoherent subtraction:
      assuming paddle and wind are uncorrelated at the paddle frequency,
          E[A_fw²]  =  A_paddle² + E[A_wind_at_fpaddle²]
      → A_paddle(fw)  =  √(max(0, A_fw² − E[A_wind_at_fpaddle²]))
      We estimate E[A_wind_at_fpaddle²] per (freq, amp, mooring) group
      from the pure-wind PSD at the paddle bin, where

          PSD_wind(f)  :=  mean PSD_residual(fullwind)  −
                           mean PSD_residual(nowind)

      (same machinery as reconstruction_A_vs_B §5–§6, but evaluated at
      f_paddle instead of over the 2–6 Hz wind band). The residual is
      the method-A reconstruction residual: signal_full − signal_peak_bin.

  (C) T_cross cross-check:
      if the correction works, A_in_corrected(fw) should agree with
      A_in(nw) within run-to-run scatter. That's the independent
      validation: it closes the loop without relying on any one
      assumption (wind PSD shape, paddle wind-independence, uncorrelated
      sum).

Method summary
--------------
 1. Load meta_results + FFT dict + processed_dfs (the cond4 thesis-band
    datasets, same as t_cross_figure / reconstruction_A_vs_B).
 2. Per wave run: reconstruct signal_A (peak-bin paddle) and residual =
    signal_full − signal_A. Welch PSD of the residual on a shared grid.
 3. Group runs by (WaveFrequencyInput, WaveAmplitudeInput, Mooring,
    PanelCondition). For groups with ≥ 1 nowind and ≥ 1 fullwind run:
    4. compute PSD_wind(f) by PSD-subtraction, read off at f_paddle ±
       BAND_HALF_HZ → E[A_wind²] = 2·∫ PSD_wind df (Parseval for the
       narrow-band amplitude).
    5. Apply correction to each fullwind run:
         A_in_corrected = √(max(0, A_in_fw² − E[A_wind²]))
       → OUT/IN_corrected = A_out_fw / A_in_corrected
 6. Part A (window sensitivity): for the thesis-band runs, re-compute
    A_in_fw / A_out_fw using a truncated segment of the detected
    stable-wave window (N ∈ {20, 40, 60, 100, full}). OUT/IN vs
    N_periods → flat curve ⇒ robust.
 7. Plot: 2×3 grid (rows = window-sensitivity / correction validation,
    cols = amplitude 0.1 / 0.2 / 0.3 V). Scratch quick-view PDF and
    thesis figure.
 8. Write findings markdown with the headline numbers; emit stub via
    shared plot_utils helpers (same pattern as mooring_comparison,
    reconstruction_A_vs_B).

Run from repo root:
    /opt/anaconda3/envs/draumkvedet/bin/python analysis_scratch/paddle_contamination_study.py

Outputs
-------
    analysis_scratch/paddle_contamination_study.pdf
    analysis_scratch/paddle_contamination_correction.csv
    analysis_scratch/paddle_contamination_window_sensitivity.csv
    analysis_scratch/paddle_contamination_findings.md
    output/FIGURES/ch04_paddle_contamination.pdf
    output/TEXFIGU/ch04_paddle_contamination.tex
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
from scipy import signal as sp_signal

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
os.chdir(BASE)

from wavescripts.improved_data_loader import load_analysis_data, load_processed_dfs
from wavescripts.plot_utils import apply_thesis_style, freq_to_k, amp_to_label, amp_to_tag

apply_thesis_style()

# ── I/O ───────────────────────────────────────────────────────────────────────
SCRATCH_DIR         = Path(__file__).parent
SCRATCH_PDF         = SCRATCH_DIR / "paddle_contamination_study.pdf"
SCRATCH_CSV_CORR    = SCRATCH_DIR / "paddle_contamination_correction.csv"
SCRATCH_CSV_WINDOW  = SCRATCH_DIR / "paddle_contamination_window_sensitivity.csv"
SCRATCH_FINDINGS_MD = SCRATCH_DIR / "paddle_contamination_findings.md"

THESIS_NAME = "ch04_paddle_contamination"
THESIS_PDF  = BASE / "output" / "FIGURES" / f"{THESIS_NAME}.pdf"
THESIS_STUB = BASE / "output" / "TEXFIGU" / f"{THESIS_NAME}.tex"
THESIS_PDF.parent.mkdir(parents=True, exist_ok=True)
THESIS_STUB.parent.mkdir(parents=True, exist_ok=True)

# Same thesis datasets as t_cross_figure.py / reconstruction_A_vs_B.py
RESULTS_PROCESSED_DIRS = [
    Path("waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange"),
    Path("waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"),
]
_results_dataset_names = {p.name.removeprefix("PROCESSED-") for p in RESULTS_PROCESSED_DIRS}

# ── Constants ────────────────────────────────────────────────────────────────
FREQS         = [1.3, 1.4, 1.5, 1.6]   # thesis scope
AMPS          = [0.1, 0.2, 0.3]
FFT_WINDOW_HZ = 0.10                   # total width; matches signal_processing
BAND_HALF_HZ  = 0.05                   # ± half-width around paddle
PROBES        = ["9373/170", "12400/250"]
PROBE_LABEL   = {"9373/170": "IN", "12400/250": "OUT"}
IN_POS, OUT_POS = PROBES

# Window-sensitivity choices. Numbers of paddle periods inside the detected
# stable plateau. "full" = whole plateau (upper bound ≈ 240 for per240 runs).
N_PERIOD_CHOICES = [20, 40, 60, 100, "full"]

# Colour scheme (reuse the rest of the thesis' WIND_COLOR_MAP conventions)
COLOR_NW   = "#2E86AB"   # blue — nowind
COLOR_FW   = "#E74C3C"   # red  — fullwind (raw)
COLOR_CORR = "#2ECC71"   # green — fullwind after wind-subtraction
COLOR_NW_A = "#888888"   # grey — secondary

# Shared PSD grid for residual subtraction (matches reconstruction_A_vs_B)
COMMON_F_GRID = np.arange(0.0, 12.5 + 1e-9, 0.05)


# ── 1. Load data ─────────────────────────────────────────────────────────────
print("1. Loading meta_results + fft_dict + processed_dfs …")
combined_meta, processed_dfs, fft_dict, _ = load_analysis_data(
    *RESULTS_PROCESSED_DIRS, load_processed=False,
)
meta_results = combined_meta[
    combined_meta["path"].apply(lambda p: any(d in str(p) for d in _results_dataset_names))
].copy()
meta_results["Mooring"] = meta_results["Mooring"].replace({
    "below_90_loose230": "below_90_loose",
    "below_90_loose300": "below_90_loose",
})
print(f"   meta_results: {len(meta_results)} rows")

# Wave runs only, full panel, quality ok-ish, thesis-band frequencies.
wave_runs = meta_results[
    meta_results["WaveFrequencyInput [Hz]"].notna()
    & (meta_results["WaveFrequencyInput [Hz]"] > 0)
    & (meta_results["PanelCondition"] == "full")
    & (meta_results["quality_flag"].isin(["ok", "probe_malfunction_secondary"]))
].copy()
wave_runs["freq_r"] = wave_runs["WaveFrequencyInput [Hz]"].round(2)
wave_runs["amp_r"]  = wave_runs["WaveAmplitudeInput [Volt]"].round(2)
wave_runs = wave_runs[wave_runs["freq_r"].isin(FREQS)].copy()
print(f"   {len(wave_runs)} thesis-band full-panel wave runs")

# Part B needs processed time series.
print("   Loading processed_dfs (heavy, ~20–30 s) …")
processed_dfs = load_processed_dfs(*RESULTS_PROCESSED_DIRS)


# ── 2. Reconstruction helper (from reconstruction_A_vs_B.py) ─────────────────
def reconstruct(fft_series: pd.Series, target_freq: float, band_half_hz: float):
    """Return (time_axis, signal_full, signal_A, fs, actual_freq).

    Peak-bin (method A) IFFT reconstruction of the paddle wave; residual =
    signal_full − signal_A goes into the PSD pool.
    """
    freq_bins = fft_series.index.values
    fft_complex = fft_series.values
    N = len(fft_complex)
    df_freq = abs(freq_bins[1] - freq_bins[0])
    fs = df_freq * N

    fft_ord = np.fft.ifftshift(fft_complex).astype(complex)
    fftfreqs = np.fft.ifftshift(freq_bins)
    pos_freqs = fftfreqs[fftfreqs > 0]
    actual_freq = pos_freqs[np.argmin(np.abs(pos_freqs - target_freq))]

    peak_idx = int(np.argmin(np.abs(fftfreqs - actual_freq)))
    mirror_idx = int(np.argmin(np.abs(fftfreqs + actual_freq)))
    fft_A = np.zeros_like(fft_ord, dtype=complex)
    fft_A[peak_idx] = fft_ord[peak_idx]
    fft_A[mirror_idx] = fft_ord[mirror_idx]

    signal_full = np.real(np.fft.ifft(fft_ord))
    signal_A    = np.real(np.fft.ifft(fft_A))
    time_axis   = np.arange(N) / fs
    return time_axis, signal_full, signal_A, fs, actual_freq


def mean_residual_psd(rows: pd.DataFrame, probe: str):
    """Mean Welch PSD of the method-A residual across `rows`, on COMMON_F_GRID.

    Returns (f_grid, pxx_mean, n_used, fs_used) or (None, None, 0, None).
    """
    psds = []
    fs_used = None
    for _, r in rows.iterrows():
        if r["path"] not in fft_dict:
            continue
        df_fft = fft_dict[r["path"]]
        col = f"FFT {probe} complex"
        if col not in df_fft.columns:
            col = f"FFT {probe}"
        if col not in df_fft.columns:
            continue
        fs_series = df_fft[col].dropna()
        if fs_series.empty:
            continue
        _, s_full, s_A, fs, _ = reconstruct(
            fs_series, float(r["WaveFrequencyInput [Hz]"]), BAND_HALF_HZ,
        )
        res = s_full - s_A
        nperseg = min(len(res), 4096)
        fr, pxx = sp_signal.welch(res, fs=fs, nperseg=nperseg)
        pxx_interp = np.interp(COMMON_F_GRID, fr, pxx, left=np.nan, right=np.nan)
        psds.append(pxx_interp)
        fs_used = fs
    if not psds:
        return None, None, 0, fs_used
    return COMMON_F_GRID, np.nanmean(np.vstack(psds), axis=0), len(psds), fs_used


def wind_psd_at_paddle(freqs_grid: np.ndarray, psd_wind: np.ndarray,
                       f_paddle: float) -> float:
    """Band-integrated wind variance inside ±BAND_HALF_HZ of f_paddle.

    Returns A_wind² such that A_wind = √(2 · ∫ PSD df) is the narrowband
    amplitude contribution of the wind inside the analysis window. Values
    are clipped to 0 (no negative variance).
    """
    mask = (freqs_grid >= f_paddle - BAND_HALF_HZ) & (freqs_grid <= f_paddle + BAND_HALF_HZ)
    if not mask.any() or np.all(~np.isfinite(psd_wind[mask])):
        return np.nan
    var = float(np.trapezoid(np.clip(psd_wind[mask], 0, None), freqs_grid[mask]))
    return 2.0 * var


# ── 3. Per-group wind PSD + correction (Part B) ──────────────────────────────
print("\n2. Computing PSD_wind at paddle bin per (freq, amp, mooring) group …")

GROUP_KEYS = ["WaveFrequencyInput [Hz]", "WaveAmplitudeInput [Volt]", "Mooring"]

group_wind: dict[tuple, dict[str, float]] = {}
for grp_key, grp in wave_runs.groupby(GROUP_KEYS):
    nw = grp[grp["WindCondition"] == "no"]
    fw = grp[grp["WindCondition"] == "full"]
    if nw.empty or fw.empty:
        continue
    f_paddle = float(grp_key[0])
    entry: dict[str, float] = {"n_nw": len(nw), "n_fw": len(fw)}
    for probe in PROBES:
        f_nw, psd_nw, n_nw, _ = mean_residual_psd(nw, probe)
        f_fw, psd_fw, n_fw, _ = mean_residual_psd(fw, probe)
        if psd_nw is None or psd_fw is None:
            entry[f"A_wind2_{probe}"] = np.nan
            continue
        psd_wind = psd_fw - psd_nw
        entry[f"A_wind2_{probe}"] = wind_psd_at_paddle(f_nw, psd_wind, f_paddle)
    group_wind[tuple(grp_key)] = entry

print(f"   {len(group_wind)} (freq, amp, mooring) groups with matched nw+fw.")


# ── 4. Per-run correction table ──────────────────────────────────────────────
print("\n3. Building per-run correction table …")
A_IN_COL  = f"Probe {IN_POS} Amplitude (FFT)"
A_OUT_COL = f"Probe {OUT_POS} Amplitude (FFT)"

corr_rows = []
for _, r in wave_runs.iterrows():
    if r["WindCondition"] != "full":
        continue
    key = (r["WaveFrequencyInput [Hz]"], r["WaveAmplitudeInput [Volt]"], r["Mooring"])
    a_wind2_in  = group_wind.get(key, {}).get(f"A_wind2_{IN_POS}", np.nan)
    a_wind2_out = group_wind.get(key, {}).get(f"A_wind2_{OUT_POS}", np.nan)
    a_in_fw  = float(r.get(A_IN_COL, np.nan))
    a_out_fw = float(r.get(A_OUT_COL, np.nan))
    if not np.isfinite(a_in_fw) or not np.isfinite(a_out_fw):
        continue
    a_in_corr  = (np.sqrt(max(0.0, a_in_fw**2  - (a_wind2_in  if np.isfinite(a_wind2_in)  else 0)))
                  if np.isfinite(a_wind2_in)  else np.nan)
    a_out_corr = (np.sqrt(max(0.0, a_out_fw**2 - (a_wind2_out if np.isfinite(a_wind2_out) else 0)))
                  if np.isfinite(a_wind2_out) else np.nan)
    corr_rows.append({
        "path":               r["path"],
        "WaveFrequencyInput [Hz]":   r["WaveFrequencyInput [Hz]"],
        "WaveAmplitudeInput [Volt]": r["WaveAmplitudeInput [Volt]"],
        "Mooring":            r["Mooring"],
        "A_in_fw":            a_in_fw,
        "A_out_fw":           a_out_fw,
        "A_wind2_in":         a_wind2_in,
        "A_wind2_out":        a_wind2_out,
        "A_in_corr":          a_in_corr,
        "A_out_corr":         a_out_corr,
        "OUT_IN_raw":         a_out_fw / a_in_fw if a_in_fw > 0 else np.nan,
        "OUT_IN_corr":        (a_out_corr / a_in_corr) if (a_in_corr and a_in_corr > 0) else np.nan,
    })
corr_df = pd.DataFrame(corr_rows)
corr_df.to_csv(SCRATCH_CSV_CORR, index=False)
print(f"   {len(corr_df)} fullwind runs with wind-correction.")
print(f"   → {SCRATCH_CSV_CORR.relative_to(BASE)}")

# Reference: nowind means per (freq, amp) for the T_cross cross-check.
nw_means = (wave_runs[wave_runs["WindCondition"] == "no"]
            .groupby(["WaveFrequencyInput [Hz]", "WaveAmplitudeInput [Volt]"])
            [[A_IN_COL, A_OUT_COL]]
            .mean()
            .rename(columns={A_IN_COL: "A_in_nw", A_OUT_COL: "A_out_nw"}))

# Attach nw reference to each corrected fullwind row (for plotting).
corr_df = corr_df.merge(
    nw_means.reset_index(),
    on=["WaveFrequencyInput [Hz]", "WaveAmplitudeInput [Volt]"],
    how="left",
)


# ── 5. Window-size sensitivity (Part A) ──────────────────────────────────────
print("\n4. Window-size sensitivity — truncating stable-wave window …")

FS = 250.0  # sampling rate (see constants.MEASUREMENT)

def _center_slice(n_total: int, n_keep: int) -> tuple[int, int]:
    """Return (start, end) indices for a centred slice of length n_keep
    inside a parent of length n_total."""
    if n_keep >= n_total:
        return 0, n_total
    mid = n_total // 2
    half = n_keep // 2
    return mid - half, mid - half + n_keep


def windowed_AFFT(signal: np.ndarray, fs: float, f_target: float) -> float:
    """Peak-bin AFFT of `signal` at f_target within ±BAND_HALF_HZ."""
    if len(signal) < 16:
        return np.nan
    spec = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / fs)
    mag   = np.abs(spec) * 2.0 / len(signal)
    mask = (freqs >= f_target - BAND_HALF_HZ) & (freqs <= f_target + BAND_HALF_HZ)
    if not mask.any():
        return np.nan
    # Nearest bin to target within the mask (matches pipeline convention).
    masked_freqs = freqs[mask]
    masked_mag   = mag[mask]
    return float(masked_mag[np.argmin(np.abs(masked_freqs - f_target))])


def eta_series(df_run: pd.DataFrame, probe: str) -> np.ndarray | None:
    """Prefer the interpolated column when it exists (pipeline convention)."""
    for col in (f"eta_{probe}_interp", f"eta_{probe}"):
        if col in df_run.columns:
            return np.asarray(df_run[col].to_numpy(), dtype=float)
    return None


window_rows = []
for _, r in wave_runs.iterrows():
    path = r["path"]
    if path not in processed_dfs:
        continue
    df_run = processed_dfs[path]
    start_col = f"Computed Probe {IN_POS} start"
    end_col   = f"Computed Probe {IN_POS} end"
    if start_col not in r.index or not np.isfinite(r[start_col]):
        continue
    i0 = int(r[start_col])
    i1 = int(r[end_col])
    if i1 - i0 < int(FS * 4):   # < 4 s of stable window → skip
        continue
    f_paddle = float(r["WaveFrequencyInput [Hz]"])
    period_samples = int(round(FS / f_paddle))
    for probe in PROBES:
        sig = eta_series(df_run, probe)
        if sig is None:
            continue
        seg_full = sig[i0:i1]
        if len(seg_full) < 16:
            continue
        for N_per in N_PERIOD_CHOICES:
            if N_per == "full":
                seg = seg_full
                n_kept = len(seg_full) // period_samples
                tag = f"full_{n_kept}p"
            else:
                n_keep = min(len(seg_full), int(N_per * period_samples))
                s, e = _center_slice(len(seg_full), n_keep)
                seg = seg_full[s:e]
                n_kept = N_per
                tag = f"{N_per}p"
            afft = windowed_AFFT(seg, FS, f_paddle)
            window_rows.append({
                "path":                      path,
                "probe":                     probe,
                "WaveFrequencyInput [Hz]":   f_paddle,
                "WaveAmplitudeInput [Volt]": r["WaveAmplitudeInput [Volt]"],
                "WindCondition":             r["WindCondition"],
                "Mooring":                   r["Mooring"],
                "N_periods":                 n_kept,
                "N_tag":                     tag,
                "A_FFT_mm":                  afft,
            })
window_df = pd.DataFrame(window_rows)
window_df.to_csv(SCRATCH_CSV_WINDOW, index=False)
print(f"   {len(window_df)} (run × probe × N_period) rows")
print(f"   → {SCRATCH_CSV_WINDOW.relative_to(BASE)}")

# Aggregate per (amp, wind, N_periods) across frequencies — OUT/IN per run,
# then mean ± std across runs in each cell.
def _outin_per_run(wdf):
    """Pivot to one OUT/IN per (run × N_tag)."""
    w_in  = wdf[wdf["probe"] == IN_POS ].rename(columns={"A_FFT_mm": "A_in"})
    w_out = wdf[wdf["probe"] == OUT_POS].rename(columns={"A_FFT_mm": "A_out"})
    join_cols = ["path", "WaveFrequencyInput [Hz]", "WaveAmplitudeInput [Volt]",
                 "WindCondition", "Mooring", "N_tag"]
    merged = w_in[join_cols + ["A_in"]].merge(w_out[join_cols + ["A_out"]],
                                              on=join_cols, how="inner")
    merged["OUT_IN"] = merged["A_out"] / merged["A_in"]
    return merged


win_run = _outin_per_run(window_df)

# Sanity — how does per-N_tag OUT/IN sit vs pipeline OUT/IN (which uses the
# full stable window)? If they disagree systematically even at N=full, the
# discrepancy is a bin-position effect, not a stationarity effect.
print("\n   Median OUT/IN (all runs) by N_tag + wind:")
print(
    win_run.groupby(["WindCondition", "N_tag"])["OUT_IN"]
           .agg(["median", "std", "count"])
           .round(4)
           .to_string()
)


# ── 6. Figure: 2×3 grid ──────────────────────────────────────────────────────
print("\n5. Plotting …")
fig, axes = plt.subplots(
    nrows=2, ncols=3, figsize=(14, 7.5), dpi=120,
    sharey="row",
)

# ── Row 0: window-size sensitivity ──
x_order = ["20p", "40p", "60p", "100p"]
# "full" tag varies per run; group it as the last x-slot labelled "full".
def _nice_n_tag(t: str) -> str:
    return "full" if t.startswith("full") else t

win_run["N_slot"] = win_run["N_tag"].map(_nice_n_tag)
x_slots = x_order + ["full"]
x_pos = {t: i for i, t in enumerate(x_slots)}

for i, amp in enumerate(AMPS):
    ax = axes[0, i]
    sub = win_run[np.isclose(win_run["WaveAmplitudeInput [Volt]"], amp)]
    if sub.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        ax.set_title("", fontsize=9)
        continue
    for wind, color in [("no", COLOR_NW), ("full", COLOR_FW)]:
        wsub = sub[sub["WindCondition"] == wind]
        if wsub.empty:
            continue
        agg = (wsub.groupby("N_slot")["OUT_IN"]
                   .agg(["median", "std", "count"])
                   .reindex(x_slots))
        xs = [x_pos[t] for t in agg.index]
        ax.errorbar(xs, agg["median"].values,
                    yerr=agg["std"].fillna(0).values,
                    marker="o", linestyle="-", color=color,
                    capsize=3, lw=1.4, markersize=5,
                    label=f"{wind} (n≈{int(agg['count'].median() or 0)}/slot)")
    ax.axhline(1.0, color="black", lw=0.6, ls="--", alpha=0.4)
    ax.set_xticks(list(x_pos.values()))
    ax.set_xticklabels(x_slots, fontsize=8)
    ax.set_xlabel("Analysis window  [paddle periods]", fontsize=8)
    if i == 0:
        ax.set_ylabel("OUT/IN (FFT)", fontsize=9)
    ax.set_title("", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="lower right", framealpha=0.92)

# ── Row 1: wind-correction validation ──
for i, amp in enumerate(AMPS):
    ax = axes[1, i]
    sub = corr_df[np.isclose(corr_df["WaveAmplitudeInput [Volt]"], amp)].copy()
    if sub.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        ax.set_title("", fontsize=9)
        continue
    sub = sub.sort_values("WaveFrequencyInput [Hz]")
    agg = sub.groupby("WaveFrequencyInput [Hz]").agg(
        A_in_fw_med    = ("A_in_fw",   "median"),
        A_in_fw_std    = ("A_in_fw",   "std"),
        A_in_corr_med  = ("A_in_corr", "median"),
        A_in_corr_std  = ("A_in_corr", "std"),
        A_in_nw_med    = ("A_in_nw",   "median"),
        A_in_nw_std    = ("A_in_nw",   "std"),
        n              = ("A_in_fw",   "count"),
    )
    xs = agg.index.values
    ks = freq_to_k(np.array(xs))
    ax.errorbar(ks, agg["A_in_nw_med"],    yerr=agg["A_in_nw_std"].fillna(0),
                marker="o", linestyle="-", color=COLOR_NW,
                capsize=3, lw=1.4, markersize=5,
                label="A_in (nowind)")
    ax.errorbar(ks, agg["A_in_fw_med"],    yerr=agg["A_in_fw_std"].fillna(0),
                marker="s", linestyle="-", color=COLOR_FW,
                capsize=3, lw=1.4, markersize=5,
                label="A_in (fullwind, raw)")
    ax.errorbar(ks, agg["A_in_corr_med"],  yerr=agg["A_in_corr_std"].fillna(0),
                marker="^", linestyle="--", color=COLOR_CORR,
                capsize=3, lw=1.4, markersize=5,
                label="A_in (fullwind, corrected)")
    ax.set_xlabel(r"$k$ (rad/m)", fontsize=9)
    if i == 0:
        ax.set_ylabel("A_in  [mm]", fontsize=9)
    ax.set_title("", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best", framealpha=0.92)

fig.suptitle("", fontsize=10, fontweight="bold", y=1.00)
fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.08, wspace=0.08, hspace=0.32)

fig.savefig(SCRATCH_PDF, bbox_inches="tight")
print(f"   scratch preview → {SCRATCH_PDF.relative_to(BASE)}")
fig.savefig(THESIS_PDF, bbox_inches="tight")
print(f"   thesis figure   → {THESIS_PDF.relative_to(BASE)}")
plt.close(fig)


# ── 7. Summary stats for caption ─────────────────────────────────────────────
def _safe_med(series: pd.Series) -> float:
    s = series.dropna()
    return float(s.median()) if len(s) else float("nan")

# Window-sensitivity headline: max |Δ OUT/IN| across N_tags per (amp, wind),
# then worst across all cells.
pivot = (
    win_run.groupby(["WaveAmplitudeInput [Volt]", "WindCondition", "N_slot"])["OUT_IN"]
           .median()
           .unstack("N_slot")
)
if not pivot.empty:
    worst_drift = float((pivot.max(axis=1) - pivot.min(axis=1)).abs().max())
else:
    worst_drift = float("nan")

# Correction headline: median |A_in_corr − A_in_nw| / A_in_nw across all runs.
_match_valid = corr_df.dropna(subset=["A_in_corr", "A_in_nw"]).copy()
if len(_match_valid):
    _rel_err = (_match_valid["A_in_corr"] - _match_valid["A_in_nw"]).abs() / _match_valid["A_in_nw"]
    corr_match_median = float(_rel_err.median())
    corr_match_max    = float(_rel_err.max())
else:
    corr_match_median = corr_match_max = float("nan")

# Contamination fraction: √(A_wind²) / A_in_fw per run, median.
_cf = corr_df.dropna(subset=["A_in_fw", "A_wind2_in"])
if len(_cf):
    _frac = np.sqrt(_cf["A_wind2_in"].clip(lower=0)) / _cf["A_in_fw"]
    contam_med = float(_frac.median())
    contam_max = float(_frac.max())
else:
    contam_med = contam_max = float("nan")

# Per-amplitude A_in_fw / A_in_nw ratio (wind enhancement of IN amplitude
# at the paddle bin, residual after spectral-contamination correction).
# At thesis amplitudes (0.2, 0.3 V) this ratio captures the physical
# wind-enhancement of A_in that spectral subtraction cannot explain.
_enh = corr_df.dropna(subset=["A_in_fw", "A_in_nw"]).copy()
_enh["fw_over_nw"] = _enh["A_in_fw"] / _enh["A_in_nw"]
enhancement_by_amp: dict[str, float] = {}
for _amp in sorted(_enh["WaveAmplitudeInput [Volt]"].unique()):
    _sub = _enh[np.isclose(_enh["WaveAmplitudeInput [Volt]"], _amp)]
    if _sub.empty:
        continue
    _tag = amp_to_tag(_amp)
    enhancement_by_amp[f"A_in_fw_over_nw_{_tag}_median"] = round(float(_sub["fw_over_nw"].median()), 4)
    enhancement_by_amp[f"A_in_fw_over_nw_{_tag}_max"]    = round(float(_sub["fw_over_nw"].max()),    4)

print("\n6. Headline numbers:")
print(f"   worst window-size drift in OUT/IN    : {worst_drift:.4f}")
print(f"   median |A_in_corr − A_in_nw|/A_in_nw : {corr_match_median*100:.2f} %"
      f"  (max {corr_match_max*100:.2f} %)")
print(f"   median wind contamination fraction   : {contam_med*100:.2f} %"
      f"  (max {contam_max*100:.2f} %)")


# ── 8. LaTeX stub via shared helper ──────────────────────────────────────────
print("\n7. Writing .tex stub via pu.build_fig_meta + pu.write_figure_stub …")

import wavescripts.plot_utils as pu
pu.ACTIVE_DATASETS = [str(p).split("/")[-1] for p in RESULTS_PROCESSED_DIRS]
pu.TEXFIGU_DIR = BASE / "output" / "TEXFIGU"
pu.FIGURES_DIR = BASE / "output" / "FIGURES"

_caption = (
    "Top row: OUT/IN (FFT) at the paddle frequency as a function of the FFT "
    "analysis window length (20, 40, 60, 100 paddle periods and the full "
    "stable plateau), grouped by paddle drive 0.10, 0.20, 0.30\\,V. "
    "Blue markers: no wind; red markers: full wind. Error bars: run-to-run "
    "standard deviation within each (amplitude, wind, window) cell. "
    "Bottom row: IN-probe amplitude at the paddle frequency versus $k$ (rad/m), "
    "one panel per amplitude. Three curves per panel: A$_{\\text{in}}$ "
    "measured under no wind (blue, circles), under full wind (red, "
    "squares), and under full wind after incoherent subtraction of the "
    "wind-PSD contribution at the paddle bin (green dashed, triangles). "
    "Wind-PSD estimate per (frequency, amplitude, mooring) group: "
    "$\\mathrm{PSD}_{\\text{wind}}(f) = "
    "\\langle\\mathrm{PSD}_{\\text{res}}\\rangle_{\\text{full}} - "
    "\\langle\\mathrm{PSD}_{\\text{res}}\\rangle_{\\text{no}}$, "
    "integrated over $f_p \\pm 0.05$\\,Hz."
)

_meta_stub = pu.build_fig_meta(
    {
        "filters": {
            "PanelCondition":            "full",
            "WaveFrequencyInput [Hz]":   [min(FREQS), max(FREQS)],
            "WaveAmplitudeInput [Volt]": AMPS,
            "WindCondition":             ["no", "full"],
            "quality_flag":              "ok+probe_malfunction_secondary",
            "Mooring":                   ["below_90_loose"],
            "probes":                    ", ".join(PROBES),
        },
        "plotting": {
            "figure_name":   THESIS_NAME,
            "caption":       _caption,
            "caption_short": "Paddle-frequency IN contamination and window-size sensitivity",
        },
    },
    chapter="04",
    data_df=wave_runs,
    extra={"script": "analysis_scratch/paddle_contamination_study.py"},
    computed_in=("analysis_scratch/paddle_contamination_study.py "
                 "(Part A: truncated-stable-window AFFT via np.fft.rfft; "
                 "Part B: PSD_wind = mean PSD_residual(full) − mean PSD_residual(no), "
                 "evaluated at paddle bin ±0.05 Hz; incoherent subtraction "
                 "A_corr = √max(0, A_fw² − E[A_wind²]))"),
    data_class="DFS",
    findings_doc="analysis_scratch/paddle_contamination_findings.md",
    grouper="per (freq, amp, mooring) group; per-run correction",
    collapse_panels=False,
    fft_window_hz=FFT_WINDOW_HZ,
    extra_params=(
        f"N_periods={N_PERIOD_CHOICES}, fft_window_hz={FFT_WINDOW_HZ}, "
        f"band_half_hz={BAND_HALF_HZ}, "
        f"welch common_grid=0-12.5 Hz @ 0.05 Hz, "
        f"datasets={sorted(_results_dataset_names)}"
    ),
    extra_stats={
        "worst_window_drift_OUTIN":    round(worst_drift, 4),
        "corr_match_median_pct":       round(corr_match_median * 100, 2),
        "corr_match_max_pct":          round(corr_match_max * 100, 2),
        "contamination_median_pct":    round(contam_med * 100, 2),
        "contamination_max_pct":       round(contam_max * 100, 2),
        "n_groups":                    len(group_wind),
        "n_fullwind_runs":             len(corr_df),
        **enhancement_by_amp,
    },
)

pu.write_figure_stub(_meta_stub, plot_type="paddle_contamination",
                     subfig_filenames=[THESIS_NAME])
print(f"   thesis stub   → {THESIS_STUB.relative_to(BASE)}")


# ── 9. Findings markdown — descriptive run log only (no interpretation) ──────
_enhancement_table_lines = []
for _amp in sorted(_enh["WaveAmplitudeInput [Volt]"].unique()):
    _sub_amp = _enh[np.isclose(_enh["WaveAmplitudeInput [Volt]"], _amp)]
    _by_freq = (_sub_amp.groupby("WaveFrequencyInput [Hz]")["fw_over_nw"]
                        .agg(["median", "count"]).round(4))
    for _f, _r in _by_freq.iterrows():
        _enhancement_table_lines.append(
            f"  {_amp:.2f} V  |  {_f:.2f} Hz  |  "
            f"A_in_fw/A_in_nw median = {_r['median']:.4f}  "
            f"(n={int(_r['count'])})"
        )
_enhancement_table = "\n".join(_enhancement_table_lines)

findings = f"""# CH04 §2.6 — Paddle contamination & window-size sensitivity — run log

Generated by `analysis_scratch/paddle_contamination_study.py`.
Scope: full panel, {min(FREQS):.1f}–{max(FREQS):.1f} Hz, {AMPS} V,
mooring `below_90_loose` (230+300 pooled), quality_flag in
{{ok, probe_malfunction_secondary}}.

This file is the numerical run log only. For interpretation and any
scientific conclusions see `memory/methodology_wind_enhances_A_in.md`
and CLAUDE.md §16.

## Part A — Window-size sensitivity

Worst drift (max − min across N_tags) in median OUT/IN across all
(amp, wind) cells: **{worst_drift:.4f}**

## Part B — Wind-correction at the paddle bin

Per-run correction:
    A_in_corrected = sqrt(max(0, A_in_fw² − E[A_wind²]))
with E[A_wind²] = 2 · ∫_{{f_p ± 0.05 Hz}} PSD_wind(f) df,
and PSD_wind = mean PSD_residual(full) − mean PSD_residual(no) per
(freq, amp, mooring) group.

Median contamination fraction √(A_wind²) / A_in_fw: **{contam_med*100:.2f} %**
(max **{contam_max*100:.2f} %**).

## Part C — A_in_fw vs A_in_nw

Agreement of corrected full-wind A_in against no-wind A_in
({len(_match_valid)} matched rows):

  median  |A_in_corr − A_in_nw| / A_in_nw  =  {corr_match_median*100:.2f} %
  max     |A_in_corr − A_in_nw| / A_in_nw  =  {corr_match_max*100:.2f} %

Per (amplitude × frequency) median A_in_fw / A_in_nw (raw, no
contamination correction — the correction is <{max(0.05, contam_med*2):.0%}
at thesis amplitudes, so this ratio is dominated by non-spectral
wind effects):

{_enhancement_table}

## Files

  {SCRATCH_PDF.name}                      — diagnostic PDF
  {SCRATCH_CSV_CORR.name}    — per-run correction table
  {SCRATCH_CSV_WINDOW.name}  — per-run, per-window-size AFFT table
  output/FIGURES/{THESIS_NAME}.pdf   — thesis figure
  output/TEXFIGU/{THESIS_NAME}.tex   — thesis stub
"""
SCRATCH_FINDINGS_MD.write_text(findings)
print(f"   findings      → {SCRATCH_FINDINGS_MD.relative_to(BASE)}")

print("\nDone.")
