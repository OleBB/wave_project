"""
H&G window stability & length-sensitivity — per240 thesis conditions
=====================================================================

Focused diagnostic in support of the H&G column rollout in processor.py.
Before we commit to [50T, 60T] as the thesis standard window, this
script answers two questions on one 4x4 figure:

  (A) Does H&G [50T, 60T] sit on a flat AFFT plateau at the three
      fullwind outlier conditions from `per40_vs_per240_outin.pdf`
      (where |Δ OUT/IN| between SNARVEI and H&G was 7-11 %)?
      If yes → H&G is the honest number and the 10 % disagreement is
               real time-varying A_in (wind enhancement).
      If no  → H&G itself needs shifting before we enshrine it.

  (B) How sensitive is AFFT to window length at fixed start=50T? If a
      line plotted against window length L ∈ {2T, 4T, 6T, 8T, 10T,
      12T, 15T} is flat, H&G is stable and we could defensibly use
      shorter windows. If it has a kink at L=10T (the no-leakage
      length), 10T really is special.

Four conditions, one row each:
  Row 1 — 1.4 Hz, 0.2 V, full wind  (outlier: Δ OUT/IN = +10.7 %)
  Row 2 — 1.6 Hz, 0.2 V, full wind  (outlier: Δ OUT/IN = −9.2 %)
  Row 3 — 1.6 Hz, 0.3 V, full wind  (outlier: Δ OUT/IN = −4.6 %)
  Row 4 — 1.4 Hz, 0.2 V, no   wind  (baseline: agreement = +0.8 %)

Four columns:
  Col 1 — sliding AFFT at IN (canonical = mean of 9373/170 + 9373/340)
          with 10T window, stepped 0.5 s, 10-100 s window-start range.
  Col 2 — sliding AFFT at OUT (12400/250), same sweep.
  Col 3 — AFFT vs window length L at fixed start=50T, both probes.
  Col 4 — sliding OUT/IN derived from cols 1 and 2 — the money plot.

Overlays on cols 1, 2, 4:
  * green shaded band = H&G window [50T, 60T]
  * grey  shaded band = SNARVEI window (from pipeline meta per probe)
  * red   dashed line = per40 paddle-stop time (40/f), shows how much
    of H&G's region is structurally unavailable to per40 runs

This script uses per240 runs only (per40 signal does not extend past
~30 s at thesis frequencies). Same cond4 lowrange datasets as the
earlier scripts.

Run from repo root:
    /opt/anaconda3/envs/draumkvedet/bin/python analysis_scratch/hg_window_stability.py

Outputs (scratch only — diagnostic, not thesis-bound):
    analysis_scratch/hg_window_stability.pdf
    analysis_scratch/hg_window_stability_summary.csv
    analysis_scratch/hg_window_stability_findings.md
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
FS             = 250.0
FFT_BAND_HZ    = 0.05

HG_START_N_T   = 50
HG_END_N_T     = 60
HG_WINDOW_N_T  = HG_END_N_T - HG_START_N_T   # 10T

SLIDING_STEP_S = 0.5
SLIDING_T_LO   = 10.0
SLIDING_T_HI   = 100.0

LENGTH_CHOICES_T = [2, 4, 6, 8, 10, 12, 15]   # window lengths at fixed start=50T
PER40_PERIODS    = 40                         # for paddle-stop overlay

# Probes
IN_PROBES = ["9373/170", "9373/340"]   # canonical IN mean
OUT_PROBE = "12400/250"

# Conditions (freq_hz, amp_V, wind, label_tag)
CONDITIONS = [
    (1.4, 0.2, "full", "outlier +10.7%"),
    (1.6, 0.2, "full", "outlier -9.2%"),
    (1.6, 0.3, "full", "outlier -4.6%"),
    (1.4, 0.2, "no",   "baseline +0.8%"),
]

RESULTS_PROCESSED_DIRS = [
    Path("waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange"),
    Path("waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"),
]
_results_dataset_names = {p.name.removeprefix("PROCESSED-") for p in RESULTS_PROCESSED_DIRS}

SCRATCH_PDF = Path(__file__).parent / "hg_window_stability.pdf"
SCRATCH_CSV = Path(__file__).parent / "hg_window_stability_summary.csv"
SCRATCH_MD  = Path(__file__).parent / "hg_window_stability_findings.md"


# ── Helpers ──────────────────────────────────────────────────────────────────
def fft_amp_at_freq(segment: np.ndarray, target_hz: float,
                    fs: float = FS, band_hz: float = FFT_BAND_HZ) -> float:
    """Peak-bin AFFT (pipeline convention). Lifted from huseby_grue_window.py."""
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


def sliding_afft(signal: np.ndarray, target_hz: float,
                 window_s: float, step_s: float = SLIDING_STEP_S,
                 fs: float = FS):
    """Slide `window_s` across signal. Returns (t_start_s, A_mm)."""
    N_win = int(round(window_s * fs))
    step  = int(round(step_s * fs))
    if N_win >= len(signal):
        return np.array([]), np.array([])
    starts = np.arange(0, len(signal) - N_win + 1, step)
    t_start = starts / fs
    A = np.full(len(starts), np.nan)
    for i, s in enumerate(starts):
        A[i] = fft_amp_at_freq(signal[s:s + N_win], target_hz, fs)
    return t_start, A


def canonical_in_signal(df_run: pd.DataFrame) -> np.ndarray | None:
    """Mean η across the two parallel IN probes at the same longitudinal
    distance. Matches the canonical pipeline IN definition."""
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


# ── Load data ────────────────────────────────────────────────────────────────
print("1. Loading meta_results + fft_dict + processed_dfs …")
combined_meta, _, _, _ = load_analysis_data(*RESULTS_PROCESSED_DIRS, load_processed=False)
meta_results = combined_meta[
    combined_meta["path"].apply(lambda p: any(d in str(p) for d in _results_dataset_names))
].copy()
meta_results["Mooring"] = meta_results["Mooring"].replace({
    "below_90_loose230": "below_90_loose",
    "below_90_loose300": "below_90_loose",
})
wave = meta_results[
    meta_results["WaveFrequencyInput [Hz]"].notna()
    & (meta_results["WaveFrequencyInput [Hz]"] > 0)
    & (meta_results["PanelCondition"] == "full")
    & (meta_results["Mooring"] == "below_90_loose")
    & (meta_results["quality_flag"].isin(["ok", "probe_malfunction_secondary"]))
    & (meta_results["WavePeriodInput"].astype(float) >= 50)   # long runs only
].copy()
print(f"   long (per240) thesis-scope runs: {len(wave)}")

print("\n2. Loading processed_dfs (~20 s) …")
processed_dfs = load_processed_dfs(*RESULTS_PROCESSED_DIRS)
print(f"   {len(processed_dfs)} time-series cached")


# ── Pick runs for each condition ─────────────────────────────────────────────
def runs_for(f: float, a: float, w: str) -> pd.DataFrame:
    return wave[
        np.isclose(wave["WaveFrequencyInput [Hz]"], f)
        & np.isclose(wave["WaveAmplitudeInput [Volt]"], a)
        & (wave["WindCondition"] == w)
    ].copy()

condition_runs: dict[tuple, pd.DataFrame] = {}
for f, a, w, tag in CONDITIONS:
    rs = runs_for(f, a, w)
    condition_runs[(f, a, w)] = rs
    print(f"   {f:.1f} Hz / {a:.2f} V / {w:>4}  ({tag}): {len(rs)} runs")


# ── Compute curves + numbers ─────────────────────────────────────────────────
print("\n3. Sliding AFFT + length-sensitivity AFFT per run …")

# Storage:
#   sliding[(f,a,w,path)][probe_name] = (t_start_s, A_mm)
#     probe_name ∈ {"IN", "OUT", "IN_9373_170", "IN_9373_340"}
#   length_scan[(f,a,w,path)][probe_name] = dict(L_T → AFFT_mm)
#   outin_slide[(f,a,w,path)] = (t_start_s, OUT/IN)
sliding: dict[tuple, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
length_scan: dict[tuple, dict[str, dict[int, float]]] = {}
outin_slide: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}

summary_rows = []

for (f, a, w, tag) in CONDITIONS:
    rs = condition_runs[(f, a, w)]
    window_s_10T = HG_WINDOW_N_T / f
    period_samples = int(round(FS / f))

    for _, r in rs.iterrows():
        path = r["path"]
        df_run = processed_dfs.get(path)
        if df_run is None:
            continue
        sig_in  = canonical_in_signal(df_run)
        sig_out = get_eta(df_run, OUT_PROBE)
        if sig_in is None or sig_out is None:
            continue

        # Sliding AFFT at 10T window
        ts_in,  A_in  = sliding_afft(sig_in,  f, window_s_10T)
        ts_out, A_out = sliding_afft(sig_out, f, window_s_10T)
        sliding[(f, a, w, path)] = {"IN": (ts_in, A_in), "OUT": (ts_out, A_out)}

        # Per-parallel-probe sliding too (cols 1 optional overlay if useful)
        for pprobe in IN_PROBES:
            s = get_eta(df_run, pprobe)
            if s is None:
                continue
            t_p, A_p = sliding_afft(s, f, window_s_10T)
            sliding[(f, a, w, path)][f"IN_{pprobe.replace('/', '_')}"] = (t_p, A_p)

        # Sliding OUT/IN (align times — both should step identically)
        if len(ts_in) == len(ts_out):
            outin_slide[(f, a, w, path)] = (ts_in, A_out / A_in)

        # Length-sensitivity: fixed start=50T, varying length
        length_scan[(f, a, w, path)] = {"IN": {}, "OUT": {}}
        start_sample = HG_START_N_T * period_samples
        for L_T in LENGTH_CHOICES_T:
            n_samples = L_T * period_samples
            end_sample = start_sample + n_samples
            if end_sample > len(sig_in) or end_sample > len(sig_out):
                length_scan[(f, a, w, path)]["IN"][L_T]  = np.nan
                length_scan[(f, a, w, path)]["OUT"][L_T] = np.nan
                continue
            length_scan[(f, a, w, path)]["IN"][L_T]  = fft_amp_at_freq(sig_in[start_sample:end_sample], f)
            length_scan[(f, a, w, path)]["OUT"][L_T] = fft_amp_at_freq(sig_out[start_sample:end_sample], f)

        # Plateau-in-HG check: median & std of sliding AFFT WITHIN [50T, 60T − 10T]
        # Actually H&G is just one window at start=50T. But we want to check
        # stability of the sliding curve in a small neighbourhood of that start.
        # Use the window starts inside [50T − 5T, 50T + 5T] = ±5T around HG start.
        hg_start_s = HG_START_N_T / f
        n_peri = 5.0 / f       # ±5T in seconds
        mask_in  = (ts_in  >= hg_start_s - n_peri) & (ts_in  <= hg_start_s + n_peri)
        mask_out = (ts_out >= hg_start_s - n_peri) & (ts_out <= hg_start_s + n_peri)
        in_med  = float(np.nanmedian(A_in [mask_in]))  if mask_in.any()  else np.nan
        in_std  = float(np.nanstd(A_in [mask_in]))     if mask_in.any()  else np.nan
        out_med = float(np.nanmedian(A_out[mask_out])) if mask_out.any() else np.nan
        out_std = float(np.nanstd(A_out[mask_out]))    if mask_out.any() else np.nan

        # AFFT at exactly HG start=50T, length=10T
        i0 = HG_START_N_T * period_samples
        i1 = HG_END_N_T   * period_samples
        a_in_hg  = fft_amp_at_freq(sig_in [i0:i1], f) if i1 <= len(sig_in)  else np.nan
        a_out_hg = fft_amp_at_freq(sig_out[i0:i1], f) if i1 <= len(sig_out) else np.nan

        summary_rows.append({
            "path":  path,
            "freq_hz":     f,
            "amp_V":       a,
            "wind":        w,
            "tag":         tag,
            "HG_AFFT_IN":  a_in_hg,
            "HG_AFFT_OUT": a_out_hg,
            "HG_OUT_IN":   a_out_hg / a_in_hg if (a_in_hg and a_in_hg > 0) else np.nan,
            "IN_plateau_med_pm5T":  in_med,
            "IN_plateau_std_pm5T":  in_std,
            "IN_plateau_cv_pm5T":   in_std / in_med if (in_med and in_med > 0) else np.nan,
            "OUT_plateau_med_pm5T": out_med,
            "OUT_plateau_std_pm5T": out_std,
            "OUT_plateau_cv_pm5T":  out_std / out_med if (out_med and out_med > 0) else np.nan,
        })

summary = pd.DataFrame(summary_rows)
summary.to_csv(SCRATCH_CSV, index=False)
print(f"   summary → {SCRATCH_CSV.relative_to(BASE)}")
print("\n4. Plateau stability (CV = std/median of 10T sliding AFFT within "
      "±5T of HG start):")
print(summary.groupby(["freq_hz", "amp_V", "wind"])[
    ["IN_plateau_cv_pm5T", "OUT_plateau_cv_pm5T"]
].median().round(4).to_string())


# ── Plot: 4 rows (conditions) × 4 cols ───────────────────────────────────────
print("\n5. Plotting …")
fig, axes = plt.subplots(4, 4, figsize=(18, 14), dpi=120)

COL_IN    = "#2E86AB"   # blue — IN amplitude
COL_OUT   = "#E74C3C"   # red  — OUT amplitude
COL_RATIO = "#444444"   # dark grey — OUT/IN
COL_HG    = "#2ECC71"   # green — H&G band
COL_SNARV = "#999999"   # grey — SNARVEI band
COL_PSTOP = "#8E44AD"   # purple — per40 paddle stop

for row_idx, (f, a, w, tag) in enumerate(CONDITIONS):
    rs = condition_runs[(f, a, w)]
    if rs.empty:
        for col_idx in range(4):
            ax = axes[row_idx, col_idx]
            ax.text(0.5, 0.5, "no runs", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
            ax.set_xticks([]); ax.set_yticks([])
        continue

    hg_start_s = HG_START_N_T / f
    hg_end_s   = HG_END_N_T   / f
    per40_stop = PER40_PERIODS / f

    # SNARVEI window from meta (per-probe) — use median across the runs in
    # this condition so we draw one band per panel; any per-run variation
    # should be small.
    def _snarvei_band(pos_col_prefix: str) -> tuple[float, float] | None:
        st = rs[f"Computed Probe {pos_col_prefix} start"].dropna()
        et = rs[f"Computed Probe {pos_col_prefix} end"].dropna()
        if st.empty or et.empty:
            return None
        return (float(st.median()) / FS, float(et.median()) / FS)

    sn_in  = _snarvei_band(IN_PROBES[0])    # 9373/170 — primary IN
    sn_out = _snarvei_band(OUT_PROBE)

    # ─ Col 1: sliding AFFT at IN ─
    ax = axes[row_idx, 0]
    for _, r in rs.iterrows():
        path = r["path"]
        if (f, a, w, path) not in sliding:
            continue
        ts, A = sliding[(f, a, w, path)]["IN"]
        ax.plot(ts, A, color=COL_IN, lw=1.0, alpha=0.6)
        # Also plot per-parallel-probe faintly
        for pp in IN_PROBES:
            key = f"IN_{pp.replace('/', '_')}"
            if key in sliding[(f, a, w, path)]:
                t_p, A_p = sliding[(f, a, w, path)][key]
                ax.plot(t_p, A_p, color=COL_IN, lw=0.6, alpha=0.25, ls="--")
    ax.axvspan(hg_start_s, hg_end_s, color=COL_HG, alpha=0.20, lw=0)
    if sn_in is not None:
        ax.axvspan(sn_in[0], sn_in[1], color=COL_SNARV, alpha=0.15, lw=0)
    ax.axvline(per40_stop, color=COL_PSTOP, ls="--", lw=1.2, alpha=0.8)
    ax.set_xlim(SLIDING_T_LO, SLIDING_T_HI)
    ax.grid(True, alpha=0.3)
    if row_idx == 3:
        ax.set_xlabel("window start [s from wavemaker start]", fontsize=9)
    ax.set_ylabel(f"{f:.1f} Hz, {a:.2f} V, {w}\n({tag})\nAFFT_IN [mm]", fontsize=9)
    if row_idx == 0:
        ax.set_title("IN (canonical mean of parallel probes)", fontsize=10)

    # ─ Col 2: sliding AFFT at OUT ─
    ax = axes[row_idx, 1]
    for _, r in rs.iterrows():
        path = r["path"]
        if (f, a, w, path) not in sliding:
            continue
        ts, A = sliding[(f, a, w, path)]["OUT"]
        ax.plot(ts, A, color=COL_OUT, lw=1.0, alpha=0.6)
    ax.axvspan(hg_start_s, hg_end_s, color=COL_HG, alpha=0.20, lw=0)
    if sn_out is not None:
        ax.axvspan(sn_out[0], sn_out[1], color=COL_SNARV, alpha=0.15, lw=0)
    ax.axvline(per40_stop, color=COL_PSTOP, ls="--", lw=1.2, alpha=0.8)
    ax.set_xlim(SLIDING_T_LO, SLIDING_T_HI)
    ax.grid(True, alpha=0.3)
    if row_idx == 3:
        ax.set_xlabel("window start [s from wavemaker start]", fontsize=9)
    if row_idx == 0:
        ax.set_title("OUT (12400/250)", fontsize=10)

    # ─ Col 3: AFFT vs window length at fixed start=50T ─
    ax = axes[row_idx, 2]
    for _, r in rs.iterrows():
        path = r["path"]
        if (f, a, w, path) not in length_scan:
            continue
        L = length_scan[(f, a, w, path)]
        Ls = LENGTH_CHOICES_T
        A_in_vals  = [L["IN"].get(lt, np.nan)  for lt in Ls]
        A_out_vals = [L["OUT"].get(lt, np.nan) for lt in Ls]
        ax.plot(Ls, A_in_vals,  color=COL_IN,  lw=1.0, alpha=0.6, marker="o", ms=4)
        ax.plot(Ls, A_out_vals, color=COL_OUT, lw=1.0, alpha=0.6, marker="s", ms=4)
    # Mark L=10T (H&G choice)
    ax.axvline(HG_WINDOW_N_T, color=COL_HG, ls="-", lw=1.2, alpha=0.7, label="H&G 10T")
    ax.set_xticks(LENGTH_CHOICES_T)
    ax.set_xlabel("window length [T]", fontsize=9)
    ax.grid(True, alpha=0.3)
    if row_idx == 0:
        ax.set_title("AFFT vs window length at start=50T\n"
                     "(blue = IN, red = OUT)", fontsize=9)

    # ─ Col 4: sliding OUT/IN ─
    ax = axes[row_idx, 3]
    for _, r in rs.iterrows():
        path = r["path"]
        if (f, a, w, path) not in outin_slide:
            continue
        ts, R = outin_slide[(f, a, w, path)]
        ax.plot(ts, R, color=COL_RATIO, lw=1.0, alpha=0.6)
    ax.axhline(1.0, color="black", lw=0.6, ls="--", alpha=0.4)
    ax.axvspan(hg_start_s, hg_end_s, color=COL_HG, alpha=0.20, lw=0)
    if sn_in is not None:
        ax.axvspan(sn_in[0], sn_in[1], color=COL_SNARV, alpha=0.15, lw=0)
    ax.axvline(per40_stop, color=COL_PSTOP, ls="--", lw=1.2, alpha=0.8)
    ax.set_xlim(SLIDING_T_LO, SLIDING_T_HI)
    ax.set_ylim(0, 1.2)
    ax.grid(True, alpha=0.3)
    if row_idx == 3:
        ax.set_xlabel("window start [s from wavemaker start]", fontsize=9)
    if row_idx == 0:
        ax.set_title("OUT/IN (sliding)\nthe money plot", fontsize=9)

# Legend on a sneaky axes outside the grid — use axes[0, 0] with proxy artists.
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
legend_handles = [
    Patch(facecolor=COL_HG,     alpha=0.20, label=f"H&G window [{HG_START_N_T}T, {HG_END_N_T}T]"),
    Patch(facecolor=COL_SNARV,  alpha=0.15, label="SNARVEI window (pipeline)"),
    Line2D([], [], color=COL_PSTOP, ls="--", lw=1.2, label=f"per40 paddle stop (40/f)"),
    Line2D([], [], color=COL_IN,  lw=1.2, label="IN canonical (mean of parallels)"),
    Line2D([], [], color=COL_IN,  lw=0.8, ls="--", label="IN per-parallel probe"),
    Line2D([], [], color=COL_OUT, lw=1.2, label="OUT (12400/250)"),
    Line2D([], [], color=COL_RATIO, lw=1.2, label="OUT/IN sliding"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=4, fontsize=8,
           bbox_to_anchor=(0.5, -0.01), frameon=True)

fig.suptitle(
    "H&G window stability & length-sensitivity at outlier + baseline conditions  "
    f"(10T sliding window, {SLIDING_STEP_S:.1f} s step)",
    fontsize=12, fontweight="bold", y=0.995,
)
fig.subplots_adjust(left=0.06, right=0.99, top=0.95, bottom=0.06, hspace=0.22, wspace=0.18)
fig.savefig(SCRATCH_PDF, bbox_inches="tight")
print(f"   → {SCRATCH_PDF.relative_to(BASE)}")
plt.close(fig)


# ── Findings markdown (descriptive run log only) ─────────────────────────────
print("\n6. Writing findings markdown …")
lines = [
    "# H&G window stability & length-sensitivity — run log",
    "",
    "Generated by `analysis_scratch/hg_window_stability.py`.  ",
    f"Scope: per240 long runs only (`WavePeriodInput >= {HG_START_N_T + HG_END_N_T - HG_WINDOW_N_T}T`), "
    "cond4 below_90_loose, full panel, quality OK.",
    "",
    "## Conditions probed",
    "",
    "| row | freq [Hz] | amp [V] | wind | tag | n_runs |",
    "|-----|-----------|---------|------|-----|--------|",
]
for (f, a, w, tag) in CONDITIONS:
    n = len(condition_runs[(f, a, w)])
    lines.append(f"| {CONDITIONS.index((f,a,w,tag))+1} | {f:.1f} | {a:.2f} | {w} | {tag} | {n} |")
lines += [
    "",
    "## Plateau stability metric",
    "",
    "CV = std/median of the 10T sliding AFFT within ±5T of the H&G "
    f"start (i.e. window-start ∈ [{HG_START_N_T-5}T, {HG_START_N_T+5}T]).  ",
    "Low CV → H&G sits on a stable plateau. High CV → transient, the "
    "10 % disagreement is window-dependent.",
    "",
    "Median CV per (freq, amp, wind) (across runs):",
    "",
    "```",
    summary.groupby(["freq_hz", "amp_V", "wind"])[
        ["IN_plateau_cv_pm5T", "OUT_plateau_cv_pm5T"]
    ].median().round(4).to_string(),
    "```",
    "",
    "## Files",
    "",
    f"  {SCRATCH_PDF.name}    — 4x4 diagnostic figure",
    f"  {SCRATCH_CSV.name}    — per-run plateau stats + HG AFFTs",
    "",
    "Interpretation intentionally omitted — see user eyeball of the PDF "
    "for the verdict on whether H&G is a stable standard.",
]
SCRATCH_MD.write_text("\n".join(lines))
print(f"   → {SCRATCH_MD.relative_to(BASE)}")

print("\nDone.")
