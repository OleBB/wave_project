"""
H&G window stability + per40 overlay — does SNARVEI reach the plateau?
=======================================================================

Extension of `hg_window_stability.py`: same 4x4 layout (4 conditions x
4 columns), now with **per40 runs overlaid on the same axes** as the
per240 runs. The question:

    At the same (freq, amp, wind), do per40's sliding AFFT curves
    ever reach the plateau that per240 sits on?

If per40's sliding curves converge to the per240 plateau within their
available window → SNARVEI is sampling a real plateau, the ratio is
trustworthy, and pooling is safe.

If per40's curves peak at a lower plateau than per240, or never
stabilise before the paddle stops (28 s at 1.4 Hz, 25 s at 1.6 Hz) →
SNARVEI is reading a transient. Per40's "plateau" is shorter than the
eyeballed window length, or never exists.

Per40 signals are shorter: paddle runs for only 40/f seconds, then the
wave decays. The sliding curve naturally terminates before 50T. To
preserve visual clarity:

  * per240 = existing colours (blue IN, red OUT, dark grey OUT/IN),
    solid lines, thicker.
  * per40  = gold (#F39C12) for IN, orange (#E67E22) for OUT, darker
    gold (#B9770E) for OUT/IN, dashed lines, slightly thinner.
  * The per40 paddle-stop vertical line (purple dashed) stays as before
    and now has physical meaning for the per40 overlay — the curve
    genuinely ends near there.

Scope: same cond4 below_90_loose, full panel, quality OK, thesis
frequencies. Both per40 and per240 must exist at the same (freq, amp,
wind) for a row to be useful.

Run from repo root:
    /opt/anaconda3/envs/draumkvedet/bin/python analysis_scratch/hg_window_stability_with_per40.py

Outputs (scratch, diagnostic only):
    analysis_scratch/hg_window_stability_with_per40.pdf
    analysis_scratch/hg_window_stability_with_per40_summary.csv
    analysis_scratch/hg_window_stability_with_per40_findings.md
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
SLIDING_T_LO   = 5.0     # earlier start — per40 plateau lives at ~15-25 s
SLIDING_T_HI   = 100.0

LENGTH_CHOICES_T = [2, 4, 6, 8, 10, 12, 15]   # fixed start=50T (per240 only)
PER40_PERIODS    = 40
PER240_THRESHOLD_T = 50   # runs with >= 50T go into the per240 pool

IN_PROBES = ["9373/170", "9373/340"]
OUT_PROBE = "12400/250"

CONDITIONS = [
    (1.4, 0.2, "full", "outlier +10.7%"),
    (1.6, 0.2, "full", "outlier -9.2%"),
    (1.6, 0.3, "full", "outlier -4.6%"),
    (1.4, 0.2, "no",   "baseline +0.8%"),
]

# Datasets — use the SAME cond4 thesis pair + the per40-rich 20260307 /
# 20260312 / 20260313 / 20260314 / 20260316 / 20260319 for per40 coverage.
# Mooring filter below narrows to below_90_loose.
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
_dataset_names = {p.name.removeprefix("PROCESSED-") for p in PROCESSED_DIRS}

SCRATCH_PDF = Path(__file__).parent / "hg_window_stability_with_per40.pdf"
SCRATCH_CSV = Path(__file__).parent / "hg_window_stability_with_per40_summary.csv"
SCRATCH_MD  = Path(__file__).parent / "hg_window_stability_with_per40_findings.md"


# ── Helpers ──────────────────────────────────────────────────────────────────
def fft_amp_at_freq(segment: np.ndarray, target_hz: float,
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
print("1. Loading meta across all thesis-relevant datasets …")
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
    & (combined_meta["quality_flag"].isin(["ok", "probe_malfunction_secondary"]))
].copy()
wave["N_input_periods"] = wave["WavePeriodInput"].astype(float)
print(f"   scope rows: {len(wave)} "
      f"(long n={int((wave['N_input_periods'] >= PER240_THRESHOLD_T).sum())}, "
      f"short n={int((wave['N_input_periods'] < PER240_THRESHOLD_T).sum())})")

print("\n2. Loading processed_dfs (~40-60 s — 14 datasets) …")
processed_dfs = load_processed_dfs(*PROCESSED_DIRS)
print(f"   {len(processed_dfs)} time-series cached")


# ── Pick runs per condition, split per40 vs per240 ────────────────────────────
def runs_for(f: float, a: float, w: str, long_runs: bool) -> pd.DataFrame:
    mask = (
        np.isclose(wave["WaveFrequencyInput [Hz]"], f)
        & np.isclose(wave["WaveAmplitudeInput [Volt]"], a)
        & (wave["WindCondition"] == w)
    )
    if long_runs:
        mask &= wave["N_input_periods"] >= PER240_THRESHOLD_T
    else:
        mask &= wave["N_input_periods"] < PER240_THRESHOLD_T
    return wave[mask].copy()

per240_runs: dict[tuple, pd.DataFrame] = {}
per40_runs:  dict[tuple, pd.DataFrame] = {}
for f, a, w, tag in CONDITIONS:
    per240_runs[(f, a, w)] = runs_for(f, a, w, long_runs=True)
    per40_runs [(f, a, w)] = runs_for(f, a, w, long_runs=False)
    print(f"   {f:.1f} Hz / {a:.2f} V / {w:>4} ({tag}): "
          f"per240 n={len(per240_runs[(f,a,w)])}, "
          f"per40 n={len(per40_runs[(f,a,w)])}")


# ── Compute sliding + length curves ──────────────────────────────────────────
print("\n3. Sliding AFFT + length-sensitivity AFFT per run …")

# Storage (tuple key includes run_type tag so per40/per240 don't clash)
#   sliding[(f,a,w,rt,path)][probe] = (ts, A)     rt ∈ {"per240", "per40"}
#   outin_slide[(f,a,w,rt,path)] = (ts, R)
#   length_scan[(f,a,w,rt,path)][probe] = {L_T: AFFT}
sliding: dict = {}
outin_slide: dict = {}
length_scan: dict = {}

summary_rows = []

for (f, a, w, tag) in CONDITIONS:
    window_s_10T = HG_WINDOW_N_T / f
    period_samples = int(round(FS / f))

    for rt, rs in [("per240", per240_runs[(f, a, w)]),
                   ("per40",  per40_runs[(f, a, w)])]:
        for _, r in rs.iterrows():
            path = r["path"]
            df_run = processed_dfs.get(path)
            if df_run is None:
                continue
            sig_in  = canonical_in_signal(df_run)
            sig_out = get_eta(df_run, OUT_PROBE)
            if sig_in is None or sig_out is None:
                continue

            # Adjust window length for per40: its paddle is alive for only
            # 40T ≈ 25-30 s, so 10T fits, 15T doesn't always. Still use 10T
            # for apples-to-apples comparison — the sliding curve will just
            # terminate earlier naturally.
            ts_in,  A_in  = sliding_afft(sig_in,  f, window_s_10T)
            ts_out, A_out = sliding_afft(sig_out, f, window_s_10T)
            sliding[(f, a, w, rt, path)] = {"IN": (ts_in, A_in), "OUT": (ts_out, A_out)}

            if len(ts_in) == len(ts_out) and len(ts_in) > 0:
                outin_slide[(f, a, w, rt, path)] = (ts_in, A_out / A_in)

            # Length scan only makes sense for per240 (needs signal past 50T).
            # For per40, skip — its signal ends before 50T at thesis freqs.
            if rt == "per240":
                length_scan[(f, a, w, rt, path)] = {"IN": {}, "OUT": {}}
                start_sample = HG_START_N_T * period_samples
                for L_T in LENGTH_CHOICES_T:
                    n_samples = L_T * period_samples
                    end_sample = start_sample + n_samples
                    if end_sample > len(sig_in) or end_sample > len(sig_out):
                        length_scan[(f, a, w, rt, path)]["IN" ][L_T] = np.nan
                        length_scan[(f, a, w, rt, path)]["OUT"][L_T] = np.nan
                        continue
                    length_scan[(f, a, w, rt, path)]["IN" ][L_T] = fft_amp_at_freq(sig_in [start_sample:end_sample], f)
                    length_scan[(f, a, w, rt, path)]["OUT"][L_T] = fft_amp_at_freq(sig_out[start_sample:end_sample], f)

            # Plateau stability at HG start (per240) or at SNARVEI (per40).
            if rt == "per240":
                hg_start_s = HG_START_N_T / f
                n_peri = 5.0 / f
                mask_in  = (ts_in  >= hg_start_s - n_peri) & (ts_in  <= hg_start_s + n_peri)
                mask_out = (ts_out >= hg_start_s - n_peri) & (ts_out <= hg_start_s + n_peri)
                region_label = "±5T around HG start"
            else:
                # Per40: check stability around SNARVEI window start
                sn_start_s = float(r.get(f"Computed Probe {IN_PROBES[0]} start", np.nan)) / FS
                n_peri = 3.0 / f   # narrower check for per40 — only a few periods available
                if np.isfinite(sn_start_s):
                    mask_in  = (ts_in  >= sn_start_s - n_peri) & (ts_in  <= sn_start_s + n_peri)
                    mask_out = (ts_out >= sn_start_s - n_peri) & (ts_out <= sn_start_s + n_peri)
                else:
                    mask_in = mask_out = np.array([], dtype=bool)
                region_label = "±3T around SNARVEI start"

            in_med  = float(np.nanmedian(A_in [mask_in]))  if mask_in.any()  else np.nan
            in_std  = float(np.nanstd(A_in [mask_in]))     if mask_in.any()  else np.nan
            out_med = float(np.nanmedian(A_out[mask_out])) if mask_out.any() else np.nan
            out_std = float(np.nanstd(A_out[mask_out]))    if mask_out.any() else np.nan

            summary_rows.append({
                "path":          path,
                "run_type":      rt,
                "freq_hz":       f,
                "amp_V":         a,
                "wind":          w,
                "tag":           tag,
                "region":        region_label,
                "IN_plateau_med":  in_med,
                "IN_plateau_std":  in_std,
                "IN_plateau_cv":   in_std / in_med if (in_med and in_med > 0) else np.nan,
                "OUT_plateau_med": out_med,
                "OUT_plateau_std": out_std,
                "OUT_plateau_cv":  out_std / out_med if (out_med and out_med > 0) else np.nan,
            })

summary = pd.DataFrame(summary_rows)
summary.to_csv(SCRATCH_CSV, index=False)
print(f"   summary → {SCRATCH_CSV.relative_to(BASE)}")
print("\n4. Plateau CV per (freq, amp, wind, run_type):")
print(summary.groupby(["freq_hz", "amp_V", "wind", "run_type"])[
    ["IN_plateau_cv", "OUT_plateau_cv"]
].median().round(4).to_string())


# ── Plot ─────────────────────────────────────────────────────────────────────
print("\n5. Plotting …")
fig, axes = plt.subplots(4, 4, figsize=(18, 14), dpi=120)

# Per240 colour family
COL_IN_240    = "#2E86AB"
COL_OUT_240   = "#E74C3C"
COL_RATIO_240 = "#444444"
# Per40 colour family (warm/gold so it reads as a distinct overlay)
COL_IN_40     = "#F39C12"
COL_OUT_40    = "#E67E22"
COL_RATIO_40  = "#B9770E"
COL_HG        = "#2ECC71"
COL_SNARV     = "#999999"
COL_PSTOP     = "#8E44AD"

for row_idx, (f, a, w, tag) in enumerate(CONDITIONS):
    rs240 = per240_runs[(f, a, w)]
    rs40  = per40_runs [(f, a, w)]
    hg_start_s = HG_START_N_T / f
    hg_end_s   = HG_END_N_T   / f
    per40_stop = PER40_PERIODS / f

    # SNARVEI band — use per40 run's median, since SNARVEI was eyeballed on per40.
    def _snarvei_band(rs: pd.DataFrame, pos: str) -> tuple[float, float] | None:
        st = rs[f"Computed Probe {pos} start"].dropna()
        et = rs[f"Computed Probe {pos} end"].dropna()
        if st.empty or et.empty:
            return None
        return (float(st.median()) / FS, float(et.median()) / FS)

    # Prefer per40 for SNARVEI band (that's what the eyeball was done on).
    rs_for_snarvei = rs40 if not rs40.empty else rs240
    sn_in  = _snarvei_band(rs_for_snarvei, IN_PROBES[0])
    sn_out = _snarvei_band(rs_for_snarvei, OUT_PROBE)

    def _plot_sliding(ax, probe_tag, col_240, col_40, y_label):
        for _, r in rs240.iterrows():
            key = (f, a, w, "per240", r["path"])
            if key not in sliding:
                continue
            ts, A = sliding[key][probe_tag]
            ax.plot(ts, A, color=col_240, lw=1.2, alpha=0.75)
        for _, r in rs40.iterrows():
            key = (f, a, w, "per40", r["path"])
            if key not in sliding:
                continue
            ts, A = sliding[key][probe_tag]
            ax.plot(ts, A, color=col_40, lw=1.0, alpha=0.75, ls="--")
        ax.axvspan(hg_start_s, hg_end_s, color=COL_HG, alpha=0.20, lw=0)
        sn = sn_in if probe_tag == "IN" else sn_out
        if sn is not None:
            ax.axvspan(sn[0], sn[1], color=COL_SNARV, alpha=0.15, lw=0)
        ax.axvline(per40_stop, color=COL_PSTOP, ls="--", lw=1.2, alpha=0.8)
        ax.set_xlim(SLIDING_T_LO, SLIDING_T_HI)
        ax.grid(True, alpha=0.3)
        if row_idx == 3:
            ax.set_xlabel("window start [s from wavemaker start]", fontsize=9)
        if y_label:
            ax.set_ylabel(
                f"{f:.1f} Hz, {a:.2f} V, {w}\n({tag})\n"
                f"n₂₄₀={len(rs240)}  n₄₀={len(rs40)}\n{y_label}",
                fontsize=8,
            )

    # Col 1 — sliding AFFT at IN
    ax = axes[row_idx, 0]
    _plot_sliding(ax, "IN", COL_IN_240, COL_IN_40, y_label="AFFT_IN [mm]")
    if row_idx == 0:
        ax.set_title("IN (canonical mean of parallel probes)", fontsize=10)

    # Col 2 — sliding AFFT at OUT
    ax = axes[row_idx, 1]
    _plot_sliding(ax, "OUT", COL_OUT_240, COL_OUT_40, y_label="")
    if row_idx == 0:
        ax.set_title("OUT (12400/250)", fontsize=10)

    # Col 3 — length scan at start=50T (per240 only — per40 can't reach)
    ax = axes[row_idx, 2]
    for _, r in rs240.iterrows():
        key = (f, a, w, "per240", r["path"])
        if key not in length_scan:
            continue
        L = length_scan[key]
        Ls = LENGTH_CHOICES_T
        A_in_vals  = [L["IN"].get(lt, np.nan)  for lt in Ls]
        A_out_vals = [L["OUT"].get(lt, np.nan) for lt in Ls]
        ax.plot(Ls, A_in_vals,  color=COL_IN_240,  lw=1.0, alpha=0.75, marker="o", ms=4)
        ax.plot(Ls, A_out_vals, color=COL_OUT_240, lw=1.0, alpha=0.75, marker="s", ms=4)
    ax.axvline(HG_WINDOW_N_T, color=COL_HG, ls="-", lw=1.2, alpha=0.7)
    ax.set_xticks(LENGTH_CHOICES_T)
    if row_idx == 3:
        ax.set_xlabel("window length [T]", fontsize=9)
    ax.grid(True, alpha=0.3)
    if row_idx == 0:
        ax.set_title(
            "AFFT vs window length at start=50T  (per240 only)\n"
            "blue=IN, red=OUT, green=H&G 10T", fontsize=9,
        )

    # Col 4 — sliding OUT/IN
    ax = axes[row_idx, 3]
    for _, r in rs240.iterrows():
        key = (f, a, w, "per240", r["path"])
        if key not in outin_slide:
            continue
        ts, R = outin_slide[key]
        ax.plot(ts, R, color=COL_RATIO_240, lw=1.2, alpha=0.75)
    for _, r in rs40.iterrows():
        key = (f, a, w, "per40", r["path"])
        if key not in outin_slide:
            continue
        ts, R = outin_slide[key]
        ax.plot(ts, R, color=COL_RATIO_40, lw=1.0, alpha=0.75, ls="--")
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
        ax.set_title("OUT/IN (sliding)  —  the money plot", fontsize=9)


from matplotlib.patches import Patch
from matplotlib.lines import Line2D
legend_handles = [
    Patch(facecolor=COL_HG,    alpha=0.20, label=f"H&G window [{HG_START_N_T}T, {HG_END_N_T}T]"),
    Patch(facecolor=COL_SNARV, alpha=0.15, label="SNARVEI window (per40 eyeball)"),
    Line2D([], [], color=COL_PSTOP, ls="--", lw=1.2, label="per40 paddle stop (40/f)"),
    Line2D([], [], color=COL_IN_240,  lw=1.4, label="per240 IN"),
    Line2D([], [], color=COL_IN_40,   lw=1.2, ls="--", label="per40 IN"),
    Line2D([], [], color=COL_OUT_240, lw=1.4, label="per240 OUT"),
    Line2D([], [], color=COL_OUT_40,  lw=1.2, ls="--", label="per40 OUT"),
    Line2D([], [], color=COL_RATIO_240, lw=1.4, label="per240 OUT/IN"),
    Line2D([], [], color=COL_RATIO_40,  lw=1.2, ls="--", label="per40 OUT/IN"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=5, fontsize=8,
           bbox_to_anchor=(0.5, -0.005), frameon=True)

fig.suptitle(
    "H&G window stability + per40 overlay — does SNARVEI reach the per240 plateau?  "
    f"(10T sliding, {SLIDING_STEP_S:.1f} s step)",
    fontsize=12, fontweight="bold", y=0.995,
)
fig.subplots_adjust(left=0.06, right=0.99, top=0.95, bottom=0.07, hspace=0.22, wspace=0.18)
fig.savefig(SCRATCH_PDF, bbox_inches="tight")
print(f"   → {SCRATCH_PDF.relative_to(BASE)}")
plt.close(fig)


# ── Findings markdown ────────────────────────────────────────────────────────
print("\n6. Writing findings markdown …")
lines = [
    "# H&G stability + per40 overlay — run log",
    "",
    "Generated by `analysis_scratch/hg_window_stability_with_per40.py`.  ",
    "Scope: full panel, below_90_loose, quality OK, thesis freqs {1.3-1.6} Hz.  ",
    "",
    "Per240: stability region is ±5T around H&G start (50T).  ",
    "Per40:  stability region is ±3T around SNARVEI start (from meta).",
    "",
    "## Cohort per condition",
    "",
    "| row | freq [Hz] | amp [V] | wind | tag | n_per240 | n_per40 |",
    "|-----|-----------|---------|------|-----|---------:|--------:|",
]
for (f, a, w, tag) in CONDITIONS:
    n240 = len(per240_runs[(f, a, w)])
    n40  = len(per40_runs[(f, a, w)])
    lines.append(f"| {CONDITIONS.index((f,a,w,tag))+1} | {f:.1f} | {a:.2f} | {w} | {tag} | {n240} | {n40} |")
lines += [
    "",
    "## Plateau CV median per (freq, amp, wind, run_type)",
    "",
    "CV = std / median of 10T sliding AFFT within the stability region.  ",
    "Low CV → curve is on a plateau. Large CV → still transient.",
    "",
    "```",
    summary.groupby(["freq_hz", "amp_V", "wind", "run_type"])[
        ["IN_plateau_cv", "OUT_plateau_cv"]
    ].median().round(4).to_string(),
    "```",
    "",
    "## Files",
    "",
    f"  {SCRATCH_PDF.name}  — 4x4 diagnostic figure with per40 overlay",
    f"  {SCRATCH_CSV.name}  — per-run plateau stats",
    "",
    "Interpretation intentionally omitted — user-eyeball of the PDF decides whether "
    "SNARVEI reaches the per240 plateau and whether pooling remains safe.",
]
SCRATCH_MD.write_text("\n".join(lines))
print(f"   → {SCRATCH_MD.relative_to(BASE)}")

print("\nDone.")
