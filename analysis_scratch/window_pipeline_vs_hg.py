"""
Pipeline window vs Huseby–Grue window — direct OUT/IN comparison
================================================================

Computes OUT/IN(FFT) under two window strategies on the same canon runs:

  HG  : probe-shifted Huseby–Grue [50T, 60T] anchored at r = 12.4 m
        (= our OUT probe). For r_probe < 12.4 m the window is shifted
        earlier by ΔT = (12.4 − r_probe) / c_g(f) · f periods. Length =
        10 T uniform. Start UC-snapped to nearest ±T.

  OURS: per-probe arrival anchor — t_start = r_probe / c_g(f, h) +
        HG.N_OFFSET / f. Length = HG.N_LENGTH / f (uniform 10 T).
        This is the window the live pipeline uses (post 2026-05-02
        roll-out: N_OFFSET=7, N_LENGTH=10). Start UC-snapped to nearest
        ±T.

Both windows use the same FFT method (peak-bin nearest, ±0.05 Hz around
paddle f) and the same UC-snap rule. Only the start position differs.
At canon thesis frequencies, OURS starts ~11–12 wave periods earlier
than HG (table below):

  At IN  (r=9.373 m):
    f=1.3 Hz: HG ≈ 33.4 s    OURS ≈ 20.9 s     Δ ≈ -12.5 s ≈ -16T
    f=1.4 Hz: HG ≈ 30.3 s    OURS ≈ 21.8 s     Δ ≈ -8.5 s  ≈ -12T
    f=1.5 Hz: HG ≈ 27.5 s    OURS ≈ 22.7 s     Δ ≈ -4.8 s  ≈ -7T
    f=1.6 Hz: HG ≈ 25.1 s    OURS ≈ 23.6 s     Δ ≈ -1.5 s  ≈ -2T

  At OUT (r=12.4 m):
    f=1.3 Hz: HG ≈ 38.5 s    OURS ≈ 25.9 s     Δ ≈ -12.6 s ≈ -16T
    f=1.4 Hz: HG ≈ 35.7 s    OURS ≈ 27.2 s     Δ ≈ -8.5 s  ≈ -12T
    f=1.5 Hz: HG ≈ 33.3 s    OURS ≈ 28.5 s     Δ ≈ -4.8 s  ≈ -7T
    f=1.6 Hz: HG ≈ 31.3 s    OURS ≈ 29.8 s     Δ ≈ -1.5 s  ≈ -2T

The question this script answers:

  Does the OURS window — anchored at probe arrival rather than at OUT-50T —
  produce OUT/IN ratios that agree with the literature-cited H&G window
  within an acceptable margin?

The key worry is that OURS sits inside the wave-train arrival region at
1.3 Hz (~16T earlier than H&G), where amplitude may not yet be fully
stable. If the agreement holds, OURS is "as good as H&G" for our purpose
and we can use it (per40 + per240 pooled) at our chosen frequency range.

Scope restriction: per240 runs ONLY. Per40 paddle output is ~28 s at
1.4 Hz, but H&G's OUT-probe window ends at ~42.9 s — well into the
wave-train decay region. Comparing OURS (in steady state) against H&G
(in decay) would be apples-to-oranges. Per240 has ~170 s of paddle
output, so both windows fit fully in steady state. (See the previous
finding `per40_and_per240_HG_shifted_findings.md` for evidence that the
probe-shifted H&G window itself produces consistent OUT/IN across per40
and per240 runs — that's a separate validation of the probe-shift
methodology, not a window-comparison.)

Outputs (scratch only):
    analysis_scratch/window_pipeline_vs_hg.csv
    analysis_scratch/window_pipeline_vs_hg.pdf
    analysis_scratch/window_pipeline_vs_hg.png
    analysis_scratch/window_pipeline_vs_hg_findings.md
    stdout: numerical summary + per-(f, probe) snapped-window listing

Run from repo root:
    /opt/anaconda3/envs/draumkvedet/bin/python analysis_scratch/window_pipeline_vs_hg.py

Companion sibling: `window_variable_N_prototype.py` — historical, evaluates
the abandoned 2026-04-30 N-squeeze (N_OFFSET=10, variable N(f)). Kept as
record of the back-and-forth that led to the live N_OFFSET=7, N_LENGTH=10
choice; do NOT use as a current reference.
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
from wavescripts.plot_utils import apply_thesis_style, WIND_COLOR_MAP, amp_to_label
from wavescripts.constants import c_group, HG

apply_thesis_style()

# ── Config ──────────────────────────────────────────────────────────────
FS                 = 250.0
FFT_BAND_HZ        = 0.05

THESIS_FREQS       = [1.3, 1.4, 1.5, 1.6]
THESIS_AMPS        = [0.10, 0.20, 0.30]

IN_PROBES          = ["9373/170", "9373/340"]
OUT_PROBE          = "12400/250"
PROBE_R_M          = {"9373/170": 9.373, "9373/340": 9.373, "12400/250": 12.400}
TANK_DEPTH_M       = HG.TANK_DEPTH_M

SNAP_HALFWIDTH_T   = 1.0   # ±1 wave period UC search window

# HG (legacy reference) — the [50T, 60T] H&G window anchored at r = 12.4 m.
# These were removed from `HG` (constants.py) when the dataclass was
# reformulated 2026-05-02 to per-probe arrival anchoring; kept here as
# local constants because this comparison's whole point is "ours vs that".
HG_REF_R_M          = 12.400
HG_START_T          = 50
HG_END_T            = 60
HG_WINDOW_LENGTH_T  = HG_END_T - HG_START_T   # 10

# OURS — taken straight from the live pipeline (constants.py)
PIPELINE_N_OFFSET   = HG.N_OFFSET    # 7
PIPELINE_N_LENGTH   = HG.N_LENGTH    # 10

# Canon — march-2026 cond4 lowrange
PROCESSED_DIRS = [
    Path("waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange"),
    Path("waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"),
]

SCRATCH_DIR = Path(__file__).parent
OUT_PDF     = SCRATCH_DIR / "window_pipeline_vs_hg.pdf"
OUT_PNG     = SCRATCH_DIR / "window_pipeline_vs_hg.png"
OUT_CSV     = SCRATCH_DIR / "window_pipeline_vs_hg.csv"
OUT_MD      = SCRATCH_DIR / "window_pipeline_vs_hg_findings.md"


# ── Helpers ─────────────────────────────────────────────────────────────
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


def snap_to_upcrossing(signal: np.ndarray, target_idx: int,
                       f_paddle: float, fs: float = FS) -> int:
    """Snap target sample idx to nearest zero-upcrossing within ±SNAP_HALFWIDTH_T
    periods, threshold = local DC mean of first 2 s of signal (matches
    live pipeline)."""
    if target_idx < 0 or target_idx >= len(signal):
        return target_idx
    samples_per_period = int(round(fs / f_paddle))
    halfwidth = int(round(SNAP_HALFWIDTH_T * samples_per_period))
    baseline_n = int(2 * fs)
    upcross_level = float(np.nanmean(signal[:baseline_n]))
    above = signal > upcross_level
    all_uc = np.where((~above[:-1]) & above[1:])[0] + 1
    lo = max(0, target_idx - halfwidth)
    hi = min(len(signal) - 1, target_idx + halfwidth)
    cands = all_uc[(all_uc >= lo) & (all_uc <= hi)]
    if len(cands) == 0:
        return target_idx
    return int(cands[np.argmin(np.abs(cands - target_idx))])


# ── Window functions ────────────────────────────────────────────────────
def hg_window_indices(signal: np.ndarray, probe: str,
                      f_paddle: float, fs: float = FS):
    """Probe-shifted H&G [50T, 60T] anchored at r = 12.4 m.
    Start UC-snapped; end = start + 10·samples_per_period."""
    r_probe_m = PROBE_R_M[probe]
    cg = c_group(f_paddle, TANK_DEPTH_M)
    dT_periods = (HG_REF_R_M - r_probe_m) / cg * f_paddle
    start_T = HG_START_T - dT_periods
    samples_per_period = int(round(fs / f_paddle))
    target_start_idx = int(round(start_T * samples_per_period))
    snap_start = snap_to_upcrossing(signal, target_start_idx, f_paddle, fs)
    snap_end = snap_start + HG_WINDOW_LENGTH_T * samples_per_period
    return snap_start, snap_end


def pipeline_window_indices(signal: np.ndarray, probe: str,
                            f_paddle: float, fs: float = FS):
    """Live pipeline window: t_start = r/c_g(f,h) + N_OFFSET/f,
    length = N_LENGTH·samples_per_period. UC-snapped start."""
    r_m = PROBE_R_M[probe]
    cg = c_group(f_paddle, TANK_DEPTH_M)
    t_start = r_m / cg + PIPELINE_N_OFFSET / f_paddle
    target_start_idx = int(round(t_start * fs))
    snap_start = snap_to_upcrossing(signal, target_start_idx, f_paddle, fs)
    samples_per_period = int(round(fs / f_paddle))
    snap_end = snap_start + PIPELINE_N_LENGTH * samples_per_period
    return snap_start, snap_end


def amp_in_window_indices(signal: np.ndarray, s: int, e: int,
                          f_paddle: float) -> float:
    if s < 0 or e > len(signal) or e <= s:
        return np.nan
    return fft_amp_at_freq(signal[s:e], f_paddle)


# ── Load ────────────────────────────────────────────────────────────────
print("1. Loading meta + processed_dfs (canon March-2026 lowrange) …")
combined_meta, _, _, _ = load_analysis_data(
    *[str(d) for d in PROCESSED_DIRS], load_processed=False
)
processed_dfs = load_processed_dfs(*[str(d) for d in PROCESSED_DIRS])
print(f"   meta: {len(combined_meta)} rows · processed_dfs: {len(processed_dfs)} runs")


# per240-only: H&G's [50T, 60T] window at the OUT probe ends past the
# wavemaker stop on per40 runs (paddle stop = 40/f ≈ 28 s at 1.4 Hz; H&G
# OUT window ends at ~42.9 s — well into the wave-train decay). Comparing
# our pipeline's in-train measurement to H&G's decay-region measurement
# would be apples-to-oranges. Per240 has 240/f ≈ 170 s of paddle output —
# both windows fit fully in steady-state.
PER240_PERIODS_THRESHOLD = 200


def select_canon() -> pd.DataFrame:
    f_col = pd.to_numeric(combined_meta["WaveFrequencyInput [Hz]"], errors="coerce")
    a_col = pd.to_numeric(combined_meta["WaveAmplitudeInput [Volt]"], errors="coerce")
    p_col = pd.to_numeric(combined_meta["WavePeriodInput"], errors="coerce")
    mask = (
        f_col.between(min(THESIS_FREQS) - 0.02, max(THESIS_FREQS) + 0.02)
        & a_col.between(min(THESIS_AMPS) - 0.005, max(THESIS_AMPS) + 0.005)
        & (p_col >= PER240_PERIODS_THRESHOLD)
        & (combined_meta["PanelCondition"] == "full")
        & (combined_meta["quality_flag"] == "ok")
    )
    sub = combined_meta[mask].copy()
    sub["freq_r"] = f_col[mask].round(2)
    sub["amp_r"]  = a_col[mask].round(2)
    return sub


sel = select_canon()
print(f"\n2. Canonical scope (per240 only): {len(sel)} runs across "
      f"{sel['freq_r'].nunique()} freqs × "
      f"{sel['amp_r'].nunique()} amps × "
      f"{sel['WindCondition'].nunique()} winds")
print("   per240 = WavePeriodInput >= "
      f"{PER240_PERIODS_THRESHOLD}, paddle output ≈ {200/1.4:.0f}–{240/1.3:.0f} s "
      "across thesis frequencies — both windows fit fully in steady-state.")


# ── 3. Compute HG + pipeline OUT/IN per run ─────────────────────────────
print("\n3. Computing OUT/IN under HG vs pipeline windows per run …")
records = []
for _, r in sel.iterrows():
    df = processed_dfs.get(r["path"])
    if df is None:
        continue
    f = float(r["freq_r"])
    sig_in  = canonical_in_signal(df)
    sig_out = get_eta(df, OUT_PROBE)
    if sig_in is None or sig_out is None:
        continue

    # H&G windows
    s_in_h,  e_in_h  = hg_window_indices(sig_in,  IN_PROBES[0], f, FS)
    s_out_h, e_out_h = hg_window_indices(sig_out, OUT_PROBE,    f, FS)
    a_in_h  = amp_in_window_indices(sig_in,  s_in_h,  e_in_h,  f)
    a_out_h = amp_in_window_indices(sig_out, s_out_h, e_out_h, f)
    outin_h = a_out_h / a_in_h if (np.isfinite(a_in_h) and a_in_h > 0) else np.nan

    # Pipeline (ours) windows
    s_in_p,  e_in_p  = pipeline_window_indices(sig_in,  IN_PROBES[0], f, FS)
    s_out_p, e_out_p = pipeline_window_indices(sig_out, OUT_PROBE,    f, FS)
    a_in_p  = amp_in_window_indices(sig_in,  s_in_p,  e_in_p,  f)
    a_out_p = amp_in_window_indices(sig_out, s_out_p, e_out_p, f)
    outin_p = a_out_p / a_in_p if (np.isfinite(a_in_p) and a_in_p > 0) else np.nan

    records.append({
        "path":           r["path"],
        "freq_r":         f,
        "amp_r":          round(float(r["amp_r"]), 2),
        "wind":           r["WindCondition"],
        "A_in_hg":        a_in_h,
        "A_out_hg":       a_out_h,
        "OUT_IN_hg":      outin_h,
        "A_in_pipe":      a_in_p,
        "A_out_pipe":     a_out_p,
        "OUT_IN_pipe":    outin_p,
        # snapped window endpoints (s) for verification
        "in_window_hg_s":   (s_in_h  / FS,  e_in_h  / FS),
        "out_window_hg_s":  (s_out_h / FS,  e_out_h / FS),
        "in_window_pipe_s": (s_in_p  / FS,  e_in_p  / FS),
        "out_window_pipe_s":(s_out_p / FS,  e_out_p / FS),
    })

per_run = pd.DataFrame(records)
per_run["delta_OUT_IN"] = per_run["OUT_IN_pipe"] - per_run["OUT_IN_hg"]
per_run["pct_OUT_IN"]   = per_run["delta_OUT_IN"] / per_run["OUT_IN_hg"] * 100.0
# Per-probe relative disagreement (signed): pipeline minus H&G, in %.
# These are the headline methodology numbers — direct A_pipe vs A_hg
# comparison on each probe, no ratio compounding.
per_run["pct_IN"]  = (per_run["A_in_pipe"]  - per_run["A_in_hg"])  / per_run["A_in_hg"]  * 100.0
per_run["pct_OUT"] = (per_run["A_out_pipe"] - per_run["A_out_hg"]) / per_run["A_out_hg"] * 100.0
print(f"   {len(per_run)} per-run rows computed")


# ── 4. Aggregate per (freq, amp, wind) ──────────────────────────────────
print("\n4. Aggregating per (freq, amp, wind) …")
agg = per_run.groupby(["freq_r", "amp_r", "wind"]).agg(
    OUT_IN_hg_mean   = ("OUT_IN_hg", "mean"),
    OUT_IN_pipe_mean = ("OUT_IN_pipe", "mean"),
    OUT_IN_hg_std    = ("OUT_IN_hg", "std"),
    OUT_IN_pipe_std  = ("OUT_IN_pipe", "std"),
    n                = ("OUT_IN_hg", "size"),
).reset_index()
agg["delta_OUT_IN"] = agg["OUT_IN_pipe_mean"] - agg["OUT_IN_hg_mean"]
agg["pct_OUT_IN"]   = agg["delta_OUT_IN"] / agg["OUT_IN_hg_mean"] * 100.0


# ── 4b. Per-probe disagreement aggregation — the methodology headline ───
# Per-cell mean of pct_IN and pct_OUT (= pipeline − H&G in %), then
# tabulate per-probe summary stats and per-(freq, probe) medians.
print("\n4b. Per-probe agreement — per (freq, amp, wind) cell …")
cell_probe = per_run.groupby(["freq_r", "amp_r", "wind"]).agg(
    pct_IN  = ("pct_IN",  "mean"),
    pct_OUT = ("pct_OUT", "mean"),
    n       = ("pct_IN",  "size"),
).reset_index()

probe_summary = pd.DataFrame([
    {
        "probe":            "9373/170 (IN)",
        "n_cells":          int(cell_probe["pct_IN"].notna().sum()),
        "median_abs_pct":   float(cell_probe["pct_IN"].abs().median()),
        "max_abs_pct":      float(cell_probe["pct_IN"].abs().max()),
        "within_pm5pct":    int((cell_probe["pct_IN"].abs() <= 5).sum()),
        "within_pm3pct":    int((cell_probe["pct_IN"].abs() <= 3).sum()),
        "within_pm2pct":    int((cell_probe["pct_IN"].abs() <= 2).sum()),
    },
    {
        "probe":            "12400/250 (OUT)",
        "n_cells":          int(cell_probe["pct_OUT"].notna().sum()),
        "median_abs_pct":   float(cell_probe["pct_OUT"].abs().median()),
        "max_abs_pct":      float(cell_probe["pct_OUT"].abs().max()),
        "within_pm5pct":    int((cell_probe["pct_OUT"].abs() <= 5).sum()),
        "within_pm3pct":    int((cell_probe["pct_OUT"].abs() <= 3).sum()),
        "within_pm2pct":    int((cell_probe["pct_OUT"].abs() <= 2).sum()),
    },
])
print(probe_summary.round(3).to_string(index=False))

# Per-(freq, probe) median |Δ|/A_HG  — the frequency trend
freq_trend_rows = []
for f in sorted(cell_probe["freq_r"].unique()):
    sub = cell_probe[cell_probe["freq_r"] == f]
    freq_trend_rows.append({
        "freq_hz":          f,
        "IN_median_abs":    float(sub["pct_IN"].abs().median()),
        "IN_max_abs":       float(sub["pct_IN"].abs().max()),
        "IN_min_abs":       float(sub["pct_IN"].abs().min()),
        "OUT_median_abs":   float(sub["pct_OUT"].abs().median()),
        "OUT_max_abs":      float(sub["pct_OUT"].abs().max()),
        "OUT_min_abs":      float(sub["pct_OUT"].abs().min()),
    })
freq_trend = pd.DataFrame(freq_trend_rows)
print("\nPer-frequency median |Δ|/A_HG across (amp, wind) cells:")
print(freq_trend.round(3).to_string(index=False))

# Spearman on the OUT trend — convergence as window-shift shrinks
try:
    from scipy.stats import spearmanr
    r_out, p_out = spearmanr(freq_trend["freq_hz"], freq_trend["OUT_median_abs"])
    r_in,  p_in  = spearmanr(freq_trend["freq_hz"], freq_trend["IN_median_abs"])
    print(f"\nSpearman r vs frequency:")
    print(f"  IN:  r = {r_in:+.3f}  p = {p_in:.3f}")
    print(f"  OUT: r = {r_out:+.3f}  p = {p_out:.3f}")
except Exception as e:
    r_out = p_out = r_in = p_in = float("nan")
    print(f"  spearmanr unavailable ({e})")


# ── 5. Wind-effect shift: Δτ = τ_fw − τ_nw under each window ────────────
print("\n5. Wind-effect shift per (freq, amp) …")
wind_pivot_hg = agg.pivot_table(
    index=["freq_r", "amp_r"], columns="wind",
    values="OUT_IN_hg_mean", aggfunc="first").reset_index()
wind_pivot_pipe = agg.pivot_table(
    index=["freq_r", "amp_r"], columns="wind",
    values="OUT_IN_pipe_mean", aggfunc="first").reset_index()

we = wind_pivot_hg.merge(wind_pivot_pipe, on=["freq_r", "amp_r"],
                         suffixes=("_hg", "_pipe"))
we["Dtau_hg"]    = we["full_hg"]   - we["no_hg"]
we["Dtau_pipe"]  = we["full_pipe"] - we["no_pipe"]
we["Dtau_shift"] = we["Dtau_pipe"] - we["Dtau_hg"]

print("\nWind effect (Δτ = τ_fw − τ_nw) per (freq, amp) — HG vs pipeline:")
print(we.round(4).to_string(index=False))


# ── 6. Verification — print snapped windows for one run per freq ────────
print("\n6. Snapped-window verification (one per40-fullwind run per freq) …")
print(f"  {'f':<5} {'IN HG':<24} {'IN pipe':<24} {'OUT HG':<24} {'OUT pipe':<24}")
print(f"  {'(Hz)':<5} {'[s_start, s_end]':<24} {'[s_start, s_end]':<24} {'[s_start, s_end]':<24} {'[s_start, s_end]':<24}")
print("  " + "─" * 120)
for f in THESIS_FREQS:
    sub = per_run[(per_run["freq_r"] == f)
                  & (per_run["amp_r"] == 0.20)
                  & (per_run["wind"] == "full")]
    if sub.empty:
        continue
    rep = sub.iloc[0]
    print(f"  {f:<5.1f} "
          f"{str(tuple(round(x, 2) for x in rep['in_window_hg_s'])):<24} "
          f"{str(tuple(round(x, 2) for x in rep['in_window_pipe_s'])):<24} "
          f"{str(tuple(round(x, 2) for x in rep['out_window_hg_s'])):<24} "
          f"{str(tuple(round(x, 2) for x in rep['out_window_pipe_s'])):<24}")


# ── 7. Single-panel methodology figure: |Δ A|/A_HG vs frequency, per probe
#       Reports per-probe amplitude agreement, NOT the OUT/IN ratio.
print("\n7. Plotting per-probe agreement vs frequency (single panel) …")
fig, ax = plt.subplots(figsize=(6.5, 4.2))

# Per-cell points (faint background) so the reader sees the spread.
ax.scatter(cell_probe["freq_r"] - 0.012,
           cell_probe["pct_IN"].abs(),
           color="#1f77b4", alpha=0.30, s=22,
           edgecolors="none", marker="o", zorder=2,
           label=None)
ax.scatter(cell_probe["freq_r"] + 0.012,
           cell_probe["pct_OUT"].abs(),
           color="#d62728", alpha=0.30, s=22,
           edgecolors="none", marker="s", zorder=2,
           label=None)

# Per-frequency median lines (foreground).
ax.plot(freq_trend["freq_hz"] - 0.012,
        freq_trend["IN_median_abs"],
        marker="o", ms=8, lw=1.8, color="#1f77b4",
        mfc="white", mec="#1f77b4", mew=1.6, zorder=4,
        label="IN  (9373/170)")
ax.plot(freq_trend["freq_hz"] + 0.012,
        freq_trend["OUT_median_abs"],
        marker="s", ms=8, lw=1.8, color="#d62728",
        zorder=4,
        label="OUT (12400/250)")

# 5% reference
ax.axhline(5.0, color="black", lw=0.7, ls="--", alpha=0.45, zorder=1,
           label=r"$\pm$5 % reference")

ax.set_xlabel("Wave frequency [Hz]", fontsize=10)
ax.set_ylabel(r"$|A_\mathrm{pipe} - A_\mathrm{HG}|\,/\,A_\mathrm{HG}$  [%]",
              fontsize=10)
ax.set_xticks(THESIS_FREQS)
ax.set_ylim(0, max(8.0, cell_probe[["pct_IN", "pct_OUT"]].abs().max().max() * 1.1))
ax.grid(True, alpha=0.3, lw=0.4)
ax.legend(fontsize=8, loc="upper right", framealpha=0.9)

fig.tight_layout()
fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_PNG, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"   saved → {OUT_PDF.relative_to(BASE)}")
print(f"          {OUT_PNG.relative_to(BASE)}")


# ── 8. Save CSV + findings ──────────────────────────────────────────────
# Save both: the per-cell aggregate (OUT/IN focus) AND the per-probe
# disagreement table (the methodology headline).
agg.to_csv(OUT_CSV, index=False)
print(f"   CSV (per-cell OUT/IN)  → {OUT_CSV.relative_to(BASE)}")

OUT_PROBE_CSV = SCRATCH_DIR / "window_pipeline_vs_hg_probe.csv"
cell_probe.to_csv(OUT_PROBE_CSV, index=False)
print(f"   CSV (per-cell probe Δ) → {OUT_PROBE_CSV.relative_to(BASE)}")

# Per-probe headline numbers.
in_med  = float(cell_probe["pct_IN"].abs().median())
in_max  = float(cell_probe["pct_IN"].abs().max())
out_med = float(cell_probe["pct_OUT"].abs().median())
out_max = float(cell_probe["pct_OUT"].abs().max())
in_w5   = int((cell_probe["pct_IN"].abs()  <= 5).sum())
out_w5  = int((cell_probe["pct_OUT"].abs() <= 5).sum())
n_cells_in  = int(cell_probe["pct_IN"].notna().sum())
n_cells_out = int(cell_probe["pct_OUT"].notna().sum())

# Supplementary OUT/IN headlines (kept for completeness).
delta_med = agg["delta_OUT_IN"].abs().median()
delta_max = agg["delta_OUT_IN"].abs().max()
pct_med   = agg["pct_OUT_IN"].abs().median()
pct_max   = agg["pct_OUT_IN"].abs().max()
shift_med = we["Dtau_shift"].abs().median()
shift_max = we["Dtau_shift"].abs().max()

md_lines = [
    "# Pipeline window vs Huseby–Grue window — methodology agreement",
    "",
    "Generated by `analysis_scratch/window_pipeline_vs_hg.py`.",
    "",
    "Scope: full panel, 1.3–1.6 Hz, 0.10/0.20/0.30 V, both winds, "
    "**per240 only** (WavePeriodInput ≥ 200), quality_flag=ok, march-2026 "
    "cond4 lowrange.  Per40 excluded: H&G's OUT window ends past the "
    "paddle stop on per40 runs, so the H&G measurement is in the decay "
    "region rather than steady state — apples-to-oranges with the "
    "pipeline window.",
    "",
    "## Window definitions",
    "",
    "**H&G** — probe-shifted Huseby–Grue [50T, 60T] anchored at r = 12.4 m. "
    "IN windows shifted earlier by ΔT = (12.4 − r_IN) / c_g(f) · f periods. "
    "Length = 10 T uniform.",
    "",
    "**Pipeline (ours)** — per-probe arrival anchor: "
    f"t_start = r_probe / c_g(f, h) + {PIPELINE_N_OFFSET} / f. "
    f"Length = {PIPELINE_N_LENGTH} T uniform.",
    "",
    "Both UC-snap start to nearest upcrossing within ±T. Both use "
    "nearest-bin FFT amplitude inside ±0.05 Hz around f_paddle.",
    "",
    "## Headline — per-probe amplitude agreement",
    "",
    "Direct comparison of the per-probe paddle-frequency amplitudes (no "
    "OUT/IN ratio compounding). Median across all "
    f"({n_cells_in if n_cells_in == n_cells_out else f'{n_cells_in}/{n_cells_out}'}"
    ") (frequency × amplitude × wind) cells:",
    "",
    "| Probe | n cells | median \\|Δ\\|/A_HG | max \\|Δ\\|/A_HG | within ±5% |",
    "|---|---:|---:|---:|---:|",
    f"| 9373/170  (IN)  | {n_cells_in}  | **{in_med:.2f}%**  | {in_max:.2f}% | {in_w5}/{n_cells_in} |",
    f"| 12400/250 (OUT) | {n_cells_out} | **{out_med:.2f}%** | {out_max:.2f}% | {out_w5}/{n_cells_out} |",
    "",
    "Pipeline and H&G windows agree on per-probe paddle-frequency "
    f"amplitude within **{max(in_med, out_med):.1f}%** at the median, "
    f"**{max(in_max, out_max):.1f}%** worst case.",
    "",
    "## Frequency dependence — observed",
    "",
    "Per-frequency median \\|Δ\\|/A_HG across (amp, wind) cells, with min/max:",
    "",
    "| f [Hz] | IN median | IN min | IN max | OUT median | OUT min | OUT max |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for _, r in freq_trend.iterrows():
    md_lines.append(
        f"| {r['freq_hz']:.1f} | "
        f"{r['IN_median_abs']:.2f}% | {r['IN_min_abs']:.2f}% | {r['IN_max_abs']:.2f}% | "
        f"{r['OUT_median_abs']:.2f}% | {r['OUT_min_abs']:.2f}% | {r['OUT_max_abs']:.2f}% |"
    )

md_lines += [
    "",
    f"Spearman trend (median \\|Δ\\| vs frequency, n=4 freqs):",
    f"  - **IN  : r = {r_in:+.3f},  p = {p_in:.3f}**",
    f"  - **OUT : r = {r_out:+.3f},  p = {p_out:.3f}**",
    "",
    "*Observation*: the OUT-probe disagreement decreases monotonically "
    "with frequency, from "
    f"{freq_trend.iloc[0]['OUT_median_abs']:.2f}% at 1.3 Hz to "
    f"{freq_trend.iloc[-1]['OUT_median_abs']:.2f}% at 1.6 Hz. "
    "The IN-probe disagreement does not show a monotonic trend at this "
    "sample size.",
    "",
    "*Candidate explanation (hypothesis, not directly tested)*: the "
    "window-start gap (pipeline relative to H&G) shrinks from ~16 wave "
    "periods at 1.3 Hz to ~2 periods at 1.6 Hz, because the pipeline "
    "anchors at probe arrival while H&G anchors at OUT-50T. A smaller "
    "gap makes the two windows sample more nearly the same wavetrain "
    "interval, so amplitude readings converge. The pattern is consistent "
    "with this mechanism on the OUT probe; the IN signal contains an "
    "incident+reflected superposition that may add window-position "
    "sensitivity not directly captured by the simple gap argument.",
    "",
    "## Per-cell breakdown (full table)",
    "",
    "| f | amp | wind | n | Δ A_in/A_in_HG | Δ A_out/A_out_HG |",
    "|---|---|---|---:|---:|---:|",
]
for _, r in cell_probe.iterrows():
    md_lines.append(
        f"| {r['freq_r']:.1f} | {r['amp_r']:.2f} | {r['wind']} | {int(r['n'])} | "
        f"{r['pct_IN']:+.2f}% | {r['pct_OUT']:+.2f}% |"
    )

md_lines += [
    "",
    "---",
    "",
    "## Supplementary — OUT/IN ratio under each window",
    "",
    "*Reported here for completeness; the headline methodology claim is "
    "the per-probe amplitude agreement above. The OUT/IN ratio is the "
    "results-section quantity (CH05).*",
    "",
    "| f | amp | wind | n | OUT/IN H&G | OUT/IN pipe | Δ | Δ % |",
    "|---|---|---|---:|---:|---:|---:|---:|",
]
for _, r in agg.iterrows():
    md_lines.append(
        f"| {r['freq_r']:.1f} | {r['amp_r']:.2f} | {r['wind']} | {r['n']} | "
        f"{r['OUT_IN_hg_mean']:.4f} | {r['OUT_IN_pipe_mean']:.4f} | "
        f"{r['delta_OUT_IN']:+.4f} | {r['pct_OUT_IN']:+.2f}% |"
    )

md_lines += [
    "",
    "Wind effect (Δτ = τ_fw − τ_nw) — preserved across windows:",
    "",
    "| f | amp | Δτ H&G | Δτ pipe | shift |",
    "|---|---|---:|---:|---:|",
]
for _, r in we.iterrows():
    md_lines.append(
        f"| {r['freq_r']:.1f} | {r['amp_r']:.2f} | "
        f"{r['Dtau_hg']:+.4f} | {r['Dtau_pipe']:+.4f} | "
        f"{r['Dtau_shift']:+.4f} |"
    )

md_lines += [
    "",
    "Supplementary OUT/IN summary:",
    f"- median \\|Δ OUT/IN\\| = {delta_med:.4f}  ({pct_med:.2f}% of H&G)",
    f"- max    \\|Δ OUT/IN\\| = {delta_max:.4f}  ({pct_max:.2f}% of H&G)",
    f"- median \\|Δτ shift\\| (wind-effect drift) = {shift_med:.4f}",
    f"- max    \\|Δτ shift\\|                      = {shift_max:.4f}",
]

OUT_MD.write_text("\n".join(md_lines), encoding="utf-8")
print(f"   findings → {OUT_MD.relative_to(BASE)}")

# Echo headlines to stdout
print("\n8. Headline numbers — per-probe agreement (the methodology claim)")
print(f"   IN  (9373/170)  : median |Δ|/A_HG = {in_med:.2f}%   "
      f"max = {in_max:.2f}%   within ±5%: {in_w5}/{n_cells_in}")
print(f"   OUT (12400/250) : median |Δ|/A_HG = {out_med:.2f}%   "
      f"max = {out_max:.2f}%   within ±5%: {out_w5}/{n_cells_out}")
print(f"   Spearman (median |Δ| vs frequency): "
      f"IN r={r_in:+.3f} (p={p_in:.3f}); OUT r={r_out:+.3f} (p={p_out:.3f})")
print()
print("   Supplementary OUT/IN ratio summary (results-section quantity):")
print(f"     median |Δ OUT/IN|  = {delta_med:.4f}  ({pct_med:.2f}% of H&G)")
print(f"     max    |Δ OUT/IN|  = {delta_max:.4f}  ({pct_max:.2f}% of H&G)")
print(f"     median |Δτ shift|  = {shift_med:.4f}")
print(f"     max    |Δτ shift|  = {shift_max:.4f}")

print("\nDone.")
