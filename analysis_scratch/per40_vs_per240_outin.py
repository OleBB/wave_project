"""
Can short (per40-style) and long (per240-style) runs be pooled for
OUT/IN (FFT) thesis results?
=====================================================================

Two windows on the paddle-frequency FFT:

  short runs (WavePeriodInput < HG_MIN_PERIODS, currently < 50T)
      → pipeline SNARVEI window, per-probe + per-freq eyeball. Read
        directly from `combined_meta["IN Amplitude (FFT)"]` /
        `combined_meta["OUT Amplitude (FFT)"]` (these are the canonical
        mean-of-parallel-probes columns written by processor2nd.py).

  long runs  (WavePeriodInput >= HG_MIN_PERIODS)
      → Huseby & Grue window [50T, 60T] from wavemaker start (t=0),
        scaled to the run's paddle frequency. Exactly 10T wide → no
        spectral leakage, no parasitic second-harmonic contamination
        (too fast to have reached the probe by 60T), no beach
        reflection (hasn't returned).
        IN side   = mean of AFFT at 9373/170 and 9373/340
                    (matches the canonical pipeline mean).
        OUT side  = AFFT at 12400/250 (single-probe canonical).

If the two methods give the same OUT/IN within run-to-run std at each
(freq, amp, wind), we pool them in CH05. If not, per240-H&G becomes
the thesis primary and per40 moves to supplementary.

Scope (thesis-matching):
    PanelCondition = full,
    Mooring        = below_90_loose (230+300 merged),
    quality_flag  in {ok, probe_malfunction_secondary},
    Frequency     in {1.3, 1.4, 1.5, 1.6} Hz,
    Amplitude     in {0.1, 0.2, 0.3} V,
    WindCondition in {no, full}.

Dynamic threshold (user-set 2026-04-21):
    HG_MIN_PERIODS = 50  — any run with >= 50 input periods gets H&G.

Run from repo root (loads processed_dfs for long runs only):
    /opt/anaconda3/envs/draumkvedet/bin/python analysis_scratch/per40_vs_per240_outin.py

Outputs:
    analysis_scratch/per40_vs_per240_outin.pdf
    analysis_scratch/per40_vs_per240_outin.csv
    analysis_scratch/per40_vs_per240_outin_findings.md
    output/FIGURES/ch04_per40_vs_per240_outin.pdf
    output/TEXFIGU/ch04_per40_vs_per240_outin.tex
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
from wavescripts.plot_utils import apply_thesis_style, freq_to_k, amp_to_label

apply_thesis_style()

# ── I/O ───────────────────────────────────────────────────────────────────────
SCRATCH_DIR = Path(__file__).parent
SCRATCH_PDF = SCRATCH_DIR / "per40_vs_per240_outin.pdf"
SCRATCH_CSV = SCRATCH_DIR / "per40_vs_per240_outin.csv"
SCRATCH_MD  = SCRATCH_DIR / "per40_vs_per240_outin_findings.md"

THESIS_NAME = "ch04_per40_vs_per240_outin"
THESIS_PDF  = BASE / "output" / "FIGURES" / f"{THESIS_NAME}.pdf"
THESIS_STUB = BASE / "output" / "TEXFIGU" / f"{THESIS_NAME}.tex"
THESIS_PDF.parent.mkdir(parents=True, exist_ok=True)
THESIS_STUB.parent.mkdir(parents=True, exist_ok=True)

# Same cond4 thesis datasets as t_cross_figure / paddle_contamination / H&G.
RESULTS_PROCESSED_DIRS = [
    Path("waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange"),
    Path("waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"),
]
_results_dataset_names = {p.name.removeprefix("PROCESSED-") for p in RESULTS_PROCESSED_DIRS}

# ── Constants ────────────────────────────────────────────────────────────────
FS             = 250.0                # sampling rate — MEASUREMENT.FS
FREQS          = [1.3, 1.4, 1.5, 1.6] # thesis scope
AMPS           = [0.1, 0.2, 0.3]
WINDS          = ["no", "full"]

HG_MIN_PERIODS  = 50                  # user-dynamic: runs with >= 50T get H&G
HG_START_N_T    = 50                  # start window at 50T from wavemaker start
HG_END_N_T      = 60                  # end   window at 60T
HG_WINDOW_N_T   = HG_END_N_T - HG_START_N_T   # 10T
FFT_BAND_HZ     = 0.05                # ± half-width, matches signal_processing

# Canonical IN = mean of these two parallel probes (per march2026_better_rearranging).
IN_PROBES  = ["9373/170", "9373/340"]
OUT_PROBE  = "12400/250"

# Plotting
WIND_COLOR   = {"no": "#2E86AB", "full": "#E74C3C"}
METHOD_MARKER = {"short": "o", "long": "D"}
METHOD_LABEL = {"short": "per40 / SNARVEI (short runs)",
                "long":  "per240 / H&G 10T (long runs)"}


# ── 1. FFT helper (mirrors pipeline convention) ──────────────────────────────
def fft_amp_at_freq(segment: np.ndarray, target_hz: float,
                    fs: float = FS, band_hz: float = FFT_BAND_HZ) -> float:
    """Single-shot FFT amplitude at `target_hz`: pick the nearest positive bin
    within ±band_hz. Normalisation: 2·|FFT|/N (pipeline convention).

    Lifted from analysis_scratch/huseby_grue_window.py for self-containment.
    """
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
    fft_vals  = np.fft.fft(seg)
    amps_pos  = 2.0 * np.abs(fft_vals[pos]) / N
    return float(amps_pos[j])


def get_eta(df_run: pd.DataFrame, pos: str) -> np.ndarray | None:
    """Prefer interpolated eta (pipeline convention)."""
    for col in (f"eta_{pos}_interp", f"eta_{pos}"):
        if col in df_run.columns:
            return df_run[col].to_numpy(dtype=float)
    return None


def hg_afft_for_run(df_run: pd.DataFrame, f_paddle: float, pos: str) -> float:
    """H&G window AFFT: signal[50T : 60T] at `f_paddle`, from sample 0 =
    wavemaker start. NaN if the window runs past the recorded signal.
    """
    sig = get_eta(df_run, pos)
    if sig is None:
        return np.nan
    period_samples = int(round(FS / f_paddle))
    i0 = HG_START_N_T * period_samples
    i1 = HG_END_N_T   * period_samples
    if i1 > len(sig):
        return np.nan
    return fft_amp_at_freq(sig[i0:i1], f_paddle)


# ── 2. Load meta ─────────────────────────────────────────────────────────────
print("1. Loading meta_results (no processed_dfs yet) …")
combined_meta, _, _, _ = load_analysis_data(*RESULTS_PROCESSED_DIRS, load_processed=False)
meta_results = combined_meta[
    combined_meta["path"].apply(lambda p: any(d in str(p) for d in _results_dataset_names))
].copy()
meta_results["Mooring"] = meta_results["Mooring"].replace({
    "below_90_loose230": "below_90_loose",
    "below_90_loose300": "below_90_loose",
})
print(f"   meta_results: {len(meta_results)} rows")


# ── 3. Scope filter ──────────────────────────────────────────────────────────
wave = meta_results[
    meta_results["WaveFrequencyInput [Hz]"].notna()
    & (meta_results["WaveFrequencyInput [Hz]"] > 0)
    & (meta_results["PanelCondition"] == "full")
    & (meta_results["Mooring"] == "below_90_loose")
    & (meta_results["quality_flag"].isin(["ok", "probe_malfunction_secondary"]))
].copy()
wave["freq_r"] = wave["WaveFrequencyInput [Hz]"].round(2)
wave["amp_r"]  = wave["WaveAmplitudeInput [Volt]"].round(2)
wave = wave[wave["freq_r"].isin(FREQS) & wave["amp_r"].isin(AMPS)
            & wave["WindCondition"].isin(WINDS)].copy()

# Split into short (pipeline SNARVEI) and long (H&G) runs.
wave["N_input_periods"] = wave["WavePeriodInput"].astype(float)
wave["is_long"] = wave["N_input_periods"] >= HG_MIN_PERIODS
print(f"   thesis-scope runs: {len(wave)} "
      f"(short n={int((~wave['is_long']).sum())}, long n={int(wave['is_long'].sum())})")


# ── 4. Pipeline AFFT for short runs (no processed_dfs needed) ────────────────
print("\n2. Reading pipeline AFFT (canonical IN/OUT) for short runs …")
short = wave[~wave["is_long"]].copy()
short["A_in_mm"]     = short["IN Amplitude (FFT)"].astype(float)
short["A_out_mm"]    = short["OUT Amplitude (FFT)"].astype(float)
short["OUT_IN"]      = short["A_out_mm"] / short["A_in_mm"]
short["method"]      = "short"
print(f"   short runs with valid OUT/IN: {int(short['OUT_IN'].notna().sum())}")


# ── 5. H&G AFFT for long runs (needs processed_dfs) ──────────────────────────
print("\n3. Loading processed_dfs for long runs (~20 s) …")
processed_dfs = load_processed_dfs(*RESULTS_PROCESSED_DIRS)
print(f"   {len(processed_dfs)} time-series cached")

print("\n4. Computing H&G AFFT per long run …")
long_rows = []
for _, r in wave[wave["is_long"]].iterrows():
    path = r["path"]
    df_run = processed_dfs.get(path)
    if df_run is None:
        continue
    f_paddle = float(r["WaveFrequencyInput [Hz]"])
    # Canonical IN: mean AFFT across both parallel probes at longitudinal 9373.
    in_afts = [hg_afft_for_run(df_run, f_paddle, p) for p in IN_PROBES]
    in_afts = [a for a in in_afts if np.isfinite(a)]
    a_in = float(np.mean(in_afts)) if len(in_afts) >= 1 else np.nan
    a_out = hg_afft_for_run(df_run, f_paddle, OUT_PROBE)
    long_rows.append({
        "path":          path,
        "freq_r":        round(float(r["WaveFrequencyInput [Hz]"]), 2),
        "amp_r":         round(float(r["WaveAmplitudeInput [Volt]"]), 2),
        "WindCondition": r["WindCondition"],
        "WaveFrequencyInput [Hz]":   f_paddle,
        "WaveAmplitudeInput [Volt]": r["WaveAmplitudeInput [Volt]"],
        "Mooring":       r["Mooring"],
        "A_in_mm":       a_in,
        "A_out_mm":      a_out,
        "OUT_IN":        a_out / a_in if (a_in and np.isfinite(a_in) and a_in > 0) else np.nan,
        "method":        "long",
        "is_long":       True,
        "N_input_periods": r["N_input_periods"],
    })
long = pd.DataFrame(long_rows)
print(f"   long runs with valid H&G OUT/IN: {int(long['OUT_IN'].notna().sum())}")


# ── 6. Unified DataFrame + CSV ───────────────────────────────────────────────
cols = ["path", "freq_r", "amp_r", "WindCondition", "Mooring",
        "A_in_mm", "A_out_mm", "OUT_IN", "method",
        "WaveFrequencyInput [Hz]", "WaveAmplitudeInput [Volt]"]
both = pd.concat([short[cols], long[cols]], axis=0, ignore_index=True)
both.to_csv(SCRATCH_CSV, index=False)
print(f"\n5. Per-run table → {SCRATCH_CSV.relative_to(BASE)}")


# ── 7. Aggregate per (freq, amp, wind, method) ───────────────────────────────
agg = (
    both.dropna(subset=["OUT_IN"])
        .groupby(["freq_r", "amp_r", "WindCondition", "method"])
        ["OUT_IN"].agg(["mean", "std", "count"])
        .reset_index()
)
agg["std"] = agg["std"].fillna(0.0)
agg["k"]  = freq_to_k(agg["freq_r"].to_numpy())
print(f"\n6. Aggregated {len(agg)} (freq, amp, wind, method) cells.")


# ── 8. Delta per (freq, amp, wind) between methods ───────────────────────────
pvt = agg.pivot_table(index=["freq_r", "amp_r", "WindCondition"],
                      columns="method", values="mean").reset_index()
pvt["delta"] = pvt["long"] - pvt["short"]
pvt["delta_rel"] = pvt["delta"] / pvt["short"]
matched = pvt.dropna(subset=["short", "long"]).copy()
if len(matched):
    abs_med = float(matched["delta"].abs().median())
    abs_max = float(matched["delta"].abs().max())
    rel_med = float(matched["delta_rel"].abs().median())
    rel_max = float(matched["delta_rel"].abs().max())
else:
    abs_med = abs_max = rel_med = rel_max = float("nan")

print("\n7. OUT/IN delta (long − short) per (freq, amp, wind):")
if len(matched):
    print(matched.round(4).to_string(index=False))
else:
    print("   (no matched cells)")
print(f"\n   median |Δ OUT/IN|      = {abs_med:.4f}")
print(f"   max    |Δ OUT/IN|      = {abs_max:.4f}")
print(f"   median |Δ OUT/IN|/short = {rel_med*100:.2f} %")
print(f"   max    |Δ OUT/IN|/short = {rel_max*100:.2f} %")


# ── 9. Figure: 1×3 (per amplitude) ───────────────────────────────────────────
print("\n8. Plotting …")
fig, axes = plt.subplots(1, 3, figsize=(14, 5), dpi=120, sharey=True)
for i, amp in enumerate(AMPS):
    ax = axes[i]
    sub = agg[np.isclose(agg["amp_r"], amp)]
    if sub.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        ax.set_title("", fontsize=10)
        continue
    for wind in WINDS:
        for method in ["short", "long"]:
            cell = sub[(sub["WindCondition"] == wind) & (sub["method"] == method)]
            if cell.empty:
                continue
            cell = cell.sort_values("k")
            # Horizontal offset between methods for readability.
            dx = {"short": -0.003, "long": +0.003}[method]
            ax.errorbar(
                cell["k"].values + dx, cell["mean"].values,
                yerr=cell["std"].fillna(0).values,
                fmt=METHOD_MARKER[method],
                color=WIND_COLOR[wind],
                mfc=WIND_COLOR[wind] if method == "short" else "white",
                mec=WIND_COLOR[wind],
                markeredgewidth=1.1,
                linestyle="-" if method == "short" else "--",
                capsize=3, lw=1.3, markersize=7, alpha=0.92,
                label=f"{wind} wind · {METHOD_LABEL[method]}",
            )
    ax.axhline(1.0, color="black", lw=0.6, ls="--", alpha=0.4)
    ax.set_xlabel("$k$ (rad/m)", fontsize=10)
    if i == 0:
        ax.set_ylabel("OUT/IN (FFT)", fontsize=10)
    ax.set_title("", fontsize=10)
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.legend(fontsize=7, loc="best", framealpha=0.92, ncol=1)

fig.suptitle("", fontsize=11, fontweight="bold", y=1.00)
fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.12, wspace=0.06)
fig.savefig(SCRATCH_PDF, bbox_inches="tight")
print(f"   scratch preview → {SCRATCH_PDF.relative_to(BASE)}")
fig.savefig(THESIS_PDF, bbox_inches="tight")
print(f"   thesis figure   → {THESIS_PDF.relative_to(BASE)}")
plt.close(fig)


# ── 10. TEXFIGU stub via shared helper ───────────────────────────────────────
print("\n9. Writing .tex stub via pu.build_fig_meta + pu.write_figure_stub …")

import wavescripts.plot_utils as pu
pu.ACTIVE_DATASETS = [str(p).split("/")[-1] for p in RESULTS_PROCESSED_DIRS]
pu.TEXFIGU_DIR = BASE / "output" / "TEXFIGU"
pu.FIGURES_DIR = BASE / "output" / "FIGURES"

_caption = (
    f"OUT/IN (FFT) at the paddle frequency versus $k$ (rad/m), split by paddle drive "
    f"{', '.join(f'{a:.2f}' for a in AMPS)}\\,V (one panel each). "
    "Two analysis-window methods per condition: pipeline SNARVEI window applied to "
    f"short runs ($N_\\mathrm{{input\\_periods}} < {HG_MIN_PERIODS}$; filled circles), "
    f"Huseby--Grue 10T window $t \\in [{HG_START_N_T}T, {HG_END_N_T}T]$ from "
    "wavemaker start applied to long runs (open diamonds). "
    "Blue: no wind; red: full wind. "
    f"Error bars: run-to-run standard deviation within each (frequency, amplitude, "
    "wind, method) cell. "
    f"Canonical IN amplitude is the mean of parallel probes 9373/170 and 9373/340; "
    f"OUT amplitude is from 12400/250 (both via the same FFT peak-bin convention). "
    f"Dataset: cond4 low-range, mooring \\texttt{{below\\_90\\_loose}} (230+300 merged), "
    f"quality flag $\\in$ \\{{ok, probe\\_malfunction\\_secondary\\}}."
)

_meta_stub = pu.build_fig_meta(
    {
        "filters": {
            "PanelCondition":            "full",
            "WaveFrequencyInput [Hz]":   [min(FREQS), max(FREQS)],
            "WaveAmplitudeInput [Volt]": AMPS,
            "WindCondition":             WINDS,
            "quality_flag":              "ok+probe_malfunction_secondary",
            "Mooring":                   ["below_90_loose"],
            "probes":                    "IN = mean(9373/170, 9373/340); OUT = 12400/250",
        },
        "plotting": {
            "figure_name":   THESIS_NAME,
            "caption":       _caption,
            "caption_short": "OUT/IN: pipeline SNARVEI (short runs) vs Huseby-Grue 10T (long runs)",
        },
    },
    chapter="04",
    data_df=wave,
    extra={"script": "analysis_scratch/per40_vs_per240_outin.py"},
    computed_in=(
        "analysis_scratch/per40_vs_per240_outin.py "
        "(short-run OUT/IN from combined_meta 'IN/OUT Amplitude (FFT)' canonical columns; "
        "long-run OUT/IN from H&G [50T, 60T] window via np.fft.fft on interpolated eta)"
    ),
    data_class="DFS",
    findings_doc="analysis_scratch/per40_vs_per240_outin_findings.md",
    grouper="per (freq, amp, wind, method) group; n-average with std",
    collapse_panels=False,
    fft_window_hz=2 * FFT_BAND_HZ,
    extra_params=(
        f"HG_MIN_PERIODS={HG_MIN_PERIODS}, HG_window=[{HG_START_N_T}T, {HG_END_N_T}T], "
        f"fft_band_hz={FFT_BAND_HZ}, "
        f"IN_probes={IN_PROBES}, OUT_probe={OUT_PROBE}, "
        f"datasets={sorted(_results_dataset_names)}"
    ),
    extra_stats={
        "n_short_runs":             int((~wave["is_long"]).sum()),
        "n_long_runs":              int(wave["is_long"].sum()),
        "n_matched_cells":          int(len(matched)),
        "delta_OUTIN_median":       round(abs_med, 4) if np.isfinite(abs_med) else "—",
        "delta_OUTIN_max":          round(abs_max, 4) if np.isfinite(abs_max) else "—",
        "delta_OUTIN_rel_median_pct": round(rel_med * 100, 2) if np.isfinite(rel_med) else "—",
        "delta_OUTIN_rel_max_pct":  round(rel_max * 100, 2) if np.isfinite(rel_max) else "—",
        "HG_min_periods":           HG_MIN_PERIODS,
        "HG_window_periods":        f"[{HG_START_N_T}T, {HG_END_N_T}T]",
    },
)

pu.write_figure_stub(_meta_stub, plot_type="per40_vs_per240_outin",
                     subfig_filenames=[THESIS_NAME])
print(f"   thesis stub   → {THESIS_STUB.relative_to(BASE)}")


# ── 11. Findings markdown (descriptive run log only) ─────────────────────────
print("\n10. Writing findings markdown …")
lines = []
lines.append("# Per40 vs per240 OUT/IN comparison — run log")
lines.append("")
lines.append(f"Generated by `analysis_scratch/per40_vs_per240_outin.py`.  ")
lines.append(f"Scope: full panel, mooring below_90_loose, quality OK, freq "
             f"{min(FREQS):.1f}–{max(FREQS):.1f} Hz, amp {AMPS} V, wind {WINDS}.")
lines.append("")
lines.append(f"HG threshold: `N_input_periods >= {HG_MIN_PERIODS}`.  ")
lines.append(f"HG window: `[{HG_START_N_T}T, {HG_END_N_T}T]` "
             f"from wavemaker start = exactly 10T (no leakage).")
lines.append("")
lines.append("## Cohort sizes")
lines.append("")
lines.append(f"- short runs (SNARVEI): **{int((~wave['is_long']).sum())}**")
lines.append(f"- long runs (H&G):     **{int(wave['is_long'].sum())}**")
lines.append(f"- matched (freq, amp, wind) cells: **{int(len(matched))}**")
lines.append("")
lines.append("## OUT/IN agreement — long vs short")
lines.append("")
lines.append(f"- median |Δ OUT/IN| (absolute) = **{abs_med:.4f}**")
lines.append(f"- max    |Δ OUT/IN| (absolute) = **{abs_max:.4f}**")
lines.append(f"- median |Δ OUT/IN| / short    = **{rel_med*100:.2f} %**")
lines.append(f"- max    |Δ OUT/IN| / short    = **{rel_max*100:.2f} %**")
lines.append("")
lines.append("## Per-cell detail (long − short)")
lines.append("")
if len(matched):
    lines.append("| freq [Hz] | amp [V] | wind | short | long | Δ | Δ / short |")
    lines.append("|-----------|---------|------|-------|------|---|-----------|")
    for _, r in matched.sort_values(["amp_r", "freq_r", "WindCondition"]).iterrows():
        lines.append(f"| {r['freq_r']:.1f} | {r['amp_r']:.2f} | {r['WindCondition']:>4} | "
                     f"{r['short']:.4f} | {r['long']:.4f} | {r['delta']:+.4f} | "
                     f"{r['delta_rel']*100:+.2f} % |")
else:
    lines.append("*(no matched cells)*")
lines.append("")
lines.append("## Files")
lines.append("")
lines.append(f"  {SCRATCH_PDF.name}")
lines.append(f"  {SCRATCH_CSV.name}")
lines.append(f"  output/FIGURES/{THESIS_NAME}.pdf")
lines.append(f"  output/TEXFIGU/{THESIS_NAME}.tex")
lines.append("")
lines.append("Interpretation/conclusions intentionally omitted here — see "
             "`memory/methodology_wind_enhances_A_in.md` and CLAUDE.md §16 for "
             "the analytical framework, and the user-eye review of the PDF for "
             "the thesis verdict on whether per40 and per240 can be pooled.")

SCRATCH_MD.write_text("\n".join(lines))
print(f"   findings → {SCRATCH_MD.relative_to(BASE)}")

print("\nDone.")
