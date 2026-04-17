"""
Sliding-window FFT at paddle frequency — fullwind per240 sweep at 9373/170
===========================================================================

Purpose
-------
The 2026-04-17 per240 sliding-AFFT exploration on 1.3 Hz fullwind runs showed
that the analysis window (21–39 s) lands on a temporary AFFT depression: the
mean AFFT across the full run is ~7.5 mm, but the pipeline reads 4.5–7.4 mm
from the window. This script extends that check to 1.3 / 1.4 / 1.5 / 1.6 Hz
at 0.1 V and 0.2 V to see whether the depression is specific to one
condition or general to fullwind per240 at 9373/170.

Method
------
- Per run: extract the 9373/170 η time series (eta_9373/170 or its _interp
  reconstruction). Slide a fixed-length window (SLIDING_WINDOW_S seconds)
  across the signal with step SLIDING_STEP_S. For each window position,
  compute the FFT amplitude at the paddle frequency (nearest positive bin
  within ±0.1 Hz, matching compute_amplitudes_from_fft).
- Plot sliding AFFT curve for each fullwind run. Overlay:
    • pipeline analysis window (green band) from meta start/end columns
    • pipeline AFFT value (red dashed horizontal) from meta
    • nowind reference per240 run at the same (freq, amp), sliding AFFT
    • stable-region mean AFFT (orange dotted line, computed over the
      stable plateau excluding ramp-up/down)

Outputs
-------
- analysis_scratch/sliding_afft_fullwind_sweep.png
- analysis_scratch/sliding_afft_fullwind_sweep_findings.md
- analysis_scratch/sliding_afft_fullwind_sweep_summary.csv
  (per-run pipeline_AFFT / stable_mean / depression_ratio)

Run
---
    conda run -n draumkvedet python analysis_scratch/sliding_afft_fullwind_sweep.py

~30–40 s (most of it loading processed_dfs).
"""

import sys, warnings, glob
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wavescripts.improved_data_loader import load_analysis_data, load_processed_dfs

# ── Configuration ─────────────────────────────────────────────────────────────
FS                   = 250.0
PROBE                = "9373/170"           # IN probe, current config
FFT_WINDOW_HZ        = 0.10                 # ±0.05 Hz search half-width (matches pipeline)
SLIDING_WINDOW_S     = 15.0                 # sliding FFT window length (seconds)
SLIDING_STEP_S       = 1.0                  # step between window positions

TARGET_FREQS         = [1.3, 1.4, 1.5, 1.6]
TARGET_AMPS          = [0.1, 0.2]

# Stable-plateau definition: fraction of post-ramp-up signal to use for the
# "stable mean AFFT" statistic. Excludes first 25 % and last 10 % of the
# non-zero sliding-AFFT curve to avoid ramp transients.
PLATEAU_HEAD_FRAC    = 0.25
PLATEAU_TAIL_FRAC    = 0.10

BASE    = Path(__file__).parent.parent
OUT_PNG = Path(__file__).parent / "sliding_afft_fullwind_sweep.png"
OUT_MD  = Path(__file__).parent / "sliding_afft_fullwind_sweep_findings.md"
OUT_CSV = Path(__file__).parent / "sliding_afft_fullwind_sweep_summary.csv"


# ── Sliding-AFFT helper ───────────────────────────────────────────────────────
def sliding_afft(signal: np.ndarray, target_freq: float, fs: float = FS,
                 window_s: float = SLIDING_WINDOW_S,
                 step_s: float = SLIDING_STEP_S,
                 search_window_hz: float = FFT_WINDOW_HZ) -> tuple[np.ndarray, np.ndarray]:
    """
    Slide a window of length window_s across signal; at each position compute
    the amplitude at target_freq using the same normalisation and bin-picking
    rule as the pipeline (2·|FFT|/N at positive bins, nearest bin within
    ±search_window_hz of target).

    Returns (t_centers_s, amplitudes_mm). Assumes signal is in mm already.
    """
    N_win  = int(round(window_s * fs))
    step   = int(round(step_s   * fs))
    if N_win >= len(signal):
        return np.array([]), np.array([])

    # Pre-compute FFT frequencies for the fixed window length
    freqs = np.fft.fftfreq(N_win, d=1.0 / fs)
    pos_mask = freqs > 0
    pos_freqs = freqs[pos_mask]

    bin_mask = (pos_freqs >= target_freq - search_window_hz) & \
               (pos_freqs <= target_freq + search_window_hz)
    if bin_mask.any():
        pos_masked_freqs = pos_freqs[bin_mask]
        nearest_idx = np.argmin(np.abs(pos_masked_freqs - target_freq))
    else:
        nearest_idx = np.argmin(np.abs(pos_freqs - target_freq))
        bin_mask = np.zeros_like(pos_freqs, dtype=bool)
        bin_mask[nearest_idx] = True
        nearest_idx = 0

    starts = np.arange(0, len(signal) - N_win + 1, step)
    t_centers = (starts + N_win / 2) / fs
    amps = np.full_like(t_centers, np.nan, dtype=float)

    for i, s in enumerate(starts):
        seg = signal[s:s + N_win]
        if np.isnan(seg).any():
            nan_frac = np.isnan(seg).mean()
            if nan_frac > 0.10:
                continue
            idx = np.arange(len(seg))
            seg = np.interp(idx, idx[~np.isnan(seg)], seg[~np.isnan(seg)])
        fft_vals = np.fft.fft(seg)
        amps_pos = 2.0 * np.abs(fft_vals[pos_mask]) / N_win
        amps[i] = amps_pos[bin_mask][nearest_idx]

    return t_centers, amps


def get_signal_mm(df: pd.DataFrame, pos: str) -> np.ndarray | None:
    """Prefer reconstructed eta_{pos}_interp; fall back to eta_{pos}. In mm."""
    for col in (f"eta_{pos}_interp", f"eta_{pos}"):
        if col in df.columns:
            return df[col].to_numpy(dtype=float)
    return None


def stable_mean(t: np.ndarray, a: np.ndarray,
                alive_threshold_frac: float = 0.50,
                head_trim_s: float = 8.0,
                tail_trim_s: float = 4.0) -> tuple[float, float, float]:
    """
    Mean AFFT over the *alive* plateau — robust to post-wavemaker decay.

    Definition:
      1. `ref` = 90th percentile of sliding AFFT.
      2. "alive" samples = those where AFFT >= alive_threshold_frac * ref.
      3. Take the LONGEST contiguous alive run (handles noisy transients).
      4. Trim head_trim_s from its start (ramp-up still settling) and
         tail_trim_s from its end (ramp-down entering).
      5. Mean of the remainder.

    Returns (mean, t_alive_start, t_alive_end) in seconds. NaN if no
    alive region.
    """
    finite = np.isfinite(a)
    if finite.sum() < 4:
        return float("nan"), float("nan"), float("nan")
    ref = np.nanpercentile(a, 90)
    alive = finite & (a >= alive_threshold_frac * ref)
    if not alive.any():
        return float("nan"), float("nan"), float("nan")

    # Longest contiguous alive run
    idx = np.where(alive)[0]
    splits = np.where(np.diff(idx) > 1)[0]
    runs = np.split(idx, splits + 1) if splits.size else [idx]
    longest = max(runs, key=len)

    t_start, t_end = t[longest[0]], t[longest[-1]]
    # Trim head and tail
    t_start_trim = t_start + head_trim_s
    t_end_trim   = t_end   - tail_trim_s
    if t_end_trim <= t_start_trim:
        return float("nan"), t_start, t_end

    mask = (t >= t_start_trim) & (t <= t_end_trim) & np.isfinite(a)
    if not mask.any():
        return float("nan"), t_start_trim, t_end_trim
    return float(np.nanmean(a[mask])), t_start_trim, t_end_trim


# ── 1. Load metadata ──────────────────────────────────────────────────────────
print("1. Loading metadata…")
dirs = sorted(glob.glob(str(BASE / "waveprocessed" / "PROCESSED-*")))
meta_all, _, _, _ = load_analysis_data(*dirs, load_processed=False)

mask = (
    (meta_all["PanelCondition"] == "full")
    & (meta_all["in_position"] == PROBE)
    & (meta_all["WaveFrequencyInput [Hz]"].isin(TARGET_FREQS))
    & (meta_all["WaveAmplitudeInput [Volt]"].isin(TARGET_AMPS))
    & (meta_all.get("quality_flag", pd.Series("ok", index=meta_all.index)) == "ok")
)
# "per240" is a wave-period-count tag — easiest way is by filename substring.
mask &= meta_all["path"].str.contains("per240", na=False)
runs = meta_all[mask].copy()
print(f"   {len(runs)} per240 runs at in=9373/170 fullpanel quality=ok")
print(runs.groupby(
    ["WaveFrequencyInput [Hz]", "WaveAmplitudeInput [Volt]", "WindCondition"]
).size().unstack(fill_value=0))

# ── 2. Load processed time series ─────────────────────────────────────────────
print("2. Loading processed time series (load_processed_dfs, slow)…")
proc_dfs = load_processed_dfs(*dirs)
print(f"   {len(proc_dfs)} processed DataFrames loaded")


# ── 3. Build lookup: for each (freq, amp, wind) pick first available run ──────
def pick_run(freq, amp, wind):
    sub = runs[
        (runs["WaveFrequencyInput [Hz]"] == freq)
        & (runs["WaveAmplitudeInput [Volt]"] == amp)
        & (runs["WindCondition"] == wind)
    ]
    # Prefer mstop30 (regular) over mstop330 (long tail) so panels are uniform.
    for tag in ("mstop30", "mstop330"):
        pick = sub[sub["path"].str.contains(tag, na=False)]
        if not pick.empty and pick["path"].iloc[0] in proc_dfs:
            return pick.iloc[0]
    # Fallback: any path present in proc_dfs
    for _, r in sub.iterrows():
        if r["path"] in proc_dfs:
            return r
    return None


# ── 4. Compute sliding AFFT and plot ──────────────────────────────────────────
print("3. Computing sliding AFFT and plotting…")
N_freq, N_amp = len(TARGET_FREQS), len(TARGET_AMPS)
fig, axes = plt.subplots(N_freq, N_amp,
                         figsize=(5.4 * N_amp, 2.6 * N_freq),
                         sharex=False)

summary_rows = []

for i_f, freq in enumerate(TARGET_FREQS):
    for i_a, amp in enumerate(TARGET_AMPS):
        ax = axes[i_f, i_a] if N_freq > 1 else axes[i_a]

        fw_row = pick_run(freq, amp, "full")
        nw_row = pick_run(freq, amp, "no")

        if fw_row is None:
            ax.text(0.5, 0.5, "no fullwind per240 run",
                    ha="center", va="center", transform=ax.transAxes, color="gray")
            ax.set_title(f"{freq:.1f} Hz, {amp:.1f} V (full)", fontsize=9)
            continue

        # Fullwind sliding AFFT
        df = proc_dfs[fw_row["path"]]
        sig = get_signal_mm(df, PROBE)
        if sig is None:
            ax.text(0.5, 0.5, "no signal column", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        t_fw, a_fw = sliding_afft(sig, target_freq=freq)
        ax.plot(t_fw, a_fw, color="tab:red", lw=1.3, label="fullwind (sliding)")

        # Pipeline window + pipeline AFFT
        s_col = f"Computed Probe {PROBE} start"
        e_col = f"Computed Probe {PROBE} end"
        pip_start, pip_end = fw_row.get(s_col), fw_row.get(e_col)
        pip_afft = fw_row.get(f"Probe {PROBE} Amplitude (FFT)", np.nan)
        if pd.notna(pip_start) and pd.notna(pip_end):
            ax.axvspan(pip_start / FS, pip_end / FS, color="tab:green",
                       alpha=0.18, label="pipeline window")
        if pd.notna(pip_afft):
            ax.axhline(pip_afft, color="tab:red", ls="--", lw=1.0,
                       label=f"pipeline AFFT = {pip_afft:.2f} mm")

        # Stable-region mean
        stable, t_s0, t_s1 = stable_mean(t_fw, a_fw)
        if np.isfinite(stable):
            ax.axhline(stable, color="tab:orange", ls=":", lw=1.1,
                       label=f"stable mean = {stable:.2f} mm")
            ax.axvline(t_s0, color="tab:orange", ls=":", lw=0.6, alpha=0.5)
            ax.axvline(t_s1, color="tab:orange", ls=":", lw=0.6, alpha=0.5)

        # Nowind reference
        nw_stable = np.nan
        if nw_row is not None:
            dfn = proc_dfs[nw_row["path"]]
            sig_n = get_signal_mm(dfn, PROBE)
            if sig_n is not None:
                t_nw, a_nw = sliding_afft(sig_n, target_freq=freq)
                nw_stable, _, _ = stable_mean(t_nw, a_nw)
                ax.plot(t_nw, a_nw, color="tab:blue", lw=1.1, alpha=0.75,
                        label=f"nowind ref (stable {nw_stable:.2f} mm)")

        # Summary row
        depression = (pip_afft / stable) if np.isfinite(stable) and stable > 0 else np.nan
        summary_rows.append({
            "freq_hz":       freq,
            "amp_V":         amp,
            "fullwind_path": Path(fw_row["path"]).name,
            "pipeline_AFFT": pip_afft,
            "stable_mean":   stable,
            "depression":    depression,
            "nowind_path":   Path(nw_row["path"]).name if nw_row is not None else None,
            "nowind_stable": nw_stable if nw_row is not None else np.nan,
        })

        ax.set_title(f"{freq:.1f} Hz, {amp:.1f} V", fontsize=9)
        ax.set_xlabel("time (s)", fontsize=8)
        ax.set_ylabel("AFFT @ f (mm)", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6, loc="lower right")

fig.suptitle(f"Sliding AFFT at {PROBE} — fullwind per240 vs nowind reference\n"
             f"(window={SLIDING_WINDOW_S}s, step={SLIDING_STEP_S}s, "
             f"bin search=±{FFT_WINDOW_HZ/2*2:.2f}Hz)",
             fontsize=10, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig(OUT_PNG, dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"   PNG → {OUT_PNG.relative_to(BASE)}")

# ── 5. Write summary CSV + findings markdown ──────────────────────────────────
df_sum = pd.DataFrame(summary_rows)
df_sum.to_csv(OUT_CSV, index=False, float_format="%.3f")
print(f"   CSV → {OUT_CSV.relative_to(BASE)}")

# Build findings
lines = []
lines.append("# Sliding AFFT @ 9373/170 — fullwind per240 sweep")
lines.append("")
lines.append(f"Generated: {pd.Timestamp.utcnow().isoformat()[:19]}Z")
lines.append("")
lines.append("## Summary table")
lines.append("")
def _md_table(df: pd.DataFrame, fmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append(fmt.format(v) if np.isfinite(v) else "—")
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + rows)

lines.append(_md_table(df_sum.drop(columns=["fullwind_path", "nowind_path"])))
lines.append("")
lines.append("## Verdict")
lines.append("")

depressions = df_sum["depression"].dropna()
frac_below_90 = (depressions < 0.90).mean() if len(depressions) else 0.0
frac_below_80 = (depressions < 0.80).mean() if len(depressions) else 0.0
median_dep = depressions.median() if len(depressions) else float("nan")

lines.append(f"- Runs analysed: {len(df_sum)} (fullwind per240 at {PROBE}).")
lines.append(f"- Median pipeline_AFFT / stable_mean_AFFT = **{median_dep:.3f}**.")
lines.append(f"- Fraction with ≥10% depression (ratio < 0.90): **{frac_below_90:.0%}**.")
lines.append(f"- Fraction with ≥20% depression (ratio < 0.80): **{frac_below_80:.0%}**.")
lines.append("")

if median_dep < 0.92:
    lines.append("**Finding:** The pipeline analysis window systematically captures a "
                 "depression relative to the stable plateau. Reproduces the 1.3 Hz result "
                 "across additional frequencies / amplitudes.")
elif median_dep < 0.98:
    lines.append("**Finding:** Small systematic depression (<8%). May be within noise; "
                 "check per-case variability.")
else:
    lines.append("**Finding:** No systematic depression — the 1.3 Hz observation does not "
                 "generalise to other fullwind per240 conditions at this probe.")

lines.append("")
lines.append("## Per-run breakdown")
lines.append("")
for _, r in df_sum.iterrows():
    lines.append(f"- **{r['freq_hz']:.1f} Hz, {r['amp_V']:.1f} V**  "
                 f"pipeline={r['pipeline_AFFT']:.2f} mm, "
                 f"stable={r['stable_mean']:.2f} mm, "
                 f"ratio={r['depression']:.3f}  "
                 f"(fullwind: `{r['fullwind_path']}`)")

lines.append("")
lines.append(f"## See also")
lines.append("")
lines.append(f"- Figure: `analysis_scratch/sliding_afft_fullwind_sweep.png`")
lines.append(f"- CSV:    `analysis_scratch/sliding_afft_fullwind_sweep_summary.csv`")
lines.append(f"- Earlier 1.3 Hz result: `analysis_scratch/sliding_afft_per240.pdf`")
lines.append(f"- Memory: `memory/open_question_per240_afft.md`")

OUT_MD.write_text("\n".join(lines) + "\n")
print(f"   MD  → {OUT_MD.relative_to(BASE)}")

print(f"\nDone. Median depression = {median_dep:.3f}  ({len(depressions)} runs)")
