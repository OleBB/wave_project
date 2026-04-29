"""
Impact of FFT peak-bin bias on OUT/IN ratio — real-data test
=============================================================

Hypothesis: if the IN and OUT probes in a given run use the same analysis
window length and see the same paddle frequency, their nearest-bin FFT
biases cancel and OUT/IN is invariant to the bias.

This script tests that directly on all fullpanel wave runs in current
probe config (in=9373/170, out=12400/250). For each run:
  - Pull the η time series for IN and OUT at the same analysis window
  - Compute nearest-bin AFFT (pipeline convention)
  - Compute parabolic-interpolated AFFT (sub-bin peak correction)
  - Compute OUT/IN for both
  - Report the distribution of Δ(OUT/IN) = ratio_parabolic − ratio_nearest

If Δ is small and unbiased across runs, the thesis OUT/IN values are
safe as-is. If Δ is systematic (biased toward + or −), the thesis needs
corrected amplitudes.

Output:
    analysis_scratch/fft_peak_bias_outin_impact.png
    analysis_scratch/fft_peak_bias_outin_impact.csv
    analysis_scratch/fft_peak_bias_outin_impact_findings.md
"""

import sys, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wavescripts.improved_data_loader import load_analysis_data, load_processed_dfs
from wavescripts.plot_utils import apply_thesis_style

apply_thesis_style()

FS            = 250.0
FFT_WINDOW_HZ = 0.10

IN_PROBE  = "9373/170"
OUT_PROBE = "12400/250"

BASE    = Path(__file__).parent.parent
OUT_PNG = Path(__file__).parent / "fft_peak_bias_outin_impact.png"
OUT_CSV = Path(__file__).parent / "fft_peak_bias_outin_impact.csv"
OUT_MD  = Path(__file__).parent / "fft_peak_bias_outin_impact_findings.md"


def nearest_and_parabolic_amp(signal: np.ndarray, target_freq: float,
                              fs: float = FS,
                              search_window_hz: float = FFT_WINDOW_HZ) -> tuple[float, float, float, float]:
    """
    Returns (A_nearest, A_parabolic, f_nearest, f_parabolic_interp).

    A_nearest:   nearest-bin amplitude (pipeline convention)
    A_parabolic: parabolic-interpolated peak amplitude (3-point fit)
                 around the nearest bin — corrects for bin-grid mismatch.
    f_nearest:   frequency of the nearest bin used
    f_parabolic: estimated true peak frequency (bin + interpolated offset)
    """
    sig = np.asarray(signal, dtype=float)
    nan_mask = np.isnan(sig)
    if nan_mask.all() or len(sig) < int(2 * fs / max(target_freq, 0.1)):
        return float("nan"), float("nan"), float("nan"), float("nan")
    if nan_mask.any():
        if nan_mask.mean() > 0.10:
            return float("nan"), float("nan"), float("nan"), float("nan")
        idx = np.arange(len(sig))
        sig = np.interp(idx, idx[~nan_mask], sig[~nan_mask])

    N = len(sig)
    fft_vals = np.fft.fft(sig)
    freqs = np.fft.fftfreq(N, d=1.0 / fs)
    pos_mask = freqs > 0
    pos_freqs = freqs[pos_mask]
    amps = 2.0 * np.abs(fft_vals[pos_mask]) / N

    # Nearest bin within ±search_window_hz
    bin_mask = (pos_freqs >= target_freq - search_window_hz) & \
               (pos_freqs <= target_freq + search_window_hz)
    if bin_mask.any():
        masked_freqs = pos_freqs[bin_mask]
        masked_amps  = amps[bin_mask]
        k = int(np.argmin(np.abs(masked_freqs - target_freq)))
        f_nearest = masked_freqs[k]
        a_nearest = masked_amps[k]
        # Find global index in full positive-freq array for parabolic fit
        global_k = int(np.where(pos_freqs == f_nearest)[0][0])
    else:
        global_k = int(np.argmin(np.abs(pos_freqs - target_freq)))
        f_nearest = pos_freqs[global_k]
        a_nearest = amps[global_k]

    # Parabolic interpolation using the nearest bin + its two neighbours.
    # Take the local peak (in case nearest-to-target isn't the local max).
    if 1 <= global_k <= len(amps) - 2:
        # Walk to local peak within ±2 bins (guards against shoulder bins)
        local_k = global_k
        for _ in range(2):
            if 1 <= local_k <= len(amps) - 2:
                neighbourhood = amps[local_k-1:local_k+2]
                shift = int(np.argmax(neighbourhood)) - 1
                if shift == 0:
                    break
                local_k += shift
        if 1 <= local_k <= len(amps) - 2:
            y_m1, y_0, y_p1 = amps[local_k-1], amps[local_k], amps[local_k+1]
            denom = (y_m1 - 2 * y_0 + y_p1)
            if denom != 0:
                delta = 0.5 * (y_m1 - y_p1) / denom  # in bins, -0.5..+0.5
                a_parabolic = y_0 - 0.25 * (y_m1 - y_p1) * delta
                f_parabolic = pos_freqs[local_k] + delta * (pos_freqs[1] - pos_freqs[0])
            else:
                a_parabolic = y_0
                f_parabolic = pos_freqs[local_k]
        else:
            a_parabolic = a_nearest
            f_parabolic = f_nearest
    else:
        a_parabolic = a_nearest
        f_parabolic = f_nearest

    return float(a_nearest), float(a_parabolic), float(f_nearest), float(f_parabolic)


# ── Load ──────────────────────────────────────────────────────────────────────
print("1. Loading metadata…")
dirs = sorted(glob.glob(str(BASE / "waveprocessed" / "PROCESSED-*")))
meta_all, _, _, _ = load_analysis_data(*dirs, load_processed=False)

mask = (
    (meta_all["PanelCondition"] == "full")
    & (meta_all["in_position"] == IN_PROBE)
    & (meta_all["out_position"] == OUT_PROBE)
    & meta_all["WaveFrequencyInput [Hz]"].notna()
    & (meta_all["WaveFrequencyInput [Hz]"] > 0)
    & (meta_all["quality_flag"] == "ok")
    & meta_all[f"Computed Probe {IN_PROBE} start"].notna()
    & meta_all[f"Computed Probe {OUT_PROBE} start"].notna()
)
runs = meta_all[mask].copy()
print(f"   {len(runs)} fullpanel wave runs at in={IN_PROBE}/out={OUT_PROBE}")

print("2. Loading processed time series…")
proc_dfs = load_processed_dfs(*dirs)

# ── Compute per-run amplitudes using both methods ─────────────────────────────
print("3. Computing amplitudes with nearest-bin and parabolic methods…")
results = []
for i, (_, row) in enumerate(runs.iterrows()):
    path = row["path"]
    freq = row["WaveFrequencyInput [Hz]"]
    df = proc_dfs.get(path)
    if df is None:
        continue

    def get_sig(pos):
        col = f"eta_{pos}_interp" if f"eta_{pos}_interp" in df.columns else f"eta_{pos}"
        if col not in df.columns:
            return None
        s = int(row[f"Computed Probe {pos} start"])
        e = int(row[f"Computed Probe {pos} end"])
        return df[col].iloc[s:e+1].to_numpy(dtype=float), s, e

    in_res = get_sig(IN_PROBE)
    out_res = get_sig(OUT_PROBE)
    if in_res is None or out_res is None:
        continue
    in_sig, in_s, in_e = in_res
    out_sig, out_s, out_e = out_res

    A_in_n, A_in_p, f_in_n, f_in_p = nearest_and_parabolic_amp(in_sig, freq)
    A_out_n, A_out_p, f_out_n, f_out_p = nearest_and_parabolic_amp(out_sig, freq)

    if not (np.isfinite(A_in_n) and np.isfinite(A_out_n)):
        continue
    if A_in_n <= 0 or A_in_p <= 0:
        continue

    results.append({
        "path":            path,
        "name":            Path(path).name,
        "freq_hz":         freq,
        "amp_V":           row["WaveAmplitudeInput [Volt]"],
        "wind":            row["WindCondition"],
        "mooring":         row.get("Mooring", "?"),
        "N_in":            in_e - in_s + 1,
        "N_out":           out_e - out_s + 1,
        "A_in_nearest":    A_in_n,
        "A_in_parabolic":  A_in_p,
        "A_out_nearest":   A_out_n,
        "A_out_parabolic": A_out_p,
        "f_in_peak":       f_in_p,
        "f_out_peak":      f_out_p,
        "outin_nearest":   A_out_n / A_in_n,
        "outin_parabolic": A_out_p / A_in_p,
        "delta_outin":     (A_out_p / A_in_p) - (A_out_n / A_in_n),
    })

df = pd.DataFrame(results)
df.to_csv(OUT_CSV, index=False, float_format="%.4f")
print(f"   {len(df)} runs processed; CSV → {OUT_CSV.relative_to(BASE)}")

# ── Plot ──────────────────────────────────────────────────────────────────────
print("4. Plotting…")
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

# (a) scatter: outin_nearest vs outin_parabolic
ax = axes[0, 0]
for wind, color in (("no", "tab:blue"), ("lowest", "tab:orange"), ("full", "tab:red")):
    sub = df[df["wind"] == wind]
    if sub.empty:
        continue
    ax.scatter(sub["outin_nearest"], sub["outin_parabolic"],
               s=10, alpha=0.6, color=color, label=f"{wind} (n={len(sub)})")
lim = [0, max(df["outin_nearest"].max(), df["outin_parabolic"].max()) * 1.05]
ax.plot(lim, lim, "k--", lw=0.8, alpha=0.5, label="y=x")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("OUT/IN (nearest-bin) — pipeline convention")
ax.set_ylabel("OUT/IN (parabolic-interpolated)")
ax.set_title("")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# (b) histogram of delta
ax = axes[0, 1]
for wind, color in (("no", "tab:blue"), ("lowest", "tab:orange"), ("full", "tab:red")):
    sub = df[df["wind"] == wind]["delta_outin"].dropna()
    if sub.empty:
        continue
    ax.hist(sub, bins=40, alpha=0.5, color=color, label=f"{wind} (μ={sub.mean():+.4f})")
ax.axvline(0, color="k", ls="--", lw=0.8)
ax.set_xlabel("Δ(OUT/IN) = parabolic − nearest")
ax.set_ylabel("count")
ax.set_title("")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# (c) delta vs frequency — does it depend on frequency?
ax = axes[1, 0]
for wind, color in (("no", "tab:blue"), ("lowest", "tab:orange"), ("full", "tab:red")):
    sub = df[df["wind"] == wind]
    if sub.empty:
        continue
    ax.scatter(sub["freq_hz"], sub["delta_outin"],
               s=10, alpha=0.6, color=color, label=wind)
ax.axhline(0, color="k", ls="--", lw=0.8)
ax.set_xlabel("Wave frequency (Hz)")
ax.set_ylabel("Δ(OUT/IN)")
ax.set_title("")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# (d) paddle frequency offset vs nominal — does this cause the bias?
ax = axes[1, 1]
df["freq_offset_in"]  = df["f_in_peak"]  - df["freq_hz"]
df["freq_offset_out"] = df["f_out_peak"] - df["freq_hz"]
# Show IN offset (where fullwind contamination lives)
for wind, color in (("no", "tab:blue"), ("full", "tab:red")):
    sub = df[df["wind"] == wind]
    if sub.empty:
        continue
    ax.scatter(sub["freq_offset_in"] * 1000, sub["delta_outin"],
               s=10, alpha=0.6, color=color, label=wind)
ax.axhline(0, color="k", ls="--", lw=0.5)
ax.axvline(0, color="k", ls="--", lw=0.5)
ax.set_xlabel("Paddle peak offset at IN probe (mHz from nominal)")
ax.set_ylabel("Δ(OUT/IN)")
ax.set_title("")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

fig.suptitle("", fontsize=11)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"   PNG → {OUT_PNG.relative_to(BASE)}")

# ── Findings ──────────────────────────────────────────────────────────────────
lines = []
lines.append("# FFT peak-bin bias: impact on OUT/IN ratio")
lines.append("")
lines.append(f"Generated: {pd.Timestamp.utcnow().isoformat()[:19]}Z")
lines.append("")
lines.append(f"**Runs analyzed**: {len(df)} fullpanel wave runs "
             f"(in={IN_PROBE}, out={OUT_PROBE}, quality_flag=ok)")
lines.append("")
lines.append("## Per-wind-condition summary")
lines.append("")
lines.append("| wind | n | mean Δ(OUT/IN) | std Δ | max |Δ| | mean |Δ|/OUT/IN |")
lines.append("|------|---|----------------|-------|----------|------------------|")
for wind in ("no", "lowest", "full"):
    sub = df[df["wind"] == wind]
    if sub.empty:
        continue
    d = sub["delta_outin"].dropna()
    rel = (d.abs() / sub["outin_nearest"].abs()).dropna()
    lines.append(f"| {wind} | {len(sub)} | {d.mean():+.5f} | {d.std():.5f} | "
                 f"{d.abs().max():.4f} | {rel.mean():.3%} |")
lines.append("")
lines.append("## Takeaway")
lines.append("")

mean_delta_all = df["delta_outin"].mean()
std_delta_all = df["delta_outin"].std()
max_abs = df["delta_outin"].abs().max()
mean_rel = (df["delta_outin"].abs() / df["outin_nearest"].abs()).mean()

if df["delta_outin"].abs().quantile(0.90) < 0.02:
    verdict = (f"**OUT/IN is robust to the FFT peak-bin bias.** 90% of runs "
               f"have |Δ(OUT/IN)| < 0.02, mean |Δ|/OUT/IN = {mean_rel:.2%}. "
               f"The bias cancels in the ratio because IN and OUT probes use "
               f"the same analysis window length (same bin grid) and see the "
               f"same paddle frequency. Existing thesis values are safe to use.")
elif df["delta_outin"].abs().quantile(0.90) < 0.05:
    verdict = (f"**OUT/IN is moderately affected.** 90% of runs have "
               f"|Δ(OUT/IN)| < 0.05, mean |Δ|/OUT/IN = {mean_rel:.2%}. "
               f"The bias mostly cancels but a few outliers may need "
               f"reviewing. Consider adding parabolic interpolation as an "
               f"optional correction for publication-grade results.")
else:
    verdict = (f"**OUT/IN is significantly affected.** Mean |Δ|/OUT/IN = {mean_rel:.2%} "
               f"across {len(df)} runs. Parabolic interpolation should be "
               f"added to `compute_amplitudes_from_fft` before quoting "
               f"absolute OUT/IN values in the thesis.")

lines.append(verdict)
lines.append("")
lines.append(f"- **Global stats**: Δ(OUT/IN) mean = {mean_delta_all:+.5f}, "
             f"std = {std_delta_all:.5f}, max |Δ| = {max_abs:.4f}")
lines.append(f"- **Frequency dependence**: see figure panel (c). If Δ(OUT/IN) has "
             f"structure in frequency, specific frequencies may be more affected.")
lines.append(f"- **Paddle-drift link**: see figure panel (d). Δ(OUT/IN) should be "
             f"near-zero when the IN paddle-peak is close to a bin, and larger "
             f"when it is off.")
lines.append("")
lines.append("## See also")
lines.append("")
lines.append(f"- Figure: `analysis_scratch/fft_peak_bias_outin_impact.png`")
lines.append(f"- CSV:    `analysis_scratch/fft_peak_bias_outin_impact.csv`")
lines.append(f"- Methodology memo: `memory/methodology_fft_peak_bin_bias.md`")

OUT_MD.write_text("\n".join(lines) + "\n")
print(f"   MD  → {OUT_MD.relative_to(BASE)}")
print(f"\nDone. {len(df)} runs, mean Δ(OUT/IN) = {mean_delta_all:+.5f}, "
      f"mean |Δ|/OUT/IN = {mean_rel:.2%}")
