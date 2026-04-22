"""
FFT amplitude extraction — method comparison
=============================================

Goal: decide what `processor.py` should use to compute
`Probe {pos} Amplitude (FFT)` — the paddle-frequency amplitude per probe.

Four methods tested:
  1. nearest_bin  — current pipeline: nearest bin of np.fft.fft within
                    ±0.1 Hz of f_paddle. Sinc-attenuated when f_paddle
                    doesn't sit exactly on a bin center.
  2. parabolic    — sub-bin quadratic interpolation over 3 bins around
                    the nearest bin. Recovers most of the sinc loss.
  3. goertzel     — single-frequency DFT evaluated at *exactly* f_paddle.
                    Bypasses the bin grid entirely.
  4. ls_fit       — least-squares fit of [1, cos(ωt), sin(ωt),
                    cos(2ωt), sin(2ωt)] to the windowed signal.
                    Same as goertzel on the f_paddle coefficient but with
                    explicit DC removal and Stokes-2f protection.

Two tests:
  §1 synthetic   — pure tone with known amplitude at controlled frequency
                   offsets; three noise conditions. Shows each method's
                   theoretical response.
  §2 real data   — nowind wave runs in the two canonical March-2026
                   folders (cross-method agreement on real signals).

Output:
  analysis_scratch/fft_method_comparison.png
  analysis_scratch/fft_method_comparison_synth.csv
  analysis_scratch/fft_method_comparison_real.csv
  analysis_scratch/fft_method_comparison_findings.md
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wavescripts.improved_data_loader import load_analysis_data, load_processed_dfs

FS = 250.0
BASE = Path(__file__).parent.parent
SCRATCH = Path(__file__).parent

OUT_PNG      = SCRATCH / "fft_method_comparison.png"
OUT_CSV_SYNTH = SCRATCH / "fft_method_comparison_synth.csv"
OUT_CSV_REAL  = SCRATCH / "fft_method_comparison_real.csv"
OUT_MD        = SCRATCH / "fft_method_comparison_findings.md"

# Two target folders — the canonical March-2026 lowrange configuration.
TARGET_DIRS = [
    BASE / "waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange",
    BASE / "waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange",
]

# Probes present in march2026_better_rearranging (per improved_data_loader.PROBE_CONFIGS)
REAL_PROBES = ["9373/170", "9373/340", "12400/250", "8804/250"]


# ═══════════════════════════════════════════════════════════════════════════
# The four methods
# ═══════════════════════════════════════════════════════════════════════════

def method_nearest_bin(signal, target_freq, fs=FS, window_hz=0.1):
    """Pipeline convention: nearest FFT bin in ±window_hz of f_paddle."""
    sig = np.asarray(signal, dtype=float)
    sig = sig - np.nanmean(sig)
    N = len(sig)
    fft_vals = np.fft.fft(sig)
    freqs = np.fft.fftfreq(N, d=1.0 / fs)
    pos_mask = freqs > 0
    pos_freqs = freqs[pos_mask]
    amps = 2.0 * np.abs(fft_vals[pos_mask]) / N

    bin_mask = (pos_freqs >= target_freq - window_hz) & (pos_freqs <= target_freq + window_hz)
    if bin_mask.any():
        masked_freqs = pos_freqs[bin_mask]
        masked_amps = amps[bin_mask]
        k_local = int(np.argmin(np.abs(masked_freqs - target_freq)))
        f_sel = masked_freqs[k_local]
        a_sel = masked_amps[k_local]
    else:
        k = int(np.argmin(np.abs(pos_freqs - target_freq)))
        a_sel = amps[k]
        f_sel = pos_freqs[k]
    return float(a_sel), float(f_sel)


def method_parabolic(signal, target_freq, fs=FS, window_hz=0.1):
    """Parabolic (quadratic) interpolation over 3 bins around the nearest bin.

    Uses the standard 3-point formula:
        delta = 0.5 · (y_{-1} − y_{+1}) / (y_{-1} − 2y_0 + y_{+1})
        A     = y_0 − 0.25 · (y_{-1} − y_{+1}) · delta
    """
    sig = np.asarray(signal, dtype=float)
    sig = sig - np.nanmean(sig)
    N = len(sig)
    fft_vals = np.fft.fft(sig)
    freqs = np.fft.fftfreq(N, d=1.0 / fs)
    pos_mask = freqs > 0
    pos_freqs = freqs[pos_mask]
    amps = 2.0 * np.abs(fft_vals[pos_mask]) / N

    # Nearest bin to target
    bin_mask = (pos_freqs >= target_freq - window_hz) & (pos_freqs <= target_freq + window_hz)
    if bin_mask.any():
        masked_freqs = pos_freqs[bin_mask]
        k_global = int(np.where(pos_freqs == masked_freqs[np.argmin(np.abs(masked_freqs - target_freq))])[0][0])
    else:
        k_global = int(np.argmin(np.abs(pos_freqs - target_freq)))

    if 1 <= k_global <= len(amps) - 2:
        y_m, y_0, y_p = amps[k_global - 1], amps[k_global], amps[k_global + 1]
        denom = (y_m - 2 * y_0 + y_p)
        df_bin = pos_freqs[1] - pos_freqs[0]
        if denom != 0:
            delta = 0.5 * (y_m - y_p) / denom  # in bins, -0.5..+0.5
            a = y_0 - 0.25 * (y_m - y_p) * delta
            f = pos_freqs[k_global] + delta * df_bin
            return float(a), float(f)
    return float(amps[k_global]), float(pos_freqs[k_global])


def method_goertzel(signal, target_freq, fs=FS):
    """Single-frequency DFT evaluated at exactly target_freq.

    Equivalent to projecting the signal onto cos(2π f t) and sin(2π f t)
    basis functions. Does NOT care about bin grid alignment.
    """
    sig = np.asarray(signal, dtype=float)
    sig = sig - np.nanmean(sig)
    N = len(sig)
    n = np.arange(N)
    arg = 2.0 * np.pi * target_freq * n / fs
    A_c = (2.0 / N) * np.sum(sig * np.cos(arg))
    A_s = (2.0 / N) * np.sum(sig * np.sin(arg))
    amplitude = np.sqrt(A_c ** 2 + A_s ** 2)
    return float(amplitude), float(target_freq)


def method_ls_fit(signal, target_freq, fs=FS, include_stokes=True):
    """Least-squares sinusoid fit at target_freq (+ optional Stokes harmonics).

    Model: y[n] = DC + A·cos(ωt) + B·sin(ωt) [+ C·cos(2ωt) + D·sin(2ωt)]
    Returns amplitude of the fundamental = √(A² + B²).

    Adding Stokes 2f column is important on paddle-driven waves where 2f
    energy would otherwise cross-contaminate the fundamental estimate.
    """
    sig = np.asarray(signal, dtype=float)
    if np.isnan(sig).any():
        mask = ~np.isnan(sig)
        sig = sig[mask]
        t = np.arange(len(sig)) / fs  # OK if gaps are small; real-data path interpolates first
    else:
        t = np.arange(len(sig)) / fs
    omega = 2.0 * np.pi * target_freq
    cols = [np.ones_like(t), np.cos(omega * t), np.sin(omega * t)]
    if include_stokes:
        cols += [np.cos(2 * omega * t), np.sin(2 * omega * t)]
    X = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(X, sig, rcond=None)
    A_c, A_s = coef[1], coef[2]
    amplitude = float(np.sqrt(A_c ** 2 + A_s ** 2))
    return amplitude, float(target_freq)


METHODS = {
    "nearest_bin": method_nearest_bin,
    "parabolic":   method_parabolic,
    "goertzel":    method_goertzel,
    "ls_fit":      method_ls_fit,
}


# ═══════════════════════════════════════════════════════════════════════════
# §1  Synthetic test: pure tone, sweep true frequency through bin grid
# ═══════════════════════════════════════════════════════════════════════════

def _wind_like_noise(N, fs, rms=0.3, rng=None):
    """Colored noise emulating wind contamination:
       low-frequency tail (0–0.5 Hz) + broadband 3–5 Hz wind-wave band.
       rms sets the RMS level of the noise in mm."""
    rng = rng or np.random.default_rng(0)
    white = rng.standard_normal(N)
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    spec = np.fft.rfft(white)
    # Low-frequency tail: 1/f below 0.5 Hz, zero above
    tail = np.where(freqs < 0.5, 1.0 / np.clip(freqs, 0.02, None), 0.0)
    # Wind-wave band: Gaussian bump centred at 4 Hz, σ=1 Hz
    wwband = np.exp(-0.5 * ((freqs - 4.0) / 1.0) ** 2)
    shaped = spec * (0.2 * tail + wwband)
    noise = np.fft.irfft(shaped, n=N)
    noise *= rms / (np.std(noise) + 1e-12)
    return noise


def run_synthetic():
    """Sweep f_true through a bin grid; compare all 4 methods."""
    print("§1 Synthetic test")

    # Window length: 10 periods at nominal 1.3 Hz at fs=250 Hz → N ≈ 1923
    f_nominal = 1.3
    n_periods = 10
    N = int(round(n_periods * FS / f_nominal))
    df_bin = FS / N
    print(f"   N={N}, fs={FS:.0f} Hz, bin spacing Δf={df_bin:.4f} Hz (≈ f/10)")

    A_true = 1.0
    rng = np.random.default_rng(42)

    # Sweep 61 points across ~1 bin around 1.3 Hz
    f_sweep = np.linspace(f_nominal - 0.6 * df_bin, f_nominal + 0.6 * df_bin, 61)

    rows = []
    t = np.arange(N) / FS

    def build_signal(scenario, f_true):
        pure = A_true * np.cos(2 * np.pi * f_true * t)
        if scenario == "clean":
            return pure
        if scenario == "white_noise":
            return pure + rng.standard_normal(N) * 0.3
        if scenario == "wind_like":
            stokes2 = 0.15 * A_true * np.cos(2 * 2 * np.pi * f_true * t)
            return pure + _wind_like_noise(N, FS, rms=0.3, rng=rng) + stokes2
        raise ValueError(scenario)

    for scenario in ["clean", "white_noise", "wind_like"]:
        for f_true in f_sweep:
            noisy = build_signal(scenario, f_true)
            row = {
                "scenario":   scenario,
                "f_true_hz":  f_true,
                "bin_offset": (f_true - f_nominal) / df_bin,   # bin widths, -0.6..+0.6
            }
            # All methods are told target_freq = f_true (we trust our knowledge)
            for name, fn in METHODS.items():
                a, f = fn(noisy, f_true)
                row[f"A_{name}"]   = a
                row[f"err_{name}"] = (a - A_true) / A_true
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV_SYNTH, index=False, float_format="%.5f")
    print(f"   synth rows: {len(df)} → {OUT_CSV_SYNTH.relative_to(BASE)}")
    return df, f_nominal, df_bin, N


# ═══════════════════════════════════════════════════════════════════════════
# §2  Real-data test: nowind wave runs in the two canon folders
# ═══════════════════════════════════════════════════════════════════════════

def run_realdata():
    """Per-probe, per-run amplitudes from all 4 methods on nowind data."""
    print("§2 Real-data test (loading 2 canon folders)")
    meta, _, _, _ = load_analysis_data(*map(str, TARGET_DIRS), load_processed=False)

    mask = (
        meta["WaveFrequencyInput [Hz]"].notna()
        & (meta["WaveFrequencyInput [Hz]"] > 0)
        & (meta["WindCondition"] == "no")
        & (meta["quality_flag"] == "ok")
    )
    runs = meta[mask].copy()
    print(f"   {len(runs)} nowind wave runs (fullpanel / reverse / nopanel all included)")

    dfs = load_processed_dfs(*map(str, TARGET_DIRS))

    rows = []
    for _, row in runs.iterrows():
        path = row["path"]
        freq = row["WaveFrequencyInput [Hz]"]
        df = dfs.get(path)
        if df is None:
            continue

        for pos in REAL_PROBES:
            start_col = f"Computed Probe {pos} start"
            end_col   = f"Computed Probe {pos} end"
            if start_col not in row or end_col not in row:
                continue
            if pd.isna(row[start_col]) or pd.isna(row[end_col]):
                continue
            s = int(row[start_col])
            e = int(row[end_col])

            eta_col = f"eta_{pos}_interp" if f"eta_{pos}_interp" in df.columns else f"eta_{pos}"
            if eta_col not in df.columns:
                continue
            sig = df[eta_col].iloc[s:e + 1].to_numpy(dtype=float)
            if len(sig) < 50:
                continue

            # Linear-interp small NaN gaps (same as pipeline)
            nan_mask = np.isnan(sig)
            if nan_mask.mean() > 0.10:
                continue
            if nan_mask.any():
                idx = np.arange(len(sig))
                sig = np.interp(idx, idx[~nan_mask], sig[~nan_mask])

            entry = {
                "path":       path,
                "folder":     Path(path).parent.name[:20],
                "name":       Path(path).name,
                "probe":      pos,
                "freq_hz":    freq,
                "amp_V":      row["WaveAmplitudeInput [Volt]"],
                "panel":      row.get("PanelCondition", "?"),
                "N_samples":  len(sig),
                "df_bin":     FS / len(sig),
            }
            for name, fn in METHODS.items():
                a, f_sel = fn(sig, freq)
                entry[f"A_{name}"] = a * 1000  # convert m → mm
                entry[f"f_{name}"] = f_sel
            rows.append(entry)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV_REAL, index=False, float_format="%.5f")
    print(f"   real rows: {len(out)} (per-probe, per-run) → {OUT_CSV_REAL.relative_to(BASE)}")
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════

def plot_results(synth_df, f_nominal, df_bin, N, real_df):
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)
    colors = {"nearest_bin": "tab:red", "parabolic": "tab:orange",
              "goertzel":    "tab:blue", "ls_fit":    "tab:green"}

    # Panels (a)(b)(c): synthetic error vs bin offset, one per scenario
    for i, scenario in enumerate(["clean", "white_noise", "wind_like"]):
        ax = fig.add_subplot(gs[i, 0])
        sub = synth_df[synth_df["scenario"] == scenario]
        for method, color in colors.items():
            ax.plot(sub["bin_offset"], sub[f"err_{method}"] * 100,
                    "-", color=color, lw=1.2, label=method, alpha=0.85)
        ax.axhline(0, color="k", lw=0.5, alpha=0.5)
        ax.set_xlim(-0.6, 0.6)
        ax.set_xlabel("true-tone offset from target (bin widths)")
        ax.set_ylabel("amplitude error (%)")
        ax.set_title(f"({chr(97 + i)}) synthetic: {scenario}  "
                     f"(N={N}, Δf={df_bin:.3f} Hz)", fontsize=10)
        ax.legend(fontsize=8, ncol=2, loc="best")
        ax.grid(True, alpha=0.3)

    # Panel (d): real-data cross-method scatter — ls_fit as reference
    ax = fig.add_subplot(gs[0, 1])
    ref = real_df["A_ls_fit"]
    for method, color in colors.items():
        if method == "ls_fit":
            continue
        ax.scatter(ref, real_df[f"A_{method}"], s=8, alpha=0.5,
                   color=color, label=method)
    lo = min(ref.min(), real_df[[c for c in real_df.columns if c.startswith("A_")]].min().min())
    hi = max(ref.max(), real_df[[c for c in real_df.columns if c.startswith("A_")]].max().max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5, label="y=x")
    ax.set_xlabel("A (ls_fit) — mm")
    ax.set_ylabel("A (other method) — mm")
    ax.set_title(f"(d) real-data: methods vs ls_fit  (n={len(real_df)})", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel (e): relative disagreement distribution (method − ls_fit) / ls_fit
    ax = fig.add_subplot(gs[1, 1])
    for method, color in colors.items():
        if method == "ls_fit":
            continue
        rel = (real_df[f"A_{method}"] - real_df["A_ls_fit"]) / real_df["A_ls_fit"] * 100
        ax.hist(rel, bins=40, alpha=0.45, color=color,
                label=f"{method}  (μ={rel.mean():+.2f}%, σ={rel.std():.2f}%)")
    ax.axvline(0, color="k", ls="--", lw=0.8)
    ax.set_xlabel("(A_method − A_ls_fit) / A_ls_fit  (%)")
    ax.set_ylabel("count")
    ax.set_title("(e) real-data: disagreement from ls_fit reference", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Panel (f): nearest_bin error vs bin-offset on REAL data
    ax = fig.add_subplot(gs[2, 1])
    real_df = real_df.copy()
    real_df["k_nearest"] = (real_df["freq_hz"] / real_df["df_bin"]).round().astype(int)
    real_df["bin_center_hz"] = real_df["k_nearest"] * real_df["df_bin"]
    real_df["bin_offset_real"] = (real_df["freq_hz"] - real_df["bin_center_hz"]) / real_df["df_bin"]
    for method, color in [("nearest_bin", "tab:red"), ("parabolic", "tab:orange")]:
        rel = (real_df[f"A_{method}"] - real_df["A_ls_fit"]) / real_df["A_ls_fit"] * 100
        ax.scatter(real_df["bin_offset_real"], rel, s=8, alpha=0.5,
                   color=color, label=method)
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.set_xlim(-0.6, 0.6)
    ax.set_xlabel("f_paddle offset from nearest bin (bin widths)")
    ax.set_ylabel("(A_method − A_ls_fit) / A_ls_fit  (%)")
    ax.set_title("(f) real-data: bin-offset predicts nearest-bin error", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("FFT amplitude extraction — method comparison", fontsize=12, y=0.995)
    fig.savefig(OUT_PNG, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"   figure → {OUT_PNG.relative_to(BASE)}")


# ═══════════════════════════════════════════════════════════════════════════
# Findings writeup
# ═══════════════════════════════════════════════════════════════════════════

def write_findings(synth_df, real_df):
    lines = []
    lines.append("# FFT amplitude extraction — method comparison")
    lines.append("")
    lines.append(f"Generated: {pd.Timestamp.utcnow().isoformat()[:19]}Z")
    lines.append("")
    lines.append("**Methods**: `nearest_bin` (pipeline), `parabolic` (3-bin quadratic interp), "
                 "`goertzel` (single-frequency DFT at exactly f_paddle), "
                 "`ls_fit` (LS with DC + fundamental + Stokes 2f basis).")
    lines.append("")

    # ── §1 synthetic ──────────────────────────────────────────────────────
    lines.append("## §1 Synthetic test")
    lines.append("")
    lines.append("Pure tone, amplitude = 1.0 mm, frequency swept across ±0.6 bin widths "
                 "around f_nominal = 1.3 Hz. Window N=1923 (10 periods at 1.3 Hz, fs=250 Hz) → "
                 "bin spacing Δf = 0.130 Hz.")
    lines.append("")
    lines.append("| scenario | method | max \\|error\\| % | mean \\|error\\| % |")
    lines.append("|---|---|---|---|")
    for scenario in ["clean", "white_noise", "wind_like"]:
        sub = synth_df[synth_df["scenario"] == scenario]
        for method in ["nearest_bin", "parabolic", "goertzel", "ls_fit"]:
            err = sub[f"err_{method}"].abs() * 100
            lines.append(f"| {scenario} | {method} | {err.max():.2f} | {err.mean():.2f} |")
    lines.append("")

    # ── §2 real data ─────────────────────────────────────────────────────
    lines.append("## §2 Real-data cross-check")
    lines.append("")
    lines.append(f"Dataset: nowind wave runs in the two canonical March-2026 folders")
    lines.append(f"({', '.join(p.name for p in TARGET_DIRS)}).")
    lines.append(f"Per-probe measurements: **n = {len(real_df)}** "
                 f"across {real_df['probe'].nunique()} probes, "
                 f"{real_df['path'].nunique()} runs, "
                 f"{real_df['freq_hz'].nunique()} frequencies.")
    lines.append("")
    lines.append("Using `ls_fit` as the reference (it is mathematically the exact paddle-frequency "
                 "amplitude, modulo Stokes-2f explicit separation). Relative disagreement = "
                 "(A_method − A_ls_fit) / A_ls_fit.")
    lines.append("")
    lines.append("| method | mean % | std % | max \\|%\\| | median \\|%\\| |")
    lines.append("|---|---|---|---|---|")
    for method in ["nearest_bin", "parabolic", "goertzel", "ls_fit"]:
        rel = (real_df[f"A_{method}"] - real_df["A_ls_fit"]) / real_df["A_ls_fit"] * 100
        lines.append(f"| {method} | {rel.mean():+.3f} | {rel.std():.3f} | "
                     f"{rel.abs().max():.2f} | {rel.abs().median():.2f} |")
    lines.append("")
    lines.append("**Breakdown by frequency** (max \\|nearest_bin − ls_fit\\|/ls_fit %):")
    lines.append("")
    lines.append("| freq_hz | n | max % | median % |")
    lines.append("|---|---|---|---|")
    for f, sub in real_df.groupby("freq_hz"):
        rel = (sub["A_nearest_bin"] - sub["A_ls_fit"]) / sub["A_ls_fit"] * 100
        lines.append(f"| {f:.3f} | {len(sub)} | {rel.abs().max():.2f} | {rel.abs().median():.2f} |")
    lines.append("")

    # ── Verdict ──────────────────────────────────────────────────────────
    lines.append("## Verdict")
    lines.append("")
    rel_nb = (real_df["A_nearest_bin"] - real_df["A_ls_fit"]) / real_df["A_ls_fit"] * 100
    rel_pb = (real_df["A_parabolic"]   - real_df["A_ls_fit"]) / real_df["A_ls_fit"] * 100
    rel_gz = (real_df["A_goertzel"]    - real_df["A_ls_fit"]) / real_df["A_ls_fit"] * 100
    lines.append(f"- `nearest_bin` disagrees with `ls_fit` by median {rel_nb.abs().median():.2f}%, "
                 f"max {rel_nb.abs().max():.2f}%.")
    lines.append(f"- `parabolic` reduces the disagreement to median {rel_pb.abs().median():.2f}%, "
                 f"max {rel_pb.abs().max():.2f}%.")
    lines.append(f"- `goertzel` agrees with `ls_fit` within median {rel_gz.abs().median():.3f}%, "
                 f"max {rel_gz.abs().max():.2f}% (numerically essentially identical).")
    lines.append("")
    lines.append("**Recommendation**: replace `compute_amplitudes_from_fft` in "
                 "`wavescripts/signal_processing.py` with an `ls_fit` evaluation at f_paddle. "
                 "It is ~15 lines of code, bin-grid-independent, and gives "
                 "Stokes-2f amplitude as a bonus output. `parabolic` is an easier "
                 "drop-in if the team prefers to keep the FFT path but corrects the "
                 "sinc-attenuation at near-zero cost.")
    lines.append("")
    lines.append("## See also")
    lines.append("")
    lines.append(f"- Figure: `{OUT_PNG.relative_to(BASE)}`")
    lines.append(f"- Synthetic CSV: `{OUT_CSV_SYNTH.relative_to(BASE)}`")
    lines.append(f"- Real-data CSV: `{OUT_CSV_REAL.relative_to(BASE)}`")
    lines.append(f"- Methodology memo: `memory/methodology_fft_peak_bin_bias.md`")

    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"   findings → {OUT_MD.relative_to(BASE)}")


# ═══════════════════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    synth_df, f_nominal, df_bin, N = run_synthetic()
    real_df = run_realdata()
    plot_results(synth_df, f_nominal, df_bin, N, real_df)
    write_findings(synth_df, real_df)
    print("\nDone.")
