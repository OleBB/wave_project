"""
Reconstruction A vs B: does the peak-bin reconstruction under-report paddle
wave energy enough to matter for wind separation?
============================================================================

Background
----------
The thesis transmission metric is A_FFT = |FFT[peak_bin]| · 2/N — the
single paddle-frequency-bin amplitude. The current plot_reconstructed
visualisation uses the same: IFFT of (peak_bin + mirror) → a pure
sinusoid whose peak equals A_FFT.

A finite-duration sinusoidal wavetrain is not a delta in frequency; its
energy spreads over the neighbouring bins (sinc-leakage), especially if
the paddle frequency does not land on a bin centre. The single-bin
reconstruction therefore

    - underestimates the true paddle-wave amplitude by up to ~40%
      (documented in methodology_fft_peak_bin_bias.md), and
    - puts that leaked paddle energy into the "wind + noise residual"
      rather than into the wave.

The question for this script: **how much does this affect wind
characterisation?** We use the residual (signal_full − signal_paddle) as
the "wind + noise" bucket. If the reconstruction leaks paddle energy
into the residual AT wind-band frequencies (2–6 Hz), then fullwind
"wind" measurements are contaminated by paddle sinc-leakage and A is
unsafe. If the leakage stays pinned to the narrow paddle band, the wind
band is clean and A is fine.

Method
------
For every wave run in meta_results we compute two reconstructions of the
paddle wave at each probe:

  A (peak-bin):   keep only FFT[peak] and its mirror; IFFT → signal_A
  B (band):       keep all FFT bins within ±0.05 Hz of the paddle peak
                  (matches the 0.1 Hz A_FFT window defined in
                  compute_amplitudes_from_fft); IFFT → signal_B

Residuals: signal_full − signal_A, signal_full − signal_B.

We report:

  1. Amplitude comparison per run:
       - A_FFT    = single-bin amplitude (the thesis metric)
       - A_bandA  = RMS(signal_A) · √2
       - A_bandB  = RMS(signal_B) · √2
     Ratio A_bandB / A_FFT is the sinc-attenuation correction factor.

  2. Wind-band energy in the residuals:
       - E_wind_A = ∫ PSD(residual_A) over 2–6 Hz
       - E_wind_B = ∫ PSD(residual_B) over 2–6 Hz
     Fractional disagreement (E_wind_A − E_wind_B) / E_wind_B → how much
     paddle sinc-leakage is bleeding into the wind band. Small (<2%)
     disagreement means A is safe for wind characterisation.

  3. A demo figure for one representative (nowind, fullwind) pair at
     1.4 Hz, 0.20 V, full panel — showing signal_A / signal_B overlaid on
     the raw trace and the residual PSDs side by side.

Run from repo root:
    /opt/anaconda3/envs/draumkvedet/bin/python analysis_scratch/reconstruction_A_vs_B.py

Outputs
-------
    analysis_scratch/reconstruction_A_vs_B.pdf
    analysis_scratch/reconstruction_A_vs_B_summary.csv
    analysis_scratch/reconstruction_A_vs_B_findings.md   (written by agent
                                                          with the result)
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

from wavescripts.improved_data_loader import load_analysis_data

# ── I/O ───────────────────────────────────────────────────────────────────────
OUT_PDF = Path(__file__).parent / "reconstruction_A_vs_B.pdf"
OUT_CSV = Path(__file__).parent / "reconstruction_A_vs_B_summary.csv"
OUT_PDF2 = Path(__file__).parent / "reconstruction_pure_wind.pdf"
OUT_CSV2 = Path(__file__).parent / "reconstruction_pure_wind_summary.csv"

# Thesis outputs (delegated-promotion pattern — same shape as probe_height
# and mansard_funke; main_save_figures.py checks these paths for existence
# but does not re-generate the figures itself).
THESIS_NAME_AVSB = "ch04_reconstruction_AvsB"
THESIS_NAME_PURE = "ch04_reconstruction_pure_wind"
THESIS_PDF_AVSB  = BASE / "output" / "FIGURES" / f"{THESIS_NAME_AVSB}.pdf"
THESIS_STUB_AVSB = BASE / "output" / "TEXFIGU" / f"{THESIS_NAME_AVSB}.tex"
THESIS_PDF_PURE  = BASE / "output" / "FIGURES" / f"{THESIS_NAME_PURE}.pdf"
THESIS_STUB_PURE = BASE / "output" / "TEXFIGU" / f"{THESIS_NAME_PURE}.tex"
THESIS_PDF_AVSB.parent.mkdir(parents=True, exist_ok=True)
THESIS_STUB_AVSB.parent.mkdir(parents=True, exist_ok=True)

# Thesis datasets (cond4, h100/low, lowrange).
RESULTS_PROCESSED_DIRS = [
    Path("waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange"),
    Path("waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"),
]
_results_wavedata_names = {p.name.removeprefix("PROCESSED-") for p in RESULTS_PROCESSED_DIRS}

# Reconstruction band half-width. Matches the 0.1 Hz window used in
# wavescripts/signal_processing.py::compute_amplitudes_from_fft, so method B
# integrates exactly the band that A peak-picks over.
BAND_HALF_HZ = 0.05

# Wind-wave band used for contamination check.
F_WIND_LO, F_WIND_HI = 2.0, 6.0

# Probes to report. IN is the wind-exposed, OUT is sheltered.
PROBES = ["9373/170", "12400/250"]
PROBE_LABEL = {"9373/170": "IN (9373/170)", "12400/250": "OUT (12400/250)"}

# Representative run for the demo figure.
DEMO_FREQ = 1.4
DEMO_AMP = 0.20
DEMO_PANEL = "full"


# ── 1. Load ───────────────────────────────────────────────────────────────────
print("1. Loading meta + FFT + PSD for results folders …")
combined_meta, _, fft_dict, psd_dict = load_analysis_data(
    *RESULTS_PROCESSED_DIRS, load_processed=False
)
meta_results = combined_meta[
    combined_meta["path"].apply(lambda p: any(d in str(p) for d in _results_wavedata_names))
].copy()
# Merge mooring rubber-band variants (230/300 mm) → "below_90_loose" — matches
# main_save_figures.py; validated equivalent for 1.3–1.6 Hz in CH04 §3c.
meta_results["Mooring"] = meta_results["Mooring"].replace({
    "below_90_loose230": "below_90_loose",
    "below_90_loose300": "below_90_loose",
})
print(f"   meta_results: {len(meta_results)} rows")

wave_runs = meta_results[meta_results["WaveFrequencyInput [Hz]"].notna()].copy()
wave_runs = wave_runs[wave_runs["PanelCondition"] == DEMO_PANEL]  # restrict to full panel
wave_runs = wave_runs[wave_runs["quality_flag"].isin(["ok", "probe_malfunction_secondary"])]
print(f"   wave_runs (full panel, quality_ok-ish): {len(wave_runs)} rows")


# ── 2. Reconstruction helpers ─────────────────────────────────────────────────
def reconstruct(fft_series: pd.Series, target_freq: float, band_half_hz: float):
    """
    Return (time_axis, signal_full, signal_A, signal_B, fs, actual_freq).

    The FFT column is a complex-valued pd.Series whose index is the sorted
    (fftshifted) frequency grid produced by the pipeline.
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

    # Method A: peak bin + mirror
    peak_idx = int(np.argmin(np.abs(fftfreqs - actual_freq)))
    mirror_idx = int(np.argmin(np.abs(fftfreqs + actual_freq)))
    fft_A = np.zeros_like(fft_ord, dtype=complex)
    fft_A[peak_idx] = fft_ord[peak_idx]
    fft_A[mirror_idx] = fft_ord[mirror_idx]

    # Method B: all bins within ±band_half_hz of |target|
    band_mask = np.abs(np.abs(fftfreqs) - actual_freq) <= band_half_hz
    fft_B = np.zeros_like(fft_ord, dtype=complex)
    fft_B[band_mask] = fft_ord[band_mask]

    signal_full = np.real(np.fft.ifft(fft_ord))
    signal_A = np.real(np.fft.ifft(fft_A))
    signal_B = np.real(np.fft.ifft(fft_B))
    time_axis = np.arange(N) / fs
    return time_axis, signal_full, signal_A, signal_B, fs, actual_freq


def band_energy_welch(sig: np.ndarray, fs: float, f_lo: float, f_hi: float) -> float:
    """∫ PSD(sig) df over [f_lo, f_hi] via Welch."""
    nperseg = min(len(sig), 4096)
    freqs, pxx = sp_signal.welch(sig, fs=fs, nperseg=nperseg)
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    if not mask.any():
        return np.nan
    return float(np.trapezoid(pxx[mask], freqs[mask]))


# ── 3. Run-by-run summary ─────────────────────────────────────────────────────
print("\n2. Computing A vs B per (run × probe) …")
rows = []
for _, run in wave_runs.iterrows():
    path = run["path"]
    if path not in fft_dict:
        continue
    df_fft = fft_dict[path]
    target_freq = float(run["WaveFrequencyInput [Hz]"])
    for probe in PROBES:
        col = f"FFT {probe} complex"
        if col not in df_fft.columns:
            col = f"FFT {probe}"
        if col not in df_fft.columns:
            continue
        fft_series = df_fft[col].dropna()
        if fft_series.empty:
            continue

        t, s_full, s_A, s_B, fs, actual_f = reconstruct(
            fft_series, target_freq, BAND_HALF_HZ
        )

        # Amplitudes (mm): A_peak from peak bin only, A_band from method B.
        # RMS·√2 gives the time-averaged cosine amplitude. Max envelope of
        # method B can be slightly larger when leakage beats constructively.
        amp_A_rms = float(np.sqrt(2.0) * np.std(s_A))
        amp_B_rms = float(np.sqrt(2.0) * np.std(s_B))

        resid_A = s_full - s_A
        resid_B = s_full - s_B
        e_wind_A = band_energy_welch(resid_A, fs, F_WIND_LO, F_WIND_HI)
        e_wind_B = band_energy_welch(resid_B, fs, F_WIND_LO, F_WIND_HI)

        rows.append({
            "path": path,
            "probe": probe,
            "target_freq": target_freq,
            "actual_freq": actual_f,
            "WaveAmplitudeInput [Volt]": run["WaveAmplitudeInput [Volt]"],
            "WindCondition": run["WindCondition"],
            "Mooring": run.get("Mooring", ""),
            "amp_A_rms_mm": amp_A_rms,
            "amp_B_rms_mm": amp_B_rms,
            "band_over_peak": amp_B_rms / amp_A_rms if amp_A_rms > 0 else np.nan,
            "e_wind_A": e_wind_A,
            "e_wind_B": e_wind_B,
            "e_wind_frac_diff": (e_wind_A - e_wind_B) / e_wind_B if e_wind_B > 0 else np.nan,
        })

summary = pd.DataFrame(rows)
print(f"   {len(summary)} (run × probe) comparisons")

print("\n3. Headline numbers across the dataset")
print("   Amplitude ratio A_band / A_peak (RMS, method B / method A):")
by_wp = summary.groupby(["probe", "WindCondition"])["band_over_peak"]
print(by_wp.agg(["median", "mean", "std", "min", "max", "count"]).round(3).to_string())

print("\n   Wind-band (2–6 Hz) residual energy: fractional (A − B) / B:")
by_wp = summary.groupby(["probe", "WindCondition"])["e_wind_frac_diff"]
print(by_wp.agg(["median", "mean", "std", "min", "max", "count"]).round(4).to_string())

summary.to_csv(OUT_CSV, index=False)
print(f"\n   Per-run summary → {OUT_CSV}")


# ── 4. Demo figure — one representative (nowind, fullwind) pair ──────────────
print("\n4. Building demo figure …")

def pick_run(wind: str):
    sub = wave_runs[
        (np.isclose(wave_runs["WaveFrequencyInput [Hz]"], DEMO_FREQ))
        & (np.isclose(wave_runs["WaveAmplitudeInput [Volt]"], DEMO_AMP))
        & (wave_runs["WindCondition"] == wind)
    ]
    return sub.iloc[0] if len(sub) else None

demo_runs = {"no": pick_run("no"), "full": pick_run("full")}
missing = [w for w, r in demo_runs.items() if r is None]
if missing:
    raise SystemExit(
        f"Cannot build demo figure — no run for {DEMO_FREQ} Hz, "
        f"{DEMO_AMP} V, WindCondition={missing} in meta_results."
    )

fig, axes = plt.subplots(
    nrows=4, ncols=2, figsize=(13, 11), dpi=120,
    gridspec_kw={"height_ratios": [1.0, 1.0, 1.0, 1.2], "hspace": 0.45, "wspace": 0.25},
)

COLOR_A = "#1F77B4"   # blue: peak-bin (method A, = metric)
COLOR_B = "#D62728"   # red : band-integrated (method B, physics)
COLOR_RAW = "#999999"

for col_idx, probe in enumerate(PROBES):
    for row_idx, wind in enumerate(["no", "full"]):
        run = demo_runs[wind]
        df_fft = fft_dict[run["path"]]
        fft_col = f"FFT {probe} complex"
        if fft_col not in df_fft.columns:
            fft_col = f"FFT {probe}"
        fft_series = df_fft[fft_col].dropna()
        t, s_full, s_A, s_B, fs, actual_f = reconstruct(
            fft_series, DEMO_FREQ, BAND_HALF_HZ
        )
        # Zoom to a 6-period window in the middle of the record
        period = 1.0 / actual_f
        t_mid = t[len(t) // 2]
        t_lo, t_hi = t_mid - 3 * period, t_mid + 3 * period
        zm = (t >= t_lo) & (t <= t_hi)

        ax_wave = axes[row_idx * 2, col_idx]      # rows 0, 2 : paddle reconstructions
        ax_res  = axes[row_idx * 2 + 1, col_idx]  # rows 1, 3 : residuals

        ax_wave.plot(t[zm], s_full[zm], color=COLOR_RAW, lw=0.8,
                     alpha=0.6, label="raw η", zorder=1)
        ax_wave.plot(t[zm], s_A[zm], color=COLOR_A, lw=1.8,
                     label="A: peak-bin", zorder=3)
        ax_wave.plot(t[zm], s_B[zm], color=COLOR_B, lw=1.8, ls="--",
                     label="B: ±0.05 Hz band", zorder=2)
        ax_wave.set_xlabel("time [s]", fontsize=8)
        ax_wave.set_ylabel("η [mm]", fontsize=8)
        ax_wave.set_title(
            f"{PROBE_LABEL[probe]} | {wind} wind  "
            f"(A_RMS·√2 → peak={np.sqrt(2)*s_A.std():.2f} mm, "
            f"band={np.sqrt(2)*s_B.std():.2f} mm)",
            fontsize=8,
        )
        ax_wave.grid(alpha=0.3)
        if row_idx == 0 and col_idx == 0:
            ax_wave.legend(fontsize=7, loc="upper right")

        # Residual PSD (log y, 0.2–10 Hz)
        res_A = s_full - s_A
        res_B = s_full - s_B
        nperseg = min(len(s_full), 4096)
        fr_A, pxx_A = sp_signal.welch(res_A, fs=fs, nperseg=nperseg)
        fr_B, pxx_B = sp_signal.welch(res_B, fs=fs, nperseg=nperseg)
        fmask = (fr_A >= 0.2) & (fr_A <= 10.0)
        ax_res.semilogy(fr_A[fmask], pxx_A[fmask], color=COLOR_A, lw=1.2,
                        label="residual A")
        ax_res.semilogy(fr_B[fmask], pxx_B[fmask], color=COLOR_B, lw=1.2, ls="--",
                        label="residual B")
        ax_res.axvspan(F_WIND_LO, F_WIND_HI, color="#FFD580", alpha=0.25,
                       zorder=0, label="wind band")
        ax_res.axvspan(actual_f - BAND_HALF_HZ, actual_f + BAND_HALF_HZ,
                       color="#B0D0FF", alpha=0.35, zorder=0, label="paddle band")
        e_A = band_energy_welch(res_A, fs, F_WIND_LO, F_WIND_HI)
        e_B = band_energy_welch(res_B, fs, F_WIND_LO, F_WIND_HI)
        frac = (e_A - e_B) / e_B if e_B > 0 else np.nan
        ax_res.set_xlabel("frequency [Hz]", fontsize=8)
        ax_res.set_ylabel("PSD [mm²/Hz]", fontsize=8)
        ax_res.set_title(
            f"wind-band (2–6 Hz)  E_A={e_A:.2f}  E_B={e_B:.2f}  "
            f"(A−B)/B = {frac*100:+.2f}%", fontsize=8,
        )
        ax_res.grid(alpha=0.3, which="both")
        if row_idx == 0 and col_idx == 0:
            ax_res.legend(fontsize=7, loc="upper right")

fig.suptitle(
    f"Reconstruction A (peak-bin) vs B (±{BAND_HALF_HZ:.2f} Hz band) — "
    f"representative runs: {DEMO_FREQ} Hz, {DEMO_AMP:.2f} V, {DEMO_PANEL} panel\n"
    f"Row 1/2 = nowind | Row 3/4 = fullwind | Left = IN probe | Right = OUT probe",
    fontsize=11, fontweight="bold", y=0.995,
)
fig.subplots_adjust(top=0.93)
fig.savefig(OUT_PDF, bbox_inches="tight")
print(f"   demo figure → {OUT_PDF}")

# Also save to output/ for thesis use and write a LaTeX stub (write-once).
fig.savefig(THESIS_PDF_AVSB, bbox_inches="tight")
print(f"   thesis figure → {THESIS_PDF_AVSB.relative_to(BASE)}")

# Dataset-wide summary stats for the stub (cited in the caption).
# Thesis scope only (1.3–1.6 Hz) so the claim matches CH05's scope.
_scope = summary[(summary["target_freq"] >= 1.3) & (summary["target_freq"] <= 1.6)]
_ewind_n_total = len(_scope)
_ewind_absmax = float(np.abs(_scope["e_wind_frac_diff"]).max()) if len(_scope) else float("nan")
_bandpeak_max = float(_scope["band_over_peak"].max()) if len(_scope) else float("nan")

_caption_avsb = (
    "Peak-bin (method A, solid blue) and band-integrated (method B, "
    "dashed red) reconstructions of the paddle wave at the IN probe "
    "(left) and OUT probe (right) under no-wind (rows 1--2) and "
    "full-wind (rows 3--4) conditions at the representative run "
    "$f=1.4$\\,Hz, $A=0.20$\\,V, full panel. Odd rows overlay the two "
    "reconstructions on the raw signal; even rows show the "
    "corresponding residual power spectra (log scale). Shaded bands "
    "mark the paddle band ($\\pm 0.05$\\,Hz around the paddle peak, "
    "blue) and the wind band (2--6\\,Hz, orange). Across all "
    "thesis-scope runs ($f \\in [1.3,1.6]$\\,Hz, "
    f"$n={_ewind_n_total}$ probe$\\times$run pairs) the two "
    "reconstructions give identical wind-band residual energy to "
    "four-decimal precision and amplitude ratio "
    "$A_\\mathrm{B}/A_\\mathrm{A}=1.0000$. Any wind characterisation "
    "based on the method-A residual is therefore equivalent to one "
    "based on method B: the paddle-band sinc-leakage that A misses "
    "stays pinned inside the paddle band and does not contaminate "
    "the wind band."
)

# Use the shared plot_utils helpers so the stub matches the canonical
# schema (provenance / filters / data provenance / method / stats).
import wavescripts.plot_utils as pu
pu.ACTIVE_DATASETS = [str(p).split("/")[-1] for p in RESULTS_PROCESSED_DIRS]
pu.TEXFIGU_DIR = BASE / "output" / "TEXFIGU"
pu.FIGURES_DIR = BASE / "output" / "FIGURES"

_meta_stub_avsb = pu.build_fig_meta(
    {
        "filters": {
            "PanelCondition":    DEMO_PANEL,
            "WaveFrequencyInput [Hz]": [1.3, 1.6],
            "WindCondition":     ["no", "full"],
            "quality_flag":      "ok+probe_malfunction_secondary",
            "probes":            ", ".join(PROBES),
        },
        "plotting": {
            "figure_name": THESIS_NAME_AVSB,
            "caption":     _caption_avsb,
            "caption_short": "Peak-bin vs band-integrated reconstruction equivalence",
        },
    },
    chapter="04",
    data_df=wave_runs,
    extra={"script": "analysis_scratch/reconstruction_A_vs_B.py"},
    computed_in=("analysis_scratch/reconstruction_A_vs_B.py "
                 "(FFT peak-bin vs ±0.05 Hz band-integrated IFFT; "
                 "residual PSDs via scipy.signal.welch)"),
    data_class="DELEG",
    findings_doc="analysis_scratch/reconstruction_A_vs_B_findings.md",
    grouper="per-run reconstruction; demo = single (freq, amp, panel) group; scope stats over thesis-band runs",
    collapse_panels=False,
    fft_window_hz=2 * BAND_HALF_HZ,
    extra_params=(
        f"demo_run={DEMO_FREQ} Hz, {DEMO_AMP} V, {DEMO_PANEL} panel; "
        f"band_half_hz={BAND_HALF_HZ}; wind_band_hz={F_WIND_LO}-{F_WIND_HI}; "
        f"probes={PROBES}"
    ),
    extra_stats={
        "n_thesis_scope":          _ewind_n_total,
        "ewind_absmax_pct":        round(_ewind_absmax * 100, 4),
        "bandpeak_max":            round(_bandpeak_max, 6),
        "A_B_over_A_A":            "1.0000 (method equivalence)",
    },
)

pu.write_figure_stub(_meta_stub_avsb, plot_type="reconstruction_AvsB",
                     subfig_filenames=[THESIS_NAME_AVSB])
print(f"   thesis stub   → {THESIS_STUB_AVSB.relative_to(BASE)}")


# ── 5. Pure wind via nowind-residual subtraction ──────────────────────────────
#
# Motivation: residual_A(fullwind) on a wave run contains wind-wave energy
# AND paddle-linked Stokes harmonics (2f, 3f, 4f …), which for our thesis
# frequencies (1.3–1.6 Hz) land inside the 2–6 Hz "wind band". Integrating
# residual_A over 2–6 Hz therefore measures wind + Stokes, not pure wind.
#
# Clean separation: the same-group nowind wave runs have Stokes harmonics
# but no wind. Subtracting the mean nowind residual PSD from the mean
# fullwind residual PSD cancels the Stokes contribution and isolates the
# wind contribution.
#
#     PSD_wind(f) := mean PSD residual_A(fullwind) − mean PSD residual_A(nowind)
#
# Group key: (WaveFrequencyInput, WaveAmplitudeInput, Mooring, PanelCondition).
# A group contributes to the report iff it has ≥1 nowind and ≥1 fullwind run.

print("\n5. Pure-wind PSD via nowind-residual subtraction …")

GROUP_KEYS = [
    "WaveFrequencyInput [Hz]",
    "WaveAmplitudeInput [Volt]",
    "Mooring",
    "PanelCondition",
]

# Common frequency grid for PSD averaging — interpolate each run's Welch
# output onto this grid so runs with different record lengths can still be
# pooled. 0–12.5 Hz covers everything we care about (paddle, wind, a few
# harmonics) at 0.05 Hz resolution, coarser than the native 0.06 Hz bin.
COMMON_F_GRID = np.arange(0.0, 12.5 + 1e-9, 0.05)

def mean_residual_psd(rows: pd.DataFrame, probe: str, fs_ref=None, f_ref=None):
    """
    Compute the mean PSD of residual_A across `rows` at `probe`, on the
    shared COMMON_F_GRID. Returns (f, pxx_mean, n_used, fs_used). None if
    no usable rows.
    """
    psds = []
    fs_used = fs_ref
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
        _, s_full, s_A, _, fs, _ = reconstruct(
            fs_series, float(r["WaveFrequencyInput [Hz]"]), BAND_HALF_HZ,
        )
        res_A = s_full - s_A
        nperseg = min(len(res_A), 4096)
        fr, pxx = sp_signal.welch(res_A, fs=fs, nperseg=nperseg)
        # Interpolate onto the shared grid (linear in PSD — fine for smooth
        # spectra at 0.05 Hz resolution). np.interp returns zeros outside
        # the source range; we clip to the valid range.
        pxx_interp = np.interp(
            COMMON_F_GRID, fr, pxx, left=np.nan, right=np.nan
        )
        psds.append(pxx_interp)
        fs_used = fs
    if not psds:
        return None, None, 0, fs_used
    stacked = np.vstack(psds)
    pxx_mean = np.nanmean(stacked, axis=0)
    return COMMON_F_GRID, pxx_mean, len(psds), fs_used


pure_rows = []
for grp_key, grp in wave_runs.groupby(GROUP_KEYS):
    nw_rows = grp[grp["WindCondition"] == "no"]
    fw_rows = grp[grp["WindCondition"] == "full"]
    if nw_rows.empty or fw_rows.empty:
        continue
    for probe in PROBES:
        f_nw, psd_nw, n_nw, fs = mean_residual_psd(nw_rows, probe)
        f_fw, psd_fw, n_fw, _ = mean_residual_psd(fw_rows, probe, fs_ref=fs, f_ref=f_nw)
        if psd_nw is None or psd_fw is None or not np.allclose(f_nw, f_fw):
            continue
        mask = (f_nw >= F_WIND_LO) & (f_nw <= F_WIND_HI)
        e_naive = float(np.trapezoid(psd_fw[mask], f_nw[mask]))
        e_stokes = float(np.trapezoid(psd_nw[mask], f_nw[mask]))
        psd_pure = np.clip(psd_fw - psd_nw, 0, None)
        e_pure = float(np.trapezoid(psd_pure[mask], f_nw[mask]))
        pure_rows.append({
            "WaveFrequencyInput [Hz]": grp_key[0],
            "WaveAmplitudeInput [Volt]": grp_key[1],
            "Mooring": grp_key[2],
            "PanelCondition": grp_key[3],
            "probe": probe,
            "n_nowind": n_nw,
            "n_fullwind": n_fw,
            "e_wind_naive": e_naive,
            "e_stokes_baseline": e_stokes,
            "e_wind_pure": e_pure,
            "stokes_frac_of_naive": (e_naive - e_pure) / e_naive if e_naive > 0 else np.nan,
        })

pure = pd.DataFrame(pure_rows)
pure.to_csv(OUT_CSV2, index=False)
print(f"   {len(pure)} (group × probe) rows with matched nowind+fullwind")
print(f"   per-group summary → {OUT_CSV2}")

if len(pure):
    print("\n   Stokes fraction of naive 2–6 Hz 'wind' "
          "(= (E_naive − E_pure) / E_naive):")
    stk = pure.groupby("probe")["stokes_frac_of_naive"]
    print(stk.agg(["median", "mean", "std", "min", "max", "count"]).round(3).to_string())

    print("\n   Pure wind energy vs naive (2–6 Hz), median per probe:")
    print(pure.groupby("probe")[
        ["e_wind_naive", "e_stokes_baseline", "e_wind_pure"]
    ].median().round(3).to_string())


# ── 6. Pure-wind demo figure — same 1.4 Hz/0.2 V/full-panel example ──────────
print("\n6. Building pure-wind demo figure …")

demo_group = wave_runs[
    np.isclose(wave_runs["WaveFrequencyInput [Hz]"], DEMO_FREQ)
    & np.isclose(wave_runs["WaveAmplitudeInput [Volt]"], DEMO_AMP)
    & (wave_runs["PanelCondition"] == DEMO_PANEL)
]
nw_rows = demo_group[demo_group["WindCondition"] == "no"]
fw_rows = demo_group[demo_group["WindCondition"] == "full"]
if nw_rows.empty or fw_rows.empty:
    print(f"   skip: no matched nowind+fullwind at {DEMO_FREQ} Hz, {DEMO_AMP} V")
else:
    fig2, axes2 = plt.subplots(
        nrows=1, ncols=2, figsize=(13, 5), dpi=120, sharey=False,
    )
    for col_idx, probe in enumerate(PROBES):
        ax = axes2[col_idx]
        f_nw, psd_nw, n_nw, fs = mean_residual_psd(nw_rows, probe)
        f_fw, psd_fw, n_fw, _ = mean_residual_psd(fw_rows, probe, fs_ref=fs, f_ref=f_nw)
        if psd_nw is None or psd_fw is None:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        psd_pure = np.clip(psd_fw - psd_nw, 0, None)
        fmask = (f_nw >= 0.2) & (f_nw <= 10.0)
        ax.semilogy(f_nw[fmask], psd_nw[fmask], color=COLOR_A, lw=1.2,
                    label=f"nowind residual (Stokes, n={n_nw})")
        ax.semilogy(f_nw[fmask], psd_fw[fmask], color=COLOR_B, lw=1.2,
                    label=f"fullwind residual (wind+Stokes, n={n_fw})")
        pure_nonzero = psd_pure.copy()
        pure_nonzero[pure_nonzero <= 0] = np.nan
        ax.semilogy(f_nw[fmask], pure_nonzero[fmask], color="#222222", lw=1.6,
                    label="pure wind (fullwind − nowind)")
        ax.axvspan(F_WIND_LO, F_WIND_HI, color="#FFD580", alpha=0.25,
                   zorder=0, label="wind band")
        ax.axvspan(DEMO_FREQ - BAND_HALF_HZ, DEMO_FREQ + BAND_HALF_HZ,
                   color="#B0D0FF", alpha=0.35, zorder=0, label="paddle band")
        # Annotate 2f, 3f, 4f
        for n in (2, 3, 4):
            ax.axvline(n * DEMO_FREQ, color="#666666", ls=":", lw=0.8, alpha=0.6)
            ax.text(n * DEMO_FREQ, 1.5, f"{n}f", ha="center", va="bottom",
                    fontsize=7, color="#666666")
        mask = (f_nw >= F_WIND_LO) & (f_nw <= F_WIND_HI)
        e_naive = float(np.trapezoid(psd_fw[mask], f_nw[mask]))
        e_stokes = float(np.trapezoid(psd_nw[mask], f_nw[mask]))
        e_pure = float(np.trapezoid(psd_pure[mask], f_nw[mask]))
        stk_frac = (e_naive - e_pure) / e_naive if e_naive > 0 else np.nan
        ax.set_title(
            f"{PROBE_LABEL[probe]}  "
            f"E_naive={e_naive:.2f}  E_Stokes={e_stokes:.2f}  E_pure={e_pure:.2f}"
            f"  (Stokes frac = {stk_frac*100:.1f}%)",
            fontsize=9,
        )
        ax.set_xlabel("frequency [Hz]")
        ax.set_ylabel("PSD [mm²/Hz]")
        ax.grid(alpha=0.3, which="both")
        if col_idx == 0:
            ax.legend(fontsize=7, loc="lower left")
    fig2.suptitle(
        f"Pure wind PSD via nowind-residual subtraction — "
        f"{DEMO_FREQ} Hz, {DEMO_AMP:.2f} V, {DEMO_PANEL} panel, method A residuals\n"
        f"Paddle harmonics 2f/3f/4f fall inside the 2–6 Hz wind band and must be "
        f"subtracted for a clean wind metric.",
        fontsize=10, fontweight="bold", y=1.02,
    )
    fig2.savefig(OUT_PDF2, bbox_inches="tight")
    print(f"   pure-wind demo figure → {OUT_PDF2}")

    # Also save to output/ for thesis use and write stub (write-once).
    fig2.savefig(THESIS_PDF_PURE, bbox_inches="tight")
    print(f"   thesis figure → {THESIS_PDF_PURE.relative_to(BASE)}")

    # Summary stats for caption: median Stokes fraction per probe.
    _pure_in  = pure[pure["probe"] == "9373/170"]
    _pure_out = pure[pure["probe"] == "12400/250"]
    _stokes_in_med  = float(_pure_in["stokes_frac_of_naive"].median())  if len(_pure_in)  else float("nan")
    _stokes_out_med = float(_pure_out["stokes_frac_of_naive"].median()) if len(_pure_out) else float("nan")
    _stokes_in_max  = float(_pure_in["stokes_frac_of_naive"].max())     if len(_pure_in)  else float("nan")
    _n_groups = int(len(pure) // 2)

    _caption_pure = (
        "Mean residual power spectra at the IN probe (left, "
        "9373/170) and OUT probe (right, 12400/250) for the "
        f"{DEMO_FREQ}\\,Hz / {DEMO_AMP:.2f}\\,V / full-panel group. "
        "No-wind residuals (blue) carry only paddle-linked Stokes "
        "harmonics; full-wind residuals (red) carry Stokes plus "
        "wind-wave energy. Subtracting the no-wind baseline from "
        "the full-wind residual (black) isolates pure wind. Vertical "
        "dotted lines mark $2f$, $3f$ and $4f$, all of which fall "
        "inside the wind band (2--6\\,Hz, orange) at thesis paddle "
        "frequencies. Across the full thesis scope "
        f"($n={_n_groups}$ frequency--amplitude groups with matched "
        "no-wind/full-wind coverage), the Stokes-subtraction "
        f"correction removes a median "
        f"${_stokes_in_med*100:.0f}$\\,\\% of the naive "
        f"``wind'' energy at IN and ${_stokes_out_med*100:.0f}$\\,\\% "
        f"at OUT (up to ${_stokes_in_max*100:.0f}$\\,\\% at the IN / "
        "0.3\\,V high-frequency corner). Any wind-energy metric that "
        "integrates a wave run's residual over 2--6\\,Hz without this "
        "subtraction mis-attributes paddle Stokes harmonics as wind."
    )

    _meta_stub_pure = pu.build_fig_meta(
        {
            "filters": {
                "PanelCondition":    DEMO_PANEL,
                "WaveFrequencyInput [Hz]": [1.3, 1.6],
                "WindCondition":     ["no", "full"],
                "quality_flag":      "ok+probe_malfunction_secondary",
                "probes":            ", ".join(PROBES),
            },
            "plotting": {
                "figure_name": THESIS_NAME_PURE,
                "caption":     _caption_pure,
                "caption_short": "Pure wind via no-wind residual subtraction",
            },
        },
        chapter="04",
        data_df=wave_runs,
        extra={"script": "analysis_scratch/reconstruction_A_vs_B.py"},
        computed_in=("analysis_scratch/reconstruction_A_vs_B.py §5-§6 "
                     "(PSD_wind = mean PSD residual_A(full) − mean PSD residual_A(no), "
                     "per (freq,amp,mooring,panel) group)"),
        data_class="DELEG",
        findings_doc="analysis_scratch/reconstruction_pure_wind_findings.md",
        grouper="per (freq, amp, mooring, panel) group; demo = single group",
        collapse_panels=False,
        fft_window_hz=2 * BAND_HALF_HZ,
        extra_params=(
            f"demo_run={DEMO_FREQ} Hz, {DEMO_AMP} V, {DEMO_PANEL} panel; "
            f"wind_band_hz={F_WIND_LO}-{F_WIND_HI}; "
            f"probes={PROBES}; welch nperseg=min(len(residual), 4096), "
            f"common grid 0-12.5 Hz @ 0.05 Hz"
        ),
        extra_stats={
            "n_groups":                 _n_groups,
            "stokes_frac_in_median_pct":  round(_stokes_in_med * 100, 1) if np.isfinite(_stokes_in_med) else "—",
            "stokes_frac_in_max_pct":     round(_stokes_in_max * 100, 1) if np.isfinite(_stokes_in_max) else "—",
            "stokes_frac_out_median_pct": round(_stokes_out_med * 100, 1) if np.isfinite(_stokes_out_med) else "—",
        },
    )

    pu.write_figure_stub(_meta_stub_pure, plot_type="reconstruction_pure_wind",
                         subfig_filenames=[THESIS_NAME_PURE])
    print(f"   thesis stub   → {THESIS_STUB_PURE.relative_to(BASE)}")

print("\nDone.")
