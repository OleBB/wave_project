"""
Visual comparison of reconstruction variants for run2 at 1.6 Hz nowind.

Variants:
  A — fill only existing pipeline NaN (VELCLIP/MONOCLIP) with LSQ sine.
      No new outlier detection. Shows what the "minimal change" baseline looks like.

  B — iterative sigma-clip LSQ: fit → residuals → flag |residual| > N·σ → refit.
      Fully self-adaptive. No hard cap, no prior knowledge of amplitude.

  C — hard cap (crest_ref × factor) + LSQ fill. Shown for comparison.

All panels: blue = clean signal, red = reconstructed sections.
Gray dashed = fitted sine across the full window.

Run:
    conda run -n draumkvedet python analysis_scratch/reconstruct_variants.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from wavescripts.improved_data_loader import load_analysis_data, load_processed_dfs

BASE = Path(__file__).parent.parent
FS   = 250.0
POS  = "9373/170"
FREQ = 1.6

PROC_DIR = BASE / "waveprocessed" / (
    "PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"
)

combined_meta, _, _, _ = load_analysis_data(PROC_DIR, load_processed=False)
processed_dfs = load_processed_dfs(PROC_DIR)

target_name = "fullpanel-nowind-amp0300-freq1600-per40-depth580-mstop30-run2.csv"
key = next(k for k in processed_dfs if target_name in k)
df  = processed_dfs[key]

meta_row = combined_meta[combined_meta["path"] == key].iloc[0]
start    = int(meta_row[f"Computed Probe {POS} start"])
end      = int(meta_row[f"Computed Probe {POS} end"])

win_t      = np.arange(end - start) / FS
omega      = 2 * np.pi * FREQ
sig_raw    = df[f"Probe {POS}"].values[start:end]
sig_interp = df[f"eta_{POS}_interp"].values[start:end]
meta_fft   = float(meta_row.get(f"Probe {POS} Amplitude (FFT)", np.nan))


# ── Helper: LSQ sine fit + fill ───────────────────────────────────────────────
def lsq_sine_fill(sig, win_t, omega, nan_mask):
    """
    Fit A·cos(ωt) + B·sin(ωt) to samples where nan_mask is False.
    Return (reconstructed_signal, amplitude, sine_curve).
    Reconstructed = original clean samples + sine fit in NaN gaps.
    """
    clean_idx = np.where(~nan_mask)[0]
    if len(clean_idx) < 4:
        return sig.copy(), np.nan, np.full_like(sig, np.nan)

    yc = sig[clean_idx] - np.nanmean(sig[clean_idx])
    Xc = np.column_stack([np.cos(omega * win_t[clean_idx]),
                          np.sin(omega * win_t[clean_idx])])
    ab, _, _, _ = np.linalg.lstsq(Xc, yc, rcond=None)
    amp  = np.sqrt(ab[0]**2 + ab[1]**2)
    mean = np.nanmean(sig[clean_idx])
    sine = mean + ab[0] * np.cos(omega * win_t) + ab[1] * np.sin(omega * win_t)

    out = sig.copy()
    out[nan_mask] = sine[nan_mask]
    return out, amp, sine


def fft_amp(sig, fs, f0, window_hz=0.1):
    n = len(sig)
    s = sig - np.nanmean(sig)
    s = np.nan_to_num(s)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    spec  = np.abs(np.fft.rfft(s))
    mask  = np.abs(freqs - f0) <= window_hz / 2
    if not mask.any():
        return np.nan
    return spec[mask][np.argmin(np.abs(freqs[mask] - f0))] * 2 / n


# ── Variant A: fill only existing pipeline NaN ────────────────────────────────
nan_pipeline = np.isnan(sig_raw)   # NaN = what VELCLIP/MONOCLIP already removed
sig_A, amp_A, sine_A = lsq_sine_fill(sig_interp, win_t, omega, nan_pipeline)
# sig_interp already has PCHIP in those gaps; re-do with sine in same gaps
sig_A_clean = sig_interp.copy()
sig_A_clean[nan_pipeline] = sine_A[nan_pipeline]
sig_A, amp_A, sine_A = lsq_sine_fill(sig_interp, win_t, omega, nan_pipeline)
fft_A = fft_amp(sig_A, FS, FREQ)
n_new_A = nan_pipeline.sum()

# ── Variant B: iterative sigma-clip LSQ ───────────────────────────────────────
SIGMA_THRESH = 3.0
MAX_ITER = 10

nan_B = np.isnan(sig_raw)   # start from existing pipeline NaN
sig_B_work = sig_interp.copy()

for iteration in range(MAX_ITER):
    clean_idx = np.where(~nan_B)[0]
    if len(clean_idx) < 4:
        break
    yc = sig_B_work[clean_idx] - np.nanmean(sig_B_work[clean_idx])
    Xc = np.column_stack([np.cos(omega * win_t[clean_idx]),
                          np.sin(omega * win_t[clean_idx])])
    ab, _, _, _ = np.linalg.lstsq(Xc, yc, rcond=None)
    mean_fit = np.nanmean(sig_B_work[clean_idx])
    sine_B   = mean_fit + ab[0] * np.cos(omega * win_t) + ab[1] * np.sin(omega * win_t)

    residuals = sig_B_work - sine_B
    res_clean = residuals[~nan_B]
    sigma     = np.std(res_clean)
    new_outliers = (~nan_B) & (np.abs(residuals) > SIGMA_THRESH * sigma)
    n_new = new_outliers.sum()
    nan_B |= new_outliers
    if n_new == 0:
        break

amp_B_val = np.sqrt(ab[0]**2 + ab[1]**2)

# Post-convergence amplitude gate: now that we have a clean fitted amplitude,
# flag any surviving sample that is physically implausible — more than
# AMP_GATE_FACTOR × fitted_amplitude below (or above) the fitted sine.
# This catches trough flanks whose local residual was within 3σ but whose
# absolute level is impossible for a wave of this amplitude.
AMP_GATE_FACTOR = 1.3
mean_fit_final = np.nanmean(sig_B_work[~nan_B])
gate_floor = mean_fit_final - amp_B_val * AMP_GATE_FACTOR
gate_ceil  = mean_fit_final + amp_B_val * AMP_GATE_FACTOR
amp_gate_outliers = (~nan_B) & ((sig_interp < gate_floor) | (sig_interp > gate_ceil))
nan_B |= amp_gate_outliers
n_amp_gate = amp_gate_outliers.sum()
print(f"Variant B post-convergence amp gate [{gate_floor:.1f}, {gate_ceil:.1f}] mm: "
      f"{n_amp_gate} additional samples flagged")

sig_B, amp_B, sine_B_final = lsq_sine_fill(sig_interp, win_t, omega, nan_B)
fft_B = fft_amp(sig_B, FS, FREQ)
n_new_B = (nan_B & ~nan_pipeline).sum()   # newly flagged (sigma-clip + amp gate)

print(f"Variant B: {iteration+1} iterations, {n_new_B} newly flagged samples "
      f"({100*n_new_B/len(nan_B):.1f}%), amp={amp_B:.2f} mm")

# ── Variant C: hard cap + LSQ (baseline from previous script) ────────────────
CAP_FACTOR = 1.5
positive_mask = sig_interp > 0
crest_ref = np.nanpercentile(sig_interp[positive_mask], 97.5)
trough_floor = -(crest_ref * CAP_FACTOR)
crest_ceil   =  (crest_ref * CAP_FACTOR)

sig_capped_C = sig_interp.copy()
newly_capped_C = (~np.isnan(sig_interp)) & ((sig_interp < trough_floor) | (sig_interp > crest_ceil))
sig_capped_C[newly_capped_C] = np.nan
nan_C = np.isnan(sig_capped_C)

sig_C, amp_C, sine_C = lsq_sine_fill(sig_capped_C, win_t, omega, nan_C)
# clamp filled regions to clean min/max
clean_C = sig_capped_C[~nan_C]
sig_C[nan_C] = np.clip(sig_C[nan_C], np.nanmin(clean_C), np.nanmax(clean_C))
fft_C = fft_amp(sig_C, FS, FREQ)
n_new_C = newly_capped_C.sum()

# ── Plotting ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True, sharey=True)
fig.suptitle(
    f"Reconstruction variants — {target_name}\n"
    f"Analysis window only.  Pipeline FFT = {meta_fft:.1f} mm  "
    f"(run3/run4 reference: ~20 mm)",
    fontsize=10,
)

t_win_abs = win_t + start / FS   # absolute time for x-axis


def plot_variant(ax, sig_orig, sig_recon, nan_mask, sine, amp, fft_val,
                 title, n_new):
    """Plot one variant. Blue = clean, red = reconstructed, gray = sine fit."""
    # Separate clean and reconstructed
    orig_only  = np.where(nan_mask, np.nan, sig_orig)
    recon_only = np.where(nan_mask, sig_recon, np.nan)

    ax.plot(t_win_abs, orig_only,  color="steelblue", lw=0.7, label="clean signal")
    ax.plot(t_win_abs, recon_only, color="red",       lw=1.0, label="reconstructed", zorder=3)
    ax.plot(t_win_abs, sine,       color="gray",      lw=0.7, ls="--", alpha=0.6,
            label=f"LSQ sine ({amp:.1f} mm)")

    ax.set_title(
        f"{title}   |   {n_new} new samples flagged  "
        f"→  FFT IN = {fft_val:.2f} mm   OUT/IN ≈ {11.03/fft_val:.3f} (using pipeline OUT=11.03 mm)",
        fontsize=8,
    )
    ax.set_ylabel("η (mm)")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)


# Row 0: current pipeline (PCHIP) for reference
orig_clean = np.where(nan_pipeline, np.nan, sig_interp)
orig_recon = np.where(nan_pipeline, sig_interp, np.nan)
axes[0].plot(t_win_abs, orig_clean, color="steelblue", lw=0.7, label="clean signal")
axes[0].plot(t_win_abs, orig_recon, color="red",       lw=1.0, label="PCHIP fill (current pipeline)")
axes[0].set_title(
    f"Current pipeline (PCHIP fill of VELCLIP/MONOCLIP NaN)   |   "
    f"FFT IN = {meta_fft:.2f} mm   OUT/IN ≈ {11.03/meta_fft:.3f}",
    fontsize=8,
)
axes[0].set_ylabel("η (mm)")
axes[0].legend(fontsize=7, loc="upper right")
axes[0].grid(True, alpha=0.3)

# Row 1: Variant A
plot_variant(axes[1], sig_interp, sig_A, nan_pipeline, sine_A, amp_A, fft_A,
             "Variant A — LSQ sine fill of existing pipeline NaN (no new detection)",
             n_new_A)

# Row 2: Variant B
plot_variant(axes[2], sig_interp, sig_B, nan_B, sine_B_final, amp_B, fft_B,
             f"Variant B — sigma-clip LSQ (σ={SIGMA_THRESH}×) + amp gate (±{AMP_GATE_FACTOR}× fitted amp)",
             n_new_B)

# Row 3: Variant C
plot_variant(axes[3], sig_interp, sig_C, nan_C, sine_C, amp_C, fft_C,
             f"Variant C — hard cap (±{CAP_FACTOR}× crest_ref = ±{crest_ref*CAP_FACTOR:.0f} mm) + LSQ fill",
             n_new_C)

axes[-1].set_xlabel("Time (s)")
plt.tight_layout()

out_path = Path(__file__).parent / "reconstruct_variants.png"
plt.savefig(out_path, dpi=150)
print(f"Saved: {out_path}")
