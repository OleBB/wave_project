"""
ARCHIVED 2026-04-15 — superseded by reconstruct_all3.py.
This was the first prototype: hard-cap + LSQ sine fill, run2 only, no edge buffer,
no derivative-guided fill. Kept for reference on how the approach evolved.

Option D reconstruction test — run2 at 1.6 Hz nowind, 0.3V.

Approach:
  1. Compute the run's own positive crest level (P97.5 of positive samples)
     as the reference amplitude — uncontaminated by the negative false troughs.
  2. Hard-cap: any sample below -(crest_ref * CAP_FACTOR) → NaN.
  3. PCHIP interpolate the new NaN gaps.
  4. Clip the PCHIP output to [floor, ceil] = the observed min/max of the
     surviving (non-capped) clean signal — so the reconstruction never exceeds
     what physically happened in that run.
  5. Compare FFT amplitude before and after reconstruction.

Run:
    conda run -n draumkvedet python analysis_scratch/reconstruct_run2_test.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

sys.path.insert(0, str(Path(__file__).parent.parent))
from wavescripts.improved_data_loader import load_analysis_data, load_processed_dfs

BASE = Path(__file__).parent.parent
FS = 250.0
POS = "9373/170"
OUT_POS = "12400/250"
FREQ = 1.6
CAP_FACTOR = 1.5   # cap troughs at -(crest_ref * CAP_FACTOR); crests at +(crest_ref * CAP_FACTOR)
# Reconstruction method: after capping, fill with physics-based LSQ sine fit
# rather than PCHIP. PCHIP fails when troughs are dense — it gets stuck at the
# cap boundary. The sine fit is globally informed by all clean samples.

# ── Load data ─────────────────────────────────────────────────────────────────
PROC_DIR = BASE / "waveprocessed" / (
    "PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"
)

combined_meta, _, _, _ = load_analysis_data(PROC_DIR, load_processed=False)
processed_dfs = load_processed_dfs(PROC_DIR)

# ── Identify the run ──────────────────────────────────────────────────────────
target_name = "fullpanel-nowind-amp0300-freq1600-per40-depth580-mstop30-run2.csv"
key = next(k for k in processed_dfs if target_name in k)
df = processed_dfs[key]

# Analysis window from meta
meta_row = combined_meta[combined_meta["path"] == key].iloc[0]
start = int(meta_row[f"Computed Probe {POS} start"])
end   = int(meta_row[f"Computed Probe {POS} end"])
freq  = float(meta_row["WaveFrequencyInput [Hz]"])

full_t = np.arange(len(df)) / FS
win_t  = np.arange(end - start) / FS   # time within window

# Raw and interp signals in analysis window
sig_raw    = df[f"Probe {POS}"].values[start:end]
sig_interp = df[f"eta_{POS}_interp"].values[start:end]
out_interp = df[f"eta_{OUT_POS}_interp"].values[start:end]

# ── Step 1: estimate crest reference from positive samples ────────────────────
positive_mask = sig_interp > 0
if positive_mask.sum() > 10:
    crest_ref = np.nanpercentile(sig_interp[positive_mask], 97.5)
else:
    crest_ref = np.nanpercentile(np.abs(sig_interp), 97.5)

trough_floor = -(crest_ref * CAP_FACTOR)
crest_ceil   =  (crest_ref * CAP_FACTOR)
print(f"Crest reference (P97.5 positive): {crest_ref:.2f} mm")
print(f"Hard cap range: [{trough_floor:.2f}, {crest_ceil:.2f}] mm")

# ── Step 2: mark samples outside cap as NaN (on top of existing NaN from pipeline) ──
sig_capped = sig_interp.copy()
newly_capped = (sig_capped < trough_floor) | (sig_capped > crest_ceil)
sig_capped[newly_capped] = np.nan

print(f"Newly capped samples: {newly_capped.sum()} "
      f"({100 * newly_capped.sum() / len(sig_capped):.1f}% of window)")

# ── Step 3: LSQ sine fit to clean samples → use as reconstruction ─────────────
# PCHIP fails when troughs are dense (gets stuck at cap boundary between troughs).
# A global LSQ sine fit at the known frequency is physically motivated and uses
# ALL clean samples simultaneously. The fitted wave is guaranteed to have the
# correct shape; only its amplitude and phase are free parameters.
nan_mask  = np.isnan(sig_capped)
clean_idx = np.where(~nan_mask)[0]

omega = 2 * np.pi * FREQ

if len(clean_idx) < 4:
    print("ERROR: too few valid samples for reconstruction")
    sig_reconstructed = sig_capped.copy()
    A_fit, B_fit = 0.0, 0.0
else:
    Xc = np.column_stack([np.cos(omega * win_t[clean_idx]),
                          np.sin(omega * win_t[clean_idx])])
    yc = sig_capped[clean_idx] - np.nanmean(sig_capped[clean_idx])
    ab, _, _, _ = np.linalg.lstsq(Xc, yc, rcond=None)
    A_fit, B_fit = ab
    amp_lsq = np.sqrt(A_fit**2 + B_fit**2)
    mean_fit = np.nanmean(sig_capped[clean_idx])
    print(f"LSQ sine fit:  amplitude={amp_lsq:.2f} mm  (clean mean={mean_fit:.2f} mm)")

    # Build full fitted sine
    sine_fit = mean_fit + A_fit * np.cos(omega * win_t) + B_fit * np.sin(omega * win_t)

    # Fill NaN/capped regions with the fitted sine; keep clean samples as-is
    sig_reconstructed = sig_capped.copy()
    fill_idx = np.where(nan_mask)[0]
    sig_reconstructed[fill_idx] = sine_fit[fill_idx]

# ── Step 4: constrain filled sections to [min_clean, max_clean] ──────────────
# The user's constraint: reconstruction cannot exceed the max/min actually
# observed in the clean parts of this run (no invented values).
clean_vals = sig_capped[~np.isnan(sig_capped)]
floor_val  = np.nanmin(clean_vals)
ceil_val   = np.nanmax(clean_vals)
# Only clip the filled (reconstructed) regions, not the original clean samples
sig_reconstructed[fill_idx] = np.clip(sig_reconstructed[fill_idx], floor_val, ceil_val)

print(f"Reconstruction clamped to [{floor_val:.2f}, {ceil_val:.2f}] mm  "
      f"(clean min/max of this run)")

# ── Step 5: FFT amplitude comparison ─────────────────────────────────────────
def fft_amp_at_freq(sig, fs, target_freq, window_hz=0.1):
    n = len(sig)
    s = sig - np.nanmean(sig)
    s = np.nan_to_num(s)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    spectrum = np.abs(np.fft.rfft(s))
    mask = np.abs(freqs - target_freq) <= window_hz / 2
    if mask.sum() == 0:
        return np.nan
    idx = np.argmin(np.abs(freqs[mask] - target_freq))
    return spectrum[mask][idx] * 2 / n

fft_original    = fft_amp_at_freq(sig_interp, FS, FREQ)
fft_reconstructed = fft_amp_at_freq(sig_reconstructed, FS, FREQ)
fft_out = fft_amp_at_freq(out_interp, FS, FREQ)
meta_fft = float(meta_row.get(f"Probe {POS} Amplitude (FFT)", np.nan))

print(f"\nFFT IN (pipeline, PCHIP):        {meta_fft:.2f} mm")
print(f"FFT IN (reconstructed):           {fft_reconstructed:.2f} mm")
print(f"FFT OUT:                          {fft_out:.2f} mm")
print(f"OUT/IN original:  {fft_out/fft_original:.3f}")
print(f"OUT/IN reconstructed: {fft_out/fft_reconstructed:.3f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig.suptitle(
    f"Option D reconstruction — {target_name}\n"
    f"Cap: ±{CAP_FACTOR}× crest_ref = [{trough_floor:.1f}, {crest_ceil:.1f}] mm   "
    f"Constrained to [{floor_val:.1f}, {ceil_val:.1f}] mm",
    fontsize=10,
)

# full time axis
full_sig = df[f"eta_{POS}_interp"].values
full_sig_raw = df[f"Probe {POS}"].values

# ── Panel 1: full run, current pipeline ──────────────────────────────────────
ax = axes[0]
t_full = np.arange(len(df)) / FS
ax.plot(t_full, full_sig, color="steelblue", lw=0.6, label=f"eta_{POS}_interp (current)")
ax.axhline(trough_floor, color="orange", lw=0.8, ls="--", label=f"cap floor {trough_floor:.1f} mm")
ax.axhline(crest_ceil,   color="orange", lw=0.8, ls="--", label=f"cap ceil {crest_ceil:.1f} mm")
ax.axvline(start / FS, color="green", lw=1.2, ls="--")
ax.axvline(end   / FS, color="red",   lw=1.2, ls="--")
ax.axvspan(start / FS, end / FS, alpha=0.07, color="green")
ax.set_ylabel("η (mm)")
ax.set_title(f"Full run — current pipeline   IN FFT = {meta_fft:.2f} mm", fontsize=9)
ax.legend(fontsize=7, loc="upper right")
ax.grid(True, alpha=0.3)

# ── Panel 2: analysis window — before vs newly-capped ────────────────────────
ax = axes[1]
t_win = win_t + start / FS

# Show original in blue, mark newly-capped regions in red
ax.plot(t_win, sig_interp, color="steelblue", lw=0.8, label="before reconstruction (PCHIP)", zorder=2)

# Highlight newly-capped spans in red
in_cap = False
cap_start = None
for i, capped in enumerate(newly_capped):
    if capped and not in_cap:
        cap_start = i
        in_cap = True
    elif not capped and in_cap:
        ax.axvspan(t_win[cap_start], t_win[i - 1], color="red", alpha=0.25, zorder=1)
        in_cap = False
if in_cap:
    ax.axvspan(t_win[cap_start], t_win[-1], color="red", alpha=0.25, zorder=1)

# Red line for the reconstructed signal over capped regions
recon_display = np.where(newly_capped | np.isnan(sig_interp), sig_reconstructed, np.nan)
ax.plot(t_win, recon_display, color="red", lw=1.0, label="reconstructed (Option D)", zorder=3)

ax.axhline(trough_floor, color="orange", lw=0.8, ls="--")
ax.axhline(crest_ceil,   color="orange", lw=0.8, ls="--")
ax.axhline(floor_val, color="purple", lw=0.8, ls=":", label=f"clamp [{floor_val:.1f}, {ceil_val:.1f}] mm")
# Show the fitted sine across the whole window so you can judge the fit quality
ax.plot(t_win, sine_fit, color="gray", lw=0.7, ls="--", alpha=0.6, label=f"LSQ sine fit (amp={amp_lsq:.1f} mm)", zorder=1)
ax.set_ylabel("η (mm)")
ax.set_title("Analysis window — newly-capped sections (red shaded), LSQ sine fit (gray)", fontsize=9)
ax.legend(fontsize=7, loc="upper right")
ax.grid(True, alpha=0.3)

# ── Panel 3: analysis window — reconstructed final ───────────────────────────
ax = axes[2]

# Blue = clean carry-through, red = reconstructed sections
recon_only = np.where(newly_capped | np.isnan(sig_interp), sig_reconstructed, np.nan)
orig_only  = np.where(newly_capped | np.isnan(sig_interp), np.nan, sig_interp)

ax.plot(t_win, orig_only,  color="steelblue", lw=0.8, label="original clean samples", zorder=2)
ax.plot(t_win, recon_only, color="red",       lw=1.0, label="reconstructed (Option D)", zorder=3)
ax.set_ylabel("η (mm)")
ax.set_title(
    f"Reconstructed signal   IN FFT = {fft_reconstructed:.2f} mm   "
    f"OUT/IN = {fft_out/fft_reconstructed:.3f}  (was {fft_out/fft_original:.3f})",
    fontsize=9,
)
ax.legend(fontsize=7, loc="upper right")
ax.grid(True, alpha=0.3)
ax.set_xlabel("Time (s)")

plt.tight_layout()
out_path = Path(__file__).parent / "reconstruct_run2_test.png"
plt.savefig(out_path, dpi=150)
print(f"\nSaved: {out_path}")
