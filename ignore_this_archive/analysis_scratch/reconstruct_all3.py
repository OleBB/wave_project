"""
Apply Variant B reconstruction (iterative sigma-clip LSQ + amplitude gate)
to run2, run3, run4 and show before/after for each.

Layout: 3 rows (one per run), 2 columns
  Left:  analysis window — blue=kept, red=reconstructed sections, gray=LSQ sine
  Right: period overlay — thin colored=before, thin gray=after, black=medians

Run:
    conda run -n draumkvedet python analysis_scratch/reconstruct_all3.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from wavescripts.improved_data_loader import load_analysis_data, load_processed_dfs

BASE     = Path(__file__).parent.parent
FS       = 250.0
POS      = "9373/170"
OUT_POS  = "12400/250"
FREQ     = 1.6
PROC_DIR = BASE / "waveprocessed" / (
    "PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"
)

# Reconstruction parameters
SIGMA_THRESH    = 3.0
MAX_ITER        = 10
AMP_GATE_FACTOR = 1.3

# Period classification thresholds
SEVERE_THRESH = -40.0
INDENT_PHASE  = (0.55, 0.85)
INDENT_THRESH = -12.0
N_PHASE       = 200

combined_meta, _, _, _ = load_analysis_data(PROC_DIR, load_processed=False)
processed_dfs = load_processed_dfs(PROC_DIR)

RUNS = [
    "fullpanel-nowind-amp0300-freq1600-per40-depth580-mstop30-run2.csv",
    "fullpanel-nowind-amp0300-freq1600-per40-depth580-mstop30-run3.csv",
    "fullpanel-nowind-amp0300-freq1600-per40-depth580-mstop30-run4.csv",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def fft_amp_at(sig, fs=FS, f0=FREQ, window_hz=0.05):
    n = len(sig)
    s = sig - np.nanmean(sig)
    s = np.nan_to_num(s)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    spec  = np.abs(np.fft.rfft(s))
    mask  = np.abs(freqs - f0) <= window_hz
    if not mask.any():
        return np.nan
    return spec[mask][np.argmin(np.abs(freqs[mask] - f0))] * 2 / n


def reconstruct_variant_b(sig_raw, sig_interp, win_t, omega):
    """Iterative sigma-clip LSQ + amplitude gate. Returns (nan_mask, sig_recon, sine, amp)."""
    nan_mask = np.isnan(sig_raw)

    for _ in range(MAX_ITER):
        clean_idx = np.where(~nan_mask)[0]
        if len(clean_idx) < 4:
            break
        yc = sig_interp[clean_idx] - np.nanmean(sig_interp[clean_idx])
        Xc = np.column_stack([np.cos(omega * win_t[clean_idx]),
                              np.sin(omega * win_t[clean_idx])])
        ab, _, _, _ = np.linalg.lstsq(Xc, yc, rcond=None)
        mean_fit = np.nanmean(sig_interp[clean_idx])
        sine     = mean_fit + ab[0]*np.cos(omega*win_t) + ab[1]*np.sin(omega*win_t)
        residuals   = sig_interp - sine
        res_clean   = residuals[~nan_mask]
        sigma       = np.std(res_clean)
        new_out     = (~nan_mask) & (np.abs(residuals) > SIGMA_THRESH * sigma)
        nan_mask   |= new_out
        if new_out.sum() == 0:
            break

    amp = np.sqrt(ab[0]**2 + ab[1]**2)

    # Amplitude gate pass
    mean_final = np.nanmean(sig_interp[~nan_mask])
    gate_floor = mean_final - amp * AMP_GATE_FACTOR
    gate_ceil  = mean_final + amp * AMP_GATE_FACTOR
    gate_out   = (~nan_mask) & ((sig_interp < gate_floor) | (sig_interp > gate_ceil))
    nan_mask  |= gate_out

    # Final refit with all flagged samples removed
    clean_idx = np.where(~nan_mask)[0]
    yc = sig_interp[clean_idx] - np.nanmean(sig_interp[clean_idx])
    Xc = np.column_stack([np.cos(omega * win_t[clean_idx]),
                          np.sin(omega * win_t[clean_idx])])
    ab, _, _, _ = np.linalg.lstsq(Xc, yc, rcond=None)
    mean_fit = np.nanmean(sig_interp[clean_idx])
    sine     = mean_fit + ab[0]*np.cos(omega*win_t) + ab[1]*np.sin(omega*win_t)
    amp      = np.sqrt(ab[0]**2 + ab[1]**2)

    # Edge buffer: dilate the nan_mask by EDGE_BUF samples on each side.
    # The raw signal at gap boundaries is unreliable (probe settling after a
    # fault event) — these 1-3 samples cause cliff artifacts at re-entry.
    EDGE_BUF = 4
    dilated = nan_mask.copy()
    edges = np.where(np.diff(nan_mask.astype(int)) != 0)[0]
    for e in edges:
        lo = max(0, e - EDGE_BUF + 1)
        hi = min(len(nan_mask), e + EDGE_BUF + 1)
        dilated[lo:hi] = True
    nan_mask = dilated

    # Derivative-guided fill: for each contiguous gap, blend forward and
    # backward extrapolations anchored to the real signal at each boundary.
    #
    #   fwd[k] = sig[i_L] + (sine[k] - sine[i_L])   <- left anchor + sine shape
    #   bwd[k] = sig[i_R] + (sine[k] - sine[i_R])   <- right anchor + sine shape
    #   fill[k] = w_R * fwd[k] + w_L * bwd[k]       <- distance-weighted blend
    #
    # Guarantees zero cliff at both gap edges. Amplitude at boundary comes from
    # the real measurement, not the global LSQ — preserves period-to-period
    # amplitude variation. The sine only provides the rate-of-change (shape).
    sig_recon = sig_interp.copy()
    n = len(nan_mask)
    # Find contiguous gap runs
    in_gap = False
    gap_start = None
    gaps = []
    for i in range(n):
        if nan_mask[i] and not in_gap:
            gap_start = i
            in_gap = True
        elif not nan_mask[i] and in_gap:
            gaps.append((gap_start, i - 1))
            in_gap = False
    if in_gap:
        gaps.append((gap_start, n - 1))

    for g_start, g_end in gaps:
        i_L = g_start - 1  # last clean sample before gap
        i_R = g_end + 1    # first clean sample after gap
        gap_len = g_end - g_start + 1

        has_left  = i_L >= 0 and not nan_mask[i_L]
        has_right = i_R < n  and not nan_mask[i_R]

        if has_left and has_right:
            # Blend fwd and bwd anchored fills
            for k in range(g_start, g_end + 1):
                w_R = (k - g_start) / (gap_len + 1)   # weight toward right boundary
                w_L = 1.0 - w_R
                fwd = sig_interp[i_L] + (sine[k] - sine[i_L])
                bwd = sig_interp[i_R] + (sine[k] - sine[i_R])
                sig_recon[k] = w_L * fwd + w_R * bwd
        elif has_left:
            for k in range(g_start, g_end + 1):
                sig_recon[k] = sig_interp[i_L] + (sine[k] - sine[i_L])
        elif has_right:
            for k in range(g_start, g_end + 1):
                sig_recon[k] = sig_interp[i_R] + (sine[k] - sine[i_R])
        else:
            sig_recon[g_start:g_end + 1] = sine[g_start:g_end + 1]

    return nan_mask, sig_recon, sine, amp


def find_upcrossings(sig):
    mean = np.nanmean(sig)
    s = sig - mean
    out = []
    for i in range(len(s) - 1):
        if np.isnan(s[i]) or np.isnan(s[i+1]):
            continue
        if s[i] < 0 and s[i+1] >= 0:
            frac = -s[i] / (s[i+1] - s[i])
            out.append(i + frac)
    return np.array(out)


def extract_periods(sig, upcrossings):
    phase_grid = np.linspace(0, 1, N_PHASE, endpoint=False)
    periods = []
    for i in range(len(upcrossings) - 1):
        i0 = int(np.floor(upcrossings[i]))
        i1 = int(np.ceil(upcrossings[i + 1]))
        if i1 >= len(sig):
            break
        chunk = sig[i0:i1]
        if len(chunk) < 10:
            continue
        t_c = np.linspace(0, 1, len(chunk), endpoint=False)
        periods.append(np.interp(phase_grid, t_c, chunk))
    return np.array(periods)


phase_grid = np.linspace(0, 1, N_PHASE, endpoint=False)
idx_phase  = np.where((phase_grid >= INDENT_PHASE[0]) &
                       (phase_grid <= INDENT_PHASE[1]))[0]

COLORS = {"clean": "steelblue", "indent": "orange", "severe": "red"}

def classify_periods(periods):
    sev, ind, cln = [], [], []
    for i, p in enumerate(periods):
        if np.nanmin(p) < SEVERE_THRESH:
            sev.append(i)
        elif np.nanmin(p[idx_phase]) < INDENT_THRESH:
            ind.append(i)
        else:
            cln.append(i)
    return cln, ind, sev


# ── Process all runs ──────────────────────────────────────────────────────────
results = []
for fname in RUNS:
    key     = next(k for k in processed_dfs if fname in k)
    df      = processed_dfs[key]
    row     = combined_meta[combined_meta["path"] == key].iloc[0]
    start   = int(row[f"Computed Probe {POS} start"])
    end     = int(row[f"Computed Probe {POS} end"])
    omega   = 2 * np.pi * FREQ
    win_t   = np.arange(end - start) / FS

    sig_raw    = df[f"Probe {POS}"].values[start:end]
    sig_interp = df[f"eta_{POS}_interp"].values[start:end]
    out_interp = df[f"eta_{OUT_POS}_interp"].values[start:end]

    meta_fft = float(row.get(f"Probe {POS} Amplitude (FFT)", np.nan))
    out_fft  = float(row.get(f"Probe {OUT_POS} Amplitude (FFT)", np.nan))

    nan_mask, sig_recon, sine, amp_lsq = reconstruct_variant_b(
        sig_raw, sig_interp, win_t, omega
    )
    n_orig_nan  = np.isnan(sig_raw).sum()
    n_total_nan = nan_mask.sum()
    n_new       = n_total_nan - n_orig_nan

    fft_recon = fft_amp_at(sig_recon)

    uc_before = find_upcrossings(sig_interp)
    uc_after  = find_upcrossings(sig_recon)
    per_before = extract_periods(sig_interp, uc_before)
    per_after  = extract_periods(sig_recon,  uc_after)
    cln_b, ind_b, sev_b = classify_periods(per_before)
    cln_a, ind_a, sev_a = classify_periods(per_after)

    results.append(dict(
        label      = fname.replace("fullpanel-nowind-amp0300-freq1600-per40-depth580-mstop30-", ""),
        win_t      = win_t + start / FS,
        sig_interp = sig_interp,
        sig_recon  = sig_recon,
        sine       = sine,
        nan_mask   = nan_mask,
        n_new      = n_new,
        meta_fft   = meta_fft,
        fft_recon  = fft_recon,
        out_fft    = out_fft,
        amp_lsq    = amp_lsq,
        per_before = per_before,
        per_after  = per_after,
        cln_b=cln_b, ind_b=ind_b, sev_b=sev_b,
        cln_a=cln_a, ind_a=ind_a, sev_a=sev_a,
    ))
    print(f"{fname[-8:]}  orig_fft={meta_fft:.2f}mm  recon_fft={fft_recon:.2f}mm  "
          f"new_flagged={n_new}  sev:{len(sev_b)}→{len(sev_a)}  "
          f"ind:{len(ind_b)}→{len(ind_a)}  cln:{len(cln_b)}→{len(cln_a)}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharey="col")
fig.suptitle(
    f"Variant B reconstruction — sigma-clip (σ={SIGMA_THRESH}×) + amp gate (±{AMP_GATE_FACTOR}× fitted)\n"
    "Left: analysis window (blue=kept, red=reconstructed)    "
    "Right: period overlay (colored=before, gray=after, black=medians)",
    fontsize=10,
)

for row_i, r in enumerate(results):
    ax_sig = axes[row_i, 0]
    ax_per = axes[row_i, 1]

    # ── Left: time-domain signal ─────────────────────────────────────────────
    kept  = np.where(r["nan_mask"], np.nan, r["sig_interp"])
    recon = np.where(r["nan_mask"], r["sig_recon"], np.nan)

    ax_sig.plot(r["win_t"], kept,         color="steelblue", lw=0.7, label="kept")
    ax_sig.plot(r["win_t"], recon,        color="red",       lw=1.0, label="reconstructed", zorder=3)
    ax_sig.plot(r["win_t"], r["sine"],    color="gray",      lw=0.6, ls="--", alpha=0.6,
                label=f"LSQ sine ({r['amp_lsq']:.1f} mm)")
    ax_sig.set_ylabel("η (mm)")
    ax_sig.set_title(
        f"{r['label']}   "
        f"FFT: {r['meta_fft']:.1f} → {r['fft_recon']:.1f} mm   "
        f"OUT/IN: {r['out_fft']/r['meta_fft']:.3f} → {r['out_fft']/r['fft_recon']:.3f}   "
        f"({r['n_new']} new samples flagged)",
        fontsize=8,
    )
    ax_sig.legend(fontsize=7, loc="upper right")
    ax_sig.grid(True, alpha=0.3)

    # ── Right: period overlay ────────────────────────────────────────────────
    # Before: colored by severity
    for i in r["cln_b"]:
        ax_per.plot(phase_grid, r["per_before"][i], color="steelblue", lw=0.5, alpha=0.4)
    for i in r["ind_b"]:
        ax_per.plot(phase_grid, r["per_before"][i], color="orange",    lw=0.5, alpha=0.5)
    for i in r["sev_b"]:
        ax_per.plot(phase_grid, r["per_before"][i], color="red",       lw=0.7, alpha=0.7)

    # After: gray
    for p in r["per_after"]:
        ax_per.plot(phase_grid, p, color="darkgray", lw=0.5, alpha=0.3)

    # Medians
    med_before = np.nanmedian(r["per_before"], axis=0)
    med_after  = np.nanmedian(r["per_after"],  axis=0)
    ax_per.plot(phase_grid, med_before, color="black",  lw=1.5, ls="--", label="median before")
    ax_per.plot(phase_grid, med_after,  color="black",  lw=1.5, ls="-",  label="median after")

    ax_per.axvspan(INDENT_PHASE[0], INDENT_PHASE[1], alpha=0.06, color="red")
    ax_per.set_title(
        f"Periods — before: {len(r['sev_b'])} sev / {len(r['ind_b'])} ind / {len(r['cln_b'])} cln   "
        f"→ after: {len(r['sev_a'])} sev / {len(r['ind_a'])} ind / {len(r['cln_a'])} cln",
        fontsize=8,
    )
    ax_per.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_per.set_xticklabels(["0\n↑cross", "¼", "½\ntrough", "¾", "1\n↑cross"])
    ax_per.legend(fontsize=7, loc="upper right")
    ax_per.grid(True, alpha=0.3)

axes[-1, 0].set_xlabel("Time (s)")
axes[-1, 1].set_xlabel("Phase (fraction of period)")

# Legend for period colours (top right panel)
handles = [
    mlines.Line2D([], [], color="steelblue", lw=1.5, label="clean (before)"),
    mlines.Line2D([], [], color="orange",    lw=1.5, label="indent (before)"),
    mlines.Line2D([], [], color="red",       lw=1.5, label="severe (before)"),
    mlines.Line2D([], [], color="darkgray",  lw=1.5, label="all (after)"),
]
axes[0, 1].legend(handles=handles, fontsize=7, loc="lower left")

plt.tight_layout()
out_path = Path(__file__).parent / "reconstruct_all3.png"
plt.savefig(out_path, dpi=150)
print(f"\nSaved: {out_path}")
