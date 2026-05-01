"""
Spectral-subtraction proof of concept — single canon case.
==========================================================

One probe (9373/170, IN-wall), one frequency / amplitude (1.4 Hz / 0.2 V),
both wind conditions. Tests whether subtracting a wind-only PSD from a
fullwind+wave run brings the recovered paddle amplitude back down to the
nowind value.

Method (per user spec, 2026-05-01):

  Nowind:
    P_nowind_clean(f) = max(0, P_nowind_run(f) - P_still(f))
    A_nowind_clean    = sqrt( sum_{|f - f_p| <= Δ}  2 P_nowind_clean(f) df )

  Fullwind:
    P_fullwind_clean(f) = max(0,  P_fullwind_run(f)
                                  - P_wind_3s(f)
                                  - P_still(f))
    A_fullwind_clean    = sqrt( sum_{|f - f_p| <= Δ}  2 P_fullwind_clean(f) df )

PSD definition: scipy.signal.periodogram, scaling='density', one-sided.
Integer-cycle windows give a discrete bin at f_p exactly (H&G + UC-snap).

Caveat: only valid if paddle wave + wind waves + instrument noise are
linearly uncorrelated (variances add). If wind couples to the paddle wave
at f_p, subtraction under-corrects there — which is precisely the H1 wind-
enhancement test.

Inputs (read from canon waveprocessed cache):
    nowind canon       — 20260327 per40
    fullwind canon     — 20260327 per40 (matched same day)
    stillwater         — 20260327 nowind+nowave-mstop30-run1
    long wind shape    — 20260326 fullwind+nowave-mstop330-run1 (381 s)

Outputs:
    analysis_scratch/spectral_subtract_canon14_psd.png
    analysis_scratch/spectral_subtract_canon14.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import os
os.chdir(BASE)

from wavescripts.improved_data_loader import load_analysis_data, load_processed_dfs
from wavescripts.plot_utils import apply_thesis_style, WIND_COLOR_MAP

# ── Configuration ────────────────────────────────────────────────────────
FS = 250.0
PROBE = "9373/170"
F_P   = 1.4
AMP_V = 0.2
DELTA = 0.05      # ±0.05 Hz integration band
SNIPPET_S = 3.0   # pre-paddle window length for P_wind_3s

DIRS = [
    "waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange",
    "waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange",
]

NOWIND_PATH    = ("/Users/ole/Kodevik/wave_project/wavedata/20260327-ProbePos4_31_FPV_2-"
                  "tett6roof-under9Mooring30-height100-lowrange/"
                  "fullpanel-nowind-amp0200-freq1400-per40-depth580-mstop30-run1.csv")
FULLWIND_PATH  = ("/Users/ole/Kodevik/wave_project/wavedata/20260327-ProbePos4_31_FPV_2-"
                  "tett6roof-under9Mooring30-height100-lowrange/"
                  "fullpanel-fullwind-amp0200-freq1400-per40-depth580-mstop30-run1.csv")
STILL_PATH     = ("/Users/ole/Kodevik/wave_project/wavedata/20260327-ProbePos4_31_FPV_2-"
                  "tett6roof-under9Mooring30-height100-lowrange/"
                  "fullpanel-nowind-nowave-depth580-mstop30-run1.csv")
LONGWIND_PATH  = ("/Users/ole/Kodevik/wave_project/wavedata/20260326-ProbePos4_31_FPV_2-"
                  "tett6roof-under9Mooring-height100-lowrange/"
                  "fullpanel-fullwind-nowave-depth580-mstop330-run1.csv")


# ── Load ──────────────────────────────────────────────────────────────────
print("Loading meta + processed_dfs …")
meta, _, _, _ = load_analysis_data(*DIRS, load_processed=False)
proc = {}
for d in DIRS:
    proc.update(load_processed_dfs(d))
print(f"  meta: {len(meta)} rows; processed_dfs: {len(proc)}")

for label, p in [("NOWIND", NOWIND_PATH), ("FULLWIND", FULLWIND_PATH),
                 ("STILL",  STILL_PATH),  ("LONGWIND", LONGWIND_PATH)]:
    if p not in proc:
        raise SystemExit(f"Missing {label} path in processed_dfs: {p}")


# ── Helpers ──────────────────────────────────────────────────────────────
def _eta_col(df: pd.DataFrame, probe: str) -> str:
    for c in (f"eta_{probe}_interp", f"eta_{probe}"):
        if c in df.columns:
            return c
    raise KeyError(f"no eta column for {probe} in df (cols: {list(df.columns)[:8]}...)")


def _hg_window_indices(meta_row, probe: str) -> tuple[int, int]:
    """Return (i_start, i_end) sample indices of the H&G window for ``probe``."""
    s = int(meta_row[f"Computed Probe {probe} start"])
    e = int(meta_row[f"Computed Probe {probe} end"])
    return s, e


def periodogram_segment(eta: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Rectangular-window one-sided periodogram, density scaling."""
    f, P = signal.periodogram(eta, fs=fs, scaling="density",
                              window="boxcar", detrend="constant")
    return f, P


def welch_segment(eta: np.ndarray, fs: float, nperseg: int) -> tuple[np.ndarray, np.ndarray]:
    """Welch with same nperseg as the run window — same df, same bin alignment."""
    f, P = signal.welch(eta, fs=fs, nperseg=nperseg, scaling="density",
                        window="boxcar", detrend="constant",
                        noverlap=0)
    return f, P


def amplitude_in_band(f: np.ndarray, P: np.ndarray, f_p: float,
                      delta: float) -> tuple[float, int]:
    """A = sqrt( sum_{|f-fp|<=delta} 2 * P * df ).

    With df > delta (coarse grid) this collapses to a single-bin pick at f_p,
    in which case A = sqrt(2 * P[bin_p] * df), matching the rfft amplitude
    convention for a rectangular window with integer cycles.
    """
    df = f[1] - f[0]
    mask = np.abs(f - f_p) <= delta
    if not mask.any():
        # Fall back to nearest single bin.
        k = int(np.argmin(np.abs(f - f_p)))
        mask = np.zeros_like(f, dtype=bool)
        mask[k] = True
    var = np.sum(2.0 * P[mask] * df)
    return float(np.sqrt(max(var, 0.0))), int(mask.sum())


# ── Per-condition extraction ──────────────────────────────────────────────
nowind_row   = meta[meta["path"] == NOWIND_PATH].iloc[0]
fullwind_row = meta[meta["path"] == FULLWIND_PATH].iloc[0]

eta_nowind   = proc[NOWIND_PATH][_eta_col(proc[NOWIND_PATH], PROBE)].to_numpy(float)
eta_fullwind = proc[FULLWIND_PATH][_eta_col(proc[FULLWIND_PATH], PROBE)].to_numpy(float)
eta_still    = proc[STILL_PATH][_eta_col(proc[STILL_PATH], PROBE)].to_numpy(float)
eta_longwind = proc[LONGWIND_PATH][_eta_col(proc[LONGWIND_PATH], PROBE)].to_numpy(float)

s_nw, e_nw = _hg_window_indices(nowind_row,   PROBE)
s_fw, e_fw = _hg_window_indices(fullwind_row, PROBE)
N_nw = e_nw - s_nw
N_fw = e_fw - s_fw
print(f"\nH&G windows at probe {PROBE}:")
print(f"  nowind   [{s_nw:>5}, {e_nw:>5}]  N={N_nw} samples ({N_nw/FS:.3f} s)")
print(f"  fullwind [{s_fw:>5}, {e_fw:>5}]  N={N_fw} samples ({N_fw/FS:.3f} s)")


# ── PSDs ─────────────────────────────────────────────────────────────────
# 1) Run windows — periodogram on the actual H&G-snapped slice.
f_nw_run, P_nw_run = periodogram_segment(eta_nowind  [s_nw:e_nw], FS)
f_fw_run, P_fw_run = periodogram_segment(eta_fullwind[s_fw:e_fw], FS)

# 2) Stillwater — Welch with nperseg = N_nw (so df matches nowind run grid).
#    Strip the very first second to avoid any startup transient.
still_clip = eta_still[int(1.0 * FS):]
f_still_nw, P_still_nw = welch_segment(still_clip, FS, nperseg=N_nw)
# Same trick on the fullwind grid (might differ by ±a few samples post-UC snap).
f_still_fw, P_still_fw = welch_segment(still_clip, FS, nperseg=N_fw)

# 3) Pre-paddle 3 s of the FULLWIND run — periodogram on first SNIPPET_N samples.
SNIPPET_N = int(SNIPPET_S * FS)
seg_pre = eta_fullwind[:SNIPPET_N]
f_w3s, P_w3s = periodogram_segment(seg_pre, FS)

# 4) Long fullwind+nowave run — Welch on the entire record at the run's nperseg
#    (this gives a smooth wind-only PSD shape on the same grid as the run window).
f_longw_fw, P_longw_fw = welch_segment(eta_longwind, FS, nperseg=N_fw)


# ── Interpolators onto each run's grid ───────────────────────────────────
def interp_psd(f_src: np.ndarray, P_src: np.ndarray,
               f_dst: np.ndarray) -> np.ndarray:
    return np.interp(f_dst, f_src, P_src, left=P_src[0], right=P_src[-1])


# Subtraction step
P_still_on_nw   = interp_psd(f_still_nw, P_still_nw, f_nw_run)
P_still_on_fw   = interp_psd(f_still_fw, P_still_fw, f_fw_run)
P_wind3s_on_fw  = interp_psd(f_w3s,      P_w3s,      f_fw_run)
P_longw_on_fw   = interp_psd(f_longw_fw, P_longw_fw, f_fw_run)

P_nw_clean      = np.clip(P_nw_run - P_still_on_nw,                                 0, None)
P_fw_clean_3s   = np.clip(P_fw_run - P_wind3s_on_fw - P_still_on_fw,                0, None)
P_fw_clean_long = np.clip(P_fw_run - P_longw_on_fw  - P_still_on_fw,                0, None)


# ── Amplitude tally ──────────────────────────────────────────────────────
df_nw = f_nw_run[1] - f_nw_run[0]
df_fw = f_fw_run[1] - f_fw_run[0]

A_nw_raw,            n_nw  = amplitude_in_band(f_nw_run, P_nw_run,        F_P, DELTA)
A_nw_clean,          _     = amplitude_in_band(f_nw_run, P_nw_clean,      F_P, DELTA)
A_fw_raw,            n_fw  = amplitude_in_band(f_fw_run, P_fw_run,        F_P, DELTA)
A_fw_clean_3s,       _     = amplitude_in_band(f_fw_run, P_fw_clean_3s,   F_P, DELTA)
A_fw_clean_long,     _     = amplitude_in_band(f_fw_run, P_fw_clean_long, F_P, DELTA)

# Reference: cached canonical FFT amplitude from meta.
A_meta_nowind   = float(nowind_row.get(  f"Probe {PROBE} Amplitude (FFT)", np.nan))
A_meta_fullwind = float(fullwind_row.get(f"Probe {PROBE} Amplitude (FFT)", np.nan))


print(f"\n=== Probe {PROBE},  f_p = {F_P} Hz,  A = {AMP_V} V,  ±{DELTA} Hz band ===")
print(f"   df_nw = {df_nw:.4f} Hz   df_fw = {df_fw:.4f} Hz   "
      f"bins integrated = {n_nw}/{n_fw}  (collapses to nearest bin if 0)")

rows = [
    ("A_meta_FFT  (cached, nowind)",       A_meta_nowind,    "—"),
    ("A_psd_raw   (P_nowind_run)",         A_nw_raw,         "should equal A_meta_FFT"),
    ("A_psd_clean (− P_still)",            A_nw_clean,       "noise-floor removed"),
    ("",                                   None,             ""),
    ("A_meta_FFT  (cached, fullwind)",     A_meta_fullwind,  "—"),
    ("A_psd_raw   (P_fullwind_run)",       A_fw_raw,         "should equal A_meta_FFT"),
    ("A_clean_3s  (− P_wind_3s − P_still)", A_fw_clean_3s,    "wind from 3 s pre-paddle"),
    ("A_clean_long (− P_longwind − P_still)", A_fw_clean_long, "wind from 381 s long run"),
]

print()
for name, val, note in rows:
    if name == "":
        print()
        continue
    if val is None or not np.isfinite(val):
        print(f"  {name:42s}    nan       {note}")
    else:
        print(f"  {name:42s}  {val:7.3f} mm  {note}")

# Residual ratios (target = 1.000 if subtraction is perfect)
def _ratio(a, b):
    return a / b if b > 0 else np.nan

print()
print("=== Residual ratios (target = 1.000) ===")
print(f"  A_nw_clean        / A_meta_FFT(nowind)   = {_ratio(A_nw_clean, A_meta_nowind):.4f}")
print(f"  A_clean_3s(fw)    / A_meta_FFT(nowind)   = {_ratio(A_fw_clean_3s, A_meta_nowind):.4f}   "
      f"(does subtraction recover the nowind amplitude?)")
print(f"  A_clean_long(fw)  / A_meta_FFT(nowind)   = {_ratio(A_fw_clean_long, A_meta_nowind):.4f}")
print(f"  A_meta_FFT(fw)    / A_meta_FFT(nowind)   = {_ratio(A_meta_fullwind, A_meta_nowind):.4f}   "
      f"(uncorrected wind enhancement; CH04 §… reference)")


# ── CSV ──────────────────────────────────────────────────────────────────
csv_rows = [
    {"quantity": "f_p_Hz",                       "value_mm": F_P,                "note": ""},
    {"quantity": "df_nowind_Hz",                 "value_mm": df_nw,              "note": ""},
    {"quantity": "df_fullwind_Hz",               "value_mm": df_fw,              "note": ""},
    {"quantity": "delta_band_Hz",                "value_mm": DELTA,              "note": ""},
    {"quantity": "A_meta_FFT_nowind_mm",         "value_mm": A_meta_nowind,      "note": "cached"},
    {"quantity": "A_psd_raw_nowind_mm",          "value_mm": A_nw_raw,           "note": "PSD-derived, should match cached"},
    {"quantity": "A_psd_clean_nowind_mm",        "value_mm": A_nw_clean,         "note": "minus P_still"},
    {"quantity": "A_meta_FFT_fullwind_mm",       "value_mm": A_meta_fullwind,    "note": "cached"},
    {"quantity": "A_psd_raw_fullwind_mm",        "value_mm": A_fw_raw,           "note": "PSD-derived, should match cached"},
    {"quantity": "A_psd_clean_fullwind_3s_mm",   "value_mm": A_fw_clean_3s,      "note": "minus P_wind_3s + P_still"},
    {"quantity": "A_psd_clean_fullwind_long_mm", "value_mm": A_fw_clean_long,    "note": "minus P_longwind + P_still"},
]
csv_path = Path("analysis_scratch/spectral_subtract_canon14.csv")
pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
print(f"\n   CSV → {csv_path}")


# ── Figure: PSDs on log-y, marking f_p ───────────────────────────────────
apply_thesis_style(usetex=False)
plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3})

fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)
COL_NW   = WIND_COLOR_MAP["no"]
COL_FW   = WIND_COLOR_MAP["full"]
COL_STILL = "#444"
COL_LONG  = "#a32"
COL_3S    = "#e09000"
COL_CLEAN = "#000"

# Panel A: nowind
ax = axes[0]
ax.semilogy(f_nw_run,    P_nw_run,         color=COL_NW,    lw=1.4, label=r"$P_\mathrm{nowind\_run}$ (10-period H&G)")
ax.semilogy(f_still_nw,  P_still_nw,       color=COL_STILL, lw=1.0, ls="--", label=r"$P_\mathrm{still}$ (Welch, $n_\mathrm{perseg}=N_\mathrm{run}$)")
ax.semilogy(f_nw_run,    np.where(P_nw_clean > 0, P_nw_clean, np.nan),
            color=COL_CLEAN, lw=1.0, ls=":",
            label=r"$P_\mathrm{nowind\_clean} = \max(0, P_\mathrm{run} - P_\mathrm{still})$")
ax.axvline(F_P, color="#888", lw=0.8, ls=":")
ax.text(F_P, ax.get_ylim()[1] * 0.5, rf"  $f_p = {F_P}$ Hz", fontsize=8, va="top", color="#888")
ax.set_xlim(0, 10)
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel(r"PSD  [$\mathrm{mm^2/Hz}$]")
ax.set_title(rf"Nowind  —  probe {PROBE},  $A_2$ ({AMP_V} V),  $f_p = {F_P}$ Hz")
ax.legend(fontsize=8, loc="upper right", framealpha=0.92)

# Panel B: fullwind
ax = axes[1]
ax.semilogy(f_fw_run,    P_fw_run,         color=COL_FW,    lw=1.4, label=r"$P_\mathrm{fullwind\_run}$ (10-period H&G)")
ax.semilogy(f_w3s,       P_w3s,            color=COL_3S,    lw=1.0, ls="--", label=r"$P_\mathrm{wind\_3s}$ (3 s pre-paddle, same run)")
ax.semilogy(f_longw_fw,  P_longw_fw,       color=COL_LONG,  lw=1.0, ls="-.", label=r"$P_\mathrm{longwind}$ (381 s fullwind+nowave)")
ax.semilogy(f_still_fw,  P_still_fw,       color=COL_STILL, lw=1.0, ls="--", label=r"$P_\mathrm{still}$ (Welch, $n_\mathrm{perseg}=N_\mathrm{run}$)")
ax.semilogy(f_fw_run,    np.where(P_fw_clean_3s > 0, P_fw_clean_3s, np.nan),
            color=COL_CLEAN, lw=1.0, ls=":",
            label=r"$P_\mathrm{fullwind\_clean}$  ($-P_\mathrm{wind\_3s}-P_\mathrm{still}$)")
ax.axvline(F_P, color="#888", lw=0.8, ls=":")
ax.text(F_P, ax.get_ylim()[1] * 0.5, rf"  $f_p = {F_P}$ Hz", fontsize=8, va="top", color="#888")
ax.set_xlim(0, 10)
ax.set_xlabel("Frequency [Hz]")
ax.set_title(rf"Fullwind  —  probe {PROBE},  $A_2$ ({AMP_V} V),  $f_p = {F_P}$ Hz")
ax.legend(fontsize=8, loc="upper right", framealpha=0.92)

# Annotate amplitude readings on each panel
note_nw = (f"$A_\\mathrm{{meta,FFT}}$ = {A_meta_nowind:.3f} mm\n"
           f"$A_\\mathrm{{PSD,raw}}$  = {A_nw_raw:.3f} mm\n"
           f"$A_\\mathrm{{PSD,clean}}$ = {A_nw_clean:.3f} mm")
axes[0].text(0.02, 0.02, note_nw, transform=axes[0].transAxes,
             fontsize=8, va="bottom", ha="left",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#aaa", alpha=0.92))

note_fw = (f"$A_\\mathrm{{meta,FFT}}$  = {A_meta_fullwind:.3f} mm\n"
           f"$A_\\mathrm{{PSD,raw}}$   = {A_fw_raw:.3f} mm\n"
           f"$A_\\mathrm{{clean,3s}}$  = {A_fw_clean_3s:.3f} mm\n"
           f"$A_\\mathrm{{clean,long}}$ = {A_fw_clean_long:.3f} mm")
axes[1].text(0.02, 0.02, note_fw, transform=axes[1].transAxes,
             fontsize=8, va="bottom", ha="left",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#aaa", alpha=0.92))

fig.suptitle("Spectral subtraction — single canon run (1.4 Hz, $A_2$, fullpanel, IN-wall)", fontsize=11)
fig.tight_layout()

png_path = Path("analysis_scratch/spectral_subtract_canon14_psd.png")
fig.savefig(png_path, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"   PNG → {png_path}")

print("\nDone.")
