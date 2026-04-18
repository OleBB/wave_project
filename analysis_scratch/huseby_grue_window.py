"""
Huseby & Grue 1.425 Hz window applied to our per240 1.4 Hz runs
================================================================

Huseby & Grue (J. Fluid Mech. 2000, §3) chose a very specific FFT
window for their 1.425 Hz wave-force experiments: **35.088 s < t <
42.105 s = exactly 10 wave periods**, with t=0 at wavemaker start.
They picked this window because:
  (i) the leading transient has passed the measurement position
      (they advise 10–15 periods after first arrival),
  (ii) the first-harmonic waves are "reasonably periodic",
  (iii) free second-harmonic parasitic waves, which travel at half the
       main-wave speed, have *not* yet reached the measurement point,
  (iv) reflections from the beach have not returned.
Their FFT is taken over exactly 10T so there is no spectral leakage.

This script transposes the same window idea onto our 1.4 Hz per240 runs
(no 1.425 Hz data in this project). Window: start = 35 s, length = 10T
at 1.4 Hz = 7.143 s → **t ∈ [35.0, 42.143] s** (exactly 10 integer
periods → no leakage).

Per40 runs are **excluded** — the paddle stops at ~28 s at 40 periods,
so the H&G window lands in ringdown. User-eyeballed stable windows
(see `snarvei_eyeballing.md`) confirm: IN probe stable 20–39 s, OUT
probe stable 27–43 s at 1.4 Hz per40 — the H&G window straddles the
end of the usable signal.

For each per240 1.4 Hz run, this script computes three AFFT values at
both probes (9373/170 and 12400/250):

  1. **Pipeline AFFT** — what the standard pipeline window yields
     (as read from the meta.json cache).
  2. **H&G AFFT** — single FFT over the [35, 42.143] s window.
  3. **Sliding-window AFFT surface** — 10T-wide window stepped across
     the signal (step 0.5 s), to visualise stability around the H&G
     region.

Then it compares OUT/IN computed three ways (pipeline / H&G /
sliding-region-mean) per (amp, wind) and plots.

Run from repo root (loads processed_dfs → ~30 s):
    conda run -n draumkvedet python analysis_scratch/huseby_grue_window.py

Outputs:
    analysis_scratch/huseby_grue_window.pdf
    analysis_scratch/huseby_grue_window_summary.csv
    analysis_scratch/huseby_grue_window_findings.md
"""

import sys
import warnings
import re
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

# ── Config ────────────────────────────────────────────────────────────────────
FS              = 250.0
TARGET_FREQ     = 1.4
WINDOW_PERIODS  = 10                          # H&G: exactly 10T, no leakage
HG_WINDOW_S     = WINDOW_PERIODS / TARGET_FREQ  # 7.143 s at 1.4 Hz
HG_START_S      = 35.0                        # H&G t=0 is wavemaker start
HG_END_S        = HG_START_S + HG_WINDOW_S    # 42.143 s

SLIDING_STEP_S  = 0.5
SLIDING_START_S = 10.0
SLIDING_END_S   = 100.0

FFT_BAND_HZ     = 0.05   # ±half-width for nearest-bin pick (matches pipeline)
PROBES          = ["9373/170", "12400/250"]
IN_POS, OUT_POS = PROBES

RESULTS_DIRS = [
    Path("waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange"),
    Path("waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"),
]

OUT_PDF = Path(__file__).parent / "huseby_grue_window.pdf"
OUT_CSV = Path(__file__).parent / "huseby_grue_window_summary.csv"
OUT_MD  = Path(__file__).parent / "huseby_grue_window_findings.md"


# ── FFT helpers (mirror the pipeline) ─────────────────────────────────────────
def fft_amp_at_freq(segment: np.ndarray, target_hz: float,
                    fs: float = FS, band_hz: float = FFT_BAND_HZ) -> float:
    """Single-shot FFT amplitude at `target_hz`: pick the nearest positive bin
    within ±band_hz. Normalisation: 2·|FFT|/N (pipeline convention)."""
    N = len(segment)
    if N < 4:
        return np.nan
    seg = segment.copy()
    if np.isnan(seg).any():
        idx = np.arange(N)
        good = ~np.isnan(seg)
        if good.sum() < N * 0.9:
            return np.nan
        seg = np.interp(idx, idx[good], seg[good])
    freqs = np.fft.fftfreq(N, d=1.0 / fs)
    pos = freqs > 0
    pos_f = freqs[pos]
    mask = (pos_f >= target_hz - band_hz) & (pos_f <= target_hz + band_hz)
    if not mask.any():
        j = int(np.argmin(np.abs(pos_f - target_hz)))
    else:
        local_idx = int(np.argmin(np.abs(pos_f[mask] - target_hz)))
        j = np.where(mask)[0][local_idx]
    fft_vals = np.fft.fft(seg)
    amps_pos = 2.0 * np.abs(fft_vals[pos]) / N
    return float(amps_pos[j])


def sliding_afft(signal: np.ndarray, target_hz: float,
                 window_s: float, step_s: float = SLIDING_STEP_S,
                 fs: float = FS, band_hz: float = FFT_BAND_HZ):
    """Slide `window_s` across signal in `step_s` steps, return (t_start, A).

    t_start is the window START (not centre) — matches H&G's [35, 42] convention.
    """
    N_win = int(round(window_s * fs))
    step  = int(round(step_s * fs))
    if N_win >= len(signal):
        return np.array([]), np.array([])
    starts = np.arange(0, len(signal) - N_win + 1, step)
    t_start = starts / fs
    A = np.full(len(starts), np.nan)
    for i, s in enumerate(starts):
        A[i] = fft_amp_at_freq(signal[s:s + N_win], target_hz, fs, band_hz)
    return t_start, A


def get_signal(df: pd.DataFrame, pos: str) -> np.ndarray | None:
    for col in (f"eta_{pos}_interp", f"eta_{pos}"):
        if col in df.columns:
            return df[col].to_numpy(dtype=float)
    return None


# ── 1. Load metadata and target runs ──────────────────────────────────────────
print("1. Loading metadata …")
meta, _, _, _ = load_analysis_data(*RESULTS_DIRS, load_processed=False)

def per_of(p):
    m = re.search(r"per(\d+)", str(p))
    return int(m.group(1)) if m else None
meta["per"] = meta["path"].apply(per_of)

target = meta[
    (meta["per"] == 240)
    & (meta["WaveFrequencyInput [Hz]"].round(2) == TARGET_FREQ)
    & (meta["PanelCondition"] == "full")
    & (meta["quality_flag"] == "ok")
    & meta["WindCondition"].isin(["no", "full"])
].copy()
target = target.sort_values(["WaveAmplitudeInput [Volt]", "WindCondition"]).reset_index(drop=True)
print(f"   {len(target)} per240 1.4 Hz full-panel runs in scope")
print(target[["WaveAmplitudeInput [Volt]", "WindCondition", "path"]].to_string(index=False))

print("\n2. Loading processed time-series …")
pdfs = load_processed_dfs(*RESULTS_DIRS)
print(f"   {len(pdfs)} time-series loaded")

# ── 2. Compute per-run AFFTs ──────────────────────────────────────────────────
print("\n3. Computing AFFTs …")
rows = []
sliding_curves = {}   # (path, probe) -> (t_start, A)

for _, r in target.iterrows():
    path = r["path"]
    df = pdfs.get(path)
    if df is None:
        print(f"   [skip] no processed DF for {Path(path).name}")
        continue

    row = {
        "path":  path,
        "amp":   float(r["WaveAmplitudeInput [Volt]"]),
        "wind":  r["WindCondition"],
    }
    for probe in PROBES:
        sig = get_signal(df, probe)
        if sig is None:
            continue
        N = len(sig)

        # H&G window — exact sample indices
        i0 = int(round(HG_START_S * FS))
        i1 = int(round(HG_END_S * FS))
        if i1 > N:
            row[f"A_hg_{probe}"] = np.nan
        else:
            row[f"A_hg_{probe}"] = fft_amp_at_freq(sig[i0:i1], TARGET_FREQ)

        # Pipeline AFFT from meta (FFT-amplitude column)
        col = f"Probe {probe} Amplitude (FFT)"
        row[f"A_pipe_{probe}"] = float(r[col]) if col in r else np.nan

        # Sliding sweep
        ts, A = sliding_afft(sig, TARGET_FREQ, window_s=HG_WINDOW_S)
        sliding_curves[(path, probe)] = (ts, A)
        mask = (ts >= SLIDING_START_S) & (ts <= SLIDING_END_S)
        if mask.sum() > 3:
            row[f"A_slide_mean_{probe}"]   = float(np.nanmean(A[mask]))
            row[f"A_slide_median_{probe}"] = float(np.nanmedian(A[mask]))
        else:
            row[f"A_slide_mean_{probe}"]   = np.nan
            row[f"A_slide_median_{probe}"] = np.nan
    rows.append(row)

df_out = pd.DataFrame(rows)

# OUT/IN computed three ways
for tag in ("pipe", "hg", "slide_mean", "slide_median"):
    c_in  = f"A_{tag}_{IN_POS}"
    c_out = f"A_{tag}_{OUT_POS}"
    if c_in in df_out.columns and c_out in df_out.columns:
        df_out[f"OUTIN_{tag}"] = df_out[c_out] / df_out[c_in]

df_out.to_csv(OUT_CSV, index=False)
print(f"   Summary → {OUT_CSV.relative_to(BASE)}")

print("\n4. OUT/IN comparison per run:")
cols_show = ["amp", "wind", "OUTIN_pipe", "OUTIN_hg", "OUTIN_slide_median"]
print(df_out[cols_show].round(4).to_string(index=False))

print("\n5. Δ (H&G − pipe) per run:")
dh = df_out["OUTIN_hg"] - df_out["OUTIN_pipe"]
for _, r in df_out.iterrows():
    d_hg = r["OUTIN_hg"] - r["OUTIN_pipe"]
    d_sm = r["OUTIN_slide_median"] - r["OUTIN_pipe"]
    print(f"   {r['amp']:.1f} V  {r['wind']:>4s}  "
          f"OUTIN_pipe={r['OUTIN_pipe']:.3f}  OUTIN_hg={r['OUTIN_hg']:.3f} (Δ={d_hg:+.3f})  "
          f"OUTIN_slide={r['OUTIN_slide_median']:.3f} (Δ={d_sm:+.3f})")

# ── 3. Plot ───────────────────────────────────────────────────────────────────
print("\n6. Plotting …")
WIND_COLOR = {"no": "#2980B9", "full": "#E74C3C"}

fig, axes = plt.subplots(3, 2, figsize=(16, 11), sharex=True)
fig.suptitle(
    f"Huseby & Grue window applied to per240 {TARGET_FREQ:.1f} Hz — "
    f"sliding 10T AFFT vs H&G single-shot [35, {HG_END_S:.2f}] s",
    fontsize=12, fontweight="bold",
)

amps_order = [0.1, 0.2, 0.3]
for i, amp in enumerate(amps_order):
    for j, probe in enumerate(PROBES):
        ax = axes[i, j]
        for _, r in target[target["WaveAmplitudeInput [Volt]"] == amp].iterrows():
            path = r["path"]
            wind = r["WindCondition"]
            key  = (path, probe)
            if key not in sliding_curves:
                continue
            ts, A = sliding_curves[key]
            col = WIND_COLOR[wind]
            ax.plot(ts, A, color=col, lw=1.1, alpha=0.85,
                    label=f"{wind} wind (path …{Path(path).name[-20:]})")

        # H&G band
        ax.axvspan(HG_START_S, HG_END_S, color="#2ECC71", alpha=0.18, lw=0,
                   label="H&G window")
        ax.axvline(HG_START_S, color="#2ECC71", lw=0.6, ls="--")
        ax.axvline(HG_END_S,   color="#2ECC71", lw=0.6, ls="--")

        # Mark H&G AFFT values (one marker per run at the midpoint)
        for _, r in target[target["WaveAmplitudeInput [Volt]"] == amp].iterrows():
            path = r["path"]
            wind = r["WindCondition"]
            hg_val = df_out[df_out["path"] == path][f"A_hg_{probe}"].values
            if len(hg_val) == 0 or np.isnan(hg_val[0]):
                continue
            ax.scatter((HG_START_S + HG_END_S) / 2, hg_val[0],
                       marker="D", s=55, color=WIND_COLOR[wind],
                       edgecolor="black", linewidth=0.5, zorder=5)

        # Pipeline AFFT horizontal lines per run
        for _, r in target[target["WaveAmplitudeInput [Volt]"] == amp].iterrows():
            path = r["path"]
            wind = r["WindCondition"]
            pipe_val = df_out[df_out["path"] == path][f"A_pipe_{probe}"].values
            if len(pipe_val) == 0 or np.isnan(pipe_val[0]):
                continue
            ax.axhline(pipe_val[0], color=WIND_COLOR[wind],
                       lw=0.7, ls=":", alpha=0.7)

        ax.set_xlim(SLIDING_START_S, SLIDING_END_S)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{amp:.1f} V — probe {probe}", fontsize=9, fontweight="bold")
        if i == 2:
            ax.set_xlabel("window start [s from wavemaker start]", fontsize=9)
        if j == 0:
            ax.set_ylabel("AFFT at 1.4 Hz  [mm]", fontsize=9)
        if i == 0 and j == 0:
            ax.legend(fontsize=6.5, loc="upper right", framealpha=0.85)

fig.text(0.5, 0.005,
         "Solid curves: sliding-AFFT, 10T = 7.14 s window. Green band: H&G [35, 42.14] s. "
         "Diamonds: H&G single-shot AFFT per run. Dotted horizontal: pipeline AFFT per run.",
         ha="center", fontsize=8, color="#444", style="italic")

fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.06, hspace=0.25)
OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PDF, bbox_inches="tight")
print(f"   Saved → {OUT_PDF.relative_to(BASE)}")

# ── 4. Findings markdown ──────────────────────────────────────────────────────
print("\n7. Writing findings markdown …")

# Per-(amp, wind) aggregated OUT/IN across runs
agg_rows = []
for amp in amps_order:
    for wind in ["no", "full"]:
        sub = df_out[(df_out["amp"] == amp) & (df_out["wind"] == wind)]
        if sub.empty:
            continue
        agg_rows.append({
            "amp":  amp, "wind": wind, "n": len(sub),
            "pipe_m":  float(sub["OUTIN_pipe"].mean()),
            "hg_m":    float(sub["OUTIN_hg"].mean()),
            "slide_m": float(sub["OUTIN_slide_median"].mean()),
        })
agg = pd.DataFrame(agg_rows)

body = []
body.append("# Huseby & Grue window applied to per240 1.4 Hz runs")
body.append("")
body.append("**Date**: 2026-04-18.")
body.append("**Script**: `analysis_scratch/huseby_grue_window.py`.")
body.append(f"**Figure**: `analysis_scratch/huseby_grue_window.pdf`.")
body.append(f"**Data**: {len(df_out)} per240 1.4 Hz full-panel quality-ok runs "
            f"(3 amplitudes × nowind+fullwind where available).")
body.append("")
body.append("## Window")
body.append("")
body.append(f"- H&G exactly: 35.088 s < t < 42.105 s at 1.425 Hz = 10T (no leakage).")
body.append(f"- Our 1.4 Hz equivalent: **[{HG_START_S:.1f}, {HG_END_S:.3f}] s = 10T** "
            f"at 1.4 Hz (also no leakage, no scaling artefact).")
body.append(f"- t=0 = wavemaker start (confirmed with user).")
body.append("- Per40 runs excluded (paddle stops ≈ 28 s; H&G window is in ringdown).")
body.append("")
body.append("## OUT/IN — three window methods")
body.append("")
body.append("| amp | wind | n | pipeline | H&G 10T | sliding-median (15–100 s, 10T window) |")
body.append("|-----|------|---|----------|---------|----------------------------------------|")
for r in agg_rows:
    body.append(f"| {r['amp']:.1f} V | {r['wind']:>4s} | {r['n']} | "
                f"{r['pipe_m']:.3f} | {r['hg_m']:.3f} | {r['slide_m']:.3f} |")
body.append("")
body.append("## Per-run detail")
body.append("")
body.append("| amp | wind | OUTIN_pipe | OUTIN_hg | Δ(H&G − pipe) | OUTIN_slide_median |")
body.append("|-----|------|-----------|----------|---------------|---------------------|")
for _, r in df_out.iterrows():
    d = r["OUTIN_hg"] - r["OUTIN_pipe"]
    body.append(f"| {r['amp']:.1f} V | {r['wind']:>4s} | {r['OUTIN_pipe']:.3f} | "
                f"{r['OUTIN_hg']:.3f} | {d:+.3f} | {r['OUTIN_slide_median']:.3f} |")
body.append("")
body.append("## Quick take (user to refine)")
body.append("")
body.append("- If the three OUT/IN columns agree within a few percent, the pipeline "
            "window is already giving H&G-equivalent numbers at 1.4 Hz. That would "
            "confirm our methodology is consistent with the published standard.")
body.append("- If the H&G window yields systematically different OUT/IN values, it "
            "points at a window-dependent bias — most likely because the pipeline "
            "window (typically 19–39 s, samples 4800–9750) is longer than 10T and "
            "may include regions before the wave train has fully settled, or it is "
            "contaminated by reflections / parasitic waves at its tail end.")
body.append("- The sliding-AFFT sweep shows where A settles vs window-start time. "
            "If H&G-window AFFT sits on the stable plateau of the sweep, it is "
            "trustworthy; if it sits on a transient, it is not.")
body.append("")
body.append("## Caveats")
body.append("")
body.append("- No 1.425 Hz data — this is an extrapolation to 1.4 Hz using the same "
            "relative window (35 s start, 10T length). Tank geometry is similar "
            "to H&G (24.6 × 0.5 × 0.6 m vs our setup) so the timing of parasitic "
            "waves and beach reflections should be comparable.")
body.append("- Only per240 runs. Per40 windows end near or before 35 s.")
body.append("- Pipeline AFFT column is read from meta.json (no re-computation).")

OUT_MD.write_text("\n".join(body), encoding="utf-8")
print(f"   Saved → {OUT_MD.relative_to(BASE)}")
print("\nDone.")
