# %%
"""
Plateau values table — A_IN, A_OUT, OUT/IN at the chosen window.
==================================================================

Companion to the ch04_plateau_overview_A{1,2,3} figures. For every
canon (f, amp, wind) cell, compute the FFT amplitude at the paddle
frequency over the chosen window:

    t_start(probe, f) = r_probe / c_g(f, h) + N_OFFSET / f
    t_end(probe, f)   = t_start + N_LENGTH / f

with N_OFFSET = 7, N_LENGTH = 10 (uniform across thesis freqs). The
window is **probe-shifted**: A_IN evaluates the IN signal over its own
window; A_OUT evaluates the OUT signal over its own (later) window.
This matches how the live pipeline computes amplitudes.

Per (f, amp, wind) cell, we report:
  * n          — number of canon runs in the cell
  * A_IN       — median across runs of the per-run window-FFT amplitude (mm)
  * A_OUT      — same for OUT
  * OUT/IN     — median across runs of the per-run ratio
  * σ(OUT/IN)  — run-to-run standard deviation of the ratio

24 cells = 4 freqs × 3 amps × 2 winds. Grouped by amp tier in the
LaTeX output (visible group label per tier) for visual scanning.
Cohort = canon March-2026 cond4 lowrange, full panel, quality_ok.

Outputs:
    output/TABLES/data/ch04_plateau_values.csv       (render-shape data)
    output/TABLES/data/ch04_plateau_values.meta.json (provenance)
    output/TABLES/ch04_plateau_values.tex            (thesis include)
    analysis_scratch/plateau_values.csv              (audit-trail companion)

Caption text is read from FIGURE_CAPTIONS["ch04_plateau_values"] in
main_save_figures.py via output/.figure_captions.json.
"""

import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

BASE = (Path(__file__).resolve().parent.parent
        if "__file__" in globals() else Path.cwd())
sys.path.insert(0, str(BASE))
os.chdir(BASE)

from wavescripts.improved_data_loader import load_analysis_data, load_processed_dfs
from wavescripts.constants import c_group, HG


# ── Config ──────────────────────────────────────────────────────────────
FS               = 250.0
FFT_BAND_HZ      = 0.05

THESIS_FREQS     = [1.3, 1.4, 1.5, 1.6]
N_OFFSET         = 7
N_LENGTH         = 10

IN_PROBES        = ["9373/170", "9373/340"]
OUT_PROBE        = "12400/250"
PROBE_R_M        = {"9373/170": 9.373, "9373/340": 9.373, "12400/250": 12.400}
TANK_DEPTH_M     = HG.TANK_DEPTH_M

WINDS            = ["no", "full"]

AMP_TIERS = [
    (0.10, "A1", r"$A_1$"),
    (0.20, "A2", r"$A_2$"),
    (0.30, "A3", r"$A_3$"),
]

PROCESSED_DIRS = [
    Path("waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange"),
    Path("waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"),
]

THESIS_NAME  = "ch04_plateau_values"
CHAPTER      = "04"
SCRIPT_REL   = "analysis_scratch/plateau_values_table.py"

DATA_DIR     = BASE / "output" / "TABLES" / "data"
RENDER_CSV   = DATA_DIR / f"{THESIS_NAME}.csv"
META_JSON    = DATA_DIR / f"{THESIS_NAME}.meta.json"
OUT_TEX      = BASE / "output" / "TABLES" / f"{THESIS_NAME}.tex"
SCRATCH_CSV  = Path(__file__).parent / "plateau_values.csv"


# ── Helpers ─────────────────────────────────────────────────────────────
def fft_amp(seg, target_hz, fs=FS, band_hz=FFT_BAND_HZ):
    N = len(seg)
    if N < 4:
        return np.nan
    s = np.asarray(seg, dtype=float).copy()
    if np.isnan(s).any():
        idx = np.arange(N); good = ~np.isnan(s)
        if good.sum() < N * 0.9:
            return np.nan
        s = np.interp(idx, idx[good], s[good])
    freqs = np.fft.fftfreq(N, d=1.0 / fs)
    pos = freqs > 0
    pos_f = freqs[pos]
    mask = (pos_f >= target_hz - band_hz) & (pos_f <= target_hz + band_hz)
    if not mask.any():
        j = int(np.argmin(np.abs(pos_f - target_hz)))
    else:
        local = int(np.argmin(np.abs(pos_f[mask] - target_hz)))
        j = np.where(mask)[0][local]
    return float(2.0 * np.abs(np.fft.fft(s)[pos])[j] / N)


def get_eta(df, pos):
    for col in (f"eta_{pos}_interp", f"eta_{pos}"):
        if col in df.columns:
            return df[col].to_numpy(dtype=float)
    return None


def can_in(df):
    sigs = [get_eta(df, p) for p in IN_PROBES]
    sigs = [s for s in sigs if s is not None]
    if not sigs:
        return None
    nmin = min(len(s) for s in sigs)
    return np.nanmean(np.vstack([s[:nmin] for s in sigs]), axis=0)


def window_fft_amp(signal, target_hz, win_start_s, win_end_s):
    """FFT amplitude at target_hz over the window [win_start, win_end]."""
    n_lo = int(round(win_start_s * FS))
    n_hi = int(round(win_end_s   * FS))
    if n_hi > len(signal) or n_lo < 0 or n_hi - n_lo < 4:
        return np.nan
    return fft_amp(signal[n_lo:n_hi], target_hz)


# ── Load ────────────────────────────────────────────────────────────────
print("Loading meta + processed_dfs (canon March-2026 lowrange) …")
combined_meta, _, _, _ = load_analysis_data(*PROCESSED_DIRS, load_processed=False)
processed_dfs = load_processed_dfs(*PROCESSED_DIRS)

f_col = pd.to_numeric(combined_meta["WaveFrequencyInput [Hz]"], errors="coerce")
a_col = pd.to_numeric(combined_meta["WaveAmplitudeInput [Volt]"], errors="coerce")
mask = (
    f_col.between(THESIS_FREQS[0] - 0.02, THESIS_FREQS[-1] + 0.02)
    & a_col.between(0.05, 0.35)
    & (combined_meta["PanelCondition"] == "full")
    & combined_meta["WindCondition"].isin(WINDS)
    # Match ch05_damping_freq's quality gate (apply_experimental_filters
    # default): keep "ok" plus "probe_malfunction_secondary".
    & combined_meta["quality_flag"].isin(["ok", "probe_malfunction_secondary"])
)
sel = combined_meta[mask].copy()
# Single-probe dropout filter — match ch05_damping_freq
# (main_save_figures.py ~line 2207). K_t from a single IN probe can't
# exceed 1: a transmissive panel always damps. K_t,probe > 1 means that
# probe's amplitude registered below the OUT probe — a single-probe
# dropout (the 9373/170 probe is known to drop out at higher frequencies).
_kt_wall = sel[f"Probe {OUT_PROBE} Amplitude (FFT)"] / sel[f"Probe {IN_PROBES[0]} Amplitude (FFT)"]
_kt_far  = sel[f"Probe {OUT_PROBE} Amplitude (FFT)"] / sel[f"Probe {IN_PROBES[1]} Amplitude (FFT)"]
_dropout = (_kt_wall > 1.0) | (_kt_far > 1.0)
if _dropout.any():
    print(f"Dropping {_dropout.sum()} runs with K_t,probe > 1 (single-probe dropout):")
    for _, _r in sel[_dropout].iterrows():
        print(f"  Kt_wall={_kt_wall[_r.name]:.3f} Kt_far={_kt_far[_r.name]:.3f}  {_r['path'].split('/')[-1]}")
    sel = sel[~_dropout].copy()
print(f"\n{len(sel)} canon runs at full panel, quality ok.")


# ── Per-run window amplitudes ───────────────────────────────────────────
print("\nComputing window FFT amplitudes per run …")
records = []
for _, r in sel.iterrows():
    f_paddle = float(r["WaveFrequencyInput [Hz]"])
    amp_v    = float(r["WaveAmplitudeInput [Volt]"])
    wind     = r["WindCondition"]

    # Match this run to a thesis (f, amp) bin
    f_match = next((f for f in THESIS_FREQS
                    if abs(f - f_paddle) < 0.02), None)
    a_match = next((a for a, _, _ in AMP_TIERS
                    if abs(a - amp_v) < 0.01), None)
    if f_match is None or a_match is None:
        continue

    df = processed_dfs.get(r["path"])
    if df is None:
        continue
    sig_in  = can_in(df)
    sig_out = get_eta(df, OUT_PROBE)
    if sig_in is None or sig_out is None:
        continue

    # Probe-shifted windows
    t_arr_in  = PROBE_R_M[IN_PROBES[0]] / c_group(f_match, TANK_DEPTH_M)
    t_arr_out = PROBE_R_M[OUT_PROBE]    / c_group(f_match, TANK_DEPTH_M)
    win_in  = (t_arr_in  + N_OFFSET / f_match,
               t_arr_in  + (N_OFFSET + N_LENGTH) / f_match)
    win_out = (t_arr_out + N_OFFSET / f_match,
               t_arr_out + (N_OFFSET + N_LENGTH) / f_match)

    a_in  = window_fft_amp(sig_in,  f_match, *win_in)
    a_out = window_fft_amp(sig_out, f_match, *win_out)
    if not (np.isfinite(a_in) and np.isfinite(a_out) and a_in > 0):
        continue

    records.append({
        "freq_hz":     f_match,
        "amp_v":       a_match,
        "wind":        wind,
        "path":        Path(str(r["path"])).name,
        "A_in_mm":     a_in,
        "A_out_mm":    a_out,
        "OUT_IN":      a_out / a_in,
    })

per_run = pd.DataFrame(records)
print(f"   {len(per_run)} per-run records.")


# ── Aggregate per (f, amp, wind) ────────────────────────────────────────
agg = (
    per_run
    .groupby(["amp_v", "freq_hz", "wind"])
    .agg(
        n         = ("OUT_IN", "size"),
        A_in_mm   = ("A_in_mm",  "median"),
        A_out_mm  = ("A_out_mm", "median"),
        OUT_IN    = ("OUT_IN",   "median"),
        OUT_IN_sd = ("OUT_IN",   "std"),
    )
    .reset_index()
)
print(f"\n{len(agg)} aggregated cells:")
print(agg.round(4).to_string(index=False))

agg.to_csv(SCRATCH_CSV, index=False)
print(f"\n   audit CSV → {SCRATCH_CSV.relative_to(BASE)}")

# ── Reshape into render-shape (one row per output table line) ──────────
# Sort matches the original LaTeX block order: amp outer (A1, A2, A3),
# then within each block freq ascending, then wind=full before wind=uten
# (alphabetical: 'full' < 'uten', which is how the original sort_values
# ["freq_hz", "wind"] resolved it).
amps_in_order = [a for a, _, _ in AMP_TIERS]
agg = agg.sort_values(
    ["amp_v", "freq_hz", "wind"],
    key=lambda s: pd.Categorical(
        s, categories=amps_in_order, ordered=True
    ) if s.name == "amp_v" else s,
).reset_index(drop=True)

render_df = agg.rename(columns={
    "freq_hz":   "freq",
    "A_in_mm":   "A_in",
    "A_out_mm":  "A_out",
    "OUT_IN":    "Kt",
    "OUT_IN_sd": "sigma_Kt",
})[["amp_v", "freq", "wind", "n", "A_in", "A_out", "Kt", "sigma_Kt"]]

DATA_DIR.mkdir(parents=True, exist_ok=True)
render_df.to_csv(RENDER_CSV, index=False)
print(f"   render CSV → {RENDER_CSV.relative_to(BASE)}")


# ── Build provenance meta.json ──────────────────────────────────────────
# Caption text + caption_short are owned by main_save_tables.py — it patches
# meta.json's caption_short field after this script runs.
amplitude_summary = ", ".join(
    f"{a:.2f}V ({lbl.replace('$','')})" for a, _, lbl in AMP_TIERS
)

meta_payload = {
    "script":          SCRIPT_REL,
    "plot_type":       "plateau_values_table",
    "chapter":         CHAPTER,
    "caption_label":   f"tab:{THESIS_NAME}",
    "caption_short":   "",
    "sections": [
        {
            "title": "Method",
            "lines": [
                "per-run window FFT amplitudes at paddle frequency f, computed over",
                "probe-shifted windows:",
                "  win_IN  = [r_IN/c_g(f) + N_off/f,  r_IN/c_g(f) + (N_off+N_len)/f]",
                "  win_OUT = [r_OUT/c_g(f) + N_off/f, r_OUT/c_g(f) + (N_off+N_len)/f]",
                f"N_offset          : {N_OFFSET} periods",
                f"N_length          : {N_LENGTH} periods",
                f"r_IN              : {PROBE_R_M[IN_PROBES[0]]} m  ({', '.join(IN_PROBES)})",
                f"r_OUT             : {PROBE_R_M[OUT_PROBE]} m  ({OUT_PROBE})",
                f"tank depth h      : {TANK_DEPTH_M} m",
                f"FFT band          : ±{FFT_BAND_HZ} Hz around target f",
                "c_g dispersion    : full ω²=gk·tanh(kh) via wavescripts.constants.c_group",
            ],
        },
        {
            "title": "Aggregation",
            "lines": [
                "per (amp, freq, wind) cell:",
                "  n         = number of canon runs",
                "  A_IN      = median across runs of per-run window-FFT amplitude",
                "  A_OUT     = same for OUT",
                "  OUT/IN    = median across runs of per-run ratio (NOT median(A_OUT)/median(A_IN))",
                "  σ(OUT/IN) = run-to-run std of the per-run ratio",
            ],
        },
        {
            "title": "Inputs",
            "lines": [
                f"frequencies [Hz]  : {', '.join(f'{f:.1f}' for f in THESIS_FREQS)}",
                f"amplitudes        : {amplitude_summary}",
                "wind conditions   : no, full",
                "filter            : PanelCondition=full, quality_flag=ok, canon March-2026 cond4 lowrange",
            ],
        },
    ],
}

META_JSON.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
print(f"   meta JSON  → {META_JSON.relative_to(BASE)}")

print("\nDone.")
