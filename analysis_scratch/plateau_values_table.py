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
LaTeX output for visual scanning. Cohort = canon March-2026 cond4
lowrange, full panel, quality_ok.

Outputs:
    output/TABLES/ch04_plateau_values.tex
    analysis_scratch/plateau_values.csv
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import os
os.chdir(BASE)

from wavescripts.improved_data_loader import load_analysis_data, load_processed_dfs
from wavescripts.constants import c_group, HG
from wavescripts.plot_utils import _lookup_central_caption


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
WIND_LABEL = {"no": "uten", "full": "full"}

PROCESSED_DIRS = [
    Path("waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange"),
    Path("waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"),
]

THESIS_NAME = "ch04_plateau_values"
OUT_TEX = BASE / "output" / "TABLES" / f"{THESIS_NAME}.tex"
OUT_CSV = Path(__file__).parent / "plateau_values.csv"
CHAPTER = "04"


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
    & (combined_meta["quality_flag"] == "ok")
)
sel = combined_meta[mask].copy()
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

agg.to_csv(OUT_CSV, index=False)
print(f"\n   CSV → {OUT_CSV.relative_to(BASE)}")


# ── LaTeX table ─────────────────────────────────────────────────────────
def fmt_num(v, decimals=2):
    if pd.isna(v):
        return r"\textendash"
    return rf"$\num{{{v:.{decimals}f}}}$"


# Render: blocks per amp, each with 8 rows (4 freqs × 2 winds).
amp_label_lookup = {a: lbl for a, _, lbl in AMP_TIERS}
body_lines = []
amps_in_order = [a for a, _, _ in AMP_TIERS]
for i, a in enumerate(amps_in_order):
    body_lines.append(
        f"    \\multicolumn{{7}}{{l}}{{\\textbf{{{amp_label_lookup[a]}}} "
        f"($V = {a:.2f}$ V)}} \\\\"
    )
    sub = agg[agg["amp_v"] == a].sort_values(["freq_hz", "wind"])
    for _, row in sub.iterrows():
        cells = [
            rf"$\num{{{row['freq_hz']:.1f}}}$",
            WIND_LABEL[row["wind"]],
            rf"$\num{{{int(row['n'])}}}$",
            fmt_num(row["A_in_mm"],   2),
            fmt_num(row["A_out_mm"],  2),
            fmt_num(row["OUT_IN"],    3),
            fmt_num(row["OUT_IN_sd"], 3) if pd.notna(row["OUT_IN_sd"]) else r"\textendash",
        ]
        body_lines.append("    " + " & ".join(cells) + r" \\")
    if i != len(amps_in_order) - 1:
        body_lines.append("    \\midrule")


caption_full  = _lookup_central_caption(THESIS_NAME, kind="full")
caption_short = _lookup_central_caption(THESIS_NAME, kind="short")
if caption_full and caption_short:
    caption_block = (
        f"  \\caption[{caption_short}]{{\n"
        f"    {caption_full}\n"
        f"  }}\n"
    )
elif caption_full:
    caption_block = f"  \\caption{{\n    {caption_full}\n  }}\n"
else:
    caption_block = "  \\caption{\n    % TODO: write caption\n  }\n"

from datetime import datetime as _dt
immutable = "\n".join([
    "%! TEX root = ../main.tex",
    "% ==============================================================",
    "% IMMUTABLE — generated automatically, do not edit this block",
    "%",
    "% — Provenance ───────────────────────────────────────────────────",
    "%   script            : analysis_scratch/plateau_values_table.py",
    "%   plot_type         : plateau_values_table",
    f"%   chapter           : {CHAPTER}",
    f"%   generated_at      : {_dt.now().isoformat(timespec='seconds')}",
    f"%   caption_label     : tab:{THESIS_NAME}",
    f"%   caption_short     : {caption_short}",
    "%",
    "% — Method ────────────────────────────────────────────────────",
    "%   per-run window FFT amplitudes at paddle frequency f, computed over",
    "%   probe-shifted windows:",
    "%     win_IN  = [r_IN/c_g(f) + N_off/f,  r_IN/c_g(f) + (N_off+N_len)/f]",
    "%     win_OUT = [r_OUT/c_g(f) + N_off/f, r_OUT/c_g(f) + (N_off+N_len)/f]",
    f"%   N_offset          : {N_OFFSET} periods",
    f"%   N_length          : {N_LENGTH} periods",
    f"%   r_IN              : {PROBE_R_M[IN_PROBES[0]]} m  ({', '.join(IN_PROBES)})",
    f"%   r_OUT             : {PROBE_R_M[OUT_PROBE]} m  ({OUT_PROBE})",
    f"%   tank depth h      : {TANK_DEPTH_M} m",
    f"%   FFT band          : ±{FFT_BAND_HZ} Hz around target f",
    "%   c_g dispersion    : full ω²=gk·tanh(kh) via wavescripts.constants.c_group",
    "%",
    "% — Aggregation ───────────────────────────────────────────────",
    "%   per (amp, freq, wind) cell:",
    "%     n         = number of canon runs",
    "%     A_IN      = median across runs of per-run window-FFT amplitude",
    "%     A_OUT     = same for OUT",
    "%     OUT/IN    = median across runs of per-run ratio (NOT median(A_OUT)/median(A_IN))",
    "%     σ(OUT/IN) = run-to-run std of the per-run ratio",
    "%",
    "% — Inputs ────────────────────────────────────────────────────",
    f"%   frequencies [Hz]  : {', '.join(f'{f:.1f}' for f in THESIS_FREQS)}",
    f"%   amplitudes        : {', '.join(f'{a:.2f}V ({lbl})' for a, _, lbl in AMP_TIERS).replace('$','')}",
    "%   wind conditions   : no, full",
    "%   filter            : PanelCondition=full, quality_flag=ok, canon March-2026 cond4 lowrange",
    "%",
    "% ── end immutable block ─────────────────────────────────────────",
])


table_body = (
    "\\begin{table}[hbt]\n"
    "  \\centering\n"
    + caption_block
    + f"  \\label{{tab:{THESIS_NAME}}}\n"
    "  \\begin{tabular}{ccccccc}\n"
    "    \\toprule\n"
    "    $f$ [\\unit{\\hertz}] &\n"
    "      vind &\n"
    "      $n$ &\n"
    "      $A_\\mathrm{Inn}$ [\\unit{\\milli\\meter}] &\n"
    "      $A_\\mathrm{Ut}$ [\\unit{\\milli\\meter}] &\n"
    "      $K_t$ &\n"
    "      $\\sigma (K_t)$ \\\\\n"
    "    \\midrule\n"
    + "\n".join(body_lines) + "\n"
    "    \\bottomrule\n"
    "  \\end{tabular}\n"
    "\\end{table}\n"
)

OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
# Note: header has 7 columns (f, vind, n, A_IN, A_OUT, OUT/IN, σ).
# But the multicolumn-amp rows above use \multicolumn{7}{l}{...} which
# spans the same 7 → matching column count.
OUT_TEX.write_text(immutable + "\n" + table_body, encoding="utf-8")
print(f"   TEX → {OUT_TEX.relative_to(BASE)}")
