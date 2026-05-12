"""K_t vs A_in per-run scatter — every canon March-2026 run individually.

Mirrors plateau_values_table.py's filter and window-FFT logic exactly,
but scatters each individual run instead of aggregating to medians.

Blue  = no-wind runs
Red   = full-wind runs
"""
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = (Path(__file__).resolve().parent.parent
        if "__file__" in globals() else Path.cwd())
sys.path.insert(0, str(BASE))
os.chdir(BASE)

from wavescripts.improved_data_loader import load_analysis_data, load_processed_dfs
from wavescripts.constants import c_group, HG


# ── Config (identical to plateau_values_table.py) ───────────────────────
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
AMPS             = [0.10, 0.20, 0.30]
PROCESSED_DIRS = [
    Path("waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange"),
    Path("waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"),
]


# ── Helpers (copied from plateau_values_table.py) ───────────────────────
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

    f_match = next((f for f in THESIS_FREQS if abs(f - f_paddle) < 0.02), None)
    a_match = next((a for a in AMPS if abs(a - amp_v) < 0.01), None)
    if f_match is None or a_match is None:
        continue

    df = processed_dfs.get(r["path"])
    if df is None:
        continue
    sig_in  = can_in(df)
    sig_out = get_eta(df, OUT_PROBE)
    if sig_in is None or sig_out is None:
        continue

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
        "freq_hz":  f_match,
        "amp_v":    a_match,
        "wind":     wind,
        "path":     Path(str(r["path"])).name,
        "A_in_mm":  a_in,
        "A_out_mm": a_out,
        "Kt":       a_out / a_in,
    })

per_run = pd.DataFrame(records)
print(f"   {len(per_run)} per-run records.")
print(f"   no-wind:   {(per_run['wind']=='no').sum()}")
print(f"   full-wind: {(per_run['wind']=='full').sum()}")

OUT_CSV = Path(__file__).parent / "scatter_kt_vs_ain_per_run.csv"
per_run.to_csv(OUT_CSV, index=False)
print(f"   per-run CSV → {OUT_CSV.relative_to(BASE)}")


# ── Scatter ─────────────────────────────────────────────────────────────
COLOR = {"no": "tab:blue", "full": "tab:red"}
MARKER = {0.10: "o", 0.20: "s", 0.30: "^"}   # circle / square / triangle-up
AMP_LABEL = {0.10: "A1 (0.10 V)", 0.20: "A2 (0.20 V)", 0.30: "A3 (0.30 V)"}

fig, ax = plt.subplots(figsize=(8.0, 5.5))

for wind in ("no", "full"):
    for amp in (0.10, 0.20, 0.30):
        sub = per_run[(per_run["wind"] == wind) & (per_run["amp_v"] == amp)]
        if sub.empty:
            continue
        ax.scatter(
            sub["A_in_mm"], sub["Kt"],
            c=COLOR[wind],
            marker=MARKER[amp],
            s=40,
            alpha=0.7,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )

# Two-group legend: wind (color) + amplitude (marker)
from matplotlib.lines import Line2D
wind_handles = [
    Line2D([0], [0], color="w", marker="o", markerfacecolor=COLOR["no"],
           markeredgecolor="white", markersize=8,
           label=f"Uten vind  (n={int((per_run['wind']=='no').sum())})"),
    Line2D([0], [0], color="w", marker="o", markerfacecolor=COLOR["full"],
           markeredgecolor="white", markersize=8,
           label=f"Full vind  (n={int((per_run['wind']=='full').sum())})"),
]
amp_handles = [
    Line2D([0], [0], color="w", marker=MARKER[a], markerfacecolor="0.4",
           markeredgecolor="white", markersize=8, label=AMP_LABEL[a])
    for a in (0.10, 0.20, 0.30)
]
leg1 = ax.legend(handles=wind_handles, title="Vind",
                 loc="lower right", frameon=True)
ax.add_artist(leg1)
ax.legend(handles=amp_handles, title="Amplitude",
          loc="lower center", frameon=True)

ax.set_xlabel(r"$A_\mathrm{Inn}$ [mm]  (per-run window-FFT amplitude on IN side)")
ax.set_ylabel(r"$K_t = A_\mathrm{Ut} / A_\mathrm{Inn}$")
ax.set_title(
    "Transmisjonskoeffisient mot innkommende amplitude — alle målte canon-runs\n"
    f"(March-2026 lowrange fullpanel quality_ok, N={len(per_run)} runs)"
)
ax.grid(True, alpha=0.3)

# Light vertical guides at the nominal amplitude bands
for x, lbl in [(7.7, "A1 ~7.5 mm"), (15.1, "A2 ~15 mm"), (22.3, "A3 ~22 mm")]:
    ax.axvline(x, color="0.8", linestyle=":", linewidth=0.8, zorder=1)

plt.tight_layout()
OUT_PDF = Path(__file__).parent / "scatter_kt_vs_ain_per_run.pdf"
plt.savefig(OUT_PDF, bbox_inches="tight")
plt.savefig(OUT_PDF.with_suffix(".png"), dpi=150, bbox_inches="tight")
print(f"\nsaved: {OUT_PDF}")
print(f"saved: {OUT_PDF.with_suffix('.png')}")
