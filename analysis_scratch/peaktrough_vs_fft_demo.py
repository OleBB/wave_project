"""
Peak-trough vs FFT amplitude — pedagogical time-series figure (IN probe only).

Worst-case demonstration: 0.1 V fullwind across the four thesis frequencies
(1.3, 1.4, 1.5, 1.6 Hz). At the IN probe (9373/170, fully exposed to wind)
the percentile (peak-to-trough/2) amplitude is inflated by wind ripple riding
on the paddle wave. Same analysis window, same data, two amplitude
estimators — the gap IS the wind contamination.

Shows ONLY the chosen analysis window slice (no macro, no zoom). The window
extents and both amplitude values are pulled directly from meta — pipeline
is the source of truth.

Layout:
    [1.3 Hz panel]  IN-probe η(t) inside its analysis window
    [1.4 Hz panel]  same
    [1.5 Hz panel]  same
    [1.6 Hz panel]  same

Overlays (each panel):
    solid green  : ±A_FFT  (pipeline `Probe {pos} Amplitude (FFT)`)
    dashed red   : ±A_p2t  (pipeline `Probe {pos} Amplitude` — (P99.5−P0.5)/2
                            on the same window)
    text badge   : f, A_FFT, A_p2t, percent inflation

Output:
    output/timeseries_exploration/peaktrough_vs_fft_in_4freq_01V_fullwind.pdf  (+ .png)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

from wavescripts.improved_data_loader import load_analysis_data, load_processed_dfs
from wavescripts.plot_utils import apply_thesis_style, WIND_COLOR_MAP

FS = 250.0
BASE = Path("/Users/ole/Kodevik/wave_project")
OUTDIR = BASE / "output/timeseries_exploration"
OUTDIR.mkdir(parents=True, exist_ok=True)

TARGET_DIR = BASE / "waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"
DATADIR    = BASE / "wavedata/20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"

IN_PROBE = "9373/170"
AMP_V    = 0.1
FREQS    = (1.3, 1.4, 1.5, 1.6)

# Canon run filename per frequency (per240, fullwind, 0.1 V, mstop30, run1).
RUN_FOR_FREQ = {
    f: str(DATADIR / f"fullpanel-fullwind-amp0100-freq{int(f*1000):04d}-per240-depth580-mstop30-run1.csv")
    for f in FREQS
}

# Visual constants
YLIM_MM    = (-18.0, 18.0)
WIND_COLOR = WIND_COLOR_MAP["full"]   # red — fullwind
FFT_LINE   = "#1B7F3A"                # green — paddle-only carrier amplitude
P2T_LINE   = "#C8312D"                # red   — peak-trough amplitude

apply_thesis_style()


# ─── Load once ───────────────────────────────────────────────────────────
print("Loading …")
meta, _, _, _ = load_analysis_data(str(TARGET_DIR), load_processed=False)
proc = load_processed_dfs(str(TARGET_DIR))


def load_one(f_hz: float) -> dict:
    csv = RUN_FOR_FREQ[f_hz]
    row = meta[meta["path"] == csv].iloc[0]
    df  = proc[csv]

    s_idx = int(row[f"Computed Probe {IN_PROBE} start"])
    e_idx = int(row[f"Computed Probe {IN_PROBE} end"])
    eta_col = f"eta_{IN_PROBE}_interp" if f"eta_{IN_PROBE}_interp" in df.columns else f"eta_{IN_PROBE}"
    eta = df[eta_col].iloc[s_idx:e_idx + 1].to_numpy(dtype=float)
    t   = np.arange(s_idx, e_idx + 1) / FS

    return {
        "f_hz":  f_hz,
        "t":     t,
        "eta":   eta,
        "ws":    s_idx / FS,
        "we":    e_idx / FS,
        "a_fft": float(row[f"Probe {IN_PROBE} Amplitude (FFT)"]),
        "a_p2t": float(row[f"Probe {IN_PROBE} Amplitude"]),
    }


runs = [load_one(f) for f in FREQS]
print()
print(f"  IN probe {IN_PROBE} · 0.1 V · fullwind (canon 20260327):")
print(f"  {'f [Hz]':>7}  {'A_FFT [mm]':>11}  {'A_p2t [mm]':>11}  {'inflasjon':>10}  window [s]")
for r in runs:
    inflate = (r["a_p2t"] / r["a_fft"] - 1) * 100
    print(f"  {r['f_hz']:>7.2f}  {r['a_fft']:>11.3f}  {r['a_p2t']:>11.3f}  "
          f"{inflate:>+9.1f}%  [{r['ws']:.2f}, {r['we']:.2f}]  ({r['we']-r['ws']:.2f} s)")


# ─── Plot ────────────────────────────────────────────────────────────────
def _apply_ticks(ax, *, x_major, x_minor, y_major, y_minor):
    ax.xaxis.set_major_locator(MultipleLocator(x_major))
    ax.xaxis.set_minor_locator(MultipleLocator(x_minor))
    ax.yaxis.set_major_locator(MultipleLocator(y_major))
    ax.yaxis.set_minor_locator(MultipleLocator(y_minor))
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(True, which="minor", alpha=0.12)


def _draw_amplitude_bands(ax, a_fft: float, a_p2t: float):
    for y in (+a_fft, -a_fft):
        ax.axhline(y, color=FFT_LINE, lw=1.2, alpha=0.95, zorder=4)
    for y in (+a_p2t, -a_p2t):
        ax.axhline(y, color=P2T_LINE, ls=(0, (4, 2)), lw=1.2, alpha=0.95, zorder=4)


def _badge(ax, f_hz: float, a_fft: float, a_p2t: float):
    inflation = (a_p2t / a_fft - 1.0) * 100.0
    f_no = f"{f_hz:g}".replace(".", ",")
    txt = (rf"$f$ = {f_no} Hz" "\n"
           rf"$A_{{\mathrm{{FFT}}}}$ = {a_fft:.2f} mm" "\n"
           rf"$A_{{\mathrm{{topp\text{{-}}bunn}}}}$ = {a_p2t:.2f} mm" "\n"
           rf"inflasjon = {inflation:+.0f} \%")
    ax.text(
        0.995, 0.96, txt,
        transform=ax.transAxes, va="top", ha="right",
        fontsize=8.5, color="#222",
        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                  ec="#888", lw=0.5, alpha=0.95),
    )


fig = plt.figure(figsize=(6.27, 8.0))
gs  = fig.add_gridspec(len(FREQS), 1, hspace=0.45)
axes = [fig.add_subplot(gs[i, 0]) for i in range(len(FREQS))]

for ax, r in zip(axes, runs):
    ax.plot(r["t"], r["eta"], color=WIND_COLOR, lw=0.8)
    ax.scatter(r["t"][::6], r["eta"][::6], s=5, color=WIND_COLOR,
               alpha=0.45, edgecolor="none")
    ax.axhline(0, color="#888", lw=0.5, alpha=0.6)
    _draw_amplitude_bands(ax, r["a_fft"], r["a_p2t"])
    ax.set_xlim(r["ws"], r["we"])
    ax.set_ylim(YLIM_MM)
    _apply_ticks(ax, x_major=1.0, x_minor=0.2, y_major=5.0, y_minor=1.0)
    _badge(ax, r["f_hz"], r["a_fft"], r["a_p2t"])

axes[-1].set_xlabel("Tid [s]")

# η label above the top axis (matches inspirational_timeseries layout)
axes[0].set_ylabel(r"$\eta$ [mm]",
                   rotation=0, ha="left", va="bottom", fontsize=10)
fig.canvas.draw()
_renderer = fig.canvas.get_renderer()
_ticks = [tk for tk in axes[0].yaxis.get_ticklabels()
          if tk.get_visible() and tk.get_text().strip()]
if _ticks:
    _left_disp = min(tk.get_window_extent(renderer=_renderer).x0 for tk in _ticks)
    _x_axes = axes[0].transAxes.inverted().transform((_left_disp, 0))[0]
    axes[0].yaxis.set_label_coords(_x_axes, 1.05)

legend_handles = [
    Line2D([0], [0], color=FFT_LINE, lw=1.4,
           label=r"$\pm A_{\mathrm{FFT}}$  (kun paddelfrekvens)"),
    Line2D([0], [0], color=P2T_LINE, lw=1.4, ls=(0, (4, 2)),
           label=r"$\pm A_{\mathrm{topp\text{-}bunn}}$  ((P$_{99,5}$$-$P$_{0,5}$)/2 over analysevinduet)"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=2,
           frameon=True, framealpha=0.95, fontsize=8.5,
           bbox_to_anchor=(0.5, -0.015))

out_pdf = OUTDIR / "peaktrough_vs_fft_in_4freq_01V_fullwind.pdf"
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_pdf.with_suffix(".png"), dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"\n  → {out_pdf.relative_to(BASE)}  (+ .png)")
print("Done.")
