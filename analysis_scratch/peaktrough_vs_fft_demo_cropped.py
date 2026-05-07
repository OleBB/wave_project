"""
Peak-trough vs FFT amplitude — pedagogical IN-probe time series, 4 frequencies,
*cropped to a shared 5-second window* (23.2 → 28.2 s).

Variant of analysis_scratch/peaktrough_vs_fft_demo.py with three changes:
  1. Shared x-range across all panels (5.0 s window, exactly the same number
     of cycles visible per frequency: 6.5 → 7 → 7.5 → 8 cycles).
  2. Taller figure (less whitespace per panel).
  3. Coarser grid (no minor ticks) and thinner amplitude lines.

Note on the 1.6 Hz panel: its analysis window opens at 23.32 s, so the leftmost
0.12 s of the displayed slice is just outside the window. The signal is the
continuous recording; the amplitude bands shown are still the canonical
pipeline values computed over the full analysis window.

Output:
    output/timeseries_exploration/peaktrough_vs_fft_in_4freq_01V_fullwind_cropped.pdf
    (+ .png)
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

# Shared display window (seconds) — same 5 s slice across all panels.
T0_S, T1_S = 23.2, 28.2

RUN_FOR_FREQ = {
    f: str(DATADIR / f"fullpanel-fullwind-amp0100-freq{int(f*1000):04d}-per240-depth580-mstop30-run1.csv")
    for f in FREQS
}

# Visual constants
YLIM_MM    = (-18.0, 18.0)
WIND_COLOR = WIND_COLOR_MAP["full"]
FFT_LINE   = "#1B7F3A"   # green  — paddle-only carrier amplitude
P99_LINE   = "#C8312D"   # red    — percentile (P99,5−P0,5)/2
PHI_LINE   = "#1F4E9D"   # blue   — phase-locked T/4, 3T/4 mean

apply_thesis_style()


# ─── Load once ───────────────────────────────────────────────────────────
print("Loading …")
meta, _, _, _ = load_analysis_data(str(TARGET_DIR), load_processed=False)
proc = load_processed_dfs(str(TARGET_DIR))


def load_one(f_hz: float) -> dict:
    csv = RUN_FOR_FREQ[f_hz]
    row = meta[meta["path"] == csv].iloc[0]
    df  = proc[csv]

    eta_col = f"eta_{IN_PROBE}_interp" if f"eta_{IN_PROBE}_interp" in df.columns else f"eta_{IN_PROBE}"
    eta_full = df[eta_col].to_numpy(dtype=float)
    t_full   = np.arange(len(eta_full)) / FS

    s_idx = int(row[f"Computed Probe {IN_PROBE} start"])
    e_idx = int(row[f"Computed Probe {IN_PROBE} end"])

    return {
        "f_hz":   f_hz,
        "t":      t_full,
        "eta":    eta_full,
        "ws":     s_idx / FS,
        "we":     e_idx / FS,
        "a_fft":  float(row[f"Probe {IN_PROBE} Amplitude (FFT)"]),
        "a_p99":  float(row[f"Probe {IN_PROBE} Amplitude"]),
        "a_phi":  float(row[f"Probe {IN_PROBE} Amplitude (phase) mean"]),
    }


runs = [load_one(f) for f in FREQS]
print()
print(f"  IN probe {IN_PROBE} · 0.1 V · fullwind (canon 20260327):")
print(f"  {'f [Hz]':>7}  {'A_FFT':>7}  {'A_p99':>7}  {'A_phi':>7}  {'inflasjon':>10}  cycles in 5 s")
for r in runs:
    inflate = (r["a_p99"] / r["a_fft"] - 1) * 100
    n_cyc   = (T1_S - T0_S) * r["f_hz"]
    print(f"  {r['f_hz']:>7.2f}  {r['a_fft']:>7.3f}  {r['a_p99']:>7.3f}  "
          f"{r['a_phi']:>7.3f}  {inflate:>+9.1f}%  {n_cyc:>5.2f}")


# ─── Plot ────────────────────────────────────────────────────────────────
def _draw_amplitude_bands(ax, a_fft: float, a_p99: float, a_phi: float):
    for y in (+a_fft, -a_fft):
        ax.axhline(y, color=FFT_LINE, lw=0.7, alpha=0.95, zorder=4)
    for y in (+a_p99, -a_p99):
        ax.axhline(y, color=P99_LINE, ls=(0, (4, 2)), lw=0.7, alpha=0.95, zorder=4)
    for y in (+a_phi, -a_phi):
        ax.axhline(y, color=PHI_LINE, ls=(0, (1, 2)), lw=0.7, alpha=0.95, zorder=4)


def _badge(ax, f_hz: float, a_fft: float, a_p99: float, a_phi: float):
    p99_pct = (a_p99 / a_fft - 1.0) * 100.0
    phi_pct = (a_phi / a_fft - 1.0) * 100.0
    f_no = f"{f_hz:g}".replace(".", ",")
    txt = (rf"$f$ = {f_no} Hz" "\n"
           rf"$A_{{\mathrm{{FFT}}}}$ = {a_fft:.2f} mm" "\n"
           rf"$A_{{\mathrm{{p}}}}$ = {a_p99:.2f} mm  ({p99_pct:+.0f} %)" "\n"
           rf"$A_{{\phi}}$ = {a_phi:.2f} mm  ({phi_pct:+.0f} %)")
    ax.text(
        0.995, 0.96, txt,
        transform=ax.transAxes, va="top", ha="right",
        fontsize=8.5, color="#222",
        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                  ec="#888", lw=0.5, alpha=0.95),
    )


fig, axes = plt.subplots(
    len(FREQS), 1,
    figsize=(6.27, 9.6),       # taller — was 8.0
    sharex=True,
    gridspec_kw={"hspace": 0.18},
)

for ax, r in zip(axes, runs):
    m = (r["t"] >= T0_S) & (r["t"] <= T1_S)
    ax.plot(r["t"][m], r["eta"][m], color=WIND_COLOR, lw=0.9)
    ax.axhline(0, color="#888", lw=0.5, alpha=0.6)
    _draw_amplitude_bands(ax, r["a_fft"], r["a_p99"], r["a_phi"])
    ax.set_xlim(T0_S, T1_S)
    ax.set_ylim(YLIM_MM)
    # Coarser grid: major only, no minor ticks/grid.
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.yaxis.set_major_locator(MultipleLocator(5.0))
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.grid(True, which="major", alpha=0.30)
    _badge(ax, r["f_hz"], r["a_fft"], r["a_p99"], r["a_phi"])

axes[-1].set_xlabel("Tid [s]")

# η label above the top axis
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
    Line2D([0], [0], color=FFT_LINE, lw=0.9,
           label=r"$\pm A_{\mathrm{FFT}}$"),
    Line2D([0], [0], color=P99_LINE, lw=0.9, ls=(0, (4, 2)),
           label=r"$\pm A_{\mathrm{p}}$"),
    Line2D([0], [0], color=PHI_LINE, lw=0.9, ls=(0, (1, 2)),
           label=r"$\pm A_{\phi}$"),
]
# Place legend in the empty horizontal band ABOVE the top axis (the η label
# sits at top-left only, so there's clear space to the right of it).
fig.legend(handles=legend_handles, loc="lower right", ncol=3,
           frameon=True, framealpha=0.95, fontsize=7.5,
           handlelength=2.2, columnspacing=1.6,
           bbox_to_anchor=(1.0, 1.005),
           bbox_transform=axes[0].transAxes)

out_pdf = OUTDIR / "peaktrough_vs_fft_in_4freq_01V_fullwind_cropped.pdf"
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_pdf.with_suffix(".png"), dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"\n  → {out_pdf.relative_to(BASE)}  (+ .png)")
print("Done.")
