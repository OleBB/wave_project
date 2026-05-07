"""
CH04 §5 — Three amplitude estimators on the IN probe, $A_1$ fullwind, four
frequencies. Pedagogical figure that motivates FFT amplitude as the primary
metric.

Each panel shows IN probe η(t) inside a shared 5-second slice of the
analysis window for one of the four thesis frequencies (1.3, 1.4, 1.5,
1.6 Hz). Three horizontal amplitude bands per panel:
  ── solid green   : ±A_FFT  (paddle-only, nearest-bin FFT)
  -- red dashed    : ±A_p    ((P99,5−P0,5)/2 percentile)
  ·· blue dotted   : ±A_φ    (phase-locked T/4, 3T/4 sample mean)
A horizontal value-legend above each axis carries the marker + value +
inflation percent vs A_FFT for that estimator.

The 5 s slice is in absolute recording time (23.2 → 28.2 s) — caption
calls this "tid i sekunder fra start" (start = start of the recording).
The amplitude values shown are the canonical pipeline values computed
over each run's full 10-period analysis window — NOT recomputed on the
visible 5 s slice.

Note on the 1.6 Hz panel: its 10-period analysis window opens at 23.32 s,
so the leftmost ~0.12 s of the visible slice (about 1 sample on screen)
is just outside that window. Signal continuity is preserved.

Output:
    output/FIGURES/ch04_amp_methods_a1_fullwind.pdf
    output/TEXFIGU/ch04_amp_methods_a1_fullwind.tex
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
import wavescripts.plot_utils as pu

FS = 250.0
BASE = Path("/Users/ole/Kodevik/wave_project")
FIGURES_DIR = BASE / "output" / "FIGURES"
TEXFIGU_DIR = BASE / "output" / "TEXFIGU"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TEXFIGU_DIR.mkdir(parents=True, exist_ok=True)

FIGURE_NAME = "ch04_amp_methods_a1_fullwind"

TARGET_DIR = BASE / "waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"
DATADIR    = BASE / "wavedata/20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"

IN_PROBE = "9373/170"
AMP_V    = 0.1
FREQS    = (1.3, 1.4, 1.5, 1.6)

# Absolute-time slice (seconds) — same 5 s of the recording across all panels.
# X-axis displays absolute recording time (caption: "fra start", i.e. from
# the start of the recording).
T0_ABS, T1_ABS = 23.2, 28.2
WINDOW_S = T1_ABS - T0_ABS

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

# Match thesis body font — same registration as inspirational_timeseries.py
from matplotlib import font_manager as _fm
_NCM_DIR = "/usr/local/texlive/2025/texmf-dist/fonts/opentype/public/newcomputermodern"
for _fname in ("NewCM10-Regular.otf", "NewCM10-Bold.otf",
               "NewCM10-Italic.otf", "NewCM10-BoldItalic.otf",
               "NewCMMath-Regular.otf"):
    try:
        _fm.fontManager.addfont(f"{_NCM_DIR}/{_fname}")
    except Exception as _e:
        print(f"   warn: could not register {_fname}: {_e}")
plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["NewComputerModern10", "DejaVu Serif"],
    "mathtext.fontset": "cm",
})

pu.ACTIVE_DATASETS = [p.name for p in
                     sorted(BASE.glob("waveprocessed/PROCESSED-*"))]
pu.TEXFIGU_DIR = TEXFIGU_DIR
pu.FIGURES_DIR = FIGURES_DIR


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
        "row":    row,
        "t":      t_full,
        "eta":    eta_full,
        "ws":     s_idx / FS,
        "we":     e_idx / FS,
        "a_fft":  float(row[f"Probe {IN_PROBE} Amplitude (FFT)"]),
        "a_p99":  float(row[f"Probe {IN_PROBE} Amplitude"]),
        "a_phi":  float(row[f"Probe {IN_PROBE} Amplitude (phase) mean"]),
        "phi_n":  int(row[f"Probe {IN_PROBE} Amplitude (phase) n"]),
    }


runs = [load_one(f) for f in FREQS]
print()
print(f"  IN probe {IN_PROBE} · A_1 (0.1 V) · fullwind (canon 20260327):")
print(f"  {'f [Hz]':>7}  {'A_FFT':>7}  {'A_p':>7}  {'A_phi':>7}  {'cycles in 5 s':>14}")
for r in runs:
    n_cyc = WINDOW_S * r["f_hz"]
    print(f"  {r['f_hz']:>7.2f}  {r['a_fft']:>7.3f}  {r['a_p99']:>7.3f}  "
          f"{r['a_phi']:>7.3f}  {n_cyc:>14.2f}")


# ─── Plot ────────────────────────────────────────────────────────────────
def _draw_amplitude_bands(ax, a_fft: float, a_p99: float, a_phi: float):
    for y in (+a_fft, -a_fft):
        ax.axhline(y, color=FFT_LINE, lw=0.7, alpha=0.95, zorder=4)
    for y in (+a_p99, -a_p99):
        ax.axhline(y, color=P99_LINE, ls=(0, (4, 2)), lw=0.7, alpha=0.95, zorder=4)
    for y in (+a_phi, -a_phi):
        ax.axhline(y, color=PHI_LINE, ls=(0, (1, 2)), lw=0.7, alpha=0.95, zorder=4)


def _freq_badge(ax, f_hz: float):
    f_no = f"{f_hz:g}".replace(".", ",")
    ax.text(
        0.995, 0.96, rf"$f$ = {f_no} Hz",
        transform=ax.transAxes, va="top", ha="right",
        fontsize=8.5, color="#222",
        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                  ec="#888", lw=0.5, alpha=0.95),
    )


def _value_legend(ax, a_fft: float, a_p99: float, a_phi: float):
    p99_pct = (a_p99 / a_fft - 1.0) * 100.0
    phi_pct = (a_phi / a_fft - 1.0) * 100.0
    handles = [
        Line2D([0], [0], color=FFT_LINE, lw=0.9,
               label=rf"$\pm A_{{\mathrm{{FFT}}}}$ = {a_fft:.2f} mm"),
        Line2D([0], [0], color=P99_LINE, lw=0.9, ls=(0, (4, 2)),
               label=rf"$\pm A_{{\mathrm{{p}}}}$ = {a_p99:.2f} mm "
                     rf"({p99_pct:+.0f} %)"),
        Line2D([0], [0], color=PHI_LINE, lw=0.9, ls=(0, (1, 2)),
               label=rf"$\pm A_{{\phi}}$ = {a_phi:.2f} mm "
                     rf"({phi_pct:+.0f} %)"),
    ]
    ax.legend(handles=handles, loc="lower left",
              bbox_to_anchor=(0.0, 1.005),
              ncol=3, frameon=False, fontsize=7.5,
              handlelength=2.0, columnspacing=1.4,
              handletextpad=0.4, borderpad=0.0)


fig, axes = plt.subplots(
    len(FREQS), 1,
    figsize=(6.27, 9.6),
    sharex=True,
    gridspec_kw={"hspace": 0.32},
)

for ax, r in zip(axes, runs):
    m = (r["t"] >= T0_ABS) & (r["t"] <= T1_ABS)
    # x-axis = absolute recording time ("fra start")
    ax.plot(r["t"][m], r["eta"][m], color=WIND_COLOR, lw=0.9)
    ax.axhline(0, color="#888", lw=0.5, alpha=0.6)
    _draw_amplitude_bands(ax, r["a_fft"], r["a_p99"], r["a_phi"])
    ax.set_xlim(T0_ABS, T1_ABS)
    ax.set_ylim(YLIM_MM)
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.yaxis.set_major_locator(MultipleLocator(5.0))
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.grid(True, which="major", alpha=0.30)
    _value_legend(ax, r["a_fft"], r["a_p99"], r["a_phi"])
    _freq_badge(ax, r["f_hz"])

axes[-1].set_xlabel("Tid [s]")

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

out_pdf = FIGURES_DIR / f"{FIGURE_NAME}.pdf"
fig.savefig(out_pdf)
plt.close(fig)
print(f"\n  → {out_pdf.relative_to(BASE)}")


# ─── TEXFIGU stub ────────────────────────────────────────────────────────
def _f(x, n=4):
    try:
        return f"{float(x):.{n}f}"
    except (TypeError, ValueError):
        return "NA"


def _per_freq_stats(runs):
    out = {}
    for r in runs:
        f_hz = r["f_hz"]
        tag = f"{int(f_hz*10):02d}"   # 13, 14, 15, 16
        out[f"f{tag}_A_FFT_mm"]     = _f(r["a_fft"], 3)
        out[f"f{tag}_A_p_mm"]       = _f(r["a_p99"], 3)
        out[f"f{tag}_A_phi_mm"]     = _f(r["a_phi"], 3)
        out[f"f{tag}_p_inflate_pct"]   = _f((r["a_p99"]/r["a_fft"] - 1) * 100, 1)
        out[f"f{tag}_phi_inflate_pct"] = _f((r["a_phi"]/r["a_fft"] - 1) * 100, 1)
        out[f"f{tag}_window_abs_s"] = f"[{r['ws']:.3f}, {r['we']:.3f}]"
        out[f"f{tag}_phi_n_cycles"] = str(r["phi_n"])
    out["display_window_abs_s"] = f"[{T0_ABS:.2f}, {T1_ABS:.2f}]"
    out["probe"]                = IN_PROBE
    out["amp_input_V"]          = _f(AMP_V, 2)
    out["wind_condition"]       = "fullwind"
    out["sampling_rate_Hz"]     = _f(FS, 1)
    return out


_meta = pu.build_fig_meta(
    {
        "filters": {
            "PanelCondition":            "full",
            "WaveAmplitudeInput [Volt]": AMP_V,
            "WindCondition":             "full",
            "WaveFrequencyInput [Hz]":   list(FREQS),
            "quality_flag":              "ok",
        },
        "plotting": {
            "figure_name": FIGURE_NAME,
        },
    },
    chapter="04",
    extra={"script": "analysis_scratch/peaktrough_vs_fft_demo_cropped.py"},
    computed_in="analysis_scratch/peaktrough_vs_fft_demo_cropped.py "
                "(per-freq IN-probe time-series render)",
    data_class="DELEG",
    findings_doc="memory/methodology_window_mean_baseline.md",
    fft_window_hz=0.1,
    extra_params=(
        f"runs=4 (1.3/1.4/1.5/1.6 Hz, fullpanel/fullwind/A_1/per240/mstop30, "
        f"canon dataset 20260327-...lowrange). "
        f"probe=IN:{IN_PROBE} only. "
        f"display window = absolute [{T0_ABS:.2f}, {T1_ABS:.2f}] s "
        f"(x-axis shows absolute recording time, 'fra start'). "
        f"per-panel amplitude values (A_FFT, A_p, A_phi) are pulled from "
        f"each run's meta.json — computed over each run's own 10-period "
        f"analysis window (Computed Probe {IN_PROBE} start/end), NOT over "
        f"the visible 5 s slice. "
        f"A_FFT = nearest-bin FFT magnitude at f_paddle (0.1 Hz half-window). "
        f"A_p   = (P99.5 − P0.5)/2 over the analysis window. "
        f"A_phi = mean of |signed| samples at u+T/4 and u+3T/4 across the "
        f"detected upcrossings inside the analysis window. "
        f"Bands: ±A_FFT (solid green), ±A_p (dashed red), "
        f"±A_phi (dotted blue). "
        f"Per-panel value legend above the axis shows marker + value + "
        f"percent inflation vs A_FFT. f badge in upper-right corner. "
        f"Typeset in NewComputerModern10 to match thesis body font."
    ),
    extra_stats=_per_freq_stats(runs),
)

pu.write_figure_stub(_meta, plot_type="amp_methods_demo",
                     subfig_filenames=[FIGURE_NAME], force=True)
print(f"   stub → output/TEXFIGU/{FIGURE_NAME}.tex")
print("Done.")
