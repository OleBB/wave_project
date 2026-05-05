"""
Inspirational time-series figures for Ch. 4 §5 opening.

Two figures, identical layout — nowind vs fullwind canon run at 1.4 Hz, 0.2 V,
per240, fullpanel. Macro (full recording, ~52 s) + micro (5-period zoom) for
both IN and OUT probes. The zoom window follows each probe's own H&G window,
so the OUT zoom sits later in time than IN (wave arrival is ~5 s later at OUT).

Outputs:
    output/FIGURES/ch04_inspirational_nowind.pdf
    output/FIGURES/ch04_inspirational_fullwind.pdf
    output/TEXFIGU/ch04_inspirational_nowind.tex
    output/TEXFIGU/ch04_inspirational_fullwind.tex

Caption text is sourced from FIGURE_CAPTIONS / FIGURE_CAPTIONS_SHORT in
main_save_figures.py (single source of truth) via the JSON cache. Stub
bodies are rewritten on every run (force=True) so the latest authored
captions land — do not hand-edit the .tex bodies.

See also the exploratory variants in
    analysis_scratch/inspirational_timeseries_C_variants.py
which writes to output/timeseries_exploration/ for side-by-side comparison
of figure height / y-tick density / zoom-ylim choices.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
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

# Shared y-limit for the η(t) panels (2026-05-05). Hardcoded so both
# the nowind and fullwind figures use the same scale and the amplitude
# difference (and the wind-induced ripple on top) is immediately
# obvious side-by-side. Resolves the per-run _sym_ylim mismatch flagged
# in the in-source TODO above the helper.
YLIM_MM = (-24.0, 24.0)

# Shared x-limit for the FULL (macro) η(t) panels only — covers the
# wave train through the longest H&G window end + a 10-s tail. Zoom
# panels stay anchored on each run's own H&G window so the visible
# slice still tracks per-run wave arrival.
MACRO_XLIM_S = (0.0, 45.0)

TARGET_DIR = BASE / "waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"
DATADIR    = BASE / "wavedata/20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"

RUNS = {
    "nowind":   DATADIR / "fullpanel-nowind-amp0200-freq1400-per240-depth580-mstop30-run1.csv",
    "fullwind": DATADIR / "fullpanel-fullwind-amp0200-freq1400-per240-depth580-mstop30-run1.csv",
}

IN_PROBE  = "9373/170"
OUT_PROBE = "12400/250"
FREQ  = 1.4
AMP_V = 0.2

# Title wording per wind condition (Norwegian, thesis-facing).
TITLE_WIND = {
    "nowind":   "Tidsserier, uten vind",
    "fullwind": "Tidsserier, med full vind",
}

# Thesis-wide colour convention: wind condition → colour (CLAUDE.md).
# The badge border + signal colour both pick this up.
WIND_KEY = {"nowind": "no", "fullwind": "full"}
WIN_COLOR = "#F1B24A"  # amber — shaded H&G window + zoom connectors

# Captions live centrally in main_save_figures.py (FIGURE_CAPTIONS /
# FIGURE_CAPTIONS_SHORT). pu.write_figure_stub looks them up by figure_name
# via output/.figure_captions.json. Nothing to edit here.


apply_thesis_style()

# Match thesis body font — NewComputerModern OTFs shipped with TeX Live.
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


def _load_run(wind_tag: str):
    csv = str(RUNS[wind_tag])
    row = meta[meta["path"] == csv].iloc[0]
    df  = proc[csv]
    t   = np.arange(len(df)) / FS

    # ── Mean-level handling — this script does NOT zero the signal here ────
    # `eta_{pos}` and `eta_{pos}_interp` arrive from the pipeline already
    # mean-zeroed against a per-run stillwater anchor. The chain is:
    #
    #   1. processor.ensure_stillwater_columns (processor.py:72) computes a
    #      Stillwater value PER RUN, PER PROBE, and writes it into meta as
    #      "Stillwater Probe {pos}":
    #        - nowave runs (no paddle) → median of the FULL run
    #        - wave runs (with paddle) → mean of the first
    #          STILLWATER.PRE_WAVE_S = 2.0 s, before the wave front arrives
    #      Self-calibrating: each run uses its own water level, so wind
    #      setup (which can shift the level by mm and takes ~10 min to
    #      decay after wind off) is automatically tracked.
    #
    #   2. processor._zero_and_smooth_signals (processor.py:759) applies it:
    #          eta_{pos} = -(raw_ULS - stillwater)
    #      The minus sign converts raw probe distance (sensor-to-water,
    #      decreasing as water rises) into surface elevation (positive up).
    #
    # By construction, eta_{pos} has ≈0 mean across the pre-wave window for
    # every run. Verified for the canon nowind run used here:
    #   IN  9373/170 — pre-wave (0–2 s) eta mean = −0.011 mm
    #   OUT 12400/250 — pre-wave (0–2 s) eta mean = −0.009 mm
    #
    # Crest-vs-trough asymmetry (max+min ≈ +1.6 mm at IN, +1.6 mm at OUT)
    # is therefore NOT a baseline error — it's Stokes-2 nonlinearity:
    # nonlinear waves have taller, steeper crests and shallower, broader
    # troughs. For ka ≈ 0.13 at IN, second-order theory predicts ≈ +1 mm
    # of (max+min) asymmetry, which matches what's plotted.
    def get_eta(probe):
        col = f"eta_{probe}_interp"
        if col not in df.columns:
            col = f"eta_{probe}"
        return df[col].to_numpy(dtype=float)

    eta_in  = get_eta(IN_PROBE)
    eta_out = get_eta(OUT_PROBE)

    def window_s(probe):
        s = int(row[f"Computed Probe {probe} start"])
        e = int(row[f"Computed Probe {probe} end"])
        return s / FS, e / FS

    in_ws,  in_we  = window_s(IN_PROBE)
    out_ws, out_we = window_s(OUT_PROBE)

    return {
        "row":     row,
        "t":       t,
        "eta_in":  eta_in,
        "eta_out": eta_out,
        "in_ws":   in_ws,  "in_we":  in_we,
        "out_ws":  out_ws, "out_we": out_we,
        "ka_in":   float(row["IN ka (FFT)"]),
        "ka_out":  float(row["OUT ka (FFT)"]),
    }

# Per-run symmetric y-limit helper. Retained for variants that want a
# data-driven scale; the canonical thesis figures use the hardcoded
# YLIM_MM module constant instead so nowind/fullwind are directly
# comparable side-by-side.
def _sym_ylim(*sigs, pad=0.15):
    y = np.concatenate(sigs)
    lo, hi = np.nanpercentile(y, [0.5, 99.5])
    m = max(abs(lo), abs(hi))
    return -m * (1 + pad), m * (1 + pad)


def _apply_ticks(ax, *, x_major, x_minor, y_major, y_minor):
    ax.xaxis.set_major_locator(MultipleLocator(x_major))
    ax.xaxis.set_minor_locator(MultipleLocator(x_minor))
    ax.yaxis.set_major_locator(MultipleLocator(y_major))
    ax.yaxis.set_minor_locator(MultipleLocator(y_minor))
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(True, which="minor", alpha=0.12)


def make_figure(wind_tag: str, data: dict, out_pdf: Path) -> None:
    """Geometry: zoom-tall variant — (10×13) figsize, height_ratios=[1,2,1,2]."""
    color = WIND_COLOR_MAP[WIND_KEY[wind_tag]]

    eta_in, eta_out = data["eta_in"], data["eta_out"]
    t = data["t"]
    in_ws, in_we   = data["in_ws"],  data["in_we"]
    out_ws, out_we = data["out_ws"], data["out_we"]
    ylim = YLIM_MM           # shared across nowind / fullwind for direct comparison
    macro_xlim = MACRO_XLIM_S  # shared across nowind / fullwind for direct comparison
    x_cutoff = macro_xlim[1]   # legacy name; preserved for any later inline use

    # Zoom on the BEGINNING of each probe's H&G window — shows the moment
    # the measurement window opens. The zoom x-range is the TRUE computed
    # window start through start + 5 periods (no lead-in), so the visible
    # slice is the first 5 cycles inside the analysis window.
    zoom_width = 5.0 / FREQ
    in_zx0,  in_zx1  = in_ws,  in_ws  + zoom_width
    out_zx0, out_zx1 = out_ws, out_ws + zoom_width

    fig = plt.figure(figsize=(6.27, 8.15))
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 2, 1, 2], hspace=0.42)
    ax_in_full  = fig.add_subplot(gs[0, 0])
    ax_in_zoom  = fig.add_subplot(gs[1, 0])
    # Macros share x (0 … x_cutoff). Zooms do NOT share x — each follows its
    # own probe's H&G window, so OUT zoom shifts right (later wave arrival).
    ax_out_full = fig.add_subplot(gs[2, 0], sharex=ax_in_full)
    ax_out_zoom = fig.add_subplot(gs[3, 0])

    for ax_full, ax_zoom, sig, probe, label, zx0, zx1 in [
        (ax_in_full,  ax_in_zoom,  eta_in,  IN_PROBE,  "Innkommende bølge",
         in_zx0,  in_zx1),
        (ax_out_full, ax_out_zoom, eta_out, OUT_PROBE, "Utgående bølge",
         out_zx0, out_zx1),
    ]:
        ax_full.plot(t, sig, color=color, lw=0.5)
        ax_full.axvspan(zx0, zx1, color=WIN_COLOR, alpha=0.55, zorder=0)
        ax_full.axhline(0, color="#888", lw=0.5, alpha=0.6)
        ax_full.set_xlim(*macro_xlim)
        ax_full.set_ylim(ylim)
        ax_full.text(
            0.005, 0.92, label,
            transform=ax_full.transAxes, va="top", ha="left",
            fontsize=10, color="black",
            bbox=dict(boxstyle="square,pad=0.25", fc="white",
                      ec="#888", lw=0.5, alpha=0.97),
        )

        m = (t >= zx0) & (t <= zx1)
        ax_zoom.plot(t[m], sig[m], color=color, lw=1.3)
        ax_zoom.scatter(t[m][::6], sig[m][::6], s=6, color=color, alpha=0.55,
                        edgecolor="none")
        ax_zoom.axhline(0, color="#888", lw=0.5, alpha=0.6)
        ax_zoom.set_xlim(zx0, zx1)
        ax_zoom.set_ylim(ylim)

        for xx in (zx0, zx1):
            con = ConnectionPatch(
                xyA=(xx, ylim[0]), coordsA=ax_full.transData,
                xyB=(xx, ylim[1]), coordsB=ax_zoom.transData,
                color=WIN_COLOR, lw=0.7, alpha=0.9, zorder=0,
            )
            fig.add_artist(con)

    ax_in_zoom.set_xlabel("Tid [s]")
    ax_out_zoom.set_xlabel("Tid [s]")
    for ax in (ax_in_full, ax_out_full):
        _apply_ticks(ax, x_major=5.0, x_minor=1.0, y_major=5.0, y_minor=1.0)
    for ax in (ax_in_zoom, ax_out_zoom):
        _apply_ticks(ax, x_major=0.5, x_minor=0.1, y_major=5.0, y_minor=1.0)

    # Horizontal y-axis label above the top pane's leftmost tick label —
    # mirrors the ch04_plateau_overview convention. All four panes share the
    # same y-range, so a single label on ax_in_full identifies them all and
    # the rotated side labels are removed for maximum horizontal space.
    ax_in_full.set_ylabel(r"$\eta$ [mm]",
                          rotation=0, ha="left", va="bottom", fontsize=10)
    fig.canvas.draw()
    _renderer = fig.canvas.get_renderer()
    _ticks = [t for t in ax_in_full.yaxis.get_ticklabels()
              if t.get_visible() and t.get_text().strip()]
    if _ticks:
        _left_disp = min(t.get_window_extent(renderer=_renderer).x0
                         for t in _ticks)
        _x_axes = ax_in_full.transAxes.inverted().transform((_left_disp, 0))[0]
        ax_in_full.yaxis.set_label_coords(_x_axes, 1.02)

    # No in-figure title: LaTeX \caption{} handles identification in the
    # thesis. ka_IN and ka_OUT are already in the immutable stub block
    # (extra_stats → stat:ka_in / stat:ka_out) and in the caption text.
    fig.savefig(out_pdf)
    # fig.savefig(out_pdf.with_suffix(".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"   → {out_pdf.relative_to(BASE)}  (+ .png)")


def _f(x, n=4):
    try:
        return f"{float(x):.{n}f}"
    except (TypeError, ValueError):
        return "NA"


def _inspirational_extra_stats(row, data, wind_tag):
    """Pack everything useful from meta + derived quantities for one run."""
    out = {}
    # ka is the primary wave descriptor — always surfaced
    out["ka_inn"] = _f(data["ka_in"],  4)
    out["ka_Ut"]  = _f(data["ka_out"], 4)
    # Per-side dispersion / physics from meta
    for side_short, side_long in (("inn", "IN"), ("Ut", "OUT")):
        out[f"A_{side_short}_FFT_mm"]         = _f(row[f"{side_long} Amplitude (FFT)"],            3)
        out[f"wavelength_{side_short}_m"]     = _f(row[f"{side_long} Wavelength (FFT)"],           4)
        out[f"wavenumber_{side_short}_per_m"] = _f(row[f"{side_long} Wavenumber (FFT)"],           4)
        out[f"period_{side_short}_s"]         = _f(row[f"{side_long} WavePeriod (FFT)"],           5)
        out[f"celerity_{side_short}_m_s"]     = _f(row[f"{side_long} Celerity (FFT)"],             4)
        out[f"Hm0_{side_short}_mm"]           = _f(row[f"{side_long} Significant Wave Height Hm0"], 3)
        out[f"Hs_{side_short}_mm"]            = _f(row[f"{side_long} Significant Wave Height Hs"],  3)
        out[f"Froude_{side_short}"]           = _f(row[f"{side_long} Froude (FFT)"],               5)
        out[f"Ursell_{side_short}"]           = _f(row[f"{side_long} Ursell (FFT)"],               5)
        out[f"wind_over_c_{side_short}"]      = _f(row[f"{side_long} Wind/Celerity (FFT)"],        4)
        out[f"f_over_fPM_{side_short}"]       = _f(row[f"{side_long} f/f_PM (FFT)"],               4)
        out[f"wave_stability_{side_short}"]   = _f(row[f"{side_long} wave_stability"],             4)
        out[f"period_cv_{side_short}"]        = _f(row[f"{side_long} period_amplitude_cv"],        4)
    # Per-probe: alternate amplitudes (method cross-check)
    for side_short, probe in (("inn", IN_PROBE), ("Ut", OUT_PROBE)):
        out[f"A_{side_short}_LS_mm"]          = _f(row[f"Probe {probe} Amplitude (LS)"],           3)
        out[f"A_{side_short}_Stk2_mm"]        = _f(row[f"Probe {probe} Amplitude Stokes2 (LS)"],   3)
        out[f"A_{side_short}_percentile_mm"]  = _f(row[f"Probe {probe} Amplitude"],                3)
        out[f"A_{side_short}_PSD_mm"]         = _f(row[f"Probe {probe} Amplitude (PSD)"],          3)
        out[f"A_{side_short}_cycles_mean"]    = _f(row[f"Probe {probe} Amplitude (cycles) mean"],  3)
        out[f"A_{side_short}_phase_mean"]     = _f(row[f"Probe {probe} Amplitude (phase) mean"],   3)
        out[f"DC_{side_short}_mm"]            = _f(row[f"Probe {probe} DC (LS)"],                  4)
        out[f"residualRMS_{side_short}_mm"]   = _f(row[f"Probe {probe} Residual RMS (LS)"],        4)
        # H&G window diagnostics
        out[f"window_{side_short}_s"]         = f"[{int(row[f'Computed Probe {probe} start'])/FS:.3f}, {int(row[f'Computed Probe {probe} end'])/FS:.3f}]"
        hg_shift = row.get(f"Probe {probe} hg_snap_shift")
        if hg_shift is not None:
            out[f"hg_snap_shift_{side_short}_samples"] = _f(hg_shift, 1)
    # Per-run OUT/IN — the thesis central metric, cross-check
    out["meta_OUT_over_IN"] = _f(row["OUT/IN (FFT)"], 4)
    # Dataset / setup metadata
    out["file_date"]          = str(row.get("file_date", "NA"))[:10]
    out["mooring"]            = str(row.get("Mooring", "NA"))
    out["probe_height_mm"]    = _f(row.get("probe_height_mm"), 1)
    out["probe_range_mode"]   = str(row.get("probe_range_mode", "NA"))
    out["water_depth_mm"]     = _f(row.get("water_depth_mm", 580), 1)
    # kd depth regime
    try:
        k_m = float(row["OUT Wavenumber (FFT)"])
        h_m = float(row.get("water_depth_mm", 580)) / 1000.0
        kd  = k_m * h_m
        regime = ("shallow"      if kd < np.pi/10 else
                  "intermediate" if kd < np.pi     else
                  "deep")
        out["kd_Ut"]         = _f(kd, 3)
        out["depth_regime"]  = regime
    except Exception:
        pass
    # Input / pipeline sanity values
    out["input_freq_Hz"]     = _f(FREQ, 2)
    out["input_amp_V"]       = _f(AMP_V, 2)
    out["sampling_rate_Hz"]  = _f(FS, 1)
    out["zoom_periods"]      = "5"
    out["wind_condition"]    = wind_tag
    return out


def _write_stub(wind_tag: str, data: dict, figure_name: str) -> None:
    """TEXFIGU stub via pu.write_figure_stub (force=True).

    Caption text is looked up centrally from FIGURE_CAPTIONS in
    main_save_figures.py via output/.figure_captions.json — no caption
    is passed in meta here.
    """
    stub_path = TEXFIGU_DIR / f"{figure_name}.tex"

    _meta = pu.build_fig_meta(
        {
            "filters": {
                "PanelCondition":            "full",
                "WaveFrequencyInput [Hz]":   FREQ,
                "WaveAmplitudeInput [Volt]": AMP_V,
                "WindCondition":             WIND_KEY[wind_tag],
                "quality_flag":              "ok",
            },
            "plotting": {
                "figure_name": figure_name,
            },
        },
        chapter="04",
        extra={"script": "analysis_scratch/inspirational_timeseries.py"},
        computed_in="analysis_scratch/inspirational_timeseries.py (single-run time-series render)",
        data_class="DELEG",
        findings_doc="memory/methodology_hg_window_kills_peak_bias.md",
        fft_window_hz=0.1,
        extra_params=(
            f"run_path={RUNS[wind_tag].relative_to(BASE)}. "
            f"probes=IN:{IN_PROBE}+OUT:{OUT_PROBE}. "
            f"zoom=5 periods at the start of each probe's H&G window "
            f"(0.5-period lead-in + 4.5 periods inside the window; "
            f"macro = full recording, ~{(data['out_we']+10):.1f} s cutoff). "
            f"H&G window: arrival-anchored [t_arr + 7T, t_arr + 17T] with "
            f"t_arr = r/c_g(f, h); start UC-snapped within ±T, end UC-snapped "
            f"to the 10th upcrossing with ±0.5 T guard. "
            f"Amber vspan on macro = zoom extent; zoom extent sits at each "
            f"probe's own H&G window start so OUT zoom sits later in time "
            f"(wave-group lag). "
            f"Colour convention: blue = nowind, red = fullwind — CLAUDE.md "
            f"thesis-wide WIND_COLOR_MAP. "
            f"Typeset in NewComputerModern10 (OTFs from TeXLive's "
            f"newcomputermodern package, registered via font_manager)."
        ),
        extra_stats=_inspirational_extra_stats(data["row"], data, wind_tag),
    )

    # force=True — body is always rewritten so the latest central-dict
    # caption text lands. Hand edits to the .tex body don't survive a re-run.
    pu.write_figure_stub(_meta, plot_type="inspirational_timeseries",
                         subfig_filenames=[figure_name], force=True)
    print(f"   stub → {stub_path.relative_to(BASE)}")


# ─── Build both figures ──────────────────────────────────────────────────
for wind_tag in ("nowind", "fullwind"):
    print(f"\n{wind_tag} …")
    data = _load_run(wind_tag)
    figure_name = f"ch04_inspirational_{wind_tag}"
    make_figure(wind_tag, data, FIGURES_DIR / f"{figure_name}.pdf")
    _write_stub(wind_tag, data, figure_name)

print("\nDone.")
