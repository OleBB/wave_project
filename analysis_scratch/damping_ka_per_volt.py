"""
CH05 §4 — OUT/IN vs ka, per-voltage, fully standalone.

Produces three standalone thesis-grade figures (one per paddle voltage) +
three matching TEXFIGU stubs. Combines per240 and per40 runs on the same
panel, with per240 on the canonical thesis colour pair (blue = nowind,
red = fullwind) and per40 on a magenta/turquoise pair that is visually
separable from the red/blue without breaking the thesis colour
convention (feedback_wind_color_convention.md).

Data scope — same as main_save_figures.py `_pv_damping_ka`:
  - two canon lowrange folders (2026-03-26 / 2026-03-27)
  - PanelCondition = full
  - 1.3–1.6 Hz
  - quality_flag ∈ {ok, NaN}
  - per240 and per40 only (per15 excluded — window too short for H&G)

Outputs (one per voltage):
    output/FIGURES/ch05_damping_ka_10V.pdf
    output/FIGURES/ch05_damping_ka_20V.pdf
    output/FIGURES/ch05_damping_ka_30V.pdf
    output/TEXFIGU/ch05_damping_ka_10V.tex
    output/TEXFIGU/ch05_damping_ka_20V.tex
    output/TEXFIGU/ch05_damping_ka_30V.tex
"""

import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import font_manager as _fm
from matplotlib.ticker import MultipleLocator

from wavescripts.improved_data_loader import load_analysis_data
from wavescripts.plot_utils import (apply_thesis_style, apply_horizontal_ylabel,
                                    WIND_COLOR_MAP, amp_to_label, amp_to_tag)
from wavescripts.plotter import _freq_marker
import wavescripts.plot_utils as pu


BASE        = Path("/Users/ole/Kodevik/wave_project")
FIGURES_DIR = BASE / "output" / "FIGURES"
TEXFIGU_DIR = BASE / "output" / "TEXFIGU"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TEXFIGU_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIRS = [
    BASE / "waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange",
    BASE / "waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange",
]

KA_COL    = "IN ka (FFT)"
RATIO_COL = "OUT/IN (FFT)"

# Captions live centrally in main_save_figures.py (FIGURE_CAPTIONS /
# FIGURE_CAPTIONS_SHORT). pu.write_figure_stub looks them up by figure_name
# via output/.figure_captions.json. Nothing to edit here.

# Wind colour palette: per240 uses the canonical thesis WIND_COLOR_MAP
# (blue / red, identical to ch05_damping_freq); per40 uses lighter tints
# of the same hues so the wind→colour convention reads consistently across
# all CH05 figures and the per-tag distinction is a brightness step rather
# than a hue jump.
import matplotlib.colors as _mcolors


def _lighten(c: str, mix: float = 0.55) -> tuple:
    """Blend colour ``c`` with white. mix=0 → unchanged, mix=1 → white."""
    r, g, b = _mcolors.to_rgb(c)
    return (r + (1.0 - r) * mix,
            g + (1.0 - g) * mix,
            b + (1.0 - b) * mix)


PER_WIND_COLOR = {
    ("per240", "no"):   WIND_COLOR_MAP["no"],            # canonical blue
    ("per240", "full"): WIND_COLOR_MAP["full"],          # canonical red
    ("per40",  "no"):   _lighten(WIND_COLOR_MAP["no"]),   # light blue
    ("per40",  "full"): _lighten(WIND_COLOR_MAP["full"]), # light red / pink
}

WIND_LABEL = {"no": "uten vind", "full": "full vind"}


# ─── NCM font (thesis body) ───────────────────────────────────────────────
apply_thesis_style()
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

pu.ACTIVE_DATASETS = [p.name for p in RESULTS_DIRS]
pu.TEXFIGU_DIR = TEXFIGU_DIR
pu.FIGURES_DIR = FIGURES_DIR


# ─── Load & filter ────────────────────────────────────────────────────────
print("Loading …")
meta, _, _, _ = load_analysis_data(*[str(p) for p in RESULTS_DIRS],
                                   load_processed=False)

m = meta.copy()
m = m[m["PanelCondition"] == "full"]
m = m[m["WaveFrequencyInput [Hz]"].between(1.3, 1.6, inclusive="both")]
if "quality_flag" in m.columns:
    m = m[m["quality_flag"].isna() | (m["quality_flag"] == "ok")]
m = m.dropna(subset=[KA_COL, RATIO_COL, "WindCondition",
                     "WaveAmplitudeInput [Volt]",
                     "WaveFrequencyInput [Hz]"])

# Tag per-run length
_per40  = re.compile(r"per40(?!\d)")
_per240 = re.compile(r"per240")
m["per_tag"] = np.where(m["path"].str.contains(_per40,  na=False), "per40",
                 np.where(m["path"].str.contains(_per240, na=False), "per240",
                          "other"))
m = m[m["per_tag"].isin(("per240", "per40"))].copy()

print(f"  {len(m)} runs (per240={int((m['per_tag']=='per240').sum())}, "
      f"per40={int((m['per_tag']=='per40').sum())})")

ALL_WINDS = sorted(m["WindCondition"].unique())
ALL_FREQS = sorted(m["WaveFrequencyInput [Hz]"].unique())
ALL_VOLTS = sorted(m["WaveAmplitudeInput [Volt]"].unique())


# ─── Axis envelope — shared across the three voltages for direct comparison
_pad_x = 0.05 * (m[KA_COL].max() - m[KA_COL].min())
_pad_y = 0.05 * (m[RATIO_COL].max() - m[RATIO_COL].min())
XLIM = (max(0.0, m[KA_COL].min()    - _pad_x), m[KA_COL].max()    + _pad_x)
YLIM = (max(0.0, m[RATIO_COL].min() - _pad_y), m[RATIO_COL].max() + _pad_y)


# Per-(amplitude, frequency) marker encoding via plotter._freq_marker:
#   A1 (0.10 V): circle family — full / 3/4 / right-half / upper-quarter
#   A2 (0.20 V): tall rectangle rotated 0° / 45° / 90° / 135°
#   A3 (0.30 V): triangle pointing up / left / down / right
# The four orientations within each amp family map to f = 1.3 / 1.4 / 1.5 / 1.6 Hz
# (sorted ascending). This gives 12 visually-distinct markers for the
# (amp × freq) combinations on top of the per-tag × wind colour encoding.
THESIS_FREQS = [1.3, 1.4, 1.5, 1.6]
FREQ_IDX = {f: i for i, f in enumerate(THESIS_FREQS)}


def _make_figure(sub: pd.DataFrame,
                 volt: "float | None" = None) -> plt.Figure:
    """One scatter axis with shared style across per-volt and combined views.

    volt=None        → combined (all 3 amps overlaid)
    volt=<float>     → single-amplitude view (single shape family)

    Marker family encodes amplitude (A1=circle, A2=rectangle, A3=triangle);
    orientation/fill within each family encodes frequency (1.3 → 1.6 Hz).
    Colour always encodes (per_tag × wind) per PER_WIND_COLOR. Ticks, grid,
    xlim/ylim, and legend layout are identical between modes.
    """
    fig, ax = plt.subplots(figsize=(8, 5.5))

    combined = volt is None
    amp_iter = ALL_VOLTS if combined else [volt]

    for amp_val in amp_iter:
        sub_amp = sub[np.isclose(sub["WaveAmplitudeInput [Volt]"], amp_val)]
        for per_tag in ("per240", "per40"):
            for wind in ALL_WINDS:
                for freq, fi in FREQ_IDX.items():
                    sel = sub_amp[(sub_amp["per_tag"] == per_tag) &
                                  (sub_amp["WindCondition"] == wind) &
                                  np.isclose(sub_amp["WaveFrequencyInput [Hz]"], freq)]
                    if sel.empty:
                        continue
                    ax.scatter(sel[KA_COL], sel[RATIO_COL],
                               marker=_freq_marker(amp_val, fi),
                               color=PER_WIND_COLOR[(per_tag, wind)],
                               s=55, alpha=0.85,
                               edgecolors="black", linewidths=0.35,
                               zorder=3)

    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.7, alpha=0.55)
    ax.set_xlim(XLIM); ax.set_ylim(YLIM)
    ax.set_xlabel(r"$ka$  (Inn, målt)", fontsize=10)
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.xaxis.set_minor_locator(MultipleLocator(0.01))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.02))
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(True, which="minor", alpha=0.10)

    # Kjøringstype × vind legend — colour-coded, same in both modes.
    wind_handles = [
        mlines.Line2D([], [], color=PER_WIND_COLOR[("per240", "no")],
                      marker="o", ls="None", ms=6, mec="black", mew=0.3,
                      label=f"Lang tidsserie · {WIND_LABEL['no']}"),
        mlines.Line2D([], [], color=PER_WIND_COLOR[("per240", "full")],
                      marker="o", ls="None", ms=6, mec="black", mew=0.3,
                      label=f"Lang tidsserie · {WIND_LABEL['full']}"),
        mlines.Line2D([], [], color=PER_WIND_COLOR[("per40",  "no")],
                      marker="o", ls="None", ms=6, mec="black", mew=0.3,
                      label=f"Kort tidsserie · {WIND_LABEL['no']}"),
        mlines.Line2D([], [], color=PER_WIND_COLOR[("per40",  "full")],
                      marker="o", ls="None", ms=6, mec="black", mew=0.3,
                      label=f"Kort tidsserie · {WIND_LABEL['full']}"),
    ]
    leg_w = ax.legend(handles=wind_handles, title="Kjøringstype · vind",
                      title_fontsize=8, fontsize=8,
                      loc="lower right", framealpha=0.92)
    ax.add_artist(leg_w)

    if combined:
        # Marker matrix inset — explicit 3×4 grid of every (amp × freq) marker.
        # Replaces the prior "Frekvens (one amp exemplar) + Amplitude (one
        # freq exemplar)" pair so the reader can read the actual marker for
        # any (amp, freq) cell directly. Bottom-centre, where data is sparse.
        AMPS = [0.10, 0.20, 0.30]
        ax_legend = ax.inset_axes([0.30, 0.04, 0.40, 0.22])
        ax_legend.set_facecolor("white")
        for i, amp_v in enumerate(AMPS):
            for j, fh in enumerate(THESIS_FREQS):
                ax_legend.scatter(
                    j, len(AMPS) - 1 - i,
                    marker=_freq_marker(amp_v, FREQ_IDX[fh]),
                    color="gray", s=70,
                    edgecolor="black", linewidth=0.4,
                )
        ax_legend.set_xlim(-0.5, len(THESIS_FREQS) - 0.5)
        ax_legend.set_ylim(-0.7, len(AMPS) - 0.3)
        ax_legend.set_xticks(range(len(THESIS_FREQS)))
        ax_legend.set_xticklabels([f"{f:.1f} Hz" for f in THESIS_FREQS],
                                    fontsize=8)
        ax_legend.set_yticks(range(len(AMPS)))
        ax_legend.set_yticklabels(
            [amp_to_label(v) for v in reversed(AMPS)], fontsize=9
        )
        ax_legend.tick_params(length=0, pad=2)
        for spine in ax_legend.spines.values():
            spine.set_edgecolor("#999")
            spine.set_linewidth(0.6)
        ax_legend.set_title("Amplitude · Frekvens", fontsize=9, pad=4)
    else:
        # Per-volt (single amp) view — Frekvens legend uses the actual amp's
        # marker family, no abstraction needed. Kept as a regular legend.
        freq_handles = [
            mlines.Line2D([], [], color="gray",
                          marker=_freq_marker(volt, fi),
                          ls="None", ms=7, mec="black", mew=0.35,
                          label=f"{f:.1f} Hz")
            for f, fi in FREQ_IDX.items()
        ]
        ax.legend(handles=freq_handles, title="Frekvens",
                   title_fontsize=8, fontsize=8,
                   loc="upper left", framealpha=0.92)

    fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.13)
    apply_horizontal_ylabel(ax, r"$K_t$", fontsize=12)
    return fig


def _f(x, n=4):
    try:
        return f"{float(x):.{n}f}"
    except (TypeError, ValueError):
        return "NA"


def _per_volt_stats(sub: pd.DataFrame) -> dict:
    out = {}
    for per_tag in ("per240", "per40"):
        for wind in ALL_WINDS:
            key_tag = f"{per_tag}_{wind}"
            sel = sub[(sub["per_tag"] == per_tag) &
                      (sub["WindCondition"] == wind)]
            out[f"n_{key_tag}"] = f"{len(sel)}"
            if len(sel):
                out[f"mean_ratio_{key_tag}"] = _f(sel[RATIO_COL].mean(), 4)
                out[f"std_ratio_{key_tag}"]  = _f(sel[RATIO_COL].std(),  4)
                out[f"ka_lo_{key_tag}"]      = _f(sel[KA_COL].min(),     4)
                out[f"ka_hi_{key_tag}"]      = _f(sel[KA_COL].max(),     4)
    # Per-frequency cluster counts
    for freq in ALL_FREQS:
        out[f"n_freq_{freq:.2f}Hz".replace(".", "p")] = f"{int((np.isclose(sub['WaveFrequencyInput [Hz]'], freq)).sum())}"
    # Per-wind means across per-tags (the headline wind effect)
    for wind in ALL_WINDS:
        sel_w = sub[sub["WindCondition"] == wind]
        out[f"mean_ratio_{wind}_all"] = _f(sel_w[RATIO_COL].mean(), 4) if len(sel_w) else "NA"
        out[f"n_{wind}_all"]          = f"{len(sel_w)}"
    return out


def _combined_stats(sub: pd.DataFrame) -> dict:
    """Stats for the all-amplitudes combined figure."""
    out = _per_volt_stats(sub)
    # Per-amp counts so the immutable block records the marker-shape mapping.
    for v in ALL_VOLTS:
        out[f"n_{amp_to_tag(v)}"] = f"{int((np.isclose(sub['WaveAmplitudeInput [Volt]'], v)).sum())}"
    return out


def _write_stub(sub: pd.DataFrame, volt: float, figure_name: str) -> None:
    volt_tag = amp_to_tag(volt)
    stats = _per_volt_stats(sub)

    # Caption text is looked up centrally from FIGURE_CAPTIONS in
    # main_save_figures.py via output/.figure_captions.json — no caption
    # is passed in meta here.
    _meta = pu.build_fig_meta(
        {
            "filters": {
                "PanelCondition":            "full",
                "WaveFrequencyInput [Hz]":   "1.3–1.6",
                "WaveAmplitudeInput [Volt]": volt,
                "WindCondition":             "no + full",
                "quality_flag":              "ok",
            },
            "plotting": {
                "figure_name": figure_name,
            },
        },
        chapter="05",
        extra={"script": "analysis_scratch/damping_ka_per_volt.py"},
        computed_in=("analysis_scratch/damping_ka_per_volt.py "
                     f"(single-voltage scatter, {volt_tag})"),
        data_class="DELEG",
        findings_doc="memory/methodology_wind_enhances_A_in.md",
        fft_window_hz=0.1,
        extra_params=(
            # Data slicing
            f"amplitude tier {volt_tag} (paddle V = {volt:.2f} V → "
            f"nominal measured amplitude at IN ≈ "
            f"{ {0.10: 7.5, 0.20: 15.0, 0.30: 21.5}.get(round(volt, 2), 0.0):.1f} mm). "
            f"frequency range = 1.3–1.6 Hz (thesis scope). "
            f"panel condition = full. quality_flag ∈ {{ok, NaN}}. "
            f"per-tags included: per240 + per40 "
            f"(per15 excluded — window too short for H&G [50T, 60T]). "
            f"datasets: {', '.join(p.name for p in RESULTS_DIRS)}. "
            # Axis + visual
            f"x-axis = IN ka (FFT) computed per run from measured IN-side "
            f"wavenumber and amplitude (not derived from dispersion). "
            f"y-axis = OUT/IN (FFT) from meta.json (canonical IN/OUT mean "
            f"of same-distance probes; see CLAUDE.md §5). "
            f"Dashed reference: ratio = 1 (no damping). "
            # Colour convention
            f"Colour convention: per240 (long runs) uses thesis-wide "
            f"RED = fullwind, BLUE = nowind (feedback_wind_color_convention.md). "
            f"per40 (short runs) uses magenta (#D946EF) for fullwind and "
            f"turquoise (#00D4BC) for nowind — chosen for hue-distance from "
            f"red/blue without crossing into green/orange. "
            # Typeset
            f"Typeset in NewComputerModern10 (OTFs from TeXLive's "
            f"newcomputermodern package, registered via font_manager)."
        ),
        extra_stats=stats,
    )

    # force=True — always rewrite the stub (immutable block + body) so
    # the latest central-dict caption text lands. Hand edits to the .tex
    # body do not survive a re-run.
    pu.write_figure_stub(_meta, plot_type="damping_ka",
                         subfig_filenames=[figure_name],
                         force=True)
    print(f"   stub → output/TEXFIGU/{figure_name}.tex")


def _write_combined_stub(sub: pd.DataFrame, figure_name: str) -> None:
    """Stub for the all-amplitudes combined ka figure."""
    _meta = pu.build_fig_meta(
        {
            "filters": {
                "PanelCondition":            "full",
                "WaveFrequencyInput [Hz]":   "1.3–1.6",
                "WaveAmplitudeInput [Volt]": "0.10, 0.20, 0.30",
                "WindCondition":             "no + full",
                "quality_flag":              "ok",
            },
            "plotting": {
                "figure_name": figure_name,
            },
        },
        chapter="05",
        extra={"script": "analysis_scratch/damping_ka_per_volt.py"},
        computed_in=("analysis_scratch/damping_ka_per_volt.py "
                     "(combined all-amplitudes overview)"),
        data_class="DELEG",
        findings_doc="memory/methodology_wind_enhances_A_in.md",
        fft_window_hz=0.1,
        extra_params=(
            f"all three amplitude tiers (A1/A2/A3) overlaid on one panel. "
            f"frequency range = 1.3–1.6 Hz (thesis scope). "
            f"panel condition = full. quality_flag ∈ {{ok, NaN}}. "
            f"per-tags included: per240 + per40 "
            f"(per15 excluded — window too short for H&G [50T, 60T]). "
            f"datasets: {', '.join(p.name for p in RESULTS_DIRS)}. "
            f"x-axis = IN ka (FFT) computed per run from measured IN-side "
            f"wavenumber and amplitude (not derived from dispersion). "
            f"y-axis = OUT/IN (FFT) from meta.json (canonical IN/OUT mean "
            f"of same-distance probes; see CLAUDE.md §5). "
            f"Dashed reference: ratio = 1 (no damping). "
            f"Encoding: marker shape → amplitude tier "
            f"(○ A1, □ A2, △ A3); colour → per-tag × wind "
            f"(blue per240·nowind, red per240·fullwind, "
            f"turquoise per40·nowind, magenta per40·fullwind). "
            f"Axes / ticks / grid identical to ch05_damping_ka_A1/A2/A3 for "
            f"direct visual comparison. "
            f"Typeset in NewComputerModern10."
        ),
        extra_stats=_combined_stats(sub),
    )
    pu.write_figure_stub(_meta, plot_type="damping_ka",
                         subfig_filenames=[figure_name],
                         force=True)
    print(f"   stub → output/TEXFIGU/{figure_name}.tex")


# ─── Build three standalone figures ───────────────────────────────────────
for volt in ALL_VOLTS:
    volt_tag = amp_to_tag(volt)
    figure_name = f"ch05_damping_ka_{volt_tag}"
    sub = m[np.isclose(m["WaveAmplitudeInput [Volt]"], volt)]

    print(f"\n{volt_tag}: {len(sub)} runs")
    fig = _make_figure(sub, volt=volt)
    out_pdf = FIGURES_DIR / f"{figure_name}.pdf"
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"   → {out_pdf.relative_to(BASE)}")
    _write_stub(sub, volt, figure_name)


# ─── Combined figure (all amplitudes overlaid) ────────────────────────────
print(f"\nCombined (all amps): {len(m)} runs")
fig = _make_figure(m, volt=None)
out_pdf = FIGURES_DIR / "ch05_damping_ka.pdf"
fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print(f"   → {out_pdf.relative_to(BASE)}")
_write_combined_stub(m, "ch05_damping_ka")

print("\nDone.")
