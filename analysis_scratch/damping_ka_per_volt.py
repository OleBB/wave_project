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
from wavescripts.plot_utils import apply_thesis_style, WIND_COLOR_MAP, amp_to_label, amp_to_tag
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

# V11 magenta palette — chosen in damping_ka_exploration (user-approved).
PER_WIND_COLOR = {
    ("per240", "no"):   WIND_COLOR_MAP.get("no",   "#1f77b4"),  # canonical blue
    ("per240", "full"): WIND_COLOR_MAP.get("full", "#d62728"),  # canonical red
    ("per40",  "no"):   "#00D4BC",                              # turquoise
    ("per40",  "full"): "#D946EF",                              # magenta (fuchsia)
}

WIND_LABEL = {"no": "uten vind", "full": "med vind"}


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


def _make_figure(sub: pd.DataFrame, volt: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for per_tag in ("per240", "per40"):
        for wind in ALL_WINDS:
            sel = sub[(sub["per_tag"] == per_tag) &
                      (sub["WindCondition"] == wind)]
            if sel.empty:
                continue
            ax.scatter(sel[KA_COL], sel[RATIO_COL],
                       marker="o",
                       color=PER_WIND_COLOR[(per_tag, wind)],
                       s=50, alpha=0.85,
                       edgecolors="black", linewidths=0.35,
                       zorder=3)

    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.7, alpha=0.55)
    ax.set_xlim(XLIM); ax.set_ylim(YLIM)
    ax.set_xlabel(r"$ka$  (Inn, målt)", fontsize=10)
    ax.set_ylabel(r"$A_\mathrm{Ut}/A_\mathrm{inn}$", fontsize=10)
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.xaxis.set_minor_locator(MultipleLocator(0.01))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.02))
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(True, which="minor", alpha=0.10)

    handles = [
        mlines.Line2D([], [], color=PER_WIND_COLOR[("per240", "no")],
                      marker="o", ls="None", ms=6, mec="black", mew=0.3,
                      label=f"per240 · {WIND_LABEL['no']}"),
        mlines.Line2D([], [], color=PER_WIND_COLOR[("per240", "full")],
                      marker="o", ls="None", ms=6, mec="black", mew=0.3,
                      label=f"per240 · {WIND_LABEL['full']}"),
        mlines.Line2D([], [], color=PER_WIND_COLOR[("per40",  "no")],
                      marker="o", ls="None", ms=6, mec="black", mew=0.3,
                      label=f"per40 · {WIND_LABEL['no']}"),
        mlines.Line2D([], [], color=PER_WIND_COLOR[("per40",  "full")],
                      marker="o", ls="None", ms=6, mec="black", mew=0.3,
                      label=f"per40 · {WIND_LABEL['full']}"),
    ]
    ax.legend(handles=handles, title="Kjøringstype · vind",
              title_fontsize=8, fontsize=8,
              loc="lower left", framealpha=0.92)
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


# ─── Build three standalone figures ───────────────────────────────────────
for volt in ALL_VOLTS:
    volt_tag = amp_to_tag(volt)
    figure_name = f"ch05_damping_ka_{volt_tag}"
    sub = m[np.isclose(m["WaveAmplitudeInput [Volt]"], volt)]

    print(f"\n{volt_tag}: {len(sub)} runs")
    fig = _make_figure(sub, volt)
    out_pdf = FIGURES_DIR / f"{figure_name}.pdf"
    # out_pgf = FIGURES_DIR / f"{figure_name}.pgf"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pgf, bbox_inches="tight")
    plt.close(fig)
    print(f"   → {out_pdf.relative_to(BASE)}")
    _write_stub(sub, volt, figure_name)

print("\nDone.")
