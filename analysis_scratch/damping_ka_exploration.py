"""
CH05 §4 — OUT/IN vs ka exploration.

Produces several layout candidates for polishing ch05_damping_ka. All read
meta_results (combined_meta filtered to the two 2026-03-26/27 canon folders
at 1.3–1.6 Hz, full panel, quality_flag ok).

Variants:
    V1_single_clean.png          — one panel, marker = amp, colour = wind, NCM font
    V2_by_amp_3panel.png         — 3 panels (0.10 / 0.20 / 0.30 V), shared axes
    V3_single_per_freq.png       — one panel, marker shape per frequency (reveals
                                    the freq-clustering along the ka axis)
    V4_by_amp_per_freq.png       — 3 panels + per-freq markers (most legend-heavy)

Output:
    output/damping_ka_exploration/*.png

No PDFs or TEXFIGU stubs written — promotion happens after you pick.
"""

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
from wavescripts.plot_utils import apply_thesis_style, WIND_COLOR_MAP


BASE = Path("/Users/ole/Kodevik/wave_project")
OUT_DIR = BASE / "output" / "damping_ka_exploration"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIRS = [
    BASE / "waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange",
    BASE / "waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange",
]


# ─── NCM font ─────────────────────────────────────────────────────────────
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


# ─── Load & filter ────────────────────────────────────────────────────────
print("Loading meta_results (canon lowrange folders only) …")
meta, _, _, _ = load_analysis_data(*[str(p) for p in RESULTS_DIRS], load_processed=False)

# Replicate meta_results filter from main_save_figures.py §~335.
# Thesis scope: 1.3–1.6 Hz, fullpanel, quality_flag ok.
m = meta.copy()
m = m[m["PanelCondition"] == "full"]
m = m[m["WaveFrequencyInput [Hz]"].between(1.3, 1.6, inclusive="both")]
if "quality_flag" in m.columns:
    m = m[m["quality_flag"].isna() | (m["quality_flag"] == "ok")]
m = m.dropna(subset=["IN ka (FFT)", "OUT/IN (FFT)",
                     "WindCondition", "WaveAmplitudeInput [Volt]",
                     "WaveFrequencyInput [Hz]"])
print(f"  filtered to {len(m)} runs")
if m.empty:
    print("ERROR: no runs survived filter — aborting.")
    sys.exit(1)

KA = "IN ka (FFT)"
RATIO = "OUT/IN (FFT)"
ALL_AMPS  = sorted(m["WaveAmplitudeInput [Volt]"].unique())
ALL_WINDS = sorted(m["WindCondition"].unique())
ALL_FREQS = sorted(m["WaveFrequencyInput [Hz]"].unique())

AMP_MARKERS  = {ALL_AMPS[i]: mk for i, mk in zip(range(len(ALL_AMPS)),
                                                  ("o", "s", "^", "D"))}
FREQ_MARKERS = {ALL_FREQS[i]: mk for i, mk in zip(range(len(ALL_FREQS)),
                                                   ("o", "s", "^", "D", "v", "P"))}

# Shared axis limits (common across variants for fair comparison)
_pad_x = 0.05 * (m[KA].max() - m[KA].min())
_pad_y = 0.05 * (m[RATIO].max() - m[RATIO].min())
XLIM = (max(0.0, m[KA].min() - _pad_x),      m[KA].max()  + _pad_x)
YLIM = (max(0.0, m[RATIO].min() - _pad_y),   m[RATIO].max() + _pad_y)


def _decorate(ax, *, xlabel=True, ylabel=True, title=None):
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.7, alpha=0.55)
    ax.set_xlim(XLIM); ax.set_ylim(YLIM)
    if xlabel: ax.set_xlabel(r"$ka$  (Inn, målt)", fontsize=10)
    if ylabel: ax.set_ylabel(r"$K_t$", fontsize=10)
    if title:  ax.set_title(title, fontsize=10)
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.xaxis.set_minor_locator(MultipleLocator(0.01))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.02))
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(True, which="minor", alpha=0.10)


def _scatter_amp_markers(ax, df):
    """Colour = wind, marker = amplitude."""
    for amp in ALL_AMPS:
        mk = AMP_MARKERS[amp]
        sub_amp = df[np.isclose(df["WaveAmplitudeInput [Volt]"], amp)]
        for wind in ALL_WINDS:
            sub = sub_amp[sub_amp["WindCondition"] == wind]
            if sub.empty:
                continue
            ax.scatter(sub[KA], sub[RATIO],
                       marker=mk, color=WIND_COLOR_MAP.get(wind, "gray"),
                       s=42, alpha=0.80, edgecolors="black", linewidths=0.3,
                       zorder=3)


def _scatter_freq_markers(ax, df):
    """Colour = wind, marker = frequency."""
    for freq in ALL_FREQS:
        mk = FREQ_MARKERS[freq]
        sub_f = df[np.isclose(df["WaveFrequencyInput [Hz]"], freq)]
        for wind in ALL_WINDS:
            sub = sub_f[sub_f["WindCondition"] == wind]
            if sub.empty:
                continue
            ax.scatter(sub[KA], sub[RATIO],
                       marker=mk, color=WIND_COLOR_MAP.get(wind, "gray"),
                       s=48, alpha=0.82, edgecolors="black", linewidths=0.3,
                       zorder=3)


def _legend_wind(ax, *, loc="upper right"):
    h = [mlines.Line2D([], [], color=WIND_COLOR_MAP.get(w, "gray"),
                       marker="o", linestyle="None", markersize=6,
                       label={"no": "uten vind",
                              "full": "med vind",
                              "lowest": "liten vind"}.get(w, w))
         for w in ALL_WINDS]
    return ax.legend(handles=h, title="Vind", title_fontsize=8,
                     fontsize=8, loc=loc, framealpha=0.92)


def _legend_amp(ax, *, loc="lower left"):
    h = [mlines.Line2D([], [], color="#444",
                       marker=AMP_MARKERS[a], linestyle="None", markersize=6,
                       markeredgecolor="black", markeredgewidth=0.3,
                       label=f"{a:.2f}\u00a0V")
         for a in ALL_AMPS]
    return ax.legend(handles=h, title="Amplitude", title_fontsize=8,
                     fontsize=8, loc=loc, framealpha=0.92)


def _legend_freq(ax, *, loc="lower left"):
    h = [mlines.Line2D([], [], color="#444",
                       marker=FREQ_MARKERS[f], linestyle="None", markersize=6,
                       markeredgecolor="black", markeredgewidth=0.3,
                       label=f"{f:.2f}\u00a0Hz")
         for f in ALL_FREQS]
    return ax.legend(handles=h, title="Frekvens", title_fontsize=8,
                     fontsize=8, loc=loc, framealpha=0.92)


# ─── V1: single clean panel, markers by amplitude ────────────────────────
fig, ax = plt.subplots(figsize=(8, 5.5))
_scatter_amp_markers(ax, m)
_decorate(ax)
leg1 = _legend_wind(ax, loc="upper right"); ax.add_artist(leg1)
_legend_amp(ax, loc="lower left")
fig.tight_layout()
fig.savefig(OUT_DIR / "V1_single_clean.png", dpi=150, bbox_inches="tight")
plt.close(fig)


# ─── V2: 3-panel by amplitude, shared axes ───────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5.0), sharex=True, sharey=True)
for ax, amp in zip(axes, ALL_AMPS):
    sub = m[np.isclose(m["WaveAmplitudeInput [Volt]"], amp)]
    _scatter_amp_markers(ax, sub)
    _decorate(ax, title=f"{amp:.2f}\u00a0V",
              ylabel=(ax is axes[0]))
    if ax is axes[0]:
        _legend_wind(ax, loc="upper right")
fig.tight_layout()
fig.savefig(OUT_DIR / "V2_by_amp_3panel.png", dpi=150, bbox_inches="tight")
plt.close(fig)


# ─── V3: single panel, markers by frequency ──────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5.5))
_scatter_freq_markers(ax, m)
_decorate(ax)
leg1 = _legend_wind(ax, loc="upper right"); ax.add_artist(leg1)
_legend_freq(ax, loc="lower left")
fig.tight_layout()
fig.savefig(OUT_DIR / "V3_single_per_freq.png", dpi=150, bbox_inches="tight")
plt.close(fig)


# ─── V4: 3-panel by amplitude + per-freq markers ─────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5.0), sharex=True, sharey=True)
for ax, amp in zip(axes, ALL_AMPS):
    sub = m[np.isclose(m["WaveAmplitudeInput [Volt]"], amp)]
    _scatter_freq_markers(ax, sub)
    _decorate(ax, title=f"{amp:.2f}\u00a0V",
              ylabel=(ax is axes[0]))
    if ax is axes[0]:
        _legend_wind(ax, loc="upper right")
    if ax is axes[-1]:
        _legend_freq(ax, loc="lower left")
fig.tight_layout()
fig.savefig(OUT_DIR / "V4_by_amp_per_freq.png", dpi=150, bbox_inches="tight")
plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# Per-240 only — V5 … V8 are V1 … V4 but filtered to per240 runs
# ══════════════════════════════════════════════════════════════════════════
m240 = m[m["path"].str.contains("per240", na=False)].copy()
print(f"\nper240 subset: {len(m240)} runs "
      f"(from {len(m)} total thesis-scope runs)")
if m240.empty:
    print("no per240 runs in filtered set — skipping V5–V8")
else:
    # Tighten axis limits to the per240 subset alone (otherwise they re-use
    # the combined-data envelope and the plots look empty on the edges).
    _pad_x240 = 0.05 * (m240[KA].max() - m240[KA].min())
    _pad_y240 = 0.05 * (m240[RATIO].max() - m240[RATIO].min())
    XLIM_240 = (max(0.0, m240[KA].min() - _pad_x240),      m240[KA].max()  + _pad_x240)
    YLIM_240 = (max(0.0, m240[RATIO].min() - _pad_y240),   m240[RATIO].max() + _pad_y240)

    # Which amps / freqs actually survive in per240?
    AMPS_240  = sorted(m240["WaveAmplitudeInput [Volt]"].unique())
    WINDS_240 = sorted(m240["WindCondition"].unique())
    FREQS_240 = sorted(m240["WaveFrequencyInput [Hz]"].unique())

    def _decorate_240(ax, *, xlabel=True, ylabel=True, title=None):
        ax.axhline(1.0, color="black", linestyle="--", linewidth=0.7, alpha=0.55)
        ax.set_xlim(XLIM_240); ax.set_ylim(YLIM_240)
        if xlabel: ax.set_xlabel(r"$ka$  (Inn, målt)", fontsize=10)
        if ylabel: ax.set_ylabel(r"$K_t$", fontsize=10)
        if title:  ax.set_title(title, fontsize=10)
        ax.xaxis.set_major_locator(MultipleLocator(0.05))
        ax.xaxis.set_minor_locator(MultipleLocator(0.01))
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.02))
        ax.grid(True, which="major", alpha=0.30)
        ax.grid(True, which="minor", alpha=0.10)

    # ─── V5: single clean, amp markers ──
    fig, ax = plt.subplots(figsize=(8, 5.5))
    _scatter_amp_markers(ax, m240)
    _decorate_240(ax)
    leg1 = _legend_wind(ax, loc="upper right"); ax.add_artist(leg1)
    _legend_amp(ax, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "V5_per240_single_clean.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ─── V6: 3-panel by amplitude ──
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.0), sharex=True, sharey=True)
    for ax, amp in zip(axes, ALL_AMPS):
        sub = m240[np.isclose(m240["WaveAmplitudeInput [Volt]"], amp)]
        _scatter_amp_markers(ax, sub)
        _decorate_240(ax, title=f"{amp:.2f}\u00a0V",
                      ylabel=(ax is axes[0]))
        if ax is axes[0]:
            _legend_wind(ax, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "V6_per240_by_amp_3panel.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ─── V7: single, freq markers ──
    fig, ax = plt.subplots(figsize=(8, 5.5))
    _scatter_freq_markers(ax, m240)
    _decorate_240(ax)
    leg1 = _legend_wind(ax, loc="upper right"); ax.add_artist(leg1)
    _legend_freq(ax, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "V7_per240_single_per_freq.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ─── V8: 3-panel by amplitude + per-freq markers ──
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.0), sharex=True, sharey=True)
    for ax, amp in zip(axes, ALL_AMPS):
        sub = m240[np.isclose(m240["WaveAmplitudeInput [Volt]"], amp)]
        _scatter_freq_markers(ax, sub)
        _decorate_240(ax, title=f"{amp:.2f}\u00a0V",
                      ylabel=(ax is axes[0]))
        if ax is axes[0]:
            _legend_wind(ax, loc="upper right")
        if ax is axes[-1]:
            _legend_freq(ax, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "V8_per240_by_amp_per_freq.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nper240 summary:")
    print(f"  winds: {WINDS_240}")
    print(f"  amps:  {[f'{a:.2f}' for a in AMPS_240]}")
    print(f"  freqs: {[f'{f:.2f}' for f in FREQS_240]}")
    print(f"  ka range:    [{m240[KA].min():.4f}, {m240[KA].max():.4f}]")
    print(f"  ratio range: [{m240[RATIO].min():.4f}, {m240[RATIO].max():.4f}]")


# ══════════════════════════════════════════════════════════════════════════
# V9 — per240 + per40 combined, marked by run-length
#      per240 keeps the canonical thesis colours (red / blue)
#      per40  uses turquoise (nowind) / pink (fullwind) for visual separation
# ══════════════════════════════════════════════════════════════════════════
import re
_per40_re  = re.compile(r"per40(?!\d)")
_per240_re = re.compile(r"per240")

m_per40  = m[m["path"].str.contains(_per40_re,  na=False)].copy()
m_per240 = m[m["path"].str.contains(_per240_re, na=False)].copy()

# Colour map: (per-tag, wind) → colour
PER_WIND_COLOR = {
    ("per240", "no"):   WIND_COLOR_MAP.get("no",   "#1f77b4"),   # canonical blue
    ("per240", "full"): WIND_COLOR_MAP.get("full", "#d62728"),   # canonical red
    ("per40",  "no"):   "#00C7B1",  # turquoise
    ("per40",  "full"): "#E07BB6",  # pink
}

# Axis envelope: union of both subsets (so the plot doesn't crop)
_combo = pd.concat([m_per40, m_per240])
_pad_x_c = 0.05 * (_combo[KA].max()    - _combo[KA].min())
_pad_y_c = 0.05 * (_combo[RATIO].max() - _combo[RATIO].min())
XLIM_C = (max(0.0, _combo[KA].min()    - _pad_x_c),  _combo[KA].max()    + _pad_x_c)
YLIM_C = (max(0.0, _combo[RATIO].min() - _pad_y_c),  _combo[RATIO].max() + _pad_y_c)


def _scatter_per_wind(ax, df, per_tag):
    """Colour keyed by (per_tag, wind); marker keyed by amplitude."""
    for amp in ALL_AMPS:
        mk = AMP_MARKERS[amp]
        sub_amp = df[np.isclose(df["WaveAmplitudeInput [Volt]"], amp)]
        for wind in ALL_WINDS:
            sub = sub_amp[sub_amp["WindCondition"] == wind]
            if sub.empty:
                continue
            ax.scatter(sub[KA], sub[RATIO],
                       marker=mk,
                       color=PER_WIND_COLOR.get((per_tag, wind), "gray"),
                       s=42, alpha=0.82, edgecolors="black", linewidths=0.3,
                       zorder=3)


fig, ax = plt.subplots(figsize=(8, 5.5))
# per240 first (lower zorder so per40 overlays — both subsets are drawn).
_scatter_per_wind(ax, m_per240, "per240")
_scatter_per_wind(ax, m_per40,  "per40")

ax.axhline(1.0, color="black", linestyle="--", linewidth=0.7, alpha=0.55)
ax.set_xlim(XLIM_C); ax.set_ylim(YLIM_C)
ax.set_xlabel(r"$ka$  (Inn, målt)", fontsize=10)
ax.set_ylabel(r"$K_t$", fontsize=10)
ax.xaxis.set_major_locator(MultipleLocator(0.05))
ax.xaxis.set_minor_locator(MultipleLocator(0.01))
ax.yaxis.set_major_locator(MultipleLocator(0.1))
ax.yaxis.set_minor_locator(MultipleLocator(0.02))
ax.grid(True, which="major", alpha=0.30)
ax.grid(True, which="minor", alpha=0.10)

# Legend 1: 4-row colour legend keyed by (per, wind)
wind_label = {"no": "uten vind", "full": "med vind"}
handles_colour = [
    mlines.Line2D([], [],
                  color=PER_WIND_COLOR[("per240", "no")],
                  marker="o", linestyle="None", markersize=6,
                  markeredgecolor="black", markeredgewidth=0.3,
                  label=f"per240 · {wind_label['no']}"),
    mlines.Line2D([], [],
                  color=PER_WIND_COLOR[("per240", "full")],
                  marker="o", linestyle="None", markersize=6,
                  markeredgecolor="black", markeredgewidth=0.3,
                  label=f"per240 · {wind_label['full']}"),
    mlines.Line2D([], [],
                  color=PER_WIND_COLOR[("per40", "no")],
                  marker="o", linestyle="None", markersize=6,
                  markeredgecolor="black", markeredgewidth=0.3,
                  label=f"per40 · {wind_label['no']}"),
    mlines.Line2D([], [],
                  color=PER_WIND_COLOR[("per40", "full")],
                  marker="o", linestyle="None", markersize=6,
                  markeredgecolor="black", markeredgewidth=0.3,
                  label=f"per40 · {wind_label['full']}"),
]
leg_colour = ax.legend(handles=handles_colour,
                       title="Kjøringstype · vind",
                       title_fontsize=8, fontsize=8,
                       loc="upper right", framealpha=0.92)
ax.add_artist(leg_colour)

# Legend 2: amplitude marker shapes (colour-neutral)
_legend_amp(ax, loc="lower left")

fig.tight_layout()
fig.savefig(OUT_DIR / "V9_combined_per240_per40_single_clean.png",
            dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"\nCombined per240+per40: per240={len(m_per240)} · per40={len(m_per40)} "
      f"runs (of {len(m)} thesis-scope total)")


# ══════════════════════════════════════════════════════════════════════════
# V10 / V11 / V12 — distinguishing per40 fullwind from per240 red
#   Keep per240 = canonical thesis RED (CLAUDE.md feedback_wind_color_convention).
#   Vary per40-fullwind along the red→pink→magenta→violet axis.
#   Also brighten per40-nowind turquoise slightly so the pair scales together.
# ══════════════════════════════════════════════════════════════════════════

PER40_COLOR_OPTIONS = {
    "V10_hotpink": {
        "nowind":   "#00C7B1",  # same turquoise as V9
        "fullwind": "#FF1493",  # deep/hot pink — higher saturation, bluer hue
    },
    "V11_magenta": {
        "nowind":   "#00D4BC",  # slightly brighter turquoise
        "fullwind": "#D946EF",  # magenta (Tailwind fuchsia-500) — strong blue undertone
    },
    "V12_violet": {
        "nowind":   "#14C4C4",  # cyan-leaning turquoise
        "fullwind": "#9333EA",  # violet — leaves red family entirely
    },
}


def _render_combined(per40_colors, out_name):
    """V9 recipe but with overridable per40 colours."""
    per_wind_color = {
        ("per240", "no"):   WIND_COLOR_MAP.get("no",   "#1f77b4"),
        ("per240", "full"): WIND_COLOR_MAP.get("full", "#d62728"),
        ("per40",  "no"):   per40_colors["nowind"],
        ("per40",  "full"): per40_colors["fullwind"],
    }

    def _sc(ax, df, per_tag):
        for amp in ALL_AMPS:
            mk = AMP_MARKERS[amp]
            sub_amp = df[np.isclose(df["WaveAmplitudeInput [Volt]"], amp)]
            for wind in ALL_WINDS:
                sub = sub_amp[sub_amp["WindCondition"] == wind]
                if sub.empty:
                    continue
                ax.scatter(sub[KA], sub[RATIO],
                           marker=mk,
                           color=per_wind_color[(per_tag, wind)],
                           s=42, alpha=0.82, edgecolors="black", linewidths=0.3,
                           zorder=3)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    _sc(ax, m_per240, "per240")
    _sc(ax, m_per40,  "per40")

    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.7, alpha=0.55)
    ax.set_xlim(XLIM_C); ax.set_ylim(YLIM_C)
    ax.set_xlabel(r"$ka$  (Inn, målt)", fontsize=10)
    ax.set_ylabel(r"$K_t$", fontsize=10)
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.xaxis.set_minor_locator(MultipleLocator(0.01))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.02))
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(True, which="minor", alpha=0.10)

    wl = {"no": "uten vind", "full": "med vind"}
    handles = [
        mlines.Line2D([], [], color=per_wind_color[("per240", "no")],
                      marker="o", ls="None", ms=6, mec="black", mew=0.3,
                      label=f"per240 · {wl['no']}"),
        mlines.Line2D([], [], color=per_wind_color[("per240", "full")],
                      marker="o", ls="None", ms=6, mec="black", mew=0.3,
                      label=f"per240 · {wl['full']}"),
        mlines.Line2D([], [], color=per_wind_color[("per40",  "no")],
                      marker="o", ls="None", ms=6, mec="black", mew=0.3,
                      label=f"per40 · {wl['no']}"),
        mlines.Line2D([], [], color=per_wind_color[("per40",  "full")],
                      marker="o", ls="None", ms=6, mec="black", mew=0.3,
                      label=f"per40 · {wl['full']}"),
    ]
    leg = ax.legend(handles=handles, title="Kjøringstype · vind",
                    title_fontsize=8, fontsize=8,
                    loc="upper right", framealpha=0.92)
    ax.add_artist(leg)
    _legend_amp(ax, loc="lower left")

    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{out_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


for tag, cols in PER40_COLOR_OPTIONS.items():
    _render_combined(cols, f"{tag}_combined_per240_per40")


print("\nDone. Output PNGs:")
for p in sorted(OUT_DIR.glob("V*.png")):
    print(f"  {p.relative_to(BASE)}")
print(f"\nFull thesis scope summary: {len(m)} runs | "
      f"{len(ALL_WINDS)} winds ({ALL_WINDS}) | "
      f"{len(ALL_AMPS)} amps ({[f'{a:.2f}' for a in ALL_AMPS]}) | "
      f"{len(ALL_FREQS)} freqs ({[f'{f:.2f}' for f in ALL_FREQS]})")
print(f"ka range: [{m[KA].min():.4f}, {m[KA].max():.4f}]")
print(f"ratio range: [{m[RATIO].min():.4f}, {m[RATIO].max():.4f}]")
