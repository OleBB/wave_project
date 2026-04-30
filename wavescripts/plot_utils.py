#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_utils.py
=============
All plotting infrastructure for the wavescripts project.

Contents
--------
STYLE & CONSTANTS
    WIND_COLOR_MAP, PANEL_STYLES, PANEL_MARKERS, MARKER_STYLES, MARKERS
    LEGEND_CONFIGS, apply_legend, draw_anchored_text, apply_thesis_style

LABEL BUILDER
    make_label

SAVE INFRASTRUCTURE
    build_fig_meta       — extract meta from plotvariables dict
    build_filename       — canonical filename from meta
    _save_figure         — save .pdf and/or .pgf
    write_figure_stub    — write .tex stub (once, then hands-off)
    save_and_stub        — combined entry point for plotter functions

Output directories (mirror your TeX project, copy manually)
------------------------------------------------------------
    output/FIGURES/   →  /Users/ole/main/FIGURES/
    output/TEXFIGU/   →  /Users/ole/main/TEXFIGU/
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

from wavescripts.wave_physics import calculate_wavenumbers_vectorized


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL SAVE FLAGS
# ═══════════════════════════════════════════════════════════════════════════════

# Set to True when exporting final thesis figures (requires LaTeX + slow).
# False = PDF only (fast prototype mode).  Override with plot_utils.SAVE_PGF = True.
SAVE_PGF: bool = False

# Folder names of the active PROCESSED_DIRS — written into every stub's immutable block.
# Set once at startup: import wavescripts.plot_utils as _pu; _pu.ACTIVE_DATASETS = [...]
ACTIVE_DATASETS: list[str] = []

# ── Physics parameters used for x-axis conversion ────────────────────────────
# Note: calculate_wavenumbers_vectorized expects depth in mm (it applies MM_TO_M internally)
# and returns k in rad/m.
_TANK_DEPTH_MM: float = 580.0   # nominal lab depth in mm (depth580 in filenames)


def freq_to_k(
    freqs,
    depth_mm: float = _TANK_DEPTH_MM,
) -> np.ndarray:
    """Convert paddle input frequencies [Hz] to wavenumber k [rad/m].

    Uses the full dispersion relation ω² = gk·tanh(kd). depth_mm is in mm
    (as expected by calculate_wavenumbers_vectorized); returned k is in rad/m.

    Replaces the old freq_to_kL (which multiplied by a reference panel
    length to give a dimensionless kL). Axes now plot pure wavenumber; no
    reference length is implied.
    """
    freqs = np.asarray(freqs, dtype=float)
    return calculate_wavenumbers_vectorized(freqs, np.full_like(freqs, depth_mm))


def add_freq_axis(
    ax,
    depth_mm: float = _TANK_DEPTH_MM,
    label: str = "frequency (Hz)",
):
    """Add a secondary x-axis showing input frequency [Hz] above a k axis.

    Uses the full dispersion relation ω² = gk·tanh(kd) — same as freq_to_k.
    Forward (k → Hz): analytical, no iteration.
    Inverse (Hz → k): calls freq_to_k (iterative solver already implemented).

    Returns the secondary Axes so the caller can adjust tick labels if needed.
    """
    g = 9.81  # m/s²
    d = depth_mm * 1e-3  # mm → m

    def _k_to_freq(k):
        k = np.asarray(k, dtype=float)
        # guard against k=0 to avoid sqrt(0*tanh(0)) edge case
        with np.errstate(invalid="ignore", divide="ignore"):
            omega = np.where(k > 0, np.sqrt(g * k * np.tanh(k * d)), 0.0)
        return omega / (2 * np.pi)

    def _freq_to_k(f):
        f = np.asarray(f, dtype=float)
        return freq_to_k(f, depth_mm=depth_mm)

    secax = ax.secondary_xaxis("top", functions=(_k_to_freq, _freq_to_k))
    secax.set_xlabel(label, fontsize=8)
    secax.tick_params(labelsize=7)
    return secax


# ═══════════════════════════════════════════════════════════════════════════════
# STYLE & CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

WIND_COLOR_MAP = {
    "full":    "#D62728",   # red
    "lowest":  "#2CA02C",   # green
    "no":      "#1F77B4",   # blue
}


# ── Reader-facing amplitude labels ────────────────────────────────────────
# 2026-04-24: paddle drive voltages (0.10/0.20/0.30 V) are a lab-specific
# signal. For thesis plots and captions, we use A_1/A_2/A_3 — reader-friendly
# tiered amplitude identifiers. Within the thesis scope (1.3–1.6 Hz) the
# mapping to measured amplitude (IN-mean FFT) is well-defined (±3 %):
#     A_1 ≈ 7.5 mm     (paddle V = 0.10 V)
#     A_2 ≈ 15.0 mm    (paddle V = 0.20 V)
#     A_3 ≈ 21.5 mm    (paddle V = 0.30 V)
# The linearity 0.2/0.1 = 2.0 is exact; 0.3/0.1 = 2.87 shows a mild paddle
# sub-linearity at highest amplitude. See the 2026-04-24 paddle-voltage →
# measured-amplitude audit (170 nowind+fullpanel runs).
_AMP_TIERS = (
    (0.10, r"$A_1$", "A1",  7.5),
    (0.20, r"$A_2$", "A2", 15.0),
    (0.30, r"$A_3$", "A3", 21.5),
)
AMP_LABEL = {v: lbl for v, lbl, _, _ in _AMP_TIERS}   # reader-facing (mathtext)
AMP_TAG   = {v: t   for v, _, t, _ in _AMP_TIERS}     # file-name tag
AMP_MM    = {v: mm  for v, _, _, mm in _AMP_TIERS}    # nominal measured mm


def amp_to_label(v, default: Optional[str] = None) -> str:
    """Reader-facing amplitude label (e.g. 0.20 → ``$A_2$``).

    Use this in axis titles, legends, subfigure captions. If ``v`` is not
    one of the canonical tiers, returns ``default`` (fallback: ``"V = x"``
    formatted string).
    """
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default if default is not None else "—"
    for canon, lbl in AMP_LABEL.items():
        if abs(v - canon) < 1e-3:
            return lbl
    return default if default is not None else f"V = {v:.2f}"


WIND_LABEL = {
    "no":      "uten",
    "lowest":  "liten",
    "full":    "full",
}


def wind_to_label(w, default: Optional[str] = None) -> str:
    """Reader-facing wind label (e.g. ``"no"`` → ``"uten"``).

    Norwegian short forms for legend entries on damping / transmission
    plots: uten (no wind), liten (lowest), full (full). Legends title the
    group as ``"vind"``. Unknown wind tags fall back to the raw string.
    """
    try:
        return WIND_LABEL.get(str(w), str(w) if default is None else default)
    except Exception:
        return default if default is not None else "—"


def amp_to_tag(v, default: Optional[str] = None) -> str:
    """File-name tag for one paddle amplitude (0.20 → ``"A2"``).

    Replaces the old ``"{int(round(v*100)):02d}V"`` convention. Fallback for
    non-canonical values preserves the old tag so legacy data never breaks.
    """
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default if default is not None else "NA"
    for canon, tag in AMP_TAG.items():
        if abs(v - canon) < 1e-3:
            return tag
    return default if default is not None else f"{int(round(v * 100)):02d}V"

PANEL_STYLES = {
    "no":      "solid",
    "full":    "dashed",
    "reverse": "dashdot",
}

PANEL_MARKERS = {
    "no":      "o",
    "full":    "s",
    "reverse": "^",
}

MARKER_STYLES = {
    "full":    "*",
    "no":      "<",
    "lowest":  ">",
}

MARKERS = [
    'o', 's', '^', 'v', 'D', '*', 'P', 'X', 'p', 'h',
    '+', 'x', '.', ',', '|', '_', 'd', '<', '>', '1', '2', '3', '4',
]

LEGEND_CONFIGS = {
    "outside_right":      {"loc": "center left",  "bbox_to_anchor": (1.02, 0.5)},
    "outside_left":       {"loc": "center right", "bbox_to_anchor": (-0.02, 0.5)},
    "inside":             {"loc": "best"},
    "inside_upper_right": {"loc": "upper right"},
    "inside_upper_left":  {"loc": "upper left"},
    "below": {
        "loc": "upper center",
        "bbox_to_anchor": (0.5, -0.15),
        "ncol": 3,
    },
    "above": {
        "loc": "lower center",
        "bbox_to_anchor": (0.5, 1.02),
        "ncol": 3,
    },
    "none": None,
}

_LEGEND_PROPS = {
    "framealpha":    0.9,
    "fontsize":      8,
    "labelspacing":  0.3,
    "handlelength":  1.5,
    "handletextpad": 0.5,
}


def apply_legend(ax: plt.Axes, plotvariables: dict) -> None:
    """
    Apply legend to *ax* based on plotvariables["plotting"]["legend"].
    Silently does nothing if no handles exist or legend is 'none'/None.
    """
    legend_pos = plotvariables.get("plotting", {}).get("legend", None)
    if not legend_pos or legend_pos == "none":
        return
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    config = LEGEND_CONFIGS.get(legend_pos)
    if config is None:
        return
    ncol = min(len(labels), 5) if legend_pos in ("below", "above") else 1
    ax.legend(**config, ncol=ncol, **_LEGEND_PROPS)


def draw_anchored_text(ax: plt.Axes, txt: str = "Figuren",
                        loc: str = "upper left", fontsize: int = 9,
                        facecolor: str = "white", edgecolor: str = "gray",
                        alpha: float = 0.85) -> None:
    """Add a small framed text box anchored to a corner of *ax*."""
    at = AnchoredText(
        txt, loc=loc,
        prop=dict(size=fontsize, color="black"),
        frameon=True, pad=0.3,
    )
    at.patch.set_facecolor(facecolor)
    at.patch.set_edgecolor(edgecolor)
    at.patch.set_alpha(alpha)
    at.patch.set_boxstyle("round,pad=0.4,rounding_size=0.2")
    ax.add_artist(at)


def apply_horizontal_ylabel(ax, label: str, *,
                            fontsize: Optional[int] = None,
                            y_offset: float = 1.02) -> None:
    """
    Place ``label`` horizontally above the leftmost edge of ``ax``'s y-tick
    labels — matches the ch04_plateau_overview / ch05_damping_all_data_scatter
    convention. Drops the rotated side label so the plot uses the full
    horizontal width.

    Call this AFTER subplots_adjust / tight_layout, BEFORE savefig — it draws
    the canvas once to read the current tick-label positions.

    For multi-pane figures with shared y-axis, call only on the topmost pane.
    """
    kwargs = dict(rotation=0, ha="left", va="bottom")
    if fontsize is not None:
        kwargs["fontsize"] = fontsize
    ax.set_ylabel(label, **kwargs)
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ticks = [t for t in ax.yaxis.get_ticklabels()
             if t.get_visible() and t.get_text().strip()]
    if not ticks:
        return
    left_disp = min(t.get_window_extent(renderer=renderer).x0 for t in ticks)
    x_axes = ax.transAxes.inverted().transform((left_disp, 0))[0]
    ax.yaxis.set_label_coords(x_axes, y_offset)


def apply_thesis_style(usetex: bool = False) -> None:
    """
    Apply thesis-quality matplotlib rcParams.

    Parameters
    ----------
    usetex : bool
        False → fast draft (matplotlib mathtext)
        True  → final quality (requires LaTeX install, slow compile)

    Usage
    -----
    Call once at the top of your notebook/script before any plotting:
        from wavescripts.plot_utils import apply_thesis_style
        apply_thesis_style()             # draft (NCM via OTF, no LaTeX)
        apply_thesis_style(usetex=True)  # final (LaTeX + Computer Modern)

    In draft mode (default), NewComputerModern OTFs shipped with TeX Live
    are registered with matplotlib's FontManager so text in the figures
    matches the thesis body font without a LaTeX round-trip. Math uses the
    built-in Computer Modern mathtext set (visually identical to NCM for
    standard symbols). If NCM OTFs can't be located, falls back silently
    to DejaVu Serif.
    """
    # Register NCM OTFs once per process (no-op after first call).
    if not usetex:
        _register_ncm_fonts()

    serif_list = (["Computer Modern"] if usetex
                  else ["NewComputerModern10", "DejaVu Serif"])

    plt.rcParams.update({
        "text.usetex":       usetex,
        "font.family":       "serif",
        "font.serif":        serif_list,
        "mathtext.fontset":  "cm",
        "font.size":         10,
        "axes.labelsize":    10,
        "axes.titlesize":    11,
        "legend.fontsize":   9,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "figure.figsize":    (5.5, 3.8),
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "savefig.format":    "pdf",
        "lines.linewidth":   1.2,
        "axes.linewidth":    0.8,
        "grid.alpha":        0.3,
        "axes.grid":         True,
    })


_NCM_REGISTERED = False


def _register_ncm_fonts() -> None:
    """Register NewComputerModern OpenType files with matplotlib.

    Looks up the OTFs under TeX Live's texmf-dist tree. Idempotent — safe
    to call repeatedly. If the files aren't present (no TeX Live install),
    prints a single warning and returns without raising.
    """
    global _NCM_REGISTERED
    if _NCM_REGISTERED:
        return
    from matplotlib import font_manager as _fm
    # Known TeX Live locations (macOS MacPorts/Homebrew, Linux TUG installer).
    _candidates = [
        Path("/usr/local/texlive/2025/texmf-dist/fonts/opentype/public/newcomputermodern"),
        Path("/usr/local/texlive/2024/texmf-dist/fonts/opentype/public/newcomputermodern"),
        Path("/opt/homebrew/texlive/texmf-dist/fonts/opentype/public/newcomputermodern"),
    ]
    ncm_dir = next((p for p in _candidates if p.exists()), None)
    if ncm_dir is None:
        print("   warn: NewComputerModern OTFs not found on this machine — "
              "falling back to DejaVu Serif.")
        _NCM_REGISTERED = True
        return
    for _fname in ("NewCM10-Regular.otf", "NewCM10-Bold.otf",
                   "NewCM10-Italic.otf", "NewCM10-BoldItalic.otf",
                   "NewCMMath-Regular.otf"):
        try:
            _fm.fontManager.addfont(str(ncm_dir / _fname))
        except Exception as _e:
            print(f"   warn: could not register {_fname}: {_e}")
    _NCM_REGISTERED = True
    
def _top_k_indices(values: np.ndarray, k: int) -> np.ndarray:
    """
    Fast selection of top k indices using partial sorting.
    
    Parameters
    ----------
    values : np.ndarray
        Array of numeric values
    k : int
        Number of top values to select
    
    Returns
    -------
    np.ndarray
        Indices of top k values, sorted in descending order
    """
    if k is None or k <= 0 or k >= values.size:
        return np.arange(values.size)
    
    # Use argpartition for O(n) selection
    part = np.argpartition(values, -k)[-k:]
    
    # Sort the selected indices by their values (descending)
    return part[np.argsort(values[part])[::-1]]


# ═══════════════════════════════════════════════════════════════════════════════
# LABEL BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def make_label(row) -> str:
    """
    Short legend label from a metadata row (pd.Series or dict).
    Format: W:full_P:reverse_A:0.10V_f:1.3Hz
    Only includes fields that are present and non-None.
    """
    parts = []
    wind  = row.get("WindCondition")
    panel = row.get("PanelCondition")
    amp   = row.get("WaveAmplitudeInput [Volt]")
    freq  = row.get("WaveFrequencyInput [Hz]")
    if wind  is not None: parts.append(f"W:{wind}")
    if panel is not None: parts.append(f"P:{panel}")
    if amp   is not None: parts.append(f"A:{float(amp):.2f}V")
    if freq  is not None: parts.append(f"f:{freq}Hz")
    return "_".join(parts) if parts else "unknown"


# ═══════════════════════════════════════════════════════════════════════════════
# SAVE INFRASTRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════

FIGURES_DIR = Path("output/FIGURES")
TEXFIGU_DIR = Path("output/TEXFIGU")

# ── Centralised figure-caption lookup ─────────────────────────────────────────
# main_save_figures.py owns FIGURE_CAPTIONS (full) + FIGURE_CAPTIONS_SHORT
# (LOF entries) and writes both to output/.figure_captions.json on import.
# Both inline plotters (same process) and delegated subprocess scripts read
# captions from that JSON file via the helper below, keyed by figure_name
# (== .tex stem == .pdf stem == \label suffix). Empty value (or missing key)
# → stub renders a TODO placeholder for full, omits the [short] arg for short.
#
# JSON cache format:
#   { "full":  {<figure_name>: <full caption text>, ...},
#     "short": {<figure_name>: <short caption text for LOF>, ...} }

_CAPTIONS_CACHE_PATH = Path("output/.figure_captions.json")


def _lookup_central_caption(figure_name: str, *, kind: str = "full") -> str:
    """Return the user-authored caption for ``figure_name`` from the JSON
    cache written by main_save_figures.py, or ``""`` if missing/unreadable.

    Parameters
    ----------
    figure_name : str
        Stub filename without .tex (== .pdf stem == \\label suffix).
    kind : {'full', 'short'}
        'full'  → text for the figure body's \\caption{...}.
        'short' → text for the optional [short] arg (LOF entry).
    """
    if not figure_name:
        return ""
    if not _CAPTIONS_CACHE_PATH.exists():
        return ""
    try:
        data = json.loads(_CAPTIONS_CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    if not isinstance(data, dict):
        return ""
    bucket = data.get(kind, {})
    if not isinstance(bucket, dict):
        return ""
    val = bucket.get(figure_name, "")
    return val if isinstance(val, str) else ""


# ── Filename format helpers ────────────────────────────────────────────────────

def _fmt_condition(val) -> str:
    """['no','full'] → 'no-full'  |  'reverse' → 'reverse'  |  None → 'all'"""
    if val is None:
        return "all"
    if isinstance(val, (list, tuple)):
        return "-".join(str(v).lower() for v in val)
    return str(val).lower()


def _fmt_amp(val) -> str:
    """0.1 → '0100'  |  [0.1,0.2] → '0100-0200'  |  None → 'allamp'"""
    if val is None:
        return "allamp"
    vals = val if isinstance(val, (list, tuple)) else [val]
    return "-".join(f"{float(v) * 1000:04.0f}" for v in vals)


def _fmt_freq(val) -> str:
    """0.65 → '0650'  |  [0.65,1.3] → '0650-1300'  |  None → 'allfreq'"""
    if val is None:
        return "allfreq"
    vals = val if isinstance(val, (list, tuple)) else [val]
    return "-".join(f"{float(v) * 1000:04.0f}" for v in vals)


def _fmt_probes(val) -> str:
    """['9373/170','12545/250'] → '9373-170og12545-250'  |  None → 'allprobes'"""
    if val is None:
        return "allprobes"
    def _fmt_one(p):
        s = str(p).replace("/", "-")
        return s
    if isinstance(val, (list, tuple)):
        return "og".join(_fmt_one(p) for p in val)
    return _fmt_one(val)


def build_filename(plot_type: str, meta: dict) -> str:
    """
    Build canonical figure filename (no extension) from meta dict.

    When ``figure_name`` is set in *meta*, it is returned directly —
    keeping filenames short and human-readable.

    Fallback pattern (used only when figure_name is absent):
        {chapter}_{plot_type}_{panel}panel-{wind}wind-amp{amp}-freq{freq}-probe{probes}

    Example fallback:
        '05_timeseries_reversepanel-fullwind-amp0100-freq0650-probe2og3'
    """
    if meta.get("figure_name"):
        return meta["figure_name"]
    chapter = str(meta.get("chapter", "00"))
    panel   = _fmt_condition(meta.get("panel"))
    wind    = _fmt_condition(meta.get("wind"))
    amp     = _fmt_amp(meta.get("amplitude"))
    freq    = _fmt_freq(meta.get("frequency"))
    probes  = _fmt_probes(meta.get("probes"))
    return (
        f"{chapter}_{plot_type}_"
        f"{panel}panel-{wind}wind-"
        f"amp{amp}-freq{freq}-"
        f"probe{probes}"
    )


def build_fig_meta(plotvariables: dict,
                   chapter: str = "05",
                   extra: Optional[dict] = None,
                   data_df=None,
                   *,
                   computed_in: Optional[str] = None,
                   data_class: Optional[str] = None,
                   findings_doc: Optional[str] = None,
                   run_category: Optional[str] = None,
                   quality_flag: Optional[str] = None,
                   mooring: Optional[str] = None,
                   grouper: Optional[str] = None,
                   collapse_panels: Optional[bool] = None,
                   fft_window_hz: Optional[float] = None,
                   extra_params: Optional[str] = None,
                   extra_stats: Optional[dict] = None,
                   max_run_paths: int = 20) -> dict:
    """
    Extract figure metadata from a plotvariables dict.

    The returned dict drives ``write_figure_stub`` — it populates the
    "IMMUTABLE" comment block at the top of every ``output/TEXFIGU/*.tex``
    stub so the full scientific provenance of the figure is visible
    without re-reading the generator script.

    Parameters
    ----------
    plotvariables : dict
        Standard plot-config dict with 'filters' and 'plotting' keys.
    chapter : str
        Two-digit chapter prefix, e.g. '05'.
    extra : dict, optional
        Additional free-form fields merged into the final meta dict.
    data_df : pandas.DataFrame, optional
        The filtered dataframe that went into the figure. When provided:
          - n_runs              : int(len(data_df))
          - in_probes_used      : unique IN-probe compositions (e.g. "9373/170+9373/340")
          - out_probes_used     : unique OUT-probe compositions
          - probe_configs       : unique probe configurations (e.g. "march2026_better_rearranging")
          - non_final_config_n  : count of rows not using the latest probe config
          - run_paths           : full CSV paths when ``len(data_df) <= max_run_paths``
                                  (wavedata/<folder>/<file>.csv per row); otherwise
                                  left blank so the ``datasets`` block carries the
                                  provenance instead.

    Keyword-only schema fields (all optional, blank when unset):

    computed_in : str
        Where the plotted values were actually calculated, e.g.
        ``"filters.py::damping_grouper -> plotter.py::plot_damping_freq"``.
        Use this to point at the function that did the real aggregation /
        computation, not just the plotting wrapper.
    data_class : str
        One of ``"META"``, ``"DFS"``, ``"DELEG"``, ``"CSV"`` — matches the
        ``# [DATA: X]`` tag in the main_save_figures.py cell.
    findings_doc : str
        Relative path to an ``analysis_scratch/<name>_findings.md`` that
        discusses the figure in depth (if any).
    run_category, quality_flag, mooring : str
        Filter predicates that shaped which runs entered the figure.
    grouper : str
        Name of the grouping function used (``"damping_grouper"``,
        ``"damping_all_amplitude_grouper"``, or ``""`` when none).
    collapse_panels : bool
        Whether the grouper collapsed fullpanel + reversepanel runs.
    fft_window_hz : float
        Bandwidth of the FFT amplitude window (default 0.1 Hz in
        ``compute_amplitudes_from_fft``).
    extra_params : str
        Free-form line for script-specific numerical parameters, e.g.
        ``"band_half_hz=0.05, wind_band_hz=2-6, fs=250"``.
    extra_stats : dict
        Summary statistics cited in the caption, rendered as ``stat:<key>``
        lines in the immutable block. Makes the caption numbers
        reproducible from the stub alone.
    max_run_paths : int
        Threshold below which the contributing CSV paths are listed in
        full. When the data has more rows than this, the ``run_paths``
        slot stays empty and only ``datasets`` is written.
    """
    f = plotvariables.get("filters", {})
    p = plotvariables.get("plotting", {})
    meta = {
        "chapter":         chapter,
        "panel":           f.get("PanelCondition"),
        "wind":            f.get("WindCondition"),
        "amplitude":       f.get("WaveAmplitudeInput [Volt]"),
        "frequency":       f.get("WaveFrequencyInput [Hz]"),
        "probes":          p.get("probes"),
        "figsize":         p.get("figsize"),
        "caption":         p.get("caption"),
        "figure_name":     p.get("figure_name"),
        "draft":           p.get("draft", False),
        # New schema fields (blank when the caller doesn't supply them):
        "computed_in":     computed_in,
        "data_class":      data_class,
        "findings_doc":    findings_doc,
        "run_category":    run_category if run_category is not None else f.get("run_category"),
        "quality_flag":    quality_flag if quality_flag is not None else f.get("quality_flag"),
        "mooring":         mooring     if mooring     is not None else f.get("Mooring"),
        "grouper":         grouper,
        "collapse_panels": collapse_panels,
        "fft_window_hz":   fft_window_hz,
        "extra_params":    extra_params,
        "extra_stats":     dict(extra_stats) if extra_stats else {},
    }
    if data_df is not None and hasattr(data_df, "columns") and len(data_df):
        meta["n_runs"] = int(len(data_df))
        # Probe composition per row — distinct strings tell the reader
        # which probes contributed. If >1 distinct, the data spans eras.
        for col in ("in_probes_used", "out_probes_used"):
            if col in data_df.columns:
                uniq = sorted({str(v) for v in data_df[col].dropna().unique() if str(v).strip()})
                if uniq:
                    meta[col] = uniq if len(uniq) > 1 else uniq[0]
        # Probe config names if available. Falls back to mapping via
        # file_date (per-row) or file_dates (per-group, list; from
        # damping_all_amplitude_grouper).
        if "probe_config_name" in data_df.columns:
            uniq = sorted({str(v) for v in data_df["probe_config_name"].dropna().unique() if str(v).strip()})
            if uniq:
                meta["probe_configs"] = uniq if len(uniq) > 1 else uniq[0]
        elif "file_date" in data_df.columns or "file_dates" in data_df.columns:
            try:
                from wavescripts.improved_data_loader import (
                    get_configuration_for_date, PROBE_CONFIGS,
                )
                from datetime import datetime as _dt
                final_name = PROBE_CONFIGS[-1].name
                # Collect date strings (per-row or pooled from per-group lists)
                date_strs = []
                if "file_date" in data_df.columns:
                    date_strs = [str(v) for v in data_df["file_date"].dropna()]
                if "file_dates" in data_df.columns:
                    for cell in data_df["file_dates"].dropna():
                        if isinstance(cell, (list, tuple)):
                            date_strs.extend(str(v) for v in cell)
                        else:
                            date_strs.append(str(cell))
                cfg_names = set()
                non_final = 0
                for s in date_strs:
                    try:
                        fd = _dt.fromisoformat(s)
                        cfg = get_configuration_for_date(fd)
                        cfg_names.add(cfg.name)
                        if cfg.name != final_name:
                            non_final += 1
                    except Exception:
                        continue
                if cfg_names:
                    uniq = sorted(cfg_names)
                    meta["probe_configs"] = uniq if len(uniq) > 1 else uniq[0]
                if non_final > 0:
                    meta["non_final_config_n"] = non_final
            except Exception:
                pass
        # run_paths — full CSV paths when the contributing set is small
        # enough to be worth listing verbatim. Above the threshold we
        # leave it blank so the datasets block carries the provenance.
        if "path" in data_df.columns and meta["n_runs"] <= max_run_paths:
            paths = [str(v) for v in data_df["path"].dropna().unique() if str(v).strip()]
            if paths:
                # Strip any repo-root absolute prefix to keep the stub
                # portable and the block narrow.
                rel_paths = []
                for raw in paths:
                    raw = raw.strip()
                    if "wavedata/" in raw:
                        raw = raw[raw.index("wavedata/"):]
                    rel_paths.append(raw)
                meta["run_paths"] = rel_paths
    if extra:
        meta.update(extra)
    return meta


# ── File writers ───────────────────────────────────────────────────────────────

def _save_figure(fig: plt.Figure, filename: str,
                 save_pdf: bool = True,
                 save_pgf: bool = True) -> list[Path]:
    """
    Save fig to FIGURES_DIR as .pdf and/or .pgf.

    PGF is gated by the module-level SAVE_PGF flag (default False) so the
    prototype run stays fast.  Set plot_utils.SAVE_PGF = True before import
    (or patch at runtime) when exporting final thesis figures.

    PGF failures are also caught and warned rather than crashing — PGF
    requires a full LaTeX install and can fail on special characters.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    # Tight save — apply_thesis_style sets savefig.bbox=tight; pad_inches=0.02
    # trims the default 0.1" breathing room so the saved PDF runs flush with
    # the content (matches the all_data_damping_scatter / inspirational
    # convention adopted 2026-04-30).
    if save_pdf:
        p = FIGURES_DIR / f"{filename}.pdf"
        fig.savefig(p, pad_inches=0.02)
        saved.append(p)
        print(f"  Saved: {p}")
    if save_pgf and SAVE_PGF:
        p = FIGURES_DIR / f"{filename}.pgf"
        try:
            fig.savefig(p, pad_inches=0.02)
            saved.append(p)
            print(f"  Saved: {p}")
        except Exception as exc:
            print(f"  WARNING: PGF save failed for {filename} — {exc.__class__.__name__}: {exc!s:.120}")
    return saved


def _label_probe(filename: str, fallback_idx: int) -> str:
    m = re.search(r"probe(\w+)", filename)
    return m.group(1) if m else str(fallback_idx + 1)


def _build_subfigure_block(filename: str, label_suffix: str = "",
                            width: str = "0.48",
                            subcaption: str = "TODO") -> str:
    """Subfigure environment for one PDF.

    The ``\\label`` is locked to ``fig:{filename}`` so the figure-name ↔
    label invariant (``figure_name == .pdf stem == .tex stem == \\label
    suffix``) holds for subfigures too. The ``label_suffix`` parameter is
    retained for back-compat with existing callers but ignored.
    """
    return (
        f"  \\begin{{subfigure}}[b]{{{width}\\linewidth}}\n"
        f"    \\centering\n"
        f"    \\includegraphics[width=\\linewidth]{{FIGURES/{filename}.pdf}}\n"
        f"    \\caption{{{subcaption}}}\n"
        f"    \\label{{fig:{filename}}}\n"
        f"  \\end{{subfigure}}"
    )


# ── Immutable-block schema (keeps the stub audit-friendly) ────────────────────
#
# A stub's comment block answers "everything you'd ever want to know about
# this figure" without re-reading the script that generated it. The schema
# below defines the deterministic set of slots. Every slot is printed even
# when empty so a future reader always finds the answer in the same place
# (the user explicitly prefers empty slots over silently-absent slots).
#
# When the stub is regenerated, only this block is refreshed — the figure
# body (\caption text, \label, subfigure layout) is preserved so
# hand-edited captions survive re-runs. Use force=True in write_figure_stub
# to clobber the body too (typically only right after the initial write).

_IMMUTABLE_OPEN  = "% =============================================================="
_IMMUTABLE_CLOSE = "% ── end immutable block ─────────────────────────────────────────"


def _fmt_stub_value(val) -> str:
    """Render a stub-block value — None/empty become '' (blank slot)."""
    if val is None:
        return ""
    if isinstance(val, bool):
        return str(val)
    if isinstance(val, (list, tuple)):
        return ", ".join(str(v) for v in val)
    return str(val)


def _build_immutable_block(meta: dict, plot_type: str,
                            subfig_filenames: Optional[list[str]] = None) -> list[str]:
    """
    Construct the immutable comment block for a figure stub.

    Returns a list of lines (without a trailing newline) ready to be joined
    with ``"\\n"``. The block starts with ``%! TEX root`` so it's valid at
    the top of a .tex file; it ends with an explicit close marker so
    surgical updates can replace it precisely.
    """
    from datetime import datetime as _dt

    def L(key, val, *, width=18):
        """Format one `%   key: val` line with the stub value renderer."""
        return f"%   {key:<{width}}: {_fmt_stub_value(val)}"

    figure_name   = meta.get("figure_name") or ""
    label         = f"fig:{figure_name}" if figure_name else ""

    # Caption text shown in the IMMUTABLE comment block — read from the same
    # central source the LaTeX body uses. Manual; no auto-derivation from full.
    caption_full  = (meta.get("caption")
                     or _lookup_central_caption(figure_name, kind="full"))
    caption_short = (meta.get("caption_short")
                     or _lookup_central_caption(figure_name, kind="short"))

    subfig_files  = subfig_filenames or [figure_name] if figure_name else []
    datasets_list = list(ACTIVE_DATASETS) if ACTIVE_DATASETS else []
    run_paths     = meta.get("run_paths") or []
    extra_stats   = meta.get("extra_stats") or {}

    lines = [
        "%! TEX root = ../main.tex",
        _IMMUTABLE_OPEN,
        "% IMMUTABLE — generated automatically, do not edit this block",
        "%",
        "% — Provenance ───────────────────────────────────────────────────",
        L("script",        meta.get("script", "plotter.py")),
        L("computed_in",   meta.get("computed_in")),
        L("plot_type",     plot_type),
        L("data_class",    meta.get("data_class")),
        L("generated_at",  _dt.now().isoformat(timespec="seconds")),
        L("findings_doc",  meta.get("findings_doc")),
        "%",
        "% — Thesis context ──────────────────────────────────────────────",
        L("chapter",       meta.get("chapter")),
        L("caption_label", label),
        L("caption_short", caption_short),
        "%",
        "% — Filters ────────────────────────────────────────────────────",
        L("panel",         meta.get("panel")),
        L("wind",          meta.get("wind")),
        L("amplitude [V]", meta.get("amplitude")),
        L("frequency [Hz]",meta.get("frequency")),
        L("probes",        meta.get("probes")),
        L("run_category",  meta.get("run_category")),
        L("quality_flag",  meta.get("quality_flag")),
        L("mooring",       meta.get("mooring")),
        "%",
        "% — Data provenance ────────────────────────────────────────────",
        L("n_runs",              meta.get("n_runs")),
        L("in_probes_used",      meta.get("in_probes_used")),
        L("out_probes_used",     meta.get("out_probes_used")),
        L("probe_configs",       meta.get("probe_configs")),
        L("non_final_config_n",  meta.get("non_final_config_n")),
        "%   datasets        :",
    ]
    for ds in datasets_list:
        lines.append(f"%     {ds}")
    if not datasets_list:
        lines.append("%     (none registered — set plot_utils.ACTIVE_DATASETS)")

    lines.append("%   run_paths       :")
    if run_paths:
        for rp in run_paths:
            lines.append(f"%     {rp}")
    else:
        # Either the figure aggregates too many runs to list (>max_run_paths
        # passed to build_fig_meta) or the source data is not row-based.
        lines.append("%     (omitted — aggregate over > threshold runs, see datasets)")

    lines += [
        "%",
        "% — Method / numerical parameters ─────────────────────────────",
        L("grouper",         meta.get("grouper")),
        L("collapse_panels", meta.get("collapse_panels")),
        L("fft_window_hz",   meta.get("fft_window_hz")),
        L("extra_params",    meta.get("extra_params")),
        "%",
        "% — Summary stats cited in caption ────────────────────────────",
    ]
    if extra_stats:
        for k, v in extra_stats.items():
            lines.append(f"%   stat:{k:<12} : {_fmt_stub_value(v)}")
    else:
        lines.append("%   (none registered — pass extra_stats=... to build_fig_meta)")

    lines += [
        "%",
        "% — Subfigures available ───────────────────────────────────────",
    ]
    if subfig_files:
        for pf in subfig_files:
            lines.append(f"%     FIGURES/{pf}.pdf")
    else:
        lines.append("%     (none)")

    lines += [_IMMUTABLE_CLOSE, ""]
    return lines


def _replace_immutable_block(existing: str, new_block: str) -> str:
    """
    Surgically replace the immutable block in an existing stub, preserving
    the figure body (caption text etc.). Returns the stub text to write.

    The block is delimited by ``_IMMUTABLE_OPEN`` at the top (after ``%! TEX
    root``) and ``_IMMUTABLE_CLOSE`` at the bottom. If the existing stub
    was written by an older version of this file (different delimiters or
    none at all), the whole stub is replaced — captions in that case were
    already stored in an auto-generated template and will be re-seeded.
    """
    close_idx = existing.find(_IMMUTABLE_CLOSE)
    if close_idx < 0:
        # No recognisable close marker → legacy stub, replace entirely.
        return new_block + existing_body_fallback(existing)
    # Preserve everything after the close marker (and its trailing newline).
    body_start = existing.find("\n", close_idx) + 1
    body = existing[body_start:]
    return new_block + body


def existing_body_fallback(existing: str) -> str:
    """
    For legacy stubs with no close marker, extract the LaTeX body
    (``\\begin{figure}`` onwards). Returns empty string if the body isn't
    found — caller will have the new template inject a fresh TODO body.
    """
    m = re.search(r"\\begin\{figure\}", existing)
    if not m:
        return ""
    return existing[m.start():]


def write_figure_stub(meta: dict, plot_type: str,
                      subfig_filenames: Optional[list[str]] = None,
                      subfig_captions: Optional[list[str]] = None,
                      force: bool = True,
                      width: str = "\\linewidth",
                      subfig_layout: str = "row",
                      thispagestyle: Optional[str] = None) -> None:
    """
    Write a LaTeX figure stub in TEXFIGU_DIR.

    Default behaviour (``force=True``):
      Stub body (caption text, label, subfigure layout) is rewritten on
      every call from ``meta`` and the central ``FIGURE_CAPTIONS`` dict
      (via the JSON cache). The single-source-of-truth invariant requires
      this — caption edits in ``main_save_figures.py`` must always land
      on regen.

    With ``force=False``:
      - Stub absent    → write from scratch.
      - Stub present   → refresh ONLY the immutable block between
                         ``_IMMUTABLE_OPEN`` and ``_IMMUTABLE_CLOSE``;
                         preserve the LaTeX body. Use only if you're
                         intentionally hand-editing a body (which the
                         project convention says you shouldn't).

    Parameters
    ----------
    meta : dict
        From ``build_fig_meta()``. Drives the immutable block.
    plot_type : str
        e.g. 'timeseries', 'psd', 'damping_freq', 'swell_scatter'.
    subfig_filenames : list[str], optional
        1 → single ``\\includegraphics``; 2+ → subfigure layout.
        None → single figure, filename from ``meta['figure_name']``.
    subfig_captions : list[str], optional
        Per-subfigure captions (same length as subfig_filenames). When
        omitted, each subfig caption is looked up from ``FIGURE_CAPTIONS``
        by its .pdf basename.
    force : bool
        Default True — rewrite the whole stub. See above for opt-out.
    """
    TEXFIGU_DIR.mkdir(parents=True, exist_ok=True)
    stub_filename = meta.get("figure_name") or build_filename(plot_type, meta)
    tex_path      = TEXFIGU_DIR / f"{stub_filename}.tex"

    new_block_lines = _build_immutable_block(meta, plot_type, subfig_filenames)
    new_block = "\n".join(new_block_lines)

    # ── Caption / figure-body template (used only when we write a new stub
    # from scratch, or when force=True). Hand-edited captions in existing
    # stubs are preserved by the surgical-update path above.
    #
    # Resolution order (first non-empty wins) for full + short separately:
    #   1. meta["caption"] / meta["caption_short"]  (explicit per-call override)
    #   2. FIGURE_CAPTIONS[figure_name] / FIGURE_CAPTIONS_SHORT[figure_name]
    #      via the JSON cache (single source of truth, edited in
    #      main_save_figures.py)
    #   3. ""                                       (full → TODO placeholder
    #                                                short → omit [short] arg,
    #                                                LaTeX uses full for LOF)
    _caption_full  = (meta.get("caption")
                      or _lookup_central_caption(stub_filename, kind="full"))
    _caption_short = (meta.get("caption_short")
                      or _lookup_central_caption(stub_filename, kind="short"))
    if _caption_full:
        if _caption_short:
            _caption_block = (
                f"  \\caption[{_caption_short}]{{\n"
                f"    {_caption_full}\n"
                "  }\n"
            )
        else:
            _caption_block = (
                f"  \\caption{{\n"
                f"    {_caption_full}\n"
                "  }\n"
            )
    else:
        _caption_block = (
            "  \\caption{\n"
            "    % TODO: write caption\n"
            "  }\n"
        )

    _figure_name = meta.get("figure_name") or stub_filename
    _label = f"fig:{_figure_name}"

    # Optional page-style override — e.g. ``thispagestyle="empty"`` drops
    # the page number on the page where the float lands. Issued inside
    # the figure environment so it applies to whichever page LaTeX picks.
    _pagestyle_line = (f"  \\thispagestyle{{{thispagestyle}}}\n"
                       if thispagestyle else "")

    subfig_files = subfig_filenames or [stub_filename]
    if len(subfig_files) == 1:
        body = (
            "\\begin{figure}[htbp]\n"
            "  \\centering\n"
            + _pagestyle_line
            + f"  \\includegraphics[width={width}]{{FIGURES/{subfig_files[0]}.pdf}}\n"
            + _caption_block
            + f"  \\label{{{_label}}}\n"
            "\\end{figure}\n"
        )
    else:
        # Subfigure layout:
        #   "row"    (default) — side-by-side at 0.48\linewidth, separated by
        #                        \hfill. LaTeX wraps to multi-row as needed.
        #                        figure placement = htbp.
        #   "column"           — stacked vertically at \linewidth, separated by
        #                        \\[1ex]. Suited to full-page figures (~3+
        #                        subfigures, page-only float).
        #                        figure placement = p (full-page float).
        if subfig_layout == "column":
            sub_w  = "1.0"
            sub_sep = "\n  \\\\[1ex]\n"
            placement = "p"
        else:   # "row"
            sub_w  = "0.48"
            sub_sep = "\n  \\hfill\n"
            placement = "htbp"

        subfigs = []
        for i, pf in enumerate(subfig_files):
            # Same resolution order as the parent caption: explicit kwarg first,
            # then central FIGURE_CAPTIONS by subfig pdf basename, then TODO.
            explicit = (subfig_captions[i] if subfig_captions and i < len(subfig_captions)
                        else "")
            subcap = explicit or _lookup_central_caption(pf) or "TODO"
            subfigs.append(_build_subfigure_block(pf, _label_probe(pf, i),
                                                  width=sub_w,
                                                  subcaption=subcap))
        body = (
            f"\\begin{{figure}}[{placement}]\n"
            "  \\centering\n"
            + _pagestyle_line
            + sub_sep.join(subfigs) + "\n"
            + _caption_block
            + f"  \\label{{{_label}}}\n"
            "\\end{figure}\n"
        )

    if tex_path.exists() and not force:
        existing = tex_path.read_text(encoding="utf-8")
        refreshed = _replace_immutable_block(existing, new_block)
        if refreshed == existing:
            print(f"  Stub already current: {tex_path.name}")
        else:
            tex_path.write_text(refreshed, encoding="utf-8")
            print(f"  Stub immutable block refreshed: {tex_path.name}")
        return

    tex_path.write_text(new_block + body, encoding="utf-8")
    print(f"  Stub created: {tex_path.name}")


def resolve_caption(plotting: dict,
                    default_template: str = "",
                    slots: dict | None = None,
                    fn_name: str = "plot") -> str:
    """
    Return the user-supplied ``plotting["caption"]`` template, formatted with
    *slots* if it contains ``{placeholders}``. Empty string if no caption is
    set — the caller (``write_figure_stub``) then falls back to the central
    ``FIGURE_CAPTIONS`` dict via the JSON cache.

    No agent-written defaults, no terminal print, no clipboard side-effect.
    The ``default_template`` and ``fn_name`` parameters are accepted for
    back-compat with existing call sites but ignored — the only source of
    caption text is ``plotting["caption"]`` (per-call override) or the
    central dict (looked up downstream by ``write_figure_stub``).

    Slot interpolation is kept for back-compat with the few inline cells
    that still parameterise their captions with ``{n_runs}`` / ``{wind_conds}``
    etc. After those cells migrate (Commit 3), the formatting branch is
    unused and this function can be reduced to a passthrough.
    """
    template = plotting.get("caption", "")
    if not template:
        return ""
    if slots and "{" in template:
        return template.format(**slots)
    return template


class TextRegistry:
    """
    Centralised registry for language-specific on-figure text.

    Each plotter creates one instance and calls ``T(slot_name, default=...)``
    wherever a human-readable string is drawn (title, suptitle, xlabel, ylabel,
    legend title, legend entries, prose annotations — NOT math labels, numeric
    ticks, or category tokens). The user can override any slot from
    ``plotvariables["plotting"]["text"]``::

        "text": {
            "ylabel": "Støygulv [mm]",
            "title": {"h100 / high": "Lav høyde", "h272 / high": "Ref."},
            "legend_threshold": "Terskel max({k_sigma:.0f}σ, {k_q:.0f}q)",
        }

    An override may be a plain string or a dict keyed by facet value (pass
    ``facet_key=`` to the call). Missing facet keys silently fall back to the
    plotter's default. Every resolved string runs through ``str.format`` with
    the registry's slot dict, so overrides can embed computed values
    (``{n_runs}``, ``{highlight_keyword}``, …). Unknown format keys raise
    ``KeyError`` — consistent with ``resolve_caption``. Override keys that
    match no requested slot print a warning at ``report()`` time (typo guard).

    Discoverability: after the plot is drawn, call ``T.report()`` to print the
    slot names the plotter actually used.
    """

    def __init__(self, plotting: dict, slots: dict | None = None,
                 fn_name: str = "plot") -> None:
        self._overrides = dict(((plotting or {}).get("text") or {}))
        self._slots = dict(slots or {})
        self._fn_name = fn_name
        self._requested: set[str] = set()

    def __call__(self, slot_name: str, default: str, *,
                 facet_key=None, extra_slots: dict | None = None) -> str:
        self._requested.add(slot_name)
        override = self._overrides.get(slot_name, None)

        if isinstance(override, dict):
            template = override.get(facet_key, default) if facet_key is not None else default
        elif override is not None:
            template = override
        else:
            template = default

        merged = self._slots if extra_slots is None else {**self._slots, **extra_slots}
        try:
            return template.format(**merged)
        except KeyError as e:
            raise KeyError(
                f"[{self._fn_name}] text slot '{slot_name}': unknown format "
                f"key {e}. Available slots: {sorted(merged)}"
            ) from None

    def report(self) -> None:
        print(f"\n[{self._fn_name}] text slots: {sorted(self._requested)}")
        unused = set(self._overrides) - self._requested
        if unused:
            print(
                f"[{self._fn_name}] WARNING: unused text override key(s) "
                f"{sorted(unused)} — typo? known slots: {sorted(self._requested)}"
            )


def add_draft_stamp(fig: plt.Figure) -> None:
    """Overlay a large red DRAFT watermark diagonally across the figure."""
    fig.text(0.5, 0.5, "DRAFT",
             fontsize=80, color="red", alpha=0.55,
             ha="center", va="center",
             rotation=35, fontweight="bold",
             transform=fig.transFigure,
             zorder=9999)


def save_and_stub(fig: plt.Figure,
                  meta: dict,
                  plot_type: str,
                  subfig_filenames: Optional[list[str]] = None,
                  force_stub: bool = True) -> None:
    """
    Save figure files and write the LaTeX stub in one call.

    Always saves both PDF (fast LaTeX build) and PGF (final quality).

    Parameters
    ----------
    fig : plt.Figure
    meta : dict
        From build_fig_meta().
    plot_type : str
        e.g. 'timeseries', 'psd', 'damping_freq', 'swell_scatter'
    subfig_filenames : list[str], optional
        When the stub should reference multiple separate subfigure PDFs.
        None → stub references only the single figure being saved now.
    force_stub : bool
        Default True — every regen rewrites the stub body so the latest
        caption text from the central FIGURE_CAPTIONS dict lands. Pass
        False only if you have a special reason to preserve a hand-edited
        body (which the project's invariant says you shouldn't).

    Example
    -------
    if plotvariables["plotting"].get("save_plot"):
        meta = build_fig_meta(plotvariables, chapter="05",
                              extra={"script": "plotter.py::plot_timeseries"})
        save_and_stub(fig, meta, plot_type="timeseries")
    """
    if meta.get("draft"):
        add_draft_stamp(fig)
    filename = build_filename(plot_type, meta)
    _save_figure(fig, filename, save_pdf=True, save_pgf=True)
    write_figure_stub(meta, plot_type,
                      subfig_filenames=subfig_filenames,
                      force=force_stub)
