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

import re
from pathlib import Path
from typing import Optional

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText


# ═══════════════════════════════════════════════════════════════════════════════
# STYLE & CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

WIND_COLOR_MAP = {
    "full":    "#D62728",   # red
    "lowest":  "#2CA02C",   # green
    "no":      "#1F77B4",   # blue
}

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
        apply_thesis_style()             # draft
        apply_thesis_style(usetex=True)  # final
    """
    plt.rcParams.update({
        "text.usetex":       usetex,
        "font.family":       "serif",
        "font.serif":        ["Computer Modern"] if usetex else ["DejaVu Serif"],
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
    """[2,3] → '2og3'  |  2 → '2'  |  None → 'allprobes'"""
    if val is None:
        return "allprobes"
    if isinstance(val, (list, tuple)):
        return "og".join(str(int(p)) for p in val)
    return str(int(val))


def build_filename(plot_type: str, meta: dict) -> str:
    """
    Build canonical figure filename (no extension) from meta dict.

    Pattern:
        {chapter}_{plot_type}_{panel}panel-{wind}wind-amp{amp}-freq{freq}-probe{probes}

    Example:
        '05_timeseries_reversepanel-fullwind-amp0100-freq0650-probe2og3'
    """
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
                   extra: Optional[dict] = None) -> dict:
    """
    Extract figure metadata from a plotvariables dict.

    Parameters
    ----------
    plotvariables : dict
        Standard plot-config dict with 'filters' and 'plotting' keys.
    chapter : str
        Two-digit chapter prefix, e.g. '05'.
    extra : dict, optional
        Additional immutable fields for the stub comments,
        e.g. {"run_id": "2024-11-03_run2", "script": "main.py"}.
    """
    f = plotvariables.get("filters", {})
    p = plotvariables.get("plotting", {})
    meta = {
        "chapter":   chapter,
        "panel":     f.get("PanelCondition"),
        "wind":      f.get("WindCondition"),
        "amplitude": f.get("WaveAmplitudeInput [Volt]"),
        "frequency": f.get("WaveFrequencyInput [Hz]"),
        "probes":    p.get("probes"),
        "figsize":   p.get("figsize"),
    }
    if extra:
        meta.update(extra)
    return meta


# ── File writers ───────────────────────────────────────────────────────────────

def _save_figure(fig: plt.Figure, filename: str,
                 save_pdf: bool = True,
                 save_pgf: bool = True) -> list[Path]:
    """
    Save fig to FIGURES_DIR as .pdf and/or .pgf.
    Returns list of saved paths (used to populate stub comments).
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    if save_pdf:
        p = FIGURES_DIR / f"{filename}.pdf"
        fig.savefig(p)
        saved.append(p)
        print(f"  Saved: {p}")
    if save_pgf:
        p = FIGURES_DIR / f"{filename}.pgf"
        fig.savefig(p)
        saved.append(p)
        print(f"  Saved: {p}")
    return saved


def _label_probe(filename: str, fallback_idx: int) -> str:
    m = re.search(r"probe(\w+)", filename)
    return m.group(1) if m else str(fallback_idx + 1)


def _build_subfigure_block(filename: str, label_suffix: str,
                            width: str = "0.48") -> str:
    return (
        f"  \\begin{{subfigure}}[b]{{{width}\\linewidth}}\n"
        f"    \\centering\n"
        f"    \\includegraphics[width=\\linewidth]{{FIGURES/{filename}.pdf}}\n"
        f"    \\caption{{TODO}}\n"
        f"    \\label{{fig:TODO_{label_suffix}}}\n"
        f"  \\end{{subfigure}}"
    )


def write_figure_stub(meta: dict, plot_type: str,
                      panel_filenames: Optional[list[str]] = None,
                      force: bool = False) -> None:
    """
    Write a LaTeX figure stub to TEXFIGU_DIR.

    Created ONCE — re-running the plot script will NOT overwrite your
    edited caption unless force=True.

    Parameters
    ----------
    meta : dict
        From build_fig_meta(). Used for filename and immutable comment block.
    plot_type : str
        e.g. 'timeseries', 'psd', 'damping_freq', 'swell_scatter'
    panel_filenames : list[str], optional
        Filenames (no extension) of individual panel PDFs.
        1 file  → single \\includegraphics
        2+ files → \\subfigure layout (arrange freely in Texifier)
        None → single panel using build_filename(plot_type, meta)
    force : bool
        Overwrite existing stub — WIPES caption edits.
        Tip: commit to git first.
    """
    TEXFIGU_DIR.mkdir(parents=True, exist_ok=True)
    stub_filename = build_filename(plot_type, meta)
    tex_path      = TEXFIGU_DIR / f"{stub_filename}.tex"

    if tex_path.exists() and not force:
        print(f"  Stub exists (not overwriting): {tex_path.name}")
        return

    # ── Immutable comment block ───────────────────────────────────────────────
    known_keys = {"chapter", "panel", "wind", "amplitude", "frequency",
                  "probes", "figsize", "script"}

    def _line(key, val):
        if isinstance(val, list):
            val = ", ".join(str(v) for v in val)
        return f"%   {key:<16}: {val}"

    comment_lines = [
        "%! TEX root = ../main.tex",
        "% " + "=" * 60,
        "% IMMUTABLE — generated automatically, do not edit this block",
        f"%   script          : {meta.get('script', 'plotter.py')}",
        f"%   plot_type       : {plot_type}",
        _line("chapter",   meta.get("chapter",   "?")),
        _line("panel",     meta.get("panel",     "?")),
        _line("wind",      meta.get("wind",      "?")),
        _line("amplitude", meta.get("amplitude", "?")),
        _line("frequency", meta.get("frequency", "?")),
        _line("probes",    meta.get("probes",    "?")),
    ]
    for k, v in meta.items():
        if k not in known_keys and v is not None:
            comment_lines.append(_line(k, v))

    panels = panel_filenames or [stub_filename]
    comment_lines += [
        "%",
        "% PANELS AVAILABLE:",
        *[f"%   FIGURES/{pf}.pdf" for pf in panels],
        "% " + "=" * 60,
        "",
    ]

    # ── Figure body ───────────────────────────────────────────────────────────
    if len(panels) == 1:
        body = (
            "\\begin{figure}[htbp]\n"
            "  \\centering\n"
            f"  \\includegraphics[width=0.9\\linewidth]{{FIGURES/{panels[0]}.pdf}}\n"
            "  \\caption[Short caption for LOF]{\n"
            "    % TODO: write caption\n"
            "  }\n"
            f"  \\label{{fig:TODO_{panels[0][-25:]}}}\n"
            "\\end{figure}\n"
        )
    else:
        subfigs = []
        for i, pf in enumerate(panels):
            subfigs.append(_build_subfigure_block(pf, _label_probe(pf, i)))
        body = (
            "\\begin{figure}[htbp]\n"
            "  \\centering\n"
            + "\n  \\hfill\n".join(subfigs) + "\n"
            "  \\caption[Short caption for LOF]{\n"
            "    % TODO: write caption\n"
            "  }\n"
            f"  \\label{{fig:TODO_{stub_filename[-30:]}}}\n"
            "\\end{figure}\n"
        )

    tex_path.write_text("\n".join(comment_lines) + body, encoding="utf-8")
    print(f"  Stub created: {tex_path.name}")


def save_and_stub(fig: plt.Figure,
                  meta: dict,
                  plot_type: str,
                  panel_filenames: Optional[list[str]] = None,
                  save_pdf: bool = True,
                  save_pgf: bool = True,
                  force_stub: bool = False) -> None:
    """
    Save figure files and write the LaTeX stub in one call.

    This is the single function every plotter function calls at the end
    when save_plot=True.

    Parameters
    ----------
    fig : plt.Figure
    meta : dict
        From build_fig_meta().
    plot_type : str
        e.g. 'timeseries', 'psd', 'damping_freq', 'swell_scatter'
    panel_filenames : list[str], optional
        When the stub should reference multiple separate panel PDFs.
        None → stub references only the single figure being saved now.
    save_pdf, save_pgf : bool
    force_stub : bool
        Overwrite existing stub (wipes caption edits — commit to git first).

    Example
    -------
    if plotvariables["plotting"].get("save_plot"):
        meta = build_fig_meta(plotvariables, chapter="05",
                              extra={"script": "plotter.py::plot_timeseries"})
        save_and_stub(fig, meta, plot_type="timeseries")
    """
    filename = build_filename(plot_type, meta)
    _save_figure(fig, filename, save_pdf=save_pdf, save_pgf=save_pgf)
    write_figure_stub(meta, plot_type,
                      panel_filenames=panel_filenames,
                      force=force_stub)
