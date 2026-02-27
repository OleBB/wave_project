#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_style.py
=============
Visual constants and styling helpers for wavescripts plots.

Contents:
    - Colour maps and marker/line style lookups
    - Legend configuration presets
    - Anchored text helper
    - Thesis-quality rcParams (apply_thesis_style)
"""

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText


# ── Colour / marker / line-style maps ─────────────────────────────────────────

WIND_COLOR_MAP = {
    "full":    "#D62728",   # red
    "lowest":  "#2CA02C",   # green
    "no":      "#1F77B4",   # blue
}

MARKERS = [
    'o', 's', '^', 'v', 'D', '*', 'P', 'X', 'p', 'h',
    '+', 'x', '.', ',', '|', '_', 'd', '<', '>', '1', '2', '3', '4'
]

PANEL_STYLES = {
    "no":      "solid",
    "full":    "dashed",
    "reverse": "solid",
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


# ── Legend presets ─────────────────────────────────────────────────────────────

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


def apply_legend(ax, plotvariables: dict) -> None:
    """
    Apply legend to *ax* based on plotvariables["plotting"]["legend"].
    Falls back silently if no handles exist or legend is "none".
    """
    legend_pos = plotvariables.get("plotting", {}).get("legend", None)
    if legend_pos is None or legend_pos == "none":
        return

    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return

    config = LEGEND_CONFIGS.get(legend_pos)
    if config is None:
        return

    ncol = min(len(labels), 5) if legend_pos in ("below", "above") else 1
    ax.legend(**config, ncol=ncol, **_LEGEND_PROPS)


# ── Anchored annotation box ────────────────────────────────────────────────────

def draw_anchored_text(ax, txt="Figuren", loc="upper left",
                       fontsize=9, facecolor="white",
                       edgecolor="gray", alpha=0.85) -> None:
    """Add a small framed text box anchored to a corner of *ax*."""
    at = AnchoredText(
        txt,
        loc=loc,
        prop=dict(size=fontsize, color="black"),
        frameon=True,
        pad=0.3,
    )
    at.patch.set_facecolor(facecolor)
    at.patch.set_edgecolor(edgecolor)
    at.patch.set_alpha(alpha)
    at.patch.set_boxstyle("round,pad=0.4,rounding_size=0.2")
    ax.add_artist(at)


# ── Thesis rcParams ────────────────────────────────────────────────────────────

def apply_thesis_style(usetex: bool = False) -> None:
    """
    Apply thesis-quality matplotlib rcParams.

    Parameters
    ----------
    usetex : bool
        True  → renders text through LaTeX (beautiful, slow, requires LaTeX install)
        False → uses matplotlib's mathtext (fast, good enough for drafts)

    Usage
    -----
    Call once at the top of a notebook or script:
        from wavescripts.plot_style import apply_thesis_style
        apply_thesis_style()          # draft
        apply_thesis_style(usetex=True)  # final
    """
    plt.rcParams.update({
        # Font
        "text.usetex":        usetex,
        "font.family":        "serif",
        "font.serif":         ["Computer Modern"] if usetex else ["DejaVu Serif"],
        "font.size":          10,
        "axes.labelsize":     10,
        "axes.titlesize":     11,
        "legend.fontsize":    9,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,

        # Figure
        "figure.figsize":     (5.5, 3.8),   # ~ half A4 width
        "figure.dpi":         150,           # screen preview

        # Saving
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.format":     "pdf",

        # Lines / axes
        "lines.linewidth":    1.2,
        "axes.linewidth":     0.8,
        "grid.alpha":         0.3,
        "axes.grid":          True,
    })
