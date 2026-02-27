#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_swell.py
=============
P2 vs P3 swell amplitude scatter plots.

Thesis-ready:
    plot_swell_scatter()   ← replaces plot_p2_vs_p3_scatter
                              + best features from plot_swell_p2_vs_p3_by_wind

Archive candidates (do not import):
    old_plot_p2_vs_p3_scatter   ← no wind/panel encoding, prototype
    plot_swell_p2_vs_p3_by_wind ← Δ mean feature folded into plot_swell_scatter
    plot_p2_p3_bars             ← never used

What was merged from plot_swell_p2_vs_p3_by_wind:
    - Shared axis limits across all panels (lo/hi computed globally)
    - Δ mean annotation in each panel
    - Proper axis labels only on edge subplots
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wavescripts.constants import (
    ProbeColumns as PC,
    GlobalColumns as GC,
)
from wavescripts.filters import filter_for_amplitude_plot
from wavescripts.plot_style import WIND_COLOR_MAP, PANEL_MARKERS
from wavescripts.plot_utils import (
    build_fig_meta, build_filename,
    _save_figure, write_figure_stub,
)


# ── Band definitions ──────────────────────────────────────────────────────────

BAND_COLS = {
    "Swell": PC.SWELL_AMPLITUDE_PSD,
    "Wind":  PC.WIND_AMPLITUDE_PSD,
    "Total": PC.TOTAL_AMPLITUDE_PSD,
}


# ── Internal drawing primitive ────────────────────────────────────────────────

def _draw_swell_scatter_ax(ax: plt.Axes,
                            band_amplitudes: pd.DataFrame,
                            p2_col: str,
                            p3_col: str,
                            band_name: str,
                            shared_lim: Optional[Tuple[float, float]] = None,
                            annotate_delta: bool = True,
                            show_legend: bool = True) -> None:
    """
    Draw one P2 vs P3 scatter panel onto *ax*.

    Parameters
    ----------
    ax : plt.Axes
    band_amplitudes : pd.DataFrame
        Already-filtered dataframe.
    p2_col, p3_col : str
        Column names for P2 and P3 amplitudes.
    band_name : str
        Used for title and missing-data message.
    shared_lim : (float, float), optional
        (lo, hi) axis limits. If None, computed from this panel's data.
    annotate_delta : bool
        Show Δ mean = P3 - P2 in upper-left corner.
    show_legend : bool
    """
    has_wind  = GC.WIND_CONDITION  in band_amplitudes.columns
    has_panel = GC.PANEL_CONDITION in band_amplitudes.columns

    if p2_col not in band_amplitudes.columns or \
       p3_col not in band_amplitudes.columns:
        ax.text(0.5, 0.5, "Missing\ncolumns",
                ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="gray")
        ax.set_title(f"{band_name} band", fontweight="bold")
        return

    p2_all = band_amplitudes[p2_col].to_numpy()
    p3_all = band_amplitudes[p3_col].to_numpy()
    valid  = np.isfinite(p2_all) & np.isfinite(p3_all)

    if valid.sum() == 0:
        ax.text(0.5, 0.5, "No valid\ndata",
                ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="gray")
        ax.set_title(f"{band_name} band", fontweight="bold")
        return

    # ── Axis limits ───────────────────────────────────────────────────────────
    if shared_lim is not None:
        lo, hi = shared_lim
    else:
        lo = min(p2_all[valid].min(), p3_all[valid].min())
        hi = max(p2_all[valid].max(), p3_all[valid].max())
        pad = 0.05 * (hi - lo) if hi > lo else 0.1
        lo, hi = lo - pad, hi + pad

    # ── Scatter points ────────────────────────────────────────────────────────
    winds  = band_amplitudes[GC.WIND_CONDITION].unique()  if has_wind  else [None]
    panels = band_amplitudes[GC.PANEL_CONDITION].unique() if has_panel else [None]

    for wind in winds:
        for panel in panels:
            mask = np.ones(len(band_amplitudes), dtype=bool)
            if wind  is not None: mask &= (band_amplitudes[GC.WIND_CONDITION]  == wind)
            if panel is not None: mask &= (band_amplitudes[GC.PANEL_CONDITION] == panel)

            if mask.sum() == 0:
                continue

            color  = WIND_COLOR_MAP.get(wind, "steelblue") if wind  else "steelblue"
            marker = PANEL_MARKERS.get(panel, "o")         if panel else "o"
            label  = f"{wind}/{panel}" if (wind and panel) else (wind or panel or "")

            ax.scatter(
                p2_all[mask], p3_all[mask],
                s=60, alpha=0.75,
                color=color, marker=marker,
                edgecolors="white", linewidths=0.5,
                label=label,
                rasterized=True,
            )

    # ── Reference line y = x ─────────────────────────────────────────────────
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.9, alpha=0.45,
            zorder=1, label="y = x")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_title(f"{band_name} band", fontweight="bold", fontsize=9)

    # ── Δ mean annotation ─────────────────────────────────────────────────────
    if annotate_delta:
        delta = (p3_all[valid] - p2_all[valid]).mean()
        ax.text(0.04, 0.96, f"Δ mean = {delta:+.3f} mm",
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=7.5, color="#333",
                bbox=dict(boxstyle="round,pad=0.2",
                          facecolor="white", alpha=0.6, edgecolor="none"))

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=6.5, loc="lower right",
                      framealpha=0.85, markerscale=0.85)


def _compute_shared_lim(band_amplitudes: pd.DataFrame,
                         col_templates: dict) -> Tuple[float, float]:
    """Compute shared axis limits across all bands."""
    all_vals = []
    for template in col_templates.values():
        for probe in (2, 3):
            col = template.format(i=probe)
            if col in band_amplitudes.columns:
                vals = band_amplitudes[col].to_numpy()
                all_vals.append(vals[np.isfinite(vals)])

    if not all_vals:
        return 0.0, 1.0

    combined = np.concatenate(all_vals)
    lo, hi = combined.min(), combined.max()
    pad = 0.05 * (hi - lo) if hi > lo else 0.1
    return lo - pad, hi + pad


# ── Main function ─────────────────────────────────────────────────────────────

def plot_swell_scatter(meta_df: pd.DataFrame,
                       plotvariables: dict,
                       chapter: str = "05",
                       share_axes: bool = True,
                       annotate_delta: bool = True) -> None:
    """
    P2 vs P3 amplitude scatter for Swell, Wind, and Total bands.

    show_plot=True  → all three bands side-by-side in one exploration figure,
                      plus a data-summary panel on the right
    save_plot=True  → saves one PDF/PGF per band, writes one .tex stub
                      with three subfigures

    Replaces:
        plot_p2_vs_p3_scatter       (all bands, wind+panel encoding)
        plot_swell_p2_vs_p3_by_wind (Δ mean, shared limits, edge-only labels)

    Parameters
    ----------
    meta_df : pd.DataFrame
        Full combined_meta_sel (filtering applied internally).
    plotvariables : dict
        Standard plotvariables dict. Relevant plotting keys:
            show_plot      : bool (default True)
            save_plot      : bool (default False)
            save_pgf       : bool (default True)
            figsize        : tuple for exploration figure
    chapter : str
    share_axes : bool
        If True, all three band panels use the same axis limits.
    annotate_delta : bool
        Show Δ mean = mean(P3 - P2) in each panel.
    """
    plotting  = plotvariables.get("plotting", {})
    show_plot = plotting.get("show_plot", True)
    save_plot = plotting.get("save_plot", False)

    # ── Filter ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("plot_swell_scatter — filtering data")
    band_amplitudes = filter_for_amplitude_plot(meta_df, plotvariables)

    if band_amplitudes.empty:
        print("No data after filtering — aborting.")
        return

    n = len(band_amplitudes)
    print(f"  {n} rows remaining after filter")

    # ── Shared axis limits ────────────────────────────────────────────────────
    shared_lim = _compute_shared_lim(band_amplitudes, BAND_COLS) \
                 if share_axes else None

    # ── Exploration figure (all bands + info panel) ───────────────────────────
    if show_plot:
        n_bands = len(BAND_COLS)
        figsize = plotting.get("figsize", (14, 5))

        # Main bands + narrow info panel on the right
        fig = plt.figure(figsize=figsize)
        gs  = fig.add_gridspec(1, n_bands + 1,
                               width_ratios=[1] * n_bands + [0.38])
        axes    = [fig.add_subplot(gs[0, i]) for i in range(n_bands)]
        info_ax = fig.add_subplot(gs[0, -1])
        info_ax.axis("off")

        # Data summary text
        def _fmt(vals, fmt="{:.3f}"):
            return ", ".join(fmt.format(v) for v in sorted(vals)
                             if v is not None and str(v) != "nan")

        has_wind  = GC.WIND_CONDITION  in band_amplitudes.columns
        has_panel = GC.PANEL_CONDITION in band_amplitudes.columns
        has_freq  = GC.WAVE_FREQUENCY_INPUT in band_amplitudes.columns
        has_amp   = GC.WAVE_AMPLITUDE_INPUT in band_amplitudes.columns

        lines = [
            "DATA SUMMARY",
            "─" * 22,
            f"n = {n}",
            "",
        ]
        if has_wind:
            for w in sorted(band_amplitudes[GC.WIND_CONDITION].unique()):
                c = (band_amplitudes[GC.WIND_CONDITION] == w).sum()
                lines.append(f"W:{w}  n={c}")
        if has_panel:
            lines.append("")
            for p in sorted(band_amplitudes[GC.PANEL_CONDITION].unique()):
                c = (band_amplitudes[GC.PANEL_CONDITION] == p).sum()
                lines.append(f"P:{p}  n={c}")
        if has_freq:
            lines += ["", "Freq [Hz]:"]
            lines.append(_fmt(band_amplitudes[GC.WAVE_FREQUENCY_INPUT].unique()))
        if has_amp:
            lines += ["", "Amp [V]:"]
            lines.append(_fmt(band_amplitudes[GC.WAVE_AMPLITUDE_INPUT].unique(), "{:.2f}"))

        info_ax.text(0.05, 0.97, "\n".join(lines),
                     transform=info_ax.transAxes,
                     fontsize=7, verticalalignment="top",
                     fontfamily="monospace",
                     bbox=dict(boxstyle="round", facecolor="wheat",
                               alpha=0.3, edgecolor="none"))

        for ax, (band_name, template) in zip(axes, BAND_COLS.items()):
            _draw_swell_scatter_ax(
                ax, band_amplitudes,
                p2_col=template.format(i=2),
                p3_col=template.format(i=3),
                band_name=band_name,
                shared_lim=shared_lim,
                annotate_delta=annotate_delta,
                show_legend=True,
            )
            ax.set_xlabel("P2 amplitude [mm]", fontsize=9)
            ax.set_ylabel("P3 amplitude [mm]", fontsize=9)

        fig.suptitle("P2 vs P3 Amplitude — Swell / Wind / Total",
                     fontsize=12, fontweight="bold", y=1.01)
        plt.tight_layout()
        plt.show()

    # ── Save: one figure per band ─────────────────────────────────────────────
    if save_plot:
        panel_filenames = []
        meta_base = build_fig_meta(plotvariables, chapter=chapter,
                                   extra={"script": "plot_swell.py::plot_swell_scatter"})

        for band_name, template in BAND_COLS.items():
            fig_s, ax_s = plt.subplots(figsize=(4.5, 4.2))

            _draw_swell_scatter_ax(
                ax_s, band_amplitudes,
                p2_col=template.format(i=2),
                p3_col=template.format(i=3),
                band_name=band_name,
                shared_lim=shared_lim,
                annotate_delta=annotate_delta,
                show_legend=True,
            )
            ax_s.set_xlabel("P2 amplitude [mm]", fontsize=9)
            ax_s.set_ylabel("P3 amplitude [mm]", fontsize=9)
            fig_s.tight_layout()

            band_meta = {**meta_base,
                         "band": band_name.lower()}
            fname = build_filename(f"swell_scatter_{band_name.lower()}", band_meta)
            _save_figure(fig_s, fname,
                         save_pdf=True,
                         save_pgf=plotting.get("save_pgf", True))
            panel_filenames.append(fname)
            plt.close(fig_s)

        # One stub with three subfigures — arrange freely in Texifier
        write_figure_stub(meta_base, "swell_scatter",
                          panel_filenames=panel_filenames)
