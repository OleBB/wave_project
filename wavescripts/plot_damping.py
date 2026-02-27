#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_damping.py
===============
Damping ratio (P3/P2) plot functions.

Exploration (seaborn, never saved to thesis):
    explore_damping_vs_freq()
    explore_damping_vs_amp()

Thesis-ready (matplotlib, individual panels saveable):
    plot_damping_freq()       ← replaces plot_damping_results
    plot_damping_scatter()    ← unchanged, now lives here

Internal helper:
    _draw_damping_freq_ax()   ← shared drawing logic

Archive candidates (do not import):
    plot_damping_results      ← superseded by plot_damping_freq
    plot_damping_pro          ← broken labels, superseded
    plot_damping_combined     ← superseded
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from wavescripts.constants import GlobalColumns as GC
from wavescripts.plot_style import WIND_COLOR_MAP, PANEL_MARKERS, apply_legend
from wavescripts.plot_utils import (
    build_fig_meta, build_filename,
    _save_figure, write_figure_stub,
)


# ── Internal drawing primitive ────────────────────────────────────────────────

def _draw_damping_freq_ax(ax: plt.Axes,
                          stats_df: pd.DataFrame,
                          panel: str,
                          wind: str) -> None:
    """
    Draw damping ratio vs frequency onto a single axes object.

    This is the shared primitive used by both the exploration grid
    (show_plot) and the per-panel save loop (save_plot).
    Keeping the drawing logic here means it only exists once.

    Parameters
    ----------
    ax : plt.Axes
    stats_df : pd.DataFrame
        Output from damping_all_amplitude_grouper().
    panel : str
        Panel condition to subset, e.g. 'reverse'.
    wind : str
        Wind condition to subset, e.g. 'full'.
    """
    mask   = ((stats_df[GC.PANEL_CONDITION_GROUPED] == panel) &
              (stats_df[GC.WIND_CONDITION] == wind))
    subset = stats_df[mask]

    if subset.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        ax.set_title(f"{panel} / {wind}", fontsize=9)
        return

    # Unity reference line — P3/P2 = 1 means no damping
    ax.axhline(1.0, color="black", linestyle="--",
               linewidth=0.8, alpha=0.4, label="Unity (no damping)")

    for amp in sorted(subset[GC.WAVE_AMPLITUDE_INPUT].unique()):
        amp_data = subset[subset[GC.WAVE_AMPLITUDE_INPUT] == amp] \
                       .sort_values(GC.WAVE_FREQUENCY_INPUT)

        ax.errorbar(
            amp_data[GC.WAVE_FREQUENCY_INPUT],
            amp_data["mean_P3P2"],
            yerr=amp_data["std_P3P2"],
            marker="o",
            label=f"{amp:.2f} V",
            capsize=3,
            alpha=0.8,
            linewidth=1.4,
        )

    ax.set_xlabel("Frequency [Hz]", fontsize=9)
    ax.set_ylabel("P3/P2", fontsize=9)
    ax.set_title(f"{panel}panel / {wind}wind", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Amplitude", fontsize=7, title_fontsize=7)


# ── Exploration functions (seaborn) ───────────────────────────────────────────

def explore_damping_vs_freq(df: pd.DataFrame,
                             plotvariables: dict) -> None:
    """
    Seaborn facet: P3/P2 vs frequency, faceted by amplitude.
    For exploration only — not saveable as individual thesis panels.
    """
    x = GC.WAVE_FREQUENCY_INPUT
    g = sns.relplot(
        data=df.sort_values(x),
        x=x,
        y="mean_P3P2",
        hue=GC.WIND_CONDITION,
        palette=WIND_COLOR_MAP,
        style=GC.PANEL_CONDITION_GROUPED,
        style_order=["no", "all"],
        col=GC.WAVE_AMPLITUDE_INPUT,
        kind="line",
        marker=True,
        facet_kws={"sharex": True, "sharey": True},
        height=3.0,
        aspect=1.2,
        errorbar=None,
    )
    for ax, (amp, sub) in zip(g.axes.flat,
                               df.groupby(GC.WAVE_AMPLITUDE_INPUT)):
        for (wind, panel), gsub in sub.groupby(
                [GC.WIND_CONDITION, GC.PANEL_CONDITION_GROUPED]):
            ax.errorbar(gsub[x], gsub["mean_P3P2"], yerr=gsub["std_P3P2"],
                        fmt="none", capsize=3, alpha=0.5)

    sns.move_legend(g, "lower center",
                    bbox_to_anchor=(0.5, 1), ncol=3,
                    title=None, frameon=False)
    g.figure.suptitle("Damping P3/P2 vs Frequency  [explore]",
                       y=1.04, fontsize=11)
    plt.tight_layout()
    plt.show()


def explore_damping_vs_amp(df: pd.DataFrame,
                            plotvariables: dict) -> None:
    """
    Seaborn facet: P3/P2 vs amplitude, faceted by frequency.
    For exploration only.
    """
    x = GC.WAVE_AMPLITUDE_INPUT
    sns.set_style("ticks", {"axes.grid": True})
    g = sns.relplot(
        data=df.sort_values(x),
        x=x,
        y="mean_P3P2",
        hue=GC.WIND_CONDITION,
        palette=WIND_COLOR_MAP,
        style=GC.PANEL_CONDITION_GROUPED,
        style_order=["no", "all"],
        col=GC.WAVE_FREQUENCY_INPUT,
        kind="line",
        marker=True,
        facet_kws={"sharex": True, "sharey": True},
        height=3.0,
        aspect=1.2,
        errorbar=None,
    )
    for ax, (freq, sub) in zip(g.axes.flat,
                                df.groupby(GC.WAVE_FREQUENCY_INPUT)):
        for (wind, panel), gsub in sub.groupby(
                [GC.WIND_CONDITION, GC.PANEL_CONDITION_GROUPED]):
            ax.errorbar(gsub[x], gsub["mean_P3P2"], yerr=gsub["std_P3P2"],
                        fmt="none", capsize=3, alpha=0.5)

    sns.move_legend(g, "lower center",
                    bbox_to_anchor=(0.5, 1), ncol=3,
                    title=None, frameon=False)
    g.figure.suptitle("Damping P3/P2 vs Amplitude  [explore]",
                       y=1.04, fontsize=11)
    plt.tight_layout()
    plt.show()


# ── Thesis-ready function ─────────────────────────────────────────────────────

def plot_damping_freq(stats_df: pd.DataFrame,
                      plotvariables: dict,
                      chapter: str = "05") -> None:
    """
    Damping ratio (P3/P2) vs frequency.

    show_plot=True  → full panel×wind grid in one figure (exploration)
    save_plot=True  → saves one PDF/PGF per (panel, wind) combination
                      and writes a single .tex stub listing all panels

    The same _draw_damping_freq_ax() primitive draws both views,
    so what you see in exploration is exactly what gets saved.

    Parameters
    ----------
    stats_df : pd.DataFrame
        Output from damping_all_amplitude_grouper().
    plotvariables : dict
        Standard plotvariables dict. Relevant plotting keys:
            show_plot : bool (default True)
            save_plot : bool (default False)
            save_pgf  : bool (default True)
            figsize   : tuple for the exploration grid
    chapter : str
        Two-digit chapter prefix for filenames.
    """
    plotting  = plotvariables.get("plotting", {})
    show_plot = plotting.get("show_plot", True)
    save_plot = plotting.get("save_plot", False)

    panel_conditions = sorted(stats_df[GC.PANEL_CONDITION_GROUPED].unique())
    wind_conditions  = sorted(stats_df[GC.WIND_CONDITION].unique())

    # ── Exploration: full grid in one figure ──────────────────────────────────
    if show_plot:
        n_rows = len(panel_conditions)
        n_cols = len(wind_conditions)
        figsize = plotting.get("figsize", (4.5 * n_cols, 3.5 * n_rows))

        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=figsize,
                                 squeeze=False,
                                 sharex=True, sharey=True)

        for i, panel in enumerate(panel_conditions):
            for j, wind in enumerate(wind_conditions):
                _draw_damping_freq_ax(axes[i, j], stats_df, panel, wind)

        fig.suptitle("Damping Ratio P3/P2 vs Frequency", fontsize=13, y=1.0)
        plt.tight_layout()
        plt.show()

    # ── Save: one figure per (panel, wind) panel ──────────────────────────────
    if save_plot:
        panel_filenames = []
        meta_base = build_fig_meta(plotvariables, chapter=chapter,
                                   extra={"script": "plot_damping.py::plot_damping_freq"})

        for panel in panel_conditions:
            for wind in wind_conditions:
                fig_s, ax_s = plt.subplots(figsize=(5.0, 3.8))
                _draw_damping_freq_ax(ax_s, stats_df, panel, wind)
                fig_s.tight_layout()

                # Per-panel meta overrides for filename
                panel_meta = {**meta_base,
                              "panel": panel,
                              "wind":  wind}
                fname = build_filename("damping_freq", panel_meta)
                _save_figure(fig_s, fname,
                             save_pdf=True,
                             save_pgf=plotting.get("save_pgf", True))
                panel_filenames.append(fname)
                plt.close(fig_s)

        # One stub that references all panels — arrange freely in Texifier
        stub_meta = {**meta_base,
                     "panel": panel_conditions,
                     "wind":  wind_conditions}
        write_figure_stub(stub_meta, "damping_freq",
                          panel_filenames=panel_filenames)


# ── Scatter (unchanged logic, moved here, save_plot added) ───────────────────

def plot_damping_scatter(stats_df: pd.DataFrame,
                         plotvariables: Optional[dict] = None,
                         show_errorbars: bool = True,
                         size_by_amplitude: bool = False,
                         chapter: str = "05") -> None:
    """
    Single scatter: P3/P2 ratio for all conditions, coloured by wind.

    Parameters
    ----------
    stats_df : pd.DataFrame
        Output from damping_all_amplitude_grouper().
    plotvariables : dict, optional
        If provided, uses show_plot / save_plot / figsize from plotting section.
        If None, defaults to show=True, save=False.
    show_errorbars : bool
    size_by_amplitude : bool
        Vary marker size by WaveAmplitudeInput.
    chapter : str
    """
    if plotvariables is None:
        plotvariables = {"plotting": {"show_plot": True, "save_plot": False}}

    plotting  = plotvariables.get("plotting", {})
    show_plot = plotting.get("show_plot", True)
    save_plot = plotting.get("save_plot", False)
    figsize   = plotting.get("figsize", (10, 6))

    sns.set_style("ticks", {"axes.grid": True})
    fig, ax = plt.subplots(figsize=figsize)

    plot_data = stats_df.sort_values(GC.WAVE_FREQUENCY_INPUT)

    scatter_kwargs = dict(
        data=plot_data,
        x=GC.WAVE_FREQUENCY_INPUT,
        y="mean_P3P2",
        hue=GC.WIND_CONDITION,
        palette=WIND_COLOR_MAP,
        style=GC.PANEL_CONDITION_GROUPED,
        style_order=["no", "all"],
        alpha=0.75,
        ax=ax,
        legend="auto",
    )
    if size_by_amplitude:
        scatter_kwargs["size"]  = GC.WAVE_AMPLITUDE_INPUT
        scatter_kwargs["sizes"] = (50, 200)

    sns.scatterplot(**scatter_kwargs)

    if show_errorbars and "std_P3P2" in plot_data.columns:
        for wind in plot_data[GC.WIND_CONDITION].unique():
            wd = plot_data[plot_data[GC.WIND_CONDITION] == wind]
            ax.errorbar(wd[GC.WAVE_FREQUENCY_INPUT], wd["mean_P3P2"],
                        yerr=wd["std_P3P2"],
                        fmt="none",
                        ecolor=WIND_COLOR_MAP.get(wind, "gray"),
                        elinewidth=1, capsize=3, alpha=0.4, zorder=1)

    # Unity line
    ax.axhline(1.0, color="black", linestyle="--",
               linewidth=0.8, alpha=0.4, label="Unity")

    ax.set_xlabel("Frequency [Hz]", fontsize=11)
    ax.set_ylabel("P3/P2  (mean ± std)", fontsize=11)
    ax.set_title("Damping Ratio: All Conditions", fontsize=12, fontweight="bold")
    ax.legend(loc="best", framealpha=0.9, fontsize=9)
    plt.tight_layout()

    if save_plot:
        meta = build_fig_meta(plotvariables, chapter=chapter,
                              extra={"script": "plot_damping.py::plot_damping_scatter"})
        fname = build_filename("damping_scatter", meta)
        _save_figure(fig, fname,
                     save_pdf=True,
                     save_pgf=plotting.get("save_pgf", True))
        write_figure_stub(meta, "damping_scatter",
                          panel_filenames=[fname])

    if show_plot:
        plt.show()
    else:
        plt.close(fig)
