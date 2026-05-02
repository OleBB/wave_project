#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plotter.py  (integration guide)
================================
This file shows HOW to wire plot_utils and plot_style into your existing
plotter functions. It is NOT a full rewrite — it shows the pattern for
two representative functions (one signal plot, one scalar/scatter plot)
so you can apply the same pattern to the rest yourself.

Import changes at the top of your real plotter.py
--------------------------------------------------
Replace the ad-hoc constants and helpers with:

    from wavescripts.plot_style import (
        WIND_COLOR_MAP, PANEL_STYLES, PANEL_MARKERS,
        LEGEND_CONFIGS, apply_legend, draw_anchored_text,
        apply_thesis_style,
    )
    from wavescripts.plot_utils import (
        build_fig_meta, save_and_stub, make_label,
        build_filename,            # if you need the name without saving
    )

Then delete from plotter.py:
    - WIND_COLOR_MAP, MARKERS, PANEL_STYLES, MARKER_STYLES, LEGEND_CONFIGS
    - draw_anchored_text()
    - _apply_legend(), _apply_legend_3()   ← pick one → now apply_legend()
    - _make_label(), _make_label_2()       ← now make_label()
"""

# ─────────────────────────────────────────────────────────────────────────────
# These imports replace the constants/helpers defined inline in plotter.py
# ─────────────────────────────────────────────────────────────────────────────
from wavescripts.plot_style import (
    WIND_COLOR_MAP, PANEL_STYLES, PANEL_MARKERS,
    apply_legend, draw_anchored_text,
)
from wavescripts.plot_utils import (
    build_fig_meta, save_and_stub, make_label,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN A — Signal / timeseries plot
# (maps to your plot_filtered / plot_overlayed family)
# ─────────────────────────────────────────────────────────────────────────────

def plot_timeseries(processed_dfs: dict,
                    df_sel: pd.DataFrame,
                    plotvariables: dict,
                    chapter: str = "05") -> None:
    """
    Overlay/separate timeseries plots with standardised save.

    New keys used from plotvariables["plotting"]:
        show_plot  : bool  — display the figure (default True)
        save_plot  : bool  — save pdf/pgf + tex stub (default False)
        save_pgf   : bool  — also save .pgf (default True)
    """
    plotting = plotvariables.get("plotting", {})
    show_plot = plotting.get("show_plot", True)
    save_plot = plotting.get("save_plot", False)

    chosenprobe = plotvariables["processing"]["chosenprobe"]
    figsize     = plotting.get("figsize", (10, 6))

    fig, ax = plt.subplots(figsize=figsize)

    for _, row in df_sel.iterrows():
        path_key = row["path"]
        if path_key not in processed_dfs:
            continue

        df_ma    = processed_dfs[path_key]
        color    = WIND_COLOR_MAP.get(row.get("WindCondition"), "black")
        lstyle   = PANEL_STYLES.get(row.get("PanelCondition", ""), "solid")
        label    = make_label(row)

        rangestart = plotvariables["processing"].get("rangestart")
        rangeend   = plotvariables["processing"].get("rangeend")
        df_cut = df_ma.loc[rangestart:rangeend]

        t0      = df_cut["Date"].iloc[0]
        time_ms = (df_cut["Date"] - t0).dt.total_seconds() * 1000

        probe_num  = str(chosenprobe).split()[-1]
        zeroed_col = f"eta_{probe_num}"
        y_data = df_cut[zeroed_col] if zeroed_col in df_cut.columns \
                 else df_cut[chosenprobe]

        ax.plot(time_ms, y_data, label=label, color=color, linestyle=lstyle)

    ax.set_xlabel("Time [ms]")
    ax.set_ylabel(f"η [mm]")
    ax.set_title(f"{chosenprobe}")

    apply_legend(ax, plotvariables)
    fig.tight_layout()

    # ── Save ─────────────────────────────────────────────────────────────────
    if save_plot:
        meta = build_fig_meta(plotvariables, chapter=chapter,
                              extra={"script": "plotter.py::plot_timeseries"})
        save_and_stub(fig, meta, plot_type="timeseries",
                      save_pgf=plotting.get("save_pgf", True))

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN B — was Swell/Wind/Total scatter; removed 2026-05-02 along with
# the underlying PSD-band columns. See plot_damping_scatter (in plotter.py)
# for a maintained scatter pattern using FFT amplitude columns.
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN C — Multi-probe: save panels separately, stub combines them
#
# Use this when you want probe2 and probe3 as separate PDFs but a single
# two-subfigure .tex stub (so you can rearrange in Texifier).
# ─────────────────────────────────────────────────────────────────────────────

def plot_timeseries_multiprobe(processed_dfs: dict,
                               df_sel: pd.DataFrame,
                               plotvariables: dict,
                               chapter: str = "05") -> None:
    """
    One subplot per probe, saved as SEPARATE panel PDFs.
    The .tex stub uses \\subfigure and references both files.

    This gives you maximum flexibility in Texifier:
    — keep as subfigure side-by-side (the stub default)
    — or pull one panel out into its own \\begin{figure} manually
    """
    from wavescripts.plot_utils import (
        build_fig_meta, build_filename,
        _save_figure, write_figure_stub,
    )

    plotting  = plotvariables.get("plotting", {})
    show_plot = plotting.get("show_plot", True)
    save_plot = plotting.get("save_plot", False)
    probes    = plotting.get("probes", [2, 3])

    panel_filenames = []

    for probe_num in probes:
        fig, ax = plt.subplots(figsize=(plotting.get("figsize", (10, 4))))

        for _, row in df_sel.iterrows():
            path_key = row["path"]
            if path_key not in processed_dfs:
                continue
            df_ma  = processed_dfs[path_key]
            color  = WIND_COLOR_MAP.get(row.get("WindCondition"), "black")
            lstyle = PANEL_STYLES.get(row.get("PanelCondition", ""), "solid")

            zeroed_col = f"eta_{probe_num}"
            if zeroed_col not in df_ma.columns:
                continue

            t0      = df_ma["Date"].iloc[0]
            time_ms = (df_ma["Date"] - t0).dt.total_seconds() * 1000
            ax.plot(time_ms, df_ma[zeroed_col],
                    label=make_label(row), color=color, linestyle=lstyle)

        ax.set_xlabel("Time [ms]")
        ax.set_ylabel(f"η probe {probe_num} [mm]")
        apply_legend(ax, plotvariables)
        fig.tight_layout()

        if save_plot:
            # Build per-probe meta and filename
            probe_meta = build_fig_meta(plotvariables, chapter=chapter)
            probe_meta["probes"] = probe_num          # single probe per panel
            fname = build_filename("timeseries", probe_meta)
            _save_figure(fig, fname,
                         save_pdf=True,
                         save_pgf=plotting.get("save_pgf", True))
            panel_filenames.append(fname)

        if show_plot:
            plt.show()
        else:
            plt.close(fig)

    # Write ONE stub that references both panel files
    if save_plot and panel_filenames:
        stub_meta = build_fig_meta(plotvariables, chapter=chapter,
                                   extra={"script": "plotter.py::plot_timeseries_multiprobe"})
        stub_meta["probes"] = probes   # full list for stub filename
        write_figure_stub(stub_meta, "timeseries",
                          panel_filenames=panel_filenames)
