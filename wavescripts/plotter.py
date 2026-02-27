#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plotter.py
==========
Thesis-bound plot functions for the wavescripts project.

Every function here:
  - accepts show_plot and save_plot toggles in plotvariables["plotting"]
  - calls save_and_stub() when save_plot=True
  - produces output to output/FIGURES/ and output/TEXFIGU/

Sections
--------
  PROBE AMPLITUDE PROFILE     plot_all_probes
  DAMPING                     plot_damping_freq, plot_damping_scatter
  SWELL / P2 vs P3            plot_swell_scatter
  FREQUENCY SPECTRUM          plot_frequency_spectrum
  RECONSTRUCTED SIGNAL        plot_reconstructed, plot_reconstructed_rms
  RAMP DETECTION              gather_ramp_data, plot_ramp_detection
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

from wavescripts.signal_processing import get_positive_spectrum
from wavescripts.filters import filter_for_amplitude_plot
from wavescripts.constants import (
    SIGNAL, MEASUREMENT,
    ProbeColumns as PC,
    GlobalColumns as GC,
    CalculationResultColumns as RC,
)
from wavescripts.plot_utils import (
    WIND_COLOR_MAP, PANEL_STYLES, PANEL_MARKERS, MARKER_STYLES,
    LEGEND_CONFIGS, apply_legend, draw_anchored_text, make_label,
    build_fig_meta, build_filename,
    _save_figure, write_figure_stub, save_and_stub,
    _top_k_indices,
)


# ═══════════════════════════════════════════════════════════════════════════════
# PROBE AMPLITUDE PROFILE
# ═══════════════════════════════════════════════════════════════════════════════

def plot_all_probes(meta_df: pd.DataFrame,
                    plotvariables: dict,
                    chapter: str = "05") -> None:
    """
    Amplitude at each probe position (P1→P4) as a line plot.
    One line per experimental run, coloured by wind, styled by panel.
    """
    plotting  = plotvariables.get("plotting", {})
    show_plot = plotting.get("show_plot", True)
    save_plot = plotting.get("save_plot", False)

    probe_x      = [1, 1.1, 1.2, 1.25]
    probe_labels = ["P1", "P2", "P3", "P4"]

    fig, ax = plt.subplots(figsize=plotting.get("figsize") or (10, 6))

    for _, row in meta_df.iterrows():
        color   = WIND_COLOR_MAP.get(row.get("WindCondition"), "black")
        lstyle  = PANEL_STYLES.get(row.get("PanelCondition", ""), "solid")
        label   = make_label(row)

        y = [row.get(f"Probe {i} Amplitude", np.nan) for i in range(1, 5)]
        ax.plot(probe_x, y, linewidth=2, label=label,
                linestyle=lstyle, marker="o", color=color)

        for x, yi in zip(probe_x, y):
            if np.isfinite(yi):
                ax.annotate(f"{yi:.2f}", xy=(x, yi),
                            xytext=(6, 6), textcoords="offset points",
                            fontsize=8, color=color)

    ax.set_xlabel("Probe position (spacing not to scale)")
    ax.set_ylabel("Amplitude [mm]")
    ax.set_xticks(probe_x)
    ax.set_xticklabels(probe_labels)
    ax.grid(True)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, color="gray")
    ax.minorticks_on()
    apply_legend(ax, plotvariables)
    fig.tight_layout()

    if save_plot:
        meta = build_fig_meta(plotvariables, chapter=chapter,
                              extra={"script": "plotter.py::plot_all_probes"})
        save_and_stub(fig, meta, plot_type="amplitude_probes")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# DAMPING
# ═══════════════════════════════════════════════════════════════════════════════

def _draw_damping_freq_ax(ax: plt.Axes,
                           stats_df: pd.DataFrame,
                           panel: str,
                           wind: str) -> None:
    """
    Draw damping ratio P3/P2 vs frequency onto a single axes.
    Shared primitive used by both show_plot grid and save_plot loop.
    """
    mask   = ((stats_df[GC.PANEL_CONDITION_GROUPED] == panel) &
              (stats_df[GC.WIND_CONDITION] == wind))
    subset = stats_df[mask]

    if subset.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        ax.set_title(f"{panel} / {wind}", fontsize=9)
        return

    ax.axhline(1.0, color="black", linestyle="--",
               linewidth=0.8, alpha=0.4, label="Unity (no damping)")

    for amp in sorted(subset[GC.WAVE_AMPLITUDE_INPUT].unique()):
        amp_data = (subset[subset[GC.WAVE_AMPLITUDE_INPUT] == amp]
                    .sort_values(GC.WAVE_FREQUENCY_INPUT))
        ax.errorbar(
            amp_data[GC.WAVE_FREQUENCY_INPUT],
            amp_data["mean_P3P2"],
            yerr=amp_data["std_P3P2"],
            marker="o", label=f"{amp:.2f} V",
            capsize=3, alpha=0.8, linewidth=1.4,
        )

    ax.set_xlabel("Frequency [Hz]", fontsize=9)
    ax.set_ylabel("P3/P2", fontsize=9)
    ax.set_title(f"{panel}panel / {wind}wind", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Amplitude", fontsize=7, title_fontsize=7)


def plot_damping_freq(stats_df: pd.DataFrame,
                      plotvariables: dict,
                      chapter: str = "05") -> None:
    """
    Damping ratio P3/P2 vs frequency.

    show_plot → full panel×wind grid in one figure
    save_plot → one PDF/PGF per (panel, wind) cell + one .tex stub

    The same _draw_damping_freq_ax() draws both, so what you see
    in exploration is exactly what gets saved.

    Input: output from damping_all_amplitude_grouper()
    """
    plotting  = plotvariables.get("plotting", {})
    show_plot = plotting.get("show_plot", True)
    save_plot = plotting.get("save_plot", False)

    panel_conditions = sorted(stats_df[GC.PANEL_CONDITION_GROUPED].unique())
    wind_conditions  = sorted(stats_df[GC.WIND_CONDITION].unique())
    n_rows, n_cols   = len(panel_conditions), len(wind_conditions)

    if show_plot:
        figsize = plotting.get("figsize") or (4.5 * n_cols, 3.5 * n_rows)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize,
                                 squeeze=False, sharex=True, sharey=True)
        for i, panel in enumerate(panel_conditions):
            for j, wind in enumerate(wind_conditions):
                _draw_damping_freq_ax(axes[i, j], stats_df, panel, wind)
        fig.suptitle("Damping Ratio P3/P2 vs Frequency", fontsize=13, y=1.0)
        plt.tight_layout()
        plt.show()

    if save_plot:
        panel_filenames = []
        meta_base = build_fig_meta(
            plotvariables, chapter=chapter,
            extra={"script": "plotter.py::plot_damping_freq"})

        for panel in panel_conditions:
            for wind in wind_conditions:
                fig_s, ax_s = plt.subplots(figsize=(5.0, 3.8))
                _draw_damping_freq_ax(ax_s, stats_df, panel, wind)
                fig_s.tight_layout()

                panel_meta = {**meta_base, "panel": panel, "wind": wind}
                fname = build_filename("damping_freq", panel_meta)
                _save_figure(fig_s, fname,
                             save_pgf=plotting.get("save_pgf", True))
                panel_filenames.append(fname)
                plt.close(fig_s)

        stub_meta = {**meta_base,
                     "panel": panel_conditions,
                     "wind":  wind_conditions}
        write_figure_stub(stub_meta, "damping_freq",
                          panel_filenames=panel_filenames)


def plot_damping_scatter(stats_df: pd.DataFrame,
                         plotvariables: Optional[dict] = None,
                         show_errorbars: bool = True,
                         size_by_amplitude: bool = False,
                         chapter: str = "05") -> None:
    """
    Single scatter: P3/P2 ratio for all conditions, coloured by wind.
    Input: output from damping_all_amplitude_grouper()
    """
    if plotvariables is None:
        plotvariables = {"plotting": {"show_plot": True, "save_plot": False}}

    plotting  = plotvariables.get("plotting", {})
    show_plot = plotting.get("show_plot", True)
    save_plot = plotting.get("save_plot", False)

    sns.set_style("ticks", {"axes.grid": True})
    fig, ax = plt.subplots(figsize=plotting.get("figsize") or (10, 6))
    plot_data = stats_df.sort_values(GC.WAVE_FREQUENCY_INPUT)

    scatter_kwargs = dict(
        data=plot_data, x=GC.WAVE_FREQUENCY_INPUT, y="mean_P3P2",
        hue=GC.WIND_CONDITION, palette=WIND_COLOR_MAP,
        style=GC.PANEL_CONDITION_GROUPED, style_order=["no", "all"],
        alpha=0.75, ax=ax, legend="auto",
    )
    if size_by_amplitude:
        scatter_kwargs["size"]  = GC.WAVE_AMPLITUDE_INPUT
        scatter_kwargs["sizes"] = (50, 200)

    sns.scatterplot(**scatter_kwargs)

    if show_errorbars and "std_P3P2" in plot_data.columns:
        for wind in plot_data[GC.WIND_CONDITION].unique():
            wd = plot_data[plot_data[GC.WIND_CONDITION] == wind]
            ax.errorbar(wd[GC.WAVE_FREQUENCY_INPUT], wd["mean_P3P2"],
                        yerr=wd["std_P3P2"], fmt="none",
                        ecolor=WIND_COLOR_MAP.get(wind, "gray"),
                        elinewidth=1, capsize=3, alpha=0.4, zorder=1)

    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.4)
    ax.set_xlabel("Frequency [Hz]", fontsize=11)
    ax.set_ylabel("P3/P2  (mean ± std)", fontsize=11)
    ax.set_title("Damping Ratio: All Conditions", fontsize=12, fontweight="bold")
    ax.legend(loc="best", framealpha=0.9, fontsize=9)
    plt.tight_layout()

    if save_plot:
        meta = build_fig_meta(plotvariables, chapter=chapter,
                              extra={"script": "plotter.py::plot_damping_scatter"})
        save_and_stub(fig, meta, plot_type="damping_scatter",
                      save_pgf=plotting.get("save_pgf", True))

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# SWELL / P2 vs P3
# ═══════════════════════════════════════════════════════════════════════════════

_BAND_COLS = {
    "Swell": PC.SWELL_AMPLITUDE_PSD,
    "Wind":  PC.WIND_AMPLITUDE_PSD,
    "Total": PC.TOTAL_AMPLITUDE_PSD,
}


def _draw_swell_scatter_ax(ax, band_amplitudes, p2_col, p3_col,
                            band_name, shared_lim=None,
                            annotate_delta=True, show_legend=True):
    has_wind  = GC.WIND_CONDITION  in band_amplitudes.columns
    has_panel = GC.PANEL_CONDITION in band_amplitudes.columns

    if p2_col not in band_amplitudes.columns or \
       p3_col not in band_amplitudes.columns:
        ax.text(0.5, 0.5, "Missing\ncolumns", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="gray")
        ax.set_title(f"{band_name} band", fontweight="bold")
        return

    p2 = band_amplitudes[p2_col].to_numpy()
    p3 = band_amplitudes[p3_col].to_numpy()
    valid = np.isfinite(p2) & np.isfinite(p3)

    if valid.sum() == 0:
        ax.text(0.5, 0.5, "No valid\ndata", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="gray")
        ax.set_title(f"{band_name} band", fontweight="bold")
        return

    lo, hi = shared_lim if shared_lim else (
        min(p2[valid].min(), p3[valid].min()),
        max(p2[valid].max(), p3[valid].max()),
    )
    if hi <= lo:
        lo, hi = 0.0, max(1.0, hi)
    pad = 0.05 * (hi - lo)
    lo, hi = lo - pad, hi + pad

    winds  = band_amplitudes[GC.WIND_CONDITION].unique()  if has_wind  else [None]
    panels = band_amplitudes[GC.PANEL_CONDITION].unique() if has_panel else [None]

    for wind in winds:
        for panel in panels:
            mask = np.ones(len(band_amplitudes), dtype=bool)
            if wind  is not None: mask &= (band_amplitudes[GC.WIND_CONDITION]  == wind)
            if panel is not None: mask &= (band_amplitudes[GC.PANEL_CONDITION] == panel)
            if mask.sum() == 0:
                continue
            ax.scatter(
                p2[mask], p3[mask], s=60, alpha=0.75,
                color=WIND_COLOR_MAP.get(wind, "steelblue") if wind else "steelblue",
                marker=PANEL_MARKERS.get(panel, "o") if panel else "o",
                edgecolors="white", linewidths=0.5,
                label=f"{wind}/{panel}" if (wind and panel) else (wind or panel or ""),
                rasterized=True,
            )

    ax.plot([lo, hi], [lo, hi], "k--", lw=0.9, alpha=0.45, zorder=1)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, lw=0.5)
    ax.set_title(f"{band_name} band", fontweight="bold", fontsize=9)

    if annotate_delta:
        delta = (p3[valid] - p2[valid]).mean()
        ax.text(0.04, 0.96, f"Δ mean = {delta:+.3f} mm",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=7.5, color="#333",
                bbox=dict(boxstyle="round,pad=0.2",
                          facecolor="white", alpha=0.6, edgecolor="none"))

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=6.5, loc="lower right",
                      framealpha=0.85, markerscale=0.85)


def _swell_shared_lim(band_amplitudes):
    all_vals = []
    for template in _BAND_COLS.values():
        for probe in (2, 3):
            col = template.format(i=probe)
            if col in band_amplitudes.columns:
                v = band_amplitudes[col].to_numpy()
                all_vals.append(v[np.isfinite(v)])
    if not all_vals:
        return 0.0, 1.0
    combined = np.concatenate(all_vals)
    lo, hi = combined.min(), combined.max()
    pad = 0.05 * (hi - lo) if hi > lo else 0.1
    return lo - pad, hi + pad


def plot_swell_scatter(meta_df: pd.DataFrame,
                       plotvariables: dict,
                       chapter: str = "05",
                       share_axes: bool = True,
                       annotate_delta: bool = True) -> None:
    """
    P2 vs P3 amplitude scatter for Swell, Wind, and Total bands.

    show_plot → all three bands side-by-side + data summary panel
    save_plot → one PDF/PGF per band + one .tex stub with three subfigures

    Replaces:
        plot_p2_vs_p3_scatter         (all bands, wind+panel encoding)
        plot_swell_p2_vs_p3_by_wind   (Δ mean, shared limits — merged in)

    Input: combined_meta_sel (filtering applied internally)
    """
    plotting  = plotvariables.get("plotting", {})
    show_plot = plotting.get("show_plot", True)
    save_plot = plotting.get("save_plot", False)

    print("\n" + "=" * 50)
    print("plot_swell_scatter — filtering")
    band_amplitudes = filter_for_amplitude_plot(meta_df, plotvariables)
    if band_amplitudes.empty:
        print("No data after filtering — aborting.")
        return
    print(f"  {len(band_amplitudes)} rows remaining")

    shared_lim = _swell_shared_lim(band_amplitudes) if share_axes else None

    if show_plot:
        figsize = plotting.get("figsize") or (14, 5)
        fig = plt.figure(figsize=figsize)
        n_bands = len(_BAND_COLS)
        gs   = fig.add_gridspec(1, n_bands + 1, width_ratios=[1] * n_bands + [0.38])
        axes = [fig.add_subplot(gs[0, i]) for i in range(n_bands)]
        info_ax = fig.add_subplot(gs[0, -1])
        info_ax.axis("off")

        # Summary text
        has_wind  = GC.WIND_CONDITION  in band_amplitudes.columns
        has_panel = GC.PANEL_CONDITION in band_amplitudes.columns
        lines = ["DATA SUMMARY", "─" * 20, f"n = {len(band_amplitudes)}", ""]
        if has_wind:
            for w in sorted(band_amplitudes[GC.WIND_CONDITION].unique()):
                c = (band_amplitudes[GC.WIND_CONDITION] == w).sum()
                lines.append(f"W:{w}  n={c}")
        if has_panel:
            lines.append("")
            for p in sorted(band_amplitudes[GC.PANEL_CONDITION].unique()):
                c = (band_amplitudes[GC.PANEL_CONDITION] == p).sum()
                lines.append(f"P:{p}  n={c}")
        info_ax.text(0.05, 0.97, "\n".join(lines),
                     transform=info_ax.transAxes, fontsize=7,
                     verticalalignment="top", fontfamily="monospace",
                     bbox=dict(boxstyle="round", facecolor="wheat",
                               alpha=0.3, edgecolor="none"))

        for ax, (band_name, template) in zip(axes, _BAND_COLS.items()):
            _draw_swell_scatter_ax(
                ax, band_amplitudes,
                p2_col=template.format(i=2),
                p3_col=template.format(i=3),
                band_name=band_name,
                shared_lim=shared_lim,
                annotate_delta=annotate_delta,
            )
            ax.set_xlabel("P2 amplitude [mm]", fontsize=9)
            ax.set_ylabel("P3 amplitude [mm]", fontsize=9)

        fig.suptitle("P2 vs P3 — Swell / Wind / Total",
                     fontsize=12, fontweight="bold", y=1.01)
        plt.tight_layout()
        plt.show()

    if save_plot:
        panel_filenames = []
        meta_base = build_fig_meta(
            plotvariables, chapter=chapter,
            extra={"script": "plotter.py::plot_swell_scatter"})

        for band_name, template in _BAND_COLS.items():
            fig_s, ax_s = plt.subplots(figsize=(4.5, 4.2))
            _draw_swell_scatter_ax(
                ax_s, band_amplitudes,
                p2_col=template.format(i=2),
                p3_col=template.format(i=3),
                band_name=band_name,
                shared_lim=shared_lim,
                annotate_delta=annotate_delta,
            )
            ax_s.set_xlabel("P2 amplitude [mm]", fontsize=9)
            ax_s.set_ylabel("P3 amplitude [mm]", fontsize=9)
            fig_s.tight_layout()

            band_meta = {**meta_base, "band": band_name.lower()}
            fname = build_filename(f"swell_{band_name.lower()}", band_meta)
            _save_figure(fig_s, fname, save_pgf=plotting.get("save_pgf", True))
            panel_filenames.append(fname)
            plt.close(fig_s)

        write_figure_stub(meta_base, "swell_scatter",
                          panel_filenames=panel_filenames)


# ═══════════════════════════════════════════════════════════════════════════════
# FREQUENCY SPECTRUM
# ═══════════════════════════════════════════════════════════════════════════════

def plot_frequency_spectrum(fft_dict: dict,
                             meta_df: pd.DataFrame,
                             plotvariables: dict,
                             data_type: str = "fft",
                             chapter: str = "05") -> tuple:
    """
    Frequency spectrum (FFT or PSD), faceted by probe / wind / panel.

    Returns (fig, axes).
    """
    plotting = plotvariables.get("plotting", {})
    show_plot = plotting.get("show_plot", True)
    save_plot = plotting.get("save_plot", False)

    facet_by       = plotting.get("facet_by", None)
    probes         = plotting.get("probes", [1])
    if not isinstance(probes, (list, tuple)):
        probes = [probes]
    n_peaks        = plotting.get("peaks", None)
    log_scale      = plotting.get("logaritmic", False)
    max_points     = plotting.get("max_points", 120)
    show_grid      = plotting.get("grid", True)
    xlim           = plotting.get("xlim", (0, 10))
    linewidth      = plotting.get("linewidth", 1.0)
    fontsize       = 7

    col_prefix = "Pxx" if data_type.lower() == "psd" else "FFT"
    ylabel     = col_prefix

    base_freq_val = plotvariables.get("filters", {}).get("WaveFrequencyInput [Hz]")
    base_freq = None
    if isinstance(base_freq_val, (list, tuple, np.ndarray, pd.Series)):
        base_freq = float(base_freq_val[0]) if len(base_freq_val) > 0 else None
    elif base_freq_val is not None:
        base_freq = float(base_freq_val)
    use_locators = base_freq is not None and base_freq > 0

    if facet_by == "probe":
        facet_groups = list(probes)
        facet_labels = [f"Probe {p}" for p in facet_groups]
    elif facet_by == "wind":
        facet_groups = list(pd.unique(meta_df["WindCondition"]))
        facet_labels = [f"Wind: {w}" for w in facet_groups]
    elif facet_by == "panel":
        facet_groups = list(pd.unique(meta_df["PanelCondition"]))
        facet_labels = [f"Panel: {p}" for p in facet_groups]
    else:
        facet_groups = [None]
        facet_labels = ["All Data"]

    n_facets = len(facet_groups)
    default_figsize = (12, 4 * n_facets) if n_facets > 1 else (18, 10)
    figsize = plotting.get("figsize") or default_figsize

    fig, axes = plt.subplots(n_facets, figsize=figsize,
                              sharex=True, squeeze=False, dpi=120)
    axes = axes.flatten()

    for facet_idx, (group, facet_label) in enumerate(zip(facet_groups, facet_labels)):
        ax = axes[facet_idx]

        subset = (meta_df[meta_df["WindCondition"] == group] if facet_by == "wind"
                  else meta_df[meta_df["PanelCondition"] == group] if facet_by == "panel"
                  else meta_df)

        for _, row in subset.iterrows():
            path = row["path"]
            if path not in fft_dict:
                continue

            df_fft    = fft_dict[path]
            windcond  = row.get("WindCondition", "unknown")
            color     = WIND_COLOR_MAP.get(windcond, "black")
            panelcond = row.get("PanelCondition", "unknown")
            lstyle    = PANEL_STYLES.get(panelcond, "solid")
            pk_marker = MARKER_STYLES.get(windcond, ".")
            label_base = make_label(row)

            probes_to_plot = [group] if facet_by == "probe" else probes

            for probe_num in probes_to_plot:
                col = f"{col_prefix} {probe_num}"
                if col not in df_fft:
                    continue
                df_pos = get_positive_spectrum(df_fft)
                y = df_pos[col].dropna().iloc[:max_points]
                if y.empty:
                    continue
                x = y.index.values

                plot_label = (label_base if facet_by == "probe"
                              else f"{label_base}_P{probe_num}" if len(probes_to_plot) > 1
                              else label_base)

                ax.plot(x, y.values, linewidth=linewidth, label=plot_label,
                        linestyle=lstyle, color=color, antialiased=False)

                if n_peaks and n_peaks > 0:
                    from wavescripts.plot_utils import _top_k_indices
                    top_idx = _top_k_indices(y.values, n_peaks)
                    ax.scatter(x[top_idx], y.values[top_idx],
                               color=color, s=80, zorder=5,
                               marker=pk_marker, linewidths=0.7)

        ax.set_title(facet_label, fontsize=fontsize, fontweight="normal")
        ax.set_ylabel(ylabel, fontsize=fontsize)
        if log_scale:
            ax.set_yscale("log")
        if xlim:
            ax.set_xlim(xlim)
        if use_locators:
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(base_freq))
            ax.xaxis.set_major_locator(ticker.MultipleLocator(2 * base_freq))
        else:
            ax.xaxis.set_major_locator(ticker.MaxNLocator(10))
        ax.tick_params(axis="both", labelsize=8)
        if show_grid:
            ax.grid(which="major", linestyle="--", alpha=0.6)
            ax.grid(which="minor", linestyle="-.", alpha=0.3)
        apply_legend(ax, plotvariables)

    axes[-1].set_xlabel(col_prefix, fontsize=fontsize)
    plt.tight_layout()

    if save_plot:
        meta = build_fig_meta(plotvariables, chapter=chapter,
                              extra={"script": "plotter.py::plot_frequency_spectrum",
                                     "data_type": data_type})
        save_and_stub(fig, meta,
                      plot_type=f"spectrum_{data_type}",
                      save_pgf=plotting.get("save_pgf", True))

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return fig, axes


# ═══════════════════════════════════════════════════════════════════════════════
# RECONSTRUCTED SIGNAL
# ═══════════════════════════════════════════════════════════════════════════════

def plot_reconstructed(fft_dict: Dict[str, pd.DataFrame],
                        filtrert_frequencies: pd.DataFrame,
                        freqplotvariables: dict,
                        data_type: str = "fft",
                        chapter: str = "05") -> Tuple[Optional[plt.Figure], Optional[np.ndarray]]:
    """
    Reconstructed wave vs wind component for a SINGLE experiment.
    Facet by probe if plotting["facet_by"] == "probe".
    """
    meta_df  = filtrert_frequencies.copy()
    plotting = freqplotvariables.get("plotting", {})
    show_plot = plotting.get("show_plot", True)
    save_plot = plotting.get("save_plot", False)

    probes    = plotting.get("probes", [1])
    probes    = [probes] if not isinstance(probes, (list, tuple)) else probes
    facet_by  = plotting.get("facet_by", None)
    show_grid = plotting.get("grid", True)
    linewidth = plotting.get("linewidth", 1.2)
    dual_yaxis = plotting.get("dual_yaxis", True)
    show_full  = plotting.get("show_full_signal", False)
    show_stats = plotting.get("show_amplitude_stats", True)
    fontsize   = 9

    if len(fft_dict) == 0:
        print("ERROR: fft_dict is empty")
        return None, None
    if len(fft_dict) > 1:
        print(f"WARNING: plotting only first of {len(fft_dict)} experiments")

    path   = list(fft_dict.keys())[0]
    df_fft = fft_dict[path]
    row    = meta_df[meta_df["path"] == path]
    if len(row) == 0:
        print(f"ERROR: no metadata for {path}")
        return None, None
    row = row.iloc[0]

    windcond   = row.get("WindCondition", "unknown")
    panelcond  = row.get("PanelCondition", "unknown")
    target_freq = row.get(GC.WAVE_FREQUENCY_INPUT, None)
    if not target_freq or target_freq <= 0:
        print(f"ERROR: invalid target frequency {target_freq}")
        return None, None

    color_swell = WIND_COLOR_MAP.get(windcond, "black")
    color_wind  = "darkred" if dual_yaxis else "orange"
    color_full  = "gray"
    lstyle      = {"no": "-", "full": "--", "reverse": "-."}.get(panelcond, "-")

    n_subplots = len(probes) if facet_by == "probe" else 1
    figsize    = plotting.get("figsize") or (16, 5 * n_subplots if n_subplots > 1 else 7)
    fig, axes  = plt.subplots(n_subplots, 1, figsize=figsize,
                               squeeze=False, dpi=120)
    axes = axes.flatten()
    amplitude_comparison = []

    for subplot_idx in range(n_subplots):
        ax_s = axes[subplot_idx]
        ax_w = ax_s.twinx() if dual_yaxis else ax_s
        probes_here = [probes[subplot_idx]] if facet_by == "probe" else probes
        title = (f"Probe {probes[subplot_idx]}" if facet_by == "probe"
                 else f"{windcond} wind / {panelcond} panel / {target_freq:.3f} Hz")

        for probe_num in probes_here:
            col = f"FFT {probe_num} complex"
            if col not in df_fft:
                col = f"FFT {probe_num}"
            if col not in df_fft:
                print(f"Skipping probe {probe_num}: column not found")
                continue

            fft_series  = df_fft[col].dropna()
            freq_bins   = fft_series.index.values
            fft_complex = fft_series.values
            N           = len(fft_complex)
            df_freq     = freq_bins[1] - freq_bins[0]
            sr          = abs(df_freq * N)
            fftfreqs    = np.fft.fftfreq(N, d=1/sr)

            fft_ord = np.zeros(N, dtype=complex)
            for i, tf in enumerate(fftfreqs):
                ci = np.argmin(np.abs(freq_bins - tf))
                if np.abs(freq_bins[ci] - tf) < 1e-6:
                    fft_ord[i] = fft_complex[ci]

            signal_full  = np.real(np.fft.ifft(fft_ord))
            time_axis    = np.arange(N) / sr
            pos_freqs    = fftfreqs[fftfreqs > 0]
            actual_freq  = pos_freqs[np.argmin(np.abs(pos_freqs - target_freq))]
            peak_idx     = np.where(np.abs(fftfreqs - actual_freq) < 1e-6)[0][0]
            mirror_idx   = np.where(np.abs(fftfreqs + actual_freq) < 1e-6)[0][0]

            fft_swell = np.zeros_like(fft_ord, dtype=complex)
            fft_swell[peak_idx]   = fft_ord[peak_idx]
            fft_swell[mirror_idx] = fft_ord[mirror_idx]
            signal_swell = np.real(np.fft.ifft(fft_swell))
            signal_wind  = signal_full - signal_swell

            lp = "" if facet_by == "probe" else f"P{probe_num} "
            if show_full:
                ax_s.plot(time_axis, signal_full, lw=linewidth*0.7,
                          label=f"{lp}full", color=color_full, alpha=0.4, zorder=1)
            ax_s.plot(time_axis, signal_swell, lw=linewidth*1.5,
                      label=f"{lp}wave ({actual_freq:.4f}Hz)",
                      linestyle=lstyle, color=color_swell, alpha=0.9, zorder=3)
            ax_w.plot(time_axis, signal_wind, lw=linewidth,
                      label=f"{lp}wind",
                      linestyle=":" if dual_yaxis else "--",
                      color=color_wind, alpha=0.7, zorder=2)

        ax_s.set_title(title, fontsize=fontsize+2, fontweight="bold", pad=15)
        ax_s.set_xlabel("Time [s]", fontsize=fontsize)
        if dual_yaxis:
            ax_s.set_ylabel("Swell Amplitude", fontsize=fontsize, color=color_swell)
            ax_w.set_ylabel("Wind+Noise Amplitude", fontsize=fontsize, color=color_wind)
        else:
            ax_s.set_ylabel("Amplitude", fontsize=fontsize)
        if show_grid:
            ax_s.grid(which="major", linestyle="--", alpha=0.3)
            ax_s.grid(which="minor", linestyle=":", alpha=0.15)
            ax_s.minorticks_on()
        lines_s, labs_s = ax_s.get_legend_handles_labels()
        lines_w, labs_w = (ax_w.get_legend_handles_labels() if dual_yaxis else ([], []))
        if lines_s + lines_w:
            ax_s.legend(lines_s + lines_w, labs_s + labs_w,
                        loc="upper right", fontsize=fontsize, framealpha=0.95)
        ax_s.axhline(0, color="black", lw=0.5, alpha=0.3)

    plt.suptitle(f"{Path(path).stem}\n{windcond} / {panelcond} / {target_freq:.2f} Hz",
                 fontsize=fontsize+3, fontweight="bold", y=0.995)
    plt.tight_layout()

    if save_plot:
        meta = build_fig_meta(freqplotvariables, chapter=chapter,
                              extra={"script": "plotter.py::plot_reconstructed"})
        save_and_stub(fig, meta, plot_type="reconstructed",
                      save_pgf=plotting.get("save_pgf", True))

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return fig, axes


def plot_reconstructed_rms(fft_dict: dict,
                            filtrert_frequencies: pd.DataFrame,
                            freqplotvariables: dict,
                            data_type: str = "fft",
                            chapter: str = "05"):
    """
    Reconstructed signal with RMS amplitude comparison.
    Loops over all experiments in fft_dict (unlike plot_reconstructed).
    Returns (fig, axes) of the last figure created.
    """
    fig, axes = None, None
    for path, df_fft in fft_dict.items():
        single_meta = filtrert_frequencies[filtrert_frequencies["path"] == path]
        if single_meta.empty:
            continue
        fig, axes = plot_reconstructed(
            {path: df_fft},
            single_meta,
            freqplotvariables,
            data_type=data_type,
            chapter=chapter,
        )
    return fig, axes


# ═══════════════════════════════════════════════════════════════════════════════
# RAMP DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def gather_ramp_data(dfs: dict,
                     combined_meta_sel: pd.DataFrame,
                     sigma_factor: float = SIGNAL.BASELINE_SIGMA_FACTOR) -> pd.DataFrame:
    """
    Pre-compute ramp detection data for all experiments and probes.
    Returns a flat DataFrame with one row per (experiment × probe).
    Used to feed RampDetectionBrowser (in plot_quicklook.py).
    """
    records = []
    for path, df in dfs.items():
        meta_row = combined_meta_sel[combined_meta_sel[GC.PATH] == path]
        if len(meta_row) == 0:
            continue
        meta_row = meta_row.iloc[0]

        for i in range(1, MEASUREMENT.NUM_PROBES + 1):
            col_raw  = f"Probe {i}"
            col_eta  = f"eta_{i}"
            col_ma   = f"Probe {i}_ma"
            col_start = PC.START.format(i=i)
            col_end   = PC.END.format(i=i)

            if col_raw not in df.columns or col_eta not in df.columns:
                continue

            good_start = meta_row.get(col_start)
            good_end   = meta_row.get(col_end)
            if good_start is None or pd.isna(good_start):
                continue
            if good_end is None or pd.isna(good_end):
                continue

            good_start = int(good_start)
            good_end   = int(good_end)
            signal     = df[col_eta].to_numpy()
            raw        = df[col_raw].to_numpy()
            ma         = df[col_ma].to_numpy() if col_ma in df.columns else signal

            n_base      = min(good_start,
                              int(SIGNAL.BASELINE_DURATION_SEC * MEASUREMENT.SAMPLING_RATE))
            base_region = signal[:n_base] if n_base > 10 else signal[:50]
            base_std    = np.std(base_region) if len(base_region) > 0 else 1.0
            threshold   = sigma_factor * base_std
            exceeded    = np.where(np.abs(signal) > threshold)[0]
            first_motion = int(exceeded[0]) if len(exceeded) > 0 else good_start

            t0      = df["Date"].iat[0]
            time_ms = (df["Date"] - t0).dt.total_seconds().to_numpy() * MEASUREMENT.M_TO_MM

            records.append({
                GC.PATH:                  path,
                "experiment":             Path(path).stem,
                "probe":                  i,
                "data_col":               col_raw,
                GC.WIND_CONDITION:        meta_row.get(GC.WIND_CONDITION, "unknown"),
                GC.PANEL_CONDITION:       meta_row.get(GC.PANEL_CONDITION, "unknown"),
                GC.WAVE_FREQUENCY_INPUT:  meta_row.get(GC.WAVE_FREQUENCY_INPUT, np.nan),
                GC.WAVE_AMPLITUDE_INPUT:  meta_row.get(GC.WAVE_AMPLITUDE_INPUT, np.nan),
                "time_ms":        time_ms,
                "raw":            raw,
                "signal":         signal,
                "ma":             ma,
                "baseline_mean":  meta_row.get(PC.STILLWATER.format(i=i), 0.0),
                "baseline_std":   base_std,
                "threshold":      threshold,
                "first_motion_idx": first_motion,
                "good_start_idx": good_start,
                "good_end_idx":   good_end,
                "good_range":     good_end - good_start,
            })

    df_out = pd.DataFrame(records)
    print(f"\nRamp data: {df_out['experiment'].nunique()} experiments, "
          f"{len(df_out)} rows")
    return df_out


def plot_ramp_detection(df: pd.DataFrame,
                         meta_sel,
                         data_col: str,
                         signal: np.ndarray,
                         baseline_mean: float,
                         threshold: float,
                         first_motion_idx: int,
                         good_start_idx: int,
                         good_range: int,
                         good_end_idx: Optional[int] = None,
                         peaks: Optional[np.ndarray] = None,
                         peak_amplitudes=None,
                         ramp_peak_indices: Optional[np.ndarray] = None,
                         title: str = "Ramp Detection") -> Tuple[plt.Figure, plt.Axes]:
    """Diagnostic plot for ramp detection on a single experiment/probe."""
    if "Date" not in df.columns:
        raise ValueError("df must contain a 'Date' column")
    if data_col not in df.columns:
        raise ValueError(f"df must contain '{data_col}' column")

    t0      = df["Date"].iat[0]
    time_ms = (df["Date"] - t0).dt.total_seconds().to_numpy() * MEASUREMENT.M_TO_MM
    raw     = df[data_col].to_numpy()
    n       = len(time_ms)

    gs_i = max(0, min(int(good_start_idx), n - 2))
    ge_i = (max(gs_i + 1, min(int(good_end_idx), n - 1))
            if good_end_idx is not None
            else max(gs_i + 1, min(gs_i + int(good_range), n - 1)))

    fig, ax = plt.subplots(figsize=(15, 7))
    fig.suptitle(title)

    ax.plot(time_ms, raw, color="lightgray", alpha=0.6, label="Raw")
    ax.plot(time_ms, signal, color="black", lw=2, label=f"Smoothed {data_col}")
    ax.axhline(baseline_mean, color="blue", linestyle="--",
               label=f"Baseline = {baseline_mean:.2f} mm")
    ax.axhline(baseline_mean + threshold, color="red", linestyle=":", alpha=0.7)
    ax.axhline(baseline_mean - threshold, color="red", linestyle=":", alpha=0.7)
    ax.axvline(time_ms[first_motion_idx], color="orange", lw=2, linestyle="--",
               label=f"First motion #{first_motion_idx}")
    ax.axvline(time_ms[gs_i], color="green", lw=3, label=f"Stable start #{gs_i}")
    ax.axvline(time_ms[ge_i], color="purple", lw=2, linestyle="--",
               label=f"End #{ge_i}")
    ax.axvspan(time_ms[gs_i], time_ms[ge_i], color="green", alpha=0.08)

    if peaks is not None and len(peaks) > 0:
        pk = np.asarray(peaks, dtype=int)
        pk = pk[(pk >= 0) & (pk < n)]
        ax.plot(time_ms[pk], signal[pk], "ro", ms=6, alpha=0.7, label="Peaks")
    if ramp_peak_indices is not None and len(ramp_peak_indices) > 0:
        rpi = np.asarray(ramp_peak_indices, dtype=int)
        rpi = rpi[(rpi >= 0) & (rpi < n)]
        ax.plot(time_ms[rpi], signal[rpi], "o", color="lime", ms=10,
                markeredgecolor="darkgreen", markeredgewidth=2,
                label=f"Ramp-up ({len(rpi)} peaks)")

    try:
        path_val = (meta_sel["path"] if isinstance(meta_sel, pd.Series)
                    else meta_sel["path"].iloc[0])
        ax.set_title(f"{Path(str(path_val)).stem}  →  {data_col}",
                     fontsize=14, pad=20)
    except Exception:
        pass

    amp_in = float(meta_sel.get(GC.WAVE_AMPLITUDE_INPUT, 0) or 0)
    zoom   = max(amp_in * 100, 15)
    ax.set_ylim(baseline_mean - zoom, baseline_mean + zoom)
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Water level [mm]")
    ax.grid(True, alpha=0.1)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    fig.tight_layout()
    return fig, ax
