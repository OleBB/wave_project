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
# seaborn imported lazily (heavy) — used only in plot_damping_scatter

from datetime import datetime

from wavescripts.constants import (
    MEASUREMENT,
    SIGNAL,
)
from wavescripts.improved_data_loader import get_configuration_for_date
from wavescripts.constants import (
    CalculationResultColumns as RC,
)
from wavescripts.constants import (
    GlobalColumns as GC,
)
from wavescripts.constants import (
    ProbeColumns as PC,
)
from wavescripts.filters import filter_for_amplitude_plot
from wavescripts.plot_utils import (
    LEGEND_CONFIGS,
    MARKER_STYLES,
    PANEL_MARKERS,
    PANEL_STYLES,
    WIND_COLOR_MAP,
    _save_figure,
    _top_k_indices,
    apply_legend,
    apply_thesis_style,
    build_fig_meta,
    build_filename,
    draw_anchored_text,
    make_label,
    resolve_caption,
    save_and_stub,
    write_figure_stub,
)
# get_positive_spectrum imported lazily (pulls in scipy.signal, ~2s)

# ═══════════════════════════════════════════════════════════════════════════════
# PROBE AMPLITUDE PROFILE
# ═══════════════════════════════════════════════════════════════════════════════


def plot_all_probes(
    meta_df: pd.DataFrame, plotvariables: dict, chapter: str = "05"
) -> None:
    """
    Amplitude at each probe position (P1→P4) as a line plot.
    One line per experimental run, coloured by wind, styled by panel.
    """
    plotting = plotvariables.get("plotting", {})
    show_plot = plotting.get("show_plot", False)
    save_plot = plotting.get("save_plot", False)

    # Collect all non-null amplitude columns across the combined meta
    all_amp_cols = [
        c for c in meta_df.columns
        if c.startswith("Probe ") and c.endswith(" Amplitude")
        and "FFT" not in c and "PSD" not in c
        and meta_df[c].notna().any()
    ]

    # Group columns by longitudinal distance (the part before "/", e.g. "9373/170" → "9373").
    # Runs from different probe configs share the same longitudinal positions even when
    # lateral offsets differ.  We average laterals when both are present for a given run.
    from collections import defaultdict as _dd
    long_groups: dict = _dd(list)
    for c in all_amp_cols:
        pos = c.replace("Probe ", "").replace(" Amplitude", "")
        long_mm = pos.split("/")[0]
        long_groups[long_mm].append(c)

    sorted_long = sorted(long_groups.keys(), key=int)
    probe_x     = list(range(len(sorted_long)))
    probe_labels = [f"P {lng}" for lng in sorted_long]

    fig, ax = plt.subplots(figsize=plotting.get("figsize") or (10, 6))

    for _, row in meta_df.iterrows():
        color  = WIND_COLOR_MAP.get(row.get("WindCondition"), "black")
        lstyle = PANEL_STYLES.get(row.get("PanelCondition", ""), "solid")
        label  = make_label(row)

        y = []
        for lng in sorted_long:
            vals = [row.get(c, np.nan) for c in long_groups[lng]]
            valid = [v for v in vals if np.isfinite(v)]
            y.append(float(np.mean(valid)) if valid else np.nan)

        ax.plot(
            probe_x,
            y,
            linewidth=2,
            label=label,
            linestyle=lstyle,
            marker="o",
            color=color,
        )

        for x, yi in zip(probe_x, y):
            if np.isfinite(yi):
                ax.annotate(
                    f"{yi:.2f}",
                    xy=(x, yi),
                    xytext=(6, 6),
                    textcoords="offset points",
                    fontsize=8,
                    color=color,
                )

    ax.set_xlabel("Probe position (spacing not to scale) parallels are averaged")
    ax.set_ylabel("Amplitude [mm]")
    ax.set_xticks(probe_x)
    ax.set_xticklabels(probe_labels)
    ax.grid(True)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, color="gray")
    ax.minorticks_on()
    apply_legend(ax, plotvariables)
    fig.tight_layout()

    if save_plot:
        meta = build_fig_meta(
            plotvariables,
            chapter=chapter,
            extra={"script": "plotter.py::plot_all_probes"},
        )
        save_and_stub(fig, meta, plot_type="amplitude_probes")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# DAMPING
# ═══════════════════════════════════════════════════════════════════════════════


def _draw_damping_freq_ax(
    ax: plt.Axes, stats_df: pd.DataFrame, panel: str, wind: str
) -> None:
    """
    Draw damping ratio OUT/IN vs frequency onto a single axes.
    Shared primitive used by both show_plot grid and save_plot loop.
    """
    mask = (stats_df[GC.PANEL_CONDITION_GROUPED] == panel) & (
        stats_df[GC.WIND_CONDITION] == wind
    )
    subset = stats_df[mask]

    if subset.empty:
        ax.text(
            0.5,
            0.5,
            "No data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="gray",
        )
        ax.set_title(f"{panel} / {wind}", fontsize=9)
        return

    ax.axhline(
        1.0,
        color="black",
        linestyle="--",
        linewidth=0.8,
        alpha=0.4,
        label="Unity (no damping)",
    )

    for amp in sorted(subset[GC.WAVE_AMPLITUDE_INPUT].unique()):
        amp_data = subset[subset[GC.WAVE_AMPLITUDE_INPUT] == amp].sort_values(
            GC.WAVE_FREQUENCY_INPUT
        )
        ax.errorbar(
            amp_data[GC.WAVE_FREQUENCY_INPUT],
            amp_data["mean_out_in"],
            yerr=amp_data["std_out_in"],
            marker="o",
            label=f"{amp:.2f} V",
            capsize=3,
            alpha=0.8,
            linewidth=1.4,
        )

    ax.set_xlabel("Frequency [Hz]", fontsize=9)
    ax.set_ylabel("OUT/IN", fontsize=9)
    ax.set_title(f"{panel}panel / {wind}wind", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Amplitude", fontsize=7, title_fontsize=7)


def _make_damping_freq_fig(
    stats_df: pd.DataFrame, panel: str, amp: float, figsize: tuple = (5, 4)
) -> plt.Figure:
    """
    Single axes: OUT/IN vs frequency for one (panel, amplitude) combination.
    Colour = wind condition. No internal faceting — LaTeX arranges subfigures.
    """
    subset = stats_df[
        (stats_df[GC.PANEL_CONDITION_GROUPED] == panel) &
        (stats_df[GC.WAVE_AMPLITUDE_INPUT] == amp)
    ]
    fig, ax = plt.subplots(figsize=figsize)
    for wind, grp in subset.groupby(GC.WIND_CONDITION):
        grp = grp.sort_values(GC.WAVE_FREQUENCY_INPUT)
        ax.errorbar(
            grp[GC.WAVE_FREQUENCY_INPUT], grp["mean_out_in"],
            yerr=grp["std_out_in"],
            label=wind, color=WIND_COLOR_MAP.get(wind),
            marker="o", markersize=5, linewidth=1.4, capsize=3,
        )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.4)
    ax.set_xlabel("Frequency [Hz]", fontsize=9)
    ax.set_ylabel("OUT/IN (FFT)", fontsize=9)
    ax.set_title(f"{panel} panel  |  {amp:.2f} V", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(title="wind", fontsize=8, title_fontsize=8)
    fig.tight_layout()
    return fig


def plot_damping_freq(
    stats_df: pd.DataFrame, plotvariables: dict, chapter: str = "05"
) -> None:
    """
    Damping ratio OUT/IN vs frequency.

    One figure per (panel_condition × amplitude) — no internal faceting.
    Faceting is done in LaTeX via the texfigu stub.
    Colour = wind condition.

    show_plot → one window per subfigure (REPL verification)
    save_plot → one PDF/PGF per subfigure + one .tex stub listing all

    Input: output from damping_all_amplitude_grouper()
    """
    plotting = plotvariables.get("plotting", {})
    show_plot = plotting.get("show_plot", False)
    save_plot = plotting.get("save_plot", False)
    figsize   = plotting.get("figsize", (5, 4))

    panel_conditions = sorted(stats_df[GC.PANEL_CONDITION_GROUPED].unique())
    amplitudes       = sorted(stats_df[GC.WAVE_AMPLITUDE_INPUT].unique())

    if show_plot:
        for panel in panel_conditions:
            for amp in amplitudes:
                fig = _make_damping_freq_fig(stats_df, panel, amp, figsize=figsize)
                plt.show()

    if save_plot:
        subfig_filenames = []
        meta_base = build_fig_meta(
            plotvariables,
            chapter=chapter,
            extra={"script": "plotter.py::plot_damping_freq"},
        )
        figure_name = plotting.get("figure_name") or build_filename("damping_freq", meta_base)
        i = 1
        for panel in panel_conditions:
            for amp in amplitudes:
                fig_s = _make_damping_freq_fig(stats_df, panel, amp, figsize=figsize)
                fname = f"{figure_name}_{i}"
                _save_figure(fig_s, fname, save_pgf=True)
                subfig_filenames.append(fname)
                plt.close(fig_s)
                i += 1

        stub_meta = {**meta_base, "panel": panel_conditions, "amplitude": amplitudes, "wind": "allwind"}
        write_figure_stub(stub_meta, "damping_freq", subfig_filenames=subfig_filenames,
                          force=plotting.get("force_stub", False))


def plot_damping_scatter(
    stats_df: pd.DataFrame,
    plotvariables: Optional[dict] = None,
    show_errorbars: bool = True,
    size_by_amplitude: bool = False,
    chapter: str = "05",
) -> None:
    """
    Single scatter: OUT/IN ratio for all conditions, coloured by wind.
    Input: output from damping_all_amplitude_grouper()
    """
    if plotvariables is None:
        plotvariables = {"plotting": {"show_plot": True, "save_plot": False}}

    plotting = plotvariables.get("plotting", {})
    show_plot = plotting.get("show_plot", False)
    save_plot = plotting.get("save_plot", False)

    import seaborn as sns
    sns.set_style("ticks", {"axes.grid": True})
    fig, ax = plt.subplots(figsize=plotting.get("figsize") or (10, 6))
    plot_data = stats_df.sort_values(GC.WAVE_FREQUENCY_INPUT)

    scatter_kwargs = dict(
        data=plot_data,
        x=GC.WAVE_FREQUENCY_INPUT,
        y="mean_out_in",
        hue=GC.WIND_CONDITION,
        palette=WIND_COLOR_MAP,
        style=GC.PANEL_CONDITION_GROUPED,
        style_order=["no", "all"],
        alpha=0.75,
        ax=ax,
        legend="auto",
    )
    if size_by_amplitude:
        scatter_kwargs["size"] = GC.WAVE_AMPLITUDE_INPUT
        scatter_kwargs["sizes"] = (50, 200)

    sns.scatterplot(**scatter_kwargs)

    if show_errorbars and "std_out_in" in plot_data.columns:
        for wind in plot_data[GC.WIND_CONDITION].unique():
            wd = plot_data[plot_data[GC.WIND_CONDITION] == wind]
            ax.errorbar(
                wd[GC.WAVE_FREQUENCY_INPUT],
                wd["mean_out_in"],
                yerr=wd["std_out_in"],
                fmt="none",
                ecolor=WIND_COLOR_MAP.get(wind, "gray"),
                elinewidth=1,
                capsize=3,
                alpha=0.4,
                zorder=1,
            )

    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.4)
    ax.set_xlabel("Frequency [Hz]", fontsize=11)
    ax.set_ylabel("OUT/IN  (mean ± std)", fontsize=11)
    ax.set_title("Damping Ratio: All Conditions", fontsize=12, fontweight="bold")
    ax.legend(loc="best", framealpha=0.9, fontsize=9)
    plt.tight_layout()

    if save_plot:
        meta = build_fig_meta(
            plotvariables,
            chapter=chapter,
            extra={"script": "plotter.py::plot_damping_scatter"},
        )
        save_and_stub(
            fig,
            meta,
            plot_type="damping_scatter",
            save_pgf=True,
        )

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# SWELL / IN vs OUT
# ═══════════════════════════════════════════════════════════════════════════════

_BAND_COLS = {
    "Swell": PC.SWELL_AMPLITUDE_PSD,
    "Wind": PC.WIND_AMPLITUDE_PSD,
    "Total": PC.TOTAL_AMPLITUDE_PSD,
}


def _draw_swell_scatter_ax(
    ax,
    band_amplitudes,
    in_col,
    out_col,
    band_name,
    shared_lim=None,
    annotate_delta=True,
    show_legend=True,
):
    has_wind = GC.WIND_CONDITION in band_amplitudes.columns
    has_panel = GC.PANEL_CONDITION in band_amplitudes.columns

    if in_col not in band_amplitudes.columns or out_col not in band_amplitudes.columns:
        ax.text(
            0.5,
            0.5,
            "Missing\ncolumns",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=9,
            color="gray",
        )
        ax.set_title(f"{band_name} band", fontweight="bold")
        return

    p2 = band_amplitudes[in_col].to_numpy()
    p3 = band_amplitudes[out_col].to_numpy()
    valid = np.isfinite(p2) & np.isfinite(p3)

    if valid.sum() == 0:
        ax.text(
            0.5,
            0.5,
            "No valid\ndata",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=9,
            color="gray",
        )
        ax.set_title(f"{band_name} band", fontweight="bold")
        return

    lo, hi = (
        shared_lim
        if shared_lim
        else (
            min(p2[valid].min(), p3[valid].min()),
            max(p2[valid].max(), p3[valid].max()),
        )
    )
    if hi <= lo:
        lo, hi = 0.0, max(1.0, hi)
    pad = 0.05 * (hi - lo)
    lo, hi = lo - pad, hi + pad

    winds = band_amplitudes[GC.WIND_CONDITION].unique() if has_wind else [None]
    panels = band_amplitudes[GC.PANEL_CONDITION].unique() if has_panel else [None]

    for wind in winds:
        for panel in panels:
            mask = np.ones(len(band_amplitudes), dtype=bool)
            if wind is not None:
                mask &= band_amplitudes[GC.WIND_CONDITION] == wind
            if panel is not None:
                mask &= band_amplitudes[GC.PANEL_CONDITION] == panel
            if mask.sum() == 0:
                continue
            ax.scatter(
                p2[mask],
                p3[mask],
                s=60,
                alpha=0.75,
                color=WIND_COLOR_MAP.get(wind, "steelblue") if wind else "steelblue",
                marker=PANEL_MARKERS.get(panel, "o") if panel else "o",
                edgecolors="white",
                linewidths=0.5,
                label=f"{wind}/{panel}" if (wind and panel) else (wind or panel or ""),
                rasterized=True,
            )

    ax.plot([lo, hi], [lo, hi], "k--", lw=0.9, alpha=0.45, zorder=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, lw=0.5)
    ax.set_title(f"{band_name} band", fontweight="bold", fontsize=9)

    if annotate_delta:
        delta = (p3[valid] - p2[valid]).mean()
        ax.text(
            0.04,
            0.96,
            f"Δ mean = {delta:+.3f} mm",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.5,
            color="#333",
            bbox=dict(
                boxstyle="round,pad=0.2", facecolor="white", alpha=0.6, edgecolor="none"
            ),
        )

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                fontsize=6.5, loc="lower right", framealpha=0.85, markerscale=0.85
            )


def _in_out_probes_from_df(df: pd.DataFrame) -> tuple[int, int]:
    """Read in_probe/out_probe numbers directly from metadata columns."""
    if "in_probe" in df.columns and "out_probe" in df.columns:
        in_vals = df["in_probe"].dropna().unique()
        out_vals = df["out_probe"].dropna().unique()
        if len(in_vals) > 1 or len(out_vals) > 1:
            print(f"WARNING: multiple probe configs in dataset — using most common")
        in_p = int(df["in_probe"].dropna().mode()[0]) if len(in_vals) > 0 else 2
        out_p = int(df["out_probe"].dropna().mode()[0]) if len(out_vals) > 0 else 3
        return in_p, out_p
    return 2, 3


def _in_out_positions_from_df(df: pd.DataFrame) -> tuple[str, str]:
    """Return position strings (e.g. '9373/170', '12545') for in/out probes."""
    from wavescripts.improved_data_loader import get_configuration_for_date
    from datetime import datetime
    in_p, out_p = _in_out_probes_from_df(df)
    try:
        file_date = datetime.fromisoformat(str(df["file_date"].dropna().iloc[0]))
        cfg = get_configuration_for_date(file_date)
        col_names = cfg.probe_col_names()
        return col_names[in_p], col_names[out_p]
    except Exception:
        return str(in_p), str(out_p)


def _swell_shared_lim(band_amplitudes, in_pos: str = "2", out_pos: str = "3"):
    all_vals = []
    for template in _BAND_COLS.values():
        for pos in (in_pos, out_pos):
            col = template.format(i=pos)
            if col in band_amplitudes.columns:
                v = band_amplitudes[col].to_numpy()
                all_vals.append(v[np.isfinite(v)])
    if not all_vals:
        return 0.0, 1.0
    combined = np.concatenate(all_vals)
    lo, hi = combined.min(), combined.max()
    pad = 0.05 * (hi - lo) if hi > lo else 0.1
    return lo - pad, hi + pad


def plot_swell_scatter(
    meta_df: pd.DataFrame,
    plotvariables: dict,
    chapter: str = "05",
    share_axes: bool = True,
    annotate_delta: bool = True,
    figsize: Tuple = (14, 5),
) -> None:
    """
    IN vs OUT amplitude scatter for Swell, Wind, and Total bands.
    as calculated for in "Probe {i} Swell Amplitude (PSD)"
    PC.SWELL_AMPLITUDE_PSD and PC.WIND_AMPLITUDE_PSD,

    in_probe/out_probe are derived automatically from file_date via ProbeConfiguration.

    show_plot → all three bands side-by-side + data summary panel
    save_plot → one PDF/PGF per band + one .tex stub with three subfigures

    Input: combined_meta_sel (filtering applied internally)
    """
    plotting = plotvariables.get("plotting", {})
    show_plot = plotting.get("show_plot", False)
    save_plot = plotting.get("save_plot", False)

    print("\n" + "=" * 50)
    print("plot_swell_scatter — filtering")
    band_amplitudes = filter_for_amplitude_plot(meta_df, plotvariables)
    if band_amplitudes.empty:
        print("No data after filtering — aborting.")
        return
    print(f"  {len(band_amplitudes)} rows remaining")

    in_pos, out_pos = _in_out_positions_from_df(band_amplitudes)
    print(f"  probe config: IN=Probe {in_pos}, OUT=Probe {out_pos}")

    shared_lim = _swell_shared_lim(band_amplitudes, in_pos, out_pos) if share_axes else None

    if show_plot:
        # figsize = plotting.get("figsize") or (14, 5)
        fig = plt.figure(figsize=figsize)
        n_bands = len(_BAND_COLS)
        gs = fig.add_gridspec(1, n_bands + 1, width_ratios=[1] * n_bands + [0.38])
        axes = [fig.add_subplot(gs[0, i]) for i in range(n_bands)]
        info_ax = fig.add_subplot(gs[0, -1])
        info_ax.axis("off")

        # Summary text
        has_wind = GC.WIND_CONDITION in band_amplitudes.columns
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
        info_ax.text(
            0.05,
            0.97,
            "\n".join(lines),
            transform=info_ax.transAxes,
            fontsize=7,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3, edgecolor="none"),
        )

        for ax, (band_name, template) in zip(axes, _BAND_COLS.items()):
            _draw_swell_scatter_ax(
                ax,
                band_amplitudes,
                in_col=template.format(i=in_pos),
                out_col=template.format(i=out_pos),
                band_name=band_name,
                shared_lim=shared_lim,
                annotate_delta=annotate_delta,
            )
            ax.set_xlabel(f"Probe {in_pos} IN amplitude [mm]", fontsize=9)
            ax.set_ylabel(f"Probe {out_pos} OUT amplitude [mm]", fontsize=9)

        fig.suptitle(
            "IN vs OUT — Swell / Wind / Total", fontsize=12, fontweight="bold", y=1.01
        )
        plt.tight_layout()
        plt.show()

    if save_plot:
        subfig_filenames = []
        meta_base = build_fig_meta(
            plotvariables,
            chapter=chapter,
            extra={"script": "plotter.py::plot_swell_scatter"},
        )

        for band_name, template in _BAND_COLS.items():
            fig_s, ax_s = plt.subplots(figsize=(4.5, 4.2))
            _draw_swell_scatter_ax(
                ax_s,
                band_amplitudes,
                in_col=template.format(i=in_pos),
                out_col=template.format(i=out_pos),
                band_name=band_name,
                shared_lim=shared_lim,
                annotate_delta=annotate_delta,
            )
            ax_s.set_xlabel(f"Probe {in_pos} IN amplitude [mm]", fontsize=9)
            ax_s.set_ylabel(f"Probe {out_pos} OUT amplitude [mm]", fontsize=9)
            fig_s.tight_layout()

            band_meta = {**meta_base, "band": band_name.lower()}
            fname = build_filename(f"swell_{band_name.lower()}", band_meta)
            _save_figure(fig_s, fname, save_pgf=True)
            subfig_filenames.append(fname)
            plt.close(fig_s)

        write_figure_stub(meta_base, "swell_scatter", subfig_filenames=subfig_filenames)


# ═══════════════════════════════════════════════════════════════════════════════
# FREQUENCY SPECTRUM
# ═══════════════════════════════════════════════════════════════════════════════


def plot_frequency_spectrum(
    fft_dict: dict,
    meta_df: pd.DataFrame,
    plotvariables: dict,
    data_type: str = "fft",
    chapter: str = "05",
) -> tuple:
    """
    Frequency spectrum (FFT or PSD), faceted by probe / wind / panel.

    Returns (fig, axes).
    """
    plotting = plotvariables.get("plotting", {})
    show_plot = plotting.get("show_plot", False)
    save_plot = plotting.get("save_plot", False)

    facet_by = plotting.get("facet_by", None)
    probes = plotting.get("probes", [])
    if not isinstance(probes, (list, tuple)):
        probes = [probes]
    n_peaks = plotting.get("peaks", None)
    log_scale = plotting.get("logaritmic", False)
    max_points = plotting.get("max_points", 120)
    show_grid = plotting.get("grid", True)
    xlim = plotting.get("xlim", (0, 10))
    linewidth = plotting.get("linewidth", 1.0)
    fontsize = 7

    col_prefix = "Pxx" if data_type.lower() == "psd" else "FFT"
    ylabel = col_prefix

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

    fig, axes = plt.subplots(
        n_facets, figsize=figsize, sharex=True, squeeze=False, dpi=120
    )
    axes = axes.flatten()

    for facet_idx, (group, facet_label) in enumerate(zip(facet_groups, facet_labels)):
        ax = axes[facet_idx]

        subset = (
            meta_df[meta_df["WindCondition"] == group]
            if facet_by == "wind"
            else meta_df[meta_df["PanelCondition"] == group]
            if facet_by == "panel"
            else meta_df
        )

        for _, row in subset.iterrows():
            path = row["path"]
            if path not in fft_dict:
                continue

            df_fft = fft_dict[path]
            windcond = row.get("WindCondition", "unknown")
            color = WIND_COLOR_MAP.get(windcond, "black")
            panelcond = row.get("PanelCondition", "unknown")
            lstyle = PANEL_STYLES.get(panelcond, "solid")
            pk_marker = MARKER_STYLES.get(windcond, ".")
            label_base = make_label(row)

            probes_to_plot = [group] if facet_by == "probe" else probes

            for probe_num in probes_to_plot:
                col = f"{col_prefix} {probe_num}"
                if col not in df_fft:
                    continue
                from wavescripts.signal_processing import get_positive_spectrum
                df_pos = get_positive_spectrum(df_fft)
                y = df_pos[col].dropna().iloc[:max_points]
                if y.empty:
                    continue
                x = y.index.values

                plot_label = (
                    label_base
                    if facet_by == "probe"
                    else f"{label_base}_P{probe_num}"
                    if len(probes_to_plot) > 1
                    else label_base
                )

                ax.plot(
                    x,
                    y.values,
                    linewidth=linewidth,
                    label=plot_label,
                    linestyle=lstyle,
                    color=color,
                    antialiased=False,
                )

                if n_peaks and n_peaks > 0:
                    from wavescripts.plot_utils import _top_k_indices

                    top_idx = _top_k_indices(y.values, n_peaks)
                    ax.scatter(
                        x[top_idx],
                        y.values[top_idx],
                        color=color,
                        s=80,
                        zorder=5,
                        marker=pk_marker,
                        linewidths=0.7,
                    )

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

    # ── caption ──────────────────────────────────────────────────────────────
    _wind_conds = sorted(meta_df["WindCondition"].dropna().unique().tolist())
    _panel_conds = sorted(meta_df["PanelCondition"].dropna().unique().tolist())
    _caption_slots = {
        "n_runs":        len(meta_df),
        "n_wind":        int(meta_df["WindCondition"].isin(["full", "lowest"]).sum()),
        "n_stillwater":  int((meta_df["WindCondition"] == "no").sum()),
        "wind_conds":    ", ".join(_wind_conds) if _wind_conds else "none",
        "panel_conds":   ", ".join(_panel_conds) if _panel_conds else "none",
        "probes":        ", ".join(str(p) for p in probes),
        "xlim_lo":       xlim[0],
        "xlim_hi":       xlim[1],
        "data_type":     data_type.upper(),
    }
    _default_caption = (
        "{data_type} of the free surface at each wave gauge "
        "({n_runs} runs: {n_wind} wind, {n_stillwater} stillwater). "
        "Wind conditions: {wind_conds}. "
        "Frequency range shown: {xlim_lo}--{xlim_hi}\\,Hz."
    )
    _caption = resolve_caption(
        plotting, _default_caption, _caption_slots,
        fn_name="plot_frequency_spectrum",
    )

    if save_plot:
        meta_base = build_fig_meta(
            {**plotvariables, "plotting": {**plotting, "caption": _caption}},
            chapter=chapter,
            extra={
                "script": "plotter.py::plot_frequency_spectrum",
                "data_type": data_type,
            },
        )
        save_separate = plotting.get("save_separate", True)

        if not save_separate:
            # Current behaviour — save the whole faceted figure as one file
            save_and_stub(
                fig,
                meta_base,
                plot_type=f"spectrum_{data_type}",
                save_pgf=True,
                force_stub=plotting.get("force_stub", False),
            )
        else:
            # Save one figure per facet panel
            subfig_filenames = []

            for group, facet_label in zip(facet_groups, facet_labels):
                fig_s, ax_s = plt.subplots(
                    figsize=plotting.get("figsize_single", (9, 5))
                )

                # Determine subset same way as main loop
                subset = (
                    meta_df[meta_df["WindCondition"] == group]
                    if facet_by == "wind"
                    else meta_df[meta_df["PanelCondition"] == group]
                    if facet_by == "panel"
                    else meta_df
                )

                probes_to_plot = [group] if facet_by == "probe" else probes

                for _, row in subset.iterrows():
                    path = row["path"]
                    if path not in fft_dict:
                        continue
                    df_fft = fft_dict[path]
                    color = WIND_COLOR_MAP.get(row.get("WindCondition"), "black")
                    lstyle = PANEL_STYLES.get(row.get("PanelCondition", ""), "solid")
                    pk_marker = MARKER_STYLES.get(row.get("WindCondition"), ".")
                    label_base = make_label(row)

                    for probe_num in probes_to_plot:
                        col = f"{col_prefix} {probe_num}"
                        if col not in df_fft:
                            continue
                        from wavescripts.signal_processing import get_positive_spectrum
                        df_pos = get_positive_spectrum(df_fft)
                        y = df_pos[col].dropna().iloc[:max_points]
                        if y.empty:
                            continue
                        x = y.index.values
                        ax_s.plot(
                            x,
                            y.values,
                            linewidth=linewidth,
                            label=label_base,
                            linestyle=lstyle,
                            color=color,
                            antialiased=False,
                        )
                        if n_peaks and n_peaks > 0:
                            top_idx = _top_k_indices(y.values, n_peaks)
                            ax_s.scatter(
                                x[top_idx],
                                y.values[top_idx],
                                color=color,
                                s=80,
                                zorder=5,
                                marker=pk_marker,
                                linewidths=0.7,
                            )

                ax_s.set_title(facet_label, fontsize=9)
                ax_s.set_xlabel(col_prefix, fontsize=9)
                ax_s.set_ylabel(ylabel, fontsize=9)
                if xlim:
                    ax_s.set_xlim(xlim)
                if log_scale:
                    ax_s.set_yscale("log")
                if use_locators:
                    ax_s.xaxis.set_minor_locator(ticker.MultipleLocator(base_freq))
                    ax_s.xaxis.set_major_locator(ticker.MultipleLocator(2 * base_freq))
                if show_grid:
                    ax_s.grid(which="major", linestyle="--", alpha=0.6)
                    ax_s.grid(which="minor", linestyle="-.", alpha=0.3)
                apply_legend(ax_s, plotvariables)
                fig_s.tight_layout()

                # Per-panel meta — override probe for filename
                panel_meta = {**meta_base}
                if facet_by == "probe":
                    panel_meta["probes"] = group
                fname = build_filename(f"spectrum_{data_type}", panel_meta)
                _save_figure(fig_s, fname, save_pgf=True)
                subfig_filenames.append(fname)
                plt.close(fig_s)

            # One stub with all panels as subfigures
            write_figure_stub(
                meta_base, f"spectrum_{data_type}", subfig_filenames=subfig_filenames,
                force=plotting.get("force_stub", False),
            )

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return fig, axes


# ═══════════════════════════════════════════════════════════════════════════════
# RECONSTRUCTED SIGNAL
# ═══════════════════════════════════════════════════════════════════════════════


def plot_reconstructed(
    fft_dict: Dict[str, pd.DataFrame],
    filtrert_frequencies: pd.DataFrame,
    freqplotvariables: dict,
    data_type: str = "fft",
    chapter: str = "05",
) -> Tuple[Optional[plt.Figure], Optional[np.ndarray]]:
    """
    Reconstructed wave vs wind component for a SINGLE experiment.
    Facet by probe if plotting["facet_by"] == "probe".
    """
    meta_df = filtrert_frequencies.copy()
    plotting = freqplotvariables.get("plotting", {})
    show_plot = plotting.get("show_plot", False)
    save_plot = plotting.get("save_plot", False)

    probes = plotting.get("probes", [])
    probes = [probes] if not isinstance(probes, (list, tuple)) else probes
    facet_by = plotting.get("facet_by", None)
    show_grid = plotting.get("grid", True)
    linewidth = plotting.get("linewidth", 1.2)
    dual_yaxis = plotting.get("dual_yaxis", True)
    show_full = plotting.get("show_full_signal", False)
    show_stats = plotting.get("show_amplitude_stats", True)
    fontsize = 9

    if len(fft_dict) == 0:
        print("ERROR: fft_dict is empty")
        return None, None
    if len(fft_dict) > 1:
        print(f"WARNING: plotting only first of {len(fft_dict)} experiments")

    path = list(fft_dict.keys())[0]
    df_fft = fft_dict[path]
    row = meta_df[meta_df["path"] == path]
    if len(row) == 0:
        print(f"ERROR: no metadata for {path}")
        return None, None
    row = row.iloc[0]

    windcond = row.get("WindCondition", "unknown")
    panelcond = row.get("PanelCondition", "unknown")
    target_freq = row.get(GC.WAVE_FREQUENCY_INPUT, None)
    if not target_freq or target_freq <= 0:
        print(f"ERROR: invalid target frequency {target_freq}")
        return None, None

    color_swell = WIND_COLOR_MAP.get(windcond, "black")
    color_wind = "darkred" if dual_yaxis else "orange"
    color_full = "gray"
    lstyle = {"no": "-", "full": "--", "reverse": "-."}.get(panelcond, "-")

    n_subplots = len(probes) if facet_by == "probe" else 1
    if n_subplots == 0:
        print("plot_reconstructed: no probes selected, nothing to plot")
        return None, None
    figsize = plotting.get("figsize") or (16, 5 * n_subplots if n_subplots > 1 else 7)
    fig, axes = plt.subplots(n_subplots, 1, figsize=figsize, squeeze=False, dpi=120)
    axes = axes.flatten()
    amplitude_comparison = []

    for subplot_idx in range(n_subplots):
        ax_s = axes[subplot_idx]
        ax_w = ax_s.twinx() if dual_yaxis else ax_s
        probes_here = [probes[subplot_idx]] if facet_by == "probe" else probes
        title = (
            f"Probe {probes[subplot_idx]}"
            if facet_by == "probe"
            else f"{windcond} wind / {panelcond} panel / {target_freq:.3f} Hz"
        )

        for probe_num in probes_here:
            col = f"FFT {probe_num} complex"
            if col not in df_fft:
                col = f"FFT {probe_num}"
            if col not in df_fft:
                available = [c for c in df_fft.columns if c.startswith("FFT") and "complex" in c]
                print(f"Skipping probe {probe_num}: column not found. Available: {available}")
                continue

            fft_series = df_fft[col].dropna()
            if fft_series.empty:
                print(f"Skipping probe {probe_num}: column '{col}' is all-NaN")
                continue
            freq_bins = fft_series.index.values      # sorted: negative → positive
            fft_complex = fft_series.values
            N = len(fft_complex)
            df_freq = abs(freq_bins[1] - freq_bins[0])
            sr = df_freq * N                          # ≈ sampling rate (Hz)

            # freq_bins is the sorted (fftshifted) order from sort_index().
            # ifftshift converts it back to numpy's natural FFT order [0, pos, neg].
            fft_ord = np.fft.ifftshift(fft_complex).astype(complex)
            fftfreqs = np.fft.ifftshift(freq_bins)

            signal_full = np.real(np.fft.ifft(fft_ord))
            time_axis = np.arange(N) / sr
            pos_freqs = fftfreqs[fftfreqs > 0]
            actual_freq = pos_freqs[np.argmin(np.abs(pos_freqs - target_freq))]
            peak_idx   = np.argmin(np.abs(fftfreqs - actual_freq))
            mirror_idx = np.argmin(np.abs(fftfreqs + actual_freq))

            fft_swell = np.zeros_like(fft_ord, dtype=complex)
            fft_swell[peak_idx] = fft_ord[peak_idx]
            fft_swell[mirror_idx] = fft_ord[mirror_idx]
            signal_swell = np.real(np.fft.ifft(fft_swell))
            signal_wind = signal_full - signal_swell

            lp = "" if facet_by == "probe" else f"P{probe_num} "
            if show_full:
                ax_s.plot(
                    time_axis,
                    signal_full,
                    lw=linewidth * 0.7,
                    label=f"{lp}full",
                    color=color_full,
                    alpha=0.4,
                    zorder=1,
                )
            ax_s.plot(
                time_axis,
                signal_swell,
                lw=linewidth * 1.5,
                label=f"{lp}wave ({actual_freq:.4f}Hz)",
                linestyle=lstyle,
                color=color_swell,
                alpha=0.9,
                zorder=3,
            )
            ax_w.plot(
                time_axis,
                signal_wind,
                lw=linewidth,
                label=f"{lp}wind",
                linestyle=":" if dual_yaxis else "--",
                color=color_wind,
                alpha=0.7,
                zorder=2,
            )

        ax_s.set_title(title, fontsize=fontsize + 2, fontweight="bold", pad=15)
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
        lines_w, labs_w = ax_w.get_legend_handles_labels() if dual_yaxis else ([], [])
        if lines_s + lines_w:
            ax_s.legend(
                lines_s + lines_w,
                labs_s + labs_w,
                loc="upper right",
                fontsize=fontsize,
                framealpha=0.95,
            )
        ax_s.axhline(0, color="black", lw=0.5, alpha=0.3)

    plt.suptitle(
        f"{Path(path).stem}\n{windcond} / {panelcond} / {target_freq:.2f} Hz",
        fontsize=fontsize + 3,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()

    if save_plot:
        meta = build_fig_meta(
            freqplotvariables,
            chapter=chapter,
            extra={"script": "plotter.py::plot_reconstructed"},
        )
        save_and_stub(
            fig,
            meta,
            plot_type="reconstructed",
            save_pgf=True,
        )

    if show_plot:
        plt.show(block=False)
    else:
        plt.close(fig)

    return fig, axes


def plot_reconstructed_rms(
    fft_dict: dict,
    filtrert_frequencies: pd.DataFrame,
    freqplotvariables: dict,
    data_type: str = "fft",
    chapter: str = "05",
):
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


def gather_ramp_data(
    dfs: dict,
    combined_meta_sel: pd.DataFrame,
    sigma_factor: float = SIGNAL.BASELINE_SIGMA_FACTOR,
) -> pd.DataFrame:
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

        # Derive position strings from eta_ columns present in this df
        positions = [c[len("eta_"):] for c in df.columns if c.startswith("eta_")]

        for pos in positions:
            col_raw = f"Probe {pos}"
            col_eta = f"eta_{pos}"
            col_ma = f"Probe {pos}_ma"
            col_start = PC.START.format(i=pos)
            col_end = PC.END.format(i=pos)

            if col_raw not in df.columns or col_eta not in df.columns:
                continue

            good_start = meta_row.get(col_start)
            good_end = meta_row.get(col_end)
            if good_start is None or pd.isna(good_start):
                continue
            if good_end is None or pd.isna(good_end):
                continue

            good_start = int(good_start)
            good_end = int(good_end)
            signal = df[col_eta].to_numpy()
            raw = df[col_raw].to_numpy()
            ma = df[col_ma].to_numpy() if col_ma in df.columns else signal
            col_interp = f"eta_{pos}_interp"
            signal_interp = df[col_interp].to_numpy() if col_interp in df.columns else signal

            n_base = min(
                good_start,
                int(SIGNAL.BASELINE_DURATION_SEC * MEASUREMENT.SAMPLING_RATE),
            )
            base_region = signal[:n_base] if n_base > 10 else signal[:50]
            base_std = np.std(base_region) if len(base_region) > 0 else 1.0
            threshold = sigma_factor * base_std
            exceeded = np.where(np.abs(signal) > threshold)[0]
            first_motion = int(exceeded[0]) if len(exceeded) > 0 else good_start

            _sw = meta_row.get(PC.STILLWATER.format(i=pos), np.nan)
            baseline_mean_val = (
                float(_sw) if pd.notna(_sw)
                else (float(np.nanmean(base_region)) if len(base_region) > 0 else 0.0)
            )

            t0 = df["Date"].iat[0]
            time_ms = (
                df["Date"] - t0
            ).dt.total_seconds().to_numpy() * MEASUREMENT.M_TO_MM

            records.append(
                {
                    GC.PATH: path,
                    "experiment": Path(path).stem,
                    "probe": pos,
                    "data_col": col_raw,
                    GC.WIND_CONDITION: meta_row.get(GC.WIND_CONDITION, "unknown"),
                    GC.PANEL_CONDITION: meta_row.get(GC.PANEL_CONDITION, "unknown"),
                    GC.WAVE_FREQUENCY_INPUT: meta_row.get(
                        GC.WAVE_FREQUENCY_INPUT, np.nan
                    ),
                    GC.WAVE_AMPLITUDE_INPUT: meta_row.get(
                        GC.WAVE_AMPLITUDE_INPUT, np.nan
                    ),
                    "time_ms": time_ms,
                    "raw": raw,
                    "signal": signal,
                    "signal_interp": signal_interp,
                    "ma": ma,
                    "baseline_mean": baseline_mean_val,
                    "baseline_std": base_std,
                    "threshold": threshold,
                    "first_motion_idx": first_motion,
                    "good_start_idx": good_start,
                    "good_end_idx": good_end,
                    "good_range": good_end - good_start,
                }
            )

    df_out = pd.DataFrame(records)
    print(
        f"\nRamp data: {df_out['experiment'].nunique()} experiments, {len(df_out)} rows"
    )
    return df_out


def plot_ramp_detection(
    df: pd.DataFrame,
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
    title: str = "Ramp Detection",
    signal_interp: Optional[np.ndarray] = None,
    expected_sine: Optional[np.ndarray] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Diagnostic plot for ramp detection on a single experiment/probe."""
    if "Date" not in df.columns:
        raise ValueError("df must contain a 'Date' column")
    if data_col not in df.columns:
        raise ValueError(f"df must contain '{data_col}' column")

    t0 = df["Date"].iat[0]
    time_ms = (df["Date"] - t0).dt.total_seconds().to_numpy() * MEASUREMENT.M_TO_MM
    raw = df[data_col].to_numpy()
    n = len(time_ms)

    gs_i = max(0, min(int(good_start_idx), n - 2))
    ge_i = (
        max(gs_i + 1, min(int(good_end_idx), n - 1))
        if good_end_idx is not None
        else max(gs_i + 1, min(gs_i + int(good_range), n - 1))
    )

    fig, ax = plt.subplots(figsize=(15, 7))
    fig.suptitle(title)

    ax.plot(time_ms, raw, color="lightgray", alpha=0.6, label="Raw")
    if signal_interp is not None:
        ax.plot(time_ms, signal_interp, color="steelblue", lw=1.2, alpha=0.7, label="Cleaned (interp)")
    ax.plot(time_ms, signal, color="black", lw=1.5, alpha=0.8, label=f"Cleaned (gaps=NaN)")
    if expected_sine is not None:
        ax.plot(time_ms, expected_sine, color="darkorange", lw=1.5, alpha=0.8,
                linestyle="--", label="Expected sine (FFT-fit)")
    ax.axhline(
        baseline_mean,
        color="blue",
        linestyle="--",
        label=f"Baseline = {baseline_mean:.2f} mm",
    )
    ax.axhline(baseline_mean + threshold, color="red", linestyle=":", alpha=0.7)
    ax.axhline(baseline_mean - threshold, color="red", linestyle=":", alpha=0.7)
    ax.axvline(
        time_ms[first_motion_idx],
        color="orange",
        lw=2,
        linestyle="--",
        label=f"First motion #{first_motion_idx}",
    )
    ax.axvline(time_ms[gs_i], color="green", lw=3, label=f"Stable start #{gs_i}")
    ax.axvline(
        time_ms[ge_i], color="purple", lw=2, linestyle="--", label=f"End #{ge_i}"
    )
    ax.axvspan(time_ms[gs_i], time_ms[ge_i], color="green", alpha=0.08)

    if peaks is not None and len(peaks) > 0:
        pk = np.asarray(peaks, dtype=int)
        pk = pk[(pk >= 0) & (pk < n)]
        ax.plot(time_ms[pk], signal[pk], "ro", ms=6, alpha=0.7, label="Peaks")
    if ramp_peak_indices is not None and len(ramp_peak_indices) > 0:
        rpi = np.asarray(ramp_peak_indices, dtype=int)
        rpi = rpi[(rpi >= 0) & (rpi < n)]
        ax.plot(
            time_ms[rpi],
            signal[rpi],
            "o",
            color="lime",
            ms=10,
            markeredgecolor="darkgreen",
            markeredgewidth=2,
            label=f"Ramp-up ({len(rpi)} peaks)",
        )

    try:
        path_val = (
            meta_sel["path"]
            if isinstance(meta_sel, pd.Series)
            else meta_sel["path"].iloc[0]
        )
        ax.set_title(f"{Path(str(path_val)).stem}  →  {data_col}", fontsize=14, pad=20)
    except Exception:
        pass

    amp_in = float(meta_sel.get(GC.WAVE_AMPLITUDE_INPUT, 0) or 0)
    zoom = max(amp_in * 100, 15)
    ax.set_ylim(baseline_mean - zoom, baseline_mean + zoom)
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Water level [mm]")
    ax.grid(True, alpha=0.1)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    fig.tight_layout()
    return fig, ax


# ═══════════════════════════════════════════════════════════════════════════════
# PROBE NOISE FLOOR
# ═══════════════════════════════════════════════════════════════════════════════


def plot_probe_noise_floor(
    combined_meta: pd.DataFrame,
    processed_dfs: dict,
    probe_positions: list[str],
    plotvariables: dict,
    *,
    window_s: float = 0.2,
    exclude_keywords: tuple[str, ...] = ("nestenstille",),
    amp_cap_mm: float = 0.5,
    chapter: str = "04",
) -> tuple[plt.Figure, pd.DataFrame]:
    """
    Compute and plot the stillwater probe noise floor.

    Noise floor = minimum windowed (P97.5-P2.5)/2 amplitude across short
    sliding windows of length *window_s* seconds. The short window ensures
    slow tank sloshing (2-10 s period) appears as a DC offset inside the
    window and does not inflate the percentile spread. What remains is pure
    probe noise (electronics jitter + capillary ripples).

    Parameters
    ----------
    combined_meta   : full metadata DataFrame (all runs)
    processed_dfs   : dict {path: DataFrame} with eta_{pos} columns
    probe_positions : list of position strings, e.g. ["8804/250", "9373/170"]
    plotvariables   : dict with keys
        "filters"  — unused here (stillwater is selected automatically)
        "plotting" — {
            "show_plot": bool,
            "save_plot": bool,
            "figure_name": str,
            "caption": str  (optional) — template string for the LaTeX caption.
                Omit to use the built-in default (which is also printed to
                terminal so you can copy-edit it). Supports named slots:
                  {window_ms}   window length [ms]
                  {n_runs}      accepted stillwater runs
                  {n_flagged}   excluded runs
                  {amp_cap_mm}  amplitude cap threshold [mm]
                Example:
                  "caption": (
                      "Noise floor of each wave gauge over {n_runs} "
                      "stillwater recordings ({window_ms:.0f}\\,ms window)."
                  )
        }
    window_s        : sliding window length in seconds (default 0.2 s)
    exclude_keywords: run filenames containing any of these are flagged/excluded
    amp_cap_mm      : runs whose windowed-min exceeds this [mm] are also flagged
    chapter         : thesis chapter string for save_and_stub (default "04")

    Returns
    -------
    fig : matplotlib Figure
    summary : DataFrame  — per-probe mean/std/min/max noise floor [mm],
              index = probe position string
    """
    from wavescripts.constants import MEASUREMENT

    fs = MEASUREMENT.SAMPLING_RATE
    window_n = int(window_s * fs)

    plotting = plotvariables.get("plotting", {})
    show_plot = plotting.get("show_plot", False)
    save_plot = plotting.get("save_plot", False)
    figure_name = plotting.get("figure_name", "ch04_probe_noise_floor")

    # --- select stillwater runs ---
    is_stillwater = (
        combined_meta["WindCondition"].eq("no")
        & combined_meta["WaveFrequencyInput [Hz]"].isna()
    )
    meta_sw = combined_meta[is_stillwater].copy()

    # --- compute windowed-minimum noise floor per run × probe ---
    rows = []
    for _, row in meta_sw.iterrows():
        df = processed_dfs.get(row["path"])
        if df is None:
            continue
        entry = {"run": Path(str(row["path"])).name}
        for pos in probe_positions:
            col = f"eta_{pos}"
            if col not in df.columns:
                entry[pos] = np.nan
                continue
            sig = df[col].values
            step = max(1, window_n // 2)
            from numpy.lib.stride_tricks import sliding_window_view
            windows = sliding_window_view(sig, window_n)[::step]  # (n_wins, window_n)
            # drop windows with too many NaNs
            valid = np.sum(~np.isnan(windows), axis=1) >= window_n // 2
            windows = windows[valid]
            if windows.size == 0:
                entry[pos] = np.nan
                continue
            p_hi = np.nanpercentile(windows, 97.5, axis=1)
            p_lo = np.nanpercentile(windows, 2.5,  axis=1)
            entry[pos] = float(np.nanmin((p_hi - p_lo) / 2))
        rows.append(entry)

    if not rows:
        raise ValueError(
            "No stillwater runs found in processed_dfs. "
            "Load with load_processed=True first."
        )

    sw_all = pd.DataFrame(rows)

    # --- flag and exclude bad runs ---
    probe_cols_present = [p for p in probe_positions if p in sw_all.columns]
    name_flag = sw_all["run"].apply(
        lambda r: any(kw in r for kw in exclude_keywords)
    )
    amp_flag = sw_all[probe_cols_present].max(axis=1) > amp_cap_mm
    bad = name_flag | amp_flag
    sw_clean = sw_all[~bad].copy()

    # --- summary statistics ---
    summary = sw_clean[probe_cols_present].agg(["mean", "std", "min", "max"]).T
    summary.index.name = "probe"

    # --- plot ---
    apply_thesis_style()
    fig, ax = plt.subplots(figsize=(max(4, len(probe_cols_present) * 1.1), 4))

    x = np.arange(len(probe_cols_present))
    means = summary["mean"].values
    stds = summary["std"].fillna(0).values

    ax.bar(x, means, yerr=stds, capsize=4, color="steelblue", alpha=0.85,
           error_kw={"elinewidth": 1.2, "ecolor": "navy"})

    # overlay individual run dots
    for i, pos in enumerate(probe_cols_present):
        vals = sw_clean[pos].dropna().values
        ax.scatter(
            np.full(len(vals), i),
            vals,
            s=20, color="white", edgecolors="navy", linewidths=0.8, zorder=3
        )

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(probe_cols_present, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Noise floor [mm]")
    ax.set_title(
        f"Probe noise floor — min windowed amplitude ({window_s*1000:.0f} ms window)\n"
        f"n={len(sw_clean)} stillwater runs accepted, {bad.sum()} flagged",
        fontsize=10,
    )
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    if show_plot:
        plt.show()

    # ── caption ──────────────────────────────────────────────────────────────
    _caption_slots = {
        "window_ms": window_s * 1e3,
        "n_runs":    len(sw_clean),
        "n_flagged": int(bad.sum()),
        "amp_cap_mm": amp_cap_mm,
    }
    _default_caption = (
        "Probe noise floor estimated as the minimum windowed "
        "(P$_{{97.5}}$--P$_{{2.5}}$)/2 amplitude over {window_ms:.0f}\\,ms "
        "sliding windows of {n_runs} stillwater recordings "
        "({n_flagged} run(s) excluded — name keyword or windowed minimum "
        "above {amp_cap_mm}\\,mm). "
        "Short windows suppress slow tank sloshing so only electronic "
        "jitter and capillary ripples remain. "
        "Error bars: standard deviation across runs. "
        "White dots: individual run values."
    )
    _caption = resolve_caption(
        plotting, _default_caption, _caption_slots,
        fn_name="plot_probe_noise_floor",
    )

    if save_plot:
        meta = build_fig_meta(
            {**plotvariables, "plotting": {**plotting, "caption": _caption}},
            chapter=chapter,
        )
        save_and_stub(fig, meta, plot_type="probe_noise_floor")

    return fig, summary


# ═══════════════════════════════════════════════════════════════════════════════
# PARALLEL RATIO
# ═══════════════════════════════════════════════════════════════════════════════


def plot_parallel_ratio(
    combined_meta: pd.DataFrame,
    plotvariables: dict,
    *,
    chapter: str = "04",
) -> plt.Figure:
    """
    Plot parallel_ratio (wall-side / far-side amplitude) vs wave frequency,
    coloured by WindCondition, one subplot per PanelCondition.

    parallel_ratio ≈ 1  →  lateral symmetry
    parallel_ratio > 1  →  wall-side probe sees larger amplitude
    parallel_ratio < 1  →  far-side probe sees larger amplitude

    Parameters
    ----------
    combined_meta : full metadata DataFrame
    plotvariables : dict with keys
        "filters"  — optional pre-filter dict (passed to apply_experimental_filters)
        "plotting" — {
            "show_plot":   bool,
            "save_plot":   bool,
            "figure_name": str,
            "caption":     str  (optional template, slots: {n_runs}, {n_panels})
        }
    chapter : thesis chapter string for save_and_stub (default "04")

    Caption slots
    -------------
    {n_runs}    total wave runs included
    {n_panels}  number of PanelCondition subplots
    """
    from wavescripts.filters import apply_experimental_filters

    plotting = plotvariables.get("plotting", {})
    show_plot   = plotting.get("show_plot",   False)
    save_plot   = plotting.get("save_plot",   False)
    figure_name = plotting.get("figure_name", "ch04_parallel_ratio")

    # --- filter to wave runs with a valid parallel_ratio ---
    df = apply_experimental_filters(combined_meta, plotvariables)
    df = df[df["WaveFrequencyInput [Hz]"].notna() & df["parallel_ratio"].notna()].copy()

    panel_vals = sorted(df["PanelCondition"].dropna().unique())
    n_panels = len(panel_vals)
    if n_panels == 0:
        raise ValueError("No rows with valid parallel_ratio after filtering.")

    apply_thesis_style()
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(4 * n_panels, 4),
        sharey=True,
        squeeze=False,
    )

    wind_order = ["no", "lowest", "full"]

    for ax, panel in zip(axes[0], panel_vals):
        sub = df[df["PanelCondition"] == panel]
        for wind in wind_order:
            grp = sub[sub["WindCondition"] == wind]
            if grp.empty:
                continue
            color = WIND_COLOR_MAP.get(wind, "gray")
            # mean ± std per frequency
            stats = grp.groupby("WaveFrequencyInput [Hz]")["parallel_ratio"].agg(["mean", "std", "count"])
            ax.errorbar(
                stats.index, stats["mean"],
                yerr=stats["std"].fillna(0),
                fmt="o-", capsize=3, lw=1.4, ms=5,
                color=color, label=wind,
            )
        ax.axhline(1.0, color="black", lw=0.8, ls="--", alpha=0.5)
        ax.set_title(f"panel: {panel}", fontsize=10)
        ax.set_xlabel("Frequency [Hz]")
        ax.grid(True, alpha=0.3)

    axes[0][0].set_ylabel("Parallel ratio (wall / far)")
    axes[0][0].legend(title="Wind", fontsize=8)
    fig.suptitle("Lateral symmetry — parallel probe ratio", fontsize=11)
    fig.tight_layout()

    if show_plot:
        plt.show()

    # ── caption ──────────────────────────────────────────────────────────────
    _caption_slots = {
        "n_runs":   len(df),
        "n_panels": n_panels,
        "panel_conditions": ", ".join(str(p) for p in panel_vals),
    }
    _default_caption = (
        "Ratio of wall-side to far-side probe amplitude at the same longitudinal "
        "distance, for {n_runs} wave runs across {n_panels} panel "
        "condition(s) ({panel_conditions}). "
        "A ratio of 1 indicates lateral symmetry. "
        "Deviations indicate wall reflections or wind-driven lateral asymmetry. "
        "Error bars: standard deviation across runs at the same frequency. "
        "Dashed line: ratio = 1."
    )
    _caption = resolve_caption(
        plotting, _default_caption, _caption_slots,
        fn_name="plot_parallel_ratio",
    )

    if save_plot:
        meta = build_fig_meta(
            {**plotvariables, "plotting": {**plotting, "caption": _caption}},
            chapter=chapter,
        )
        save_and_stub(fig, meta, plot_type="parallel_ratio")

    return fig
