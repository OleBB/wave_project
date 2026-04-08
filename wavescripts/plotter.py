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
  WAVE STABILITY              plot_wave_stability
  TIME-SERIES OVERVIEW        plot_timeseries_overview
  INSTRUMENT DIAGNOSTICS      plot_sound_speed
  WIND CHARACTERISATION       plot_wind_snr, plot_td_vs_fft
  WAVE DETECTION              plot_first_arrival
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
    # ── SUBFIGURE SIZING RULE ─────────────────────────────────────────────────
    # Always use fixed subplots_adjust for any _make_*_fig helper that produces
    # a series of subfigures. NEVER use tight_layout() here — it adjusts margins
    # per content (legend size, tick label width, title length), so each saved
    # file gets different internal proportions even at the same figsize. That
    # breaks downstream LaTeX faceting where all subfigures must align.
    # Tune these values once for the plot type, then keep them identical across
    # every subfigure in the series.
    fig.subplots_adjust(left=0.14, right=0.97, top=0.90, bottom=0.13)
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
    wind_conditions  = sorted(stats_df[GC.WIND_CONDITION].unique())
    n_runs           = int(stats_df["n_runs"].sum()) if "n_runs" in stats_df.columns else len(stats_df)

    # Allow caption at plotvariables top level (string) as a convenience
    _top_caption = plotvariables.get("caption")
    if isinstance(_top_caption, str) and "caption" not in plotting:
        plotting = {**plotting, "caption": _top_caption}

    _caption_slots = {
        "n_runs":      n_runs,
        "n_panels":    len(panel_conditions),
        "panels":      ", ".join(panel_conditions),
        "n_amps":      len(amplitudes),
        "amps":        ", ".join(f"{a:.2f}\\,V" for a in amplitudes),
        "n_wind":      len(wind_conditions),
        "wind_conds":  ", ".join(wind_conditions),
    }
    _default_caption = (
        "Damping ratio OUT/IN (FFT amplitude at paddle frequency) versus "
        "wave frequency, for {panels} panel condition(s). "
        "Colour encodes wind condition ({wind_conds}); "
        "each line shows one amplitude ({amps}). "
        "Errorbars: standard deviation across repeated runs. "
        "Dashed line: ratio = 1 (no damping)."
    )
    _caption = resolve_caption(plotting, _default_caption, _caption_slots,
                               fn_name="plot_damping_freq")

    if show_plot:
        for panel in panel_conditions:
            for amp in amplitudes:
                fig = _make_damping_freq_fig(stats_df, panel, amp, figsize=figsize)
                plt.show()

    if save_plot:
        subfig_filenames = []
        meta_base = build_fig_meta(
            {**plotvariables, "plotting": {**plotting, "caption": _caption}},
            chapter=chapter,
            extra={"script": "plotter.py::plot_damping_freq"},
        )
        figure_name    = plotting.get("figure_name") or build_filename("damping_freq", meta_base)
        subfig_captions = []
        for panel in panel_conditions:
            for amp in amplitudes:
                fig_s = _make_damping_freq_fig(stats_df, panel, amp, figsize=figsize)
                amp_tag = f"{int(round(amp * 100)):02d}V"
                fname = f"{figure_name}_{panel}_{amp_tag}"
                _save_figure(fig_s, fname, save_pgf=True)
                subfig_filenames.append(fname)
                subfig_captions.append(f"{panel.capitalize()} panel, ${amp:.2f}$\\,V")
                plt.close(fig_s)

        stub_meta = {**meta_base, "panel": panel_conditions, "amplitude": amplitudes, "wind": "allwind"}
        write_figure_stub(stub_meta, "damping_freq", subfig_filenames=subfig_filenames,
                          subfig_captions=subfig_captions,
                          force=plotting.get("force_stub", False))


def _make_damping_scatter_fig(
    stats_df: pd.DataFrame, panel: str, figsize: tuple = (5, 4)
) -> plt.Figure:
    """
    Single axes: OUT/IN vs frequency for one panel condition.
    Colour = wind condition. Marker size = amplitude. No internal faceting.
    """
    import seaborn as sns
    subset = stats_df[stats_df[GC.PANEL_CONDITION_GROUPED] == panel]
    fig, ax = plt.subplots(figsize=figsize)

    sns.scatterplot(
        data=subset.sort_values(GC.WAVE_FREQUENCY_INPUT),
        x=GC.WAVE_FREQUENCY_INPUT, y="mean_out_in",
        hue=GC.WIND_CONDITION, palette=WIND_COLOR_MAP,
        size=GC.WAVE_AMPLITUDE_INPUT, sizes=(40, 160),
        alpha=0.80, ax=ax, legend="auto",
    )

    if "std_out_in" in subset.columns:
        for wind, grp in subset.groupby(GC.WIND_CONDITION):
            ax.errorbar(
                grp[GC.WAVE_FREQUENCY_INPUT], grp["mean_out_in"],
                yerr=grp["std_out_in"],
                fmt="none", ecolor=WIND_COLOR_MAP.get(wind, "gray"),
                elinewidth=1, capsize=3, alpha=0.4, zorder=1,
            )

    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.4)
    ax.set_xlabel("Frequency [Hz]", fontsize=9)
    ax.set_ylabel("OUT/IN (FFT)", fontsize=9)
    ax.set_title(f"{panel} panel", fontsize=9)
    ax.legend(title="wind / amp", fontsize=7, title_fontsize=7)
    ax.grid(True, alpha=0.3)
    # Fixed margins — see subfigure sizing rule in _make_damping_freq_fig
    fig.subplots_adjust(left=0.14, right=0.97, top=0.90, bottom=0.13)
    return fig


def plot_damping_scatter(
    stats_df: pd.DataFrame,
    plotvariables: Optional[dict] = None,
    chapter: str = "05",
) -> None:
    """
    OUT/IN scatter vs frequency, one figure per panel condition.
    Colour = wind. Size = amplitude. No internal faceting — LaTeX arranges subfigures.

    show_plot → one window per panel (REPL verification)
    save_plot → one PDF/PGF per panel + .tex stub

    Input: output from damping_all_amplitude_grouper()
    """
    if plotvariables is None:
        plotvariables = {"plotting": {"show_plot": True, "save_plot": False}}

    plotting  = plotvariables.get("plotting", {})
    show_plot = plotting.get("show_plot", False)
    save_plot = plotting.get("save_plot", False)
    figsize   = plotting.get("figsize", (5, 4))

    panel_conditions = sorted(stats_df[GC.PANEL_CONDITION_GROUPED].unique())
    amplitudes       = sorted(stats_df[GC.WAVE_AMPLITUDE_INPUT].unique())
    wind_conditions  = sorted(stats_df[GC.WIND_CONDITION].unique())
    n_runs           = int(stats_df["n_runs"].sum()) if "n_runs" in stats_df.columns else len(stats_df)

    # Allow caption at plotvariables top level (string) as a convenience
    _top_caption = plotvariables.get("caption")
    if isinstance(_top_caption, str) and "caption" not in plotting:
        plotting = {**plotting, "caption": _top_caption}

    _caption_slots = {
        "n_runs":     n_runs,
        "n_panels":   len(panel_conditions),
        "panels":     ", ".join(panel_conditions),
        "n_wind":     len(wind_conditions),
        "wind_conds": ", ".join(wind_conditions),
        "n_amps":     len(amplitudes),
        "amps":       ", ".join(f"{a:.2f}\\,V" for a in amplitudes),
    }
    _default_caption = (
        "OUT/IN damping ratio versus wave frequency, all amplitudes combined. "
        "{panels} panel condition(s); colour = wind condition ({wind_conds}); "
        "marker size = wave amplitude ({amps}). "
        "Errorbars: standard deviation across runs."
    )
    _caption = resolve_caption(plotting, _default_caption, _caption_slots,
                               fn_name="plot_damping_scatter")

    if show_plot:
        for panel in panel_conditions:
            fig = _make_damping_scatter_fig(stats_df, panel, figsize=figsize)
            plt.show()

    if save_plot:
        subfig_filenames = []
        meta_base = build_fig_meta(
            {**plotvariables, "plotting": {**plotting, "caption": _caption}},
            chapter=chapter,
            extra={"script": "plotter.py::plot_damping_scatter"},
        )
        figure_name     = plotting.get("figure_name") or build_filename("damping_scatter", meta_base)
        subfig_captions = []
        for panel in panel_conditions:
            fig_s = _make_damping_scatter_fig(stats_df, panel, figsize=figsize)
            fname = f"{figure_name}_{panel}"
            _save_figure(fig_s, fname, save_pgf=True)
            subfig_filenames.append(fname)
            subfig_captions.append(f"{panel.capitalize()} panel")
            plt.close(fig_s)

        stub_meta = {**meta_base, "panel": panel_conditions, "wind": "allwind"}
        write_figure_stub(stub_meta, "damping_scatter", subfig_filenames=subfig_filenames,
                          subfig_captions=subfig_captions,
                          force=plotting.get("force_stub", False))


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

    # ── caption ──────────────────────────────────────────────────────────────
    _top_caption = plotvariables.get("caption")
    if isinstance(_top_caption, str) and "caption" not in plotting:
        plotting = {**plotting, "caption": _top_caption}
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


def _build_sw_fits(combined_meta_sel: pd.DataFrame, positions: list[str]) -> dict:
    """
    Fit per-(date, probe) linear polynomials to nowave stillwater levels
    for nowind and fullwind conditions separately.

    Returns dict keyed by (date, pos), value is dict with keys "no" and "full",
    each a callable f(mtime_seconds) -> predicted raw probe stillwater [mm],
    or None if fewer than 1 data point available.

    x-axis: file modification time in seconds (os.path.getmtime).
    Fits are linear (deg=1) when ≥2 points, constant when exactly 1 point.
    """
    import os
    fits: dict = {}
    dates = combined_meta_sel[GC.FILE_DATE].dropna().unique()

    for date in dates:
        day_mask = combined_meta_sel[GC.FILE_DATE] == date
        day_rows  = combined_meta_sel[day_mask]
        nowave_mask = day_rows[GC.WAVE_FREQUENCY_INPUT].isna()

        for pos in positions:
            sw_col = PC.STILLWATER.format(i=pos)
            result: dict = {}

            for wind_cond in ("no", "full"):
                if sw_col not in day_rows.columns:
                    result[wind_cond] = None
                    continue
                sel = day_rows[
                    nowave_mask &
                    (day_rows[GC.WIND_CONDITION] == wind_cond) &
                    day_rows[sw_col].notna()
                ]
                if len(sel) == 0:
                    result[wind_cond] = None
                    continue

                mtimes   = np.array([os.path.getmtime(p) for p in sel[GC.PATH]])
                sw_vals  = sel[sw_col].values.astype(float)
                valid    = np.isfinite(sw_vals)
                mtimes   = mtimes[valid]
                sw_vals  = sw_vals[valid]

                if len(mtimes) == 0:
                    result[wind_cond] = None
                elif len(mtimes) == 1:
                    v = sw_vals[0]
                    result[wind_cond] = lambda t, _v=v: _v
                else:
                    t0 = mtimes.mean()
                    coeffs = np.polyfit(mtimes - t0, sw_vals, deg=1)
                    result[wind_cond] = lambda t, _c=coeffs, _t0=t0: np.polyval(_c, t - _t0)

            fits[(date, pos)] = result

    return fits


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
    import os

    # Collect all probe positions present in any df
    all_positions: list[str] = list({
        c[len("eta_"):] for df in dfs.values() for c in df.columns
        if c.startswith("eta_") and not c.endswith("_interp")
    })

    # Pre-compute nowind + fullwind stillwater regression curves per (date, probe)
    sw_fits = _build_sw_fits(combined_meta_sel, all_positions)

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

            # Find first SUSTAINED motion using rolling RMS over half a wave period.
            # A single sample exceeding threshold is almost always probe oscillation
            # or electrical noise — not the wave front. Requiring a half-period window
            # of elevated RMS filters these out while still detecting the true wave
            # arrival within ~half a period of accuracy.
            freq_row = meta_row.get(GC.WAVE_FREQUENCY_INPUT, np.nan)
            if pd.notna(freq_row) and float(freq_row) > 0:
                _half_period = max(int(MEASUREMENT.SAMPLING_RATE / (2.0 * float(freq_row))), 10)
            else:
                _half_period = 50  # fallback: ~0.2 s
            _sig_filled = np.where(np.isnan(signal), 0.0, signal)
            _rms_kernel = np.ones(_half_period) / _half_period
            _rolling_rms = np.sqrt(np.convolve(_sig_filled ** 2, _rms_kernel, mode="same"))
            exceeded = np.where(_rolling_rms > threshold)[0]
            first_motion = int(exceeded[0]) if len(exceeded) > 0 else good_start

            _sw = meta_row.get(PC.STILLWATER.format(i=pos), np.nan)
            if pd.notna(_sw):
                baseline_mean_val = float(_sw)
            else:
                # Stillwater column missing or NaN — derive from raw pre-wave window.
                # Must use raw (probe distance units), NOT eta_ (already zeroed → ~0),
                # because raw_display = -(raw - baseline_mean) needs raw units here.
                raw_base = raw[:n_base] if n_base > 10 else raw[:50]
                baseline_mean_val = float(np.nanmedian(raw_base)) if len(raw_base) > 0 else float(np.nanmedian(raw))

            # Stillwater references from pre-computed regression curves (time-aware)
            file_date   = meta_row.get(GC.FILE_DATE, "")
            wind_cond   = meta_row.get(GC.WIND_CONDITION, "no")
            run_mtime   = os.path.getmtime(path)
            fit         = sw_fits.get((file_date, pos), {})

            nowind_fn   = fit.get("no")
            fullwind_fn = fit.get("full")

            sw_nowind_ref   = float(nowind_fn(run_mtime))   if nowind_fn   else np.nan
            sw_fullwind_ref = float(fullwind_fn(run_mtime)) if fullwind_fn else np.nan

            # Wind setup in probe-distance units: positive = water level dropped
            # (fullwind raises raw probe reading → larger probe distance → lower water)
            sw_wind_setup = sw_fullwind_ref - sw_nowind_ref

            # Delta vs the curve appropriate for this run's wind condition
            sw_expected = sw_fullwind_ref if wind_cond == "full" else sw_nowind_ref
            sw_delta = baseline_mean_val - sw_expected  # probe units; positive = water lower than fit

            t0 = df["Date"].iat[0]
            time_ms = (
                df["Date"] - t0
            ).dt.total_seconds().to_numpy() * MEASUREMENT.M_TO_MM

            import re as _re
            _mstop_m = _re.search(r"mstop(\d+)", Path(path).name)
            _mstop_s = int(_mstop_m.group(1)) if _mstop_m else None

            def _get(col):
                v = meta_row.get(col, np.nan)
                return float(v) if v is not None and not pd.isna(v) else np.nan

            records.append(
                {
                    GC.PATH: path,
                    "experiment": Path(path).stem,
                    "probe": pos,
                    "data_col": col_raw,
                    GC.WIND_CONDITION: meta_row.get(GC.WIND_CONDITION, "unknown"),
                    GC.PANEL_CONDITION: meta_row.get(GC.PANEL_CONDITION, "unknown"),
                    GC.WAVE_FREQUENCY_INPUT: meta_row.get(GC.WAVE_FREQUENCY_INPUT, np.nan),
                    GC.WAVE_AMPLITUDE_INPUT: meta_row.get(GC.WAVE_AMPLITUDE_INPUT, np.nan),
                    # Wave physics for this probe — displayed in browser info panel
                    "ka_fft":         _get(f"Probe {pos} ka (FFT)"),
                    "period_fft":     _get(f"Probe {pos} WavePeriod (FFT)"),
                    "wavelength_fft": _get(f"Probe {pos} Wavelength (FFT)"),
                    "amp_fft":        _get(f"Probe {pos} Amplitude (FFT)"),
                    "amp_td":         _get(f"Probe {pos} Amplitude"),
                    "sw_nowind_ref":   sw_nowind_ref,
                    "sw_fullwind_ref": sw_fullwind_ref,
                    "sw_wind_setup":   sw_wind_setup,
                    "sw_delta":        sw_delta,
                    # Inter-run timing — for settle-time display in browser
                    "inter_run_gap_s":   _get("inter_run_gap_s"),
                    "prev_run_category": str(meta_row.get("prev_run_category") or ""),
                    "prev_run_wind":     str(meta_row.get("prev_run_wind") or ""),
                    "mstop_s":           _mstop_s,
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
    wave_info: Optional[dict] = None,
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

    # Transform raw to eta coordinate: -(raw - baseline_mean)
    # baseline_mean is the per-run stillwater in raw probe units (~100 mm absolute).
    # eta = -(probe - SW), so raw_as_eta = -(raw - baseline_mean).
    # This centers the raw signal at 0, same axis as cleaned eta — dropouts visible as jumps.
    raw_display = -(np.asarray(raw, dtype=float) - baseline_mean)
    ax.plot(time_ms, raw_display, color="lightgray", alpha=0.6, label="Raw (η-centered)")
    if signal_interp is not None:
        ax.plot(time_ms, signal_interp, color="steelblue", lw=1.2, alpha=0.7, label="Cleaned (interp)")
    ax.plot(time_ms, signal, color="black", lw=1.5, alpha=0.8, label="Cleaned (gaps=NaN)")
    if expected_sine is not None:
        ax.plot(time_ms, expected_sine, color="darkorange", lw=1.5, alpha=0.8,
                linestyle="--", label="Expected sine (FFT-fit)")
    # Reference lines in eta coordinate (centered at 0)
    ax.axhline(0.0, color="blue", linestyle="--", label="Stillwater (0 mm)")
    ax.axhline(+threshold, color="red", linestyle=":", alpha=0.7)
    ax.axhline(-threshold, color="red", linestyle=":", alpha=0.7)
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

    amp_in = meta_sel.get(GC.WAVE_AMPLITUDE_INPUT, 0)
    try:
        amp_in = float(amp_in)
    except (TypeError, ValueError):
        amp_in = 0.0
    if not np.isfinite(amp_in):
        amp_in = 0.0
    zoom = max(amp_in * 100, 15)
    ax.set_ylim(-zoom, zoom)  # eta coordinate: centered at 0
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Water level [mm]")
    ax.grid(True, alpha=0.1)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.7)

    if wave_info:
        def _wi(v, unit="", fmt=".3f"):
            try:
                return f"{float(v):{fmt}} {unit}".strip() if np.isfinite(float(v)) else "—"
            except (TypeError, ValueError):
                return "—"
        units = {"ka": "", "T": "s", "λ": "m", "A(FFT)": "mm", "A(TD)": "mm"}
        wave_parts = [f"{k} = {_wi(wave_info.get(k), units.get(k, ''))}"
                      for k in ["ka", "T", "λ", "A(FFT)", "A(TD)"]]
        wind_cond  = wave_info.get("wind_cond", "no")
        ref_label  = "fullwind fit" if wind_cond == "full" else "nowind fit"
        sw_d  = wave_info.get("sw_delta")
        sw_ws = wave_info.get("sw_wind_setup")
        try:
            sw_str = f"ΔSW = {float(sw_d):+.2f} mm vs {ref_label}" if np.isfinite(float(sw_d)) else f"ΔSW = — (no {ref_label} available)"
        except (TypeError, ValueError):
            sw_str = f"ΔSW = — (no {ref_label} available)"
        try:
            ws_str = f"wind setup = {float(sw_ws):+.2f} mm probe-dist (↑ = WL dropped)" if np.isfinite(float(sw_ws)) else "wind setup = —"
        except (TypeError, ValueError):
            ws_str = "wind setup = —"
        ax.text(
            0.01, 0.04, "   ".join(wave_parts) + "\n" + sw_str + "   |   " + ws_str,
            transform=ax.transAxes,
            fontsize=9, family="monospace",
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#aaaaaa", alpha=0.85),
        )

    return fig, ax


# ═══════════════════════════════════════════════════════════════════════════════
# PROBE NOISE FLOOR
# ═══════════════════════════════════════════════════════════════════════════════


def plot_probe_noise_floor(
    combined_meta: pd.DataFrame,
    probe_positions: list[str],
    plotvariables: dict,
    *,
    processed_dfs: dict | None = None,
    group_by: list[str] | None = None,
    highlight_keyword: str | None = "wavemakeroff-1hour",
    exclude_keywords: tuple[str, ...] = ("nestenstille",),
    k_sigma: float = 3.0,
    k_q: float = 2.0,
    probe_number_map: dict[str, int] | None = None,
    chapter: str = "04",
) -> tuple[list[plt.Figure], pd.DataFrame]:
    """
    Compute and plot the ultrasound probe noise floor from stillwater runs.

    Metrics are computed from ALL accepted stillwater runs (no single "gold
    standard" run required).  When group_by is set, stats are computed and
    plotted separately per hardware configuration group — because the noise
    floor depends on probe height / range mode and there is no single gold
    standard that covers all configurations.

    Answers three questions per (probe, config group):
      1. Precision  — noise_95pct_amp_mm, noise_rms_mm
      2. Bias       — bias_vs_ref_mm (relative to cross-probe mean within group)
      3. Threshold  — detection_threshold_mm = max(k_sigma·σ, k_q·q)

    All metrics work on deviations from the run mean — valid at any probe
    height (100 mm, 136 mm, 272 mm, …) without zeroing.

    Data sources (all from combined_meta unless noted):
      mean_level_mm        "Stillwater Probe {pos}"  — per-run mean level
      noise_rms_mm         "Probe {pos} Stillwater Std"  — std of raw signal
      noise_95pct_amp_mm   "Probe {pos} Amplitude"  — (P97.5−P2.5)/2
      bias_vs_ref_mm       mean_level_mm − mean(all probes, same group)
      quantization_step_mm P5 of nonzero |diff(eta_{pos})|  — needs processed_dfs
      detection_threshold_mm  max(k_sigma·rms, k_q·q)

    Parameters
    ----------
    combined_meta     : full metadata DataFrame (all runs)
    probe_positions   : list of position strings, e.g. ["8804/250", "9373/170"]
    plotvariables     : dict with "filters" (unused) and "plotting" sub-dict
    processed_dfs     : optional {path: DataFrame} for quantization_step_mm.
                        When None, quantization_step_mm is NaN and threshold
                        uses k_sigma·σ only.
    group_by          : list of combined_meta column names to facet by,
                        e.g. ["probe_height_mm", "probe_range_mode"].
                        When None a single subplot for all runs is drawn.
    highlight_keyword : filename substring — matching runs are marked with a
                        gold star on the plot (visual only, not used for stats).
                        Pass None to disable.
    exclude_keywords  : filenames with any of these substrings are shown as
                        excluded outliers and omitted from statistics.
    k_sigma           : σ multiplier for threshold  (default 3)
    k_q               : quantization-step multiplier  (default 2)
    probe_number_map  : optional {position: probe_number} for x-axis labels
    chapter           : thesis chapter string for save_and_stub

    Returns
    -------
    figs    : list of matplotlib Figures, one per group (in group sort order)
    summary : DataFrame  columns [group, probe, mean_level_mm, noise_rms_mm,
                noise_95pct_amp_mm, noise_95pct_std_mm, n_runs,
                quantization_step_mm, bias_vs_ref_mm, detection_threshold_mm]
              When group_by is None, the "group" column is "all".
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    plotting = plotvariables.get("plotting", {})
    show_plot = plotting.get("show_plot", False)
    save_plot = plotting.get("save_plot", False)

    # ── select stillwater runs ────────────────────────────────────────────────
    is_stillwater = (
        combined_meta["WindCondition"].eq("no")
        & combined_meta["WaveFrequencyInput [Hz]"].isna()
    )
    meta_sw = combined_meta[is_stillwater].copy()
    if meta_sw.empty:
        raise ValueError("No stillwater runs (WindCondition='no', no wave freq).")
    meta_sw["_run"] = meta_sw["path"].apply(lambda p: Path(str(p)).name)

    # ── probes that actually have amplitude columns ───────────────────────────
    probe_cols_present = [p for p in probe_positions
                          if f"Probe {p} Amplitude" in meta_sw.columns]
    if not probe_cols_present:
        raise ValueError(
            "No 'Probe {pos} Amplitude' columns in combined_meta stillwater rows. "
            "Run the pipeline (main.py) first."
        )

    # ── flat per-run table ────────────────────────────────────────────────────
    _grp_cols = [c for c in (group_by or []) if c in meta_sw.columns]
    rows = []
    for _, row in meta_sw.iterrows():
        entry = {"run": row["_run"], "path": row["path"]}
        for gc in _grp_cols:
            entry[gc] = row[gc]
        for pos in probe_cols_present:
            entry[f"amp_{pos}"]   = row.get(f"Probe {pos} Amplitude",      np.nan)
            entry[f"level_{pos}"] = row.get(f"Stillwater Probe {pos}",     np.nan)
            entry[f"rms_{pos}"]   = row.get(f"Probe {pos} Stillwater Std", np.nan)
        rows.append(entry)
    sw_all = pd.DataFrame(rows)

    # ── exclude bad runs ──────────────────────────────────────────────────────
    is_excluded = sw_all["run"].apply(
        lambda r: any(kw in r for kw in exclude_keywords)
    )
    is_highlight = (
        sw_all["run"].apply(lambda r: highlight_keyword in r)
        if highlight_keyword else pd.Series(False, index=sw_all.index)
    )
    sw_accepted = sw_all[~is_excluded].copy()
    is_hl_acc   = is_highlight[~is_excluded]

    # ── define groups ─────────────────────────────────────────────────────────
    if _grp_cols:
        def _grp_label(r):
            parts = []
            for c in _grp_cols:
                v = r[c]
                parts.append(f"h{int(v)}" if c == "probe_height_mm" else str(v))
            return " / ".join(parts)
        sw_accepted["_group"] = sw_accepted.apply(_grp_label, axis=1)
        # sort: height descending, then range mode alphabetically
        def _sort_key(lbl):
            parts = lbl.split(" / ")
            h = int(parts[0].lstrip("h")) if parts[0].startswith("h") else 0
            return (-h, parts[1] if len(parts) > 1 else "")
        groups = sorted(sw_accepted["_group"].unique(), key=_sort_key)
    else:
        sw_accepted["_group"] = "all"
        groups = ["all"]

    # ── quantization step: one estimate per probe (from first available run) ──
    quant_steps: dict[str, float] = {}
    if processed_dfs is not None:
        for path in sw_accepted["path"]:
            df_run = processed_dfs.get(path)
            if df_run is None:
                continue
            for pos in probe_cols_present:
                if pos in quant_steps:
                    continue
                src = (f"eta_{pos}_interp" if f"eta_{pos}_interp" in df_run.columns
                       else f"eta_{pos}")
                if src not in df_run.columns:
                    continue
                sig = np.asarray(df_run[src].dropna(), dtype=float)
                if len(sig) < 2:
                    continue
                diffs = np.abs(np.diff(sig))
                nz = diffs[diffs > 0]
                if len(nz):
                    quant_steps[pos] = float(np.percentile(nz, 5))
            if len(quant_steps) == len(probe_cols_present):
                break   # all probes covered

    # ── per-group × per-probe summary ────────────────────────────────────────
    records = []
    for grp in groups:
        sub = sw_accepted[sw_accepted["_group"] == grp]
        lev_vals = {}
        for pos in probe_cols_present:
            amp_vals = sub[f"amp_{pos}"].dropna()
            rms_vals = sub[f"rms_{pos}"].dropna()
            lev_vals_ = sub[f"level_{pos}"].dropna()
            quant    = quant_steps.get(pos, np.nan)

            mean_amp  = float(amp_vals.mean())  if len(amp_vals)  else np.nan
            mean_rms  = float(rms_vals.mean())  if len(rms_vals)  else np.nan
            mean_lev  = float(lev_vals_.mean()) if len(lev_vals_) else np.nan
            lev_vals[pos] = mean_lev

            thr_sigma = k_sigma * mean_rms if np.isfinite(mean_rms) else np.nan
            thr_quant = k_q     * quant    if np.isfinite(quant)    else np.nan
            candidates = [v for v in (thr_sigma, thr_quant) if np.isfinite(v)]
            det_thr   = float(max(candidates)) if candidates else np.nan

            records.append({
                "group":                  grp,
                "probe":                  pos,
                "mean_level_mm":          mean_lev,
                "noise_rms_mm":           mean_rms,
                "noise_95pct_amp_mm":     mean_amp,
                "noise_95pct_std_mm":     float(amp_vals.std()) if len(amp_vals) > 1 else np.nan,
                "n_runs":                 len(amp_vals),
                "quantization_step_mm":   quant,
                "detection_threshold_mm": det_thr,
            })
        # bias relative to cross-probe mean level within this group
        _lvs = [v for v in lev_vals.values() if np.isfinite(v)]
        ref = float(np.mean(_lvs)) if _lvs else np.nan
        for rec in records[-len(probe_cols_present):]:
            lv = lev_vals.get(rec["probe"], np.nan)
            rec["bias_vs_ref_mm"] = (lv - ref) if np.isfinite(lv) and np.isfinite(ref) else np.nan

    summary = pd.DataFrame(records)

    # ── plot: one standalone figure per group ────────────────────────────────────
    apply_thesis_style()
    n_probes = len(probe_cols_present)
    bar_w    = 0.55
    x = np.arange(n_probes)
    sw_excl = sw_all[is_excluded]
    _has_excl = not sw_excl.empty
    _has_hl   = bool(is_hl_acc.any())
    _has_quant_any = bool(quant_steps)

    _leg_handles = [
        Patch(facecolor="steelblue", alpha=0.75),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor="navy", markersize=7, linewidth=0),
        Line2D([0], [0], color="crimson", linewidth=1.8, linestyle="--"),
    ]
    _leg_labels = [
        f"Mean 95% noise amp.  (±1σ)",
        "Per-run value",
        f"Threshold  max({k_sigma:.0f}σ,  {k_q:.0f}q)  [mm]",
    ]
    if _has_quant_any:
        _leg_handles.append(Line2D([0], [0], color="dimgrey", linewidth=1.0, linestyle=":"))
        _leg_labels.append("Quantization half-step  q/2  [mm]")
    if _has_hl:
        _leg_handles.append(Line2D([0], [0], marker="*", color="w",
                                   markerfacecolor="gold", markeredgecolor="darkorange",
                                   markersize=11, linewidth=0))
        _leg_labels.append(f"Highlighted run  ({highlight_keyword})")
    if _has_excl:
        _leg_handles.append(Line2D([0], [0], marker="x", color="dimgrey",
                                   markersize=8, linewidth=0, markeredgewidth=1.3))
        _leg_labels.append("Excluded (not settled)")

    figs = []
    for grp in groups:
        sub      = sw_accepted[sw_accepted["_group"] == grp]
        sub_hl   = sub[is_hl_acc[sub.index].values]
        sub_norm = sub[~is_hl_acc[sub.index].values]
        grp_sum  = summary[summary["group"] == grp].set_index("probe")

        fig, ax = plt.subplots(1, 1, figsize=(max(4, n_probes * 1.3 + 1.5), 4.2))

        means_bar = np.array([grp_sum.loc[p, "noise_95pct_amp_mm"]  if p in grp_sum.index else np.nan
                               for p in probe_cols_present], dtype=float)
        stds_bar  = np.array([grp_sum.loc[p, "noise_95pct_std_mm"]  if p in grp_sum.index else np.nan
                               for p in probe_cols_present], dtype=float)
        stds_bar  = np.where(np.isfinite(stds_bar), stds_bar, 0.0)

        ax.bar(x, means_bar, width=bar_w, yerr=stds_bar, capsize=4,
               color="steelblue", alpha=0.75,
               error_kw={"elinewidth": 1.2, "ecolor": "navy"})

        for xi, pos in enumerate(probe_cols_present):
            thr = grp_sum.loc[pos, "detection_threshold_mm"] if pos in grp_sum.index else np.nan
            if np.isfinite(thr):
                ax.hlines(thr, xi - bar_w * 0.46, xi + bar_w * 0.46,
                          colors="crimson", linewidths=1.8, linestyles="--", zorder=4)
            q = grp_sum.loc[pos, "quantization_step_mm"] if pos in grp_sum.index else np.nan
            if np.isfinite(q):
                ax.hlines(q / 2, xi - bar_w * 0.38, xi + bar_w * 0.38,
                          colors="dimgrey", linewidths=1.0, linestyles=":", zorder=3)

        for pi, pos in enumerate(probe_cols_present):
            vals = sub_norm[f"amp_{pos}"].dropna().values
            if len(vals):
                ax.scatter(np.full(len(vals), x[pi]), vals,
                           s=20, color="white", edgecolors="navy",
                           linewidths=0.8, zorder=5)
            vals_hl = sub_hl[f"amp_{pos}"].dropna().values
            if len(vals_hl):
                ax.scatter(np.full(len(vals_hl), x[pi]), vals_hl,
                           s=80, marker="*", color="gold", edgecolors="darkorange",
                           linewidths=0.9, zorder=6)
            vals_ex = sw_excl[f"amp_{pos}"].dropna().values if f"amp_{pos}" in sw_excl.columns else []
            if len(vals_ex):
                ax.scatter(np.full(len(vals_ex), x[pi]), vals_ex,
                           s=35, marker="x", color="dimgrey",
                           linewidths=1.3, zorder=4)

        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(x)
        if probe_number_map:
            xlabels = [f"#{probe_number_map[p]}\n{p}" if p in probe_number_map else p
                       for p in probe_cols_present]
        else:
            xlabels = probe_cols_present
        ax.set_xticklabels(xlabels, rotation=0, ha="center", fontsize=9)
        n_grp = len(sub)
        ax.set_title(grp if grp != "all" else f"all configs  (n={n_grp} runs)", fontsize=10)
        ax.set_ylabel("Stillwater 95% noise amplitude  [mm]")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(_leg_handles, _leg_labels, fontsize=8, loc="upper right", framealpha=0.9)

        plt.tight_layout()
        if show_plot:
            plt.show()
        figs.append(fig)

    # ── caption ──────────────────────────────────────────────────────────────
    _quant_vals = sorted({round(v, 3) for v in quant_steps.values() if np.isfinite(v)})
    if _quant_vals:
        if len(_quant_vals) == 1:
            _quant_str = (
                f"Quantization step $q = {_quant_vals[0]:.2f}$\\,mm "
                "(P5 of nonzero sample-to-sample differences); "
            )
        else:
            _quant_str = (
                f"Quantization step $q = {_quant_vals[0]:.2f}$--${_quant_vals[-1]:.2f}$\\,mm "
                "per probe (P5 of nonzero sample-to-sample differences); "
            )
    else:
        _quant_str = ""
    _caption_slots = {
        "n_accepted": len(sw_accepted),
        "n_excluded": int(is_excluded.sum()),
        "k_sigma":    k_sigma,
        "k_q":        k_q,
    }
    _default_caption = (
        "Stillwater 95\\% noise amplitude $(P_{{97.5}} - P_{{2.5}})/2$ per "
        "ultrasound wave gauge, with no waves and no wind. "
        "Each panel shows one hardware configuration "
        "(probe height above still water / range mode). "
        "Blue bars: mean across accepted stillwater runs within each configuration "
        "(error bars: ±1\\,std). "
        "White dots: individual run values. "
        + (_quant_str)
        + "Dashed red line: detection threshold "
        "$\\max({k_sigma:.0f}\\,\\sigma,\\; {k_q:.0f}\\,q)$ "
        "per probe, where $\\sigma$ is the rms noise; "
        "wave amplitudes below this line are indistinguishable from stillwater noise."
    )
    _caption = resolve_caption(
        plotting, _default_caption, _caption_slots,
        fn_name="plot_probe_noise_floor",
    )

    if save_plot:
        meta_fig = build_fig_meta(
            {**plotvariables, "plotting": {**plotting, "caption": _caption}},
            chapter=chapter,
        )
        save_and_stub(figs[0], meta_fig, plot_type="probe_noise_floor")

    return figs, summary


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
    # Allow caption at plotvariables top level (string) as a convenience
    _top_caption = plotvariables.get("caption")
    if isinstance(_top_caption, str) and "caption" not in plotting:
        plotting = {**plotting, "caption": _top_caption}

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


# ═══════════════════════════════════════════════════════════════════════════════
# WAVE STABILITY
# ═══════════════════════════════════════════════════════════════════════════════


def plot_wave_stability(
    combined_meta: pd.DataFrame,
    probe_positions: list,
    plotvariables: dict,
    chapter: str = "04",
    stability_threshold: float = 0.85,
) -> plt.Figure:
    """
    Wave-train stability (autocorrelation at lag-1-period) vs frequency.

    One subplot per probe, coloured by wind condition. A threshold line at
    *stability_threshold* marks the boundary below which the IN probe signal
    is dominated by wind-wave noise rather than the paddle wave. This motivates
    using FFT amplitude (not time-domain) for OUT/IN under full wind.

    Parameters
    ----------
    combined_meta : DataFrame (all runs)
    probe_positions : list of "dist/lat" position strings
    plotvariables : dict with "filters" and "plotting" sub-dicts.
        Recognised "plotting" keys:
            show_plot          : bool  (default False)
            save_plot          : bool  (default False)
            figure_name        : str
            force_stub         : bool  (default False)
            figsize            : tuple (default auto)
            caption            : str  (optional template with {n_runs},
                                       {threshold}, {n_probes})
    chapter : str
    stability_threshold : float
        Horizontal reference line drawn on each subplot (default 0.85).
    """
    from wavescripts.filters import apply_experimental_filters

    plotting = plotvariables.get("plotting", {})
    show_plot  = plotting.get("show_plot",  False)
    save_plot  = plotting.get("save_plot",  False)
    figure_name = plotting.get("figure_name", "ch04_wave_stability")
    force_stub  = plotting.get("force_stub",  False)

    _top_caption = plotvariables.get("caption")
    if isinstance(_top_caption, str) and "caption" not in plotting:
        plotting = {**plotting, "caption": _top_caption}

    # ── Filter and build tidy data frame ─────────────────────────────────────
    sel = apply_experimental_filters(combined_meta, plotvariables)
    # keep only wave runs with a measured frequency
    sel = sel[sel["WaveFrequencyInput [Hz]"].notna()].copy()

    rows = []
    for _, row in sel.iterrows():
        for pos in probe_positions:
            col    = f"Probe {pos} wave_stability"
            cv_col = f"Probe {pos} period_amplitude_cv"
            if col not in sel.columns:
                continue
            val = row.get(col)
            if pd.isna(val):
                continue
            rows.append({
                "probe":          pos,
                "freq":           row["WaveFrequencyInput [Hz]"],
                "wind":           row.get("WindCondition"),
                "amplitude":      row.get("WaveAmplitudeInput [Volt]"),
                "wave_stability": float(val),
                "period_cv":      float(row.get(cv_col, np.nan)),
            })

    stab_df = pd.DataFrame(rows)

    n_runs   = len(sel)
    n_probes = len(probe_positions)

    _caption_slots = {
        "n_runs":    n_runs,
        "threshold": stability_threshold,
        "n_probes":  n_probes,
    }
    _default_caption = (
        "Wave-train stability (autocorrelation at lag-1-period) versus wave "
        "frequency for {n_probes} probes, {n_runs} wave runs. "
        "Colour encodes wind condition. "
        "Dashed line: quality threshold {threshold} — "
        "runs below this are dominated by wind-wave noise at the IN probe."
    )
    _caption = resolve_caption(
        plotting, _default_caption, _caption_slots,
        fn_name="plot_wave_stability",
    )

    # ── Draw figure ───────────────────────────────────────────────────────────
    n_cols  = len(probe_positions)
    figsize = plotting.get("figsize", (3.2 * n_cols, 3.5))
    apply_thesis_style()
    fig, axes = plt.subplots(1, n_cols, figsize=figsize, sharey=True)
    if n_cols == 1:
        axes = [axes]

    for ax, pos in zip(axes, probe_positions):
        sub = stab_df[stab_df["probe"] == pos]
        for wind, grp in sub.groupby("wind"):
            agg = (grp.groupby("freq")["wave_stability"]
                      .agg(mean="mean", std="std")
                      .reset_index())
            ax.errorbar(
                agg["freq"], agg["mean"], yerr=agg["std"],
                label=wind,
                color=WIND_COLOR_MAP.get(wind, "gray"),
                marker="o", markersize=5, linewidth=1.2, capsize=3,
            )
        ax.axhline(
            stability_threshold, color="gray",
            linestyle="--", linewidth=0.8, alpha=0.7,
            label=f"threshold {stability_threshold}",
        )
        ax.set_title(pos, fontsize=9)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylim(0, 1.05)

    axes[0].set_ylabel("wave_stability")
    axes[-1].legend(title="wind", fontsize=8, loc="lower left")
    fig.suptitle("Wavetrain stability vs frequency", fontsize=10)
    # Fixed margins — content-independent so all subfigures align in LaTeX
    fig.subplots_adjust(left=0.10, right=0.97, top=0.88, bottom=0.14, wspace=0.08)

    if show_plot:
        plt.show()

    if save_plot:
        FIGURES_DIR = Path("output/FIGURES")
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fname = figure_name
        fig.savefig(FIGURES_DIR / f"{fname}.pdf")
        fig.savefig(FIGURES_DIR / f"{fname}.pgf")
        print(f"  Saved: output/FIGURES/{fname}.pdf")
        meta = build_fig_meta(
            {**plotvariables, "plotting": {**plotting,
                                           "figure_name": figure_name,
                                           "caption": _caption}},
            chapter=chapter,
        )
        write_figure_stub(
            meta, plot_type="wave_stability",
            subfig_filenames=[fname],
            force=force_stub,
        )

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# TIME-SERIES OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════


def plot_timeseries_overview(
    combined_meta: pd.DataFrame,
    processed_dfs: dict,
    plotvariables: dict,
    chapter: str = "04",
) -> plt.Figure:
    """
    Static time-series grid for selected runs — CH04 §5.

    Layout: rows = probes (from plotvariables["plotting"]["probes"]),
            columns = runs selected by plotvariables["filters"].

    Each cell shows eta_{pos} [mm] vs time [s] with:
      - light-grey band for the detected stable-window (good_start → good_end)
      - column title = short run description (wind condition, freq, amplitude)

    Parameters
    ----------
    combined_meta : DataFrame (all runs)
    processed_dfs : dict {path: DataFrame} with eta_{pos} columns
    plotvariables : dict with "filters" and "plotting" sub-dicts.
        Recognised "plotting" keys:
            show_plot      : bool  (default False)
            save_plot      : bool  (default False)
            figure_name    : str
            force_stub     : bool  (default False)
            probes         : list[str]  — which probes to show as rows
            max_runs       : int   — cap on columns shown (default 4)
            xlim           : tuple (t_start, t_end) [s], or None for full run
            ylim           : tuple (y_lo, y_hi) [mm], or None for auto (shared)
            figsize        : tuple  (auto if omitted)
            caption        : str  (optional template with {n_runs},
                                   {freq_hz}, {wind_conds}, {probe_list})
    chapter : str
    """
    from wavescripts.filters import apply_experimental_filters
    from wavescripts.constants import MEASUREMENT, PC

    fs = MEASUREMENT.SAMPLING_RATE

    plotting     = plotvariables.get("plotting", {})
    show_plot    = plotting.get("show_plot",  False)
    save_plot    = plotting.get("save_plot",  False)
    figure_name  = plotting.get("figure_name", "ch04_timeseries")
    force_stub   = plotting.get("force_stub",  False)
    probe_positions = plotting.get("probes", [])
    max_runs     = plotting.get("max_runs", 4)
    xlim         = plotting.get("xlim",  None)
    ylim         = plotting.get("ylim",  None)

    _top_caption = plotvariables.get("caption")
    if isinstance(_top_caption, str) and "caption" not in plotting:
        plotting = {**plotting, "caption": _top_caption}

    # ── Select runs ───────────────────────────────────────────────────────────
    sel = apply_experimental_filters(combined_meta, plotvariables)
    sel = sel[sel["WaveFrequencyInput [Hz]"].notna()].copy()
    if len(sel) > max_runs:
        sel = sel.iloc[:max_runs]

    n_runs   = len(sel)
    n_probes = len(probe_positions)

    if n_runs == 0 or n_probes == 0:
        raise ValueError("plot_timeseries_overview: no runs or no probes selected.")

    # ── Caption slots ─────────────────────────────────────────────────────────
    freq_vals  = sorted(sel["WaveFrequencyInput [Hz]"].dropna().unique())
    wind_vals  = sorted(sel["WindCondition"].dropna().unique())
    _caption_slots = {
        "n_runs":      n_runs,
        "freq_hz":     ", ".join(f"{f:.2f}" for f in freq_vals),
        "wind_conds":  ", ".join(wind_vals),
        "probe_list":  ", ".join(probe_positions),
    }
    _default_caption = (
        "Time series of free-surface elevation at {n_probes} probes "
        "for {n_runs} selected runs "
        "(f = {freq_hz}\\,Hz; wind: {wind_conds}). "
        "Grey shading marks the detected stable-wave window used for amplitude "
        "and FFT analysis."
    )
    _default_caption = _default_caption.replace("{n_probes}", str(n_probes))
    _caption = resolve_caption(
        plotting, _default_caption, _caption_slots,
        fn_name="plot_timeseries_overview",
    )

    # ── Build figure ──────────────────────────────────────────────────────────
    figsize = plotting.get("figsize", (3.5 * n_runs, 2.8 * n_probes))
    apply_thesis_style()
    fig, axes = plt.subplots(
        n_probes, n_runs,
        figsize=figsize,
        sharey="row" if ylim is None else False,
        sharex=False,
        squeeze=False,
    )

    for col_i, (_, run_row) in enumerate(sel.iterrows()):
        path = run_row["path"]
        df   = processed_dfs.get(path)
        freq  = run_row.get("WaveFrequencyInput [Hz]")
        wind  = run_row.get("WindCondition", "?")
        amp   = run_row.get("WaveAmplitudeInput [Volt]")
        col_title = f"{freq:.2f} Hz  {wind}  {amp:.2f} V"

        for row_i, pos in enumerate(probe_positions):
            ax = axes[row_i][col_i]

            if df is None:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=8, color="gray")
                ax.set_title(col_title if row_i == 0 else "", fontsize=8)
                continue

            eta_col    = f"eta_{pos}"
            interp_col = f"eta_{pos}_interp"
            sig_col    = eta_col if interp_col not in df.columns else interp_col

            if sig_col not in df.columns:
                ax.text(0.5, 0.5, f"no {eta_col}", transform=ax.transAxes,
                        ha="center", va="center", fontsize=8, color="gray")
                continue

            # Build time axis
            t = np.arange(len(df)) / fs

            # Stable-window bounds from meta
            start_col = PC.START.format(i=pos)
            end_col   = PC.END.format(i=pos)
            gs = run_row.get(start_col)
            ge = run_row.get(end_col)

            sig = df[sig_col].values
            ax.plot(t, sig, color=WIND_COLOR_MAP.get(wind, "#1F77B4"),
                    linewidth=0.6, alpha=0.85)

            # Shade stable window
            if pd.notna(gs) and pd.notna(ge):
                t_gs = int(gs) / fs
                t_ge = int(ge) / fs
                ax.axvspan(t_gs, t_ge, color="gray", alpha=0.12, linewidth=0)

            if xlim:
                ax.set_xlim(xlim)
            if ylim:
                ax.set_ylim(ylim)

            if row_i == 0:
                ax.set_title(col_title, fontsize=8)
            if col_i == 0:
                ax.set_ylabel(f"{pos}\n[mm]", fontsize=8)
            if row_i == n_probes - 1:
                ax.set_xlabel("Time [s]", fontsize=8)

    # Fixed margins — content-independent so all subfigures align in LaTeX
    fig.subplots_adjust(left=0.12, right=0.97, top=0.92, bottom=0.10,
                        hspace=0.35, wspace=0.15)

    if show_plot:
        plt.show()

    if save_plot:
        FIGURES_DIR = Path("output/FIGURES")
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fname = figure_name
        fig.savefig(FIGURES_DIR / f"{fname}.pdf")
        fig.savefig(FIGURES_DIR / f"{fname}.pgf")
        print(f"  Saved: output/FIGURES/{fname}.pdf")
        meta = build_fig_meta(
            {**plotvariables, "plotting": {**plotting,
                                           "figure_name": figure_name,
                                           "caption": _caption}},
            chapter=chapter,
        )
        write_figure_stub(
            meta, plot_type="timeseries_overview",
            subfig_filenames=[fname],
            force=force_stub,
        )

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# INSTRUMENT DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_sound_speed(
    combined_meta: pd.DataFrame,
    plotvariables: dict,
    chapter: str = "04",
) -> Optional[plt.Figure]:
    """
    Speed-of-sound (and inferred air temperature) per run vs date — CH04.

    Quantifies the worst-case probe amplitude scale error arising from
    lab temperature variation across sessions.

    Parameters
    ----------
    combined_meta : DataFrame — all runs; must have sound_speed_mean_ms /
                    sound_speed_std_ms columns (added by pipeline).
    plotvariables : standard dict with "filters" and "plotting" sub-dicts.
        Recognised "plotting" keys: show_plot, save_plot, draft, force_stub,
        figsize, figure_name, caption.
    """
    plotting   = plotvariables.get("plotting", {})
    show_plot  = plotting.get("show_plot",  False)
    save_plot  = plotting.get("save_plot",  False)
    force_stub = plotting.get("force_stub", False)
    figsize    = plotting.get("figsize",    (10, 3))

    _c_df = combined_meta[
        ["file_date", "experiment_folder",
         "sound_speed_mean_ms", "sound_speed_std_ms"]
    ].dropna(subset=["sound_speed_mean_ms"]).copy()
    _c_df["file_date"]    = pd.to_datetime(_c_df["file_date"])
    _c_df["T_approx_C"]   = (_c_df["sound_speed_mean_ms"] - 331.0) / 0.606
    _c_ref                = 343.0
    _c_df["scale_err_pct"] = (_c_df["sound_speed_mean_ms"] - _c_ref).abs() / _c_ref * 100

    n_runs    = len(_c_df)
    worst_pct = _c_df["scale_err_pct"].max()

    _slots = {
        "n_runs":      n_runs,
        "c_ref":       _c_ref,
        "worst_pct":   f"{worst_pct:.3f}",
        "worst_mm_10": f"{worst_pct * 0.1:.4f}",
    }
    _default = (
        "Speed of sound in air per run, measured by the probe hardware "
        "({n_runs} runs). "
        "Right axis: approximate air temperature from "
        r"$c_\mathrm{air}\approx 331 + 0.606\,T$. "
        "Dashed line: {c_ref}\\,m/s reference ($\\approx$20\\,\\textdegree C). "
        "Worst-case amplitude scale error: {worst_pct}\\,\\% "
        "({worst_mm_10}\\,mm on a 10\\,mm wave). "
        "For OUT/IN ratios the error cancels exactly."
    )
    _caption = resolve_caption(plotting, _default, _slots,
                               fn_name="plot_sound_speed")

    def _make_fig() -> plt.Figure:
        apply_thesis_style()
        fig, ax = plt.subplots(figsize=figsize)
        sc = ax.scatter(_c_df["file_date"], _c_df["sound_speed_mean_ms"],
                        c=_c_df["T_approx_C"], cmap="coolwarm", s=14, zorder=3)
        ax.errorbar(_c_df["file_date"], _c_df["sound_speed_mean_ms"],
                    yerr=_c_df["sound_speed_std_ms"],
                    fmt="none", color="gray", alpha=0.4, lw=0.8)
        ax2 = ax.twinx()
        ax2.set_ylim([(y - 331.0) / 0.606 for y in ax.get_ylim()])
        ax2.set_ylabel("Air temperature [°C]", color="gray", fontsize=9)
        ax.axhline(_c_ref, ls="--", color="k", lw=0.7,
                   label=f"c = {_c_ref} m/s  (~20 °C ref)")
        ax.set_ylabel("$c_\\mathrm{air}$ [m/s]")
        ax.set_xlabel("Date")
        ax.legend(fontsize=8)
        fig.colorbar(sc, ax=ax, label="T [°C]", fraction=0.03, pad=0.12)
        fig.autofmt_xdate()
        fig.tight_layout()
        return fig

    fig = _make_fig()
    if show_plot:
        plt.show()

    if save_plot:
        meta = build_fig_meta(
            {**plotvariables, "plotting": {**plotting, "caption": _caption}},
            chapter=chapter,
            extra={"script": "plotter.py::plot_sound_speed"},
        )
        save_and_stub(fig, meta, plot_type="sound_speed",
                      force_stub=force_stub)

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# WIND CHARACTERISATION
# ═══════════════════════════════════════════════════════════════════════════════

def plot_wind_snr(
    combined_meta: pd.DataFrame,
    combined_psd_dict: dict,
    plotvariables: dict,
    chapter: str = "04",
) -> Optional[plt.Figure]:
    """
    Spectral SNR = A_paddle_FFT / A_wind_FFT per probe vs frequency — CH04 §4.

    Wind-noise amplitude A_wind_FFT is derived by integrating the mean fullwind
    nowave PSD over the 0.1 Hz FFT window at each paddle frequency.
    SNR < 5 is unreliable; SNR < 3 is wind-dominated.

    Parameters
    ----------
    combined_meta     : DataFrame — all runs.
    combined_psd_dict : {path: DataFrame} with "Pxx {pos}" columns.
    plotvariables     : standard dict.
        Extra "plotting" keys: probes (list[str]), fft_window_hz (float, default 0.1).
    """
    from wavescripts.filters import apply_experimental_filters as _aef

    plotting       = plotvariables.get("plotting", {})
    show_plot      = plotting.get("show_plot",     False)
    save_plot      = plotting.get("save_plot",     False)
    force_stub     = plotting.get("force_stub",    False)
    probes         = plotting.get("probes",        [])
    fft_window_hz  = plotting.get("fft_window_hz", 0.1)
    figsize        = plotting.get("figsize",       None)

    # ── Baseline: mean PSD from fullwind nowave runs ──────────────────────────
    _fw_mask  = (combined_meta["WaveFrequencyInput [Hz]"].isna()
                 & (combined_meta["WindCondition"] == "full"))
    _fw_paths = set(combined_meta.loc[_fw_mask, "path"].values)
    _fw_psds  = {k: v for k, v in combined_psd_dict.items() if k in _fw_paths}

    if not _fw_psds:
        print("plot_wind_snr: no fullwind nowave PSD data — aborting.")
        return None

    _psd_sample = next(v for v in _fw_psds.values() if v is not None and len(v) > 1)
    _psd_freqs  = _psd_sample.index.values.astype(float)

    _mean_pxx: dict[str, np.ndarray] = {}
    for pos in probes:
        col   = f"Pxx {pos}"
        stack = [v[col].values for v in _fw_psds.values()
                 if v is not None and col in v.columns]
        if stack:
            _mean_pxx[pos] = np.vstack(stack).mean(axis=0)

    _paddle_freqs = sorted(combined_meta["WaveFrequencyInput [Hz]"].dropna().unique())

    _wind_amp_fft: dict[str, dict] = {}
    for pos, pxx in _mean_pxx.items():
        _wind_amp_fft[pos] = {}
        for f in _paddle_freqs:
            mask = ((_psd_freqs >= f - fft_window_hz / 2) &
                    (_psd_freqs <= f + fft_window_hz / 2))
            if mask.any():
                _wind_amp_fft[pos][f] = float(
                    np.sqrt(2.0 * np.trapezoid(pxx[mask], _psd_freqs[mask]))
                )

    # ── SNR from wave runs ────────────────────────────────────────────────────
    _wave = _aef(
        combined_meta[combined_meta["WaveFrequencyInput [Hz]"].notna()].copy(),
        plotvariables,
    )
    _snr_rows = []
    for _, row in _wave.iterrows():
        freq  = row.get("WaveFrequencyInput [Hz]")
        if not np.isfinite(float(freq)):
            continue
        f_key = min(_paddle_freqs, key=lambda f: abs(f - freq))
        for pos in probes:
            a_paddle = row.get(f"Probe {pos} Amplitude (FFT)", np.nan)
            a_wind   = _wind_amp_fft.get(pos, {}).get(f_key, np.nan)
            if (np.isfinite(float(a_paddle)) and np.isfinite(float(a_wind))
                    and float(a_wind) > 0):
                _snr_rows.append({
                    "freq": float(freq),
                    "wind": row.get("WindCondition"),
                    "probe": pos,
                    "SNR":  float(a_paddle) / float(a_wind),
                })

    snr_df     = pd.DataFrame(_snr_rows)
    n_wave_runs = _wave["path"].nunique()

    _slots = {
        "n_wave_runs": n_wave_runs,
        "n_fw_psds":   len(_fw_psds),
        "fft_window":  fft_window_hz,
    }
    _default = (
        "Spectral SNR per probe: ratio of paddle-frequency FFT amplitude "
        "to wind-noise amplitude integrated over the {fft_window}\\,Hz FFT window "
        "at each paddle frequency "
        "({n_wave_runs} wave runs; wind-noise baseline from "
        "{n_fw_psds} full-wind no-wave runs). "
        "Horizontal lines: SNR~=~10 (dotted grey), 5 (dashed black), "
        "3 (dotted red, critical). "
        "SNR~$<$~3 indicates wind dominates the FFT measurement."
    )
    _caption = resolve_caption(plotting, _default, _slots,
                               fn_name="plot_wind_snr")

    probes_in = [p for p in probes if p in snr_df.get("probe", pd.Series()).values]
    fs        = figsize or (4.5 * max(len(probes_in), 1), 4)

    def _make_fig() -> plt.Figure:
        apply_thesis_style()
        fig, axes = plt.subplots(1, len(probes_in), figsize=fs, sharey=True)
        if len(probes_in) == 1:
            axes = [axes]
        for ax, pos in zip(axes, probes_in):
            sub = snr_df[snr_df["probe"] == pos]
            for wc, grp in sub.groupby("wind"):
                agg = grp.groupby("freq")["SNR"].median()
                ax.plot(agg.index, agg.values, marker="o", lw=1.2, markersize=6,
                        color=WIND_COLOR_MAP.get(wc, "grey"), label=wc)
            ax.axhline(10, color="0.6",     lw=0.7, ls=":",  label="SNR=10")
            ax.axhline(5,  color="k",       lw=0.8, ls="--", label="SNR=5")
            ax.axhline(3,  color="tomato",  lw=0.8, ls=":",  label="SNR=3")
            ax.set_xlabel("Paddle frequency [Hz]")
            if ax is axes[0]:
                ax.set_ylabel(
                    r"Spectral SNR  ($A_\mathrm{paddle}\,/\,A_\mathrm{wind,FFT}$)"
                )
            ax.set_title(pos, fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    fig = _make_fig()
    if show_plot:
        plt.show()

    if save_plot:
        meta = build_fig_meta(
            {**plotvariables, "plotting": {**plotting, "caption": _caption}},
            chapter=chapter,
            extra={"script": "plotter.py::plot_wind_snr"},
        )
        save_and_stub(fig, meta, plot_type="wind_snr", force_stub=force_stub)

    return fig


def plot_td_vs_fft(
    combined_meta: pd.DataFrame,
    plotvariables: dict,
    chapter: str = "04",
) -> Optional[plt.Figure]:
    """
    Time-domain vs FFT amplitude — motivates FFT as the only valid OUT/IN metric.

    Row 0: scatter A_td vs A_FFT per probe (1:1 line = dashed).
    Row 1: ratio A_FFT/A_td vs paddle frequency.

    Under no wind the ratio ≈ 1 everywhere.  Under full wind at the IN probe
    the ratio drops sharply (time-domain is wind-dominated).  The OUT probe
    remains near 1 even under full wind (panel blocks wind fetch).

    Parameters
    ----------
    combined_meta : DataFrame — wave runs only (WaveFrequencyInput not NaN).
    plotvariables : standard dict.
        Extra "plotting" key: probes (list[str]).
    """
    from wavescripts.filters import apply_experimental_filters as _aef

    plotting   = plotvariables.get("plotting", {})
    show_plot  = plotting.get("show_plot",  False)
    save_plot  = plotting.get("save_plot",  False)
    force_stub = plotting.get("force_stub", False)
    probes     = plotting.get("probes",     [])
    figsize    = plotting.get("figsize",    None)

    wave_meta = _aef(
        combined_meta[combined_meta["WaveFrequencyInput [Hz]"].notna()].copy(),
        plotvariables,
    )

    n_runs     = len(wave_meta)
    wind_conds = sorted(wave_meta["WindCondition"].dropna().unique())

    _slots = {
        "n_runs":     n_runs,
        "wind_conds": ", ".join(wind_conds),
    }
    _default = (
        "Time-domain amplitude $A_\\mathrm{{td}}$ vs.\\ FFT amplitude "
        "$A_\\mathrm{{FFT}}$ at the paddle frequency per probe "
        "({n_runs} wave runs; wind: {wind_conds}). "
        "Top row: scatter; dashed = 1:1 line. "
        "Bottom row: ratio $A_\\mathrm{{FFT}}/A_\\mathrm{{td}}$ vs.\\ frequency. "
        "Ratio $\\to 1$ under no wind; $\\to 0$ at the IN probe under full wind "
        "(time-domain dominated by wind waves). "
        "Confirms $A_\\mathrm{{FFT}}$ as the only valid amplitude metric under wind."
    )
    _caption = resolve_caption(plotting, _default, _slots,
                               fn_name="plot_td_vs_fft")

    probes_avail = [p for p in probes
                    if (f"Probe {p} Amplitude"       in wave_meta.columns and
                        f"Probe {p} Amplitude (FFT)" in wave_meta.columns)]
    fs = figsize or (4 * max(len(probes_avail), 1), 8)

    def _make_fig() -> plt.Figure:
        apply_thesis_style()
        fig, axes = plt.subplots(2, len(probes_avail), figsize=fs)
        if len(probes_avail) == 1:
            axes = axes.reshape(2, 1)

        for ci, pos in enumerate(probes_avail):
            td_col  = f"Probe {pos} Amplitude"
            fft_col = f"Probe {pos} Amplitude (FFT)"
            sub = wave_meta[
                [td_col, fft_col, "WindCondition", "WaveFrequencyInput [Hz]"]
            ].dropna(subset=[td_col, fft_col]).copy()
            sub["ratio"] = sub[fft_col] / sub[td_col]

            ax_sc = axes[0, ci]
            ax_rt = axes[1, ci]

            for wc, grp in sub.groupby("WindCondition"):
                c = WIND_COLOR_MAP.get(wc, "grey")
                ax_sc.scatter(grp[td_col], grp[fft_col],
                              s=12, alpha=0.45, color=c, label=wc)
                agg = grp.groupby("WaveFrequencyInput [Hz]")["ratio"].median()
                ax_rt.plot(agg.index, agg.values,
                           marker="o", lw=1.0, markersize=5, color=c, label=wc)

            lim = sub[[td_col, fft_col]].max().max() * 1.08
            ax_sc.plot([0, lim], [0, lim], "k--", lw=0.7, alpha=0.4)
            ax_sc.set_xlim(0, lim); ax_sc.set_ylim(0, lim)
            ax_sc.set_xlabel("$A_\\mathrm{td}$ [mm]")
            if ci == 0:
                ax_sc.set_ylabel("$A_\\mathrm{FFT}$ [mm]")
            ax_sc.set_title(pos, fontsize=9)
            ax_sc.legend(fontsize=7)
            ax_sc.grid(True, alpha=0.3)

            ax_rt.axhline(1.0, color="k", lw=0.7, ls="--", alpha=0.5)
            ax_rt.set_ylim(0, 1.15)
            ax_rt.set_xlabel("Paddle frequency [Hz]")
            if ci == 0:
                ax_rt.set_ylabel("$A_\\mathrm{FFT}\\,/\\,A_\\mathrm{td}$")
            ax_rt.set_title(f"{pos}  ratio", fontsize=9)
            ax_rt.legend(fontsize=7)
            ax_rt.grid(True, alpha=0.3)

        fig.tight_layout()
        return fig

    fig = _make_fig()
    if show_plot:
        plt.show()

    if save_plot:
        meta = build_fig_meta(
            {**plotvariables, "plotting": {**plotting, "caption": _caption}},
            chapter=chapter,
            extra={"script": "plotter.py::plot_td_vs_fft"},
        )
        save_and_stub(fig, meta, plot_type="td_vs_fft", force_stub=force_stub)

    return fig


def plot_first_arrival(
    combined_meta: pd.DataFrame,
    processed_dfs: dict,
    plotvariables: dict,
    chapter: str = "04",
) -> Optional[plt.Figure]:
    """
    First wave arrival time at each probe vs distance from paddle — CH04 §6.

    No-wind runs only (cleanest signal).  Parallel probes at the same
    longitudinal distance are averaged with half-range error bars.
    Noise floor is taken from the mean Amplitude of stillwater rows in
    combined_meta (pipeline-computed, no extra processing needed).

    Parameters
    ----------
    combined_meta : DataFrame — all runs.
    processed_dfs : {path: DataFrame} with eta_{pos} columns.
    plotvariables : standard dict.
        Extra "plotting" keys:
            probes           : list[str]
            threshold_factor : float  (default 5.0 × noise floor)
            window_s         : float  (rolling window [s], default 2.5)
            min_arrival_s    : float  (ignore arrivals before this, default 0.5)
    """
    from wavescripts.wave_detection import find_first_arrival as _find_arr

    plotting         = plotvariables.get("plotting", {})
    show_plot        = plotting.get("show_plot",        False)
    save_plot        = plotting.get("save_plot",        False)
    force_stub       = plotting.get("force_stub",       False)
    probes           = plotting.get("probes",           [])
    threshold_factor = plotting.get("threshold_factor", 5.0)
    window_s         = plotting.get("window_s",         2.5)
    min_arrival_s    = plotting.get("min_arrival_s",    0.5)
    figsize          = plotting.get("figsize",          (9, 5))

    _fs = MEASUREMENT.SAMPLING_RATE

    # ── Noise floor from stillwater rows already in combined_meta ────────────
    sw_mask     = (combined_meta["WaveFrequencyInput [Hz]"].isna()
                   & (combined_meta["WindCondition"] == "no"))
    noise_floor = {}
    for pos in probes:
        col  = f"Probe {pos} Amplitude"
        vals = (combined_meta.loc[sw_mask, col].dropna()
                if col in combined_meta.columns else pd.Series(dtype=float))
        if not vals.empty:
            noise_floor[pos] = float(vals.mean())

    # ── Detect arrivals — no-wind wave runs ───────────────────────────────────
    nowind_wave = combined_meta[
        combined_meta["WaveFrequencyInput [Hz]"].notna()
        & (combined_meta["WindCondition"] == "no")
    ]

    rows = []
    for _, row in nowind_wave.iterrows():
        df   = processed_dfs.get(row["path"])
        if df is None:
            continue
        freq = float(row["WaveFrequencyInput [Hz]"])
        for pos in probes:
            noise = noise_floor.get(pos)
            if not noise or noise <= 0:
                continue
            eta_col = f"eta_{pos}"
            if eta_col not in df.columns:
                continue
            sig      = df[eta_col].dropna().values
            _, t_s   = _find_arr(sig, noise, fs=_fs,
                                 threshold_factor=threshold_factor,
                                 window_s=window_s)
            rows.append({
                "freq_hz":   freq,
                "probe":     pos,
                "dist_mm":   int(pos.split("/")[0]),
                "arrival_s": t_s,
            })

    if not rows:
        print("plot_first_arrival: no arrivals detected — check threshold_factor.")
        return None

    arr_df   = pd.DataFrame(rows)
    plot_df  = arr_df[arr_df["arrival_s"] > min_arrival_s].copy()
    n_runs   = nowind_wave["path"].nunique()
    freqs    = sorted(plot_df["freq_hz"].dropna().unique())

    thresh_strs = ", ".join(
        f"{pos}: {threshold_factor * noise_floor[pos]:.2f}\\,mm"
        for pos in probes if pos in noise_floor
    )
    _slots = {
        "n_runs":      n_runs,
        "threshold":   threshold_factor,
        "thresh_strs": thresh_strs,
        "window_s":    window_s,
        "min_arr":     min_arrival_s,
    }
    _default = (
        "First wave arrival time at each probe vs.\\ distance from the paddle, "
        "no-wind runs ({n_runs} runs). "
        "Detection: rolling {window_s}\\,s window exceeds "
        "{threshold}$\\times$ stillwater noise floor ({thresh_strs}). "
        "Arrivals $\\leq${min_arr}\\,s excluded as instrument transients. "
        "Error bars: half-range across parallel probes at the same "
        "longitudinal distance."
    )
    _caption = resolve_caption(plotting, _default, _slots,
                               fn_name="plot_first_arrival")

    freq_colors = dict(zip(
        freqs, plt.cm.rainbow(np.linspace(0, 1, max(len(freqs), 1)))
    ))
    agg = (
        plot_df
        .groupby(["freq_hz", "dist_mm"])["arrival_s"]
        .agg(mean="mean", err=lambda x: (x.max() - x.min()) / 2)
        .reset_index()
    )

    def _make_fig() -> plt.Figure:
        apply_thesis_style()
        fig, ax = plt.subplots(figsize=figsize)
        for freq, grp in agg.groupby("freq_hz"):
            grp_s = grp.sort_values("dist_mm")
            ax.errorbar(grp_s["dist_mm"], grp_s["mean"], yerr=grp_s["err"],
                        marker="o", markersize=8, linewidth=1.2, capsize=4,
                        color=freq_colors[freq], label=f"{freq:.2f} Hz")
        for pos in probes:
            d = int(pos.split("/")[0])
            ax.axvline(d, color="0.75", lw=0.4, ls="--", zorder=0)
            ax.text(d, 1.01, pos, fontsize=7, ha="center", va="bottom",
                    color="0.4", transform=ax.get_xaxis_transform())
        ax.set_xlabel("Probe distance from paddle [mm]")
        ax.set_ylabel("First arrival [s]")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], fontsize=8, title="frequency")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    fig = _make_fig()
    if show_plot:
        plt.show()

    if save_plot:
        meta = build_fig_meta(
            {**plotvariables, "plotting": {**plotting, "caption": _caption}},
            chapter=chapter,
            extra={"script": "plotter.py::plot_first_arrival"},
        )
        save_and_stub(fig, meta, plot_type="first_arrival",
                      force_stub=force_stub)

    return fig
    return fig
