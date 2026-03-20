#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_quicklook.py
=================
Exploration and interactive tools — NOT for thesis output.

Functions here:
  - Never have save_plot options
  - Never write to output/FIGURES/ or output/TEXFIGU/
  - Are for interactive inspection, comparison, and debugging

Contents
--------
SEABORN EXPLORATION     explore_damping_vs_freq, explore_damping_vs_amp
INTERACTIVE BROWSERS    SignalBrowserFiltered (Qt), RampDetectionBrowser (Qt)
INTERACTIVE EXPORT      save_interactive_plot (Plotly HTML)
DEVELOPER TOOLS         plot_all_markers, plot_rgb
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# seaborn and plotly imported lazily inside the functions that need them

from wavescripts.constants import GlobalColumns as GC
from wavescripts.plot_utils import WIND_COLOR_MAP, MARKERS


# ═══════════════════════════════════════════════════════════════════════════════
# SEABORN EXPLORATION
# ═══════════════════════════════════════════════════════════════════════════════

def explore_damping_vs_freq_old(df: pd.DataFrame,
                                plotvariables: dict) -> None:
    """Old version — kept for reference. Use explore_damping_vs_freq instead."""
    import seaborn as sns
    x = GC.WAVE_FREQUENCY_INPUT
    if df.empty:
        print("explore_damping_vs_freq_old: no data to plot (empty dataframe)")
        return
    g = sns.relplot(
        data=df.sort_values(x),
        x=x, y="mean_out_in",
        hue=GC.WIND_CONDITION, palette=WIND_COLOR_MAP,
        style=GC.PANEL_CONDITION_GROUPED, style_order=["no", "all"],
        col=GC.WAVE_AMPLITUDE_INPUT,
        kind="line", marker=True,
        facet_kws={"sharex": True, "sharey": True},
        height=3.0, aspect=1.2, errorbar=None,
    )
    for ax, (amp, sub) in zip(g.axes.flat,
                               df.groupby(GC.WAVE_AMPLITUDE_INPUT)):
        for (wind, panel), gsub in sub.groupby(
                [GC.WIND_CONDITION, GC.PANEL_CONDITION_GROUPED]):
            ax.errorbar(gsub[x], gsub["mean_out_in"], yerr=gsub["std_out_in"],
                        fmt="none", capsize=3, alpha=0.5)
    sns.move_legend(g, "lower center",
                    bbox_to_anchor=(0.5, 1), ncol=3,
                    title=None, frameon=False)
    g.figure.suptitle("Damping OUT/IN vs Frequency  [quicklook — old]",
                       y=1.04, fontsize=11)
    plt.show()


def _effective_yerr(gsub: pd.DataFrame, fallback_rel: float) -> pd.Series:
    """Effective error bar: std for multi-run groups, fallback % for single runs.

    fallback_rel: relative uncertainty for single-run groups (e.g. 0.10 = ±10%).
    Shown with dashed errorbar style so single-run uncertainty is visually distinct.
    """
    yerr = gsub["std_out_in"].copy()
    single = yerr.isna() & gsub["n_runs"].eq(1) if "n_runs" in gsub.columns else yerr.isna()
    yerr[single] = gsub.loc[single, "mean_out_in"] * fallback_rel
    return yerr


def _explore_damping(
    df: pd.DataFrame,
    plotvariables: dict,
    x_col: str,
    facet_col: str,
    title: str,
) -> None:
    """Shared implementation for explore_damping_vs_freq and explore_damping_vs_amp."""
    import seaborn as sns
    plotting = plotvariables.get("plotting", {})
    fallback_rel = plotting.get("single_run_rel_error", 0.10)
    if df.empty:
        print(f"{title}: no data to plot (empty dataframe)")
        return
    sns.set_style("ticks", {"axes.grid": True})
    g = sns.relplot(
        data=df.sort_values(x_col),
        x=x_col, y="mean_out_in",
        hue=GC.WIND_CONDITION, palette=WIND_COLOR_MAP,
        style=GC.PANEL_CONDITION_GROUPED, style_order=["no", "all"],
        col=facet_col,
        kind="line", marker=True,
        facet_kws={"sharex": True, "sharey": True},
        height=3.0, aspect=1.2, errorbar=None,
    )
    for ax, (_, sub) in zip(g.axes.flat, df.groupby(facet_col)):
        for (wind, panel), gsub in sub.groupby(
                [GC.WIND_CONDITION, GC.PANEL_CONDITION_GROUPED]):
            yerr = _effective_yerr(gsub, fallback_rel)
            # multi-run: solid caps; single-run fallback: dashed/lighter
            is_fallback = gsub["std_out_in"].isna() if "std_out_in" in gsub.columns else pd.Series(False, index=gsub.index)
            ax.errorbar(gsub[x_col], gsub["mean_out_in"], yerr=yerr,
                        fmt="none", capsize=3, alpha=0.5)
            # mark single-run points with an open circle overlay
            sr = gsub[is_fallback]
            if not sr.empty:
                color = WIND_COLOR_MAP.get(wind, "gray")
                ax.scatter(sr[x_col], sr["mean_out_in"],
                           s=60, facecolors="none", edgecolors=color,
                           linewidths=1.2, zorder=5)
    sns.move_legend(g, "lower center",
                    bbox_to_anchor=(0.5, 1), ncol=3,
                    title=None, frameon=False)
    g.figure.suptitle(f"{title}  [quicklook]", y=1.04, fontsize=11)
    plt.show()


def explore_damping_vs_freq(df: pd.DataFrame,
                             plotvariables: dict) -> None:
    """
    Seaborn facet: OUT/IN vs frequency, one column per amplitude.
    Single-run groups get a fallback error bar (±single_run_rel_error, default 10%)
    shown with an open-circle marker overlay.
    """
    _explore_damping(df, plotvariables,
                     x_col=GC.WAVE_FREQUENCY_INPUT,
                     facet_col=GC.WAVE_AMPLITUDE_INPUT,
                     title="Damping OUT/IN vs Frequency")


def explore_damping_vs_amp(df: pd.DataFrame,
                            plotvariables: dict) -> None:
    """
    Seaborn facet: OUT/IN vs amplitude, one column per frequency.
    Single-run groups get a fallback error bar (±single_run_rel_error, default 10%)
    shown with an open-circle marker overlay.
    """
    _explore_damping(df, plotvariables,
                     x_col=GC.WAVE_AMPLITUDE_INPUT,
                     facet_col=GC.WAVE_FREQUENCY_INPUT,
                     title="Damping OUT/IN vs Amplitude")


# Qt browser classes moved to plot_browsers.py — import from there directly.


# ═══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE EXPORT
# ═══════════════════════════════════════════════════════════════════════════════

def save_interactive_plot(df: pd.DataFrame,
                           filename: str = "damping_analysis.html") -> None:
    """Save an interactive Plotly HTML for sharing / exploring in a browser."""
    import plotly.express as px
    # Only pass error_y if there are actual non-NaN std values (plotly can't handle all-NaN)
    has_std = "std_out_in" in df.columns and df["std_out_in"].notna().any()
    fig = px.line(
        df,
        x="WaveFrequencyInput [Hz]",
        y="mean_out_in",
        color=GC.WIND_CONDITION,
        color_discrete_map=WIND_COLOR_MAP,
        error_y="std_out_in" if has_std else None,
        hover_data=["WaveFrequencyInput [Hz]", "WaveAmplitudeInput [Volt]"],
        title="Interactive Damping Analysis",
        markers=True,
    )
    fig.write_html(filename)
    print(f"Interactive plot saved: {filename}")


# ═══════════════════════════════════════════════════════════════════════════════
# DEVELOPER TOOLS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_all_markers() -> None:
    """Visual reference sheet for all matplotlib marker styles."""
    n_cols = 6
    n_rows = (len(MARKERS) + n_cols - 1) // n_cols
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, marker in enumerate(MARKERS):
        x = (i % n_cols) * 2
        y = -(i // n_cols) * 2
        ax.plot(x, y, marker=marker, markersize=20,
                color="red", markeredgecolor="black", markeredgewidth=2)
        ax.text(x, y - 0.6, f"'{marker}'",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_xlim(-1, n_cols * 2)
    ax.set_ylim(-n_rows * 2, 1)
    ax.axis("off")
    ax.set_title("Matplotlib Marker Styles", fontsize=16, fontweight="bold", pad=20)
    plt.show()


def plot_stillwater_fit(dfs: dict, meta: "pd.DataFrame", cfg,
                        date: str | None = None) -> None:
    """Diagnostic plot of the stillwater drift fit for one folder / cfg.

    X-axis is run index (sequential order) because file_date has day precision
    only — no time-of-day is stored in metadata.

    Parameters
    ----------
    dfs   : processed_dfs dict (must be loaded with load_processed=True)
    meta  : combined_meta or single-folder meta
    cfg   : ProbeConfiguration for the date of interest
    date  : optional "YYYY-MM-DD" string to restrict to a single day,
            e.g. date="2026-03-07". Without this, all days in cfg range
            are shown together.

    Shows per-probe subplots with:
      - Blue circles    : no-wave runs (full-run median, high weight)
      - Orange triangles: wave runs (first PRE_WAVE_S seconds, low weight)
      - Black line      : fitted drift (Stillwater Probe {pos} from meta)
    """
    import os
    import matplotlib.dates as mdates
    from datetime import datetime as _dt
    from pathlib import Path as _Path
    from wavescripts.constants import MEASUREMENT, STILLWATER, STILLWATER_EXCLUDE, GlobalColumns as _GC

    _PRE_WAVE_N = int(STILLWATER.PRE_WAVE_S * MEASUREMENT.SAMPLING_RATE)

    # Restrict to cfg date range
    meta_t = pd.to_datetime(meta["file_date"])
    in_range = meta_t >= pd.Timestamp(cfg.valid_from)
    if cfg.valid_until is not None:
        in_range &= meta_t < pd.Timestamp(cfg.valid_until)
    # Optionally narrow to a single day
    if date is not None:
        in_range &= meta_t.dt.strftime("%Y-%m-%d") == date
    meta = meta[in_range].copy().reset_index(drop=True)
    if meta.empty:
        print(f"No rows match cfg '{cfg.name}'" + (f" date={date}" if date else "") + ".")
        return

    wind_str = meta[_GC.WIND_CONDITION].astype(str).str.strip().str.lower()
    nowind   = wind_str.isin(["no", "", "nan", "none"])
    nowave   = meta[_GC.WAVE_FREQUENCY_INPUT].isna() | meta["path"].astype(str).str.lower().str.contains("nowave")

    nowind_rows = meta[nowind].assign(
        _is_nowave=nowave.reindex(meta[nowind].index).fillna(False),
        _mtime=meta[nowind]["path"].apply(
            lambda p: _dt.fromtimestamp(os.path.getmtime(p)) if os.path.exists(p) else None
        ),
    ).reset_index(drop=True)

    col_names = cfg.probe_col_names()
    n = len(col_names)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5), sharey=False)
    if n == 1:
        axes = [axes]

    folder_name = _Path(meta["path"].iloc[0]).parent.name
    title = f"Stillwater drift — {folder_name}" + (f"  [{date}]" if date else "")

    # Build wind timeline: sort all runs by mtime, shade spans where wind != "no"
    all_mtimes = meta["path"].apply(
        lambda p: _dt.fromtimestamp(os.path.getmtime(p)) if os.path.exists(p) else None
    )
    wind_timeline = pd.DataFrame({
        "mtime": all_mtimes,
        "wind":  meta[_GC.WIND_CONDITION].astype(str).str.strip().str.lower(),
    }).dropna(subset=["mtime"]).sort_values("mtime").reset_index(drop=True)
    # Append a sentinel row so the last span has an end
    if not wind_timeline.empty:
        sentinel = wind_timeline.iloc[[-1]].copy()
        sentinel["mtime"] = wind_timeline["mtime"].iloc[-1] + pd.Timedelta(minutes=5)
        wind_timeline = pd.concat([wind_timeline, sentinel], ignore_index=True)

    _wind_colors = {"full": ("#d62728", 0.15), "lowest": ("#ff7f0e", 0.12)}

    for ax, (i, pos) in zip(axes, col_names.items()):
        probe_col = f"Probe {pos}"

        # Wind bands — shade spans between consecutive runs where wind != "no"
        for j in range(len(wind_timeline) - 1):
            wc = wind_timeline.loc[j, "wind"]
            if wc in _wind_colors:
                color, alpha = _wind_colors[wc]
                ax.axvspan(wind_timeline.loc[j, "mtime"],
                           wind_timeline.loc[j + 1, "mtime"],
                           color=color, alpha=alpha, linewidth=0,
                           label=f"wind: {wc}")

        # Collect data points for scatter AND for recomputing the poly fit
        pts_x, pts_v, pts_w = [], [], []

        for _, row in nowind_rows.iterrows():
            df_run = dfs.get(row["path"])
            if df_run is None or probe_col not in df_run.columns:
                continue
            n_samp = None if row["_is_nowave"] else _PRE_WAVE_N
            src = df_run[probe_col].iloc[:n_samp] if n_samp else df_run[probe_col]
            v = float(pd.to_numeric(src, errors="coerce").dropna().median())
            x = row["_mtime"]
            if x is None:
                continue
            fname = _Path(row["path"]).name
            excluded = any(kw in fname for kw in STILLWATER_EXCLUDE)
            if excluded:
                ax.plot(x, v, "x", color="red", ms=8, mew=2, zorder=4,
                        label="excluded")
            elif row["_is_nowave"]:
                ax.plot(x, v, "o", color="steelblue", ms=7, zorder=3,
                        label="nowave (full run)")
                pts_x.append(x.timestamp()); pts_v.append(v); pts_w.append(5)
            else:
                ax.plot(x, v, "^", color="darkorange", ms=6, zorder=3,
                        label=f"wave (pre-{STILLWATER.PRE_WAVE_S:.0f}s)")
                pts_x.append(x.timestamp()); pts_v.append(v); pts_w.append(1)

        # Smooth poly fit curve through the non-excluded data points
        if len(pts_x) >= 2:
            ts = np.array(pts_x)
            t0 = ts[0]
            coeffs = np.polyfit(ts - t0, pts_v, deg=1, w=np.array(pts_w, dtype=float))
            # Extend curve across the full axis range (wind bands may push x beyond data pts)
            all_ts = [t.timestamp() for t in wind_timeline["mtime"] if t is not None]
            t_lo = min(all_ts + list(ts))
            t_hi = max(all_ts + list(ts))
            t_smooth = np.linspace(t_lo, t_hi, 200)
            v_smooth = np.polyval(coeffs, t_smooth - t0)
            ax.plot([_dt.fromtimestamp(t) for t in t_smooth], v_smooth,
                    "-", color="black", lw=1.5, zorder=2, label="poly fit")

        ax.set_title(pos, fontsize=9)
        ax.set_xlabel("time of day")
        ax.set_ylabel("level [mm]")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.tick_params(axis="x", labelrotation=30)

    handles, labels = axes[0].get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        seen.setdefault(l, h)
    axes[0].legend(seen.values(), seen.keys(), fontsize=7)

    fig.suptitle(title, fontsize=10)
    plt.show()


def plot_rgb() -> None:
    """Visual comparison of wind condition colour palettes."""
    palettes = {
        "Current (D3-inspired)":   [WIND_COLOR_MAP["full"],
                                     WIND_COLOR_MAP["lowest"],
                                     WIND_COLOR_MAP["no"]],
        "Standard Science (D3)":   ["#d62728", "#2ca02c", "#1f77b4"],
        "High-Visibility (Indigo)": ["#E31A1C", "#33A02C", "#3F51B5"],
    }
    x = np.linspace(0, 10, 200)
    plt.figure(figsize=(12, 8))
    for i, (name, colors) in enumerate(palettes.items()):
        offset = i * 2.5
        for j, (label, color) in enumerate(
                zip(["Full", "Lowest", "No"], colors)):
            plt.plot(x, np.sin(x + j * 0.5) + offset - j * 0.5,
                     color=color, lw=3, label=f"{name} — {label}")
    plt.title("Wind Condition Palette Comparison", fontsize=15)
    plt.yticks([])
    plt.xlabel("X-axis")
    plt.grid(True, axis="x", alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    plt.show()


if __name__ == "__main__":
    plot_all_markers()
    plot_rgb()
