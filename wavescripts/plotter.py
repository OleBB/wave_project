#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 16:27:38 2025

@author: gpt
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.ticker as ticker
from matplotlib.widgets import Slider, CheckButtons
from typing import Mapping, Any, Optional, Sequence


WIND_COLORS = {
    "full":"red",
    "no": "blue",
    "lowest":"green"
}

MARKERS = ['o', 's', '^', 'v', 'D', '*', 'P', 'X', 'p', 'h', 
           '+', 'x', '.', ',', '|', '_', 'd', '<', '>', '1', '2', '3', '4']

PANEL_STYLES = {
    "no": "solid",
    "full": "dashed",
    "reverse":"solid"
}

MARKER_STYLES = {
    "full": "*",
    "no": "<",
    "lowest": ">"
}

LEGEND_CONFIGS = {
    "outside_right": {"loc": "center left", "bbox_to_anchor": (1.02, 0.5)},
    "outside_left": {"loc": "center right", "bbox_to_anchor": (-0.02, 0.5)},
    "inside": {"loc": "best"},
    "inside_upper_right": {"loc": "upper right"},
    "inside_upper_left": {"loc": "upper left"},
    "below": {"loc": "upper center", "bbox_to_anchor": (0.5, -0.15), "ncol": 3},
    "above": {"loc": "lower center", "bbox_to_anchor": (0.5, 1.02), "ncol": 3},
    "none": None
}

def _apply_legend(ax, freqplotvar):
    """bruker plottevariablene og kobler mot konfiggen i toppen."""
    plotting = freqplotvar.get("plotting", {})
    legend_pos = plotting.get("legend", "outside_right")
    
    config = LEGEND_CONFIGS.get(legend_pos)
    if config is not None:
        ax.legend(**config)


# ------------------------------------------------------------
# Short label builder (prevents huge legend)
# ------------------------------------------------------------
def _make_label(row):
    panel = row.get("PanelCondition", "")
    wind  = row.get("WindCondition", "")
    amp   = row.get("WaveAmplitudeInput [Volt]", "")
    freq  = row.get("WaveFrequencyInput [Hz]", "")

    return f"{panel}panel-{wind}wind-amp{amp}-freq{freq}"

# ------------------------------------------------------------
# Core plot function (single dataset)
# ------------------------------------------------------------
def plot_column(df, start, end, chosenprobe, title="", ax=None,
                color="black", linestyle="-"):

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(df[chosenprobe].iloc[start:end],
            label=title,
            color=color,
            linestyle=linestyle)

    ax.set_title(title)
    return ax

# ------------------------------------------------------------
# ... handles the input from main and runs plot_filtered
# ... Choose to plot separate plots or a combined overlaid plot
# ------------------------------------------------------------
def plotter_selection(processed_dfs, df_sel, plotvariables):

    manual_start = plotvariables["processing"]["rangestart"]
    manual_end   = plotvariables["processing"]["rangeend"]
    
    """Estimated Probe 1 start"""
    
    """... tenker å lage en knapp som velger 
    av eller på manuell range vs calculated range 
    den dataaen må jo lagres til meta.json"""
    # debug - print types and a small sample
    print("type(df_sel) =", type(df_sel))
    try:
        print("df_sel sample (first 5):", list(df_sel)[:5])
    except Exception:
        print("Could not list df_sel")
    #print("type(auto_ranges) =", type(auto_ranges))
    #print("auto_ranges keys (first 10):", list(auto_ranges.keys())[:10])

    # ---- compute plot ranges per path ----
    plot_ranges = {}
    
    for path in df_sel["path"]: #pleide å være processed_dfs
        #auto_start, auto_end = auto_ranges[path]
        start = manual_start if manual_start is not None else None#siste her pleide å være auto
        end   = manual_end   if manual_end   is not None else None
        plot_ranges[path] = (start, end)
    
    print('plot ranges variable:',plot_ranges)    
    # ---- RUN SEPARATE PLOTS ----
    if plotvariables["plotting"]["separate"]:
        for path, df_ma in processed_dfs.items():

            plot_start, plot_end = plot_ranges[path]

            runtime_vars = {
                **plotvariables["processing"],
                **plotvariables["plotting"],
                "rangestart": plot_start,
                "rangeend": plot_end,
            }

            plot_filtered(
                processed_dfs={path: df_ma},
                df_sel=df_sel[df_sel["path"] == path],
                **runtime_vars
            )

    # ---- RUN OVERLAYED PLOT ----
    if  plotvariables["plotting"]["overlay"]:
        plot_overlayed(
            processed_dfs,
            df_sel,
            plot_ranges,    # <-- instead of auto_ranges
            plotvariables
        )

# ------------------------------------------------------------
# Main function: filters metadata, smooths, colors, styles, plots
# ------------------------------------------------------------
def plot_filtered(processed_dfs,
                  df_sel,
                  **runtime_vars):
    #unpack plotvariables/kwargs:
    chosenprobe = runtime_vars["chosenprobe"]
    rangestart  = runtime_vars["rangestart"]
    rangeend    = runtime_vars["rangeend"]
    win         = runtime_vars["win"]
    figsize     = runtime_vars.get("figsize")

    
    figsize = (10,6)
    fig, ax = plt.subplots(figsize=figsize)

    for idx, row in df_sel.iterrows():
        
        path_key = row["path"]
        df_ma   = processed_dfs[path_key]
        print("Columns for", row["path"], df_ma.columns.tolist())

        # Color based on wind
        windcond = row["WindCondition"]
        color = WIND_COLORS.get(windcond, "black")

        # Linestyle based on panel condition
        panelcond = row["PanelCondition"]
        linestyle = "--" if "full" in panelcond else "-"

        # Short label for legend
        label = _make_label(row)
        
        print("start_idx =", rangestart)
        print("end_idx   =", rangeend)
        print("df len    =", len(df_ma))
        df_cut = df_ma.loc[rangestart:rangeend]
        print("df_cut len:", len(df_cut))
        #print(df_cut["Date"].head())
        
        # Convert Date column to milliseconds relative to the start
        t0 = df_cut["Date"].iloc[0]
        time_ms = (df_cut["Date"] - t0).dt.total_seconds() * 1000
    
        # NEW — automatically uses zeroed signal if it exists
        probe_num = chosenprobe.split()[-1] # extracts "1" from "Probe 1"
        zeroed_col = f"eta_{probe_num}"
        
        if zeroed_col in df_cut.columns:
            y_data = df_cut[zeroed_col]
            ylabel = f"{zeroed_col} (zeroed)"
        else:
            y_data = df_cut[chosenprobe]
            ylabel = chosenprobe

        ax.plot(time_ms, y_data, label=label, color=color, linestyle=linestyle)
        ax.set_ylabel(ylabel)

    ax.set_xlabel("Milliseconds")
    ax.set_ylabel(chosenprobe)
    ax.set_title(f"{chosenprobe} — smoothed (window={win})")
    ax.legend()

    plt.show()

def plot_overlayed(processed_dfs, df_sel, plot_ranges, plotvariables):
    """
    Overlay multiple datasets on the same axes,
    aligning each on its own good_start_idx, and using
    the same legend style as plot_filtered (_make_label).
    """

    chosenprobe = plotvariables["processing"]["chosenprobe"]
    figsize     = plotvariables["plotting"]["figsize"] or (12, 6)

    fig, ax = plt.subplots(figsize=figsize)

    for idx, row in df_sel.iterrows():
        path = row["path"]
        
        # Color based on wind
        windcond = row["WindCondition"]
        color = WIND_COLORS.get(windcond, "black")
        
        # skip if this path isn't in processed_dfs/auto_ranges
        if path not in processed_dfs or path not in plot_ranges:
            continue

        df_ma = processed_dfs[path]
        start_idx, end_idx = plot_ranges[path]

        # slice the good part
        df_cut = df_ma.iloc[start_idx:end_idx]

        if df_cut.empty:
            continue

        # time in ms relative to local start
        t0 = df_cut["Date"].iloc[0]
        time_ms = (df_cut["Date"] - t0).dt.total_seconds() * 1000

        # use your existing label function
        label = _make_label(row)

        ax.plot(time_ms,
                df_cut[chosenprobe],
                label=label,
                color=color)

    ax.set_xlabel("Time [ms, aligned at clean start]")
    ax.set_ylabel(chosenprobe)
    ax.set_title(f"{chosenprobe} — overlayed clean segments")
    ax.legend()
    plt.tight_layout()
    plt.show()

#%% ## Ramp detection


def plot_ramp_detection(df, meta_sel, data_col,
                        signal,
                        baseline_mean,
                        threshold,
                        first_motion_idx,
                        good_start_idx,
                        good_range,
                        peaks=None,
                        peak_amplitudes=None,
                        ramp_peak_indices=None,
                        title="Ramp Detection Debug"):
    t0 = df["Date"].iat[0]
    time = (df["Date"]-t0).dt.total_seconds() *1000
    raw = df[data_col].values #bør jeg sette minus for å flippe hele greien?

    plt.figure(figsize=(15, 7))

    # 1. Plot raw + smoothed
    plt.plot(time, raw, color="lightgray", alpha=0.6, label="Raw signal")
    plt.plot(time, signal, color="black", linewidth=2, label=f"Smoothed {data_col}")

    # 2. Baseline & threshold
    plt.axhline(baseline_mean, color="blue", linestyle="--", label=f"Baseline = {baseline_mean:.2f} mm")
    plt.axhline(baseline_mean + threshold, color="red", linestyle=":", alpha=0.7)
    plt.axhline(baseline_mean - threshold, color="red", linestyle=":", alpha=0.7)

    # 3. First motion
    plt.axvline(time[first_motion_idx], color="orange", linewidth=2, linestyle="--",
                label=f"First motion #{first_motion_idx}")

    # 4. Good stable interval
    good_start_idx = int(good_start_idx)
    good_end_idx = min(len(time) - 1, good_start_idx + int(good_range) - 1)

    plt.axvline(time[good_start_idx], color="green", linewidth=3, label=f"Stable start #{good_start_idx}")
    plt.axvline(time[good_end_idx], color="purple", linewidth=2, linestyle="--", label=f"End #{good_end_idx}")
    plt.axvspan(time[good_start_idx], time[good_end_idx], color="green", alpha=0.15, label="Stable region")

    # 5. Optional: highlight detected peaks and ramp-up
    if peaks is not None:
        plt.plot(time[peaks], signal[peaks], "ro", markersize=6, alpha=0.7, label="Detected peaks")
    if ramp_peak_indices is not None:
        plt.plot(time[ramp_peak_indices], signal[ramp_peak_indices],
                 "o", color="lime", markersize=10, markeredgecolor="darkgreen", markeredgewidth=2,
                 label=f"Ramp-up ({len(ramp_peak_indices)} peaks)")

    # ────────────────── MAGIC ZOOM THAT MAKES THE WAVE VISIBLE ──────────────────
    zoom_margin = 15  # mm above/below baseline — perfect for your ±8 mm waves
    plt.ylim(baseline_mean - zoom_margin, baseline_mean + zoom_margin)
    # ─────────────────────────────────────────────────────────────────────────────

    # Title from filename
    # New – works whether metadataframe is DataFrame or single row (Series)
    path_value = meta_sel["path"] if isinstance(meta_sel, pd.Series) else meta_sel["path"].iloc[0]
    filename = str(path_value).split("/")[-1]
    plt.title(f"{filename}  →  {data_col}", fontsize=14, pad=20)

    plt.xlabel("Time [ms]")
    plt.ylabel("Water level [mm]")
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    plt.grid(True, alpha=0.1)
    plt.tight_layout()
    plt.show()

#%% physical probe plot
def plot_all_probes(meta_df :pd.DataFrame, ampvar:dict) -> None:

    panel_styles = {
        "no": "solid",
        "full": "dashed",
        "reverse":"dashdot"
        }
    

    figsize = ampvar.get("plotting", {}).get("figsize")
    fig, ax = plt.subplots(figsize=figsize)

    probelocations = [9200, 9500, 12444, 12455]
    probelocations = [1, 1.1, 1.2, 1.25]
    newsymbol = ["x","*",".","v","o","x"]

    probelocations = [1, 1.1, 1.2, 1.25]
    xlabels = ["P1", "P2", "P3", "P4"]

    for idx, row in meta_df.iterrows():
        #path = row["path"]

        windcond = row["WindCondition"]
        colla = WIND_COLORS.get(windcond, "black")
        
        panelcond = row["PanelCondition"]
        linjestil = panel_styles.get(panelcond)
        
        marker = "o"

        label = _make_label(row)
        
        xliste = []
        yliste = []

        for i in range(1,5):
            x = probelocations[i-1]
            y = row[f"Probe {i} Amplitude"]
            #print(f'x is {x} and y is: {y}')
            xliste.append(x)
            yliste.append(y)
        
        # --- her plottes --- #
        ax.plot(xliste,yliste, linewidth=2, label=label, linestyle=linjestil,marker=marker, color=colla)
        
        # annotate each point with its value (formatted)
        for x, y in zip(xliste, yliste):
            ax.annotate(
                f"{y:.2f}",          # format as needed
                xy=(x, y),
                xytext=(6, 6),       # offset in points to avoid overlapping the marker
                textcoords="offset points",
                fontsize=8,
                color=colla
            )

    ax.set_xlabel("Probenes avstand er ikke representert korrekt visuelt")
    ax.set_ylabel("amplitude in mm")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid()
    ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='gray')
    ax.minorticks_on()
    ax.set_xticks(probelocations)
    ax.set_xticklabels(xlabels)
    plt.show()
# %% Facet amp og freq

def facet_plot_freq_vs_mean(df, ampvar):
    # df should be your aggregated stats (mean_P3P2, std_P3P2)
    x='WaveFrequencyInput [Hz]'
    g = sns.relplot(
        data=df.sort_values([x]),
        x=x,
        y='mean_P3P2',
        hue='WindCondition',          # color by condition
        palette=WIND_COLORS,
        style='PanelConditionGrouped',# differentiate panel
        style_order=["no", "all"],  
        col='WaveAmplitudeInput [Volt]',  # one column per amplitude
        kind='line',
        marker=True,
        facet_kws={'sharex': True, 'sharey': True},
        height=3.0,
        aspect=1.2,
        errorbar=None                 # add std manually if desired
    )
    # Optional: manually draw std error bars per facet
    for ax, ((amp), sub) in zip(g.axes.flat, df.groupby('WaveAmplitudeInput [Volt]')):
        for (wind, panel), gsub in sub.groupby(['WindCondition', 'PanelConditionGrouped']):
            ax.errorbar(gsub[x], gsub['mean_P3P2'], yerr=gsub['std_P3P2'],
                        fmt='none', capsize=3, alpha=0.6)
    sns.move_legend(
    ax, "lower center",
    bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False)
    plt.tight_layout()
    plt.show()


def facet_plot_amp_vs_mean(df, ampvar):
    # df should be your aggregated stats (mean_P3P2, std_P3P2)
    x='WaveAmplitudeInput [Volt]'
    sns.set_style("ticks",{'axes.grid' : True})
    g = sns.relplot(
        data=df.sort_values([x]),
        x=x,
        y='mean_P3P2',
        hue='WindCondition',          # color by condition
        palette=WIND_COLORS,
        style='PanelConditionGrouped',# differentiate panel
        style_order=["no", "all"],
        col='WaveFrequencyInput [Hz]',  # one column per amplitude
        kind='line',
        marker=True,
        facet_kws={'sharex': True, 'sharey': True},
        height=3.0,
        aspect=1.2,
        errorbar=None,              # add std manually if desired
    )
    # Optional: manually draw std error bars per facet
    for ax, ((amp), sub) in zip(g.axes.flat, df.groupby('WaveFrequencyInput [Hz]')):
        for (wind, panel), gsub in sub.groupby(['WindCondition', 'PanelConditionGrouped']):
            ax.errorbar(gsub[x], gsub['mean_P3P2'], yerr=gsub['std_P3P2'],
                        fmt='none', capsize=3, alpha=0.6)
    sns.move_legend(
    ax, "lower center",
    bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False)
    plt.tight_layout()
    plt.show()
    


def facet_amp(df, ampvar):
    # df should be your aggregated stats (mean_P3P2, std_P3P2)
    fig, ax = plt.subplots()
    x="Wavenumber" #WaveFrequencyInput [Hz]"
    sns.set_style("ticks",{'axes.grid' : True})
    g = sns.scatterplot(
        data=df.sort_values([x]),
        x=x,
        y='mean_P3P2',
        hue='WindCondition',          # color by condition
        palette=WIND_COLORS,
        style='PanelConditionGrouped',# differentiate panel
        style_order=["no", "all"],
        #col='WaveFrequencyInput [Hz]',  # one column per amplitude
        #kind='line',
        marker=True,
        #facet_kws={'sharex': True, 'sharey': True},
        #height=3.0,
        #aspect=1.2,
        #errorbar=True,              # add std manually if desired
    )
    # Optional: manually draw std error bars per facet
    collas = ["red", "green", "blue"]
            # Add errorbars matching the marker colors
    # for xi, yi, err, c in zip(x, df["mean_P3P2"], df["std_P3P2"], collas):
        # ax.errorbar(xi, yi, yerr=err, fmt='none', ecolor=c, elinewidth=1.5, capsize=6)

    plt.tight_layout()
    plt.show()







def plot_damping_combined_2(
    df,
    *,
    filters: Mapping[str, Any],
    plotting: Mapping[str, Any],
    x_col: str = "kL",
    y_col: str = "mean_P3P2",
    err_col: str = "std_P3P2",          # column that holds the error magnitude
    hue_col: str = "WindCondition",
    figsize: Optional[tuple] = None,
    separate: bool = False,
    overlay: bool = False,
    annotate: bool = False,
) -> None:
    """
    Plot mean P3/P2 versus kL with optional error bars.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame that already contains the columns needed for the plot
        (e.g. `kL`, `mean`, `std`, `WindCondition` …).

    filters, plotting : dict‑like
        Dictionaries that come from ``amplitudeplotvariables``.
        Only the entries used by this function are listed in the signature;
        the rest are ignored but kept for forward compatibility.

    x_col, y_col, err_col, hue_col : str
        Column names for the x‑axis, y‑axis, error‑bars and the grouping
        variable (wind condition).

    figsize : tuple | None
        Figure size. If ``None`` the default size from ``matplotlib`` is used.

    separate : bool
        If ``True`` each wind condition gets its own subplot (vertical stack).
        If ``False`` everything is drawn in a single Axes.

    overlay : bool
        When ``separate=True`` you can either **overlay** the data from all
        conditions on one subplot (``overlay=True``) or keep them in separate
        sub‑axes (the default).

    annotate : bool
        Annotate each point with its exact ``mean`` value (rounded to 2 d.p.).
    """

    # ------------------------------------------------------------------ #
    # 1️⃣  Apply the simple filters that live in ``filters`` (if you need
    #     more sophisticated filtering you can do it before calling this
    #     function – the example below only shows the most common ones)
    # ------------------------------------------------------------------ #
    # Example: keep only the requested wind conditions
    wind_sel = filters.get("WindCondition", None)
    if wind_sel is not None:
        df = df[df[hue_col].isin(wind_sel)]

    # ------------------------------------------------------------------ #
    # 2️⃣  Set up the figure / axes
    # ------------------------------------------------------------------ #
    if figsize is None:
        figsize = (10, 6) if not separate else (10, 3 * len(df[hue_col].unique()))

    if separate:
        # One row per wind condition (unless overlay=True → a single Axes)
        n_rows = 1 if overlay else len(df[hue_col].unique())
        fig, axes = plt.subplots(
            n_rows, 1, figsize=figsize, sharex=True, sharey=True, squeeze=False
        )
        axes = axes.ravel()
    else:
        fig, ax = plt.subplots(figsize=figsize)
        axes = [ax]  # unify handling later

    # ------------------------------------------------------------------ #
    # 3️⃣  Plot each condition
    # ------------------------------------------------------------------ #
    conditions = df[hue_col].unique()
    for i, cond in enumerate(sorted(conditions)):
        sub = df[df[hue_col] == cond]

        # Choose the Axes we will draw on
        if separate and not overlay:
            ax = axes[i]
        else:
            ax = axes[0]

        # Scatter + errorbars
        ax.errorbar(
            sub[x_col],
            sub[y_col],
            yerr=sub[err_col],
            fmt="o",
            capsize=4,
            label=cond,
            markersize=5,
            linestyle="none",
        )

        # Optional annotation of each point
        if annotate:
            for _, row in sub.iterrows():
                ax.annotate(
                    f"{row[y_col]:.2f}",
                    (row[x_col], row[y_col]),
                    textcoords="offset points",
                    xytext=(0, 5),
                    ha="center",
                    fontsize=8,
                )

        # Axis cosmetics – only add legend / labels once
        if i == 0:
            ax.set_xlabel(r"$kL$ (wavenumber $\times$ geometry length)")
            ax.set_ylabel("Mean P3/P2")
            ax.grid(True, which="both", ls=":", linewidth=0.5)
            ax.minorticks_on()

        if not separate or overlay:
            # All curves share the same Axes → add a legend once
            ax.legend(title="Wind condition")
        else:
            # Separate sub‑plots → title each subplot
            ax.set_title(f"Wind condition: {cond}")

    plt.tight_layout()
    plt.show()





def plot_damping_combined(
    df: pd.DataFrame,
    *,
    amplitudeplotvariables: dict[str, Any],
) -> None:
    """
    Plot y_col vs x_col with optional error bars, colored by hue_col.
    Column names are taken from df (preferred) or defaulted if missing.
    Plot options (figsize, separate, overlay, annotate) are read from amplitudeplotvariables['plotting'].
    """

    # 1) Resolve column names from the DataFrame (fall back to defaults if absent)
    default_x = "kL"
    default_y = "mean_P3P2"
    default_err = "std_P3P2"
    default_hue = "WindCondition"

    x_col = default_x if default_x in df.columns else df.columns[0]
    y_col = default_y if default_y in df.columns else df.columns[1] if len(df.columns) > 1 else default_y
    err_col = default_err if default_err in df.columns else None
    hue_col = default_hue if default_hue in df.columns else None

    # 2) Resolve plotting options from config
    plotting = amplitudeplotvariables.get("plotting", {})
    figsize = plotting.get("figsize", (10, 6))
    separate = bool(plotting.get("separate", False))
    overlay = bool(plotting.get("overlay", False))
    annotate = bool(plotting.get("annotate", False))

    # 3) Sanity checks
    missing = [c for c in [x_col, y_col] if c not in df.columns]
    if missing:
        raise KeyError(f"Required column(s) missing in df: {missing}")
    if err_col is not None and err_col not in df.columns:
        err_col = None  # silently disable error bars if column not present

    # 4) Prepare figure/axes
    if separate and hue_col and hue_col in df.columns:
        groups = list(df[hue_col].dropna().unique())
        n = len(groups)
        fig, axes = plt.subplots(n, 1, figsize=(figsize[0], max(figsize[1], 3) if isinstance(figsize, tuple) else 6), sharex=True)
        if n == 1:
            axes = [axes]
    else:
        fig, ax = plt.subplots(figsize=figsize)
        axes = [ax]

    # 5) Plotting helper
    def draw_one(ax, data, label=None):
        x = data[x_col].values
        y = data[y_col].values
        if err_col is not None:
            yerr = data[err_col].values
            ax.errorbar(x, y, yerr=yerr, fmt='o-', lw=1.5, ms=5, capsize=3,
                        color=WIND_COLORS.get(label, None) if label is not None else None,
                        label=label)
        else:
            ax.plot(x, y, 'o-', lw=1.5, ms=5,
                    color=WIND_COLORS.get(label, None) if label is not None else None,
                    label=label)

        if annotate:
            for xi, yi in zip(x, y):
                ax.annotate(f"{yi:.3g}", (xi, yi), textcoords="offset points", xytext=(5, 4), fontsize=8)

    # 6) Draw data
    if separate and hue_col and hue_col in df.columns:
        for ax, g in zip(axes, df.groupby(hue_col)):
            label, sub = g
            draw_one(ax, sub.sort_values(x_col), label=str(label))
            ax.set_title(f"{hue_col}: {label}")
            ax.set_ylabel(y_col)
            ax.grid(True, which='major', alpha=0.35)
            ax.grid(True, which='minor', alpha=0.15, linestyle=':')
            ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
        axes[-1].set_xlabel(x_col)
    else:
        ax = axes[0]
        if hue_col and hue_col in df.columns and not overlay:
            # Plot each hue as its own line in same axes
            for label, sub in df.groupby(hue_col):
                draw_one(ax, sub.sort_values(x_col), label=str(label))
        else:
            # Overlay or no hue: draw once with a generic label
            label = None
            if hue_col and hue_col in df.columns and overlay:
                label = "overlay"
            draw_one(ax, df.sort_values(x_col), label=label)

        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.grid(True, which='major', alpha=0.35)
        ax.grid(True, which='minor', alpha=0.15, linestyle=':')
        ax.xaxis.set_major_locator(ticker.MaxNLocator(6))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
        # Legend only if meaningful labels exist
        handles, labels = ax.get_legend_handles_labels()
        if labels and any(lab for lab in labels):
            ax.legend(loc="best")

    fig.tight_layout()
    plt.show()




def plot_damping_combined_old(
    df: pd.DataFrame,
    amplitudeplotvariables: dict[str, Any]
) -> None:
    """
    Plot mean P3/P2 versus kL with optional error bars and wind-condition colors.
    """
    colors = WIND_COLORS
    filters =  amplitudeplotvariables.get("filters", {})
    plotting = amplitudeplotvariables.get("plotting", {})
    
    # Default colors: use provided mapping or try to pull from `plotting`
    if colors is None:
        colors = plotting.get("WIND_COLORS", None) if isinstance(plotting, Mapping) else None
    if colors is None:
        # fallback palette if some conditions are missing in mapping
        colors = {}

    # Apply simple filters
    wind_sel = filters.get("WindCondition")
    if wind_sel is not None:
        df = df[df[hue_col].isin(wind_sel)]

    # Determine conditions
    conditions = sorted(df[hue_col].dropna().unique())
    if len(conditions) == 0:
        raise ValueError("No data to plot after filtering.")

    # Figure / axes setup
    multi_axes = separate and not overlay
    n_rows = len(conditions) if multi_axes else 1
    if figsize is None:
        figsize = (10, 6) if n_rows == 1 else (10, 3 * n_rows)

    fig, axes = plt.subplots(
        n_rows, 1, figsize=figsize, sharex=True, sharey=True, squeeze=False
    )
    axes = axes.ravel()

    # Simple fallback palette for any unmapped condition
    fallback = ["C0", "C1", "C2", "C3", "C4", "C5"]

    # Plot each condition
    for i, cond in enumerate(conditions):
        sub = df[df[hue_col] == cond]
        ax = axes[i] if multi_axes else axes[0]

        color = colors.get(cond, fallback[i % len(fallback)])

        ax.errorbar(
            sub[x_col],
            sub[y_col],
            yerr=sub[err_col],
            fmt="o",
            capsize=4,
            label=cond,
            markersize=5,
            linestyle="none",
            color=color,
            ecolor=color,
        )

        if annotate:
            for _, row in sub.iterrows():
                ax.annotate(
                    f"{row[y_col]:.2f}",
                    (row[x_col], row[y_col]),
                    textcoords="offset points",
                    xytext=(0, 5),
                    ha="center",
                    fontsize=8,
                )

        # Cosmetics per-axes
        ax.grid(True, which="both", ls=":", linewidth=0.5)
        ax.minorticks_on()
        if multi_axes:
            ax.set_title(f"Wind condition: {cond}")

    # Shared labels and legend
    axes[0].set_xlabel("$kL$ (wavenumber \\times geometry length)")
    axes[0].set_ylabel("Mean P3/P2")

    if not multi_axes:
        axes[0].legend(title="Wind condition")

    plt.tight_layout()
    plt.show()
    
    
    
    

# %% Frequency plots 

def plot_frequencyspectrum(fft_dict:dict, meta_df: pd.DataFrame, freqplotvar:dict) -> None:
    panel_styles = {
        "no": "solid",
        "full": "dashed",
        "reverse":"solid"
    }
    marker_styles = {
        "full": "*",
        "no": "o",
        "reverse": "^"
    }
    
    base_freq = freqplotvar.get("filters", {}).get("WaveFrequencyInput [Hz]")
    base_freq = base_freq[0]
    
    plotting = freqplotvar.get("plotting", {})
    log_scale = plotting.get("logaritmic", False)
    n_peaks = plotting.get("peaks", None)
    
    probes = plotting.get("probes", 1)

    figsize = freqplotvar.get("plotting", {}).get("figsize")
    fig, ax = plt.subplots(figsize=figsize)
    

    for idx, row in meta_df.iterrows():
        path = row["path"]
        
        if path not in fft_dict:
            continue
        
        df_fft = fft_dict[path]
        
        # print('print')
        # print(df_fft.values)
        # sys.exit()

        windcond = row["WindCondition"]
        colla = WIND_COLORS.get(windcond, "black")
        
        panelcond = row["PanelCondition"]
        linjestil = panel_styles.get(panelcond)

        peak_marker = marker_styles.get(windcond, ".")
        
        # marker = "o"

        label = _make_label(row)
        stopp = 100
        
        for i in probes:
            selected_probe = f"Probe {i}"
            y = df_fft[f"FFT {i}"].head(stopp).dropna()
            x = y.index
            top_indices = y.nlargest(n_peaks).index
            top_values = y[top_indices]
            # print(f'x is {x} and y is: {y}')
            ax.plot(x,
                    y, 
                    linewidth=2, 
                    label=label+"_"+selected_probe, 
                    linestyle=linjestil,
                    marker=None, 
                    color=colla,
                    )
            ax.scatter(top_indices, top_values, 
                  color=colla, s=100, zorder=5, 
                  marker=peak_marker, edgecolors=None, linewidths=0.7)

     # ===== AXES SCALING =====
    if log_scale:
        ax.set_yscale('log')
    
    # ===== AXIS LIMITS =====
    ax.set_xlim(0, 10)
    
    # ===== TICK CONFIGURATION =====
    # Minor ticks at base frequency intervals (e.g., every 1.3 Hz)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(base_freq))
    
    # Major ticks at 2× base frequency (e.g., every 2.6 Hz)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2 * base_freq))
    
    # ===== GRID STYLE =====
    ax.grid(which='major', linestyle='--', alpha=0.6)
    ax.grid(which='minor', linestyle='-.', alpha=0.3)
    
    # ===== REFERENCE LINES =====
    # Vertical lines at frequency multiples
    multiples = np.arange(base_freq, 10, base_freq)  # Skip 0, stop at xlim
    for freq in multiples:
        ax.axvline(freq, color='gray', linewidth=0.6, 
                   linestyle=':', alpha=0.5, zorder=0)
    
    # ===== LABELS AND LEGEND =====
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('FFT Magnitude', fontsize=12)
    _apply_legend(ax, freqplotvar)

    # plt.tight_layout()
    plt.show()



def plot_powerspectraldensity(psd_dict:dict, meta_df: pd.DataFrame, freqplotvar:dict) -> None:
    panel_styles = {
        "no": "solid",
        "full": "dashed",
        "reverse":"solid"
    }
    marker_styles = {
        "full": "*",
        "no": "<",
        "lowest": ">"
    }
    
    base_freq = freqplotvar.get("filters", {}).get("WaveFrequencyInput [Hz]")
    base_freq = base_freq[0]
    
    plotting = freqplotvar.get("plotting", {})
    log_scale = plotting.get("logaritmic", False)
    n_peaks = plotting.get("peaks", None)
    
    probes = plotting.get("probes", 1)

    figsize = freqplotvar.get("plotting", {}).get("figsize")
    fig, ax = plt.subplots(figsize=figsize)
    
    legend_position = plotting.get("legend")
    
    for idx, row in meta_df.iterrows():
        path = row["path"]
        
        if path not in psd_dict:
            continue
        
        df_fft = psd_dict[path]
        
        # print('print')
        # print(df_fft.values)
        # sys.exit()
        
        windcond = row["WindCondition"]
        colla = WIND_COLORS.get(windcond, "black")
        
        panelcond = row["PanelCondition"]
        linjestil = panel_styles.get(panelcond)

        peak_marker = marker_styles.get(windcond, ".")
        
        # marker = "o"

        label = _make_label(row)
        stopp = 100
        
        for i in probes:
            selected_probe = f"Probe {i}"
            y = df_fft[f"Pxx {i}"].head(stopp).dropna()
            x = y.index
            top_indices = y.nlargest(n_peaks).index
            top_values = y[top_indices]
            # print(f'x is {x} and y is: {y}')
            ax.plot(x,
                    y, 
                    linewidth=2, 
                    label=label+"_"+selected_probe, 
                    linestyle=linjestil,
                    marker=None, 
                    color=colla,
                    )
            ax.scatter(top_indices, top_values, 
                  color=colla, s=100, zorder=5, 
                  marker=peak_marker, edgecolors=None, linewidths=0.7)

    # ===== AXES SCALING =====
    if log_scale:
        ax.set_yscale('log')
    
    # ===== AXIS LIMITS =====
    ax.set_xlim(0, 10)
    
    # ===== TICK CONFIGURATION =====
    # Minor ticks at base frequency intervals (e.g., every 1.3 Hz)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(base_freq))
    
    # Major ticks at 2× base frequency (e.g., every 2.6 Hz)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2 * base_freq))
    
    # ===== GRID STYLE =====
    ax.grid(which='major', linestyle='--', alpha=0.6)
    ax.grid(which='minor', linestyle='-.', alpha=0.3)
    
    # ===== REFERENCE LINES =====
    # Vertical lines at frequency multiples
    multiples = np.arange(base_freq, 10, base_freq)  # Skip 0, stop at xlim
    for freq in multiples:
        ax.axvline(freq, color='gray', linewidth=0.6, 
                   linestyle=':', alpha=0.5, zorder=0)
    
    # ===== LABELS AND LEGEND =====
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('PSD', fontsize=12)
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    _apply_legend(ax, freqplotvar)
    # plt.tight_layout()
    plt.show()


def plot_facet_frequencyspectrum(fft_dict: dict, meta_df: pd.DataFrame, freqplotvar: dict) -> tuple:
    
    # Extract config
    plotting = freqplotvar.get("plotting", {})
    facet = plotting.get("facet", False)
    probes = plotting.get("probes", [1])
    n_peaks = plotting.get("peaks", 7)
    log_scale = plotting.get("logarithmic", False)
    
    panel_styles = {
        "no": "solid",
        "full": "dashed",
        "reverse":"solid"
    }
    PANEL_STYLES = panel_styles
    marker_styles = {
        "full": "*",
        "no": "<",
        "lowest": ">"
    }
    MARKER_STYLES=marker_styles
    
    base_freq = freqplotvar.get("filters", {}).get("WaveFrequencyInput [Hz]")
    base_freq = base_freq[0]
    
    plotting = freqplotvar.get("plotting", {})
    log_scale = plotting.get("logaritmic", False)
    n_peaks = plotting.get("peaks", None)
    
    probes = plotting.get("probes", 1)

    figsize = freqplotvar.get("plotting", {}).get("figsize")
    fig, ax = plt.subplots(figsize=figsize)
    
    legend_position = plotting.get("legend")
    
    # ===== CREATE FIGURE =====
    if facet:
        # Create subplots - one per probe
        n_probes = len(probes)
        fig, axes = plt.subplots(n_probes, 1, figsize=(12, 4*n_probes), sharex=True)
        
        # Handle single probe case (axes won't be array)
        if n_probes == 1:
            axes = [axes]
    else:
        # Single plot for all
        fig, ax = plt.subplots(figsize=(12, 6))
        axes = [ax] * len(probes)  # Reuse same ax for all probes
    
    # ===== PLOTTING LOOP =====
    for idx, row in meta_df.iterrows():
        path = row["path"]
        
        if path not in fft_dict:
            continue
        
        df_fft = fft_dict[path]
        
        # Styling
        windcond = row["WindCondition"]
        colla = WIND_COLORS.get(windcond, "black")
        
        panelcond = row["PanelCondition"]
        linjestil = PANEL_STYLES.get(panelcond, "solid")
        peak_marker = MARKER_STYLES.get(windcond, ".")
        
        label = _make_label(row)
        stopp = 100
        
        # Plot each probe
        for probe_idx, probe_num in enumerate(probes):
            ax = axes[probe_idx]  # Get correct subplot
            
            selected_probe = f"Probe {probe_num}"
            y = df_fft[f"FFT {probe_num}"].head(stopp).dropna()
            x = y.index
            
            top_indices = y.nlargest(n_peaks).index
            top_values = y[top_indices]
            
            # Label handling for facet vs single plot
            if facet:
                plot_label = label  # No need to add probe name (it's in subplot title)
            else:
                plot_label = f"{label}_{selected_probe}"
            
            # Plot line
            ax.plot(x, y, 
                    linewidth=2, 
                    label=plot_label, 
                    linestyle=linjestil,
                    color=colla)
            
            # Plot peaks
            ax.scatter(top_indices, top_values, 
                      color=colla, s=100, zorder=5, 
                      marker=peak_marker, edgecolors='black', linewidths=0.7)
    
    # ===== FORMATTING =====
    for probe_idx, probe_num in enumerate(probes):
        ax = axes[probe_idx]
        
        # Log scale
        if log_scale:
            ax.set_yscale('log')
        
        # Grid
        ax.grid(which='major', linestyle='--', alpha=0.6)
        ax.grid(which='minor', linestyle='-.', alpha=0.3)
        
        # Labels
        if facet:
            ax.set_title(f"Probe {probe_num}", fontsize=12, fontweight='bold')
            ax.set_ylabel('FFT Magnitude')
        else:
            ax.set_ylabel('FFT Magnitude')
        
        # Legend
        _apply_legend(ax, freqplotvar)
        
        # X-axis limits and ticks
        ax.set_xlim(0, 10)
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(base_freq))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2 * base_freq))
    
    # X-label only on bottom plot
    axes[-1].set_xlabel('Frequency (Hz)', fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    return (fig, axes) if facet else (fig, axes[0])


# def _top_k_indices(values: np.ndarray, k: int) -> np.ndarray:
#     # Faster than pandas nlargest for numeric arrays; returns indices into values
#     if k is None or k <= 0 or k >= values.size:
#         return np.arange(values.size)
#     # argpartition gives k largest in O(n); then sort those k if you need ordered peaks
#     part = np.argpartition(values, -k)[-k:]
#     # Sort descending for nicer visuals
#     return part[np.argsort(values[part])[::-1]]




def _top_k_indices(values: np.ndarray, k: int) -> np.ndarray:
    if k is None or k <= 0 or k >= values.size:
        return np.arange(values.size)
    part = np.argpartition(values, -k)[-k:]
    return part[np.argsort(values[part])[::-1]]

def plot_facet_condition_frequencyspectrum(
    fft_dict: dict, meta_df: pd.DataFrame, freqplotvar: dict
) -> tuple:
    plotting = freqplotvar.get("plotting", {})
    facet_by = plotting.get("facet_by", None)  # None, 'probe', 'wind', 'panel'
    probes = plotting.get("probes", [1,2,3,4])  # must be iterable
    n_peaks = plotting.get("peaks", None)
    max_points = plotting.get("max_points", 100)  # your 'stopp'

    # Robust base_freq extraction
    base_freq_val = freqplotvar.get("filters", {}).get("WaveFrequencyInput [Hz]")
    base_freq = None
    if isinstance(base_freq_val, (list, tuple, np.ndarray, pd.Series)):
        base_freq = float(base_freq_val[0]) if len(base_freq_val) > 0 else None
    elif base_freq_val is not None:
        base_freq = float(base_freq_val)
    # base_freq must be positive to use MultipleLocator
    use_locators = base_freq is not None and base_freq > 0

    # Decide facets
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
        facet_labels = [""]

    n_facets = len(facet_groups)
    print(f"[DEBUG] facet_by={facet_by}, n_facets={n_facets}, facet_groups={facet_groups}")

    fig, axes = plt.subplots(n_facets, 1, figsize=(12, 4 * n_facets), sharex=True, dpi=120)
    if n_facets == 1:
        axes = [axes]

    for facet_idx, (group, facet_label) in enumerate(zip(facet_groups, facet_labels)):
        ax = axes[facet_idx]

        # Filter data for this facet
        if facet_by == "wind":
            subset = meta_df[meta_df["WindCondition"] == group]
        elif facet_by == "panel":
            subset = meta_df[meta_df["PanelCondition"] == group]
        else:
            subset = meta_df

        print(f"[DEBUG] facet {facet_idx}: label={facet_label}, subset_rows={len(subset)}")

        # Plot rows
        any_plotted = False
        for _, row in subset.iterrows():
            path = row["path"]
            if path not in fft_dict:
                continue
            df_fft = fft_dict[path]

            windcond = row.get("WindCondition")
            colla = WIND_COLORS.get(windcond, "C0")
            panelcond = row.get("PanelCondition")
            linjestil = PANEL_STYLES.get(panelcond, "solid")
            peak_marker = MARKER_STYLES.get(windcond, ".")

            if facet_by == "probe":
                probe_num = group
                col = f"FFT {probe_num}"
                if col not in df_fft:
                    continue
                y = df_fft[col].dropna().iloc[:max_points]
                if y.empty:
                    continue
                x = y.index.values
                ax.plot(x, y.values, linewidth=1.5, linestyle=linjestil, color=colla, antialiased=False)
                if n_peaks and n_peaks > 0:
                    vals = y.values
                    top_idx_local = _top_k_indices(vals, n_peaks)
                    ax.scatter(x[top_idx_local], vals[top_idx_local], color=colla, s=36, zorder=5,
                               marker=peak_marker, edgecolors='none')
                any_plotted = True
            else:
                # Plot all requested probes on this facet
                for probe_num in probes:
                    col = f"FFT {probe_num}"
                    if col not in df_fft:
                        continue
                    y = df_fft[col].dropna().iloc[:max_points]
                    if y.empty:
                        continue
                    x = y.index.values
                    ax.plot(x, y.values, linewidth=1.0, linestyle=linjestil, color=colla, antialiased=False)
                    if n_peaks and n_peaks > 0:
                        vals = y.values
                        top_idx_local = _top_k_indices(vals, n_peaks)
                        ax.scatter(x[top_idx_local], vals[top_idx_local], color=colla, s=24, zorder=5,
                                   marker=peak_marker, edgecolors='none')
                    any_plotted = True

        ax.set_title(facet_label, fontweight='bold')

        # Keep locator logic simple for testing; add MultipleLocator only if valid
        ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
        if use_locators:
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(base_freq))
            ax.xaxis.set_major_locator(ticker.MultipleLocator(2 * base_freq))

        # Temporarily disable legend/grid while debugging
        # _apply_legend(ax, freqplotvar)
        # ax.grid()

        print(f"[DEBUG] facet {facet_idx}: any_plotted={any_plotted}")

    fig.tight_layout()
    return fig, axes

# %%
def plot_frequency_spectrum(
    fft_dict: dict,
    meta_df: pd.DataFrame,
    freqplotvar: dict
) -> tuple:
    """
    Flexible frequency spectrum plotter with extensive customization options.
    
    Parameters
    ----------
    fft_dict : dict
        Dictionary mapping file paths to FFT DataFrames
    meta_df : pd.DataFrame
        Metadata with columns: path, WindCondition, PanelCondition, etc.
    freqplotvar : dict
        Configuration with structure:
        {
            "filters": {"WaveFrequencyInput [Hz]": [value], ...},
            "plotting": {
                "figsize": tuple or None,
                "facet_by": "probe" | "wind" | "panel" | None,
                "probes": [1, 2, 3, 4],
                "peaks": int or None,
                "logaritmic": bool,
                "legend": "inside" | "outside_right" | "below" | "above" | None,
                "max_points": int (default 100),
                "xlim": tuple or None,
                "grid": bool (default True),
                "show": bool (default True)
            }
        }
    
    Returns
    -------
    tuple
        (fig, axes) - matplotlib figure and axes objects
    """
    
    # ===== STYLE DEFINITIONS =====
    PANEL_STYLES = {
        "no": "solid",
        "full": "dashed",
        "reverse": "solid"
    }
    
    MARKER_STYLES = {
        "full": "*",
        "no": "<",
        "lowest": ">"
    }
    
    # ===== EXTRACT CONFIGURATION =====
    plotting = freqplotvar.get("plotting", {})
    
    facet_by = plotting.get("facet_by", None)  # None, 'probe', 'wind', 'panel'
    probes = plotting.get("probes", [1])
    if not isinstance(probes, (list, tuple)):
        probes = [probes]
    
    n_peaks = plotting.get("peaks", None)
    log_scale = plotting.get("logaritmic", False)
    max_points = plotting.get("max_points", 100)
    legend_position = plotting.get("legend", "outside_right")
    show_grid = plotting.get("grid", True)
    show_plot = plotting.get("show", True)
    xlim = plotting.get("xlim", (0, 10))
    
    # Extract base frequency for tick locators
    base_freq_val = freqplotvar.get("filters", {}).get("WaveFrequencyInput [Hz]")
    base_freq = None
    if isinstance(base_freq_val, (list, tuple, np.ndarray, pd.Series)):
        base_freq = float(base_freq_val[0]) if len(base_freq_val) > 0 else None
    elif base_freq_val is not None:
        base_freq = float(base_freq_val)
    use_locators = base_freq is not None and base_freq > 0
    
    # ===== DETERMINE FACET STRUCTURE =====
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
    
    # ===== CREATE FIGURE =====
    default_figsize = (12, 4 * n_facets) if n_facets > 1 else (12, 6)
    figsize = plotting.get("figsize", default_figsize)
    
    fig, axes = plt.subplots(
        n_facets, 1,
        figsize=figsize,
        sharex=True,
        squeeze=False
    )
    axes = axes.flatten()  # Always work with 1D array
    
    # ===== PLOTTING LOOP =====
    for facet_idx, (group, facet_label) in enumerate(zip(facet_groups, facet_labels)):
        ax = axes[facet_idx]
        
        # Filter data for this facet
        if facet_by == "wind":
            subset = meta_df[meta_df["WindCondition"] == group]
        elif facet_by == "panel":
            subset = meta_df[meta_df["PanelCondition"] == group]
        else:
            subset = meta_df
        
        # Plot each row in the subset
        for _, row in subset.iterrows():
            path = row["path"]
            
            if path not in fft_dict:
                continue
            
            df_fft = fft_dict[path]
            
            # Extract styling information
            windcond = row.get("WindCondition", "unknown")
            colla = WIND_COLORS.get(windcond, "black")
            panelcond = row.get("PanelCondition", "unknown")
            linjestil = PANEL_STYLES.get(panelcond, "solid")
            peak_marker = MARKER_STYLES.get(windcond, ".")
            
            # Generate label
            label_base = _make_label(row) if "_make_label" in dir() else f"{windcond}_{panelcond}"
            
            # Determine which probes to plot for this facet
            if facet_by == "probe":
                probes_to_plot = [group]  # Only plot the faceted probe
            else:
                probes_to_plot = probes  # Plot all requested probes
            
            # Plot each probe
            for probe_num in probes_to_plot:
                col = f"FFT {probe_num}"
                
                if col not in df_fft:
                    continue
                
                # Extract data
                y = df_fft[col].dropna().iloc[:max_points]
                if y.empty:
                    continue
                
                x = y.index.values
                
                # Create label for this line
                if facet_by == "probe":
                    plot_label = label_base
                elif len(probes_to_plot) > 1:
                    plot_label = f"{label_base}_P{probe_num}"
                else:
                    plot_label = label_base
                
                # Plot line
                ax.plot(
                    x, y.values,
                    linewidth=1.5,
                    label=plot_label,
                    linestyle=linjestil,
                    color=colla,
                    antialiased=True
                )
                
                # Plot peaks if requested
                if n_peaks and n_peaks > 0:
                    vals = y.values
                    top_idx_local = _top_k_indices(vals, n_peaks)
                    ax.scatter(
                        x[top_idx_local],
                        vals[top_idx_local],
                        color=colla,
                        s=80,
                        zorder=5,
                        marker=peak_marker,
                        edgecolors=None,#denna er visstnok dyr
                        linewidths=0.7
                    )
        
        # ===== FORMATTING FOR THIS FACET =====
        
        # Title
        if facet_label:
            ax.set_title(facet_label, fontsize=12, fontweight='bold')
        
        # Y-axis
        ax.set_ylabel('FFT Magnitude', fontsize=11)
        if log_scale:
            ax.set_yscale('log')
        
        # X-axis limits
        if xlim:
            ax.set_xlim(xlim)
        
        # Tick locators
        if use_locators:
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(base_freq))
            ax.xaxis.set_major_locator(ticker.MultipleLocator(2 * base_freq))
        else:
            ax.xaxis.set_major_locator(ticker.MaxNLocator(10))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(8))
        
        # Grid
        if show_grid:
            ax.grid(which='major', linestyle='--', alpha=0.6)
            ax.grid(which='minor', linestyle='-.', alpha=0.3)
        
        # Legend
        _apply_legend(ax, freqplotvar)
    
    # ===== FINAL TOUCHES =====
    
    # X-label only on bottom plot
    axes[-1].set_xlabel('Frequency (Hz)', fontsize=12)
    
    plt.tight_layout()
    
    if show_plot:
        plt.show()
    
    return fig, axes


def _top_k_indices_2(values: np.ndarray, k: int) -> np.ndarray:
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


def _apply_legend_2(ax, freqplotvar: dict):
    """
    Apply legend to axis based on configuration.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to apply legend to
    freqplotvar : dict
        Configuration dictionary with plotting.legend key
    """
    legend_position = freqplotvar.get("plotting", {}).get("legend", None)
    
    if legend_position is None:
        return
    
    handles, labels = ax.get_legend_handles_labels()
    
    if not handles:
        return
    
    if legend_position == "inside":
        ax.legend(loc='best', framealpha=0.9)
    elif legend_position == "outside_right":
        ax.legend(
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            framealpha=0.9
        )
    elif legend_position == "below":
        ax.legend(
            bbox_to_anchor=(0.5, -0.15),
            loc='upper center',
            ncol=min(len(labels), 4),
            framealpha=0.9
        )
    elif legend_position == "above":
        ax.legend(
            bbox_to_anchor=(0.5, 1.15),
            loc='lower center',
            ncol=min(len(labels), 4),
            framealpha=0.9
        )


def _make_label(row: pd.Series) -> str:
    """
    Create a descriptive label from a metadata row.
    
    Parameters
    ----------
    row : pd.Series
        Row from metadata DataFrame
    
    Returns
    -------
    str
        Formatted label string
    """
    parts = []
    
    if "WindCondition" in row:
        parts.append(f"W:{row['WindCondition']}")
    
    if "PanelCondition" in row:
        parts.append(f"P:{row['PanelCondition']}")
    
    # Add other relevant fields as needed
    # if "WaveAmplitudeInput [Volt]" in row:
    #     parts.append(f"A:{row['WaveAmplitudeInput [Volt]']}")
    
    return "_".join(parts) if parts else "Unknown"





# %% hjelpemiddel
"""Plott alle markører"""
import matplotlib.pyplot as plt

def plot_all_markers():
    markers = ['o', 's', '^', 'v', 'D', '*', 'P', 'X', 'p', 'h', 
               '+', 'x', '.', ',', '|', '_', 'd', '<', '>', '1', '2', '3', '4']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    n_cols = 6
    n_rows = (len(markers) + n_cols - 1) // n_cols
    
    for i, marker in enumerate(markers):
        row = i // n_cols
        col = i % n_cols
        
        x = col * 2
        y = -row * 2
        
        ax.plot(x, y, marker=marker, markersize=20, 
                color='red', markeredgecolor='black', markeredgewidth=2)
        
        ax.text(x, y - 0.6, f"'{marker}'", ha='center', fontsize=10, fontweight='bold')
    
    ax.set_xlim(-1, n_cols * 2)
    ax.set_ylim(-n_rows * 2, 1)
    ax.axis('off')
    ax.set_title('Matplotlib Marker Styles', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print('main called')
    plot_all_markers()
    #legg ved fleire hjelpefunksjoner