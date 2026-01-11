#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 16:27:38 2025

@author: gpt
"""

import matplotlib.pyplot as plt
import seaborn as sns
import os


wind_colors = {
    "full":"red",
    "no": "blue",
    "lowest":"green"
}

# ------------------------------------------------------------
# Short label builder (prevents huge legend)
# ------------------------------------------------------------
def make_label(row):
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

    # Mapping for consistent colors
    wind_colors = {
        "full":"red",
        "no":"blue",
        "lowest":"green"
    }
    figsize = (10,6)
    fig, ax = plt.subplots(figsize=figsize)

    for idx, row in df_sel.iterrows():
        
        path_key = row["path"]
        df_ma   = processed_dfs[path_key]
        print("Columns for", row["path"], df_ma.columns.tolist())

        # Color based on wind
        windcond = row["WindCondition"]
        color = wind_colors.get(windcond, "black")

        # Linestyle based on panel condition
        panelcond = row["PanelCondition"]
        linestyle = "--" if "full" in panelcond else "-"

        # Short label for legend
        label = make_label(row)
        
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
    the same legend style as plot_filtered (make_label).
    """
    # Mapping for consistent colors
    wind_colors = {
        "full":"red",
        "no":"blue",
        "lowest":"green"
    }
    chosenprobe = plotvariables["processing"]["chosenprobe"]
    figsize     = plotvariables["plotting"]["figsize"] or (12, 6)

    fig, ax = plt.subplots(figsize=figsize)

    for idx, row in df_sel.iterrows():
        path = row["path"]
        
        # Color based on wind
        windcond = row["WindCondition"]
        color = wind_colors.get(windcond, "black")
        
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
        label = make_label(row)

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

#%% ##

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

#%%
def plot_all_probes(meta_df, ampvar):
    wind_colors = {
        "full":"red",
        "no": "blue",
        "lowest":"green"
    }
    panel_styles = {
        "no": "solid",
        "full": "dashed",
        "reverse":"dashdot"
        }
    
    figsize = (10,6)
    fig, ax = plt.subplots(figsize=figsize)

    probelocations = [9200, 9500, 12444, 12455]
    probelocations = [1, 1.1, 1.2, 1.25]
    newsymbol = ["x","*",".","v","o","x"]

    probelocations = [1, 1.1, 1.2, 1.25]
    xlabels = ["P1", "P2", "P3", "P4"]

    for idx, row in meta_df.iterrows():
        #path = row["path"]

        windcond = row["WindCondition"]
        colla = wind_colors.get(windcond, "black")
        
        panelcond = row["PanelCondition"]
        linjestil = panel_styles.get(panelcond)
        
        marker = "o"

        label = make_label(row)
        
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
    ax.set_title(f"Utkast: Prober 1,2,3,4, (merk: arbitrær x-akse.)avstand P1-P2=30cm, avstand P2-P3/P4= 3,04m ")
    ax.legend()
    plt.tight_layout()
    ax.grid()
    ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='gray')
    ax.minorticks_on()
    ax.set_xticks(probelocations)
    ax.set_xticklabels(xlabels)
    plt.show()
# %%


def facet_plot_freq_vs_mean(df, ampvar):
    # df should be your aggregated stats (mean_P3P2, std_P3P2)
    x='WaveFrequencyInput [Hz]'
    g = sns.relplot(
        data=df.sort_values([x]),
        x=x,
        y='mean_P3P2',
        hue='WindCondition',          # color by condition
        palette=wind_colors,
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


def facet_plot_freq_vs_mean(df, ampvar):
    # df should be your aggregated stats (mean_P3P2, std_P3P2)
    x='WaveAmplitudeInput [Volt]'
    sns.set_style("ticks",{'axes.grid' : True})
    g = sns.relplot(
        data=df.sort_values([x]),
        x=x,
        y='mean_P3P2',
        hue='WindCondition',          # color by condition
        palette=wind_colors,
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
    
# %%

def facet_amp(df, ampvar):
    # df should be your aggregated stats (mean_P3P2, std_P3P2)
    x='WaveAmplitudeInput [Volt]'
    sns.set_style("ticks",{'axes.grid' : True})
    g = sns.relplot(
        data=df.sort_values([x]),
        x=x,
        y='mean_P3P2',
        hue='WindCondition',          # color by condition
        palette=wind_colors,
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



# %%



def plot_damping_2(df, plotvariables):
    xvar="WaveFrequencyInput [Hz]"
    wind_colors = {
        "full": "red",
        "no": "blue",
        "lowest": "green",
    }
    panel_markers = {
        "no": "o",
        "full": "s",
        "reverse": "D",
    }
    panel_styles = {
        "no": "solid",
        "full": "dashed",
        "reverse": "dashdot",
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    # Ensure sorted x for nicer visuals
    df = df.sort_values([ 'PanelConditionGrouped', 'WindCondition', xvar ])

    # One scatter per (panel, wind) group
    for (panel, wind), sub in df.groupby(['PanelConditionGrouped', 'WindCondition'], sort=False):
        color = wind_colors.get(wind, 'black')
        marker = panel_markers.get(panel, 'o')
        linestyle = panel_styles.get(panel, 'solid')

        # Use scatter for points; optionally use plot for connecting line
        ax.scatter(sub[xvar], sub['mean_P3P2'],
                   label=f'{panel} | {wind}',
                   color=color, marker=marker, alpha=0.85)
        ax.plot(sub[xvar], sub['mean_P3P2'],
                color=color, linestyle=linestyle, linewidth=1.5, alpha=0.7)

        # Optional annotations
        # for x, y in zip(sub[xvar], sub['mean_P3P2']):
        #     ax.annotate(f'{y:.2f}', (x, y), textcoords='offset points', xytext=(6, 6), fontsize=8, color=color)

    ax.set_xlabel('kL (wavenumber × geometry length)' if xvar == 'WaveFrequencyInput [Hz]' else xvar)
    ax.set_ylabel('Mean P3/P2 in mm')
    ax.set_title('Damping (mean P3/P2 vs frequency)')
    ax.grid(True)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='gray')
    ax.minorticks_on()

    # Build a clean legend without duplicates
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), title='Panel | Wind', ncol=2)

    plt.tight_layout()
    plt.show()
    
# %%
    











def plot_damping_old(grouped_df, ampvar):
    wind_colors = {
        "full":"red",
        "no": "blue",
        "lowest":"green"
    }
    panel_styles = {
        "no": "solid",
        "full": "dashed",
        "reverse":"dashdot"
        }
    
    figsize = (10,6)
    fig, ax = plt.subplots(figsize=figsize)

    probelocations = [9200, 9500, 12444, 12455]
    probelocations = [1, 1.1, 1.2, 1.25]
    newsymbol = ["x","*",".","v","o","x"]

    # probelocations = [1, 1.1, 1.2, 1.25]
    # xlabels = ["P1", "P2", "P3", "P4"]
    for idx, row in groped_df.iterrows():
        #path = row["path"]

        windcond = row["WindCondition"]
        colla = wind_colors.get(windcond, "black")
        
        panelcond = row["PanelConditionGrouped"]
        
        marker = "o"

        label = make_label(row)
        
        xliste = []
        yliste = []

        
        # --- her plottes --- #
        ax.scatter(meta_df['WaveFrequencyInput [Hz]'], meta_df['mean_P3P2'], linewidth=2, label=label, marker=marker, color=colla)
        
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

    ax.set_xlabel("kL (wavenumber x geometry length")
    ax.set_ylabel("Mean P3/P2 in mm")
    ax.set_title(f"dempning")
    ax.legend()
    plt.tight_layout()
    ax.grid()
    ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='gray')
    ax.minorticks_on()
    ax.set_xticks(probelocations)
    #ax.set_xticklabels()
    plt.show()


# %%




from typing import Mapping, Any, Sequence, Optional
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
    import matplotlib.pyplot as plt  # local import – safe for optional use

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




from typing import Mapping, Any, Optional, Sequence

def plot_damping_combined(
    df,
    *,
    filters: Mapping[str, Any],
    plotting: Mapping[str, Any],  # kept for forward compatibility
    x_col: str = "kL",
    y_col: str = "mean_P3P2",
    err_col: str = "std_P3P2",
    hue_col: str = "WindCondition",
    figsize: Optional[tuple] = None,
    separate: bool = False,
    overlay: bool = False,
    annotate: bool = False,
) -> None:
    """
    Plot mean P3/P2 versus kL with optional error bars and wind-condition colors.
    """
    colors = wind_colors

    # Default colors: use provided mapping or try to pull from `plotting`
    if colors is None:
        colors = plotting.get("wind_colors", None) if isinstance(plotting, Mapping) else None
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



















# 