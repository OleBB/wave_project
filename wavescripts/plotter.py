#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 16:27:38 2025

@author: gpt
"""

import matplotlib.pyplot as plt
import os


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
def plot_selection(processed_dfs, df_sel, auto_ranges, plotvariables):

    manual_start = plotvariables["processing"]["rangestart"]
    manual_end   = plotvariables["processing"]["rangeend"]
    plot_separate = plotvariables["plotting"]["separate"]
    plot_overlay = plotvariables["plotting"]["overlay"]

    # ---- compute plot ranges per path ----
    plot_ranges = {}
    for path in processed_dfs:
        auto_start, auto_end = auto_ranges[path]
        start = manual_start if manual_start is not None else auto_start
        end   = manual_end   if manual_end   is not None else auto_end
        plot_ranges[path] = (start, end)

    # ---- SEPARATE PLOTS ----
    if plot_separate:
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

    # ---- OVERLAYED PLOT ----
    if plot_overlay:
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
   


    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, row in df_sel.iterrows():
        

        path_key = row["path"]
        df_ma   = processed_dfs[path_key]
        #print("Columns for", row["path"], df_ma.columns.tolist())

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
        print(df_cut["Date"].head())
        
        # Convert Date column to milliseconds relative to the start
        t0 = df_cut["Date"].iloc[0]
        time_ms = (df_cut["Date"] - t0).dt.total_seconds() * 1000
    
         # Plot it
        ax.plot(time_ms, df_cut[chosenprobe],
                label=label,
                color=color,
                linestyle=linestyle) 

    ax.set_xlabel("Milliseconds")
    ax.set_ylabel(chosenprobe)
    ax.set_title(f"{chosenprobe} — smoothed (win={win})")
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


def plot_ramp_debug(df, data_col, debug_info, title="Ramp Detection Debug"):

    signal = debug_info["signal"]
    baseline_mean = debug_info["baseline_mean"]
    threshold = debug_info["threshold"]
    first_motion_idx = debug_info["first_motion_idx"]
    good_start_idx, good_end_idx = debug_info["good_start"], debug_info["good_end"]

    time = df["Date"]

    plt.figure(figsize=(14, 6))
    plt.plot(time, df[data_col], label="Raw", alpha=0.4)
    plt.plot(time, signal, label="Smoothed", linewidth=2)

    plt.axhline(baseline_mean, color="blue", linestyle="--", label="Baseline mean")
    plt.axhline(baseline_mean + threshold, color="red", linestyle="--", label="+threshold")
    plt.axhline(baseline_mean - threshold, color="red", linestyle="--")

    plt.axvline(time.iloc[first_motion_idx], color="orange", linestyle="--", label="first motion")
    plt.axvline(time.iloc[good_start_idx], color="green", linestyle="--", label="good start")
    plt.axvline(time.iloc[good_end_idx], color="purple", linestyle="--", label="good end")

    plt.axvspan(time.iloc[good_start_idx], time.iloc[good_end_idx], color="green", alpha=0.15)

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel(data_col)
    plt.legend()
    plt.tight_layout()
    plt.show()







