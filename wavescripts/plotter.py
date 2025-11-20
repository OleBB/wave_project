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
# Main function: filters metadata, smooths, colors, styles, plots
# ------------------------------------------------------------
def plot_filtered(processed_dfs,
                  df_sel,
                  amp=None,
                  freq=None,
                  wind=None,
                  tunnel=None,
                  mooring=None,
                  chosenprobe=None,
                  rangestart=0,
                  rangeend=None,
                  data_cols=None,
                  win=1,
                  figsize=None
                  ):
    
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

def plot_multiple(processed_dfs, df_sel, auto_ranges, plotvariables):
    """
    Plot all datasets on top of each other,
    aligned so that their wave starts at t = 0 ms.
    """

    chosenprobe = plotvariables["processing"]["chosenprobe"]
    figsize     = plotvariables["plotting"]["figsize"] or (12, 6)

    fig, ax = plt.subplots(figsize=figsize)

    for idx, row in df_sel.iterrows():

        path = row["path"]
        df = processed_dfs[path]

        start_idx, end_idx = auto_ranges[path]
        df_cut = df.iloc[start_idx:end_idx]

        # Convert timestamps to milliseconds relative to local start
        t0 = df_cut["Date"].iloc[0]
        time_ms = (df_cut["Date"] - t0).dt.total_seconds() * 1000

        # Label for the legend
        label = f"{row['WindCondition']} — {row['Mooring']} — {row['WaveAmplitudeInput [Volt]']}"

        ax.plot(time_ms, df_cut[chosenprobe], label=label)

    ax.set_xlabel("Time [ms, aligned]")
    ax.set_ylabel(chosenprobe)
    ax.set_title(f"Overlayed Waves — Aligned by Ramp-up End")
    ax.legend()
    plt.show()


def debug_plot_ramp_detection(df, data_col,
                              signal,
                              baseline_mean,
                              threshold,
                              first_motion_idx,
                              good_start_idx,
                              good_end_idx,
                              title="Ramp Detection Debug"):
    """
    Visualizes the detection process:
    - raw data
    - smoothed signal used for detection
    - baseline region
    - threshold
    - motion point
    - good region window
    """

    time = df["Date"]  # datetime index

    plt.figure(figsize=(14, 6))

    # --- Raw probe signal ---
    plt.plot(time, df[data_col],
             label="Raw signal", alpha=0.4)

    # --- Smoothed detection signal ---
    plt.plot(time, signal,
             label="Smoothed (detect)", linewidth=2)

    # --- Baseline mean ---
    plt.axhline(baseline_mean, color="blue", linestyle="--",
                label=f"Baseline mean = {baseline_mean:.3f}")

    # --- Threshold ---
    plt.axhline(baseline_mean + threshold, color="red", linestyle="--",
                label=f"+ Threshold ({threshold:.3f})")
    plt.axhline(baseline_mean - threshold, color="red", linestyle="--")

    # --- First motion ---
    plt.axvline(time.iloc[first_motion_idx],
                color="orange", linestyle="--", linewidth=2,
                label=f"First motion index = {first_motion_idx}")

    # --- Good interval start / end ---
    plt.axvline(time.iloc[good_start_idx],
                color="green", linestyle="--", linewidth=2,
                label=f"Good start = {good_start_idx}")

    plt.axvline(time.iloc[good_end_idx],
                color="purple", linestyle="--", linewidth=2,
                label=f"Good end = {good_end_idx}")

    # --- Shade good region ---
    plt.axvspan(time.iloc[good_start_idx],
                time.iloc[good_end_idx],
                color="green", alpha=0.15)

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel(data_col)
    plt.legend()
    plt.tight_layout()
    plt.show()








