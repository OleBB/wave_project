#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 16:27:38 2025

@author: gpt
"""

import matplotlib.pyplot as plt
import os
from wavescripts.processor import find_resting_levels, remove_outliers, apply_moving_average, compute_simple_amplitudes

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
def plot_filtered(meta_df,
                  dfs,
                  amp=None,
                  freq=None,
                  wind=None,
                  chosenprobe=None,
                  rangestart=0,
                  rangeend=None,
                  data_cols=None,
                  win=1,
                  figsize=None
                  ):
    
    # - By default all columns will be processed
    if data_cols is None:
        data_cols = ["Probe 1","Probe 2","Probe 3","Probe 4"]
    # --- Filtering based on requested conditions ---
    df_sel = meta_df.copy()

    if amp is not None:
        df_sel = df_sel[df_sel["WaveAmplitudeInput [Volt]"] == amp]

    if freq is not None:
        df_sel = df_sel[df_sel["WaveFrequencyInput [Hz]"] == freq]

    if wind is not None:
        df_sel = df_sel[df_sel["WindCondition"] == wind]

    if df_sel.empty:
        print("No matching datasets found.")
        return

    # Mapping for consistent colors
    wind_colors = {
        "full":   "red",
        "no": "blue",
        "lowest": "green"
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, row in df_sel.iterrows():

        path_key = row["path"]
        df_raw   = dfs[path_key]
        
        print("Columns for", row["path"], df_raw.columns.tolist())

        # Apply moving average
        df_ma = apply_moving_average(df_raw, data_cols, win=win)
        # - For smooth signals, a simple location of n biggest is taken.
        n_amplitudes = 10
        average_simple_amplitude = compute_simple_amplitudes(df_ma, data_cols, n_amplitudes) 
        # Color based on wind
        windcond = row["WindCondition"]
        color = wind_colors.get(windcond, "black")

        # Linestyle based on panel condition
        panelcond = row["PanelCondition"]
        linestyle = "--" if "full" in panelcond else "-"

        # Short label for legend
        label = make_label(row)

        # Plot it
        ax.plot(df_ma[chosenprobe].iloc[rangestart:rangeend],
                label=label,
                color=color,
                linestyle=linestyle)

    ax.set_xlabel("Milliseconds")
    ax.set_ylabel(chosenprobe)
    ax.set_title(f"{chosenprobe} — smoothed (win={win})")
    ax.legend()

    plt.show()














