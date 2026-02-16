#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 16:27:38 2025

@author: gpt
"""
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.ticker as ticker
from matplotlib.widgets import Slider, CheckButtons
from typing import Mapping, Any, Optional, Sequence, Tuple, Dict
from matplotlib.lines import Line2D
from matplotlib.offsetbox import (AnchoredOffsetbox, AuxTransformBox,
                                  DrawingArea, TextArea, VPacker)
from matplotlib.patches import Circle, Ellipse
from matplotlib.offsetbox import AnchoredText
from matplotlib.offsetbox import AnchoredOffsetbox   # or AnchoredOffsetBox

from wavescripts.signal_processing import get_positive_spectrum

from wavescripts.filters import filter_for_amplitude_plot

from wavescripts.constants import MEASUREMENT
from wavescripts.constants import SIGNAL, RAMP, MEASUREMENT, get_smoothing_window
from wavescripts.constants import (
    ProbeColumns as PC, 
    GlobalColumns as GC, 
    ColumnGroups as CG,
    CalculationResultColumns as RC
)

WIND_COLORS = {
    "full": "red",
    "no": "blue",
    "lowest": "green"
}

def draw_anchored_text(ax, txt="Figuren", loc="upper left", fontsize=9,
                       facecolor="white", edgecolor="gray", alpha=0.85):
    """
    Add a small text box anchored to a corner of *ax*.
    """
    at = AnchoredText(
        txt,
        loc=loc,               # any of the usual legend locations
        prop=dict(size=fontsize, color="black"),
        frameon=True,
        pad=0.3,               # padding inside the box (in points)
    )
    # style the surrounding rectangle
    at.patch.set_facecolor(facecolor)
    at.patch.set_edgecolor(edgecolor)
    at.patch.set_alpha(alpha)
    # optional rounded corners
    at.patch.set_boxstyle("round,pad=0.4,rounding_size=0.2")
    ax.add_artist(at)


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
        
def _apply_legend_3(ax, freqplotvar: dict):
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
    
    # Common legend properties for better readability
    legend_props = {
        'framealpha': 0.9,
        'fontsize': 8,  # Smaller font
        'labelspacing': 0.3,  # Tighter spacing between entries
        'handlelength': 1.5,  # Shorter line samples
        'handletextpad': 0.5,  # Less space between line and text
    }
    
    if legend_position == "inside":
        ax.legend(loc='best', **legend_props)
    elif legend_position == "outside_right":
        ax.legend(
            bbox_to_anchor=(1.02, 1),
            loc='upper left',
            **legend_props
        )
    elif legend_position == "below":
        # Multi-column layout for horizontal space
        ncol = min(len(labels), 5)  # Up to 5 columns
        ax.legend(
            bbox_to_anchor=(0.5, -0.2),
            loc='upper center',
            ncol=ncol,
            **legend_props
        )
    elif legend_position == "above":
        ncol = min(len(labels), 5)
        ax.legend(
            bbox_to_anchor=(0.5, 1.05),
            loc='lower center',
            ncol=ncol,
            **legend_props
        )

def _top_k_indices(values: np.ndarray, k: int) -> np.ndarray:
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



# ------------------------------------------------------------
# Short label builder (prevents huge legend)
# ------------------------------------------------------------
def _make_label(row):
    panel = row.get("PanelCondition", "")
    wind  = row.get("WindCondition", "")
    amp   = row.get("WaveAmplitudeInput [Volt]", "")
    freq  = row.get("WaveFrequencyInput [Hz]", "")

    return f"{panel}panel-{wind}wind-amp{amp}-freq{freq}"

#Claude
def _make_label_2(row: pd.Series) -> str:
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
                        good_end_idx,
                        peaks=None,
                        peak_amplitudes=None,
                        ramp_peak_indices=None,
                        title="Ramp Detection Debug"):
    # --- Build time (ms) and raw as NumPy arrays ---
    if "Date" not in df.columns:
        raise ValueError("df must contain a 'Date' column")
    if data_col not in df.columns:
        raise ValueError(f"df must contain the '{data_col}' column")

    t0 = df["Date"].iat[0]
    # time in milliseconds

    time_ms = (df["Date"] - t0).dt.total_seconds().to_numpy() * MEASUREMENT.M_TO_MM
    raw = df[data_col].to_numpy()

    n = len(time_ms)
    if len(signal) != n:
        raise ValueError(f"signal length ({len(signal)}) != df length ({n})")
    if not (0 <= first_motion_idx < n):
        raise ValueError(f"first_motion_idx out of bounds: {first_motion_idx}")

    # --- Compute good range safely (don’t overwrite inputs) ---
    good_start_i = int(good_start_idx)
    # If good_end_idx is provided, honor it; otherwise derive from good_range
    if good_end_idx is not None:
        good_end_i = int(good_end_idx)
    else:
        good_end_i = good_start_i + int(good_range)

    # Clamp to valid bounds and ensure start < end
    good_start_i = max(0, min(good_start_i, n - 2))
    good_end_i = max(good_start_i + 1, min(good_end_i, n - 1))

    # --- Create figure/axes and start plotting immediately ---
    fig, ax = plt.subplots(figsize=(15, 7))
    fig.suptitle(title)

    # 1) Plot raw + smoothed
    ax.plot(time_ms, raw, color="lightgray", alpha=0.6, label="Raw signal")
    ax.plot(time_ms, signal, color="black", linewidth=2, label=f"Smoothed {data_col}")

    # Optional: reference sine over the selected range
    # NOTE: time_ms is in milliseconds; the angular frequency below must be in rad/ms.
    # If you prefer Hz, convert to seconds and use 2*pi*f*t_sec instead.
    amp_in = float(meta_sel["WaveAmplitudeInput [Volt]"])
    amp = amp_in if amp_in is not None else 20

    t_cut = time_ms[good_start_i:good_end_i]

    # Example sine using your form: sin(omega * t_ms), where omega is in rad/ms.
    omega_rad_per_ms = 0.004 * 1.3  # adjust to your wave
    sinewave = baseline_mean + (100.0 * amp) * np.sin(omega_rad_per_ms * t_cut)

    # Uncomment to draw the sine reference
    ax.plot(t_cut, sinewave, color="red", linestyle="--", label="Ref sine")

    # 2) Baseline & threshold
    ax.axhline(baseline_mean, color="blue", linestyle="--", label=f"Baseline = {baseline_mean:.2f} mm")
    ax.axhline(baseline_mean + threshold, color="red", linestyle=":", alpha=0.7)
    ax.axhline(baseline_mean - threshold, color="red", linestyle=":", alpha=0.7)

    # 3) First motion
    ax.axvline(time_ms[first_motion_idx], color="orange", linewidth=2, linestyle="--",
               label=f"First motion #{first_motion_idx}")

    # 4) Good stable interval
    ax.axvline(time_ms[good_start_i], color="green", linewidth=3, label=f"Stable start #{good_start_i}")
    ax.axvline(time_ms[good_end_i], color="purple", linewidth=2, linestyle="--", label=f"End #{good_end_i}")
    ax.axvspan(time_ms[good_start_i], time_ms[good_end_i], color="green", alpha=0.08, label="Stable region")

    # 5) Optional: peaks and ramp-up
    if peaks is not None and len(peaks) > 0:
        peaks = np.asarray(peaks, dtype=int)
        peaks = peaks[(peaks >= 0) & (peaks < n)]
        ax.plot(time_ms[peaks], signal[peaks], "ro", markersize=6, alpha=0.7, label="Detected peaks")
    if ramp_peak_indices is not None and len(ramp_peak_indices) > 0:
        rpi = np.asarray(ramp_peak_indices, dtype=int)
        rpi = rpi[(rpi >= 0) & (rpi < n)]
        ax.plot(time_ms[rpi], signal[rpi],
                "o", color="lime", markersize=10, markeredgecolor="darkgreen", markeredgewidth=2,
                label=f"Ramp-up ({len(rpi)} peaks)")

    # Zoom around baseline to make waves visible
    zoom_margin = amp*100  # 0.1amp * 100 = 10 mm? 
    print(f"zoom margin er {zoom_margin}")
    ax.set_ylim(baseline_mean - zoom_margin, baseline_mean + zoom_margin)
    
    """TODO:
        
        #markere peaks, enten med tickers eller grid
    """
    
    # Title from metadata path
    try:
        path_value = meta_sel["path"] if isinstance(meta_sel, pd.Series) else meta_sel["path"].iloc[0]
        filename = str(path_value).split("/")[-1]
        ax.set_title(f"{filename}  →  {data_col}", fontsize=14, pad=20)
    except Exception:
        pass  # keep suptitle only if path missing

    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Water level [mm]")
    ax.grid(True, alpha=0.1)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    fig.tight_layout()
    return fig, ax



"""
def old_plot_ramp_detection(df, meta_sel, data_col,
                        signal,
                        baseline_mean,
                        threshold,
                        first_motion_idx,
                        good_start_idx,
                        good_range,
                        good_end_idx,
                        peaks=None,
                        peak_amplitudes=None,
                        ramp_peak_indices=None,
                        title="Ramp Detection Debug"):
    t0 = df["Date"].iat[0]
    time = (df["Date"]-t0).dt.total_seconds() *1000
    raw = df[data_col].values #bør jeg sette minus for å flippe hele greien?

    plt.figure(figsize=(15, 7))
    
    
    amp = meta_sel["WaveAmplitudeInput [Volt]"]

    time_cut = time[good_start_idx:good_end_idx]
    # print(time_cut)
    sinewave = 100*amp*np.sin(0.004*1.3*time_cut)+baseline_mean
    
    # 1. Plot raw + smoothed + sine
    plt.plot(time, raw, color="lightgray", alpha=0.6, label="Raw signal")
    plt.plot(time, signal, color="black", linewidth=2, label=f"Smoothed {data_col}")
    # plt.plot(time_cut, sinewave, color="red", linestyle="--")

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
    plt.tight_layout()"""


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
    g, "lower center",
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
    g, "lower center",
    bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False)
    plt.tight_layout()
    plt.show()
    


def plot_damping_scatter(stats_df: pd.DataFrame, 
                         save_path: str = None,
                         show_errorbars: bool = True,
                         size_by_amplitude: bool = True,
                         figsize: tuple = (10, 6)):
    """
    Single scatter plot showing all damping data points.
    
    Args:
        stats_df: Output from damping_all_amplitude_grouper()
        save_path: Optional path to save figure
        show_errorbars: Add error bars for std_P3P2
        size_by_amplitude: Vary marker size by amplitude
        figsize: Figure size (width, height)
    """
    
    WIND_COLORS = {
        "full": "red",
        "no": "blue",
        "lowest": "green"
    }
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.set_style("ticks", {'axes.grid': True})
    
    # Sort for cleaner plotting
    plot_data = stats_df.sort_values([GC.WAVE_FREQUENCY_INPUT])
    
    # Prepare kwargs for scatterplot
    scatter_kwargs = {
        'data': plot_data,
        'x': GC.WAVE_FREQUENCY_INPUT,
        'y': 'mean_P3P2',
        'hue': GC.WIND_CONDITION,
        'palette': WIND_COLORS,
        'style': GC.PANEL_CONDITION_GROUPED,
        'style_order': ["no", "all"],
        'alpha': 0.7,
        'ax': ax,
        'legend': 'auto',  # Better legend handling
    }
    
    # Optionally add size encoding
    if size_by_amplitude:
        scatter_kwargs['size'] = GC.WAVE_AMPLITUDE_INPUT
        scatter_kwargs['sizes'] = (50, 200)
    else:
        scatter_kwargs['s'] = 80  # Fixed size
    
    # Main scatter plot
    sns.scatterplot(**scatter_kwargs)
    
    # Add error bars if requested
    if show_errorbars and 'std_P3P2' in plot_data.columns:
        # Vectorized approach - much faster than loop
        for wind in plot_data[GC.WIND_CONDITION].unique():
            wind_data = plot_data[plot_data[GC.WIND_CONDITION] == wind]
            color = WIND_COLORS.get(wind, 'gray')
            
            ax.errorbar(
                wind_data[GC.WAVE_FREQUENCY_INPUT],
                wind_data['mean_P3P2'],
                yerr=wind_data['std_P3P2'],
                fmt='none',
                ecolor=color,
                elinewidth=1,
                capsize=3,
                alpha=0.4,
                zorder=1  # Behind the markers
            )
    
    # Formatting
    ax.set_xlabel('Frequency [Hz]', fontsize=12)
    ax.set_ylabel('P3/P2 (mean ± std)', fontsize=12)
    ax.set_title('Damping Ratio: All Conditions', fontsize=14, fontweight='bold')
    
    # Improve legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, 
              loc='best', 
              frameon=True, 
              framealpha=0.9,
              fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved to: {save_path}")
    
    plt.show()



# %%
def plot_damping_results(stats_df: pd.DataFrame, 
                         save_path: str = None,
                         figsize: tuple = (14, 10)):
    """
    Simple plotter for damping aggregation results.
    
    Creates subplots showing P3/P2 ratio vs. frequency for different conditions.
    
    Args:
        stats_df: Output from damping_all_amplitude_grouper()
        save_path: Optional path to save figure
        figsize: Figure size (width, height)
    """
    
    # Get unique conditions
    panel_conditions = stats_df[GC.PANEL_CONDITION_GROUPED].unique()
    wind_conditions = stats_df[GC.WIND_CONDITION].unique()
    
    n_panels = len(panel_conditions)
    n_winds = len(wind_conditions)
    
    fig, axes = plt.subplots(n_panels, n_winds, figsize=figsize, squeeze=False, sharex=True, sharey=True)
    fig.suptitle('Damping Ratio (P3/P2) vs Frequency', fontsize=16, y=0.995)
    
    for i, panel in enumerate(panel_conditions):
        for j, wind in enumerate(wind_conditions):
            ax = axes[i, j]
            
            # Filter data for this subplot
            mask = (stats_df[GC.PANEL_CONDITION_GROUPED] == panel) & \
                   (stats_df[GC.WIND_CONDITION] == wind)
            subset = stats_df[mask]
            
            if subset.empty:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                       transform=ax.transAxes)
                ax.set_title(f'{panel} / {wind}')
                continue
            
            # Plot each amplitude as separate line
            for amp in subset[GC.WAVE_AMPLITUDE_INPUT].unique():
                amp_data = subset[subset[GC.WAVE_AMPLITUDE_INPUT] == amp]
                amp_data = amp_data.sort_values(GC.WAVE_FREQUENCY_INPUT)
                
                ax.errorbar(
                    amp_data[GC.WAVE_FREQUENCY_INPUT],
                    amp_data['mean_P3P2'],
                    yerr=amp_data['std_P3P2'],
                    marker='o',
                    label=f'{amp:.2f} V',
                    capsize=3,
                    alpha=0.7
                )
            
            ax.set_xlabel('Frequency [Hz]')
            ax.set_ylabel('P3/P2')
            ax.set_title(f'{panel}panel / {wind}wind')
            ax.grid(True, alpha=0.3)
            ax.legend(title='Amplitude', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()


# Usage:
# stats = damping_all_amplitude_grouper(combined_meta_df)
# plot_damping_results(stats, save_path='damping_plot.png')


# %% funker greit - gemini
def plot_damping_pro(df: pd.DataFrame, amplitudeplotvariables: dict):
    # 1. Setup & Defaults
    plotting = amplitudeplotvariables.get("plotting", {})
    figsize = plotting.get("figsize", (12, 7))
    
    # Use Semantic naming for the "Story"
    x_col, y_col = GC.WAVE_AMPLITUDE_INPUT, "mean_P3P2"
    err_col, hue_col = "std_P3P2", "WindCondition"
    
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=figsize)

    # 2. The "Baseline" - Gives the data a benchmark
    # If P3/P2 is a damping ratio, 1.0 is often a critical threshold.
    ax.axhline(1.0, color='black', linestyle='--', alpha=0.5, label='Unity (No Damping)')

    # 3. Plotting with Seaborn for better "Storytelling"
    # We use lineplot because it handles the aggregation and error bands elegantly
    palette = "viridis" # Or your custom WIND_COLORS
    
    sns.lineplot(
        data=df, x=x_col, y=y_col, hue=hue_col, 
        marker="o", err_style="bars", err_kws={'capsize': 3},
        palette=palette, ax=ax, markersize=8, linewidth=2
    )

    # 4. Adding a "Rug" to show data distribution
    sns.rugplot(data=df, x=x_col, hue=hue_col, ax=ax, alpha=0.5)

    # 5. Scientific Polish
    ax.set_title("Damping Ratio ($P_3/P_2$) vs. Dimensionless Wavenumber ($kL$)", 
                 fontsize=14, pad=15, loc='left', fontweight='bold')
    ax.set_xlabel(f"Dimensionless Wavenumber ${x_col}$", fontsize=12)
    ax.set_ylabel("Damping Ratio $\zeta$ (Mean $P_3/P_2$)", fontsize=12)
    
    # Use Log scale if kL covers multiple orders of magnitude
    # ax.set_xscale('log') 

    # Clean up legend
    ax.legend(title="Wind Condition", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 6. Annotation: The "Insight"
    # You can programmatically highlight the peak damping
    max_idx = df[y_col].idxmax()
    ax.annotate('Peak Damping', 
                xy=(df.loc[max_idx, x_col], df.loc[max_idx, y_col]),
                xytext=(20, 20), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color='red'))

    plt.tight_layout()
    plt.show()

# %%
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





    
# %% plot daming (basert på den fleksible plot_freq_spectrum)
# # begynte å kopiere den under... men starter med å bruke groupern til å hente ny amplitude. fra band_amplitudes
# def plot_swell_damping(
#     fft_dict: dict,
#     band_amp: pd.DataFrame,
#     meta_df: pd.DataFrame,
#     freqplotvar: dict, 
#     data_type: str = "fft"
# ) -> tuple:
#     """
#     Flexible swell plotter with extensive customization options.
    
#     Parameters
#     ----------
#     # fft_dict : dict
#         # Dictionary mapping file paths to FFT/PSD DataFrames
#     band_amp: pd. DataFrame
#         Kalkulert fra FFTen - 
        
#         Index(['path', 'Probe 1 swell amplitude', 'Probe 1 wind_waves amplitude',
#        'Probe 1 total amplitude', 'Probe 2 swell amplitude',
#        'Probe 2 wind_waves amplitude', 'Probe 2 total amplitude',
#        'Probe 3 swell amplitude', 'Probe 3 wind_waves amplitude',
#        'Probe 3 total amplitude', 'Probe 4 swell amplitude',
#        'Probe 4 wind_waves amplitude', 'Probe 4 total amplitude'],
#       dtype='object')

#     meta_df : pd.DataFrame
#         Metadata with columns: path, WindCondition, PanelCondition, etc.
#     freqplotvar : dict
#         Configuration with structure:
#         {
#             "filters": {"WaveFrequencyInput [Hz]": [value], ...},
#             "plotting": {
#                 "figsize": tuple or None,
#                 "facet_by": "probe" | "wind" | "panel" | None,
#                 "probes": [1, 2, 3, 4],
#                 "peaks": int or None,
#                 "logaritmic": bool,
#                 "legend": "inside" | "outside_right" | "below" | "above" | None,
#                 "max_points": int (default 100),
#                 "xlim": tuple or None,
#                 "grid": bool (default True),
#                 "show": bool (default True)
#             }
#         }
    
#     Returns
#     -------
#     tuple
#         (fig, axes) - matplotlib figure and axes objects
#     """
    
#     # ===== STYLE DEFINITIONS =====
#     PANEL_STYLES = {
#         "no": "solid",
#         "full": "dashed",
#         "reverse": "dashdot"
#     }
    
#     MARKER_STYLES = {
#         "full": "*",
#         "no": "<",
#         "lowest": ">"
#     }
#     #TODO: vurdere å markere ulike prober.. 
#     #TODO: videre, fikse legend til å være mer beskrivende
#     # ===== EXTRACT CONFIGURATION =====
#     plotting = freqplotvar.get("plotting", {})
    
#     facet_by = plotting.get("facet_by", None)  # None, 'probe', 'wind', 'panel'
#     probes = plotting.get("probes", [1])
#     if not isinstance(probes, (list, tuple)):
#         probes = [probes]
    
#     n_peaks = plotting.get("peaks", None)
#     log_scale = plotting.get("logaritmic", False)
#     max_points = plotting.get("max_points", 120)
#     legend_position = plotting.get("legend", "outside_right")
#     show_grid = plotting.get("grid", True)
#     show_plot = plotting.get("show", True)
#     xlim = plotting.get("xlim", (0, 10))
#     linewidth = plotting.get("linewidth", 1.0)
#     fontsize = 7
    
#     # Extract base frequency for tick locators
#     #not-todo: eventuelt gjøre om denne til å hente fra meta-sel, dersom man skulle ville hatt flere frekvenser men det blir kanksje dumt for fft'en.
#     base_freq_val = freqplotvar.get("filters", {}).get("WaveFrequencyInput [Hz]")
#     base_freq = None
#     if isinstance(base_freq_val, (list, tuple, np.ndarray, pd.Series)):
#         base_freq = float(base_freq_val[0]) if len(base_freq_val) > 0 else None
#     elif base_freq_val is not None:
#         base_freq = float(base_freq_val)
#     use_locators = base_freq is not None and base_freq > 0
    
#     if data_type.lower() == "psd":
#        col_prefix = "Pxx"
#        ylabel = "PSD"
#     else:
#        col_prefix = "FFT"
#        ylabel = "FFT"
    
#     # ===== DETERMINE FACET STRUCTURE =====
#     if facet_by == "probe":
#         facet_groups = list(probes)
#         facet_labels = [f"Probe {p}" for p in facet_groups]
#     elif facet_by == "wind":
#         facet_groups = list(pd.unique(meta_df["WindCondition"]))
#         facet_labels = [f"Wind: {w}" for w in facet_groups]
#     elif facet_by == "panel":
#         facet_groups = list(pd.unique(meta_df["PanelCondition"]))
#         facet_labels = [f"Panel: {p}" for p in facet_groups]
#     else:
#         facet_groups = [None]
#         facet_labels = ["All Data"]
    
#     n_facets = len(facet_groups)
    
#     # ===== CREATE FIGURE =====
#     default_figsize = (12, 4 * n_facets) if n_facets > 1 else (18,10)
#     figsize = plotting.get("figsize") or default_figsize

#     fig, axes = plt.subplots(
#         n_facets, 
#         figsize=figsize,
#         sharex=True,
#         squeeze=False,
#         dpi=120
#     )
#     axes = axes.flatten()  # Always work with 1D array
    
#     # ===== PLOTTING LOOP =====
#     for facet_idx, (group, facet_label) in enumerate(zip(facet_groups, facet_labels)):
#         ax = axes[facet_idx]
        
#         # Filter data for this facet
#         if facet_by == "wind":
#             subset = meta_df[meta_df["WindCondition"] == group]
#         elif facet_by == "panel":
#             subset = meta_df[meta_df["PanelCondition"] == group]
#         else:
#             subset = meta_df
        
#         # Plot each row in the subset
#         for _, row in subset.iterrows():
#             path = row["path"]
            
#             if path not in fft_dict:
#                 continue
            
#             df_fft = fft_dict[path]
            
#             # Extract styling information
#             windcond = row.get("WindCondition", "unknown")
#             colla = WIND_COLORS.get(windcond, "black")
#             panelcond = row.get("PanelCondition", "unknown")
#             linjestil = PANEL_STYLES.get(panelcond, "solid")
#             peak_marker = MARKER_STYLES.get(windcond, ".")
            
#             # Generate label
#             label_base = _make_label(row) if "_make_label" in dir() else f"{windcond}_{panelcond}"
            
#             # Determine which probes to plot for this facet
#             if facet_by == "probe":
#                 probes_to_plot = [group]  # Only plot the faceted probe
#             else:
#                 probes_to_plot = probes  # Plot all requested probes
            
#             # Plot each probe
#             for probe_num in probes_to_plot:
#                 col = f"{col_prefix} {probe_num}"
                
#                 if col not in df_fft:
#                     continue
                
#                 # Extract data
#                 y = df_fft[col].dropna().iloc[:max_points]
#                 if y.empty:
#                     continue
                
#                 x = y.index.values
                
#                 # Create label for this line
#                 if facet_by == "probe":
#                     plot_label = label_base
#                 elif len(probes_to_plot) > 1:
#                     plot_label = f"{label_base}_P{probe_num}"
#                 else:
#                     plot_label = label_base
                
#                 # Plot line
#                 ax.plot(
#                     x, y.values,
#                     linewidth=linewidth,
#                     label=plot_label,
#                     linestyle=linjestil,
#                     color=colla,
#                     antialiased=False #merk
#                 )
                
#                 # Plot peaks if requested
#                 if n_peaks and n_peaks > 0:
#                     vals = y.values
#                     top_idx_local = _top_k_indices(vals, n_peaks)
#                     ax.scatter(
#                         x[top_idx_local],
#                         vals[top_idx_local],
#                         color=colla,
#                         s=80,
#                         zorder=5,
#                         marker=peak_marker,
#                         edgecolors=None,#denna er visstnok dyr
#                         linewidths=0.7
#                     )
        
#         # ===== FORMATTING FOR THIS FACET =====
        
#         # Title
#         if facet_label:
#             ax.set_title(facet_label, fontsize=fontsize, fontweight='normal')
        
#         # Y-axis
#         ax.set_ylabel(ylabel, fontsize=fontsize)
#         if log_scale:
#             ax.set_yscale('log')
        
#         # X-axis limits
#         if xlim:
#             ax.set_xlim(xlim)
        
#         # Tick locators
#         if use_locators:
#             ax.xaxis.set_minor_locator(ticker.MultipleLocator(base_freq))
#             ax.xaxis.set_major_locator(ticker.MultipleLocator(2 * base_freq))
#         else:
#             ax.xaxis.set_major_locator(ticker.MaxNLocator(10))
#             ax.yaxis.set_major_locator(ticker.MaxNLocator(8))
#         ax.tick_params(axis='both', labelsize=8)
        
#         # ax.set_frame_on(False) didnt help
#         # Grid
#         if show_grid:
#             ax.grid(which='major', linestyle='--', alpha=0.6)
#             ax.grid(which='minor', linestyle='-.', alpha=0.3)
        
#         # Legend
#         _apply_legend_3(ax, freqplotvar)
    
#     # ===== FINAL TOUCHES =====
    
#     # X-label only on bottom plot
#     axes[-1].set_xlabel(col_prefix, fontsize=fontsize)
    
#     draw_anchored_text(ax,"tekstboks")
    
#     plt.tight_layout()
    
#     if show_plot:
#         plt.show()
    
#     return fig, axes
# %% fleksibel fft-plotter
def plot_frequency_spectrum(
    fft_dict: dict,
    meta_df: pd.DataFrame,
    freqplotvar: dict, 
    data_type: str = "fft"
) -> tuple:
    """
    Flexible frequency spectrum plotter with extensive customization options.
    
    Parameters
    ----------
    fft_dict : dict
        Dictionary mapping file paths to FFT/PSD DataFrames
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
        "reverse": "dashdot"
    }
    
    MARKER_STYLES = {
        "full": "*",
        "no": "<",
        "lowest": ">"
    }
    #TODO: vurdere å markere ulike prober.. 
    #TODO: videre, fikse legend til å være mer beskrivende
    # ===== EXTRACT CONFIGURATION =====
    plotting = freqplotvar.get("plotting", {})
    
    facet_by = plotting.get("facet_by", None)  # None, 'probe', 'wind', 'panel'
    probes = plotting.get("probes", [1])
    if not isinstance(probes, (list, tuple)):
        probes = [probes]
    
    n_peaks = plotting.get("peaks", None)
    log_scale = plotting.get("logaritmic", False)
    max_points = plotting.get("max_points", 120)
    legend_position = plotting.get("legend", "outside_right")
    show_grid = plotting.get("grid", True)
    show_plot = plotting.get("show", True)
    xlim = plotting.get("xlim", (0, 10))
    linewidth = plotting.get("linewidth", 1.0)
    fontsize = 7
    
    # Extract base frequency for tick locators
    #not-todo: eventuelt gjøre om denne til å hente fra meta-sel, dersom man skulle ville hatt flere frekvenser men det blir kanksje dumt for fft'en.
    base_freq_val = freqplotvar.get("filters", {}).get("WaveFrequencyInput [Hz]")
    base_freq = None
    if isinstance(base_freq_val, (list, tuple, np.ndarray, pd.Series)):
        base_freq = float(base_freq_val[0]) if len(base_freq_val) > 0 else None
    elif base_freq_val is not None:
        base_freq = float(base_freq_val)
    use_locators = base_freq is not None and base_freq > 0
    
    if data_type.lower() == "psd":
       col_prefix = "Pxx"
       ylabel = "PSD"
    else:
       col_prefix = "FFT"
       ylabel = "FFT"
    
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
    default_figsize = (12, 4 * n_facets) if n_facets > 1 else (18,10)
    figsize = plotting.get("figsize") or default_figsize

    fig, axes = plt.subplots(
        n_facets, 
        figsize=figsize,
        sharex=True,
        squeeze=False,
        dpi=120
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
                col = f"{col_prefix} {probe_num}"
                
                if col not in df_fft:
                    continue
                df_fft_pos = get_positive_spectrum(df_fft)
                # Extract data
                y = df_fft_pos[col].dropna().iloc[:max_points]

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
                    linewidth=linewidth,
                    label=plot_label,
                    linestyle=linjestil,
                    color=colla,
                    antialiased=False #merk
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
            ax.set_title(facet_label, fontsize=fontsize, fontweight='normal')
        
        # Y-axis
        ax.set_ylabel(ylabel, fontsize=fontsize)
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
        ax.tick_params(axis='both', labelsize=8)
        
        # ax.set_frame_on(False) didnt help
        # Grid
        if show_grid:
            ax.grid(which='major', linestyle='--', alpha=0.6)
            ax.grid(which='minor', linestyle='-.', alpha=0.3)
        
        # Legend
        _apply_legend_3(ax, freqplotvar)
    
    # ===== FINAL TOUCHES =====
    
    # X-label only on bottom plot
    axes[-1].set_xlabel(col_prefix, fontsize=fontsize)
    
    draw_anchored_text(ax,"tekstboks")
    
    plt.tight_layout()
    
    if show_plot:
        plt.show()
    
    return fig, axes

# %% ifft plot - dekomponert
def i_plot_reconstructed(fft_dict: dict, 
                               filtrert_frequencies: pd.DataFrame,
                               freqplotvariables: dict,
                               data_type="fft") -> tuple: 
    meta_df = filtrert_frequencies.copy()
    plotting = freqplotvariables.get("plotting", {})
    
    facet_by = plotting.get("facet_by", None)  # None, 'probe', 'wind', 'panel'
    probes = plotting.get("probes", [1])
    if not isinstance(probes, (list, tuple)):
        probes = [probes]
    
    n_peaks = plotting.get("peaks", None)
    log_scale = plotting.get("logaritmic", False)
    max_points = plotting.get("max_points", 120)
    legend_position = plotting.get("legend", "outside_right")
    show_grid = plotting.get("grid", True)
    show_plot = plotting.get("show", True)
    xlim = plotting.get("xlim", (0, 10))
    linewidth = plotting.get("linewidth", 1.0)
    fontsize = 7
    # Extract base frequency for tick locators
    #not-todo: eventuelt gjøre om denne til å hente fra meta-sel, dersom man skulle ville hatt flere frekvenser men det blir kanksje dumt for fft'en.
    base_freq_val = freqplotvariables.get("filters", {}).get("WaveFrequencyInput [Hz]")
    base_freq = None
    if isinstance(base_freq_val, (list, tuple, np.ndarray, pd.Series)):
        base_freq = float(base_freq_val[0]) if len(base_freq_val) > 0 else None
    elif base_freq_val is not None:
        base_freq = float(base_freq_val)
    use_locators = base_freq is not None and base_freq > 0
    
    if data_type.lower() == "psd":
       col_prefix = "Pxx"
       ylabel = "PSD"
    else:
       col_prefix = "FFT"
       ylabel = "FFT"
    
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
    default_figsize = (12, 4 * n_facets) if n_facets > 1 else (18,10)
    figsize = plotting.get("figsize") or default_figsize    
    
    fig, axes = plt.subplots(
        n_facets, 
        figsize=figsize,
        sharex=True,
        squeeze=False,
        dpi=120
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
                col = f"{col_prefix} {probe_num}"
                
                if col not in df_fft:
                    continue
                
                # Extract data
                y = df_fft[col].dropna()
                f = y.values
                window = SIGNAL.FFT_FREQUENCY_WINDOW #ønsker å lete ved min forventa frekvens.
                target_freq = row[GC.WAVE_FREQUENCY_INPUT]
                mask = (f >= target_freq - window) & (f <= target_freq + window)
                # #reconstruere
                y_single_f = y[mask].values
                y_rest_f = y[~mask].values
                print(f"Path: {path} | Probe: {probe_num} | Target freq: {target_freq:.3f} Hz | Window: ±{window:.3f} Hz")
                print(f"Frequency range in data: {f.min():.3f} – {f.max():.3f} Hz (n={len(f)} bins)")
                print(f"Points selected for single wave: {mask.sum()}  (mask range: {f[mask].min() if mask.any() else '—'} – {f[mask].max() if mask.any() else '—'})")
                
                if mask.sum() == 0:
                    print("→ No frequencies matched → skipping ifft")
                    continue   # or handle gracefully
                # if y.empty:
                    # continue
                
                x = y.index.values
                y_single_wave = np.fft.ifft(y_single_f)
                y_rest_wave = np.fft.ifft(y_rest_f)
                # Create label for this line
                if facet_by == "probe":
                    plot_label = label_base
                elif len(probes_to_plot) > 1:
                    plot_label = f"{label_base}_P{probe_num}"
                else:
                    plot_label = label_base
                
                # Plot line
                ax.plot(
                    x, y_single_f,
                    linewidth=linewidth,
                    label=plot_label,
                    linestyle=linjestil,
                    color=colla,
                    antialiased=False #merk
                )
                # Plot line
                ax.plot(
                    x, y_rest_f,
                    linewidth=linewidth,
                    label=plot_label,
                    linestyle="-.",
                    color=colla,
                    antialiased=False #merk
                )
                
                # Plot peaks if requested
                if n_peaks and n_peaks > 0:
                    vals = y
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
            ax.set_title(facet_label, fontsize=fontsize, fontweight='normal')
        
        # Y-axis
        ax.set_ylabel(ylabel, fontsize=fontsize)
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
        ax.tick_params(axis='both', labelsize=8)
        
        # ax.set_frame_on(False) didnt help
        # Grid
        if show_grid:
            ax.grid(which='major', linestyle='--', alpha=0.6)
            ax.grid(which='minor', linestyle='-.', alpha=0.3)
        
        # Legend
        _apply_legend_3(ax, freqplotvariables)
    
    # ===== FINAL TOUCHES =====
   
    return fig, axes


# %%

def old_claude_plot_reconstructed(
    fft_dict: Dict[str, pd.DataFrame],
    filtrert_frequencies: pd.DataFrame,
    freqplotvariables: dict,
    data_type: str = "fft"
) -> Tuple[Optional[plt.Figure], Optional[np.ndarray]]:
    """
    Plot reconstructed time-domain signals: isolated swell frequency vs everything else.
    
    Shows:
    - Pure swell component (target frequency only)
    - Everything else (wind + noise)
    """
    meta_df = filtrert_frequencies.copy()
    plotting = freqplotvariables.get("plotting", {})
    
    # Configuration
    facet_by = plotting.get("facet_by", None)
    probes = plotting.get("probes", [1])
    probes = [probes] if not isinstance(probes, (list, tuple)) else probes
    
    show_grid = plotting.get("grid", True)
    show_plot = plotting.get("show", True)
    linewidth = plotting.get("linewidth", 1.2)
    fontsize = 9
    
    # Frequency window for swell extraction (tight window around target)
    window = 0.05  # Hz - captures just the swell peak
    
    ylabel = "Amplitude"
    
    # Determine facets
    if facet_by == "probe":
        facet_groups = probes
        facet_labels = [f"Probe {p}" for p in facet_groups]
    elif facet_by == "wind":
        facet_groups = pd.unique(meta_df["WindCondition"]).tolist()
        facet_labels = [f"Wind: {w}" for w in facet_groups]
    elif facet_by == "panel":
        facet_groups = pd.unique(meta_df["PanelCondition"]).tolist()
        facet_labels = [f"Panel: {p}" for p in facet_groups]
    else:
        facet_groups = [None]
        facet_labels = ["All Data"]
    
    n_facets = len(facet_groups)
    
    # Create figure
    default_figsize = (16, 5 * n_facets) if n_facets > 1 else (16, 7)
    figsize = plotting.get("figsize") or default_figsize
    
    fig, axes = plt.subplots(
        n_facets, 1,
        figsize=figsize,
        sharex=False,
        squeeze=False,
        dpi=120
    )
    axes = axes.flatten()
    
    # Plotting loop
    for facet_idx, (group, facet_label) in enumerate(zip(facet_groups, facet_labels)):
        ax = axes[facet_idx]
        
        # Subset data for this facet
        if facet_by == "wind":
            subset = meta_df[meta_df["WindCondition"] == group]
        elif facet_by == "panel":
            subset = meta_df[meta_df["PanelCondition"] == group]
        else:
            subset = meta_df
        
        if len(subset) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(facet_label, fontsize=fontsize)
            continue
        
        for row_idx, row in subset.iterrows():
            path = row["path"]
            if path not in fft_dict:
                continue
            
            df_fft = fft_dict[path]
            
            # Styling
            windcond = row.get("WindCondition", "unknown")
            color = WIND_COLORS.get(windcond, "black")
            panelcond = row.get("PanelCondition", "unknown")
            linestyle = PANEL_STYLES.get(panelcond, "-")
            
            # Get target swell frequency
            target_freq = row.get(GC.WAVE_FREQUENCY_INPUT, None)
            if target_freq is None or target_freq <= 0:
                print(f"Skipping {Path(path).name}: invalid target frequency")
                continue
            
            # Label
            label_base = f"{windcond}/{panelcond}"
            
            # Probes to plot in this facet
            probes_to_plot = [group] if facet_by == "probe" else probes
            
            for probe_num in probes_to_plot:
                # ═══════════════════════════════════════════════
                # USE COMPLEX FFT COEFFICIENTS!
                # ═══════════════════════════════════════════════
                col = f"FFT {probe_num} complex"
                
                if col not in df_fft:
                    print(f"Column {col} not found, trying without 'complex'...")
                    col = f"FFT {probe_num}"
                    if col not in df_fft:
                        print(f"  → Still not found, skipping")
                        continue
                
                # Extract FULL FFT series
                fft_series = df_fft[col].dropna()
                
                if len(fft_series) == 0:
                    continue
                
                # Get frequency bins and complex FFT coefficients
                freq_bins = fft_series.index.values
                fft_complex = fft_series.values
                
                print(f"\n{'='*60}")
                print(f"Processing: {Path(path).stem} | Probe {probe_num}")
                print(f"Target swell frequency: {target_freq:.3f} Hz")
                print(f"Frequency range: {freq_bins.min():.2f} to {freq_bins.max():.2f} Hz")
                print(f"FFT length: {len(fft_complex)}")
                print(f"FFT dtype: {fft_complex.dtype}")
                
                # ═══════════════════════════════════════════════
                # Find bins near target frequency (both +/- for symmetry)
                # ═══════════════════════════════════════════════
                mask_target = np.abs(np.abs(freq_bins) - target_freq) <= window
                
                n_target = mask_target.sum()
                print(f"Swell bins found (±{window} Hz): {n_target}")
                
                if n_target == 0:
                    print(f"  → No frequency bins near {target_freq:.3f} Hz! Skipping.")
                    continue
                
                target_freqs = freq_bins[mask_target]
                print(f"  → Swell freq bins: {target_freqs}")
                
                # ═══════════════════════════════════════════════
                # Create two versions: swell-only and wind-only
                # ═══════════════════════════════════════════════
                fft_swell = np.zeros_like(fft_complex, dtype=complex)
                fft_swell[mask_target] = fft_complex[mask_target]
                
                fft_wind = fft_complex.copy()
                fft_wind[mask_target] = 0  # Zero out the swell
                
                # ═══════════════════════════════════════════════
                # Reconstruct time-domain signals
                # ═══════════════════════════════════════════════
                try:
                    signal_swell = np.fft.ifft(fft_swell)
                    signal_wind = np.fft.ifft(fft_wind)
                    
                    # Take real part (imaginary should be negligible)
                    signal_swell = np.real(signal_swell)
                    signal_wind = np.real(signal_wind)
                    
                    # Also reconstruct full signal for comparison
                    signal_full = np.fft.ifft(fft_complex)
                    signal_full = np.real(signal_full)
                    
                    print(f"  → Swell component: {len(signal_swell)} samples")
                    print(f"     Range: {signal_swell.min():.4f} to {signal_swell.max():.4f}")
                    print(f"     RMS: {np.sqrt(np.mean(signal_swell**2)):.4f}")
                    print(f"  → Wind component: {len(signal_wind)} samples")
                    print(f"     Range: {signal_wind.min():.4f} to {signal_wind.max():.4f}")
                    print(f"     RMS: {np.sqrt(np.mean(signal_wind**2)):.4f}")
                    
                except Exception as e:
                    print(f"  → IFFT failed: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                
                # ═══════════════════════════════════════════════
                # Create time axis
                # ═══════════════════════════════════════════════
                n_samples = len(signal_swell)
                # Sampling rate from frequency resolution
                df_freq = freq_bins[1] - freq_bins[0]
                sampling_rate = abs(df_freq * n_samples)
                time_axis = np.arange(n_samples) / sampling_rate
                
                print(f"  → Sampling rate: {sampling_rate:.1f} Hz")
                print(f"  → Duration: {time_axis[-1]:.2f} s")
                
                # ═══════════════════════════════════════════════
                # Plot reconstructed signals
                # ═══════════════════════════════════════════════
                plot_label = (
                    f"{label_base}_P{probe_num}" if len(probes_to_plot) > 1 
                    else label_base
                )
                
                # Plot swell (thick, prominent)
                ax.plot(
                    time_axis, signal_swell,
                    linewidth=linewidth * 1.5,
                    label=f"{plot_label} (swell {target_freq:.2f}Hz)",
                    linestyle=linestyle,
                    color=color,
                    alpha=0.9,
                    zorder=3
                )
                
                # Plot wind/rest (thinner, lighter)
                ax.plot(
                    time_axis, signal_wind,
                    linewidth=linewidth * 0.8,
                    label=f"{plot_label} (wind+noise)",
                    linestyle=":",
                    color=color,
                    alpha=0.5,
                    zorder=2
                )
                
                # Optionally plot full signal for reference (very light)
                # ax.plot(
                #     time_axis, signal_full,
                #     linewidth=linewidth * 0.5,
                #     label=f"{plot_label} (full)",
                #     linestyle="-",
                #     color='gray',
                #     alpha=0.2,
                #     zorder=1
                # )
        
        # ═══════════════════════════════════════════════
        # Facet formatting
        # ═══════════════════════════════════════════════
        ax.set_title(facet_label, fontsize=fontsize + 2, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        
        ax.tick_params(axis='both', labelsize=fontsize - 1)
        
        if show_grid:
            ax.grid(which='major', linestyle='--', alpha=0.3, linewidth=0.8)
            ax.grid(which='minor', linestyle=':', alpha=0.15, linewidth=0.5)
            ax.minorticks_on()
        
        # Legend
        handles, labels = ax.get_legend_handles_labels()
        if len(handles) > 0:
            ax.legend(
                loc='upper right',
                fontsize=fontsize - 1,
                framealpha=0.95,
                ncol=1 if len(handles) <= 8 else 2
            )
        
        # Add zero line for reference
        ax.axhline(0, color='black', linewidth=0.5, alpha=0.3, zorder=0)
    
    # ═══════════════════════════════════════════════
    # Final touches
    # ═══════════════════════════════════════════════
    plt.suptitle('Signal Decomposition: Swell vs Wind Components', 
                 fontsize=fontsize + 4, fontweight='bold', y=0.995)
    fig.tight_layout()
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return fig, axes

# %%
def plot_reconstructed(
    fft_dict: Dict[str, pd.DataFrame],
    filtrert_frequencies: pd.DataFrame,
    freqplotvariables: dict,
    data_type: str = "fft"
) -> Tuple[Optional[plt.Figure], Optional[np.ndarray]]:
    """
    Plot reconstructed swell vs wind for a SINGLE experiment.
    Can facet by probe to show P2 and P3 in separate subplots.
    """
    meta_df = filtrert_frequencies.copy()
    plotting = freqplotvariables.get("plotting", {})
    
    # Color/style mappings
    WIND_COLORS = {
        "full": "red",
        "no": "blue",
        "lowest": "green"
    }
    
    PANEL_STYLES = {
        "no": "-",
        "full": "--",
        "reverse": "-."
    }
    
    # Configuration
    facet_by = plotting.get("facet_by", None)
    probes = plotting.get("probes", [1])
    probes = [probes] if not isinstance(probes, (list, tuple)) else probes
    
    show_grid = plotting.get("grid", True)
    show_plot = plotting.get("show_plot", True)
    linewidth = plotting.get("linewidth", 1.2)
    fontsize = 9
    
    show_full_signal = plotting.get("show_full_signal", False)
    dual_yaxis = plotting.get("dual_yaxis", True)
    show_amplitude_stats = plotting.get("show_amplitude_stats", True)
    
    # Validate: should have exactly ONE experiment
    if len(fft_dict) == 0:
        print("ERROR: fft_dict is empty!")
        return None, None
    
    if len(fft_dict) > 1:
        print(f"WARNING: fft_dict contains {len(fft_dict)} experiments. Only plotting the first one.")
        print("         To plot multiple, call this function in a loop.")
    
    # Get the single experiment
    path = list(fft_dict.keys())[0]
    df_fft = fft_dict[path]
    
    # Get metadata
    path_meta = meta_df[meta_df["path"] == path]
    
    if len(path_meta) == 0:
        print(f"ERROR: No metadata found for {path}")
        return None, None
    
    row = path_meta.iloc[0]
    
    # Styling
    windcond = row.get("WindCondition", "unknown")
    color_swell = WIND_COLORS.get(windcond, "black")
    color_wind = "darkred" if dual_yaxis else "orange"
    color_full = "gray"
    panelcond = row.get("PanelCondition", "unknown")
    linestyle = PANEL_STYLES.get(panelcond, "-")
    
    # Get target frequency
    target_freq = row.get(GC.WAVE_FREQUENCY_INPUT, None)
    if target_freq is None or target_freq <= 0:
        print(f"ERROR: Invalid target frequency: {target_freq}")
        return None, None
    
    # Storage for amplitude comparison
    amplitude_comparison = []
    
    # ═══════════════════════════════════════════════
    # Determine subplot layout based on facet_by
    # ═══════════════════════════════════════════════
    if facet_by == "probe":
        n_subplots = len(probes)
        subplot_labels = [f"Probe {p}" for p in probes]
        facet_mode = "probe"
    else:
        n_subplots = 1
        subplot_labels = [f"{Path(path).stem}"]
        facet_mode = "single"
    
    # Create figure
    figsize = plotting.get("figsize", (16, 5 * n_subplots) if n_subplots > 1 else (16, 7))
    fig, axes = plt.subplots(n_subplots, 1, figsize=figsize, squeeze=False, dpi=120)
    axes = axes.flatten()
    
    # ═══════════════════════════════════════════════
    # Plot each subplot
    # ═══════════════════════════════════════════════
    for subplot_idx in range(n_subplots):
        ax_swell = axes[subplot_idx]
        
        if dual_yaxis:
            ax_wind = ax_swell.twinx()
        else:
            ax_wind = ax_swell
        
        # Determine which probe(s) to plot in this subplot
        if facet_mode == "probe":
            probes_to_plot = [probes[subplot_idx]]
            subplot_title = subplot_labels[subplot_idx]
        else:
            probes_to_plot = probes
            subplot_title = f"{windcond} wind / {panelcond} panel / {target_freq:.3f} Hz"
        
        # Process each probe for this subplot
        for probe_num in probes_to_plot:
            col = f"FFT {probe_num} complex"
            
            if col not in df_fft:
                col = f"FFT {probe_num}"
                if col not in df_fft:
                    print(f"Skipping probe {probe_num}: column {col} not found")
                    continue
            
            fft_series = df_fft[col].dropna()
            if len(fft_series) == 0:
                print(f"Skipping probe {probe_num}: empty data")
                continue
            
            freq_bins = fft_series.index.values
            fft_complex = fft_series.values
            
            print(f"\n{'='*60}")
            print(f"Processing: {Path(path).stem}")
            print(f"Probe: {probe_num}")
            print(f"Target swell: {target_freq:.3f} Hz")
            
            # Reorder FFT from sorted to fftfreq order
            N = len(fft_complex)
            df_freq = freq_bins[1] - freq_bins[0]
            sampling_rate = abs(df_freq * N)
            correct_fftfreq_order = np.fft.fftfreq(N, d=1/sampling_rate)
            
            fft_reordered = np.zeros(N, dtype=complex)
            for i, target_f in enumerate(correct_fftfreq_order):
                closest_idx = np.argmin(np.abs(freq_bins - target_f))
                if np.abs(freq_bins[closest_idx] - target_f) < 1e-6:
                    fft_reordered[i] = fft_complex[closest_idx]
            
            # Reconstruct full signal
            signal_full = np.real(np.fft.ifft(fft_reordered))
            time_axis = np.arange(N) / sampling_rate
            
            # Find target frequency
            pos_mask = correct_fftfreq_order > 0
            pos_freqs = correct_fftfreq_order[pos_mask]
            
            closest_pos_idx = np.argmin(np.abs(pos_freqs - target_freq))
            actual_freq = pos_freqs[closest_pos_idx]
            
            peak_idx = np.where(np.abs(correct_fftfreq_order - actual_freq) < 1e-6)[0][0]
            mirror_idx = np.where(np.abs(correct_fftfreq_order + actual_freq) < 1e-6)[0][0]
            
            # Create swell-only FFT
            fft_swell = np.zeros_like(fft_reordered, dtype=complex)
            fft_swell[peak_idx] = fft_reordered[peak_idx]
            fft_swell[mirror_idx] = fft_reordered[mirror_idx]
            
            # Reconstruct
            signal_swell = np.real(np.fft.ifft(fft_swell))
            signal_wind = signal_full - signal_swell
            
            # Amplitude analysis
            full_peak = np.max(np.abs(signal_full))
            swell_peak = np.max(np.abs(signal_swell))
            wind_peak = np.max(np.abs(signal_wind))
            
            full_p2p = np.max(signal_full) - np.min(signal_full)
            swell_p2p = np.max(signal_swell) - np.min(signal_swell)
            
            full_rms = np.sqrt(np.mean(signal_full**2))
            swell_rms = np.sqrt(np.mean(signal_swell**2))
            wind_rms = np.sqrt(np.mean(signal_wind**2))
            
            peak_diff = full_peak - swell_peak
            peak_ratio = (full_peak / swell_peak) if swell_peak > 0 else np.nan
            peak_percent = (peak_diff / full_peak * 100) if full_peak > 0 else np.nan
            
            # Store comparison
            amplitude_comparison.append({
                'experiment': Path(path).stem,
                'probe': probe_num,
                'wind': windcond,
                'panel': panelcond,
                'target_freq': target_freq,
                'full_peak': full_peak,
                'swell_peak': swell_peak,
                'wind_peak': wind_peak,
                'full_rms': full_rms,
                'swell_rms': swell_rms,
                'wind_rms': wind_rms,
                'peak_diff': peak_diff,
                'peak_ratio': peak_ratio,
                'peak_percent_diff': peak_percent,
                'full_p2p': full_p2p,
                'swell_p2p': swell_p2p
            })
            
            if show_amplitude_stats:
                print(f"  Peak: Full={full_peak:.4f}, Swell={swell_peak:.4f}, Diff={peak_percent:+.2f}%")
                print(f"  RMS:  Full={full_rms:.4f}, Swell={swell_rms:.4f}, Wind={wind_rms:.4f}")
            
            # Plot labels
            if facet_mode == "probe":
                # When faceting by probe, no need for probe number in label
                label_prefix = ""
            else:
                # When all probes on one plot, need probe labels
                label_prefix = f"P{probe_num} "
            
            # Full signal
            if show_full_signal:
                ax_swell.plot(
                    time_axis, signal_full,
                    linewidth=linewidth * 0.7,
                    label=f"{label_prefix}full",
                    linestyle="-",
                    color=color_full,
                    alpha=0.4,
                    zorder=1
                )
            
            # Swell component
            ax_swell.plot(
                time_axis, signal_swell,
                linewidth=linewidth * 1.5,
                label=f"{label_prefix}swell ({actual_freq:.2f}Hz)",
                linestyle=linestyle,
                color=color_swell,
                alpha=0.9,
                zorder=3
            )
            
            # Wind component
            ax_wind.plot(
                time_axis, signal_wind,
                linewidth=linewidth * (1.2 if dual_yaxis else 1.0),
                label=f"{label_prefix}wind",
                linestyle=":" if dual_yaxis else "--",
                color=color_wind,
                alpha=0.7 if dual_yaxis else 0.8,
                zorder=2
            )
        
        # ═══════════════════════════════════════════════
        # Formatting for this subplot
        # ═══════════════════════════════════════════════
        ax_swell.set_title(subplot_title, fontsize=fontsize + 2, fontweight='bold', pad=15)
        ax_swell.set_xlabel('Time (s)', fontsize=fontsize)
        
        if dual_yaxis:
            ax_swell.set_ylabel('Swell Amplitude', fontsize=fontsize, color=color_swell)
            ax_wind.set_ylabel('Wind+Noise Amplitude', fontsize=fontsize, color=color_wind)
            ax_swell.tick_params(axis='y', labelcolor=color_swell, labelsize=fontsize - 1)
            ax_wind.tick_params(axis='y', labelcolor=color_wind, labelsize=fontsize - 1)
        else:
            ax_swell.set_ylabel('Amplitude', fontsize=fontsize)
            ax_swell.tick_params(axis='y', labelsize=fontsize - 1)
        
        ax_swell.tick_params(axis='x', labelsize=fontsize - 1)
        
        if show_grid:
            ax_swell.grid(which='major', linestyle='--', alpha=0.3, linewidth=0.8)
            ax_swell.grid(which='minor', linestyle=':', alpha=0.15, linewidth=0.5)
            ax_swell.minorticks_on()
        
        # Legend
        lines_swell, labels_swell = ax_swell.get_legend_handles_labels()
        
        if dual_yaxis:
            lines_wind, labels_wind = ax_wind.get_legend_handles_labels()
            all_lines = lines_swell + lines_wind
            all_labels = labels_swell + labels_wind
        else:
            all_lines = lines_swell
            all_labels = labels_swell
        
        if all_lines:
            ax_swell.legend(
                all_lines, all_labels,
                loc='upper right',
                fontsize=fontsize,
                framealpha=0.95
            )
        
        ax_swell.axhline(0, color='black', linewidth=0.5, alpha=0.3, zorder=0)
        if dual_yaxis:
            ax_wind.axhline(0, color='darkred', linewidth=0.5, alpha=0.2, zorder=0)
    
    # Overall title
    overall_title = f"{Path(path).stem}\n{windcond} wind / {panelcond} panel / {target_freq:.3f} Hz"
    plt.suptitle(overall_title, fontsize=fontsize + 3, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    # Print summary
    if amplitude_comparison and show_amplitude_stats:
        print(f"\n{'='*60}")
        print("AMPLITUDE COMPARISON")
        print(f"{'='*60}")
        df_comparison = pd.DataFrame(amplitude_comparison)
        print(df_comparison.to_string(index=False))
        print(f"{'='*60}\n")
    
    return fig, axes

def non_facet_plot_reconstructed(
    fft_dict: Dict[str, pd.DataFrame],
    filtrert_frequencies: pd.DataFrame,
    freqplotvariables: dict,
    data_type: str = "fft"
) -> Tuple[Optional[plt.Figure], Optional[np.ndarray]]:
    """
    Plot reconstructed swell vs wind for a SINGLE experiment.
    
    Input should contain exactly ONE experiment.
    For multiple experiments, call this function multiple times in a loop.
    """
    meta_df = filtrert_frequencies.copy()
    plotting = freqplotvariables.get("plotting", {})
    
    # Color/style mappings
    WIND_COLORS = {
        "full": "red",
        "no": "blue",
        "lowest": "green"
    }
    
    PANEL_STYLES = {
        "no": "-",
        "full": "--",
        "reverse": "-."
    }
    
    # Configuration
    probes = plotting.get("probes", [1])
    probes = [probes] if not isinstance(probes, (list, tuple)) else probes
    
    show_grid = plotting.get("grid", True)
    show_plot = plotting.get("show_plot", True)
    linewidth = plotting.get("linewidth", 1.2)
    fontsize = 9
    
    show_full_signal = plotting.get("show_full_signal", False)
    dual_yaxis = plotting.get("dual_yaxis", True)
    show_amplitude_stats = plotting.get("show_amplitude_stats", True)
    
    # Validate: should have exactly ONE experiment
    if len(fft_dict) == 0:
        print("ERROR: fft_dict is empty!")
        return None, None
    
    if len(fft_dict) > 1:
        print(f"WARNING: fft_dict contains {len(fft_dict)} experiments. Only plotting the first one.")
        print("         To plot multiple, call this function in a loop.")
    
    # Get the single experiment
    path = list(fft_dict.keys())[0]
    df_fft = fft_dict[path]
    
    # Get metadata
    path_meta = meta_df[meta_df["path"] == path]
    
    if len(path_meta) == 0:
        print(f"ERROR: No metadata found for {path}")
        return None, None
    
    row = path_meta.iloc[0]
    
    # Styling
    windcond = row.get("WindCondition", "unknown")
    color_swell = WIND_COLORS.get(windcond, "black")
    color_wind = "darkred" if dual_yaxis else "orange"
    color_full = "gray"
    panelcond = row.get("PanelCondition", "unknown")
    linestyle = PANEL_STYLES.get(panelcond, "-")
    
    # Get target frequency
    target_freq = row.get(GC.WAVE_FREQUENCY_INPUT, None)
    if target_freq is None or target_freq <= 0:
        print(f"ERROR: Invalid target frequency: {target_freq}")
        return None, None
    
    # Storage for amplitude comparison
    amplitude_comparison = []
    
    # Create figure - SINGLE PLOT
    figsize = plotting.get("figsize", (16, 7))
    fig, ax_swell = plt.subplots(1, 1, figsize=figsize, dpi=120)
    
    if dual_yaxis:
        ax_wind = ax_swell.twinx()
    else:
        ax_wind = ax_swell
    
    # Process each probe
    for probe_num in probes:
        col = f"FFT {probe_num} complex"
        
        if col not in df_fft:
            col = f"FFT {probe_num}"
            if col not in df_fft:
                print(f"Skipping probe {probe_num}: column {col} not found")
                continue
        
        fft_series = df_fft[col].dropna()
        if len(fft_series) == 0:
            print(f"Skipping probe {probe_num}: empty data")
            continue
        
        freq_bins = fft_series.index.values
        fft_complex = fft_series.values
        
        print(f"\n{'='*60}")
        print(f"Processing: {Path(path).stem}")
        print(f"Probe: {probe_num}")
        print(f"Target swell: {target_freq:.3f} Hz")
        
        # Reorder FFT from sorted to fftfreq order
        N = len(fft_complex)
        df_freq = freq_bins[1] - freq_bins[0]
        sampling_rate = abs(df_freq * N)
        correct_fftfreq_order = np.fft.fftfreq(N, d=1/sampling_rate)
        
        fft_reordered = np.zeros(N, dtype=complex)
        for i, target_f in enumerate(correct_fftfreq_order):
            closest_idx = np.argmin(np.abs(freq_bins - target_f))
            if np.abs(freq_bins[closest_idx] - target_f) < 1e-6:
                fft_reordered[i] = fft_complex[closest_idx]
        
        # Reconstruct full signal
        signal_full = np.real(np.fft.ifft(fft_reordered))
        time_axis = np.arange(N) / sampling_rate
        
        # Find target frequency
        pos_mask = correct_fftfreq_order > 0
        pos_freqs = correct_fftfreq_order[pos_mask]
        
        closest_pos_idx = np.argmin(np.abs(pos_freqs - target_freq))
        actual_freq = pos_freqs[closest_pos_idx]
        
        peak_idx = np.where(np.abs(correct_fftfreq_order - actual_freq) < 1e-6)[0][0]
        mirror_idx = np.where(np.abs(correct_fftfreq_order + actual_freq) < 1e-6)[0][0]
        
        # Create swell-only FFT
        fft_swell = np.zeros_like(fft_reordered, dtype=complex)
        fft_swell[peak_idx] = fft_reordered[peak_idx]
        fft_swell[mirror_idx] = fft_reordered[mirror_idx]
        
        # Reconstruct
        signal_swell = np.real(np.fft.ifft(fft_swell))
        signal_wind = signal_full - signal_swell
        
        # Amplitude analysis
        full_peak = np.max(np.abs(signal_full))
        swell_peak = np.max(np.abs(signal_swell))
        wind_peak = np.max(np.abs(signal_wind))
        
        full_p2p = np.max(signal_full) - np.min(signal_full)
        swell_p2p = np.max(signal_swell) - np.min(signal_swell)
        
        full_rms = np.sqrt(np.mean(signal_full**2))
        swell_rms = np.sqrt(np.mean(signal_swell**2))
        wind_rms = np.sqrt(np.mean(signal_wind**2))
        
        peak_diff = full_peak - swell_peak
        peak_ratio = (full_peak / swell_peak) if swell_peak > 0 else np.nan
        peak_percent = (peak_diff / full_peak * 100) if full_peak > 0 else np.nan
        
        # Store comparison
        amplitude_comparison.append({
            'experiment': Path(path).stem,
            'probe': probe_num,
            'wind': windcond,
            'panel': panelcond,
            'target_freq': target_freq,
            'full_peak': full_peak,
            'swell_peak': swell_peak,
            'wind_peak': wind_peak,
            'full_rms': full_rms,
            'swell_rms': swell_rms,
            'wind_rms': wind_rms,
            'peak_diff': peak_diff,
            'peak_ratio': peak_ratio,
            'peak_percent_diff': peak_percent,
            'full_p2p': full_p2p,
            'swell_p2p': swell_p2p
        })
        
        if show_amplitude_stats:
            print(f"  Peak: Full={full_peak:.4f}, Swell={swell_peak:.4f}, Diff={peak_percent:+.2f}%")
            print(f"  RMS:  Full={full_rms:.4f}, Swell={swell_rms:.4f}, Wind={wind_rms:.4f}")
        
        # Plot
        probe_label = f"P{probe_num}"
        
        # Full signal
        if show_full_signal:
            ax_swell.plot(
                time_axis, signal_full,
                linewidth=linewidth * 0.7,
                label=f"{probe_label} full",
                linestyle="-",
                color=color_full,
                alpha=0.4,
                zorder=1
            )
        
        # Swell component
        ax_swell.plot(
            time_axis, signal_swell,
            linewidth=linewidth * 1.5,
            label=f"{probe_label} swell ({actual_freq:.2f}Hz)",
            linestyle=linestyle,
            color=color_swell,
            alpha=0.9,
            zorder=3
        )
        
        # Wind component
        ax_wind.plot(
            time_axis, signal_wind,
            linewidth=linewidth * (1.2 if dual_yaxis else 1.0),
            label=f"{probe_label} wind",
            linestyle=":" if dual_yaxis else "--",
            color=color_wind,
            alpha=0.7 if dual_yaxis else 0.8,
            zorder=2
        )
    
    # Formatting
    title = f"{Path(path).stem}\n{windcond} wind / {panelcond} panel / {target_freq:.3f} Hz"
    ax_swell.set_title(title, fontsize=fontsize + 2, fontweight='bold', pad=20)
    ax_swell.set_xlabel('Time (s)', fontsize=fontsize)
    
    if dual_yaxis:
        ax_swell.set_ylabel('Swell Amplitude', fontsize=fontsize, color=color_swell)
        ax_wind.set_ylabel('Wind+Noise Amplitude', fontsize=fontsize, color=color_wind)
        ax_swell.tick_params(axis='y', labelcolor=color_swell, labelsize=fontsize - 1)
        ax_wind.tick_params(axis='y', labelcolor=color_wind, labelsize=fontsize - 1)
    else:
        ax_swell.set_ylabel('Amplitude', fontsize=fontsize)
        ax_swell.tick_params(axis='y', labelsize=fontsize - 1)
    
    ax_swell.tick_params(axis='x', labelsize=fontsize - 1)
    
    if show_grid:
        ax_swell.grid(which='major', linestyle='--', alpha=0.3, linewidth=0.8)
        ax_swell.grid(which='minor', linestyle=':', alpha=0.15, linewidth=0.5)
        ax_swell.minorticks_on()
    
    # Legend
    lines_swell, labels_swell = ax_swell.get_legend_handles_labels()
    
    if dual_yaxis:
        lines_wind, labels_wind = ax_wind.get_legend_handles_labels()
        all_lines = lines_swell + lines_wind
        all_labels = labels_swell + labels_wind
    else:
        all_lines = lines_swell
        all_labels = labels_swell
    
    if all_lines:
        ax_swell.legend(
            all_lines, all_labels,
            loc='upper right',
            fontsize=fontsize,
            framealpha=0.95
        )
    
    ax_swell.axhline(0, color='black', linewidth=0.5, alpha=0.3, zorder=0)
    if dual_yaxis:
        ax_wind.axhline(0, color='darkred', linewidth=0.5, alpha=0.2, zorder=0)
    
    plt.tight_layout()
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    # Print summary
    if amplitude_comparison and show_amplitude_stats:
        print(f"\n{'='*60}")
        print("AMPLITUDE COMPARISON")
        print(f"{'='*60}")
        df_comparison = pd.DataFrame(amplitude_comparison)
        print(df_comparison.to_string(index=False))
        print(f"{'='*60}\n")
    
    return fig, ax_swell
# %%
def old_plot_reconstructed(

    fft_dict: Dict[str, pd.DataFrame],
    filtrert_frequencies: pd.DataFrame,
    freqplotvariables: dict,
    data_type: str = "fft"
) -> Tuple[Optional[plt.Figure], Optional[np.ndarray]]:
    """
    Plot reconstructed swell as a pure sine wave vs the rest of the signal.
    Uses dual y-axes to make both components visible.
    """
    meta_df = filtrert_frequencies.copy()
    plotting = freqplotvariables.get("plotting", {})
    
    # Configuration
    facet_by = plotting.get("facet_by", None)
    probes = plotting.get("probes", [1])
    probes = [probes] if not isinstance(probes, (list, tuple)) else probes
    
    show_grid = plotting.get("grid", True)
    show_plot = plotting.get("show_plot", True)
    linewidth = plotting.get("linewidth", 1.2)
    fontsize = 9
    
    # New options for reconstruction plot
    show_full_signal = plotting.get("show_full_signal", False)  # Default OFF
    dual_yaxis = plotting.get("dual_yaxis", True)  # Use dual y-axes by default
    
    # Determine facets
    if facet_by == "probe":
        facet_groups = probes
        facet_labels = [f"Probe {p}" for p in facet_groups]
    elif facet_by == "wind":
        facet_groups = pd.unique(meta_df["WindCondition"]).tolist()
        facet_labels = [f"Wind: {w}" for w in facet_groups]
    elif facet_by == "panel":
        facet_groups = pd.unique(meta_df["PanelCondition"]).tolist()
        facet_labels = [f"Panel: {p}" for p in facet_groups]
    else:
        facet_groups = [None]
        facet_labels = ["All Data"]
    
    n_facets = len(facet_groups)
    
    # Create figure
    default_figsize = (16, 5 * n_facets) if n_facets > 1 else (16, 7)
    figsize = plotting.get("figsize", default_figsize)
    
    fig, axes = plt.subplots(
        n_facets, 1,
        figsize=figsize,
        sharex=False,
        squeeze=False,
        dpi=120
    )
    axes = axes.flatten()
    
    # Plotting loop
    for facet_idx, (group, facet_label) in enumerate(zip(facet_groups, facet_labels)):
        ax_swell = axes[facet_idx]
        
        # Create second y-axis only if dual_yaxis is True
        if dual_yaxis:
            ax_wind = ax_swell.twinx()
        else:
            ax_wind = ax_swell  # Use same axis
        
        # Subset data for this facet
        if facet_by == "wind":
            subset = meta_df[meta_df["WindCondition"] == group]
        elif facet_by == "panel":
            subset = meta_df[meta_df["PanelCondition"] == group]
        else:
            subset = meta_df
        
        if len(subset) == 0:
            ax_swell.text(0.5, 0.5, 'No data', ha='center', va='center', 
                   transform=ax_swell.transAxes, fontsize=12)
            ax_swell.set_title(facet_label, fontsize=fontsize)
            continue
        
        for row_idx, row in subset.iterrows():
            path = row["path"]
            if path not in fft_dict:
                continue
            
            df_fft = fft_dict[path]
            
            # Styling
            windcond = row.get("WindCondition", "unknown")
            color_swell = GC.WIND_COLORS.get(windcond, "black")
            color_wind = "darkred" if dual_yaxis else "orange"  # Different color for single axis
            color_full = "gray"
            panelcond = row.get("PanelCondition", "unknown")
            linestyle = PANEL_STYLES.get(panelcond, "-")
            
            # Get target swell frequency
            target_freq = row.get(GC.WAVE_FREQUENCY_INPUT, None)
            if target_freq is None or target_freq <= 0:
                print(f"Skipping {Path(path).name}: invalid target frequency")
                continue
            
            label_base = f"{windcond}/{panelcond}"
            probes_to_plot = [group] if facet_by == "probe" else probes
            
            for probe_num in probes_to_plot:
                col = f"FFT {probe_num} complex"
                
                if col not in df_fft:
                    col = f"FFT {probe_num}"
                    if col not in df_fft:
                        continue
                
                fft_series = df_fft[col].dropna()
                if len(fft_series) == 0:
                    continue
                
                freq_bins = fft_series.index.values
                fft_complex = fft_series.values
                
                print(f"\n{'='*60}")
                print(f"Processing: {Path(path).stem} | Probe {probe_num}")
                print(f"Target swell: {target_freq:.3f} Hz")
                
                # Reorder FFT from sorted to fftfreq order
                N = len(fft_complex)
                df_freq = freq_bins[1] - freq_bins[0]
                sampling_rate = abs(df_freq * N)
                correct_fftfreq_order = np.fft.fftfreq(N, d=1/sampling_rate)
                
                fft_reordered = np.zeros(N, dtype=complex)
                for i, target_f in enumerate(correct_fftfreq_order):
                    closest_idx = np.argmin(np.abs(freq_bins - target_f))
                    if np.abs(freq_bins[closest_idx] - target_f) < 1e-6:
                        fft_reordered[i] = fft_complex[closest_idx]
                
                # Reconstruct full signal
                signal_full = np.real(np.fft.ifft(fft_reordered))
                time_axis = np.arange(N) / sampling_rate
                
                # Find target frequency bins
                pos_mask = correct_fftfreq_order > 0
                pos_freqs = correct_fftfreq_order[pos_mask]
                
                closest_pos_idx = np.argmin(np.abs(pos_freqs - target_freq))
                actual_freq = pos_freqs[closest_pos_idx]
                
                peak_idx = np.where(np.abs(correct_fftfreq_order - actual_freq) < 1e-6)[0][0]
                mirror_idx = np.where(np.abs(correct_fftfreq_order + actual_freq) < 1e-6)[0][0]
                
                # Create swell-only FFT
                fft_swell = np.zeros_like(fft_reordered, dtype=complex)
                fft_swell[peak_idx] = fft_reordered[peak_idx]
                fft_swell[mirror_idx] = fft_reordered[mirror_idx]
                
                # Reconstruct
                signal_swell = np.real(np.fft.ifft(fft_swell))
                signal_wind = signal_full - signal_swell
                
                print(f"  → Swell RMS: {np.sqrt(np.mean(signal_swell**2)):.4f}")
                print(f"  → Wind RMS: {np.sqrt(np.mean(signal_wind**2)):.4f}")
                
                # Plot
                plot_label = (
                    f"{label_base}_P{probe_num}" if len(probes_to_plot) > 1 
                    else label_base
                )
                
                # Full signal (optional)
                if show_full_signal:
                    ax_swell.plot(
                        time_axis, signal_full,
                        linewidth=linewidth * 0.7,
                        label=f"{plot_label} (full)",
                        linestyle="-",
                        color=color_full,
                        alpha=0.4,
                        zorder=1
                    )
                
                # Swell component
                ax_swell.plot(
                    time_axis, signal_swell,
                    linewidth=linewidth * 1.5,
                    label=f"{plot_label} (swell {actual_freq:.2f}Hz)",
                    linestyle=linestyle,
                    color=color_swell,
                    alpha=0.9,
                    zorder=3
                )
                
                # Wind component
                ax_wind.plot(
                    time_axis, signal_wind,
                    linewidth=linewidth * (1.2 if dual_yaxis else 1.0),
                    label=f"{plot_label} (wind+noise)",
                    linestyle=":" if dual_yaxis else "--",
                    color=color_wind,
                    alpha=0.7 if dual_yaxis else 0.8,
                    zorder=2
                )
        
        # Formatting
        ax_swell.set_title(facet_label, fontsize=fontsize + 2, fontweight='bold')
        ax_swell.set_xlabel('Time (s)', fontsize=fontsize)
        
        if dual_yaxis:
            ax_swell.set_ylabel('Swell Amplitude', fontsize=fontsize, color=color_swell)
            ax_wind.set_ylabel('Wind+Noise Amplitude', fontsize=fontsize, color=color_wind)
            ax_swell.tick_params(axis='y', labelcolor=color_swell, labelsize=fontsize - 1)
            ax_wind.tick_params(axis='y', labelcolor=color_wind, labelsize=fontsize - 1)
        else:
            ax_swell.set_ylabel('Amplitude', fontsize=fontsize)
            ax_swell.tick_params(axis='y', labelsize=fontsize - 1)
        
        ax_swell.tick_params(axis='x', labelsize=fontsize - 1)
        
        if show_grid:
            ax_swell.grid(which='major', linestyle='--', alpha=0.3, linewidth=0.8)
            ax_swell.grid(which='minor', linestyle=':', alpha=0.15, linewidth=0.5)
            ax_swell.minorticks_on()
        
        # Legend
        lines_swell, labels_swell = ax_swell.get_legend_handles_labels()
        
        if dual_yaxis:
            lines_wind, labels_wind = ax_wind.get_legend_handles_labels()
            all_lines = lines_swell + lines_wind
            all_labels = labels_swell + labels_wind
        else:
            all_lines = lines_swell
            all_labels = labels_swell
        
        if all_lines:
            ax_swell.legend(
                all_lines, all_labels,
                loc='upper right',
                fontsize=fontsize - 1,
                framealpha=0.95
            )
        
        ax_swell.axhline(0, color='black', linewidth=0.5, alpha=0.3, zorder=0)
        if dual_yaxis:
            ax_wind.axhline(0, color='darkred', linewidth=0.5, alpha=0.2, zorder=0)
    
    title_suffix = "(Dual Y-axes)" if dual_yaxis else "(Single Y-axis)"
    plt.suptitle(f'Signal Decomposition: Swell vs Wind+Noise {title_suffix}', 
                 fontsize=fontsize + 4, fontweight='bold', y=0.995)
    fig.tight_layout()
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return fig, axes
# %%
def plot_reconstructed_rms(
    fft_dict: Dict[str, pd.DataFrame],
    filtrert_frequencies: pd.DataFrame,
    freqplotvariables: dict,
    data_type: str = "fft"
) -> Tuple[Optional[plt.Figure], Optional[np.ndarray]]:
    """
    Plot reconstructed swell as a pure sine wave vs the rest of the signal.
    Calculates and reports amplitude differences between full signal and swell extraction.
    """
    meta_df = filtrert_frequencies.copy()
    plotting = freqplotvariables.get("plotting", {})
    
    # Configuration
    facet_by = plotting.get("facet_by", None)
    probes = plotting.get("probes", [1])
    probes = [probes] if not isinstance(probes, (list, tuple)) else probes
    
    show_grid = plotting.get("grid", True)
    show_plot = plotting.get("show_plot", True)
    linewidth = plotting.get("linewidth", 1.2)
    fontsize = 9
    
    show_full_signal = plotting.get("show_full_signal", False)
    dual_yaxis = plotting.get("dual_yaxis", True)
    show_amplitude_stats = plotting.get("show_amplitude_stats", True)  # New option
    
    # Determine facets
    if facet_by == "probe":
        facet_groups = probes
        facet_labels = [f"Probe {p}" for p in facet_groups]
    elif facet_by == "wind":
        facet_groups = pd.unique(meta_df["WindCondition"]).tolist()
        facet_labels = [f"Wind: {w}" for w in facet_groups]
    elif facet_by == "panel":
        facet_groups = pd.unique(meta_df["PanelCondition"]).tolist()
        facet_labels = [f"Panel: {p}" for p in facet_groups]
    else:
        facet_groups = [None]
        facet_labels = ["All Data"]
    
    n_facets = len(facet_groups)
    
    # Create figure
    default_figsize = (16, 5 * n_facets) if n_facets > 1 else (16, 7)
    figsize = plotting.get("figsize", default_figsize)
    
    fig, axes = plt.subplots(
        n_facets, 1,
        figsize=figsize,
        sharex=False,
        squeeze=False,
        dpi=120
    )
    axes = axes.flatten()
    
    # Storage for amplitude comparison stats
    amplitude_comparison = []
    
    # Plotting loop
    for facet_idx, (group, facet_label) in enumerate(zip(facet_groups, facet_labels)):
        ax_swell = axes[facet_idx]
        
        if dual_yaxis:
            ax_wind = ax_swell.twinx()
        else:
            ax_wind = ax_swell
        
        # Subset data for this facet
        if facet_by == "wind":
            subset = meta_df[meta_df["WindCondition"] == group]
        elif facet_by == "panel":
            subset = meta_df[meta_df["PanelCondition"] == group]
        else:
            subset = meta_df
        
        if len(subset) == 0:
            ax_swell.text(0.5, 0.5, 'No data', ha='center', va='center', 
                   transform=ax_swell.transAxes, fontsize=12)
            ax_swell.set_title(facet_label, fontsize=fontsize)
            continue
        
        for row_idx, row in subset.iterrows():
            path = row["path"]
            if path not in fft_dict:
                continue
            
            df_fft = fft_dict[path]
            
            # Styling
            windcond = row.get("WindCondition", "unknown")
            color_swell = WIND_COLORS.get(windcond, "black")
            color_wind = "darkred" if dual_yaxis else "orange"
            color_full = "gray"
            panelcond = row.get("PanelCondition", "unknown")
            linestyle = PANEL_STYLES.get(panelcond, "-")
            
            # Get target swell frequency
            target_freq = row.get(GC.WAVE_FREQUENCY_INPUT, None)
            if target_freq is None or target_freq <= 0:
                print(f"Skipping {Path(path).name}: invalid target frequency")
                continue
            
            label_base = f"{windcond}/{panelcond}"
            probes_to_plot = [group] if facet_by == "probe" else probes
            
            for probe_num in probes_to_plot:
                col = f"FFT {probe_num} complex"
                
                if col not in df_fft:
                    col = f"FFT {probe_num}"
                    if col not in df_fft:
                        continue
                
                fft_series = df_fft[col].dropna()
                if len(fft_series) == 0:
                    continue
                
                freq_bins = fft_series.index.values
                fft_complex = fft_series.values
                
                print(f"\n{'='*60}")
                print(f"Processing: {Path(path).stem} | Probe {probe_num}")
                print(f"Target swell: {target_freq:.3f} Hz")
                
                # Reorder FFT from sorted to fftfreq order
                N = len(fft_complex)
                df_freq = freq_bins[1] - freq_bins[0]
                sampling_rate = abs(df_freq * N)
                correct_fftfreq_order = np.fft.fftfreq(N, d=1/sampling_rate)
                
                fft_reordered = np.zeros(N, dtype=complex)
                for i, target_f in enumerate(correct_fftfreq_order):
                    closest_idx = np.argmin(np.abs(freq_bins - target_f))
                    if np.abs(freq_bins[closest_idx] - target_f) < 1e-6:
                        fft_reordered[i] = fft_complex[closest_idx]
                
                # Reconstruct full signal
                signal_full = np.real(np.fft.ifft(fft_reordered))
                time_axis = np.arange(N) / sampling_rate
                
                # Find target frequency bins
                pos_mask = correct_fftfreq_order > 0
                pos_freqs = correct_fftfreq_order[pos_mask]
                
                closest_pos_idx = np.argmin(np.abs(pos_freqs - target_freq))
                actual_freq = pos_freqs[closest_pos_idx]
                
                peak_idx = np.where(np.abs(correct_fftfreq_order - actual_freq) < 1e-6)[0][0]
                mirror_idx = np.where(np.abs(correct_fftfreq_order + actual_freq) < 1e-6)[0][0]
                
                # Create swell-only FFT
                fft_swell = np.zeros_like(fft_reordered, dtype=complex)
                fft_swell[peak_idx] = fft_reordered[peak_idx]
                fft_swell[mirror_idx] = fft_reordered[mirror_idx]
                
                # Reconstruct
                signal_swell = np.real(np.fft.ifft(fft_swell))
                signal_wind = signal_full - signal_swell
                
                # ═══════════════════════════════════════════════
                # AMPLITUDE ANALYSIS
                # ═══════════════════════════════════════════════
                
                # Peak-to-peak amplitudes
                full_p2p = np.max(signal_full) - np.min(signal_full)
                swell_p2p = np.max(signal_swell) - np.min(signal_swell)
                wind_p2p = np.max(signal_wind) - np.min(signal_wind)
                
                # Peak amplitudes (max absolute value)
                full_peak = np.max(np.abs(signal_full))
                swell_peak = np.max(np.abs(signal_swell))
                wind_peak = np.max(np.abs(signal_wind))
                
                # RMS values
                full_rms = np.sqrt(np.mean(signal_full**2))
                swell_rms = np.sqrt(np.mean(signal_swell**2))
                wind_rms = np.sqrt(np.mean(signal_wind**2))
                
                # Differences and ratios
                peak_diff = full_peak - swell_peak
                peak_ratio = (full_peak / swell_peak) if swell_peak > 0 else np.nan
                peak_percent = (peak_diff / full_peak * 100) if full_peak > 0 else np.nan
                
                rms_diff = full_rms - swell_rms
                rms_ratio = (full_rms / swell_rms) if swell_rms > 0 else np.nan
                
                # Store comparison
                amplitude_comparison.append({
                    'path': Path(path).stem,
                    'probe': probe_num,
                    'wind': windcond,
                    'panel': panelcond,
                    'full_peak': full_peak,
                    'swell_peak': swell_peak,
                    'wind_peak': wind_peak,
                    'full_rms': full_rms,
                    'swell_rms': swell_rms,
                    'wind_rms': wind_rms,
                    'peak_diff': peak_diff,
                    'peak_ratio': peak_ratio,
                    'peak_percent_diff': peak_percent,
                    'full_p2p': full_p2p,
                    'swell_p2p': swell_p2p
                })
                
                # Print detailed comparison
                print(f"\n  AMPLITUDE COMPARISON:")
                print(f"  {'='*50}")
                print(f"  Peak Amplitude:")
                print(f"    Full signal:  {full_peak:8.4f}")
                print(f"    Swell only:   {swell_peak:8.4f}")
                print(f"    Difference:   {peak_diff:8.4f} ({peak_percent:+.2f}%)")
                print(f"    Ratio:        {peak_ratio:.4f}x")
                print(f"    Wind peak:    {wind_peak:8.4f}")
                print(f"\n  Peak-to-Peak:")
                print(f"    Full signal:  {full_p2p:8.4f}")
                print(f"    Swell only:   {swell_p2p:8.4f}")
                print(f"\n  RMS:")
                print(f"    Full signal:  {full_rms:8.4f}")
                print(f"    Swell only:   {swell_rms:8.4f}")
                print(f"    Difference:   {rms_diff:8.4f}")
                print(f"    Ratio:        {rms_ratio:.4f}x")
                print(f"    Wind RMS:     {wind_rms:8.4f}")
                
                # Plot
                plot_label = (
                    f"{label_base}_P{probe_num}" if len(probes_to_plot) > 1 
                    else label_base
                )
                
                # Full signal (optional)
                if show_full_signal:
                    ax_swell.plot(
                        time_axis, signal_full,
                        linewidth=linewidth * 0.7,
                        label=f"{plot_label} (full, pk={full_peak:.2f})",
                        linestyle="-",
                        color=color_full,
                        alpha=0.4,
                        zorder=1
                    )
                
                # Swell component
                swell_label = f"{plot_label} (swell {actual_freq:.2f}Hz, pk={swell_peak:.2f})"
                if show_amplitude_stats and show_full_signal:
                    swell_label += f" [{peak_percent:+.1f}%]"
                
                ax_swell.plot(
                    time_axis, signal_swell,
                    linewidth=linewidth * 1.5,
                    label=swell_label,
                    linestyle=linestyle,
                    color=color_swell,
                    alpha=0.9,
                    zorder=3
                )
                
                # Wind component
                ax_wind.plot(
                    time_axis, signal_wind,
                    linewidth=linewidth * (1.2 if dual_yaxis else 1.0),
                    label=f"{plot_label} (wind+noise, pk={wind_peak:.2f})",
                    linestyle=":" if dual_yaxis else "--",
                    color=color_wind,
                    alpha=0.7 if dual_yaxis else 0.8,
                    zorder=2
                )
        
        # Formatting
        ax_swell.set_title(facet_label, fontsize=fontsize + 2, fontweight='bold')
        ax_swell.set_xlabel('Time (s)', fontsize=fontsize)
        
        if dual_yaxis:
            ax_swell.set_ylabel('Swell Amplitude', fontsize=fontsize, color=color_swell)
            ax_wind.set_ylabel('Wind+Noise Amplitude', fontsize=fontsize, color=color_wind)
            ax_swell.tick_params(axis='y', labelcolor=color_swell, labelsize=fontsize - 1)
            ax_wind.tick_params(axis='y', labelcolor=color_wind, labelsize=fontsize - 1)
        else:
            ax_swell.set_ylabel('Amplitude', fontsize=fontsize)
            ax_swell.tick_params(axis='y', labelsize=fontsize - 1)
        
        ax_swell.tick_params(axis='x', labelsize=fontsize - 1)
        
        if show_grid:
            ax_swell.grid(which='major', linestyle='--', alpha=0.3, linewidth=0.8)
            ax_swell.grid(which='minor', linestyle=':', alpha=0.15, linewidth=0.5)
            ax_swell.minorticks_on()
        
        # Legend
        lines_swell, labels_swell = ax_swell.get_legend_handles_labels()
        
        if dual_yaxis:
            lines_wind, labels_wind = ax_wind.get_legend_handles_labels()
            all_lines = lines_swell + lines_wind
            all_labels = labels_swell + labels_wind
        else:
            all_lines = lines_swell
            all_labels = labels_swell
        
        if all_lines:
            ax_swell.legend(
                all_lines, all_labels,
                loc='upper right',
                fontsize=fontsize - 1,
                framealpha=0.95
            )
        
        ax_swell.axhline(0, color='black', linewidth=0.5, alpha=0.3, zorder=0)
        if dual_yaxis:
            ax_wind.axhline(0, color='darkred', linewidth=0.5, alpha=0.2, zorder=0)
    
    title_suffix = "(Dual Y-axes)" if dual_yaxis else "(Single Y-axis)"
    plt.suptitle(f'Signal Decomposition: Swell vs Wind+Noise {title_suffix}', 
                 fontsize=fontsize + 4, fontweight='bold', y=0.995)
    fig.tight_layout()
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    # Print summary table
    if amplitude_comparison and show_amplitude_stats:
        print(f"\n{'='*80}")
        print("AMPLITUDE COMPARISON SUMMARY")
        print(f"{'='*80}")
        df_comparison = pd.DataFrame(amplitude_comparison)
        print(df_comparison.to_string(index=False))
        print(f"{'='*80}\n")
    
    return fig, axes
# %%
from pathlib import Path
from typing import Optional, Tuple

def plot_swell_p2_vs_p3_by_wind(
    band_amplitudes: pd.DataFrame,
    meta_df: pd.DataFrame = None,           # kept for compatibility, but currently unused
    figsize_per_facet: Tuple[float, float] = (4.5, 4.2),
    ncols: int = 3,
    alpha: float = 0.85,
    annotate: bool = False,
    show: bool = True,
    save_path: Optional[Path] = None,
    title: Optional[str] = "Swell amplitude: Before (P2) vs After (P3) by wind",
    xlabel: str = "P2 (before) amplitude (mm)",
    ylabel: str = "P3 (after) amplitude (mm)",
    wind_order: Optional[Sequence[str]] = None,
) -> Tuple[Optional[plt.Figure], Optional[np.ndarray]]:
    """
    Create faceted scatter plots comparing swell amplitudes before (P2) and after (P3),
    with one subplot per wind condition.

    Features:
    - Shared axis limits across facets
    - y = x reference line
    - Optional filename annotation
    - Mean difference (Δ) displayed in each facet
    """
    # ────────────────────────────────────────────────
    # Column names (using your global constants)
    # ────────────────────────────────────────────────
    path_col  = GC.PATH
    wind_col  = GC.WIND_CONDITION
    panel_col = GC.PANEL_CONDITION          # kept for future use / compatibility
    p2_col    = PC.SWELL_AMPLITUDE_PSD.format(i=2)
    p3_col    = PC.SWELL_AMPLITUDE_PSD.format(i=3)

    # ────────────────────────────────────────────────
    # Prepare data – start from band_amplitudes
    # ────────────────────────────────────────────────
    df = band_amplitudes.copy()

    required_cols = [path_col, wind_col, p2_col, p3_col]
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    # Keep only rows with valid (non-null, finite) P2 & P3 values
    df = df[df[[p2_col, p3_col]].notna().all(axis=1)].copy()
    df[p2_col] = df[p2_col].astype(float)
    df[p3_col] = df[p3_col].astype(float)

    # Also require wind condition to be known
    df = df[df[wind_col].notna()].copy()

    if df.empty:
        print("No valid (finite P2/P3 + known wind) data to plot.")
        return None, None

    # ────────────────────────────────────────────────
    # Determine wind categories & order
    # ────────────────────────────────────────────────
    present_winds = df[wind_col].unique()

    if wind_order is not None:
        winds = [w for w in wind_order if w in present_winds]
        winds += [w for w in present_winds if w not in wind_order]
    else:
        winds = sorted(present_winds)

    if not winds:
        print("No wind conditions remain after filtering.")
        return None, None

    # ────────────────────────────────────────────────
    # Shared axis limits with small padding
    # ────────────────────────────────────────────────
    p2_all = df[p2_col].values
    p3_all = df[p3_col].values
    lo = min(p2_all.min(), p3_all.min())
    hi = max(p2_all.max(), p3_all.max())

    if hi <= lo:
        lo, hi = 0.0, max(1.0, hi * 1.1 if hi > 0 else 1.0)

    pad = 0.05 * (hi - lo)
    lim = (lo - pad, hi + pad)

    # ────────────────────────────────────────────────
    # Create figure & grid of subplots
    # ────────────────────────────────────────────────
    n_facets = len(winds)
    ncols = max(1, min(ncols, n_facets))
    nrows = (n_facets + ncols - 1) // ncols
    fig_size = (figsize_per_facet[0] * ncols, figsize_per_facet[1] * nrows)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=fig_size,
        squeeze=False,
        dpi=120
    )
    axes_flat = axes.flat

    # ────────────────────────────────────────────────
    # Plot each wind condition facet
    # ────────────────────────────────────────────────
    for i, wind in enumerate(winds):
        ax = axes_flat[i]
        sub = df[df[wind_col] == wind]

        if sub.empty:
            ax.set_visible(False)
            continue

        x = sub[p2_col].values
        y = sub[p3_col].values

        # Point color (use WIND_COLORS if defined)
        color = WIND_COLORS.get(wind, "#1f77b4") if "WIND_COLORS" in globals() else "#1f77b4"

        ax.scatter(
            x, y,
            s=40,
            c=color,
            edgecolor="white",
            linewidth=0.7,
            alpha=alpha,
            rasterized=True
        )

        # Reference line
        ax.plot(lim, lim, color="#888", ls="--", lw=1.0, label="y = x")

        # Title + count
        ax.set_title(f"Wind: {wind}\n(n = {len(sub)})", fontsize=9)

        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.grid(True, alpha=0.3, lw=0.6)

        # Axis labels – only on left column and bottom row
        if i % ncols == 0:
            ax.set_ylabel(ylabel, fontsize=9)
        if i >= (nrows - 1) * ncols:
            ax.set_xlabel(xlabel, fontsize=9)

        # Optional: annotate with file stem
        if annotate:
            for xi, yi, p in zip(x, y, sub[path_col].astype(str)):
                stem = Path(p).stem
                ax.annotate(
                    stem,
                    (xi, yi),
                    xytext=(6, 4),
                    textcoords="offset points",
                    fontsize=6.5,
                    alpha=0.80,
                    color="#444"
                )

        # Mean difference in top-left corner
        delta_mean = (y - x).mean()
        ax.text(
            0.03, 0.97,
            f"Δ mean = {delta_mean:+.3f} mm",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=8,
            color="#444"
        )

    # Hide unused subplots
    for ax in axes_flat[n_facets:]:
        ax.set_visible(False)

    # Final figure touches
    if title:
        fig.suptitle(title, fontsize=11, y=0.995)

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, axes
# %% claude  - caller filter internt
def plot_p2_vs_p3_scatter(meta_df: pd.DataFrame, filter_vars: dict):
    """
    Plot P2 vs P3 amplitudes for different spectral bands with detailed metadata.
    
    Args:
        meta_df: Full metadata dataframe (BEFORE filtering)
        filter_vars: Dictionary with filter settings (swellplotvariables)
    """
    
    # Color/style mappings
    WIND_COLORS = {
        "full": "red",
        "no": "blue",
        "lowest": "green"
    }
    
    PANEL_MARKERS = {
        "no": "o",
        "full": "s",
        "reverse": "^"
    }
    
    band_constants = {
        'Swell': PC.SWELL_AMPLITUDE_PSD,
        'Wind': PC.WIND_AMPLITUDE_PSD,
        'Total': PC.TOTAL_AMPLITUDE_PSD,
    }
    
    # Check if filtering is active
    overordnet = filter_vars.get("overordnet", {})
    chooseAll = overordnet.get("chooseAll", False)
    chooseFirst = overordnet.get("chooseFirst", False)
    filters_active = not chooseAll  # Filters only active if NOT chooseAll
    # Apply filtering
    print("\n" + "="*60)
    print("FILTERING DATA FOR PLOT")
    print("="*60)
    band_amplitudes = filter_for_amplitude_plot(meta_df, filter_vars)
    
    # CHECK IF DATAFRAME IS EMPTY AFTER FILTERING
    if len(band_amplitudes) == 0:
        print("\n" + "="*60)
        print("ERROR: No data remaining after filtering!")
        return
    
    print(f"\n✓ Filtered data has {len(band_amplitudes)} rows\n")
    
    # Check which metadata columns exist
    has_wind = GC.WIND_CONDITION in band_amplitudes.columns
    has_panel = GC.PANEL_CONDITION in band_amplitudes.columns
    has_freq = GC.WAVE_FREQUENCY_INPUT in band_amplitudes.columns
    has_amp = GC.WAVE_AMPLITUDE_INPUT in band_amplitudes.columns
    
    # Extract metadata
    n_points = len(band_amplitudes)
    unique_winds = band_amplitudes[GC.WIND_CONDITION].unique() if has_wind else ['N/A']
    unique_panels = band_amplitudes[GC.PANEL_CONDITION].unique() if has_panel else ['N/A']
    unique_freqs = band_amplitudes[GC.WAVE_FREQUENCY_INPUT].unique() if has_freq else ['N/A']
    unique_amps = band_amplitudes[GC.WAVE_AMPLITUDE_INPUT].unique() if has_amp else ['N/A']
    
    # Create figure
    n_bands = len(band_constants)
    fig = plt.figure(figsize=(14, 5))
    
    # Create gridspec for main plots + info panel
    gs = fig.add_gridspec(1, n_bands + 1, width_ratios=[1, 1, 1, 0.4])
    axes = [fig.add_subplot(gs[0, i]) for i in range(n_bands)]
    info_ax = fig.add_subplot(gs[0, -1])
    info_ax.axis('off')
    
    # Build info text
    info_text = "DATA SUMMARY\n" + "="*25 + "\n\n"
    info_text += f"N points: {n_points}\n\n"
    
    info_text += "Wind:\n"
    for w in unique_winds:
        count = (band_amplitudes[GC.WIND_CONDITION] == w).sum() if has_wind else 0
        info_text += f"  • {w}: {count}\n"
    
    info_text += "\nPanel:\n"
    for p in unique_panels:
        count = (band_amplitudes[GC.PANEL_CONDITION] == p).sum() if has_panel else 0
        info_text += f"  • {p}: {count}\n"
    
    info_text += f"\nFreq [Hz]:\n"
    for f in unique_freqs:
        if f != 'N/A':
            info_text += f"  • {f:.3f}\n"
    
    info_text += f"\nAmp [V]:\n"
    for a in unique_amps:
        if a != 'N/A':
            info_text += f"  • {a:.2f}\n"
    
    if filters_active:
        info_text += "\n" + "="*25 + "\nFILTERS APPLIED\n" + "="*25 + "\n"
        filters = filter_vars.get('filters', {})
        for key, val in filters.items():
            if val is not None:
                # Shorten long lists
                if isinstance(val, (list, tuple)) and len(val) > 3:
                    val_str = f"[{val[0]}, ..., {val[-1]}]"
                else:
                    val_str = str(val)
                info_text += f"{key}:\n  {val_str}\n"
    else:
        info_text += "\n" + "="*25 + "\nNO FILTERS\n" + "="*25 + "\n"
        info_text += "(chooseAll=True)\n"
        if chooseFirst:
            info_text += "(chooseFirst=True)\n"
    
    info_ax.text(0.05, 0.95, info_text, 
                 transform=info_ax.transAxes,
                 fontsize=7,
                 verticalalignment='top',
                 fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Plot each band
    for ax, (band_name, constant_template) in zip(axes, band_constants.items()):
        p2_col = constant_template.format(i=2)
        p3_col = constant_template.format(i=3)
        
        # Check if columns exist
        if p2_col not in band_amplitudes.columns or p3_col not in band_amplitudes.columns:
            ax.text(0.5, 0.5, f'Missing\ncolumns', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_title(f'{band_name} Band', fontweight='bold')
            continue
        
        p2 = band_amplitudes[p2_col].to_numpy()
        p3 = band_amplitudes[p3_col].to_numpy()
        
        # Color by wind, marker by panel
        if has_wind and has_panel:
            for wind in unique_winds:
                for panel in unique_panels:
                    mask = (band_amplitudes[GC.WIND_CONDITION] == wind) & \
                           (band_amplitudes[GC.PANEL_CONDITION] == panel)
                    
                    if mask.sum() > 0:
                        ax.scatter(
                            p2[mask], 
                            p3[mask], 
                            alpha=0.7,
                            color=WIND_COLORS.get(wind, 'gray'),
                            marker=PANEL_MARKERS.get(panel, 'o'),
                            s=80,
                            label=f'{wind}/{panel}',
                            edgecolors='black',
                            linewidths=0.5
                        )
        else:
            # Fallback: simple scatter
            ax.scatter(p2, p3, alpha=0.7, s=80, edgecolors='black', linewidths=0.5)
        
        # Reference line
        valid_mask = ~(np.isnan(p2) | np.isnan(p3))
        if valid_mask.sum() > 0:
            lim = max(p2[valid_mask].max(), p3[valid_mask].max()) * 1.05
            ax.plot([0, lim], [0, lim], 'k--', linewidth=1, alpha=0.5, zorder=1)
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
        
        ax.set_title(f'{band_name} Band', fontweight='bold')
        ax.set_xlabel('P2 amplitude', fontsize=10)
        ax.set_ylabel('P3 amplitude', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Add legend
        if has_wind and has_panel:
            handles, labels = ax.get_legend_handles_labels()
            if len(handles) > 0:
                ax.legend(fontsize=7, loc='upper left', framealpha=0.9)
    
    plt.suptitle('P2 vs P3 Amplitude Comparison', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()
    
# %%
def old_plot_p2_vs_p3_scatter(band_amplitudes):
    band_name = ['Swell', 'Wind', 'Total']
    fig, axes = plt.subplots(1, len(band_name), figsize=(12, 4), sharex=False, sharey=False)
    band_constants = {
        'Swell': PC.SWELL_AMPLITUDE_PSD,
        'Wind': PC.WIND_AMPLITUDE_PSD,
        'Total': PC.TOTAL_AMPLITUDE_PSD,
    }
    
    for ax, (band_name, constant_template) in zip(axes, band_constants.items()):
        p2_in = band_amplitudes[constant_template.format(i=2)].to_numpy()
        p3_in = band_amplitudes[constant_template.format(i=3)].to_numpy()
        # print(p2)
        # print(p3)
        # CREATE VALID MASK - remove NaN and inf
        valid_mask = np.isfinite(p2_in) & np.isfinite(p3_in)
        p2 = p2_in[valid_mask]
        p3 = p3_in[valid_mask]
        
        ax.scatter(p2, p3, alpha=0.7)
        
        # Calculate limits
        lim = max(p2.max(), p3.max()) * 1.05 if len(p2) else 1.0

        # Plot reference line FIRST (or use zorder)
        ax.plot([0, lim], [0, lim], 'k--', linewidth=1, label='y=x', zorder=1)
        
        # Set axis limits to show the reference line
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        
        ax.set_title(f'{band_name}')
        ax.set_xlabel('P2 amplitude')
        ax.set_ylabel('P3 amplitude')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')  # Optional: makes it a square plot
    
    plt.tight_layout()
    plt.show()
    
def plot_p2_p3_bars(band_amplitudes):
    bands = ['Swell', 'Wind', 'Total']
    for _, row in band_amplitudes.iterrows():
        values_p2 = [row[PC.SWELL_AMPLITUDE_PSD.format(i=2)] for b in bands]
        values_p3 = [row[PC.SWELL_AMPLITUDE_PSD.format(i=3)] for b in bands]

        x = np.arange(len(bands))
        w = 0.35

        plt.figure(figsize=(8, 4))
        plt.bar(x - w/2, values_p2, width=w, label='P2')
        plt.bar(x + w/2, values_p3, width=w, label='P3')
        plt.xticks(x, bands)
        plt.ylabel('Amplitude')
        plt.title(row[40:])
        plt.legend()
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()




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