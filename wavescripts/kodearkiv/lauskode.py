#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 15:25:36 2025

@author: ole
"""

for path, (start, end) in auto_ranges.items():
    mask = meta["path"] == path
    meta.loc[mask, "auto_start"] = good_start_idx
    meta.loc[mask, "auto_end"]   = good_end_idx
    """OOOPS MÅ VÆRE TILPASSET PROBE OGSÅ"""
    

meta_volt = row["WaveAmplitudeInput [Volt]"]
meta_freq = row["WaveFrequencyInput [Hz]"]
meta_per = row["WavePeriodInput"]

df_sel["Calculated end"] = end #

#-------- fra plotteR_selection

for path in df_sel["path"]: #pleide å være processed_dfs
    auto_start, auto_end = auto_ranges[path]
    start = manual_start if manual_start is not None else auto_start
    end   = manual_end   if manual_end   is not None else auto_end
    plot_ranges[path] = (start, end)
    
#--- fra inni process_selsected _data
    #----fra inni forloopen
        #print('df_ma inside process_selected_data where, \n the first number of samples will become Nan because thats how the function works \n',df_ma.head())


#--- fra processorr. trenger vel ikke denne for den har jeg kanskje lagret et annet ste
#... 
def debug_plot_ramp_detection(df, data_col,
                              signal,
                              baseline_mean,
                              threshold,
                              first_motion_idx,
                              good_start_idx,
                              good_end_idx,
                              title="Ramp Detection Debug"):

    time = df["Date"]

    plt.figure(figsize=(14, 6))

    # Raw probe signal
    plt.plot(time, df[data_col], label="Raw signal", alpha=0.4)

    # Smoothed detection signal
    plt.plot(time, signal, label="Smoothed (detect)", linewidth=2)

    # Baseline mean
    plt.axhline(baseline_mean, color="blue", linestyle="--",
                label=f"Baseline mean = {baseline_mean:.3f}")

    # Threshold region
    plt.axhline(baseline_mean + threshold, color="red", linestyle="--",
                label=f"+ Threshold ({threshold:.3f})")
    plt.axhline(baseline_mean - threshold, color="red", linestyle="--")

    # First motion
    plt.axvline(time.iloc[first_motion_idx],
                color="orange", linestyle="--", linewidth=2,
                label=f"First motion @ {first_motion_idx}")

    # Good interval start / end
    plt.axvline(time.iloc[good_start_idx],
                color="green", linestyle="--", linewidth=2,
                label=f"Good start @ {good_start_idx}")

    plt.axvline(time.iloc[good_end_idx],
                color="purple", linestyle="--", linewidth=2,
                label=f"Good end @ {good_end_idx}")

    # Shaded good region
    plt.axvspan(time.iloc[good_start_idx],
                time.iloc[good_end_idx],
                color="green", alpha=0.15)

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel(data_col)
    plt.legend()
    plt.tight_layout()
    plt.show()