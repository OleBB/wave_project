#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 15:25:36 2025

@author: ole
"""







# %%



"""claude plot ubrukt"""

meta_df = meta_sel.copy()

psd_plot_config = {
    "overordnet": {"chooseAll": False}, 
    "filters": {
        # Filter which files/runs to include
        "path_pattern": None,  # e.g., "wind" to match files with "wind" in path
        "exclude_pattern": None,  # e.g., "test" to exclude test runs
        "meta_filters": {  # Filter based on metadata columns
            # "WaveAmplitudeInput [Volt]": [0.1, 0.2, 0.3],
            # "WindCondition": ["no", "lowest"],
        },
        # Filter which probes/columns to plot
        "probes": [1, 2, 3, 4],  # None for all, or list like [1, 3]
        "freq_range": (0, 10),
    },
    "processing": {
        "normalize": False,
        "smooth": False,
        "smooth_window": 5,
        "average_probes": False,  # average across selected probes
        "log_scale": False,
    },
    "plotting": {
        "figsize": (10, 6),
        "linewidth": 1.5,
        "alpha": 0.7,  # lower for many lines
        "show_legend": True,
        "legend_ncol": 2,
        "grid": True,
        "title": None,
        "color_by": None,  # "probe", "file", or a metadata column name
        "separate_by_probe": False,  # create subplots per probe
    }   
}

def filter_runs(psd_dict, meta_df, filters):
    """Filter which runs/files to include based on path and metadata"""
    runs_to_plot = list(psd_dict.keys())
    
    # Filter by path pattern
    if filters["path_pattern"] is not None:
        runs_to_plot = [r for r in runs_to_plot if filters["path_pattern"] in str(r)]
    
    if filters["exclude_pattern"] is not None:
        runs_to_plot = [r for r in runs_to_plot if filters["exclude_pattern"] not in str(r)]
    
    # Filter by metadata
    if filters["meta_filters"] and meta_df is not None:
        for col, values in filters["meta_filters"].items():
            if values is not None and col in meta_df.columns:
                valid_runs = meta_df[meta_df[col].isin(values)].index.tolist()
                runs_to_plot = [r for r in runs_to_plot if r in valid_runs]
    
    return runs_to_plot

def get_probe_columns(df, probes):
    """Get column names for specified probes"""
    if probes is None:
        # Return all Pxx columns
        return [col for col in df.columns if col.startswith('Pxx')]
    else:
        # Return specific probe columns
        probe_cols = []
        for p in probes:
            # Adjust this based on your actual column naming
            col_name = f'Pxx_probe{p}'  # or 'Pxx Probe {p}' or however they're named
            if col_name in df.columns:
                probe_cols.append(col_name)
        return probe_cols

def plot_psd_from_dict(psd_dict, meta_df, config):
    """Plot PSDs from dictionary of dataframes"""
    
    # Filter runs
    if config["overordnet"]["chooseAll"]:
        runs_to_plot = list(psd_dict.keys())
    else:
        runs_to_plot = filter_runs(psd_dict, meta_df, config["filters"])
    
    print(f"Plotting {len(runs_to_plot)} runs")
    
    # Setup plot
    plot_cfg = config["plotting"]
    proc_cfg = config["processing"]
    
    if plot_cfg["separate_by_probe"]:
        # Create subplots for each probe
        probes = config["filters"]["probes"] or [1, 2, 3, 4]
        fig, axes = plt.subplots(len(probes), 1, figsize=(10, 3*len(probes)), sharex=True)
        if len(probes) == 1:
            axes = [axes]
    else:
        fig, ax = plt.subplots(figsize=plot_cfg["figsize"] or (10, 6))
        axes = [ax]
    
    # Color mapping
    if plot_cfg["color_by"] == "probe":
        colors = plt.cm.tab10(np.linspace(0, 1, 4))
        color_map = {f'Pxx_probe{i}': colors[i-1] for i in range(1, 5)}
    elif plot_cfg["color_by"] == "file":
        colors = plt.cm.viridis(np.linspace(0, 1, len(runs_to_plot)))
        color_map = {run: colors[i] for i, run in enumerate(runs_to_plot)}
    elif plot_cfg["color_by"] and meta_df is not None:
        # Color by metadata column
        unique_vals = meta_df[plot_cfg["color_by"]].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_vals)))
        color_map = {val: colors[i] for i, val in enumerate(unique_vals)}
    else:
        color_map = None
    
    # Plot each run
    for run_path in runs_to_plot:
        df = psd_dict[run_path]
        freq = df.iloc[:, 0]  # First column is frequency
        
        # Get probe columns to plot
        probe_cols = get_probe_columns(df, config["filters"]["probes"])
        
        # Filter frequency range
        freq_range = config["filters"]["freq_range"]
        mask = (freq >= freq_range[0]) & (freq <= freq_range[1])
        
        if plot_cfg["separate_by_probe"]:
            # Plot each probe in its own subplot
            for idx, probe_col in enumerate(probe_cols):
                data = df[probe_col].values[mask]
                
                if proc_cfg["normalize"]:
                    data = data / data.max()
                if proc_cfg["smooth"]:
                    from scipy.ndimage import uniform_filter1d
                    data = uniform_filter1d(data, size=proc_cfg["smooth_window"])
                
                # Determine color
                if color_map and plot_cfg["color_by"] == "file":
                    color = color_map[run_path]
                elif color_map and plot_cfg["color_by"] in meta_df.columns:
                    color = color_map[meta_df.loc[run_path, plot_cfg["color_by"]]]
                else:
                    color = None
                
                label = f"{run_path}" if len(runs_to_plot) < 10 else None
                
                axes[idx].plot(freq[mask], data, 
                             linewidth=plot_cfg["linewidth"],
                             alpha=plot_cfg["alpha"],
                             color=color,
                             label=label)
                axes[idx].set_ylabel(f"{probe_col}")
                axes[idx].grid(plot_cfg["grid"], which="both", ls="--", alpha=0.3)
        else:
            # Plot all in one axis
            for probe_col in probe_cols:
                data = df[probe_col].values[mask]
                
                if proc_cfg["normalize"]:
                    data = data / data.max()
                if proc_cfg["smooth"]:
                    from scipy.ndimage import uniform_filter1d
                    data = uniform_filter1d(data, size=proc_cfg["smooth_window"])
                
                # Determine color
                if color_map and plot_cfg["color_by"] == "probe":
                    color = color_map[probe_col]
                elif color_map and plot_cfg["color_by"] == "file":
                    color = color_map[run_path]
                elif color_map and plot_cfg["color_by"] in meta_df.columns:
                    color = color_map[meta_df.loc[run_path, plot_cfg["color_by"]]]
                else:
                    color = None
                
                # Create informative label
                label_parts = []
                if len(runs_to_plot) < 10:
                    label_parts.append(str(run_path).split('/')[-1][:20])  # shortened filename
                label_parts.append(probe_col)
                label = " - ".join(label_parts)
                
                ax.plot(freq[mask], data,
                       linewidth=plot_cfg["linewidth"],
                       alpha=plot_cfg["alpha"],
                       color=color,
                       label=label)
    
    # Styling
    for axis in axes:
        if proc_cfg["log_scale"]:
            axis.set_yscale('log')
        axis.grid(plot_cfg["grid"], which="both", ls="--", alpha=0.3)
    
    axes[-1].set_xlabel("Frequency (Hz)")
    
    if not plot_cfg["separate_by_probe"]:
        ax.set_ylabel("PSD" if not proc_cfg["normalize"] else "Normalized PSD")
        if plot_cfg["show_legend"] and len(runs_to_plot) * len(probe_cols) <= 20:
            ax.legend(ncol=plot_cfg["legend_ncol"], fontsize='small')
    
    if plot_cfg["title"]:
        fig.suptitle(plot_cfg["title"])
    
    plt.tight_layout()
    plt.show()
    
    return fig, axes, runs_to_plot

# Usage
fig, axes, plotted = plot_psd_from_dict(psd_dictionary, meta_df, psd_plot_config)

# %%


chooseAll = False
plotvar = {
    "filters": {
        "amp": 0.1, #0.1, 0.2, 0.3 
        "freq": 1.3, #bruk et tall  
        "per": None, #bruk et tall #brukes foreløpig kun til find_wave_range, ennå ikke knyttet til filtrering
        "wind": "lowest", #full, no, lowest, all
        "tunnel": None,
        "mooring": "low"
    },
    "processing": {
        "chosenprobe": "Probe 2",
        "rangestart": None,
        "rangeend": None,
        "data_cols": ["Probe 2"],#her kan jeg velge fler, må huske [listeformat]
        "win": 11
    },
    "plotting": {
        "figsize": None,
        "separate":True,
        "overlay": False
        
    }
    
}

#unpack filters:
amp = plotvar["filters"]["amp"]
freq = plotvar["filters"]["freq"]
per = plotvar["filters"]["per"]

print(amp,freq)
print(amp+freq)





df = meta_sel.copy()


amp_rows = df[df["WaveAmplitudeInput [Volt]"] == amp]
freq_rows = df[df["WaveFrequencyInput [Hz]"] == freq]
both = df[(df["WaveAmplitudeInput [Volt]"] == amp) & (df["WaveFrequencyInput [Hz]"] == freq)]


# rows where col == "apple"
print(both)

#%%

def filter_for_amplitude_plot2(meta_df, plotvar,chooseAll=True):
    
    if chooseAll:
        return meta_df
    
    #unpack filters:
    amp = plotvar["filters"]["amp"]
    freq = plotvar["filters"]["freq"]
    per = plotvar["filters"]["per"]
        
    wanted = "amp"
    df = meta_df.copy()
    
    for idx, row in df.iterrows():
        
        if row["WaveAmplitudeInput [Volt]"] == amp:
            #add row to filtered df
            df[df["c"]]
    
    return filtered_df


# rows where col == "apple"
apple_rows = df[df["col"] == "apple"]
print(apple_rows)



#%%
fr = meta_sel.copy()
for idx, row, in fr.iterrows():
    P1 = row["Probe 1 Amplitude"]
    P2 = row["Probe 2 Amplitude"]
    P3 = row["Probe 3 Amplitude"]
    P4 = row["Probe 4 Amplitude"]
    
    if P1 != 0:
        fr.at[idx, "P2/P1"] = P2/P1
        
    if P2 != 0:
        fr.at[idx, "P3/P2"] = P3/P2
    
    if P3 != 0:
        fr.at[idx, "P4/P3"] = P4/P3



#%%
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
    
    
#
#GROKKOLINI:
    
    Recommended New Workflow (2025 Best Practice)
text1. load_or_update()               → raw dfs + basic meta
2. One single big meta DataFrame  → filter once (your df_sel)
3. Add computed columns directly in meta (no row-wise loop!)
4. update_processed_metadata(meta) → save
5. (Optional) Save processed DataFrames with new columns (eta, filtered, etc.)

#%%
print("Stillwater values and their types:")
for i in range(1,5):
    val = stillwater[f"Stillwater Probe {i}"]
    print(f"  Stillwater Probe {i} → {val!r}  (type: {type(val).__name__})")

#%% skulle debugge når 

#The real culprit is this line:
#sw = stillwater[f"Stillwater Probe {i}"]
print("FINDING THE EXACT FILE AND COLUMN THAT CAUSES THE ERROR\n" + "-"*60)

for path in dfs.keys():
    df = dfs[path]
    print(f"File: {Path(path).name}")
    for i in range(1, 5):
        possible_names = [f"Probe {i}", f"probe {i}", f"Probe{i}", f"probe{i}", str(i)]
        found = False
        for name in possible_names:
            if name in df.columns:
                col = name
                found = True
                break
        if not found:
            print(f"  Probe {i} column NOT FOUND in this file!")
            continue

        # Check if column is numeric
        series = df[col]
        if series.dtype == "object" or str(series.dtype).startswith("<U"):
            # definitely has strings
            bad_examples = series[pd.to_numeric(series, errors='coerce').isna()].unique()[:10]
            print(f"  BROKEN → '{col}' contains strings! Examples: {list(bad_examples)}")
        else:
            print(f"  OK → '{col}' is numeric ({series.dtype})")


#%%.-- denna er vel bare å slette
################### OLD 
def old_find_wave_range(
    df,  
    df_sel,  # detta er kun metadata for de utvalgte
    data_col,            
    detect_win=1,
    baseline_seconds=2.0,
    sigma_factor=5.0,
    skip_periods=None,
    keep_periods=None,
    range_plot=True
):
    #VARIABEL over^
    """
    Detect the start and end of the stable wave interval.

    Signal behavior (based on Ole's description):
      - baseline mean ~271
      - wave begins ~5 sec after baseline ends
      - ramp-up lasts ~12 periods
      - stable region has peaks ~280, troughs ~260

    Returns (good_start_idx, good_range)
    """

    # ==========================================================
    # AUTO-CALCULATE SKIP & KEEP PERIODS BASED ON REAL SIGNAL
    # ==========================================================
    importertfrekvens = 1# TK TK BYTT UT med en ekte import fra metadata
    # ---- Skip: baseline (5 seconds) + ramp-up (12 periods) ----
    if skip_periods is None:
        baseline_skip_periods = int(5 * importertfrekvens)        # 5 seconds worth of periods
        ramp_skip_periods     = 12 #VARIABEL # fixed from your signal observations
        skip_periods          = baseline_skip_periods + ramp_skip_periods

    # ---- Keep: x stable periods ----
    if keep_periods is None:
        keep_periods = 8#int(input_per-8) #VARIABEL
        
    # ==========================================================
    # PROCESSING STARTS HERE
    # ==========================================================
    
    """TODO TK: LEGGE TIL SAMMENLIKNING AV PÅFØLGENDE 
    BØLGE FOR Å SE STABILT signal. 
    OG Se på alle bølgetoppene"""
    
    """TODO TK: SJEKKE OM lengdene på periodene er like"""
    
    """TODO TK: """
    print(df.columns)
    print(df.head())
    print(df.dtypes)
    print(f"data_col before signal.. {data_col}")
    # 1) Smooth signal
    signal = (
        df[data_col]
        .rolling(window=detect_win, min_periods=1)
        .mean()
        .fillna(0)
        .values
    )

    # 2) Sample rate
    dt = (df["Date"].iloc[1] - df["Date"].iloc[0]).total_seconds()
    Fs = 1.0 / dt

    # 3) Baseline mean/std
    baseline_samples = int(baseline_seconds * Fs)
    baseline = signal[:baseline_samples]
    baseline_mean = np.mean(baseline)
    baseline_std  = np.std(baseline)

    threshold = sigma_factor * baseline_std
    movement = np.abs(signal - baseline_mean)

    # 4) First actual movement above noise floor
    first_motion_idx = np.argmax(movement > threshold)

    # 5) Convert frequency → samples per period
    T = 1.0 / float(importertfrekvens)
    samples_per_period = int(Fs * T)

    # 6) Final "good" window
    good_start_idx = first_motion_idx + skip_periods * samples_per_period
    good_range   = round(keep_periods * samples_per_period)

    # Clamp so that the last value is a real one
    good_start_idx = min(good_start_idx, len(signal) - 1)
    #good_range   = min(good_range,   len(signal) - 1)
    
    print('nu printes good start_idx', good_start_idx)
    print('nu printes good range', good_range)
    from wavescripts.plotter import plot_ramp_detection
    if range_plot:
        print('nu kjøres range_plot')
        plot_ramp_detection(
            df=df,
            df_sel=df_sel,
            data_col=data_col,
            signal=signal,
            baseline_mean=baseline_mean,
            threshold=threshold,
            first_motion_idx=first_motion_idx,
            good_start_idx=good_start_idx,
            good_range=good_range,
            title=f"Ramp Detection Debug – {data_col}"
        )

    debug_info = {
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "first_motion_idx": first_motion_idx,
        "samples_per_period": samples_per_period,
        "skip_periods": skip_periods,
        "keep_periods": keep_periods
    }

    return good_start_idx, good_range, debug_info

### klippet vekk fra processor:
    
    ################### GAMMAL KODE ########################
    def old_process_selected_data_old(
        dfs: dict[str, pd.DataFrame],
        meta_sel: pd.DataFrame,
        meta_full: pd.DataFrame,   # full meta of the experiment
        debug: bool = True,
        win: int = 10,
    ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:

        # 1. Ensure stillwater levels exist (idempotent — safe to call anytime)
        meta_full = ensure_stillwater_columns(dfs, meta_full)

        # Extract the values (they are the same in every row!)
        stillwater = {f"Stillwater Probe {i}": meta_full[f"Stillwater Probe {i}"].iloc[0] for i in range(1,5)}

        for i in range(1,5):
            val = stillwater[f"Stillwater Probe {i}"]
            print(f"  Stillwater Probe {i} → {val!r}  (type: {type(val).__name__})")
        print(f'stillwater = hva , jo: {stillwater}')
        # 2. Process only selected runs
        processed_dfs = {}
        for _, row in meta_sel.iterrows():
            path = row["path"]
            df = dfs[path].copy()
            # Clean columns once and for all
            for i in range(1, 5):
                col = f"Probe {i}"                    
            for i in range(1, 5):
                probe = f"Probe {2}"
                sw = stillwater[f"Stillwater Probe {i}"] 
                df[f"eta_{i}"] = df[probe] - sw
                df[f"{probe}_ma"] = df[f"eta_{i}"].rolling(win, center=False).mean()
        
            processed_dfs[path] = df
        
        # Add stillwater columns to meta_sel too (in case they were missing)
        for col, val in stillwater.items():
            if col not in meta_sel.columns:
                meta_sel[col] = val
        #for kolonne, value in dfs[riktig path]:
        
            # === DEBUG === #
        #find_wave_range(df_raw, df_sel,data_col=probe, detect_win=detect_window, debug=False)   
        
        # Ensure meta_sel knows which folder to save into
        if "PROCESSED_folder" not in meta_sel.columns:
            if "PROCESSED_folder" in meta_full.columns:
                folder = meta_full["PROCESSED_folder"].iloc[0]
            elif "experiment_folder" in meta_full.columns:
                folder = "PROCESSED-" + meta_full["experiment_folder"].iloc[0]
            else:
                raw_folder = Path(meta_full["path"].iloc[0]).parent.name
                folder = f"PROCESSED-{raw_folder}"
            
            meta_sel["PROCESSED_folder"] = folder
            print(f"Set PROCESSED_folder = {folder}")

        update_processed_metadata(meta_sel)

        return processed_dfs, meta_sel

    #######################
    # =============================================== 
    # === OLD OLD OLD Take in a filtered subset then process === #
    # ===============================================
    PROBES = ["Probe 1", "Probe 2", "Probe 3", "Probe 4"]
    def older_process_selected_data_old(dfs, df_sel, plotvariables):
        processed = {}
        debug_data ={}
        win      = plotvariables["processing"]["win"]

        for _, row in df_sel.iterrows():
            path = row["path"]
            df_raw = dfs[path]
            print("type(df_raw) =", type(df_raw))
        
            detect_window = 1 #10 er default i find_wave_range
            
            # === PROCESS ALL THE PROBES === #
            for probe in PROBES: #loope over alle 4 kolonnene
                print(f'probe in loop is: {probe}')
                # --- Apply moving avg. to the selected df_ma for each >probe in PROBES< 
                df_ma = apply_moving_average(df_raw, data_col=probe, win=win)
                print(f'df-ma per {probe} sitt tail:',df_ma[probe].tail())
                
                # find the start of the signal and optionally run the debug-plot
                start, end, debug_info = find_wave_range(df_raw, 
                                         df_sel,    
                                         data_col=probe,
                                         detect_win=detect_window, 
                                         debug=False) 
                df_sel[f"Computed {probe} start"] = start
            processed[path] = df_ma
            debug_data[path] = debug_info
        return processed, df_sel, debug_data 


    ######################