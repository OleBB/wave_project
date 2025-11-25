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