#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 10:21:49 2025

@author: ole
"""

import os
from pathlib import Path
import pandas as pd
import copy
import sys

from wavescripts.improved_data_loader import load_or_update
from wavescripts.filters import filter_chosen_files
from wavescripts.processor import process_selected_data
from wavescripts.processor2nd import process_processed_data 
from wavescripts.constants import SIGNAL, RAMP, MEASUREMENT, get_smoothing_window
from wavescripts.constants import (
    ProbeColumns as PC, 
    GlobalColumns as GC, 
    ColumnGroups as CG,
    CalculationResultColumns as RC,
    PlottPent as PP,
    WIND_COLOR_MAP

)


from PyQt5.QtWidgets import QApplication

# ── Filters ───────────────────────────────────────────────────────────────────
from wavescripts.filters import (
    apply_experimental_filters,
    damping_grouper,
    damping_all_amplitude_grouper,
    filter_dataframe,
    filter_for_amplitude_plot,
    filter_for_damping,
    filter_for_frequencyspectrum,
)

# ── Thesis plots ──────────────────────────────────────────────────────────────
from wavescripts.plotter import (
    gather_ramp_data,
    plot_all_probes,
    plot_damping_freq,
    plot_damping_scatter,
    plot_frequency_spectrum,
    plot_ramp_detection,
    plot_reconstructed,
    plot_reconstructed_rms,
    plot_swell_scatter,
)

# ── Quicklook / exploration ───────────────────────────────────────────────────
from wavescripts.plot_quicklook import (
    explore_damping_vs_amp,
    explore_damping_vs_freq,
    RampDetectionBrowser,
    save_interactive_plot,
    SignalBrowserFiltered,
)

# %%
file_dir = Path(__file__).resolve().parent
os.chdir(file_dir) # Make the script always run from the folder where THIS file lives

"""
Overordnet: Enhver mappe er en egen kjøring, som deler samme vanndyp og probestilltilstand.
En rekke prossesseringer skjer på likt for hele mappen.
Og så er det kode som sammenlikner data når hele mappen er prosessert en gang
"""

# List of dataset paths you want to process
dataset_paths = [
    #Path("/Users/ole/Kodevik/wave_project/wavedata/20251110-tett6roof-lowM-ekte580"),  # per15
    # Path("/Users/ole/Kodevik/wave_project/w-avedata/20251110-tett6roof-lowMooring"), #mstop 10
    
    # Path("/Users/ole/Kodevik/wave_project/wavedata/20251110-tett6roof-lowMooring-2"), #per15 (few runs)
    Path("/Users/ole/Kodevik/wave_project/wavedata/20251112-tett6roof"),
    Path("/Users/ole/Kodevik/wave_project/wavedata/20251112-tett6roof-lowM-579komma8"),
    Path("/Users/ole/Kodevik/wave_project/wavedata/20251113-tett6roof"),
    Path("/Users/ole/Kodevik/wave_project/wavedata/20251113-tett6roof-loosepaneltaped"),

    Path("/Users/ole/Kodevik/wave_project/wavedata/20251113-tett6roof-probeadjusted"),
    
]

#%% kjør
# Initialize containers for all results
all_meta_sel = []
all_processed_dfs = []
all_fft_dicts = []  
all_psd_dicts = []  

processvariables = {
    "overordnet": {
        "chooseAll": True,
        "chooseFirst": False, #velger første i hver mappe
    },
    "filters": {
        "WaveAmplitudeInput [Volt]": [0.1],  # 0.1, 0.2, 0.3 
        "WaveFrequencyInput [Hz]": 1.3,  # bruk et tall  
        "WavePeriodInput": None,  # bruk et tall #brukes foreløpig kun til find_wave_range, ennå ikke knyttet til filtrering
        "WindCondition": None,#["full"],  # full, no, lowest
        "TunnelCondition": None,
        "Mooring": "low",
        "PanelCondition": None #["reverse"]#, "reverse"],  # no, full, reverse, 
    }, 
    "prosessering": {
        "total_reset": False, #laster også csv'ene på nytt
        "force_recompute": False, #kjører alt på nytt, ignorerer gammal json
        "debug": False,
        "smoothing_window": 10, #kontrollere denne senere
        "find_range": False,
        "range_plot": False,    
    },
}

prosessering = processvariables.get("prosessering", {})
total_reset = prosessering.get("total_reset", False)
if total_reset:
    input("TOTAL RESET! press enter if you want to continue")

for i, data_path in enumerate(dataset_paths):
    print(f"\n{'='*50}")
    print(f"Processing dataset {i+1}/{len(dataset_paths)}: {data_path.name}")
    print(f"{'='*50}")
    try:
        force = prosessering.get("force_recompute", False)
        dfs, meta = load_or_update(data_path, force_recompute=force, total_reset=total_reset)
        
        print('# === Filter === #')
        meta_sel = filter_chosen_files(meta, processvariables)
        
        print('# === Single probe process === #')
        processed_dfs, meta_sel, psd_dictionary, fft_dictionary = process_selected_data(dfs, meta_sel, meta, processvariables)
        
        print('# === Probe comparison processing === #')
        meta_sel = process_processed_data(psd_dictionary, fft_dictionary, meta_sel, meta, processvariables)
        
        # Collect results
        all_meta_sel.append(meta_sel)
        all_processed_dfs.append(processed_dfs)
        all_fft_dicts.append(fft_dictionary)
        all_psd_dicts.append(psd_dictionary)
        
        print(f"Successfully processed {len(meta_sel)} selections from {data_path.name}")
        
    except Exception as e:
        print(f"Error processing {data_path.name}: {str(e)}")
        continue

print(f"\n{'='*50}")
print(f"PROCESSING COMPLETE - Total datasets processed: {len(all_meta_sel)}")
print(f"Total selections across all datasets: {sum(len(sel) for sel in all_meta_sel)}")
print(f"{'='*50}")

if all_meta_sel:
    # Combine metadata
    combined_meta_sel = pd.concat(all_meta_sel, ignore_index=True)
    print("\nCombined meta_selections ready for analysis")
    
    # Combine FFT dictionaries
    combined_fft_dict = {}
    for fft_dict in all_fft_dicts:
        combined_fft_dict.update(fft_dict)
        
    # Combine psd dictionaries
    combined_psd_dict = {}
    for psd_dict in all_psd_dicts:
        combined_psd_dict.update(psd_dict)
    
    #combine processed dicts
    combined_processed_dfs = {}
    for processed_dict in all_processed_dfs:
        combined_processed_dfs.update(processed_dict)
        
  
        
    
    print(f"Combined processed dictionary contains {len(combined_processed_dfs)} experiments")
    print(f"Combined FFT dictionary contains {len(combined_fft_dict)} experiments")
    print(f"Combined metadata contains {len(combined_meta_sel)} rows")
    # Verify they match
    dfs_paths = set(combined_processed_dfs.keys())
    fft_paths = set(combined_fft_dict.keys())
    meta_paths = set(combined_meta_sel['path'].unique())
    print(f"\nPaths in processed_dfs dict: {len(dfs_paths)}")
    print(f"Paths in FFT dict: {len(fft_paths)}")
    print(f"Unique paths in metadata: {len(meta_paths)}")
    print(f"Matching paths: {len(fft_paths & meta_paths)}")
    
    del all_meta_sel, all_processed_dfs, all_fft_dicts, all_psd_dicts, dfs, meta, meta_sel, processed_dfs, dfs_paths, fft_paths,meta_paths 
    import gc
    gc.collect()
    print("bosset e tatt ut")
    
    # - Compare results between different datasets
# """PRINT RESULTS"""
# from wavescripts.wavestudyer import wind_damping_analysis
# damping_analysis_results = wind_damping_analysis(combined_meta_sel)

# %%
# TOdo - sjekke ekte sample rate
# todo - fjerne outliers



# %% [markdown] 
# Nå, hvordan beveger bølgen seg gjennom tanken. Vi kikker på den gjennomsnittlige bølgeamplituden sett ved hver av de fire måleprobene. I plottet under er avstandene mellom probene ikke korrekt representert, bare rekkefølgen. 
# . - . 
# %% fysisk plott
amplitudeplotvariables = {
    "overordnet": {
        "chooseAll": False,
        "chooseFirst": False,
        "chooseFirstUnique": False,
        
    },
    "filters": {
        "WaveAmplitudeInput [Volt]": [0.1,0.2,0.3], #0.1, 0.2, 0.3 
        "WaveFrequencyInput [Hz]": [1.3, 0.65], #bruk et tall  
        "WavePeriodInput": None, #bruk et tall #brukes foreløpig kun til find_wave_range, ennå ikke knyttet til filtrering
        "WindCondition": ["no", "lowest", "full"], #full, no, lowest, all
        "TunnelCondition": None,
        "Mooring": "low",
        "PanelCondition": ["full", "reverse", "no"], # no, full, reverse, 
        
    },
    "processing": {
        "chosenprobe": "Probe 2",
        "rangestart": None,
        "rangeend": None,
        "data_cols": ["Probe 2"],#her kan jeg velge fler, må huske [listeformat]
        "win": 11
    },
    "plotting": {
        "figsize": [20,10],
        "separate":True,
        "overlay": False,
        "annotate": True   
    }   
}

"""unikt filter for å se på amplitudene sjæl"""
m_filtrert = apply_experimental_filters(combined_meta_sel, amplitudeplotvariables)
# Plot_all_probes plotter alt den tar inn
plot_all_probes(m_filtrert, amplitudeplotvariables)
print("======== Amplituder P1234 PLOTTA ===========")

#%% grouper - slår i hop
damping_groupedruns_df, damping_pivot_wide = damping_grouper(combined_meta_sel)
# %% lagrer en interaktiv fil som man kan leke med
save_interactive_plot(damping_groupedruns_df)
print("tøff interaktiv greie i nettleser")
# %%


# %% damping variables initiert
dampingplotvariables = {
    "overordnet": {
        "chooseAll": True,
        "chooseFirst": False,
        "chooseFirstUnique": False,
    }, 
    "filters": {
        "WaveAmplitudeInput [Volt]": [0.1, 0.2, 0.3], #0.1, 0.2, 0.3 
        "WaveFrequencyInput [Hz]": [1.3, 0.65], #bruk et tall  
        "WavePeriodInput": None, #bruk et tall #brukes foreløpig kun til find_wave_range, ennå ikke knyttet til filtrering
        "WindCondition": ["no", "lowest", "full"], #full, no, lowest, all
        "TunnelCondition": None,
        # "Mooring": None,
        "PanelCondition": None #["full", "reverse"], # no, full, reverse, 
        
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
        "overlay": False,
        "annotate": True   
    }   
}

damping_filtrert = filter_for_damping(damping_groupedruns_df, dampingplotvariables["filters"])

# %% plotting damping frequencies seaborn

# explore_facet_plot_freq_vs_mean(damping_filtrert, dampingplotvariables)
explore_damping_vs_freq(damping_filtrert, dampingplotvariables)

# %% plotting damping amplitudes seaborn 
explore_damping_vs_amp(damping_filtrert, dampingplotvariables)




# %% slår alle i hop
dampingplotvariables = {
    "overordnet": {
        "chooseAll": False, 
        "chooseFirst": False,
    }, 
    "filters": {
        "WaveAmplitudeInput [Volt]": [0.1],# 0.2, 0.3], #0.1, 0.2, 0.3 
        "WaveFrequencyInput [Hz]": [1.3],# 0.65], #bruk et tall  
        "WavePeriodInput": None, #bruk et tall #brukes foreløpig kun til find_wave_range, ennå ikke knyttet til filtrering
        "WindCondition": ["no", "lowest", "full"], #full, no, lowest, all
        "TunnelCondition": None,
        #"Mooring": None,
        "PanelCondition": ["no", "full", "reverse"], # no, full, reverse, 
        
    },
    "processing": {
        "chosenprobe": 1, #[1,2,3,4]
        "rangestart": None,
        "rangeend": None,
        "data_cols": ["Probe 2"],
        "win": 11 #Ingen av disse er egt i bruk
    },
    "plotting": {
        "show_plot": True,
        "save_plot": True,
        "figsize": None,
        "separate":False,
        "facet_by": None, #wind, panel, probe 
        "overlay": False,
        "annotate": True, 
        "legend": "outside_right", # inside, below, above #med mer!
        "logaritmic": False, 
        "peaks": 7, 
        "probes": [1,2,3,4],
    }   
}


damping_groupedallruns_df  = damping_all_amplitude_grouper(combined_meta_sel)
# %% plotter seaborn facet med dempning 
plot_damping_freq(damping_groupedallruns_df,dampingplotvariables)
# %%

# uviss...
# damping_all_amplitudes_filtrert = filter_for_damping(damping_groupedallruns_df, dampingplotvariables["filters"])

# %% plotter damping scatter 
plot_damping_scatter(damping_groupedallruns_df, dampingplotvariables)
#kunne lagt til plotvariabler her og.. 


# %% FFT-SPEKTRUM filter initiert
freqplotvariables = {
    "overordnet": {
        "chooseAll": True, 
        "chooseFirst": False,
        "chooseFirstUnique": False,
    }, 
    "filters": {
        "WaveAmplitudeInput [Volt]": [0.1],# 0.2, 0.3], #0.1, 0.2, 0.3 
        "WaveFrequencyInput [Hz]": [1.3],# 0.65], #bruk et tall  
        "WavePeriodInput": None, #bruk et tall #brukes foreløpig kun til find_wave_range, ennå ikke knyttet til filtrering
        "WindCondition": ["no", "lowest", "full"], #full, no, lowest, all
        "TunnelCondition": None,
        "Mooring": None,
        "PanelCondition": "reverse", #["no", "full", "reverse"], # no, full, reverse,  #kan grupperes i filters.
        
    },
    "processing": {
        "chosenprobe": 1, #[1,2,3,4]
        "rangestart": None,
        "rangeend": None,
        "data_cols": ["Probe 2"],
        "win": 11 #Ingen av disse er egt i bruk
    },
    "plotting": {
        "show_plot": True,
        "save_plot": True,
        "figsize": (10,12), #(10,10),
        "linewidth": 0.7,
        "separate":False,
        "facet_by": "probe", #wind", #wind, panel, probe 
        "overlay": False, #
        "annotate": False, #
        "max_points": 120, #spørs på oppløsning av fft'en.
        "xlim": (0,5.2), #4x 1.3
        "legend": "inside", #"outside_right", # inside, below, above #med mer!
        "logaritmic": False, 
        "peaks": 3, 
        "probes": [2,3]
    }   
}
#lærte noe nytt - #dict.get(key, default) only falls back when the key is missing.

filtrert_frequencies = filter_for_frequencyspectrum(combined_meta_sel, freqplotvariables)
print("filter applied")

# %% plotter alt den tar inn...
fig, axes = plot_frequency_spectrum(
    combined_fft_dict,
    filtrert_frequencies,
    freqplotvariables,
    data_type="fft"
)
# %% plotter PSD facet, men bare noe
fig, axes = plot_frequency_spectrum(
    combined_psd_dict,  # Your PSD data dictionary
    filtrert_frequencies, 
    freqplotvariables,
    data_type="psd"
)

# %% NY GROUPER - Eller er det en avsporing. må ha dataen i metasel først uansett. 
"""################################"""
#%% NY gruppering - slår i hop - nye navn
#hopp over foreløig from wavescripts.filters import swell_grouper
#hopp over swell_groupedruns_df, swell_pivot_wide = swell_grouper(combined_meta_sel)
# %% damping variables initiert
swellplotvariables = {
    "overordnet": {
        "chooseAll": True, 
        "chooseFirst": False,
        "chooseFirstUnique": True,
    }, 
    "filters": {
        "WaveAmplitudeInput [Volt]": [0.1, 0.2, 0.3],# 0.2, 0.3], #0.1, 0.2, 0.3 
        "WaveFrequencyInput [Hz]": [1.3],# 0.65], #bruk et tall  
        "WavePeriodInput": None, #bruk et tall #brukes foreløpig kun til find_wave_range, ennå ikke knyttet til filtrering
        "WindCondition": ["no", "lowest", "full"], #full, no, lowest, all
        "TunnelCondition": None,
        "Mooring": None,
        "PanelCondition": "reverse", #["no", "full", "reverse"], # no, full, reverse,  #kan grupperes i filters.
        
    },
    "processing": {
        "chosenprobe": 1, #[1,2,3,4]
        "rangestart": None,
        "rangeend": None,
        "data_cols": ["Probe 2"],
        "win": 11 #Ingen av disse er egt i bruk
    },
    "plotting": {
        "show_plot": True,
        "save_plot": True,
        "figsize": (10,12), #(10,10),
        "linewidth": 0.7,
        "separate":False,
        "facet_by": "probe", #wind", #wind, panel, probe 
        "overlay": False, #
        "annotate": False, #
        "max_points": 120, #spørs på oppløsning av fft'en.
        "xlim": (0,5.2), #4x 1.3
        "legend": "inside", #"outside_right", # inside, below, above #med mer!
        "logaritmic": False, 
        "peaks": 3, 
        "probes": [2,3]
    }   
}

# from wavescripts.filters import filter_for_swell
swell_filtrert = filter_for_amplitude_plot(combined_meta_sel, swellplotvariables)
band_amplitudes = swell_filtrert
# %% plotting damping frequencies seaborn
# funkekje! # facet_swell(damping_filtrert, swellplotvariables)

# %% claude, som caller filter internt
plot_swell_scatter(combined_meta_sel, swellplotvariables)

# %% seaborn plot med 3 swell facets, full, svak og null wind. 
plot_swell_scatter(band_amplitudes, combined_meta_sel)


# %% damping wide - ikke i bruk
# wide = damping_pivot_wide
# print(wide.columns.tolist())

# mean_cols = [c for c in wide.columns if c.startswith("mean_P3P2_")]
# wide_means = wide[["WaveAmplitudeInput [Volt]", "PanelConditionGrouped"] + mean_cols]

# mask = (
#     (wide["WaveAmplitudeInput [Volt]"] == 0.5)
#     & (wide["PanelConditionGrouped"] == "all")
# )
# row = wide.loc[mask]

# wide["delta_mean_P3P2_Windyyyy"] = (
#     wide["mean_P3P2_lowest"] - wide["mean_P3P2_full"]
# )


# %% todo: lage funksjon for å kjøre range_plot utenom prosessering

# from wavescripts.plotter import plot_ramp_detection

# forløkke velge fil. 
# hente ut index
# figr, axr  = plot_ramp_detection(df, meta_sel, data_col, signal, baseline_mean, threshold, first_motion_idx, good_start_idx, good_range, good_end_idx)

# %% åssen endrer bølgetallet seg?

colls = [CG.fft_wave_dimension_cols(i) for i in range(1,5)]
meta_sel_wavenumberstudy = combined_meta_sel.copy()
hey = [CG.FFT_WAVENUMBER_COLS].copy()

# %% FFT-SPEKTRUM  initiert
from wavescripts.plotter import plot_frequency_spectrum
freqplotvariables = {
    "overordnet": {
        "chooseAll": False, 
        "chooseFirst": True,
        "chooseFirstUnique": False,
    }, 
    "filters": {
        "WaveAmplitudeInput [Volt]": [0.1],# 0.2, 0.3], #0.1, 0.2, 0.3 
        "WaveFrequencyInput [Hz]": [1.3],# 0.65], #bruk et tall  
        "WavePeriodInput": None, #bruk et tall #brukes foreløpig kun til find_wave_range, ennå ikke knyttet til filtrering
        "WindCondition": ["no", "lowest", "full"], #full, no, lowest, all
        "TunnelCondition": None,
        "Mooring": None,
        "PanelCondition": "reverse", #["no", "full", "reverse"], # no, full, reverse,  #kan grupperes i filters.
        
    },
    "processing": {
        "chosenprobe": 1, #[1,2,3,4]
        "rangestart": None,
        "rangeend": None,
        "data_cols": ["Probe 2"],
        "win": 11 #Ingen av disse er egt i bruk
    },
    "plotting": {
        "show_plot": True,
        "save_plot": True,
        "save_separate": True,
        "figsize": (18,18), #(10,10),
        "linewidth": 1,
        "grid": True,
        "show_full_signal": True,
        "dual_yaxis": False, #for å skalere opp vindstøyen og se den tydeligere.
        "separate":True,
        "facet_by": "probe", #wind", #wind, panel, probe 
        "overlay": False, #
        "annotate": False, #
        "max_points": 120, #spørs på oppløsning av fft'en.
        "xlim": (0,5.2), #4x 1.3
        "legend": None, #"outside_right", # inside, below, above #med mer!
        "logaritmic": False, 
        "peaks": 3, 
        "probes": [2,3]
    }   
}
#lærte noe nytt - #dict.get(key, default) only falls back when the key is missing.

filtrert_frequencies = filter_for_frequencyspectrum(combined_meta_sel, freqplotvariables)

# %% kopiert fra oven plotter fft facet
fig, axes = plot_frequency_spectrum(
    combined_fft_dict,
    filtrert_frequencies,
    freqplotvariables,
    data_type="fft"
)


# les av fft_dict -> les av tabell. loope probe 2 og 3. 
# plotte probe 2 dekomponert. 
# %%

# fig, axes = plot_reconstructed(combined_fft_dict, 
#                                filtrert_frequencies,
#                                freqplotvariables,
#                                data_type="fft")
# %% Hent ut fft og matchende paths.
fft_paths = set(combined_fft_dict.keys())
meta_paths = set(combined_meta_sel['path'].unique())
matching_paths = fft_paths & meta_paths

# Create filtered versions
filtered_fft_dict = {p: combined_fft_dict[p] for p in matching_paths}
filtered_meta = combined_meta_sel[combined_meta_sel['path'].isin(matching_paths)]

print(f"\nReady to plot {len(filtered_fft_dict)} experiments")

# Plot a single experiment
single_path = list(filtered_fft_dict.keys())[1] #velg én
single_meta = filtered_meta[filtered_meta['path'] == single_path]

fig, ax = plot_reconstructed(
    {single_path: filtered_fft_dict[single_path]},
    single_meta,
    freqplotvariables
)
# %% SIGNALplot - for å sammenlikne amplitude med sann amplitude
fig, axes = plot_reconstructed(combined_fft_dict, 
                               filtrert_frequencies,
                               freqplotvariables,
                               data_type="fft")

# %% Kjør interaktiv plotter av dekomponert signal.
# ── Launch ─────────────────────────────────────────────────────

app = QApplication.instance() or QApplication(sys.argv)
browser = SignalBrowserFiltered(filtered_fft_dict, filtered_meta, freqplotvariables)
browser.show()

# %%



# %%

ramp_df = gather_ramp_data(combined_processed_dfs, combined_meta_sel)

# Launch browser
app = QApplication.instance() or QApplication(sys.argv)
browser = RampDetectionBrowser(ramp_df)
browser.show()
# %%


# %% debug som fant ut at noen runs ikke har Wave, så de filtreres ut 
# # Debug: Check what your FFT data actually looks like
# paf = list(combined_fft_dict.keys())[0]  # Get first path
# df_fft = combined_fft_dict[paf]

# print("FFT DataFrame info:")
# print(f"Shape: {df_fft.shape}")
# print(f"Index (first 10): {df_fft.index[:10].tolist()}")
# print(f"Index (last 10): {df_fft.index[-10:].tolist()}")
# print(f"Columns: {df_fft.columns.tolist()}")
# print(f"\nIndex name: {df_fft.index.name}")
# print(f"Index dtype: {df_fft.index.dtype}")

# # Check if there's a frequency column instead
# if 'frequency' in df_fft.columns or 'freq' in df_fft.columns:
#     print("\n⚠️ Frequency is a COLUMN, not the index!")

# %% __main__


# if __name__ == "__main__":
    # print('running main')
