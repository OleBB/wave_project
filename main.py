#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 10:21:49 2025

@author: ole
"""

import os
from pathlib import Path
import pandas as pd
file_dir = Path(__file__).resolve().parent
os.chdir(file_dir) # Make the script always run from the folder where THIS file lives
from wavescripts.data_loader import load_or_update
from wavescripts.filters import filter_chosen_files
from wavescripts.processor import process_selected_data
from wavescripts.processor2nd import process_processed_data

"""
Overordnet: Enhver mappe er en egen kjøring, som deler samme vanndyp og probestilltilstand.
En rekke prossesseringer skjer på likt for hele mappen.
Og så er det kode som sammenlikner data når hele mappen er prosessert en gang
"""

# List of dataset paths you want to process
dataset_paths = [
    #Path("/Users/ole/Kodevik/wave_project/wavedata/20251110-tett6roof-lowM-ekte580"),  # per15
    
    Path("/Users/ole/Kodevik/wave_project/wavedata/20251110-tett6roof-lowMooring"),
    Path("/Users/ole/Kodevik/wave_project/wavedata/20251110-tett6roof-lowMooring-2"),
    Path("/Users/ole/Kodevik/wave_project/wavedata/20251112-tett6roof"),
    # Path("/Users/ole/Kodevik/wave_project/wavedata/20251112-tett6roof-lowM-579komma8"),
    # Path("/Users/ole/Kodevik/wave_project/wavedata/20251113-tett6roof"),
    # Path("/Users/ole/Kodevik/wave_project/wavedata/20251113-tett6roof-loosepaneltaped"),
    
    # Path("/Users/ole/Kodevik/wave_project/wavedata/20251113-tett6roof-probeadjusted"),
    
]
#%%
# Initialize containers for all results
all_meta_sel = []
all_processed_dfs = []

processvariables = {
    "overordnet": {
        "chooseAll": True,
        "chooseFirst": False,
    },
    "filters": {
        "amp": [0.1],  # 0.1, 0.2, 0.3 
        "freq": 1.3,  # bruk et tall  
        "per": None,  # bruk et tall #brukes foreløpig kun til find_wave_range, ennå ikke knyttet til filtrering
        "wind": None,#["full"],  # full, no, lowest
        "tunnel": None,
        "mooring": "low",
        "panel": None #["reverse"]#, "reverse"],  # no, full, reverse, 
    }, 
    "prosessering": {
        "debug": False,
        "smoothing window": 10, 
        "find_range": True,
        "range_plot": False,    
        "force_recompute": False,
    },
}

# Loop through each dataset
for i, data_path in enumerate(dataset_paths):
    print(f"\n{'='*50}")
    print(f"Processing dataset {i+1}/{len(dataset_paths)}: {data_path.name}")
    print(f"{'='*50}")
    try:
        prosessering = processvariables.get("prosessering", {})
        force =prosessering.get("force_recompute", False)
        dfs, meta = load_or_update(data_path, force_recompute=force)
        
        print('# === Filter === #') #dette filteret er egentlig litt unøding, når jeg ønsker å prossesere hele sulamitten
        meta_sel = filter_chosen_files(meta, processvariables)
        
        print('# === Single probe process === #')
        processed_dfs, meta_sel, psd_dictionary, fft_dictionary = process_selected_data(dfs, meta_sel, meta, processvariables)
        
        print('arbeider her, med FFT av alle disse per folder')
        print('# === FTT on each separate signal, saved to a dict of dfs')
        # ftt_dfs = process_psd(processed_dfs)
        
        print('# === Probe comparison processing === #')
        meta_sel = process_processed_data(meta_sel)
        all_meta_sel.append(meta_sel)
        all_processed_dfs.append(processed_dfs)
        print(f"Successfully processed {len(meta_sel)} selections from {data_path.name}")
        
    except Exception as e:
        print(f"Error processing {data_path.name}: {str(e)}")
        continue
print(f"\n{'='*50}")
print(f"PROCESSING COMPLETE - Total datasets processed: {len(all_meta_sel)}")
print(f"Total selections across all datasets: {sum(len(sel) for sel in all_meta_sel)}")
print(f"{'='*50}")

if all_meta_sel:
    combined_meta_sel = pd.concat(all_meta_sel, ignore_index=True)
    print("\nCombined meta_selections ready for analysis:")
    print(combined_meta_sel.head())
    
    # You can now analyze the combined data
    # For example:
    # - Count selections by dataset
    # - Analyze properties across all selections
    # - Compare results between different datasets
# """PRINT RESULTS"""
# from wavescripts.wavestudyer import wind_damping_analysis
# damping_analysis_results = wind_damping_analysis(combined_meta_sel)



# %% fysisk plott
chooseAll = False
amplitudeplotvariables = {
    "filters": {
        "WaveAmplitudeInput [Volt]": [0.1,0.2,0.3], #0.1, 0.2, 0.3 
        "WaveFrequencyInput [Hz]": 1.3, #bruk et tall  
        "WavePeriodInput": None, #bruk et tall #brukes foreløpig kun til find_wave_range, ennå ikke knyttet til filtrering
        "WindCondition": ["no", "lowest", "full"], #full, no, lowest, all
        "TunnelCondition": None,
        "Mooring": "low",
        "PanelCondition": ["full", "reverse"], # no, full, reverse, 
        
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
from wavescripts.filters import filter_for_amplitude_plot
m_filtrert = filter_for_amplitude_plot(combined_meta_sel, amplitudeplotvariables, chooseAll)
# %% Plot_all_probes plotter alt den tar inn
from wavescripts.plotter import plot_all_probes
plot_all_probes(m_filtrert, amplitudeplotvariables)

print("======== Amplituder P1234 PLOTTA ===========")

#%% grouper - slå i hop
from wavescripts.filters import damping_grouper
damping_groupedruns_df, damping_pivot_wide = damping_grouper(combined_meta_sel)
# %%
chooseAll = False
dampingplotvariables = {
    "overordnet": {"chooseAll": False}, 
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

from wavescripts.filters import filter_for_damping
damping_filtrert = filter_for_damping(damping_groupedruns_df, dampingplotvariables["filters"])

# %% facet damping seaborn
from wavescripts.plotter import facet_plot_freq_vs_mean
facet_plot_freq_vs_mean(damping_filtrert, dampingplotvariables)

# %% damping amp seaborn 
from wavescripts.plotter import facet_plot_amp_vs_mean
facet_plot_amp_vs_mean(damping_filtrert, dampingplotvariables)

# %% grouped
chooseAll = False
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
        "figsize": None,
        "separate":False,
        "facet_by": "panel", #wind, panel, probe 
        "overlay": False,
        "annotate": True, 
        "legend": "outside_right", # inside, below, above #med mer!
        "logaritmic": False, 
        "peaks": 7, 
        "probes": [1,2,3,4],
    }   
}


"""Slå alle i hop"""
from wavescripts.filters import damping_all_amplitude_grouper
damping_groupedallruns_df  = damping_all_amplitude_grouper(combined_meta_sel)

# damping_all_amplitudes_filtrert = filter_for_damping(damping_groupedallruns_df, dampingplotvariables["filters"])

from wavescripts.plotter import facet_amp
facet_amp(damping_groupedallruns_df, dampingplotvariables)

# %% FFT-SPEKTRUM filter
freqplotvariables = {
    "overordnet": {
        "chooseAll": False, 
        "chooseFirst": False,
        "chooseFirstUnique": True,
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
        "figsize": None,
        "separate":False,
        "facet_by": "wind", #wind, panel, probe 
        "overlay": False, #
        "annotate": False, #
        "xlim": (0,5),
        "legend": None, #"outside_right", # inside, below, above #med mer!
        "logaritmic": False, 
        "peaks": 3, 
        "probes": [2, 3],
        "runs": 1
        
        
    }   
}

from wavescripts.filters import filter_for_frequencyspectrum
filtrert_frequencies = filter_for_frequencyspectrum(meta_sel, freqplotvariables)

# %% cLAUdE - gi fn nytt navn seinar
from wavescripts.plotter import plot_frequency_spectrum
fig, axes = plot_frequency_spectrum(
    fft_dictionary,
    filtrert_frequencies,
    freqplotvariables
)

# %% forsøk på facet plot
from wavescripts.plotter import plot_facet_frequencyspectrum
fig, axes = plot_facet_frequencyspectrum(fft_dictionary, filtrert_frequencies, freqplotvariables)
# %%
from wavescripts.plotter import plot_facet_condition_frequencyspectrum
fig_fft, axes_fft = plot_facet_condition_frequencyspectrum(fft_dictionary, filtrert_frequencies, freqplotvariables)

# %% 
from wavescripts.plotter import plot_frequencyspectrum
plot_frequencyspectrum(fft_dictionary,filtrert_frequencies, freqplotvariables)

from wavescripts.plotter import plot_powerspectraldensity
plot_powerspectraldensity(psd_dictionary, filtrert_frequencies, freqplotvariables)

# %%


# python
import pandas as pd
import matplotlib.pyplot as plt

# df_plot has columns from your psd_dictionary (as in your example)
first_cols = {k: d.iloc[:,0] for k, d in psd_dictionary.items()}
df_plot = pd.concat(first_cols, axis=1)
# Get only first half of the dictionary items
halfway = len(psd_dictionary) // 2
first_half_items = dict(list(psd_dictionary.items())[:halfway])

# %%


# first_cols = {k: d.iloc[:, 2] for k, d in first_half_items.items()}
# df_plot = pd.concat(first_cols, axis=1)

# fig, ax = plt.subplots(figsize=(7, 4))

# # Iterate columns for full control
# for name in df_plot.columns:
#     ax.plot(df_plot.index, df_plot[name], label=str(name), linewidth=1.5, marker=None)

# ax.set_xlabel("freq (Hz)")
# # ax.set_ylabel("PSD")
# ax.set_xlim(0, 10)
# ax.grid(True, which="both", ls="--", alpha=0.3)
# # ax.legend(title="Series", ncol=2)  # or remove if not needed
# plt.tight_layout()
# plt.show()


# %% test av psd dict
# Get only first half of the dictionary items
halfway = len(psd_dictionary) // 2
first_half_items = dict(list(psd_dictionary.items())[:halfway])

# Extract both columns
col1_data = {k: d.iloc[:, 1] for k, d in first_half_items.items()}
col2_data = {k: d.iloc[:, 2] for k, d in first_half_items.items()}

df_col1 = pd.concat(col1_data, axis=1)
df_col2 = pd.concat(col2_data, axis=1)

# A4 size in inches (portrait: 8.27 x 11.69, landscape: 11.69 x 8.27)
fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27), dpi=300)

# Plot column 1 in first facet
for name in df_col1.columns:
    short_name = str(name)[66:120]
    axes[0].plot(df_col1.index, df_col1[name], label=short_name, linewidth=1.5)
axes[0].set_xlabel("freq (Hz)")
axes[0].set_ylabel("PSD")
axes[0].set_title("Column 1")
axes[0].set_xlim(0, 10)
axes[0].grid(True, which="both", ls="--", alpha=0.3)

# Plot column 2 in second facet
for name in df_col2.columns:
    short_name = str(name)[66:120]
    axes[1].plot(df_col2.index, df_col2[name], label=short_name, linewidth=1.5)
axes[1].set_xlabel("freq (Hz)")
axes[1].set_ylabel("PSD")
axes[1].set_title("Column 2")
axes[1].set_xlim(0, 10)
axes[1].grid(True, which="both", ls="--", alpha=0.3)

# Make y-axis limits equal
y_min = min(axes[0].get_ylim()[0], axes[1].get_ylim()[0])
y_max = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
axes[0].set_ylim(y_min, y_max)
axes[1].set_ylim(y_min, y_max)

# Add shared legend below the plots
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.02), 
           ncol=4, frameon=False)

plt.tight_layout()

# Save as high-resolution image for direct printing
plt.savefig('plot_A4.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
# %%
import numpy as np

def compute_amplitude_by_band(psd_dictionary, freq_bands=None):
    """Compute amplitude for specific frequency bands from PSD"""
    if freq_bands is None:
        freq_bands = {
            'swell': (1.0, 1.6),
            'wind_waves': (3.0, 10.0),
            'total': (0.0, 10.0)
        }
    
    results = []
    for path, psd_df in psd_dictionary.items():
        row_out = {'path': path}
        
        for i in range(1, 5):
            col = f'Pxx {i}'
            if col not in psd_df.columns:
                continue
                
            freq_res = psd_df.index[1] - psd_df.index[0]
            
            for band_name, (f_low, f_high) in freq_bands.items():
                band_mask = (psd_df.index >= f_low) & (psd_df.index <= f_high)
                variance = psd_df[band_mask][col].sum() * freq_res
                amplitude = 2 * np.sqrt(variance)
                row_out[f'Probe {i} {band_name} amplitude'] = amplitude
        
        results.append(row_out)
    
    return pd.DataFrame(results)

# Use it:
band_amplitudes = compute_amplitude_by_band(psd_dictionary)
print(band_amplitudes)
# %% amp by band

def compute_amplitude_by_band(psd_dictionary, freq_bands=None, verbose=False):
    """Compute amplitude for specific frequency bands from PSD"""
    if freq_bands is None:
        freq_bands = {
            'swell': (1.0, 1.6),
            'wind_waves': (3.0, 10.0),
            'total': (0.0, 10.0)
        }
    
    results = []
    for path, psd_df in psd_dictionary.items():
        row_out = {'path': path}
        
        if verbose:
            print(f"\n=== Path: {path} ===")
            print(f"Freq range: {psd_df.index.min():.3f} to {psd_df.index.max():.3f} Hz")
        
        for i in range(1, 5):
            col = f'Pxx {i}'
            if col not in psd_df.columns:
                continue
            
            # Calculate frequency resolution
            freq_res = psd_df.index[1] - psd_df.index[0]
            
            if verbose and i == 1:
                print(f"Frequency resolution: {freq_res:.4f} Hz")
            
            for band_name, (f_low, f_high) in freq_bands.items():
                band_mask = (psd_df.index >= f_low) & (psd_df.index <= f_high)
                n_points = band_mask.sum()
                
                if n_points == 0:
                    if verbose:
                        print(f"  {band_name}: NO DATA POINTS in band [{f_low}, {f_high}] Hz")
                    row_out[f'Probe {i} {band_name} amplitude'] = 0.0
                    continue
                
                # Integrate PSD to get variance
                psd_band = psd_df[band_mask][col]
                variance = psd_band.sum() * freq_res
                std_dev = np.sqrt(variance)
                
                # Standard wave amplitude estimate (peak-to-trough ≈ 2σ for sinusoid)
                amplitude = 2 * std_dev
                
                if verbose and i == 1:
                    print(f"  {band_name} [{f_low}-{f_high} Hz]: {n_points} points, "
                          f"variance={variance:.6f}, amplitude={amplitude:.4f}")
                
                row_out[f'Probe {i} {band_name} amplitude'] = amplitude
        
        results.append(row_out)
    
    return pd.DataFrame(results)
# Run with diagnostics
band_amplitudes = compute_amplitude_by_band(psd_dictionary, verbose=True)
print("\n=== Results ===")
print(band_amplitudes)
# %% amp by band
import numpy as np
import pandas as pd
def compute_amplitude_by_band(psd_dictionary, freq_bands=None):
    """Compute band amplitudes by integrating PSD using the actual frequency axis."""
    if freq_bands is None:
        freq_bands = {
            'swell': (0.0, 2.9999),
            'wind_waves': (3, 10.0),
            'total': (0.0, 10.0),
        }

    results = []
    for path, psd_df in psd_dictionary.items():
        row_out = {'path': path}
        freqs = psd_df.index.to_numpy(dtype=float)

        for i in range(1, 5):
            col = f'Pxx {i}'
            if col not in psd_df.columns:
                continue

            for band_name, (f_low, f_high) in freq_bands.items():
                mask = (freqs >= f_low) & (freqs <= f_high)
                if mask.sum() < 2:
                    amplitude = 0.0
                else:
                    f_band = freqs[mask]
                    psd_band = psd_df.loc[mask, col].to_numpy(dtype=float)
                    variance = np.trapezoid(psd_band, x=f_band)
                    amplitude = 2.0 * np.sqrt(variance)

                row_out[f'Probe {i} {band_name} amplitude'] = amplitude

        results.append(row_out)

    return pd.DataFrame(results)

# Example usage:
band_amplitudes = compute_amplitude_by_band(psd_dictionary)
print(band_amplitudes)

# %% band scatter
import matplotlib.pyplot as plt

def plot_p2_vs_p3_scatter(band_amplitudes):
    bands = ['swell', 'wind_waves', 'total']
    fig, axes = plt.subplots(1, len(bands), figsize=(12, 4), sharex=False, sharey=False)

    for ax, band in zip(axes, bands):
        p2 = band_amplitudes[f'Probe 2 {band} amplitude'].to_numpy()
        p3 = band_amplitudes[f'Probe 3 {band} amplitude'].to_numpy()
        ax.scatter(p2, p3, alpha=0.7)
        lim = max(p2.max(), p3.max()) * 1.05 if len(p2) else 1.0
        ax.plot([0, lim], [0, lim], 'k--', linewidth=1)  # y = x reference
        ax.set_title(f'{band}')
        ax.set_xlabel('P2 amplitude')
        ax.set_ylabel('P3 amplitude')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# Example:
plot_p2_vs_p3_scatter(band_amplitudes)

# %% band bars looop

import numpy as np
import matplotlib.pyplot as plt

def plot_p2_p3_bars(band_amplitudes):
    bands = ['swell', 'wind_waves', 'total']
    for _, row in band_amplitudes.iterrows():
        path = row['path']
        values_p2 = [row[f'Probe 2 {b} amplitude'] for b in bands]
        values_p3 = [row[f'Probe 3 {b} amplitude'] for b in bands]

        x = np.arange(len(bands))
        w = 0.35

        plt.figure(figsize=(8, 4))
        plt.bar(x - w/2, values_p2, width=w, label='P2')
        plt.bar(x + w/2, values_p3, width=w, label='P3')
        plt.xticks(x, bands)
        plt.ylabel('Amplitude')
        plt.title(path)
        plt.legend()
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()

# Example:
plot_p2_p3_bars(band_amplitudes)



# %% fft leik



"""FFT """
# Get only first half of the dictionary items
halfway = len(fft_dictionary) // 2
first_half_items = dict(list(fft_dictionary.items())[:halfway])

# Extract both columns
col1_data = {k: d.iloc[:, 1] for k, d in first_half_items.items()}
col2_data = {k: d.iloc[:, 2] for k, d in first_half_items.items()}

df_col1 = pd.concat(col1_data, axis=1)
df_col2 = pd.concat(col2_data, axis=1)

# A4 size in inches (portrait: 8.27 x 11.69, landscape: 11.69 x 8.27)
fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27), dpi=300)

# Plot column 1 in first facet
for name in df_col1.columns:
    short_name = str(name)[66:120]
    axes[0].plot(df_col1.index, df_col1[name], label=short_name, linewidth=1.5)
axes[0].set_xlabel("freq (Hz)")
axes[0].set_ylabel("Magnitude")
axes[0].set_title("P2")
axes[0].set_xlim(0, 10)
axes[0].grid(True, which="both", ls="--", alpha=0.3)

# Plot column 2 in second facet
for name in df_col2.columns:
    short_name = str(name)[66:120]
    axes[1].plot(df_col2.index, df_col2[name], label=short_name, linewidth=1.5)
axes[1].set_xlabel("freq (Hz)")
axes[1].set_ylabel("Magnitude")
axes[1].set_title("P3")
axes[1].set_xlim(0, 10)
axes[1].grid(True, which="both", ls="--", alpha=0.3)

# Make y-axis limits equal
y_min = min(axes[0].get_ylim()[0], axes[1].get_ylim()[0])
y_max = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
axes[0].set_ylim(y_min, y_max)
axes[1].set_ylim(y_min, y_max)

# Add shared legend below the plots
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.02), 
           ncol=4, frameon=False)

plt.tight_layout()

# Save as high-resolution image for direct printing
# plt.savefig('plot_A4.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
# %%


import numpy as np
from scipy.signal import welch, detrend

def dominant_wave(x, fs, band=(0.04, 2)):
    """
    x: signal (1D), fs: sampling rate (Hz), band: expected wave band in Hz
    returns: f0 (Hz), amplitude (peak), phase (rad)
    """
    # 1) Preprocess
    x = detrend(x, type='linear')

    # 2) Welch PSD (tune nperseg for resolution; overlap smooths)
    nperseg = int(8 * fs)  # ~8 s segments, adjust as needed
    f, Pxx = welch(x, fs=fs, window='hann', nperseg=nperseg, noverlap=nperseg//2, nfft=4*nperseg)

    # 3) Band-limit and find max
    m = (f >= band[0]) & (f <= band[1])
    if m.sum() < 3:
        return np.nan, 0.0, np.nan
    f_band, P_band = f[m], Pxx[m]
    k = np.argmax(P_band)

    # 4) Quadratic (parabolic) interpolation on log power for sub-bin frequency
    # Use three points: k-1, k, k+1 (guard edges)
    if 0 < k < len(P_band) - 1:
        y0, y1, y2 = np.log(P_band[k-1] + 1e-20), np.log(P_band[k] + 1e-20), np.log(P_band[k+1] + 1e-20)
        # Offset of peak from center bin (in bins)
        delta = 0.5 * (y0 - y2) / (y0 - 2*y1 + y2)
        df = f_band[1] - f_band[0]
        f0 = f_band[k] + delta * df
    else:
        f0 = f_band[k]

    # 5) Sinusoid least-squares fit at f0: x(t) ≈ a*sin(2πf0 t) + b*cos(2πf0 t)
    t = np.arange(len(x)) / fs
    s = np.sin(2*np.pi*f0*t)
    c = np.cos(2*np.pi*f0*t)
    A = np.column_stack([s, c])
    coef, _, _, _ = np.linalg.lstsq(A, x, rcond=None)
    a, b = coef
    amplitude_peak = np.hypot(a, b)            # peak amplitude
    phase = np.arctan2(b, a)                   # phase for cos-sin convention

    return float(f0), float(amplitude_peak), float(phase)

first_df = next(iter(processed_dfs.values()))
x = first_df["eta_2"]
x = x[6000:12000]
fs = 250
f0, amppeak, phase = dominant_wave(x, fs)

def fft_peak_amplitude(x, window=None):
    N = len(x)
    if window is None:
        window = np.ones(N)
    xw = x * window
    X = np.fft.rfft(xw)  # single-sided
    k = np.argmax(np.abs(X[1:])) + 1  # avoid DC
    # Coherent gain: amplitude reduction of the window for a bin-aligned tone
    CG = window.sum() / N
    # Single-sided: double non-DC/non-Nyquist bins
    A_peak = (2.0 * np.abs(X[k])) / (N * CG)
    f0_bin = k  # bin index; convert to Hz with fs*(k/N)
    return A_peak, k

a_peak, k = fft_peak_amplitude(x)




# %% damping


import seaborn as sns
import matplotlib as plt
xvar = "WaveFrequencyInput [Hz]"

df = damping_groupedruns_df.copy()
sns.set_theme(style='whitegrid')
ax = sns.scatterplot(
    data=df.sort_values(xvar),
    x=xvar,
    y='mean_P3P2',
    hue='WindCondition',
    style='PanelConditionGrouped',
    markers=True
)



# %%


wide = damping_pivot_wide
print(wide.columns.tolist())

mean_cols = [c for c in wide.columns if c.startswith("mean_P3P2_")]
wide_means = wide[["WaveAmplitudeInput [Volt]", "PanelConditionGrouped"] + mean_cols]

mask = (
    (wide["WaveAmplitudeInput [Volt]"] == 0.5)
    & (wide["PanelConditionGrouped"] == "all")
)
row = wide.loc[mask]

wide["delta_mean_P3P2_Windyyyy"] = (
    wide["mean_P3P2_lowest"] - wide["mean_P3P2_full"]
)
# %% Kult plot med errorbar

# stats has columns: WaveAmplitudeInput [Volt], PanelConditionGrouped, WindCondition, mean_P3P2, std_P3P2, ...
sns.lineplot(
    data=damping_groupedruns_df,
    x='WaveFrequencyInput [Hz]',
    y='mean_P3P2',
    hue='WindCondition',
    style='PanelConditionGrouped',
    marker='o',
    # errorbar=None  # we already have std; seaborn would otherwise estimate from raw data
)

# Add error bars manually using matplotlib if desired
for (pc, w), g in damping_groupedruns_df.groupby(['PanelConditionGrouped', 'WindCondition']):
    plt.errorbar(g['WaveAmplitudeInput [Volt]'], g['mean_P3P2'], yerr=g['std_P3P2'], fmt='none', alpha=0.3)
plt.show()

# %%

import numpy as np
df =damping_groupedruns_df
# Palette mapping to reuse the same color for lines and error bars
winds = df['WindCondition'].dropna().unique().tolist()
palette = sns.color_palette('tab10', n_colors=len(winds))
color_map = dict(zip(winds, palette))

g = sns.FacetGrid(
    data=df,
    col='PanelConditionGrouped',
    hue='WindCondition',
    palette=color_map,
    sharex=True,
    sharey=True,
    height=3.0,
    aspect=1.2,
    col_wrap=None  # set an int to wrap columns if you have many panels
)

# Draw mean lines with markers
g.map_dataframe(
    sns.lineplot,
    x='WaveAmplitudeInput [Volt]',
    y='mean_P3P2',
    marker='o',
    err_style=None  # disable seaborn’s internal error depiction
)

# Add std error bars manually for each hue in each facet
for ax, (panel_cond, sub_panel) in zip(g.axes.flat, df.groupby('PanelConditionGrouped', sort=False)):
    for wind, sub in sub_panel.groupby('WindCondition', sort=False):
        ax.errorbar(
            sub['WaveFrequencyInput [Hz]'],
            sub['mean_P3P2'],
            yerr=sub['std_P3P2'],
            fmt='none',
            capsize=3,
            color=color_map[wind],
            alpha=0.8
        )

g.add_legend(title='Wind')
g.set_axis_labels('WaveAmplitudeInput [Volt]', 'mean P3/P2')
g.set_titles(col_template='{col_name}')



# %% damping comb - under arbeid gpt plot

from wavescripts.filters import filter_dataframe
m_damping_filtrert = filter_dataframe(
    damping_groupedruns_df,
    amplitudeplotvariables, 
    ignore_missing_columns=True
)

from wavescripts.plotter import plot_damping_combined
plot_damping_combined(
    m_damping_filtrert,
    amplitudeplotvariables=amplitudeplotvariables
)



