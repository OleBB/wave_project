#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 08:41:03 2025
@author: ole
"""


import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from wavescripts.improved_data_loader import load_or_update
from wavescripts.filters import filter_chosen_files
from wavescripts.processor import process_selected_data
from wavescripts.processor2nd import process_processed_data 

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
    # Path("/Users/ole/Kodevik/wave_project/wavedata/20251112-tett6roof-lowM-579komma8"),
    # Path("/Users/ole/Kodevik/wave_project/wavedata/20251113-tett6roof"),
    # Path("/Users/ole/Kodevik/wave_project/wavedata/20251113-tett6roof-loosepaneltaped"),
    
    # Path("/Users/ole/Kodevik/wave_project/wavedata/20251113-tett6roof-probeadjusted"),
    
]
#%% kjør
# Initialize containers for all results
all_meta_sel = []
all_processed_dfs = []

processvariables = {
    "overordnet": {
        "chooseAll": False,
        "chooseFirst": True,
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
        "total_reset": False, #laster også csv'ene på nytt
        "force_recompute": True, #kjører alt på nytt, ignorerer gammal json
        "debug": True,
        "smoothing_window": 10, #kontrollere denne senere
        "find_range": True,
        "range_plot": False,    
    },
}
#todo: fikse slik at jeg kan plotte range, eller kjøre ting på nytt, uten å 
#   reloade csv.'ene. det trengs vel bare 1 gang.
#todo: bli enig om hva som er forskjellen på force recompute og full resett (tror dei e like no)? 
# Loop through each dataset
for i, data_path in enumerate(dataset_paths):
    print(f"\n{'='*50}")
    print(f"Processing dataset {i+1}/{len(dataset_paths)}: {data_path.name}")
    print(f"{'='*50}")
    try:
        prosessering = processvariables.get("prosessering", {})
        total_reset =prosessering.get("total_reset", False)
        if total_reset:
            input("TOTAL RESET! press enter if you want to continue")
        force =prosessering.get("force_recompute", False)
        dfs, meta = load_or_update(data_path, force_recompute=force, total_reset=total_reset)
        
        print('# === Filter === #') #dette filteret er egentlig litt unøding, når jeg ønsker å prossesere hele sulamitten
        meta_sel = filter_chosen_files(meta, processvariables)
        
        print('# === Single probe process === #')
        processed_dfs, meta_sel, psd_dictionary, fft_dictionary = process_selected_data(dfs, meta_sel, meta, processvariables)
        
        # print('# === FTT on each separate signal, saved to a dict of dfs')

        
        print('# === Probe comparison processing === #')
        meta_sel = process_processed_data(psd_dictionary, fft_dictionary, meta_sel, meta, processvariables)
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

# %%

#TODO: rydde i maintester. kopierte det over fra main.

# %%

#dagens mål: implementere likningen fra John. 
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

first_df = next(iter(processed_dfs.values()))
time_series_full = first_df[["Date", "eta_2"]]

start = meta_sel.iloc[0]["Computed Probe 2 start"]
end = meta_sel.iloc[0]["Computed Probe 2 end"]

time_series = time_series_full.iloc[int(start):int(end)]
# time_series = time_series_full

dt = 0.004
signal = time_series["eta_2"].values
n_samples = len(signal)
time = np.arange(n_samples)*dt

time_series.iloc[:,1].plot()
# %%
ctotal = 0
number_of_frequencies = 300
frequencies = np.linspace(0.04,60,number_of_frequencies)
# frequencies = np.linspace(1.28,1.32,number_of_frequencies)
# frequencies = np.array(1,)
signal = np.asarray(signal)          # shape (N,)
time  = np.asarray(time)           # shape (N,)
N  = 970

frequencies = np.asarray(frequencies)  # shape (F,)
fourier_coeffs = np.zeros(len(frequencies), dtype=complex)

# %% nested for loop
""" NESTED FOR LOOP """
kof = np.zeros(len(frequencies), dtype=complex)
c   = np.zeros(N, dtype=complex)
for i, f in enumerate(frequencies):
    w = 2 * np.pi * f
    ctotal = 0.0 + 0.0j
    for n in range(N):  # med n=0
        c[n] = signal[n] * np.exp(-1j * w * time[n])
        if n == 2:
            print(c)
        ctotal += c[n]
    kof[i] = (ctotal * dt) / N
# %%
plt.plot(frequencies, np.abs(kof), '-')
plt.xlabel('Frequency')
plt.xlim(0,10)
plt.ylabel(' - ')
plt.grid(True)
plt.show()
# %% vektorisert indre loop
""" Vektorisert indre loop"""
for i, freq in enumerate(frequencies):
    omega = 2 * np.pi * freq
    
    # Compute Fourier coefficient using trapezoidal integration
    integrand = signal * np.exp(-1j * omega * time)
    fourier_coeffs[i] = np.sum(integrand) * dt / n_samples
    
    if i == 0:  # Debug: print first few values
        print(f"Frequency {freq:.3f} Hz: {integrand[:3]}")

plt.plot(frequencies, np.abs(fourier_coeffs), '-', color='green')
plt.xlabel('Frequency')
plt.xlim(0,10)
plt.ylabel(' - ')
plt.grid(True)
plt.show()
# %%
"""Vectorisert ytre loop også"""
window = np.hanning(n_samples)
h_signal = signal * window
omega = 2 * np.pi * frequencies[:, np.newaxis] #reshape til (200, 1)
f_coeffs = np.sum(h_signal * np.exp(-1j * omega * time), axis=1)*dt#/n_samples

# plt.plot(frequencies, np.abs(f_coeffs), '-', color='red')
# plt.xlabel('Frequency')
# plt.xlim(0,10)
# plt.ylabel(' - ')
# plt.grid(True)
# plt.show()
# %%
"""FFT"""
fft_vals = np.fft.fft(h_signal)
fft_freqs = np.fft.fftfreq(len(h_signal), d=1/250)

positive_freq_idx = fft_freqs >= 0
fft_freqs_pos = fft_freqs[positive_freq_idx]
fft_magnitude = np.abs(fft_vals[positive_freq_idx])

# plt.plot(fft_freqs_pos, np.abs(fft_magnitude), '-.', color='red')
# plt.xlabel('Frequency')
# plt.xlim(0,10)
# plt.ylabel(' - ')
# plt.grid(True)
# plt.show()
# %%
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# Manual method
axes[0].stem(frequencies, np.abs(f_coeffs), '-o')
axes[0].plot(frequencies, np.abs(f_coeffs), '-')
axes[0].set_xlabel('Frequency (Hz)')
axes[0].set_xlim(0,10)
axes[0].set_ylabel('Amplitude')
axes[0].set_title('Manual Fourier Transform')
axes[0].grid(True)

# FFT method
axes[1].stem(fft_freqs_pos, np.abs(fft_magnitude),'-o')
axes[1].plot(fft_freqs_pos, np.abs(fft_magnitude),'-')
axes[1].set_xlabel('Frequency (Hz)')
axes[1].set_xlim(0,10)
axes[1].set_ylabel('Amplitude')
axes[1].set_title('NumPy FFT')
# axes[1].set_xlim(frequencies[0], frequencies[-1])  # Same range as manual
axes[1].grid(True)

plt.tight_layout()
plt.show()
# %%


# Original signal (970 samples)
signal_original = h_signal.copy()
n_original = len(signal_original)

# Zero-padded signal (4x longer)
n_padded = 4 * n_original
signal_padded = np.pad(signal_original, (0, n_padded - n_original), mode='constant')

# Compute FFT for both
fft_original = np.fft.fft(signal_original)
fft_padded = np.fft.fft(signal_padded)

freqs_original = np.fft.fftfreq(n_original, dt)
freqs_padded = np.fft.fftfreq(n_padded, dt)

# Plot comparison
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Original (coarse)
mask_orig = (freqs_original >= 0) & (freqs_original <= 10)
axes[0].stem(freqs_original[mask_orig], np.abs(fft_original[mask_orig]), basefmt=' ')
axes[0].plot(freqs_original[mask_orig], np.abs(fft_original[mask_orig]), 'r-', linewidth=2)
axes[0].set_title(f'Original FFT ({n_original} points) - Coarser frequency bins')
axes[0].set_xlabel('Frequency (Hz)')
axes[0].set_ylabel('Amplitude')
axes[0].grid(True, alpha=0.3)

# Zero-padded (smooth)
mask_pad = (freqs_padded >= 0) & (freqs_padded <= 10)
axes[1].stem(freqs_padded[mask_pad], np.abs(fft_padded[mask_pad]), basefmt=' ')
axes[1].plot(freqs_padded[mask_pad], np.abs(fft_padded[mask_pad]), 'r-', linewidth=2)
axes[1].set_title(f'Zero-Padded FFT ({n_padded} points) - Smoother, same resolution')
axes[1].set_xlabel('Frequency (Hz)')
axes[1].set_ylabel('Amplitude')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
# %%
# If you know it's ~1.30 Hz, fit a sinusoid at that frequency
target_freq = 1.3

# Test frequencies around target
test_freqs = np.linspace(1.25, 1.35, 1000)

# For each frequency, compute how well a sinusoid fits
def compute_fit_quality(freq, h_signal, time):
    omega = 2 * np.pi * freq
    # Project signal onto cosine and sine at this frequency
    cos_component = np.sum(h_signal * np.cos(omega * time))
    sin_component = np.sum(h_signal * np.sin(omega * time))
    # Amplitude of best-fit sinusoid
    amplitude = np.sqrt(cos_component**2 + sin_component**2)
    return amplitude

fit_quality = np.array([compute_fit_quality(f, h_signal, time) for f in test_freqs])

# Find best frequency
best_idx = np.argmax(fit_quality)
best_frequency = test_freqs[best_idx]

print(f"Best fit frequency: {best_frequency:.6f} Hz")

# Plot
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Fit quality vs frequency
axes[0].plot(test_freqs, fit_quality, linewidth=2)
axes[0].axvline(best_frequency, color='red', linestyle='--', 
                label=f'Best: {best_frequency:.4f} Hz')
axes[0].set_xlabel('Frequency (Hz)')
axes[0].set_ylabel('Fit Quality')
axes[0].set_title('Frequency Fit Quality')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Reconstruct signal at best frequency
omega_best = 2 * np.pi * best_frequency
cos_comp = np.sum(signal * np.cos(omega_best * time))
sin_comp = np.sum(signal * np.sin(omega_best * time))
amplitude = 2 * np.sqrt(cos_comp**2 + sin_comp**2) / n_samples
phase = np.arctan2(sin_comp, cos_comp)
fitted_signal = amplitude * np.cos(omega_best * time - phase)

axes[1].plot(time, signal, 'b-', alpha=0.5, label='Original signal')
axes[1].plot(time, fitted_signal, 'r-', linewidth=2, 
             label=f'Fitted {best_frequency:.4f} Hz sinusoid')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Amplitude')
axes[1].set_title('Signal vs Best Fit')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(0, min(3.4, time[-1]))  # Show first 2 seconds

plt.tight_layout()
plt.show()


# %%

plt.plot(time,h_signal, 'rx', label='hanning')
plt.plot(time, signal, 'b-', alpha=0.5, label='Original signal')

# %%


kof = f_coeffs
import numpy as np
import matplotlib.pyplot as plt

# Assuming you already have:
# frequencies = np.array([1.3, 2.6, 5.2])  # in Hz (or rad/s)
# kof = ...  # shape (len(frequencies),), likely complex

plt.figure(figsize=(8, 5))

# Magnitude
plt.subplot(2, 2, 1)
plt.plot(frequencies, np.abs(kof), marker='o')
plt.title('Magnitude |kof|')
plt.xlabel('Frequency')
plt.ylabel('|kof|')
plt.grid(True)

# Real part
plt.subplot(2, 2, 2)
plt.plot(frequencies, np.real(kof), marker='o', color='tab:blue')
plt.title('Real(kof)')
plt.xlabel('Frequency')
plt.ylabel('Real')
plt.grid(True)

# Imag part
plt.subplot(2, 2, 3)
plt.plot(frequencies, np.imag(kof), marker='o', color='tab:orange')
plt.title('Imag(kof)')
plt.xlabel('Frequency')
plt.ylabel('Imag')
plt.grid(True)

# Phase
plt.subplot(2, 2, 4)
phase = np.angle(kof)
plt.plot(frequencies, phase, marker='o', color='tab:green')
plt.title('Phase(kof)')
plt.xlabel('Frequency')
plt.ylabel('Phase (rad)')
plt.grid(True)

plt.tight_layout()
plt.show()



# %%





import matplotlib.ticker as mticker

first_df = next(iter(psd_dictionary.values()))
# python
ax = first_df[["Pxx 1", "Pxx 2", "Pxx 3", "Pxx 4"]].plot()
ax.set_xlim(0, 6)
ax.set_ylim(1e-6, 40)
ax.minorticks_on()
ax.xaxis.set_major_locator(mticker.MultipleLocator(0.5))   # major every 0.5 (adjust)

ax.grid(True, which="major")
#ax.grid(True, which="minor", linestyle="--")

# python
# ax.set_xscale("log")
# ax.set_yscale("log")   # or "symlog" if values span zero

# %%
#
# python
import pandas as pd
import matplotlib.pyplot as plt

# df_plot has columns from your psd_dictionary (as in your example)
first_cols = {k: d.iloc[:, 0] for k, d in psd_dictionary.items()}
df_plot = pd.concat(first_cols, axis=1)

fig, ax = plt.subplots(figsize=(7, 4))

# Iterate columns for full control
for name in df_plot.columns:
    ax.plot(df_plot.index, df_plot[name], label=str(name), linewidth=1.5, marker=None)

ax.set_xlabel("freq (Hz)")
# ax.set_ylabel("PSD")
ax.set_xlim(0, 10)
ax.grid(True, which="both", ls="--", alpha=0.3)
#ax.legend(title="Series", ncol=2)  # or remove if not needed
plt.tight_layout()
plt.show()

# %% FFT







#%% TODO: correlate

np.correlate()



#%% - 
from wavescripts.wavestudyer import compare_probe_amplitudes_and_lag, amplitude_overview, full_tank_diagnostics, wind_damping_analysis 
summary_df = wind_damping_analysis(meta_sel)

# %% 
"""skriver et sammendrag av dempningen i """
from wavescripts.wavestudyer import wind_damping_analysis
damping_analysis_results = wind_damping_analysis(meta_sel)


#%%
from wavescripts.wavestudyer import damping
oppgradert_meta_sel = damping(meta_sel)




#%% - Her plotter man en enkeltkjøring oppå en annen


chooseAll = True

amplitudeplotvariables = {
    "filters": {
        "amp": 0.1, #0.1, 0.2, 0.3 
        "freq": 1.3, #bruk et tall  
        "per": None, #bruk et tall #brukes foreløpig kun til find_wave_range, ennå ikke knyttet til filtrering
        "wind": ["no", "lowest", "full"], #full, no, lowest, all
        "tunnel": None,
        "mooring": "low",
        "panel": ["full", "no"], # no, full, reverse, 
        
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

from wavescripts.filters import filter_for_amplitude_plot
m_filtrert = filter_for_amplitude_plot(oppgradert_meta_sel, amplitudeplotvariables, chooseAll)

"""Plot amplitude summary plotter alt den tar inn"""
from wavescripts.plotter import plot_all_probes
plot_all_probes(m_filtrert, amplitudeplotvariables)




#%% lage diagnose senere
summary = full_tank_diagnostics(processed_dfs)
#TODO få resultater fra disse over i run_and_save_report.py
#%%
overview = amplitude_overview(processed_dfs, window_ms)
#to måter å få ut første verdien fra dictionary
#next(iter(processed_dfs.values())) #uten .values får man key, eller så kan man skrive .keys, og .item gir tuple med key og value.
# Pick any file
df = list(processed_dfs.values())[0]
print('name:' ,df.head())

print("Raw amplitudes (before any fix):")
for i in range(1,5):
    col = f"eta_{i}"
    if col in df.columns:
        amp = (df[col].quantile(0.99) - df[col].quantile(0.01)) / 2
        print(f"  Probe {i}: {amp:.1f} mm  →  {'PROBABLY BAD' if amp > 50 else 'OK'}")
#%%
col1 = "eta_2"
col2 = "eta_3"
start_ms = 6000
end_ms = 7000
result = compare_probe_amplitudes_and_lag(df,col1, col2, start_ms, end_ms)

res = compare_probe_amplitudes_and_lag(df, col1, col2, start_ms, end_ms)

window = df.loc[start_ms:end_ms]
t = (window.index - window.index[0]) #/ 1000  #seconds

import matplotlib.pyplot as plt
plt.figure(figsize=(10,4))
plt.plot(t, window["eta_2"], label="Probe 2", alpha=0.8)
plt.plot(t - res["lag_ms"], window["eta_3"], label=f"Probe 3 (shifted -{res['lag_ms']:.0f}ms)", alpha=0.8)
plt.legend()
plt.xlabel("milliseconds")
plt.ylabel("Elevation [mm]")
plt.title("Perfect alignment after time-shift correction")
plt.grid(alpha=0.3)
plt.show()

#%% - Create a "report"  - but currently its really just a table with df data. 
from wavescripts.create_report import markdown_report_from_df
markdown_report_from_df(df,
                            title="Wave Tank Analysis Report",
                            subtitle="a preliminary draft",
                            out_path="report.md",
                            plots_folder="reportplots",
                            save_plots=True,
                            max_rows=50)

#%%

# Step 1: Load raw data + basic metadata
from wavescripts.data_loader import load_or_, _processed_metadata
dfs, meta = load_or_("wavedata/20251110-tett6roof-lowM-ekte580")
import pandas as pd
# Step 2: YOUR ANALYSIS — you modify meta and/or dfs as much as you want
for key, df in dfs.items():
    path = Path(key)
    row = meta[meta["path"] == key].iloc[0]

    # Example: compute zeroed waves and significant height
    stillwater = df["Probe 1"].iloc[:250].mean()
    eta = df["Probe 1"] - stillwater

    #  the metadata row (in-place)
    meta.loc[meta["path"] == key, "Computed Probe 1 start"] = float(stillwater)
    meta.loc[meta["path"] == key, "Hs"] = float(4 * eta.std())
    meta.loc[meta["path"] == key, "T_z"] = float(0.71)  # or proper zero-crossing
    meta.loc[meta["path"] == key, "Processed at"] = pd.Timestamp("now")

    # Optionally save d DataFrame back (with eta, filtered, etc.)
    df["eta_1"] = eta
    dfs[key] = df  # will be saved next time you call load_or_ or manually
_processed_metadata(meta)


#%%

# processing.py — clean and professional
from wavescripts.data_loader import load_or_, _processed_metadata
from wavescripts.processor import process_selected_data


from wavescripts.data_loader import load_meta_from_processed
meta = load_meta_from_processed("PROCESSED-20251110-tett6roof-lowM-ekte580")


#%%
"""... KANSKJE DISPLAYE JSON ENTRYEN SÅ
#JEG VET HVA SOM BLE KJØRT. ELLER DISPLAY MOORING SOM TILLEGGSINFO I PLOTT?
# altså, nedenfor printes mappe og fil... tanken var å fange opp  
#dersom jeg kjører ulik mooring- og tunnelcondition
#sånn den er nå så kjører den bare dersom alle parameterrne stemmer"""


