#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 10:21:49 2025

@author: ole
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from wavescripts.data_loader import load_or_update
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
    
    Path("/Users/ole/Kodevik/wave_project/wavedata/20251110-tett6roof-lowMooring"),
    Path("/Users/ole/Kodevik/wave_project/wavedata/20251110-tett6roof-lowMooring-2"),
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
        "debug": True,
        "smoothing_window": 10, #kontrollere denne senere
        "find_range": True,
        "range_plot": False,    
        "force_recompute": True,
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

#%% grouper - slår i hop
from wavescripts.filters import damping_grouper
damping_groupedruns_df, damping_pivot_wide = damping_grouper(combined_meta_sel)
# %% damping variables initiert
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

# %% plotting damping frequencies seaborn
from wavescripts.plotter import facet_plot_freq_vs_mean
facet_plot_freq_vs_mean(damping_filtrert, dampingplotvariables)

# %% plotting damping amplitudes seaborn 
from wavescripts.plotter import facet_plot_amp_vs_mean
facet_plot_amp_vs_mean(damping_filtrert, dampingplotvariables)

# %% slår alle i hop
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


from wavescripts.filters import damping_all_amplitude_grouper
damping_groupedallruns_df  = damping_all_amplitude_grouper(combined_meta_sel)

# damping_all_amplitudes_filtrert = filter_for_damping(damping_groupedallruns_df, dampingplotvariables["filters"])
# %% plotter damping_groupedallruns
from wavescripts.plotter import facet_amp
facet_amp(damping_groupedallruns_df, dampingplotvariables)

# %% FFT-SPEKTRUM filter initiert
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

from wavescripts.filters import filter_for_frequencyspectrum
filtrert_frequencies = filter_for_frequencyspectrum(meta_sel, freqplotvariables)


# %% plotter fft facet
from wavescripts.plotter import plot_frequency_spectrum
fig, axes = plot_frequency_spectrum(
    fft_dictionary,
    filtrert_frequencies,
    freqplotvariables,
    data_type="fft"
)
# %% plotter PSD facet
fig, axes = plot_frequency_spectrum(
    psd_dictionary,  # Your PSD data dictionary
    filtrert_frequencies, 
    freqplotvariables,
    data_type="psd"
)


# %% amplitude by BAND

import pandas as pd
from typing import Dict, Tuple, Optional, Mapping, Iterable

def compute_amplitude_by_band(
    psd_dict: Mapping[str, pd.DataFrame],
    *,
    freq_bands: Optional[Dict[str, Tuple[float, float]]] = None,
    probes: Iterable[int] = (1, 2, 3, 4),
    verbose: bool = False,
    integration: str = "sum",          # "sum"  → simple Δf * Σ PSD
                                        # "trapez" → np.trapezoid on the real freq axis
    freq_resolution: Optional[float] = None,
) -> pd.DataFrame:
    """
    Compute wave‑amplitude estimates for a set of frequency bands from PSD data.

    Parameters
    ----------
    psd_dict : mapping of ``path → pd.DataFrame``
        Each DataFrame must be indexed by frequency (Hz) and contain columns
        named ``'Pxx 1'``, ``'Pxx 2'``, … for the different probes.
    freq_bands : dict, optional
        Mapping ``band_name → (f_low, f_high)`` in Hz.  If omitted the
        classic three‑band set is used:

        .. code-block:: python

            {
                "swell":      (1.0, 1.6),
                "wind_waves": (3.0, 10.0),
                "total":      (0.0, 10.0),
            }

    probes : iterable of int, default (1,2,3,4)
        Which probe columns (``'Pxx i'``) to process.
    verbose : bool, default ``False``
        Print a short diagnostic for each file / band (mirrors the second
        version you posted).
    integration : {"sum", "trapez"}, default ``"sum"``
        * ``"sum"`` – assumes a *uniform* frequency spacing and computes the
          variance as ``Δf * Σ PSD``.  This is the fastest option and matches
          the first two snippets.
        * ``"trapez"`` – uses ``np.trapezoid`` on the *actual* frequency axis,
          which is more accurate when the spacing is irregular (third snippet).
    freq_resolution : float, optional
        Explicit frequency resolution (Δf).  If ``None`` and ``integration=="sum"``,
        the function derives Δf from the first two frequency points of each
        DataFrame (the original behaviour).

    Returns
    -------
    pd.DataFrame
        One row per ``path`` with columns

        ``'Probe {i} {band_name} amplitude'``

        containing the peak‑to‑trough amplitude estimate
        $A = 2\sqrt{\mathrm{variance}}$.
    """

    # ----------------------------------------------------------------------
    #  Default frequency‑band definitions (kept from the first two versions)
    # ----------------------------------------------------------------------
    if freq_bands is None:
        freq_bands = {
            "swell":      (0.0, 2.6),
            "wind_waves": (2.60000001, 16.0),
            "total":      (0.0, 16.0),
        }

    # ----------------------------------------------------------------------
    #  Validate the chosen integration method
    # ----------------------------------------------------------------------
    if integration not in {"sum", "trapez"}:
        raise ValueError("integration must be either 'sum' or 'trapz'")

    # ----------------------------------------------------------------------
    #  Main loop over all PSD files (paths)
    # ----------------------------------------------------------------------
    rows = []
    for path, df in psd_dict.items():
        # Store results for this path
        row = {"path": path}

        if verbose:
            print(f"\n=== Path: {path} ===")
            print(f"  Frequency range: {df.index.min():.3f}–{df.index.max():.3f} Hz")
            if integration == "sum":
                # Δf will be derived later; show a placeholder now
                print("  Integration method: sum (Δf * Σ PSD)")

        # ------------------------------------------------------------------
        #1 Determine frequency resolution if needed (only for "sum")
        # ------------------------------------------------------------------
        if integration == "sum":
            # Assume uniform spacing – take the difference of the first two points.
            # If the user supplied an explicit value, honour it.
            if freq_resolution is None:
                # Guard against a single‑point index (unlikely for a PSD)
                if len(df.index) < 2:
                    raise ValueError(f"Not enough frequency points in {path} to infer Δf")
                freq_res = float(df.index[1] - df.index[0])
            else:
                freq_res = float(freq_resolution)

            if verbose:
                print(f"  Frequency resolution Δf: {freq_res:.6f} Hz")

        # ------------------------------------------------------------------
        # Loop over the requested probes
        # ------------------------------------------------------------------
        for i in probes:
            col = f"Pxx {i}"
            if col not in df.columns:
                if verbose:
                    print(f"  Probe {i}: column '{col}' missing → skipped")
                continue

            # ----------------------------------------------------------------
            # Loop over the frequency bands
            # ----------------------------------------------------------------
            for band_name, (f_low, f_high) in freq_bands.items():
                # Build a mask for the current band
                mask = (df.index >= f_low) & (df.index <= f_high)
                n_points = int(mask.sum())

                # ------------------------------------------------------------
                #  Empty‑band handling (kept from version 2)
                # ------------------------------------------------------------
                if n_points == 0:
                    amplitude = 0.0
                    if verbose:
                        print(
                            f"  Probe {i} – {band_name}: "
                            f"NO DATA POINTS in [{f_low}, {f_high}] Hz → amplitude=0"
                        )
                    row[f"Probe {i} {band_name} amplitude"] = amplitude
                    continue

                # ------------------------------------------------------------
                #  Compute variance → amplitude
                # ------------------------------------------------------------
                if integration == "sum":
                    # Simple rectangular integration: Δf * Σ PSD
                    variance = df.loc[mask, col].sum() * freq_res
                else:  # "trapz"
                    # Use the true frequency axis for trapezoidal integration.
                    freqs = df.index.to_numpy(dtype=float)[mask]
                    psd_vals = df.loc[mask, col].to_numpy(dtype=float)
                    variance = np.trapezoid(psd_vals, x=freqs)

                # Standard deviation and peak‑to‑trough amplitude
                std_dev = np.sqrt(variance)
                amplitude = 2.0 * std_dev

                if verbose and i == probes[0]:  # print once per band, first probe
                    print(
                        f"  Probe {i} – {band_name} [{f_low}-{f_high}] Hz: "
                        f"{n_points} points, variance={variance:.6e}, amplitude={amplitude:.4f}"
                    )

                row[f"Probe {i} {band_name} amplitude"] = amplitude

        rows.append(row)

    # ----------------------------------------------------------------------
    #  Convert list‑of‑dicts → DataFrame (preserves column order)
    # ----------------------------------------------------------------------
    return pd.DataFrame(rows)


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
        plt.title(path[40:])
        plt.legend()
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()

# Example:
plot_p2_p3_bars(band_amplitudes)




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



# %% damping wide


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



# %% __main__


# if __name__ == "__main__":
    # print('running main')
