#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wave Signal Processor

@author: ole
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

class WaveProcessor:
    def __init__(self, folder_path, output_dir, probe_cols=[1,2,3,4],
                 fixed_samples=None, start_cut=None, probe_ranges=None,
                 stillwater_samples=3000, window_size=50, signal_freq=1.3):
        """
        probe_ranges: dict {probe_idx: (start_cut, end_index)}
        """
        self.folder_path = Path(folder_path)
        self.output_dir = Path(output_dir)
        self.probe_cols = probe_cols
        self.fixed_samples = fixed_samples
        self.start_cut = start_cut
        self.stillwater_samples = stillwater_samples
        self.window_size = window_size
        self.signal_freq = signal_freq
        self.probe_ranges = probe_ranges or {}  # optional per-probe ranges

        # Internal storage
        self.csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv') and not f.endswith('stats.csv')]
        self.file_names = []
        self.cut_probe_arrays = []
        self.corrected_probes_mm = []
        self.avg_resting_mm = []
        self.offsets = []
        self.probes_ma = []
        self.time_axes_ma = []
        self.time_axes = []
        self.data_dict = {}

    # -------------------
    # Utility functions
    # -------------------
    @staticmethod
    def moving_average(x, w):
        if len(x) < 2 or w < 2:
            return x  # return as-is if too short
        return np.convolve(x, np.ones(w)/w, mode='valid')

    @staticmethod
    def compute_amplitudes(peaks, troughs, signal, time_axis, max_amplitudes=10):
        amplitudes = []
        time_pairs = []
        for i in range(min(len(peaks), len(troughs), max_amplitudes)):
            if i + 1 < len(troughs) and peaks[i] < troughs[i + 1]:
                if peaks[i] < len(signal) and troughs[i] < len(signal):
                    amp = abs(signal[peaks[i]] - signal[troughs[i]]) / 2
                    amplitudes.append(amp)
                    time_pairs.append((time_axis[peaks[i]], time_axis[troughs[i]]))
            elif peaks[i] < troughs[i]:
                if peaks[i] < len(signal) and troughs[i] < len(signal):
                    amp = abs(signal[peaks[i]] - signal[troughs[i]]) / 2
                    amplitudes.append(amp)
                    time_pairs.append((time_axis[peaks[i]], time_axis[troughs[i]]))
        return amplitudes[:max_amplitudes], time_pairs[:max_amplitudes]

    # -------------------
    # Loading and processing
    # -------------------
    def load_selected_files(self, selected_indices):
        selected_files = [self.csv_files[i] for i in selected_indices]
        probe_arrays = [[] for _ in range(len(self.probe_cols))]
        timestamps = []
        min_length = float('inf')

        for file in selected_files:
            df = pd.read_csv(self.folder_path / file, header=None)
            ts = pd.to_datetime(df[0])
            for i, col in enumerate(self.probe_cols):
                probe_arrays[i].append(np.array(df[col].to_numpy()) *1000)  #millimeter 
            timestamps.append((ts - ts.iloc[0]).dt.total_seconds() * 1000)
            self.file_names.append(file)
            min_length = min(min_length, len(df))

        if self.fixed_samples is not None and self.fixed_samples < min_length:
            min_length = self.fixed_samples

        # Truncate probes per probe_ranges
        self.cut_probe_arrays = []
        self.time_axes = []

        for i, probe_list in enumerate(probe_arrays):
            start, end = self.probe_ranges.get(i, (self.start_cut or 0, self.fixed_samples or min_length))
            truncated_probes = []
            truncated_times = []

            for arr, t in zip(probe_list, timestamps):
                s = min(start, len(arr))
                e = min(end, len(arr))
                if e <= s:
                    print(f"Skipping Probe {i+1}, File {len(truncated_probes)}: invalid range ({s}-{e})")
                    truncated_probes.append(np.array([]))
                    truncated_times.append(np.array([]))
                    continue
                truncated_probes.append(arr[s:e])
                truncated_times.append(t[s:e])
            
            self.cut_probe_arrays.append(truncated_probes)
            self.time_axes.append(truncated_times)

    def compute_resting_levels(self):
        rest_files = [f for f in self.csv_files if 'nowind' in f.lower() and 'baddata' not in f.lower()]
        if not rest_files:
            raise ValueError("No valid 'no wind' files found for resting level")
    
        self.resting_probes_all = []
        for f in rest_files:
            df = pd.read_csv(self.folder_path / f, header=None)
            probes = []
            for col in self.probe_cols:
                data = df[col].to_numpy()[:self.stillwater_samples]*1000 #millimeter
                if not data.size:
                    raise ValueError(f"No valid data in column {col} of file {f}")
                probes.append(data)
            self.resting_probes_all.append(probes)
    
        # Compute average resting levels (no *1000)
        self.avg_resting_mm = [
            np.nanmean(np.concatenate([self.resting_probes_all[f][i] for f in range(len(self.resting_probes_all))]))
            for i in range(len(self.probe_cols))
        ]
        print(f"Average resting levels (mm): {self.avg_resting_mm}")
    
        # Offsets to align to Probe 3
        ref_idx = 2  # Probe 3
        self.offsets = [self.avg_resting_mm[ref_idx] - self.avg_resting_mm[i] for i in range(len(self.probe_cols))]
        print(f"Offsets relative to Probe 3: {self.offsets}")
    
        # Correct probes (no *1000)
        self.corrected_probes_mm = [
            [np.array(arr + self.offsets[i]) if len(arr) > 0 else np.array([]) for arr in self.cut_probe_arrays[i]]
            for i in range(len(self.probe_cols))
        ]
        self.plot_resting_levels()


    def apply_moving_average(self):
        n_files = len(self.cut_probe_arrays[0])  # assuming at least 1 probe
        n_probes = len(self.probe_cols)

        self.probes_ma = []
        self.time_axes_ma = []

        for probe_idx in range(n_probes):
            probe_list_ma = []
            probe_time_list = []

            for file_idx in range(n_files):
                arr = self.corrected_probes_mm[probe_idx][file_idx]
                time_axis = self.time_axes[probe_idx][file_idx]

                if len(arr) < 2:
                    print(f"Skipping Probe {probe_idx+1}, File {file_idx}: too short ({len(arr)} samples)")
                    probe_list_ma.append(np.array([]))
                    probe_time_list.append(np.array([]))
                    continue

                win = min(self.window_size, len(arr))
                ma = self.moving_average(arr, win)
                ma_time = time_axis[win-1:win-1+len(ma)]

                probe_list_ma.append(ma)
                probe_time_list.append(ma_time)
                
            self.probes_ma.append(probe_list_ma)
            self.time_axes_ma.append(probe_time_list)

    # -------------------
    # Plot amplitudes
    # -------------------
    def plot_amplitude_comparison(self, probe_idx, file_idx, output_dir=None, max_peaks=10):
        signal = np.array(self.probes_ma[probe_idx][file_idx])
        time_axis = np.array(self.time_axes_ma[probe_idx][file_idx])
        stillwater = self.avg_resting_mm[probe_idx]
    
        #print(f"Probe {probe_idx+1}, File {file_idx}: stillwater = {stillwater} mm")  # Debug
    
        if len(signal) < 2:
            print(f"Cannot plot Probe {probe_idx+1}, File {file_idx}: signal too short")
            return np.nan
    
        # Shift signal to center around 0 relative to stillwater
        signal_shifted = signal - stillwater
    
        dt_ms = np.mean(np.diff(time_axis)) if len(time_axis) > 1 else 4.0
        fs = 1000 / dt_ms
        min_distance = int(fs / (2 * self.signal_freq))
    
        peaks, _ = find_peaks(signal_shifted, distance=min_distance, prominence=0.5)
        troughs, _ = find_peaks(-signal_shifted, distance=min_distance, prominence=0.5)
        amps, time_pairs = self.compute_amplitudes(peaks, troughs, signal_shifted, time_axis, max_amplitudes=max_peaks)
        avg_amp = np.mean(amps) if amps else np.nan
    
        # Save results
        key = f'amplitudes_probe{probe_idx+1}_file{file_idx}'
        self.data_dict[key] = {
            'peaks_indices': peaks,
            'troughs_indices': troughs,
            'amplitudes': amps,
            'time_pairs': time_pairs,
            'average_amplitude': avg_amp
        }
    
        # Plot
        plt.figure(figsize=(10, 4))
        plt.plot(time_axis, signal_shifted, label='Smoothed Signal')
        if len(peaks) > 0:
            plt.plot(time_axis[peaks], signal_shifted[peaks], 'ro', label='Peaks')
        if len(troughs) > 0:
            plt.plot(time_axis[troughs], signal_shifted[troughs], 'go', label='Troughs')
        plt.axhline(0, color='red', linestyle='--', label='Stillwater Level')
        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude (mm relative to stillwater)")
        plt.title(f"Probe {probe_idx+1}, {self.file_names[file_idx]} Amplitude: {avg_amp:.2f} mm")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            plt.savefig(Path(output_dir) / f'probe{probe_idx+1}_file{file_idx}_amplitude.png')
        plt.show(block=False)
        return avg_amp
   
    
    def plot_resting_levels(self):
        if not self.resting_probes_all:
            raise ValueError("self.resting_probes_all is empty or not set")
        
        plt.figure(figsize=(10, 6))
        for probe_idx in range(len(self.probe_cols)):
            all_samples = [self.resting_probes_all[f][probe_idx] for f in range(len(self.resting_probes_all))]
            concatenated = np.concatenate(all_samples)  # Remove *1000 if data is in mm
            plt.plot(concatenated, label=f"Probe {probe_idx+1}")
        plt.xlabel("Sample index")
        plt.ylabel("Raw amplitude [mm]")
        plt.title("Resting level samples from 'no wind' runs")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    