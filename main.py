#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main runner for wave signal processing
"""

from scripts.wave_processor import WaveProcessor
from pathlib import Path
import numpy as np

# ---------------- Paths ----------------
project_root = Path(__file__).parent
data_folder = project_root / "wavedata"
results_folder = project_root / "waveresults"

# Ensure results folder exists
results_folder.mkdir(exist_ok=True)

# ---------------- Create processor ----------------
probe_window = 2675 # 14per/1.3Hz = 10.7s . 10.7s/250 samples/sec. = 2675. 
probe2start = 4500 #fra 18K til 28K ms
probe3start = 6000
probe_ranges = {
    1: (probe2start, probe2start+probe_window),  # Probe 2
    2: (probe3start, probe3start+probe_window),  # Probe 3
}

processor = WaveProcessor(
    folder_path=data_folder,
    output_dir=results_folder,
    probe_ranges=probe_ranges,
    stillwater_samples=2000,
    window_size=50
)

# ---------------- Select all CSV files ----------------
selected_indices = list(range(len(processor.csv_files)))  # automatically all CSVs
selected_indices = [0]
processor.load_selected_files(selected_indices)

# ---------------- Compute resting levels and offsets ----------------
processor.compute_resting_levels()

# ---------------- Apply moving average ----------------
processor.apply_moving_average()

# ---------------- Plot amplitudes for Probe 2 and Probe 3 ----------------
avg_amps_probe2 = []
avg_amps_probe3 = []

for i in range(len(selected_indices)):
    avg_amps_probe2.append(processor.plot_amplitude_comparison(probe_idx=1, file_idx=i, output_dir=results_folder))
    avg_amps_probe3.append(processor.plot_amplitude_comparison(probe_idx=2, file_idx=i, output_dir=results_folder))

# ---------------- Compare Probe 2 vs Probe 3 ----------------
print("\nProbe 2 vs Probe 3 amplitude comparison:")
for i, fname in enumerate(processor.file_names):
    p2 = avg_amps_probe2[i]
    p3 = avg_amps_probe3[i]
    if not np.isnan(p2) and not np.isnan(p3):
        diff = p2 - p3
        ratio = p2/p3 if p3 != 0 else np.nan
        print(f"{fname}: Probe2={p2:.2f} mm, Probe3={p3:.2f} mm, Diff={diff:.2f}, Ratio={ratio:.2f}")
    else:
        print(f"{fname}: insufficient amplitude data")

# Example: plot stillwater for first file 
processor.plot_resting_levels(resting_probes_all=0)



"""
# Create batch object
batch = SignalBatch(data_folder)

# -------------------
# Processing pipeline
# -------------------
fs = 100       # sampling frequency
lowcut = 1.0   # Hz
highcut = 3.0  # Hz

# Apply processing
batch.apply_filter(lowcut, highcut, fs)
batch.normalize_all()

# Save processed signals
for sig in batch.signals:
    sig.save_csv(results_folder)

# Plot individually
batch.plot_all(overlay=False)

# Plot overlayed
batch.plot_all(overlay=True)

# Compute FFT for all signals and save plots
fft_results = batch.compute_fft_all()
import matplotlib.pyplot as plt
for name, (freqs, mag) in fft_results.items():
    plt.figure(figsize=(8,4))
    plt.plot(freqs, mag)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.title(f"FFT: {name}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, f"fft_{name}.png"))
    plt.close()
"""


