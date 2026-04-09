#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
"""
Created on Thu Nov  6 20:51:51 2025

@author: Ole
"""
# %%
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# --- USER SETTINGS ---
# Root directory containing your folders
root_dir = r"/Users/ole/Kodevik/wave_project/pressuredata/"

# Custom legend labels — edit these to your liking.
# Key = folder name, Value = label shown in legend.
# Any folder not listed here will use its folder name as label.
LEGEND_LABELS = {
    "20251106-fullwindUtenProbe2-fullpanel-amp0100-freq1300":
        "Full vind, uten probe foran, med panel, med genererte bølger (06.11)",

    "20251106-fullwindUtenProbe2-fullpanel":
        "Full vind, uten probe foran, med panel, uten bølger (06.11)",

    "20251105-fullwindUtenProbe2-fullpanel":
        "Full vind, med probe foran, med panel, uten bølger (05.11)",

    "20251104-fullwind":
        "Full vind, med probe foran, uten panel, uten bølger (04.11)",

    "20251106-lowestwindUtenProbe2-fullpanel-amp0100-freq1300":
        "Laveste vind, uten probe foran, med panel, med genererte bølger (06.11)",

    "20251105-lowestwindUtenProbe2-fullpanel":
        "Laveste vind, uten probe foran, med panel, uten bølger (05.11)",

    "20251105-lowestwind-fullpanel":
        "Laveste vind, med probe foran, med panel, uten bølger (05.11)",

    "20251105-lowestwind":
        "Laveste vind, med probe foran, uten panel, uten bølger (05.11)",

    "20251104-lowestwind":
        "Laveste vind, med probe foran, uten panel, uten bølger (04.11)",
}


# Pattern to extract height (e.g., moh000, moh010, ...)
height_pattern = re.compile(r"moh(\d+)", re.IGNORECASE)

def read_mean_from_file(file_path):
    """Reads a text file and returns the mean of all numeric values."""
    values = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().replace('"', '').replace(',', '.')
            if line:
                try:
                    values.append(float(line))
                except ValueError:
                    pass
    return np.mean(values) if values else np.nan


def process_folder(folder_path):
    """Processes one folder, returning sorted (heights, mean_values)."""
    heights = []
    means = []

    for fname in os.listdir(folder_path):
        if fname.lower().endswith(".txt"):
            match = height_pattern.search(fname)
            if match:
                height = int(match.group(1))
                file_path = os.path.join(folder_path, fname)
                mean_val = read_mean_from_file(file_path)
                heights.append(height)
                means.append(mean_val)

    # Sort by height
    heights, means = zip(*sorted(zip(heights, means)))
    return np.array(heights), np.array(means)

#%%
# --- MAIN PLOTTING ---
def wind_plotter(selected_folders, name="windprofile"):
    markers = ['o', 's', '^', 'D', '*', 'x', '+', 'P', '|', '<', '>']
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = os.path.expanduser("~/Kodevik/wave_project/windresults")
    os.makedirs(fig_path, exist_ok=True)

    datasets = []
    for i, folder in enumerate(selected_folders):
        folder_path = os.path.join(root_dir, folder)
        if not os.path.isdir(folder_path):
            print(f"Skipping (not found): {folder_path}")
            continue
        heights, means = process_folder(folder_path)
        datasets.append((heights, means, markers[i % len(markers)], LEGEND_LABELS.get(folder, folder)))

    for suffix, log_scale in [("linear", False), ("log", True)]:
        fig, ax = plt.subplots(figsize=(8, 6))
        for heights, means, marker, label in datasets:
            ax.plot(means, heights, marker=marker, label=label)
        ax.set_ylabel("Høyde over vannet [mm]")
        ax.set_xlabel("Vindfart [m/s]")
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.legend(fontsize=8)
        if log_scale:
            ax.set_yscale('log')
            ax.set_ylim(5, 400)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}'))
            ax.set_title("Vindprofil (logaritmisk høydeskala)")
        else:
            ax.set_ylim(0, 380)
            ax.yaxis.set_major_locator(plt.MultipleLocator(20))
            ax.yaxis.set_minor_locator(plt.MultipleLocator(5))
            ax.set_title("Vindprofil (lineær skala)")
        fig.tight_layout()
        save_name = os.path.join(fig_path, f"{name}_{suffix}_{ts}.pdf")
        fig.savefig(save_name, bbox_inches="tight")
        print(f"Saved: {save_name}")
        plt.show()
#%%
if __name__ == "__main__":
    selected_folders_full = [
        "20251106-fullwindUtenProbe2-fullpanel-amp0100-freq1300",
        "20251106-fullwindUtenProbe2-fullpanel",
        "20251105-fullwindUtenProbe2-fullpanel",
        "20251104-fullwind"
    ]
    selected_folders_low = [
        "20251106-lowestwindUtenProbe2-fullpanel-amp0100-freq1300",
        "20251105-lowestwindUtenProbe2-fullpanel",
        "20251105-lowestwind-fullpanel",
        "20251105-lowestwind",
        "20251104-lowestwind",
    ]
    print("-RUN 1-")
    wind_plotter(selected_folders_low, name="windprofile_lowestwind")
    print("-RUN 2-")
    wind_plotter(selected_folders_full, name="windprofile_fullwind")
