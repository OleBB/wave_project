#!/usr/bin/env python3
"""
Save publication-quality figures for the thesis.

Run from terminal when you want to regenerate all output figures:
    conda activate draumkvedet
    cd /Users/ole/Kodevik/wave_project
    python main_save_figures.py

Saves to figures/ directory (created if it doesn't exist).
Set save_plot=True in each section when the figure is ready to export.

Requires processed data in waveprocessed/. Run main.py first if missing or stale.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from wavescripts.constants import MEASUREMENT
from wavescripts.filters import (
    apply_experimental_filters,
    damping_all_amplitude_grouper,
    filter_for_frequencyspectrum,
)
from wavescripts.improved_data_loader import load_analysis_data, load_processed_dfs
from wavescripts.plot_utils import WIND_COLOR_MAP
from wavescripts.plotter import (
    plot_all_probes,
    plot_damping_freq,
    plot_damping_scatter,
    plot_frequency_spectrum,
    plot_swell_scatter,
)
from wavescripts.wave_detection import find_first_arrival

FS = MEASUREMENT.SAMPLING_RATE

try:
    file_dir = Path(__file__).resolve().parent
except NameError:
    file_dir = Path.cwd()
os.chdir(file_dir)

figures_dir = file_dir / "figures"
figures_dir.mkdir(exist_ok=True)

# ── Dataset(s) ────────────────────────────────────────────────────────────────
PROCESSED_DIRS = [
    Path("waveprocessed/PROCESSED-20260307-ProbPos4_31_FPV_2-tett6roof"),
]

# ── Load from cache ───────────────────────────────────────────────────────────
print("Loading analysis data...")
combined_meta, _, combined_fft_dict, combined_psd_dict = load_analysis_data(
    *PROCESSED_DIRS
)
processed_dfs = load_processed_dfs(*PROCESSED_DIRS)

# ── Figure: amplitude all probes ──────────────────────────────────────────────
# TODO: set save_plot=True and configure save path when ready
amplitudeplotvariables = {
    "overordnet": {"chooseAll": True, "chooseFirst": False, "chooseFirstUnique": False},
    "filters": {
        "WaveAmplitudeInput [Volt]": None,
        "WaveFrequencyInput [Hz]":   None,
        "WindCondition":             ["no", "lowest", "full"],
        "TunnelCondition":           None,
        "Mooring":                   "low",
        "PanelCondition":            None,
    },
    "plotting": {
        "figsize":    [7, 4],
        "separate":   True,
        "overlay":    False,
        "annotate":   True,
        "save_plot":  False,  # ← set True when ready
    },
}
# m_filtrert = apply_experimental_filters(combined_meta, amplitudeplotvariables)
# plot_all_probes(m_filtrert, amplitudeplotvariables)

# ── Figure: damping vs frequency ──────────────────────────────────────────────
# TODO
# damping_groupedallruns_df = damping_all_amplitude_grouper(combined_meta)
# plot_damping_freq(damping_groupedallruns_df, ...)
# plot_damping_scatter(damping_groupedallruns_df, ...)

# ── Figure: FFT spectrum ──────────────────────────────────────────────────────
# TODO
# filtrert = filter_for_frequencyspectrum(combined_meta, freqplotvariables)
# plot_frequency_spectrum(combined_fft_dict, filtrert, freqplotvariables, data_type="fft")

# ── Figure: swell scatter ─────────────────────────────────────────────────────
# TODO
# plot_swell_scatter(combined_meta, swellplotvariables)

"""
WAVE DETECTION
"""

# ── Figure: first wave arrival ────────────────────────────────────────────────
# Detection parameters
THRESHOLD_FACTOR = 2.0    # 2× stillwater noise floor
WINDOW_S         = 0.5    # rolling window length [s]
MIN_ARRIVAL_S    = 0.5    # arrivals below this are wind-wave artefacts

# Probe positions active in this dataset (must match the loaded config)
PROBE_POSITIONS = ["8804/250", "9373/170", "9373/340", "12545/250"]

# Noise floor: mean stillwater amplitude per probe from combined_meta
_sw_mask = (
    combined_meta["WaveFrequencyInput [Hz]"].isna()
    & (combined_meta["WindCondition"] == "no")
)
_noise_floor = {
    col.replace("Probe ", "").replace(" Amplitude", ""):
        combined_meta.loc[_sw_mask, col].mean()
    for col in combined_meta.columns
    if col.startswith("Probe ") and col.endswith(" Amplitude")
    and "FFT" not in col and "PSD" not in col
    and combined_meta.loc[_sw_mask, col].notna().any()
}

# Detect first arrival for every wave run × probe
_arrival_rows = []
for _, row in combined_meta[combined_meta["WaveFrequencyInput [Hz]"].notna()].iterrows():
    df = processed_dfs.get(row["path"])
    if df is None:
        continue
    for pos in PROBE_POSITIONS:
        eta_col = f"eta_{pos}"
        if eta_col not in df.columns:
            continue
        noise = _noise_floor.get(pos)
        if not noise or noise <= 0:
            continue
        sig = df[eta_col].dropna().values
        idx, t_s = find_first_arrival(sig, noise,
                                      fs=FS,
                                      threshold_factor=THRESHOLD_FACTOR,
                                      window_s=WINDOW_S)
        _arrival_rows.append({
            "run":        Path(row["path"]).name,
            "freq_hz":    row["WaveFrequencyInput [Hz]"],
            "amp_volt":   row.get("WaveAmplitudeInput [Volt]"),
            "wind":       row.get("WindCondition"),
            "panel":      row.get("PanelCondition"),
            "probe":      pos,
            "dist_mm":    int(pos.split("/")[0]),
            "arrival_idx": idx,
            "arrival_s":  t_s,
        })

arrival_df = pd.DataFrame(_arrival_rows)
print(f"Arrival detections: {arrival_df['arrival_s'].notna().sum()} / {len(arrival_df)} probe-runs")

# ── Plot A: faceted by frequency ──────────────────────────────────────────────
_plot_facet = arrival_df.dropna(subset=["arrival_s"])
sns.set_style("ticks", {"axes.grid": True})
g = sns.relplot(
    data=_plot_facet, x="dist_mm", y="arrival_s",
    hue="wind", col="freq_hz",
    kind="line", marker="o", dashes=False,
    errorbar=None, markersize=10, linewidth=0.8,
    height=3.5, aspect=1.3,
    palette=WIND_COLOR_MAP,
    facet_kws={"sharey": True},
)
for ax in g.axes.flat:
    ax.set_xlabel("Probe distance from paddle [mm]")
    ax.set_ylabel("First arrival [s]")
g.figure.suptitle(
    f"First wave arrival  (threshold = {THRESHOLD_FACTOR}× noise floor,"
    f"  window = {WINDOW_S} s)",
    y=1.02, fontsize=9,
)
plt.tight_layout()
# plt.savefig(figures_dir / "first_arrival_faceted.pdf", bbox_inches="tight")
plt.show()

# ── Plot B: all frequencies, single axes, parallel probes averaged ────────────
WIND_LS = {"no": "-", "lowest": "--", "full": ":"}
_plot_single = arrival_df[arrival_df["arrival_s"] > MIN_ARRIVAL_S].copy()
_freqs_sorted = sorted(_plot_single["freq_hz"].dropna().unique())
_freq_colors  = {f: c for f, c in zip(
    _freqs_sorted, plt.cm.rainbow(np.linspace(0, 1, len(_freqs_sorted)))
)}

_agg = (
    _plot_single
    .groupby(["wind", "freq_hz", "dist_mm"])["arrival_s"]
    .agg(mean="mean", err=lambda x: (x.max() - x.min()) / 2)
    .reset_index()
)

fig, ax = plt.subplots(figsize=(9, 5))
for (wind, freq), grp in _agg.groupby(["wind", "freq_hz"]):
    grp_s = grp.sort_values("dist_mm")
    ax.errorbar(grp_s["dist_mm"], grp_s["mean"], yerr=grp_s["err"],
                marker="o", markersize=8, linewidth=1.2, capsize=4,
                color=_freq_colors[freq],
                linestyle=WIND_LS.get(wind, "-"),
                label=f"{freq} Hz / {wind}")

ax.set_xlabel("Probe distance from paddle [mm]")
ax.set_ylabel("First arrival [s]")
ax.set_title(
    f"First wave arrival — all frequencies  "
    f"(threshold {THRESHOLD_FACTOR}× noise,  arrivals > {MIN_ARRIVAL_S} s shown)"
)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], fontsize=8, title="freq / wind")
ax.grid(True, alpha=0.3)
plt.tight_layout()
# plt.savefig(figures_dir / "first_arrival_all_freq.pdf", bbox_inches="tight")
plt.show()

print("main_save_figures: no figures saved yet (all sections are stubs).")
print(f"Output directory: {figures_dir}")
