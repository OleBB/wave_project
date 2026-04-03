#!/usr/bin/env python3
"""
Inline cell-by-cell wave data analysis.

Open in Zed and run cells individually (Shift+Enter or your keybinding).
Plots appear inline in the Zed output panel.

Requires processed data in waveprocessed/. Run main.py first if missing or stale.

For interactive Qt browsers → use main_explore_browser.py (run from terminal).
For saving publication figures → use main_save_figures.py.
"""

# %% ── imports ────────────────────────────────────────────────────────────────
import time

start0 = time.perf_counter()

%matplotlib inline
import os
from pathlib import Path

import numpy as np
import pandas as pd

from wavescripts.constants import ColumnGroups as CG
from wavescripts.constants import GlobalColumns as GC
from wavescripts.constants import ProbeColumns as PC
from wavescripts.filters import (
    apply_experimental_filters,
    damping_all_amplitude_grouper,
    damping_grouper,
    filter_for_amplitude_plot,
    filter_for_damping,
    filter_for_frequencyspectrum,
)

from wavescripts.improved_data_loader import load_analysis_data, load_processed_dfs, ANALYSIS_PROBES
from wavescripts.plot_quicklook import (
    explore_damping_vs_amp,
    explore_damping_vs_freq,
    save_interactive_plot,
)

from wavescripts.plotter import (
    plot_all_probes,
    plot_damping_freq,
    plot_damping_scatter,
    plot_frequency_spectrum,
    plot_reconstructed,
    plot_swell_scatter,
)
end0 = time.perf_counter()
print(f"imports  {end0 - start0:.4f} s")


# %%
try:
    file_dir = Path(__file__).resolve().parent
except NameError:
    file_dir = Path.cwd()
os.chdir(file_dir)

# ── load from cache (fast — no reprocessing) ───────────────────────────────
import time
start = time.perf_counter()

PROCESSED_DIRS = [
    # ── Nov 2025: probe 1 at 18000 mm, roof not fully sealed ──────────────────
    # Path("waveprocessed/PROCESSED-20251005-sixttry6roof-highMooring"),        # probe 1 på 18000 mm; tak ikke tetta helt
    # ── Nov 2025: probe 1 moved to 8804 mm, lowMooring ────────────────────────
    # Path("waveprocessed/PROCESSED-20251110-tett6roof-lowM-ekte580"),          # mange kjøringer med -per15
    # Path("waveprocessed/PROCESSED-20251110-tett6roof-lowMooring"),            # noen kjøringer med -per30
    # Path("waveprocessed/PROCESSED-20251110-tett6roof-lowMooring-2"),          # et par kjøringer med -per15
    # Path("waveprocessed/PROCESSED-20251112-tett6roof"),
    # Path("waveprocessed/PROCESSED-20251113-tett6roof"),
    # Path("waveprocessed/PROCESSED-20251113-tett6roof-loosepaneltaped"),
    # Path("waveprocessed/PROCESSED-20251113-tett6roof-probeadjusted"),
    # ── Mar 2026: new probe positions (march2026_rearranging config) ───────────
    # Path("waveprocessed/PROCESSED-20260305-newProbePos-tett6roof"),           # in=9373/170, out=11800/250 — transitional
    # Path("waveprocessed/PROCESSED-20260306-newProbePos-tett6roof"),           # in=9373/170, out=11800/250 — transitional
    # ── Mar 2026: final probe positions (march2026_better_rearranging) ─────────
    # Path("waveprocessed/PROCESSED-20260307-ProbPos4_31_FPV_2-tett6roof"),    # disse bør være greie bortsett fra de steile
    Path("waveprocessed/PROCESSED-20260312-ProbPos4_31_FPV_2-tett6roof"),      # noen filer med FALSEDATE (riktig dato fra mappe)
    Path("waveprocessed/PROCESSED-20260313-ProbePos4_31_FPV_2-tett6roof"),     # noen filer med FALSEDATE?
    Path("waveprocessed/PROCESSED-20260314-ProbePos4_31_FPV_2-tett6roof"),     # noen filer med FALSEDATE?
    Path("waveprocessed/PROCESSED-20260316-ProbePos4_31_FPV_2-tett6roof"),     # noen filer med FALSEDATE?
    Path("waveprocessed/PROCESSED-20260316-ProbePos4_31_FPV_2-tett6roof-under9Mooring"),
    Path("waveprocessed/PROCESSED-20260319-ProbePos4_31_FPV_2-tett6roof-under9Mooring"),
    Path("waveprocessed/PROCESSED-20260321-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-RENAMED"),
    # ── Mar 2026: probe lowered — height136 (transitional, 1 dag) ─────────────
    Path("waveprocessed/PROCESSED-20260323-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height136"),  # h136/high, 1 dag
    # ── Mar 2026: probe lowered to height100 ──────────────────────────────────
    Path("waveprocessed/PROCESSED-20260323-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100"),
    Path("waveprocessed/PROCESSED-20260324-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100"),
    Path("waveprocessed/PROCESSED-20260325-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100"),
    Path("waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100"),
    # ── Mar 2026: lowrange switch enabled ─────────────────────────────────────
    Path("waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange"),
    Path("waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"),
]

combined_meta, processed_dfs, combined_fft_dict, combined_psd_dict = load_analysis_data(
    *PROCESSED_DIRS, load_processed=False
)
# %%
processed_dfs: dict = {}  # loaded lazily below (wind-only section)

# Paths present in both metadata and FFT dict
matching_paths = set(combined_fft_dict.keys()) & set(combined_meta["path"].unique())
filtered_fft_dict = {p: combined_fft_dict[p] for p in matching_paths}
filtered_meta = combined_meta[combined_meta["path"].isin(matching_paths)].copy()

print(f"Loaded: {len(combined_meta)} total rows, {len(filtered_fft_dict)} wave experiments")
print(f"PanelCondition values: {sorted(combined_meta['PanelCondition'].dropna().unique())}")
print(f"WindCondition values:  {sorted(combined_meta['WindCondition'].dropna().unique())}")
print(f"Frequencies [Hz]:      {sorted(combined_meta['WaveFrequencyInput [Hz]'].dropna().unique())}")

end = time.perf_counter()
print(f"read_parquet and other stuff took {end - start:.4f} s")
# %% ── amplitude — all probes physical layout ─────────────────────────────────
amplitudeplotvariables = {
    "overordnet": {
        "chooseAll": False,
        "chooseFirst": False,
        "chooseFirstUnique": False,
    },
    "filters": {
        "WaveAmplitudeInput [Volt]": None,
        "WaveFrequencyInput [Hz]":   (0.5,1.5),
        "WavePeriodInput":           (29,),
        "WindCondition":             ["full"],
        "TunnelCondition":           None,
        "Mooring":                   None,
        "PanelCondition":            None, #"no", #"["reverse", "full"],
    },
    "plotting": {
        "figsize":   [7, 4],
        "separate":  True,
        "overlay":   False,
        "annotate":  True,
    },
}

m_filtrert = apply_experimental_filters(combined_meta, amplitudeplotvariables)
plot_all_probes(m_filtrert, amplitudeplotvariables)

# %% ── damping grouper + interactive HTML ─────────────────────────────────────
damping_groupedruns_df, damping_pivot_wide = damping_grouper(combined_meta)
save_interactive_plot(damping_groupedruns_df)
# %%
import importlib
import wavescripts.plot_quicklook as pq
importlib.reload(pq)
from wavescripts.plot_quicklook import explore_damping_vs_freq, explore_damping_vs_amp

# %% ── damping vs frequency ───────────────────────────────────────────────────
dampingplotvariables = {
    "overordnet": {"chooseAll": False, "chooseFirst": False, "chooseFirstUnique": False},
    "filters": {
        "WaveAmplitudeInput [Volt]": (0.1,0.3),
        "WaveFrequencyInput [Hz]":   (1.3,1.7),
        "WavePeriodInput":           240,
        "WindCondition":             None,
        "TunnelCondition":           None,
        "PanelCondition":            None, #"reverse",#"full"],
    },
    "plotting": {"figsize": None, "separate": True, "overlay": False, "annotate": True, "single_run_rel_error": 0.10},
}

damping_filtrert = filter_for_damping(damping_groupedruns_df, dampingplotvariables)
explore_damping_vs_freq(damping_filtrert, dampingplotvariables)

# %% ── damping vs amplitude ───────────────────────────────────────────────────
explore_damping_vs_amp(damping_filtrert, dampingplotvariables)

# %% ── damping all amplitudes grouped ────────────────────────────────────────
dampingplotvariables_all = {
    "overordnet": {"chooseAll": False, "chooseFirst": False},
    "filters": {
        "WaveAmplitudeInput [Volt]": 0.1,
        "WaveFrequencyInput [Hz]":   (0.0,1.5),
        "WavePeriodInput":           None,
        "WindCondition":             ["no", "full"], #"lowest"
        "TunnelCondition":           None,
        "PanelCondition":            None,
    },
    "plotting": {
        "show_plot":  True,
        # "save_plot":  False, go to main_save_figures for saving
        "figsize":    (7, 3),
        "separate":   True,
        "facet_by":   None,
        "overlay":    False,
        "annotate":   True,
        "legend":     "outside_right",
        "logaritmic": False,
        "peaks":      7,
        "probes":     ANALYSIS_PROBES,
    },
}

#TODO: fix this to handle actual input... doesnt seem to react to filters.
#TODO: seems like the other plots does it as well..
damping_groupedallruns_df = damping_all_amplitude_grouper(combined_meta)
plot_damping_freq(damping_groupedallruns_df, dampingplotvariables_all)

# %% ── damping scatter ────────────────────────────────────────────────────────
plot_damping_scatter(damping_groupedallruns_df, dampingplotvariables_all)

# %% ── FFT spectrum — filter config ──────────────────────────────────────────
# Adjust filters to match the PanelCondition values in your dataset.
# Run the load cell first and check the printed PanelCondition values.
freqplotvariables = {
    "overordnet": {
        "chooseAll": False,
        "chooseFirst": False,
        "chooseFirstUnique": True,
    },
    "filters": {
        "WaveAmplitudeInput [Volt]": [0.1],
        "WaveFrequencyInput [Hz]":   [1.5],
        "WavePeriodInput":           None,
        "WindCondition":             ["no", "lowest", "full"],
        "TunnelCondition":           None,
        "Mooring":                   None,
        "PanelCondition":            None,  # set to match your data, e.g. "full"
    },
    "plotting": {
        "show_plot":   True,
        # "save_plot":   False, go to main_save_figures for saving
        "figsize":     (5, 5),
        "linewidth":   0.7,
        "facet_by":    "probe",
        "max_points":  120,
        "xlim":        (0, 5.2),
        "legend":      "inside",
        "logaritmic":  False,
        "peaks":       3,
        "probes":      ["12400/250", "9373/340"],
    },
}

filtrert_frequencies = filter_for_frequencyspectrum(combined_meta, freqplotvariables)
print(f"Frequency filter: {len(filtrert_frequencies)} runs matched")

# %% ── FFT spectrum plot ───────────────────────────────────────────────────────
fig, axes = plot_frequency_spectrum(
    combined_fft_dict, filtrert_frequencies, freqplotvariables, data_type="fft"
)

# %% ── PSD spectrum plot ──────────────────────────────────────────────────────
fig, axes = plot_frequency_spectrum(
    combined_psd_dict, filtrert_frequencies, freqplotvariables, data_type="psd"
)

# %% ── swell scatter ──────────────────────────────────────────────────────────
swellplotvariables = {
    "overordnet": {
        "chooseAll": False,
        "chooseFirst": False,
        "chooseFirstUnique": True,
    },
    "filters": {
        "WaveAmplitudeInput [Volt]": [0.1, 0.2, 0.3],
        "WaveFrequencyInput [Hz]":   None,
        "WavePeriodInput":           None,
        "WindCondition":             ["no", "lowest", "full"],
        "TunnelCondition":           None,
        "Mooring":                   None,
        "PanelCondition":            None,  # set to match your data
    },
    "plotting": {
        "show_plot":  True,
        # "save_plot":  False, go to main_save_figures for saving
        "figsize":    (5, 5),
        "linewidth":  0.7,
        "facet_by":   "probe",
        "max_points": 120,
        "xlim":       (0, 5.2),
        "legend":     "inside",
        "logaritmic": False,
        "peaks":      3,
        "probes":     ["12400/250", "9373/170"],
    },
}

plot_swell_scatter(combined_meta, swellplotvariables)

# %% ── wavenumber study ───────────────────────────────────────────────────────
_probe_positions = ANALYSIS_PROBES
wavenumber_cols = [f"Probe {pos} Wavenumber (FFT)" for pos in _probe_positions]
fft_dimension_cols = [CG.fft_wave_dimension_cols(pos) for pos in _probe_positions]
meta_wavenumber = combined_meta[["path"] + [c for c in wavenumber_cols if c in combined_meta.columns]].copy()
print(meta_wavenumber.describe())

# %% ── reconstructed signal — single experiment ───────────────────────────────
# Pick one experiment to inspect its reconstructed time-domain signal.
single_path = filtrert_frequencies["path"].iloc[1]
single_meta = filtrert_frequencies.iloc[[1]]

fig, ax = plot_reconstructed(
    {single_path: filtered_fft_dict[single_path]}, single_meta, freqplotvariables
)

# %% ── reconstructed signal — all filtered experiments ───────────────────────
_recon_paths = {p: filtered_fft_dict[p] for p in filtrert_frequencies["path"] if p in filtered_fft_dict}
fig, axes = plot_reconstructed(
    _recon_paths, filtrert_frequencies, freqplotvariables, data_type="fft"
)

# %% - Wind section
"""
#
#
# ==========================================================================================================================
# WIND-ONLY ANALYSIS
# Runs with no wave input (WaveFrequencyInput NaN) to characterise wind-only
# surface response. Compare wind conditions against stillwater baseline (no wind).
# ==========================================================================================================================
"""
# %%  see the stillwater level — load processed_dfs if not yet in memory
from wavescripts.improved_data_loader import load_processed_dfs
from pathlib import Path
if not processed_dfs:
    PROCESSED_DIRS = sorted(Path("waveprocessed").glob("PROCESSED-*"))
    processed_dfs = load_processed_dfs(*PROCESSED_DIRS)

# %%  plot stillwater drift fit — pick a date to inspect
import importlib
import wavescripts.plot_quicklook as pql
importlib.reload(pql)
from wavescripts.plot_quicklook import plot_stillwater_fit
from wavescripts.improved_data_loader import get_configuration_for_date
from datetime import datetime as _dt

# Auto-pick first date that has nowave runs. Override to inspect a specific day:
#   _sw_date = "2026-03-07"
_nowave_dates = sorted(set(
    pd.to_datetime(combined_meta.loc[combined_meta["WaveFrequencyInput [Hz]"].isna(), "file_date"]
    .dropna()).dt.strftime("%Y-%m-%d").tolist()
))
print(f"Dates with nowave runs: {_nowave_dates}")
_sw_date = _nowave_dates[0]   # change index to pick a different day
print(f"Plotting: {_sw_date}")
_sw_cfg = get_configuration_for_date(_dt.strptime(_sw_date, "%Y-%m-%d"))
plot_stillwater_fit(processed_dfs, combined_meta, _sw_cfg, date=_sw_date)






# %%
import importlib
import wavescripts.constants as _c
importlib.reload(_c)
import wavescripts.plot_quicklook as pql
importlib.reload(pql)
from wavescripts.plot_quicklook import plot_stillwater_fit
plot_stillwater_fit(processed_dfs, combined_meta, _sw_cfg, date="2026-03-07")







# %%
#%% ── wind-only — filter runs ────────────────────────────────────────────────
from pathlib import Path as _Path
import matplotlib.pyplot as plt
from scipy.signal import welch as _welch
from wavescripts.constants import MEASUREMENT

_FS = MEASUREMENT.SAMPLING_RATE

_meta_nowave     = combined_meta[combined_meta["WaveFrequencyInput [Hz]"].isna()].copy()
_meta_wind_only  = _meta_nowave[_meta_nowave["WindCondition"].isin(["full", "lowest"])].copy()
_meta_stillwater = _meta_nowave[_meta_nowave["WindCondition"] == "no"].copy()

# All nowave runs together (stillwater = baseline, wind-only = signal of interest)
_meta_nowave_all = _meta_nowave.copy()

print(f"Wind-only runs ({len(_meta_wind_only)}):")
for _, r in _meta_wind_only.iterrows():
    print(f"  [{r['WindCondition']:7s}]  {_Path(r['path']).name}")
print(f"\nStillwater baseline ({len(_meta_stillwater)}):")
for _, r in _meta_stillwater.iterrows():
    print(f"  {_Path(r['path']).name}")

# %% ── wind-only — PSD dict from cache (nowave PSDs computed in pipeline) ────
_nowave_paths = set(_meta_nowave_all["path"].values)
_wind_psd_dict = {k: v for k, v in combined_psd_dict.items() if k in _nowave_paths}
print(f"wind_psd_dict: {len(_wind_psd_dict)} nowave runs from cache")

# %% ── wind-only — PSD spectrum plot ─────────────────────────────────────────
_wind_psd_plotvars = {
    "overordnet": {"chooseAll": True, "chooseFirst": False, "chooseFirstUnique": True},
    "filters": {
        "WaveFrequencyInput [Hz]": None,
        "WindCondition":           None,
        "PanelCondition":          None,
    },
    "plotting": {
        "show_plot":  True,
        # "save_plot":  False, go to main_save_figures for saving
        "figsize":    (11, 4 * 4),
        "linewidth":  1.0,
        "facet_by":   "probe",
        "max_points": 500,
        "xlim":       (0, 5),
        "legend":     "inside",
        "logaritmic": False,
        "peaks":      0,
        "probes":     ANALYSIS_PROBES,
    },
}

fig, axes = plot_frequency_spectrum(
    _wind_psd_dict, _meta_nowave_all, _wind_psd_plotvars, data_type="psd"
)



# %% ── lazy-load processed_dfs (needed for stats, stillwater plot, arrival) ──
if not processed_dfs:
    print("Loading processed_dfs (~75 MB, ~20 s)...")
    _t0 = time.perf_counter()
    processed_dfs = load_processed_dfs(*PROCESSED_DIRS)
    print(f"Loaded {len(processed_dfs)} runs in {time.perf_counter() - _t0:.1f} s")

# %% ── wind-only — statistics (mean setup + RMS per probe) ───────────────────
_PROBE_POSITIONS = ANALYSIS_PROBES
_stats_rows = []

for _, row_meta in _meta_nowave_all.iterrows():
    df = processed_dfs.get(row_meta["path"])
    if df is None:
        continue
    rec = {
        "file":          _Path(row_meta["path"]).name,
        "WindCondition": row_meta["WindCondition"],
    }
    for pos in _PROBE_POSITIONS:
        eta_col = f"eta_{pos}"
        if eta_col in df.columns:
            sig = df[eta_col].dropna()
            rec[f"mean_mm {pos}"] = round(sig.mean(), 3)
            rec[f"rms_mm  {pos}"] = round(sig.std(),  3)
    _stats_rows.append(rec)

_stats_df = pd.DataFrame(_stats_rows).sort_values("WindCondition")
print(_stats_df.to_string(index=False))

# %% ── investigate: wind-only growth 9373 → 12400 vs claimed wave growth ─────
import pandas as pd

# 1. The suspicious wave run
_wave_run = combined_meta[
    (combined_meta["WaveFrequencyInput [Hz]"] == 0.65) &
    (combined_meta["WaveAmplitudeInput [Volt]"] == 0.1) &
    (combined_meta["WindCondition"] == "full")
].copy()

print("=== Wave run(s) at 0.65 Hz, 0.1 V, full wind ===")
cols = ["path", "PanelCondition", "Mooring",
        "Probe 9373/170 Amplitude", "Probe 9373/250 Amplitude",
        "Probe 9373/340 Amplitude", "Probe 12400/250 Amplitude",
        "Probe 12400/170 Amplitude", "Probe 12400/340 Amplitude",
        "in_position", "out_position", "OUT/IN (FFT)"]
print(_wave_run[[c for c in cols if c in _wave_run.columns]].T.to_string())

# 2. Nowave + full wind: what do the probes read?
_nowave_full = combined_meta[
    combined_meta["WaveFrequencyInput [Hz]"].isna() &
    (combined_meta["WindCondition"] == "full")
].copy()

print(f"\n=== Nowave + full wind runs: {len(_nowave_full)} ===")
amp_cols = [c for c in _nowave_full.columns
            if "Amplitude" in c and "FFT" not in c and "PSD" not in c]
print(_nowave_full[["path", "PanelCondition", "Mooring"] + amp_cols].to_string())

# 3. Stillwater baseline (no wind, no wave)
_stillwater = combined_meta[
    combined_meta["WaveFrequencyInput [Hz]"].isna() &
    (combined_meta["WindCondition"] == "no")
].copy()

print(f"\n=== Stillwater (no wind, no wave): {len(_stillwater)} ===")
print(_stillwater[["path"] + amp_cols].to_string())

# %%
_wave = combined_meta[combined_meta["WaveFrequencyInput [Hz]"].notna()].copy()
_wave["ratio_170_vs_340"] = _wave["Probe 9373/170 Amplitude"] / _wave["Probe 9373/340 Amplitude"]
print(_wave[["WaveFrequencyInput [Hz]", "WaveAmplitudeInput [Volt]", "WindCondition",
             "PanelCondition", "Probe 9373/170 Amplitude", "Probe 9373/340 Amplitude",
             "ratio_170_vs_340"]].sort_values("ratio_170_vs_340", ascending=False).to_string())

# %% ── repl_out helper — tee stdout to repl/<name>.txt ───────────────────────
import contextlib, sys as _sys
from pathlib import Path as _Path

@contextlib.contextmanager
def repl_out(filename: str):
    """Write cell output to repl/<filename> while still printing to terminal."""
    _outdir = _Path("repl")
    _outdir.mkdir(exist_ok=True)
    _orig = _sys.stdout
    class _Tee:
        def __init__(self, f): self._f = f
        def write(self, s): _orig.write(s); self._f.write(s)
        def flush(self): _orig.flush(); self._f.flush()
    with open(_outdir / filename, "w") as _f:
        _sys.stdout = _Tee(_f)
        try:
            yield
        finally:
            _sys.stdout = _orig
    print(f"→ saved to repl/{filename}")

"""
åssen bruke repl out:
with repl_out("filnavn.txt"):
    print(f"Nowave+fullwind rows: {len(funksjon)}")
"""
# %% ── stillwater — overview + zoom ──────────────────────────────────────────
import matplotlib.pyplot as plt
_sw_row = _meta_stillwater.iloc[2]
_sw_df  = processed_dfs.get(_sw_row["path"])

if _sw_df is None:
    print("processed_dfs not loaded — run the lazy-load cell first")
else:
    _eta_cols = sorted([c for c in _sw_df.columns if c.startswith("eta_")])
    _t = np.arange(len(_sw_df)) / _FS
    _colors = plt.cm.tab10(np.linspace(0, 0.9, len(_eta_cols)))

    fig, (ax_ov, ax_zm) = plt.subplots(2, 1, figsize=(14, 6),
                                        gridspec_kw={"height_ratios": [2, 1]})
    for col, color in zip(_eta_cols, _colors):
        label = col.replace("eta_", "")
        ax_ov.plot(_t, _sw_df[col], lw=0.6, label=label, color=color)

    ax_ov.set_xlabel("Time [s]")
    ax_ov.set_ylabel("η [mm]")
    ax_ov.set_title(f"Stillwater — {_Path(_sw_row['path']).name}", fontsize=10)
    ax_ov.legend(fontsize=8, loc="upper right")
    ax_ov.grid(True, alpha=0.3)

    _mid = len(_sw_df) // 2
    _zm_slice = slice(_mid - 12, _mid + 13)
    for col, color in zip(_eta_cols, _colors):
        ax_zm.plot(_t[_zm_slice], _sw_df[col].iloc[_zm_slice],
                   lw=1.2, marker=".", markersize=5,
                   label=col.replace("eta_", ""), color=color)

    ax_zm.set_xlabel("Time [s]")
    ax_zm.set_ylabel("η [mm]")
    ax_zm.set_title("Zoom — 25 samples", fontsize=9)
    ax_zm.grid(True, alpha=0.3)

    plt.show()
    print(_sw_row["path"])
    print(f"{len(_sw_df)} samples  |  {len(_sw_df)/_FS:.1f} s")

# %% ── stillwater — probe uncertainty statistics ──────────────────────────────
# Noise floor = minimum (P97.5-P2.5)/2 over short sliding windows.
# Window must be SHORT (0.2 s) so that slow tank sloshing (2–10 s period)
# appears as a flat DC offset within the window and does not contribute to
# the percentile spread. What remains is pure probe noise: rapid sample-to-
# sample jitter from electronics and surface tension capillary ripples.

_SW_WINDOW_S  = 0.2    # window length [s] — shorter than slowest slosh period
_SW_WINDOW_N  = int(_SW_WINDOW_S * _FS)

_probe_cols = [p for p in ANALYSIS_PROBES if any(
    f"eta_{p}" in df.columns for df in processed_dfs.values()
)]

_sw_rows = []
for _, _row in _meta_stillwater.iterrows():
    _df = processed_dfs.get(_row["path"])
    if _df is None:
        continue
    _entry = {"run": _Path(_row["path"]).name}
    for _pos in _probe_cols:
        _col = f"eta_{_pos}"
        if _col not in _df.columns:
            _entry[_pos] = np.nan
            continue
        _sig = _df[_col].values
        # slide window, compute (P97.5-P2.5)/2 in each window
        _amps = []
        for _s in range(0, len(_sig) - _SW_WINDOW_N, _SW_WINDOW_N // 2):
            _chunk = _sig[_s : _s + _SW_WINDOW_N]
            _chunk = _chunk[~np.isnan(_chunk)]
            if len(_chunk) < _SW_WINDOW_N // 2:
                continue
            _amps.append((np.percentile(_chunk, 97.5) - np.percentile(_chunk, 2.5)) / 2)
        _entry[_pos] = min(_amps) if _amps else np.nan
    _sw_rows.append(_entry)

_sw_stats_all = pd.DataFrame(_sw_rows)
print(f"=== Stillwater probe noise floor — min over {_SW_WINDOW_S}s windows [mm] ===")
print(_sw_stats_all.to_string(index=False))

# --- flag: hard cap on the windowed minimum (truly broken probe / very bad run) ---
_STILLWATER_EXCLUDE = ["nestenstille"]         # known bad by name (kept for reference)
_STILLWATER_AMP_CAP = 0.5                      # mm — flag if windowed min still exceeds this

_name_flag = _sw_stats_all["run"].apply(
    lambda r: any(kw in r for kw in _STILLWATER_EXCLUDE)
)
_amp_flag = _sw_stats_all[_probe_cols].max(axis=1) > _STILLWATER_AMP_CAP

_flagged = _sw_stats_all[_name_flag | _amp_flag]
if not _flagged.empty:
    print(f"\n⚠ Flagged (name match or windowed min > {_STILLWATER_AMP_CAP} mm):")
    print(_flagged.to_string(index=False))

_sw_stats = _sw_stats_all[~(_name_flag | _amp_flag)].copy()
print(f"\n=== Accepted runs: {len(_sw_stats)} / {len(_sw_stats_all)} ===")
print(_sw_stats.to_string(index=False))

# Noise floor: mean of windowed-minimum across accepted runs
_sw_summary = _sw_stats[_probe_cols].agg(["mean", "std", "min", "max"]).T
_sw_summary.index.name = "probe"
print("\n=== Per-probe noise floor summary [mm] ===")
print(_sw_summary.round(4).to_string())

# %% ── speed-of-sound / lab temperature diagnostic ────────────────────────────
# The probe hardware measures the speed of sound in air per sample and logs it
# as column "Mach" in every raw CSV.  The pipeline now extracts mean±std per
# run into combined_meta as sound_speed_mean_ms / sound_speed_std_ms.
#
# c_air ≈ 331 + 0.606 × T_Celsius  [m/s]  →  T ≈ (c − 331) / 0.606
# (dry air; humidity adds ~0.1–0.3 % at typical indoor RH — negligible here)
#
# This cell:
#   1. Plots c_air (mean per run) vs date — shows seasonal/daily lab temperature
#   2. Compares November 2025 vs March 2026 sessions
#   3. Identifies the late-March high-humidity sessions by their lower c_air
#   4. Quantifies the worst-case amplitude scale error
#
# Physical bottom line:
#   - Total spread: ~1.24 m/s across all sessions = 0.36 %
#   - Scale error on 10 mm wave: ~0.036 mm  (well below 0.25 mm target)
#   - Hardware almost certainly self-compensates (measures c to compute distance)
#   - For OUT/IN ratios: error is ZERO (same air column, both probes, same moment)

_c_df = combined_meta[["file_date", "experiment_folder", "sound_speed_mean_ms", "sound_speed_std_ms"]].dropna(subset=["sound_speed_mean_ms"]).copy()
_c_df["file_date"] = pd.to_datetime(_c_df["file_date"])
_c_df["T_approx_C"] = (_c_df["sound_speed_mean_ms"] - 331.0) / 0.606

print("=== Speed-of-sound per run ===")
print(_c_df[["file_date", "experiment_folder", "sound_speed_mean_ms", "sound_speed_std_ms", "T_approx_C"]].round(3).to_string(index=False))

# Scale error relative to a fixed assumed c_ref (probe factory default ~ 343 m/s)
_c_ref = 343.0
_c_df["scale_error_pct"] = (_c_df["sound_speed_mean_ms"] - _c_ref).abs() / _c_ref * 100
print(f"\nWorst-case scale error: {_c_df['scale_error_pct'].max():.3f} %")
print(f"  → for a 10 mm wave: {_c_df['scale_error_pct'].max() * 0.1:.4f} mm")
print(f"  → for a 30 mm wave: {_c_df['scale_error_pct'].max() * 0.3:.4f} mm")
print(f"  → for OUT/IN ratio: 0.000 mm (systematic cancels in ratio)")

fig_cs, ax_cs = plt.subplots(figsize=(10, 3))
ax_cs.scatter(_c_df["file_date"], _c_df["sound_speed_mean_ms"],
              c=_c_df["T_approx_C"], cmap="coolwarm", s=12, zorder=3)
ax_cs.errorbar(_c_df["file_date"], _c_df["sound_speed_mean_ms"],
               yerr=_c_df["sound_speed_std_ms"], fmt="none", color="gray", alpha=0.4, lw=0.8)
ax2_cs = ax_cs.twinx()
ax2_cs.set_ylabel("Approx. air temp [°C]", color="gray")
ax2_cs.set_ylim([(y - 331) / 0.606 for y in ax_cs.get_ylim()])
ax_cs.set_ylabel("c_air [m/s]")
ax_cs.set_title("Speed-of-sound per run (hardware measurement, 'Mach' column)")
ax_cs.axhline(343.0, ls="--", color="k", lw=0.7, label="c = 343 m/s (~20 °C reference)")
ax_cs.legend(fontsize=8)
fig_cs.tight_layout()
plt.show()

# %% ── first wave arrival detection ───────────────────────────────────────────
# For each wave run × probe: find the first time the rolling amplitude exceeds
# threshold_factor × stillwater noise floor.
# Physics: fast long-wave precursors may arrive well before the _SNARVEI_CALIB start.
from wavescripts.wave_detection import find_first_arrival

_THRESHOLD_FACTOR = 5.0   # detection at 2× noise floor !!todo: EDIT! figure out a good value.
_WINDOW_S         = 2.5   # rolling window length [s]
_PROBE_POSITIONS  = ANALYSIS_PROBES

# Noise floor per probe from stillwater summary (mean across runs)
_noise_floor = _sw_summary["mean"].to_dict()   # {"8804/250": 0.34, ...}

_arrival_rows = []
for _, _row in combined_meta[combined_meta["WaveFrequencyInput [Hz]"].notna()].iterrows():
    _df = processed_dfs.get(_row["path"])
    if _df is None:
        continue
    for _pos in _PROBE_POSITIONS:
        _eta_col = f"eta_{_pos}"
        if _eta_col not in _df.columns:
            continue
        _noise = _noise_floor.get(_pos)
        if _noise is None or _noise <= 0:
            continue
        _sig = _df[_eta_col].dropna().values
        _idx, _t_s = find_first_arrival(_sig, _noise,
                                         fs=_FS,
                                         threshold_factor=_THRESHOLD_FACTOR,
                                         window_s=_WINDOW_S)
        _arrival_rows.append({
            "run":           _Path(_row["path"]).name,
            "freq_hz":       _row["WaveFrequencyInput [Hz]"],
            "amp_volt":      _row.get("WaveAmplitudeInput [Volt]"),
            "wind":          _row.get("WindCondition"),
            "panel":         _row.get("PanelCondition"),
            "probe":         _pos,
            "dist_mm":       int(_pos.split("/")[0]),
            "arrival_idx":   _idx,
            "arrival_s":     _t_s,
            "snarvei_s":     None,   # placeholder — fill from _SNARVEI_CALIB if needed
        })

_arrival_df = pd.DataFrame(_arrival_rows)
print(f"Arrival detections: {_arrival_df['arrival_s'].notna().sum()} / {len(_arrival_df)} probe-runs")
print(_arrival_df.sort_values(["freq_hz", "probe"]).to_string(index=False))

# %% ── first arrival — stacked subplots, one per frequency ───────────────────
from wavescripts.plot_utils import WIND_COLOR_MAP as _WCM

_plot_df  = _arrival_df.dropna(subset=["arrival_s"])
_freqs    = sorted(_plot_df["freq_hz"].dropna().unique())
_winds    = ["no", "lowest", "full"]

fig, axes = plt.subplots(len(_freqs), 1, figsize=(7, 2.5 * len(_freqs)), sharey=False)
if len(_freqs) == 1:
    axes = [axes]

for ax, freq in zip(axes, _freqs):
    _sub = _plot_df[_plot_df["freq_hz"] == freq]
    for wind in _winds:
        _w = _sub[_sub["wind"] == wind].sort_values("dist_mm")
        if _w.empty:
            continue
        ax.plot(_w["dist_mm"], _w["arrival_s"],
                marker="o", linewidth=0.8, markersize=7,
                color=_WCM.get(wind, "gray"), label=wind)
    # label each probe position with a dashed vertical line + name at top
    _ref = _sub.sort_values("dist_mm").drop_duplicates("probe")
    for _, _r in _ref.iterrows():
        ax.axvline(_r["dist_mm"], color="0.75", linewidth=0.4, linestyle="--", zorder=0)
        ax.text(_r["dist_mm"], 1.01, _r["probe"],
                fontsize=7, ha="center", va="bottom", color="0.4",
                transform=ax.get_xaxis_transform())
    ax.set_title(f"{freq} Hz", fontsize=9)
    ax.set_ylabel("First arrival [s]")
    ax.grid(True, linewidth=0.4)
    ax.legend(fontsize=8)

axes[-1].set_xlabel("Probe distance from paddle [mm]")
fig.suptitle(
    f"First wave arrival  (threshold={_THRESHOLD_FACTOR}× noise floor, window={_WINDOW_S} s)",
    fontsize=9, y=1.01,
)
plt.show()

# %% ── wind-only — overview + zoom (solo & parallel probes) ───────────────────
_wind_row = _meta_wind_only.iloc[1]
_wind_df  = processed_dfs.get(_wind_row["path"])

if _wind_df is None:
    print("processed_dfs not loaded — run the lazy-load cell first")
else:
    _wind_cond = _wind_row.get("WindCondition", "?")
    _run_name  = _Path(_wind_row["path"]).name
    _t = np.arange(len(_wind_df)) / _FS
    _mid = len(_wind_df) // 2
    _zm_slice = slice(_mid - 125, _mid + 125)

    # Group eta columns by longitudinal distance
    _all_eta = sorted([c for c in _wind_df.columns if c.startswith("eta_") and not c.endswith("_interp")])
    from collections import defaultdict
    _by_dist = defaultdict(list)
    for _col in _all_eta:
        _pos  = _col.replace("eta_", "")          # e.g. "9373/170"
        _dist = int(_pos.split("/")[0])
        _by_dist[_dist].append(_col)

    _solo_cols     = [cols[0] for cols in _by_dist.values() if len(cols) == 1]
    _parallel_cols = [col for cols in _by_dist.values() if len(cols) > 1 for col in cols]
    print(f"eta cols in df:  {_all_eta}")
    print(f"solo:     {_solo_cols}")
    print(f"parallel: {_parallel_cols}")

    def _wind_plot(eta_cols, subtitle, separate=False):
        _colors = [plt.cm.tab10(i / 10) for i in range(len(eta_cols))]
        if separate:
            n = len(eta_cols)
            fig, axes = plt.subplots(2, n, figsize=(7 * n, 6),
                                     gridspec_kw={"height_ratios": [2, 1]},
                                     sharey="row")
            if n == 1:
                axes = axes.reshape(2, 1)
            for i, (col, color) in enumerate(zip(eta_cols, _colors)):
                label = col.replace("eta_", "")
                axes[0, i].plot(_t, _wind_df[col], lw=0.6, color=color)
                axes[1, i].plot(_t[_zm_slice], _wind_df[col].iloc[_zm_slice], lw=1.0, color=color)
                axes[0, i].set_title(f"{label}", fontsize=10)
                axes[0, i].set_xlabel("Time [s]")
                axes[0, i].set_ylabel("η [mm]")
                axes[0, i].grid(True, alpha=0.3)
                axes[1, i].set_xlabel("Time [s]")
                axes[1, i].set_ylabel("η [mm]")
                axes[1, i].set_title("Zoom — 1 s window", fontsize=9)
                axes[1, i].grid(True, alpha=0.3)
            fig.suptitle(f"Wind-only ({_wind_cond}) — {subtitle} — {_run_name}", fontsize=10)
        else:
            fig, (ax_ov, ax_zm) = plt.subplots(2, 1, figsize=(14, 6),
                                                gridspec_kw={"height_ratios": [2, 1]})
            for col, color in zip(eta_cols, _colors):
                label = col.replace("eta_", "")
                ax_ov.plot(_t, _wind_df[col], lw=0.6, label=label, color=color)
                ax_zm.plot(_t[_zm_slice], _wind_df[col].iloc[_zm_slice],
                           lw=1.0, label=label, color=color)
            ax_ov.set_xlabel("Time [s]")
            ax_ov.set_ylabel("η [mm]")
            ax_ov.set_title(f"Wind-only ({_wind_cond}) — {subtitle} — {_run_name}", fontsize=10)
            ax_ov.legend(fontsize=8, loc="upper right")
            ax_ov.grid(True, alpha=0.3)
            ax_zm.set_xlabel("Time [s]")
            ax_zm.set_ylabel("η [mm]")
            ax_zm.set_title("Zoom — 1 s window (250 samples)", fontsize=9)
            ax_zm.grid(True, alpha=0.3)

        plt.show()

    if _solo_cols:
        _wind_plot(_solo_cols,     "solo probes", separate=True)
    if _parallel_cols:
        _wind_plot(_parallel_cols, "parallel probes")

    print(f"{len(_wind_df)} samples  |  {len(_wind_df)/_FS:.1f} s  |  wind: {_wind_cond}")
    print(f"Solo:     {[c.replace('eta_','') for c in _solo_cols]}")
    print(f"Parallel: {[c.replace('eta_','') for c in _parallel_cols]}")

# %% ── wind background — A: broadband (time-domain) amplitude, all probes ──────
# "Probe {pos} Amplitude" = (P97.5 - P2.5) / 2 of the FULL time-domain signal.
# Includes ALL frequencies: paddle + wind waves + noise.
# For nowave+wind runs: this IS the wind amplitude (no paddle wave present).
# For wave+wind runs at IN probe: wind (~9 mm) >> paddle (2–5 mm at 0.1 V)
#   → A_td is wind-dominated → OUT/IN from time-domain is meaningless under full wind.
# For wave+wind runs at OUT probe: wind is sheltered (~0.9 mm) → A_td still usable.

_fullwind_nowave = _meta_wind_only[_meta_wind_only["WindCondition"] == "full"].copy()
_nowind_nowave   = _meta_stillwater.copy()
_fullwind_wave   = filtered_meta[filtered_meta["WindCondition"] == "full"].copy()
_nowind_wave     = filtered_meta[filtered_meta["WindCondition"] == "no"].copy()

print("=== Broadband (time-domain) amplitude: fullwind NOWAVE runs per probe ===")
print(f"{'probe':12s}  {'n':>4}  {'median':>8}  {'std':>7}  {'min':>7}  {'max':>7}  mm")
_bg_td = {}   # {pos: median A_td_wind} — wind background level
for _pos in ANALYSIS_PROBES:
    _col  = f"Probe {_pos} Amplitude"
    _vals = _fullwind_nowave[_col].dropna() if _col in _fullwind_nowave.columns else pd.Series(dtype=float)
    if _vals.empty:
        continue
    _bg_td[_pos] = _vals.median()
    print(f"  {_pos:12s}  {len(_vals):4d}  {_vals.median():8.3f}  {_vals.std():7.3f}  "
          f"{_vals.min():7.3f}  {_vals.max():7.3f}")

if "_sw_summary" in vars():
    print("\n=== Wind/noise ratio per probe (broadband wind / stillwater noise floor) ===")
    for _pos in ANALYSIS_PROBES:
        _nf = _sw_summary.loc[_pos, "mean"] if _pos in _sw_summary.index else np.nan
        _aw = _bg_td.get(_pos, np.nan)
        if np.isfinite(_nf) and np.isfinite(_aw) and _nf > 0:
            print(f"  {_pos:12s}  wind={_aw:.3f} mm  noise={_nf:.3f} mm  ratio={_aw/_nf:.0f}×")

print("\n=== Broadband amplitude in wave+fullwind vs wave+nowind (median across all freqs) ===")
print(f"{'probe':12s}  {'A_td nowind':>12}  {'A_td fullwind':>14}  {'wind/nowind':>11}  → wind adds this factor to td")
for _pos in ANALYSIS_PROBES:
    _col  = f"Probe {_pos} Amplitude"
    _v_nw = _nowind_wave[_col].dropna().median()   if _col in _nowind_wave.columns   else np.nan
    _v_fw = _fullwind_wave[_col].dropna().median() if _col in _fullwind_wave.columns else np.nan
    if np.isfinite(_v_nw) and np.isfinite(_v_fw):
        print(f"  {_pos:12s}  {_v_nw:12.3f}  {_v_fw:14.3f}  {_v_fw/_v_nw:11.2f}×")

# %% ── wind background — B: spectral contamination at paddle frequencies ───────
#
# KEY THEORY: The pipeline FFT amplitude uses a 0.1 Hz window centred on f_paddle.
# Wind waves are broadband but have a low-frequency tail. Wind energy AT f_paddle
# directly contaminates the FFT amplitude even when most wind energy is at 3–5 Hz.
#
# From Parseval / one-sided PSD:
#   For incoherent broadband noise with PSD Pxx(f) [mm²/Hz],
#   the equivalent sinusoidal amplitude within a band B = 0.1 Hz is:
#
#       A_wind_FFT(f) = sqrt( 2 · ∫_{f-B/2}^{f+B/2}  Pxx(f') df' )
#
#   For a sine wave of amplitude A at exactly f₀:  Pxx(f₀) = A²/(2·df)
#   → A = sqrt(2 · Pxx · df).  Wind waves are NOT lines — they occupy a band —
#   so we integrate over 0.1 Hz rather than using a single bin.
#
# A_wind_FFT(f) is the measurement floor for the FFT amplitude at frequency f.
# Any measured FFT amplitude less than ≈ 5× A_wind_FFT(f) is unreliable.
#
# TWO COMPETING BIASES in fullwind wave runs:
#   (1) CONTAMINATION: wind energy at f_paddle adds to A_paddle_FFT(IN)
#       → measured A_IN is inflated → OUT/IN biased DOWNWARD (more damping than real)
#       (IN probe exposed to wind; OUT probe sheltered → asymmetric contamination)
#   (2) COHERENCE LOSS: paddle wave phase jitter under wind spreads FFT energy
#       → A_IN_FFT underestimates the true paddle amplitude (wave_stability < 1)
#       → OUT/IN biased UPWARD (less damping than real)
#   Which dominates is frequency- and amplitude-dependent. This cell quantifies (1).

_FFT_WINDOW_HZ = 0.1  # window used by compute_amplitudes_from_fft

_paddle_freqs = sorted(combined_meta["WaveFrequencyInput [Hz]"].dropna().unique())
print(f"Paddle frequencies: {[f'{f:.2f}' for f in _paddle_freqs]} Hz")

if not _wind_psd_dict:
    print("_wind_psd_dict is empty — check nowave paths in combined_psd_dict")
else:
    _psd_sample = next(v for v in _wind_psd_dict.values() if v is not None and len(v) > 1)
    _psd_freqs  = _psd_sample.index.values.astype(float)
    _df_psd     = float(_psd_freqs[1] - _psd_freqs[0])
    print(f"PSD frequency resolution: {_df_psd:.4f} Hz  "
          f"(Pxx amplitude unit: mm²/Hz)")

    # Average PSD over fullwind nowave runs only
    _fw_paths     = set(_fullwind_nowave["path"].values)
    _wind_psds_fw = {k: v for k, v in _wind_psd_dict.items() if k in _fw_paths}
    print(f"Fullwind nowave PSDs: {len(_wind_psds_fw)}")

    _mean_pxx = {}   # {pos: array(n_freqs)}
    _std_pxx  = {}
    for _pos in ANALYSIS_PROBES:
        _col = f"Pxx {_pos}"
        _stack = [v[_col].values for v in _wind_psds_fw.values()
                  if v is not None and _col in v.columns]
        if not _stack:
            continue
        _arr = np.vstack(_stack)
        _mean_pxx[_pos] = _arr.mean(axis=0)
        _std_pxx[_pos]  = _arr.std(axis=0)

    # Integrate Pxx over 0.1 Hz band at each paddle frequency
    _wind_amp_fft = {}   # {pos: {freq: A_wind_FFT_mm}}
    for _pos, _pxx in _mean_pxx.items():
        _wind_amp_fft[_pos] = {}
        for _f in _paddle_freqs:
            _mask = ((_psd_freqs >= _f - _FFT_WINDOW_HZ / 2) &
                     (_psd_freqs <= _f + _FFT_WINDOW_HZ / 2))
            if not _mask.any():
                continue
            _A = np.sqrt(2.0 * np.trapezoid(_pxx[_mask], _psd_freqs[_mask]))
            _wind_amp_fft[_pos][_f] = _A

    # ── Print contamination table ─────────────────────────────────────────────
    print(f"\n=== A_wind_FFT (mm): wind noise amplitude in 0.1 Hz FFT window at paddle frequencies ===")
    print(f"  (= minimum detectable signal; SNR < 5× = unreliable FFT measurement)")
    _hdr = f"  {'probe':12s}" + "".join(f"  {f:6.2f}Hz" for f in _paddle_freqs)
    print(_hdr)
    for _pos in ANALYSIS_PROBES:
        if _pos not in _wind_amp_fft:
            continue
        _row = f"  {_pos:12s}"
        for _f in _paddle_freqs:
            _a = _wind_amp_fft[_pos].get(_f, np.nan)
            _row += f"  {_a:8.4f}" if np.isfinite(_a) else f"  {'NaN':>8}"
        print(_row)

    # ── Plot: mean wind PSD ± 1σ per probe, paddle-freq markers annotated ─────
    _n_probe = len(_mean_pxx)
    fig, axes = plt.subplots(_n_probe, 1, figsize=(11, 3.2 * _n_probe), sharex=True)
    if _n_probe == 1:
        axes = [axes]
    for ax, (_pos, _pxx_m) in zip(axes, _mean_pxx.items()):
        _pxx_s = _std_pxx.get(_pos, np.zeros_like(_pxx_m))
        _pxx_lo = np.maximum(_pxx_m - _pxx_s, 1e-12)
        ax.fill_between(_psd_freqs, _pxx_lo, _pxx_m + _pxx_s,
                        alpha=0.2, color="steelblue")
        ax.semilogy(_psd_freqs, _pxx_m, lw=1.0, color="steelblue",
                    label=f"mean ± 1σ  (n={len(_wind_psds_fw)})")
        _ybot, _ytop = ax.get_ylim()
        for _f in _paddle_freqs:
            _a = _wind_amp_fft.get(_pos, {}).get(_f, np.nan)
            ax.axvline(_f, color="tomato", lw=0.7, ls="--", alpha=0.75)
            if np.isfinite(_a):
                ax.text(_f + 0.02, _ytop * 0.6,
                        f"A={_a:.4f}", fontsize=6, color="tomato",
                        ha="left", va="top", rotation=90)
        ax.set_xlim(0, 5.5)
        ax.set_ylabel("PSD [mm²/Hz]", fontsize=8)
        ax.set_title(_pos, fontsize=9)
        ax.grid(True, alpha=0.25, which="both")
        ax.legend(fontsize=7)
    axes[-1].set_xlabel("Frequency [Hz]")
    fig.suptitle(
        "Wind PSD (fullwind nowave) — mean ± 1σ per probe\n"
        "Red dashes = paddle frequencies  |  annotations = A_wind_FFT(f) in 0.1 Hz band [mm]",
        fontsize=9,
    )
    plt.tight_layout()
    plt.show()

# %% ── wind background — C: spectral SNR and coherence on wave runs ────────────
#
# Spectral SNR = A_paddle_FFT / A_wind_FFT(f_paddle)
#
#   SNR > 10  → clean: wind < 10% of signal
#   SNR 3–10  → use with caution: wind is 10–33% of signal
#   SNR < 3   → unreliable: wind dominates FFT amplitude
#
# Note: this measures bias (1) — contamination that inflates A_IN → depresses OUT/IN.
# Bias (2) — coherence loss — is captured by wave_stability (< 1.0 under full wind):
#   a stable sinusoid has wave_stability ≈ 1; a phase-jittered wave < 1.
#   Loss of stability means FFT underestimates A_IN → inflates OUT/IN.
# Both biases affect the same runs. SNR + wave_stability together constrain the
# net measurement error, even if we cannot separate the two effects from the signal alone.

if not _wind_amp_fft:
    print("Run cell B first to build _wind_amp_fft")
else:
    _snr_rows = []
    for _, _row in filtered_meta.iterrows():
        _freq = _row.get("WaveFrequencyInput [Hz]")
        if not np.isfinite(float(_freq)):
            continue
        _f_key = min(_paddle_freqs, key=lambda _f: abs(_f - _freq))
        for _pos in ANALYSIS_PROBES:
            _fft_col  = f"Probe {_pos} Amplitude (FFT)"
            _a_paddle = _row.get(_fft_col, np.nan)
            _a_wind   = _wind_amp_fft.get(_pos, {}).get(_f_key, np.nan)
            if not (np.isfinite(float(_a_paddle)) and np.isfinite(float(_a_wind)) and float(_a_wind) > 0):
                continue
            _snr_rows.append({
                "freq":       float(_freq),
                "amp_v":      _row.get("WaveAmplitudeInput [Volt]"),
                "wind":       _row.get("WindCondition"),
                "panel":      _row.get("PanelCondition"),
                "probe":      _pos,
                "A_paddle":   float(_a_paddle),
                "A_wind_FFT": float(_a_wind),
                "SNR":        float(_a_paddle) / float(_a_wind),
            })

    _snr_df = pd.DataFrame(_snr_rows)
    print(f"SNR rows: {len(_snr_df)}")

    # Summary table: median SNR per probe × wind × frequency
    _snr_agg = (
        _snr_df.groupby(["probe", "wind", "freq"])["SNR"]
        .agg(median="median", min="min", n="count")
        .reset_index()
    )
    print("\n=== Spectral SNR: A_paddle_FFT / A_wind_FFT(f) ===")
    print(f"  {'probe':12s}  {'wind':8s}  {'freq':5s}  {'median':>8}  {'min':>6}  {'n':>4}  flag")
    for _, r in _snr_agg.sort_values(["probe", "wind", "freq"]).iterrows():
        flag = ""
        if r["median"] < 3:
            flag = "  !! CRITICAL — wind dominates"
        elif r["median"] < 5:
            flag = "  ⚠ LOW — use with caution"
        print(f"  {r['probe']:12s}  {r['wind']:8s}  {r['freq']:5.2f}  "
              f"{r['median']:8.1f}  {r['min']:6.1f}  {int(r['n']):4d}{flag}")

    # ── Plot: SNR vs frequency per probe, coloured by wind condition ──────────
    _wcm = {"no": "steelblue", "lowest": "goldenrod", "full": "tomato"}
    _probes_in = [p for p in ANALYSIS_PROBES if p in _snr_df["probe"].values]
    fig, axes = plt.subplots(1, len(_probes_in), figsize=(4.5 * len(_probes_in), 4), sharey=True)
    if len(_probes_in) == 1:
        axes = [axes]
    for ax, _pos in zip(axes, _probes_in):
        _sub = _snr_df[_snr_df["probe"] == _pos]
        for _wc, _grp in _sub.groupby("wind"):
            _agg = _grp.groupby("freq")["SNR"].median()
            ax.plot(_agg.index, _agg.values, marker="o", lw=1.2, markersize=6,
                    color=_wcm.get(_wc, "grey"), label=_wc)
        ax.axhline(10, color="0.6", lw=0.7, ls=":",  label="SNR=10")
        ax.axhline(5,  color="k",   lw=0.8, ls="--", label="SNR=5")
        ax.axhline(3,  color="tomato", lw=0.8, ls=":", label="SNR=3 critical")
        ax.set_xlabel("Paddle freq [Hz]")
        if ax is axes[0]:
            ax.set_ylabel("Spectral SNR  (A_paddle / A_wind_FFT)")
        ax.set_title(_pos, fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.suptitle(
        "Spectral SNR = A_paddle_FFT / A_wind_FFT(f_paddle)\n"
        "Quantifies bias (1): wind contamination at paddle frequency inflates A_IN → depresses OUT/IN",
        fontsize=9,
    )
    plt.tight_layout()
    plt.show()

    # ── wave_stability (bias 2: coherence loss) ───────────────────────────────
    _stab_col = "IN wave_stability"
    if _stab_col in filtered_meta.columns:
        print(f"\n=== IN probe wave_stability under full wind ===")
        print(f"  (< 1.0 = phase jitter → FFT underestimates A_IN → OUT/IN inflated)")
        _wstab = (
            filtered_meta[filtered_meta["WindCondition"] == "full"]
            .groupby("WaveFrequencyInput [Hz]")[_stab_col]
            .agg(median="median", min="min", n="count")
            .rename(columns={"median": "median_stability"})
        )
        print(_wstab.round(3).to_string())
        print(f"\n  nowind reference:")
        _wstab_nw = (
            filtered_meta[filtered_meta["WindCondition"] == "no"]
            .groupby("WaveFrequencyInput [Hz]")[_stab_col]
            .agg(median="median", n="count")
        )
        print(_wstab_nw.round(3).to_string())
        print(f"\n  Stability drop = coherence loss fraction = how much FFT amplitude is underestimated.")
        print(f"  If stability drops from 0.97 to 0.60: FFT amp underestimated by ~(1 - 0.60/0.97) = {1 - 0.60/0.97:.0%}.")
    else:
        print(f"\n('{_stab_col}' not in filtered_meta — run processor2nd.py to add generic IN/OUT columns)")

# %% ── wind background — D: time-domain vs FFT amplitude (wave runs) ──────────
#
# Under no wind: A_td ≈ A_FFT  (no wind background; both measure the paddle wave)
# Under full wind at IN:  A_td >> A_FFT
#   A_td²  ≈ A_paddle² + A_wind_broad²    (all-frequency incoherent sum, ~9 mm wind)
#   A_FFT² ≈ A_paddle² + A_wind_FFT(f)²  (0.1 Hz band only, much smaller)
# Under full wind at OUT: A_td ≈ A_FFT
#   panel shelters wind → A_wind_broad_OUT ≈ 0.9 mm, small vs A_paddle
#
# Ratio A_FFT/A_td shows what fraction of the signal energy is at the paddle frequency.
# At IN probe + fullwind: ratio << 1 → time-domain is useless for OUT/IN.
# The scatter and ratio plots together show why A_FFT is the only valid metric.

_wcm = {"no": "steelblue", "lowest": "goldenrod", "full": "tomato"}

fig, axes = plt.subplots(2, len(ANALYSIS_PROBES),
                         figsize=(4 * len(ANALYSIS_PROBES), 8))

for _ci, _pos in enumerate(ANALYSIS_PROBES):
    _td_col  = f"Probe {_pos} Amplitude"
    _fft_col = f"Probe {_pos} Amplitude (FFT)"
    if _td_col not in filtered_meta.columns or _fft_col not in filtered_meta.columns:
        continue
    _sub = filtered_meta[[_td_col, _fft_col, "WindCondition",
                           "WaveFrequencyInput [Hz]"]].dropna(subset=[_td_col, _fft_col])
    _sub = _sub[_sub["WaveFrequencyInput [Hz]"].notna()].copy()
    _sub["ratio"] = _sub[_fft_col] / _sub[_td_col]

    ax_sc = axes[0, _ci]   # scatter: A_td vs A_FFT
    ax_rt = axes[1, _ci]   # ratio A_FFT/A_td vs frequency

    for _wc, _grp in _sub.groupby("WindCondition"):
        _c = _wcm.get(_wc, "grey")
        ax_sc.scatter(_grp[_td_col], _grp[_fft_col], s=12, alpha=0.45, color=_c, label=_wc)
        _agg = _grp.groupby("WaveFrequencyInput [Hz]")["ratio"].median()
        ax_rt.plot(_agg.index, _agg.values, marker="o", lw=1.0, markersize=5,
                   color=_c, label=_wc)

    _lim = _sub[[_td_col, _fft_col]].max().max() * 1.08
    ax_sc.plot([0, _lim], [0, _lim], "k--", lw=0.7, alpha=0.4, label="1:1")
    ax_sc.set_xlim(0, _lim); ax_sc.set_ylim(0, _lim)
    ax_sc.set_xlabel("A_td [mm]  (all-freq, P97.5−P2.5)/2")
    ax_sc.set_ylabel("A_FFT [mm]  (0.1 Hz at paddle freq)" if _ci == 0 else "")
    ax_sc.set_title(_pos, fontsize=9)
    ax_sc.legend(fontsize=7)
    ax_sc.grid(True, alpha=0.3)

    ax_rt.axhline(1.0, color="k", lw=0.7, ls="--", alpha=0.5)
    ax_rt.set_ylim(0, 1.15)
    ax_rt.set_xlabel("Paddle freq [Hz]")
    ax_rt.set_ylabel("A_FFT / A_td  (1.0 = identical)" if _ci == 0 else "")
    ax_rt.set_title(f"{_pos}  ratio", fontsize=9)
    ax_rt.legend(fontsize=7)
    ax_rt.grid(True, alpha=0.3)

fig.suptitle(
    "A_td (all-frequency) vs A_FFT (paddle frequency only) per probe\n"
    "Ratio → 1 under no wind; → 0 at IN probe under full wind → time-domain is wind-dominated\n"
    "OUT probe ratio ≈ 1 even under wind (panel blocks wind) → A_td and A_FFT agree at OUT",
    fontsize=9,
)
plt.tight_layout()
plt.show()

# %% ── wind background — E: rolling RMS stationarity + half-period threshold ───
# Rolling RMS tests whether wind background is stationary (constant level through run).
# Non-stationarity (rubber-band splash, wind gusts) means a single threshold won't work.
# Half-period window (~78 samples at 1.6 Hz) is the detection timescale of
# RampDetectionBrowser — shows the worst-case RMS level the detector sees before
# the paddle wave arrives, under full wind.

_WIND_RMS_PROBE = "9373/170"
_WIND_RMS_WIN_S = 1.0
_WIND_RMS_WIN_N = int(_WIND_RMS_WIN_S * _FS)
_HW_TARGET      = 1.6                              # Hz — shortest paddle period
_HW_WIN_N       = int(0.5 / _HW_TARGET * _FS)     # half-period in samples (~78)
_HW_STEP        = _HW_WIN_N // 2

print(f"Long-window RMS:    {_WIND_RMS_WIN_N} samples = {_WIND_RMS_WIN_S:.1f} s")
print(f"Half-period window: {_HW_WIN_N} samples = {_HW_WIN_N / _FS * 1000:.0f} ms "
      f"(half-period at {_HW_TARGET} Hz)")

if not processed_dfs:
    print("processed_dfs not loaded — run the lazy-load cell first")
else:
    _rms_runs = []
    for _, _row in _fullwind_nowave.iterrows():
        _df = processed_dfs.get(_row["path"])
        if _df is None:
            continue
        _eta_col = f"eta_{_WIND_RMS_PROBE}"
        if _eta_col not in _df.columns:
            continue
        _sig = np.where(np.isnan(_df[_eta_col].values), 0.0, _df[_eta_col].values)
        # 1 s non-overlapping windows
        _n_win  = len(_sig) // _WIND_RMS_WIN_N
        _t_ctr  = (np.arange(_n_win) + 0.5) * _WIND_RMS_WIN_S
        _rms_1s = np.array([
            np.sqrt(np.mean(_sig[i * _WIND_RMS_WIN_N:(i + 1) * _WIND_RMS_WIN_N] ** 2))
            for i in range(_n_win)
        ])
        # half-period overlapping windows
        _starts  = np.arange(0, len(_sig) - _HW_WIN_N, _HW_STEP)
        _rms_hw  = np.array([
            np.sqrt(np.mean(_sig[s:s + _HW_WIN_N] ** 2)) for s in _starts
        ])
        _rms_runs.append({
            "run":      _Path(_row["path"]).name,
            "t_ctr":    _t_ctr,
            "rms_1s":   _rms_1s,
            "cv_1s":    _rms_1s.std() / _rms_1s.mean() if _rms_1s.mean() > 0 else np.nan,
            "hw_mean":  _rms_hw.mean(),
            "hw_p99":   np.percentile(_rms_hw, 99),
            "hw_max":   _rms_hw.max(),
        })

    print(f"\n=== Rolling RMS ({_WIND_RMS_WIN_S} s) — {_WIND_RMS_PROBE} fullwind nowave ===")
    print(f"  {'run':60s}  {'mean_1s':>8}  {'std_1s':>7}  {'CV':>5}  {'hw_p99':>7}  {'hw_max':>7}  mm")
    for r in _rms_runs:
        _stationarity = "NONSTATIONARY" if r["cv_1s"] > 0.15 else "ok"
        print(f"  {r['run'][:58]:58s}  {r['rms_1s'].mean():8.3f}  "
              f"{r['rms_1s'].std():7.3f}  {r['cv_1s']:5.3f}  "
              f"{r['hw_p99']:7.3f}  {r['hw_max']:7.3f}  {_stationarity}")

    if _rms_runs:
        # Summary across runs
        _hw_p99s = np.array([r["hw_p99"] for r in _rms_runs])
        _hw_maxs = np.array([r["hw_max"] for r in _rms_runs])
        print(f"\n  Half-period RMS across ALL fullwind nowave runs:")
        print(f"    mean p99 = {_hw_p99s.mean():.3f} ± {_hw_p99s.std():.3f} mm")
        print(f"    overall max = {_hw_maxs.max():.3f} mm  (worst-case detector sees)")
        print(f"    → first-motion threshold > {_hw_p99s.mean():.3f} mm : < 1% false positives from wind")
        print(f"    → first-motion threshold > {_hw_maxs.max():.3f} mm : zero false positives")
        if "_noise_floor" in vars() and _WIND_RMS_PROBE in _noise_floor:
            _nf = _noise_floor[_WIND_RMS_PROBE]
            print(f"    stillwater noise floor at {_WIND_RMS_PROBE}: {_nf:.3f} mm")
            print(f"    wind p99 / noise floor = {_hw_p99s.mean()/_nf:.0f}× "
                  f"→ noise-floor threshold gives false positives in wind")

        # Plot: rolling 1 s RMS for up to 4 runs
        _n_plot = min(len(_rms_runs), 4)
        fig, axes = plt.subplots(_n_plot, 1, figsize=(12, 3 * _n_plot), sharex=False)
        if _n_plot == 1:
            axes = [axes]
        for ax, r in zip(axes, _rms_runs[:_n_plot]):
            ax.plot(r["t_ctr"], r["rms_1s"], lw=0.9, color="steelblue")
            ax.axhline(r["rms_1s"].mean(),              color="k",   lw=0.8, ls="--",
                       label=f"mean {r['rms_1s'].mean():.3f}")
            ax.axhline(r["rms_1s"].mean() + r["rms_1s"].std(), color="0.5", lw=0.6, ls=":")
            ax.axhline(r["hw_p99"],                     color="tomato", lw=0.7, ls="--",
                       label=f"hw_p99 {r['hw_p99']:.3f}")
            ax.set_title(f"{r['run'][:80]}  (CV={r['cv_1s']:.3f})", fontsize=8)
            ax.set_ylabel("RMS [mm]")
            ax.set_xlabel("Time [s]")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        fig.suptitle(
            f"Rolling 1 s RMS — {_WIND_RMS_PROBE}  fullwind nowave\n"
            "Flat = stationary (CV < 0.15); spikes = splash or gust events\n"
            "Red dashes = half-period p99 (worst-case for first-motion detector)",
            fontsize=9, y=1.01,
        )
        plt.tight_layout()
        plt.show()

# %% ── first arrival — no-wind only, all frequencies ─────────────────────────
_MIN_ARRIVAL_S = 0.5   # below this = instrument/wind transient, ignored
_plot_df2 = _arrival_df[
    (_arrival_df["wind"] == "no") & (_arrival_df["arrival_s"] > _MIN_ARRIVAL_S)
].copy()

_freqs_sorted = sorted(_plot_df2["freq_hz"].dropna().unique())
_freq_colors  = {f: c for f, c in zip(_freqs_sorted,
                  plt.cm.rainbow(np.linspace(0, 1, len(_freqs_sorted))))}

# Average parallel probes (same dist_mm, same freq) → mean ± half-range
_agg = (
    _plot_df2
    .groupby(["freq_hz", "dist_mm"])["arrival_s"]
    .agg(mean="mean", err=lambda x: (x.max() - x.min()) / 2)
    .reset_index()
)

fig, ax = plt.subplots(figsize=(9, 5))
for freq, grp in _agg.groupby("freq_hz"):
    grp_s = grp.sort_values("dist_mm")
    ax.errorbar(grp_s["dist_mm"], grp_s["mean"], yerr=grp_s["err"],
                marker="o", markersize=8, linewidth=1.2, capsize=4,
                color=_freq_colors[freq], label=f"{freq} Hz")

# probe position labels at top
for _pos in _PROBE_POSITIONS:
    _d = int(_pos.split("/")[0])
    ax.axvline(_d, color="0.75", linewidth=0.4, linestyle="--", zorder=0)
    ax.text(_d, 1.01, _pos, fontsize=7, ha="center", va="bottom",
            color="0.4", transform=ax.get_xaxis_transform())

ax.set_xlabel("Probe distance from paddle [mm]")
ax.set_ylabel("First arrival [s]")
ax.set_title("First wave arrival — no wind, all frequencies")
_handles, _labels = ax.get_legend_handles_labels()
ax.legend(_handles[::-1], _labels[::-1], fontsize=8, title="frequency")
ax.grid(True, alpha=0.3)
plt.show()

_thresh_str = ",  ".join(
    f"{pos} → {_THRESHOLD_FACTOR * _noise_floor[pos]:.2f} mm"
    for pos in _PROBE_POSITIONS if pos in _noise_floor
)
print(
    f"Caption: First arrival time at each probe vs. distance from paddle, no-wind runs only. "
    f"Detection threshold: {_THRESHOLD_FACTOR}× stillwater noise floor "
    f"({_thresh_str}), "
    f"rolling window {_WINDOW_S} s. "
    f"Arrivals ≤ {_MIN_ARRIVAL_S} s excluded as instrument transients. "
    f"Error bars show half-range across parallel probes at the same longitudinal distance."
)

# %% ── quick-load: one dataset for arrival detection ─────────────────────────
# Load only the last PROCESSED_DIR. Skipped if already loaded this session.
if "_arr_meta" not in vars():
    _arr_dir  = sorted(Path("waveprocessed").glob("PROCESSED-*"))[-1]
    print(f"Loading: {_arr_dir.name}")
    _arr_meta, _arr_dfs, _, _ = load_analysis_data(_arr_dir, load_processed=True)
    print(f"  {len(_arr_meta)} runs  |  {len(_arr_dfs)} timeseries loaded")
else:
    print(f"_arr_meta already in memory ({len(_arr_meta)} runs) — skipping load")

# %% ── upcrossing-based arrival detection ─────────────────────────────────────
def _find_arrival_upcross(signal, fs, target_freq, threshold_mm,
                           min_period_factor=0.25, max_period_factor=4.0):
    """
    Detect first wave arrival using zero-upcrossing cycle analysis.

    Does NOT look for the target frequency specifically. Instead it finds
    all zero-upcrossings, measures each cycle's period and peak-to-trough
    amplitude, and returns the first cycle whose amplitude exceeds
    threshold_mm and whose period is plausible (between min and max factor
    × target period).

    This captures the ramp buildup naturally: the first few cycles often
    have longer/varying periods at low amplitude, then amplitude grows as
    the wave train develops. The first cycle above threshold is the arrival.

    Parameters
    ----------
    signal           : 1-D array, already zeroed (eta_ column)
    fs               : sampling rate [Hz]
    target_freq      : nominal wave frequency [Hz], used only to set the
                       plausible period window (not for FFT)
    threshold_mm     : amplitude [mm] above which a cycle counts as arrived
    min_period_factor: reject cycles shorter than this × (1/target_freq).
                       Filters out noise crossings. Default 0.25.
    max_period_factor: reject cycles longer than this × (1/target_freq).
                       Filters out slow sloshing. Default 4.0.

    Returns
    -------
    (idx, t_s) : sample index and time [s] of the START of the first
                 detected cycle. (None, None) if never detected.
    """
    sig = np.where(np.isnan(signal), 0.0, np.asarray(signal, dtype=float))
    target_period_s   = 1.0 / target_freq
    min_cycle_samples = int(min_period_factor * target_period_s * fs)
    max_cycle_samples = int(max_period_factor * target_period_s * fs)

    # zero-upcrossings: sample where sig goes from negative to non-negative
    crossings = np.where((sig[:-1] < 0) & (sig[1:] >= 0))[0]

    for i in range(len(crossings) - 1):
        c0 = crossings[i]
        c1 = crossings[i + 1]
        cycle_len = c1 - c0

        if cycle_len < min_cycle_samples or cycle_len > max_cycle_samples:
            continue   # noise crossing or slow slosh — skip

        chunk = sig[c0:c1]
        amp   = chunk.max() - chunk.min()   # peak-to-trough

        if amp >= threshold_mm:
            return int(c0), c0 / fs

    return None, None

_THRESH_FACTOR      = 2.0   # × noise floor
_MIN_PERIOD_FACTOR  = 0.25  # reject crossings shorter than 25% of target period (noise)
_MAX_PERIOD_FACTOR  = 4.0   # reject crossings longer than 4× target period (sloshing)

_periodic_rows = []
for _, _row in _arr_meta[
        _arr_meta["WaveFrequencyInput [Hz]"].notna() &
        (_arr_meta["WindCondition"] == "no")
].iterrows():
    _df = _arr_dfs.get(_row["path"])
    if _df is None:
        continue
    _freq = float(_row["WaveFrequencyInput [Hz]"])
    for _pos in _PROBE_POSITIONS:
        _eta_col = f"eta_{_pos}"
        if _eta_col not in _df.columns:
            continue
        _noise = _noise_floor.get(_pos)
        if _noise is None or _noise <= 0:
            continue
        _sig = _df[_eta_col].values
        _idx, _t_s = _find_arrival_upcross(
            _sig, _FS, _freq,
            threshold_mm=_THRESH_FACTOR * _noise,
            min_period_factor=_MIN_PERIOD_FACTOR,
            max_period_factor=_MAX_PERIOD_FACTOR,
        )
        _periodic_rows.append({
            "run":       _Path(_row["path"]).name,
            "freq_hz":   _freq,
            "amp_volt":  _row.get("WaveAmplitudeInput [Volt]"),
            "probe":     _pos,
            "dist_mm":   int(_pos.split("/")[0]),
            "arrival_s": _t_s,
        })

_periodic_df = pd.DataFrame(_periodic_rows)
print(f"Period-based detections: {_periodic_df['arrival_s'].notna().sum()} / {len(_periodic_df)}")

# %% ── period-based arrival — plot ────────────────────────────────────────────
_pagg = (
    _periodic_df.dropna(subset=["arrival_s"])
    .groupby(["freq_hz", "dist_mm"])["arrival_s"]
    .agg(mean="mean", err=lambda x: (x.max() - x.min()) / 2)
    .reset_index()
)
_pfreqs   = sorted(_pagg["freq_hz"].unique())
_pcolors  = {f: c for f, c in zip(_pfreqs,
              plt.cm.rainbow(np.linspace(0, 1, len(_pfreqs))))}

fig, ax = plt.subplots(figsize=(9, 5))
for freq, grp in _pagg.groupby("freq_hz"):
    grp_s = grp.sort_values("dist_mm")
    ax.errorbar(grp_s["dist_mm"], grp_s["mean"], yerr=grp_s["err"],
                marker="o", markersize=8, linewidth=1.2, capsize=4,
                color=_pcolors[freq], label=f"{freq} Hz")

for _pos in _PROBE_POSITIONS:
    _d = int(_pos.split("/")[0])
    ax.axvline(_d, color="0.75", linewidth=0.4, linestyle="--", zorder=0)
    ax.text(_d, 1.01, _pos, fontsize=7, ha="center", va="bottom",
            color="0.4", transform=ax.get_xaxis_transform())

ax.set_xlabel("Probe distance from paddle [mm]")
ax.set_ylabel("First arrival [s]")
ax.set_title(f"First wave arrival — upcrossing-based detection, no wind  "
             f"(threshold {_THRESH_FACTOR}× noise, period window "
             f"{_MIN_PERIOD_FACTOR}–{_MAX_PERIOD_FACTOR}× target)")
_h, _l = ax.get_legend_handles_labels()
ax.legend(_h[::-1], _l[::-1], fontsize=8, title="frequency")
ax.grid(True, alpha=0.3)
plt.show()

_thresh_str2 = ",  ".join(
    f"{pos} → {_THRESH_FACTOR * _noise_floor[pos]:.2f} mm"
    for pos in _PROBE_POSITIONS if pos in _noise_floor
)
print(
    f"Caption: First arrival detected by zero-upcrossing cycle analysis. "
    f"Each cycle's peak-to-trough amplitude is compared to {_THRESH_FACTOR}× "
    f"stillwater noise floor per probe ({_thresh_str2}). "
    f"Cycles outside {_MIN_PERIOD_FACTOR}–{_MAX_PERIOD_FACTOR}× the target period are rejected. "
    f"No-wind runs only. "
    f"Error bars show half-range across parallel probes at the same longitudinal distance."
)

# %% ── pre-arrival zero-upcrossing frequency ───────────────────────────────────
# For each nowind run × probe: take the signal from t=0 to the period-based
# arrival. Count zero-upcrossings and compute their mean frequency.
# Question: are the pre-arrival oscillations the target frequency (just too weak
# to trigger), a sub-harmonic, or a different mode entirely?

def _upcrossing_freq(signal, fs):
    """Mean frequency from zero upcrossings. Returns NaN if < 2 crossings."""
    sig = np.asarray(signal, dtype=float)
    sig = sig[~np.isnan(sig)]
    if len(sig) < 4:
        return np.nan
    crossings = np.where((sig[:-1] < 0) & (sig[1:] >= 0))[0]
    if len(crossings) < 2:
        return np.nan
    periods_s = np.diff(crossings) / fs
    return 1.0 / np.mean(periods_s)

_pre_rows = []
for _, _row in _periodic_df.dropna(subset=["arrival_s"]).iterrows():
    _run_meta = combined_meta[combined_meta["path"].str.endswith(_row["run"])].iloc[0]
    _df = processed_dfs.get(_run_meta["path"])
    if _df is None:
        continue
    _eta_col = f"eta_{_row['probe']}"
    if _eta_col not in _df.columns:
        continue
    _arr_idx = int(_row["arrival_s"] * _FS)
    _pre_sig = _df[_eta_col].values[:_arr_idx]
    _f_pre = _upcrossing_freq(_pre_sig, _FS)
    _pre_rows.append({
        "run":        _row["run"],
        "freq_hz":    _row["freq_hz"],
        "probe":      _row["probe"],
        "dist_mm":    _row["dist_mm"],
        "arrival_s":  _row["arrival_s"],
        "pre_freq":   _f_pre,
        "ratio":      _f_pre / _row["freq_hz"] if np.isfinite(_f_pre) else np.nan,
    })

_pre_df = pd.DataFrame(_pre_rows)
print(_pre_df[["freq_hz", "probe", "arrival_s", "pre_freq", "ratio"]]
      .sort_values(["freq_hz", "probe"])
      .to_string(index=False, float_format="{:.3f}".format))

# %% ── pre-arrival frequency — plot ratio to target ──────────────────────────
_probes_sorted = sorted(_pre_df["probe"].dropna().unique())

# mean ± half-range across runs at the same frequency (panel/amp shouldn't matter pre-arrival)
_pre_agg = (
    _pre_df.dropna(subset=["ratio"])
    .groupby(["probe", "freq_hz"])["ratio"]
    .agg(mean="mean", err=lambda x: (x.max() - x.min()) / 2, n="count")
    .reset_index()
)

fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True, sharex=True)
for ax, pos in zip(axes.flat, _probes_sorted):
    _sub = _pre_agg[_pre_agg["probe"] == pos]
    ax.errorbar(_sub["freq_hz"], _sub["mean"], yerr=_sub["err"],
                marker="o", linewidth=1.0, markersize=7, capsize=4,
                color="steelblue")
    # annotate n per point
    for _, r in _sub.iterrows():
        ax.text(r["freq_hz"], r["mean"] + r["err"] + 0.02,
                f"n={int(r['n'])}", fontsize=6, ha="center", color="0.5")
    ax.axhline(1.0, color="k", linewidth=0.8, linestyle="--")
    ax.set_title(pos, fontsize=9)
    ax.set_xlabel("Target frequency [Hz]")
    ax.set_ylabel("pre-arrival freq / target freq")
    ax.grid(True, alpha=0.3)

fig.suptitle("Pre-arrival oscillation frequency relative to target wave\n"
             "(mean ± half-range across runs, no wind)", fontsize=10)
plt.show()

# %% ── tank swell tail — mstop90 runs ────────────────────────────────────────
# mstopXX naming convention: recording continues XX seconds AFTER the wavemaker stops.
# Total recording = wave_duration + XX s.
# Wave stop time = total_duration - XX s  (derived per-run from actual signal length).

import re as _re

def _parse_mstop_tail(path):
    """Extract the mstopXX tail duration from filename. Returns tail_s or None."""
    m = _re.search(r"mstop(\d+)", _Path(path).name)
    return int(m.group(1)) if m else None

_mstop_meta = combined_meta[combined_meta["path"].str.contains("mstop90")]
print(f"mstop90 runs: {len(_mstop_meta)}")
print(_mstop_meta[["path", "WaveFrequencyInput [Hz]", "WaveAmplitudeInput [Volt]",
                    "WindCondition", "PanelCondition"]].to_string(index=False))

# %% ── swell tail — wind-only baseline PSD (same date as mstop90 runs) ────────
from scipy.signal import welch as _welch

_mstop_date = "2026-03-23"
_wind_baseline_meta = combined_meta[
    (combined_meta["file_date"].astype(str).str.startswith(_mstop_date)) &
    (combined_meta["WindCondition"] == "full") &
    (combined_meta["WaveFrequencyInput [Hz]"].isna())
]
print(f"Wind-only baseline runs ({_mstop_date}): {len(_wind_baseline_meta)}")

_wind_psd_baseline  = {}   # {pos: (freqs, Pxx_mean)}
_wind_ts_baseline   = {}   # {pos: signal_array}  — one representative run
for _, _row in _wind_baseline_meta.iterrows():
    _df = processed_dfs.get(_row["path"])
    if _df is None:
        continue
    for _pos in _probe_cols:
        _col = f"eta_{_pos}"
        if _col not in _df.columns:
            continue
        _sig = _df[_col].values
        _f, _p = _welch(np.where(np.isnan(_sig), 0.0, _sig), fs=_FS, nperseg=4096)
        if _pos not in _wind_psd_baseline:
            _wind_psd_baseline[_pos] = (_f, _p)
            _wind_ts_baseline[_pos]  = _sig          # keep first run as reference trace
        else:
            _wind_psd_baseline[_pos] = (_f, (_wind_psd_baseline[_pos][1] + _p) / 2)

# %% ── swell tail — combined timeseries + backwards PSD, one probe per row ────
# Layout: 4 rows (probes ordered by distance) × 2 columns (timeseries | PSD)
# Timeseries: full recording, wave-stop marker + clearance marker
# PSD: backwards 2s windows, red→green = early→late in tail

_WIN_S           = 2.0
_WIN_N           = int(_WIN_S * _FS)
_NPERSEG         = min(512, _WIN_N)
_SWELL_BAND      = (0.0, 2.0)
_SLOSH_THRESHOLD = 3.0    # × wind-only swell energy — tune after inspection

def _swell_ratio(sig, fs, baseline_f, baseline_p, band, nperseg):
    """Swell-band PSD energy ratio vs wind-only baseline."""
    sig = np.where(np.isnan(sig), 0.0, sig)
    f, p = _welch(sig, fs=fs, nperseg=nperseg)
    mask   = (f >= band[0]) & (f <= band[1])
    mask_b = (baseline_f >= band[0]) & (baseline_f <= band[1])
    e_sig  = np.trapezoid(p[mask],         f[mask])         if mask.any()   else 0.0
    e_base = np.trapezoid(baseline_p[mask_b], baseline_f[mask_b]) if mask_b.any() else 1.0
    return e_sig / max(e_base, 1e-12)

# probes sorted near→far from paddle
_probes_by_dist = sorted(_probe_cols, key=lambda p: int(p.split("/")[0]))
_mstop_sorted   = _mstop_meta.sort_values("WaveFrequencyInput [Hz]").reset_index(drop=True)

# ── pre-compute everything for all runs ──────────────────────────────────────
_run_data = []   # list of dicts, one per run
for _, _row in _mstop_sorted.iterrows():
    _df = processed_dfs.get(_row["path"])
    if _df is None:
        continue
    _tail_s   = _parse_mstop_tail(_row["path"])
    _total_s  = len(_df) / _FS
    _stop_idx = int((_total_s - _tail_s) * _FS)
    _t_full   = np.arange(len(_df)) / _FS

    _entry = {
        "df":        _df,
        "t_full":    _t_full,
        "wave_stop": _total_s - _tail_s,
        "stop_idx":  _stop_idx,
        "label":     f"{_row.get('WaveAmplitudeInput [Volt]', '?')}V  "
                     f"{_row.get('WaveFrequencyInput [Hz]', '?')}Hz  "
                     f"{_row.get('WindCondition', '?')}wind",
        "probes":    {},   # {pos: {ratios, t_centres, clearance_s, cmap_arr}}
    }
    for pos in _probes_by_dist:
        _col = f"eta_{pos}"
        if _col not in _df.columns or pos not in _wind_psd_baseline:
            continue
        _tail  = _df[_col].values[_stop_idx:]
        _bf, _bp = _wind_psd_baseline[pos]
        _n_wins  = len(_tail) // _WIN_N
        _ratios, _t_centres = [], []
        for i in range(_n_wins):
            _seg = _tail[i * _WIN_N : (i + 1) * _WIN_N]
            _ratios.append(_swell_ratio(_seg, _FS, _bf, _bp, _SWELL_BAND, _NPERSEG))
            _t_centres.append((i + 0.5) * _WIN_S)
        _clearance_s = None
        for i in range(_n_wins - 1, -1, -1):
            if _ratios[i] > _SLOSH_THRESHOLD:
                _clearance_s = _t_centres[i] + _WIN_S / 2
                break
        _entry["probes"][pos] = {
            "tail":        _tail,
            "ratios":      _ratios,
            "t_centres":   _t_centres,
            "clearance_s": _clearance_s,
            "cmap_arr":    plt.cm.RdYlGn(np.linspace(0.1, 0.9, max(_n_wins, 1))),
        }
    _run_data.append(_entry)

# ── one figure per probe, runs as columns ────────────────────────────────────
n_runs = len(_run_data)

for pos in _probes_by_dist:
    _dist_mm = int(pos.split("/")[0])
    _bf, _bp = _wind_psd_baseline.get(pos, (None, None))
    if _bf is None:
        continue

    fig, axes = plt.subplots(2, n_runs,
                             figsize=(5.5 * n_runs, 7),
                             gridspec_kw={"height_ratios": [2, 1.5]})
    if n_runs == 1:
        axes = axes[:, np.newaxis]

    for col_i, rd in enumerate(_run_data):
        ax_ts  = axes[0, col_i]
        ax_psd = axes[1, col_i]
        pd_    = rd["probes"].get(pos)

        ax_ts.set_title(rd["label"], fontsize=8)

        if pd_ is None:
            ax_ts.set_visible(False)
            ax_psd.set_visible(False)
            continue

        _col = f"eta_{pos}"

        # timeseries
        ax_ts.plot(rd["t_full"], rd["df"][_col], lw=0.5, color="steelblue")
        # overlay wind-only reference (same length as full recording, or cropped)
        _wind_ref = _wind_ts_baseline.get(pos)
        if _wind_ref is not None:
            _n_ref = min(len(_wind_ref), len(rd["t_full"]))
            ax_ts.plot(rd["t_full"][:_n_ref], _wind_ref[:_n_ref],
                       lw=0.4, color="darkorange", alpha=0.5, label="wind-only ref")
        ax_ts.axvline(rd["wave_stop"], color="red", lw=1.0, linestyle="--",
                      label=f"stop {rd['wave_stop']:.1f} s")
        if pd_["clearance_s"] is not None:
            ax_ts.axvline(rd["wave_stop"] + pd_["clearance_s"],
                          color="darkgreen", lw=1.2, linestyle="--",
                          label=f"+{pd_['clearance_s']:.0f} s clear")
        ax_ts.legend(fontsize=7, loc="upper right")
        ax_ts.grid(True, alpha=0.3)
        if col_i == 0:
            ax_ts.set_ylabel("η [mm]", fontsize=8)

        # PSD cascade
        _n_wins = len(pd_["t_centres"])
        _label_every = max(1, _n_wins // 8)
        for i, (tc, r) in enumerate(zip(pd_["t_centres"], pd_["ratios"])):
            _seg = pd_["tail"][i * _WIN_N : (i + 1) * _WIN_N]
            _seg = np.where(np.isnan(_seg), 0.0, _seg)
            _f, _p = _welch(_seg, fs=_FS, nperseg=_NPERSEG)
            _lbl = f"{tc:.0f}s ×{r:.1f}" if i % _label_every == 0 else "_nolegend_"
            ax_psd.semilogy(_f, _p, color=pd_["cmap_arr"][i], lw=0.8, label=_lbl)
        ax_psd.semilogy(_bf, _bp, color="darkorange", lw=1.5,
                        linestyle="--", label="wind-only")
        ax_psd.axvspan(*_SWELL_BAND, color="0.92", zorder=0)
        ax_psd.set_xlim(0, 6)
        ax_psd.grid(True, alpha=0.3, which="both")
        ax_psd.legend(fontsize=6, ncol=1, loc="lower center")
        _clr = pd_["clearance_s"]
        ax_psd.set_xlabel("Frequency [Hz]", fontsize=8)
        ax_psd.set_title(f"{'clear +'+str(int(_clr))+' s' if _clr else 'already clear'}",
                         fontsize=8, color="darkred" if _clr else "green")
        if col_i == 0:
            ax_psd.set_ylabel("PSD [mm²/Hz]", fontsize=8)

    fig.suptitle(
        f"Probe  {pos}  —  {_dist_mm} mm from paddle\n"
        f"threshold = {_SLOSH_THRESHOLD}×  |  red→green = early→late in tail",
        fontsize=10
    )
    plt.show()
#TODO: is the wind noise centered at the same baseline in these plots?
# TODO: does wind "increase" the baseline? does it move water backwards in the tank?
#todo: should we remove any obvious outliers from this plot?

# %% ── tail period tracking — seiche emergence after mstop ───────────────────
# After the wavemaker stops, paddle-frequency energy dissipates and lower-
# frequency (longer-period) seiche modes emerge.
#
# Physical prediction:
#   - Phase 1 (ramp-down): ~paddle frequency, decaying amplitude
#   - Phase 2 (free decay): still ~paddle freq, further decay each wall-bounce
#   - Phase 3 (seiche): period grows toward tank seiche harmonics
#       T1 = 2L/c  (L~25m, c=sqrt(g*d)=sqrt(9.81*0.580)~2.39 m/s → T1~20.9 s)
#       T2 = T1/2 ~10.4 s,  T3 = T1/3 ~7.0 s
#
# Method: zero-upcrossing period tracking (per cycle, no windowing).
#   eta_ is already zeroed to the stillwater level, so each time the signal
#   crosses zero upward is one wave cycle boundary. The period of cycle i is
#   simply t_upcross[i+1] - t_upcross[i]. One period estimate per wave cycle —
#   no window, no averaging, no smearing across the very evolution we want to see.
#
# Start with nowind runs — clean physics, no wind-wave interference.
# Second cell (below) overlays fullwind runs for comparison.

import re as _re2

# ── seiche reference constants ────────────────────────────────────────────────
_TANK_LENGTH_M   = 25.0      # approximate tank length [m]
_DEPTH_M         = 0.580     # water depth [m]
_C_SHALLOW       = float(np.sqrt(9.81 * _DEPTH_M))   # ~2.39 m/s
_T_SEICHE        = [2 * _TANK_LENGTH_M / _C_SHALLOW / n for n in (1, 2, 3)]  # [T1, T2, T3]

# ── upcrossing filter bounds ──────────────────────────────────────────────────
# Reject implausibly short or long "periods" to filter noise spikes and DC drift
_MIN_PERIOD_S    = 0.15    # shorter than this = noise crossing, not a real wave
_MAX_PERIOD_S    = 30.0    # longer than this = slow drift crossing, not seiche

# ── helper: zero-upcrossing period series ────────────────────────────────────
def _upcross_periods(sig, fs, min_period_s=_MIN_PERIOD_S, max_period_s=_MAX_PERIOD_S):
    """
    Return (t_uc, T_uc): time of each upcrossing and the period of that cycle.

    sig is expected to be already zeroed (eta_ column — zero = stillwater).
    t_uc[i]  = time [s] of the i-th zero-upcrossing
    T_uc[i]  = t_uc[i+1] - t_uc[i]  (period of cycle starting at upcrossing i)

    Crossings with T < min_period_s or T > max_period_s are excluded.
    Returns (array, array) — both empty if fewer than 2 valid crossings.
    """
    sig   = np.where(np.isnan(sig), 0.0, np.asarray(sig, dtype=float))
    cross = np.where((sig[:-1] < 0) & (sig[1:] >= 0))[0]
    if len(cross) < 2:
        return np.array([]), np.array([])
    t_uc = cross / fs
    T_uc = np.diff(t_uc)
    valid = (T_uc >= min_period_s) & (T_uc <= max_period_s)
    return t_uc[:-1][valid], T_uc[valid]

# ── select nowind per240 runs from 20260313 (cleanest long dataset) ──────────
_tail_meta_nw = combined_meta[
    (combined_meta["WindCondition"] == "no") &
    (combined_meta["path"].str.contains("20260313")) &
    (combined_meta["WaveFrequencyInput [Hz]"].notna()) &
    (combined_meta["run_category"] == "standard")
].sort_values("WaveFrequencyInput [Hz]").reset_index(drop=True)

print(f"Nowind tail-period runs (20260313): {len(_tail_meta_nw)}")
print(_tail_meta_nw[["WaveFrequencyInput [Hz]", "WaveAmplitudeInput [Volt]",
                      "PanelCondition"]].to_string())

# %% ── tail period tracking — compute ────────────────────────────────────────
# Ensure processed_dfs is loaded
if not processed_dfs:
    print("Loading processed_dfs (~75 MB, ~20 s)...")
    _t0 = time.perf_counter()
    processed_dfs = load_processed_dfs(*PROCESSED_DIRS)
    print(f"Loaded {len(processed_dfs)} runs in {time.perf_counter() - _t0:.1f} s")

# Use the IN probe — most exposed, clearest wave signal
_in_pos  = "9373/170"
_eta_col = f"eta_{_in_pos}"

_period_tracks = []   # list of dicts, one per run

for _, row in _tail_meta_nw.iterrows():
    path = row["path"]
    df   = processed_dfs.get(path)
    if df is None or _eta_col not in df.columns:
        continue

    freq_hz = float(row["WaveFrequencyInput [Hz]"])
    amp_v   = float(row["WaveAmplitudeInput [Volt]"])
    panel   = row.get("PanelCondition", "?")

    # good_end from pipeline (last clean wave sample index)
    _end_col = f"Computed Probe {_in_pos} end"
    good_end = row.get(_end_col)
    if good_end is None or pd.isna(good_end):
        m = _re2.search(r"mstop(\d+)", _Path(path).name)
        if m:
            good_end = max(0, len(df) - int(int(m.group(1)) * _FS))
        else:
            continue
    good_end = int(good_end)

    sig_full = df[_eta_col].values
    t_full   = np.arange(len(sig_full)) / _FS

    sig_tail = sig_full[good_end:]
    t_tail   = t_full[good_end:]    # absolute time stamps of the tail

    if len(sig_tail) < int(0.5 * _FS):   # need at least 0.5 s of tail
        continue

    # Per-cycle period: t of each upcrossing + T[i] = t[i+1]-t[i]
    # t_uc is relative to tail start (sample 0 of sig_tail)
    t_uc, T_uc = _upcross_periods(sig_tail, _FS)

    _period_tracks.append({
        "path":     path,
        "freq_hz":  freq_hz,
        "amp_v":    amp_v,
        "panel":    panel,
        "t_full":   t_full,
        "sig_full": sig_full,
        "good_end": good_end,
        # t_uc offset to absolute run time for consistent x-axis with timeseries
        "t_uc":     t_uc + t_tail[0],
        "T_uc":     T_uc,
        "paddle_T": 1.0 / freq_hz,
    })

print(f"Computed upcrossing period tracks for {len(_period_tracks)} runs")
for tr in _period_tracks:
    print(f"  {tr['freq_hz']:.2f} Hz  {int(tr['amp_v']*1000):d} mV  "
          f"{tr['panel']:8}  {len(tr['T_uc'])} cycles in tail")

# %% ── tail period tracking — plot ────────────────────────────────────────────
# Layout: one column per run, two rows:
#   Row 0 — full eta_ signal, good_end dashed line
#   Row 1 — per-cycle period (semilogy scatter), one dot per wave cycle
#            x-axis = time of upcrossing (absolute), y-axis = period of that cycle
#
# Reading the period plot:
#   • Dots start near 1/freq_hz (paddle period) immediately after good_end
#   • If seiche physics is right, dots climb toward T3~7s then T2~10s
#   • Gaps = no zero crossings in that interval (signal below noise or decayed)
#   • The climb should be monotonic (lower frequencies survive longer)

_n_runs = len(_period_tracks)
if _n_runs == 0:
    print("No tracks computed — check processed_dfs is loaded and 20260313 data exists")
else:
    fig, axes = plt.subplots(2, _n_runs,
                             figsize=(max(4, 3.5 * _n_runs), 7),
                             sharey="row")
    if _n_runs == 1:
        axes = axes[:, np.newaxis]

    _palette = plt.cm.viridis(np.linspace(0.15, 0.85, _n_runs))

    for ci, tr in enumerate(_period_tracks):
        ax_ts  = axes[0, ci]
        ax_per = axes[1, ci]
        col    = _palette[ci]

        # ── timeseries ─────────────────────────────────────────────────────
        ax_ts.plot(tr["t_full"], tr["sig_full"], lw=0.4, color="steelblue")
        ax_ts.axvline(tr["t_full"][tr["good_end"]], color="red",
                      lw=1.0, ls="--", label="good_end")
        ax_ts.set_title(
            f"{tr['freq_hz']:.2f} Hz  {int(tr['amp_v']*1000):d} mV  {tr['panel']}",
            fontsize=8
        )
        ax_ts.set_xlabel("t [s]", fontsize=7)
        if ci == 0:
            ax_ts.set_ylabel("η [mm]", fontsize=8)
        ax_ts.legend(fontsize=6, loc="upper right")
        ax_ts.grid(True, alpha=0.3)

        # ── per-cycle period track ─────────────────────────────────────────
        if len(tr["T_uc"]) > 0:
            ax_per.semilogy(tr["t_uc"], tr["T_uc"],
                            "o", ms=3, color=col, alpha=0.8)

        ax_per.axhline(tr["paddle_T"], color="grey", ls="--", lw=0.8,
                       label=f"paddle T={tr['paddle_T']:.2f} s")
        for k, Ts in enumerate(_T_SEICHE, 1):
            ax_per.axhline(Ts, color=f"C{k+1}", ls=":", lw=0.9,
                           label=f"seiche T{k}={Ts:.1f} s")

        ax_per.axvline(tr["t_full"][tr["good_end"]], color="red", lw=0.8, ls="--")
        ax_per.set_xlim(tr["t_full"][tr["good_end"]] - 2, tr["t_full"][-1] + 2)
        ax_per.set_ylim(_MIN_PERIOD_S * 0.8, _MAX_PERIOD_S * 1.2)
        ax_per.set_xlabel("t [s]", fontsize=7)
        if ci == 0:
            ax_per.set_ylabel("Cycle period [s]  (log)", fontsize=8)
        ax_per.legend(fontsize=6, loc="upper left")
        ax_per.grid(True, alpha=0.3, which="both")

    fig.suptitle(
        f"Tail period evolution — per zero-upcrossing cycle — nowind, 20260313, "
        f"IN probe ({_in_pos})\n"
        f"Each dot = one wave cycle.  "
        f"Seiche T₁={_T_SEICHE[0]:.1f} s  "
        f"(L={_TANK_LENGTH_M:.0f} m, d={_DEPTH_M*1000:.0f} mm)",
        fontsize=9
    )
    plt.tight_layout()
    plt.show()
    # What to look for:
    #   1. Do dots start near paddle_T and climb monotonically?
    #   2. Do they approach T3 (~7 s) or T2 (~10 s) within the tail window?
    #   3. Dot density = number of cycles per unit time → thins as period grows
    #   4. Higher-amplitude runs should have more dots (more energy to decay)

# %% ── tail period tracking — overlay wind vs nowind ─────────────────────────
# Compare fullwind vs nowind: does wind clear the tank faster?
# Use mstop90 runs from 20260307 (fullwind, 1.3 Hz) — longest available tail.
# For nowind comparison at 1.3 Hz: use matching runs from _period_tracks.
#
# NOTE: 20260307 nowind runs do NOT have mstop90 — only fullwind does.
# The 20260313 per240-mstop30 nowind runs give ~30 s of clean tail.

_mstop90_meta = combined_meta[
    (combined_meta["path"].str.contains("20260307")) &
    (combined_meta["WaveFrequencyInput [Hz]"].notna()) &
    (combined_meta["run_category"] == "standard")
].sort_values("WindCondition").reset_index(drop=True)
print(f"\nmstop90 runs (20260307): {len(_mstop90_meta)}")
print(_mstop90_meta[["WaveFrequencyInput [Hz]", "WaveAmplitudeInput [Volt]",
                      "WindCondition", "PanelCondition",
                      "path"]].to_string(index=False))

# %% ── wind vs nowind period track — compute and overlay ─────────────────────
_wind_tracks = []

for _, row in _mstop90_meta.iterrows():
    path = row["path"]
    df   = processed_dfs.get(path)
    if df is None or _eta_col not in df.columns:
        continue

    freq_hz = float(row["WaveFrequencyInput [Hz]"])
    wind    = row.get("WindCondition", "?")

    _end_col = f"Computed Probe {_in_pos} end"
    good_end = row.get(_end_col)
    if good_end is None or pd.isna(good_end):
        m = _re2.search(r"mstop(\d+)", _Path(path).name)
        if m:
            good_end = max(0, len(df) - int(int(m.group(1)) * _FS))
        else:
            continue
    good_end = int(good_end)

    sig_full = df[_eta_col].values
    t_full   = np.arange(len(sig_full)) / _FS
    sig_tail = sig_full[good_end:]
    t_tail   = t_full[good_end:]

    if len(sig_tail) < int(0.5 * _FS):
        continue

    t_uc, T_uc = _upcross_periods(sig_tail, _FS)

    _wind_tracks.append({
        "freq_hz":  freq_hz,
        "wind":     wind,
        "good_end": good_end,
        "t_full":   t_full,
        "sig_full": sig_full,
        "t_uc":     t_uc + t_tail[0],
        "T_uc":     T_uc,
        "paddle_T": 1.0 / freq_hz,
    })

# Pull matching nowind 1.3 Hz tracks from _period_tracks
_nw_1300 = [tr for tr in _period_tracks if abs(tr["freq_hz"] - 1.3) < 0.05]
_all_compare = _wind_tracks + _nw_1300

print(f"\nWind comparison tracks: {len(_all_compare)}")
for tr in _all_compare:
    wind_lbl = tr.get("wind", "no")
    n_cyc    = len(tr["T_uc"])
    t_span   = (tr["t_uc"][-1] - tr["t_full"][tr["good_end"]]) if n_cyc else 0
    print(f"  {tr['freq_hz']:.2f} Hz  {wind_lbl:8}  {n_cyc} cycles  tail ~{t_span:.0f} s")

# %% ── wind vs nowind — plot comparison ──────────────────────────────────────
if _all_compare:
    fig, (ax_ts, ax_per) = plt.subplots(2, 1, figsize=(12, 7), sharex=False)

    _wind_colors = {"full": "orangered", "lowest": "goldenrod", "no": "steelblue"}

    for tr in _all_compare:
        wind_lbl = tr.get("wind", "no")
        col      = _wind_colors.get(wind_lbl, "grey")
        t_offset = tr["t_full"][tr["good_end"]]   # re-zero to tail start

        ax_ts.plot(tr["t_full"] - t_offset, tr["sig_full"],
                   lw=0.5, color=col, alpha=0.6,
                   label=f"{tr['freq_hz']:.2f} Hz {wind_lbl}")
        ax_ts.axvline(0, color=col, ls="--", lw=0.7)

        if len(tr["T_uc"]) > 0:
            ax_per.semilogy(tr["t_uc"] - t_offset, tr["T_uc"],
                            "o", ms=3, alpha=0.75, color=col,
                            label=f"{wind_lbl} ({len(tr['T_uc'])} cycles)")

    for k, Ts in enumerate(_T_SEICHE, 1):
        ax_per.axhline(Ts, color=f"C{k+2}", ls=":", lw=1.0,
                       label=f"seiche T{k}={Ts:.1f} s")
    ax_per.axhline(1.0 / 1.3, color="grey", ls="--", lw=0.8,
                   label="paddle T (1.3 Hz)")

    ax_ts.set_xlabel("t since good_end [s]", fontsize=9)
    ax_ts.set_ylabel("η [mm]", fontsize=9)
    ax_ts.set_title("Tail signal — wind vs no-wind (re-zeroed to good_end)", fontsize=10)
    ax_ts.legend(fontsize=7)
    ax_ts.grid(True, alpha=0.3)

    ax_per.set_xlabel("t since good_end [s]", fontsize=9)
    ax_per.set_ylabel("Cycle period [s]  (log)", fontsize=9)
    ax_per.set_ylim(_MIN_PERIOD_S * 0.8, _MAX_PERIOD_S * 1.2)
    ax_per.set_title(
        f"Per-cycle period evolution — does wind clear the tank faster?\n"
        f"Seiche harmonics: T₁={_T_SEICHE[0]:.1f} s, T₂={_T_SEICHE[1]:.1f} s, "
        f"T₃={_T_SEICHE[2]:.1f} s  |  each dot = one wave cycle",
        fontsize=10
    )
    ax_per.legend(fontsize=7)
    ax_per.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.show()
    # What to look for:
    #   • Does fullwind curve reach longer periods EARLIER → faster seiche emergence?
    #   • Does fullwind have FEWER dots in tail → faster decay to below-noise floor?
    #   • Do both wind conditions converge to the same seiche period eventually?
    #   • Nowind tail is only ~30 s; fullwind is ~90 s — the extra 60 s of fullwind
    #     data past the nowind range is the most interesting for decay characterisation.
else:
    print("No comparison tracks — check that 20260307 data is in PROCESSED_DIRS and processed_dfs is loaded")

# %% ══════════════════════════════════════════════════════════════════════════
# STOKES WAVE ANALYSIS
# Theory: for a progressive wave of amplitude a, wavenumber k, depth d:
#   η(t) ≈ a₁ cos(ωt)                              (linear / 1st order)
#         + a₂ cos(2ωt)                             (2nd harmonic)
#         + a₃ cos(3ωt)                             (3rd harmonic)
#
# Stokes finite-depth 2nd-order (Dean & Dalrymple §2.4):
#   a₂ = a₁ · (ka₁/4) · cosh(kd)·(2+cosh(2kd)) / sinh³(kd)
#   → deep water limit: a₂/a₁ → ka₁/2
#
# Stokes 3rd-order (deep water, leading term):
#   a₃ = a₁ · (3/8)·(ka₁)²
#   → valid when kd >> 1; finite-depth 3rd order is more complex
#
# Ursell number: Ur = ka / (kd)³  (measures nonlinearity relative to depth)
#   Ur << 1: linear theory good   Ur ~ 1: Stokes 2nd/3rd order   Ur >> 1: KdV regime
#
# Depth regime (d = 580 mm):
#   kd < π/10 ≈ 0.31 : shallow   kd > π ≈ 3.14 : deep   else: intermediate
#   At 0.65 Hz: kd ≈ 1.2 (intermediate)   At 1.3 Hz: kd ≈ 3.9 (deep)
# ══════════════════════════════════════════════════════════════════════════════

# %% ── Stokes 1: dispersion relation + depth regime ───────────────────────────
from scipy.optimize import brentq as _brentq

_G_GRAVITY = 9.81    # m/s²  # TODO GET FROM constants.py
_TANK_DEPTH = 0.580  # m (from depth580 filenames) # TODO GET FROM constants.py

def _solve_k(freq_hz, d=_TANK_DEPTH, g=_G_GRAVITY):
    """Solve ω² = g·k·tanh(k·d) for k [m⁻¹] via Brent's method."""
    omega = 2 * np.pi * freq_hz
    k_deep = omega**2 / g   # deep-water starting guess (lower bound)
    def _residual(k): return g * k * np.tanh(k * d) - omega**2
    return _brentq(_residual, k_deep * 0.01, k_deep * 20.0)

_SHALLOW_KD = np.pi / 10   # kd < this → shallow water # TODO GET FROM constants.py
_DEEP_KD    = np.pi         # kd > this → deep water # TODO GET FROM constants.py

_disp_rows = []
for _f in sorted(combined_meta["WaveFrequencyInput [Hz]"].dropna().unique()):
    _k  = _solve_k(_f)
    _kd = _k * _TANK_DEPTH
    _L  = 2 * np.pi / _k
    _c  = _L * _f
    if _kd < _SHALLOW_KD:
        _regime = "SHALLOW"
    elif _kd > _DEEP_KD:
        _regime = "deep"
    else:
        _regime = "intermediate"
    _disp_rows.append({
        "f [Hz]": _f,
        "k [m⁻¹]": round(_k, 4),
        "kd": round(_kd, 3),
        "L [m]": round(_L, 3),
        "c [m/s]": round(_c, 3),
        "regime": _regime,
    })

_disp_df = pd.DataFrame(_disp_rows)
print("=== Dispersion relation at tank depth d =", _TANK_DEPTH, "m ===")
print(_disp_df.to_string(index=False))
print(f"\nShallow threshold:  kd < {_SHALLOW_KD:.3f} (L > {2*np.pi/(_SHALLOW_KD/_TANK_DEPTH):.2f} m)")
print(f"Deep threshold:     kd > {_DEEP_KD:.3f}    (L < {2*np.pi/(_DEEP_KD/_TANK_DEPTH):.2f} m)")

# Cross-check against pipeline Wavenumber (FFT) for IN probe
_k_cols = [c for c in combined_meta.columns if "Wavenumber (FFT)" in c]
if _k_cols:
    print(f"\n=== Pipeline wavenumber vs dispersion-solved k ===")
    _k_check = (
        combined_meta.dropna(subset=_k_cols[:1] + ["WaveFrequencyInput [Hz]"])
        .groupby("WaveFrequencyInput [Hz]")[_k_cols[0]]
        .median()
    )
    for _f, _k_pipe in _k_check.items():
        _k_theory = _solve_k(_f)
        print(f"  {_f:.2f} Hz:  pipeline={_k_pipe:.4f}  theory={_k_theory:.4f}  "
              f"diff={abs(_k_pipe - _k_theory)/_k_theory*100:.1f}%")

# %% ── Stokes 2: second harmonic — measured vs theory ────────────────────────
#
# For each wave run: extract FFT amplitude at f_paddle and 2·f_paddle.
# Compare measured a₂ to Stokes 2nd-order prediction.
#
# The complex FFT columns in combined_fft_dict allow reading amplitude
# at any frequency bin, not just the one the pipeline uses as the paddle peak.
#
# Stokes prediction: a₂_theory = a₁ · C₂(ka₁, kd)
# where C₂(ka, kd) = (ka/4) · cosh(kd)·(2+cosh(2kd)) / sinh³(kd)
#
# If measured a₂ ≈ a₂_theory → wave is Stokes-like (nonlinear but not breaking)
# If measured a₂ >> a₂_theory → additional nonlinearity, tank reflection, or beating
# If measured a₂ << a₂_theory → wave was weak / linear

def _stokes2_ratio(ka, kd):
    """a₂/a₁ for 2nd-order Stokes wave in finite depth."""
    sh = np.sinh(kd)
    return (ka / 4.0) * np.cosh(kd) * (2.0 + np.cosh(2.0 * kd)) / sh**3

def _stokes3_ratio_deep(ka):
    """a₃/a₁ for 3rd-order Stokes wave, deep-water limit."""
    return (3.0 / 8.0) * ka**2

def _fft_amplitude_at(fft_df, pos, target_hz, window_hz=0.15):
    """Return |FFT {pos}| at the positive-frequency bin nearest to target_hz.

    "FFT {pos}" stores |fft_vals|/N for ALL frequencies (positive and negative).
    For a sine of amplitude A: the positive-freq bin holds A/2 and the
    negative-freq bin also holds A/2.  The normalization cancels in any
    ratio between harmonics, so this function is correct for computing
    a₂/a₁, a₃/a₁, etc.  To get physical amplitude in mm, use
    "Probe {pos} Amplitude (FFT)" from combined_meta instead.
    """
    col = f"FFT {pos}"
    if col not in fft_df.columns:
        return np.nan
    _freqs  = fft_df.index.values.astype(float)
    # Positive frequencies only — avoid accidentally matching negative-freq mirror
    _pmask  = _freqs > 0
    _wmask  = np.abs(_freqs - target_hz) <= window_hz / 2
    _mask   = _pmask & _wmask
    if not _mask.any():
        return np.nan
    _vals   = fft_df.loc[_mask, col].values.astype(float)
    _fsub   = _freqs[_mask]
    return float(_vals[np.argmin(np.abs(_fsub - target_hz))])

_stokes_rows = []
for _, _row in filtered_meta.iterrows():
    _freq = _row.get("WaveFrequencyInput [Hz]")
    if not np.isfinite(float(_freq)):
        continue
    _path   = _row["path"]
    _fft_df = combined_fft_dict.get(_path)
    if _fft_df is None:
        continue
    _k  = _solve_k(_freq)
    _kd = _k * _TANK_DEPTH
    for _pos in ANALYSIS_PROBES:
        # Physical amplitude (mm) from pipeline — correctly calibrated
        _a1_mm = float(_row.get(f"Probe {_pos} Amplitude (FFT)", np.nan))
        if not np.isfinite(_a1_mm) or _a1_mm <= 0:
            continue
        # FFT-unit values for harmonic ratios (scale of |fft_vals|/N cancels in ratio)
        _a1_fft = _fft_amplitude_at(_fft_df, _pos, _freq)
        _a2_fft = _fft_amplitude_at(_fft_df, _pos, 2 * _freq)
        _a3_fft = _fft_amplitude_at(_fft_df, _pos, 3 * _freq)
        if not np.isfinite(_a1_fft) or _a1_fft <= 0:
            continue
        # ka uses physical amplitude (a₁ in m, k in m⁻¹)
        _ka  = _k * _a1_mm / 1000.0
        _Ur  = _ka / _kd**3
        # Harmonic ratios (dimensionless, scale-independent)
        _r2  = _a2_fft / _a1_fft if np.isfinite(_a2_fft) else np.nan
        _r3  = _a3_fft / _a1_fft if np.isfinite(_a3_fft) else np.nan
        # Stokes theoretical ratios (dimensionless)
        _r2_th = _stokes2_ratio(_ka, _kd)
        _r3_th = _stokes3_ratio_deep(_ka)
        _stokes_rows.append({
            "freq":            _freq,
            "amp_v":           _row.get("WaveAmplitudeInput [Volt]"),
            "wind":            _row.get("WindCondition"),
            "panel":           _row.get("PanelCondition"),
            "probe":           _pos,
            "ka":              _ka,
            "kd":              _kd,
            "Ursell":          _Ur,
            "a1_mm":           _a1_mm,
            "a2_over_a1":      _r2,                                    # measured ratio
            "a3_over_a1":      _r3,
            "a2_theory_ratio": _r2_th,                                 # Stokes prediction (ratio)
            "a3_theory_ratio": _r3_th,
            "a2_mm":           _a1_mm * _r2   if np.isfinite(_r2)   else np.nan,  # absolute mm
            "a2_theory_mm":    _a1_mm * _r2_th,
            "meas_over_theory":_r2 / _r2_th   if np.isfinite(_r2) and _r2_th > 0 else np.nan,
        })

_stokes_df = pd.DataFrame(_stokes_rows)
print(f"Stokes comparison rows: {len(_stokes_df)}")

# Summary: median a₂/a₁ measured vs theoretical at IN probe
_s2_agg = (
    _stokes_df[_stokes_df["probe"] == "9373/170"]
    .groupby(["freq", "wind"])
    .agg(
        ka_med   = ("ka",              "median"),
        kd       = ("kd",              "first"),
        Ursell   = ("Ursell",          "median"),
        a2_meas  = ("a2_over_a1",      "median"),
        a2_th    = ("a2_theory_ratio", "median"),
        ratio    = ("meas_over_theory","median"),
        n        = ("a1_mm",           "count"),
    )
    .reset_index()
)
print("\n=== a₂/a₁: measured vs Stokes theory at IN probe (9373/170) ===")
print(f"  {'f':5s}  {'wind':8s}  {'ka':7s}  {'kd':5s}  {'Ur':7s}  "
      f"{'a₂/a₁ meas':>12}  {'a₂/a₁ theory':>13}  {'meas/theory':>11}  n")
for _, r in _s2_agg.iterrows():
    print(f"  {r['freq']:5.2f}  {r['wind']:8s}  {r['ka_med']:7.4f}  {r['kd']:5.3f}  "
          f"{r['Ursell']:7.5f}  {r['a2_meas']:12.5f}  {r['a2_th']:13.5f}  "
          f"{r['ratio']:11.3f}  {int(r['n'])}")
print("\n  ratio > 1 → more 2nd-harmonic energy than Stokes predicts (reflection? beating? tank resonance?)")
print("  ratio < 1 → less than Stokes (dissipation, or deeper-water → theory overpredicts for kd > π)")

# %% ── Stokes 3: measured a₂ vs theory scatter ────────────────────────────────
# One point per run per probe. 1:1 line = perfect Stokes. Deviation = residual
# nonlinearity not captured by the finite-depth 2nd-order formula.
_wcm = {"no": "steelblue", "lowest": "goldenrod", "full": "tomato"}
_probes_stk = [p for p in ANALYSIS_PROBES if p in _stokes_df["probe"].values]
_ns = len(_probes_stk)
fig, axes = plt.subplots(1, _ns, figsize=(4.5 * _ns, 4.5), sharey=False)
if _ns == 1:
    axes = [axes]
for ax, _pos in zip(axes, _probes_stk):
    _sub = _stokes_df[_stokes_df["probe"] == _pos].dropna(subset=["a2_mm", "a2_theory_mm"])
    for _wc, _grp in _sub.groupby("wind"):
        ax.scatter(_grp["a2_theory_mm"], _grp["a2_mm"], s=20, alpha=0.5,
                   color=_wcm.get(_wc, "grey"), label=_wc)
    _lim = _sub[["a2_mm", "a2_theory_mm"]].max().max() * 1.1
    ax.plot([0, _lim], [0, _lim], "k--", lw=0.8, label="1:1 Stokes")
    ax.set_xlabel("a₂ predicted — Stokes 2nd-order finite-depth [mm]")
    ax.set_ylabel("a₂ measured — FFT at 2f [mm]" if ax is axes[0] else "")
    ax.set_title(_pos, fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
fig.suptitle(
    "2nd harmonic: measured vs Stokes 2nd-order (finite-depth)\n"
    "1:1 = perfect Stokes   above line = extra nonlinearity or reflections   "
    "below line = weaker than predicted",
    fontsize=9,
)
plt.tight_layout()
plt.show()

# Also: a₂_ratio vs Ursell (should show departure from Stokes at high Ursell)
fig, ax = plt.subplots(figsize=(7, 4))
for _pos, _grp in _stokes_df.groupby("probe"):
    _sub = _grp.dropna(subset=["meas_over_theory", "Ursell"])
    ax.scatter(_sub["Ursell"], _sub["meas_over_theory"], s=12, alpha=0.4,
               label=_pos)
ax.axhline(1.0, color="k", lw=0.8, ls="--", label="Stokes theory")
ax.set_xlabel("Ursell number  Ur = ka/(kd)³")
ax.set_ylabel("(a₂ measured) / (a₂ Stokes theory)")
ax.set_title("Departure from Stokes 2nd-order vs Ursell number", fontsize=9)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlim(left=0)
plt.tight_layout()
plt.show()

# %% ── Stokes 4: crest-trough asymmetry ───────────────────────────────────────
#
# Stokes waves are vertically asymmetric: crests are higher than troughs are deep.
# For a 2nd-order Stokes wave:
#   η_crest ≈ a₁ + a₂        (harmonics add at crest)
#   η_trough ≈ −a₁ + a₂      (signs alternate; a₂ reduces trough depth)
#   Asymmetry A = η_crest / |η_trough|  = (a₁+a₂) / (a₁−a₂)  ≈ 1 + 2·a₂/a₁
#
# Theory: A ≈ 1 + ka/2  (deep water)   →  A > 1 always for Stokes waves
# Linear wave: A = 1 exactly.
#
# Wind waves: broad spectrum, approximately symmetric (A ≈ 1).
# Under full wind at IN probe: signal is wind-dominated → asymmetry ≈ 1.
# Under no wind: asymmetry should grow with ka.
#
# Measurement: from time series, cycle-by-cycle crest max and trough min.
# Uses the stable wave window (middle 50% of run to avoid ramp + tail contamination).

if not processed_dfs:
    print("processed_dfs not loaded — run the lazy-load cell first")
else:
    _asym_rows = []
    for _, _row in filtered_meta[
            filtered_meta["WindCondition"] == "no"].iterrows():
        _df = processed_dfs.get(_row["path"])
        if _df is None:
            continue
        _freq = _row.get("WaveFrequencyInput [Hz]")
        if not np.isfinite(float(_freq)):
            continue
        _k  = _solve_k(_freq)
        for _pos in ANALYSIS_PROBES:
            _eta_col = f"eta_{_pos}"
            if _eta_col not in _df.columns:
                continue
            _sig = _df[_eta_col].values
            # Use middle 50% of run to dodge ramp and tail
            _s0 = len(_sig) // 4
            _s1 = 3 * len(_sig) // 4
            _win = _sig[_s0:_s1]
            _win = _win[~np.isnan(_win)]
            if len(_win) < 50:
                continue
            # Zero-upcrossing cycle extraction
            _upc = np.where((_win[:-1] < 0) & (_win[1:] >= 0))[0]
            if len(_upc) < 3:
                continue
            _T_samples = _FS / _freq          # expected samples per period
            _crests = []
            _troughs = []
            for _i in range(len(_upc) - 1):
                _c0, _c1 = _upc[_i], _upc[_i + 1]
                _cycle_len = _c1 - _c0
                if not (0.5 * _T_samples < _cycle_len < 2.0 * _T_samples):
                    continue
                _chunk = _win[_c0:_c1]
                _crests.append(_chunk.max())
                _troughs.append(_chunk.min())
            if len(_crests) < 3:
                continue
            _a1 = _row.get(f"Probe {_pos} Amplitude (FFT)", np.nan)
            _ka = _k * float(_a1) / 1000 if np.isfinite(float(_a1)) and float(_a1) > 0 else np.nan
            _asym_rows.append({
                "freq":    _freq,
                "amp_v":   _row.get("WaveAmplitudeInput [Volt]"),
                "probe":   _pos,
                "ka":      _ka,
                "a1_mm":   float(_a1) if np.isfinite(float(_a1)) else np.nan,
                "crest_median": np.median(_crests),
                "trough_median": abs(np.median(_troughs)),
                "n_cycles": len(_crests),
                "asymmetry": np.median(_crests) / abs(np.median(_troughs)),
            })

    _asym_df = pd.DataFrame(_asym_rows)
    print(f"Crest-trough asymmetry rows: {len(_asym_df)}")

    if not _asym_df.empty:
        # Theory line: A_theory = 1 + ka  (approx, from a₂/a₁ ≈ ka/2 and A ≈ 1 + 2a₂/a₁)
        _ka_theory = np.linspace(0, _asym_df["ka"].max() * 1.1, 200)
        _A_theory  = 1 + _ka_theory   # deep-water approximation

        _probe_colors = dict(zip(ANALYSIS_PROBES, plt.cm.tab10(np.linspace(0, 0.9, len(ANALYSIS_PROBES)))))
        fig, ax = plt.subplots(figsize=(8, 5))
        for _pos, _grp in _asym_df.dropna(subset=["ka", "asymmetry"]).groupby("probe"):
            ax.scatter(_grp["ka"], _grp["asymmetry"], s=20, alpha=0.5,
                       color=_probe_colors.get(_pos, "grey"), label=_pos)
        ax.plot(_ka_theory, _A_theory, "k--", lw=1.2,
                label="Stokes 2nd-order: A ≈ 1 + ka  (deep water)")
        ax.axhline(1.0, color="0.7", lw=0.6, ls=":")
        ax.set_xlabel("ka  (wave steepness)")
        ax.set_ylabel("Crest / |trough|  (asymmetry ratio)")
        ax.set_title(
            "Wave crest-trough asymmetry vs ka — no-wind runs\n"
            "Stokes: A > 1 (crests pointier, troughs flatter).  Linear: A = 1.",
            fontsize=9
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        print("\n=== Asymmetry summary per frequency / probe ===")
        _asym_agg = (
            _asym_df.groupby(["freq", "probe"])
            .agg(ka_med=("ka","median"), A_med=("asymmetry","median"),
                 A_std=("asymmetry","std"), n=("n_cycles","sum"))
            .reset_index()
        )
        print(_asym_agg.round(4).to_string(index=False))

# %% ══════════════════════════════════════════════════════════════════════════
# RUN SIMILARITY ANALYSIS — amp0100-freq1300
#
# "How similar are repeated runs at the same nominal conditions?"
# The answer has multiple layers:
#
# (a) WAVEMAKER REPEATABILITY — does the paddle deliver the same wave amplitude
#     and shape every time? → compare A_FFT and a₂/a₁ across all runs.
#
# (b) "TOO CLOSE" EFFECT — insufficient stillwater recovery between runs biases
#     the next run: residual sloshing rides on top of the new paddle wave,
#     distorting amplitude, phase coherence, and harmonic content.
#     Diagnostic: A_FFT and wave_stability vs inter_run_gap_s.
#
# (c) "COLD START" — first wave run of the day (or after a long idle period).
#     The tank is fully settled but the water temperature and wavemaker may
#     behave differently. Diagnostic: first-run-of-day vs subsequent.
#
# (d) RECORDING LENGTH — per40 (40 wave periods ≈ 31 s at 1.3 Hz) vs per240
#     (240 periods ≈ 185 s). A longer run captures more of the wave field and
#     gives a more stable spectral average, but may also sample a time-varying
#     process (wavemaker drift, tank warming). The FFT amplitude should converge
#     to the same value if the run is stationary; if not → something evolves.
# ══════════════════════════════════════════════════════════════════════════════

# %% ── Similarity 1: data inventory for amp0100-freq1300 ─────────────────────
_SIM_FREQ = 1.3   # Hz
_SIM_AMP  = 0.1   # V

_sim_all = combined_meta[
    (combined_meta["WaveFrequencyInput [Hz]"] == _SIM_FREQ) &
    (combined_meta["WaveAmplitudeInput [Volt]"] == _SIM_AMP)
].copy().sort_values("file_date").reset_index(drop=True)

print(f"=== All runs at {_SIM_FREQ} Hz / {_SIM_AMP} V ===")
print(f"Total: {len(_sim_all)} runs")
print(f"\nBreakdown:")
print(_sim_all.groupby(["PanelCondition", "WindCondition"]).size().rename("n").reset_index().to_string(index=False))
if "WavePeriodInput" in _sim_all.columns:
    print(f"\nBy WavePeriodInput (recording length):")
    print(_sim_all["WavePeriodInput"].value_counts().sort_index().to_string())
print(f"\nDate range: {_sim_all['file_date'].min()} → {_sim_all['file_date'].max()}")

# %% ── Similarity 2: amplitude variability across runs ────────────────────────
# For each run, compare A_FFT at IN probe (9373/170) across all experimental variables.
# Shows wavemaker repeatability and condition-driven differences.

_SIM_IN   = "9373/170"
_SIM_OUT  = "12400/250"
_fft_in   = f"Probe {_SIM_IN} Amplitude (FFT)"
_fft_out  = f"Probe {_SIM_OUT} Amplitude (FFT)"

_sim_plot = _sim_all.dropna(subset=[_fft_in]).copy()
_sim_plot["run_index"] = np.arange(len(_sim_plot))

# ── amplitude vs run index, coloured by condition ─────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
_cmap_wind  = {"no": "steelblue", "lowest": "goldenrod", "full": "tomato"}
_cmap_panel = {"no": "steelblue", "full": "seagreen", "reverse": "darkorange"}

for ax, (_col, _cmap, _title) in zip(axes, [
    (_fft_in,  _cmap_wind,  f"IN probe ({_SIM_IN}) A_FFT — coloured by WindCondition"),
    (_fft_in,  _cmap_panel, f"IN probe ({_SIM_IN}) A_FFT — coloured by PanelCondition"),
    (_fft_out, _cmap_wind,  f"OUT probe ({_SIM_OUT}) A_FFT — coloured by WindCondition"),
]):
    _cond_col = "WindCondition" if "Wind" in _title else "PanelCondition"
    for _cond, _grp in _sim_plot.groupby(_cond_col):
        ax.scatter(_grp["run_index"], _grp[_col], s=25, alpha=0.7,
                   color=_cmap.get(_cond, "grey"), label=_cond)
    ax.set_ylabel("A_FFT [mm]")
    ax.set_title(_title, fontsize=9)
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel(f"Run index (sorted by date, n={len(_sim_plot)})")
fig.suptitle(
    f"Amplitude variability — {_SIM_FREQ} Hz, {_SIM_AMP} V\n"
    "Each point = one run. Spread within a condition = wavemaker variability + tank state.",
    fontsize=9,
)
plt.tight_layout()
plt.show()

# Print CV for each condition
print("=== A_FFT CV (std/mean) per condition — wavemaker repeatability ===")
for _cond, _grp in _sim_plot.groupby(["WindCondition", "PanelCondition"]):
    _v = _grp[_fft_in].dropna()
    if len(_v) < 2:
        continue
    print(f"  wind={_cond[0]:8s}  panel={_cond[1]:8s}  "
          f"n={len(_v):3d}  mean={_v.mean():.3f}  std={_v.std():.3f}  "
          f"CV={_v.std()/_v.mean():.3f}  [mm]")

# %% ── Similarity 3: "too close" effect — inter_run_gap_s ────────────────────
# Does wave amplitude or stability correlate with time since the previous run ended?
# Short gap → residual sloshing may contaminate the new run.
#
"""Notes todo
so the too close effect must take in to account what the previous run was. long-wave runs create way more sloshing.
"""

if "inter_run_gap_s" in _sim_plot.columns:
    _gap = _sim_plot.dropna(subset=["inter_run_gap_s", _fft_in, "wave_stability"]).copy() \
        if "wave_stability" in _sim_plot.columns else \
        _sim_plot.dropna(subset=["inter_run_gap_s", _fft_in]).copy()
    _gap = _gap[_gap["WindCondition"] == "no"]   # nowind only — cleaner signal

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    _cmap_period = {}
    if "WavePeriodInput" in _gap.columns:
        _periods = sorted(_gap["WavePeriodInput"].dropna().unique())
        _period_colors = {p: c for p, c in zip(_periods, plt.cm.Set1(np.linspace(0, 0.8, len(_periods))))}
        for _p, _grp in _gap.groupby("WavePeriodInput"):
            axes[0].scatter(_grp["inter_run_gap_s"], _grp[_fft_in],
                            s=25, alpha=0.6, color=_period_colors.get(_p, "grey"), label=f"per{int(_p)}")
    else:
        axes[0].scatter(_gap["inter_run_gap_s"], _gap[_fft_in], s=25, alpha=0.6, color="steelblue")

    axes[0].set_xlabel("inter_run_gap_s [s]  (time since previous run ended)")
    axes[0].set_ylabel(f"A_FFT at IN probe [mm]")
    axes[0].set_title("'Too close' effect: A_FFT vs gap to previous run", fontsize=9)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    if "wave_stability" in _gap.columns:
        for _p, _grp in _gap.groupby("WavePeriodInput") if "WavePeriodInput" in _gap.columns \
                else [(None, _gap)]:
            _c = _period_colors.get(_p, "steelblue") if _p is not None else "steelblue"
            _lbl = f"per{int(_p)}" if _p is not None else ""
            axes[1].scatter(_grp["inter_run_gap_s"], _grp["wave_stability"],
                            s=25, alpha=0.6, color=_c, label=_lbl)
        axes[1].set_xlabel("inter_run_gap_s [s]")
        axes[1].set_ylabel("wave_stability (1 = perfectly coherent)")
        axes[1].set_title("Wave coherence vs gap — residual sloshing disrupts phase?", fontsize=9)
        axes[1].set_ylim(0, 1.05)
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        f"'Too close' effect — {_SIM_FREQ} Hz, {_SIM_AMP} V, no-wind only\n"
        "Short gap → previous run's sloshing not fully decayed → biased amplitude/stability",
        fontsize=9,
    )
    plt.tight_layout()
    plt.show()

    # Also: flag by prev_run_category and prev_run_wind
    if "prev_run_category" in _gap.columns:
        print("=== A_FFT by prev_run_category ===")
        _prev_agg = _gap.groupby("prev_run_category")[_fft_in].agg(
            median="median", std="std", n="count")
        print(_prev_agg.round(3).to_string())

# %% ── Similarity 4: "cold start" — first run of day vs later ────────────────
# First run of the day: tank has rested overnight → stillwater, but wavemaker
# cold, water may be at different temperature, sloshing from previous day gone.
# Hypothesis: cold start runs may have slightly different amplitude (wavemaker
# hydraulics take a few runs to warm up and reach steady state).
#
"""notes, todo
... so the cold starts needs to account for periods, as the per240 will, in my estimation (but, really lets quantify this too)
have a more stable amplitude.
""""

if "file_date" in _sim_all.columns:
    _sim_dated = _sim_all.dropna(subset=[_fft_in]).copy()
    _sim_dated["file_date_day"] = pd.to_datetime(
        _sim_dated["file_date"].astype(str).str[:10], errors="coerce"
    )
    # "Cold start" = first wave run at this freq+amp on this day
    _sim_dated = _sim_dated.sort_values(["file_date_day", "file_date"])
    _sim_dated["run_num_today"] = (
        _sim_dated.groupby("file_date_day").cumcount()
    )
    _sim_dated["is_cold_start"] = _sim_dated["run_num_today"] == 0

    _nowind_dated = _sim_dated[_sim_dated["WindCondition"] == "no"]
    if len(_nowind_dated) > 0:
        fig, ax = plt.subplots(figsize=(9, 4))
        for _cold, _grp in _nowind_dated.groupby("is_cold_start"):
            _label = "cold start (1st run today)" if _cold else "subsequent runs"
            _c     = "tomato" if _cold else "steelblue"
            ax.scatter(_grp["file_date_day"].astype(str), _grp[_fft_in],
                       s=30, alpha=0.7, color=_c, label=_label)
        ax.set_xlabel("Date")
        ax.set_ylabel("A_FFT at IN probe [mm]")
        ax.tick_params(axis="x", rotation=45)
        ax.set_title(
            f"Cold-start effect — {_SIM_FREQ} Hz, {_SIM_AMP} V, no wind\n"
            "Red = first wave run of the day; blue = subsequent runs on same day",
            fontsize=9,
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        _cold_v = _nowind_dated[_nowind_dated["is_cold_start"]][_fft_in].dropna()
        _warm_v = _nowind_dated[~_nowind_dated["is_cold_start"]][_fft_in].dropna()
        print(f"Cold start:  n={len(_cold_v)}  median={_cold_v.median():.3f}  std={_cold_v.std():.3f} mm")
        print(f"Subsequent:  n={len(_warm_v)}  median={_warm_v.median():.3f}  std={_warm_v.std():.3f} mm")
        if len(_cold_v) > 0 and len(_warm_v) > 0:
            _diff = _cold_v.median() - _warm_v.median()
            print(f"Cold − warm: {_diff:+.3f} mm  ({_diff/_warm_v.median()*100:+.1f}%)")

# %% ── Similarity 5: recording length — per40 vs per240 ──────────────────────
# Does the measurement depend on how many wave periods were recorded?
# Both should give the same A_FFT if the wave is stationary.
# Deviations suggest: (a) the stable window selection is different, or (b)
# the wave is not stationary across 240 periods (wavemaker drift, tank warming).
# Also compare spectral shape: do shorter runs catch the ramp, inflating harmonics?
#
"""
todo:
    - second plot not showing.
    - first plot needs more distinct colors.
"""

if "WavePeriodInput" in _sim_all.columns:
    _per_nowind = _sim_all[
        (_sim_all["WindCondition"] == "no") &
        (_sim_all["WavePeriodInput"].notna())
    ].dropna(subset=[_fft_in]).copy()

    _per_groups = sorted(_per_nowind["WavePeriodInput"].unique())
    print(f"\n=== Recording length comparison — {_SIM_FREQ} Hz, {_SIM_AMP} V, no wind ===")
    print(f"{'WavePeriodInput':>16}  {'n':>4}  {'A_FFT_IN median':>16}  "
          f"{'std':>6}  {'CV':>6}  {'stability_med':>14}")
    _stab_col = "wave_stability"
    for _p in _per_groups:
        _g = _per_nowind[_per_nowind["WavePeriodInput"] == _p]
        _v = _g[_fft_in].dropna()
        _s = _g[_stab_col].dropna() if _stab_col in _g.columns else pd.Series(dtype=float)
        if _v.empty:
            continue
        _stab_str = f"{_s.median():.3f}" if not _s.empty else "n/a"
        print(f"  per{int(_p):3d}  ({_p*1/_SIM_FREQ:5.0f} s)  "
              f"{len(_v):4d}  {_v.median():16.3f}  {_v.std():6.3f}  "
              f"{_v.std()/_v.median():6.3f}  {_stab_str}")

    if len(_per_groups) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        _period_colors2 = {p: c for p, c in zip(
            _per_groups, plt.cm.Set1(np.linspace(0, 0.8, len(_per_groups))))}
        for _p, _grp in _per_nowind.groupby("WavePeriodInput"):
            _c = _period_colors2.get(_p, "grey")
            _lbl = f"per{int(_p)}  ({_p/_SIM_FREQ:.0f} s)"
            axes[0].scatter(_grp["PanelCondition"].map(
                {"no": 0, "full": 1, "reverse": 2}).fillna(-1),
                _grp[_fft_in], s=25, alpha=0.6, color=_c, label=_lbl)
        axes[0].set_xticks([0, 1, 2])
        axes[0].set_xticklabels(["no panel", "full panel", "reverse"])
        axes[0].set_ylabel("A_FFT at IN probe [mm]")
        axes[0].set_title("A_FFT by panel condition", fontsize=9)
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        if _stab_col in _per_nowind.columns:
            for _p, _grp in _per_nowind.groupby("WavePeriodInput"):
                _c = _period_colors2.get(_p, "grey")
                axes[1].scatter(_grp[_fft_in], _grp[_stab_col],
                                s=25, alpha=0.6, color=_c,
                                label=f"per{int(_p)}")
            axes[1].set_xlabel("A_FFT at IN probe [mm]")
            axes[1].set_ylabel("wave_stability")
            axes[1].set_title("Stability vs amplitude — by recording length", fontsize=9)
            axes[1].legend(fontsize=8)
            axes[1].grid(True, alpha=0.3)

        fig.suptitle(
            f"Recording length effect — {_SIM_FREQ} Hz, {_SIM_AMP} V, no wind\n"
            "If per40 ≈ per240: wave is stationary and FFT amplitude is robust.  "
            "Divergence = non-stationarity (ramp contamination, wavemaker drift).",
            fontsize=9,
        )
        plt.tight_layout()
        plt.show()

# %% ── Similarity 6: spectral shape comparison ────────────────────────────────
# For the most controlled subset (nopanel, nowind), compare normalized FFT spectra
# across all per240 runs. Are the spectral shapes consistent, or do harmonic
# ratios vary run-to-run?
# Normalized by a₁ so runs with slightly different amplitudes can be compared.
# If shapes are consistent: wavemaker is repeatable in spectral character.
# If 2nd harmonic varies: interaction with panel/tank sloshing differs.

_SIM_CTRL = _sim_all[
    (_sim_all["WindCondition"] == "no") &
    (_sim_all["PanelCondition"] == "no")
].copy()
if "WavePeriodInput" in _SIM_CTRL.columns:
    _long_mask = _SIM_CTRL["WavePeriodInput"] == _SIM_CTRL["WavePeriodInput"].max()
    _SIM_CTRL  = _SIM_CTRL[_long_mask]

print(f"Spectral shape comparison — {_SIM_FREQ} Hz, {_SIM_AMP} V, nopanel, nowind: "
      f"{len(_SIM_CTRL)} runs")

_spec_to_plot = []
for _, _row in _SIM_CTRL.iterrows():
    _fft_df = combined_fft_dict.get(_row["path"])
    if _fft_df is None:
        continue
    _col = f"FFT {_SIM_IN} complex"
    if _col not in _fft_df.columns:
        _col = f"FFT {_SIM_IN}"
        if _col not in _fft_df.columns:
            continue
    _freqs = _fft_df.index.values.astype(float)
    _amp   = np.abs(_fft_df[_col].values) if np.iscomplexobj(_fft_df[_col].values) \
             else _fft_df[_col].values.astype(float)
    # Fundamental amplitude for normalization
    _f_mask = np.abs(_freqs - _SIM_FREQ) < 0.1
    _a1_norm = _amp[_f_mask].max() if _f_mask.any() else 1.0
    if _a1_norm <= 0:
        continue
    _spec_to_plot.append({
        "run":    _Path(_row["path"]).name,
        "date":   str(_row.get("file_date", ""))[:10],
        "freqs":  _freqs,
        "amp_norm": _amp / _a1_norm,   # normalized by fundamental
        "a1":     _a1_norm,
    })

if _spec_to_plot:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    _colors = plt.cm.viridis(np.linspace(0, 0.9, len(_spec_to_plot)))

    for _sp, _c in zip(_spec_to_plot, _colors):
        # Full spectrum up to 5 Hz
        _m = _sp["freqs"] <= 5.5
        axes[0].semilogy(_sp["freqs"][_m], np.maximum(_sp["amp_norm"][_m], 1e-6),
                         lw=0.7, alpha=0.6, color=_c, label=_sp["date"])
        # Zoom: harmonics 1–4
        _m2 = _sp["freqs"] <= 4.5
        axes[1].semilogy(_sp["freqs"][_m2], np.maximum(_sp["amp_norm"][_m2], 1e-4),
                         lw=0.7, alpha=0.6, color=_c)

    for ax in axes:
        for _n in range(1, 5):
            ax.axvline(_n * _SIM_FREQ, color="tomato", lw=0.6, ls="--", alpha=0.5)
        ax.set_xlabel("Frequency [Hz]")
        ax.grid(True, alpha=0.2, which="both")
    axes[0].set_ylabel("Normalized amplitude (a₁ = 1)")
    axes[0].legend(fontsize=6, ncol=2, loc="upper right")
    axes[0].set_title("Full spectrum (normalized by a₁)", fontsize=9)
    axes[1].set_title("Harmonic zoom — consistency of a₂/a₁, a₃/a₁", fontsize=9)
    fig.suptitle(
        f"Spectral shape repeatability — {_SIM_FREQ} Hz, {_SIM_AMP} V, nopanel, nowind\n"
        f"Red dashes = harmonics at {_SIM_FREQ}, {2*_SIM_FREQ}, {3*_SIM_FREQ}, {4*_SIM_FREQ} Hz\n"
        "Overlapping curves = wavemaker repeatable.  Spread = run-to-run variability.",
        fontsize=9,
    )
    plt.tight_layout()
    plt.show()

    _a2_vals = [
        _fft_amplitude_at(combined_fft_dict[_row["path"]], _SIM_IN, 2 * _SIM_FREQ)
        / max(_fft_amplitude_at(combined_fft_dict[_row["path"]], _SIM_IN, _SIM_FREQ), 1e-9)
        for _, _row in _SIM_CTRL.iterrows()
        if combined_fft_dict.get(_row["path"]) is not None
    ]
    _a2_vals = [v for v in _a2_vals if np.isfinite(v)]
    if _a2_vals:
        _k_ctrl  = _solve_k(_SIM_FREQ)
        # Physical amplitude in mm — from pipeline's calibrated column

        _a1_ctrl_mm = _SIM_CTRL[f"Probe {_SIM_IN} Amplitude (FFT)"].dropna().median()
        _ka_ctrl = _k_ctrl * _a1_ctrl_mm / 1000.0   # k [m⁻¹] × a [m]
        _a2_stk  = _stokes2_ratio(_ka_ctrl, _k_ctrl * _TANK_DEPTH)
        print(f"\n  a₂/a₁ across {len(_a2_vals)} control runs:")
        print(f"    median = {np.median(_a2_vals):.5f}  std = {np.std(_a2_vals):.5f}  "
              f"CV = {np.std(_a2_vals)/np.median(_a2_vals):.3f}")
        print(f"    a₁ = {_a1_ctrl_mm:.3f} mm  ka = {_ka_ctrl:.4f}  kd = {_k_ctrl*_TANK_DEPTH:.3f}")
        print(f"    Stokes 2nd-order theory:  a₂/a₁ = {_a2_stk:.5f}")
        print(f"    measured/theory = {np.median(_a2_vals)/_a2_stk:.2f}×")

# %% ══════════════════════════════════════════════════════════════════════════
# REFLECTION ANALYSIS
#
# The panel is not perfectly transparent: some incident wave energy reflects
# back toward the paddle, creating a partial standing wave between paddle and
# panel.  Reflected energy appears at the IN-side probes (8804 and 9373) as an
# amplitude modulation that varies with frequency:
#
#   A_measured(x) = sqrt( A_i² + A_r² + 2·A_i·A_r·cos(2kx + φ) )
#
# where x is measured from the panel toward the paddle (so the probe closest
# to the panel, 9373, has the smallest x).
#
# Two approaches used here:
#
# (1) AMPLITUDE RATIO — compare A_FFT with/without panel at each probe.
#     A_panel/A_nopanel > 1  →  probe near an ANTINODE  →  lower bound on R
#     A_panel/A_nopanel < 1  →  probe near a NODE       →  different bound
#     The ratio oscillates with frequency as k changes and the node/antinode
#     pattern sweeps past the probe position.
#
# (2) TWO-PROBE COMPLEX METHOD — use the phase-coherent complex FFTs at both
#     8804/250 and 9373/170 recorded in the same run (same time origin) to
#     solve simultaneously for A_i and A_r:
#
#         F_A = a·e^{ikx_A} + b·e^{-ikx_A}    (probe A at x_A = 8.804 m)
#         F_B = a·e^{ikx_B} + b·e^{-ikx_B}    (probe B at x_B = 9.373 m)
#
#     → a = (F_B·e^{-ikx_A} - F_A·e^{-ikx_B}) / (2i·sin(kΔx))
#     → b = (F_A·e^{ikx_B}  - F_B·e^{ikx_A})  / (2i·sin(kΔx))
#     → R = |b| / |a|
#
#     ill-conditioned when sin(k·Δx) ≈ 0  (Δx = 0.569 m → nπ at k ≈ nπ/0.569)
#
# Physical note: 8804 and 9373 differ in LATERAL position (250 vs 170 mm).
# The two-probe formula assumes 1D plane waves.  Lateral non-uniformity adds
# noise but should not cause systematic bias for a narrow-banded paddle wave.
#
# The reflection coefficient is measured at the probe array, not the panel
# face.  Propagation loss (tiny in a wave tank) is not corrected for.
# ══════════════════════════════════════════════════════════════════════════════

# %% ── Reflection 0: data inventory ──────────────────────────────────────────
# What nopanel / fullpanel / reversepanel nowind data is loaded?
# If nopanel data is sparse → check which PROCESSED_DIRS folders are commented
# out above (20260307 and earlier contain the main nopanel dataset).

_REFL_AMP   = 0.1   # V — use lowest amplitude for cleanest standing-wave signal
_REFL_UP    = "8804/250"   # upstream probe
_REFL_IN    = "9373/170"   # IN probe (closer to panel)
_REFL_X_UP  = 8.804        # m — longitudinal position
_REFL_X_IN  = 9.373        # m
_REFL_DX    = _REFL_X_IN - _REFL_X_UP  # = 0.569 m — probe separation

_refl_wave = combined_meta[
    (combined_meta["WaveAmplitudeInput [Volt]"] == _REFL_AMP) &
    (combined_meta["WindCondition"] == "no")
].copy()

print("=== Reflection analysis — data inventory ===")
print(f"Amplitude filter: {_REFL_AMP} V,  WindCondition: no")
print(f"\nPanelCondition × WaveFrequencyInput [Hz]  (run counts):\n")

_refl_counts = (
    _refl_wave.groupby(["PanelCondition", "WaveFrequencyInput [Hz]"])
    .size()
    .unstack("WaveFrequencyInput [Hz]")
    .fillna(0).astype(int)
)
print(_refl_counts.to_string())

_panel_types = sorted(_refl_wave["PanelCondition"].dropna().unique())
print(f"\nPanelCondition values present: {_panel_types}")
print(f"Frequencies with nopanel data: "
      f"{sorted(_refl_wave[_refl_wave['PanelCondition']=='no']['WaveFrequencyInput [Hz]'].dropna().unique())}")

# Probe data availability per panel condition
print(f"\nProbe amplitude column availability (median non-NaN count per panel condition):")
for _pc in _panel_types:
    _sub = _refl_wave[_refl_wave["PanelCondition"] == _pc]
    for _pr in [_REFL_UP, _REFL_IN]:
        _col = f"Probe {_pr} Amplitude (FFT)"
        if _col in _sub.columns:
            _n_valid = _sub[_col].notna().sum()
            print(f"  {_pc:12s}  {_pr}:  {_n_valid} valid rows out of {len(_sub)}")

if "no" not in _panel_types:
    print("\n⚠  NO nopanel runs in current PROCESSED_DIRS.")
    print("   The nopanel control data is likely in the commented-out 20260307 folder.")
    print("   To enable, uncomment the line:")
    print("   # Path('waveprocessed/PROCESSED-20260307-ProbPos4_31_FPV_2-tett6roof')")
    print("   in the PROCESSED_DIRS list near the top of this file.")
else:
    _n_nopanel = len(_refl_wave[_refl_wave["PanelCondition"] == "no"])
    print(f"\n✓  {_n_nopanel} nopanel runs available. Proceeding with reflection analysis.")

# %% ── Reflection 1: amplitude ratio — panel vs nopanel at both upstream probes
# For each frequency: median A_FFT with panel / without panel.
# ratio > 1 → probe is near an antinode of the reflected standing wave
# ratio < 1 → probe is near a node

_refl_nowind = _refl_wave.copy()  # already filtered to WindCondition == "no"

_r1_rows = []
for _f in sorted(_refl_nowind["WaveFrequencyInput [Hz]"].dropna().unique()):
    _fsub = _refl_nowind[_refl_nowind["WaveFrequencyInput [Hz]"] == _f]
    _row  = {"f [Hz]": _f}
    for _pr in [_REFL_IN, _REFL_UP]:
        _col = f"Probe {_pr} Amplitude (FFT)"
        if _col not in _fsub.columns:
            continue
        for _pc in ["no", "full", "reverse"]:
            _vals = _fsub.loc[_fsub["PanelCondition"] == _pc, _col].dropna()
            _key  = f"A_{_pc[:3]}_{_pr.replace('/', '_')}"
            _row[_key] = float(_vals.median()) if len(_vals) else np.nan
    # Ratios: panel / nopanel
    for _pr in [_REFL_IN, _REFL_UP]:
        _p = _pr.replace("/", "_")
        _a_no  = _row.get(f"A_no_{_p}",  np.nan)
        _a_ful = _row.get(f"A_ful_{_p}", np.nan)
        _a_rev = _row.get(f"A_rev_{_p}", np.nan)
        if np.isfinite(_a_no) and _a_no > 0:
            _row[f"ratio_full_{_p}"]    = _a_ful / _a_no if np.isfinite(_a_ful) else np.nan
            _row[f"ratio_reverse_{_p}"] = _a_rev / _a_no if np.isfinite(_a_rev) else np.nan
    _r1_rows.append(_row)

_r1_df = pd.DataFrame(_r1_rows)

print("=== Amplitude ratio: A_panel / A_nopanel  (nowind, 0.1 V) ===")
_print_cols = ["f [Hz]"] + [c for c in _r1_df.columns if "ratio" in c]
if len(_print_cols) > 1:
    print(_r1_df[_print_cols].round(4).to_string(index=False))
    print("\nInterpretation:  ratio > 1.0 → probe near antinode (constructive with reflected wave)")
    print("                 ratio < 1.0 → probe near node     (destructive)")
    print("                 ratio = 1.0 → no reflection, or probe exactly between node and antinode")
else:
    print("No nopanel data available for ratio computation — check data inventory above.")

# ── Plot ──────────────────────────────────────────────────────────────────
_ratio_cols_in  = [c for c in _r1_df.columns if "ratio" in c and _REFL_IN.replace("/","_") in c]
_ratio_cols_up  = [c for c in _r1_df.columns if "ratio" in c and _REFL_UP.replace("/","_") in c]

if _ratio_cols_in or _ratio_cols_up:
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, sharey=True)
    _cmap_pc = {"full": "seagreen", "reverse": "darkorange"}
    _freqs_plot = _r1_df["f [Hz]"].values

    for _ax, _rcols, _pr_label in zip(
        axes,
        [_ratio_cols_in, _ratio_cols_up],
        [f"IN probe  ({_REFL_IN})", f"Upstream probe  ({_REFL_UP})"],
    ):
        for _rc in _rcols:
            _pc_key = "full" if "full" in _rc else "reverse"
            _vals = _r1_df[_rc].values
            _mask = np.isfinite(_vals)
            if _mask.any():
                _ax.plot(_freqs_plot[_mask], _vals[_mask], "o-",
                         color=_cmap_pc[_pc_key], ms=5, lw=1.4,
                         label=f"{_pc_key}panel / nopanel")
        _ax.axhline(1.0, color="0.5", lw=0.8, ls="--")
        _ax.set_ylabel("A_panel / A_nopanel")
        _ax.set_title(_pr_label, fontsize=9)
        _ax.legend(fontsize=8)
        _ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Frequency [Hz]")
    fig.suptitle(
        "Standing-wave amplitude modulation — panel vs nopanel  (nowind, 0.1 V)\n"
        "Ratio > 1 = probe near antinode of reflected standing wave.\n"
        "Ratio oscillates with frequency as node/antinode sweeps past the probe.",
        fontsize=9,
    )
    plt.tight_layout()
    plt.show()

    # Also print k·Δx and expected standing-wave modulation at each probe
    print("\n=== k·Δx and probe distance from panel (qualitative) ===")
    _panel_x   = 11.0   # m — approximate panel centre (between 9373 and 12400 mm)
    _x_in_from_panel  = _panel_x - _REFL_X_IN   # distance from panel to IN probe
    _x_up_from_panel  = _panel_x - _REFL_X_UP   # distance from panel to upstream probe
    print(f"  Approximate panel position: {_panel_x:.1f} m from paddle")
    print(f"  9373/170 distance to panel: ~{_x_in_from_panel*1000:.0f} mm")
    print(f"  8804/250 distance to panel: ~{_x_up_from_panel*1000:.0f} mm")
    print(f"  Probe separation: {_REFL_DX*1000:.0f} mm\n")
    print(f"  {'f [Hz]':>7}  {'k [m⁻¹]':>9}  {'λ [m]':>7}  "
          f"{'k·Δx':>6}  {'k·x_IN':>7}  {'k·x_UP':>7}  "
          f"{'2k·x_IN/π':>10}  {'2k·x_UP/π':>10}")
    for _f in sorted(_refl_nowind["WaveFrequencyInput [Hz]"].dropna().unique()):
        _k  = _solve_k(_f)
        _L  = 2 * np.pi / _k
        _kdx = _k * _REFL_DX
        _kx_in = _k * _x_in_from_panel
        _kx_up = _k * _x_up_from_panel
        # 2k·x / π = 0 → antinode, 1 → node, 2 → antinode ...
        print(f"  {_f:7.2f}  {_k:9.3f}  {_L:7.3f}  "
              f"{_kdx:6.3f}  {_kx_in:7.3f}  {_kx_up:7.3f}  "
              f"{2*_kx_in/np.pi:10.3f}  {2*_kx_up/np.pi:10.3f}")

# %% ── Reflection 2: two-probe complex method ─────────────────────────────────
# For each fullpanel nowind run: extract complex FFT at paddle frequency for
# both 8804/250 and 9373/170 (same CSV → phase-coherent).  Solve:
#
#   a = (F_B·exp(-ikx_A) - F_A·exp(-ikx_B)) / (2i·sin(kΔx))   # incident
#   b = (F_A·exp(ikx_B)  - F_B·exp(ikx_A))  / (2i·sin(kΔx))   # reflected
#
# R = |b| / |a|,  conditioned on |sin(kΔx)| > _REFL_COND_THRESH
#
# Applies to fullpanel runs (where there IS a reflector).
# Nopanel runs give R ≈ 0 — a useful sanity check.

_REFL_COND_THRESH = 0.30   # |sin(kΔx)| < this → skip (ill-conditioned)
_REFL_WIN_HZ      = 0.08   # frequency search window around paddle freq

def _fft_complex_at(fft_df, pos, target_hz, window_hz=_REFL_WIN_HZ):
    """Return the raw complex FFT value at the positive-freq bin nearest target_hz.
    Uses 'FFT {pos} complex' (np.fft.fft output; scale N·A/2).
    Only searches positive frequencies — negative-freq mirror is excluded.
    """
    col = f"FFT {pos} complex"
    if col not in fft_df.columns:
        return np.nan + 0j
    _freqs = fft_df.index.values.astype(float)
    _pmask = _freqs > 0
    _wmask = np.abs(_freqs - target_hz) <= window_hz / 2
    _mask  = _pmask & _wmask
    if not _mask.any():
        return np.nan + 0j
    _vals  = fft_df.loc[_mask, col].values.astype(complex)
    _fsub  = _freqs[_mask]
    return _vals[np.argmin(np.abs(_fsub - target_hz))]

_r2_rows = []
_r2_panel_conditions = ["full", "reverse", "no"]  # include nopanel as R≈0 check

_r2_wave = combined_meta[
    (combined_meta["WaveAmplitudeInput [Volt]"] == _REFL_AMP) &
    (combined_meta["WindCondition"] == "no") &
    (combined_meta["PanelCondition"].isin(_r2_panel_conditions))
].copy()

for _, _row in _r2_wave.iterrows():
    _path = _row["path"]
    _fft_df = combined_fft_dict.get(_path)
    if _fft_df is None:
        continue
    _f = _row.get("WaveFrequencyInput [Hz]", np.nan)
    if not np.isfinite(_f):
        continue
    _pc = _row.get("PanelCondition", "?")
    _k  = _solve_k(_f)
    _kdx = _k * _REFL_DX
    _cond = abs(np.sin(_kdx))   # conditioning of the denominator

    # Complex FFT at both probes
    _F_A = _fft_complex_at(_fft_df, _REFL_UP, _f)  # 8804/250
    _F_B = _fft_complex_at(_fft_df, _REFL_IN, _f)  # 9373/170
    if not (np.isfinite(_F_A.real) and np.isfinite(_F_B.real)):
        continue

    if _cond < _REFL_COND_THRESH:
        _R = np.nan   # ill-conditioned — skip
    else:
        _xA = _REFL_X_UP
        _xB = _REFL_X_IN
        _denom = 2j * np.sin(_k * (_xB - _xA))
        _a_inc = (_F_B * np.exp(-1j * _k * _xA) - _F_A * np.exp(-1j * _k * _xB)) / _denom
        _b_ref = (_F_A * np.exp( 1j * _k * _xB) - _F_B * np.exp( 1j * _k * _xA)) / _denom
        _R = abs(_b_ref) / max(abs(_a_inc), 1e-30)

    _r2_rows.append({
        "f [Hz]": _f, "PanelCondition": _pc,
        "k [m⁻¹]": _k, "kΔx": _kdx, "|sin(kΔx)|": _cond,
        "R": _R,
    })

_r2_df = pd.DataFrame(_r2_rows)

# ── Aggregate: median R per (freq, panel_condition) ──────────────────────
if len(_r2_df) > 0:
    _r2_agg = (
        _r2_df.dropna(subset=["R"])
        .groupby(["f [Hz]", "PanelCondition"])["R"]
        .agg(R_median="median", R_std="std", n="count")
        .reset_index()
    )
    print("=== Two-probe reflection coefficient R = |A_reflected| / |A_incident| ===")
    print(f"Only runs where |sin(kΔx)| > {_REFL_COND_THRESH} are used (conditioning filter).\n")
    print(_r2_agg.round(4).to_string(index=False))
    print(f"\nNote: R is measured at the probe array ({_REFL_X_IN:.3f} m from paddle),")
    print(f"      not at the panel face.  Propagation loss (small) is not corrected.")

    # ── Plot ────────────────────────────────────────────────────────────
    _cmap_pc = {"full": "seagreen", "reverse": "darkorange", "no": "steelblue"}
    fig, axes = plt.subplots(2, 1, figsize=(10, 7),
                             gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

    for _pc, _grp in _r2_agg.groupby("PanelCondition"):
        _grp_s = _grp.sort_values("f [Hz]")
        _f_   = _grp_s["f [Hz]"].values
        _R_   = _grp_s["R_median"].values
        _err_ = _grp_s["R_std"].fillna(0).values
        _c    = _cmap_pc.get(_pc, "0.5")
        axes[0].errorbar(_f_, _R_, yerr=_err_, fmt="o-", color=_c, ms=5,
                         lw=1.4, capsize=3, label=f"{_pc}panel")

    axes[0].set_ylabel("R = |A_r| / |A_i|")
    axes[0].set_ylim(bottom=0)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Two-probe reflection coefficient", fontsize=9)

    # Conditioning: |sin(kΔx)| vs frequency (lower → result less reliable)
    _cond_df = (
        _r2_df.groupby("f [Hz]")["|sin(kΔx)|"].first().reset_index().sort_values("f [Hz]")
    )
    axes[1].bar(_cond_df["f [Hz]"], _cond_df["|sin(kΔx)|"],
                width=0.03, color="0.6", label="|sin(kΔx)|")
    axes[1].axhline(_REFL_COND_THRESH, color="tomato", lw=0.8, ls="--",
                    label=f"threshold {_REFL_COND_THRESH}")
    axes[1].set_ylabel("|sin(kΔx)|")
    axes[1].set_xlabel("Frequency [Hz]")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title("Conditioning of the two-probe method  (low = ill-conditioned)", fontsize=9)

    fig.suptitle(
        f"Two-probe complex method — R vs frequency  (nowind, 0.1 V)\n"
        f"Probes: {_REFL_UP} (upstream) and {_REFL_IN} (IN),  separation = {_REFL_DX*1000:.0f} mm\n"
        "Nopanel should give R ≈ 0 (sanity check).  Fullpanel gives true panel reflection.",
        fontsize=9,
    )
    plt.tight_layout()
    plt.show()
else:
    print("No data available for two-probe analysis — check data inventory.")

# %% ── Reflection 3: standing wave pattern check ─────────────────────────────
# If the amplitude ratio oscillates with a predictable k·x pattern, it confirms
# the reflection interpretation.
#
# For a partial standing wave (incident A_i + reflected A_r with R = A_r/A_i):
#
#   A(x) = A_i · sqrt(1 + R² + 2R·cos(2k(x_panel - x) + φ))
#
# where x_panel - x is the distance from the probe to the panel (waves travel in
# -x direction after reflecting).
#
# Comparison: if we take the ratio for the two probes:
#   A(x_IN) / A(x_UP)  should oscillate with the 2kΔx pattern
#   without panel: ratio ≈ 1 (no standing wave → flat across frequency)
#   with panel:    ratio oscillates → peaks and troughs as frequency varies
#
# This cell tests: is the nopanel in/upstream ratio flat, and does the panel
# introduce the expected frequency-dependent modulation?

_r3_rows = []
for _, _row in combined_meta[
    (combined_meta["WaveAmplitudeInput [Volt]"] == _REFL_AMP) &
    (combined_meta["WindCondition"] == "no")
].iterrows():
    _f   = _row.get("WaveFrequencyInput [Hz]", np.nan)
    _pc  = _row.get("PanelCondition", "?")
    if not np.isfinite(_f):
        continue
    _col_in = f"Probe {_REFL_IN} Amplitude (FFT)"
    _col_up = f"Probe {_REFL_UP} Amplitude (FFT)"
    _a_in = float(_row.get(_col_in, np.nan))
    _a_up = float(_row.get(_col_up, np.nan))
    if not (np.isfinite(_a_in) and np.isfinite(_a_up) and _a_up > 0):
        continue
    _r3_rows.append({
        "f [Hz]": _f, "PanelCondition": _pc,
        "ratio_IN_over_UP": _a_in / _a_up,
        "k [m⁻¹]": _solve_k(_f),
    })

_r3_df = pd.DataFrame(_r3_rows)

if len(_r3_df) > 0:
    _r3_agg = (
        _r3_df.groupby(["f [Hz]", "PanelCondition"])["ratio_IN_over_UP"]
        .agg(median="median", std="std", n="count")
        .reset_index()
    )

    _cmap_pc = {"no": "steelblue", "full": "seagreen", "reverse": "darkorange"}
    fig, ax = plt.subplots(figsize=(10, 5))

    for _pc, _grp in _r3_agg.groupby("PanelCondition"):
        _grp_s = _grp.sort_values("f [Hz]")
        _f_    = _grp_s["f [Hz]"].values
        _r_    = _grp_s["median"].values
        _err_  = _grp_s["std"].fillna(0).values
        _c     = _cmap_pc.get(_pc, "0.5")
        ax.errorbar(_f_, _r_, yerr=_err_, fmt="o-", color=_c, ms=5,
                    lw=1.4, capsize=3, label=f"{_pc}panel")

    ax.axhline(1.0, color="0.4", lw=0.8, ls="--")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(f"A({_REFL_IN}) / A({_REFL_UP})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title(
        f"Amplitude ratio between probes: {_REFL_IN} / {_REFL_UP}  (nowind, 0.1 V)\n"
        "Nopanel: should be flat near 1 (no standing wave modulation).\n"
        "Fullpanel: oscillates with frequency if reflected wave present at the IN probe.\n"
        "Divergence between nopanel and fullpanel curves = standing-wave signature.",
        fontsize=9,
    )
    plt.tight_layout()
    plt.show()

    print("\n=== IN/UP ratio by panel condition ===")
    print(_r3_agg.round(4).to_string(index=False))

    # Qualitative comparison: nopanel vs fullpanel ratio at same frequencies
    _no_r   = _r3_agg[_r3_agg["PanelCondition"] == "no"].set_index("f [Hz]")["median"]
    _ful_r  = _r3_agg[_r3_agg["PanelCondition"] == "full"].set_index("f [Hz]")["median"]
    _common = _no_r.index.intersection(_ful_r.index)
    if len(_common) > 0:
        print("\n=== Panel effect on IN/UP ratio (fullpanel / nopanel) ===")
        print(f"  {'f [Hz]':>7}  {'nopanel':>8}  {'fullpanel':>10}  {'difference':>12}")
        for _f in sorted(_common):
            _diff = _ful_r[_f] - _no_r[_f]
            print(f"  {_f:7.2f}  {_no_r[_f]:8.4f}  {_ful_r[_f]:10.4f}  {_diff:+12.4f}")
    else:
        print("\n⚠  No overlapping frequencies between nopanel and fullpanel — "
              "cannot compute direct comparison.")
        print("   Frequencies in nopanel: "
              f"{sorted(_no_r.index.tolist())}")
        print("   Frequencies in fullpanel: "
              f"{sorted(_ful_r.index.tolist())}")
else:
    print("No data available for inter-probe ratio analysis.")
