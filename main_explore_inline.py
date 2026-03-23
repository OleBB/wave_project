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

from pandas._libs.lib import fast_unique_multiple_list_gen
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
    # Path("waveprocessed/PROCESSED-20251005-sixttry6roof-highMooring"), #denne har probe 1 på 18000. Men husk at taket ikke var tetta helt.
    # Path("waveprocessed/PROCESSED-20251110-tett6roof-lowM-ekte580"), # mange kjøringer med -per15
    # Path("waveprocessed/PROCESSED-20251110-tett6roof-lowMooring"), # noen kjøringer med  -per30-
    # Path("waveprocessed/PROCESSED-20251110-tett6roof-lowMooring-2"), #et par kjøringer med -per15-
    # Path("waveprocessed/PROCESSED-20251112-tett6roof"),
    # Path("waveprocessed/PROCESSED-20251113-tett6roof"),
    # Path("waveprocessed/PROCESSED-20251113-tett6roof-loosepaneltaped"),
    # Path("waveprocessed/PROCESSED-20251113-tett6roof-probeadjusted"),
    # Path("waveprocessed/PROCESSED-20260305-newProbePos-tett6roof"),
    # Path("waveprocessed/PROCESSED-20260306-newProbePos-tett6roof"),
    Path("waveprocessed/PROCESSED-20260307-ProbPos4_31_FPV_2-tett6roof"),
    Path("waveprocessed/PROCESSED-20260312-ProbPos4_31_FPV_2-tett6roof"),
    Path("waveprocessed/PROCESSED-20260313-ProbePos4_31_FPV_2-tett6roof"),
    Path("waveprocessed/PROCESSED-20260314-ProbePos4_31_FPV_2-tett6roof"),
    Path("waveprocessed/PROCESSED-20260316-ProbePos4_31_FPV_2-tett6roof"),
    Path("waveprocessed/PROCESSED-20260316-ProbePos4_31_FPV_2-tett6roof-under9Mooring"),
    Path("waveprocessed/PROCESSED-20260319-ProbePos4_31_FPV_2-tett6roof-under9Mooring"),
]

combined_meta, processed_dfs, combined_fft_dict, combined_psd_dict = load_analysis_data(
    *PROCESSED_DIRS, load_processed=False
)
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
        "WavePeriodInput":           None,
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

# %% ── damping vs frequency ───────────────────────────────────────────────────
dampingplotvariables = {
    "overordnet": {"chooseAll": False, "chooseFirst": False, "chooseFirstUnique": False},
    "filters": {
        "WaveAmplitudeInput [Volt]": (0.1,0.3),
        "WaveFrequencyInput [Hz]":   (0.1,1.8),
        "WavePeriodInput":           None,
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
        "WaveFrequencyInput [Hz]":   [1.7],
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
# %%  see the stillwater leve.

import importlib
import wavescripts.plot_quicklook as pql
importlib.reload(pql)
from wavescripts.plot_quicklook import plot_stillwater_fit
plot_stillwater_fit(processed_dfs, combined_meta, cfg)
from wavescripts.improved_data_loader import load_processed_dfs
from pathlib import Path
PROCESSED_DIRS = sorted(Path("waveprocessed").glob("PROCESSED-*"))
processed_dfs = load_processed_dfs(*PROCESSED_DIRS)






# %%
import importlib
import wavescripts.constants as _c
importlib.reload(_c)
import wavescripts.plot_quicklook as pql
importlib.reload(pql)
from wavescripts.plot_quicklook import plot_stillwater_fit
plot_stillwater_fit(processed_dfs, combined_meta, cfg, date="2026-03-07")







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

# %% ── period-based arrival detection (experimental) ─────────────────────────
# Instead of a broadband amplitude threshold, slide a window of exactly N periods
# at the target frequency and compute the FFT amplitude at that frequency.
# When that narrow-band amplitude first exceeds a threshold, the wave has arrived.
# This rejects wind-wave energy (wrong frequency) and is probe-noise-independent.

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
for _, _row in combined_meta[
        combined_meta["WaveFrequencyInput [Hz]"].notna() &
        (combined_meta["WindCondition"] == "no")
].iterrows():
    _df = processed_dfs.get(_row["path"])
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
ax.set_title(f"First wave arrival — period-based detection, no wind  "
             f"(window = {_N_PERIODS} periods)")
_h, _l = ax.get_legend_handles_labels()
ax.legend(_h[::-1], _l[::-1], fontsize=8, title="frequency")
ax.grid(True, alpha=0.3)
plt.show()

_thresh_str2 = ",  ".join(
    f"{pos} → {_THRESH_FACTOR * _noise_floor[pos]:.2f} mm"
    for pos in _PROBE_POSITIONS if pos in _noise_floor
)
print(
    f"Caption: First arrival detected by sliding a {_N_PERIODS}-period window and "
    f"measuring FFT amplitude at the target frequency. "
    f"Threshold: {_THRESH_FACTOR}× stillwater noise floor per probe "
    f"({_thresh_str2}). "
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

_mstop_date = "2026-03-07"
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
