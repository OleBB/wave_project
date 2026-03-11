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

from wavescripts.improved_data_loader import load_analysis_data, load_processed_dfs
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
    Path("waveprocessed/PROCESSED-20260307-ProbPos4_31_FPV_2-tett6roof"),
    Path("waveprocessed/PROCESSED-20251112-tett6roof"),
    Path("waveprocessed/PROCESSED-20251113-tett6roof"),
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
        "WaveFrequencyInput [Hz]":   0.65,
        "WavePeriodInput":           None,
        "WindCondition":             ["full"],
        "TunnelCondition":           None,
        "Mooring":                   "low",
        "PanelCondition":            "no", #"["reverse", "full"],
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
        "WaveAmplitudeInput [Volt]": None,
        "WaveFrequencyInput [Hz]":   None,
        "WavePeriodInput":           None,
        "WindCondition":             None,
        "TunnelCondition":           None,
        "PanelCondition":            "reverse",#"full"],
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
        "WaveAmplitudeInput [Volt]": None,
        "WaveFrequencyInput [Hz]":   None,
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
        "probes":     ["9373/170", "12545/250", "9373/340", "8804/250"],
    },
}

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
        "WaveFrequencyInput [Hz]":   [1.3],
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
        "probes":      ["12545/250", "9373/340"],
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
        "chooseAll": True,
        "chooseFirst": False,
        "chooseFirstUnique": True,
    },
    "filters": {
        "WaveAmplitudeInput [Volt]": [0.1, 0.2, 0.3],
        "WaveFrequencyInput [Hz]":   [1.3],
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
        "probes":     ["12545/250", "9373/340"],
    },
}

plot_swell_scatter(combined_meta, swellplotvariables)

# %% ── wavenumber study ───────────────────────────────────────────────────────
_probe_positions = ["9373/170", "12545/250", "9373/340", "8804/250"]
wavenumber_cols = [f"Probe {pos} Wavenumber (FFT)" for pos in _probe_positions]
fft_dimension_cols = [CG.fft_wave_dimension_cols(pos) for pos in _probe_positions]
meta_wavenumber = combined_meta[["path"] + [c for c in wavenumber_cols if c in combined_meta.columns]].copy()
print(meta_wavenumber.describe())

# %% ── reconstructed signal — single experiment ───────────────────────────────
# Pick one experiment to inspect its reconstructed time-domain signal.
single_path = filtrert_frequencies["path"].iloc[0]
single_meta = filtrert_frequencies.iloc[[0]]

fig, ax = plot_reconstructed(
    {single_path: filtered_fft_dict[single_path]}, single_meta, freqplotvariables
)

# %% ── reconstructed signal — all filtered experiments ───────────────────────
_recon_paths = {p: filtered_fft_dict[p] for p in filtrert_frequencies["path"] if p in filtered_fft_dict}
fig, axes = plot_reconstructed(
    _recon_paths, filtrert_frequencies, freqplotvariables, data_type="fft"
)


"""
#
#
# =============================================================================
# WIND-ONLY ANALYSIS
# Runs with no wave input (WaveFrequencyInput NaN) to characterise wind-only
# surface response. Compare wind conditions against stillwater baseline (no wind).
# =============================================================================
"""
# %% ── wind-only — filter runs ────────────────────────────────────────────────
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

# %% ── wind-only — lazy-load processed_dfs (skipped during normal load) ─────
if not processed_dfs:
    print("Loading processed_dfs (~75 MB, ~20 s)...")
    _t0 = time.perf_counter()
    processed_dfs = load_processed_dfs(*PROCESSED_DIRS)
    print(f"Loaded {len(processed_dfs)} runs in {time.perf_counter() - _t0:.1f} s")

# %% ── wind-only — build PSD dict (same format as psd_dict) ──────────────────
# Compute Welch PSD for all nowave runs so we can reuse plot_frequency_spectrum.
_NPERSEG = 4096

_wind_psd_dict = {}
for path, df in processed_dfs.items():
    if path not in _meta_nowave_all["path"].values:
        continue
    eta_cols = [c for c in df.columns if c.startswith("eta_")]
    if not eta_cols:
        continue
    records = {}
    freqs_ref = None
    for eta_col in eta_cols:
        pos = eta_col[len("eta_"):]          # "eta_9373/170" → "9373/170"
        sig = df[eta_col].dropna().values
        if len(sig) < _NPERSEG:
            continue
        f, pxx = _welch(sig, fs=_FS, nperseg=_NPERSEG)
        if freqs_ref is None:
            freqs_ref = f
        records[f"Pxx {pos}"] = pxx
    if records and freqs_ref is not None:
        _wind_psd_dict[path] = pd.DataFrame(records, index=pd.Index(freqs_ref, name="Frequencies"))

print(f"Built wind_psd_dict for {len(_wind_psd_dict)} nowave runs")

# %% ── wind-only — PSD spectrum plot ─────────────────────────────────────────
_wind_psd_plotvars = {
    "overordnet": {"chooseAll": True, "chooseFirst": False, "chooseFirstUnique": False},
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
        "probes":     ["9373/170", "12545/250", "9373/340", "8804/250"],
    },
}

fig, axes = plot_frequency_spectrum(
    _wind_psd_dict, _meta_nowave_all, _wind_psd_plotvars, data_type="psd"
)



# %% ── wind-only — statistics (mean setup + RMS per probe) ───────────────────
_PROBE_POSITIONS = ["9373/170", "12545/250", "9373/340", "8804/250"]
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

# %% ── investigate: wind-only growth 9373 → 12545 vs claimed wave growth ─────
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
        "Probe 9373/340 Amplitude", "Probe 12545/250 Amplitude",
        "Probe 12545/170 Amplitude", "Probe 12545/340 Amplitude",
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
_wave["ratio_170_vs_340"] = _wave["Probe 12545/170 Amplitude"] / _wave["Probe 12545/340 Amplitude"]
print(_wave[["WaveFrequencyInput [Hz]", "WaveAmplitudeInput [Volt]", "WindCondition",
             "PanelCondition", "Probe 12545/170 Amplitude", "Probe 12545/340 Amplitude",
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
_sw_row = _meta_stillwater.iloc[0]
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

    plt.tight_layout()
    plt.show()
    print(f"{len(_sw_df)} samples  |  {len(_sw_df)/_FS:.1f} s")

# %% ── stillwater — probe uncertainty statistics ──────────────────────────────
# "Probe {pos} Amplitude" = (P97.5 - P2.5) / 2 — already in combined_meta for all runs.
# No need to reload processed_dfs; just filter stillwater rows and pivot.
_amp_cols = [c for c in _meta_stillwater.columns
             if c.startswith("Probe ") and c.endswith(" Amplitude")
             and "FFT" not in c and "PSD" not in c
             and _meta_stillwater[c].notna().any()]

_sw_stats = (
    _meta_stillwater[["path"] + _amp_cols]
    .copy()
    .rename(columns={"path": "run"})
)
_sw_stats["run"] = _sw_stats["run"].apply(lambda p: _Path(p).name)
_sw_stats = _sw_stats.rename(columns={c: c.replace("Probe ", "").replace(" Amplitude", "")
                                       for c in _amp_cols})

print("=== Stillwater noise amplitude per run [mm] (pipeline definition: (P97.5−P2.5)/2) ===")
print(_sw_stats.to_string(index=False))

# Summary across runs
_probe_cols = [c.replace("Probe ", "").replace(" Amplitude", "") for c in _amp_cols]
_sw_summary = _sw_stats[_probe_cols].agg(["mean", "std", "min", "max"]).T
_sw_summary.index.name = "probe"
print("\n=== Per-probe summary across all stillwater runs [mm] ===")
print(_sw_summary.round(4).to_string())

# %% ── first wave arrival detection ───────────────────────────────────────────
# For each wave run × probe: find the first time the rolling amplitude exceeds
# threshold_factor × stillwater noise floor.
# Physics: fast long-wave precursors may arrive well before the _SNARVEI_CALIB start.
from wavescripts.wave_detection import find_first_arrival

_THRESHOLD_FACTOR = 2.0   # detection at 2× noise floor
_WINDOW_S         = 0.5   # rolling window length [s]
_PROBE_POSITIONS  = ["8804/250", "9373/170", "9373/340", "12545/250"]  # adjust to your layout

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

# %% ── first arrival — plot arrival time vs distance ─────────────────────────
import seaborn as sns
from wavescripts.plot_utils import WIND_COLOR_MAP as _WCM
sns.set_style("ticks", {"axes.grid": True})
_plot_df = _arrival_df.dropna(subset=["arrival_s"])
g = sns.relplot(
    data=_plot_df, x="dist_mm", y="arrival_s",
    hue="wind", col="freq_hz",
    kind="line", marker="o", dashes=False,
    errorbar=None, markersize=10, linewidth=0.8,
    height=3.5, aspect=1.3,
    palette=_WCM,
    facet_kws={"sharey": True},
)
for ax in g.axes.flat:
    ax.set_xlabel("Probe distance from paddle [mm]")
    ax.set_ylabel("First arrival [s]")
g.figure.suptitle(
    f"First wave arrival  (threshold = {_THRESHOLD_FACTOR}× noise floor,"
    f"  window = {_WINDOW_S} s)",
    y=1.02, fontsize=9,
)
plt.tight_layout()
plt.show()

# %% -- no facet
sns.set_style("ticks", {"axes.grid": True})
_plot_df = _arrival_df.dropna(subset=["arrival_s"])
g = sns.scatterplot(
    data=_plot_df, x="dist_mm", y="arrival_s",
    hue="wind", col="freq_hz",
    kind="line", marker="o", dashes=False,
    errorbar=None, markersize=10, linewidth=0.8,
    height=3.5, aspect=1.3,
    palette=_WCM,
)
for ax in g.axes.flat:
    ax.set_xlabel("Probe distance from paddle [mm]")
    ax.set_ylabel("First arrival [s]")
g.figure.suptitle(
    f"First wave arrival  (threshold = {_THRESHOLD_FACTOR}× noise floor,"
    f"  window = {_WINDOW_S} s)",
    y=1.02, fontsize=9,
)
plt.tight_layout()
plt.show()

# %% ── wind-only — overview + zoom (solo & parallel probes) ───────────────────
_wind_row = _meta_wind_only.iloc[0]
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
    _all_eta = sorted([c for c in _wind_df.columns if c.startswith("eta_")])
    from collections import defaultdict
    _by_dist = defaultdict(list)
    for _col in _all_eta:
        _pos  = _col.replace("eta_", "")          # e.g. "9373/170"
        _dist = int(_pos.split("/")[0])
        _by_dist[_dist].append(_col)

    _solo_cols     = [cols[0] for cols in _by_dist.values() if len(cols) == 1]
    _parallel_cols = [col for cols in _by_dist.values() if len(cols) > 1 for col in cols]

    def _wind_plot(eta_cols, subtitle):
        _colors = plt.cm.tab10(np.linspace(0, 0.9, max(len(eta_cols), 1)))
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

        plt.tight_layout()
        plt.show()

    _wind_plot(_solo_cols,     "solo probes")
    _wind_plot(_parallel_cols, "parallel probes")

    print(f"{len(_wind_df)} samples  |  {len(_wind_df)/_FS:.1f} s  |  wind: {_wind_cond}")
    print(f"Solo:     {[c.replace('eta_','') for c in _solo_cols]}")
    print(f"Parallel: {[c.replace('eta_','') for c in _parallel_cols]}")

# %% ── first arrival — single plot, all frequencies, no wind-waves ────────────
_MIN_ARRIVAL_S = 0.5   # below this = wind-wave false detection, ignore
_plot_df2 = _arrival_df[_arrival_df["arrival_s"] > _MIN_ARRIVAL_S].copy()

_WIND_LS = {"no": "-", "lowest": "--", "full": ":"}  # linestyle by wind
_freqs_sorted = sorted(_plot_df2["freq_hz"].dropna().unique())
_freq_colors  = {f: c for f, c in zip(_freqs_sorted,
                  plt.cm.rainbow(np.linspace(0, 1, len(_freqs_sorted))))}

# Average parallel probes (same dist_mm, same run/wind/freq) → mean ± half-range
_agg = (
    _plot_df2
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
                linestyle=_WIND_LS.get(wind, "-"),
                label=f"{freq} Hz")

ax.set_xlabel("Probe distance from paddle [mm]")
ax.set_ylabel("First arrival [s]")
ax.set_title(
    f"First wave arrival — all frequencies  "
    f"(threshold {_THRESHOLD_FACTOR}× noise,  arrivals > {_MIN_ARRIVAL_S} s shown)"
)
_handles, _labels = ax.get_legend_handles_labels()
ax.legend(_handles[::-1], _labels[::-1], fontsize=8, title="freq / wind")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%  ----- push d tale to end
import dtale
# med et forsøk på å få mere info på skjermen, med at
# tittelkolonnen er høyere, så de under kan være bredere
# dtale.app.initialize(
#     custom_css="""
#       .rt-th {
#         white-space: normal !important;   /* allow wrapping */
#         word-break: break-word;           /* break long tokens */
#       }
#     """
# )
dtale.show(combined_meta, host="localhost").open_browser()


print(combined_meta.columns.to_list())
