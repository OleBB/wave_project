#!/usr/bin/env python3
"""
Save publication-quality figures and tables for the thesis.

Run from terminal:
    conda activate draumkvedet
    cd /Users/ole/Kodevik/wave_project
    python main_save_figures.py

Each section corresponds to a thesis chapter and figure/table number.
Set save_plot=True (or call save_and_stub) when a figure is ready to export.
Requires processed cache. Run main.py first if stale.

INPUT KEYS  (experimental conditions):
    WaveAmplitudeInput [Volt]   — paddle drive voltage (0.1 V / 0.2 V)
    WaveFrequencyInput [Hz]     — paddle frequency (0.65–1.9 Hz)
    PanelCondition              — full / reverse / no
    WindCondition               — full / lowest / no

OUTPUT KEYS (measured results):
    OUT/IN (FFT)                — damping ratio, FFT amplitude at paddle freq only
                                  (wind waves excluded — they are characterised separately)
    ka                          — wavenumber × amplitude, found per probe per run
                                  (not pre-calculated; measured from the actual wave)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FIGURE INDEX
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Status legend:  ✓ ready   ~ draft (DRAFT stamp)   ✗ placeholder (blank fig)

CHAPTER 04 — METHODOLOGY
  §1   ch04_probe_noise_floor      ~  Stillwater noise floor per probe / hw config
  §2   ch04_stillwater_timing      ✗  Swell decay time vs wait time  [TODO]
  §3   ch04_parallel_ratio         ~  Wall/far-side amplitude ratio vs frequency
  §3b  ch04_probe_height           ~  Probe height & range-mode validity (4 cond × 4 probe)
  §3c  ch04_mooring_comparison     ✓  Mooring rubber band length: loose230 vs loose300 (delegated)
  §3d  ch04_sound_speed             ~  Speed-of-sound / lab temperature drift
  §3e  ch04_parallel_probe_agreement ~  9373/170 vs 9373/340 — justifies mean-IN canonical ref
  §4-1 ch04_wind_psd               ~  Wind PSD per probe (nowave runs)
  §4-2 ch04_wind_reflection        ✗  Wind reflection from panel     [TODO]
  §4-3 ch04_fft_wave               ~  FFT spectrum at paddle freq (1.3 Hz example)
  §4-4 ch04_wind_snr               ~  Spectral SNR: paddle / wind noise per probe
  §4-5 ch04_td_vs_fft              ~  A_td vs A_FFT: why FFT is required
  §4f  ch04_reconstruction_AvsB    ~  Peak-bin (A) vs band-integrated (B) reconstruction equivalence
  §4g  ch04_reconstruction_pure_wind ~ Pure wind via no-wind residual subtraction (Stokes removal)
  §5   ch04_timeseries_overview    ~  Full time-series with stable-window band
  §6   ch04_first_arrival          ~  First wave arrival vs probe distance
  §7   ch04_wave_stability         ~  Wave stability and period_cv vs frequency
  §8   ch04_lateral_nowind         ~  Lateral equality (parallel ratio, no-wind)
  §9   ch04_amplitude_profile      ~  Amplitude at every probe, all runs

CHAPTER 05 — RESULTS
  §1   ch05_damping_freq           ✓  OUT/IN (FFT) vs frequency  ← primary result
  §2   ch05_damping_scatter        ✓  OUT/IN scatter vs amplitude
  §3   ch05_damping_wind_delta     ~  Wind effect on damping (delta plot)
  §3b  ch05_t_cross                ~  T_cross: honest wind effect via clean nowind reference
  §4   ch05_damping_ka             ✗  Damping vs ka (wavenumber × amplitude) [TODO]
  §5   ch05_swell_scatter          —  DROPPED — kept commented in place; see §5 cell
  §6   ch05_reconstructed          ~  FFT-reconstructed paddle signal
  §7   ch05_damping_all_data_scatter  ~  Supplementary: OUT/IN across ALL conditions (combined_meta)

DIAGNOSTICS
  D1   diag_13hz_consistency        ✗  1.3 Hz cross-session consistency check [TODO]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# %% ── dev: reload modules (run this cell after editing wavescripts/) ─────────
import importlib, wavescripts.plotter as _pm, wavescripts.filters as _fm

importlib.reload(_pm); importlib.reload(_fm);

from wavescripts.plotter import (plot_probe_noise_floor, plot_parallel_ratio,
                                  plot_frequency_spectrum, plot_wave_stability,
                                  plot_timeseries_overview,
                                  plot_damping_freq, plot_damping_scatter,
                                  plot_damping_wind_delta)
from wavescripts.filters import apply_experimental_filters as _aef


# %% ----------- Velkommen ----------------------------
import os
from datetime import datetime as _dt
from pathlib import Path

import time

import matplotlib
matplotlib.use("Agg")   # non-interactive — plt.show() is a no-op; avoids hanging
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
from wavescripts.improved_data_loader import load_analysis_data, load_processed_dfs, ANALYSIS_PROBES, get_configuration_for_date
from wavescripts.plot_utils import (WIND_COLOR_MAP, apply_thesis_style,
                                    add_draft_stamp, build_fig_meta, save_and_stub)
from wavescripts.plotter import (
    plot_all_probes,
    plot_damping_freq,
    plot_damping_ka,
    plot_damping_scatter,
    plot_damping_wind_delta,
    plot_fft_peak_bias_cancellation,
    plot_first_arrival,
    plot_frequency_spectrum,
    plot_parallel_ratio,
    plot_probe_noise_floor,
    plot_reconstructed,
    plot_sound_speed,
    plot_swell_scatter,
    plot_td_vs_fft,
    plot_timeseries_overview,
    plot_wave_stability,
    plot_wind_snr,
)
from wavescripts.wave_detection import find_first_arrival
# %%
FS = MEASUREMENT.SAMPLING_RATE

try:
    file_dir = Path(__file__).resolve().parent
except NameError:
    file_dir = Path.cwd()
os.chdir(file_dir)

# ── Datasets ──────────────────────────────────────────────────────────────────
# Two named datasets from a single load:
#   ALL_PROCESSED_DIRS   → combined_meta  — all sessions → CH04 methodology
#   RESULTS_PROCESSED_DIRS → meta_results — two validated sessions → CH05 results
#
# Why single load: loading FFT/PSD parquets twice costs ~5 min vs ~3 min once.
# meta_results is a filtered DataFrame subset — essentially free to derive.
#
# Why two sets: older sessions have interpolation artefacts at ≥1.6 Hz but are
# valid for noise floor, probe characterisation, wind PSD, etc. (CH04). Result
# figures (OUT/IN vs freq/kL, damping vs amplitude) must use only the two
# validated lowrange/h100 folders.

ALL_PROCESSED_DIRS = [
    # ── Nov 2025: probe 1 at 18000 mm, roof not fully sealed ──────────────────
    Path("waveprocessed/PROCESSED-20251005-sixttry6roof-highMooring"),
    # ── Nov 2025: probe 1 moved to 8804 mm, lowMooring ────────────────────────
    Path("waveprocessed/PROCESSED-20251110-tett6roof-lowM-ekte580"),
    Path("waveprocessed/PROCESSED-20251110-tett6roof-lowMooring"),
    Path("waveprocessed/PROCESSED-20251110-tett6roof-lowMooring-2"),
    Path("waveprocessed/PROCESSED-20251112-tett6roof"),
    Path("waveprocessed/PROCESSED-20251113-tett6roof"),
    Path("waveprocessed/PROCESSED-20251113-tett6roof-loosepaneltaped"),
    Path("waveprocessed/PROCESSED-20251113-tett6roof-probeadjusted"),
    # ── Mar 2026: new probe positions (march2026_rearranging config) ───────────
    Path("waveprocessed/PROCESSED-20260305-newProbePos-tett6roof"),           # in=9373/170, out=11800/250 — transitional
    Path("waveprocessed/PROCESSED-20260306-newProbePos-tett6roof"),           # in=9373/170, out=11800/250 — transitional
    # ── Mar 2026: final probe positions (march2026_better_rearranging) ─────────
    Path("waveprocessed/PROCESSED-20260307-ProbPos4_31_FPV_2-tett6roof"),
    Path("waveprocessed/PROCESSED-20260312-ProbPos4_31_FPV_2-tett6roof"),
    Path("waveprocessed/PROCESSED-20260313-ProbePos4_31_FPV_2-tett6roof"),
    Path("waveprocessed/PROCESSED-20260314-ProbePos4_31_FPV_2-tett6roof"),
    Path("waveprocessed/PROCESSED-20260316-ProbePos4_31_FPV_2-tett6roof"),
    Path("waveprocessed/PROCESSED-20260316-ProbePos4_31_FPV_2-tett6roof-under9Mooring"),
    Path("waveprocessed/PROCESSED-20260319-ProbePos4_31_FPV_2-tett6roof-under9Mooring"),
    Path("waveprocessed/PROCESSED-20260321-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-RENAMED"),
    # ── Mar 2026: probe lowered — height136 (transitional, 1 dag) ─────────────
    Path("waveprocessed/PROCESSED-20260323-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height136"),
    # ── Mar 2026: probe lowered to height100 ──────────────────────────────────
    Path("waveprocessed/PROCESSED-20260323-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100"),
    Path("waveprocessed/PROCESSED-20260324-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100"),
    Path("waveprocessed/PROCESSED-20260325-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100"),
    Path("waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100"),
    # ── Mar 2026: lowrange switch enabled ─────────────────────────────────────
    Path("waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange"),
    Path("waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"),
]

# Only the two validated sessions — results analysis (CH05)
# lowrange mode, h100, final probe config (march2026_better_rearranging)
RESULTS_PROCESSED_DIRS = [
    Path("waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange"),
    Path("waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"),
]
PROCESSED_DIRS = ALL_PROCESSED_DIRS  # backward-compat alias

# Register results-quality datasets for figure stubs
import wavescripts.plot_utils as _pu
_pu.ACTIVE_DATASETS = [p.name for p in RESULTS_PROCESSED_DIRS]

# ── Load from cache ───────────────────────────────────────────────────────────
print("Loading analysis data...")
combined_meta, _, combined_fft_dict, combined_psd_dict = load_analysis_data(
    *ALL_PROCESSED_DIRS
)

# %%
# processed_dfs is heavy (~75 MB). Loaded when needed by sections below.
processed_dfs = load_processed_dfs(*ALL_PROCESSED_DIRS)

# ── Results subset (CH05) ─────────────────────────────────────────────────────
# meta_results: only the two validated folders, used for all CH05 result figures.
# combined_meta: all sessions, used for CH04 methodology characterisation.
#
# NOTE: combined_meta["path"] points to the raw CSV in wavedata/<date>-<exp>/file.csv.
# RESULTS_PROCESSED_DIRS names are PROCESSED-<date>-<exp> — the "PROCESSED-" prefix
# does not appear in the path strings. Strip it to match the wavedata folder name.
_results_wavedata_names = {p.name.removeprefix("PROCESSED-") for p in RESULTS_PROCESSED_DIRS}
meta_results = combined_meta[
    combined_meta["path"].apply(lambda p: any(d in str(p) for d in _results_wavedata_names))
].copy()
# Merge mooring rubber band variants — validated equal for 1.3–1.6 Hz (CH04 §3c result).
# Δ = −3% to +0.4% at 1.4–1.6 Hz, 0.2/0.3V — within ±7% SW measurement uncertainty.
# See analysis_scratch/mooring_comparison_findings.md.
meta_results["Mooring"] = meta_results["Mooring"].replace({
    "below_90_loose230": "below_90_loose",
    "below_90_loose300": "below_90_loose",
})

# ── Canonical IN/OUT reference = mean of all probes at the same distance ─────
# As of 2026-04-18 this is computed in processor2nd.py::_update_more_metrics
# and stored permanently in meta.json. The mean lives in "IN Amplitude (FFT)"
# and "OUT Amplitude (FFT)"; per-era contributors are listed in
# "in_probes_used" / "out_probes_used". The earlier post-load hook
# (wavescripts/mean_in_probe.py) has been archived under
# analysis_scratch/archive/2026-04-18_mean_in_probe_hook/.
# No runtime transformation needed — meta_results already carries the
# canonical columns as loaded from cache.


# ── Placeholder helper ────────────────────────────────────────────────────────
def _save_placeholder(figure_name: str, section_label: str, chapter: str) -> None:
    """Save a red-stamped DRAFT placeholder for a not-yet-implemented figure."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.text(0.5, 0.55, section_label, ha="center", va="center",
            transform=ax.transAxes, fontsize=13, fontweight="bold", color="#444")
    ax.text(0.5, 0.38, "Not yet implemented", ha="center", va="center",
            transform=ax.transAxes, fontsize=10, color="#888", style="italic")
    ax.set_axis_off()
    meta = build_fig_meta(
        {"filters": {}, "plotting": {
            "figure_name": figure_name,
            "draft": True,
            "caption": f"PLACEHOLDER — {section_label}. Not yet implemented.",
        }},
        chapter=chapter,
    )
    save_and_stub(fig, meta, plot_type=figure_name, force_stub=True)
    plt.close(fig)


# =============================================================================
# CHAPTER 04 — METHODOLOGY
# =============================================================================
# %%

# %%
"""
── CH04 § 1 — Probe uncertainty / noise floor ───────────────────────────────
Goal: show the stillwater noise amplitude per probe and hardware configuration,
and derive the minimum detectable wave amplitude (detection threshold).

Three questions answered per (probe, config):
  1. Precision  — how much does the reading fluctuate in still water?
  2. Bias       — do probes agree on the mean water level within a config?
  3. Threshold  — what is the smallest detectable wave amplitude?

Data: combined_meta stillwater rows (WindCondition=="no", WaveFrequencyInput NaN).
      processed_dfs needed for quantization_step_mm (optional but recommended).

Groups: probe_height_mm × probe_range_mode — 4 hardware configurations:
  h272/high  (default pre-2026-03-23)
  h136/high
  h100/high
  h100/low

Metrics (all from combined_meta, shift-invariant — valid at any probe height):
  noise_95pct_amp_mm   (P97.5−P2.5)/2   mean across accepted runs in group
  noise_rms_mm         std(raw signal)   mean across accepted runs in group
  mean_level_mm        median level      mean across accepted runs in group
  bias_vs_ref_mm       mean_level − cross-probe mean (within group)
  quantization_step_mm P5 of nonzero |diff(η)|  from processed_dfs
  detection_threshold_mm  max(k_sigma·σ,  k_q·q)   default max(3σ, 2q)
"""

from datetime import datetime as _dt
from wavescripts.improved_data_loader import get_configuration_for_date
# Probe numbers derived from current config (hardware IDs, fixed across configs):
_active_cfg = get_configuration_for_date(_dt(2026, 3, 15))
_PROBE_NUM_MAP = {pos: num for num, pos in _active_cfg.probe_col_names().items()}

_pv_noise_floor = {
    "filters": {},
    "plotting": {
        "show_plot": True,
        "save_plot": True,            # DRAFT — noise floor plot not yet polished
        "draft":     True,
        "figure_name": "ch04_probe_noise_floor",
        "force_stub": True,
        "caption": (
            "Stillwater 95\\% noise amplitude $(P_{{97.5}} - P_{{2.5}})/2$ per "
            "ultrasound wave gauge, with no waves and no wind. "
            "Each panel shows one hardware configuration "
            "(probe height above still water / range mode). "
            "Blue bars: mean across accepted stillwater runs within each configuration "
            "(error bars: \\pm 1\\,std). "
            "White dots: individual run values. "
            "Quantization step $q = 0.03$--$0.05$\\,mm per probe "
            "(P5 of nonzero sample-to-sample differences); "
            "Dashed red line: detection threshold $\\max(3\\,\\sigma,\\; 2\\,q)$ "
            "per probe, where $\\sigma$ is the rms noise; "
            "wave amplitudes below this line are indistinguishable from stillwater noise."
        ),
    },
}

start = time.perf_counter()
_figs_nf, _noise_summary = plot_probe_noise_floor(
    combined_meta, ANALYSIS_PROBES, _pv_noise_floor,
    group_by=["probe_height_mm", "probe_range_mode"],
    processed_dfs = processed_dfs,
    highlight_keyword="wavemakeroff-1hour",  # visual star only, no effect on metrics
    probe_number_map=_PROBE_NUM_MAP,
)
print("\n=== Probe noise floor summary [mm] ===")
print(_noise_summary.round(4).to_string())
end = time.perf_counter()
print(f"probe uncertainty-plot took {end - start:.4f} s")






# %%
"""
── CH04 § 2 — Stillwater timing (how long to wait between runs) ─────────────
Goal: show that long-wave swell from previous runs decays over time, and that
wind dramatically shortens the required waiting time.

Data: repeated stillwater runs at different times after wave runs; look at
low-frequency PSD content in eta_* columns over time.

Figures:
  - Plot:  PSD of eta at the OUT probe vs time-after-wave (semi-log, low freqs)
  - Note:  wind-only runs show near-immediate settling(return to wind-wave spectrum) — physical explanation
           (wind chops suppress long-wave coherence in the tank).
"""
# TODO: implement stillwater timing figure
_save_placeholder("ch04_stillwater_timing", "CH04 §2 — Stillwater timing", chapter="04")


# %%
"""
── CH04 § 3 — Probe placement: longitudinal and lateral effects ─────────────
Goal: show what parallel probes tell us — lateral uniformity without wind,
lateral asymmetry with wind. Also: why the longitudinal positions were chosen.

Data: combined_meta, parallel_ratio column, no-wind wave runs. But,
the probes placed downstream are to be trusted more, because no interference from mooring and panel.

Figures:
  - Plot:  parallel_ratio vs frequency, coloured by WindCondition
  - Plot:  parallel_ratio vs frequency, coloured by PanelCondition (reflection)
  - Table: parallel_ratio summary (mean, std) by wind/panel group
"""

""" PRINTOUT
Ratio of wall-side to far-side probe amplitude at the same longitudinal distance, for 154 wave runs across 1 panel configurations. A ratio of 1 indicates lateral symmetry. Deviations indicate wall reflections or wind-driven lateral asymmetry. Error bars: standard deviation across runs at the same frequency. Dashed line: ratio = 1.
"""

_pv_parallel_ratio = {
    "filters": {},
    "plotting": {
        "show_plot": True,
        "save_plot": True,            # DRAFT — parallel ratio not yet polished
        "draft":     True,
        "figure_name": "ch04_parallel_ratio",
        "force_stub": True,
        "caption": (
            "Ratio of wall-side to far-side probe amplitude at the same longitudinal "
            "distance, for {n_runs} wave runs across {n_panels} panel condition(s) "
            "({panel_conditions}). "
            "A ratio of 1 indicates lateral symmetry. "
            "Deviations indicate wall reflections or wind-driven lateral asymmetry. "
            r"Error bars: standard deviation across runs at the same frequency. "
            "Dashed line: ratio = 1."
        ),
    },
}
start = time.perf_counter()
_fig_pr = plot_parallel_ratio(combined_meta, _pv_parallel_ratio)
end = time.perf_counter()
print(f"Lateral symmetry of plot_parallel_ratio {end-start:.4f} seconds")

_pv_parallel_ratio_scatter = {
    "filters": {},
    "plotting": {
        **_pv_parallel_ratio["plotting"],
        "scatter":     True,
        "figure_name": "ch04_parallel_ratio_scatter",
        "caption": (
            "Same data as the parallel-ratio figure, plotted as individual run "
            "points (no grouping). Each dot = one run. Colour = wind condition; "
            "marker shape = wave amplitude. Use to identify outlier runs."
        ),
    },
}
plot_parallel_ratio(combined_meta, _pv_parallel_ratio_scatter)

# %%
"""
── CH04 § 3b — Probe height & range-mode validity ────────────────────────────
Goal: characterise how probe height above the water surface and the
hardware range mode affect (a) the stillwater noise floor and (b) the
wind-background amplitude per probe. Four hardware conditions:

    cond1: h272/high   standard, pre-2026-03-23 (longest air path)
    cond2: h136/high   transitional (1 day, n=1 stillwater)
    cond3: h100/high   WRONG mode (h is below the 130 mm window minimum)
    cond4: h100/low    correct lowrange — used for all CH05 results

Result: cond1 has ~3× higher stillwater noise at the IN probe than cond4,
consistent with longer-path acoustic attenuation. The OUT probe is
sheltered by the panel and shows ~10× lower wind background than the
wind-exposed probes (IN, parallel, upstream). cond3 is the source of
the P2-probe-malfunction runs flagged by the pipeline.

Generated by analysis_scratch/probe_height_figure.py. Writes PDF + stub
directly into output/. Re-run after a recompute changes amplitude
columns. Full per-condition analysis text:
    analysis_scratch/probe_height_wind_findings.md (rewritten 2026-04-17)
"""

_ch04_ph_fig  = Path("output/FIGURES/ch04_probe_height.pdf")
_ch04_ph_stub = Path("output/TEXFIGU/ch04_probe_height.tex")
if not (_ch04_ph_fig.exists() and _ch04_ph_stub.exists()):
    print("  ch04_probe_height missing — run "
          "`python analysis_scratch/probe_height_figure.py` to generate it.")
else:
    print(f"  ch04_probe_height: figure OK → {_ch04_ph_fig}")

# %%
"""
── CH04 § 3c — Mooring rubber band length: loose230 vs loose300 ─────────────
Scientific result: rubber band length (230 mm vs 300 mm, below-water at −90 mm)
has NO detectable systematic effect on OUT/IN(FFT) within the main thesis
range. Across 1.4–1.6 Hz at 0.2/0.3 V with matched wind conditions, the
two moorings agree to within |Δ| ≤ 5.9 %. The 1.3 Hz no-wind loose230
outlier is a mooring-independent standing-wave artefact (IN probe near a
pressure node), documented separately. Merge both as `below_90_loose`.

Generated by analysis_scratch/mooring_comparison.py. Writes PDF + stub
directly into output/. Full per-condition table (including the 0.1 V
low-SNR caveats and 1.3 Hz anomaly):
    analysis_scratch/mooring_comparison_findings.md
"""

_ch04_mc_fig  = Path("output/FIGURES/ch04_mooring_comparison.pdf")
_ch04_mc_stub = Path("output/TEXFIGU/ch04_mooring_comparison.tex")
if not (_ch04_mc_fig.exists() and _ch04_mc_stub.exists()):
    print("  ch04_mooring_comparison missing — run "
          "`python analysis_scratch/mooring_comparison.py` to generate it.")
else:
    print(f"  ch04_mooring_comparison: figure OK → {_ch04_mc_fig}")

# %%
"""
── CH04 § 3d — Speed-of-sound / lab temperature drift ───────────────────────
Goal: show that lab temperature variation introduces < 0.4 % amplitude scale
error, and that this cancels exactly for OUT/IN ratios.
Data: sound_speed_mean_ms / sound_speed_std_ms in combined_meta (pipeline).
"""

_pv_sound_speed = {
    "filters": {},
    "plotting": {
        "show_plot":   True,
        "save_plot":   True,            # DRAFT — not yet polished
        "draft":       True,
        "figure_name": "ch04_sound_speed",
        "force_stub":  True,
        "figsize":     (10, 3),
    },
}

plot_sound_speed(combined_meta, _pv_sound_speed, chapter="04")

# %%
"""
── CH04 § 3e — Parallel-probe agreement (mean-IN validation) ────────────────
Methodology support for the CH05 decision to use mean(9373/170, 9373/340)
as the canonical IN reference. Shows that the two probes — at the same
longitudinal distance from the paddle — agree to within ±5% for nowind
and ±10% for fullwind 0.2–0.3 V runs across the thesis band. The
fullwind 0.1 V regime has larger probe-to-probe scatter (low SNR,
wind contamination dominates each probe differently). That's exactly
where the mean is most valuable (reduces single-probe noise).

Result: 62/78 thesis-scope runs consistent at 10% threshold. Mean IN
is justified for the headline result; single-run outliers (like the
0.3 V nowind H&G-window dip in analysis_scratch/huseby_grue_window.pdf)
are absorbed by averaging.

Generated by analysis_scratch/parallel_probe_agreement.py. Writes PDF
+ stub directly into output/.
"""

_ch04_ppa_fig  = Path("output/FIGURES/ch04_parallel_probe_agreement.pdf")
_ch04_ppa_stub = Path("output/TEXFIGU/ch04_parallel_probe_agreement.tex")
if not (_ch04_ppa_fig.exists() and _ch04_ppa_stub.exists()):
    print("  ch04_parallel_probe_agreement missing — run "
          "`python analysis_scratch/parallel_probe_agreement.py` to generate it.")
else:
    print(f"  ch04_parallel_probe_agreement: figure OK → {_ch04_ppa_fig}")

# _pv_probe_height = {
#     "filters": {"run_category": "standard"},
#     "plotting": {
#         "show_plot": False,
#         "save_plot": False,
#         "figure_name": "ch04_probe_height",
#         "caption": "TODO",
#     },
# }

# %%
"""
── CH04 § 4-1 — Wind characterisation ─────────────────────────────────────────
Goal: characterise what the wind does to the water surface — spectrum, spatial
extent, interaction with the panel.

Subtopics:
  4a. Wind-wave PSD at each probe (broadband, 2–10 Hz dominant)
  4b. Wind-only amplitude vs probe position (SNR context)
  4c. Wind-only amplitude: IN probe (~10 mm) vs OUT probe (~0.9 mm) —
      panel attenuates wind waves almost completely at 12400 mm
  4d. Lateral coherence: cross-correlate /170 and /340 at same distance
      (coherent = tank-wide fetch; incoherent = local turbulence)

Data: combined_psd_dict (nowave entries), nowave+fullwind rows of combined_meta.

Figures:
  - Plot:  wind PSD per probe, fullwind vs stillwater overlay (log y-axis)
  - Plot:  wind-only amplitude vs longitudinal distance, bar per probe
  - Plot:  cross-correlation coefficient /170 vs /340 for fullwind runs
"""
from wavescripts.filters import apply_experimental_filters as _aef

_pv_wind_psd = {
    "filters": {
        "WaveFrequencyInput [Hz]": None,
        "WindCondition":           None,
        "PanelCondition":          None,
        # exclude diagnostic/experimental runs by filename keyword
        "exclude_run_keywords": ["nestenstille", "mstop"],
    },
    "plotting": {
        "show_plot":     True,
        "save_plot":     True,          # set True when ready
        "figure_name":   "ch04_wind_psd",
        "force_stub":    True,
        "figsize":       (11, 4 * 4),
        "linewidth":     1.0,
        "facet_by":      "probe",
        "probes":        ANALYSIS_PROBES,
        "xlim":          (0, 5),
        "logaritmic":    False,
        "peaks":         0,
        "max_points":    500,
        "grid":          True,
        "legend":        "inside",

        "caption": (
            "POWER spectral density (PSD) of the free surface at each wave gauge "
            "during wind-only runs (no paddle waves). "
            "All {n_runs} nowave runs overlaid; colour encodes wind condition. "
            "Stillwater runs (no wind) shown as baseline. "
            "Wind energy is concentrated above 2\\,Hz — "
            "the paddle frequency range (0.65--1.9\\,Hz) is unaffected."
        ),
    },
}

_meta_nowave_all = combined_meta[combined_meta["WaveFrequencyInput [Hz]"].isna()].copy()
_meta_nowave     = _aef(_meta_nowave_all, _pv_wind_psd)
_nowave_paths    = set(_meta_nowave["path"])
_wind_psd_dict   = {k: v for k, v in combined_psd_dict.items() if k in _nowave_paths}

start = time.perf_counter()
_fig_wind_psd, _ = plot_frequency_spectrum(
    _wind_psd_dict, _meta_nowave, _pv_wind_psd, data_type="psd", chapter="04"
)
end = time.perf_counter()
print(f"Wind PSD plot took {end - start:.4f} s")



# %%
"""
── CH04 § 4-2 — Wind wave-reflection from panel ─────────────────────────────────────────
Goal: find out the reflection — spectrum, spatial
extent, interaction with the panel.

Data: combined_psd_dict (nowave entries), nowave+fullwind rows of combined_meta.

Figures:
  - Plot:
"""
# TODO: implement wind reflection figure
_save_placeholder("ch04_wind_reflection", "CH04 §4-2 — Wind reflection from panel", chapter="04")

# %%
"""
── CH04 § 4-3 — FFT spectrum: paddle frequency peak ────────────────────────
Goal: show what the FFT looks like for a wave run — narrow peak at the paddle
frequency, wind condition overlaid. Motivates using FFT amplitude (not
time-domain) for OUT/IN. One representative frequency (e.g. 1.3 Hz).

Data: combined_fft_dict, wave runs.

Figures:
  - plot_frequency_spectrum with data_type="fft", facet_by="probe"
"""

_pv_fft_wave = {
    "filters": {
        "WaveAmplitudeInput [Volt]": 0.2,
        "WaveFrequencyInput [Hz]":   1.3,
        "WindCondition":             None,
        "PanelCondition":            "full",
        "run_category":              "standard",
    },
    "plotting": {
        "show_plot":   True,
        "save_plot":   True,           # DRAFT — FFT wave example not yet polished
        "draft":       True,
        "figure_name": "ch04_fft_wave",
        "force_stub":  True,
        "figsize":     (11, 4 * 4),
        "linewidth":   0.8,
        "facet_by":    "probe",
        "probes":      ANALYSIS_PROBES,
        "xlim":        (0, 5),
        "logaritmic":  False,
        "peaks":       3,
        "max_points":  500,
        "grid":        True,
        "legend":      "inside",
        "caption": (
            "FFT amplitude spectrum of the free surface during wave runs "
            "(paddle frequency 1.3\\,Hz, amplitude 0.2\\,V, full panel). "
            "Each panel shows one probe; colour encodes wind condition. "
            "The narrow paddle-frequency peak is the target signal used "
            "for OUT/IN ratio computation."
        ),
    },
}

_fft_wave_meta = _aef(combined_meta, _pv_fft_wave)
_fft_wave_paths = set(_fft_wave_meta["path"])
_fft_wave_dict  = {k: v for k, v in combined_fft_dict.items() if k in _fft_wave_paths}

_fig_fft_wave, _ = plot_frequency_spectrum(
    _fft_wave_dict, _fft_wave_meta, _pv_fft_wave, data_type="fft", chapter="04"
)

# %%
"""
── CH04 § 4-4 — Spectral SNR: paddle signal vs wind noise ───────────────────
Goal: quantify how much of the FFT amplitude at paddle frequencies is
wind noise. SNR < 5 = unreliable; SNR < 3 = dominated by wind.
Data: combined_meta (wave runs + FFT amplitudes) + combined_psd_dict (nowave PSDs).
TODO: ... the 1.3 hz wave is noticably different.. but why, this is the run i have waay more data on .somehing is wrong.
"""

_pv_wind_snr = {
    "filters": {
        "WaveAmplitudeInput [Volt]": None,
        "WaveFrequencyInput [Hz]":   None,
        "WindCondition":             None,
        "PanelCondition":            None,
    },
    "plotting": {
        "show_plot":      True,
        "save_plot":      True,         # DRAFT — not yet polished
        "draft":          True,
        "figure_name":    "ch04_wind_snr",
        "force_stub":     True,
        "probes":         ANALYSIS_PROBES,
        "fft_window_hz":  0.1,
    },
}

plot_wind_snr(combined_meta, combined_psd_dict, _pv_wind_snr, chapter="04")

# %%
"""
── CH04 § 4-5 — Time-domain vs FFT amplitude: why A_FFT is required ─────────
Goal: demonstrate that time-domain amplitude is wind-dominated at the IN probe
under full wind, making OUT/IN from A_td meaningless. FFT amplitude isolates
the paddle frequency and is unaffected by broadband wind energy.
Data: combined_meta wave rows (Probe {pos} Amplitude and Probe {pos} Amplitude (FFT)).
"""

_pv_td_vs_fft = {
    "filters": {
        "min_periods":               10,
        "WaveAmplitudeInput [Volt]": None,
        "WaveFrequencyInput [Hz]":   None,
        "WindCondition":             None,
        "PanelCondition":            None,
    },
    "plotting": {
        "show_plot":   True,
        "save_plot":   True,            # DRAFT — not yet polished
        "draft":       True,
        "figure_name": "ch04_td_vs_fft",
        "force_stub":  True,
        "probes":      ANALYSIS_PROBES,
    },
}

plot_td_vs_fft(combined_meta, _pv_td_vs_fft, chapter="04")

_pv_td_vs_fft_scatter = {
    "filters": {**_pv_td_vs_fft["filters"]},
    "plotting": {
        **_pv_td_vs_fft["plotting"],
        "scatter":     True,
        "figure_name": "ch04_td_vs_fft_scatter",
        "caption": (
            "Same data as the td-vs-fft figure. "
            "Bottom row: individual run ratios as scatter (no median line). "
            "Use to identify outlier runs driving dips at specific kL."
        ),
    },
}
plot_td_vs_fft(combined_meta, _pv_td_vs_fft_scatter, chapter="04")

# %%
"""
── CH04 § 4b — FFT peak-bin bias cancels in OUT/IN ──────────────────────────
Goal: establish that although nearest-bin FFT amplitudes are biased by
paddle-drift vs bin-grid alignment (sinc attenuation up to ~40% for
individual amplitudes), the OUT/IN ratio is robust because IN and OUT
probes use matching analysis window lengths → same bin grid → bias
cancels. Empirical: mean |Δ(OUT/IN)|/OUT/IN < 0.5% across ~360 runs.

Data: precomputed CSV at analysis_scratch/fft_peak_bias_outin_impact.csv
(generated by analysis_scratch/fft_peak_bias_outin_impact.py, which
loads processed_dfs). If the CSV is missing or stale, re-run that script.

This figure documents the headline methodology safeguard: the thesis's
primary OUT/IN (FFT) result is not an artifact of FFT binning.
"""

_pv_fft_peak_bias = {
    "filters": {},
    "plotting": {
        "show_plot":   True,
        "save_plot":   True,            # DRAFT — polish on review
        "draft":       True,
        "figure_name": "ch04_fft_peak_bias_cancellation",
        "force_stub":  True,
    },
}
plot_fft_peak_bias_cancellation(_pv_fft_peak_bias, chapter="04")

# %%
"""
── CH04 § 4c — Mansard-Funke reflection coefficient ─────────────────────────
Goal: direct measurement of the panel's reflection coefficient R using the
two-probe Mansard-Funke method on 8804/250 (upstream) + 9373/170 (IN probe).

Result: R ≈ 0.05-0.07 across all moorings at 0.2 V nowind — well below the
R = 0.20 value that was previously assumed. Supports the decision to NOT
apply a standing-wave correction to OUT/IN (FFT).

The figure is generated by analysis_scratch/mansard_funke.py (which also
writes the .tex stub directly to output/TEXFIGU/). If the figure needs
refreshing after a recompute, re-run that script. This cell is a
no-op placeholder that documents the figure's existence in the thesis
cell order.
"""

_ch04_mf_fig  = Path("output/FIGURES/ch04_mansard_funke_reflection.pdf")
_ch04_mf_stub = Path("output/TEXFIGU/ch04_mansard_funke_reflection.tex")
if not (_ch04_mf_fig.exists() and _ch04_mf_stub.exists()):
    print("  ch04_mansard_funke_reflection missing — run "
          "`python analysis_scratch/mansard_funke.py` to generate it.")
else:
    print(f"  ch04_mansard_funke_reflection: figure OK → {_ch04_mf_fig}")

# %%
"""
── CH04 § 4d — Standing-wave correction test (negative evidence) ────────────
Goal: show that applying a standing-wave correction at R = 0.20 to the raw
OUT/IN(FFT) curve creates a violent zigzag not present in the raw data.
This is the complementary result to §4c — raw data shows no node/antinode
fingerprint at the predicted frequency spacing, putting an upper bound of
R ≲ 0.05 on the panel reflection coefficient.

Generated by analysis_scratch/sw_correction.py. Writes figure + stub
directly to output/. Re-run after any pipeline change that alters OUT/IN.
"""

_ch04_sw_fig  = Path("output/FIGURES/ch04_sw_correction_test.pdf")
_ch04_sw_stub = Path("output/TEXFIGU/ch04_sw_correction_test.tex")
if not (_ch04_sw_fig.exists() and _ch04_sw_stub.exists()):
    print("  ch04_sw_correction_test missing — run "
          "`python analysis_scratch/sw_correction.py` to generate it.")
else:
    print(f"  ch04_sw_correction_test: figure OK → {_ch04_sw_fig}")

# %%
"""
── CH04 § 4e — Sliding-window FFT stability at the IN probe ─────────────────
Goal: show that the paddle-frequency FFT amplitude at the IN probe
(9373/170) is stable across the run for fullwind per240 conditions. Any
apparent discrepancies between the pipeline AFFT (short analysis window)
and alternative FFT windows reflect FFT bin-grid alignment (see §4b),
not physical within-run transients. This is a companion figure to §4b —
§4b establishes the bias mechanism and OUT/IN cancellation; §4e shows
that the underlying signal is genuinely stable (so the bias is the only
suspect when short/long-window FFT values disagree).

Generated by analysis_scratch/sliding_afft_fullwind_sweep.py. Writes
PDF + stub directly into output/. A deeper 1.5 Hz / 0.2 V zoom lives
in analysis_scratch/sliding_afft_15hz_02v_zoom.{py,pdf} for reference.
"""

_ch04_sa_fig  = Path("output/FIGURES/ch04_sliding_afft_stability.pdf")
_ch04_sa_stub = Path("output/TEXFIGU/ch04_sliding_afft_stability.tex")
if not (_ch04_sa_fig.exists() and _ch04_sa_stub.exists()):
    print("  ch04_sliding_afft_stability missing — run "
          "`python analysis_scratch/sliding_afft_fullwind_sweep.py` to generate it.")
else:
    print(f"  ch04_sliding_afft_stability: figure OK → {_ch04_sa_fig}")

# %%
"""
── CH04 § 4f — Reconstruction A vs B: wind-separation safety check ──────────
Goal: justify the peak-bin FFT reconstruction (method A) that underpins the
A_FFT thesis metric by showing it carries the same wind-band (2–6 Hz)
residual energy as a full band-integrated reconstruction (method B).

Result: across every thesis-scope run (f in [1.3, 1.6] Hz, n=156 probe×run
pairs), (A-B)/B = 0 in wind-band energy and A_B/A_A = 1.0000 in amplitude —
identical to four-decimal precision. Paddle-band sinc-leakage that A misses
stays inside the paddle band; wind characterisation based on the method-A
residual is equivalent to one based on method B.

Generated by analysis_scratch/reconstruction_A_vs_B.py. Writes PDF + stub
directly into output/. Full analysis:
    analysis_scratch/reconstruction_A_vs_B_findings.md
"""

_ch04_rab_fig  = Path("output/FIGURES/ch04_reconstruction_AvsB.pdf")
_ch04_rab_stub = Path("output/TEXFIGU/ch04_reconstruction_AvsB.tex")
if not (_ch04_rab_fig.exists() and _ch04_rab_stub.exists()):
    print("  ch04_reconstruction_AvsB missing — run "
          "`python analysis_scratch/reconstruction_A_vs_B.py` to generate it.")
else:
    print(f"  ch04_reconstruction_AvsB: figure OK → {_ch04_rab_fig}")

# %%
"""
── CH04 § 4g — Pure-wind PSD via no-wind residual subtraction ───────────────
Goal: quantify the Stokes-harmonic contamination of the 2–6 Hz "wind band"
on wave runs. At thesis paddle frequencies (1.3–1.6 Hz), the 2f/3f/4f
harmonics fall directly inside 2–6 Hz. The naive "wind energy" is therefore
wind + Stokes; subtracting a matched no-wind residual cancels the Stokes
contribution and leaves pure wind.

Result: Stokes removes a median 14 % of the naive wind-band energy at IN
and 23 % at OUT, up to 59 % at the IN / 0.3 V high-frequency corner. Any
wind metric that integrates a wave run's residual over 2–6 Hz without this
subtraction mis-attributes paddle Stokes as wind.

Generated by analysis_scratch/reconstruction_A_vs_B.py (same script as
§4f, bottom half). Writes PDF + stub directly into output/. Full analysis:
    analysis_scratch/reconstruction_pure_wind_findings.md
"""

_ch04_pw_fig  = Path("output/FIGURES/ch04_reconstruction_pure_wind.pdf")
_ch04_pw_stub = Path("output/TEXFIGU/ch04_reconstruction_pure_wind.tex")
if not (_ch04_pw_fig.exists() and _ch04_pw_stub.exists()):
    print("  ch04_reconstruction_pure_wind missing — run "
          "`python analysis_scratch/reconstruction_A_vs_B.py` to generate it.")
else:
    print(f"  ch04_reconstruction_pure_wind: figure OK → {_ch04_pw_fig}")

# %%
"""
── CH04 § 5 — What does a full signal look like? ────────────────────────────
Goal: show the full signal for a select few runs — stillwater baseline,
wavemaker ramp, stable wavetrain, decay. Wind-wave noise visible at IN probe
vs clean signal at OUT probe.

Layout: rows = probes, columns = runs selected by filters.
Grey band = detected stable-window used for all amplitude/FFT analysis.
"""

_pv_timeseries = {
    "filters": {
        # Pick a representative condition — adjust as needed:
        "WaveFrequencyInput [Hz]": 1.3,
        "WaveAmplitudeInput [Volt]": 0.2,
        "WindCondition": None,      # None = all wind conditions
        "PanelCondition": "full",
        # "run_category": "standard",
    },
    "plotting": {
        "show_plot":   True,
        "save_plot":   True,           # DRAFT — timeseries overview not yet polished
        "draft":       True,
        "figure_name": "ch04_timeseries_overview",
        "force_stub":  True,
        "probes":      ["9373/170", "12400/250"],   # IN and OUT only
        "max_runs":    4,           # cap columns; reduce if too crowded
        "xlim":        None,        # e.g. (0, 60) to zoom; None = full run
        "ylim":        None,        # e.g. (-30, 30); None = auto per row
        # caption printed on first run — paste the one-liner here:
        # "caption": "...",
    },
}

_fig_ts = plot_timeseries_overview(combined_meta, processed_dfs, _pv_timeseries)

# %%
"""
── CH04 § 6 — Wave-range detection ──────────────────────────────────────────
Goal: explain and validate _SNARVEI_CALIB. Show how the stable wavetrain
window is detected: (1) threshold crossing, (2) ramp-up skip, (3) n periods.

Data: processed_dfs, Computed Probe {pos} start/end columns.

Figures:
  - Plot:  single run with detected start/end marked, one probe panel per row
  - Plot:  start sample vs frequency (all probes) — show _SNARVEI_CALIB points
"""

_pv_first_arrival = {
    "filters": {},
    "plotting": {
        "show_plot":        True,
        "save_plot":        True,       # DRAFT — threshold not yet calibrated
        "draft":            True,
        "figure_name":      "ch04_first_arrival",
        "force_stub":       True,
        "probes":           ANALYSIS_PROBES,
        "threshold_factor": 5.0,        # TODO: calibrate per-probe after noise floor analysis
        "window_s":         2.5,
        "min_arrival_s":    0.5,
        "figsize":          (9, 5),
    },
}

plot_first_arrival(combined_meta, processed_dfs, _pv_first_arrival, chapter="04")

# %%
"""
── CH04 § 7 — Autocorrelation A: wavetrain stability ────────────────────────
Goal: show wave_stability and period_cv as quality metrics. Demonstrate that
fullwind + low amplitude (0.1 V) degrades IN probe stability, while OUT probe
stays clean.

Data: combined_meta, wave_stability {pos} and period_cv {pos} columns.

Figures:
  - Plot:  wave_stability vs frequency, faceted by probe, coloured by wind
  - Plot:  period_cv vs frequency, same layout
  - Note:  this motivates use of FFT amplitude (not time-domain) for OUT/IN
"""

_pv_wave_stability = {
    "filters": {
        "min_periods":               10,
        "WaveAmplitudeInput [Volt]": None,
        "WaveFrequencyInput [Hz]":   (0.9,1.6),
        "WindCondition":             None,
        "PanelCondition":            "full",
        # "run_category": "standard",   # re-enable after --force-recompute
    },
    "plotting": {
        "show_plot":   True,
        "save_plot":   True,          # DRAFT — wave stability not yet polished
        "draft":       True,
        "figure_name": "ch04_wave_stability",
        "force_stub":  True,
        "figsize":     (10, 3.5),
        "probes":      ANALYSIS_PROBES,
        # caption printed to terminal on first run — paste the one-liner here:
        # "caption": "...",
    },
}

_fig_stab = plot_wave_stability(combined_meta, ANALYSIS_PROBES, _pv_wave_stability)

# %%
"""
── CH04 § 8 — Autocorrelation B: lateral wave equality ──────────────────────
Goal: show that the paddle wave is laterally uniform (parallel probes agree)
under no-wind conditions, and that full wind introduces lateral asymmetry.

Data: combined_meta, parallel_ratio column, wave_stability columns.

Figures:
  - Plot:  parallel_ratio vs frequency, no-wind runs (should be ~1.0)
  - Plot:  parallel_ratio vs frequency, fullwind runs (asymmetry visible?)
  - Table: mean parallel_ratio ± std by (WindCondition, frequency)
"""

# Lateral equality uses the same plot_parallel_ratio function (already defined in §3),
# but filtered to a single wind condition at a time for the per-wind breakdown.
_pv_lateral_nowind = {
    "filters": {"WindCondition": "no", "run_category": "standard"},
    "plotting": {
        "show_plot":   True,
        "save_plot":   True,          # DRAFT — lateral equality not yet polished
        "draft":       True,
        "figure_name": "ch04_lateral_nowind",
        "force_stub":  True,
        "caption": (
            "Wall-side to far-side amplitude ratio at matched longitudinal distance, "
            "no-wind runs only. A ratio of 1 indicates the paddle wave is laterally "
            r"uniform. Dashed line: ratio = 1."
        ),
    },
}
_fig_lat_nw = plot_parallel_ratio(combined_meta, _pv_lateral_nowind)

_pv_lateral_nowind_scatter = {
    "filters": {**_pv_lateral_nowind["filters"]},
    "plotting": {
        **_pv_lateral_nowind["plotting"],
        "scatter":     True,
        "figure_name": "ch04_lateral_nowind_scatter",
        "caption": (
            "Same data as the lateral-nowind figure, plotted as individual run "
            "points (no grouping). Each dot = one run. Use to identify outliers."
        ),
    },
}
plot_parallel_ratio(combined_meta, _pv_lateral_nowind_scatter)

# %% - perhaps skip this one. its the physical plot.
# """
# ── CH04 § 9 — Amplitude profile across all probes ───────────────────────────
# Goal: show measured amplitude at each probe position for all runs, giving a
# # physical overview of how wave energy is distributed along the tank.
# Colour = wind condition, linestyle = panel condition.
# Data: combined_meta wave rows, all Probe {pos} Amplitude columns.
# """

# _pv_all_probes = {
#     "filters": {
#         "WaveAmplitudeInput [Volt]": None,
#         "WaveFrequencyInput [Hz]":   None,
#         "WindCondition":             None,
#         "PanelCondition":            None,
#     },
#     "plotting": {
#         "show_plot":   True,
#         "save_plot":   False,            # this one is mostly
#         "draft":       True,
#         "figure_name": "ch04_amplitude_profile",
#         "force_stub":  True,
#         "figsize":     (10, 6),
#         "annotate":    False,
#     },
# }

# _ap_meta = apply_experimental_filters(
#     combined_meta[combined_meta["WaveFrequencyInput [Hz]"].notna()], _pv_all_probes
# )
# plot_all_probes(_ap_meta, _pv_all_probes, chapter="04")

# =============================================================================
# CHAPTER 05 — RESULTS
# =============================================================================

# %%
"""
── CH05 § 1 — Damping overview: OUT/IN vs frequency ─────────────────────────
THE central result. "How much is left of the paddle-frequency wave after
travelling through the panel geometry?"

x-axis:  WaveFrequencyInput [Hz]  (or ka — wavenumber × amplitude)
y-axis:  OUT/IN (FFT) — always FFT-based, never time-domain
colour:  WindCondition  (no / lowest / full)
facets:  PanelCondition  ×  WaveAmplitudeInput [Volt]

Key question: does wind change the damping? If yes: how much, and at which
frequencies?

Data: combined_meta → damping_all_amplitude_grouper → plot_damping_freq.

Figures:
  - plot_damping_freq: OUT/IN vs freq, errorbars (std or ±10% for n=1)
  - Same plot with ka on x-axis (requires wavenumber column)
"""

_pv_damping_freq = {
    "filters": {
        "WaveAmplitudeInput [Volt]": (0.1, 0.3),
        # Thesis scope: 1.3–1.6 Hz inclusive. Below 1.3 Hz the two
        # validated lowrange folders have no nowind runs (fullwind-only
        # → no wind-effect comparison possible). Above 1.6 Hz the
        # 0.2/0.3 V data is unreliable (dropout zone) and the lone 1.7
        # Hz/0.30V point has no nowind counterpart. Cross-condition
        # patterns (all dates, all probe configs) are shown in the
        # supplementary all-data scatter (CH05 §7). See memory/MEMORY.md
        # "Scope boundary".
        "WaveFrequencyInput [Hz]":   (1.3, 1.6),
        "WindCondition":             None,
        "PanelCondition":            None,
    },
    "plotting": {
        "show_plot":  True,
        "save_plot":  True,          # set True when figure is ready for thesis
        "force_stub": True,
        "figure_name": "ch05_damping_freq",
        "figsize":    (7, 3),
        "annotate":   True,
        "legend":     "outside_right",
        "probes":     ANALYSIS_PROBES,
        "caption": (
            "Damping ratio OUT/IN (FFT amplitude at paddle frequency) versus wave frequency. "
            "Colour encodes wind condition ({wind_conds}); "
            "each line shows one amplitude ({amps}). "
            "Errorbars: standard deviation across repeated runs. "
            "Dashed line: ratio = 1 (no damping)."
        ),
    },
}

_damping_meta   = _aef(meta_results, _pv_damping_freq)
_damping_grouped = damping_all_amplitude_grouper(_damping_meta)
plot_damping_freq(_damping_grouped, _pv_damping_freq)

# %%
"""
── CH05 § 2 — Damping vs amplitude ──────────────────────────────────────────
Secondary result. Is there an amplitude dependence? (Expected: small effect
at these steepnesses, but worth showing explicitly.)

x-axis:  WaveAmplitudeInput [Volt]
y-axis:  OUT/IN (FFT)
colour:  WaveFrequencyInput [Hz]
facets:  PanelCondition  ×  WindCondition

Figures:
  - plot_damping_scatter or similar
"""

_pv_damping_scatter = {
    "filters": {
        "WaveAmplitudeInput [Volt]": None,
        # Thesis scope: 1.3–1.6 Hz (see _pv_damping_freq).
        "WaveFrequencyInput [Hz]":   (1.3, 1.6),
        "WindCondition":             None,
        "PanelCondition":            None,
        # "run_category":            "standard",   # re-enable after --force-recompute
    },
    "plotting": {
        "show_plot":   True,
        "save_plot":   True,         # set True when figure is ready for thesis
        "figure_name": "ch05_damping_scatter",
        "force_stub":  True,
        "figsize":     (5, 4),
        "caption": "UT/INN damping ratio versus wave frequency, all amplitudes combined. all panel condition(s); colour = wind condition (full, no); marker size = wave amplitude (0.10\,V, 0.20\,V, 0.30\,V, 0.60\,V). Errorbars: standard deviation across runs."
}}

_scatter_meta   = _aef(meta_results, _pv_damping_scatter)
_scatter_grouped = damping_all_amplitude_grouper(_scatter_meta)
plot_damping_scatter(_scatter_grouped, _pv_damping_scatter)

# %%
"""
── CH05 § 3 — Wind effect on damping ────────────────────────────────────────
The single key question of the thesis, isolated:
"Does adding wind increase or decrease damping by the panel?"

Show: OUT/IN (no wind) vs OUT/IN (full wind) at matched frequency/amplitude/panel.
Expected: wind may add energy at IN → apparent increase in damping if using
time-domain; FFT-based OUT/IN removes this artefact and shows the true effect.

Figures:
  - Plot:  delta-OUT/IN (full wind minus no wind) vs frequency
  - Table: OUT/IN summary — (frequency × panel) with wind as columns
"""

_pv_damping_wind_delta = {
    "filters": {
        "WaveAmplitudeInput [Volt]": None,
        # Thesis scope: 1.3–1.6 Hz (see _pv_damping_freq).
        "WaveFrequencyInput [Hz]":   (1.3, 1.6),
        "WindCondition":             ["no", "full"],
        "PanelCondition":            None,
    },
    "plotting": {
        "show_plot":   True,
        "save_plot":   True,
        "draft":       True,
        "figure_name": "ch05_damping_wind_delta",
        "force_stub":  True,
        "figsize":     (6, 5),
        "ref_wind":    "no",
        "target_wind": "full",
    },
}

_wind_delta_meta    = _aef(meta_results, _pv_damping_wind_delta)
_wind_delta_grouped = damping_all_amplitude_grouper(_wind_delta_meta)
plot_damping_wind_delta(_wind_delta_grouped, _pv_damping_wind_delta, chapter="05")

# %%
"""
── CH05 § 3b — T_cross: wind effect via clean nowind reference ──────────────
Alternative wind-effect metric. Plots three transmission curves per
amplitude:

    (OUT/IN)_nw   = A_out^nw / A_in^nw    (blue, clean baseline)
    T_cross       = A_out^fw / A_in^nw    (green, honest wind effect)
    (OUT/IN)_fw   = A_out^fw / A_in^fw    (red, standard metric)

Green uses the *clean* nowind IN amplitude as reference, so wind
contamination of the IN probe FFT (see CLAUDE.md §16) doesn't corrupt
the denominator. The green–blue gap is the honest wind effect; the
red–blue gap is what the standard metric shows. Where they disagree
(mostly at 1.5–1.6 Hz), the standard metric underestimates the true
wind effect by up to ~50% because A_in^fw is inflated.

Prerequisite: the paddle output (A_in) must be approximately
wind-independent. Validated at 0.2 V / 0.3 V within ±15% across the
thesis band; noisier at 0.1 V (low SNR).

Generated by analysis_scratch/t_cross_figure.py. Writes three PDFs
(one per amplitude) + a subfigure stub.
"""

_ch05_tcross_stub = Path("output/TEXFIGU/ch05_t_cross.tex")
_ch05_tcross_pdfs = [Path(f"output/FIGURES/ch05_t_cross_{v}V.pdf") for v in ("10", "20", "30")]
if not (all(p.exists() for p in _ch05_tcross_pdfs) and _ch05_tcross_stub.exists()):
    print("  ch05_t_cross missing — run "
          "`python analysis_scratch/t_cross_figure.py` to generate it.")
else:
    print(f"  ch05_t_cross: 3 figures OK → {_ch05_tcross_pdfs[0].parent}")

# %%
"""
── CH05 § 4 — Wave steepness: ka as axis variable ───────────────────────────
All damping plots should optionally show ka on the x-axis instead of Hz.

ka is not pre-calculated from dispersion — it is found per probe per run from
the measured wavenumber k (from FFT phase or zero-crossing period) and the
measured amplitude a at that probe position. This reflects the actual wave
seen by each probe, not the theoretical incident wave.

Note: the panel changes both amplitude AND ka between IN and OUT. The
frequency changes little, but amplitude can drop up to ~95%. Report both
IN-side ka and OUT-side ka.

Requires: wavenumber column in combined_meta (computed in processor2nd).
"""
# IN ka (FFT) is fully populated (108/108 wave runs, verified 2026-04-17).
# plot_damping_ka takes raw meta_df, no grouper needed — each run is one scatter point.
_pv_damping_ka = {
    "filters": {
        "PanelCondition": "full",
        # Thesis scope: 1.3–1.6 Hz (see _pv_damping_freq).
        "WaveFrequencyInput [Hz]":   (1.3, 1.6),
        "min_periods": 10,
    },
    "plotting": {
        "show_plot": False,
        "save_plot": True,
        "draft": True,
        "figure_name": "ch05_damping_ka",
        "force_stub": False,
        "caption": (
            "OUT/IN damping ratio versus wave steepness $ka$ at the incident probe "
            "(9373/170), full panel condition. "
            "Colour encodes wind condition; marker encodes wave amplitude. "
            "Each point is one run. Dashed line: ratio = 1 (no damping). "
            "Data: two validated sessions (2026-03-26/27, lowrange mode)."
        ),
    },
}

_ka_meta = apply_experimental_filters(meta_results, _pv_damping_ka)
from wavescripts.plotter import plot_damping_ka
plot_damping_ka(_ka_meta, _pv_damping_ka, chapter="05")

# Per-amplitude variants of the same plot — one figure per input voltage
# (0.10/0.20/0.30 V). Shared x/y limits across the three so they read as
# panels of the same underlying figure. Same filtered data as the all-amps
# overview above.
_pv_damping_ka_by_amp = {
    **_pv_damping_ka,
    "plotting": {
        **_pv_damping_ka["plotting"],
        "figure_name": "ch05_damping_ka_by_amp",
        "facet_by_amp": True,
        "force_stub": True,
        "caption": (
            "OUT/IN damping ratio versus wave steepness $ka$ at the incident probe "
            "(9373/170), full panel condition, split by input amplitude "
            "(0.10/0.20/0.30\\,V). Colour encodes wind condition. Axes are "
            "shared across the three sub-figures for direct comparison. "
            "Each point is one run. Dashed line: ratio = 1 (no damping). "
            "Data: two validated sessions (2026-03-26/27, lowrange mode)."
        ),
    },
}
plot_damping_ka(_ka_meta, _pv_damping_ka_by_amp, chapter="05")

# %%
"""
── CH05 § 5 — DROPPED: Swell / wind / total band amplitude scatter ──────────
Kept here commented-out as a reminder of what we tried and rejected.

Original idea: plot IN vs OUT amplitude for the "Swell" / "Wind" / "Total"
PSD bands (columns `Probe {pos} Swell Amplitude (PSD)` etc., populated by
processor2nd.py). Intent was to show which energy bands the panel
attenuates.

Why dropped (2026-04-18 design chat):
  - The swell/wind cut doesn't match the physics we actually care about.
    The relevant contrast is paddle-frequency wave (e.g. 1.3 Hz) vs
    wind-wave band (3–5 Hz) vs everything else (drift skirt below ~1.3 Hz,
    instrument noise above ~5 Hz up to Nyquist = 125 Hz).
  - If / when a band-residual plot is wanted, do it on-demand from
    `combined_psd_dict` in the plotter. The PSDs are already loaded; band
    integration is one line. No need to freeze band definitions into
    meta.json columns — that's the premature-columnisation failure mode
    we're explicitly avoiding.
  - The existing `SWELL / WIND / TOTAL` amplitude-PSD columns in
    processor2nd.py could also go away on the next major pipeline pass,
    once we're sure nothing else depends on them.

Return here if a multi-band residual scatter becomes useful in CH05 or CH04
wind characterisation. See chat history / session log for the discussion
that led here.

_pv_swell_scatter = {
    "filters": {
        "WaveAmplitudeInput [Volt]": [0.1, 0.2, 0.3],
        "WaveFrequencyInput [Hz]":   (1.3, 1.6),
        "WindCondition":             None,
        "PanelCondition":            None,
    },
    "plotting": {
        "show_plot":   True,
        "save_plot":   True,
        "draft":       True,
        "figure_name": "ch05_swell_scatter",
        "force_stub":  True,
    },
}

plot_swell_scatter(meta_results, _pv_swell_scatter, chapter="05")
"""

# %%
"""
── CH05 § 6 — Reconstructed wave signal ─────────────────────────────────────
Goal: show the FFT-reconstructed paddle-frequency signal alongside the raw
time-series. Illustrates what A_FFT actually isolates from the full signal.
Data: combined_fft_dict, one representative run (1.3 Hz, 0.2 V, full panel).
"""

_pv_reconstructed = {
    "filters": {
        "WaveAmplitudeInput [Volt]": 0.2,
        "WaveFrequencyInput [Hz]":   1.3,
        "WindCondition":             None,
        "PanelCondition":            "full",
    },
    "plotting": {
        "show_plot":    True,
        "save_plot":    True,           # DRAFT — not yet polished
        "draft":        True,
        "figure_name":  "ch05_reconstructed",
        "force_stub":   True,
        "facet_by":     "probe",
        "probes":       ["9373/170", "12400/250"],
        "linewidth":    0.8,
        "grid":         True,
        "legend":       "inside",
        "xlim":         None,
        "max_points":   500,
    },
}

_recon_meta  = apply_experimental_filters(meta_results, _pv_reconstructed)
_recon_paths = {p: combined_fft_dict[p]
                for p in _recon_meta["path"] if p in combined_fft_dict}
if _recon_paths:
    plot_reconstructed(_recon_paths, _recon_meta, _pv_reconstructed,
                       data_type="fft", chapter="05")
else:
    print("ch05_reconstructed: no matching runs found — check filters.")

# %%
"""
── CH05 § 7 — All-data damping scatter (supplementary) ──────────────────────
The thesis-headline CH05 figures (§1–§6) use meta_results — two validated
lowrange folders, cond4, scoped to 1.3–1.6 Hz where both wind conditions
exist. This supplementary figure does the opposite: it plots OUT/IN(FFT)
for **every** quality-ok full-panel wave run ever recorded, across all
four hardware conditions (cond1 h272/high, cond2 h136/high, cond3
h100/high WRONG, cond4 h100/low) and the legacy Nov-2025 probe config.
Purpose: cross-condition pattern check — does the damping curve shape
depend on hardware configuration? If cond1 (pre-March) and cond4 overlap
in the thesis band, that supports the generalisability of the main
result.

Generated by analysis_scratch/all_data_damping_scatter.py. Writes PDF
+ stub directly into output/.
"""

_ch05_alldata_fig  = Path("output/FIGURES/ch05_damping_all_data_scatter.pdf")
_ch05_alldata_stub = Path("output/TEXFIGU/ch05_damping_all_data_scatter.tex")
if not (_ch05_alldata_fig.exists() and _ch05_alldata_stub.exists()):
    print("  ch05_damping_all_data_scatter missing — run "
          "`python analysis_scratch/all_data_damping_scatter.py` to generate it.")
else:
    print(f"  ch05_damping_all_data_scatter: figure OK → {_ch05_alldata_fig}")


# =============================================================================
# WAVE DETECTION (diagnostic, possibly CH04 § 6)
# =============================================================================

# %%
"""
── First wave arrival ────────────────────────────────────────────────────────
Detection of first wave energy arriving at each probe. Useful for validating
_SNARVEI_CALIB start-sample estimates and understanding wave group velocity.
"""
# Detection parameters
THRESHOLD_FACTOR = 2.0    # 2× stillwater noise floor
WINDOW_S         = 0.5    # rolling window length [s]
MIN_ARRIVAL_S    = 0.5    # arrivals below this are wind-wave artefacts

PROBE_POSITIONS = ANALYSIS_PROBES

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

# Arrival detection — needs processed_dfs
# processed_dfs = load_processed_dfs(*PROCESSED_DIRS)   # uncomment if needed
_arrival_rows = []
# for _, row in combined_meta[combined_meta["WaveFrequencyInput [Hz]"].notna()].iterrows():
#     df = processed_dfs.get(row["path"])
#     if df is None:
#         continue
#     for pos in PROBE_POSITIONS:
#         eta_col = f"eta_{pos}"
#         if eta_col not in df.columns:
#             continue
#         noise = _noise_floor.get(pos)
#         if not noise or noise <= 0:
#             continue
#         sig = df[eta_col].dropna().values
#         idx, t_s = find_first_arrival(sig, noise, fs=FS,
#                                       threshold_factor=THRESHOLD_FACTOR,
#                                       window_s=WINDOW_S)
#         _arrival_rows.append({
#             "run": Path(row["path"]).name,
#             "freq_hz":    row["WaveFrequencyInput [Hz]"],
#             "amp_volt":   row.get("WaveAmplitudeInput [Volt]"),
#             "wind":       row.get("WindCondition"),
#             "panel":      row.get("PanelCondition"),
#             "probe":      pos,
#             "dist_mm":    int(pos.split("/")[0]),
#             "arrival_idx": idx,
#             "arrival_s":  t_s,
#         })

arrival_df = pd.DataFrame(_arrival_rows)
if not arrival_df.empty:
    print(f"Arrival detections: {arrival_df['arrival_s'].notna().sum()} / {len(arrival_df)}")

# ── Plot A: faceted by frequency ──────────────────────────────────────────────
# if not arrival_df.empty:
#     _plot_facet = arrival_df.dropna(subset=["arrival_s"])
#     sns.set_style("ticks", {"axes.grid": True})
#     g = sns.relplot(
#         data=_plot_facet, x="dist_mm", y="arrival_s",
#         hue="wind", col="freq_hz",
#         kind="line", marker="o", dashes=False,
#         errorbar=None, markersize=10, linewidth=0.8,
#         height=3.5, aspect=1.3,
#         palette=WIND_COLOR_MAP,
#         facet_kws={"sharey": True},
#     )
#     for ax in g.axes.flat:
#         ax.set_xlabel("Probe distance from paddle [mm]")
#         ax.set_ylabel("First arrival [s]")
#     g.figure.suptitle(
#         f"First wave arrival  (threshold = {THRESHOLD_FACTOR}× noise floor,"
#         f"  window = {WINDOW_S} s)",
#         y=1.02, fontsize=9,
#     )
#     plt.tight_layout()
#     plt.show()

# ── Plot B: all frequencies, single axes ──────────────────────────────────────
# if not arrival_df.empty:
#     WIND_LS = {"no": "-", "lowest": "--", "full": ":"}
#     _plot_single = arrival_df[arrival_df["arrival_s"] > MIN_ARRIVAL_S].copy()
#     _freqs_sorted = sorted(_plot_single["freq_hz"].dropna().unique())
#     _freq_colors  = {f: c for f, c in zip(
#         _freqs_sorted, plt.cm.rainbow(np.linspace(0, 1, len(_freqs_sorted)))
#     )}
#     _agg = (
#         _plot_single
#         .groupby(["wind", "freq_hz", "dist_mm"])["arrival_s"]
#         .agg(mean="mean", err=lambda x: (x.max() - x.min()) / 2)
#         .reset_index()
#     )
#     apply_thesis_style()
#     fig, ax = plt.subplots(figsize=(9, 5))
#     for (wind, freq), grp in _agg.groupby(["wind", "freq_hz"]):
#         grp_s = grp.sort_values("dist_mm")
#         ax.errorbar(grp_s["dist_mm"], grp_s["mean"], yerr=grp_s["err"],
#                     marker="o", markersize=8, linewidth=1.2, capsize=4,
#                     color=_freq_colors[freq],
#                     linestyle=WIND_LS.get(wind, "-"),
#                     label=f"{freq} Hz / {wind}")
#     ax.set_xlabel("Probe distance from paddle [mm]")
#     ax.set_ylabel("First arrival [s]")
#     ax.set_title(
#         f"First wave arrival — all frequencies  "
#         f"(threshold {THRESHOLD_FACTOR}× noise,  arrivals > {MIN_ARRIVAL_S} s shown)"
#     )
#     handles, labels = ax.get_legend_handles_labels()
#     ax.legend(handles[::-1], labels[::-1], fontsize=8, title="freq / wind")
#     plt.tight_layout()
#     _meta = {
#         "chapter": "04", "panel": None, "wind": None,
#         "amplitude": None, "frequency": None,
#         "probes": PROBE_POSITIONS, "script": "main_save_figures.py::first_arrival",
#     }
#     save_and_stub(fig, _meta, "first_arrival")
#     plt.show()

# %% ── DIAGNOSTICS ───────────────────────────────────────────────────────────
# D1 — 1.3 Hz cross-session consistency check
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE: 1.3 Hz is the most over-represented frequency — many early sessions
# used it as a reference wave. The "drop" at 1.3 Hz visible in result plots
# (ch05_damping_freq etc.) may be a data-quality artefact from older sessions
# rather than a physical effect. This figure checks consistency.
#
# IMPLEMENTATION TODO:
#   1. Load ALL PROCESSED_DIRS (the full commented-out list), not just the two
#      active ones. Use load_analysis_data() with the complete list.
#   2. Filter combined_meta to WaveFrequencyInput = 1.3 Hz, PanelCondition=full,
#      both amplitudes, all wind conditions.
#   3. Add a "session" column derived from the PROCESSED_DIR folder name (date
#      substring), so each session date is a distinct x-category or color group.
#   4. Plot OUT/IN (FFT) per session, one subplot per (WindCondition, amplitude)
#      combination. X-axis = session date, y-axis = OUT/IN (FFT).
#   5. If sessions agree → the 1.3 Hz drop is real physics.
#      If older sessions are outliers → restrict to the two reliable folders.
#
# NOTE: Use a SEPARATE load_analysis_data() call here with ALL_PROCESSED_DIRS
# so the main combined_meta (two folders only) is not affected.
#
# ALL_PROCESSED_DIRS (for this diagnostic only):
_ALL_PROCESSED_DIRS_FOR_13HZ = [
    # Path("waveprocessed/PROCESSED-20251005-sixttry6roof-highMooring"),
    # Path("waveprocessed/PROCESSED-20251110-tett6roof-lowM-ekte580"),
    # Path("waveprocessed/PROCESSED-20251110-tett6roof-lowMooring"),
    # Path("waveprocessed/PROCESSED-20251110-tett6roof-lowMooring-2"),
    # Path("waveprocessed/PROCESSED-20251112-tett6roof"),
    # Path("waveprocessed/PROCESSED-20251113-tett6roof"),
    # Path("waveprocessed/PROCESSED-20251113-tett6roof-loosepaneltaped"),
    # Path("waveprocessed/PROCESSED-20251113-tett6roof-probeadjusted"),
    # Path("waveprocessed/PROCESSED-20260307-ProbPos4_31_FPV_2-tett6roof"),
    # Path("waveprocessed/PROCESSED-20260312-ProbPos4_31_FPV_2-tett6roof"),
    # Path("waveprocessed/PROCESSED-20260313-ProbePos4_31_FPV_2-tett6roof"),
    # Path("waveprocessed/PROCESSED-20260314-ProbePos4_31_FPV_2-tett6roof"),
    # Path("waveprocessed/PROCESSED-20260316-ProbePos4_31_FPV_2-tett6roof"),
    # Path("waveprocessed/PROCESSED-20260316-ProbePos4_31_FPV_2-tett6roof-under9Mooring"),
    # Path("waveprocessed/PROCESSED-20260319-ProbePos4_31_FPV_2-tett6roof-under9Mooring"),
    # Path("waveprocessed/PROCESSED-20260321-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-RENAMED"),
    # Path("waveprocessed/PROCESSED-20260323-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height136"),
    # Path("waveprocessed/PROCESSED-20260323-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100"),
    # Path("waveprocessed/PROCESSED-20260324-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100"),
    # Path("waveprocessed/PROCESSED-20260325-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100"),
    # Path("waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100"),
    Path("waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange"),
    Path("waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"),
]
# To activate: uncomment all lines above, then replace _save_placeholder below
# with a real plot using load_analysis_data(*_ALL_PROCESSED_DIRS_FOR_13HZ).

_save_placeholder(
    "diag_13hz_consistency",
    "D1 — 1.3 Hz cross-session consistency check\n[TODO: load all sessions, plot OUT/IN per session-date at 1.3 Hz]",
    chapter="diag",
)

print("main_save_figures.py — all figure sections complete.")

# TODO: check the phase on the sine vs signal comparison.
# BIG TODO: change all plots with freq on x-axis to kL.

# %%
# Quick sanity-check: aggregated OUT/IN at 1.2-1.7 Hz, fullpanel, no/full wind.
# Note: damping_all_amplitude_grouper renames OUT/IN (FFT) → mean_out_in
# (with std_out_in) in its aggregated output.
from wavescripts.filters import damping_all_amplitude_grouper
_g = damping_all_amplitude_grouper(meta_results[
    meta_results["WaveFrequencyInput [Hz]"].between(1.2, 1.7) &
    meta_results["PanelCondition"].eq("full") &
    meta_results["WindCondition"].isin(["no", "full"])
])
print(_g[["WaveFrequencyInput [Hz]", "WaveAmplitudeInput [Volt]",
          "WindCondition", "mean_out_in", "std_out_in", "n_runs"]].to_string())
