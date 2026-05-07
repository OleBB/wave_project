#!/usr/bin/env python3
# %%

"""USER UPDATE: disabled some plots on may 3rd.  """

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
    WaveAmplitudeInput [Volt]   — paddle drive voltage (0.1 / 0.2 / 0.3 V).
                                  On reader-facing plots this is relabelled as
                                  amplitude tiers A_1 / A_2 / A_3 (see
                                  wavescripts.plot_utils.amp_to_label /
                                  amp_to_tag). At the thesis scope (1.3–1.6 Hz)
                                  the nominal measured amplitudes are
                                  A_1 ≈ 7.5 mm, A_2 ≈ 15 mm, A_3 ≈ 21.5 mm
                                  (±3 % within-frequency, audited 2026-04-24).
    WaveFrequencyInput [Hz]     — paddle frequency (0.65–1.9 Hz)
    PanelCondition              — full / reverse / no
    WindCondition               — full / lowest / no

OUTPUT KEYS (reader-facing axes on plots):
    OUT/IN (FFT)  — damping ratio (= A_Ut/A_inn at paddle frequency). The
                    thesis primary metric. FFT amplitude at the paddle bin only
                    (wind waves are on a separate axis, not inflating this).
    k  (rad/m)    — wavenumber on plot x-axes. Derived from input Hz via the
                    full dispersion relation ω² = g·k·tanh(kd); replaces the
                    older "kL" axis (2026-04-24 migration).
    ka            — wavenumber × amplitude, measured per probe per run. Primary
                    wave descriptor (§19 CLAUDE.md); encodes wavelength (in k)
                    and steepness (in a). Used for CH05 §4 per-voltage panels.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
END-TO-END PROCESS FLOW — what happens when you run `python main_save_figures.py`
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 1. IMPORTS & MODULE RELOAD (lines ~110–145)
    importlib.reload(wavescripts.plotter, wavescripts.filters). Load plotter
    API + helpers. Cheap.

 2. DELEGATED-HELPER REGISTRATION (lines ~190–225)
    Define _run_delegated_if_missing(script_rel, outputs, label). Any cell
    tagged [DELEG] later calls this to subprocess-run a scratch script only
    when its declared output files are absent. Set REGENERATE_DELEGATED=True
    at module top to force every delegated script to re-run.

 3. DATASET REGISTRY (lines ~306–365)
    ALL_PROCESSED_DIRS      — every PROCESSED-* folder (CH04 methodology).
    RESULTS_PROCESSED_DIRS  — the 2 canon March-2026 lowrange folders (CH05).
    _pu.ACTIVE_DATASETS     — RESULTS_PROCESSED_DIRS names; gets written into
                              every figure stub's IMMUTABLE provenance block.

 4. LIGHT LOAD (lines ~366–375)
    load_analysis_data(*ALL_PROCESSED_DIRS, load_processed=False)
      → combined_meta (DataFrame, one row per run, all folders)
      → combined_fft_dict / combined_psd_dict (wave runs only)
      → processed_dfs = {}  (empty placeholder; deferred to gates)
    Cost: ~2 s.

 5. RESULTS SUBSET (lines ~377–394)
    meta_results = combined_meta filtered to RESULTS_PROCESSED_DIRS.
    Mooring-variant merge (below_90_loose230 / loose300 → below_90_loose)
    lands here (CH04 §3c established they're equivalent within ±7%).

 6. CH04 METHODOLOGY FIGURES (lines ~430–1322) — all [META] or [DELEG]
    Order roughly matches the thesis chapter outline (§19 CLAUDE.md):
      §1    noise floor            (is our signal above noise?)
      §2    stillwater timing      (how long between runs — TODO)
      §3    probe placement        (lateral symmetry, probe-to-probe calib)
      §4-x  wind & FFT methodology (what is the wind; why FFT not TD)
      §4b–n amplitude-method / window-choice diagnostics
      §5    inspirational timeseries (IN/OUT macro + 5-period zoom)
      §7    wave stability, period_cv
      §8    lateral equality (nowind)
      §9    amplitude profile along tank (cell commented)

 7. CH05 RESULTS FIGURES (lines ~1324–1800) — all [META] or [DELEG]
      §1    damping vs frequency         ← primary result
      §2    damping scatter vs amplitude
      §3    wind delta
      §3b   T_cross (clean nowind reference)
      §4    damping vs ka (overview + per-voltage standalones)
      §5    swell scatter (DROPPED)
      §6    reconstructed paddle signal
      §7    all-data scatter (supplementary)

 8. MEDIUM LOAD GATE (lines ~1930–1966)
    Loads processed_dfs for the 2 canon March-2026 folders (~180 runs,
    ~12 MB, ~45 s first time). Needed by [DFS-canon] cells immediately
    below: CH04 §5b timeseries overview, §6 first arrival. _loaded_dirs
    tracks which folders are already in memory so the gate is idempotent.

 9. CH04 §5b / §6 (DFS-CANON CELLS) (lines ~1968–2039)

10. HEAVY LOAD GATE (lines ~2040–2066)
    Loads processed_dfs for the remaining 23 folders (~+65 MB, ~+2 min).
    Currently only the D1 cross-session diagnostic is a [DFS-all] consumer
    (placeholder). Skip if you don't need that diagnostic.

11. DIAGNOSTICS (lines ~2068–2131)
    D1 — 1.3 Hz cross-session consistency (placeholder).

12. FINAL SANITY PRINT (lines ~2133–end)
    damping_all_amplitude_grouper aggregated OUT/IN at 1.2–1.7 Hz, fullpanel,
    no/full wind. Printed to stdout; no figure saved.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PLOT TAXONOMY — what each kind of figure shows
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Five recurring shapes show up in this file. Knowing which shape a figure is
tells you what to put on the axes and what the points/bars represent.

  (I) Single-run demo — ONE canonical run, illustrates what a reader sees
      Examples: ch04_fft_wave (one FFT), ch04_inspirational_{nowind,fullwind}
      (one time series). Data: a specific csv from a specific folder.
      Tag: usually [DELEG] (the scratch script hard-codes the run path).

 (II) All-runs scatter — each point is ONE run; colour/marker encodes condition
      Examples: ch05_damping_ka_{10,20,30}V, ch05_damping_scatter,
      ch05_damping_all_data_scatter, ch04_td_vs_fft_scatter.
      Data: apply_experimental_filters(meta_results, _pv…) → N-row DataFrame.
      Tag: [META] or [DELEG] — one row of meta_results per dot.

(III) Grouped aggregate — mean + errorbar per (freq, wind, amp, …) cell
      Examples: ch05_damping_freq, ch04_parallel_ratio, ch04_probe_noise_floor.
      Data: damping_grouper(meta_results, …) → per-cell stats_df. Each point
      is a mean over runs sharing the same experimental keys.

 (IV) Multi-probe overlay — curves or bars at several probe positions at once
      Examples: ch04_wind_psd (PSD per probe), ch04_lateral_nowind (wall/far
      asymmetry), ch04_fft_wave (IN vs OUT bar grid).
      Data: fft_dict / psd_dict / eta_{pos} columns keyed by probe position.

  (V) Methodology / decision — comparison figures that justify a choice
      Examples: ch04_fft_method_comparison (4 FFT amplitude methods),
      ch04_reconstruction_AvsB, ch04_sw_correction_test, ch04_paddle_contamination,
      ch04_per40_and_per240_HG_shifted, ch04_fft_peak_bias_cancellation.
      These are usually [DELEG] and produce a side-by-side or before-vs-after.

"All available runs" in the thesis-result context = meta_results (the two
March-2026 lowrange folders). "All runs across all sessions" = combined_meta
(CH04 methodology, supplementary scatter).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FIGURE INDEX
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Status legend:    ✓ ready   ~ draft (DRAFT stamp)   ✗ placeholder (blank fig)
                  — dropped (cell kept as a marker)

Data-class tag:   [META]       combined_meta + FFT/PSD dicts — loaded up front (~2 s)
                  [DFS-canon]  additionally needs processed_dfs from the 2 canon
                               March-2026 lowrange folders — loaded at the MEDIUM
                               LOAD GATE (~45 s first time, ~180 runs, ~12 MB)
                  [DFS-all]    needs processed_dfs from ALL folders — loaded at
                               the HEAVY LOAD GATE (~+2 min for the remaining
                               23 folders, ~75 MB total)
                  [DELEG]      subprocess-calls a scratch script; data-agnostic here
                               (the subprocess loads whatever it needs on its own)
                  [CSV]        reads a pre-computed CSV (regenerated via [DELEG] helper)
                  [OUTDATED]   cell commented out (dropped/superseded). Body kept as a
                               marker; description records the replacement figure name.
                               The agent should NEVER cite or re-enable these without
                               the user's explicit go-ahead.

Every cell starts with a `# [DATA: X]` line matching one of the six tags. If
you add a cell, add its tag too; the gates rely on [DFS-*] cells sitting below
their respective gate. Figures in the index are listed in thesis order — the
two [DFS-*] entries are flagged "(below MEDIUM gate)" to signal their physical
file location.

Three-tier load philosophy:
  Tier 1 (Light, top):       combined_meta + FFT/PSD dicts. Fast REPL iteration.
  Tier 2 (Medium, § 5/§ 6):   adds canon-only processed_dfs. Enough for thesis
                               methodology figures that only need a few
                               representative runs from the canonical dataset.
  Tier 3 (Heavy, bottom):    adds remaining 23 folders' processed_dfs. Only
                               needed for cross-session diagnostics (currently
                               a placeholder; D1 cross-session consistency).

Figure-stub invariants (2026-04-24 durable rule):
  Every figure-generation script guarantees four things on regeneration —
  (a) IMMUTABLE block freshly rewritten, (b) valid `\\begin{figure}` env,
  (c) `\\label{fig:NAME}` matches the stub filename `NAME.tex`, (d) user-
  authored captions in the Python `CAPTIONS` dict are applied verbatim.
  See memory/feedback_figure_stub_invariants.md.

CHAPTER 04 — METHODOLOGY
  §1    ch04_probe_noise_floor           [OUTDATED]  replaced by ch04_probe_noise_floor_table  (4-panel bar plot superseded 2026-05-06)
        ch04_probe_noise_floor_table     [DELEG] ✓  Reader-facing table: stillwater noise per probe, h272/high (innledende) vs h100/low (endelig)
  # §2    ch04_stillwater_timing           [META]  ✗  Swell decay time vs wait time  [TODO]
  §3    ch04_parallel_ratio              [META]  ~  Wall/far-side amplitude ratio vs frequency
        ch04_parallel_ratio_scatter      [META]  ~     └─ per-run scatter sibling
  §3b   ch04_probe_height                [DELEG] ✓  Probe height & range-mode validity
  §3c   ch04_mooring_comparison          [DELEG] ✓  Mooring rubber band length: loose230 vs loose300
  §3d   ch04_sound_speed                 [META]  ~  Speed-of-sound / lab temperature drift
  §3e   ch04_parallel_probe_agreement    [OUTDATED]  replaced by ch04_parallel_probe_agreement_by_freq
        ch04_parallel_probe_agreement_by_freq  [DELEG] ✓  9373/170 vs 9373/340 — mean-IN canonical ref, faceted by frequency (2x2 grid)
        ch04_parallel_probe_agreement_bland_altman  [DELEG] ✓  Single-panel Bland-Altman (mean vs % disagreement) — same data, 4 freqs in one figure
        ch04_parallel_probe_psd_agreement      [DELEG] ✓  table: paired t-test + variance-reduction stats (output/TABLES/)
        ch04_parallel_probe_psd_agreement_simple [DELEG] ✓     └─ 4-col reader-facing summary: f · N · <A> · Δ%
  §3f   ch04_depth_regime                [DELEG] ✓  Depth-regime map: kd vs f at d=0.58 m + deep-water-approx error
  §4-1  ch04_wind_psd                    [META]  ~  Wind PSD per probe (nowave runs)
  §4-2  ch04_wind_reflection             [META]  ✗  Wind reflection from panel  [TODO]
  §4-3  ch04_fft_wave                    [DELEG] ✓  FFT spectrum at paddle freq (1.4 Hz canon — 2×2 bar grid w/ Δ)
  §4-3b ch04_reconstructed                [META]  ✓  FFT-reconstructed paddle signal — 4 stacked panels (wind × probe), A4-tall
  §4-4  ch04_wind_snr                    [META]  ~  Spectral SNR: paddle / wind noise per probe
  §4-5  ch04_td_vs_fft                   [META]  ~  A_td vs A_FFT: why FFT is required
        ch04_td_vs_fft_scatter           [META]  ~     └─ per-run scatter sibling
  §4b   ch04_fft_peak_bias_cancellation  [CSV]   ~  Peak-bin FFT bias cancels in OUT/IN ratio
  §4c   ch04_mansard_funke_reflection    [DELEG] ✓  Mansard–Funke reflection coefficient
  §4d   ch04_sw_correction_test          [DELEG] ✓  Standing-wave correction test — negative evidence
  §4e   ch04_sliding_afft_stability      [DELEG] ✓  Sliding-window FFT stability at IN probe
  §4f   ch04_reconstruction_AvsB         [DELEG] ✓  Peak-bin (A) vs band-integrated (B) equivalence
  §4g   ch04_reconstruction_pure_wind    [DELEG] ✓  Pure wind via no-wind residual subtraction
  §4h   ch04_fft_method_comparison       [DELEG] ~  4-method FFT comparison (nearest/parabolic/goertzel/ls_fit)
  §4i   ch04_fft_window_length_sens      [DELEG] ~  Window-length sensitivity (N ∈ {5,8,10,12,15,20})
  §4j   ch04_fft_window_position_sens    [DELEG] ~  Window-position sensitivity (T_ref ∈ [40,80]T)
  §4k   ch04_fft_window_position_trace   [DELEG] ~  Visual: sweep windows overlaid on η(t)
  §4L   ch04_paddle_contamination        [DELEG-HEAVY] ~  Paddle-freq IN contamination + wind correction + T_cross validation  (cell moved below HEAVY LOAD GATE)
  §4m   ch04_per40_and_per240_HG_shifted [DELEG] ~  Per40+per240 pooling under probe-shifted H&G window (the answer: yes)
  §4n   ch04_hg_per40_window_fitness_f{13,14,15,16}  [DELEG] ~  Proposed H&G window (N=15 + UC-snap) overlaid on η(t), per40+per240, all 4 thesis freqs
        ch04_window_intervals            [DELEG] ~     └─ companion table: IN/OUT window intervals + samples-per-period at each thesis freq
  §4o   ch04_window_choice               [DELEG] ~  Post-squeeze window (N_off=7, N_len=10 uniform): window vs nowind eyeball + fullwind empirical plateau
        ch04_window_choice_nowind        [DELEG] ~     └─ companion table (nowind, eyeball plateau)
        ch04_window_choice_fullwind      [DELEG] ~     └─ companion table (fullwind, empirical plateau ±2%)
        ch04_plateau_overview_A{1,2,3}   [DELEG] ~     └─ reader-facing plateau (sliding A_FFT, 4×2 IN/OUT panels, both winds), one per amp tier
        ch04_plateau_values              [DELEG] ~     └─ appendix table (A_IN, A_OUT, OUT/IN at chosen window per (f, amp, wind))
        ch04_tidsvindu                   [DELEG] ~     └─ main-text table (c_g, IN/OUT vindu, Δt, vindusbredde — 5×4, read from meta)
  §4p   ch04_wind_transition_overview    [DELEG] ~  Wind transition (ramp-up + decay, full + 60 s zoom — IN/OUT only, μ₀-subtracted, ±15 mm)
        ch04_wind_rampup_full / _zoom60  [DELEG] ~     └─ subfigs: 20260314 fan 0 → max
        ch04_wind_decay_full  / _zoom60  [DELEG] ~     └─ subfigs: 20260327 fan max → 0 (endofday)
  §4q   ch04_wind_pre_paddle_psd         [DELEG] ~  3 s pre-paddle ensemble PSD vs 5 long nowave-fullwind runs (IN-wall) — validates pre-paddle window as wind sample
        ch04_wind_pre_paddle_table       [CSV]   ~     └─ companion summary table (4 probes × {long σ, 3 s σ, Δ%, 3 s scatter})
        ch04_wind_qc_control_chart       [DELEG] ~     └─ appendix QC: σ_η per ok run, chronological, full+no+lowest
        ch04_wind_qc_boxplot             [DELEG] ~     └─ appendix QC: σ_η distribution by (wind × date)
        ch04_wind_setup_baseline_table   [CSV]   ~     └─ appendix: 3-nowind-vs-3-fullwind |Δη| at OUT per transition, all March datasets
  §5    ch04_inspirational_nowind        [DELEG] ~  Reading a time series — nowind canon (macro + 5-period zoom)
        ch04_inspirational_fullwind      [DELEG] ~     └─ same layout, fullwind canon (wind-wave clutter visible at IN)
        ch04_amp_methods_a1_fullwind     [DELEG] ~     └─ three amplitude estimators (FFT, p99, φ-locked) on IN, A_1 fullwind, 4 freqs (5 s slices)
  §5b   ch04_wind_pre_paddle_overlay     [DELEG] ~  Pre-paddle overlay: wind-wave bg at upstream IN/8804 vs panel-shadowed OUT (1.3 Hz, 0.2 V)
        ch04_per40_overlay_t10-21        [DELEG] ~  Per40 ramp-up + settle, fw vs nw overlay at IN (1.3 Hz, 0.2 V) — highway phase shift visible
        ch04_per40_overlay_t40-51        [DELEG] ~  Per40 ramp-down + decay, fw vs nw overlay at IN (1.3 Hz, 0.2 V) — phase shift persists, decay similar
  §5b   ch04_timeseries_overview         [!disabled] ~  Full time-series with stable-window band  (below MEDIUM gate) (? redundant because of hg_per40_window_fitness)
  §6    ch04_first_arrival               [DFS-canon] ~  First wave arrival vs probe distance      (below MEDIUM gate)
  §7    ch04_wave_stability              [META]  ~  Wave stability and period_cv vs frequency
  §8    ch04_lateral_nowind              [META]  ~  Lateral equality (parallel ratio, no-wind)
        ch04_lateral_nowind_scatter      [META]  ~     └─ per-run scatter sibling
  §9    ch04_amplitude_profile           [META]  ✗  Amplitude at every probe, all runs  [cell commented]

CHAPTER 05 — RESULTS
  §1    ch05_damping_freq                [META]  ✓  OUT/IN (FFT) vs frequency  ← primary result
  §1b   ch05_damping_freq_table          [DELEG] ✓     └─ companion table: per-amp K_t,uten / K_t,vind / ΔK_t across 1.3–1.6 Hz
  §2    ch05_damping_scatter             [META]  ✓  OUT/IN scatter vs amplitude
  §3    ch05_wind_effect_table           [DELEG] ~  Wind effect: per-(freq,amp) ΔK_t + % gains/reductions table
        ch05_wind_effect_table_by_amp    [DELEG] ✓     └─ same data, sorted amp-outer / freq-inner (sibling layout)
  §3a   ch05_transmission_wind_ratios    [DELEG] ~  R_IN, R_OUT, R_T per (freq, amp) — main thesis table
        ch05_transmission_wind_amplitudes [DELEG] ~     └─ supporting/appendix: A_IN, A_OUT, T per (freq, amp, wind)
  §3b   ch05_t_cross                     [DELEG] ✓  T_cross: honest wind effect via clean nowind ref
  §4    ch05_damping_ka                  [META]  ~  Damping vs ka (wavenumber × amplitude)
        ch05_damping_ka_{A1,A2,A3}       [DELEG] ✓     └─ standalone per-amplitude-tier (per240+per40, magenta palette)
  §4a   ch05_damping_ka_fit              [DELEG] ~  Damping vs ka — same data as §4, with per-wind poly-2 fit overlay
        ch05_damping_ka_fit_{A1,A2,A3}   [DELEG] ~     └─ per-amp subfigs with R² annotation
  §4b   ch05_mooring_focus_at_1_3hz_ka   [DELEG] ~  Mooring + panelretning at 1.30 Hz — single A4 page, 3 stacked subfigs
        ch05_mooring_focus_at_1_3hz_ka_{A1,A2,A3}  [DELEG] ~     └─ per-amp subfigs (loaded from above)
        ch05_mooring_focus_at_1_3hz_table  [DELEG] ~     └─ companion table: K_t per wind, ΔK_t pp, T-økn %, D-red % per (amp, panel, mooring)
  §4c   ch05_full_vs_reverse_at_1_3hz_ka  [DELEG] ~  Combined: same data as §4b, all 3 amps in one scatter (axes match §4)
  §5    (removed — was Swell/Wind/Total band scatter; PSD-band columns dropped 2026-05-02)
  §6    (moved → CH04 §4-3b as ch04_reconstructed)
  §7    ch05_damping_all_data_scatter    [DELEG] ✓  Supplementary: OUT/IN across ALL conditions

DIAGNOSTICS
  D1    diag_13hz_consistency            [META]  ✗  1.3 Hz cross-session consistency  [TODO]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# %% ── dev: reload modules (run this cell after editing wavescripts/) ─────────
import importlib, wavescripts.plotter as _pm, wavescripts.filters as _fm

from numpy._core.numeric import False_

importlib.reload(_pm); importlib.reload(_fm);

from wavescripts.plotter import (plot_probe_noise_floor, plot_parallel_ratio,
                                  plot_frequency_spectrum, plot_wave_stability,
                                  plot_timeseries_overview,
                                  plot_damping_freq, plot_damping_scatter)
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
    plot_fft_peak_bias_cancellation,
    plot_first_arrival,
    plot_frequency_spectrum,
    plot_parallel_ratio,
    plot_probe_noise_floor,
    plot_reconstructed,
    plot_reconstructed_combined,
    plot_sound_speed,
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


# ── Delegated figure helper ───────────────────────────────────────────────────
# Some CH04/CH05 figures are built by standalone scripts in analysis_scratch/
# instead of inline plotter calls — either because the figure is bespoke
# (e.g. reconstruction A-vs-B has a custom 4-row layout) or because the
# analysis itself is separate from the plotting (e.g. Mansard–Funke).
# Each scratch script writes its PDF + TEXFIGU stub directly to output/.
#
# `_run_delegated_if_missing` gives main_save_figures.py the "one-stop-shop"
# property: running this file end-to-end produces every figure, regenerating
# via subprocess when expected outputs are absent. Set REGENERATE_DELEGATED
# to True to force every delegated script to re-run (use when the pipeline
# data changed). Default False: run only when outputs are missing.
import argparse
import subprocess
import sys

REGENERATE_DELEGATED = False

# ── CLI flags (parsed once at module load) ────────────────────────────────────
# Default (no flags): run end-to-end, including both load gates.
#   --skip-heavy   : skip the HEAVY LOAD GATE (D1 cross-session diagnostic).
#                    Medium gate still runs → CH04 §5b/§6 DFS-canon cells produce.
#   --skip-dfs     : skip BOTH gates → only the Light tier (CH04 §1…§9, CH05 §1…§7)
#                    runs. Nothing that needs processed_dfs executes. Fast.
#   --regenerate   : equivalent to setting REGENERATE_DELEGATED=True (forces
#                    every delegated script to re-run even if outputs exist).
# Safe to import this file in non-CLI contexts too (Jupyter, tests) —
# parse_known_args tolerates unknown/missing argv.
_cli = argparse.ArgumentParser(add_help=False)
_cli.add_argument("--skip-heavy", action="store_true")
_cli.add_argument("--skip-dfs",   action="store_true")
_cli.add_argument("--regenerate", action="store_true")
_args, _ = _cli.parse_known_args()
SKIP_HEAVY = _args.skip_heavy or _args.skip_dfs
SKIP_DFS   = _args.skip_dfs
if _args.regenerate:
    REGENERATE_DELEGATED = True
if SKIP_DFS:
    print("CLI: --skip-dfs set → light tier only (no processed_dfs)")
elif SKIP_HEAVY:
    print("CLI: --skip-heavy set → medium tier ok, heavy gate skipped")

def _run_delegated_if_missing(
    script_rel: str,
    outputs: list[Path],
    label: str | None = None,
    *,
    force: bool | None = None,
    timeout_s: int = 900,
) -> None:
    """Run ``analysis_scratch/<script>`` if any expected output is missing.

    Parameters
    ----------
    script_rel : str
        Repo-relative path of the scratch script.
    outputs : list[Path]
        Files the script is expected to write. Checked with ``exists()``;
        write-once stubs + existing PDFs both qualify as "already there".
    label : str, optional
        Short label for the status line. Defaults to the first output stem.
    force : bool, optional
        Run the script even if all outputs are already present. Defaults
        to the module-level ``REGENERATE_DELEGATED`` toggle.
    timeout_s : int
        Kill the subprocess after this many seconds. Default 900 (15 min).

    Never raises — the cell's downstream existence check still fires a
    visible warning if the figure truly didn't land.
    """
    label = label or Path(outputs[0]).stem
    missing = [p for p in outputs if not p.exists()]
    if force is None:
        force = REGENERATE_DELEGATED
    if not missing and not force:
        print(f"  {label}: OK ({len(outputs)} output(s) present)")
        return
    reason = "REGENERATE_DELEGATED=True" if force else f"{len(missing)} missing"
    print(f"  {label}: running {script_rel} ({reason})")
    try:
        r = subprocess.run(
            # sys.executable = the same interpreter this file is running in,
            # so the subprocess inherits the conda env (draumkvedet) whether
            # main_save_figures.py is run from CLI, Zed REPL, or a notebook.
            [sys.executable, script_rel],
            cwd=str(file_dir),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        print(f"    {label}: TIMEOUT after {timeout_s}s — script killed")
        return
    if r.returncode != 0:
        tail = (r.stderr or "(no stderr)")[-400:].rstrip()
        print(f"    {label}: FAILED (rc={r.returncode}); stderr tail: {tail}")
        return
    still_missing = [p.name for p in outputs if not p.exists()]
    if still_missing:
        print(f"    {label}: ran but outputs still missing → {still_missing}")
        return
    print(f"    {label}: regenerated {len(outputs)} output(s)")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE CAPTIONS — single source of truth for thesis caption text.
# ══════════════════════════════════════════════════════════════════════════════
# Edit the right-hand string for each figure. Empty string → the stub's
# \caption{} body gets a TODO placeholder so a glance at this dict tells
# you what's still unwritten.
#
# Keys are figure_name == .tex stem == .pdf stem == \label{fig:...} suffix.
# Multi-subfigure stubs have one key for the parent \caption{} (the .tex
# stem) plus one key per included PDF (subfig basename).
#
# This dict is read by both inline plotters (same Python process) and
# delegated subprocess scripts via output/.figure_captions.json (rewritten
# below on every import). Run `python main_save_figures.py --regenerate`
# to refresh every TEXFIGU stub from these strings (force=True on stub
# writes — required for caption edits to land).
#
# Plain LaTeX text. No {slot} interpolation, no agent-written defaults,
# no clipboard side-effects.
# ──────────────────────────────────────────────────────────────────────────────
FIGURE_CAPTIONS: dict[str, str] = {

    # ── CHAPTER 04 — METHODOLOGY ─────────────────────────────────────────────

    # § 1 — Probe noise floor (parent + 4 subfigs) — OUTDATED, kept for backwards refs
    # "ch04_probe_noise_floor":          "Hver enkelt probes støygulv", #these four figs are replaced by the table now
    # "ch04_probe_noise_floor_group0":   "Prober hengende høyt over vannet.",
    # "ch04_probe_noise_floor_group1":   "",
    # "ch04_probe_noise_floor_group2":   "",
    # "ch04_probe_noise_floor_group3":   "Prober hengende lavt over vannet.",
    # § 1 (replacement) — reader-facing noise-floor table
    "ch04_probe_noise_floor_table":    "Oversikt over støygulvet til prober ved innledende og endelig oppsett.",   # TODO: write caption

    # § 2 — Stillwater timing (placeholder)
    # "ch04_stillwater_timing":          "", !archived

    # § 3 — Probe placement / parallel probes
    # "ch04_parallel_ratio":             "",
    # "ch04_parallel_ratio_scatter":     "",
    "ch04_probe_height":               "",
    "ch04_mooring_comparison":         "",
    "ch04_sound_speed":                "",
    "ch04_parallel_probe_agreement":   "",
    "ch04_parallel_probe_agreement_by_freq": "Forskjell mellom høyre og venstre prober. Hvert vindu viser én frekvens og tre amplituder.",
    "ch04_parallel_probe_agreement_bland_altman": "Forskjell mellom parallelle prober. ",
    "ch04_parallel_probe_psd_agreement":     "Samsvar mellom parallelle prober",
    "ch04_parallel_probe_psd_agreement_simple": "",   # TODO: caption — "the two parallel probes agree to within ~2% at every thesis frequency"
    "ch04_depth_regime":               "",

    # § 4 — Wind characterisation / FFT methodology
    "ch04_wind_psd":                   "",
    "ch04_wind_reflection":            "",
    "ch04_fft_wave":                   "Frekvensspekter for bølge på \qty{1.4}{\hertz}, amplitudevalg $A_2$. Oppe uten vind, ned med vind. Innkommende til venstre, utgående til høyre.",
    "ch04_reconstructed":              "Hovedfrekvensen fra bølgen \qty{1.4}{\hertz}, amplitudevalg $A_2$, resten av signalet er separert ut.  Stablet ovenfra og ned: Innkommende uten vind, Utgående uten vind, Innkommende med vind, Utgående med vind. Felles x- og y-akser for alle fire paneler.",
    "ch04_wind_snr":                   "",
    "ch04_td_vs_fft":                  "",
    "ch04_td_vs_fft_scatter":          "",
    "ch04_fft_peak_bias_cancellation": "",
    "ch04_mansard_funke_reflection":   "",
    "ch04_sw_correction_test":         "",
    "ch04_sliding_afft_stability":     "",
    "ch04_reconstruction_AvsB":        "",
    "ch04_reconstruction_pure_wind":   "",
    "ch04_paddle_contamination":       "",
    "ch04_per40_and_per240_HG_shifted":"",
    "ch04_hg_per40_window_fitness_f13": "",
    "ch04_hg_per40_window_fitness_f14": "",
    "ch04_hg_per40_window_fitness_f15": "",
    "ch04_hg_per40_window_fitness_f16": "",
    "ch04_window_intervals":           "",
    "ch04_window_choice":              "",
    "ch04_window_choice_nowind":       "",
    "ch04_window_choice_fullwind":     "",
    "ch04_plateau_overview_A1":        "Glidende gjennomsnitt av amplitude. $A_1$. Innkommende til venstre, utgående til høyre. Vertikale linjer indikerer estimerte tider for andre effekter.",
    "ch04_plateau_overview_A2":        "Glidende gjennomsnitt av amplitude. $A_2$. Innkommende til venstre, utgående til høyre. Vertikale linjer indikerer estimerte tider for andre effekter.",
    "ch04_plateau_overview_A3":        "Glidende gjennomsnitt av amplitude. $A_3$. Innkommende til venstre, utgående til høyre. Vertikale linjer indikerer estimerte tider for andre effekter.",
    "ch04_plateau_values":             "Beregnet amplitude fra hvert tidsvindu. Samlet for alle tre amplituder.Inngående og utgående. Transmisjonskoeffisient, og dens standardavvik.",
    "ch04_tidsvindu":                  "Frekvensenes tidsvinduer",

    # § 4p — Wind transition overview (parent + 4 subfigs: ramp-up + decay × full + zoom)
    "ch04_wind_transition_overview":   "Vindens påvirkning på vannets nivå. Merk: Ulike x-akser.",
    "ch04_wind_rampup_full":           "Fra null vind til full vind.",
    "ch04_wind_rampup_zoom60":         "Zoomet på 60 s.",
    "ch04_wind_decay_full":            "Fra full vind til null vind.",
    "ch04_wind_decay_zoom60":          "Zoomet på 60 s.",

    # § 4q — Pre-paddle wind PSD validation (3 s snippet vs long nowave runs)
    "ch04_wind_pre_paddle_psd":        "",
    "ch04_wind_pre_paddle_table":      "",
    "ch04_wind_qc_control_chart":      "",
    "ch04_wind_qc_boxplot":            "",
    "ch04_wind_setup_baseline_table":  "Målt endring i vannstand ved å se på utgående probe. Fire datasett.",

    # § 5 — Reading a time series (inspirational opener) #NOTE: used raw string r"" because of python newline break.
    "ch04_inspirational_nowind":       r"Tidsserie for bølgen \qty{1.4}{\hertz}, amplitudevalg $A_2$, uten vind. Nærbilde av de første fem periodene i tidsvinduet. ka, inn: \num{0.1287},   ka, ut:  \num{0.0878}",#todo: consider changing these numbers if the pipeline changes... if amplitude changes slightly..
    "ch04_inspirational_fullwind":     r"Tidsserie for bølgen \qty{1.4}{\hertz}, amplitudevalg $A_2$, med vind. Nærbilde av de første fem periodene i tidsvinduet. ka, inn:   \num{0.1531}, ka, ut: \num{0.0979}", #however these are illustrative plots...and i think the precision perhaps doesnt matter too much in plots. tables are more important.

    # § 5 — Three amplitude estimators (FFT vs percentile vs phase-locked) — IN probe, A_1 fullwind, 4 frequencies
    "ch04_amp_methods_a1_fullwind":    r"Nærbilde av tidsserier for innkommende bølge, $A_1$. Full vind for alle fire frekevensene. Samme tidsakse i sekunder fra start.  Verdiene er beregnet fra hver bølges eget vindu på 10 perioder.",

    # § 5b — Highway-effect visual evidence (per40 overlay + pre-paddle)
    "ch04_wind_pre_paddle_overlay":    "To like bølger, med og uten vind, lagt oppå hverandre.",
    "ch04_per40_overlay_t10-21":       "",
    "ch04_per40_overlay_t40-51":       "",

    # § 6–9 — Wave-range detection, autocorrelation, lateral
    "ch04_first_arrival":              "",
    "ch04_timeseries_overview":        "",
    "ch04_wave_stability":             "",
    "ch04_lateral_nowind":             "",
    "ch04_lateral_nowind_scatter":     "",

    # ── CHAPTER 05 — RESULTS ─────────────────────────────────────────────────

    # § 1 — Damping vs frequency (parent + 3 subfigs)
    "ch05_damping_freq":               "Transmisjonskoeffisient per frekvens. Endelig oppsett. Usikkerhetsstolper viser standardavvik. Ingen stolpe hvis kun én serie er brukt.",
    "ch05_damping_freq_full_A1":       "Amplitudevalg $A_1$",
    "ch05_damping_freq_full_A2":       "Amplitudevalg $A_2$",
    "ch05_damping_freq_full_A3":       "Amplitudevalg $A_3$",
    "ch05_damping_freq_table":         "",   # TODO: caption — per-amp K_t,uten, K_t,vind, ΔK_t across 1.3–1.6 Hz, mirrors ch05_damping_freq layout

    # § 2 — Damping vs amplitude
    "ch05_damping_scatter":            "Transmisjonskoeffisient per frekvens. Endelig oppsett. Samlet figur med alle tre amplitudene. Usikkerhetsstolper er fjernet.",

    # § 3 — Wind effect (table; replaces the old scatter ch05_damping_wind_delta,
    #         archived 2026-04-28).
    "ch05_wind_effect_table":          "",
    "ch05_wind_effect_table_by_amp":   "",   # TODO: caption — same data as ch05_wind_effect_table, sorted amp-outer / freq-inner

    # § 3a — Transmission/wind ratio summary tables (per-(f, amp) main + supporting)
    "ch05_transmission_wind_ratios":     "",
    "ch05_transmission_wind_amplitudes": "",

    # § 3b — T_cross (parent + 3 subfigs)
    "ch05_t_cross":                    "",
    "ch05_t_cross_A1":                 "",
    "ch05_t_cross_A2":                 "",
    "ch05_t_cross_A3":                 "",

    # § 4 — Damping vs ka
    "ch05_damping_ka":                 "Transmisjonskoeffisient per bølgesteilhet $ka$. Alle tre amplitudevalg. ",
    "ch05_damping_ka_A1":              "Transmisjonskoeffisient per bølgesteilhet $ka$. Amplitudevalg $A_1$.",
    "ch05_damping_ka_A2":              "Transmisjonskoeffisient per bølgesteilhet $ka$. Amplitudevalg $A_2$.",
    "ch05_damping_ka_A3":              "Transmisjonskoeffisient per bølgesteilhet $ka$. Amplitudevalg $A_3$.",

    # § 4a — Damping vs ka with per-wind poly-2 fits # tror æ droppe denne
    "ch05_damping_ka_fit":             "TODO: skriv hovedteksten. Samme data som figur \\ref{fig:ch05_damping_ka}, uten overlagt kurvetilpasning (kombinert kombinert visning blir for trang for tilpasningskurver).",
    "ch05_damping_ka_fit_A1":          "TODO: skriv hovedteksten. Samme data som figur \\ref{fig:ch05_damping_ka_A1}, med en grad-2-polynom-tilpasning per vindkondisjon (per240 + per40 slått sammen). Kurvetilpasning + $R^2$-verdier i figuren viser at uten-vind-data følger en tydelig kurve mens med-vind-data har større spredning på samme amplitude.",
    "ch05_damping_ka_fit_A2":          "TODO: skriv hovedteksten. Som figur \\ref{fig:ch05_damping_ka_fit_A1}, men for $A_2$.",
    "ch05_damping_ka_fit_A3":          "TODO: skriv hovedteksten. Som figur \\ref{fig:ch05_damping_ka_fit_A1}, men for $A_3$.",

    # § 4b — Mooring + panelretning at 1.30 Hz (single-page A4, 3 subfigs)
    "ch05_mooring_focus_at_1_3hz_ka":    "Sammenlikning av transmisjon for tre amplituder ved frekvens \\qty{1.3}{\hertz}", #"TODO: skriv hovedteksten. Sammenligning av transmisjon $K_t$ ved 1.30 Hz på tvers av mooring (below\\_90 vs above\\_50) og panelretning (normal vs revers), for hver av amplitudevalgene $A_1$, $A_2$, $A_3$. Fyrer (hule) markører for å se overlappende punkter.",
    "ch05_mooring_focus_at_1_3hz_ka_A1": " $A_1$.",
    "ch05_mooring_focus_at_1_3hz_ka_A2": " $A_2$.",
    "ch05_mooring_focus_at_1_3hz_ka_A3": " $A_3$.",
    # § 4b — Companion table for the figure above.
    "ch05_mooring_focus_at_1_3hz_table": "Tall til figur \ref{fig:ch05_mooring_focus_at_1_3hz_ka}. Transmisjon for panelrekken fortøyd på ulike måter. Merk: kun for \qty{1.3}{\hertz}. Antall (n) kjøringer.",
    # claude kladd). Per $\\Delta K_t$, transmisjonsforhold $K_{t,\\text{vind}}/K_{t,\\text{uten}}$ ($>1$ = vind slipper mer bølge gjennom), og dempningsforhold $D_{\\text{vind}}/D_{\\text{uten}}$ med $D = 1 - K_t$ ($<1$ = vind reduserer panelets demping). Tomt felt for revers $\\cdot$ below\\_90 — denne kombinasjonen ble ikke kjørt.",

    # § 4c — Combined scatter (same data as §4b, all amps in one panel).
    "ch05_full_vs_reverse_at_1_3hz_ka":  "Samme data som figur \\ref{fig:ch05_mooring_focus_at_1_3hz_ka}, men alle tre amplituder samlet i én figur. Akser matcher figur \\ref{fig:ch05_damping_ka} for direkte sammenligning.",

    # § 6 — moved to CH04 §4-3b as ch04_reconstructed (paired with ch04_fft_wave)

    # § 7 — All-data scatter (supplementary)
    "ch05_damping_all_data_scatter":   "Alle kjøringer. Vi skiller primært mellom det endelige oppsettet og alle andre oppsett.",


    # ── DIAGNOSTICS ──────────────────────────────────────────────────────────
    "diag_13hz_consistency":           "",
}

# ──────────────────────────────────────────────────────────────────────────────
# SHORT CAPTIONS for the List of Figures (LOF entry).
#
# Edit when you want a different LOF entry from the full \caption{} body.
# Empty string → no [short] argument is emitted; LaTeX falls back to the full
# caption for the LOF (its default behaviour). Author manually — no automatic
# truncation, no first-sentence extraction.
#
# Subfigures don't have short captions (their text never appears in LOF), so
# this dict only contains parent figure_names.
# ──────────────────────────────────────────────────────────────────────────────
FIGURE_CAPTIONS_SHORT: dict[str, str] = {

    # ── CHAPTER 04 ───────────────────────────────────────────────────────────
    "ch04_probe_noise_floor":          "Probes støygulv",
    "ch04_probe_noise_floor_table":    "",   # TODO: short caption
    # "ch04_stillwater_timing":          "", !archived
    # "ch04_parallel_ratio":             "", !disabled
    # "ch04_parallel_ratio_scatter":     "", !disabled
    "ch04_probe_height":               "",
    "ch04_mooring_comparison":         "",
    "ch04_sound_speed":                "",
    # "ch04_parallel_probe_agreement":   "", !disabled
    "ch04_parallel_probe_agreement_by_freq": "Parallelle prober, fire frekvenser.",
    "ch04_parallel_probe_agreement_bland_altman": "",
    "ch04_parallel_probe_psd_agreement":     "",
    "ch04_parallel_probe_psd_agreement_simple": "",
    "ch04_depth_regime":               "",
    # "ch04_wind_psd":                   "", - moved to main_save_archive.py
    # "ch04_wind_reflection":            "", - moved to main_save_archive.py
    "ch04_fft_wave":                   "Fourierspektrum med og uten vind.",
    "ch04_reconstructed":              "Rekonstruerte bølger",
    "ch04_wind_snr":                   "",
    "ch04_td_vs_fft":                  "",
    "ch04_td_vs_fft_scatter":          "",
    "ch04_fft_peak_bias_cancellation": "",
    "ch04_mansard_funke_reflection":   "",
    "ch04_sw_correction_test":         "",
    "ch04_sliding_afft_stability":     "",
    "ch04_reconstruction_AvsB":        "",
    "ch04_reconstruction_pure_wind":   "",
    "ch04_paddle_contamination":       "",
    "ch04_per40_and_per240_HG_shifted":"",
    "ch04_hg_per40_window_fitness_f13": "",
    "ch04_hg_per40_window_fitness_f14": "",
    "ch04_hg_per40_window_fitness_f15": "",
    "ch04_hg_per40_window_fitness_f16": "",
    "ch04_window_intervals":           "",
    "ch04_window_choice":              "",
    "ch04_window_choice_nowind":       "",
    "ch04_window_choice_fullwind":     "",
    "ch04_plateau_overview_A1":        "Platå, A1",
    "ch04_plateau_overview_A2":        "Platå, A2",
    "ch04_plateau_overview_A3":        "Platå, A3",
    "ch04_plateau_values":             "",
    "ch04_tidsvindu":                  "Frekvensenes tidsvindu",
    "ch04_wind_transition_overview":   "",
    "ch04_wind_pre_paddle_psd":        "",
    "ch04_wind_pre_paddle_table":      "",
    "ch04_wind_qc_control_chart":      "",
    "ch04_wind_qc_boxplot":            "",
    "ch04_wind_setup_baseline_table":  "",
    "ch04_inspirational_nowind":       "Tidsserie uten vind",
    "ch04_inspirational_fullwind":     "Tidsserie med vind",
    "ch04_amp_methods_a1_fullwind":    "Tre amplitudemetoder, $A_1$ med vind",
    "ch04_wind_pre_paddle_overlay":    "Vindsignal før padlestart",
    "ch04_per40_overlay_t10-21":       "Per40 oppstart, fasestart skifter",
    "ch04_per40_overlay_t40-51":       "Per40 nedstart, fase fortsatt skiftet",
    "ch04_first_arrival":              "",
    "ch04_timeseries_overview":        "",
    "ch04_wave_stability":             "",
    "ch04_lateral_nowind":             "",
    "ch04_lateral_nowind_scatter":     "",

    # ── CHAPTER 05 ───────────────────────────────────────────────────────────
    "ch05_damping_freq":               "",
    "ch05_damping_scatter":            "Transmisjon per frekvens",
    "ch05_damping_freq_table":         "",
    "ch05_wind_effect_table":          "",
    "ch05_wind_effect_table_by_amp":   "",
    "ch05_transmission_wind_ratios":     "",
    "ch05_transmission_wind_amplitudes": "",
    "ch05_t_cross":                    "",
    "ch05_damping_ka":                 "Transmisjon per ka. Samlet.",
    "ch05_damping_ka_A1":              "Transmisjon per ka. A1",
    "ch05_damping_ka_A2":              "Transmisjon per ka. A2",
    "ch05_damping_ka_A3":              "Transmisjon per ka. A3",
    "ch05_damping_ka_fit":             "Transmisjon per ka, samlet (uten kurvetilpasning).",
    "ch05_damping_ka_fit_A1":          "Transmisjon per ka, $A_1$ — med kurvetilpasning per vind.",
    "ch05_damping_ka_fit_A2":          "Transmisjon per ka, $A_2$ — med kurvetilpasning per vind.",
    "ch05_damping_ka_fit_A3":          "Transmisjon per ka, $A_3$ — med kurvetilpasning per vind.",
    "ch05_mooring_focus_at_1_3hz_ka":   "Mooring + panelretning ved 1.30 Hz.",
    "ch05_mooring_focus_at_1_3hz_ka_A1": "Mooring + panelretning, $A_1$.",
    "ch05_mooring_focus_at_1_3hz_ka_A2": "Mooring + panelretning, $A_2$.",
    "ch05_mooring_focus_at_1_3hz_ka_A3": "Mooring + panelretning, $A_3$.",
    "ch05_mooring_focus_at_1_3hz_table": "Mooring + panelretning ved 1.30 Hz — hardtall.",
    "ch05_full_vs_reverse_at_1_3hz_ka":  "Mooring + panelretning ved 1.30 Hz — alle amplituder samlet.",
    "ch05_damping_all_data_scatter":   "",

    # ── DIAGNOSTICS ──────────────────────────────────────────────────────────
    "diag_13hz_consistency":           "",
}

# Persist for delegated subprocess scripts that import write_figure_stub
# (which reads this JSON via _lookup_central_caption). Gitignored — source
# of truth is the two dicts above, this file is a runtime artefact.
import json as _json
_captions_path = file_dir / "output" / ".figure_captions.json"
_captions_path.parent.mkdir(parents=True, exist_ok=True)
_captions_path.write_text(
    _json.dumps(
        {"full": FIGURE_CAPTIONS, "short": FIGURE_CAPTIONS_SHORT},
        indent=2, ensure_ascii=False,
    ),
    encoding="utf-8",
)

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
# figures (OUT/IN vs freq/k, damping vs amplitude) must use only the two
# validated lowrange/h100 folders.

ALL_PROCESSED_DIRS = [
    # ── Nov 2025: probe 1 at 18000 mm, roof not fully sealed ──────────────────
    Path("waveprocessed/PROCESSED-20251005-sixttry6roof-highMooring"),
    # ── Nov 2025: probe 1 moved to 8804 mm, lowMooring ────────────────────────
    # NOTE (2026-05-05): the `lowM-ekte580` folder is intentionally absent.
    # That source dataset (wavedata/20251110-tett6roof-lowM-ekte580) contains
    # only per15 runs, which improved_data_loader filters out at discovery
    # (too short for the analysis window). main.py skips it → no PROCESSED-*
    # folder is produced. Old caches that contained it predate the per15/per30
    # discovery filter.
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
# combined_meta + FFT/PSD dicts are cheap (~2 s, a few MB). Loaded up front.
# processed_dfs (raw 250 Hz time series, ~75 MB) is deferred — only the
# [DATA: DFS] cells below the Heavy load gate need it. See the gate cell
# for where the deferred load actually happens.
print("Loading analysis data (meta + FFT/PSD; deferring processed_dfs)...")
combined_meta, _, combined_fft_dict, combined_psd_dict = load_analysis_data(
    *ALL_PROCESSED_DIRS, load_processed=False
)
processed_dfs: dict = {}   # placeholder; real load happens at the Heavy load gate

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
# %% !disabled  (replaced by ch04_probe_noise_floor_table — sibling below)
# [DATA: OUTDATED]  — replaced by analysis_scratch/probe_noise_floor_table.py
# Original cell: 4-panel bar plot of stillwater noise per (probe, hw config)
# generated by plot_probe_noise_floor in plotter.py. Superseded 2026-05-06
# by the simpler reader-facing table that compares only the two endpoint
# configs (h272/high → h100/low). The 4-panel form had two intermediate
# configs (h136/high, n=1; h100/high, transitional) that didn't carry the
# methodology message — the reader-facing question is "innledende vs endelig"
# (initial vs final), and a 4-row × 3-column table answers it cleanly.
#
# Agent: do NOT cite, regen, or re-enable this cell without an explicit
# user request. Output PDFs (ch04_probe_noise_floor_group{0..3}.pdf) and
# stub on disk are intentionally left in place (project convention: never
# rm artefacts; remove via `git rm` if the user decides they should go).
# The plot_probe_noise_floor function itself stays in plotter.py — it's
# called by analysis_scratch/probe_noise_floor_table.py to compute the
# summary numbers that feed the new table.
#
# To re-enable: uncomment the docstring + plot_probe_noise_floor call,
# flip the FIGURE INDEX tag from "[OUTDATED]" back to "[META]", and change
# this header back to "# [DATA: META]".
#
# """
# ── CH04 § 1 — Probe uncertainty / noise floor ───────────────────────────────
# Goal: show the stillwater noise amplitude per probe and hardware configuration,
# and derive the minimum detectable wave amplitude (detection threshold).
#
# Three questions answered per (probe, config):
#   1. Precision  — how much does the reading fluctuate in still water?
#   2. Bias       — do probes agree on the mean water level within a config?
#   3. Threshold  — what is the smallest detectable wave amplitude?
#
# Data: combined_meta stillwater rows (WindCondition=="no", WaveFrequencyInput NaN).
#       processed_dfs needed for quantization_step_mm (optional but recommended).
#
# Groups: probe_height_mm × probe_range_mode — 4 hardware configurations:
#   h272/high  (default pre-2026-03-23)
#   h136/high
#   h100/high
#   h100/low
#
# Metrics (all from combined_meta, shift-invariant — valid at any probe height):
#   noise_95pct_amp_mm   (P99.5−P0.5)/2   mean across accepted runs in group  [legacy col name; values are 99 % half-range]
#   noise_rms_mm         std(raw signal)   mean across accepted runs in group
#   mean_level_mm        median level      mean across accepted runs in group
#   bias_vs_ref_mm       mean_level − cross-probe mean (within group)
#   quantization_step_mm P5 of nonzero |diff(η)|  from processed_dfs
#   detection_threshold_mm  max(k_sigma·σ,  k_q·q)   default max(3σ, 2q)
# """
#
# from datetime import datetime as _dt
# from wavescripts.improved_data_loader import get_configuration_for_date
# # Probe numbers derived from current config (hardware IDs, fixed across configs):
# _active_cfg = get_configuration_for_date(_dt(2026, 3, 15))
# _PROBE_NUM_MAP = {pos: num for num, pos in _active_cfg.probe_col_names().items()}
#
# _pv_noise_floor = {
#     "filters": {},
#     "plotting": {
#         "show_plot": True,
#         "save_plot": True,
#         "draft":     False,
#         "figure_name": "ch04_probe_noise_floor",
#         "force_stub": True,
#         "figsize":        (3.4, 3.0),
#         "ylim":           (0, 0.5),
#         "xtick_fontsize": 7,
#         "show_excluded":  False,
#             "text": {
#                 "ylabel": "Støyamplitude (99 %)  [mm]",
#                 "legend_mean_amp":    "Gjennomsnittlig støyamplitude  (±1σ)",
#                 "legend_per_run":     "Per kjøring",
#                 "legend_threshold":   "Terskel  max({k_sigma:.0f}σ, {k_q:.0f}q)  [mm]",
#                 "legend_quantization":"Halvt kvantiseringssteg  q/2  [mm]",
#             },
#     },
# }
#
# _figs_nf, _noise_summary = plot_probe_noise_floor(
#     combined_meta, ANALYSIS_PROBES, _pv_noise_floor,
#     group_by=["probe_height_mm", "probe_range_mode"],
#     processed_dfs = processed_dfs,
#     highlight_keyword=None,
#     probe_number_map=_PROBE_NUM_MAP,
# )

# %%
# [DATA: DELEG]  — analysis_scratch/probe_noise_floor_table.py
"""
── CH04 § 1 — Probe noise floor table (innledende vs endelig) ───────────────
Replaces the 4-panel bar plot with a simple reader-facing table that
compares the initial setup (h272/high) to the final canon setup (h100/low).
Four rows (one per probe position), three columns:
    - Støyamplitude (99 %), innledende [mm]
    - Støyamplitude (99 %), endelig [mm]
    - Forbedring (innledende / endelig)

Method note: "Støyamplitude (99 %)" = (P99.5 − P0.5) / 2 of stillwater
signal, mean across accepted stillwater runs in each config. Same metric
as the y-axis of the (now outdated) 4-panel figure, so numbers are
directly comparable.

Generated by analysis_scratch/probe_noise_floor_table.py. Calls
plot_probe_noise_floor under the hood (no re-implementation), then
filters to the two endpoint configs and pivots into a side-by-side
table. Writes:
    output/TABLES/ch04_probe_noise_floor_table.tex   (thesis include)
    analysis_scratch/ch04_probe_noise_floor_table.csv (companion)
"""

_run_delegated_if_missing(
    "analysis_scratch/probe_noise_floor_table.py",
    [Path("output/TABLES/ch04_probe_noise_floor_table.tex")],
    label="ch04_probe_noise_floor_table",
)







# removed: # ── CH04 § 2 — Stillwater timing (how long to wait between runs) ─────────────

# %%
# [DATA: META]
# """
# ── CH04 § 3 — Probe placement: longitudinal and lateral effects ─────────────
# Goal: show what parallel probes tell us — lateral uniformity without wind,
# lateral asymmetry with wind. Also: why the longitudinal positions were chosen.

# Data: combined_meta, parallel_ratio column, no-wind wave runs. But,
# the probes placed downstream are to be trusted more, because no interference from mooring and panel.

# Figures:
#   - Plot:  parallel_ratio vs frequency, coloured by WindCondition
#   - Plot:  parallel_ratio vs frequency, coloured by PanelCondition (reflection)
#   - Table: parallel_ratio summary (mean, std) by wind/panel group
# """

# """ PRINTOUT
# Ratio of wall-side to far-side probe amplitude at the same longitudinal distance, for 154 wave runs across 1 panel configurations. A ratio of 1 indicates lateral symmetry. Deviations indicate wall reflections or wind-driven lateral asymmetry. Error bars: standard deviation across runs at the same frequency. Dashed line: ratio = 1.
# """

# _pv_parallel_ratio = {
#     "filters": {},
#     "plotting": {
#         "show_plot": True,
#         "save_plot": True,            # DRAFT — parallel ratio not yet polished
#         "draft":     True,
#         "figure_name": "ch04_parallel_ratio",
#         "force_stub": True,
#     },
# }
# start = time.perf_counter()
# _fig_pr = plot_parallel_ratio(combined_meta, _pv_parallel_ratio)
# end = time.perf_counter()
# print(f"Lateral symmetry of plot_parallel_ratio {end-start:.4f} seconds")

# _pv_parallel_ratio_scatter = {
#     "filters": {},
#     "plotting": {
#         **_pv_parallel_ratio["plotting"],
#         "scatter":     True,
#         "figure_name": "ch04_parallel_ratio_scatter",
#     },
# }
# plot_parallel_ratio(combined_meta, _pv_parallel_ratio_scatter)

# %%
# [DATA: DELEG]  — analysis_scratch/probe_height_figure.py
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

_run_delegated_if_missing(
    "analysis_scratch/probe_height_figure.py",
    [Path("output/FIGURES/ch04_probe_height.pdf"),
     Path("output/TEXFIGU/ch04_probe_height.tex")],
    label="ch04_probe_height",
)

# %%
# [DATA: DELEG]  — analysis_scratch/mooring_comparison.py
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

_run_delegated_if_missing(
    "analysis_scratch/mooring_comparison.py",
    [Path("output/FIGURES/ch04_mooring_comparison.pdf"),
     Path("output/TEXFIGU/ch04_mooring_comparison.tex")],
    label="ch04_mooring_comparison",
)

# %%
# [DATA: META]
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
        "draft":       False,
        "figure_name": "ch04_sound_speed",
        "force_stub":  True,
        "figsize":     (10, 3),
    },
}

plot_sound_speed(combined_meta, _pv_sound_speed, chapter="04")

# %% !disabled  (sibling below)
# [DATA: OUTDATED]  — replaced by analysis_scratch/parallel_probe_agreement_by_freq.py
# Original cell: analysis_scratch/parallel_probe_agreement.py (3-panel scatter
# + disagreement-vs-V + heatmap). Superseded 2026-05-02 by the per-frequency
# facet sibling below (ch04_parallel_probe_agreement_by_freq) — the 3-panel
# form gave too much surface area to defend given the wall/far probes'
# unequal noise-floor spec; the freq-faceted 2x2 scatter + the PSD agreement
# table together carry the methodology. Cell body kept as a marker so the
# §3e history stays visible in this file.
#
# Agent: do NOT cite, regen, or re-enable this cell without an explicit
# user request. Output PDF + TEXFIGU stub on disk are intentionally left in
# place (project convention: never rm artefacts; remove via `git rm` if the
# user decides they should go).
#
# To re-enable: uncomment the docstring + _run_delegated_if_missing call,
# flip the FIGURE INDEX tag back from "[OUTDATED]" to "[DELEG] ✓", and
# change this header back to "# [DATA: DELEG]".
#
# """
# ── CH04 § 3e — Parallel-probe agreement (mean-IN validation) ────────────────
# Methodology support for the CH05 decision to use mean(9373/170, 9373/340)
# as the canonical IN reference. Shows that the two probes — at the same
# longitudinal distance from the paddle — agree to within ±5% for nowind
# and ±10% for fullwind 0.2–0.3 V runs across the thesis band. The
# fullwind 0.1 V regime has larger probe-to-probe scatter (low SNR,
# wind contamination dominates each probe differently). That's exactly
# where the mean is most valuable (reduces single-probe noise).
#
# Result: 62/78 thesis-scope runs consistent at 10% threshold. Mean IN
# is justified for the headline result; single-run outliers (like the
# 0.3 V nowind H&G-window dip in analysis_scratch/huseby_grue_window.pdf)
# are absorbed by averaging.
#
# Generated by analysis_scratch/parallel_probe_agreement.py. Writes PDF
# + stub directly into output/.
# """
#
# _run_delegated_if_missing(
#     "analysis_scratch/parallel_probe_agreement.py",
#     [Path("output/FIGURES/ch04_parallel_probe_agreement.pdf"),
#      Path("output/TEXFIGU/ch04_parallel_probe_agreement.tex")],
#     label="ch04_parallel_probe_agreement",
# )

# %%
# [DATA: DELEG]  — analysis_scratch/parallel_probe_agreement_by_freq.py
"""
── CH04 § 3e (sibling) — Parallel-probe agreement, faceted by frequency ─────
Per-frequency facet of the §3e scatter (panel a only). 2x2 grid, one panel
per thesis paddle frequency (1.3, 1.4, 1.5, 1.6 Hz). Same data and same
±5% / ±10% identity bands as §3e, but separated by frequency so any
frequency-dependent deviation between the two parallel probes is visible
without cross-frequency mixing. Companion to PSD-based agreement analysis
in analysis_scratch/parallel_probe_psd_agreement.py (paired t-tests +
amplitude-variance reduction table).

Generated by analysis_scratch/parallel_probe_agreement_by_freq.py.
"""

_run_delegated_if_missing(
    "analysis_scratch/parallel_probe_agreement_by_freq.py",
    [Path("output/FIGURES/ch04_parallel_probe_agreement_by_freq.pdf"),
     Path("output/TEXFIGU/ch04_parallel_probe_agreement_by_freq.tex")],
    label="ch04_parallel_probe_agreement_by_freq",
)

# %%
# [DATA: DELEG]  — analysis_scratch/parallel_probe_agreement_bland_altman.py
"""
── CH04 § 3e (single-panel sibling) — Parallel-probe Bland-Altman ───────────
Same n=80 runs, same scope as the 2×2 facet above. Replotted as a single
panel: x = mean amplitude, y = signed disagreement (%). Marker convention
copies ch05_damping_ka (_freq_marker): outline shape = amplitude tier
(○ A1, □ A2, △ A3); orientation/fill = paddle frequency. Colour = wind.
Lets the reader compare wind / amp / freq trends in disagreement at a
glance, where the facet shows per-frequency scatter intuition.
"""
_run_delegated_if_missing(
    "analysis_scratch/parallel_probe_agreement_bland_altman.py",
    [Path("output/FIGURES/ch04_parallel_probe_agreement_bland_altman.pdf"),
     Path("output/TEXFIGU/ch04_parallel_probe_agreement_bland_altman.tex")],
    label="ch04_parallel_probe_agreement_bland_altman",
)

# %%
# [DATA: DELEG]  — analysis_scratch/parallel_probe_psd_agreement.py
"""
── CH04 § 3e (sibling table) — Parallel-probe PSD agreement statistics ──────
Methodology table that quantifies the visual agreement shown in
ch04_parallel_probe_agreement_by_freq:
  - Per-frequency mean dB difference between far and wall probes
  - Paired t-test against H0: zero difference
  - Pearson correlation of band-integrated amplitudes across runs
  - Per-probe σA and the σA of their simple mean
  - ΔVar(mean) — % change in Var(½(A_wall + A_far)) vs the better single
    probe (positive ⇒ averaging worsens precision)

Output table lands in output/TABLES/ (NOT output/FIGURES/). The script also
emits a 3-panel scratch PDF (overlay + difference spectrum + zoom) and
prints the same stats to stdout.

Generated by analysis_scratch/parallel_probe_psd_agreement.py.
"""

_run_delegated_if_missing(
    "analysis_scratch/parallel_probe_psd_agreement.py",
    [Path("output/TABLES/ch04_parallel_probe_psd_agreement.tex")],
    label="ch04_parallel_probe_psd_agreement",
)

# %%
# [DATA: DELEG]  — analysis_scratch/parallel_probe_psd_agreement_simple.py
"""
── CH04 § 3e (sibling, simple) — Parallel-probe PSD agreement (4 cols) ──────
Reader-facing summary of the same data as the 10-column statistical table.
One row per thesis frequency: f, N, mean amplitude across both probes,
signed disagreement (far − wall) in percent. Headline number for the
thesis prose: "the two parallel probes agree to within ~2 % across all
four thesis frequencies (n=80 paired runs each)".

Both versions coexist; one will be removed once the thesis prose settles.
Generated by analysis_scratch/parallel_probe_psd_agreement_simple.py.
"""

_run_delegated_if_missing(
    "analysis_scratch/parallel_probe_psd_agreement_simple.py",
    [Path("output/TABLES/ch04_parallel_probe_psd_agreement_simple.tex")],
    label="ch04_parallel_probe_psd_agreement_simple",
)

# %%
# [DATA: DELEG]  — subprocess-calls analysis_scratch/depth_regime_map.py
# """
# ── CH04 § 3f — Depth-regime map (kd vs f) at d = 0.58 m ─────────────────────
# Two-panel figure that justifies the full dispersion relation ω² = g·k·tanh(kd)
# used throughout the pipeline (wavescripts.constants.c_group,
# wavescripts.plot_utils.freq_to_k).

#     Top panel: kd vs paddle frequency at d = 580 mm. Shaded regime bands
#     (deep / intermediate / shallow) with thresholds kd=π and kd=π/10.
#     Run-frequency markers sized by n_runs, coded by regime (colour + shape).
#     Vertical band marks the thesis scope (1.3–1.6 Hz). Secondary right-hand
#     axis shows wavelength λ in metres.

#     Bottom panel: relative error in λ if the deep-water approximation
#     λ_deep = g/(2π f²) were used instead of full dispersion, as % on a
#     symlog y-axis. Quantifies "by how much does the full dispersion matter".

# At 580 mm depth, thesis-scope runs (1.3–1.6 Hz) are comfortably deep
# (kd > π, deep-water approximation error < 0.11 %). Sub-1 Hz frequencies
# drift into intermediate water (kd < π at 1.0 Hz; up to ~15 % λ-error at
# 0.7 Hz) — consistent with the bottom-motion observation from 2026-03-12
# annotated on the figure. Scope boundary `f < 1 Hz out of scope`
# (MEMORY.md) has a direct physical rationale visible here.

# Delegated build — see analysis_scratch/depth_regime_map.py.
# """
# _run_delegated_if_missing(
#     "analysis_scratch/depth_regime_map.py",
#     [Path("output/FIGURES/ch04_depth_regime.pdf"),
#      Path("output/TEXFIGU/ch04_depth_regime.tex")],
#     label="ch04_depth_regime",
# )

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
# [DATA: DELEG]  — subprocess-calls analysis_scratch/fft_wave_spectrum.py
"""
── CH04 § 4-3 — FFT spectrum: paddle frequency peak ────────────────────────
Goal: teach the reader the discrete FFT spectrum of a paddle wave. A 2×2
grid of bar plots at the actual FFT bin centres for a canonical 1.4 Hz,
0.2 V, fullpanel, per240 run (canon 20260327). Rows = wind condition,
columns = probe side (Inn / Ut). Horizontal dotted guides at each row's
paddle peaks; vertical guides at f, 2f, 3f, 4f (labels on the top axis);
Δ and A_Ut/A_inn arrow in the Utgående panels.

Typeset in NewComputerModern (OTFs registered directly from TeX Live).
Bin widths per probe are recorded in the TEXFIGU stub's extra_stats
(they differ by ±1 sample between probes because the H&G window end is
UC-snapped independently per probe).

Delegated build — see analysis_scratch/fft_wave_spectrum.py for details.
"""
_run_delegated_if_missing(
    "analysis_scratch/fft_wave_spectrum.py",
    [Path("output/FIGURES/ch04_fft_wave.pdf"),
     Path("output/TEXFIGU/ch04_fft_wave.tex")],
    label="ch04_fft_wave",
)

# %%
# [DATA: META]  — reads combined_fft_dict (NOT processed_dfs); lives above gate
"""
── CH04 § 4-3b — Reconstructed wave signal (combined nowind / fullwind) ─────
Goal: show the FFT-reconstructed paddle-frequency signal alongside the raw
time-series, stacked top-to-bottom for both wind conditions and both probe
positions on a single A4-tall page. Pairs naturally with §4-3 (the FFT
spectrum) — same data, time-domain view of what the single-bin reconstruction
isolates from the full signal.

Layout (top → bottom): Innkommende uten vind, Utgående uten vind,
Innkommende med vind, Utgående med vind. Single shared symmetric y-axis.
Data: combined_fft_dict, one representative run per wind condition
(1.4 Hz, 0.2 V, full panel).
"""

_pv_reconstructed = {
    "filters": {
        "WaveAmplitudeInput [Volt]": 0.2,
        "WaveFrequencyInput [Hz]":   1.4,
        "WindCondition":             ["no", "full"],
        "PanelCondition":            "full",
    },
    "plotting": {
        "show_plot":    False,
        "save_plot":    True,
        "draft":        False,
        "figure_name":  "ch04_reconstructed",
        "force_stub":   True,
        "probes":       ["9373/170", "12400/250"],
        "linewidth":    0.8,
        "show_full_signal":  False,
        "grid":         True,
        "figsize":      (8, 11),
    },
}

_recon_meta  = apply_experimental_filters(meta_results, _pv_reconstructed)
_recon_paths = {p: combined_fft_dict[p]
                for p in _recon_meta["path"] if p in combined_fft_dict}
if _recon_paths:
    plot_reconstructed_combined(_recon_paths, _recon_meta, _pv_reconstructed,
                                data_type="fft", chapter="04")
else:
    print("ch04_reconstructed: no matching runs found — check filters.")

# %%
# [DATA: META]  — reads combined_fft_dict + combined_psd_dict
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
# [DATA: META]
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
    },
}
plot_td_vs_fft(combined_meta, _pv_td_vs_fft_scatter, chapter="04")

# %% !disabled - either repurpose or archive
# [DATA: CSV]  — analysis_scratch/fft_peak_bias_outin_impact.py regens CSV;
#                then plotter reads it
# """
# ── CH04 § 4b — FFT peak-bin bias cancels in OUT/IN ──────────────────────────
# Goal: establish that although nearest-bin FFT amplitudes are biased by
# paddle-drift vs bin-grid alignment (sinc attenuation up to ~40% for
# individual amplitudes), the OUT/IN ratio is robust because IN and OUT
# probes use matching analysis window lengths → same bin grid → bias
# cancels. Empirical: mean |Δ(OUT/IN)|/OUT/IN < 0.5% across ~360 runs.

# Data: precomputed CSV at analysis_scratch/fft_peak_bias_outin_impact.csv
# (generated by analysis_scratch/fft_peak_bias_outin_impact.py, which
# loads processed_dfs). If the CSV is missing or stale, re-run that script.

# This figure documents the headline methodology safeguard: the thesis's
# primary OUT/IN (FFT) result is not an artifact of FFT binning.
# """

# # The plotter reads a per-run CSV that the scratch script generates;
# # regenerate the CSV via subprocess if it's missing, then call the plotter.
# _run_delegated_if_missing(
#     "analysis_scratch/fft_peak_bias_outin_impact.py",
#     [Path("analysis_scratch/fft_peak_bias_outin_impact.csv")],
#     label="fft_peak_bias_outin_impact.csv",
# )
# _pv_fft_peak_bias = {
#     "filters": {},
#     "plotting": {
#         "show_plot":   True,
#         "save_plot":   True,            # DRAFT — polish on review
#         "draft":       True,
#         "figure_name": "ch04_fft_peak_bias_cancellation",
#         "force_stub":  True,
#     },
# }
# plot_fft_peak_bias_cancellation(_pv_fft_peak_bias, chapter="04")

# %%
# [DATA: DELEG]  — analysis_scratch/mansard_funke.py
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

_run_delegated_if_missing(
    "analysis_scratch/mansard_funke.py",
    [Path("output/FIGURES/ch04_mansard_funke_reflection.pdf"),
     Path("output/TEXFIGU/ch04_mansard_funke_reflection.tex")],
    label="ch04_mansard_funke_reflection",
)

# %%
# [DATA: DELEG]  — analysis_scratch/sw_correction.py
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

_run_delegated_if_missing(
    "analysis_scratch/sw_correction.py",
    [Path("output/FIGURES/ch04_sw_correction_test.pdf"),
     Path("output/TEXFIGU/ch04_sw_correction_test.tex")],
    label="ch04_sw_correction_test",
)

# %%
# [DATA: DELEG]  — analysis_scratch/sliding_afft_fullwind_sweep.py
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

_run_delegated_if_missing(
    "analysis_scratch/sliding_afft_fullwind_sweep.py",
    [Path("output/FIGURES/ch04_sliding_afft_stability.pdf"),
     Path("output/TEXFIGU/ch04_sliding_afft_stability.tex")],
    label="ch04_sliding_afft_stability",
)

# %%
# [DATA: DELEG]  — analysis_scratch/reconstruction_A_vs_B.py (same script as §4g)
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

_run_delegated_if_missing(
    "analysis_scratch/reconstruction_A_vs_B.py",
    [Path("output/FIGURES/ch04_reconstruction_AvsB.pdf"),
     Path("output/TEXFIGU/ch04_reconstruction_AvsB.tex"),
     # Same script also produces the §4g pair — list all four here so
     # one invocation satisfies both cells, and the §4g cell below is
     # a pure existence check that hits the cache.
     Path("output/FIGURES/ch04_reconstruction_pure_wind.pdf"),
     Path("output/TEXFIGU/ch04_reconstruction_pure_wind.tex")],
    label="ch04_reconstruction_AvsB",
)

# %%
# [DATA: DELEG]  — analysis_scratch/reconstruction_A_vs_B.py (same script as §4f)
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

# §4g shares its scratch script with §4f above — both figures are produced
# by the same call to reconstruction_A_vs_B.py. This cell is therefore a
# pure existence check: if §4f's call succeeded, the pure-wind outputs are
# already present and the helper short-circuits.
_run_delegated_if_missing(
    "analysis_scratch/reconstruction_A_vs_B.py",
    [Path("output/FIGURES/ch04_reconstruction_pure_wind.pdf"),
     Path("output/TEXFIGU/ch04_reconstruction_pure_wind.tex")],
    label="ch04_reconstruction_pure_wind",
)

# %%
# [DATA: DELEG]  — analysis_scratch/fft_method_comparison.py
"""
── CH04 § 4h — FFT amplitude extraction, 4-method comparison ────────────────
Observation: on 128 per-probe nowind measurements (canon March-2026 lowrange
folders, 5 frequencies), the four methods `nearest_bin` / `parabolic` /
`goertzel` / `ls_fit` agree within 0.4 % max and 0.04 % median. On synthetic
pure-tone sweeps with non-integer-cycle windows, `nearest_bin` shows up to
38 % sinc attenuation at half-bin offsets; `goertzel` and `ls_fit` stay
< 1 %.

Candidate explanation (hypothesis, not verified): the H&G 10-period window
produces bin k=10 within 0.03 bin widths of f_paddle, so the nearest-bin
reading approaches the unbiased DFT value for an integer-cycle tone.

Script uses only the two canon folders — no full-dataset load required.
Current outputs live in analysis_scratch/ (PNG + findings.md); PDF + TEX
promotion to output/ is pending.

See: analysis_scratch/fft_method_comparison_findings.md
"""

_run_delegated_if_missing(
    "analysis_scratch/fft_method_comparison.py",
    [Path("analysis_scratch/fft_method_comparison.png"),
     Path("analysis_scratch/fft_method_comparison_findings.md")],
    label="ch04_fft_method_comparison",
)

# %%
# [DATA: DELEG]  — analysis_scratch/fft_window_sensitivity_lsfit.py
"""
── CH04 § 4i — Window-length sensitivity (N ∈ {5, 8, 10, 12, 15, 20}) ───────
Observation: across 51 per240 fullpanel runs in the canon March-2026
lowrange folders, median OUT/IN drift relative to the pipeline default
N=10 periods stays ≤ 1 % at every N ∈ {5, 8, 12, 15, 20} for both nowind
and fullwind. Max |drift| (excluding one run with RECON-aborted probe
quality) is < 5 %.

Candidate explanation (hypothesis): N=10 sits within a wider plateau
[8, 15]p of interchangeable window lengths.

Complements earlier per240 window-size study
(analysis_scratch/paddle_contamination_window_sensitivity.csv from
2026-04-21, which tested N ∈ [20, 100]p and found max drift 1.17 %).

Script uses only the two canon folders. PDF/TEX promotion pending.

See: analysis_scratch/fft_window_sensitivity_lsfit_findings.md
"""

_run_delegated_if_missing(
    "analysis_scratch/fft_window_sensitivity_lsfit.py",
    [Path("analysis_scratch/fft_window_sensitivity_lsfit.png"),
     Path("analysis_scratch/fft_window_sensitivity_lsfit_findings.md")],
    label="ch04_fft_window_length_sens",
)

# %%
# [DATA: DELEG]  — analysis_scratch/fft_window_position_sensitivity_lsfit.py
"""
── CH04 § 4j — Window-position sensitivity (T_ref ∈ [40, 80]T) ──────────────
Observation: sliding a fixed 10-period window across T_ref from 40 to 80
periods (1T step), 51 per240 runs show three drift patterns relative to
the pipeline default T_ref=50T:

  - Nowind: −3.3 % at T_ref=40T; within ±0.6 % for T_ref ∈ [45, 65]T.
  - Fullwind: within ±1 % for T_ref ∈ [40, 55]T; monotonically negative
    from T_ref=55T, reaching −2.7 % at T_ref=80T.
  - Intersection of sub-1 % windows (both conditions): T_ref ∈ [45, 55]T.

Candidate explanations (hypotheses, not verified in this sweep):
Region 1 (T_ref < 45T nowind) may overlap wave-envelope build-up at OUT
(ref. memory note "envelope-back arrives at r=12.4 m at t ≈ 31 T"). Region 3
(T_ref > 55T fullwind) is consistent with a time-dependent A_in
wind-enhancement extending the static effect documented in
methodology_wind_enhances_A_in.md. Other mechanisms (reflections, wind
ramp profile, probe-specific response) not ruled out.

Script uses only the two canon folders. PDF/TEX promotion pending.

See: analysis_scratch/fft_window_position_sensitivity_lsfit_findings.md
"""

_run_delegated_if_missing(
    "analysis_scratch/fft_window_position_sensitivity_lsfit.py",
    [Path("analysis_scratch/fft_window_position_sensitivity_lsfit.png"),
     Path("analysis_scratch/fft_window_position_sensitivity_lsfit_findings.md")],
    label="ch04_fft_window_position_sens",
)

# %%
# [DATA: DELEG]  — analysis_scratch/fft_window_position_sensitivity_trace.py
"""
── CH04 § 4k — Visual: sweep windows overlaid on η(t) ───────────────────────
Pedagogical companion to §4j. Shows the actual probe time series (f=1.4 Hz,
A=0.2 V per240) from the canon March-2026 folders, with 9 representative
window positions (T_ref ∈ {40, 45, …, 80}T, every 5T) drawn as coloured
rectangles. Pipeline default T_ref=50T highlighted.

Layout: 2 rows (nowind, fullwind) × 2 cols (IN 9373/170, OUT 12400/250).
Probe-shift (ΔT ≈ 7.59 periods at 1.4 Hz between OUT and IN) is visible
as a horizontal shift of the window rectangles between columns.

Script uses only the two canon folders. PDF/TEX promotion pending.
"""

_run_delegated_if_missing(
    "analysis_scratch/fft_window_position_sensitivity_trace.py",
    [Path("analysis_scratch/fft_window_position_sensitivity_trace.png")],
    label="ch04_fft_window_position_trace",
)

# %% ── §4L moved below the HEAVY LOAD GATE ─────────────────────────────────
# `ch04_paddle_contamination` is the slowest §4 DELEG (~minutes). Relocated
# below the HEAVY LOAD GATE so a `--regen` pass can finish §4 + §5 quickly
# and reach this only when explicitly continuing past the gate.

# %%
# [DATA: DELEG]  — analysis_scratch/per40_and_per240_HG_shifted.py
"""
── CH04 § 4m — Per40 + per240 under probe-shifted H&G window ────────────────
Methodology question: we have ~4× more per40 (short) runs than per240 (long)
runs. Can we pool both into CH05? Applies the probe-shifted Huseby–Grue
window to both run types, then checks whether OUT/IN(FFT) agrees.

Physics: H&G's [35.088 s, 42.105 s] = [50T, 60T] at 1.425 Hz was measured
at r = 12.4 m — exactly our OUT probe. For closer probes (IN at 9.373 m),
shift the same window back by group-velocity travel time:
    ΔT(f) = (12.4 − r_probe) / c_group(f) · f    [periods]
    1.3 Hz → 6.6 T earlier;  1.4 Hz → 7.6 T;  1.5 Hz → 8.7 T;  1.6 Hz → 9.9 T

The shifted IN window fits inside the per40 wavetrain at every thesis
frequency → both run types analyzable with one methodology.

Result: per40 and per240 agree within 1–2 % median → pooling justified for
CH05. (The earlier mixed-method comparison ch04_per40_vs_per240_outin
was archived 2026-04-29; see ignore_this_archive/ARCHIVE_NOTES.md.)

Writes output/FIGURES/ch04_per40_and_per240_HG_shifted.pdf and stub directly.
See memory/methodology_hg_probe_shifted.md.
"""

_run_delegated_if_missing(
    "analysis_scratch/per40_and_per240_HG_shifted.py",
    [Path("output/FIGURES/ch04_per40_and_per240_HG_shifted.pdf"),
     Path("output/TEXFIGU/ch04_per40_and_per240_HG_shifted.tex")],
    label="ch04_per40_and_per240_HG_shifted",
)

# %% TODO - consider repurpose this - we are no longer using hg window, we use our "earliest possible" window, because the analysis said its better for our limited per40 runs.
# [DATA: DELEG]  — analysis_scratch/hg_per40_window_fitness.py
"""
── CH04 § 4n — Proposed H&G window fitness on per40+per240, all 4 freqs ─────
Visual companion to §4m: confirms the proposed H&G window placement sits
inside the wavetrain at every thesis frequency (1.3, 1.4, 1.5, 1.6 Hz).

Proposed formula (CH04 §4 methodology):
    t_start = r / c_g(f, h) + N_offset / f       (seconds)
    t_end   = t_start + 10 / f                   (10T H&G window)

with N_offset = 15 periods (5 wavemaker-ramp + 10 H&G "10 periods after
arrival" safety). Start snapped to nearest zero-upcrossing within ±T of
the theoretical start (matches the live pipeline's snap rule).

Per frequency, one figure with 4 rows (per40 nowind / per40 fullwind /
per240 nowind / per240 fullwind) × 2 cols (IN, OUT). Green band is the
snapped window; purple dotted line is per40 paddle stop (40/f). All four
rows share x-axis [0, 60] s so the reader compares plateaus directly.

The figures answer: does the proposed window fall inside the user's
eyeballed plateau at each frequency? — visually, yes at all four. Tightest
fit at 1.6 Hz OUT where window end coincides with plateau end; comfortable
margin at 1.3–1.5 Hz at both probes.

Outputs (4 figures + 4 stubs):
    output/FIGURES/ch04_hg_per40_window_fitness_f{13,14,15,16}.pdf
    output/TEXFIGU/ch04_hg_per40_window_fitness_f{13,14,15,16}.tex
"""

_run_delegated_if_missing(
    "analysis_scratch/hg_per40_window_fitness.py",
    [Path("output/FIGURES/ch04_hg_per40_window_fitness_f13.pdf"),
     Path("output/FIGURES/ch04_hg_per40_window_fitness_f14.pdf"),
     Path("output/FIGURES/ch04_hg_per40_window_fitness_f15.pdf"),
     Path("output/FIGURES/ch04_hg_per40_window_fitness_f16.pdf"),
     Path("output/TEXFIGU/ch04_hg_per40_window_fitness_f13.tex"),
     Path("output/TEXFIGU/ch04_hg_per40_window_fitness_f14.tex"),
     Path("output/TEXFIGU/ch04_hg_per40_window_fitness_f15.tex"),
     Path("output/TEXFIGU/ch04_hg_per40_window_fitness_f16.tex")],
    label="ch04_hg_per40_window_fitness",
)

# %%
# [DATA: DELEG]  — analysis_scratch/window_intervals_table.py
"""
── CH04 § 4n (companion table) — H&G window intervals per thesis frequency ──
Numerical companion to ch04_hg_per40_window_fitness_f{13,14,15,16}: a single
LaTeX table that tabulates the H&G window's [t_start, t_end] in seconds
at the IN and OUT probes, plus samples-per-period (Fs / f), at each thesis
frequency (1.3, 1.4, 1.5, 1.6 Hz).

Values are THEORETICAL (pre-snap) — the deterministic output of the
proposed formula `t_start = r/c_g(f, h) + 15/f`, length 10T. The actual
per-run windows snap to ±T upcrossings; that variability is shown
visually in the §4n PDFs but isn't table-friendly.

This table is the "what window did we use?" reference. Whenever the
formula changes (N_offset, window length, depth), re-running this script
updates the table — and the IMMUTABLE block at the top of the .tex file
records the formula parameters used at generation time.

Writes output/TABLES/ch04_window_intervals.tex directly.
"""

_run_delegated_if_missing(
    "analysis_scratch/window_intervals_table.py",
    [Path("output/TABLES/ch04_window_intervals.tex")],
    label="ch04_window_intervals",
)

# %%
# [DATA: DELEG]  — analysis_scratch/window_choice_figure.py
"""
── CH04 § 4o — Window-choice visual: Option B vs nowind eyeball + fullwind empirical ──
Geometry-only figure (no per-run data) showing the post-squeeze Option B
window per (f, probe) overlaid against:

  • upper grey-hash band  → nowind plateau bounds eyeballed in
                            RampDetectionBrowser (snarvei_eyeballing.md, 0.2 V)
  • lower red-hash band   → fullwind empirical plateau (per40 sliding A_FFT,
                            ±2 % relaxed criterion, n=1 per cell) from
                            per40_plateau_end_aggregated.csv

Option B parameters: t_start = r/c_g(f, h) + 10/f, length N(f)/f, with
N(f) = {1.3:10, 1.4:13, 1.5:13, 1.6:13}. Markers t_arr (▼ blue),
t_paras (▽ red), per40 paddle stop (✕ purple) for orientation.

Visual claim: the Option B window sits inside both nowind and fullwind
plateau bounds at every (f, probe) cell, except (1.3 Hz IN fullwind) where
the empirical plateau ends ~2 s before the window does — that cell is
also bound by the parasitic 2f arrival and is the binding case.

Outputs:
    output/FIGURES/ch04_window_choice.pdf
    output/TEXFIGU/ch04_window_choice.tex
"""

_run_delegated_if_missing(
    "analysis_scratch/window_choice_figure.py",
    [Path("output/FIGURES/ch04_window_choice.pdf"),
     Path("output/TEXFIGU/ch04_window_choice.tex")],
    label="ch04_window_choice",
)

# %%
# [DATA: DELEG]  — analysis_scratch/window_choice_table.py
"""
── CH04 § 4o (companion tables) — Option B window choice per (f, probe) ─────
Two LaTeX tables in parallel column structure (one per wind condition).
Columns: f, probe, t_arr, [t_start, t_end], Δ, t_2f, plateau, N.

Plateau column source differs by wind:
  • nowind   → eyeball plateau (snarvei_eyeballing.md, RampDetectionBrowser, 0.2 V)
  • fullwind → empirical plateau (per40 sliding A_FFT, ±2 % relaxed) from
                per40_plateau_end_aggregated.csv

Numbers come from the same Option B formula as the figure above.

Outputs:
    output/TABLES/ch04_window_choice_nowind.tex
    output/TABLES/ch04_window_choice_fullwind.tex
"""

_run_delegated_if_missing(
    "analysis_scratch/window_choice_table.py",
    [Path("output/TABLES/ch04_window_choice_nowind.tex"),
     Path("output/TABLES/ch04_window_choice_fullwind.tex")],
    label="ch04_window_choice_table",
)

# %%
# [DATA: DELEG]  — analysis_scratch/plateau_overview.py
"""
── CH04 § 4o — Plateau overview at A_1 / A_2 / A_3 (sliding A_FFT) ──────────
Three reader-facing figures, one per amplitude tier (A_1=0.10 V, A_2=0.20 V,
A_3=0.30 V). Each laid out 4 rows (f) × 2 cols (IN, OUT), with both wind
conditions overlaid (blue=nowind, red=fullwind). Sliding A_FFT(t) at the
paddle frequency, window length matches the chosen post-squeeze rule
(N_off = 7, N_len = 10 — uniform across all four thesis frequencies).

Each panel shows: individual canon runs (thin lines), the chosen window
(green band), and four reference vertical lines — paddle stop (40/f),
2nd harmonic arrival (r/c_g(2f)), back-wall reflection arrival
((2L−r)/c_phase, L=25 m), and the long-wave first-motion arrival
(r/√(gh)).

Visual claim: the chosen window encloses a flat A_FFT plateau at every
(f, probe, wind, amp) cell.

Outputs:
    output/FIGURES/ch04_plateau_overview_A{1,2,3}.pdf
    output/TEXFIGU/ch04_plateau_overview_A{1,2,3}.tex
"""

_run_delegated_if_missing(
    "analysis_scratch/plateau_overview.py",
    [Path("output/FIGURES/ch04_plateau_overview_A1.pdf"),
     Path("output/FIGURES/ch04_plateau_overview_A2.pdf"),
     Path("output/FIGURES/ch04_plateau_overview_A3.pdf"),
     Path("output/TEXFIGU/ch04_plateau_overview_A1.tex"),
     Path("output/TEXFIGU/ch04_plateau_overview_A2.tex"),
     Path("output/TEXFIGU/ch04_plateau_overview_A3.tex")],
    label="ch04_plateau_overview",
)

# %%
# [DATA: DELEG]  — analysis_scratch/plateau_values_table.py
"""
── CH04 § 4o (companion table) — Plateau A_FFT values inside chosen window ──
Numerical companion to ch04_plateau_overview_A{1,2,3}: per (f, amp, wind)
cell, the median A_IN, A_OUT, and OUT/IN ratio computed over the chosen
window (probe-shifted, N_off=7, N_len=10). Plus per-cell run-to-run σ on
the ratio and the cell's run count n.

24 cells (4 freqs × 3 amps × 2 winds), grouped into three amp blocks.
Bridges the figure (visual plateau) to the CH05 result (OUT/IN values).

Outputs:
    output/TABLES/ch04_plateau_values.tex
    analysis_scratch/plateau_values.csv
"""

_run_delegated_if_missing(
    "analysis_scratch/plateau_values_table.py",
    [Path("output/TABLES/ch04_plateau_values.tex")],
    label="ch04_plateau_values",
)

# %%
# [DATA: DELEG]  — analysis_scratch/tidsvindu_table.py
"""
── CH04 § 4o — Tidsvindu (main-text companion to plateau figure) ────────────
Compact 5-row × 4-freq-col table read directly from canon meta:

    c_g, Innkommende vindu [s], Utgående vindu [s], Δt, Vindusbredde [s]

Window times come from `Computed Probe {pos} start / end` in meta (median
across canon runs at each thesis frequency). The table therefore tracks
whatever window the pipeline currently produces — when the pipeline is
updated to a new N_off / N_len, regen this table and the numbers follow.

Inferred N_offset / N_length recorded in the immutable provenance block
of the generated .tex so it's always clear which window the table reflects.

Outputs:
    output/TABLES/ch04_tidsvindu.tex
    analysis_scratch/tidsvindu.csv
"""

_run_delegated_if_missing(
    "analysis_scratch/tidsvindu_table.py",
    [Path("output/TABLES/ch04_tidsvindu.tex")],
    label="ch04_tidsvindu",
)

# %%
# [DATA: DELEG]  — analysis_scratch/wind_decay_timeseries.py
"""
── CH04 § 4p — Wind transition overview (ramp-up + decay, full + zoom) ──────
Four-panel composite stacked on a single page: how the tank surface responds
when the fan is switched on (ramp-up) and switched off (decay). For each
direction we show the full record (minutes) and a 60 s zoom (seconds).

Two single-run picks — both using the march2026_better_rearranging probe
config so the IN and OUT positions are directly comparable:
  Ramp-up: 20260314 fullpanel-fromZeroWinToMaxWin-run1.csv  (fan: 0 → max)
  Decay:   20260327 experimental-fromMaxToZeroWin-...-endofday.csv (fan: max → 0)

Per-probe baseline μ₀ subtracted (ramp-up = first 2 s, decay = last 2 s) so
the wind-driven setup tilt and chop envelope are read against a known still-
water reference. Y-axis clipped to ±15 mm, common across all four panels.

Generated by analysis_scratch/wind_decay_timeseries.py. Writes 4 PDFs (one
per subfigure) plus a column-layout multi-subfig TEXFIGU stub.
"""

_run_delegated_if_missing(
    "analysis_scratch/wind_decay_timeseries.py",
    [Path("output/FIGURES/ch04_wind_rampup_full.pdf"),
     Path("output/FIGURES/ch04_wind_rampup_zoom60.pdf"),
     Path("output/FIGURES/ch04_wind_decay_full.pdf"),
     Path("output/FIGURES/ch04_wind_decay_zoom60.pdf"),
     Path("output/TEXFIGU/ch04_wind_transition_overview.tex")],
    label="ch04_wind_transition_overview",
)

# %%
# [DATA: DELEG]  — analysis_scratch/wind_2s_vs_360s.py
"""
── CH04 § 4q — Pre-paddle wind PSD validation ───────────────────────────────
Does the first 3 s of a wave run (before any paddle motion has propagated to
any probe — √(gh) = 2.39 m/s, closest probe at 8804 mm → safe to 3.68 s)
reproduce the wind PSD measured on long nowave+fullwind runs?

Canon datasets only (PROCESSED-20260326-*-lowrange + PROCESSED-20260327-*-
lowrange — final probe config). Five long fullwind+nowave reference runs
(durations 31, 33, 63, 360, 381 s), 70 fullwind+wave runs contributing one
3 s pre-paddle snippet each.

Result (IN-wall σ_η, the relevant wind metric):
  long-runs ensemble : 4.28 mm
  3 s pre-paddle     : 4.04 mm  (−5.5 %)
  per-snippet σ noise: 0.83 mm
PSD shape and ~3.7 Hz wind-wave peak match between methods. The pre-paddle
window is therefore a defensible wind sample, scaling the wind-characterisation
dataset from ~14 long runs to several hundred snippets across all wave runs.

Generated by analysis_scratch/wind_2s_vs_360s.py. Writes 1 PDF + 1 TEXFIGU stub.
"""

_run_delegated_if_missing(
    "analysis_scratch/wind_2s_vs_360s.py",
    [Path("output/FIGURES/ch04_wind_pre_paddle_psd.pdf"),
     Path("output/TEXFIGU/ch04_wind_pre_paddle_psd.tex")],
    label="ch04_wind_pre_paddle_psd",
)

# %%
# [DATA: CSV]  — analysis_scratch/wind_pre_paddle_table.py
"""
── CH04 § 4q — Pre-paddle wind summary table (companion to PSD figure) ─────
Four-row table: per probe, long-run σ_η vs 3 s pre-paddle σ_η, % delta, and
3 s 1σ scatter across 70 wave runs. Reads
analysis_scratch/wind_2s_vs_360s_stats_3s.csv (produced by the cell above)
and renders a LaTeX table to output/TABLES/.

Caption is blank in FIGURE_CAPTIONS — the user populates it later. The
immutable block flags the OUT-probe noise-floor caveat (12400/250 σ ≈
probe stillwater noise floor; the −8.6 % delta is sampling noise, not
window-length bias) and the long-run duration spread (31–381 s, only 2 of
5 ≥ 360 s) so the caption can phrase it correctly.
"""

_run_delegated_if_missing(
    "analysis_scratch/wind_pre_paddle_table.py",
    [Path("output/TABLES/ch04_wind_pre_paddle_table.tex")],
    label="ch04_wind_pre_paddle_table",
)

# %%
# [DATA: DELEG]  — analysis_scratch/wind_qc_3s.py (precondition CSV)
"""
── CH04 appendix — QC inputs ───────────────────────────────────────────────
Runs the per-run σ_η computation over the canon campaign so the thesis
QC plots downstream have their data CSV. Side products (PNGs in
analysis_scratch/) are kept for scratch iteration.
"""

_run_delegated_if_missing(
    "analysis_scratch/wind_qc_3s.py",
    [Path("analysis_scratch/wind_qc_3s_per_run.csv")],
    label="ch04_wind_qc_3s_inputs",
)

# %%
# [DATA: DELEG]  — analysis_scratch/wind_qc_3s_thesis.py
"""
── CH04 appendix — Wind-QC thesis figures ──────────────────────────────────
Two diagnostic figures promoted to thesis quality:
    output/FIGURES/ch04_wind_qc_control_chart.pdf
    output/FIGURES/ch04_wind_qc_boxplot.pdf
+ matching TEXFIGU stubs.

Captions are blank in FIGURE_CAPTIONS — user populates later. Each stub's
immutable block records two caveats requested 2026-05-01:
  (1) These QC plots are stratified only by WindCondition × date so far;
      other run-level factors (Mooring, PanelCondition, run-type
      per40/per240, time-of-day) are NOT yet decomposed and may account
      for part of the within-group scatter. Refine before publication.
  (2) OUT (12400/250) σ_η sits in the probe's stillwater noise floor
      envelope (0.14–0.36 mm), so apparent OUT-side scatter is partly
      probe noise, not genuine wind-wave variability.

Inputs:
    analysis_scratch/wind_qc_3s_per_run.csv              (cell above)
    analysis_scratch/wind_2s_vs_360s_per_long_run_3s.csv (CH04 §4q PSD cell)
"""

_run_delegated_if_missing(
    "analysis_scratch/wind_qc_3s_thesis.py",
    [Path("output/FIGURES/ch04_wind_qc_control_chart.pdf"),
     Path("output/FIGURES/ch04_wind_qc_boxplot.pdf"),
     Path("output/TEXFIGU/ch04_wind_qc_control_chart.tex"),
     Path("output/TEXFIGU/ch04_wind_qc_boxplot.tex")],
    label="ch04_wind_qc_thesis",
)

# %%
# [DATA: DELEG]  — analysis_scratch/wind_setup_baseline_3v3.py (precondition CSV)
"""
── CH04 appendix — 3v3 wind-setup at OUT (data prep) ──────────────────────
Sweeps all March 2026 PROCESSED dirs that have both nowind and fullwind
runs. For each detected nowind↔fullwind transition, reads the absolute
`Stillwater Probe 12400/250` baseline (mm; ULS reads distance DOWN to
water — lower number = higher water level), takes the last n_pre ≤ 3
nowind runs vs the first n_post ≤ 3 fullwind runs, and reports the
absolute water rise at OUT under wind.

Per-transition CSV is the precondition for the appendix table cell below.
Side products (per-transition stdout block) are useful for scratch
inspection.

See analysis_scratch/wind_setup_baseline_3v3_investigation.md for the
narrative + key observations (cross-dataset reproducibility ~0.01 mm
under strict 3+3 sampling at ~1.30 mm; OFF > ON direction asymmetry of
0.1–0.4 mm in 3 of 4 datasets).
"""

_run_delegated_if_missing(
    "analysis_scratch/wind_setup_baseline_3v3.py",
    [Path("analysis_scratch/wind_setup_baseline_3v3_results.csv")],
    label="ch04_wind_setup_baseline_inputs",
)

# %%
# [DATA: CSV]  — analysis_scratch/wind_setup_baseline_3v3_table.py
"""
── CH04 appendix — 3v3 wind-setup baseline table ──────────────────────────
Reads analysis_scratch/wind_setup_baseline_3v3_results.csv (cell above) and
renders the per-transition appendix table to output/TABLES/. One row per
detected nowind↔fullwind transition, midrule between datasets, per-dataset
mean magnitude shown at the bottom of each block.

Caption is blank in FIGURE_CAPTIONS — user populates later. The immutable
block records:
  (1) The OFF > ON direction asymmetry caveat (0.1–0.4 mm in 3 of 4
      datasets) and the two untested candidate explanations.
  (2) The 20260326 small-sample artefact (0 strict 3+3 transitions on
      that day; its lower 0.88 mm mean is a sampling artefact, not a real
      day-to-day difference).
  (3) The pipeline anchor convention for `Stillwater Probe 12400/250`.
"""

_run_delegated_if_missing(
    "analysis_scratch/wind_setup_baseline_3v3_table.py",
    [Path("output/TABLES/ch04_wind_setup_baseline_table.tex")],
    label="ch04_wind_setup_baseline_table",
)

# %%
# [DATA: DELEG]  — analysis_scratch/inspirational_timeseries.py
"""
── CH04 § 5 — Reading a time series (inspirational opener) ──────────────────
Two narrative figures at the chapter-5 threshold: one nowind, one fullwind,
both at the canon run (fullpanel, 1.4 Hz, 0.2 V, per240). Macro (full 52 s)
+ 5-period micro zoom for IN and OUT probes. Zoom windows follow each probe's
own H&G window, so the OUT zoom sits ~5 s later than IN (group-velocity lag).
The fullwind figure visually demonstrates wind-wave clutter riding on the
paddle tone at IN (9373/170, exposed) while OUT (12400/250, sheltered by the
panel) stays close to a clean sinusoid — the direct motivation for using
FFT amplitude (not time-domain percentile) as the OUT/IN metric.

Generated by analysis_scratch/inspirational_timeseries.py. Writes both PDFs +
both TEXFIGU stubs directly into output/. Tweakable variants kept in
analysis_scratch/inspirational_timeseries_C_variants.py
(output/timeseries_exploration/).
"""

_run_delegated_if_missing(
    "analysis_scratch/inspirational_timeseries.py",
    [Path("output/FIGURES/ch04_inspirational_nowind.pdf"),
     Path("output/FIGURES/ch04_inspirational_fullwind.pdf"),
     Path("output/TEXFIGU/ch04_inspirational_nowind.tex"),
     Path("output/TEXFIGU/ch04_inspirational_fullwind.tex")],
    label="ch04_inspirational_timeseries",
)


# %%
# [DATA: DELEG]  — analysis_scratch/peaktrough_vs_fft_demo_cropped.py
"""
── CH04 § 5 — Three amplitude estimators on the IN probe (A_1, fullwind) ────
Four-row stack (1.3, 1.4, 1.5, 1.6 Hz) of η(t) at IN (9373/170) inside a
shared 5 s slice of the analysis window, A_1 (0.1 V) fullwind, canon dataset
20260327. Three horizontal amplitude bands per panel:
    ── solid green   : ±A_FFT  (paddle-only)
    -- red dashed    : ±A_p    ((P99,5−P0,5)/2 percentile)
    ·· blue dotted   : ±A_φ    (phase-locked T/4, 3T/4 sample mean)

A horizontal value-legend above each axis carries the marker + value +
percent inflation vs A_FFT for that estimator. f badge in upper-right
corner. x-axis is shifted to relative time [0, 5] s ("fra t=0").

Reader takeaway: at the wind-exposed IN probe, A_p inflates above the
carrier (+87/+49/+48/+57 % across 1.3-1.6 Hz, 0.1 V fullwind) while A_φ
sits below it; FFT is the only estimator that tracks the paddle amplitude
robustly under wind. Direct visual motivation for the OUT/IN (FFT) metric
used as the central thesis result.

Generated by analysis_scratch/peaktrough_vs_fft_demo_cropped.py. The
sibling analysis_scratch/peaktrough_vs_fft_demo.py keeps the per-frequency
analysis-window-only variant (no shared-time crop) for inspection.
"""

_run_delegated_if_missing(
    "analysis_scratch/peaktrough_vs_fft_demo_cropped.py",
    [Path("output/FIGURES/ch04_amp_methods_a1_fullwind.pdf"),
     Path("output/TEXFIGU/ch04_amp_methods_a1_fullwind.tex")],
    label="ch04_amp_methods_a1_fullwind",
)


# %%
# [DATA: DELEG]  — analysis_scratch/wind_doppler_arrival_shift.py
"""
── CH04 § 5b — Pre-paddle wind background, per-probe overlay ────────────────
Three-row stack of η(t) for the first 25 s of one canonical run pair
(1.3 Hz, 0.2 V, full panel), fullwind vs nowind overlaid at each of:
  - 8804/250 (upstream)
  - 9373/170 (IN)
  - 12400/250 (OUT, panel-shadowed)

Shows that the wind-wave background is real and substantial at the upstream
IN region (~5-10 mm RMS) but essentially absent at OUT (panel shadows wind
fetch). Visual companion to the highway-effect investigation: confirms there
IS a wind-wave field at the source for the paddle wave to "join", but no
such field exists past the panel.
"""
_run_delegated_if_missing(
    "analysis_scratch/wind_doppler_arrival_shift.py",
    [Path("output/FIGURES/ch04_wind_pre_paddle_overlay.pdf"),
     Path("output/TEXFIGU/ch04_wind_pre_paddle_overlay.tex")],
    label="ch04_wind_pre_paddle_overlay",
)


# %%
# [DATA: DELEG]  — analysis_scratch/per40_full_chirp.py
"""
── CH04 § 5b — Per40 ramp-up overlay (fullwind vs nowind, t = 10-21 s) ──────
Single-axis overlay of η(t) at IN (9373/170) for one per40 canonical run
pair (1.3 Hz, 0.2 V, full panel), zoomed to the wave-train arrival window.
Raw + 25-sample-smoothed traces for both wind conditions; faint paddle-period
grid anchored to the first detected upcrossing per condition.

Reader takeaway: the fullwind paddle wave arrives with the SAME amplitude and
shape as nowind, but its zero-upcrossings consistently sit ~100-150 ms earlier
— direct visualization of the highway-effect phase shift.
"""
_run_delegated_if_missing(
    "analysis_scratch/per40_full_chirp.py",
    [Path("output/FIGURES/ch04_per40_overlay_t10-21.pdf"),
     Path("output/FIGURES/ch04_per40_overlay_t40-51.pdf"),
     Path("output/TEXFIGU/ch04_per40_overlay_t10-21.tex"),
     Path("output/TEXFIGU/ch04_per40_overlay_t40-51.tex")],
    label="ch04_per40_overlay_t10-21",
)


# %%
# [DATA: DELEG]  — analysis_scratch/per40_full_chirp.py (same script as above)
"""
── CH04 § 5b — Per40 ramp-down overlay (fullwind vs nowind, t = 40-51 s) ────
Companion to ch04_per40_overlay_t10-21. Same overlay axes, zoomed to the
ramp-down + decay portion of the per40 wave train. Confirms that:
  - The phase shift between fw and nw persists through the entire steady
    portion (red still leads blue).
  - The wave-train decay length is roughly the same in both conditions —
    wind does NOT extend the wave's life past paddle stop.

Both PDFs are produced by the same per40_full_chirp.py call; the cell above
already triggers the build, this cell just declares the dependency for
auditing.
"""
_run_delegated_if_missing(
    "analysis_scratch/per40_full_chirp.py",
    [Path("output/FIGURES/ch04_per40_overlay_t40-51.pdf"),
     Path("output/TEXFIGU/ch04_per40_overlay_t40-51.tex")],
    label="ch04_per40_overlay_t40-51",
)



# =============================================================================
# CHAPTER 05 — RESULTS
# =============================================================================

# %%
# [DATA: META]
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
        "show_plot":  False,
        "save_plot":  True,          # set True when figure is ready for thesis
        "force_stub": True,
        "figure_name": "ch05_damping_freq",
        "figsize":    (7, 3),
        "annotate":   True,
        "legend":     "outside_right",
        "probes":     ANALYSIS_PROBES,
        # Stack the three (A1/A2/A3) subfigures vertically as a full-page
        # float (\\begin{figure}[p], each subfig at \\linewidth, separated
        # by \\\\[1ex]). Default would be "row" (0.48\\linewidth + \\hfill).
        "subfig_layout": "column",
    },
}

_damping_meta   = _aef(meta_results, _pv_damping_freq)
# Pool across moorings (2026-05-05): drop the Mooring column so
# damping_all_amplitude_grouper skips it as a grouping key. Each
# (freq, amp, panel, wind) cell then pools all canon moorings into a
# single row (n-weighted mean / true std / total n_runs), matching the
# CH05 §1b/§3/§3b tables that were fixed on 2026-05-05. Without this drop,
# the per-point n_runs tokens written into the figure stub would be
# per-mooring (e.g. "no-1.40Hz:n=2; no-1.40Hz:n=3") and disagree with the
# pooled table totals. See memory/finding_wind_effect_table_aggregation_bias.md
# and analysis_scratch/damping_freq_table.py for the same pattern.
_damping_meta = _damping_meta.drop(columns=["Mooring"], errors="ignore")
_damping_grouped = damping_all_amplitude_grouper(_damping_meta)
plot_damping_freq(_damping_grouped, _pv_damping_freq)

# %%
# [DATA: DELEG]  — analysis_scratch/damping_freq_table.py
"""
── CH05 § 1b — Damping-vs-frequency table (companion to ch05_damping_freq) ─
Per amplitude tier (A1/A2/A3), tabulates K_t at no-wind, K_t at full-wind,
and ΔK_t across the four thesis frequencies. Three row-blocks mirror the
three stacked subfigures of ch05_damping_freq so the reader can read off
the plotted values without counting from a graph.

Generated by analysis_scratch/damping_freq_table.py. Writes the table
directly into output/TABLES/.
"""

_run_delegated_if_missing(
    "analysis_scratch/damping_freq_table.py",
    [Path("output/TABLES/ch05_damping_freq_table.tex")],
    label="ch05_damping_freq_table",
)

# %%
# [DATA: META]
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
        "show_plot":   False,
        "save_plot":   True,         # set True when figure is ready for thesis
        "figure_name": "ch05_damping_scatter",
        "force_stub":  True,
        "figsize":     (5, 7),
    },
}

_scatter_meta   = _aef(meta_results, _pv_damping_scatter)
_scatter_grouped = damping_all_amplitude_grouper(_scatter_meta)
plot_damping_scatter(_scatter_grouped, _pv_damping_scatter)

# %%
# [DATA: DELEG]  — analysis_scratch/wind_effect_table.py
"""
── CH05 § 3 — Wind effect on transmission/damping (table) ───────────────────
Per (freq, amp) cell within the thesis scope (1.3–1.6 Hz, full panel,
quality_flag=ok, both wind conditions present):

    K_t,uten                       = mean OUT/IN(FFT) at no-wind
    K_t,vind                       = mean OUT/IN(FFT) at full-wind
    ΔK_t      = K_t,vind − K_t,uten   (signed transmission change, pp)
    % T-gain  = ΔK_t / K_t,uten · 100 (relative transmission change)
    % D-red   = (D_nw − D_fw)/D_nw · 100   where D = 1 − K_t
                                   (relative damping reduction)

Replaces the earlier ch05_damping_wind_delta scatter (archived 2026-04-28
to ignore_this_archive/wavescripts/plot_damping_wind_delta_2026-04-28.py).
The same data is now presented as a table — more thesis-friendly when the
goal is precise per-cell numbers, and the cell count is small (4 freqs ×
3 amps = 12 rows).

Generated by analysis_scratch/wind_effect_table.py. Writes:
    output/TABLES/ch05_wind_effect_table.tex   (thesis include)
    analysis_scratch/wind_effect_table.csv     (human-readable)
"""

_run_delegated_if_missing(
    "analysis_scratch/wind_effect_table.py",
    [Path("output/TABLES/ch05_wind_effect_table.tex")],
    label="ch05_wind_effect_table",
)

# %%
# [DATA: DELEG]  — analysis_scratch/wind_effect_table_by_amp.py
"""
── CH05 § 3 sibling — Wind-effect table, amp-outer / freq-inner sort ────────
Same data, filters, and column set as ch05_wind_effect_table — only the row
order differs. Three row-blocks (A1/A2/A3), four rows each (1.3–1.6 Hz),
midrule between amp blocks. Useful when the thesis paragraph reads "for A1,
wind shifts K_t from … to …" rather than "at 1.3 Hz, the three amplitudes …".

Both layouts coexist for now; one will be removed once the thesis prose
settles on which sort order reads better.

Generated by analysis_scratch/wind_effect_table_by_amp.py. Writes:
    output/TABLES/ch05_wind_effect_table_by_amp.tex   (thesis include)
    analysis_scratch/wind_effect_table_by_amp.csv     (human-readable)
"""

_run_delegated_if_missing(
    "analysis_scratch/wind_effect_table_by_amp.py",
    [Path("output/TABLES/ch05_wind_effect_table_by_amp.tex")],
    label="ch05_wind_effect_table_by_amp",
)

# %%
# [DATA: DELEG]  — analysis_scratch/wind_effect_per_condition.py (precondition CSV)
"""
── CH05 § 3a precondition — Per-run wind/no-wind amplitudes ─────────────────
Runs the LS-fit-based per-(run, probe) amplitude extraction over the canon
March-2026 lowrange folders, fullpanel, quality_flag=ok, 1.3–1.6 Hz × A1/A2/A3
(A3@1.6 Hz excluded for high-amp dropout per feedback_freq_amp_limits.md).

Drives the `analysis_scratch/wind_effect_ratios_summary.png` plot inline AND
the per-run CSV consumed by the table cell below. Registering only the
per-run CSV as the gate output — the script also writes long/ratios CSVs
and several scatter PNGs as side products, kept for scratch iteration.
"""

_run_delegated_if_missing(
    "analysis_scratch/wind_effect_per_condition.py",
    [Path("analysis_scratch/wind_effect_per_condition_per_run.csv")],
    label="ch05_wind_effect_per_condition",
)

# %%
# [DATA: DELEG]  — analysis_scratch/transmission_wind_tables.py
"""
── CH05 § 3a — Transmission/wind ratio summary tables ──────────────────────
Reads analysis_scratch/wind_effect_per_condition_per_run.csv (cell above),
applies symmetric NaN filtering (drop a run if either A_IN or A_OUT is
missing), and emits two LaTeX tables to output/TABLES/:

  - ch05_transmission_wind_ratios.tex      (main; one row per (f, A))
       R_IN, R_OUT, R_T mean ± propagated σ + n_runs per wind state.
  - ch05_transmission_wind_amplitudes.tex  (supporting; one row per (f, A, wind))
       A_IN, A_OUT mean ± std (mm) + T mean ± std + n_runs.

Both ratio columns are ratio-of-means at the (f, A) level (matches the
R_IN / R_OUT plot in wind_effect_per_condition.py); σ is independent-Gaussian
propagation of the run-mean estimators. Console preview (markdown snippet for
A1/A2/A3 tiers) is intentional — user copies it from the terminal.

Captions are blank in FIGURE_CAPTIONS — populated by the user later. The
immutable block records provenance (source CSV path, datasets, filters,
canonicalisation) and the σ-propagation formula.
"""

_run_delegated_if_missing(
    "analysis_scratch/transmission_wind_tables.py",
    [Path("output/TABLES/ch05_transmission_wind_ratios.tex"),
     Path("output/TABLES/ch05_transmission_wind_amplitudes.tex")],
    label="ch05_transmission_wind_tables",
)

# %%
# [DATA: DELEG]  — analysis_scratch/t_cross_figure.py
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

_run_delegated_if_missing(
    "analysis_scratch/t_cross_figure.py",
    [Path("output/TEXFIGU/ch05_t_cross.tex"),
     *(Path(f"output/FIGURES/ch05_t_cross_{t}.pdf") for t in ("A1", "A2", "A3"))],
    label="ch05_t_cross",
)

# %%
# [DATA: META]
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
# Per-voltage figures (A1/A2/A3) AND the combined all-amplitudes figure
# (ch05_damping_ka) are all produced by analysis_scratch/damping_ka_per_volt.py.
# All four share the same axes/ticks/grid/colour scheme; the combined view
# uses marker shape (○ A1, □ A2, △ A3) to encode amplitude on top of the
# per-tag × wind colour encoding (blue/red for per240, turquoise/magenta
# for per40 — feedback_wind_color_convention.md).
_run_delegated_if_missing(
    "analysis_scratch/damping_ka_per_volt.py",
    [Path("output/TEXFIGU/ch05_damping_ka.tex"),
     Path("output/TEXFIGU/ch05_damping_ka_A1.tex"),
     Path("output/TEXFIGU/ch05_damping_ka_A2.tex"),
     Path("output/TEXFIGU/ch05_damping_ka_A3.tex"),
     Path("output/FIGURES/ch05_damping_ka.pdf"),
     *(Path(f"output/FIGURES/ch05_damping_ka_{t}.pdf") for t in ("A1", "A2", "A3"))],
    label="ch05_damping_ka_per_volt",
)

# %%
# [DATA: DELEG]  — analysis_scratch/damping_ka_per_volt_with_fit.py
"""
── CH05 § 4a — Damping vs ka with poly-2 fit per wind condition ─────────────
Same data and visual encoding as ch05_damping_ka_* above, with a degree-2
polynomial fit overlaid per (amplitude, wind), pooled across per_tag
(per240 + per40). Each per-amp subfigure carries an in-axis annotation with
n and R² per wind condition.

Reader observation that motivates this view: at A1, no-wind data tracks a
clean curve (R² ≈ 0.98) while with-wind data scatters more (R² ≈ 0.65)
even with more datapoints. The fits make that visible directly. The wind
× ka interaction tightens at A2 and A3 (R² > 0.86 either wind).

Combined view (`ch05_damping_ka_fit.pdf`) skips the fit overlay — 6
overlapping curves at all 3 amps would be too dense to add information.

Generated by analysis_scratch/damping_ka_per_volt_with_fit.py.
"""
_run_delegated_if_missing(
    "analysis_scratch/damping_ka_per_volt_with_fit.py",
    [Path("output/TEXFIGU/ch05_damping_ka_fit.tex"),
     Path("output/TEXFIGU/ch05_damping_ka_fit_A1.tex"),
     Path("output/TEXFIGU/ch05_damping_ka_fit_A2.tex"),
     Path("output/TEXFIGU/ch05_damping_ka_fit_A3.tex"),
     Path("output/FIGURES/ch05_damping_ka_fit.pdf"),
     *(Path(f"output/FIGURES/ch05_damping_ka_fit_{t}.pdf")
       for t in ("A1", "A2", "A3"))],
    label="ch05_damping_ka_per_volt_with_fit",
)

# %%
# [DATA: DELEG]  — analysis_scratch/mooring_focus_at_1_3hz_ka.py
"""
── CH05 § 4b — Mooring + panelretning at 1.30 Hz (single-page A4) ───────────
Three subfigures (A1/A2/A3) stacked vertically (subfig_layout="column") on
one A4 page float, generated by analysis_scratch/mooring_focus_at_1_3hz_ka.py.

Restricts to 1.30 Hz so reverse-panel data (only at 0.65 + 1.30 Hz, all on
above_50 mooring) can be compared side-by-side with normal panel data.
The dominant visual story is the **mooring effect** (canon below_90 vs
above_50: ΔK_t ≈ +0.04 nowind, +0.12 fullwind on full panel); panelretning
(normal vs revers) is a secondary observation visible only on above_50.

Encoding (per subfigure):
  hue   : mooring × wind — below_90 → blue/red (WIND_COLOR_MAP);
                            above_50 → cyan / bright pink
  shape : panelretning × amp — normal = ○ □ △ (A1/A2/A3);
                                revers = 6/5/4-point star (A1/A2/A3)
  fill  : all hollow (overlapping points readable)

Caption text comes from FIGURE_CAPTIONS / FIGURE_CAPTIONS_SHORT in this
file (looked up via output/.figure_captions.json).
"""
_run_delegated_if_missing(
    "analysis_scratch/mooring_focus_at_1_3hz_ka.py",
    [Path("output/TEXFIGU/ch05_mooring_focus_at_1_3hz_ka.tex"),
     *(Path(f"output/FIGURES/ch05_mooring_focus_at_1_3hz_ka_{t}.pdf")
       for t in ("A1", "A2", "A3"))],
    label="ch05_mooring_focus_at_1_3hz_ka",
)

# %%
# [DATA: DELEG]  — analysis_scratch/mooring_focus_at_1_3hz_table.py
"""
── CH05 § 4b — Mooring + panelretning at 1.30 Hz: companion table ───────────
Hard numbers for the figure above. Same data, same scope (1.30 Hz only,
panels ∈ {full, reverse}, moorings ∈ {below_90, above_50}). One row per
(amp, panel, mooring) cell with K_t per wind, ΔK_t in pp, K_t-økn %, D-red %.

The reverse · below_90 cell is honestly absent — reverse panel was never
run on the below_90 mooring.

Generated by analysis_scratch/mooring_focus_at_1_3hz_table.py. Writes:
    output/TABLES/ch05_mooring_focus_at_1_3hz_table.tex   (thesis include)
    analysis_scratch/mooring_focus_at_1_3hz_table.csv     (companion CSV)
"""
_run_delegated_if_missing(
    "analysis_scratch/mooring_focus_at_1_3hz_table.py",
    [Path("output/TABLES/ch05_mooring_focus_at_1_3hz_table.tex")],
    label="ch05_mooring_focus_at_1_3hz_table",
)

# %%
# [DATA: DELEG]  — analysis_scratch/full_vs_reverse_at_1_3hz_ka.py
"""
── CH05 § 4c — Combined K_t vs ka at 1.30 Hz (all 3 amps in one scatter) ────
Same data, scope, and visual language as §4b (mooring_focus_at_1_3hz_ka),
but with all three amplitudes pooled in a single scatter instead of
3 stacked subfigures.

x-axis = freq_to_k(1.30 Hz) × `IN Amplitude (FFT)` — paddle-only ka,
NOT the wind-contaminated `IN ka (FFT)` pipeline column. x/y limits
match ch05_damping_ka so the reader can stack the figures visually.

Encoding:
  hue   : mooring × wind — below_90 → blue/red (WIND_COLOR_MAP);
                            above_50 → cyan / bright pink
  shape : panelretning × amp — normal = ○ □ △ (A1/A2/A3);
                                revers = 6/5/4-point star
  fill  : all hollow

Caption text comes from FIGURE_CAPTIONS / FIGURE_CAPTIONS_SHORT in this
file (looked up via output/.figure_captions.json).
"""
_run_delegated_if_missing(
    "analysis_scratch/full_vs_reverse_at_1_3hz_ka.py",
    [Path("output/FIGURES/ch05_full_vs_reverse_at_1_3hz_ka.pdf"),
     Path("output/TEXFIGU/ch05_full_vs_reverse_at_1_3hz_ka.tex")],
    label="ch05_full_vs_reverse_at_1_3hz_ka",
)

# %%
# [DATA: DELEG]  — analysis_scratch/all_data_damping_scatter.py
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

_run_delegated_if_missing(
    "analysis_scratch/all_data_damping_scatter.py",
    [Path("output/FIGURES/ch05_damping_all_data_scatter.pdf"),
     Path("output/TEXFIGU/ch05_damping_all_data_scatter.tex")],
    label="ch05_damping_all_data_scatter",
)


# =============================================================================
# WAVE DETECTION (diagnostic, possibly CH04 § 6)
# =============================================================================

# %%
# [DATA: META]  — legacy exploration, superseded by CH04 §6 (ch04_first_arrival).
#                 Plotting all commented out; still computes _noise_floor from meta.
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


# %% ═══════════════════════════════════════════════════════════════════════════
# ███████████████████████████████████████████████████████████████████████████████
# █                                                                             █
# █   MEDIUM LOAD GATE — processed_dfs for the two canon March-2026 folders     █
# █   (~180 runs, ~12 MB, ~45 s first time). Enough for every [DFS-canon]       █
# █   cell below — §5, §6.                                                      █
# █                                                                             █
# █   If you only need the light figures, STOP here. All cells above this       █
# █   gate use combined_meta + FFT/PSD dicts only.                              █
# █                                                                             █
# ███████████████████████████████████████████████████████████████████████████████
# ═══════════════════════════════════════════════════════════════════════════════
# [DATA: DFS-canon gate]
#
# _loaded_dirs tracks which PROCESSED-* dirs are already in `processed_dfs`, so
# the heavy gate further down only reads the remaining folders and never re-
# deserialises canon. Both gates are idempotent — re-running the cell is safe.
_loaded_dirs: set  # forward declaration
try:
    _loaded_dirs  # noqa: F821
except NameError:
    _loaded_dirs = set()

if SKIP_DFS:
    print("Medium load gate — skipped (CLI: --skip-dfs). "
          "Exiting before any DFS-canon or diagnostic cells run.")
    sys.exit(0)

_canon_missing = [d for d in RESULTS_PROCESSED_DIRS if d not in _loaded_dirs]
if _canon_missing:
    print(f"Medium load gate — loading canon processed_dfs "
          f"({len(_canon_missing)} folder(s), ~12 MB, ~45 s first time)…")
    _t_gate = time.time()
    _new_dfs = load_processed_dfs(*_canon_missing)
    processed_dfs.update(_new_dfs)
    _loaded_dirs.update(_canon_missing)
    print(f"  +{len(_new_dfs)} DataFrames in {time.time() - _t_gate:.1f} s "
          f"(processed_dfs: {len(processed_dfs)} total)")
else:
    print(f"Medium load gate — canon already loaded "
          f"(processed_dfs: {len(processed_dfs)} total), skipping")


# %% !disabled  - this is perhaps redundant because of hg_per40_window_fitnes
# [DATA: DFS-canon]
# """
# ── CH04 § 5 — What does a full signal look like? ────────────────────────────
# Goal: show the full signal for a select few runs — stillwater baseline,
# wavemaker ramp, stable wavetrain, decay. Wind-wave noise visible at IN probe
# vs clean signal at OUT probe.

# Layout: rows = probes, columns = runs selected by filters.
# Grey band = detected stable-window used for all amplitude/FFT analysis.
# """

# _pv_timeseries = {
#     "filters": {
#         # Pick a representative condition — adjust as needed:
#         "WaveFrequencyInput [Hz]": 1.3,
#         "WaveAmplitudeInput [Volt]": 0.2,
#         "WindCondition": None,      # None = all wind conditions
#         "PanelCondition": "full",
#         # "run_category": "standard",
#     },
#     "plotting": {
#         "show_plot":   True,
#         "save_plot":   True,           # DRAFT — timeseries overview not yet polished
#         "draft":       True,
#         "figure_name": "ch04_timeseries_overview",
#         "force_stub":  True,
#         "probes":      ["9373/170", "12400/250"],   # IN and OUT only
#         "max_runs":    4,           # cap columns; reduce if too crowded
#         "xlim":        None,        # e.g. (0, 60) to zoom; None = full run
#         "ylim":        None,        # e.g. (-30, 30); None = auto per row
#         # caption printed on first run — paste the one-liner here:
#         # "caption": "...",
#     },
# }

# _fig_ts = plot_timeseries_overview(combined_meta, processed_dfs, _pv_timeseries)



# %% ═══════════════════════════════════════════════════════════════════════════
# ███████████████████████████████████████████████████████████████████████████████
# █                                                                             █
# █   HEAVY LOAD GATE — adds processed_dfs for the remaining (non-canon)        █
# █   folders (~23 folders, ~+2 min on top of the canon load above).            █
# █                                                                             █
# █   Only [DFS-all] cells below need this. Currently the only consumer is the  █
# █   D1 1.3 Hz cross-session consistency diagnostic (placeholder). If you are  █
# █   only producing CH04 §5 / §6 or anything above, SKIP this cell.            █
# █                                                                             █
# ███████████████████████████████████████████████████████████████████████████████
# ═══════════════════════════════════════════════════════════════════════════════
# [DATA: DFS-all gate]
if SKIP_HEAVY:
    print("Heavy load gate — skipped (CLI: --skip-heavy). "
          "Exiting before any DFS-all or diagnostic cells run.")
    sys.exit(0)

_remaining_dirs = [d for d in ALL_PROCESSED_DIRS if d not in _loaded_dirs]
if _remaining_dirs:
    print(f"Heavy load gate — loading remaining processed_dfs "
          f"({len(_remaining_dirs)} folder(s), ~65 MB, ~2 min)…")
    _t_gate = time.time()
    _new_dfs = load_processed_dfs(*_remaining_dirs)
    processed_dfs.update(_new_dfs)
    _loaded_dirs.update(_remaining_dirs)
    print(f"  +{len(_new_dfs)} DataFrames in {time.time() - _t_gate:.1f} s "
          f"(processed_dfs: {len(processed_dfs)} total)")
else:
    print(f"Heavy load gate — all folders already loaded "
          f"(processed_dfs: {len(processed_dfs)} total), skipping")


# %% ── HEAVY DELEG: §4 cells relocated below the heavy gate ─────────────────
# Cells that hang main_save_figures.py for minutes when REGENERATE_DELEGATED
# is on. They are subprocess-driven (do NOT consume `processed_dfs`), so
# their position is purely an execution-order convenience. The heavy gate is
# the natural breakpoint where you can stop a `--regen` pass without missing
# any §4 / §5 publication output.

# %%
# [DATA: DELEG]  — analysis_scratch/paddle_contamination_study.py
"""
── CH04 § 4L — Paddle-frequency IN contamination + window-size sensitivity ──
Three methodology questions on one 2×3 figure (rows: window-length / wind
correction; cols: 0.1, 0.2, 0.3 V):

  (A) Window-size sensitivity — OUT/IN vs N_periods ∈ {20, 40, 60, 100, full}.
      Flat curve ⇒ metric robust to window choice.
  (B) Incoherent wind-subtraction on A_in under fullwind, using PSD_wind
      estimated from (fullwind − nowind) residual PSD:
          A_in_corrected = √(max(0, A_in_fw² − E[A_wind²]))
  (C) T_cross cross-check: A_in_corrected(fw) vs A_in(nw) — closes the loop
      without relying on any single assumption.

Headline findings (see analysis_scratch/paddle_contamination_findings.md
and memory/methodology_wind_enhances_A_in.md):
  - Window drift < 0.012 across N sweep (metric robust).
  - Contamination ~2 % at the paddle bin.
  - Wind *enhances* A_in by 10–17 % at 1.5–1.6 Hz, 0.2–0.3 V — NOT
    spectral contamination. Reframes the T_cross vs OUT/IN_fw gap as a
    physically meaningful wind-on-IN effect.

Script writes output/FIGURES/ch04_paddle_contamination.pdf and
output/TEXFIGU/ch04_paddle_contamination.tex directly.
"""

_run_delegated_if_missing(
    "analysis_scratch/paddle_contamination_study.py",
    [Path("output/FIGURES/ch04_paddle_contamination.pdf"),
     Path("output/TEXFIGU/ch04_paddle_contamination.tex")],
    label="ch04_paddle_contamination",
)


# %% ── DIAGNOSTICS ───────────────────────────────────────────────────────────
# [DATA: META]  — [TODO] cells below are placeholders for diagnostic checks
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
# (Done 2026-04-24: x-axis is now pure wavenumber $k$ (rad/m), not $kL$.
#  See memory/feedback_kL_to_k_migration.md for the rename map.)

# %%
# [DATA: META]  — Quick sanity-check, prints only (no figure saved).
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
