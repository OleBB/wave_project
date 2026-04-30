#!/usr/bin/env python3
# %%
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

Every cell starts with a `# [DATA: X]` line matching one of the five tags. If
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
  §1    ch04_probe_noise_floor           [META]  ~  Stillwater noise floor per probe / hw config
  §2    ch04_stillwater_timing           [META]  ✗  Swell decay time vs wait time  [TODO]
  §3    ch04_parallel_ratio              [META]  ~  Wall/far-side amplitude ratio vs frequency
        ch04_parallel_ratio_scatter      [META]  ~     └─ per-run scatter sibling
  §3b   ch04_probe_height                [DELEG] ✓  Probe height & range-mode validity
  §3c   ch04_mooring_comparison          [DELEG] ✓  Mooring rubber band length: loose230 vs loose300
  §3d   ch04_sound_speed                 [META]  ~  Speed-of-sound / lab temperature drift
  §3e   ch04_parallel_probe_agreement    [DELEG] ✓  9373/170 vs 9373/340 — mean-IN canonical ref
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
  §4L   ch04_paddle_contamination        [DELEG] ~  Paddle-freq IN contamination + wind correction + T_cross validation
  §4m   ch04_per40_and_per240_HG_shifted [DELEG] ~  Per40+per240 pooling under probe-shifted H&G window (the answer: yes)
  §4n   ch04_hg_per40_window_fitness_f{13,14,15,16}  [DELEG] ~  Proposed H&G window (N=15 + UC-snap) overlaid on η(t), per40+per240, all 4 thesis freqs
        ch04_window_intervals            [DELEG] ~     └─ companion table: IN/OUT window intervals + samples-per-period at each thesis freq
  §5    ch04_inspirational_nowind        [DELEG] ~  Reading a time series — nowind canon (macro + 5-period zoom)
        ch04_inspirational_fullwind      [DELEG] ~     └─ same layout, fullwind canon (wind-wave clutter visible at IN)
  §5b   ch04_timeseries_overview         [DFS-canon] ~  Full time-series with stable-window band  (below MEDIUM gate)
  §6    ch04_first_arrival               [DFS-canon] ~  First wave arrival vs probe distance      (below MEDIUM gate)
  §7    ch04_wave_stability              [META]  ~  Wave stability and period_cv vs frequency
  §8    ch04_lateral_nowind              [META]  ~  Lateral equality (parallel ratio, no-wind)
        ch04_lateral_nowind_scatter      [META]  ~     └─ per-run scatter sibling
  §9    ch04_amplitude_profile           [META]  ✗  Amplitude at every probe, all runs  [cell commented]

CHAPTER 05 — RESULTS
  §1    ch05_damping_freq                [META]  ✓  OUT/IN (FFT) vs frequency  ← primary result
  §2    ch05_damping_scatter             [META]  ✓  OUT/IN scatter vs amplitude
  §3    ch05_wind_effect_table           [DELEG] ~  Wind effect: per-(freq,amp) Δτ + % gains/reductions table
  §3b   ch05_t_cross                     [DELEG] ✓  T_cross: honest wind effect via clean nowind ref
  §4    ch05_damping_ka                  [META]  ~  Damping vs ka (wavenumber × amplitude)
        ch05_damping_ka_{A1,A2,A3}       [DELEG] ✓     └─ standalone per-amplitude-tier (per240+per40, magenta palette)
  §5    ch05_swell_scatter               [META]  —  DROPPED (cell commented in place)
  §6    (moved → CH04 §4-3b as ch04_reconstructed)
  §7    ch05_damping_all_data_scatter    [DELEG] ✓  Supplementary: OUT/IN across ALL conditions

DIAGNOSTICS
  D1    diag_13hz_consistency            [META]  ✗  1.3 Hz cross-session consistency  [TODO]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# %% ── dev: reload modules (run this cell after editing wavescripts/) ─────────
import importlib, wavescripts.plotter as _pm, wavescripts.filters as _fm

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

    # § 1 — Probe noise floor (parent + 4 subfigs)
    "ch04_probe_noise_floor":          "",
    "ch04_probe_noise_floor_group0":   "",
    "ch04_probe_noise_floor_group1":   "",
    "ch04_probe_noise_floor_group2":   "",
    "ch04_probe_noise_floor_group3":   "",

    # § 2 — Stillwater timing (placeholder)
    "ch04_stillwater_timing":          "",

    # § 3 — Probe placement / parallel probes
    "ch04_parallel_ratio":             "",
    "ch04_parallel_ratio_scatter":     "",
    "ch04_probe_height":               "",
    "ch04_mooring_comparison":         "",
    "ch04_sound_speed":                "",
    "ch04_parallel_probe_agreement":   "",
    "ch04_depth_regime":               "",

    # § 4 — Wind characterisation / FFT methodology
    "ch04_wind_psd":                   "",
    "ch04_wind_reflection":            "",
    "ch04_fft_wave":                   "Frekvensspekter for bølge på \qty{1.4}{\hertz}, amplitudevalg $A_2$.",
    "ch04_reconstructed":              "Hovedmoden fra bølgen \qty{1.4}{\hertz}, amplitudevalg $A_2$, resten av signalet er separert ut. Stablet ovenfra og ned: Innkommende uten vind, Utgående uten vind, Innkommende med vind, Utgående med vind. Felles y-akse for alle fire paneler.",
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

    # § 5 — Reading a time series (inspirational opener)
    "ch04_inspirational_nowind":       "Tidsserie for bølgen \qty{1.4}{\hertz}, amplitudevalg $A_2$, uten vind.",
    "ch04_inspirational_fullwind":     "Tidsserie for bølgen \qty{1.4}{\hertz}, amplitudevalg $A_2$, med vind.",

    # § 6–9 — Wave-range detection, autocorrelation, lateral
    "ch04_first_arrival":              "",
    "ch04_timeseries_overview":        "",
    "ch04_wave_stability":             "",
    "ch04_lateral_nowind":             "",
    "ch04_lateral_nowind_scatter":     "",

    # ── CHAPTER 05 — RESULTS ─────────────────────────────────────────────────

    # § 1 — Damping vs frequency (parent + 3 subfigs)
    "ch05_damping_freq":               "",
    "ch05_damping_freq_full_A1":       "",
    "ch05_damping_freq_full_A2":       "",
    "ch05_damping_freq_full_A3":       "",

    # § 2 — Damping vs amplitude
    "ch05_damping_scatter":            "",

    # § 3 — Wind effect (table; replaces the old scatter ch05_damping_wind_delta,
    #         archived 2026-04-28).
    "ch05_wind_effect_table":          "",

    # § 3b — T_cross (parent + 3 subfigs)
    "ch05_t_cross":                    "",
    "ch05_t_cross_A1":                 "",
    "ch05_t_cross_A2":                 "",
    "ch05_t_cross_A3":                 "",

    # § 4 — Damping vs ka
    "ch05_damping_ka":                 "",
    "ch05_damping_ka_A1":              "",
    "ch05_damping_ka_A2":              "",
    "ch05_damping_ka_A3":              "",

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
    "ch04_probe_noise_floor":          "",
    "ch04_stillwater_timing":          "",
    "ch04_parallel_ratio":             "",
    "ch04_parallel_ratio_scatter":     "",
    "ch04_probe_height":               "",
    "ch04_mooring_comparison":         "",
    "ch04_sound_speed":                "",
    "ch04_parallel_probe_agreement":   "",
    "ch04_depth_regime":               "",
    "ch04_wind_psd":                   "",
    "ch04_wind_reflection":            "",
    "ch04_fft_wave":                   "",
    "ch04_reconstructed":              "Rekonstrukerte bølger",
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
    "ch04_inspirational_nowind":       "",
    "ch04_inspirational_fullwind":     "",
    "ch04_first_arrival":              "",
    "ch04_timeseries_overview":        "",
    "ch04_wave_stability":             "",
    "ch04_lateral_nowind":             "",
    "ch04_lateral_nowind_scatter":     "",

    # ── CHAPTER 05 ───────────────────────────────────────────────────────────
    "ch05_damping_freq":               "",
    "ch05_damping_scatter":            "",
    "ch05_wind_effect_table":          "",
    "ch05_t_cross":                    "",
    "ch05_damping_ka":                 "",
    "ch05_damping_ka_A1":              "",
    "ch05_damping_ka_A2":              "",
    "ch05_damping_ka_A3":              "",
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
# %%
# [DATA: META]
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
        # ↓↓↓ NEW — add this block ↓↓↓
            "text": {
                "ylabel": "Støyamplitude (95 %)  [mm]",
                "legend_mean_amp":    "Gjennomsnittlig støyamplitude  (±1σ)",
                "legend_per_run":     "Per kjøring",
                "legend_threshold":   "Terskel  max({k_sigma:.0f}σ, {k_q:.0f}q)  [mm]",
                "legend_quantization":"Halvt kvantiseringssteg  q/2  [mm]",
                "legend_highlight":   "Uthevet kjøring  ({highlight_keyword})",
                "legend_excluded":    "Ekskludert (ikke satt seg)",
                "title": {
                    "h272 / high": "h=272 mm — referanseoppsett",
                    "h100 / low":  "h=100 mm - endelig oppsett ",
                    # groups not listed here keep their default title
                },
            },
            # ↑↑↑ end new block ↑↑↑
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
# [DATA: META]  — [TODO] cell currently empty
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
# [DATA: META]
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
    },
}
plot_parallel_ratio(combined_meta, _pv_parallel_ratio_scatter)

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
        "draft":       True,
        "figure_name": "ch04_sound_speed",
        "force_stub":  True,
        "figsize":     (10, 3),
    },
}

plot_sound_speed(combined_meta, _pv_sound_speed, chapter="04")

# %%
# [DATA: DELEG]  — analysis_scratch/parallel_probe_agreement.py
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

_run_delegated_if_missing(
    "analysis_scratch/parallel_probe_agreement.py",
    [Path("output/FIGURES/ch04_parallel_probe_agreement.pdf"),
     Path("output/TEXFIGU/ch04_parallel_probe_agreement.tex")],
    label="ch04_parallel_probe_agreement",
)

# %%
# [DATA: DELEG]  — subprocess-calls analysis_scratch/depth_regime_map.py
"""
── CH04 § 3f — Depth-regime map (kd vs f) at d = 0.58 m ─────────────────────
Two-panel figure that justifies the full dispersion relation ω² = g·k·tanh(kd)
used throughout the pipeline (wavescripts.constants.c_group,
wavescripts.plot_utils.freq_to_k).

    Top panel: kd vs paddle frequency at d = 580 mm. Shaded regime bands
    (deep / intermediate / shallow) with thresholds kd=π and kd=π/10.
    Run-frequency markers sized by n_runs, coded by regime (colour + shape).
    Vertical band marks the thesis scope (1.3–1.6 Hz). Secondary right-hand
    axis shows wavelength λ in metres.

    Bottom panel: relative error in λ if the deep-water approximation
    λ_deep = g/(2π f²) were used instead of full dispersion, as % on a
    symlog y-axis. Quantifies "by how much does the full dispersion matter".

At 580 mm depth, thesis-scope runs (1.3–1.6 Hz) are comfortably deep
(kd > π, deep-water approximation error < 0.11 %). Sub-1 Hz frequencies
drift into intermediate water (kd < π at 1.0 Hz; up to ~15 % λ-error at
0.7 Hz) — consistent with the bottom-motion observation from 2026-03-12
annotated on the figure. Scope boundary `f < 1 Hz out of scope`
(MEMORY.md) has a direct physical rationale visible here.

Delegated build — see analysis_scratch/depth_regime_map.py.
"""
_run_delegated_if_missing(
    "analysis_scratch/depth_regime_map.py",
    [Path("output/FIGURES/ch04_depth_regime.pdf"),
     Path("output/TEXFIGU/ch04_depth_regime.tex")],
    label="ch04_depth_regime",
)

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
# [DATA: META]  — reads combined_psd_dict (loaded alongside combined_meta)
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
# [DATA: META]  — [TODO] cell currently empty
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

# %%
# [DATA: CSV]  — analysis_scratch/fft_peak_bias_outin_impact.py regens CSV;
#                then plotter reads it
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

# The plotter reads a per-run CSV that the scratch script generates;
# regenerate the CSV via subprocess if it's missing, then call the plotter.
_run_delegated_if_missing(
    "analysis_scratch/fft_peak_bias_outin_impact.py",
    [Path("analysis_scratch/fft_peak_bias_outin_impact.csv")],
    label="fft_peak_bias_outin_impact.csv",
)
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

# %%
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

# NOTE: CH04 §5b (ch04_timeseries_overview, grid of runs) and §6 (ch04_first_arrival)
# have been relocated to below the Heavy load gate at the bottom of this file.
# They are the only two figure cells that need processed_dfs (raw time series).

# %%
# [DATA: META]
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
# [DATA: META]
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
    },
}
_fig_lat_nw = plot_parallel_ratio(combined_meta, _pv_lateral_nowind)

_pv_lateral_nowind_scatter = {
    "filters": {**_pv_lateral_nowind["filters"]},
    "plotting": {
        **_pv_lateral_nowind["plotting"],
        "scatter":     True,
        "figure_name": "ch04_lateral_nowind_scatter",
    },
}
plot_parallel_ratio(combined_meta, _pv_lateral_nowind_scatter)

# %%
# [DATA: META]  — cell body currently commented out ("perhaps skip this one")
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
_damping_grouped = damping_all_amplitude_grouper(_damping_meta)
plot_damping_freq(_damping_grouped, _pv_damping_freq)

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
        "show_plot":   True,
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

    τ_nw                           = mean OUT/IN(FFT) at no-wind
    τ_fw                           = mean OUT/IN(FFT) at full-wind
    Δτ        = τ_fw − τ_nw        (signed transmission change, pp)
    % T-gain  = Δτ / τ_nw · 100    (relative transmission change)
    % D-red   = (D_nw − D_fw)/D_nw · 100   where D = 1 − τ
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
        "force_stub": True,
    },
}

_ka_meta = apply_experimental_filters(meta_results, _pv_damping_ka)
from wavescripts.plotter import plot_damping_ka
plot_damping_ka(_ka_meta, _pv_damping_ka, chapter="05")

# Per-voltage standalone variants — three independent figures (one per paddle
# voltage 0.10 / 0.20 / 0.30 V), each with its own TEXFIGU stub so the thesis
# caption is fully hand-authored per panel (\caption{} body left blank; a
# suggested draft is parked in the stub's IMMUTABLE extra_params for reference).
# Also combines per240 and per40 runs: per240 uses the canonical thesis blue/red
# (WIND_COLOR_MAP), per40 uses turquoise (#00D4BC, nowind) + magenta (#D946EF,
# fullwind) for visual separation without breaking the wind-colour convention.
# Delegated build — see analysis_scratch/damping_ka_per_volt.py for details.
_run_delegated_if_missing(
    "analysis_scratch/damping_ka_per_volt.py",
    [Path("output/TEXFIGU/ch05_damping_ka_A1.tex"),
     Path("output/TEXFIGU/ch05_damping_ka_A2.tex"),
     Path("output/TEXFIGU/ch05_damping_ka_A3.tex"),
     *(Path(f"output/FIGURES/ch05_damping_ka_{t}.pdf") for t in ("A1", "A2", "A3"))],
    label="ch05_damping_ka_per_volt",
)

# %%
# [DATA: META]  — cell body DROPPED (kept as a marker)
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


# %%
# [DATA: DFS-canon]
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
# [DATA: DFS-canon]
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
