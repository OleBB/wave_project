

# %%
# [DATA: META]  — [TODO] cell currently empty
# """
# ── CH04 § 2 — Stillwater timing (how long to wait between runs) ─────────────
# Goal: show that long-wave swell from previous runs decays over time, and that
# wind dramatically shortens the required waiting time.

# Data: repeated stillwater runs at different times after wave runs; look at
# low-frequency PSD content in eta_* columns over time.

# Figures:
#   - Plot:  PSD of eta at the OUT probe vs time-after-wave (semi-log, low freqs)
#   - Note:  wind-only runs show near-immediate settling(return to wind-wave spectrum) — physical explanation
#            (wind chops suppress long-wave coherence in the tank).
# """
# # TODO: implement stillwater timing figure
# _save_placeholder("ch04_stillwater_timing", "CH04 §2 — Stillwater timing", chapter="04")


# %% !disabled - this one is pretty old. Other psd plots have come further.
# [DATA: META]  — reads combined_psd_dict (loaded alongside combined_meta)
# """
# ── CH04 § 4-1 — Wind characterisation ─────────────────────────────────────────
# Goal: characterise what the wind does to the water surface — spectrum, spatial
# extent, interaction with the panel.

# Subtopics:
#   4a. Wind-wave PSD at each probe (broadband, 2–10 Hz dominant)
#   4b. Wind-only amplitude vs probe position (SNR context)
#   4c. Wind-only amplitude: IN probe (~10 mm) vs OUT probe (~0.9 mm) —
#       panel attenuates wind waves almost completely at 12400 mm
#   4d. Lateral coherence: cross-correlate /170 and /340 at same distance
#       (coherent = tank-wide fetch; incoherent = local turbulence)

# Data: combined_psd_dict (nowave entries), nowave+fullwind rows of combined_meta.

# Figures:
#   - Plot:  wind PSD per probe, fullwind vs stillwater overlay (log y-axis)
#   - Plot:  wind-only amplitude vs longitudinal distance, bar per probe
#   - Plot:  cross-correlation coefficient /170 vs /340 for fullwind runs
# """
# from wavescripts.filters import apply_experimental_filters as _aef

# _pv_wind_psd = {
#     "filters": {
#         "WaveFrequencyInput [Hz]": None,
#         "WindCondition":           None,
#         "PanelCondition":          None,
#         # exclude diagnostic/experimental runs by filename keyword
#         "exclude_run_keywords": ["nestenstille", "mstop"],
#     },
#     "plotting": {
#         "show_plot":     True,
#         "save_plot":     True,          # set True when ready
#         "figure_name":   "ch04_wind_psd",
#         "force_stub":    True,
#         "figsize":       (11, 4 * 4),
#         "linewidth":     1.0,
#         "facet_by":      "probe",
#         "probes":        ANALYSIS_PROBES,
#         "xlim":          (0, 5),
#         "logaritmic":    False,
#         "peaks":         0,
#         "max_points":    500,
#         "grid":          True,
#         "legend":        "inside",
#     },
# }

# _meta_nowave_all = combined_meta[combined_meta["WaveFrequencyInput [Hz]"].isna()].copy()
# _meta_nowave     = _aef(_meta_nowave_all, _pv_wind_psd)
# _nowave_paths    = set(_meta_nowave["path"])
# _wind_psd_dict   = {k: v for k, v in combined_psd_dict.items() if k in _nowave_paths}

# start = time.perf_counter()
# _fig_wind_psd, _ = plot_frequency_spectrum(
#     _wind_psd_dict, _meta_nowave, _pv_wind_psd, data_type="psd", chapter="04"
# )
# end = time.perf_counter()
# print(f"Wind PSD plot took {end - start:.4f} s")

# %%
# [DATA: META]  — [TODO] cell empty - reflection study has ben done in mansard-funke - this can be removed
# """
# ── CH04 § 4-2 — Wind wave-reflection from panel ─────────────────────────────────────────
# Goal: find out the reflection — spectrum, spatial
# extent, interaction with the panel.

# Data: combined_psd_dict (nowave entries), nowave+fullwind rows of combined_meta.

# Figures:
#   - Plot:
# """
# # TODO: implement wind reflection figure
# _save_placeholder("ch04_wind_reflection", "CH04 §4-2 — Wind reflection from panel", chapter="04")
#


# NOTE: CH04 §5b (ch04_timeseries_overview, grid of runs) and §6 (ch04_first_arrival)
# have been relocated to below the Heavy load gate at the bottom of this file.
# They are currently the only two figure cells that need processed_dfs (raw time series).

# %% !disabled but, TODO: consider repurposing this
# [DATA: META]
# """
# ── CH04 § 7 — Autocorrelation A: wavetrain stability ────────────────────────
# Goal: show wave_stability and period_cv as quality metrics. Demonstrate that
# fullwind + low amplitude (0.1 V) degrades IN probe stability, while OUT probe
# stays clean.

# Data: combined_meta, wave_stability {pos} and period_cv {pos} columns.

# Figures:
#   - Plot:  wave_stability vs frequency, faceted by probe, coloured by wind
#   - Plot:  period_cv vs frequency, same layout
#   - Note:  this motivates use of FFT amplitude (not time-domain) for OUT/IN
# """

# _pv_wave_stability = {
#     "filters": {
#         "min_periods":               10,
#         "WaveAmplitudeInput [Volt]": None,
#         "WaveFrequencyInput [Hz]":   (0.9,1.6),
#         "WindCondition":             None,
#         "PanelCondition":            "full",
#         # "run_category": "standard",   # re-enable after --force-recompute
#     },
#     "plotting": {
#         "show_plot":   True,
#         "save_plot":   True,          # DRAFT — wave stability not yet polished
#         "draft":       True,
#         "figure_name": "ch04_wave_stability",
#         "force_stub":  True,
#         "figsize":     (10, 3.5),
#         "probes":      ANALYSIS_PROBES,
#         # caption printed to terminal on first run — paste the one-liner here:
#         # "caption": "...",
#     },
# }

# _fig_stab = plot_wave_stability(combined_meta, ANALYSIS_PROBES, _pv_wave_stability)

# %% !disabled
# [DATA: META]
# """
# ── CH04 § 8 — Autocorrelation B: lateral wave equality ──────────────────────
# Goal: show that the paddle wave is laterally uniform (parallel probes agree)
# under no-wind conditions, and that full wind introduces lateral asymmetry.

# Data: combined_meta, parallel_ratio column, wave_stability columns.

# Figures:
#   - Plot:  parallel_ratio vs frequency, no-wind runs (should be ~1.0)
#   - Plot:  parallel_ratio vs frequency, fullwind runs (asymmetry visible?)
#   - Table: mean parallel_ratio ± std by (WindCondition, frequency)
# """

# # Lateral equality uses the same plot_parallel_ratio function (already defined in §3),
# # but filtered to a single wind condition at a time for the per-wind breakdown.
# _pv_lateral_nowind = {
#     "filters": {"WindCondition": "no", "run_category": "standard"},
#     "plotting": {
#         "show_plot":   True,
#         "save_plot":   True,          # DRAFT — lateral equality not yet polished
#         "draft":       True,
#         "figure_name": "ch04_lateral_nowind",
#         "force_stub":  True,
#     },
# }
# _fig_lat_nw = plot_parallel_ratio(combined_meta, _pv_lateral_nowind)

# _pv_lateral_nowind_scatter = {
#     "filters": {**_pv_lateral_nowind["filters"]},
#     "plotting": {
#         **_pv_lateral_nowind["plotting"],
#         "scatter":     True,
#         "figure_name": "ch04_lateral_nowind_scatter",
#     },
# }
# plot_parallel_ratio(combined_meta, _pv_lateral_nowind_scatter)

# %% !disabled
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



# %% !disabled - not really relevant as a plot
# [DATA: DFS-canon]
# """
# ── CH04 § 6 — Wave-range detection ──────────────────────────────────────────
# Goal: explain and validate _SNARVEI_CALIB. Show how the stable wavetrain
# window is detected: (1) threshold crossing, (2) ramp-up skip, (3) n periods.

# Data: processed_dfs, Computed Probe {pos} start/end columns.

# Figures:
#   - Plot:  single run with detected start/end marked, one probe panel per row
#   - Plot:  start sample vs frequency (all probes) — show _SNARVEI_CALIB points
# """

# _pv_first_arrival = {
#     "filters": {},
#     "plotting": {
#         "show_plot":        True,
#         "save_plot":        True,       # DRAFT — threshold not yet calibrated
#         "draft":            True,
#         "figure_name":      "ch04_first_arrival",
#         "force_stub":       True,
#         "probes":           ANALYSIS_PROBES,
#         "threshold_factor": 5.0,        # TODO: calibrate per-probe after noise floor analysis
#         "window_s":         2.5,
#         "min_arrival_s":    0.5,
#         "figsize":          (9, 5),
#     },
# }

# plot_first_arrival(combined_meta, processed_dfs, _pv_first_arrival, chapter="04")
