

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
