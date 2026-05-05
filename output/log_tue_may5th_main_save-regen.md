(draumkvedet) ole@eduroam-193-157-165-233 wave_project % python main_save_figures.py --regenerate
Note: gap between 'nov_normalt_oppsett' and 'march2026_rearranging' (2026-01-01 00:00:00 – 2026-03-04 00:00:00), no config needed if no data exists
✓ Validated 4 probe configurations
Loading analysis data (meta + FFT/PSD; deferring processed_dfs)...
   No spectra cache — recomputing FFT/PSD...
   No spectra cache — recomputing FFT/PSD...
   Loaded 2 FFT / 3 PSD spectra from cache
   Loaded 2 FFT / 2 PSD spectra from cache
   Loaded 4 FFT / 4 PSD spectra from cache
   Loaded 1 FFT / 4 PSD spectra from cache
   Loaded 6 FFT / 7 PSD spectra from cache
   Loaded 9 FFT / 9 PSD spectra from cache
   Loaded 6 FFT / 8 PSD spectra from cache
   Loaded 9 FFT / 10 PSD spectra from cache
   Loaded 12 FFT / 12 PSD spectra from cache
   Loaded 12 FFT / 16 PSD spectra from cache
   Loaded 22 FFT / 23 PSD spectra from cache
   Loaded 22 FFT / 28 PSD spectra from cache
   Loaded 31 FFT / 31 PSD spectra from cache
   Loaded 26 FFT / 36 PSD spectra from cache
   Loaded 33 FFT / 36 PSD spectra from cache
   Loaded 33 FFT / 36 PSD spectra from cache
   Loaded 34 FFT / 44 PSD spectra from cache
   Loaded 35 FFT / 43 PSD spectra from cache
   Loaded 41 FFT / 43 PSD spectra from cache
   Loaded 48 FFT / 54 PSD spectra from cache
   Loaded 50 FFT / 60 PSD spectra from cache
   Loaded 73 FFT / 86 PSD spectra from cache
load_analysis_data: 609 rows, 511 FFT experiments
/Users/ole/Kodevik/wave_project/wavescripts/plotter.py:2374: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
  plt.show()
/Users/ole/Kodevik/wave_project/wavescripts/plotter.py:2374: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
  plt.show()
/Users/ole/Kodevik/wave_project/wavescripts/plotter.py:2374: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
  plt.show()
/Users/ole/Kodevik/wave_project/wavescripts/plotter.py:2374: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
  plt.show()

[plot_probe_noise_floor] text slots: ['legend_excluded', 'legend_highlight', 'legend_mean_amp', 'legend_per_run', 'legend_threshold', 'title', 'ylabel']
[plot_probe_noise_floor] WARNING: unused text override key(s) ['legend_quantization'] — typo? known slots: ['legend_excluded', 'legend_highlight', 'legend_mean_amp', 'legend_per_run', 'legend_threshold', 'title', 'ylabel']
  Saved: output/FIGURES/ch04_probe_noise_floor_group0.pdf
  Saved: output/FIGURES/ch04_probe_noise_floor_group1.pdf
  Saved: output/FIGURES/ch04_probe_noise_floor_group2.pdf
  Saved: output/FIGURES/ch04_probe_noise_floor_group3.pdf
  Stub created: ch04_probe_noise_floor.tex

=== Probe noise floor summary [mm] ===
          group      probe  mean_level_mm  noise_rms_mm  noise_95pct_amp_mm  noise_95pct_std_mm  n_runs  quantization_step_mm  detection_threshold_mm  bias_vs_ref_mm
0   h272 / high   9373/170       271.9400        0.1404              0.2891              0.0843      20                   NaN                  0.4212         -0.1302
1   h272 / high  12400/250       272.3381        0.1245              0.2633              0.0792      16                   NaN                  0.3735          0.2680
2   h272 / high   9373/340       271.9335        0.0638              0.1283              0.1069      20                   NaN                  0.1913         -0.1367
3   h272 / high   8804/250       272.0690        0.1155              0.2430              0.0878      20                   NaN                  0.3464         -0.0012
4   h136 / high   9373/170       136.3400        0.0445              0.0450                 NaN       1                   NaN                  0.1334          0.2300
5   h136 / high  12400/250       135.9200        0.0307              0.0900                 NaN       1                   NaN                  0.0920         -0.1900
6   h136 / high   9373/340       136.2600        0.0445              0.0450                 NaN       1                   NaN                  0.1336          0.1500
7   h136 / high   8804/250       135.9200        0.0250              0.0900                 NaN       1                   NaN                  0.0749         -0.1900
8   h100 / high   9373/170       100.5206        0.0676              0.1500              0.1299      17                   NaN                  0.2028          0.1193
9   h100 / high  12400/250       100.7206        0.0421              0.0868              0.0609      17                   NaN                  0.1264          0.3193
10  h100 / high   9373/340       100.1665        0.0691              0.1235              0.0867      17                   NaN                  0.2073         -0.2349
11  h100 / high   8804/250       100.1976        0.0722              0.1429              0.0846      17                   NaN                  0.2167         -0.2037
12   h100 / low   9373/170       100.8644        0.0547              0.1016              0.0816       9                   NaN                  0.1642          0.0778
13   h100 / low  12400/250       101.0533        0.0574              0.0872              0.0642       9                   NaN                  0.1721          0.2667
14   h100 / low   9373/340       100.6144        0.0403              0.0935              0.1151       9                   NaN                  0.1210         -0.1722
15   h100 / low   8804/250       100.6144        0.0303              0.0644              0.0904       9                   NaN                  0.0909         -0.1722
probe uncertainty-plot took 1.1127 s
  ch04_probe_height: running analysis_scratch/probe_height_figure.py (REGENERATE_DELEGATED=True)
    ch04_probe_height: regenerated 2 output(s)
  ch04_mooring_comparison: running analysis_scratch/mooring_comparison.py (REGENERATE_DELEGATED=True)
    ch04_mooring_comparison: regenerated 2 output(s)
/Users/ole/Kodevik/wave_project/wavescripts/plotter.py:3025: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
  plt.show()
  Saved: output/FIGURES/ch04_sound_speed.pdf
  Stub created: ch04_sound_speed.tex
  ch04_parallel_probe_agreement_by_freq: running analysis_scratch/parallel_probe_agreement_by_freq.py (REGENERATE_DELEGATED=True)
    ch04_parallel_probe_agreement_by_freq: regenerated 2 output(s)
  ch04_parallel_probe_psd_agreement: running analysis_scratch/parallel_probe_psd_agreement.py (REGENERATE_DELEGATED=True)
    ch04_parallel_probe_psd_agreement: regenerated 1 output(s)
  ch04_parallel_probe_psd_agreement_simple: running analysis_scratch/parallel_probe_psd_agreement_simple.py (REGENERATE_DELEGATED=True)
    ch04_parallel_probe_psd_agreement_simple: regenerated 1 output(s)
  ch04_fft_wave: running analysis_scratch/fft_wave_spectrum.py (REGENERATE_DELEGATED=True)
    ch04_fft_wave: FAILED (rc=1); stderr tail: Traceback (most recent call last):
  File "/Users/ole/Kodevik/wave_project/analysis_scratch/fft_wave_spectrum.py", line 239, in <module>
    fig.savefig(out_pgf, bbox_inches="tight")
                ^^^^^^^
NameError: name 'out_pgf' is not defined. Did you mean: 'out_pdf'?
  [✓] quality_flag gate: excluded 6 flagged run(s) (use quality_flag='all' to include)

--- Starting Filter Process (126 rows) ---
  [✓] WaveAmplitudeInput [Volt] == 0.2........................ kept 32 rows (removed 94)
  [✓] WaveFrequencyInput [Hz].. == 1.4........................ kept 5 rows (removed 27)
  [✓] WindCondition............ isin(['no', 'full']).......... kept 5 rows (removed 0)
  [✓] PanelCondition........... == full....................... kept 5 rows (removed 0)
--- Filter Final: 5 rows remaining (Total removed: 121) ---

  Saved: output/FIGURES/ch04_reconstructed.pdf
  Stub created: ch04_reconstructed.tex
  [✓] quality_flag gate: excluded 29 flagged run(s) (use quality_flag='all' to include)

--- Starting Filter Process (489 rows) ---
--- Filter Final: 489 rows remaining (Total removed: 0) ---

/Users/ole/Kodevik/wave_project/wavescripts/plotter.py:3173: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
  plt.show()
  Saved: output/FIGURES/ch04_wind_snr.pdf
  Stub created: ch04_wind_snr.tex
  [✓] quality_flag gate: excluded 29 flagged run(s) (use quality_flag='all' to include)

--- Starting Filter Process (489 rows) ---
  [✓] min_periods < 10: excluded 40 run(s) with too-short analysis window
--- Filter Final: 449 rows remaining (Total removed: 40) ---

/Users/ole/Kodevik/wave_project/wavescripts/plotter.py:3293: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
  plt.show()
  Saved: output/FIGURES/ch04_td_vs_fft.pdf
  Stub created: ch04_td_vs_fft.tex
  [✓] quality_flag gate: excluded 29 flagged run(s) (use quality_flag='all' to include)

--- Starting Filter Process (489 rows) ---
  [✓] min_periods < 10: excluded 40 run(s) with too-short analysis window
--- Filter Final: 449 rows remaining (Total removed: 40) ---

/Users/ole/Kodevik/wave_project/wavescripts/plotter.py:3293: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
  plt.show()
  Saved: output/FIGURES/ch04_td_vs_fft_scatter.pdf
  Stub created: ch04_td_vs_fft_scatter.tex
  ch04_mansard_funke_reflection: running analysis_scratch/mansard_funke.py (REGENERATE_DELEGATED=True)
    ch04_mansard_funke_reflection: regenerated 2 output(s)
  ch04_sw_correction_test: running analysis_scratch/sw_correction.py (REGENERATE_DELEGATED=True)
    ch04_sw_correction_test: regenerated 2 output(s)
  ch04_sliding_afft_stability: running analysis_scratch/sliding_afft_fullwind_sweep.py (REGENERATE_DELEGATED=True)
    ch04_sliding_afft_stability: regenerated 2 output(s)
  ch04_reconstruction_AvsB: running analysis_scratch/reconstruction_A_vs_B.py (REGENERATE_DELEGATED=True)
    ch04_reconstruction_AvsB: regenerated 4 output(s)
  ch04_reconstruction_pure_wind: running analysis_scratch/reconstruction_A_vs_B.py (REGENERATE_DELEGATED=True)
    ch04_reconstruction_pure_wind: regenerated 2 output(s)
  ch04_fft_method_comparison: running analysis_scratch/fft_method_comparison.py (REGENERATE_DELEGATED=True)
    ch04_fft_method_comparison: regenerated 2 output(s)
  ch04_fft_window_length_sens: running analysis_scratch/fft_window_sensitivity_lsfit.py (REGENERATE_DELEGATED=True)
    ch04_fft_window_length_sens: regenerated 2 output(s)
  ch04_fft_window_position_sens: running analysis_scratch/fft_window_position_sensitivity_lsfit.py (REGENERATE_DELEGATED=True)
    ch04_fft_window_position_sens: regenerated 2 output(s)
  ch04_fft_window_position_trace: running analysis_scratch/fft_window_position_sensitivity_trace.py (REGENERATE_DELEGATED=True)
    ch04_fft_window_position_trace: regenerated 1 output(s)
  ch04_per40_and_per240_HG_shifted: running analysis_scratch/per40_and_per240_HG_shifted.py (REGENERATE_DELEGATED=True)
    ch04_per40_and_per240_HG_shifted: regenerated 2 output(s)
  ch04_hg_per40_window_fitness: running analysis_scratch/hg_per40_window_fitness.py (REGENERATE_DELEGATED=True)
    ch04_hg_per40_window_fitness: regenerated 8 output(s)
  ch04_window_intervals: running analysis_scratch/window_intervals_table.py (REGENERATE_DELEGATED=True)
    ch04_window_intervals: regenerated 1 output(s)
  ch04_window_choice: running analysis_scratch/window_choice_figure.py (REGENERATE_DELEGATED=True)
    ch04_window_choice: regenerated 2 output(s)
  ch04_window_choice_table: running analysis_scratch/window_choice_table.py (REGENERATE_DELEGATED=True)
    ch04_window_choice_table: regenerated 2 output(s)
  ch04_plateau_overview: running analysis_scratch/plateau_overview.py (REGENERATE_DELEGATED=True)
    ch04_plateau_overview: regenerated 6 output(s)
  ch04_plateau_values: running analysis_scratch/plateau_values_table.py (REGENERATE_DELEGATED=True)
    ch04_plateau_values: regenerated 1 output(s)
  ch04_tidsvindu: running analysis_scratch/tidsvindu_table.py (REGENERATE_DELEGATED=True)
    ch04_tidsvindu: regenerated 1 output(s)
  ch04_wind_transition_overview: running analysis_scratch/wind_decay_timeseries.py (REGENERATE_DELEGATED=True)
    ch04_wind_transition_overview: regenerated 5 output(s)
  ch04_wind_pre_paddle_psd: running analysis_scratch/wind_2s_vs_360s.py (REGENERATE_DELEGATED=True)
    ch04_wind_pre_paddle_psd: regenerated 2 output(s)
  ch04_wind_pre_paddle_table: running analysis_scratch/wind_pre_paddle_table.py (REGENERATE_DELEGATED=True)
    ch04_wind_pre_paddle_table: regenerated 1 output(s)
  ch04_wind_qc_3s_inputs: running analysis_scratch/wind_qc_3s.py (REGENERATE_DELEGATED=True)
    ch04_wind_qc_3s_inputs: regenerated 1 output(s)
  ch04_wind_qc_thesis: running analysis_scratch/wind_qc_3s_thesis.py (REGENERATE_DELEGATED=True)
    ch04_wind_qc_thesis: regenerated 4 output(s)
  ch04_wind_setup_baseline_inputs: running analysis_scratch/wind_setup_baseline_3v3.py (REGENERATE_DELEGATED=True)
    ch04_wind_setup_baseline_inputs: regenerated 1 output(s)
  ch04_wind_setup_baseline_table: running analysis_scratch/wind_setup_baseline_3v3_table.py (REGENERATE_DELEGATED=True)
    ch04_wind_setup_baseline_table: regenerated 1 output(s)
  ch04_inspirational_timeseries: running analysis_scratch/inspirational_timeseries.py (REGENERATE_DELEGATED=True)
    ch04_inspirational_timeseries: regenerated 4 output(s)
  [✓] quality_flag gate: excluded 6 flagged run(s) (use quality_flag='all' to include)

--- Starting Filter Process (126 rows) ---
  [✓] WaveAmplitudeInput [Volt] range(0.1 to 0.3)............. kept 103 rows (removed 23)
  [✓] WaveFrequencyInput [Hz].. range(1.3 to 1.6)............. kept 80 rows (removed 23)
--- Filter Final: 80 rows remaining (Total removed: 46) ---

Input dataframe shape: (80, 650)
Unique PanelCondition: ['full']
Unique WindCondition: ['full', 'no']

Number of unique grouping combinations: 24

Group sizes (rows per group):
count    24.000000
mean      3.333333
std       2.078182
min       2.000000
25%       2.000000
50%       3.000000
75%       3.250000
max      10.000000
dtype: float64

After aggregation — stats shape: (24, 20)
Columns in stats: ['WaveAmplitudeInput [Volt]', 'WaveFrequencyInput [Hz]', 'PanelCondition', 'WindCondition', 'mean_out_in', 'std_out_in', 'n_runs', 'paths', 'mean_kL', 'mean_A_8804/250', 'mean_A_9373/170', 'mean_A_9373/340', 'mean_A_12400/250', 'in_probes_used', 'out_probes_used', 'in_position', 'out_position', 'ain_disagree_frac_mean', 'aout_disagree_frac_mean', 'file_dates']

WARNING: 10 groups have ≤ 2 runs:
    WaveAmplitudeInput [Volt]  WaveFrequencyInput [Hz] PanelCondition WindCondition  n_runs
3                         0.1                      1.4           full            no       2
5                         0.1                      1.5           full            no       2
7                         0.1                      1.6           full            no       2
9                         0.2                      1.3           full            no       2
11                        0.2                      1.4           full            no       2
13                        0.2                      1.5           full            no       2
15                        0.2                      1.6           full            no       2
17                        0.3                      1.3           full            no       2
19                        0.3                      1.4           full            no       2
21                        0.3                      1.5           full            no       2
  Saved: output/FIGURES/ch05_damping_freq_full_A1.pdf
  Saved: output/FIGURES/ch05_damping_freq_full_A2.pdf
  Saved: output/FIGURES/ch05_damping_freq_full_A3.pdf
  Stub created: ch05_damping_freq.tex
  ch05_damping_freq_table: running analysis_scratch/damping_freq_table.py (REGENERATE_DELEGATED=True)
    ch05_damping_freq_table: regenerated 1 output(s)
  [✓] quality_flag gate: excluded 6 flagged run(s) (use quality_flag='all' to include)

--- Starting Filter Process (126 rows) ---
  [✓] WaveFrequencyInput [Hz].. range(1.3 to 1.6)............. kept 80 rows (removed 46)
--- Filter Final: 80 rows remaining (Total removed: 46) ---

Input dataframe shape: (80, 651)
Unique PanelCondition: ['full']
Unique WindCondition: ['full', 'no']

Number of unique grouping combinations: 24

Group sizes (rows per group):
count    24.000000
mean      3.333333
std       2.078182
min       2.000000
25%       2.000000
50%       3.000000
75%       3.250000
max      10.000000
dtype: float64

After aggregation — stats shape: (24, 21)
Columns in stats: ['WaveAmplitudeInput [Volt]', 'WaveFrequencyInput [Hz]', 'PanelCondition', 'WindCondition', 'Mooring', 'mean_out_in', 'std_out_in', 'n_runs', 'paths', 'mean_kL', 'mean_A_8804/250', 'mean_A_9373/170', 'mean_A_9373/340', 'mean_A_12400/250', 'in_probes_used', 'out_probes_used', 'in_position', 'out_position', 'ain_disagree_frac_mean', 'aout_disagree_frac_mean', 'file_dates']

WARNING: 10 groups have ≤ 2 runs:
    WaveAmplitudeInput [Volt]  WaveFrequencyInput [Hz] PanelCondition WindCondition  n_runs
3                         0.1                      1.4           full            no       2
5                         0.1                      1.5           full            no       2
7                         0.1                      1.6           full            no       2
9                         0.2                      1.3           full            no       2
11                        0.2                      1.4           full            no       2
13                        0.2                      1.5           full            no       2
15                        0.2                      1.6           full            no       2
17                        0.3                      1.3           full            no       2
19                        0.3                      1.4           full            no       2
21                        0.3                      1.5           full            no       2
  Saved: output/FIGURES/ch05_damping_scatter_full.pdf
  Stub created: ch05_damping_scatter.tex
  ch05_wind_effect_table: running analysis_scratch/wind_effect_table.py (REGENERATE_DELEGATED=True)
    ch05_wind_effect_table: regenerated 1 output(s)
  ch05_wind_effect_table_by_amp: running analysis_scratch/wind_effect_table_by_amp.py (REGENERATE_DELEGATED=True)
    ch05_wind_effect_table_by_amp: regenerated 1 output(s)
  ch05_wind_effect_per_condition: running analysis_scratch/wind_effect_per_condition.py (REGENERATE_DELEGATED=True)
    ch05_wind_effect_per_condition: regenerated 1 output(s)
  ch05_transmission_wind_tables: running analysis_scratch/transmission_wind_tables.py (REGENERATE_DELEGATED=True)
    ch05_transmission_wind_tables: regenerated 2 output(s)
  ch05_t_cross: running analysis_scratch/t_cross_figure.py (REGENERATE_DELEGATED=True)
    ch05_t_cross: regenerated 4 output(s)
  ch05_damping_ka_per_volt: running analysis_scratch/damping_ka_per_volt.py (REGENERATE_DELEGATED=True)
    ch05_damping_ka_per_volt: regenerated 8 output(s)
  ch05_damping_all_data_scatter: running analysis_scratch/all_data_damping_scatter.py (REGENERATE_DELEGATED=True)
    ch05_damping_all_data_scatter: regenerated 2 output(s)
Medium load gate — loading canon processed_dfs (2 folder(s), ~12 MB, ~45 s first time)…
   Loaded 46 processed DataFrames from PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange
   Loaded 132 processed DataFrames from PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange
  +132 DataFrames in 43.8 s (processed_dfs: 132 total)
Heavy load gate — loading remaining processed_dfs (22 folder(s), ~65 MB, ~2 min)…
   Loaded 7 processed DataFrames from PROCESSED-20251005-sixttry6roof-highMooring
   Loaded 9 processed DataFrames from PROCESSED-20251110-tett6roof-lowMooring
   Loaded 10 processed DataFrames from PROCESSED-20251110-tett6roof-lowMooring-2
   Loaded 53 processed DataFrames from PROCESSED-20251112-tett6roof
   Loaded 63 processed DataFrames from PROCESSED-20251113-tett6roof
   Loaded 96 processed DataFrames from PROCESSED-20251113-tett6roof-loosepaneltaped
   Loaded 108 processed DataFrames from PROCESSED-20251113-tett6roof-probeadjusted
   Loaded 118 processed DataFrames from PROCESSED-20260305-newProbePos-tett6roof
   Loaded 141 processed DataFrames from PROCESSED-20260306-newProbePos-tett6roof
   Loaded 195 processed DataFrames from PROCESSED-20260307-ProbPos4_31_FPV_2-tett6roof
   Loaded 231 processed DataFrames from PROCESSED-20260312-ProbPos4_31_FPV_2-tett6roof
   Loaded 268 processed DataFrames from PROCESSED-20260313-ProbePos4_31_FPV_2-tett6roof
   Loaded 311 processed DataFrames from PROCESSED-20260314-ProbePos4_31_FPV_2-tett6roof
   Loaded 315 processed DataFrames from PROCESSED-20260316-ProbePos4_31_FPV_2-tett6roof
   Loaded 317 processed DataFrames from PROCESSED-20260316-ProbePos4_31_FPV_2-tett6roof-under9Mooring
   Loaded 333 processed DataFrames from PROCESSED-20260319-ProbePos4_31_FPV_2-tett6roof-under9Mooring
   Loaded 342 processed DataFrames from PROCESSED-20260321-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-RENAMED
   Loaded 349 processed DataFrames from PROCESSED-20260323-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height136
   Loaded 409 processed DataFrames from PROCESSED-20260323-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100
   Loaded 445 processed DataFrames from PROCESSED-20260324-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100
   Loaded 473 processed DataFrames from PROCESSED-20260325-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100
   Loaded 477 processed DataFrames from PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100
  +477 DataFrames in 56.2 s (processed_dfs: 609 total)
  ch04_paddle_contamination: running analysis_scratch/paddle_contamination_study.py (REGENERATE_DELEGATED=True)
    ch04_paddle_contamination: regenerated 2 output(s)
  Saved: output/FIGURES/diag_13hz_consistency.pdf
  Stub created: diag_13hz_consistency.tex
main_save_figures.py — all figure sections complete.
Input dataframe shape: (94, 651)
Unique PanelCondition: ['full']
Unique WindCondition: ['full', 'no']

Number of unique grouping combinations: 31

Group sizes (rows per group):
count    31.000000
mean      3.032258
std       2.105293
min       1.000000
25%       2.000000
50%       2.000000
75%       3.500000
max      10.000000
dtype: float64

After aggregation — stats shape: (31, 21)
Columns in stats: ['WaveAmplitudeInput [Volt]', 'WaveFrequencyInput [Hz]', 'PanelCondition', 'WindCondition', 'Mooring', 'mean_out_in', 'std_out_in', 'n_runs', 'paths', 'mean_kL', 'mean_A_8804/250', 'mean_A_9373/170', 'mean_A_9373/340', 'mean_A_12400/250', 'in_probes_used', 'out_probes_used', 'in_position', 'out_position', 'ain_disagree_frac_mean', 'aout_disagree_frac_mean', 'file_dates']

WARNING: 16 groups have ≤ 2 runs:
    WaveAmplitudeInput [Volt]  WaveFrequencyInput [Hz] PanelCondition WindCondition  n_runs
4                         0.1                      1.4           full            no       2
6                         0.1                      1.5           full            no       2
8                         0.1                      1.6           full            no       2
9                         0.1                      1.7           full          full       1
10                        0.2                      1.2           full          full       2
12                        0.2                      1.3           full            no       2
14                        0.2                      1.4           full            no       2
16                        0.2                      1.5           full            no       2
18                        0.2                      1.6           full            no       2
19                        0.2                      1.7           full          full       1
20                        0.2                      1.7           full            no       1
22                        0.3                      1.3           full            no       2
24                        0.3                      1.4           full            no       2
26                        0.3                      1.5           full            no       2
29                        0.3                      1.7           full          full       1
30                        0.3                      1.7           full            no       1
    WaveFrequencyInput [Hz]  WaveAmplitudeInput [Volt] WindCondition  mean_out_in  std_out_in  n_runs
0                       1.2                        0.1          full     0.825395    0.009743       3
1                       1.3                        0.1          full     0.786101    0.046343      10
2                       1.3                        0.1            no     0.660448    0.028073       9
3                       1.4                        0.1          full     0.721131    0.086318       3
4                       1.4                        0.1            no     0.502334    0.034339       2
5                       1.5                        0.1          full     0.695186    0.074377       3
6                       1.5                        0.1            no     0.452680    0.016190       2
7                       1.6                        0.1          full     0.613237    0.028959       4
8                       1.6                        0.1            no     0.366845    0.004564       2
9                       1.7                        0.1          full     0.529436         NaN       1
10                      1.2                        0.2          full     0.853545    0.013388       2
11                      1.3                        0.2          full     0.857740    0.021778       3
12                      1.3                        0.2            no     0.810555    0.007927       2
13                      1.4                        0.2          full     0.759130    0.055270       3
14                      1.4                        0.2            no     0.686926    0.002257       2
15                      1.5                        0.2          full     0.732744    0.048795       3
16                      1.5                        0.2            no     0.594339    0.020122       2
17                      1.6                        0.2          full     0.684498    0.028117       3
18                      1.6                        0.2            no     0.485909    0.000005       2
19                      1.7                        0.2          full     0.665842         NaN       1
20                      1.7                        0.2            no     0.411961         NaN       1
21                      1.3                        0.3          full     0.832655    0.017334       4
22                      1.3                        0.3            no     0.829650    0.002272       2
23                      1.4                        0.3          full     0.766135    0.006945       4
24                      1.4                        0.3            no     0.671612    0.003026       2
25                      1.5                        0.3          full     0.729367    0.012101       4
26                      1.5                        0.3            no     0.611721    0.002613       2
27                      1.6                        0.3          full     0.707070    0.012405       6
28                      1.6                        0.3            no     0.560979    0.083715       5
29                      1.7                        0.3          full     0.662135         NaN       1
30                      1.7                        0.3   (draumkvedet)(draumk(draumk(drau(dra(dr(d(((draumkvedet) ole@eduroam-193-157-165-233 wave_project %