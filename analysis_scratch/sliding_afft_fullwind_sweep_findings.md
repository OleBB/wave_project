# Sliding AFFT @ 9373/170 — fullwind per240 sweep

Generated: 2026-04-17T13:58:27Z

## Summary table

| freq_hz | amp_V | pipeline_AFFT | matched_mid_AFFT | plateau_FFT | plateau_sliding_std | mean_in_pipeline_win | depression | nowind_plateau_FFT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.300 | 0.100 | 5.208 | 5.263 | 7.654 | 0.219 | 5.035 | 0.680 | 4.661 |
| 1.300 | 0.200 | 15.798 | 15.453 | 14.632 | 0.898 | 9.611 | 1.080 | 9.654 |
| 1.400 | 0.100 | 8.337 | 7.882 | 7.886 | 0.311 | 8.280 | 1.057 | 7.284 |
| 1.400 | 0.200 | 16.147 | 16.761 | 17.153 | 0.767 | 16.294 | 0.941 | 13.278 |
| 1.500 | 0.100 | 8.608 | 8.403 | 7.990 | 0.416 | 5.411 | 1.077 | 5.542 |
| 1.500 | 0.200 | 14.524 | 12.818 | 7.330 | 0.918 | 7.924 | 1.981 | 14.909 |
| 1.600 | 0.100 | 9.117 | 9.307 | 7.611 | 0.378 | 9.638 | 1.198 | 7.113 |
| 1.600 | 0.200 | 19.911 | 19.905 | 16.919 | 0.839 | 19.124 | 1.177 | 14.942 |

## Verdict

- Runs analysed: **8** fullwind per240 runs at 9373/170.
- `depression` = pipeline_AFFT / plateau_FFT. Range: **0.68 to 1.98**; median **1.08**.
- Mean absolute bias vs plateau-FFT: **2.17 mm** (24% relative).

**Headline finding:** The pipeline's short-window FFT is an inconsistent estimator of the paddle-frequency amplitude at 9373/170 under fullwind. Direction of bias flips between conditions — no fixed correction applies:

  - **1.3 Hz / 0.1 V**: pipeline 5.2 vs plateau-FFT 7.7 mm → under-reads (-32%)
  - **1.3 Hz / 0.2 V**: pipeline 15.8 vs plateau-FFT 14.6 mm → over-reads (+8%)
  - **1.4 Hz / 0.1 V**: pipeline 8.3 vs plateau-FFT 7.9 mm → over-reads (+6%)
  - **1.4 Hz / 0.2 V**: pipeline 16.1 vs plateau-FFT 17.2 mm → under-reads (-6%)
  - **1.5 Hz / 0.1 V**: pipeline 8.6 vs plateau-FFT 8.0 mm → over-reads (+8%)
  - **1.5 Hz / 0.2 V**: pipeline 14.5 vs plateau-FFT 7.3 mm → over-reads (+98%)
  - **1.6 Hz / 0.1 V**: pipeline 9.1 vs plateau-FFT 7.6 mm → over-reads (+20%)
  - **1.6 Hz / 0.2 V**: pipeline 19.9 vs plateau-FFT 16.9 mm → over-reads (+18%)

Most extreme: **1.5 Hz / 0.2 V** pipeline reads ~2× plateau-FFT; **1.3 Hz / 0.1 V** pipeline under-reads by ~32%.

## Diagnosing cause: matched_mid_AFFT column

The new `matched_mid_AFFT` column takes a same-length FFT slice deep in the plateau (after t_s0 + 20 s, same sample count as the pipeline window). This controls for bin-resolution: if `matched_mid_AFFT ≈ plateau_FFT`, the signal genuinely has lower amplitude there. If `matched_mid_AFFT ≈ pipeline_AFFT`, both windows see the same signal and any plateau_FFT / pipeline_AFFT gap is a bin-alignment artifact of the short window.

Interpretation per case:

  - **1.3 / 0.1 V**: pipeline=5.2, mid-slice=5.3, plateau=7.7 — mid-slice matches pipeline → plateau_FFT differs; both short windows see the same signal, long FFT reads a different value (long-window leakage or paddle-drift within run)
  - **1.3 / 0.2 V**: pipeline=15.8, mid-slice=15.5, plateau=14.6 — no significant discrepancy — all three within 10%
  - **1.4 / 0.1 V**: pipeline=8.3, mid-slice=7.9, plateau=7.9 — no significant discrepancy — all three within 10%
  - **1.4 / 0.2 V**: pipeline=16.1, mid-slice=16.8, plateau=17.2 — no significant discrepancy — all three within 10%
  - **1.5 / 0.1 V**: pipeline=8.6, mid-slice=8.4, plateau=8.0 — no significant discrepancy — all three within 10%
  - **1.5 / 0.2 V**: pipeline=14.5, mid-slice=12.8, plateau=7.3 — mid in between — genuine time-varying amplitude during the run
  - **1.6 / 0.1 V**: pipeline=9.1, mid-slice=9.3, plateau=7.6 — mid-slice matches pipeline → plateau_FFT differs; both short windows see the same signal, long FFT reads a different value (long-window leakage or paddle-drift within run)
  - **1.6 / 0.2 V**: pipeline=19.9, mid-slice=19.9, plateau=16.9 — mid-slice matches pipeline → plateau_FFT differs; both short windows see the same signal, long FFT reads a different value (long-window leakage or paddle-drift within run)

**Takeaway:** the `pipeline_AFFT` discrepancies have **mixed causes** — some from bin-resolution, some from genuine transient bursts (notably 1.5 Hz 0.2 V where the signal really is stronger during the pipeline window than later). The pipeline's short-window FFT should not be trusted as an absolute amplitude estimator; for quantitative results use either a longer window FFT or sub-bin peak interpolation.

## Per-run breakdown

- **1.3 Hz, 0.1 V**  pipeline=5.21, matched-mid=5.26, plateau-FFT=7.65, mean-in-window=5.03 mm, ratio=0.680  (fullwind: `fullpanel-fullwind-amp0100-freq1300-per240-depth580-mstop30-run1.csv`)
- **1.3 Hz, 0.2 V**  pipeline=15.80, matched-mid=15.45, plateau-FFT=14.63, mean-in-window=9.61 mm, ratio=1.080  (fullwind: `fullpanel-fullwind-amp0200-freq1300-per240-depth580-mstop30-run1.csv`)
- **1.4 Hz, 0.1 V**  pipeline=8.34, matched-mid=7.88, plateau-FFT=7.89, mean-in-window=8.28 mm, ratio=1.057  (fullwind: `fullpanel-fullwind-amp0100-freq1400-per240-depth580-mstop30-run2.csv`)
- **1.4 Hz, 0.2 V**  pipeline=16.15, matched-mid=16.76, plateau-FFT=17.15, mean-in-window=16.29 mm, ratio=0.941  (fullwind: `fullpanel-fullwind-amp0200-freq1400-per240-depth580-mstop30-run1.csv`)
- **1.5 Hz, 0.1 V**  pipeline=8.61, matched-mid=8.40, plateau-FFT=7.99, mean-in-window=5.41 mm, ratio=1.077  (fullwind: `fullpanel-fullwind-amp0100-freq1500-per240-depth580-mstop30-run1.csv`)
- **1.5 Hz, 0.2 V**  pipeline=14.52, matched-mid=12.82, plateau-FFT=7.33, mean-in-window=7.92 mm, ratio=1.981  (fullwind: `fullpanel-fullwind-amp0200-freq1500-per240-depth580-mstop30-run1.csv`)
- **1.6 Hz, 0.1 V**  pipeline=9.12, matched-mid=9.31, plateau-FFT=7.61, mean-in-window=9.64 mm, ratio=1.198  (fullwind: `fullpanel-fullwind-amp0100-freq1600-per240-depth580-mstop30-run1.csv`)
- **1.6 Hz, 0.2 V**  pipeline=19.91, matched-mid=19.91, plateau-FFT=16.92, mean-in-window=19.12 mm, ratio=1.177  (fullwind: `fullpanel-fullwind-amp0200-freq1600-per240-depth580-mstop30-run1.csv`)

## See also

- Figure: `analysis_scratch/sliding_afft_fullwind_sweep.png`
- CSV:    `analysis_scratch/sliding_afft_fullwind_sweep_summary.csv`
- Earlier 1.3 Hz result: `analysis_scratch/sliding_afft_per240.pdf`
- Memory: `memory/open_question_per240_afft.md`
