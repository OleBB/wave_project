# Sliding AFFT @ 9373/170 — fullwind per240 sweep

Generated: 2026-05-02T14:55:32Z

## Summary table

| freq_hz | amp_V | pipeline_AFFT | matched_mid_AFFT | plateau_FFT | plateau_sliding_std | mean_in_pipeline_win | depression | nowind_plateau_FFT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.300 | 0.100 | 7.789 | 7.599 | 7.654 | 0.219 | 4.804 | 1.018 | 4.661 |
| 1.300 | 0.200 | 15.305 | 15.713 | 14.632 | 0.898 | 9.342 | 1.046 | 9.655 |
| 1.400 | 0.100 | 8.426 | 7.879 | 7.886 | 0.311 | 8.092 | 1.069 | 7.284 |
| 1.400 | 0.200 | 15.911 | 16.531 | 17.153 | 0.767 | 15.650 | 0.928 | 13.279 |
| 1.500 | 0.100 | 5.319 | 5.783 | 7.990 | 0.416 | 5.070 | 0.666 | 5.543 |
| 1.500 | 0.200 | 15.487 | 13.247 | 7.330 | 0.918 | 7.854 | 2.113 | 14.909 |
| 1.600 | 0.100 | 9.246 | 9.800 | 7.611 | 0.378 | 8.902 | 1.215 | 7.113 |
| 1.600 | 0.200 | 19.416 | 20.158 | 16.919 | 0.839 | 17.759 | 1.148 | 15.000 |

## Verdict

- Runs analysed: **8** fullwind per240 runs at 9373/170.
- `depression` = pipeline_AFFT / plateau_FFT. Range: **0.67 to 2.11**; median **1.06**.
- Mean absolute bias vs plateau-FFT: **2.19 mm** (25% relative).

**Headline finding:** The pipeline's short-window FFT is an inconsistent estimator of the paddle-frequency amplitude at 9373/170 under fullwind. Direction of bias flips between conditions — no fixed correction applies:

  - **1.3 Hz / 0.1 V**: pipeline 7.8 vs plateau-FFT 7.7 mm → matches (+2%)
  - **1.3 Hz / 0.2 V**: pipeline 15.3 vs plateau-FFT 14.6 mm → matches (+5%)
  - **1.4 Hz / 0.1 V**: pipeline 8.4 vs plateau-FFT 7.9 mm → over-reads (+7%)
  - **1.4 Hz / 0.2 V**: pipeline 15.9 vs plateau-FFT 17.2 mm → under-reads (-7%)
  - **1.5 Hz / 0.1 V**: pipeline 5.3 vs plateau-FFT 8.0 mm → under-reads (-33%)
  - **1.5 Hz / 0.2 V**: pipeline 15.5 vs plateau-FFT 7.3 mm → over-reads (+111%)
  - **1.6 Hz / 0.1 V**: pipeline 9.2 vs plateau-FFT 7.6 mm → over-reads (+21%)
  - **1.6 Hz / 0.2 V**: pipeline 19.4 vs plateau-FFT 16.9 mm → over-reads (+15%)

Most extreme: **1.5 Hz / 0.2 V** pipeline reads ~2× plateau-FFT; **1.3 Hz / 0.1 V** pipeline under-reads by ~32%.

## Diagnosing cause: matched_mid_AFFT column

The new `matched_mid_AFFT` column takes a same-length FFT slice deep in the plateau (after t_s0 + 20 s, same sample count as the pipeline window). This controls for bin-resolution: if `matched_mid_AFFT ≈ plateau_FFT`, the signal genuinely has lower amplitude there. If `matched_mid_AFFT ≈ pipeline_AFFT`, both windows see the same signal and any plateau_FFT / pipeline_AFFT gap is a bin-alignment artifact of the short window.

Interpretation per case:

  - **1.3 / 0.1 V**: pipeline=7.8, mid-slice=7.6, plateau=7.7 — no significant discrepancy — all three within 10%
  - **1.3 / 0.2 V**: pipeline=15.3, mid-slice=15.7, plateau=14.6 — no significant discrepancy — all three within 10%
  - **1.4 / 0.1 V**: pipeline=8.4, mid-slice=7.9, plateau=7.9 — no significant discrepancy — all three within 10%
  - **1.4 / 0.2 V**: pipeline=15.9, mid-slice=16.5, plateau=17.2 — no significant discrepancy — all three within 10%
  - **1.5 / 0.1 V**: pipeline=5.3, mid-slice=5.8, plateau=8.0 — mid-slice matches pipeline → plateau_FFT differs; both short windows see the same signal, long FFT reads a different value (long-window leakage or paddle-drift within run)
  - **1.5 / 0.2 V**: pipeline=15.5, mid-slice=13.2, plateau=7.3 — mid in between — genuine time-varying amplitude during the run
  - **1.6 / 0.1 V**: pipeline=9.2, mid-slice=9.8, plateau=7.6 — mid-slice matches pipeline → plateau_FFT differs; both short windows see the same signal, long FFT reads a different value (long-window leakage or paddle-drift within run)
  - **1.6 / 0.2 V**: pipeline=19.4, mid-slice=20.2, plateau=16.9 — mid-slice matches pipeline → plateau_FFT differs; both short windows see the same signal, long FFT reads a different value (long-window leakage or paddle-drift within run)

**Takeaway:** the `pipeline_AFFT` discrepancies have **mixed causes** — some from bin-resolution, some from genuine transient bursts (notably 1.5 Hz 0.2 V where the signal really is stronger during the pipeline window than later). The pipeline's short-window FFT should not be trusted as an absolute amplitude estimator; for quantitative results use either a longer window FFT or sub-bin peak interpolation.

## Per-run breakdown

- **1.3 Hz, 0.1 V**  pipeline=7.79, matched-mid=7.60, plateau-FFT=7.65, mean-in-window=4.80 mm, ratio=1.018  (fullwind: `fullpanel-fullwind-amp0100-freq1300-per240-depth580-mstop30-run1.csv`)
- **1.3 Hz, 0.2 V**  pipeline=15.31, matched-mid=15.71, plateau-FFT=14.63, mean-in-window=9.34 mm, ratio=1.046  (fullwind: `fullpanel-fullwind-amp0200-freq1300-per240-depth580-mstop30-run1.csv`)
- **1.4 Hz, 0.1 V**  pipeline=8.43, matched-mid=7.88, plateau-FFT=7.89, mean-in-window=8.09 mm, ratio=1.069  (fullwind: `fullpanel-fullwind-amp0100-freq1400-per240-depth580-mstop30-run2.csv`)
- **1.4 Hz, 0.2 V**  pipeline=15.91, matched-mid=16.53, plateau-FFT=17.15, mean-in-window=15.65 mm, ratio=0.928  (fullwind: `fullpanel-fullwind-amp0200-freq1400-per240-depth580-mstop30-run1.csv`)
- **1.5 Hz, 0.1 V**  pipeline=5.32, matched-mid=5.78, plateau-FFT=7.99, mean-in-window=5.07 mm, ratio=0.666  (fullwind: `fullpanel-fullwind-amp0100-freq1500-per240-depth580-mstop30-run1.csv`)
- **1.5 Hz, 0.2 V**  pipeline=15.49, matched-mid=13.25, plateau-FFT=7.33, mean-in-window=7.85 mm, ratio=2.113  (fullwind: `fullpanel-fullwind-amp0200-freq1500-per240-depth580-mstop30-run1.csv`)
- **1.6 Hz, 0.1 V**  pipeline=9.25, matched-mid=9.80, plateau-FFT=7.61, mean-in-window=8.90 mm, ratio=1.215  (fullwind: `fullpanel-fullwind-amp0100-freq1600-per240-depth580-mstop30-run1.csv`)
- **1.6 Hz, 0.2 V**  pipeline=19.42, matched-mid=20.16, plateau-FFT=16.92, mean-in-window=17.76 mm, ratio=1.148  (fullwind: `fullpanel-fullwind-amp0200-freq1600-per240-depth580-mstop30-run1.csv`)

## See also

- Figure: `analysis_scratch/sliding_afft_fullwind_sweep.png`
- CSV:    `analysis_scratch/sliding_afft_fullwind_sweep_summary.csv`
- Earlier 1.3 Hz result: `analysis_scratch/sliding_afft_per240.pdf`
- Memory: `memory/open_question_per240_afft.md`
