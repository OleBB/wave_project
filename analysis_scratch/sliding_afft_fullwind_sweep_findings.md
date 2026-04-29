# Sliding AFFT @ 9373/170 — fullwind per240 sweep

Generated: 2026-04-29T07:50:09Z

## Summary table

| freq_hz | amp_V | pipeline_AFFT | matched_mid_AFFT | plateau_FFT | plateau_sliding_std | mean_in_pipeline_win | depression | nowind_plateau_FFT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.300 | 0.100 | 7.891 | 7.599 | 7.654 | 0.219 | 5.171 | 1.031 | 4.662 |
| 1.300 | 0.200 | 16.971 | 15.562 | 14.632 | 0.898 | 8.813 | 1.160 | 9.655 |
| 1.400 | 0.100 | 8.373 | 7.879 | 7.886 | 0.311 | 8.338 | 1.062 | 7.284 |
| 1.400 | 0.200 | 16.479 | 16.319 | 17.153 | 0.767 | 16.632 | 0.961 | 13.278 |
| 1.500 | 0.100 | 8.597 | 8.399 | 7.990 | 0.416 | 5.524 | 1.076 | 5.542 |
| 1.500 | 0.200 | 14.403 | 13.247 | 7.330 | 0.918 | 7.983 | 1.965 | 14.909 |
| 1.600 | 0.100 | 9.656 | 10.022 | 7.611 | 0.378 | 9.354 | 1.269 | 7.113 |
| 1.600 | 0.200 | 19.612 | 19.998 | 16.919 | 0.839 | 18.291 | 1.159 | 14.993 |

## Verdict

- Runs analysed: **8** fullwind per240 runs at 9373/170.
- `depression` = pipeline_AFFT / plateau_FFT. Range: **0.96 to 1.96**; median **1.12**.
- Mean absolute bias vs plateau-FFT: **2.02 mm** (22% relative).

**Headline finding:** The pipeline's short-window FFT is an inconsistent estimator of the paddle-frequency amplitude at 9373/170 under fullwind. Direction of bias flips between conditions — no fixed correction applies:

  - **1.3 Hz / 0.1 V**: pipeline 7.9 vs plateau-FFT 7.7 mm → matches (+3%)
  - **1.3 Hz / 0.2 V**: pipeline 17.0 vs plateau-FFT 14.6 mm → over-reads (+16%)
  - **1.4 Hz / 0.1 V**: pipeline 8.4 vs plateau-FFT 7.9 mm → over-reads (+6%)
  - **1.4 Hz / 0.2 V**: pipeline 16.5 vs plateau-FFT 17.2 mm → matches (-4%)
  - **1.5 Hz / 0.1 V**: pipeline 8.6 vs plateau-FFT 8.0 mm → over-reads (+8%)
  - **1.5 Hz / 0.2 V**: pipeline 14.4 vs plateau-FFT 7.3 mm → over-reads (+96%)
  - **1.6 Hz / 0.1 V**: pipeline 9.7 vs plateau-FFT 7.6 mm → over-reads (+27%)
  - **1.6 Hz / 0.2 V**: pipeline 19.6 vs plateau-FFT 16.9 mm → over-reads (+16%)

Most extreme: **1.5 Hz / 0.2 V** pipeline reads ~2× plateau-FFT; **1.3 Hz / 0.1 V** pipeline under-reads by ~32%.

## Diagnosing cause: matched_mid_AFFT column

The new `matched_mid_AFFT` column takes a same-length FFT slice deep in the plateau (after t_s0 + 20 s, same sample count as the pipeline window). This controls for bin-resolution: if `matched_mid_AFFT ≈ plateau_FFT`, the signal genuinely has lower amplitude there. If `matched_mid_AFFT ≈ pipeline_AFFT`, both windows see the same signal and any plateau_FFT / pipeline_AFFT gap is a bin-alignment artifact of the short window.

Interpretation per case:

  - **1.3 / 0.1 V**: pipeline=7.9, mid-slice=7.6, plateau=7.7 — no significant discrepancy — all three within 10%
  - **1.3 / 0.2 V**: pipeline=17.0, mid-slice=15.6, plateau=14.6 — mid in between — genuine time-varying amplitude during the run
  - **1.4 / 0.1 V**: pipeline=8.4, mid-slice=7.9, plateau=7.9 — no significant discrepancy — all three within 10%
  - **1.4 / 0.2 V**: pipeline=16.5, mid-slice=16.3, plateau=17.2 — no significant discrepancy — all three within 10%
  - **1.5 / 0.1 V**: pipeline=8.6, mid-slice=8.4, plateau=8.0 — no significant discrepancy — all three within 10%
  - **1.5 / 0.2 V**: pipeline=14.4, mid-slice=13.2, plateau=7.3 — mid-slice matches pipeline → plateau_FFT differs; both short windows see the same signal, long FFT reads a different value (long-window leakage or paddle-drift within run)
  - **1.6 / 0.1 V**: pipeline=9.7, mid-slice=10.0, plateau=7.6 — mid-slice matches pipeline → plateau_FFT differs; both short windows see the same signal, long FFT reads a different value (long-window leakage or paddle-drift within run)
  - **1.6 / 0.2 V**: pipeline=19.6, mid-slice=20.0, plateau=16.9 — mid-slice matches pipeline → plateau_FFT differs; both short windows see the same signal, long FFT reads a different value (long-window leakage or paddle-drift within run)

**Takeaway:** the `pipeline_AFFT` discrepancies have **mixed causes** — some from bin-resolution, some from genuine transient bursts (notably 1.5 Hz 0.2 V where the signal really is stronger during the pipeline window than later). The pipeline's short-window FFT should not be trusted as an absolute amplitude estimator; for quantitative results use either a longer window FFT or sub-bin peak interpolation.

## Per-run breakdown

- **1.3 Hz, 0.1 V**  pipeline=7.89, matched-mid=7.60, plateau-FFT=7.65, mean-in-window=5.17 mm, ratio=1.031  (fullwind: `fullpanel-fullwind-amp0100-freq1300-per240-depth580-mstop30-run1.csv`)
- **1.3 Hz, 0.2 V**  pipeline=16.97, matched-mid=15.56, plateau-FFT=14.63, mean-in-window=8.81 mm, ratio=1.160  (fullwind: `fullpanel-fullwind-amp0200-freq1300-per240-depth580-mstop30-run1.csv`)
- **1.4 Hz, 0.1 V**  pipeline=8.37, matched-mid=7.88, plateau-FFT=7.89, mean-in-window=8.34 mm, ratio=1.062  (fullwind: `fullpanel-fullwind-amp0100-freq1400-per240-depth580-mstop30-run2.csv`)
- **1.4 Hz, 0.2 V**  pipeline=16.48, matched-mid=16.32, plateau-FFT=17.15, mean-in-window=16.63 mm, ratio=0.961  (fullwind: `fullpanel-fullwind-amp0200-freq1400-per240-depth580-mstop30-run1.csv`)
- **1.5 Hz, 0.1 V**  pipeline=8.60, matched-mid=8.40, plateau-FFT=7.99, mean-in-window=5.52 mm, ratio=1.076  (fullwind: `fullpanel-fullwind-amp0100-freq1500-per240-depth580-mstop30-run1.csv`)
- **1.5 Hz, 0.2 V**  pipeline=14.40, matched-mid=13.25, plateau-FFT=7.33, mean-in-window=7.98 mm, ratio=1.965  (fullwind: `fullpanel-fullwind-amp0200-freq1500-per240-depth580-mstop30-run1.csv`)
- **1.6 Hz, 0.1 V**  pipeline=9.66, matched-mid=10.02, plateau-FFT=7.61, mean-in-window=9.35 mm, ratio=1.269  (fullwind: `fullpanel-fullwind-amp0100-freq1600-per240-depth580-mstop30-run1.csv`)
- **1.6 Hz, 0.2 V**  pipeline=19.61, matched-mid=20.00, plateau-FFT=16.92, mean-in-window=18.29 mm, ratio=1.159  (fullwind: `fullpanel-fullwind-amp0200-freq1600-per240-depth580-mstop30-run1.csv`)

## See also

- Figure: `analysis_scratch/sliding_afft_fullwind_sweep.png`
- CSV:    `analysis_scratch/sliding_afft_fullwind_sweep_summary.csv`
- Earlier 1.3 Hz result: `analysis_scratch/sliding_afft_per240.pdf`
- Memory: `memory/open_question_per240_afft.md`
