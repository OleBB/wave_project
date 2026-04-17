# Sliding AFFT @ 9373/170 — fullwind per240 sweep

Generated: 2026-04-17T10:58:02Z

## Summary table

| freq_hz | amp_V | pipeline_AFFT | stable_mean | depression | nowind_stable |
| --- | --- | --- | --- | --- | --- |
| 1.300 | 0.100 | 5.208 | 5.240 | 0.994 | 4.573 |
| 1.300 | 0.200 | 15.798 | 10.102 | 1.564 | 9.574 |
| 1.400 | 0.100 | 8.337 | 8.391 | 0.994 | 7.741 |
| 1.400 | 0.200 | 16.147 | 17.183 | 0.940 | 14.427 |
| 1.500 | 0.100 | 8.608 | 5.029 | 1.712 | 5.526 |
| 1.500 | 0.200 | 14.524 | 8.573 | 1.694 | 9.438 |
| 1.600 | 0.100 | 9.117 | 9.803 | 0.930 | 9.057 |
| 1.600 | 0.200 | 19.911 | 19.792 | 1.006 | 14.964 |

## Verdict

- Runs analysed: 8 (fullwind per240 at 9373/170).
- Median pipeline_AFFT / stable_mean_AFFT = **1.000**.
- Fraction with ≥10% depression (ratio < 0.90): **0%**.
- Fraction with ≥20% depression (ratio < 0.80): **0%**.

**Finding:** No systematic depression — the 1.3 Hz observation does not generalise to other fullwind per240 conditions at this probe.

## Per-run breakdown

- **1.3 Hz, 0.1 V**  pipeline=5.21 mm, stable=5.24 mm, ratio=0.994  (fullwind: `fullpanel-fullwind-amp0100-freq1300-per240-depth580-mstop30-run1.csv`)
- **1.3 Hz, 0.2 V**  pipeline=15.80 mm, stable=10.10 mm, ratio=1.564  (fullwind: `fullpanel-fullwind-amp0200-freq1300-per240-depth580-mstop30-run1.csv`)
- **1.4 Hz, 0.1 V**  pipeline=8.34 mm, stable=8.39 mm, ratio=0.994  (fullwind: `fullpanel-fullwind-amp0100-freq1400-per240-depth580-mstop30-run2.csv`)
- **1.4 Hz, 0.2 V**  pipeline=16.15 mm, stable=17.18 mm, ratio=0.940  (fullwind: `fullpanel-fullwind-amp0200-freq1400-per240-depth580-mstop30-run1.csv`)
- **1.5 Hz, 0.1 V**  pipeline=8.61 mm, stable=5.03 mm, ratio=1.712  (fullwind: `fullpanel-fullwind-amp0100-freq1500-per240-depth580-mstop30-run1.csv`)
- **1.5 Hz, 0.2 V**  pipeline=14.52 mm, stable=8.57 mm, ratio=1.694  (fullwind: `fullpanel-fullwind-amp0200-freq1500-per240-depth580-mstop30-run1.csv`)
- **1.6 Hz, 0.1 V**  pipeline=9.12 mm, stable=9.80 mm, ratio=0.930  (fullwind: `fullpanel-fullwind-amp0100-freq1600-per240-depth580-mstop30-run1.csv`)
- **1.6 Hz, 0.2 V**  pipeline=19.91 mm, stable=19.79 mm, ratio=1.006  (fullwind: `fullpanel-fullwind-amp0200-freq1600-per240-depth580-mstop30-run1.csv`)

## See also

- Figure: `analysis_scratch/sliding_afft_fullwind_sweep.png`
- CSV:    `analysis_scratch/sliding_afft_fullwind_sweep_summary.csv`
- Earlier 1.3 Hz result: `analysis_scratch/sliding_afft_per240.pdf`
- Memory: `memory/open_question_per240_afft.md`
