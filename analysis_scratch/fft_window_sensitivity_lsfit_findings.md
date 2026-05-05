# FFT window-length sensitivity — LS fit at varying N_periods

Generated: 2026-05-05T10:23:12Z

**Dataset**: fullpanel per240 wave runs, quality_flag=ok, from the two canonical
March-2026 lowrange folders. n_runs = 51 total; complete-sweep runs = 51.

**Method**: for each probe in each run, extract N × samples_per_period samples
starting at the pipeline's `Computed Probe {pos} start`; compute amplitude by
least-squares sinusoid fit at f_paddle (bin-grid-independent). Canonical IN =
mean(9373/170, 9373/340); OUT = 12400/250. OUT/IN per run per N.

**Sweep**: N ∈ [5, 8, 10, 12, 15, 20] periods. Reference: N = 10p (the pipeline H&G default).

## Global drift summary

| N_periods | median drift % | max \|drift\| % | n_runs |
|---|---|---|---|
| 5p | +0.105 | 7.67 | 51 |
| 8p | -0.160 | 2.65 | 51 |
| **10p (ref)** | 0.000 | 0.00 | — |
| 12p | +0.084 | 2.33 | 51 |
| 15p | +0.261 | 4.89 | 51 |
| 20p | +0.238 | 7.19 | 51 |

## Per wind condition

### no (n_runs = 17)

| N_periods | median drift % | max \|drift\| % |
|---|---|---|
| 5p | -2.313 | 7.67 |
| 8p | -0.548 | 2.65 |
| **10p (ref)** | 0.000 | 0.00 |
| 12p | +0.485 | 2.33 |
| 15p | +0.987 | 4.89 |
| 20p | +1.596 | 7.19 |

### full (n_runs = 34)

| N_periods | median drift % | max \|drift\| % |
|---|---|---|
| 5p | +0.291 | 2.95 |
| 8p | -0.010 | 1.95 |
| **10p (ref)** | 0.000 | 0.00 |
| 12p | -0.037 | 1.74 |
| 15p | +0.108 | 3.55 |
| 20p | -0.255 | 4.00 |

## Per frequency (median drift from N=10p)

| freq (Hz) | 5p % | 8p % | 12p % | 15p % | 20p % | n_runs |
|---|---|---|---|---|---|---|
| 0.80 | +1.399 | -0.016 | +1.743 | +2.469 | +3.573 | 1 |
| 0.90 | -1.434 | -0.187 | +0.651 | +1.001 | +1.561 | 1 |
| 1.00 | +0.194 | +0.167 | -0.149 | -0.452 | -0.187 | 1 |
| 1.10 | +0.217 | -0.281 | -0.186 | +0.191 | +1.062 | 1 |
| 1.20 | +0.737 | -0.065 | -0.869 | -1.373 | -1.478 | 1 |
| 1.30 | -1.635 | -0.586 | +0.296 | +1.382 | +1.342 | 16 |
| 1.40 | -0.725 | -0.532 | +0.124 | +0.755 | +0.870 | 9 |
| 1.50 | -1.048 | -0.186 | +0.257 | +0.488 | +0.661 | 9 |
| 1.60 | +0.498 | +0.180 | -0.222 | -0.662 | -1.196 | 9 |
| 1.70 | +1.310 | +0.671 | -0.293 | -1.262 | -2.857 | 3 |

## Takeaway

- **Global max \|drift\|** across all runs, all N ≠ 10p: **7.67%**
- **Worst-case median drift** (worst N vs 10p): **1.561%**

- **Drift exceeds 3%** — the choice of N matters more than expected. Investigate.

- Complements the earlier `paddle_contamination_window_sensitivity.csv` study which
  reported a maximum drift of 1.17% across a larger N range (20p–100p) — consistent
  with this finding.

## See also

- Figure: `analysis_scratch/fft_window_sensitivity_lsfit.png`
- Data: `analysis_scratch/fft_window_sensitivity_lsfit.csv`
- Prior sweep: `analysis_scratch/paddle_contamination_window_sensitivity.csv`
- Method: `analysis_scratch/fft_method_comparison_findings.md`
- Plateau: `analysis_scratch/hg_window_stability_findings.md`
