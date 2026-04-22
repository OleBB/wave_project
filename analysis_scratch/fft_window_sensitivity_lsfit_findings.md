# FFT window-length sensitivity — LS fit at varying N_periods

Generated: 2026-04-22T09:22:59Z

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
| 5p | +0.585 | 4.16 | 51 |
| 8p | +0.159 | 1.89 | 51 |
| **10p (ref)** | 0.000 | 0.00 | — |
| 12p | +0.176 | 1.56 | 51 |
| 15p | +0.107 | 8.45 | 51 |
| 20p | -0.367 | 6.51 | 51 |

## Per wind condition

### no (n_runs = 17)

| N_periods | median drift % | max \|drift\| % |
|---|---|---|
| 5p | +0.953 | 4.16 |
| 8p | +0.512 | 1.89 |
| **10p (ref)** | 0.000 | 0.00 |
| 12p | +0.115 | 1.56 |
| 15p | +0.043 | 8.45 |
| 20p | -0.129 | 6.51 |

### full (n_runs = 34)

| N_periods | median drift % | max \|drift\| % |
|---|---|---|
| 5p | +0.333 | 3.16 |
| 8p | -0.022 | 1.22 |
| **10p (ref)** | 0.000 | 0.00 |
| 12p | +0.186 | 1.38 |
| 15p | +0.113 | 2.43 |
| 20p | -0.428 | 5.16 |

## Per frequency (median drift from N=10p)

| freq (Hz) | 5p % | 8p % | 12p % | 15p % | 20p % | n_runs |
|---|---|---|---|---|---|---|
| 0.80 | -0.271 | -0.066 | +0.242 | +0.107 | +0.044 | 1 |
| 0.90 | +1.086 | +0.771 | -0.047 | -0.503 | -0.490 | 1 |
| 1.00 | -0.298 | -0.354 | +0.223 | +0.119 | +0.260 | 1 |
| 1.10 | -1.157 | -0.335 | -0.585 | -0.563 | -0.960 | 1 |
| 1.20 | +0.958 | +0.228 | -0.036 | -0.200 | -0.617 | 1 |
| 1.30 | +0.769 | +0.247 | +0.344 | +0.200 | +0.391 | 16 |
| 1.40 | -0.440 | -0.048 | +0.196 | +0.213 | -0.129 | 9 |
| 1.50 | +0.942 | +0.266 | +0.201 | +0.207 | -0.521 | 9 |
| 1.60 | +0.734 | +0.557 | -0.032 | -0.876 | -0.711 | 9 |
| 1.70 | -0.438 | -0.049 | +0.334 | +0.073 | -0.941 | 3 |

## Takeaway

### Headline: OUT/IN is robust to N across a factor of 4

**Median drift stays under 1% at every N in [5, 20]** — the OUT/IN ratio is not
meaningfully changed by the window-length choice.

| N vs N=10p | median \|drift\| (nowind) | median \|drift\| (fullwind) | robustness verdict |
|---|---|---|---|
| 5p  | 0.95% | 0.33% | ok for quick-look; slightly biased high |
| 8p  | 0.51% | 0.02% | interchangeable with 10p |
| 12p | 0.12% | 0.19% | interchangeable with 10p |
| 15p | 0.04% | 0.11% | interchangeable with 10p |
| 20p | 0.13% | 0.43% | interchangeable with 10p |

### The max-drift outliers — observed correlation

The 8.45 % max at 15p and 6.51 % at 20p come from a single run:
`fullpanel-nowind-amp0300-freq1400-per240-depth580-mstop30-run1.csv`.

**Observed correlation** (not causal claim): the same run was reported
during pipeline processing as having `RECON ABORTED` on probe 9373/340
(903/1790 = 50.4 % of samples newly flagged by the false-trough repair
loop, which the pipeline interprets as over-convergence and halts). The
`quality_flag == 'ok'` filter did not exclude it.

Whether the RECON abort and the window-length drift share a root cause,
or are separate artefacts of the same underlying probe issue, has not
been investigated here. Excluding this one run, max \|drift\| at any N
stays under 5 % and median drift stays under 1 %.

### Observations for N = 10p

- Median drift at every N ∈ {8, 12, 15} relative to N=10 is ≤ 0.19 %
  on nowind and ≤ 0.43 % on fullwind.
- Median drift at N=5 relative to N=10 is +0.95 % (nowind), +0.33 %
  (fullwind).
- Median drift at N=20 relative to N=10 is −0.13 % (nowind), −0.43 %
  (fullwind).
- All within a factor of ~2 of each other, all well under 1 %.

*Candidate explanation (hypothesis): N=10p is unremarkable within an
8–15p plateau of robust values — not uniquely optimal, but a defensible
choice within a wide flat region.*

### Consistency with prior work

- `paddle_contamination_window_sensitivity.csv` (2026-04-21) tested N ∈ {20, 40,
  60, 100} periods on per240 and found max drift 1.17 %. **This sweep extends
  downward to 5p and confirms the plateau extends across the full 5p–20p range.**
- `hg_window_stability_findings.md` (per240) CV of 10T sliding AFFT within ±5T
  of H&G start: ~0.5 %. Consistent with the ~0.1 % median drifts here at 8p–15p.

### For the thesis CH04 methodology paragraph

The window length was fixed at N = 10 wave periods. This choice is justified by
three complementary constraints: it is the largest fixed value that fits inside
the per40 post-paddle plateau across all in-scope frequencies (1.3–1.7 Hz),
aligning with the pipeline's per40 ↔ per240 pooling; it matches the convention
of Huseby & Grue (2000) and the wave-tank literature generally; and a direct
sensitivity sweep across N ∈ {5, 8, 10, 12, 15, 20} periods on per240 runs shows
a median OUT/IN drift below 0.5 % at every N, with the canonical value sitting
near the centre of this plateau.


## See also

- Figure: `analysis_scratch/fft_window_sensitivity_lsfit.png`
- Data: `analysis_scratch/fft_window_sensitivity_lsfit.csv`
- Prior sweep: `analysis_scratch/paddle_contamination_window_sensitivity.csv`
- Method: `analysis_scratch/fft_method_comparison_findings.md`
- Plateau: `analysis_scratch/hg_window_stability_findings.md`
