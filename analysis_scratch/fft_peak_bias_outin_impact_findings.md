# FFT peak-bin bias: impact on OUT/IN ratio

Generated: 2026-04-17T13:04:53Z

**Runs analyzed**: 367 fullpanel wave runs (in=9373/170, out=12400/250, quality_flag=ok)

## Per-wind-condition summary

| wind | n | mean Δ(OUT/IN) | std Δ | max |Δ| | mean |Δ|/OUT/IN |
|------|---|----------------|-------|----------|------------------|
| no | 180 | -0.00035 | 0.00358 | 0.0302 | 0.155% |
| full | 187 | -0.00826 | 0.02817 | 0.1640 | 0.760% |

## Takeaway

**OUT/IN is robust to the FFT peak-bin bias.** 90% of runs have |Δ(OUT/IN)| < 0.02, mean |Δ|/OUT/IN = 0.46%. The bias cancels in the ratio because IN and OUT probes use the same analysis window length (same bin grid) and see the same paddle frequency. Existing thesis values are safe to use.

- **Global stats**: Δ(OUT/IN) mean = -0.00438, std = 0.02062, max |Δ| = 0.1640
- **Frequency dependence**: see figure panel (c). If Δ(OUT/IN) has structure in frequency, specific frequencies may be more affected.
- **Paddle-drift link**: see figure panel (d). Δ(OUT/IN) should be near-zero when the IN paddle-peak is close to a bin, and larger when it is off.

## See also

- Figure: `analysis_scratch/fft_peak_bias_outin_impact.png`
- CSV:    `analysis_scratch/fft_peak_bias_outin_impact.csv`
- Methodology memo: `memory/methodology_fft_peak_bin_bias.md`
