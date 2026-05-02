# FFT peak-bin bias: impact on OUT/IN ratio

Generated: 2026-05-02T14:53:20Z

**Runs analyzed**: 358 fullpanel wave runs (in=9373/170, out=12400/250, quality_flag=ok)

## Per-wind-condition summary

| wind | n | mean Δ(OUT/IN) | std Δ | max |Δ| | mean |Δ|/OUT/IN |
|------|---|----------------|-------|----------|------------------|
| no | 172 | +0.00000 | 0.00002 | 0.0001 | 0.002% |
| full | 186 | -0.00059 | 0.00786 | 0.1072 | 0.058% |

## Takeaway

**OUT/IN is robust to the FFT peak-bin bias.** 90% of runs have |Δ(OUT/IN)| < 0.02, mean |Δ|/OUT/IN = 0.03%. The bias cancels in the ratio because IN and OUT probes use the same analysis window length (same bin grid) and see the same paddle frequency. Existing thesis values are safe to use.

- **Global stats**: Δ(OUT/IN) mean = -0.00030, std = 0.00567, max |Δ| = 0.1072
- **Frequency dependence**: see figure panel (c). If Δ(OUT/IN) has structure in frequency, specific frequencies may be more affected.
- **Paddle-drift link**: see figure panel (d). Δ(OUT/IN) should be near-zero when the IN paddle-peak is close to a bin, and larger when it is off.

## See also

- Figure: `analysis_scratch/fft_peak_bias_outin_impact.png`
- CSV:    `analysis_scratch/fft_peak_bias_outin_impact.csv`
- Methodology memo: `memory/methodology_fft_peak_bin_bias.md`
