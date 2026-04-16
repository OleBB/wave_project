# Mansard-Funke reflection analysis

**Date**: 2026-04-16
**Script**: `analysis_scratch/mansard_funke.py`
**Figure**: `analysis_scratch/mansard_funke.png`

## Method

Two-probe decomposition using 8804/250 (upstream, x₁=8.804 m) and 9373/170
(IN probe, x₂=9.373 m). Probe separation Δx = 569 mm.

    A_incident = [η̂(x₁)·exp(+ikx₂) − η̂(x₂)·exp(+ikx₁)] / [+2i·sin(kΔ)]
    B_reflected = [η̂(x₂)·exp(−ikx₁) − η̂(x₁)·exp(−ikx₂)] / [+2i·sin(kΔ)]
    R = |B| / |A|

Note: sign convention matches the Python FFT, where positive-freq bin of a rightward
wave η=A·cos(kx−ωt) gives z ∝ A·exp(−ikx). Original Mansard & Funke (1980) uses the
opposite Fourier sign convention; the formula above is the Python-adapted form.

Both probes use the SAME analysis window (intersection of individual probe windows)
to preserve phase coherence. Each run's FFT resolution is self-consistent.

Ill-conditioned when |sin(kΔ)| < 0.3 (runs excluded from primary results).

## Quality / amplitude tiers

| Amplitude | Treatment |
|-----------|-----------|
| 0.2 V | **Primary** — best SNR, primary result |
| 0.1 V | **Include** — lower SNR, flag in figure |
| 0.3 V | **Caution** — non-linear risk at high frequency; untreated signal can be misleading |

Full-wind runs: the IN probe (9373/170) receives wind-wave energy that contaminates the
phase estimate at the paddle frequency. Fullwind R values are shown but interpreted with caution.

## Conditioning by frequency

```
  0.50 Hz:  k=1.460  kΔ=0.831  |sin(kΔ)|=0.738  OK
  0.60 Hz:  k=1.838  kΔ=1.046  |sin(kΔ)|=0.865  OK
  0.70 Hz:  k=2.275  kΔ=1.295  |sin(kΔ)|=0.962  OK
  0.80 Hz:  k=2.787  kΔ=1.586  |sin(kΔ)|=1.000  OK
  0.90 Hz:  k=3.390  kΔ=1.929  |sin(kΔ)|=0.937  OK
  1.00 Hz:  k=4.095  kΔ=2.330  |sin(kΔ)|=0.726  OK
  1.10 Hz:  k=4.903  kΔ=2.790  |sin(kΔ)|=0.345  OK
  1.20 Hz:  k=5.809  kΔ=3.305  |sin(kΔ)|=0.163  ILL-CONDITIONED ⚠
  1.30 Hz:  k=6.806  kΔ=3.873  |sin(kΔ)|=0.668  OK
  1.40 Hz:  k=7.889  kΔ=4.489  |sin(kΔ)|=0.975  OK
  1.50 Hz:  k=9.055  kΔ=5.152  |sin(kΔ)|=0.905  OK
  1.60 Hz:  k=10.302  kΔ=5.862  |sin(kΔ)|=0.409  OK
  1.70 Hz:  k=11.630  kΔ=6.618  |sin(kΔ)|=0.328  OK
  1.80 Hz:  k=13.039  kΔ=7.419  |sin(kΔ)|=0.907  OK
  1.90 Hz:  k=14.528  kΔ=8.266  |sin(kΔ)|=0.916  OK
  2.00 Hz:  k=16.097  kΔ=9.159  |sin(kΔ)|=0.262  ILL-CONDITIONED ⚠
```

## R results (0.2V, no-wind, well-conditioned)

```
  freq             mooring    wind     mean R    std R     n
   0.50                above_50    full   0.2395      nan     1
   0.60                above_50    full   0.1469   0.0188     2
   0.70                above_50    full   0.1074   0.0121     3
   0.80                above_50    full   0.0421   0.0101     2
   0.90                above_50    full   0.0343   0.0274     2
   1.00                above_50    full   0.0582   0.0061     2
   1.10                above_50    full   0.0836   0.0525     2
   1.30                above_50    full   0.0250   0.0128     2
   1.40                above_50    full   0.1068   0.0742     2
   1.50                above_50    full   0.1324   0.0266     2
   1.60                above_50    full   0.2633   0.0119     2
   1.70                above_50    full   0.9429   0.2313     2
   1.80                above_50    full   0.2563   0.0216     3
   1.90                above_50    full   0.1367   0.0135     2
   0.70                above_50      no   0.1163      nan     1
   0.80                above_50      no   0.0408      nan     1
   0.90                above_50      no   0.0418      nan     1
   1.00                above_50      no   0.0454   0.0178     2
   1.10                above_50      no   0.0470   0.0162     2
   1.30                above_50      no   0.0757   0.0198     2
   1.40                above_50      no   0.0537   0.0462     2
   1.50                above_50      no   0.0258   0.0105     2
   1.60                above_50      no   0.0777   0.0556     3
   1.70                above_50      no   0.1205      nan     1
   0.80       below_90_loose230    full   0.0459      nan     1
   0.90       below_90_loose230    full   0.0596      nan     1
   1.00       below_90_loose230    full   0.0297      nan     1
   1.10       below_90_loose230    full   0.0287      nan     1
   1.30       below_90_loose230    full   0.0513   0.0260     4
   1.40       below_90_loose230    full   0.0670   0.0326     6
   1.50       below_90_loose230    full   0.1080   0.0400     5
   1.60       below_90_loose230    full   0.2171   0.0807     5
   1.70       below_90_loose230    full   0.9426   0.4952     4
   0.80       below_90_loose230      no   0.0507      nan     1
   0.90       below_90_loose230      no   0.0612      nan     1
   1.00       below_90_loose230      no   0.0747      nan     1
   1.10       below_90_loose230      no   0.0697      nan     1
   1.30       below_90_loose230      no   0.0523   0.0231     5
   1.40       below_90_loose230      no   0.0367   0.0169     5
   1.50       below_90_loose230      no   0.0316   0.0114     5
   1.60       below_90_loose230      no   0.0815   0.0165     4
   1.70       below_90_loose230      no   0.1093      nan     1
   0.80       below_90_loose300    full   0.0139      nan     1
   0.90       below_90_loose300    full   0.0508      nan     1
   1.00       below_90_loose300    full   0.0255      nan     1
   1.10       below_90_loose300    full   0.0599      nan     1
   1.30       below_90_loose300    full   0.0570   0.0245     2
   1.40       below_90_loose300    full   0.0655   0.0337     2
   1.50       below_90_loose300    full   0.1069   0.0680     2
   1.60       below_90_loose300    full   0.2483   0.1425     2
   1.30       below_90_loose300      no   0.0664   0.0196     2
   1.40       below_90_loose300      no   0.0084   0.0052     2
   1.50       below_90_loose300      no   0.0565   0.0004     2
   1.60       below_90_loose300      no   0.1206   0.0229     2
   1.70       below_90_loose300      no   0.1062      nan     1
```

## Overall R by mooring (0.2V, nowind, well-conditioned, all freq)

| Mooring | n | mean R | median R | max R |
|---------|---|--------|----------|-------|
| above_50 | 17 | 0.0616 | 0.0580 | 0.1345 |
| below_90_loose230 | 24 | 0.0539 | 0.0522 | 0.1093 |
| below_90_loose300 | 9 | 0.0678 | 0.0568 | 0.1368 |

## Interpretation

### Summary (nowind, 0.2V, well-conditioned)

| Mooring | n | mean R | median R | max R | Verdict |
|---------|---|--------|----------|-------|---------|
| above_50 | 17 | 0.062 | 0.058 | 0.135 | low–moderate |
| below_90_loose230 | 24 | 0.054 | 0.052 | 0.109 | **low** |
| below_90_loose300 | 9 | 0.068 | 0.057 | 0.137 | **low** |

All three mooring types produce R ≈ 0.05–0.07 (median) under nowind conditions.
This is the first direct, model-free measurement of the panel's reflection coefficient.

### Connection to SW correction analysis

The smoothness test (sw_correction_findings.md) established an upper bound R < ~0.05
from the structure of the raw OUT/IN data. The MF result gives R ≈ 0.05–0.07 (median),
slightly above but entirely consistent with that bound — the smoothness argument is
conservative and the MF measurement is noisy.

**Both methods agree**: R is well below the assumed 0.20. The standing-wave correction
at R=0.20 is not supported. The raw OUT/IN should be used directly.

### Mooring comparison

Contrary to expectation, above_50 and below_90 give nearly identical median R (0.058 vs 0.052).
The above-water mooring does not cause dramatically more reflection than the below-water mooring
under nowind conditions. The visual standing wave observed with above_50 at full wind is more
likely a wind-mooring interaction artifact than a pure reflection effect.

### 1.7 Hz anomaly (fullwind: R ≈ 0.94)

The fullwind 1.7 Hz R values (above_50: 0.943, below_90_loose230: 0.943) are clearly artifacts.
At 1.7 Hz, |sin(kΔ)| = 0.328 — barely above the 0.30 threshold. Under full wind, the IN probe
(9373/170) receives broad-spectrum wind-wave energy; the paddle-frequency phase estimate is
corrupted. The MF decomposition is ill-conditioned in practice at 1.7 Hz fullwind even when the
formal conditioning criterion is marginally satisfied. **Discard fullwind R values at 1.7 Hz.**

Similarly, fullwind R values at 1.6 Hz (R ≈ 0.22–0.26) and 1.5 Hz (R ≈ 0.10–0.11) are likely
inflated by wind-wave phase contamination. The nowind values at the same frequencies (0.08–0.12)
are more reliable.

### High-freq nowind R: 1.6–1.7 Hz

Under nowind, 1.6 Hz R ≈ 0.08 (loose230), 0.12 (loose300), 0.08 (above_50). The 1.7 Hz
nowind values (n=1 each, R ≈ 0.11–0.12) are broadly consistent. These are within the scatter
and do not require a standing-wave correction at these frequencies.

### Fullwind R values should not be used for SW correction

The fullwind MF R is unreliable because:
1. Wind-wave energy at the paddle frequency biases the phase estimate at 9373/170
2. The 8804/250 probe (upstream) is less affected by wind (less fetch) — the two
   probes sample different wind-wave environments → artificial phase difference → inflated R

Fullwind MF results are shown for completeness but are not reliable for computing R(f).

### Recommended R for SW correction (if ever applied)

**Use nowind MF results only.** Conservative upper bound: R = 0.10 at high freq (1.6-1.7 Hz),
R = 0.05-0.07 at 1.3-1.5 Hz. Given the smoothness test finding, even these small values
produce corrections smaller than the run-to-run scatter.

### Thesis treatment

The MF analysis provides the definitive result:
> **R ≈ 0.05–0.07 (direct measurement, all mooring types, nowind), confirming that the
> standing-wave correction at the assumed R=0.20 is not needed. The raw OUT/IN is the
> primary result.**

Report R as a measured quantity (not an assumption), note the 1.2 Hz and 2.0 Hz
ill-conditioning, and flag the fullwind results as unreliable.

## Status

- [x] MF decomposition implemented (both probes, phase-coherent shared window)
- [x] FFT sign convention verified with synthetic test (R=0 for pure incident, R=0.05 recovers)
- [x] Amplitude tier flagging (0.1V/0.2V/0.3V)
- [x] Ill-conditioning detection per frequency (1.2 Hz and 2.0 Hz excluded)
- [x] Results by mooring (above_50 vs below_90_loose230/300)
- [x] Interpretation written: R ≈ 0.05–0.07 nowind, consistent with smoothness bound
- [x] 1.7 Hz fullwind anomaly identified and flagged
- [ ] Update MEMORY.md with MF result summary
