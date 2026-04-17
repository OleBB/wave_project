# Mansard-Funke reflection analysis

**Date**: 2026-04-17
**Script**: `analysis_scratch/mansard_funke.py`
**Figure**: `analysis_scratch/mansard_funke.png`

## Method

Two-probe decomposition using 8804/250 (upstream, x₁=8.804 m) and 9373/170
(IN probe, x₂=9.373 m). Probe separation Δx = 569 mm.

    A_incident = [η̂(x₁)·exp(−ikx₂) − η̂(x₂)·exp(−ikx₁)] / [−2i·sin(kΔ)]
    B_reflected = [η̂(x₂)·exp(ikx₁) − η̂(x₁)·exp(ikx₂)] / [−2i·sin(kΔ)]
    R = |B| / |A|

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
   1.10                above_50    full   0.0851   0.0660     2
   1.30                above_50    full   0.0643   0.0314     2
   1.40                above_50    full   0.1422   0.1116     2
   1.50                above_50    full   0.1288   0.0407     2
   1.60                above_50    full   0.3312   0.0983     2
   1.70                above_50    full   0.8622   0.3162     2
   1.80                above_50    full   0.2442   0.0199     3
   1.90                above_50    full   0.1140   0.0101     2
   0.70                above_50      no   0.1163      nan     1
   0.80                above_50      no   0.0408      nan     1
   0.90                above_50      no   0.0435      nan     1
   1.00                above_50      no   0.0454   0.0178     2
   1.10                above_50      no   0.0470   0.0162     2
   1.30                above_50      no   0.0651   0.0224     2
   1.40                above_50      no   0.0663   0.0267     2
   1.50                above_50      no   0.0225   0.0065     2
   1.60                above_50      no   0.0453   0.0247     3
   1.70                above_50      no   0.1188      nan     1
   0.80       below_90_loose230    full   0.0459      nan     1
   0.90       below_90_loose230    full   0.0596      nan     1
   1.00       below_90_loose230    full   0.0297      nan     1
   1.10       below_90_loose230    full   0.0273      nan     1
   1.30       below_90_loose230    full   0.0878   0.0672     4
   1.40       below_90_loose230    full   0.0973   0.0163     6
   1.50       below_90_loose230    full   0.1395   0.0274     5
   1.60       below_90_loose230    full   0.2380   0.0840     5
   1.70       below_90_loose230    full   0.8455   0.5122     4
   0.80       below_90_loose230      no   0.0507      nan     1
   0.90       below_90_loose230      no   0.0640      nan     1
   1.00       below_90_loose230      no   0.0747      nan     1
   1.10       below_90_loose230      no   0.0697      nan     1
   1.30       below_90_loose230      no   0.0486   0.0228     5
   1.40       below_90_loose230      no   0.0369   0.0202     5
   1.50       below_90_loose230      no   0.0255   0.0130     5
   1.60       below_90_loose230      no   0.0597   0.0190     4
   1.70       below_90_loose230      no   0.1864      nan     1
   0.80       below_90_loose300    full   0.0139      nan     1
   0.90       below_90_loose300    full   0.0239      nan     1
   1.00       below_90_loose300    full   0.0255      nan     1
   1.10       below_90_loose300    full   0.0526      nan     1
   1.30       below_90_loose300    full   0.0428   0.0272     2
   1.40       below_90_loose300    full   0.0992   0.0164     2
   1.50       below_90_loose300    full   0.1259   0.0196     2
   1.60       below_90_loose300    full   0.3046   0.0668     2
   1.30       below_90_loose300      no   0.0481   0.0052     2
   1.40       below_90_loose300      no   0.0129   0.0099     2
   1.50       below_90_loose300      no   0.0450   0.0045     2
   1.60       below_90_loose300      no   0.1146   0.0152     2
   1.70       below_90_loose300      no   0.1075      nan     1
```

## Overall R by mooring (0.2V, nowind, well-conditioned, all freq)

| Mooring | n | mean R | median R | max R |
|---------|---|--------|----------|-------|
| above_50 | 17 | 0.0558 | 0.0474 | 0.1188 |
| below_90_loose230 | 24 | 0.0516 | 0.0493 | 0.1864 |
| below_90_loose300 | 9 | 0.0610 | 0.0481 | 0.1254 |

## Interpretation

### Mooring comparison
- **above_50** (panel ~5 cm above water, stiff mooring): R = [filled in after run]
  This is the mooring where the standing wave was visually observed at full wind.
- **below_90_loose230/300** (panel 9 cm below water, loose line): R = [filled in after run]
  Consistent with the smoothness test result (sw_correction_findings.md) which showed
  R < 0.05 as the data-supported upper bound.

### Why above_50 may have higher R
A stiff above-water mooring constrains the panel more rigidly. Under incoming waves,
a rigid panel acts more like a partial breakwater → more reflection. A loose below-water
mooring allows the panel to move with the wave → less reflection → more transmission.

### Connection to SW correction analysis
The sw_correction_findings.md smoothness test showed that R=0.20 was inconsistent
with the raw OUT/IN data. The MF measurement here provides a direct, model-free R(f).

### Amplitude effect (0.3V warning)
At 0.3V and high frequency (1.6+ Hz), wave steepness increases and the signal may
contain harmonics or be affected by the reconstruction procedure. MF assumes a single
sinusoidal component — harmonics would bias both |A| and |B| estimates. Treat 0.3V
R results above 1.5 Hz as indicative only.

## Status

- [x] MF decomposition implemented
- [x] Both probes use same analysis window (phase-coherent)
- [x] Amplitude tier flagging (0.1V/0.2V/0.3V)
- [x] Ill-conditioning detection per frequency
- [x] Results by mooring (above_50 vs below_90)
- [ ] Quantitative comparison of R(above_50) vs R(below_90) after run
- [ ] Connect to SW correction: does measured R explain observed smooth curve?
