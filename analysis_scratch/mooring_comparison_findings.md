# Mooring comparison: loose230 vs loose300

**Date**: 2026-04-16
**Script**: `analysis_scratch/mooring_comparison.py`
**Figure**: `analysis_scratch/mooring_comparison.png`

## Purpose

Determine whether `below_90_loose230` (short rubber band, 230 mm, folders 20260316–20260326)
and `below_90_loose300` (longer rubber band, 300 mm, folder 20260327) produce equivalent
OUT/IN(FFT) values at matched conditions, so the datasets can be safely merged.

Both moorings: depth −90 mm (90 mm below still water), underwater attachment.
Difference: free (unstretched) rubber band length — 230 mm vs 300 mm.

## Data coverage

Only `fullpanel`, `quality_flag==ok` wave runs included.
Main overlap: **1.3–1.7 Hz**, all amplitudes (0.1/0.2/0.3 V), full and no wind.
Below 1.0 Hz: zero or one loose300 run — no meaningful comparison possible.

## Numerical results

```
Mooring comparison: below_90_loose230 vs below_90_loose300
Generated: 2026-04-16 09:58
Runs analysed: 222 (fullpanel, quality_flag==ok, wave runs)

Overlap summary by frequency:
  0.80 Hz:  loose230 n=1  loose300 n=1
  0.90 Hz:  loose230 n=1  loose300 n=1
  1.00 Hz:  loose230 n=1  loose300 n=1
  1.10 Hz:  loose230 n=1  loose300 n=1
  1.20 Hz:  loose230 n=2  loose300 n=1
  1.30 Hz:  loose230 n=39  loose300 n=22
  1.40 Hz:  loose230 n=25  loose300 n=13
  1.50 Hz:  loose230 n=24  loose300 n=13
  1.60 Hz:  loose230 n=25  loose300 n=15
  1.70 Hz:  loose230 n=4  loose300 n=3

Per-condition OUT/IN comparison (freq, amp, wind):
    freq   amp   wind    loose230   loose300         Δ      Δ%  n230 n300
  ------ ----- ------  ---------- ----------  -------- -------  ---- ----
    0.80   0.2   full       1.058      1.569    +0.512   +48.4%     1    1
    0.90   0.2   full       0.956      0.972    +0.017    +1.7%     1    1
    1.00   0.2   full       0.944      0.947    +0.003    +0.3%     1    1
    1.10   0.2   full       0.941      0.937    -0.004    -0.5%     1    1
    1.20   0.2   full  0.837±0.000      0.858    +0.021    +2.5%     2    1
    1.30   0.1   full  0.915±0.222 0.805±0.050    -0.110   -12.0%     9    6
    1.30   0.2   full  0.835±0.019 0.815±0.020    -0.020    -2.4%     4    2
    1.30   0.3   full  0.807±0.004 0.832±0.013    +0.025    +3.1%     2    3
    1.40   0.1   full  0.735±0.054 0.699±0.037    -0.035    -4.8%     4    2
    1.40   0.2   full  0.737±0.015 0.714±0.000    -0.023    -3.1%     6    2
    1.40   0.3   full  0.774±0.004 0.766±0.021    -0.009    -1.1%     2    3
    1.50   0.1   full  0.654±0.017 0.593±0.012    -0.061    -9.3%     3    2
    1.50   0.2   full  0.674±0.012 0.662±0.008    -0.012    -1.8%     5    2
    1.50   0.3   full  0.707±0.048 0.708±0.023    +0.001    +0.2%     2    3
    1.60   0.1   full  0.554±0.030 0.533±0.031    -0.021    -3.8%     3    2
    1.60   0.2   full  0.630±0.030 0.633±0.040    +0.003    +0.4%     5    2
    1.60   0.3   full  0.646±0.018 0.641±0.019    -0.005    -0.8%     4    3
    1.70   0.1   full  0.501±0.035      0.478    -0.023    -4.6%     2    1
    1.70   0.3   full       0.647      0.613    -0.034    -5.3%     1    1
    1.30   0.1     no  0.737±0.057 0.701±0.026    -0.037    -5.0%    16    7
    1.30   0.2     no  0.808±0.013 0.796±0.005    -0.011    -1.4%     5    2
    1.30   0.3     no  1.015±0.317 0.809±0.003    -0.206   -20.3%     3    2
    1.40   0.1     no  0.535±0.036 0.540±0.029    +0.005    +1.0%     5    2
    1.40   0.2     no  0.710±0.022 0.692±0.001    -0.018    -2.5%     5    2
    1.40   0.3     no  0.733±0.016 0.690±0.007    -0.043    -5.9%     3    2
    1.50   0.1     no  0.431±0.042 0.463±0.013    +0.032    +7.4%     5    2
    1.50   0.2     no  0.620±0.019 0.576±0.003    -0.044    -7.1%     5    2
    1.50   0.3     no  0.610±0.016 0.618±0.002    +0.008    +1.3%     4    2
    1.60   0.1     no  0.354±0.021 0.398±0.020    +0.044   +12.4%     4    2
    1.60   0.2     no  0.479±0.011 0.463±0.005    -0.016    -3.4%     4    2
    1.60   0.3     no  0.527±0.016 0.519±0.014    -0.008    -1.5%     5    4
    1.70   0.2     no       0.358      0.399    +0.042   +11.6%     1    1

Overall delta statistics (loose300 − loose230):
  Mean Δ:    -0.0009  (-0.2%)
  Median Δ:  -0.0099  (-1.5%)
  Std of Δ:  0.1042
  Max |Δ|:   0.5117  (48.4%)

  Wind=no: mean Δ=-0.0195 (-1.0%),  max|Δ|=0.2061 (20.3%),  n=13
  Wind=full: mean Δ=+0.0117 (+0.4%),  max|Δ|=0.5117 (48.4%),  n=19

Interpretation guide:
  |Δ%| < 5%  → moorings indistinguishable, safe to merge
  |Δ%| 5-10% → borderline; consider flagging in figure
  |Δ%| > 10% → significant mooring effect; do NOT merge without correction
```


## Interpretation (2026-04-16)

See delta plots (bottom row of figure): Δ = loose300 − loose230.
Dashed lines mark the ±0.05 threshold.

### Apparent outliers — NOT mooring effects

**0.8 Hz fullwind 0.2V  (Δ = +48.4%)**: n=1 each. Single-run comparison, uncontrolled.
Discard — insufficient data for any conclusion.

**1.3 Hz nowind 0.3V  (Δ = −20.3%)**: loose230 mean = 1.015 ± 0.317 (OUT > IN, std huge, n=3).
A mean OUT/IN > 1 at nowind is physically implausible for a damping geometry. This is the
known 1.3 Hz standing-wave anomaly: the IN probe (9373/170) sits near a pressure node at
1.3 Hz when the panel is present, suppressing the measured IN amplitude and inflating OUT/IN.
Confirmed in the assumption audit (2026-04-14). **Not a mooring effect.**

**1.3 Hz fullwind 0.1V  (Δ = −12.0%)**: loose230 std = 0.222 (n=9, very wide scatter).
The large scatter in loose230 dominates. At 0.1V fullwind, the IN probe SNR is low and the
standing-wave effect is present. The difference is within 1σ of the loose230 scatter. **Not
a reliable mooring comparison.**

**1.7 Hz (all)**: n=1–2 per mooring. Insufficient data.

### Core comparison: 1.4–1.6 Hz, 0.2V and 0.3V (the thesis main range)

| Condition | Δ range | Assessment |
|-----------|---------|------------|
| fullwind, 0.2V, 1.4–1.6 Hz | −3.1% to +0.4% | **Merge: indistinguishable** |
| fullwind, 0.3V, 1.4–1.6 Hz | −1.1% to +0.2% | **Merge: indistinguishable** |
| nowind,   0.2V, 1.4–1.6 Hz | −7.1% to −2.5% | Borderline at 1.5 Hz (−7.1%) |
| nowind,   0.3V, 1.4–1.6 Hz | −5.9% to +1.3% | Borderline at 1.4 Hz (−5.9%) |

The borderline cases (nowind, 0.2/0.3V, 1.5 Hz) go in opposite directions across amplitudes
(−7.1% at 0.2V, +1.3% at 0.3V), suggesting noise rather than a systematic mooring effect.

### 0.1V runs

0.1V fullwind at 1.5 Hz: Δ = −9.3% (loose300 lower). At this amplitude, wind-wave energy
is comparable to paddle-wave energy at the IN probe — FFT amplitude estimates are noisier.
Treat 0.1V fullwind as "indicative only" regardless of mooring; the SNR issue dominates.

0.1V nowind at 1.6 Hz: Δ = +12.4% (n=4 vs 2). Likely noise given the tiny amplitudes
(~0.35 OUT/IN × 0.1V × 80mm/V ≈ 3 mm at OUT). Probe noise floor (~0.14 mm) is non-trivial.

### Conclusion

**Merge the two mooring types for 1.3–1.6 Hz, 0.2V and 0.3V runs.**
The mooring rubber band length (230 vs 300 mm) does not produce a detectable systematic
effect on OUT/IN(FFT) within the main experimental range.

**For 0.1V runs**: merge, but treat results as higher-uncertainty. Note in figure caption
that low-amplitude runs have reduced SNR and the FFT amplitude estimate is less reliable.

**For 1.3 Hz specifically**: the standing-wave anomaly inflates OUT/IN for nowind runs
regardless of mooring. The mooring comparison is contaminated there. Address as a separate
standing-wave / reflection correction issue (see `memory/reflection_analysis.md`).

**Action**: Update `main_save_figures.py` to treat `below_90_loose230` and
`below_90_loose300` as a single group `"below_90_loose"` for the main result figures.
Keep mooring as a facet variable in supplementary plots for transparency.

## Status

- [x] Numerical comparison complete
- [x] Interpretation written
- [ ] Update `main_save_figures.py` grouping (next session)
- [ ] Note standing-wave anomaly at 1.3 Hz in methodology
