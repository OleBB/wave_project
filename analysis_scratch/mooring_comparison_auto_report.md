# Mooring comparison: loose230 vs loose300

**Date**: 2026-04-29
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
Generated: 2026-04-29 09:47
Runs analysed: 215 (fullpanel, quality_flag==ok, wave runs)

Overlap summary by frequency:
  0.80 Hz:  loose230 n=1  loose300 n=1
  0.90 Hz:  loose230 n=1  loose300 n=1
  1.00 Hz:  loose230 n=1  loose300 n=1
  1.10 Hz:  loose230 n=1  loose300 n=1
  1.20 Hz:  loose230 n=2  loose300 n=1
  1.30 Hz:  loose230 n=37  loose300 n=22
  1.40 Hz:  loose230 n=25  loose300 n=12
  1.50 Hz:  loose230 n=23  loose300 n=13
  1.60 Hz:  loose230 n=25  loose300 n=14
  1.70 Hz:  loose230 n=4  loose300 n=3

Per-condition OUT/IN comparison (freq, amp, wind):
    freq   amp   wind    loose230   loose300         Δ      Δ%  n230 n300
  ------ ----- ------  ---------- ----------  -------- -------  ---- ----
    0.80   0.2   full       1.046      4.430    +3.384  +323.4%     1    1
    0.90   0.2   full       0.951      1.679    +0.728   +76.6%     1    1
    1.00   0.2   full       0.919      0.857    -0.062    -6.7%     1    1
    1.10   0.2   full       0.968      0.963    -0.005    -0.5%     1    1
    1.20   0.2   full  0.823±0.029      0.869    +0.046    +5.6%     2    1
    1.30   0.1   full  0.801±0.052 0.761±0.025    -0.041    -5.1%     9    6
    1.30   0.2   full  0.814±0.029 0.791±0.035    -0.023    -2.8%     4    2
    1.30   0.3   full  0.820±0.074 0.822±0.033    +0.002    +0.3%     2    3
    1.40   0.1   full  0.763±0.067 0.711±0.030    -0.052    -6.8%     4    2
    1.40   0.2   full  0.747±0.049 0.717±0.018    -0.030    -4.0%     6    2
    1.40   0.3   full  0.779±0.016 0.770±0.020    -0.008    -1.1%     2    3
    1.50   0.1   full  0.684±0.077 0.591±0.030    -0.093   -13.6%     3    2
    1.50   0.2   full  0.684±0.043 0.643±0.012    -0.042    -6.1%     5    2
    1.50   0.3   full  0.728±0.041 0.721±0.012    -0.006    -0.8%     2    3
    1.60   0.1   full  0.580±0.079 0.560±0.006    -0.020    -3.4%     3    2
    1.60   0.2   full  0.647±0.037 0.626±0.072    -0.021    -3.3%     5    2
    1.60   0.3   full  0.679±0.024 0.654±0.015    -0.025    -3.7%     4    3
    1.70   0.1   full  0.546±0.064      0.529    -0.017    -3.1%     2    1
    1.70   0.3   full       0.682      0.599    -0.084   -12.2%     1    1
    1.30   0.1     no  0.704±0.052 0.695±0.006    -0.009    -1.3%    16    7
    1.30   0.2     no  0.789±0.025 0.771±0.017    -0.018    -2.2%     5    2
    1.30   0.3     no       0.824 0.801±0.017    -0.023    -2.8%     1    2
    1.40   0.1     no  0.535±0.031 0.558±0.024    +0.023    +4.3%     5    2
    1.40   0.2     no  0.715±0.021 0.693±0.004    -0.022    -3.1%     5    2
    1.40   0.3     no  0.723±0.016      0.719    -0.003    -0.5%     3    1
    1.50   0.1     no  0.431±0.043 0.465±0.039    +0.034    +7.9%     5    2
    1.50   0.2     no  0.625±0.017 0.588±0.005    -0.037    -6.0%     4    2
    1.50   0.3     no  0.607±0.011 0.618±0.003    +0.010    +1.7%     4    2
    1.60   0.1     no  0.351±0.028 0.391±0.007    +0.040   +11.3%     4    2
    1.60   0.2     no  0.497±0.008 0.485±0.007    -0.012    -2.4%     4    2
    1.60   0.3     no  0.549±0.015 0.519±0.010    -0.030    -5.5%     5    3
    1.70   0.2     no       0.376      0.425    +0.049   +13.1%     1    1

Overall delta statistics (loose300 − loose230):
  Mean Δ:    +0.1136  (+10.8%)
  Median Δ:  -0.0174  (-2.6%)
  Std of Δ:  0.6119
  Max |Δ|:   3.3836  (323.4%)

  Wind=no: mean Δ=+0.0001 (+1.1%),  max|Δ|=0.0491 (13.1%),  n=13
  Wind=full: mean Δ=+0.1912 (+17.5%),  max|Δ|=3.3836 (323.4%),  n=19

Interpretation guide:
  |Δ%| < 5%  → moorings indistinguishable, safe to merge
  |Δ%| 5-10% → borderline; consider flagging in figure
  |Δ%| > 10% → significant mooring effect; do NOT merge without correction
```


## Interpretation

See delta plots (bottom row of figure): Δ = loose300 − loose230.
Dashed lines mark the ±0.05 threshold (5% of a typical OUT/IN ≈ 1).

## Next steps

- If |Δ%| < 5% across all conditions: merge moorings, annotate in methodology.
- If |Δ%| 5–10% at specific conditions: flag those points; consider mooring as a covariate.
- If |Δ%| > 10%: do NOT merge; treat mooring as a separate experimental variable.
