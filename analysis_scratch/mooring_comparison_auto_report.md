# Mooring comparison: loose230 vs loose300

**Date**: 2026-04-20
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
Generated: 2026-04-20 16:57
Runs analysed: 217 (fullpanel, quality_flag==ok, wave runs)

Overlap summary by frequency:
  0.80 Hz:  loose230 n=1  loose300 n=1
  0.90 Hz:  loose230 n=1  loose300 n=1
  1.00 Hz:  loose230 n=1  loose300 n=1
  1.10 Hz:  loose230 n=1  loose300 n=1
  1.20 Hz:  loose230 n=2  loose300 n=1
  1.30 Hz:  loose230 n=37  loose300 n=22
  1.40 Hz:  loose230 n=25  loose300 n=12
  1.50 Hz:  loose230 n=24  loose300 n=13
  1.60 Hz:  loose230 n=25  loose300 n=13
  1.70 Hz:  loose230 n=4  loose300 n=3

Per-condition OUT/IN comparison (freq, amp, wind):
    freq   amp   wind    loose230   loose300         Δ      Δ%  n230 n300
  ------ ----- ------  ---------- ----------  -------- -------  ---- ----
    0.80   0.2   full       1.058      1.569    +0.511   +48.3%     1    1
    0.90   0.2   full       0.972      0.971    -0.001    -0.1%     1    1
    1.00   0.2   full       0.943      0.953    +0.010    +1.0%     1    1
    1.10   0.2   full       0.950      0.937    -0.013    -1.4%     1    1
    1.20   0.2   full  0.835±0.001      0.862    +0.026    +3.2%     2    1
    1.30   0.1   full  0.831±0.065 0.983±0.253    +0.152   +18.3%     9    6
    1.30   0.2   full  0.911±0.138 0.806±0.013    -0.105   -11.5%     4    2
    1.30   0.3   full  0.829±0.049 0.834±0.022    +0.005    +0.6%     2    3
    1.40   0.1   full  0.750±0.054 0.710±0.031    -0.040    -5.4%     4    2
    1.40   0.2   full  0.749±0.045 0.716±0.017    -0.033    -4.5%     6    2
    1.40   0.3   full  0.768±0.006 0.759±0.020    -0.009    -1.1%     2    3
    1.50   0.1   full  0.689±0.076 0.747±0.189    +0.058    +8.5%     3    2
    1.50   0.2   full  0.686±0.048 0.651±0.019    -0.034    -5.0%     5    2
    1.50   0.3   full  0.719±0.027 0.716±0.003    -0.002    -0.3%     2    3
    1.60   0.1   full  0.588±0.053 0.544±0.007    -0.045    -7.6%     3    2
    1.60   0.2   full  0.629±0.036 0.611±0.051    -0.018    -2.9%     5    2
    1.60   0.3   full  0.685±0.020 0.654±0.022    -0.030    -4.4%     4    3
    1.70   0.1   full  0.483±0.019      0.468    -0.015    -3.2%     2    1
    1.70   0.3   full       1.017      0.970    -0.047    -4.6%     1    1
    1.30   0.1     no  0.687±0.044 0.702±0.027    +0.014    +2.1%    16    7
    1.30   0.2     no  0.800±0.022 0.795±0.001    -0.005    -0.7%     5    2
    1.30   0.3     no       0.821 0.825±0.011    +0.004    +0.4%     1    2
    1.40   0.1     no  0.545±0.039 0.546±0.032    +0.002    +0.3%     5    2
    1.40   0.2     no  0.719±0.023 0.702±0.001    -0.017    -2.3%     5    2
    1.40   0.3     no  0.733±0.022      0.696    -0.037    -5.0%     3    1
    1.50   0.1     no  0.432±0.039 0.458±0.027    +0.026    +5.9%     5    2
    1.50   0.2     no  0.624±0.020 0.588±0.008    -0.037    -5.9%     5    2
    1.50   0.3     no  0.608±0.014 0.620±0.005    +0.012    +2.0%     4    2
    1.60   0.1     no  0.339±0.018 0.385±0.000    +0.046   +13.5%     4    2
    1.60   0.2     no  0.485±0.009 0.478±0.008    -0.007    -1.5%     4    2
    1.60   0.3     no  0.537±0.007 0.506±0.007    -0.031    -5.8%     5    2
    1.70   0.2     no       0.359      0.399    +0.040   +11.1%     1    1

Overall delta statistics (loose300 − loose230):
  Mean Δ:    +0.0118  (+1.3%)
  Median Δ:  -0.0063  (-0.9%)
  Std of Δ:  0.1006
  Max |Δ|:   0.5112  (48.3%)

  Wind=no: mean Δ=+0.0007 (+1.1%),  max|Δ|=0.0458 (13.5%),  n=13
  Wind=full: mean Δ=+0.0195 (+1.5%),  max|Δ|=0.5112 (48.3%),  n=19

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
