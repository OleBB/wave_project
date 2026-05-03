# Mooring comparison: loose230 vs loose300

**Date**: 2026-05-03
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
Generated: 2026-05-03 16:42
Runs analysed: 214 (fullpanel, quality_flag==ok, wave runs)

Overlap summary by frequency:
  0.80 Hz:  loose230 n=1  loose300 n=1
  0.90 Hz:  loose230 n=1  loose300 n=1
  1.00 Hz:  loose230 n=1  loose300 n=1
  1.10 Hz:  loose230 n=1  loose300 n=1
  1.20 Hz:  loose230 n=2  loose300 n=1
  1.30 Hz:  loose230 n=37  loose300 n=22
  1.40 Hz:  loose230 n=25  loose300 n=13
  1.50 Hz:  loose230 n=23  loose300 n=13
  1.60 Hz:  loose230 n=24  loose300 n=14
  1.70 Hz:  loose230 n=4  loose300 n=3

Per-condition OUT/IN comparison (freq, amp, wind):
    freq   amp   wind    loose230   loose300         Δ      Δ%  n230 n300
  ------ ----- ------  ---------- ----------  -------- -------  ---- ----
    0.80   0.2   full       0.959      0.974    +0.016    +1.6%     1    1
    0.90   0.2   full       0.932      0.944    +0.012    +1.2%     1    1
    1.00   0.2   full       0.933      0.952    +0.019    +2.0%     1    1
    1.10   0.2   full       0.929      0.941    +0.012    +1.3%     1    1
    1.20   0.2   full  0.873±0.001      0.850    -0.024    -2.7%     2    1
    1.30   0.1   full  0.785±0.060 0.767±0.041    -0.018    -2.3%     9    6
    1.30   0.2   full  0.865±0.025 0.826±0.014    -0.039    -4.5%     4    2
    1.30   0.3   full  0.821±0.018 0.830±0.012    +0.009    +1.1%     2    3
    1.40   0.1   full  0.724±0.057 0.697±0.020    -0.027    -3.7%     4    2
    1.40   0.2   full  0.757±0.045 0.727±0.007    -0.030    -4.0%     6    2
    1.40   0.3   full  0.761±0.007 0.758±0.024    -0.003    -0.4%     2    3
    1.50   0.1   full  0.849±0.210 0.618±0.000    -0.231   -27.2%     3    2
    1.50   0.2   full  0.689±0.080 0.659±0.038    -0.030    -4.4%     5    2
    1.50   0.3   full  0.712±0.011 0.721±0.015    +0.008    +1.2%     2    3
    1.60   0.1   full  0.601±0.073 0.562±0.003    -0.039    -6.6%     3    2
    1.60   0.2   full  0.644±0.041 0.636±0.069    -0.009    -1.3%     5    2
    1.60   0.3   full  0.676±0.022 0.655±0.018    -0.021    -3.0%     4    3
    1.70   0.1   full  0.532±0.050      0.536    +0.004    +0.8%     2    1
    1.70   0.3   full       0.680      0.610    -0.070   -10.4%     1    1
    1.30   0.1     no  0.654±0.029 0.659±0.026    +0.005    +0.7%    16    7
    1.30   0.2     no  0.802±0.018 0.805±0.008    +0.003    +0.4%     5    2
    1.30   0.3     no       0.803 0.831±0.003    +0.028    +3.5%     1    2
    1.40   0.1     no  0.538±0.040 0.509±0.032    -0.029    -5.4%     5    2
    1.40   0.2     no  0.715±0.033 0.692±0.002    -0.023    -3.2%     5    2
    1.40   0.3     no  0.736±0.022 0.676±0.003    -0.060    -8.1%     3    2
    1.50   0.1     no  0.436±0.033 0.444±0.016    +0.007    +1.6%     5    2
    1.50   0.2     no  0.620±0.025 0.592±0.019    -0.028    -4.5%     4    2
    1.50   0.3     no  0.608±0.015 0.605±0.004    -0.002    -0.4%     4    2
    1.60   0.1     no  0.342±0.023 0.373±0.006    +0.031    +9.1%     4    2
    1.60   0.2     no  0.500±0.007 0.487±0.002    -0.012    -2.5%     4    2
    1.60   0.3     no  0.547±0.017 0.521±0.011    -0.026    -4.8%     4    3
    1.70   0.2     no       0.373      0.415    +0.043   +11.4%     1    1

Overall delta statistics (loose300 − loose230):
  Mean Δ:    -0.0164  (-2.0%)
  Median Δ:  -0.0104  (-1.8%)
  Std of Δ:  0.0468
  Max |Δ|:   0.2306  (27.2%)

  Wind=no: mean Δ=-0.0049 (-0.2%),  max|Δ|=0.0596 (11.4%),  n=13
  Wind=full: mean Δ=-0.0242 (-3.2%),  max|Δ|=0.2306 (27.2%),  n=19

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
