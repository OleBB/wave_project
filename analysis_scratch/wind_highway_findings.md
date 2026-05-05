# Highway-effect tests 1 & 2 (2026-05-05)

Per-run analysis on canon, full panel, ok quality, non-period-aliased: n=41 fullwind runs.

## Test 1 — Δt vs pre-paddle σ at source (8804/250)

Within-cell variation in σ_pre under fullwind shows whether the
highway effect tracks the *amount* of wind-wave activity. If the
wind background is ~steady within a cell, this test reduces to an
across-cell correlation, which is harder to interpret because
frequency and amplitude vary too.

## Test 2 — Δt vs paddle amplitude A_paddle

Highway prediction: |Δt| ∝ 1/A. Bigger paddle envelope outpaces
the wind-wave head-start.

## Per-cell summary

             n  dt_mean_ms  dt_std_ms  sigma_pre_mean  sigma_pre_std  A_paddle_mean
f_hz amp_V                                                                         
1.3  0.1    10     -178.36      47.78            3.87           0.48           7.56
     0.2     3     -132.67      24.44            3.46           0.85          14.74
     0.3     4      -92.00      13.27            3.63           0.65          21.53
1.4  0.1     3     -130.67      40.86            3.68           0.80           8.05
     0.2     3     -154.67       4.62            3.46           0.14          15.20
     0.3     4     -159.00      10.52            3.54           0.29          22.46
1.5  0.1     2      -92.00      53.74            4.03           0.14           7.65
     0.3     4     -124.00      17.74            3.48           0.84          21.40
1.6  0.1     3     -162.00      42.33            3.38           0.14           7.13
     0.3     5     -195.47      18.20            3.58           0.32          21.00

## Files
- wind_highway_test1_dt_vs_sigma.png — Test 1 scatter + fits
- wind_highway_test2_dt_vs_amp.png — Test 2 scatter + 1/A fits
- wind_highway_collapse.png — Δt vs σ_pre/A combined
- wind_highway_per_run.csv — per-fw-run table
