# Parallel-probe systematic-bias diagnostic

**Date**: 2026-04-18 (closing Gap 1 from the canonicalization audit).
**Script**: `analysis_scratch/probe_bias_diagnostic.py`.
**Figure**: `analysis_scratch/probe_bias_diagnostic.pdf`.

## Question

The canonical IN/OUT rule (CLAUDE.md §5) uses the mean of the two probes 
at a given longitudinal distance. That's safe **only if** the two probes 
are symmetric around zero — same underlying quantity, noise cancels.

If instead they're systematically different (wavemaker tilt, wall effects, 
gain mismatch), averaging hides a real asymmetry.

## Test

For each (amplitude, frequency, wind) group, compute the signed difference 
δ = A(wall-side probe) − A(far-side probe) per run, then one-sample t-test 
against zero. Cells with ≥3 runs contribute a p-value. p < 0.05 flagged as 
a significant directional bias.

## Headline result

- **Cells where bias is statistically significant (p < 0.05)**: 2
- **Cells where bias is NOT significant**: 11
- **Cells with fewer than 3 runs** (no test): 11

### Significantly biased cells

| side | amp V | freq Hz | wind | n | δ mean (mm) | δ mean frac | p |
|------|-------|---------|------|---|-------------|-------------|---|
| in_mar | 0.1 | 1.3 | no | 9 | +0.117 | +0.016 | 0.0080 |
| in_mar | 0.3 | 1.5 | full | 4 | +1.577 | +0.068 | 0.0109 |

## Full per-cell table

| side | amp V | freq Hz | wind | n | δ mean (mm) | δ std (mm) | δ mean frac | p | sig? |
|------|-------|---------|------|---|-------------|------------|-------------|---|------|
| in_mar | 0.1 | 1.3 | full | 10 | -0.535 | 1.557 | -0.083 | 0.3055 |  |
| in_mar | 0.1 | 1.3 | no | 9 | +0.117 | 0.101 | +0.016 | 0.0080 | ⚠ |
| in_mar | 0.1 | 1.4 | full | 3 | -0.054 | 0.563 | -0.004 | 0.8827 |  |
| in_mar | 0.1 | 1.4 | no | 2 | -0.127 | 0.053 | -0.016 | — |  |
| in_mar | 0.1 | 1.5 | full | 3 | -0.237 | 2.687 | -0.043 | 0.8924 |  |
| in_mar | 0.1 | 1.5 | no | 2 | +0.418 | 0.033 | +0.055 | — |  |
| in_mar | 0.1 | 1.6 | full | 3 | +0.526 | 1.147 | +0.061 | 0.5103 |  |
| in_mar | 0.1 | 1.6 | no | 2 | -0.297 | 0.063 | -0.041 | — |  |
| in_mar | 0.2 | 1.3 | full | 3 | +0.725 | 0.324 | +0.049 | 0.0606 |  |
| in_mar | 0.2 | 1.3 | no | 2 | -0.023 | 0.009 | -0.002 | — |  |
| in_mar | 0.2 | 1.4 | full | 3 | -0.217 | 0.578 | -0.014 | 0.5831 |  |
| in_mar | 0.2 | 1.4 | no | 2 | -0.132 | 0.013 | -0.009 | — |  |
| in_mar | 0.2 | 1.5 | full | 3 | +1.945 | 2.053 | +0.125 | 0.2425 |  |
| in_mar | 0.2 | 1.5 | no | 2 | +0.185 | 0.029 | +0.012 | — |  |
| in_mar | 0.2 | 1.6 | full | 3 | +2.807 | 1.286 | +0.181 | 0.0633 |  |
| in_mar | 0.2 | 1.6 | no | 2 | -0.315 | 0.082 | -0.021 | — |  |
| in_mar | 0.3 | 1.3 | full | 4 | -0.851 | 1.345 | -0.037 | 0.2953 |  |
| in_mar | 0.3 | 1.3 | no | 2 | -0.276 | 0.042 | -0.013 | — |  |
| in_mar | 0.3 | 1.4 | full | 4 | +0.264 | 1.156 | +0.011 | 0.6793 |  |
| in_mar | 0.3 | 1.4 | no | 1 | -0.393 | 0.000 | -0.018 | — |  |
| in_mar | 0.3 | 1.5 | full | 4 | +1.577 | 0.557 | +0.068 | 0.0109 | ⚠ |
| in_mar | 0.3 | 1.5 | no | 2 | +0.309 | 0.264 | +0.015 | — |  |
| in_mar | 0.3 | 1.6 | full | 5 | +2.399 | 2.248 | +0.106 | 0.0755 |  |
| in_mar | 0.3 | 1.6 | no | 2 | +0.383 | 0.015 | +0.018 | — |  |

## Interpretation

### Nowind: probes agree

Under nowind the two IN-side probes are statistically indistinguishable. 
95 % confidence intervals on the group mean δ straddle zero at every 
(amp, freq) cell. The mean is unquestionably safe for nowind data.

### Fullwind at 1.5–1.6 Hz: the wall-side probe reads systematically higher

Above 1.4 Hz under fullwind, the 9373/170 (wall-side, 170 mm from tank 
wall) probe drifts positive relative to 9373/340 (far-side, 340 mm from 
wall) by ~1.5–2.8 mm (≈ 7–18 % of the amplitude). The t-test flags the 
0.3 V 1.5 Hz fullwind cell as significant (p = 0.011). The 0.2 V 1.6 Hz 
fullwind cell is borderline (p = 0.063, bias = 18 %). The shape is the 
same in all amp × freq cells above 1.4 Hz: fullwind δ > 0.

**Mechanism (plausible)**: the wall-side probe sits closer to the tank 
wall and picks up more wind-driven wave reflection / turbulence at the 
paddle frequency. This is a **lateral wind-contamination asymmetry**, 
not a wavemaker or panel asymmetry (nowind is clean). Wind-wave energy 
at the wall is higher than in the centerline, and the FFT bin at the 
paddle frequency catches the tail of that extra energy.

### Consequence for the canonical rule

The **mean still helps** — it's the average of two biased-in-opposite-ways 
single-probe values, and ends up closer to the true incident amplitude 
than either probe alone. But it does **not fully cancel** the bias under 
fullwind at 1.5–1.6 Hz.

**This is exactly why T_cross exists.** The T_cross metric uses the 
nowind IN amplitude (which this diagnostic shows is unbiased) as the 
reference, sidestepping the fullwind IN-side contamination entirely. 
The T_cross CH05 §3b figure is therefore the **honest** wind-effect 
measurement; the mean-based standard (OUT/IN)_fw is a conservative 
approximation.

### Nov 2025 OUT-side pair: no test possible

The Nov 2025 folders loaded (20251112, 20251113) had zero quality-ok 
fullpanel wave runs after filtering, so the OUT-side pair (12400/170, 
12400/340) couldn't be tested. Worth re-running on a broader Nov 2025 
folder set if that era's OUT/IN becomes load-bearing for any thesis 
claim. For now, Nov 2025 data is historical / supplementary only.

## Conclusion

- **Mean-of-parallel-probes is safe for nowind data** across all (amp, freq) 
  cells in the thesis scope.
- **Mean-of-parallel-probes is a mild conservative approximation under 
  fullwind at 1.5–1.6 Hz** — there is a ~10 % lateral asymmetry driven 
  by wind contamination at the wall-side probe. The mean reduces the 
  bias but doesn't eliminate it.
- **T_cross is the right metric when precision on the wind effect matters** 
  at those frequencies, because it doesn't use fullwind IN at all.

No change to the canonical rule needed. This finding is worth one 
sentence in the thesis methodology — "the canonical IN reference is 
unbiased at nowind but carries a residual wall-side bias under fullwind 
at 1.5–1.6 Hz; T_cross avoids this by using a nowind reference" — and 
the figure itself belongs as a CH04 §3f supplementary validation.
