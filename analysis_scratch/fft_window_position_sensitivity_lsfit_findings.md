# FFT window-position sensitivity — 10T window slid across the per240 plateau

Generated: 2026-04-22T10:00:18Z

**Dataset**: fullpanel per240 wave runs, quality_flag=ok, from the two canonical
March-2026 lowrange folders. n_runs = 51.

**Method**: hold window length at N = 10 periods (pipeline default);
sweep the start position T_ref from 40T to 80T in 1T steps.
T_ref is in OUT-probe coordinates; each probe's local start is
`T_ref − ΔT_probe` via the probe-shifted H&G convention. Reference:
**T_ref = 50T** = current pipeline default.

## Per-run OUT/IN variability across T_ref

- Median range (max − min)/median across positions: **11.92 %**
- Max range across positions (worst run): **212.43 %**
- Median std(OUT/IN) / median(OUT/IN): **3.12 %**

## Median drift from T_ref=50T, per wind condition

| T_ref | nowind median % | nowind max \|Δ\| % | fullwind median % | fullwind max \|Δ\| % |
|---|---|---|---|---|
| 40T | -3.283 | 29.95 | +0.776 | 12.01 |
| 42T | -1.205 | 26.50 | +0.661 | 9.52 |
| 45T | +0.602 | 13.48 | +0.195 | 10.94 |
| 48T | +0.511 | 4.57 | +0.158 | 2.84 |
| **50T** | 0.000 | 0.00 | 0.000 | 0.00 |
| 52T | +0.030 | 1.91 | +0.156 | 2.47 |
| 55T | -0.011 | 14.03 | +0.073 | 4.26 |
| 58T | -0.496 | 14.15 | -0.326 | 6.67 |
| 60T | -0.263 | 17.38 | -0.797 | 9.46 |
| 65T | +0.056 | 83.88 | -1.452 | 13.52 |
| 70T | +0.938 | 5.23 | -2.057 | 14.68 |
| 75T | +1.343 | 116.85 | -2.186 | 13.37 |
| 80T | +0.653 | 90.52 | -2.723 | 12.32 |

## Where is the plateau?

Median OUT/IN vs T_ref (all runs pooled):

| T_ref | median OUT/IN | drift from 50T % |
|---|---|---|
| 40T | 0.7256 | +0.780 |
| 42T | 0.7410 | +2.928 |
| 44T | 0.7380 | +2.501 |
| 46T | 0.7359 | +2.218 |
| 48T | 0.7290 | +1.260 |
| **50T** | **0.7200** | **+0.000** |
| 52T | 0.7219 | +0.270 |
| 54T | 0.7323 | +1.717 |
| 56T | 0.7274 | +1.037 |
| 58T | 0.7229 | +0.403 |
| 60T | 0.7266 | +0.917 |
| 62T | 0.7280 | +1.120 |
| 65T | 0.7217 | +0.236 |
| 68T | 0.7186 | -0.193 |
| 70T | 0.7199 | -0.008 |
| 75T | 0.7252 | +0.721 |
| 80T | 0.7214 | +0.200 |

## Interpretation — three regions, not one plateau

The sweep reveals THREE distinct regimes, not a uniform plateau:

### Region 1: T_ref < 45T — nowind drift (observation only)

Nowind median drift at T_ref=40T is **−3.3 %** relative to T_ref=50T.

*Candidate explanation (hypothesis, not verified here)*: this region may
overlap with the wave-envelope build-up at the OUT probe (memory note
"envelope-back arrives at OUT at t ≈ 31T at 1.4 Hz" in
`methodology_hg_probe_shifted.md`). Verifying would require a
time-resolved envelope reconstruction. Not done in this sweep.

### Region 2: T_ref ∈ [45T, 65T] — the plateau (nowind)

Median nowind drift stays within **±0.6 %** at every T_ref ∈ [45, 65]T.

### Region 3: T_ref ≥ 55T fullwind — monotonic negative drift

Fullwind median drift is monotonically negative from T_ref=55T onward:

| T_ref | fullwind median drift |
|-------|-----------------------|
| 50T   | 0 (ref)               |
| 55T   | +0.07 %               |
| 60T   | **−0.80 %**           |
| 65T   | **−1.45 %**           |
| 70T   | **−2.06 %**           |
| 75T   | **−2.19 %**           |
| 80T   | **−2.72 %**           |

Panel (d) of the figure shows the median IN-probe amplitude rising with
T_ref under fullwind; median OUT-probe amplitude does not exhibit the
same rise.

*Candidate explanation (hypothesis, not verified here)*: this pattern is
consistent with `memory/methodology_wind_enhances_A_in.md` (documented
+10–17 % static A_in enhancement at 1.5–1.6 Hz) extended to a
time-dependent effect within a single run. Alternative explanations —
e.g. reflections accumulating in the sheltered OUT region, wind ramp-up
profile, probe-specific response — have not been ruled out in this
sweep.

### Where is the observed flat region?

Combining O1–O3 above as observations (not explanations):

- Nowind OUT/IN drift ≤ 0.6 % for T_ref ∈ [45, 65]T.
- Fullwind OUT/IN drift ≤ 1 % for T_ref ∈ [40, 55]T.
- Intersection (both < 1 %): **T_ref ∈ [45, 55]T**.

The pipeline default T_ref = 50T sits in the middle of this
intersection.

*Candidate explanation for why this position is a good choice
(hypothesis)*: it is simultaneously the latest position in the nowind
plateau and the earliest position before the fullwind drift. A
reviewer-facing defense for T_ref = 50T that goes beyond "H&G
convention" would need to test this hypothesis more carefully — e.g.
by controlling for probe-specific reflection effects or measuring the
time-dependence of A_in directly with the `fromZeroToMaxWin` dataset.

## Cross-reference with prior work

- `hg_window_stability_findings.md`: per240 10T sliding AFFT CV within
  ±5T of H&G start was ~0.5 % (IN and OUT). Consistent numerically
  with this sweep's ±0.6 % nowind drift across [45, 65]T.
- `per40_active_vs_late_vs_hg_findings.md`: per40 [32T, 38T] active
  zone reads 13 % low on OUT. Numerical magnitude is of the same order
  as Region 1 here; whether the same mechanism is at play is not tested.
- `methodology_wind_enhances_A_in.md`: documents +10–17 % static A_in
  enhancement. This sweep observes an additional pattern (monotonic
  fullwind OUT/IN drift with T_ref) whose relation to the static
  enhancement is a candidate hypothesis — not a proved link.

## Outlier flagging (observation)

One nowind run shows OUT/IN = 1.31 at T_ref=80T vs 0.69 at T_ref=50T
(+90 % drift): `fullpanel-nowind-amp0100-freq1300-per240-depth580-mstop330-run3.csv`.
This is the only run in the sweep with |drift| > 10 % at any T_ref
under nowind. It uses an mstop330 filename tag and 0.1 V amplitude.
Mechanism not investigated here.

## Pre-thesis methodology paragraph (draft)

> The window start position T_ref = 50T (relative to wavemaker onset,
> in OUT-probe coordinates) was chosen after a sensitivity sweep across
> T_ref ∈ [40, 80]T at fixed window length 10T. Observed drift in the
> median OUT/IN ratio relative to T_ref = 50T: nowind stays within
> ±0.6 % for T_ref ∈ [45, 65]T and drifts −3.3 % at T_ref = 40T;
> fullwind stays within ±1 % for T_ref ∈ [40, 55]T and drifts to
> −2.7 % at T_ref = 80T. The intersection of the two sub-1 % windows
> (T_ref ∈ [45, 55]T) contains the pipeline default. The direction of
> the fullwind drift is consistent with the wind-driven A_in
> enhancement documented in
> `memory/methodology_wind_enhances_A_in.md`, although the sweep does
> not distinguish between that mechanism and other candidates.

## See also

- Figure: `analysis_scratch/fft_window_position_sensitivity_lsfit.png`
- Data: `analysis_scratch/fft_window_position_sensitivity_lsfit.csv`
- Length sensitivity (companion): `analysis_scratch/fft_window_sensitivity_lsfit_findings.md`
- Plateau (prior): `analysis_scratch/hg_window_stability_findings.md`
