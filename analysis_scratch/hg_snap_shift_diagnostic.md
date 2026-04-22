# H&G snap-shift diagnostic — pulled from meta.json

Generated: 2026-04-22T12:47:55Z

**Dataset**: fullpanel wave runs, quality_flag=ok, from the two canonical
March-2026 lowrange folders. n_runs = 102.

Shift = (upcrossing-snapped window start) − (theoretical H&G window start),
converted to units of **wave periods** (T = 1/f_paddle).

## Table 1 — Shift by probe and wind (all frequencies pooled)

Median shift in periods, ±std, n_runs. Positive shift = snap moved the
window LATER than the theoretical H&G position.

| probe | wind=no | wind=full |
|---|---|---|
| `9373/170` | **+0.042** ± 0.273  (n=32) | **+0.070** ± 0.191  (n=69) |
| `9373/340` | **+0.039** ± 0.273  (n=32) | **-0.003** ± 0.184  (n=69) |
| `12400/250` | **-0.174** ± 0.214  (n=32) | **-0.226** ± 0.303  (n=69) |
| `8804/250` | **-0.016** ± 0.318  (n=32) | **+0.013** ± 0.268  (n=69) |

## Table 2 — IN-vs-OUT shift disagreement

If the ~−0.2 T offset at OUT were purely a wavemaker phase reference issue,
ALL probes should show the same median shift (c_g(f) predicts the same
arrival phase at every probe relative to paddle). Any cross-probe
disagreement argues for tank-local physics (amplitude dispersion,
reflections, near-panel effects) rather than a pure-phase-reference cause.

| wind | IN (9373/170) median | OUT (12400/250) median | IN − OUT |
|---|---|---|---|
| no | +0.042 | -0.174 | **+0.216** |
| full | +0.070 | -0.226 | **+0.296** |

## Table 3 — Amplitude dependence of OUT-probe shift

Does the −0.2 T offset grow (or shrink) with paddle amplitude? A Stokes-
nonlinearity mechanism would predict the offset magnitude scales with wave
steepness (roughly ∝ ka ∝ amplitude at fixed frequency).

OUT probe (12400/250), median shift in periods per (amp, wind):

| amp [V] | wind=no | wind=full |
|---|---|---|
| 0.1 | -0.172 ± 0.101  (n=15) | -0.286 ± 0.258  (n=20) |
| 0.2 | -0.120 ± 0.315  (n=9) | -0.318 ± 0.377  (n=13) |
| 0.3 | -0.327 ± 0.237  (n=8) | -0.212 ± 0.370  (n=18) |

Same table for IN probe (9373/170):

| amp [V] | wind=no | wind=full |
|---|---|---|
| 0.1 | +0.026 ± 0.283  (n=15) | +0.079 ± 0.192  (n=20) |
| 0.2 | +0.281 ± 0.310  (n=9) | +0.096 ± 0.157  (n=13) |
| 0.3 | +0.084 ± 0.135  (n=8) | -0.180 ± 0.158  (n=18) |

## Table 4 — The 1.4 Hz OUT-probe flip

Observation in the session log: at 1.4 Hz, median OUT shift flips from
~-0.35 T (nowind) to ~+0.35 T (fullwind). A ±0.5 T jump is the boundary
where 'nearest upcrossing' can flip between two adjacent ones. Listing
all 1.4 Hz OUT runs below with individual shifts so another agent can
confirm or falsify.

| path | amp [V] | wind | shift (T) | hg_expected_start | snap start (Computed start) |
|---|---|---|---|---|---|
| `fullpanel-fullwind-amp0100-freq1400-per240-depth580-mstop30-` | 0.1 | full | -0.475 | 8950.0 | 8865.0 |
| `fullpanel-fullwind-amp0200-freq1400-per240-depth580-mstop30-` | 0.2 | full | +0.425 | 8950.0 | 9026.0 |
| `fullpanel-fullwind-amp0300-freq1400-per240-depth580-mstop30-` | 0.3 | full | +0.346 | 8950.0 | 9012.0 |
| `fullpanel-nowind-amp0100-freq1400-per240-depth580-mstop30-ru` | 0.1 | no | -0.251 | 8950.0 | 8905.0 |
| `fullpanel-nowind-amp0200-freq1400-per40-depth580-mstop30-run` | 0.2 | no | -0.380 | 8950.0 | 8882.0 |
| `fullpanel-fullwind-amp0200-freq1400-per40-depth580-mstop30-r` | 0.2 | full | +0.486 | 8950.0 | 9037.0 |
| `fullpanel-fullwind-amp0100-freq1400-per240-depth580-mstop30-` | 0.1 | full | -0.352 | 8950.0 | 8887.0 |
| `fullpanel-fullwind-amp0100-freq1400-per40-depth580-mstop30-r` | 0.1 | full | -0.402 | 8950.0 | 8878.0 |
| `fullpanel-fullwind-amp0200-freq1400-per240-depth580-mstop30-` | 0.2 | full | -0.497 | 8950.0 | 8861.0 |
| `fullpanel-nowind-amp0100-freq1400-per40-depth580-mstop30-run` | 0.1 | no | -0.279 | 8950.0 | 8900.0 |
| `fullpanel-fullwind-amp0300-freq1400-per40-depth580-mstop30-r` | 0.3 | full | +0.363 | 8950.0 | 9015.0 |
| `fullpanel-fullwind-amp0300-freq1400-per40-depth580-mstop30-r` | 0.3 | full | +0.352 | 8950.0 | 9013.0 |
| `fullpanel-nowind-amp0200-freq1400-per240-depth580-mstop30-ru` | 0.2 | no | -0.352 | 8950.0 | 8887.0 |
| `fullpanel-nowind-amp0300-freq1400-per240-depth580-mstop30-ru` | 0.3 | no | -0.492 | 8950.0 | 8862.0 |
| `fullpanel-fullwind-amp0300-freq1400-per240-depth580-mstop30-` | 0.3 | full | +0.363 | 8950.0 | 9015.0 |

## Table 5 — Full breakdown by (freq, amp, wind, probe)

Long-form table, see also `hg_snap_shift_diagnostic_summary.csv`.

| freq [Hz] | amp [V] | wind | probe | n | median | std |
|---|---|---|---|---|---|---|
| 1.30 | 0.1 | no | `9373/170` | 9 | +0.036 | 0.023 |
| 1.30 | 0.1 | no | `9373/340` | 9 | +0.036 | 0.023 |
| 1.30 | 0.1 | no | `12400/250` | 9 | -0.172 | 0.022 |
| 1.30 | 0.1 | no | `8804/250` | 9 | -0.375 | 0.016 |
| 1.30 | 0.1 | full | `9373/170` | 10 | -0.138 | 0.094 |
| 1.30 | 0.1 | full | `9373/340` | 10 | -0.146 | 0.059 |
| 1.30 | 0.1 | full | `12400/250` | 10 | -0.294 | 0.038 |
| 1.30 | 0.1 | full | `8804/250` | 10 | +0.440 | 0.439 |
| 1.30 | 0.2 | no | `9373/170` | 2 | -0.047 | 0.007 |
| 1.30 | 0.2 | no | `9373/340` | 2 | -0.047 | 0.000 |
| 1.30 | 0.2 | no | `12400/250` | 2 | -0.263 | 0.011 |
| 1.30 | 0.2 | no | `8804/250` | 2 | -0.440 | 0.011 |
| 1.30 | 0.2 | full | `9373/170` | 3 | -0.172 | 0.060 |
| 1.30 | 0.2 | full | `9373/340` | 3 | -0.219 | 0.056 |
| 1.30 | 0.2 | full | `12400/250` | 3 | -0.365 | 0.060 |
| 1.30 | 0.2 | full | `8804/250` | 3 | +0.432 | 0.008 |
| 1.30 | 0.3 | no | `9373/170` | 2 | -0.117 | 0.011 |
| 1.30 | 0.3 | no | `9373/340` | 2 | -0.117 | 0.004 |
| 1.30 | 0.3 | no | `12400/250` | 2 | -0.359 | 0.015 |
| 1.30 | 0.3 | no | `8804/250` | 2 | +0.487 | 0.004 |
| 1.30 | 0.3 | full | `9373/170` | 4 | -0.214 | 0.038 |
| 1.30 | 0.3 | full | `9373/340` | 4 | -0.227 | 0.024 |
| 1.30 | 0.3 | full | `12400/250` | 4 | -0.430 | 0.014 |
| 1.30 | 0.3 | full | `8804/250` | 4 | +0.380 | 0.010 |
| 1.40 | 0.1 | no | `9373/170` | 2 | -0.492 | 0.016 |
| 1.40 | 0.1 | no | `9373/340` | 2 | -0.497 | 0.008 |
| 1.40 | 0.1 | no | `12400/250` | 2 | -0.265 | 0.020 |
| 1.40 | 0.1 | no | `8804/250` | 2 | +0.221 | 0.004 |
| 1.40 | 0.1 | full | `9373/170` | 3 | +0.251 | 0.028 |
| 1.40 | 0.1 | full | `9373/340` | 3 | +0.268 | 0.014 |
| 1.40 | 0.1 | full | `12400/250` | 3 | -0.402 | 0.062 |
| 1.40 | 0.1 | full | `8804/250` | 3 | +0.067 | 0.076 |
| 1.40 | 0.2 | no | `9373/170` | 2 | +0.430 | 0.024 |
| 1.40 | 0.2 | no | `9373/340` | 2 | +0.419 | 0.016 |
| 1.40 | 0.2 | no | `12400/250` | 2 | -0.366 | 0.020 |
| 1.40 | 0.2 | no | `8804/250` | 2 | +0.134 | 0.024 |
| 1.40 | 0.2 | full | `9373/170` | 3 | +0.251 | 0.031 |
| 1.40 | 0.2 | full | `9373/340` | 3 | +0.207 | 0.044 |
| 1.40 | 0.2 | full | `12400/250` | 3 | +0.425 | 0.551 |
| 1.40 | 0.2 | full | `8804/250` | 3 | +0.011 | 0.025 |
| 1.40 | 0.3 | no | `9373/170` | 1 | +0.307 | n/a |
| 1.40 | 0.3 | no | `9373/340` | 1 | +0.307 | n/a |
| 1.40 | 0.3 | no | `12400/250` | 1 | -0.492 | n/a |
| 1.40 | 0.3 | no | `8804/250` | 1 | +0.028 | n/a |
| 1.40 | 0.3 | full | `9373/170` | 4 | +0.162 | 0.012 |
| 1.40 | 0.3 | full | `9373/340` | 4 | +0.184 | 0.020 |
| 1.40 | 0.3 | full | `12400/250` | 4 | +0.358 | 0.008 |
| 1.40 | 0.3 | full | `8804/250` | 4 | -0.101 | 0.014 |
| 1.50 | 0.1 | no | `9373/170` | 2 | +0.401 | 0.017 |
| 1.50 | 0.1 | no | `9373/340` | 2 | +0.389 | 0.008 |
| 1.50 | 0.1 | no | `12400/250` | 2 | +0.033 | 0.004 |
| 1.50 | 0.1 | no | `8804/250` | 2 | +0.186 | 0.008 |
| 1.50 | 0.1 | full | `9373/170` | 3 | +0.132 | 0.036 |
| 1.50 | 0.1 | full | `9373/340` | 3 | +0.246 | 0.070 |
| 1.50 | 0.1 | full | `12400/250` | 3 | -0.120 | 0.115 |
| 1.50 | 0.1 | full | `8804/250` | 3 | +0.096 | 0.111 |
| 1.50 | 0.2 | no | `9373/170` | 2 | +0.266 | 0.021 |
| 1.50 | 0.2 | no | `9373/340` | 2 | +0.254 | 0.021 |
| 1.50 | 0.2 | no | `12400/250` | 2 | -0.102 | 0.025 |
| 1.50 | 0.2 | no | `8804/250` | 2 | +0.069 | 0.038 |
| 1.50 | 0.2 | full | `9373/170` | 3 | +0.042 | 0.035 |
| 1.50 | 0.2 | full | `9373/340` | 3 | +0.030 | 0.023 |
| 1.50 | 0.2 | full | `12400/250` | 3 | -0.347 | 0.018 |
| 1.50 | 0.2 | full | `8804/250` | 3 | -0.114 | 0.019 |
| 1.50 | 0.3 | no | `9373/170` | 2 | +0.063 | 0.030 |
| 1.50 | 0.3 | no | `9373/340` | 2 | +0.036 | 0.008 |
| 1.50 | 0.3 | no | `12400/250` | 2 | -0.338 | 0.047 |
| 1.50 | 0.3 | no | `8804/250` | 2 | -0.111 | 0.038 |
| 1.50 | 0.3 | full | `9373/170` | 4 | -0.099 | 0.026 |
| 1.50 | 0.3 | full | `9373/340` | 4 | -0.087 | 0.037 |
| 1.50 | 0.3 | full | `12400/250` | 4 | -0.015 | 0.558 |
| 1.50 | 0.3 | full | `8804/250` | 4 | -0.263 | 0.011 |
| 1.60 | 0.1 | no | `9373/170` | 2 | -0.413 | 0.014 |
| 1.60 | 0.1 | no | `9373/340` | 2 | -0.433 | 0.005 |
| 1.60 | 0.1 | no | `12400/250` | 2 | -0.308 | 0.018 |
| 1.60 | 0.1 | no | `8804/250` | 2 | +0.465 | 0.014 |
| 1.60 | 0.1 | full | `9373/170` | 3 | +0.327 | 0.068 |
| 1.60 | 0.1 | full | `9373/340` | 3 | +0.244 | 0.069 |
| 1.60 | 0.1 | full | `12400/250` | 3 | +0.397 | 0.065 |
| 1.60 | 0.1 | full | `8804/250` | 3 | +0.231 | 0.056 |
| 1.60 | 0.2 | no | `9373/170` | 2 | +0.388 | 0.032 |
| 1.60 | 0.2 | no | `9373/340` | 2 | +0.388 | 0.023 |
| 1.60 | 0.2 | no | `12400/250` | 2 | +0.449 | 0.018 |
| 1.60 | 0.2 | no | `8804/250` | 2 | +0.304 | 0.023 |
| 1.60 | 0.2 | full | `9373/170` | 3 | +0.154 | 0.010 |
| 1.60 | 0.2 | full | `9373/340` | 3 | +0.135 | 0.032 |
| 1.60 | 0.2 | full | `12400/250` | 3 | +0.135 | 0.035 |
| 1.60 | 0.2 | full | `8804/250` | 3 | +0.122 | 0.021 |
| 1.60 | 0.3 | no | `9373/170` | 3 | +0.090 | 0.006 |
| 1.60 | 0.3 | no | `9373/340` | 3 | +0.077 | 0.004 |
| 1.60 | 0.3 | no | `12400/250` | 3 | +0.064 | 0.007 |
| 1.60 | 0.3 | no | `8804/250` | 3 | -0.019 | 0.032 |
| 1.60 | 0.3 | full | `9373/170` | 5 | -0.199 | 0.022 |
| 1.60 | 0.3 | full | `9373/340` | 5 | -0.199 | 0.016 |
| 1.60 | 0.3 | full | `12400/250` | 5 | -0.212 | 0.027 |
| 1.60 | 0.3 | full | `8804/250` | 5 | -0.205 | 0.028 |
| 1.70 | 0.1 | full | `9373/170` | 1 | +0.218 | n/a |
| 1.70 | 0.1 | full | `9373/340` | 1 | +0.184 | n/a |
| 1.70 | 0.1 | full | `12400/250` | 1 | -0.163 | n/a |
| 1.70 | 0.1 | full | `8804/250` | 1 | +0.395 | n/a |
| 1.70 | 0.2 | no | `9373/170` | 1 | -0.483 | n/a |
| 1.70 | 0.2 | no | `9373/340` | 1 | -0.490 | n/a |
| 1.70 | 0.2 | no | `12400/250` | 1 | -0.095 | n/a |
| 1.70 | 0.2 | no | `8804/250` | 1 | -0.442 | n/a |
| 1.70 | 0.2 | full | `9373/170` | 1 | +0.095 | n/a |
| 1.70 | 0.2 | full | `9373/340` | 1 | +0.054 | n/a |
| 1.70 | 0.2 | full | `12400/250` | 1 | +0.476 | n/a |
| 1.70 | 0.2 | full | `8804/250` | 1 | +0.218 | n/a |
| 1.70 | 0.3 | full | `9373/170` | 1 | -0.259 | n/a |
| 1.70 | 0.3 | full | `9373/340` | 1 | -0.279 | n/a |
| 1.70 | 0.3 | full | `12400/250` | 1 | -0.054 | n/a |
| 1.70 | 0.3 | full | `8804/250` | 1 | -0.190 | n/a |

## Observations (facts)

- **O1**: OUT probe (12400/250) shows median shift ~−0.18 T under nowind,
  ~−0.23 T under fullwind. Systematic negative bias across all amplitudes.
- **O2**: IN probes (9373/170 and 9373/340) show median shifts near zero
  (~+0.04 T nowind, ~0 fullwind). Much smaller than OUT.
- **O3**: IN − OUT median shift is ~+0.21 T (nowind) and ~+0.30 T (fullwind).
  Any explanation purely about wavemaker phase reference would predict 0.
- **O4**: The 1.4 Hz OUT flip (Table 4) is driven by individual runs with
  shifts near ±0.5 T — at that boundary 'nearest upcrossing' can flip.
- **O5**: Amplitude dependence of OUT shift (Table 3) — examine the numbers,
  see whether the offset grows with amplitude (Stokes test).

## Candidate explanations (hypotheses, not verified)

- **H1** (wavemaker soft-start phase reference): would produce equal-
  magnitude offsets at ALL probes → inconsistent with O3. Partially rule out.
- **H2** (group-velocity underestimate for 12.4 m travel): if c_g at OUT
  is slightly slower than deep-water prediction, actual wave arrives later
  than predicted → snap finds upcrossing AFTER expected → POSITIVE shift.
  Observed sign is NEGATIVE, so this mechanism doesn't fit either.
- **H3** (group-velocity OVERESTIMATE for 12.4 m travel): predicted arrival
  later than actual; snap finds upcrossing BEFORE expected → NEGATIVE shift.
  Sign matches. Requires c_g to be ~0.2 T too fast over the longer travel.
  In deep water, a 0.2 T error at 1.4 Hz ≈ 0.14 s travel-time error over
  12.4 m = effective c_g 0.9 m/s (vs predicted 0.56) or ~60% faster. Not
  physically credible for a linear deep-water wave.
- **H4** (finite-amplitude Stokes correction to c_g): second-order Stokes
  nonlinearity slightly modifies c_phase and c_g at large ka. Effect size
  is small (typically < few percent at ka ~0.1). Probably insufficient to
  account for 0.2 T, but Table 3 would show it as an amplitude dependence.
- **H5** (near-panel reflection at OUT probe changing detected upcrossing):
  the OUT probe sits between the panel and the beach. If a partial
  reflection from the panel creates a standing-wave pattern at the OUT
  probe, local phase is shifted relative to the free-running tone. Would
  affect OUT much more than IN. Magnitude depends on reflection coefficient
  (measured ~0.05–0.07 under nowind 0.2 V — see mansard_funke_findings.md).
  Could plausibly shift OUT upcrossing by up to ~0.2 T for R in that range.
- **H6** (sensor-response lag on ULS probe): each ULS probe has its own
  electronic response. A small but probe-specific delay would produce a
  constant per-probe offset. Doesn't depend on frequency, amplitude, or wind.
  Would require bench measurement to verify.

To go further: the amplitude-dependence in Table 3 is the most useful
discriminator — if shift magnitude scales with amplitude, H4 / H5 gain
support; if flat, H6 or H3 (rejected for different reasons) are in play.

## Sanity check: are 9373/170 and 9373/340 really parallel?

If the two "parallel" probes are at the same longitudinal distance (= 9373 mm
from paddle), they should see the same wave phase at the same absolute
sample. The snap-shift array gives us a direct per-run pairwise test:

```
delta = (Probe 9373/170 hg_snap_shift) - (Probe 9373/340 hg_snap_shift)   [samples]
```

If truly parallel → delta ≈ 0 per run (up to probe-specific noise).

Filtered to thesis freqs 1.3-1.7 Hz, quality=ok (n = 83):

| condition       | median delta | std delta | zero-delta rate |
|-----------------|--------------|-----------|-----------------|
| nowind  (n=32)  | +1 sample    | 1.6       | 25.0 %          |
| fullwind (n=51) | +1 sample    | 10.3      | 7.8 %           |

1 sample at 250 Hz = 4 ms. Under deep-water `c_g ≈ 0.55 m/s`, 4 ms of travel
time corresponds to ~2.2 mm longitudinal offset.

**Under nowind**: median = +1 sample → ~2 mm longitudinal difference at most.
25 % of runs snap to the IDENTICAL sample. Std = 1.6 samples means typical
per-run difference is within ±2 samples = ±4 mm equivalent. Probes are
parallel to ~2 mm tolerance.

**Under fullwind**: median unchanged (+1 sample), but std blows up to 10
samples. Wind-induced noise causes the detected upcrossings to jitter between
adjacent candidate samples on one probe vs the other. Not a physical position
change — just upcrossing-detection jitter on noisy signals.

**Candidate explanations for the +1-sample median offset** (not verified;
any or none could be right):
- H7a — genuine mechanical mounting offset of ~2 mm between the two probes
- H7b — probe-specific ULS electronic response lag of ~4 ms between channels
- H7c — systematic algorithmic bias in upcrossing detection on two independent
        noise realisations of the same wave

Distinguishing these would need a bench measurement (lag) or precise optical
survey (position).

**Verdict**: probes are effectively parallel for all practical analysis.

### Implication for the canonical-IN averaging decision

The pipeline's canonical IN is `mean(9373/170, 9373/340)` for the
`march2026_better_rearranging` config. Two independent lines of evidence
support this choice:

1. **Amplitude agreement** — existing methodology figure
   `ch04_parallel_probe_agreement` (`analysis_scratch/parallel_probe_agreement.py`):
   the two probes agree within ±5 % under nowind and ±10 % under
   fullwind 0.2–0.3 V. Averaging reduces single-probe noise.

2. **Phase agreement** (this diagnostic): the two probes snap to the
   same upcrossing sample (or within ±1) for the majority of nowind
   runs. They literally see the same wavefront at the same time.

The mean-IN is not smoothing over two different measurements — it is
averaging two independent readings of the same physical quantity. The
two probes provide statistical noise reduction, not spatial averaging.
This is the stronger (and physically cleaner) justification.

## For another agent wanting to re-derive these numbers

1. Re-run this script (no arguments): `python analysis_scratch/hg_snap_shift_diagnostic.py`
2. Raw per-run data: `hg_snap_shift_diagnostic.csv`
3. Breakdown table: `hg_snap_shift_diagnostic_summary.csv` (long form)
4. Underlying columns in meta.json (per probe):
   - `Probe {pos} hg_snap_shift` — signed shift in samples
   - `Probe {pos} hg_expected_start` — pre-snap theoretical H&G start
   - `Computed Probe {pos} start/end` — post-snap window (used by FFT/LS)
5. Convert to periods: `shift_periods = shift_samples / round(fs / f_paddle)`

