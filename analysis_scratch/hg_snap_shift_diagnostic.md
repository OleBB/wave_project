# H&G snap-shift diagnostic — pulled from meta.json

Generated: 2026-05-02T16:03:26Z

**Dataset**: fullpanel wave runs, quality_flag=ok, from the two canonical
March-2026 lowrange folders. n_runs = 103.

Shift = (upcrossing-snapped window start) − (theoretical H&G window start),
converted to units of **wave periods** (T = 1/f_paddle).

## Table 1 — Shift by probe and wind (all frequencies pooled)

Median shift in periods, ±std, n_runs. Positive shift = snap moved the
window LATER than the theoretical H&G position.

| probe | wind=no | wind=full |
|---|---|---|
| `9373/170` | **+0.286** ± 0.294  (n=33) | **+0.056** ± 0.240  (n=69) |
| `9373/340` | **+0.281** ± 0.295  (n=33) | **+0.083** ± 0.232  (n=69) |
| `12400/250` | **+0.052** ± 0.233  (n=33) | **+0.035** ± 0.250  (n=69) |
| `8804/250` | **-0.073** ± 0.245  (n=33) | **-0.061** ± 0.252  (n=69) |

## Table 2 — IN-vs-OUT shift disagreement

If the ~−0.2 T offset at OUT were purely a wavemaker phase reference issue,
ALL probes should show the same median shift (c_g(f) predicts the same
arrival phase at every probe relative to paddle). Any cross-probe
disagreement argues for tank-local physics (amplitude dispersion,
reflections, near-panel effects) rather than a pure-phase-reference cause.

| wind | IN (9373/170) median | OUT (12400/250) median | IN − OUT |
|---|---|---|---|
| no | +0.286 | +0.052 | **+0.234** |
| full | +0.056 | +0.035 | **+0.021** |

## Table 3 — Amplitude dependence of OUT-probe shift

Does the −0.2 T offset grow (or shrink) with paddle amplitude? A Stokes-
nonlinearity mechanism would predict the offset magnitude scales with wave
steepness (roughly ∝ ka ∝ amplitude at fixed frequency).

OUT probe (12400/250), median shift in periods per (amp, wind):

| amp [V] | wind=no | wind=full |
|---|---|---|
| 0.1 | +0.104 ± 0.177  (n=15) | -0.026 ± 0.216  (n=20) |
| 0.2 | -0.007 ± 0.227  (n=9) | -0.084 ± 0.338  (n=13) |
| 0.3 | +0.397 ± 0.254  (n=9) | +0.071 ± 0.222  (n=18) |

Same table for IN probe (9373/170):

| amp [V] | wind=no | wind=full |
|---|---|---|
| 0.1 | +0.318 ± 0.258  (n=15) | +0.083 ± 0.269  (n=20) |
| 0.2 | -0.269 ± 0.352  (n=9) | +0.218 ± 0.160  (n=13) |
| 0.3 | +0.317 ± 0.119  (n=9) | +0.073 ± 0.084  (n=18) |

## Table 4 — The 1.4 Hz OUT-probe flip

Observation in the session log: at 1.4 Hz, median OUT shift flips from
~-0.35 T (nowind) to ~+0.35 T (fullwind). A ±0.5 T jump is the boundary
where 'nearest upcrossing' can flip between two adjacent ones. Listing
all 1.4 Hz OUT runs below with individual shifts so another agent can
confirm or falsify.

| path | amp [V] | wind | shift (T) | hg_expected_start | snap start (Computed start) |
|---|---|---|---|---|---|
| `fullpanel-fullwind-amp0100-freq1400-per240-depth580-mstop30-` | 0.1 | full | +0.480 | 6816.0 | 6902.0 |
| `fullpanel-fullwind-amp0200-freq1400-per240-depth580-mstop30-` | 0.2 | full | +0.341 | 6816.0 | 6877.0 |
| `fullpanel-fullwind-amp0300-freq1400-per240-depth580-mstop30-` | 0.3 | full | +0.240 | 6816.0 | 6859.0 |
| `fullpanel-nowind-amp0100-freq1400-per240-depth580-mstop30-ru` | 0.1 | no | -0.302 | 6816.0 | 6762.0 |
| `fullpanel-nowind-amp0200-freq1400-per40-depth580-mstop30-run` | 0.2 | no | -0.397 | 6816.0 | 6745.0 |
| `fullpanel-fullwind-amp0200-freq1400-per40-depth580-mstop30-r` | 0.2 | full | +0.391 | 6816.0 | 6886.0 |
| `fullpanel-fullwind-amp0100-freq1400-per240-depth580-mstop30-` | 0.1 | full | -0.402 | 6816.0 | 6744.0 |
| `fullpanel-fullwind-amp0100-freq1400-per40-depth580-mstop30-r` | 0.1 | full | -0.447 | 6816.0 | 6736.0 |
| `fullpanel-fullwind-amp0200-freq1400-per240-depth580-mstop30-` | 0.2 | full | +0.425 | 6816.0 | 6892.0 |
| `fullpanel-nowind-amp0100-freq1400-per40-depth580-mstop30-run` | 0.1 | no | -0.313 | 6816.0 | 6760.0 |
| `fullpanel-fullwind-amp0300-freq1400-per40-depth580-mstop30-r` | 0.3 | full | +0.246 | 6816.0 | 6860.0 |
| `fullpanel-nowind-amp0300-freq1400-per40-depth580-mstop30-run` | 0.3 | no | +0.419 | 6816.0 | 6891.0 |
| `fullpanel-fullwind-amp0300-freq1400-per40-depth580-mstop30-r` | 0.3 | full | +0.263 | 6816.0 | 6863.0 |
| `fullpanel-nowind-amp0200-freq1400-per240-depth580-mstop30-ru` | 0.2 | no | -0.380 | 6816.0 | 6748.0 |
| `fullpanel-nowind-amp0300-freq1400-per240-depth580-mstop30-ru` | 0.3 | no | +0.447 | 6816.0 | 6896.0 |
| `fullpanel-fullwind-amp0300-freq1400-per240-depth580-mstop30-` | 0.3 | full | +0.279 | 6816.0 | 6866.0 |

## Table 5 — Full breakdown by (freq, amp, wind, probe)

Long-form table, see also `hg_snap_shift_diagnostic_summary.csv`.

| freq [Hz] | amp [V] | wind | probe | n | median | std |
|---|---|---|---|---|---|---|
| 1.30 | 0.1 | no | `9373/170` | 9 | +0.318 | 0.019 |
| 1.30 | 0.1 | no | `9373/340` | 9 | +0.318 | 0.022 |
| 1.30 | 0.1 | no | `12400/250` | 9 | +0.109 | 0.031 |
| 1.30 | 0.1 | no | `8804/250` | 9 | -0.078 | 0.014 |
| 1.30 | 0.1 | full | `9373/170` | 10 | +0.083 | 0.062 |
| 1.30 | 0.1 | full | `9373/340` | 10 | +0.091 | 0.056 |
| 1.30 | 0.1 | full | `12400/250` | 10 | -0.018 | 0.036 |
| 1.30 | 0.1 | full | `8804/250` | 10 | -0.240 | 0.055 |
| 1.30 | 0.2 | no | `9373/170` | 2 | +0.216 | 0.004 |
| 1.30 | 0.2 | no | `9373/340` | 2 | +0.203 | 0.000 |
| 1.30 | 0.2 | no | `12400/250` | 2 | +0.029 | 0.004 |
| 1.30 | 0.2 | no | `8804/250` | 2 | -0.169 | 0.011 |
| 1.30 | 0.2 | full | `9373/170` | 3 | +0.036 | 0.032 |
| 1.30 | 0.2 | full | `9373/340` | 3 | +0.016 | 0.051 |
| 1.30 | 0.2 | full | `12400/250` | 3 | -0.125 | 0.055 |
| 1.30 | 0.2 | full | `8804/250` | 3 | -0.344 | 0.021 |
| 1.30 | 0.3 | no | `9373/170` | 2 | +0.138 | 0.011 |
| 1.30 | 0.3 | no | `9373/340` | 2 | +0.143 | 0.011 |
| 1.30 | 0.3 | no | `12400/250` | 2 | -0.094 | 0.015 |
| 1.30 | 0.3 | no | `8804/250` | 2 | -0.266 | 0.007 |
| 1.30 | 0.3 | full | `9373/170` | 4 | +0.013 | 0.017 |
| 1.30 | 0.3 | full | `9373/340` | 4 | +0.016 | 0.029 |
| 1.30 | 0.3 | full | `12400/250` | 4 | -0.224 | 0.012 |
| 1.30 | 0.3 | full | `8804/250` | 4 | -0.362 | 0.014 |
| 1.40 | 0.1 | no | `9373/170` | 2 | +0.469 | 0.008 |
| 1.40 | 0.1 | no | `9373/340` | 2 | +0.461 | 0.012 |
| 1.40 | 0.1 | no | `12400/250` | 2 | -0.307 | 0.008 |
| 1.40 | 0.1 | no | `8804/250` | 2 | +0.165 | 0.004 |
| 1.40 | 0.1 | full | `9373/170` | 3 | +0.263 | 0.057 |
| 1.40 | 0.1 | full | `9373/340` | 3 | +0.229 | 0.062 |
| 1.40 | 0.1 | full | `12400/250` | 3 | -0.402 | 0.523 |
| 1.40 | 0.1 | full | `8804/250` | 3 | +0.056 | 0.034 |
| 1.40 | 0.2 | no | `9373/170` | 2 | +0.363 | 0.024 |
| 1.40 | 0.2 | no | `9373/340` | 2 | +0.355 | 0.020 |
| 1.40 | 0.2 | no | `12400/250` | 2 | -0.388 | 0.012 |
| 1.40 | 0.2 | no | `8804/250` | 2 | +0.073 | 0.024 |
| 1.40 | 0.2 | full | `9373/170` | 3 | +0.151 | 0.006 |
| 1.40 | 0.2 | full | `9373/340` | 3 | +0.134 | 0.046 |
| 1.40 | 0.2 | full | `12400/250` | 3 | +0.391 | 0.042 |
| 1.40 | 0.2 | full | `8804/250` | 3 | -0.095 | 0.020 |
| 1.40 | 0.3 | no | `9373/170` | 2 | +0.265 | 0.020 |
| 1.40 | 0.3 | no | `9373/340` | 2 | +0.254 | 0.020 |
| 1.40 | 0.3 | no | `12400/250` | 2 | +0.433 | 0.020 |
| 1.40 | 0.3 | no | `8804/250` | 2 | -0.034 | 0.016 |
| 1.40 | 0.3 | full | `9373/170` | 4 | +0.047 | 0.015 |
| 1.40 | 0.3 | full | `9373/340` | 4 | +0.067 | 0.017 |
| 1.40 | 0.3 | full | `12400/250` | 4 | +0.254 | 0.018 |
| 1.40 | 0.3 | full | `8804/250` | 4 | -0.193 | 0.012 |
| 1.50 | 0.1 | no | `9373/170` | 2 | -0.314 | 0.004 |
| 1.50 | 0.1 | no | `9373/340` | 2 | -0.323 | 0.000 |
| 1.50 | 0.1 | no | `12400/250` | 2 | +0.332 | 0.004 |
| 1.50 | 0.1 | no | `8804/250` | 2 | -0.003 | 0.716 |
| 1.50 | 0.1 | full | `9373/170` | 3 | -0.395 | 0.458 |
| 1.50 | 0.1 | full | `9373/340` | 3 | -0.431 | 0.486 |
| 1.50 | 0.1 | full | `12400/250` | 3 | +0.180 | 0.100 |
| 1.50 | 0.1 | full | `8804/250` | 3 | +0.293 | 0.108 |
| 1.50 | 0.2 | no | `9373/170` | 2 | -0.428 | 0.030 |
| 1.50 | 0.2 | no | `9373/340` | 2 | -0.449 | 0.008 |
| 1.50 | 0.2 | no | `12400/250` | 2 | +0.207 | 0.021 |
| 1.50 | 0.2 | no | `8804/250` | 2 | +0.359 | 0.025 |
| 1.50 | 0.2 | full | `9373/170` | 3 | +0.311 | 0.033 |
| 1.50 | 0.2 | full | `9373/340` | 3 | +0.341 | 0.040 |
| 1.50 | 0.2 | full | `12400/250` | 3 | -0.084 | 0.019 |
| 1.50 | 0.2 | full | `8804/250` | 3 | +0.144 | 0.003 |
| 1.50 | 0.3 | no | `9373/170` | 2 | +0.344 | 0.038 |
| 1.50 | 0.3 | no | `9373/340` | 2 | +0.320 | 0.030 |
| 1.50 | 0.3 | no | `12400/250` | 2 | -0.027 | 0.047 |
| 1.50 | 0.3 | no | `8804/250` | 2 | +0.153 | 0.030 |
| 1.50 | 0.3 | full | `9373/170` | 4 | +0.162 | 0.027 |
| 1.50 | 0.3 | full | `9373/340` | 4 | +0.159 | 0.033 |
| 1.50 | 0.3 | full | `12400/250` | 4 | -0.284 | 0.022 |
| 1.50 | 0.3 | full | `8804/250` | 4 | +0.018 | 0.015 |
| 1.60 | 0.1 | no | `9373/170` | 2 | -0.067 | 0.023 |
| 1.60 | 0.1 | no | `9373/340` | 2 | -0.074 | 0.014 |
| 1.60 | 0.1 | no | `12400/250` | 2 | +0.026 | 0.018 |
| 1.60 | 0.1 | no | `8804/250` | 2 | -0.183 | 0.014 |
| 1.60 | 0.1 | full | `9373/170` | 3 | -0.353 | 0.068 |
| 1.60 | 0.1 | full | `9373/340` | 3 | -0.423 | 0.023 |
| 1.60 | 0.1 | full | `12400/250` | 3 | -0.250 | 0.081 |
| 1.60 | 0.1 | full | `8804/250` | 3 | -0.353 | 0.027 |
| 1.60 | 0.2 | no | `9373/170` | 2 | -0.288 | 0.027 |
| 1.60 | 0.2 | no | `9373/340` | 2 | -0.295 | 0.018 |
| 1.60 | 0.2 | no | `12400/250` | 2 | -0.199 | 0.009 |
| 1.60 | 0.2 | no | `8804/250` | 2 | -0.369 | 0.023 |
| 1.60 | 0.2 | full | `9373/170` | 3 | +0.455 | 0.010 |
| 1.60 | 0.2 | full | `9373/340` | 3 | +0.423 | 0.013 |
| 1.60 | 0.2 | full | `12400/250` | 3 | +0.449 | 0.561 |
| 1.60 | 0.2 | full | `8804/250` | 3 | +0.449 | 0.004 |
| 1.60 | 0.3 | no | `9373/170` | 3 | +0.436 | 0.010 |
| 1.60 | 0.3 | no | `9373/340` | 3 | +0.423 | 0.004 |
| 1.60 | 0.3 | no | `12400/250` | 3 | +0.410 | 0.010 |
| 1.60 | 0.3 | no | `8804/250` | 3 | +0.327 | 0.006 |
| 1.60 | 0.3 | full | `9373/170` | 5 | +0.109 | 0.029 |
| 1.60 | 0.3 | full | `9373/340` | 5 | +0.122 | 0.016 |
| 1.60 | 0.3 | full | `12400/250` | 5 | +0.090 | 0.023 |
| 1.60 | 0.3 | full | `8804/250` | 5 | +0.096 | 0.019 |
| 1.70 | 0.1 | full | `9373/170` | 1 | +0.429 | n/a |
| 1.70 | 0.1 | full | `9373/340` | 1 | +0.374 | n/a |
| 1.70 | 0.1 | full | `12400/250` | 1 | -0.048 | n/a |
| 1.70 | 0.1 | full | `8804/250` | 1 | -0.503 | n/a |
| 1.70 | 0.2 | no | `9373/170` | 1 | -0.388 | n/a |
| 1.70 | 0.2 | no | `9373/340` | 1 | -0.401 | n/a |
| 1.70 | 0.2 | no | `12400/250` | 1 | -0.007 | n/a |
| 1.70 | 0.2 | no | `8804/250` | 1 | -0.361 | n/a |
| 1.70 | 0.2 | full | `9373/170` | 1 | +0.218 | n/a |
| 1.70 | 0.2 | full | `9373/340` | 1 | +0.211 | n/a |
| 1.70 | 0.2 | full | `12400/250` | 1 | -0.395 | n/a |
| 1.70 | 0.2 | full | `8804/250` | 1 | +0.306 | n/a |
| 1.70 | 0.3 | full | `9373/170` | 1 | -0.163 | n/a |
| 1.70 | 0.3 | full | `9373/340` | 1 | -0.190 | n/a |
| 1.70 | 0.3 | full | `12400/250` | 1 | +0.116 | n/a |
| 1.70 | 0.3 | full | `8804/250` | 1 | -0.061 | n/a |

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

## For another agent wanting to re-derive these numbers

1. Re-run this script (no arguments): `python analysis_scratch/hg_snap_shift_diagnostic.py`
2. Raw per-run data: `hg_snap_shift_diagnostic.csv`
3. Breakdown table: `hg_snap_shift_diagnostic_summary.csv` (long form)
4. Underlying columns in meta.json (per probe):
   - `Probe {pos} hg_snap_shift` — signed shift in samples
   - `Probe {pos} hg_expected_start` — pre-snap theoretical H&G start
   - `Computed Probe {pos} start/end` — post-snap window (used by FFT/LS)
5. Convert to periods: `shift_periods = shift_samples / round(fs / f_paddle)`

