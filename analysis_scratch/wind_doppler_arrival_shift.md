# Wind arrival-shift study (2026-05-05)

**Headline observation**: under fullwind, the paddle wave H&G
window snaps **132 ms (≈0.18 T) earlier at IN**
than under nowind — and the shift is **approximately the same size**
**at every probe**, regardless of distance from the paddle.

## What I expected vs what I found

Initial hypothesis (H1): wind-driven surface current adds to c_g.
Prediction: Δt(r) = -r·U/c_g², linear through origin.

**Result**: Δt(r) is essentially **flat** in r in the cleanest cells,
not linear. Cleanest example — 1.3 Hz, A=0.2 V (canon, full panel):

    probe    r_m   dt_T  n_fw  n_nw    dt_ms
 8804/250  8.804 -0.174     3     2 -134.000
 9373/170  9.373 -0.172     3     2 -132.667
 9373/340  9.373 -0.175     3     2 -134.667
12400/250 12.400 -0.174     3     2 -134.000

All four probes shift by ~134 ms, irrespective of distance
(r ranges from 8.8 m to 12.4 m, a 41 % spread). A Doppler model
would predict the OUT shift to be 12.4/8.8 = 1.41× the 8804 shift.

Across all 9 'clean' cells (no period-aliasing on any probe), a
uniform-shift model fits better than a linear-Doppler model in 7/9.

## What this rules out

- **Simple uniform-current Doppler.** Killed by the flat Δt(r).
- **Detection-threshold bias from envelope amplitude (H3 from
  earlier discussion).** Would scale with envelope slope at each
  probe; OUT envelope is ~5× smaller than IN, so OUT shift should
  be much larger if H3 dominated. Observed: OUT shift ≈ IN shift.

## What survives — candidate explanations

*Candidate H4* — **Wave-source timing shift.** Under fullwind there
are pre-existing wind-waves at the paddle. The first detectable
'paddle-frequency' upcrossing forms slightly earlier because the
wind-wave carrier and the paddle wave constructively combine. The
wave field then propagates at normal c_g and arrives uniformly
earlier at every probe. Predicts a flat Δt(r). **Consistent.**

*Candidate H7* — **Paddle hardware response under wind load.** Air
drag on the paddle face could shift its actual motion onset by
~100 ms relative to the command signal. A trigger-time shift at
the source predicts a flat Δt(r) at all probes. **Consistent.**
Test: inspect paddle command/feedback channel if logged.

*Candidate H8* — **Snap-anchoring on wind-wave upcrossings.** The
H&G snap finds the nearest zero-upcrossing within ±T of the
theoretical start. Under fullwind at IN/8804 there are wind-wave
upcrossings every ~0.2-0.3 s — the snap may bias toward an earlier
one. **Inconsistent with the uniform shift**: OUT (no wind waves —
verified in the pre-paddle plot) should not have this bias, yet
OUT also shifts by the same amount.

## Pre-paddle sanity check (paddle_start.png)

Pre-paddle η at OUT shows essentially no wind-wave activity
(panel does shadow effectively). Pre-paddle η at 8804 and IN
shows clear wind-wave noise (~5-10 mm). Both fw and nw recordings
appear to share the same recorded t=0; no obvious paddle-trigger
offset is visible. **H7 hardware shift cannot be confirmed from
η alone — would need the paddle command channel.**

## Per-leg 'apparent current' if you insist on Doppler

 f_hz  amp_V  U_paddle_to_8804_mm_s  U_8804_to_9373_mm_s  U_9373_to_12400_mm_s
  1.3    0.1                    5.0                 37.5                  -9.9
  1.3    0.2                    5.5                 -0.9                   0.2
  1.3    0.3                    2.9                 13.4                   0.6
  1.4    0.1                    2.7                 30.3                 -27.1
  1.4    0.2                    4.3                 18.3                 -73.0
  1.4    0.3                    4.1                 24.1                  -3.4
  1.5    0.1                   -6.2                 55.9                  17.4
  1.5    0.2                    4.4               -305.3                  62.5
  1.5    0.3                    2.8                 16.2                   4.6
  1.6    0.1                    2.7                 25.7                   1.3
  1.6    0.2                  -13.8                 19.3                  19.9
  1.6    0.3                    3.9                 20.9                   0.3

These numbers are physically suspect: the paddle→8804 leg shows
a tiny consistent ~3-5 mm/s, the 8804→9373 leg shows much larger
20-50 mm/s, and the 9373→12400 (panel-shadow) leg fluctuates
wildly including negative values. None of this is consistent with
a real surface current. The decomposition is the math forcing a
flat Δt(r) into a linear-Doppler frame.

## Interpretation

The wave field arrives uniformly earlier at every probe under
fullwind. This is a **source-side or detection-side** effect, not
a propagation effect. Most likely candidates are H4 (wave-
generation timing shifted by pre-existing wind-wave field) or
H7 (paddle hardware response under wind load).

**This does not refute the wind→IN coupling story** documented in
`methodology_wind_enhances_A_in.md` (10-17% A_in enhancement). The
amplitude enhancement and the timing shift are two distinct
observations that may share a single source (wind interacts with
wave generation or first cycles) but are independent measurements.

## Files
- wind_doppler_arrival_shift.csv — per-cell, per-probe table
- wind_doppler_arrival_shift_legs.csv — per-leg apparent current
- wind_doppler_arrival_shift_dr.png — Δt vs r scatter (key plot)
- wind_doppler_arrival_shift_eta.png — η(t) overlay 1.3 Hz 0.2 V
- wind_doppler_arrival_shift_paddle_start.png — pre-paddle sanity
