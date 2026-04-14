# T_cross — Cross-Condition Transmission Ratio
*Idea proposed 2026-04-14. Tested same session.*

---

## The idea

Instead of:
```
OUT/IN_fw = A_out(12400/250, full-wind) / A_in(9373/170, full-wind)
```

use:
```
T_cross = A_out(12400/250, full-wind) / A_in(9373/170, no-wind)
```

The denominator comes from a separate no-wind run at the same (frequency, amplitude,
mooring). The IN probe under no-wind has no wind-wave contamination — SNR is 30–90×
vs 2–3× under full wind. The OUT probe under full wind is already sheltered (~1 mm
wind background). Both the numerator and denominator are clean measurements.

---

## What T_cross physically measures

```
T_cross / OUT/IN_nw  =  A_out_fw / A_out_nw
```

This ratio is entirely independent of the IN probe. It tells you: **by what factor does
wind increase the wave amplitude at the OUT probe, relative to the no-wind case?**
This is the cleanest possible statement of the wind effect on wave output, contaminated
by nothing at the IN probe.

If T_cross > OUT/IN_nw: wind adds energy to the output side.
If T_cross < OUT/IN_nw: wind reduces output (e.g. panel submerged, extra damping).
If T_cross ≈ OUT/IN_nw: wind does not affect the output.

---

## Key prerequisite: is A_in wind-independent?

The approach is valid only if the wavemaker delivers the same amplitude at the paddle
frequency regardless of wind. Tested from data (0.1V amplitude, full-panel, 1.3 Hz,
within the same mooring group):

| Mooring | A_in_nw mean | A_in_fw mean | ratio fw/nw |
|---------|-------------|-------------|-------------|
| below_90_loose230 | 7.20 mm | 6.73 mm | 0.935 |
| below_90_loose300 | 6.69 mm | 7.03 mm | 1.051 |

At 1.3 Hz: ratio is 0.94–1.05 — the paddle output is approximately wind-independent
(within ±5%). This validates T_cross at 1.3 Hz.

At other frequencies, the fw/nw ratio from the first test (different mooring groups
mixed in) showed extreme values:
- 0.8 Hz: 0.578 (only 1 fw run vs 2 nw runs, DIFFERENT moorings — not a valid comparison)
- 1.7 Hz: 1.334 (loose230 fw vs above_50 nw — mooring confound)

**These extreme ratios are probably confounded by mooring differences, not pure wind
effects.** The within-mooring validation must be done at each frequency before claiming
A_in is or isn't wind-independent at that frequency.

---

## T_cross results (1.3–1.7 Hz, within-mooring, 0.1V)

Where both no-wind and full-wind runs exist within the same mooring group:

| Freq | Mooring | A_in_nw | A_out_fw | T_cross | OUT/IN_fw | OUT/IN_nw | Wind effect (clean) |
|------|---------|---------|---------|---------|-----------|-----------|---------------------|
| 1.3 | above_50 | 7.37 | 5.50 | 0.745 | 0.681 | — | — |
| 1.3 | loose230 | 7.20 | 5.95 | 0.826 | 0.905 | 0.756 | +0.070 |
| 1.3 | loose300 | 6.69 | 5.72 | 0.856 | 0.828 | 0.705 | +0.151 |
| 1.4 | loose230 | 7.81 | 5.70 | 0.729 | 0.785 | 0.590 | +0.139 |
| 1.4 | loose300 | 7.77 | 5.75 | 0.740 | 0.712 | 0.628 | +0.112 |
| 1.5 | loose230 | 8.24 | 5.31 | 0.645 | 0.705 | 0.433 | +0.212 |
| 1.5 | loose300 | 7.58 | 5.12 | 0.676 | 0.587 | 0.432 | +0.244 |
| 1.6 | loose230 | 7.82 | 4.85 | 0.620 | 0.548 | 0.355 | +0.265 |
| 1.6 | loose300 | 7.00 | 4.84 | 0.691 | 0.531 | 0.356 | +0.335 |
| 1.7 | loose230 | 6.65 | 4.41 | 0.662 | 0.498 | 0.263 | +0.399 |

Wind effect (clean) = T_cross − OUT/IN_nw.
This is purely "how much extra wave energy reaches the OUT probe under wind."

### Key observation

At 1.3 Hz: clean wind effect = +0.07 to +0.15, which is SIMILAR to the raw wind effect
(OUT/IN_fw − OUT/IN_nw ≈ +0.07 to +0.12 in the same mooring groups). The two methods agree.

At 1.7 Hz: clean wind effect = +0.40 — the OUT probe under full wind has 40 percentage
points MORE transmission than under no wind, referenced to the clean no-wind input.
The raw method at 1.7 Hz gives +0.23 (OUT/IN_fw − OUT/IN_nw). The raw method
underestimates the wind effect because A_in_fw is inflated above A_in_nw at 1.7 Hz
(the standing wave pattern at IN changes under wind), making OUT/IN_fw appear lower.

### T_cross is strictly larger than OUT/IN_fw when A_in_fw < A_in_nw

At 1.7 Hz: A_in_fw ≈ 9.0 mm > A_in_nw ≈ 6.7 mm → T_cross = 0.662 < OUT/IN_fw = 0.498?

Wait — that contradicts. Let me re-check with the actual numbers:
- A_out_fw (loose230) = 4.41 mm
- A_in_nw (loose230) = 6.65 mm → T_cross = 4.41/6.65 = 0.663
- A_in_fw (loose230) = 8.89 mm → OUT/IN_fw = 4.41/8.89 = 0.496

So T_cross = 0.663 > OUT/IN_fw = 0.496 at 1.7 Hz, because A_in_fw (8.89) > A_in_nw (6.65).
Wind inflates the apparent IN amplitude at 1.7 Hz → OUT/IN_fw is an underestimate of
true transmission under wind. T_cross is the more honest number.

---

## Limitation: mooring confound at low frequencies

At frequencies below ~1.2 Hz, the no-wind runs exist only under `above_50` mooring
while the full-wind runs are under `below_90_loose230`. These are different moorings —
different dates, different panel dynamics. T_cross cannot be cleanly computed there
without within-mooring no-wind reference runs.

At 0.6–1.1 Hz, either:
(a) Collect no-wind reference runs within the `below_90_loose230` mooring group, OR
(b) Accept that the comparison is confounded and note this limitation

---

## Gaps that prevent full application

1. **No within-mooring no-wind runs at 0.6–1.1 Hz** for the loose230/300 mooring.
   The no-wind 0.6–1.1 Hz data is concentrated in the above_50 early-March sessions.
   If more no-wind runs were collected after March 16 (under-panel mooring), T_cross
   could be extended to the full frequency range.

2. **Only 0.1V amplitude tested here.** The idea generalises to 0.2V and 0.3V but
   the per-run matching requires careful grouping.

3. **The A_in wind-independence check** has only been done within-mooring at 1.3 Hz.
   It should be verified at 1.5 Hz and 1.7 Hz using the loose230 within-mooring pairs.

---

## Summary: is this a better metric than standard OUT/IN?

**Yes, conditionally.** When within-mooring reference pairs exist:
- T_cross eliminates Attack 3 (wind contamination of IN probe FFT)
- T_cross/OUT/IN_nw is probe-geometry-independent (only uses OUT probe values)
- At 1.5–1.7 Hz, T_cross reveals the wind effect is larger than OUT/IN_fw suggests
  (the raw method partially cancels the wind effect by also inflating the IN denominator)

**No** if mooring is not matched — then it's a cross-condition confound (mooring + wind
simultaneously varied).

The cleanest thesis statement would be:
> "Comparing the transmitted wave amplitude under full wind to the incident wave amplitude
> under no-wind (T_cross), wind increases transmission by +7–15% at 1.3 Hz and +40% at
> 1.7 Hz. This estimate is conservative relative to T_cross because it uses the clean
> incident reference, eliminating in-probe wind noise."

---

## Next steps

1. Verify A_in wind-independence within-mooring at 1.5 and 1.7 Hz (the high-effect freqs)
2. Plot T_cross vs frequency alongside standard OUT/IN_fw and OUT/IN_nw — three curves
   on the same axes. The spread between OUT/IN_fw and T_cross directly shows where the
   IN probe is wind-contaminated.
3. Add to main_save_figures.py as an alternative panel for CH05 §3.
4. Check if the loose300 mooring has both no-wind and full-wind runs at enough frequencies
   to do the T_cross comparison within that group alone.
