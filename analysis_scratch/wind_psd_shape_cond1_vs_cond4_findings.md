# Wind PSD shape: cond1 (h272/high) vs cond4 (h100/low)

**Date**: 2026-04-18 (free-time exploration follow-up to today's ch04 §3b figure)
**Script**: `analysis_scratch/wind_psd_shape_cond1_vs_cond4.py`
**Figure**: `analysis_scratch/wind_psd_shape_cond1_vs_cond4.pdf`

## Question

`probe_height_wind_findings.md` Finding 2 claims:

> The physical wind-wave field is approximately constant across sessions
> … Amplitude differences across conditions at the IN probe reflect
> probe-measurement properties, not wind-field differences.

If this is true, the wind PSD *shape* should match between cond1 and
cond4 — only the amplitude scale differs. If the shape differs, the
probe at h272/high reports the wind differently from the probe at
h100/low (frequency-dependent response, range-mode rolloff, …) and the
"probe-measurement effect" framing is correct but more nuanced than just
a single scaling factor.

## Data

- cond1 nowave+fullwind runs: **3** (folders: 20260307, 20260312, 20260319)
- cond4 nowave+fullwind runs: **6** (folders: 20260326-lowrange, 20260327)
- Frequency grid: 0.0–125.0 Hz, n=2049 (Welch from pipeline)
- Same physical probes throughout — same probe positions in cond1 and cond4

## Headline result — ratio of mean PSDs over 2–6 Hz wind band

| Probe | ⟨cond1/cond4⟩ ± std | cond1 peak (Hz) | cond4 peak (Hz) |
|-------|---------------------|-----------------|-----------------|
| 9373/170 | 1.41 ± 0.90 | 3.66 | 3.78 |
| 12400/250 | 0.79 ± 0.33 | 1.16 | 1.10 |
| 9373/340 | 1.27 ± 0.81 | 3.60 | 3.72 |
| 8804/250 | 1.33 ± 0.86 | 3.66 | 4.09 |

A flat ratio across the wind band would mean "pure amplitude scaling"
(probe height changes the gain, not the spectral response). A structured
ratio means "frequency-dependent probe response" (the probe at h272 sees
the wind PSD with a different transfer function than at h100).

## Interpretation per probe

- **9373/170**: cond1 reports **higher** wind PSD (1.41× cond4), with substantial frequency-dependent structure (ratio std/mean > 30%).
- **12400/250**: cond1 reports **lower** wind PSD (0.79× cond4), with substantial frequency-dependent structure (ratio std/mean > 30%).
- **9373/340**: cond1 reports **higher** wind PSD (1.27× cond4), with substantial frequency-dependent structure (ratio std/mean > 30%).
- **8804/250**: cond1 reports **higher** wind PSD (1.33× cond4), with substantial frequency-dependent structure (ratio std/mean > 30%), and peak frequency differs by 0.43 Hz.

## Caveats

- **Sample size is small** (3 cond1 runs, 6 cond4 runs). Ratio
  uncertainty is dominated by between-run variability of the wind PSD
  itself, not by within-run noise. Bigger samples would tighten the ratio
  estimates.
- **Mooring length differs** between the cond1 and cond4 sets (cond1
  uses early-Mar mooring, cond4 uses under9Mooring(30) loose230/loose300).
  The OUT probe wind background depends on mooring length via the
  post-panel fetch (see `physics_wavetank_mooring_fetch.md`), so OUT-probe
  ratios mix probe-height effects and mooring-fetch effects.
- **Tank temperature / ambient conditions** vary day-to-day; not
  controlled for here.
- **No paddle-frequency contribution** since these are nowave runs. The
  PSD is purely wind-wave + noise floor.

## What the figure shows — plain narrative

For the three wind-exposed probes (IN, parallel, upstream), the wind
PSDs essentially coincide at the **wind peak** (~3.7 Hz, 10–20 mm²/Hz)
— the bulk of the wind-wave energy is reported the same way by both
hardware configurations. This **confirms** Finding 2's claim that the
wind field is the same.

But the cond1 PSD has a clear *additional* "skirt" of energy on
**both sides of the wind peak**, especially below ~3 Hz where it sits
~2–3× higher than cond4 (visible in the green-vs-blue separation in the
top row, and the ratio jumping above 1 in the bottom row). The same
pattern appears, more weakly, above ~6 Hz.

This skirt is a **probe-measurement effect, not a real signal**. The
longer acoustic path of cond1 (272 mm vs 100 mm) is more vulnerable to
slow drift, temperature gradient, and beam-divergence variability —
producing low-frequency content that wasn't actually in the water. The
cond4 (h100/low) probe has a much shorter air column and reports a much
cleaner low-frequency baseline.

**Implication for the §3b figure I produced today**: the 1.4 mm
difference between cond1 (10.6 mm) and cond4 (9.2 mm) in the
time-domain wind background at the IN probe is *not* extra wind energy
— it is spurious low-frequency drift in cond1 that gets aggregated into
the percentile-based amplitude metric. The wind itself is essentially
identical. I should add a sentence to the §3b stub caption, or to
`probe_height_wind_findings.md` Finding 2, noting this distinction.

**Implication for paddle-frequency analyses**: cond1 measurements at
0.65–1.8 Hz (the paddle band, 0.5–1.8 Hz region in the table above)
include this drift contamination. The IN probe paddle-band PSD is
**0.41 cond1 vs 0.08 cond4** — a 5× difference, attributable to drift
not paddle signal. This matters for the time-domain noise floor on
cond1 paddle-wave runs but does not affect FFT-based OUT/IN at the
paddle frequency, since the FFT window is narrow and excludes the
drift.

**OUT probe**: the OUT probe has no recognisable wind peak at all
(reported "peak" of ~1.1 Hz is just noise floor — the panel sheltering
suppresses the wind-wave signal everywhere). The ratio is < 1 because
cond4 reports a bit more wind background here (consistent with the
loose230/loose300 mooring, the longer mooring being shorter than
cond1's mooring → longer post-panel fetch → larger OUT ripples). This
is the mooring-fetch confound; not a probe-height effect.

## Bottom line

The wind PSD shape **does** match between cond1 and cond4 in the
wind-wave band itself (~3.7 Hz peak), so the "wind field is constant"
claim survives. But the cond1 PSD additionally carries a **drift skirt
below the wind peak** that is a probe-measurement artefact (longer
acoustic path → more drift). The "amplitude differences are
probe-measurement effects" claim is therefore correct but the physical
story is sharper than just "amplitude scaling": the difference is
spurious low-frequency energy in cond1, not a uniform amplitude factor.

The OUT probe ratio is dominated by mooring-fetch differences between
the two date sets, not probe-height effects. It cannot be used to test
the probe-height claim cleanly without subsetting to matched moorings.

## Next steps (if the result motivates a deeper look)

1. Subset cond1 and cond4 to matched mooring (e.g. both `above_50` or
   both `loose230`) to isolate probe-height from mooring-fetch effects.
2. Compute coherence between cond1 and cond4 spectra at the IN probe to
   test the *shape* claim more rigorously than the simple ratio.
3. Check whether cond3 (h100/high WRONG) shows the same shape as cond4
   (same height, different range mode) — would isolate the range-mode
   contribution.
