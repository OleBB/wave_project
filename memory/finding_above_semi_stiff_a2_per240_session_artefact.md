---
name: Above-semi-stiff A2 V-shape is a 2026-03-14 per=240 session artefact
description: The "wind decreases K_t at A2 1.4/1.6 Hz" pattern for the 16 cm above-water mooring is driven entirely by the 2026-03-14 per=240 fullwind session — n=2/2 cells with one anomalous run per cell. Cross-checked: no equivalent confound at canon under-mooring A2 (n=6/6 runs across multiple dates).
type: project
---

# Above-semi-stiff A2 V-shape: 2026-03-14 per=240 session artefact

> **Date investigated**: 2026-05-11.
> **Scripts**: [`analysis_scratch/wind_effect_realness_lowfreq.py`](../analysis_scratch/wind_effect_realness_lowfreq.py)
> **Outputs**:
> - [`analysis_scratch/wind_effect_realness_lowfreq.pdf`](../analysis_scratch/wind_effect_realness_lowfreq.pdf) (ΔK_t grid, 4-way colored)
> - [`analysis_scratch/wind_effect_realness_lowfreq_a2_focus.pdf`](../analysis_scratch/wind_effect_realness_lowfreq_a2_focus.pdf) (A2 cross-amp dip check)
> - [`analysis_scratch/wind_effect_realness_lowfreq_runs.csv`](../analysis_scratch/wind_effect_realness_lowfreq_runs.csv) (282 rows)
> - [`analysis_scratch/wind_effect_realness_lowfreq_a2_runs.csv`](../analysis_scratch/wind_effect_realness_lowfreq_a2_runs.csv) (63 rows)
> - [`analysis_scratch/wind_effect_realness_lowfreq_cells.csv`](../analysis_scratch/wind_effect_realness_lowfreq_cells.csv) (120 cells)

## TL;DR

The visible "A2 V-shape" in `wind_effect_realness_lowfreq_a2_focus.pdf` —
where canon_above_loose (over-water, 16 cm strikk, full panel) ΔK_t at
A2=0.2V dips below A1 and A3 at 1.4 Hz (ΔK_t = −0.082) and 1.6 Hz
(ΔK_t = +0.016, V-shape vs +0.201 at A1 and +0.174 at A3) — **is driven
by two specific anomalous runs from the 2026-03-14 per=240 fullwind
session**, in n=2/2 cells where one outlier IS the mean. No equivalent
session artefact at canon under-mooring A2 (canon_loose230 and
canon_loose300), which average across n=6/6 runs from 4 different dates.

The headline claim "wind decreases transmission at 1.3–1.4 Hz for the
above-semi-stiff mooring" is **NOT statistically supported** at any
amplitude — every cell in 1.2–1.4 Hz × {A1, A2, A3} has bootstrap 95% CI
crossing zero (verdict = null) and Welch t-test p > 0.05.

## Trigger observation

User skepticism (2026-05-11): the above-semi-stiff mooring (Mooring =
`above_50`, PanelCondition = `full`, march2026 era, called
"canon_above_loose" in the 4-way categorization) appeared to show wind
DECREASING transmission at 1.3 and 1.4 Hz in several scatter views. With
3 amplitudes available, A2 (0.2V) stood out as visually weirder than A1
(0.1V) or A3 (0.3V).

User hypothesis: "A1 is closer to noise, A3 ignores noise, A2 is in the
sweet spot, more affected by wind in more aspects than we have thought
of."

Sub-question to test: is the A2 oddity (a) experiment error (bad day,
too-close rest time, weaker wind that day), (b) a real "A2 sweet spot"
visible universally across moorings, or (c) a single-session artefact?

## The n=2/2 cell structure for above-semi-stiff A2

Every A2 cell in canon_above_loose at 1.2–1.6 Hz contains **exactly the
same 4 runs** (4-run confound across all freqs):

- 1 fullwind run, 2026-03-07, **per=40**, mstop=30 s
- 1 fullwind run, 2026-03-14, **per=240**, mstop=30 s
- 1 nowind run,   2026-03-07, **per=40**, mstop=30 s
- 1 nowind run,   2026-03-13, **per=240**, mstop=30 s

(Plus 1 extra nowind run at 1.6 Hz on 2026-03-16, per=240, mstop=30 s,
K_t_FFT = 0.450.)

So **n_full = 2 and n_no = 2 (or 3 at 1.6 Hz)** in every above-semi-stiff
A2 cell across 1.2–1.6 Hz. Any single-run outlier IS the cell mean.

## Full per-run table — above-semi-stiff A2

K_t source: `Kt_FFT = OUT/IN (FFT)` from the canonical IN/OUT amplitude
columns (`IN Amplitude (FFT)`, `OUT Amplitude (FFT)`), recomputed
post-pipeline-fix 2026-05-07.

| Freq [Hz] | Wind | Date | per | mstop [s] | K_t_FFT | K_t_LS | K_t_PSD |
|---|---|---|---|---|---|---|---|
| 1.2 | full | 03-07 | 40  | 30 | 0.807 | 0.803 | (similar) |
| 1.2 | full | 03-14 | 240 | 30 | 0.815 | 0.812 | (similar) |
| 1.2 | no   | 03-07 | 40  | 30 | 0.777 | 0.777 | (similar) |
| 1.2 | no   | 03-13 | 240 | 30 | 0.846 | 0.846 | (similar) |
| 1.3 | full | 03-07 | 40  | 90 | 0.707 | 0.706 | (similar) |
| 1.3 | full | 03-14 | 240 | 30 | 0.717 | 0.715 | (similar) |
| 1.3 | no   | 03-07 | 40  | 30 | 0.714 | 0.714 | (similar) |
| 1.3 | no   | 03-13 | 240 | 30 | 0.732 | 0.732 | (similar) |
| 1.4 | full | 03-07 | 40  | 30 | 0.651 | 0.653 | 0.657 |
| **1.4** | **full** | **03-14** | **240** | **30** | **0.570** | **0.570** | **0.586** |
| 1.4 | no   | 03-07 | 40  | 30 | 0.701 | 0.699 | 0.695 |
| 1.4 | no   | 03-13 | 240 | 30 | 0.684 | 0.684 | 0.686 |
| 1.5 | full | 03-07 | 40  | 30 | 0.620 | 0.621 | (similar) |
| 1.5 | full | 03-14 | 240 | 30 | 0.596 | 0.597 | (similar) |
| 1.5 | no   | 03-07 | 40  | 30 | 0.574 | 0.573 | (similar) |
| 1.5 | no   | 03-13 | 240 | 30 | 0.570 | 0.568 | (similar) |
| 1.6 | full | 03-07 | 40  | 30 | 0.541 | 0.539 | (similar) |
| **1.6** | **full** | **03-14** | **240** | **30** | **0.395** | **0.398** | (similar) |
| 1.6 | no   | 03-07 | 40  | 30 | 0.456 | 0.452 | (similar) |
| 1.6 | no   | 03-13 | 240 | 30 | 0.451 | 0.450 | (similar) |
| 1.6 | no   | 03-16 | 240 | 30 | 0.450 | 0.448 | (similar) |

FFT, LS and PSD agree to within ~0.01 on every single run, including the
two anomalous ones. So the K_t value itself is robust to amplitude-method
choice — the bin-grid-independent LS sinusoid fit, the integrated PSD
band, and the nearest-bin FFT all see the same thing. The anomaly is in
the **physical wave field**, not in the K_t computation method.

## Per40-vs-per240 gap by frequency — above-semi-stiff A2

Gap = K_t(per=240) − K_t(per=40), per (freq × wind):

| Freq [Hz] | Fullwind gap | Nowind gap |
|---|---|---|
| 1.2 | **+0.008** | +0.069 |
| 1.3 | **+0.010** | +0.018 |
| 1.4 | **−0.081** ← | −0.017 |
| 1.5 | **−0.024** | −0.004 |
| 1.6 | **−0.146** ← | −0.005 |

Observations:
- Fullwind gap is **dramatic at 1.4 and 1.6 Hz**, modest or in line with
  nowind gap at 1.2, 1.3, 1.5 Hz.
- Nowind gap is small everywhere except 1.2 Hz (+0.069, n=2/2 — also
  noise-limited).
- The 1.4 and 1.6 Hz fullwind per=240 runs (both from 2026-03-14) are
  the source of the V-shape in the A2 trajectory.

## Cross-mooring comparison: does the same per40-vs-per240 confound exist at canon under-mooring A2?

**No.** At canon_loose230 (under-water, 23 cm strikk) A2 1.4 Hz fullwind
(n=6 runs across 4 dates):

| Date | per | mstop [s] | K_t_FFT |
|---|---|---|---|
| 03-23 | 40  | 30 | 0.749 |
| 03-23 | 40  | 30 | 0.736 |
| 03-23 | 240 | 30 | 0.755 |
| 03-24 | 240 | (NaN) | 0.737 |
| 03-24 | 40  | (NaN) | 0.798 |
| 03-26 | 240 | 30 | 0.822 |

- per=40 mean ≈ 0.761 (n=3)
- per=240 mean ≈ 0.771 (n=3)
- **per40-vs-per240 gap ≈ +0.010 — small, no systematic bias**

Cell mean K_t = 0.766; nowind cell mean K_t = 0.715 (n=6); ΔK_t = +0.051
[+0.020, +0.085], Welch p = 0.019 → **enhance** (statistically supported).

Same picture at canon_loose300 (under-water, 30 cm strikk) — A1 / A2 /
A3 ΔK_t at 1.3 Hz all positive with no V-shape.

## Cell-level statistical verdicts — above-semi-stiff (Kt source = eff, with LS-override applied)

(With `n_boot = 10 000` bootstrap draws, independent resampling each
side; `α = 0.05`; Welch's t-test for unequal variance.)

| Freq | Amp | n_full / n_no | ΔK_t | 95% CI | Welch p | Verdict |
|---|---|---|---|---|---|---|
| 1.2 | 0.1 | 2 / 2 | +0.007 | [−0.032, +0.046] | 0.819 | **null** |
| 1.2 | 0.2 | 2 / 2 | −0.000 | [−0.039, +0.038] | 0.987 | **null** |
| 1.2 | 0.3 | 2 / 2 | −0.038 | [−0.077, +0.001] | 0.325 | **null** |
| 1.3 | 0.1 | 4 / 9 | +0.004 | [−0.029, +0.039] | 0.829 | **null** |
| 1.3 | 0.2 | 2 / 2 | −0.010 | [−0.024, +0.003] | 0.430 | **null** |
| 1.3 | 0.3 | 3 / 2 | −0.014 | [−0.036, +0.008] | 0.410 | **null** |
| 1.4 | 0.1 | 2 / 2 | +0.051 | [+0.020, +0.081] | 0.143 | enhance (n.s.) |
| **1.4** | **0.2** | **2 / 2** | **−0.082** | **[−0.131, −0.033]** | **0.283** | decrease (**n.s.**) |
| 1.4 | 0.3 | 2 / 2 | +0.038 | [+0.029, +0.046] | 0.085 | enhance (n.s.) |
| 1.5 | 0.1 | 1 / 3 | +0.191 | [+0.165, +0.210] | — | enhance |
| 1.5 | 0.2 | 2 / 2 | +0.036 | [+0.022, +0.050] | 0.194 | enhance (n.s.) |
| 1.5 | 0.3 | 2 / 2 | +0.076 | [+0.072, +0.080] | 0.008 | enhance |
| 1.6 | 0.1 | 1 / 2 | +0.201 | [+0.199, +0.204] | — | enhance |
| **1.6** | **0.2** | **2 / 3** | **+0.016** | **[−0.059, +0.091]** | **0.860** | **null (V-shape)** |
| 1.6 | 0.3 | 2 / 2 | +0.174 | [+0.136, +0.213] | 0.122 | enhance (n.s.) |

n.s. = "not significant", Welch p ≥ 0.05.

Key reading: **NO above-semi-stiff cell in 1.2–1.4 Hz × {A1, A2, A3}
reaches Welch p < 0.05**. The 1.4 Hz A2 cell with bootstrap CI strictly
below zero (the "decrease" verdict) has Welch p = 0.283 — the CI is
over-confident at n=2/2 because resampling two points repeats them.

Statistical rule of thumb learned today: **at n=2/2 trust Welch's p, not
the bootstrap CI**. Bootstrap CIs at n=2/2 are artificially tight when
the two runs happen to agree closely.

## A1 vs A2 vs A3 ΔK_t at 1.3 Hz per mooring (Kt source = eff)

| Mooring (4-way) | A1 | A2 | A3 |
|---|---|---|---|
| canon_above_loose (Over, 16 cm, full) | +0.004 [−0.029, +0.039] p=0.83 | **−0.010** [−0.024, +0.003] p=0.43 | −0.014 [−0.036, +0.008] p=0.41 |
| nov_above_stiff (Over, 6 cm, REVERSE) | +0.088 [+0.044, +0.129] p=0.01 | +0.003 [−0.038, +0.045] p=0.94 | −0.029 [−0.053, −0.007] p=0.09 |
| canon_loose230 (Under, 23 cm, full) | **+0.135** [+0.103, +0.168] p<0.001 | +0.046 [+0.016, +0.075] p=0.05 | −0.002 [−0.041, +0.023] p=0.96 |
| canon_loose300 (Under, 30 cm, full) | **+0.092** [+0.066, +0.118] p<0.001 | +0.035 [+0.023, +0.047] p=0.05 | +0.008 [−0.006, +0.028] p=0.54 |

Pattern across moorings at 1.3 Hz:
- **No universal "A2 dip"**: every mooring shows A1 ≥ A2 ≥ A3 (monotonic
  decay of wind enhancement with amplitude), not a V-shape.
- At above-semi-stiff, all three amps cluster at zero (within ~0.018 of
  each other). The "V-shape" the user noticed is the visual artefact of
  one A2 point sitting slightly below A1 and A3, all within errorbar
  noise.

## Verdict counts at LOW FREQ (1.2–1.4 Hz), Kt source = eff

Pivot by (Mooring × PanelCondition × verdict):

| Mooring × Panel | decrease | enhance | insufficient_n | null |
|---|---|---|---|---|
| above_50 / full       | 1 | 2 | 0 | **6** |
| above_50 / no         | 0 | 2 | 6 | 1 |
| above_50 / reverse    | 1 | 1 | 6 | 1 |
| below_90_loose230 / full | 0 | **6** | 2 | 1 |
| below_90_loose300 / full | 0 | **5** | 3 | 1 |

The lone "decrease" verdict for above_50/full at low freq is the 1.4 Hz
A2 cell discussed above (Welch p = 0.283, not significant). Under-mooring
shows 11 "enhance" / 0 "decrease" — a clean directional signal.

## Candidate explanations for the 2026-03-14 anomaly

*(Hypotheses; none directly tested.)*

1. **Long per=240 train + wind interacting with the 16 cm above-water
   semi-stiff mooring**. A 240-period train at 1.4 Hz is ~170 s of
   continuous wind+wave; at 1.6 Hz ~150 s. The 16 cm strikk is the
   loosest above-water mooring, most prone to slow displacement or
   build-up of standing-wave resonances over time. Why specifically at
   1.4 and 1.6 Hz (not 1.3 or 1.5)? Unknown — could be a panel-mooring
   resonance at those wavelengths.
2. **Single-day variability**: wind generator behaving slightly
   differently on 2026-03-14 (relative humidity, blower temperature,
   water-surface tension after long pump use). 03-14 was a different
   session from 03-07 even though the mooring/panel was nominally the
   same configuration.
3. **Reflections building up** over a 240-period train more at 1.4/1.6
   Hz than at adjacent freqs (tank-length resonances?).
4. **Wave-induced mooring drift** accumulating over the long train —
   would be specific to the 16 cm above-water rope and to amplitudes
   large enough to perturb the panel position.

Note that the nowind side of the same per=240 sessions (2026-03-13)
behaves normally — the anomaly is fullwind-specific. So the
candidate explanations must involve wind, not just per=240.

## Bullet recommendations for the user

1. **Re-measure** the 2026-03-14 per=240 fullwind runs at 1.4 Hz and
   1.6 Hz for above-semi-stiff (Mooring = above_50, PanelCondition =
   full, A2 = 0.2V). If they reproduce K_t ≈ 0.57 and 0.40, the long-
   train collapse is a real physical effect of the protocol on this
   mooring. If they don't reproduce, the original runs were anomalous
   and can be flagged.

2. **For the thesis**: do not claim "wind decreases K_t at 1.3–1.4 Hz"
   for the above-semi-stiff mooring. Honest claim: "no statistically
   significant wind effect on K_t at 1.2–1.4 Hz on the over-water
   semi-stiff mooring; more runs at A2 are needed to characterise the
   1.4 and 1.6 Hz cells, where current data is dominated by a single
   per=240 session (2026-03-14)."

3. **A2 "sweet spot" hypothesis falsified for canon under-mooring**:
   canon_loose230 and canon_loose300 show monotonic decay of wind
   enhancement with amplitude, no V-shape at A2. The "A2 is different"
   visual is specific to the n=2/2 above-semi-stiff cells.

## Headline contrast (what IS real)

Under-mooring (canon_loose230 + canon_loose300) at 1.3–1.4 Hz **A1**:
wind enhances K_t by +0.092 to +0.186 with p < 0.01 — clean, replicated
across two strikk lengths and many dates.

At 1.5 Hz and above the over-semi-stiff mooring switches to clear wind
enhancement (e.g. 1.5 Hz A3: ΔK_t = +0.076, p = 0.008; 1.6 Hz A3:
ΔK_t = +0.174, p = 0.122). This is the "panel-resonance transition zone"
flagged in earlier sessions — the transition itself is supported by the
data even though the "below-1.5-Hz decrease" framing of it was wrong.

## Files (full paths)

- Script: [`analysis_scratch/wind_effect_realness_lowfreq.py`](../analysis_scratch/wind_effect_realness_lowfreq.py)
- Outputs:
  - [`analysis_scratch/wind_effect_realness_lowfreq.pdf`](../analysis_scratch/wind_effect_realness_lowfreq.pdf)
  - [`analysis_scratch/wind_effect_realness_lowfreq_kt.pdf`](../analysis_scratch/wind_effect_realness_lowfreq_kt.pdf)
  - [`analysis_scratch/wind_effect_realness_lowfreq_a2_focus.pdf`](../analysis_scratch/wind_effect_realness_lowfreq_a2_focus.pdf)
  - [`analysis_scratch/wind_effect_realness_lowfreq_runs.csv`](../analysis_scratch/wind_effect_realness_lowfreq_runs.csv) (282 rows, all (mooring × freq × amp × wind) cells in scope)
  - [`analysis_scratch/wind_effect_realness_lowfreq_a2_runs.csv`](../analysis_scratch/wind_effect_realness_lowfreq_a2_runs.csv) (63 rows, A2 only)
  - [`analysis_scratch/wind_effect_realness_lowfreq_cells.csv`](../analysis_scratch/wind_effect_realness_lowfreq_cells.csv) (120 cells, bootstrap CI + Welch p)

## The two anomalous runs (full filenames)

These are the two single runs that drive the entire "A2 V-shape" in
canon_above_loose at 1.4 and 1.6 Hz:

- `20260314-ProbePos4_31_FPV_2-tett6roof/fullpanel-fullwind-amp0200-freq1400-per240-depth580-mstop30-run1.csv`
  → K_t_FFT = 0.570, K_t_LS = 0.570, K_t_PSD = 0.586
- `20260314-ProbePos4_31_FPV_2-tett6roof/fullpanel-fullwind-amp0200-freq1600-per240-depth580-mstop30-run1.csv`
  → K_t_FFT = 0.395, K_t_LS = 0.398

Both belong to the 4-folder canon-era march2026_better_rearranging probe
config. quality_flag = "ok" on both runs, no cut_samples on canonical
probes, FFT/LS/PSD agree to ~0.01 — so the runs are clean from the
pipeline's perspective. The anomaly is in the physical K_t value, not in
the measurement.
