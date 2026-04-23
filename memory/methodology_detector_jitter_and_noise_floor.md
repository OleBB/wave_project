---
name: Detector jitter and probe noise floor
description: Why upcrossing spacings inside the H&G window jitter under wind — quantitative link between probe noise floor, wave slope, and zero-crossing detection uncertainty
type: project
---

# Detector jitter is noise_floor / wave_slope

> **Update 2026-04-23, commit `b57d4a6`**. The H&G window end is now snapped
> to the 10th detected zero-upcrossing (was: fixed `start + 10·round(Fs/f)`).
> Observation B below — η ≠ 0 at window end — is **resolved by construction**
> for the nowind case and substantially reduced under fullwind. The remaining
> fullwind residual is the detector-jitter footprint on the 10th upcrossing
> itself (which no window choice can remove at the sample-grid level). The
> derivation below is retained for historical context; numbers after the update
> would differ, see the "Updated numbers (post-b57d4a6)" appendix.

**TL;DR.** The zero-upcrossing detector used to snap the H&G window has a
per-cycle timing uncertainty equal to `σ_signal / |dη/dt|_zero`, where
`σ_signal` is the noise amplitude on the probe at the crossing and
`|dη/dt|_zero = 2π·f·A` is the slope of the wave at zero. Under nowind,
probe noise (~0.1 mm) predicts ~0.2-sample jitter — below the irreducible
sample-grid quantization floor of ~0.7 samples, so nowind spacings look
like pure quantization. Under fullwind, wind-background amplitude at the
probe (~2–3 mm at the IN side) predicts ~4–6 samples of jitter, which is
what we observe. The pipeline metric (FFT amplitude) integrates over all
10 cycles and averages this jitter out, so the phenomenon is visible in
the time domain but invisible in the frequency domain.

Derived during the 2026-04-23 session triggered by the user's visual
observation that:
  (a) troughs before the yellow zoom band differ by ~1 between IN and OUT
      macros in `ch04_inspirational_nowind.pdf`;
  (b) the last wave inside the 10-period zoom does not end at η = 0 — a
      little in nowind, more in fullwind.

Both observations turn out to share an underlying mechanism with the
(already-documented) snap shifts and the (already-measured) probe noise
floor.


## 1 — The canon run and reference parameters

Data source (both runs): `PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange`.

- Nowind CSV: `fullpanel-nowind-amp0200-freq1400-per240-depth580-mstop30-run1.csv`
- Fullwind CSV: `fullpanel-fullwind-amp0200-freq1400-per240-depth580-mstop30-run1.csv`

Reference numbers:
- Sampling rate `Fs = 250` Hz
- Paddle frequency `f = 1.4` Hz
- True period in samples `T = Fs/f = 178.571`
- `samples_per_period = int(round(Fs/f)) = 179` (from `wave_detection.py:93`)
- Window length = `10 · samples_per_period = 1790` samples (pipeline
  preserves this on snap; only the start is snapped, the end rides along
  by the same offset)
- True 10 periods = `10 · 178.571 = 1785.71` samples
- **Window overshoot** = `1790 − 1785.71 = 4.29 samples = 0.024 T` past
  the 10th upcrossing
- Wave amplitude at 1.4 Hz, 0.2 V paddle: `A ≈ 15` mm (IN, nowind)
- Slope of the paddle wave at zero: `|dη/dt|_zero = 2π·f·A = 131.9 mm/s`


## 2 — Observation A: the ~1-trough IN/OUT asymmetry

Source of numbers: `output/timeseries_exploration/insp_snap_diagnostic_{nowind,fullwind}.pdf`
+ console prints from `analysis_scratch/inspirational_timeseries_snap_diagnostic.py`.

Snap shifts for this run (from meta.json: `Probe {pos} hg_snap_shift`
column in samples, divided by `samples_per_period` for periods):

| Wind | Probe | Shift [samples] | Shift [T] |
|---|---|---|---|
| nowind   | 9373/170 (IN)  | +80 | **+0.448** |
| nowind   | 12400/250 (OUT)| −63 | **−0.353** |
| nowind   | **net IN − OUT** |   | **+0.801** |
| fullwind | 9373/170 (IN)  | +46 | +0.258 |
| fullwind | 12400/250 (OUT)| −89 | −0.498 |
| fullwind | **net IN − OUT** |   | **+0.756** |

The net IN − OUT shift is ~0.8 T in both conditions (0.80 nowind vs
0.76 fullwind, within single-run noise). One wave period ≈ one trough.

**Observation A**: the visible ~1-trough asymmetry in trough count
between IN and OUT macros is a direct consequence of the net 0.8-T
snap shift difference between the two probes' H&G windows.

*Candidate explanation (hypothesis, see `hg_snap_shift_diagnostic.md`
§"Candidate explanations")*: the opposite-signed shifts at IN and OUT
are consistent with a per-probe-position effective phase reference
(H5: near-panel reflection at OUT creates a local standing-wave phase
offset; other hypotheses listed). Parallel-probe agreement (Sanity
check in the same .md) rules out any per-probe hardware lag at the
≤0.01 T level.


## 3 — Observation B: η at window end ≠ 0

η values at the snapped window endpoints, read from the processed
`eta_` column of the time series (`processed_dfs[path][col]`):

| Wind | Probe | η at start [mm] | η at end [mm] |
|---|---|---|---|
| nowind   | 9373/170  | −0.78 | **−3.01** |
| nowind   | 12400/250 | −0.27 | **−1.56** |
| fullwind | 9373/170  | +0.60 | **−5.06** |
| fullwind | 12400/250 | −0.43 | **−4.55** |

The start values are all near zero (the snap landed on a raw-signal
upcrossing, which is an eta-downcrossing given the inverted ULS
convention: raw distance decreases when water rises). The end values
are clearly non-zero, consistent with η crossing through zero going
downward somewhere before the window end, then proceeding into the
downstroke.

Converting to phase-past-last-upcrossing via `arcsin(|η_end|/A)/(2π)`:

| Wind | Probe | Implied drift past 10th UC [T] |
|---|---|---|
| nowind   | 9373/170  | **0.032** |
| nowind   | 12400/250 | **0.017** |
| fullwind | 9373/170  | **0.054** |
| fullwind | 12400/250 | **0.049** |

The nowind OUT value (0.017 T) matches the pure-quantization prediction
(0.024 T) within the measurement precision. The other three have
**extra drift on top of the quantization baseline** — more under fullwind
(~0.05 T) than nowind (~0.03 T).

**Observation B**: the 10-period H&G window does not end at a zero
crossing. Nowind: ~0.02–0.03 T overshoot. Fullwind: ~0.05 T overshoot.


## 4 — Mechanism check: upcrossing spacing within the window

If the wave frequency drifts mid-window, inter-upcrossing spacings
should trend across the 10 periods. If only the first upcrossing is
biased, spacings should be uniform. If the detector jitters on a noisy
signal, spacings should be random but zero-mean.

Replicated the pipeline upcrossing detection (rolling-mean smoothing
with wind-dependent window size per `wavescripts/constants.py` — 1 for
nowind, 15 for fullwind) and measured all inter-upcrossing spacings
inside the H&G snapped window.

Reference: for a pure 1.4 Hz sinusoid, integer-rounding of continuous
upcrossing times produces spacings that alternate between 178 and 179
samples (true T = 178.571), with an irreducible ~0.7-sample std from
sample-grid quantization alone.

| Wind | Probe | mean [samples] | std [samples] | range | in {178,179} | other |
|---|---|---|---|---|---|---|
| nowind   | 9373/170  | 178.56 | **0.83** | 177–180 | 7/9  | 2 (177, 180) |
| nowind   | 12400/250 | 178.60 | **1.02** | 177–180 | 6/10 | 4 (177×2, 180×2) |
| fullwind | 9373/170  | 178.00 | **3.94** | 172–185 | 4/9  | 5 (172, 173, 176, 183, 185) |
| fullwind | 12400/250 | 178.22 | **2.90** | 175–184 | 3/9  | 6 (175×2, 176, 177, 182, 184) |

Empirical frequency `Fs / mean_spacing`:

| Wind | Probe | Empirical f [Hz] |
|---|---|---|
| nowind   | 9373/170  | 1.4001 |
| nowind   | 12400/250 | 1.3998 |
| fullwind | 9373/170  | 1.4045 |
| fullwind | 12400/250 | 1.4027 |

**Observation C**: nowind spacings have std ~0.8–1.0 samples and cluster
in {177, 178, 179, 180} — consistent with pure sample-grid quantization.
Mean = 178.57 matches the true period exactly.

**Observation D**: fullwind spacings have std ~3–4 samples, with
individual cycles spanning 172–185. Mean still matches 1.4 Hz within
~0.3 %. Spacings are not trending — they are scattered around the mean.

**Observation E**: fullwind empirical frequency is slightly higher than
the set 1.4 Hz (by 0.3–0.5 %). Small but non-zero.

Observations C and D together rule out "wave frequency shifts mid-window"
as the driver of Observation B's extra drift — the mean spacing is
frequency-correct in both conditions. What changes under wind is the
**variance** of the detector placement of each zero crossing.


## 5 — Linking detector jitter to the probe noise floor

For a sinusoid `η(t) = A·sin(2πft)` corrupted by additive Gaussian noise
of standard deviation `σ_signal`, the detected zero-crossing time has
uncertainty

    σ_t = σ_signal / |dη/dt|_zero = σ_signal / (2π·f·A)

This is the standard result for level-crossing detection under noise.

### Stillwater noise floor (source: `output/FIGURES/ch04_probe_noise_floor_group3.pdf`)

Mean stillwater "støyamplitude (95 %)" per probe for h=100 mm, cond4
(our canon setup):

| Probe | Mean noise ≈ 2σ [mm] | Implied σ_signal [mm] |
|---|---|---|
| 9373/170  | 0.10 | ~0.05 |
| 12400/250 | 0.09 | ~0.045 |
| 9373/340  | 0.09 | ~0.045 |
| 8804/250  | 0.06 | ~0.03 |

### Predicted jitter per cycle

At `f = 1.4` Hz, `A = 15` mm, `|dη/dt|_zero = 131.9 mm/s`:

| Condition | σ_signal at probe [mm] | σ_t (s) | σ_t (samples @ 250 Hz) |
|---|---|---|---|
| nowind stillwater floor | 0.05 | 0.00038 | **~0.10** |
| fullwind wind-background estimate | 2.5 | 0.019 | **~4.7** |

The fullwind σ_signal of ~2.5 mm is a working estimate based on CLAUDE.md
§16 ("at the IN probe, wind-background signal is roughly 1-3 mm range"
per the probe-height discussion). It has not been directly read off a
figure here; the matching predicted jitter (4.7 samples vs observed 3.94)
is consistent with this estimate but does not itself measure the wind
background.

### Comparison to measured jitter

| Condition | Predicted σ_t [samples] | Quantization floor [samples] | Observed [samples] | Interpretation |
|---|---|---|---|---|
| nowind  | 0.10 | 0.7 | 0.83–1.02 | **quantization-limited**; noise floor ≪ quantization |
| fullwind | 4.7  | 0.7 | 2.90–3.94 | **noise-limited**; wind background drives the jitter |

**Observation F**: nowind detector jitter matches the sample-grid
quantization floor within measurement noise. The stillwater noise floor
(0.09–0.10 mm at IN) is too small to project past quantization — it
contributes ~0.1 samples, lost in the 0.7-sample quantization band.

**Observation G**: fullwind detector jitter (~3–4 samples) is consistent
with `σ_t ≈ σ_wind_bg / (2πfA)` for `σ_wind_bg ≈ 2` mm. The actual
fullwind σ_signal at the IN probe should be looked up from the
wind-background measurement (CH04 §4-1) to close this loop with a
direct number rather than an estimate.


## 6 — Consequence for the η-at-end observation

Returning to Observation B. The drift past the last upcrossing has
two components:

1. **Window-length quantization** (constant, every run at 1.4 Hz):
   `0.024 T` from `int(round(Fs/f))` rounding up from 178.571 to 179.
2. **Detector jitter accumulated across the window**: if each of the
   10 upcrossings is placed with independent σ_t of ~4 samples under
   fullwind, the last upcrossing's detection is still ±4 samples on
   that specific cycle — it does not random-walk. But the cycles
   *inside* the window each shift by ~4 samples, and the average wave
   phase at the window end relative to the 10th crossing is
   dominated by wherever the 10th crossing lands relative to the
   true 10T position.

Rough decomposition (for this specific run):

| Wind | Probe | 0.024 baseline | Observed drift | Extra attributable to detector | Fits σ_t budget? |
|---|---|---|---|---|---|
| nowind   | IN  | 0.024 | 0.032 | 0.008 | yes (σ_t ~ 0.01 T = 0.17 ms = 0.04 samples) |
| nowind   | OUT | 0.024 | 0.017 | −0.007 | yes (within quantization noise) |
| fullwind | IN  | 0.024 | 0.054 | 0.030 | yes (σ_t ~ 0.03 T ≈ 5 samples) |
| fullwind | OUT | 0.024 | 0.049 | 0.025 | yes |

The fullwind "extra drift" of 0.025–0.030 T converts to ~4–5 samples,
matching the observed per-cycle σ_t of 3–4 samples. Consistent.


## 7 — What this means for the pipeline

Zero impact on the reported FFT amplitude:

- The FFT integrates over all 10 cycles. Detector jitter that scatters
  individual upcrossings around their true positions is zero-mean and
  averages out.
- Spectral leakage from the 0.024 T window overshoot: paddle frequency
  falls on bin `1.4·1790/250 = 10.02`, a 0.02-bin offset from integer.
  Sinc factor `sinc(0.02) = 0.9993`, so ~0.07 % attenuation.
- Cross-method agreement (CH04 §4h, 4-method comparison across 128
  nowind measurements, < 0.04 % median) corroborates that this drift
  is invisible in the frequency domain.

Direct impact on the time-domain visual:

- η at the window end is not zero. ~0.02–0.05 T worth of "incomplete
  last period" is visible. This is a figure-rendering artifact, not a
  measurement artifact.
- The ~1-trough difference in trough count between IN and OUT macros
  (before the yellow zoom band starts) is the snap-shift net asymmetry
  (Observation A), which is unrelated to detector jitter — it's
  per-probe effective phase reference.


## 8 — Open questions / follow-ups

- The fullwind empirical frequency bias (Observation E, +0.3–0.5 %)
  is small but non-zero and consistent between probes. *Candidate
  explanations (not tested here)*: wind-induced surface current Doppler,
  wind-wave coupling modulating the effective period at the probe, or a
  detector bias that systematically places upcrossings slightly early
  under wind-chop-modulated baselines. Worth a dedicated check across
  all fullwind runs.
- The wind-background σ_signal at the IN probe under fullwind should
  be read from the CH04 §4-1 figure (wind characterisation) rather than
  estimated at "2–3 mm". That would close Observation G with a direct
  number and let the prediction `σ_t = σ_wind_bg / (2πfA)` become an
  assertion rather than a consistency check.
- Cheap pipeline improvement: use float `samples_per_period` and round
  the window endpoints separately (or snap the END to an upcrossing
  too). Would remove the 0.024 T window-length quantization baseline.
  Cosmetic only — no FFT impact.


## Updated numbers (post-`b57d4a6`, UC-snap end)

After the pipeline change, the canon runs show:

| Run | Probe | Window length | η at start | η at end |
|---|---|---|---|---|
| nowind   | 9373/170  | 1786 samples (10.0016 T) | −0.78 | **−0.79 mm** |
| nowind   | 12400/250 | 1787 samples (10.0072 T) | −0.27 | **−0.44 mm** |
| fullwind | 9373/170  | 1781 samples (9.9736 T)  | +0.60 | **−3.43 mm** |
| fullwind | 12400/250 | 1782 samples (9.9792 T)  | −0.43 | **−1.12 mm** |

Nowind: η_start and η_end now match (both near the raw-upcrossing level).
Observation B is resolved for the nowind case.

Fullwind: η_end is substantially smaller than the fixed-length values
(−3.4/−1.1 mm vs −5.1/−4.5 mm before). The residual is the detector-jitter
footprint on the 10th upcrossing itself — each upcrossing has ±4-sample
placement noise under wind (Section 4), so the "10th UC" lands ±4 samples
off the true 10T position. This irreducible uncertainty translates to
~0.02 T of residual drift at the window end, matching the observed
fullwind η_end magnitudes.

Amplitude agreement tightens:

| Probe | Condition | FFT [mm] | LS [mm] | Δ (%) |
|---|---|---|---|---|
| 9373/170  | nowind   | 15.3308 | 15.3381 | 0.05 |
| 12400/250 | nowind   | 10.6635 | 10.6705 | 0.07 |
| 9373/170  | fullwind | 16.0708 | 16.0549 | 0.10 |
| 12400/250 | fullwind | 11.7289 | 11.7189 | 0.09 |

Max Δ = 0.10 % (was up to 0.4 % in CH04 §4h before UC-snap end).

Sections 1–7 above describe the pre-change pipeline. The physics
(detector jitter = σ_signal / wave_slope, noise floor limits, etc.) is
unchanged; only the window-length quantization contribution (0.024 T
baseline) has been eliminated by the pipeline fix.

### Population-level validation (commit `eefba87`)

After a full --force-recompute across all 25 datasets and the ±0.5 T
sanity guard (commit `eefba87`), the window-length distribution across
1012 thesis-scope probe-runs (1.3–1.7 Hz, fullpanel, quality_flag=ok):

| Probe | wind | n | min [samples] | max | median | std |
|---|---|---|---|---|---|---|
| 9373/170  | no   | 133 | −11 | +7  | 0  | 3.6 |
| 9373/170  | full | 120 | −13 | +41 | 0  | 7.8 |
| 9373/340  | no   | 133 | −11 | +9  | +1 | 4.1 |
| 9373/340  | full | 120 | −72 | +26 | 0  | 10.3 |
| 12400/250 | no   | 133 | −11 | +7  | −2 | 4.0 |
| 12400/250 | full | 120 | −17 | +22 | +2 | 6.8 |
| 8804/250  | no   | 133 | −8  | +18 | 0  | 4.3 |
| 8804/250  | full | 120 | −15 | +20 | 0  | 6.0 |

Shifts measured as `(Computed end − Computed start) − 10·samples_per_period`.

Nowind: std 3.6–4.3 samples = ~0.02 T, dominated by sample-grid
quantization (each UC snaps to the nearest integer sample, accumulated
over 10 crossings).

Fullwind: std 6–10 samples = ~0.04 T, matching the accumulated detector
jitter budget `√10·(σ_signal/(2πfA))` for `σ_signal ≈ 2` mm at 1.4 Hz.

**0/1012 runs exceeded the ±0.5 T guard** — all runs either got a clean
UC-snap end or correctly fell back to fixed length. The pre-guard
version (commit `b57d4a6`) had 38/120 fullwind 9373/170 runs with
|shift| > 100 samples, concentrated at the pathological 1.3 Hz × 0.1 V
× fullwind × exposed-probe combination where wind chop creates spurious
near-zero crossings. Guard resolves this.


## For another agent wanting to re-derive these numbers

- H&G window snap code: `wavescripts/wave_detection.py:165–209`.
- Sample-per-period quantization: `wavescripts/wave_detection.py:93`
  (`samples_per_period = int(round(Fs/f))`).
- Smoothing windows: `wavescripts/constants.py:70–73`.
- Single-run visual: `analysis_scratch/inspirational_timeseries_snap_diagnostic.py`
  → writes `output/timeseries_exploration/insp_snap_diagnostic_{nowind,fullwind}.pdf`.
- Population snap-shift stats: `analysis_scratch/hg_snap_shift_diagnostic.{py,md}`.
- Stillwater noise floor: `output/FIGURES/ch04_probe_noise_floor_group3.pdf`,
  generated by `analysis_scratch/probe_height_figure.py` (or its CH04 §1 sibling).
- The upcrossing-spacing diagnostic is an inline one-off from the session;
  to reproduce: load the processed run, re-run the rolling-mean smoother,
  compute `np.where((~above[:-1]) & above[1:])[0] + 1`, intersect with the
  H&G snapped window, `np.diff`.
