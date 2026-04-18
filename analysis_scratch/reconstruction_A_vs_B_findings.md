# Reconstruction A vs B — is method A safe for wind separation?

**Date**: 2026-04-18
**Script**: `analysis_scratch/reconstruction_A_vs_B.py`
**Figure**: `analysis_scratch/reconstruction_A_vs_B.pdf`
**Per-run data**: `analysis_scratch/reconstruction_A_vs_B_summary.csv`
**Related**: `methodology_fft_peak_bin_bias.md`, CLAUDE.md §6 ("FFT-based OUT/IN under fullwind — two competing biases"), `wind_psd_shape_cond1_vs_cond4_findings.md`.

## Question

`plot_reconstructed` (and the FFT amplitude metric `A_FFT = |FFT[peak_bin]| · 2/N` that drives `OUT/IN (FFT)`) uses the single paddle-frequency peak bin. A finite-duration wavetrain's energy is not a δ at that bin — it spreads over neighbouring bins (sinc-leakage + coherence loss + wind contamination in the paddle band). Method A under-reports the true paddle amplitude.

For the thesis, the concrete worry is:

> If method A leaks paddle-linked energy into what we call the **wind + noise residual**, then our characterisation of wind is contaminated by mis-attributed paddle energy. That would make A unsafe as the basis for separating wind.

This script tests that worry empirically.

## Data

- `meta_results`, full panel, `quality_flag in {ok, probe_malfunction_secondary}`: 101 wave runs.
- Probes IN (9373/170) and OUT (12400/250) reconstructed on each run ⇒ 202 (run × probe) comparisons.

## Method

For each (run × probe), operate on the cached FFT spectrum:

- **A (peak-bin)**: zero all bins except the exact paddle peak + its real-signal mirror. IFFT → `signal_A`.
- **B (±0.05 Hz band)**: zero all bins except those within ±0.05 Hz of the paddle peak (both positive and negative-frequency sides). IFFT → `signal_B`. The band width matches the 0.1 Hz window in `compute_amplitudes_from_fft`, so B integrates exactly the band over which A peak-picks.

Residuals: `residual_X = signal_full − signal_X`. Wind-band energy: Welch PSD of the residual, integrated over **2–6 Hz** (the wind-wave band documented in CH04 §3b probe-height characterisation).

Reported per (run × probe):

- Amplitudes: `amp_X_rms = √2 · std(signal_X)` (mm), and the ratio `amp_B / amp_A` as a "how much more wave would we have if we integrated the band".
- Wind-band energy: `E_wind_A`, `E_wind_B`, and `(E_wind_A − E_wind_B) / E_wind_B`.

## Headline result

### Wind-band (2–6 Hz) residual energy — the safety test

| probe      | wind | (A−B)/B median | (A−B)/B mean | std | min | max | n |
|------------|------|---------------:|-------------:|----:|----:|----:|--:|
| 12400/250  | full | 0.0000 | −0.0000 | 0.0000 | −0.0001 | 0.0000 | 70 |
| 12400/250  | no   | 0.0000 | −0.0000 | 0.0000 | −0.0000 | 0.0000 | 31 |
| 9373/170   | full | 0.0000 |  0.0000 | 0.0000 | −0.0000 | 0.0000 | 70 |
| 9373/170   | no   | 0.0000 |  0.0000 | 0.0000 |  0.0000 | 0.0000 | 31 |

**(A−B)/B is zero to five decimal places across every single run.** The two residuals carry **identical** wind-band energy. This is not approximation error; it is structural: the paddle band (±0.05 Hz around 1.3–1.6 Hz) and the wind band (2–6 Hz) are disjoint, so whatever extra energy method B pulls out of the residual lives entirely inside the paddle band. The wind band stays clean under either reconstruction.

### Paddle-band amplitude (for context, not the primary result)

**Full dataset (all freqs, full-panel wave runs):**

| probe      | wind | amp_B/amp_A median | mean | std | max | n |
|------------|------|-------------------:|-----:|----:|----:|--:|
| 12400/250  | full | 1.000 | 1.001 | 0.008 | 1.063 | 70 |
| 12400/250  | no   | 1.000 | 1.000 | 0.000 | 1.002 | 31 |
| 9373/170   | full | 1.000 | 1.008 | 0.053 | 1.443 | 70 |
| 9373/170   | no   | 1.000 | 1.000 | 0.000 | 1.001 | 31 |

**Restricted to thesis scope (1.3–1.6 Hz):**

| probe      | wind | median | mean | std | max | n |
|------------|------|-------:|-----:|----:|----:|--:|
| 9373/170   | full | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 48 |
| 9373/170   | no   | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 30 |

**Inside thesis scope, A and B agree at every single run to four decimal places.** No sinc-leakage, no paddle-band wind contamination, nothing.

The full-dataset 1.443 max comes from a **single out-of-scope outlier run** at 0.8 Hz / 0.2 V / per40 (`fullpanel-fullwind-amp0200-freq0800-per40-run1.csv`, 20260327). Investigation: the FFT at the IN probe has two peaks of near-equal magnitude in adjacent bins — `|FFT|=26110` at 0.782 Hz and `|FFT|=26240` at 0.818 Hz. This is a **split paddle peak**, not wind energy bleeding into the paddle band. Accompanying red flags: `IN wave_stability = 0.745` (vs 0.953 at OUT on the same run), `IN period_amplitude_cv = 0.453` (vs 0.037 at OUT). Time-domain amplitude 18.4 mm, PSD band-integrated 11.6 mm, peak-bin FFT 7.5 mm — disagree by factors of 2, showing the IN signal is not a coherent monochromatic wave on this run. Likely mechanism: the short per40 record (50 s) at 0.8 Hz caught a paddle-frequency drift or wind-induced coherence loss. This combination never happens in thesis-scope per240 runs (300 s records, 6× finer bin grid).

`quality_flag = "ok"` lets this run through the current filter, but a stability-based filter (e.g. `IN wave_stability > 0.9`) would catch it. Not a concern for the thesis but worth logging.

Nowind runs across the whole dataset: A ≡ B (max ratio 1.001). No sinc-leakage in this data — paddle lands on a bin centre.

## Demo figure walk-through (`reconstruction_A_vs_B.pdf`)

Representative (nowind, fullwind) pair at 1.4 Hz, 0.20 V, full panel, under9Mooring30. Rows 1/2 are nowind, rows 3/4 are fullwind. Left column IN probe, right column OUT probe.

- **Top of each row-pair** (time series): raw η overlaid with `signal_A` (solid blue) and `signal_B` (dashed red). Nowind: A and B overlap exactly. Fullwind: still visually indistinguishable at this run — paddle frequency lands close to a bin centre so A ≈ B here. The A-band/A-peak maximum of 1.44 happens at different runs in the tail.
- **Bottom of each row-pair** (residual PSDs): shaded blue = paddle band ±0.05 Hz, shaded orange = wind band 2–6 Hz. The blue-line and red-dashed-line residual PSDs overlap *everywhere outside the paddle band*. Within the paddle band they differ (B has zero there by construction; A has whatever the band's sinc-leakage + wind contamination amounts to). The "(A−B)/B = +0.00%" in each subtitle is the integrated wind-band fraction — exactly zero to two decimal places.
- **IN / fullwind residual PSD** (bottom-left) shows dramatically more energy in 2–6 Hz than the nowind row above it (`E_A = 5.83` vs `0.42`): this is the real wind-wave peak at ~3.5–4 Hz. A and B report the same number for it.
- **OUT / fullwind residual** (bottom-right) shows only a small wind-band bump (`E_B = 0.46` vs nowind `0.13`), consistent with the panel sheltering the OUT probe from wind — same finding as the probe-height / wind-background work.

## Conclusion — is A safe?

**Yes, for wind characterisation.** The paddle-band leakage that method A misses stays inside the paddle band. No paddle-linked energy is bleeding into 2–6 Hz where wind lives. Any wind-separation analysis (residual PSDs, wind-band integrals, wind-vs-paddle energy budgets) gives the same answer under A as under B.

Method A is also **metric-faithful** — the reconstructed waveform's peak-to-peak amplitude equals the `A_FFT` number reported in CH05. A reader can measure the wave with a ruler and recover the OUT/IN ratio. Keep A for CH05.

## What the result does NOT say

- **Method A is, in principle, not unbiased as a paddle amplitude estimator.** The 44 % tail in the full dataset shows that the mechanism exists — but it requires a split paddle peak (short records + paddle drift or coherence loss). In the thesis scope (1.3–1.6 Hz per240), this mechanism does not trigger: `amp_B / amp_A = 1.0000` at every run. The broader peak-bin bias documented in `methodology_fft_peak_bin_bias.md` (up to 40 % from inter-bin sinc-attenuation) is real in principle but also not observable in this thesis subset — paddle frequencies land on or near bin centres on per240 records.
- The `OUT/IN (FFT)` *ratio* is still robust because the same under-reporting bias acts on IN and OUT (per `methodology_fft_peak_bin_bias.md`, mean |Δ|/ratio = 0.46% across 367 runs). The scatter above is for absolute amplitudes per probe, not ratios.

## Next step — the CH04 §x methodology figure

The demo figure here is the core of what needs to go into the methodology chapter. Suggested adaptation for thesis-grade:

1. Pick **one** fullwind run where the `amp_B/amp_A` ratio is near the 95th percentile (not median), so the gap between A and B is visible. The median run shows nothing interesting because A ≈ B there. A representative-of-the-bias run teaches the reader something.
2. Two rows only — IN and OUT, both at fullwind. Skip nowind (nothing to show).
3. In the residual PSD panels, annotate "paddle-band difference A vs B = X %, wind-band difference = +0.00 %" as the punch-line. That is the whole point: A and B differ inside the narrow paddle band and agree everywhere else.
4. Caption cites the dataset-wide statistic `(A−B)/B = 0.0000 ± 0.0000` across 202 (run × probe) comparisons — that is the validation number.

The figure then supports a paragraph in the methodology chapter that says:

> We use the peak-bin amplitude `A_FFT` as the transmission metric. Although this under-reports the true paddle amplitude in individual runs (up to 44 % at IN probes under wind), the under-reported energy stays within ±0.05 Hz of the paddle frequency. A band-integrated reconstruction (method B) produces residuals identical to the peak-bin reconstruction (method A) in the 2–6 Hz wind-wave band, so any wind characterisation based on the A residual is equivalent to one based on the B residual.
