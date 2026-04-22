# FFT amplitude extraction — method comparison

Generated: 2026-04-22

**Methods tested** (all try to extract paddle-wave amplitude at a known
target frequency from a windowed probe signal):

| method        | what it does                                                                                                              |
|---------------|---------------------------------------------------------------------------------------------------------------------------|
| `nearest_bin` | current pipeline. `np.fft.fft`, pick nearest bin to `f_paddle` in ±0.1 Hz. Sinc-attenuated when `f_paddle` is off-bin.    |
| `parabolic`   | 3-bin quadratic interpolation (generic Gaussian-peak formula, NOT sinc-corrected). Recovers only part of the sinc loss.  |
| `goertzel`    | single-frequency DFT evaluated at *exactly* `f_paddle`. Bypasses the bin grid.                                           |
| `ls_fit`      | least-squares fit of `[DC, cos(ωt), sin(ωt), cos(2ωt), sin(2ωt)]`. Same as goertzel + explicit Stokes-2f protection.     |

## §1 Synthetic test — confirms the theoretical bias

Pure tone, A = 1.0 mm, frequency swept ±0.6 bin widths around f_nominal =
1.3 Hz. Window N=1923 (10 periods), fs=250 Hz → bin spacing Δf = 0.130 Hz.
All methods are told `target_freq = f_true` (i.e. we know the paddle
frequency exactly).

| scenario    | method        | max \|err\| % | mean \|err\| % |
|-------------|---------------|---------------|----------------|
| clean       | nearest_bin   | **38.01**     | 15.83          |
| clean       | parabolic     | 31.33         | 14.68          |
| clean       | goertzel      | 0.83          | 0.50           |
| clean       | ls_fit        | **0.00**      | 0.00           |
| white_noise | nearest_bin   | 37.13         | 15.92          |
| white_noise | parabolic     | 31.73         | 14.79          |
| white_noise | goertzel      | 2.95          | 0.88           |
| white_noise | ls_fit        | 2.74          | 0.77           |
| wind_like   | nearest_bin   | 38.02         | 15.85          |
| wind_like   | parabolic     | 31.24         | 14.70          |
| wind_like   | goertzel      | 2.00          | 0.60           |
| wind_like   | ls_fit        | 1.79          | 0.45           |

**Observations** (see figure panels a–c):

- `nearest_bin` shows the classic sawtooth error pattern: 0 % at bin
  centres, rising to ~38 % at ±½ bin offset. Matches sinc theory exactly.
- `parabolic` (generic Gaussian formula) recovers only ~6 percentage
  points of the loss. The sinc main-lobe isn't parabolic at the
  extremes — a sinc-correct Jacobsen estimator would do better, but
  even that won't reach Goertzel/ls_fit accuracy.
- `goertzel` and `ls_fit` are flat across the whole sweep. In clean
  conditions they are numerically exact (ls_fit) or exact to machine
  noise (goertzel). Under noise they pick up a small stochastic
  scatter (~0.5 % RMS) that does not depend on bin alignment.

## §2 Real-data cross-check

Dataset: nowind wave runs in the two canonical March-2026 lowrange folders
(20260326, 20260327). Per-probe measurements: **n = 128** across 4
probes, 32 runs, 5 frequencies (1.3–1.7 Hz).

Using `ls_fit` as reference. Relative disagreement = (A_method − A_ls_fit) / A_ls_fit.

| method        | mean %   | std %  | max \|%\| | median \|%\| |
|---------------|----------|--------|-----------|--------------|
| `nearest_bin` | −0.049   | 0.102  | 0.40      | 0.04         |
| `parabolic`   | −0.048   | 0.102  | 0.39      | 0.04         |
| `goertzel`    | +0.015   | 0.132  | 0.33      | 0.08         |
| `ls_fit`      | 0.000    | 0.000  | 0.00      | 0.00         |

**Observation**: all four methods agree to within 0.4 % on real data.
The 38 % synthetic bias is not present here.

### Observed bin alignment in the current cache

The pipeline window length is `N = round(10·fs/f_paddle) ± 1`. At
fs = 250 Hz the bin grid is `k·fs/N`, and the bin at `k = 10` sits at
`10·fs/N`. Measured offsets from the real-data CSV:

| f_paddle | N samples | bin 10 freq | offset from f_paddle |
|----------|-----------|-------------|----------------------|
| 1.3 Hz   | 1921      | 1.3014 Hz   | **−0.011 bins**      |
| 1.4 Hz   | 1791      | 1.3959 Hz   | **+0.029 bins**      |
| 1.5 Hz   | 1671      | 1.4961 Hz   | **+0.026 bins**      |
| 1.6 Hz   | 1561      | 1.6015 Hz   | **−0.009 bins**      |
| 1.7 Hz   | 1471      | 1.6995 Hz   | **+0.003 bins**      |

Worst real-data bin offset is 0.03 bins; the sinc attenuation at that
offset is `|sinc(π·0.03)| = 0.9985` (math, not inference) → implied
theoretical loss ~0.15 %.

**Candidate explanation (hypothesis)**: the 10-period window makes bin 10
land on f_paddle. Whether this was an intended consequence of the H&G
convention or an incidental result of the wave-physics choice is not
documented. The observation stands independently of the explanation.

## Cross-reference: the 40 % number in `methodology_fft_peak_bin_bias.md`

That memo's 40 % number is a theoretical rectangular-window worst case
for a non-integer-cycle window. This sweep did not observe it in the
current cache.

Note (added after reviewer pushback 2026-04-22): reading the pre-H&G
`find_wave_range` code shows both window endpoints were snapped to
zero-upcrossings, producing integer-cycle windows in that era as well.
A direct re-run of the 4-method comparison on pre-H&G cache would be
needed to observe whether the 40 % appears there; without that data the
status of the pre-H&G realised bias is unobserved.

## Recommendation

**Observed**: on 128 nowind per-probe measurements, the 4 methods agree
to 0.4 % max / 0.04 % median. On this basis the current
`compute_amplitudes_from_fft` does not require replacement to hit the
0.25 mm precision target. **Hypothesis**: replacing with `ls_fit` would
buy at most ~0.15 % on absolute amplitude (from eliminating the
theoretical sinc attenuation); this has not been measured directly.

**Consider an optional upgrade path**: add an `ls_fit`-based function
(~15 lines) that returns both the fundamental amplitude AND the Stokes-2f
amplitude in one pass. Benefits:

1. Stokes-2f per probe in `meta.json` as a first-class column (useful
   for wave-steepness and non-linearity analysis, currently requires
   the sliding-FFT scripts in `analysis_scratch/`).
2. A direct residual signal `y − model` per probe — paddle-free,
   ready-made for wind-wave characterisation.
3. Bin-grid-independence: future analyses that use non-integer-period
   windows (e.g. looking at only the first 5 periods, or the last 3)
   stay unbiased automatically.

This upgrade is **not urgent for the OUT/IN thesis result** but would
strengthen any CH04 methodology section that reports individual probe
amplitudes in absolute mm.

## Thesis methodology text (draft, for CH04)

> The paddle-wave amplitude at each probe is computed from the Fourier
> spectrum of the H&G-windowed signal. The window spans ten wave
> periods (N = round(10·fs/f_paddle) samples) starting at [50T, 60T]
> after wavemaker onset, anchored at the most-downstream probe
> (r = 12.4 m) and probe-shifted earlier by the group-velocity travel
> time. The FFT bin grid has spacing Δf = fs/N, and for this choice
> of N we observe that the bin at k=10 lies within 0.03 bin widths of
> f_paddle across the thesis frequency range (1.3–1.7 Hz). The
> corresponding sinc-leakage attenuation, |sinc(π·0.03)| = 0.9985,
> implies ~0.15 % amplitude loss at that offset. Cross-validation
> against a bin-grid-independent least-squares sinusoid fit at exactly
> f_paddle, on 128 per-probe measurements from the nowind canonical
> dataset, gave a median relative disagreement of 0.04 % and maximum
> 0.40 %. Individual absolute amplitudes are reported as peak-bin
> values throughout. The OUT/IN ratio is observed to be robust to
> method choice: on 367 independent runs the mean |Δ(OUT/IN)|/OUT/IN
> between nearest-bin and parabolic-interpolated amplitudes is 0.46 %.

## See also

- Figure: `analysis_scratch/fft_method_comparison.png`
- Synthetic CSV: `analysis_scratch/fft_method_comparison_synth.csv`
- Real-data CSV: `analysis_scratch/fft_method_comparison_real.csv`
- Earlier (pre-H&G) findings: `memory/methodology_fft_peak_bin_bias.md`
- Related: `analysis_scratch/sliding_afft_fullwind_sweep_findings.md`
  (pre-H&G, demonstrated the bias in real data)
- Related: `analysis_scratch/fft_peak_bias_outin_impact_findings.md`
  (pre-H&G, OUT/IN ratio robustness)
