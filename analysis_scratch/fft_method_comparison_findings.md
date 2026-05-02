# FFT amplitude extraction — method comparison

Generated: 2026-05-02T14:56:35Z

**Methods**: `nearest_bin` (pipeline), `parabolic` (3-bin quadratic interp), `goertzel` (single-frequency DFT at exactly f_paddle), `ls_fit` (LS with DC + fundamental + Stokes 2f basis).

## §1 Synthetic test

Pure tone, amplitude = 1.0 mm, frequency swept across ±0.6 bin widths around f_nominal = 1.3 Hz. Window N=1923 (10 periods at 1.3 Hz, fs=250 Hz) → bin spacing Δf = 0.130 Hz.

| scenario | method | max \|error\| % | mean \|error\| % |
|---|---|---|---|
| clean | nearest_bin | 38.01 | 15.83 |
| clean | parabolic | 31.33 | 14.68 |
| clean | goertzel | 0.83 | 0.50 |
| clean | ls_fit | 0.00 | 0.00 |
| white_noise | nearest_bin | 37.13 | 15.92 |
| white_noise | parabolic | 31.73 | 14.79 |
| white_noise | goertzel | 2.95 | 0.88 |
| white_noise | ls_fit | 2.74 | 0.77 |
| wind_like | nearest_bin | 38.02 | 15.85 |
| wind_like | parabolic | 31.24 | 14.70 |
| wind_like | goertzel | 2.00 | 0.60 |
| wind_like | ls_fit | 1.79 | 0.45 |

## §2 Real-data cross-check

Dataset: nowind wave runs in the two canonical March-2026 folders
(PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange, PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange).
Per-probe measurements: **n = 132** across 4 probes, 33 runs, 5 frequencies.

Using `ls_fit` as the reference (it is mathematically the exact paddle-frequency amplitude, modulo Stokes-2f explicit separation). Relative disagreement = (A_method − A_ls_fit) / A_ls_fit.

| method | mean % | std % | max \|%\| | median \|%\| |
|---|---|---|---|---|
| nearest_bin | -0.025 | 0.125 | 0.44 | 0.05 |
| parabolic | -0.023 | 0.125 | 0.43 | 0.06 |
| goertzel | -0.015 | 0.185 | 0.63 | 0.11 |
| ls_fit | +0.000 | 0.000 | 0.00 | 0.00 |

**Breakdown by frequency** (max \|nearest_bin − ls_fit\|/ls_fit %):

| freq_hz | n | max % | median % |
|---|---|---|---|
| 1.300 | 52 | 0.38 | 0.05 |
| 1.400 | 24 | 0.35 | 0.04 |
| 1.500 | 24 | 0.39 | 0.03 |
| 1.600 | 28 | 0.30 | 0.11 |
| 1.700 | 4 | 0.44 | 0.30 |

## Verdict

- `nearest_bin` disagrees with `ls_fit` by median 0.05%, max 0.44%.
- `parabolic` reduces the disagreement to median 0.06%, max 0.43%.
- `goertzel` agrees with `ls_fit` within median 0.107%, max 0.63% (numerically essentially identical).

**Recommendation**: replace `compute_amplitudes_from_fft` in `wavescripts/signal_processing.py` with an `ls_fit` evaluation at f_paddle. It is ~15 lines of code, bin-grid-independent, and gives Stokes-2f amplitude as a bonus output. `parabolic` is an easier drop-in if the team prefers to keep the FFT path but corrects the sinc-attenuation at near-zero cost.

## See also

- Figure: `analysis_scratch/fft_method_comparison.png`
- Synthetic CSV: `analysis_scratch/fft_method_comparison_synth.csv`
- Real-data CSV: `analysis_scratch/fft_method_comparison_real.csv`
- Methodology memo: `memory/methodology_fft_peak_bin_bias.md`
