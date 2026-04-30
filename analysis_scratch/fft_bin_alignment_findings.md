# FFT bin alignment and sinc leakage on the canon dataset

Reproduce: `python analysis_scratch/fft_bin_alignment_check.py`
Outputs:   `fft_bin_alignment_table.csv`, `fft_bin_alignment.png`

## What this checks

For a single-bin FFT amplitude estimate to be unbiased, the target frequency
must land exactly on a bin centre. Bins sit at multiples of `Δf = fs/N`.
If the target lands `δ` bin widths off, the nearest-bin reading is
attenuated by `|sinc(δ)|`; at `δ = 0.5` the attenuation is ~36 %.

The H&G window in this pipeline holds **N ≈ 10 paddle periods** by
construction (UC-snapped at both ends since 2026-04-23), so:

```
N ≈ 10·fs/f_paddle   ⇒   Δf = fs/N ≈ f_paddle/10   ⇒   bin 10 ≈ f_paddle
```

Any leftover misalignment comes from rounding `N` to an integer number of
samples and from per-run UC-snap drift.

## Math, in 4 lines

```python
df            = fs / N
bin10         = 10 * df
offset_bins   = (f_target - bin10) / df
sinc_atten    = 1 - abs(np.sinc(offset_bins))   # numpy: sinc(x) = sin(πx)/(πx)
```

(See `analysis_scratch/fft_bin_alignment_check.py`, function
`measure_alignment`. Single-bin nearest-bin amplitude is what
`compute_amplitudes_from_fft` in `wavescripts/signal_processing.py:226`
returns.)

## Measured on canon (60 probe-runs, 1.3–1.6 Hz × 0.2 V × full panel × 3 probes × both winds)

| quantity | value |
|---|---|
| bin width Δf = fs/N | 0.129 – 0.161 Hz |
| \|offset\| from bin 10, median | **0.016 bin widths** |
| \|offset\| from bin 10, max | 0.080 bin widths |
| sinc attenuation, median | **0.04 %** |
| sinc attenuation, max | 1.05 % (one outlier; next-largest ~0.4 %) |

The 1.05 % outlier is a fullwind run where the H&G end-snap fell back to
fixed length under the spurious-UC guard (commit `eefba87`).

## Visual

`fft_bin_alignment.png`:

* **(a)** FFT magnitude spectrum (1.4 Hz canon, 9373/170, nowind), with
  every bin centre marked. The 1.4 Hz red line lies essentially on top of
  bin 10.
* **(b)** Theoretical sinc-attenuation curve `1 − |sinc(δ)|` overlaid with
  all 60 measured (`offset_bins`, `sinc_atten_pct`) points. Every point
  sits in the flat bottom of the curve.

## Citable one-liner

> The H&G window is constructed to span an integer number of paddle
> cycles, so on the canonical dataset the paddle frequency lands within
> 0.08 bin widths of the 10th FFT bin (median 0.016) — the resulting
> nearest-bin sinc attenuation is below 0.1 % on average.

## Cross-checks (already in repo)

* 4-method comparison (nearest-bin / parabolic / Goertzel / LS-fit) on 128
  per-probe nowind measurements: 0.4 % max, 0.04 % median —
  `analysis_scratch/fft_method_comparison_findings.md`.
* FFT vs LS-fit amplitude on canon after UC-end-snap: 0.10 % max —
  `memory/methodology_hg_window_kills_peak_bias.md`,
  `memory/session_2026-04-23.md`.
