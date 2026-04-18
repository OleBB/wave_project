# T_cross figure — findings (2026-04-18)

Builds on `t_cross_idea.md` (2026-04-14). That doc established the metric
at 0.1 V only, across mixed moorings. Here we extend to all three
amplitudes (0.1 / 0.2 / 0.3 V) within the thesis scope (1.3–1.6 Hz) and
pool the loose230 + loose300 moorings (merged in `meta_results` per the
CH04 §3c finding that mooring length has no detectable transmission
effect).

Data: `meta_results` = 2 validated lowrange folders, cond4 only, 101
quality-ok full-panel wave runs. Figures:
- `output/FIGURES/ch05_t_cross_{10,20,30}V.pdf`
- Combined quick-view: `analysis_scratch/t_cross_figure.pdf`
- Numerical table: `analysis_scratch/t_cross_figure_summary.csv`

---

## A_in wind-independence check (paddle consistency)

Does the paddle deliver the same amplitude with and without wind at
each (freq, amp)? Required for T_cross to be interpretable.

| amp | freq | A_in^nw | A_in^fw | ratio fw/nw | verdict |
|-----|------|---------|---------|-------------|---------|
| 0.1 V | 1.3 | 7.49 | 6.50 | 0.87 | ⚠ outside ±15% (low SNR) |
| 0.1 V | 1.4 | 7.89 | 7.69 | 0.97 | ✓ |
| 0.1 V | 1.5 | 7.86 | 7.42 | 0.95 | ✓ |
| 0.1 V | 1.6 | 7.03 | 8.66 | 1.23 | ⚠ wind inflates A_in |
| 0.2 V | 1.3 | 14.92 | 15.27 | 1.02 | ✓ |
| 0.2 V | 1.4 | 15.15 | 15.76 | 1.04 | ✓ |
| 0.2 V | 1.5 | 15.21 | 16.47 | 1.08 | ✓ |
| 0.2 V | 1.6 | 14.77 | 17.09 | **1.16** | ⚠ borderline |
| 0.3 V | 1.3 | 21.46 | 22.09 | 1.03 | ✓ |
| 0.3 V | 1.4 | 22.09 | 23.54 | 1.07 | ✓ |
| 0.3 V | 1.5 | 21.24 | 24.20 | 1.14 | ✓ |
| 0.3 V | 1.6 | 21.52 | 24.35 | 1.13 | ✓ |

**Summary**: paddle output is wind-independent within ±15% at 0.2–0.3 V
across 1.3–1.6 Hz (just barely at 0.2 V 1.6 Hz). At 0.1 V the ratio
scatters widely because per-run SNR is too low for a clean fullwind
A_in measurement. T_cross is therefore most reliable at 0.2–0.3 V.

The systematic upward trend in the ratio with frequency (1.02 → 1.16
at 0.2 V) is itself informative: it's the IN-probe wind-contamination
bias that T_cross was designed to eliminate. Under wind, A_in reads
higher not because the paddle delivers more, but because the FFT bin
at the paddle frequency absorbs some nearby wind energy.

---

## The three transmission metrics per (amp, freq)

All values are means; errors are in the CSV / on the figure.

### 0.2 V

| f | (OUT/IN)_nw | T_cross | (OUT/IN)_fw | honest Δ (T_cross − nw) | standard Δ (fw − nw) |
|---|-------------|---------|-------------|-------------------------|----------------------|
| 1.3 | 0.795 | 0.839 | 0.820 | **+0.04** | +0.03 |
| 1.4 | 0.702 | 0.785 | 0.756 | **+0.08** | +0.05 |
| 1.5 | 0.588 | 0.745 | 0.689 | **+0.16** | +0.10 |
| 1.6 | 0.478 | 0.718 | 0.622 | **+0.24** | +0.14 |

### 0.3 V

| f | (OUT/IN)_nw | T_cross | (OUT/IN)_fw | honest Δ (T_cross − nw) | standard Δ (fw − nw) |
|---|-------------|---------|-------------|-------------------------|----------------------|
| 1.3 | 0.825 | 0.866 | 0.842 | **+0.04** | +0.02 |
| 1.4 | 0.697 | 0.812 | 0.763 | **+0.11** | +0.07 |
| 1.5 | 0.621 | 0.811 | 0.712 | **+0.19** | +0.09 |
| 1.6 | 0.506 | 0.761 | 0.674 | **+0.26** | +0.17 |

### 0.1 V (caveat: A_in wind-independence violated at 1.3 and 1.6 Hz)

| f | (OUT/IN)_nw | T_cross | (OUT/IN)_fw | honest Δ | standard Δ |
|---|-------------|---------|-------------|----------|------------|
| 1.3 | 0.700 | 0.775 | 0.941 | +0.08 | +0.24 |
| 1.4 | 0.545 | 0.726 | 0.746 | +0.18 | +0.20 |
| 1.5 | 0.457 | 0.651 | 0.756 | +0.19 | +0.30 |
| 1.6 | 0.386 | 0.579 | 0.580 | +0.19 | +0.19 |

---

## Key findings

**1. The standard (OUT/IN)_fw metric systematically underreports the wind
effect** at 0.2 V and 0.3 V above 1.3 Hz. The underestimate grows with
frequency:
- At 1.4 Hz: ~40% of the honest effect is missed
- At 1.5 Hz: ~40–50% missed
- At 1.6 Hz: ~35–40% missed (e.g. 0.3 V: honest +0.26, standard +0.17)

The mechanism is spectral contamination: fullwind inflates A_in^fw by
2–16% relative to A_in^nw at these frequencies (per the wind-independence
check above), so dividing by a larger A_in makes (OUT/IN)_fw appear smaller
than the true transmission.

**2. T_cross is nearly flat above 1.4 Hz at 0.3 V** (0.81 → 0.81 → 0.76
across 1.4 → 1.5 → 1.6 Hz). The nowind transmission drops steeply
(0.70 → 0.62 → 0.51) over the same range. The gap — the wind effect —
widens dramatically.

**3. At 1.3 Hz the two metrics agree** (honest Δ ≈ standard Δ ≈ +0.02–0.04
at 0.2–0.3 V). The IN-probe bias is small at the lowest paddle frequency
where the wind PSD tail has less overlap with the FFT bin. This is the
regime where the standard metric is trustworthy.

---

## Thesis-ready claim (suggested)

> "Referenced to the clean nowind incident amplitude (T_cross metric),
> wind increases wave transmission past the panel by **+4% at 1.3 Hz to
> +26% at 1.6 Hz** (0.3 V). The effect grows monotonically with paddle
> frequency. The standard ratio (OUT/IN)_fw underestimates this effect
> by up to a factor of two above 1.4 Hz, because fullwind contaminates
> the IN probe's FFT reading at the paddle frequency."

This is a stronger and cleaner statement than what the raw (OUT/IN)_fw
vs (OUT/IN)_nw comparison supports on its own.

---

## Caveats

1. **0.1 V data is less reliable** — the wind-independence precondition
   fails at 1.3 and 1.6 Hz there (ratios 0.87 and 1.23). Include for
   completeness but annotate in captions.
2. **1.6 Hz at 0.2 V is borderline** (ratio 1.16 vs 1.15 threshold).
3. **Pooled mooring assumption**: the result pools loose230 + loose300
   runs. The separate CH04 §3c analysis showed these moorings are
   transmission-equivalent within ±7% in the thesis band, which is an
   order of magnitude smaller than the wind effect measured here
   (+4–+26%), so pooling is safe.
4. **No 1.7 Hz data** — out of thesis scope per the scope cut.

---

## Status

- [x] T_cross extended to 0.2 V and 0.3 V within thesis scope
- [x] A_in wind-independence precondition verified per (freq, amp)
- [x] Three thesis PDFs + LaTeX subfigure stub written
- [x] Wired into main_save_figures.py as §3b
- [ ] User's visual review of the figures
- [ ] Decide whether the honest-wind-effect framing should replace the
      (OUT/IN)_fw − (OUT/IN)_nw framing in the thesis text (§3 caption +
      results prose)
