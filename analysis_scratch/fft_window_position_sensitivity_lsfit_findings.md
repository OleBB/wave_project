# FFT window-position sensitivity — 10T window slid across the per240 plateau

Generated: 2026-05-05T10:24:04Z

**Dataset**: fullpanel per240 wave runs, quality_flag=ok, from the two canonical
March-2026 lowrange folders. n_runs = 51.

**Method**: hold window length at N = 10 periods (pipeline default);
sweep the start position T_ref from 40T to 80T in 1T steps.
T_ref is in OUT-probe coordinates; each probe's local start is
`T_ref − ΔT_probe` via the probe-shifted H&G convention. Reference:
**T_ref = 50T** = current pipeline default.

## Per-run OUT/IN variability across T_ref

- Median range (max − min)/median across positions: **11.92 %**
- Max range across positions (worst run): **212.43 %**
- Median std(OUT/IN) / median(OUT/IN): **3.12 %**

## Median drift from T_ref=50T, per wind condition

| T_ref | nowind median % | nowind max \|Δ\| % | fullwind median % | fullwind max \|Δ\| % |
|---|---|---|---|---|
| 40T | -3.253 | 28.79 | +0.776 | 12.01 |
| 42T | -1.095 | 28.04 | +0.661 | 9.52 |
| 45T | +0.630 | 13.52 | +0.195 | 10.94 |
| 48T | +0.486 | 4.55 | +0.158 | 2.84 |
| **50T** | 0.000 | 0.00 | 0.000 | 0.00 |
| 52T | +0.019 | 2.85 | +0.156 | 2.47 |
| 55T | +0.120 | 11.51 | +0.073 | 4.26 |
| 58T | -0.599 | 6.29 | -0.326 | 6.67 |
| 60T | -0.404 | 17.28 | -0.797 | 9.46 |
| 65T | +0.087 | 83.72 | -1.452 | 13.52 |
| 70T | +0.911 | 10.40 | -2.057 | 14.68 |
| 75T | +1.370 | 116.85 | -2.186 | 13.37 |
| 80T | +0.670 | 90.52 | -2.723 | 12.32 |

## Where is the plateau?

Median OUT/IN vs T_ref (all runs pooled):

| T_ref | median OUT/IN | drift from 50T % |
|---|---|---|
| 40T | 0.7256 | +0.013 |
| 42T | 0.7411 | +2.146 |
| 44T | 0.7380 | +1.724 |
| 46T | 0.7394 | +1.922 |
| 48T | 0.7338 | +1.151 |
| **50T** | **0.7255** | **+0.000** |
| 52T | 0.7315 | +0.823 |
| 54T | 0.7323 | +0.943 |
| 56T | 0.7272 | +0.237 |
| 58T | 0.7229 | -0.361 |
| 60T | 0.7266 | +0.149 |
| 62T | 0.7280 | +0.351 |
| 65T | 0.7217 | -0.527 |
| 68T | 0.7186 | -0.953 |
| 70T | 0.7199 | -0.769 |
| 75T | 0.7252 | -0.046 |
| 80T | 0.7214 | -0.562 |

## Takeaway (interpret manually before publishing)

- Per-run OUT/IN across T_ref ∈ [40, 80]T varies by median **11.92 %**
  (max − min). This is the "how sensitive is OUT/IN to where we start the window" answer.

- See panel (a) of the figure for per-run drift traces vs T_ref.
  Panel (d) shows median A_in and A_out independently — the plateau regions are visible.

- Cross-reference with `hg_window_stability_findings.md`:
  - per240 10T sliding AFFT CV within ±5T of H&G start was ~0.5 % (IN and OUT).
  - This sweep extends the ±5T to ±30T — confirms / refutes plateau beyond pipeline default.

## See also

- Figure: `analysis_scratch/fft_window_position_sensitivity_lsfit.png`
- Data: `analysis_scratch/fft_window_position_sensitivity_lsfit.csv`
- Length sensitivity (companion): `analysis_scratch/fft_window_sensitivity_lsfit_findings.md`
- Plateau (prior): `analysis_scratch/hg_window_stability_findings.md`
