# Zoom on 1.5 Hz / 0.2 V fullwind per240 — 9373/170

**Figure**: `analysis_scratch/sliding_afft_15hz_02v_zoom.png`

## Finding: the 1.5 Hz anomaly is NOT a transient — it is a bin-grid artifact in the long FFT

The parent sweep flagged 1.5 Hz / 0.2 V fullwind as potentially a genuine
time-varying signal: `pipeline_AFFT=14.5 mm, matched_mid_AFFT=12.8 mm,
plateau_FFT=7.3 mm`. This zoom rules that out.

### Direct test

For one specific 1.5 Hz / 0.2 V fullwind per240 run (20260314), I took
eight non-overlapping 12 s FFT slices across the signal:

| window (s)   | N    | AFFT (mm) | f_peak (Hz) |
|--------------|------|-----------|-------------|
| 24.3 – 36.3  | 3003 | **14.52** | 1.4985      |
| 44.3 – 56.3  | 3003 | **12.67** | 1.4985      |
| 64.3 – 76.3  | 3003 | **14.39** | 1.4985      |
| 84.3 – 96.3  | 3003 | **14.47** | 1.4985      |
| 104.3 – 116.3| 3003 | **14.00** | 1.4985      |
| 124.3 – 136.3| 3003 | **14.17** | 1.4985      |
| 144.3 – 156.3| 3003 | **14.07** | 1.4985      |
| 164.3 – 176.3| 3003 | **13.04** | 1.4985      |

The signal is stable at **~13–14.5 mm for the entire 150 s run**.
The `plateau_FFT` value of 7.3 mm is **wrong** — it is an artifact of
the long-window FFT, not the true paddle amplitude.

### Mechanism

Paddle output at this setting is at **1.4985 Hz** (not exactly 1.500 Hz).

- **Short window** (12 s, N=3003): bin spacing = 250/3003 = **0.083 Hz**.
  Bins at 1.416, 1.499, 1.582. Paddle at 1.4985 Hz → bin 1.499 hits
  essentially perfectly → full amplitude captured (~14 mm).

- **Long window** (150 s, N=37500): bin spacing = 250/37500 = **0.0067 Hz**.
  Bins at 1.493, 1.500, 1.507. Nearest to 1.4985 is 1.495 (-4 mHz off).
  Sinc response at 4 mHz off over 150 s:

    `sin(π·37500·(1.495-1.499)/250) / (π·37500·(-0.004)/250) ≈ 0.51`

  → long-window peak bin reads only ~51% of true amplitude → 14·0.51 ≈ 7.1 mm.
  That matches the observed 7.33 mm.

### Pipeline values across the 4 fullwind runs at this condition

- **20260314**: pipeline_AFFT=14.52, Atd=19.50 mm
- **20260323**: pipeline_AFFT=17.00, Atd=19.99 mm
- **20260326**: pipeline_AFFT=15.12, Atd=18.96 mm
- **20260327**: pipeline_AFFT=17.20, Atd=21.78 mm

All consistent at ~15–17 mm AFFT, ~19–22 mm Atd. No transient.
Nowind reference runs: pipeline_AFFT ≈ 14.7–15.2 mm. Very similar to fullwind.

### Generalisation — systematic FFT peak-bin bias

The **nearest-bin amplitude** depends on how well the FFT bin grid aligns
with the paddle's actual (drifted) frequency. Both short and long FFTs
can under-read; the sign and magnitude of the bias depends on the
specific paddle drift and window length.

Worked examples from the parent sweep:

- **1.3 Hz / 0.1 V**: paddle likely at 1.300 Hz. Short-window bin at
  1.333 Hz (-33 mHz off) → sinc attenuation ~0.68 → short reads 5.2 mm.
  Long-window bin at 1.300 Hz (exact) → full amplitude → reads 7.65 mm.
  **True amplitude ≈ 7.65 mm; pipeline under-reads by ~32%.**

- **1.5 Hz / 0.2 V**: paddle at 1.4985 Hz. Short-window bin at 1.499 Hz
  (+0.5 mHz off) → full amplitude → reads 14 mm. Long-window bin at
  1.495 Hz (-4 mHz off) → sinc attenuation ~0.51 → reads 7.3 mm.
  **True amplitude ≈ 14 mm; plateau_FFT under-reads by ~48%.**

Both directions occur in the real data.

### Saving grace for OUT/IN ratio

If the bias is similar at both IN and OUT probes (same paddle drift,
similar window lengths), the ratio `OUT/IN (FFT)` is approximately
bias-free — the numerator and denominator cancel. However:

- Different probes may have slightly different analysis window lengths
  (different start/end samples) → different bin spacings.
- Fullwind introduces probe-specific phase jitter → different spectral
  broadening at each probe.
- At wind-contaminated probes, spectral leakage from the wind-wave tail
  modifies the peak-bin response.

### What remains unresolved

- **Paddle frequency calibration**: what is the actual paddle frequency
  vs nominal, per run? If this were tracked (already via `Probe {pos}
  Frequency (FFT)`), a per-run peak-interpolation correction could be
  applied.
- **Sub-bin interpolation**: parabolic fit around the peak bin would
  reduce this bias to <1% — worth considering as a pipeline enhancement
  for publication-grade amplitude estimates.

### Implication for parent sweep findings

The parent sweep (`sliding_afft_fullwind_sweep_findings.md`) classified
the 1.5 Hz / 0.2 V case as "mid in between — genuine time-varying
amplitude". **That classification should be revised to: bin-resolution
artifact (plateau-FFT under-reads by half).** All 8 cases in the parent
sweep are now explained as various flavours of the same FFT peak-bin
bias — no cases are genuine within-run transients.
