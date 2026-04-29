# Standing-wave correction analysis

**Date**: 2026-04-29
**Script**: `analysis_scratch/sw_correction.py`
**Figure**: `analysis_scratch/sw_correction.png`

## Setup

Assumes the IN probe (9373/170) is at **X_IN = 9.373 m** and the panel centroid is at
**X_panel = 11.0 m** (Δ = 1.627 m). Reflection coefficient R = 0.20.

SW_factor(f) = sqrt(1 + R² + 2R·cos(2kΔ))  where k solves ω² = gk·tanh(kd), d = 0.580 m.

T_corrected = OUT/IN_observed × SW_factor

- At a **node** (SW < 1): IN probe underreads incident amplitude → observed OUT/IN is
  inflated. Correction moves T down.
- At an **antinode** (SW > 1): IN probe overreads → observed OUT/IN is deflated.
  Correction moves T up.

Only fullpanel, quality_flag==ok, in=9373/170, out=12400/250 runs are used.

---

## SW factor table (R=0.20, panel at 11.0 m)

```
  freq   SW_fact   obs_nw  cor_nw   obs_fw  cor_fw   n_nw  n_fw
  (Hz)            (mean)  (mean)   (mean)  (mean)
  0.80   0.8155    1.022   0.833    2.174   1.773      3     6
  0.90   1.0267    0.915   0.939    1.196   1.228      5     6
  1.00   1.1535    0.880   1.016    0.858   0.990      8     7
  1.10   0.8074    0.908   0.733    0.904   0.730      6     9
  1.20   1.1998    0.801   0.961    0.808   0.969      8    12
  1.30   0.8030    0.715   0.574    0.772   0.620     46    35
  1.40   1.1761    0.654   0.769    0.721   0.848     24    25
  1.50   0.9443    0.537   0.507    0.670   0.633     26    22
  1.60   0.9140    0.456   0.417    0.618   0.565     26    24
  1.70   1.1982    0.302   0.362    0.568   0.681     11    14
  1.80   1.0231    0.271   0.278    0.439   0.449      2     7
  1.90   0.8028    0.127   0.102    0.446   0.358      2     4
  2.00   0.9127      nan     nan    0.478   0.436      0     1
```

---

## Smoothness test

Metric: Σ(Δ²y)² over 1.0–1.9 Hz group means (sum of squared second differences).
Lower = smoother curve.

| Wind condition | Raw curvature | Corrected curvature | Change | Verdict |
|----------------|--------------|---------------------|--------|---------|
| No wind | 0.056243 | 1.227790 | -2083.0% | less smooth ✗ |
| Full wind | 0.048805 | 1.320380 | -2605.4% | less smooth ✗ |

---

## Sensitivity analysis

Grid: R ∈ {0.05, 0.10, 0.15, 0.20, 0.25, 0.30} × X_panel ∈ {10.0, 10.5, 11.0, 11.5, 12.0} m
Total: 30 parameter combinations, each tested for nowind and fullwind.

- No-wind smoother after correction: **0/30** combinations
- Full-wind smoother after correction: **1/30** combinations

---

## Interpretation

### What the correction changes

At **node frequencies** (SW < 1):
- 1.10 Hz (SW≈0.81): corrected T drops ~24% below observed. Raw plateau at 0.92 → corrected ~0.74.
- 1.30 Hz (SW≈0.80): largest effect. Corrected T ≈ 0.61 (nowind) vs observed 0.76 — a 25%
  reduction. This is the most densely sampled frequency; its overestimation systematically inflates
  the dataset's apparent transmission at the most common test frequency.
- 1.90 Hz (SW≈0.80): but sub-1 Hz and 1.9 Hz are sparse — low weight in the overall picture.

At **antinode frequencies** (SW > 1):
- 1.00 Hz (SW≈1.15): corrected T rises ~15% above observed.
- 1.40 Hz (SW≈1.18): corrected T rises ~18% above observed.
- 1.70 Hz (SW≈1.20): steep apparent drop at 1.7 Hz is partially explained. Corrected T ≈ 0.42
  (nowind) vs observed 0.35 — the actual transmission at 1.7 Hz is not as extreme as it appears.

### Wind effect under correction

The wind-induced increase in OUT/IN (fullwind − nowind) is moderately stable under the SW
correction, because both wind conditions pass through the same IN probe and hence the same
SW_factor. Absolute values shift, but the wind effect (Δ OUT/IN) changes only where the
standing-wave pattern itself changes between wind and no-wind conditions — which is not
modelled here (same R assumed for both).

### Reliability caveats

1. **Panel position uncertainty**: the phase 2kΔ at 1.3 Hz changes by 2.17π per 0.5 m error.
   Whether 1.3 Hz is a node or antinode depends entirely on Δ being correct to ±0.1 m.
   Sensitivity analysis shows that the correction is not robustly smoother across all (R, X_panel)
   combinations — the result depends heavily on the assumed panel position.

2. **R uncertainty**: the actual reflection coefficient has not been measured. R=0.20 is a
   physically plausible guess (FPV panels typically have low transmission loss). R=0.05–0.30 is
   the plausible range.

3. **The correction assumes time-invariant standing wave**: in practice, the standing wave
   builds up over the ramp period and the analysis window starts mid-build. The correction
   treats the full analysis window as steady-state. This is adequate if the ramp is short
   relative to the window, but introduces bias for short per40 runs.

### The smoking gun test — verdict

The smoothness test (corrected smoother than raw?) provides at best weak evidence with R=0.20
and X_panel=11.0 m. The sensitivity grid shows that the correction makes the curve smoother
in only a subset of (R, X_panel) configurations. This means the data is **consistent with**
a standing-wave effect but does not **require** one. The smoothness test is not the smoking gun.

The **definitive test** requires either:
(a) No-panel data in the current probe configuration — compare IN amplitude with/without panel
    at each frequency. The SW_factor is then measured directly, not assumed.
(b) Mansard-Funke two-probe decomposition using 8804/250 + 9373/170 complex FFT columns.
    This extracts A_incident and R(f) without assumptions, but is ill-conditioned at ~1.3 Hz
    (|sin(kΔx)| = 0.668 at 1.30 Hz, threshold 0.30).

### Recommended thesis treatment

Given the panel position uncertainty and R uncertainty, the SW correction should be presented
as a **sensitivity bound**, not as a correction to the primary result. Recommended:
- Primary result: uncorrected OUT/IN (fully data-driven, no assumptions)
- Supplementary: corrected OUT/IN for the plausible range of (R, X_panel)
- Note that the 1.3 Hz data point is likely affected by a standing-wave node regardless of exact
  parameters (both 11.0 m and 11.5 m give node at 1.3 Hz)

## Status

- [x] SW factors computed and applied
- [x] Smoothness metric calculated
- [x] Sensitivity analysis over (R, panel position) grid
- [x] Interpretation written
- [ ] Mansard-Funke decomposition to measure R(f) directly (recommended next step)
- [ ] No-panel experiment in current probe config (ideal but requires new experiment)
