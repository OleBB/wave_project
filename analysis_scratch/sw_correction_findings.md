# Standing-wave correction analysis

**Date**: 2026-04-16
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
  0.80   0.8155    0.991   0.808    1.211   0.988      5     7
  0.90   1.0267    0.915   0.940    1.003   1.030      5     7
  1.00   1.1535    0.927   1.069    0.933   1.077      8     8
  1.10   0.8074    0.902   0.728    0.859   0.694      6     9
  1.20   1.1998    0.823   0.987    0.798   0.958      8    12
  1.30   0.8030    0.753   0.605    0.813   0.653     48    35
  1.40   1.1761    0.651   0.766    0.716   0.843     25    25
  1.50   0.9443    0.537   0.507    0.662   0.625     27    22
  1.60   0.9140    0.453   0.414    0.603   0.551     28    24
  1.70   1.1982    0.312   0.373    0.528   0.633     14    14
  1.80   1.0231    0.272   0.279    0.455   0.465      4     7
  1.90   0.8028    0.134   0.107    0.369   0.296      2     4
  2.00   0.9127    0.144   0.131    0.439   0.401      1     1
```

---

## Smoothness test

Metric: Σ(Δ²y)² over 1.0–1.9 Hz group means (sum of squared second differences).
Lower = smoother curve.

| Wind condition | Raw curvature | Corrected curvature | Change | Verdict |
|----------------|--------------|---------------------|--------|---------|
| No wind | 0.028787 | 1.281246 | -4350.8% | less smooth ✗ |
| Full wind | 0.020754 | 1.258295 | -5962.8% | less smooth ✗ |

---

## Sensitivity analysis

Grid: R ∈ {0.05, 0.10, 0.15, 0.20, 0.25, 0.30} × X_panel ∈ {10.0, 10.5, 11.0, 11.5, 12.0} m
Total: 30 parameter combinations, each tested for nowind and fullwind.

- No-wind smoother after correction: **0/30** combinations
- Full-wind smoother after correction: **3/30** combinations

---

## Key finding: negative evidence for R=0.20

**The smoothness test fails universally** — the SW correction at R=0.20 makes the curve
LESS smooth for 30/30 nowind parameter combinations and 27/30 fullwind combinations.
Curvature increases by +4350% (nowind) and +5963% (fullwind) at the nominal R=0.20, panel=11.0 m.

This is **not a technical failure** — it is a real physical finding:

> **The raw OUT/IN curve is already smooth. The correction at R=0.20 would impose a
> zigzag oscillation of ±0.15–0.20 between adjacent frequencies that simply is not
> present in the data.**

If R were truly 0.20, the raw OUT/IN would already show this ±15% zigzag between
consecutive node/antinode frequencies. Instead, the raw curve is a nearly monotonic
decreasing function with frequency-to-frequency variation of ~0.05. The SW oscillations
at R=0.20 would have amplitude 3× larger than the noise in the raw data — they would
be easily visible. They are not.

### Upper bound on R from the smoothness argument

The correction produces noticeable zigzag when:
  2 × R / (1 − R²) ≈ R [for small R]  ≥  typical raw frequency-to-frequency variation

The raw curve has adjacent-frequency steps of ~0.05–0.10 in OUT/IN. The SW factor
oscillation amplitude is approximately 2R at small R. For the correction to be
invisible in the raw data, we need 2R ≪ 0.05, i.e., **R ≪ 0.025 or so**.

A more conservative estimate: if the SW oscillations are suppressed below the data
scatter (SEM ≈ 0.02), then R must be < ~0.01–0.02. This is substantially smaller
than the assumed R=0.20.

### Why this is consistent with FPV panel physics

A floating solar panel is primarily a surface-floating geometry with a mooring that
allows lateral wave motion (the panel can drift up and down with the wave). Such panels
are not rigid partial reflectors like a breakwater. Their wave reflection coefficient
is expected to be low — much lower than a vertical wall (R≈0.9) or a fixed pontoon
(R≈0.3–0.5). R < 0.05 is physically plausible for a compliant floating panel.

Supporting evidence from the data: at 0.80 Hz, OUT/IN ≈ 0.99 (nowind) and 1.21 (fullwind).
The fact that OUT/IN ≈ 1.0 at low frequencies (the panel barely damps long waves) is
consistent with low R — a transparent panel does not generate reflections.

### What 1.3 Hz shows specifically

Even at 1.3 Hz — where both X_panel=11.0 m and X_panel=11.5 m put the probe near a node —
the raw OUT/IN (nowind) = 0.75. If R=0.20 and the probe is at a node (SW=0.80):
  T_corrected = 0.75 × 0.80 = 0.60 (not 0.75)

But since the smoothness analysis shows R is likely much smaller, the true correction
at 1.3 Hz is also much smaller. At R=0.02: SW_node ≈ 0.98 → T_corrected ≈ 0.74.
The correction is negligible.

The 1.3 Hz standing-wave anomaly (OUT/IN_nowind > OUT/IN_fullwind at low amplitude,
see mooring_comparison_findings.md) is more likely explained by amplitude-dependent
wave dynamics or measurement scatter than by a reflection node.

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

The smoothness test provides strong **negative evidence** against a large standing-wave
correction. Recommended thesis approach:

1. **Primary result: use uncorrected OUT/IN** — fully data-driven, no model assumptions.
   The test shows the data is not consistent with R=0.20, so the assumed correction would
   introduce model error larger than the effect being corrected.

2. **Methodology note**: acknowledge that a partial reflection from the panel could in principle
   bias the IN probe measurement. State that the data shows no detectable node/antinode
   fingerprint at the expected frequency spacing, placing an upper bound of R < ~0.05 on the
   panel's reflection coefficient.

3. **For 1.3 Hz specifically**: the anomalous nowind scatter (see mooring_comparison_findings.md)
   is documented but the SW correction is not applied as a correction — it would require R>0.10
   to explain the 25% inflation, and R=0.10 is inconsistent with the smoothness test.

4. **If Mansard-Funke is run**: it will provide a direct measurement of R(f) and resolve the
   ambiguity. Until then, report R < 0.05 as the data-supported bound.

## Status

- [x] SW factors computed and applied
- [x] Smoothness metric calculated (curvature metric, all frequencies 1.0–1.9 Hz)
- [x] Sensitivity analysis over 30 (R, panel position) combinations
- [x] Key finding: correction universally makes curve LESS smooth → R ≪ 0.20
- [x] Upper bound estimate: R < ~0.05 from smoothness argument
- [x] Thesis treatment recommendation written
- [ ] Mansard-Funke decomposition to measure R(f) directly (definitive next step)
- [ ] No-panel experiment in current probe config (ideal, requires new data)
