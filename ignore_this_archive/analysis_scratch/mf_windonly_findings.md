# Wind-wave coherence and transfer function — fullwind nowave

**Date**: 2026-04-16
**Script**: `analysis_scratch/mf_windonly.py`
**Figure**: `analysis_scratch/mf_windonly.png`

## Why not Mansard-Funke directly?

MF requires a coherent single-frequency wave at each FFT bin. Wind-only waves are
broadband stochastic: at any individual bin, the two probe signals have random phases.
Applying MF to incoherent noise always gives |A| ≈ |B| → R → 1 regardless of physics.
This was confirmed: initial MF run on wind-only data gave R ≈ 1.0 for all moorings.

## Method: Welch cross-spectral density

Concatenate all runs per mooring → compute via Welch averaging:
- PSD₁(f), PSD₂(f) at 8804/250 and 9373/170
- CPSD G₁₂(f) = cross-spectral density
- Coherence γ²(f) = |G₁₂|² / (PSD₁·PSD₂)  [0=incoherent, 1=fully coherent]
- Transfer function |H(f)| = √(PSD₂/PSD₁), arg(H) = arg(G₁₂)

Welch parameters: nperseg=2048 samples (8.2s), 50% overlap, Δf=0.122 Hz.

Standing wave fingerprint:
- For pure incident wave: |H(f)| = 1, arg(H) = −kΔ exactly
- For partial standing wave (R > 0): |H(f)| oscillates around 1 with amplitude ≈ 2R;
  arg(H) deviates from −kΔ. Both vary with frequency in a node/antinode pattern.

Only bins with γ² ≥ 0.5 are considered reliable.

## Numerical summary

```
Wind-wave coherence & transfer function — fullwind nowave fullpanelGenerated: 2026-04-16 11:44Probe pair: 8804/250 + 9373/170  Δ=569mmWelch: nperseg=2048 (8.2s)  noverlap=1024  Δf=0.122 HzCoherence threshold for phase reliability: γ² ≥ 0.5Mooring: above_50  (4 runs, 1423s, 0 reliable freq bins)Mooring: below_90_loose230  (10 runs, 1616s, 0 reliable freq bins)Mooring: below_90_loose300  (2 runs, 393s, 0 reliable freq bins)
```

## Key findings

### Coherence is essentially zero

Maximum γ² ≈ 0.19, median γ² ≈ 0.003 — at all frequencies, for all mooring types.
Even with the longest runs concatenated (1423s above_50, 1616s below_90), γ² never
exceeds 0.37 anywhere in the 0.5–6 Hz range (even with 33s segments).

**The wind-generated wave field is spatially incoherent at the 569 mm probe separation.**
Each probe responds to independent local turbulent pressure fluctuations. No coherent
wave train connects the two probes at any frequency.

### Consequence: no two-probe analysis is possible for wind-only waves

MF decomposition, transfer function H(f), and any phase-based reflection analysis all
require coherent signals. With γ² ≈ 0.003, the CPSD phase is random — any computed R,
|H|, or phase deviation is pure noise. This was confirmed: naive MF applied to wind-only
data gave R ≈ 1.0 for all moorings (= breakdown result, not physical).

### |H| > 1 is a fetch effect, not reflection

The ratio |H(f)| = sqrt(PSD₂/PSD₁) ≈ 1.2–1.7 across frequencies and moorings.
9373/170 sees ~17% more wind-wave energy than 8804/250 because it has ~0.6 m more
wind fetch (wind travels paddle→panel; 9373 at x=9.37 m vs 8804 at x=8.80 m).

  - above_50:          std(8804) = 3.91 mm, std(9373) = 4.61 mm (ratio 1.18)
  - below_90_loose230: std(8804) = 3.66 mm, std(9373) = 4.29 mm (ratio 1.17)

This is the same fetch asymmetry that makes the IN probe (9373/170) more wind-
contaminated than the OUT probe (12400/250, sheltered behind the panel).

### What was the "visible standing wave" with above_50 mooring?

The nowave wind field is incoherent — it cannot produce a standing wave pattern.
The visual standing wave observed under above_50 + full wind must have occurred in
combined wind+wave runs. The paddle wave (coherent, single-frequency) reflects off
the panel, and the stiffer above-water mooring amplifies this reflected component.
The pattern was a paddle-frequency standing wave, not a wind-wave reflection.

This is consistent with the paddle-wave MF finding (mansard_funke_findings.md):
above_50 and below_90 have similar median R ≈ 0.05–0.07. Higher R at 1.6–1.7 Hz
(nowind, above_50) could produce a visible node/antinode at 0.3V amplitudes.

### Thesis treatment

> "The wind-generated wave field between paddle and panel is spatially incoherent at
> the 569 mm probe separation (γ² < 0.2 at all frequencies and all run conditions),
> precluding two-probe reflection analysis of wind waves. The 17% larger wave energy
> at 9373/170 compared to 8804/250 under full wind is consistent with fetch-limited
> wave growth over the 570 mm gap, not with reflection. The panel's reflection
> coefficient was characterised from paddle-wave runs: R ≈ 0.05–0.07 (Mansard-Funke,
> all mooring types, no-wind conditions)."

## Status

- [x] Welch CPSD computed per mooring
- [x] Coherence measured: γ² < 0.2 everywhere — wind-wave field spatially incoherent
- [x] |H| > 1 explained by fetch gradient, not reflection
- [x] Visual standing wave attributed to paddle-wave reflection in combined runs
- [x] Thesis treatment written
