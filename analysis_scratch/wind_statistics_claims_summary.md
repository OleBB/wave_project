# Wind-statistics claims — what we can and cannot defend

Date: 2026-04-20 (kurtosis convention corrected same day)
Scope: what the pitot-based wind stats (TI, skewness, kurtosis, mean/median) in
`windscripts/windprofile_combined.py` fig11 actually let us claim in the thesis,
and what references those claims lean on.

## Kurtosis convention (important)

LabVIEW's **1D Sample Statistics / Basic Statistics** VIs (2016 vintage)
return **excess kurtosis** — i.e. the kurtosis is defined with
Gaussian = 0 (NI documentation: `kurtosis = μ4/σ⁴ − 3`).
The `Kurt` column in the mAstats files is therefore already excess kurt;
`windprofile_combined.py::process_folder` previously subtracted 3 a *second*
time (bug fixed same day). All numbers below are excess kurtosis under the
NI / Fisher convention (Gaussian = 0).

## Measured numbers (from fig11, 2026-04-20, post-fix)

| Quantity                | Full vind (red, n=65)   | Laveste vind (green, n=66) |
|-------------------------|-------------------------|----------------------------|
| TI [%]                  | median 1.50 (0.61–2.26) | median 1.76 (1.21–4.45)    |
| Skewness                | median +0.01            | median +0.04               |
| Excess kurtosis (Gauss=0)| median **+7.03** (+3.4 … +8.6) | median **+6.80** (+4.7 … +8.6) |

All three metrics are means across 1-second blocks within each run, then
aggregated across all runs that measured a given height. Blocks are sampled at
100 Hz → each block carries ~100 samples.

(Previously published numbers in this document — `+3.44` / `+3.77` — were
under-reported by exactly 3 because the pipeline subtracted 3 from an already-
excess LabVIEW value. The corrected values are roughly +3 higher.)

## What the numbers mean

- **Low TI (~1.5–1.8 %)** — the fluctuation magnitude relative to the mean is
  small at both wind conditions, and low compared to typical fan-driven flows.
- **Symmetric fluctuations (|skew| ≲ 0.05)** — no systematic gust asymmetry at
  either wind condition. Fast lulls and fast gusts occur with equal frequency.
- **Strongly heavy tails (excess kurtosis ≈ +7, i.e. total κ ≈ 10)** — at BOTH
  wind conditions, not just fullwind. The pitot signal contains many more
  extreme deviations from the mean than a Gaussian distribution would produce.
  That κ ≈ 10 is well above canonical turbulence (κ ≈ 3 for velocity) and
  well above atmospheric boundary-layer values (typically κ ≈ 3–4).

## Claims and defensibility

### (1) "Tunnel TI stays below ~3 % at all measured heights"

- **Defensible.** This is a factual statement about our measured data, not
  a claim about passing any published quality threshold.
- There is **no single 5 % TI threshold** that a referee would recognise. Better
  to avoid "passes the 5 % clean-tunnel test" phrasing. Different fields use
  different benchmarks:
  - Aerospace-grade low-turbulence tunnels: TI < ~0.5 %
  - Industrial/small research tunnels: 1–5 % typical
  - Atmospheric-boundary-layer tunnels: 5–20 % by design

References for TI classification:
- **Mehta R.D. & Bradshaw P. (1979)**, "Design rules for small low speed wind
  tunnels", *Aeronautical Journal* 83, 443–449.
- **Barlow J.B., Rae W.H. & Pope A. (1999)**, *Low-Speed Wind Tunnel Testing*,
  3rd ed., Wiley.
- **Plate E.J. (ed.) (1982)**, *Engineering Meteorology*, Elsevier. Discusses
  ABL-tunnel TI ranges.

### (2) "Fluctuations are symmetric (|skew| < 0.15)"

- **Defensible.** Standard interpretation of low skewness = no systematic
  asymmetry.

### (3) "Fullwind shows more non-Gaussian intermittency than lowestwind"

- **Wrong.** Medians are close (fullwind +7.03 vs lowestwind +6.80, both
  excess). Both wind conditions are similarly heavy-tailed. Do not claim a
  monotone wind-strength effect on the tail heaviness.
- **Do not use** this claim.

### (4) "Heavy tails indicate sub-second gust intermittency in the wind"

- **Overclaim.** In canonical turbulence, velocity kurtosis ≈ 3 (excess ≈ 0)
  (Pope 2000, *Turbulent Flows*, ch. 6). Excess kurt of **+7** at the
  *velocity* level is far above typical flow values and is almost certainly
  dominated by instrument/processing response rather than flow intermittency:
  - Pitot mechanical response lag at 100 Hz
  - Tubing acoustics or transducer resonance
  - Electrical spike events (the `|skew|<5 & kurt<50` spike filter in
    `process_folder` already discards the worst of these)
  - Small sample size (~100 points per block) makes kurtosis estimation noisy
- True small-scale intermittency in Kolmogorov cascades is found in velocity
  *derivatives*, not in velocity itself (Frisch 1995, *Turbulence: The Legacy
  of A.N. Kolmogorov*).
- Independent cross-check on the water surface: the wave-probe eta time
  series measured under the same fullwind shows median excess kurtosis
  ≈ **−0.6** (exposed probes) to +0.14 (panel-sheltered) — i.e. the water
  surface driven by this airflow is **near-Gaussian**. If the airflow truly
  had κ ≈ 10 intermittency at 100 Hz, the surface would be expected to
  inherit some of it; it does not. See
  `analysis_scratch/windwave_eta_statistics_findings.md`.

**Safer wording for the thesis:**
> "The pitot signal has elevated excess kurtosis (median ≈ +7 under NI /
> Fisher convention, equivalent to κ ≈ 10), substantially above canonical
> turbulent-velocity values. An independent water-surface measurement under
> the same airflow gives near-Gaussian block statistics, so the pitot
> heavy-tail signal is most plausibly an instrument-response characteristic
> rather than a genuine property of the airflow."

### (5) Log-law fit (fig10, separate concern)

- Fit across the full height range (8 mm – 245 mm) yields
  `u* ≈ 0.02 m/s`, `z₀ ≈ 10⁻⁶⁰ m`, `R² < 0.04` at both wind conditions.
- **Honest finding**: the profile is NOT a canonical log-law over our range.
  The tunnel roof at 380 mm confines the flow and produces an approximately
  uniform-speed core above ~50 mm. The log region (if it exists) is confined
  to the lowest few tens of mm.
- Vollestad & Jensen (2021) reports log-law fits at their three highest wind
  speeds using PIV — they have finer spatial resolution in the boundary
  layer and more points in the log region.
- In the thesis: report the bad fit honestly as part of the uncertainty
  discussion. Do not selectively pick a sub-range to "get" log-law unless
  the physical reason for the cutoff is stated.

## Future work (not thesis-blocking)

- If raw time-series becomes available (not just block stats), recompute
  kurtosis on the raw signal and on its time-derivative. The derivative
  kurtosis is where true Kolmogorov intermittency shows up.
- Compare the pitot kurtosis distribution against a hot-wire reference
  measurement in the same tunnel if one is ever taken.
- Characterise the pitot + transducer frequency response to bound the
  instrument-artifact contribution to the excess kurtosis number.

## Files

- `windscripts/windprofile_combined.py` — fig10 (log-law fit) and fig11
  (turbulence character: TI, skew, excess kurt vs height).
- `windscripts/vollestad_overlay.py` — axes-aligned overlay of our fig5 with
  Vollestad's published profile (methodology-comparison figure).
- `windresults/windprofile_turbulence_character_*.pdf` — fig11 outputs.
- `windresults/windprofile_loglaw_fit_*.pdf` — fig10 outputs.
- `windresults/vollestad_comparison_overlay.png` — overlay output.
