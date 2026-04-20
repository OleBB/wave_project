# Wind-statistics claims — what we can and cannot defend

Date: 2026-04-20
Scope: what the pitot-based wind stats (TI, skewness, kurtosis, mean/median) in
`windscripts/windprofile_combined.py` fig11 actually let us claim in the thesis,
and what references those claims lean on.

## Measured numbers (from fig11, 2026-04-20)

| Quantity                | Full vind (red, n=36)  | Laveste vind (green, n=34) |
|-------------------------|-----------------------|----------------------------|
| TI [%]                  | median 1.24 (0.6–2.3) | median 1.71 (1.2–4.3)      |
| Skewness                | median +0.01          | median +0.04               |
| Excess kurtosis (κ − 3) | median +3.44 (0.7–5.2)| median +3.77 (1.7–5.0)     |

All three metrics are means across 1-second blocks within each run, then
aggregated across all runs that measured a given height. Blocks are sampled at
100 Hz → each block carries ~100 samples.

## What the numbers mean

- **Low TI (<3 %)** — the fluctuation magnitude relative to the mean is small at
  both wind conditions, and low compared to typical fan-driven flows.
- **Symmetric fluctuations (|skew| ≲ 0.15)** — no systematic gust asymmetry at
  either wind condition. Fast lulls and fast gusts occur with equal frequency.
- **Heavy tails (excess kurtosis ≈ +3.5, i.e. total κ ≈ 6.5)** — at BOTH wind
  conditions, not just fullwind. The pitot signal has significantly more
  extreme deviations from the mean than a Gaussian distribution would produce.

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

- **Wrong.** Initially claimed based on misreading fig11. Actual numbers show
  lowestwind is *slightly* more peaked than fullwind (medians 3.77 vs 3.44).
  Both are similarly non-Gaussian.
- **Do not use** this claim.

### (4) "Heavy tails indicate sub-second gust intermittency in the wind"

- **Overclaim.** In canonical turbulence, velocity kurtosis ≈ 3
  (Pope 2000, *Turbulent Flows*, ch. 6). Excess kurt of +3.5 at the *velocity*
  level is unusual and is at least as likely to be an instrument/processing
  artifact as real flow intermittency:
  - Pitot mechanical response lag at 100 Hz
  - Tubing acoustics or transducer resonance
  - Electrical spike events (the `|skew|<5 & kurt<50` spike filter in
    `process_folder` already discards the worst of these)
  - Small sample size (~100 points per block) makes kurtosis estimation noisy
- True small-scale intermittency in Kolmogorov cascades is found in velocity
  *derivatives*, not in velocity itself (Frisch 1995, *Turbulence: The Legacy
  of A.N. Kolmogorov*).

**Safer wording for the thesis:**
> "Elevated excess kurtosis (median ≈ +3.5) indicates non-Gaussian
> fluctuations in the pitot signal. Whether this reflects genuine small-scale
> intermittency of the airflow or instrument-response characteristics of the
> pitot probe cannot be resolved from block statistics alone."

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
