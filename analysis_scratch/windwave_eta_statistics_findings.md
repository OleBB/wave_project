# Wave-probe surface-elevation statistics under wind-only, no-wave runs

Date: 2026-04-20
Scope: wave-probe analogue of `windscripts/windprofile_combined.py` fig11 —
σ, skewness, excess kurtosis of the eta_{pos} surface-elevation time series
under wind-only no-wave runs, to test whether the pitot's excess kurt ≈ +3.5
is physical intermittency of the airflow or an instrument/processing artifact.

Script:   `analysis_scratch/windwave_eta_statistics.py`
Figure:   `analysis_scratch/windwave_eta_statistics.pdf`
CSV:      `analysis_scratch/windwave_eta_statistics_summary.csv`

## What was measured

For each wind-only no-wave run and each probe position, on the zeroed
`eta_{pos}` time series with the first 5 s trimmed:

| Quantity        | Meaning                                                 |
|-----------------|---------------------------------------------------------|
| σ_η     [mm]    | std of surface elevation — wind-driven roughness        |
| skew            | asymmetry of the surface-elevation PDF                  |
| excess kurt     | κ − 3 — heavy-tailedness (Gaussian = 0)                  |
| (mean−med)/σ    | same asymmetry sanity check used for the pitot          |

Filter: `WaveFrequencyInput [Hz]` is NaN · `WindCondition ∈ {full, lowest}` ·
`run_category == "nowave_control"` (excludes `fromZeroToMaxWind`,
`fromMaxToZero`, experimental ramps, probe-malfunction partials).

## Runs available in the dataset

| Subset                          | Count |
|---------------------------------|-------|
| fullwind + nowave + standard    | 16    |
| lowestwind + nowave             | **0** |

All 44 lowestwind runs in `combined_meta` have paddle waves. There is no
wind-only lowestwind data to compare against. This is a dataset gap, not a
filter mistake — the pitot/lowestwind comparison simply has no wave-probe
counterpart. All numbers below are fullwind only.

## Measured numbers (medians across 16 fullwind runs)

| Probe       | σ_η [mm] (med, range) | Skew (med) | Excess kurt (med, range)      | Role                 |
|-------------|----------------------|------------|-------------------------------|----------------------|
| `8804/250`  | 3.77  (3.27 – 4.15)  | +0.21      | −0.63 (−0.86 … +2.79)         | upstream, exposed    |
| `9373/170`  | 4.36  (3.80 – 4.96)  | +0.25      | −0.65 (−0.86 … +15.3*)        | IN, wall-side, exposed |
| `9373/340`  | 4.28  (3.75 – 4.65)  | +0.20      | −0.69 (−0.81 … +1.24)         | parallel, far-side, exposed |
| `12400/250` | 0.35  (0.28 – 0.44)  | +0.02      | +0.14 (−0.08 … +0.76)         | OUT, panel-sheltered |

(*) Four outlier points with excess kurt > 3 (clipped in the figure, drawn as
'×') are all from 2026-03-07 / 2026-03-14 / 2026-03-19 — before the probe was
lowered to height100 on 2026-03-21. At the pre-height100 setting the probe
sat close to the water surface and dropouts under fullwind are already a
known issue (see `memory/known_baddata_ultrasound_16hz.md` / `MEMORY.md`).
Excluding those four runs, every probe's excess kurt falls inside |κ−3| < 1.

## Direct comparison to pitot stats (from 2026-04-20 session)

| Quantity              | Pitot fullwind (n=36) | Wave probes fullwind (n=16, excl. 4 outliers) |
|-----------------------|----------------------|------------------------------------------------|
| Skewness (median)     | +0.01                | +0.20 (exposed), +0.02 (sheltered)             |
| Excess kurt (median)  | **+3.44**            | **−0.6 to −0.7** (exposed), +0.14 (sheltered)   |

The water surface under the same wind is **not** heavy-tailed. It is
mildly **sub-Gaussian** (flatter than normal) on the three exposed probes.

## Verdict: pitot artifact vs physical intermittency

- **Defensible**: the surface elevation under the tunnel's fullwind is
  close to Gaussian, with a small positive skew and a mildly sub-Gaussian
  (negative excess kurt) PDF on all three exposed probes. Sheltered OUT is
  near-Gaussian — consistent with noise-floor-dominated signal at σ ≈ 0.35 mm.
- **The pitot's excess kurt ≈ +3.5 is not matched in the water surface it
  drives.** If the airflow truly had κ = 6.5 intermittency in its velocity
  fluctuations, the surface — which is forced by that same airflow — would
  be expected to inherit at least some heavy-tailedness. It does not.
- **Best supported conclusion**: the pitot excess-kurt signal is dominated
  by instrument response, not flow physics. Candidates (already listed in
  `wind_statistics_claims_summary.md`): pitot mechanical lag at 100 Hz,
  tubing acoustics, transducer resonance, residual electrical spikes.
- **Safer wording for the thesis**: the pitot's heavy tails should be
  reported as an instrument characteristic. The water-surface PDF under
  the same wind is near-Gaussian and does not support a claim of
  sub-second velocity-level intermittency.

## Caveats

- Water is a low-pass filter. Airflow with κ = 6.5 at 100 Hz could still
  drive a σ-scale surface with κ ≈ 3 because the surface integrates
  energy across frequencies and the highest-frequency pitot events
  dissipate before coupling to any surface mode. So the wave-probe result
  is not a *proof* the airflow is Gaussian — it is a proof that **the
  excess kurt of the pitot is not reproduced on any surface-observable
  quantity**, which is what the thesis would otherwise have tried to
  claim. The artifact-vs-intermittency uncertainty shifts in the
  direction of "artifact".
- Sample size is small (n=16). The medians are stable but the ranges
  are not a rigorous distributional claim.
- No comparison for lowestwind is possible — see dataset gap above.
- The panel-sheltered probe σ_η ≈ 0.35 mm is ~2.5× the 0.14 mm
  stillwater noise floor of `12400/250` (CLAUDE.md §16). So even the
  sheltered signal is detectably above noise — the sub-panel wind
  leakage or downstream air circulation is not zero.

## Future work (not thesis-blocking)

- Record a wind-only nowave run with a pitot + wave probes simultaneously
  and compare their block-statistics on a second-by-second basis.
- If pitot raw time series becomes available, compute the same block
  statistics on the pitot and compare. Could confirm or rule out
  instrument resonance at specific frequencies.
- Revisit the pre-height100 outliers (2026-03-07, 14, 19) with the
  dropout-detection logic — they may have undetected spikes dominating
  the kurtosis number.
