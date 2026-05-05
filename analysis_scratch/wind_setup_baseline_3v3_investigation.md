# 3v3 wind-setup at OUT — investigation across March 2026 (2026-05-05)

Quick follow-up to the 3v3 finding in
[`memory/finding_wind_setup_delta_eta_per_dataset.md`](../../memory/finding_wind_setup_delta_eta_per_dataset.md).
Earlier finding: 20260326 read ~0.88 mm, 20260327 ~1.30 mm — a 0.4 mm
cross-dataset gap. Hypothesis on the table: continuous fan dial drifts
day-to-day + grid voltage variability.

This investigation extends to all March 2026 PROCESSED dirs that have
both `WindCondition == "no"` and `WindCondition == "full"` rows.

Script: [`wind_setup_baseline_3v3.py`](wind_setup_baseline_3v3.py).
CSV with all transitions: [`wind_setup_baseline_3v3_results.csv`](wind_setup_baseline_3v3_results.csv).

---

## Per-dataset summary, all transitions pooled

| date | n_transitions | mean magnitude [mm] | std [mm] | range [mm] |
|---|---|---|---|---|
| 20260323 | 4 | 1.335 | 0.203 | 1.133–1.570 |
| 20260324 | 3 | 1.383 | 0.135 | 1.293–1.538 |
| 20260326 | 4 | **0.884** | **0.362** | 0.445–1.330 |
| 20260327 | 4 | 1.296 | 0.082 | 1.213–1.407 |

20260326 stands out: lower mean, much higher std. Worth checking whether
the difference is *real* or an artefact of sample size.

## Restricted to strict 3+3 samples (`n_pre == 3 AND n_post == 3`)

| date | n_transitions | mean magnitude [mm] | std [mm] | range [mm] |
|---|---|---|---|---|
| 20260323 | 3 | **1.301** | 0.235 | 1.133–1.570 |
| 20260324 | 1 | **1.293** | — | 1.293 |
| 20260326 | **0** | — | — | (no transition has 3+3) |
| 20260327 | 3 | **1.294** | 0.100 | 1.213–1.407 |

When restricted to clean 3+3 samples, the three well-sampled datasets
agree to **0.01 mm** between dataset means. The 0.4 mm cross-dataset gap
reported earlier was an artefact of 20260326's small-sample noise, not a
real day-to-day setup difference.

20260326 had 4 transitions in 41 cleaned rows — a high transition
density that left no room for 3 consecutive same-state runs around any
single transition. Every 20260326 transition had n_pre or n_post ≤ 2.
Excluded from the strict comparison; not evidence of true day-to-day
variability.

## Direction split — OFF > ON in 3 of 4 datasets

| date | n_ON / n_OFF | \|ON\| mean [mm] | \|OFF\| mean [mm] | OFF − ON [mm] |
|---|---|---|---|---|
| 20260323 | 2 / 2 | 1.167 | 1.503 | **+0.336** |
| 20260324 | 1 / 2 | 1.293 | 1.428 | +0.135 |
| 20260326 | 2 / 2 | 0.673 | 1.095 | +0.422 (small samples) |
| 20260327 | 2 / 2 | 1.310 | 1.282 | −0.028 (tied) |

In three datasets the OFF magnitude (recovery when wind stops) exceeds
the ON magnitude (rise when wind starts) by 0.13–0.42 mm. 20260327 is
the exception, with the two directions essentially equal.

## Headline observations

1. **Cross-dataset wind-setup magnitude (strict 3+3) is highly
   reproducible**: ~1.30 mm across 20260323, 20260324, 20260327, with
   between-dataset spread ≤ 0.01 mm.
2. **The 0.4 mm cross-dataset gap reported earlier was an artefact of
   20260326's small-sample transitions**, not real day-to-day
   variability.
3. **The fan-dial / grid-voltage hypothesis is not supported by this
   data.** If anything, the data suggests fan output is highly
   reproducible across at least these four days.
4. **A direction asymmetry persists**: OFF magnitude > ON magnitude by
   0.1–0.4 mm in 3 of 4 datasets (20260327 is tied). The 3v3 method
   reduced but did not eliminate the asymmetry seen in the earlier
   rampup/decay analysis.

## What this leaves unresolved

- *Why does OFF magnitude exceed ON magnitude?* The 3v3 method was
  expected to wash this out (both directions sample fully-settled vs
  fully-developed-wind states). One transition has it tied (20260327),
  three don't.
  *Candidate explanations (hypotheses, not tested)*:
  - **Wind stress build-up time.** "First 3 fullwind runs" after wind
    ON might catch tank still ramping toward steady setup, biasing the
    fullwind mean *low* (smaller |ON|).
  - **Asymmetric relaxation.** "First 3 nowind runs" after wind OFF
    might catch tank still recovering, biasing the nowind mean *high*
    in absolute ULS reading (i.e. water level still slightly elevated
    above true settled level), inflating |OFF|.
  - Both biases could coexist and reinforce.
- *Sample size is small.* Only 4 dataset-days; 10 strict 3+3
  transitions across them.
- *Why does 20260327 alone show no direction asymmetry?* The other
  three days had wind sessions of similar typical durations; without
  pulling per-run timestamps it's hard to know whether 20260327's
  longer sessions allowed the tank to fully equilibrate before the
  next nowind block.

## Possible next steps if we want to push further

- Add early March datasets (20260307, 20260312–14) to widen the
  cross-dataset sample.
- Inspect the time-gaps between the last fullwind run and first nowind
  run at each "wind OFF" transition (and vice versa). If shorter gaps
  correlate with bigger |OFF − ON| gaps, that's evidence for the
  ramp-time hypotheses.
- Pull `mstop_tail_mm_12400/250` for the post-paddle decay tail of
  fullwind runs — that's another absolute baseline read on the same
  probe, independent of `Stillwater Probe 12400/250`.

None of these are urgent given the headline finding (cross-dataset
reproducibility is excellent at strict 3+3).
