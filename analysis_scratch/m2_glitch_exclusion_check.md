# M2 — Did the rolling-RMS-flagged glitch runs inflate the cond1 wind average?

**Date**: 2026-04-18 (free-time exploration follow-up)
**Backing script**: ad-hoc; output captured below.
**Source data**: `analysis_scratch/rolling_rms_stationarity.csv` (16 nowave+fullwind runs across all dates) +
`probe_height_figure.py`'s cond1/cond4 cuts.

## Question

`probe_height_wind_findings.md` Finding 2 (open item): "were the
above_50 glitchy runs included in the 10.575 ± 0.425 mm cond1 wind
average?" If so, re-compute excluding them.

`memory/MEMORY.md` notes that the rolling-RMS check found three
regimes — severe sensor-glitch (max/med > 3.5+, ±60 mm spikes in η,
NOT physical), likely rubber-band splash (max/med 1.5–3.5), and normal
wind variation.

## Severity classification of the 16 nowave+fullwind runs

Threshold: `9373/170_max_med > 3.5` ⇒ severe glitch. Between 1.5 and
3.5 ⇒ likely rubber-band. Below 1.5 ⇒ normal.

| regime | count | dataset distribution |
|---|---|---|
| severe | 3 | All cond1 (above_50 mooring), 20260307 / 20260314 dates |
| likely_rubber_band | 14 | mix of cond1, cond3, cond4 — cond4 is here too |
| normal (max/med < 1.5) | **0** | none of the runs qualify |

**Key observation:** every nowave+fullwind run in the dataset is at
least "likely_rubber_band" by this metric. There is no "clean" baseline
in the data — the rubber band threshold is hit by all runs, and severe
glitches are confined to cond1 above_50 dates.

## Effect on the §3b figure averages

`Probe 9373/170 Amplitude` (the IN probe, time-domain percentile):

| condition | filter | n | mean (mm) | std (mm) |
|---|---|---|---|---|
| cond1 h272/high | all runs | 5 | **10.575** | 0.425 |
| cond1 h272/high | exclude severe (max/med > 3.5) | 2 | 10.453 | 0.718 |
| cond4 h100/low | all runs | 6 | **9.178** | 0.460 |
| cond4 h100/low | exclude severe | 6 | 9.178 | 0.460 |

Excluding the three severe-glitch cond1 runs drops the cond1 average
from 10.575 to 10.453 mm — a 0.12 mm reduction. The cond1−cond4 gap
shrinks from 1.40 mm to 1.28 mm. The bulk of the difference survives
glitch exclusion.

The same pattern appears at the other wind-exposed probes
(9373/340: 9.95 → 9.41 mm; 8804/250: 8.70 → 8.54 mm). No probe shows
a glitch-driven dominant artifact — the gap is real (or, per
`wind_psd_shape_cond1_vs_cond4_findings.md`, a drift-skirt artefact
that is present in *all* cond1 runs, not just the severe ones).

## Conclusion

**The 1.4 mm cond1-vs-cond4 difference is NOT primarily caused by the
above_50 glitchy runs.** Excluding them changes the cond1 mean by only
~1%. The PSD shape comparison (today's
`wind_psd_shape_cond1_vs_cond4` analysis) gives the better
explanation: cond1 has a drift skirt below ~3 Hz that is present in
*all* cond1 runs and absent from cond4, contributing systematic
spurious low-frequency energy that the time-domain percentile metric
aggregates as "amplitude".

The §3b figure values (10.575 ± 0.425 mm cond1 vs 9.178 ± 0.460 mm
cond4 at the IN probe) are correct as reported; the difference is real
in the time-domain metric, just not from the severe-glitch runs and
not from a real wind-amplitude difference.

## Side observation

Severe glitch (max/med > 3.5) is confined to **cond1 above_50 mooring**
on 20260307 and 20260314. It does not appear at cond3 or cond4 (the
later h100 dates). This is consistent with the longer-air-path
mechanism — the same physical reason cond1 has the drift skirt may
also be why cond1 occasionally produces severe spike events when the
medium has a transient gradient or droplet near the probe face.

## Status

- [x] Question answered: glitch exclusion changes cond1 average by ~1%, not 10%.
- [x] Cross-references the PSD-shape finding (today, separate file).
- [ ] No code changes — the §3b figure caption is correct as written.
