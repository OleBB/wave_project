# 3v3 wind-setup at OUT — investigation across March 2026

*Updated 2026-05-05.*

Self-contained writeup of the appendix table
[`output/TABLES/ch04_wind_setup_baseline_table.tex`](../output/TABLES/ch04_wind_setup_baseline_table.tex).
Source CSV: [`wind_setup_baseline_3v3_results.csv`](wind_setup_baseline_3v3_results.csv).
Generators: [`wind_setup_baseline_3v3.py`](wind_setup_baseline_3v3.py) (data),
[`wind_setup_baseline_3v3_table.py`](wind_setup_baseline_3v3_table.py) (LaTeX rendering).
Wired into `main_save_figures.py` near the other CH04 wind cells (legend §4q).

---

## Method

For each March 2026 PROCESSED dir that has both nowind and fullwind runs:

1. Sort runs chronologically by file mtime.
2. Restrict to canonical `WindCondition ∈ {no, full}` and OUT baseline
   `Stillwater Probe 12400/250 ∈ (80, 120)` mm (drops one 200.05 mm
   probe-malfunction outlier on 20260326).
3. Walk the sequence; whenever consecutive runs change WindCondition
   (e.g. no → full), record a "transition". For each transition, take
   the last `n_pre ≤ 3` runs of the BEFORE state and the first
   `n_post ≤ 3` runs of the AFTER state. Average `Stillwater Probe
   12400/250` on each side.
4. Setup magnitude `|Δη_OUT|` = water rise at OUT under wind =
   `mean(nowind) − mean(fullwind)`. ULS reads distance DOWN to water,
   so positive `mean(nowind) − mean(fullwind)` = water rose during wind.
5. Each transition is independently a "wind ON" (av → på) or
   "wind OFF" (på → av) measurement. Both directions of the same wind
   session are reported as separate rows.

---

## Headline numbers

### Per-dataset summary, all transitions

| date | mooring | n trans. | mean \|Δη\| [mm] | range [mm] |
|---|---|---|---|---|
| 20260323 | M9-23 / highrange | 4 | 1.335 | 1.133–1.570 |
| 20260324 | M9-23 / highrange | 3 | 1.383 | 1.293–1.538 |
| 20260326 | M9-23 / lowrange  | 4 | **0.884** | **0.445–1.330** |
| 20260327 | M9-30 / lowrange  | 4 | 1.296 | 1.213–1.407 |

### Strict 3+3 only (`n_pre == 3 AND n_post == 3`) — diagnostic, not in published table

| date | n trans. | mean \|Δη\| [mm] |
|---|---|---|
| 20260323 | 3 | 1.301 |
| 20260324 | 1 | 1.293 |
| 20260326 | **0** | — |
| 20260327 | 3 | 1.294 |

When restricted to clean 3+3 samples, the three well-sampled datasets
agree to **~0.01 mm between dataset means at ~1.30 mm**. 20260326 has
zero strict 3+3 transitions; its lower per-day mean is a small-sample
artefact of high transition density (4 transitions in 41 cleaned rows
left no room for 3 consecutive same-state runs around any single
transition).

The strict-3+3 cutoff was an arbitrary choice; it's diagnostic only and
is not shown in the published table. The n column in the table makes
the underlying sample sizes visible per row.

---

## Per-transition timing (chronological per day)

| date | dir | \|Δη\| | first pre → first post → last post | pre-span | gap | post-span | n |
|---|---|---|---|---|---|---|---|
| 20260323 | av→på | +1.200 | 13:02 → 13:16 → 13:20 |  8m36s |   5m18s |  3m56s | 3/3 |
| 20260323 | på→av | +1.570 | 13:20 → 16:55 → 17:03 |  3m24s | 211m32s |  7m59s | 3/3 |
| 20260323 | av→på | +1.133 | 17:26 → 17:49 → 17:54 |  9m38s |  12m41s |  4m57s | 3/3 |
| 20260323 | på→av | +1.435 | 18:51 → 19:16 → 19:20 |  7m56s |  17m27s |  3m31s | 3/2 |
| 20260324 | på→av | +1.538 | 14:56 → 15:31 → 15:48 | 18m55s |  16m08s | 17m01s | 2/3 |
| 20260324 | av→på | +1.293 | 18:38 → 19:04 → 19:09 | 14m46s |  11m07s |  4m53s | 3/3 |
| 20260324 | på→av | +1.317 | 19:30 → 19:59 → 19:59 | 11m23s |  17m52s |  0m00s | 3/1 |
| 20260326 | av→på | +0.445 | 11:50 → 12:22 → 12:25 |  3m30s |  28m37s |  2m41s | 2/2 |
| 20260326 | på→av | +0.860 | 12:22 → 13:51 → 13:58 |  2m41s |  86m11s |  7m12s | 2/2 |
| 20260326 | av→på | +0.902 | 13:51 → 14:04 → 14:19 |  7m12s |   6m04s | 14m34s | 2/3 |
| 20260326 | på→av | +1.330 | 16:44 → 17:18 → 17:18 | 15m28s |  19m01s |  0m00s | 3/1 |
| 20260327 | av→på | +1.213 | 11:26 → 11:49 → 11:52 | 14m26s |   8m57s |  2m58s | 3/3 |
| 20260327 | på→av | +1.263 | 12:38 → 15:16 → 15:26 | 11m27s | 146m58s |  9m33s | 3/3 |
| 20260327 | av→på | +1.407 | 16:26 → 16:49 → 16:58 |  7m20s |  16m07s |  9m01s | 3/3 |
| 20260327 | på→av | +1.300 | 17:58 → 18:43 → 18:43 | 16m50s |  27m27s |  0m00s | 3/1 |

`gap` = first_post − last_pre. For an ON transition, this is the time
between the last settled-nowind reading and the first wind-on reading;
the actual moment wind was turned on falls somewhere inside this gap.

---

## Observations

### O1 — Cross-dataset reproducibility is excellent under strict 3+3

Three datasets (20260323, 20260324, 20260327) agree to ~0.01 mm at
~1.30 mm under strict 3+3 sampling. The fan-dial / grid-voltage
hypothesis (continuous dial drift between days) is **not supported** by
this data — fan output looks highly reproducible across at least these
four days.

### O2 — Direction asymmetry: \|OFF\| > \|ON\| in 3 of 4 datasets

| date | \|ON\| mean | \|OFF\| mean | OFF − ON [mm] |
|---|---|---|---|
| 20260323 | 1.167 | 1.503 | +0.336 |
| 20260324 | 1.293 | 1.428 | +0.135 |
| 20260326 | 0.673 | 1.095 | +0.422 (small-sample) |
| 20260327 | 1.310 | 1.282 | −0.028 (tied) |

20260327 is the outlier in the *opposite direction*: ON ≈ OFF. The
other three days have OFF magnitude exceeding ON magnitude by
0.13–0.42 mm. *Candidate explanations (hypotheses, not directly
tested)*:
- ON measurement biased low — wind hadn't fully developed when we
  started averaging.
- OFF measurement biased high — residual setup hadn't fully relaxed
  when we started averaging.
- Both at once.

### O3 — 20260326 lower mean is small-sample, not a real day-to-day difference

20260326 had **all four transitions small-sample** (`n_pre < 3` OR
`n_post < 3`). Critically, its single late-day transition (på→av #2,
+1.330 mm with 3/1 sampling) lands on par with the other datasets'
~1.30 mm mean. The day's lower per-day mean (0.884) is dragged down by
the first three transitions, all of which were measured in tight,
sparse windows.

### O4 — Seiche bias is plausible on short-gap ON transitions

`ch04_wind_rampup` figure shows a clean **8.2 s seiche** at OUT under
wind, amplitude ~1 mm at start of wind, decaying to ~0.5 mm by ~5 min.

`Stillwater Probe 12400/250` is computed from the first ~1 s of each
recording — that captures only ~1/8 of one seiche period, so each
individual fullwind reading is **phase-biased by ±A**. For 3
random-phase samples the mean uncertainty is `A/√6 ≈ 0.4·A`:

| seiche state | uncertainty in 3-sample mean |
|---|---|
| right after wind-on (A ≈ 1 mm) | ±0.4 mm |
| 5 min into wind (A ≈ 0.5 mm)   | ±0.2 mm |

This is the same scale as our magnitude variations. The within-block
spread of the 3 fullwind readings is consistent:

| transition | post-block ULS values [mm] | spread | comment |
|---|---|---|---|
| 20260326 av→på #2 | 99.990, 99.900, 99.550 | **0.440** | wide — seiche-suspect |
| 20260327 av→på #2 | 99.880, 99.980, 100.070 | 0.190 | mild |
| 20260327 av→på #1 | 99.470, 99.470, 99.560 | 0.090 | tight |
| (most others)    | …                       | < 0.20 | tight |

**Worst-case transition for seiche bias: 20260326 av→på #2** — only
6 min gap (wind only just established when we sampled), and post-block
spans 14m34s with 0.44 mm spread between readings. That's exactly the
signature you'd expect.

Long-gap ON transitions (≥ 15 min) and all OFF transitions are below
the 0.2 mm bias threshold, so they're robust.

### O5 — Bracketing nowind-nowave drift is at noise floor

Settled-to-settled drift across each day's bracketing nowind-nowave
runs is +0.013 to +0.027 mm (well inside OUT probe noise floor
0.13 mm). Wind setup is fully reversible — once wind stops and the
tank settles, OUT level returns to its pre-wind value to better than
probe noise.

---

## Why this method was chosen

The earlier rampup/decay cross-direction analysis used 2 s zero-windows
+ 5 s wind-windows on single transition recordings. It produced a
**direction-asymmetric ~0.6 mm gap** (decay > rampup) that turned out
to be a windowing artefact: the 2 s and 5 s averaging windows weren't
the same physical state in the two directions, and the post-fans-off
operator-walk-back time landed in the wind-window for decay.

The 3v3 method uses **fully-settled stillwater means** on both sides
of the wind state change, separated by the actual moment wind was
turned on/off. Both directions of the same wind session can be
measured, and direction asymmetry then reflects physical asymmetry
between turn-on and turn-off transients (O2), not windowing.

---

## Caveats currently flagged in the table's IMMUTABLE block

1. **Direction asymmetry** (O2). 0.1–0.4 mm bias in 3 of 4 datasets.
2. **Small-sample artefact** on 20260326 (O3).
3. **Seiche bias** on short-gap ON transitions (O4). Worst case
   20260326 av→på #2.
4. **Pipeline anchor convention** — `Stillwater Probe 12400/250` for
   fullwind runs is anchored to the first 1 s of THAT run (which may
   itself be wind-on); we report raw ULS readings on both sides so
   the difference is correct, but the absolute baseline column is not
   a deviation from a settled-tank reference.

---

## What would close the open questions

These are listed in priority order; none are urgent for the appendix
table to stand on its own:

- **O2 direction asymmetry**. Pull the per-run η(t) signal for the
  3 fullwind runs of one ON transition vs the 3 fullwind runs of the
  paired OFF transition. If the ON-side fullwind mean is biased low,
  it should improve as we extend the averaging window past the first
  1 s. Conversely, OFF-side bias would show up as a slowly-decaying
  baseline within the post-block recording.
- **O4 seiche-bias on 20260326 av→på #2**. Quick check: load the
  three fullwind runs' η(t), Welch-PSD them, see if the 0.122 Hz
  (8.2 s) peak amplitude is consistent with the 0.44 mm spread we
  observed in the means.
- **More datasets**. Adding pre-March datasets (20260307, 20260312–14,
  if they have both nowind and fullwind runs) would widen the
  reproducibility test from n=4 to n=8+ days. Cheap.

---

## Files

| file | role |
|---|---|
| `analysis_scratch/wind_setup_baseline_3v3.py` | Data: per-transition CSV (with mtimes) |
| `analysis_scratch/wind_setup_baseline_3v3_table.py` | LaTeX: appendix table |
| `analysis_scratch/wind_setup_baseline_3v3_results.csv` | per-transition data (input to table) |
| `analysis_scratch/wind_setup_baseline_3v3_investigation.md` | this writeup |
| `output/TABLES/ch04_wind_setup_baseline_table.tex` | thesis appendix table |

Memory: [`memory/finding_wind_setup_delta_eta_per_dataset.md`](../../memory/finding_wind_setup_delta_eta_per_dataset.md) — older
narrative, currently flags "3v3 supersedes earlier cross-direction
framing" with the per-dataset ~1.30 mm headline.
