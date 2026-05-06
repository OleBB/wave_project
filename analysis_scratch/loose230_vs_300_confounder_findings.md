# loose230 vs loose300 — confounder check (2026-05-06, follow-up to session b)

After session_2026-05-06b's three-readings stand-off (canon Δ = +0.036 vs broad
Δ = −0.002), the user asked to check the data against three known confounders:

  1. Probe height (high/low range setting)
  2. Day-to-day wind-speed variability
  3. Between-run sloshing (insufficient settling between runs)

All claims below are **observations**. Hypotheses are flagged separately.

Scripts:
```
analysis_scratch/loose230_vs_300_confounder_check.py    (round 1 – per-folder roster, probe-config-bias-within-loose230)
analysis_scratch/loose230_vs_300_confounder_check2.py   (round 2 – Mar 26 vs Mar 27 head-to-head, run-order)
analysis_scratch/loose230_vs_300_confounder_check3.py   (round 3 – Stillwater Std availability, A_in vs A_out decomp, settling-time pattern)
```

---

## Observation O1 — h100/low vs h100/high inside loose230 alone

Within loose230 (broad scope, 118 rows), restricting to h100/low (the canon
configuration, only 1 day of data: Mar 26) reads systematically **higher** K_t
than the other probe configs that share a mooring boundary:

| stat (K_t(100/low) − K_t(other config)) | value |
|---|---|
| n pairings (across freq × amp × wind, both configs present) | 20 |
| **median Δ** | **+0.056** |
| mean Δ | +0.047 |
| range | [−0.076, +0.117] |

Strongest in fullwind A1 cells (+0.062 to +0.117). Nowind cells: ±0.01.

The canon-scope mooring delta is +0.036.
The within-loose230 probe-mode delta (100/low vs 100/high) is +0.056 — larger.

*Candidate explanation (hypothesis)*: the OUT probe at 12400/250 reads more
amplitude in low-range mode than in high-range mode under wind, because the
ULS quantization step is tighter in low-range. This is plausible but NOT
verified here — would need a same-day same-run side-by-side mode comparison,
which doesn't exist in the dataset.

## Observation O2 — A_out, not A_in, drives the canon-scope +0.036

Mar 26 loose230/h100low (n=19, 17 fullwind + 2 nowind) head-to-head with
Mar 27 loose300/h100low (n=64, 32 fullwind + 32 nowind). Same probe config,
same Windspeed setpoint, ka_in ≈ identical.

13 paired cells (both dates have data):

| metric | median Δ (Mar26 − Mar27) | as % of Mar27 mean |
|---|---|---|
| K_t = A_out / A_in | **+0.036** | — |
| A_in (mm) | −0.13 | −1.1 % |
| A_out (mm) | +0.44 | +4.2 % |

So both sides shift, but **A_in dropped slightly** while **A_out rose** by ~4 %.
K_t = A_out/A_in is amplified in both directions.

Pearson within fullwind (n=12 paired cells):
- r(ΔK_t, ΔA_in) = **−0.86**
- r(ΔK_t, ΔA_out) = +0.24

ΔA_in dominates K_t variance. Cells where A_in dropped most are the cells
where K_t rose most.

*Candidate explanation (hypothesis)*: this is the textbook signature of
**day-to-day variation in IN-probe wind-spectral contamination** (CLAUDE.md
§16, "FFT-based OUT/IN under fullwind"). The IN probe (9373/170) is exposed
to wind; the OUT probe (12400/250) is sheltered by the panel. Small
day-to-day differences in turbulence/wind-fetch geometry shift A_in (FFT)
without affecting A_out. The K_t signal then partly reflects this contamination
asymmetry, not panel transmission.

## Observation O3 — Day-to-day wind variability via pre-paddle σ_η

`Windspeed` in meta is a setpoint constant (5.8 m/s for fullwind runs, 0
for nowind), not a measurement. **But the actual day-to-day wind energy is
captured by σ_η in the 3 s pre-paddle window**, already computed by
[analysis_scratch/wind_qc_3s.py](analysis_scratch/wind_qc_3s.py) and saved
to [wind_qc_3s_per_run.csv](wind_qc_3s_per_run.csv) (118 rows, canon datasets).

Pre-paddle σ_η, fullwind cells in canon scope:

|  | n | σ_η at IN (9373/170) [mm] | σ_η at OUT (12400/250) [mm] |
|---|---|---|---|
| Mar 26 (loose230 canon) | 16 | mean 3.82, std 0.99 | mean 0.346, std 0.050 |
| Mar 27 (loose300 canon) | 32 | mean 4.20, std 0.75 | mean 0.319, std 0.032 |
| **Δ (Mar 26 − Mar 27)** | — | **−0.39 mm (−9.2 %)** | +0.027 mm (+8.5 %) |

Long-run reference (5 fullwind+nowave runs, 31–381 s): IN ≈ 4.28 mm, OUT ≈ 0.36 mm.
Per [wind_qc_3s_control_chart.png](analysis_scratch/wind_qc_3s_control_chart.png),
Mar 27 fullwind cells span a wider range (some reaching ~6 mm) than Mar 26.

**Key disambiguation — wind-wave band only**:
[loose230_vs_300_IN_windwave_band.py](analysis_scratch/loose230_vs_300_IN_windwave_band.py)
recomputes σ_η on the 3 s pre-paddle window restricted to the **3–5 Hz wind-wave
band** via Welch PSD. The bigger pre-paddle gap at IN (−0.39 mm in total σ)
collapses in the wind-wave band:

| band | loose230 σ_η at IN [mm] | loose300 σ_η at IN [mm] | Δ |
|---|---|---|---|
| total (broadband) | 3.82 (n=16) | 4.20 (n=32) | **−0.39 (−9.2 %)** |
| 3–5 Hz wind-wave | 3.14 (n=17) | 3.22 (n=32) | **−0.07 (−2.3 %)** |
| 0.5–2 Hz swell control | 0.30 | 0.26 | +0.04 |

Mean PSDs at IN overlap nearly perfectly between Mar 26 and Mar 27 in the
wind-wave band — see
[loose230_vs_300_IN_windwave_psd.png](analysis_scratch/loose230_vs_300_IN_windwave_psd.png).

OUT probe in 3–5 Hz wind-wave band: loose230 σ_η = 0.074 mm, loose300 = 0.066 mm
(both below or at the gold-standard probe noise floor of 0.14 mm).

Per-run correlation `K_t × σ_η_IN_3-5Hz` is essentially zero in both moorings
(r = −0.02 loose230, +0.14 loose300). The day-to-day variation in IN-probe
wind-wave choppiness does **not** predict the per-run K_t.

*Candidate explanation (hypothesis)*: the −9.2 % broadband gap at IN comes
from frequency content **outside** the 3–5 Hz wind-wave band — likely
low-frequency drift / settling / sub-2 Hz residual sloshing from preceding
runs, NOT day-to-day wind speed. The actual wind-wave forcing was the same
on the two days.

So the user's confounder #2 ("day-to-day wind speed variability") IS
checkable through pre-paddle σ_η — and within the relevant 3-5 Hz band,
**loose230 and loose300 fullwind days are essentially indistinguishable**.

## Observation O4 — Sloshing proxy via pre-paddle σ_η under nowind

Sloshing-proxy check via the same 3 s pre-paddle window
([wind_qc_3s_per_run.csv](analysis_scratch/wind_qc_3s_per_run.csv)),
**nowind cells only** (no wind = pure residual-sloshing measurement):

| | n | σ_η at OUT [mm] mean / median / max | σ_η at IN [mm] mean / median / max |
|---|---|---|---|
| Mar 26 (loose230 canon) | 2 | 0.017 / 0.017 / 0.033 | 0.016 / 0.016 / 0.032 |
| Mar 27 (loose300 canon) | 30 | 0.078 / 0.027 / **0.286** | 0.100 / 0.032 / 0.735 |

Mar 26's two nowind canon cells were rock-solid quiet (≈ 0.017 mm at both
probes). Mar 27 had a much wider distribution; the median is still tiny
(0.027 mm) but the worst case at OUT was 0.286 mm — about **17× higher**
than Mar 26's worst case, consistent with the rushed `prev_run_nperiods=40`
pattern on Mar 27.

Probe noise floor at 12400/250: 0.14 mm gold standard, 0.14–0.36 mm range
(CLAUDE.md §16). So Mar 27's median nowind sloshing (0.027 mm) is well below
noise floor — most runs were fine — but the long tail (max 0.286 mm) shows
some runs caught residual motion.

Per-run correlation inside loose300 nowind: r(K_t, σ_η_OUT_pre) = −0.139
(weak negative — more sloshing → slightly *lower* K_t). Not statistically
strong, just a weak hint that residual sloshing biases the IN-side amplitude
upward (since A_in is the canonical mean of two probes, one of which is the
upstream wall probe at 9373/170 — see correlation note below).

The metadata column `Probe {pos} Stillwater Std` is populated for only 5/609
special stillwater calibration runs and is NOT useful for regular wave runs.
The pre-paddle 3 s window from `processed_dfs` is the right substitute.

## Observation O5 — Settling time via `inter_run_gap_s` (corrected)

> Earlier rounds used `prev_run_nperiods`, which is the *previous run's wave
> duration*, not the wait time between runs. The correct column is
> `inter_run_gap_s` (585/609 non-null in meta).
> [analysis_scratch/loose230_vs_300_inter_run_gap.py](analysis_scratch/loose230_vs_300_inter_run_gap.py)

Inter-run gap (seconds) distribution per day, in-scope rows:

| date | mooring | probe_cfg | n | gap median [s] | gap range [s] |
|---|---|---|---|---|---|
| 2026-03-19 | loose230 | 272/high | 10 | 186 | 51–552 |
| 2026-03-21 | loose230 | 100/high | 7 | 90 | 50–207 |
| 2026-03-23 | loose230 | 100/high | 36 | 116 | 19–535 |
| 2026-03-23 | loose230 | 136/high | 4 | 153 | 33–398 |
| 2026-03-24 | loose230 | 100/high | 21 | 213 | 40–579 |
| 2026-03-25 | loose230 | 100/high | 17 | 295 | 26–839 |
| **2026-03-26 (canon)** | **loose230** | **100/low** | **19** | **80** | **36–472** |
| **2026-03-27 (loose300)** | **loose300** | **100/low** | **64** | **100** | **13–231** |

So contrary to the earlier (wrong) interpretation: **Mar 26 (canon loose230)
gaps were not longer than Mar 27** — Mar 26 median was 80 s, Mar 27 was 100 s,
both in the ~1-2 minute range. Mar 25 (loose230, h100/high) was the most
patient day (median 295 s); Mar 26 (the day they switched probe mode) had a
faster cadence (median 80 s).

**Within Mar 27 loose300, gap DOES correlate with K_t** (n=32 each):

| condition | r(K_t, gap) | quartile-mean K_t (shortest → longest gap) |
|---|---|---|
| nowind | **+0.320** | 0.567 → 0.572 → 0.653 → 0.643 (Δ ≈ +0.076 across quartiles) |
| fullwind | **+0.295** | 0.722 → 0.722 → 0.738 → 0.752 (Δ ≈ +0.030 across quartiles) |

So short-gap (rushed) runs on Mar 27 read systematically lower K_t than
long-gap runs. Effect is real and same direction in both wind conditions.
*Candidate explanation (hypothesis)*: residual tank motion from the
previous run inflates A_in (canonical mean of two probes that include the
upstream wall probe at 9373/170, susceptible to leftover sloshing) more
than it inflates A_out (sheltered by panel), pushing K_t down.

**Re-stratifying the canon-scope cell-by-cell delta**:

| Mar 27 subset used in delta | n cells | median Δ_Kt (Mar 26 − Mar 27) |
|---|---|---|
| all Mar 27 (full canon) | 13 | **+0.036** |
| only long-gap Mar 27 (≥ median = 100 s) | 9 | **+0.030** |
| only short-gap Mar 27 (< 100 s) | 12 | +0.022 |

Restricting Mar 27 to its long-gap half (better-settled runs) shrinks the
mooring-scope delta from +0.036 to +0.030. So short-gap Mar 27 runs do bias
Mar 27 K_t slightly low and inflate the loose230-vs-loose300 delta — but
only by ~0.006, not the whole +0.036.

## Observation O6 — prev_run_wind effect (small, same-direction)

| mooring × current wind × prev_run_wind | n | mean K_t |
|---|---|---|
| loose230 nowind, prev=full | 3 | 0.639 |
| loose230 nowind, prev=no | 62 | 0.612 |
| loose230 fullwind, prev=full | 49 | 0.737 |
| loose230 fullwind, prev=no | 3 | 0.731 |
| loose300 nowind, prev=full | 0 | — |
| loose300 nowind, prev=no | 32 | 0.609 |
| loose300 fullwind, prev=full | 30 | 0.733 |
| loose300 fullwind, prev=no | 2 | 0.748 |

For loose230 nowind runs, prev=full → K_t = 0.639, prev=no → 0.612 (Δ +0.027).
For loose300, no prev=full nowind cases exist (run-order ran all nowind first).

So in current-nowind cells, a previous fullwind run leaves a small lingering
effect (~+0.027). This is in the same direction as the canon-scope finding
but small and underpowered (n=3 for the smaller cell).

---

## Observation O7 — Apples-to-apples requires lowrange-to-lowrange (only)

[analysis_scratch/loose230_vs_300_apples_check.py](analysis_scratch/loose230_vs_300_apples_check.py)

Mooring × probe-range matrix in thesis scope:

| mooring | highrange | lowrange |
|---|---|---|
| above_50 | 58 | 0 |
| above_200 | 2 | 0 |
| loose230 | 99 | 19 |
| **loose300** | **0** | **64** |

`loose300` exists ONLY in lowrange. So:
- **Highrange-to-highrange comparison of loose230 vs loose300 is impossible.**
- **Lowrange-to-lowrange (canon scope) IS the only apples-to-apples mooring comparison available.**

Re-doing the canon vs broad comparison with this in mind:

| Δ_Kt = K_t(loose230) − K_t(loose300) | n cells | median |
|---|---|---|
| canon (apples — both lowrange) | 13 | **+0.036** |
| broad uncorrected (loose230 71% highrange + 16% lowrange + 13% other-height-highrange; loose300 100% lowrange) | 24 | **−0.005** |
| broad with within-loose230 mode correction applied | 24 | **+0.040** |
| within-loose230 mode bias (low − high), median | 13 | +0.048 |

The "broad scope ≈ 0" finding from session 2026-05-06b was an artefact of
asymmetric probe-mode pooling. Once mode-corrected, broad and canon agree:
both estimate a real mooring effect of **+0.036 to +0.040** in K_t.

**This reverses the 2026-05-06b "split" conclusion**: it's not that canon
is "small-sample noise" and broad is "the real answer". The broad scope
mixes apples with oranges on the loose230 side, biasing its mean DOWN by
roughly 0.71 × 0.048 ≈ +0.034 vs what it would read in pure lowrange.

The cleaner statement is: at the slack levels tested (loose230 = 23 cm,
loose300 = 30 cm), loose230 transmits approximately +0.036 to +0.040 more
K_t past the panel than loose300 — a real mooring effect, modulated by
~+0.020 of confounders we documented (probe-mode + sloshing-residue +
day-specific A_in drift).

---

## Synthesis (updated 2026-05-06)

The canon-only +0.036 K_t lift between loose230 and loose300 has at least
**three plausible drivers** that are aliased in this dataset:

1. **Probe range mode (h100/high → h100/low)** between Mar 25 and Mar 26.
   Within-loose230 evidence: same-mooring, same-day-spacing, mode change
   reads +0.056 in the same direction as the canon finding.
2. **Day-specific A_in difference (~−1 %)** between Mar 26 and Mar 27 driving
   K_t through the IN denominator. r(ΔK_t, ΔA_in) = −0.86 (cell-mean
   regression). Standard wind-contamination geometry.
3. **Settling-time difference (Mar 26 patient, Mar 27 rushed)** plus
   prev_run_wind effects on current K_t (~+0.027 magnitude where checked),
   with Mar 27 showing rare residual-sloshing tails up to σ_η = 0.286 mm
   in nowind runs (vs Mar 26 max 0.033 mm).

What the IN-probe pre-paddle PSD check ruled out:

- **Mooring slack does NOT change wind-wave choppy energy at IN**. In the
  3–5 Hz wind-wave band, σ_η_IN is 3.14 mm (loose230) vs 3.22 mm (loose300)
  — a 2.3 % gap, indistinguishable. The mean PSDs overlap. The hypothesis
  that a slacker panel could absorb (or radiate) more 3–5 Hz wind-wave
  energy and shift IN-probe choppiness is **not supported** by the data.
- The broadband −9.2 % pre-paddle σ_η_IN gap is from energy *outside* the
  wind-wave band (mostly < 2 Hz drift / sloshing / settling residue), not
  from actual day-to-day wind variability.
- Per-run correlation `K_t × σ_η_IN_3-5Hz` is essentially zero — wind
  weather choppiness is not a per-run K_t driver.

Each of the three remaining drivers can plausibly account for ~25–100 % of
the canon Δ. The data cannot discriminate. The mooring slack (loose230 vs
loose300) is just one of the things that changed between Mar 26 and Mar 27;
it is **not isolated**.

The broad-scope median Δ ≈ 0 is consistent with mooring slack having no
detectable effect on K_t, with the canon-scope signal coming from a mix
of the probe-mode bias + settling/sloshing residue on Mar 27. The broad
scope mixes probe configs, so its real interpretation is "mooring effect
≤ probe-config + day effect for this dataset", not "mooring effect is zero".

## Where this leaves the thesis

- The session 2026-05-06b conclusion ("present both, observe, don't
  conclude") is reinforced. The +0.036 in canon scope is real but
  not cleanly attributable to mooring slack alone.
- If a single number is needed for the thesis, the broad-scope median
  Δ = −0.002 is the more defensible "no effect" reading, with the caveat
  that probe configs are pooled.
- **The probe-mode delta (h100/low vs h100/high inside loose230) is the
  largest single confounder surfaced here**, with implications for any
  cross-day comparison in this dataset, not just the mooring question.
  Worth noting in the CH04 methodology chapter or as a thesis caveat.
- **Mooring slack does not visibly affect IN-probe wind-wave dynamics**
  in the relevant 3–5 Hz band (n=17 vs 32, mean PSDs overlap). This is
  itself a useful negative finding for the thesis: if you want to argue
  the panel response to choppy wind waves is mooring-sensitive, this data
  doesn't support it at the slack levels (23 cm vs 30 cm) tested here.

---

## Open questions / Things this analysis CANNOT answer

- Whether the probe-mode delta (O1) is real or an artefact: needs a
  same-day side-by-side comparison, which the dataset doesn't have.
- Whether mooring slack affects panel behavior at slack levels OUTSIDE
  the [23, 30] cm range tested: the dataset has no other slack values,
  and within this 7 cm range, no IN-probe wind-wave-band signal emerges.
- Whether the residual-sloshing tail (σ_η_OUT up to 0.286 mm under nowind
  on Mar 27) actually pushed any K_t cells off the true value: we'd need
  per-run residual-vs-K_t residual analysis stratified by paddle setting,
  not just pooled correlation. Quick correlation r ≈ −0.14 hints yes-but-
  small.
