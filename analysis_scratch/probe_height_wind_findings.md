# Probe Height, Range Mode, and Wind Background — Analysis Findings
*Original autonomous session: 2026-03-30*
*Reviewed and substantially corrected: 2026-04-17 (Ole + assistant walk-through)*

> **Review summary**: the 2026-03-30 agent reached several wrong conclusions
> by speculating about physical mechanisms (probe swaps, water disturbances,
> quantization) without checking the raw signal or the experimental log. The
> measurements (tables) are kept; the interpretations have been rewritten.
> Obvious errors in the original version are called out as they appear.

---

## Context

A bug was found and fixed during the original session:

**Bug**: `_extract_probe_height()` and `_extract_probe_range_mode()` in
`improved_data_loader.py` were called with `filename` (CSV basename) but the
`-height100`, `-height136`, `-lowrange` keywords are in the **folder name**.
Same root cause as the earlier mooring bug.

**Fix**: Pass `str(file_path)` instead of `filename` at `improved_data_loader.py:472–473`.
Pipeline re-run with `--force-recompute`.

After the fix, condition counts across the main analysis dataset
(2026 probe config):

| Condition | Runs | Folders | Notes |
|-----------|------|---------|-------|
| cond1: height272, high-range (standard) | 201 | 8 (March 7 – 21) | Baseline, no hardware mistakes |
| cond2: height136, high-range | 7 | 1 (March 23) | Small dataset; probably highrange (unconfirmed in log) |
| cond3: height100, high-range (**user error** — forgot to switch to lowrange after lowering probes) | 129 | 4 (March 23 – 26) | Hardware out of spec; P2-malfunction runs likely originate here |
| cond4: height100, low-range (**correct**) | 132 | 2 (March 26 – 27) | Used for all thesis-results figures (`meta_results`) |

**Probe identity is stable across the entire experiment**: the same four
physical ultrasound probes (numbered 1–4) were used throughout. Only their
positions changed. Any claim of "probe calibration drift between old and new
physical units" in the original doc was fabricated by the agent.

---

## Probe hardware geometry recap

| Condition | Probe height | Range mode | Window (mm) | Still-water position | Crest headroom | Trough headroom |
|-----------|-------------|------------|-------------|---------------------|----------------|-----------------|
| cond1 | 272 mm | high | 130–350 mm | 272 mm ✓ in window | +142 mm | +78 mm |
| cond2 | 136 mm | high | 130–350 mm | 136 mm ✓ just inside | +6 mm | +214 mm |
| cond3 | 100 mm | **high** (wrong) | 130–350 mm | 100 mm ✗ below min | −30 mm | +250 mm |
| cond4 | 100 mm | **low** (correct) | 30–250 mm | 100 mm ✓ centered | +70 mm | +150 mm |

---

## Finding 1: Stillwater noise floor by condition

Noise floor = (P97.5 − P2.5) / 2 of `eta` signal in nowave+nowind runs.
Values in mm.

| Condition | 9373/170 (IN) | 12400/250 (OUT) | 9373/340 (par) | 8804/250 (up) | n_runs |
|-----------|:-------------:|:---------------:|:--------------:|:-------------:|:------:|
| cond1 h272/high | **0.289 ± 0.094** | **0.263 ± 0.079** | 0.148 ± 0.111 | 0.240 ± 0.084 | 16 |
| cond2 h136/high | 0.045 (n=1) | 0.090 (n=1) | 0.045 (n=1) | 0.090 (n=1) | 1 |
| cond3 h100/high (wrong) | 0.158 ± 0.139 | 0.071 ± 0.050 | 0.127 ± 0.087 | 0.126 ± 0.085 | 12 |
| cond4 h100/low (correct) | **0.102 ± 0.082** | **0.087 ± 0.064** | 0.093 ± 0.115 | 0.064 ± 0.090 | 9 |

### Corrected interpretation

**Cond1 (h272) has the HIGHEST noise floor — this is EXPECTED physics, not
surprising.** Ultrasound time-of-flight measurement accuracy degrades with
acoustic path length: a longer column of air between probe and water
surface means more time for attenuation, beam divergence, and temperature-
gradient-induced speed-of-sound drift to accumulate. h272 has the longest
air column (272 mm) of all four conditions; h100 has the shortest. The
noise-floor ordering h272 > h136 > h100 is the ordering we'd expect on
acoustic-path-length grounds alone, independent of range-mode choice.

*The original doc attributed the h272 noise floor to "probe calibration
drift between old and new physical probes" and "residual tank motion in
March 7–21 stillwater periods". No probes were ever swapped, and agent
did not check the raw signal to confirm any tank motion. Those
explanations are removed.*

**Cond2 (h136, n=1)** is too small a dataset to interpret. The single
reported value (0.045 mm at IN) could be (a) a genuinely quiet run, or
(b) a probe readout limited by hardware quantization (if signal variation
is smaller than one reported distance step, the P97.5 − P2.5 can collapse
to a small value). With n=1 we cannot distinguish these. Mark as
**low-confidence**; do not use for comparisons.

*The original doc claimed the probe was "stuck at a quantization level"
because of the +6 mm crest headroom. Speculative — no raw-signal
inspection, no hardware-quantization measurement.*

**Cond3 (h100, wrong highrange)** noise floor is 1.5–2× higher than cond4
at most probes. This is consistent with operating below the 130 mm
acoustic-window minimum: the hardware may return plausible-looking values
but its internal averaging and nonlinearity are not calibrated for the
out-of-spec geometry. **Cond3 is also the condition from which the known
P2-malfunction runs originate** (user confirmation 2026-04-17) — the same
range-mode mistake that misaligned the acoustic window also caused
detectable probe malfunctions in the wave runs.

Within-cond3 variation: the 20260323 folder (first day with probes at
h100) shows higher IN-probe noise (0.271 mm) than later cond3 folders.
Cause unconfirmed — likely reflects either (a) continued probe
malfunction effects or (b) settling of the probe mount. *The original
doc attributed this to "water disturbance from repositioning" without
inspecting the signal; dropped as unsupported.*

**Cond4 (h100/lowrange)** is the cleanest:
- IN probe (9373/170): 0.102 mm mean
- OUT probe (12400/250): 0.087 mm
- The 20260327 folder (under9Mooring30) shows the lowest noise of the
  whole experiment: 9373/170 = 0.065 mm, 8804/250 = 0.031 mm,
  9373/340 = 0.052 mm, 12400/250 = 0.087 mm. Likely the best-settled
  water of the experiment.

---

## Finding 2: Wind background amplitude by condition

Amplitude in nowave+fullwind runs. **Probe mapping**: P1=9373/170,
P2=12400/250, P3=9373/340, P4=8804/250 (`march2026_better_rearranging`
config, all conditions 1–4).

| Condition | 9373/170 (IN) | 12400/250 (OUT) | 9373/340 (par) | 8804/250 (up) | n_runs |
|-----------|:-------------:|:---------------:|:--------------:|:-------------:|:------:|
| cond1 h272/high | 10.575 ± 0.425 | 0.907 ± 0.158 | 9.951 ± 0.620 | 8.700 ± 0.319 | 5 |
| cond3 h100/high (wrong) | 9.851 ± 0.226 | 0.857 ± 0.060 | 10.200 ± 0.700 | 8.515 ± 0.120 | 2 |
| cond4 h100/low loose230 | 8.983 ± 0.447 | 1.016 ± 0.149 | 9.034 ± 0.217 | 7.745 ± 0.182 | 4 |
| cond4 h100/low loose300 | 9.568 ± 0.045 | 0.817 ± 0.004 | 8.707 ± 0.853 | 8.442 ± 0.067 | 2 |

### Corrected interpretation

**The physical wind-wave field is approximately constant across
sessions**: the wind fan is driven by a stepless 0-to-max wheel and held
near the same mark session-to-session (measured ~5.9–6.0 m/s across the
dates). Tank water level varies by ±0.5 mm. Neither drives inter-
condition differences in reported wind amplitude.

**Amplitude differences across conditions at the IN probe reflect
probe-measurement properties, not wind-field differences.** Going from
h272/high (~10.6 mm) to h100/low (~9.2 mm) is a 1.4 mm shift over the
same physical wind. Candidate mechanisms:
- Different noise characteristics in high vs low range mode
- Probe geometry at h100 places the probe face closer to wave crests
  (~30 mm clearance at 10 mm wave + surface displacement), potentially
  altering acoustic reflection characteristics
- Small-n self-selection (2–5 runs per condition)

*The original doc concluded "wind amplitude is consistent across
conditions, a property of the wind-wave field, not the probe". This is
backwards: the wind-wave field IS consistent (fan and water-level
measurements confirm); it's the probe that reports it differently.*

**OUT probe variation (0.82–1.02 mm) has a physical cause missed by the
agent: mooring length → post-panel fetch.** When mooring lines are
longer, the front panels extend further back (downstream), shortening
the free-water fetch between the back of the panel row and the OUT probe
at 12400 mm. Less fetch → smaller wind-generated ripples at OUT.
Evidence:

| Mooring | loose230 | loose300 | difference |
|---------|----------|----------|------------|
| cond4 OUT wind amp | 1.016 mm | 0.817 mm | −0.199 mm (longer mooring → smaller ripple) |

The direction matches the physical model. The agent had the data but
did not see the pattern. **This mooring-length → post-panel fetch
mechanism is not previously documented** — added as a TODO to promote
into its own memory note.

### Addendum 2026-04-18: PSD shape comparison sharpens the picture

A direct cond1-vs-cond4 PSD comparison at all four probes
(`analysis_scratch/wind_psd_shape_cond1_vs_cond4.{py,pdf,_findings.md}`)
shows that the wind-wave **peak itself** (~3.7 Hz, 10–20 mm²/Hz at
wind-exposed probes) is essentially identical between conditions —
which **confirms** the "wind field is constant" claim above. The
amplitude difference between cond1 and cond4 at the IN probe (10.575
mm vs 9.178 mm = +1.4 mm) is **not** due to a different wind field; it
is due to a **drift skirt** below ~3 Hz that is present in cond1's
spectrum and absent from cond4's. The longer 272 mm acoustic path is
more vulnerable to slow temperature/medium drift, producing
low-frequency content that the percentile-based amplitude metric
(P97.5 − P2.5)/2 aggregates into the reported "amplitude".

This sharpens the framing: the difference is not a uniform amplitude
scaling factor between conditions — it is **frequency-localised
artefact energy below the wind band, exclusive to cond1**. Practical
consequence: cond1 IN-probe time-domain amplitudes are systematically
inflated by about 1–1.5 mm relative to the actual wind. FFT-based
OUT/IN at the paddle frequency is not affected (narrow-band window
excludes the drift skirt).

---

## Finding 3: Signal-to-noise — framing clarified

The original doc conflated three different quantities under the single
label "SNR":

| Quantity | Formula | Meaning | When it matters |
|---|---|---|---|
| **Dynamic range** | A_wind / σ_stillwater | Headroom between probe noise floor and full-wind signal | Probe hardware sanity check |
| **SNR (no-wind run)** | A_paddle / σ_stillwater | Paddle wave vs still-water noise | No-wind wave runs |
| **SNR (wind run)** | A_paddle / A_wind | Paddle wave vs wind-wave contamination at the same probe | **THE relevant metric for full-wind OUT/IN** |

**What the original doc called "Wind/SW SNR"** is the **dynamic-range**
quantity. It answers "how wide is the probe's headroom between stillwater
quiet and full-wind noisy", not "can I detect a paddle wave?" Its values
of 3–12× at OUT and 37–90× at IN describe probe headroom, and the
agent's conclusion "better SNR at h100" just means "quieter stillwater
at h100".

**The thesis-relevant SNR — the one that determines whether an OUT/IN
ratio is trustworthy under wind** — is `A_paddle / A_wind` at the OUT
probe:
- A_wind at OUT ≈ 0.85–1.02 mm (Finding 2)
- A_paddle at OUT varies by condition. At 0.2 V nowind with high
  transmission: ~5–10 mm → SNR ≈ 5–10× (workable)
- At conditions with low transmission (small OUT/IN), A_paddle at OUT
  may drop to 1–2 mm → SNR ≈ 1–2× (**wind-contaminated; be cautious**)

For the IN probe under wind, wind waves ride on top of the paddle wave,
so A_paddle at IN is hard to read from time-domain alone — this
motivates the use of FFT amplitude at the paddle frequency (see
`CLAUDE.md §16` and CH04 §4-5 figure).

*The original doc's SNR claims are removed; replaced with the table
above. Downstream conclusions that relied on the conflated definition
are not automatically correct — cond4 is preferred for other reasons
(correct range mode, lower stillwater noise floor) which survive.*

---

## Finding 4: How defective runs are actually handled

*Corrected statement replacing the original "cond3 data is physically
usable" claim.*

**Defective runs are handled by two overlapping mechanisms**:

1. **User-annotated bad-data markers** — where the researcher noticed
   a problem at the time (P2 probe malfunction, aborted run, etc.) and
   either renamed or excluded the file.
2. **Script-detected quality flags** (`processor.py::_write_quality_flags`):
   - `probe_malfunction_secondary` / `probe_malfunction_critical` —
     stuck segments or DC steps in the analysis window
   - `dropout_critical` — >2% unrecoverable NaN in IN/OUT window
   - `in_probe_low_snr` — no-wind wave_stability < 0.35 at IN probe

The default filter used by thesis figures (`apply_experimental_filters`
with `quality_flag` default) excludes `*_critical` and
`in_probe_low_snr`, keeps `probe_malfunction_secondary` (auxiliary probe
broken but IN/OUT usable).

*Cond3 data is kept wherever the run passes the quality checks;
defective cond3 runs are caught by the script. What the original doc
lacked was a clear audit trail showing per-run why each flag was (or
was not) applied.*

**Open item (new 2026-04-17)**: produce a per-run **quality-flag
audit** document that traces, for every run, which flagging layers
were evaluated and why the final `quality_flag` value came out the way
it did. This would enable confident review of individual runs and
early detection of any layer that's silently misclassifying. See
"Open items" below.

**Verified**: `meta_results` (the two thesis-result folders
20260326-`lowrange` and 20260327-`lowrange`) contains only cond4 runs.
Thesis headline results are not exposed to cond3 contamination.

---

## Finding 5: Rubber-band splash does not affect the analysis-critical folders

(Unchanged from original — this finding held up.)

- Rubber-band splash only occurred in some `under9Mooring` folders
  (not `under9Mooring30`).
- The splash, if present, would appear as non-stationary
  high-frequency bursts in the raw signal.
- The 20260327 folder (under9Mooring30, cond4 loose300) is the
  cleanest reference for per-folder wind background at h100/lowrange.

A rolling-RMS stationarity check across the fullwind nowave runs would
identify which specific runs (if any) contain the rubber-band
artifact — **not yet done**, see next steps.

---

## Recommended per-folder noise floor and wind background

For the main damping analysis (2026-03-07 onward,
`march2026_better_rearranging` config):

**Stillwater noise floor** (2× detection threshold in parentheses):

| Probe | Cond1 h272 | Cond3 h100/wrong | Cond4 h100/low |
|-------|-----------|-----------------|----------------|
| 9373/170 (IN) | ~0.29 mm (0.58 mm) | ~0.16 mm (0.32 mm) | ~0.10 mm (0.20 mm) |
| 12400/250 (OUT) | ~0.26 mm (0.52 mm) | ~0.07 mm (0.14 mm) | ~0.09 mm (0.18 mm) |
| 9373/340 | ~0.15 mm (0.30 mm) | ~0.13 mm (0.26 mm) | ~0.09 mm (0.18 mm) |
| 8804/250 | ~0.24 mm (0.48 mm) | ~0.13 mm (0.26 mm) | ~0.06 mm (0.12 mm) |

**Wind background amplitude**:

| Probe | All conditions (range) | Notes |
|-------|----------------------|-------|
| 9373/170 (IN) | 8.9–10.6 mm | Variation is probe-measurement effect (range mode / height), not wind variation |
| 12400/250 (OUT) | 0.82–1.02 mm | loose300 mooring (longer) → smaller ripple (fetch mechanism) |
| 9373/340 | 8.7–10.2 mm | Similar to IN probe |
| 8804/250 | 7.7–8.7 mm | ~1 mm below IN |

---

## Open items / next steps

1. **Per-run quality-flag audit note (HIGH — user-requested 2026-04-17)**:
   debug document enumerating each flagging layer in
   `processor.py::_write_quality_flags` and walking through how it was
   applied per run. Format: one section per layer, with a table of runs
   it flagged and the triggering metric value.

2. **Mooring-length → post-panel fetch memory note**: documented here as
   an aside; promote to its own memory file under
   `memory/physics_wavetank_mooring_fetch.md`.

3. **Rolling RMS stationarity** on nowave+fullwind runs (rubber-band
   splash detection) — not yet done.

4. **PSD comparison across conditions**: plot mean PSD for cond1 vs cond4
   at the same probe. Does probe height/range mode affect the spectral
   shape of the wind-wave field, or only the amplitude reported? If shape
   is unchanged, conditions are directly comparable.

5. **Per-folder pipeline column `wind_rms_{pos}`**: one scalar per folder
   from nowave+fullwind runs. Would enable per-folder first-motion
   threshold in `RampDetectionBrowser`.

6. **Two pre-existing traceback errors** from the 2026-03-30 recompute
   (verify status from the 2026-04-17 `run_20260417_145335_force.log`
   before closing):
   - `20251112-tett6roof`: `ensure_stillwater_columns` → `pd.to_datetime`
     format error on `file_date`
   - `20260324-under9Mooring-height100`: `find_wave_range` →
     `mstop_sec` is NaN for some run

---

## What changed in the 2026-04-17 review

- Removed the **probe-swap speculation**: no physical probes were ever
  swapped; the agent invented this to explain the h272 noise floor.
- Replaced the **"h272 is unexpectedly noisy"** narrative with the
  correct acoustic-path-length physics (longer path → more noise; the
  ordering h272 > h136 > h100 is the EXPECTED ordering).
- Removed the **h136 "stuck at quantization"** claim (speculative,
  n=1).
- Removed the **"water disturbance on 20260323"** claim (agent
  speculation without signal inspection).
- Replaced the **"wind amplitude consistent across conditions"**
  conclusion: wind field IS consistent (fan + water level confirm); the
  probe reports it differently depending on range mode and height.
- **Added the mooring-length → post-panel fetch mechanism** as the
  physical explanation for the OUT-probe wind variation between
  loose230 and loose300.
- **Rewrote Finding 3 (SNR)**: three distinct quantities disambiguated;
  "Wind/SW SNR" correctly relabelled as "dynamic range"; thesis-
  relevant SNR = A_paddle / A_wind at OUT.
- **Rewrote Finding 4**: replaced "cond3 physically usable" with a
  description of the actual quality-flagging mechanism and a TODO for
  the per-run audit.
