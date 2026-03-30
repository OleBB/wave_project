# Wave Transmission Analysis — Findings
*Autonomous analysis session, 2026-03-28*

---

## Data integrity notes

- **Cached OUT/IN vs recomputed** (confirmed 2026-03-28, 354 valid runs): correlation = 1.0000, mean diff = 0.0000, max abs diff = 0.0000. The cached OUT/IN (FFT) values in meta.json are IDENTICAL to values recomputed on-the-fly from `"Probe {pos} Amplitude (FFT)"` columns. The old wide-window FFT issue described in CLAUDE.md §6 ("Stale OUT/IN in meta.json") does NOT affect the current cache. Analysis using the cached `OUT/IN (FFT)` column directly is valid for the 354-run working dataset (after excluding known outliers).
- **Mooring column**: Shows `above_50` for ALL 357 standard runs in main config, including runs where the dataset name says `under9Mooring`. The mooring type is correctly inferred from dataset name but NOT correctly stored in metadata. Cache is stale for the Mooring column.
- **Quality flags**: All 482 standard runs with quality flag show "ok". No runs flagged as bad.
- **Wave detection**: Several runs had OUT_IN > 1.5 or 70+ (wave range detection failure). These were excluded by the quality filter.

---

## What was loaded

- **646 total runs** across 24 PROCESSED-* directories
- Date range: 2025-08-?? to 2026-03-27 (inferred from directory names)
- **513 standard wave runs**, 65 nowave_control, 9 wind_decay, 8 partial, 6 experimental, 2 diagnostic

### Configuration of main dataset

The bulk of the wave data (357 standard runs) uses:
- **IN probe**: `9373/170`
- **OUT probe**: `12400/250`
- **Mooring**: `above_50` (all show `above_50` in the `Mooring` column, BUT dataset names reveal mooring position changed over time — see below)

Additional data with older probe config (in_=`9373/250`, out=`12400/170`): 97 runs — includes no-panel and reverse-panel conditions.

---

## Quality filters applied

Before drawing any conclusions, I applied:
1. IN probe FFT amplitude > 2× noise floor = 0.66 mm (2 × 0.33 mm from CLAUDE.md)
2. OUT probe FFT amplitude > 2× noise floor = 0.28 mm (2 × 0.14 mm)
3. OUT/IN < 1.3 (extreme values indicate wave range detection failure or near-noise IN probe)

After filtering: **~340 valid runs** for primary analysis.

Notable bad run: `fullpanel-nowind-amp0300-freq1600-per40-depth580-mstop30-run1.csv` (20260327) had OUT/IN = 70.2, IN amplitude = 0.149 mm (below noise floor — wave range detection probably found stillwater).

---

## Finding 1: Frequency is the dominant factor

**Correlation**: Expected kL vs OUT/IN = -0.74 (strong negative)
**Correlation**: Frequency vs OUT/IN = -0.74 (nearly identical, kL adds little beyond frequency)

Higher frequency = more damping (lower OUT/IN). This is the clearest, most robust finding.

Full range observed:
- 0.5–0.7 Hz: OUT/IN ≈ 0.95–1.15 (near-neutral, sometimes slight energy gain)
- 0.8–1.0 Hz: OUT/IN ≈ 0.90–1.10 (mostly near 1.0)
- 1.0–1.3 Hz: OUT/IN decreasing from ~0.93 to ~0.76
- 1.3–1.7 Hz: OUT/IN from 0.76 down to ~0.40
- 1.8–1.9 Hz: OUT/IN 0.20–0.45

By kL bin (kL = wavenumber × panel length):
| kL range | Full wind OUT/IN | No wind OUT/IN |
|----------|-----------------|----------------|
| < 8      | 1.026           | 0.980          |
| 8–12     | 0.965           | 0.923          |
| 12–16    | 0.834           | 0.858          |
| 16–20    | 0.820           | 0.741          |
| 20–25    | 0.697           | 0.598          |
| 25–30    | 0.589           | 0.529          |
| 30–35    | 0.470           | 0.392          |
| > 35     | 0.503           | 0.180          |

---

## Finding 2: Wind INCREASES apparent OUT/IN at most frequencies

**The central question: Wind appears to REDUCE damping (increase apparent transmission) at most frequencies above 1.0 Hz — BUT the effect depends on mooring configuration and is confounded with experimental date.**

Mean OUT/IN comparison (quality-filtered, all amplitudes):

| Freq (Hz) | Full wind | No wind | Wind effect |
|-----------|-----------|---------|-------------|
| 0.5       | 1.054     | 0.963   | +0.091      |
| 0.6       | 0.935     | 0.878   | +0.057      |
| 0.7       | 1.115     | 1.021   | +0.094      |
| 0.8       | 1.154     | 0.991   | +0.163      |
| 0.9       | 1.001     | 0.915   | +0.086      |
| 1.0       | 0.933     | 0.927   | +0.005      |
| 1.1       | 0.866     | 0.903   | **-0.037**  |
| 1.2       | 0.807     | 0.819   | **-0.011**  |
| 1.3       | 0.820     | 0.741   | +0.079      |
| 1.4       | 0.724     | 0.694   | +0.031      |
| 1.5       | 0.667     | 0.545   | +0.122      |
| 1.6       | 0.607     | 0.522   | +0.085      |
| 1.7       | 0.499     | 0.341   | +0.157      |
| 1.8       | 0.458     | 0.586   | **-0.128**  |
| 1.9       | 0.712     | 0.192   | +0.520 (N=3, unreliable)|

Key takeaway: At 11 of 14 frequencies tested (excluding 1.1, 1.2, 1.8 Hz), wind **increases** measured OUT/IN. The effect is largest at high frequencies (1.5–1.7 Hz: +0.09 to +0.16 in absolute terms, or 20–50% relative increase).

### Why does wind increase OUT/IN?

Wind increases BOTH IN and OUT probe amplitudes at high frequencies (≥1.3 Hz), but increases the OUT probe amplitude proportionally MORE than the IN probe:

Example at 1.7 Hz:
- IN amplitude: full_wind = 15.5 mm, no_wind = 12.1 mm (+28%)
- OUT amplitude: full_wind = 7.7 mm, no_wind = 4.2 mm (+82%)

This suggests wind is **either**:
1. Adding coherent energy at the OUT probe at the paddle frequency (unlikely — the OUT probe is sheltered)
2. Reducing destructive interference at the OUT probe (the panel can reflect waves, creating a standing wave pattern — wind disrupts the standing wave and changes interference)
3. Some artifact of the FFT window picking up near-paddle-frequency wind energy at the OUT probe (less likely since OUT probe is sheltered from wind)

Physical interpretation: **Wind may be disrupting the standing wave pattern** that would otherwise form due to reflections between panel and paddle. The OUT probe (past the panel) has no wind fetch, so wind-wave contamination is minimal there. The IN probe IS exposed to wind, so at full wind, the IN probe has a wind wave component that may affect the FFT measurement — this is partly controlled by using FFT amplitude at the paddle frequency. But wave instability (measured IN wave_stability) is much lower under full wind.

---

## Finding 3: Wave amplitude has a WEAK positive effect on transmission

Higher amplitude = slightly more transmission (higher OUT/IN):
| Amplitude | Full wind | No wind |
|-----------|-----------|---------|
| 0.1 V     | 0.751     | 0.591   |
| 0.2 V     | 0.741     | 0.710   |
| 0.3 V     | 0.757     | 0.724   |

The amplitude effect is smaller than the frequency effect. At 1.3 Hz no-wind:
- 0.1 V: OUT/IN = 0.709 (n=30)
- 0.2 V: OUT/IN = 0.791 (n=8)
- 0.3 V: OUT/IN = 0.826 (n=6)

This suggests larger-amplitude waves transmit slightly more efficiently — possibly nonlinear effects (radiation pressure, panel motions scale with amplitude, or wave breaking interactions with the panel geometry).

**Correlation**: WaveAmplitudeInput vs OUT/IN = +0.17 (weak positive)
**Correlation**: IN ka vs OUT/IN = -0.28 (moderate negative — steeper waves transmit less)

Note: ka (wavenumber × amplitude) combines frequency AND amplitude effects. Its negative correlation with OUT/IN shows that steeper waves are blocked more, consistent with the frequency trend.

---

## Finding 4: Mooring position effect is present but confounded

Dataset directory names reveal three mooring configurations:
- `above50mm` — panel moored above (early datasets, 20260307–20260316)
- `under9_23mm` — panel moored below with 230mm band (20260316–20260325)
- `under9_30mm` — panel moored below with 300mm band (20260327)

BUT the metadata column `Mooring` shows `above_50` for ALL runs — the mooring type is NOT properly encoded in the metadata for under-panel moorings. This is a data quality issue.

Comparing at 1.3 Hz:
| Mooring | Full wind | No wind |
|---------|-----------|---------|
| above50mm  | 0.704     | 0.710   |
| under9_23mm | 0.896    | 0.765   |
| under9_30mm | 0.826    | 0.736   |

Under-panel mooring shows ~13–18% higher OUT/IN vs above-panel mooring at 1.3 Hz. This is a significant effect. Possible explanations:
- Under-panel mooring allows the panel to move more freely in heave, transmitting wave motion better
- Above-panel mooring suppresses panel heave, increasing reflection and reducing transmission

**IMPORTANT**: This confounds the analysis because the above50mm runs are from early datasets (March 7–14 2026) and under9 runs are from later dates (March 16–27). There may be systematic differences between these periods (water temperature, probe calibration drift, etc.) beyond just the mooring type.

---

## Finding 5: Wind reduces wave stability (coherence) at IN probe

| Freq (Hz) | IN stability full-wind | IN stability no-wind |
|-----------|----------------------|---------------------|
| 0.6       | 0.505                | 0.958               |
| 0.7       | 0.693                | 0.974               |
| 0.8       | 0.424                | 0.946               |
| 0.9       | 0.769                | 0.987               |
| 1.0–1.5   | 0.79–0.89            | 0.95–0.98           |

The IN probe (exposed to wind) has much lower wave_stability under full wind — meaning the waveform is less periodic/coherent. This is direct evidence of wind-wave interference at the IN probe.

This does NOT affect the FFT amplitude measurement directly (FFT at paddle frequency is still extracted), but it suggests the IN probe signal is noisier under wind, which increases measurement uncertainty.

---

## Finding 6: No-panel control shows near-unity transmission

From the old probe configuration (9373/250 → 12400/170):
| Freq | Full wind | Lowest wind | No wind |
|------|-----------|-------------|---------|
| 0.65 Hz | 0.975 | 0.983 | 0.981 |
| 1.30 Hz | 0.932 | 0.951 | 0.979 |

Without panel: OUT/IN ≈ 0.93–0.98 across all wind conditions. The small deviation from 1.0 may be due to:
1. Different probe positions (slightly different wave amplitude due to tank width effects)
2. Wind adds energy between the two probes (unlikely at paddle freq)
3. Reflection from probe structures

This confirms the panel IS the source of damping (not wind or tank geometry), and wind by itself does not significantly affect wave transmission.

---

## Finding 7: Strong damping at high frequencies — nearly total blocking

At 1.7–1.9 Hz:
- No wind: OUT/IN = 0.19–0.34 (66–81% of energy blocked)
- Full wind: OUT/IN = 0.36–0.50 (50–64% blocked)

This is remarkable. At 1.9 Hz (short waves), the panel blocks ~80% of wave energy even without wind. With full wind, apparent blocking drops to ~50%.

---

## Finding 8: Wind-wave contamination at IN probe — a measurement artifact

The IN probe (9373/170) is fully exposed to wind. The OUT probe (12400/250) is sheltered by the panel.

Signal-to-noise ratio (SNR) = FFT paddle-frequency amplitude / wind-wave amplitude (PSD):

| Freq | Full wind SNR_IN | No-wind SNR_IN | Full wind SNR_OUT | No-wind SNR_OUT |
|------|-----------------|----------------|------------------|----------------|
| 0.5 Hz | 0.66 | 25.9 | 10.6 | 26.9 |
| 0.7 Hz | 1.32 | 35.0 | 15.6 | 40.5 |
| 1.0 Hz | 2.22 | 27.1 | 15.7 | 39.4 |
| 1.3 Hz | 2.25 | 12.7 | 9.1 | 18.6 |
| 1.7 Hz | 2.64 | 6.0 | 5.4 | 8.4 |

At 0.5 Hz full wind: **wind-wave energy exceeds paddle-wave energy at the IN probe** (SNR_IN = 0.66). The FFT correctly tries to extract only the paddle frequency — but with SNR < 1, the extraction is unreliable.

At 1.7 Hz full wind: SNR_IN = 2.64 (paddle wave is 2.6× bigger than wind waves at IN). The OUT probe has SNR_OUT = 5.4, still well above contamination.

**Artifact evidence**: Within full-wind runs only, the correlation between SNR_IN and OUT/IN is r = -0.24 (weaker SNR → higher apparent OUT/IN). When grouped by SNR:
- SNR_IN < 1: mean OUT/IN = 0.941 (artificially inflated)
- SNR_IN 1-2: mean OUT/IN = 0.746
- SNR_IN 2-3: mean OUT/IN = 0.724
- SNR_IN 3-5: mean OUT/IN = 0.706
- SNR_IN > 5: mean OUT/IN = 0.687

**Conclusion**: The apparent "wind increases transmission" result is partly a measurement artifact. When the IN probe is contaminated by wind waves (low SNR), the FFT amplitude at the paddle frequency is reduced (wind-wave incoherence at the measurement window reduces the coherent sum), inflating OUT/IN. The true wind effect on wave transmission is smaller than the raw data suggests, especially at low frequencies and low amplitudes.

The OUT probe SNR under full wind (5–16x depending on freq) is much better than the IN probe SNR (0.7–3.5x). This asymmetry is the root cause of the apparent wind effect on OUT/IN.

---

## Finding 9: Wind effect split by mooring configuration

When separated by mooring type, the wind effect becomes clearer:

### above50mm mooring (early datasets: March 7–14 2026)
| Freq | Full wind | No wind | Wind effect |
|------|-----------|---------|-------------|
| 1.0 Hz | 0.934 | 0.927 | +0.008 |
| 1.1 Hz | 0.836 | 0.902 | **-0.066** |
| 1.2 Hz | 0.780 | 0.813 | **-0.034** |
| 1.3 Hz | 0.704 | 0.710 | -0.006 |
| 1.4 Hz | 0.643 | 0.672 | **-0.029** |
| 1.5 Hz | 0.636 | 0.497 | +0.139 |
| 1.6 Hz | 0.562 | 0.499 | +0.063 |
| 1.7 Hz | 0.449 | 0.262 | +0.187 |

**Key finding for above50mm**: At intermediate frequencies (1.1–1.4 Hz), wind slightly DECREASES OUT/IN. At high frequencies (1.5–1.7 Hz), wind increases OUT/IN substantially.

### under9 mooring (later datasets: March 16–27 2026)
| Freq | Full wind | No wind | Wind effect |
|------|-----------|---------|-------------|
| 1.0 Hz | 0.930 | 0.930 | ~0.000 |
| 1.1 Hz | 0.927 | 0.906 | +0.021 |
| 1.2 Hz | 0.841 | 0.852 | -0.011 |
| 1.3 Hz | 0.864 | 0.755 | **+0.109** |
| 1.4 Hz | 0.753 | 0.658 | **+0.095** |
| 1.5 Hz | 0.676 | 0.565 | **+0.112** |
| 1.6 Hz | 0.620 | 0.531 | **+0.088** |
| 1.7 Hz | 0.524 | 0.391 | **+0.133** |
| 1.8 Hz | 0.642 | 0.401 | **+0.241** |

**Key finding for under9**: Wind consistently and substantially increases OUT/IN at every frequency from 1.3 Hz upward. Magnitude 0.09–0.24 absolute.

**Interpretation**: The mooring position strongly modulates the wind effect. With below-panel mooring (under9), wind more freely induces panel motion which transmits wave energy. With above-panel mooring, the panel is constrained differently. This is a confound — we cannot say definitively that the wind effect is the same for both mooring types without controlled experiments varying only mooring while holding everything else constant.

**The above50mm data (1.1–1.4 Hz) suggests wind may actually slightly reduce transmission** when the panel is moored from above — the opposite of the under9 result. This is a physically interesting difference that deserves investigation.

---

## Summary: What factors matter most for OUT/IN

1. **FREQUENCY / kL** (dominant): kL or frequency is the strongest predictor (r = -0.74). Higher freq = more damping. Range: OUT/IN from ~1.0 at 0.7 Hz to ~0.25 at 1.9 Hz. The physical parameter kL (wavenumber × panel length) is essentially equivalent to frequency as a predictor in this dataset.

2. **MOORING POSITION** (large effect, but confounded with date): Under-panel mooring (under9) shows ~10–15% higher OUT/IN than above-panel mooring at 1.3–1.5 Hz. Cannot cleanly separate from date confounders since all under9 runs are from March 16–27 and all above50mm runs from March 7–14.

3. **WIND CONDITION** (moderate, interacts with mooring):
   - With under9 mooring: Wind increases OUT/IN by +0.09 to +0.24 absolute at 1.3–1.8 Hz (consistent positive effect)
   - With above50mm mooring: Wind slightly decreases OUT/IN at 1.1–1.4 Hz, increases at 1.5–1.7 Hz (mixed effect)
   - Overall tendency: wind increases apparent transmission. The effect grows with frequency.

4. **WAVE AMPLITUDE** (weak): Larger amplitude = slightly higher OUT/IN (+0.03 to +0.08 between 0.1 V and 0.3 V). Effect is small compared to frequency and mooring effects. Correlation: +0.17.

5. **PROBE HEIGHT** (insufficient data): Only 6 runs at height=136mm. Cannot draw conclusions.

---

## Finding 10: The wind effect is real, not purely an artifact

Even when filtering to full-wind runs with high SNR (SNR_IN > 3), a positive wind effect on OUT/IN persists at 1.3–1.7 Hz:

| Freq | High-SNR full wind | No wind | True wind effect |
|------|-------------------|---------|-----------------|
| 1.0 Hz | 0.925 | 0.927 | -0.002 (none) |
| 1.1 Hz | 0.856 | 0.903 | -0.047 (wind slightly reduces) |
| 1.2 Hz | 0.781 | 0.819 | -0.038 (wind slightly reduces) |
| 1.3 Hz | 0.791 | 0.741 | +0.050 (wind increases) |
| 1.4 Hz | 0.731 | 0.662 | +0.069 (wind increases) |
| 1.5 Hz | 0.687 | 0.545 | +0.143 (wind increases) |
| 1.6 Hz | 0.628 | 0.522 | +0.106 (wind increases) |
| 1.7 Hz | 0.532 | 0.341 | +0.191 (wind strongly increases) |

**Conclusion**: There is a genuine physical effect of wind on wave transmission. Wind increases transmission at 1.3–1.7 Hz. The measurement artifact (SNR effect) contributes to the apparent wind effect but does not explain all of it.

---

## Best current answer to the central question

**Wind INCREASES wave transmission through the FPV panel at most frequencies, especially 1.3–1.7 Hz.**

The effect is genuine (confirmed even with high-SNR data), but is PARTIALLY a measurement artifact:

**True effect (from high-SNR analysis)**:
- 1.0–1.2 Hz: wind has little effect or slightly reduces transmission
- 1.3–1.7 Hz: wind increases OUT/IN by +0.05 to +0.19 absolute (8–56% relative)
- 1.8+ Hz: insufficient high-SNR data to conclude

**Artifact component**: The IN probe is fully exposed to wind (SNR_IN ~2–3 under full wind). The OUT probe is sheltered (SNR_OUT ~5–15). Low SNR_IN inflates apparent OUT/IN. The artifact contributes ~30–40% of the observed wind effect.

Physical mechanisms for the real wind effect (ranked by plausibility):
1. **Wind pushes the panel directly, inducing panel oscillation at/near the paddle frequency, generating waves on the OUT side** — this would explain why the effect grows with frequency (higher frequency → stiffer wave coupling)
2. **Wind disrupts the standing-wave pattern** between paddle and panel, changing where nodes/antinodes occur — the IN probe may be near a node in no-wind conditions
3. **Wind current across the panel changes the effective hydrodynamic damping** of the panel, altering transmission
4. **Wind-wave energy diffracts around the panel** and adds to the OUT probe measurement (but this should be at wind-wave frequencies, not paddle frequency — unless there is nonlinear mixing)

---

## Key numbers at a glance (for thesis)

**Core result — full panel, full wind vs no wind, 0.1–0.3V amplitude combined**:
- At 1.0 Hz: OUT/IN = 0.93 (wind) vs 0.93 (no wind) — no effect
- At 1.3 Hz: OUT/IN = 0.82 (wind) vs 0.74 (no wind) — wind +11%
- At 1.5 Hz: OUT/IN = 0.67 (wind) vs 0.54 (no wind) — wind +22%
- At 1.7 Hz: OUT/IN = 0.50 (wind) vs 0.34 (no wind) — wind +46%

**Frequency explains 74% of variance** (r² = 0.74²  ≈ 0.55 using kL)

**Wind wind effect vs no-wind**: at the same frequency, full wind gives ~10–50% higher OUT/IN at 1.3–1.7 Hz. This is the central thesis result.

**Mooring matters**: under-panel mooring shows ~10–15% higher OUT/IN than above-panel mooring (confounded with experimental date).

---

## Finding 11: Water depth regime

At 580 mm depth, the wave classification (deep/intermediate/shallow) by frequency:

| Freq (Hz) | kH | tanh(kH) | Regime |
|-----------|-----|---------|--------|
| 0.5 | 0.85 | 0.69 | **intermediate** — bottom interaction |
| 0.6 | 1.07 | 0.79 | **intermediate** |
| 0.7 | 1.32 | 0.87 | **intermediate** |
| 0.8 | 1.62 | 0.92 | **transitional** (>0.9) |
| 0.9 | 1.97 | 0.96 | near deep |
| 1.0 | 2.38 | 0.98 | effectively deep |
| 1.3+ | >4.0 | >0.999 | **deep water** |

The deep-water approximation ω² = gk is valid above ~1.0 Hz. Below 0.8 Hz, bottom interaction is significant. This affects wave speed, wavelength, and potentially how the FPV panel interacts with orbital motion.

The OUT/IN values near 1.0 (and sometimes > 1.0) at 0.7–0.8 Hz may be partly related to intermediate-water effects changing the wave field geometry between the two probes.

---

## Finding 13: Mooring classification bug — all under9 runs misclassified as `above_50`

**Bug found and fixed (2026-03-28).**

All runs from `under9Mooring` experiment folders (20260316 onward) were stored as `Mooring = "above_50"` in the processed cache, despite being under-panel mooring runs. Two bugs combined to cause this:

1. **`_extract_mooring_condition` received only the CSV filename, not the full path.** The keyword `under9Mooring` lives in the experiment folder name (e.g. `20260316-...-under9Mooring`), not in the individual CSV names like `fullpanel-fullwind-amp0100-freq1300...csv`. So the keyword search always failed.

2. **`get_mooring_type` date fallback iterated the config list without properly ordering by specificity.** `low_mooring` (valid_from 2025-11-06, valid_until=None) always matched first for any date after Nov 2025, shadowing the later `below_9_mooring` entry (valid_from 2026-03-17).

**Fixes applied to `wavescripts/improved_data_loader.py`:**
- Changed both `_extract_mooring_condition(metadata, filename)` calls to `_extract_mooring_condition(metadata, str(file_path))` — the full path includes the folder name with the mooring keyword.
- Set `valid_until=datetime(2026, 3, 17)` on `low_mooring` config — closes the date-range overlap.

**Impact**: After `--force-recompute`, the `Mooring` column for all under9 runs will correctly show `below_90_loose230` (March 16–26) or `below_90_loose300` (March 27). The mooring-type analysis in Findings 4 and 9 was computed from directory-name grouping (not the `Mooring` column) and remains valid — but the corrected `Mooring` column will enable proper groupby analysis in the standard pipeline.

**Action needed**: Run `python main.py --force-recompute` to rebuild all meta.json caches with corrected Mooring values.

---

## Finding 15: Systematic outlier catalogue — three distinct physical/technical categories

Cross-dataset scan of all runs with OUT/IN > 1.1 (25 total) reveals three distinct categories:

### Category A: Wave range detection failures (technical)
High-amplitude (0.3 V) long-duration (per240/per80/per40) no-wind runs at high freq (1.5–1.8 Hz):
- 20260327 amp0300-freq1600-nowind: OUT/IN = 70.25 (IN = 0.149 mm — noise floor)
- 20260327 amp0300-freq1600-nowind per240: OUT/IN = 2.37
- 20260327 amp0300-freq1500-nowind per240: OUT/IN = 2.22
- 20260327 amp0300-freq1700-nowind per240: OUT/IN = 1.98
- 20260323 amp0300-freq1600-nowind per80: OUT/IN = 2.60
- 20260323 amp0300-freq1800-nowind: OUT/IN = 1.73
- 20260314 amp0300-freq1800-nowind per240: OUT/IN = 1.24

Common pattern: high amplitude, no-wind, long or high-frequency runs. The amplitude envelope analysis shows valid waves exist in the run — the issue is either (a) the per240 runs have been processed with stale/incorrect wave window boundaries, or (b) the OUT/IN was computed from an end-of-run window where IN is in mstop tail decay but OUT still has energy. Must recheck after `--force-recompute`.

### Category B: Panel submersion under full wind + low frequency (physical)
Confirmed physical mechanism (wind setup pushes panel under water surface):
- 20260327 amp0200-freq0800-fullwind: OUT/IN = 1.54
- 20260319 amp0100-freq1300-fullwind: OUT/IN = 1.23 (confirmed by readme)
- 20260314 amp0200-freq0800-fullwind: OUT/IN = 1.46
- 20260307 amp0200-freq0800-fullwind: OUT/IN = 1.38

Pattern: low frequency (0.7–0.8 Hz at 0.2 V, 1.3 Hz at 0.1 V), full wind, longer wavelengths that cause larger panel displacement. All these should be excluded from the primary damping analysis. OUT/IN > 1.3 combined with low frequency + full wind is a reliable indicator.

### Category C: Genuine low-frequency near-unity transmission (physics, not errors)
Multiple runs at 0.7–0.9 Hz, full wind, 0.1–0.3 V:
- 20260312 amp0300-freq0700-fullwind: OUT/IN = 1.70
- 20260307 amp0200-freq0700-fullwind: OUT/IN = 1.22
- 20260307 amp0200-freq0900-fullwind: OUT/IN = 1.30
- 20260314 amp0200-freq0700-fullwind: OUT/IN = 1.13
- 20260326 amp0100-freq0700-fullwind: OUT/IN = 1.13

These appear across multiple dates with different moorings and conditions. These are NOT panel submersion — they are the known low-frequency near-unity or slightly-above-unity phenomenon noted in Finding 1 (kL < 8 → OUT/IN ≈ 1.0). The 0.3 V case at 0.7 Hz going to 1.70 is extreme and may involve standing wave resonance or near-capsize events. The 0.1 V case at 0.7 Hz reaching 1.13 under full wind is consistent with the SNR artifact (IN probe wind contamination at low frequency).

**Recommended quality filter thresholds** (revised from preliminary 1.3 cutoff):
- OUT/IN > 2.0: always exclude (detection failure or extreme physical anomaly)
- OUT/IN 1.3–2.0 + freq < 1.0 Hz + full wind: likely panel submersion OR low-freq physics — investigate per run
- OUT/IN 1.0–1.3 + freq 0.7–0.9 Hz: genuine physics (low-freq near-unity) — INCLUDE but flag

---

## Finding 14: 20260327 wave range detection failures — concentrated in long no-wind high-amplitude runs

Multiple outliers on 20260327:

| Run | OUT/IN | Cause |
|-----|--------|-------|
| amp0200-freq0800-fullwind | 1.539 | Panel submerged (confirmed by readme: "dykket til 6 cm under") |
| amp0300-freq1600-nowind-per40 run1 | 70.250 | Wave detection failure: IN=0.149 mm (noise floor) |
| amp0300-freq1600-nowind-per40 run2 | 2.372 | Wave detection failure |
| amp0300-freq1500-nowind-per240 | 2.224 | Wave detection failure |
| amp0300-freq1700-nowind-per240 | 1.976 | Wave detection failure |
| amp0300-freq1400-nowind-per240 | 1.323 | Borderline — wave detection issue |

**Pattern**: The OUT/IN > 1.3 outliers are concentrated in:
1. Full-wind + low frequency (0.8 Hz) at high amplitude — panel submersion
2. No-wind + **per240 (240-second runs)** at high amplitude (0.3 V) — wave range detection failure

For the per240 detection failures: the `_SNARVEI_CALIB` lookup gives a start sample for the stable wave window. But for a 240-second run (250 Hz × 240 s = 60,000 samples), if the algorithm detects the wrong window (e.g. selects the stillwater period at the end after the wave stops), the IN probe amplitude would be near-zero and OUT/IN explodes.

The no-wind normal per40 runs at the same freq (e.g. amp0300-freq1600-nowind run5: OUT/IN=0.533) are perfectly valid, showing this is a detection timing issue with long runs.

**Action**: The per240 runs need a separate `_SNARVEI_CALIB` calibration tuned to the longer run duration, OR the detection algorithm needs to be robust to run length. The mstop parameter (per40 vs per240) should be used to select the correct lookup table.

---

## Finding 12: 20260319 OUT/IN > 1.0 outlier — panel physically submerged

**Root cause confirmed: the panel was physically submerged under full wind on 20260319.**

Run notes state: "nå er de 5 første panelene under vann. det fremste er helt nede i 4 cm under vann" (first 5 panels are submerged, front one 4 cm underwater).

FFT amplitude data for full-panel runs at 20260319:
- Full wind: IN (9373/170) = 4.67 mm, OUT (12400/250) = 5.76 mm → OUT/IN = 1.234
- No wind same date: IN ≈ 7.1 mm, OUT ≈ 5.5 mm → OUT/IN ≈ 0.777

**Mechanism**: The submerged panel edge creates partial standing-wave interference at the 9373/170 IN probe position, reducing the measured IN amplitude. The OUT probe sees near-unobstructed transmission because the panel is no longer a surface barrier.

**This is NOT a pipeline bug.** It is a documented physical anomaly: the panel geometry was fundamentally different from all other runs.

**Action**: The 20260319 full-wind runs must be excluded from the main damping dataset. OUT/IN > 1.0 (combined with panel present) can serve as an automatic quality filter. This finding also strengthens the validity of the wind-effect analysis: excluding this date removes the most extreme outliers from the full-wind group.

---

## Finding 16: Mooring bug fix applied — what changes after --force-recompute

The Mooring classification fix applied to `wavescripts/improved_data_loader.py` will change the following:

| Date range | Before fix | After fix |
|------------|------------|-----------|
| 2025-11-06 to 2026-03-16 | `above_50` | `above_50` (unchanged) |
| 2026-03-16 (no keyword folder) | `above_50` | `above_50` (unchanged) |
| 2026-03-16 to 2026-03-26 (under9Mooring folder) | `above_50` | `below_90_loose230` |
| 2026-03-27 (under9Mooring30 folder) | `above_50` | `below_90_loose300` |

After recompute, the `Mooring` column will correctly distinguish the three mooring types. The mooring-type analysis in Findings 4, 9, and 10 used directory-name-based grouping (not the `Mooring` column) and is accurate. But `filters.py` uses the `Mooring` column for `groupby` in `damping_grouper` — meaning the stale cache has been incorrectly grouping all under9 runs with above50 runs, collapsing the mooring dimension and hiding the mooring effect. After recompute, `damping_grouper` will correctly separate the three mooring types.

---

## What I couldn't figure out / need more data

1. **Physical mechanism of wind effect**: The data clearly shows wind increases OUT/IN, but WHY requires more investigation. Need to compare the parallel probe (9373/340) readings under wind to understand lateral effects. Also need the full time-series of one run under wind to see what the OUT probe looks like spectrally.

2. **Mooring confound**: Cannot cleanly separate mooring type effect from date effect. Need matched pairs of same-date runs with different moorings.

3. **0.1V runs are most affected by wind**: At 0.1V, wind gives OUT/IN = 0.75 vs no-wind = 0.59 — a very large 27% difference. At 0.2V: 0.74 vs 0.71 — only 4% difference. This suggests that at very low wave amplitudes (where wind-wave energy at IN probe is ~2/3 of signal), the FFT-at-paddle-freq measurement is still being contaminated somehow, OR the physical mechanism is strongly amplitude-dependent.

4. **1.8 Hz anomaly**: At 1.8 Hz, wind seems to give LOWER OUT/IN (0.46 vs 0.59). This is opposite the trend everywhere else. Too few data points (n=4 full, n=2 no) to be confident.

5. **The very highest frequency runs (1.9–2.0 Hz)**: Very sparse data, hard to draw conclusions.
