# Probe Height, Range Mode, and Wind Background — Analysis Findings
*Autonomous session, 2026-03-30*

---

## Context

A bug was found and fixed during this session:

**Bug**: `_extract_probe_height()` and `_extract_probe_range_mode()` in `improved_data_loader.py`
were called with `filename` (the CSV basename, e.g. `fullpanel-...-run1.csv`), but the keywords
`-height100`, `-height136`, `-lowrange` are in the **folder name**, not the CSV name. Same root
cause as the earlier mooring bug.

**Fix**: Changed both calls at line 472–473 to pass `str(file_path)` instead of `filename`.
**Pipeline re-run** with `--force-recompute` to bake corrected values into all caches.

After the fix, condition counts across the main analysis dataset (Mar 2026 probe config):

| Condition | Runs | Folders |
|-----------|------|---------|
| cond1: height272, high-range (standard) | 201 | 8 (March 7 – 21) |
| cond2: height136, high-range (borderline) | 7 | 1 (March 23) |
| cond3: height100, high-range (wrong mode) | 129 | 4 (March 23 – 26) |
| cond4: height100, low-range (correct) | 132 | 2 (March 26 – 27) |

---

## Probe hardware geometry recap

| Condition | Probe height | Range mode | Window (mm) | Still-water position | Crest headroom | Trough headroom |
|-----------|-------------|------------|-------------|---------------------|----------------|-----------------|
| cond1 | 272 mm | high | 130–350 mm | 272 mm ✓ in window | +142 mm | +78 mm |
| cond2 | 136 mm | high | 130–350 mm | 136 mm ✓ just inside | +6 mm | +214 mm |
| cond3 | 100 mm | high | 130–350 mm | 100 mm ✗ below min | −30 mm | +250 mm |
| cond4 | 100 mm | low | 30–250 mm | 100 mm ✓ centered | +70 mm | +150 mm |

---

## Finding 1: Stillwater noise floor by condition

Noise floor = (P97.5 − P2.5) / 2 of `eta` signal in nowave+nowind runs. Values in mm.

| Condition | 9373/170 (IN) | 12400/250 (OUT) | 9373/340 (par) | 8804/250 (up) | n_runs |
|-----------|:-------------:|:---------------:|:--------------:|:-------------:|:------:|
| cond1 h272/high | **0.289 ± 0.094** | **0.263 ± 0.079** | 0.148 ± 0.111 | 0.240 ± 0.084 | 16 |
| cond2 h136/high | 0.045 (n=1) | 0.090 (n=1) | 0.045 (n=1) | 0.090 (n=1) | 1 |
| cond3 h100/high (wrong) | 0.158 ± 0.139 | 0.071 ± 0.050 | 0.127 ± 0.087 | 0.126 ± 0.085 | 12 |
| cond4 h100/low (correct) | **0.102 ± 0.082** | **0.087 ± 0.064** | 0.093 ± 0.115 | 0.064 ± 0.090 | 9 |

### Observations

**Condition 1 (h272) has the HIGHEST noise floor** across all probes — unexpected given it is the
"standard" configuration. The IN probe at 9373/170 averages 0.289 mm; the OUT probe 0.263 mm.
This likely reflects a combination of (a) probe calibration drift between the old physical probe
units and the new ones used from March 23 onward, and (b) the March 7–21 sessions possibly having
more residual tank motion in the stillwater periods. It cannot be attributed to probe height since
the h100 conditions have *lower* noise.

**Condition 2 (h136) appears spuriously quiet** (0.045 mm for most probes, single run).
This is not a real low-noise floor — it's the quantization resolution of the sensor. With only
6 mm of crest headroom, the probe is essentially stuck at one quantization level in still water.
The output is nearly constant → amplitude ≈ 0. These runs are not usable for noise-floor
characterization.

**Condition 3 (h100, wrong mode) has elevated but usable noise** — approximately 1.4–2.0× higher
than condition 4 (the correct setup) for most probes. The elevation is consistent with operating
outside the stated 130 mm minimum:

| Probe | cond3 mean | cond4 mean | ratio |
|-------|-----------|-----------|-------|
| 9373/170 (IN) | 0.158 mm | 0.102 mm | 1.55× |
| 9373/340 | 0.127 mm | 0.093 mm | 1.36× |
| 8804/250 | 0.126 mm | 0.064 mm | 1.96× |
| 12400/250 (OUT) | 0.071 mm | 0.087 mm | 0.82× (reversed — OUT probe is cleaner in cond3) |

The noise elevation is moderate. Cond3 folders are not fundamentally unreliable for wave analysis —
they have elevated but finite uncertainty.

**Important within-cond3 variation**: The 20260323 folder (first day of h100 probe lowering) has
much higher IN-probe noise (0.271 mm) than later folders (0.135, 0.045 mm). The act of
repositioning the probe likely disturbed the water, and the first-day stillwater runs capture
residual motion. Later cond3 folders are cleaner.

**Condition 4 (h100/lowrange) is the cleanest** for most probes:
- IN probe (9373/170): 0.102 mm mean
- OUT probe (12400/250): 0.087 mm
- The 20260327 folder (under9Mooring30, no rubber-band splash) has remarkably low noise:
  9373/170 = 0.065 mm, 8804/250 = 0.031 mm, 9373/340 = 0.052 mm, 12400/250 = 0.087 mm.
  This is likely the best-settled water of the entire experiment.

---

## Finding 2: Wind background amplitude by condition

Noise floor = (P97.5 − P2.5) / 2 of `eta` signal in nowave+fullwind runs.
**Probe mapping for all runs**: P1=9373/170, P2=12400/250, P3=9373/340, P4=8804/250
(march2026_better_rearranging config, applies to all conditions 1–4).

| Condition | 9373/170 (IN) | 12400/250 (OUT) | 9373/340 (par) | 8804/250 (up) | n_runs |
|-----------|:-------------:|:---------------:|:--------------:|:-------------:|:------:|
| cond1 h272/high | **10.575 ± 0.425** | 0.907 ± 0.158 | 9.951 ± 0.620 | 8.700 ± 0.319 | 5 |
| cond3 h100/high (wrong) | 9.851 ± 0.226 | 0.857 ± 0.060 | 10.200 ± 0.700 | 8.515 ± 0.120 | 2 |
| cond4 h100/low loose230 | 8.983 ± 0.447 | 1.016 ± 0.149 | 9.034 ± 0.217 | 7.745 ± 0.182 | 4 |
| cond4 h100/low loose300 | 9.568 ± 0.045 | 0.817 ± 0.004 | 8.707 ± 0.853 | 8.442 ± 0.067 | 2 |

### Observations

**Wind amplitude at exposed probes is very consistent across conditions: ~9–11 mm** (IN probe,
9373/170). The probe height and range mode do not significantly change the measured wind-wave
amplitude. This is the primary result: the measured wind background is a property of the tank's
wind-wave field, not of the probe geometry.

The small systematic decrease from cond1 to cond4 (~1.4 mm at IN probe, 10.6 → 9.2 mm) could
reflect:
- Slightly lower wind speed / fetch variation between session dates
- The probe at h100 is closer to the water surface → slightly different sampling of the wave
  field (the sensor face at 100 mm is only 28 mm above the wave crest at typical 72 mm amplitude)
- Not a hardware artifact — the signal looks physically consistent in both conditions

**OUT probe (12400/250) is remarkably consistent and low: 0.82–1.02 mm** across all conditions.
The panel shelter is effective regardless of probe height or mooring type. This is the key
result for the damping analysis: the wind background at the OUT probe is small (~1 mm) and
stable, well below the typical wave amplitudes being measured.

**Mooring type has negligible effect on wind amplitude** (cond1 only, h272):
- above_50: IN = 10.164 mm, OUT = 0.883 mm
- below_90_loose230: IN = 10.848 mm, OUT = 0.923 mm
- Difference: ~0.7 mm (7%), within run-to-run variability (std ~0.2–0.3 mm)

Wind waves are generated by surface wind–water interaction, not by mooring–panel coupling.
The mooring type changes the panel dynamics (which affects reflected/transmitted waves) but
does not meaningfully change the background wind-wave field at the probe locations.

---

## Finding 3: Wind-to-stillwater SNR by condition

| Condition | IN probe (9373/170) | OUT probe (12400/250) |
|-----------|--------------------:|---------------------:|
| cond1 h272/high | Wind/SW = 36.6× | Wind/SW = 3.45× |
| cond3 h100/high (wrong) | Wind/SW = 62.4× | Wind/SW = 12.0× |
| cond4 h100/low (correct) | Wind/SW = 90.3× | Wind/SW = 10.9× |

The Wind/SW ratio is driven primarily by the **lower stillwater noise floor** in h100 conditions,
not by higher wind amplitude. The probe at h100 simply has better precision in still water.

**OUT probe SNR is low (3–12×)**: the wind amplitude (~0.9 mm) is only 3–12× above the
stillwater noise floor (~0.07–0.26 mm). This means:
- Under wind, the OUT probe signal is dominated by wind waves (~1 mm)
- A paddle wave would need amplitude >> 1 mm at the OUT probe to be clearly distinguishable
  from wind background on amplitude alone
- For the standard wave amplitude (0.1 V ≈ ~5 mm at OUT), SNR is fine. For very attenuated
  transmission at high frequency where OUT amplitude drops to ~1 mm, wind contamination
  becomes significant.

**IN probe SNR is high (37–90×)**: wind amplitude (~10 mm) >> stillwater noise floor (~0.1–0.3 mm).
Wave arrival detection at the IN probe is straightforward even under full wind.

---

## Finding 4: Condition 3 wave data is physically usable

For amp0300 (largest waves) in cond3 (h100, highrange = wrong mode):
- IN probe minimum reading: ~60–63 mm (water surface 37–40 mm ABOVE still-water level)
- No readings below 60 mm (no hard clipping at any probe)
- Amplitudes (27–30 mm) are consistent with equivalent cond4 runs (27–30 mm)

**The hardware continues to return valid measurements below the 130 mm stated minimum.**
The probe operates correctly at 60–100 mm despite being outside its nominal high-range window.
The stated "130 mm minimum" appears to be a nominal accuracy threshold, not a hard cutoff.

However, the elevated stillwater noise (1.5–2× vs cond4) means cond3 data carries slightly
higher amplitude uncertainty. Detection threshold for wave arrival should use the per-folder
noise floor, not the overall mean.

---

## Finding 5: Rubber-band splash does not affect the analysis-critical folders

Per the researcher's note:
- Rubber-band splash only occurred in some `under9Mooring` folders (not `under9Mooring30`)
- The splash, if present, would appear as non-stationary high-frequency bursts in the raw signal
- The 20260327 folder (under9Mooring30, cond4 loose300) is the cleanest reference for
  per-folder wind background at h100/lowrange

A rolling-RMS stationarity check across the fullwind nowave runs would identify which specific
runs (if any) contain the rubber-band artifact. This has not yet been done — see next steps.

---

## Recommended per-folder noise floor and wind background (updated values)

For the main damping analysis (2026-03-07 onward, march2026_better_rearranging config):

**Stillwater noise floor** (2× detection threshold in parentheses):

| Probe | Cond1 h272 | Cond3 h100/wrong | Cond4 h100/low |
|-------|-----------|-----------------|----------------|
| 9373/170 (IN) | ~0.29 mm (0.58 mm) | ~0.16 mm (0.32 mm) | ~0.10 mm (0.20 mm) |
| 12400/250 (OUT) | ~0.26 mm (0.52 mm) | ~0.07 mm (0.14 mm) | ~0.09 mm (0.18 mm) |
| 9373/340 | ~0.15 mm (0.30 mm) | ~0.13 mm (0.26 mm) | ~0.09 mm (0.18 mm) |
| 8804/250 | ~0.24 mm (0.48 mm) | ~0.13 mm (0.26 mm) | ~0.06 mm (0.12 mm) |

**Wind background amplitude** (for first-motion threshold and SNR estimation):

| Probe | All conditions (range) | Notes |
|-------|----------------------|-------|
| 9373/170 (IN) | 8.9–10.6 mm | Consistent; higher at h272 by ~1 mm |
| 12400/250 (OUT) | 0.82–1.02 mm | Very consistent across all conditions |
| 9373/340 | 8.7–10.2 mm | Similar to IN probe |
| 8804/250 | 7.7–8.7 mm | ~1 mm below IN (less fetch from roof) |

---

## Open items / next steps

1. **Rolling RMS stationarity check**: for each nowave+fullwind run in cond3 and cond4 folders,
   compute rolling RMS (1 s windows) across the full run and check if it is stationary. Identify
   which runs (if any) show the rubber-band burst pattern (sudden high-amplitude, rapid-decay
   events in the 9373-probe group).

2. **PSD comparison across conditions**: using `_wind_psd_dict`, plot mean PSD for cond1 vs cond4
   at the same probe. Key question: does the probe height/range mode affect the spectral shape
   of the wind-wave field, or only the amplitude? If spectral shape is unchanged, the conditions
   are directly comparable for wave analysis.

3. **Per-folder pipeline column**: `wind_rms_{pos}` — one scalar per folder (from nowave+fullwind
   runs in that folder). Analogous to per-run stillwater anchor but aggregated to folder level.
   Would live in `processor2nd.py:compute_inter_run_timing()` or a new function. Enables
   per-folder first-motion threshold in `RampDetectionBrowser`.

4. **Cond3 wave analysis validity**: with the noise floor and wave data checked, the 129 wave runs
   in cond3 folders appear usable. Recommend keeping them in the analysis dataset but flagging
   them with their elevated noise floor so the detection threshold is applied correctly per folder.

5. **Two traceback errors in recompute**:
   - `20251112-tett6roof`: `ensure_stillwater_columns` → `pd.to_datetime` format error on `file_date`
   - `20260324-under9Mooring-height100`: `find_wave_range` → `mstop_sec` is NaN for some run
   Both are pre-existing bugs, not caused by the height/range fix.
