# Core Assumption Audit — OUT/IN (FFT) as Transmission Metric
*Written 2026-04-14. Intended for critical scrutiny by the next agent.*

---

## The claim under attack

The project uses:

```
OUT/IN (FFT) = FFT_amplitude(12400/250) / FFT_amplitude(9373/170)
```

measured at the paddle frequency (narrow ±0.05 Hz window) as the **wave transmission
coefficient** through the FPV panel.

**The implicit assumptions this bundles together:**

1. `FFT(9373/170)` = incident wave amplitude  
2. `FFT(12400/250)` = transmitted wave amplitude  
3. Dividing them cancels wavemaker variability and gives a geometry-only ratio  
4. The ±0.05 Hz FFT window contains only paddle-frequency energy (no wind contamination)  
5. The result is stable across the measurement window (not a snapshot of a transient)  
6. Without a panel, the ratio should approach 1.0 (baseline check)

**Every one of these is questionable.** The sections below attack each in order of physical
severity, with specific tests the next agent can run.

---

## Attack 1 — The IN probe measures incident + reflected wave, not incident alone

### Physical argument

The IN probe sits at 9373 mm from the paddle. The panel sits at approximately
`x_panel ≈ 10500–11000 mm` (centroid unknown; noted as `_panel_x = 11.0 m` in
`main_explore_inline.py` Reflection 1 cell). The panel **reflects** some fraction R
of the incoming wave energy back toward the paddle.

Between the paddle and the panel, the wave field is a superposition:

```
η(x, t) = A_i cos(kx − ωt) + A_r cos(−kx − ωt + φ)
```

where A_i = incident amplitude, A_r = R·A_i = reflected amplitude, k = wavenumber.

The measured FFT amplitude at position x_in = 9.373 m is NOT A_i. It is:

```
A_meas(x_in) = A_i × √(1 + R² + 2R cos(2k·Δ))
```

where **Δ = x_panel − x_in ≈ 11.0 − 9.373 = 1.627 m**.

This factor oscillates between:
- **Maximum (antinode)**: √(1 + R)² = 1 + R  → measured IN inflated by factor (1+R)  
- **Minimum (node)**: √(1 − R)² = 1 − R  → measured IN deflated by factor (1−R)

The measured OUT/IN is therefore:

```
OUT/IN_meas = T / √(1 + R² + 2R cos(2k·Δ))
```

where T = A_transmitted / A_i is the TRUE transmission coefficient we want.

**This is a frequency-dependent systematic error. The sign alternates.**

### Magnitude estimate

From no-panel control data (Finding 6 in findings.md):
- OUT/IN without panel ≈ 0.93–0.98 at 0.65–1.3 Hz
- The panel IS damping, so R > 0

From the amplitude ratio A(9373)/A(8804) oscillation (Reflection 3 design in project_tasks.md):
rough estimate R ≈ 0.1–0.3 for intermediate frequencies.

With R = 0.20:
- At antinode: measured IN is 20% inflated → OUT/IN_meas is 17% LOWER than true T
- At node: measured IN is 20% deflated → OUT/IN_meas is 25% HIGHER than true T

**A ±20% systematic error, oscillating with frequency.** Not a second-order correction.

### Specific test to run

**Test 1a — Node/antinode fingerprint:**

Compute the standing-wave phase angle `φ(f) = 2k(f) × 1.627 mod 2π` at each
experimental frequency. Use the full dispersion relation `ω² = g·k·tanh(k·d)` with d = 580 mm.

Then check: do the dips in the OUT/IN vs frequency curve (e.g. the 1.1–1.2 Hz region
where wind+no-wind both show slightly lower transmission than neighbors) correspond to
φ ≈ 0 (antinode — IN inflated, OUT/IN falsely low)?

And: does OUT/IN > 1.0 at 0.7–0.8 Hz correspond to φ ≈ π (node — IN deflated, OUT/IN
falsely high)?

Code needed:
```python
from scipy.optimize import brentq
g, d = 9.81, 0.580
def dispersion(k, f):
    omega = 2*np.pi*f
    return omega**2 - g*k*np.tanh(k*d)

freqs = np.arange(0.65, 1.95, 0.05)
k_vals = [brentq(dispersion, 0.01, 100, args=(f,)) for f in freqs]
Delta = 1.627  # m — update if panel centroid is confirmed
phase = [(2*k*Delta) % (2*np.pi) for k in k_vals]
# phase near 0 → antinode → IN inflated → OUT/IN biased LOW
# phase near π → node    → IN deflated → OUT/IN biased HIGH
```

Then overlay `phase` on the `OUT/IN vs frequency` plot for no-wind, full-panel runs.
If peaks and troughs in OUT/IN align with node/antinode positions: the standing wave
contamination is real and significant.

**Critical unknown**: the exact position of the panel centroid (or front edge) along the tank.
`_panel_x = 11.0 m` is approximate. A 0.1 m error in Δ shifts the phase by
`2 × k × 0.1 ≈ 0.8 rad` at 1.3 Hz, which is not negligible.  
→ **Verify panel centroid from lab drawings or measure from experimental notes.**

**Test 1b — Two-probe separation method (Mansard & Funke):**

The reflection analysis cells in `main_explore_inline.py` (Reflection 2, ~lines 2790–3225)
implement the two-probe decomposition using probes at 9373/170 and 8804/250 (Δx = 569 mm):

```
FFT_complex(f, x1) = A_i · e^(ikx1) + A_r · e^(-ikx1)
FFT_complex(f, x2) = A_i · e^(ikx2) + A_r · e^(-ikx2)
```

Solve for A_i, A_r → R = |A_r/A_i|.  
This gives the true incident amplitude A_i for each frequency, enabling the correct
transmission coefficient T = A_out / A_i.

**This is the definitive fix, but it requires running the reflection cells first (currently ⬜ in
project_tasks.md).** Note the ill-conditioning warning: the method fails when `|sin(k·Δx)| < 0.30`
(near f ≈ 1.3 Hz and 1.7 Hz for Δx = 569 mm). These are the exact frequencies most
important to the thesis — so the ill-conditioning is a serious limitation of this approach.

**Test 1c — Check if A(9373/170) with panel < A(9373/170) without panel:**

If the panel creates a partial standing wave, the amplitude at 9373/170 WITH panel should
oscillate around the amplitude WITHOUT panel. Specifically, for frequencies where the phase
argument is near antinode, A_with_panel > A_no_panel; for node, A_with_panel < A_no_panel.

```python
# Compare mean A_in (no-panel) vs A_in (full-panel) per frequency, no-wind only
# Use combined_meta columns: "Probe 9373/170 Amplitude (FFT)", PanelCondition
nopanel = combined_meta.query("PanelCondition == 'no' and WindCondition == 'no'")
fullpanel = combined_meta.query("PanelCondition == 'full' and WindCondition == 'no'")
# groupby WaveFrequencyInput, compare mean amplitudes
```

If the A_in ratio (no-panel / full-panel) oscillates with frequency in a pattern consistent
with the phase angle from Test 1a, this is strong evidence for standing wave contamination.

---

## Attack 2 — The probes are not symmetric in the tank

### Physical argument

- IN probe: longitudinal 9373 mm, lateral **170 mm** (near wall)
- OUT probe: longitudinal 12400 mm, lateral **250 mm** (centerline)

These measure the wave field at different cross-sectional positions. Even without the panel,
they measure different things. The **no-panel control (Finding 6) gives OUT/IN ≈ 0.93–0.98**
at two frequencies. This 2–7% offset is real and must be accounted for.

But the critical question is: **is this offset frequency-dependent?**

If the wave field has any lateral non-uniformity (wall reflections, edge diffraction,
wind-driven lateral setup), the correction factor will vary with frequency. Applying a
single constant correction factor from two data points (0.65 Hz and 1.30 Hz) is insufficient.

### Known lateral asymmetry evidence

The parallel probe at 9373/340 (same longitudinal distance as IN, other lateral side)
shows 4× spread in stillwater amplitude across runs (CLAUDE.md §16: 0.075–0.315 mm).
This probe is unreliable. But the comparison between 9373/170 and 9373/340 under wave
conditions would reveal lateral non-uniformity at the IN probe longitudinal station.

Column: `"Probe 9373/340 Amplitude (FFT)"` vs `"Probe 9373/170 Amplitude (FFT)"`.
The ratio `parallel_ratio = 9373/340 / 9373/170` is presumably already computed in the
pipeline (mentioned in Q4 of questions.md and in plotter.py API).

### Specific test to run

**Test 2a — No-panel OUT/IN vs frequency:**

This is the baseline calibration test. The no-panel data exists (old probe config
9373/250 → 12400/170) for 0.65 Hz and 1.30 Hz. But the current probe config
(9373/170 → 12400/250) may not have no-panel runs at every frequency.

```python
nopanel = combined_meta.query(
    "PanelCondition == 'no' and WindCondition == 'no'"
    " and in_position == '9373/170' and out_position == '12400/250'"
)
# group by WaveFrequencyInput, compute mean OUT/IN
```

If no-panel OUT/IN is flat (constant ≈ 0.95) across 0.65–1.9 Hz: the asymmetry is a
geometric constant, safely corrected by dividing all panel-present results by ~0.95.

If no-panel OUT/IN varies with frequency: every OUT/IN measurement has a frequency-
dependent systematic bias that is NOT removed by the current pipeline. The corrected
transmission would be:

```
T_corrected(f) = OUT/IN_panel(f) / OUT/IN_nopanel(f)
```

**Test 2b — Parallel ratio vs frequency:**

```python
# For no-wind, full-panel runs only:
combined_meta["parallel_ratio"] = (
    combined_meta["Probe 9373/340 Amplitude (FFT)"]
    / combined_meta["Probe 9373/170 Amplitude (FFT)"]
)
# groupby WaveFrequencyInput, plot mean ± std
# If flat → lateral uniformity at IN station
# If oscillating → lateral standing wave / wall reflection at IN station
```

A frequency-dependent parallel ratio is the smoking gun for Attack 1 as well —
it would confirm that the standing wave pattern is laterally inhomogeneous (consistent
with wall-side vs center probe measuring different standing wave phases).

---

## Attack 3 — The FFT window captures wind energy at the paddle frequency

### Physical argument

The FFT amplitude uses a ±0.05 Hz window centered on the target frequency. Wind waves
are described as "broadband above ~2 Hz" (CLAUDE.md §16), but this is a spectral
centroid description. The wind energy spectral density does not go to zero below 2 Hz —
it has a low-frequency tail.

**Evidence from Finding 8 (findings.md):**

| Freq | IN probe SNR (wind/paddle) | OUT probe SNR |
|------|---------------------------|---------------|
| 0.5 Hz | **0.66** (wind > paddle) | 10.6 |
| 0.7 Hz | 1.32 | 15.6 |
| 1.0 Hz | 2.22 | 15.7 |
| 1.3 Hz | 2.25 | 9.1 |
| 1.7 Hz | 2.64 | 5.4 |

At the IN probe, paddle-wave SNR against wind-wave background is 2–3 at the most
important frequencies. This is not measurement-noise contamination — **this is
coherence contamination**: the FFT estimate of the paddle-frequency amplitude is
degraded by incoherent wind-wave energy at nearby frequencies.

The FFT amplitude at frequency f over window T is:
```
A_FFT(f) = (2/T) |∫ η(t) e^(-2πift) dt|
```

If η(t) = A_paddle sin(2πf₀t) + η_wind(t), then the measured A_FFT(f₀) includes
leakage from η_wind within the ±0.05 Hz bin. This leakage is not zero even if the
wind spectrum peaks at 3+ Hz, because the FFT has sidelobes.

More critically: the wind waves are **incoherent** — their phase is random run-to-run.
In some runs they add to the paddle wave in the measurement window; in others they
subtract. This adds **random variance** to OUT/IN estimates under wind.

### Specific test to run

**Test 3a — Compare FFT amplitude to PSD-derived amplitude:**

The PSD amplitude (`"Probe {pos} Amplitude (PSD)"`) uses Welch's method across the
full run and averages. It is less sensitive to incoherent noise within a single window.

Compare `"Probe 9373/170 Amplitude (FFT)"` vs `"Probe 9373/170 Amplitude (PSD)"` for
full-wind runs vs no-wind runs per frequency. If FFT amplitude is systematically LOWER
under wind (not higher), this indicates the incoherence artifact: random phase wind
waves partially cancel the coherent paddle wave in the FFT integral.

**Test 3b — Compare wind-only PSD at paddle frequency:**

From nowave + fullwind runs, extract PSD power in the ±0.05 Hz window around each
target frequency:

```python
# Already computed in main_explore_inline.py wind cells (Cell A)
# wind_psd_dict: {path: DataFrame(index=Frequencies, cols="Pxx {pos}")}
# For each nowave+fullwind run, sum Pxx in [f_target ± 0.05 Hz]
# Convert to amplitude: A_wind = sqrt(2 * Pxx_sum * df)
# Compare with A_paddle from combined_meta["Probe 9373/170 Amplitude (FFT)"]
```

If A_wind_at_paddle_freq > 0.1 × A_paddle: the ±0.05 Hz window is contaminated for
that frequency under full wind. Report as SNR = A_paddle / A_wind.

**Test 3c — High-SNR filter test:**

Finding 10 already did this: filtering to SNR_IN > 3 reduces but does not eliminate the
wind effect. The residual wind effect at high SNR is the **true physical wind effect**.

The exact numbers from Finding 10 (high-SNR full-wind vs no-wind):
```
1.3 Hz: +0.050 (+7%)
1.5 Hz: +0.143 (+26%)  
1.7 Hz: +0.191 (+56%)
```

These are the best current estimates of the TRUE wind effect, but they need verification
by running the test in the current pipeline (with corrected Mooring column and quality
flags). Finding 10 was computed before the mooring bug fix and before `dropout_critical`
flags were added.

---

## Attack 4 — The measurement window may include reflections from the far end of the tank

### Physical argument

After the wavemaker starts, waves travel down the tank. They reach the far end (assume
tank length ~25 m) and reflect back (partially — beach absorbers reduce but don't
eliminate reflections). The reflected wave from the far end arrives back at the IN probe
at time:

```
t_reflect ≈ (2 × L_tank) / c_group
```

At 1.0 Hz in deep water: c_group = g/(4πf) ≈ 0.78 m/s. Travel time = 2 × 25 / 0.78 ≈ 64 s.
At 1.3 Hz: c_group ≈ 0.60 m/s → 83 s.
At 0.7 Hz: c_group ≈ 1.12 m/s → 45 s.

For per40 runs (40 s total recording), the far-end reflection may not arrive within the
recording window. For per240 runs (240 s recording), it definitely arrives and may
contaminate the stable wave window if `_SNARVEI_CALIB` start + window duration extends
past the reflection arrival time.

This is particularly relevant for the per240 no-wind high-amplitude outliers identified
in Finding 14 — but those were likely wave detection failures, not far-end reflection
contamination. The more insidious case is **partial contamination** in per40 runs at
low frequencies where c_group is high.

### Specific test to run

**Test 4a — Estimate far-end reflection arrival time:**

```python
import numpy as np
g = 9.81
L_tank = 25.0  # m — VERIFY from lab drawings
d = 0.580
freqs = [0.65, 0.70, 0.80, 0.90, 1.0, 1.3, 1.5, 1.7, 1.9]
for f in freqs:
    # Solve dispersion for k
    omega = 2*np.pi*f
    # Use k from combined_meta for a representative run at this frequency
    # Or solve numerically as in Test 1a
    k = brentq(lambda k: omega**2 - g*k*np.tanh(k*d), 0.01, 100)
    c_group = (g / (2*omega)) * (1 + 2*k*d/np.sinh(2*k*d))
    t_reflect = 2 * L_tank / c_group
    print(f"{f:.2f} Hz: c_g = {c_group:.2f} m/s, t_reflect = {t_reflect:.0f} s")
```

Compare `t_reflect` against the analysis window boundaries:
- Window start: `_SNARVEI_CALIB` sample / 250 Hz (check `wavescripts/wave_detection.py`)
- Window end: `mstop_sec` (from combined_meta column `mstop_sec` or inferred from filename)

If `t_reflect < mstop_sec` for any frequency: far-end contamination is possible in
those runs. Flag these runs.

**Test 4b — Check for amplitude non-stationarity within the stable window:**

If far-end reflections arrive and contaminate the signal, the amplitude within the stable
window is not constant — it increases when the reflected wave superimposes. Compute a
rolling FFT amplitude within the stable window (e.g. 10-period sub-windows):

```python
# For a selected per40 run at low frequency, full-panel, no-wind:
# rolling_amp_IN[t] = FFT amplitude of η(t-5s : t+5s) at probe 9373/170
# If amplitude jumps mid-window: far-end reflection detected
# Column to use: "cut_samples_9373/170" — if nonzero, window already cropped
```

---

## Attack 5 — The OUT probe has its own confound: partial wind shelter

### Physical argument

The OUT probe (12400/250) is described as "past the panel, almost no wind" (CLAUDE.md
intro). But **"almost no wind"** is not "no wind." From Finding 2 in
probe_height_wind_findings.md:

```
Wind background at OUT probe (12400/250): 0.82–1.02 mm across all conditions
```

This is 1 mm RMS of wind-wave amplitude at the OUT probe despite being sheltered by
the panel. At high frequencies (1.7–1.9 Hz) where the transmitted wave amplitude drops
to ~1–3 mm, the wind background at the OUT probe is 30–100% of the transmitted wave.

More subtly: the wind shelter provided by the panel is not perfect. At full wind, there
is some air flow around/over the panel edges, and wind can still generate waves in the
short fetch between the panel and the OUT probe. This energy is at wind-wave frequencies
(2–5 Hz) and does NOT contaminate the FFT estimate at paddle frequency. But it DOES
add to the time-domain OUT amplitude.

However, there is a second effect: **wind-driven currents and surface drift.** If wind
creates a horizontal surface current past the panel into the OUT region, this current
could modulate the wave speed and therefore the apparent amplitude at 12400/250.

### Specific test to run

**Test 5a — Quantify wind background at OUT probe relative to transmitted wave amplitude:**

For each frequency under full wind:
```python
# wind_background_out[f] from nowave+fullwind runs
# transmitted_amplitude[f] from "Probe 12400/250 Amplitude (FFT)" in wave runs
# SNR_OUT[f] = transmitted_amplitude[f] / wind_background_out[f]
```

Finding 8 gives this for some frequencies:
- 1.7 Hz, full wind: SNR_OUT = 5.4 → wind background = 7.7/5.4 ≈ 1.4 mm

At 1.9 Hz where OUT/IN ≈ 0.2 and IN amplitude ≈ 10 mm, OUT amplitude ≈ 2 mm.
Wind background at OUT ≈ 1 mm. SNR_OUT ≈ 2.0. This is marginal.

**Implication**: At 1.7–1.9 Hz, the OUT probe measurement under full wind is
significantly affected by residual wind background. The FFT extraction at the paddle
frequency mitigates this (wind energy is at different frequencies), but the
incoherence / leakage problem from Attack 3 applies to the OUT probe too.

---

## Attack 6 — Probe configuration correctness end-to-end

### Background: the probe switch

Between Nov 2025 and Mar 2026, the physical probes were rearranged. The config system
(`PROBE_CONFIGS` in `improved_data_loader.py`) is the authoritative source — and the user
confirms it is rock solid. Two separate questions remain:

1. Does the **cached pipeline output** (meta.json `in_position`/`out_position`) correctly
   reflect the config for every run?
2. Does the **analysis code** use config-driven columns, or does it hardcode position strings
   that are only valid for the current config?

Config transitions (from CLAUDE.md §8):

| Config | Valid | in_position | out_position |
|--------|-------|-------------|--------------|
| `initial_setup` | Aug 2025 | `9373/250` | `12400/170` |
| `nov_normalt_oppsett` | Nov 2025 | `9373/250` | `12400/170` |
| `march2026_rearranging` | Mar 4–7 2026 | `9373/170` | `11800/250` |
| `march2026_better_rearranging` | Mar 7+ 2026 | `9373/170` | `12400/250` |

### What is actually loaded

As of 2026-04-14, `main_explore_inline.py` PROCESSED_DIRS contains:
- All Nov 2025 folders: **commented out** (old config)
- Transitional Mar 4–6 folders: **commented out**
- **20260307 folder: commented out** (first day of current config — contains no-panel runs)
- 20260312 onward: **loaded** — all current config only

**The currently loaded dataset is single-config.** No probe-switch collisions can occur
with the current PROCESSED_DIRS. This is safe — but only as long as older folders
remain commented out.

### The no-panel data situation (operationally urgent)

The no-panel baseline runs for the **current config** (9373/170 → 12400/250) are in the
**commented-out 20260307 folder.** The code at line 3022 of `main_explore_inline.py`
explicitly checks and prints:

```
⚠  NO nopanel runs in current PROCESSED_DIRS.
   The nopanel control data is likely in the commented-out 20260307 folder.
```

**Finding 6 (findings.md) reported no-panel OUT/IN ≈ 0.93–0.98** — but this was computed
from the **old probe config** (in=`9373/250`, out=`12400/170`). The lateral positions of
IN and OUT are **swapped** between old and new configs. That 0.93–0.98 baseline **cannot
be applied** to the current config data.

The current-config no-panel data exists physically (20260307 folder), but has never been
loaded into the analysis.

### Hardcoded position strings

Several analysis cells hardcode current-config position strings:

| Location | String | Effect on old-config data |
|----------|--------|--------------------------|
| `main_explore_inline.py:1283` | `_WIND_RMS_PROBE = "9373/170"` | Silent NaN for old-config runs |
| `main_explore_inline.py:1932` | `_in_pos = "9373/170"` | Silent NaN |
| `main_explore_inline.py:2607–2608` | `_SIM_IN/OUT = "9373/170"/"12400/250"` | Silent NaN |
| `main_explore_inline.py:2985` | `_REFL_IN = "9373/170"` | Silent NaN |

**For the current single-config PROCESSED_DIRS, all hardcoded strings are correct.**
If old-config folders were ever uncommented and mixed in, those runs would return NaN
from any cell using these strings and be silently excluded — no error, just missing data.

The core pipeline (`damping_grouper`, `processor2nd.py`) does NOT hardcode positions —
it reads from `in_position`/`out_position` columns. That path is config-driven and correct.

### Specific tests to run

**Test 6a — Verify in_position/out_position are set correctly for all loaded runs:**

```python
pos_check = combined_meta[combined_meta["run_category"] == "standard"][
    ["file_date", "in_position", "out_position"]
]
unexpected = pos_check[
    (pos_check["in_position"] != "9373/170") | (pos_check["out_position"] != "12400/250")
]
print(f"Runs with unexpected in/out position: {len(unexpected)}")
if len(unexpected):
    print(unexpected.to_string())
# Expected: 0 rows. Any NaN or wrong string = stale meta.json → force-recompute needed.
```

**Test 6b — Load 20260307 and run Reflection 0:**

Uncomment the 20260307 line in PROCESSED_DIRS, reload `combined_meta`, and run the
Reflection 0 cell (~line 2978 of `main_explore_inline.py`). This retrieves the no-panel
calibration data for the current config.

Expected output:
```
✓  N nopanel runs available.
Frequencies with nopanel data: [0.65, 0.70, ...]
```

The 20260307 comment says "disse bør være greie bortsett fra de steile" (fine except
steep waves). At 0.1 V amplitude, no steepness issues are expected.

**Test 6c — Verify Finding 6 came from the old config:**

```python
nopanel = combined_meta.query("PanelCondition == 'no'")
print(nopanel[["file_date", "in_position", "out_position",
               "WaveFrequencyInput [Hz]", "OUT/IN (FFT)"]].to_string())
```

With the current PROCESSED_DIRS (20260307 commented out), this should return 0 rows.
This confirms: the no-panel baseline currently used in any analysis is either missing
entirely, or was transferred from the old config (invalid). Either way, the Reflection 0
cell will warn about it.

**Test 6d — After loading 20260307, compare no-panel OUT/IN with panel OUT/IN:**

```python
nopanel_curr = combined_meta.query(
    "PanelCondition == 'no' and WindCondition == 'no'"
    " and in_position == '9373/170'"
).groupby("WaveFrequencyInput [Hz]")["OUT/IN (FFT)"].mean()

fullpanel_curr = combined_meta.query(
    "PanelCondition == 'full' and WindCondition == 'no'"
    " and in_position == '9373/170'"
).groupby("WaveFrequencyInput [Hz]")["OUT/IN (FFT)"].mean()

print("No-panel OUT/IN (current config, new baseline):")
print(nopanel_curr)
print("\nFull-panel / no-panel ratio (corrected T):")
print(fullpanel_curr / nopanel_curr)
```

If no-panel OUT/IN is not flat across frequency: the geometric correction is
frequency-dependent, and every current OUT/IN value must be divided by this
per-frequency factor to get the true transmission coefficient.

---

## Summary: Which attacks are most severe?

| Attack | Effect on OUT/IN | Direction | Fixable? |
|--------|-----------------|-----------|---------|
| 1: Reflection at IN probe | ±20% at each frequency | Alternating | Yes — two-probe decomposition |
| 2: Lateral probe asymmetry | 2–7% constant, possibly more at high freq | Systematic | Yes — no-panel correction |
| 3: Wind contamination of FFT | ~30–40% of observed wind effect | Inflates apparent wind effect | Partially — SNR filter |
| 4: Far-end reflections | Unknown — only relevant for per40 at low freq | Systematic at specific freq | Yes — check window timing |
| 5: Residual wind at OUT probe | ~50% of OUT amplitude at 1.9 Hz | Inflates OUT/IN under wind | Partially — SNR filter on OUT |
| 6: No-panel baseline wrong config / missing | ~5% correction factor unknown | Unknown sign | Yes — uncomment 20260307 folder, run Reflection 0 |

**Attack 1 is the most physically fundamental.** If R ≈ 0.2 (reasonable), the standing
wave creates ±20% oscillation in apparent OUT/IN vs the TRUE transmission. The
observed OUT/IN > 1.0 at 0.7–0.8 Hz is direct evidence that the IN probe is near a
node (measuring less than A_i) at those frequencies. The OUT/IN < expected at
1.1–1.2 Hz may similarly reflect an antinode at the IN probe.

**Attack 6 is operationally the easiest to resolve:** the data exists (20260307 folder),
the code already warns about it (line 3022), and uncommenting one line gives the no-panel
baseline for the current probe config. The config system is rock-solid — the issue is
purely that the relevant folder is commented out of PROCESSED_DIRS.

---

## Recommended execution order for the next agent

1. **Run Test 6a** — verify `in_position`/`out_position` are correct for all loaded runs.
   Expected: 0 mismatches. Takes 5 seconds.

2. **Run Test 6b** — uncomment 20260307 in PROCESSED_DIRS, reload, run Reflection 0.
   This gives the no-panel baseline for the current config (has never been loaded).
   Takes 2 minutes.

3. **Run Test 1a** — compute the standing wave phase at each frequency, overlay on
   OUT/IN vs frequency plot. If phase and OUT/IN dips/peaks align, Attack 1 is confirmed
   and the paper needs the two-probe reflection analysis.

3. **Run Test 2a** — no-panel OUT/IN vs frequency in current config (if any data exists).
   This directly measures the geometric correction factor.

4. **Run Test 3b** — wind PSD at paddle frequency from nowave+fullwind runs. Gives the
   per-frequency SNR and quantifies how much of the wind effect is artifact.

5. **Run Reflection 2 cells** in `main_explore_inline.py` (~line 2790) — the two-probe
   decomposition. Note: ill-conditioned at f ≈ 1.3 Hz and 1.7 Hz. Report which
   frequencies are reliable and which are not.

---

## What a valid OUT/IN metric would look like

Given all the above, the correct transmission coefficient estimate is:

```
T(f) = A_transmitted(f) / A_incident(f)
```

where:
- `A_incident(f)` = true incident amplitude, separated from reflected using two-probe
  Mansard–Funke decomposition (probes at 8804/250 and 9373/170, separation = 569 mm)
- `A_transmitted(f)` = FFT amplitude at 12400/250 at paddle frequency, after subtracting
  estimated wind-background contribution

This is a non-trivial analysis requiring:
1. Complex FFT at both upstream probes (already stored: `"FFT 9373/170 complex"`,
   `"FFT 8804/250 complex"`)
2. Solving the 2×2 linear system per run per frequency
3. Checking conditioning (`|sin(kΔx)| ≥ 0.30`) and flagging ill-conditioned runs
4. Applying a wind-background subtraction at the OUT probe

**Until this analysis is done, all OUT/IN values carry an unknown systematic error
that could be ±20% at any given frequency, and the "wind increases transmission" finding
could be partly an artifact of the standing wave changing the apparent IN amplitude
under wind conditions (if wind changes R by disrupting the standing wave pattern).**

---

*All referenced code is in `wavescripts/` and `main_explore_inline.py`. All referenced
columns are in `combined_meta`. The reflection analysis cells are at approximately
lines 2790–3225 of `main_explore_inline.py` and are currently marked ⬜ in
`project_tasks.md` (not yet run).*
