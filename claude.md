# Wave Project — Context for Code Assistants

---

## WHAT THIS PROJECT IS — READ THIS FIRST

**Thesis experiment**: A floating solar panel (FPV) geometry is placed in a wave tank. Paddle waves are generated at 0.65–1.9 Hz. The central question is:

> **Does wind increase or decrease how much of an incoming wave is transmitted past the panel geometry?**

**Key metric**: `OUT/IN (FFT)` — ratio of wave amplitude *past* the panel to incident wave amplitude, computed at the paddle frequency only (narrow 0.1 Hz FFT window). Time-domain amplitude is NOT used for damping — it includes wind-wave energy.

**Experiment variables**: `WaveFrequencyInput [Hz]` · `WaveAmplitudeInput [Volt]` (0.1 V / 0.2 V) · `WindCondition` (full / lowest / no) · `PanelCondition` (full / reverse / no) · `Mooring`

**Probes**: 4 wave gauges identified by physical position `"longitudinal_mm/lateral_mm"`:
- `9373/170` — IN probe, between paddle and panel, fully exposed to wind
- `12400/250` — OUT probe, past panel, almost no wind (panel blocks wind fetch)
- `9373/340` — parallel to IN probe, same longitudinal distance, other lateral side
- `8804/250` — upstream probe, closest to wavemaker

**Pipeline**: `main.py` → raw CSVs in `wavedata/` → processed cache in `waveprocessed/PROCESSED-*/` → exploration in `main_explore_inline.py` (Zed REPL) → publication figures in `main_save_figures.py`

**Known physical complication**: at full wind + low amplitude (0.1 V), the IN probe signal is ~2/3 wind-wave energy — time-domain OUT/IN is meaningless for damping. FFT amplitude at the paddle frequency is the only trustworthy metric.

---

## 0. Current investigation (pick up here next session)

### ⚠ New dataset 13 — still needs processing

**Run `main.py` with `total_reset=False, force_recompute=False`** — datasets 1–12 skip instantly, only dataset 13 (`20260313-ProbePos4_31_FPV_2-tett6roof`) runs the full pipeline. Also need `force_recompute=True` afterwards to pick up the `ak`→`ka` rename and dynamic hard cap changes.

---

### Session summary (2026-03-14) — exploration analysis in `main_explore_inline.py`

#### `ak` → `ka` rename + "Expected *" global wave dimensions

All `ak` column names and attributes renamed to `ka` throughout:
- `constants.py`: `GlobalColumns.KA`, `CalculationResultColumns.KA`, `KA_FFT`, all lists updated
- `wave_physics.py`: local variable + output dict key
- `improved_data_loader.py`: computed-fields init list
- `processor.py`: `"Probe {pos} ka (FFT)"` column name

Global wave dimensions (computed from `WaveFrequencyInput`, not measured) renamed with `"Expected "` prefix to distinguish from FFT-measured values:
- `"Wavenumber"` → `"Expected Wavenumber"`, `"Wavelength"` → `"Expected Wavelength"`, etc.
- `ColumnGroups.GLOBAL_WAVE_DIMENSION_COLS` updated; `filters.py` auto-picks up via `GC.KL` attribute

#### Dynamic hard cap (`processor.py`)

Replaced fixed `CLIP.WAVE_MM = 200` with:
- No-wind wave runs: `clip_mm = CLIP.WAVE_CLIP_FACTOR × volt` (`WAVE_CLIP_FACTOR = 270`)
- Wind wave runs: `clip_mm = WAVE_CLIP_FACTOR × (volt + WIND_BASE_VOLT)` (`WIND_BASE_VOLT = 0.05`)
- Stillwater / nowave: unchanged (`CLIP.NOWIND_MM`)

#### Wave arrival analysis (`main_explore_inline.py`)

**Threshold-based detection** (`find_first_arrival`, existing):
- Fullwind runs always show `arrival_s = 0.000` — wind noise already exceeds `2× noise floor` at t=0. Cannot use time-domain threshold crossing to detect wave arrival under wind.
- Nowind runs: physically coherent arrival times 4–26 s, increasing with frequency (higher freq → shorter wavelength → lower group velocity → later arrival).

**Period-based detection** (`_find_arrival_periodic`, new cell ~L720):
- Slides a window of exactly `N=3` periods at the target frequency, computes FFT amplitude at that bin
- Rejects wind-wave energy (wrong frequency) — both wind and no-wind runs now give physical arrivals
- Nowind arrivals are ~4–7 s later than threshold-based detections (threshold triggers on weak leading edge; period method requires coherent wave energy)
- `_THRESH_FACTOR = 2.0`, `_N_PERIODS = 3`

**Pre-arrival upcrossing frequency** (`_upcrossing_freq`, new cell ~L760):
- Takes the signal from `t=0` to the period-based arrival time, counts zero-upcrossings
- Computes `ratio = pre_arrival_freq / target_freq`
- Plots as 2×2 grid (one subplot per probe) with mean ± half-range across runs
- Investigates whether pre-arrival oscillations are the target wave (just below threshold), a sub-harmonic, or a different mode (possible long-wave precursor)

**Physical note — fast long-wave precursor (OPEN)**:
- Some nowind arrivals are anomalously early (e.g. 1 s at 8804 mm). Max possible surface wave speed in a 580 mm tank is `√(gd) ≈ 2.4 m/s` → min travel time 3.7 s. A 1 s arrival exceeds this limit.
- Hypothesis: wavemaker soft-start ramp generates a broad-spectrum transient including very long-wave (kd→0) components. Known as "piston transient" or "evanescent modes" in lab flumes.
- To investigate: plot first 5–10 s of a few nowind runs at 8804/250 across frequencies.

#### Stillwater noise floor — short-window minimum method (new, ~L497)

The `(P97.5−P2.5)/2` amplitude over a full stillwater run conflates two things:
1. **Residual tank sloshing** — long-period (2–10 s), decays over minutes, makes unsettled runs look noisy
2. **True probe noise** — rapid sample-to-sample jitter from electronics + capillary ripples, constant regardless of settling

New approach: slide a **0.2 s window** (50 samples — shorter than the slowest slosh period) and take the **minimum window amplitude** across the run. Sloshing appears as a flat DC offset within 0.2 s and contributes nothing to `P97.5−P2.5`. Only probe noise remains.

Additional filters:
- `_STILLWATER_EXCLUDE = ["nestenstille"]` — name-based exclusion of known unsettled runs
- `_STILLWATER_AMP_CAP = 0.5 mm` — hard cap on the windowed minimum (flags truly broken probes)

With this method, even the nestenstille run may yield a valid noise floor estimate (its quiet patches are fine).

#### Tank swell tail analysis — mstop90 runs (`main_explore_inline.py` ~L951)

**`mstopXX` naming convention**: recording continues `XX` seconds *after* the wavemaker stops. Total recording = wave duration + XX s. Wave stop time = `total_duration − XX`. `_parse_mstop_tail(path)` reads this from the filename.

**Only mstop90 used** for this analysis — some mstop30 runs are mislabelled (should be mstop0), making their tail duration unreliable.

**Available mstop90 runs** (all from 2026-03-07, fullpanel, fullwind, 1.3 Hz):
- `amp0100-freq1300-per40` × 2 runs
- `amp0200-freq1300-per40` × 1 run
- Total recording: ~120.8 s; wave stop at ~30.8 s; tail = 90 s

**Why lower-frequency runs leave more sloshing**: longer wavelength → excites lower tank resonance modes → more energy stored, slower decay. A 0.65 Hz run is expected to leave dramatically more residual sloshing than a 1.3 Hz run. This is a known dispersion effect — when more runs with different frequencies are available, sort by frequency to compare.

**Backwards PSD stepping** (`_swell_ratio`, `_SLOSH_THRESHOLD`):
- Tail divided into 2 s windows; stepped backwards from end
- Each window: Welch PSD compared to wind-only baseline (same date, fullwind-nowave)
- `_swell_ratio` = integral of swell-band PSD (0–2 Hz) / integral of wind-only baseline in same band
- When ratio > `_SLOSH_THRESHOLD = 3.0` → sloshing still present
- Walking backwards, the last window that exceeds threshold = **clearance time**
- Clearance time shown in subplot title; color scale: red = high sloshing ratio, green = wind-only level
- **Tune `_SLOSH_THRESHOLD`** — 3.0 is first guess, adjust after inspecting the plots

**What the PSD comparison actually measures**: the sloshing appears as excess energy below 2 Hz relative to the wind-only baseline. The wind-only PSD defines the "noise floor" of the free-surface motion at each frequency. Anything above it in the 0–2 Hz band after the wave stops is residual tank resonance / sloshing.

**Time series overview**: wrapped in `if False:` — re-enable to plot all runs. The overview plots are useful for visual sanity checking the wave-stop marker.

---

### Next session — start here

**0. Process dataset 13** — run `main.py` with `total_reset=False, force_recompute=False`

**A. Tank swell clearance** — run the backwards PSD cells, read off clearance times per probe per run. Tune `_SLOSH_THRESHOLD` until it agrees with visual inspection of the time series. Key question: how many seconds after a 1.3 Hz fullwind run before the tank is ready?

**B. Pre-arrival frequency analysis** — run the period-based + upcrossing cells, inspect the 2×2 ratio plot. Key question: is `ratio ≈ 1` (weak leading edge of target wave) or something lower (precursor mode)?

**C. Finalize wind/probe-uncertainty plots → `main_save_figures.py`**
- Stillwater noise floor table is now computed with the short-window method — ready to format for thesis
- Wind PSD plot (~line 314) and probe-uncertainty plot (~line 479) — polish and copy to `main_save_figures.py`

**D. Primary thesis result plots**
- `damping_vs_freq`: OUT/IN (FFT) vs frequency, split by wind and amplitude
- `wave_stability` vs frequency/wind
- `parallel_ratio` vs frequency
- Panel reflection investigation

---

### ✅ Probe distances corrected — cache regenerated, RESOLVED

**All probe distances were physically re-measured and corrected (this session).** Two distances changed:

| Probe group | Old (estimated) | New (measured) | Affected configs |
|-------------|----------------|----------------|-----------------|
| OUT probe (main parallel pair) | `12545 mm` | `12400 mm` | `initial_setup`, `nov_normalt_oppsett`, `march2026_better_rearranging` |
| Temporary OUT probe (Mar 4–6 only) | `12300 mm` | `11800 mm` | `march2026_rearranging` only |

**Every column name in the cache that contained `12545` or `12300` is now stale.** This includes:
- `"Probe 12545/250 Amplitude"`, `"Probe 12545/170 Amplitude"`, `"Probe 12545/340 Amplitude"` → now `12400`
- `"Computed Probe 12545/250 start"` / `end` → now `12400`
- `"wave_stability 12545/250"`, `"period_cv 12545/250"` → now `12400`
- `"samples_clipped_12545/250"`, `"max_gap_12545/250"` → now `12400`
- `"eta_12545/250"` in `processed_dfs` → now `12400`
- `"FFT 12545/250"`, `"Pxx 12545/250"` → now `12400`
- `out_position` column values `"12545/170"` / `"12545/250"` → now `12400`
- Same pattern for `12300` → `11800` for the March 4–6 dataset

**Files changed**:
- `wavescripts/improved_data_loader.py` — `PROBE_CONFIGS` distances updated
- `wavescripts/wave_detection.py` — `_SNARVEI_CALIB` and `_PROBE_GROUP` renamed; `11800` calibration values recalculated at distance fraction 0.802 (between 9373 and 12400): `[(0.65, 4030), (0.70, 4150), (1.30, 6160), (1.60, 6700)]`
- `main_explore_inline.py`, `main_explore_browser.py`, `main_save_figures.py` — all hardcoded probe position strings updated
- `claude.md` — all sections updated

**Cache already regenerated** with `force_recompute=True`. Columns in `combined_meta` now use `12400` / `11800`. `out_position` values are `"12400/250"` (or `"12400/170"` for older configs). Still worth eyeballing `_SNARVEI_CALIB["11800"]` in `RampDetectionBrowser` for the Mar 4–6 datasets — those calibration points are first estimates only.

---

### Design note — minimising hardcoded probe position strings

**Goal**: `improved_data_loader.py` (`PROBE_CONFIGS`) is the single source of truth for all probe distances and lateral positions. No other file should contain hardcoded distance numbers in executable code. Notes and comments are fine.

**Current state after audit:**

| Location | Status | What's there |
|----------|--------|-------------|
| `processor.py`, `processor2nd.py`, `signal_processing.py` | already data-driven | All use `cfg.probe_col_names()` — zero hardcoded positions |
| `plotter.py`, `plot_browsers.py` | already data-driven | Discover positions dynamically from data columns / `combined_meta` |
| `wave_detection.py` `_SNARVEI_CALIB` | acceptable as-is | Calibration table (manually eyeballed values) — keys are distance strings, unavoidably manual |
| `wave_detection.py` `_PROBE_GROUP` | ✅ done | Auto-generated from `PROBE_CONFIGS` at module load — no longer a hardcoded dict |
| `main_explore_inline.py` probe lists | partially reducible | `plotvariables["probes"]` entries are intentional per-plot choices — cannot be fully automated. But the repeated 4-probe `_PROBE_POSITIONS` constant could be a shared import. |
| `main_explore_browser.py` probe lists | partially reducible | Same as above |
| `main_save_figures.py` `PROBE_POSITIONS` | partially reducible | Same as above |

**Both improvements implemented** — `_PROBE_GROUP` is auto-generated, `ANALYSIS_PROBES` is defined in `improved_data_loader.py` and imported everywhere.

**What stays hardcoded (intentionally):**
- `plotvariables["probes"]` lists that select a specific subset for a specific plot — these are deliberate editorial choices, not config duplicates
- `_SNARVEI_CALIB` keys and values — keys must match `_PROBE_GROUP` distance prefixes; values are manually eyeballed sample indices that cannot be derived from `PROBE_CONFIGS`

---

**RESOLVED**: The apparent ~3x wave growth (`nopanel, fullwind, 0.65 Hz, 0.1 V`) was a **pipeline artifact** — root cause was incomplete `find_wave_range` logic. The amplitude window included the wavemaker ramp-up phase, inflating the OUT amplitude. Fixed by improving `_SNARVEI_CALIB`. `main.py` has been re-run; ratios are now plausible.

### New columns added (both in cache after latest `main.py` run)

**`parallel_ratio`** (in `combined_meta`, computed by `processor2nd.py`):
- `= Probe {wall_pos} Amplitude / Probe {far_pos} Amplitude` (wall = smaller lateral, e.g. `/170`; far = larger, e.g. `/340`)
- Detects lateral asymmetry between the parallel probe pair. Near 1.0 = probes agree. Far from 1.0 = asymmetry (wall reflection, lateral wind effect, etc.)
- Added via `ProbeConfiguration.parallel_pair()` in `improved_data_loader.py` — auto-detects which two probes share the same longitudinal distance.

**`wave_stability {pos}`** and **`period_cv {pos}`** (in `combined_meta`, computed by `processor.py` in `run_find_wave_ranges`):
- `wave_stability` = autocorrelation of the wave signal (within the stable window) at lag = 1 period. Near 1.0 = clean repeating wave; lower = phase/amplitude drift across periods.
- `period_cv` = coefficient of variation (std/mean) of per-period peak-to-trough amplitudes. Low = uniform amplitude; high = wave ramping or irregular.
- Computed using FFT-based autocorrelation (`np.fft.rfft`) and `wave_upcrossings` from `debug_info`.
- Available for every probe, every wave run where `find_wave_range` succeeded.

### Key physical finding from wave_stability / period_cv

**Wind-zone context** (confirmed by experiment):
- Probes up to ~9373 mm: **heavy wind exposure** — short wind waves ride on top of paddle wave
- Probes ~11800–12700 mm: **almost no wind** — panel damps the short wind waves almost completely
- Probes at ~18000 mm: wind builds up again over the new fetch after the panel

**Observed in data (March 2026)**:
- `9373/170` (IN probe, full wind exposure): `wave_stability` drops to 0.55–0.74 at low amplitude (0.1V) + full wind. `period_cv` reaches 0.3–0.6 — the signal is dominated by wind-wave noise, not the paddle wave.
- `12400/250` (OUT probe, protected by panel): `wave_stability` stays at 0.93–0.96 even with full wind. `period_cv` stays below 0.10.
- No-wind runs: both probes show `wave_stability` ~0.94–0.96 and `period_cv` ~0.02–0.05 — clean waves.

**Implication for OUT/IN ratio**: full-wind + low-amplitude (0.1V) runs have unreliable IN amplitude at `9373/170`. The time-domain percentile amplitude includes wind-wave energy, inflating the denominator and deflating OUT/IN. Use `wave_stability < 0.85` or `period_cv > 0.15` on the IN probe as a quality flag before trusting OUT/IN.

### Two important next investigations

**1. Quantify wind noise at IN probe using nowave+fullwind runs — DONE (March 2026)**
Wind-only amplitude at full wind (March 2026 mean of 2 nowave+fullwind runs):
- `9373/170`: **10.18 mm** (IN probe — fully exposed to wind)
- `9373/340`: **9.48 mm** (parallel probe — same exposure)
- `8804/250`: **8.41 mm** (upstream probe)
- `12400/250`: **0.88 mm** (OUT probe — panel kills the wind almost completely)

SNR = wave_amplitude / wind_only_amplitude for fullwind wave runs:

| Voltage | IN probe SNR (`9373/170`) | Wind fraction at IN | OUT probe SNR (`12400/250`) | Wind fraction at OUT |
|---------|--------------------------|---------------------|-----------------------------|----------------------|
| 0.1V    | 1.3–1.6                  | **63–77%**          | 5–8                         | 13–20%               |
| 0.2V    | 1.7–3.1                  | **32–61%**          | 10–16                       | 6–10%                |

**Conclusion**: for fullwind runs, time-domain OUT/IN is not measuring damping — it is measuring `(OUT signal) / (paddle wave + wind noise at IN)`. The 0.1V denominator is 2/3 wind. The 0.2V denominator is ≥ 1/3 wind. **Neither is trustworthy for damping conclusions under fullwind conditions using time-domain amplitude.**

**DONE**: `damping_grouper` in `filters.py` now recomputes OUT/IN from `"Probe {pos} Amplitude (FFT)"` (narrow 0.1 Hz window, paddle frequency only) instead of time-domain amplitude. Wind waves are a real physical phenomenon to characterize separately — they should not be folded into the damping ratio. FFT amplitude at the paddle frequency is the correct basis for OUT/IN.

**2. Panel reflection affecting IN probe**
When the panel is present, it reflects incoming waves back toward the wavemaker. The IN probe at `9373/170` sits between the wavemaker and the panel — it may be measuring a superposition of the incident wave and the reflected wave, not the pure incident amplitude. This would make OUT/IN systematically wrong for panel runs (the denominator is inflated by the reflection). To investigate: (a) compare IN probe amplitude with-panel vs without-panel for the same wave condition and no wind, (b) check if the effect is frequency-dependent (reflection coefficient varies with frequency), (c) consider whether FFT amplitude (single-frequency) is less affected than time-domain percentile amplitude, since standing-wave nodes/antinodes depend on probe position relative to wavelength.

### Outlier removal pipeline — Phase 2 DONE (March 2026, exp/probepos branch)

Three-layer outlier removal implemented in `wavescripts/processor.py` (`_zero_and_smooth_signals`):

1. **Hard cap** (Phase 1, already in place): `±200 mm` wave runs, `±5 mm` stillwater-only runs
2. **Velocity filter**: `|diff[k-1]| > CLIP.DIFF_MM=10` AND `|diff[k]| > CLIP.DIFF_MM` AND opposite signs → spike at sample k. Buffer of `CLIP.VEL_BUFFER=2` samples removed on each side (shoulder contamination). Prints `VELCLIP` diagnostic.
3. **Isolated sample check**: single valid sample between NaN gaps on both sides → also set to NaN. Prints `ISOCLIP` diagnostic.

**Display layers** added to `processed_dfs`:
- `eta_{pos}` — truth column, NaN where bad data removed
- `eta_{pos}_interp` — pchip-interpolated display copy; gaps ≤ `max_interp_gap` bridged, longer gaps stay NaN
- `Probe {pos}_ma` — `_eta_for_ma` (linearly filled then re-masked at long gaps) → rolling mean for upcrossing detection

**Dynamic `max_interp_gap`**: wave runs use `int(fs / (4 * freq))` = 1/4 wavelength; nowave/stillwater use `CLIP.INTERP_MAX_GAP=10` fallback.

**New quality columns in `combined_meta`** (after `main.py` run):
- `samples_clipped_{pos}` — total samples set to NaN for that probe/run (all three layers)
- `max_gap_{pos}` — longest consecutive NaN run in `eta_{pos}` for that probe/run

**Constants added to `CLIP` in `constants.py`**: `DIFF_MM=10.0`, `INTERP_MAX_GAP=10`, `VEL_BUFFER=2`

**`RampDetectionBrowser` fixes and improvements** (`plot_browsers.py` + `plotter.py`):
- Fixed `baseline_mean` NaN crash: falls back to `nanmean(base_region)` if `Stillwater Probe {pos}` is NaN in meta
- Three signal layers: Raw (lightgray), Cleaned interp (steelblue), Cleaned NaN (black)
- Fixed probe filter crash for `/`-separated position strings (`int()` → string comparison)
- **Expected sine toggle**: checkbox "Show expected sine" — FFTs the stable window, extracts amplitude+phase at target freq, overlays `baseline_mean + A·sin(2π·f·t + φ)` as dashed orange line

**NOTE**: `main.py` has NOT yet been re-run with these changes. Run with `force_recompute=True` to regenerate cache and verify:
- Zero `VELCLIP` messages on clean runs
- `ISOCLIP` fires for isolated sample at idx ~37794 in `reversepanel-nowind-amp0300-freq1300...run1 → 12400/340`
- `samples_clipped_*` and `max_gap_*` columns appear in `combined_meta`

---

### Pipeline changes made today

**Nowave PSDs now in cache** (`processor.py` + `signal_processing.py`):
- New function `compute_nowave_psd` in `signal_processing.py` computes broadband Welch PSD for all nowave runs (full `eta_` signal, `nperseg=4096`, no wave-range window).
- Called in `process_selected_data` after the wave PSD, merged into `psd_dict` before saving.
- `psd_dict` / `combined_psd_dict` now contains **all 143 runs** (132 wave + 11 nowave) — no format change.
- `main_explore_inline.py` wind section: `_wind_psd_dict` is now a one-line filter of `combined_psd_dict` — no `processed_dfs` needed for the PSD plot.
- `processed_dfs` lazy-load is still needed for the stats cell (343), stillwater plot (439), and arrival detection (506).

### Next session — start here

**0. Process the new dataset first** ⚠ NOT YET DONE
- Run `main.py` with `total_reset=False, force_recompute=False`
- Expect: datasets 1–12 print `✓ Already processed — skipping`, dataset 13 runs the full pipeline
- Dataset 3 (`20251110-tett6roof-lowMooring`) will also re-run (it lacks `processed_dfs.parquet` / `fft_spectra.parquet` / `psd_spectra.parquet`). It will now print `Both anchors share the same timestamp — using flat value` instead of crashing.
- After run: spot-check `combined_meta` for dataset 13 rows — check `Probe 12400/250 Amplitude`, `wave_stability 12400/250`, `Stillwater Probe 1`–`4` are non-NaN and physically plausible. Use `wavetable` alias to open dtale.

**1. Quick sanity checks**
- Open `RampDetectionBrowser`, enable "Show expected sine" checkbox, verify dashed orange line fits stable wave at a few runs
- Spot-check `11800/250` metrics for the March 4–6 datasets are non-NaN and physically plausible:
```python
_mar46 = combined_meta[combined_meta["file_date"].str.startswith("2026-03-0")]
print(_mar46[["file_date", "Probe 11800/250 Amplitude", "wave_stability 11800/250", "Probe 11800/250 Amplitude (FFT)"]].to_string())
```

**A. Finalize wind/probe-uncertainty plots → `main_save_figures.py`**
- In `main_explore_inline.py`, the wind section has a working probe-uncertainty/stillwater plot (around line 479) and wind PSD plots. Review these, polish, then copy the finalized `plotvariables` + function calls to `main_save_figures.py` following the explore→publication chain (§18).
- The wind PSD plot at ~line 314 is also a candidate — shows broadband wind spectrum per probe with stillwater baseline.

**B. More plots and visualisations**
- `damping_vs_freq`: OUT/IN ratio vs frequency, split by wind condition and amplitude — the primary damping result. Uses `explore_damping_vs_freq` / `plot_damping_freq`.
- `wave_stability` vs frequency/wind — show where the IN probe quality degrades. Useful as a companion figure to the damping plot.
- `parallel_ratio` vs frequency — does lateral asymmetry depend on frequency or wind?
- Panel reflection investigation (see §0 item 2): plot IN amplitude with-panel vs without-panel, no wind, across frequencies.

**C. Tables**
- Probe noise floor table: stillwater amplitude per probe, formatted for thesis (already computed in cell 479 — extract to `plotter.py` or a LaTeX stub).
- SNR table: wind-only amplitude vs wave amplitude at 0.1V and 0.2V — already computed this session, formalize it.
- OUT/IN summary table: mean ± std per (frequency, amplitude, wind, panel) group — the core damping result in tabular form.

### Still open
- **Ramp detection for `11800/*` probes — FIXED**: `_SNARVEI_CALIB["11800"]` and `_PROBE_GROUP` in `wave_detection.py` are now correct. Ramp detection returns the right `good_start_idx` / `good_end_idx` for `11800/250` runs. Post-`_eta` processing was suspected broken (wrong amplitude / wave_stability / FFT) but plots are now working normally after `force_recompute` — status unclear. **Still worth verifying**: in REPL check `combined_meta` columns `"Probe 11800/250 Amplitude"` and `"wave_stability 11800/250"` for the March 4–6 datasets are non-NaN and physically plausible. If NaN: check `_extract_stillwater_levels` for probe 2 in `march2026_rearranging` (NaN stillwater → entire `eta_11800/250` = NaN → all downstream metrics silently fail).
- `9373/250` noise floor 0.600 mm in Nov 2025 vs `9373/170` ~0.32 mm in March 2026 — likely different physical probes, not a position effect. No action needed unless it affects a key result.
- **Wind-wave characterization**: cross-correlate `/170` and `/340` at the same longitudinal distance for fullwind runs to test lateral coherence of wind waves. Use `processed_dfs` eta columns + `scipy.signal.correlate`.
- **NaN wave_stability runs**: a handful of runs show NaN for wave_stability (e.g. some 0.7 Hz, 1.4 Hz full-wind runs). These are cases where `find_wave_range` returned `None` start/end. Investigate in `RampDetectionBrowser`.
- Lateral ratio sanity check (no-wind runs, `parallel_ratio` should be ~1.0):
```python
_wave = combined_meta[combined_meta["WaveFrequencyInput [Hz]"].notna()].copy()
print(_wave[["WaveFrequencyInput [Hz]", "WindCondition", "PanelCondition", "parallel_ratio"]]
      .sort_values("parallel_ratio", ascending=False).to_string())
```

**Dead theories — do not revisit without new evidence**:
- Wall reflection at `12400/170` creating constructive interference
- Wind skewing wave crests laterally
- Wrong `out_position` assignment
- Wind-wave contamination inflating time-domain amplitude (now quantified via `wave_stability`/`period_cv` — effect is real but understood)

---

## 1. Project overview

Wave-tank experiment analysis pipeline:

- Raw CSV runs in `wavedata/`
- `main.py` processes CSVs → cache in `waveprocessed/PROCESSED-*`
- Exploration scripts load processed cache, never raw CSVs
- Probes identified by **physical position**, not probe number 1–4

Repo: `https://github.com/OleBB/wave_project`

---

## 2. Environment

- OS: macOS, Editor: Zed, conda (never pip)
- Active local env: `draumkvedet`
- Exported/shared env name: `draumeriket`

```yaml
name: draumeriket
channels:
  - defaults
dependencies:
  - python=3.11
  - spyder=6.1.0
  - notebook, spyder-notebook, spyder-unittest
  - numpy, scipy, pandas, matplotlib, seaborn, plotly
  - pytest, sympy, pyarrow, tabulate
```

---

## 3. Entry points (repo root)

| File | Role | How to run |
|------|------|------------|
| `main.py` | Full pipeline: CSV → processed cache | `python main.py` |
| `main_explore_inline.py` | Primary analysis playground, `# %%` cells | Open in Zed REPL |
| `main_explore_browser.py` | Qt GUIs for interactive run browsing | `python main_explore_browser.py` |
| `main_save_figures.py` | Batch LaTeX/PGF figure export | `python main_save_figures.py` |
| `dtale_meta.py` | Open `combined_meta` in dtale browser, nothing else | `python dtale_meta.py` or shell alias `wavetable` |

`main_explore_browser.py` forces `matplotlib.use("Qt5Agg")` — run from terminal, not REPL.

Shell alias `wavetable` is saved in `~/.zshrc` → `cd ~/Kodevik/wave_project && conda activate draumkvedet && python dtale_meta.py`. Type `wavetable` from any terminal to open the table instantly.

See §19 for the full three-phase call hierarchy and plotting script roles.

---

## 4. Data loading

```python
combined_meta, processed_dfs, combined_fft_dict, combined_psd_dict = load_analysis_data(
    *PROCESSED_DIRS, load_processed=False   # default — fast path, ~2 s
)
```

- `load_processed=False` (default): skips 75 MB `processed_dfs.parquet`, loads meta + FFT/PSD only (~2 s)
- `load_processed=True`: also loads full time-series `processed_dfs` (~+20 s)
- `processed_dfs` is lazy-loaded in `main_explore_inline.py` just before the wind-only section:
  ```python
  if not processed_dfs:
      processed_dfs = load_processed_dfs(*PROCESSED_DIRS)
  ```
- `waveprocessed/` is **gitignored** — all caches are local, regenerated by `main.py`
- The 3 dataset directories are loaded **in parallel** via `ThreadPoolExecutor` (I/O-bound)

### What each variable contains

- `combined_meta`: DataFrame, one row per run (wave + nowave), all runs
- `processed_dfs`: `{csv_path: DataFrame}` of zeroed+smoothed time series (empty if `load_processed=False`)
- `combined_fft_dict`: `{csv_path: DataFrame}` for **wave runs only** — columns `"FFT {pos}"` + `"FFT {pos} complex"`
- `combined_psd_dict`: `{csv_path: DataFrame}` — columns `"Pxx {pos}"`

### FFT/PSD parquet storage

- Complex columns split into `col_real` / `col_imag` float32 pairs on save, recombined to complex128 on load
- All floats downcast to float32 to halve file size
- On load: bulk-cast all float32 → float64 once, recombine complex once, then split by path via `groupby` (not per-path boolean masking)

### `repl_out` — tee stdout to file

```python
with repl_out("filename.txt"):
    print(...)   # goes to terminal AND repl/filename.txt
```

Defined in `main_explore_inline.py`. Output files live in `repl/` (gitignored).

---

## 5. Probe naming convention (CRITICAL)

### Always `distance_mm/lateral_mm`

Every probe position is always written as `"longitudinal/lateral"` — even for probes with a unique longitudinal distance:

| Probe | Position string |
|-------|----------------|
| 9373 mm from paddle, center (250 mm) | `"9373/250"` |
| 9373 mm from paddle, near wall (170 mm) | `"9373/170"` |
| 9373 mm from paddle, far side (340 mm) | `"9373/340"` |
| 12400 mm, center | `"12400/250"` |
| 12400 mm, near wall | `"12400/170"` |
| 12400 mm, far side | `"12400/340"` |
| 8804 mm, center | `"8804/250"` |

`probe_col_name()` always returns `f"{dist}/{lat}"` — no parallel-detection logic.

**Do not** use plain-number names like `"9373"`, `"12400"`, `"8804"` — these were the old convention, replaced in Mar 2026.

### Column name patterns

- Raw signal: `"Probe 9373/250"`
- Processed elevation: `"eta_9373/250"`
- Smoothed: `"Probe 9373/250_ma"`
- Amplitude (time-domain, percentile): `"Probe 9373/250 Amplitude"` ← used by `plot_all_probes` and `damping_grouper`
- FFT amplitude: `"Probe 9373/250 Amplitude (FFT)"`
- PSD amplitude: `"Probe 9373/250 Amplitude (PSD)"`
- FFT spectrum: `"FFT 9373/250"`, `"FFT 9373/250 complex"`
- PSD spectrum: `"Pxx 9373/250"`

**Do not** reintroduce probe numbers (1–4) in user-facing code.

### `in_position` / `out_position` in combined_meta

- Set by `processor2nd.py` from `ProbeConfiguration.in_probe` / `out_probe` via `probe_col_name()`
- Stored as position strings: `"9373/250"`, `"12400/170"`, etc.
- Used by `damping_grouper` to recompute OUT/IN ratio on-the-fly from plain amplitude columns

---

## 6. Known pitfalls / gotchas

### `apply_dtypes` destroys position strings with `/`

`apply_dtypes` in `improved_data_loader.py` calls `pd.to_numeric(..., errors="coerce")` on all columns not in `NON_FLOAT_COLUMNS`. Position strings containing `/` (e.g. `"12400/170"`) become **NaN**. Plain-number strings (e.g. `"9373"`) become floats (`9373.0`).

**Fix already applied**: `in_position` and `out_position` are now in `NON_FLOAT_COLUMNS`.

**Rule**: Any new string-typed column whose value may contain `/`, letters, or other non-numeric characters **must** be added to `NON_FLOAT_COLUMNS`. Forgetting this causes silent NaN corruption that is very hard to debug.

### `np.percentile` propagates NaN in matrix amplitude computation

`_compute_matrix_amplitudes` in `signal_processing.py` builds a matrix of probe samples and calls `np.nanpercentile`. If `np.percentile` (without `nan`) is used instead, **any probe with even 1 NaN sample in its range gets NaN amplitude** — including all nowave runs (which use the full signal range). Fixed by changing to `np.nanpercentile`.

### Stale `OUT/IN (FFT)` in meta.json

`meta.json` may contain `OUT/IN (FFT)` values computed with an old wide FFT window (`0.5 Hz`, `argmax`) that picks up wind-wave peaks instead of paddle-wave peaks. Do not trust cached `OUT/IN (FFT)`.

`damping_grouper` now recomputes OUT/IN on-the-fly from `"Probe {pos} Amplitude (FFT)"` columns (paddle frequency, narrow 0.1 Hz window). It falls back to the cached value only if recomputation yields 0 valid rows (prints a diagnostic).

### Two amplitude types — not interchangeable

| Column | Source | Used by |
|--------|--------|---------|
| `"Probe {pos} Amplitude"` | Percentile of time-domain signal | `plot_all_probes` |
| `"Probe {pos} Amplitude (FFT)"` | FFT peak near target frequency | Old OUT/IN cached values |

Always use `"Probe {pos} Amplitude"` (no suffix) for OUT/IN ratio computation.

### FFT amplitude window

`compute_amplitudes_from_fft` uses `window=0.1` Hz and `argmin(abs(masked_freqs - target_freq))` (nearest bin). Old code used `window=0.5` Hz + `argmax`, which picked up wind-wave peaks for low-amplitude runs.

### `_SNARVEI` probe name matching

`find_wave_range` in `wave_detection.py` uses `_PROBE_GROUP` dict to map all lateral variants of a probe to a distance group (e.g. `"Probe 12400/170"` → `"12400"`). If a new probe position is added, it **must** be added to `_PROBE_GROUP` — otherwise range detection falls back to `2 * samples_per_period` (stillwater phase), giving near-zero amplitudes and OUT/IN ≈ 0.1.

---

## 7. Wave range detection (`_SNARVEI_CALIB`)

Defined in `wavescripts/wave_detection.py`. Multi-point linear interpolation of stable-wave start sample vs frequency, calibrated by eyeballing `RampDetectionBrowser`.

```python
_SNARVEI_CALIB = {
    "8804":  [(0.65, 3975), (1.30, 4700), (1.80, 6000)],
    "9373":  [(0.65, 4075), (0.70, 3750), (1.30, 4800), (1.60, 5500)],
    "11800": [(0.65, 4030), (0.70, 4150), (1.30, 6160), (1.60, 6700)],  # march2026_rearranging only; needs eyeballing
    "12400": [(0.65, 4020), (0.70, 4250), (1.30, 6500), (1.60, 7000)],
}
```

**Key insight**: ramp-up duration (13–20 periods) dominates the start time — wave travel time (< 5 s) is a minor secondary effect. All probes in a run see their first stable peak at nearly the same sample index, with only a small per-probe offset from travel time. The "first stable peak" is the second visible peak in the ramp: the first peak is still part of the wavemaker's soft-start program and is unreliable.

- Format: `(freq_hz, start_sample)` sorted by frequency; samples at 250 Hz (ms / 4)
- Interpolates linearly between points; extrapolates linearly beyond the range
- `_PROBE_GROUP` maps every probe column name variant to a distance group key
- To add a calibration point: eyeball start in `RampDetectionBrowser`, convert ms → samples (/4), add tuple
- `8804` group has 3 points (0.65, 1.30, 1.80 Hz) — extrapolates outside that range

### TODO: investigate wavemaker ramp-up shape

The wavemaker controller uses frequency-dependent acceleration profiles — higher frequencies have a different (longer?) soft-start program. This means the signal **before** the eyeballed good-start index is not simply "stillwater + linear ramp": it contains a wavemaker-programmed pre-ramp that varies by frequency.

Before relying on the region before `good_start_idx` for anything (e.g. stillwater baseline, ramp characterization), we must understand what the controller actually does in that window. The `_SNARVEI_CALIB` start values are conservative eyeballs at the first clearly stable period — the true stable onset may be 1–2 periods earlier or later depending on frequency. Needs systematic inspection in `RampDetectionBrowser` across frequencies.

---

## 8. Probe configurations over time

Defined in `improved_data_loader.py` as `PROBE_CONFIGS`:

| Config name | Valid from | in_pos | out_pos | Notes |
|-------------|-----------|--------|---------|-------|
| `initial_setup` | Aug 2025 | `9373/250` | `12400/170` | Probe 1 far back at 18000 mm |
| `nov_normalt_oppsett` | Nov 10 2025 | `9373/250` | `12400/170` | Probe 1 moved to 8804 mm |
| `march2026_rearranging` | Mar 4 2026 | `9373/170` | `11800/250` | Temporary, 2 days |
| `march2026_better_rearranging` | Mar 7 2026 | `9373/170` | `12400/250` | Current layout |

`get_configuration_for_date(file_date)` selects the right config.

---

## 9. Run types

- **Wave runs**: `WaveFrequencyInput [Hz]` > 0 — appear in `fft_dict` / `psd_dict`
- **Nowave runs**: `WaveFrequencyInput [Hz]` is NaN or `"nowave"` in filename
  - Stillwater: `WindCondition == "no"`
  - Wind-only: `WindCondition in {"full", "lowest"}`
- Both amp and freq tags must be present in filename to set wave parameters

---

## 10. Core modules (`wavescripts/`)

- **`improved_data_loader.py`**: `ProbeConfiguration`, `PROBE_CONFIGS`, `load_analysis_data`, `load_processed_dfs`, `save_spectra_dicts`, `load_spectra_dicts`, `apply_dtypes`, `NON_FLOAT_COLUMNS`
- **`processor.py`**: `process_selected_data` — full pipeline called by `main.py`
- **`processor2nd.py`**: post-processing after main pipeline — sets `in_position`, `out_position`, `OUT/IN (FFT)`, band amplitudes
- **`signal_processing.py`**: `compute_fft_with_amplitudes`, `compute_psd_with_amplitudes`, `compute_amplitudes_from_fft`
- **`filters.py`**: `apply_experimental_filters`, `filter_for_frequencyspectrum`, `damping_grouper`, `damping_all_amplitude_grouper`
- **`plotter.py`**: `plot_all_probes`, `plot_damping_freq`, `plot_frequency_spectrum`, `plot_reconstructed`, `plot_swell_scatter`
- **`plot_quicklook.py`**: `explore_damping_vs_freq`, `explore_damping_vs_amp`, `save_interactive_plot` — no Qt, no save_plot
- **`plot_browsers.py`**: `SignalBrowserFiltered`, `RampDetectionBrowser` (Qt, only imported when used)
- **`constants.py`**: `MEASUREMENT` (sampling rate 250 Hz), `GlobalColumns (GC)`, `ProbeColumns (PC)`, `ColumnGroups (CG)`

---

## 11. Damping / OUT/IN analysis

`explore_damping_vs_freq` (in `plot_quicklook.py`) uses `damping_grouper` from `filters.py`.

`damping_grouper`:
- Groups by: `WaveFrequencyInput [Hz]`, `WaveAmplitudeInput [Volt]`, `WindCondition`, `PanelCondition`, `Mooring`
- Recomputes `OUT/IN` from `"Probe {in_position} Amplitude"` / `"Probe {out_position} Amplitude"` per row
- Requires `in_position` and `out_position` to be valid strings in `combined_meta` (not NaN, not float)
- Falls back to cached `OUT/IN (FFT)` with a diagnostic print if recompute fails

`damping_all_amplitude_grouper`: same grouping, but across all amplitude levels.

---

## 12. Wind-only analysis

In `main_explore_inline.py` (lazy-loaded section):

- Filters `combined_meta` for nowave runs
- Builds `wind_psd_dict` using `scipy.signal.welch` on `eta_{pos}` columns from `processed_dfs`
- Same dict format as `psd_dict`: `{path: DataFrame(index=Frequencies, cols="Pxx {pos}")}`
- Plots with `plot_frequency_spectrum(..., data_type="psd", facet_by="probe")`
- Computes mean (wind setup) and std (RMS fluctuations) per probe

---

## 13. Debugging tips

### Inspect a single filtered run

```python
from wavescripts.filters import apply_experimental_filters
_sel = apply_experimental_filters(combined_meta, myplotvariables)
amp_cols = [c for c in _sel.columns if "Amplitude" in c and "FFT" not in c and "PSD" not in c]
print(_sel[["path", "file_date", "in_position", "out_position", "OUT/IN (FFT)"] + amp_cols].T.to_string())
```

`.T` (transpose) is essential — with 1 row and many columns it prints much more readably.

### View a DataFrame interactively (Zed REPL)

- Last expression in a cell: renders as HTML table inline
- `df.to_clipboard()` → paste into Numbers/Excel
- `df.to_html("/tmp/x.html"); import subprocess; subprocess.run(["open", "/tmp/x.html"])`

### Reload a module without restarting REPL

```python
import importlib
import wavescripts.filters as f
importlib.reload(f)
```

---

## 14. Git workflow

- Never commit to `main` directly
- Branch: `git checkout -b exp/<what-you-try>`
- Safety snapshot: `git commit -am "safety: working before I break it"`
- Merge to main after experiment works, then delete branch
- `waveprocessed/` is gitignored — never commit it

---

## 15. Testing (pytest)

```bash
pytest -q                          # all tests
pytest -q tests/test_sandkasse.py  # single file
pytest -q -k test_name             # single test
pytest -vv / -s / -x               # verbose / show prints / stop at first fail
```

When changing analysis logic, propose tests using small synthetic data that assert on key outputs (peak counts, amplitudes, wavenumbers).

---

## 16. Physical assumptions — always in mind

These are not negotiable. Every analysis decision must be consistent with them.

### Measurement precision
- **Target resolution: 0.25 mm** (quarter millimeter). No discrepancy is too small to investigate.
- Stillwater noise floor is **probe-dependent** — measured as `"Probe {pos} Amplitude"` = (P97.5−P2.5)/2 from no-wind, no-wave runs:

Reference data: `wave_project/dtale-probe-uncertainty.csv` (5 rows, paths included).

Run identity per row:

| Row | Path (short) | Status |
|-----|-------------|--------|
| 1 | `20260307/.../nestenstille.csv` | ⚠ **outlier** — "almost still", water not settled |
| 2 | `20260307/.../nowave-depth580-run1.csv` | Normal |
| 3 | `20260307/.../nowave-depth580-run2.csv` | Normal |
| 4 | `20260307/.../wavemakeroff-1hour-stillwater.csv` | ✓ **gold standard** — most settled |
| 5 | `20251112/.../nopanel-nowind-nowave-per40-run1.csv` | Nov 2025, different probe config |

Measured noise floor per probe (excluding row 1 outlier):

| Probe | Gold std (row 4) | Runs 2–4 range | Notes |
|-------|-----------------|---------------|-------|
| `8804/250` | 0.330 mm | 0.315–0.350 | **~0.33 mm** — stable |
| `8804/170` | — | 0.260 (row 5) | Single Nov-2025 measurement |
| `9373/170` | 0.330 mm | 0.305–0.330 | **~0.32 mm** — stable |
| `9373/250` | — | 0.600 (row 5) | ⚠ Nov-2025 only — suspiciously high; probe calibration issue? |
| `9373/340` | 0.075 mm | 0.075–0.315 | **Unreliable — 4× spread across settled runs** |
| `12400/250` | 0.130 mm | 0.130–0.165 | **~0.14 mm — quietest, most stable** |
| `12400/170` | — | 0.305 (row 5) | Single Nov-2025 measurement |
| `12400/340` | — | 0.255 (row 5) | Single Nov-2025 measurement |

- **Gold standard noise floor**: use row 4 (`wavemakeroff-1hour`) values — tank maximally settled.
- `9373/340` high variability (0.075–0.315 mm across runs on same day) is unexplained — probe sensitivity or positioning issue.
- `9373/250` = 0.600 mm in Nov 2025 while `9373/170` ≈ 0.32 mm in March 2026 — same longitudinal distance, factor-of-2 difference. Likely probe-specific calibration difference between the two physical probes used at those times.
- Detection threshold: **2× probe noise floor** individually. For `12400/250` → ~0.26 mm; for `8804/250` / `9373/170` → ~0.65 mm.
- Any amplitude below the probe's own noise floor is indistinguishable from noise — must be flagged, not reported as signal.

### Wave physics
- **Wind waves exist only above ~2 Hz** — no wind-wave energy at paddle frequencies (0.65–1.8 Hz) in the PSD sense.
- BUT wind waves (3–5 Hz, broad, erratic) **ride on top** of the paddle wave in the time domain. Time-domain percentile amplitudes include ALL frequency content. FFT amplitude at the target frequency does not.
- **Two amplitude types are not interchangeable**:
  - `"Probe {pos} Amplitude"` = (P97.5−P2.5)/2 of time-domain signal — includes wind waves
  - `"Probe {pos} Amplitude (FFT)"` = FFT peak within 0.1 Hz of target — paddle-wave only
- The **OUT/IN ratio** must always be computed from `"Probe {pos} Amplitude (FFT)"` (paddle frequency only). Time-domain amplitude includes wind-wave energy which inflates the IN probe under fullwind conditions, making OUT/IN meaningless for damping. Wind waves are a real physical phenomenon to characterize separately, not noise to average into the damping ratio.

### Probe geometry
- Parallel probes at the same longitudinal distance (e.g. `9373/170` and `9373/340`) are **not redundant** — they measure lateral wave non-uniformity. A factor-of-2 difference between them is physically meaningful and must be explained, not averaged away silently.
- Wall-side probe (`/170`) is closer to the tank wall — susceptible to wall reflections and wind-driven lateral asymmetry.
- Center probe (`/250`) is the most representative single measurement of the 1D wave field.

### Wave arrival
- First stable wave energy arrives at ~12400 mm in approximately **10 seconds** from paddle start (frequency-dependent).
- Wavemaker ramp-up (13–20 periods) dominates the pre-stable window — not wave travel time.
- Anything arriving before ~0.5 s at any probe is a wind-wave or instrument artifact, not a paddle wave.

### Stillwater as ground truth
- Stillwater (no wind, no wave) defines the true zero and the noise floor for each probe.
- Every probe's `"Probe {pos} Amplitude"` in a stillwater run is a direct noise floor measurement.
- This noise floor must be measured fresh per probe position — it is not transferable across configurations.

---

## 17. Rigorous analysis workflow

Precision standard: **0.25 mm**. Nothing is too small to ignore. Follow this sequence whenever a new result or anomaly appears.

### 1. Before trusting any amplitude
- [ ] Verify `in_position` and `out_position` are correct for the run's date (`get_configuration_for_date`)
- [ ] Confirm the stillwater noise floor for the relevant probe and date — is the signal above 2× noise?
- [ ] Check for NaN in the amplitude columns (`_sel.T.to_string()` with transpose)

### 2. Before trusting an OUT/IN ratio
- [ ] Check both time-domain AND FFT amplitude — do they agree? If not, wind-wave contamination is likely.
- [ ] Compare parallel probes at the same distance — do they agree within ~10%? Factor-of-2 disagreement requires investigation.
- [ ] Confirm n (number of runs) — for n=1, apply ±10% fallback errorbar, not a hard conclusion.
- [ ] Never trust cached `OUT/IN (FFT)` — always recompute from `"Probe {pos} Amplitude"` columns.

### 3. Diagnosing an anomaly
Systematic elimination order:
1. **Noise**: is the amplitude above 2× stillwater noise floor for that probe?
2. **Wind-wave contamination**: compare time-domain vs FFT amplitude — does the anomaly survive FFT isolation?
3. **Lateral asymmetry**: compare both parallel probes — is one side inflated?
4. **Probe config error**: was `in_position`/`out_position` assigned correctly for that run's date?
5. **Data quality**: are there NaN samples? Was the run cut short? Check `processed_dfs` time series directly.
6. **Physics**: only after 1–5 are ruled out, conclude the effect is real.

### 4. Plotting
- Always show errorbars. Use `std` when n>1, ±10% fallback when n=1.
- Parallel probes: average and show half-range errorbar — never plot both as independent points without comment.
- Y-axis must always be shared (`sharey=True`) when comparing across conditions or frequencies.
- Color = the physically meaningful primary variable. Linestyle = secondary modifier (e.g. wind condition).

### 5. Recording results
- If a value changes after pipeline fix (e.g. re-running `main.py`), note both the old and new value and what changed.
- `repl_out("filename.txt")` to capture diagnostic prints permanently.
- Update `CLAUDE.md §0` (current investigation) whenever a conclusion changes.

---

## 18. Script architecture and call hierarchy

### The three phases

```
PHASE 1 — PIPELINE (run once, or when data changes)
─────────────────────────────────────────────────────────────────
  main.py
    ├─ processor.py           raw CSV → zeroed+smoothed time series,
    │                          FFT, PSD, wave-range detection,
    │                          stillwater anchor, probe noise floors
    ├─ processor2nd.py        post-processing: in/out positions,
    │                          OUT/IN ratio, band amplitudes
    └─ improved_data_loader.py  saves → waveprocessed/PROCESSED-*/
                                          meta.json, fft.parquet,
                                          psd.parquet, processed_dfs.parquet

PHASE 2 — EXPLORATION (human analysis, after pipeline)
─────────────────────────────────────────────────────────────────
  main_explore_inline.py      # %% cells in Zed REPL — primary playground
  main_explore_browser.py     Qt GUIs — interactive browsing / calibration
                              (forces Qt5Agg, run from terminal)

  Both load waveprocessed/ cache — NEVER raw CSVs.
  All save_plot keys are permanently False here.

PHASE 3 — EXPORT (when a plot is ready)
─────────────────────────────────────────────────────────────────
  main_save_figures.py        copy plotvariables + call from exploration,
                              set save_plot=True, run as script
                              → output/FIGURES/  (PDF + PGF)
                              → output/TEXFIGU/  (LaTeX stubs, written once)
```

### Plotting script hierarchy

| Script | Role | Stability |
|--------|------|-----------|
| `plot_utils.py` | Style + save infrastructure: `apply_thesis_style`, `save_and_stub`, `build_fig_meta`, `WIND_COLOR_MAP` | **Core — never dead code** |
| `plotter.py` | Reusable publication-grade plot functions: `plot_all_probes`, `plot_damping_freq`, `plot_frequency_spectrum`, `plot_swell_scatter` | **Core — stable public API** |
| `plot_quicklook.py` | Fast exploratory functions: `explore_damping_vs_freq`, `explore_damping_vs_amp` — no save_plot, no TeX stubs | Exploratory — **will accumulate dead code** |
| `plot_browsers.py` | Qt interactive browsers: `SignalBrowserFiltered`, `RampDetectionBrowser` — diagnostic / calibration only | Diagnostic — stable but narrow scope |

### Explore → publication call chain

```
main_explore_inline.py
  │  (experiment, iterate, all save_plot=False)
  │  "looks right"
  ▼
main_save_figures.py
  │  (copy plotvariables dict + function call, set save_plot=True)
  │  calls
  ▼
plotter.py  (stable, reusable plot function)
  │  calls at the end
  ▼
plot_utils.save_and_stub(fig, meta, plot_type)
  ├─ output/FIGURES/{filename}.pdf   ← include in LaTeX
  ├─ output/FIGURES/{filename}.pgf   ← PGF native
  └─ output/TEXFIGU/{filename}.tex   ← stub written ONCE, never overwritten
```

`plot_quicklook.py` functions are **outside this chain** — they are never called from `main_save_figures.py`. Once an exploratory function matures into a publishable plot, it either calls an existing `plotter.py` function or a new one is added to `plotter.py`.

### Expected evolution

- `plot_quicklook.py` will grow dead functions as the analysis moves on — this is intentional. Only functions actively imported in `main_explore_*.py` should be considered live.
- `plotter.py` grows slowly and deliberately — every function here has a corresponding call in `main_save_figures.py`.
- The stubs in `output/TEXFIGU/` are write-once: captions and `\label` are edited by hand after generation, never regenerated (use `force_stub=True` only after a git commit).

---

## 19. Thesis structure and key variables

### Where keys appear

**Methodology plots (Ch04)** — diagnostic. Keys shown only as needed (e.g. frequency matters for wave-range detection; wind condition matters for noise floor). No need to show all keys on every methodology figure.

**Results plots and tables (Ch05)** — every wave-data figure must give the reader enough context to know what wave they are looking at. The reader-facing keys are:
- `ka` — the primary wave descriptor (see below). Replaces raw frequency + voltage for the reader.
- `PanelCondition` — always shown (it is the geometry variable being studied)
- `WindCondition` — always shown (it is the forcing variable; the central question)

`WaveAmplitudeInput [Volt]` and `WaveFrequencyInput [Hz]` are **writer/script-facing** — useful in code and internal tables, but not reader-friendly in figures. They are encoded inside `ka`.

**Script-facing keys** (used in filters, column names, `plotvariables`):
- `WaveAmplitudeInput [Volt]` — 0.1 V / 0.2 V
- `WaveFrequencyInput [Hz]` — 0.65–1.9 Hz
- `PanelCondition` — full / reverse / no
- `WindCondition` — full / lowest / no

**Reader-facing output keys** (shown on figures):
- `OUT/IN (FFT)` — damping ratio. Always from `"Probe {pos} Amplitude (FFT)"` (paddle freq only, 0.1 Hz window). Wind waves excluded.
- `ka` — wavenumber × amplitude, measured per probe per run (not pre-calculated). Encodes both wavelength (hidden in k) and wave steepness (via a). Almost an all-in-one wave descriptor for the reader.

### Thesis chapter outline (`main_save_figures.py` is the backbone)

**Chapter 04 — Methodology:**
1. Probe uncertainty / noise floor — stillwater amplitude per probe, detection threshold
2. Stillwater timing — how long between runs; low-freq swell decay; wind shortens wait
3. Probe placement — longitudinal/lateral effects, what parallel probes reveal
4. Wind characterisation — wind PSD, spatial extent, lateral coherence, SNR at IN vs OUT
5. Full signal overview — annotated time-domain: ramp, stable train, wind riding on wave
6. Wave-range detection — _SNARVEI_CALIB, threshold crossing, stable wavetrain window
7. Autocorrelation A — wavetrain stability (`wave_stability`, `period_cv`)
8. Autocorrelation B — lateral equality (parallel probes, wind vs no-wind)
9. (additional steps TBD from processor / processor2nd logic)

**Chapter 05 — Results:**
1. Damping vs frequency — OUT/IN (FFT) vs Hz (and vs ka). The central result.
2. Damping vs amplitude — weaker effect, but absence of effect is itself a finding.
3. Wind effect on damping — the single key question: "How does wind affect damping?"
   Formally: "How much of the paddle-frequency wave survives through the panel geometry, when wind is added?"

### The ka debate

`ka` is not trivial to define because the panel changes both amplitude and effective wavenumber between IN and OUT:
- Frequency changes very little (panel does not alter wave period significantly).
- Amplitude can drop up to ~95% through the panel geometry.
- The IN-side `ka` (at `9373/170`, no-panel run) represents the "undisturbed" incident wave from the wavemaker — the ideal reference.
- In reality with panel present, IN probe sees incident + reflected wave superposition. OUT probe sees transmitted wave only.
- Both IN-side and OUT-side `ka` should be reported separately where relevant.
- For cross-run comparison, use the no-panel IN-side `ka` as the reference axis (closest to "what the wavemaker delivers").

### Water depth regime — important wave physics context

Tank depth is ~580 mm (from filenames: `depth580`). Wave classification by depth-to-wavelength ratio:

| Regime | Condition | Effect |
|--------|-----------|--------|
| Deep water | d > λ/2 | Waves don't feel the bottom. Standard dispersion ω² = gk applies. |
| Intermediate | λ/20 < d < λ/2 | Partial bottom interaction. Full dispersion ω² = gk·tanh(kd). |
| Shallow water | d < λ/20 | Waves press against the bottom. Speed limited by depth: c = √(gd), independent of frequency. "Speed limits apply." |

At 580 mm depth, the regime depends on frequency. Higher frequencies (shorter λ) are deeper-water; lower frequencies (longer λ) may enter intermediate water. **This must be checked per frequency** — it affects dispersion, wave speed, and potentially how the panel interacts with the wave. The correct dispersion relation is always ω² = gk·tanh(kd); the deep-water simplification ω² = gk is only valid when kd >> 1.

**Observed (2026-03-12)**: visible wave-induced water movement at the tank bottom at low frequencies. This is the direct physical signature of intermediate/shallow water — orbital motion is no longer confined to the surface but reaches the bottom. Confirms that at least the lowest frequencies (0.65–0.7 Hz, longest wavelengths) are not in the deep-water regime at 580 mm depth.

**TODO**: compute kd for each frequency at d=580 mm and classify each run. Flag any runs in the shallow/intermediate transition zone — the physics there differs from the deep-water assumption and should be noted in the thesis. The bottom-motion observation is supporting evidence for the methodology chapter.

---

## 20. Rules for this assistant

> **For orientation**: §18 has the full three-phase call hierarchy and where each script fits. §5 has probe naming. §6 has known pitfalls. §0 is the current open investigation.

- Never reintroduce probe numbers (1–4) in user-facing code
- Always use `dist/lateral` position strings — never plain-number names
- When adding columns that are strings (especially with `/`), add them to `NON_FLOAT_COLUMNS`
- When adding new plots: accept `plotvariables` dict with `filters` + `plotting` keys; reuse `plot_frequency_spectrum` / `plot_reconstructed`
- When touching stillwater: honor anchor rules (prefer nowind+nowave; fall back to first 1s)
- When touching data loading: go through `load_analysis_data()` unless there is a clear reason not to
- Propose a branch name (`exp/<topic>`) for any non-trivial change
- If this file disagrees with the actual code, ask for clarification
