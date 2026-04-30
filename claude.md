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

## 0. How to use this document

**Current tasks and session state** → `MEMORY.md` is auto-loaded and has a START HERE
section at the top with the end goal, current state, and immediate next steps.
For the full task list see `memory/project_tasks.md`. For the latest session changelog
see `memory/session_2026-04-22.md`.

**Document map — find it here:**

| Topic | Section |
|-------|---------|
| Entry points, how to run | §3 |
| Data loading, what each variable contains | §4 |
| Probe naming, column name patterns | §5 — read this before touching any column names |
| Known pitfalls and silent failures | §6 — read this before adding new columns |
| Wave range detection, `_SNARVEI_CALIB` | §7 |
| Probe configurations over time | §8 |
| Core modules and their roles | §10 |
| Damping / OUT/IN analysis | §11 |
| Physical assumptions (noise floor, wave physics) | §16 |
| Rigorous analysis workflow | §17 |
| Three-phase architecture, explore → publication chain | §18 |
| Thesis structure, `ka`, key result variables | §19 |
| Rules for this assistant | §20 |

**Three rules that override everything else:**

1. **Probe names are always `"distance_mm/lateral_mm"` strings** — never plain integers, never probe numbers 1–4. `"9373/170"` is correct. `9373` is wrong.
2. **`OUT/IN` always uses FFT amplitude** — `"Probe {pos} Amplitude (FFT)"`, never `"Probe {pos} Amplitude"` (time-domain includes wind waves and is meaningless for damping under full wind).
3. **Any new string-typed column must go in `NON_FLOAT_COLUMNS`** in `improved_data_loader.py` — `apply_dtypes` calls `pd.to_numeric(errors="coerce")` on everything else, silently turning strings into NaN.

**`ignore_this_archive/` is dead code.** Everything under that top-level folder
is superseded / one-off / legacy. Do NOT read it to understand current state,
do NOT import from it, do NOT cite its findings as authoritative. When in
doubt about whether something is still live, check whether it's under
`ignore_this_archive/` — if yes, ignore it. See `ignore_this_archive/ARCHIVE_NOTES.md`
for the dumping rule, the per-file explanation of why each item was archived,
and the short list of known cosmetic dangling references.

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
| `wavetables/dtale_meta.py` | Open `combined_meta` in dtale browser, nothing else | `python wavetables/dtale_meta.py` or shell alias `wavetable` |

`main_explore_browser.py` forces `matplotlib.use("Qt5Agg")` — run from terminal, not REPL.

Shell alias `wavetable` is saved in `~/.zshrc` → `cd ~/Kodevik/wave_project && conda activate draumkvedet && python wavetables/dtale_meta.py`. Type `wavetable` from any terminal to open the table instantly.

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
- FFT spectrum: `"FFT 9373/250"`, `"FFT 9373/250 complex"`
- PSD spectrum: `"Pxx 9373/250"`

**Amplitude columns — one per method**, all in mm, all at the same probe position:

| column | what it measures | suffix convention |
|---|---|---|
| `"Probe 9373/250 Amplitude"` | (P99.5 − P0.5)/2 percentile over the whole window | **no suffix (legacy)** — TODO rename to `Amplitude (percentile)` for consistency; see `session_2026-04-22.md` follow-ups |
| `"Probe 9373/250 Amplitude (FFT)"` | nearest-bin FFT magnitude at f_paddle | `(method)` |
| `"Probe 9373/250 Amplitude (PSD)"` | integrated PSD variance over ±0.1 Hz | `(method)` |
| `"Probe 9373/250 Amplitude (LS)"` | fundamental of LS sinusoid fit at f_paddle | `(method)` |
| `"Probe 9373/250 Amplitude Stokes2 (LS)"` | 2nd-harmonic from same LS fit | `Stokes2 (method)` |
| `"Probe 9373/250 Amplitude (cycles) {stat}"` | per-cycle (max − min)/2 between zero-upcrossings | `(cycles) mean/std/n/list` |
| `"Probe 9373/250 Amplitude (phase) {stat}"` | per-cycle phase-locked sample at T/4 and 3T/4 | `(phase) mean/std/n/list` |

`{stat}` in `cycles` and `phase` is one of `mean`, `std`, `n`, `list` (the `list` column is a Python list, protected by `NON_FLOAT_COLUMNS`).

**Canonical suffix pattern going forward**: `(method_tag) [stat]` with method_tag in parentheses. Legacy `Amplitude` with no suffix remains in downstream plotter code (`plot_all_probes`, `damping_grouper`) until the rename is carried out.

**Do not** reintroduce probe numbers (1–4) in user-facing code.

### `in_position` / `out_position` in combined_meta

- Set by `processor2nd.py` from `ProbeConfiguration.in_probe` / `out_probe` via `probe_col_name()`
- Stored as position strings: `"9373/250"`, `"12400/170"`, etc.
- Refer to the **reference probe** — still a single-probe identifier
- The canonical IN/OUT values are in the generic `IN Amplitude (FFT)` / `OUT Amplitude (FFT)` columns (see below)

### Canonical IN/OUT = mean of all probes sharing the reference distance (CRITICAL, 2026-04-18)

The pipeline's `OUT/IN (FFT)` is computed as `A_out_canonical / A_in_canonical`, where each canonical amplitude is the **mean of all probes at the same longitudinal distance as the reference probe**:

- `march2026_better_rearranging`: IN = mean(9373/170, 9373/340); OUT = 12400/250 alone
- `march2026_rearranging`:        IN = mean(9373/170, 9373/340); OUT = 11800/250 alone
- `nov_normalt_oppsett`:          IN = 9373/250 alone; OUT = mean(12400/170, 12400/340)
- `initial_setup`:                IN = 9373/250 alone; OUT = mean(12400/170, 12400/340)

The single-probe ratio (old behaviour) is gone — it's replaced everywhere in meta.json by the mean-based ratio. Per-probe `Probe {pos} Amplitude (FFT)` columns still exist for oddity inspection.

**Columns written to meta.json by `processor2nd.py::_update_more_metrics`**:

- `IN Amplitude (FFT)` / `OUT Amplitude (FFT)` — canonical means
- Same pattern for `IN ka (FFT)`, `IN Wavenumber (FFT)`, `IN Wavelength (FFT)`, `IN WavePeriod (FFT)`, `IN Celerity (FFT)`, `IN Significant Wave Height Hm0`, `IN Significant Wave Height Hs`, `IN Froude (FFT)`, `IN Wind/Celerity (FFT)`, `IN f/f_PM (FFT)`, `IN Ursell (FFT)` — also means
- `IN wave_stability`, `IN period_amplitude_cv` — **reference-probe only** (per-probe quality metrics; averaging makes less sense)
- `in_probes_used` / `out_probes_used` — e.g. `"9373/170+9373/340"` — tells the reader which probes contributed per row
- `ain_disagree_frac` / `aout_disagree_frac` — `(max − min) / mean` over the contributing probes; 0 when the side has a single probe

**`damping_grouper`** (`filters.py`) recomputes OUT/IN from the generic `IN Amplitude (FFT)` / `OUT Amplitude (FFT)` columns, **not** from `Probe {in_position} Amplitude (FFT)` anymore.

**Don't silently merge**: `build_fig_meta(plotvariables, data_df=...)` automatically adds `in_probes_used`, `out_probes_used`, `probe_configs`, `non_final_config_n` to the stub's immutable comment block whenever the dataframe is passed in. That way every figure documents which probes contributed to its data.

**Archived**: the earlier post-load hook `wavescripts/mean_in_probe.py` lives under `analysis_scratch/archive/2026-04-18_mean_in_probe_hook/`. Superseded by the pipeline-level computation.

---

## 6. Known pitfalls / gotchas

### `apply_dtypes` destroys position strings with `/`

`apply_dtypes` in `improved_data_loader.py` calls `pd.to_numeric(..., errors="coerce")` on all columns not in `NON_FLOAT_COLUMNS`. Position strings containing `/` (e.g. `"12400/170"`) become **NaN**. Plain-number strings (e.g. `"9373"`) become floats (`9373.0`).

**Fix already applied**: `in_position` and `out_position` are now in `NON_FLOAT_COLUMNS`.

**Rule**: Any new string-typed column whose value may contain `/`, letters, or other non-numeric characters **must** be added to `NON_FLOAT_COLUMNS`. Forgetting this causes silent NaN corruption that is very hard to debug.

### `np.percentile` propagates NaN in matrix amplitude computation

`_compute_matrix_amplitudes` in `signal_processing.py` builds a matrix of probe samples and calls `np.nanpercentile`. If `np.percentile` (without `nan`) is used instead, **any probe with even 1 NaN sample in its range gets NaN amplitude** — including all nowave runs (which use the full signal range). Fixed by changing to `np.nanpercentile`.

### Stale `OUT/IN (FFT)` in meta.json (historical; resolved for current pipeline)

Historical note: pre-2026-04-18 `meta.json` files contained `OUT/IN (FFT)` computed with an old wide FFT window (0.5 Hz, argmax) that picked up wind-wave peaks instead of paddle. After the canonicalization session (2026-04-18) the canonical `IN Amplitude (FFT)` / `OUT Amplitude (FFT)` columns are computed by `processor2nd.py::_update_more_metrics` using the narrow (0.1 Hz, nearest-bin) method — safe to trust on any cache rebuilt after that date.

`damping_grouper` (`filters.py`) recomputes OUT/IN on-the-fly from the canonical `IN Amplitude (FFT)` / `OUT Amplitude (FFT)` columns, not from per-probe `Probe {in_position} Amplitude (FFT)` anymore. It falls back to the cached `OUT/IN (FFT)` only if the canonical columns are missing (prints a diagnostic).

### Six amplitude methods — not interchangeable

| Column | Source | What it sees |
|--------|--------|--------------|
| `"Probe {pos} Amplitude"` | (P99.5−P0.5)/2 of time-domain signal, whole window | Paddle + wind + Stokes — **legacy**; rename to `(percentile)` deferred; see §5 table |
| `"Probe {pos} Amplitude (FFT)"` | nearest-bin magnitude at f_paddle | Paddle tone only (FFT filters everything else) |
| `"Probe {pos} Amplitude (PSD)"` | integrated PSD variance over ±0.1 Hz | Paddle tone + broadband tail in a 0.2 Hz band |
| `"Probe {pos} Amplitude (LS)"` | LS sinusoid fit at exactly f_paddle | Paddle tone only — bin-grid-independent |
| `"Probe {pos} Amplitude (cycles) mean"` | per-cycle `(max − min)/2` from zero-upcrossings | Paddle + wind-on-top, per cycle |
| `"Probe {pos} Amplitude (phase) mean"` | per-cycle sample reading at `u+T/4`, `u+3T/4` | Paddle tone's quarter-period amplitude |

**OUT/IN uses `(FFT)` or `(LS)` — never the time-domain methods.** See §5 for the full suffix convention; see CH04 §4h comparison figure for cross-method agreement (< 0.4 % on 128 nowind measurements).

### FFT amplitude window

`compute_amplitudes_from_fft` uses `window=0.1` Hz and `argmin(abs(masked_freqs - target_freq))` (nearest bin). Old code used `window=0.5` Hz + `argmax`, which picked up wind-wave peaks for low-amplitude runs.

### Snap fires on raw ULS, FFT runs on η — sign-flipped (KNOWN, 2026-04-29)

**Observation:** the reconstructed paddle-frequency signal in `ch05_reconstructed` starts at sample 0 going **negative** (slope of first 5 samples = `[-1, -1, -1, -1]`; FFT phase at the paddle bin = +91° instead of the −90° expected for a sine starting at zero with positive slope).

**Mechanism (directly traceable):**
- [`wave_detection.py:62`](wavescripts/wave_detection.py:62) — upcrossing detection runs on `signal_smooth = rolling_mean(df[data_col])`, where `data_col` is the **raw ULS probe** (distance from sensor down to the water surface). Rising raw signal = water level falling.
- [`processor.py:759`](wavescripts/processor.py:759) — `eta = -(raw - stillwater)` flips the sign so positive η means water up.
- [`signal_processing.py:374`](wavescripts/signal_processing.py:374) — the FFT (and LS / cycles / phase metrics) consume `eta_{pos}_interp`, the sign-flipped signal.

A "raw upcrossing" is therefore an **η downcrossing** at the same sample index. Window endpoints are correct integer-cycle markers, but they're zero crossings of the *wrong* sign — the reconstructed fundamental's phase is offset by π.

**Effect on amplitude metrics:** none. FFT magnitude, LS amplitude, per-cycle (max − min)/2 and percentile amplitudes all ignore the absolute phase. Sinc leakage stays zero (integer cycles preserved). The bug only shows up visually in phase-sensitive plots like reconstructed waveforms.

**Effect on per-cycle metrics:** `cycles` and `phase` amplitudes that rely on between-upcrossings spans are also computed against raw-signal upcrossings. Because the spans are full periods either way, this does not bias the magnitude — but the phase-locked sample indices in `(phase) mean` are at η-trough quarters rather than η-crest quarters. Mean amplitude is unaffected (symmetric extraction); only the sign of intermediate values would flip. Not investigated further.

**Status:** noted, not fixed (2026-04-29). Fixing requires switching the upcrossing detector to `eta_{pos}` (or inverting the threshold), then `python main.py --force-recompute` on all datasets — every cached `Computed Probe {pos} start/end` shifts by ~half a period. Defer to a dedicated pipeline session.

### `_SNARVEI` probe name matching — archived

The old `_SNARVEI_CALIB` + `_PROBE_GROUP` eyeballed calibration was replaced by the deterministic H&G window + ±T snap in 2026-04-21 (see §7). Archived data lives in `constants.py` as `SNARVEI_ARCHIVE_START` / `_SNARVEI_ARCHIVE_END` and is still referenced by `RampDetectionBrowser` for visual calibration, but `find_wave_range` no longer uses it. Adding a new probe position no longer requires updating `_PROBE_GROUP` — `find_wave_range` derives the distance from the probe column name directly.

### Lessons from 2026-04-18 (big canonicalization session)

General principles learned the hard way this day — worth internalising before you refactor the data model:

**Post-load hooks are fragile.** The morning's `mean_in_probe.py` was a post-load transformation applied only inside `main_save_figures.py`. It worked but left `meta.json` and in-memory `meta_results` with different definitions of `OUT/IN (FFT)` — and any script that loaded meta *without* calling the hook silently used the single-probe ratio. **Rule**: if a change affects the *semantic meaning* of a metadata column, put it in the pipeline (`processor2nd.py`), not in a post-load hook. One source of truth, readable from meta.json.

**Silent merging is the enemy.** When a metric is computed by combining multiple probes / runs / conditions, the merge *must* be visible. Mechanism we settled on: `build_fig_meta(data_df=…)` auto-writes `in_probes_used`, `out_probes_used`, `probe_configs`, `non_final_config_n` into the TEXFIGU stub's IMMUTABLE comment block. Every plotter in `plotter.py` passes its filtered frame (`meta_df`, `stats_df`, or `band_amplitudes`) through this channel so the stub *always* documents which probes contributed. **Rule**: when you write a new plotter, pass the filtered data via `data_df=` to `build_fig_meta`.

**Averaging can still hide bias.** If two parallel probes are *systematically* different (e.g. wall-side vs far-side under wind), the mean is still biased, just *less* than a single probe. The `probe_bias_diagnostic.py` found significant directional bias at fullwind ≥ 1.5 Hz (wall-side reads +7–18 % higher, wind-contamination-driven). Nowind is clean. The mean is still defensible but is a conservative approximation; T_cross is the honest fullwind metric at high freq. **Rule**: after any multi-probe averaging, run a signed-difference + t-test check to confirm the residual bias is acceptable.

**Recomputes are not free but are often right.** A `python main.py --force-recompute` takes ~20 min for 25 datasets and updates every `meta.json`. When a canonical column changes meaning, *do the recompute* so the on-disk state matches the new science. Keeping the old hook around for "convenience" breeds divergence.

**Archive, don't delete.** Retired scripts go to `analysis_scratch/archive/<yyyy-mm-dd>_<name>/` with a README explaining why, what replaced them, and when the archived pattern could still be useful. Git history is authoritative but archive folders are faster to scan when looking for precedent.

---

## 7. Wave range detection (H&G window + ±T upcrossing snap)

Two-step procedure in `wavescripts/wave_detection.py::find_wave_range`:

### Step 1 — probe-shifted Huseby & Grue window (deterministic)

```python
_start_T, _end_T = hg_window_for_probe(r_probe_m, f_paddle)
good_start_idx   = round(_start_T * samples_per_period)
good_end_idx     = round(_end_T   * samples_per_period)
```

Window length is ALWAYS `10 × samples_per_period` samples (10 wave periods). The H&G reference `[50·T, 60·T]` is anchored at **r = 12.400 m** (HG.REF_R_M; corresponds to our OUT probe / the rail position, which is ~10 mm closer to paddle than H&G 2000's actual 12.41 m — the 10 mm gives a global ~0.026 T offset that doesn't matter for analysis; see `analysis_scratch/hg_snap_shift_diagnostic.md`).

For probes closer to the paddle, the window is shifted **earlier** by `ΔT = (REF_R_M − r_probe) / c_group(f, depth) · f` periods. `c_group` uses the full dispersion `ω² = gk·tanh(kh)` with h = 0.58 m; at thesis frequencies (1.3–1.7 Hz) this matches the deep-water shortcut `g/(4πf)` to < 0.1 % — see `wavescripts/constants.py::c_group`.

Result: at the OUT probe, the window spans samples `[50T, 60T]`. At the IN probe (r = 9.373 m, ΔT ≈ 7.6 periods at 1.4 Hz), it spans `[42.4T, 52.4T]` from wavemaker onset.

### Step 2 — ±T upcrossing snap (pipeline default since commit `71e67c5`, 2026-04-22)

After Step 1, the theoretical start is snapped to the **nearest zero-upcrossing of the raw ULS signal** within ±1 full wave period. Both window endpoints shift by the same amount (preserving the 10-period length). The snapped window is guaranteed to contain an integer number of cycles of the actual measured wave train — maximum FFT alignment.

Result: `Computed Probe {pos} start/end` = snap-adjusted window (used by FFT / LS / cycles / phase). Three diagnostic columns per probe record the snap:
- `Probe {pos} hg_expected_start` — pre-snap theoretical H&G start
- `Probe {pos} hg_expected_end`   — pre-snap theoretical H&G end
- `Probe {pos} hg_snap_shift`     — signed sample difference (snap − expected)

Typical snap shifts observed on canon data: IN probes near ±0 (c_g formula well-matched), OUT probe consistently ~−0.2 T (a separate physics effect still under investigation — candidates H5 near-panel reflection / H6 probe-specific lag). Full analysis in `analysis_scratch/hg_snap_shift_diagnostic.md`.

### Archived: `_SNARVEI_CALIB` (pre-2026-04-21)

The old eyeballed start-sample calibration lives in `constants.py` as `SNARVEI_ARCHIVE_START` / `_SNARVEI_ARCHIVE_END`, retained for `RampDetectionBrowser` calibration reference but NOT used by the current pipeline. The SNARVEI pipeline also snapped to zero-upcrossings (start + end), so integer-cycle windows were coherent in that era too — the 40 % sinc worst case from `memory/methodology_fft_peak_bin_bias.md` was never realised in either pre- or post-H&G pipeline. See `memory/methodology_hg_window_kills_peak_bias.md`.

### TODO: investigate wavemaker ramp-up shape

The wavemaker controller uses frequency-dependent acceleration profiles — higher frequencies have a different (longer?) soft-start program. The region **before** `good_start_idx` is not simply "stillwater + linear ramp" but contains a wavemaker-programmed pre-ramp that varies with frequency. Worth systematic inspection in `RampDetectionBrowser` if anyone needs to use the ramp region for stillwater baseline or probe characterisation.

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
- **`experimental-fromZeroToMaxWin` / `fromZeroToMaxWind` runs** — wind ramp-up runs (no paddle). Wind increases from zero to maximum while all probes record. Used to characterise wind setup (water level tilt), wind-wave growth, and time constants. Dates: 20260314, 20260326, 20260327. NOT standard wave or nowave runs — treat separately.

---

## 10. Core modules (`wavescripts/`)

- **`improved_data_loader.py`**: `ProbeConfiguration`, `PROBE_CONFIGS`, `load_analysis_data`, `load_processed_dfs`, `save_spectra_dicts`, `load_spectra_dicts`, `apply_dtypes`, `NON_FLOAT_COLUMNS`
- **`processor.py`**: `process_selected_data` — full pipeline called by `main.py`
- **`processor2nd.py`**: post-processing after main pipeline — sets `in_position`, `out_position`, `OUT/IN (FFT)`, band amplitudes
- **`signal_processing.py`**: `compute_fft_with_amplitudes`, `compute_psd_with_amplitudes`, `compute_amplitudes_from_fft`, `compute_amplitudes_from_lsfit`, `compute_lsfit_with_amplitudes` (LS sinusoid fit + Stokes-2f, added 2026-04-22)
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
- Recomputes `OUT/IN` per row from the canonical `IN Amplitude (FFT)` / `OUT Amplitude (FFT)` columns (which are themselves means of contributing probes at the same longitudinal distance; see §5 Canonical IN/OUT section)
- Falls back to the cached `OUT/IN (FFT)` value only if the canonical columns are missing (prints a diagnostic)

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

- Don't commit to `main` directly, without asking
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
- Stillwater noise floor is **probe-dependent** — measured as `"Probe {pos} Amplitude"` = (P99.5−P0.5)/2 from no-wind, no-wave runs:

Reference data: the 5-row source table was a dtale dump (now at
`ignore_this_archive/dtale-probe_uncertainty_tables.csv`); the table below
is the authoritative copy.

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
  - `"Probe {pos} Amplitude"` = (P99.5−P0.5)/2 of time-domain signal — includes wind waves
  - `"Probe {pos} Amplitude (FFT)"` = FFT peak within 0.1 Hz of target — paddle-wave only
- The **OUT/IN ratio** must always be computed from `"Probe {pos} Amplitude (FFT)"` (paddle frequency only). Time-domain amplitude includes wind-wave energy which inflates the IN probe under fullwind conditions, making OUT/IN meaningless for damping. Wind waves are a real physical phenomenon to characterize separately, not noise to average into the damping ratio.

### FFT-based OUT/IN under fullwind — two competing biases (CRITICAL)

When wind is on during a paddle-wave run, two competing biases act on the FFT amplitude at the paddle frequency. Both affect the **IN probe** (9373/170, fully exposed to wind). The **OUT probe** (12400/250) is sheltered by the panel and largely unaffected by both.

**(1) Spectral contamination — biases OUT/IN DOWNWARD**
Wind has a broadband PSD with a low-frequency tail. Even though most wind energy is at 3–5 Hz, there is non-zero wind energy within the 0.1 Hz FFT window at the paddle frequency. This adds spurious amplitude to A_IN_FFT:
- A_IN_FFT is inflated → OUT/IN appears lower than the true transmission
- Effect is larger at lower paddle frequencies (where the wind PSD tail is higher) and at low wave amplitudes (0.1 V), where wind energy can dominate A_IN
- The asymmetry is key: IN probe is contaminated, OUT probe is not → the bias is always downward

**(2) Phase jitter (coherence loss) — biases OUT/IN UPWARD**
Wind-induced turbulence and wind waves cause small cycle-to-cycle phase variations in the paddle wave. This spreads FFT energy away from the exact paddle frequency, reducing the FFT peak:
- A_IN_FFT is deflated → OUT/IN appears higher than true transmission
- Quantified by `wave_stability` column in `combined_meta` (values < 1 indicate jitter)
- More pronounced at higher frequencies and for longer runs

**Net effect:** At 0.1 V fullwind, bias (1) likely dominates (IN reads high → OUT/IN deflated). At 0.2–0.3 V, bias (2) may be comparable. The observed wind *increase* in OUT/IN at 1.5–1.6 Hz survives both biases — bias (1) would suppress it, yet the increase is still clearly seen, meaning the true effect is at least as large as measured and probably larger.

**Implication:** Measured OUT/IN under fullwind is a conservative (lower-bound) estimate of true transmission when bias (1) dominates. Never report fullwind OUT/IN without acknowledging this.

**Data reliability limit:** `WaveAmplitudeInput > 1.6 Hz` at 0.2 V or 0.3 V is unreliable — frequent dropouts at high amplitude + high frequency. Exclude from main results. See `memory/feedback_freq_amp_limits.md`.

### Wind setup — tank water level tilt under fullwind

Wind pushes water leeward (toward the panel / OUT probe side). This creates a mean water level slope along the tank:
- OUT probe (12400/250, leeward, sheltered from wind noise) shows a **noticeable mean level rise** under fullwind
- IN probe (9373/170, upwind) shows a corresponding mean level drop
- The effect is visible and measurable in the `eta_` time series as a nonzero mean during wind-only runs

**Effect on amplitude metrics:**
- FFT amplitude at paddle frequency: **unaffected** — DC level shift is at f=0, not at the paddle frequency
- Time-domain percentile amplitude (P99.5−P0.5)/2: **unaffected** — symmetric percentiles cancel any mean offset
- Wave speed / dispersion: minor effect via slightly changed local depth at each probe

**Dataset for quantifying this:** `experimental-fromZeroToMaxWin` / `fromZeroToMaxWind` runs (multiple dates: 20260314, 20260326, 20260327). These ramp wind from zero to maximum while recording all probes — the mean level drift at OUT probe (12400/250) is directly visible as the wind ramps up. Use these to characterise the setup magnitude and its time constant.

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

### `main_save_figures.py` — three-tier data-load gates (2026-04-22)

For REPL iteration, `main_save_figures.py` splits data loads into three progressively heavier tiers. Cells are tagged accordingly — stop executing before the next gate if you don't need the heavier data:

| Tier | Gate | Cost | What's loaded | Tag |
|---|---|---|---|---|
| 1 — Light | top of file | ~2 s | `combined_meta` + FFT/PSD dicts (all 25 folders) | `[META]` / `[DELEG]` / `[CSV]` |
| 2 — Medium | MEDIUM LOAD GATE | ~45 s | + `processed_dfs` for the 2 canon March-2026 lowrange folders (~180 runs, ~12 MB) | `[DFS-canon]` |
| 3 — Heavy | HEAVY LOAD GATE | ~+2 min | + remaining 23 folders' `processed_dfs` (~800 runs total, ~75 MB) | `[DFS-all]` |

Gates track loaded folders via `_loaded_dirs` — re-running the medium gate is idempotent; the heavy gate loads only the delta on top of whatever's already in `processed_dfs`. Currently §5 / §6 are the only [DFS-canon] consumers; no [DFS-all] cells exist (D1 placeholder would be the first).

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

### Observation-vs-inference discipline (durable rule)

**Documentation in this project reports observations. Explanations of WHY are kept separate and clearly flagged as hypotheses.**

- State numbers, patterns, correlations as observed.
- If a mechanism is offered, mark it `*Candidate explanation (hypothesis)*:` and keep it on a separate line / paragraph from the observation.
- Do NOT use causal phrasing ("this causes", "this is because", "this was chosen to") unless the causal link has been directly tested in the current data.
- Do NOT attribute design intent to past choices unless there is documented intent. N=10 periods is what H&G (2000) used; *why* is a candidate explanation, not a fact.
- Titles and descriptions that embed a causal claim are bugs. Prefer "X is observed to have Y" over "X does Y because Z".
- When updating a memo after a correction, add an "Observations" section and leave old inferential text as "retained for historical context". Don't silently rewrite past claims into correct ones — the revision itself is data.

Past mistake this rule exists to prevent (2026-04-22): a session memo claimed "H&G window kills the peak-bin bias by construction". A reviewer pointed out that the pre-H&G SNARVEI pipeline also produced integer-cycle windows (via zero-upcrossing snap), so both eras have coherent sampling. The bias was theoretical in both cases, and the memo's framing overstated what the data showed. Lesson: if the observation is "4 methods agree to 0.04 %", write that — not "the H&G window solves the problem".

