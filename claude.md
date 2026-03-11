# Wave Project — Context for Code Assistants

## 0. Current investigation (pick up here next session)

**Problem**: `explore_damping_vs_freq` shows ~3x wave growth for `nopanel, fullwind, 0.65 Hz, 0.1 V`.
Physics says no-panel + full-wind should show growth, but 3x seems too high. Investigating whether it's real or a measurement artifact.

**What we know so far** (from diagnostic output):

The run `nopanel-fullwind-amp0100-freq0650-...-20251112`:
- `in_position = "9373/250"` → amplitude **13.24 mm**
- `out_position = "12545/170"` → amplitude gives the 3x ratio visually in plot
- But `"12545/340"` (same longitudinal distance, other lateral) → **21.33 mm** (only 1.6x)

**The core anomaly**: two probes at the same distance from paddle (12545 mm) disagree by a factor of ~2. Either:
- The `out_position` assignment (`12545/170`) is wrong — maybe `12545/340` or an average should be used
- There is real lateral non-uniformity from wind

**Nowave-fullwind baseline**: the `nopanel-fullwind-nowave` Nov 12 run previously had all-NaN plain amplitudes. **Fixed** — was caused by `np.percentile` propagating NaN in the matrix path. Now uses `np.nanpercentile`. Re-run `main.py` to confirm.

**Physics note**: wind waves exist only above ~2 Hz — no wind-wave energy at 0.65 Hz in the PSD sense. BUT wind waves (3–5 Hz, broad spectrum, erratic) ride on top of the paddle wave in the time domain. The `"Probe {pos} Amplitude"` values are **time-domain percentile amplitudes** — they include ALL frequency content, not just 0.65 Hz.

**Better metric for this investigation**: `"Probe {pos} Amplitude (FFT)"` at exactly 0.65 Hz isolates just the paddle wave and is immune to wind-wave contamination. Compare the FFT amplitudes (not time-domain) at `12545/170` vs `12545/340` across runs to test whether the asymmetry survives frequency-isolation.

**The real question**: two probes at same distance from paddle (12545 mm) disagree by ~2x. Possible causes:
- Wind-wave noise inflating time-domain amplitude more on the 170 side (near wall)
- Wall reflection at 170 mm side creating constructive interference at 0.65 Hz (would survive FFT isolation)
- Wind skewing the wave crest laterally (would only appear in wind runs)
- `out_probe=3` (`12545/170`) is simply the wrong choice — `12545/340` or average may be better

**Next steps**:
1. Re-run the ratio check using `"Probe 12545/170 Amplitude (FFT)"` vs `"Probe 12545/340 Amplitude (FFT)"` — if asymmetry disappears → wind-wave noise in time-domain is the cause
2. If FFT asymmetry persists → geometry/reflection or wind skew — check whether it appears in no-wind runs too
3. Decide whether `out_position` should be `12545/170`, `12545/340`, or averaged — update `nov_normalt_oppsett` config if needed
4. Learn from `ensure_stillwater` in `processor.py` and sample 1-2 seconds of wind data (before wave action) from wave-runs, and use whole dataset for no-wave runs.

**Diagnostic code** (run in `main_explore_inline.py`):
```python
# Lateral asymmetry check
_wave = combined_meta[combined_meta["WaveFrequencyInput [Hz]"].notna()].copy()
_wave["ratio_170_vs_340"] = _wave["Probe 12545/170 Amplitude"] / _wave["Probe 12545/340 Amplitude"]
print(_wave[["WaveFrequencyInput [Hz]", "WaveAmplitudeInput [Volt]", "WindCondition",
             "PanelCondition", "Probe 12545/170 Amplitude", "Probe 12545/340 Amplitude",
             "ratio_170_vs_340"]].sort_values("ratio_170_vs_340", ascending=False).to_string())
```

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
| `main_save_figures.py` | (WIP) batch LaTeX/PGF figure export | `python main_save_figures.py` |

`main_explore_browser.py` forces `matplotlib.use("Qt5Agg")` — run from terminal, not REPL.

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
| 12545 mm, center | `"12545/250"` |
| 12545 mm, near wall | `"12545/170"` |
| 12545 mm, far side | `"12545/340"` |
| 8804 mm, center | `"8804/250"` |

`probe_col_name()` always returns `f"{dist}/{lat}"` — no parallel-detection logic.

**Do not** use plain-number names like `"9373"`, `"12545"`, `"8804"` — these were the old convention, replaced in Mar 2026.

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
- Stored as position strings: `"9373/250"`, `"12545/170"`, etc.
- Used by `damping_grouper` to recompute OUT/IN ratio on-the-fly from plain amplitude columns

---

## 6. Known pitfalls / gotchas

### `apply_dtypes` destroys position strings with `/`

`apply_dtypes` in `improved_data_loader.py` calls `pd.to_numeric(..., errors="coerce")` on all columns not in `NON_FLOAT_COLUMNS`. Position strings containing `/` (e.g. `"12545/170"`) become **NaN**. Plain-number strings (e.g. `"9373"`) become floats (`9373.0`).

**Fix already applied**: `in_position` and `out_position` are now in `NON_FLOAT_COLUMNS`.

**Rule**: Any new string-typed column whose value may contain `/`, letters, or other non-numeric characters **must** be added to `NON_FLOAT_COLUMNS`. Forgetting this causes silent NaN corruption that is very hard to debug.

### `np.percentile` propagates NaN in matrix amplitude computation

`_compute_matrix_amplitudes` in `signal_processing.py` builds a matrix of probe samples and calls `np.nanpercentile`. If `np.percentile` (without `nan`) is used instead, **any probe with even 1 NaN sample in its range gets NaN amplitude** — including all nowave runs (which use the full signal range). Fixed by changing to `np.nanpercentile`.

### Stale `OUT/IN (FFT)` in meta.json

`meta.json` may contain `OUT/IN (FFT)` values computed with an old wide FFT window (`0.5 Hz`, `argmax`) that picks up wind-wave peaks instead of paddle-wave peaks. Do not trust cached `OUT/IN (FFT)`.

`damping_grouper` now recomputes OUT/IN on-the-fly from `"Probe {pos} Amplitude"` (plain time-domain, percentile-based) columns. It falls back to the cached value only if recomputation yields 0 valid rows (prints a diagnostic).

### Two amplitude types — not interchangeable

| Column | Source | Used by |
|--------|--------|---------|
| `"Probe {pos} Amplitude"` | Percentile of time-domain signal | `plot_all_probes`, `damping_grouper` |
| `"Probe {pos} Amplitude (FFT)"` | FFT peak near target frequency | Old OUT/IN cached values |

Always use `"Probe {pos} Amplitude"` (no suffix) for OUT/IN ratio computation.

### FFT amplitude window

`compute_amplitudes_from_fft` uses `window=0.1` Hz and `argmin(abs(masked_freqs - target_freq))` (nearest bin). Old code used `window=0.5` Hz + `argmax`, which picked up wind-wave peaks for low-amplitude runs.

### `_SNARVEI` probe name matching

`find_wave_range` in `wave_detection.py` uses `_PROBE_GROUP` dict to map all lateral variants of a probe to a distance group (e.g. `"Probe 12545/170"` → `"12545"`). If a new probe position is added, it **must** be added to `_PROBE_GROUP` — otherwise range detection falls back to `2 * samples_per_period` (stillwater phase), giving near-zero amplitudes and OUT/IN ≈ 0.1.

---

## 7. Wave range detection (`_SNARVEI_CALIB`)

Defined in `wavescripts/wave_detection.py`. Multi-point linear interpolation of stable-wave start sample vs frequency, calibrated by eyeballing `RampDetectionBrowser`.

```python
_SNARVEI_CALIB = {
    "8804":  [(0.65, 5104), (1.30, 4700)],
    "9373":  [(0.65, 5154), (0.70, 3750), (1.30, 4800), (1.60, 5500)],
    "12545": [(0.65, 5654), (0.70, 4250), (1.30, 6500), (1.60, 7000)],
}
```

- Format: `(freq_hz, start_sample)` sorted by frequency; samples at 250 Hz (ms / 4)
- Interpolates linearly between points; extrapolates linearly beyond the range
- `_PROBE_GROUP` maps every probe column name variant to a distance group key
- To add a calibration point: eyeball start in `RampDetectionBrowser`, convert ms → samples (/4), add tuple
- `8804` group only has 2 points (0.65 and 1.30 Hz) — extrapolates for other frequencies

---

## 8. Probe configurations over time

Defined in `improved_data_loader.py` as `PROBE_CONFIGS`:

| Config name | Valid from | in_pos | out_pos | Notes |
|-------------|-----------|--------|---------|-------|
| `initial_setup` | Aug 2025 | `9373/250` | `12545/170` | Probe 1 far back at 18000 mm |
| `nov14_normalt_oppsett` | Nov 10 2025 | `9373/250` | `12545/170` | Probe 1 moved to 8804 mm |
| `march2026_rearranging` | Mar 4 2026 | `9373/170` | `12300/250` | Temporary, 2 days |
| `march2026_better_rearranging` | Mar 7 2026 | `9373/170` | `12545/250` | Current layout |

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
- **`plot_quicklook.py`**: `SignalBrowserFiltered`, `RampDetectionBrowser` (Qt)
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

## 16. Rules for this assistant

- Never reintroduce probe numbers (1–4) in user-facing code
- Always use `dist/lateral` position strings — never plain-number names
- When adding columns that are strings (especially with `/`), add them to `NON_FLOAT_COLUMNS`
- When adding new plots: accept `plotvariables` dict with `filters` + `plotting` keys; reuse `plot_frequency_spectrum` / `plot_reconstructed`
- When touching stillwater: honor anchor rules (prefer nowind+nowave; fall back to first 1s)
- When touching data loading: go through `load_analysis_data()` unless there is a clear reason not to
- Propose a branch name (`exp/<topic>`) for any non-trivial change
- If this file disagrees with the actual code, ask for clarification
