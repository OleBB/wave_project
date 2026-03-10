# Project Overview

- **Repo URL**: `https://github.com/OleBB/wave_project`
- **Project type**: Python data / signal / wave analysis (Jupyter + Spyder)
- **Primary goals for this assistant**:
  - Understand the structure and purpose of this repo
  - Help design, refactor, and test analysis code
  - Help keep experiments organized via git branches
  - Help write and improve pytest tests

If you (the chatbot) need context, ask before assuming missing details.

---

## Git Workflow (Solo)

**Core rules**

- Never commit directly to `main`.
- Work in experiment branches:  
  `git checkout -b exp/<descriptive-name>`
- Commit small, often, with clear messages.
- When an experiment works:
  - `git checkout main`
  - `git pull`
  - `git merge exp/<descriptive-name>`
  - `git push`
  - Optionally tag: `git tag v0.1-baseline`
  - Delete branch: `git branch -d exp/<descriptive-name>`

**Safety net**

Before risky changes:

```bash
git commit -am "safety: everything working before I break it"
```

This ensures I can always restore a working state by using `git log` and `git checkout <commit-sha>`.

**How you (the assistant) should use this**

- When suggesting changes, propose a branch name: `exp/<what-you-try>`.
- When generating commit messages, keep them short and descriptive, e.g.  
  `exp: try random forest, F1=0.68`.

---

## Testing (pytest)

The project uses **pytest** for automated checks.

**Common commands**

- Run one file: `pytest -q test_sandkasse.py`
- Run all tests in current folder: `pytest -q`
- Run a single test function:  
  `pytest -q test_sandkasse.py -k test_svaret`
- Verbose: `pytest -vv`
- Stop at first failure: `pytest -x`
- Show `print()` output: `pytest -s`
- Re-run only last failures: `pytest --last-failed`
- List fixtures: `pytest --fixtures`
- Help: `pytest -h`

**Why tests matter here**

- Prevent silent regressions in signal/data processing
- Encode assumptions (e.g. monotonic timestamps, sampling rate, peak distances)
- Make refactoring safe
- Provide fast, automated feedback instead of manual visual checks

**How you (the assistant) should use this**

- When changing logic, propose or update tests that:
  - Use small synthetic data
  - Check key metrics (e.g. peak counts, frequencies)
  - Verify error handling (e.g. raising `ValueError` on bad input)
- Prefer:
  - A few “smoke tests” for end-to-end pipeline
  - “Golden” tests with known outputs/metrics

---

## Environment

Target environment (conda):

```yaml
name: draumeriket
channels:
  - defaults
dependencies:
  - python=3.11
  - spyder=6.1.0
  - notebook
  - spyder-notebook
  - scipy
  - pytest
  - numpy
  - matplotlib
  - sympy
  - pandas
  - plotly
  - seaborn
  - pyarrow
  - tabulate
  - spyder-unittest
```

**How you (the assistant) should use this**

- Assume Python 3.11 and the above packages are available.
- Prefer numpy/pandas/scipy for core computations.
- For plotting, suggest matplotlib/plotly/seaborn.

---

## Notebooks and `.gitignore`

- Some notebooks are ignored by git.
- Only selected notebooks are shared/committed.

**How you (the assistant) should behave**

- When proposing new notebooks, assume they may need to be added explicitly (not all are tracked).
- For reusable code, prefer Python modules over big notebooks, and then import them into notebooks.

---

## How to Help Me Most Effectively

When responding, you (the chatbot) should:

1. **Ask for file content** before editing:
   - e.g., “Paste `wave_analysis.py`” or “Show the relevant notebook cell.”
2. **Propose branch + commit messages** when suggesting non-trivial changes.
3. **Suggest tests** alongside any important new logic or refactor.
4. **Keep examples small and runnable** with synthetic data.
5. **Explain assumptions explicitly** (sampling rate, units, signal length, etc.).

If something in this file conflicts with the actual codebase, ask for clarification instead of guessing.```markdown
## Project: wave_project – quick context for code assistants

### What this repo does

Wave tank experiment analysis pipeline:

- Load raw CSV runs from `wavedata/`
- Process & cache to `waveprocessed/PROCESSED-*`
- Compute FFT/PSD & wave physics
- Explore runs via inline notebooks or Qt browsers
- (Later) export publication-quality figures with LaTeX/PGF

Main constraints:

- Must handle multiple probe layouts over time
- Probes are identified by **physical position**, not 1–4
- Need fast reload from cached processed data
- Need robust handling of “nowave / nowind / wind-only” runs

---

## High-level workflow

### 1. Solo git workflow (from README)

Always work on branches:

```bash
git checkout -b exp/what-im-trying
# work, then:
git add .
git commit -m "exp: short message"
git checkout main
git pull
git merge exp/what-im-trying
git push
git branch -d exp/what-im-trying
```

Safety snapshot before risky edits:

```bash
git commit -am "safety: everything working before I break it"
```

### 2. Environments & requirements

Conda env (`draumeriket`):

- `python=3.11`
- `spyder`, `notebook`, `spyder-notebook`, `spyder-unittest`
- `numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn`, `plotly`
- `pytest`, `sympy`, `pyarrow`, `tabulate`

(Prefers conda over pip.)

### 3. Entry points / scripts

At repo root:

- `main.py`  
  Full batch processing:
  - Load raw CSVs from `wavedata/…`
  - Use `wavescripts/processor.py` & `wavescripts/signal_processing.py`
  - Compute stillwater, wave ranges, FFT/PSD, amplitudes
  - Save to `waveprocessed/PROCESSED-*` (parquet + meta JSON)
  - **No plotting, no Qt**

- `main_explore_inline.py`  
  Inline exploration (Zed REPL, `%matplotlib inline`):
  - Calls `load_analysis_data(*PROCESSED_DIRS)`
  - Works with:
    - `combined_meta` (all runs)
    - `processed_dfs` (zeroed+smoothed time series)
    - `combined_fft_dict`, `combined_psd_dict` (wave runs only)
  - Contains:
    - Amplitude / damping / FFT / PSD plots
    - Swell scatter, wavenumber checks
    - Example reconstructed-signal plots for a single run and all runs
    - Wind-only analysis (nowave + wind, plus stillwater baseline)

- `main_explore_browser.py`  
  Qt GUIs (run from terminal, not REPL):
  - `SignalBrowserFiltered` → pick runs; calls `plot_reconstructed`
  - `RampDetectionBrowser` → inspect ramp detection
  - Uses Qt backend (`Qt5Agg`), free-floating matplotlib windows

- `main_save_figures.py`  
  Stub for later:
  - Will call plotting functions with `save_plot/save_pgf`
  - Target: LaTeX-ready figures to `figures/` or similar

---

## Core modules (wavescripts/)

- `improved_data_loader.py`
  - `ProbeConfiguration`, `PROBE_CONFIGS`
  - `load_meta_json(processed_dir)`
    - Handles `meta.json` as list of records
    - Casts types via `apply_dtypes`
    - Converts `"file_date"` to datetime
  - `load_analysis_data(*processed_dirs, load_processed=False)`
    - Always loads `combined_meta` and FFT/PSD from parquet cache
    - `load_processed=False` (default): skips loading `processed_dfs` (saves ~75 MB + ~20 s)
    - `load_processed=True`: also loads full time-series `processed_dfs` (needed for wind-only analysis)
    - Both `main_explore_inline.py` and `main_explore_browser.py` pass `load_processed=True`
  - `save_spectra_dicts(fft_dict, psd_dict, cache_dir)`
    - Saves `fft_spectra.parquet` and `psd_spectra.parquet`
    - All float columns downcast to **float32** to halve file size
    - Complex columns (e.g. `"FFT pos complex"`) split into `col_real` + `col_imag` float32 pairs
  - `load_spectra_dicts(cache_dir)`
    - Loads parquet; upcasts float32 → float64
    - Recombines `_real`/`_imag` pairs back into complex128 columns

- `processor.py`
  - `process_selected_data(...)` (called by `main.py`)
  - `ensure_stillwater_columns(dfs, meta, cfg)`
    - Uses “no-wind/no-wave” anchor runs to compute stillwater:
      - Anchor condition:
        - `WindCondition == "no"` or missing  
        - **AND** nowave (`WaveFrequencyInput` NaN or `"nowave"` in path)
      - If no such anchors:
        - Falls back to `WindCondition == "no"` runs with waves,
        - Samples only the **first 1 second** (pre-wave stillwater)
    - Computes per-probe stillwater level and noise std
  - Calls:
    - `find_wave_range` (wave onset/offset)
    - `compute_fft_with_amplitudes`
    - `compute_psd_with_amplitudes`
    - `calculate_wavenumbers_vectorized`, `calculate_wavedimensions`, `calculate_windspeed`

- `signal_processing.py`
  - `compute_fft_with_amplitudes(processed_dfs, meta_wave, cfg, fs, debug=False)`
    - Builds `fft_dict` with columns:
      - `"FFT {pos}"`, `"FFT {pos} complex"` (index = Frequencies)
    - Stores per-run FFT amplitude/frequency/period back into `meta`
    - Skips runs where `WaveFrequencyInput [Hz]` is NaN or ≤ 0
  - `compute_psd_with_amplitudes(...)`
    - Builds `psd_dict` with `"Pxx {pos}"` columns
    - Computes band amplitudes and writes to `meta`:
      - `"Probe {pos} Amplitude (PSD)"`, `"Swell Amplitude (PSD)"`, `"Wind Amplitude (PSD)"`, etc.
  - Debug prints are minimal (per-run summaries, not per-sample spam)

- `plotter.py`
  - `plot_all_probes(meta_sel, plotvariables)`
  - `plot_damping_freq`, `plot_damping_scatter`
  - `plot_frequency_spectrum(fft_or_psd_dict, meta_df, plotvariables, data_type="fft")`
    - `fft_dict` or `psd_dict` style:
      - dict[path → DataFrame with `"FFT {pos}"` / `"Pxx {pos}"`]
    - `plotvariables["plotting"]["probes"]` is a list of **position strings**
      - e.g. `["9373/170", "12545"]`
    - `facet_by` can be `"probe" | "wind" | "panel"`
  - `plot_swell_scatter(...)`
  - `plot_reconstructed(fft_dict, filtrert_frequencies, freqplotvariables, data_type="fft")`
    - Expects **single run** if called with `{path: df}` and matching `meta`
    - Handles:
      - `plotting["probes"]` = list of position strings
      - `plotting["facet_by"] == "probe"` → multiple subplots
      - Safely returns early if no probes selected
    - Returns `(fig, axes)`; uses `plt.show(block=False)` when `show_plot=True`
  - `gather_ramp_data`, `plot_ramp_detection` for ramp browser

- `plot_quicklook.py`
  - `SignalBrowserFiltered(QMainWindow)`
    - Takes `fft_dict`, `meta_df`, `freqplotvariables`
    - Derives probe positions from columns starting with `"FFT "`
      - Builds checkboxes labeled by position (e.g. `"12545"`, `"9373/340"`)
    - On selection:
      - Copies `plotvars`, updates `"plotting"` section
      - Calls `plot_reconstructed({path: df}, single_meta, plotvars)`
      - Resizes figure to ~¾ screen
  - `RampDetectionBrowser(QMainWindow)`
    - Steps through ramp detection results (`gather_ramp_data` output)
    - Zoom control; shows full ramp detection plot per run
  - `explore_damping_vs_freq`, `explore_damping_vs_amp`
  - `save_interactive_plot(df, filename="damping_analysis.html")` (plotly)

- `filters.py`
  - Central place for `plotvariables["filters"]` handling:
    - `apply_experimental_filters`
    - `filter_for_frequencyspectrum`
    - `damping_grouper`, `damping_all_amplitude_grouper`
    - `filter_for_amplitude_plot`, etc.

- `constants.py`
  - `MEASUREMENT`:
    - `SAMPLING_RATE = 250.0`
    - `STILLWATER_SAMPLES = 250`
  - `GlobalColumns (GC)`:
    - e.g. `WIND_CONDITION`, `PANEL_CONDITION`, `WAVE_FREQUENCY_INPUT`, `PATH`
  - `ProbeColumns (PC)`:
    - Templates like `"Probe {i} Amplitude"`, `"Probe {i} Amplitude (FFT)"`, …
  - `ColumnGroups (CG)`:
    - Helpers to list groups of columns (amplitude, FFT, PSD, setup)

---

## Probe naming & column conventions

### Probe positions

Refactored from numeric `Probe 1–4` to physical positions:

- Logical probes:
  - 1 → `"9373/170"`
  - 2 → `"12545"`
  - 3 → `"9373/340"`
  - 4 → `"8804"`
- `ProbeConfiguration.probe_col_names()` returns `{1: "9373/170", 2: "12545", ...}`

### Raw & processed time series

For each run in `processed_dfs[path]`:

- Time: `"Date"`
- Raw probes: `"Probe 9373/170"`, `"Probe 12545"`, ...
- Processed elevation:
  - `"eta_9373/170"`, `"eta_12545"`, `"eta_9373/340"`, `"eta_8804"`
- Smoothed:
  - `"Probe 9373/170_ma"`, etc.
- Other:
  - `"Mach"` (wind tunnel info)

### Metadata columns (combined_meta)

Per run:

- Base:
  - `"path"`, `"experiment_folder"`, `"file_date"`
  - `"WindCondition"`, `"PanelCondition"`, `"Mooring"`, `"TunnelCondition"`
  - `"WaveAmplitudeInput [Volt]"`, `"WaveFrequencyInput [Hz]"`, `"WavePeriodInput"`
- Stillwater & geometry:
  - `"Probe i mm from paddle"`, `"Probe i mm from wall"`
  - `"Stillwater Probe {i}"` (original numbering)
  - `"Stillwater Probe 9373/170"`, etc. (position-based)
- FFT/PSD amplitudes:
  - `"Probe {pos} Amplitude"`
  - `"Probe {pos} Amplitude (FFT)"`, `"Probe {pos} Amplitude (PSD)"`
  - `"Probe {pos} Swell Amplitude (PSD)"`, `"Probe {pos} Wind Amplitude (PSD)"`, `"Total Amplitude (PSD)"`
- Wave physics:
  - `"Probe {pos} Wavenumber (FFT)"`, `"Wavelength (FFT)"`, `"kL (FFT)"`, `"ak (FFT)"`, `"tanh(kH) (FFT)"`, `"Celerity (FFT)"`
  - Global: `"Waveperiod"`, `"Wavenumber"`, `"Wavelength"`, `"kL"`, `"ak"`, `"kH"`, `"tanh(kH)"`, `"Celerity"`
- Ratios & damping:
  - `"P2/P1 (FFT)"`, `"P3/P2 (FFT)"`, `"P4/P3 (FFT)"`, `"OUT/IN (FFT)"`
- Swell scatter support:
  - `"Probe {pos} Swell Amplitude (PSD)"`, `"Probe {pos} Wind Amplitude (PSD)"`

Note: Nowave runs (`WaveFrequencyInput NaN`) have PSD amplitude columns as NaN but do appear in `combined_meta`.

---

## Run types (important for filtering)

From `combined_meta`:

- **Wave runs**:
  - `WaveFrequencyInput [Hz]` > 0
  - Appear in `fft_dict` / `psd_dict`
- **Nowave runs** (no paddle wave):
  - `WaveFrequencyInput [Hz]` NaN **OR** `"nowave"` in filename
  - Includes:
    - stillwater: `WindCondition == "no"`
    - wind-only: `WindCondition in {"full","lowest"}` + nowave
- **Old/odd filenames**:
  - E.g. `"fullpanel-nowind-NO-depth580-..."`:
    - No `-amp####-` and no `-freq####-` → treated as nowave
  - Rule in `_extract_wave_parameters`:
    - Both amp and freq must be present to set wave parameters

---

## How FFT/PSD caches are built

`main.py` computes FFT/PSD and calls `save_spectra_dicts(fft_dict, psd_dict, cache_dir)` to write:
- `waveprocessed/PROCESSED-*/fft_spectra.parquet`
- `waveprocessed/PROCESSED-*/psd_spectra.parquet`

These files store all FFT/PSD data including **complex columns** (split to float32 real+imag).

`load_analysis_data` then loads from cache (fast path):
1. Load `meta.json` → `combined_meta`
2. Try `load_spectra_dicts(processed_dir)` → `fft_dict`, `psd_dict` from parquet
3. If no cache: fall back to recomputing from `processed_dfs` (slow)
4. If `load_processed=True`: also load `processed_dfs.parquet` (~75 MB)

`waveprocessed/` is **gitignored** — all parquet/json cache files are local only, regenerated by `main.py`.

Nowave runs are **not** in `fft_dict` / `psd_dict` but remain in `combined_meta` and `processed_dfs`.

---

## Wind-only analysis (current approach)

In `main_explore_inline.py`:

- Filter:
  - `_meta_nowave = combined_meta[WaveFrequencyInput is NaN]`
  - `_meta_wind_only = _meta_nowave[WindCondition in {"full","lowest"}]`
  - `_meta_stillwater = _meta_nowave[WindCondition == "no"]`
- Build `wind_psd_dict` from `processed_dfs` using `scipy.signal.welch` on `eta_{pos}` columns:
  - Same format as `psd_dict`: `{path: DataFrame(index = Frequencies, cols = "Pxx {pos}")}`
- Plot:
  - `plot_frequency_spectrum(wind_psd_dict, _meta_nowave, plotvars, data_type="psd")`
  - `facet_by="probe"`; log scale; xlim (0, 5 Hz)
- Stats:
  - For each nowave run and each probe:
    - Mean(`eta_{pos}`) → wind setup
    - Std(`eta_{pos}`) → RMS surface fluctuations

---

## Pytest usage (from README, shortened)

Useful commands (run from repo root):

- Run all tests:
  ```bash
  pytest -q
  ```

- Single file:
  ```bash
  pytest -q tests/test_sandkasse.py
  ```

- Single test:
  ```bash
  pytest -q tests/test_sandkasse.py -k test_svaret
  ```

- Verbose / show prints / stop early:
  ```bash
  pytest -vv
  pytest -s
  pytest -x
  ```

Rationale: tests codify assumptions (sampling rate, monotonic timestamps, wave range, probe mapping), so refactors don’t silently break the physics.

---

## What a helper model should know / respect

- **Do not** reintroduce probe numbers (1–4) in user-facing code; use position strings consistently.
- When adding new plots:
  - Accept `plotvariables` with `filters` + `plotting`.
  - Use the same column naming patterns.
  - Prefer `plot_frequency_spectrum` / `plot_reconstructed` rather than duplicating logic.
- When touching stillwater logic:
  - Honor the new anchor rules (prefer nowind+nowave; fall back to first 1s of nowind+wave).
- When adding new analysis:
  - Prefer to extend `main_explore_inline.py` and optionally mirror into `main_save_figures.py`.
- When touching data loading:
  - Always go through `load_analysis_data()` unless there is a very good reason not to.
```



[Bruker]

please give me a .md file


[gpt-5.1]

```markdown
# claude.md — Wave Project overview for code assistants

## 1. User + environment context

- OS: macOS
- Editor: Zed
- Package manager: conda (not pip) <ref: index=2648723 firstWord=3 lastWord=8/>
- Main Jupyter/REPL env: `draumkvedet` (local), but this repo’s exported env for others is `draumeriket` <ref: index=2648739 firstWord=20 lastWord=28/><ref: index=2648696 firstWord=60 lastWord=80/>

Conda env (shared / exportable):

```yaml
name: draumeriket
channels:
  - defaults
dependencies:
  - python=3.11
  - spyder=6.1.0
  - notebook
  - spyder-notebook
  - scipy
  - pytest
  - numpy
  - matplotlib
  - sympy
  - pandas
  - plotly
  - seaborn
  - pyarrow
  - tabulate
  - spyder-unittest
```
<ref: index=2648696 firstWord=80 lastWord=112/>

Assume these libs are available when proposing code.

---

## 2. High-level purpose

This repo is a **wave-tank experiment analysis pipeline**:

- Raw CSV experiments in `wavedata/`
- `main.py` runs the full processing pipeline:
  - load CSVs
  - compute stillwater levels
  - detect usable wave ranges
  - compute FFT/PSD and wave-physics quantities
  - cache results in `waveprocessed/PROCESSED-*`
- Exploration scripts load **processed data**, not raw CSVs

Processed caches contain:

- `meta.json`: one row per run (all runs, including “nowave”) with typed columns <ref: index=2648707 firstWord=40 lastWord=51/>
- `processed_dfs.parquet`: all zeroed+smoothed time series, keyed by original CSV path <ref: index=2648711 firstWord=537 lastWord=553/>
- FFT/PSD dicts are recomputed from `processed_dfs` on load (fast, no CSV parsing) <ref: index=2648705 firstWord=981 lastWord=993/>

---

## 3. Entry points (root of repo)

From `project_structure.md` (assistant memory): <ref: index=2648739 firstWord=40 lastWord=80/>

| File                     | Role                                                       | How to run                            |
|--------------------------|------------------------------------------------------------|---------------------------------------|
| `main.py`                | Full pipeline: CSV → processed cache (meta + dfs)         | `python main.py`                      |
| `main_explore_inline.py` | Inline, cell-based exploration + plots (Zed / Jupyter)    | Open in Zed, run `# %%` cells         |
| `main_explore_browser.py`| Qt GUIs for exploring runs interactively                  | `python main_explore_browser.py`      |
| `main_save_figures.py`   | (WIP) batch export of LaTeX/PGF figures                   | `python main_save_figures.py`         |

Important notes:

- `main_explore_inline.py` is the **primary analysis playground** <ref: index=2648739 firstWord=80 lastWord=96/>
- `main_explore_browser.py` explicitly forces Qt backend:

  ```python
  import matplotlib
  matplotlib.use("Qt5Agg")
  ```
  <ref: index=2648671 firstWord=1 lastWord=8/>

---

## 4. Data structures you must respect

### 4.1. Metadata (`combined_meta`)

Loaded via `load_analysis_data(*PROCESSED_DIRS, load_processed=True)` in `wavescripts/improved_data_loader.py`:

```python
combined_meta, processed_dfs, fft_dict, psd_dict = load_analysis_data(
    *PROCESSED_DIRS, load_processed=True
)
```

- `combined_meta`: all runs (wave + nowave), typed with `apply_dtypes`
- `processed_dfs`: `{path: DataFrame}` of processed signals (empty dict if `load_processed=False`)
- `fft_dict`: `{path: DataFrame}` for **wave runs only** — includes complex columns
- `psd_dict`: `{path: DataFrame}` for **wave runs only**

Type enforcement example:

- `WaveFrequencyInput [Hz]` is coerced to `float64`, missing/invalid → `NaN` <ref: index=2648705 firstWord=1041 lastWord=1066/>

### 4.2. Processed time series (`processed_dfs`)

Each `processed_dfs[path]` DataFrame (per run) typically contains:

- Time: a timestamp column (often `"Date"` or similar)
- Raw probes: `"Probe {pos}"` columns; “pos” is **physical position**, not 1–4
- Processed elevation: `eta_{pos}` columns (same position convention) <ref: index=2648711 firstWord=457 lastWord=460/>
- Other channels (e.g. `"Mach"` for wind tunnel state)

### 4.3. FFT and PSD dicts

`psd_dict` example from a real run: <ref: index=2648735 firstWord=1 lastWord=15/>

```python
psd_dict[path].columns
# ['Pxx 9373/170', 'Pxx 12545', 'Pxx 9373/340', 'Pxx 8804']

psd_dict[path].index.name  # 'Frequencies'
```

- Keys are **original CSV paths**
- Columns are `"Pxx {pos}"` where `{pos}` is the physical probe identifier

`fft_dict` uses analogous `"FFT {pos}"` (magnitude) and `"FFT {pos} complex"` (complex128) columns. Both are always present — stored in parquet as float32 real+imag pairs and recombined on load.

---

## 5. Probe naming (critical convention)

Probes are identified by **position**, not index:

- Position strings come from `ProbeConfiguration.probe_col_name` and `.probe_col_names()` <ref: index=2648718 firstWord=69 lastWord=88/>
- Examples: `"9373/170"`, `"12545"`, `"9373/340"`, `"8804"`
- Raw columns: `"Probe {pos}"`
- Processed: `"eta_{pos}"`, `"Probe {pos}_ma"`, etc.

Do **not** introduce `"Probe 1"`/`"Probe 2"` style names in new code; always go through the configuration or reuse existing patterns.

---

## 6. Run types and wave parameters

Filename parsing lives in `wavescripts/improved_data_loader.py` / `_extract_wave_parameters`-style helpers:

- `-amp####-` → `WaveAmplitudeInput [Volt]` (scaled)  
- `-freq####-` → `WaveFrequencyInput [Hz]` (scaled) <ref: index=2648758 firstWord=478 lastWord=483/>
- Other tags: `-per####-`, `-depth####`, `-mstop####`, `-run#` <ref: index=2648758 firstWord=485 lastWord=495/>

Key semantics (from recent work and REPL logs): <ref: index=2648730 firstWord=1113 lastWord=1124/><ref: index=2648753 firstWord=1 lastWord=24/>

- **Wave runs**: both amp and freq present → positive `WaveFrequencyInput [Hz]`
- **Nowave runs**:
  - No amp/freq tags → both wave fields remain `NaN`
  - Or filenames explicitly containing `nowave` / special tags; these are treated as no-wave even if a stray `freq####` appears
- Stillwater anchors (no wind, no wave) are used to compute baseline “stillwater” levels
- Wind-only runs: nowave + `WindCondition` in `{"full", "lowest"}`

When filtering:

- Wave runs: `meta["WaveFrequencyInput [Hz]"].notna() & (meta["WaveFrequencyInput [Hz]"] > 0)` <ref: index=2648705 firstWord=1015 lastWord=1019/>
- Nowave: `WaveFrequencyInput [Hz]` is `NaN` (and/or filename tags)

---

## 7. Plotting + browsers

### 7.1. Core plotting module

`wavescripts/plotter.py` provides:

- High-level plotting functions (damping, spectra, reconstruction, swell scatter, etc.)
- `plot_reconstructed(fft_dict, meta, freqplotvariables, data_type="fft")`
  - Returns `(fig, axes)`
  - For Qt uses `plt.show()` (the Qt backend is set in `main_explore_browser.py`) <ref: index=2648699 firstWord=914 lastWord=919/>

When adding plots:

- Prefer reusing `plot_reconstructed` / `plot_frequency_spectrum` instead of writing ad-hoc plotting in scripts
- Accept and extend existing `plotvariables` / `freqplotvariables` dicts rather than inventing new argument shapes

### 7.2. Qt exploration (`plot_quicklook.py` + `main_explore_browser.py`)

`wavescripts/plot_quicklook.py` defines Qt browsers: <ref: index=2648691 firstWord=13 lastWord=18/><ref: index=2648691 firstWord=34 lastWord=37/>

- `SignalBrowserFiltered(fft_dict, meta_df, freqplotvariables)`:
  - Lists runs and probe checkboxes (positions derived from `"FFT {pos}"` columns)
  - On selection: calls `plot_reconstructed({path: fft_df}, single_meta, plotvars)` directly — complex columns are in cache now, no on-demand recompute needed
  - Resizes figure to ~¾ of screen via `_resize_to_fraction`
- `RampDetectionBrowser`:
  - Steps through ramp detection results (`gather_ramp_data`)
  - Uses `plot_ramp_detection`-style logic and also resizes plots to ~¾ screen <ref: index=2648688 firstWord=435 lastWord=437/><ref: index=2648675 firstWord=1 lastWord=6/>

---

## 8. Git + workflow expectations

(Short version, extracted from README and prior instructions) <ref: index=2648696 firstWord=1 lastWord=40/>

- Work on **branches**, not `main`:
  - `git checkout -b exp/<short-description>`
- Commit small, descriptive steps
- Merge to `main` only after experiments are stable
- Before risky refactors, make a “safety” commit so you can roll back easily
- Use `pytest` for quick feedback:
  - `pytest -q` for all tests
  - `pytest -q path/to/test_file.py -k test_name` for focused runs

---

## 9. How you (the assistant) should behave

When proposing changes or analysis:

1. **Ask for file contents** (or relevant parts) before editing anything non-trivial.
2. Suggest a **branch name** (`exp/<topic>`) when the change is more than a one-liner.
3. Propose or update **pytest tests** whenever changing analysis logic:
   - Use small synthetic signals
   - Assert on key outputs (e.g. amplitudes, wavenumbers, wave ranges)
4. Respect the existing conventions:
   - Position-based probe naming
   - `combined_meta` + `processed_dfs` + `fft_dict`/`psd_dict` data model
   - Wave/nowave classification rules
5. If this document disagrees with code in the repo, **ask for clarification** instead of guessing.
```
