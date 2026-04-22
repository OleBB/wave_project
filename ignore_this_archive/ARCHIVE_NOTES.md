# Archive notes

Context for `ignore_this_archive/`. Read this file if you need to know *why*
something was moved here; treat everything else in this folder as
DEAD / SUPERSEDED / ONE-OFF.

Rule for you and any agent:
- Do NOT read the other files here to understand current code or results.
- Do NOT import from anywhere here.
- Do NOT cite results from here as authoritative.
- If you look for precedent here, treat it as history only.

**Dumping rule**: when in doubt, `mv` it here. Git history preserves the
original path, so nothing is ever lost. Storage is not a constraint.

---

## Why things here got archived (2026-04-22)

### `wavescripts/`

| Item | Why archived |
|---|---|
| `plotter_old.py` | 2 396-line predecessor of `wavescripts/plotter.py`. Grep showed **0 live imports** across the repo at time of archive. |

### `wavescripts_arkiv/`

Moved wholesale — this was already an in-repo archive of reserve/old variants
(`reserve_*.py`, `old_*.py`, `lauskode.py`, etc.) kept during the 2025–2026
refactors. None of it is imported by live code. Grouped here so the top-level
`wavescripts/` contains only the 15 live modules.

### `analysis_scratch/` — the one-off diagnostic scripts

All of these were standalone investigations. Their **headline finding** is
either (a) already baked into the live methodology in `CLAUDE.md` / `memory/`,
or (b) superseded by a successor script that IS wired into
`main_save_figures.py`. Each kept its own `_findings.md` companion so the
archived context stays self-contained.

| Script | Why archived — what it asked vs. what's live now |
|---|---|
| `hg_window_stability.py` · `hg_window_stability_with_per40.py` | **Superseded by** `analysis_scratch/fft_window_sensitivity_lsfit.py` (now §4i of `main_save_figures.py`). Both were early attempts at characterising the stability of the Huseby–Grue plateau; the LS-fit sweep replaced them with a cleaner 4-method comparison on 128 real measurements. |
| `hg_travel_time_shift_check.py` | One-off sanity check during the H&G probe-shift design (confirmed the probe-shift formula against real IN/OUT time lags). Conclusion captured in `memory/methodology_hg_probe_shifted.md`; the live diagnostic is `analysis_scratch/hg_snap_shift_diagnostic.py` (NOT archived). |
| `per40_active_vs_late_vs_hg.py` | **Superseded by** `analysis_scratch/per40_and_per240_HG_shifted.py` (now §4n). Exploratory window-zone comparison; the shifted-H&G version is the clean answer. |
| `recon_before_after.py` · `recon_check.py` · `reconstruct_all3.py` · `reconstruct_variants.py` | **All four superseded by** `analysis_scratch/reconstruction_A_vs_B.py` (now §4f + §4g). These were iterative exploration of reconstruction variants; the A-vs-B script is the final reproducible form. |
| `quicklook_1600hz.py` | Single-frequency diagnostic for the 1.6 Hz dropout concern. Conclusion absorbed into `memory/feedback_freq_amp_limits.md` (scope exclusion) and `memory/known_baddata_ultrasound_16hz.md`. |
| `rolling_rms_stationarity.py` | Stationarity check that used to live outside the pipeline. The quality-flag system in `processor.py` now catches the same failure modes via `quality_flag` / `wave_stability` / `period_amplitude_cv`. |
| `mooring_sw_bias.py` | Negative-result diagnostic — asked whether mooring geometry biases the standing-wave correction. Answer: no. Captured briefly in `memory/session_2026-04-16.md`. |
| `mf_windonly.py` | One-off test of the Mansard–Funke reflection method on pure-wind runs. Not a thesis artefact. Mansard–Funke proper is live at §4c (`mansard_funke.py`). |
| `sw_probe_ratio.py` | Standing-wave probe-ratio diagnostic. Output captured in its findings sibling; SW correction itself lives at §4d (`sw_correction.py`, a separate script). |
| `windwave_eta_statistics.py` | Wind-only η statistics one-off (η-PDF shape, skew, kurtosis). Numbers are in `memory/archive/session_2026-04-20.md`; the LabVIEW-Kurt bug fix it triggered is in `windprofile_combined.py`. |
| `wind_psd_shape_cond1_vs_cond4.py` | Wind-PSD shape comparison between two tunnel roof configs. Absorbed into the live `ch04_wind_psd` plotter call in §4-1. Details in `memory/project_band_residual_decision.md` and `memory/methodology_reconstruction_and_stokes.md`. |
| `probe_bias_diagnostic.py` | Probe inter-calibration check across parallel probes under fullwind. Conclusion: wall-side (`/170`) reads higher at high-freq fullwind; mean-IN absorbs it. Live version is §3e (`parallel_probe_agreement.py`) + the bias caveat in `CLAUDE.md` §6. |
| `probe_height_analysis.py` | Precursor to `probe_height_figure.py` (the live §3b figure). Early exploration; the `_figure.py` sibling is the polished version. |
| `period_overlay.py` | Phase-lock diagnostic on one 1.6 Hz ultrasound-glitch run. Conclusion: 100 % of periods affected — now in `memory/known_baddata_ultrasound_16hz.md`. |
| `run_assumption_tests.py` | FFT/window/stationarity assumption audit. Passed → negative result, no live figure. Raw output lives alongside as `run_assumption_tests_output.txt`. |

### Top-level files

| Item | Why archived |
|---|---|
| `about-claude.md` · `rules_for_agent2.md` | Superseded by `CLAUDE.md` (Mar 20 predecessors). |
| `main.ipynb` | Superseded by `main.py` + `main_explore_inline.py` (2.1 MB legacy notebook, 0 live refs). |
| `damping_analysis.html` | One-off dtale HTML dump, Mar 30. |
| `fft_vs_td_17apr.md` | Dated note; result is baked into `CLAUDE.md` §16 (time-domain vs FFT amplitude rule). |
| `notes.txt` | Generic scratch. |
| `dtale-probe_uncertainty_tables.csv` | Only referenced from `CLAUDE.md` §16 (with a *mismatched* filename at that). The §16 table itself was copied from this CSV and remains authoritative. |
| `plotsettings.json` | Only referenced from the already-archived `wavescripts_arkiv/rester_mainTester.py`. |
| `apr16-repl/` · `repl-16apr/` · `repl/` · `claude-prompt-backup/` | Dated REPL snapshots + old agent prompt backups. |
| `wavenotebooks/` (most `.ipynb`) | Legacy Jupyter notebooks (sept/oct/nov/jan/feb) superseded by `main_explore_inline.py`. |
| `wavezarchive/` | Pre-existing local dump folder the user had on `.gitignore`; folded in here for consistency. |

---

## Known dangling references (cosmetic only, nothing breaks)

Markdown breadcrumbs that now point at archived files. The scripts that emit
them still run fine; only the link text is stale. Updated in live memory
docs on 2026-04-22; left as-is in session logs and `_findings.md` files
because those are historical records.

- `analysis_scratch/fft_window_sensitivity_lsfit.py:403` — emits the string
  `Plateau: analysis_scratch/hg_window_stability_findings.md` into its output
  findings file. The referenced file now lives here.
- `analysis_scratch/fft_window_position_sensitivity_lsfit_findings.md` —
  references archived `hg_window_stability_findings.md` and
  `per40_active_vs_late_vs_hg_findings.md`.
- Various `analysis_scratch/*_findings.md` cross-reference other archived
  findings. All cosmetic.
- `main_save_figures.py` still has *comment* pointers to
  `analysis_scratch/huseby_grue_window.pdf` (line ~626) and
  `analysis_scratch/sliding_afft_15hz_02v_zoom.{py,pdf}` (line ~968). Neither
  was archived — they stay live under `analysis_scratch/`.

## Recovery

```bash
# Find the rename / deletion commit for a specific path:
git log --diff-filter=DR --follow -- ignore_this_archive/<path>

# Pull a file back out:
git mv ignore_this_archive/<path> <original_location>

# Or just view a past version without restoring:
git show <sha>:<original_path>
```
