# ignore_this_archive/

**Everything in this folder (aptly named "ignore_this_archive") is DEAD / SUPERSEDED / ONE-OFF.**

Rule for you and any agent:
- Do NOT read anything here to understand current code.
- Do NOT import from anywhere here.
- Do NOT cite results from here as authoritative.
- If you look for precedent here, treat it as history only.

**Dumping rule**: when in doubt, `mv` it here. Git history preserves the
original path, so nothing is ever lost. Storage is not a constraint.

## Subfolders

### `wavescripts/`
- `plotter_old.py` — 2 396-line predecessor of `wavescripts/plotter.py`.
  Zero live imports at time of archive (2026-04-22).

### `wavescripts_arkiv/`
- Pre-existing in-repo archive of old `processor`/`plotter`/`data_loader`
  variants. Moved here wholesale; no contents changed.

### `analysis_scratch/`
One-off diagnostic scripts that produced a scratch PDF/PNG + findings
markdown. Either:
  - superseded by a delegated script now in `analysis_scratch/` (top-level)
    and wired into `main_save_figures.py`, OR
  - a negative-result / diagnostic whose headline is already captured in
    `memory/` or a `_findings.md` and doesn't need to re-run.

Each script's companion files (`.pdf`, `.png`, `_summary.csv`,
`_findings.md`) were moved together so the archived context is
self-contained.

## Known dangling references (cosmetic only, won't break execution)

- `analysis_scratch/fft_window_sensitivity_lsfit.py:403` emits a markdown
  breadcrumb pointing at `analysis_scratch/hg_window_stability_findings.md`,
  which now lives under `ignore_this_archive/analysis_scratch/`. The
  sibling script still runs fine; its output markdown just has a stale
  relative link.
- `main_save_figures.py` still has *comment* pointers to
  `analysis_scratch/huseby_grue_window.pdf` (line ~626) and
  `analysis_scratch/sliding_afft_15hz_02v_zoom.{py,pdf}` (line ~968).
  Neither file was archived — they stay live under `analysis_scratch/`.

## Recovery

```bash
git log --diff-filter=R --follow -- ignore_this_archive/<path>
git mv ignore_this_archive/<path> <original_location>
```
