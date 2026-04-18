# Archived 2026-04-18 — mean_in_probe post-load hook (superseded)

This module (`wavescripts/mean_in_probe.py`) was the first implementation
of the mean-IN canonical reference — a post-load transformation applied
to `meta_results` in `main_save_figures.py`.

**Why archived**: the logic was moved into `processor2nd.py::_update_more_metrics`
so `meta.json` on disk carries the canonical columns directly. A post-load
hook is no longer needed. See commit message for the pipeline replacement.

The archived version hardcoded `9373/170` + `9373/340`. The pipeline
replacement generalizes via `ProbeConfiguration` so the same rule applies
to every probe era (e.g. Nov 2025 uses OUT = mean of 12400/170 + 12400/340
while IN stays single at 9373/250).

Kept here for quick reference if the hook pattern is ever useful again
(e.g. for iterating without `--force-recompute`).
