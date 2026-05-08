"""Shared save-side utilities for the figure / table orchestrators.

Currently exports `_run_delegated_if_missing`, the helper that lets
`main_save_figures.py` and `main_save_tables.py` invoke standalone
data-side scripts (under ``analysis_scratch/``) only when their
declared output files are absent. Both orchestrators import the
function from here so they share one regenerate-toggle.

Setting ``REGENERATE_DELEGATED = True`` (either by mutating this
module attribute or via the orchestrator's ``--regenerate`` CLI flag)
forces every delegated script to re-run even if outputs exist.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Repo root — `wavescripts/` sits one directory below the repo root.
file_dir = Path(__file__).resolve().parent.parent

# Default: only run delegated scripts when their outputs are missing.
# Either orchestrator may flip this to True before calling the helper.
REGENERATE_DELEGATED = False


def _run_delegated_if_missing(
    script_rel: str,
    outputs: list[Path],
    label: str | None = None,
    *,
    force: bool | None = None,
    timeout_s: int = 900,
) -> None:
    """Run ``analysis_scratch/<script>`` if any expected output is missing.

    Parameters
    ----------
    script_rel : str
        Repo-relative path of the scratch script.
    outputs : list[Path]
        Files the script is expected to write. Checked with ``exists()``;
        write-once stubs + existing PDFs both qualify as "already there".
    label : str, optional
        Short label for the status line. Defaults to the first output stem.
    force : bool, optional
        Run the script even if all outputs are already present. Defaults
        to the module-level ``REGENERATE_DELEGATED`` toggle.
    timeout_s : int
        Kill the subprocess after this many seconds. Default 900 (15 min).

    Never raises — the cell's downstream existence check still fires a
    visible warning if the figure truly didn't land.
    """
    label = label or Path(outputs[0]).stem
    missing = [p for p in outputs if not p.exists()]
    if force is None:
        force = REGENERATE_DELEGATED
    if not missing and not force:
        print(f"  {label}: OK ({len(outputs)} output(s) present)")
        return
    reason = "REGENERATE_DELEGATED=True" if force else f"{len(missing)} missing"
    print(f"  {label}: running {script_rel} ({reason})")
    try:
        r = subprocess.run(
            # sys.executable = the same interpreter this file is running in,
            # so the subprocess inherits the conda env (draumkvedet) whether
            # the orchestrator is run from CLI, Zed REPL, or a notebook.
            [sys.executable, script_rel],
            cwd=str(file_dir),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        print(f"    {label}: TIMEOUT after {timeout_s}s — script killed")
        return
    if r.returncode != 0:
        tail = (r.stderr or "(no stderr)")[-400:].rstrip()
        print(f"    {label}: FAILED (rc={r.returncode}); stderr tail: {tail}")
        return
    still_missing = [p.name for p in outputs if not p.exists()]
    if still_missing:
        print(f"    {label}: ran but outputs still missing → {still_missing}")
        return
    print(f"    {label}: regenerated {len(outputs)} output(s)")
