#!/usr/bin/env python3
"""
sync_captions.py — patch caption blocks in rendered .tex stubs in-place.

Caption-only edit workflow (Option D from
``memory/workflow_caption_only_edits_design_note.md``):

    1. User edits a value in ``FIGURE_CAPTIONS`` / ``TABLE_CAPTIONS``
       (these dicts live in ``main_save_figures.py`` / ``main_save_tables.py``
       and are persisted at import-time to ``output/.figure_captions.json``
       and ``output/.table_captions.json``).
    2. User runs:

           python analysis_scratch/sync_captions.py

       The script walks every ``output/TEXFIGU/*.tex`` and ``output/TABLES/*.tex``
       file, finds the ``% >>> CAPTION-SYNC START`` ... ``% <<< CAPTION-SYNC END``
       sentinel block, and rewrites the caption inside it from the JSON cache.
    3. Done. No data reload, no figure regen.

Bootstrap: a stub gets sentinels the first time it is rendered by an
updated renderer (``wavescripts/table_render.py``,
``wavescripts/plot_utils.py``, ``analysis_scratch/parallel_probe_psd_agreement.py``).
For older stubs, run the underlying script once — after that, all
caption edits are sync-only.

Use ``--dry-run`` to preview changes without writing files.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG_CAPTIONS_JSON = REPO_ROOT / "output" / ".figure_captions.json"
TAB_CAPTIONS_JSON = REPO_ROOT / "output" / ".table_captions.json"
TEXFIGU_DIR = REPO_ROOT / "output" / "TEXFIGU"
TABLES_DIR = REPO_ROOT / "output" / "TABLES"

SENTINEL_START = "% >>> CAPTION-SYNC START (do not edit this block; sync_captions.py overwrites)"
SENTINEL_START_PREFIX = "% >>> CAPTION-SYNC START"  # tolerant match
SENTINEL_END = "% <<< CAPTION-SYNC END"

# Regex for finding the figure/table label outside the sentinel block.
LABEL_RE = re.compile(r"\\label\{(fig|tab):([^}]+)\}")


def render_caption_block(caption_full: str, caption_short: str,
                         indent: str = "  ") -> str:
    """Render the lines that go INSIDE the sentinel block.

    Mirrors the renderers in table_render.py / plot_utils.py exactly:
    - both full + short → ``\\caption[short]{full}`` (multi-line layout)
    - full only         → ``\\caption{full}``
    - empty             → TODO placeholder (matches existing stubs)
    """
    if caption_full and caption_short:
        return (
            f"{indent}\\caption[{caption_short}]{{\n"
            f"{indent}  {caption_full}\n"
            f"{indent}}}"
        )
    if caption_full:
        return (
            f"{indent}\\caption{{\n"
            f"{indent}  {caption_full}\n"
            f"{indent}}}"
        )
    return (
        f"{indent}\\caption{{\n"
        f"{indent}  % TODO: write caption "
        f"(edit FIGURE_CAPTIONS/TABLE_CAPTIONS)\n"
        f"{indent}}}"
    )


def wrap_with_sentinels(caption_full: str, caption_short: str,
                        indent: str = "  ") -> str:
    """Full sentinel block, ready to splice into a .tex file.

    Trailing newline NOT included — caller handles surrounding newlines.
    """
    body = render_caption_block(caption_full, caption_short, indent=indent)
    return (
        f"{SENTINEL_START}\n"
        f"{body}\n"
        f"{SENTINEL_END}"
    )


def _find_sentinel_lines(lines: list[str]) -> list[tuple[int, int]]:
    """Return [(start_idx, end_idx), ...] for each sentinel block found.

    Indices point at the SENTINEL lines themselves (inclusive).
    """
    blocks: list[tuple[int, int]] = []
    start: int | None = None
    for i, line in enumerate(lines):
        stripped = line.rstrip()
        if stripped.startswith(SENTINEL_START_PREFIX):
            if start is not None:
                # malformed — restart on the new START
                start = i
            else:
                start = i
        elif stripped == SENTINEL_END:
            if start is not None:
                blocks.append((start, i))
                start = None
    return blocks


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {"full": {}, "short": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_captions(name: str, kind: str, fig_json: dict, tab_json: dict
                      ) -> tuple[str, str]:
    """Return (full, short) caption text for ``name`` of ``kind``.

    kind = ``fig`` or ``tab``. Missing entries → empty string.
    """
    src = fig_json if kind == "fig" else tab_json
    full = (src.get("full") or {}).get(name, "") or ""
    short = (src.get("short") or {}).get(name, "") or ""
    return full, short


def patch_file(tex_path: Path, fig_json: dict, tab_json: dict, *,
               dry_run: bool = False) -> tuple[str, str]:
    """Patch a single .tex file in-place. Returns (status, detail).

    status in {OK, SKIP, WARN}. detail is a short human-readable string.
    """
    text = tex_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    # Locate the sentinel block(s).
    blocks = _find_sentinel_lines(lines)
    if not blocks:
        return ("WARN", "no sentinels — run the data/figure script once to bootstrap")
    if len(blocks) > 1:
        return ("WARN", f"{len(blocks)} sentinel blocks found — file is malformed; skipped")

    start_idx, end_idx = blocks[0]

    # Find the label OUTSIDE the sentinel block.
    label_match = None
    for i, line in enumerate(lines):
        if start_idx <= i <= end_idx:
            continue
        m = LABEL_RE.search(line)
        if m:
            label_match = m
            break
    if label_match is None:
        return ("WARN", "no \\label{fig:...} or \\label{tab:...} found; skipped")

    kind = label_match.group(1)  # 'fig' or 'tab'
    name = label_match.group(2)
    full, short = _resolve_captions(name, kind, fig_json, tab_json)

    new_block = wrap_with_sentinels(full, short, indent="  ")
    new_block_lines = new_block.splitlines()

    # Splice: lines[:start_idx] + new_block_lines + lines[end_idx+1:]
    new_lines = lines[:start_idx] + new_block_lines + lines[end_idx + 1:]
    new_text = "\n".join(new_lines)
    # Preserve trailing newline if original had one.
    if text.endswith("\n") and not new_text.endswith("\n"):
        new_text += "\n"

    if new_text == text:
        return ("OK",
                f"no change (caption: {len(full)} chars / short: {len(short)} chars)")

    if not dry_run:
        tex_path.write_text(new_text, encoding="utf-8")

    verb = "would update" if dry_run else "updated"
    return ("OK",
            f"{verb} (caption: {len(full)} chars / short: {len(short)} chars)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would change but don't write files.")
    args = parser.parse_args(argv)

    fig_json = _load_json(FIG_CAPTIONS_JSON)
    tab_json = _load_json(TAB_CAPTIONS_JSON)

    targets: list[Path] = []
    if TEXFIGU_DIR.exists():
        targets.extend(sorted(TEXFIGU_DIR.glob("*.tex")))
    if TABLES_DIR.exists():
        targets.extend(sorted(TABLES_DIR.glob("*.tex")))

    if not targets:
        print(f"No .tex files found under {TEXFIGU_DIR} or {TABLES_DIR}.")
        return 1

    n_synced = 0
    n_skipped = 0
    n_unchanged = 0
    for path in targets:
        rel = path.relative_to(REPO_ROOT)
        try:
            status, detail = patch_file(path, fig_json, tab_json,
                                        dry_run=args.dry_run)
        except Exception as exc:
            print(f"ERROR: {rel} — {exc}")
            n_skipped += 1
            continue

        if status == "WARN":
            print(f"WARN: {rel} — {detail}")
            n_skipped += 1
        else:
            print(f"OK:   {rel}  ({detail})")
            if "no change" in detail:
                n_unchanged += 1
            else:
                n_synced += 1

    if args.dry_run:
        print(f"\nDry run: would update {n_synced} files, "
              f"unchanged {n_unchanged}, skipped {n_skipped}.")
    else:
        print(f"\nSynced {n_synced} files, unchanged {n_unchanged}, "
              f"skipped {n_skipped}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
