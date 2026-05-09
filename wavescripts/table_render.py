"""Generic LaTeX table renderer.

Reads a numeric CSV + a ``.meta.json`` provenance sidecar, assembles a
``\\begin{table} … \\end{table}`` float, and writes it to ``out_tex_path``.

This is a pure presentation layer: data computation lives in the table's
own ``analysis_scratch`` script, which writes the CSV + JSON pair and
then calls :func:`render_table` to produce the thesis ``.tex``.

Provenance comment block layout
-------------------------------
The IMMUTABLE block at the top of the rendered ``.tex`` is reconstructed
from the meta dict. Top-level keys (``script``, ``plot_type``,
``chapter``, ``caption_label``, ``caption_short``) populate the
"Provenance" section. ``generated_at`` is regenerated at render time.

Additional sections come from ``meta["sections"]``:

    "sections": [
        {"title": "Filters",
         "lines": ["panel             : full", "wind              : no, full"]},
        {"title": "Data provenance",
         "lines": ["n_cells           : 12",
                   "n_runs (nw + fw)  : 80",
                   "datasets        :",
                   "  PROCESSED-…",
                   "  PROCESSED-…"]},
        ...
    ]

Each ``lines`` entry is the literal comment-body text. Lines starting
with two spaces are emitted with ``%     `` (extra-indented for sub-items
like dataset enumeration); other lines get ``%   ``.
"""
from __future__ import annotations

import json
from datetime import datetime as _dt
from pathlib import Path
from typing import Callable

import pandas as pd


_PROVENANCE_KEYS = (
    ("script", "script"),
    ("plot_type", "plot_type"),
    ("chapter", "chapter"),
    ("generated_at", "generated_at"),
    ("caption_label", "caption_label"),
    ("caption_short", "caption_short"),
)
_PROVENANCE_KEY_WIDTH = 18
_SECTION_TOTAL_LEN = 64


def _section_header(title: str) -> str:
    """``% — TITLE ──...`` padded to a total of ``_SECTION_TOTAL_LEN`` chars."""
    head = f"% — {title} "
    pad = max(0, _SECTION_TOTAL_LEN - len(head))
    return head + "─" * pad


def _end_block_line() -> str:
    head = "% ── end immutable block "
    pad = max(0, _SECTION_TOTAL_LEN - len(head))
    return head + "─" * pad


def _kv_line(key: str, value: object) -> str:
    val = "" if value is None else str(value)
    if val == "":
        return f"%   {key.ljust(_PROVENANCE_KEY_WIDTH)}:"
    return f"%   {key.ljust(_PROVENANCE_KEY_WIDTH)}: {val}"


def _emit_body_line(line: str) -> str:
    return f"%   {line}"


def _render_immutable_block(meta: dict, label: str) -> str:
    generated_at = _dt.now().isoformat(timespec="seconds")
    prov_values = {
        "script": meta.get("script", ""),
        "plot_type": meta.get("plot_type", ""),
        "chapter": meta.get("chapter", ""),
        "generated_at": generated_at,
        "caption_label": meta.get("caption_label") or label,
        "caption_short": meta.get("caption_short", ""),
    }

    lines: list[str] = [
        "%! TEX root = ../main.tex",
        "% " + "=" * 62,
        "% IMMUTABLE — generated automatically, do not edit this block",
        "%",
        _section_header("Provenance"),
    ]
    for display_key, meta_key in _PROVENANCE_KEYS:
        lines.append(_kv_line(display_key, prov_values[meta_key]))
    lines.append("%")

    for sec in meta.get("sections", []):
        title = sec.get("title", "")
        body = sec.get("lines", []) or []
        lines.append(_section_header(title))
        for ln in body:
            lines.append(_emit_body_line(ln))
        lines.append("%")

    lines.append(_end_block_line())
    return "\n".join(lines)


def _render_caption_block(
    caption: str | None,
    short_caption: str | None,
) -> str:
    """Render the caption block, wrapped in CAPTION-SYNC sentinels.

    The sentinels let ``analysis_scratch/sync_captions.py`` patch caption
    text in-place without re-rendering the table — see the design note at
    ``memory/workflow_caption_only_edits_design_note.md`` (Option D).
    """
    if caption and short_caption:
        body = (
            f"  \\caption[{short_caption}]{{\n"
            f"    {caption}\n"
            f"  }}"
        )
    elif caption:
        body = (
            f"  \\caption{{\n"
            f"    {caption}\n"
            f"  }}"
        )
    else:
        body = (
            "  \\caption{\n"
            "    % TODO: write caption "
            "(edit FIGURE_CAPTIONS/TABLE_CAPTIONS)\n"
            "  }"
        )
    return (
        "% >>> CAPTION-SYNC START "
        "(do not edit this block; sync_captions.py overwrites)\n"
        f"{body}\n"
        "% <<< CAPTION-SYNC END\n"
    )


def _render_body(
    df: pd.DataFrame,
    columns: list[str],
    column_headers: list[str],
    column_spec: str,
    row_groups: list[tuple[str | None, Callable[[pd.DataFrame], pd.DataFrame]]],
    cell_format: dict[str, Callable[[pd.Series], str]],
    indent: str = "    ",
) -> str:
    if len(columns) != len(column_headers):
        raise ValueError(
            f"columns ({len(columns)}) and column_headers "
            f"({len(column_headers)}) must have the same length"
        )
    n_cols = len(columns)

    body: list[str] = [
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\small",
        f"  \\begin{{tabular}}{{{column_spec}}}",
        f"{indent}\\toprule",
        f"{indent}{' & '.join(column_headers)} \\\\",
        f"{indent}\\midrule",
    ]

    for i, (group_label, filter_fn) in enumerate(row_groups):
        sub = filter_fn(df)
        if i > 0:
            body.append(f"{indent}\\midrule")
        if group_label is not None:
            body.append(
                f"{indent}\\multicolumn{{{n_cols}}}{{l}}{{{group_label}}} \\\\"
            )
        for _, row in sub.iterrows():
            cells = [cell_format[c](row) for c in columns]
            body.append(f"{indent}{' & '.join(cells)} \\\\")

    body.append(f"{indent}\\bottomrule")
    body.append("  \\end{tabular}")
    return "\n".join(body)


def render_table(
    csv_path: Path,
    meta_path: Path,
    out_tex_path: Path,
    *,
    columns: list[str],
    column_headers: list[str],
    column_spec: str,
    cell_format: dict[str, Callable[[pd.Series], str]],
    row_groups: list[tuple[str | None, Callable[[pd.DataFrame], pd.DataFrame]]],
    label: str,
    caption: str | None = None,
    short_caption: str | None = None,
) -> None:
    """Render a CSV + meta.json pair into a thesis-style LaTeX table.

    See module docstring for the meta dict shape and the caption logic.
    """
    df = pd.read_csv(csv_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    immutable = _render_immutable_block(meta, label)

    body = _render_body(
        df,
        columns=columns,
        column_headers=column_headers,
        column_spec=column_spec,
        row_groups=row_groups,
        cell_format=cell_format,
    )

    caption_block = _render_caption_block(caption, short_caption)

    full = (
        immutable
        + "\n"
        + body
        + "\n"
        + caption_block
        + f"  \\label{{{label}}}\n"
        + "\\end{table}\n"
    )

    out_tex_path.parent.mkdir(parents=True, exist_ok=True)
    out_tex_path.write_text(full, encoding="utf-8")
