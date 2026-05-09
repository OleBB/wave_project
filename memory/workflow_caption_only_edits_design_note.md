---
name: workflow_caption_only_edits_design_note
description: Design note — caption-only edits should not require re-rendering data. Current pain point + why the render-only table pipeline works better + path-forward sketches.
type: project
---
# Workflow design note — caption-only edits shouldn't trigger data regen

> **STATUS: IMPLEMENTED 2026-05-09** — Option D (in-place caption
> patcher with sentinel-bracketed blocks) is live. The sync tool is
> [`analysis_scratch/sync_captions.py`](../analysis_scratch/sync_captions.py).
> Renderers updated: `wavescripts/table_render.py`,
> `wavescripts/plot_utils.py`,
> `analysis_scratch/parallel_probe_psd_agreement.py`. Stubs get sentinels
> the first time they're re-rendered after this date; older stubs warn
> "no sentinels — run the data/figure script once to bootstrap" and skip.
> Workflow: edit caption in `FIGURE_CAPTIONS` / `TABLE_CAPTIONS` →
> re-run `main_save_tables.py` (or `main_save_figures.py`) once to
> refresh the JSON cache → run `python analysis_scratch/sync_captions.py`.
> Done.

## The pain point (recorded 2026-05-09 from user)

> "this tex stub needs regen — just because of the caption.
>  but really — I shouldnt need to regen the tex stub all the time, to do minor edits.
>  that's my system — and it's not working that well to be fair.
>  so note that I want to have a new system — more 'user-based'.
>  basically — I won't overwrite the tex files all the time.
>  problem is — hard to keep track.
>  the table-pipeline works better... having precomputed everything (it's just CSVs right)
>  then all edits to the table are super quick."

## What's happening today

For most figures and some tables, the workflow is:

```
edit FIGURE_CAPTIONS / TABLE_CAPTIONS
   ↓
re-run main_save_tables.py  (writes output/.{figure,table}_captions.json)
   ↓
re-run the data/figure script (reads JSON, embeds caption into .tex stub)
   ↓
.tex stub on disk now has the new caption
```

Every caption edit requires re-rendering the underlying artefact —
even for trivial typo fixes. The user wants the caption layer to be
edit-able WITHOUT triggering data/figure regen.

### Concrete example flagged today

```python
"ch04_amp_methods_a1_fullwind":
    r"Nærbilde av tidsserier for innkommende bølge, $A_1$. Full vind for
    alle fire frekevensene. Samme tidsakse i sekunder fra start.
    Verdiene er beregnet fra hver bølges eget vindu på 10 perioder.",
```

This caption was edited in `FIGURE_CAPTIONS`. To make it land in the
rendered `.tex` stub, the underlying figure script has to re-run —
which loads PSD data and re-draws the figure. Wasteful for what is
just a sentence change.

## Why the table-render pipeline already works better (partial fix)

The 2026-05-08 split into `main_save_figures.py` (data) +
`main_save_tables.py` (render-only) made *table* caption edits fast:

```
edit TABLE_CAPTIONS in main_save_tables.py
   ↓
re-run main_save_tables.py  (~50 ms per cell — reads CSV + meta.json,
                              re-emits .tex with new caption)
   ↓
.tex on disk has the new caption
```

No CSV regen, no PSD reload. **This works well** — the user has
already noticed and approved the pattern.

The wrinkle: tables that still render from `main_save_figures.py`
cells (caption-centralised but data-script-bound) don't benefit from
this — they fall back to the slow path. See the comment at the top
of `main_save_tables.py`:

```
Captions-only (rendered from main_save_figures.py cells):
  ch04_window_choice_nowind / fullwind
  ch04_tidsvindu
  ch04_wind_setup_baseline_table
  ch05_wind_effect_table / _by_amp
  ch05_transmission_wind_ratios / _amplitudes
```

## What the user wants — "user-based"

The intent (paraphrased): a caption-only edit should:
- Not reload data.
- Not re-render figures.
- Land in the relevant `.tex` stub in ~seconds, not minutes.
- Not require the user to remember which scripts to re-run.

In other words: the *caption layer* should decouple from the
*content layer* the way the render-only table pipeline already does.

## Path-forward sketches (for future agent / future session)

Three ways to extend the table-pipeline pattern to figures.

### Option A — Caption-only update script

A new `analysis_scratch/sync_captions_to_stubs.py` that:
1. Reads `output/.figure_captions.json` (already maintained).
2. For each `output/TEXFIGU/*.tex` and `output/TABLES/*.tex`, parses
   the existing `\caption[short]{full}` block and replaces it with the
   current JSON value.
3. Leaves everything else (the IMMUTABLE block, body, `\label`,
   `\input` of subfigures, etc.) untouched.

Speed: ~200 ms for the whole tree.
Risk: regex-based caption parsing must handle Norwegian special chars,
math mode `$...$`, `\ref{}`, `\texttt{}`, multi-line captions, etc.
Probably tractable but needs careful testing.

### Option B — \caption{\input{captions/<name>.tex}}

Each `.tex` stub references the caption indirectly:

```latex
\caption[\input{captions/ch04_amp_methods_a1_fullwind.short.tex}]{%
  \input{captions/ch04_amp_methods_a1_fullwind.tex}%
}
```

A `caption_export.py` script writes one tiny file per caption from the
JSON. User edits captions by either editing the dicts + running the
export OR editing the `.tex` files directly.

Speed: same as A.
Risk: many small files (one per caption × 2 short/full = lots of
files). Pollution of the `output/` tree. But conceptually clean.

### Option C — LaTeX macro-based key/value lookup

Define a single `output/captions_definitions.tex` that contains all
captions as macros:

```latex
\newcommand{\captionFigChAmpMethodsAOneFullwind}{Nærbilde av tidsserier ...}
```

Each `.tex` stub does:

```latex
\caption[\shortCaptionFigChAmpMethodsAOneFullwind]{%
  \captionFigChAmpMethodsAOneFullwind%
}
```

A `caption_macros.py` script writes the macros file from the JSON.

Speed: same as A.
Risk: macro names are fragile (camelCase from snake_case), and LaTeX
errors if a key is missing have ugly error messages.

### Option D — In-place caption patcher (most pragmatic)

Variant of A but using a sentinel-line approach instead of regex:

When a `.tex` stub is first written, the caption block is bracketed
by special comments:

```latex
% --- caption start (auto-generated, sync with main_save_*.py) ---
\caption[Short]{%
  Full caption.%
}
% --- caption end ---
```

A `sync_captions.py` script finds these blocks and replaces them
in-place using line-based matching, no regex magic. Simple, robust.

User can also hand-edit the captions outside the brackets if they want
LaTeX-only flexibility (e.g., `\subref{}`, hyperlinks) without losing
sync — the script only touches inside the brackets.

This is probably the best path. Hybrid of "user owns captions in
LaTeX" and "writer owns captions in Python" — they pick whichever is
faster for the moment.

## Recommendation

When the user has 30 minutes for plumbing work:

1. Pick **Option D** (in-place caption patcher with sentinel comments).
2. Modify the existing renderers to emit the sentinel-bracketed caption block.
3. Add `analysis_scratch/sync_captions.py` that walks `output/{TEXFIGU,TABLES}/*.tex`,
   finds the sentinel block, replaces it with the current JSON value.
4. Update the session-startup checklist in MEMORY.md to include
   `python analysis_scratch/sync_captions.py` as the post-edit step.

Caption edits then take ~1 second instead of minutes; no risk of
forgetting which scripts to re-run; the sentinel comments give the
user a hand-edit escape hatch.

## Until then — the manual workaround

For figures (caption-bound to data render):

```bash
# After editing FIGURE_CAPTIONS, re-run only the affected figure script:
python analysis_scratch/<the-relevant-figure-script>.py
```

For tables already in the render-only pipeline:

```bash
# After editing TABLE_CAPTIONS:
python main_save_tables.py
```

These both work but are slower than the proposed "Option D" patcher.

## Cross-references

- `main_save_tables.py` — top docstring documents the render-only convention (the table pipeline).
- `main_save_figures.py` — figures still bound to data render.
- 2026-05-09 durable rule "captions are user-owned" lives in
  `memory/MEMORY.md` (or should — propagate if not there yet). The
  caption-only-edit workflow is the practical extension of that rule:
  if captions are user-owned, editing them should be one-step.
