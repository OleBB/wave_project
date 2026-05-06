# Session 2026-05-06 (continuation, "pt2") — tables, curve fits, loose230 vs loose300

Continuation of [`session_2026-05-06_panel_mooring_probecfg.md`](session_2026-05-06_panel_mooring_probecfg.md).
Where the morning was about exploring (mooring × panel × wind) data cuts,
the afternoon was about hardening the tools the figures need, then
chasing one specific finding the new tools surfaced (the within-canon
mooring-slack effect).

Observation-vs-inference rule (CLAUDE.md §20) followed throughout: every
"X causes Y" claim is flagged as a *candidate explanation (hypothesis)*,
not a finding.

---

## Thread 6 — Companion table for the mooring_focus figure

User asked: "what accompanying table should we use to display the hard
numbers?" Decision: a wind-effect-style table mirroring the layout of
`ch05_wind_effect_table_by_amp` but with `(panel, mooring)` as the inner
dimension instead of frequency, restricted to 1.30 Hz.

Built `analysis_scratch/mooring_focus_at_1_3hz_table.py`. 9 rows
(3 amps × 3 panel·mooring combos); the `revers · below_90` cell is
honestly absent (never run in the experiment).

**Headline numbers** (full table at
`output/TABLES/ch05_mooring_focus_at_1_3hz_table.tex`):
- A1 normal below_90: ΔK_t = +0.119, ratio_Kt = 1.180
- A1 normal above_50: ratio_Kt ≈ 1.007 (no wind effect — above-water
  mooring locks the panel down)
- A1 revers above_50: ratio_Kt = 1.142 (n=2 fullwind, soft)

**Wired into main_save_figures.py** as a `[DELEG]` cell after the
ch05_mooring_focus_at_1_3hz_ka figure, with placeholder caption + entry
in FIGURE INDEX `§4b`.

Files:
```
analysis_scratch/mooring_focus_at_1_3hz_table.py
analysis_scratch/mooring_focus_at_1_3hz_table.csv
output/TABLES/ch05_mooring_focus_at_1_3hz_table.tex
```

---

## Thread 7 — Wind-effect tables: column overhaul

User feedback in-thesis: the existing tables had three columns
(`$\Delta K_t$ [pp]`, `$K_t$-økn. [%]`, `D-red. [%]`) that were either
unit-confused (percentage points where decimal would do) or
self-explanatorily impossible (`D-red.` = "damping reduction" — needed a
better name or removal).

**Applied uniformly to three table scripts** (canon mooring focus + the
two wind-effect siblings):

| Old column | New column | Format |
|---|---|---|
| `$\Delta K_t$ [pp]` (e.g. +11.9) | `$\Delta K_t$` (e.g. +0.119) | signed decimal |
| `$K_t$-økn. [%]` (e.g. +18.0) | `$K_{t,\text{vind}}/K_{t,\text{uten}}$` (e.g. 1.180) | unsigned ratio |
| `D-red. [%]` (e.g. +34.7) | `$D_{\text{vind}}/D_{\text{uten}}$` (e.g. 0.653) | unsigned ratio, D = 1−K_t |

Sign conventions now self-evident from the column names. The freq-outer
sibling (`wind_effect_table.py`) was updated alongside the by_amp sibling
to keep the two-table set consistent, even though the user only mentioned
the by_amp variant — since the user has since said they dropped the
freq-version, this isn't load-bearing.

Files touched:
```
analysis_scratch/wind_effect_table.py            (formulas, header, immutable, docstring)
analysis_scratch/wind_effect_table_by_amp.py     (formulas, header, immutable)
analysis_scratch/mooring_focus_at_1_3hz_table.py (formulas, header, immutable)
main_save_figures.py                              (placeholder caption updated)
3 regenerated .tex files under output/TABLES/
```

---

## Thread 8 — Damping-vs-ka with per-wind curve fits

Triggered by a thesis paragraph the user had drafted about A1: "for the
wind data we see larger spread per frequency, despite there being more
data points. Without-wind data lies clearly along a curve."

**Built `analysis_scratch/damping_ka_per_volt_with_fit.py`**: copied from
`damping_ka_per_volt.py`, added a degree-2 polynomial fit per (amplitude,
wind), pooled across per_tag (per240 + per40). 3 per-amp PDFs + 1
combined (no fits in combined, too dense). Each per-amp subfigure carries
an in-axis annotation with `n` and `R²` per wind condition.

**R² results** quantify the user's observation:

| amp | wind | n | R² (poly-2) |
|---|---|---|---|
| A1 | uten | 15 | 0.979 |
| A1 | med | 19 | 0.647 |
| A2 | uten | 8 | 0.988 |
| A2 | med | 12 | 0.861 |
| A3 | uten | 9 | 0.990 |
| A3 | med | 17 | 0.958 |

A1 fullwind R² = 0.65 vs nowind R² = 0.98 — exactly what the user wrote.
At A2/A3 the fullwind R² recovers to >0.86, so A1 is genuinely the
problem cell.

**Wired into main_save_figures.py** as `[DELEG]` cell + 4 captions
(short filled, long marked TODO) + FIGURE INDEX `§4a`.

**One iteration that was reverted**: tried a second "per-frequency-mean"
dashed line on top, in case the smooth fit was being fooled by
ka-overlap between adjacent frequencies. Lines tracked closely and added
visual noise. User asked to drop. Reverted.

Files:
```
analysis_scratch/damping_ka_per_volt_with_fit.py
output/FIGURES/ch05_damping_ka_fit{,_A1,_A2,_A3}.pdf
output/TEXFIGU/ch05_damping_ka_fit{,_A1,_A2,_A3}.tex
main_save_figures.py                              (DELEG cell + 4 caption entries + INDEX)
```

---

## Thread 9 — A1 / 1.4 Hz outlier → loose230 vs loose300 mooring slack

User spotted an oddity in `ch05_damping_ka_fit_A1.pdf`: at 1.4 Hz, two
points with similar ka (~0.103) differ by 0.17 in K_t. Drilled into the
underlying meta to identify the driver.

**Inspection result**: the two near-identical-ka points differ by
**Mooring** — `below_90_loose300` (30 cm rubber-band slack) gave
K_t = 0.641; `below_90_loose230` (23 cm slack) gave K_t = 0.812. Same
per_tag, same wind, same paddle setting, same probe config (h100/low).
A_in differs by 13 % between the two moorings, A_out also differs — both
push K_t in the same direction.

This was the within-canon analogue of the morning's mooring effect
(above_50 vs below_90), now visible inside `below_90` itself.

### 9a — Loose230 vs loose300, canon scope

Built `analysis_scratch/loose230_vs_loose300_table.py` and
`analysis_scratch/loose230_vs_loose300_freq_scatter.py`. Same canon scope
as `wind_effect_table_by_amp` (h100/low only, 1.3–1.6 Hz). Table has 24
cells; only 13 have data from both moorings (loose230 is sparse —
mostly fullwind, mostly n=1 per cell at A2/A3).

**Headline (canon, 13 cells)**:
- median Δ = K_t,230 − K_t,300 = **+0.036**
- mean Δ = +0.040
- range = −0.019 to +0.137
- pattern: at A1/A2 fullwind, loose230 transmits 5–14 % more; at A3 the
  difference vanishes.

Files:
```
analysis_scratch/loose230_vs_loose300_table.py
analysis_scratch/loose230_vs_loose300_freq_scatter.py
analysis_scratch/loose230_vs_loose300_table.csv
analysis_scratch/loose230_vs_loose300_freq_scatter_summary.csv
output/TABLES/ch05_loose230_vs_loose300_table.tex
output/FIGURES/ch05_loose230_vs_loose300_freq_A{1,2,3}.pdf
```

### 9b — Provenance check vs old findings doc

User asked: did I rely on the prior `mooring_comparison_findings.md` /
`mooring_comparison.py` or do my own analysis?

**Confirmed independent**: read the old `mooring_comparison.py` only for
filename/naming conventions; never executed it; never quoted from
`_findings.md`. My script queried `load_analysis_data` directly from the
current `waveprocessed/` cache.

Discovered the old findings doc reports very different numbers — at
1.3 Hz / A1 / fullwind the old doc has loose230 K_t = 0.915 (n=9) vs my
0.830 (n=4), and the sign of Δ even flipped. Two reasons:
1. **Scope** (dominant): old doc pooled loose230 from Mar 16-26 across
   ~9 folders / 4 probe configs; mine restricted to h100/low only (the
   thesis canon scope).
2. **Pipeline drift**: H&G window roll-out (2026-05-02) and window-mean
   baseline (2026-05-05) shifted K_t values modestly per cell.

The scope difference is the dominant driver. The old doc isn't
"better" or "worse" — it answers a different question (broader sample,
more confounds).

### 9c — Loose230 vs loose300, broad scope (per user request)

User decision: "keep the canon, and do a separate identical analysis with
the broad scope too. then it's easy to compare afterwards."

Built sibling scripts:
```
analysis_scratch/loose230_vs_loose300_table_broad.py
analysis_scratch/loose230_vs_loose300_freq_scatter_broad.py
output/TABLES/ch05_loose230_vs_loose300_table_broad.tex
output/FIGURES/ch05_loose230_vs_loose300_freq_broad_A{1,2,3}.pdf
```

Broad scope = all PROCESSED-* folders, filtered by Mooring tag in
`{below_90_loose230, below_90_loose300}`. loose230 picks up Mar 16-26
data across 4 probe configs; loose300 unchanged (only Mar 27 has it).

**Headline (broad, 24 cells, ALL cells comparable)**:
- median Δ = **−0.002** (essentially zero)
- mean Δ = +0.009
- range = −0.038 to +0.104
- pattern: most cells now sit near Δ ≈ 0; the canon-only "loose230 +0.04"
  signal collapses to noise level

| stat | canon | broad |
|---|---|---|
| n_cells with both moorings | 13 | 24 |
| median Δ | +0.036 | −0.002 |
| A1 / 1.4 / med Δ (the original outlier) | +0.137 | +0.051 |
| A1 / 1.6 / med Δ | +0.050 | −0.006 (sign flip) |

### 9d — Three readings, no resolution; observe and present

User offered three readings:
1. Canon-only signal was small-sample noise; the broad sample reveals the
   moorings are equivalent.
2. Broad scope adds 9 days of setup drift, washing out a real per-day
   mooring effect that canon caught (Mar 26 vs Mar 27, controlled).
3. Truth in between — small real effect (~+0.01 in K_t) inflated to +0.036
   by canon-only small-sample noise.

User's view (recorded for thesis context):
- Reading 1 has credence because the previous probe-config bias check
  (`probe_config_bias_likeforlike_all.pdf` from morning session) showed
  probe config doesn't biase K_t much, so broad-scope pooling is fair.
- Reading 2 is weakened by known confounds in the older runs:
  wind-speed inconsistencies, "I was lazy and didn't wait for tank
  settling between runs". So more-decent data trumphs canon precision.
- Reading 3 is also reasonable. Don't conclude — observe and present.

**Decision**: present both tables side-by-side in the thesis (or footnote
the divergence); don't pick a winner. The data can't discriminate.

---

## File index — afternoon additions

### Scripts
```
analysis_scratch/mooring_focus_at_1_3hz_table.py
analysis_scratch/damping_ka_per_volt_with_fit.py
analysis_scratch/loose230_vs_loose300_table.py
analysis_scratch/loose230_vs_loose300_table_broad.py
analysis_scratch/loose230_vs_loose300_freq_scatter.py
analysis_scratch/loose230_vs_loose300_freq_scatter_broad.py
```
Plus column-overhaul edits (no rename) to:
```
analysis_scratch/wind_effect_table.py
analysis_scratch/wind_effect_table_by_amp.py
```

### Output PDFs (under output/FIGURES/)
```
ch05_damping_ka_fit{,_A1,_A2,_A3}.pdf
ch05_loose230_vs_loose300_freq_A{1,2,3}.pdf            (canon)
ch05_loose230_vs_loose300_freq_broad_A{1,2,3}.pdf      (broad)
```

### Output tables (under output/TABLES/)
```
ch05_mooring_focus_at_1_3hz_table.tex
ch05_loose230_vs_loose300_table.tex                     (canon)
ch05_loose230_vs_loose300_table_broad.tex               (broad)
```
Plus regenerated:
```
ch05_wind_effect_table.tex
ch05_wind_effect_table_by_amp.tex
```

### Output stubs (under output/TEXFIGU/)
```
ch05_damping_ka_fit{,_A1,_A2,_A3}.tex
```

### main_save_figures.py changes
- New `[DELEG]` cell for `damping_ka_per_volt_with_fit` (§4a)
- New `[DELEG]` cell for `mooring_focus_at_1_3hz_table` (companion to §4b)
- 4 + 1 + 4 new entries in FIGURE_CAPTIONS / FIGURE_CAPTIONS_SHORT
- 2 new FIGURE INDEX lines (§4a, §4b table)
- Caption text updated for `ch05_mooring_focus_at_1_3hz_table` to match
  the new ratio columns

### This memo
```
analysis_scratch/session_2026-05-06b_tables_curves_loose_mooring.md
```

---

## Things explicitly **not** wired into main_save_figures.py

The `loose230_vs_loose300_*` family (both scopes — 4 scripts, 6 figures,
2 tables) is left as scratch. The mooring-slack finding is honest but
inconclusive (Reading 1/2/3 stand-off above), so it isn't ready for a
thesis spot. If it becomes a thesis section later, wiring is mechanical:
add 4 `_run_delegated_if_missing` calls + 8 caption entries + 2 FIGURE
INDEX lines.

---

## Carry-overs to the next session

1. **Caption prose**: 7 `TODO:` placeholders in FIGURE_CAPTIONS waiting
   for hand-written text — `ch05_mooring_focus_at_1_3hz_ka{,_A1,_A2,_A3}`,
   `ch05_mooring_focus_at_1_3hz_table`, `ch05_damping_ka_fit{,_A1,_A2,_A3}`.
   Short captions are filled in.
2. **Decide loose230 vs loose300 fate**: thesis section, footnote, or
   drop entirely. Both scopes (canon + broad) are documented and
   reproducible if revisited.
3. **The `_with_fit` script** is wired in but produces an extra combined
   PDF (`ch05_damping_ka_fit.pdf`) that has no fit overlay — kept for
   layout symmetry with `ch05_damping_ka.pdf`. If unused in the thesis
   it can be dropped from the wiring outputs list.
