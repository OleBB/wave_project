---
name: handoff_2026-05-09_probe_agreement_ka_regime
description: Handoff for next agent — two coexisting probe-agreement tables (Variant A simple + ka span column, Variant B regime-stratified). Document open decisions and where each version lives.
type: project
---
# Handoff — probe-agreement tables: ka column (Variant A) + regime stratification (Variant B)

Context: the user spotted that the simple probe-agreement table tells a
mixed story (low-ka wind-dominated regime vs high-ka wave-dominated
regime where the far probe clips peaks). They wanted to TRY both
"add a ka column" (small change) and "stratify by ka regime" (new
table) and decide later which fits the thesis prose.

Both implementations now ship side-by-side. This note documents what
landed, what's open, and what the next agent should pick up.

## TL;DR

- **Variant A** = the existing simple table, NOW with a `ka spenn`
  column showing the cell's ka range. Same shape (4 freqs × 2 wind
  blocks), one extra column. Reader can see at-a-glance which rows are
  wind-dominated (small ka) vs wave-dominated (large ka).
- **Variant B** = a NEW small companion table that pools runs across
  (amp × freq) per (regime × wind), where regime is decided per-run by
  `ka < 0.15` (wind-dominated) vs `ka ≥ 0.15` (wave-dominated). 4 rows
  total, with skewness and empirical 90 % interval per cell.
- Both tables are wired through main_save_tables.py. Both .tex files
  exist on disk and compile.
- **Captions are user-owned** per the durable rule established
  2026-05-09 — captions in TABLE_CAPTIONS for both are deliberately
  short / empty; suggested longer wording lives ONLY in the IMMUTABLE
  TEX STUB sections (so I don't re-violate the rule). The user picks
  the final caption text.

## Files touched

| File | Role | Change |
|---|---|---|
| `analysis_scratch/parallel_probe_psd_agreement_simple.py` | data script | (1) Builds run_id → ka map from canon meta. (2) Threads ka_per_run through `per_freq_simple`, which now emits `ka_min`, `ka_max`, plus internal `_per_run_ka` aligned with `_per_run`. (3) Computes the regime-stratified pooled stats (Variant B) and writes `ch04_parallel_probe_psd_agreement_by_ka_regime.{csv,meta.json}` to `output/TABLES/data/`. |
| `main_save_tables.py` | render + captions | (1) Simple-table cell adds `ka_span` column (5 cols total, `column_spec = "ccccc"`). (2) New render cell at line ~400+ for the regime table — 6 cols, two row-groups by regime. (3) New entries in `TABLE_CAPTIONS` and `TABLE_CAPTIONS_SHORT` for `ch04_parallel_probe_psd_agreement_by_ka_regime`. |
| `output/TABLES/ch04_parallel_probe_psd_agreement_simple.tex` | rendered output | Variant A. |
| `output/TABLES/ch04_parallel_probe_psd_agreement_by_ka_regime.tex` | rendered output | Variant B (NEW). |

## Variant A — what's in the rendered table

Columns: `f [Hz]` · `N` · `ka spenn` · `Δ̄ (fjern−nær) [%]` · `σ_Δ [%]`

Two row-groups (`\itshape Uten vind` / `\itshape Full vind`), 4 rows each.
Each row's N is the count of paddle-at-this-f runs (paddle-freq filter
landed earlier the same day; not new in this turn).

Rendered numbers (current state):

| f | N | ka spenn | Δ̄ % | σ_Δ % |
|---|---|---|---|---|
| **Uten vind** | | | | |
| 1.30 | 13 | [0.05, 0.15] | −0.98 | 1.06 |
| 1.40 | 6 | [0.06, 0.18] | +1.87 | 1.04 |
| 1.50 | 6 | [0.07, 0.19] | −2.55 | 1.89 |
| 1.60 | 7 | [0.07, 0.22] | +1.10 | 2.21 |
| **Full vind** | | | | |
| 1.30 | 17 | [0.05, 0.16] | +0.13 | 4.81 |
| 1.40 | 10 | [0.06, 0.18] | +0.52 | 6.20 |
| 1.50 | 10 | [0.07, 0.22] | −4.58 | 12.98 |
| 1.60 | 11 | [0.08, 0.25] | −9.26 | 10.11 |

## Variant B — what's in the rendered table

Columns: `vind` · `N` · `Δ̄ [%]` · `σ_Δ [%]` · `γ_1` · `[P_5, P_95] [%]`

Row-groups by regime (`\itshape Vind-dominert (ka < 0,15)` /
`\itshape Bølge-dominert (ka ≥ 0,15)`), 2 rows each (Uten / Full vind).

Rendered numbers (current state):

| | regime | wind | N | Δ̄ % | σ_Δ % | γ_1 | [P5, P95] |
|---|---|---|---|---|---|---|---|
| 1 | Vind-dominert | Uten | 24 | −0.32 | 2.19 | −0.04 | [−4.3, +3.3] |
| 2 | Vind-dominert | Full | 29 | −1.09 | 9.55 | −1.22 | [−19.6, +10.7] |
| 3 | Bølge-dominert | Uten | 8 | −0.19 | 2.20 | −1.09 | [−3.6, +1.7] |
| 4 | Bølge-dominert | Full | 19 | −5.72 | 8.16 | −0.96 | [−20.7, +2.5] |

### Reading

- **Vind-dominert × Uten vind** is the cleanest probe-agreement cell:
  γ_1 = −0.04 (essentially symmetric), Δ̄ ≈ 0, σ ~ 2 %. As expected.
- **Vind-dominert × Full vind**: scatter blows up (σ = 9.55 %), and the
  distribution becomes heavily left-skewed (γ_1 = −1.22). The wide
  P_95 = +10.7 alongside a deep P_5 = −19.6 quantifies the asymmetry.
- **Bølge-dominert × Full vind**: the *systematic* bias is bigger (Δ̄ =
  −5.7 %) and the skew is still strong (γ_1 = −0.96). Together with
  Vind-dominert × Full this forms the "far probe clips peaks" story.
- **Bølge-dominert × Uten vind** is small N (8) — the γ_1 = −1.09
  estimate is noisy. Don't over-interpret a single number; flag in
  prose if you cite it.

## Open decisions for the user / next agent

1. **Which variant to use in the thesis chapter, or both.**
   - Variant A is a drop-in upgrade of the existing simple table. Same
     row layout, one extra column. Lowest disruption — your existing
     prose around the table mostly still applies.
   - Variant B is a new cell that tells the regime story directly. No
     freq breakdown — replaces freq granularity with regime granularity.
   - Recommended: **A inline in the methodology paragraph; B in an
     appendix or a follow-up paragraph if the regime story matters
     enough to call out**. They're not redundant — A shows the freq
     breakdown with regime hint, B shows the regime contrast pooled.

2. **Caption text.** Both captions are deliberately short / empty
   (`TABLE_CAPTIONS` entries: `r"Samsvar mellom parallelle prober. ..."`
   for A; `""` for B). Longer suggested wording lives in each table's
   IMMUTABLE TEX STUB under the "Caption suggestion" or "Reading"
   section. User owns the final caption per the 2026-05-09 durable
   rule. **Do NOT have the agent expand TABLE_CAPTIONS** — only flag
   suggestions in the stub.

3. **ka regime threshold.** Currently `KA_REGIME_THRESHOLD = 0.15`
   (constant at the top of `parallel_probe_psd_agreement_simple.py`).
   This split corresponds roughly to the user's verbal description of
   the regimes (1.3 Hz any amp + 1.4–1.5 × A1 below; 1.5–1.6 × A2/A3
   above). Tunable knob. Try 0.12 if you want more runs in the
   wave-dominated bin; try 0.20 to make the regimes more contrastful
   (smaller wave-dominated bin, but more clearly above the threshold).

4. **The Bølge-dominert × Uten vind cell has N = 8** — small. The γ_1 =
   −1.09 estimate is unreliable at that N. Two ways to handle:
   - Live with it and note the N=8 caveat in prose.
   - Drop the row entirely with a footnote.
   - Pool across wind for that regime (one number per regime instead of
     four total). Loses the wind contrast within regime but gives more
     reliable per-regime stats.

5. **Far-probe-clips-peaks claim is hypothesis, not verified.** The
   data is consistent with this mechanism (negative Δ̄ + negative skew
   under high-ka full-wind), but no direct verification (e.g., looking
   at the per-run time series for the worst Δ runs to confirm clipping
   behaviour). If you want to make the claim hard, add a small
   diagnostic figure showing one or two example time-series excerpts
   with the clipped peaks visible. Optional.

## Where the data flows

```
parallel_probe_psd_agreement_simple.py main()
  ├─ _build_run_ka_map()                   ← loads canon meta, builds run_id → ka
  ├─ for wind in (nowind, fullwind):
  │    └─ _rows_for_wind(wind, run_ka_map)
  │         └─ for fh in TARGET_FREQS:
  │              ├─ _load_psd_data_from_project(target_freqs=[fh], wind_label=...)
  │              ├─ harmonize_grid(...)
  │              └─ per_freq_simple(..., ka_per_run=...)   ← computes Δ_i, ka_min/max
  │                   returns row dict with `_per_run` (Δ_i array)
  │                                          `_per_run_ka` (ka_i array)
  │                                          ka_min, ka_max, diff_pct, std_pct
  │
  ├─ rows : list of 8 row dicts (4 freqs × 2 winds)
  │
  ├─ Variant A CSV write:
  │    csv_rows = strip _-prefix keys from rows
  │    pd.DataFrame(csv_rows).to_csv(RENDER_CSV)
  │       → output/TABLES/data/ch04_parallel_probe_psd_agreement_simple.csv
  │
  ├─ Variant B aggregation:
  │    For each wind: concat all _per_run + _per_run_ka across freqs
  │    Split by ka < / ≥ KA_REGIME_THRESHOLD → wind_dom, wave_dom subsets
  │    Compute n, mean, std, skew, p5, p95 per (regime, wind) → 4 rows
  │    Write CSV + meta.json
  │       → output/TABLES/data/ch04_parallel_probe_psd_agreement_by_ka_regime.csv
  │
  └─ Both meta.json files written with caption-label, sections, etc.

main_save_tables.py (render-only, on import)
  ├─ TABLE_CAPTIONS / TABLE_CAPTIONS_SHORT defined; persisted to JSON.
  ├─ Cell for ch04_parallel_probe_psd_agreement_simple → renders Variant A .tex
  └─ Cell for ch04_parallel_probe_psd_agreement_by_ka_regime → renders Variant B .tex
```

## Quick smoke test

```bash
# Regenerate both CSVs:
python analysis_scratch/parallel_probe_psd_agreement_simple.py

# Render both .tex files:
python main_save_tables.py

# Sanity check:
ls output/TABLES/ch04_parallel_probe_psd_agreement_*.tex
ls output/TABLES/data/ch04_parallel_probe_psd_agreement_*
```

Expected output:
- `output/TABLES/ch04_parallel_probe_psd_agreement_simple.tex` (Variant A)
- `output/TABLES/ch04_parallel_probe_psd_agreement_by_ka_regime.tex` (Variant B)
- `.tex` of three other related tables (the merged ones, untouched).

## Implication for the thesis K_t story (carried over from prior turn)

The far-probe-clips-peaks observation has a direct implication for the
canonical IN amplitude (mean of two probes):

> "For runs at $f \ge 1{,}5$ Hz under full wind, the canonical IN
> amplitude is a conservative (low) estimate because the far-probe
> (9373/340) intermittently clips wave peaks. Reported $K_t$ in this
> regime is therefore an upper bound on the true transmission."

That sentence is worth working into the methodology chapter alongside
either Variant A or Variant B (or both). It converts a noise observation
into a methodological caveat — earns reviewer trust.

## Loose ends

- The **Bland-Altman figure** (`fig:ch04_parallel_probe_agreement_bland_altman`,
  rendered elsewhere) is the visual companion. Verify it visually shows
  the asymmetric dot pattern under fullwind that the table's negative
  skewness implies. If the figure shows symmetric clouds and the table
  shows γ_1 = −0.91, something is off.
- **N = 8 cell** in Variant B (Bølge-dominert × Uten vind) — see open
  decision #4 above.
- **Caption text** is the user's call. The TEX STUB has suggestions —
  open the `.tex` files and look at the `% — Caption suggestion ` /
  `% — Reading ` sections.
- The 4 individual per-(range × wind) tables were dropped earlier
  (replaced by the merged tables `_lowrange_merged` / `_highrange_merged`).
  The simple table + Variant B are the in-text content; merged tables
  are the appendix companions.
