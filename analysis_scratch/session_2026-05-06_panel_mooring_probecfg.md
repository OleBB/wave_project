# Session 2026-05-06 — Panel comparison + mooring effect + probe-config bias

Exploratory session. Started from "is 0.65 Hz data in this plot?" → cascaded
into reverse-vs-full panel comparison → mooring effect across panels → probe
config as a confound inside one mooring → final thesis-style 3-panel figure.

Followed the observation-vs-inference rule (CLAUDE.md §20) throughout: every
"X is happening because Y" claim is flagged as a *candidate explanation
(hypothesis)*, not a finding.

> ⚠ **Mid-session correction (Thread 2 → 2c)**: the standalone full-vs-reverse
> comparison in 2a did not restrict by mooring. Reverse panel only exists on
> `above_50`; the full-panel side pooled all moorings, including below_90
> which transmits more. The "full panel benefits much more from wind than
> reverse" inference (Δ_wind = +0.082 vs +0.027) was an artifact of that mix.
> When restricted to `Mooring == above_50` on both sides, the corrected Δ_wind
> for full panel at 1.30 Hz is **+0.021** — comparable to reverse's +0.027.
> Use Thread 2c numbers, not 2a. The 2a files are kept on disk marked
> `SUPERSEDED` for traceability.

---

## Thread 1 — Does the all-data scatter contain 0.65 Hz?

**Question**: Is `output/FIGURES/ch05_damping_all_data_scatter.pdf` showing
0.65 Hz?

**Answer (verified)**: No. Filter requires `PanelCondition == "full"`. The
0.65 Hz dataset (41 rows from Nov 2025) is split 25 reverse / 16 no — zero
full-panel runs at 0.65 Hz. So 0.65 Hz drops out, not because of a frequency
cutoff, but because the full panel was never run at 0.65 Hz.

**Variant scatter plots** (panel filter relaxed to include the missing data):
- **V1**: `PanelCondition ∈ {full, reverse}`, all hardware → 401 rows, 19 at
  0.65 Hz. Reverse-panel data only at 0.65 + 1.30 Hz, sparse.
- **V2**: same as V1 but exclude canon (`cond4_h100_low`) → 299 rows, all
  hollow markers (earlier hardware only).

Files:
```
analysis_scratch/all_data_damping_scatter_variants.py
analysis_scratch/all_data_damping_scatter_v1_full_reverse.pdf
analysis_scratch/all_data_damping_scatter_v2_full_reverse_no_canon.pdf
analysis_scratch/all_data_damping_scatter_variants_summary.csv
output/FIGURES/ch05_damping_all_data_scatter_v1_full_reverse.pdf
output/FIGURES/ch05_damping_all_data_scatter_v2_full_reverse_no_canon.pdf
```

---

## Thread 2 — Reverse vs full panel comparison

Reverse panel data only exists at **two frequencies in the full record**:
0.65 Hz (19 runs) and 1.30 Hz (23 runs). So "comparing reverse to full" is
necessarily a head-to-head at those two cells.

### 2a. Full vs reverse at 1.30 Hz — INITIAL (mooring-mixed, SUPERSEDED)

> ⚠ **SUPERSEDED by Thread 2c (2026-05-06).** This pass did not restrict by
> mooring. Reverse panel only physically exists on `above_50` mooring, but
> the full-panel side pooled all moorings (above_50 + below_90_loose230 +
> below_90_loose300). After Thread 4 showed below_90 moorings transmit
> 0.05–0.10 more K_t than above_50, this comparison was re-run with
> `Mooring == above_50` for both sides. The headline below is retained for
> historical context — DO NOT cite the +0.082 number.

| | nowind | fullwind | Δ wind |
|---|---|---|---|
| full *(mixed moorings)* | 0.688 | 0.770 | +0.082 |
| reverse | 0.665 | 0.692 | +0.027 |

Original (now-retracted) inference: "wind boosts full panel transmission
much more than reverse". This effect did not survive mooring restriction
(see 2c).

Files (superseded — kept on disk for traceability):
```
analysis_scratch/full_vs_reverse_comparison.py            (SUPERSEDED)
analysis_scratch/full_vs_reverse_comparison.pdf           (SUPERSEDED)
analysis_scratch/full_vs_reverse_comparison_summary.csv   (SUPERSEDED)
output/FIGURES/ch05_full_vs_reverse_comparison.pdf        (SUPERSEDED)
```

### 2c. Full vs reverse at 1.30 Hz — CORRECTED (above_50 only)

After Thread 4 quantified the mooring effect, 2a was re-run with
`Mooring == above_50` on both sides — the only mooring where reverse
panel exists, so it forces like-for-like at the mooring axis.

| | nowind | fullwind | Δ wind |
|---|---|---|---|
| full *(above_50 only)* | **0.661** | **0.682** | **+0.021** |
| reverse | 0.665 | 0.692 | +0.027 |

**Corrected observation**: at 1.30 Hz on above_50 mooring, full and
reverse give nearly the same K_t and a similarly small wind boost
(+0.02 to +0.03). The "full panel benefits much more from wind"
inference from 2a was an artifact of mixing in below_90 full-panel
data, which carries a much larger wind boost (Thread 4 Row A: above_50
fullwind 0.612 → below_90 fullwind 0.766 at 1.4 Hz A2).

**Caveat baked in**: this corrected comparison still pools across
amplitude tiers (matching 2a's structure). For the cleanest
like-for-like at (panel, mooring, amp, wind, freq), use the Row B
panels in `transmission_mooring_panel_*.pdf` from Thread 4 — they
facet by amp. The amp-pooled deltas above are slightly different from
Row B's per-amp deltas because K_t varies with amp.

Files:
```
analysis_scratch/full_vs_reverse_comparison_above50.py
analysis_scratch/full_vs_reverse_comparison_above50.pdf
analysis_scratch/full_vs_reverse_comparison_above50_summary.csv
output/FIGURES/ch05_full_vs_reverse_comparison_above50.pdf
```

### 2b. 0.65 Hz reverse vs 0.70 Hz full (nearest-neighbour comparison)

Δf = 0.05 Hz, Δλ ≈ 0.30 m. Same panel can't be compared at the same freq;
this is the closest indirect comparison.

| | nowind | fullwind | Δ wind |
|---|---|---|---|
| 0.70 Hz, full | 1.007 (n=2) | 1.017 (n=6) | +0.011 |
| 0.65 Hz, reverse | 0.965 (n=12) | 0.954 (n=7) | −0.011 |

**Observations**:
- Both K_t are within ±5 % of unity — the panel barely attenuates at
  λ ≈ 3 m.
- Full panel gives K_t > 1 (apparent amplification).
- Wind effects are sign-flipped between the two cells (+0.011 vs −0.011)
  but smaller than within-cell σ (~0.01) → consistent with noise.

*Candidate explanation (hypothesis, user)*: long waves are too long /
energetic to "see" the panel; wind just adds noise where the geometric
signal is ≈ 0. Supported but not proven by the data — see Thread 3.

> ⚠ **Same mooring-mix caveat applies to 2b as flagged in 2a.** The
> 0.70 Hz full-panel data exists on both above_50 and below_90_loose230,
> so this comparison also mixes moorings on the full-panel side. The
> n_no=2 / n_full=6 cells are too small to redo per-mooring with any
> confidence; flagging rather than re-running. The K_t > 1 oddity at
> 0.70 Hz full panel is interesting but should be revisited per-mooring
> if it ever matters.

Files:
```
analysis_scratch/freq_065_vs_070_comparison.py
analysis_scratch/freq_065_vs_070_comparison.pdf
analysis_scratch/freq_065_vs_070_comparison_summary.csv
output/FIGURES/ch05_freq_065_vs_070_comparison.pdf
```

---

## Thread 3 — σ ratio of K_t per frequency

Generalises the per-cell σ values from Thread 2: for each (freq, panel) cell
where both nowind and fullwind have ≥3 runs, compute
σ_fullwind / σ_nowind.

**Observation**:

| Band | n cells | median σ ratio |
|---|---|---|
| f < 1.0 Hz | 3 | **1.30** |
| f ≥ 1.0 Hz | 9 | **0.90** |

At low freq wind inflates within-cell spread. At higher freq it doesn't
(may even tighten). Matches the user's "wind = noise where panel is
inactive" framing.

**Cell-level oddities flagged**:
- 0.80 Hz full panel: σ_ratio = 8.0 — driven by σ_nowind = 0.0018 from
  n=4 runs that happened to land near-identical. Not a real σ estimate.
- 1.10 Hz full panel: σ_ratio = 3.3 — moderate sample (n=6/9).
- 1.30 Hz reverse: σ_ratio = 0.33 — wind dramatically tightens; small
  fullwind sample (n=6) so wide CI.

Files:
```
analysis_scratch/std_ratio_vs_freq.py
analysis_scratch/std_ratio_vs_freq.pdf
analysis_scratch/std_ratio_vs_freq_summary.csv
output/FIGURES/ch05_std_ratio_vs_freq.pdf
```

---

## Thread 4 — Transmission across mooring × panel

Original ask: extended IN-amplitude comparison across mooring × panel
conditions. **User pivot mid-script**: switch y-axis from IN amplitude to
K_t (transmission ratio).

### 4a. Combined wind/mooring/panel matrix (2×3 facets)

Two complementary halves in one figure (each held the other axis constant
because no mooring covers all 3 panels and no panel covers all 4 moorings):
- **Row A — mooring effect** (PanelCondition = full): 3 moorings.
- **Row B — panel effect** (Mooring = above_50): 3 panels.
- **Cols** — amplitude tier (A1, A2, A3).

Two visual iterations:
1. First pass: mooring/panel as colour, wind as marker.
2. After user feedback: wind=colour (red/blue thesis convention via
   `WIND_COLOR_MAP`), mooring/panel as marker shape (`PANEL_MARKERS`
   for row B; `D, v, P` from `MARKERS` for row A).

Files:
```
analysis_scratch/in_amp_mooring_panel_compare.py   # script (name kept; content is K_t)
analysis_scratch/transmission_mooring_panel_compare.pdf
analysis_scratch/transmission_mooring_panel_compare_summary.csv
output/FIGURES/ch05_transmission_mooring_panel_compare.pdf
```

### 4b. Wind-split versions (one figure per wind condition)

User asked for two new figures, splitting by wind, with above_50 vs the
two below_90 moorings being the primary visual contrast.

**Encoding inside each figure**:
- Mooring: above_50 → blue ◆ (cool), below_90_loose230 → orange ▼,
  below_90_loose300 → dark red ✚. above-vs-below = cool-vs-warm hue
  (primary). 230 vs 300 = shade in warm family (secondary).
- Panel (Row B): green / purple / grey with `PANEL_MARKERS` shapes —
  palette deliberately distinct from Row A.
- Wind preserved as figure-title colour (red title for med vind, blue for
  uten) so the wind=colour convention survives at figure level.

After mooring nomenclature was clarified by the user, labels were updated
to be physically meaningful and a small probe-config-composition footer
was added (see Thread 5 for context).

**Observations** (Row A, A2, full panel):
- Below-water moorings transmit more energy than above_50 by ~0.05–0.10
  K_t under nowind, growing to ~0.18–0.21 under fullwind at 1.6 Hz.
- *Candidate explanation*: above_50 mooring physically restricts panel
  motion → more damping. Not tested independently of probe-config
  variation (see Thread 5).

**Observations** (Row B, A2, above_50, 1.30 Hz):
- full panel: K_t = 0.723 (nowind) / 0.712 (fullwind).
- reverse panel: K_t = 0.683 / 0.685.
- no panel:    K_t = 0.983 / 1.023 (sanity check passes — no panel ≈ 1).
- Reverse panel is *slightly more* attenuating than full at 1.30 Hz
  (Δ ≈ 0.04). Surprising, but holds under both winds.

Files:
```
analysis_scratch/transmission_mooring_panel_by_wind.py
analysis_scratch/transmission_mooring_panel_nowind.pdf
analysis_scratch/transmission_mooring_panel_fullwind.pdf
output/FIGURES/ch05_transmission_mooring_panel_nowind.pdf
output/FIGURES/ch05_transmission_mooring_panel_fullwind.pdf
```

---

## Thread 5 — Probe-config bias inside below_90_loose230

User-provided context (verified against meta):

| Mooring | n | probe configs |
|---|---|---|
| `above_50` | 308 | 1 (h272/high) — clean |
| `below_90_loose230` | 208 | 4 (h100/high 137; h100/low 46; h136/high 7; h272/high 18) — mixed |
| `below_90_loose300` | 86 | 1 (h100/low — canon) — clean |

User assumption stated explicitly: *probe config is a measurement-uncertainty
axis, not a physics axis*. The Thread 4 mooring comparison pools across
configs implicitly. We tested whether that pooling is defensible.

### 5a. Raw cut — pool amp, group by (cfg, freq, wind)

| stat | inter-config spread (max−min mean K_t) |
|---|---|
| median | **0.130** |
| max | 0.205 |
| reference: within-cell σ at 1.3–1.6 Hz | ~0.07 |

Looks bad on its face — 2× within-cell σ. But strongly confounded:
- Cells pool amp (different configs may have been run at different amp mixes).
- Configs are aliased with **dates** (Mar 19 → 26).
- Most cells aren't sampled by all configs → "spread" partly compares
  different cells, not different configs at the same cell.

Files:
```
analysis_scratch/probe_config_bias_within_mooring.py
analysis_scratch/probe_config_bias_within_mooring.pdf
analysis_scratch/probe_config_bias_within_mooring_summary.csv
```

### 5b. Like-for-like — restrict to cells where ≥2 configs sampled

Three threshold iterations, each tightening the n-per-config bar:

| MIN_RUNS_PER_CFG | cells surviving | median Δ_cfg | max | 90th % |
|---|---|---|---|---|
| ≥3 | 2 / 18 | 0.032 | 0.062 | — |
| ≥2 | 3 / 22 | 0.024 | 0.062 | 0.055 |
| ≥1 (all data) | 22 / 49 | **0.030** | 0.149 | 0.110 |

**Within-(cell, cfg) σ ≈ 0.025** across all three views. So the median
inter-config spread is **comparable to or smaller than the within-cell σ**
under the like-for-like restriction.

The ≥3-runs cell with the largest spread (0.062 at 1.3 Hz fullwind A1) is
`100/high` vs `100/low` — same height, different range mode. *Candidate
explanation (hypothesis)*: range mode might genuinely affect ULS dynamic
range under fullwind. Not testable with 4+4 runs.

The "all data" cut's tail is dominated by **small-n + fullwind-noise**
combinations: most large-spread cells have 1 run per config, so the
"spread" is three single observations from the noisy fullwind distribution,
not a comparison of means.

Files:
```
analysis_scratch/probe_config_bias_likeforlike.py            # parametrised script
analysis_scratch/probe_config_bias_likeforlike.pdf           # n≥3 (initial)
analysis_scratch/probe_config_bias_likeforlike_summary.csv
analysis_scratch/probe_config_bias_likeforlike_cells.csv
analysis_scratch/probe_config_bias_likeforlike_n2plus.pdf    # n≥2
analysis_scratch/probe_config_bias_likeforlike_n2plus_summary.csv
analysis_scratch/probe_config_bias_likeforlike_n2plus_cells.csv
analysis_scratch/probe_config_bias_likeforlike_all.pdf       # all (n≥1)
analysis_scratch/probe_config_bias_likeforlike_all_summary.csv
analysis_scratch/probe_config_bias_likeforlike_all_cells.csv
```

### 5c. User visual observation on the all-cells plot

User: *"I see almost all orange squares (100/low) above the others. 100/low
is canon. This implies our supposed increased precision from using canon
means LOWER TRANSMISSION."*

Observed but not tested. The pattern is real-looking visually and is most
prominent in fullwind cells. With the per-cell sample sizes available, we
can't formally test "canon is biased high" vs "small-sample variance".

### 5d. Strategy discussion — table or visual?

Decided: **plot only**, no formal table.

Reasoning:
- Numbers in a table read as claims; the data doesn't support claims at the
  per-cell level.
- Configs are aliased with dates → ANOVA-aware reader would dismiss a table.
- The visual carries the soft pattern honestly without forcing magnitude.
- Matches the user's own stated philosophy and the project's
  observation-vs-inference rule.

Suggested caption text for the figure (if it's promoted to thesis):
> Lines per probe-konfigurasjon innenfor `below_90_loose230` / full panel.
> Punktene er rå middel per (freq, vind, amp), uten n-grense. På celler
> der ≥2 konfigurasjoner er samplet, er median spredning mellom
> konfigurasjoner ~0.03 i K_t — sammenlignbart med innen-celle σ ved
> 1.3–1.6 Hz (~0.07). Sample per (celle, cfg) er typisk 1–3 kjøringer;
> konfigurasjon er sammenfilt med målingsdato. Den synlige tendensen til
> at canon (100/low) ligger over de andre er observert, ikke testet.

### 5e. Final thesis-style 3-panel figure

Mirrored `plot_damping_freq` visual language:
- 3 separate subfigures (one per amplitude tier).
- x = k (with paddle freq on top axis, full dispersion via `freq_to_k`).
- y = K_t, shared y-range across subfigs for stacked reading.
- horizontal `$K_t$` y-label above leftmost tick.
- Marker per amp tier (○ A1, □ A2, △ A3) — thesis convention.

**Two visual iterations**:
1. Initial palette: blue / orange / green / red. User flagged conflict
   with `WIND_COLOR_MAP` (red = fullwind, blue = nowind).
2. Revised palette: avoiding red/blue entirely. 136/high dropped (only
   3 runs in slice). Final colours: 100/high → green (#2CA02C),
   **100/low → purple #7B3F99 (canon)**, 272/high → gold-brown (#A0721B).
   Wind → linestyle (solid no, dashed full).

Files:
```
analysis_scratch/probe_config_bias_thesis_style.py
analysis_scratch/probe_config_bias_thesis_A1.pdf
analysis_scratch/probe_config_bias_thesis_A2.pdf
analysis_scratch/probe_config_bias_thesis_A3.pdf
output/FIGURES/ch05_probe_config_bias_A1.pdf
output/FIGURES/ch05_probe_config_bias_A2.pdf
output/FIGURES/ch05_probe_config_bias_A3.pdf
```

---

## Assumptions baked into this session

1. **Pooling across hardware (canon + earlier) is defensible** for cross-
   condition comparisons in Threads 2–4. Carried over from existing all-data
   scatter convention. The Thread 5 like-for-like check supports this *for
   probe configs within `below_90_loose230` only*; not generalised.

2. **`PanelCondition` is the geometry variable of interest, not the
   wavemaker amplitude or wind**. All filters preserve panel as a stratum,
   never collapse over it.

3. **OUT/IN (FFT) is the only damping metric used** (CLAUDE.md §16 rule).
   Time-domain amplitude excluded — it includes wind waves and is
   meaningless under fullwind.

4. **Wind has only two trustworthy levels in this analysis**: `no` and
   `full`. `lowest` exists in some cells but was excluded throughout for
   visual simplicity (and because the central thesis question is "wind
   vs no wind").

5. **Thresholds applied throughout**:
   - `quality_flag == "ok"`
   - `OUT/IN (FFT) ∈ [0.1, 2.0]` (extreme outlier guard from
     all_data_damping_scatter convention)
   - `WaveFrequencyInput < 2.0 Hz` (drops the lone 2.0 Hz point — same
     reason as all_data_damping_scatter)

6. **`above_200` mooring dropped** (only 7 rows total, all at 1.3 Hz) —
   too sparse to use anywhere.

7. **`136/high` probe config dropped from the final thesis-style figure**
   only — kept in the diagnostic plots. Reason: 3 runs in the
   below_90_loose230/full slice is too sparse for a usable line.

---

## What we did *not* do (open questions for follow-up)

- **Mooring effect across panels other than full** is not testable —
  reverse and no panels exist only on `above_50` mooring.
- **Probe-config bias was tested only inside `below_90_loose230`**.
  `above_50` and `below_90_loose300` each have a single config so are
  safe-by-construction; nothing was tested for them.
- **Date confound was flagged but not removed**. To do so would need a
  cell where the same config was run on multiple dates — that subset
  isn't substantial enough in the current cache.
- **The "100/low above others" observation was not formally tested**. The
  data per cell is too sparse for a credible test; intentionally left as
  a visual pattern with caveats.
- **Headline reverse-vs-full comparison** (Thread 2a) covers only 1.30
  Hz. To extend would require running the wave tank with reverse panel at
  more frequencies — out of scope here.

---

## File index — all artefacts created this session

### Scripts
```
analysis_scratch/all_data_damping_scatter_variants.py
analysis_scratch/full_vs_reverse_comparison.py            (SUPERSEDED — see 2c)
analysis_scratch/full_vs_reverse_comparison_above50.py    (corrected)
analysis_scratch/freq_065_vs_070_comparison.py
analysis_scratch/std_ratio_vs_freq.py
analysis_scratch/in_amp_mooring_panel_compare.py
analysis_scratch/transmission_mooring_panel_by_wind.py
analysis_scratch/probe_config_bias_within_mooring.py
analysis_scratch/probe_config_bias_likeforlike.py
analysis_scratch/probe_config_bias_thesis_style.py
```

### Scratch PDFs (visual previews)
```
analysis_scratch/all_data_damping_scatter_v1_full_reverse.pdf
analysis_scratch/all_data_damping_scatter_v2_full_reverse_no_canon.pdf
analysis_scratch/full_vs_reverse_comparison.pdf            (SUPERSEDED)
analysis_scratch/full_vs_reverse_comparison_above50.pdf    (corrected)
analysis_scratch/freq_065_vs_070_comparison.pdf
analysis_scratch/std_ratio_vs_freq.pdf
analysis_scratch/transmission_mooring_panel_compare.pdf
analysis_scratch/transmission_mooring_panel_nowind.pdf
analysis_scratch/transmission_mooring_panel_fullwind.pdf
analysis_scratch/probe_config_bias_within_mooring.pdf
analysis_scratch/probe_config_bias_likeforlike.pdf            (n≥3)
analysis_scratch/probe_config_bias_likeforlike_n2plus.pdf     (n≥2)
analysis_scratch/probe_config_bias_likeforlike_all.pdf        (n≥1, all)
analysis_scratch/probe_config_bias_thesis_A1.pdf
analysis_scratch/probe_config_bias_thesis_A2.pdf
analysis_scratch/probe_config_bias_thesis_A3.pdf
```

### Scratch CSVs (numeric data)
```
analysis_scratch/all_data_damping_scatter_variants_summary.csv
analysis_scratch/full_vs_reverse_comparison_summary.csv            (SUPERSEDED)
analysis_scratch/full_vs_reverse_comparison_above50_summary.csv    (corrected)
analysis_scratch/freq_065_vs_070_comparison_summary.csv
analysis_scratch/std_ratio_vs_freq_summary.csv
analysis_scratch/transmission_mooring_panel_compare_summary.csv
analysis_scratch/probe_config_bias_within_mooring_summary.csv
analysis_scratch/probe_config_bias_likeforlike_summary.csv
analysis_scratch/probe_config_bias_likeforlike_cells.csv
analysis_scratch/probe_config_bias_likeforlike_n2plus_summary.csv
analysis_scratch/probe_config_bias_likeforlike_n2plus_cells.csv
analysis_scratch/probe_config_bias_likeforlike_all_summary.csv
analysis_scratch/probe_config_bias_likeforlike_all_cells.csv
```

### Output PDFs (under `output/FIGURES/` — promoted but not yet wired into thesis)
```
output/FIGURES/ch05_damping_all_data_scatter_v1_full_reverse.pdf
output/FIGURES/ch05_damping_all_data_scatter_v2_full_reverse_no_canon.pdf
output/FIGURES/ch05_full_vs_reverse_comparison.pdf            (SUPERSEDED)
output/FIGURES/ch05_full_vs_reverse_comparison_above50.pdf    (corrected)
output/FIGURES/ch05_freq_065_vs_070_comparison.pdf
output/FIGURES/ch05_std_ratio_vs_freq.pdf
output/FIGURES/ch05_transmission_mooring_panel_compare.pdf
output/FIGURES/ch05_transmission_mooring_panel_nowind.pdf
output/FIGURES/ch05_transmission_mooring_panel_fullwind.pdf
output/FIGURES/ch05_probe_config_bias_A1.pdf
output/FIGURES/ch05_probe_config_bias_A2.pdf
output/FIGURES/ch05_probe_config_bias_A3.pdf
```

### This memo
```
analysis_scratch/session_2026-05-06_panel_mooring_probecfg.md
```

---

## Naming-debt note

`analysis_scratch/in_amp_mooring_panel_compare.py` was written intending IN
amplitude as the y-axis, then mid-script the user pivoted to K_t. The script
content is now K_t but the filename still says `in_amp_…`. Outputs went to
`transmission_…` paths (correct). If this script is ever revived, rename it
or fold the K_t logic into a `transmission_mooring_panel_compare.py` and
delete the old name.
