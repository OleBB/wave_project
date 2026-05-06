Let me check the actual ka spread per (amp, freq) to see how bad the collision is.

You're right — the fit ignores frequency, and at A1 especially there's real ka overlap between adjacent frequencies. Per-(amp, freq) ka ranges:

| amp | f=1.3 | f=1.4 | f=1.5 | f=1.6 |
|---|---|---|---|---|
| **A1** | 0.053–0.099 | 0.066–0.119 | 0.073–0.120 | 0.084–0.138 |
| A2 | 0.104–0.139 | 0.127–0.170 | 0.151–0.187 | 0.168–0.211 |
| A3 | 0.156–0.186 | 0.186–0.214 | 0.209–0.255 | 0.236–0.280 |

At A1, ka ≈ 0.10 could come from 1.3, 1.4, or 1.5 Hz — and those frequencies have very different mean K_t (0.73 vs 0.63 vs 0.60). The fit treats them as the same x → it eats some of what's actually frequency variation as "scatter".

Per-(amp, freq) mean K_t shows it cleanly:

| amp | f=1.3 | f=1.4 | f=1.5 | f=1.6 |
|---|---|---|---|---|
| A1 | 0.727 | 0.634 | 0.598 | 0.515 |
| A2 | 0.839 | 0.730 | 0.677 | 0.605 |
| A3 | 0.832 | 0.735 | 0.690 | 0.637 |

K_t drops monotonically with frequency at every amp. At fixed amp, *frequency is the dominant variable*, and ka is mostly a proxy for it.

**What this means for the fit:**
- The poly-2 K_t(ka) is mostly tracking K_t(f) with f re-expressed as k·a
- But because ka-clouds overlap between adjacent freqs, points from two different freqs at the same ka have different K_t
- The "extra scatter under wind" we see may be partly real wind-driven scatter and partly the frequency-overlap effect manifesting differently between wind conditions

**Proposed fix (cleanest):** fit through **per-frequency means** instead of through all individual points. So each (amp, wind) gets a fit that connects 4 points (one per frequency, each a mean ± std at that frequency's mean ka). This:
- Puts frequency back in explicitly — the curve passes through 4 frequency-distinct cells
- Removes ka-overlap as a source of scatter (since means are separated)
- Stays visually identical to the current style (one line per wind)
- Makes the wind story crisper: are the *means* spread, or is it just within-cell variance under wind?

Alternative is to keep the current fit but acknowledge the caveat in the caption. Or fit per (amp, freq, wind) and show 4 lines per wind — too cluttered IMO.

Want me to swap in the per-frequency-mean fit as a new variant (preserving the current `_with_fit` script)? Two scripts side-by-side would let you compare.
---
Adding a per-frequency-mean line on top of the existing smooth fit. Same colour, dashed style, mean markers at each frequency point — visually distinct without competing for the colour dimension.

Now insert the per-freq-mean line right after the existing fit loop:

Done. Each per-amp subfigure now has two overlaid curves per wind condition:

- **Solid line** — poly-2 fit through all individual points (existing). R² in the annotation.
- **Dashed line + filled markers** — line through the 4 per-frequency-mean points (new). Removes the ka-overlap-between-frequencies confound by collapsing each frequency to one (mean ka, mean K_t) point first.

Same colour for both lines per wind condition (red = med vind, blue = uten vind), so the wind dimension stays clean. The annotation box now reads:

```
heltrukken: poly-2 fit (alle punkter)
stiplet:    linje gjennom 4 frekvensgjennomsnitt
  uten vind: n=15, R² = 0.979
  med vind:  n=19, R² = 0.647
```

(Numbers shown are for A1; the `R²` is for the smooth fit only — the dashed line just connects 4 dots, no fit-quality metric needed.)

What to look for visually:
- If solid and dashed lines overlap closely → the smooth fit isn't being fooled by ka overlap; it's tracking the per-freq means correctly
- If they diverge (likely especially under wind at A1) → the smooth fit is fitting through scatter that the per-freq view shows is structured by frequency

Combined view (`ch05_damping_ka_fit.pdf`) still skips both overlays — too dense.

Files (overwritten in place):
- 3 per-amp PDFs + 1 combined under `output/FIGURES/ch05_damping_ka_fit*.pdf`
- 4 stubs under `output/TEXFIGU/`
- Script: [analysis_scratch/damping_ka_per_volt_with_fit.py](analysis_scratch/damping_ka_per_volt_with_fit.py)
---
Two actions: revert the dashed line, then investigate the 1.4 Hz spread at A1.

Reverted the dashed line — back to just the smooth fit.

For your A1 / 1.4 Hz mystery: yes, the table will answer it. Here are the 5 points at that cell:

| ka | K_t | per_tag | wind | Mooring | A_in [mm] | A_out [mm] |
|---|---|---|---|---|---|---|
| 0.066 | 0.527 | per40 | uten | below_90_**loose300** | 8.01 | 4.22 |
| 0.069 | 0.478 | per240 | uten | below_90_**loose300** | 8.10 | 3.87 |
| **0.103** | **0.641** | per240 | med | below_90_**loose300** | 8.22 | 5.27 |
| **0.105** | **0.812** | per240 | med | below_90_**loose230** | 7.29 | 5.93 |
| 0.119 | 0.710 | per40 | med | below_90_loose300 | 7.95 | 5.65 |

The two near-identical-ka points you spotted (0.103 vs 0.105) differ by **mooring** — `loose300` (30 cm slack) vs `loose230` (23 cm slack). Same per_tag, same wind, same paddle setting; the 0.17 K_t gap is the mooring effect, exactly the same physics we documented in the `ch05_mooring_focus_at_1_3hz_table` (where the canon-vs-above_50 mooring gap was +0.04 to +0.12 in K_t). Now we see a **within-canon** mooring-slack effect too: tighter 230 transmits more (K_t = 0.81), looser 300 transmits less (K_t = 0.64) at this cell.

Two breakdowns inside that gap:
- **A_in**: loose300 = 8.22 mm vs loose230 = 7.29 mm. The 300 mooring's IN reads 13% higher → drags K_t down.
- **A_out**: loose230 = 5.93 mm vs loose300 = 5.27 mm. The tighter mooring also passes more wave through → drags K_t up.

Both effects compound. Real physics, not noise.

**Will the existing tables show this?** The `ch05_wind_effect_table_by_amp` won't — it pools across moorings (Mooring column dropped before the grouper, per the 2026-05-05 fix). To surface this you'd need either:
- A new mooring-split table (per (amp, freq, mooring, wind)) — exposes that the canon dataset's "noise" includes a real loose230 vs loose300 effect.
- Or just a footnote in the caption acknowledging mooring-induced spread is present.

The poly-2 fit at A1 has R² = 0.65 under wind partly because of this — the smooth curve treats the mooring-driven 0.17 gap as scatter. With only 1 loose230 fullwind point at 1.4 Hz vs 2 loose300 fullwind points, there's no way for the fit to know.

**Quickest answer to your "do we need to inspect the data" question**: yes, but I just inspected it for you 👆. The driver is the within-canon mooring-slack difference (loose230 vs loose300), which the current canon-pooled tables don't surface. Want me to spin up a new mooring-aware variant of the wind-effect table so this is visible?

Files (overwritten):
- 3 per-amp PDFs back to single-fit-line style at `output/FIGURES/ch05_damping_ka_fit_A{1,2,3}.pdf`
- Script: `analysis_scratch/damping_ka_per_volt_with_fit.py` (reverted, freq-mean line code removed)

Here's the full canon comparison. Both moorings have data at 13 of 24 (amp, freq, wind) cells.

**All cells, sorted by amp / freq / wind:**

| Amp | f [Hz] | wind | K_t (loose230, n) | K_t (loose300, n) | Δ = 230−300 |
|---|---|---|---|---|---|
| A1 | 1.3 | uten | 0.646 (n=2) | 0.665 (n=7) | **−0.019** |
| A1 | 1.3 | med | 0.830 (n=4) | 0.757 (n=6) | **+0.073** |
| A1 | 1.4 | uten | — | 0.502 (n=2) | — |
| A1 | 1.4 | med | 0.812 (n=1) | 0.675 (n=2) | **+0.137** |
| A1 | 1.5 | uten | — | 0.453 (n=2) | — |
| A1 | 1.5 | med | 0.764 (n=1) | 0.661 (n=2) | **+0.103** |
| A1 | 1.6 | uten | — | 0.367 (n=2) | — |
| A1 | 1.6 | med | 0.647 (n=1) | 0.597 (n=2) | **+0.050** |
| A2 | 1.3 | uten | — | 0.811 (n=2) | — |
| A2 | 1.3 | med | 0.882 (n=1) | 0.846 (n=2) | +0.036 |
| A2 | 1.4 | med | 0.822 (n=1) | 0.728 (n=2) | **+0.094** |
| A2 | 1.5 | med | 0.789 (n=1) | 0.705 (n=2) | **+0.084** |
| A2 | 1.6 | med | 0.687 (n=1) | 0.683 (n=2) | +0.003 |
| A3 | 1.3 | med | 0.819 (n=1) | 0.837 (n=3) | −0.018 |
| A3 | 1.4 | med | 0.757 (n=1) | 0.769 (n=3) | −0.012 |
| A3 | 1.5 | med | 0.727 (n=1) | 0.730 (n=3) | −0.003 |
| A3 | 1.6 | med | 0.699 (n=2) | 0.711 (n=3) | −0.011 |

(All other (amp, freq, uten) cells at A2/A3 have *no* loose230 runs at all.)

**Headline summary** (13 comparable cells):
- median Δ = **+0.036** (loose230 transmits 3.6 % more on average)
- mean Δ = **+0.040**
- range: **−0.019 to +0.137**

**Pattern by amplitude tier** (under fullwind, where most comparisons exist):
- **A1**: Δ = +0.073, +0.137, +0.103, +0.050 → loose230 consistently +5 to +14 % higher
- **A2**: Δ = +0.036, +0.094, +0.084, +0.003 → loose230 higher except at 1.6
- **A3**: Δ = −0.018, −0.012, −0.003, −0.011 → moorings essentially equal (loose300 marginally higher)

So the mooring-slack effect **is real, sizeable at low amplitudes, and disappears at A3**. *Candidate explanation (hypothesis, not tested)*: at low amplitude the panel motion is small enough that the extra mooring slack genuinely lets the panel respond more passively (less restoring force = less reflection); at high amplitude the panel hits the slack regime in both moorings and the difference is washed out. Worth a sentence in the thesis but flagged as a hypothesis.

**Methodology caveats baked into this comparison:**

1. **The two moorings are aliased with date** — loose230 = Mar 26, loose300 = Mar 27. So the differences could partly be daily setup drift (water level recheck, fan position, temperature). Within a day, only the mooring changes — but you have one day per mooring. Not a controlled crossover.

2. **loose230 has almost no nowind runs at A2/A3** — Mar 26 ran a partial matrix focused on A1 1.3 + one fullwind run per other cell. Mar 27 ran the systematic matrix. So 11 of 24 cells have only loose300, and the nowind comparison only exists at A1 1.3.

3. **Per-cell n is tiny for loose230** (often n=1 fullwind). The +0.137 outlier at A1 1.4 fullwind is one-vs-two runs — it's the single observation the user spotted in the figure.

**Implication for the wind_effect_table_by_amp:**
The 2026-05-05 fix that pools across moorings (drops Mooring before grouper) silently merges these two rigs. The pooled K_t means hide the +0.04 median mooring gap. Probably fine for the headline thesis result (the canon-vs-above_50 mooring split is the bigger story) but worth knowing about. If you want to surface this, easiest is a footnote — or a new sibling table that adds Mooring as an explicit row dimension.

---


Let me peek at the previous files first to align with what was done.

Building both. First, a focused script that produces the LaTeX table — using the structure I already pulled together. Then a frequency scatter plot per amp tier.

Table done. Now the frequency scatter:

Both delivered. Quick summary of what each shows:

**Table** — `output/TABLES/ch05_loose230_vs_loose300_table.tex` (24 cells, 13 with both moorings):

| Headline | Value |
|---|---|
| median Δ = K_t,230 − K_t,300 | **+0.036** |
| mean Δ | +0.040 |
| range | −0.019 to +0.137 |

Outer = amp, inner = freq, then wind. Cells where only one mooring sampled show "—". Columns: K_t per mooring (with n), absolute Δ, ratio K_t,230 / K_t,300.

**Frequency scatter** — 3 PDFs (`output/FIGURES/ch05_loose230_vs_loose300_freq_A{1,2,3}.pdf`):

- x = frequency
- y = K_t (mean ± std where n>1)
- colour = wind (red/blue)
- marker = mooring × amp tier:
  - loose300 → ○ □ △ (normal shapes — primary/well-sampled mooring)
  - loose230 → 6/5/4-point star (re-using the "secondary/sparse" marker family from earlier CH05 plots)
- linestyle = mooring (solid loose300, dashed loose230)
- all hollow markers

The visual story matches the table: at A1 the dashed line (loose230) sits visibly above the solid (loose300) under wind across all 4 frequencies; at A2 it's smaller; at A3 the two lines are essentially on top of each other (mooring-difference vanishes at high amplitude).

Files:
- TeX: [output/TABLES/ch05_loose230_vs_loose300_table.tex](output/TABLES/ch05_loose230_vs_loose300_table.tex)
- CSV: [analysis_scratch/loose230_vs_loose300_table.csv](analysis_scratch/loose230_vs_loose300_table.csv)
- 3 PDFs: [A1](output/FIGURES/ch05_loose230_vs_loose300_freq_A1.pdf) · [A2](output/FIGURES/ch05_loose230_vs_loose300_freq_A2.pdf) · [A3](output/FIGURES/ch05_loose230_vs_loose300_freq_A3.pdf)
- Scripts: [loose230_vs_loose300_table.py](analysis_scratch/loose230_vs_loose300_table.py) · [loose230_vs_loose300_freq_scatter.py](analysis_scratch/loose230_vs_loose300_freq_scatter.py)

**Methodology caveats baked into both**:
- Mooring is aliased with date (loose230 = Mar 26 only, loose300 = Mar 27 only) — daily setup drift can't be separated from mooring effect
- loose230 has nowind data only at A1 / 1.3 Hz; all other nowind comparisons impossible
- Per-cell n for loose230 fullwind is mostly 1 — the +0.137 outlier is a 1-vs-2 comparison

Not yet wired into main_save_figures.py. Want me to add the cell + caption entries when you've decided whether these earn a thesis spot?

Let me verify the provenance of my numbers by comparing to what the old findings doc claimed.

Independent analysis — but the comparison reveals an important scope difference.

**My script vs the old findings doc:**

| Cell | Old findings (Apr 20) | My script (today) |
|---|---|---|
| 1.3 Hz / A1 / fullwind, loose230 K_t | **0.915 (n=9)** | **0.830 (n=4)** |
| 1.3 Hz / A1 / fullwind, loose300 K_t | 0.805 (n=6) | 0.757 (n=6) |
| → Δ | **−0.110** | **+0.073** |

The **sign of Δ even flipped**. So no, I did *not* base anything on the old doc — and good thing, because the old numbers don't match the current cache.

**Two reasons for the difference:**

1. **Scope**: the old doc pooled loose230 from **folders 20260316–26** (~9 folders covering h272/high, h100/high, h136/high, and h100/low probe configs). My script restricts to the **two canon lowrange folders only** (Mar 26 + Mar 27, both h100/low) — matching `wind_effect_table_by_amp.py`'s canon scope. That's why my n_230 = 4 instead of 9.

2. **Pipeline drift since Apr 20**: the H&G window roll-out (May 2) and the window-mean baseline change (May 5) both touched K_t values. Differences are small per cell (~0.01) but accumulate; old doc's K_t = 0.915 vs my 0.830 is too big to be only pipeline drift, so the scope difference is the dominant factor.

**What I actually did:**
- Read the old `mooring_comparison.py` only for **filename conventions** (THESIS_NAME pattern) — never executed the old script, never imported its findings.
- Built `loose230_vs_loose300_table.py` and `loose230_vs_loose300_freq_scatter.py` fresh, calling `load_analysis_data(*RESULTS_DIRS, …)` directly from the current `waveprocessed/` cache.
- Filter is identical to `wind_effect_table_by_amp.py` (canon lowrange folders, full panel, quality_flag=ok, 1.3–1.6 Hz).

**Question for you:** the canon-only scope makes my numbers consistent with the headline thesis tables, but it sharply reduces n for loose230 (4 vs 9 at the headline cell). Do you want me to also produce a **broad-scope variant** (loose230 from Mar 16–26 across all probe configs) to mirror what the old doc was doing? That would:
- Trade strict scope-matching with the canon thesis result for more statistical power on the mooring question
- Re-introduce the probe-config-as-confound problem (which we previously showed doesn't bias K_t much within below_90_loose230, so this is probably fine)
- Make the numbers comparable to the old doc — which should give the user a sanity-check that the pipeline drift isn't huge

Or stick with canon-only — clean, but n=4 limits how strongly we can claim anything.


Now also update the immutable-block scope text and the RESULTS_DIRS dump (which prints all folders — fine, just larger):

Now do the same for the freq scatter broad version:

Now update the output filenames in the broad freq-scatter script:

Both broad-scope versions delivered. The headline comparison is dramatic:

| stat | canon (h100/low only) | broad (all configs) |
|---|---|---|
| cells with both moorings | 13 | **24** (all of them) |
| median Δ = K_t,230 − K_t,300 | **+0.036** | **−0.002** |
| mean Δ | +0.040 | +0.009 |
| range | −0.019 to +0.137 | −0.038 to +0.104 |

**The "loose230 transmits more" pattern almost vanishes at broad scope.** Median Δ goes from +0.036 (canon) to ≈0 (broad). The mooring effect that the canon-only data suggested isn't robust to widening the sample.

Cell-by-cell, the divergences worth flagging:

| Cell | Canon Δ | Broad Δ | What changed |
|---|---|---|---|
| A1 / 1.4 / med | **+0.137** | +0.051 | Outlier in canon shrinks 3× |
| A1 / 1.6 / med | +0.050 | −0.006 | Sign flips |
| A2 / 1.5 / med | +0.084 | +0.004 | Effect collapses |
| A2 / 1.4 / med | +0.094 | +0.038 | Halved |

A few cells stay consistent (A1 / 1.5 / med: +0.103 canon vs +0.104 broad — almost identical). But on the whole, broad scope says the moorings are equivalent; canon scope said loose230 is ~4 % higher.

**Three readings, take your pick:**

1. **The canon-only signal was small-sample noise**. Broadening to ~6× more loose230 runs revealed the effect averages out. The thesis should report "no defensible mooring-slack effect" and stop here.
2. **The broad scope added drift from 9 days of setup variation**, washing out a real per-setup mooring effect that canon caught (Mar 26 vs Mar 27, controlled). Canon is the trustworthy reading; broad mixes apples and oranges.
3. **Truth is in between**: there's a real but small mooring effect (~+0.01 in K_t), inflated to +0.036 by canon-only small-sample noise.

I'd lean toward (3) but the data can't really discriminate. The honest thing in the thesis is probably to show **both** tables side by side, point out the divergence, and let the reader see the data limit.

Files (all new — none overwriting the canon-scope versions):

| Type | Canon | Broad |
|---|---|---|
| Table .tex | [output/TABLES/ch05_loose230_vs_loose300_table.tex](output/TABLES/ch05_loose230_vs_loose300_table.tex) | [output/TABLES/ch05_loose230_vs_loose300_table_broad.tex](output/TABLES/ch05_loose230_vs_loose300_table_broad.tex) |
| CSV | [analysis_scratch/loose230_vs_loose300_table.csv](analysis_scratch/loose230_vs_loose300_table.csv) | [analysis_scratch/loose230_vs_loose300_table_broad.csv](analysis_scratch/loose230_vs_loose300_table_broad.csv) |
| Plots A1/A2/A3 | [`ch05_loose230_vs_loose300_freq_A{1,2,3}.pdf`](output/FIGURES/) | [`ch05_loose230_vs_loose300_freq_broad_A{1,2,3}.pdf`](output/FIGURES/) |
| Script | [loose230_vs_loose300_table.py](analysis_scratch/loose230_vs_loose300_table.py), [loose230_vs_loose300_freq_scatter.py](analysis_scratch/loose230_vs_loose300_freq_scatter.py) | [loose230_vs_loose300_table_broad.py](analysis_scratch/loose230_vs_loose300_table_broad.py), [loose230_vs_loose300_freq_scatter_broad.py](analysis_scratch/loose230_vs_loose300_freq_scatter_broad.py) |

Want me to also wire either version into main_save_figures.py? My instinct: not yet — let you read the figures first and decide whether the divergence warrants thesis space (it's a methodology footnote at most, given the inconclusive picture).