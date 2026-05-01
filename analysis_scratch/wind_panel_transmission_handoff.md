# Wind effect on panel transmission — handoff for verification

**Date written**: 2026-05-01
**Author**: claude-opus-4-7 (current session)
**Audience**: next agent / human reviewer
**Purpose**: present observations from a multi-step analysis so they can be
verified against the cached data, with locations of every script and CSV.

> Read this with the project's observation-vs-inference discipline in mind
> (CLAUDE.md §20 + memory/MEMORY.md). All numbers below are observations.
> Where mechanism is offered, it is flagged `*Candidate explanation
> (hypothesis)*:` and is **not** the agent's claim of fact. Do not promote
> the hypotheses to text without independent verification.

---

## 1. The question

> Does wind increase or decrease how much of an incoming wave is
> transmitted past the panel geometry?
> (CLAUDE.md §0 — primary thesis question.)

The session walked through four nested sub-questions:

1. Across canon (f, A) cells, how does wind change A_IN, A_OUT, and the
   IN→OUT ratio at the paddle frequency?
2. Is the IN-side wind enhancement spectral contamination at f_p, or
   coherent coupling?
3. How does panel transmission compare to bare-tank propagation, with and
   without wind?
4. Is the bare-tank "growth under wind" finding from step 3 robust to
   IN-side wind background correction?

Sub-question 4 turned up a bug in the step-3 aggregator. The corrected
picture is below. **The bug is the most important thing in this doc.**

---

## 2. Scripts and outputs

All scripts are under `analysis_scratch/`. All outputs are in
`analysis_scratch/` unless otherwise noted. None of these have been
promoted to `output/`.

| Step | Script | Outputs |
|---|---|---|
| 1 — wind effect across canon cells | `wind_effect_per_condition.py` | `wind_effect_per_condition_per_run.csv` (72 rows), `wind_effect_per_condition_long.csv` (88 rows), `wind_effect_per_condition_ratios.csv` (11 rows), `wind_effect_ratios_summary.png`, `wind_effect_scatter_f14_{A1,A2,A3}.png` |
| 2 — spectral subtraction at canon 1.4 Hz | `spectral_subtract_canon14.py` | `spectral_subtract_canon14.csv`, `spectral_subtract_canon14_psd.png` |
| 3 — panel vs nopanel + November bridge | `nopanel_panel_comparison.py` | `nopanel_panel_comparison_per_run.csv` (122 rows), `nopanel_panel_comparison_per_condition.csv` (37 rows), `nopanel_panel_RD_summary.csv` (11 rows), `nopanel_panel_RD_summary.png` — **CONTAINS THE BUG, see §5** |
| 4 — sanity check on November bridge | `nopanel_bridge_in_sanity.py` | `nopanel_bridge_in_sanity_per_run.csv` (16 rows), `nopanel_bridge_in_sanity_summary.csv` (6 rows), `nopanel_bridge_in_sanity.png` |

Probe positions, separations, and pipeline assumptions are documented in
each script's docstring.

---

## 3. Headline observations (numbers only)

All amplitudes from least-squares fits at f_p on the 10-period H&G window
(see CLAUDE.md §5, §7). PanelCondition == "full", quality_flag == "ok",
canon March 2026 lowrange folders unless stated otherwise.

### 3.1 Wind ratios per (f, A), R = wind/nowind

Source: `wind_effect_per_condition_ratios.csv`, columns `R_IN_mean`,
`R_OUT`, `R_T_IN_mean`.

| amp | f [Hz] | R_IN | R_OUT | R_T |
|---:|---:|---:|---:|---:|
| A₁ | 1.3 | 0.99 | 1.18 | 1.19 |
| A₁ | 1.4 | 0.97 | 1.39 | 1.43 |
| A₁ | 1.5 | 1.05 | 1.59 | 1.52 |
| A₁ | 1.6 | 1.16 | 1.93 | 1.66 |
| A₂ | 1.3 | 1.01 | 1.07 | 1.06 |
| A₂ | 1.4 | 1.03 | 1.14 | 1.10 |
| A₂ | 1.5 | 1.03 | 1.27 | 1.23 |
| A₂ | 1.6 | 1.06 | 1.50 | 1.41 |
| A₃ | 1.3 | 1.04 | 1.04 | 1.00 |
| A₃ | 1.4 | 1.03 | 1.17 | 1.14 |
| A₃ | 1.5 | 1.09 | 1.30 | 1.19 |

(A₃ at 1.6 Hz excluded by `EXCLUDE_CONDITIONS` per
`memory/feedback_freq_amp_limits.md`.)

Patterns observed in this table:
- R_IN ≤ 1.16 in every cell.
- R_OUT and R_T grow with frequency at every amplitude tier.
- R_OUT ≥ R_T ≥ R_IN in every row.

### 3.2 Pre-paddle σ at OUT under fullwind

Source: same CSV, column `sigma_pre_OUT_full_mm`. Range across the 11
condition cells: 0.30–0.37 mm.

CLAUDE.md §16 records the OUT probe (12400/250) stillwater noise floor
at 0.14–0.36 mm (gold standard 0.14 mm). The fullwind σ_pre at OUT is
within that envelope.

### 3.3 Spectral subtraction at canon (1.4 Hz, A₂, IN-wall)

Source: `spectral_subtract_canon14.csv`,
`spectral_subtract_canon14_psd.png`.

| quantity | value [mm] |
|---|---:|
| A_meta_FFT, nowind | 14.959 |
| A_PSD-raw, nowind | 14.969 |
| A_PSD-clean (− P_still), nowind | 14.969 |
| A_meta_FFT, fullwind | 16.413 |
| A_PSD-raw, fullwind | 16.424 |
| A_PSD-clean (− P_wind_3s − P_still), fullwind | 16.423 |

Wind PSD level at 1.4 Hz (from 3 s pre-paddle): ~0.03 mm²/Hz. Run PSD at
the f_p bin: ~10² mm²/Hz. The PSD-clean amplitude differs from PSD-raw
by ≤ 0.001 mm.

### 3.4 March-strict panel vs bare-tank, no-wind only

Source: `nopanel_panel_RD_summary.csv` columns prefixed `march_`. Panel
runs at 3.027 m, March nopanel at 2.427 m, extrapolated by
`G(L) = exp(ln(G(L₀))/L₀ · L)`.

| amp | f | T_panel,no | G_nopanel,no(2.43 m) | k_no [1/m] | G_nopanel,no(3 m) | D_no |
|---:|---:|---:|---:|---:|---:|---:|
| A₁ | 1.3 | 0.659 | 0.962 | −0.016 | 0.953 | 0.691 |
| A₁ | 1.4 | 0.509 | 0.971 | −0.012 | 0.964 | 0.528 |
| A₁ | 1.5 | 0.433 | 0.968 | −0.013 | 0.960 | 0.450 |
| A₁ | 1.6 | 0.337 | 0.968 | −0.013 | 0.960 | 0.351 |
| A₂ | 1.3 | 0.776 | 0.971 | −0.012 | 0.964 | 0.805 |
| A₂ | 1.4 | 0.689 | — | — | — | — |
| A₂ | 1.5 | 0.587 | — | — | — | — |
| A₂ | 1.6 | 0.474 | 0.976 | −0.010 | 0.970 | 0.488 |
| A₃ | * | — | — | — | — | — |

D_no_wind cells without a march nopanel match are NaN. A₃ is uncomputable
in march-strict because no march nopanel runs were taken at A₃.

### 3.5 November nopanel @ 1.3 Hz — the sanity check

Source: `nopanel_bridge_in_sanity_per_run.csv` (16 per-run rows; n=16).

Per-run G = A_OUT_LS / A_IN_LS at 1.3 Hz across all amplitudes and both
wind states:

| date | amp | wind | A_IN [mm] | A_OUT [mm] | G |
|---|---|---|---:|---:|---:|
| 2025-11-10 | A1 | full | 0.32 | NaN | — |
| 2025-11-10 | A2 | full | 0.42 | NaN | — |
| 2025-11-10 | A3 | full | 0.20 | NaN | — |
| 2025-11-12 | A1 | full | 8.03 | 8.15 | 1.014 |
| 2025-11-12 | A2 | full | 15.88 | 16.25 | 1.023 |
| 2025-11-12 | A3 | full | 23.40 | 22.65 | 0.968 |
| 2025-11-13 | A1 | full | 8.02 | 8.66 | 1.081 |
| 2025-11-13 | A2 | full | 15.70 | 15.19 | 0.968 |
| 2025-11-13 | A3 | full | 23.78 | 22.97 | 0.966 |
| 2025-11-12 | A1 | no | 7.61 | 7.45 | 0.979 |
| 2025-11-12 | A2 | no | 15.13 | 14.89 | 0.984 |
| 2025-11-12 | A3 | no | 22.37 | 21.79 | 0.974 |
| 2025-11-13 | A1 | no | 7.63 | 7.79 | 1.020 |
| 2025-11-13 | A1 | no | 7.54 | 7.55 | 1.001 |
| 2025-11-13 | A2 | no | 14.83 | 14.82 | 0.999 |
| 2025-11-13 | A3 | no | 22.11 | 21.76 | 0.984 |

Per-run G across all 13 finite rows: range **0.97–1.08**.

3 s pre-paddle wind correction (uncorrelated PSD subtraction at f_p):
shifts G by less than 0.005 in every cell. Wind contribution at f_p is
0.3 % at nowind, 1.3–5.6 % at fullwind, of A_IN.

The 20251110 runs are `per15` (15-period total wave train, ~11.5 s of
paddle motion at 1.3 Hz). The OUT H&G window at 12400 mm sits past the
recording's wave content → A_OUT = NaN. IN at 9373 mm is closer, so its
H&G window catches some post-paddle settling and the LS fit returns a
small-but-finite value (~0.3 mm), not a wave amplitude.

---

## 4. What changed when these analyses were run

Four facts came out of the session that did not exist beforehand. They
are not yet recorded in any memory file:

1. R_T (wind/nowind transmission ratio) is > 1 at every (f, A) cell
   except (A₃, 1.3 Hz) which is essentially 1.0. Magnitude grows with f.
2. R_OUT > R_IN at every cell. The wind-driven amplitude change is
   concentrated downstream of the panel.
3. Spectral subtraction at canon 1.4 Hz removes ~0.001 mm of the 1.45 mm
   IN-side wind enhancement. Wind energy at f_p is three orders of
   magnitude too small to explain the IN amplitude shift via uncorrelated
   superposition.
4. Per-run November nopanel G at 1.3 Hz is in 0.97–1.08 across all 13
   valid runs (3 dropped due to per15 H&G mismatch), both wind states.

---

## 5. The bug

Location: `analysis_scratch/nopanel_panel_comparison.py`, `_gain` helper
near line 348 of the file (post-fix line numbers may differ if anyone
edits it).

```python
def _gain(series, f, a, w):
    A_in,  n_in  = _lookup(series, f, a, w, "A_IN_mean_mm")
    A_out, n_out = _lookup(series, f, a, w, "A_OUT_mean_mm")
    ...
    return A_out / A_in, min(n_in, n_out)
```

`A_in` and `A_out` come from the per-condition aggregation, where each
column is `mean(grp[col].dropna())`. **The dropna runs independently per
column.** A run with finite A_IN but NaN A_OUT contributes its A_IN to
the mean but not to A_OUT — and vice versa. The two means are then
divided as if they came from the same set of runs.

In the November nopanel cell (A2, fullwind, f=1.3 Hz):

- 3 candidate runs (1 from each November date)
- per15 run (20251110): A_IN = 0.42 mm (post-paddle settling), A_OUT = NaN
- per40 run (20251112): A_IN = 15.88 mm, A_OUT = 16.25 mm
- per40 run (20251113): A_IN = 15.70 mm, A_OUT = 15.19 mm

Aggregator output:
- A_in_arr = [0.42, 15.88, 15.70] → mean = 10.67 mm
- A_out_arr = [16.25, 15.19] → mean = 15.72 mm
- G = 15.72 / 10.67 = **1.474**

Per-run truth:
- runs with both finite: G = 1.023 and 0.968
- mean per-run G = **0.996**

The previously-reported `nov_G_nopanel_wind_at_3m = 1.474` and the entire
"bare-tank wind amplification" finding are artifacts of asymmetric NaN
handling.

### The fix has not been applied

`nopanel_panel_comparison.py` and `nopanel_panel_RD_summary.csv` still
contain the buggy numbers. The verifier should:
1. Re-run with symmetric NaN handling (drop runs where either A_IN or
   A_OUT is NaN, then aggregate); OR
2. Add a `run_category in {"per40","per240"}` filter that excludes the
   per15 runs upstream.

Both fixes give the same answer at this cell. Pick whichever is more
defensible.

---

## 6. What's reliable, what's not

| Result | Reliable? | Where to verify |
|---|---|---|
| R_IN, R_OUT, R_T per (f, A) (table 3.1) | Reliable | per-run rows in `wind_effect_per_condition_per_run.csv` agree with the LS pipeline; aggregation is symmetric |
| Spectral subtraction at canon 1.4 Hz (table 3.3) | Reliable | A_meta vs A_PSD-raw match within 0.07 % — pipeline FFT and the script's periodogram give the same answer |
| March-strict D_no_wind (table 3.4) | Reliable for the cells where it's not NaN. n_nopanel is 0–2, so confidence intervals are wide. | Verify by re-running with symmetric NaN guard; the same CSV row should reappear |
| November bridge G_nopanel,wind ≈ 1.47 (in `nopanel_panel_RD_summary.csv`) | **NOT reliable** — bug | See §5 |
| D_wind, R_D in November bridge (in `nopanel_panel_RD_summary.csv`) | **NOT reliable** — depends on the buggy G | See §5 |
| Per-run G in November = 0.97–1.08 (table 3.5) | Reliable | `nopanel_bridge_in_sanity_per_run.csv` directly |

---

## 7. What the next agent should do to verify

In order:

1. **Reproduce table 3.5 from scratch.** Re-run
   `analysis_scratch/nopanel_bridge_in_sanity.py` and compare to the
   committed CSVs. If anything changes, investigate.
2. **Verify the bug exists in `nopanel_panel_comparison.py`.** Pull the
   per-run rows for `series == "nov_nopanel"`, `freq_hz == 1.3`,
   `amp_v == 0.2`, `wind == "full"` from
   `nopanel_panel_comparison_per_run.csv`. Confirm there are 3 rows, that
   one has A_OUT_mm = NaN, and that running
   `mean(A_OUT.dropna())/mean(A_IN.dropna())` reproduces 1.474 while
   running `mean(A_OUT/A_IN)` over the rows where both are finite gives
   ~0.996.
3. **Apply the fix.** Either (a) symmetric NaN handling in the
   aggregator: drop a run from a condition cell if either A_IN or A_OUT
   is NaN; or (b) filter `run_category in {"per40","per240"}` upstream.
   Re-run, regenerate `nopanel_panel_comparison_per_condition.csv`,
   `nopanel_panel_RD_summary.csv`, `nopanel_panel_RD_summary.png`.
   Report the new D_now, D_wind, R_D values.
4. **Cross-check the corrected November bridge against table 3.5.** With
   the fix, the November bridge cells in `nopanel_panel_RD_summary.csv`
   should give G_nopanel ≈ 1.0 in both wind states, D ≈ T_panel, and
   R_D ≈ R_T (from table 3.1). Confirm.
5. **Decide on march-strict A2 1.4/1.5 Hz and all A₃**: nopanel data for
   these does not exist in March. Should the corrected summary leave
   them as NaN, fall back to November nopanel, or flag explicitly?
6. **Decide whether to update memory.** The two findings worth
   memorialising IF VERIFIED are:
   - The bug pattern (asymmetric NaN handling in mean-of-means
     aggregators) and its fix.
   - The corrected R_T pattern (downstream, not upstream, drives the
     wind effect on the canon transmission metric).
   Existing memory file `methodology_wind_enhances_A_in.md` may need an
   addendum, not a rewrite, per the project's observation-vs-inference
   discipline.

---

## 8. What the next agent should NOT do

- Do not promote the existing `nopanel_panel_RD_summary.csv` numbers to
  any thesis figure or table without applying the fix.
- Do not write causal-language thesis prose from the corrected numbers
  alone. The data are 11 (f, A) cells with n = 1–14 runs per cell, no
  nopanel + fullwind in the canon period, and a cross-campaign bridge
  that mixes probe layouts. Causal claims need additional data (esp.
  a march nopanel + fullwind dataset) or independent corroboration.
- Do not delete the buggy outputs. Per memory file
  `feedback_never_delete_outputs.md`, regenerated outputs overwrite in
  place; do not `rm` the existing files.
- Do not assume the IN-side spectral subtraction result at 1.4 Hz
  (table 3.3) generalises to other (f, A) cells without re-running.
  *Candidate explanation (hypothesis)*: wind couples coherently to the
  paddle wave at the IN probe — but this is one cell on one probe.

---

## 9. Open methodological questions raised but not closed

1. The `quality_flag == "ok"` filter does not catch per15 runs whose H&G
   window falls past the wave train. Worth a separate quality check
   (e.g. require `Computed Probe {pos} end ≤ N_samples − 100` or test
   that the LS residual ratio is below a threshold).
2. November and March probe layouts have IN/OUT laterals swapped (Nov:
   IN-centre + OUT-wall; March: IN-wall+far + OUT-centre). The
   sensitivity of G_nopanel to that swap was not tested.
3. The November nopanel data at 1.3 Hz alone cannot test wind-frequency
   dependence of bare-tank propagation. To test whether bare-tank is
   wind-independent at 1.4–1.6 Hz (where R_T is largest in table 3.1)
   requires more nopanel + fullwind runs at higher frequency.
4. The OUT-side R_OUT > R_IN observation has a *candidate explanation
   (hypothesis)*: wind transfers energy to the wave system somewhere
   between IN and OUT, e.g. across the panel. But the data here cannot
   distinguish "panel becomes more transparent" from "wind sets up a
   current that changes the panel's effective interaction" from "wind
   modifies the IN amplitude denominator slightly differently than OUT".
   Do not state any of these as fact.
