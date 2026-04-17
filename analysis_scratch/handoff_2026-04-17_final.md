# Handoff — 2026-04-17 (end of day)

**Previous handoff**: `handoff_2026-04-17.md` (post-interruption snapshot).
This file supersedes it.

---

## State at handoff

- `main` is **clean**, pushed to origin through commit `37df161`.
- Pipeline cache is **fresh** (`waveprocessed/run_20260417_145335_force.log`).
  646 runs across 25 datasets. **9 RECON ABORTs**, all ≥1.8 Hz or 0.8 Hz per11
  (all out of scope) — no in-scope aborts.
- User just finished a walk-through review of
  `probe_height_wind_findings.md` (rewritten to remove fabricated
  explanations from the 2026-03-30 agent).
- Today's commits (oldest → newest):
  ```
  da22837 sliding AFFT sweep (1.3–1.6 Hz × 0.1/0.2 V)
  96e2e93 sliding AFFT sweep: add matched_mid_AFFT diagnostic
  cb52fce sliding AFFT: zoom on 1.5 Hz / 0.2 V — anomaly ruled out
  5a6268a OUT/IN bias-cancellation test on 367 real runs
  983215e checkpoint interrupted 04-17 session + annotate handoff
  5e0488a CH04 methodology figure: FFT peak-bin bias cancels in OUT/IN
  daf7085 refresh FFT peak-bias figure from post-recompute cache
  5cafa1c fix: CH04 §3c mooring comparison groupby key name
  7ff3356 damping_grouper: make full+reverse collapse opt-in
  ea0911d fix: aggregated grouper output uses mean_out_in, not OUT/IN (FFT)
  ef3abd6 promote sliding AFFT stability figure to CH04 §4e; drop scratch PNGs
  16f3ff2 promote Mansard-Funke + SW correction figures to CH04 (§4c, §4d)
  37df161 rewrite probe_height_wind findings — correct wrong conclusions
  ```

---

## What got done today (in thematic groups)

### 1. Methodology investigation: FFT peak-bin bias

**Arc**: user flagged a per240 AFFT-discrepancy PDF from earlier. Investigation:
- Parent sweep (`sliding_afft_fullwind_sweep.py`) at 1.3–1.6 Hz × 0.1/0.2 V.
- Zoom on 1.5 Hz / 0.2 V (`sliding_afft_15hz_02v_zoom.py`) — ruled out transient.
- Root cause identified: **FFT nearest-bin amplitude is biased** by paddle-drift
  vs bin-grid alignment (sinc attenuation up to ~40% for individual AFFTs).
- OUT/IN robustness test on 367 real runs (`fft_peak_bias_outin_impact.py`):
  mean |Δ|/OUT/IN = **0.46%**, 95% of runs < 0.02 absolute → **bias cancels in
  OUT/IN ratio**. Thesis OUT/IN values safe.

**Artifacts**:
- `memory/methodology_fft_peak_bin_bias.md` — methodology memo
- `memory/open_question_per240_afft.md` — original question resolved
- CH04 §4b figure `ch04_fft_peak_bias_cancellation.pdf` (plot function in
  `wavescripts/plotter.py::plot_fft_peak_bias_cancellation`)

### 2. Pipeline verification (post 8804 end-calibration)

- 2026-04-16b session added `_SNARVEI_END_CALIB["8804"]` — this session's
  `run_20260417_145335_force.log` confirms **mstop330-run3 RECON ABORT is
  gone**; 9 remaining RECON ABORTs are all out-of-scope.
- Pipeline is stable; `meta_results` = cond4-only (both `-lowrange` folders).

### 3. Three new CH04 methodology figures promoted

All use the same promotion pattern: scratch script writes PDF + `.tex` stub
directly into `output/FIGURES/` + `output/TEXFIGU/`. PNGs dropped to save
space (user preference).

| Section | Figure | Source script |
|---|---|---|
| CH04 §4b | FFT peak-bin bias cancels in OUT/IN | `plot_fft_peak_bias_cancellation` (plotter.py) |
| CH04 §4c | Mansard-Funke reflection coefficient (R≈0.05–0.07) | `analysis_scratch/mansard_funke.py` |
| CH04 §4d | Standing-wave correction test (negative evidence) | `analysis_scratch/sw_correction.py` |
| CH04 §4e | Sliding AFFT stability at IN probe | `analysis_scratch/sliding_afft_fullwind_sweep.py` |

§4b is a proper plotter function (reads CSV). §4c–§4e are
scratch-script-driven — `main_save_figures.py` has documentation cells that
verify the figure files exist and point to the script for regeneration.

### 4. Refactor: `damping_grouper` collapse is now opt-in

Both `damping_grouper` and `damping_all_amplitude_grouper` in
`wavescripts/filters.py` took a new `collapse_panels: bool = False` parameter.

- **Default (False)**: `PanelCondition` preserved (`full`, `reverse`, `no` as
  separate groups).
- **Opt-in (True)**: old behaviour (full+reverse → "all" via
  `PanelConditionGrouped`), now with a LOUD print warning.

All 7 `plotter.py` consumers bulk-renamed from `GC.PANEL_CONDITION_GROUPED` to
`GC.PANEL_CONDITION`. REPL quicklook (`main_explore_inline.py`) passes
`collapse_panels=True` explicitly to preserve behaviour.

Thesis impact: **none**. All CH05 callers use `meta_results` (zero reverse
rows); CH04 §3c mooring cell pre-filters to `PanelCondition == "full"`.
The collapse was a no-op on the thesis path — this change removes it as a
latent landmine.

Open research question enabled: `memory/open_question_fullpanel_vs_reversepanel.md`.

### 5. Bug C confirmed already implemented

The `in_probe_low_snr` quality flag check at `processor.py:1607–1623` works
as designed. Audit confirmed 3 runs correctly flagged (ws = 0.198, 0.323,
0.335). The 2026-04-17 handoff's "slip through" concern was stale
(`wave_stability` values had shifted slightly after recompute).

### 6. Critical bug fixes in `main_save_figures.py`

- CH04 §3c mooring groupby used a column the grouper no longer produces
  (`PanelCondition` → `PanelConditionGrouped`). Fixed, then re-fixed after
  the grouper refactor.
- Ad-hoc 1.3 Hz audit cell at the tail referenced `"OUT/IN (FFT)"` which
  the grouper aggregates to `mean_out_in`. Fixed.

### 7. Walk-through of `probe_height_wind_findings.md`

User flagged that the 2026-03-30 agent had "lots of wrong conclusions".
Walk-through surfaced and corrected:

- **No probe swap ever** (agent invented one).
- **h272 highest-noise is EXPECTED physics** (longer acoustic path → more
  noise), not surprising. Correct ordering is h272 > h136 > h100.
- **cond3 = user forgot range-mode switch** after lowering probes.
  P2-malfunction runs likely originate from this period.
- **h136 "stuck at quantization"**: speculative, n=1, dropped.
- **20260323 "water disturbance"**: agent speculation without signal
  check, dropped.
- **Wind amplitude "consistent across conditions"**: wind field IS
  consistent (fan + water-level confirm); probe reports differ because of
  range-mode + height combination.
- **OUT-probe wind amplitude variation (0.82–1.02 mm)**: physical mechanism
  the agent missed — **longer mooring → panels extend further back →
  shorter post-panel fetch → smaller OUT ripples**. Evidence in the data:
  loose230 OUT = 1.016 mm vs loose300 OUT = 0.817 mm.
- **SNR framing**: agent conflated dynamic range with SNR. Thesis-relevant
  SNR = `A_paddle / A_wind` at OUT, not `A_wind / σ_stillwater`.
- **Finding 4 "cond3 physically usable"**: replaced with description of
  actual quality-flag mechanism + TODO for per-run audit.

---

## Open items for next agent

### CRITICAL

1. **Per-run quality-flag audit** (HIGH, user-requested 2026-04-17).
   Produce a debug document that enumerates each flagging layer in
   `processor.py::_write_quality_flags` and walks through how it classifies
   each run. Format suggestion: one section per layer, with a table of runs
   it flagged and the triggering metric value. User wants to review which
   runs got flagged and why.

2. **1.6 Hz chaos in parallel_ratio** (see
   `memory/known_baddata_ultrasound_16hz.md`). User confirmed it's
   false-reading bad-day ultrasound data — not a physical effect. Needs:
   (a) identify the specific date(s); (b) either add a per-run
   `quality_flag` or filter the affected runs from CH04 §3 figures.
   **Check first**: is the bad day in `meta_results` (should not be,
   since meta_results is cond4-only)?

### HIGH

3. **9373 snarvei Day 2 eyeballing** — user-only decision. Day 2 values
   (mstop30 per40 nowind, 0.2 V) suggest end at 37/37/36/35 s vs current
   `_SNARVEI_END_CALIB["9373"]` = 39/38/36/36. See
   `analysis_scratch/snarvei_eyeballing.md`. Apply or not?

4. **Mooring-length → post-panel fetch** — promote the insight from
   `probe_height_wind_findings.md` into its own memory file
   (`memory/physics_wavetank_mooring_fetch.md`). The physics:
   longer mooring → front panels extend further back (downstream) →
   shorter free-water fetch from back-of-panel-row to OUT probe at
   12400 mm → smaller wind-generated ripples at OUT. Data: loose230
   OUT wind = 1.016 mm; loose300 OUT wind = 0.817 mm.

### MEDIUM

5. **CH05 figure review**: user re-ran main_save_figures.py earlier today
   after the recompute; new CH04/CH05 PDFs exist in `output/FIGURES/`.
   They haven't been reviewed cell-by-cell yet.

6. **probe_height_analysis.py → CH04 §3b figure**. Currently text-only
   analysis. Would fill the CH04 §3b placeholder and justify the
   height100/lowrange hardware choice with data. Needs new plotting code.

7. **Open question: fullpanel vs reversepanel damping**. See
   `memory/open_question_fullpanel_vs_reversepanel.md`. 61 reverse-panel
   runs exist (all Nov 2025, old probe config). Not in meta_results.
   Could be a thesis appendix or future paper.

### LOW / BACKGROUND

8. **Rolling RMS stationarity** on nowave+fullwind runs — rubber-band
   splash detection. From the probe_height_wind_findings.md TODOs.

9. **PSD comparison across conditions** (cond1 vs cond4 at same probe).
   Question: does probe height/range affect spectral shape of wind-wave
   field, or only amplitude?

10. **Per-folder `wind_rms_{pos}` pipeline column**. One scalar per folder
    from nowave+fullwind runs. Would enable per-folder first-motion
    threshold in `RampDetectionBrowser`.

11. **Sub-bin FFT interpolation** (parabolic peak fit) in
    `compute_amplitudes_from_fft`. Would reduce individual-AFFT bias to
    <1%. Not needed for OUT/IN (already bias-robust per the 367-run
    test). Only matters if the thesis ever quotes individual AFFT values
    in mm. See `memory/methodology_fft_peak_bin_bias.md`.

---

## Scratch-folder inventory (what's promoted vs what's not)

| Scratch artifact | Status | Chapter slot |
|---|---|---|
| `fft_peak_bias_outin_impact.*` | ✅ promoted | CH04 §4b |
| `mansard_funke.*` | ✅ promoted | CH04 §4c |
| `sw_correction.*` | ✅ promoted | CH04 §4d |
| `sliding_afft_fullwind_sweep.*` | ✅ promoted | CH04 §4e |
| `sliding_afft_15hz_02v_zoom.*` | Companion; not separately promoted | — |
| `sliding_afft_per240.pdf` | Superseded by sweep; kept for history | — |
| `mooring_comparison.*` | Already in CH04 §3c (custom cell) | CH04 §3c |
| `sw_probe_ratio.*` | Superseded by Mansard-Funke | — |
| `mf_windonly.*` | Supplementary (wind-only MF) | Appendix candidate |
| `mooring_sw_bias.py` | Supplementary | Appendix candidate |
| `probe_height_analysis.py` | Text-only; no figure yet | CH04 §3b (TODO) |
| `recon_before_after.py` | Algorithm diagnostic | Methodology reserve |
| `recon_check.py` | Algorithm diagnostic | — |
| `reconstruct_all3.py` / `reconstruct_variants.py` | Algorithm experiments | — |
| `period_overlay.py` | Exploratory | — |
| `quicklook_1600hz.py` | Bad-data debug | Do not promote |

---

## Memory index (written/updated this session)

- `memory/MEMORY.md` — START HERE updated to reflect this session's state
- `memory/methodology_fft_peak_bin_bias.md` — NEW, primary methodology finding
- `memory/open_question_per240_afft.md` — resolved (was open)
- `memory/open_question_fullpanel_vs_reversepanel.md` — NEW
- `memory/known_baddata_ultrasound_16hz.md` — NEW
- `memory/project_tasks.md` — updated priority list
- `analysis_scratch/handoff_2026-04-17.md` — earlier handoff (preserved)
- `analysis_scratch/handoff_2026-04-17_final.md` — THIS FILE

---

## Reminders for the next agent

- Don't commit `.claude/worktrees/...` metadata — leave untracked.
- The user prefers PDFs only (no PNGs). When promoting new scratch figures,
  use the pattern from `mansard_funke.py` / `sw_correction.py` /
  `sliding_afft_fullwind_sweep.py`: save PDF + write `.tex` stub directly.
- `main_save_figures.py` for scratch-driven figures uses lightweight
  documentation cells (file-existence check + print). The scratch script
  is authoritative for the figure itself.
- When a previous agent's findings look "too confident" or invoke an
  unlisted physical mechanism, **check by asking the user or inspecting
  the raw signal** before trusting. Today's walk-through caught multiple
  fabricated explanations that sounded plausible.
- `meta_results` is a 2-folder subset (both `-lowrange`); CH05 headline
  results are insulated from cond3 and older-probe-config data.
- The user is walking a thesis to completion. Prefer small, reviewable
  steps and commit often.
