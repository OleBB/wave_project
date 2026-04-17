# Handoff — 2026-04-17 (final)

**This supersedes all earlier `handoff_2026-04-*.md` files for this date.**
The prior handoff written earlier in the day (`handoff_2026-04-17.md`)
and the first version of this file are retained for history but
`handoff_2026-04-17_final.md` (this one) is the authoritative pointer.

---

## TL;DR — 30-second orientation

- `main` is **clean and pushed** through commit `0d54525`. Worktree
  metadata in `.claude/` is correctly untracked. Nothing pending.
- **Pipeline is fresh**: `waveprocessed/run_20260417_145335_force.log`.
  646 runs, 25 datasets, 9 RECON ABORTs — all at ≥1.8 Hz or 0.8 Hz
  per11 (all out of scope). `meta_results` = two `-lowrange` folders,
  cond4 only.
- **Headline methodology result**: FFT peak-bin bias discovered, tested
  on 367 real runs, shown to cancel in OUT/IN (mean |Δ|/OUT/IN = 0.46%).
  Thesis OUT/IN values are safe as-is.
- **Three new CH04 methodology figures** + the rewrite of a previously-
  misleading findings doc + two diagnostic tools (quality-flag audit,
  rolling-RMS stationarity).
- **Immediate next agent priorities** (ordered, see details below):
  1. CH05 figure review pass (user + agent together)
  2. probe_height_analysis → CH04 §3b figure (fills placeholder)
  3. Promote open-question follow-ups if the user picks them

## How to get your bearings

If you have 5 minutes, read:

1. `memory/MEMORY.md` START HERE (pipeline + current direction)
2. `memory/session_2026-04-17.md` (today's session log, chronological)
3. `analysis_scratch/handoff_2026-04-17_final.md` (this file)

If you have 30 minutes, also read:

4. `memory/methodology_fft_peak_bin_bias.md` (the critical methodology finding)
5. `memory/project_tasks.md` (full priority list)
6. `analysis_scratch/probe_height_wind_findings.md` (rewritten today — example of what to watch for)

## End goal (from CLAUDE.md)

**Does wind increase or decrease how much of an incoming wave is
transmitted past the FPV panel geometry?** Measured via `OUT/IN (FFT)`
at the paddle frequency. Primary result lives in CH05 §1, §3, §4 of
`main_save_figures.py`.

---

## What got done today (thematic summary)

### 1. Methodology: FFT peak-bin bias

Started from an interrupted 04-17 PDF that suggested per240 fullwind
1.3 Hz runs had a "depression" in AFFT at the pipeline window. After
a sweep + zoom + 367-run impact test:

- **The "depression" was a methodology artifact**: comparing a sliding-
  FFT mean to a single-FFT value for narrow-band signals.
- **The real issue is FFT peak-bin sinc-attenuation** when the bin
  grid doesn't align with the paddle's drifted frequency. Up to ~40%
  under-read on individual AFFT values.
- **OUT/IN ratio is robust**: 0.46% mean relative error over 367 real
  runs. The bias cancels because IN and OUT use matching analysis
  windows → same bin grid → same bias.
- Artifacts: `memory/methodology_fft_peak_bin_bias.md`,
  CH04 §4b figure `plot_fft_peak_bias_cancellation` in plotter.py,
  three scratch scripts (`sliding_afft_*`, `fft_peak_bias_outin_impact`).

### 2. Pipeline verification

Recompute with the new 8804 end calibration from 2026-04-16b. Clean
result: `mstop330-run3` no longer RECON ABORTs, 9 remaining ABORTs all
out of scope. `meta_results` (cond4 only) uninfected.

### 3. Three new CH04 methodology figures

| Slot | Figure | Source | Result |
|---|---|---|---|
| §4b | FFT peak-bin bias cancels in OUT/IN | plotter.py + CSV | 0.46% mean error |
| §4c | Mansard-Funke reflection R | `analysis_scratch/mansard_funke.py` | R ≈ 0.05–0.07 |
| §4d | SW correction test (negative) | `analysis_scratch/sw_correction.py` | R ≪ 0.20; no correction needed |
| §4e | Sliding AFFT stability at IN | `analysis_scratch/sliding_afft_fullwind_sweep.py` | Signal stable across run |

**Promotion pattern** (for future scratch→thesis work):

- Scratch script writes PDF + `.tex` stub directly into
  `output/FIGURES/` and `output/TEXFIGU/`. User prefers PDF-only (no
  PNG) — PDF serves both quick-view and thesis.
- `main_save_figures.py` has a lightweight documentation cell that
  verifies the figure exists and points at the scratch script for
  regeneration. No actual work done there — the scratch script is
  authoritative.

### 4. Refactor: damping_grouper collapse is opt-in

`damping_grouper` and `damping_all_amplitude_grouper` in
`wavescripts/filters.py` previously mapped `PanelCondition` {full,
reverse} → "all" via `PanelConditionGrouped`. This silently averaged
full and reverse panels together — a potentially physically-distinct
pair.

Now both take `collapse_panels: bool = False`. Default: separate
groups. Opt-in True: old behaviour, with a loud warning.

**Thesis impact: none.** All CH05 callers use `meta_results` which has
zero reverse-panel rows. CH04 §3c pre-filters to `PanelCondition == "full"`.
This was a no-op on the thesis path — change removes a latent landmine
and enables a future fullpanel-vs-reversepanel study.

### 5. probe_height_wind_findings.md rewrite

The 2026-03-30 autonomous agent had produced a findings doc with
several fabricated explanations. User and assistant walked through it
and rewrote it. Key corrections:

- **No physical probes were ever swapped.** The same four ultrasound
  probes (numbered 1–4) were used throughout the entire experiment;
  only positions changed.
- **h272 is noisiest is EXPECTED physics** (longer acoustic path → more
  drift / attenuation). The ordering h272 > h136 > h100 is predicted.
- **cond3 (h100/high) was a user-mode-switch mistake**, and is where
  the P2-malfunction runs originate.
- **Wind field IS consistent across sessions** (measured fan 5.9–6.0 m/s,
  water ±0.5 mm). Probe reports differ because of range-mode + height
  combination, not wind variation.
- **New insight: mooring length → post-panel fetch → OUT ripple amplitude**
  (undocumented physics). Longer mooring → panels extend further back →
  shorter free-water fetch from back of panel to OUT probe → smaller
  ripples. Evidence: loose230 OUT = 1.016 mm, loose300 OUT = 0.817 mm.
  Now in `memory/physics_wavetank_mooring_fetch.md`.
- **"SNR"** in the original doc was dynamic range, not SNR. The
  thesis-relevant SNR is `A_paddle / A_wind` at OUT, not `A_wind /
  σ_stillwater`.

### 6. Two diagnostic tools (end of day)

**Per-run quality-flag audit** — `analysis_scratch/quality_flag_audit.py`
+ `.md` + `.csv`. Enumerates each flagging layer in
`processor.py::_write_quality_flags`, tables which runs each layer
flagged and the triggering metric, plus a near-threshold diagnostic
section. 27/646 runs flagged (23 dropout_critical, 3 in_probe_low_snr,
1 probe_malfunction_critical).

**Rolling-RMS stationarity check** for fullwind+nowave runs —
`analysis_scratch/rolling_rms_stationarity.py` + `.pdf` + `.csv` +
`.md`. Tests if the wind-background signal is stationary. Surprise
finding: detects **three regimes**, not just rubber-band splash:

- Severe sensor-glitch (above_50 20260307 / 20260314 runs, max/med 4–5+,
  ±60 mm spikes in η — NOT physical)
- Likely rubber-band (some loose230, max/med 1.5–2.5)
- Normal wind variation (borderline CV ~ 0.25)

Recommendation: exclude severe-glitch runs from any wind-background
averaging.

---

## Open items for next agent — ranked

### HIGH priority / actionable now

**H1. CH05 figure review pass.**
`output/FIGURES/ch05_*.pdf` files were regenerated after the recompute
and the grouper refactor. They haven't been cell-by-cell reviewed
yet. Expected order of scrutiny:
- `ch05_damping_freq.pdf` (the headline figure — OUT/IN vs frequency,
  split by wind)
- `ch05_damping_wind_delta_*.pdf` (the wind-effect delta)
- `ch05_damping_ka_full.pdf` (ka axis version)
- `ch05_damping_scatter_full.pdf` (amplitude dependence)
- `ch05_swell_scatter.pdf` and `ch05_reconstructed.pdf`

For each: does the shape match physical expectations? Are errorbars
present? Are the 1.6 Hz bad-ultrasound runs polluting anything? Does
the damping-vs-frequency pattern make sense?

Needs user's eye + agent editing of captions / axes / labels as
feedback arrives.

**H2. probe_height_analysis.py → CH04 §3b figure** (fills placeholder).
Currently text-only analysis. Needs new plotting code to turn the
stillwater-noise-floor vs condition numbers into a visual. Proposed:
2×1 panel figure — (a) noise floor per probe × per condition, (b) wind
amplitude per probe × per condition. Uses the same scratch-driven
promotion pattern as the new §4c–§4e figures.

**H3. 1.6 Hz bad-ultrasound filter** (user-blocked).
User knows this data is bad; needs to provide the specific date(s) so
a `quality_flag` rule or folder exclusion can be applied. See
`memory/known_baddata_ultrasound_16hz.md`. Once the date is known,
it's ~10 min of work.

### MEDIUM priority

**M1. Fullpanel vs reversepanel damping investigation.**
61 reverse-panel runs exist (all Nov 2025, old probe config). Question
design in `memory/open_question_fullpanel_vs_reversepanel.md`. Bigger
investigation — multiple sessions. The mooring-geometry change when
reversing is a confounder to work around.

**M2. Apply rolling-RMS stationarity findings**
to `probe_height_wind_findings.md` Finding 2 (wind amplitude per
condition). Question: were the above_50 glitchy runs included in the
10.575 ± 0.425 mm average? If so, re-compute excluding them.

**M3. Update `reflection_analysis.md`**
to note the mooring-fetch mechanism affects the OUT-probe wind floor
(see `physics_wavetank_mooring_fetch.md`). Current doc treats wind at
OUT as mooring-independent — partially wrong.

### LOW priority / background

**L1. Sub-bin parabolic interpolation** in `compute_amplitudes_from_fft`.
Would reduce individual-AFFT bias to <1%. Not needed for OUT/IN (already
bias-robust per today's finding). Only matters if thesis ever quotes
individual AFFT values in mm.

**L2. PSD comparison across conditions** (cond1 vs cond4 same probe).
Does probe height/range affect spectral shape of wind-wave field, or
only the amplitude reported?

**L3. Per-folder `wind_rms_{pos}` pipeline column**. One scalar per
folder from nowave+fullwind runs. Would enable per-folder first-motion
threshold in `RampDetectionBrowser`.

**L4. 9373 Day 2 eyeballing decision** (user-only). Apply 37/37/36/35 s
or keep current 39/38/36/36 s for `_SNARVEI_END_CALIB["9373"]`? See
`analysis_scratch/snarvei_eyeballing.md`.

---

## Where things live — quick index

### Thesis figures (ready, in `output/FIGURES/`)

| Chapter section | Figure file |
|---|---|
| CH04 §1 probe noise floor | `ch04_probe_noise_floor_group{0,1,2,3}.pdf` |
| CH04 §3 parallel probes | `ch04_parallel_ratio{,_scatter}.pdf` |
| CH04 §3c mooring comparison | `ch04_mooring_comparison.pdf` |
| CH04 §3d sound speed | `ch04_sound_speed.pdf` |
| CH04 §4-5 td vs FFT | `ch04_td_vs_fft{,_scatter}.pdf` |
| **CH04 §4b FFT peak-bin bias** | `ch04_fft_peak_bias_cancellation.pdf` NEW |
| **CH04 §4c Mansard-Funke** | `ch04_mansard_funke_reflection.pdf` NEW |
| **CH04 §4d SW correction** | `ch04_sw_correction_test.pdf` NEW |
| **CH04 §4e Sliding AFFT** | `ch04_sliding_afft_stability.pdf` NEW |
| CH04 §5 timeseries | `ch04_timeseries_overview.pdf` |
| CH04 §6 first arrival | `ch04_first_arrival.pdf` |
| CH04 §7 wave stability | `ch04_wave_stability.pdf` |
| CH04 §8 lateral nowind | `ch04_lateral_nowind{,_scatter}.pdf` |
| CH04 wind characterisation | `ch04_wind_psd.pdf`, `ch04_wind_snr.pdf`, `ch04_wind_reflection.pdf`, `ch04_fft_wave.pdf` |
| CH05 §1 damping vs freq | `ch05_damping_freq.pdf`, `ch05_damping_freq_all_{10,20,30}V.pdf` |
| CH05 §2 damping vs amp | `ch05_damping_scatter{,_full}.pdf` |
| CH05 §3 wind delta | `ch05_damping_wind_delta_{all,full}_{10,20,30}V.pdf` |
| CH05 §4 damping vs ka | `ch05_damping_ka_full.pdf` |
| CH05 §5 swell scatter | `ch05_swell_scatter.pdf` |
| CH05 §6 reconstructed | `ch05_reconstructed.pdf` |
| CH04 placeholders | `ch04_stillwater_timing.pdf`, `ch04_probe_height.pdf` |

### Diagnostic tools (scratch)

| Tool | Script | Output |
|---|---|---|
| Quality-flag audit (per run) | `analysis_scratch/quality_flag_audit.py` | `.md` + `.csv` |
| Rolling-RMS stationarity | `analysis_scratch/rolling_rms_stationarity.py` | `.md` + `.pdf` + `.csv` |
| Sliding AFFT sweep | `analysis_scratch/sliding_afft_fullwind_sweep.py` | `.md` + `.pdf` + `.csv` |
| Sliding AFFT zoom (1.5 Hz) | `analysis_scratch/sliding_afft_15hz_02v_zoom.py` | `.md` + `.pdf` |
| FFT peak-bias OUT/IN impact | `analysis_scratch/fft_peak_bias_outin_impact.py` | `.md` + `.csv` + `.png` |
| Mansard-Funke | `analysis_scratch/mansard_funke.py` | scratch PDF + thesis PDF + stub |
| SW correction | `analysis_scratch/sw_correction.py` | same pattern |

### Memory (key references)

| File | Purpose |
|---|---|
| `memory/MEMORY.md` | START HERE — always current state |
| `memory/methodology_fft_peak_bin_bias.md` | Headline methodology finding |
| `memory/physics_wavetank_mooring_fetch.md` | Today's new physics insight |
| `memory/open_question_fullpanel_vs_reversepanel.md` | Open research question |
| `memory/open_question_per240_afft.md` | Resolved (was the starting thread) |
| `memory/known_baddata_ultrasound_16hz.md` | The 1.6 Hz chaos context |
| `memory/project_tasks.md` | Full priority list |
| `memory/session_2026-04-17.md` | Today's session log |

---

## Working patterns for next agent

- **User prefers PDF-only** (no PNG) for figures. When promoting a
  scratch figure, use the pattern from
  `mansard_funke.py` / `sw_correction.py` / `sliding_afft_fullwind_sweep.py`:
  save PDF to `analysis_scratch/{name}.pdf` (scratch quick-view) AND to
  `output/FIGURES/ch04_{name}.pdf` (thesis), plus write a `.tex` stub
  in `output/TEXFIGU/`.
- **Thesis-integrated plotter functions** (in `wavescripts/plotter.py`)
  are for reusable plots that take `combined_meta` + `plotvariables`.
  Use this when the figure is data-driven with logic worth centralising.
  For one-off methodology figures, keep the logic in a scratch script.
- **Don't commit `.claude/worktrees/...` metadata** — leave untracked.
- **`git add -u` is safer than `git add .`** in this repo because
  `.claude/` should not be staged.
- **Commit often with descriptive messages**; the user has expressed
  appreciation for commit-as-you-go.
- **When an earlier doc looks confident without backing its claims
  with the raw signal or a log check, treat it with suspicion.**
  Today's walk-through caught multiple fabricated explanations that
  sounded plausible.
- **`meta_results` is a 2-folder subset** (both `-lowrange`, both
  cond4). All CH05 headline figures use this. CH04 figures use
  `combined_meta` (all folders) for broader methodology characterisation.
- **When in doubt, ask the user for experimental context** before
  concluding a physical mechanism. Many "surprising" results are
  artefacts the user can identify immediately.

---

## Timestamp

- Final push commit: `0d54525` (rolling-RMS stationarity)
- Session started picking up work at the state of `dcbacd0` (previous
  day's "worksession done" commit)
- Total commits today: 17
- All on `origin/main`. Nothing pending.
