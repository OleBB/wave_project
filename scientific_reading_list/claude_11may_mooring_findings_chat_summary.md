## Discussion summary — chain of investigation

**Q1: The visible outlier in `ch05_damping_overmooring_scatter` at A3, 1.4 Hz nowind looks too high. What is it?**

Pulled the four matching above_50 A3 1.4 Hz runs. Found two nowind runs:
- mar 12 per40: K_t = 0.705 (normal)
- mar 13 per240: K_t = 0.831 (the outlier — IN-probe 9373/340 reads 14.59 mm vs mar 12's 21.46 mm)

**Found:** the outlier is `20260313/.../fullpanel-nowind-amp0300-freq1400-per240-mstop30-run1.csv`. K_t,probe>1 dropout filter rejected the same probe at 1.005 (just over threshold), exposing the rest of the unreliable run.

**→ Q2: What in meta.json flags this run as bad?**

Cross-checked diagnostic columns against siblings. `cut_samples_9373/340 = 45` on the outlier, `0` on both siblings. Wider check showed only 5 runs across the dataset have `quality_flag = "ok"` AND `cut_samples > 0` on a canonical IN/OUT probe.

**Found:** `cut_samples > 0` is the clean discriminator — catches exactly the failed runs without false positives.

**→ Q3: What does Amplitude (LS) say for this run?**

LS sinusoid-fit amplitude at f_paddle is bin-grid-independent and robust to gaps. Compared LS / PSD / percentile / FFT per probe.

**Found:** LS, PSD, and percentile all read normal (~21 mm for 9373/340); only FFT drops to 14.59 mm. The 45 `cut_samples` broke integer-cycle coherence in the H&G-snapped FFT window. **K_t under LS = 0.701 — same as the mar 12 sibling.** The wave wasn't broken; the FFT measurement was.

**→ Q4: Should we use LS for the canonical K_t pipeline-wide?**

Discussed pipeline-change cost. Settled on a surgical override rule.

**Found:** Per-row 3-way override — if `|Kt_FFT − Kt_LS| > 0.05` AND `|Kt_FFT − Kt_PSD| > 0.05` AND `|Kt_LS − Kt_PSD| < 0.05`, swap to Kt_LS. Asymmetric in FFT's favour (FFT stays unless both alternatives disagree with it AND agree with each other). Fires on exactly 3 of 399 runs across the scatter scope (clean separation: next-nearest gap is 0.03). Implemented in 3 scripts + LaTeX stub provenance + lower-left "n = X data estimert med LS" annotation.

**→ Q5: Apply the same fix to the wired-in thesis figures?**

Applied LS override + annotation to all 6 scatter figures (3 k-axis + 3 ka-axis). Wired the 2 missing k-axis under/over views into `main_save_figures.py` with FIGURE_INDEX + FIGURE_CAPTIONS + DELEG cell.

**Found:** Six scatter figures now uniformly use the override; per-view counts (n=1, 2, 3) add correctly across the under/over/all views. The k-axis parent `ch05_damping_all_data_scatter` also got the encoding promoted to mooring×panel categorization (instead of hardware-based), matching its ka sibling.

**→ Q6: Will LS also fix the 1.6 Hz fullwind spike in `ch05_damping_freq_overmooring_A3`?**

Compared FFT vs LS K_t for the 5 above_50 A3 1.4 Hz runs after dropout.

**Found:** No. FFT and LS agree to 4 decimals at every cell mean. The wide errorbar at 1.6 Hz fullwind (n=2, σ=0.049) is **real measurement variability** between two genuinely-different per40+per240 sessions, not a method artefact. Cosmetic styling revert (basic markersize, tiny coloured errorbars) addressed the visual alarm without changing the data.

**→ Q7: At A2 1.4 Hz over-mooring, nowind sits ABOVE fullwind — real or artefact?**

Pulled all 4 cells. FFT/LS/PSD agree on every run; no cut_samples; per40-vs-per240 disagree by +0.081.

**Found:** Real. All-hardware (h272/high cond1). The mar 14 per240 run was acquired in a descending-frequency sweep with a 51-min pause and a `nowave_control` predecessor (the only one across all 24 A2 above_50 runs). Different session shapes produce different K_t at this freq.

**→ Q8: Does A2 differ more by day-day than the other amps?**

ANOVA on K_t residuals grouped by date.

**Found:** **No, A2 has the SMALLEST date-effect** (η² = 0.25) of the three amplitudes. A1 has the largest (η² = 0.43, p = 0.04). What looks like A2 instability is small-n + session-context variance, not day-condition variance.

**→ Q9: Is A2's K_t variance driven by IN or OUT probe?**

Decomposed cell-level CV into A_IN_relstd vs A_OUT_relstd.

**Found:** Under fullwind, **OUT probe is the bigger contributor** (relstd 5–15% vs IN 4–7%). Parallel IN probes also disagree laterally (median 2.5%, max 19.6%) — wind drives lateral wave-field non-uniformity that the IN-pair averaging only partially cancels.

**→ Q10: Does the over-mooring deserve its own aggregated `damping_freq` plot, or is the scatter more honest?**

Counted cell n's: most over-mooring cells have n=1–2.

**Found:** The aggregated triplet is misleading. Recommended dropping it in favour of the scatter, which is statistically self-defensive. The user agreed.

**→ Q11: The over-mooring scatter also has blue-above-red at low freqs — is the wind effect inverted across the full freq range?**

Tabulated ΔK_t per (amp × freq) at above_50 across 0.5–1.9 Hz.

**Found:** **Wind effect sign flips at ~1.5 Hz at above_50.** Below 1.5 Hz wind reduces K_t (negative Δ); above 1.5 Hz wind strongly enhances K_t (Δ up to +0.32 at 1.9 Hz). Transition zone is real and physically interesting.

**→ Q12: Is the IN/OUT probe labelling correct? Could the over-mooring transition be an averaging-scheme artefact?**

Cross-checked 275 above_50 rows against PROBE_CONFIGS expectations.

**Found:** All 275 labels match. Two probe-geometry configs contribute (`nov_normalt_oppsett` with IN single + OUT paired, `march2026_better_rearranging` with IN paired + OUT single). The 28 `march2026_rearranging` runs use OUT=11800/250 which is only valid for nopanel runs — but those are not in the scatter scope anyway.

**→ Q13: Split over-mooring scatter by config to see if averaging scheme matters.**

Built a config-split scatter (canon vs Nov 2025).

**Found:** At 1.3 Hz where both configs overlap, they agree on direction (wind enhances slightly). **The 1.5 Hz transition is canon-only data — the Nov 2025 era never measured at 1.4+ Hz.** So the transition is a real canon-era finding, not a probe-geometry-averaging artefact.

**→ Q14 (user's notebook context): "above_50" tag conflates THREE physical changes between Nov and Canon — strikk stiffness (6 cm vs 16 cm), panel orientation (reverse vs full), and probe averaging.**

Re-split the scatter into 4 explicit mooring conditions:
- canon_loose300 (under, 30 cm strikk)
- canon_loose230 (under, 23 cm strikk)
- above_semi_stiff (over, 16 cm strikk, full panel, canon era) 

#!HEY, yes HEY YOU! I JUST CHANGED THIS TERM... (IT SAID CANON LOOSE, BUT ITS NOT CANON AND ITS NOT LOOSE ENOUGH.)
- nov_above_stiff (over, 6 cm strikk, reverse panel, Nov 2025)

**Found:** 4-way scatter with vivid palette. Mooring height (over/under) is the single biggest visual axis — over-water moorings sit ~0.20 K_t below under-water at 1.5+ Hz. Stiffness within under-water (23 vs 30 cm) matters very little. Nov stiff has only two frequencies measured; not enough to characterize.

**→ Q15: At 1.3 Hz (densest dataset, 108 runs across 4 moorings), how does the wind effect vary across conditions?**

Built per-amplitude breakdown.

**Found:** **The wind effect at 1.3 Hz is strongly amplitude-dependent and mooring-dependent.** Under-water moorings show wind enhancement that decays with amplitude (+0.13 at A1 → +0.01 at A3). semi-stiff above-water (loose 16 cm) shows ≈ zero effect at all amps. Nov stiff above-water (6 cm + reverse) shows sign flip with amplitude (+0.09 at A1 → −0.03 at A3). The "wind enhances K_t at 1.3 Hz" headline is primarily an A1 effect.

**→ Q16 (user's intuition): "the OUT probe is MORE sheltered with longer moorings"**

Tested wind-induced ΔA_IN% vs ΔA_OUT% per (mooring × amp).

**Found:** Not a simple strikk-length effect. The asymmetry is **over-vs-under-water**: under-water moorings, wind boosts A_OUT (+5–6 %) and leaves A_IN flat; over-water moorings, wind boosts A_IN (+3–5 %) and leaves A_OUT flat (panel above water casts air shadow). Universal pattern: **the IN side gets the fetch, the OUT side is sheltered**, just expressed differently for over- vs under-water geometries.

**→ Q17: What do the broader signal-quality diagnostics in meta say?**

Checked LS Residual RMS, wave_stability, period_amplitude_cv across all 4 moorings at 1.3 Hz.

**Found:** **OUT-shelter hypothesis confirmed universally.** RRMS_IN increases by ~6× under wind (broadband wind chop); RRMS_OUT only ~2×. wave_stability_IN drops by 0.09–0.17; wave_stability_OUT drops by < 0.01. CV_IN goes from 3% to 18–25%; CV_OUT stays at 4–5%. **The panel acts as a frequency-selective filter** — wind broadband noise (peak 3–5 Hz) stays on IN; only paddle-band signal makes it through to OUT.

**→ Q18 (user's mechanism): "the over-mooring geometry can be LIFTED by wind. That's why mooring point was moved down."**

Tested with ε = ΔK_t / (1 − K_t_nowind) — "fraction of remaining blockage wind removes". If lift is constant in frequency, ε should be flat for over-mooring.

**Found:** **Under-water moorings have ε ≈ 0.31 ± 0.04 across 1.3–1.7 Hz** (rock-steady "universal" wind-enhancement coefficient). **semi-stiff above-water (16 cm strikk) has ε freq-dependent: ≈ 0 below 1.5 Hz, jumps to +0.25 at 1.5 Hz, converges to ε ≈ 0.32 at 1.6+ Hz.** The transition at 1.5 Hz matches a panel-resonance prediction (m=4.55 kg + estimated k≈400 N/m → f_n = 1.5 Hz). Refined the lift hypothesis: panel is **wave-compliant below f_n** (moves with waves → effectively transparent → wind can't enhance), **rigid-blocker above f_n** (wind removes ~32% of remaining blockage just like under-water).

**→ Q19: Restrict the ε(f) figure to the thesis band 1.2–1.6 Hz for the cleanest view.**

Re-rendered with narrow x-range.

**Found:** In 1.2–1.6 Hz, the over-vs-under gap closes monotonically from +0.31 at 1.3 Hz to +0.11 at 1.6 Hz. Visually clean — single image carries the transition story.

**→ Q20 (user's intuition): "the low freqs are dominated by wind noise at the IN frequency. Their waves are very long."**

Tested wind-induced ΔA_IN at the paddle bin (FFT) vs total envelope (percentile) across frequencies.

**Found:** **At the FFT paddle bin, wind contribution is largest at HIGH freq (+8 mm at 1.9 Hz/A3), not low freq.** But at the percentile envelope, **wind adds 5–8 mm of chop to A_IN at every frequency**. The FFT correctly filters the chop out of the paddle bin; the percentile sees it all. So the user's intuition is correct in a refined form — wind adds noise on top of waves, the panel filters it, and the *measurement method* determines whether you see this.

**→ Q21: Try the percentile and phase-locked metrics for K_t.**

Built the 3-method ε comparison for canon_loose230.

**Found:** **FFT and LS agree closely** (the panel-resonance story is method-independent within the paddle-frequency family). **Percentile K_t drops dramatically under wind at low freq** (ε = −1.05 at 1.2 Hz) — quantifies the panel's wind-wave filter function. **Phase-locked K_t produces K_t > 1 cells under wind** (artefact: wind-perturbed upcrossings on IN scramble the cycle-detection-based sampling, undersampling A_IN_phase). Phase-locked is unreliable under wind for K_t.

**→ Final synthesis: three methodology-grade observations now visible in one figure:**

1. FFT-based K_t is the right metric for the headline "wind enhances paddle transmission" story.
2. The panel is an effective wind-wave low-pass filter (visualized by the gap between FFT and percentile lines).
3. Phase-locked K_t is broken under wind (cycle-detection failure on IN) — should be flagged in methodology.

The full chain ran from "one suspicious dot in one figure" all the way to a mechanistically-grounded panel-resonance interpretation of the over-mooring transition zone and a clear distinction between three K_t measurement methods. Every step was driven by a specific question the data exposed, and each answer generated the next question.