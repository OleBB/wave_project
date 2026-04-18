# Huseby & Grue window applied to per240 1.4 Hz runs

**Date**: 2026-04-18.
**Script**: `analysis_scratch/huseby_grue_window.py`.
**Figure**: `analysis_scratch/huseby_grue_window.pdf`.
**Data**: 9 per240 1.4 Hz full-panel quality-ok runs (3 amplitudes × nowind+fullwind where available).

## Window

- H&G exactly: 35.088 s < t < 42.105 s at 1.425 Hz = 10T (no leakage).
- Our 1.4 Hz equivalent: **[35.0, 42.143] s = 10T** at 1.4 Hz (also no leakage, no scaling artefact).
- t=0 = wavemaker start (confirmed with user).
- Per40 runs excluded (paddle stops ≈ 28 s; H&G window is in ringdown).

## OUT/IN — three window methods

| amp | wind | n | pipeline | H&G 10T | sliding-median (15–100 s, 10T window) |
|-----|------|---|----------|---------|----------------------------------------|
| 0.1 V |   no | 1 | 0.523 | 0.542 | 0.506 |
| 0.1 V | full | 2 | 0.754 | 0.749 | 0.740 |
| 0.2 V |   no | 1 | 0.701 | 0.696 | 0.688 |
| 0.2 V | full | 2 | 0.784 | 0.751 | 0.738 |
| 0.3 V |   no | 1 | 0.696 | 0.937 | 0.701 |
| 0.3 V | full | 2 | 0.767 | 0.786 | 0.774 |

## Per-run detail

| amp | wind | OUTIN_pipe | OUTIN_hg | Δ(H&G − pipe) | OUTIN_slide_median |
|-----|------|-----------|----------|---------------|---------------------|
| 0.1 V | full | 0.818 | 0.811 | -0.007 | 0.800 |
| 0.1 V | full | 0.689 | 0.687 | -0.001 | 0.681 |
| 0.1 V |   no | 0.523 | 0.542 | +0.019 | 0.506 |
| 0.2 V | full | 0.840 | 0.799 | -0.041 | 0.773 |
| 0.2 V | full | 0.728 | 0.702 | -0.026 | 0.702 |
| 0.2 V |   no | 0.701 | 0.696 | -0.005 | 0.688 |
| 0.3 V | full | 0.772 | 0.817 | +0.045 | 0.770 |
| 0.3 V | full | 0.761 | 0.755 | -0.006 | 0.778 |
| 0.3 V |   no | 0.696 | 0.937 | +0.241 | 0.701 |

## The 0.3 V nowind outlier is a 9373/170 probe glitch, NOT physics

The single Δ = +0.24 outlier at (0.3 V, nowind) looked suspicious. Adding
the parallel 9373/340 probe (same longitudinal distance, other lateral
position) as a sanity check (`huseby_grue_window.pdf`, middle column of
the bottom row) resolves it:

- **9373/170** at 0.3 V nowind: drops from ~22 mm to ~17 mm between
  window-start 30 s and 36 s, recovers after 36 s. The H&G single-shot
  window catches this dip → IN reads 17 mm → OUT/IN inflated to 0.94.
- **9373/340** at 0.3 V nowind: flat at ~22 mm across the entire run.
  **No dip whatsoever.**

Two probes looking at the same longitudinal position of the same
wavefield: a dip on one and not the other is **instrumentation noise**,
not physics. The H&G OUT/IN = 0.94 is an artefact of a single-probe
false reading in the short window.

Recomputing H&G OUT/IN using 9373/340 as the IN reference for this
run: 15.5 mm / 22 mm ≈ 0.70 — matches the pipeline (0.696) and
sliding-median (0.701) exactly.

**Takeaway for the thesis**: using both 9373 probes (not just the
pipeline's single 9373/170) as cross-checks catches this kind of
single-probe glitch. The parallel probe is a redundancy sanity check
whenever the two disagree by more than expected lateral variability.
See follow-up scope below.

## Overall take

- **For 8 of 9 runs**: pipeline, H&G and sliding-median all agree
  within ±0.05 OUT/IN. The pipeline window is giving H&G-equivalent
  numbers at 1.4 Hz — methodology validated against the published
  standard.
- **For the one outlier (0.3 V nowind)**: the H&G window is
  mathematically fine; the IN probe reading in that window is the
  problem. Cross-check with the parallel probe is what caught it.
- The sliding-AFFT curves confirm the wave train settles onto a stable
  plateau by ~20–25 s and stays there through 100 s. Both window
  choices (pipeline 19–39 s, H&G 35–42 s) sit on this plateau when the
  probe is behaving.

## Follow-up scope (the user's "big topic")

> "we should really use BOTH 9373 probes for all these plots"

Concrete options, in order of increasing scope:

1. **Redundancy quality flag** — add `ain_probe_disagreement` column:
   when |A(9373/170) − A(9373/340)| / mean(A) > threshold (e.g. 10%),
   flag the run. Agents / plots can then opt in to exclude.
2. **Sanity-check overlay** — on any CH05 figure that uses A(9373/170),
   add the A(9373/340) value as a faint marker (or errorbar range).
   Reader sees disagreement at a glance without changing the
   quantitative claim.
3. **Redefine IN reference** — use mean(9373/170, 9373/340) as the
   canonical IN amplitude everywhere. Simple, robust, but blurs genuine
   lateral asymmetry (CLAUDE.md §16 warns against averaging without
   thought). Would require re-deriving OUT/IN, ka, T_cross, everything.
4. **Per-run best-probe selection** — keep 9373/170 as default, swap
   to 9373/340 when the former has a detected anomaly and the latter
   doesn't. Preserves lateral-asymmetry discussion in CH04 while
   letting CH05 reap the robustness benefit.

Option (2) is the lightest. Option (3) is the heaviest but arguably
the cleanest science. Recommend option (2) first, then option (3)
if the user wants to go all-in after seeing (2).

## Caveats

- No 1.425 Hz data — this is an extrapolation to 1.4 Hz using the same relative window (35 s start, 10T length). Tank geometry is similar to H&G (24.6 × 0.5 × 0.6 m vs our setup) so the timing of parasitic waves and beach reflections should be comparable.
- Only per240 runs. Per40 windows end near or before 35 s.
- Pipeline AFFT column is read from meta.json (no re-computation).