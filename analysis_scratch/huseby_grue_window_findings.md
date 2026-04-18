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

## Quick take (user to refine)

- If the three OUT/IN columns agree within a few percent, the pipeline window is already giving H&G-equivalent numbers at 1.4 Hz. That would confirm our methodology is consistent with the published standard.
- If the H&G window yields systematically different OUT/IN values, it points at a window-dependent bias — most likely because the pipeline window (typically 19–39 s, samples 4800–9750) is longer than 10T and may include regions before the wave train has fully settled, or it is contaminated by reflections / parasitic waves at its tail end.
- The sliding-AFFT sweep shows where A settles vs window-start time. If H&G-window AFFT sits on the stable plateau of the sweep, it is trustworthy; if it sits on a transient, it is not.

## Caveats

- No 1.425 Hz data — this is an extrapolation to 1.4 Hz using the same relative window (35 s start, 10T length). Tank geometry is similar to H&G (24.6 × 0.5 × 0.6 m vs our setup) so the timing of parasitic waves and beach reflections should be comparable.
- Only per240 runs. Per40 windows end near or before 35 s.
- Pipeline AFFT column is read from meta.json (no re-computation).