# Assumption Audit — Test Results
*Run 2026-04-14. Script: `analysis_scratch/run_assumption_tests.py`*
*Full output: `analysis_scratch/run_assumption_tests_output.txt`*

---

## Test 6a — Config applied correctly to all cached runs

**Result: CLEAN.**

All 337 standard runs have `in_position = "9373/170"`, `out_position = "12400/250"`.
Zero NaN, zero mismatches. The pipeline correctly applied `get_configuration_for_date`
to every run. No stale meta.json in the loaded dataset.

---

## Test 6b/6d — No-panel baseline for current probe config

**Result: CRITICAL GAP. No-panel data does not exist for the current probe configuration.**

The 20260307 folder contains 54 runs, all `PanelCondition == "full"`. The first day
of the current config went straight into full-panel measurements — no-panel control
was never run.

Nov 2025 folders (the only no-panel data in the entire dataset) have 52 no-panel wave
runs. All are at `in_position = "9373/250"`, `out_position = "12400/170"` — the **old**
probe configuration. Only two frequencies tested: **0.65 Hz and 1.30 Hz**.

```
Old-config no-panel OUT/IN (9373/250 → 12400/170):
  0.65 Hz:  mean = 0.983, range [0.921, 1.016], n = 24
  1.30 Hz:  mean = 0.965, range [0.825, 1.023], n = 28
```

These numbers come from measuring `A(12400/170) / A(9373/250)`. The current config
measures `A(12400/250) / A(9373/170)` — **different lateral positions for both IN and
OUT.** The 0.97 ± 0.03 result cannot be applied as a geometric correction factor.

### Raw data confirms no-panel experiments were never run in March 2026

Searched `wavedata/` for all CSVs with "nopanel" in filename:
- No nopanel files exist in any March 2026 folder (20260305 through 20260327)
- All nopanel raw CSVs are in `20251113-tett6roof-loosepaneltaped` — Nov 2025, old config

This is definitive. The gap is in the experimental design, not in data loading or caching.

### Why the Reflection 0 warning was correct but misdirected

The code at line 3022 warns "nopanel data is likely in the commented-out 20260307
folder." This was wrong — 20260307 has only full-panel data. The actual no-panel data
is in the Nov 2025 folders, but those are in the old config. The comment in the code
was presumably written from memory and is incorrect.

### Consequence

**The geometric correction factor (Attack 2) and the standing wave baseline (Attack 1)
cannot be computed from existing data.** The two paths forward are:

A. Run a no-panel control experiment in the current probe configuration, at minimum
   covering 0.65, 0.80, 1.0, 1.3, 1.5, 1.7, 1.9 Hz (the key frequencies where the
   standing wave analysis predicts alternating node/antinode positions).

B. Use the two-probe Mansard–Funke decomposition (Reflection 2 cells) on the existing
   full-panel data. This extracts A_incident directly from the two complex FFT columns
   (`"FFT 9373/170 complex"` and `"FFT 8804/250 complex"`) without needing a no-panel
   reference. It is ill-conditioned at f ≈ 1.3 Hz and 1.7 Hz (|sin(kΔx)| < 0.30),
   but those are the most important frequencies.

---

## Test 1a — Standing wave phase vs observed OUT/IN

**Result: PLAUSIBLE PATTERN, BUT UNCONSTRAINED WITHOUT EXACT PANEL POSITION.**

Full table (R = 0.20 assumed, panel centroid at 11.0 m, Δ = 1.627 m):

```
Freq   phase/π   SW factor   obs nw    obs fw   corr nw   corr fw   node?
0.60   1.904π    1.1925      0.878     0.916     1.047     1.093     antinode
0.70   0.357π    1.1019      1.021     1.255     1.125     1.382     antinode
0.80   0.887π    0.8155      0.991     1.180     0.808     0.962     NODE ↑
0.90   1.511π    1.0267      0.915     0.952     0.940     0.977     antinode
1.00   0.241π    1.1535      0.926     0.925     1.068     1.067     antinode
1.10   1.078π    0.8074      0.919     0.893     0.742     0.721     NODE ↑
1.20   0.017π    1.1998      0.836     0.822     1.003     0.987     antinode
1.30   1.050π    0.8030      0.763     0.835     0.613     0.671     NODE ↑
1.40   0.172π    1.1761      0.661     0.740     0.778     0.871     antinode
1.50   1.379π    0.9443      0.546     0.671     0.516     0.634     NODE (mild)
1.60   0.671π    0.9140      0.508     0.604     0.464     0.552     NODE (mild)
1.70   0.046π    1.1982      0.348     0.553     0.417     0.663     antinode
1.80   1.505π    1.0231      0.354     0.491     0.362     0.502     antinode
1.90   1.048π    0.8028      0.173     0.452     0.139     0.363     NODE ↑
```

### What the phase pattern predicts — and matches

Node at **0.80 Hz** (SW=0.815): IN probe measures 81.5% of incident amplitude.
The panel IS blocking 0.8 Hz waves — the ~1.0 observed OUT/IN is a partial illusion.
True T ≈ 0.81 (no-wind) and 0.96 (full-wind). Still near-unity, but the full-wind
apparent "amplification" (1.18) drops to 0.96 after correction — much more physically
reasonable.

Node at **1.10 Hz** (SW=0.807): The observed out/IN ≈ 0.92 (no-wind) is corrected to
T ≈ 0.74. This is consistent with the overall damping trend from 1.0 → 1.3 Hz, instead
of the anomalous plateau in the raw data at 1.0–1.2 Hz.

Node at **1.30 Hz** (SW=0.803): All the 1.3 Hz data (the most densely sampled frequency)
has the IN probe near a node. This means we systematically OVERESTIMATE transmission at
1.3 Hz by ~25%. The "corrected" T ≈ 0.61 (no-wind) and 0.67 (full-wind).
Wind effect at 1.3 Hz survives: +0.06 in corrected units (vs +0.07 raw).

Antinode at **1.70 Hz** (SW=1.198): The IN probe is near an antinode — measuring 20%
MORE than A_incident. We UNDERESTIMATE transmission at 1.7 Hz. Corrected T ≈ 0.42
(no-wind) and 0.66 (full-wind). The apparent steep drop at 1.7 Hz is softened: true
damping is 58% rather than the observed 65% at no-wind.
Wind effect at 1.7 Hz GROWS: +0.24 corrected (vs +0.21 raw).

### The critical uncertainty: panel position

The phase 2kΔ at 1.3 Hz changes by **2.17π for every 0.5 m error** in panel centroid
position. This means:

| Assumed panel centre | Phase at 1.30 Hz | 1.30 Hz is a... |
|---------------------|-----------------|-----------------|
| 10.5 m | 2kΔ mod 2π = −1.17π → −0.17π | antinode (SW=1.19) |
| 11.0 m | 2kΔ mod 2π = 1.05π | NODE (SW=0.80) |
| 11.5 m | 2kΔ mod 2π = 3.27π mod 2π = 1.27π | NODE (SW=0.86) |
| 12.0 m | 2kΔ mod 2π = 5.49π mod 2π = 1.49π | NODE → antinode boundary |

The current assumed value (11.0 m) makes 1.3 Hz a node. So does 11.5 m. 10.5 m makes
it an antinode. The qualitative conclusion (1.3 Hz node → OUT/IN inflated) is robust
to ~0.5 m panel position error.

But the **specific SW correction factor (0.803 vs 1.199)** is not robust: confirming
it to ±5% requires knowing the panel centroid to ±0.1 m.

### The smoking gun: corrected OUT/IN vs frequency should be smoother

If the standing wave effect is real, the corrected OUT/IN column (×SW_factor) should
show a smoother monotonic decrease with frequency than the raw OUT/IN. The raw data has
visible wiggles at 1.0–1.2 Hz (plateau/local minimum) that could be the node/antinode
fingerprint. A quantitative smoothness test (e.g. second-order differences) would
be diagnostic.

---

## Test 4a — Far-end reflection arrival time

**Result: ONLY 0.65–0.70 Hz IN PER40 RUNS IS AT RISK — and sub-1 Hz is out of scope.**

```
Freq    c_group    t_reflect   Within per40?   Within per240?
0.65    1.44 m/s    34.7 s       YES ⚠            YES ⚠
0.70    1.33 m/s    37.5 s       YES ⚠            YES ⚠
0.80    1.13 m/s    44.2 s       no               YES ⚠
0.90+   < 1.0 m/s  > 50 s       no               YES ⚠
```

(Tank length assumed 25 m — verify from lab drawings.)

At 0.65 and 0.70 Hz in per40 runs: the far-end reflection arrives at ~35–38 s. The
stable wave window ends at mstop (typically ~30 s after recording start for per40 runs).
Whether the reflection arrives BEFORE or AFTER mstop depends on the exact mstop time.

For per40 runs with mstop < 35 s: the reflection arrives after the analysis window ends.
No contamination.
For per40 runs with mstop > 37 s: the reflection may contaminate the tail of the window.

**Recommended action**: check `combined_meta["mstop_sec"]` for 0.65–0.70 Hz per40 runs.
If `mstop_sec > 35`, flag those runs as potentially reflection-contaminated.

For all frequencies ≥ 0.80 Hz in per40 runs: no contamination risk.
For per240 runs at any frequency: reflection arrives within the recording (but analysis
window is the stable wave portion only — if mstop is before t_reflect, still clean).

---

## Summary of audit findings

| Attack | Verdict | Severity |
|--------|---------|---------|
| 1: Reflection at IN probe | Plausible ±20% effect. Pattern consistent with observed OUT/IN wiggles. Cannot quantify without exact panel position (±0.1 m needed) or two-probe decomposition. | **High** |
| 2: Lateral probe asymmetry | Cannot assess — no-panel data in current config does not exist. | **Unknown** |
| 3: Wind FFT contamination | Confirmed (Finding 8, F10). Quantified: ~30–40% of raw wind effect is artifact. True wind effect at 1.3–1.7 Hz: +5–25% (from high-SNR filter). | **Moderate** |
| 4: Far-end reflections | Low risk for ≥ 0.80 Hz per40 runs. Check mstop_sec for 0.65–0.70 Hz. | **Low–Moderate** |
| 5: Residual wind at OUT | Confirmed (OUT wind background ~1 mm). Marginal at 1.9 Hz. FFT at paddle freq mitigates but doesn't eliminate. | **Moderate** |
| 6: Probe config correctness | CLEAN — all 337 standard runs correctly assigned. No-panel baseline for current config: **does not exist**. | Attack 6a: Low / Attack 6b: **Critical** |

### The no-panel gap is the most actionable finding

Everything else (standing wave, wind contamination) can be partially mitigated by
analysis. The missing no-panel control is a data gap that requires either:
- A new experiment (ideal), or
- The Mansard–Funke two-probe decomposition as a substitute (limited by conditioning)

The standing wave correction can only be applied with confidence once either the no-panel
OUT/IN baseline or the Mansard–Funke R estimate is available.

---

## Recommended next actions (priority order)

1. **Check if no-panel experiments were conducted but in a folder not yet cached.**
   Search `wavedata/` raw CSVs for any run with `nopanel` or similar keyword after
   2026-03-07. Use: `find wavedata -name "*nopanel*" -newer wavedata/20260307`
   or check folder names systematically.

2. **Run Reflection 2 cells** (main_explore_inline.py ~line 3137) on the current
   full-panel data. Extracts A_incident using the 8804/250 + 9373/170 probe pair.
   Will give R(f) directly — no panel position needed. Note ill-conditioning at 1.3 Hz.

3. **Compute standing wave correction for all frequencies and plot corrected OUT/IN.**
   Use SW_factor from Test 1a column, applying per-row based on the run's frequency.
   Compare raw vs corrected curves for visual smoothness test.
   Code: `combined_meta["SW_factor"] = combined_meta["WaveFrequencyInput [Hz]"].map(sw_factor_dict)`

4. **Verify panel centroid position from lab drawings or physical measurement.**
   The phase calculation is exquisitely sensitive (±0.1 m → ±0.2π at 1.3 Hz).
   Write it into CLAUDE.md §7 once confirmed.

5. **Flag 0.65–0.70 Hz per40 runs** where `mstop_sec > 35` as potentially
   reflection-contaminated. These are rare but should be excluded from low-frequency
   analysis.
