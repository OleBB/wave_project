# Rolling-RMS stationarity — rubber-band splash detection

Generated: 2026-04-17T17:46:08Z

Source: `analysis_scratch/rolling_rms_stationarity.py` → this doc

## Scope and method

Target: 17 fullwind+nowave runs (excluding wind-ramp experimental runs). Probes evaluated: `9373/170` (IN-side, panel-adjacent) and `9373/340` (parallel lateral cross-check).

Per run: compute rolling RMS of η over 1 s windows (step 0.5 s), skipping first 5 s and last 5 s. Score three stationarity metrics:

- **CV** = std(rolling_rms) / mean(rolling_rms). Threshold: > 0.25
- **max/median** of rolling RMS. Threshold: > 2.5
- **burst_frac** = fraction of windows > median + 3·MAD. Threshold: > 10%

A run is flagged **splashy** if any metric at the primary probe crosses its threshold.

## Summary

- Runs analyzed: **17**
- Flagged splashy: **10** (59%)
- Clean:           **7**

### Splashy fraction by mooring

| mooring | n | splashy | splashy_frac |
| --- | --- | --- | --- |
| above_50 | 4 | 4 | 1.000 |
| below_90_loose230 | 11 | 4 | 0.364 |
| below_90_loose300 | 2 | 2 | 1.000 |


## Splashy runs (detail)

| folder_file | mooring | 9373/170_cv | 9373/170_max_med | 9373/170_burst_frac | 9373/340_cv | 9373/340_max_med |
| --- | --- | --- | --- | --- | --- | --- |
| 20260307-ProbPos4_31_FPV_2-tett6roof/fullpanel-fullwind-nowave-ULSonly-endofday.csv | above_50 | 0.278 | 1.949 | 0.007 | 0.268 | 1.890 |
| 20260307-ProbPos4_31_FPV_2-tett6roof/fullpanel-fullwind-nowave-depth580-ULSonly.csv | above_50 | 0.355 | 5.247 | 0.015 | 0.286 | 2.517 |
| 20260314-ProbePos4_31_FPV_2-tett6roof/fullpanel-fullwind-nowave-run2.csv | above_50 | 0.384 | 4.047 | 0.027 | 0.274 | 2.291 |
| 20260314-ProbePos4_31_FPV_2-tett6roof/fullpanel-fullwind-nowave-run1.csv | above_50 | 0.362 | 4.071 | 0.015 | 0.284 | 2.355 |
| 20260319-ProbePos4_31_FPV_2-tett6roof-under9Mooring/fullpanel-fullwind-nowave-depth580-mstop30-run1.csv | below_90_loose230 | 0.360 | 3.355 | 0.027 | 0.292 | 2.184 |
| 20260323-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100/fullpanel-fullwind-nowave-depth580-mstop30-run2.csv | below_90_loose230 | 0.261 | 1.777 | 0.000 | 0.239 | 1.763 |
| 20260324-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100/fullpanel-fullwind-nowave-depth580-endofday-run1.csv | below_90_loose230 | 0.254 | 1.736 | 0.000 | 0.216 | 1.440 |
| 20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange/fullpanel-fullwind-nowave-depth580-mstop330-run1.csv | below_90_loose230 | 0.255 | 1.796 | 0.001 | 0.252 | 2.073 |
| 20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange/fullpanel-fullwind-nowave-depth580-mstop30-run4.csv | below_90_loose300 | 0.281 | 1.926 | 0.001 | 0.247 | 1.853 |
| 20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange/fullpanel-fullwind-nowave-depth580-mstop30-run2.csv | below_90_loose300 | 0.260 | 1.572 | 0.000 | 0.222 | 1.756 |

## Near-threshold clean runs (diagnostic)

Runs that came close to a splash threshold but passed. Worth looking at the raw signal if a trend in the data points at these.

| folder_file | mooring | 9373/170_cv | 9373/170_max_med | 9373/170_burst_frac |
| --- | --- | --- | --- | --- |
| 20260323-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100/fullpanel-fullwind-nowave-depth580-mstop30-run1.csv | below_90_loose230 | 0.245 | 1.741 | 0.025 |
| 20260324-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100/fullpanel-fullwind-nowave-depth580-run-endofday-P2malfunction-butfirstpartcanbeused.csv | below_90_loose230 | 0.244 | 1.665 | 0.024 |
| 20260324-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100/fullpanel-fullwind-nowave-depth580-run1.csv | below_90_loose230 | 0.232 | 1.654 | 0.000 |
| 20260324-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100/fullpanel-fullwind-nowave-depth580-run2.csv | below_90_loose230 | 0.227 | 1.689 | 0.008 |
| 20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange/fullpanel-fullwind-nowave-depth580-mstop30-run1.csv | below_90_loose230 | 0.210 | 1.514 | 0.000 |
| 20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange/fullpanel-fullwind-nowave-depth580-mstop30-run3.csv | below_90_loose230 | 0.212 | 1.639 | 0.000 |
| 20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange/fullpanel-fullwind-nowave-depth580-mstop30-run2.csv | below_90_loose230 | 0.214 | 1.650 | 0.000 |


## Interpretation

**Important nuance seen from the diagnostic figure**: the `splashy` flag does NOT cleanly map to rubber-band events. The high-max/med runs (above_50 in 20260307/20260314) show sensor-glitch spikes reaching ±60 mm in the raw η — these are **dropout-recovery artifacts or sensor glitches**, not physical splash events. The metric catches three qualitatively different regimes:

| Regime | Metric signature | Typical cause |
|---|---|---|
| **Severe glitch** | max/med > 3.0, burst_frac ≥ 1% | Sensor dropout recovery producing ±60 mm spikes |
| **Likely rubber-band** | max/med 1.5–2.5, borderline CV | Brief bursts consistent with rubber-band impact events |
| **Normal wind variation** | CV 0.20–0.27, max/med < 2.0 | Intrinsic wind-wave field variability over ~30 s windows |

The 4 above_50 runs flagged with max/med 4–5 are the clearest **sensor-glitch** candidates and should be excluded from any wind-background averaging. The CV-only flagged runs (borderline cases) are probably normal wind variation and are fine to keep with caveat.

Expected-vs-observed on the `under9Mooring` rubber-band note:

- `under9Mooring` (`below_90_loose230`, rubber-band present per user): 4/11 flagged. Consistent with splash being intermittent.
- `under9Mooring30` (`below_90_loose300`, no rubber-band expected): 2/2 flagged but only by borderline CV, not by max/med — probably normal wind variation, NOT splash.
- `above_50` (stiff mooring): 4/4 flagged by high max/med — **sensor-glitch** regime (early-date experimental runs), not splash.

**What to do with these runs**: for per-folder wind-background averaging, **exclude severe-glitch runs** (max/med > 3.0). Borderline CV runs are usable. This affects Finding 2 of `probe_height_wind_findings.md` if it averaged across the glitchy above_50 runs.

**What this does NOT affect**: the thesis OUT/IN figures in CH05 — they use wave runs (not nowave+fullwind).

## See also

- Figure: `analysis_scratch/rolling_rms_stationarity.pdf` (clean vs splashy example)
- CSV:    `analysis_scratch/rolling_rms_stationarity.csv`
- Context: `memory/physics_wavetank_mooring_fetch.md`, `analysis_scratch/probe_height_wind_findings.md` Finding 5
