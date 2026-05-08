This is a research task — I'll search systematically. Let me read the three pipeline files and then scan the analysis scripts.

Let me delegate the cross-script audit to an Explore agent — too many files to scan tightly here.

The agent's verdict that "0 RAW bypass" looks too clean — my earlier grep flagged `parallel_probe_psd.py` as reading raw `Probe {pos}`. Verifying directly.

Verified — the agent was directionally right. The `f"Probe {probe}"` hit in `parallel_probe_psd.py` is just an axis title, not a column read. One caveat: `per40_full_chirp.py` does read `eta_{PROBE}` *without* the interp fallback — gets cleaning, no pchip. Otherwise the picture holds.

## What the pipeline does to time-series data

[processor.py:650–903](wavescripts/processor.py:650) `_zero_and_smooth_signals` writes 4 columns per probe `{pos}` per run. Each new column is built from the previous one:

| Column | Layers applied | Result |
|---|---|---|
| `Probe {pos}` | none — raw ULS reading | source of truth, untouched |
| `eta_{pos}` | 0a–0d + 1 + 2 + 3, then sign flip | zeroed elevation with **NaN holes** where dropouts/faults were detected |
| `eta_{pos}_interp` | + pchip fill of gaps shorter than `fs/(4f) ≈ λ/4` | **fully cleaned** — short gaps repaired, long gaps stay NaN |
| `Probe {pos}_ma` | linear-fill + rolling mean (`get_smoothing_window`) | smoothed; for visualisation, not metrics |

**The 7 cleaning layers (all NaN-out, no value substitution):**
- 0a `CEILCLIP` — raw within ±band of 198.7 mm hardware ceiling
- 0b `STEPCLIP` — DC step mid-run (rolling-median diff peak); NaNs everything from step onward + half-window safety buffer
- 0c `STUCK` — rolling std < `STUCK_STD_MM` AND segment mean off run-median by > 5 mm (catches "flat at wrong level")
- 0d `RANGECLIP` — η below probe-range floor (`probe_height_mm - max_mm`); ceiling deliberately not clipped
- 1 `CLIP` — hard cap on |η| > `clip_mm` (voltage-scaled for wave runs; off for stillwater)
- 2 `VELCLIP` — single-sample sign-reversal spikes (`DIFF_MM`) + sustained monotone drift (`DIFF_MONO_MM`, no-wind only) + `±VEL_BUFFER` shoulder removal
- 3 `ISOCLIP` — single non-NaN samples sandwiched by NaN

**Then** `eta_{pos}_interp` = `PchipInterpolator(eta_{pos})` on the valid samples, with `_remask_long_gaps` re-NaN-ing any gap longer than `max_interp_gap`.

`processor2nd.py` does not touch the time series — only metadata columns (`OUT/IN`, `IN ka (FFT)`, `parallel_ratio`, etc.).

## Which non-pipeline scripts ignore those fixes

After agent audit + spot verification, the buckets are:

**RAW (Bypass — read `Probe {pos}` unprocessed):** **None** for analysis. The only time `Probe {pos}` is read is for axis titles or for cached `Amplitude (PSD)/(LS)/(percentile)/(cycles)/(phase)` *meta-columns* — those are scalars per run already computed by the pipeline. No script computes Welch/FFT directly on the raw column.

**CLIPPED-ONLY (read `eta_{pos}` without interp fallback — NaN holes survive):**
- `analysis_scratch/per40_full_chirp.py` (lines 171, 259, 367) — sets `eta_col = f"eta_{PROBE}"` directly. Whether this matters depends on what it does with NaN; worth a flag.
- `wavescripts/wavestudyer.py` — diagnostic, handles NaN via `.dropna()`.

**CLEAN (use `eta_{pos}_interp`, with `eta_{pos}` only as fallback when interp absent):** the dominant pattern. Idiom across the repo:
```python
col = f"eta_{pos}_interp" if f"eta_{pos}_interp" in df.columns else f"eta_{pos}"
```
This is in every `wind_*.py`, `timeseries_*.py`, `plateau_*.py`, `inspirational_timeseries*.py`, `wavescripts/plot_browsers.py`, etc. The cached `fft_dict` and `psd_dict` (used by `parallel_probe_psd_agreement.py`, `paddle_contamination_study.py`, the canon FFT/PSD path everywhere) are computed by [signal_processing.py:_extract_probe_signal:49–51](wavescripts/signal_processing.py:49) which uses `eta_{pos}_interp` first, then `eta_{pos}`.

**SMOOTHED (read `Probe {pos}_ma`):** None in current analysis. Column exists in cache, only `plot_browsers.py` references it (for the rolling-mean overlay).

## Implication for the 1.6 Hz nowind 1.23 anomaly

Pipeline catches & pchip-fills *short* dropouts but the parallel-probe table is downstream of that:

- Cached PSD ← `welch(eta_{pos}_interp[7T_window])`. If the dropout is **shorter than `λ/4`** (≈ 39 samples at 1.6 Hz), pchip fills it — the local amplitude on the rising flank is *imputed*, not measured. Pchip is monotone-cubic, so it doesn't overshoot, but it can't recover energy that was never recorded — the cycle in question contributes a suppressed amplitude to the FFT/PSD on that probe only. That's exactly the asymmetric mechanism that pulls the geometric-mean ratio away from 1.
- If the dropout is **longer than `λ/4`**, the gap stays NaN, `_extract_probe_signal` falls back to the *longest contiguous segment* — vegg and langt can end up with different effective lengths and bin alignment.

So the right diagnostic plot for the suspect runs is overlay all three of `Probe {pos}`, `eta_{pos}` (raw with holes), `eta_{pos}_interp` for both parallel probes — that shows visually whether pchip silently suppressed the amplitude on one probe.

Want me to write that diagnostic, or do you want to eyeball the runs first in `RampDetectionBrowser` / `SignalBrowserFiltered` ([wavescripts/plot_browsers.py](wavescripts/plot_browsers.py)) and decide based on what you see?