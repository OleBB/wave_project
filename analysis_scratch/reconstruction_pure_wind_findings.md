# Pure wind = fullwind residual − nowind residual (Stokes baseline)

**Date**: 2026-04-18
**Script**: `analysis_scratch/reconstruction_A_vs_B.py` §5–§6 (second half)
**Figure**: `analysis_scratch/reconstruction_pure_wind.pdf` (1.4 Hz / 0.20 V demo)
**Per-group table**: `analysis_scratch/reconstruction_pure_wind_summary.csv`
**Related**: `reconstruction_A_vs_B_findings.md` (the A-vs-B safety check that motivated this).

## Motivation

The A-vs-B check established that method A's residual gives the same 2–6 Hz wind-band energy as method B's. What it did **not** address: the residual itself is not pure wind. A wave run's residual contains

    paddle-linked Stokes harmonics (2f, 3f, 4f, …) + wind-wave energy + broadband noise

For paddle frequencies 1.3–1.6 Hz, the Stokes harmonics fall directly inside the 2–6 Hz wind band (`2f = 2.6–3.2 Hz`, `3f = 3.9–4.8 Hz`, `4f = 5.2–6.4 Hz`). Integrating `residual_A` over 2–6 Hz therefore measures **wind + Stokes**, not wind alone.

The clean separator exists in the dataset: **nowind wave runs** at the same (frequency, amplitude, mooring, panel) have exactly the same Stokes harmonic content and zero wind. Subtracting the mean nowind residual PSD from the mean fullwind residual PSD cancels the Stokes contribution.

    PSD_pure_wind(f) := mean PSD(residual_A | fullwind) − mean PSD(residual_A | nowind)

Integrated over 2–6 Hz, this is the honest "what did the wind add" number.

## Data

- `meta_results`, full panel, `quality_flag in {ok, probe_malfunction_secondary}`, `Mooring` collapsed (`loose230/300 → below_90_loose`).
- 13 matched (freq × amp) groups, 26 (group × probe) comparisons.
- Frequencies 1.3–1.7 Hz at 0.1 / 0.2 / 0.3 V. Frequencies below 1.3 Hz only have fullwind coverage in this dataset — no match possible.

Per-run PSDs (Welch, nperseg ≤ 4096) are interpolated onto a shared 0.05 Hz grid (0–12.5 Hz) before averaging so runs with different record lengths can be pooled.

## Headline result — Stokes fraction of the naive "wind" energy

`(E_naive − E_pure) / E_naive` where `E_naive = ∫PSD(fullwind residual)` over 2–6 Hz and `E_pure` is after nowind subtraction (positive-clipped).

| probe       | median | mean | std  | min   | max  | n  |
|-------------|-------:|-----:|-----:|------:|-----:|---:|
| 12400/250 (OUT) | 0.234 | 0.215 | 0.075 | 0.086 | 0.330 | 13 |
| 9373/170  (IN)  | 0.143 | 0.195 | 0.191 | 0.010 | 0.593 | 13 |

**20–25 % of what we measure as wind in the 2–6 Hz band is actually paddle Stokes harmonics.** Not 0%, not 90%; a real but correctable bias.

## Scaling with amplitude and frequency

Stokes energy scales as `(ka)²`. That shows up cleanly in the per-group data:

| freq / amp | IN Stokes frac | OUT Stokes frac |
|-----------:|---------------:|----------------:|
| 1.3 / 0.1  |  1 % |  24 % |
| 1.3 / 0.2  |  7 % |  17 % |
| 1.3 / 0.3  | 29 % |  30 % |
| 1.4 / 0.1  |  1 % |  18 % |
| 1.4 / 0.2  |  8 % |  28 % |
| 1.4 / 0.3  | 31 % |  33 % |
| 1.5 / 0.1  |  7 % |  28 % |
| 1.5 / 0.2  | 22 % |  23 % |
| 1.5 / 0.3  | **52 %** | 19 % |
| 1.6 / 0.1  |  1 % |  11 % |
| 1.6 / 0.2  | 20 % |  25 % |
| 1.6 / 0.3  | **59 %** | 14 % |
| 1.7 / 0.2  | 14 % |   9 % |

Two patterns stand out:

- **IN at 0.3 V crosses 50 % Stokes by 1.5 Hz.** The fullwind residual at the IN probe is more than half Stokes harmonics at this corner of the parameter space. Any wind metric that ignores Stokes is untrustworthy here.
- **OUT is 10–30 % Stokes almost everywhere**, with weak frequency dependence. The panel suppresses wind-wave energy at OUT substantially more than it suppresses the transmitted Stokes harmonic — so what reaches OUT is a relatively Stokes-rich mixture even when the Stokes amplitude itself is tiny in absolute terms.

## Demo figure walk-through (`reconstruction_pure_wind.pdf`, 1.4 Hz / 0.20 V)

Both probes, mean residual PSDs in blue (nowind = Stokes baseline), red (fullwind = wind + Stokes), black (pure wind = fullwind − nowind).

**IN probe (left)**

- **Blue** line shows a clean 2f = 2.8 Hz Stokes peak at ~1.5 mm²/Hz, smaller bumps at 3f = 4.2 Hz and 4f = 5.6 Hz. Background everywhere else is ~0.01 mm²/Hz.
- **Red** line shows the same Stokes peaks plus a broad wind-wave contribution filling the 2.5–5 Hz region up to ~8 mm²/Hz. The 2.8 Hz peak is a superposition of Stokes + wind-at-2.8 Hz.
- **Black** (pure wind) has the Stokes peaks cleanly removed. The true wind spectrum peaks at ~3.5–4 Hz, **not** at 2.8 Hz — the apparent peak in red at 2.8 Hz was a Stokes artefact.
- `E_naive = 5.35`, `E_Stokes = 0.42`, `E_pure = 4.94`, Stokes fraction **7.7 %** — small correction for this run.

**OUT probe (right)**

- Stokes peaks at 2f / 3f / 4f visible in both blue and red; wind adds a much smaller broadband contribution (panel shelters the OUT probe from wind fetch).
- Pure wind (black) is almost flat at ~0.05 mm²/Hz across the wind band — wind essentially does not reach OUT as a coherent wave field.
- `E_naive = 0.52`, `E_Stokes = 0.16`, `E_pure = 0.37`, Stokes fraction **28.4 %** — the naive "wind at OUT" number is inflated by a quarter.

## Caveat — subtraction can over- or under-correct on peak-by-peak basis

The nowind Stokes baseline is built from n=2–9 runs per group; for small n it inherits run-to-run variability. If a nowind group happens to have a slightly *lower* paddle amplitude than the fullwind group, its Stokes 2f peak is also lower → subtraction under-corrects → the black curve lies above the true wind PSD at 2f. Conversely if nowind has a slightly higher paddle amplitude, subtraction over-corrects → `fullwind − nowind < 0` at 2f, clipped to zero.

At higher-amplitude / higher-frequency runs (1.5–1.6 Hz / 0.3 V), the nowind Stokes energy can exceed the fullwind Stokes energy — see rows 18, 24 of the CSV where `E_Stokes > 0.5·E_naive`. Mechanism: wind-induced phase jitter broadens the paddle peak and its harmonics, lowering the peak PSD (same effect that reduces A_FFT under fullwind — see CLAUDE.md §6). That makes `fullwind − nowind` go slightly negative at the Stokes peaks in those conditions; the `clip(0, None)` step absorbs it. Signed-integral variants would show the effect more cleanly but the clipped version is the right number for "what energy did the wind add".

The effect is a 5–20 % uncertainty on `E_pure` at high amplitude, larger than the A-vs-B discrepancy (which was 0 %). This is the real noise floor for wind characterisation in this dataset.

## What this means for the thesis

1. **For CH05 transmission results (OUT/IN of `A_FFT`)**: irrelevant. Stokes contamination is in the *residual*, not the paddle peak. `A_FFT` and therefore OUT/IN are unaffected.
2. **For CH04 wind-characterisation figures**: any "wind at the IN probe" or "wind at the OUT probe" number that integrates the 2–6 Hz residual of a **wave** run is 20–30 % Stokes on average, up to 60 % at the worst corners. Either (a) subtract a matched nowind baseline like this analysis does, or (b) use nowave + fullwind runs for wind characterisation (which is what `probe_height_figure` / CH04 §3b already does — those are Stokes-free).
3. **For the methodology figure that presents method A**: the pure-wind panel here is a clean complement to the A-vs-B panel. Together they say:
   - Method A is safe: the paddle under-reporting doesn't leak into 2–6 Hz (A-vs-B check).
   - But residual energy in 2–6 Hz is not pure wind: 20 % is Stokes harmonics that live there because `2f` / `3f` / `4f` fall inside the wind band for our paddle frequencies. A nowind subtraction (or use of nowave runs) is required to isolate wind.

## Possible next steps

- Extend the subtraction to panel=reverse / panel=no runs if we ever need wind characterisation there; same method applies.
- Produce `pure_wind_energy_in` / `pure_wind_energy_out` as a per-(freq, amp, mooring) summary table for the methodology chapter — use this alongside the T_cross metric as complementary wind-effect descriptors.
- Consider a variant that uses a narrower wind band (e.g. 3.5–5 Hz, avoiding `2f` at 1.3–1.6 Hz paddle) — makes the Stokes correction smaller but costs SNR. Probably not worth it given the subtraction works well.
