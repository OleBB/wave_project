You're right — the 0.5 mm threshold rejects the crude little ~0.3 mm waves the user can clearly see. Wrong knob. Switching from amplitude-hysteresis to **time-lockout**: after an upcrossing, lock out further upcrossings for ~T_paddle/4 (~190 ms at 1.3 Hz). That handles the "hovering near zero with multiple crossings" problem without rejecting any real wave that's slower than ~4× paddle frequency.

Pairing it with slightly more aggressive smoothing (25 samples = 100 ms — still safely below T_paddle/4).

Now update the detail plot annotation — replace the hyst-band shading with a lockout label.

The lockout works. Now look at the nowind T_i panel (middle left):

- Earliest detections at t≈3 s register **T_i ≈ 5–7 s** — a single very-long-wavelength transient pulse from the wavemaker's first impulse
- Then steady descent: T_i ~3 s at t=5 s, ~2 s at t=7 s, ~1.5 s at t=9 s
- The user's **"3 waves between t=5 and t=9"** are now visible as 3 distinct points in that window with T_i ≈ 1.5–3 s
- Amplitudes (bottom panel) confirm these are real but tiny (~0.5–1 mm) — well below the 0.5 mm hysteresis I'd been using before, which is why they got rejected

Under fullwind (right) the lockout doesn't reject wind-wave upcrossings either, so you see the dense wind-wave content sprayed across T_i ≈ 0.2–0.5 s the whole time — there's no clean precursor to recover under fullwind because the wind drowns it. That's a limitation of any "look at zero crossings" approach for fullwind, not a fix-able detector issue.

**New take on the chirp**: it's not a clean monotonic frequency sweep. There's an isolated long-wavelength impulse (T~5-7 s = ~0.15-0.2 Hz) at the very start, then a separate organized chirp from T~3 s down to T_pad starting around t=5 s. Two distinct features, not one continuous sweep. Worth noting if anyone wants to characterize the wavemaker's startup transient separately.

Want me to push the same lockout fix into the period grid plot, or leave it as a detail-only finding for now?
---
Cleaner. The chirp curve at left is now read straightforwardly: starts at T_i ≈ 5.5 s at t=3 s, jumps down to ~3 s during the t=5–9 region (the user's "3 waves"), then a smooth descent through 1.5 s at t=11, settling at T_pad=0.77 s by t=14.4 s.

Right panel under fullwind: wind-wave T_i clusters near 0.2-0.4 s the whole time, and the paddle settling shows up as the line dropping into the same T_pad band around t=15s.

Want me to push on Tests 1 (Δt vs σ_pre) and 2 (Δt vs A) now, or look at another cell first? 
 ---
 
Honest verdict: the simple highway model doesn't fit. Here's what each test says.

## Test 1 — uninformative (not failing, just unable to test)

σ_pre at 8804 under fullwind is **3.54 mm median, range 2.5–4.8 mm**. That's tight — wind setting is binary (no/full), and the wind itself doesn't vary much across runs within "fullwind". The within-cell σ_pre standard deviation is only ~0.1–0.8 mm. There simply isn't enough wind-amplitude variation to test "busier highway → bigger boost". Within-cell regression slopes are scattered and random (some +, some −), overall slope = −2.6 ms/mm (essentially zero).

To actually test this you'd need the **`experimental-fromZeroToMaxWin` runs** (CLAUDE.md §9) — those ramp wind from zero to maximum, giving a continuous wind-strength axis. With paddle ON during a wind-ramp, you'd get true σ_pre variation.

## Test 2 — partially fails, exposes a frequency-dependent split

The 1/A prediction works **only at 1.3 Hz**:
- 1.3 Hz: |Δt| 178 → 133 → 92 ms across A = 7.6 → 14.7 → 21.5 mm. Clean 1/A shrinkage. ✓

At higher frequencies the pattern **inverts**:
- 1.4 Hz: |Δt| 131 → 155 → 159 ms — flat to slightly INCREASING with A
- 1.5 Hz: 92 → 124 ms (only 2 amps available) — INCREASES
- 1.6 Hz: 162 → 195 ms — INCREASES

So at 1.6 Hz, **bigger paddle gives BIGGER |Δt|** — opposite of highway-1/A.

## Collapse plot — model wrong

σ_pre/A as a single explanatory variable: scatter is wide, fit slope is only −37 ms per dimensionless unit, no clear collapse. The 1.6 Hz, 0.3V cluster sits at low x but high |Δt|; the 1.3 Hz, 0.1V cluster sits at high x and high |Δt|. They straddle the fit line in opposite ways.

## What this means

1. **Highway-as-1/A is wrong.** The data ranks the cells by frequency much more cleanly than by amplitude or σ_pre.
2. **The frequency split is the new puzzle**: 1.3 Hz behaves *as if* the highway helps less when the paddle is bigger; 1.6 Hz behaves *as if* the highway helps MORE when the paddle is bigger. These two regimes need separate explanations or a unified model that's amplitude-AND-frequency dependent.
3. **The original observation (uniform Δt across r) still stands** — that's what told us this is a source-side or detection-side effect. But the *mechanism* isn't the simple "wind-wave priming proportional to noise floor" story. It's something that scales with paddle amplitude differently at different frequencies.

## Candidate refinements (hypotheses)

- **Wind-induced wave-current interaction at the paddle face**: Stokes drift from the wind-wave field at the source. Magnitude could depend on (paddle frequency)² because higher-freq waves are more confined to the surface.
- **Wind-wave / paddle-wave nonlinear coupling**: bigger paddle waves at higher frequency interact more strongly with wind-wave field. Coupling strength could go like A · σ_wind / (some scale).
- **Frequency-dependent panel response under wind load**: even though OUT shift equals IN shift, the OUT shift might be partly the panel responding to wind on its top surface in a frequency-dependent way.

## Suggested next step

Build a (freq, A) heatmap of |Δt| with proper error bars, and overlay it with the same heatmap of A_in enhancement (which we know exists from `methodology_wind_enhances_A_in.md`). If they correlate cell-by-cell, the two phenomena share a mechanism. If they don't, they're independent and we have two puzzles.

The findings doc, CSV, and three plots are saved. Want me to do the (freq, A) heatmap correlation with A_in enhancement next?

--- 

Three cells are still period-aliased at the cell-mean level (positive Δt of 190, 498, 465 ms = ~0.7 T — clearly wrong sign). Let me NaN those and re-run so they don't pollute the correlation.

## Verdict: the two effects are independent

**Correlation across 10 (alias-filtered) cells**:
- Pearson r = −0.085, p = 0.82
- Spearman ρ = −0.067, p = 0.85

Statistically indistinguishable from zero. The wind-induced amplitude enhancement and the wind-induced Δt are **not measuring the same underlying mechanism**.

## What each heatmap actually shows

**A_in enhancement (middle)**: clean monotonic pattern — grows with frequency. 1.3 Hz cells are essentially zero (−0.8% to +3.5%), 1.4 Hz hovers around +3%, 1.5 Hz jumps to +9.5%, **1.6 Hz reaches +15.6% at 0.1 V and +10.2% at 0.3 V**. This is exactly the documented pattern (`methodology_wind_enhances_A_in.md`). Frequency-dominated.

**Δt (left)**: no obvious monotonic structure. 1.3 Hz spans −92 to −178 ms across A. 1.4 Hz is tightly clustered around −130 to −160 ms. 1.5/1.6 Hz at 0.3V give the largest |Δt| (~−195 ms). **Amplitude-dominated, not frequency-dominated.**

## What this tells us

The two wind effects on the paddle wave are governed by different physics:

- **A_in enhancement**: wind boosts the FFT amplitude at the paddle frequency, more strongly at higher frequencies. Likely related to wind-wave spectral content overlapping the paddle band more at higher f, or nonlinear coupling between wind waves and paddle wave that grows with frequency.

- **Δt (highway)**: wind shifts the wave's apparent phase at the source, more strongly at higher amplitude (at least at 1.5–1.6 Hz). Likely related to wind-driven Stokes drift at the surface near the paddle, which would scale with paddle wave amplitude (not the wind-wave spectral overlap).

These can both exist without sharing a mechanism. The thesis story now needs to treat them as **two distinct wind effects**, not as two faces of one phenomenon.

## What the (1.3 Hz, 0.1V) cell does

This is the most-sampled cell (n_nw=9, n_fw=10) so most reliable. **A_enh = −0.8%** (essentially no enhancement) but **Δt = −178 ms** (strong shift). The two effects clearly decouple here — wind affects the timing without affecting the amplitude. Strong evidence that they're independent phenomena.

## Files

- [analysis_scratch/wind_highway_heatmap.py](analysis_scratch/wind_highway_heatmap.py)
- [analysis_scratch/wind_highway_heatmap.png](analysis_scratch/wind_highway_heatmap.png)
- [analysis_scratch/wind_highway_heatmap.csv](analysis_scratch/wind_highway_heatmap.csv)

What now? Three natural directions:
1. Move on, document the two-effects-not-one finding to memory, leave the mechanism-of-Δt as an open question for a future session
2. Pursue Δt mechanism via the `experimental-fromZeroToMaxWin` ramp datasets — those let us test "Δt vs wind strength" with a real continuous wind axis instead of binary
3. Test whether Δt correlates with mean water-level setup at OUT (wind setup magnitude per run) — another independent wind metric

Pick one or call it for now?

---

## In plain words

We started with one observation: under wind, the paddle wave seems to arrive at every probe about 130-200 ms earlier than under no wind.

**First we asked**: is the wind blowing the water surface forward, so the wave rides on a moving "river" and travels faster?

**Answer: no.** If that were true, the far probes would see a bigger time-shift than the near probes (more distance to "speed up" over). Instead, every probe — near, middle, far, even the one tucked behind the panel where there's no wind at all — shows almost the exact same shift. So whatever's happening, it's not the wind making the wave travel faster.

---

**Then we proposed**: maybe the wind has been making little choppy waves at the surface long before the paddle even moves. When the paddle finally starts, its first wave doesn't have to "build itself up from a dead-calm pond" — it joins an already-moving surface and gets a head-start. The whole wave train is born a fraction of a second earlier, and that head-start carries through to every probe equally. That's the **"highway effect"** — the wind has already paved a moving highway for the paddle wave to drive onto.

This was a satisfying story. So we tried to confirm it with two predictions:

**Prediction 1**: a busier highway (more wind chop at the source) should give a bigger head-start.
**Reality**: we couldn't really test this. Our wind setting is binary (wind on, wind off). When wind is "on", it's roughly the same strength every time. So we don't have variation to measure against. Inconclusive — would need a wind-strength sweep.

**Prediction 2**: a bigger paddle wave should outrun the highway head-start (because once your paddle is making big waves, the little wind ripples matter less proportionally).
**Reality**: this works at the lowest frequency (1.3 Hz) — bigger paddle, smaller shift, exactly as predicted. But at higher frequencies (1.5, 1.6 Hz) the pattern flips: bigger paddle gives a *bigger* shift. So the simple highway story can't be the full picture.

---

**Then we cross-checked** against another known wind effect: we already knew from earlier work that wind boosts the paddle wave's amplitude by 10-15% at high frequencies. We thought: if these two effects (the time-shift and the amplitude-boost) come from the same physical mechanism, they should appear in the same cells together. Cells with big amplitude boost should also have big time shift.

**Reality**: no correlation at all. The cells with the biggest amplitude boost (high frequency) are not the cells with the biggest time shift. The most striking case: at 1.3 Hz / 0.1V (our most-measured cell), wind makes essentially zero amplitude change (-0.8%), yet the time shift is one of the largest we see (-178 ms). That's the smoking gun — wind can shift the timing without touching the amplitude.

---

## Where this leaves us

1. **Wind makes the wave arrive earlier** — real, ~150 ms typical, observed across all probes equally.
2. **It's not Doppler.** Confirmed.
3. **The "highway" intuition is probably *partly* right, but the simple version of it is wrong.** It works at low frequency but breaks down at high frequency.
4. **Wind has at least two separate effects on the wave**, not one: a timing shift and an amplitude boost. They live in different cells of the (frequency, amplitude) grid, so they have different physical origins.

The mechanism of the timing shift remains open. The most promising next step would be to use the wind-ramp experiments (where wind is gradually turned up while waves are running) — those would give us a continuous "wind strength" axis instead of just on/off, and we could finally test whether more wind → more shift.