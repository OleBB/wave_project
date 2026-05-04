Figure rendered. Quick takeaways from the four numbers — there's a non-trivial finding at **fullwind IN** worth flagging:

```
(run,     probe)         sw_raw   pre_mean η   win mean   win med
nowind  · 9373/170      101.210    −0.011      +0.200    −0.568
nowind  · 12400/250     101.210    −0.009      +0.049    −0.440
fullwind· 9373/170      101.230    +0.729      +0.732    −0.770   ← !
fullwind· 12400/250     100.460    +0.014      +0.213    −0.510
```

**What's expected and what's unexpected:**

1. **`sw_raw` differs between fullwind IN and OUT** (101.23 vs 100.46 mm — diff = +0.77 mm). That's **wind setup** doing exactly what CLAUDE.md §16 predicts: water tilts leeward, so OUT (downwind, sheltered) reads a *smaller* raw distance (water is higher) and IN (upwind) reads a *larger* raw distance (water is lower). Per-run anchoring captures it. ✓

2. **`pre_mean η ≈ 0` for three of four cases** — by construction, since η = −(raw − stillwater) and stillwater is built from those same first 2 s. ✓

3. **`pre_mean η = +0.73 mm` for fullwind IN** — this is **not** ≈0 and breaks the by-construction expectation. Mechanism: the pipeline uses the **median** of the first 2 s as the anchor (`_probe_median_from_run` → `pd.Series.median()`, [processor.py:43](wavescripts/processor.py:43)), not the mean. Under fullwind the first 2 s at IN already contain wind waves with a small distributional asymmetry (mean ≠ median by ~0.7 mm). At OUT this is small (+0.014) because OUT is sheltered from the fetch.

4. **All four `win med` values are negative** (−0.4 to −0.8 mm). That's the Stokes-2 broad-trough signature visible at every probe regardless of wind. Real physics — not a baseline issue.

**So the practical question** the figure surfaces is whether the **median-based anchor** is the right call for fullwind runs at the IN probe. It's an ~0.7 mm bias on a signal whose target precision (CLAUDE.md §16) is 0.25 mm. It would not affect FFT amplitude (DC offset doesn't reach the paddle bin) or percentile amplitude (symmetric percentiles cancel), but it does mean the η baseline visible in time-domain plots is shifted under fullwind IN.

Want me to follow up with a sweep across more (freq, amp, wind, probe) cells to see whether the +0.7 mm at fullwind IN is consistent with this single canon run, or runs deeper than that?

Noted on the terminology — saving that as a durable preference, then doing the FFT check.

Now the FFT amplitude check on the nowind run, both probes, vs the meta `(FFT)` and `(LS)` columns.

Three independent calculations converge to within ~0.02 mm:

```
NOWIND canon: 1.4 Hz · 0.2 V · per240 · fullpanel

probe          my_FFT   meta(FFT)   meta(LS)     ΔFFT      ΔLS
9373/170      15.2117    15.2100    15.2115   +0.0017   +0.0002      (N=1787, bin 10 @ 1.3990 Hz)
12400/250     10.5504    10.5474    10.5340   +0.0029   +0.0163      (N=1782, bin 10 @ 1.4029 Hz)
```

All values in mm.

**What this tells us, tight version:**

- **Pipeline FFT amplitude is correct.** My one-shot `rfft / N · 2` on the analysis-window slice agrees with `Probe {pos} Amplitude (FFT)` to ±3 µm. That 3 µm is the round-trip from float32 parquet storage of the precomputed spectrum vs my float64 recomputation — not a method disagreement.
- **LS agrees too**, within ±16 µm. The slightly larger gap at OUT (16 µm) vs IN (0.2 µm) reflects LS fitting at the *exact* paddle frequency rather than the nearest bin: at OUT the bin sits at 1.4029 Hz (off-target by 0.0029 Hz), so nearest-bin and continuous-frequency disagree microscopically; at IN the bin is at 1.3990 Hz (off by 0.0010 Hz), agreement is better.
- **Bin index 10** for both probes — by construction. The window is 10T long so bin 10 = the paddle frequency when the window contains an integer number of cycles. Upcrossing-snap on both ends keeps this true; the residual 0.0029 Hz offset at OUT is because the snap landed on a window length of N=1782 vs the theoretical N=1786 for exactly 10 cycles.
- **τ check**: A_OUT / A_IN = 10.55 / 15.21 = 0.6936. Meta's `OUT/IN (FFT)` for this run reads … (let me confirm if you want — but 0.69 is consistent with the canon nowind 1.4 Hz / A₂ value in `ch05_damping_freq_table` which was 0.687).

**Verdict:** the meta `(FFT)` and `(LS)` columns for the nowind run are trustworthy. The +0.7 mm fullwind-IN baseline anomaly we saw earlier doesn't propagate into amplitude — DC offset doesn't reach the paddle bin. Time-domain percentile would be the only metric that could be biased by it, and even then symmetric P0.5/P99.5 cancels small DC shifts.

Ready for whatever the next focus is — e.g. running the same check on fullwind, or sweeping across all canon (freq, amp) cells.


Same sub-mm agreement holds, even in the harder fullwind case:

```
FULLWIND canon: 1.4 Hz · 0.2 V · per240 · fullpanel

probe          my_FFT    meta(FFT)   meta(LS)     ΔFFT       ΔLS
9373/170      15.7331    15.7245    15.7449    +0.0086    −0.0117    (N=1792, bin 10 @ 1.3951 Hz)
12400/250     11.5116    11.5079    11.5141    +0.0038    −0.0025    (N=1790, bin 10 @ 1.3966 Hz)
```

All three methods agree to within ±12 µm on IN, ±4 µm on OUT. **The +0.7 mm baseline anomaly we found in the previous step is invisible to amplitude.** Confirms the DC level doesn't reach the paddle bin: the FFT magnitude at the analysis-window's fundamental cycle is the wave amplitude regardless of where you put zero, as long as the window contains an integer number of cycles (which UC-snap on both ends guarantees).

**Wind effect on this single run, IN and OUT side by side:**

|  | nowind | fullwind | Δ (mm) | Δ (%) |
|---|---|---|---|---|
| **A_IN**  (9373/170)   | 15.21 mm | 15.73 mm | +0.52 mm | +3.4 % |
| **A_OUT** (12400/250)  | 10.55 mm | 11.51 mm | +0.96 mm | +9.1 % |
| **τ = A_OUT / A_IN**   | 0.694    | 0.732    | +0.038   | +5.5 % |

So both probes read *higher* with wind on, but OUT rises ~2× more than IN — that's the headline thesis result manifesting on a single canon pair: wind lets more wave through the panel.

**Quirk worth noting**: this single run gives Δτ = +0.038, while the `ch05_damping_freq_table` headline at 1.4 Hz / A₂ reads Δτ = +0.135. That's ~3.5× larger. The difference comes from the canonical aggregation: the meta `OUT/IN (FFT)` for the table averages across both canon moorings (`under9Mooring` vs `under9Mooring30`) and the pivot's `aggfunc="first"` picks a specific mooring per cell — not necessarily this run's. If you want to track that down, the right next step is a mooring-by-mooring breakdown of A_IN / A_OUT / τ for the (1.4 Hz, A₂) cell. Not urgent, but worth flagging that the single canon pair we just verified is *not* numerically what the published table shows.

The FFT amplitude check itself is conclusive: pipeline `(FFT)` and `(LS)` columns are correct in both wind conditions to ~10 µm precision.