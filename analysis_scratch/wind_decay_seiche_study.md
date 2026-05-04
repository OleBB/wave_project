```
Rampup 20260314  (T=360s, zero=first 2.0s, wind=last N s)
   N(s)  | IN Δη  OUT Δη  | OUT−IN
   ------+----------------+--------
       5 | +1.21  +0.78  | −0.43
      10 | +1.38  +0.81  | −0.57
      20 | +1.56  +0.80  | −0.76
      30 | +1.57  +0.81  | −0.76
      60 | +1.71  +0.80  | −0.91

Decay 20260327_eod  (T=1387s, zero=last 2.0s, wind=first N s)
   N(s)  | IN Δη  OUT Δη  | OUT−IN
   ------+----------------+--------
       5 | +1.32  +1.38  | +0.06
      10 | +1.22  +1.32  | +0.10
      20 | +1.04  +0.79  | −0.25
      30 | +0.78  +0.57  | −0.21
      60 | +0.48  +0.32  | −0.16
```

**Key observations**

- **Rampup OUT is rock-stable at +0.80 mm** across every window from 5–60 s. So OUT *is* at steady state — case (a) doesn't apply to OUT.
- **Rampup IN keeps climbing** (+1.21 → +1.71) as the window widens. That means IN was **higher *earlier* in the latter portion of the recording and dropping toward the end** — not a still-evolving steady state, but a downward drift in IN over the last minute or so. OUT doesn't show it.
- **The OUT−IN inversion (IN > OUT) is present at every window length** for the rampup, including 30 s and 60 s. So it's not a "wind hadn't reached steady state" artefact — it's a real property of the 20260314 record.
- **Decay at 5–10 s gives the clean wind-setup signature** (OUT > IN by 0.06–0.10 mm). At 20 s+ the first-N-seconds window is starting to include the post-jog phase where the user turned fans off, and Δη collapses toward zero — confirming that 5 s is the right size for the wind window (and validates your jog-buffer rule empirically).

So your case (a) is **not** the explanation for the rampup inversion. The rampup is actually showing IN drifting downward over its last minute while OUT holds steady — which is closer to a slow seiche or a tank-redistribution transient than a "wind not yet developed" issue. Different mooring/height geometry on 20260314 (no `height100`, no `under9Mooring30`) is still on the table as a candidate explanation. The 5-s legend reading you're using for the figure is fine; it just can't be read as confirming the wind-setup direction the way the decay does.
---
---

---
---
---
---

---
---



```
                                                p99.5-p0.5
slab [s]    probe   mean(Δη)  std    /2 (~amp)   peak f [Hz]  peak T [s]
─────────────────────────────────────────────────────────────────────────
[ 60, 120]  IN      +1.51    4.44    10.17       0.122        8.19
[ 60, 120]  OUT     +0.75    0.53     1.20       0.122        8.19
[120, 180]  IN      +1.46    4.20     9.60       0.244        4.10
[120, 180]  OUT     +0.80    0.43     1.12       0.122        8.19
[180, 240]  IN      +1.76    4.95    11.35       0.122        8.19
[180, 240]  OUT     +0.75    0.40     0.99       0.122        8.19
[240, 300]  IN      +1.58    4.75    11.14       0.244        4.10
[240, 300]  OUT     +0.79    0.38     0.94       0.122        8.19
[300, 360]  IN      +1.71    4.93    10.68       0.488        2.05
[300, 360]  OUT     +0.80    0.36     0.93       0.122        8.19
```

(Δη is mean over the slab minus mean over the first 2 s of the recording, using `eta_*_interp`. Welch peak restricted to 0.05–0.5 Hz seiche band.)

**Headline answers**

(a) **Yes — the seiche is stable.** OUT shows a clean **0.122 Hz peak (T = 8.19 s)** in every single 60-s slab from t = 60 s through t = 360 s. Five slabs, identical bin. Period locked to ~8.2 s. (At IN the algorithm picks 2nd or 4th harmonics of the same line in two slabs because IN is buried in much louder wind-wave clutter — but the underlying 0.122 Hz line is clearly the same.)

(b) **The median (mean) at OUT is rock-stable at +0.78 ± 0.02 mm** across the full 60–360 s window relative to the pre-wind first-2-s reference. Std on OUT is only ~0.4 mm — the seiche oscillates ±1 mm peak-to-peak around that clean +0.78 mm offset.

**The interesting bit you'll want for the thesis**

Look at OUT vs IN beyond the seiche question:

- **OUT std ~0.4 mm, peak amp ~1 mm.** Sheltered by the panel, no wind-wave clutter. You're seeing only the seiche on top of the wind setup.
- **IN std ~4.5 mm, peak amp ~10 mm.** IN is fully exposed to the wind fetch, so wind waves dominate its signal. The same ~8.2 s seiche is buried under ±10 mm of wind-wave clutter.

That **is** the wind setup picture you'd expect: OUT (leeward, sheltered) sits at a clean +0.78 mm above pre-wind. IN (upwind, exposed) sits at a noisier ~+1.5 mm — and the apparent IN > OUT inversion is a **wind-wave skew/clutter bias on IN's time-average**, not a real water-level inversion. (Consistent with your morning's finding that wave median sits ~0.5 mm above true stillwater under wind.)

If you want the cleanest single-figure number to characterize wind setup at this run: **OUT = +0.78 ± 0.02 mm above pre-wind, dominant seiche period 8.2 s.**



# USER: So OUT reaches steady pretty quickly?

Yes — and it's striking how fast.

```
slab [s]    probe  mean(Δη)   std
─────────────────────────────────
[ 0, 10]    OUT    ~0.00     0.15    ← pre-wind: probe quiet
[10, 15]    OUT    -0.01     0.16    ← still quiet (you're jogging / fan starting)
[15, 20]    OUT    +1.16     0.80    ← wind first arrives, OUT *overshoots* slightly
[20, 30]    OUT    +0.73     0.70    ← settled
[30, 40]    OUT    +0.66     0.73
[40, 50]    OUT    +0.64     0.65
[50, 60]    OUT    +0.71     0.65
[60–360]    OUT    +0.75–0.80 (every 60-s slab), std ≈ 0.4
```

OUT goes from **0 → +1.16 mm in ~5 s** (small overshoot at first arrival), then settles to its long-run mean **+0.73 mm by t≈20–30 s**. From t=20 s onward, every 60-s mean lands within ±0.07 mm of +0.78 mm. So OUT essentially reaches its steady wind-on level within **~10–15 s of wind first reaching the probe**, and the only motion left after that is the ±1 mm, 8.2-s seiche riding on top.

**By contrast IN**:
```
[15, 20]    IN     +0.70     0.49    ← starting to feel wind
[20, 30]    IN     +1.22     4.18    ← wind waves SATURATING (std × 8)
[30, 40]    IN     +2.06     5.08    ← fully developed wind-wave clutter
[40, 360]   IN     +1.1 … +2.1, std ~4–5 mm
```

IN's *mean* is never stable — it bounces between +1.1 and +2.1 — because wind-wave clutter (std ~4–5 mm, peak amp ~10 mm) doesn't average out at 60-s windows. The wind-wave field itself takes maybe ~15 s to saturate, but after that the noisy time-average never settles the way OUT does.

**Implication for the figure**: OUT is the cleaner record of the wind-setup magnitude (+0.78 mm) and reaches it almost immediately. IN's apparent mean is contaminated by wind-wave skew, so you can't read wind-setup off it directly without filtering the wind-wave band out.