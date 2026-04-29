Treat the wind “noise” as just another component of the signal that you **model and separate**, not something you must perfectly remove globally.

For your use case (regular wave at known \(f_p\), wind waves on top), the clean, standard approach is:

---

### 1. Don’t chase a perfect global mean

With wind, the “mean level” wiggles a bit and can drift slowly. That’s normal.

You do **not** need a single magic mean level that makes all your records zero. For amplitude and transmission, what you need is:

- a good estimate of the **component at \(f_p\)** in each 10T window,
- and a way to keep **low‑frequency wind + drift** from contaminating that.

---

### 2. Use LS in each window to separate “wave at \(f_p\)” from “slow stuff”

In each 10T window at IN and OUT, fit:

```math
η(t_i) ≈ C_0 + C_1 t_i
       + A_c cos(2π f_p t_i)
       + A_s sin(2π f_p t_i)
```

- \(C_0\) = local mean level (includes wind set‑up, any offset).
- \(C_1\) = local slow drift (if any).
- \(A_c, A_s\) = coefficients of the paddle component at \(f_p\).
- Wave amplitude: \(A = \sqrt{A_c^2 + A_s^2}\).

Everything that’s **not** at exactly \(f_p\) (most wind waves, mean, slow slosh) either:

- gets absorbed in \(C_0, C_1\), or
- sits in the residual and doesn’t bias \(A\) unless it has strong energy exactly at \(f_p\).

This is the key: **you don’t need to know the true mean level**; you just model it per window. That’s how you make your amplitudes robust to wind.

---

### 3. Use band‑pass only for detection / alignment

Wind waves mainly hurt:

- zero‑crossing based phase detection,
- plateau finding.

Solve that by making a **filtered copy** of the signal, only for timing:

1. Band‑pass filter around \(f_p\) (e.g. \([0.8 f_p, 1.2 f_p]\)).
2. On this filtered copy:
   - detect the first zero upcrossing after the plateau,
   - or align IN/OUT via cross‑correlation.
3. Use those sample indices on the **raw** signal for the LS fit above.

So:

- Wind noise is suppressed in the phase / window‑placement step.
- The LS fit then handles mean and any remaining slow drift.

---

### 4. For your question about “what mean to use”

With wind noise, the “mean to use” for amplitude extraction is simply:

- the **local mean \(C_0\) from the LS fit**, or
- the **window average** you subtract before FFT.

You report amplitudes relative to that local baseline; the 30‑s calm‑water mean is just a reference for geometry/theory, not something you subtract in each amplitude calculation.

---

If you show me one short sample (e.g. plot or description: typical wave height, typical wind‑noise std), I can suggest how strong your band‑pass should be and whether you really need the linear trend term \(C_1\), or if a simple DC term \(C_0\) is enough for your windows.
