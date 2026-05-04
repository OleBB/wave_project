You’re basically seeing three things at once:  

1. **Your amplitude methods agree extremely well.**  
2. **Tiny DC offsets are present but not driving the amplitude.**  
3. **Windowing is creating a residual bias (non‑zero mean) even after global de‑meaning.**

Let me walk through the numbers.

---

## 1. How well do the amplitudes agree?

For probe 9373/170:

- my_FFT: 15.2117  
- meta(FFT): 15.2100  
- meta(LS): 15.2115  

Differences:

- ΔFFT = +0.0017 ⇒ about $0.0017 / 15.21 \approx 0.011\%$  
- ΔLS  = +0.0002 ⇒ about $0.0002 / 15.21 \approx 0.001\%$

For probe 12400/250:

- my_FFT: 10.5504  
- meta(FFT): 10.5474  
- meta(LS): 10.5340  

Differences:

- ΔFFT = +0.0029 ⇒ about $0.0029 / 10.55 \approx 0.028\%$  
- ΔLS  = +0.0163 ⇒ about $0.0163 / 10.55 \approx 0.15\%$

What you learn:

- Your FFT implementation and the “canonical” FFT are virtually identical (differences in the $10^{-3}$ range).  
- Least‑squares (LS) amplitude is also close, but visibly more sensitive for the second probe (likely to leakage / off‑bin fitting, or a slightly different model).

So, **your amplitude extraction is consistent and numerically stable**. The differences are at the level where window choice, exact bin frequency, and minor preprocessing details fully explain them.

---

## 2. What about the offsets?

Offsets:

- sw_raw (both probes): 101.210 → some absolute water level / hardware offset
- pre_mean η:
  - 9373/170: −0.011
  - 12400/250: −0.009  
  ⇒ global mean of the *detrended* time series is essentially zero (offset removed correctly).
- win mean (mean after windowing):
  - 9373/170: +0.200
  - 12400/250: +0.049
- win med (median after windowing):
  - 9373/170: −0.568
  - 12400/250: −0.440

What this tells you:

1. **Before windowing**, you’ve removed the mean correctly (pre_mean η is near zero). So your basic “subtract mean before FFT” is working.
2. **After windowing**, the mean is no longer zero: the window mean is +0.200 or +0.049, and the median is negative.  
   - This is expected: applying a **non‑symmetric window to a finite record** can reintroduce a small DC component and skew median/mean.
   - Also, if the dominant sinusoid doesn’t start and end at zero crossings, windowing can shift the average.

3. The amplitude differences (ΔFFT, ΔLS) are **tiny compared to these offsets**:
   - Even with windowed mean on the order of $0.05$–$0.20$ (relative to amplitudes ≈ 10–15), your main spectral line height barely moves in the third decimal place.

So you learn that **your primary amplitude estimate at 1.4 Hz is robust to these small offsets**. The DC / very-low-frequency content is not significantly contaminating that line.

---

## 3. Implications for your processing chain

From these numbers, I’d conclude:

1. **Your current practice (de‑mean → window → FFT) is working fine** for single, clean spectral lines like this 1.4 Hz test.
2. **Adding or removing a small constant offset does not meaningfully affect your main wave amplitude**, as long as:
   - you subtract the mean over the full record, and  
   - your wave frequency is well inside the band (not near DC).
3. **Windowing does subtly change “offset-like” statistics** (window mean, median), but that’s normal and not a sign of a bug.

If you want to be extra “by the book”:

- Always:
  1. Subtract global mean of $\eta(t)$ over the record.
  2. Optionally detrend linearly.
  3. Then apply window and FFT.
- For **LS fitting**, be aware it can be a bit more sensitive to how you model the trend and DC term. If needed, explicitly include a constant in the LS model and/or fit over a carefully chosen interval.

---

## 4. High‑level “what do I learn?”

Summarizing in plain language:

- Your **two probes and three amplitude methods are self‑consistent**; differences are very small.
- The **small vertical offset / baseline level is not driving your amplitude estimate** for the 1.4 Hz component.
- The **main wave amplitude is reliable to better than about 0.1–0.2%** under these conditions.
- Windowing and subtle residual offsets mostly show up in DC/very‑low frequencies, not in your 1.4 Hz line.

If you want, you can send a short snippet of how you compute `pre_mean η`, `win mean`, and `win med` (pseudo‑code or Python/MATLAB), and I can comment on whether any small refinement would help further reduce these residual biases.

What changes with FULLWIND compared to NOWIND is:

1. The **amplitude extraction is still very consistent** between methods.  
2. The **background (“mean level”) physics has changed** a lot for one probe.  
3. Your statistics (pre_mean, win mean, win med) now clearly show **asymmetry and drift / setup** under wind.

Let’s go through it.

---

## 1. Amplitude comparison: still solid

For FULLWIND:

- 9373/170:
  - my_FFT: 15.7331
  - meta(FFT): 15.7245  → ΔFFT = +0.0086  (~0.055 %)
  - meta(LS): 15.7449  → ΔLS  = −0.0117 (~0.074 %)

- 12400/250:
  - my_FFT: 11.5116
  - meta(FFT): 11.5079  → ΔFFT = +0.0038 (~0.033 %)
  - meta(LS): 11.5141  → ΔLS  = −0.0025 (~0.022 %)

So even in wind:

- All three methods still agree to within about **0.05–0.1 %**.  
- This is slightly worse than NOWIND (where discrepancies were ~0.01–0.03 %), but still very good.

**Conclusion:** your amplitude extraction (my_FFT vs meta(FFT) vs meta(LS)) is still reliable; wind has not broken the FFT/LS machinery.

---

## 2. Offsets: big change for 9373/170 under wind

Offsets:

- Probe 9373/170 (FULLWIND):
  - sw_raw: 101.230
  - pre_mean η: +0.729
  - win mean: +0.732
  - win med: −0.770  ← big sign flip between mean and median

- Probe 12400/250 (FULLWIND):
  - sw_raw: 100.460
  - pre_mean η: +0.014
  - win mean: +0.213
  - win med: −0.510

Compare to NOWIND:

- NOWIND pre_mean η were basically zero (−0.011, −0.009).  
- NOWIND win mean were small (~0.05 to 0.20).  
- NOWIND win med were negative but moderate.

What you “learn” from FULLWIND numbers:

1. **Probe 9373/170 has a large positive mean offset under wind**:  
   - pre_mean η = +0.729 (before windowing).  
   - win mean ≈ +0.732 (after windowing, essentially unchanged).

   That’s a noticeable setup relative to the wave amplitude (~15.7). It looks like a **wind‑induced water level rise or drift** at that probe.

2. **Probe 12400/250 remains almost unbiased before windowing**:  
   - pre_mean η = +0.014 (very small).
   - So the *global mean* of the detrended record is still close to zero; the raw offset is mostly removed.

3. **Windowing under wind introduces or emphasizes asymmetry**:
   - For 9373/170, the windowed mean is +0.732 but the windowed median is −0.770.  
   - For 12400/250, win mean = +0.213, win med = −0.510.

   Mean > 0, median < 0 means the distribution of $\eta$ is **skewed**. High, intermittent positive excursions (e.g. steep or breaking crests, spray, turbulence spikes) pull the mean up, while most values are slightly negative relative to some baseline, so the median stays negative.

4. **Wind has broken the “nice symmetric sinusoid” assumption.**  
   Under NOWIND, the signal is close to sinusoidal and your statistics behave nicely.  
   Under wind, the surface elevation is more **non‑Gaussian, skewed, and possibly slowly drifting**.

---

## 3. What this implies for your processing

From these numbers, methodologically:

1. **Amplitude estimation at 1.4 Hz is still fine**  
   - The consistency between your FFT and the reference methods is strong: wind noise and setup are not significantly corrupting the 1.4 Hz line amplitude.

2. **You cannot interpret the mean level as “just a constant offset” anymore, especially for 9373/170.**  
   - A pre_mean η of +0.729 suggests real physics: wind setup, seiche, or slow drift at that probe location.  
   - Don’t blindly re‑zero everything; that mean shift might be part of what you want to measure/understand.

3. **Detrending and segmentation become more important.**

   Under wind, it’s often better to:

   - Split the record into shorter **stationary segments** (e.g. 30–60 s),
   - Remove a mean or linear trend from each segment,
   - Window each segment and compute spectra,
   - Average the segment spectra (Welch‑style).

   This reduces the impact of slow drifts and large, rare events on the spectrum.

4. **Median vs mean tells you about wave shape under wind.**

   - The fact that win median is significantly **negative** while the mean is positive indicates:
     - More frequent “below-baseline” positions with occasional large positive excursions (steep crests, spray hitting the probe, etc.).
   - This is typical when wind generates sharper crests and gentler troughs and/or when the probe signal includes some **non‑wave noise** (splashes).

---

## 4. In one sentence: what do you learn?

- Under wind, your **spectral amplitude algorithms still work**, but the **background water level and shape of the free-surface signal change strongly**, reflected in large mean shifts at one probe and strong skewness (mean vs median disagreement); this tells you that you’re now in a non‑ideal, non‑symmetric, noisy wave field where detrending/segmentation and careful physical interpretation of the “offset” are essential.

If you like, next step we can:

- Look at one FULLWIND time series and PSD,  
- Decide how to segment, detrend, and interpret the mean/setup vs the 1.4 Hz wave amplitude.