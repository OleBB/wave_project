Most of what you wrote is accurate in spirit and would be understood as “normal” by wave‑tank/coastal engineering reviewers. The main points to check against the literature are:

1. **Use of FFT peak amplitude as $H$ for regular waves**

   In regular (monochromatic) wavemaker experiments, it is absolutely standard to:
   - Identify the paddle (or target) frequency.
   - Take a narrow-band measure of amplitude at that frequency from the spectrum (FFT or PSD).
   - Form a transmission/reflection coefficient as
     ```math
     K_t \;=\; \frac{H_t}{H_i}
     ```
   where $H$ is a *monochromatic* amplitude or height.

   Whether $H$ is computed via:
   - direct FFT peak amplitude at $f_\text{paddle}$,
   - a narrow-band spectral moment (band-integrated PSD),
   - or a time-domain fit to a sinusoid,

   is rarely spelled out in detail, and different labs use different flavors. Your statements that:
   - $H_{m0} = 4\sqrt{m_0}$ is standard for irregular seas;
   - peak FFT amplitude is widely used for regular runs,

   match what’s in Goda, Hughes, and the coastal/wave‑tank practice. The exact numerical definition of “amplitude” is often buried in the methods section or not fully specified.

2. **Metric‑faithful vs physics‑faithful reconstructions**

   - Your “metric‑faithful” reconstruction:
     - Take the FFT, zero everything except the single peak bin, IFFT, and interpret the resulting pure sinusoid as the wave corresponding to the metric.
     - This is exactly what your metric is: $\lvert \text{FFT}[f_{\text{peak}}]\rvert \cdot 2/N$, so the time‑series reconstruction is perfectly consistent with the reported number.

   - Your “physics‑faithful” reconstruction:
     - Recognizes that a finite window and off‑bin frequency cause leakage ($\text{sinc}$ spreading).
     - The physical energy is spread across a small band of bins, and the single peak bin can be biased low by up to $\mathcal{O}(40\%)$ when $\omega$ is off‑bin.
     - A band reconstruction (± a few bins, or FWHM) recovers that energy and yields a larger physical amplitude.

   This is textbook DSP behavior applied to wave tanks. The literature rarely walks through this carefully, but the effect is real, and your description is correct.

3. **What the wave‑tank literature actually expects**

   For **data processing**, the expectations are:

   - **Transparency and internal consistency** matter more than the specific choice of metric.
   - You must be clear about:
     - How $H_i$ and $H_t$ are defined.
     - How you compute them from the signals (FFT length, window, whether you integrate over a band, etc.).
   - For **regular waves**, a reviewer will not be surprised by:
     - Taking the amplitude from a single spectral line at the paddle frequency.
     - Using that amplitude to form $K_t$.

   Where reviewers get more demanding is:
   - If your conclusions hinge on small differences in $K_t$ (say a few percent), and your amplitude extraction method has known biases at the tens‑of‑percent level when the frequency is off‑bin, they may ask you to:
     - Quantify the bias;
     - Show that it doesn’t change your conclusions; or
     - Adopt a more robust metric (sub‑bin interpolation or band integration).

   The fact that you’ve already checked 367 runs and found the **OUT/IN ratio robust within $\sim 0.5\%$** even when individual amplitudes differ by up to $40\%$ due to leakage is exactly the kind of reassurance a careful reviewer would want.

4. **On the two “fixes”: sub‑bin interpolation vs band integration**

   - **Sub‑bin interpolation** (Quinn, parabolic, Jacobsen):
     - Known and widely used in signal processing, radar, acoustics, etc.
     - Rarely named explicitly in wave‑tank work, but the *idea* of “refining the peak frequency and amplitude” is not foreign.
     - If you cite appropriate DSP references and clearly describe it, reviewers will accept it as a reasonable improvement.

   - **Band‑integrated PSD** (Huseby & Grue, $H_{m0}$):
     - Fully within the wave‑tank/coastal tradition.
     - Integrating PSD around the peak and taking
       ```math
       A \;=\; \sqrt{2 \int_{\text{band}} P_{xx}(f)\,\mathrm{d}f}
       ```
       is directly analogous to the $H_{m0}$ practice and Huseby & Grue’s band‑energy method.
     - This will be intuitively acceptable to reviewers, especially if you connect it to $m_0$ and the standard $H_{m0}$ definition.

   So your description of the two “published fixes” and their status is accurate.

5. **What people actually plot**

   - It is true that:
     - Most papers show raw probe time‑series (possibly low‑pass filtered).
     - And/or spectra ($S(f)$, $P_{xx}(f)$).
   - **FFT‑reconstructed waveforms are rare**, and when they appear, they are usually for illustrative purposes, and essentially metric‑faithful:
     - “Here is the component at the paddle frequency we’re using for $K_t$.”
   - In that sense, your Option A style reconstruction matches implicit practice: the picture illustrates the metric, not the exact physical waveform including all leakage details.

6. **Are your A/B/C options reasonable in the eyes of the literature?**

   - **Option A — metric‑faithful + disclosure**

     This is entirely defensible and aligned with community norms:

     - Keep single‑bin FFT amplitude for your $A_\text{FFT}$ metric.
     - In the caption/methods, state explicitly something like:

       > “The reconstructed wave shown is the single paddle‑bin IFFT. Its amplitude matches the reported FFT amplitude metric $A_{\text{FFT}}$. Because of windowing and off‑bin frequency, this single‑bin amplitude systematically underestimates the true paddle amplitude by a factor related to sinc‑leakage; see Section CH04.x.”

     - This is basically what a lot of published work does in practice, just with less explicit discussion of the bias.

   - **Option B — physics‑faithful + metric reconciliation**

     Scientifically, this is the strongest for a **methodology chapter**:

     - Reconstruct:
       - (1) single‑bin sinusoid (metric‑faithful),
       - (2) band‑integrated reconstruction (physics‑faithful),
       - and plot both over the raw probe time‑series.
     - Show visually that:
       - The band‑integrated reconstruction tracks the actual wave.
       - The single‑bin one is smaller but defines the metric used in $K_t$.

     This is very much in line with what Goda, Hughes, and ITTC guidelines would encourage under “demonstrate that your signal processing method captures the main wave component appropriately.” It explains the metric and its limitations clearly.

     Reviewers are likely to appreciate this as careful and “honest,” not as overcomplicating things, especially if you keep the explanation compact.

   - **Option C — upgrade the metric itself**

     - Moving from single‑bin amplitude to band‑integrated amplitude or sub‑bin interpolated amplitude as your *primary* $H$ will:
       - Reduce the bias at the individual amplitude level.
       - Shift all $K_t$ values slightly.
     - Because you already know the *ratio* $K_t$ is robust (within about $0.5\%$ across your parameter range), the scientific payoff is small compared to the disruption (recomputing, replotting, and re‑stating all numbers).

     In wave‑tank literature, reviewers usually won’t insist on such a change unless:
     - Your specific study focuses on fine accuracy in amplitude extraction itself, or
     - Someone explicitly challenges the use of single‑bin amplitudes.

     As long as you:
     - Document your current metric clearly, and
     - Provide the methodology figure showing that the choice doesn’t materially affect $K_t$ in your dataset,

     Option C is *not* expected of you.

7. **Direct answer: what does the literature “expect” you to do?**

   - Define $H_i$ and $H_t$ clearly (FFT window, bin choice, any band integration).
   - Use a standard, defensible metric for regular waves — single‑bin FFT amplitude is acceptable and commonly used.
   - Demonstrate that your metric gives a stable $K_t$ and is consistent with the observed time‑series (your CH04 figure does this).
   - Acknowledge known limitations (sinc‑leakage, off‑bin attenuation) and, if relevant, show that they don’t change the physical interpretation.

   If you implement:
   - **Option A** as your default for CH05 (results),
   - **Option B** in CH04 (methods) with a clear, single figure and short discussion,

   you are *well within* what wave‑tank / coastal engineering literature expects. You’re actually being more explicit about processing assumptions than many published papers, which is a good thing.

If you’d like, I can help you draft:
- A precise methods paragraph that would read comfortably to a coastal‑engineering reviewer, and
- A caption + panel layout for the CH04 “metric vs physical wave” figure.
