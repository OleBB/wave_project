Here’s a structured summary of the key issues you raised and how we addressed them. You can copy‑paste sections into your notes or thesis.

---

# 1. Measuring “main wave” amplitude vs “total” amplitude

**Issue you raised**

- FFT at the paddle frequency gives a clean amplitude.  
- You wanted to compare this to some notion of “total amplitude” of the surface elevation (including wind waves and everything else) and needed:
  - a definition,
  - a name,
  - and a way to relate them.

**Key points**

- Model in a window:
  ```math
  \eta(t) = \eta_{f_p}(t) + \eta_\text{wind}(t) + \text{noise},
  ```
  with
  ```math
  \eta_{f_p}(t) \approx A\cos(2\pi f_p t + \phi).
  ```

- **Main wave (fundamental) amplitude**  
  From LS/FFT at $f_p$:
  ```math
  \eta(t_i) \approx C_0 + A_c\cos(2\pi f_p t_i) + A_s\sin(2\pi f_p t_i),
  \quad
  A = \sqrt{A_c^2 + A_s^2}.
  ```
  This $A$ is your **fundamental amplitude at the paddle frequency** in that window.

- **“Total amplitude” / total energy**  
  For the full signal in that window (demeaned), compute variance:
  ```math
  \sigma_\eta^2 = \frac{1}{N}\sum \eta_n^2,
  \quad
  H_s = 4 \sigma_\eta.
  ```
  $H_s$ is the **significant wave height**; $\sigma_\eta$ or $H_s/2$ can serve as a “total amplitude” measure.

- **Relating the two**  
  For a pure sinusoid of amplitude $A$:
  ```math
  \sigma_\eta = \frac{A}{\sqrt{2}}, \quad H_s = 4\sigma_\eta = 2\sqrt{2}\,A.
  ```
  You can form ratios like:
  ```math
  \text{fraction of variance at } f_p \approx \frac{A^2/2}{\sigma_\eta^2},
  \quad
  \frac{A}{H_s} \ \text{or}\ \frac{A}{2\sigma_\eta}.
  ```

**How to name things**

- $A$ : “fundamental amplitude at the paddle frequency”.  
- $\sigma_\eta$ : “total standard deviation of surface elevation in the window”.  
- $H_s$ : “total significant wave height in the window”.

---

# 2. How to measure amplitude with wind waves riding on top

**Issue you raised**

- Wind waves superimposed on the regular wave distort individual crests.
- Should you:
  - take max–min within each period, or
  - lock to the known period/frequency and look at the signal at that phase?

**Key points**

- **Don’t rely on max–min within single periods**: individual crests are randomly inflated/deflated by wind; it’s noisy and not a robust fundamental measure.

- **Use a frequency‑locked fit and average across many cycles**:
  - Fit at known $f_p$:
    ```math
    \eta(t_i) \approx C_0 + A_c\cos(2\pi f_p t_i) + A_s\sin(2\pi f_p t_i),
    ```
    $A = \sqrt{A_c^2 + A_s^2}$, $\phi = \text{atan2}(-A_s, A_c)$.
  - This gives a **best estimate of the underlying regular wave** in that window.

- **To study how wind distorts crests**, use residuals and/or phase‑locked averages:
  - Reconstruct fitted fundamental:
    ```math
    \hat{\eta}_{f_p}(t_i) = A \cos(2\pi f_p t_i + \phi),
    ```
  - Residual (wind+noise): $r(t_i) = \eta(t_i) - \hat{\eta}_{f_p}(t_i)$.
  - Examine $r(t_i)$ near crest phase of $\hat{\eta}_{f_p}$ to quantify:
    - mean bias at crest (systematic wind effect),
    - rms scatter (random wind effect).

- **Phase‑averaging**:
  - Define phase $\theta(t_i) = 2\pi f_p t_i + \phi$.
  - Bin by phase and average $\eta$ in each bin → “typical waveform” and its variability as a function of phase.

---

# 3. Phase‑locked / phase‑averaged methods vs real‑world design practice

**Issue you raised**

- Are phase‑locked analyses “real science”?  
- How do they relate to what engineers do for real floating structures?

**Key points**

- **Yes, phase‑locked analysis is standard** in:
  - regular‑wave hydrodynamics (RAOs, diffraction/radiation),
  - PIV/LDV under regular waves,
  - breaking wave impact studies,
  - CFD of periodic waves (cycle/phase‑averaging).

- In **field / design practice** (irregular seas), the focus is on:
  - statistical/spectral parameters: $H_s$, $T_p$, spectrum shape,
  - RAOs in frequency domain,
  - response spectra and extreme statistics.

- Design is done with:
  ```math
  S_\eta(\omega) \xrightarrow{\text{RAO}} S_\text{response}(\omega) \to \text{RMS / extremes},
  ```
  not with a fixed global phase.

- Your lab work fits naturally into this framework:
  - you measure deterministic, frequency‑resolved response at $f_p$,
  - this is analogous to RAOs used in design.

---

# 4. Reflections, travel time, and long‑wave transients

**Issue you raised**

- For a 1.3 Hz wave in a 25 m tank, when do reflections return?  
- What is the fast disturbance you see at 9 m after ~3 s (faster than your 1.3 Hz phase speed)?  
- What happens to that long‑wave/seiche signal over time?

**Key points**

- **Reflection time** (first main reflection from far end back to the paddle region):
  - Path: $2L = 50\ \text{m}$.
  - Phase speed $c$ from dispersion (finite depth) or deep‑water approximation:
    ```math
    c \approx \frac{gT}{2\pi}.
    ```
  - Reflection time:
    ```math
    t_\text{refl} \approx \frac{2L}{c}.
    ```
  - For your case, this is on the order of 40–45 s (depending on actual depth).

- **Fast initial disturbance**:
  - Apparent speed: $\sim 3\ \text{m/s}$, much faster than the 1.3 Hz waves.
  - This matches the **shallow‑water long‑wave speed**:
    ```math
    c_\text{long} \approx \sqrt{gh},
    ```
    e.g. $h \approx 0.9\ \text{m} \Rightarrow c \approx 3\ \text{m/s}$.
  - This is a **long‑wave pulse / basin adjustment / seiche‑type mode**, excited by the paddle moving the whole water column at start‑up.

- **What happens to the long wave/seiche**:
  - It runs to the far end, reflects, and sets up a **standing long‑wave mode** (seiche).
  - It decays gradually due to:
    - boundary friction,
    - absorption at beach,
    - viscous effects.
  - During your steady wave‑train period, what remains is a **small, slow background motion** $M(t)$ (setup/tilt).

- In your 10T analysis windows:
  - You model:
    ```math
    \eta(t_i) \approx C_0 + C_1 t_i + A_c\cos(2\pi f_p t_i) + A_s\sin(2\pi f_p t_i),
    ```
  - $C_0$ and $C_1 t$ capture the slow seiche/drift $M(t)$,
  - $A$ captures the 1.3 Hz amplitude.

- **Speed limit point**:
  - For gravity waves in an incompressible tank, $\sqrt{gh}$ is the **upper limit** on gravity‑wave propagation speed.
  - Faster signals (sound in water, structural vibrations) do not produce the free‑surface motions you are measuring.

---

# 5. Parasitic wavemaker effects

**Issue you raised**

- Beyond seiche and reflections, when should you worry about parasitic wavemaker effects?

**Key points**

Parasitic effects to consider:

1. **Start‑up/shut‑down transients** – you’re already handling these by discarding early time and fitting $C_0$, $C_1$.

2. **Reflections/re‑reflections at the wavemaker** – relevant if:
   - your analysis window overlaps incoming reflections,
   - you see beating or standing‑wave patterns.
   → Mitigation: keep analysis well before strong reflections, use multiple probes to check spatial patterns.

3. **Non‑ideal wavemaker motion** (what’s usually meant by “parasitic”):
   - extra harmonics (2$f_p$, 3$f_p$…),
   - low‑frequency surge,
   - high‑frequency mechanical vibration,
   - clipping at stroke limits.

You should:

- Compare **command vs measured paddle motion**:
  - check for non‑sinusoidal components at other frequencies.
- Check probe spectra for:
  - unexplained peaks that correlate with paddle motion.

Rule of thumb:

- Worry if spurious components within ~0.5–3$f_p$ exceed a few percent of the main harmonic and affect your conclusions.
- Otherwise, they are background that you filter out or ignore.

---

# 6. Wind forcing: what matters and how to characterise it

**Issues you raised**

- You have wind over the tank as well as paddle waves.
- What do you need to know about the wind?
- How does wind‑wave literature usually treat this?

**Key points**

What’s normally documented:

1. **Mean wind speed and profile**
   - Reference wind speed at known height:
     ```math
     U(z_\text{ref})
     ```
   - Possibly a vertical profile to show boundary‑layer shape.

2. **Fetch and uniformity**
   - Distance over water from wind inlet to your probes (fetch).
   - Whether wind is reasonably uniform along this fetch.
   - Fans run long enough to establish a steady wind field.

3. **Wind‑induced setup and currents**
   - Wind causes:
     - mean water‑level slope (setup),
     - mean current in water.
   - You handle this by:
     - discarding initial wind transients,
     - removing local mean/trend in each analysis window.

4. **Wind‑wave spectrum**
   - Wind‑only runs give $S_\eta(f)$:
     - show background energy,
     - spectral shape and peak.

For your experiment:

- Use pre‑paddle windows as **per‑run snapshots** of the wind‑wave field.
- Use long wind‑only runs as **reference** for stationarity and spectral shape.
- Compare wind‑on vs wind‑off to see physical effects on:
  - $A_\text{in}(f_p)$,
  - $A_\text{out}(f_p)$,
  - transmission $T(f_p)$,
  - and total variance/spectrum.

---

# 7. 2 s vs 3 s pre‑paddle windows: representativeness and variability

**Issues you raised**

- You started with 2 s pre‑paddle windows and then tested 3 s.
- You wanted to know:
  - how representative these short windows are vs long wind‑only runs,
  - whether 3 s is worth the change,
  - what to put in main text vs appendix.

**Key quantitative results**

From comparisons of 360 s wind‑only runs vs 2 s and 3 s snippets (numbers are representative):

- For each probe, you computed:
  - long‑run $\sigma_\eta$ (and $H_s$),
  - mean and std of $\sigma_\eta$ over many short snippets.

Example for 3 s vs long‑run (ensemble across 5 long runs and 70 3‑s snippets):

- **8804/250 (upstream)**  
  Long: $\sigma = 3.617\ \text{mm}$, $H_s = 14.467\ \text{mm}$  
  3 s: $\sigma = 3.76\ \text{mm}$ (+3.9 %), std(σ) ≈ 0.64 mm.

- **9373/170 (IN wall)**  
  Long: $\sigma = 4.281\ \text{mm}$  
  3 s: $\sigma = 4.042\ \text{mm}$ (−5.5 %), std(σ) ≈ 0.83 mm.

- **9373/340 (IN far)**  
  Long: $\sigma = 4.078\ \text{mm}$  
  3 s: $\sigma = 4.123\ \text{mm}$ (+1.1 %), std(σ) ≈ 0.60 mm.

- **12400/250 (OUT)**  
  Long: $\sigma = 0.36\ \text{mm}$  
  3 s: $\sigma = 0.329\ \text{mm}$ (−8.6 %), std(σ) ≈ 0.044 mm.

Comparing 2 s vs 3 s snippets:

- Bias vs long‑run changes only a little; in some probes 3 s is marginally better.
- Per‑snippet noise (std(σ) over snippets) reduces by about **15–20 %** going from 2 s → 3 s, consistent with longer window → more samples.

**Conclusions**

- The short pre‑paddle windows (2–3 s) are **statistically consistent** with the 360 s reference:
  - differences in $\sigma_\eta$ usually within 5–10 %.
- 3 s provides:
  - modestly lower bias (at some probes),
  - and clearly lower per‑snippet noise (~15–20 %).
- 3 s is safe in terms of time margin before the wave train arrives and is a good compromise.

**Recommended usage**

- Adopt **3 s pre‑paddle windows** as standard for wind‑only characterisation across runs.
- Treat these as per‑run measures of:
  - wind‑wave $H_s$,
  - wind‑wave spectra (approximately),
  - and run‑to‑run variability of the wind forcing.

---

# 8. What belongs in main text vs appendix

**Main text**

Keep high‑level decisions and their justification:

- Use of 3 s pre‑paddle windows:
  - bias vs long‑run within ~5–10 % for σ,
  - per‑run variability of σ/Hs (e.g. 15–20 % CV at IN probes).
- Statement that:
  - pre‑paddle windows are representative of wind‑only conditions,
  - they quantify run‑to‑run variability of the wind background.
- One compact table or figure summarizing:
  - long‑run vs 3 s σ and their differences,
  - run‑to‑run std(σ).

**Appendix**

Put supporting details:

- Full 2 s vs 3 s comparison:
  - side‑by‑side table with Δ(%),
  - per‑snippet std for 2 s and 3 s,
  - % reduction in noise.
- Full ensemble stats:
  - means and stds for mean, σ, $H_s$ across long runs and all pre‑paddle snippets.
- Spectral comparisons:
  - 360 s vs 2 s vs 3 s spectrum envelopes.
- Any QC plots (time‑series of σ over runs, histograms/boxplots).

---

If you’d like, I can next help you draft:

- a short “Wave and Wind Forcing” methods subsection that captures the essentials of all this,
- and a matching “Data quality and representativeness” appendix section structure.