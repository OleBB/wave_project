What you’re seeing is a very typical and actually quite *nice* situation: your probes are precise enough that, in still water, their noise is dominated by the digital resolution of the measurement system rather than by wild random fluctuations. That has very concrete implications for how to think about “uncertainty” and “minimum detectable wave amplitude”.

I’ll break this down in a way you can re‑read later when rested.

---

## 1. What the still‑water data actually look like

In still water, a “perfect” instrument would measure a perfectly flat line. In reality you see:

- a **slowly varying mean level** (small drift over many seconds), plus
- **small high‑frequency jitter** sample‑to‑sample.

In your zoomed snapshot, that jitter looks like:

- values toggling between a few discrete numbers, e.g. `0.10028` and `0.10036` m,
- with rare excursions to `0.10020` or `0.10044` m.

Two important observations:

1. These values only take *discrete* steps, not a continuum.
2. The step size is of order  

$$
\Delta z \approx 0.00008 \ \text{m} = 0.08 \ \text{mm}.$$

That means:

- physically, the water and sensor are probably fluctuating with some small amplitude,
- but numerically, the recorded values can only jump in increments of about $0.08\,$mm.

This is **quantization**.

---

## 2. Quantization: why the signal seems to jump between 2–3 levels

Any digital measurement system ultimately represents your continuous water surface height $\eta(t)$ as discrete numeric values:

$$$  
\eta_\text{meas}(t) = Q\bigl( \eta_\text{true}(t) + \text{analog noise} \bigr),
$$

where $Q(\cdot)$ is a *quantizer* that maps a continuous input to one of several discrete output values:

- Imagine the vertical axis divided into bins of width $q$ (the *quantization step*).
- Any true value inside a bin is rounded to the bin’s central value (or lower/upper edge, depending on the hardware).

If the *true* variation of the signal over short time scales is **smaller than or comparable to** $q$, then:

- most of the time the measured signal will sit on **one or two adjacent quantization levels**;
- occasional slightly larger excursions will flip it to the next bin.

That’s exactly what your zoomed snippet shows: you’re mostly quantization‑limited.

When you zoom out (larger vertical scale, longer time window), this looks like a very steady signal with tiny jitter. When you zoom in too far, you suddenly *notice* the discrete nature and it looks like “jitter between two lines”.

So this is not an illusion; it’s a real property of the instrument:

- it has **finite vertical resolution**,
- and your *noise* is small enough that you’re often bumping into that resolution limit.

---

## 3. How you quantify noise and resolution from the data

In your pipeline, you’ve defined several metrics from still‑water data:

1. **RMS noise** (random uncertainty)

   For each probe and configuration you compute the standard deviation of the still‑water signal:

   $
   \sigma = \sqrt{\frac{1}{N - 1} \sum_{i=1}^N (\eta_i - \bar{\eta})^2}\,,
   $

   where $\eta_i$ is the recorded surface elevation (in mm) in still water, and $\bar{\eta}$ is its mean.

   This gives you a measure of **random high‑frequency variability**, in mm.

2. **95 % noise amplitude**

   Instead of assuming Gaussian statistics, you take a robust percentile‑based amplitude:

   $
   A_{95} = \frac{P_{97.5} - P_{2.5}}{2},
   $

   where $P_{x}$ is the $x$‑th percentile of the still‑water samples. For roughly Gaussian noise you expect:

   $
   A_{95} \approx 1.96 \,\sigma.
   $

   So this is basically “half the central 95 % spread” around the mean water level.

3. **Quantization step** (effective vertical resolution)

   From your processed time series, you compute:

   $
   \Delta \eta_i = |\eta_{i+1} - \eta_i|
   $

   and then take

   $
   q = \text{P5}\bigl(\Delta \eta_i \mid \Delta \eta_i > 0\bigr),
   $

   i.e. the 5th percentile of non‑zero sample‑to‑sample differences. This is clever because:

   - zeros correspond to repeated identical values,
   - the smallest non‑zero differences are dominated by the quantization step,
   - taking P5 discards rare larger jumps and gives a robust estimate of the *typical minimum resolved step*.

   That’s how you ended up with `q = 0.03`–`0.05` mm per probe in your summary table.

4. **Detection threshold**

   You then define a minimum detectable wave amplitude:

   $
   A_\text{th} = \max(3\sigma,\; 2q).
   $

   Interpretation:

   - $3\sigma$ is a “3‑sigma” noise threshold: a signal of amplitude $< 3\sigma$ is very likely to be buried in random noise.
   - $2q$ ensures that any “detectable” wave change is at least **two quantization steps**, i.e. clearly above the digital resolution limit.
   - Taking the maximum ensures you are conservative with respect to both effects: random noise and digitization.

In your current data:

- some probes/configs are **noise‑limited**: $3\sigma > 2q$ → threshold $\approx 3\sigma$,
- others are **quantization‑limited**: $3\sigma < 2q$ → threshold $= 2q$.

---

## 4. Why the red detection line is always above the blue noise bar

In the plot:

- blue bar = $\text{mean } A_{95}$ per probe,
- dashed red line = $A_\text{th} = \max(3\sigma, 2q)$.

Because, for Gaussian‑ish noise,

$
A_{95} \approx 1.96\sigma,
$

and your threshold is $3\sigma$ or larger, you *expect*:

$
A_\text{th} \gtrsim 3\sigma \approx 1.5 \times A_{95}.
$

So the red line should sit comfortably above the blue bar for all probes. That has a clear physical meaning:

- the blue bar says: “this is the *typical* still‑water noise amplitude”.
- the red line says: “I won’t claim to see a wave unless its amplitude is clearly larger than the typical noise”.

That’s exactly how a detection threshold should behave.

---

## 5. How this affects your wave measurements

### 5.1. Minimum resolvable wave amplitude

The single most important practical consequence is:

- **Any wave whose amplitude is below $A_\text{th}$ is not reliably detectable.**

For example, if for some probe in some configuration:

- $\sigma = 0.03$ mm, $q = 0.05$ mm,
- then $3\sigma = 0.09$ mm, $2q = 0.10$ mm → $A_\text{th} = 0.10$ mm.

This means:

- waves with **peak amplitude** of, say, $0.03$ or $0.05$ mm will produce changes on the order of one quantization level and/or be comparable to random jitter;
- you cannot safely distinguish them from noise.

By contrast, a 1 mm wave is a factor 10 above the threshold and will be very clearly visible.

For your **spectral analysis** or **amplitude statistics**:

- low‑amplitude components near or below $A_\text{th}$ will be strongly contaminated by noise,
- for scientific interpretation, you may want to clip or at least mark frequencies / amplitudes below that threshold as “unreliable”.

### 5.2. Bias vs noise

From still water, you also get:

- **mean_level_mm**: median or mean still‑water level per probe,
- **bias_vs_ref_mm**: systematic offset relative to the cross‑probe mean.

This gives you a **systematic error** (bias) per probe. For later wave runs, you can:

- subtract these biases to align all probes to a common reference,
- while still associating them with a random uncertainty of about $\sigma$.

So your **total probe uncertainty** has two components:

1. **Random**: $\sigma$ (RMS noise), plus quantization derivative.
2. **Systematic**: bias vs reference (constant offset).

### 5.3. Interpretation of small differences between probes

When you compare wave amplitudes measured by different probes:

- small differences on the order of a few $\sigma$ or a few $q$ are not physically significant; they’re within the expected “noise + resolution” envelope.
- only differences much larger than $A_\text{th}$ per probe are evidence of genuinely different responses (e.g. due to position or calibration).

Your still‑water analysis gives you a concrete way to say:

> “Differences below X mm between probes are within the instrument uncertainty and should not be over‑interpreted.”

Where X is some multiple of $\sigma$ or $q$ based on your tolerance.

---

## 6. How you might describe this in your thesis

Here’s a compact but precise way you could explain this process in your methods:

> In order to characterise the measurement uncertainty of the ultrasound wave gauges, we recorded still‑water signals (no wind, no imposed waves) at the full sampling rate. For each probe and hardware configuration, we compute the RMS still‑water noise $\sigma$ (standard deviation of the raw signal) and a robust 95 % noise amplitude $(P_{97.5} - P_{2.5})/2$, where $P_{x}$ denotes the $x$‑th percentile. The effective vertical resolution (quantization step) $q$ is estimated as the 5th percentile of non‑zero sample‑to‑sample differences $|\Delta \eta|$ in still water. In many configurations the short‑term fluctuations are comparable to this quantization step, so the recorded signal often toggles between only two or three discrete levels when viewed at high zoom; this indicates that the gauges are quantization‑limited in still water rather than dominated by analog noise.
>
> We define the minimum detectable wave amplitude as
> $
> A_\text{th} = \max(3\,\sigma,\; 2\,q),
> $
> i.e. the larger of a 3‑sigma noise threshold and twice the quantization step. Wave amplitudes below $A_\text{th}$ cannot be reliably distinguished from still‑water noise and digitization effects, and are therefore treated as sub‑threshold. This detection threshold is used in subsequent analysis to interpret small wave amplitudes and to assess the significance of differences between probes.

You can add:

> For each probe and configuration we also compute the mean still‑water level and a bias relative to the configuration‑averaged level. These systematic offsets are used to align probes to a common reference level, while the noise and quantization metrics define the random measurement uncertainty.

---

## 7. Mental summary for future you

When you come back to this tired:

- The signal looks like it flips between 2–3 levels because the **instrument’s vertical resolution is finite** and your noise is small.
- You measured:
  - **RMS noise** $\sigma$,
  - **95 % noise amplitude** $A_{95}$,
  - **quantization step** $q$ from the smallest nonzero sample jumps.
- You define a **detection threshold** $A_\text{th} = \max(3\sigma, 2q)$ for each probe/config.
- The **red line above the blue bar** in the plot is exactly what this definition predicts and what you want: a conservative line above typical noise.
- Any wave smaller than that line is statistically indistinguishable from still‑water noise + digitization, so you shouldn’t over‑interpret it.

If you want later, you can paste a small chunk of the code that computes $q$ and the thresholds, and I can annotate it line‑by‑line in the same conceptual terms.
