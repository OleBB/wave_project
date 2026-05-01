Below is a structured critical‑review questionnaire that Claude can use while scrutinising your thesis. It’s organised by topic and phrased as concrete questions/checks.

---

# A. Overall structure and scientific argument

1. **Research questions and scope**
   - Are the main research questions stated clearly and early?
   - Is it unambiguous what is *in* scope (e.g. regular waves at $f_p$, transmission under wind) and what is *out* of scope (full irregular seas, fully developed wind‑sea modeling, etc.)?
   - Do all later analyses clearly connect back to these questions?

2. **Logical flow**
   - Does each chapter/section follow logically from the previous one?
   - Are major claims supported at the point they are made, or are justifications deferred too far (or never given)?
   - Is it always clear whether a statement is:
     - an observation from your data,
     - a standard result from literature,
     - or your interpretation/guess?

3. **Use of literature**
   - Are key concepts (seiche, shallow‑water speed, RAOs, wind‑wave spectra, etc.) properly referenced?
   - Where you claim “this is standard practice” (e.g. discarding transients, using RAOs), is at least one credible source cited (ITTC, a recognised lab paper, standard text)?
   - Are you honest about what the literature does *not* say explicitly (e.g. long‑wave transients from startup), vs what you are inferring?

---

# B. Experimental setup and boundary conditions

1. **Tank and boundaries**
   - Are tank dimensions (length, width, depth) clearly stated?
   - Are all boundaries described: type of wavemaker, opposite wall / beach, side walls?
   - Is it clear where each probe is located (coordinates, relation to wavemaker and structure) and how accurate those positions are?

2. **Wavemaker**
   - Is the wavemaker type (piston/flap, control mode) described sufficiently?
   - Are command and actual motion distinguished, and do you show or describe evidence that the motion was a clean sinusoid at $f_p$ (or acknowledge and quantify deviations)?

3. **Wind generation**
   - Is the wind facility (fans, ducting, opening) described clearly?
   - Are the reference height and position of the wind measurements stated?
   - Is fetch (distance from wind inlet to each probe) clearly described and used consistently?

4. **Structure/obstacle (for transmission)**
   - Is the geometry and position of the porous/damping section described well enough that another lab could reproduce it?
   - Are relevant length scales (relative to wavelength and depth) given?

---

# C. Wave and wind measurement: sensors and calibration

1. **Wave gauges**
   - Are gauge types, sampling rate, resolution and calibration methods clearly specified?
   - Is there evidence that the calibration is stable across days (or a statement about how that was checked)?

2. **Wind measurement**
   - Is the type of anemometer and its calibration described?
   - Are limitations acknowledged (e.g. only one height, no full profile)?
   - Are you clear when “wind speed” is sensor reading vs nominal fan setting?

3. **Synchronisation**
   - Is it clear how time synchronisation between paddle, gauges, and wind sensors was handled?
   - Are any known timing offsets documented and accounted for?

---

# D. Signal processing and windowing (waves)

1. **Choice of window lengths**
   - Are the 3 s pre‑paddle windows justified quantitatively using your comparisons with long wind‑only runs?
   - Is the choice of 10T windows for fundamental amplitude estimation justified (e.g. enough cycles to estimate $A$, short enough to assume stationarity)?
   - Are margins to contamination by reflections and by the ramp‑up of the wave train clearly shown (time diagrams or figures)?

2. **Treatment of transients**
   - Do you clearly mark and discard initial transients:
     - wind startup,
     - long‑wave/seiche startup,
     - wave‑train ramp‑up?
   - Do you show at least one example time series with these regions visually indicated?

3. **Mean and trend removal**
   - In each analysis window, do you consistently state what was removed:
     - constant only ($C_0$),
     - constant + linear trend ($C_0 + C_1 t$),
     - or more?
   - Is there a clear justification for using (or not using) $C_1 t$ (i.e. to handle slow seiche / setup drift)?

4. **Harmonic fitting at the paddle frequency**
   - Is the fitting model written down explicitly (e.g. $\eta = C_0 + C_1 t + A_c\cos(2\pi f_p t) + A_s\sin(2\pi f_p t)$)?
   - Is it clear how you obtain $A$ and $\phi$ from $A_c$ and $A_s$?
   - Is there a check that residuals (observed minus fitted) are consistent with “noise + wind waves” (e.g. no obvious leftover fundamental)?

5. **Spectral analysis**
   - Are PSD estimation methods specified (windowing, segment length, averaging)?
   - When comparing 3 s snippets vs 360 s runs, do you clearly distinguish:
     - median spectra,
     - sample‑to‑sample variability (envelopes),
     - resolution limitations of short windows?

---

# E. Long‑wave/seiche and reflection control

1. **Identification of long‑wave transients**
   - Do you demonstrate (with plots and numbers) that the fast initial disturbance propagates at about $\sqrt{gh}$ and is therefore a long wave, not the 1.3 Hz phase speed?
   - Do you clearly distinguish between:
     - the long‑wave pulse,
     - its subsequent seiche‑like oscillation,
     - and the arrival of the regular 1.3 Hz wave train?

2. **Handling of seiche and setup**
   - Is it explicit that slow variations (long‑wave, setup, drift) are captured by $C_0$ and $C_1$ within each window and thus excluded from the $f_p$ amplitude?
   - Is there any residual seiche that could be large enough to bias amplitude estimates? If so, is this acknowledged and bounded?

3. **Reflection timing**
   - Is the expected time of first significant reflection from the far end computed and shown?
   - Are analysis windows chosen to avoid strong contamination by reflections (and is that shown visually for at least one run)?

---

# F. Wind characterisation and variability

1. **Wind‑only background vs pre‑paddle snippets**
   - Is the comparison between 360 s wind‑only runs and 3 s pre‑paddle snippets clearly presented?
   - Do you show:
     - long‑run mean and σ,
     - snippet mean and σ,
     - and snippet‑to‑snippet std of σ,
     for each probe?
   - Do you quantify bias (Δ%) and variability so readers can judge if 3 s is “good enough”?

2. **Run‑to‑run variability**
   - Is there a figure or table summarising how $\sigma_\eta$ or $H_s$ varies across all runs for a given wind condition?
   - Is the magnitude of this variability (e.g. ±15–20 % at IN probes) clearly stated and discussed in terms of implications for uncertainty in transmission estimates?

3. **Spectral consistency**
   - Do you provide at least one spectral comparison:
     - long wind‑only vs ensemble of 3 s snippets?
   - Does the text explain whether spectral shape is stable over time, or whether there are systematic drifts (e.g. more high‑frequency noise later in the campaign)?

4. **Possible fan/voltage drift**
   - Do you address your own suspicion about voltage fluctuations by:
     - looking for trends in σ/Hs across runs,
     - or correlating wave statistics with measured wind speeds where available?
   - Are conclusions about fan stability modest and evidence‑based (e.g. “no obvious long‑term trend within X %”) rather than speculative?

---

# G. Wavemaker parasitic effects and harmonics

1. **Evidence of purity of the fundamental**
   - Is there a spectrum of a representative wave gauge showing the dominance of the line at $f_p$?
   - Are higher harmonics (2$f_p$, 3$f_p$) present and, if so, are their magnitudes quantified relative to the fundamental?

2. **Non‑ideal wavemaker motion**
   - Is any deviation between commanded and measured paddle motion discussed?
   - If such deviations exist, is it clear whether they affect:
     - the fundamental at $f_p$,
     - or mostly higher harmonics / low‑frequency components?

3. **Impact on results**
   - For the frequencies of interest (at and near $f_p$), is there a clear argument that parasitic components are small compared to the main wave, or that they are handled appropriately (e.g. not included in $A$ at $f_p$)?

---

# H. Main wave amplitude, “total” variance, and transmission

1. **Definition of amplitude metrics**
   - Is the fundamental amplitude at $f_p$ defined clearly from the LS fit?
   - Is the “total variance” / $H_s$ in a window defined clearly (and compatible with standard definitions)?

2. **Separation of main wave from background**
   - Do you explain that $A$ at $f_p$ represents the regular (paddle‑forced) component, while $\sigma_\eta$ and $H_s$ include wind waves and other noise?
   - If you use ratios like $A/H_s$ or variance fractions, are they explained physically (e.g. “fraction of energy at the fundamental”)?

3. **Transmission calculation**
   - Is the transmission coefficient $T(f_p)$ defined clearly (e.g. as $A_\text{out} / A_\text{in}$ or some height-based equivalent)?
   - Is it clear how any differences in background noise between IN and OUT are accounted for (e.g. by focusing solely on the line at $f_p$)?

---

# I. Uncertainty, sensitivity, and robustness

1. **Uncertainty quantification**
   - Do you provide error bars or confidence intervals on key quantities: $A_\text{in}$, $A_\text{out}$, $T$?
   - Are these uncertainties tied back to measured variability:
     - in the 3 s wind snippets,
     - between repeated runs,
     - between different windows within a run?

2. **Sensitivity to choices**
   - Is there at least a brief discussion of how sensitive results are to:
     - window length (2 s vs 3 s pre‑paddle, or 8T vs 10T for wave amplitude),
     - detrending choice (mean vs mean+trend)?
   - If alternative choices were tested (e.g. 2 s vs 3 s), are they summarised and concluded on in a way that supports the final choices?

3. **Potential biases**
   - Are known limitations (e.g. residual seiche, imperfect wind uniformity, finite tank length, non‑ideal wavemaker motion) acknowledged explicitly?
   - Is there at least a qualitative argument that these effects are small compared to the main patterns you report?

---

# J. Clarity of figures, tables, and explanations

1. **Figures**
   - Are plots labeled clearly (axes, units, probe IDs, wind conditions)?
   - Wherever interpretation is non‑trivial (e.g. seiche vs wave‑train arrival, spectrum envelopes), does the caption and text guide the reader enough to understand what to look at?

2. **Tables**
   - Are key summary tables (like the long‑run vs 3 s σ comparison) kept compact and used to support specific claims in the main text?
   - Are more detailed tables appropriately relegated to the appendix?

3. **Consistency**
   - Are probe names, symbols, and colours used consistently across figures and text (e.g. IN‑wall always the same colour)?
   - Are notations like $f_p$, $H_s$, $\sigma_\eta$, $A$, etc. defined once and used consistently?

---

Claude can go through this checklist section by section and, for each question, mark:

- **Yes**: clearly and adequately addressed.
- **Partly**: addressed but needs clarification or stronger evidence.
- **No / unclear**: gap in explanation, missing plot/table, or unsupported assumption.

Any “Partly” or “No” answers become targeted edits or additions.