Let me read the key PDFs you added and extract specific findings that map to your chapters.

Plant 1982 read. Let me read the others.

Specific findings from each PDF, mapped to chapters where they apply.

## Miles 1957 → §03 Teori (wind-wave generation)
- **Eq. result:** energy transfer rate to a wave of speed $c$ ∝ profile curvature $-U''(y)$ at the critical height where $U(y)=c$.
- **Minimum wind for initiation:** ~100 cm/s against laminar dissipation. Your lowest wind (3.9 m/s) is well above; you can state your regime is firmly above the Miles threshold.
- **Direct quote you can use:** "The results also may have application to the flutter of membranes and panels." Miles himself flagged the membrane application — perfect lead-in for your thesis topic.

## Phillips 1957 → §03 Teori (initial stage / "vinden generer bølger")
- **Two-stage growth:** initial (linear-in-$t$, resonance with turbulent pressure components moving at $c$) and principal (until nonlinearity dominates).
- **Initial-stage wave spectrum:** $\overline{\zeta^2} \sim \overline{p^2}\, t / (2\sqrt{2}\,\rho^2 U_c g)$.
- **Most prominent wavelength:** $\lambda_{cr} \approx 1.7$ cm (capillary–gravity minimum). You see wind waves at 3–5 Hz in your spectra (§FFT) — these are *short* gravity waves, *above* the Phillips initial scale, consistent with finite-fetch development.

## Plant 1982 → §03 + §05 (wind effect on damping)
- **Master formula:** $\beta = (0.04\pm0.02)\,u_*^2\,\omega\cos\theta/c^2$. Holds across two orders of magnitude in $u_*/c$.
- **Applicable range:** $g/(2\pi U_{10})$ to 20 Hz — this brackets your wind-wave band (3–5 Hz) entirely.
- **Concrete number for your case:** with $u_* \approx 0.2$ m/s (estimated from your 6.1 m/s log-profile) and $f = 1.4$ Hz ($c \approx 1.1$ m/s), $\beta \sim 10^{-3}$ s⁻¹ — small over your 7 s window. Useful for arguing that *your generated wave* is essentially unaffected by direct wind input over the tank residence time, so the damping you observe is due to the panel, not direct wind growth/dissipation of the carrier.
- **Slope limit (Eq. 25):** total mean-square upwind/downwind slope ≤ 0.04 ± 0.02. Your wind-wave band must obey this. Useful sanity check.

## Janssen 1989 → §04 Vindmålinger / §grensesjikt
- **Wave-induced stress = 20%–100% of total wind stress**, strongly dependent on wave age $c_p/u_*$.
- **Young wind sea (lab tank) ⇒ strong coupling.** Your $c_p/u_* < 4$ regime sits squarely in the young-wave domain, so wave-induced stress matters: the log-law roughness length $z_0$ is *not* purely Charnock — it includes a wave-age contribution.
- **Implication for your wind profile fits:** when you compare your profile to Vollestad & Jensen, differences in fetch/water-depth shift wave age and hence $z_0$. This is *the* citation to explain why your profile differs even with the same fans.

## Mansard & Funke 1980 → §04 Bølgemålinger (probe positions)
- **Two-probe Goda-Suzuki fails when $x/L = n/2$** — the equations become singular. With your probes at 8.80 m and 9.37 m (Δ = 0.57 m), check: at 1.3 Hz, $\lambda = 0.92$ m so $\Delta/\lambda = 0.62$ — close to the half-wavelength singularity. **This is a real risk** for your two-probe setup at low frequencies.
- **Three-probe least-squares fix:** their Eqs. (1)–(8) give the formulation. You have probes at 8.80, 9.37, and 12.4 m — exactly the three-probe geometry. Implementing this would let you separate genuine incident amplitude from reflections (your strand reflects 3–10%) and would tighten your transmission ratio $T$.
- **Sensitivity benefits over 2-probe:** robustness to transversal modes, harmonics, and signal noise — exactly the artefacts your wind cases suffer from.

## Welch 1967 → §04 Fourierdekomposisjon
- **Method:** divide record into K segments of length L (with optional overlap), apply window $W_1$ (parabolic) or $W_2$ (Parzen), FFT each, average.
- **Variance reduction:** $\text{Var}(\hat{P}) \propto P^2/K$. With 50% overlap and $W_1$, the effective variance reduction is 11/18 per nonoverlap unit — almost as good as nonoverlapping.
- **Equivalent degrees of freedom:** $\approx 2.8\,N\Delta f$. For your 7 s window at 250 Hz, going to 4 overlapping 2-s segments gives ~11 EDOF instead of 1.
- **Direct application:** your wind-on FFT plots will be much cleaner if you average periodograms à la Welch instead of one big FFT. Your "vindbølger synlig på 3-5 Hz" claim becomes statistically defensible.

## Watanabe, Utsunomiya & Wang 2004 → §03 Teori (membran/elastisk lag)
- **Governing equation for mat-like VLFS:** $D\nabla^4 w - \sigma^2\gamma w + \rho g w = p(x,y)$ — standard plate equation on elastic foundation. This is the equation Maugsten/McGuire/Jacobsen all build from.
- **Free-edge BCs:** zero shear $\partial^3 w/\partial n^3 + (2-\nu)\partial^3 w / \partial n \partial s^2 = 0$ and zero moment. Your back end is free, front end moored — this is the relevant BC.
- **Frequency-domain modal expansion (dry vs. wet modes):** explains directly the wet-mode language used by McGuire 2024.
- **Use as the *canonical* citation** when you write "linear hydroelastic theory of pontoon-type VLFS" anywhere in §03.

## Jeffreys 1925 → §03 Teori (historical paragraph only)
- Sheltering hypothesis: separation over crests creates pressure phase-shifted from slope.
- Cited by Miles 1957 (model 2) and Phillips 1957 as the predecessor.
- **Use it only if you write a 2-sentence history paragraph** — otherwise skip.

## Liu et al. 2024 (FPV review) → §forord/intro and §06 Discussion
- Confirms two main offshore-FPV categories: **rigid pontoon** and **flexible/membrane** — your work is in the second.
- **HDPE floats with semi-rigid/flexible connections** is the dominant offshore design.
- **Thin-film a-Si modules** are explicitly noted as enabling membrane-type FPV ("can directly interact with water, moving with the waves") — this is the technological motivation for studying *deformable* (not rigid) structures, which is what your panels approximate.
- Useful as the up-to-date context citation in the introduction; replaces older Trapani 2015 if you only have room for one FPV review.

## Concrete next steps for the thesis
1. **Highest-leverage technical addition:** implement Mansard–Funke 3-probe analysis on your existing data. Your geometry already supports it.
2. **Highest-leverage prose addition:** in §04 Vindmålinger, cite Janssen 1989 to explain the wind-profile difference vs. Vollestad & Jensen.
3. **Cleanest spectrum result:** rerun §FFT plots with Welch (overlapping segments + window) — your 3–5 Hz wind-wave peak will sharpen.
4. **Theory framing:** anchor §03 elastic-membrane subsection in Watanabe 2004 plate equation, then cite McGuire/Jacobsen/Maugsten as your specific implementations.