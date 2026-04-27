Honest answer: rank your effort by what can carry a defensible scientific claim from *only* your observables. You have $\eta(t)$ at three streamwise probes — that's it. So your scientific claims live or die on what you can extract from $\eta(t)$.

## What your data can support, ranked

**1 — `T(f, ka, wind)` with error bars.** This is your one number per condition. Everything else builds on it. Get this rock-solid first.

What "rock-solid" means:
- Multiple runs per condition (you have these — verify how many)
- Error bars from run-to-run scatter, not just FFT noise within one run
- One figure: $T$ vs $kl$ (or vs frequency), three amplitude markers, with/without wind as paired points
- A defensible window-and-FFT pipeline that you can describe in two sentences

If this figure doesn't survive scrutiny, nothing built on top of it does. **Spend disproportionate time here.**

**2 — Shape check against linear theory.** Compare your no-wind $T(kl)$ curve to McGuire 2024 Fig. 7.15 (or Cho & Kim 1998 experimental Fig. for membrane). Even a *qualitative* match — "transmission decreases with $kl$, perfect-transmission dip near $kl \approx X$" — is a thesis-defensible claim. The exact numbers don't have to match (your stiffness differs from theirs); the *trend* should.

**3 — `|R|` via Goda–Suzuki where probe spacing allows.** You have probes at 8.80 m and 9.37 m (Δ = 0.57 m). At 1.4–1.6 Hz the spacing is workable; at 1.3 Hz you're near the half-wavelength singularity. Implement and report which frequencies are extractable. Then plot the energy deficit $\delta = 1 - |T|^2 - |R|^2$ vs $kl$ for those frequencies — wind on / wind off. **This is your headline figure if it works.**

**4 — Sutherland steepness-independence test.** You have three amplitudes per frequency. Sutherland et al. 2017 (your supervisor group's paper) predicts damping is *independent* of $ka$ for inextensible covers. Plot $T$ vs $ka$ at fixed $f$. Is the slope zero? If yes, your panels behave inextensibly. If no, you've found something different from Sutherland — both outcomes are publishable.

**5 — Mean water-level shift.** Quantify wind setup vs wind speed. One small figure. One paragraph in §06. Not the headline, but a free observation from data you already have.

## What your data *cannot* support — be explicit about this

State these as limitations in §06 to pre-empt examiner questions:

- **Mode shape of panel response** — you have surface elevation, not panel deflection. You cannot say which mode (rigid heave, pitch, mode 1, mode 2…) is excited.
- **Heave vs pitch decomposition** — same reason. Maugsten predicts comparable symmetric/antisymmetric in your $kl$ regime; you cannot directly verify this without video or accelerometers.
- **Direct flow-separation observation** — Vollestad & Jensen 2021 had PIV; you have none. You can only *infer* flow separation indirectly from the energy deficit.
- **Decomposition of "where the wind energy goes"** — you can quantify total deficit growth $\Delta\delta_\text{wind}$, but you cannot separate "wind dissipation in the air column" from "wind-driven pumping of the panel" from "wind-modified wave breaking on the panel front." Three mechanisms, one observable.
- **Generalization beyond your single configuration** — you tested one tension, one panel chain, one water depth, one fetch. Don't extrapolate.

## Concrete top priority — pick one thing for this week

Lock down item 1. Specifically:

- For each (f, $ka$, wind) condition, list how many independent runs you have.
- Plot $T$ for each individual run as a faint dot, mean as a marker, std as an error bar. **One figure with all conditions on it.**
- Look at it. If the error bars are tight (< 2% say), you're golden. If they're wide, find out why (drift, paddle warmup, panel position, day-to-day water level) before doing anything else.

Until that figure exists and looks clean, do not invest in items 2–5. The temptation to start on the headline figure (item 3) before item 1 is solid is real and dangerous — a noisy energy deficit looks like a wind effect even when it's just measurement noise.

## Broader scope, later

Once item 1 is locked:
- Item 2 in 2 days
- Item 3 in 4–7 days (Goda–Suzuki implementation)
- Items 4 and 5 in 1 day each

That timeline gets you to a complete results chapter in roughly two weeks of focused work. Discussion (§06) pulls in the literature you've now built up; conclusion (§07) writes itself once §05 and §06 are stable.

Don't broaden until item 1 is the bedrock. Everything you said about "wind adds energy to the system" in your abstract has to land back on this single curve.