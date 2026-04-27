# Agent brief — hydroelastic verification of wave-tank transmission data

## Purpose

This file briefs a fresh agent on a set of analyses to run against the existing
wave-tank data for Ole Sandok's MEK5960 thesis. The thesis studies wind + wave
effects on a chain of floating plates connected by elastic bands (a model of a
membrane-type floating PV system).

The user has already built the experiment, processed time series, and computed
transmission coefficients $T = a_{\text{out}}/a_{\text{in}}$. What is missing is
a layer of quantitative comparison against the linear hydroelastic theory
documented in two UiO master theses (McGuire 2024, Maugsten 2023) and the
canonical wave-tank reflection-analysis methods (Goda–Suzuki 1976,
Mansard–Funke 1980).

The agent's job is to *add* analyses on top of existing data, not to re-do the
data acquisition. Treat the existing time series and FFT pipeline as ground
truth.

---

## Hard constraints

1. **Do not edit any `.tex` file without explicit user permission.** The user
   writes the prose. The agent produces figures, tables, numbers, and code.
2. **Never write Norwegian.** Comments, prose, captions, log output — English
   only.
3. **The `.bib` file is *not* a `.tex` file.** It can be appended to, but ask
   first; the user uses Better-BibTeX with specific citekey conventions.
4. Write code into a clearly-marked working directory (suggest
   `/Users/ole/main/agent_work/` — create if missing). Do not scatter scripts
   across the project.
5. Output figures as both `.pdf` (for `\includegraphics`) and `.png` (for
   inspection). Place them under `FIGURES/agent/` so the user can later move
   them.

---

## Experiment summary (so the agent has context)

- **Tank:** UiO long wave tank, 26.8 m × 0.50 m × 0.96 m, water depth 0.580 m.
- **Wave maker:** hydraulic paddle at one end. Drive signal in volts;
  amplitude is approximately linear in volts (Osyka 2016).
- **Beach:** mesh-stack absorber, 3–10 % reflection.
- **Geometry under test:** a chain of 24 thin plastic plates joined by two
  long rubber bands (lengthwise), moored only at the upstream end with a 30 cm
  rope at 9 cm depth. Plate widths: first 12 are 8 cm, last 12 are 12 cm.
  Total chain length $l = 2.6$ m.
- **Wind tunnel:** partial roof, 18.6 m long, 6.4 m of open water upstream of
  the tunnel. Two Heylo FD4000 fans pulling air through. Two wind levels:
  3.9 m/s and 6.1 m/s at 10 cm above water.
- **Wave probes (ultrasonic, General Acoustics, 250 Hz, 0.18 mm resolution):**
  - Position 1: 8.80 m from paddle (parallel/auxiliary)
  - Position 2: 9.37 m from paddle (incident reference)
  - Position 3: 12.4 m from paddle (transmitted)
  - Geometry sits between roughly 9.5 m and 12.1 m
- **Wave conditions tested:**
  - Frequencies: 1.3, 1.4, 1.5, 1.6 Hz
  - Amplitudes: $A_1, A_2, A_3$ (drive 0.1, 0.2, 0.3 V → ~8, 16, 24 mm)
  - Steepness $ka$: 0.054 to 0.247
- **Analysis window:** 7 periods between t = 35 s and 42 s (Huseby & Grue
  2000), shifted by group-velocity travel time for upstream probes.

---

## Theoretical anchor — what to verify

### A. Energy conservation (McGuire 2024, Eq. 2.128)

For an impermeable, lossless floating elastic plate under linear theory:

$$|\hat{R}|^2 + |\hat{T}|^2 = |\hat{A}|^2$$

i.e. reflected² + transmitted² should equal incident² for each wavenumber.
Any deviation from unity is dissipation: viscous, gap leakage,
panel-to-panel friction, or (in the wind-on case) wind-induced losses.

**This is the single most useful diagnostic** the experiment can produce
because it converts a noisy two-quantity comparison ($a_\text{in}, a_\text{out}$)
into a one-quantity test ($1 - |T|^2 - |R|^2 \ge 0$).

### B. Dimensionless parameters (McGuire §2.13, Maugsten §2.7)

The hydroelastic literature parameterizes the problem on
- $\Pi_3 = kl$ (wavenumber × plate length): wavelength-to-length ratio
- $\Pi_1 = D/(\rho g l^4)$ (bending stiffness, McGuire) or
- $G_3 = T k^2/(\rho g)$ (membrane tension, Maugsten)
- $\Pi_8 = k\hat{A}$ (steepness)

The user's experiment sweeps $kl$ from ~18 to ~27 and steepness from 0.05 to
0.25 at one tension/stiffness setting. **All results should be plotted against
$kl$**, not against frequency, so they sit on the same axis as McGuire's and
Maugsten's published curves.

### C. Wind growth scaling (Plant 1982, Eq. 1)

For wind-generated short waves visible in the spectrum at 3–5 Hz:

$$\beta = (0.04 \pm 0.02)\, u_*^2 \omega \cos\theta / c^2$$

valid for $g/(2\pi U_{10}) < f < 20$ Hz. Relevant for arguing that the
direct wind input to the **carrier wave** at 1.3–1.6 Hz is negligible over
the 7-period window, so observed wind-on damping changes are due to
panel/structure interaction, not direct wind input on the carrier.

---

## Concrete analyses to perform

### Analysis 1 — Implement Mansard–Funke 3-probe reflection separation

**Why:** The two-probe Goda–Suzuki method has singularities at probe
spacings $\Delta x / \lambda = n/2$. With probes at 8.80 m and 9.37 m
(spacing 0.57 m), at 1.3 Hz the spacing is 0.62 wavelengths — close to
the $n=1$ singularity. The user has *three* streamwise probes available
(8.80, 9.37, 12.4 m) — exactly the geometry Mansard & Funke 1980 designed for.

**Reference:** Mansard & Funke (1980), *Proc. 17th ICCE*, ASCE, 154–172.
PDF in `/Users/ole/Zotero/storage/69XZDVRF/`. Equations (1)–(8).

**Steps:**
1. For each (frequency, amplitude, wind) configuration, take time series from
   probes 1, 2, 3 over the same Huseby–Grue window.
2. Compute Fourier coefficients at the carrier frequency for each probe.
3. Set up the least-squares system from M&F Eq. (6)–(8) for incident and
   reflected complex amplitudes $C_I, C_R$.
4. Solve the overdetermined system. Output $|\hat{R}/\hat{A}|$ and a
   condition number / residual norm per case.
5. Sanity check: in no-panel runs, $|\hat{R}|$ should reflect only the beach
   (3–10 %).

**Caveat:** Probes 1, 2 are upstream of the panel, probe 3 is downstream.
The standard M&F derivation assumes all three probes are on the same side of
the reflector. So the **3-probe reflection analysis must use only probes 1
and 2 (both upstream of the panel)**. Probe 3 then independently gives
$|\hat{T}/\hat{A}|$ as the user already computes. The 2 vs 3-probe distinction
above is a misstatement — fix it: with two upstream probes the user is doing
Goda–Suzuki, not M&F. Honestly evaluate whether the existing two-probe
spacing is workable across all four frequencies, and report which frequencies
hit the $n/2$ degeneracy. If degeneracy is unavoidable, recommend a
follow-up experiment with a third upstream probe.

### Analysis 2 — Energy-conservation deficit plot

**Output:** one figure with three panels (1.3, 1.4, 1.5, 1.6 Hz on x-axis as
$kl$):
- Panel a: $|\hat{T}/\hat{A}|^2$ vs $kl$, colored by amplitude tier
- Panel b: $|\hat{R}/\hat{A}|^2$ vs $kl$, same coloring
- Panel c: **deficit** $\delta = 1 - |\hat{T}|^2 - |\hat{R}|^2$ vs $kl$

For each panel, plot two curves: wind-off and wind-on.

**Interpretation guide:**
- $\delta \approx 0$ in wind-off ⇒ linear theory holds, dissipation is
  small.
- $\delta > 0$ in wind-off ⇒ viscous/friction dissipation in the panel chain.
- $\delta_\text{wind} - \delta_\text{no-wind}$ ⇒ extra dissipation introduced
  by the wind. **This is the headline scientific result** rephrased as energy
  conservation.

### Analysis 3 — Welch-averaged spectra for wind-on cases

**Why:** Single-shot FFTs of wind-on data are dominated by stochastic
wind-wave variance. Welch (1967) periodogram averaging with 50 % overlap
and Hann or Welch ($1-t^2$) window reduces variance roughly by $K$ (number of
segments) at the cost of frequency resolution.

**Steps:**
1. Re-process each 7 s analysis window using Welch's method:
   - Segment length: 1 s (250 samples)
   - Overlap: 50 %
   - Window: Hann
   - This gives ~13 effective segments → variance reduction by ~10×
2. Compare to single-FFT spectra side-by-side. Quantify peak SNR
   improvement at the carrier frequency vs. the 3–5 Hz wind-wave band.
3. Output a figure for one representative case (e.g. 1.4 Hz, $A_2$, full
   wind) showing single-FFT, Welch with Hann, Welch with Welch window.
   Caption notes equivalent degrees of freedom (~2.8·N·Δf).

**Reference:** Welch (1967), IEEE Trans. AU 15, 70–73.
PDF in `/Users/ole/Zotero/storage/92JW659P/`.

### Analysis 4 — Plant 1982 growth-rate sanity check

**Why:** Reviewer might ask whether the wind directly amplifies the carrier
wave over the analysis window. The answer should be "no" but it must be
quantified.

**Steps:**
1. Estimate $u_*$ from the user's measured wind profile by fitting a log-law
   $\bar{u}(z) = (u_*/\kappa) \ln(z/z_0)$ to the data in
   `FIGURES/04_windprofile_both.pdf`. Use $\kappa = 0.41$.
2. Compute $\beta = 0.04 \cdot u_*^2 \omega / c^2$ for each carrier
   frequency, with $c = \omega/k$.
3. Compute fractional amplitude growth over the 7-period window:
   $\Delta a / a = \beta \cdot 7T$.
4. Output a small table: $(f, c, u_*, \beta, \Delta a/a)$.

The user should expect $\Delta a/a \ll 1$, justifying the claim that
wind-on changes in $|T|$ are due to panel response, not direct wind input
to the carrier.

### Analysis 5 — Maugsten short-wave regime check

**Why:** Maugsten 2023 predicts that for short waves (the user's regime,
$kl \approx 18$–$27$), symmetric and antisymmetric modes have comparable
amplitude. This means the panel chain should pitch and heave in comparable
measure.

**Steps:**
1. From video footage of the panels (if available — ask user) or from the
   transmitted-wave phase relative to incident, infer whether the panel
   chain shows pitching.
2. If video is not available, simply state in the deliverable that this is
   a qualitative prediction not directly testable with surface-elevation
   data alone, and recommend a future PIV or video-tracking experiment.

---

## Suggested deliverable

A single Markdown report at `/Users/ole/main/agent_work/REPORT.md` containing:

1. **Methods** section: code snippets for each analysis (Python, numpy, scipy)
2. **Results** section: numerical results per condition, with figures
   referenced
3. **Figures** generated under `FIGURES/agent/`
4. **Open questions** and **recommended follow-up experiments**

The user will read the report and decide which figures to drop into the
thesis. The agent **does not edit `CHAPTERS/*.tex`** unless the user
explicitly says so.

---

## File map

| What | Where |
|------|-------|
| Time-series data | (ask user — likely `FIGURES/` source data or external drive) |
| FFT pipeline source | (ask user — referenced in CH04 §FFT) |
| Existing TEXFIGU input | `/Users/ole/main/TEXFIGU/` |
| Bibliography | `/Users/ole/main/masterbibliography.bib` |
| Zotero PDFs | `/Users/ole/Zotero/storage/<key>/` |
| Key reference PDFs already added | <ul><li>Miles 1957: `68BFK2UA`</li><li>Phillips 1957: `CYP4U7YT`</li><li>Plant 1982: `IZP8AGIG`</li><li>Janssen 1989: `C2HT6TG5`</li><li>Mansard & Funke 1980: `69XZDVRF`</li><li>Welch 1967: `92JW659P`</li><li>Watanabe et al. 2004: `E9SMDT5E`</li><li>Jeffreys 1925: `HUZE6RWB`</li><li>Liu et al. 2024 FPV review: `223FQETL`</li><li>McGuire 2024: `4KMIFWXJ`</li><li>Maugsten 2023: `ITWH76CI`</li><li>Jacobsen 2024: `CVTWZB7J`</li></ul> |

---

## Priorities

If time is limited, run analyses in this order:

1. **Analysis 2 (energy-conservation deficit)** — single most informative
   figure. Needs Analysis 1 first, but if reflection cannot be cleanly
   extracted, plot $|T|^2$ deficit vs. unity as a lower bound on dissipation.
2. **Analysis 3 (Welch spectra)** — improves every wind-on figure already
   in CH04. Cheap and high-impact.
3. **Analysis 1 (Mansard–Funke / Goda–Suzuki)** — depends on probe-spacing
   feasibility check. If the spacing is degenerate, this becomes a
   recommended follow-up rather than a current-data result.
4. **Analysis 4 (Plant)** — short, defensive. One table.
5. **Analysis 5 (Maugsten short-wave)** — qualitative; only if video exists.

---

## What "done" looks like

The user can read the agent's `REPORT.md`, look at the deficit plot, and
write a single paragraph in CH06 Discussion that says:
*"We tested the linear-theory energy-conservation identity
$|T|^2 + |R|^2 = 1$ across our 12 wave conditions. In the wind-off case the
deficit is X ± Y %, consistent with viscous and structural dissipation. In
the wind-on case the deficit grows to Z ± W %, with the additional Z − X %
attributable to wind-induced losses."*

That paragraph, plus the energy-deficit figure, is the scientific payload of
the wind/no-wind comparison.
