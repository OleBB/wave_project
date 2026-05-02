## Parallel-probe agreement at the IN reference distance

**Table X.** Pairwise comparison of the two parallel IN-side probes (9373/170 = wall, 9373/340 = far) across the four thesis paddle frequencies. All statistics computed from N = 80 paired runs (panel-full, quality-ok, both probes present) drawn from the canonical March 2026 lowrange dataset. Sign conventions and column definitions in the footnotes.

| f [Hz] | N | Δ̄ [dB]ᵃ | σ_Δ [dB] | *p*ᵇ | r(A) ᶜ | σA_wall [mm]ᵈ | σA_far [mm]ᵈ | σA_mean [mm]ᵈ | ΔVar(mean) [%]ᵉ |
|:------:|:-:|:------:|:--------:|:-------:|:-----------:|:------------:|:-----------:|:------------:|:--------------:|
| 1.30 | 80 | −0.07 | 0.76 | 0.39 | +0.996 | 4.81 | **4.77** | 4.78 | +0.6 |
| 1.40 | 80 | −0.15 | 0.62 | **0.034** | +0.991 | 4.77 | **4.68** | 4.71 | +1.5 |
| 1.50 | 80 | −0.17 | 0.65 | **0.024** | +0.989 | 5.42 | **5.14** | 5.27 | +4.8 |
| 1.60 | 80 | +0.02 | 2.44 | 0.93 | +0.992 | 5.81 | **5.45** | 5.62 | +6.2 |

**Bolding rules.**  *p* column: bold ⇒ paired *t*-test significant at α = 0.05.  σA columns: bold ⇒ smallest σ in that row (always the far probe — see Interpretation).

**Footnotes.**
- ᵃ Δ̄ = mean across runs of (10·log₁₀ P_far(f) − 10·log₁₀ P_wall(f)) at the PSD bin nearest *f*. Negative ⇒ far reads lower power than wall.
- ᵇ Two-sided paired *t*-test, H₀: Δ̄ = 0 dB.
- ᶜ Pearson correlation across the N runs between the two probes' band-integrated amplitudes A = √(2 · ∫ S(f) df) over a ±0.1 Hz window centred on *f*.
- ᵈ σA = standard deviation across runs of A (units: mm). Pooled across condition cells, so it includes systematic variation from amplitude tier and wind setting in addition to measurement reproducibility.
- ᵉ Percent change in Var(½(A_wall + A_far)) relative to min(Var(A_wall), Var(A_far)). Positive ⇒ averaging *worsens* precision compared with the better single probe.

### Verdict per frequency

| f [Hz] | Verdict |
|:------:|:--------|
| 1.30 | Probes statistically indistinguishable; either suffices. |
| 1.40 | Small but significant mean offset (far ≈ 3.4 % lower power); single-probe precision is essentially equal. |
| 1.50 | Small but significant mean offset (far ≈ 3.8 % lower power); far slightly more precise. |
| 1.60 | No detectable mean offset; σ_Δ inflated by run-level variability; far meaningfully more precise. |

---

## Headline findings

- **Mean spectral level.** The two probes agree to within |Δ̄| ≤ 0.17 dB (≲ 4 % of power) at every thesis frequency. The far probe reads marginally lower at 1.3–1.5 Hz; the offset crosses statistical significance at 1.4 and 1.5 Hz (*p* = 0.034, 0.024) but is below the practical resolution at which thesis amplitudes are reported.
- **Cross-run correlation.** Band-integrated amplitudes correlate at r ≥ 0.989 at every frequency. The two probes are sampling the same wave field, not two independent samples of it.
- **Per-probe precision.** The far probe has the smaller σA at every frequency (by 0.5 % at 1.3 Hz, rising to 6 % at 1.6 Hz).
- **Variance reduction from averaging.** None: ΔVar(mean) is positive at every frequency (+0.6 % to +6.2 %). Averaging the two probes is *worse* than using the far probe alone.

## Interpretation

A correlation of r ≈ 0.99 between two measurements means that essentially all of the run-to-run variation in one probe is shared with the other; almost none is independent. Both probes are responding to the same paddle-driven incident wave, with negligible probe-specific contribution from independent wind-driven chop. In that regime the classical variance reduction from averaging — which assumes partially independent measurements — does not apply. For two correlated observations with variances V_w, V_f and Pearson correlation ρ,

> Var(½(A_w + A_f)) = ¼ (V_w + V_f + 2 ρ √(V_w V_f)),

so ρ → 1 collapses the average's variance toward (½(σ_w + σ_f))² rather than min(V_w, V_f). When V_w ≠ V_f — as here, where the wall probe is consistently the noisier of the two — the mean inherits part of the wall probe's excess variance and lands above the better single estimator's variance. This is exactly the +0.6 % to +6.2 % inflation reported in the rightmost column.

Two practical conclusions follow. First, the **far probe is adopted as the marginally better single estimator of the monochromatic IN amplitude**. Second, the canonical IN reference value used elsewhere in this work — the *mean* of the two parallel probes — is retained for symmetry with the OUT side and for definitional simplicity, not because it improves precision; the precision penalty is small (≤ 6 %) and the resulting estimate sits between the two probes by construction, so it remains a defensible summary of the lateral wave field.