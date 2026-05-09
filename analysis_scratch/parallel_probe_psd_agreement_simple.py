"""
Parallel-probe PSD agreement — SIMPLE table.

Sibling of analysis_scratch/parallel_probe_psd_agreement.py. Same data,
same band-integrated amplitudes; collapses the 10-column statistical
breakdown into a 4-column reader-facing summary that says one thing:
"the two parallel probes agree to within ~X % at each thesis frequency."

Columns: $f$ [Hz] · $N$ · $\\langle A \\rangle$ [mm] · $\\Delta$ (far−wall) [%]
where Δ is the mean across runs of (A_far − A_wall) / ½(A_far + A_wall),
expressed in percent and signed (positive ⇒ far reads higher than wall).

Outputs:
    output/TABLES/data/ch04_parallel_probe_psd_agreement_simple.csv       (render-shape)
    output/TABLES/data/ch04_parallel_probe_psd_agreement_simple.meta.json (provenance)

Caption text is owned by main_save_tables.py (TABLE_CAPTIONS).
"""

import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

BASE = (Path(__file__).resolve().parent.parent
        if "__file__" in globals() else Path.cwd())
sys.path.insert(0, str(BASE))
os.chdir(BASE)

# Reuse the data path + heavy lifting from the full-stats sibling so both
# tables are guaranteed to be reading the same runs and the same band
# integrals — only the rendering differs.
from analysis_scratch.parallel_probe_psd_agreement import (
    PROBES, TARGET_FREQS, BAND_HALFWIDTH_HZ, F_MAX_HZ, N_GRID,
    _load_psd_data_from_project, harmonize_grid, stack_runs,
    _band_amplitudes,
)


THESIS_NAME = "ch04_parallel_probe_psd_agreement_simple"
CHAPTER     = "04"
SCRIPT_REL  = "analysis_scratch/parallel_probe_psd_agreement_simple.py"

DATA_DIR    = BASE / "output" / "TABLES" / "data"
RENDER_CSV  = DATA_DIR / f"{THESIS_NAME}.csv"
META_JSON   = DATA_DIR / f"{THESIS_NAME}.meta.json"

# Variant B output (new 2026-05-09): ka-regime stratified pooled table.
REGIME_THESIS_NAME = "ch04_parallel_probe_psd_agreement_by_ka_regime"
REGIME_CSV         = DATA_DIR / f"{REGIME_THESIS_NAME}.csv"
REGIME_META_JSON   = DATA_DIR / f"{REGIME_THESIS_NAME}.meta.json"

# Threshold separating "wind-dominated" (low ka) from "wave-dominated"
# (high ka) regimes. 0.15 is the cut suggested by the (freq × amp) split:
# 1.3 Hz any amp + 1.4–1.5 Hz × A1 sit below; 1.5–1.6 Hz × A2/A3 sit above.
KA_REGIME_THRESHOLD = 0.15


def per_freq_simple(harmonized, f_grid, probes, target_freqs, halfwidth,
                    ka_per_run=None):
    """Per-frequency: N runs, mean Δ%, std Δ%.

    Per-run signed relative difference Δ_i = (A_far,i − A_wall,i) /
    [½(A_far,i + A_wall,i)] × 100, in percent. Reported as:
      - diff_pct  : mean across runs (probes' systematic bias)
      - std_pct   : std across runs   (run-to-run scatter; matches the
                                       per-run cloud in the Bland-Altman
                                       figure ch04_parallel_probe_agreement_bland_altman)

    The ⟨A⟩ column was dropped 2026-05-09 — at fixed f the per-run mix
    of voltage tiers (A1/A2/A3) is unbalanced, so ⟨A⟩ averages over a
    different blend at each frequency and isn't a clean physical scale.
    """
    stack_a = stack_runs(harmonized, probes[0])
    stack_b = stack_runs(harmonized, probes[1])
    rows = []
    for fh in target_freqs:
        a_a = _band_amplitudes(stack_a, f_grid, fh - halfwidth, fh + halfwidth)
        a_b = _band_amplitudes(stack_b, f_grid, fh - halfwidth, fh + halfwidth)
        finite = np.isfinite(a_a) & np.isfinite(a_b)
        n = int(finite.sum())
        if n == 0:
            rows.append(dict(freq=fh, n=0, diff_pct=np.nan, std_pct=np.nan,
                             ka_min=np.nan, ka_max=np.nan,
                             _per_run=np.array([], dtype=float),
                             _per_run_ka=np.array([], dtype=float)))
            continue
        a_a = a_a[finite]
        a_b = a_b[finite]
        # Per-run relative difference. Signed: positive ⇒ far reads higher.
        per_run = (a_b - a_a) / (0.5 * (a_a + a_b)) * 100.0
        # Per-run ka, aligned with the same `finite` mask. NaN if not provided.
        if ka_per_run is not None and len(ka_per_run) == len(stack_a):
            ka_finite = np.asarray(ka_per_run, dtype=float)[finite]
        else:
            ka_finite = np.full(n, np.nan)
        ka_valid = ka_finite[np.isfinite(ka_finite)]
        ka_min = float(ka_valid.min()) if len(ka_valid) else float("nan")
        ka_max = float(ka_valid.max()) if len(ka_valid) else float("nan")
        rows.append(dict(
            freq=fh, n=n,
            diff_pct=float(per_run.mean()),
            std_pct=float(per_run.std(ddof=1)) if n > 1 else float("nan"),
            ka_min=ka_min, ka_max=ka_max,
            # Raw per-run Δ + ka values, retained on the row dict for
            # downstream pooled-statistic computation (skewness, percentiles,
            # ka-regime stratification). The CSV writer strips keys starting
            # with `_` before saving — they're Python-level internal handoffs,
            # not render columns.
            _per_run=per_run.copy(),
            _per_run_ka=ka_finite.copy(),
        ))
    return rows


WIND_LABELS = ("nowind", "fullwind")
WIND_DISPLAY = {"nowind": "Uten vind", "fullwind": "Full vind"}


def _build_run_ka_map():
    """One-time meta load → run_id → IN ka (FFT) mapping for the canon
    March-2026 lowrange folders. Used by _rows_for_wind to thread ka per
    run through to per_freq_simple. Stand-alone import (no caching) —
    the project loader caches at the parquet level, so this is cheap.
    """
    from wavescripts.improved_data_loader import load_analysis_data
    canon_dirs = [
        BASE / "waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange",
        BASE / "waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange",
    ]
    all_meta, _, _, _ = load_analysis_data(*[str(d) for d in canon_dirs],
                                           load_processed=False)
    run_ka = {}
    for _, r in all_meta.iterrows():
        path = str(r["path"])
        # Same run_id convention as _load_psd_data_from_project (parent
        # folder + file stem) so the map joins correctly.
        run_id = f"{Path(path).parent.name}/{Path(path).stem}"
        run_ka[run_id] = float(r.get("IN ka (FFT)", np.nan))
    return run_ka


def _rows_for_wind(wind_label, run_ka_map=None):
    """Compute one row per target freq, restricted to runs paddled at that
    target freq AND under the requested wind condition.

    2026-05-09 update: previously the loader was called once per wind and
    the same set of runs (mixed paddle frequencies) was reused for every
    row. That measured probe agreement at the *PSD band level* — useful
    as a generic calibration check but mixed paddle-at-this-freq runs
    with paddle-at-other-freq runs (where the band amplitude is just
    noise floor / wind tail). Now the loader is called once per
    (wind, target_freq) pair via the `target_freqs=[fh]` filter, so each
    row's N counts only runs that were actually paddled at that freq.
    Result: cleaner Δ̄ / σ_Δ that answer "do the probes agree on the
    paddle wave?" rather than "do they agree on whatever PSD power is
    at this band?".
    """
    rows = []
    for fh in TARGET_FREQS:
        # _load_psd_data_from_project's `target_freqs` arg matches paddle
        # WaveFrequencyInput within ±0.005 Hz. Passing [fh] restricts to
        # runs paddled at this target freq only.
        psd_data = _load_psd_data_from_project(
            range_label="lowrange",
            wind_label=wind_label,
            target_freqs=[fh],
        )
        if not psd_data:
            rows.append(dict(freq=fh, n=0, diff_pct=np.nan, std_pct=np.nan,
                             wind=wind_label))
            continue
        f_grid, harmonized = harmonize_grid(psd_data, n_grid=N_GRID,
                                            f_max=F_MAX_HZ)
        # ka_per_run aligned with stack_runs() ordering = harmonized.keys().
        if run_ka_map is not None:
            ka_per_run = np.array([run_ka_map.get(rid, np.nan)
                                   for rid in harmonized.keys()])
        else:
            ka_per_run = None
        # per_freq_simple operates over a list; pass [fh] to compute just
        # this row.
        sub = per_freq_simple(harmonized, f_grid, PROBES, [fh],
                              BAND_HALFWIDTH_HZ, ka_per_run=ka_per_run)
        for r in sub:
            r["wind"] = wind_label
            rows.append(r)
    return rows


def main():
    """Build the 2-block (uten / full vind) version of the simple table.

    The pooled (single-block) version mixed both wind conditions per row,
    which masked the reader's primary question — "do the probes agree the
    same with and without wind?" 2026-05-09: split into wind-keyed rows so
    each block answers the question directly. Renderer (main_save_tables.py)
    uses row_groups to draw the two sub-sections.
    """
    print("Loading meta to build run_id → ka map ...")
    run_ka_map = _build_run_ka_map()
    print(f"  {len(run_ka_map)} runs in canon meta with ka.")

    print("Loading PSD data — split by wind condition ...")
    rows = []
    for wind_label in WIND_LABELS:
        wind_rows = _rows_for_wind(wind_label, run_ka_map=run_ka_map)
        rows.extend(wind_rows)
        print(f"  {wind_label}: {sum(r['n'] for r in wind_rows)} runs total")

    print()
    print(f"  {'wind':>10}  {'f [Hz]':>6}  {'N':>3}  "
          f"{'mean Δ [%]':>11}  {'std Δ [%]':>10}")
    for r in rows:
        print(f"  {r['wind']:>10}  {r['freq']:>6.2f}  {r['n']:>3d}  "
              f"{r['diff_pct']:>+11.2f}  {r['std_pct']:>10.2f}")
    print()

    # Pooled distribution-shape stats per wind condition (2026-05-09).
    # Pooled across (amp, freq) within each wind because amplitude is the
    # dimension probes disagree most on, so pooling across amplitudes is
    # exactly what we want to characterise. N becomes large enough for
    # reliable skewness (32 uten, 48 full).
    from scipy import stats as _stats

    def _pooled_shape(wind_label):
        diffs = np.concatenate([
            r["_per_run"] for r in rows
            if r["wind"] == wind_label and len(r["_per_run"]) > 0
        ]) if rows else np.array([])
        if len(diffs) < 3:
            return None
        return {
            "n":     int(len(diffs)),
            "mean":  float(diffs.mean()),
            "std":   float(diffs.std(ddof=1)),
            "skew":  float(_stats.skew(diffs, bias=False)),
            "p5":    float(np.percentile(diffs, 5)),
            "p95":   float(np.percentile(diffs, 95)),
            "min":   float(diffs.min()),
            "max":   float(diffs.max()),
        }

    pooled_no   = _pooled_shape("nowind")   or {}
    pooled_full = _pooled_shape("fullwind") or {}

    print(f"\nPooled distribution shape (across amp × freq, per wind):")
    for label, p in (("nowind", pooled_no), ("fullwind", pooled_full)):
        if not p:
            continue
        print(f"  {label}: N={p['n']:3d}  mean={p['mean']:+.2f}%  "
              f"std={p['std']:.2f}%  skew={p['skew']:+.3f}  "
              f"[P5,P95]=[{p['p5']:+.1f}, {p['p95']:+.1f}]")

    # Render-shape CSV — strip the internal `_per_run` arrays.
    csv_rows = [{k: v for k, v in r.items() if not k.startswith("_")}
                for r in rows]
    render_df = pd.DataFrame(csv_rows)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    render_df.to_csv(RENDER_CSV, index=False)
    print(f"\nrender CSV → {RENDER_CSV.relative_to(BASE)}")

    # ── Variant B: ka-regime stratified pooled table ──────────────────────
    # Pool runs across (amp × freq) per (regime × wind), where regime is
    # decided per-run by ka < KA_REGIME_THRESHOLD ("wind-dominated") or
    # ≥ threshold ("wave-dominated"). Each cell pools enough runs for
    # reliable σ / γ₁ / percentiles; the regime contrast (rather than the
    # freq contrast) carries the table's argument.
    def _pool_by_regime(diffs, kas):
        """Return wind-dominated / wave-dominated subsets of (Δ, ka) pairs."""
        kas = np.asarray(kas, dtype=float)
        diffs = np.asarray(diffs, dtype=float)
        valid = np.isfinite(kas) & np.isfinite(diffs)
        diffs = diffs[valid]; kas = kas[valid]
        wind_dom = diffs[kas <  KA_REGIME_THRESHOLD]
        wave_dom = diffs[kas >= KA_REGIME_THRESHOLD]
        return wind_dom, wave_dom

    def _stats_block(diffs):
        if len(diffs) < 3:
            return None
        return {
            "n":     int(len(diffs)),
            "mean":  float(diffs.mean()),
            "std":   float(diffs.std(ddof=1)),
            "skew":  float(_stats.skew(diffs, bias=False)),
            "p5":    float(np.percentile(diffs, 5)),
            "p95":   float(np.percentile(diffs, 95)),
        }

    regime_rows = []
    for wind_label in WIND_LABELS:
        wind_diffs = np.concatenate([
            r["_per_run"] for r in rows
            if r["wind"] == wind_label and len(r["_per_run"]) > 0
        ]) if rows else np.array([])
        wind_kas = np.concatenate([
            r["_per_run_ka"] for r in rows
            if r["wind"] == wind_label and len(r["_per_run_ka"]) > 0
        ]) if rows else np.array([])
        wind_dom, wave_dom = _pool_by_regime(wind_diffs, wind_kas)
        for regime_key, subset in (("wind_dominated", wind_dom),
                                   ("wave_dominated", wave_dom)):
            stats_block = _stats_block(subset)
            if stats_block is None:
                regime_rows.append(dict(
                    regime=regime_key, wind=wind_label,
                    n=int(len(subset)),
                    mean=np.nan, std=np.nan, skew=np.nan,
                    p5=np.nan, p95=np.nan,
                ))
            else:
                regime_rows.append(dict(
                    regime=regime_key, wind=wind_label, **stats_block,
                ))

    regime_df = pd.DataFrame(regime_rows)
    regime_df.to_csv(REGIME_CSV, index=False)
    print(f"regime CSV → {REGIME_CSV.relative_to(BASE)}")

    # Print regime summary to console.
    print(f"\nVariant B — pooled by ka regime "
          f"(threshold ka = {KA_REGIME_THRESHOLD}):")
    print(f"  {'regime':>16}  {'wind':>10}  {'N':>3}  {'mean':>7}  "
          f"{'std':>6}  {'skew':>6}  {'[P5,P95]':>16}")
    for r in regime_rows:
        if r['n'] < 3:
            print(f"  {r['regime']:>16}  {r['wind']:>10}  {r['n']:>3d}  "
                  f"(too few runs)")
            continue
        print(f"  {r['regime']:>16}  {r['wind']:>10}  {r['n']:>3d}  "
              f"{r['mean']:>+7.2f}  {r['std']:>6.2f}  "
              f"{r['skew']:>+6.2f}  [{r['p5']:>+5.1f}, {r['p95']:>+5.1f}]")

    # Provenance meta.json — per-wind summaries so the headline tells the
    # wind effect on probe agreement directly.
    freq_list = ", ".join(f"{r['freq']:.1f}" for r in rows
                          if r["wind"] == WIND_LABELS[0])

    def _wind_stats(wlabel):
        subset = [r for r in rows
                  if r["wind"] == wlabel and np.isfinite(r["diff_pct"])]
        if not subset:
            return None
        n_total = sum(r["n"] for r in subset)
        n_per_freq = subset[0]["n"]
        worst_mean = max(abs(r["diff_pct"]) for r in subset)
        sigma_lo = min(r["std_pct"] for r in subset
                       if np.isfinite(r["std_pct"]))
        sigma_hi = max(r["std_pct"] for r in subset
                       if np.isfinite(r["std_pct"]))
        return {
            "n_total": n_total, "n_per_freq": n_per_freq,
            "worst_mean": worst_mean,
            "sigma_lo": sigma_lo, "sigma_hi": sigma_hi,
        }

    no_stats   = _wind_stats("nowind")   or {}
    full_stats = _wind_stats("fullwind") or {}

    def _hl(stats, label):
        if not stats:
            return [f"{label:<11}: (no data)"]
        return [
            f"{label:<11}: |Δ̄|_max = {stats['worst_mean']:.2f} %, "
            f"σ_Δ = {stats['sigma_lo']:.1f}–{stats['sigma_hi']:.1f} %, "
            f"N = {stats['n_per_freq']}/freq",
        ]

    meta_payload = {
        "script":          SCRIPT_REL,
        "plot_type":       "parallel_probe_psd_agreement_simple_table",
        "chapter":         CHAPTER,
        "caption_label":   f"tab:{THESIS_NAME}",
        "caption_short":   "",
        "sections": [
            {
                "title": "Method",
                "lines": [
                    f"probes              : {PROBES[0]} (wall) vs {PROBES[1]} (far)",
                    f"band                : ±{BAND_HALFWIDTH_HZ:.2f} Hz around f, integrated PSD",
                    "Per-run Δ           : 100 · (A_far − A_wall) / [½ · (A_far + A_wall)]",
                    "                      Sign: + ⇒ far reads higher than wall.",
                    "Per-row run set     : only runs paddled at the row's target",
                    "                      frequency (WaveFrequencyInput == f ±0.005 Hz).",
                    "                      So each row's N counts only paddle-at-this-f",
                    "                      runs — the Δ̄ / σ_Δ values reflect probe",
                    "                      agreement on the actual paddle wave, not",
                    "                      on PSD band power across mixed paddle freqs.",
                    "Δ̄ (column 3)        : mean of Δ across the N runs (per cell).",
                    "                      Probes' systematic bias on the paddle wave.",
                    "σ_Δ (column 4)      : std of Δ across the N runs (per cell).",
                    "                      Run-to-run scatter (matches the",
                    "                      individual dots in the Bland-Altman fig).",
                    "Layout              : two row-blocks — `Uten vind` then",
                    "                      `Full vind`. Each block has 4 freq rows.",
                    "                      The split makes the wind effect on probe",
                    "                      agreement directly readable.",
                ],
            },
            {
                "title": "Inputs",
                "lines": [
                    f"N (uten/full)       : {no_stats.get('n_per_freq', 'n/a')} / "
                    f"{full_stats.get('n_per_freq', 'n/a')}  per freq",
                    "data scope          : panel-full, quality-ok, both probes present,",
                    "                      canon March-2026 lowrange folders.",
                    f"target frequencies  : {freq_list} Hz",
                ],
            },
            {
                "title": "Headline",
                "lines": [
                    *_hl(no_stats,   "Uten vind"),
                    *_hl(full_stats, "Full vind"),
                    "Reading             : (a) probes are well-calibrated under no-wind",
                    "                      (small Δ̄, small σ_Δ); wind inflates BOTH the",
                    "                      systematic bias and the run-to-run scatter.",
                    "                      (b) the wall- vs far-probe disagreement under",
                    "                      wind is what motivates using the MEAN of both",
                    "                      probes as the canonical IN amplitude.",
                    "companion figure    : fig:ch04_parallel_probe_agreement_bland_altman",
                    "                      (per-run scatter visualised).",
                    "companion table     : ch04_parallel_probe_psd_agreement_lowrange_merged.tex",
                    "                      (additional probe-agreement metrics).",
                ],
            },
            {
                # Distribution-shape diagnostics, pooled per wind condition
                # across (amp × freq). Pooling rationale: amplitude is the
                # dimension probes disagree most on, so pooling across amp
                # is exactly what we want to summarise. N=32/48 is large
                # enough for usable skewness estimates (per-cell N=6–17 is
                # too small for cell-level skewness).
                "title": "Pooled shape diagnostics (across amp × freq, per wind)",
                "lines": [
                    f"Uten vind  : N={pooled_no.get('n', 'n/a'):>3}  "
                    f"mean={pooled_no.get('mean', float('nan')):+.2f}%  "
                    f"std={pooled_no.get('std', float('nan')):.2f}%  "
                    f"skew={pooled_no.get('skew', float('nan')):+.3f}  "
                    f"[P5,P95]=[{pooled_no.get('p5', float('nan')):+.1f}, "
                    f"{pooled_no.get('p95', float('nan')):+.1f}]%",
                    f"Full vind  : N={pooled_full.get('n', 'n/a'):>3}  "
                    f"mean={pooled_full.get('mean', float('nan')):+.2f}%  "
                    f"std={pooled_full.get('std', float('nan')):.2f}%  "
                    f"skew={pooled_full.get('skew', float('nan')):+.3f}  "
                    f"[P5,P95]=[{pooled_full.get('p5', float('nan')):+.1f}, "
                    f"{pooled_full.get('p95', float('nan')):+.1f}]%",
                    "Reading skew         : |γ₁| < 0.5 ≈ symmetric (σ_Δ ± reads",
                    "                       both ways); 0.5–1 = moderately skewed;",
                    "                       > 1 = heavily skewed (σ overstates one",
                    "                       side, understates the other).",
                    "Reading [P5, P95]    : empirical 90 % interval — no Gaussian",
                    "                       assumption. Compare width to ±2σ to",
                    "                       gauge tail behaviour.",
                ],
            },
            {
                # Caption text is user-owned (lives in TABLE_CAPTIONS in
                # main_save_tables.py). Suggestions are flagged here in the
                # stub for the user to copy + adapt — this section is the
                # ONLY place this script proposes caption wording.
                "title": "Caption suggestion (for reference)",
                "lines": [
                    "If you want both math symbols rendered, ensure $...$ wraps:",
                    r"  $\bar{\Delta}$  and  $\sigma_{\Delta}$  (math-mode).",
                    "",
                    "One-sentence form:",
                    r"  Samsvar mellom parallelle prober ved padlefrekvens.",
                    r"  $\bar{\Delta}$ er probenes systematiske amplitudeforskjell;",
                    r"  $\sigma_{\Delta}$ er løp-til-løp-spredning.",
                    "",
                    "Two-sentence form (adds the wind-effect punchline):",
                    r"  Samsvar mellom parallelle prober ved padlefrekvens.",
                    r"  Probene er essensielt identiske uten vind",
                    r"  ($|\bar{\Delta}| < 0{,}3\,\%$, $\sigma_{\Delta} \approx 2\,\%$);",
                    r"  vinden firedobler spredningen og introduserer et",
                    r"  systematisk avvik på opp mot $3\,\%$.",
                ],
            },
        ],
    }
    META_JSON.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
    print(f"meta JSON  → {META_JSON.relative_to(BASE)}")

    # ── Variant B regime-table meta.json ──────────────────────────────────
    REGIME_LABEL = {
        "wind_dominated": "Vind-dominert (ka < %.2f)" % KA_REGIME_THRESHOLD,
        "wave_dominated": "Bølge-dominert (ka ≥ %.2f)" % KA_REGIME_THRESHOLD,
    }
    regime_payload = {
        "script":          SCRIPT_REL,
        "plot_type":       "parallel_probe_psd_agreement_by_ka_regime_table",
        "chapter":         CHAPTER,
        "caption_label":   f"tab:{REGIME_THESIS_NAME}",
        "caption_short":   "",
        "sections": [
            {
                "title": "Method",
                "lines": [
                    f"Threshold           : ka = {KA_REGIME_THRESHOLD}",
                    "Wind-dominated      : ka < threshold (low amp and/or low freq —",
                    "                      the wave is small enough that wind chop",
                    "                      and lateral asymmetry dominate Δ).",
                    "Wave-dominated      : ka ≥ threshold (steeper waves; under wind",
                    "                      the far probe (9373/340) struggles to",
                    "                      track the rapid up-stroke after the",
                    "                      trough, producing systematic negative Δ).",
                    "Pooling             : (regime × wind), across (amp × freq).",
                    "                      Pooling across amp is intentional —",
                    "                      amplitude is the dimension probes",
                    "                      disagree most on, so summarising it out",
                    "                      is exactly the contrast we want.",
                    "Δ̄ / σ_Δ / γ_1       : mean / std / Fisher-Pearson skew of Δ%.",
                    "[P5, P95]           : empirical 90 % interval (no Gaussian).",
                ],
            },
            {
                "title": "Reading",
                "lines": [
                    "(a) Compare wind-dominated vs wave-dominated within each wind",
                    "    block: σ should grow markedly under the wave-dominated",
                    "    regime (probe-struggle dominates) compared to the wind-",
                    "    dominated regime (wind-chop dominates).",
                    "(b) Skewness γ_1 should be ≈ 0 in symmetric regimes (uten",
                    "    vind, both regimes; full vind, low ka) but turn",
                    "    negative under (full vind × wave-dominated) where the",
                    "    far probe systematically clips peaks.",
                    "(c) [P5, P95] reads as the empirical 90 % range of Δ —",
                    "    asymmetric width = asymmetric distribution.",
                ],
            },
            {
                "title": "Companion table + figure",
                "lines": [
                    "tab:ch04_parallel_probe_psd_agreement_simple — same data",
                    "    stratified by (wind × freq) instead of (wind × regime).",
                    "fig:ch04_parallel_probe_agreement_bland_altman — per-run",
                    "    scatter visualised.",
                ],
            },
        ],
    }
    REGIME_META_JSON.write_text(json.dumps(regime_payload, indent=2),
                                 encoding="utf-8")
    print(f"regime meta → {REGIME_META_JSON.relative_to(BASE)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
