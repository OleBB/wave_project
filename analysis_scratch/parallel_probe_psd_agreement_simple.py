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


def per_freq_simple(harmonized, f_grid, probes, target_freqs, halfwidth):
    """Per-frequency: N runs, mean amplitude, mean signed disagreement %."""
    stack_a = stack_runs(harmonized, probes[0])
    stack_b = stack_runs(harmonized, probes[1])
    rows = []
    for fh in target_freqs:
        a_a = _band_amplitudes(stack_a, f_grid, fh - halfwidth, fh + halfwidth)
        a_b = _band_amplitudes(stack_b, f_grid, fh - halfwidth, fh + halfwidth)
        finite = np.isfinite(a_a) & np.isfinite(a_b)
        n = int(finite.sum())
        if n == 0:
            rows.append(dict(freq=fh, n=0, mean_amp=np.nan, diff_pct=np.nan))
            continue
        a_a = a_a[finite]
        a_b = a_b[finite]
        mean_amp = float(0.5 * (a_a.mean() + a_b.mean()))
        # Per-run relative difference, then mean across runs.
        # Signed: positive ⇒ far probe (b) reads higher than wall (a).
        per_run = (a_b - a_a) / (0.5 * (a_a + a_b)) * 100.0
        rows.append(dict(freq=fh, n=n,
                         mean_amp=mean_amp,
                         diff_pct=float(per_run.mean())))
    return rows


def main():
    print("Loading PSD data ...")
    psd_data = _load_psd_data_from_project()
    f_grid, harmonized = harmonize_grid(psd_data, n_grid=N_GRID, f_max=F_MAX_HZ)
    rows = per_freq_simple(harmonized, f_grid, PROBES, TARGET_FREQS,
                           BAND_HALFWIDTH_HZ)

    print()
    print(f"  {'f [Hz]':>6}  {'N':>3}  {'<A> [mm]':>9}  {'Δ (far-wall) [%]':>17}")
    for r in rows:
        print(f"  {r['freq']:>6.2f}  {r['n']:>3d}  {r['mean_amp']:>9.3f}  "
              f"{r['diff_pct']:>+17.2f}")
    print()

    # Render-shape CSV.
    render_df = pd.DataFrame(rows)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    render_df.to_csv(RENDER_CSV, index=False)
    print(f"render CSV → {RENDER_CSV.relative_to(BASE)}")

    # Provenance meta.json.
    n_runs = int(rows[0]["n"]) if rows else 0
    freq_list = ", ".join(f"{r['freq']:.1f}" for r in rows)
    finite_rows = [r for r in rows if np.isfinite(r["diff_pct"])]
    worst = max((abs(r["diff_pct"]) for r in finite_rows), default=float("nan"))

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
                    f"probes            : {PROBES[0]} (wall) vs {PROBES[1]} (far)",
                    f"band              : ±{BAND_HALFWIDTH_HZ:.2f} Hz around f, integrated PSD",
                    "<A>               : 0.5·(mean A_wall + mean A_far) across runs",
                    "Δ (far−wall) [%]  : mean across runs of",
                    "                    100 · (A_far − A_wall) / [½ · (A_far + A_wall)]",
                    "                    Sign: + ⇒ far reads higher than wall.",
                ],
            },
            {
                "title": "Inputs",
                "lines": [
                    f"N runs            : {n_runs}",
                    "data scope        : panel-full, quality-ok, both probes present,",
                    "                    canon March-2026 lowrange folders.",
                    f"target frequencies: {freq_list} Hz",
                ],
            },
            {
                "title": "Headline",
                "lines": [
                    f"worst |Δ|         : {worst:.2f} %  (across all rows)",
                    "companion table   : ch04_parallel_probe_psd_agreement.tex",
                    "                    (full 10-column statistical breakdown)",
                ],
            },
        ],
    }
    META_JSON.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
    print(f"meta JSON  → {META_JSON.relative_to(BASE)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
