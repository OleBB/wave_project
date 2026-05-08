"""
Parallel-probe PSD agreement — SIMPLE table.

Sibling of analysis_scratch/parallel_probe_psd_agreement.py. Same data,
same band-integrated amplitudes; collapses the 10-column statistical
breakdown into a 4-column reader-facing summary that says one thing:
"the two parallel probes agree to within ~X % at each thesis frequency."

Columns: $f$ [Hz] · $N$ · $\\langle A \\rangle$ [mm] · $\\Delta$ (far−wall) [%]
where Δ is the mean across runs of (A_far − A_wall) / ½(A_far + A_wall),
expressed in percent and signed (positive ⇒ far reads higher than wall).

Output:
    output/TABLES/ch04_parallel_probe_psd_agreement_simple.tex
"""

import sys
from pathlib import Path
from datetime import datetime as _dt

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

# Reuse the data path + heavy lifting from the full-stats sibling so both
# tables are guaranteed to be reading the same runs and the same band
# integrals — only the rendering differs.
from analysis_scratch.parallel_probe_psd_agreement import (
    PROBES, TARGET_FREQS, BAND_HALFWIDTH_HZ, F_MAX_HZ, N_GRID,
    _load_psd_data_from_project, harmonize_grid, stack_runs,
    _band_amplitudes,
)
from wavescripts.plot_utils import _lookup_central_caption


THESIS_TABLE_NAME = "ch04_parallel_probe_psd_agreement_simple"
BASE = Path(__file__).resolve().parent.parent
OUT_TEX = BASE / "output" / "TABLES" / f"{THESIS_TABLE_NAME}.tex"


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
            rows.append(dict(freq=fh, n=0, mean_amp=np.nan,
                             diff_pct=np.nan))
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


def write_simple_tex_table(rows, out_path):
    # Caption is read from output/.table_captions.json (written by main_save_tables.py).
    # Run main_save_tables.py once before this script to populate the cache.
    captions_json = BASE / "output" / ".table_captions.json"
    caption_full  = _lookup_central_caption(THESIS_TABLE_NAME, kind="full",  json_path=captions_json)
    caption_short = _lookup_central_caption(THESIS_TABLE_NAME, kind="short", json_path=captions_json)
    if caption_full and caption_short:
        caption_block = (f"  \\caption[{caption_short}]{{\n"
                         f"    {caption_full}\n  }}\n")
    elif caption_full:
        caption_block = f"  \\caption{{\n    {caption_full}\n  }}\n"
    else:
        caption_block = ("  \\caption{\n"
                         "    % TODO: write caption "
                         "(edit FIGURE_CAPTIONS in main_save_figures.py)\n"
                         "  }\n")

    n_runs = rows[0]["n"] if rows else 0
    freq_list = ", ".join(f"{r['freq']:.1f}" for r in rows)

    # Worst (largest |Δ|) — surfaced into the immutable block so the
    # reader's "they agree to within X %" claim is grounded in a number.
    finite_rows = [r for r in rows if np.isfinite(r["diff_pct"])]
    worst = max((abs(r["diff_pct"]) for r in finite_rows), default=float("nan"))

    immutable = "\n".join([
        "%! TEX root = ../main.tex",
        "% ==============================================================",
        "% IMMUTABLE — generated automatically, do not edit this block",
        "%",
        "% — Provenance ───────────────────────────────────────────────────",
        "%   script            : analysis_scratch/parallel_probe_psd_agreement_simple.py",
        "%   plot_type         : parallel_probe_psd_agreement_simple_table",
        "%   chapter           : 04",
        f"%   generated_at      : {_dt.now().isoformat(timespec='seconds')}",
        f"%   caption_label     : tab:{THESIS_TABLE_NAME}",
        f"%   caption_short     : {caption_short or ''}",
        "%",
        "% — Method ────────────────────────────────────────────────────",
        f"%   probes            : {PROBES[0]} (wall) vs {PROBES[1]} (far)",
        f"%   band              : ±{BAND_HALFWIDTH_HZ:.2f} Hz around f, integrated PSD",
        "%   <A>               : 0.5·(mean A_wall + mean A_far) across runs",
        "%   Δ (far−wall) [%]  : mean across runs of",
        "%                       100 · (A_far − A_wall) / [½ · (A_far + A_wall)]",
        "%                       Sign: + ⇒ far reads higher than wall.",
        "%",
        "% — Inputs ────────────────────────────────────────────────────",
        f"%   N runs            : {n_runs}",
        "%   data scope        : panel-full, quality-ok, both probes present,",
        "%                       canon March-2026 lowrange folders.",
        f"%   target frequencies: {freq_list} Hz",
        "%",
        "% — Headline ──────────────────────────────────────────────────",
        f"%   worst |Δ|         : {worst:.2f} %  (across all rows)",
        "%   companion table   : ch04_parallel_probe_psd_agreement.tex",
        "%                       (full 10-column statistical breakdown)",
        "%",
        "% ── end immutable block ─────────────────────────────────────────",
    ])

    body_lines = []
    for r in rows:
        f_cell  = f"\\num{{{r['freq']:.2f}}}"
        n_cell  = f"\\num{{{r['n']}}}"
        a_cell  = (f"\\num{{{r['mean_amp']:.2f}}}"
                   if np.isfinite(r["mean_amp"]) else "n/a")
        d_cell  = (f"\\num{{{r['diff_pct']:+.2f}}}"
                   if np.isfinite(r["diff_pct"]) else "n/a")
        body_lines.append(f"    {f_cell} & {n_cell} & {a_cell} & {d_cell} \\\\")

    table_body = (
        "\\begin{table}[hbt]\n"
        "  \\centering\n"
        "  \\small\n"
        + caption_block
        + f"  \\label{{tab:{THESIS_TABLE_NAME}}}\n"
        "  \\begin{tabular}{cccc}\n"
        "    \\toprule\n"
        "    $f$ [\\unit{\\hertz}] &\n"
        "      $N$ &\n"
        "      $\\langle A \\rangle$ [\\unit{\\milli\\metre}] &\n"
        "      $\\Delta$ (far$-$wall) [\\%] \\\\\n"
        "    \\midrule\n"
        + "\n".join(body_lines) + "\n"
        "    \\bottomrule\n"
        "  \\end{tabular}\n"
        "\\end{table}\n"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(immutable + "\n" + table_body)
    print(f"Saved -> {out_path}")


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

    write_simple_tex_table(rows, OUT_TEX)
    print("\nDone.")


if __name__ == "__main__":
    main()
