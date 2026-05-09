"""
Parallel-probe PSD agreement analysis (CH04 §3 supporting analysis)
====================================================================

Quantifies how the two parallel probes 9373/170 (wall-side) and 9373/340
(far-side) agree on the wind-wave PSD across a set of runs, and what is
gained by averaging them.

NB on labels
------------
The user spec named the two probes "IN" / "OUT". In this project both
9373/170 and 9373/340 sit at the SAME longitudinal distance (9373 mm
from the wavemaker = the IN side); they differ only in lateral position.
The actual OUT probe is 12400/250. To avoid confusion downstream, this
script uses "wall" and "far" throughout. Edit `PROBES` to swap.

Assumed data structure (after the adapter runs)
-----------------------------------------------
    psd_data : dict[str, dict[str, dict[str, np.ndarray]]]
        psd_data[run_id][probe] = {"f": f_array, "Pxx": pxx_array}
    All runs share the same `f_array` (linear-spaced, enforced by the
    `harmonize_grid` adapter; units: Hz).  Pxx in mm²/Hz.

The analysis functions below take that simple dict and use only NumPy,
SciPy, and Matplotlib. Only `_load_psd_data_from_project` is project-
specific (it converts the project's `{csv_path: DataFrame}` cache into
the simple format above). Replace it with your own loader to use this
script outside this project.

Outputs
-------
- analysis_scratch/parallel_probe_psd_agreement.pdf
- A printed summary table to stdout
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# === Configuration ==========================================================

PROBES = ("9373/170", "9373/340")  # wall-side, far-side (parallel IN probes)
PROBE_LABELS = {"9373/170": "wall (170)", "9373/340": "far (340)"}

# Paddle frequencies (Hz) and a band half-width for narrow-band integration.
# Set to a single fundamental + harmonics if desired; the default lists the
# four thesis frequencies so the harmonic-test table covers all of them.
TARGET_FREQS = (1.3, 1.4, 1.5, 1.6)
BAND_HALFWIDTH_HZ = 0.10        # ±0.1 Hz around each f to integrate PSD

F_MAX_HZ = 10.0                  # crop x-axis (paddle peak + wind tail)
N_GRID = 2048                    # resolution of the harmonized common grid


# === Analysis core (pure NumPy/SciPy) =======================================

def harmonize_grid(psd_data, n_grid=N_GRID, f_max=F_MAX_HZ):
    """Resample each (run, probe) PSD onto a common linear-f grid via interp."""
    f_target = np.linspace(0.0, f_max, n_grid)
    out = {}
    for run_id, run_dict in psd_data.items():
        out[run_id] = {}
        for probe, d in run_dict.items():
            f, pxx = np.asarray(d["f"]), np.asarray(d["Pxx"])
            valid = np.isfinite(pxx) & np.isfinite(f)
            if valid.sum() < 2:
                out[run_id][probe] = np.full_like(f_target, np.nan)
                continue
            out[run_id][probe] = np.interp(
                f_target, f[valid], pxx[valid], left=np.nan, right=np.nan
            )
    return f_target, out


def stack_runs(harmonized, probe):
    """Stack one probe's PSD across runs → shape (n_runs, n_freqs)."""
    rows = [harmonized[r][probe] for r in harmonized if probe in harmonized[r]]
    if not rows:
        raise ValueError(f"No runs contain probe {probe!r}.")
    return np.vstack(rows)


def run_averaged(harmonized, probes):
    """Mean / std / quantile band per probe, averaged in linear power units."""
    summary = {}
    for p in probes:
        st = stack_runs(harmonized, p)
        summary[p] = dict(
            stack=st,
            mean=np.nanmean(st, axis=0),
            std=np.nanstd(st, axis=0, ddof=1),
            p05=np.nanquantile(st, 0.05, axis=0),
            p95=np.nanquantile(st, 0.95, axis=0),
            n=np.sum(np.isfinite(st), axis=0),
        )
    return summary


def safe_db(p):
    """10·log10 with a NaN floor for non-positive values (avoids -inf)."""
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    pos = p > 0
    out[pos] = 10.0 * np.log10(p[pos])
    return out


def per_freq_paired_diff(stack_a, stack_b):
    """
    Per-bin paired difference in dB across runs: d_r(f) = 10log P_b - 10log P_a.
    Returns (diff_db, mean, std, p_value, n_per_bin).
    p_value is from a per-bin paired t-test (H0: mean = 0 dB).
    """
    diff_db = safe_db(stack_b) - safe_db(stack_a)            # (n_runs, n_freqs)
    mean = np.nanmean(diff_db, axis=0)
    std = np.nanstd(diff_db, axis=0, ddof=1)
    n = np.sum(np.isfinite(diff_db), axis=0)
    se = std / np.sqrt(np.where(n > 0, n, 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, mean / se, np.nan)
    pval = np.where(n > 1, 2 * stats.t.sf(np.abs(t), df=np.maximum(n - 1, 1)),
                    np.nan)
    return diff_db, mean, std, pval, n


def _band_amplitudes(stack, f, lo, hi):
    """Per-run band-integrated amplitude: A = sqrt(2 · ∫ S(f) df) over [lo, hi]."""
    m = (f >= lo) & (f <= hi)
    if m.sum() < 2:
        return np.full(stack.shape[0], np.nan)
    int_p = np.trapezoid(stack[:, m], f[m], axis=1)
    return np.sqrt(2.0 * np.maximum(int_p, 0.0))


def harmonic_summary(harmonized, f, probes, target_freqs, halfwidth):
    """
    For each target frequency: per-run dB difference at the bin, paired t-test,
    cross-run correlation of band-integrated amplitudes, and variance of
    the simple mean compared to each single-probe variance.

    Also returns the mean of per-run linear power ratios `P_b[bin] / P_a[bin]`
    (for the reader-friendly Option-D table) and the per-run amplitudes
    (for downstream non-parametric spread reporting if desired).
    """
    stack_a = stack_runs(harmonized, probes[0])
    stack_b = stack_runs(harmonized, probes[1])
    rows = []
    for fh in target_freqs:
        # Nearest bin for the per-bin dB test
        bin_idx = int(np.argmin(np.abs(f - fh)))
        pa = stack_a[:, bin_idx]
        pb = stack_b[:, bin_idx]
        d_db = safe_db(pb) - safe_db(pa)
        d_db = d_db[np.isfinite(d_db)]
        if len(d_db) >= 2:
            tt = stats.ttest_rel(safe_db(pb), safe_db(pa), nan_policy="omit")
            mean_db, std_db, p_val = float(np.mean(d_db)), float(np.std(d_db, ddof=1)), float(tt.pvalue)
        else:
            mean_db, std_db, p_val = np.nan, np.nan, np.nan

        # Geometric mean of per-run power ratios = 10^(mean_dB / 10).
        # Geometric is the natural mean for ratios (and the natural pair to
        # the dB math); arithmetic mean blows up when one run has a small
        # P_wall, which we observed at 1.6 Hz nowind (1 run inflated the
        # arithmetic mean to 1.50 while the median stayed near 1.00).
        ratio_mean = float(10 ** (mean_db / 10.0)) if np.isfinite(mean_db) else np.nan

        # Band-integrated amplitudes for variance / correlation
        a_a = _band_amplitudes(stack_a, f, fh - halfwidth, fh + halfwidth)
        a_b = _band_amplitudes(stack_b, f, fh - halfwidth, fh + halfwidth)
        finite = np.isfinite(a_a) & np.isfinite(a_b)
        n_band = int(finite.sum())
        if n_band >= 3:
            r = float(np.corrcoef(a_a[finite], a_b[finite])[0, 1])
        else:
            r = np.nan
        v_a = float(np.var(a_a[finite], ddof=1)) if n_band > 1 else np.nan
        v_b = float(np.var(a_b[finite], ddof=1)) if n_band > 1 else np.nan
        cov = r * np.sqrt(v_a * v_b) if (n_band > 1 and np.isfinite(r)) else 0.0
        v_mean = 0.25 * (v_a + v_b + 2 * cov) if n_band > 1 else np.nan
        red_vs_min = (1 - v_mean / min(v_a, v_b)) * 100 if (
            n_band > 1 and min(v_a, v_b) > 0) else np.nan

        rows.append(dict(
            freq=fh, n=int(len(d_db)),
            mean_db=mean_db, std_db=std_db, p_value=p_val,
            ratio_mean=ratio_mean,
            n_band=n_band, corr=r,
            std_a=np.sqrt(v_a) if np.isfinite(v_a) else np.nan,
            std_b=np.sqrt(v_b) if np.isfinite(v_b) else np.nan,
            std_mean=np.sqrt(v_mean) if np.isfinite(v_mean) else np.nan,
            reduction_pct=red_vs_min,
        ))
    return rows


# === Plotting (pure matplotlib) =============================================

def plot_three_panel(f, summary, diff_mean, diff_std, target_freqs, out_path,
                     probes=PROBES):
    """A) overlay of run-averaged PSDs with p05–p95 band; B) difference
    spectrum (dB) ± 1σ across runs; C) zoom around paddle harmonics."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    colors = {probes[0]: "tab:blue", probes[1]: "tab:orange"}

    # Panel A — overlay of run-averaged PSDs
    ax = axes[0]
    for p in probes:
        s = summary[p]
        ax.plot(f, s["mean"], color=colors[p], lw=1.3, label=PROBE_LABELS[p])
        ax.fill_between(f, s["p05"], s["p95"], color=colors[p], alpha=0.18)
    for fh in target_freqs:
        ax.axvline(fh, color="grey", ls=":", lw=0.5, alpha=0.4)
    ax.set_yscale("log"); ax.set_xlim(0, F_MAX_HZ)
    ax.set_xlabel(r"$f$ [Hz]")
    ax.set_ylabel(r"PSD $S_{\eta\eta}$ [mm$^{2}$/Hz]")
    ax.set_title("(A) Run-averaged PSD ± p05–p95 band")
    ax.grid(True, which="both", alpha=0.3); ax.legend(loc="upper right")

    # Panel B — difference spectrum in dB
    ax = axes[1]
    ax.plot(f, diff_mean, color="black", lw=1.2, label="mean across runs")
    ax.fill_between(f, diff_mean - diff_std, diff_mean + diff_std,
                    color="grey", alpha=0.3, label=r"$\pm 1\sigma$")
    ax.axhline(0, color="red", ls="--", lw=0.8, alpha=0.7)
    for fh in target_freqs:
        ax.axvline(fh, color="grey", ls=":", lw=0.5, alpha=0.4)
    ax.set_xlim(0, F_MAX_HZ)
    ax.set_xlabel(r"$f$ [Hz]")
    ax.set_ylabel(r"$10\log_{10}(P_\mathrm{far}/P_\mathrm{wall})$ [dB]")
    ax.set_title("(B) Difference spectrum (paired across runs)")
    ax.grid(True, alpha=0.3); ax.legend(loc="upper right", fontsize=8)

    # Panel C — zoom around paddle harmonics
    ax = axes[2]
    f_lo = max(0.0, min(target_freqs) - 0.4)
    f_hi = max(target_freqs) + 0.4
    for p in probes:
        s = summary[p]
        m = (f >= f_lo) & (f <= f_hi)
        ax.plot(f[m], s["mean"][m], color=colors[p], lw=1.5,
                label=PROBE_LABELS[p])
        ax.fill_between(f[m], s["p05"][m], s["p95"][m],
                        color=colors[p], alpha=0.18)
    for fh in target_freqs:
        ax.axvline(fh, color="grey", ls=":", lw=0.6, alpha=0.5)
    ax.set_yscale("log"); ax.set_xlim(f_lo, f_hi)
    ax.set_xlabel(r"$f$ [Hz]")
    ax.set_title(f"(C) Zoom: {f_lo:.1f}–{f_hi:.1f} Hz (paddle band)")
    ax.grid(True, which="both", alpha=0.3); ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved -> {out_path}")
    plt.close(fig)


# === Thesis-table writer (LaTeX tabular -> output/TABLES/) ==================

THESIS_TABLE_BASE = "ch04_parallel_probe_psd_agreement"

# Range × wind cells produce one table each. Keys map to file-name suffixes
# (and FIGURE_CAPTIONS keys in main_save_figures.py).
RANGE_LABELS = {"lowrange": "low", "highrange": "high"}
WIND_LABELS = {"nowind": "no", "fullwind": "full"}


def _table_name(range_label, wind_label):
    """e.g. ('lowrange', 'nowind') -> 'ch04_parallel_probe_psd_agreement_lowrange_nowind'."""
    return f"{THESIS_TABLE_BASE}_{range_label}_{wind_label}"


def _format_body_lines(rows: list[dict]) -> list[str]:
    """Render harmonic-summary rows into LaTeX body lines (one per freq).

    Shared between the per-(range × wind) table and the merged-winds table
    so the two presentations are guaranteed identical at the cell level.
    """
    lines = []
    for r in rows:
        # "Beste probe" = single probe with smallest std(A). Mean is excluded
        # from the comparison (averaging is judged separately by Variansøkning).
        if np.isfinite(r["std_a"]) and np.isfinite(r["std_b"]):
            best = "nær" if r["std_a"] < r["std_b"] else "fjern"
        else:
            best = "n/a"
        f_cell = f"\\num{{{r['freq']:.2f}}}"
        ratio_cell = (f"\\num{{{r['ratio_mean']:.2f}}}"
                      if np.isfinite(r["ratio_mean"]) else "n/a")
        rho_cell = (f"\\num{{{r['corr']:+.3f}}}"
                    if np.isfinite(r["corr"]) else "n/a")
        best_cell = best
        # Sign: positive % = averaging WORSENS precision vs the better single
        # probe. `reduction_pct` is the fractional REDUCTION (negative when
        # worse), so negate.
        v_cell = (f"\\num{{{-r['reduction_pct']:+.1f}}}"
                  if np.isfinite(r["reduction_pct"]) else "n/a")
        lines.append(
            f"    {f_cell} & {ratio_cell} & {rho_cell} & {best_cell} & {v_cell} \\\\"
        )
    return lines


def write_merged_winds_tex_table(
    rows_by_wind: dict[str, list[dict]],
    out_path: Path,
    *,
    range_label: str,
    scope_notes: dict,
) -> None:
    """Render two wind conditions in ONE .tex with sub-section headers.

    Style modelled on tab:ch04_wind_setup_baseline_table — one combined
    \\caption + \\label, body uses \\multicolumn{5}{l}{\\itshape <wind>} per
    sub-section. Cell formatting is identical to the per-cell tables
    (shared via _format_body_lines), so the merged table reads as the
    union of the per-cell tables, not a different analysis.

    Wind ordering: nowind (baseline) above, fullwind (perturbation) below.
    """
    from datetime import datetime as _dt
    from wavescripts.plot_utils import _lookup_central_caption

    table_name = f"{THESIS_TABLE_BASE}_{range_label}_merged"
    # Caption lives in TABLE_CAPTIONS (main_save_tables.py) → persisted at
    # output/.table_captions.json. The default _lookup_central_caption
    # path is .figure_captions.json, so pass json_path explicitly.
    base_dir = Path(__file__).resolve().parent.parent
    _captions_json = base_dir / "output" / ".table_captions.json"
    caption_full  = _lookup_central_caption(table_name, kind="full",
                                            json_path=_captions_json)
    caption_short = _lookup_central_caption(table_name, kind="short",
                                            json_path=_captions_json)
    # Wrap the caption in CAPTION-SYNC sentinels so the caption can be
    # rewritten in-place later via analysis_scratch/sync_captions.py.
    if caption_full and caption_short:
        _cap_inner = (f"  \\caption[{caption_short}]{{\n"
                      f"    {caption_full}\n  }}")
    elif caption_full:
        _cap_inner = f"  \\caption{{\n    {caption_full}\n  }}"
    else:
        _cap_inner = ("  \\caption{\n"
                      "    % TODO: write caption "
                      "(edit TABLE_CAPTIONS in main_save_tables.py)\n"
                      "  }")
    caption_block = (
        "% >>> CAPTION-SYNC START "
        "(do not edit this block; sync_captions.py overwrites)\n"
        f"{_cap_inner}\n"
        "% <<< CAPTION-SYNC END\n"
    )

    WIND_HEADER = {"nowind": "Uten vind", "fullwind": "Full vind"}

    # Per-wind n-runs lines for the immutable provenance block.
    n_lines = []
    for wkey in ("nowind", "fullwind"):
        rs = rows_by_wind.get(wkey) or []
        if rs:
            n_str = ", ".join(f"{r['freq']:.1f}Hz:n={r['n']}" for r in rs)
            n_lines.append(f"%   N runs ({wkey:>8}): {n_str}")

    immutable = "\n".join([
        "%! TEX root = ../main.tex",
        "% ==============================================================",
        "% IMMUTABLE — generated automatically, do not edit this block",
        "%",
        "% — Provenance ───────────────────────────────────────────────────",
        "%   script            : analysis_scratch/parallel_probe_psd_agreement.py",
        "%   plot_type         : parallel_probe_psd_agreement_table_merged",
        "%   chapter           : 04",
        f"%   generated_at      : {_dt.now().isoformat(timespec='seconds')}",
        f"%   caption_label     : tab:{table_name}",
        f"%   caption_short     : {caption_short or ''}",
        "%",
        "% — Method ────────────────────────────────────────────────────",
        "%   Same per-row analysis as the per-(range × wind) tables; cells",
        "%   formatted via the shared _format_body_lines helper. Two wind",
        "%   conditions are stacked in a single \\begin{table}, separated",
        "%   by italic sub-section headers (\\multicolumn{5}{l}{\\itshape …}),",
        "%   following the tab:ch04_wind_setup_baseline_table convention.",
        "%   Pairwise comparison of 9373/170 (wall) and 9373/340 (far) at",
        "%   each thesis paddle frequency (1.3, 1.4, 1.5, 1.6 Hz).",
        "%   P_fjern/P_nær : geometric mean across runs of P_far[bin]/P_wall[bin].",
        "%   Pearsons ρ    : Pearson correlation across runs of band-integrated A.",
        "%   Beste probe   : single probe with smallest std(A).",
        "%   Variansøkning [%] : % change in Var(½(A_wall+A_far)) vs the better single.",
        "%",
        "% — Inputs ────────────────────────────────────────────────────",
        *n_lines,
        f"%   probe-range setup : {range_label}",
        f"%   data scope (uten) : {scope_notes.get((range_label, 'nowind'), '')}",
        f"%   data scope (full) : {scope_notes.get((range_label, 'fullwind'), '')}",
        "%",
        "% ── end immutable block ─────────────────────────────────────────",
    ])

    # Body: nowind block, midrule, fullwind block. Sub-section headers
    # are \multicolumn{5}{l}{\itshape ...} rows, not text outside tabular.
    body_chunks = []
    first = True
    for wkey in ("nowind", "fullwind"):
        rs = rows_by_wind.get(wkey) or []
        if not rs:
            continue
        if not first:
            body_chunks.append("    \\midrule")
        body_chunks.append(
            f"    \\multicolumn{{5}}{{l}}{{\\itshape {WIND_HEADER[wkey]}}} \\\\"
        )
        body_chunks.extend(_format_body_lines(rs))
        first = False
    body_str = "\n".join(body_chunks)

    table_body = (
        "\\begin{table}[hbt]\n"
        "  \\centering\n"
        "  \\small\n"
        + caption_block
        + f"  \\label{{tab:{table_name}}}\n"
        "  \\begin{tabular}{ccccc}\n"
        "    \\toprule\n"
        "      f [\\unit{\\hertz}] &\n"
        "      $P_\\mathrm{fjern}/P_\\mathrm{nær}$ (snitt) &\n"
        "      Pearsons $\\rho$ &\n"
        "      Beste probe &\n"
        "      Variansøkning ved snitt [\\%]\\\\\n"
        "    \\midrule\n"
        + body_str + "\n"
        "    \\bottomrule\n"
        "  \\end{tabular}\n"
        "\\end{table}\n"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(immutable + "\n" + table_body, encoding="utf-8")


def write_tex_table(rows, out_path, *, range_label, wind_label, scope_note):
    """Render `rows` (from harmonic_summary) as a thesis-ready LaTeX tabular.

    Reader-friendly Option-D layout (5 columns):
      Bølgefrekvens | P_fjern/P_nær (snitt) | Pearsons ρ | Beste probe |
      Variansøkning ved snitt [%]

    `range_label`  in {'lowrange', 'highrange'}  — probe-hardware sensitivity
    `wind_label`   in {'nowind', 'fullwind'}     — wind condition filter
    `scope_note`   one-line description of the folder/wind subset, dropped
                   into the IMMUTABLE inputs block so the .tex documents what
                   it was generated from.
    """
    from datetime import datetime as _dt
    from wavescripts.plot_utils import _lookup_central_caption

    table_name = _table_name(range_label, wind_label)
    caption_full = _lookup_central_caption(table_name, kind="full")
    caption_short = _lookup_central_caption(table_name, kind="short")
    # Wrap the caption in CAPTION-SYNC sentinels so the caption can be
    # rewritten in-place later via analysis_scratch/sync_captions.py.
    if caption_full and caption_short:
        _cap_inner = (f"  \\caption[{caption_short}]{{\n"
                      f"    {caption_full}\n  }}")
    elif caption_full:
        _cap_inner = f"  \\caption{{\n    {caption_full}\n  }}"
    else:
        _cap_inner = ("  \\caption{\n"
                      "    % TODO: write caption "
                      "(edit FIGURE_CAPTIONS in main_save_figures.py)\n"
                      "  }")
    caption_block = (
        "% >>> CAPTION-SYNC START "
        "(do not edit this block; sync_captions.py overwrites)\n"
        f"{_cap_inner}\n"
        "% <<< CAPTION-SYNC END\n"
    )

    n_per_freq = ", ".join(f"{r['freq']:.1f}Hz:n={r['n']}" for r in rows)
    freq_list = ", ".join(f"{r['freq']:.1f}" for r in rows)
    immutable = "\n".join([
        "%! TEX root = ../main.tex",
        "% ==============================================================",
        "% IMMUTABLE — generated automatically, do not edit this block",
        "%",
        "% — Provenance ───────────────────────────────────────────────────",
        "%   script            : analysis_scratch/parallel_probe_psd_agreement.py",
        "%   plot_type         : parallel_probe_psd_agreement_table",
        "%   chapter           : 04",
        f"%   generated_at      : {_dt.now().isoformat(timespec='seconds')}",
        f"%   caption_label     : tab:{table_name}",
        f"%   caption_short     : {caption_short or ''}",
        "%",
        "% — Method ────────────────────────────────────────────────────",
        "%   Pairwise comparison of 9373/170 (wall) and 9373/340 (far) at",
        "%   each thesis paddle frequency (1.3, 1.4, 1.5, 1.6 Hz).",
        "%   P_fjern/P_nær : geometric mean across runs of P_far[bin]/P_wall[bin]",
        "%               at the PSD bin nearest f, computed as 10^(mean_dB/10).",
        "%               1.00 = perfect calibration agreement. Geometric mean is",
        "%               used because it pairs naturally with the dB math and is",
        "%               robust to a single run with near-zero P_wall.",
        "%   Pearsons ρ : Pearson correlation across runs of band-integrated",
        "%               amplitudes A = sqrt(2·∫ S(f) df) over ±0.1 Hz of f.",
        "%               High ρ ⇒ the two probes track the same physical wave.",
        "%   Beste probe: the single probe with the smallest std(A) across runs",
        "%               (lowest scatter ⇒ better single estimator).",
        "%   Variansøkning ved snitt [%] : % change in Var(½(A_wall + A_far))",
        "%               vs the smaller of Var(A_wall), Var(A_far). Positive",
        "%               ⇒ averaging worsens precision relative to the better",
        "%               single probe (the two probes are not independent).",
        "%",
        "% — Inputs ────────────────────────────────────────────────────",
        f"%   N runs per freq   : {n_per_freq}  (panel-full, quality-ok, both probes)",
        f"%   wind condition    : {wind_label}",
        f"%   probe-range setup : {range_label}",
        f"%   data scope        : {scope_note}",
        f"%   target frequencies: {freq_list} Hz",
        "%",
        "% ── end immutable block ─────────────────────────────────────────",
    ])

    body_lines = _format_body_lines(rows)

    table_body = (
        "\\begin{table}[hbt]\n"
        "  \\centering\n"
        "  \\small\n"
        + caption_block
        + f"  \\label{{tab:{table_name}}}\n"
        "  \\begin{tabular}{ccccc}\n"
        "    \\toprule\n"
        "      f [\\unit{\\hertz}] &\n"
        "      $P_\\mathrm{fjern}/P_\\mathrm{nær}$ (snitt) &\n"
        "      Pearsons $\\rho$ &\n"
        "      Beste probe &\n"
        "      Variansøkning ved snitt [\\%]\\\\\n"
        "    \\midrule\n"
        + "\n".join(body_lines) + "\n"
        "    \\bottomrule\n"
        "  \\end{tabular}\n"
        "\\end{table}\n"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(immutable + "\n" + table_body)
    print(f"Saved -> {out_path}")


# === Stdout reporting =======================================================

def print_summary_table(rows, n_runs):
    print()
    print("=" * 80)
    print(f"PARALLEL-PROBE PSD AGREEMENT  (n_runs = {n_runs})")
    print("=" * 80)
    print("Per-bin paired comparison at each target frequency")
    print(f"  {'f [Hz]':>7}  {'n':>4}  {'mean dB':>8}  "
          f"{'sd dB':>8}  {'p (paired t)':>12}  {'r(A_w,A_f)':>11}")
    for r in rows:
        print(f"  {r['freq']:>7.2f}  {r['n']:>4d}  "
              f"{r['mean_db']:>+8.2f}  {r['std_db']:>8.2f}  "
              f"{r['p_value']:>12.3g}  {r['corr']:>+11.3f}")

    print()
    print("Band-integrated amplitude (±0.1 Hz) — variance / std reduction")
    print(f"  {'f [Hz]':>7}  {'n':>4}  {'sd A_wall':>10}  {'sd A_far':>10}  "
          f"{'sd A_mean':>10}  {'var reduce vs best [%]':>23}")
    for r in rows:
        print(f"  {r['freq']:>7.2f}  {r['n_band']:>4d}  "
              f"{r['std_a']:>10.4f}  {r['std_b']:>10.4f}  "
              f"{r['std_mean']:>10.4f}  {r['reduction_pct']:>23.1f}")
    print("=" * 80)
    print("Note: variance pools across all runs in the subset (incl. condition")
    print("variation). Restrict the filter in main() to a single (freq, amp,")
    print("wind) cell to estimate pure measurement reproducibility.")


def print_implications(rows):
    """Plain-language summary of the headline numbers."""
    if not rows:
        return
    # Pick the bin closest to the canonical 1.4 Hz paddle frequency
    canonical = min(rows, key=lambda r: abs(r["freq"] - 1.4))
    fh = canonical["freq"]
    print()
    print("PLAIN-LANGUAGE IMPLICATIONS (canonical f = {:.2f} Hz)".format(fh))
    print("-" * 80)
    print(f"  Typical far-vs-wall PSD difference: {canonical['mean_db']:+.2f} dB "
          f"(p = {canonical['p_value']:.3g}).")
    if np.isfinite(canonical["corr"]):
        print(f"  Cross-run correlation of band amplitudes: r = "
              f"{canonical['corr']:+.3f}.")
    if np.isfinite(canonical["reduction_pct"]):
        sign = "reduces" if canonical["reduction_pct"] > 0 else "INCREASES"
        print(f"  Averaging both probes {sign} amplitude variance by "
              f"~{abs(canonical['reduction_pct']):.1f}% vs the better single probe.")
    p = canonical["p_value"]
    if not np.isfinite(p):
        verdict = "  (Too few runs for a verdict — increase the filter scope.)"
    elif p > 0.05 and abs(canonical["mean_db"]) < 1.0:
        verdict = ("  -> Probes are statistically indistinguishable at this band; "
                   "either is sufficient. Averaging still trims variance.")
    else:
        verdict = ("  -> Probes differ systematically at this band; "
                   "report the mean and disclose the spread.")
    print(verdict)


# === Project-specific data loader (the only non-portable bit) ===============

def _load_psd_data_from_project(target_freqs=TARGET_FREQS,
                                only_fullpanel=True,
                                range_label="lowrange",
                                wind_label=None):
    """
    Convert the project's `{csv_path: DataFrame}` PSD cache into the simple
    `{run_id: {probe: {f, Pxx}}}` format the analysis core consumes.

    Parameters
    ----------
    target_freqs : iterable of float
        Paddle frequencies to keep (within ±0.005 Hz).
    only_fullpanel : bool
        Restrict to PanelCondition == "full" (default true; methodology check).
    range_label : {"lowrange", "highrange"}
        Probe-hardware sensitivity setup. Selects a different folder pool:
          - "lowrange"  : 26 + 27 March 2026 height100-lowrange folders
                          (parallel-probe canon, low-sensitivity hardware).
          - "highrange" : 16 + 19 March 2026 under9Mooring folders
                          (parallel-probe canon, high-sensitivity hardware,
                           272 mm above tank). Smaller n (~11 thesis runs).
        The poor-quality high-range 100mm batch and the 136mm batch are
        deliberately excluded per CLAUDE-level decision (2026-05-06).
    wind_label : {"nowind", "fullwind", None}
        WindCondition filter. None pools all wind conditions (legacy).
    """
    from wavescripts.improved_data_loader import load_analysis_data

    base = Path(__file__).parent.parent
    if range_label == "lowrange":
        target_dirs = [
            base / "waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange",
            base / "waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange",
        ]
    elif range_label == "highrange":
        target_dirs = [
            base / "waveprocessed/PROCESSED-20260316-ProbePos4_31_FPV_2-tett6roof-under9Mooring",
            base / "waveprocessed/PROCESSED-20260319-ProbePos4_31_FPV_2-tett6roof-under9Mooring",
        ]
    else:
        raise ValueError(f"range_label must be 'lowrange' or 'highrange', got {range_label!r}")

    meta, _proc, _fft, psd_dict = load_analysis_data(
        *map(str, target_dirs), load_processed=False
    )
    mask = (
        meta["WaveFrequencyInput [Hz]"].notna()
        & meta["WaveFrequencyInput [Hz]"].apply(
            lambda v: any(abs(v - fh) < 0.005 for fh in target_freqs))
        & (meta["quality_flag"] == "ok")
    )
    if only_fullpanel:
        mask &= (meta["PanelCondition"] == "full")
    if wind_label == "nowind":
        mask &= (meta["WindCondition"] == "no")
    elif wind_label == "fullwind":
        mask &= (meta["WindCondition"] == "full")
    elif wind_label is not None:
        raise ValueError(f"wind_label must be 'nowind', 'fullwind', or None, got {wind_label!r}")

    sel_paths = meta.loc[mask, "path"].tolist()
    sel_paths = [p for p in sel_paths if p in psd_dict]
    print(f"  {len(sel_paths)} runs selected for {range_label}/{wind_label or 'pooled'}.")

    psd_data = {}
    for path in sel_paths:
        df = psd_dict[path]
        # Include parent folder in the run_id; the two canon March-2026
        # folders share filenames (only the mooring config differs in the
        # folder name), so Path(path).stem alone collides.
        run_id = f"{Path(path).parent.name}/{Path(path).stem}"
        per_run = {}
        for probe in PROBES:
            col = f"Pxx {probe}"
            if col not in df.columns:
                continue
            ser = df[col].dropna()
            if len(ser) < 2:
                continue
            per_run[probe] = {
                "f": ser.index.to_numpy(),
                "Pxx": ser.to_numpy(),
            }
        # Only keep runs where BOTH probes have data — the paired t-test
        # downstream requires row-aligned stacks.
        if all(p in per_run for p in PROBES):
            psd_data[run_id] = per_run
    print(f"  {len(psd_data)} runs retained (both parallel probes present).")
    return psd_data


# === Driver =================================================================

def main():
    """Generate one .tex table per (range, wind) cell — 4 tables total.

    Also keeps the legacy three-panel diagnostic PDF, regenerated from the
    pooled lowrange (no wind filter) for reference.
    """
    base = Path(__file__).resolve().parent.parent
    tables_dir = base / "output" / "TABLES"

    # The four reader-facing tables (Option D, 5 cols, Norwegian headers).
    scope_notes = {
        ("lowrange", "nowind"):
            "low-range hardware, 100mm above; under9Mooring + under9Mooring30 (Mar 26+27 2026); WindCondition=no",
        ("lowrange", "fullwind"):
            "low-range hardware, 100mm above; under9Mooring + under9Mooring30 (Mar 26+27 2026); WindCondition=full",
        ("highrange", "nowind"):
            "high-range hardware, 272mm above; under9Mooring (Mar 16+19 2026); WindCondition=no",
        ("highrange", "fullwind"):
            "high-range hardware, 272mm above; under9Mooring (Mar 16+19 2026); WindCondition=full",
    }
    last_rows = None
    last_harmonized = None
    last_f = None
    rows_cache: dict[tuple[str, str], list] = {}   # (range, wind) → rows
    for range_label in ("lowrange", "highrange"):
        for wind_label in ("nowind", "fullwind"):
            print(f"\n=== {range_label} / {wind_label} ===")
            psd_data = _load_psd_data_from_project(
                range_label=range_label, wind_label=wind_label
            )
            if not psd_data:
                print(f"  No runs — skipping {range_label}/{wind_label}.")
                continue
            f, harmonized = harmonize_grid(psd_data, n_grid=N_GRID, f_max=F_MAX_HZ)
            rows = harmonic_summary(harmonized, f, PROBES, TARGET_FREQS,
                                    BAND_HALFWIDTH_HZ)
            rows_cache[(range_label, wind_label)] = rows

            # The four per-(range × wind) .tex files were dropped 2026-05-09
            # — superseded by the two merged tables below
            # (ch04_parallel_probe_psd_agreement_{low,high}range_merged).
            # write_tex_table() is intentionally left in the module so the
            # individual outputs can be re-enabled by uncommenting one line:
            #     write_tex_table(rows, tables_dir / f"{_table_name(...)}.tex",
            #                     range_label=..., wind_label=...,
            #                     scope_note=scope_notes[(..., ...)])
            print_summary_table(rows, n_runs=len(harmonized))
            if range_label == "lowrange":
                last_rows = rows
                last_harmonized = harmonized
                last_f = f

    # Merged-winds tables (2026-05-09): each (range) gets one table that
    # stacks both wind conditions via \multicolumn{5}{l}{\itshape ...}
    # sub-section headers. Style modelled on tab:ch04_wind_setup_baseline_table.
    # Cell formatting reuses _format_body_lines so the merged table is
    # cell-for-cell identical to the union of the per-(range × wind) tables.
    for merge_range in ("lowrange", "highrange"):
        nw_rows = rows_cache.get((merge_range, "nowind"))
        fw_rows = rows_cache.get((merge_range, "fullwind"))
        if nw_rows and fw_rows:
            print(f"\n=== {merge_range} merged (uten + full vind) ===")
            out_tex = tables_dir / f"{THESIS_TABLE_BASE}_{merge_range}_merged.tex"
            write_merged_winds_tex_table(
                {"nowind": nw_rows, "fullwind": fw_rows},
                out_tex,
                range_label=merge_range,
                scope_notes=scope_notes,
            )
            print(f"  Wrote {out_tex.relative_to(base)}")

    # Legacy 3-panel diagnostic PDF (pooled lowrange, all winds) — kept for the
    # pre-existing CH04 figure pipeline. Falls back to the lowrange+nowind
    # subset if the pooled load returns nothing for some reason.
    print("\n=== Diagnostic PDF (pooled lowrange, all winds) ===")
    psd_pooled = _load_psd_data_from_project(
        range_label="lowrange", wind_label=None
    )
    if psd_pooled:
        f, harmonized = harmonize_grid(psd_pooled, n_grid=N_GRID, f_max=F_MAX_HZ)
        summary = run_averaged(harmonized, PROBES)
        stack_a = stack_runs(harmonized, PROBES[0])
        stack_b = stack_runs(harmonized, PROBES[1])
        _, diff_mean, diff_std, _, _ = per_freq_paired_diff(stack_a, stack_b)
        out_pdf = Path(__file__).parent / "parallel_probe_psd_agreement.pdf"
        plot_three_panel(f, summary, diff_mean, diff_std, TARGET_FREQS, str(out_pdf))
        # Plain-language headline from the pooled data (handy console summary).
        rows_pooled = harmonic_summary(harmonized, f, PROBES, TARGET_FREQS,
                                       BAND_HALFWIDTH_HZ)
        print_implications(rows_pooled)


if __name__ == "__main__":
    main()
