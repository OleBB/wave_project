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
    """
    stack_a = stack_runs(harmonized, probes[0])
    stack_b = stack_runs(harmonized, probes[1])
    rows = []
    for fh in target_freqs:
        # Nearest bin for the per-bin dB test
        bin_idx = int(np.argmin(np.abs(f - fh)))
        d_db = safe_db(stack_b[:, bin_idx]) - safe_db(stack_a[:, bin_idx])
        d_db = d_db[np.isfinite(d_db)]
        if len(d_db) >= 2:
            tt = stats.ttest_rel(safe_db(stack_b[:, bin_idx]),
                                 safe_db(stack_a[:, bin_idx]),
                                 nan_policy="omit")
            mean_db, std_db, p_val = float(np.mean(d_db)), float(np.std(d_db, ddof=1)), float(tt.pvalue)
        else:
            mean_db, std_db, p_val = np.nan, np.nan, np.nan

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
                                only_fullpanel=True):
    """
    Convert the project's `{csv_path: DataFrame}` PSD cache into the simple
    `{run_id: {probe: {f, Pxx}}}` format the analysis core consumes.

    Filter defaults: fullpanel, quality_flag=ok, WaveFrequencyInput in
    `target_freqs` (broad pool so the per-bin paired test has decent n).
    Edit the `mask` block below to reshape.
    """
    from wavescripts.improved_data_loader import load_analysis_data

    base = Path(__file__).parent.parent
    target_dirs = [
        base / "waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange",
        base / "waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange",
    ]
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
    sel_paths = meta.loc[mask, "path"].tolist()
    sel_paths = [p for p in sel_paths if p in psd_dict]
    print(f"  {len(sel_paths)} runs selected from canon March-2026 lowrange.")

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
    print("Loading PSD data ...")
    psd_data = _load_psd_data_from_project()

    f, harmonized = harmonize_grid(psd_data, n_grid=N_GRID, f_max=F_MAX_HZ)
    summary = run_averaged(harmonized, PROBES)
    stack_a = stack_runs(harmonized, PROBES[0])
    stack_b = stack_runs(harmonized, PROBES[1])
    _, diff_mean, diff_std, _, _ = per_freq_paired_diff(stack_a, stack_b)

    rows = harmonic_summary(harmonized, f, PROBES, TARGET_FREQS,
                            BAND_HALFWIDTH_HZ)

    out_pdf = Path(__file__).parent / "parallel_probe_psd_agreement.pdf"
    plot_three_panel(f, summary, diff_mean, diff_std, TARGET_FREQS, str(out_pdf))
    print_summary_table(rows, n_runs=len(harmonized))
    print_implications(rows)


if __name__ == "__main__":
    main()
