"""
T_cross figure — CH05 §3b alternative wind-effect metric
=========================================================

Three curves on the same axes, per amplitude:
  OUT/IN_nw   = A_out^nw / A_in^nw   (blue)  — the nowind baseline
  T_cross     = A_out^fw / A_in^nw   (green) — OUT under wind, referenced
                                               to the *clean* nowind IN
  OUT/IN_fw   = A_out^fw / A_in^fw   (red)   — standard fullwind metric

Green vs blue = the honest wind effect on transmission (clean incident
reference). Red vs blue = the effect as the standard metric shows it.
Their disagreement at higher frequency is the IN-probe wind-contamination
bias (CLAUDE.md §16).

The three curves reduce to the identity T_cross / (OUT/IN_nw) = A_out^fw
/ A_out^nw — a ratio that lives entirely on the OUT probe, which is
sheltered by the panel and immune to the IN-contamination problem.

Prerequisite: A_in must be approximately wind-independent at each
frequency within a matched mooring group (paddle output should not
depend on whether the fan is on). The original T_cross idea doc
(`t_cross_idea.md`) showed the loose230/loose300 moorings satisfy this
at 1.3 Hz within ±5%. Here we pool loose230+loose300 (already merged
in meta_results per the CH04 §3c finding that mooring length has no
detectable transmission effect in the thesis band), which gives enough
samples per (freq, amp) to compute SEM.

Run from repo root:
    conda run -n draumkvedet python analysis_scratch/t_cross_figure.py

Outputs:
    analysis_scratch/t_cross_figure.pdf             (combined quick-view)
    analysis_scratch/t_cross_figure_summary.csv      (per-(freq,amp) numbers)
    output/FIGURES/ch05_t_cross_{10,20,30}V.pdf      (thesis subfigures)
    output/TEXFIGU/ch05_t_cross.tex                  (stub, write-once)
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import os
os.chdir(BASE)

from wavescripts.improved_data_loader import load_analysis_data
from wavescripts.plot_utils import apply_thesis_style, freq_to_k, add_freq_axis, amp_to_label, amp_to_tag

apply_thesis_style()

# ── I/O ────────────────────────────────────────────────────────────────────────
SCRATCH_PDF = Path(__file__).parent / "t_cross_figure.pdf"
SCRATCH_CSV = Path(__file__).parent / "t_cross_figure_summary.csv"
THESIS_BASE = "ch05_t_cross"
STUB_PATH   = BASE / "output" / "TEXFIGU" / f"{THESIS_BASE}.tex"

# meta_results: two validated lowrange folders, cond4 only
RESULTS_DIRS = [
    Path("waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange"),
    Path("waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"),
]

# Canonical IN reference is the mean of 9373/170 and 9373/340, written
# by processor2nd.py as "IN Amplitude (FFT)" (pipeline-level since
# 2026-04-18; see CLAUDE.md §5). OUT is the per-era canonical as well,
# in "OUT Amplitude (FFT)".
IN_POS, OUT_POS = "IN", "OUT"
IN_FFT  = "IN Amplitude (FFT)"
OUT_FFT = "OUT Amplitude (FFT)"

FREQS = [1.3, 1.4, 1.5, 1.6]   # thesis scope
AMPS  = [0.1, 0.2, 0.3]

# ── 1. Load ────────────────────────────────────────────────────────────────────
print("1. Loading meta_results …")
meta, _, _, _ = load_analysis_data(*RESULTS_DIRS, load_processed=False)
# Match main_save_figures.py: merge loose230/loose300 → below_90_loose
meta["Mooring"] = meta["Mooring"].replace({
    "below_90_loose230": "below_90_loose",
    "below_90_loose300": "below_90_loose",
})
# Canonical IN/OUT now live in meta.json (pipeline-level since
# 2026-04-18). No runtime hook needed.

wave = meta[
    meta["WaveFrequencyInput [Hz]"].notna()
    & (meta["WaveFrequencyInput [Hz]"] > 0)
    & (meta["PanelCondition"] == "full")
    & (meta["quality_flag"] == "ok")
    & meta[IN_FFT].notna()
    & meta[OUT_FFT].notna()
    & (meta[IN_FFT] > 0)
].copy()
wave["freq_r"] = wave["WaveFrequencyInput [Hz]"].round(2)
wave["amp_r"]  = wave["WaveAmplitudeInput [Volt]"].round(2)
print(f"   {len(wave)} wave runs available")

# ── 2. Compute per-(freq, amp) stats ──────────────────────────────────────────
def mean_std_n(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan, np.nan, 0
    return float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0, int(len(vals))

rows = []
for amp in AMPS:
    for f in FREQS:
        sub = wave[(wave["amp_r"] == amp) & (wave["freq_r"] == f)]
        nw = sub[sub["WindCondition"] == "no"]
        fw = sub[sub["WindCondition"] == "full"]

        A_in_nw_m, A_in_nw_s, n_nw   = mean_std_n(nw[IN_FFT])
        A_out_nw_m, A_out_nw_s, _    = mean_std_n(nw[OUT_FFT])
        A_in_fw_m, A_in_fw_s, n_fw   = mean_std_n(fw[IN_FFT])
        A_out_fw_m, A_out_fw_s, _    = mean_std_n(fw[OUT_FFT])

        # Per-run OUT/IN values (mean, std)
        outin_nw = (nw[OUT_FFT] / nw[IN_FFT]).values
        outin_fw = (fw[OUT_FFT] / fw[IN_FFT]).values
        outin_nw_m, outin_nw_s, _ = mean_std_n(outin_nw)
        outin_fw_m, outin_fw_s, _ = mean_std_n(outin_fw)

        # T_cross: A_out_fw / A_in_nw. Compute per-fw-run then divide by
        # the mean A_in_nw (bootstrap / SEM from the numerator spread).
        if n_nw > 0 and n_fw > 0 and A_in_nw_m > 0:
            tcross_values = fw[OUT_FFT].values / A_in_nw_m
            tcross_m, tcross_s, _ = mean_std_n(tcross_values)
            # Propagate the nowind denominator uncertainty (quadrature).
            rel_den = (A_in_nw_s / A_in_nw_m) if A_in_nw_m > 0 else 0.0
            rel_num = (tcross_s / tcross_m) if tcross_m > 0 else 0.0
            rel_tot = np.sqrt(rel_num**2 + rel_den**2)
            tcross_s_total = tcross_m * rel_tot
        else:
            tcross_m, tcross_s_total = np.nan, np.nan

        rows.append({
            "amp": amp, "freq": f, "k": float(freq_to_k(np.array([f]))[0]),
            "n_nw": n_nw, "n_fw": n_fw,
            "A_in_nw": A_in_nw_m, "A_in_nw_std": A_in_nw_s,
            "A_in_fw": A_in_fw_m, "A_in_fw_std": A_in_fw_s,
            "A_out_nw": A_out_nw_m, "A_out_nw_std": A_out_nw_s,
            "A_out_fw": A_out_fw_m, "A_out_fw_std": A_out_fw_s,
            "outin_nw": outin_nw_m, "outin_nw_std": outin_nw_s,
            "outin_fw": outin_fw_m, "outin_fw_std": outin_fw_s,
            "tcross":   tcross_m,   "tcross_std":   tcross_s_total,
        })
df = pd.DataFrame(rows)
df.to_csv(SCRATCH_CSV, index=False)
print(f"   Stats → {SCRATCH_CSV.relative_to(BASE)}")

# Also verify A_in wind-independence (paddle output consistent with/without wind)
print("\n2. A_in wind-independence check per (freq, amp) — paddle consistency")
print(f"   {'amp':>4} {'freq':>5}  {'A_in_nw':>8} ± {'std':<5}  {'A_in_fw':>8} ± {'std':<5}  {'ratio fw/nw':>11}")
for _, r in df.iterrows():
    if r["n_nw"] == 0 or r["n_fw"] == 0:
        continue
    ratio = r["A_in_fw"] / r["A_in_nw"] if r["A_in_nw"] > 0 else np.nan
    flag = "⚠" if (ratio < 0.90 or ratio > 1.15) else ""
    print(f"   {r['amp']:>4.1f} {r['freq']:>5.1f}  "
          f"{r['A_in_nw']:>8.3f} ± {r['A_in_nw_std']:<5.3f}  "
          f"{r['A_in_fw']:>8.3f} ± {r['A_in_fw_std']:<5.3f}  "
          f"{ratio:>10.3f}  {flag}")

# ── 3. Plot — 3 separate thesis PDFs + combined scratch preview ──────────────
print("\n3. Plotting …")

COLOR_NW     = "#2E86AB"  # blue
COLOR_TCROSS = "#2ECC71"  # green
COLOR_FW     = "#E74C3C"  # red


def draw_t_cross_ax(ax, sub: pd.DataFrame, amp: float):
    """Three curves on one axes: OUT/IN_nw, T_cross, OUT/IN_fw."""
    if sub.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        return

    x = sub["k"].values

    def _line(y, yerr, color, label, marker, ls="-"):
        mask = np.isfinite(y)
        if mask.sum() == 0:
            return
        ax.errorbar(x[mask], y[mask],
                    yerr=(yerr[mask] if yerr is not None else None),
                    fmt=marker + ls, color=color, label=label,
                    capsize=3, markersize=5, lw=1.6, alpha=0.92)

    _line(sub["outin_nw"].values, sub["outin_nw_std"].values,
          COLOR_NW, r"$(OUT/IN)_{nw}$  clean baseline", "o")
    _line(sub["tcross"].values, sub["tcross_std"].values,
          COLOR_TCROSS, r"$T_{\mathrm{cross}}$  $= A_{out}^{fw} / A_{in}^{nw}$", "s")
    _line(sub["outin_fw"].values, sub["outin_fw_std"].values,
          COLOR_FW, r"$(OUT/IN)_{fw}$  standard metric", "^")

    # Shade the honest wind effect (green − blue) where both exist
    mask = np.isfinite(sub["outin_nw"].values) & np.isfinite(sub["tcross"].values)
    if mask.sum() >= 2:
        ax.fill_between(x[mask], sub["outin_nw"].values[mask],
                        sub["tcross"].values[mask],
                        color=COLOR_TCROSS, alpha=0.12, lw=0,
                        label="honest wind effect")

    ax.axhline(1.0, color="black", lw=0.6, ls="--", alpha=0.4)
    ax.set_xlabel(r"$k$ (rad/m)", fontsize=9)
    ax.set_ylabel("transmission", fontsize=9)
    ax.set_title("", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="lower left", framealpha=0.92)
    add_freq_axis(ax)


# Combined preview (3 panels side-by-side)
fig = plt.figure(figsize=(16, 4.6))
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.32,
                       left=0.05, right=0.99, top=0.84, bottom=0.18)
for i, amp in enumerate(AMPS):
    ax = fig.add_subplot(gs[0, i])
    draw_t_cross_ax(ax, df[df["amp"] == amp].sort_values("k"), amp)
fig.suptitle("", fontsize=11, fontweight="bold", y=0.95)
fig.text(0.5, 0.02,
         r"Green line uses the clean nowind IN amplitude as reference, sidestepping "
         r"in-probe wind contamination. The green–blue gap is the honest wind effect; "
         r"the red–blue gap is what the raw $(OUT/IN)_{fw}$ metric shows. "
         r"Where they disagree, the raw metric is biased.",
         ha="center", fontsize=8, color="#444", style="italic")
SCRATCH_PDF.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(SCRATCH_PDF, bbox_inches="tight")
print(f"   Scratch preview → {SCRATCH_PDF.relative_to(BASE)}")
plt.close(fig)

# Thesis subfigures: one PDF per amplitude
thesis_names = []
for amp in AMPS:
    fig_s, ax_s = plt.subplots(figsize=(6, 3.6))
    draw_t_cross_ax(ax_s, df[df["amp"] == amp].sort_values("k"), amp)
    fig_s.subplots_adjust(left=0.12, right=0.97, top=0.84, bottom=0.15)
    amp_tag = amp_to_tag(amp)
    fname = f"{THESIS_BASE}_{amp_tag}"
    fpath = BASE / "output" / "FIGURES" / f"{fname}.pdf"
    fpath.parent.mkdir(parents=True, exist_ok=True)
    fig_s.savefig(fpath, bbox_inches="tight")
    print(f"   Thesis PDF    → {fpath.relative_to(BASE)}")
    plt.close(fig_s)
    thesis_names.append(fname)

# ── 4. LaTeX stub ─────────────────────────────────────────────────────────────
print("\n4. Writing .tex stub …")
STUB_PATH.parent.mkdir(parents=True, exist_ok=True)

# Headline number for the caption: max (T_cross − OUT/IN_fw) across the
# thesis band. Expresses "how much the standard metric underestimates
# the true wind effect at worst".
valid = df.dropna(subset=["tcross", "outin_fw"])
if not valid.empty:
    diff = valid["tcross"] - valid["outin_fw"]
    worst_idx = diff.abs().idxmax()
    worst = valid.loc[worst_idx]
    worst_str = (f"{worst['tcross']:.3f} vs {worst['outin_fw']:.3f} at "
                 f"$f = {worst['freq']:.1f}$\\,Hz, ${worst['amp']:.2f}$\\,V")
    worst_diff_abs = float(diff.abs().max())
else:
    worst_str = "—"
    worst_diff_abs = float("nan")

_caption = (
    r"Wind effect on transmission measured three ways: "
    r"$(OUT/IN)_{nw}$ (blue, nowind baseline); "
    r"$T_{\mathrm{cross}} = A_{out}^{fw}/A_{in}^{nw}$ (green, fullwind OUT referenced "
    r"to the clean nowind IN); and $(OUT/IN)_{fw}$ (red, standard fullwind metric). "
    r"The green--blue gap is the wind effect measured with the clean incident "
    r"reference. The red--blue gap is the same effect as reported by the standard "
    r"ratio; the two disagree where the IN probe is contaminated by wind noise at "
    r"the paddle frequency (CLAUDE.md \S 16). Worst-case divergence between the "
    rf"two metrics: {worst_str}. Each subfigure is one input amplitude; "
    r"$n$ per point in \texttt{analysis\_scratch/t\_cross\_figure\_summary.csv}."
)

# Use the shared plot_utils helpers so the stub matches the canonical
# schema (provenance / filters / data provenance / method / stats).
import wavescripts.plot_utils as pu
pu.ACTIVE_DATASETS = [str(p).split("/")[-1] for p in RESULTS_DIRS]
pu.TEXFIGU_DIR = BASE / "output" / "TEXFIGU"
pu.FIGURES_DIR = BASE / "output" / "FIGURES"

_meta_stub = pu.build_fig_meta(
    {
        "filters": {
            "PanelCondition":            "full",
            "WaveFrequencyInput [Hz]":   [min(FREQS), max(FREQS)],
            "WaveAmplitudeInput [Volt]": AMPS,
            "WindCondition":             ["no", "full"],
            "quality_flag":              "ok",
            "Mooring":                   ["below_90_loose"],
        },
        "plotting": {
            "figure_name": THESIS_BASE,
            "caption":     _caption,
            "caption_short": "T_cross: wind effect via clean nowind reference",
        },
    },
    chapter="05",
    data_df=wave,
    extra={"script": "analysis_scratch/t_cross_figure.py"},
    computed_in=("analysis_scratch/t_cross_figure.py "
                 "(groupby freq+amp+wind → mean/std of IN/OUT Amplitude (FFT); "
                 "T_cross = A_out^fw / mean(A_in^nw))"),
    data_class="DELEG",
    findings_doc="analysis_scratch/t_cross_figure_findings.md",
    grouper="manual (groupby freq, amp, wind)",
    collapse_panels=False,
    fft_window_hz=0.1,
    extra_params=(
        f"frequencies_hz={FREQS}, amplitudes_V={AMPS}, "
        "winds=['no','full'], moorings merged to below_90_loose"
    ),
    extra_stats={
        "worst_abs_tcross_minus_outinfw": round(worst_diff_abs, 4) if np.isfinite(worst_diff_abs) else "—",
        "worst_condition": worst_str,
        "n_freqs": len(FREQS),
        "n_amps":  len(AMPS),
    },
)

subfig_captions = [f"${a:.2f}$\\,V" for a in AMPS]
pu.write_figure_stub(_meta_stub, plot_type="t_cross",
                     subfig_filenames=thesis_names,
                     subfig_captions=subfig_captions)
print(f"   Wrote stub    → {STUB_PATH.relative_to(BASE)}")

print("\nDone.")
