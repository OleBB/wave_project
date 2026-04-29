"""
H&G window fitness check — does the snapped H&G window fit inside per40?
=========================================================================

Successor to the (archived) `hg_window_stability_with_per40.py`. That
earlier figure asked: "does SNARVEI's eyeballed window for per40 reach
the same plateau that per240 sits on?" — by overlaying sliding-AFFT
curves.

This script asks the *current* version of the same question:

    Now that the pipeline uses the probe-shifted Huseby–Grue window
    with start + end snapped to zero-upcrossings, does that window
    sit inside the per40 wavetrain — or does it drift past paddle
    stop into ringdown / wind-only territory?

Approach: for canonical thesis conditions (1.4 Hz, 0.2 V, full panel,
quality_flag=ok), pick one representative run from each of the four
(run_type × wind) cells. Plot η(t) at IN and OUT probes; shade the
ACTUAL snapped H&G window read from meta.json's "Computed Probe {pos}
start/end" columns. Mark t_paddle_stop (= N_input_periods / f_paddle).

Visual verdict:
  - per240: paddle stops at ~171 s — H&G window (~[42, 49] s at IN,
    ~[35, 42] s at OUT after snap) sits well inside the wavetrain.
  - per40 : paddle stops at ~28.6 s — H&G window may push past stop
    at the OUT probe. The question is whether the *signal* there is
    still the wavemaker-driven train or already ringdown.

Run from repo root:
    /opt/anaconda3/envs/draumkvedet/bin/python analysis_scratch/hg_per40_window_fitness.py

Outputs (scratch, diagnostic only — not a thesis figure):
    analysis_scratch/hg_per40_window_fitness_f13.{pdf,png}
    analysis_scratch/hg_per40_window_fitness_f14.{pdf,png}
    analysis_scratch/hg_per40_window_fitness_f15.{pdf,png}
    analysis_scratch/hg_per40_window_fitness_f16.{pdf,png}
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

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import os
os.chdir(BASE)

from wavescripts.improved_data_loader import load_analysis_data, load_processed_dfs
from wavescripts.plot_utils import apply_thesis_style, WIND_COLOR_MAP

apply_thesis_style()

# ── Config ──────────────────────────────────────────────────────────────
FS              = 250.0
TARGET_FREQS    = [1.3, 1.4, 1.5, 1.6]   # all four thesis-scope frequencies
TARGET_AMP      = 0.2
PER240_THRESHOLD_T = 50

PROBES = ["9373/170", "12400/250"]
PROBE_LABEL = {"9373/170": "IN  (9373/170)", "12400/250": "OUT (12400/250)"}

# Canon — march-2026 cond4 lowrange
PROCESSED_DIRS = [
    Path("waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange"),
    Path("waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"),
]

SCRATCH_DIR = Path(__file__).parent


# ── Load ────────────────────────────────────────────────────────────────
print("1. Loading meta + processed_dfs (canon March-2026 lowrange) …")
combined_meta, _, _, _ = load_analysis_data(*PROCESSED_DIRS, load_processed=False)
processed_dfs = load_processed_dfs(*PROCESSED_DIRS)
print(f"   meta: {len(combined_meta)} rows · processed_dfs: {len(processed_dfs)} runs")


# ── Helper: filter + pick one representative run per (run_type, wind) ───
def _safe_get_amp(s):
    return pd.to_numeric(s, errors="coerce")


def select_canon(target_freq: float) -> pd.DataFrame:
    mask = (
        np.isclose(_safe_get_amp(combined_meta["WaveFrequencyInput [Hz]"]),
                   target_freq, atol=0.02)
        & np.isclose(_safe_get_amp(combined_meta["WaveAmplitudeInput [Volt]"]),
                     TARGET_AMP, atol=0.01)
        & (combined_meta["PanelCondition"] == "full")
        & (combined_meta["quality_flag"] == "ok")
    )
    sub = combined_meta[mask].copy()
    sub["N_input_periods"] = pd.to_numeric(sub["WavePeriodInput"], errors="coerce")
    sub["run_type"] = np.where(sub["N_input_periods"] >= PER240_THRESHOLD_T,
                               "per240", "per40")
    return sub


def pick_run(scope: pd.DataFrame, run_type: str, wind: str):
    s = scope[(scope["run_type"] == run_type) & (scope["WindCondition"] == wind)]
    if s.empty:
        return None
    # Prefer runs with both probes' "Computed Probe ... start/end" populated
    for _, r in s.iterrows():
        if all(pd.notna(r.get(f"Computed Probe {p} start")) and
               pd.notna(r.get(f"Computed Probe {p} end"))
               for p in PROBES):
            return r
    return s.iloc[0]


# ── Per-frequency figure builder ────────────────────────────────────────
def build_figure_for(target_freq: float):
    sel = select_canon(target_freq)
    print(f"\n— f={target_freq:.2f} Hz: {len(sel)} canonical runs "
          f"(per40 n={int((sel['run_type']=='per40').sum())}, "
          f"per240 n={int((sel['run_type']=='per240').sum())})")

    cells = [(rt, w) for rt in ("per40", "per240") for w in ("no", "full")]
    picks = {(rt, w): pick_run(sel, rt, w) for rt, w in cells}
    for (rt, w), r in picks.items():
        if r is None:
            print(f"   WARN no run for ({rt}, {w}) at {target_freq:.2f} Hz")
        else:
            print(f"   ({rt}, {w}) → {Path(str(r['path'])).name}")

    fig, axes = plt.subplots(4, 2, figsize=(11, 10), sharex=False)

    row_order = [
        ("per40",  "no"),   # row 0
        ("per40",  "full"), # row 1
        ("per240", "no"),   # row 2
        ("per240", "full"), # row 3
    ]

    for row_i, (rt, wind) in enumerate(row_order):
        run = picks.get((rt, wind))
        color = WIND_COLOR_MAP[wind]
        n_periods = float(run["N_input_periods"]) if run is not None else np.nan
        f_paddle  = (float(run["WaveFrequencyInput [Hz]"])
                     if run is not None else target_freq)
        t_paddle_stop = (n_periods / f_paddle
                         if np.isfinite(n_periods) else np.nan)

        for col_i, probe in enumerate(PROBES):
            ax = axes[row_i, col_i]
            if run is None:
                ax.text(0.5, 0.5, "no run", ha="center", va="center",
                        transform=ax.transAxes, color="gray")
                ax.set_xticks([]); ax.set_yticks([])
                continue

            df = processed_dfs.get(run["path"])
            eta_col = f"eta_{probe}"
            if df is None or eta_col not in df.columns:
                ax.text(0.5, 0.5, f"no {eta_col}", ha="center", va="center",
                        transform=ax.transAxes, color="gray", fontsize=8)
                continue

            sig = df[eta_col].values
            t   = np.arange(len(sig)) / FS

            x_max = 60.0 if rt == "per40" else 80.0
            m = t <= x_max
            ax.plot(t[m], sig[m], color=color, lw=0.45, alpha=0.85)

            s_col = f"Computed Probe {probe} start"
            e_col = f"Computed Probe {probe} end"
            gs = run.get(s_col)
            ge = run.get(e_col)
            if pd.notna(gs) and pd.notna(ge):
                t_gs = float(gs) / FS
                t_ge = float(ge) / FS
                ax.axvspan(t_gs, t_ge, color="#2ECC71", alpha=0.22, lw=0,
                           label="snapped H&G window")
                ax.axvline(t_gs, color="#2ECC71", lw=0.6, ls="--", alpha=0.7)
                ax.axvline(t_ge, color="#2ECC71", lw=0.6, ls="--", alpha=0.7)
                ax.text(0.5 * (t_gs + t_ge), 0.92, f"[{t_gs:.1f}, {t_ge:.1f}] s",
                        transform=ax.get_xaxis_transform(),
                        ha="center", va="top", fontsize=7,
                        color="#1A6E2A", alpha=0.85)

            if np.isfinite(t_paddle_stop) and t_paddle_stop <= x_max:
                ax.axvline(t_paddle_stop, color="#7F3FBF", lw=1.0, ls=":")
                ax.text(t_paddle_stop + 0.3, 0.05,
                        f"paddle stop ({t_paddle_stop:.1f} s)",
                        transform=ax.get_xaxis_transform(), fontsize=7,
                        color="#7F3FBF", va="bottom", ha="left")

            ax.set_xlim(0, x_max)
            ax.grid(True, alpha=0.25, lw=0.4)
            ax.tick_params(labelsize=8)
            if col_i == 0:
                ax.set_ylabel(f"{rt} · {wind} wind\n$\\eta$ [mm]", fontsize=8)
            if row_i == 3:
                ax.set_xlabel("time from wavemaker start [s]", fontsize=8)
            if row_i == 0:
                ax.set_title("", fontsize=10)  # filled in by user later
            ax.text(0.99, 0.96, PROBE_LABEL[probe], transform=ax.transAxes,
                    ha="right", va="top", fontsize=7, color="#444",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#bbb",
                              alpha=0.85, lw=0.4))

    fig.suptitle("", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    f_tag = f"f{int(round(target_freq * 10))}"
    out_pdf = SCRATCH_DIR / f"hg_per40_window_fitness_{f_tag}.pdf"
    out_png = SCRATCH_DIR / f"hg_per40_window_fitness_{f_tag}.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"   saved → {out_pdf.relative_to(BASE)}")
    print(f"          {out_png.relative_to(BASE)}")


# ── Loop over all four thesis frequencies ───────────────────────────────
for _f in TARGET_FREQS:
    build_figure_for(_f)
