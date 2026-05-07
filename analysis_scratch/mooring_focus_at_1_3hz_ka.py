"""
Mooring focus at 1.30 Hz — K_t vs ka, one figure per amplitude
================================================================

Iteration on `full_vs_reverse_at_1_3hz_ka.py`. Goal: drop dimensions and
let the mooring story breathe. The previous figure mashed amp tiers
together along ka and used 6 marker shapes + 4 colours + fill style. Too
much.

This variant:
  - **Three figures**, one per amp (A1, A2, A3) — amp dimension moves to
    the file/page, freeing visual space and giving each panel a cleaner
    horizontal spread of ka.
  - **Mooring is the primary contrast**:
      below_90 (canon + tidligere lumped together) → normal blue / red
      above_50                                      → cyan / bright pink
  - **All markers hollow** so overlapping points are visible.
  - **Hardware (canon vs tidligere) collapsed** — no fill distinction;
    just lumped under "below_90" for the visual. The previous
    probe-config diagnostics already established that the canon vs
    earlier hardware split inside below_90_loose230 doesn't carry a
    statistically defensible bias.
  - **Panel orientation × amp** kept as marker shape (consistent with the
    previous variant's marker family — even though amp is now one-figure):
        normal panel  → ○ A1, □ A2, △ A3
        revers panel  → 6-point star A1, 5-point star A2, 4-point star A3

Outputs (one PDF per amp + one combined column-stacked stub):
    analysis_scratch/mooring_focus_at_1_3hz_ka_A{1,2,3}.pdf
    output/FIGURES/ch05_mooring_focus_at_1_3hz_ka_A{1,2,3}.pdf
    output/TEXFIGU/ch05_mooring_focus_at_1_3hz_ka.tex      (single A4 page,
                                                            3 subfigs stacked)
    analysis_scratch/mooring_focus_at_1_3hz_ka_summary.csv
"""

import sys
import warnings
from pathlib import Path
import glob

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import patheffects as pe
from matplotlib.ticker import MultipleLocator

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import os
os.chdir(BASE)

from wavescripts.improved_data_loader import load_analysis_data
from wavescripts.plot_utils import (
    WIND_COLOR_MAP, amp_to_label, apply_thesis_style, freq_to_k,
    apply_horizontal_ylabel,
)
import wavescripts.plot_utils as pu

apply_thesis_style()

TARGET_FREQ = 1.30

# ── 1. Load & filter ───────────────────────────────────────────────────────────
print("1. Loading processed folders …")
all_dirs = sorted(glob.glob(str(BASE / "waveprocessed" / "PROCESSED-*")))
meta, _, _, _ = load_analysis_data(*all_dirs, load_processed=False)

def mooring_group(m):
    if m in ("below_90_loose230", "below_90_loose300"):
        return "below_90"
    if m == "above_50":
        return "above_50"
    return "other"
meta["moor_grp"] = meta["Mooring"].apply(mooring_group)

sel = meta[
    (meta["WaveFrequencyInput [Hz]"] == TARGET_FREQ)
    & meta["PanelCondition"].isin(["full", "reverse"])
    & (meta["quality_flag"] == "ok")
    & meta["OUT/IN (FFT)"].notna()
    & meta["IN Amplitude (FFT)"].notna()
    & meta["WindCondition"].isin(["no", "full"])
    & (meta["OUT/IN (FFT)"].between(0.1, 2.0))
    & meta["moor_grp"].isin(["below_90", "above_50"])
].copy()

k_const = float(freq_to_k(np.array([TARGET_FREQ]))[0])
sel["a_m"] = sel["IN Amplitude (FFT)"].astype(float) / 1000.0
sel["ka"]  = k_const * sel["a_m"]
sel["amp_v"] = sel["WaveAmplitudeInput [Volt]"].apply(lambda v: round(float(v), 2))

print(f"   {len(sel)} rows; k({TARGET_FREQ} Hz) = {k_const:.3f} rad/m")

# ── 2. Visual constants ────────────────────────────────────────────────────────
# Mooring × wind → colour. 4 colours, hollow markers everywhere.
COLOR = {
    ("below_90", "no"):   WIND_COLOR_MAP["no"],     # #1F77B4 — blue
    ("below_90", "full"): WIND_COLOR_MAP["full"],   # #D62728 — red
    ("above_50", "no"):   "#00CED1",                # cyan (DarkTurquoise)
    ("above_50", "full"): "#FF1493",                # bright pink (DeepPink)
}
# Panel orientation × amp → shape. Even though each figure shows one amp
# tier, we keep the same marker family from the previous variant so a reader
# flipping between figures doesn't have to re-learn the convention.
# (N, 1, 0) = matplotlib regular star with N points.
PANEL_AMP_MARKER = {
    ("full",    0.10): "o",
    ("full",    0.20): "s",
    ("full",    0.30): "^",
    ("reverse", 0.10): (6, 1, 0),   # 6-point star
    ("reverse", 0.20): (5, 1, 0),   # 5-point star
    ("reverse", 0.30): (4, 1, 0),   # 4-point star
}
PANEL_LABEL = {"full": "normal", "reverse": "revers"}
WIND_LABEL   = {"no": "uten vind", "full": "med vind"}
MOORING_LABEL = {"below_90": "Under",
                 "above_50": "Over"}

MARKER_SIZE = 90        # bigger so hollow rings are clearly visible
EDGE_LW     = 1.6
ALPHA       = 0.85

AMP_TIERS = [(0.10, "A1", "0.1V"),
             (0.20, "A2", "0.2V"),
             (0.30, "A3", "0.3V")]

# Per-amp horizontal extent — fixed 0.018-wide window centered on each
# amp's data midpoint. Same width across A1/A2/A3 so the three subfigures
# read at the same horizontal scale; 0.018 just fits A3 (widest natural
# spread, ~0.026 raw → 0.018 with slight clipping at the edges) while
# leaving A1 (narrowest spread) still readable rather than collapsed.
KA_WINDOW_WIDTH = 0.018
amp_x_window = {
    amp_v: (
        (sel.loc[sel["amp_v"] == amp_v, "ka"].min()
         + sel.loc[sel["amp_v"] == amp_v, "ka"].max()) / 2 - KA_WINDOW_WIDTH / 2,
        (sel.loc[sel["amp_v"] == amp_v, "ka"].min()
         + sel.loc[sel["amp_v"] == amp_v, "ka"].max()) / 2 + KA_WINDOW_WIDTH / 2,
    )
    for amp_v, _, _ in AMP_TIERS
    if (sel["amp_v"] == amp_v).any()
}
print(f"   ka windows per amp (width={KA_WINDOW_WIDTH}): {amp_x_window}")

# Shared y-range across all 3 amp figures for visual stacking.
y_lo = sel["OUT/IN (FFT)"].min() - 0.02
y_hi = sel["OUT/IN (FFT)"].max() + 0.02
print(f"   shared y-range: [{y_lo:.3f}, {y_hi:.3f}]")

# ── 3. Plot — one figure per amp ───────────────────────────────────────────────
summary_rows = []

for amp_v, amp_tag, amp_v_lbl in AMP_TIERS:
    sub = sel[sel["amp_v"] == amp_v]
    if sub.empty:
        print(f"   [{amp_tag}] no rows — skipping")
        continue

    fig, ax = plt.subplots(figsize=(7.0, 4.6))

    for (panel, moor, wind), grp in sub.groupby(
        ["PanelCondition", "moor_grp", "WindCondition"]
    ):
        if grp.empty:
            continue
        color  = COLOR[(moor, wind)]
        marker = PANEL_AMP_MARKER[(panel, amp_v)]
        ax.scatter(
            grp["ka"], grp["OUT/IN (FFT)"],
            facecolors="none", edgecolors=color, marker=marker,
            s=MARKER_SIZE, linewidths=EDGE_LW, alpha=ALPHA,
            zorder=3,
            path_effects=[
                pe.Stroke(linewidth=EDGE_LW + 1.0, foreground="black"),
                pe.Normal(),
            ],
        )
        summary_rows.append(dict(
            amp_tag=amp_tag, amp_v=amp_v,
            panel=panel, moor_grp=moor, wind=wind,
            n=int(len(grp)),
            ka_mean=float(grp["ka"].mean()),
            Kt_mean=float(grp["OUT/IN (FFT)"].mean()),
            Kt_std=float(grp["OUT/IN (FFT)"].std()) if len(grp) > 1 else None,
        ))

    ax.axhline(1.0, color="black", lw=0.6, ls="--", alpha=0.5)
    ax.set_xlim(*amp_x_window[amp_v])
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel(r"$ka$  (per kjøring; $k(1.30\,\mathrm{Hz})\cdot a_\mathrm{IN}$)",
                   fontsize=10)
    # No ax.set_title — caption belongs in the .tex stub (user-edited).
    ax.grid(which="major", alpha=0.30, lw=0.6)
    ax.grid(which="minor", alpha=0.15, lw=0.4)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(MultipleLocator(0.025))
    # Horizontal $K_t$ above leftmost tick → frees the left margin for max
    # plotting width (matches the ch05_damping_freq convention).
    apply_horizontal_ylabel(ax, r"$K_t$", fontsize=12)

    # Two-block legend.
    moor_wind_handles = [
        mlines.Line2D(
            [], [],
            color=COLOR[(m, w)],
            linestyle="-",      # line visible
            linewidth=2.0,
            marker=None,        # no marker
            label=f"{MOORING_LABEL[m]} · {WIND_LABEL[w]}",
        )
        for m in ["below_90", "above_50"]
        for w in ["no", "full"]
    ]

    panel_handles = [
        mlines.Line2D([], [], color="black",
                      marker=PANEL_AMP_MARKER[(p, amp_v)],
                      ms=10, lw=0, mfc="none", mec="black",
                      mew=EDGE_LW, label=PANEL_LABEL[p],
                      path_effects=[
                          pe.Stroke(linewidth=EDGE_LW + 1.0, foreground="black"),
                          pe.Normal(),
                      ])
        for p in ["full", "reverse"]
    ]
    leg1 = ax.legend(handles=moor_wind_handles, loc="upper right",
                      fontsize=8, framealpha=0.92,
                      title="Moring · vind", title_fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=panel_handles, loc="lower right",
               fontsize=8.5, framealpha=0.92,
               title="Panelretning", title_fontsize=8.5,
               bbox_to_anchor=(0.995, 0.005))

    fig.tight_layout()
    scratch = (Path(__file__).parent
                / f"mooring_focus_at_1_3hz_ka_{amp_tag}.pdf")
    out_pdf = (BASE / "output" / "FIGURES"
                / f"ch05_mooring_focus_at_1_3hz_ka_{amp_tag}.pdf")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(scratch, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    print(f"   [{amp_tag}] Saved → {scratch.relative_to(BASE)}")
    print(f"   [{amp_tag}] Saved → {out_pdf.relative_to(BASE)}")
    plt.close(fig)

# Summary CSV.
csv_path = (Path(__file__).parent
             / "mooring_focus_at_1_3hz_ka_summary.csv")
pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
print(f"\n   Summary → {csv_path.relative_to(BASE)}")

# ── 4. Combined TEXFIGU stub ──────────────────────────────────────────────────
# Single .tex file with subfig_layout="column" → 3 subfigs stacked vertically
# on one A4 page float. Mirrors the ch05_damping_freq pattern in plot_damping_freq.
# Caption text comes from FIGURE_CAPTIONS / FIGURE_CAPTIONS_SHORT in
# main_save_figures.py (looked up via output/.figure_captions.json).
COMBINED_NAME = "ch05_mooring_focus_at_1_3hz_ka"
SUBFIG_NAMES  = [f"{COMBINED_NAME}_{tag}"
                  for amp_v, tag, _ in AMP_TIERS
                  if (sel["amp_v"] == amp_v).any()]
SUBFIG_CAPS   = [f"{tag} ({lbl})"
                  for amp_v, tag, lbl in AMP_TIERS
                  if (sel["amp_v"] == amp_v).any()]

# Provenance datasets, like other DELEG figures.
pu.ACTIVE_DATASETS = [Path(d).name for d in all_dirs]

_meta_stub = pu.build_fig_meta(
    {
        "filters": {
            "PanelCondition":            "full, reverse",
            "WaveFrequencyInput [Hz]":   f"{TARGET_FREQ}",
            "WaveAmplitudeInput [Volt]": "0.10, 0.20, 0.30",
            "WindCondition":             "no, full",
            "Mooring":                   "below_90_loose230, below_90_loose300, above_50",
            "quality_flag":              "ok",
        },
        "plotting": {"figure_name": COMBINED_NAME},
    },
    chapter="05",
    extra={"script": "analysis_scratch/mooring_focus_at_1_3hz_ka.py"},
    computed_in=("analysis_scratch/mooring_focus_at_1_3hz_ka.py "
                 "(single-freq mooring + panelretning scatter, ka on x)"),
    data_class="DELEG",
    findings_doc=None,
    fft_window_hz=0.1,
    extra_params=(
        f"freq = {TARGET_FREQ} Hz only. ka per run from "
        f"k({TARGET_FREQ} Hz) × IN Amplitude (FFT) [mm] / 1000. "
        "Three subfigures, one per amplitude tier (A1/A2/A3); within each, "
        "x-axis is ka (natural per-run spread). Mooring: "
        "below_90 (canon, lumping below_90_loose230 + below_90_loose300, "
        "and pooling tidligere + cond4 hardware) gets WIND_COLOR_MAP "
        "(blue uten / red med vind); above_50 gets cyan / bright pink. "
        "Panelretning by marker family: normal = ○ □ △; revers = "
        "6/5/4-point star (matching A1/A2/A3). All markers hollow. "
        "Subfigures stacked vertically (subfig_layout='column') as one "
        "A4 page float."
    ),
    extra_stats={
        "n_runs_total": int(len(sel)),
        "n_below_90":   int((sel["moor_grp"] == "below_90").sum()),
        "n_above_50":   int((sel["moor_grp"] == "above_50").sum()),
        "n_normal":     int((sel["PanelCondition"] == "full").sum()),
        "n_revers":     int((sel["PanelCondition"] == "reverse").sum()),
        "n_per_amp_A1": int((sel["amp_v"] == 0.10).sum()),
        "n_per_amp_A2": int((sel["amp_v"] == 0.20).sum()),
        "n_per_amp_A3": int((sel["amp_v"] == 0.30).sum()),
        "k_const_radm": round(k_const, 4),
    },
)

pu.write_figure_stub(
    _meta_stub,
    plot_type="mooring_focus_ka",
    subfig_filenames=SUBFIG_NAMES,
    subfig_captions=SUBFIG_CAPS,
    subfig_layout="column",
    force=True,
)
print(f"   Combined stub → output/TEXFIGU/{COMBINED_NAME}.tex")

print("\nDone.")
