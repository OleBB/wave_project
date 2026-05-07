"""
Mooring + panelretning + frekvens at 1.3–1.6 Hz — K_t vs ka, per amplitude
==========================================================================

Combines the data scope of two existing per-amp ka figures:

  - `ch05_mooring_focus_at_1_3hz_ka_{A1,A2,A3}` — 1.30 Hz only,
    mooring ∈ {below_90, above_50}, panel ∈ {full, reverse}.
  - `ch05_damping_ka_fit_{A1,A2,A3}` — 1.3–1.6 Hz, canon below_90 only,
    panel = full only, per_tag ∈ {per240, per40}.

The new figure (per amp, A1/A2/A3) shows BOTH datasets overlaid: at
1.30 Hz the mooring/panel cross is fully populated; at 1.4–1.6 Hz only
the canon-lowrange below_90 + full-panel cluster is present (since
revers panel and above_50 mooring were never run above 1.30 Hz).

x-axis = paddle-only ka per run = `IN Wavenumber (FFT) × IN Amplitude
(FFT) [m]`. Both factors are FFT-measured, paddle-tone-only — explicitly
NOT the pipeline `IN ka (FFT)` column (which mixes FFT wavenumber with
time-domain percentile amplitude and is wind-contaminated; see
CLAUDE.md §16).

Encoding (per subfigure):
  hue   : mooring × wind — below_90 → blue/red (WIND_COLOR_MAP);
                            above_50 → cyan / bright pink
          (matches `ch05_mooring_focus_at_1_3hz_ka_*`).
  shape : panel × freq —
            normal panel (full):    `_freq_marker(amp_v, freq_idx)` per freq
                                    (○/wedge cycle for A1, rotated rect for
                                    A2, oriented triangle for A3 — same as
                                    `ch05_damping_ka_fit_*`).
            revers panel (reverse): 6/5/4-point star for A1/A2/A3 (only
                                    1.30 Hz exists — same as
                                    `ch05_mooring_focus_at_1_3hz_ka_*`).
  fill  : all hollow (overlapping points readable).

Outputs:
    analysis_scratch/mooring_panel_freq_ka_{A1,A2,A3}.pdf       (scratch)
    analysis_scratch/mooring_panel_freq_ka_summary.csv
    output/FIGURES/ch05_mooring_panel_freq_ka_{A1,A2,A3}.pdf    (thesis)
    output/TEXFIGU/ch05_mooring_panel_freq_ka.tex               (combined stub,
                                                                 3 stacked subfigs)
"""

import sys
import re
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
    WIND_COLOR_MAP, apply_thesis_style, apply_horizontal_ylabel,
)
from wavescripts.plotter import _freq_marker
import wavescripts.plot_utils as pu

apply_thesis_style()

# ── 1. Load all processed folders ──────────────────────────────────────────────
print("1. Loading processed folders …")
all_dirs = sorted(glob.glob(str(BASE / "waveprocessed" / "PROCESSED-*")))
meta, _, _, _ = load_analysis_data(*all_dirs, load_processed=False)
print(f"   {len(meta)} total rows from {len(all_dirs)} folders")

# ── 2. Build the union dataset ─────────────────────────────────────────────────
# Tag mooring group (below_90 lumps loose230 + loose300, matches
# mooring_focus_at_1_3hz_ka.py).
def mooring_group(m):
    if m in ("below_90_loose230", "below_90_loose300"):
        return "below_90"
    if m == "above_50":
        return "above_50"
    return "other"
meta["moor_grp"] = meta["Mooring"].apply(mooring_group)

# Tag per_tag (matches damping_ka_per_volt_with_fit.py).
_per40  = re.compile(r"per40(?!\d)")
_per240 = re.compile(r"per240")
meta["per_tag"] = np.where(meta["path"].str.contains(_per40,  na=False), "per40",
                    np.where(meta["path"].str.contains(_per240, na=False), "per240",
                             "other"))

# Common filter: panels, wind, moorings, quality, FFT validity.
common_mask = (
    meta["PanelCondition"].isin(["full", "reverse"])
    & (meta["quality_flag"] == "ok")
    & meta["OUT/IN (FFT)"].notna()
    & meta["IN Amplitude (FFT)"].notna()
    & meta["IN Wavenumber (FFT)"].notna()
    & meta["WindCondition"].isin(["no", "full"])
    & meta["OUT/IN (FFT)"].between(0.1, 2.0)
    & meta["moor_grp"].isin(["below_90", "above_50"])
)

# A. Broad 1.30 Hz scope (mooring_focus_at_1_3hz_ka).
mask_A = common_mask & (meta["WaveFrequencyInput [Hz]"] == 1.30)

# B. Narrow 1.4-1.6 Hz scope (damping_ka_fit): canon-lowrange folders only,
# panel=full only, per240+per40 only.
CANON_LOWRANGE_HINTS = ("20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange",
                        "20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange")
canon_path_mask = meta["path"].str.contains("|".join(re.escape(h) for h in CANON_LOWRANGE_HINTS),
                                             na=False, regex=True)
mask_B = (
    common_mask
    & meta["WaveFrequencyInput [Hz]"].isin([1.40, 1.50, 1.60])
    & (meta["PanelCondition"] == "full")
    & canon_path_mask
    & meta["per_tag"].isin(["per240", "per40"])
)

sel = meta[mask_A | mask_B].copy()
sel["amp_v"] = sel["WaveAmplitudeInput [Volt]"].apply(lambda v: round(float(v), 2))

# Paddle-only ka: FFT wavenumber × FFT amplitude (mm → m).
sel["ka"] = (sel["IN Wavenumber (FFT)"].astype(float)
             * sel["IN Amplitude (FFT)"].astype(float) / 1000.0)

print(f"\n2. Scope union: {len(sel)} rows total")
print(f"   - 1.30 Hz (mooring_focus scope): {int(mask_A.sum())}")
print(f"   - 1.4-1.6 Hz (damping_ka_fit canon scope): {int(mask_B.sum())}")
print(f"   ka range: [{sel['ka'].min():.3f}, {sel['ka'].max():.3f}]")
print(f"   K_t range: [{sel['OUT/IN (FFT)'].min():.3f}, {sel['OUT/IN (FFT)'].max():.3f}]")

print("\n   Counts per (freq, amp, panel, mooring, wind):")
print(sel.groupby(["WaveFrequencyInput [Hz]", "amp_v", "PanelCondition",
                    "moor_grp", "WindCondition"]).size().to_string())

# ── 3. Visual constants ────────────────────────────────────────────────────────
COLOR = {
    ("below_90", "no"):   WIND_COLOR_MAP["no"],     # blue
    ("below_90", "full"): WIND_COLOR_MAP["full"],   # red
    ("above_50", "no"):   "#00CED1",                # cyan
    ("above_50", "full"): "#FF1493",                # pink
}
MOORING_LABEL = {"below_90": "Under", "above_50": "Over"}
WIND_LABEL    = {"no": "uten vind", "full": "med vind"}
PANEL_LABEL   = {"full": "normal", "reverse": "revers"}

THESIS_FREQS = [1.30, 1.40, 1.50, 1.60]
FREQ_IDX     = {f: i for i, f in enumerate(THESIS_FREQS)}

# Reverse-panel marker family — only used at 1.30 Hz (no revers data above).
REVERSE_MARKER = {0.10: (6, 1, 0), 0.20: (5, 1, 0), 0.30: (4, 1, 0)}

MARKER_SIZE = 90
EDGE_LW     = 1.6
ALPHA       = 0.85

AMP_TIERS = [(0.10, "A1"), (0.20, "A2"), (0.30, "A3")]

# Shared y range across the 3 figures so they stack comparably.
y_lo = sel["OUT/IN (FFT)"].min() - 0.02
y_hi = sel["OUT/IN (FFT)"].max() + 0.02
print(f"\n   shared y-range: [{y_lo:.3f}, {y_hi:.3f}]")

# Shared x range (over the full union).
x_lo = sel["ka"].min() - 0.005
x_hi = sel["ka"].max() + 0.005
print(f"   shared x-range: [{x_lo:.3f}, {x_hi:.3f}]")

# ── 4. Plot — one figure per amp tier ──────────────────────────────────────────
summary_rows = []

for amp_v, amp_tag in AMP_TIERS:
    sub = sel[sel["amp_v"] == amp_v]
    if sub.empty:
        print(f"   [{amp_tag}] no rows — skipping")
        continue

    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    # Loop ordered for consistent legend / draw order.
    for (panel, moor, wind, freq), grp in sub.groupby(
        ["PanelCondition", "moor_grp", "WindCondition", "WaveFrequencyInput [Hz]"]
    ):
        if grp.empty:
            continue
        color  = COLOR[(moor, wind)]
        if panel == "full":
            fi = FREQ_IDX.get(round(float(freq), 2))
            if fi is None:
                continue
            marker = _freq_marker(amp_v, fi)
        else:
            marker = REVERSE_MARKER[amp_v]

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
            freq=float(freq),
            n=int(len(grp)),
            ka_mean=float(grp["ka"].mean()),
            Kt_mean=float(grp["OUT/IN (FFT)"].mean()),
            Kt_std=float(grp["OUT/IN (FFT)"].std()) if len(grp) > 1 else None,
        ))

    ax.axhline(1.0, color="black", lw=0.6, ls="--", alpha=0.5)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel(r"$ka$  (per kjøring; "
                   r"$k_\mathrm{FFT}\cdot a_\mathrm{IN,\,FFT}$)",
                   fontsize=10)
    ax.grid(which="major", alpha=0.30, lw=0.6)
    ax.grid(which="minor", alpha=0.15, lw=0.4)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(MultipleLocator(0.025))
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.xaxis.set_minor_locator(MultipleLocator(0.01))
    apply_horizontal_ylabel(ax, r"$K_t$", fontsize=12)

    # Legend block 1: Moring × vind (4 coloured swatches).
    moor_wind_handles = [
        mlines.Line2D(
            [], [],
            color=COLOR[(m, w)],
            linestyle="-", linewidth=2.0, marker=None,
            label=f"{MOORING_LABEL[m]} · {WIND_LABEL[w]}",
        )
        for m in ["below_90", "above_50"]
        for w in ["no", "full"]
    ]

    # Legend block 2: Panel × frekvens.
    # 4 freq markers for "normal" (full panel) at this amp + 1 star for "revers"
    # (only 1.30 Hz). Even if the data has fewer freqs at this amp, show the
    # full marker family for legibility (matches damping_ka_fit's behaviour).
    _legend_path_effects = [
        pe.Stroke(linewidth=EDGE_LW + 1.0, foreground="black"),
        pe.Normal(),
    ]
    panel_freq_handles = [
        mlines.Line2D([], [], color="black",
                      marker=_freq_marker(amp_v, FREQ_IDX[f]),
                      ms=10, lw=0, mfc="none", mec="black",
                      mew=EDGE_LW,
                      label=f"{PANEL_LABEL['full']} · {f:.1f} Hz",
                      path_effects=_legend_path_effects)
        for f in THESIS_FREQS
    ]
    panel_freq_handles.append(
        mlines.Line2D([], [], color="black",
                      marker=REVERSE_MARKER[amp_v],
                      ms=10, lw=0, mfc="none", mec="black",
                      mew=EDGE_LW,
                      label=f"{PANEL_LABEL['reverse']} · 1.3 Hz",
                      path_effects=_legend_path_effects)
    )

    leg1 = ax.legend(handles=moor_wind_handles, loc="upper right",
                      fontsize=8, framealpha=0.92,
                      title="Moring · vind", title_fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=panel_freq_handles, loc="lower right",
               fontsize=8, framealpha=0.92,
               title="Panel · frekvens", title_fontsize=8,
               bbox_to_anchor=(0.995, 0.005))

    fig.tight_layout()
    scratch = (Path(__file__).parent
                / f"mooring_panel_freq_ka_{amp_tag}.pdf")
    out_pdf = (BASE / "output" / "FIGURES"
                / f"ch05_mooring_panel_freq_ka_{amp_tag}.pdf")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(scratch, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    print(f"   [{amp_tag}] Saved → {scratch.relative_to(BASE)}")
    print(f"   [{amp_tag}] Saved → {out_pdf.relative_to(BASE)}")
    plt.close(fig)

# Summary CSV.
csv_path = (Path(__file__).parent / "mooring_panel_freq_ka_summary.csv")
pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
print(f"\n   Summary → {csv_path.relative_to(BASE)}")

# ── 5. Combined TEXFIGU stub ──────────────────────────────────────────────────
COMBINED_NAME = "ch05_mooring_panel_freq_ka"
SUBFIG_NAMES  = [f"{COMBINED_NAME}_{tag}"
                  for amp_v, tag in AMP_TIERS
                  if (sel["amp_v"] == amp_v).any()]
SUBFIG_CAPS   = [f"{tag}" for amp_v, tag in AMP_TIERS
                  if (sel["amp_v"] == amp_v).any()]

pu.ACTIVE_DATASETS = [Path(d).name for d in all_dirs]

_meta_stub = pu.build_fig_meta(
    {
        "filters": {
            "PanelCondition":            "full, reverse",
            "WaveFrequencyInput [Hz]":   "1.30, 1.40, 1.50, 1.60",
            "WaveAmplitudeInput [Volt]": "0.10, 0.20, 0.30",
            "WindCondition":             "no, full",
            "Mooring":                   "below_90_loose230, below_90_loose300, above_50",
            "quality_flag":              "ok",
        },
        "plotting": {"figure_name": COMBINED_NAME},
    },
    chapter="05",
    extra={"script": "analysis_scratch/mooring_panel_freq_ka.py"},
    computed_in=("analysis_scratch/mooring_panel_freq_ka.py "
                 "(mooring_focus + damping_ka_fit data scopes overlaid)"),
    data_class="DELEG",
    findings_doc=None,
    fft_window_hz=0.1,
    extra_params=(
        "Union of two scopes: (A) 1.30 Hz, all folders, panel ∈ {full, reverse}, "
        "mooring ∈ {below_90 (loose230+loose300 lumped), above_50}; "
        "(B) 1.4–1.6 Hz, only canon-lowrange folders "
        "(20260326 + 20260327), panel = full, per_tag ∈ {per240, per40}, "
        "mooring = below_90. "
        "x = paddle-only ka per run = `IN Wavenumber (FFT)` × "
        "`IN Amplitude (FFT)` [m] — both factors FFT-measured, paddle-tone-only, "
        "NOT the wind-contaminated `IN ka (FFT)` pipeline column. "
        "Encoding: mooring × wind → colour (below_90 → WIND_COLOR_MAP, "
        "above_50 → cyan/pink); panel × freq → marker (normal panel uses "
        "`_freq_marker(amp, freq_idx)` cycling per freq; revers panel uses "
        "6/5/4-point star at 1.30 Hz only). All hollow markers."
    ),
    extra_stats={
        "n_runs_total":      int(len(sel)),
        "n_at_1_30hz":       int(mask_A.sum()),
        "n_at_1_4_1_6hz":    int(mask_B.sum()),
        "n_below_90":        int((sel["moor_grp"] == "below_90").sum()),
        "n_above_50":        int((sel["moor_grp"] == "above_50").sum()),
        "n_normal":          int((sel["PanelCondition"] == "full").sum()),
        "n_revers":          int((sel["PanelCondition"] == "reverse").sum()),
        "n_per_amp_A1":      int((sel["amp_v"] == 0.10).sum()),
        "n_per_amp_A2":      int((sel["amp_v"] == 0.20).sum()),
        "n_per_amp_A3":      int((sel["amp_v"] == 0.30).sum()),
    },
)

pu.write_figure_stub(
    _meta_stub,
    plot_type="mooring_panel_freq_ka",
    subfig_filenames=SUBFIG_NAMES,
    subfig_captions=SUBFIG_CAPS,
    subfig_layout="column",
    force=True,
)
print(f"   Combined stub → output/TEXFIGU/{COMBINED_NAME}.tex")

print("\nDone.")
