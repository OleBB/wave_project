"""
Mooring og panelretning ved 1.30 Hz — K_t vs ka
=================================================

Originally framed as "full vs reverse panel" (script and filename still
carry that name for traceability). After running it the dominant visual
story turned out to be the **mooring effect** (canon = below_90 vs
above_50): the canon/above_50 K_t gap is +0.043 nowind / +0.121 fullwind,
while the panel-direction gap on above_50 is < 0.01 either wind. Panel
direction is a *secondary* observation visible only on above_50 — canon
mooring was never run with revers panelretning.

Read this plot as: mooring is primary, panelretning is the within-mooring
sub-comparison.

Same data as `full_vs_reverse_at_1_3hz.py`, restyled with ka on x. ka
clusters the points into 3 amp groups along x; a connecting line through
per-amp means added more clutter than it resolved (4 whole + 2 dashed
lines, all clamped to the same 3 ka values). Removed.

Panel orientation moves into a parallel marker family. The thesis-wide
"normal" panel keeps the standard ○ □ △ amp shapes; "revers panelretning"
gets a distinct star / X family that doesn't echo the standard set.

Naming note: the dataset's `PanelCondition` is `full` / `reverse`; the
reader-facing labels are "normal" (since the panel sits in its physically
designed orientation) and "revers panelretning" (panel rotated). Internal
filter keeps the data-column names.

Encoding:
  x      : ka (per-run, k(1.30 Hz) × IN Amplitude (FFT) [mm] / 1000)
  y      : K_t
  marker : amp tier × panel
             normal panel  → ○ A1, □ A2, △ A3
             revers panel  → ✡ A1, ★ A2, ✕ A3
  hue    : wind (red med, blue uten)
  hue    : mooring × wind (below_90 → blue/red; above_50 → cyan/bright pink)
           — palette synced with `mooring_focus_at_1_3hz_ka.py` so the
           reader can compare the combined and per-amp variants directly
  fill   : hardware (filled = canon cond4_h100_low, hollow = earlier)

Outputs:
    analysis_scratch/full_vs_reverse_at_1_3hz_ka.pdf
    output/FIGURES/ch05_full_vs_reverse_at_1_3hz_ka.pdf
    analysis_scratch/full_vs_reverse_at_1_3hz_ka_summary.csv
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
from matplotlib.ticker import MultipleLocator

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import os
os.chdir(BASE)

from wavescripts.improved_data_loader import load_analysis_data
from wavescripts.constants import PROBE_HEIGHT_DEFAULT_MM
from wavescripts.plot_utils import (
    WIND_COLOR_MAP, amp_to_label, apply_thesis_style, freq_to_k,
    apply_horizontal_ylabel,
)

# Per-(panel, amp) markers. Reader sees panel via shape family
# (round/square/triangle vs 6/5/4-point star), not via linestyle.
# Synced with `mooring_focus_at_1_3hz_ka.py` per-amp variant so the reader
# can compare the combined and per-amp figures directly.
PANEL_AMP_MARKER = {
    ("full",    0.10): "o",
    ("full",    0.20): "s",
    ("full",    0.30): "^",
    ("reverse", 0.10): (6, 1, 0),   # 6-point star
    ("reverse", 0.20): (5, 1, 0),   # 5-point star
    ("reverse", 0.30): (4, 1, 0),   # 4-point star
}
PANEL_LABEL = {"full": "normal", "reverse": "revers"}

apply_thesis_style()

SCRATCH_PDF = Path(__file__).parent / "full_vs_reverse_at_1_3hz_ka.pdf"
SCRATCH_CSV = Path(__file__).parent / "full_vs_reverse_at_1_3hz_ka_summary.csv"
OUT_PDF     = BASE / "output" / "FIGURES" / "ch05_full_vs_reverse_at_1_3hz_ka.pdf"
OUT_PDF.parent.mkdir(parents=True, exist_ok=True)

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

# Per-run ka.
k_const = float(freq_to_k(np.array([TARGET_FREQ]))[0])
sel["a_m"] = sel["IN Amplitude (FFT)"].astype(float) / 1000.0   # mm → m
sel["ka"]  = k_const * sel["a_m"]
sel["amp_v"] = sel["WaveAmplitudeInput [Volt]"].apply(lambda v: round(float(v), 2))
print(f"   {len(sel)} rows; k({TARGET_FREQ} Hz) = {k_const:.3f} rad/m")
print(f"   ka range: [{sel['ka'].min():.3f}, {sel['ka'].max():.3f}]")

# ── 2. Visual constants ────────────────────────────────────────────────────────
WIND_LABEL  = {"no": "uten vind", "full": "med vind"}
MARKER_SIZE = 90        # match per-amp `mooring_focus_at_1_3hz_ka.py`
EDGE_LW     = 1.6
ALPHA       = 0.85

COLOR = {
    ("below_90", "no"):   WIND_COLOR_MAP["no"],     # #1F77B4 — blue
    ("below_90", "full"): WIND_COLOR_MAP["full"],   # #D62728 — red
    ("above_50", "no"):   "#00CED1",                # cyan (DarkTurquoise)
    ("above_50", "full"): "#FF1493",                # bright pink (DeepPink)
}
MOORING_LABEL = {"below_90": "Under", "above_50": "Over"}

# ── 3. Plot ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.5, 5.4))

# Scatter all individual runs. Marker = (panel, amp); colour = (mooring, wind).
# All hollow — matches per-amp `mooring_focus_at_1_3hz_ka.py`. Hardware (canon
# vs earlier) lumped under below_90; the probe-config diagnostics already
# established that the canon vs earlier hardware split inside below_90_loose230
# doesn't carry a statistically defensible bias.
mean_rows = []
for (panel, moor, wind, amp_v), grp in sel.groupby(
    ["PanelCondition", "moor_grp", "WindCondition", "amp_v"]
):
    if grp.empty:
        continue
    color = COLOR[(moor, wind)]
    marker = PANEL_AMP_MARKER.get((panel, amp_v), "X")
    ax.scatter(
        grp["ka"], grp["OUT/IN (FFT)"],
        facecolors="none", edgecolors=color, marker=marker,
        s=MARKER_SIZE, linewidths=EDGE_LW, alpha=ALPHA,
        zorder=3,
    )
    # Bookkeeping (mean stats kept for the CSV, not plotted).
    mean_rows.append(dict(
        panel=panel, moor_grp=moor, wind=wind, amp_v=amp_v,
        n=int(len(grp)),
        ka_mean=float(grp["ka"].mean()),
        Kt_mean=float(grp["OUT/IN (FFT)"].mean()),
        Kt_std=float(grp["OUT/IN (FFT)"].std()) if len(grp) > 1 else None,
    ))

ax.axhline(1.0, color="black", lw=0.6, ls="--", alpha=0.5)
ax.set_xlabel(r"$ka$  (per kjøring; $k(1.30\,\mathrm{Hz})\cdot a_\mathrm{IN}$)",
               fontsize=10)
ax.grid(which="major", alpha=0.30, lw=0.6)
ax.grid(which="minor", alpha=0.15, lw=0.4)
ax.yaxis.set_major_locator(MultipleLocator(0.05))
ax.yaxis.set_minor_locator(MultipleLocator(0.025))
# Horizontal $K_t$ above leftmost tick — matches per-amp variant.
apply_horizontal_ylabel(ax, r"$K_t$", fontsize=12)

# X- and Y-range matched to ch05_damping_ka so the reader can compare scales.
ax.set_xlim(0.045, 0.29)
ax.set_ylim(0.34, 0.91)

# Two-block legend — same structure as per-amp variant. Panel block expanded
# to 6 entries (2 panels × 3 amps) since the combined figure pools all amps.
moor_wind_handles = [
    mlines.Line2D(
        [], [],
        color=COLOR[(m, w)],
        linestyle="-",
        linewidth=2.0,
        marker=None,
        label=f"{MOORING_LABEL[m]} · {WIND_LABEL[w]}",
    )
    for m in ["below_90", "above_50"]
    for w in ["no", "full"]
]

panel_handles = [
    mlines.Line2D([], [], color="black",
                  marker=PANEL_AMP_MARKER[(p, v)],
                  ms=10, lw=0, mfc="none", mec="black",
                  mew=EDGE_LW,
                  label=f"{PANEL_LABEL[p]} · {amp_to_label(v)}")
    for p in ["full", "reverse"]
    for v in [0.10, 0.20, 0.30]
]

leg1 = ax.legend(handles=moor_wind_handles, loc="upper right",
                  fontsize=8, framealpha=0.92,
                  title="Moring · vind", title_fontsize=8)
ax.add_artist(leg1)
ax.legend(handles=panel_handles, loc="lower right",
           fontsize=8.5, framealpha=0.92,
           title="Panelretning", title_fontsize=8.5,
           ncol=2, bbox_to_anchor=(0.995, 0.005))

fig.tight_layout()

SCRATCH_PDF.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(SCRATCH_PDF, bbox_inches="tight", pad_inches=0.02)
print(f"\n   Saved → {SCRATCH_PDF.relative_to(BASE)}")
fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.02)
print(f"   Saved → {OUT_PDF.relative_to(BASE)}")
plt.close(fig)

mean_df = pd.DataFrame(mean_rows)
mean_df.to_csv(SCRATCH_CSV, index=False)
print(f"   Summary → {SCRATCH_CSV.relative_to(BASE)}\n")
print(mean_df.round(4).to_string(index=False))

# ── 4. TEXFIGU stub ───────────────────────────────────────────────────────────
# Caption text comes from FIGURE_CAPTIONS / FIGURE_CAPTIONS_SHORT in
# main_save_figures.py via output/.figure_captions.json.
import wavescripts.plot_utils as pu

THESIS_NAME = "ch05_full_vs_reverse_at_1_3hz_ka"
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
        "plotting": {"figure_name": THESIS_NAME},
    },
    chapter="05",
    extra={"script": "analysis_scratch/full_vs_reverse_at_1_3hz_ka.py"},
    computed_in=("analysis_scratch/full_vs_reverse_at_1_3hz_ka.py "
                 "(combined K_t vs ka scatter at 1.30 Hz, all 3 amps)"),
    data_class="DELEG",
    findings_doc=None,
    fft_window_hz=0.1,
    extra_params=(
        f"freq = {TARGET_FREQ} Hz only. ka per run from "
        f"k({TARGET_FREQ} Hz) × IN Amplitude (FFT) [mm] / 1000 — paddle-only "
        "FFT amplitude (NOT the wind-contaminated `IN ka (FFT)` pipeline column "
        "which mixes FFT wavenumber with time-domain percentile amplitude). "
        "Mooring colours: below_90 (canon, lumping below_90_loose230 + "
        "below_90_loose300, hardware pooled) gets WIND_COLOR_MAP "
        "(blue uten / red med vind); above_50 gets cyan / bright pink. "
        "Panelretning by marker family: normal = ○ □ △; revers = "
        "6/5/4-point star (matching A1/A2/A3). All markers hollow. "
        "x/y limits matched to ch05_damping_ka so the figures stack visually."
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
    plot_type="full_vs_reverse_at_1_3hz_ka",
    subfig_filenames=[THESIS_NAME],
)
out_stub = BASE / "output" / "TEXFIGU" / f"{THESIS_NAME}.tex"
print(f"   Stub → {out_stub.relative_to(BASE)}")

print("\nDone.")
