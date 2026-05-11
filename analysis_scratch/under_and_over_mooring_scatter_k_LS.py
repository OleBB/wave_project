"""
Under- and over-mooring damping scatter — k-axis, LS-amplitude prototype
=========================================================================

Prototype sibling of `under_and_over_mooring_scatter_k.py`. Identical
data subset, filtering, categorization, and visual encoding — the only
change is the K_t source:

  K_t  =  mean( Probe {p} Amplitude (LS) for p in out_probes_used )
          ─────────────────────────────────────────────────────────
          mean( Probe {p} Amplitude (LS) for p in in_probes_used )

i.e. the per-probe LS sinusoid fit at f_paddle (bin-grid-independent),
averaged across the same probes that contribute to the canonical
`IN Amplitude (FFT)` / `OUT Amplitude (FFT)` columns (per row's
`in_probes_used` / `out_probes_used`).

Motivation: at A3 1.4 Hz nowind above_50 the run from 20260313 reads
K_t,FFT = 0.831 — well above its fullwind counterparts at 0.74. The
LS reading is K_t,LS = 0.701, in line with the rest of the field.
Mechanism: 45 cut_samples on Probe 9373/340 destroyed integer-cycle
coherence in the H&G-snapped window and leaked FFT amplitude out of
the paddle bin; the LS fit is robust to such gaps. See memory note.

Output is scratch — PDFs go to analysis_scratch/, no thesis stub, no
output/FIGURES/. If the LS plots look right we can promote them.

Run from repo root:
    conda run -n draumkvedet python analysis_scratch/under_and_over_mooring_scatter_k_LS.py
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
from wavescripts.plot_utils import (
    WIND_COLOR_MAP, amp_to_label, apply_thesis_style,
    apply_horizontal_ylabel, freq_to_k, add_freq_axis,
)

apply_thesis_style()

# ── I/O — scratch only ─────────────────────────────────────────────────────────
OUT_OVER  = Path(__file__).parent / "under_and_over_mooring_scatter_k_LS_over.pdf"
OUT_UNDER = Path(__file__).parent / "under_and_over_mooring_scatter_k_LS_under.pdf"
OUT_CSV   = Path(__file__).parent / "under_and_over_mooring_scatter_k_LS_summary.csv"

K_COL = "IN Wavenumber (FFT)"
A_COL = "IN Amplitude (FFT)"   # only used to compute ka for axis envelope

# ── 1. Load everything ─────────────────────────────────────────────────────────
print("1. Loading all processed folders …")
all_dirs = sorted(glob.glob(str(BASE / "waveprocessed" / "PROCESSED-*")))
print(f"   {len(all_dirs)} folders")
meta, _, _, _ = load_analysis_data(*all_dirs, load_processed=False)
print(f"   {len(meta)} total rows")

# ── 2. Filter (identical to the FFT sibling) ──────────────────────────────────
wave = meta[
    meta["WaveFrequencyInput [Hz]"].notna()
    & (meta["WaveFrequencyInput [Hz]"] > 0)
    & meta["PanelCondition"].isin(["full", "reverse"])
    & meta["WindCondition"].isin(["no", "full"])
    & (meta["quality_flag"] == "ok")
    & meta["OUT/IN (FFT)"].notna()
    & meta[K_COL].notna()
    & meta[A_COL].notna()
    & meta["in_probes_used"].notna()
    & meta["out_probes_used"].notna()
].copy()

wave_clip = wave[(wave["OUT/IN (FFT)"] <= 2.0) & (wave["OUT/IN (FFT)"] >= 0.1)].copy()
wave_clip = wave_clip[wave_clip["WaveFrequencyInput [Hz]"] < 2.0].copy()
wave_clip = wave_clip[wave_clip["Mooring"] != "above_200"].copy()
print(f"   {len(wave_clip)} runs after the standard filter")

# ── 3. Compute K_t from LS amplitudes (canonical mean of in/out probes) ───────
def _compute_kt_ls(row):
    in_probes  = [p.strip() for p in str(row["in_probes_used"]).split("+")]
    out_probes = [p.strip() for p in str(row["out_probes_used"]).split("+")]
    try:
        a_in_vals  = [row[f"Probe {p} Amplitude (LS)"] for p in in_probes]
        a_out_vals = [row[f"Probe {p} Amplitude (LS)"] for p in out_probes]
        a_in  = float(np.nanmean(a_in_vals))
        a_out = float(np.nanmean(a_out_vals))
        if a_in <= 0 or not np.isfinite(a_in) or not np.isfinite(a_out):
            return np.nan
        return a_out / a_in
    except KeyError:
        return np.nan

wave_clip["K_t_LS"] = wave_clip.apply(_compute_kt_ls, axis=1)
n_missing_ls = int(wave_clip["K_t_LS"].isna().sum())
if n_missing_ls:
    print(f"   {n_missing_ls} runs missing LS amplitude — dropped from LS view")
    wave_clip = wave_clip[wave_clip["K_t_LS"].notna()].copy()
print(f"   {len(wave_clip)} runs have a valid K_t_LS")

# Sanity: clip outliers the same way FFT was clipped, but only for plotting.
n_extreme_ls = ((wave_clip["K_t_LS"] > 2.0) | (wave_clip["K_t_LS"] < 0.1)).sum()
wave_clip = wave_clip[(wave_clip["K_t_LS"] <= 2.0) & (wave_clip["K_t_LS"] >= 0.1)].copy()
print(f"   {len(wave_clip)} after K_t_LS clip ({n_extreme_ls} extreme)")

# x-axis: k from dispersion (same as FFT sibling).
wave_clip["k"] = freq_to_k(wave_clip["WaveFrequencyInput [Hz]"].values)
wave_clip["ka"] = (wave_clip[K_COL].astype(float)
                   * wave_clip[A_COL].astype(float) / 1000.0)

# ── 4. Category (mooring × panel) — identical to the FFT sibling ──────────────
def _category(row):
    m = row["Mooring"]; p = row["PanelCondition"]
    if m == "below_90_loose300" and p == "full":     return "below_loose300_full"
    if m == "below_90_loose230" and p == "full":     return "below_loose230_full"
    if m == "above_50"          and p == "full":     return "above_50_full"
    if m == "above_50"          and p == "reverse":  return "above_50_reverse"
    return "other"

wave_clip["category"] = wave_clip.apply(_category, axis=1)
wave_clip = wave_clip[wave_clip["category"] != "other"].copy()

print("\n2. Counts per category × wind:")
print(wave_clip.groupby(["category", "WindCondition"]).size().unstack(fill_value=0).to_string())
print(f"   total: {len(wave_clip)} runs")

# ── 5. Direct FFT-vs-LS comparison (focused diagnostic) ───────────────────────
print("\n3. K_t difference FFT vs LS — top 15 by |ΔK_t|:")
wave_clip["dKt_ls_minus_fft"] = wave_clip["K_t_LS"] - wave_clip["OUT/IN (FFT)"]
top = wave_clip.assign(absd=wave_clip["dKt_ls_minus_fft"].abs()) \
                .nlargest(15, "absd")
cols_show = ["WaveFrequencyInput [Hz]", "WaveAmplitudeInput [Volt]",
             "WindCondition", "Mooring", "PanelCondition",
             "OUT/IN (FFT)", "K_t_LS", "dKt_ls_minus_fft"]
print(top[cols_show + ["path"]].to_string())

summary = (wave_clip.groupby(["category", "WindCondition"])
                     .agg(n=("K_t_LS", "count"),
                          kt_fft_mean=("OUT/IN (FFT)", "mean"),
                          kt_ls_mean=("K_t_LS", "mean"),
                          kt_fft_std=("OUT/IN (FFT)", "std"),
                          kt_ls_std=("K_t_LS", "std"))
                     .reset_index())
print("\n4. Summary (mean K_t and std per category × wind):")
print(summary.to_string(index=False))
summary.to_csv(OUT_CSV, index=False)
print(f"\n   Summary → {OUT_CSV.relative_to(BASE)}")

# ── 6. Encoding (identical to the FFT sibling) ────────────────────────────────
ABOVE_FULLWIND_COLOR = "#B22222"
ABOVE_NOWIND_COLOR   = "#4682B4"

COLORS = {
    ("below_loose300_full", "no"):   WIND_COLOR_MAP["no"],
    ("below_loose300_full", "full"): WIND_COLOR_MAP["full"],
    ("below_loose230_full", "no"):   "#9ECAE1",
    ("below_loose230_full", "full"): "#F4815A",
    ("above_50_full",       "no"):   ABOVE_NOWIND_COLOR,
    ("above_50_full",       "full"): ABOVE_FULLWIND_COLOR,
    ("above_50_reverse",    "no"):   ABOVE_NOWIND_COLOR,
    ("above_50_reverse",    "full"): ABOVE_FULLWIND_COLOR,
}
HOLLOW_CATEGORIES = {"above_50_reverse"}
CATEGORY_LABELS = {
    "below_loose300_full": "Under, 30 cm",
    "below_loose230_full": "Under, 23 cm",
    "above_50_full":       "Over",
    "above_50_reverse":    "Over, revers",
}
CATEGORY_ORDER = [
    "above_50_full", "above_50_reverse",
    "below_loose230_full", "below_loose300_full",
]
MARKERS = {
    "below_loose300_full": {0.10: "o",  0.20: "s",  0.30: "^"},
    "below_loose230_full": {0.10: "o",  0.20: "s",  0.30: "^"},
    "above_50_full":       {0.10: "D",  0.20: "P",  0.30: "h"},
    "above_50_reverse":    {0.10: "D",  0.20: "P",  0.30: "h"},
}
MARKER_SIZE = 55
ALPHA = 0.75
EDGE_LW = 0.4

def _round_amp(v): return round(float(v), 2)

# Shared envelope.
_k_min = float(wave_clip["k"].min())
_k_max = float(wave_clip["k"].max())
_x_pad = max(0.15, 0.02 * (_k_max - _k_min))
XLIM = (_k_min - _x_pad, _k_max * 1.02)
YLIM = (0.1, 1.05)

THESIS_K_LO = float(freq_to_k(np.array([1.3]))[0])
THESIS_K_HI = float(freq_to_k(np.array([1.6]))[0])

# ── 7. Plot helper — identical to the FFT sibling, but y is K_t_LS ────────────
def _make_view(sub: pd.DataFrame, *,
               categories: list[str],
               out_pdf: Path,
               view_label: str,
               draw_fits: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(6.27, 9.69))
    plot_order = [c for c in CATEGORY_ORDER if c in categories]
    for cat in plot_order:
        sub_cat = sub[sub["category"] == cat]
        if sub_cat.empty:
            continue
        for wind in ("no", "full"):
            for amp_v in (0.10, 0.20, 0.30):
                s = sub_cat[(sub_cat["WindCondition"] == wind)
                            & (sub_cat["WaveAmplitudeInput [Volt]"].apply(_round_amp) == amp_v)]
                if s.empty: continue
                color  = COLORS.get((cat, wind), "gray")
                marker = MARKERS[cat][amp_v]
                sz = MARKER_SIZE * (1.6 if cat.startswith("above_50") else 1.0)
                hollow = cat in HOLLOW_CATEGORIES
                face_color = "none" if hollow else color
                edge_color = color  if hollow else "black"
                edge_lw    = 1.1    if hollow else EDGE_LW
                ax.scatter(
                    s["k"], s["K_t_LS"],
                    facecolors=face_color, edgecolors=edge_color,
                    marker=marker, s=sz, linewidths=edge_lw, alpha=ALPHA,
                    zorder=3 if cat == "below_loose300_full" else 2,
                )
                if draw_fits:
                    k_cell = s["k"].to_numpy(float)
                    kt_cell = s["K_t_LS"].to_numpy(float)
                    if len(k_cell) >= 2 and k_cell.std() > 1e-9:
                        p = np.polyfit(k_cell, kt_cell, deg=1)
                        x_line = np.array([k_cell.min(), k_cell.max()])
                        y_line = np.polyval(p, x_line)
                        ax.plot(
                            x_line, y_line, color=color, lw=0.6, alpha=0.55,
                            linestyle=(0, (3, 1.5)) if hollow else "-",
                            zorder=4 if cat == "below_loose300_full" else 3,
                            solid_capstyle="round",
                        )

    ax.axvspan(THESIS_K_LO, THESIS_K_HI,
               color=WIND_COLOR_MAP["no"], alpha=0.07, lw=0, zorder=1)
    _ann_y = YLIM[1] - 0.04
    ax.text(THESIS_K_HI - 0.05, _ann_y,
            "Hovedfokus\n1,3–1,6 Hz", ha="right", va="top",
            fontsize=8, color="#1F618D", alpha=0.85,
            bbox=dict(boxstyle="round,pad=0.2",
                      facecolor="white", alpha=0.75, edgecolor="none"))

    ax.axhline(1.0, color="black", lw=0.6, ls="--", alpha=0.5)
    ax.set_xlabel("$k$", fontsize=11)
    ax.set_ylabel("")
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.grid(which="major", alpha=0.30, lw=0.6)
    ax.grid(which="minor", alpha=0.15, lw=0.4)
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)

    secax = add_freq_axis(ax)
    secax.set_xlabel(r"Frekvens (Hz)", fontsize=9)
    _used_freqs = sorted(sub[sub["category"].isin(categories)]
                         ["WaveFrequencyInput [Hz]"].unique())
    secax.set_xticks(_used_freqs)
    secax.set_xticklabels([f"{f:.1f}" for f in _used_freqs])
    secax.tick_params(labelsize=7)

    # Legends.
    config_handles = []
    for cat in plot_order[::-1]:
        is_above = cat.startswith("above_50")
        cat_marker_exemplar = "D" if is_above else "o"
        cat_msize = 11 if is_above else 9
        hollow = cat in HOLLOW_CATEGORIES
        for wind, wlabel in (("full", "full vind"), ("no", "uten vind")):
            wind_color = COLORS[(cat, wind)]
            mfc = "none"      if hollow else wind_color
            mec = wind_color  if hollow else "black"
            mew = 1.1         if hollow else 0.4
            config_handles.append(
                mlines.Line2D([], [],
                              marker=cat_marker_exemplar, linestyle="None",
                              markerfacecolor=mfc, markeredgecolor=mec,
                              markeredgewidth=mew, markersize=cat_msize,
                              label=f"{CATEGORY_LABELS[cat]}, {wlabel}")
            )
    amp_handles = []
    has_below = any(c.startswith("below_") for c in plot_order)
    has_above = any(c.startswith("above_50") for c in plot_order)
    for v in (0.10, 0.20, 0.30):
        if has_below:
            amp_handles.append(
                mlines.Line2D([], [], color="black",
                              marker=MARKERS["below_loose300_full"][v], linestyle="None",
                              markersize=8, markerfacecolor="lightgray",
                              markeredgecolor="black", markeredgewidth=0.3,
                              label=f"{amp_to_label(v)}{'  (under)' if has_above else ''}"))
        if has_above:
            amp_handles.append(
                mlines.Line2D([], [], color="black",
                              marker=MARKERS["above_50_full"][v], linestyle="None",
                              markersize=10, markerfacecolor="lightgray",
                              markeredgecolor="black", markeredgewidth=0.3,
                              label=f"{amp_to_label(v)}{'  (over)' if has_below else ''}"))
    amp_legend_ncol = 2 if (has_below and has_above) else 1

    leg_cfg = ax.legend(handles=config_handles, loc="upper right",
                        bbox_to_anchor=(0.995, 0.995),
                        fontsize=7.5, framealpha=0.92,
                        title="Konfigurasjon (LS)", title_fontsize=8,
                        ncol=2 if len(config_handles) >= 4 else 1)
    ax.add_artist(leg_cfg)
    fig.canvas.draw()
    leg_cfg_bbox = leg_cfg.get_window_extent().transformed(ax.transAxes.inverted())
    amp_anchor_y = leg_cfg_bbox.y0 - 0.010
    ax.legend(handles=amp_handles, loc="upper right",
              bbox_to_anchor=(0.995, amp_anchor_y),
              fontsize=7, framealpha=0.92,
              title="Amplitude", title_fontsize=8, ncol=amp_legend_ncol)

    sub_view = sub[sub["category"].isin(categories)]
    n_total = len(sub_view)
    print(f"\n   {view_label}:  n = {n_total} kjøringer (LS)")

    fig.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.06)
    apply_horizontal_ylabel(ax, r"$K_t^{\,\mathrm{(LS)}}$", fontsize=12)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    print(f"   Saved → {out_pdf.relative_to(BASE)}")
    plt.close(fig)


# ── 8. Build the two views ────────────────────────────────────────────────────
print("\n5. Building views …")
_make_view(wave_clip,
           categories=["above_50_full", "above_50_reverse"],
           out_pdf=OUT_OVER,
           view_label="overmooring (LS)")
_make_view(wave_clip,
           categories=["below_loose300_full", "below_loose230_full"],
           out_pdf=OUT_UNDER,
           view_label="undermooring (LS)")

print("\nDone.")
