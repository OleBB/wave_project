"""
Under- and over-mooring damping scatter — k-axis siblings
==========================================================

k-axis companions to the existing
`analysis_scratch/all_data_damping_scatter_ka.py` under/over views.
Mirrors that script's data subset and (mooring × panel) categorization
exactly; only the x-axis differs:

  x-axis = `k` = freq_to_k(WaveFrequencyInput [Hz])  [rad/m]

The parent k-axis figure (`all_data_damping_scatter.py`) uses a
different, hardware-based categorization and is intentionally NOT
modified here — the user keeps that figure as-is. This script
produces only the under and over views, on the k-axis, with the
mooring×panel category scheme.

Differences from the ka-axis sibling:
  - x-axis is k (rad/m), not ka.
  - The top frequency axis is added (`add_freq_axis`), with ticks
    at every used frequency.
  - The thesis-scope band 1.3–1.6 Hz is shown as a light-blue
    `axvspan` with a small "Hovedfokus 1,3–1,6 Hz" annotation, same
    convention as `all_data_damping_scatter.py`.
  - Only two views are produced (under, over); there is no "all"
    k-axis view from this script (the parent figure already covers
    that visual slot via a different lens).
  - Shared XLIM/YLIM computed from the full filtered subset so the
    two views are directly comparable.

Run from repo root:
    conda run -n draumkvedet python analysis_scratch/under_and_over_mooring_scatter_k.py

Outputs:
    output/FIGURES/ch05_damping_undermooring_scatter.pdf
    output/FIGURES/ch05_damping_overmooring_scatter.pdf
    output/TEXFIGU/ch05_damping_undermooring_scatter.tex
    output/TEXFIGU/ch05_damping_overmooring_scatter.tex
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

# ── K_t SOURCE: FFT canonical, with three-way LS+PSD fallback ─────────────────
# IMPORTANT — read before changing anything that touches y-values.
#
# K_t plotted on this figure is NOT a flat `OUT/IN (FFT)` read. Per row we
# compute the canonical FFT K_t and ALSO the LS and PSD K_t (using each
# row's `in_probes_used` / `out_probes_used` to know which probes to
# average), then apply a per-row override:
#
#     if |Kt_FFT − Kt_LS| > 0.05  AND  |Kt_FFT − Kt_PSD| > 0.05
#                                AND  |Kt_LS  − Kt_PSD| < 0.05:
#         use Kt_LS         # FFT is the odd one out; LS+PSD agree → trust them
#     else:
#         use Kt_FFT        # canonical
#
# Why: at A3 1.4 Hz nowind above_50 the mar13 run reads Kt_FFT = 0.831
# (an outlier above its fullwind counterparts at ~0.74); Kt_LS = 0.701
# and Kt_PSD = 0.715 agree perfectly and put it cleanly in the field.
# Mechanism: 45 `cut_samples` on probe 9373/340 broke integer-cycle
# coherence in the H&G-snapped FFT window. LS (sinusoid fit at f_paddle)
# and PSD (variance integrated over ±0.1 Hz) are robust to that. See
# the diagnostic memo and CLAUDE.md §6 / §17.
#
# Asymmetric in FFT's favour: ONE substitute method disagreeing isn't
# enough — we require both LS and PSD to disagree with FFT AND to agree
# with each other. Threshold 0.05 sits well above measurement noise:
# scanning all 399 in-scope runs the rule fires on exactly 3 (mar13 1.4Hz
# A3 above_50, mar7 1.1Hz A2 above_50, mar19 1.5Hz A1 loose230), with the
# next-nearest non-trigger at |Kt_FFT − Kt_LS| ≤ 0.03. None of the
# three rescued runs is in the canon thesis-band scope (1.3–1.6 Hz, h100
# lowrange) — they only show up in this supplementary scatter, which
# spans the broader cross-condition record.
#
# Symbol shown on plot is still `K_t` (method-independent). The override
# count and per-row triggers are printed at runtime and recorded in the
# figure-stub provenance block. To go back to pure FFT, set
# OVERRIDE_THRESHOLD = float("inf") below — the rule will then never fire.
#
# Sibling: analysis_scratch/all_data_damping_scatter_ka.py implements the
# identical rule (ka-axis view of the same data subset). Keep the two in
# sync if you change anything here.
OVERRIDE_THRESHOLD = 0.05

# ── I/O ────────────────────────────────────────────────────────────────────────
CHAPTER = "05"

K_COL = "IN Wavenumber (FFT)"
A_COL = "IN Amplitude (FFT)"

# ── 1. Load everything ─────────────────────────────────────────────────────────
print("1. Loading all processed folders …")
all_dirs = sorted(glob.glob(str(BASE / "waveprocessed" / "PROCESSED-*")))
print(f"   {len(all_dirs)} folders")
meta, _, _, _ = load_analysis_data(*all_dirs, load_processed=False)
print(f"   {len(meta)} total rows")

# ── 2. Filter & categorize by (mooring × panel) ───────────────────────────────
# Same gate as analysis_scratch/all_data_damping_scatter_ka.py.
wave = meta[
    meta["WaveFrequencyInput [Hz]"].notna()
    & (meta["WaveFrequencyInput [Hz]"] > 0)
    & meta["PanelCondition"].isin(["full", "reverse"])
    & meta["WindCondition"].isin(["no", "full"])
    & (meta["quality_flag"] == "ok")
    & meta["OUT/IN (FFT)"].notna()
    & meta[K_COL].notna()
    & meta[A_COL].notna()
    # in/out_probes_used drive the per-method K_t recomputation below.
    # If they're missing on a row, we can't apply the three-way override
    # rule, so drop the row from this figure's scope.
    & meta["in_probes_used"].notna()
    & meta["out_probes_used"].notna()
].copy()

n_extreme = ((wave["OUT/IN (FFT)"] > 2.0) | (wave["OUT/IN (FFT)"] < 0.1)).sum()
wave_clip = wave[(wave["OUT/IN (FFT)"] <= 2.0) & (wave["OUT/IN (FFT)"] >= 0.1)].copy()
print(f"   {len(wave)} wave/quality=ok runs in scope ({n_extreme} extreme outliers clipped)")

n_drop_2hz = int((wave_clip["WaveFrequencyInput [Hz]"] >= 2.0).sum())
wave_clip = wave_clip[wave_clip["WaveFrequencyInput [Hz]"] < 2.0].copy()
print(f"   {len(wave_clip)} after dropping {n_drop_2hz} run(s) at f >= 2.0 Hz")

# Drop the 3 oddball above_200 mooring runs — not in any thesis category.
n_above200 = int((wave_clip["Mooring"] == "above_200").sum())
wave_clip = wave_clip[wave_clip["Mooring"] != "above_200"].copy()
if n_above200:
    print(f"   dropped {n_above200} runs at Mooring=above_200")

# x-axis: k from dispersion (same as the parent k-axis figure).
wave_clip["k"] = freq_to_k(wave_clip["WaveFrequencyInput [Hz]"].values)

# Keep paddle-only ka around too, for stats / legend annotation.
wave_clip["ka"] = (wave_clip[K_COL].astype(float)
                   * wave_clip[A_COL].astype(float) / 1000.0)

# ── K_t override rule — see top-of-file IMMUTABLE block for full reasoning ───
# Per row: recompute K_t under each of the three amplitude methods (FFT, LS,
# PSD) using `in_probes_used` / `out_probes_used` to know which probes
# contribute. Then keep canonical FFT unless FFT contradicts BOTH LS and
# PSD, AND those two agree with each other — in which case use LS.
def _kt_method(row, method_suffix):
    inp = [p.strip() for p in str(row["in_probes_used"]).split("+")]
    out = [p.strip() for p in str(row["out_probes_used"]).split("+")]
    try:
        a_in  = float(np.nanmean([row[f"Probe {p} Amplitude{method_suffix}"] for p in inp]))
        a_out = float(np.nanmean([row[f"Probe {p} Amplitude{method_suffix}"] for p in out]))
        if a_in <= 0 or not np.isfinite(a_in) or not np.isfinite(a_out):
            return np.nan
        return a_out / a_in
    except KeyError:
        return np.nan

wave_clip["Kt_FFT_recomp"] = wave_clip.apply(lambda r: _kt_method(r, " (FFT)"), axis=1)
wave_clip["Kt_LS"]         = wave_clip.apply(lambda r: _kt_method(r, " (LS)"),  axis=1)
wave_clip["Kt_PSD"]        = wave_clip.apply(lambda r: _kt_method(r, " (PSD)"), axis=1)

# Canonical FFT K_t — the cached one (matches downstream pipeline columns
# exactly). The recomputed _recomp version above is for sanity-checking and
# should agree to numerical precision.
wave_clip["Kt_canon"] = wave_clip["OUT/IN (FFT)"].astype(float)

# Three-way override decision — see top-of-file block.
_dF = (wave_clip["Kt_canon"] - wave_clip["Kt_LS"]).abs()
_dP = (wave_clip["Kt_canon"] - wave_clip["Kt_PSD"]).abs()
_dLP = (wave_clip["Kt_LS"]    - wave_clip["Kt_PSD"]).abs()
_override = (_dF > OVERRIDE_THRESHOLD) & (_dP > OVERRIDE_THRESHOLD) & (_dLP < OVERRIDE_THRESHOLD)
wave_clip["Kt_override"] = _override
wave_clip["Kt_eff"] = np.where(_override, wave_clip["Kt_LS"], wave_clip["Kt_canon"])

print(f"\n   K_t override (FFT → LS) fires on "
      f"{int(_override.sum())} of {len(wave_clip)} runs "
      f"(threshold |ΔKt| > {OVERRIDE_THRESHOLD}, LS+PSD agreement required)")
if _override.any():
    _cols = ["WaveFrequencyInput [Hz]", "WaveAmplitudeInput [Volt]",
             "WindCondition", "Mooring", "PanelCondition",
             "Kt_canon", "Kt_LS", "Kt_PSD"]
    print(wave_clip[_override][_cols].to_string())
    print("   Override paths:")
    for _p in wave_clip[_override]["path"]:
        print(f"     {_p.split('/wavedata/', 1)[1]}")

print(f"\n   k range: [{wave_clip['k'].min():.3f}, {wave_clip['k'].max():.3f}]")
print(f"   freq range: [{wave_clip['WaveFrequencyInput [Hz]'].min():.2f}, "
      f"{wave_clip['WaveFrequencyInput [Hz]'].max():.2f}] Hz")

# Compose the (mooring × panel) category — above_50 split by panel.
def _category(row):
    m = row["Mooring"]; p = row["PanelCondition"]
    if m == "below_90_loose300" and p == "full":     return "below_loose300_full"
    if m == "below_90_loose230" and p == "full":     return "below_loose230_full"
    if m == "above_50"          and p == "full":     return "above_50_full"
    if m == "above_50"          and p == "reverse":  return "above_50_reverse"
    return "other"

wave_clip["category"] = wave_clip.apply(_category, axis=1)
n_other = int((wave_clip["category"] == "other").sum())
if n_other:
    print(f"   {n_other} runs in 'other' category dropped")
    wave_clip = wave_clip[wave_clip["category"] != "other"].copy()

print("\n2. Counts per category × wind:")
print(wave_clip.groupby(["category", "WindCondition"]).size().unstack(fill_value=0).to_string())
print(f"   total: {len(wave_clip)} runs")

# ── 3. Encoding — identical to the ka sibling ─────────────────────────────────
ABOVE_FULLWIND_COLOR = "#B22222"   # firebrick (muted dark red)
ABOVE_NOWIND_COLOR   = "#4682B4"   # steel blue (muted blue)

COLORS = {
    ("below_loose300_full", "no"):   WIND_COLOR_MAP["no"],   # standard blue
    ("below_loose300_full", "full"): WIND_COLOR_MAP["full"], # standard red
    ("below_loose230_full", "no"):   "#9ECAE1",              # light blue
    ("below_loose230_full", "full"): "#F4815A",              # orange salmon
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
    "above_50_full",
    "above_50_reverse",
    "below_loose230_full",
    "below_loose300_full",
]

MARKERS = {
    "below_loose300_full": {0.10: "o",  0.20: "s",  0.30: "^"},
    "below_loose230_full": {0.10: "o",  0.20: "s",  0.30: "^"},
    "above_50_full":       {0.10: "D",  0.20: "P",  0.30: "h"},
    "above_50_reverse":    {0.10: "D",  0.20: "P",  0.30: "h"},
}
WIND_LABEL = {"no": "uten vind", "full": "full vind"}
MARKER_SIZE = 55
ALPHA = 0.75
EDGE_LW = 0.4


def _round_amp(v): return round(float(v), 2)


# ── Shared axis envelope so the under/over views compare 1:1 ──────────────────
# Derived from the full filtered subset (all categories) so both views share
# the same x and y box.
_k_min = float(wave_clip["k"].min())
_k_max = float(wave_clip["k"].max())
_x_pad = max(0.15, 0.02 * (_k_max - _k_min))
XLIM = (_k_min - _x_pad, _k_max * 1.02)
YLIM = (0.1, 1.05)

# Thesis-scope band in k-space (1.3–1.6 Hz) — same as the parent k-axis figure.
THESIS_K_LO = float(freq_to_k(np.array([1.3]))[0])
THESIS_K_HI = float(freq_to_k(np.array([1.6]))[0])


def _make_view(sub: pd.DataFrame, *,
               categories: list[str],
               figure_name: str,
               plot_type: str,
               view_label: str,
               draw_fits: bool = True) -> None:
    """Build one scatter figure on the k-axis, save PDF + stub."""
    # A4 portrait minus 1-inch margins. Same aspect as ka sibling.
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
                if s.empty:
                    continue
                color  = COLORS.get((cat, wind), "gray")
                marker = MARKERS[cat][amp_v]
                sz = MARKER_SIZE * (1.6 if cat.startswith("above_50") else 1.0)
                hollow = cat in HOLLOW_CATEGORIES
                face_color = "none" if hollow else color
                edge_color = color  if hollow else "black"
                edge_lw    = 1.1    if hollow else EDGE_LW
                # y-source: Kt_eff = per-row FFT→LS override (see top-of-file block).
                ax.scatter(
                    s["k"], s["Kt_eff"],
                    facecolors=face_color, edgecolors=edge_color,
                    marker=marker, s=sz,
                    linewidths=edge_lw, alpha=ALPHA,
                    zorder=3 if cat == "below_loose300_full" else 2,
                )
                # Per-cell linear fit (one line per (cat × amp × wind))
                # over the cell's actual k range.
                if draw_fits:
                    k_cell = s["k"].to_numpy(float)
                    # Kt_eff: per-row FFT→LS override rule (see top-of-file block).
                    kt_cell = s["Kt_eff"].to_numpy(float)
                    if len(k_cell) >= 2 and k_cell.std() > 1e-9:
                        p = np.polyfit(k_cell, kt_cell, deg=1)
                        x_line = np.array([k_cell.min(), k_cell.max()])
                        y_line = np.polyval(p, x_line)
                        ax.plot(
                            x_line, y_line,
                            color=color, lw=0.6, alpha=0.55,
                            linestyle=(0, (3, 1.5)) if hollow else "-",
                            zorder=4 if cat == "below_loose300_full" else 3,
                            solid_capstyle="round",
                        )

    # Thesis-scope band → light-blue axvspan + annotation. Same convention
    # as analysis_scratch/all_data_damping_scatter.py.
    ax.axvspan(THESIS_K_LO, THESIS_K_HI,
               color=WIND_COLOR_MAP["no"], alpha=0.07, lw=0, zorder=1)
    # Place the annotation just inside the band, near the top of the
    # data area. Y position uses YLIM ceiling minus a small margin.
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

    # LS-override count annotation (lower-left corner of the data area).
    # See top-of-file IMMUTABLE block for the override rule. Always rendered,
    # even when n=0, so the reader knows the check was applied.
    _n_ls_swap = int(sub[sub["category"].isin(categories)]["Kt_override"].sum())
    ax.text(
        XLIM[0] + 0.02 * (XLIM[1] - XLIM[0]),
        YLIM[0] + 0.025,
        f"n = {_n_ls_swap} data estimert med LS",
        ha="left", va="bottom",
        fontsize=8, color="#555555", alpha=0.90,
        bbox=dict(boxstyle="round,pad=0.25",
                  facecolor="white", alpha=0.80, edgecolor="none"),
        zorder=5,
    )

    # Top axis: every frequency actually present in this view as a tick.
    secax = add_freq_axis(ax)
    secax.set_xlabel(r"Frekvens (Hz)", fontsize=9)
    _used_freqs = sorted(sub[sub["category"].isin(categories)]
                         ["WaveFrequencyInput [Hz]"].unique())
    secax.set_xticks(_used_freqs)
    secax.set_xticklabels([f"{f:.1f}" for f in _used_freqs])
    secax.tick_params(labelsize=7)

    # Configuration legend (cat × wind) — only entries relevant to this view.
    config_handles = []
    for cat in plot_order[::-1]:
        is_above = cat.startswith("above_50")
        cat_marker_exemplar = "D" if is_above else "o"
        cat_msize = 11 if is_above else 9
        hollow = cat in HOLLOW_CATEGORIES
        for wind, wlabel in (("full", "full vind"),
                             ("no",   "uten vind")):
            wind_color = COLORS[(cat, wind)]
            mfc = "none"      if hollow else wind_color
            mec = wind_color  if hollow else "black"
            mew = 1.1         if hollow else 0.4
            config_handles.append(
                mlines.Line2D([], [],
                              marker=cat_marker_exemplar, linestyle="None",
                              markerfacecolor=mfc,
                              markeredgecolor=mec, markeredgewidth=mew,
                              markersize=cat_msize,
                              label=f"{CATEGORY_LABELS[cat]}, {wlabel}")
            )

    # Amplitude legend.
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
                              label=f"{amp_to_label(v)}{'  (under)' if has_above else ''}")
            )
        if has_above:
            amp_handles.append(
                mlines.Line2D([], [], color="black",
                              marker=MARKERS["above_50_full"][v], linestyle="None",
                              markersize=10, markerfacecolor="lightgray",
                              markeredgecolor="black", markeredgewidth=0.3,
                              label=f"{amp_to_label(v)}{'  (over)' if has_below else ''}")
            )
    amp_legend_ncol = 2 if (has_below and has_above) else 1

    leg_cfg = ax.legend(handles=config_handles, loc="upper right",
                        bbox_to_anchor=(0.995, 0.995),
                        fontsize=7.5, framealpha=0.92,
                        title="Konfigurasjon", title_fontsize=8,
                        ncol=2 if len(config_handles) >= 4 else 1)
    ax.add_artist(leg_cfg)

    fig.canvas.draw()
    leg_cfg_bbox = leg_cfg.get_window_extent().transformed(ax.transAxes.inverted())
    amp_anchor_y = leg_cfg_bbox.y0 - 0.010

    ax.legend(handles=amp_handles, loc="upper right",
              bbox_to_anchor=(0.995, amp_anchor_y),
              fontsize=7, framealpha=0.92,
              title="Amplitude", title_fontsize=8,
              ncol=amp_legend_ncol)

    sub_view = sub[sub["category"].isin(categories)]
    n_total = len(sub_view)
    print(f"\n   {view_label}:  n = {n_total} kjøringer")

    fig.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.06)
    apply_horizontal_ylabel(ax, r"$K_t$", fontsize=12)

    # Save → output/FIGURES + stub
    out_pdf  = BASE / "output" / "FIGURES" / f"{figure_name}.pdf"
    out_stub = BASE / "output" / "TEXFIGU" / f"{figure_name}.tex"
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_stub.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    print(f"   Saved → {out_pdf.relative_to(BASE)}")

    # Stub
    import wavescripts.plot_utils as pu
    pu.ACTIVE_DATASETS = [Path(d).name for d in all_dirs]
    cats_str = ", ".join(categories)
    _meta_stub = pu.build_fig_meta(
        {
            "filters": {
                "PanelCondition":   "full + reverse" if "above_50" in categories else "full",
                "WindCondition":    "no, full",
                "quality_flag":     "ok",
                "categories":       cats_str,
            },
            "plotting": {"figure_name": figure_name},
        },
        chapter=CHAPTER,
        extra={"script": "analysis_scratch/under_and_over_mooring_scatter_k.py"},
        computed_in=("analysis_scratch/under_and_over_mooring_scatter_k.py "
                     f"(view: {view_label})"),
        data_class="DELEG",
        findings_doc=None,
        fft_window_hz=0.1,
        extra_params=(
            f"all PROCESSED-* folders ({len(all_dirs)}). Same filter as the "
            f"ka sibling ch05_damping_all_data_scatter_ka, restricted to "
            f"categories: {cats_str}. "
            f"x-axis = k (rad/m) from dispersion: ω² = gk·tanh(kh), h=0.58 m. "
            f"Top frequency axis with one tick per used frequency. "
            f"Thesis-scope band (1.3–1.6 Hz) shaded as a light-blue axvspan. "
            f"Encoding: same (mooring × panel) scheme as the ka sibling — "
            f"colour shade encodes mooring (within red/blue families), "
            f"marker family encodes mooring family (○/□/△ for below; "
            f"diamond/plus/hex for above_50, hollow on reverse panel). "
            f"Shared XLIM/YLIM across the two views (under, over). "
            f"K_t SOURCE: canonical OUT/IN (FFT) with per-row LS+PSD override "
            f"— if |Kt_FFT−Kt_LS|>{OVERRIDE_THRESHOLD} AND "
            f"|Kt_FFT−Kt_PSD|>{OVERRIDE_THRESHOLD} AND "
            f"|Kt_LS−Kt_PSD|<{OVERRIDE_THRESHOLD}, use Kt_LS (FFT broken on "
            f"that run — typically by cut_samples breaking window cycle "
            f"coherence). Override fires on 3/399 in-scope runs across the "
            f"full dataset, none of which is in the canon thesis-band scope. "
            f"See script's top-of-file IMMUTABLE block."
        ),
        extra_stats={
            "n_total":  len(sub_view),
            **{f"n_{c}": int((sub_view["category"] == c).sum()) for c in categories},
            "k_min":    float(sub_view["k"].min()),
            "k_max":    float(sub_view["k"].max()),
            "ka_min":   float(sub_view["ka"].min()),
            "ka_max":   float(sub_view["ka"].max()),
            "freq_min": float(sub_view["WaveFrequencyInput [Hz]"].min()),
            "freq_max": float(sub_view["WaveFrequencyInput [Hz]"].max()),
            # n_kt_override = runs in this view where the FFT-vs-LS+PSD
            # rule swapped Kt_FFT for Kt_LS. See top-of-script IMMUTABLE.
            "n_kt_override": int(sub_view["Kt_override"].sum()),
            "kt_override_threshold": OVERRIDE_THRESHOLD,
        },
    )
    pu.write_figure_stub(_meta_stub, plot_type=plot_type,
                         subfig_filenames=[figure_name],
                         thispagestyle="empty")
    print(f"   Stub → {out_stub.relative_to(BASE)}")
    plt.close(fig)


# ── 4. Build the two views ────────────────────────────────────────────────────
print("\n3. Building views …")
VIEWS = [
    {"name": "under", "categories": ["below_loose300_full", "below_loose230_full"],
     "figure_name": "ch05_damping_undermooring_scatter",
     "plot_type":   "damping_undermooring_scatter",
     "view_label":  "undermooring only (loose300 + loose230), k-axis",
     "draw_fits":   True},
    {"name": "over",  "categories": ["above_50_full", "above_50_reverse"],
     "figure_name": "ch05_damping_overmooring_scatter",
     "plot_type":   "damping_overmooring_scatter",
     "view_label":  "overmooring only (above_50 full + reverse), k-axis",
     "draw_fits":   True},
]

for view in VIEWS:
    _make_view(wave_clip,
               categories=view["categories"],
               figure_name=view["figure_name"],
               plot_type=view["plot_type"],
               view_label=view["view_label"],
               draw_fits=view["draw_fits"])

print("\nDone.")
