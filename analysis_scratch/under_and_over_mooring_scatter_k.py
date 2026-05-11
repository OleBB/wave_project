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
                ax.scatter(
                    s["k"], s["OUT/IN (FFT)"],
                    facecolors=face_color, edgecolors=edge_color,
                    marker=marker, s=sz,
                    linewidths=edge_lw, alpha=ALPHA,
                    zorder=3 if cat == "below_loose300_full" else 2,
                )
                # Per-cell linear fit (one line per (cat × amp × wind))
                # over the cell's actual k range.
                if draw_fits:
                    k_cell = s["k"].to_numpy(float)
                    kt_cell = s["OUT/IN (FFT)"].to_numpy(float)
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
            f"Shared XLIM/YLIM across the two views (under, over)."
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
