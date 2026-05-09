"""
All-data damping scatter (CH05 supplementary) — ka variant
==========================================================

Companion to `all_data_damping_scatter.py`. Same scope (cross-condition
overview), x-axis is paddle-only ka per run.

x-axis = `IN Wavenumber (FFT)` × `IN Amplitude (FFT)` [m]. Both factors
are FFT-measured, paddle-tone-only quantities (the pipeline column
`IN ka (FFT)` mixes FFT wavenumber with the time-domain percentile
amplitude, which inflates ka under wind by the wind-wave energy on
top of the paddle wave — misleading for steepness, see CLAUDE.md §16).

Visual language (updated 2026-05-09 — the k-axis sibling still uses the
old hardware-fill convention; this ka variant adopts the
mooring×panel-category scheme established by
analysis_scratch/all_data_damping_scatter_by_mooring.py):

  - Wind condition  → colour hue (red family = full, blue family = no)
  - (Mooring × panel) → colour shade + marker family
      below_loose300 × full    : standard red / blue       | ○ A1, □ A2, △ A3
      below_loose230 × full    : light salmon / light blue | ○ A1, □ A2, △ A3
      above_50 (panel pooled) : pink / turquoise          | ✶ A1 (6-pt),
                                                            ★ A2 (5-pt),
                                                            ✦ A3 (4-pt)

The above_50 pooling (full panel + reverse panel together) is justified
in the appendix table tab:app_panel_pooling — |Δ| ≤ 0.10 in 100% of
cells where both panels were tested.

Run from repo root:
    conda run -n draumkvedet python analysis_scratch/all_data_damping_scatter_ka.py

Outputs:
    analysis_scratch/all_data_damping_scatter_ka.pdf       (scratch quick-view)
    analysis_scratch/all_data_damping_scatter_ka_summary.csv
    output/FIGURES/ch05_damping_all_data_scatter_ka.pdf    (thesis supplementary)
    output/TEXFIGU/ch05_damping_all_data_scatter_ka.tex    (stub; caption from
                                                            FIGURE_CAPTIONS in
                                                            main_save_figures.py)
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
    apply_horizontal_ylabel,
)

apply_thesis_style()

# ── I/O ────────────────────────────────────────────────────────────────────────
SCRATCH_PDF = Path(__file__).parent / "all_data_damping_scatter_ka.pdf"
SCRATCH_CSV = Path(__file__).parent / "all_data_damping_scatter_ka_summary.csv"

THESIS_NAME = "ch05_damping_all_data_scatter_ka"
OUT_PDF  = BASE / "output" / "FIGURES" / f"{THESIS_NAME}.pdf"
OUT_STUB = BASE / "output" / "TEXFIGU" / f"{THESIS_NAME}.tex"
CHAPTER  = "05"

K_COL = "IN Wavenumber (FFT)"
A_COL = "IN Amplitude (FFT)"

# ── 1. Load everything ─────────────────────────────────────────────────────────
print("1. Loading all processed folders …")
all_dirs = sorted(glob.glob(str(BASE / "waveprocessed" / "PROCESSED-*")))
print(f"   {len(all_dirs)} folders")
meta, _, _, _ = load_analysis_data(*all_dirs, load_processed=False)
print(f"   {len(meta)} total rows")

# ── 2. Filter & categorize by (mooring × panel) ───────────────────────────────
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

# Paddle-only ka: FFT wavenumber × FFT amplitude (mm → m).
wave_clip["ka"] = (wave_clip[K_COL].astype(float)
                   * wave_clip[A_COL].astype(float) / 1000.0)

print(f"\n   ka range: [{wave_clip['ka'].min():.3f}, {wave_clip['ka'].max():.3f}]")
print(f"   freq range: [{wave_clip['WaveFrequencyInput [Hz]'].min():.2f}, "
      f"{wave_clip['WaveFrequencyInput [Hz]'].max():.2f}] Hz")

# Compose the (mooring × panel) category — above_50 split by panel
# orientation (2026-05-09): per user request, reverse panel data on
# above_50 must be visually distinguishable from full panel. Below
# moorings still keep the "_full" suffix because they only have full
# panel data in canon. The pooling justification (app_panel_pooling)
# still holds for the matching table; this split applies to the
# scatter figure only.
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

# ── 3. Save summary CSV ───────────────────────────────────────────────────────
summary = (wave_clip.groupby(["category", "WindCondition", "PanelCondition"])
                     .agg(n=("path", "count"),
                          freq_min=("WaveFrequencyInput [Hz]", "min"),
                          freq_max=("WaveFrequencyInput [Hz]", "max"),
                          ka_min=("ka", "min"),
                          ka_max=("ka", "max"),
                          out_in_mean=("OUT/IN (FFT)", "mean"),
                          out_in_std=("OUT/IN (FFT)", "std"))
                     .reset_index())
summary.to_csv(SCRATCH_CSV, index=False)
print(f"\n   Summary → {SCRATCH_CSV.relative_to(BASE)}")

# ── 4. Plot — single figure, mooring×panel categories pooled per by_mooring ──
# Above_50 (over-mooring) palette (2026-05-09, professional pass):
# steel blue (nowind) + firebrick (fullwind). Both are classical matplotlib
# named colours — clearly in the red/blue family, more muted than the
# canon vibrant red/blue (loose300) and clearly distinct from the light
# salmon/light blue (loose230). Below moorings unchanged.
#
# History (over-mooring colour iterations):
#   morning : turquoise / pink  (#17BECF / #E377C2)  — felt cartoonish
#   mid-day : yellow / purple   (#FFD60A / #9467BD)  — too saturated
#   evening : turquoise / magenta (#17BECF / #D81B7A) — bright, garish
#   final   : steel blue / firebrick (#4682B4 / #B22222) — muted, classical
ABOVE_FULLWIND_COLOR = "#B22222"   # firebrick (muted dark red)
ABOVE_NOWIND_COLOR   = "#4682B4"   # steel blue (muted blue)

COLORS = {
    ("below_loose300_full", "no"):   WIND_COLOR_MAP["no"],   # standard blue
    ("below_loose300_full", "full"): WIND_COLOR_MAP["full"], # standard red
    ("below_loose230_full", "no"):   "#9ECAE1",              # light blue
    ("below_loose230_full", "full"): "#F4815A",              # orange salmon
                                                             # (shifted from
                                                             # #FCAE91 to widen
                                                             # gap to magenta)
    ("above_50_full",       "no"):   ABOVE_NOWIND_COLOR,
    ("above_50_full",       "full"): ABOVE_FULLWIND_COLOR,
    ("above_50_reverse",    "no"):   ABOVE_NOWIND_COLOR,
    ("above_50_reverse",    "full"): ABOVE_FULLWIND_COLOR,
}

# Categories where the marker is rendered HOLLOW (no fill, colour goes
# to the outline). Currently used to distinguish above_50 reverse from
# above_50 full panel.
HOLLOW_CATEGORIES = {"above_50_reverse"}

CATEGORY_LABELS = {
    "below_loose300_full": "Under, 30 cm",
    "below_loose230_full": "Under, 23 cm",
    "above_50_full":       "Over",                # mm dropped — implicit
    "above_50_reverse":    "Over, revers",        # only flag the deviation
}

CATEGORY_ORDER = [
    "above_50_full",         # bottom — paint under
    "above_50_reverse",      # bottom-ish; hollow stars sit just above above_50_full
    "below_loose230_full",
    "below_loose300_full",   # top — canonical reference most visible
]

# Marker per (category, amp). Below moorings use the project amp-tier
# convention; above_50 (both panels) uses three visually distinct shapes
# — diamond / filled-plus / hexagon. Pentagon (`p`) was tried for A2 but
# read as too similar to the hexagon at small thesis-figure sizes; the
# filled plus (`P`) breaks the polygon pattern entirely and pops cleanly.
# All three render hollow (face='none', colour to outline) for the
# reverse-panel split. Full vs reverse panel is encoded by fill (see
# HOLLOW_CATEGORIES), not shape.
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

# ── Shared axis envelope so the 3 views (all / under / over) compare 1:1 ─────
# Computed from the all-data subset so each view sits inside the same box.
XLIM = (0.0, max(0.36, wave_clip["ka"].max() * 1.02))
YLIM = (0.1, 1.05)   # ceiling at 1.05 — drops K_t > 1 outliers (probe
                     # dropouts) and tightens the y-axis, freeing the
                     # top band for the stacked legends (2026-05-09).


def _make_view(sub: pd.DataFrame, *,
               categories: list[str],
               figure_name: str,
               plot_type: str,
               view_label: str,
               draw_fits: bool = True) -> None:
    """Build one scatter figure restricted to ``categories``, save PDF + stub.

    Same axes / encoding as the parent (all-data) figure — only the
    category set changes per view. Shared XLIM/YLIM keeps the three
    views directly comparable.
    """
    # Sized for A4 portrait minus 1-inch margins (8.27" × 11.69" →
    # 6.27" × 9.69" printable area). Tall aspect gives the K_t axis more
    # vertical real estate than the previous near-square (8.5, 8.0).
    fig, ax = plt.subplots(figsize=(6.27, 9.69))

    # Plot order: same as in CATEGORY_ORDER, restricted to this view.
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
                # Reverse-panel above_50 cells render hollow: facecolour
                # is "none", outline gets the wind colour. Full panel
                # (and all below moorings) render solid with black outline.
                hollow = cat in HOLLOW_CATEGORIES
                face_color = "none" if hollow else color
                edge_color = color  if hollow else "black"
                edge_lw    = 1.1    if hollow else EDGE_LW
                ax.scatter(
                    s["ka"], s["OUT/IN (FFT)"],
                    facecolors=face_color, edgecolors=edge_color,
                    marker=marker, s=sz,
                    linewidths=edge_lw, alpha=ALPHA,
                    zorder=3 if cat == "below_loose300_full" else 2,
                )
                # Thin per-cell linear fit (one line per (cat × amp × wind))
                # — direct visual companion to the table rows. Span = cell's
                # actual ka range. Style: very thin + low alpha so dots stay
                # primary; colour matches dot face for visual binding to the
                # cell. Reverse-panel cells use a dashed line so the line
                # itself echoes the hollow-marker convention. Skip if the
                # cell has fewer than 2 unique ka values. Disabled in the
                # combined view (draw_fits=False) — 18 lines on 399 dots
                # was too dense to read.
                if draw_fits:
                    ka_cell = s["ka"].to_numpy(float)
                    kt_cell = s["OUT/IN (FFT)"].to_numpy(float)
                    if len(ka_cell) >= 2 and ka_cell.std() > 1e-9:
                        p = np.polyfit(ka_cell, kt_cell, deg=1)
                        x_line = np.array([ka_cell.min(), ka_cell.max()])
                        y_line = np.polyval(p, x_line)
                        ax.plot(
                            x_line, y_line,
                            color=color, lw=0.6, alpha=0.55,
                            linestyle=(0, (3, 1.5)) if hollow else "-",
                            zorder=4 if cat == "below_loose300_full" else 3,
                            solid_capstyle="round",
                        )

    ax.axhline(1.0, color="black", lw=0.6, ls="--", alpha=0.5)
    ax.set_xlabel(r"$ka$  (Inn, målt)", fontsize=11)
    ax.set_ylabel("")
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax.grid(which="major", alpha=0.30, lw=0.6)
    ax.grid(which="minor", alpha=0.15, lw=0.4)
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)

    # Configuration legend (cat × wind) — only entries relevant to this view.
    # Reverse-panel above_50 entries render hollow (face='none'); the wind
    # colour goes to the marker outline so the legend mirrors the figure.
    # n-counts dropped from labels 2026-05-09 — they live in the companion
    # tables (tab:ch05_damping_*_scatter_ka_table) where they're easier to
    # compare across cells.
    config_handles = []
    for cat in plot_order[::-1]:
        is_above = cat.startswith("above_50")
        # Legend exemplar = the A1 marker from the family (parallel to
        # below where exemplar = "o" = MARKERS[*]/0.10). Picks the family's
        # "smoothest" shape so the cat × wind legend reads compactly.
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

    # Amplitude legend — only the marker family present in this view.
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

    # Both legends stack in the upper-right corner (2026-05-09 layout):
    # Konfigurasjon legend on top (wide — typically 2 columns), Amplitude
    # legend immediately below it. Y-axis ceiling at 1.05 frees the top
    # band of the data area; legends sit there without occluding dots.
    # Amplitude legend's anchor y is computed from Konfigurasjon's
    # rendered bbox so the spacing is correct regardless of how many
    # rows Konfigurasjon ends up using.
    leg_cfg = ax.legend(handles=config_handles, loc="upper right",
                        bbox_to_anchor=(0.995, 0.995),
                        fontsize=7.5, framealpha=0.92,
                        title="Konfigurasjon", title_fontsize=8,
                        ncol=2 if len(config_handles) >= 4 else 1)
    ax.add_artist(leg_cfg)

    # Force a draw so leg_cfg has a real bbox; transform it into axes
    # coordinates and drop the amplitude legend just below it.
    fig.canvas.draw()
    leg_cfg_bbox = leg_cfg.get_window_extent().transformed(ax.transAxes.inverted())
    amp_anchor_y = leg_cfg_bbox.y0 - 0.010   # tiny gap

    ax.legend(handles=amp_handles, loc="upper right",
              bbox_to_anchor=(0.995, amp_anchor_y),
              fontsize=7, framealpha=0.92,
              title="Amplitude", title_fontsize=8,
              ncol=amp_legend_ncol)

    n_total = len(sub[sub["category"].isin(categories)])
    print(f"\n   {view_label}:  n = {n_total} kjøringer")

    fig.subplots_adjust(left=0.10, right=0.98, top=0.95, bottom=0.06)
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
    sub_view = sub[sub["category"].isin(categories)]
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
        extra={"script": "analysis_scratch/all_data_damping_scatter_ka.py"},
        computed_in=("analysis_scratch/all_data_damping_scatter_ka.py "
                     f"(view: {view_label})"),
        data_class="DELEG",
        findings_doc=None,
        fft_window_hz=0.1,
        extra_params=(
            f"all PROCESSED-* folders ({len(all_dirs)}). Same filter as the "
            f"parent ch05_damping_all_data_scatter_ka, restricted to "
            f"categories: {cats_str}. "
            f"x-axis = paddle-only ka = `IN Wavenumber (FFT)` × `IN Amplitude "
            f"(FFT)` [m] (FFT-measured, paddle-tone-only). "
            f"Encoding: same as parent — colour shade encodes mooring (within "
            f"red/blue families); marker family encodes mooring family "
            f"(○/□/△ for below; 6/5/4-pt stars for above_50). "
            f"Shared XLIM/YLIM with the parent for direct visual comparison."
        ),
        extra_stats={
            "n_total":  len(sub_view),
            **{f"n_{c}": int((sub_view["category"] == c).sum()) for c in categories},
            "ka_min":   float(sub_view["ka"].min()),
            "ka_max":   float(sub_view["ka"].max()),
        },
    )
    pu.write_figure_stub(_meta_stub, plot_type=plot_type,
                         subfig_filenames=[figure_name],
                         thispagestyle="empty")
    print(f"   Stub → {out_stub.relative_to(BASE)}")
    plt.close(fig)


# ── 4./5./6. Build all three views ────────────────────────────────────────────
print("\n3. Building views …")
VIEWS = [
    # Combined view: regression lines OFF — 18 thin lines over 399 dots
    # was visually overcrowded. The under/over subviews keep their lines
    # since they're sparse enough to read; the combined view leans on the
    # tables for per-cell numbers (tab:ch05_damping_all_data_scatter_ka_table).
    {"name": "all",   "categories": CATEGORY_ORDER,
     "figure_name": "ch05_damping_all_data_scatter_ka",
     "plot_type":   "damping_all_data_scatter_ka",
     "view_label":  "all data (loose300 + loose230 + above_50 split by panel)",
     "draw_fits":   False},
    {"name": "under", "categories": ["below_loose300_full", "below_loose230_full"],
     "figure_name": "ch05_damping_undermooring_scatter_ka",
     "plot_type":   "damping_undermooring_scatter_ka",
     "view_label":  "undermooring only (loose300 + loose230)",
     "draw_fits":   True},
    {"name": "over",  "categories": ["above_50_full", "above_50_reverse"],
     "figure_name": "ch05_damping_overmooring_scatter_ka",
     "plot_type":   "damping_overmooring_scatter_ka",
     "view_label":  "overmooring only (above_50 full + reverse, panel-split)",
     "draw_fits":   True},
]

for view in VIEWS:
    _make_view(wave_clip,
               categories=view["categories"],
               figure_name=view["figure_name"],
               plot_type=view["plot_type"],
               view_label=view["view_label"],
               draw_fits=view["draw_fits"])

# Also write the all-data scratch sibling for backward compat
fig_all_pdf_scratch = SCRATCH_PDF
fig_all_pdf_thesis = BASE / "output" / "FIGURES" / "ch05_damping_all_data_scatter_ka.pdf"
import shutil
shutil.copyfile(fig_all_pdf_thesis, fig_all_pdf_scratch)
print(f"\n   Scratch sibling → {fig_all_pdf_scratch.relative_to(BASE)}")

print("\nDone.")
