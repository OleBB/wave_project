"""
CH05 § 1 sibling — Damping vs frequency, over-mooring (above_50)
================================================================

Mirror of the `ch05_damping_freq_full_A{1,2,3}` three-figure column split
(main_save_figures.py around line ~2113) but restricted to **above_50**
mooring and with **PanelCondition pooled** (full + reverse → "all").

  - Frequency band: 1.3–1.6 Hz inclusive
  - Wind:           no / full only (drops "lowest")
  - Mooring:        above_50 only
  - Panel:          full + reverse pooled via collapse_panels=True
  - Same K_t,probe>1 dropout filter as the parent cell

The pooling justification (above_50 full vs reverse) is in the appendix
table `tab:app_panel_pooling` (referenced by analysis_scratch/all_data_damping_scatter_ka.py).

Outputs:
    output/FIGURES/ch05_damping_freq_overmooring_A1.pdf  (+ .pgf)
    output/FIGURES/ch05_damping_freq_overmooring_A2.pdf  (+ .pgf)
    output/FIGURES/ch05_damping_freq_overmooring_A3.pdf  (+ .pgf)
    output/TEXFIGU/ch05_damping_freq_overmooring.tex

Run from repo root:
    conda run -n draumkvedet python analysis_scratch/damping_freq_overmooring.py

Decision notes (2026-05-11):
  • Standalone script (option B) rather than a new cell in main_save_figures.py:
    above_50 data is not in `meta_results` (which is restricted to the two
    canon below_90 folders). A delegated script in analysis_scratch/ is the
    established precedent (cf. mooring_focus_at_1_3hz_ka.py,
    all_data_damping_scatter_ka.py).
  • Instead of calling `plot_damping_freq`, we drive its private helpers
    `_make_damping_freq_fig`, `_save_figure`, `build_fig_meta`, and
    `write_figure_stub` directly so the output filenames are exactly
    `ch05_damping_freq_overmooring_A{1,2,3}.pdf` and the stub is
    `ch05_damping_freq_overmooring.tex`. The public `plot_damping_freq`
    uses `f"{figure_name}_{panel}_{amp_tag}"` and writes the stub to
    `{figure_name}.tex` — there is no `(figure_name, panel)` setting that
    yields both the desired pdf names AND a non-clobbering stub
    filename in one shot.
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

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import os
os.chdir(BASE)

from wavescripts.improved_data_loader import load_analysis_data
from wavescripts.filters import (
    apply_experimental_filters as _aef,
    damping_all_amplitude_grouper,
)
from wavescripts.plotter import _make_damping_freq_fig
from wavescripts.plot_utils import (
    _save_figure,
    apply_thesis_style,
    amp_to_tag,
    amp_to_label,
    build_fig_meta,
    write_figure_stub,
)
import wavescripts.plot_utils as _pu

apply_thesis_style()

FIGURE_NAME = "ch05_damping_freq_overmooring"
CHAPTER = "05"


# ── 1. Load all processed folders (above_50 lives outside RESULTS_PROCESSED_DIRS) ──
print("1. Loading all processed folders …")
all_dirs = sorted(glob.glob(str(BASE / "waveprocessed" / "PROCESSED-*")))
print(f"   {len(all_dirs)} folders")
meta, _, _, _ = load_analysis_data(*all_dirs, load_processed=False)
print(f"   {len(meta)} total rows loaded")


# ── 2. Filter: 1.3–1.6 Hz × {no, full} wind × {full, reverse} panel × above_50 ──
_pv = {
    "filters": {
        "WaveAmplitudeInput [Volt]": (0.1, 0.3),
        "WaveFrequencyInput [Hz]":   (1.3, 1.6),     # thesis scope band
        "WindCondition":             ["no", "full"], # drop "lowest"
        # Panel: keep BOTH "full" and "reverse" — pooled below via
        # collapse_panels=True. Justification in tab:app_panel_pooling.
        "PanelCondition":            ["full", "reverse"],
    },
    "plotting": {
        "show_plot":  False,
        "save_plot":  True,
        "force_stub": True,
        "figure_name": FIGURE_NAME,
        "figsize":    (7, 3),
        "annotate":   True,
        "legend":     "outside_right",
        "subfig_layout": "column",
        # Use the over-mooring (above_50) palette — muted firebrick /
        # steel blue — so this figure shares its colour language with
        # ch05_damping_overmooring_scatter{,_ka}. Same hex codes as
        # ABOVE_FULLWIND_COLOR / ABOVE_NOWIND_COLOR in
        # analysis_scratch/under_and_over_mooring_scatter_k.py.
        "wind_color_map": {"no": "#4682B4", "full": "#B22222"},
    },
}
_meta = _aef(meta, _pv)
print(f"\n2. After _aef filters: {len(_meta)} rows")

# Mooring-restrict to above_50.
if "Mooring" not in _meta.columns:
    raise RuntimeError("Mooring column missing from meta — cannot filter to above_50.")
n_before = len(_meta)
_meta = _meta[_meta["Mooring"] == "above_50"].copy()
print(f"   After Mooring=='above_50': {len(_meta)} rows (was {n_before})")

if _meta.empty:
    raise RuntimeError(
        "No above_50 rows survived the filters. Check that PROCESSED-* folders "
        "for the above_50 mooring exist in waveprocessed/ and contain runs in "
        "the 1.3–1.6 Hz band with wind in {no, full}."
    )

print("\n   PanelCondition × WindCondition coverage (rows):")
print(_meta.groupby(["PanelCondition", "WindCondition"]).size().unstack(fill_value=0).to_string())

# Tighten the stub's `datasets:` provenance to folders that actually
# contributed rows.
_active_dates = {str(d).replace("-", "") for d in _meta["file_date"].astype(str).unique()}
_pu.ACTIVE_DATASETS = sorted(
    Path(d).name for d in all_dirs
    if any(date in Path(d).name for date in _active_dates)
)
print(f"\n   Active provenance folders: {_pu.ACTIVE_DATASETS}")


# ── 3. K_t,probe>1 dropout filter (same as parent cell) ──────────────────────
_A_in_wall = _meta.get("Probe 9373/170 Amplitude (FFT)")
_A_in_far  = _meta.get("Probe 9373/340 Amplitude (FFT)")
_A_out_ctr = _meta.get("Probe 12400/250 Amplitude (FFT)")
if _A_in_wall is not None and _A_in_far is not None and _A_out_ctr is not None:
    _kt_wall = _A_out_ctr / _A_in_wall
    _kt_far  = _A_out_ctr / _A_in_far
    _dropout = (_kt_wall > 1.0) | (_kt_far > 1.0)
    if _dropout.any():
        print(f"\n3. Dropping {_dropout.sum()} runs with K_t,probe > 1 (single-probe dropout):")
        for _, _r in _meta[_dropout].iterrows():
            print(f"   Kt_wall={_kt_wall[_r.name]:.3f} Kt_far={_kt_far[_r.name]:.3f}  "
                  f"{_r['path'].split('/')[-1]}")
        _meta = _meta[~_dropout].copy()
    else:
        print("\n3. No K_t,probe > 1 dropout runs detected.")
else:
    print("\n3. WARNING: per-probe FFT amplitude columns missing — skipping dropout filter.")


# ── 4. Group: pool full + reverse panel → "all" via collapse_panels=True ─────
print("\n4. Aggregating with collapse_panels=True (full + reverse → 'all') …")
_grouped = damping_all_amplitude_grouper(_meta, collapse_panels=True)

# `_make_damping_freq_fig` filters on PanelCondition. After collapse_panels
# the grouper writes to PanelConditionGrouped; reinstate PanelCondition
# with a single pooled label.
if "PanelCondition" not in _grouped.columns:
    if "PanelConditionGrouped" in _grouped.columns:
        _grouped["PanelCondition"] = _grouped["PanelConditionGrouped"]
    else:
        raise RuntimeError(
            "Neither PanelCondition nor PanelConditionGrouped present in grouper output."
        )
PANEL_LABEL = _grouped["PanelCondition"].iloc[0]  # "all" by convention
print(f"   pooled panel label = '{PANEL_LABEL}'  (rows: {len(_grouped)})")


# ── 5. Diagnostics: per-(amp × wind × freq) n_runs coverage ─────────────────
print("\n5. n_runs per (amp × wind × freq) — verify above_50 coverage:")
_pivot = (_grouped.assign(amp_tag=_grouped["WaveAmplitudeInput [Volt]"]
                          .map({0.10: "A1", 0.20: "A2", 0.30: "A3"})
                          .fillna(_grouped["WaveAmplitudeInput [Volt]"].astype(str)))
                  .pivot_table(index=["amp_tag", "WindCondition"],
                               columns="WaveFrequencyInput [Hz]",
                               values="n_runs",
                               aggfunc="sum",
                               fill_value=0))
print(_pivot.to_string())

_expected_freqs = [1.3, 1.4, 1.5, 1.6]
_empty = []
for _amp_tag in ("A1", "A2", "A3"):
    for _wind in ("no", "full"):
        for _f in _expected_freqs:
            try:
                _n = int(_pivot.loc[(_amp_tag, _wind), _f])
            except KeyError:
                _n = 0
            if _n == 0:
                _empty.append((_amp_tag, _wind, _f))
if _empty:
    print(f"\n   WARNING: {len(_empty)} empty (amp × wind × freq) cells:")
    for _amp_tag, _wind, _f in _empty:
        print(f"     {_amp_tag} × {_wind:>4s} × {_f:.2f} Hz : NO DATA")
else:
    print("\n   All (amp × wind × freq) cells populated.")


# ── 6. Render: one PDF per amplitude, stub listing all three ────────────────
print("\n6. Rendering subfigures …")
amplitudes = sorted(_grouped["WaveAmplitudeInput [Volt]"].unique())

# Caption-slot stats for the stub.
_fw_med = _grouped[_grouped["WindCondition"] == "full"]["mean_out_in"].median()
_nw_med = _grouped[_grouped["WindCondition"] == "no"]["mean_out_in"].median()
_extra_stats = {
    "median_fullwind_OUTIN": round(float(_fw_med), 4) if not pd.isna(_fw_med) else None,
    "median_nowind_OUTIN":   round(float(_nw_med), 4) if not pd.isna(_nw_med) else None,
    "n_panels":     1,                  # pooled
    "n_amplitudes": len(amplitudes),
    "n_total_points": int(len(_grouped)),
    "n_runs_total":   int(_grouped["n_runs"].sum()),
    "points_with_n1": int((_grouped["n_runs"] == 1).sum()),
    "mooring":       "above_50",
    "panel_pooling": "full+reverse pooled (collapse_panels=True; "
                     "justification: tab:app_panel_pooling)",
}
for amp in amplitudes:
    sub = _grouped[_grouped["WaveAmplitudeInput [Volt]"] == amp].sort_values(
        ["WindCondition", "WaveFrequencyInput [Hz]"]
    )
    tokens = [
        f"{r['WindCondition']}-{r['WaveFrequencyInput [Hz]']:.2f}Hz:n={int(r['n_runs'])}"
        for _, r in sub.iterrows()
    ]
    _extra_stats[f"n_per_point_{amp_to_tag(amp)}"] = "; ".join(tokens)

meta_base = build_fig_meta(
    _pv,
    chapter=CHAPTER,
    extra={"script": "analysis_scratch/damping_freq_overmooring.py"},
    data_df=_grouped,
    computed_in="filters.py::damping_all_amplitude_grouper(collapse_panels=True) "
                "→ plotter.py::_make_damping_freq_fig",
    data_class="META",
    findings_doc=None,
    grouper="damping_all_amplitude_grouper",
    collapse_panels=True,
    fft_window_hz=0.1,
    extra_params="window=0.1 Hz, amp_column='OUT/IN (FFT)' (paddle freq only); "
                 "Mooring=above_50; PanelCondition in {full, reverse} pooled",
    extra_stats=_extra_stats,
)

subfig_filenames = []
subfig_captions  = []
for amp in amplitudes:
    fig = _make_damping_freq_fig(_grouped, PANEL_LABEL, amp, figsize=(7, 3))
    fname = f"{FIGURE_NAME}_{amp_to_tag(amp)}"   # ch05_damping_freq_overmooring_A1 etc.
    _save_figure(fig, fname, save_pgf=True)
    subfig_filenames.append(fname)
    subfig_captions.append(f"Over-forankring (above\\_50, panel pooled), {amp_to_label(amp)}")
    plt.close(fig)
    print(f"   wrote FIGURES/{fname}.pdf  (+ .pgf)")

stub_meta = {**meta_base,
             "figure_name": FIGURE_NAME,
             "panel": [PANEL_LABEL],
             "amplitude": amplitudes,
             "wind":      "allwind"}
write_figure_stub(stub_meta, "damping_freq",
                  subfig_filenames=subfig_filenames,
                  subfig_captions=subfig_captions,
                  force=True,
                  subfig_layout="column")
print(f"   wrote TEXFIGU/{FIGURE_NAME}.tex")

print("\nDone.")
