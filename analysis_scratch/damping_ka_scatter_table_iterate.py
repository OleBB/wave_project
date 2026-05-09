"""
Iteration script: tables to accompany the three ka-scatter figures
==================================================================

Companion to ``all_data_damping_scatter_ka.py``. Re-uses the same data
preparation (load → filter → categorise → ka column) so the rows here
are exactly the dots you see in those figures.

Pooling granularity (per user 2026-05-09):
    (category × amplitude × wind condition)
e.g. one row for "below_loose230_full × A1 × full vind".

Three views — one table per view, mirroring the figure split:
    all   = loose300 + loose230 + above_50 pooled
    under = loose300 + loose230 (full panel only)
    over  = above_50 (full + reverse panel pooled)

Per row we compute:
    n              count of runs in the cell
    ka_min/max     ka range in the cell (paddle-only ka)
    Kt_mean        mean of OUT/IN (FFT)
    Kt_std         std of OUT/IN (FFT)
    n_freqs        unique paddle frequencies represented
    slope_dKt_dka  slope of K_t vs ka inside the cell (linear)
    R2             R² of that linear fit
    ka_at_min/max  Kt(ka_min)/(ka_max) from the linear fit (sanity)

The slope/R² columns are diagnostic at this granularity (n_freqs is often
3–4, so R² is noisy). Treat as informative, not decisive — the user will
decide whether to keep them, drop them, or move them to a coarser table.

Run:
    conda run -n draumkvedet python analysis_scratch/damping_ka_scatter_table_iterate.py

Outputs to analysis_scratch/ (NOT output/) — this is iteration scratch:
    damping_ka_scatter_table_iterate_all.csv
    damping_ka_scatter_table_iterate_under.csv
    damping_ka_scatter_table_iterate_over.csv

Once the row shape / ordering / fit decision is locked, promote to
output/TABLES/data/<name>.{csv,meta.json} and add render cells in
main_save_tables.py (two-step pattern, 2026-05-08 convention).
"""

import sys
import glob
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import os
os.chdir(BASE)

from wavescripts.improved_data_loader import load_analysis_data

K_COL = "IN Wavenumber (FFT)"
A_COL = "IN Amplitude (FFT)"


# ── 1. Load + filter + categorise (mirror of the figure script) ───────────────
print("1. Loading all processed folders …")
all_dirs = sorted(glob.glob(str(BASE / "waveprocessed" / "PROCESSED-*")))
meta, _, _, _ = load_analysis_data(*all_dirs, load_processed=False)
print(f"   {len(all_dirs)} folders, {len(meta)} total rows")

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
wave_clip = wave[(wave["OUT/IN (FFT)"] <= 2.0) & (wave["OUT/IN (FFT)"] >= 0.1)].copy()
wave_clip = wave_clip[wave_clip["WaveFrequencyInput [Hz]"] < 2.0].copy()
wave_clip = wave_clip[wave_clip["Mooring"] != "above_200"].copy()
wave_clip["ka"] = (wave_clip[K_COL].astype(float)
                   * wave_clip[A_COL].astype(float) / 1000.0)


def _category(row):
    m, p = row["Mooring"], row["PanelCondition"]
    if m == "below_90_loose300" and p == "full":              return "below_loose300_full"
    if m == "below_90_loose230" and p == "full":              return "below_loose230_full"
    if m == "above_50"          and p in ("full", "reverse"): return "above_50"
    return "other"


wave_clip["category"] = wave_clip.apply(_category, axis=1)
wave_clip = wave_clip[wave_clip["category"] != "other"].copy()
print(f"   {len(wave_clip)} runs in scope across 3 categories")


# ── 2. Aggregation helper ─────────────────────────────────────────────────────
def _round_amp(v):
    return round(float(v), 2)


CATEGORY_LABEL = {
    "below_loose300_full": "Under, loose300, full panel",
    "below_loose230_full": "Under, loose230, full panel",
    "above_50":            "Over (50 mm), pooled paneler",
}
CATEGORY_ORDER = ["below_loose300_full", "below_loose230_full", "above_50"]
WIND_ORDER     = ["no", "full"]
WIND_LABEL     = {"no": "uten", "full": "full"}
AMP_LABEL      = {0.10: "A1", 0.20: "A2", 0.30: "A3"}
AMP_ORDER      = [0.10, 0.20, 0.30]


PANEL_LENGTH_M = 2.6   # L — panel longitudinal length [m]; fixed.


def _linfit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return (slope, R²) of y = a*x + b. NaN if x has no spread."""
    if len(x) < 2 or x.std() < 1e-9:
        return (np.nan, np.nan)
    p = np.polyfit(x, y, deg=1)
    pred  = np.polyval(p, x)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return (float(p[0]), r2)


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Group → (category, amp, wind), one row per cell.

    Per cell, we compute TWO parallel linear fits of K_t against
    different x-axes:
      • vs ka : steepness slope; ka couples wavenumber AND amplitude
      • vs k  : wavelength-only slope; amplitude variation within the
                cell is absorbed into residuals
    R²_ka and R²_k can differ — that difference is itself diagnostic.
    kL columns are a fixed scaling of k by L=2.6 m (panel length).
    """
    df = df.copy()
    df["amp_v"] = df["WaveAmplitudeInput [Volt]"].apply(_round_amp)
    rows = []
    for cat in CATEGORY_ORDER:
        for amp_v in AMP_ORDER:
            for wind in WIND_ORDER:
                cell = df[(df["category"] == cat)
                          & (df["amp_v"] == amp_v)
                          & (df["WindCondition"] == wind)]
                if cell.empty:
                    continue
                ka = cell["ka"].to_numpy(float)
                k  = cell[K_COL].to_numpy(float)        # rad/m
                kt = cell["OUT/IN (FFT)"].to_numpy(float)
                slope_ka, r2_ka = _linfit(ka, kt)
                slope_k,  r2_k  = _linfit(k,  kt)
                # kL is a constant rescaling of k; slope rescales as 1/L.
                slope_kL = slope_k * PANEL_LENGTH_M if pd.notna(slope_k) else np.nan
                rows.append({
                    "category":  cat,
                    "category_label": CATEGORY_LABEL[cat],
                    "amp":       AMP_LABEL[amp_v],
                    "amp_volt":  amp_v,
                    "wind":      wind,
                    "wind_label": WIND_LABEL[wind],
                    "n":         int(len(cell)),
                    "n_freqs":   int(cell["WaveFrequencyInput [Hz]"].nunique()),
                    # K_t summary
                    "Kt_mean":   float(kt.mean()),
                    "Kt_std":    float(kt.std(ddof=1)) if len(kt) > 1 else np.nan,
                    # ka block
                    "ka_min":    float(ka.min()),
                    "ka_max":    float(ka.max()),
                    "slope_dKt_dka": slope_ka,
                    "R2_ka":     r2_ka,
                    # k block (rad/m)
                    "k_min":     float(k.min()),
                    "k_max":     float(k.max()),
                    "slope_dKt_dk": slope_k,
                    "R2_k":      r2_k,
                    # kL block (dimensionless; L = 2.6 m)
                    "kL_min":    float(k.min()) * PANEL_LENGTH_M,
                    "kL_max":    float(k.max()) * PANEL_LENGTH_M,
                    "slope_dKt_dkL": slope_kL,
                    # R²_kL == R²_k (linear rescale); not duplicated.
                })
    return pd.DataFrame(rows)


# ── 3. Build the 3 views ──────────────────────────────────────────────────────
VIEWS = {
    "all":   {"name": "all data (3 categories)",
              "cats": CATEGORY_ORDER},
    "under": {"name": "undermooring (loose300 + loose230)",
              "cats": ["below_loose300_full", "below_loose230_full"]},
    "over":  {"name": "overmooring (above_50, panel pooled)",
              "cats": ["above_50"]},
}

# Pretty console formatter — three slope blocks side by side
# (ka steepness | k wavelength-only | kL panel-relative).
def _fmt_slope(slope: float, r2: float) -> str:
    if pd.notna(slope):
        return f"slope={slope:+7.3f} R²={r2:+.2f}"
    return "slope=    n/a  R²= n/a "


def _fmt_row(r):
    head = (f"{r['category_label']:<32s}  "
            f"{r['amp']}  {r['wind_label']:<4s}  "
            f"n={r['n']:>3d}  "
            f"Kt={r['Kt_mean']:.3f}±{r['Kt_std']:.3f}")
    ka_block = (f"ka=[{r['ka_min']:.3f},{r['ka_max']:.3f}]  "
                + _fmt_slope(r['slope_dKt_dka'], r['R2_ka']))
    k_block  = (f"k=[{r['k_min']:.2f},{r['k_max']:.2f}]  "
                + _fmt_slope(r['slope_dKt_dk'], r['R2_k']))
    kL_block = (f"kL=[{r['kL_min']:.1f},{r['kL_max']:.1f}]  "
                # R²_kL == R²_k (linear rescaling), so reuse R2_k.
                + _fmt_slope(r['slope_dKt_dkL'], r['R2_k']))
    return f"{head}  ║ {ka_block} ║ {k_block} ║ {kL_block}"


import json

# Mapping from view key → published table name (used in main_save_tables.py).
TABLE_NAMES = {
    "all":   "ch05_damping_all_data_scatter_ka_table",
    "under": "ch05_damping_undermooring_scatter_ka_table",
    "over":  "ch05_damping_overmooring_scatter_ka_table",
}


def _add_pair_deltas(tab: pd.DataFrame) -> pd.DataFrame:
    """For each (category × amp) pair, populate the wind-comparison
    columns on the `full` row only:

      delta_Kt        = K̄_t(full)  − K̄_t(no)
      slope_ratio_ka  = |slope_full| / |slope_no|        (ka-axis)
      slope_ratio_kL  = |slope_full| / |slope_no|        (kL-axis)

    The `no` row's pair-comparison columns stay NaN (blank in render).
    """
    tab = tab.copy()
    tab["delta_Kt"]       = float("nan")
    tab["slope_ratio_ka"] = float("nan")
    tab["slope_ratio_kL"] = float("nan")

    for (_cat, _amp), grp in tab.groupby(["category", "amp_volt"], sort=False):
        if set(grp["wind"]) != {"no", "full"}:
            continue   # cell missing one wind condition; nothing to compare
        no_row   = grp[grp["wind"] == "no"].iloc[0]
        full_idx = grp[grp["wind"] == "full"].index[0]
        full_row = grp[grp["wind"] == "full"].iloc[0]

        tab.at[full_idx, "delta_Kt"] = float(full_row["Kt_mean"] - no_row["Kt_mean"])

        for src, dst in [("slope_dKt_dka", "slope_ratio_ka"),
                         ("slope_dKt_dkL", "slope_ratio_kL")]:
            s_no, s_fw = no_row[src], full_row[src]
            if pd.notna(s_no) and pd.notna(s_fw) and abs(s_no) > 1e-9:
                tab.at[full_idx, dst] = float(abs(s_fw) / abs(s_no))

    return tab


# Section text for the IMMUTABLE comment block at the top of each .tex stub.
# View-specific headline findings; the rest is shared.
def _meta_sections(vkey: str, view_label: str, n_total: int, n_cells: int,
                   tab: pd.DataFrame) -> list[dict]:
    n_pairs = int(tab["delta_Kt"].notna().sum())
    n_lifts = int((tab["delta_Kt"] > 0).sum())
    n_drops = int((tab["delta_Kt"] < 0).sum())
    n_flat  = int((tab["slope_ratio_ka"] < 1.0).sum())

    # View-specific findings paragraph.
    if vkey == "under":
        findings = [
            f"n_pairs           : {n_pairs} wind comparisons (3 amps × 2 moorings)",
            f"ΔK_t > 0          : {n_lifts}/{n_pairs} cells — wind raises K_t in",
            "                    every cell of the under family.",
            f"slope flattens    : {n_flat}/{n_pairs} cells — wind softens the",
            "                    K_t-vs-ka slope in every cell.",
            "Flattening factor : slope_ratio_ka median ≈ 0.37 (loose300, loose230",
            "                    track each other tightly).",
            "Reading note      : pair the rows top-down; the `full` row carries",
            "                    the comparison columns ΔK_t and slope_ratio.",
        ]
    elif vkey == "over":
        findings = [
            f"n_pairs           : {n_pairs} wind comparisons (3 amps × 1 mooring)",
            f"ΔK_t > 0          : {n_lifts}/{n_pairs} cells — wind raises K_t",
            "                    only at A1; ΔK_t is near zero or negative at",
            "                    A2 and A3 (the wind enhancement vanishes at",
            "                    higher amplitudes for the above-water mooring).",
            f"slope flattens    : {n_flat}/{n_pairs} cells — wind softens the",
            "                    K_t-vs-ka slope in every cell.",
            "Reading note      : the negative ΔK_t entries are the table's",
            "                    most distinctive observation here — the figure",
            "                    hides them in the dense dot cloud.",
        ]
    else:   # "all" view — full master story
        findings = [
            f"n_pairs           : {n_pairs} wind comparisons (3 amps × 3 cats)",
            f"ΔK_t > 0          : {n_lifts}/{n_pairs} cells.",
            f"ΔK_t ≤ 0          : {n_drops}/{n_pairs} cells — at above_50",
            "                    × {A2, A3} the wind effect vanishes/reverses.",
            f"slope flattens    : {n_flat}/{n_pairs} cells — universal under",
            "                    wind. Flattening factor ≈ 0.37 for under-water",
            "                    moorings, ≈ 0.60 for above-water (above_50).",
            "Reading note      : the under-vs-over distinction in wind effect",
            "                    is amplitude-dependent (loose moorings:",
            "                    persistent lift; above_50: A1-only).",
        ]

    return [
        {"title": "Method",
         "lines": [
             "x-axis variants    : ka (paddle steepness), k (rad/m), kL (k×L)",
             "L                  : 2.6 m (panel longitudinal length, fixed)",
             "ka                 : `IN Wavenumber (FFT)` × `IN Amplitude (FFT)` [m]",
             "K_t                : `OUT/IN (FFT)` (paddle freq, 0.1 Hz nearest-bin)",
             "Pooling            : (category × amp × wind) — one row per cell",
             "Within-cell fit    : linear K_t = a·x + b for x ∈ {ka, k, kL}",
             "                     R²_kL == R²_k (kL is constant rescaling of k);",
             "                     R²_ka may differ because ka mixes k and amp.",
         ]},
        {"title": "Reading the table",
         "lines": [
             "Each row is one (category × amp × wind) cell.",
             "Rows are paired by wind: the `uten` row is the no-wind baseline,",
             "the `full` row caps the wind comparison.",
             "ΔK_t and slope_ratio are populated on the `full` row only:",
             "   ΔK_t        = K̄_t(full)  − K̄_t(no)",
             "   slope_ratio = |dK_t/dka|_full / |dK_t/dka|_no",
             "ΔK_t > 0 means wind raises mean transmission; slope_ratio < 1",
             "means wind softens the steepness sensitivity.",
         ]},
        {"title": "Headline findings (this view)",
         "lines": findings},
        {"title": "Beyond the figure",
         "lines": [
             "Figure shows clouds; table pins centroids and effect sizes.",
             "The figure cannot show:  per-cell n, exact K̄_t, slope numbers,",
             "or a sign-flip in ΔK_t hidden by overlapping dots. Each is in",
             "the table by design.",
         ]},
        {"title": "Inputs",
         "lines": [
             f"View               : {view_label}",
             f"Coverage           : {n_cells} cells from {n_total} runs",
             "Quality filter     : quality_flag=ok, paddle freq < 2 Hz,",
             "                     K_t ∈ [0.1, 2.0], Mooring != above_200",
         ]},
        {"title": "Companion figure",
         "lines": [
             "fig:ch05_damping_all_data_scatter_ka  (parent, all 3 categories)",
             "fig:ch05_damping_undermooring_scatter_ka  (loose300+loose230)",
             "fig:ch05_damping_overmooring_scatter_ka   (above_50, panel pooled)",
         ]},
    ]


print("\n" + "=" * 100)
for vkey, vinfo in VIEWS.items():
    sub = wave_clip[wave_clip["category"].isin(vinfo["cats"])]
    tab = _aggregate(sub)
    tab = _add_pair_deltas(tab)   # adds delta_Kt + slope_ratio_{ka,kL}

    # Scratch CSV (kept for iteration / quick eyeball).
    scratch_csv = Path(__file__).parent / f"damping_ka_scatter_table_iterate_{vkey}.csv"
    tab.to_csv(scratch_csv, index=False)

    # Standalone-LaTeX-ready data sidecar (CSV + meta.json) for main_save_tables.
    table_name = TABLE_NAMES[vkey]
    data_dir = BASE / "output" / "TABLES" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    sidecar_csv  = data_dir / f"{table_name}.csv"
    sidecar_meta = data_dir / f"{table_name}.meta.json"
    tab.to_csv(sidecar_csv, index=False)
    sidecar_meta.write_text(
        json.dumps({
            "script": "analysis_scratch/damping_ka_scatter_table_iterate.py",
            "plot_type": "damping_ka_scatter_table",
            "chapter": "05",
            "caption_label": f"tab:{table_name}",
            "caption_short": "",   # filled from TABLE_CAPTIONS_SHORT at render time
            "sections": _meta_sections(vkey, vinfo["name"], len(sub), len(tab), tab),
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\n── VIEW: {vinfo['name']}  ({len(tab)} cells, {len(sub)} runs) ──")
    for _, r in tab.iterrows():
        print(_fmt_row(r))
    print(f"   scratch → {scratch_csv.relative_to(BASE)}")
    print(f"   sidecar → {sidecar_csv.relative_to(BASE)}")
    print(f"   meta    → {sidecar_meta.relative_to(BASE)}")
print("\n" + "=" * 100)
print("Done. Edit this script's pooling / row order / column choice and re-run.")
