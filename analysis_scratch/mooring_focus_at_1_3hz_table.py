"""
Mooring + panelretning at 1.30 Hz — wind-effect table
======================================================

Companion to `analysis_scratch/mooring_focus_at_1_3hz_ka.py`.
Same data, same scope (1.30 Hz only, panels ∈ {full, reverse}, moorings ∈
{below_90, above_50}). Where the figure shows the points, this table
gives the hard numbers.

Sibling structure to `wind_effect_table_by_amp.py`:
  outer sort = amplitude  (A1 / A2 / A3, one row block each)
  inner sort = (panel, mooring), so each block has up to 3 rows:
    normal · below_90
    normal · above_50
    revers · above_50
  (revers · below_90 doesn't exist — reverse panel was never run on the
   below_90 mooring.)

Columns mirror the wind_effect tables: K_t per wind, ΔK_t in pp, K_t-økn %,
D-red %. Per-cell n shown alongside K_t since several cells have n ≤ 4.

Outputs:
    output/TABLES/data/ch05_mooring_focus_at_1_3hz_table.csv       (render-shape data)
    output/TABLES/data/ch05_mooring_focus_at_1_3hz_table.meta.json (provenance)
    output/TABLES/ch05_mooring_focus_at_1_3hz_table.tex            (thesis include)
    analysis_scratch/mooring_focus_at_1_3hz_table.csv              (audit-trail companion)

Caption text is read from FIGURE_CAPTIONS["ch05_mooring_focus_at_1_3hz_table"]
in main_save_figures.py via output/.figure_captions.json.
"""

import json
import os
import sys
import warnings
from pathlib import Path
import glob

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

BASE = (Path(__file__).resolve().parent.parent
        if "__file__" in globals() else Path.cwd())
sys.path.insert(0, str(BASE))
os.chdir(BASE)

from wavescripts.improved_data_loader import load_analysis_data
from wavescripts.plot_utils import amp_to_label

# ── I/O ────────────────────────────────────────────────────────────────────
SCRATCH_CSV = Path(__file__).parent / "mooring_focus_at_1_3hz_table.csv"
THESIS_NAME = "ch05_mooring_focus_at_1_3hz_table"
DATA_DIR    = BASE / "output" / "TABLES" / "data"
RENDER_CSV  = DATA_DIR / f"{THESIS_NAME}.csv"
META_JSON   = DATA_DIR / f"{THESIS_NAME}.meta.json"
OUT_TEX     = BASE / "output" / "TABLES" / f"{THESIS_NAME}.tex"
CHAPTER     = "05"
SCRIPT_REL  = "analysis_scratch/mooring_focus_at_1_3hz_table.py"

TARGET_FREQ  = 1.30
THESIS_AMPS  = [0.10, 0.20, 0.30]

# ── 1. Load + filter ───────────────────────────────────────────────────────
print("1. Loading all processed folders …")
all_dirs = sorted(glob.glob(str(BASE / "waveprocessed" / "PROCESSED-*")))
meta, _, _, _ = load_analysis_data(*all_dirs, load_processed=False)
print(f"   {len(meta)} rows total")

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
    & meta["WindCondition"].isin(["no", "full"])
    & (meta["OUT/IN (FFT)"].between(0.1, 2.0))
    & meta["moor_grp"].isin(["below_90", "above_50"])
].copy()
sel["amp_v"] = sel["WaveAmplitudeInput [Volt]"].apply(lambda v: round(float(v), 2))
print(f"   {len(sel)} rows at 1.30 Hz, panels ∈ {{full, reverse}}, "
      f"moor ∈ {{below_90, above_50}}")

# ── 2. Aggregate per (amp, panel, mooring, wind) — pool hardware ───────────
agg = (sel.groupby(["amp_v", "PanelCondition", "moor_grp", "WindCondition"])
          ["OUT/IN (FFT)"]
          .agg(["mean", "std", "count"])
          .reset_index()
          .rename(columns={"mean": "Kt", "std": "Kt_std", "count": "n",
                            "PanelCondition": "panel",
                            "WindCondition": "wind",
                            "moor_grp": "mooring"}))
print(f"\n   {len(agg)} (amp, panel, mooring, wind) cells")

# Pivot to one row per (amp, panel, mooring) with wind columns side-by-side.
pivot_K  = agg.pivot_table(index=["amp_v", "panel", "mooring"],
                            columns="wind", values="Kt", aggfunc="mean").reset_index()
pivot_n  = agg.pivot_table(index=["amp_v", "panel", "mooring"],
                            columns="wind", values="n",  aggfunc="sum").reset_index()
pivot_sd = agg.pivot_table(index=["amp_v", "panel", "mooring"],
                            columns="wind", values="Kt_std", aggfunc="mean").reset_index()

table = pivot_K.rename(columns={"no": "Kt_nw", "full": "Kt_fw"})
table["std_nw"] = pivot_sd["no"]
table["std_fw"] = pivot_sd["full"]
table["n_nw"]   = pivot_n["no"].astype("Int64")
table["n_fw"]   = pivot_n["full"].astype("Int64")
table["Delta_Kt"]  = table["Kt_fw"] - table["Kt_nw"]    # absolute change in K_t
table["ratio_Kt"]  = table["Kt_fw"] / table["Kt_nw"]    # K_t,vind / K_t,uten
_D_nw = 1.0 - table["Kt_nw"]
_D_fw = 1.0 - table["Kt_fw"]
table["ratio_D"]   = _D_fw / _D_nw                       # D_vind / D_uten,  D = 1 - K_t

# Drop cells where either wind condition is missing (no Δ to compute).
n_before = len(table)
table = table.dropna(subset=["Kt_nw", "Kt_fw"]).reset_index(drop=True)
n_dropped = n_before - len(table)
if n_dropped:
    print(f"   {n_dropped} cell(s) dropped — at least one wind missing")

# Sort: amplitude OUTER, then panel (normal first), then mooring
# (below_90 first → "canon" sits at top of each block).
PANEL_ORDER   = {"full": 0, "reverse": 1}
MOORING_ORDER = {"below_90": 0, "above_50": 1}
table["_p_ord"] = table["panel"].map(PANEL_ORDER)
table["_m_ord"] = table["mooring"].map(MOORING_ORDER)
table = table.sort_values(["amp_v", "_p_ord", "_m_ord"]).reset_index(drop=True)
table = table.drop(columns=["_p_ord", "_m_ord"])

print(f"   {len(table)} cells in final table\n")
print(table.round(3).to_string(index=False))

# ── 3. Save companion CSV (audit trail, full numeric columns) ──────────────
SCRATCH_CSV.parent.mkdir(parents=True, exist_ok=True)
table.to_csv(SCRATCH_CSV, index=False)
print(f"\n   audit CSV → {SCRATCH_CSV.relative_to(BASE)}")

# ── 4. Reshape into render-shape (one row per output table line) ───────────
# Same shape as the audit CSV — one row per (amp, panel, mooring) — but with
# the display-ready label columns added. Compound cells (Kt + n) stay as
# separate columns; the cell_format callable joins them into '0.658\,(23)'.
PANEL_LBL   = {"full": "normal", "reverse": "revers"}
MOORING_LBL = {"below_90": "below\\_90", "above_50": "above\\_50"}

render_rows: list[dict] = []
for _, r in table.iterrows():
    a = float(r["amp_v"])
    render_rows.append({
        "amp_v":             a,
        # Original script repeats the amp label on every row of a block, so
        # `amp_to_label(a)` is unconditional here too — keeps the rendered
        # cells byte-identical to the pre-migration baseline.
        "amp_label_display": amp_to_label(a),
        "panel":             r["panel"],
        "panel_label":       PANEL_LBL[r["panel"]],
        "mooring":           r["mooring"],
        "mooring_label":     MOORING_LBL[r["mooring"]],
        "Kt_nw":             float(r["Kt_nw"]),
        "Kt_fw":             float(r["Kt_fw"]),
        "n_nw":              int(r["n_nw"]) if pd.notna(r["n_nw"]) else None,
        "n_fw":              int(r["n_fw"]) if pd.notna(r["n_fw"]) else None,
        "Delta_Kt":          float(r["Delta_Kt"]),
        "ratio_Kt":          float(r["ratio_Kt"]),
        "ratio_D":           float(r["ratio_D"]),
    })

render_df = pd.DataFrame(render_rows)

DATA_DIR.mkdir(parents=True, exist_ok=True)
render_df.to_csv(RENDER_CSV, index=False)
print(f"   render CSV → {RENDER_CSV.relative_to(BASE)}")


# ── 5. Build provenance meta.json ──────────────────────────────────────────
# Caption text + caption_short are owned by main_save_tables.py — it patches
# meta.json's caption_short field after this script runs.
n_total_runs = int(table["n_nw"].fillna(0).sum() + table["n_fw"].fillna(0).sum())

meta_payload = {
    "script":          SCRIPT_REL,
    "plot_type":       "mooring_focus_at_1_3hz_table",
    "chapter":         CHAPTER,
    "caption_label":   f"tab:{THESIS_NAME}",
    "caption_short":   "",
    "sections": [
        {
            "title": "Filters",
            "lines": [
                "panel             : full, reverse",
                "wind              : no, full",
                "mooring           : below_90 (canon: loose230 + loose300),",
                "                    above_50",
                f"amplitude [V]     : {', '.join(f'{a:.1f}' for a in THESIS_AMPS)}",
                f"frequency [Hz]    : {TARGET_FREQ}",
                "quality_flag      : ok",
            ],
        },
        {
            "title": "Data provenance",
            "lines": [
                f"n_cells           : {len(table)}",
                f"n_runs (nw + fw)  : {n_total_runs}",
                "datasets        :",
                *[f"  {Path(d).name}" for d in all_dirs],
            ],
        },
        {
            "title": "Method",
            "lines": [
                "grouper           : groupby(amp, panel, mooring, wind) → mean K_t",
                "sort_order        : amplitude outer, then panel (normal first),",
                "                    then mooring (below_90 first within each panel).",
                "mooring pooling   : below_90 lumps loose230 + loose300; both panels",
                "                    pool hardware (cond4 + earlier) — see",
                "                    analysis_scratch/probe_config_bias_likeforlike.pdf",
                "                    for the bias check.",
                "metric_definitions:",
                "  K_t                 : OUT/IN(FFT), narrow 0.1 Hz window at f_paddle",
                "  Kt_nw / Kt_fw       : mean K_t at no- / full-wind",
                "  n_nw  / n_fw        : run count per (amp, panel, mooring, wind)",
                "  Delta_Kt            : Kt_fw - Kt_nw  (signed decimal fraction)",
                "  ratio_Kt            : Kt_fw / Kt_nw  (>1 = wind passes more wave)",
                "  ratio_D             : D_fw / D_nw    (D = 1 - K_t)",
                "                        <1 = wind erodes panel damping",
            ],
        },
    ],
}

META_JSON.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
print(f"   meta JSON  → {META_JSON.relative_to(BASE)}")

print("\nDone.")
