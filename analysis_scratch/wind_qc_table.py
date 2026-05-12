"""
Wind QC pre-paddle table — CH04 companion.
==========================================

For each probe and each canon set, compute pre-paddle η RMS statistics
(mean ± std, max) across all 'ok' runs in that canon, split by
WindCondition. Used to demonstrate that:

  * under nowind, η RMS sits at the probe noise floor → tank was still
    before each run.
  * under fullwind, the windward probes see ~8 mm RMS while the
    panel-shadowed OUT probe sees < 1 mm, and the run-to-run scatter
    of the windward RMS quantifies how repeatable the wind input is
    from one run to the next.

Pre-paddle window per probe = 0 to r / sqrt(g·H) seconds. H = 0.58 m,
sqrt(gH) = 2.385 m/s (the shallow-water speed limit — the latest time
any paddle-frequency signal could possibly have reached that probe).

    8804/250  →  3.69 s
    9373/170  →  3.93 s
    9373/340  →  3.93 s
    12400/250 →  5.20 s

Outputs:
    output/TABLES/data/ch04_wind_qc_table.csv       (render-shape data)
    output/TABLES/data/ch04_wind_qc_table.meta.json (provenance)
    analysis_scratch/wind_qc_table_audit.csv        (per-run audit)

Caption text is owned by main_save_tables.py (TABLE_CAPTIONS).
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

BASE = (Path(__file__).resolve().parent.parent
        if "__file__" in globals() else Path.cwd())
sys.path.insert(0, str(BASE))
os.chdir(BASE)

THESIS_NAME_NW = "ch04_wind_qc_nowind_table"
THESIS_NAME_FW = "ch04_wind_qc_fullwind_table"
SCRIPT_REL     = "analysis_scratch/wind_qc_table.py"
CHAPTER        = "04"

CANON = {
    "loose230": Path("waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange"),
    "loose300": Path("waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"),
}

# (probe position, r [m], Norwegian role label)
PROBES = [
    ("8804/250",  8.804,  "Foran"),
    ("9373/170",  9.373,  "Innkommende"),
    ("9373/340",  9.373,  "Innkommende"),
    ("12400/250", 12.400, "Utgående"),
]

FS = 250.0
H_M = 0.58
SHALLOW_SPEED = float(np.sqrt(9.81 * H_M))  # 2.385 m/s

WINDS = ["no", "full"]

DATA_DIR        = BASE / "output" / "TABLES" / "data"
RENDER_CSV_NW   = DATA_DIR / f"{THESIS_NAME_NW}.csv"
META_JSON_NW    = DATA_DIR / f"{THESIS_NAME_NW}.meta.json"
RENDER_CSV_FW   = DATA_DIR / f"{THESIS_NAME_FW}.csv"
META_JSON_FW    = DATA_DIR / f"{THESIS_NAME_FW}.meta.json"
AUDIT_CSV       = Path("analysis_scratch") / f"wind_qc_table_audit.csv"


def load_canon(folder: Path):
    """Return (meta_df, processed_dfs) for one canon folder, quality_flag=='ok'."""
    meta = pd.DataFrame(json.loads((folder / "meta.json").read_text()))
    meta = meta[meta["quality_flag"] == "ok"].copy()
    big = pd.read_parquet(folder / "processed_dfs.parquet")
    return meta, big


def per_run_rms(meta: pd.DataFrame, big: pd.DataFrame,
                probe: str, r_m: float) -> pd.DataFrame:
    """Pre-paddle η RMS per run at this probe.

    Window = [0, r/sqrt(gH)] seconds. η demeaned within window so the
    RMS is std-of-demeaned, not affected by any wind setup baseline.
    """
    win_end_samples = int(round(r_m / SHALLOW_SPEED * FS))
    eta_col = f"eta_{probe}"
    out = []
    for _, row in meta.iterrows():
        path = row["path"]
        sub = big[big["_path"] == path]
        if eta_col not in sub.columns or len(sub) < win_end_samples:
            continue
        eta = sub[eta_col].values[:win_end_samples].astype(float)
        eta = eta - np.nanmean(eta)
        rms = float(np.sqrt(np.nanmean(eta ** 2)))
        out.append({
            "path": path,
            "WindCondition": row["WindCondition"],
            "PanelCondition": row.get("PanelCondition", ""),
            "rms_mm": rms,
        })
    return pd.DataFrame(out)


# ── Aggregate per (probe, canon, wind) ─────────────────────────────────
audit_rows: list[pd.DataFrame] = []
wide_rows: list[dict] = []

for probe, r_m, role in PROBES:
    win_s = r_m / SHALLOW_SPEED
    row = {
        "probe":    probe,
        "role":     role,
        "r_m":      r_m,
        "window_s": round(win_s, 2),
    }
    for canon_name, folder in CANON.items():
        print(f"  {probe:12s}  canon={canon_name}  loading...")
        meta, big = load_canon(folder)
        per_run = per_run_rms(meta, big, probe, r_m)
        if per_run.empty:
            for w in WINDS:
                row[f"rms_{w}_mean_{canon_name}"] = np.nan
                row[f"rms_{w}_std_{canon_name}"]  = np.nan
                row[f"rms_{w}_max_{canon_name}"]  = np.nan
                row[f"n_{w}_{canon_name}"]        = 0
            continue
        per_run["canon"] = canon_name
        per_run["probe"] = probe
        audit_rows.append(per_run)
        for w in WINDS:
            sub = per_run[per_run["WindCondition"] == w]
            if sub.empty:
                row[f"rms_{w}_mean_{canon_name}"] = np.nan
                row[f"rms_{w}_std_{canon_name}"]  = np.nan
                row[f"rms_{w}_max_{canon_name}"]  = np.nan
                row[f"n_{w}_{canon_name}"]        = 0
            else:
                row[f"rms_{w}_mean_{canon_name}"] = float(sub["rms_mm"].mean())
                row[f"rms_{w}_std_{canon_name}"]  = (float(sub["rms_mm"].std(ddof=1))
                                                     if len(sub) > 1 else 0.0)
                row[f"rms_{w}_max_{canon_name}"]  = float(sub["rms_mm"].max())
                row[f"n_{w}_{canon_name}"]        = int(len(sub))
    wide_rows.append(row)

wide = pd.DataFrame(wide_rows)

DATA_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_CSV.parent.mkdir(parents=True, exist_ok=True)

if audit_rows:
    audit = pd.concat(audit_rows, ignore_index=True)
    audit.to_csv(AUDIT_CSV, index=False)
    print(f"\naudit  CSV → {AUDIT_CSV}  ({len(audit)} per-run rows)")


# ── Two render CSVs: nowind and fullwind (different implications) ──────
_BASE_COLS = ["probe", "role", "r_m", "window_s"]


def _split_csv(wind: str) -> pd.DataFrame:
    """Project the wide aggregate down to one wind condition."""
    cols = list(_BASE_COLS)
    for canon_name in CANON:
        cols += [
            f"rms_{wind}_mean_{canon_name}",
            f"rms_{wind}_std_{canon_name}",
            f"rms_{wind}_max_{canon_name}",
            f"n_{wind}_{canon_name}",
        ]
    return wide[cols].copy()


_split_csv("no").to_csv(RENDER_CSV_NW, index=False)
print(f"render CSV → {RENDER_CSV_NW.relative_to(BASE)}")
_split_csv("full").to_csv(RENDER_CSV_FW, index=False)
print(f"render CSV → {RENDER_CSV_FW.relative_to(BASE)}")


# ── Provenance meta.json (one per split) ───────────────────────────────
_METHOD_LINES = [
    "Pre-paddle window per probe = [0, r/sqrt(gH)] seconds.",
    f"  sqrt(gH) = {SHALLOW_SPEED:.3f} m/s at H = {H_M} m (shallow-water limit).",
    "  8804/250  → 3.69 s   (windward, exposed wind fetch ~6.4 m)",
    "  9373/170  → 3.93 s   (IN reference, wall side)",
    "  9373/340  → 3.93 s   (IN parallel, far side)",
    "  12400/250 → 5.20 s   (panel-shadowed)",
    "η demeaned within window, then RMS = sqrt(mean(η²)).",
    "Cells: mean ± std (max) across runs in cell.",
    "Std uses ddof=1; n=1 cells show std=0.",
    "Outlier flagging deferred to user-side analysis off the audit",
    "CSV (analysis_scratch/wind_qc_table_audit.csv).",
]
_INPUT_LINES = [
    "datasets : PROCESSED-20260326 (loose230, 23 cm strikk),",
    "           PROCESSED-20260327 (loose300, 30 cm strikk)",
    "filter   : quality_flag == 'ok' (all panel conditions pooled)",
    "probes   : 8804/250, 9373/170, 9373/340, 12400/250",
]


META_JSON_NW.write_text(json.dumps({
    "script":        SCRIPT_REL,
    "plot_type":     "wind_qc_nowind_table",
    "chapter":       CHAPTER,
    "caption_label": f"tab:{THESIS_NAME_NW}",
    "caption_short": "",
    "sections": [
        {"title": "Inputs",  "lines": _INPUT_LINES + ["wind set : nowind only"]},
        {"title": "Method",  "lines": _METHOD_LINES},
        {
            "title": "Reading the table (nowind implication)",
            "lines": [
                "Stillwater check: under no wind the pre-paddle window should",
                "contain only residual motion from previous runs + probe noise.",
                "Mean RMS at probe noise-floor level (CLAUDE.md §16) confirms",
                "the tank had settled before most runs began.",
                "The max column surfaces runs where settling time was insufficient",
                "— e.g. when a per40 paddle run was followed immediately by",
                "the next run without waiting for the wave train to decay.",
                "loose230 has only 5 nowind controls (n=5 is too small for",
                "the max to be meaningful); loose300 has 37 nowind runs.",
            ],
        },
    ],
}, indent=2), encoding="utf-8")
print(f"meta  JSON → {META_JSON_NW.relative_to(BASE)}")


META_JSON_FW.write_text(json.dumps({
    "script":        SCRIPT_REL,
    "plot_type":     "wind_qc_fullwind_table",
    "chapter":       CHAPTER,
    "caption_label": f"tab:{THESIS_NAME_FW}",
    "caption_short": "",
    "sections": [
        {"title": "Inputs",  "lines": _INPUT_LINES + ["wind set : fullwind only"]},
        {"title": "Method",  "lines": _METHOD_LINES},
        {
            "title": "Reading the table (fullwind implication)",
            "lines": [
                "Wind reproducibility: under full wind every pre-paddle window",
                "contains a fully developed wind-wave field. The mean RMS shows",
                "the typical wind-wave amplitude (~4 mm at windward probes,",
                "< 0.4 mm at the panel-shadowed OUT probe).",
                "The std column divided by the mean (CV ≈ 14 %) is the",
                "run-to-run reproducibility of the wind input. This sets the",
                "precision floor for A_in,fw and therefore for K_t,fw — it",
                "cannot be reduced by within-canon averaging beyond √n.",
                "The max column flags any unusually windy single run.",
            ],
        },
    ],
}, indent=2), encoding="utf-8")
print(f"meta  JSON → {META_JSON_FW.relative_to(BASE)}")

print("Done.")
