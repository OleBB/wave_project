"""
Pre-paddle wind summary table — CH04 §4q companion.
====================================================

Renders the four-probe summary that accompanies ch04_wind_pre_paddle_psd:

    probe          | long-run sigma | 3 s mean sigma | delta (%) | 3 s 1-sigma scatter

Source CSV (must exist; produced by analysis_scratch/wind_2s_vs_360s.py):
    analysis_scratch/wind_2s_vs_360s_stats_3s.csv
    analysis_scratch/wind_2s_vs_360s_per_long_run_3s.csv

Outputs:
    output/TABLES/data/ch04_wind_pre_paddle_table.csv       (render-shape data)
    output/TABLES/data/ch04_wind_pre_paddle_table.meta.json (provenance)
    analysis_scratch/wind_pre_paddle_table.csv              (audit-trail companion)

Caption text is owned by main_save_tables.py (TABLE_CAPTIONS).

Immutable block records:
  - which CSV was read
  - long-run reference set: 5 fullwind+nowave runs, durations 31, 33, 63,
    360, 381 s (only 2 of 5 are >= 360 s; phrase carefully in caption)
  - OUT-probe (12400/250) noise-floor caveat: long-run sigma 0.36 mm
    sits in the probe's measured stillwater-noise range (0.14-0.36 mm
    per CLAUDE.md §16); the -8.6 % delta vs the 3 s ensemble is
    consistent with sampling noise, NOT a window-length bias.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd

BASE = (Path(__file__).resolve().parent.parent
        if "__file__" in globals() else Path.cwd())
sys.path.insert(0, str(BASE))
os.chdir(BASE)


SRC_CSV   = Path("analysis_scratch/wind_2s_vs_360s_stats_3s.csv")
LONG_CSV  = Path("analysis_scratch/wind_2s_vs_360s_per_long_run_3s.csv")

THESIS_NAME = "ch04_wind_pre_paddle_table"
CHAPTER     = "04"
SCRIPT_REL  = "analysis_scratch/wind_pre_paddle_table.py"

DATA_DIR    = BASE / "output" / "TABLES" / "data"
RENDER_CSV  = DATA_DIR / f"{THESIS_NAME}.csv"
META_JSON   = DATA_DIR / f"{THESIS_NAME}.meta.json"
SCRATCH_CSV = Path("analysis_scratch/wind_pre_paddle_table.csv")

# Probes in row order (label as it appears in the CSV → display label).
PROBE_ORDER = [
    ("8804/250 (upstream)",   "8804/250 (upstream)"),
    ("9373/170 (IN, wall)",   "9373/170 (IN, wall)"),
    ("9373/340 (IN, far)",    "9373/340 (IN, far)"),
    ("12400/250 (OUT)",       "12400/250 (OUT)"),
]

# ── Read inputs ──────────────────────────────────────────────────────────
if not SRC_CSV.exists():
    raise SystemExit(
        f"Missing {SRC_CSV}. Run analysis_scratch/wind_2s_vs_360s.py first."
    )
stats = pd.read_csv(SRC_CSV)

if not LONG_CSV.exists():
    raise SystemExit(
        f"Missing {LONG_CSV}. Run analysis_scratch/wind_2s_vs_360s.py first."
    )
long_runs = pd.read_csv(LONG_CSV)
n_long_runs    = long_runs["long_run"].nunique()
long_durations = sorted(long_runs.drop_duplicates("long_run")["duration_s"].tolist())


# ── Pivot stats CSV into one row per probe with both segments ────────────
def _row(probe_label: str) -> dict:
    sub = stats[stats["probe"] == probe_label]
    long_row = sub[sub["segment"].str.startswith("long")].iloc[0]
    short_row = sub[sub["segment"].str.startswith("3 s")].iloc[0]
    sigma_long = float(long_row["sigma_mm"])
    sigma_3s   = float(short_row["sigma_mm"])
    sigma_3s_scatter = float(short_row["sigma_std"])
    delta_pct = 100.0 * (sigma_3s - sigma_long) / sigma_long
    return {
        "probe":               probe_label,
        "sigma_long_mm":       sigma_long,
        "sigma_3s_mm":         sigma_3s,
        "delta_pct":           delta_pct,
        "sigma_3s_scatter_mm": sigma_3s_scatter,
        "n_long":              int(long_row["n"]),
        "n_3s":                int(short_row["n"]),
    }


records = [_row(label) for label, _ in PROBE_ORDER]
n_long_seg = int(records[0]["n_long"])
n_3s       = int(records[0]["n_3s"])

# Audit-trail CSV — full per-probe breakdown, including counts.
SCRATCH_CSV.parent.mkdir(parents=True, exist_ok=True)
audit_df = pd.DataFrame(records)
audit_df.to_csv(SCRATCH_CSV, index=False)
print(f"audit CSV → {SCRATCH_CSV}")

# Render-shape CSV (same columns; the renderer reads the cell-format keys).
DATA_DIR.mkdir(parents=True, exist_ok=True)
audit_df.to_csv(RENDER_CSV, index=False)
print(f"render CSV → {RENDER_CSV.relative_to(BASE)}")


# ── Provenance meta.json ──────────────────────────────────────────────────
durations_str = ", ".join(f"{int(d)}" for d in long_durations)

meta_payload = {
    "script":          SCRIPT_REL,
    "plot_type":       "wind_pre_paddle_table",
    "chapter":         CHAPTER,
    "caption_label":   f"tab:{THESIS_NAME}",
    "caption_short":   "",
    "sections": [
        {
            "title": "Inputs",
            "lines": [
                f"stats CSV         : {SRC_CSV}",
                f"per-long-run CSV  : {LONG_CSV}",
                "datasets          : PROCESSED-20260326-*-lowrange,",
                "                    PROCESSED-20260327-*-lowrange (canon)",
                f"long-run set      : {n_long_runs} fullwind+nowave runs",
                f"long-run durations: {durations_str} s  (only 2 of {n_long_runs} are ≥ 360 s)",
                f"3 s ensemble      : {n_3s} fullwind+wave runs, one 3 s pre-paddle",
                "                    snippet each (paddle-free: √(gh)=2.39 m/s,",
                "                    closest probe at 8804 mm → safe to 3.68 s)",
            ],
        },
        {
            "title": "Method",
            "lines": [
                "sigma_eta = std(eta), Hs = 4 sigma_eta, mean = mean(eta).",
                "long-run column     : ensemble mean of per-run sigma over long runs",
                "3 s mean column     : ensemble mean of per-snippet sigma over 70 runs",
                "delta column        : (sigma_3s − sigma_long) / sigma_long, percent",
                "3 s 1σ scatter      : std (ddof=1) of per-snippet sigma across 70 runs",
            ],
        },
        {
            "title": "Caveats",
            "lines": [
                "OUT probe (12400/250):",
                "  long-run sigma = 0.36 mm; 3 s sigma = 0.33 mm; delta = -8.6 %.",
                "  The probe's measured stillwater noise floor is ~0.14 mm",
                "  (gold standard, CLAUDE.md §16) and ranges 0.14-0.36 mm",
                "  across settled stillwater runs. The -8.6 % delta sits inside",
                "  this noise-floor envelope, so it is consistent with sampling",
                "  noise, not a window-length bias. Caption should reflect this",
                "  when interpreting the OUT row.",
                "Long-run reference set: 5 runs total, durations vary 31-381 s",
                "  (median 63 s; only 2 of 5 are ≥ 360 s). Phrase as e.g. ",
                '  "five fullwind+nowave runs, durations 31-381 s" — not "5 × 360 s".',
            ],
        },
    ],
}

META_JSON.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
print(f"meta JSON  → {META_JSON.relative_to(BASE)}")

print("Done.")
