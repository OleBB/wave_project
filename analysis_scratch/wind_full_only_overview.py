"""
Quick wind-field reconnaissance — meta-only, no time-series, no FFT.

Filters combined_meta for *steady* full-wind nowave runs (paddle off, fan at
full power for the whole recording — NOT ramp/decay) and tabulates the
already-cached per-probe time-domain amplitude (P99.5−P0.5)/2 in mm.

That column is the wind-chop envelope per probe (no paddle tone present),
so it directly answers:
  • Is the wind field reproducible across runs on the same day?
  • Did wind energy change between sessions?
  • What's the spatial pattern (upstream / IN / parallel / OUT)?
  • Wall-side vs far-side lateral asymmetry at r=9373 mm?

Outputs:
    analysis_scratch/wind_full_only_overview.csv         — one row per run
    analysis_scratch/wind_full_only_overview_summary.csv — grouped by (date, mooring, height)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from wavescripts.improved_data_loader import load_analysis_data

BASE = Path("/Users/ole/Kodevik/wave_project")
DATASET_DIRS = sorted(BASE.glob("waveprocessed/PROCESSED-*"))

# Probe set & friendly labels
PROBES = ["8804/250", "9373/170", "9373/340", "12400/250"]
LABELS = {
    "8804/250":  "8804/250 (upstream)",
    "9373/170":  "9373/170 (IN, wall)",
    "9373/340":  "9373/340 (IN, far)",
    "12400/250": "12400/250 (OUT)",
}

# ── Load light tier (combined_meta only) ──────────────────────────────────
print(f"Loading {len(DATASET_DIRS)} processed datasets …")
meta, _, _, _ = load_analysis_data(
    *[str(d) for d in DATASET_DIRS], load_processed=False,
)
print(f"  → {len(meta)} runs in combined_meta")

# ── Filter: steady fullwind, no paddle, no ramp ──────────────────────────
# 1. Paddle off    → WaveFrequencyInput is NaN
# 2. Fan steady    → WindCondition == "full"
# 3. No ramp/decay → exclude any path with frommax / fromzero / frommin
# 4. Trustworthy   → quality_flag == "ok"
ramp_pat = r"(?i)frommax|fromzero|frommin"
sel = (
    (meta["WindCondition"] == "full") &
    (meta["WaveFrequencyInput [Hz]"].isna()) &
    (meta["quality_flag"] == "ok") &
    (~meta["path"].str.contains(ramp_pat, regex=True, na=False))
)
fw = meta.loc[sel].copy()
print(f"  → {len(fw)} steady-fullwind nowave runs after filter")
if len(fw) == 0:
    raise SystemExit("No runs match the filter — nothing to tabulate.")

# ── Build per-run table ───────────────────────────────────────────────────
# Pull the time-domain amplitude (P99.5−P0.5)/2 for each probe; columns
# missing for a given dataset's probe config show up as NaN.
amp_cols = {}
for p in PROBES:
    col = f"Probe {p} Amplitude"
    if col in fw.columns:
        amp_cols[p] = col
    else:
        # Some early datasets may not have this probe column at all.
        fw[col] = np.nan
        amp_cols[p] = col

# Run filename for human reading
fw["run_filename"] = fw["path"].str.split("/").str[-1]

# Recording duration (s) — useful for spotting unusually short or long runs.
# Inferred from sample count if the column exists; fallback "" otherwise.
if "n_samples" in fw.columns:
    fw["duration_s"] = (fw["n_samples"] / 250.0).round(1)
elif "Number of Samples" in fw.columns:
    fw["duration_s"] = (fw["Number of Samples"] / 250.0).round(1)
else:
    fw["duration_s"] = np.nan

per_run_cols = [
    "file_date", "Mooring", "probe_height_mm", "probe_range_mode",
    "PanelCondition", "duration_s",
    *[amp_cols[p] for p in PROBES],
    "run_filename",
]
per_run_cols = [c for c in per_run_cols if c in fw.columns]
per_run = fw[per_run_cols].copy()
per_run = per_run.sort_values(["file_date", "Mooring", "probe_height_mm",
                                "PanelCondition", "run_filename"])

# Round amplitudes to 0.01 mm — well below the 0.25 mm precision target,
# but keeps the spread readable.
for p in PROBES:
    c = amp_cols[p]
    if c in per_run.columns:
        per_run[c] = per_run[c].astype(float).round(2)

# ── Per-run print ─────────────────────────────────────────────────────────
print("\n=== STEADY FULLWIND NOWAVE — PER RUN ===")
with pd.option_context("display.width", 220,
                       "display.max_columns", 50,
                       "display.max_colwidth", 70):
    print(per_run.to_string(index=False))

per_run_csv = Path(__file__).parent / "wind_full_only_overview.csv"
per_run.to_csv(per_run_csv, index=False)
print(f"\n  → {per_run_csv.relative_to(BASE)}")

# ── Aggregated summary by (date, mooring, height, panel) ──────────────────
group_keys = [k for k in
              ("file_date", "Mooring", "probe_height_mm", "PanelCondition")
              if k in per_run.columns]

agg = (
    per_run
    .groupby(group_keys, dropna=False)
    .agg(
        n_runs=("run_filename", "count"),
        **{
            f"A_{p}_mean": (amp_cols[p], "mean") for p in PROBES if amp_cols[p] in per_run.columns
        },
        **{
            f"A_{p}_std":  (amp_cols[p], "std")  for p in PROBES if amp_cols[p] in per_run.columns
        },
    )
    .reset_index()
)

# Round
for c in agg.columns:
    if c.startswith("A_"):
        agg[c] = agg[c].astype(float).round(2)

# Re-order: keys, n, then per probe (mean, std) interleaved
ordered = list(group_keys) + ["n_runs"]
for p in PROBES:
    for stat in ("mean", "std"):
        col = f"A_{p}_{stat}"
        if col in agg.columns:
            ordered.append(col)
agg = agg[ordered]

print("\n=== STEADY FULLWIND NOWAVE — GROUPED BY (date, mooring, height, panel) ===")
with pd.option_context("display.width", 220, "display.max_columns", 50):
    print(agg.to_string(index=False))

agg_csv = Path(__file__).parent / "wind_full_only_overview_summary.csv"
agg.to_csv(agg_csv, index=False)
print(f"\n  → {agg_csv.relative_to(BASE)}")

# ── Quick spatial check: OUT/IN, far/wall, IN-pair mean ──────────────────
def _div(a, b):
    return (a / b).where((b > 0) & a.notna() & b.notna())

per_run["A_IN_mean"] = per_run[[amp_cols["9373/170"], amp_cols["9373/340"]]].mean(axis=1)
per_run["OUT/IN_chop"]   = _div(per_run[amp_cols["12400/250"]], per_run["A_IN_mean"]).round(3)
per_run["wall/far"]      = _div(per_run[amp_cols["9373/170"]],
                                 per_run[amp_cols["9373/340"]]).round(3)
per_run["upstream/IN"]   = _div(per_run[amp_cols["8804/250"]], per_run["A_IN_mean"]).round(3)

print("\n=== SPATIAL RATIOS (per run) ===")
ratio_cols = ["file_date", "Mooring", "PanelCondition",
              "A_IN_mean", "OUT/IN_chop", "wall/far", "upstream/IN", "run_filename"]
ratio_cols = [c for c in ratio_cols if c in per_run.columns]
with pd.option_context("display.width", 220,
                       "display.max_columns", 50,
                       "display.max_colwidth", 70):
    print(per_run[ratio_cols].sort_values(
        ["file_date", "Mooring", "run_filename"]).to_string(index=False))

print("\nDone.")
