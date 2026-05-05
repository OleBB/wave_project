"""
Cross-direction Δη check (2026-05-05).
=======================================

Follow-up to memory/finding_wind_setup_delta_eta_per_dataset.md, which observed
OUT-probe Δη differing between rampup (20260314, +0.78 mm) and decay (20260327,
+1.38 mm). The memo's TODO: run the same Δη calculation on a CROSS-DIRECTION
file from each dataset to discriminate setup-geometry vs day-to-day variability.

This script reuses the Δη definition from `wind_decay_timeseries.py::plot_run_zoom`
(commit 2026-05-04):

    Δη = mean(η, wind_window) − mean(η, zero_window)

with `zero_secs=2.0`, `wind_secs=5.0`, applied to whichever end of the recording
is settled given the run direction:

    fromZero…   →  zero=first 2 s,  wind=last 5 s
    fromMax…    →  zero=last 2 s,   wind=first 5 s

Files (q1=b: 6 total, 2 from 20260314 + 4 from 20260327):

    20260314 rampup (original observation): fullpanel-fromZeroWinToMaxWin-run1.csv
    20260314 decay  (cross-direction):      fullpanel-fromMaxWinToZeroWin-run1.csv
    20260327 rampup-A (cross-direction):    experimental-fromZeroToMaxWind-depth580-mstop30-run3.csv
    20260327 rampup-B (cross-direction):    experimental-fromZeroToMaxWin-depth580-mstop330-run1.csv
    20260327 decay-A (original observation): experimental-fromMaxToZeroWin-depth580-mstop30-run-endofday.csv
    20260327 decay-B (decay sibling):        experimental-fromMaxToZeroWin-depth580.csv

Output: stdout table only. Memo update is done by hand after reading the result.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from wavescripts.improved_data_loader import load_analysis_data, load_processed_dfs

FS = 250.0
BASE = Path("/Users/ole/Kodevik/wave_project")

ZERO_SECS = 2.0
WIND_SECS = 5.0
PROBES = ["9373/170", "12400/250"]   # IN, OUT

# Each entry: (dataset_tag, direction, processed_dir, run_csv_relative_to_wavedata)
# direction ∈ {"rampup", "decay"} maps to zero_kind ∈ {"first", "last"}.
DIR_20260314 = "PROCESSED-20260314-ProbePos4_31_FPV_2-tett6roof"
DIR_20260327 = "PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"

RUNS = [
    ("20260314", "rampup",
     DIR_20260314,
     "20260314-ProbePos4_31_FPV_2-tett6roof/fullpanel-fromZeroWinToMaxWin-run1.csv"),
    ("20260314", "decay",
     DIR_20260314,
     "20260314-ProbePos4_31_FPV_2-tett6roof/fullpanel-fromMaxWinToZeroWin-run1.csv"),
    ("20260327", "rampup-A",
     DIR_20260327,
     "20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange/"
     "experimental-fromZeroToMaxWind-depth580-mstop30-run3.csv"),
    ("20260327", "rampup-B",
     DIR_20260327,
     "20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange/"
     "experimental-fromZeroToMaxWin-depth580-mstop330-run1.csv"),
    ("20260327", "decay-A",
     DIR_20260327,
     "20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange/"
     "experimental-fromMaxToZeroWin-depth580-mstop30-run-endofday.csv"),
    ("20260327", "decay-B",
     DIR_20260327,
     "20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange/"
     "experimental-fromMaxToZeroWin-depth580.csv"),
]


def _zero_kind_for(direction: str) -> str:
    """rampup → 'first' (zero at start, wind at end);
    decay → 'last' (zero at end, wind at start)."""
    if direction.startswith("rampup"):
        return "first"
    if direction.startswith("decay"):
        return "last"
    raise ValueError(f"unknown direction {direction!r}")


# A wind-transition recording has whole-record std of order ~mm at every
# probe. A stillwater recording (or a misnamed wind file with no wind ever
# on) has OUT std *below* the OUT probe noise floor (~0.13 mm) since OUT
# averages out paddle waves too. The 20260327 'fromMaxToZeroWin-depth580.csv'
# file fits this pattern (std=0.028 mm); a weak-but-real wind recording like
# the canonical 20260327 endofday decay (std=0.234 mm) does not. Using 0.05 mm
# as the threshold cleanly excludes the degenerate file while keeping every
# real wind transition.
DEGENERATE_OUT_STD_THRESHOLD_MM = 0.05


def compute_delta_eta(df, probe: str, zero_kind: str,
                      zero_secs: float = ZERO_SECS,
                      wind_secs: float = WIND_SECS
                      ) -> tuple[float, float, float, float]:
    """Returns (Δη, baseline, wind_baseline, full_record_std) in mm."""
    col = f"eta_{probe}_interp" if f"eta_{probe}_interp" in df.columns else f"eta_{probe}"
    if col not in df.columns:
        return float("nan"), float("nan"), float("nan"), float("nan")

    eta = df[col].to_numpy(dtype=float)
    t   = np.arange(len(eta)) / FS
    T   = float(t[-1])

    if zero_kind == "first":
        zero_mask = (t >= 0.0) & (t <= zero_secs)
        wind_mask = (t >= T - wind_secs) & (t <= T)
    else:  # "last"
        zero_mask = (t >= T - zero_secs) & (t <= T)
        wind_mask = (t >= 0.0) & (t <= wind_secs)

    baseline      = float(np.nanmean(eta[zero_mask]))
    wind_baseline = float(np.nanmean(eta[wind_mask]))
    full_std      = float(np.nanstd(eta))
    return wind_baseline - baseline, baseline, wind_baseline, full_std


# ── load both processed dirs once each ─────────────────────────────────────
_proc_cache: dict[str, dict] = {}

def _proc(dir_name: str) -> dict:
    if dir_name not in _proc_cache:
        target = str(BASE / "waveprocessed" / dir_name)
        load_analysis_data(target, load_processed=False)
        _proc_cache[dir_name] = load_processed_dfs(target)
    return _proc_cache[dir_name]


# ── compute ────────────────────────────────────────────────────────────────
print(f"\nΔη cross-direction check  —  zero_secs={ZERO_SECS}, wind_secs={WIND_SECS}")
print(f"   degenerate OUT-std cutoff (likely stillwater): < {DEGENERATE_OUT_STD_THRESHOLD_MM} mm")
print(f"{'='*108}")
header = (f"{'dataset':<12} {'direction':<10} {'IN [mm]':>10} {'OUT [mm]':>10} "
          f"{'OUT std':>10}  {'csv':<48}")
print(header)
print("-" * 108)

results: list[dict] = []
for tag, direction, proc_dir, run_rel in RUNS:
    proc = _proc(proc_dir)
    run_csv_abs = str(BASE / "wavedata" / run_rel)
    if run_csv_abs not in proc:
        print(f"{tag:<12} {direction:<10}     ?     ?  MISSING from processed cache:")
        print(f"             {run_rel}")
        continue
    df = proc[run_csv_abs]
    zk = _zero_kind_for(direction)

    d_in,  _, _, std_in  = compute_delta_eta(df, "9373/170",  zk)
    d_out, _, _, std_out = compute_delta_eta(df, "12400/250", zk)

    degenerate = (not np.isnan(std_out)) and std_out < DEGENERATE_OUT_STD_THRESHOLD_MM

    results.append({
        "dataset": tag, "direction": direction,
        "delta_in": d_in, "delta_out": d_out,
        "std_out": std_out, "degenerate": degenerate,
        "csv_short": Path(run_rel).name,
    })

    in_str  = "  n/a" if np.isnan(d_in)  else f"{d_in:+.2f}"
    out_str = "  n/a" if np.isnan(d_out) else f"{d_out:+.2f}"
    std_str = "  n/a" if np.isnan(std_out) else f"{std_out:.3f}"
    flag    = "  ← degenerate (likely stillwater)" if degenerate else ""
    print(f"{tag:<12} {direction:<10} {in_str:>10} {out_str:>10} {std_str:>10}  "
          f"{Path(run_rel).name:<48}{flag}")

print("-" * 108)


# ── per-dataset summary (OUT probe is the clean wind-setup metric) ─────────
print("\nPer-dataset OUT-probe Δη summary (degenerate runs excluded):")
for tag in ("20260314", "20260327"):
    sub = [r for r in results if r["dataset"] == tag and not r["degenerate"]]
    vals = [r["delta_out"] for r in sub if not np.isnan(r["delta_out"])]
    if not vals:
        continue
    arr = np.array(vals)
    spread = arr.max() - arr.min() if len(arr) > 1 else 0.0
    print(f"   {tag}:  n={len(arr)}  values=[{', '.join(f'{v:+.2f}' for v in arr)}] mm"
          f"   mean={arr.mean():+.2f}  spread={spread:.2f} mm")

# Cross-dataset comparison verdict — degenerate runs excluded
out_2026_03_14 = [r["delta_out"] for r in results
                  if r["dataset"] == "20260314" and not r["degenerate"]
                  and not np.isnan(r["delta_out"])]
out_2026_03_27 = [r["delta_out"] for r in results
                  if r["dataset"] == "20260327" and not r["degenerate"]
                  and not np.isnan(r["delta_out"])]
if out_2026_03_14 and out_2026_03_27:
    m14, m27 = np.mean(out_2026_03_14), np.mean(out_2026_03_27)
    sp14 = (np.max(out_2026_03_14) - np.min(out_2026_03_14)) if len(out_2026_03_14) > 1 else 0.0
    sp27 = (np.max(out_2026_03_27) - np.min(out_2026_03_27)) if len(out_2026_03_27) > 1 else 0.0
    cross_gap = abs(m14 - m27)
    biggest_within = max(sp14, sp27)
    print(f"\n   between-dataset gap (|Δmean|) = {cross_gap:.2f} mm")
    print(f"   largest within-dataset spread = {biggest_within:.2f} mm "
          f"(20260314 spread {sp14:.2f}, 20260327 spread {sp27:.2f})")
    if biggest_within > 0 and cross_gap < biggest_within:
        verdict = ("→ within-dataset spread ≥ between-dataset gap "
                   "⇒ day-to-day / run-to-run variability dominates.")
    elif biggest_within > 0 and cross_gap >= 2.0 * biggest_within:
        verdict = ("→ between-dataset gap ≫ within-dataset spread "
                   "⇒ dataset/setup geometry dominates.")
    else:
        verdict = "→ inconclusive (between-dataset gap and within-dataset spread are similar)."
    print(f"   {verdict}")

print("\nDone.")
