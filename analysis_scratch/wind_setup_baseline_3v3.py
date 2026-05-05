"""
Wind-setup at OUT via raw absolute baseline — 3-nowind-vs-3-fullwind.
======================================================================

Cleaner than the rampup/decay window method (which has a direction-dependent
asymmetry). Reads `Stillwater Probe 12400/250` from each run's meta — that's
the absolute ULS-reading baseline (mm; ULS reads distance DOWN to water
surface, so lower number = higher water level).

For each canon -lowrange folder: sort runs chronologically by file mtime,
find each nowind→fullwind transition, take the last 3 nowind runs before
the transition and the first 3 fullwind runs after, average each group.

    Δη_OUT (water rise under wind, mm) = mean(nowind) − mean(fullwind)

Positive = water at OUT rose during wind, as expected for a tank tilted
leeward by wind stress.

Output: stdout table only.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from wavescripts.improved_data_loader import load_analysis_data

BASE = Path("/Users/ole/Kodevik/wave_project")

# All March 2026 datasets — script silently skips those with no
# nowind→fullwind transition or no usable rows.
DATASETS = sorted(p.name for p in (BASE / "waveprocessed").glob("PROCESSED-202603*"))

OUT_COL = "Stillwater Probe 12400/250"


# Sanity range on the absolute baseline — a healthy ULS reading at OUT in our
# tank sits ~95–105 mm. Anything outside this is a probe-malfunction / clip
# event (e.g. one 200.050 mm outlier in 20260326) and is dropped.
SANE_RANGE_MM = (80.0, 120.0)


def chronological_meta(processed_dir: str) -> pd.DataFrame:
    target = str(BASE / "waveprocessed" / processed_dir)
    meta, _, _, _ = load_analysis_data(target, load_processed=False)
    m = meta.copy()
    m["mtime"] = m["path"].apply(
        lambda p: os.path.getmtime(p) if os.path.exists(p) else np.nan
    )
    m = m.sort_values("mtime").reset_index(drop=True)
    m["fname"] = m["path"].apply(lambda p: Path(p).name)
    return m


def restrict_to_canonical_states(m: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows where WindCondition is canonical ("no" or "full") AND
    OUT baseline reads sane. Preserves chronological order; transition
    files (fromZero…/fromMax… with WindCondition outside {no, full}) and
    malfunctioning rows are dropped so consecutive-state detection works."""
    keep = m["WindCondition"].isin({"no", "full"})
    keep &= m[OUT_COL].between(*SANE_RANGE_MM, inclusive="both")
    return m[keep].reset_index(drop=True)


def find_transitions(m: pd.DataFrame, before: str, after: str) -> list[int]:
    """Return indices i where m['WindCondition'][i-1] == before and
    m['WindCondition'][i] == after, AFTER restrict_to_canonical_states."""
    wc = m["WindCondition"].to_list()
    return [i for i in range(1, len(wc)) if wc[i - 1] == before and wc[i] == after]


def report_transition(m: pd.DataFrame, idx_after: int, n: int = 3) -> dict | None:
    """At transition `idx_after` (first 'after'-state row), take the last n
    'before'-state runs and first n 'after'-state runs, average OUT_COL."""
    wc = m["WindCondition"].to_list()
    before_state = wc[idx_after - 1]
    after_state  = wc[idx_after]

    pre_idx, i = [], idx_after - 1
    while i >= 0 and len(pre_idx) < n:
        if wc[i] == before_state:
            pre_idx.insert(0, i)
        else:
            break
        i -= 1

    post_idx, j = [], idx_after
    while j < len(wc) and len(post_idx) < n:
        if wc[j] == after_state:
            post_idx.append(j)
        else:
            break
        j += 1

    pre  = m.iloc[pre_idx]
    post = m.iloc[post_idx]
    pre_vals  = pre[OUT_COL].to_numpy()
    post_vals = post[OUT_COL].to_numpy()

    if len(pre_vals) < 1 or len(post_vals) < 1:
        return None

    # Timing per transition. mtime is the file-write time of the CSV (close
    # to end-of-recording for fresh files). The "sample" used in OUT_COL is
    # the first 1 s of each recording (per-run stillwater anchor) so the
    # mtime is a slight upper bound on when each sample was taken — for
    # ordering and inter-run intervals on the minute scale this is fine.
    return {
        "before_state": before_state,
        "after_state":  after_state,
        "pre_idx":      pre_idx,
        "post_idx":     post_idx,
        "pre_files":    pre["fname"].to_list(),
        "post_files":   post["fname"].to_list(),
        "pre_vals":     pre_vals,
        "post_vals":    post_vals,
        "pre_mean":     float(pre_vals.mean()),
        "post_mean":    float(post_vals.mean()),
        # Wind setup as water rise (mm). ULS reads distance DOWN to water, so
        # water rise = pre_mean − post_mean.
        "delta_water_rise_mm": float(pre_vals.mean() - post_vals.mean()),
        # mtime of the first run in the pre-block (= start of the 3 same-state
        # runs preceding the transition).
        "first_pre_mtime":   float(pre["mtime"].iloc[0]),
        # mtime of the last run in the pre-block (= the most-recent same-state
        # reading right before the wind state flipped).
        "last_pre_mtime":    float(pre["mtime"].iloc[-1]),
        # mtime of the first run in the post-block (= the first run AFTER the
        # wind flipped state — the "transition moment" in our data).
        "first_post_mtime":  float(post["mtime"].iloc[0]),
        # mtime of the last run in the post-block (= the last reading we
        # average into the post-side baseline).
        "last_post_mtime":   float(post["mtime"].iloc[-1]),
    }


all_results: list[dict] = []

for ds in DATASETS:
    m_full = chronological_meta(ds)
    if OUT_COL not in m_full.columns:
        continue
    m = restrict_to_canonical_states(m_full)
    has_no   = (m["WindCondition"] == "no").any()
    has_full = (m["WindCondition"] == "full").any()
    if not (has_no and has_full):
        continue   # no possible transitions on this day
    print(f"\n{'='*88}\n{ds}\n{'='*88}")
    print(f"   {len(m_full)} runs total  →  {len(m)} after restrict to canonical "
          f"WindCondition + OUT-baseline in {SANE_RANGE_MM} mm")

    # Both directions: nowind→fullwind (wind ON) and fullwind→nowind (wind OFF).
    for before, after, label in [("no", "full", "wind ON"),
                                  ("full", "no", "wind OFF")]:
        idxs = find_transitions(m, before, after)
        print(f"\n--- {label}  ({before} → {after})  : {len(idxs)} transitions ---")
        for ti, idx in enumerate(idxs):
            r = report_transition(m, idx, n=3)
            if r is None:
                print(f"  [transition #{ti+1} at row {idx}] no usable rows")
                continue
            print(f"\n  transition #{ti+1} @ row {idx}:")
            print(f"    last  {len(r['pre_idx'])} {before:<4} mean OUT baseline = "
                  f"{r['pre_mean']:.3f} mm  "
                  f"(values: {[f'{v:.3f}' for v in r['pre_vals']]})")
            print(f"    first {len(r['post_idx'])} {after:<4} mean OUT baseline = "
                  f"{r['post_mean']:.3f} mm  "
                  f"(values: {[f'{v:.3f}' for v in r['post_vals']]})")
            water_rise = r["delta_water_rise_mm"]
            sign = ("water RISES" if water_rise > 0 else
                    "water DROPS" if water_rise < 0 else "no change")
            print(f"    Δ (pre − post) = {water_rise:+.3f} mm  →  "
                  f"{sign} at OUT going {before} → {after}")

            # Timing line — three key timestamps + intervals.
            from datetime import datetime as _dt_fmt
            t1 = _dt_fmt.fromtimestamp(r["first_pre_mtime"]).strftime("%H:%M:%S")
            t2 = _dt_fmt.fromtimestamp(r["first_post_mtime"]).strftime("%H:%M:%S")
            t3 = _dt_fmt.fromtimestamp(r["last_post_mtime"]).strftime("%H:%M:%S")
            pre_span_s  = int(r["last_pre_mtime"]   - r["first_pre_mtime"])
            gap_s       = int(r["first_post_mtime"] - r["last_pre_mtime"])
            post_span_s = int(r["last_post_mtime"]  - r["first_post_mtime"])
            print(f"    timing: first pre={t1}, first post={t2}, last post={t3}  "
                  f"(pre span {pre_span_s}s, gap {gap_s}s, post span {post_span_s}s)")

            all_results.append({
                "dataset": ds, "label": label,
                "before": before, "after": after,
                # Row index in the chronologically-sorted, cleaned df —
                # smaller = earlier in the day. Lets downstream callers
                # (e.g. the appendix table) recover chronological order
                # within a dataset rather than wind-ON-first / wind-OFF-after.
                "transition_row_idx": idx,
                "n_pre": len(r["pre_idx"]), "n_post": len(r["post_idx"]),
                "pre_mean": r["pre_mean"], "post_mean": r["post_mean"],
                "water_rise_mm": water_rise,
                "magnitude_mm": abs(water_rise),
                # Timing — file mtimes (≈ end-of-recording timestamps).
                "first_pre_mtime":  r["first_pre_mtime"],
                "last_pre_mtime":   r["last_pre_mtime"],
                "first_post_mtime": r["first_post_mtime"],
                "last_post_mtime":  r["last_post_mtime"],
            })


# ── per-dataset summary across both directions ────────────────────────────
print(f"\n\n{'='*88}\nPer-dataset summary (magnitudes, both directions pooled)\n{'='*88}")
df_results = pd.DataFrame(all_results)
if not df_results.empty:
    df_results["date"] = df_results["dataset"].str.extract(r"PROCESSED-(\d{8})-")[0]
    summary = (df_results
               .groupby(["date", "dataset"])["magnitude_mm"]
               .agg(["count", "mean", "std", "min", "max"])
               .round(3)
               .reset_index())
    print(summary.to_string(index=False))

    csv_out = BASE / "analysis_scratch" / "wind_setup_baseline_3v3_results.csv"
    df_results.to_csv(csv_out, index=False)
    print(f"\n   CSV → {csv_out.relative_to(BASE)}")

print("\nDone.")
