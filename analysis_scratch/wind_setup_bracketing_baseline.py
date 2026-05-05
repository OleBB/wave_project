"""
Bracketing-baseline drift at OUT (12400/250) — follow-up (2026-05-05).
======================================================================

Reuses analysis_scratch/wind_qc_3s_per_run.csv (produced by wind_qc_3s.py).
That CSV stores, for every "ok" run in the two canon -lowrange folders,
the per-probe mean and std of the FIRST 3 s of recording. The η in the
processed cache is already zeroed against a per-run stillwater anchor (the
first 1 s of THIS run by default — see processor.py), so:

  • For NOWIND-NOWAVE runs: the anchor is genuinely settled stillwater, so
    `mean_12400/250` ≈ first-3 s mean − first-1 s mean ≈ 0 by construction
    AT t=0 of that run, BUT we are comparing the same column across runs,
    each anchored separately. The CHRONOLOGICAL drift across nowind-nowave
    runs reflects the day's accumulated bias of "first 1 s anchor at run
    start" relative to the day's first stillwater anchor — i.e. how much
    the absolute water level drifts between settled states. This is the
    bracketing-Δ the question is about.

  • For FULLWIND-NOWAVE runs: the anchor is the first 1 s of THAT run,
    which is ALREADY wind-on. `mean_12400/250` is then the difference
    between two wind-on means (3 s vs 1 s of the same run) and is near
    zero BY CONSTRUCTION. It does NOT report the wind-setup magnitude.
    To recover absolute wind setup at OUT during a fullwind run, one
    would need the raw `Stillwater Probe 12400/250` baseline column from
    meta.json or a run anchored to a separate stillwater reference.
    Not in this script's scope.

So this script reports ONLY the nowind-nowave bracketing comparison (start
vs end of day) — the question the user asked. Fullwind-nowave numbers are
shown on the plot for completeness but are explicitly flagged as
self-anchored and not interpretable as wind setup.

Outputs:
    analysis_scratch/wind_setup_bracketing_baseline.png
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from wavescripts.plot_utils import apply_thesis_style, WIND_COLOR_MAP

apply_thesis_style()

BASE = Path("/Users/ole/Kodevik/wave_project")
CSV  = BASE / "analysis_scratch/wind_qc_3s_per_run.csv"
OUT  = BASE / "analysis_scratch/wind_setup_bracketing_baseline.png"

df = pd.read_csv(CSV)
df["dt"] = pd.to_datetime(df["run_mtime"], unit="s")
df = df.sort_values("run_mtime").reset_index(drop=True)

# Nowind-nowave (bracketing settled states).
nw = df[(df["WindCondition"] == "no") & (df["is_wave"] == False)].copy()
# Fullwind-nowave (in-wind baseline at OUT — first 3 s sees fully developed wind).
fw = df[(df["WindCondition"] == "full") & (df["is_wave"] == False)].copy()
# Lowwind-nowave (if any).
lo = df[(df["WindCondition"] == "lowest") & (df["is_wave"] == False)].copy()

print(f"\nBracketing-baseline drift at OUT (12400/250)\n{'='*62}")
print(f"Source: {CSV.relative_to(BASE)}\n")

print(f"Nowind-nowave runs:  n={len(nw)}")
print(nw[["file_date", "run_idx", "mean_12400/250", "sigma_12400/250"]]
      .to_string(index=False))

# Per-day bracket: first vs last nowind-nowave run on each day.
print("\nPer-day brackets (last − first nowind-nowave on each day):")
for d, sub in nw.groupby("file_date"):
    sub = sub.sort_values("run_mtime")
    if len(sub) < 2:
        print(f"   {d}: only n={len(sub)} nowind-nowave run — no bracket.")
        continue
    first, last = sub.iloc[0], sub.iloc[-1]
    delta = last["mean_12400/250"] - first["mean_12400/250"]
    hours = (last["run_mtime"] - first["run_mtime"]) / 3600.0
    print(f"   {d}: Δη_OUT (last − first) = {delta:+.3f} mm "
          f"over {hours:.1f} h, n_between={len(sub)-2}")

print(f"\nFullwind-nowave runs (FLAG: self-anchored, not wind setup):  n={len(fw)}")
print(f"   `mean_12400/250` here = mean(first 3 s, η-zeroed) where the η-zero")
print(f"   is the first 1 s of THE SAME RUN. Both windows are wind-on, so the")
print(f"   value is near zero by construction. It does NOT report wind setup.")
if len(fw):
    print(f"   range:  min={fw['mean_12400/250'].min():+.2f}  "
          f"max={fw['mean_12400/250'].max():+.2f}  "
          f"mean={fw['mean_12400/250'].mean():+.2f}  "
          f"std={fw['mean_12400/250'].std():.2f}  mm")

print(f"\nLowwind-nowave runs:  n={len(lo)}")
if len(lo):
    print(f"   range:  min={lo['mean_12400/250'].min():+.2f}  "
          f"max={lo['mean_12400/250'].max():+.2f}  mm")

# ── Plot ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(10, 6.0), sharex=False,
                         gridspec_kw={"hspace": 0.45})

date_offsets = {d: i for i, d in enumerate(sorted(df["file_date"].unique()))}

for ax, date in zip(axes, sorted(df["file_date"].unique())):
    sub = df[df["file_date"] == date].sort_values("run_mtime")
    sub_nw = sub[(sub["WindCondition"] == "no") & (sub["is_wave"] == False)]
    sub_fw = sub[(sub["WindCondition"] == "full") & (sub["is_wave"] == False)]
    sub_lo = sub[(sub["WindCondition"] == "lowest") & (sub["is_wave"] == False)]

    ax.scatter(sub_fw["dt"], sub_fw["mean_12400/250"], s=42,
               color=WIND_COLOR_MAP.get("full", "#C84F4F"),
               edgecolor="none", alpha=0.85,
               label=f"full-wind nowave (n={len(sub_fw)}, self-anchored — flag)")
    if len(sub_lo):
        ax.scatter(sub_lo["dt"], sub_lo["mean_12400/250"], s=42,
                   color=WIND_COLOR_MAP.get("lowest", "#3B9C57"),
                   edgecolor="none", alpha=0.85,
                   label=f"low-wind nowave (n={len(sub_lo)})")
    ax.scatter(sub_nw["dt"], sub_nw["mean_12400/250"], s=70,
               color=WIND_COLOR_MAP.get("no", "#1F4E8C"),
               edgecolor="black", linewidth=0.6, marker="s", zorder=5,
               label=f"nowind-nowave (n={len(sub_nw)})")
    # Connect bracketing nowind-nowave with a line so the day's drift is visible.
    if len(sub_nw) >= 2:
        ax.plot(sub_nw["dt"], sub_nw["mean_12400/250"],
                color=WIND_COLOR_MAP.get("no", "#1F4E8C"),
                lw=0.8, ls="--", alpha=0.7, zorder=4)

    ax.axhline(0, color="#444", lw=0.6, alpha=0.5)
    # Reference line at the day's first nowind-nowave value to show drift
    # relative to that anchor.
    if len(sub_nw):
        first_nw = sub_nw.iloc[0]["mean_12400/250"]
        if abs(first_nw) > 1e-6:
            ax.axhline(first_nw, color=WIND_COLOR_MAP.get("no", "#1F4E8C"),
                       lw=0.5, ls=":", alpha=0.5)
    ax.set_ylabel("OUT 12400/250\nfirst-3 s mean η [mm]")
    ax.set_title(f"{date}  —  per-run baseline at OUT, chronological")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=8)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("recording time")

fig.suptitle("Bracketing nowind-nowave baseline vs fullwind in-run baseline at OUT",
             fontsize=11)
fig.savefig(OUT, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n   PNG → {OUT.relative_to(BASE)}")
print("\nDone.")
