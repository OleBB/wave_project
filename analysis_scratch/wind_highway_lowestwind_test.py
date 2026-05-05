"""
Highway-effect wind-strength test using lowestwind (Nov 2025) data
==================================================================

Until now, wind has been binary (no/full) so we couldn't test the
highway-effect prediction "more wind → bigger Δt". Nov 2025 data has a
THIRD wind condition: lowestwind = 3.8 m/s (vs fullwind = 6 m/s),
giving us a 3-point wind-strength axis at 1.3 Hz.

Coverage (Nov 2025, IN = 9373/250 era):
  PROCESSED-20251112-tett6roof + PROCESSED-20251113-tett6roof-loosepaneltaped
  Panel: no (clean baseline) and reverse (parallel test)
  1.3 Hz × {0.1, 0.2, 0.3} V × {no, lowest, full} wind
  → 2 folders × 3 amp × 3 wind = 18 runs per panel condition

Wind speeds:
  no       = 0 m/s
  lowest   = 3.8 m/s
  full     = 6.0 m/s

Test: plot Δt vs wind speed per amplitude. If the highway effect is
monotonic in wind strength, Δt(lowest) should sit between Δt(no) and
Δt(full).

Outputs:
  analysis_scratch/wind_highway_lowestwind_test.png
  analysis_scratch/wind_highway_lowestwind_test.csv
  analysis_scratch/wind_highway_lowestwind_test.md
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

OUT_DIR = BASE / "analysis_scratch"
FS = 250.0

NOV_FOLDERS = [
    BASE / "waveprocessed/PROCESSED-20251112-tett6roof",
    BASE / "waveprocessed/PROCESSED-20251113-tett6roof-loosepaneltaped",
]
WIND_SPEED = {"no": 0.0, "lowest": 3.8, "full": 6.0}
PROBE_IN  = "9373/250"   # Nov 2025 IN reference
PROBE_OUT = "12400/170"  # Nov 2025 OUT reference (for cross-check)
AMPS = [0.1, 0.2, 0.3]
F_HZ = 1.3
PANELS = ["no", "reverse"]   # avoid 'full' panel (very few runs)


def load_nov_meta() -> pd.DataFrame:
    rows = []
    for d in NOV_FOLDERS:
        for r in json.load(open(d / "meta.json")):
            r["_folder"] = d.name
            rows.append(r)
    df = pd.DataFrame(rows)
    df = df[df["WaveFrequencyInput [Hz]"] == F_HZ]
    df = df[df["quality_flag"] == "ok"]
    df = df[df["WaveAmplitudeInput [Volt]"].isin(AMPS)]
    df = df[df["WindCondition"].isin(WIND_SPEED.keys())]
    df = df[df["PanelCondition"].isin(PANELS)]
    return df.copy()


def per_run_table(meta: pd.DataFrame) -> pd.DataFrame:
    """Build per-run table with snap shift in seconds for IN and OUT."""
    out = meta.copy()
    out["snap_in_s"]  = out[f"Probe {PROBE_IN} hg_snap_shift"]  / FS
    out["snap_out_s"] = out[f"Probe {PROBE_OUT} hg_snap_shift"] / FS
    out["wind_ms"]    = out["WindCondition"].map(WIND_SPEED)
    out["snap_in_T"]  = out["snap_in_s"]  * F_HZ
    out["snap_out_T"] = out["snap_out_s"] * F_HZ
    keep = [
        "_folder", "path", "PanelCondition", "WindCondition", "wind_ms",
        "WaveAmplitudeInput [Volt]",
        "snap_in_s", "snap_in_T", "snap_out_s", "snap_out_T",
    ]
    return out[keep].rename(columns={"WaveAmplitudeInput [Volt]": "amp_V"})


def cell_aggregates(per_run: pd.DataFrame) -> pd.DataFrame:
    """For each (panel, amp, wind) cell: mean snap_in, count, std."""
    g = (per_run.groupby(["PanelCondition", "amp_V", "WindCondition", "wind_ms"])
         .agg(snap_in_s_mean=("snap_in_s", "mean"),
              snap_in_s_std= ("snap_in_s", "std"),
              snap_out_s_mean=("snap_out_s", "mean"),
              snap_out_s_std= ("snap_out_s", "std"),
              n=("path", "count"))
         .reset_index())
    # Δt vs nowind baseline per (panel, amp)
    nw = g[g["WindCondition"] == "no"][
        ["PanelCondition", "amp_V", "snap_in_s_mean", "snap_out_s_mean"]
    ].rename(columns={"snap_in_s_mean": "snap_in_nw_s",
                      "snap_out_s_mean": "snap_out_nw_s"})
    g = g.merge(nw, on=["PanelCondition", "amp_V"], how="left")
    g["dt_in_ms"]  = (g["snap_in_s_mean"]  - g["snap_in_nw_s"])  * 1000
    g["dt_out_ms"] = (g["snap_out_s_mean"] - g["snap_out_nw_s"]) * 1000
    return g


def plot_wind_axis(cells: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    amp_color = {0.1: "#1f77b4", 0.2: "#2ca02c", 0.3: "#d62728"}
    panel_marker = {"no": "o", "reverse": "s"}

    for ax, probe_label, dt_col in zip(
        axes, [f"IN ({PROBE_IN})", f"OUT ({PROBE_OUT})"],
        ["dt_in_ms", "dt_out_ms"]
    ):
        for panel in PANELS:
            for amp in AMPS:
                sub = cells[(cells["PanelCondition"] == panel)
                            & (cells["amp_V"] == amp)].sort_values("wind_ms")
                if len(sub) < 2:
                    continue
                ax.plot(sub["wind_ms"], sub[dt_col],
                        marker=panel_marker[panel], color=amp_color[amp],
                        ms=10, lw=1.5, mec="k", mew=0.6,
                        label=f"panel={panel}, A={amp} V")
                # n-annotations
                for _, r in sub.iterrows():
                    ax.annotate(f'n={int(r["n"])}',
                                (r["wind_ms"], r[dt_col]),
                                xytext=(5, 5), textcoords="offset points",
                                fontsize=7, color="grey")
        ax.axhline(0, color="k", lw=0.3)
        ax.set_xlabel("wind speed [m/s]  (no=0, lowest=3.8, full=6.0)")
        ax.set_xticks([0, 3.8, 6.0])
        ax.set_title(f"Δt at {probe_label}", fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Δt = snap − snap_nowind  [ms]")
    fig.suptitle(f"Highway-effect wind-strength test  —  1.3 Hz, Nov 2025 data\n"
                 f"prediction: |Δt| should grow monotonically with wind",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    print("Loading Nov 2025 meta...")
    meta = load_nov_meta()
    print(f"  {len(meta)} runs at 1.3 Hz, panel ∈ {PANELS}, ok quality")

    per_run = per_run_table(meta)
    cells = cell_aggregates(per_run)
    csv = OUT_DIR / "wind_highway_lowestwind_test.csv"
    cells.to_csv(csv, index=False)
    print(f"  → {csv.relative_to(BASE)}")

    print("\n=== Cell aggregates (Δt at IN, ms) ===")
    pivot_in = cells.pivot_table(
        index=["PanelCondition", "amp_V"], columns="WindCondition",
        values="dt_in_ms"
    )[["no", "lowest", "full"]]
    print(pivot_in.round(0).to_string())

    print("\n=== Cell aggregates (Δt at OUT, ms) ===")
    pivot_out = cells.pivot_table(
        index=["PanelCondition", "amp_V"], columns="WindCondition",
        values="dt_out_ms"
    )[["no", "lowest", "full"]]
    print(pivot_out.round(0).to_string())

    print("\n=== Monotonicity check (sign of |Δt(lowest)| vs |Δt(full)|) ===")
    print("  POSITIVE = lowestwind shift is between zero and fullwind shift "
          "(supports highway monotonicity)")
    rows = []
    for panel in PANELS:
        for amp in AMPS:
            sub = cells[(cells["PanelCondition"] == panel)
                        & (cells["amp_V"] == amp)]
            if len(sub) < 3:
                continue
            dt_lo  = float(sub[sub["WindCondition"] == "lowest"]["dt_in_ms"].iloc[0])
            dt_fu  = float(sub[sub["WindCondition"] == "full"]["dt_in_ms"].iloc[0])
            ratio  = dt_lo / dt_fu if dt_fu != 0 else np.nan
            mono   = "monotonic" if (0 < ratio < 1.2) else "NON-monotonic"
            rows.append({"panel": panel, "amp_V": amp,
                         "dt_lo": dt_lo, "dt_fu": dt_fu,
                         "lo/fu_ratio": ratio,
                         "wind_speed_ratio": 3.8 / 6.0,
                         "verdict": mono})
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    print("\nPlotting wind-strength scan...")
    plot_wind_axis(cells, OUT_DIR / "wind_highway_lowestwind_test.png")
    print("  → analysis_scratch/wind_highway_lowestwind_test.png")


if __name__ == "__main__":
    main()
