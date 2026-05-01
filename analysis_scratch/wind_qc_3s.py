"""
Quality-control + variability map of the wind forcing.

For every run in the two canon -lowrange datasets (quality_flag == "ok"),
take the first 3 s of recording (pre-paddle for wave runs; opening 3 s for
nowave runs — both are pre-disturbance because √(gh) = 2.39 m/s and the
closest probe is 8804 mm away → safe up to 3.68 s) and compute per probe:

  η̄, σ_η, Hs = 4 σ_η.

Two products:

(a) Control chart: σ_η per probe vs chronological run index, coloured by
    WindCondition, with a horizontal reference band from the 5 long
    fullwind+nowave runs.

(b) Boxplots grouped by (WindCondition × date) per probe, showing within-
    group scatter and between-day drift.

Loads the two canon folders only.

Outputs:
    analysis_scratch/wind_qc_3s_per_run.csv
    analysis_scratch/wind_qc_3s_control_chart.png
    analysis_scratch/wind_qc_3s_boxplot.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wavescripts.improved_data_loader import load_analysis_data, load_processed_dfs

FS = 250.0
BASE = Path("/Users/ole/Kodevik/wave_project")

TARGET_DIRS = [
    BASE / "waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange",
    BASE / "waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange",
]

# 5 long fullwind+nowave reference runs — used to build the control-chart
# reference band (long-run σ_η range across them, per probe).
LONG_RUN_CSVS = [
    str(BASE / "wavedata/20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange/fullpanel-fullwind-nowave-depth580-mstop30-run1.csv"),
    str(BASE / "wavedata/20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange/fullpanel-fullwind-nowave-depth580-mstop30-run2.csv"),
    str(BASE / "wavedata/20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange/fullpanel-fullwind-nowave-depth580-mstop330-run1.csv"),
    str(BASE / "wavedata/20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange/fullpanel-fullwind-nowave-depth580-mstop30-run2.csv"),
    str(BASE / "wavedata/20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange/fullpanel-fullwind-nowave-depth580-mstop30-run4.csv"),
]

PROBES = ["9373/170", "12400/250"]  # IN-wall + OUT — the two panels
LABELS = {
    "9373/170":  "9373/170 (IN, wall)",
    "12400/250": "12400/250 (OUT)",
}
WIND_COLOR = {"full": "#FEA11B", "no": "#1E9C68", "lowest": "#888888"}
WIND_MARKER = {"full": "o", "no": "s", "lowest": "^"}

SNIPPET_S = 3.0
SNIPPET_N = int(SNIPPET_S * FS)

# ── Load: TWO canon folders only ─────────────────────────────────────────
print(f"Loading {len(TARGET_DIRS)} canon datasets …")
meta, _, _, _ = load_analysis_data(*[str(d) for d in TARGET_DIRS], load_processed=False)
proc = {}
for d in TARGET_DIRS:
    proc.update(load_processed_dfs(str(d)))
print(f"  → {len(meta)} runs, {len(proc)} time-series")

# ── Helpers ──────────────────────────────────────────────────────────────
def _eta(df, probe):
    col = f"eta_{probe}_interp" if f"eta_{probe}_interp" in df.columns else f"eta_{probe}"
    if col not in df.columns:
        return None
    return df[col].to_numpy(dtype=float)


def _stats(arr):
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return np.nan, np.nan, np.nan
    return float(np.mean(arr)), float(np.std(arr, ddof=1)), 4.0 * float(np.std(arr, ddof=1))


# ── Per-run snippet stats (every quality=ok run) ─────────────────────────
sel = (meta["quality_flag"] == "ok") & (meta["WindCondition"].isin(["full", "no", "lowest"]))
work = meta.loc[sel].copy().reset_index(drop=True)
work = work.sort_values("run_mtime").reset_index(drop=True)
print(f"  {len(work)} ok runs with WindCondition ∈ {{full,no,lowest}}")

rows = []
for idx, row in work.iterrows():
    path = row["path"]
    if path not in proc:
        continue
    df = proc[path]
    if len(df) < SNIPPET_N:
        continue
    rec = {
        "run_idx":       idx,
        "file_date":     str(row.get("file_date", ""))[:10],
        "WindCondition": row["WindCondition"],
        "is_wave":       bool(pd.notna(row.get("WaveFrequencyInput [Hz]"))),
        "WaveFreq_Hz":   row.get("WaveFrequencyInput [Hz]"),
        "WaveAmp_V":     row.get("WaveAmplitudeInput [Volt]"),
        "run_mtime":     row.get("run_mtime"),
        "path":          path,
    }
    for probe in PROBES:
        eta = _eta(df, probe)
        if eta is None:
            continue
        seg = eta[:SNIPPET_N]
        if not np.all(np.isfinite(seg)):
            continue
        m, s, h = _stats(seg)
        rec[f"mean_{probe}"]  = m
        rec[f"sigma_{probe}"] = s
        rec[f"Hs_{probe}"]    = h
    rows.append(rec)

per_run = pd.DataFrame(rows)
print(f"  per-run snippet stats: {len(per_run)} rows")

csv_out = Path(__file__).parent / "wind_qc_3s_per_run.csv"
per_run.to_csv(csv_out, index=False)
print(f"  → {csv_out.relative_to(BASE)}")

# ── Long-run reference σ_η range (per probe) ─────────────────────────────
long_present = [p for p in LONG_RUN_CSVS if p in proc]
long_ref = {}   # probe → (sigma_min, sigma_mean, sigma_max)
for probe in PROBES:
    sigmas = []
    for lp in long_present:
        eta = _eta(proc[lp], probe)
        if eta is None:
            continue
        _, s, _ = _stats(eta)
        if np.isfinite(s):
            sigmas.append(s)
    sigmas = np.array(sigmas) if sigmas else np.array([np.nan])
    long_ref[probe] = (np.nanmin(sigmas), np.nanmean(sigmas), np.nanmax(sigmas))
    print(f"  long-run σ_η at {probe}: "
          f"min={long_ref[probe][0]:.3f}  mean={long_ref[probe][1]:.3f}  max={long_ref[probe][2]:.3f}  mm  "
          f"(n={len(sigmas)})")

# ── (a) Control chart — σ_η vs chronological run index ───────────────────
fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True)
for ax, probe in zip(axes, PROBES):
    sub = per_run.dropna(subset=[f"sigma_{probe}"]).copy().reset_index(drop=True)

    # long-run reference band
    smin, smean, smax = long_ref[probe]
    ax.axhspan(smin, smax, color="#444", alpha=0.10, label=f"long-run σ range  [{smin:.2f}, {smax:.2f}] mm")
    ax.axhline(smean, color="#444", lw=0.8, ls="--",
                label=f"long-run σ mean = {smean:.2f} mm")

    # snippets coloured by WindCondition; marker shape carries wave-vs-nowave
    for wc in ("full", "lowest", "no"):
        for is_wave in (True, False):
            mask = (sub["WindCondition"] == wc) & (sub["is_wave"] == is_wave)
            if not mask.any():
                continue
            x = np.arange(len(sub))[mask]
            y = sub.loc[mask, f"sigma_{probe}"].to_numpy(dtype=float)
            label = f"{wc} · {'wave' if is_wave else 'nowave'}  (n={mask.sum()})"
            ax.scatter(x, y,
                       color=WIND_COLOR[wc],
                       marker="o" if is_wave else "x",
                       s=28 if is_wave else 60,
                       linewidths=1.4,
                       alpha=0.85, label=label)

    # date dividers (vertical lines between dates)
    if len(sub):
        date_changes = sub["file_date"].ne(sub["file_date"].shift()).cumsum()
        for d in sub["file_date"].unique():
            first = (sub["file_date"] == d).idxmax()
            ax.axvline(first - 0.5, color="#bbb", lw=0.6, alpha=0.7)
            ax.text(first, ax.get_ylim()[1] * 0.97, f" {d}", fontsize=8,
                    va="top", ha="left", color="#666",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white",
                              ec="none", alpha=0.85))

    ax.set_ylabel(r"$\sigma_\eta$  [mm]")
    ax.set_title(f"{LABELS[probe]} — first {SNIPPET_S:g} s of every ok run, sorted by mtime",
                 fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.92)

axes[1].set_xlabel("Run index (chronological)")
fig.suptitle(
    "Wind QC — control chart of pre-disturbance σ_η across the canon campaign",
    fontsize=12,
)
fig.tight_layout()

cc_out = Path(__file__).parent / "wind_qc_3s_control_chart.png"
fig.savefig(cc_out, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"  → {cc_out.relative_to(BASE)}")

# ── (b) Boxplots: WindCondition × date ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=False)
for ax, probe in zip(axes, PROBES):
    sub = per_run.dropna(subset=[f"sigma_{probe}"])
    # Make group label "wind/date"
    sub_grp = sub.assign(_grp=sub["WindCondition"] + " · " + sub["file_date"]).copy()
    groups = sorted(sub_grp["_grp"].unique())
    data = [sub_grp.loc[sub_grp["_grp"] == g, f"sigma_{probe}"].to_numpy(dtype=float)
            for g in groups]
    counts = [len(d) for d in data]

    bp = ax.boxplot(data, labels=[f"{g}\n(n={n})" for g, n in zip(groups, counts)],
                    patch_artist=True, showfliers=True, whis=(5, 95))
    for patch, g in zip(bp["boxes"], groups):
        wc = g.split(" · ")[0]
        patch.set_facecolor(WIND_COLOR.get(wc, "#999"))
        patch.set_alpha(0.45)
    for med in bp["medians"]:
        med.set_color("black"); med.set_linewidth(1.6)

    smin, smean, smax = long_ref[probe]
    ax.axhspan(smin, smax, color="#444", alpha=0.10)
    ax.axhline(smean, color="#444", lw=0.8, ls="--",
                label=f"long-run σ mean")

    ax.set_ylabel(r"$\sigma_\eta$  [mm]")
    ax.set_title(f"{LABELS[probe]}", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=20)
    ax.legend(loc="upper right", fontsize=8)

fig.suptitle(
    f"Wind QC — distribution of pre-disturbance σ_η by (WindCondition × date)\n"
    f"first {SNIPPET_S:g} s, canon datasets (20260326+27 -lowrange), quality_flag=ok",
    fontsize=11,
)
fig.tight_layout()

bp_out = Path(__file__).parent / "wind_qc_3s_boxplot.png"
fig.savefig(bp_out, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"  → {bp_out.relative_to(BASE)}")

# ── Group summary ────────────────────────────────────────────────────────
print("\n=== Group medians (σ_η, mm) ===")
for probe in PROBES:
    print(f"\n  {LABELS[probe]}")
    sub = per_run.dropna(subset=[f"sigma_{probe}"]).copy()
    sub["grp"] = sub["WindCondition"] + " · " + sub["file_date"]
    grp = (sub.groupby("grp")[f"sigma_{probe}"]
              .agg(["count", "median", "mean", "std", "min", "max"])
              .round(3))
    print(grp.to_string())

print("\nDone.")
