"""
Highway-effect tests 1 & 2 (2026-05-05)
=======================================

The "highway effect" hypothesis (in-house name): wind-wave activity at
the paddle / source region acts as a moving carrier that the paddle
wave can join immediately, producing a uniform earlier arrival at every
probe. Two falsifiable predictions:

  Test 1 — busier highway = bigger boost.
      |Δt| should correlate with pre-paddle wind-wave RMS at the source
      region (8804/250). Within-cell and across-cell.

  Test 2 — bigger paddle outpaces the highway.
      |Δt| should shrink as paddle amplitude grows. The paddle envelope
      reaches the upcrossing detector faster, so any wind-wave head-start
      matters less proportionally. Functional form: |Δt| ∝ 1/A or
      Δt ∝ σ_pre / A (collapse plot).

Per-run Δt definition:
    Δt_run = snap_fw_run − mean(snap_nw_in_same_cell)
where snap is from `Probe 9373/170 hg_snap_shift` (sample units, /FS to s).

Filters: canon (March-2026 lowrange, 2 folders), full panel, ok quality,
1.3-1.6 Hz, 0.1-0.3 V, both wind conditions.

Outputs:
    analysis_scratch/wind_highway_test1_dt_vs_sigma.png   Test 1 scatter
    analysis_scratch/wind_highway_test2_dt_vs_amp.png     Test 2 scatter
    analysis_scratch/wind_highway_collapse.png            σ_pre/A combined
    analysis_scratch/wind_highway_per_run.csv             one row per fw run
    analysis_scratch/wind_highway_findings.md
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

CANON = [
    BASE / "waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange",
    BASE / "waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange",
]

FREQS = [1.3, 1.4, 1.5, 1.6]
AMPS  = [0.1, 0.2, 0.3]
PROBE_DT     = "9373/170"   # IN: where Δt is measured (any probe works,
                            # they all see the same uniform shift)
PROBE_SOURCE = "8804/250"   # source-region probe for σ_pre
PRE_WINDOW   = (3.0, 8.0)   # seconds — pre-paddle window for σ measurement


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_canon_meta() -> pd.DataFrame:
    rows = []
    for d in CANON:
        for r in json.load(open(d / "meta.json")):
            r["_folder"] = d.name
            rows.append(r)
    df = pd.DataFrame(rows)
    df = df[df["WaveFrequencyInput [Hz]"].notna()]
    df = df[df["PanelCondition"] == "full"]
    df = df[df["quality_flag"] == "ok"]
    df = df[df["WaveFrequencyInput [Hz]"].isin(FREQS)]
    df = df[df["WaveAmplitudeInput [Volt]"].isin(AMPS)]
    df = df[df["WindCondition"].isin(["no", "full"])]
    return df.copy()


def load_processed():
    print("Loading processed_dfs from canon (~45s)...")
    dfs = [pd.read_parquet(d / "processed_dfs.parquet") for d in CANON]
    big = pd.concat(dfs, ignore_index=True)
    print(f"  {len(big):,} rows across {big['_path'].nunique()} runs")
    return big


# ---------------------------------------------------------------------------
# Per-run table assembly
# ---------------------------------------------------------------------------
def build_per_run_table(meta: pd.DataFrame, big: pd.DataFrame) -> pd.DataFrame:
    """For each fullwind run, compute:
      - Δt = snap_fw - mean(snap_nw in same cell)   [in seconds]
      - σ_pre at source probe (8804) over PRE_WINDOW
      - A_paddle  = cell's mean nowind IN Amplitude (FFT)      [mm]
    Drop period-aliased runs (|Δt| > 0.5 T).
    """
    snap_col = f"Probe {PROBE_DT} hg_snap_shift"
    amp_col  = "IN Amplitude (FFT)"
    eta_col  = f"eta_{PROBE_SOURCE}"

    # Build fast lookup of nowind baseline per cell
    nw_baseline = (
        meta[meta["WindCondition"] == "no"]
        .groupby(["WaveFrequencyInput [Hz]",
                  "WaveAmplitudeInput [Volt]"])
        .agg(snap_nw_mean=(snap_col, "mean"),
             A_nw_mean=(amp_col, "mean"),
             n_nw=(snap_col, "count"))
        .reset_index()
    )

    # σ_pre per fullwind run from processed_dfs
    fw_paths = meta[meta["WindCondition"] == "full"]["path"].tolist()
    print(f"  computing σ_pre for {len(fw_paths)} fullwind runs at "
          f"{PROBE_SOURCE} (window t ∈ {PRE_WINDOW} s)...")
    i0 = int(PRE_WINDOW[0] * FS)
    i1 = int(PRE_WINDOW[1] * FS)
    sigma_per_path = {}
    for p in fw_paths:
        eta = big[big["_path"] == p][eta_col].values
        if len(eta) >= i1:
            seg = eta[i0:i1]
            sigma_per_path[p] = float(np.nanstd(seg - np.nanmean(seg)))
        else:
            sigma_per_path[p] = np.nan
    # Also for nowind reference (comparison)
    nw_paths = meta[meta["WindCondition"] == "no"]["path"].tolist()
    sigma_pre_nw_perpath = {}
    for p in nw_paths:
        eta = big[big["_path"] == p][eta_col].values
        if len(eta) >= i1:
            seg = eta[i0:i1]
            sigma_pre_nw_perpath[p] = float(np.nanstd(seg - np.nanmean(seg)))

    print(f"  σ_pre nowind: median = "
          f"{np.nanmedian(list(sigma_pre_nw_perpath.values())):.2f} mm")
    print(f"  σ_pre fullwind: median = "
          f"{np.nanmedian(list(sigma_per_path.values())):.2f} mm")

    # Build long table per fw run
    fw_meta = meta[meta["WindCondition"] == "full"].merge(
        nw_baseline, on=["WaveFrequencyInput [Hz]",
                         "WaveAmplitudeInput [Volt]"], how="left"
    )
    fw_meta["snap_fw_s"] = fw_meta[snap_col] / FS
    fw_meta["snap_nw_s"] = fw_meta["snap_nw_mean"] / FS
    fw_meta["dt_s"] = fw_meta["snap_fw_s"] - fw_meta["snap_nw_s"]
    fw_meta["dt_T"] = fw_meta["dt_s"] * fw_meta["WaveFrequencyInput [Hz]"]
    fw_meta["sigma_pre_mm"] = fw_meta["path"].map(sigma_per_path)
    fw_meta["A_paddle_mm"] = fw_meta["A_nw_mean"]

    keep_cols = [
        "path", "_folder",
        "WaveFrequencyInput [Hz]", "WaveAmplitudeInput [Volt]", "Mooring",
        "snap_fw_s", "snap_nw_s", "n_nw",
        "dt_s", "dt_T",
        "sigma_pre_mm", "A_paddle_mm",
    ]
    out = fw_meta[keep_cols].rename(columns={
        "WaveFrequencyInput [Hz]": "f_hz",
        "WaveAmplitudeInput [Volt]": "amp_V",
    })

    # Drop period-aliased (|Δt_T| > 0.5)
    pre_n = len(out)
    out = out[out["dt_T"].abs() < 0.5]
    print(f"  kept {len(out)}/{pre_n} fw runs after period-alias filter")
    return out


# ---------------------------------------------------------------------------
# Test 1 — Δt vs σ_pre
# ---------------------------------------------------------------------------
def plot_test1(per_run: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))

    cells = sorted(per_run.groupby(["f_hz", "amp_V"]).groups.keys())
    cmap = plt.get_cmap("viridis")
    n = len(cells)

    overall_x, overall_y = [], []
    for i, (f, a) in enumerate(cells):
        sub = per_run[(per_run["f_hz"] == f) & (per_run["amp_V"] == a)]
        if sub.empty:
            continue
        color = cmap(i / max(n - 1, 1))
        ax.scatter(sub["sigma_pre_mm"], sub["dt_s"] * 1000,
                   color=color, s=50, edgecolor="k", lw=0.5,
                   label=f"{f} Hz, {a} V (n={len(sub)})")
        overall_x.extend(sub["sigma_pre_mm"].dropna().tolist())
        overall_y.extend((sub["dt_s"] * 1000).dropna().tolist())
        # Within-cell regression if >= 3 points and σ varies
        if len(sub) >= 3 and sub["sigma_pre_mm"].std() > 0.1:
            m, b = np.polyfit(sub["sigma_pre_mm"], sub["dt_s"] * 1000, 1)
            xx = np.linspace(sub["sigma_pre_mm"].min(),
                              sub["sigma_pre_mm"].max(), 20)
            ax.plot(xx, m * xx + b, color=color, lw=0.8, alpha=0.6)

    # Overall trend line
    if len(overall_x) > 5:
        m, b = np.polyfit(overall_x, overall_y, 1)
        xx = np.linspace(min(overall_x), max(overall_x), 50)
        ax.plot(xx, m * xx + b, "k--", lw=1.5,
                label=f"all-runs fit: {m:.1f} ms/mm")

    ax.axhline(0, color="k", lw=0.3)
    ax.set_xlabel(f"σ_pre at {PROBE_SOURCE} (t ∈ {PRE_WINDOW} s) [mm]")
    ax.set_ylabel("Δt = snap_fw − mean(snap_nw) [ms]")
    ax.set_title("Test 1 — wind-wave RMS at source vs paddle-wave Δt\n"
                 "Highway prediction: more wind activity → larger |Δt| (negative shift)",
                 fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Test 2 — Δt vs paddle amplitude
# ---------------------------------------------------------------------------
def plot_test2(per_run: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    freq_color = {1.3: "#1b9e77", 1.4: "#d95f02",
                  1.5: "#7570b3", 1.6: "#e7298a"}
    amp_marker = {0.1: "o", 0.2: "s", 0.3: "^"}

    # Left: Δt vs A
    ax = axes[0]
    for f in FREQS:
        for a in AMPS:
            sub = per_run[(per_run["f_hz"] == f) & (per_run["amp_V"] == a)]
            if sub.empty:
                continue
            ax.scatter(sub["A_paddle_mm"], sub["dt_s"] * 1000,
                       color=freq_color[f], marker=amp_marker[a], s=70,
                       edgecolor="k", lw=0.5,
                       label=f"{f} Hz {a}V" if a == 0.2 else None)
    # Per-frequency 1/A fit
    for f in FREQS:
        sub = per_run[per_run["f_hz"] == f].dropna(
            subset=["A_paddle_mm", "dt_s"])
        if len(sub) >= 3 and sub["A_paddle_mm"].std() > 0.1:
            x = sub["A_paddle_mm"].values
            y = sub["dt_s"].values * 1000
            # Fit y = c / A  →  c = mean(y * A)
            c = np.mean(y * x)
            xx = np.linspace(x.min(), x.max(), 50)
            ax.plot(xx, c / xx, color=freq_color[f], lw=1, alpha=0.6,
                    ls="--", label=f"{f} Hz: c/A fit  c={c:.0f}")
    ax.axhline(0, color="k", lw=0.3)
    ax.set_xlabel("paddle amplitude A_paddle (cell-mean nw IN A_FFT) [mm]")
    ax.set_ylabel("Δt [ms]")
    ax.set_title("Test 2 — paddle amplitude vs Δt\n"
                 "Highway prediction: |Δt| shrinks as A grows (1/A)",
                 fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    # Right: |Δt| vs A on log-log to see the 1/A trend as slope -1
    ax = axes[1]
    for f in FREQS:
        sub = per_run[per_run["f_hz"] == f].dropna(
            subset=["A_paddle_mm", "dt_s"])
        if sub.empty:
            continue
        ax.loglog(sub["A_paddle_mm"], sub["dt_s"].abs() * 1000, "o",
                  color=freq_color[f], ms=8, mec="k", mew=0.5,
                  label=f"{f} Hz")
    # Reference 1/A line
    if len(per_run):
        x_ref = np.array([per_run["A_paddle_mm"].min(),
                          per_run["A_paddle_mm"].max()])
        y_ref = 100 * x_ref.max() / x_ref  # scaled so it lands in plot
        ax.plot(x_ref, y_ref, "k--", lw=1, label="slope = -1 (1/A)")
    ax.set_xlabel("A_paddle [mm] (log)")
    ax.set_ylabel("|Δt| [ms] (log)")
    ax.set_title("Test 2 (log-log) — slope of −1 confirms 1/A scaling",
                 fontsize=11)
    ax.grid(which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Combined collapse plot — Δt vs σ_pre/A
# ---------------------------------------------------------------------------
def plot_collapse(per_run: pd.DataFrame, out_path: Path) -> None:
    sub = per_run.dropna(subset=["sigma_pre_mm", "A_paddle_mm", "dt_s"]).copy()
    sub["x_collapse"] = sub["sigma_pre_mm"] / sub["A_paddle_mm"]

    fig, ax = plt.subplots(figsize=(10, 7))
    freq_color = {1.3: "#1b9e77", 1.4: "#d95f02",
                  1.5: "#7570b3", 1.6: "#e7298a"}
    amp_marker = {0.1: "o", 0.2: "s", 0.3: "^"}
    for (f, a), grp in sub.groupby(["f_hz", "amp_V"]):
        ax.scatter(grp["x_collapse"], grp["dt_s"] * 1000,
                   color=freq_color[f], marker=amp_marker[a], s=70,
                   edgecolor="k", lw=0.5,
                   label=f"{f} Hz, {a} V")
    # Linear fit (highway prediction: Δt ≈ -k · σ_pre / A)
    if len(sub) > 5:
        m, b = np.polyfit(sub["x_collapse"], sub["dt_s"] * 1000, 1)
        xx = np.linspace(sub["x_collapse"].min(),
                          sub["x_collapse"].max(), 50)
        ax.plot(xx, m * xx + b, "k--", lw=1.5,
                label=f"linear fit  slope = {m:.0f} ms / (mm/mm)")

    ax.axhline(0, color="k", lw=0.3)
    ax.set_xlabel("σ_pre / A_paddle  (dimensionless)")
    ax.set_ylabel("Δt [ms]")
    ax.set_title("Highway-collapse plot — Δt vs σ_pre/A\n"
                 "If model right, all cells lie on one line",
                 fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> None:
    print("Loading canon meta...")
    meta = load_canon_meta()
    print(f"  {len(meta)} ok runs")

    big = load_processed()

    print("\nBuilding per-run table...")
    per_run = build_per_run_table(meta, big)
    csv = OUT_DIR / "wind_highway_per_run.csv"
    per_run.to_csv(csv, index=False)
    print(f"  → {csv.relative_to(BASE)} ({len(per_run)} fw runs)")

    print("\n=== Per-cell summary ===")
    summary = per_run.groupby(["f_hz", "amp_V"]).agg(
        n=("path", "count"),
        dt_mean_ms=("dt_s", lambda s: s.mean() * 1000),
        dt_std_ms=("dt_s",  lambda s: s.std() * 1000),
        sigma_pre_mean=("sigma_pre_mm", "mean"),
        sigma_pre_std=("sigma_pre_mm", "std"),
        A_paddle_mean=("A_paddle_mm", "mean"),
    ).round(2)
    print(summary.to_string())

    print("\nPlotting Test 1 (Δt vs σ_pre)...")
    plot_test1(per_run, OUT_DIR / "wind_highway_test1_dt_vs_sigma.png")
    print("  → analysis_scratch/wind_highway_test1_dt_vs_sigma.png")

    print("\nPlotting Test 2 (Δt vs A_paddle)...")
    plot_test2(per_run, OUT_DIR / "wind_highway_test2_dt_vs_amp.png")
    print("  → analysis_scratch/wind_highway_test2_dt_vs_amp.png")

    print("\nPlotting collapse (Δt vs σ_pre/A)...")
    plot_collapse(per_run, OUT_DIR / "wind_highway_collapse.png")
    print("  → analysis_scratch/wind_highway_collapse.png")

    md = OUT_DIR / "wind_highway_findings.md"
    with open(md, "w") as f:
        f.write(_findings(per_run, summary))
    print(f"\nFindings → {md.relative_to(BASE)}")


def _findings(per_run, summary) -> str:
    n = len(per_run)
    lines = [
        "# Highway-effect tests 1 & 2 (2026-05-05)",
        "",
        f"Per-run analysis on canon, full panel, ok quality, "
        f"non-period-aliased: n={n} fullwind runs.",
        "",
        "## Test 1 — Δt vs pre-paddle σ at source (8804/250)",
        "",
        "Within-cell variation in σ_pre under fullwind shows whether the",
        "highway effect tracks the *amount* of wind-wave activity. If the",
        "wind background is ~steady within a cell, this test reduces to an",
        "across-cell correlation, which is harder to interpret because",
        "frequency and amplitude vary too.",
        "",
        "## Test 2 — Δt vs paddle amplitude A_paddle",
        "",
        "Highway prediction: |Δt| ∝ 1/A. Bigger paddle envelope outpaces",
        "the wind-wave head-start.",
        "",
        "## Per-cell summary",
        "",
        summary.to_string(),
        "",
        "## Files",
        "- wind_highway_test1_dt_vs_sigma.png — Test 1 scatter + fits",
        "- wind_highway_test2_dt_vs_amp.png — Test 2 scatter + 1/A fits",
        "- wind_highway_collapse.png — Δt vs σ_pre/A combined",
        "- wind_highway_per_run.csv — per-fw-run table",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
