"""
Highway-effect Δt vs A_in-enhancement cross-correlation (2026-05-05)
====================================================================

Two known wind effects on the paddle wave:
  - **Δt**: snap shift fw vs nw, uniform across all probes (~100–250 ms
    earlier under fullwind), origin still under investigation.
  - **A_in enhancement**: ~10–17% increase in IN Amplitude (FFT) under
    fullwind at 1.5–1.6 Hz, documented in
    `memory/methodology_wind_enhances_A_in.md`.

Question: do these two effects correlate cell-by-cell across (freq, A)?
  - If YES → likely share one mechanism (wind interacts with the wave
    at the source/IN region in a way that shifts both phase and amplitude).
  - If NO → independent phenomena, need separate stories.

Method
------
For each (freq, amp) cell on canon, full panel, ok quality:
  Δt_cell      = mean(snap_fw) − mean(snap_nw)                    [ms]
  A_enh_cell   = (mean(A_in_fw) − mean(A_in_nw)) / mean(A_in_nw)  [%]

Plots:
  - Heatmap of |Δt| over (freq, A)
  - Heatmap of A_enh over (freq, A)
  - Scatter of A_enh vs |Δt|, one point per cell, with correlation

Outputs:
  analysis_scratch/wind_highway_heatmap.png
  analysis_scratch/wind_highway_heatmap.csv
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
from scipy.stats import pearsonr, spearmanr

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
SNAP_COL = "Probe 9373/170 hg_snap_shift"
AMP_COL  = "IN Amplitude (FFT)"


def load_canon_meta() -> pd.DataFrame:
    rows = []
    for d in CANON:
        for r in json.load(open(d / "meta.json")):
            rows.append(r)
    df = pd.DataFrame(rows)
    df = df[df["WaveFrequencyInput [Hz]"].notna()]
    df = df[df["PanelCondition"] == "full"]
    df = df[df["quality_flag"] == "ok"]
    df = df[df["WaveFrequencyInput [Hz]"].isin(FREQS)]
    df = df[df["WaveAmplitudeInput [Volt]"].isin(AMPS)]
    df = df[df["WindCondition"].isin(["no", "full"])]
    return df.copy()


def cell_aggregates(meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for f in FREQS:
        for a in AMPS:
            cell = meta[(meta["WaveFrequencyInput [Hz]"] == f)
                        & (meta["WaveAmplitudeInput [Volt]"] == a)]
            if cell.empty:
                continue
            nw = cell[cell["WindCondition"] == "no"]
            fw = cell[cell["WindCondition"] == "full"]

            # snap shift: keep only non-period-aliased runs
            T_samp = FS / f
            snap_nw = nw[SNAP_COL].dropna() / FS  # in seconds
            snap_fw = fw[SNAP_COL].dropna() / FS
            # Reject runs where snap landed > 0.5 T from theoretical
            snap_nw = snap_nw[(snap_nw * f).abs() < 0.5]
            snap_fw = snap_fw[(snap_fw * f).abs() < 0.5]

            # Amplitude (FFT) at IN
            A_nw = nw[AMP_COL].dropna()
            A_fw = fw[AMP_COL].dropna()

            row = {
                "f_hz": f, "amp_V": a,
                "n_nw": len(snap_nw), "n_fw": len(snap_fw),
                "snap_nw_s": snap_nw.mean() if len(snap_nw) else np.nan,
                "snap_fw_s": snap_fw.mean() if len(snap_fw) else np.nan,
                "A_nw_mm":   A_nw.mean()   if len(A_nw) else np.nan,
                "A_fw_mm":   A_fw.mean()   if len(A_fw) else np.nan,
                "A_nw_std":  A_nw.std()    if len(A_nw) > 1 else np.nan,
                "A_fw_std":  A_fw.std()    if len(A_fw) > 1 else np.nan,
            }
            row["dt_ms"]    = (row["snap_fw_s"] - row["snap_nw_s"]) * 1000
            row["dt_T"]     = row["dt_ms"] / 1000 * f
            row["A_enh_pct"] = (row["A_fw_mm"] - row["A_nw_mm"]) / row["A_nw_mm"] * 100
            # Cell-mean alias filter: even if individual runs survived the
            # ±0.5 T per-run filter, the fw and nw cell means can land on
            # different upcrossings. Flag |Δt_T| > 0.4 as suspicious.
            row["alias_flag"] = abs(row["dt_T"]) > 0.4
            if row["alias_flag"]:
                row["dt_ms"] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def plot_heatmap(cells: pd.DataFrame, out_path: Path) -> None:
    """Three panels: |Δt| heatmap, A_enh heatmap, scatter with correlation."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6),
                              gridspec_kw={"width_ratios": [1, 1, 1.2]})

    # Heatmap 1: |Δt|
    pivot_dt = cells.pivot(index="amp_V", columns="f_hz", values="dt_ms")
    ax = axes[0]
    im = ax.imshow(pivot_dt.values, aspect="auto", cmap="RdBu_r",
                   vmin=-250, vmax=250,
                   origin="lower",
                   extent=[FREQS[0] - 0.05, FREQS[-1] + 0.05,
                           AMPS[0] - 0.05, AMPS[-1] + 0.05])
    ax.set_xticks(FREQS); ax.set_yticks(AMPS)
    ax.set_xlabel("frequency [Hz]"); ax.set_ylabel("amplitude [V]")
    ax.set_title("Δt = snap_fw − snap_nw [ms]\n(negative = fw earlier)",
                 fontsize=11)
    for i, a in enumerate(pivot_dt.index):
        for j, f in enumerate(pivot_dt.columns):
            v = pivot_dt.iloc[i, j]
            if pd.notna(v):
                ax.text(f, a, f"{v:.0f}", ha="center", va="center",
                        fontsize=11, fontweight="bold",
                        color="white" if abs(v) > 100 else "black")
    plt.colorbar(im, ax=ax, label="ms")

    # Heatmap 2: A_in enhancement %
    pivot_amp = cells.pivot(index="amp_V", columns="f_hz", values="A_enh_pct")
    ax = axes[1]
    im = ax.imshow(pivot_amp.values, aspect="auto", cmap="RdBu_r",
                   vmin=-30, vmax=30, origin="lower",
                   extent=[FREQS[0] - 0.05, FREQS[-1] + 0.05,
                           AMPS[0] - 0.05, AMPS[-1] + 0.05])
    ax.set_xticks(FREQS); ax.set_yticks(AMPS)
    ax.set_xlabel("frequency [Hz]"); ax.set_ylabel("amplitude [V]")
    ax.set_title("A_in enhancement = (A_fw − A_nw) / A_nw [%]\n"
                 "(positive = wind boosts A_in)", fontsize=11)
    for i, a in enumerate(pivot_amp.index):
        for j, f in enumerate(pivot_amp.columns):
            v = pivot_amp.iloc[i, j]
            if pd.notna(v):
                ax.text(f, a, f"{v:+.1f}%", ha="center", va="center",
                        fontsize=11, fontweight="bold",
                        color="white" if abs(v) > 15 else "black")
    plt.colorbar(im, ax=ax, label="%")

    # Scatter: A_enh vs |Δt|
    ax = axes[2]
    valid = cells.dropna(subset=["dt_ms", "A_enh_pct"])
    freq_color = {1.3: "#1b9e77", 1.4: "#d95f02",
                  1.5: "#7570b3", 1.6: "#e7298a"}
    amp_marker = {0.1: "o", 0.2: "s", 0.3: "^"}
    for _, r in valid.iterrows():
        ax.scatter(r["A_enh_pct"], r["dt_ms"],
                   color=freq_color[r["f_hz"]],
                   marker=amp_marker[r["amp_V"]], s=150,
                   edgecolor="k", lw=0.7,
                   label=f'{r["f_hz"]} Hz, {r["amp_V"]} V')
        ax.annotate(f'{r["f_hz"]}, {r["amp_V"]}',
                    (r["A_enh_pct"], r["dt_ms"]),
                    xytext=(5, 5), textcoords="offset points", fontsize=7)

    if len(valid) > 3:
        x = valid["A_enh_pct"].values
        y = valid["dt_ms"].values
        m, b = np.polyfit(x, y, 1)
        xx = np.linspace(x.min(), x.max(), 50)
        ax.plot(xx, m * xx + b, "k--", lw=1.5, alpha=0.6,
                label=f"linear fit  slope = {m:.1f} ms/%")
        r_pearson, p_pearson = pearsonr(x, y)
        r_spearman, p_spearman = spearmanr(x, y)
        ax.text(0.02, 0.02,
                f"Pearson r = {r_pearson:+.3f}  (p = {p_pearson:.3f})\n"
                f"Spearman ρ = {r_spearman:+.3f}  (p = {p_spearman:.3f})\n"
                f"n cells = {len(valid)}",
                transform=ax.transAxes, fontsize=10, va="bottom",
                bbox=dict(boxstyle="round,pad=0.4", fc="white",
                          ec="grey", alpha=0.85))

    ax.axhline(0, color="k", lw=0.3)
    ax.axvline(0, color="k", lw=0.3)
    ax.set_xlabel("A_in enhancement [%]")
    ax.set_ylabel("Δt [ms] (negative = fw earlier)")
    ax.set_title("Cell-by-cell correlation: A enhancement vs Δt\n"
                 "color = freq; marker = amp tier (○ 0.1, □ 0.2, △ 0.3)",
                 fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper right", ncol=2)

    fig.suptitle("Wind effects on the paddle wave: phase-shift Δt vs amplitude enhancement\n"
                 "Canon, full panel, ok quality, period-alias filtered",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    print("Loading canon meta...")
    meta = load_canon_meta()
    print(f"  {len(meta)} ok runs")

    print("\nAggregating per cell...")
    cells = cell_aggregates(meta)
    csv = OUT_DIR / "wind_highway_heatmap.csv"
    cells.to_csv(csv, index=False)
    print(f"  → {csv.relative_to(BASE)}")

    print("\n=== Per-cell table ===")
    cols = ["f_hz", "amp_V", "n_nw", "n_fw",
            "A_nw_mm", "A_fw_mm", "A_enh_pct",
            "dt_ms"]
    print(cells[cols].round(2).to_string(index=False))

    valid = cells.dropna(subset=["dt_ms", "A_enh_pct"])
    if len(valid) > 3:
        r_p, p_p = pearsonr(valid["A_enh_pct"], valid["dt_ms"])
        r_s, p_s = spearmanr(valid["A_enh_pct"], valid["dt_ms"])
        print(f"\nCorrelation across {len(valid)} cells:")
        print(f"  Pearson  r = {r_p:+.3f}, p = {p_p:.4f}")
        print(f"  Spearman ρ = {r_s:+.3f}, p = {p_s:.4f}")

    print("\nPlotting heatmap + scatter...")
    plot_heatmap(cells, OUT_DIR / "wind_highway_heatmap.png")
    print("  → analysis_scratch/wind_highway_heatmap.png")


if __name__ == "__main__":
    main()
