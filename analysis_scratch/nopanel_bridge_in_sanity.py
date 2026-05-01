"""
Sanity check: is the November bare-tank G ≈ 1.5 at 1.3 Hz a real propagating
amplification, or partly an inflated A_IN under wind background?

Method (per user spec, 2026-05-01):

  For each November nopanel run at 1.3 Hz (both winds, all paddle amplitudes):

  - IN probe = 9373/250 (centre, exposed to wind)
  - OUT probe = 12400/170 (wall, ostensibly less exposed)

  At each probe and run:

  1. A_LS  — LS fit fundamental on the 10-period H&G window (matches
             prior pipeline; primary measurement).
  2. A_PSDraw — periodogram on the same window: A² = 2 · P_run[bin_p] · df.
  3. P_pre[f_p] — periodogram on the first 3 s pre-paddle, interpolated
             onto the run's frequency grid at f_p.
  4. A_clean = sqrt(max(0, A_PSDraw² − 2 · P_pre[f_p] · df))
             — what's left of the paddle bin after removing the *uncorrelated*
             wind variance at f_p (Wiener-style coherent-power subtraction).
  5. σ_pre = std of first 3 s.

  Then:
     G_raw   = A_OUT_LS    / A_IN_LS
     G_clean = A_OUT_clean / A_IN_clean

If G_clean still ≈ 1.5 under fullwind, the bare-tank amplification is real.
If G_clean drops back near 1.0, it was an A_IN-inflation artefact.

Output:
    analysis_scratch/nopanel_bridge_in_sanity_per_run.csv
    analysis_scratch/nopanel_bridge_in_sanity_summary.csv
    analysis_scratch/nopanel_bridge_in_sanity.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import os
os.chdir(BASE)

from wavescripts.improved_data_loader import load_analysis_data, load_processed_dfs
from wavescripts.plot_utils import apply_thesis_style, WIND_COLOR_MAP

FS = 250.0
F_P = 1.3                # bridge target
SNIPPET_S = 3.0
SNIPPET_N = int(SNIPPET_S * FS)

DATASET_DIRS = [
    "waveprocessed/PROCESSED-20251110-tett6roof-lowM-ekte580",
    "waveprocessed/PROCESSED-20251112-tett6roof",
    "waveprocessed/PROCESSED-20251113-tett6roof-loosepaneltaped",
]

IN_PROBE  = "9373/250"
OUT_PROBE = "12400/170"
AMP_LABEL = {0.1: "A1", 0.2: "A2", 0.3: "A3"}


# ── Helpers ──────────────────────────────────────────────────────────────
def _eta_col(df: pd.DataFrame, probe: str) -> str | None:
    for c in (f"eta_{probe}_interp", f"eta_{probe}"):
        if c in df.columns:
            return c
    return None


def _hg_window(meta_row, probe: str) -> tuple[int, int] | None:
    s_col = f"Computed Probe {probe} start"
    e_col = f"Computed Probe {probe} end"
    if s_col not in meta_row or e_col not in meta_row:
        return None
    try:
        s = int(meta_row[s_col]); e = int(meta_row[e_col])
    except (TypeError, ValueError):
        return None
    if e <= s:
        return None
    return s, e


def ls_amplitude(eta_seg: np.ndarray, fs: float, f: float) -> float:
    n = eta_seg.size
    if n < 4 or not np.all(np.isfinite(eta_seg)):
        return np.nan
    t = np.arange(n) / fs
    arg = 2.0 * np.pi * f * t
    A_mat = np.column_stack([np.sin(arg), np.cos(arg), np.ones(n)])
    coeffs, *_ = np.linalg.lstsq(A_mat, eta_seg, rcond=None)
    a, b, _c = coeffs
    return float(np.sqrt(a * a + b * b))


def periodogram_density(eta: np.ndarray, fs: float):
    f, P = signal.periodogram(eta, fs=fs, scaling="density",
                              window="boxcar", detrend="constant")
    return f, P


def amplitude_at_bin(f: np.ndarray, P: np.ndarray, f_target: float) -> float:
    """A = sqrt(2 · P[nearest bin] · df) — matches LS amplitude on integer-cycle windows."""
    if f.size < 2:
        return np.nan
    df = f[1] - f[0]
    k = int(np.argmin(np.abs(f - f_target)))
    return float(np.sqrt(2.0 * max(0.0, P[k]) * df))


def wind_amplitude_at_bin(eta: np.ndarray, fs: float, f_target: float,
                          n_pre: int, df_run: float) -> float:
    """Estimate the uncorrelated wind contribution to A at f_target,
    using the first n_pre samples (pre-paddle) and matching the run's df."""
    if eta.size < n_pre + 4:
        return np.nan
    f_pre, P_pre = periodogram_density(eta[:n_pre], fs)
    if f_pre.size < 2:
        return np.nan
    P_at_fp = float(np.interp(f_target, f_pre, P_pre,
                              left=P_pre[0], right=P_pre[-1]))
    return float(np.sqrt(2.0 * max(0.0, P_at_fp) * df_run))


# ── Load ──────────────────────────────────────────────────────────────────
print(f"Loading {len(DATASET_DIRS)} November folders …")
meta, _, _, _ = load_analysis_data(*DATASET_DIRS, load_processed=False)
proc = {}
for d in DATASET_DIRS:
    proc.update(load_processed_dfs(d))
print(f"  meta {len(meta)} rows; processed_dfs {len(proc)}")


# ── Filter to November nopanel @ 1.3 Hz ──────────────────────────────────
def _round1(x):
    try:
        return round(float(x), 2)
    except (TypeError, ValueError):
        return np.nan

work = meta[
    (meta["quality_flag"] == "ok") &
    (meta["WindCondition"].isin(["no", "full"])) &
    (meta["PanelCondition"] == "no") &
    (meta["WaveFrequencyInput [Hz]"].apply(_round1) == F_P) &
    (meta["in_position"] == IN_PROBE) &
    (meta["out_position"] == OUT_PROBE)
].copy()
print(f"  → {len(work)} November nopanel runs at f_p={F_P} Hz "
      f"(IN={IN_PROBE}, OUT={OUT_PROBE})")


# ── Per-run measurements ─────────────────────────────────────────────────
rows = []
for _, row in work.iterrows():
    p = row["path"]
    if p not in proc:
        continue
    df = proc[p]
    amp_v = round(float(row["WaveAmplitudeInput [Volt]"]), 2)

    rec = {
        "path":     p,
        "amp_v":    amp_v,
        "amp_tag":  AMP_LABEL.get(amp_v, f"A?({amp_v})"),
        "wind":     row["WindCondition"],
        "file_date": str(row.get("file_date", ""))[:10],
    }

    for tag, probe in [("IN", IN_PROBE), ("OUT", OUT_PROBE)]:
        col = _eta_col(df, probe)
        win = _hg_window(row, probe)
        if col is None or win is None:
            for k in ("A_LS", "A_PSDraw", "A_wind_at_fp", "A_clean", "sigma_pre", "df_run", "N_run"):
                rec[f"{tag}_{k}"] = np.nan
            continue
        eta = df[col].to_numpy(float)
        s, e = win
        seg = eta[s:e]
        N = seg.size
        # 1. LS
        rec[f"{tag}_A_LS"] = ls_amplitude(seg, FS, F_P)
        # 2. PSD raw on the H&G window
        f_run, P_run = periodogram_density(seg, FS)
        df_run = f_run[1] - f_run[0] if f_run.size >= 2 else np.nan
        rec[f"{tag}_A_PSDraw"] = amplitude_at_bin(f_run, P_run, F_P)
        rec[f"{tag}_df_run"] = df_run
        rec[f"{tag}_N_run"]  = N
        # 3+4. Wind at f_p from 3 s pre-paddle, then coherent subtraction
        A_wind_fp = wind_amplitude_at_bin(eta, FS, F_P, SNIPPET_N, df_run)
        rec[f"{tag}_A_wind_at_fp"] = A_wind_fp
        A_raw = rec[f"{tag}_A_PSDraw"]
        if np.isfinite(A_raw) and np.isfinite(A_wind_fp):
            A_clean = float(np.sqrt(max(0.0, A_raw * A_raw - A_wind_fp * A_wind_fp)))
        else:
            A_clean = np.nan
        rec[f"{tag}_A_clean"] = A_clean
        # 5. σ over first 3 s
        seg_pre = eta[:SNIPPET_N]
        rec[f"{tag}_sigma_pre"] = (float(np.std(seg_pre, ddof=1))
                                   if seg_pre.size >= 4 and np.all(np.isfinite(seg_pre))
                                   else np.nan)

    # Per-run gains
    rec["G_raw"]   = rec["OUT_A_LS"]    / rec["IN_A_LS"]    if rec.get("IN_A_LS")    else np.nan
    rec["G_PSDraw"] = rec["OUT_A_PSDraw"]/ rec["IN_A_PSDraw"] if rec.get("IN_A_PSDraw") else np.nan
    rec["G_clean"] = rec["OUT_A_clean"] / rec["IN_A_clean"] if rec.get("IN_A_clean") else np.nan
    rows.append(rec)

per_run = pd.DataFrame(rows)
per_run.to_csv("analysis_scratch/nopanel_bridge_in_sanity_per_run.csv", index=False)
print(f"   CSV → analysis_scratch/nopanel_bridge_in_sanity_per_run.csv ({len(per_run)} rows)")

if per_run.empty:
    print("\nNo runs available — abort.")
    sys.exit(0)


# ── Aggregate per (amp, wind) ────────────────────────────────────────────
def _agg(grp, col):
    arr = grp[col].dropna().to_numpy()
    return (float(arr.mean()) if arr.size else np.nan,
            float(arr.std(ddof=1)) if arr.size > 1 else np.nan,
            int(arr.size))


sum_rows = []
for (amp, wind), g in per_run.groupby(["amp_v", "wind"], sort=True):
    rec = {"amp_v": amp, "amp_tag": AMP_LABEL.get(amp, "A?"), "wind": wind, "n": len(g)}
    for col in ("IN_A_LS", "IN_A_PSDraw", "IN_A_wind_at_fp", "IN_A_clean", "IN_sigma_pre",
                "OUT_A_LS", "OUT_A_PSDraw", "OUT_A_wind_at_fp", "OUT_A_clean", "OUT_sigma_pre",
                "G_raw", "G_PSDraw", "G_clean"):
        m, s, n = _agg(g, col)
        rec[f"{col}_mean"] = m
        rec[f"{col}_std"]  = s
        rec[f"{col}_n"]    = n
    # Wind-contribution fraction at IN
    rec["IN_wind_frac_pct"] = (
        100.0 * rec["IN_A_wind_at_fp_mean"] / rec["IN_A_PSDraw_mean"]
        if rec["IN_A_PSDraw_mean"] and np.isfinite(rec["IN_A_PSDraw_mean"]) else np.nan
    )
    rec["G_clean_minus_G_raw"] = rec["G_clean_mean"] - rec["G_raw_mean"]
    sum_rows.append(rec)

summary = pd.DataFrame(sum_rows).sort_values(["amp_v", "wind"]).reset_index(drop=True)
summary.to_csv("analysis_scratch/nopanel_bridge_in_sanity_summary.csv", index=False)
print(f"   CSV → analysis_scratch/nopanel_bridge_in_sanity_summary.csv ({len(summary)} rows)")


# ── Console table ────────────────────────────────────────────────────────
print(f"\n=== November nopanel @ {F_P} Hz — IN-side sanity check ===\n"
      f"   IN: {IN_PROBE} (centre, exposed)   OUT: {OUT_PROBE} (wall)\n"
      f"   3 s pre-paddle wind PSD subtracted at f_p, coherent-power style.\n")

disp = summary[[
    "amp_tag", "wind", "n",
    "IN_sigma_pre_mean", "IN_A_LS_mean", "IN_A_wind_at_fp_mean", "IN_A_clean_mean",
    "IN_wind_frac_pct",
    "OUT_A_LS_mean", "OUT_sigma_pre_mean",
    "G_raw_mean", "G_clean_mean", "G_clean_minus_G_raw",
]].copy()
disp.columns = [
    "amp", "wind", "n",
    "σ_pre_IN", "A_IN_LS", "A_IN_wind@fp", "A_IN_clean",
    "wind_frac_IN_%",
    "A_OUT_LS", "σ_pre_OUT",
    "G_raw", "G_clean", "ΔG",
]
for c in disp.columns:
    if disp[c].dtype == float:
        disp[c] = disp[c].round(3)
print(disp.to_string(index=False))

# Cross-row delta: how much does the wind correction shift G under fullwind vs nowind?
print("\n=== Verdict per amplitude tier ===")
for amp in sorted(per_run["amp_v"].unique()):
    n  = summary[(summary["amp_v"] == amp) & (summary["wind"] == "no")]
    fw = summary[(summary["amp_v"] == amp) & (summary["wind"] == "full")]
    if n.empty or fw.empty:
        continue
    G_now_raw    = float(n["G_raw_mean"].iloc[0])
    G_now_clean  = float(n["G_clean_mean"].iloc[0])
    G_wind_raw   = float(fw["G_raw_mean"].iloc[0])
    G_wind_clean = float(fw["G_clean_mean"].iloc[0])
    print(f"  {AMP_LABEL[amp]} ({amp:.1f} V):  "
          f"G_now: raw={G_now_raw:.3f} → clean={G_now_clean:.3f}   |   "
          f"G_wind: raw={G_wind_raw:.3f} → clean={G_wind_clean:.3f}   "
          f"(Δ_clean = {G_wind_clean - G_now_clean:+.3f})")


# ── Figure ───────────────────────────────────────────────────────────────
apply_thesis_style(usetex=False)
plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3})

fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))

# Left: G_raw vs G_clean per (amp, wind), bars
amps = sorted(per_run["amp_v"].unique())
x = np.arange(len(amps)) * 2.0
width = 0.40

for j, wind in enumerate(["no", "full"]):
    color = WIND_COLOR_MAP[wind]
    g_raw = []
    g_clean = []
    for a in amps:
        sub = summary[(summary["amp_v"] == a) & (summary["wind"] == wind)]
        g_raw.append(  float(sub["G_raw_mean"].iloc[0])   if not sub.empty else np.nan)
        g_clean.append(float(sub["G_clean_mean"].iloc[0]) if not sub.empty else np.nan)
    offs = -width / 2 if j == 0 else +width / 2
    axes[0].bar(x + offs - width * 0.1, g_raw,  width * 0.45, color=color, alpha=0.55,
                label=f"{wind} · G_raw")
    axes[0].bar(x + offs + width * 0.45, g_clean, width * 0.45, color=color, alpha=0.95,
                edgecolor="black", linewidth=0.5,
                label=f"{wind} · G_clean")

axes[0].axhline(1.0, color="#444", lw=0.7, ls="--")
axes[0].set_xticks(x)
axes[0].set_xticklabels([f"{AMP_LABEL[a]}\n({a:.1f} V)" for a in amps])
axes[0].set_ylabel(r"$G = A_\mathrm{OUT}/A_\mathrm{IN}$")
axes[0].set_title(rf"Bare-tank gain @ $f_p={F_P}$ Hz — raw vs IN-cleaned")
axes[0].legend(fontsize=7, loc="upper left", ncol=2, framealpha=0.92)

# Right: A_IN components per run (so user can see how small the wind-at-fp correction is)
ax = axes[1]
for wind in ("no", "full"):
    sub = per_run[per_run["wind"] == wind]
    if sub.empty:
        continue
    color = WIND_COLOR_MAP[wind]
    ax.scatter(sub["IN_A_LS"], sub["IN_A_wind_at_fp"],
               color=color, s=42, alpha=0.85, edgecolors="white", linewidths=0.5,
               label=f"{wind}  (n={len(sub)})")

ax.set_xlabel(r"$A_\mathrm{IN,LS}$ at $f_p$ [mm]")
ax.set_ylabel(r"$A_\mathrm{IN,wind@f_p}$ from 3 s pre-paddle [mm]")
ax.set_title(r"Wind contribution to IN amplitude at $f_p$")
# y=x reference and 10 % line
xx = np.linspace(0, ax.get_xlim()[1] if ax.get_xlim()[1] > 0 else 30, 50)
ax.plot(xx, xx, color="#888", lw=0.6, ls=":", label="y = x")
ax.plot(xx, 0.10 * xx, color="#aaa", lw=0.6, ls="--", label="y = 0.10 x")
ax.legend(fontsize=7, loc="upper right")

fig.suptitle(rf"November nopanel @ {F_P} Hz — IN-side wind-background sanity check", fontsize=11)
fig.tight_layout()

png = Path("analysis_scratch/nopanel_bridge_in_sanity.png")
fig.savefig(png, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"\n   PNG → {png}")

print("\nDone.")
