"""
Wind effect quantification at f_p — IN, OUT, transmission, phase.
=================================================================

For each canonical wave condition (frequency × paddle amplitude) and each
probe (9373/170 IN-wall, 9373/340 IN-far, 12400/250 OUT):

  * Per run, on the 10-period H&G wave window, fit
        y(t) = a sin(2π f_p t) + b cos(2π f_p t) + c
    by least squares ⇒ amplitude A = √(a²+b²), phase φ = atan2(b, a).
  * Per run, compute σ_η on the first 3 s of recording at the same probe
    (pre-paddle wind background).
  * Compute ensemble mean / std per (f, A, probe, wind state).

Quantities reported per (f, A):
    A̅_IN,nowind / A̅_IN,wind    R_IN  = A̅_IN,wind / A̅_IN,nowind
    A̅_OUT,nowind / A̅_OUT,wind   R_OUT = A̅_OUT,wind / A̅_OUT,nowind
    T = A_OUT / A_IN  per IN-probe AND per canonical IN-mean
    R_T = T_wind / T_now
    σ̅_pre_3s at IN-wall + OUT, mean ± std
    Δφ̅ = circular mean(φ_OUT − φ_IN_wall),   ΔΔφ = Δφ̅_wind − Δφ̅_nowind

Outputs:
    analysis_scratch/wind_effect_per_condition_per_run.csv
    analysis_scratch/wind_effect_per_condition_long.csv
    analysis_scratch/wind_effect_per_condition_ratios.csv
    analysis_scratch/wind_effect_ratios_summary.png
    analysis_scratch/wind_effect_scatter_f14_A1.png
    analysis_scratch/wind_effect_scatter_f14_A2.png
    analysis_scratch/wind_effect_scatter_f14_A3.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import os
os.chdir(BASE)

from wavescripts.improved_data_loader import load_analysis_data, load_processed_dfs
from wavescripts.plot_utils import apply_thesis_style, WIND_COLOR_MAP

FS = 250.0
PROBES = {
    "IN_wall": "9373/170",
    "IN_far":  "9373/340",
    "OUT":     "12400/250",
}
SNIPPET_S = 3.0
SNIPPET_N = int(SNIPPET_S * FS)

CANON_FREQS  = [1.3, 1.4, 1.5, 1.6]
CANON_AMPS_V = [0.1, 0.2, 0.3]
# Per memory feedback_freq_amp_limits.md: A_3 (0.3V) at 1.6Hz is dropout-prone.
EXCLUDE_CONDITIONS = {(1.6, 0.3)}

DIRS = [
    "waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange",
    "waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange",
]

# Map paddle voltage → reader-facing amplitude tier label.
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


def ls_fit_at_freq(eta_seg: np.ndarray, fs: float, f: float):
    """Return A, φ from y = a sin(2πf t) + b cos(2πf t) + c.

    A = √(a² + b²),   φ = atan2(b, a)  (phase of the leading sine convention).
    """
    n = eta_seg.size
    if n < 4 or not np.all(np.isfinite(eta_seg)):
        return np.nan, np.nan
    t = np.arange(n) / fs
    arg = 2.0 * np.pi * f * t
    A_mat = np.column_stack([np.sin(arg), np.cos(arg), np.ones(n)])
    coeffs, *_ = np.linalg.lstsq(A_mat, eta_seg, rcond=None)
    a, b, _c = coeffs
    A   = float(np.sqrt(a * a + b * b))
    phi = float(np.arctan2(b, a))
    return A, phi


def sigma_pre(eta: np.ndarray, n_samples: int) -> float:
    seg = eta[:n_samples]
    if seg.size < 4 or not np.all(np.isfinite(seg)):
        return np.nan
    return float(np.std(seg, ddof=1))


def circ_mean(angles: np.ndarray) -> tuple[float, float]:
    """Circular mean and std of angles in radians."""
    angles = np.asarray(angles, dtype=float)
    angles = angles[np.isfinite(angles)]
    if angles.size == 0:
        return np.nan, np.nan
    z = np.exp(1j * angles).mean()
    mean_phi = float(np.angle(z))
    R = abs(z)
    # Circular std (Mardia): √(−2 ln R).
    std_phi = float(np.sqrt(max(0.0, -2.0 * np.log(R)))) if R > 0 else np.nan
    return mean_phi, std_phi


def wrap_pi(x: float) -> float:
    return float((x + np.pi) % (2.0 * np.pi) - np.pi)


# ── Load ──────────────────────────────────────────────────────────────────
print("Loading meta + processed_dfs (canon -lowrange × 2) …")
meta, _, _, _ = load_analysis_data(*DIRS, load_processed=False)
proc = {}
for d in DIRS:
    proc.update(load_processed_dfs(d))
print(f"  meta: {len(meta)} rows; processed_dfs: {len(proc)}")


# ── Filter to canon wave runs ────────────────────────────────────────────
def _round1(x):
    try:
        return round(float(x), 2)
    except (TypeError, ValueError):
        return np.nan

sel = (
    (meta["PanelCondition"] == "full") &
    (meta["quality_flag"]   == "ok") &
    (meta["WindCondition"].isin(["no", "full"])) &
    (meta["WaveFrequencyInput [Hz]"].apply(_round1).isin(CANON_FREQS)) &
    (meta["WaveAmplitudeInput [Volt]"].apply(_round1).isin(CANON_AMPS_V))
)
work = meta.loc[sel].copy()
print(f"  → {len(work)} canon wave runs (full/no, ok)")


# ── Per-run quantities ───────────────────────────────────────────────────
rows = []
for _, row in work.iterrows():
    p = row["path"]
    if p not in proc:
        continue
    df = proc[p]
    f_p = float(row["WaveFrequencyInput [Hz]"])
    amp_v = float(row["WaveAmplitudeInput [Volt]"])
    if (round(f_p, 2), round(amp_v, 2)) in EXCLUDE_CONDITIONS:
        continue

    rec = {
        "path":     p,
        "freq_hz":  round(f_p, 2),
        "amp_v":    round(amp_v, 2),
        "amp_tag":  AMP_LABEL.get(round(amp_v, 2), f"A?({amp_v})"),
        "wind":     row["WindCondition"],
        "run_category": row.get("run_category", ""),
        "file_date":    str(row.get("file_date", ""))[:10],
        "mooring":      row.get("Mooring", ""),
    }

    for tag, probe in PROBES.items():
        col = _eta_col(df, probe)
        if col is None:
            rec[f"A_{tag}"]            = np.nan
            rec[f"phi_{tag}"]          = np.nan
            rec[f"sigma_pre_{tag}"]    = np.nan
            rec[f"A_FFT_meta_{tag}"]   = np.nan
            continue
        eta = df[col].to_numpy(dtype=float)
        win = _hg_window(row, probe)
        if win is None:
            rec[f"A_{tag}"] = rec[f"phi_{tag}"] = np.nan
        else:
            s, e = win
            A, phi = ls_fit_at_freq(eta[s:e], FS, f_p)
            rec[f"A_{tag}"]   = A
            rec[f"phi_{tag}"] = phi

        rec[f"sigma_pre_{tag}"]  = sigma_pre(eta, SNIPPET_N)
        rec[f"A_FFT_meta_{tag}"] = float(row.get(f"Probe {probe} Amplitude (FFT)", np.nan))

    rows.append(rec)

per_run = pd.DataFrame(rows)
print(f"  → {len(per_run)} per-run rows")

# IN-mean (canonical) amplitude per run.
per_run["A_IN_mean"]  = per_run[["A_IN_wall", "A_IN_far"]].mean(axis=1)
per_run["sigma_pre_IN_mean"] = per_run[["sigma_pre_IN_wall", "sigma_pre_IN_far"]].mean(axis=1)

# Per-run phase difference OUT − IN_wall (wrapped to (-π, π]).
def _dphi(row):
    a, b = row["phi_OUT"], row["phi_IN_wall"]
    if not (np.isfinite(a) and np.isfinite(b)):
        return np.nan
    return wrap_pi(a - b)

per_run["dphi_OUT_minus_IN_wall"] = per_run.apply(_dphi, axis=1)

# Per-run transmission columns.
per_run["T_wall"] = per_run["A_OUT"] / per_run["A_IN_wall"]
per_run["T_far"]  = per_run["A_OUT"] / per_run["A_IN_far"]
per_run["T_mean"] = per_run["A_OUT"] / per_run["A_IN_mean"]

per_run.to_csv("analysis_scratch/wind_effect_per_condition_per_run.csv", index=False)
print("   CSV → analysis_scratch/wind_effect_per_condition_per_run.csv")


# ── Long-format aggregation per (freq, amp, wind, probe) ─────────────────
LONG_PROBES = ["IN_wall", "IN_far", "OUT", "IN_mean"]

long_rows = []
for (f_p, amp_v, wind), grp in per_run.groupby(["freq_hz", "amp_v", "wind"], sort=True):
    for tag in LONG_PROBES:
        A_arr     = grp[f"A_{tag}"].dropna().to_numpy()
        sig_arr   = grp[f"sigma_pre_{tag}"].dropna().to_numpy() if tag != "IN_mean" else \
                    grp["sigma_pre_IN_mean"].dropna().to_numpy()
        Afft_arr  = grp.get(f"A_FFT_meta_{tag}", pd.Series(dtype=float)).dropna().to_numpy()
        phi_arr   = grp[f"phi_{tag}"].dropna().to_numpy() if tag != "IN_mean" else np.array([])
        phi_mean, phi_std = circ_mean(phi_arr) if phi_arr.size else (np.nan, np.nan)
        long_rows.append({
            "freq_hz":   f_p,
            "amp_v":     amp_v,
            "amp_tag":   AMP_LABEL.get(amp_v, f"A?({amp_v})"),
            "wind":      wind,
            "probe_tag": tag,
            "n_runs":    int(A_arr.size),
            "A_mean_mm": float(A_arr.mean()) if A_arr.size else np.nan,
            "A_std_mm":  float(A_arr.std(ddof=1)) if A_arr.size > 1 else np.nan,
            "A_FFT_meta_mean_mm": float(Afft_arr.mean()) if Afft_arr.size else np.nan,
            "sigma_pre_mean_mm":  float(sig_arr.mean()) if sig_arr.size else np.nan,
            "sigma_pre_std_mm":   float(sig_arr.std(ddof=1)) if sig_arr.size > 1 else np.nan,
            "phi_mean_rad": phi_mean,
            "phi_std_rad":  phi_std,
        })

long_df = pd.DataFrame(long_rows)
long_df.to_csv("analysis_scratch/wind_effect_per_condition_long.csv", index=False)
print(f"   CSV → analysis_scratch/wind_effect_per_condition_long.csv "
      f"({len(long_df)} rows)")


# ── Wide ratio table per (freq, amp) ─────────────────────────────────────
def _pivot_A(probe_tag: str, wind: str) -> pd.Series:
    sub = long_df[(long_df["probe_tag"] == probe_tag) & (long_df["wind"] == wind)]
    return sub.set_index(["freq_hz", "amp_v"])["A_mean_mm"]


ratio_rows = []
for (f_p, amp_v), _ in per_run.groupby(["freq_hz", "amp_v"]):
    rec = {"freq_hz": f_p, "amp_v": amp_v, "amp_tag": AMP_LABEL.get(amp_v, "A?")}

    for tag in ["IN_wall", "IN_far", "IN_mean", "OUT"]:
        try:
            A_now  = _pivot_A(tag, "no").loc[(f_p, amp_v)]
        except KeyError:
            A_now = np.nan
        try:
            A_wind = _pivot_A(tag, "full").loc[(f_p, amp_v)]
        except KeyError:
            A_wind = np.nan
        rec[f"A_{tag}_now_mm"]   = A_now
        rec[f"A_{tag}_wind_mm"]  = A_wind
        rec[f"R_{tag}"]          = A_wind / A_now if A_now and np.isfinite(A_now) else np.nan

    # Transmission per IN choice.
    for in_tag in ["IN_wall", "IN_far", "IN_mean"]:
        A_in_now   = rec.get(f"A_{in_tag}_now_mm", np.nan)
        A_in_wind  = rec.get(f"A_{in_tag}_wind_mm", np.nan)
        A_out_now  = rec.get("A_OUT_now_mm", np.nan)
        A_out_wind = rec.get("A_OUT_wind_mm", np.nan)
        T_now  = A_out_now  / A_in_now  if A_in_now  and np.isfinite(A_in_now)  else np.nan
        T_wind = A_out_wind / A_in_wind if A_in_wind and np.isfinite(A_in_wind) else np.nan
        rec[f"T_{in_tag}_now"]  = T_now
        rec[f"T_{in_tag}_wind"] = T_wind
        rec[f"R_T_{in_tag}"]    = T_wind / T_now if T_now and np.isfinite(T_now) else np.nan

    # σ_pre at IN_wall and OUT (means per condition).
    for tag in ["IN_wall", "OUT"]:
        for wind in ["no", "full"]:
            try:
                v = long_df.loc[
                    (long_df["probe_tag"] == tag) &
                    (long_df["wind"] == wind) &
                    (long_df["freq_hz"] == f_p) &
                    (long_df["amp_v"] == amp_v),
                    "sigma_pre_mean_mm"
                ].iloc[0]
            except (IndexError, KeyError):
                v = np.nan
            rec[f"sigma_pre_{tag}_{wind}_mm"] = v

    # Phase deltas.
    for wind in ["no", "full"]:
        sub = per_run[
            (per_run["freq_hz"] == f_p) &
            (per_run["amp_v"] == amp_v) &
            (per_run["wind"] == wind)
        ]
        dphi_arr = sub["dphi_OUT_minus_IN_wall"].dropna().to_numpy()
        m, _ = circ_mean(dphi_arr) if dphi_arr.size else (np.nan, np.nan)
        rec[f"dphi_OUT_IN_wall_{wind}_rad"] = m

    rec["dphi_shift_wind_minus_now_rad"] = wrap_pi(
        (rec.get("dphi_OUT_IN_wall_full_rad", np.nan) or 0.0) -
        (rec.get("dphi_OUT_IN_wall_no_rad",   np.nan) or 0.0)
    ) if (np.isfinite(rec.get("dphi_OUT_IN_wall_full_rad", np.nan))
          and np.isfinite(rec.get("dphi_OUT_IN_wall_no_rad", np.nan))) else np.nan

    ratio_rows.append(rec)

ratio_df = pd.DataFrame(ratio_rows).sort_values(["amp_v", "freq_hz"]).reset_index(drop=True)
ratio_df.to_csv("analysis_scratch/wind_effect_per_condition_ratios.csv", index=False)
print(f"   CSV → analysis_scratch/wind_effect_per_condition_ratios.csv "
      f"({len(ratio_df)} rows)")


# ── Console summary ──────────────────────────────────────────────────────
print("\n=== Per-condition ratios (R = wind/now) — IN_mean, OUT, T ===")
disp = ratio_df[[
    "amp_tag", "freq_hz",
    "A_IN_mean_now_mm", "A_IN_mean_wind_mm", "R_IN_mean",
    "A_OUT_now_mm",     "A_OUT_wind_mm",     "R_OUT",
    "T_IN_mean_now",    "T_IN_mean_wind",    "R_T_IN_mean",
    "sigma_pre_IN_wall_full_mm", "sigma_pre_OUT_full_mm",
]].copy()
for c in disp.columns:
    if disp[c].dtype == float:
        disp[c] = disp[c].round(3)
print(disp.to_string(index=False))

print("\n=== Phase shifts (Δφ_OUT−IN_wall, radians; ΔΔφ = wind − now) ===")
phase_disp = ratio_df[[
    "amp_tag", "freq_hz",
    "dphi_OUT_IN_wall_no_rad", "dphi_OUT_IN_wall_full_rad",
    "dphi_shift_wind_minus_now_rad",
]].copy()
for c in phase_disp.columns:
    if phase_disp[c].dtype == float:
        phase_disp[c] = phase_disp[c].round(4)
print(phase_disp.to_string(index=False))


# ── Figure (a): R_IN, R_OUT, R_T per condition ───────────────────────────
apply_thesis_style(usetex=False)
plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3})

amps_sorted = sorted([a for a in ratio_df["amp_v"].unique() if not np.isnan(a)])
fig, axes = plt.subplots(1, len(amps_sorted), figsize=(3.6 * len(amps_sorted), 4.0),
                         sharey=True)
if len(amps_sorted) == 1:
    axes = [axes]

R_COLOR = {"R_IN_mean": "#1F77B4", "R_OUT": "#2CA02C", "R_T_IN_mean": "#D62728"}
R_MARKER = {"R_IN_mean": "o", "R_OUT": "s", "R_T_IN_mean": "D"}
R_LABEL  = {"R_IN_mean": r"$R_\mathrm{IN}$ (IN mean)",
            "R_OUT":     r"$R_\mathrm{OUT}$",
            "R_T_IN_mean": r"$R_T = T_\mathrm{wind}/T_\mathrm{now}$"}

for ax, amp in zip(axes, amps_sorted):
    sub = ratio_df[ratio_df["amp_v"] == amp].sort_values("freq_hz")
    for col in ("R_IN_mean", "R_OUT", "R_T_IN_mean"):
        ax.plot(sub["freq_hz"], sub[col], "-",
                color=R_COLOR[col], marker=R_MARKER[col], lw=1.4, ms=7,
                label=R_LABEL[col])
    ax.axhline(1.0, color="#888", lw=0.7, ls="--")
    ax.set_xlabel(r"$f_p$ [Hz]")
    ax.set_title(f"{AMP_LABEL.get(amp, '?')} ({amp:.1f} V)")
    ax.set_xticks(CANON_FREQS)
    ax.legend(fontsize=8, loc="upper left")
axes[0].set_ylabel(r"ratio (wind / nowind)")
fig.suptitle("Wind effect at $f_p$ — IN-mean, OUT, transmission ratios", fontsize=11)
fig.tight_layout()

png_a = Path("analysis_scratch/wind_effect_ratios_summary.png")
fig.savefig(png_a, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"\n   PNG → {png_a}")


# ── Figure (b): scatter A_IN, A_OUT, T  vs  σ_pre_IN_wall ────────────────
def _scatter_panel_for_condition(f_p: float, amp_v: float, ax_row, color_now, color_full, lbl):
    fw = per_run[(per_run["freq_hz"] == f_p) & (per_run["amp_v"] == amp_v) &
                 (per_run["wind"] == "full")]
    nw = per_run[(per_run["freq_hz"] == f_p) & (per_run["amp_v"] == amp_v) &
                 (per_run["wind"] == "no")]

    # x = σ_pre_IN_wall on fullwind runs; nowind reference shown as horizontal line.
    cols_y = [
        ("A_IN_wall",  r"$A_\mathrm{IN,wall}$ [mm]"),
        ("A_OUT",      r"$A_\mathrm{OUT}$ [mm]"),
        ("T_wall",     r"$T = A_\mathrm{OUT}/A_\mathrm{IN,wall}$"),
    ]
    for ax, (col, ylabel) in zip(ax_row, cols_y):
        x_fw = fw["sigma_pre_IN_wall"].to_numpy(float)
        y_fw = fw[col].to_numpy(float)
        ax.scatter(x_fw, y_fw, color=color_full, s=28, alpha=0.85,
                   edgecolors="white", linewidths=0.5, label=f"fullwind (n={len(fw)})")
        # Nowind reference line and band: mean ± 1σ from nowind runs at this condition.
        y_nw = nw[col].dropna().to_numpy(float)
        if y_nw.size:
            m = float(y_nw.mean())
            s = float(y_nw.std(ddof=1)) if y_nw.size > 1 else 0.0
            ax.axhline(m, color=color_now, lw=1.0, ls="--",
                       label=f"nowind mean (n={len(y_nw)})")
            if s > 0:
                ax.axhspan(m - s, m + s, color=color_now, alpha=0.10)
        # Linear regression on fullwind points (informational).
        ok = np.isfinite(x_fw) & np.isfinite(y_fw)
        if ok.sum() >= 3:
            slope, intercept = np.polyfit(x_fw[ok], y_fw[ok], 1)
            xx = np.linspace(x_fw[ok].min(), x_fw[ok].max(), 50)
            ax.plot(xx, slope * xx + intercept, color="#444", lw=0.8, ls=":",
                    label=f"slope={slope:+.3f}")
        ax.set_xlabel(r"$\sigma_\eta^\mathrm{wind}$ at IN-wall (3 s pre-paddle) [mm]")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7, loc="best")
    ax_row[1].set_title(lbl, fontsize=10)


color_now  = WIND_COLOR_MAP["no"]
color_full = WIND_COLOR_MAP["full"]

for amp in CANON_AMPS_V:
    if (1.4, amp) in EXCLUDE_CONDITIONS:
        continue
    if amp not in per_run["amp_v"].unique():
        continue
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0))
    _scatter_panel_for_condition(
        1.4, amp, axes, color_now, color_full,
        rf"$f_p = 1.4$ Hz,  {AMP_LABEL[amp]} ({amp:.1f} V),  IN-wall + OUT, fullpanel",
    )
    fig.suptitle(rf"Per-run wind sensitivity — 1.4 Hz, {AMP_LABEL[amp]}", fontsize=11)
    fig.tight_layout()
    out = Path(f"analysis_scratch/wind_effect_scatter_f14_{AMP_LABEL[amp]}.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"   PNG → {out}")

print("\nDone.")
