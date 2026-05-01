"""
Panel transmission vs bare-tank propagation — March + November bridge.
=======================================================================

Strict March no-wind comparison + November 1.3 Hz wind bridge.

Pipeline
--------
  1. Per-run LS amplitudes at f_p on the 10-period H&G window for IN and OUT
     probes (same fit as wind_effect_per_condition.py).
  2. Aggregate per (campaign, panel_state, wind, freq, amp) to ensemble means.
  3. Compute per-metre growth rate
         k = ln(G^L) / L
     from the bare-tank (nopanel) IN→OUT gain at the actual separation L,
     extrapolate to the panel separation L_panel = 3.027 m
         G_nopanel^(L_panel) = exp(k · L_panel)
     and report
         D    = T_panel^(L_panel) / G_nopanel^(L_panel)
         R_D  = D_wind / D_no_wind
  4. Output two computation blocks side by side:

       (a) March-strict — uses only March panel + March nopanel runs.
           D_no_wind computable for every (f, A); D_wind / R_D set to NaN
           because nopanel + fullwind has no March data.
       (b) November-bridge — at 1.3 Hz only. Pairs November nopanel
           (both wind states, ~3.03 m, IN-centre + OUT-wall) with March
           panel (both wind states, ~3.03 m, IN-wall+far + OUT-centre).
           Cross-campaign + lateral-swap caveat documented in CSV columns.

NO wind-independent-k assumption is baked into the main numbers. A clearly-
labelled "ASSUMPTION" sensitivity column shows what D_wind would be in March
under k_wind = k_no, but it never feeds R_D.

Outputs
-------
    analysis_scratch/nopanel_panel_comparison_per_run.csv
    analysis_scratch/nopanel_panel_comparison_per_condition.csv
    analysis_scratch/nopanel_panel_RD_summary.csv
    analysis_scratch/nopanel_panel_RD_summary.png
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
from wavescripts.plot_utils import apply_thesis_style

FS = 250.0

# ── Probe positions per campaign ────────────────────────────────────────
# Each entry maps role → list of probe positions ("dist_mm/lat_mm" strings).
# The IN amplitude is the mean across all listed IN probes.
CAMPAIGN_PROBES = {
    "march": {
        "IN":  ["9373/170", "9373/340"],           # parallel pair
        "OUT_panel":   "12400/250",                # OUT for panel canon
        "OUT_nopanel": "11800/250",                # OUT for march nopanel rearranging
    },
    "november": {
        "IN":  ["9373/250"],                       # single centre probe
        "OUT_panel":   "12400/170",                # IN/OUT wall-side switched
        "OUT_nopanel": "12400/170",
    },
}

# Physical IN→OUT separations (m).
L_PANEL_M           = 3.027   # 12400 - 9373
L_NOPANEL_MARCH_M   = 2.427   # 11800 - 9373 (march nopanel rearrang)
L_NOPANEL_NOVEMBER_M = 3.027  # 12400 - 9373 (same as panel)

# Conditions to analyse.
CANON_FREQS  = [1.3, 1.4, 1.5, 1.6]
CANON_AMPS_V = [0.1, 0.2, 0.3]
EXCLUDE_CONDITIONS = {(1.6, 0.3)}  # high-freq high-amp dropouts

DATASET_DIRS = [
    # March panel canon
    "waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange",
    "waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange",
    # March nopanel rearrang (2-day window)
    "waveprocessed/PROCESSED-20260305-newProbePos-tett6roof",
    "waveprocessed/PROCESSED-20260306-newProbePos-tett6roof",
    "waveprocessed/PROCESSED-20260307-ProbPos4_31_FPV_2-tett6roof",
    # November bridge campaign (nov_normalt_oppsett config)
    "waveprocessed/PROCESSED-20251110-tett6roof-lowM-ekte580",
    "waveprocessed/PROCESSED-20251112-tett6roof",
    "waveprocessed/PROCESSED-20251113-tett6roof-loosepaneltaped",
]

AMP_LABEL = {0.1: "A1", 0.2: "A2", 0.3: "A3"}


# ── LS fit and helpers ───────────────────────────────────────────────────
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


def ls_fit_at_freq(eta_seg: np.ndarray, fs: float, f: float) -> float:
    """Return A from LS fit y = a sin(2πf t) + b cos(2πf t) + c.  NaN-safe."""
    n = eta_seg.size
    if n < 4 or not np.all(np.isfinite(eta_seg)):
        return np.nan
    t = np.arange(n) / fs
    arg = 2.0 * np.pi * f * t
    A_mat = np.column_stack([np.sin(arg), np.cos(arg), np.ones(n)])
    coeffs, *_ = np.linalg.lstsq(A_mat, eta_seg, rcond=None)
    a, b, _c = coeffs
    return float(np.sqrt(a * a + b * b))


def _series_for(in_pos: str, out_pos: str, panel_cond: str) -> str | None:
    """Dispatch by probe-position pair + panel state.

    Avoids brittle date-string matching: probe positions encode both the
    campaign (lateral choice differs March vs November) and the panel/nopanel
    state (different OUT distance for march nopanel).
    """
    in_pos = str(in_pos); out_pos = str(out_pos)
    # March panel canon — IN-wall+far + OUT-centre, panel installed.
    if in_pos == "9373/170" and out_pos == "12400/250" and panel_cond == "full":
        return "march_panel"
    # March nopanel rearrang — IN at 9373/170 + parallel, OUT moved to 11800/250.
    if in_pos == "9373/170" and out_pos == "11800/250" and panel_cond == "no":
        return "march_nopanel"
    # November — IN-centre + OUT-wall (lateral swap from march canon).
    if in_pos == "9373/250" and out_pos == "12400/170":
        if panel_cond == "full":
            return "nov_panel"
        if panel_cond == "no":
            return "nov_nopanel"
    return None


def _campaign_label(series: str) -> str:
    return {"march_panel": "march", "march_nopanel": "march",
            "nov_panel": "november", "nov_nopanel": "november"}.get(series, "other")


def _separation_for(series: str) -> float:
    if series == "march_panel":   return L_PANEL_M
    if series == "march_nopanel": return L_NOPANEL_MARCH_M
    if series == "nov_panel":     return L_PANEL_M
    if series == "nov_nopanel":   return L_NOPANEL_NOVEMBER_M
    return np.nan


def _probe_set_for(series: str) -> tuple[list[str], str]:
    """Return ([IN probes], OUT probe) for a series."""
    if series in ("march_panel", "march_nopanel"):
        cfg = CAMPAIGN_PROBES["march"]
    elif series in ("nov_panel", "nov_nopanel"):
        cfg = CAMPAIGN_PROBES["november"]
    else:
        return [], ""
    out = cfg["OUT_panel"] if series.endswith("_panel") else cfg["OUT_nopanel"]
    return cfg["IN"], out


# ── Load ──────────────────────────────────────────────────────────────────
print(f"Loading {len(DATASET_DIRS)} dataset folders …")
meta, _, _, _ = load_analysis_data(*DATASET_DIRS, load_processed=False)
proc = {}
for d in DATASET_DIRS:
    proc.update(load_processed_dfs(d))
print(f"  meta {len(meta)} rows, processed_dfs {len(proc)}")


# ── Filter wave runs of interest ────────────────────────────────────────
def _round1(x):
    try:
        return round(float(x), 2)
    except (TypeError, ValueError):
        return np.nan

work = meta[
    (meta["quality_flag"] == "ok") &
    (meta["WindCondition"].isin(["no", "full"])) &
    (meta["WaveFrequencyInput [Hz]"].apply(_round1).isin(CANON_FREQS)) &
    (meta["WaveAmplitudeInput [Volt]"].apply(_round1).isin(CANON_AMPS_V)) &
    (meta["PanelCondition"].isin(["full", "no"]))
].copy()

work["series"] = work.apply(
    lambda r: _series_for(r.get("in_position"),
                          r.get("out_position"),
                          r.get("PanelCondition")),
    axis=1,
)
work = work.dropna(subset=["series"])
print(f"  → {len(work)} candidate runs across "
      f"series={sorted(work['series'].unique())}")


# ── Per-run amplitudes ───────────────────────────────────────────────────
rows = []
for _, row in work.iterrows():
    p = row["path"]
    if p not in proc:
        continue
    df = proc[p]
    f_p = round(float(row["WaveFrequencyInput [Hz]"]), 2)
    amp = round(float(row["WaveAmplitudeInput [Volt]"]), 2)
    if (f_p, amp) in EXCLUDE_CONDITIONS:
        continue

    series = row["series"]
    in_probes, out_probe = _probe_set_for(series)
    L_m = _separation_for(series)

    rec = {
        "path":        p,
        "campaign":    _campaign_label(series),
        "series":      series,
        "panel_state": "panel" if series.endswith("_panel") else "nopanel",
        "wind":        row["WindCondition"],
        "freq_hz":     f_p,
        "amp_v":       amp,
        "amp_tag":     AMP_LABEL.get(amp, f"A?({amp})"),
        "file_date":   str(row.get("file_date", ""))[:10],
        "in_probes":   "+".join(in_probes),
        "out_probe":   out_probe,
        "L_separation_m": L_m,
    }

    # IN amplitudes (one column per probe + mean).
    in_vals = []
    for ip in in_probes:
        col = _eta_col(df, ip)
        win = _hg_window(row, ip)
        if col is None or win is None:
            A = np.nan
        else:
            s, e = win
            A = ls_fit_at_freq(df[col].to_numpy(float)[s:e], FS, f_p)
        rec[f"A_IN[{ip}]_mm"] = A
        if np.isfinite(A):
            in_vals.append(A)
    rec["A_IN_mean_mm"] = float(np.mean(in_vals)) if in_vals else np.nan
    rec["n_in_probes_used"] = len(in_vals)

    # OUT amplitude.
    col = _eta_col(df, out_probe)
    win = _hg_window(row, out_probe)
    if col is None or win is None:
        rec["A_OUT_mm"] = np.nan
    else:
        s, e = win
        rec["A_OUT_mm"] = ls_fit_at_freq(df[col].to_numpy(float)[s:e], FS, f_p)

    rows.append(rec)

per_run = pd.DataFrame(rows)
print(f"  per-run rows: {len(per_run)}")
per_run.to_csv("analysis_scratch/nopanel_panel_comparison_per_run.csv", index=False)
print("   CSV → analysis_scratch/nopanel_panel_comparison_per_run.csv")


# ── Aggregate per (series, freq, amp, wind) ──────────────────────────────
agg_rows = []
for (series, f, a, w), grp in per_run.groupby(["series", "freq_hz", "amp_v", "wind"], sort=True):
    A_in_arr  = grp["A_IN_mean_mm"].dropna().to_numpy()
    A_out_arr = grp["A_OUT_mm"].dropna().to_numpy()
    rec = {
        "series":   series,
        "campaign": _campaign_label(series),
        "panel_state": "panel" if series.endswith("_panel") else "nopanel",
        "freq_hz":  f,
        "amp_v":    a,
        "amp_tag":  AMP_LABEL.get(a, "A?"),
        "wind":     w,
        "L_separation_m": _separation_for(series),
        "n_runs":   int(min(len(A_in_arr), len(A_out_arr))),
        "A_IN_mean_mm":  float(A_in_arr.mean())  if A_in_arr.size  else np.nan,
        "A_IN_std_mm":   float(A_in_arr.std(ddof=1))  if A_in_arr.size  > 1 else np.nan,
        "A_OUT_mean_mm": float(A_out_arr.mean()) if A_out_arr.size else np.nan,
        "A_OUT_std_mm":  float(A_out_arr.std(ddof=1)) if A_out_arr.size > 1 else np.nan,
    }
    rec["G_at_L"] = (rec["A_OUT_mean_mm"] / rec["A_IN_mean_mm"]
                    if rec["A_IN_mean_mm"] and np.isfinite(rec["A_IN_mean_mm"])
                    else np.nan)
    agg_rows.append(rec)

agg = pd.DataFrame(agg_rows).sort_values(["series", "freq_hz", "amp_v", "wind"]).reset_index(drop=True)
agg.to_csv("analysis_scratch/nopanel_panel_comparison_per_condition.csv", index=False)
print(f"   CSV → analysis_scratch/nopanel_panel_comparison_per_condition.csv "
      f"({len(agg)} rows)")


# ── Helpers to look up an aggregated cell ────────────────────────────────
def _lookup(series, f, a, w, col):
    sel = agg[(agg["series"] == series) & (agg["freq_hz"] == f) &
              (agg["amp_v"] == a) & (agg["wind"] == w)]
    if sel.empty:
        return np.nan, 0
    return float(sel.iloc[0][col]), int(sel.iloc[0]["n_runs"])


def _gain(series, f, a, w):
    A_in,  n_in  = _lookup(series, f, a, w, "A_IN_mean_mm")
    A_out, n_out = _lookup(series, f, a, w, "A_OUT_mean_mm")
    if not (np.isfinite(A_in) and A_in > 0 and np.isfinite(A_out)):
        return np.nan, 0
    return A_out / A_in, min(n_in, n_out)


def _extrapolate_G(G_L_obs: float, L_obs: float, L_target: float) -> float:
    """G(L_target) = exp( ln(G(L_obs)) / L_obs * L_target )."""
    if not (np.isfinite(G_L_obs) and G_L_obs > 0 and L_obs > 0 and L_target > 0):
        return np.nan
    k = np.log(G_L_obs) / L_obs
    return float(np.exp(k * L_target))


def _k(G_L_obs: float, L_obs: float) -> float:
    if not (np.isfinite(G_L_obs) and G_L_obs > 0 and L_obs > 0):
        return np.nan
    return float(np.log(G_L_obs) / L_obs)


# ── Wide summary per (freq, amp): march-strict + november-bridge ────────
summary_rows = []
for f in CANON_FREQS:
    for a in CANON_AMPS_V:
        if (f, a) in EXCLUDE_CONDITIONS:
            continue

        rec = {"freq_hz": f, "amp_v": a, "amp_tag": AMP_LABEL.get(a, "A?")}
        rec["L_panel_m"] = L_PANEL_M

        # ── March block ──────────────────────────────────────────
        # Panel transmission @ 3.027 m (no extrapolation, panel sits at L_PANEL_M).
        T_now,  n_T_now  = _gain("march_panel", f, a, "no")
        T_wind, n_T_wind = _gain("march_panel", f, a, "full")

        # Nopanel gain at march L = 2.427 m, then extrapolate to 3.027 m.
        G_no_now,  n_G_no_now  = _gain("march_nopanel", f, a, "no")
        G_no_wind, n_G_no_wind = _gain("march_nopanel", f, a, "full")
        k_no_now  = _k(G_no_now,  L_NOPANEL_MARCH_M)
        k_no_wind = _k(G_no_wind, L_NOPANEL_MARCH_M)
        G_no_now_at_3  = _extrapolate_G(G_no_now,  L_NOPANEL_MARCH_M, L_PANEL_M)
        G_no_wind_at_3 = _extrapolate_G(G_no_wind, L_NOPANEL_MARCH_M, L_PANEL_M)
        D_now_march  = T_now  / G_no_now_at_3  if G_no_now_at_3  and np.isfinite(G_no_now_at_3)  else np.nan
        D_wind_march = T_wind / G_no_wind_at_3 if G_no_wind_at_3 and np.isfinite(G_no_wind_at_3) else np.nan
        R_D_march    = (D_wind_march / D_now_march
                        if np.isfinite(D_now_march) and D_now_march != 0 and np.isfinite(D_wind_march)
                        else np.nan)

        rec.update({
            "march_T_panel_now":             T_now,
            "march_T_panel_wind":            T_wind,
            "march_G_nopanel_now_at_2_427m": G_no_now,
            "march_G_nopanel_wind_at_2_427m": G_no_wind,
            "march_L_nopanel_m":             L_NOPANEL_MARCH_M,
            "march_k_nopanel_now":           k_no_now,
            "march_k_nopanel_wind":          k_no_wind,
            "march_G_nopanel_now_at_3m":     G_no_now_at_3,
            "march_G_nopanel_wind_at_3m":    G_no_wind_at_3,
            "march_D_now":                   D_now_march,
            "march_D_wind":                  D_wind_march,
            "march_R_D":                     R_D_march,
            "march_n_panel_now":             n_T_now,
            "march_n_panel_wind":            n_T_wind,
            "march_n_nopanel_now":           n_G_no_now,
            "march_n_nopanel_wind":          n_G_no_wind,
        })

        # March-only sensitivity: ASSUMPTION k_wind == k_no.
        # Yields D_wind under that assumption — never feeds R_D in main results.
        if np.isfinite(k_no_now):
            G_no_assumeknow_at_3 = float(np.exp(k_no_now * L_PANEL_M))
            D_wind_assumeknow = (T_wind / G_no_assumeknow_at_3
                                 if G_no_assumeknow_at_3 and np.isfinite(T_wind)
                                 else np.nan)
            R_D_assumeknow = (D_wind_assumeknow / D_now_march
                              if D_now_march and np.isfinite(D_now_march) else np.nan)
        else:
            D_wind_assumeknow = np.nan
            R_D_assumeknow = np.nan
        rec["ASSUMPTION_kwind_eq_know_march_D_wind"] = D_wind_assumeknow
        rec["ASSUMPTION_kwind_eq_know_march_R_D"]    = R_D_assumeknow

        # ── November bridge block (only meaningful at 1.3 Hz with both winds) ──
        # Panel side from MARCH, nopanel side from NOVEMBER. Cross-campaign mix.
        G_no_nov_now,  n_G_nov_now  = _gain("nov_nopanel", f, a, "no")
        G_no_nov_wind, n_G_nov_wind = _gain("nov_nopanel", f, a, "full")
        # Nov nopanel sits at 3.027 m → no extrapolation needed.
        D_now_bridge  = T_now  / G_no_nov_now  if G_no_nov_now  and np.isfinite(G_no_nov_now)  else np.nan
        D_wind_bridge = T_wind / G_no_nov_wind if G_no_nov_wind and np.isfinite(G_no_nov_wind) else np.nan
        R_D_bridge    = (D_wind_bridge / D_now_bridge
                         if np.isfinite(D_now_bridge) and D_now_bridge != 0 and np.isfinite(D_wind_bridge)
                         else np.nan)
        rec.update({
            "nov_G_nopanel_now_at_3m":   G_no_nov_now,
            "nov_G_nopanel_wind_at_3m":  G_no_nov_wind,
            "nov_n_nopanel_now":         n_G_nov_now,
            "nov_n_nopanel_wind":        n_G_nov_wind,
            "bridge_D_now_marchpanel_novNopanel":  D_now_bridge,
            "bridge_D_wind_marchpanel_novNopanel": D_wind_bridge,
            "bridge_R_D_marchpanel_novNopanel":    R_D_bridge,
        })

        # Provenance flags.
        rec["panel_source"]   = "march_canon"
        rec["nopanel_source_march"] = "march_rearrang_2.427m_extrapolated"
        rec["nopanel_source_bridge"] = ("november_3.027m_no_extrapolation"
                                        if any(np.isfinite([G_no_nov_now, G_no_nov_wind]))
                                        else "no_data")
        rec["bridge_caveat"]  = ("Nov uses IN-centre (9373/250) + OUT-wall (12400/170); "
                                 "march panel uses IN-wall+far + OUT-centre. Lateral swap.")

        summary_rows.append(rec)

summary = pd.DataFrame(summary_rows).sort_values(["amp_v", "freq_hz"]).reset_index(drop=True)
summary.to_csv("analysis_scratch/nopanel_panel_RD_summary.csv", index=False)
print(f"   CSV → analysis_scratch/nopanel_panel_RD_summary.csv ({len(summary)} rows)")


# ── Console — what's computable, what's missing ──────────────────────────
print("\n=== March-strict block ===")
disp = summary[[
    "amp_tag", "freq_hz",
    "march_T_panel_now", "march_G_nopanel_now_at_2_427m", "march_k_nopanel_now",
    "march_G_nopanel_now_at_3m", "march_D_now",
    "march_T_panel_wind", "march_D_wind", "march_R_D",
    "march_n_panel_now", "march_n_nopanel_now", "march_n_nopanel_wind",
]].copy()
for c in disp.columns:
    if disp[c].dtype == float:
        disp[c] = disp[c].round(3)
print(disp.to_string(index=False))

print("\n=== November bridge block (1.3 Hz) ===")
b = summary[summary["freq_hz"] == 1.3][[
    "amp_tag",
    "march_T_panel_now", "nov_G_nopanel_now_at_3m", "bridge_D_now_marchpanel_novNopanel",
    "march_T_panel_wind", "nov_G_nopanel_wind_at_3m", "bridge_D_wind_marchpanel_novNopanel",
    "bridge_R_D_marchpanel_novNopanel",
    "nov_n_nopanel_now", "nov_n_nopanel_wind",
]].copy()
for c in b.columns:
    if b[c].dtype == float:
        b[c] = b[c].round(3)
print(b.to_string(index=False))


print("\n=== Sensitivity (ASSUMPTION: k_nopanel,wind = k_nopanel,no) ===")
sens = summary[[
    "amp_tag", "freq_hz", "march_D_now",
    "ASSUMPTION_kwind_eq_know_march_D_wind",
    "ASSUMPTION_kwind_eq_know_march_R_D",
]].copy()
for c in sens.columns:
    if sens[c].dtype == float:
        sens[c] = sens[c].round(3)
print(sens.to_string(index=False))


# ── Figure: D_now (march) + R_D (november bridge) ───────────────────────
apply_thesis_style(usetex=False)
plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3})

fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))

# Left panel — D_no_wind across (f, A) for march-strict.
ax = axes[0]
amps_present = sorted([a for a in summary["amp_v"].unique() if not np.isnan(a)])
amp_color = {0.1: "#1F77B4", 0.2: "#D62728", 0.3: "#2CA02C"}
amp_marker = {0.1: "o", 0.2: "s", 0.3: "^"}
for amp in amps_present:
    sub = summary[summary["amp_v"] == amp].sort_values("freq_hz")
    ax.plot(sub["freq_hz"], sub["march_D_now"],
            color=amp_color[amp], marker=amp_marker[amp], lw=1.4, ms=7,
            label=f"{AMP_LABEL[amp]} ({amp:.1f} V)")
ax.axhline(1.0, color="#888", lw=0.7, ls="--", label=r"$D=1$ (panel = bare tank)")
ax.set_xlabel(r"$f_p$ [Hz]")
ax.set_ylabel(r"$D_\mathrm{no\,wind}$")
ax.set_title(r"March strict — panel vs bare-tank propagation, no wind")
ax.set_xticks(CANON_FREQS)
ax.legend(fontsize=8, loc="upper right")

# Right panel — November bridge at 1.3 Hz: D_no, D_wind, R_D bars per amp.
ax = axes[1]
b = summary[summary["freq_hz"] == 1.3].sort_values("amp_v").reset_index(drop=True)
x = np.arange(len(b))
width = 0.27
ax.bar(x - width, b["bridge_D_now_marchpanel_novNopanel"],   width,
       color="#1F77B4", label=r"$D_\mathrm{no\,wind}$ (bridge)")
ax.bar(x,         b["bridge_D_wind_marchpanel_novNopanel"], width,
       color="#D62728", label=r"$D_\mathrm{wind}$  (bridge)")
ax.bar(x + width, b["bridge_R_D_marchpanel_novNopanel"],    width,
       color="#444",    label=r"$R_D = D_\mathrm{wind}/D_\mathrm{no\,wind}$")
ax.axhline(1.0, color="#888", lw=0.7, ls="--")
ax.set_xticks(x)
ax.set_xticklabels([f"{lbl}\n({a:.1f} V)" for lbl, a in zip(b["amp_tag"], b["amp_v"])])
ax.set_ylabel("ratio")
ax.set_title(r"November bridge @ 1.3 Hz  —  march panel × nov nopanel")
ax.legend(fontsize=8, loc="upper left")

fig.suptitle("Panel transmission vs bare-tank propagation — strict + november bridge", fontsize=11)
fig.tight_layout()

png = Path("analysis_scratch/nopanel_panel_RD_summary.png")
fig.savefig(png, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"\n   PNG → {png}")

print("\nDone.")
