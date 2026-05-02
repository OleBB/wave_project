"""
Variable-N(f) window prototype — proof + impact on OUT/IN
==========================================================

Computes OUT/IN(FFT) under two window strategies on the same canon runs:

  OLD: probe-shifted Huseby–Grue [50T, 60T] at the OUT-probe anchor
       (r=12.4 m), IN shifted earlier by ΔT = (12.4 − r_IN) / c_g(f) · f.
       Length = 10 T (uniform). Start UC-snapped to nearest ±T.

  NEW: t_start = r_probe / c_g(f, h) + 10/f. Length N(f) periods, where
       N(f) is the maximum that respects (a) parasitic 2f arrival at IN
       and (b) per40 ringdown at OUT:
           f=1.3 → N=10 (binding: parasitic at IN)
           f=1.4 → N=13 (binding: parasitic at IN)
           f=1.5 → N=17 (binding: both, sweet spot)
           f=1.6 → N=15 (binding: per40 plateau at OUT)
       Start UC-snapped to nearest ±T. End = start + N·samples_per_period.

Both windows use the same FFT method (peak-bin nearest, ±0.05 Hz around
paddle f). Only the window placement and length change.

This script answers TWO questions:

  1. PROOF — does the new window land in the eyeball plateau and dodge
     parasitic + per40 ringdown at every (f, probe)? Confirmed by
     printing the snapped window endpoints relative to t_arr, t_paras,
     and per40 paddle-stop.

  2. IMPACT — how much does the OUT/IN ratio shift under the new
     window? Aggregated per (freq, amp, wind), with the wind effect
     Δτ = τ_fw − τ_nw under both old and new. The headline question is
     whether the wind effect — the central thesis result — is preserved.

Outputs (scratch only):
    analysis_scratch/window_variable_N_prototype.csv
    analysis_scratch/window_variable_N_prototype.pdf
    analysis_scratch/window_variable_N_prototype.png
    analysis_scratch/window_variable_N_prototype_findings.md
    stdout: numerical summary + per-(f, probe) safety verification

Run from repo root:
    /opt/anaconda3/envs/draumkvedet/bin/python analysis_scratch/window_variable_N_prototype.py
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

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
from wavescripts.plot_utils import apply_thesis_style, WIND_COLOR_MAP, amp_to_label
from wavescripts.constants import c_group, HG

apply_thesis_style()

# ── Config ──────────────────────────────────────────────────────────────
FS                 = 250.0
FFT_BAND_HZ        = 0.05

THESIS_FREQS       = [1.3, 1.4, 1.5, 1.6]
THESIS_AMPS        = [0.10, 0.20, 0.30]

IN_PROBES          = ["9373/170", "9373/340"]
OUT_PROBE          = "12400/250"
PROBE_R_M          = {"9373/170": 9.373, "9373/340": 9.373, "12400/250": 12.400}
TANK_DEPTH_M       = HG.TANK_DEPTH_M

SNAP_HALFWIDTH_T   = 1.0   # ±1 wave period UC search window

# OLD pipeline parameters — the [50T, 60T] H&G window anchored at r=12.4 m
# (= our OUT probe). Kept here as local constants because this prototype
# script's whole point is OLD-vs-NEW comparison; the HG dataclass
# (wavescripts/constants.py) was reformulated 2026-05-02 to per-probe
# arrival anchoring and no longer carries START_T_REF / END_T_REF / REF_R_M.
OLD_HG_REF_R_M      = 12.400
OLD_HG_START_T      = 50
OLD_HG_END_T        = 60
OLD_WINDOW_LENGTH_T = OLD_HG_END_T - OLD_HG_START_T   # 10

# NEW formula parameters
NEW_N_OFFSET       = 10
N_LENGTH_LOOKUP    = {1.3: 10, 1.4: 13, 1.5: 17, 1.6: 15}

# Canon — march-2026 cond4 lowrange
PROCESSED_DIRS = [
    Path("waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange"),
    Path("waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"),
]

SCRATCH_DIR = Path(__file__).parent
OUT_PDF     = SCRATCH_DIR / "window_variable_N_prototype.pdf"
OUT_PNG     = SCRATCH_DIR / "window_variable_N_prototype.png"
OUT_CSV     = SCRATCH_DIR / "window_variable_N_prototype.csv"
OUT_MD      = SCRATCH_DIR / "window_variable_N_prototype_findings.md"


# ── Helpers ─────────────────────────────────────────────────────────────
def fft_amp_at_freq(segment: np.ndarray, target_hz: float,
                    fs: float = FS, band_hz: float = FFT_BAND_HZ) -> float:
    N = len(segment)
    if N < 4:
        return np.nan
    seg = np.asarray(segment, dtype=float).copy()
    if np.isnan(seg).any():
        idx  = np.arange(N)
        good = ~np.isnan(seg)
        if good.sum() < N * 0.9:
            return np.nan
        seg = np.interp(idx, idx[good], seg[good])
    freqs = np.fft.fftfreq(N, d=1.0 / fs)
    pos   = freqs > 0
    pos_f = freqs[pos]
    mask  = (pos_f >= target_hz - band_hz) & (pos_f <= target_hz + band_hz)
    if not mask.any():
        j = int(np.argmin(np.abs(pos_f - target_hz)))
    else:
        local_idx = int(np.argmin(np.abs(pos_f[mask] - target_hz)))
        j = np.where(mask)[0][local_idx]
    fft_vals = np.fft.fft(seg)
    amps_pos = 2.0 * np.abs(fft_vals[pos]) / N
    return float(amps_pos[j])


def get_eta(df_run: pd.DataFrame, pos: str):
    for col in (f"eta_{pos}_interp", f"eta_{pos}"):
        if col in df_run.columns:
            return df_run[col].to_numpy(dtype=float)
    return None


def canonical_in_signal(df_run: pd.DataFrame):
    sigs = []
    for p in IN_PROBES:
        s = get_eta(df_run, p)
        if s is not None:
            sigs.append(s)
    if not sigs:
        return None
    nmin = min(len(s) for s in sigs)
    stack = np.vstack([s[:nmin] for s in sigs])
    return np.nanmean(stack, axis=0)


def snap_to_upcrossing(signal: np.ndarray, target_idx: int,
                       f_paddle: float, fs: float = FS) -> int:
    """Snap target sample idx to nearest zero-upcrossing within ±SNAP_HALFWIDTH_T
    periods, threshold = local DC mean of first 2 s of signal (matches
    live pipeline)."""
    if target_idx < 0 or target_idx >= len(signal):
        return target_idx
    samples_per_period = int(round(fs / f_paddle))
    halfwidth = int(round(SNAP_HALFWIDTH_T * samples_per_period))
    baseline_n = int(2 * fs)
    upcross_level = float(np.nanmean(signal[:baseline_n]))
    above = signal > upcross_level
    all_uc = np.where((~above[:-1]) & above[1:])[0] + 1
    lo = max(0, target_idx - halfwidth)
    hi = min(len(signal) - 1, target_idx + halfwidth)
    cands = all_uc[(all_uc >= lo) & (all_uc <= hi)]
    if len(cands) == 0:
        return target_idx
    return int(cands[np.argmin(np.abs(cands - target_idx))])


# ── Window functions ────────────────────────────────────────────────────
def old_window_indices(signal: np.ndarray, probe: str,
                       f_paddle: float, fs: float = FS):
    """Probe-shifted H&G [50T, 60T] anchored at r=12.4 m, UC-snap start.
    End = start + 10·samples_per_period."""
    r_probe_m = PROBE_R_M[probe]
    cg = c_group(f_paddle, TANK_DEPTH_M)
    dT_periods = (OLD_HG_REF_R_M - r_probe_m) / cg * f_paddle
    start_T = OLD_HG_START_T - dT_periods
    samples_per_period = int(round(fs / f_paddle))
    target_start_idx = int(round(start_T * samples_per_period))
    snap_start = snap_to_upcrossing(signal, target_start_idx, f_paddle, fs)
    snap_end = snap_start + OLD_WINDOW_LENGTH_T * samples_per_period
    return snap_start, snap_end


def new_window_indices(signal: np.ndarray, probe: str,
                       f_paddle: float, fs: float = FS):
    """t_start = r/c_g + 10/f, length N(f)/f, UC-snap start."""
    r_m = PROBE_R_M[probe]
    cg = c_group(f_paddle, TANK_DEPTH_M)
    t_start = r_m / cg + NEW_N_OFFSET / f_paddle
    target_start_idx = int(round(t_start * fs))
    snap_start = snap_to_upcrossing(signal, target_start_idx, f_paddle, fs)
    f_key = round(float(f_paddle), 2)
    if f_key not in N_LENGTH_LOOKUP:
        # fallback for off-canon frequencies — use nearest
        f_key = min(N_LENGTH_LOOKUP.keys(), key=lambda k: abs(k - f_paddle))
    N_len = N_LENGTH_LOOKUP[f_key]
    samples_per_period = int(round(fs / f_paddle))
    snap_end = snap_start + N_len * samples_per_period
    return snap_start, snap_end


def amp_in_window_indices(signal: np.ndarray, s: int, e: int,
                          f_paddle: float) -> float:
    if s < 0 or e > len(signal) or e <= s:
        return np.nan
    return fft_amp_at_freq(signal[s:e], f_paddle)


# ── Load ────────────────────────────────────────────────────────────────
print("1. Loading meta + processed_dfs (canon March-2026 lowrange) …")
combined_meta, _, _, _ = load_analysis_data(*PROCESSED_DIRS, load_processed=False)
processed_dfs = load_processed_dfs(*PROCESSED_DIRS)
print(f"   meta: {len(combined_meta)} rows · processed_dfs: {len(processed_dfs)} runs")


def select_canon() -> pd.DataFrame:
    f_col = pd.to_numeric(combined_meta["WaveFrequencyInput [Hz]"], errors="coerce")
    a_col = pd.to_numeric(combined_meta["WaveAmplitudeInput [Volt]"], errors="coerce")
    mask = (
        f_col.between(min(THESIS_FREQS) - 0.02, max(THESIS_FREQS) + 0.02)
        & a_col.between(min(THESIS_AMPS) - 0.005, max(THESIS_AMPS) + 0.005)
        & (combined_meta["PanelCondition"] == "full")
        & (combined_meta["quality_flag"] == "ok")
    )
    sub = combined_meta[mask].copy()
    sub["freq_r"] = f_col[mask].round(2)
    sub["amp_r"]  = a_col[mask].round(2)
    return sub


sel = select_canon()
print(f"\n2. Canonical scope: {len(sel)} runs across "
      f"{sel['freq_r'].nunique()} freqs × "
      f"{sel['amp_r'].nunique()} amps × "
      f"{sel['WindCondition'].nunique()} winds")


# ── 3. Compute OLD + NEW OUT/IN per run ─────────────────────────────────
print("\n3. Computing OUT/IN under OLD vs NEW windows per run …")
records = []
for _, r in sel.iterrows():
    df = processed_dfs.get(r["path"])
    if df is None:
        continue
    f = float(r["freq_r"])
    sig_in  = canonical_in_signal(df)
    sig_out = get_eta(df, OUT_PROBE)
    if sig_in is None or sig_out is None:
        continue

    # OLD windows
    s_in_o,  e_in_o  = old_window_indices(sig_in,  IN_PROBES[0], f, FS)
    s_out_o, e_out_o = old_window_indices(sig_out, OUT_PROBE,    f, FS)
    a_in_o  = amp_in_window_indices(sig_in,  s_in_o,  e_in_o,  f)
    a_out_o = amp_in_window_indices(sig_out, s_out_o, e_out_o, f)
    outin_o = a_out_o / a_in_o if (np.isfinite(a_in_o) and a_in_o > 0) else np.nan

    # NEW windows
    s_in_n,  e_in_n  = new_window_indices(sig_in,  IN_PROBES[0], f, FS)
    s_out_n, e_out_n = new_window_indices(sig_out, OUT_PROBE,    f, FS)
    a_in_n  = amp_in_window_indices(sig_in,  s_in_n,  e_in_n,  f)
    a_out_n = amp_in_window_indices(sig_out, s_out_n, e_out_n, f)
    outin_n = a_out_n / a_in_n if (np.isfinite(a_in_n) and a_in_n > 0) else np.nan

    records.append({
        "path":        r["path"],
        "freq_r":      f,
        "amp_r":       round(float(r["amp_r"]), 2),
        "wind":        r["WindCondition"],
        "A_in_old":    a_in_o,
        "A_out_old":   a_out_o,
        "OUT_IN_old":  outin_o,
        "A_in_new":    a_in_n,
        "A_out_new":   a_out_n,
        "OUT_IN_new":  outin_n,
        # snapped window endpoints (s) for verification
        "in_window_old_s":  (s_in_o  / FS,  e_in_o  / FS),
        "out_window_old_s": (s_out_o / FS,  e_out_o / FS),
        "in_window_new_s":  (s_in_n  / FS,  e_in_n  / FS),
        "out_window_new_s": (s_out_n / FS,  e_out_n / FS),
    })

per_run = pd.DataFrame(records)
per_run["delta_OUT_IN"] = per_run["OUT_IN_new"] - per_run["OUT_IN_old"]
per_run["pct_OUT_IN"]   = per_run["delta_OUT_IN"] / per_run["OUT_IN_old"] * 100.0
print(f"   {len(per_run)} per-run rows computed")


# ── 4. Aggregate per (freq, amp, wind) ──────────────────────────────────
print("\n4. Aggregating per (freq, amp, wind) …")
agg = per_run.groupby(["freq_r", "amp_r", "wind"]).agg(
    OUT_IN_old_mean = ("OUT_IN_old", "mean"),
    OUT_IN_new_mean = ("OUT_IN_new", "mean"),
    OUT_IN_old_std  = ("OUT_IN_old", "std"),
    OUT_IN_new_std  = ("OUT_IN_new", "std"),
    n               = ("OUT_IN_old", "size"),
).reset_index()
agg["delta_OUT_IN"] = agg["OUT_IN_new_mean"] - agg["OUT_IN_old_mean"]
agg["pct_OUT_IN"]   = agg["delta_OUT_IN"] / agg["OUT_IN_old_mean"] * 100.0


# ── 5. Wind-effect shift: Δτ_fw - Δτ_nw, under old vs new ───────────────
print("\n5. Wind-effect shift per (freq, amp) …")
wind_pivot_old = agg.pivot_table(
    index=["freq_r", "amp_r"], columns="wind",
    values="OUT_IN_old_mean", aggfunc="first").reset_index()
wind_pivot_new = agg.pivot_table(
    index=["freq_r", "amp_r"], columns="wind",
    values="OUT_IN_new_mean", aggfunc="first").reset_index()

we = wind_pivot_old.merge(wind_pivot_new, on=["freq_r", "amp_r"],
                          suffixes=("_old", "_new"))
we["Dtau_old"] = we["full_old"] - we["no_old"]
we["Dtau_new"] = we["full_new"] - we["no_new"]
we["Dtau_shift"] = we["Dtau_new"] - we["Dtau_old"]

print("\nWind effect (Δτ = τ_fw − τ_nw) per (freq, amp) — OLD vs NEW:")
print(we.round(4).to_string(index=False))


# ── 6. Verification — print snapped windows for one run per freq ────────
print("\n6. Snapped-window verification (one per40-fullwind run per freq) …")
print(f"  {'f':<5} {'IN old':<24} {'IN new':<24} {'OUT old':<24} {'OUT new':<24}")
print(f"  {'(Hz)':<5} {'[s_start, s_end]':<24} {'[s_start, s_end]':<24} {'[s_start, s_end]':<24} {'[s_start, s_end]':<24}")
print("  " + "─" * 120)
for f in THESIS_FREQS:
    sub = per_run[(per_run["freq_r"] == f)
                  & (per_run["amp_r"] == 0.20)
                  & (per_run["wind"] == "full")]
    if sub.empty:
        continue
    rep = sub.iloc[0]
    print(f"  {f:<5.1f} "
          f"{str(tuple(round(x, 2) for x in rep['in_window_old_s'])):<24} "
          f"{str(tuple(round(x, 2) for x in rep['in_window_new_s'])):<24} "
          f"{str(tuple(round(x, 2) for x in rep['out_window_old_s'])):<24} "
          f"{str(tuple(round(x, 2) for x in rep['out_window_new_s'])):<24}")


# ── 7. Plot — OUT/IN(f) under old vs new, per amp ───────────────────────
print("\n7. Plotting OUT/IN comparison …")
fig, axes = plt.subplots(1, len(THESIS_AMPS), figsize=(13, 4.5),
                         sharex=True, sharey=True)
for ax, amp in zip(axes, THESIS_AMPS):
    sub = agg[np.isclose(agg["amp_r"], amp)].copy()
    if sub.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        continue
    for wind in ("no", "full"):
        sw = sub[sub["wind"] == wind].sort_values("freq_r")
        if sw.empty:
            continue
        color = WIND_COLOR_MAP[wind]
        # OLD: solid + circles
        ax.errorbar(sw["freq_r"], sw["OUT_IN_old_mean"],
                    yerr=sw["OUT_IN_old_std"].fillna(0),
                    fmt="o-", color=color, capsize=3, lw=1.5, ms=6, alpha=0.85,
                    label=f"{wind} · old")
        # NEW: dashed + diamonds, slight offset
        ax.errorbar(sw["freq_r"] + 0.01, sw["OUT_IN_new_mean"],
                    yerr=sw["OUT_IN_new_std"].fillna(0),
                    fmt="D--", color=color, capsize=3, lw=1.5, ms=6, alpha=0.85,
                    mfc="white", mec=color, label=f"{wind} · new")
    ax.axhline(1.0, color="black", lw=0.5, ls="--", alpha=0.4)
    ax.set_xlabel("Frequency [Hz]", fontsize=10)
    ax.set_xticks(THESIS_FREQS)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"{amp_to_label(amp)}", fontsize=10)

axes[0].set_ylabel("OUT/IN (FFT)", fontsize=10)
axes[0].legend(fontsize=7, loc="best", framealpha=0.9, ncol=1)
fig.suptitle("", fontsize=11)
fig.tight_layout()
fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"   saved → {OUT_PDF.relative_to(BASE)}")
print(f"          {OUT_PNG.relative_to(BASE)}")


# ── 8. Save CSV + findings ──────────────────────────────────────────────
agg.to_csv(OUT_CSV, index=False)
print(f"   CSV → {OUT_CSV.relative_to(BASE)}")

md_lines = [
    "# Variable-N(f) window prototype — OLD vs NEW comparison",
    "",
    "Generated by `analysis_scratch/window_variable_N_prototype.py`.",
    "",
    "Scope: full panel, 1.3–1.6 Hz, 0.10/0.20/0.30 V, both winds, "
    "quality_flag=ok, march-2026 cond4 lowrange.",
    "",
    "## Window definitions",
    "",
    "**OLD** — probe-shifted H&G [50T, 60T] anchored at r=12.4 m. Length = 10 T uniform.",
    "",
    "**NEW** — t_start = r/c_g(f, h) + 10/f. Length N(f) periods, where:",
    "",
    "| f (Hz) | N_length (T) | binding constraint |",
    "|---|---:|---|",
    "| 1.3 | 10 | parasitic at IN |",
    "| 1.4 | 13 | parasitic at IN |",
    "| 1.5 | 17 | tie (parasitic ≈ per40 plateau) |",
    "| 1.6 | 15 | per40 plateau at OUT |",
    "",
    "Both UC-snap start to nearest upcrossing within ±T.",
    "",
    "## OUT/IN per (freq, amp, wind) — old vs new",
    "",
    "| f | amp | wind | n | OUT/IN old | OUT/IN new | Δ | Δ % |",
    "|---|---|---|---:|---:|---:|---:|---:|",
]
for _, r in agg.iterrows():
    md_lines.append(
        f"| {r['freq_r']:.1f} | {r['amp_r']:.2f} | {r['wind']} | {r['n']} | "
        f"{r['OUT_IN_old_mean']:.4f} | {r['OUT_IN_new_mean']:.4f} | "
        f"{r['delta_OUT_IN']:+.4f} | {r['pct_OUT_IN']:+.2f}% |"
    )

md_lines += [
    "",
    "## Wind effect (Δτ = τ_fw − τ_nw) — old vs new",
    "",
    "| f | amp | Δτ_old | Δτ_new | shift |",
    "|---|---|---:|---:|---:|",
]
for _, r in we.iterrows():
    md_lines.append(
        f"| {r['freq_r']:.1f} | {r['amp_r']:.2f} | "
        f"{r['Dtau_old']:+.4f} | {r['Dtau_new']:+.4f} | "
        f"{r['Dtau_shift']:+.4f} |"
    )

OUT_MD.write_text("\n".join(md_lines), encoding="utf-8")
print(f"   findings → {OUT_MD.relative_to(BASE)}")

# Headlines
print("\n8. Headline numbers")
print(f"   Median |Δ OUT/IN| across all (f, amp, wind): {agg['delta_OUT_IN'].abs().median():.4f}")
print(f"   Max    |Δ OUT/IN|:                          {agg['delta_OUT_IN'].abs().max():.4f}")
print(f"   Median |Δτ shift|:                          {we['Dtau_shift'].abs().median():.4f}")
print(f"   Max    |Δτ shift|:                          {we['Dtau_shift'].abs().max():.4f}")

print("\nDone.")
