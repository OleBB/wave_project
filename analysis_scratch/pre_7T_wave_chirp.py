"""
Pre-7T-start wave-chirp characterization (2026-05-05)
=====================================================

Before we claim "wind makes the paddle wave arrive earlier" based on the
snap-shift Δt, we need to see what's actually in η(t) BEFORE the 7T-start
window opens. The wavemaker's soft-start ramp produces a chirp:

  dead-flat (~3 s) → tiny long-period motion → growing, period-shortening
  wave train → settles at paddle frequency → enters 7T-start window.

If the snap latches onto a chirp-region upcrossing (NOT yet at paddle
frequency), the apparent Δt under fullwind may be measuring a chirp-vs-
wind interaction, not a propagation effect.

Method
------
Per (freq, amp, wind) cell on canon March-2026 lowrange + full panel,
take ONE canonical run (first ok run). At IN (9373/170):

  1. Detect zero upcrossings of eta_9373/170 from t=0 to t_expected + 2T.
  2. Compute instantaneous period T_i = t_{uc,i+1} - t_{uc,i}.
  3. Compute per-cycle amplitude (max - min)/2 between consecutive uc's.
  4. Find time-to-settle: first uc where |T_i - T_paddle|/T_paddle < 0.10
     for that cycle AND the next 2 cycles (3-in-a-row stability).

Outputs
-------
analysis_scratch/pre_7T_wave_chirp_eta.png         η(t) grid 4×3 cells
analysis_scratch/pre_7T_wave_chirp_period.png      T_i(t) grid 4×3 cells
analysis_scratch/pre_7T_wave_chirp_detail_1.3Hz_0.2V.png  detail panel
analysis_scratch/pre_7T_wave_chirp_settle.csv      time-to-settle table
analysis_scratch/pre_7T_wave_chirp.md              findings
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
WINDS = ["no", "full"]
PROBE = "9373/170"     # IN reference; fully exposed to wind
WIND_COLOR = {"no": "#1f77b4", "full": "#d62728"}

SETTLE_TOL = 0.10      # ±10% of T_paddle
SETTLE_RUN = 3         # 3 consecutive cycles required


# ---------------------------------------------------------------------------
# Loaders
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
    df = df[df["WindCondition"].isin(WINDS)]
    return df.copy()


def load_processed(canon_dirs):
    """Load processed_dfs once for both canon folders, return dict of paths."""
    print("Loading processed_dfs from canon folders (~45s)...")
    dfs = []
    for d in canon_dirs:
        dfs.append(pd.read_parquet(d / "processed_dfs.parquet"))
    big = pd.concat(dfs, ignore_index=True)
    print(f"  {len(big):,} rows across {big['_path'].nunique()} runs")
    return big


# ---------------------------------------------------------------------------
# Wave-chirp analysis per run
# ---------------------------------------------------------------------------
def _light_smooth(eta: np.ndarray, win_samples: int = 25) -> np.ndarray:
    """Rolling-mean smoother to suppress sub-100ms ripple. Doesn't touch
    structure at the wave timescales we care about (T_pad >= 625 ms,
    T_pad/4 = 156 ms for 1.6 Hz — well above 100 ms window)."""
    if win_samples <= 1:
        return eta
    kern = np.ones(win_samples) / win_samples
    return np.convolve(eta, kern, mode="same")


def detect_upcrossings(eta: np.ndarray,
                       lockout_samples: int = 0) -> np.ndarray:
    """Return sample indices where eta crosses zero with positive slope.

    Time-lockout: after each upcrossing, ignore further candidates for
    `lockout_samples` samples. This rejects spurious extra crossings
    that occur when a slow wave hovers near zero (small ripples cause
    multiple crossings within one real cycle), without rejecting any
    real wave whose period exceeds the lockout interval.

    Works on the AMPLITUDE-AGNOSTIC sign of the signal — small precursor
    waves down to noise level still register as long as they're spaced
    more than `lockout_samples` apart.
    """
    s = eta - np.nanmean(eta[:int(3 * FS)])  # remove pre-paddle DC
    ucs = []
    last = -lockout_samples - 1
    for i in range(1, len(s)):
        if s[i - 1] <= 0 < s[i] and (i - last) > lockout_samples:
            ucs.append(i)
            last = i
    return np.array(ucs, dtype=int)


def lockout_for_paddle(t_paddle: float, fraction: float = 0.25) -> int:
    """Lockout length = fraction · T_paddle, in samples.
    Default 1/4 of paddle period = max detectable freq is 4·f_paddle."""
    return int(round(fraction * t_paddle * FS))


def per_cycle_metrics(eta: np.ndarray, ucs: np.ndarray) -> pd.DataFrame:
    """Period and amplitude between each pair of consecutive upcrossings."""
    rows = []
    for i in range(len(ucs) - 1):
        i0, i1 = ucs[i], ucs[i + 1]
        cycle = eta[i0:i1]
        T_i = (i1 - i0) / FS
        if len(cycle) >= 2:
            amp_i = (np.nanmax(cycle) - np.nanmin(cycle)) / 2
        else:
            amp_i = np.nan
        rows.append({
            "uc_idx":  i0,
            "uc_time": i0 / FS,
            "T_i":     T_i,
            "amp_i":   amp_i,
            "f_i":     1.0 / T_i if T_i > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def time_to_settle(metrics: pd.DataFrame, T_paddle: float,
                   tol: float = SETTLE_TOL,
                   run: int = SETTLE_RUN) -> tuple[float, int]:
    """First upcrossing time where T_i is within ±tol of T_paddle for `run`
    consecutive cycles. Returns (time, uc_index) or (np.nan, -1)."""
    rel_err = (metrics["T_i"] - T_paddle).abs() / T_paddle
    in_band = rel_err < tol
    for i in range(len(in_band) - run + 1):
        if in_band.iloc[i:i + run].all():
            return float(metrics["uc_time"].iloc[i]), i
    return np.nan, -1


# ---------------------------------------------------------------------------
# Pick canonical run per cell
# ---------------------------------------------------------------------------
def pick_runs(meta: pd.DataFrame) -> dict:
    """Return {(f, amp, wind): meta_row} — first ok run per cell."""
    chosen = {}
    for f_hz in FREQS:
        for amp in AMPS:
            for w in WINDS:
                cell = meta[
                    (meta["WaveFrequencyInput [Hz]"] == f_hz)
                    & (meta["WaveAmplitudeInput [Volt]"] == amp)
                    & (meta["WindCondition"] == w)
                ]
                if not cell.empty:
                    chosen[(f_hz, amp, w)] = cell.iloc[0]
    return chosen


# ---------------------------------------------------------------------------
# Plot: η(t) grid 4×3, fw vs nw overlaid
# ---------------------------------------------------------------------------
def plot_eta_grid(meta: pd.DataFrame, big: pd.DataFrame,
                  chosen: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(len(FREQS), len(AMPS),
                             figsize=(15, 12), sharex=False, sharey=False)
    eta_col = f"eta_{PROBE}"

    for i, f_hz in enumerate(FREQS):
        T_paddle = 1.0 / f_hz
        for j, amp in enumerate(AMPS):
            ax = axes[i, j]
            for w in WINDS:
                key = (f_hz, amp, w)
                if key not in chosen:
                    continue
                row = chosen[key]
                run_eta = big[big["_path"] == row["path"]][eta_col].values
                t = np.arange(len(run_eta)) / FS

                # Theoretical 7T-start at IN (identical fw/nw)
                t_exp = row[f"Probe {PROBE} hg_expected_start"] / FS
                # Snapped start (per run)
                t_snap = row[f"Computed Probe {PROBE} start"] / FS
                xlim = (0, t_exp + 2 * T_paddle)

                ax.plot(t, run_eta, color=WIND_COLOR[w], lw=0.6,
                        alpha=0.85, label=w)
                ax.axvline(t_snap, color=WIND_COLOR[w], ls="-",
                           lw=1.0, alpha=0.5)

            # Reference markers
            ax.axvline(t_exp, color="k", ls=":", lw=0.7,
                       label="7T-start (theory)")
            ax.axhline(0, color="k", lw=0.3)
            ax.set_xlim(xlim)
            ax.set_title(f"{f_hz} Hz, {amp} V", fontsize=9)
            ax.grid(alpha=0.3)
            if j == 0:
                ax.set_ylabel("η [mm]")
            if i == len(FREQS) - 1:
                ax.set_xlabel("time [s]")
            if i == 0 and j == 0:
                ax.legend(fontsize=7)

    fig.suptitle(f"η(t) at IN ({PROBE}) — pre-7T-start chirp + arrival\n"
                 f"vertical lines: black-dotted=7T theory, "
                 f"colored-solid=actual snap", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot: instantaneous period T_i(t) grid 4×3, fw vs nw overlaid
# ---------------------------------------------------------------------------
def plot_period_grid(meta: pd.DataFrame, big: pd.DataFrame, chosen: dict,
                     out_path: Path, settle_table: list) -> None:
    fig, axes = plt.subplots(len(FREQS), len(AMPS),
                             figsize=(15, 12), sharex=False)
    eta_col = f"eta_{PROBE}"

    for i, f_hz in enumerate(FREQS):
        T_paddle = 1.0 / f_hz
        for j, amp in enumerate(AMPS):
            ax = axes[i, j]
            for w in WINDS:
                key = (f_hz, amp, w)
                if key not in chosen:
                    continue
                row = chosen[key]
                eta = big[big["_path"] == row["path"]][eta_col].values
                t_exp = row[f"Probe {PROBE} hg_expected_start"] / FS
                t_snap = row[f"Computed Probe {PROBE} start"] / FS

                # restrict analysis window to start..t_exp + 2T
                end_idx = int((t_exp + 2 * T_paddle) * FS)
                eta_win = eta[:end_idx]
                eta_sm = _light_smooth(eta_win)
                lock = lockout_for_paddle(T_paddle)
                ucs = detect_upcrossings(eta_sm, lockout_samples=lock)
                metrics = per_cycle_metrics(eta_win, ucs)

                t_settle, idx_settle = time_to_settle(metrics, T_paddle)
                settle_table.append({
                    "f_hz": f_hz, "amp_V": amp, "wind": w,
                    "n_upcrossings": len(ucs),
                    "t_first_uc":  metrics["uc_time"].min() if len(metrics) else np.nan,
                    "t_settle":    t_settle,
                    "t_exp_7T":    t_exp,
                    "t_snap":      t_snap,
                    "settle_minus_snap": (t_settle - t_snap) if np.isfinite(t_settle) else np.nan,
                })

                ax.plot(metrics["uc_time"], metrics["T_i"], "o-",
                        color=WIND_COLOR[w], lw=0.8, ms=3, alpha=0.85,
                        label=w)
                if np.isfinite(t_settle):
                    ax.axvline(t_settle, color=WIND_COLOR[w], ls="--",
                               lw=0.8, alpha=0.7)

            # Reference: paddle period and tolerance band
            ax.axhline(T_paddle, color="k", lw=0.5)
            ax.axhline(T_paddle * (1 + SETTLE_TOL), color="grey", lw=0.3, ls=":")
            ax.axhline(T_paddle * (1 - SETTLE_TOL), color="grey", lw=0.3, ls=":")
            ax.axvline(t_exp, color="k", ls=":", lw=0.6)

            ax.set_title(f"{f_hz} Hz, {amp} V (T_pad={T_paddle*1000:.0f} ms)",
                         fontsize=9)
            ax.grid(alpha=0.3)
            ax.set_ylim(0, 3 * T_paddle)
            if j == 0:
                ax.set_ylabel("T_i [s]")
            if i == len(FREQS) - 1:
                ax.set_xlabel("time of upcrossing [s]")
            if i == 0 and j == 0:
                ax.legend(fontsize=7)

    fig.suptitle(f"Instantaneous period T_i(t) at IN ({PROBE}) — chirp toward "
                 f"paddle period\n"
                 f"dashed verticals = settle time (3 cycles within ±10%); "
                 f"dotted vertical = 7T-start theory", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Detail plot for one cell: 3 rows × 2 cols (fw|nw)
# ---------------------------------------------------------------------------
def plot_detail(meta: pd.DataFrame, big: pd.DataFrame, chosen: dict,
                out_path: Path, f_hz: float = 1.3, amp: float = 0.2) -> None:
    eta_col = f"eta_{PROBE}"
    T_paddle = 1.0 / f_hz

    fig, axes = plt.subplots(3, 2, figsize=(15, 9), sharex="col", sharey="row")
    for col, w in enumerate(WINDS):
        key = (f_hz, amp, w)
        if key not in chosen:
            continue
        row = chosen[key]
        eta = big[big["_path"] == row["path"]][eta_col].values
        t_exp = row[f"Probe {PROBE} hg_expected_start"] / FS
        t_snap = row[f"Computed Probe {PROBE} start"] / FS

        end_idx = int((t_exp + 2 * T_paddle) * FS)
        eta_win = eta[:end_idx]
        t = np.arange(len(eta_win)) / FS
        eta_sm = _light_smooth(eta_win)
        lock = lockout_for_paddle(T_paddle)
        ucs = detect_upcrossings(eta_sm, lockout_samples=lock)
        metrics = per_cycle_metrics(eta_win, ucs)
        t_settle, _ = time_to_settle(metrics, T_paddle)

        clr = WIND_COLOR[w]

        # Row 0: η(t) with upcrossings marked
        ax = axes[0, col]
        ax.plot(t, eta_win, color=clr, lw=0.6, alpha=0.4, label="raw")
        ax.plot(t, eta_sm, color=clr, lw=0.7,
                label=f"smoothed (25-sample = 100 ms)")
        ax.axhline(0, color="k", lw=0.3)
        for uc in ucs:
            ax.axvline(uc / FS, color="grey", lw=0.3, alpha=0.5)
        ax.axvline(t_exp, color="k", ls=":", lw=0.7, label="7T theory")
        ax.axvline(t_snap, color=clr, lw=1.0, alpha=0.7,
                   label=f"snap ({t_snap:.3f}s)")
        if np.isfinite(t_settle):
            ax.axvline(t_settle, color="orange", ls="--", lw=1.0,
                       label=f"settle ({t_settle:.3f}s)")
        lock_ms = lock * 1000 / FS
        ax.set_title(f"{w} — {f_hz} Hz, {amp} V "
                     f"(lockout {lock_ms:.0f} ms)", fontsize=11)
        ax.set_ylabel("η [mm]")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)

        # Row 1: T_i(t) — log y-axis to show the full sweep from precursor
        # (T_i can reach 5-10 s for the earliest detectable cycles) down to
        # T_pad. Log compresses the long-tail without truncating data.
        ax = axes[1, col]
        ax.plot(metrics["uc_time"], metrics["T_i"], "o-", color=clr,
                lw=0.8, ms=4)
        ax.axhline(T_paddle, color="k", lw=0.5,
                   label=f"T_pad = {T_paddle*1000:.0f} ms")
        ax.axhline(T_paddle * (1 + SETTLE_TOL), color="grey", lw=0.3, ls=":")
        ax.axhline(T_paddle * (1 - SETTLE_TOL), color="grey", lw=0.3, ls=":")
        ax.axvline(t_exp, color="k", ls=":", lw=0.7)
        if np.isfinite(t_settle):
            ax.axvline(t_settle, color="orange", ls="--", lw=1.0)
        ax.set_ylabel("T_i [s]")
        ax.set_ylim(0, 10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Row 2: amp_i(t)
        ax = axes[2, col]
        ax.plot(metrics["uc_time"], metrics["amp_i"], "o-", color=clr,
                lw=0.8, ms=4)
        ax.axvline(t_exp, color="k", ls=":", lw=0.7)
        if np.isfinite(t_settle):
            ax.axvline(t_settle, color="orange", ls="--", lw=1.0)
        ax.set_ylabel("per-cycle (max−min)/2 [mm]")
        ax.set_xlabel("time [s]")
        ax.grid(alpha=0.3)

    fig.suptitle(f"Pre-7T-start chirp detail at IN ({PROBE})  —  "
                 f"f={f_hz} Hz, A={amp} V",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> None:
    print("Loading canon meta...")
    meta = load_canon_meta()
    print(f"  {len(meta)} ok runs after filters")

    chosen = pick_runs(meta)
    print(f"  picked {len(chosen)} canonical runs (one per cell × wind)")

    big = load_processed(CANON)

    print("\nPlotting η(t) grid...")
    plot_eta_grid(meta, big, chosen, OUT_DIR / "pre_7T_wave_chirp_eta.png")
    print("  → analysis_scratch/pre_7T_wave_chirp_eta.png")

    print("\nPlotting period chirp grid + computing settle times...")
    settle_rows = []
    plot_period_grid(meta, big, chosen,
                     OUT_DIR / "pre_7T_wave_chirp_period.png",
                     settle_rows)
    print("  → analysis_scratch/pre_7T_wave_chirp_period.png")

    settle_df = pd.DataFrame(settle_rows)
    csv = OUT_DIR / "pre_7T_wave_chirp_settle.csv"
    settle_df.to_csv(csv, index=False)
    print(f"  → {csv.relative_to(BASE)} ({len(settle_df)} rows)")

    print("\n=== Time-to-settle [s] (3 cycles within ±10% of T_paddle) ===")
    piv_settle = settle_df.pivot_table(
        index=["f_hz", "amp_V"], columns="wind", values="t_settle"
    )
    piv_exp = settle_df.pivot_table(
        index=["f_hz", "amp_V"], columns="wind", values="t_exp_7T",
        aggfunc="first"
    )
    piv_snap = settle_df.pivot_table(
        index=["f_hz", "amp_V"], columns="wind", values="t_snap"
    )
    print("\n  t_settle (when waves first reach paddle period for 3 cycles):")
    print(piv_settle.round(2).to_string())
    print("\n  t_exp_7T (theoretical 7T-start, identical fw/nw):")
    print(piv_exp.round(2).to_string())
    print("\n  t_snap (where pipeline actually anchored):")
    print(piv_snap.round(2).to_string())
    print("\n  Δ(snap − settle) per wind — POSITIVE means snap is anchored "
          "AFTER waves settled to paddle period (good); NEGATIVE means snap "
          "is anchored DURING the chirp (bad — measuring non-paddle period):")
    delta = settle_df.assign(
        snap_minus_settle=lambda d: d["t_snap"] - d["t_settle"]
    ).pivot_table(
        index=["f_hz", "amp_V"], columns="wind", values="snap_minus_settle"
    )
    print(delta.round(2).to_string())

    print("\nPlotting detail for 1.3 Hz, 0.2 V (cleanest cell)...")
    plot_detail(meta, big, chosen,
                OUT_DIR / "pre_7T_wave_chirp_detail_1.3Hz_0.2V.png",
                f_hz=1.3, amp=0.2)
    print("  → analysis_scratch/pre_7T_wave_chirp_detail_1.3Hz_0.2V.png")

    # Findings markdown
    md = OUT_DIR / "pre_7T_wave_chirp.md"
    with open(md, "w") as f:
        f.write(_findings_md(settle_df))
    print(f"\nFindings → {md.relative_to(BASE)}")


def _findings_md(settle_df: pd.DataFrame) -> str:
    delta = settle_df.assign(
        snap_minus_settle=lambda d: d["t_snap"] - d["t_settle"]
    ).pivot_table(
        index=["f_hz", "amp_V"], columns="wind", values="snap_minus_settle"
    )
    lines = [
        "# Pre-7T-start wave-chirp characterization (2026-05-05)",
        "",
        "## Question",
        "",
        "Before claiming the snap-shift Δt under fullwind reflects 'earlier",
        "paddle-wave arrival', we need to confirm that the snap is anchored",
        "in the steady-state paddle-frequency wave train, NOT in the",
        "wavemaker's pre-paddle chirp (long-period → paddle-period sweep).",
        "",
        "## Method",
        "",
        "Per (freq, amp, wind) cell on canon, take one canonical run.",
        "At IN (9373/170): detect zero upcrossings of η, compute",
        "instantaneous period T_i between consecutive upcrossings. Define",
        "'settle time' as the first upcrossing where |T_i − T_pad|/T_pad < 0.10",
        "for 3 consecutive cycles.",
        "",
        "Compare:",
        "- t_exp_7T  — theoretical 7T-start (identical fw/nw)",
        "- t_snap    — where the pipeline actually anchored (per-run)",
        "- t_settle  — first cycle where T_i is within ±10% of T_pad",
        "",
        "## Verdict per cell — Δ(snap − settle)",
        "",
        "POSITIVE = snap anchored AFTER chirp settled to paddle period (good)",
        "NEGATIVE = snap anchored DURING chirp (bad — measuring non-paddle T)",
        "",
        delta.round(2).to_string(),
        "",
        "## Implication for the highway-effect Δt",
        "",
        "If Δ(snap − settle) is consistently POSITIVE for both fw and nw,",
        "the snap is comfortably in the steady-state wave train and the",
        "earlier-snap-under-fullwind observation is a real timing shift of",
        "the steady-state wave field — the highway effect stands.",
        "",
        "If Δ(snap − settle) is NEGATIVE under fullwind but POSITIVE under",
        "nowind (or vice versa), the apparent Δt may be partly due to the",
        "snap landing on different parts of the chirp in the two conditions.",
        "Further investigation needed.",
        "",
        "## Files",
        "- pre_7T_wave_chirp_eta.png — η(t) grid 4×3 cells",
        "- pre_7T_wave_chirp_period.png — T_i(t) chirp grid 4×3 cells",
        "- pre_7T_wave_chirp_detail_1.3Hz_0.2V.png — detailed 3×2 panel",
        "- pre_7T_wave_chirp_settle.csv — per-cell settle table",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
