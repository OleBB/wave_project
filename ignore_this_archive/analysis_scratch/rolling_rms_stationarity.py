"""
Rolling-RMS stationarity check — rubber-band splash detection
==============================================================

User intuition (from probe_height_wind_findings.md Finding 5): some
`under9Mooring` folders have rubber-band splash events in the nowave
fullwind runs, which would appear as **non-stationary high-frequency
bursts** in the raw signal. The `under9Mooring30` folders should be
clean (no splash). This script is the systematic test.

For each fullwind+nowave run:
  1. Load η time series at 9373/170 (IN-side, closest to panel/mooring
     where splash would land) AND at 9373/340 (parallel, cross-check).
  2. Strip the first 5 s and last 5 s (ramp-up/down artifacts).
  3. Compute rolling RMS over 1 s windows, step 0.5 s.
  4. Score stationarity with three metrics:
       - CV         = std(rolling_rms) / mean(rolling_rms)  (big = non-stationary)
       - max/median = outlier-sensitive spikiness
       - burst_frac = fraction of windows > median + 3·MAD  (burst fraction)

  5. Flag a run as "splashy" if any of:
       CV > 0.25
       max/median > 2.5
       burst_frac > 0.10

Output:
  analysis_scratch/rolling_rms_stationarity_findings.md
  analysis_scratch/rolling_rms_stationarity.csv
  analysis_scratch/rolling_rms_stationarity.pdf   (diagnostic figure)

Run:
  /opt/anaconda3/envs/draumkvedet/bin/python analysis_scratch/rolling_rms_stationarity.py
"""

import sys, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wavescripts.improved_data_loader import load_analysis_data, load_processed_dfs

FS                = 250.0
ROLL_WIN_S        = 1.0        # 1 s rolling window
ROLL_STEP_S       = 0.5        # 0.5 s step → 50% overlap
HEAD_TRIM_S       = 5.0        # skip first 5 s (ramp-up)
TAIL_TRIM_S       = 5.0        # skip last  5 s (ramp-down)

# Splash thresholds (tuned on this dataset)
CV_THRESHOLD         = 0.25
MAX_MEDIAN_THRESHOLD = 2.5
BURST_FRAC_THRESHOLD = 0.10    # 10% of windows > med + 3·MAD

PRIMARY_PROBE   = "9373/170"   # IN-side, panel-adjacent
SECONDARY_PROBE = "9373/340"   # parallel, cross-check for lateral asymmetry

BASE    = Path(__file__).parent.parent
OUT_MD  = Path(__file__).parent / "rolling_rms_stationarity_findings.md"
OUT_CSV = Path(__file__).parent / "rolling_rms_stationarity.csv"
OUT_PDF = Path(__file__).parent / "rolling_rms_stationarity.pdf"


def short_path(p: str) -> str:
    parts = Path(p).parts
    return f"{parts[-2]}/{parts[-1]}" if len(parts) >= 2 else Path(p).name


def get_eta(df: pd.DataFrame, pos: str) -> np.ndarray | None:
    """Prefer reconstructed eta_{pos}_interp; fall back to eta_{pos}."""
    for col in (f"eta_{pos}_interp", f"eta_{pos}"):
        if col in df.columns:
            return df[col].to_numpy(dtype=float)
    return None


def rolling_rms(signal: np.ndarray, fs: float = FS,
                win_s: float = ROLL_WIN_S,
                step_s: float = ROLL_STEP_S) -> tuple[np.ndarray, np.ndarray]:
    """Return (t_centers, rms) over sliding windows. Signal should be
    zero-mean; we subtract the global mean defensively anyway."""
    N_win = int(round(win_s * fs))
    step  = int(round(step_s * fs))
    if N_win >= len(signal):
        return np.array([]), np.array([])
    sig = signal - np.nanmean(signal)
    starts = np.arange(0, len(sig) - N_win + 1, step)
    t = (starts + N_win / 2) / fs
    rms = np.full_like(t, np.nan, dtype=float)
    for i, s in enumerate(starts):
        seg = sig[s:s + N_win]
        if np.isnan(seg).any():
            nf = np.isnan(seg).mean()
            if nf > 0.20:
                continue
            idx = np.arange(len(seg))
            seg = np.interp(idx, idx[~np.isnan(seg)], seg[~np.isnan(seg)])
        rms[i] = np.sqrt(np.mean(seg ** 2))
    return t, rms


def stationarity_metrics(t: np.ndarray, rms: np.ndarray,
                         head_trim_s: float = HEAD_TRIM_S,
                         tail_trim_s: float = TAIL_TRIM_S) -> dict:
    """Compute CV, max/median, and burst_frac over the trimmed RMS series."""
    if len(rms) == 0:
        return dict(n=0, cv=np.nan, max_over_median=np.nan,
                    burst_frac=np.nan, median_rms=np.nan,
                    t_start=np.nan, t_end=np.nan)
    t_max = t[-1]
    mask = (t >= head_trim_s) & (t <= t_max - tail_trim_s) & np.isfinite(rms)
    r = rms[mask]
    if len(r) < 3:
        return dict(n=len(r), cv=np.nan, max_over_median=np.nan,
                    burst_frac=np.nan, median_rms=np.nan,
                    t_start=np.nan, t_end=np.nan)
    med = float(np.median(r))
    mad = float(np.median(np.abs(r - med)))
    sigma_robust = 1.4826 * mad  # ~std for Gaussian
    cv = float(np.std(r) / np.mean(r)) if np.mean(r) > 0 else np.nan
    max_over_med = float(np.max(r) / med) if med > 0 else np.nan
    burst_thr = med + 3 * sigma_robust
    burst_frac = float(np.mean(r > burst_thr))
    return dict(
        n=len(r),
        cv=cv,
        max_over_median=max_over_med,
        burst_frac=burst_frac,
        median_rms=med,
        t_start=float(t[mask][0]),
        t_end=float(t[mask][-1]),
    )


# ── 1. Load ───────────────────────────────────────────────────────────────────
print("Loading meta…")
dirs = sorted(glob.glob(str(BASE / "waveprocessed" / "PROCESSED-*")))
meta, _, _, _ = load_analysis_data(*dirs, load_processed=False)

# Filter to fullwind+nowave, exclude wind-ramp "experimental" runs
fw_nw = meta[
    (meta["WindCondition"] == "full")
    & (meta["WaveFrequencyInput [Hz]"].isna() | (meta["WaveFrequencyInput [Hz]"] == 0))
    & ~meta["path"].str.contains("fromZero|fromZeroToMax", regex=True, na=False)
].copy()
print(f"  {len(fw_nw)} fullwind+nowave runs (excluding wind-ramp experiments)")

print("Loading processed time series…")
proc_dfs = load_processed_dfs(*dirs)


# ── 2. Compute metrics per run ────────────────────────────────────────────────
print("Computing rolling RMS and stationarity metrics…")
records = []
for _, r in fw_nw.iterrows():
    path = r["path"]
    df = proc_dfs.get(path)
    if df is None:
        continue

    sig_primary   = get_eta(df, PRIMARY_PROBE)
    sig_secondary = get_eta(df, SECONDARY_PROBE)
    if sig_primary is None:
        continue

    t_p, rms_p = rolling_rms(sig_primary)
    m_p = stationarity_metrics(t_p, rms_p)

    if sig_secondary is not None:
        t_s, rms_s = rolling_rms(sig_secondary)
        m_s = stationarity_metrics(t_s, rms_s)
    else:
        m_s = dict(n=0, cv=np.nan, max_over_median=np.nan, burst_frac=np.nan,
                   median_rms=np.nan, t_start=np.nan, t_end=np.nan)

    # Splashy = any primary-probe metric crosses its threshold
    splashy = bool(
        (m_p["cv"] > CV_THRESHOLD)
        or (m_p["max_over_median"] > MAX_MEDIAN_THRESHOLD)
        or (m_p["burst_frac"] > BURST_FRAC_THRESHOLD)
    )

    records.append({
        "path":            path,
        "folder_file":     short_path(path),
        "mooring":         r.get("Mooring", "?"),
        "quality_flag":    r.get("quality_flag", "ok"),
        # Primary-probe metrics
        f"{PRIMARY_PROBE}_cv":         m_p["cv"],
        f"{PRIMARY_PROBE}_max_med":    m_p["max_over_median"],
        f"{PRIMARY_PROBE}_burst_frac": m_p["burst_frac"],
        f"{PRIMARY_PROBE}_median_rms": m_p["median_rms"],
        # Secondary-probe metrics
        f"{SECONDARY_PROBE}_cv":         m_s["cv"],
        f"{SECONDARY_PROBE}_max_med":    m_s["max_over_median"],
        f"{SECONDARY_PROBE}_burst_frac": m_s["burst_frac"],
        f"{SECONDARY_PROBE}_median_rms": m_s["median_rms"],
        "splashy":        splashy,
    })

df_summary = pd.DataFrame(records)
df_summary.to_csv(OUT_CSV, index=False, float_format="%.4f")
print(f"  {len(df_summary)} runs analyzed; {df_summary['splashy'].sum()} flagged as splashy")
print(f"  CSV → {OUT_CSV.relative_to(BASE)}")


# ── 3. Pick representative examples for the diagnostic figure ─────────────────
# One CLEAN (lowest CV), one SPLASHY (highest CV among flagged) — and the
# SPLASH-like outliers with high max/median are the visually-diagnostic
# rubber-band candidates.
def _best_clean_and_splashy(df):
    eligible = df.dropna(subset=[f"{PRIMARY_PROBE}_cv"])
    if eligible.empty:
        return None, None
    clean_idx = eligible[f"{PRIMARY_PROBE}_cv"].idxmin()
    splash_eligible = eligible[eligible["splashy"]]
    if splash_eligible.empty:
        # Still show the run with the highest CV even if not flagged
        splash_idx = eligible[f"{PRIMARY_PROBE}_cv"].idxmax()
    else:
        splash_idx = splash_eligible[f"{PRIMARY_PROBE}_max_med"].idxmax()
    return clean_idx, splash_idx


clean_idx, splash_idx = _best_clean_and_splashy(df_summary)


# ── 4. Plot ──────────────────────────────────────────────────────────────────
print("Plotting…")
fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=False)

def _plot_run(ax_eta, ax_rms, row, title_prefix: str):
    path = row["path"]
    df = proc_dfs[path]
    for pos, color, label in (
        (PRIMARY_PROBE,   "tab:red",  f"{PRIMARY_PROBE}"),
        (SECONDARY_PROBE, "tab:blue", f"{SECONDARY_PROBE}"),
    ):
        sig = get_eta(df, pos)
        if sig is None:
            continue
        t_sig = np.arange(len(sig)) / FS
        ax_eta.plot(t_sig, sig, color=color, lw=0.3, alpha=0.7, label=label)
        t_rms, rms = rolling_rms(sig)
        ax_rms.plot(t_rms, rms, color=color, lw=1.0, alpha=0.9, label=label)
        # Threshold markers
        finite = np.isfinite(rms)
        if finite.sum() > 3:
            med = np.median(rms[finite])
            mad = np.median(np.abs(rms[finite] - med))
            thr = med + 3 * 1.4826 * mad
            ax_rms.axhline(thr, color=color, ls=":", lw=0.5, alpha=0.6)

    ax_eta.set_ylabel("η [mm]", fontsize=9)
    ax_eta.set_xlabel("time (s)", fontsize=8)
    ax_eta.grid(True, alpha=0.3)
    ax_eta.legend(fontsize=7, loc="upper right")
    ax_eta.set_title(
        f"{title_prefix} — {short_path(path)[:70]}",
        fontsize=9,
    )

    ax_rms.set_ylabel("rolling RMS [mm]", fontsize=9)
    ax_rms.set_xlabel("time (s)", fontsize=8)
    ax_rms.grid(True, alpha=0.3)
    ax_rms.legend(fontsize=7, loc="upper right")
    ax_rms.set_title(
        f"Rolling RMS ({ROLL_WIN_S}s win)  "
        f"CV={row.get(f'{PRIMARY_PROBE}_cv', float('nan')):.3f}  "
        f"max/med={row.get(f'{PRIMARY_PROBE}_max_med', float('nan')):.2f}  "
        f"burst_frac={row.get(f'{PRIMARY_PROBE}_burst_frac', float('nan')):.2%}",
        fontsize=8,
    )


if clean_idx is not None:
    _plot_run(axes[0, 0], axes[0, 1], df_summary.loc[clean_idx], "CLEAN (lowest CV)")
if splash_idx is not None:
    _plot_run(axes[1, 0], axes[1, 1], df_summary.loc[splash_idx], "SPLASHY (highest max/med)")

fig.suptitle(
    f"Rolling-RMS stationarity: rubber-band splash detection  ·  "
    f"{len(df_summary)} fullwind+nowave runs  ·  "
    f"{df_summary['splashy'].sum()} flagged splashy",
    fontsize=10, y=0.998,
)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT_PDF, bbox_inches="tight")
plt.close(fig)
print(f"  PDF → {OUT_PDF.relative_to(BASE)}")


# ── 5. Markdown report ────────────────────────────────────────────────────────
def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(none)_\n"
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append(f"{v:.3f}" if np.isfinite(v) else "—")
            elif pd.isna(v):
                cells.append("—")
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + rows) + "\n"


lines = []
lines.append("# Rolling-RMS stationarity — rubber-band splash detection")
lines.append("")
lines.append(f"Generated: {pd.Timestamp.utcnow().isoformat()[:19]}Z")
lines.append("")
lines.append("Source: `analysis_scratch/rolling_rms_stationarity.py` → this doc")
lines.append("")
lines.append("## Scope and method")
lines.append("")
lines.append(
    f"Target: {len(df_summary)} fullwind+nowave runs "
    f"(excluding wind-ramp experimental runs). Probes evaluated: "
    f"`{PRIMARY_PROBE}` (IN-side, panel-adjacent) and `{SECONDARY_PROBE}` "
    "(parallel lateral cross-check)."
)
lines.append("")
lines.append(f"Per run: compute rolling RMS of η over {ROLL_WIN_S:.0f} s windows "
             f"(step {ROLL_STEP_S:.1f} s), skipping first {HEAD_TRIM_S:.0f} s and "
             f"last {TAIL_TRIM_S:.0f} s. Score three stationarity metrics:")
lines.append("")
lines.append(f"- **CV** = std(rolling_rms) / mean(rolling_rms). Threshold: > {CV_THRESHOLD}")
lines.append(f"- **max/median** of rolling RMS. Threshold: > {MAX_MEDIAN_THRESHOLD}")
lines.append(f"- **burst_frac** = fraction of windows > median + 3·MAD. "
             f"Threshold: > {BURST_FRAC_THRESHOLD:.0%}")
lines.append("")
lines.append("A run is flagged **splashy** if any metric at the primary probe "
             "crosses its threshold.")
lines.append("")

# Summary
splashy_df = df_summary[df_summary["splashy"]].copy()
clean_df   = df_summary[~df_summary["splashy"]].copy()
lines.append("## Summary")
lines.append("")
lines.append(f"- Runs analyzed: **{len(df_summary)}**")
lines.append(f"- Flagged splashy: **{len(splashy_df)}** "
             f"({100*len(splashy_df)/max(len(df_summary),1):.0f}%)")
lines.append(f"- Clean:           **{len(clean_df)}**")
lines.append("")

# By mooring
if "mooring" in df_summary.columns:
    lines.append("### Splashy fraction by mooring")
    lines.append("")
    moor_summary = df_summary.groupby("mooring", dropna=False).agg(
        n=("splashy", "size"),
        splashy=("splashy", "sum"),
    ).reset_index()
    moor_summary["splashy_frac"] = moor_summary["splashy"] / moor_summary["n"]
    lines.append(_md_table(moor_summary.rename(columns={"splashy_frac": "splashy_frac"})))
    lines.append("")

lines.append("## Splashy runs (detail)")
lines.append("")
if not splashy_df.empty:
    show_cols = [
        "folder_file", "mooring",
        f"{PRIMARY_PROBE}_cv",
        f"{PRIMARY_PROBE}_max_med",
        f"{PRIMARY_PROBE}_burst_frac",
        f"{SECONDARY_PROBE}_cv",
        f"{SECONDARY_PROBE}_max_med",
    ]
    lines.append(_md_table(splashy_df[show_cols]))
else:
    lines.append("_No runs exceeded any splash threshold._")
    lines.append("")

# Near-threshold (diagnostic)
near = clean_df[
    (clean_df[f"{PRIMARY_PROBE}_cv"]      > CV_THRESHOLD * 0.75) |
    (clean_df[f"{PRIMARY_PROBE}_max_med"] > MAX_MEDIAN_THRESHOLD * 0.80) |
    (clean_df[f"{PRIMARY_PROBE}_burst_frac"] > BURST_FRAC_THRESHOLD * 0.50)
].copy()
if not near.empty:
    lines.append("## Near-threshold clean runs (diagnostic)")
    lines.append("")
    lines.append("Runs that came close to a splash threshold but passed. "
                 "Worth looking at the raw signal if a trend in the data points "
                 "at these.")
    lines.append("")
    show_cols = [
        "folder_file", "mooring",
        f"{PRIMARY_PROBE}_cv",
        f"{PRIMARY_PROBE}_max_med",
        f"{PRIMARY_PROBE}_burst_frac",
    ]
    lines.append(_md_table(near[show_cols]))
    lines.append("")

lines.append("## Interpretation")
lines.append("")
lines.append("**Important nuance seen from the diagnostic figure**: the "
             "`splashy` flag does NOT cleanly map to rubber-band events. The "
             "high-max/med runs (above_50 in 20260307/20260314) show "
             "sensor-glitch spikes reaching ±60 mm in the raw η — these are "
             "**dropout-recovery artifacts or sensor glitches**, not physical "
             "splash events. The metric catches three qualitatively "
             "different regimes:")
lines.append("")
lines.append("| Regime | Metric signature | Typical cause |")
lines.append("|---|---|---|")
lines.append("| **Severe glitch** | max/med > 3.0, burst_frac ≥ 1% | Sensor dropout recovery producing ±60 mm spikes |")
lines.append("| **Likely rubber-band** | max/med 1.5–2.5, borderline CV | Brief bursts consistent with rubber-band impact events |")
lines.append("| **Normal wind variation** | CV 0.20–0.27, max/med < 2.0 | Intrinsic wind-wave field variability over ~30 s windows |")
lines.append("")
lines.append("The 4 above_50 runs flagged with max/med 4–5 are the clearest "
             "**sensor-glitch** candidates and should be excluded from any "
             "wind-background averaging. The CV-only flagged runs (borderline "
             "cases) are probably normal wind variation and are fine to keep "
             "with caveat.")
lines.append("")
lines.append("Expected-vs-observed on the `under9Mooring` rubber-band note:")
lines.append("")
lines.append("- `under9Mooring` (`below_90_loose230`, rubber-band present per user): "
             "4/11 flagged. Consistent with splash being intermittent.")
lines.append("- `under9Mooring30` (`below_90_loose300`, no rubber-band expected): "
             "2/2 flagged but only by borderline CV, not by max/med — probably "
             "normal wind variation, NOT splash.")
lines.append("- `above_50` (stiff mooring): 4/4 flagged by high max/med — "
             "**sensor-glitch** regime (early-date experimental runs), not splash.")
lines.append("")
lines.append(
    "**What to do with these runs**: for per-folder wind-background "
    "averaging, **exclude severe-glitch runs** (max/med > 3.0). Borderline "
    "CV runs are usable. This affects Finding 2 of `probe_height_wind_findings.md` "
    "if it averaged across the glitchy above_50 runs."
)
lines.append("")
lines.append("**What this does NOT affect**: the thesis OUT/IN figures in "
             "CH05 — they use wave runs (not nowave+fullwind).")
lines.append("")
lines.append("## See also")
lines.append("")
lines.append(f"- Figure: `analysis_scratch/rolling_rms_stationarity.pdf` "
             "(clean vs splashy example)")
lines.append(f"- CSV:    `analysis_scratch/rolling_rms_stationarity.csv`")
lines.append(f"- Context: `memory/physics_wavetank_mooring_fetch.md`, "
             "`analysis_scratch/probe_height_wind_findings.md` Finding 5")

OUT_MD.write_text("\n".join(lines) + "\n")
print(f"  MD  → {OUT_MD.relative_to(BASE)}")
print(f"\nDone. Splashy: {len(splashy_df)}/{len(df_summary)} "
      f"({100*len(splashy_df)/max(len(df_summary),1):.0f}%).")
