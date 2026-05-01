"""
Is the 2 s pre-paddle window representative of the long wind-only state?

CANON datasets only — the two folders ending in "-lowrange" (final
march2026_better_rearranging probe config, low-range ULS, height 100 mm).

Long-run reference: 5 long fullwind+nowave runs (~6 min each):
    20260326-...-lowrange  : 3 runs (mstop30-run1, mstop30-run2, mstop330-run1)
    20260327-...-lowrange  : 2 runs (mstop30-run2, mstop30-run4)

2 s pre-paddle ensemble: first 2 s of every fullwind wave run in both
datasets. Paddle motion has not yet arrived at any probe at t < 3.69 s
because √(gh) = 2.39 m/s and the closest probe is 8.804 m away.

For each gauge:
  η̄  = mean elevation [mm]
  σ_η = standard deviation about the mean [mm]
  Hs  = 4 σ_η = significant wave height [mm]

Outputs:
    analysis_scratch/wind_2s_vs_360s_stats.csv          — ensemble summary (compact)
    analysis_scratch/wind_2s_vs_360s_per_long_run.csv   — per-long-run breakdown
    analysis_scratch/wind_2s_vs_360s_spectrum.png       — IN probe spectrum compare
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch, periodogram

from wavescripts.improved_data_loader import load_analysis_data, load_processed_dfs

FS = 250.0
BASE = Path("/Users/ole/Kodevik/wave_project")

# Two CANON folders — both ending in "-lowrange" (cond4: h100 + low-range ULS)
TARGET_DIRS = [
    BASE / "waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange",
    BASE / "waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange",
]

# 5 long fullwind+nowave reference runs across the two canon folders.
LONG_RUN_CSVS = [
    str(BASE / "wavedata/20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange/fullpanel-fullwind-nowave-depth580-mstop30-run1.csv"),
    str(BASE / "wavedata/20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange/fullpanel-fullwind-nowave-depth580-mstop30-run2.csv"),
    str(BASE / "wavedata/20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange/fullpanel-fullwind-nowave-depth580-mstop330-run1.csv"),
    str(BASE / "wavedata/20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange/fullpanel-fullwind-nowave-depth580-mstop30-run2.csv"),
    str(BASE / "wavedata/20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange/fullpanel-fullwind-nowave-depth580-mstop30-run4.csv"),
]

PROBES = ["8804/250", "9373/170", "9373/340", "12400/250"]
LABELS = {
    "8804/250":  "8804/250 (upstream)",
    "9373/170":  "9373/170 (IN, wall)",
    "9373/340":  "9373/340 (IN, far)",
    "12400/250": "12400/250 (OUT)",
}

SNIPPET_S = 3.0          # 3 s safe at all probes: closest is 8804 mm, √(gh)=2.39 m/s → 3.68 s safety
SNIPPET_N = int(SNIPPET_S * FS)
PSD_PROBE = "9373/170"   # which probe to draw the spectrum for (IN, wall side)
TAG       = f"{SNIPPET_S:g}s"   # suffix for output filenames so prior runs are preserved

# ── Load: TWO canon folders only ─────────────────────────────────────────
print(f"Loading meta + processed_dfs for {len(TARGET_DIRS)} canon datasets …")
meta, _, _, _ = load_analysis_data(*[str(d) for d in TARGET_DIRS], load_processed=False)
proc = {}
for d in TARGET_DIRS:
    proc.update(load_processed_dfs(str(d)))
print(f"  → {len(meta)} runs in meta, {len(proc)} runs with time-series")

# ── Pick the wave runs whose first 2 s is pure-wind ──────────────────────
wave_sel = (
    (meta["WindCondition"] == "full") &
    (meta["WaveFrequencyInput [Hz]"].notna()) &
    (meta["quality_flag"] == "ok")
)
wave_paths = [p for p in meta.loc[wave_sel, "path"] if p in proc]
missing_long = [p for p in LONG_RUN_CSVS if p not in proc]
if missing_long:
    print(f"\nWARNING: {len(missing_long)} long run(s) not in cache:")
    for p in missing_long:
        print(f"  - {p}")
long_present = [p for p in LONG_RUN_CSVS if p in proc]
print(f"  long runs (present): {len(long_present)} / {len(LONG_RUN_CSVS)}")
print(f"  wave runs (fullwind, ok): {len(wave_paths)}  "
      f"(each contributes one 2 s pre-paddle snippet per probe)")
if not long_present:
    raise SystemExit("No long reference runs present — cannot continue.")


def _eta(df, probe):
    col = f"eta_{probe}_interp" if f"eta_{probe}_interp" in df.columns else f"eta_{probe}"
    if col not in df.columns:
        return None
    return df[col].to_numpy(dtype=float)


def _stats(arr):
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return np.nan, np.nan, np.nan
    mean  = float(np.mean(arr))
    sigma = float(np.std(arr, ddof=1))   # σ about the mean
    return mean, sigma, 4.0 * sigma      # mean, σ, Hs=4σ


# ── Build the tables ─────────────────────────────────────────────────────
ensemble_rows = []   # compact: 2 rows per probe (long-ensemble + 2s-ensemble)
per_run_rows  = []   # diagnostic: one row per long run per probe

for probe in PROBES:
    label = LABELS[probe]

    # Per-long-run stats (whole-record)
    long_means, long_sigmas, long_hss = [], [], []
    for lp in long_present:
        eta = _eta(proc[lp], probe)
        if eta is None:
            continue
        m, s, h = _stats(eta)
        long_means.append(m); long_sigmas.append(s); long_hss.append(h)
        per_run_rows.append({
            "probe": label, "long_run": Path(lp).parent.name + "/" + Path(lp).name,
            "duration_s": round(len(proc[lp]) / FS, 1),
            "mean_mm":  round(m, 3),
            "sigma_mm": round(s, 3),
            "Hs_mm":    round(h, 3),
        })
    n_long = len(long_means)
    ensemble_rows.append({
        "probe": label,
        "segment": f"long nowave runs (ensemble across {n_long})",
        "n": n_long,
        "mean_mm":  round(float(np.mean(long_means)),  3) if n_long else np.nan,
        "sigma_mm": round(float(np.mean(long_sigmas)), 3) if n_long else np.nan,
        "Hs_mm":    round(float(np.mean(long_hss)),    3) if n_long else np.nan,
        "mean_std":  round(float(np.std(long_means,  ddof=1)), 3) if n_long > 1 else np.nan,
        "sigma_std": round(float(np.std(long_sigmas, ddof=1)), 3) if n_long > 1 else np.nan,
        "Hs_std":    round(float(np.std(long_hss,    ddof=1)), 3) if n_long > 1 else np.nan,
    })

    # 2 s pre-paddle ensemble across all fullwind wave runs
    means, sigmas, hss = [], [], []
    for wp in wave_paths:
        eta = _eta(proc[wp], probe)
        if eta is None or len(eta) < SNIPPET_N:
            continue
        seg = eta[:SNIPPET_N]
        if not np.all(np.isfinite(seg)):
            continue
        m, s, h = _stats(seg)
        means.append(m); sigmas.append(s); hss.append(h)
    n = len(means)
    ensemble_rows.append({
        "probe": label,
        "segment": f"{SNIPPET_S:g} s pre-paddle (ensemble across {n} wave runs)",
        "n": n,
        "mean_mm":  round(float(np.mean(means)),  3) if n else np.nan,
        "sigma_mm": round(float(np.mean(sigmas)), 3) if n else np.nan,
        "Hs_mm":    round(float(np.mean(hss)),    3) if n else np.nan,
        "mean_std":  round(float(np.std(means,  ddof=1)), 3) if n > 1 else np.nan,
        "sigma_std": round(float(np.std(sigmas, ddof=1)), 3) if n > 1 else np.nan,
        "Hs_std":    round(float(np.std(hss,    ddof=1)), 3) if n > 1 else np.nan,
    })

ens_table = pd.DataFrame(ensemble_rows)
per_run_table = pd.DataFrame(per_run_rows)

print("\n=== Per-long-run η statistics (5 long fullwind+nowave runs) ===")
with pd.option_context("display.width", 220, "display.max_columns", 50,
                       "display.max_colwidth", 90):
    print(per_run_table.to_string(index=False))

print(f"\n=== Ensemble: long-runs vs {SNIPPET_S:g} s pre-paddle ===")
with pd.option_context("display.width", 220, "display.max_columns", 50,
                       "display.max_colwidth", 70):
    print(ens_table.to_string(index=False))

csv_out = Path(__file__).parent / f"wind_2s_vs_360s_stats_{TAG}.csv"
ens_table.to_csv(csv_out, index=False)
print(f"\n  → {csv_out.relative_to(BASE)}")

per_run_csv = Path(__file__).parent / f"wind_2s_vs_360s_per_long_run_{TAG}.csv"
per_run_table.to_csv(per_run_csv, index=False)
print(f"  → {per_run_csv.relative_to(BASE)}")

# ── Spectrum plot — IN probe only ────────────────────────────────────────
nperseg = int(8 * FS)   # 8 s segments → Δf = 0.125 Hz (per long-run Welch)

# Long-run PSDs — one per long run (5 PSDs at most)
long_psds = []
freqs_long = None
for lp in long_present:
    eta = _eta(proc[lp], PSD_PROBE)
    if eta is None:
        continue
    f_l, P_l = welch(eta - np.nanmean(eta), fs=FS,
                     window="hann", nperseg=nperseg,
                     scaling="density", detrend="linear")
    if freqs_long is None:
        freqs_long = f_l
    long_psds.append(P_l)
long_psds = np.vstack(long_psds)
long_p50 = np.nanpercentile(long_psds, 50, axis=0)
long_p16 = np.nanpercentile(long_psds, 16, axis=0)
long_p84 = np.nanpercentile(long_psds, 84, axis=0)

# 2 s pre-paddle snippet PSDs — one per wave run
snip_psds  = []
freqs_snip = None
for wp in wave_paths:
    eta = _eta(proc[wp], PSD_PROBE)
    if eta is None or len(eta) < SNIPPET_N:
        continue
    seg = eta[:SNIPPET_N]
    if not np.all(np.isfinite(seg)):
        continue
    f_s, P_s = periodogram(seg - np.mean(seg), fs=FS,
                           window="hann", scaling="density", detrend="linear")
    if freqs_snip is None:
        freqs_snip = f_s
    snip_psds.append(P_s)
snip_psds = np.vstack(snip_psds)
snip_p16 = np.nanpercentile(snip_psds, 16, axis=0)
snip_p50 = np.nanpercentile(snip_psds, 50, axis=0)
snip_p84 = np.nanpercentile(snip_psds, 84, axis=0)

fig, ax = plt.subplots(figsize=(11, 6))

ax.fill_between(freqs_long, long_p16, long_p84, color="#1E9C68", alpha=0.20,
                label=f"long-run 16–84 % (n={long_psds.shape[0]})")
ax.semilogy(freqs_long, long_p50, color="#1E9C68", lw=2.0,
            label=f"long-run median (n={long_psds.shape[0]}, Welch nperseg=8 s)")

ax.fill_between(freqs_snip, snip_p16, snip_p84, color="#FEA11B", alpha=0.22,
                label=f"{SNIPPET_S:g} s pre-paddle 16–84 % (n={snip_psds.shape[0]})")
ax.semilogy(freqs_snip, snip_p50, color="#FEA11B", lw=1.8, ls="--",
            label=f"{SNIPPET_S:g} s pre-paddle median (n={snip_psds.shape[0]})")

ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel(r"$S_\eta(f)$  [mm$^2$/Hz]")
ax.set_xlim(0, 16)
ax.set_title(
    f"Wind PSD — {len(long_present)} long nowave runs vs ensemble of {SNIPPET_S:g} s pre-paddle snippets\n"
    f"probe {LABELS[PSD_PROBE]} | canon datasets 20260326+27 -lowrange | fullwind"
)
ax.grid(True, which="major", alpha=0.40)
ax.grid(True, which="minor", alpha=0.18)
ax.legend(loc="upper right", fontsize=10)

png_out = Path(__file__).parent / f"wind_2s_vs_360s_spectrum_{TAG}.png"
fig.savefig(png_out, dpi=160, bbox_inches="tight")
print(f"  → {png_out.relative_to(BASE)}")

# ── Thesis-side render: PDF + PNG into output/FIGURES + TEXFIGU stub ─────
# Wired into main_save_figures.py via _run_delegated_if_missing.
THESIS_NAME = "ch04_wind_pre_paddle_psd"
thesis_dir  = BASE / "output" / "FIGURES"
thesis_dir.mkdir(parents=True, exist_ok=True)
thesis_pdf  = thesis_dir / f"{THESIS_NAME}.pdf"
thesis_png  = thesis_dir / f"{THESIS_NAME}.png"
fig.savefig(thesis_pdf, bbox_inches="tight")
fig.savefig(thesis_png, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"  → {thesis_pdf.relative_to(BASE)}  (+ .png)")

# TEXFIGU stub via plot_utils. Caption text is sourced from FIGURE_CAPTIONS in
# main_save_figures.py via output/.figure_captions.json — empty entry there
# lands a TODO placeholder in the body, edited later by hand.
import wavescripts.plot_utils as pu
pu.ACTIVE_DATASETS = [p.name for p in
                      sorted(BASE.glob("waveprocessed/PROCESSED-*"))]
pu.TEXFIGU_DIR = BASE / "output" / "TEXFIGU"
pu.FIGURES_DIR = thesis_dir
pu.TEXFIGU_DIR.mkdir(parents=True, exist_ok=True)

# Numeric stats to embed in the immutable block (single source of truth — the
# caption can reference "Hs deficit −5.5 % at IN-wall", "per-snippet σ noise
# 0.83 mm" etc. without re-deriving from the CSV).
def _row(seg, probe_label):
    sub = ens_table[(ens_table["probe"] == probe_label) &
                    (ens_table["segment"].str.startswith(seg))]
    return sub.iloc[0] if len(sub) else None

_extra_stats = {"snippet_seconds": f"{SNIPPET_S:g}"}
for _probe_label_short, _probe_full in (("upstream", LABELS["8804/250"]),
                                          ("inWall",   LABELS["9373/170"]),
                                          ("inFar",    LABELS["9373/340"]),
                                          ("Out",      LABELS["12400/250"])):
    long_row = _row("long nowave runs", _probe_full)
    snip_row = _row(f"{SNIPPET_S:g} s pre-paddle", _probe_full)
    if long_row is not None and snip_row is not None:
        ref_sigma = float(long_row["sigma_mm"])
        sni_sigma = float(snip_row["sigma_mm"])
        delta_pct = (sni_sigma - ref_sigma) / ref_sigma * 100.0 if ref_sigma else float("nan")
        _extra_stats[f"sigma_{_probe_label_short}_long_mm"] = f"{ref_sigma:.3f}"
        _extra_stats[f"sigma_{_probe_label_short}_snip_mm"] = f"{sni_sigma:.3f}"
        _extra_stats[f"sigma_{_probe_label_short}_delta_pct"] = f"{delta_pct:+.2f}"
        _extra_stats[f"Hs_{_probe_label_short}_long_mm"] = f"{float(long_row['Hs_mm']):.3f}"
        _extra_stats[f"Hs_{_probe_label_short}_snip_mm"] = f"{float(snip_row['Hs_mm']):.3f}"

_meta = pu.build_fig_meta(
    {
        "filters": {"WindCondition": "full"},
        "plotting": {"figure_name": THESIS_NAME},
    },
    chapter="04",
    extra={"script": "analysis_scratch/wind_2s_vs_360s.py"},
    computed_in="analysis_scratch/wind_2s_vs_360s.py",
    data_class="DELEG",
    findings_doc="(none)",
    extra_params=(
        f"Wind characterisation validation: does the {SNIPPET_S:g} s pre-paddle "
        f"window of a wave run reproduce the wind PSD measured on long "
        f"(~31–381 s) nowave+fullwind runs? "
        f"Canon datasets only (PROCESSED-20260326-*-lowrange + "
        f"PROCESSED-20260327-*-lowrange — final probe config "
        f"march2026_better_rearranging). Long-run set: "
        f"{len(long_present)} runs, durations 31, 33, 63, 360, 381 s. "
        f"Per-long-run Welch PSD (Hann, nperseg = 8 s, linear detrend, "
        f"Δf = 0.125 Hz) → 16–84 percentile envelope + median (green). "
        f"Pre-paddle ensemble: first {SNIPPET_S:g} s of all "
        f"{snip_psds.shape[0]} fullwind+wave runs in the same datasets — "
        f"safe because √(gh) = 2.39 m/s and the closest probe (8804 mm) "
        f"sees no paddle motion before 3.68 s. Per-snippet single periodogram "
        f"(Hann, linear detrend, Δf = {1/SNIPPET_S:.3f} Hz) → 16–84 "
        f"envelope + median (orange dashed). Probe shown: "
        f"{LABELS[PSD_PROBE]}. Colour convention is local (green = long-run, "
        f"orange = pre-paddle); does NOT use the thesis-wide WIND_COLOR_MAP "
        f"because both curves represent the same wind condition (fullwind)."
    ),
    extra_stats=_extra_stats,
)
pu.write_figure_stub(_meta, plot_type="wind_pre_paddle_psd", force=True)
print(f"  stub → output/TEXFIGU/{THESIS_NAME}.tex")

print("Done.")
