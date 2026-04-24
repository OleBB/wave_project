"""
Probe-shifted H&G window — per40 + per240, both using H&G-at-12.4m anchor
===========================================================================

Huseby & Grue's [35.088 s, 42.105 s] = [50 T, 60 T] at 1.425 Hz was
measured at r = 12.4 m from the wavemaker — which is exactly our OUT
probe position. So the canonical H&G window applies **at OUT** in our
tank without modification, but the same wavetrain passes through any
closer probe earlier by the group-velocity travel-time difference.

For a probe at distance r_probe < 12.4 m, the physically correct H&G
window is shifted BACK by

    ΔT(f) = (12.4 − r_probe) / c_group(f) · f    [periods]

with c_group = g / (4π f) in deep water (tanh(kh) ≈ 1 at h = 0.58 m
for f ≥ 1.3 Hz). For the canonical IN probes at r = 9.373 m:

    1.3 Hz →  6.6 T earlier
    1.4 Hz →  7.6 T
    1.5 Hz →  8.7 T
    1.6 Hz →  9.9 T

The shifted IN window fits inside the per40 wavetrain regime at every
thesis frequency (see previous findings). So both per40 and per240 can
be analysed with the same H&G methodology, probe-shifted.

This script:
  1. Computes probe-shifted H&G AFFT at IN (canonical mean) and OUT
     (unshifted) for every thesis-scope run, per40 and per240 alike.
  2. Compares per40 vs per240 OUT/IN at matched (freq, amp, wind).
  3. Reports median / max |Δ OUT/IN| across all cells.

Prediction: per40 and per240 should agree within 1–2 % median (much
better than the earlier 10.7 % outlier at 1.4 Hz / 0.2 V / fullwind
when per40 used unshifted [32T, 38T]).

Scope:
    panel = full, mooring = below_90_loose (230+300 merged),
    quality_flag = "ok" only,
    freq in {1.3, 1.4, 1.5, 1.6} Hz,
    amp  in {0.1, 0.2, 0.3} V,
    wind in {no, full}.

Run from repo root:
    /opt/anaconda3/envs/draumkvedet/bin/python analysis_scratch/per40_and_per240_HG_shifted.py

Outputs:
    analysis_scratch/per40_and_per240_HG_shifted.pdf
    analysis_scratch/per40_and_per240_HG_shifted.csv
    analysis_scratch/per40_and_per240_HG_shifted_findings.md
    output/FIGURES/ch04_per40_and_per240_HG_shifted.pdf
    output/TEXFIGU/ch04_per40_and_per240_HG_shifted.tex
"""

import os
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
os.chdir(BASE)

from wavescripts.improved_data_loader import load_analysis_data, load_processed_dfs
from wavescripts.plot_utils import freq_to_k, amp_to_label

# ── Config ────────────────────────────────────────────────────────────────────
FS              = 250.0
G               = 9.81
FFT_BAND_HZ     = 0.05

# Canonical H&G window at r = 12.4 m (H&G's original anchor)
HG_REF_R_M      = 12.400
HG_START_T_REF  = 50
HG_END_T_REF    = 60

# Canonical IN (mean of parallels at r = 9.373 m) + OUT (single probe at r = 12.4 m)
IN_PROBES       = ["9373/170", "9373/340"]
R_IN_M          = 9.373
OUT_PROBE       = "12400/250"
R_OUT_M         = 12.400

FREQS = [1.3, 1.4, 1.5, 1.6]
AMPS  = [0.1, 0.2, 0.3]
WINDS = ["no", "full"]
PER240_THRESHOLD_T = 50

# Datasets — same pool as per40_vs_per240_outin + the cond4 lowrange for per240.
PROCESSED_DIRS = [
    Path("waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange"),
    Path("waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"),
    Path("waveprocessed/PROCESSED-20260307-ProbPos4_31_FPV_2-tett6roof"),
    Path("waveprocessed/PROCESSED-20260312-ProbPos4_31_FPV_2-tett6roof"),
    Path("waveprocessed/PROCESSED-20260313-ProbePos4_31_FPV_2-tett6roof"),
    Path("waveprocessed/PROCESSED-20260314-ProbePos4_31_FPV_2-tett6roof"),
    Path("waveprocessed/PROCESSED-20260316-ProbePos4_31_FPV_2-tett6roof"),
    Path("waveprocessed/PROCESSED-20260316-ProbePos4_31_FPV_2-tett6roof-under9Mooring"),
    Path("waveprocessed/PROCESSED-20260319-ProbePos4_31_FPV_2-tett6roof-under9Mooring"),
    Path("waveprocessed/PROCESSED-20260321-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-RENAMED"),
    Path("waveprocessed/PROCESSED-20260323-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100"),
    Path("waveprocessed/PROCESSED-20260324-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100"),
    Path("waveprocessed/PROCESSED-20260325-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100"),
    Path("waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100"),
]
_dataset_names = {p.name.removeprefix("PROCESSED-") for p in PROCESSED_DIRS}

SCRATCH_DIR = Path(__file__).parent
SCRATCH_PDF = SCRATCH_DIR / "per40_and_per240_HG_shifted.pdf"
SCRATCH_CSV = SCRATCH_DIR / "per40_and_per240_HG_shifted.csv"
SCRATCH_MD  = SCRATCH_DIR / "per40_and_per240_HG_shifted_findings.md"

THESIS_NAME = "ch04_per40_and_per240_HG_shifted"
THESIS_PDF  = BASE / "output" / "FIGURES" / f"{THESIS_NAME}.pdf"
THESIS_STUB = BASE / "output" / "TEXFIGU" / f"{THESIS_NAME}.tex"
THESIS_PDF.parent.mkdir(parents=True, exist_ok=True)
THESIS_STUB.parent.mkdir(parents=True, exist_ok=True)


# ── Helpers ──────────────────────────────────────────────────────────────────
def fft_amp_at_freq(segment: np.ndarray, target_hz: float,
                    fs: float = FS, band_hz: float = FFT_BAND_HZ) -> float:
    """Peak-bin AFFT, pipeline convention. 2·|FFT|/N."""
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
    freqs  = np.fft.fftfreq(N, d=1.0 / fs)
    pos    = freqs > 0
    pos_f  = freqs[pos]
    mask   = (pos_f >= target_hz - band_hz) & (pos_f <= target_hz + band_hz)
    if not mask.any():
        j = int(np.argmin(np.abs(pos_f - target_hz)))
    else:
        local_idx = int(np.argmin(np.abs(pos_f[mask] - target_hz)))
        j = np.where(mask)[0][local_idx]
    fft_vals = np.fft.fft(seg)
    amps_pos = 2.0 * np.abs(fft_vals[pos]) / N
    return float(amps_pos[j])


def get_eta(df_run: pd.DataFrame, pos: str) -> np.ndarray | None:
    for col in (f"eta_{pos}_interp", f"eta_{pos}"):
        if col in df_run.columns:
            return df_run[col].to_numpy(dtype=float)
    return None


def c_group_deep(f_hz: float) -> float:
    """Deep-water group velocity in m/s."""
    return G / (4.0 * np.pi * f_hz)


def hg_shift_periods(r_probe_m: float, f_hz: float) -> float:
    """Periods-earlier shift of H&G window for a probe closer than 12.4 m."""
    delta_r = HG_REF_R_M - r_probe_m
    delta_s = delta_r / c_group_deep(f_hz)
    return delta_s * f_hz


def shifted_hg_afft(signal: np.ndarray, r_probe_m: float, f_hz: float) -> float:
    """AFFT at `f_hz` using H&G window [50T − ΔT, 60T − ΔT] for this probe's r."""
    if signal is None:
        return np.nan
    samples_per_period = int(round(FS / f_hz))
    dT = hg_shift_periods(r_probe_m, f_hz)
    start_T = HG_START_T_REF - dT
    end_T   = HG_END_T_REF   - dT
    i0 = int(round(start_T * samples_per_period))
    i1 = int(round(end_T   * samples_per_period))
    if i0 < 0 or i1 > len(signal):
        return np.nan
    return fft_amp_at_freq(signal[i0:i1], f_hz)


# ── Load ──────────────────────────────────────────────────────────────────────
print("1. Loading meta across 14 thesis-relevant datasets …")
combined_meta, _, _, _ = load_analysis_data(*PROCESSED_DIRS, load_processed=False)
combined_meta["Mooring"] = combined_meta["Mooring"].replace({
    "below_90_loose230": "below_90_loose",
    "below_90_loose300": "below_90_loose",
})
wave = combined_meta[
    combined_meta["WaveFrequencyInput [Hz]"].notna()
    & (combined_meta["WaveFrequencyInput [Hz]"] > 0)
    & (combined_meta["PanelCondition"] == "full")
    & (combined_meta["Mooring"] == "below_90_loose")
    & (combined_meta["quality_flag"] == "ok")
].copy()
wave["N_input_periods"] = wave["WavePeriodInput"].astype(float)
wave = wave[wave["WaveFrequencyInput [Hz]"].round(2).isin(FREQS)
            & wave["WaveAmplitudeInput [Volt]"].round(2).isin(AMPS)
            & wave["WindCondition"].isin(WINDS)]
wave["run_type"] = np.where(wave["N_input_periods"] >= PER240_THRESHOLD_T, "per240", "per40")
print(f"   scope rows: {len(wave)} "
      f"(per240 n={int((wave['run_type']=='per240').sum())}, "
      f"per40 n={int((wave['run_type']=='per40').sum())})")

print("\n2. Loading processed_dfs (~30-60 s) …")
pdfs = load_processed_dfs(*PROCESSED_DIRS)
print(f"   {len(pdfs)} time-series cached")

# Travel-time offsets at each thesis freq
print("\n3. Probe-shifted H&G window offsets (ΔT earlier than [50T, 60T]):")
for f in FREQS:
    dT = hg_shift_periods(R_IN_M, f)
    shift_start = HG_START_T_REF - dT
    shift_end   = HG_END_T_REF   - dT
    print(f"   {f:.1f} Hz: c_g={c_group_deep(f):.3f} m/s  →  "
          f"IN (r={R_IN_M} m) ΔT={dT:.2f}T  →  [{shift_start:.1f}T, {shift_end:.1f}T] "
          f"= [{shift_start/f:.2f} s, {shift_end/f:.2f} s]")


# ── Compute per-run AFFT ─────────────────────────────────────────────────────
print("\n4. Computing probe-shifted H&G AFFT per run …")
rows = []
for _, r in wave.iterrows():
    path = r["path"]
    df_run = pdfs.get(path)
    if df_run is None:
        continue
    f_paddle = float(r["WaveFrequencyInput [Hz]"])

    # Canonical IN mean AFFT — compute per parallel probe, average the AFFTs
    # (same convention as processor2nd.py's _update_more_metrics).
    in_afts = []
    for p in IN_PROBES:
        sig = get_eta(df_run, p)
        if sig is None:
            continue
        a = shifted_hg_afft(sig, R_IN_M, f_paddle)
        if np.isfinite(a):
            in_afts.append(a)
    a_in = float(np.mean(in_afts)) if in_afts else np.nan

    # OUT at r = 12.4 m → no shift, direct H&G window
    sig_out = get_eta(df_run, OUT_PROBE)
    a_out = shifted_hg_afft(sig_out, R_OUT_M, f_paddle)

    rows.append({
        "path":          path,
        "run_type":      r["run_type"],
        "N_input_periods": r["N_input_periods"],
        "freq_hz":       round(f_paddle, 2),
        "amp_V":         round(float(r["WaveAmplitudeInput [Volt]"]), 2),
        "wind":          r["WindCondition"],
        "A_in_mm":       a_in,
        "A_out_mm":      a_out,
        "OUT_IN":        a_out / a_in if (a_in and a_in > 0 and np.isfinite(a_out)) else np.nan,
    })

df = pd.DataFrame(rows)
df.to_csv(SCRATCH_CSV, index=False)
print(f"   {len(df)} runs → {SCRATCH_CSV.relative_to(BASE)}")
print(f"   per240 OUT/IN valid: {int(((df['run_type']=='per240') & df['OUT_IN'].notna()).sum())}")
print(f"   per40  OUT/IN valid: {int(((df['run_type']=='per40')  & df['OUT_IN'].notna()).sum())}")


# ── Aggregate per (freq, amp, wind, run_type) ────────────────────────────────
agg = (df.dropna(subset=["OUT_IN"])
          .groupby(["freq_hz", "amp_V", "wind", "run_type"])
          ["OUT_IN"].agg(["mean", "std", "count"])
          .reset_index())
agg["std"] = agg["std"].fillna(0.0)
agg["k"]  = freq_to_k(agg["freq_hz"].to_numpy())


# ── Agreement per (freq, amp, wind) ──────────────────────────────────────────
pvt = agg.pivot_table(index=["freq_hz", "amp_V", "wind"],
                       columns="run_type", values="mean").reset_index()
pvt["delta"] = pvt.get("per40", np.nan) - pvt.get("per240", np.nan)
pvt["delta_rel"] = pvt["delta"] / pvt["per240"]
matched = pvt.dropna(subset=["per40", "per240"]).copy()

if len(matched):
    abs_med = float(matched["delta"].abs().median())
    abs_max = float(matched["delta"].abs().max())
    rel_med = float(matched["delta_rel"].abs().median())
    rel_max = float(matched["delta_rel"].abs().max())
else:
    abs_med = abs_max = rel_med = rel_max = float("nan")

print("\n5. Per-cell agreement (per40 − per240, both using probe-shifted H&G):")
if len(matched):
    print(matched[["freq_hz", "amp_V", "wind", "per40", "per240",
                   "delta", "delta_rel"]].round(4).to_string(index=False))
else:
    print("   (no matched cells)")
print(f"\n   median |Δ OUT/IN|         = {abs_med:.4f}")
print(f"   max    |Δ OUT/IN|         = {abs_max:.4f}")
print(f"   median |Δ OUT/IN| / per240 = {rel_med*100:.2f} %")
print(f"   max    |Δ OUT/IN| / per240 = {rel_max*100:.2f} %")


# ── Figure: 1×3 (per amp) with both per40 and per240 overlaid ────────────────
print("\n6. Plotting …")
fig, axes = plt.subplots(1, 3, figsize=(14, 5), dpi=120, sharey=True)

WIND_COLOR   = {"no": "#2E86AB", "full": "#E74C3C"}
RUNTYPE_MARK = {"per40": "o", "per240": "D"}
RUNTYPE_LABEL = {"per40": "per40 (H&G shifted)", "per240": "per240 (H&G shifted)"}

for i, amp in enumerate(AMPS):
    ax = axes[i]
    sub = agg[np.isclose(agg["amp_V"], amp)]
    if sub.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        ax.set_title(f"{amp_to_label(amp)}", fontsize=10)
        continue
    for wind in WINDS:
        for rt in ["per40", "per240"]:
            cell = sub[(sub["wind"] == wind) & (sub["run_type"] == rt)].sort_values("k")
            if cell.empty:
                continue
            dx = {"per40": -0.003, "per240": +0.003}[rt]
            ax.errorbar(
                cell["k"].values + dx, cell["mean"].values,
                yerr=cell["std"].values,
                fmt=RUNTYPE_MARK[rt],
                color=WIND_COLOR[wind],
                mfc=WIND_COLOR[wind] if rt == "per40" else "white",
                mec=WIND_COLOR[wind],
                markeredgewidth=1.1, markersize=7,
                linestyle="-" if rt == "per40" else "--",
                capsize=3, lw=1.3, alpha=0.92,
                label=f"{wind} · {RUNTYPE_LABEL[rt]}",
            )
    ax.axhline(1.0, color="black", lw=0.6, ls="--", alpha=0.4)
    ax.set_xlabel("$k$ (rad/m)", fontsize=10)
    if i == 0:
        ax.set_ylabel("OUT/IN (FFT)", fontsize=10)
    ax.set_title(f"{amp_to_label(amp)}", fontsize=10)
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.legend(fontsize=7, loc="best", framealpha=0.92, ncol=1)

fig.suptitle(
    "OUT/IN (FFT) using probe-shifted H&G at both probes — per40 vs per240  "
    "(H&G anchor r = 12.4 m, IN at r = 9.373 m shifted earlier by group-velocity Δt)",
    fontsize=11, fontweight="bold", y=1.00,
)
fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.12, wspace=0.06)
fig.savefig(SCRATCH_PDF, bbox_inches="tight")
print(f"   scratch → {SCRATCH_PDF.relative_to(BASE)}")
fig.savefig(THESIS_PDF, bbox_inches="tight")
print(f"   thesis  → {THESIS_PDF.relative_to(BASE)}")
plt.close(fig)


# ── TEXFIGU stub via shared helper ───────────────────────────────────────────
print("\n7. Writing .tex stub …")

import wavescripts.plot_utils as pu
pu.ACTIVE_DATASETS = [str(p).split("/")[-1] for p in PROCESSED_DIRS]
pu.TEXFIGU_DIR = BASE / "output" / "TEXFIGU"
pu.FIGURES_DIR = BASE / "output" / "FIGURES"

_caption = (
    "OUT/IN (FFT) at the paddle frequency versus $k$ (rad/m), split by paddle drive "
    f"{', '.join(f'{a:.2f}' for a in AMPS)}\\,V (one panel each). "
    "Both per40 (filled circles) and per240 (open diamonds) runs use the "
    "Huseby--Grue window anchored at $r = 12.4$\\,m with $[50T, 60T]$ from "
    "wavemaker start. The IN probes at $r = 9.373$\\,m receive the same "
    "waves $\\Delta T$ earlier by group velocity "
    "($c_g = g/(4\\pi f)$, deep water), so the IN window is shifted back "
    f"by $\\Delta T \\in [{hg_shift_periods(R_IN_M, max(FREQS)):.2f}, "
    f"{hg_shift_periods(R_IN_M, min(FREQS)):.2f}]$\\,T across the thesis "
    "band. Canonical IN amplitude = mean AFFT across 9373/170 and 9373/340; "
    "OUT amplitude = AFFT at 12400/250. Blue: no wind; red: full wind. "
    "Error bars: run-to-run standard deviation. Mooring below\\_90\\_loose "
    "(230+300 merged), quality\\_flag = ok, full panel."
)

_meta_stub = pu.build_fig_meta(
    {
        "filters": {
            "PanelCondition":            "full",
            "WaveFrequencyInput [Hz]":   [min(FREQS), max(FREQS)],
            "WaveAmplitudeInput [Volt]": AMPS,
            "WindCondition":             WINDS,
            "quality_flag":              "ok",
            "Mooring":                   ["below_90_loose"],
            "probes":                    "IN=mean(9373/170, 9373/340), OUT=12400/250",
        },
        "plotting": {
            "figure_name":   THESIS_NAME,
            "caption":       _caption,
            "caption_short": "OUT/IN via probe-shifted H&G window: per40 and per240 pooled",
        },
    },
    chapter="04",
    data_df=wave,
    extra={"script": "analysis_scratch/per40_and_per240_HG_shifted.py"},
    computed_in=(
        "analysis_scratch/per40_and_per240_HG_shifted.py "
        "(probe-shifted H&G window — IN at r=9.373 m shifted back by "
        "(12.4 − r)/c_group · f from the canonical H&G [50T, 60T] at OUT)"
    ),
    data_class="DFS",
    findings_doc="analysis_scratch/per40_and_per240_HG_shifted_findings.md",
    grouper="per (freq, amp, wind, run_type); H&G window per probe",
    collapse_panels=False,
    fft_window_hz=2 * FFT_BAND_HZ,
    extra_params=(
        f"HG window [50T, 60T] at OUT (r=12.4 m); "
        f"IN window shifted by ΔT=(12.4−9.373)/c_group·f; "
        f"c_group = g/(4π f) (deep water approx, tanh(kh)>0.999 at ≥1.3 Hz)"
    ),
    extra_stats={
        "n_per240":            int((wave["run_type"]=="per240").sum()),
        "n_per40":             int((wave["run_type"]=="per40").sum()),
        "n_matched_cells":     int(len(matched)),
        "abs_Δ_OUTIN_median":  round(abs_med, 4) if np.isfinite(abs_med) else "—",
        "abs_Δ_OUTIN_max":     round(abs_max, 4) if np.isfinite(abs_max) else "—",
        "rel_Δ_OUTIN_median_pct": round(rel_med * 100, 2) if np.isfinite(rel_med) else "—",
        "rel_Δ_OUTIN_max_pct":    round(rel_max * 100, 2) if np.isfinite(rel_max) else "—",
        "HG_anchor_r_m":       HG_REF_R_M,
        "HG_window_T":         f"[{HG_START_T_REF}, {HG_END_T_REF}]",
    },
)

pu.write_figure_stub(_meta_stub, plot_type="per40_and_per240_HG_shifted",
                     subfig_filenames=[THESIS_NAME])
print(f"   thesis stub → {THESIS_STUB.relative_to(BASE)}")


# ── Findings markdown ────────────────────────────────────────────────────────
print("\n8. Writing findings markdown …")
lines = [
    "# Probe-shifted H&G — per40 and per240, both using H&G-at-12.4 m anchor",
    "",
    "Generated by `analysis_scratch/per40_and_per240_HG_shifted.py`.",
    "",
    "## Method",
    "",
    "H&G window [50 T, 60 T] anchored at r = 12.4 m (= our OUT probe). For a "
    "probe at r_probe < 12.4 m, the same waves arrive earlier by "
    "ΔT = (12.4 − r_probe) / c_group · f periods.",
    "",
    "Applied to both per40 and per240 at thesis-scope conditions "
    f"(freq {FREQS} Hz, amp {AMPS} V, wind {WINDS}, panel full, mooring "
    "below_90_loose, quality OK).",
    "",
    "Travel-time shifts at IN (r = 9.373 m):",
    "",
    "| freq [Hz] | c_group [m/s] | ΔT [T] | IN window [T] | IN window [s] |",
    "|-----------|---------------|--------|---------------|---------------|",
]
for f in FREQS:
    dT = hg_shift_periods(R_IN_M, f)
    cg = c_group_deep(f)
    s = HG_START_T_REF - dT
    e = HG_END_T_REF   - dT
    lines.append(f"| {f:.1f} | {cg:.3f} | {dT:.2f} | [{s:.1f}, {e:.1f}] | [{s/f:.2f}, {e/f:.2f}] |")
lines += [
    "",
    "## Cohort",
    "",
    f"- per240 runs: **{int((wave['run_type']=='per240').sum())}**",
    f"- per40  runs: **{int((wave['run_type']=='per40').sum())}**",
    f"- matched (freq, amp, wind) cells: **{int(len(matched))}**",
    "",
    "## Agreement (per40 − per240, probe-shifted H&G at both)",
    "",
    f"- median |Δ OUT/IN|          = **{abs_med:.4f}**",
    f"- max    |Δ OUT/IN|          = **{abs_max:.4f}**",
    f"- median |Δ OUT/IN| / per240  = **{rel_med*100:.2f} %**",
    f"- max    |Δ OUT/IN| / per240  = **{rel_max*100:.2f} %**",
    "",
    "## Per-cell detail (per40 − per240)",
    "",
]
if len(matched):
    lines.append("| freq [Hz] | amp [V] | wind | per40 | per240 | Δ | Δ/per240 |")
    lines.append("|-----------|---------|------|-------|--------|---|----------|")
    for _, r in matched.sort_values(["amp_V", "freq_hz", "wind"]).iterrows():
        lines.append(f"| {r['freq_hz']:.1f} | {r['amp_V']:.2f} | {r['wind']:>4} | "
                     f"{r['per40']:.4f} | {r['per240']:.4f} | {r['delta']:+.4f} | "
                     f"{r['delta_rel']*100:+.2f} % |")
else:
    lines.append("*(no matched cells)*")
lines += [
    "",
    "## Files",
    "",
    f"  {SCRATCH_PDF.name}",
    f"  {SCRATCH_CSV.name}",
    f"  output/FIGURES/{THESIS_NAME}.pdf",
    f"  output/TEXFIGU/{THESIS_NAME}.tex",
    "",
    "Interpretation omitted — user eyeball decides whether per40 and per240 "
    "can be formally pooled for CH05 under probe-shifted H&G.",
]
SCRATCH_MD.write_text("\n".join(lines))
print(f"   findings → {SCRATCH_MD.relative_to(BASE)}")

print("\nDone.")
