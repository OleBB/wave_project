"""
Wind-effect realness check at low frequencies — 4-way mooring split
====================================================================

Investigate whether the visible "wind decreases transmission at 1.3-1.4 Hz"
pattern in the above-water semi-stiff mooring is statistically real, OR
within errorbar noise. Now using the 4-way mooring categorization and
color/marker scheme from scatter_by_mooring_4way.pdf for visual consistency.

User question (2026-05-11):
  - above-water semi-stiff (16 cm strikk, full panel, canon era): does
    wind significantly decrease K_t at 1.3-1.4 Hz?
  - A2 (0.2V) appears to "behave oddly" — is this an experiment-error
    artefact (bad day, too-close rest time, weaker wind that day) or a
    universal pattern visible in canon (under-mooring) runs too?

4-way mooring categorization (= scatter_by_mooring_4way.pdf):
  - canon_loose300   = below_90, 30 cm strikk, full panel, canon era
  - canon_loose230   = below_90, 23 cm strikk, full panel, canon era
  - canon_above_loose= above_50, 16 cm strikk, full panel, canon era  (the user's "above semi-stiff")
  - nov_above_stiff  = above_50, 6 cm  strikk, REVERSE panel, Nov 2025 era

Scope:
  - Freqs main:  {1.3, 1.4, 1.5, 1.6} Hz
  - Freq support: {1.2} Hz (decent data, shown dimmer)
  - Excluded:    {1.7+} Hz (user: not good data)
  - Amps:        A1 (0.1V), A2 (0.2V), A3 (0.3V) — per-amp breakdown
  - Quality:     take everything, flag (no drops on quality_flag)

For each (cat, freq, amp) cell:
  ΔK_t = K_t(full) - K_t(no)
  bootstrap 95% CI (10k draws, independent resampling each side)
  Welch's t-test p-value
  verdict: decrease / enhance / null / insufficient_n

Outputs (analysis_scratch/):
  wind_effect_realness_lowfreq_runs.csv       — per-run forensic table
  wind_effect_realness_lowfreq_cells.csv      — per-cell ΔK_t stats
  wind_effect_realness_lowfreq.pdf            — ΔK_t-vs-freq, 4-way colored
  wind_effect_realness_lowfreq_kt.pdf         — K_t-vs-freq, 4-way colored
  wind_effect_realness_lowfreq_a2_focus.pdf   — A2 cross-amp dip check + 1.4 Hz A2 forensic
  wind_effect_realness_lowfreq_a2_runs.csv    — A2 per-run table w/ session context

Run from repo root:
    conda run -n draumkvedet python analysis_scratch/wind_effect_realness_lowfreq.py
"""

import sys
import warnings
from pathlib import Path
import glob

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats as scistats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import os
os.chdir(BASE)

from wavescripts.improved_data_loader import load_analysis_data, get_configuration_for_date


# ── Tunables ───────────────────────────────────────────────────────────────────
FREQS_MAIN    = (1.3, 1.4, 1.5, 1.6)
FREQS_SUPPORT = (1.2,)
FREQS_ALL     = tuple(sorted(FREQS_SUPPORT + FREQS_MAIN))
AMPS          = (0.10, 0.20, 0.30)
N_BOOT        = 10_000
RNG_SEED      = 20260511
OVERRIDE_THRESHOLD = 0.05
ALPHA         = 0.05

# 4-way categorization + colors + markers — matches scatter_by_mooring_4way.pdf
CAT_LABELS = {
    "canon_loose300":    "Under, 30 cm",
    "canon_loose230":    "Under, 23 cm",
    "canon_above_loose": "Over, 16 cm (semi-stiff)",
    "nov_above_stiff":   "Over, 6 cm + revers (Nov)",
}
CAT_ORDER_PLOT = [
    "canon_above_loose",
    "nov_above_stiff",
    "canon_loose230",
    "canon_loose300",
]
COLORS_WIND = {
    ("canon_loose300",    "no"):   "#1F77B4",
    ("canon_loose300",    "full"): "#D62728",
    ("canon_loose230",    "no"):   "#9ECAE1",
    ("canon_loose230",    "full"): "#FCAE91",
    ("canon_above_loose", "no"):   "#1ABC9C",
    ("canon_above_loose", "full"): "#E91E63",
    ("nov_above_stiff",   "no"):   "#9B59B6",
    ("nov_above_stiff",   "full"): "#F1C40F",
}
COLOR_CAT = {  # one color per mooring for ΔK_t plots — use the fullwind tone
    "canon_loose300":    "#D62728",
    "canon_loose230":    "#FCAE91",
    "canon_above_loose": "#E91E63",
    "nov_above_stiff":   "#D4A017",  # darker yellow for readability on white
}
MARKERS = {0.10: "o", 0.20: "s", 0.30: "^"}


# ── I/O ────────────────────────────────────────────────────────────────────────
OUT_RUNS      = Path(__file__).parent / "wind_effect_realness_lowfreq_runs.csv"
OUT_CELLS     = Path(__file__).parent / "wind_effect_realness_lowfreq_cells.csv"
OUT_PDF_DELTA = Path(__file__).parent / "wind_effect_realness_lowfreq.pdf"
OUT_PDF_KT    = Path(__file__).parent / "wind_effect_realness_lowfreq_kt.pdf"
OUT_PDF_A2    = Path(__file__).parent / "wind_effect_realness_lowfreq_a2_focus.pdf"
OUT_A2_RUNS   = Path(__file__).parent / "wind_effect_realness_lowfreq_a2_runs.csv"

rng = np.random.default_rng(RNG_SEED)


# ── 1. Load ────────────────────────────────────────────────────────────────────
print("1. Loading all processed folders …")
all_dirs = sorted(glob.glob(str(BASE / "waveprocessed" / "PROCESSED-*")))
print(f"   {len(all_dirs)} folders")
meta, _, _, _ = load_analysis_data(*all_dirs, load_processed=False)
print(f"   {len(meta)} total rows")


# ── 2. Scope filter — take everything, flag (no quality drops) ────────────────
def _r2(v):
    try: return round(float(v), 2)
    except (TypeError, ValueError): return np.nan

m = meta.copy()
m["freq_r"] = m["WaveFrequencyInput [Hz]"].apply(_r2)
m["amp_r"]  = m["WaveAmplitudeInput [Volt]"].apply(_r2)

in_scope = m[
    m["freq_r"].isin(FREQS_ALL)
    & m["amp_r"].isin(AMPS)
    & m["WindCondition"].isin(["no", "full"])
    & m["Mooring"].isin(["below_90_loose230", "below_90_loose300", "above_50"])
    & m["OUT/IN (FFT)"].notna()
    & m["in_probes_used"].notna()
    & m["out_probes_used"].notna()
].copy()

n_ext = ((in_scope["OUT/IN (FFT)"] > 2.0) | (in_scope["OUT/IN (FFT)"] < 0.1)).sum()
in_scope = in_scope[(in_scope["OUT/IN (FFT)"] <= 2.0) & (in_scope["OUT/IN (FFT)"] >= 0.1)].copy()
print(f"   {len(in_scope)} rows in scope (clipped {int(n_ext)} extreme K_t_FFT)")


# ── 3. Probe-geometry config (file_date → cfg → 4-way category) ───────────────
in_scope["file_date_dt"] = pd.to_datetime(in_scope["file_date"]).dt.tz_localize(None)
in_scope["cfg"] = in_scope["file_date_dt"].apply(
    lambda d: get_configuration_for_date(d).name if pd.notnull(d) else "N/A"
)

def _category(row):
    m, c, p = row["Mooring"], row["cfg"], row["PanelCondition"]
    if m == "below_90_loose300" and c == "march2026_better_rearranging": return "canon_loose300"
    if m == "below_90_loose230" and c == "march2026_better_rearranging": return "canon_loose230"
    if m == "above_50"          and c == "march2026_better_rearranging" and p == "full":     return "canon_above_loose"
    if m == "above_50"          and c == "nov_normalt_oppsett"          and p == "reverse":  return "nov_above_stiff"
    return "other"

in_scope["category"] = in_scope.apply(_category, axis=1)
n_other = int((in_scope["category"] == "other").sum())
if n_other:
    print(f"   {n_other} 'other' rows (probably mixed panel/era) — kept for forensic CSV, dropped from figures")

wave = in_scope[in_scope["category"].isin(CAT_ORDER_PLOT)].copy()
print(f"   {len(wave)} rows after 4-way categorisation")
print("\n2. Counts per (category × wind):")
print(wave.groupby(["category", "WindCondition"]).size().unstack(fill_value=0).to_string())


# ── 4. K_t three ways + LS override ───────────────────────────────────────────
def _kt(row, suffix):
    inp = [p.strip() for p in str(row["in_probes_used"]).split("+")]
    out = [p.strip() for p in str(row["out_probes_used"]).split("+")]
    try:
        a_in  = float(np.nanmean([row.get(f"Probe {p} Amplitude{suffix}", np.nan) for p in inp]))
        a_out = float(np.nanmean([row.get(f"Probe {p} Amplitude{suffix}", np.nan) for p in out]))
        if a_in <= 0 or not np.isfinite(a_in) or not np.isfinite(a_out):
            return np.nan
        return a_out / a_in
    except (KeyError, TypeError):
        return np.nan

wave["Kt_FFT"] = wave["OUT/IN (FFT)"].astype(float)
wave["Kt_LS"]  = wave.apply(lambda r: _kt(r, " (LS)"),  axis=1)
wave["Kt_PSD"] = wave.apply(lambda r: _kt(r, " (PSD)"), axis=1)

_dF  = (wave["Kt_FFT"] - wave["Kt_LS"]).abs()
_dP  = (wave["Kt_FFT"] - wave["Kt_PSD"]).abs()
_dLP = (wave["Kt_LS"]  - wave["Kt_PSD"]).abs()
wave["override_fired"] = (_dF > OVERRIDE_THRESHOLD) & (_dP > OVERRIDE_THRESHOLD) & (_dLP < OVERRIDE_THRESHOLD)
wave["Kt_eff"] = np.where(wave["override_fired"], wave["Kt_LS"], wave["Kt_FFT"])
print(f"\n   K_t override fired on {int(wave['override_fired'].sum())} of {len(wave)} rows")


# ── 5. Per-row quality flags + session context ────────────────────────────────
def _cut_any(row, side):
    probes = [p.strip() for p in str(row[f"{side}_probes_used"]).split("+")]
    vals = []
    for p in probes:
        col = f"cut_samples_{p}"
        if col in row.index:
            v = row[col]
            if pd.notna(v): vals.append(int(v))
    return any(v > 0 for v in vals)

wave["cut_in_any"]  = wave.apply(lambda r: _cut_any(r, "in"),  axis=1)
wave["cut_out_any"] = wave.apply(lambda r: _cut_any(r, "out"), axis=1)
wave["qflag_nonok"] = (wave["quality_flag"] != "ok") if "quality_flag" in wave.columns else False
wave["fft_ls_dis"]  = _dF > OVERRIDE_THRESHOLD
wave["ws_in_low"]   = (wave.get("IN wave_stability", pd.Series(np.nan, index=wave.index)).astype(float) < 0.7)
wave["any_flag"] = (wave["cut_in_any"] | wave["cut_out_any"]
                    | wave["qflag_nonok"] | wave["fft_ls_dis"]
                    | wave["ws_in_low"])

# Session context: mstop (rest), per (run length), run_number, paddle/wind config.
wave["mstop_sec"] = wave.get("Extra seconds", pd.Series(np.nan, index=wave.index))
wave["per_train"] = wave.get("WavePeriodInput", pd.Series(np.nan, index=wave.index))
wave["run_num"]   = wave.get("Run number",     pd.Series(np.nan, index=wave.index))
wave["short_path"] = wave["path"].apply(lambda p: str(p).split("/wavedata/", 1)[-1] if pd.notna(p) else "")


# ── 6. Forensic per-run CSV ───────────────────────────────────────────────────
run_cols = [
    "short_path", "file_date", "category", "Mooring", "PanelCondition", "WindCondition",
    "WaveFrequencyInput [Hz]", "WaveAmplitudeInput [Volt]",
    "per_train", "mstop_sec", "run_num",
    "Kt_FFT", "Kt_LS", "Kt_PSD", "Kt_eff", "override_fired",
    "quality_flag", "cut_in_any", "cut_out_any",
    "fft_ls_dis", "ws_in_low", "any_flag",
]
run_cols = [c for c in run_cols if c in wave.columns]
wave[run_cols].sort_values(
    ["category", "WaveFrequencyInput [Hz]", "WaveAmplitudeInput [Volt]", "WindCondition", "file_date"]
).to_csv(OUT_RUNS, index=False)
print(f"\n   Forensic per-run table → {OUT_RUNS.relative_to(BASE)}  ({len(wave)} rows)")


# ── 7. Per-cell ΔK_t with bootstrap CI + Welch's t ───────────────────────────
def _boot_diff(a, b):
    if len(a) < 1 or len(b) < 1:
        return np.nan, np.nan, np.nan
    a = np.asarray(a, float); b = np.asarray(b, float)
    diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        diffs[i] = rng.choice(a, size=len(a), replace=True).mean() - rng.choice(b, size=len(b), replace=True).mean()
    return float(diffs.mean()), float(np.percentile(diffs, 100*ALPHA/2)), float(np.percentile(diffs, 100*(1-ALPHA/2)))

def _welch(a, b):
    if len(a) < 2 or len(b) < 2: return np.nan
    return float(scistats.ttest_ind(a, b, equal_var=False, nan_policy="omit").pvalue)

def _verdict(lo, hi):
    if not np.isfinite(lo) or not np.isfinite(hi): return "insufficient_n"
    if hi < 0: return "decrease"
    if lo > 0: return "enhance"
    return "null"

records = []
for cat in CAT_ORDER_PLOT:
    gcat = wave[wave["category"] == cat]
    for freq in FREQS_ALL:
        for amp in AMPS:
            cell = gcat[(gcat["freq_r"] == freq) & (gcat["amp_r"] == amp)]
            full = cell[cell["WindCondition"] == "full"]
            no_  = cell[cell["WindCondition"] == "no"]
            for src_label, src_col in [("FFT", "Kt_FFT"), ("eff", "Kt_eff")]:
                f_arr = full[src_col].dropna().to_numpy()
                n_arr = no_[src_col].dropna().to_numpy()
                d_m, d_lo, d_hi = _boot_diff(f_arr, n_arr)
                records.append({
                    "category": cat, "freq": freq, "amp": amp, "Kt_source": src_label,
                    "n_full": len(f_arr), "n_no": len(n_arr),
                    "Kt_mean_full": float(np.nanmean(f_arr)) if len(f_arr) else np.nan,
                    "Kt_mean_no":   float(np.nanmean(n_arr)) if len(n_arr) else np.nan,
                    "Kt_std_full":  float(np.nanstd(f_arr, ddof=1)) if len(f_arr) > 1 else np.nan,
                    "Kt_std_no":    float(np.nanstd(n_arr, ddof=1)) if len(n_arr) > 1 else np.nan,
                    "dKt_mean": d_m, "dKt_ci_lo": d_lo, "dKt_ci_hi": d_hi,
                    "welch_p":  _welch(f_arr, n_arr),
                    "verdict":  _verdict(d_lo, d_hi),
                })
cells = pd.DataFrame.from_records(records)
cells.to_csv(OUT_CELLS, index=False)
print(f"   Per-cell ΔK_t stats → {OUT_CELLS.relative_to(BASE)}  ({len(cells)} cells)")


# ── 8. Headline verdict at low freq (1.2-1.4 Hz), Kt source = eff ────────────
print("\n3. Verdict summary (Kt source = eff):")
for cat in CAT_ORDER_PLOT:
    sub = cells[(cells["category"] == cat) & (cells["Kt_source"] == "eff")]
    if sub.empty: continue
    print(f"\n   {CAT_LABELS[cat]}:")
    for _, r in sub.sort_values(["freq", "amp"]).iterrows():
        d_str  = "    -- " if pd.isna(r["dKt_mean"]) else f"{r['dKt_mean']:+.3f}"
        ci_str = "[--, --]" if pd.isna(r["dKt_ci_lo"]) else f"[{r['dKt_ci_lo']:+.3f}, {r['dKt_ci_hi']:+.3f}]"
        p_str  = "--" if pd.isna(r["welch_p"]) else f"{r['welch_p']:.3f}"
        print(f"     f={r['freq']:.1f}Hz A={r['amp']:.1f}V  "
              f"n={int(r['n_full'])}/{int(r['n_no'])}  "
              f"ΔKt={d_str} {ci_str}  p={p_str}  → {r['verdict']}")


# ── 9. Figure 1 — ΔK_t vs freq, 4-way colored, faceted by cat × amp ──────────
print("\n4. Building figures …")

def _make_delta_fig(cells_df, out_pdf, src_label="eff", src_title="eff (FFT + LS-override)"):
    fig, axes = plt.subplots(
        nrows=len(CAT_ORDER_PLOT), ncols=len(AMPS),
        figsize=(11, 12), sharex=True, sharey=True,
    )
    for r, cat in enumerate(CAT_ORDER_PLOT):
        sub_cat = cells_df[(cells_df["category"] == cat) & (cells_df["Kt_source"] == src_label)]
        color = COLOR_CAT[cat]
        for c, amp in enumerate(AMPS):
            ax = axes[r, c]
            sub = sub_cat[sub_cat["amp"] == amp].sort_values("freq")
            marker = MARKERS[amp]
            for _, row in sub.iterrows():
                if pd.isna(row["dKt_mean"]): continue
                is_support = row["freq"] in FREQS_SUPPORT
                alpha_pt = 0.40 if is_support else 0.95
                ms       = 8    if is_support else 11
                lo = row["dKt_ci_lo"]; hi = row["dKt_ci_hi"]
                ax.errorbar(
                    row["freq"], row["dKt_mean"],
                    yerr=[[row["dKt_mean"] - lo], [hi - row["dKt_mean"]]],
                    fmt=marker, color=color, ecolor=color,
                    markeredgecolor="black", markeredgewidth=0.5,
                    markersize=ms, lw=1.8, alpha=alpha_pt, capsize=4,
                )
                ax.text(row["freq"], hi + 0.012,
                        f"{int(row['n_full'])}/{int(row['n_no'])}",
                        fontsize=6.5, ha="center", va="bottom", color="black", alpha=0.7)
                if not pd.isna(row["welch_p"]) and row["welch_p"] < 0.05:
                    ax.text(row["freq"], lo - 0.018,
                            f"p={row['welch_p']:.3f}*",
                            fontsize=6.5, ha="center", va="top",
                            color="darkred", alpha=0.9)
            ax.axhline(0.0, color="black", lw=0.7, ls="--", alpha=0.6)
            ax.axvspan(1.15, 1.45, color="#ffe599", alpha=0.18, lw=0)
            if r == 0: ax.set_title(f"A = {amp:.1f} V", fontsize=10)
            if c == 0:
                lbl = CAT_LABELS[cat].replace(" (semi-stiff)", "\n(semi-stiff)").replace(" + revers (Nov)", "\n+ revers (Nov)")
                ax.set_ylabel(f"{lbl}\n$\\Delta K_t$", fontsize=8)
            ax.set_xticks(list(FREQS_ALL))
            ax.set_xticklabels([f"{f:.1f}" for f in FREQS_ALL], fontsize=8)
            ax.grid(alpha=0.3, lw=0.5)
    for c in range(len(AMPS)):
        axes[-1, c].set_xlabel("Frekvens (Hz)", fontsize=9)
    fig.suptitle(
        f"Wind effect on K_t  ($\\Delta K_t$ = $K_t^{{full}} - K_t^{{no}}$)  — bootstrap 95% CI, source: {src_title}\n"
        f"4-way mooring split (color = mooring, marker = amp).  Yellow band = lavfrekvens-fokus (1.2-1.4 Hz).  "
        f"Numbers = n_full/n_no.  Asterisk = Welch p < 0.05.",
        fontsize=10, y=0.995,
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.93)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.03)
    print(f"   Saved → {out_pdf.relative_to(BASE)}")
    plt.close(fig)

_make_delta_fig(cells, OUT_PDF_DELTA, src_label="eff", src_title="eff (FFT with LS override)")


# ── 10. Figure 2 — K_t full vs no with 4-way wind colors and amp markers ─────
def _make_kt_fig(wave_df, out_pdf, src_col="Kt_eff"):
    fig, axes = plt.subplots(
        nrows=len(CAT_ORDER_PLOT), ncols=len(AMPS),
        figsize=(11, 12), sharex=True, sharey=True,
    )
    for r, cat in enumerate(CAT_ORDER_PLOT):
        gcat = wave_df[wave_df["category"] == cat]
        for c, amp in enumerate(AMPS):
            ax = axes[r, c]
            sub = gcat[gcat["amp_r"] == amp]
            marker = MARKERS[amp]
            for wind in ("no", "full"):
                w = sub[sub["WindCondition"] == wind]
                if w.empty: continue
                color = COLORS_WIND[(cat, wind)]
                # individual run dots
                ax.scatter(
                    w["freq_r"] + (0.02 if wind == "full" else -0.02),
                    w[src_col],
                    marker=marker, s=24,
                    facecolor=color, edgecolor="black",
                    alpha=0.55, linewidths=0.5, zorder=2,
                )
                agg = (w.groupby("freq_r")[src_col]
                          .agg(["mean", "std", "count"]).reset_index())
                for _, ar in agg.iterrows():
                    if ar["count"] >= 2 and pd.notna(ar["std"]):
                        ax.errorbar(
                            ar["freq_r"] + (0.02 if wind == "full" else -0.02),
                            ar["mean"], yerr=ar["std"],
                            fmt=marker, color=color, mec="black", mew=0.5,
                            ms=10, lw=1.5, alpha=0.95, capsize=3, zorder=3,
                        )
            ax.set_xticks(list(FREQS_ALL))
            ax.set_xticklabels([f"{f:.1f}" for f in FREQS_ALL], fontsize=8)
            ax.set_ylim(0.2, 1.0)
            ax.grid(alpha=0.3, lw=0.5)
            ax.axvspan(1.15, 1.45, color="#ffe599", alpha=0.18, lw=0)
            if r == 0: ax.set_title(f"A = {amp:.1f} V", fontsize=10)
            if c == 0:
                lbl = CAT_LABELS[cat].replace(" (semi-stiff)", "\n(semi-stiff)").replace(" + revers (Nov)", "\n+ revers (Nov)")
                ax.set_ylabel(f"{lbl}\n$K_t$", fontsize=8)
    for c in range(len(AMPS)):
        axes[-1, c].set_xlabel("Frekvens (Hz)", fontsize=9)
    fig.suptitle(
        "$K_t$ vs frekvens — 4-way mooring split, color = (mooring × wind), marker = amp.  "
        "Yellow band = lavfrekvens-fokus (1.2-1.4 Hz).",
        fontsize=10, y=0.995,
    )
    fig.tight_layout(); fig.subplots_adjust(top=0.94)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.03)
    print(f"   Saved → {out_pdf.relative_to(BASE)}")
    plt.close(fig)

_make_kt_fig(wave, OUT_PDF_KT, src_col="Kt_eff")


# ── 11. A2 deep-dive ─────────────────────────────────────────────────────────
# Goal: is "A2 weird" an above_semi-stiff phenomenon or a universal pattern?
#   - Plot ΔK_t(amp) per (cat × freq) — connect A1 → A2 → A3.
#   - If A2 dips below A1 and A3 universally → real physical pattern at A2.
#   - If only at canon_above_loose → mooring-specific or session artefact.
print("\n5. A2 deep-dive — does A2 anomaly appear in canon (under-mooring) too?")

# Compact table per (cat × freq × amp): ΔK_t_eff with CI, n
a2_pivot = []
for cat in CAT_ORDER_PLOT:
    for freq in FREQS_ALL:
        for amp in AMPS:
            r = cells[(cells["category"] == cat) & (cells["freq"] == freq)
                      & (cells["amp"] == amp) & (cells["Kt_source"] == "eff")].iloc[0]
            a2_pivot.append(dict(category=cat, freq=freq, amp=amp,
                                  dKt=r["dKt_mean"], lo=r["dKt_ci_lo"], hi=r["dKt_ci_hi"],
                                  n_full=r["n_full"], n_no=r["n_no"]))
a2_df = pd.DataFrame(a2_pivot)

# Figure 3: ΔK_t(amp) curves per freq, 4-way colored, w/ CI errorbars
fig, axes = plt.subplots(nrows=1, ncols=len(FREQS_ALL),
                         figsize=(14, 5), sharey=True)
for c, freq in enumerate(FREQS_ALL):
    ax = axes[c]
    for cat in CAT_ORDER_PLOT:
        sub = a2_df[(a2_df["category"] == cat) & (a2_df["freq"] == freq)].sort_values("amp")
        valid = sub[sub["dKt"].notna()]
        if valid.empty: continue
        color = COLOR_CAT[cat]
        ax.plot(valid["amp"], valid["dKt"], "-", color=color, lw=1.2, alpha=0.75)
        for _, r in valid.iterrows():
            ax.errorbar(r["amp"], r["dKt"],
                        yerr=[[r["dKt"] - r["lo"]], [r["hi"] - r["dKt"]]],
                        fmt=MARKERS[r["amp"]], color=color, ecolor=color,
                        markeredgecolor="black", markeredgewidth=0.5,
                        markersize=10, lw=1.5, alpha=0.9, capsize=4)
    ax.axhline(0, color="black", lw=0.6, ls="--", alpha=0.6)
    ax.set_xticks([0.10, 0.20, 0.30])
    ax.set_xticklabels(["A1\n0.1V", "A2\n0.2V", "A3\n0.3V"], fontsize=9)
    ax.set_title(f"f = {freq:.1f} Hz", fontsize=10)
    if c == 0: ax.set_ylabel(r"$\Delta K_t$ (full $-$ no)", fontsize=11)
    ax.grid(alpha=0.3, lw=0.5)
    # highlight A2 with a soft yellow column
    ax.axvspan(0.175, 0.225, color="#ffe599", alpha=0.25, lw=0, zorder=0)

# Legend on the rightmost panel
legend_handles = [
    mlines.Line2D([], [], marker="o", linestyle="-", color=COLOR_CAT[cat],
                  markerfacecolor=COLOR_CAT[cat], markeredgecolor="black",
                  markeredgewidth=0.4, markersize=8, lw=1.5,
                  label=CAT_LABELS[cat])
    for cat in CAT_ORDER_PLOT
]
axes[-1].legend(handles=legend_handles, loc="upper right",
                fontsize=8, framealpha=0.92, title="Forankring", title_fontsize=9)

fig.suptitle(
    r"A2 deep-dive: $\Delta K_t$(amp) for each mooring × freq.  "
    r"If A2 dips below A1 and A3 only at 'Over, 16 cm' it's mooring-specific; "
    r"if it dips everywhere it's a universal A2 effect.",
    fontsize=10, y=1.00,
)
fig.tight_layout(); fig.subplots_adjust(top=0.90)
fig.savefig(OUT_PDF_A2, bbox_inches="tight", pad_inches=0.03)
print(f"   Saved → {OUT_PDF_A2.relative_to(BASE)}")
plt.close(fig)


# ── 12. A2 forensic per-run CSV — for the user to inspect dates/sessions ────
a2_runs = wave[wave["amp_r"] == 0.20].copy()
a2_runs = a2_runs[a2_runs["freq_r"].isin([1.3, 1.4, 1.5])]  # densest A2 cells
a2_cols = [
    "category", "WaveFrequencyInput [Hz]", "WindCondition",
    "file_date", "per_train", "mstop_sec", "run_num",
    "Kt_FFT", "Kt_LS", "Kt_PSD", "Kt_eff",
    "any_flag", "cut_in_any", "cut_out_any", "qflag_nonok", "fft_ls_dis", "ws_in_low",
    "IN wave_stability", "OUT wave_stability",
    "short_path",
]
a2_cols = [c for c in a2_cols if c in a2_runs.columns]
a2_runs[a2_cols].sort_values(
    ["category", "WaveFrequencyInput [Hz]", "WindCondition", "file_date"]
).to_csv(OUT_A2_RUNS, index=False)
print(f"   A2 per-run table → {OUT_A2_RUNS.relative_to(BASE)}  ({len(a2_runs)} rows)")


# ── 13. Print the A2 above_semi-stiff 1.4 Hz cell — the worst-offender cell ──
print("\n6. A2 above_semi-stiff 1.4 Hz — the cell that triggered the question:")
focus = wave[(wave["category"] == "canon_above_loose")
             & (wave["amp_r"] == 0.20)
             & (wave["freq_r"] == 1.4)]
if focus.empty:
    print("   (no rows — check the filter)")
else:
    cols_print = ["WindCondition", "file_date", "per_train", "mstop_sec",
                  "Kt_FFT", "Kt_LS", "Kt_PSD", "any_flag", "short_path"]
    cols_print = [c for c in cols_print if c in focus.columns]
    print(focus[cols_print].sort_values(["WindCondition", "file_date"]).to_string(index=False))

# Also print A2 at 1.3 Hz across all 4 moorings — does A2 "dip" appear in canon under-mooring too?
print("\n7. A2 dip cross-check at 1.3 Hz — per mooring:")
ck = cells[(cells["amp"] == 0.20) & (cells["freq"] == 1.3) & (cells["Kt_source"] == "eff")]
print(ck[["category", "n_full", "n_no", "Kt_mean_full", "Kt_mean_no",
           "dKt_mean", "dKt_ci_lo", "dKt_ci_hi", "welch_p", "verdict"]].to_string(index=False))

print("\n8. Compare A1 vs A2 vs A3 at 1.3 Hz per mooring (ΔK_t_eff with CI):")
for cat in CAT_ORDER_PLOT:
    sub = cells[(cells["category"] == cat) & (cells["freq"] == 1.3)
                 & (cells["Kt_source"] == "eff")].sort_values("amp")
    print(f"\n   {CAT_LABELS[cat]}:")
    for _, r in sub.iterrows():
        d_str  = "    -- " if pd.isna(r["dKt_mean"]) else f"{r['dKt_mean']:+.3f}"
        ci_str = "[--, --]" if pd.isna(r["dKt_ci_lo"]) else f"[{r['dKt_ci_lo']:+.3f}, {r['dKt_ci_hi']:+.3f}]"
        p_str  = "--" if pd.isna(r["welch_p"]) else f"{r['welch_p']:.3f}"
        print(f"     A={r['amp']:.1f}V  n={int(r['n_full'])}/{int(r['n_no'])}  ΔKt={d_str} {ci_str}  p={p_str}")

print("\nDone.")
