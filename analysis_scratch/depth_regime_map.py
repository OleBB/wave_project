"""
CH04 §3f — Depth-regime map (kd vs f) at d = 0.58 m.

Two-panel figure justifying the full dispersion relation ω² = gk·tanh(kd)
for this tank.

    Top panel — kd vs frequency at d = 580 mm, with shaded regime bands
        (deep / intermediate / shallow), horizontal reference lines at
        kd = π and kd = π/10, run-frequency markers sized by n_runs and
        coded by regime (colour + marker shape), vertical band marking
        the thesis scope (1.3–1.6 Hz), and a secondary right-hand axis
        in wavelength λ (metres).

    Bottom panel — relative error in λ if the deep-water approximation
        (λ_deep = g/(2π f²)) were used instead of the full dispersion,
        shown as % over the same frequency axis. Quantifies "by how much
        does the full dispersion matter".

Data: unique WaveFrequencyInput [Hz] and n_runs from combined_meta across
ALL processed datasets (methodology figure — uses combined_meta, not
meta_results).

Outputs:
    output/FIGURES/ch04_depth_regime.pdf
    output/TEXFIGU/ch04_depth_regime.tex  (CAPTIONS-dict driven, force=True)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import font_manager as _fm
from matplotlib.ticker import MultipleLocator

from wavescripts.improved_data_loader import load_analysis_data
from wavescripts.plot_utils import apply_thesis_style, freq_to_k
import wavescripts.plot_utils as pu


BASE        = Path("/Users/ole/Kodevik/wave_project")
FIGURES_DIR = BASE / "output" / "FIGURES"
TEXFIGU_DIR = BASE / "output" / "TEXFIGU"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TEXFIGU_DIR.mkdir(parents=True, exist_ok=True)

# All processed folders — this is a methodology figure, so everything.
ALL_PROCESSED_DIRS = sorted((BASE / "waveprocessed").glob("PROCESSED-*"))

# ══════════════════════════════════════════════════════════════════════════
# USER-AUTHORED CAPTION — edit this; the stub body uses it verbatim.
# ══════════════════════════════════════════════════════════════════════════
# Empty → body gets a TODO placeholder. Non-empty → \caption[...]{...} as-is.
# force=True, so the body always reflects this variable on regeneration.
CAPTION = ""


# ─── Physics constants ────────────────────────────────────────────────────
DEPTH_M  = 0.580      # d = 580 mm (every run in this dataset)
G        = 9.81       # m/s²
F_MIN    = 0.50       # Hz — x-axis lower bound
F_MAX    = 2.50       # Hz — x-axis upper bound
THESIS_F_LO = 1.30    # Hz — thesis scope lower
THESIS_F_HI = 1.60    # Hz — thesis scope upper

# Regime thresholds (standard water-wave convention)
KD_DEEP_BOUNDARY     = np.pi          # kd > π  → deep
KD_SHALLOW_BOUNDARY  = np.pi / 10.0   # kd < π/10 → shallow

# Palette for regime markers — distinct from thesis red/blue/magenta/turquoise.
REGIME_COLOR = {
    "deep":         "#546E7A",   # steel grey
    "intermediate": "#D4940E",   # amber
    "shallow":      "#8E24AA",   # purple
}
REGIME_MARKER = {
    "deep":         "o",
    "intermediate": "s",
    "shallow":      "^",
}


# ─── NCM font ─────────────────────────────────────────────────────────────
apply_thesis_style()
_NCM_DIR = "/usr/local/texlive/2025/texmf-dist/fonts/opentype/public/newcomputermodern"
for _fname in ("NewCM10-Regular.otf", "NewCM10-Bold.otf",
               "NewCM10-Italic.otf", "NewCM10-BoldItalic.otf",
               "NewCMMath-Regular.otf"):
    try:
        _fm.fontManager.addfont(f"{_NCM_DIR}/{_fname}")
    except Exception as _e:
        print(f"   warn: could not register {_fname}: {_e}")
plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["NewComputerModern10", "DejaVu Serif"],
    "mathtext.fontset": "cm",
})
pu.ACTIVE_DATASETS = [p.name for p in ALL_PROCESSED_DIRS]
pu.TEXFIGU_DIR = TEXFIGU_DIR
pu.FIGURES_DIR = FIGURES_DIR


# ─── Load + tally runs per frequency ──────────────────────────────────────
print(f"Loading combined_meta from {len(ALL_PROCESSED_DIRS)} folder(s) …")
meta, _, _, _ = load_analysis_data(*[str(p) for p in ALL_PROCESSED_DIRS],
                                   load_processed=False)

# Wave runs only (exclude nowave/nowind diagnostics etc.)
wave = meta[meta["WaveFrequencyInput [Hz]"].notna()].copy()
wave = wave[wave["WaveFrequencyInput [Hz]"] > 0]

freq_counts = wave.groupby("WaveFrequencyInput [Hz]").size().rename("n_runs")
print(f"  {len(wave)} wave runs across {len(freq_counts)} unique frequencies")


def regime_from_kd(kd):
    if kd > KD_DEEP_BOUNDARY:
        return "deep"
    elif kd < KD_SHALLOW_BOUNDARY:
        return "shallow"
    return "intermediate"


# Compute kd for each observed frequency
run_df = freq_counts.reset_index()
run_df["k_full"]  = freq_to_k(run_df["WaveFrequencyInput [Hz]"].to_numpy(),
                              depth_mm=DEPTH_M * 1000.0)
run_df["kd"]      = run_df["k_full"] * DEPTH_M
run_df["regime"]  = run_df["kd"].apply(regime_from_kd)
run_df["lambda_full_m"] = 2 * np.pi / run_df["k_full"]

# Deep-water approximation: k_deep = ω²/g
omega = 2 * np.pi * run_df["WaveFrequencyInput [Hz]"].to_numpy()
run_df["k_deep"] = omega**2 / G
run_df["lambda_deep_m"] = 2 * np.pi / run_df["k_deep"]
# Relative error in λ from using the deep-water approximation
run_df["lambda_rel_err_pct"] = (run_df["lambda_deep_m"] - run_df["lambda_full_m"]) \
                                / run_df["lambda_full_m"] * 100.0

print("\nObserved frequencies and their regimes:")
for _, r in run_df.iterrows():
    print(f"  f={r['WaveFrequencyInput [Hz]']:.2f} Hz  "
          f"k={r['k_full']:.3f}  kd={r['kd']:.3f}  "
          f"λ={r['lambda_full_m']:.3f} m  "
          f"regime={r['regime']}  n_runs={r['n_runs']}  "
          f"λ-error (deep approx)={r['lambda_rel_err_pct']:.2f}%")


# ─── Continuous curve on fine frequency grid ──────────────────────────────
f_grid    = np.linspace(F_MIN, F_MAX, 400)
k_grid    = freq_to_k(f_grid, depth_mm=DEPTH_M * 1000.0)
kd_grid   = k_grid * DEPTH_M
lam_full  = 2 * np.pi / k_grid
lam_deep  = G / (2 * np.pi * f_grid**2)
lam_err   = (lam_deep - lam_full) / lam_full * 100.0


# ─── Figure ───────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(8, 8.2))
gs  = fig.add_gridspec(2, 1, height_ratios=[2.3, 1.0], hspace=0.12)
ax_top = fig.add_subplot(gs[0, 0])
ax_bot = fig.add_subplot(gs[1, 0], sharex=ax_top)


# ─── Top panel: kd vs f ──
# Regime bands (horizontal shaded zones by kd value)
ax_top.axhspan(KD_DEEP_BOUNDARY, kd_grid.max() * 1.2,
               color=REGIME_COLOR["deep"], alpha=0.09, zorder=0)
ax_top.axhspan(KD_SHALLOW_BOUNDARY, KD_DEEP_BOUNDARY,
               color=REGIME_COLOR["intermediate"], alpha=0.09, zorder=0)
ax_top.axhspan(0, KD_SHALLOW_BOUNDARY,
               color=REGIME_COLOR["shallow"], alpha=0.09, zorder=0)

# Regime band labels (inside the coloured strips, at the right edge)
ax_top.text(F_MAX - 0.04, (KD_DEEP_BOUNDARY + kd_grid.max() * 1.1) / 2,
            "Dypt vann ($kd > \\pi$)",
            ha="right", va="center", fontsize=9,
            color=REGIME_COLOR["deep"], weight="bold", alpha=0.8)
ax_top.text(F_MAX - 0.04, (KD_SHALLOW_BOUNDARY + KD_DEEP_BOUNDARY) / 2,
            "Mellomdypt ($\\pi/10 < kd < \\pi$)",
            ha="right", va="center", fontsize=9,
            color=REGIME_COLOR["intermediate"], weight="bold", alpha=0.8)
ax_top.text(F_MAX - 0.04, KD_SHALLOW_BOUNDARY / 2,
            "Grunnvann ($kd < \\pi/10$)",
            ha="right", va="center", fontsize=8,
            color=REGIME_COLOR["shallow"], weight="bold", alpha=0.8)

# Regime boundary dashed lines
ax_top.axhline(KD_DEEP_BOUNDARY, color="#444", lw=0.7, ls=":",
               alpha=0.7, zorder=1)
ax_top.axhline(KD_SHALLOW_BOUNDARY, color="#444", lw=0.7, ls=":",
               alpha=0.7, zorder=1)

# Thesis scope vertical band
ax_top.axvspan(THESIS_F_LO, THESIS_F_HI,
               color="#888", alpha=0.10, zorder=0)

# kd(f) curve
ax_top.plot(f_grid, kd_grid, color="black", lw=1.3, zorder=2)

# Run markers — sized by n_runs, coloured+shaped by regime
for regime, sub in run_df.groupby("regime"):
    sizes = 30 + sub["n_runs"].to_numpy() * 1.8
    ax_top.scatter(
        sub["WaveFrequencyInput [Hz]"], sub["kd"],
        s=sizes,
        marker=REGIME_MARKER[regime],
        color=REGIME_COLOR[regime],
        edgecolors="black", linewidths=0.45,
        alpha=0.88, zorder=3,
        label=f"{regime} ({int(sub['n_runs'].sum())} runs)",
    )

# Annotate sub-1 Hz bottom-motion observation from CLAUDE.md §19
_low = run_df[run_df["WaveFrequencyInput [Hz]"] < 0.75]
if not _low.empty:
    _ann_x = float(_low["WaveFrequencyInput [Hz]"].iloc[0])
    _ann_y = float(_low["kd"].iloc[0])
    ax_top.annotate(
        "bunnbevegelse observert\n(2026-03-12)",
        xy=(_ann_x, _ann_y),
        xytext=(_ann_x + 0.15, _ann_y + 1.3),
        fontsize=8, color="#444",
        arrowprops=dict(arrowstyle="->", color="#888", lw=0.6),
        bbox=dict(boxstyle="square,pad=0.2", fc="white",
                  ec="#888", lw=0.4, alpha=0.92),
        zorder=4,
    )

ax_top.set_xlim(F_MIN, F_MAX)
# y-lo slightly > 0 — the secondary λ-axis diverges at kd=0 and crashes
# matplotlib's apply_aspect if the lower limit is exactly 0.
ax_top.set_ylim(0.05, kd_grid.max() * 1.08)
ax_top.set_ylabel("$kd$  (dimensionless)", fontsize=10)
ax_top.xaxis.set_major_locator(MultipleLocator(0.5))
ax_top.xaxis.set_minor_locator(MultipleLocator(0.1))
ax_top.yaxis.set_major_locator(MultipleLocator(2.0))
ax_top.yaxis.set_minor_locator(MultipleLocator(0.5))
ax_top.grid(True, which="major", alpha=0.30)
ax_top.grid(True, which="minor", alpha=0.12)
ax_top.tick_params(axis="x", labelbottom=False)  # suppress — bot panel has ticks
ax_top.legend(loc="lower right", fontsize=8, framealpha=0.92,
              title="Regime (markers)", title_fontsize=8)

# Secondary right-hand y-axis: wavelength in metres
def _kd_to_lambda(kd):
    kd = np.asarray(kd, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(kd > 0, 2 * np.pi * DEPTH_M / kd, np.nan)
def _lambda_to_kd(lam):
    lam = np.asarray(lam, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(lam > 0, 2 * np.pi * DEPTH_M / lam, np.nan)
sec = ax_top.secondary_yaxis("right", functions=(_kd_to_lambda, _lambda_to_kd))
sec.set_ylabel(r"bølgelengde $\lambda$ (m)", fontsize=9)
sec.tick_params(labelsize=8)


# ─── Bottom panel: λ error (%) from deep-water approximation ──
ax_bot.plot(f_grid, lam_err, color="black", lw=1.3, zorder=2)
ax_bot.axvspan(THESIS_F_LO, THESIS_F_HI, color="#888", alpha=0.10, zorder=0)
ax_bot.axhline(0, color="#888", lw=0.5, alpha=0.6)

# Reference thresholds
for ref_pct, label in [(1.0, "1%"), (5.0, "5%"), (20.0, "20%")]:
    ax_bot.axhline(ref_pct, color="#bbb", lw=0.5, ls=":", alpha=0.7)
    ax_bot.text(F_MAX - 0.02, ref_pct,
                f" {label}", ha="right", va="bottom",
                fontsize=7, color="#666")

# Run markers on the error curve too (so the reader sees the spread of errors)
for regime, sub in run_df.groupby("regime"):
    sizes = 20 + sub["n_runs"].to_numpy() * 1.2
    ax_bot.scatter(
        sub["WaveFrequencyInput [Hz]"], sub["lambda_rel_err_pct"],
        s=sizes,
        marker=REGIME_MARKER[regime],
        color=REGIME_COLOR[regime],
        edgecolors="black", linewidths=0.4,
        alpha=0.85, zorder=3,
    )

ax_bot.set_xlim(F_MIN, F_MAX)
ax_bot.set_yscale("symlog", linthresh=1.0)
ax_bot.set_ylim(0, max(lam_err.max() * 1.3, 100))
ax_bot.set_xlabel("paddle-frekvens $f$ (Hz)", fontsize=10)
ax_bot.set_ylabel(r"feil i $\lambda$ ved dyptvanns-" + "\napproksimasjon (%, symlog)",
                  fontsize=9)
ax_bot.xaxis.set_major_locator(MultipleLocator(0.5))
ax_bot.xaxis.set_minor_locator(MultipleLocator(0.1))
ax_bot.grid(True, which="major", alpha=0.30)
ax_bot.grid(True, which="minor", alpha=0.12)


fig.tight_layout()

out_pdf = FIGURES_DIR / "ch04_depth_regime.pdf"
# out_pgf = FIGURES_DIR / "ch04_depth_regime.pgf"
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_pgf, bbox_inches="tight")
plt.close(fig)
print(f"\n   → {out_pdf.relative_to(BASE)}")


# ─── TEXFIGU stub ─────────────────────────────────────────────────────────
def _f(x, n=3):
    try: return f"{float(x):.{n}f}"
    except Exception: return "NA"

extra_stats = {
    "water_depth_m":      _f(DEPTH_M, 3),
    "n_runs_total":       f"{int(freq_counts.sum())}",
    "n_unique_freqs":     f"{len(freq_counts)}",
    "thesis_scope_f_lo_Hz": _f(THESIS_F_LO, 2),
    "thesis_scope_f_hi_Hz": _f(THESIS_F_HI, 2),
    "kd_thresh_deep_pi":  _f(KD_DEEP_BOUNDARY, 3),
    "kd_thresh_shallow":  _f(KD_SHALLOW_BOUNDARY, 3),
}
# Per-regime run counts
for regime, sub in run_df.groupby("regime"):
    extra_stats[f"n_runs_{regime}"] = f"{int(sub['n_runs'].sum())}"
    extra_stats[f"n_freqs_{regime}"] = f"{len(sub)}"
    extra_stats[f"kd_range_{regime}"] = f"[{sub['kd'].min():.2f}, {sub['kd'].max():.2f}]"
# Per-frequency readouts
for _, r in run_df.iterrows():
    tag = f"f{r['WaveFrequencyInput [Hz]']:.2f}Hz".replace(".", "p")
    extra_stats[f"kd_at_{tag}"]            = _f(r["kd"], 3)
    extra_stats[f"lambda_m_at_{tag}"]      = _f(r["lambda_full_m"], 3)
    extra_stats[f"lambda_err_pct_{tag}"]   = _f(r["lambda_rel_err_pct"], 2)
    extra_stats[f"n_runs_at_{tag}"]        = f"{int(r['n_runs'])}"
    extra_stats[f"regime_at_{tag}"]        = r["regime"]

_meta = pu.build_fig_meta(
    {
        "filters": {
            "PanelCondition":   None,   # methodology — all panels
            "WindCondition":    None,
            "quality_flag":     None,
        },
        "plotting": {
            "figure_name": "ch04_depth_regime",
            "caption":     CAPTION,
        },
    },
    chapter="04",
    extra={"script": "analysis_scratch/depth_regime_map.py"},
    computed_in=("analysis_scratch/depth_regime_map.py "
                 "(single-depth kd vs f map + deep-water approximation error)"),
    data_class="DELEG",
    findings_doc="CLAUDE.md §19 (water depth regime — TODO noted there)",
    extra_params=(
        f"water_depth=d={DEPTH_M} m (every run, filename tag 'depth580'). "
        f"x-axis range = {F_MIN}-{F_MAX} Hz; grid = 400 points. "
        f"Regime thresholds: deep (kd > π = {KD_DEEP_BOUNDARY:.3f}), "
        f"intermediate (π/10 < kd < π), "
        f"shallow (kd < π/10 = {KD_SHALLOW_BOUNDARY:.3f}). "
        f"Top panel uses the full dispersion relation ω² = g·k·tanh(kd). "
        f"k computed via wavescripts.plot_utils.freq_to_k (iterative solver "
        f"in calculate_wavenumbers_vectorized). Right-axis (top panel) shows "
        f"wavelength λ = 2π/k in metres via secondary_yaxis. Bottom panel "
        f"shows relative error in λ if the deep-water approximation "
        f"λ_deep = g/(2π f²) were used instead, plotted on symlog y-axis "
        f"(linear below 1%). Run markers at every unique WaveFrequencyInput "
        f"in combined_meta across all {len(ALL_PROCESSED_DIRS)} PROCESSED-* "
        f"folders, sized by n_runs at that frequency. Regime encoded by "
        f"BOTH colour and marker shape: deep={REGIME_COLOR['deep']} circle, "
        f"intermediate={REGIME_COLOR['intermediate']} square, "
        f"shallow={REGIME_COLOR['shallow']} triangle. Vertical grey band "
        f"marks thesis scope 1.3–1.6 Hz. Low-freq annotation references "
        f"bottom-motion observation logged in CLAUDE.md §19 (2026-03-12). "
        f"Typeset in NewComputerModern10."
    ),
    extra_stats=extra_stats,
)

pu.write_figure_stub(_meta, plot_type="depth_regime",
                     subfig_filenames=["ch04_depth_regime"],
                     force=True)
print(f"   stub → output/TEXFIGU/ch04_depth_regime.tex")
print("\nDone.")
