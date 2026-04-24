"""
CH04 §4-3 — FFT amplitude spectrum at the paddle frequency.

Produces a 2×2 grid showing the discrete FFT spectrum of the surface
elevation at the IN and OUT probes for a canonical run (1.4 Hz, 0.2 V,
fullpanel, per240, canon 20260327 dataset). Each panel is one run+probe;
bars sit at the actual FFT bin centres. Rows = wind condition
(nowind, fullwind); columns = probe side (Inn, Ut).

Annotations on every panel:
  - Horizontal dotted guides at that row's IN and OUT paddle peaks
  - Vertical dotted guides at f, 2f, 3f, 4f (labels on the top axis of the
    top row)
  - Peak value [mm] next to each paddle bar

Utgående panels additionally carry a two-headed Δ arrow between the row's
IN and OUT guide lines, labeled with:
    Δ = A_inn − A_Ut  [mm]
    A_Ut / A_inn      [dimensionless — thesis OUT/IN ratio]

Outputs:
  output/FIGURES/ch04_fft_wave.pdf
  output/FIGURES/ch04_fft_wave.pgf
  output/TEXFIGU/ch04_fft_wave.tex    (force-written, single-includegraphics)

Type face: NewComputerModern10 (OTFs shipped with TeX Live's
newcomputermodern package) registered directly with matplotlib. Math is
drawn with the built-in CM mathtext set; visually identical to NCM for
the symbols used here.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm
from matplotlib.ticker import MultipleLocator

from wavescripts.improved_data_loader import load_analysis_data
from wavescripts.signal_processing import get_positive_spectrum
from wavescripts.plot_utils import apply_thesis_style, WIND_COLOR_MAP
import wavescripts.plot_utils as pu


# ─── Paths & constants ────────────────────────────────────────────────────
BASE        = Path("/Users/ole/Kodevik/wave_project")
FIGURES_DIR = BASE / "output" / "FIGURES"
TEXFIGU_DIR = BASE / "output" / "TEXFIGU"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TEXFIGU_DIR.mkdir(parents=True, exist_ok=True)

TARGET_DIR = BASE / "waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"
DATADIR    = BASE / "wavedata/20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"

RUNS = {
    "nowind":   DATADIR / "fullpanel-nowind-amp0200-freq1400-per240-depth580-mstop30-run1.csv",
    "fullwind": DATADIR / "fullpanel-fullwind-amp0200-freq1400-per240-depth580-mstop30-run1.csv",
}

IN_PROBE  = "9373/170"
OUT_PROBE = "12400/250"
FREQ      = 1.4
AMP_V     = 0.2
XMAX      = 6.0
FS        = 250.0

WIND_KEY    = {"nowind": "no", "fullwind": "full"}
WIND_COLOR  = {w: WIND_COLOR_MAP[WIND_KEY[w]] for w in ("nowind", "fullwind")}
WIND_LABEL  = {"nowind": "uten vind", "fullwind": "med vind"}
SIDE_LABEL  = {"IN": "Innkommende", "OUT": "Utgående"}

HARMONICS        = [(k, k * FREQ) for k in (1, 2, 3, 4)]
HARMONIC_TICKS   = [fk for _, fk in HARMONICS if fk <= XMAX]
HARMONIC_LABELS  = [r"$f$", r"$2f$", r"$3f$", r"$4f$"][:len(HARMONIC_TICKS)]


# ─── Register NewComputerModern OTFs ──────────────────────────────────────
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
    "text.usetex":      False,
    "font.family":      "serif",
    "font.serif":       ["NewComputerModern10", "DejaVu Serif"],
    "mathtext.fontset": "cm",
})

pu.ACTIVE_DATASETS = [p.name for p in
                     sorted(BASE.glob("waveprocessed/PROCESSED-*"))]
pu.TEXFIGU_DIR = TEXFIGU_DIR
pu.FIGURES_DIR = FIGURES_DIR


# ─── Load the one canon folder ────────────────────────────────────────────
print("Loading …")
meta, _, fft_dict, _ = load_analysis_data(str(TARGET_DIR), load_processed=False)


def get_spectrum(run_csv: Path, probe: str):
    """Return (f, A_one_sided_mm, bin_width_Hz) for this run+probe."""
    csv  = str(run_csv)
    dff  = fft_dict[csv]
    dpos = get_positive_spectrum(dff)
    y    = dpos[f"FFT {probe}"].dropna()
    f    = y.index.values
    mask = (f >= 0) & (f <= XMAX)
    f    = f[mask]
    A    = 2.0 * y.values[mask]          # one-sided amplitude (matches meta)
    bin_w = float(f[1] - f[0])           # real FFT bin spacing for this probe
    return f, A, bin_w


specs      = {}     # (wind, "IN"/"OUT") → (f, A, bin_w)
peaks      = {}     # (wind, "IN"/"OUT") → peak amplitude in mm
bin_widths = {}     # same keys → bin width in Hz

for wind, csv in RUNS.items():
    for side, probe in (("IN", IN_PROBE), ("OUT", OUT_PROBE)):
        f, A, bin_w = get_spectrum(csv, probe)
        specs[(wind, side)]      = (f, A, bin_w)
        peaks[(wind, side)]      = float(A.max())
        bin_widths[(wind, side)] = bin_w

print("\nPeaks & bin widths (verified against meta.json):")
for (wind, side), A_peak in peaks.items():
    probe   = IN_PROBE if side == "IN" else OUT_PROBE
    meta_A  = float(meta[meta["path"] == str(RUNS[wind])].iloc[0]
                    [f"Probe {probe} Amplitude (FFT)"])
    bin_w   = bin_widths[(wind, side)]
    print(f"  {wind:8s} {side:3s}  A_peak={A_peak:.4f}  meta={meta_A:.4f}  "
          f"bin_width={bin_w:.5f} Hz  (N={int(round(FS/bin_w))})")

# ka values (canonical per-side means from meta.json)
KA = {
    wind: {
        "IN":  float(meta[meta["path"] == str(RUNS[wind])].iloc[0]["IN ka (FFT)"]),
        "OUT": float(meta[meta["path"] == str(RUNS[wind])].iloc[0]["OUT ka (FFT)"]),
    }
    for wind in ("nowind", "fullwind")
}
print("\nka (from meta):")
for w, d in KA.items():
    print(f"  {w:8s}  ka_inn={d['IN']:.4f}  ka_Ut={d['OUT']:.4f}")

# Tight y-cap: bumped 1.5 % over the tallest overall peak.
YMAX_ALL = max(peaks.values()) * 1.015

# Per-row IN / OUT reference peaks — drives the horizontal guides and the Δ arrow.
ROW_PEAKS = {
    wind: {"IN": peaks[(wind, "IN")], "OUT": peaks[(wind, "OUT")]}
    for wind in ("nowind", "fullwind")
}


# ─── Panel ────────────────────────────────────────────────────────────────
def draw_panel(ax, *, wind, side, show_delta, label_frac=0.88):
    f, A, _bin_w = specs[(wind, side)]
    color  = WIND_COLOR[wind]
    row_in = ROW_PEAKS[wind]["IN"]
    row_ut = ROW_PEAKS[wind]["OUT"]
    df     = f[1] - f[0]

    ax.bar(f, A, width=df * 0.9, color=color, edgecolor=color, lw=0.4,
           alpha=0.85, align="center", zorder=3)

    ax.axhline(row_in, color="#222", lw=0.7, ls=":", alpha=0.8, zorder=2)
    ax.axhline(row_ut, color="#888", lw=0.7, ls=":", alpha=0.8, zorder=2)
    for _k, fk in HARMONICS:
        if fk <= XMAX:
            ax.axvline(fk, color="#888", lw=0.5, ls=":", alpha=0.7, zorder=1)

    peak_i = int(np.argmax(A))
    f_peak, A_peak = f[peak_i], A[peak_i]
    ax.text(f_peak + 0.18, A_peak * label_frac,
            f"{A_peak:.2f}\u00a0mm",
            ha="left", va="center", fontsize=9, color="black",
            bbox=dict(boxstyle="square,pad=0.22", fc="white", ec="#888",
                      lw=0.5, alpha=0.97),
            zorder=7)

    if show_delta:
        delta_mm = row_in - row_ut
        ratio    = row_ut / row_in
        x_arr = 5.6
        ax.annotate("", xy=(x_arr, row_in), xytext=(x_arr, row_ut),
                    arrowprops=dict(arrowstyle="<->", color="#222", lw=1.1,
                                    shrinkA=0, shrinkB=0),
                    zorder=5)
        mid_y = (row_in + row_ut) / 2
        ax.text(x_arr - 0.12, mid_y,
                rf"$\Delta = {delta_mm:.2f}$" + "\u00a0mm\n"
                rf"$A_\mathrm{{Ut}}/A_\mathrm{{inn}} = {ratio:.2f}$",
                ha="right", va="center", fontsize=8.5,
                bbox=dict(boxstyle="square,pad=0.25", fc="white",
                          ec="#444", lw=0.6, alpha=0.95),
                zorder=6)

    ax.set_xlim(0, XMAX)
    ax.set_ylim(0, YMAX_ALL)
    ax.set_title(f"{SIDE_LABEL[side]}, {WIND_LABEL[wind]}", fontsize=10)
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(MultipleLocator(0.2))
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(True, which="minor", alpha=0.12)


# ─── Figure ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(8, 12), sharex=True, sharey=True)
for i, wind in enumerate(("nowind", "fullwind")):
    for j, side in enumerate(("IN", "OUT")):
        draw_panel(axes[i, j], wind=wind, side=side, show_delta=(j == 1))
        if j == 0:
            axes[i, j].set_ylabel("Amplitude [mm]")
        axes[i, j].set_xlabel("Frekvens [Hz]")
        axes[i, j].tick_params(axis="x", labelbottom=True)

for ax_top in axes[0, :]:
    sec = ax_top.secondary_xaxis("top")
    sec.set_xticks(HARMONIC_TICKS)
    sec.set_xticklabels(HARMONIC_LABELS)
    sec.tick_params(axis="x", which="major", labelsize=9, pad=2,
                    length=4, color="#444", labelcolor="#222")

fig.tight_layout()

out_pdf = FIGURES_DIR / "ch04_fft_wave.pdf"
out_pgf = FIGURES_DIR / "ch04_fft_wave.pgf"
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_pgf, bbox_inches="tight")
plt.close(fig)
print(f"   → {out_pdf.relative_to(BASE)}  (+ .pgf)")


# ─── TEXFIGU stub ─────────────────────────────────────────────────────────
# Extra redundancy — every reader-facing number we can pull from meta.json
# for both runs is echoed into the immutable stub so nothing is lost.


def _f(x, n=4):
    try:
        return f"{float(x):.{n}f}"
    except (TypeError, ValueError):
        return "NA"


def _fft_fig_extra_stats(meta, runs, peaks, bin_widths, ka,
                         in_probe, out_probe, freq, fs):
    """Pack everything useful from meta + derivations into a flat stats dict."""
    out = {}
    # ── Paddle-peak amplitudes (mm), Δ and OUT/IN ──────────────────────
    for w in ("nowind", "fullwind"):
        out[f"A_inn_{w}_mm"]   = _f(peaks[(w, "IN")],  3)
        out[f"A_Ut_{w}_mm"]    = _f(peaks[(w, "OUT")], 3)
        out[f"delta_{w}_mm"]   = _f(peaks[(w, "IN")] - peaks[(w, "OUT")], 3)
        out[f"ratio_{w}"]      = _f(peaks[(w, "OUT")] / peaks[(w, "IN")], 4)
    # ── FFT bin widths + window-length N per probe ─────────────────────
    for w in ("nowind", "fullwind"):
        bin_in  = bin_widths[(w, "IN")]
        bin_out = bin_widths[(w, "OUT")]
        out[f"bin_width_inn_{w}_Hz"] = _f(bin_in,  5)
        out[f"bin_width_Ut_{w}_Hz"]  = _f(bin_out, 5)
        out[f"N_inn_{w}_samples"]    = f"{int(round(fs / bin_in))}"
        out[f"N_Ut_{w}_samples"]     = f"{int(round(fs / bin_out))}"
    # ── ka (wavenumber × amplitude) — the primary wave descriptor ─────
    for w in ("nowind", "fullwind"):
        out[f"ka_inn_{w}"] = _f(ka[w]["IN"],  4)
        out[f"ka_Ut_{w}"]  = _f(ka[w]["OUT"], 4)
    # ── Dispersion quantities from meta (IN- and OUT-side means) ──────
    for w in ("nowind", "fullwind"):
        row = meta[meta["path"] == str(runs[w])].iloc[0]
        for side_short, side_long in (("inn", "IN"), ("Ut", "OUT")):
            out[f"wavelength_{side_short}_{w}_m"]    = _f(row[f"{side_long} Wavelength (FFT)"],   4)
            out[f"wavenumber_{side_short}_{w}_per_m"]= _f(row[f"{side_long} Wavenumber (FFT)"],   4)
            out[f"period_{side_short}_{w}_s"]        = _f(row[f"{side_long} WavePeriod (FFT)"],   5)
            out[f"celerity_{side_short}_{w}_m_s"]    = _f(row[f"{side_long} Celerity (FFT)"],     4)
            out[f"Hm0_{side_short}_{w}_mm"]          = _f(row[f"{side_long} Significant Wave Height Hm0"], 3)
            out[f"Hs_{side_short}_{w}_mm"]           = _f(row[f"{side_long} Significant Wave Height Hs"],  3)
            out[f"Froude_{side_short}_{w}"]          = _f(row[f"{side_long} Froude (FFT)"],       5)
            out[f"Ursell_{side_short}_{w}"]          = _f(row[f"{side_long} Ursell (FFT)"],       5)
            out[f"wind_over_c_{side_short}_{w}"]     = _f(row[f"{side_long} Wind/Celerity (FFT)"],4)
            out[f"f_over_fPM_{side_short}_{w}"]      = _f(row[f"{side_long} f/f_PM (FFT)"],       4)
    # ── Per-run OUT/IN (from meta — cross-check vs figure's own ratio) ─
    for w in ("nowind", "fullwind"):
        row = meta[meta["path"] == str(runs[w])].iloc[0]
        out[f"meta_OUT_over_IN_{w}"] = _f(row["OUT/IN (FFT)"], 4)
    # ── Quality: wave_stability + period_amplitude_cv per side ────────
    for w in ("nowind", "fullwind"):
        row = meta[meta["path"] == str(runs[w])].iloc[0]
        out[f"wave_stability_inn_{w}"]  = _f(row["IN wave_stability"],  4)
        out[f"wave_stability_Ut_{w}"]   = _f(row["OUT wave_stability"], 4)
        out[f"period_cv_inn_{w}"]       = _f(row["IN period_amplitude_cv"],  4)
        out[f"period_cv_Ut_{w}"]        = _f(row["OUT period_amplitude_cv"], 4)
    # ── Alternate amplitude methods (LS, percentile, cycles, phase) ───
    for w in ("nowind", "fullwind"):
        row = meta[meta["path"] == str(runs[w])].iloc[0]
        for side_short, probe in (("inn", in_probe), ("Ut", out_probe)):
            out[f"A_{side_short}_{w}_LS_mm"]     = _f(row[f"Probe {probe} Amplitude (LS)"],       3)
            out[f"A_{side_short}_{w}_Stk2_mm"]   = _f(row[f"Probe {probe} Amplitude Stokes2 (LS)"],3)
            out[f"A_{side_short}_{w}_percentile_mm"] = _f(row[f"Probe {probe} Amplitude"],        3)
            out[f"A_{side_short}_{w}_PSD_mm"]    = _f(row[f"Probe {probe} Amplitude (PSD)"],      3)
            out[f"A_{side_short}_{w}_cycles_mean"] = _f(row[f"Probe {probe} Amplitude (cycles) mean"], 3)
            out[f"A_{side_short}_{w}_phase_mean"]  = _f(row[f"Probe {probe} Amplitude (phase) mean"],  3)
            out[f"DC_{side_short}_{w}_mm"]         = _f(row[f"Probe {probe} DC (LS)"],             4)
            out[f"residualRMS_{side_short}_{w}_mm"]= _f(row[f"Probe {probe} Residual RMS (LS)"],   4)
    # ── Window bounds in seconds (per probe, per wind) ────────────────
    for w in ("nowind", "fullwind"):
        row = meta[meta["path"] == str(runs[w])].iloc[0]
        for side_short, probe in (("inn", in_probe), ("Ut", out_probe)):
            s = int(row[f"Computed Probe {probe} start"])
            e = int(row[f"Computed Probe {probe} end"])
            out[f"window_{side_short}_{w}_s"] = f"[{s/fs:.3f}, {e/fs:.3f}]"
            hg_shift = row.get(f"Probe {probe} hg_snap_shift")
            if hg_shift is not None:
                out[f"hg_snap_shift_{side_short}_{w}_samples"] = _f(hg_shift, 1)
    # ── Dataset / setup metadata (per wind — both runs same folder) ───
    for w in ("nowind", "fullwind"):
        row = meta[meta["path"] == str(runs[w])].iloc[0]
        out[f"file_date_{w}"]            = str(row.get("file_date", "NA"))[:10]
        out[f"mooring_{w}"]              = str(row.get("Mooring", "NA"))
        out[f"probe_height_{w}_mm"]      = _f(row.get("probe_height_mm"), 1)
        out[f"probe_range_mode_{w}"]     = str(row.get("probe_range_mode", "NA"))
        out[f"water_depth_{w}_mm"]       = _f(row.get("water_depth_mm", 580), 1)
    # ── Derived: kd (depth regime) from OUT-side k, h = 0.58 m ────────
    for w in ("nowind", "fullwind"):
        row = meta[meta["path"] == str(runs[w])].iloc[0]
        try:
            k_m = float(row["OUT Wavenumber (FFT)"])
            h_m = float(row.get("water_depth_mm", 580)) / 1000.0
            kd  = k_m * h_m
            regime = ("shallow"       if kd < np.pi/10 else
                      "intermediate"  if kd < np.pi     else
                      "deep")
            out[f"kd_Ut_{w}"]      = _f(kd, 3)
            out[f"depth_regime_{w}"] = regime
        except Exception:
            pass
    # ── Input targets (sanity) ────────────────────────────────────────
    out["input_freq_Hz"]     = _f(freq, 2)
    out["input_amp_V"]       = "0.20"
    out["sampling_rate_Hz"]  = _f(fs, 1)
    return out


_nw, _fw = ROW_PEAKS["nowind"], ROW_PEAKS["fullwind"]
# ══════════════════════════════════════════════════════════════════════════
# USER-AUTHORED CAPTION — edit the string below; the stub body uses it verbatim.
# ══════════════════════════════════════════════════════════════════════════
# - Empty string  → body renders with a TODO placeholder (hint to author it).
# - Non-empty     → body's \caption[short]{full} uses your text literally.
# Regenerate by running this script. force=True, so the body is always
# rewritten from this value — editing the .tex by hand won't survive.
# Author captions here, not in the .tex.
#
# No agent-drafted captions are included anywhere in the stub; the
# IMMUTABLE block only records provenance / filters / stats / method.
caption = ""

_stub_meta = pu.build_fig_meta(
    {
        "filters": {
            "PanelCondition":            "full",
            "WaveFrequencyInput [Hz]":   FREQ,
            "WaveAmplitudeInput [Volt]": AMP_V,
            "WindCondition":             None,      # both conditions shown
            "quality_flag":              "ok",
        },
        "plotting": {
            "figure_name": "ch04_fft_wave",
            "caption":     caption,
        },
    },
    chapter="04",
    extra={"script": "analysis_scratch/fft_wave_spectrum.py"},
    computed_in="analysis_scratch/fft_wave_spectrum.py (single-run FFT spectrum bar plot)",
    data_class="DELEG",
    findings_doc="memory/methodology_hg_window_kills_peak_bias.md",
    fft_window_hz=0.1,
    extra_params=(
        # Run paths — full canonical CSVs for reproducibility
        f"run_paths=nowind:{RUNS['nowind'].relative_to(BASE)} | "
        f"fullwind:{RUNS['fullwind'].relative_to(BASE)}. "
        # Probe identity — as stored in meta.json (distance_mm/lateral_mm)
        f"probes=IN:{IN_PROBE}+OUT:{OUT_PROBE}. "
        # Figure extent + discrete-bin annotations
        f"xlim=[0, {XMAX}] Hz; harmonic vertical markers at k·f for k=1..4 "
        f"(1.4, 2.8, 4.2, 5.6 Hz). Top x-axis carries the harmonic labels "
        f"(f, 2f, 3f, 4f) on the nowind row; bottom x-axis carries the "
        f"numeric frequency ticks on both rows. "
        # FFT pipeline parameters — relevant to what the reader sees
        f"fft_engine=numpy.fft.fft on windowed eta_{{pos}}_interp "
        f"(H&G probe-shifted [50T, 60T] window, start UC-snapped within ±T, "
        f"end UC-snapped to the 10th upcrossing, ±0.5 T guard; "
        f"see memory/methodology_hg_window_kills_peak_bias.md). "
        # Amplitude convention
        f"amplitude_axis = 2*|FFT|/N (one-sided real amplitude, mm); "
        f"matches Probe {{pos}} Amplitude (FFT) in meta.json to 4 dp. "
        # Δ + ratio convention in the figure
        f"Δ = A_inn − A_Ut per row (per-wind, using each row's own IN and "
        f"OUT paddle peaks); ratio = A_Ut / A_inn per row. Horizontal "
        f"reference lines match the row's own IN/OUT peaks. "
        # Bar width interpretation
        f"bar_width = 0.9 * FFT_bin_spacing (per-probe bin spacing is the "
        f"real FFT resolution fs/N for that probe's window). "
        # Reading cues
        f"Colour convention: blue = no wind (nowind), red = med vind "
        f"(fullwind) — CLAUDE.md thesis-wide WIND_COLOR_MAP. "
        # Font
        f"Typeset in NewComputerModern10 (OTFs from TeXLive's "
        f"newcomputermodern package, registered via font_manager)."
    ),
    extra_stats=_fft_fig_extra_stats(meta, RUNS, peaks, bin_widths, KA,
                                     IN_PROBE, OUT_PROBE, FREQ, FS),
)

# Force-rewrite: the pre-existing stub had a 4-subfigure body for the old
# overlay-plot layout, not the single 2×2 grid we're saving now.
pu.write_figure_stub(_stub_meta, plot_type="spectrum_fft",
                     subfig_filenames=["ch04_fft_wave"],
                     force=True,
                     width="\\linewidth")
print(f"   stub → output/TEXFIGU/ch04_fft_wave.tex")
print("\nDone.")
