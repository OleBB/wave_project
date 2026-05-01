"""
Quick-look: all 4 probes overlaid for each fromMax* wind-decay run.

Iterates through all known wind-decay runs (run_category="wind_decay",
i.e. fromMaxToZeroWin / fromMaxToNoWin / fromMaxWinToZeroWin). Each run gets
its own PNG so they can be compared side-by-side.

All runs picked here are in the march2026_better_rearranging probe config
(IN=9373/170, OUT=12400/250, parallel=9373/340, upstream=8804/250).

Outputs:
    analysis_scratch/wind_decay_timeseries_<DATE>_<tag>.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wavescripts.improved_data_loader import load_analysis_data, load_processed_dfs
from wavescripts.plot_utils import apply_thesis_style

# Thesis-grade rcParams (NewComputerModern body font + math via mathtext,
# 10pt body, 9pt ticks/legend, tight bbox on save). Idempotent — safe to
# call here for delegated scratch scripts that don't go through plotter.py.
apply_thesis_style()

FS = 250.0
BASE = Path("/Users/ole/Kodevik/wave_project")

# Each entry: (tag, processed_dir_name, run_csv_relative_to_wavedata)
# Only the two "best" runs are active — the others are commented out so the
# corresponding PNGs in analysis_scratch/ are not regenerated. The kept PNGs
# remain on disk untouched.
RUNS = [
    # ("20260314",
    #  "PROCESSED-20260314-ProbePos4_31_FPV_2-tett6roof",
    #  "20260314-ProbePos4_31_FPV_2-tett6roof/fullpanel-fromMaxWinToZeroWin-run1.csv"),
    # ("20260319",
    #  "PROCESSED-20260319-ProbePos4_31_FPV_2-tett6roof-under9Mooring",
    #  "20260319-ProbePos4_31_FPV_2-tett6roof-under9Mooring/fullpanel-fromMaxToNoWin-nowave-depth580-mstop30-run1.csv"),
    # ("20260324",
    #  "PROCESSED-20260324-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100",
    #  "20260324-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100/fullpanel-fromMaxToZeroWin-depth580-run1.csv"),
    # ("20260326_dieout",
    #  "PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange",
    #  "20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange/experimental-fromMaxToZeroWinWaitingForItToDieOut-depth580-mstop30-run1.csv"),
    # ("20260326_dieoutPart2",
    #  "PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange",
    #  "20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange/experimental-fromMaxToZeroWinWaitingForItToDieOutPart2-depth580-mstop30-run1.csv"),
    # ("20260326_run2",
    #  "PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange",
    #  "20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange/experimental-fromMaxToZeroWin-depth580-mstop30-run2.csv"),
    # ("20260327",
    #  "PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange",
    #  "20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange/experimental-fromMaxToZeroWin-depth580.csv"),
    ("20260327_endofday",
     "PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange",
     "20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange/experimental-fromMaxToZeroWin-depth580-mstop30-run-endofday.csv"),
]

# Wind ramp-UP (fromZeroToMax* / fromZeroWinToMaxWin*) — the opposite direction.
RUNS_RAMPUP = [
    ("20260314",
     "PROCESSED-20260314-ProbePos4_31_FPV_2-tett6roof",
     "20260314-ProbePos4_31_FPV_2-tett6roof/fullpanel-fromZeroWinToMaxWin-run1.csv"),
]

# march2026_better_rearranging probe set (also matches march2026_rearranging
# for the IN-side pair; missing probes are skipped silently).
PROBES = ["8804/250", "9373/170", "9373/340", "12400/250"]
LABELS = {
    "8804/250":  "8804/250 (upstream)",
    "9373/170":  "9373/170 (IN)",
    "9373/340":  "9373/340 (parallel)",
    "12400/250": "12400/250 (OUT)",
}
COLORS = {
    "8804/250":  "#888888",
    "9373/170":  "#FEA11B",   # IN  — amber orange (UiO color pallette)
    "9373/340":  "#2ca02c",
    "12400/250": "#1E9C68",   # OUT — new green (not UiO)
}


def plot_run_zoom(tag: str, processed_dir_name: str, run_csv_rel: str,
                  *, kind: str, xlim: tuple[float, float] | None = (0, 60),
                  zero_kind: str = "first", zero_secs: float = 2.0,
                  ylim: tuple[float, float] = (-15.0, 15.0),
                  probes: list[str] | None = None,
                  name_suffix: str = "",
                  x_unit: str = "seconds",
                  figsize: tuple[float, float] = (14.0, 7.5),
                  thesis_name: str | None = None) -> None:
    """Per-probe η is re-zeroed so the reader sees the wind-driven setup as
    a deviation from a known still-water moment.

    xlim       = (start_s, end_s) in seconds, or None → full record
    x_unit     = "seconds" | "minutes" — labelling + tick spacing
    zero_kind  = "first" → subtract per-probe mean of t ∈ [0, zero_secs]
    zero_kind  = "last"  → subtract per-probe mean of t ∈ [T-zero_secs, T]
    """
    from matplotlib.ticker import MultipleLocator

    target_dir = BASE / "waveprocessed" / processed_dir_name
    run_csv    = str(BASE / "wavedata" / run_csv_rel)

    _, _, _, _ = load_analysis_data(str(target_dir), load_processed=False)
    proc = load_processed_dfs(str(target_dir))
    if run_csv not in proc:
        print(f"   skip — run not in processed cache:\n     {run_csv}")
        return

    df = proc[run_csv]
    t  = np.arange(len(df)) / FS
    T  = float(t[-1])

    if xlim is None:
        xlim = (0.0, T)
    full_range = abs(xlim[0]) < 1e-9 and abs(xlim[1] - T) < 1e-3
    print(f"\n{tag} ({xlim[0]:.1f}-{xlim[1]:.1f} s, "
          f"zero={zero_kind} {zero_secs}s, x={x_unit}) …")

    if zero_kind == "first":
        zero_mask = (t >= 0.0) & (t <= zero_secs)
        zero_label = f"first {zero_secs:g} s"
    elif zero_kind == "last":
        zero_mask = (t >= T - zero_secs) & (t <= T)
        zero_label = f"last {zero_secs:g} s"
    else:
        raise ValueError(f"zero_kind must be 'first' or 'last', got {zero_kind!r}")

    m = (t >= xlim[0]) & (t <= xlim[1])

    probes_to_plot = probes if probes is not None else PROBES

    # Plotting axis unit
    if x_unit == "minutes":
        scale     = 1.0 / 60.0
        x_label   = "Tid [min]"
    elif x_unit == "seconds":
        scale     = 1.0
        x_label   = "Tid [s]"
    else:
        raise ValueError(f"x_unit must be 'seconds' or 'minutes', got {x_unit!r}")

    t_plot       = t * scale
    xlim_plot    = (xlim[0] * scale, xlim[1] * scale)
    span_plot    = xlim_plot[1] - xlim_plot[0]

    fig, ax = plt.subplots(figsize=figsize)
    for probe in probes_to_plot:
        col = f"eta_{probe}_interp" if f"eta_{probe}_interp" in df.columns else f"eta_{probe}"
        if col not in df.columns:
            continue
        eta = df[col].to_numpy(dtype=float)
        baseline = float(np.nanmean(eta[zero_mask]))
        eta_z = eta - baseline
        ax.plot(t_plot[m], eta_z[m], lw=1.0, color=COLORS[probe], alpha=0.9,
                label=f"{LABELS[probe]}  (μ₀={baseline:+.2f} mm)")

    ax.axhline(0, color="#444", lw=0.6, alpha=0.6)

    # Tick spacing — pick reasonable major/minor for the chosen x_unit / span.
    if x_unit == "minutes":
        if   span_plot > 15: major, minor = 5.0,  1.0
        elif span_plot >  6: major, minor = 2.0,  0.5
        elif span_plot >  3: major, minor = 1.0,  0.2
        else:                major, minor = 0.5,  0.1
    else:  # seconds
        major, minor = (5.0, 1.0) if span_plot <= 80 else (10.0, 2.0)

    ax.xaxis.set_major_locator(MultipleLocator(major))
    ax.xaxis.set_minor_locator(MultipleLocator(minor))
    ax.yaxis.set_major_locator(MultipleLocator(5.0))
    ax.yaxis.set_minor_locator(MultipleLocator(1.0))
    ax.grid(True, which="major", alpha=0.40)
    ax.grid(True, which="minor", alpha=0.18)
    ax.tick_params(which="both", direction="out", length=4)
    ax.tick_params(which="minor", length=2)

    ax.set_xlim(*xlim_plot)        # tight — exactly the signal range, no padding
    ax.set_ylim(*ylim)
    # Font sizes inherit from apply_thesis_style() rcParams (10pt labels,
    # 9pt ticks/legend) — no per-axis fontsize overrides here.
    ax.set_xlabel(x_label)
    ax.set_ylabel(r"$\eta - \mu_0$ [mm]")
    title_kind = "Wind decay" if kind == "decay" else "Wind ramp-up"
    span_lbl   = (f"first {int(round(span_plot))} min"
                  if x_unit == "minutes" else
                  f"first {int(round(span_plot))} s")
    if full_range:
        span_lbl = f"full record ({T:.1f} s = {T/60:.2f} min)"
    # Title is included on scratch PNGs (kept verbose for sanity) but
    # suppressed on the thesis PDF — the LaTeX \caption{} carries the
    # identification, matching the inspirational_timeseries pattern.
    # Dropping the title also saves ~0.4–0.5 in of vertical space, helping
    # the four subfigures fit a single A4 page at 1-inch margins.
    if thesis_name is None:
        ax.set_title(
            f"{title_kind} — {span_lbl} — {tag} — {Path(run_csv).name}\n"
            f"per-probe baseline μ₀ = mean of {zero_label}",
        )
    ax.legend(loc="upper right", framealpha=0.95, ncol=2)

    fname_kind = "wind_decay_timeseries" if kind == "decay" else "wind_rampup_timeseries"
    if full_range:
        range_tag = "_full"
    elif x_unit == "seconds":
        range_tag = f"_zoom{int(round(span_plot))}"
    else:
        range_tag = f"_zoom{int(round(span_plot))}min"
    out = Path(__file__).parent / f"{fname_kind}_{tag}{range_tag}{name_suffix}.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"   → {out.relative_to(BASE)}")

    # Optional thesis-side render: same figure, saved as PDF + PNG into
    # output/FIGURES under the thesis-spec name. Pairs with a multi-subfig
    # stub at output/TEXFIGU/<parent>.tex (written separately at the bottom
    # of this script).
    if thesis_name is not None:
        thesis_dir = BASE / "output" / "FIGURES"
        thesis_dir.mkdir(parents=True, exist_ok=True)
        thesis_pdf = thesis_dir / f"{thesis_name}.pdf"
        thesis_png = thesis_dir / f"{thesis_name}.png"
        fig.savefig(thesis_pdf, bbox_inches="tight")
        fig.savefig(thesis_png, dpi=160, bbox_inches="tight")
        print(f"   → {thesis_pdf.relative_to(BASE)}  (+ .png)")

    plt.close(fig)


def plot_run(tag: str, processed_dir_name: str, run_csv_rel: str,
             *, kind: str = "decay") -> None:
    """kind: 'decay' → wind_decay_timeseries_*.png ; 'rampup' → wind_rampup_timeseries_*.png"""
    target_dir = BASE / "waveprocessed" / processed_dir_name
    run_csv    = str(BASE / "wavedata" / run_csv_rel)

    print(f"\n{tag} …")
    meta, _, _, _ = load_analysis_data(str(target_dir), load_processed=False)
    proc = load_processed_dfs(str(target_dir))

    if run_csv not in proc:
        print(f"   skip — run not in processed cache:\n     {run_csv}")
        return

    df = proc[run_csv]
    t  = np.arange(len(df)) / FS

    fig, ax = plt.subplots(figsize=(12, 4.5))
    for probe in PROBES:
        col = f"eta_{probe}_interp" if f"eta_{probe}_interp" in df.columns else f"eta_{probe}"
        if col not in df.columns:
            print(f"   skip probe {probe} — no column in this dataset")
            continue
        eta = df[col].to_numpy(dtype=float)
        ax.plot(t, eta, lw=0.5, color=COLORS[probe], alpha=0.85, label=LABELS[probe])

    ax.axhline(0, color="#555", lw=0.5, alpha=0.5)
    ax.set_xlabel("Tid [s]")
    ax.set_ylabel(r"$\eta$ [mm]")
    title_kind = "Wind decay" if kind == "decay" else "Wind ramp-up"
    ax.set_title(f"{title_kind} — {tag} — {Path(run_csv).name}")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fname_kind = "wind_decay_timeseries" if kind == "decay" else "wind_rampup_timeseries"
    out = Path(__file__).parent / f"{fname_kind}_{tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   → {out.relative_to(BASE)}")


# 4-probe overview plots — left as-is on disk; commented out so they don't
# regenerate (and so the new IN/OUT colour picks don't bleed into them).
# for tag, pdir, csv in RUNS:
#     plot_run(tag, pdir, csv, kind="decay")
# for tag, pdir, csv in RUNS_RAMPUP:
#     plot_run(tag, pdir, csv, kind="rampup")

# 4-probe zoom plots — also kept on disk; commented out for the same reason.
# plot_run_zoom("20260327_endofday", ..., kind="decay", xlim=(0, 60), zero_kind="last")
# plot_run_zoom("20260314",          ..., kind="rampup", xlim=(0, 60), zero_kind="first")

# ── IN + OUT only — 4 plots: zoom + full record × decay + ramp-up ──────────
# Decay run: tank settles by the end → use last 2 s as the still-water reference.
# Ramp-up run: tank starts at rest → use first 2 s as the reference.
# Both clipped to ±15 mm. Zoom = first 60 s in seconds; full = whole record in
# minutes with x-axis tight to the actual signal length.
DECAY = dict(
    tag="20260327_endofday",
    processed_dir_name="PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange",
    run_csv_rel="20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange/experimental-fromMaxToZeroWin-depth580-mstop30-run-endofday.csv",
    kind="decay", zero_kind="last", zero_secs=2.0,
)
RAMPUP = dict(
    tag="20260314",
    processed_dir_name="PROCESSED-20260314-ProbePos4_31_FPV_2-tett6roof",
    run_csv_rel="20260314-ProbePos4_31_FPV_2-tett6roof/fullpanel-fromZeroWinToMaxWin-run1.csv",
    kind="rampup", zero_kind="first", zero_secs=2.0,
)
INOUT = ["9373/170", "12400/250"]

# figsize: tuned so the four subfigures (rampup_full + rampup_zoom +
# decay_full + decay_zoom, each at 1.0\linewidth in the .tex stub) fit a
# single A4 page at 1-inch margins. The in-figure title is suppressed for
# thesis output (see plot_run_zoom: `if thesis_name is None`) so the saved
# PDF aspect tracks figsize aspect closely (xlabel + legend only add a
# small bbox margin).
#
# Aspect ratio is what matters here, since \includegraphics[width=\linewidth]
# scales the rendered image to the textblock width (~6.27 in at a4paper +
# 1-in margins). At those settings:
#     full → displayed height ≈ 1.15 in
#     zoom → displayed height ≈ 2.22 in
# Total image stack ≈ 6.7 in, leaving ~3.0 in for the parent caption + four
# subfig captions + 3 × \\[1ex] inter-subfig spacing within the ~9.7 in A4
# textblock height. Comfortable margin even when captions grow.
#
# Zoom is intentionally taller than full (~1.95×) so the closeup gets clearly
# more vertical real estate than the overview, per the visual-rhetoric intent
# ("the zoom deserves more vertical space; the full is an overview anyway").
FIGSIZE_FULL = (14.0, 2.0)
FIGSIZE_ZOOM = (14.0, 4.5)

# Full record (minutes, tight xlim — set inside the function when xlim=None)
plot_run_zoom(**RAMPUP, xlim=None, ylim=(-15, 15),
              probes=INOUT, name_suffix="_inout", x_unit="minutes",
              figsize=FIGSIZE_FULL, thesis_name="ch04_wind_rampup_full")
plot_run_zoom(**RAMPUP, xlim=(0, 60), ylim=(-15, 15),
              probes=INOUT, name_suffix="_inout", x_unit="seconds",
              figsize=FIGSIZE_ZOOM, thesis_name="ch04_wind_rampup_zoom60")
plot_run_zoom(**DECAY,  xlim=None, ylim=(-15, 15),
              probes=INOUT, name_suffix="_inout", x_unit="minutes",
              figsize=FIGSIZE_FULL, thesis_name="ch04_wind_decay_full")
plot_run_zoom(**DECAY,  xlim=(0, 60), ylim=(-15, 15),
              probes=INOUT, name_suffix="_inout", x_unit="seconds",
              figsize=FIGSIZE_ZOOM, thesis_name="ch04_wind_decay_zoom60")

# ── Composite TEXFIGU stub: 4 subfigures stacked on a full page (column) ───
# Order, top → bottom: ramp-up full, ramp-up zoom, decay full, decay zoom.
# Captions live centrally in main_save_figures.py::FIGURE_CAPTIONS — empty
# entries here will land as TODO placeholders in the .tex stub.
import wavescripts.plot_utils as pu

pu.ACTIVE_DATASETS = [p.name for p in
                      sorted(BASE.glob("waveprocessed/PROCESSED-*"))]
pu.TEXFIGU_DIR = BASE / "output" / "TEXFIGU"
pu.FIGURES_DIR = BASE / "output" / "FIGURES"
pu.TEXFIGU_DIR.mkdir(parents=True, exist_ok=True)

_meta = pu.build_fig_meta(
    {
        "filters": {"WindCondition": "transition"},
        "plotting": {"figure_name": "ch04_wind_transition_overview"},
    },
    chapter="04",
    extra={"script": "analysis_scratch/wind_decay_timeseries.py"},
    computed_in="analysis_scratch/wind_decay_timeseries.py",
    data_class="DELEG",
    findings_doc="(none)",
    extra_params=(
        f"Two single-run time-series. "
        f"Wind ramp-up = 20260314 fullpanel-fromZeroWinToMaxWin-run1.csv "
        f"(probe config march2026_better_rearranging). "
        f"Wind decay   = 20260327 experimental-fromMaxToZeroWin-depth580-mstop30-run-endofday.csv "
        f"(same probe config). "
        f"Probes shown: IN=9373/170 (colour #FEA11B amber, fully exposed to wind), "
        f"OUT=12400/250 (colour #2EC483 green, sheltered behind panel). "
        f"Per-probe baseline μ₀ subtracted: ramp-up = mean of first 2 s "
        f"(tank at rest before fan); decay = mean of last 2 s "
        f"(tank settled by ~23 min). "
        f"Y-axis clipped to ±15 mm so the wind-setup tilt and chop envelope "
        f"are readable on the same scale across all four panels. "
        f"Subfigure order top→bottom: ramp-up full record (minutes), "
        f"ramp-up first 60 s (seconds), decay full record (minutes), "
        f"decay first 60 s (seconds)."
    ),
)
pu.write_figure_stub(
    _meta,
    plot_type="wind_transition_timeseries",
    subfig_filenames=[
        "ch04_wind_rampup_full",
        "ch04_wind_rampup_zoom60",
        "ch04_wind_decay_full",
        "ch04_wind_decay_zoom60",
    ],
    subfig_layout="column",
    force=True,
)
print(f"   stub → output/TEXFIGU/ch04_wind_transition_overview.tex")

print("\nDone.")
