"""
Parallel-probe PSD comparison (CH04 §3 supporting figure)
==========================================================

Two-panel PSD figure: one curve per run, faceted by probe.
  Left panel  = 9373/170 (wall-side)
  Right panel = 9373/340 (far-side)

Data source assumption
----------------------
This script consumes `combined_psd_dict` produced by
`wavescripts.improved_data_loader.load_analysis_data`. Expected structure:

    combined_psd_dict : dict[str, pandas.DataFrame]
        Keys   = absolute CSV paths (one per wave run)
        Values = DataFrame indexed by frequency [Hz], with columns
                 "Pxx 9373/170", "Pxx 9373/340", "Pxx 12400/250", ...
        PSD units = mm² / Hz  (η is in mm; Welch via scipy.signal.welch)

Axes choice
-----------
- Y is log:    PSD spans several decades between paddle peak and noise floor.
- X is linear: paddle peak (~1.4 Hz) and wind band (~2–6 Hz) sit close
               together; log-X compresses both into a corner.
Pass log_x=True to switch X to log if you want to inspect sub-Hz content.

How to change things
--------------------
- Probes:    edit the module-level PROBES tuple.
- Runs:      edit the boolean `mask` in the __main__ block (filters
             combined_meta; matching `path` values are fed to the
             plotting function).
- Save path: pass `out_path="…"` to plot_parallel_probe_psd, or call
             `fig.savefig("psd_parallel_probes.png", dpi=200,
             bbox_inches="tight")` on the returned figure.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wavescripts.improved_data_loader import load_analysis_data
from wavescripts.plot_utils import apply_thesis_style

apply_thesis_style()

PROBES = ("9373/170", "9373/340")
F_MAX_HZ = 10.0


def plot_parallel_probe_psd(
    combined_psd_dict: dict,
    probes: tuple = PROBES,
    f_max_hz: float | None = F_MAX_HZ,
    log_x: bool = False,
    label_fn=None,
    out_path: str | None = None,
):
    """One PSD curve per run on each panel; left=probes[0], right=probes[1]."""
    label_fn = label_fn or (lambda p: Path(p).stem[:32])
    paths = list(combined_psd_dict.keys())
    if not paths:
        raise ValueError("combined_psd_dict is empty — nothing to plot.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    cmap = plt.get_cmap("tab10" if len(paths) <= 10 else "viridis")
    colors = [cmap(i / max(len(paths) - 1, 1)) for i in range(len(paths))]

    for ax, probe in zip(axes, probes):
        col = f"Pxx {probe}"
        for i, path in enumerate(paths):
            df = combined_psd_dict[path]
            if col not in df.columns:
                continue
            # Each probe column lives on its own Welch frequency grid; the
            # union-merged DataFrame has NaN at the other probes' grid points.
            # Drop NaN per-probe so matplotlib does not break the line.
            ser = df[col].dropna()
            f = ser.index.to_numpy()
            pxx = ser.to_numpy()
            if f_max_hz is not None:
                m = f <= f_max_hz
                f, pxx = f[m], pxx[m]
            ax.plot(f, pxx, color=colors[i], lw=0.9, alpha=0.85,
                    label=label_fn(path))
        ax.set_yscale("log")
        if log_x:
            ax.set_xscale("log")
        ax.set_xlabel(r"$f$ [Hz]")
        ax.set_title(f"Probe {probe}", fontsize=10)
        ax.grid(True, which="both", alpha=0.3)

    axes[0].set_ylabel(r"PSD $S_{\eta\eta}$ [mm$^{2}$/Hz]")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center",
                   ncol=min(len(labels), 4), fontsize=7,
                   bbox_to_anchor=(0.5, -0.04))

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    if out_path is not None:
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Saved -> {out_path}")
    return fig, axes


def plot_parallel_probe_psd_per_run(
    combined_psd_dict: dict,
    probes: tuple = PROBES,
    f_max_hz: float | None = F_MAX_HZ,
    log_x: bool = False,
    title_fn=None,
    n_cols: int = 3,
    out_path: str | None = None,
):
    """One panel per run; both probes overlaid in each panel.

    Layout: ceil(n_runs / n_cols) rows × n_cols columns. Empty trailing
    cells are hidden. Each panel gets its own x-label so the bottom row
    of an L-shaped grid still reads cleanly.
    """
    title_fn = title_fn or (lambda p: Path(p).stem[:32])
    paths = list(combined_psd_dict.keys())
    if not paths:
        raise ValueError("combined_psd_dict is empty — nothing to plot.")

    n_runs = len(paths)
    n_cols = min(n_cols, n_runs)
    n_rows = (n_runs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.5 * n_cols, 3.2 * n_rows),
        sharey=True, squeeze=False,
    )
    probe_colors = {probes[0]: "tab:blue", probes[1]: "tab:orange"}

    for run_idx, path in enumerate(paths):
        ax = axes[run_idx // n_cols, run_idx % n_cols]
        df = combined_psd_dict[path]
        for probe in probes:
            col = f"Pxx {probe}"
            if col not in df.columns:
                continue
            ser = df[col].dropna()
            f = ser.index.to_numpy()
            pxx = ser.to_numpy()
            if f_max_hz is not None:
                m = f <= f_max_hz
                f, pxx = f[m], pxx[m]
            ax.plot(f, pxx, color=probe_colors.get(probe), lw=0.9,
                    alpha=0.85, label=probe)
        ax.set_yscale("log")
        if log_x:
            ax.set_xscale("log")
        if f_max_hz is not None:
            ax.set_xlim(0, f_max_hz)
        ax.set_xlabel(r"$f$ [Hz]")
        ax.set_title(title_fn(path), fontsize=8)
        ax.grid(True, which="both", alpha=0.3)

    # Hide trailing empty cells (when n_runs is not a multiple of n_cols).
    for k in range(n_runs, n_rows * n_cols):
        axes[k // n_cols, k % n_cols].set_visible(False)

    for ax in axes[:, 0]:
        ax.set_ylabel(r"PSD $S_{\eta\eta}$ [mm$^{2}$/Hz]")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center",
                   ncol=len(labels), fontsize=10,
                   bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    if out_path is not None:
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Saved -> {out_path}")
    return fig, axes


if __name__ == "__main__":
    BASE = Path(__file__).parent.parent
    TARGET_DIRS = [
        BASE / "waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange",
        BASE / "waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange",
    ]

    print("Loading meta + PSD dict for canon March-2026 lowrange …")
    meta, _proc, _fft, psd_dict = load_analysis_data(
        *map(str, TARGET_DIRS), load_processed=False
    )

    # Representative subset: 1.4 Hz, 0.2 V, fullpanel, all wind conditions.
    # Edit this mask to reshape the figure (e.g. drop the freq filter to
    # see every frequency at one amplitude, or filter by WindCondition).
    mask = (
        (meta["PanelCondition"] == "full")
        & np.isclose(meta["WaveFrequencyInput [Hz]"], 1.4, atol=0.005)
        & np.isclose(meta["WaveAmplitudeInput [Volt]"], 0.2, atol=0.01)
        & (meta["quality_flag"] == "ok")
    )
    sel_paths = meta.loc[mask, "path"].tolist()
    psd_subset = {p: psd_dict[p] for p in sel_paths if p in psd_dict}
    print(f"  {len(psd_subset)} runs selected")

    # Legend label = wind condition + short filename so curves are
    # identifiable at a glance.
    wind_by_path = dict(zip(meta["path"], meta["WindCondition"]))
    def label(p):
        return f"{wind_by_path.get(p, '?'):7s} {Path(p).stem[:30]}"

    out_pdf = Path(__file__).parent / "parallel_probe_psd.pdf"
    fig, _ = plot_parallel_probe_psd(
        psd_subset, label_fn=label, out_path=str(out_pdf)
    )
    plt.close(fig)

    # Per-run view: one panel per run, both probes overlaid in each panel.
    out_pdf2 = Path(__file__).parent / "parallel_probe_psd_per_run.pdf"
    fig2, _ = plot_parallel_probe_psd_per_run(
        psd_subset, title_fn=label, out_path=str(out_pdf2)
    )
    plt.close(fig2)
