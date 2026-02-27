#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_utils.py
=============
Infrastructure for saving figures and generating LaTeX stubs.

Every plotter function should end with:

    panels = _save_figure(fig, meta, plot_type="timeseries")
    write_figure_stub(meta, panels)

The meta dict is built from plotvariables + combined_meta_sel via
build_fig_meta().

Filename convention
-------------------
    {chapter}_{plot_type}_{panel}panel-{wind}wind-amp{amp}-freq{freq}-probe{probes}

    e.g.  05_timeseries_reversepanel-fullwind-amp0100-freq0650-probe2og3

Output directories (relative to project root, mirroring the TeX project)
---------
    output/FIGURES/   ← .pdf and .pgf  (copy → /Users/ole/main/FIGURES/)
    output/TEXFIGU/   ← .tex stubs     (copy → /Users/ole/main/TEXFIGU/)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union
import matplotlib.pyplot as plt


# ── Output roots (edit if your project layout changes) ────────────────────────

FIGURES_DIR = Path("output/FIGURES")
TEXFIGU_DIR = Path("output/TEXFIGU")


# ── Filename helpers ───────────────────────────────────────────────────────────

def _fmt_condition(val) -> str:
    """['no', 'full'] → 'no-full'   |   'reverse' → 'reverse'   |   None → 'all'"""
    if val is None:
        return "all"
    if isinstance(val, (list, tuple)):
        return "-".join(str(v).lower() for v in val)
    return str(val).lower()


def _fmt_amp(val) -> str:
    """0.1 → '0100'  |  [0.1, 0.2] → '0100-0200'  |  None → 'allamp'"""
    if val is None:
        return "allamp"
    vals = val if isinstance(val, (list, tuple)) else [val]
    return "-".join(f"{float(v)*1000:04.0f}" for v in vals)


def _fmt_freq(val) -> str:
    """0.65 → '0650'  |  [0.65, 1.3] → '0650-1300'  |  None → 'allfreq'"""
    if val is None:
        return "allfreq"
    vals = val if isinstance(val, (list, tuple)) else [val]
    return "-".join(f"{float(v)*1000:04.0f}" for v in vals)


def _fmt_probes(val) -> str:
    """[2, 3] → '2og3'  |  2 → '2'  |  None → 'allprobes'"""
    if val is None:
        return "allprobes"
    if isinstance(val, (list, tuple)):
        return "og".join(str(int(p)) for p in val)
    return str(int(val))


def build_filename(plot_type: str, meta: dict) -> str:
    """
    Build the canonical figure filename (without extension) from meta.

    Parameters
    ----------
    plot_type : str
        Short descriptor, e.g. 'timeseries', 'psd', 'scatter_p2p3'
    meta : dict
        As returned by build_fig_meta().

    Returns
    -------
    str
        e.g. '05_timeseries_reversepanel-fullwind-amp0100-freq0650-probe2og3'
    """
    chapter = str(meta.get("chapter", "00"))
    panel   = _fmt_condition(meta.get("panel"))
    wind    = _fmt_condition(meta.get("wind"))
    amp     = _fmt_amp(meta.get("amplitude"))
    freq    = _fmt_freq(meta.get("frequency"))
    probes  = _fmt_probes(meta.get("probes"))

    return (
        f"{chapter}_{plot_type}_"
        f"{panel}panel-{wind}wind-"
        f"amp{amp}-freq{freq}-"
        f"probe{probes}"
    )


# ── Meta builder ──────────────────────────────────────────────────────────────

def build_fig_meta(plotvariables: dict, chapter: str = "05",
                   extra: Optional[dict] = None) -> dict:
    """
    Extract figure metadata from a plotvariables dict.

    Parameters
    ----------
    plotvariables : dict
        Your standard plot-config dict with 'filters' and 'plotting' keys.
    chapter : str
        Two-digit chapter prefix, e.g. '05'.
    extra : dict, optional
        Any additional immutable fields to include in the stub comments,
        e.g. {"run_id": "2024-11-03_run2", "script": "analysis_swell.py"}.

    Returns
    -------
    dict
        Flat meta dict ready for build_filename() and write_figure_stub().
    """
    f = plotvariables.get("filters", {})
    p = plotvariables.get("plotting", {})

    meta = {
        "chapter":   chapter,
        "panel":     f.get("PanelCondition"),
        "wind":      f.get("WindCondition"),
        "amplitude": f.get("WaveAmplitudeInput [Volt]"),
        "frequency": f.get("WaveFrequencyInput [Hz]"),
        "probes":    p.get("probes"),
        "figsize":   p.get("figsize"),
    }

    if extra:
        meta.update(extra)

    return meta


# ── Save figure ───────────────────────────────────────────────────────────────

def _save_figure(fig: plt.Figure, filename: str,
                 save_pdf: bool = True,
                 save_pgf: bool = True) -> list[Path]:
    """
    Save fig as .pdf and/or .pgf to FIGURES_DIR.

    Returns list of saved paths (used by write_figure_stub to populate
    the PANELS AVAILABLE block).
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    saved = []

    if save_pdf:
        p = FIGURES_DIR / f"{filename}.pdf"
        fig.savefig(p)
        saved.append(p)
        print(f"  Saved: {p}")

    if save_pgf:
        p = FIGURES_DIR / f"{filename}.pgf"
        fig.savefig(p)
        saved.append(p)
        print(f"  Saved: {p}")

    return saved


# ── Stub writer ───────────────────────────────────────────────────────────────

def _build_includegraphics(filename: str, width: str = r"\linewidth") -> str:
    return (
        f"    \\includegraphics[width={width}]"
        f"{{FIGURES/{filename}.pdf}}"
    )


def _build_subfigure_block(filename: str, label_suffix: str,
                           width: str = "0.48") -> str:
    return (
        f"  \\begin{{subfigure}}[b]{{{width}\\linewidth}}\n"
        f"    \\centering\n"
        f"    \\includegraphics[width=\\linewidth]{{FIGURES/{filename}.pdf}}\n"
        f"    \\caption{{TODO}}\n"
        f"    \\label{{fig:TODO_{label_suffix}}}\n"
        f"  \\end{{subfigure}}"
    )


def write_figure_stub(meta: dict, plot_type: str,
                      panel_filenames: Optional[list[str]] = None,
                      force: bool = False) -> None:
    """
    Write a LaTeX figure stub to TEXFIGU_DIR.

    The stub is created ONCE. Re-running the plot script will NOT overwrite
    your edited caption — unless force=True.

    Parameters
    ----------
    meta : dict
        As returned by build_fig_meta().
    plot_type : str
        e.g. 'timeseries', 'psd', 'scatter_p2p3'
    panel_filenames : list[str], optional
        Filenames (without extension) of the individual panel PDFs.
        - 1 file  → single \includegraphics layout
        - 2+ files → subfigure layout
        If None, a single-panel stub is generated from build_filename().
    force : bool
        Overwrite an existing stub (wipes your caption edits — use carefully).
    """
    TEXFIGU_DIR.mkdir(parents=True, exist_ok=True)

    stub_filename = build_filename(plot_type, meta)
    tex_path = TEXFIGU_DIR / f"{stub_filename}.tex"

    if tex_path.exists() and not force:
        print(f"  Stub exists (not overwriting): {tex_path.name}")
        return

    # ── Immutable comment block ───────────────────────────────────────────────
    def _fmt_meta_line(key, val):
        if isinstance(val, list):
            val = ", ".join(str(v) for v in val)
        return f"%   {key:<16}: {val}"

    immutable_lines = [
        "%! TEX root = ../main.tex",
        "% " + "=" * 60,
        "% IMMUTABLE — do not edit below this block",
        f"%   generated_by    : {meta.get('script', 'plotter.py')}",
        f"%   plot_type       : {plot_type}",
        _fmt_meta_line("chapter",   meta.get("chapter", "?")),
        _fmt_meta_line("panel",     meta.get("panel", "?")),
        _fmt_meta_line("wind",      meta.get("wind", "?")),
        _fmt_meta_line("amplitude", meta.get("amplitude", "?")),
        _fmt_meta_line("frequency", meta.get("frequency", "?")),
        _fmt_meta_line("probes",    meta.get("probes", "?")),
    ]

    # Optional extra keys (run_id, raw_data, etc.)
    known = {"chapter","panel","wind","amplitude","frequency","probes",
             "figsize","script"}
    for k, v in meta.items():
        if k not in known:
            immutable_lines.append(_fmt_meta_line(k, v))

    # Panel file list
    if panel_filenames:
        immutable_lines.append("%")
        immutable_lines.append("% PANELS AVAILABLE:")
        for pf in panel_filenames:
            immutable_lines.append(f"%   FIGURES/{pf}.pdf")

    immutable_lines.append("% " + "=" * 60)
    immutable_lines.append("")

    # ── Figure body ───────────────────────────────────────────────────────────
    panels = panel_filenames or [stub_filename]

    if len(panels) == 1:
        figure_body = (
            "\\begin{figure}[htbp]\n"
            "  \\centering\n"
            f"  {_build_includegraphics(panels[0])}\n"
            "  \\caption[Short caption for LOF]{\n"
            "    % TODO: write caption\n"
            "  }\n"
            f"  \\label{{fig:TODO_{panels[0][-20:]}}}\n"
            "\\end{figure}"
        )
    else:
        subfigs = []
        for i, pf in enumerate(panels):
            suffix = f"probe{_label_probe(pf, i)}"
            subfigs.append(_build_subfigure_block(pf, suffix))
        joiner = "\n  \\hfill\n"
        figure_body = (
            "\\begin{figure}[htbp]\n"
            "  \\centering\n"
            + joiner.join(subfigs) + "\n"
            "  \\caption[Short caption for LOF]{\n"
            "    % TODO: write caption\n"
            "  }\n"
            f"  \\label{{fig:TODO_{stub_filename[-30:]}}}\n"
            "\\end{figure}"
        )

    stub = "\n".join(immutable_lines) + figure_body + "\n"
    tex_path.write_text(stub, encoding="utf-8")
    print(f"  Stub created: {tex_path.name}")


def _label_probe(filename: str, fallback_idx: int) -> str:
    """Extract probe number from filename for label suffix."""
    import re
    m = re.search(r"probe(\d+)", filename)
    return m.group(1) if m else str(fallback_idx + 1)


# ── Combined entry point (call this from every plotter function) ───────────────

def save_and_stub(fig: plt.Figure,
                  meta: dict,
                  plot_type: str,
                  panel_filenames: Optional[list[str]] = None,
                  save_pdf: bool = True,
                  save_pgf: bool = True,
                  force_stub: bool = False) -> None:
    """
    Save figure files and write the LaTeX stub in one call.

    This is the single function every plotter function should call
    at the end, when save_plot=True.

    Parameters
    ----------
    fig : plt.Figure
    meta : dict
        From build_fig_meta().
    plot_type : str
        e.g. 'timeseries', 'psd', 'scatter_p2p3'
    panel_filenames : list[str], optional
        If the figure stub should reference multiple separate panel PDFs
        (e.g. probe2 and probe3 saved separately), pass their filenames here.
        If None, the stub references the single figure being saved now.
    save_pdf, save_pgf : bool
        Toggle which formats to save.
    force_stub : bool
        Overwrite existing stub (wipes caption edits).

    Example
    -------
    # At the end of plot_timeseries():
    if plotvariables["plotting"].get("save_plot"):
        meta = build_fig_meta(plotvariables, chapter="05",
                              extra={"script": "analysis_swell.py"})
        save_and_stub(fig, meta, plot_type="timeseries")
    """
    filename = build_filename(plot_type, meta)
    _save_figure(fig, filename, save_pdf=save_pdf, save_pgf=save_pgf)
    write_figure_stub(meta, plot_type,
                      panel_filenames=panel_filenames,
                      force=force_stub)


# ── Label builder (kept here, used by plotter.py) ─────────────────────────────

def make_label(row) -> str:
    """
    Create a short legend label from a metadata row (Series or dict).
    Format: W:full_P:reverse
    """
    parts = []
    wind  = row.get("WindCondition")
    panel = row.get("PanelCondition")
    amp   = row.get("WaveAmplitudeInput [Volt]")
    freq  = row.get("WaveFrequencyInput [Hz]")

    if wind  is not None: parts.append(f"W:{wind}")
    if panel is not None: parts.append(f"P:{panel}")
    if amp   is not None: parts.append(f"A:{amp:.2f}V")
    if freq  is not None: parts.append(f"f:{freq}Hz")

    return "_".join(parts) if parts else "unknown"
