#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_quicklook.py
=================
Exploration and interactive tools — NOT for thesis output.

Functions here:
  - Never have save_plot options
  - Never write to output/FIGURES/ or output/TEXFIGU/
  - Are for interactive inspection, comparison, and debugging

Contents
--------
SEABORN EXPLORATION     explore_damping_vs_freq, explore_damping_vs_amp
INTERACTIVE BROWSERS    SignalBrowserFiltered (Qt), RampDetectionBrowser (Qt)
INTERACTIVE EXPORT      save_interactive_plot (Plotly HTML)
DEVELOPER TOOLS         plot_all_markers, plot_rgb
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import plotly.express as px

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QListWidget,
    QVBoxLayout, QWidget, QLabel,
)

from wavescripts.constants import GlobalColumns as GC
from wavescripts.plot_utils import WIND_COLOR_MAP, MARKERS
from wavescripts.plotter import plot_reconstructed


# ═══════════════════════════════════════════════════════════════════════════════
# SEABORN EXPLORATION
# ═══════════════════════════════════════════════════════════════════════════════

def explore_damping_vs_freq(df: pd.DataFrame,
                             plotvariables: dict) -> None:
    """
    Seaborn facet: P3/P2 vs frequency, one column per amplitude.
    Exploration only — not saveable as individual thesis panels.
    Use plot_damping_freq() for thesis output.
    """
    x = GC.WAVE_FREQUENCY_INPUT
    g = sns.relplot(
        data=df.sort_values(x),
        x=x, y="mean_P3P2",
        hue=GC.WIND_CONDITION, palette=WIND_COLOR_MAP,
        style=GC.PANEL_CONDITION_GROUPED, style_order=["no", "all"],
        col=GC.WAVE_AMPLITUDE_INPUT,
        kind="line", marker=True,
        facet_kws={"sharex": True, "sharey": True},
        height=3.0, aspect=1.2, errorbar=None,
    )
    for ax, (amp, sub) in zip(g.axes.flat,
                               df.groupby(GC.WAVE_AMPLITUDE_INPUT)):
        for (wind, panel), gsub in sub.groupby(
                [GC.WIND_CONDITION, GC.PANEL_CONDITION_GROUPED]):
            ax.errorbar(gsub[x], gsub["mean_P3P2"], yerr=gsub["std_P3P2"],
                        fmt="none", capsize=3, alpha=0.5)
    sns.move_legend(g, "lower center",
                    bbox_to_anchor=(0.5, 1), ncol=3,
                    title=None, frameon=False)
    g.figure.suptitle("Damping P3/P2 vs Frequency  [quicklook]",
                       y=1.04, fontsize=11)
    plt.tight_layout()
    plt.show()


def explore_damping_vs_amp(df: pd.DataFrame,
                            plotvariables: dict) -> None:
    """
    Seaborn facet: P3/P2 vs amplitude, one column per frequency.
    Exploration only — use plot_damping_freq() for thesis output.
    """
    x = GC.WAVE_AMPLITUDE_INPUT
    sns.set_style("ticks", {"axes.grid": True})
    g = sns.relplot(
        data=df.sort_values(x),
        x=x, y="mean_P3P2",
        hue=GC.WIND_CONDITION, palette=WIND_COLOR_MAP,
        style=GC.PANEL_CONDITION_GROUPED, style_order=["no", "all"],
        col=GC.WAVE_FREQUENCY_INPUT,
        kind="line", marker=True,
        facet_kws={"sharex": True, "sharey": True},
        height=3.0, aspect=1.2, errorbar=None,
    )
    for ax, (freq, sub) in zip(g.axes.flat,
                                df.groupby(GC.WAVE_FREQUENCY_INPUT)):
        for (wind, panel), gsub in sub.groupby(
                [GC.WIND_CONDITION, GC.PANEL_CONDITION_GROUPED]):
            ax.errorbar(gsub[x], gsub["mean_P3P2"], yerr=gsub["std_P3P2"],
                        fmt="none", capsize=3, alpha=0.5)
    sns.move_legend(g, "lower center",
                    bbox_to_anchor=(0.5, 1), ncol=3,
                    title=None, frameon=False)
    g.figure.suptitle("Damping P3/P2 vs Amplitude  [quicklook]",
                       y=1.04, fontsize=11)
    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE BROWSERS (Qt)
# ═══════════════════════════════════════════════════════════════════════════════

class SignalBrowserFiltered(QMainWindow):
    """
    Qt browser for reconstructed signal inspection.
    Select experiments from a filterable list; click to plot via plot_reconstructed().

    Usage
    -----
    app = QApplication.instance() or QApplication(sys.argv)
    browser = SignalBrowserFiltered(filtered_fft_dict, filtered_meta, freqplotvariables)
    browser.show()
    """

    def __init__(self, fft_dict: dict, meta_df: pd.DataFrame, plotvars: dict):
        super().__init__()
        self.fft_dict = fft_dict
        self.meta_df  = meta_df
        self.plotvars = copy.deepcopy(plotvars)
        self.setWindowTitle("Signal Browser")
        self.setGeometry(100, 100, 550, 900)

        from PyQt5.QtWidgets import (QComboBox, QHBoxLayout, QCheckBox,
                                      QGroupBox, QGridLayout, QSlider)
        from PyQt5.QtCore import Qt

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # ── Filters ───────────────────────────────────────────────────────────
        filter_box    = QGroupBox("Data Filters")
        filter_layout = QGridLayout()

        self.wind_filter  = QComboBox()
        self.panel_filter = QComboBox()
        self.freq_filter  = QComboBox()
        self.amp_filter   = QComboBox()

        self.wind_filter.addItems(
            ["All wind"] + sorted(meta_df["WindCondition"].dropna().unique().tolist()))
        self.panel_filter.addItems(
            ["All panel"] + sorted(meta_df["PanelCondition"].dropna().unique().tolist()))
        self.freq_filter.addItems(
            ["All freq"] + [str(f) for f in
                            sorted(meta_df["WaveFrequencyInput [Hz]"].dropna().unique())])
        self.amp_filter.addItems(
            ["All amp"] + [str(a) for a in
                           sorted(meta_df["WaveAmplitudeInput [Volt]"].dropna().unique())])

        filter_layout.addWidget(QLabel("Wind:"),   0, 0); filter_layout.addWidget(self.wind_filter,  0, 1)
        filter_layout.addWidget(QLabel("Panel:"),  0, 2); filter_layout.addWidget(self.panel_filter, 0, 3)
        filter_layout.addWidget(QLabel("Freq:"),   1, 0); filter_layout.addWidget(self.freq_filter,  1, 1)
        filter_layout.addWidget(QLabel("Amp:"),    1, 2); filter_layout.addWidget(self.amp_filter,   1, 3)
        filter_box.setLayout(filter_layout)
        layout.addWidget(filter_box)

        # ── Plot options ──────────────────────────────────────────────────────
        plot_box    = QGroupBox("Plot Options")
        plot_layout = QGridLayout()

        plot_layout.addWidget(QLabel("Probes:"), 0, 0)
        probe_row = QHBoxLayout()
        self.probe_checks = {}
        current_probes = self.plotvars.get("plotting", {}).get("probes", [2, 3])
        for p in [1, 2, 3, 4]:
            cb = QCheckBox(f"P{p}")
            cb.setChecked(p in current_probes)
            self.probe_checks[p] = cb
            probe_row.addWidget(cb)
        probe_widget = QWidget()
        probe_widget.setLayout(probe_row)
        plot_layout.addWidget(probe_widget, 0, 1, 1, 3)

        self.dual_yaxis_check = QCheckBox("Dual Y-axis")
        self.dual_yaxis_check.setChecked(
            self.plotvars.get("plotting", {}).get("dual_yaxis", True))
        plot_layout.addWidget(self.dual_yaxis_check, 1, 0, 1, 2)

        self.full_signal_check = QCheckBox("Show Full Signal")
        self.full_signal_check.setChecked(
            self.plotvars.get("plotting", {}).get("show_full_signal", False))
        plot_layout.addWidget(self.full_signal_check, 1, 2, 1, 2)

        self.facet_probe_check = QCheckBox("Facet by Probe")
        self.facet_probe_check.setChecked(
            self.plotvars.get("plotting", {}).get("facet_by") == "probe")
        plot_layout.addWidget(self.facet_probe_check, 2, 0, 1, 2)

        self.amp_stats_check = QCheckBox("Show Amplitude Stats")
        self.amp_stats_check.setChecked(
            self.plotvars.get("plotting", {}).get("show_amplitude_stats", True))
        plot_layout.addWidget(self.amp_stats_check, 2, 2, 1, 2)

        plot_layout.addWidget(QLabel("Linewidth:"), 3, 0)
        self.lw_slider = QSlider(Qt.Horizontal)
        self.lw_slider.setMinimum(1); self.lw_slider.setMaximum(30)
        self.lw_slider.setValue(
            int(self.plotvars.get("plotting", {}).get("linewidth", 1.0) * 10))
        self.lw_label = QLabel(f"{self.lw_slider.value() / 10:.1f}")
        self.lw_slider.valueChanged.connect(
            lambda v: self.lw_label.setText(f"{v/10:.1f}"))
        plot_layout.addWidget(self.lw_slider, 3, 1, 1, 2)
        plot_layout.addWidget(self.lw_label, 3, 3)

        plot_box.setLayout(plot_layout)
        layout.addWidget(plot_box)

        # ── List ──────────────────────────────────────────────────────────────
        self.count_label = QLabel()
        layout.addWidget(self.count_label)
        self.list_widget = QListWidget()
        self.list_widget.currentRowChanged.connect(self.on_select)
        layout.addWidget(self.list_widget)

        for w in [self.wind_filter, self.panel_filter,
                  self.freq_filter, self.amp_filter]:
            w.currentTextChanged.connect(self.update_list)
        self.update_list()

    def get_selected_probes(self):
        return [p for p, cb in self.probe_checks.items() if cb.isChecked()]

    def update_list(self):
        df = self.meta_df.copy()
        wind  = self.wind_filter.currentText()
        panel = self.panel_filter.currentText()
        freq  = self.freq_filter.currentText()
        amp   = self.amp_filter.currentText()
        if wind  != "All wind":  df = df[df["WindCondition"] == wind]
        if panel != "All panel": df = df[df["PanelCondition"] == panel]
        if freq  != "All freq":  df = df[df["WaveFrequencyInput [Hz]"] == float(freq)]
        if amp   != "All amp":   df = df[df["WaveAmplitudeInput [Volt]"] == float(amp)]
        df = df[df["path"].isin(self.fft_dict.keys())]

        self.list_widget.clear()
        self.current_paths = []
        for _, row in df.iterrows():
            path = row["path"]
            self.list_widget.addItem(
                f"{str(row.get('WindCondition','?')):8s} | "
                f"{str(row.get('PanelCondition','?')):8s} | "
                f"{row.get('WaveFrequencyInput [Hz]','?')} Hz | "
                f"{row.get('WaveAmplitudeInput [Volt]','?')} V | "
                f"{Path(path).stem[-30:]}"
            )
            self.current_paths.append(path)
        self.count_label.setText(f"Showing {len(self.current_paths)} experiments")

    def on_select(self, row_idx):
        if row_idx < 0 or row_idx >= len(self.current_paths):
            return
        path        = self.current_paths[row_idx]
        single_meta = self.meta_df[self.meta_df["path"] == path]
        if single_meta.empty:
            return

        plotvars = copy.deepcopy(self.plotvars)
        p = plotvars.setdefault("plotting", {})
        p["probes"]               = self.get_selected_probes()
        p["dual_yaxis"]           = self.dual_yaxis_check.isChecked()
        p["show_full_signal"]     = self.full_signal_check.isChecked()
        p["facet_by"]             = "probe" if self.facet_probe_check.isChecked() else None
        p["show_amplitude_stats"] = self.amp_stats_check.isChecked()
        p["linewidth"]            = self.lw_slider.value() / 10
        p["grid"]                 = True
        p["show_plot"]            = True
        p["save_plot"]            = False   # browser never saves

        plt.close("all")
        plot_reconstructed({path: self.fft_dict[path]}, single_meta, plotvars)


class RampDetectionBrowser(QMainWindow):
    """
    Qt browser for stepping through ramp detection results.
    Feed it the output of gather_ramp_data().

    Usage
    -----
    ramp_df = gather_ramp_data(combined_processed_dfs, combined_meta_sel)
    app = QApplication.instance() or QApplication(sys.argv)
    browser = RampDetectionBrowser(ramp_df)
    browser.show()
    """

    def __init__(self, ramp_df: pd.DataFrame):
        super().__init__()
        self.ramp_df = ramp_df
        self.setWindowTitle("Ramp Detection Browser")
        self.setGeometry(100, 100, 600, 900)

        from PyQt5.QtWidgets import (QComboBox, QGroupBox,
                                      QGridLayout, QDoubleSpinBox)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # ── Filters ───────────────────────────────────────────────────────────
        filter_box    = QGroupBox("Data Filters")
        filter_layout = QGridLayout()

        self.wind_filter  = QComboBox()
        self.wind_filter.addItems(
            ["All wind"] + sorted(ramp_df[GC.WIND_CONDITION].dropna().unique().tolist()))
        self.panel_filter = QComboBox()
        self.panel_filter.addItems(
            ["All panel"] + sorted(ramp_df[GC.PANEL_CONDITION].dropna().unique().tolist()))
        self.freq_filter = QComboBox()
        self.freq_filter.addItems(
            ["All freq"] + [str(f) for f in
                            sorted(ramp_df[GC.WAVE_FREQUENCY_INPUT].dropna().unique())])
        self.amp_filter = QComboBox()
        self.amp_filter.addItems(
            ["All amp"] + [str(a) for a in
                           sorted(ramp_df[GC.WAVE_AMPLITUDE_INPUT].dropna().unique())])
        self.probe_filter = QComboBox()
        self.probe_filter.addItems(
            ["All probes"] + [f"Probe {p}" for p in sorted(ramp_df["probe"].unique())])
        self.probe_filter.setCurrentText("Probe 2")

        for row_i, (lbl, widget) in enumerate([
            ("Wind:", self.wind_filter), ("Panel:", self.panel_filter),
            ("Freq:", self.freq_filter), ("Amp:",   self.amp_filter),
            ("Probe:", self.probe_filter),
        ]):
            filter_layout.addWidget(QLabel(lbl),  row_i // 2, (row_i % 2) * 2)
            filter_layout.addWidget(widget,        row_i // 2, (row_i % 2) * 2 + 1)

        filter_box.setLayout(filter_layout)
        layout.addWidget(filter_box)

        # ── Zoom ──────────────────────────────────────────────────────────────
        plot_box    = QGroupBox("Plot Options")
        plot_layout = QGridLayout()
        plot_layout.addWidget(QLabel("Zoom margin [mm]:"), 0, 0)
        self.zoom_spin = QDoubleSpinBox()
        self.zoom_spin.setRange(1.0, 500.0)
        self.zoom_spin.setValue(30.0)
        self.zoom_spin.setSingleStep(5.0)
        plot_layout.addWidget(self.zoom_spin, 0, 1)
        plot_box.setLayout(plot_layout)
        layout.addWidget(plot_box)

        # ── List ──────────────────────────────────────────────────────────────
        self.count_label = QLabel()
        layout.addWidget(self.count_label)
        self.list_widget = QListWidget()
        self.list_widget.currentRowChanged.connect(self.on_select)
        layout.addWidget(self.list_widget)

        for w in [self.wind_filter, self.panel_filter,
                  self.freq_filter, self.amp_filter, self.probe_filter]:
            w.currentTextChanged.connect(self.update_list)
        self.update_list()

    def update_list(self):
        df = self.ramp_df.copy()
        wind  = self.wind_filter.currentText()
        panel = self.panel_filter.currentText()
        freq  = self.freq_filter.currentText()
        amp   = self.amp_filter.currentText()
        probe = self.probe_filter.currentText()
        if wind  != "All wind":   df = df[df[GC.WIND_CONDITION] == wind]
        if panel != "All panel":  df = df[df[GC.PANEL_CONDITION] == panel]
        if freq  != "All freq":   df = df[df[GC.WAVE_FREQUENCY_INPUT] == float(freq)]
        if amp   != "All amp":    df = df[df[GC.WAVE_AMPLITUDE_INPUT] == float(amp)]
        if probe != "All probes": df = df[df["probe"] == int(probe.split()[-1])]

        self.list_widget.clear()
        self.current_rows = []
        for _, row in df.iterrows():
            self.list_widget.addItem(
                f"P{row['probe']} | "
                f"{str(row[GC.WIND_CONDITION]):8s} | "
                f"{str(row[GC.PANEL_CONDITION]):8s} | "
                f"{row[GC.WAVE_FREQUENCY_INPUT]:.2f} Hz | "
                f"{row[GC.WAVE_AMPLITUDE_INPUT]:.1f} V | "
                f"{row['experiment'][-35:]}"
            )
            self.current_rows.append(row)
        self.count_label.setText(f"Showing {len(self.current_rows)} rows")

    def on_select(self, row_idx):
        if row_idx < 0 or row_idx >= len(self.current_rows):
            return
        from wavescripts.plotter import plot_ramp_detection
        row  = self.current_rows[row_idx]
        zoom = self.zoom_spin.value()

        dummy_dates = pd.to_datetime(row["time_ms"], unit="ms")
        df_plot = pd.DataFrame({"Date": dummy_dates, row["data_col"]: row["raw"]})

        plt.close("all")
        fig, ax = plot_ramp_detection(
            df=df_plot,
            meta_sel=pd.Series({
                GC.PATH:               row[GC.PATH],
                GC.WIND_CONDITION:     row[GC.WIND_CONDITION],
                GC.PANEL_CONDITION:    row[GC.PANEL_CONDITION],
                GC.WAVE_FREQUENCY_INPUT: row[GC.WAVE_FREQUENCY_INPUT],
                GC.WAVE_AMPLITUDE_INPUT: row[GC.WAVE_AMPLITUDE_INPUT],
            }),
            data_col=row["data_col"],
            signal=row["signal"],
            baseline_mean=row["baseline_mean"],
            threshold=row["threshold"],
            first_motion_idx=row["first_motion_idx"],
            good_start_idx=row["good_start_idx"],
            good_range=row["good_range"],
            good_end_idx=row["good_end_idx"],
        )
        ax.set_ylim(row["baseline_mean"] - zoom, row["baseline_mean"] + zoom)
        plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE EXPORT
# ═══════════════════════════════════════════════════════════════════════════════

def save_interactive_plot(df: pd.DataFrame,
                           filename: str = "damping_analysis.html") -> None:
    """Save an interactive Plotly HTML for sharing / exploring in a browser."""
    fig = px.line(
        df,
        x="WaveFrequencyInput [Hz]",
        y="mean_P3P2",
        color=GC.WIND_CONDITION,
        color_discrete_map=WIND_COLOR_MAP,
        error_y="std_P3P2",
        hover_data=["WaveFrequencyInput [Hz]", "WaveAmplitudeInput [Volt]"],
        title="Interactive Damping Analysis",
        markers=True,
    )
    fig.write_html(filename)
    print(f"Interactive plot saved: {filename}")


# ═══════════════════════════════════════════════════════════════════════════════
# DEVELOPER TOOLS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_all_markers() -> None:
    """Visual reference sheet for all matplotlib marker styles."""
    n_cols = 6
    n_rows = (len(MARKERS) + n_cols - 1) // n_cols
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, marker in enumerate(MARKERS):
        x = (i % n_cols) * 2
        y = -(i // n_cols) * 2
        ax.plot(x, y, marker=marker, markersize=20,
                color="red", markeredgecolor="black", markeredgewidth=2)
        ax.text(x, y - 0.6, f"'{marker}'",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_xlim(-1, n_cols * 2)
    ax.set_ylim(-n_rows * 2, 1)
    ax.axis("off")
    ax.set_title("Matplotlib Marker Styles", fontsize=16, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.show()


def plot_rgb() -> None:
    """Visual comparison of wind condition colour palettes."""
    palettes = {
        "Current (D3-inspired)":   [WIND_COLOR_MAP["full"],
                                     WIND_COLOR_MAP["lowest"],
                                     WIND_COLOR_MAP["no"]],
        "Standard Science (D3)":   ["#d62728", "#2ca02c", "#1f77b4"],
        "High-Visibility (Indigo)": ["#E31A1C", "#33A02C", "#3F51B5"],
    }
    x = np.linspace(0, 10, 200)
    plt.figure(figsize=(12, 8))
    for i, (name, colors) in enumerate(palettes.items()):
        offset = i * 2.5
        for j, (label, color) in enumerate(
                zip(["Full", "Lowest", "No"], colors)):
            plt.plot(x, np.sin(x + j * 0.5) + offset - j * 0.5,
                     color=color, lw=3, label=f"{name} — {label}")
    plt.title("Wind Condition Palette Comparison", fontsize=15)
    plt.yticks([])
    plt.xlabel("X-axis")
    plt.grid(True, axis="x", alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_all_markers()
    plot_rgb()
