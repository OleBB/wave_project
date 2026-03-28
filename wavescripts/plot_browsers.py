#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_browsers.py
================
Qt-based interactive browsers — run from terminal via main_explore_browser.py.

NOT imported by main_explore_inline.py (REPL). PyQt5 is only loaded when this
module is explicitly imported, keeping REPL startup fast.

Contents
--------
SignalBrowserFiltered   — step through FFT signal reconstruction run by run
RampDetectionBrowser    — inspect ramp detection results for all runs
"""




from __future__ import annotations

import copy
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QListWidget,
    QVBoxLayout, QWidget, QLabel, QLineEdit,
)

from wavescripts.constants import GlobalColumns as GC
from wavescripts.plotter import plot_reconstructed, _build_sw_fits


def _resize_to_fraction(fig, fraction: float = 0.75) -> None:
    """Resize a matplotlib Qt figure window to `fraction` of the primary screen."""
    try:
        screen = QApplication.primaryScreen().availableGeometry()
        w = int(screen.width() * fraction)
        h = int(screen.height() * fraction)
        fig.canvas.manager.window.resize(w, h)
    except Exception:
        pass  # non-Qt backend or headless — silently skip


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

    def __init__(self, fft_dict: dict, meta_df: pd.DataFrame, plotvars: dict,
                 processed_dfs: dict | None = None):
        super().__init__()
        self.fft_dict      = fft_dict
        self.meta_df       = meta_df
        self.plotvars      = copy.deepcopy(plotvars)
        self.processed_dfs = processed_dfs  # optional: enables eta/residue overlays
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
        self.per_filter   = QComboBox()

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
        self.per_filter.addItems(
            ["All per"] + [str(a) for a in
                           sorted(meta_df["WavePeriodInput"].dropna().unique())])

        filter_layout.addWidget(QLabel("WindConditions:"),   0, 0); filter_layout.addWidget(self.wind_filter,  0, 1)
        filter_layout.addWidget(QLabel("PanelConditions:"),  0, 2); filter_layout.addWidget(self.panel_filter, 0, 3)
        filter_layout.addWidget(QLabel("Frequencies:"),   1, 0); filter_layout.addWidget(self.freq_filter,  1, 1)
        filter_layout.addWidget(QLabel("Amplitudes:"),    1, 2); filter_layout.addWidget(self.amp_filter,   1, 3)
        filter_layout.addWidget(QLabel("Periods:"),    2, 0); filter_layout.addWidget(self.per_filter,   2, 1)
        filter_box.setLayout(filter_layout)
        layout.addWidget(filter_box)

        # ── Plot options ──────────────────────────────────────────────────────
        plot_box    = QGroupBox("Plot Options")
        plot_layout = QGridLayout()

        plot_layout.addWidget(QLabel("Probes:"), 0, 0)
        probe_row = QHBoxLayout()
        self.probe_checks = {}
        current_probes = self.plotvars.get("plotting", {}).get("probes", [])
        # Derive position strings from fft_dict columns (e.g. "FFT 12545" → "12545")
        _sample_df = next(iter(fft_dict.values())) if fft_dict else None
        _all_positions = (
            [c[4:] for c in _sample_df.columns if c.startswith("FFT ") and "complex" not in c]
            if _sample_df is not None else []
        )
        for pos in _all_positions:
            cb = QCheckBox(pos)
            cb.setChecked(pos in current_probes)
            self.probe_checks[pos] = cb
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

        # ── Signal overlay toggles (require processed_dfs) ───────────────────
        overlay_label = QLabel("Overlays (needs processed_dfs):")
        plot_layout.addWidget(overlay_label, 3, 0, 1, 4)
        self.true_eta_check = QCheckBox("True η (NaN gaps)")
        self.true_eta_check.setChecked(False)
        plot_layout.addWidget(self.true_eta_check, 4, 0, 1, 2)
        self.repaired_eta_check = QCheckBox("Repaired η (interp)")
        self.repaired_eta_check.setChecked(False)
        plot_layout.addWidget(self.repaired_eta_check, 4, 2, 1, 2)
        self.expected_sine_check = QCheckBox("Expected sine")
        self.expected_sine_check.setChecked(False)
        plot_layout.addWidget(self.expected_sine_check, 5, 0, 1, 2)
        self.residue_check = QCheckBox("Residue (η − sine)")
        self.residue_check.setChecked(False)
        plot_layout.addWidget(self.residue_check, 5, 2, 1, 2)

        plot_layout.addWidget(QLabel("Linewidth:"), 6, 0)
        self.lw_slider = QSlider(Qt.Horizontal)
        self.lw_slider.setMinimum(1); self.lw_slider.setMaximum(30)
        self.lw_slider.setValue(
            int(self.plotvars.get("plotting", {}).get("linewidth", 1.0) * 10))
        self.lw_label = QLabel(f"{self.lw_slider.value() / 10:.1f}")
        self.lw_slider.valueChanged.connect(
            lambda v: self.lw_label.setText(f"{v/10:.1f}"))
        plot_layout.addWidget(self.lw_slider, 6, 1, 1, 2)
        plot_layout.addWidget(self.lw_label, 6, 3)

        plot_box.setLayout(plot_layout)
        layout.addWidget(plot_box)

        # ── List ──────────────────────────────────────────────────────────────
        self.count_label = QLabel()
        layout.addWidget(self.count_label)

        # ── Path field (read-only, selectable for copy) ───────────────────────
        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        self.path_edit.setPlaceholderText("CSV path — select a row to populate")
        self.path_edit.setStyleSheet("font-family: monospace; font-size: 10px;")
        layout.addWidget(self.path_edit)

        # ── Info panel ────────────────────────────────────────────────────────
        self.info_label = QLabel("—")
        self.info_label.setStyleSheet(
            "font-family: monospace; font-size: 11px; padding: 4px;"
            "background: #f0f0f0; border: 1px solid #ccc;"
        )
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        self.list_widget = QListWidget()
        self.list_widget.currentRowChanged.connect(self.on_select)
        layout.addWidget(self.list_widget)

        for w in [self.wind_filter, self.panel_filter,
                  self.freq_filter, self.amp_filter, self.per_filter]:
            w.currentTextChanged.connect(self.update_list)

        def _replot():
            self.on_select(self.list_widget.currentRow())
        for cb in [self.true_eta_check, self.repaired_eta_check,
                   self.expected_sine_check, self.residue_check]:
            cb.stateChanged.connect(lambda _: _replot())

        # Pre-compute stillwater regression curves for ΔSW info panel
        _positions = [c[4:] for c in (next(iter(fft_dict.values()), pd.DataFrame())).columns
                      if c.startswith("FFT ") and "complex" not in c]
        self._sw_fits = _build_sw_fits(meta_df, _positions)

        # Replot when any plot-option control changes
        def _replot():
            self.on_select(self.list_widget.currentRow())

        for cb in self.probe_checks.values():
            cb.stateChanged.connect(lambda _: _replot())
        for cb in [self.dual_yaxis_check, self.full_signal_check,
                   self.facet_probe_check, self.amp_stats_check,
                   self.true_eta_check, self.repaired_eta_check,
                   self.expected_sine_check, self.residue_check]:
            cb.stateChanged.connect(lambda _: _replot())
        self.lw_slider.sliderReleased.connect(_replot)

        self.update_list()

    def get_selected_probes(self):
        return [p for p, cb in self.probe_checks.items() if cb.isChecked()]

    def update_list(self):
        df = self.meta_df.copy()
        wind  = self.wind_filter.currentText()
        panel = self.panel_filter.currentText()
        freq  = self.freq_filter.currentText()
        amp   = self.amp_filter.currentText()
        per   = self.per_filter.currentText()
        if wind  != "All wind":  df = df[df["WindCondition"] == wind]
        if panel != "All panel": df = df[df["PanelCondition"] == panel]
        if freq  != "All freq":  df = df[df["WaveFrequencyInput [Hz]"] == float(freq)]
        if amp   != "All amp":   df = df[df["WaveAmplitudeInput [Volt]"] == float(amp)]
        if per   != "All per":   df = df[df["WavePeriodInput"] == float(per)]
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
                f"{row.get('WavePeriodInput','?')} s | "
                f"{Path(path).stem[-30:]}"
            )
            self.current_paths.append(path)
        self.count_label.setText(f"Showing {len(self.current_paths)} experiments")

    def _compute_expected_sine(self, signal, gs, ge, freq, fs):
        """FFT-extract amplitude+phase at target freq from stable window. Returns full-length sine."""
        stable = signal[gs:ge]
        n = len(stable)
        if n < 5:
            return None
        fft_c  = np.fft.rfft(stable - np.nanmean(stable))
        freqs_r = np.fft.rfftfreq(n, d=1.0 / fs)
        best   = int(np.argmin(np.abs(freqs_r - freq)))
        amp    = 2.0 * np.abs(fft_c[best]) / n
        phase  = np.angle(fft_c[best])
        t_all  = np.arange(len(signal)) / fs
        t_gs   = gs / fs
        return amp * np.sin(2 * np.pi * freq * (t_all - t_gs) + phase)

    def _plot_with_overlays(self, path, single_meta, probes,
                             show_true_eta, show_repaired_eta,
                             show_expected_sine, show_residue):
        """Custom plot used when any overlay toggle is on. Shows stable-window signals."""
        from wavescripts.constants import MEASUREMENT, ProbeColumns as PC
        fs = MEASUREMENT.SAMPLING_RATE
        df = self.processed_dfs.get(path)
        if df is None:
            print(f"  processed_dfs has no entry for {path}")
            return None
        meta_row = single_meta.iloc[0]
        freq = float(meta_row.get("WaveFrequencyInput [Hz]", 0) or 0)
        n_probes = len(probes)
        fig, axes = plt.subplots(n_probes, 1, figsize=(15, 4 * n_probes), sharex=True, squeeze=False)
        fig.suptitle(f"{path.split('/')[-1]}  — stable window overlays", fontsize=11)
        for ax, pos in zip(axes[:, 0], probes):
            gs_col = f"Computed Probe {pos} start"
            ge_col = f"Computed Probe {pos} end"
            gs = int(meta_row.get(gs_col, 0) or 0)
            ge = int(meta_row.get(ge_col, len(df)) or len(df))
            t = np.arange(len(df)) / fs
            t_cut = t[gs:ge]
            if show_true_eta and f"eta_{pos}" in df.columns:
                eta = df[f"eta_{pos}"].to_numpy(dtype=float)
                ax.plot(t_cut, eta[gs:ge], color="black", lw=1.2, alpha=0.85, label="True η")
            if show_repaired_eta and f"eta_{pos}_interp" in df.columns:
                eta_i = df[f"eta_{pos}_interp"].to_numpy(dtype=float)
                ax.plot(t_cut, eta_i[gs:ge], color="steelblue", lw=1.0, alpha=0.7, label="Repaired η")
            sine = None
            if (show_expected_sine or show_residue) and freq > 0:
                eta_src = df.get(f"eta_{pos}_interp", df.get(f"eta_{pos}"))
                if eta_src is not None:
                    sig = eta_src.to_numpy(dtype=float)
                    sine = self._compute_expected_sine(sig, gs, ge, freq, fs)
            if show_expected_sine and sine is not None:
                ax.plot(t_cut, sine[gs:ge], color="darkorange", lw=1.5,
                        linestyle="--", alpha=0.9, label="Expected sine")
            if show_residue and sine is not None:
                eta_r = df.get(f"eta_{pos}_interp", df.get(f"eta_{pos}"))
                if eta_r is not None:
                    residue = eta_r.to_numpy(dtype=float)[gs:ge] - sine[gs:ge]
                    ax.plot(t_cut, residue, color="crimson", lw=1.0, alpha=0.8, label="Residue")
            ax.axhline(0, color="blue", linestyle="--", lw=0.8, alpha=0.5)
            ax.set_ylabel(f"η  {pos}  [mm]")
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(True, alpha=0.15)
        axes[-1, 0].set_xlabel("Time [s]")
        fig.tight_layout()
        return fig

    def on_select(self, row_idx):
        if row_idx < 0 or row_idx >= len(self.current_paths):
            return
        path        = self.current_paths[row_idx]
        single_meta = self.meta_df[self.meta_df["path"] == path]
        if single_meta.empty:
            return

        probes = self.get_selected_probes()

        # ── Path field ────────────────────────────────────────────────────────
        self.path_edit.setText(path)

        # ── Info panel ────────────────────────────────────────────────────────
        import os
        meta_row  = single_meta.iloc[0]
        wind_cond = meta_row.get(GC.WIND_CONDITION, "no")
        file_date = meta_row.get(GC.FILE_DATE, "")
        run_mtime = os.path.getmtime(path)

        def _fmt(v, unit="", fmt=".3f"):
            return f"{v:{fmt}} {unit}".strip() if pd.notna(v) else "—"
        lines = []
        for pos in probes:
            ka  = meta_row.get(f"Probe {pos} ka (FFT)", float("nan"))
            per = meta_row.get(f"Probe {pos} WavePeriod (FFT)", float("nan"))
            wl  = meta_row.get(f"Probe {pos} Wavelength (FFT)", float("nan"))
            af  = meta_row.get(f"Probe {pos} Amplitude (FFT)", float("nan"))
            at  = meta_row.get(f"Probe {pos} Amplitude", float("nan"))
            fit = self._sw_fits.get((file_date, pos), {})
            nw_fn = fit.get("no")
            fw_fn = fit.get("full")
            sw_nw  = float(nw_fn(run_mtime))  if nw_fn  else float("nan")
            sw_fw  = float(fw_fn(run_mtime))  if fw_fn  else float("nan")
            sw_ws  = sw_fw - sw_nw
            sw_exp = sw_fw if wind_cond == "full" else sw_nw
            baseline = meta_row.get(f"Stillwater Probe {pos}", float("nan"))
            try:
                sw_d = float(baseline) - sw_exp
                d_str = f"ΔSW={sw_d:+.2f}mm"
            except (TypeError, ValueError):
                d_str = "ΔSW=—"
            try:
                ws_str = f"setup={sw_ws:+.2f}mm"
            except (TypeError, ValueError):
                ws_str = "setup=—"
            # ── tail diagnostics (seiche emergence) ──────────────────────
            tail_a   = meta_row.get(f"mstop_tail_mm_{pos}", float("nan"))
            tail_p5  = meta_row.get(f"tail_uc_period_at_5s_{pos}",  float("nan"))
            tail_p15 = meta_row.get(f"tail_uc_period_at_15s_{pos}", float("nan"))
            tail_p25 = meta_row.get(f"tail_uc_period_at_25s_{pos}", float("nan"))
            tail_clr = meta_row.get(f"tail_clear_s_{pos}",    float("nan"))
            tail_str = (
                f"tail: A₀={_fmt(tail_a,'mm')}  "
                f"T@5s={_fmt(tail_p5,'s')}  "
                f"T@15s={_fmt(tail_p15,'s')}  "
                f"T@25s={_fmt(tail_p25,'s')}  "
                f"clear={_fmt(tail_clr,'s')}"
            )
            lines.append(
                f"{pos:12s}  ka={_fmt(ka)}  T={_fmt(per,'s')}  "
                f"λ={_fmt(wl,'m')}  A(FFT)={_fmt(af,'mm')}  A(TD)={_fmt(at,'mm')}  "
                f"{d_str}  {ws_str}"
            )
            lines.append(f"{'':12s}  {tail_str}")
        self.info_label.setText("\n".join(lines) if lines else "—")
        show_true_eta    = self.true_eta_check.isChecked()
        show_repaired    = self.repaired_eta_check.isChecked()
        show_sine        = self.expected_sine_check.isChecked()
        show_residue     = self.residue_check.isChecked()
        has_overlays = (show_true_eta or show_repaired or show_sine or show_residue) \
                       and self.processed_dfs is not None

        plotvars = copy.deepcopy(self.plotvars)
        p = plotvars.setdefault("plotting", {})
        p["probes"]               = probes
        p["dual_yaxis"]           = self.dual_yaxis_check.isChecked()
        p["show_full_signal"]     = self.full_signal_check.isChecked()
        p["facet_by"]             = "probe" if self.facet_probe_check.isChecked() else None
        p["show_amplitude_stats"] = self.amp_stats_check.isChecked()
        p["linewidth"]            = self.lw_slider.value() / 10
        p["grid"]                 = True
        p["show_plot"]            = True
        p["save_plot"]            = False

        plt.close("all")
        if has_overlays:
            fig = self._plot_with_overlays(path, single_meta, probes,
                                           show_true_eta, show_repaired,
                                           show_sine, show_residue)
        else:
            fig, _ = plot_reconstructed({path: self.fft_dict[path]}, single_meta, plotvars)
        if fig is not None:
            _resize_to_fraction(fig, 0.75)


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
                                      QGridLayout, QDoubleSpinBox, QCheckBox)

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
        self.probe_filter.setCurrentText("All probes")

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
        self.sine_cb = QCheckBox("Show expected sine")
        self.sine_cb.setChecked(False)
        plot_layout.addWidget(self.sine_cb, 1, 0, 1, 2)
        plot_box.setLayout(plot_layout)
        layout.addWidget(plot_box)

        # ── Path field (read-only, selectable for copy) ───────────────────────
        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        self.path_edit.setPlaceholderText("CSV path — select a row to populate")
        self.path_edit.setStyleSheet("font-family: monospace; font-size: 10px;")
        layout.addWidget(self.path_edit)

        # ── Info panel ────────────────────────────────────────────────────────
        self.info_label = QLabel("—")
        self.info_label.setStyleSheet(
            "font-family: monospace; font-size: 11px; padding: 4px;"
            "background: #f0f0f0; border: 1px solid #ccc;"
        )
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        # ── List ──────────────────────────────────────────────────────────────
        self.count_label = QLabel()
        layout.addWidget(self.count_label)
        self.list_widget = QListWidget()
        self.list_widget.currentRowChanged.connect(self.on_select)
        layout.addWidget(self.list_widget)

        for w in [self.wind_filter, self.panel_filter,
                  self.freq_filter, self.amp_filter, self.probe_filter]:
            w.currentTextChanged.connect(self.update_list)

        # Block on_select during init — first plot fires after event loop starts
        self.list_widget.blockSignals(True)
        self.update_list()
        self.list_widget.blockSignals(False)
        # Defer initial selection so the figure renders after the window is shown
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(0, lambda: self.on_select(0))

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
        if probe != "All probes": df = df[df["probe"] == probe.split(None, 1)[-1]]

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

        # ── Path field ────────────────────────────────────────────────────────
        self.path_edit.setText(str(row.get(GC.PATH, "")))

        # ── Info panel ────────────────────────────────────────────────────────
        def _fmt(v, unit="", fmt=".3f"):
            return f"{v:{fmt}} {unit}".strip() if pd.notna(v) else "—"
        wind_cond = row.get(GC.WIND_CONDITION, "no")
        sw_d  = row.get("sw_delta")
        sw_ws = row.get("sw_wind_setup")
        try:
            ref_label = "fullwind fit" if wind_cond == "full" else "nowind fit"
            sw_str = f"ΔSW = {float(sw_d):+.2f} mm vs {ref_label}" if pd.notna(sw_d) else "ΔSW = —"
        except (TypeError, ValueError):
            sw_str = "ΔSW = —"
        try:
            ws_str = f"wind setup = {float(sw_ws):+.2f} mm (probe units)" if pd.notna(sw_ws) else "wind setup = —"
        except (TypeError, ValueError):
            ws_str = "wind setup = —"
        self.info_label.setText(
            f"ka = {_fmt(row.get('ka_fft'))}  |  "
            f"T = {_fmt(row.get('period_fft'), 's')}  |  "
            f"λ = {_fmt(row.get('wavelength_fft'), 'm')}  |  "
            f"A(FFT) = {_fmt(row.get('amp_fft'), 'mm')}  |  "
            f"A(TD) = {_fmt(row.get('amp_td'), 'mm')}\n"
            f"{sw_str}   |   {ws_str}"
        )

        dummy_dates = pd.to_datetime(row["time_ms"], unit="ms")
        df_plot = pd.DataFrame({"Date": dummy_dates, row["data_col"]: row["raw"]})

        # Expected sine: FFT-reconstruct dominant frequency from stable window
        expected_sine = None
        freq = row[GC.WAVE_FREQUENCY_INPUT]
        if self.sine_cb.isChecked() and pd.notna(freq) and float(freq) > 0:
            sig = row["signal"]
            gs = int(row["good_start_idx"])
            ge = int(row["good_end_idx"])
            stable = sig[gs:ge]
            n_stable = len(stable)
            if n_stable > 4:
                from wavescripts.constants import MEASUREMENT
                fs = MEASUREMENT.SAMPLING_RATE
                fft_c = np.fft.rfft(stable - np.nanmean(stable))
                freqs_r = np.fft.rfftfreq(n_stable, d=1.0 / fs)
                best_bin = int(np.argmin(np.abs(freqs_r - float(freq))))
                amp = 2.0 * np.abs(fft_c[best_bin]) / n_stable
                phase = np.angle(fft_c[best_bin])
                # Build full-length sine aligned to sample 0
                t_all = np.arange(len(sig)) / fs
                t_gs = gs / fs
                # No baseline_mean offset — plot is in eta coordinate (centered at 0)
                expected_sine = amp * np.sin(
                    2 * np.pi * float(freq) * (t_all - t_gs) + phase
                )

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
            signal_interp=row.get("signal_interp"),
            expected_sine=expected_sine,
            baseline_mean=row["baseline_mean"],
            threshold=row["threshold"],
            first_motion_idx=row["first_motion_idx"],
            good_start_idx=row["good_start_idx"],
            good_range=row["good_range"],
            good_end_idx=row["good_end_idx"],
            wave_info={
                "ka":          row.get("ka_fft"),
                "T":           row.get("period_fft"),
                "λ":           row.get("wavelength_fft"),
                "A(FFT)":      row.get("amp_fft"),
                "A(TD)":       row.get("amp_td"),
                "sw_delta":    row.get("sw_delta"),
                "sw_wind_setup": row.get("sw_wind_setup"),
                "wind_cond":   row.get(GC.WIND_CONDITION, "no"),
            },
        )
        ax.set_ylim(-zoom, zoom)  # eta coordinate: centered at 0
        if fig is not None:
            _resize_to_fraction(fig, 0.75)
        plt.show(block=False)
