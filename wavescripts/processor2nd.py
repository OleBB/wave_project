#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 10:37:28 2025

@author: ole
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from wavescripts.improved_data_loader import update_processed_metadata, get_configuration_for_date
from typing import Mapping, Any, Optional, Sequence, Dict, Tuple, Iterable
from wavescripts.constants import SIGNAL, RAMP, MEASUREMENT, get_smoothing_window
from wavescripts.constants import (
    ProbeColumns as PC,
    GlobalColumns as GC,
    ColumnGroups as CG,
    CalculationResultColumns as RC
)

# %% Band
def compute_amplitude_by_band(
    psd_dict: Mapping[str, pd.DataFrame],
    *,
    freq_bands: Optional[Dict[str, Tuple[float, float]]] = None,
    probes: Iterable[int] = (1, 2, 3, 4),
    verbose: bool = False,
    integration: str = "sum",          # "sum"  → simple Δf * Σ PSD
                                        # "trapez" → np.trapezoid on the real freq axis
    freq_resolution: Optional[float] = None,
) -> pd.DataFrame:
    """
    Compute wave‑amplitude estimates for a set of frequency bands from PSD data.

    Parameters
    ----------
    psd_dict : mapping of ``path → pd.DataFrame``
        Each DataFrame must be indexed by frequency (Hz) and contain columns
        named ``'Pxx 1'``, ``'Pxx 2'``, … for the different probes.
    freq_bands : dict, optional
        Mapping ``band_name → (f_low, f_high)`` in Hz.  If omitted the
        classic three‑band set is used:

        .. code-block:: python

            {
                "swell":      (1.0, 1.6),
                "wind_waves": (3.0, 10.0),
                "total":      (0.0, 10.0),
            }

    probes : iterable of int, default (1,2,3,4)
        Which probe columns (``'Pxx i'``) to process.
    verbose : bool, default ``False``
        Print a short diagnostic for each file / band (mirrors the second
        version you posted).
    integration : {"sum", "trapez"}, default ``"sum"``
        * ``"sum"`` – assumes a *uniform* frequency spacing and computes the
          variance as ``Δf * Σ PSD``.  This is the fastest option and matches
          the first two snippets.
        * ``"trapez"`` – uses ``np.trapezoid`` on the *actual* frequency axis,
          which is more accurate when the spacing is irregular (third snippet).
    freq_resolution : float, optional
        Explicit frequency resolution (Δf).  If ``None`` and ``integration=="sum"``,
        the function derives Δf from the first two frequency points of each
        DataFrame (the original behaviour).

    Returns
    -------
    pd.DataFrame
        One row per ``path`` with columns

        ``'Probe {i} {band_name} amplitude'``

        containing the peak‑to‑trough amplitude estimate
        $A = 2\sqrt{\mathrm{variance}}$.
    """

    # ----------------------------------------------------------------------
    #  Default frequency‑band definitions (kept from the first two versions)
    # ----------------------------------------------------------------------
    if freq_bands is None:
        freq_bands = {
            "Swell":      (0.0, 2.6),
            "Wind": (2.60000001, 16.0),
            "Total":      (0.0, 16.0),
        }

    # ----------------------------------------------------------------------
    #  Validate the chosen integration method
    # ----------------------------------------------------------------------
    if integration not in {"sum", "trapez"}:
        raise ValueError("integration must be either 'sum' or 'trapz'")

    # ----------------------------------------------------------------------
    #  Main loop over all PSD files (paths)
    # ----------------------------------------------------------------------
    rows = []
    for path, df in psd_dict.items():
        # Store results for this path
        row = {"path": path}

        if verbose:
            print(f"\n=== Path: {path} ===")
            print(f"  Frequency range: {df.index.min():.3f}–{df.index.max():.3f} Hz")
            if integration == "sum":
                # Δf will be derived later; show a placeholder now
                print("  Integration method: sum (Δf * Σ PSD)")

        # ------------------------------------------------------------------
        #1 Determine frequency resolution if needed (only for "sum")
        # ------------------------------------------------------------------
        if integration == "sum":
            # Assume uniform spacing – take the difference of the first two points.
            # If the user supplied an explicit value, honour it.
            if freq_resolution is None:
                # Guard against a single‑point index (unlikely for a PSD)
                if len(df.index) < 2:
                    raise ValueError(f"Not enough frequency points in {path} to infer Δf")
                freq_res = float(df.index[1] - df.index[0])
            else:
                freq_res = float(freq_resolution)

            if verbose:
                print(f"  Frequency resolution Δf: {freq_res:.6f} Hz")

        # ------------------------------------------------------------------
        # Loop over available Pxx columns (position-based: "Pxx 9373/170" etc.)
        # ------------------------------------------------------------------
        pxx_cols = [c for c in df.columns if c.startswith("Pxx ")]

        for col in pxx_cols:
            pos = col[4:]  # strip "Pxx " → "9373/170", "12545", etc.

            for band_name, (f_low, f_high) in freq_bands.items():
                mask = (df.index >= f_low) & (df.index <= f_high)
                n_points = int(mask.sum())

                if n_points == 0:
                    row[f"Probe {pos} {band_name} Amplitude (PSD)"] = 0.0
                    continue

                if integration == "sum":
                    variance = df.loc[mask, col].sum() * freq_res
                else:
                    freqs = df.index.to_numpy(dtype=float)[mask]
                    psd_vals = df.loc[mask, col].to_numpy(dtype=float)
                    variance = np.trapezoid(psd_vals, x=freqs)

                amplitude = 2.0 * np.sqrt(variance)

                if verbose:
                    print(
                        f"  Probe {pos} – {band_name} [{f_low}-{f_high}] Hz: "
                        f"{n_points} pts, amplitude={amplitude:.4f}"
                    )

                row[f"Probe {pos} {band_name} Amplitude (PSD)"] = amplitude

        rows.append(row)

    # ----------------------------------------------------------------------
    #  Convert list‑of‑dicts → DataFrame (preserves column order)
    # ----------------------------------------------------------------------
    return pd.DataFrame(rows)
# %%


def compute_inter_run_timing(meta_df: pd.DataFrame) -> pd.DataFrame:
    """Compute inter-run gaps and preceding-run context within each experiment folder.

    Within a single experiment folder (same parent directory), runs are sorted by
    file modification time — which is when LabVIEW finished writing each CSV. The
    gap between consecutive runs is the time the tank had to settle between recordings.

    Why mtime (not filename timestamp)?
      Filename dates are date-level (YYYYMMDD), not time-level. mtime gives second-
      level resolution of the actual recording order. For runs on the same day this
      is the only reliable ordering signal.

    Adds these columns:
      run_mtime            [float]  Unix timestamp of the file's mtime
      inter_run_gap_s      [float]  Seconds since the preceding run in the same folder.
                                    NaN for the first run of the day.
      prev_run_category    [str]    run_category of the preceding run ("" = first run).
                                    Useful for understanding what the tank was doing just
                                    before this run. E.g. prev_run_category="wind_decay"
                                    means the preceding run was a fan-off decay recording.
      prev_run_wind        [str]    WindCondition of the preceding run ("" = first run).
                                    Tells you whether the tank had wind before this run.
      prev_run_freq_hz     [float]  WaveFrequencyInput of the preceding run (NaN if nowave).
                                    Critical for stillwater recovery: sub-1 Hz waves deposit
                                    far more energy per cycle (longer wavelength, larger
                                    orbital depth) and take significantly longer to decay
                                    than high-frequency waves at the same amplitude.
      prev_run_nperiods    [float]  WavePeriodInput of the preceding run (NaN if nowave).
                                    Proxy for total energy deposited: per40 (short burst)
                                    leaves much less residual than per240 (long steady state).
                                    Combined with freq: sub-1Hz + per240 is the worst case.

    Practical use — stillwater recovery:
      A nowave+nowind run is only trustworthy as a noise floor reference if the tank
      has had time to settle.  Required gap depends on the preceding run:
        - prev nowave:                  no gap needed (water already at rest)
        - prev wave, ≥1 Hz, per40:      ~120 s  (small, short burst — quick decay)
        - prev wave, ≥1 Hz, per240:     ~300 s  (long steady state — more energy)
        - prev wave, sub-1 Hz, per40:   ~300 s  (low-freq orbital depth reaches bottom)
        - prev wave, sub-1 Hz, per240:  ~600 s  (worst case — deep long waves, full tank)
        - prev run had wind (any):      ~720 s  (fromMaxToZeroWin characterisation ~12 min)
      These thresholds live in ensure_stillwater_columns (_SETTLE_GAP_S) in processor.py.

    Note: NON_FLOAT_COLUMNS in improved_data_loader.py lists prev_run_category and
    prev_run_wind as str so apply_dtypes does not coerce them to NaN.
    """
    meta_df = meta_df.copy()

    # Read mtime for each file; fall back to NaN if path is missing or inaccessible
    def _safe_mtime(path: str) -> float:
        try:
            return float(os.path.getmtime(path))
        except (OSError, TypeError, ValueError):
            return float("nan")

    meta_df["run_mtime"] = meta_df["path"].apply(_safe_mtime)

    # Process each folder independently
    meta_df["_folder"] = meta_df["path"].apply(lambda p: str(Path(p).parent))

    for folder, grp in meta_df.groupby("_folder"):
        sorted_idx = grp.sort_values("run_mtime").index
        prev_row = None
        for i, idx in enumerate(sorted_idx):
            if prev_row is None:
                meta_df.at[idx, "inter_run_gap_s"]    = float("nan")
                meta_df.at[idx, "prev_run_category"]  = ""
                meta_df.at[idx, "prev_run_wind"]      = ""
                meta_df.at[idx, "prev_run_freq_hz"]   = float("nan")
                meta_df.at[idx, "prev_run_nperiods"]  = float("nan")
            else:
                meta_df.at[idx, "inter_run_gap_s"]    = (
                    float(meta_df.at[idx, "run_mtime"]) - float(meta_df.at[prev_row, "run_mtime"])
                )
                meta_df.at[idx, "prev_run_category"]  = str(meta_df.at[prev_row, "run_category"] or "")
                meta_df.at[idx, "prev_run_wind"]      = str(meta_df.at[prev_row, "WindCondition"] or "")
                meta_df.at[idx, "prev_run_freq_hz"]   = meta_df.at[prev_row, "WaveFrequencyInput [Hz]"]
                meta_df.at[idx, "prev_run_nperiods"]  = meta_df.at[prev_row, "WavePeriodInput"]
            prev_row = idx

    meta_df = meta_df.drop(columns=["_folder"])
    return meta_df


def _update_more_metrics(
    psd_dict: dict,
    fft_dict: dict,
    meta_sel: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute additional derived metrics (ratios + band amplitudes)
    and add/overwrite them in the metadata.
    """
    # Start from a clean indexed copy
    meta_indexed = meta_sel.set_index("path").copy()

    # Derive cfg once from the folder date
    file_date = datetime.fromisoformat(str(meta_indexed["file_date"].iloc[0]))
    cfg = get_configuration_for_date(file_date)
    col_names = cfg.probe_col_names()  # {1: "9373/170", 2: "12545", ...}

    # Compute OUT/IN (FFT): read in_probe/out_probe directly from table columns
    if "in_probe" in meta_indexed.columns and "out_probe" in meta_indexed.columns:
        out_in      = pd.Series(index=meta_indexed.index, dtype=float)
        in_pos_ser  = pd.Series(index=meta_indexed.index, dtype=object)
        out_pos_ser = pd.Series(index=meta_indexed.index, dtype=object)
        for (in_p, out_p), idx in meta_indexed.groupby(["in_probe", "out_probe"]).groups.items():
            in_pos  = col_names[int(in_p)]
            out_pos = col_names[int(out_p)]
            in_col  = f"Probe {in_pos} Amplitude (FFT)"
            out_col = f"Probe {out_pos} Amplitude (FFT)"
            in_pos_ser.loc[idx]  = in_pos
            out_pos_ser.loc[idx] = out_pos
            if in_col in meta_indexed.columns and out_col in meta_indexed.columns:
                out_in.loc[idx] = (
                    meta_indexed.loc[idx, out_col] / meta_indexed.loc[idx, in_col]
                )
        out_in = out_in.replace([np.inf, -np.inf], np.nan)
        meta_indexed[GC.OUT_IN_FFT]   = out_in
        meta_indexed["in_position"]   = in_pos_ser
        meta_indexed["out_position"]  = out_pos_ser

        # ── Generic IN / OUT columns ─────────────────────────────────
        # Copy position-specific columns into probe-agnostic names so
        # downstream code doesn't need to know which probe was IN/OUT.
        _GENERIC_SUFFIXES = [
            "Amplitude (FFT)",
            "WavePeriod (FFT)",
            "Wavenumber (FFT)",
            "Wavelength (FFT)",
            "ka (FFT)",
            "Celerity (FFT)",
            "Significant Wave Height Hm0",
            "Significant Wave Height Hs",
            "Froude (FFT)",
            "Wind/Celerity (FFT)",
            "f/f_PM (FFT)",
            "Ursell (FFT)",
            "wave_stability",
            "period_amplitude_cv",
        ]
        for suffix in _GENERIC_SUFFIXES:
            in_vals  = pd.Series(index=meta_indexed.index, dtype=float)
            out_vals = pd.Series(index=meta_indexed.index, dtype=float)
            for (in_p, out_p), idx in meta_indexed.groupby(["in_probe", "out_probe"]).groups.items():
                in_pos  = col_names[int(in_p)]
                out_pos = col_names[int(out_p)]
                src_in  = f"Probe {in_pos} {suffix}"
                src_out = f"Probe {out_pos} {suffix}"
                if src_in in meta_indexed.columns:
                    in_vals.loc[idx]  = meta_indexed.loc[idx, src_in]
                if src_out in meta_indexed.columns:
                    out_vals.loc[idx] = meta_indexed.loc[idx, src_out]
            meta_indexed[f"IN {suffix}"]  = in_vals
            meta_indexed[f"OUT {suffix}"] = out_vals

    # ── Parallel probe ratio ─────────────────────────────────────────
    # parallel_ratio = wall-side amplitude / far-side amplitude
    parallel = cfg.parallel_pair()
    if parallel:
        pos_wall, pos_far = parallel
        col_wall = f"Probe {pos_wall} Amplitude"
        col_far  = f"Probe {pos_far} Amplitude"
        if col_wall in meta_indexed.columns and col_far in meta_indexed.columns:
            ratio = meta_indexed[col_wall] / meta_indexed[col_far]
            meta_indexed["parallel_ratio"] = ratio.replace([np.inf, -np.inf], np.nan)

    # ── Band amplitudes ──────────────────────────────────────────────
    # Assuming compute_amplitude_by_band returns a DataFrame with "path" column
    band_amplitudes = compute_amplitude_by_band(psd_dict)

    if not band_amplitudes.empty:
        # Set same index and select only the band columns you want to add
        band_indexed = band_amplitudes.set_index("path")

        # Option A: aggressive overwrite of whatever columns come back
        meta_indexed[band_indexed.columns] = band_indexed

        # Option B: more controlled — only specific columns
        # band_cols = [c for c in band_indexed.columns if "band" in c.lower()]  # example
        # meta_indexed[band_cols] = band_indexed[band_cols]

    # You can add more blocks here later (e.g. using fft_dict)

    # Return to normal shape
    return meta_indexed.reset_index(names="path")



# %% kjøres
from wavescripts.processor import _set_output_folder
def process_processed_data(
        psd_dict: dict,
        fft_dict: dict,
        meta_sel: pd.DataFrame,
        meta_full: pd.DataFrame, #trenger kanskje ikke denne, men _set_output_folder vil ha den.
        processvariables: dict
) -> pd.DataFrame:
    """
    Forklaring:
        nu kjører vi funksjoner som krever en df som allerede har verdier for
        alle probene
    Returns:
        oppgradert meta_data_df
    Saves:
        oppdaterer json-filen
    """
    prosessering =  processvariables.get("prosessering",{})
    debug = prosessering.get("debug", False)
    force_recompute =prosessering.get("force_recompute", False)
    if debug:
        print("kjører process_processed_data fra processsor2nd.py")

    meta_sel = _update_more_metrics(psd_dict, fft_dict, meta_sel)

    # Inter-run timing: gap to previous run, preceding-run context
    meta_sel = compute_inter_run_timing(meta_sel)

    meta_sel = _set_output_folder(meta_sel, meta_full, debug)
    """VIKTIG - denne oppdaterer .JSON-filen"""
    update_processed_metadata(meta_sel, force_recompute=force_recompute)

    return meta_sel
