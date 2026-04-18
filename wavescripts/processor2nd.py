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


def compute_inter_run_timing(
    meta_df: pd.DataFrame,
    processed_dfs: Optional[dict] = None,
) -> pd.DataFrame:
    """Compute inter-run gaps and preceding-run context within each experiment folder.

    Ordering and gap calculation
    ----------------------------
    Preferred (when processed_dfs provided):
      Each CSV's "Date" column contains wall-clock timestamps recorded by LabVIEW.
        run_start  = first Date sample of the run
        run_end    = last  Date sample of the run
        inter_run_gap_s = run_start[current] − run_end[previous]
      This is the true settling time — time between the previous recording ending
      and the next one starting (manual save + click-to-start on the wavemaker PC).

    Fallback (no processed_dfs, or path not in processed_dfs):
      Uses mtime (file modification time) for ordering.
      WARNING: gap = mtime[current] − mtime[previous] includes the entire recording
      duration of the current run and is NOT the true settling time.

    Adds these columns:
      run_mtime            [float]  Unix timestamp of the file's mtime (ordering fallback)
      run_start_ts         [float]  Unix timestamp of first CSV sample (NaN if unavailable)
      run_end_ts           [float]  Unix timestamp of last  CSV sample (NaN if unavailable)
      inter_run_gap_s      [float]  True settling time (s): end of prev → start of current.
                                    NaN for the first run of the folder.
      prev_run_category    [str]    run_category of the preceding run ("" = first run).
      prev_run_wind        [str]    WindCondition of the preceding run ("" = first run).
      prev_run_freq_hz     [float]  WaveFrequencyInput of the preceding run (NaN if nowave).
      prev_run_nperiods    [float]  WavePeriodInput of the preceding run (NaN if nowave).

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

    # ── mtime (always computed — used as ordering fallback) ───────────────────
    def _safe_mtime(path: str) -> float:
        try:
            return float(os.path.getmtime(path))
        except (OSError, TypeError, ValueError):
            return float("nan")

    meta_df["run_mtime"] = meta_df["path"].apply(_safe_mtime)

    # ── CSV timestamps (preferred when processed_dfs available) ───────────────
    def _csv_timestamps(path: str):
        """Return (start_ts, end_ts) as Unix floats, or (nan, nan) if unavailable."""
        if processed_dfs is None:
            return float("nan"), float("nan")
        df = processed_dfs.get(path)
        if df is None:
            df = processed_dfs.get(str(Path(path).resolve()))
        if df is None or "Date" not in df.columns or len(df) == 0:
            return float("nan"), float("nan")
        try:
            return float(df["Date"].iloc[0].timestamp()), float(df["Date"].iloc[-1].timestamp())
        except Exception:
            return float("nan"), float("nan")

    ts_pairs = meta_df["path"].map(_csv_timestamps)
    meta_df["run_start_ts"] = ts_pairs.apply(lambda x: x[0])
    meta_df["run_end_ts"]   = ts_pairs.apply(lambda x: x[1])

    # Sort key: CSV start timestamp if available, else mtime
    meta_df["_sort_key"] = meta_df["run_start_ts"].where(
        meta_df["run_start_ts"].notna(), meta_df["run_mtime"]
    )

    # ── Process each folder independently ─────────────────────────────────────
    meta_df["_folder"] = meta_df["path"].apply(lambda p: str(Path(p).parent))

    for folder, grp in meta_df.groupby("_folder"):
        sorted_idx = grp.sort_values("_sort_key").index
        prev_row = None
        for idx in sorted_idx:
            if prev_row is None:
                meta_df.at[idx, "inter_run_gap_s"]   = float("nan")
                meta_df.at[idx, "prev_run_category"] = ""
                meta_df.at[idx, "prev_run_wind"]     = ""
                meta_df.at[idx, "prev_run_freq_hz"]  = float("nan")
                meta_df.at[idx, "prev_run_nperiods"] = float("nan")
            else:
                cur_start = meta_df.at[idx,     "run_start_ts"]
                prev_end  = meta_df.at[prev_row, "run_end_ts"]
                if not (np.isnan(cur_start) or np.isnan(prev_end)):
                    gap = cur_start - prev_end
                else:
                    # Fallback: mtime diff (includes current run duration — not true gap)
                    gap = float(meta_df.at[idx, "run_mtime"]) - float(meta_df.at[prev_row, "run_mtime"])
                meta_df.at[idx, "inter_run_gap_s"]   = gap
                meta_df.at[idx, "prev_run_category"] = str(meta_df.at[prev_row, "run_category"] or "")
                meta_df.at[idx, "prev_run_wind"]     = str(meta_df.at[prev_row, "WindCondition"] or "")
                meta_df.at[idx, "prev_run_freq_hz"]  = meta_df.at[prev_row, "WaveFrequencyInput [Hz]"]
                meta_df.at[idx, "prev_run_nperiods"] = meta_df.at[prev_row, "WavePeriodInput"]
            prev_row = idx

    meta_df = meta_df.drop(columns=["_folder", "_sort_key"])
    return meta_df


def _probes_at_same_distance(cfg, ref_probe_num: int) -> list:
    """Return position strings of all probes sharing the same longitudinal
    distance as ``ref_probe_num`` (the reference probe is included).

    Example (``march2026_better_rearranging``):
        ref_probe_num = 1  (9373/170)
        → ['9373/170', '9373/340']       # probe 1 + probe 3, same 9373 distance

    Example (``nov_normalt_oppsett``):
        ref_probe_num = 3  (12400/170)
        → ['12400/170', '12400/340']     # probe 3 + probe 4, same 12400 distance
    """
    ref_dist = cfg.distances_mm[ref_probe_num]
    return [
        cfg.probe_col_name(p)
        for p in sorted(cfg.distances_mm)
        if cfg.distances_mm[p] == ref_dist
    ]


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
        in_probes_used  = pd.Series(index=meta_indexed.index, dtype=object)
        out_probes_used = pd.Series(index=meta_indexed.index, dtype=object)

        # Canonical IN/OUT amplitudes = mean across ALL probes at the
        # same longitudinal distance as the reference probe. See the
        # module docstring and CLAUDE.md §5 for the rationale.
        for (in_p, out_p), idx in meta_indexed.groupby(["in_probe", "out_probe"]).groups.items():
            in_ref_pos  = col_names[int(in_p)]
            out_ref_pos = col_names[int(out_p)]
            in_pos_ser.loc[idx]  = in_ref_pos
            out_pos_ser.loc[idx] = out_ref_pos

            in_positions  = _probes_at_same_distance(cfg, int(in_p))
            out_positions = _probes_at_same_distance(cfg, int(out_p))
            in_probes_used.loc[idx]  = "+".join(in_positions)
            out_probes_used.loc[idx] = "+".join(out_positions)

            in_amp_cols  = [f"Probe {p} Amplitude (FFT)" for p in in_positions
                            if f"Probe {p} Amplitude (FFT)" in meta_indexed.columns]
            out_amp_cols = [f"Probe {p} Amplitude (FFT)" for p in out_positions
                            if f"Probe {p} Amplitude (FFT)" in meta_indexed.columns]
            if in_amp_cols and out_amp_cols:
                in_mean  = meta_indexed.loc[idx, in_amp_cols].mean(axis=1, skipna=True)
                out_mean = meta_indexed.loc[idx, out_amp_cols].mean(axis=1, skipna=True)
                out_in.loc[idx] = out_mean / in_mean
        out_in = out_in.replace([np.inf, -np.inf], np.nan)
        meta_indexed[GC.OUT_IN_FFT]        = out_in
        meta_indexed["in_position"]        = in_pos_ser
        meta_indexed["out_position"]       = out_pos_ser
        meta_indexed["in_probes_used"]     = in_probes_used
        meta_indexed["out_probes_used"]    = out_probes_used

        # ── Generic IN / OUT columns ─────────────────────────────────
        # Canonical wave-measurement columns, each the mean across all
        # probes at the same longitudinal distance (≥1 probe per side).
        # Per-probe "Probe {pos} ..." columns remain in the table for
        # oddity inspection.
        #
        # For each suffix we also emit {IN,OUT}_disagree_frac — the
        # (max − min) / mean across the contributing probes. 0 when
        # only one probe shares the distance. Only computed for
        # Amplitude (FFT), since other quantities (k, λ, T) should
        # match between parallel probes to within a tiny fraction.
        _MEAN_SUFFIXES = [
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
        ]
        # Quality-metric suffixes — averaging makes less sense here, so
        # copy the reference probe value (preserves historical behaviour).
        _REF_ONLY_SUFFIXES = [
            "wave_stability",
            "period_amplitude_cv",
        ]
        for suffix in _MEAN_SUFFIXES + _REF_ONLY_SUFFIXES:
            in_vals  = pd.Series(index=meta_indexed.index, dtype=float)
            out_vals = pd.Series(index=meta_indexed.index, dtype=float)
            emit_spread = suffix == "Amplitude (FFT)"
            in_spread  = pd.Series(index=meta_indexed.index, dtype=float)
            out_spread = pd.Series(index=meta_indexed.index, dtype=float)

            for (in_p, out_p), idx in meta_indexed.groupby(["in_probe", "out_probe"]).groups.items():
                if suffix in _MEAN_SUFFIXES:
                    in_positions  = _probes_at_same_distance(cfg, int(in_p))
                    out_positions = _probes_at_same_distance(cfg, int(out_p))
                else:
                    in_positions  = [col_names[int(in_p)]]
                    out_positions = [col_names[int(out_p)]]
                in_src  = [f"Probe {p} {suffix}" for p in in_positions
                           if f"Probe {p} {suffix}" in meta_indexed.columns]
                out_src = [f"Probe {p} {suffix}" for p in out_positions
                           if f"Probe {p} {suffix}" in meta_indexed.columns]
                if in_src:
                    sub = meta_indexed.loc[idx, in_src]
                    in_vals.loc[idx] = sub.mean(axis=1, skipna=True)
                    if emit_spread and len(in_src) > 1:
                        m = sub.mean(axis=1, skipna=True)
                        spread = sub.max(axis=1, skipna=True) - sub.min(axis=1, skipna=True)
                        in_spread.loc[idx] = spread / m.where(m > 0, np.nan)
                    elif emit_spread:
                        in_spread.loc[idx] = 0.0
                if out_src:
                    sub = meta_indexed.loc[idx, out_src]
                    out_vals.loc[idx] = sub.mean(axis=1, skipna=True)
                    if emit_spread and len(out_src) > 1:
                        m = sub.mean(axis=1, skipna=True)
                        spread = sub.max(axis=1, skipna=True) - sub.min(axis=1, skipna=True)
                        out_spread.loc[idx] = spread / m.where(m > 0, np.nan)
                    elif emit_spread:
                        out_spread.loc[idx] = 0.0
            meta_indexed[f"IN {suffix}"]  = in_vals
            meta_indexed[f"OUT {suffix}"] = out_vals
            if emit_spread:
                meta_indexed["ain_disagree_frac"]  = in_spread
                meta_indexed["aout_disagree_frac"] = out_spread

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
        processvariables: dict,
        processed_dfs: Optional[dict] = None,
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
    meta_sel = compute_inter_run_timing(meta_sel, processed_dfs=processed_dfs)

    meta_sel = _set_output_folder(meta_sel, meta_full, debug)
    """VIKTIG - denne oppdaterer .JSON-filen"""
    update_processed_metadata(meta_sel, force_recompute=force_recompute)

    return meta_sel
