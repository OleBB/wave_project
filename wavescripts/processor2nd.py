#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 10:37:28 2025

@author: ole
"""
import numpy as np
import pandas as pd

from wavescripts.improved_data_loader import update_processed_metadata
from typing import Mapping, Any, Optional, Sequence, Dict, Tuple, Iterable


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
        # Loop over the requested probes
        # ------------------------------------------------------------------
        for i in probes:
            col = f"Pxx {i}"
            if col not in df.columns:
                if verbose:
                    print(f"  Probe {i}: column '{col}' missing → skipped")
                continue

            # ----------------------------------------------------------------
            # Loop over the frequency bands
            # ----------------------------------------------------------------
            for band_name, (f_low, f_high) in freq_bands.items():
                # Build a mask for the current band
                mask = (df.index >= f_low) & (df.index <= f_high)
                n_points = int(mask.sum())

                # ------------------------------------------------------------
                #  Empty‑band handling (kept from version 2)
                # ------------------------------------------------------------
                if n_points == 0:
                    amplitude = 0.0
                    if verbose:
                        print(
                            f"  Probe {i} – {band_name}: "
                            f"NO DATA POINTS in [{f_low}, {f_high}] Hz → amplitude=0"
                        )
                    row[f"Probe {i} {band_name} Amplitude (PSD)"] = amplitude
                    continue

                # ------------------------------------------------------------
                #  Compute variance → amplitude
                # ------------------------------------------------------------
                if integration == "sum":
                    # Simple rectangular integration: Δf * Σ PSD
                    variance = df.loc[mask, col].sum() * freq_res
                else:  # "trapz"
                    # Use the true frequency axis for trapezoidal integration.
                    freqs = df.index.to_numpy(dtype=float)[mask]
                    psd_vals = df.loc[mask, col].to_numpy(dtype=float)
                    variance = np.trapezoid(psd_vals, x=freqs)

                # Standard deviation and peak‑to‑trough amplitude
                std_dev = np.sqrt(variance)
                amplitude = 2.0 * std_dev

                if verbose and i == probes[0]:  # print once per band, first probe
                    print(
                        f"  Probe {i} – {band_name} [{f_low}-{f_high}] Hz: "
                        f"{n_points} points, variance={variance:.6e}, amplitude={amplitude:.4f}"
                    )

                row[f"Probe {i} {band_name} Amplitude (PSD)"] = amplitude

        rows.append(row)

    # ----------------------------------------------------------------------
    #  Convert list‑of‑dicts → DataFrame (preserves column order)
    # ----------------------------------------------------------------------
    return pd.DataFrame(rows)
# %%
# def gpt_update_more_metrics(
#     psd_dict: dict,
#     fft_dict: dict,
#     meta_sel: pd.DataFrame,
# ) -> pd.DataFrame:
#     """
#     Overwrite/compute ratio columns and add band amplitudes aligned by 'path'.
#     Assumes meta_sel already contains Probe 1..4 Amplitude columns.
#     """
#     # Explicit copy and stable index
#     meta_indexed = meta_sel.set_index("path").copy()

#     # -----------------------------
#     # Ratios (fail-fast and aligned)
#     # -----------------------------
#     required_amp_cols = [
#         "Probe 1 Amplitude",
#         "Probe 2 Amplitude",
#         "Probe 3 Amplitude",
#         "Probe 4 Amplitude",
#     ]
#     missing = [c for c in required_amp_cols if c not in meta_indexed.columns]
#     if missing:
#         raise KeyError(f"Missing required amplitude columns: {missing}")

#     with np.errstate(divide="ignore", invalid="ignore"):
#         meta_indexed["P2/P1"] = meta_indexed["Probe 2 Amplitude"] / meta_indexed["Probe 1 Amplitude"]
#         meta_indexed["P3/P2"] = meta_indexed["Probe 3 Amplitude"] / meta_indexed["Probe 2 Amplitude"]
#         meta_indexed["P4/P3"] = meta_indexed["Probe 4 Amplitude"] / meta_indexed["Probe 3 Amplitude"]

#     ratio_cols = ["P2/P1", "P3/P2", "P4/P3"]
#     meta_indexed.loc[:, ratio_cols] = (
#         meta_indexed[ratio_cols]
#         .replace([np.inf, -np.inf], np.nan)
#     )

#     # ------------------------------------------
#     # Add PSD-based band amplitudes by 'path'
#     # ------------------------------------------
#     band_amplitudes = compute_amplitude_by_band(psd_dict)
#     if not isinstance(band_amplitudes, pd.DataFrame) or "path" not in band_amplitudes.columns:
#         raise ValueError("compute_amplitude_by_band(psd_dict) must return a DataFrame with a 'path' column.")

#     ba = band_amplitudes.set_index("path")
#     # Ensure we only select real columns (fail-fast on missing)
#     ba_cols = [c for c in ba.columns]
#     # Align to meta_indexed index (overwrite semantics)
#     meta_indexed.loc[:, ba_cols] = ba.reindex(meta_indexed.index)[ba_cols]

#     # ------------------------------------------
#     # Optionally, add FFT-based band amplitudes
#     # ------------------------------------------
#     # If you have a corresponding function that returns DataFrame with 'path'
#     if 'compute_fft_amplitude_by_band' in globals():
#         fft_band_amplitudes = compute_fft_amplitude_by_band(fft_dict)
#         if isinstance(fft_band_amplitudes, pd.DataFrame) and "path" in fft_band_amplitudes.columns:
#             fba = fft_band_amplitudes.set_index("path")
#             fba_cols = [c for c in fba.columns]
#             meta_indexed.loc[:, fba_cols] = fba.reindex(meta_indexed.index)[fba_cols]

#     return meta_indexed.reset_index()


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

    # ── Probe amplitude ratios ───────────────────────────────────────
    amp_cols = [
        "Probe 1 Amplitude (FFT)",
        "Probe 2 Amplitude (FFT)",
        "Probe 3 Amplitude (FFT)",
        "Probe 4 Amplitude (FFT)",
    ]

    # Only proceed if we have the needed columns
    missing = [col for col in amp_cols if col not in meta_indexed.columns]
    if missing:
        print(f"Warning: missing amplitude columns for ratios: {missing}")
        # or raise ValueError(...) if you prefer to fail loudly

    # Compute ratios directly (vectorized)
    ratios = pd.DataFrame(index=meta_indexed.index)

    ratios["P2/P1 (FFT)"] = (
        meta_indexed["Probe 2 Amplitude (FFT)"] / meta_indexed["Probe 1 Amplitude (FFT)"]
    )
    ratios["P3/P2 (FFT)"] = (
        meta_indexed["Probe 3 Amplitude (FFT)"] / meta_indexed["Probe 2 Amplitude (FFT)"]
    )
    ratios["P4/P3 (FFT)"] = (
        meta_indexed["Probe 4 Amplitude (FFT)"] / meta_indexed["Probe 3 Amplitude (FFT)"]
    )

    # Replace inf/-inf with NaN (e.g. division by zero)
    ratios = ratios.replace([np.inf, -np.inf], np.nan)

    # Assign / overwrite the ratio columns
    ratio_cols = ["P2/P1", "P3/P2", "P4/P3"]
    meta_indexed[ratio_cols] = ratios[ratio_cols]

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

# def _update_more_metrics(
#         psd_dict: dict,
#         fft_dict: dict,
#         meta_sel: pd.DataFrame
#         ) -> pd.DataFrame():
#     meta_indexed = meta_sel.set_index("path")
    
#     meta_indexed = meta_sel.set_index("path").copy()
    
#     # ==========================================================
#     #  plz help below
#     # ==========================================================
#     cols = ["path", "Probe 1 Amplitude","Probe 2 Amplitude", "Probe 3 Amplitude","Probe 4 Amplitude",]
#     sub_df = mdf[cols].copy()
#     sub_df["P2/P1"] = sub_df["Probe 2 Amplitude"] / sub_df["Probe 1 Amplitude"]
#     sub_df["P3/P2"] = sub_df["Probe 3 Amplitude"] / sub_df["Probe 2 Amplitude"]
#     sub_df["P4/P3"] = sub_df["Probe 4 Amplitude"] / sub_df["Probe 3 Amplitude"]
#     sub_df.replace([np.inf, -np.inf], np.nan, inplace=True) #infinite values  div 0
#     mdf_indexed = mdf.set_index("path") # set index to join back on "path"
#     ratios_by_path = sub_df.set_index("path")[["P2/P1", "P3/P2", "P4/P3"]]
#     mdf_indexed[["P2/P1", "P3/P2", "P4/P3"]] = ratios_by_path
#     mdf = mdf_indexed.reset_index() #reset index 
        
#     band_amplitudes = compute_amplitude_by_band(psd_dict)
    
    
    
    
#     return meta_indexed.reset_index
    

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

    meta_sel = _set_output_folder(meta_sel, meta_full, debug)
    """VIKTIG - denne oppdaterer .JSON-filen"""
    update_processed_metadata(meta_sel, force_recompute=force_recompute)
    
    return meta_sel
