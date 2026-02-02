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

""""
def probe_comparisor(meta_df):
    df = meta_df.copy()

    for idx, row, in df.iterrows():
        P1 = row["Probe 1 Amplitude"]
        P2 = row["Probe 2 Amplitude"]
        P3 = row["Probe 3 Amplitude"]
        P4 = row["Probe 4 Amplitude"]
        
        if P1 != 0:
            df.at[idx, "P2/P1"] = P2/P1
            
        if P2 != 0:
            df.at[idx, "P3/P2"] = P3/P2
        
        if P3 != 0:
            df.at[idx, "P4/P3"] = P4/P3
         
    return df"""



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
            "swell":      (0.0, 2.6),
            "wind_waves": (2.60000001, 16.0),
            "total":      (0.0, 16.0),
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
                    row[f"Probe {i} {band_name} amplitude"] = amplitude
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

                row[f"Probe {i} {band_name} amplitude"] = amplitude

        rows.append(row)

    # ----------------------------------------------------------------------
    #  Convert list‑of‑dicts → DataFrame (preserves column order)
    # ----------------------------------------------------------------------
    return pd.DataFrame(rows)

def _update_more_metrics(
        psd_dict: dict,
        fft_dict: dict,
        meta_sel: pd.DataFrame
        ) -> pd.DataFrame():
    meta_indexed = meta_sel.set_index("path")
    
    meta_indexed = meta_sel.set_index("path").copy()
    
    # ==========================================================
    #  plz help below
    # ==========================================================
    cols = ["path", "Probe 1 Amplitude","Probe 2 Amplitude", "Probe 3 Amplitude","Probe 4 Amplitude",]
    sub_df = mdf[cols].copy()
    sub_df["P2/P1"] = sub_df["Probe 2 Amplitude"] / sub_df["Probe 1 Amplitude"]
    sub_df["P3/P2"] = sub_df["Probe 3 Amplitude"] / sub_df["Probe 2 Amplitude"]
    sub_df["P4/P3"] = sub_df["Probe 4 Amplitude"] / sub_df["Probe 3 Amplitude"]
    sub_df.replace([np.inf, -np.inf], np.nan, inplace=True) #infinite values  div 0
    mdf_indexed = mdf.set_index("path") # set index to join back on "path"
    ratios_by_path = sub_df.set_index("path")[["P2/P1", "P3/P2", "P4/P3"]]
    mdf_indexed[["P2/P1", "P3/P2", "P4/P3"]] = ratios_by_path
    mdf = mdf_indexed.reset_index() #reset index 
        
    band_amplitudes = compute_amplitude_by_band(psd_dict)
    
    
    
    
    return meta_indexed.reset_index
    

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
