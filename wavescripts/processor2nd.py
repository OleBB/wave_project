#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 10:37:28 2025

@author: ole
"""
import numpy as np
import pandas as pd

from wavescripts.data_loader import update_processed_metadata


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
         
    return df


def process_processed_data(
        meta_sel: pd.DataFrame,
) -> pd.DataFrame:
    """ nu kjører vi funksjoner som krever en df som allerede har verdier for alle probene"""
    mdf = meta_sel.copy()
    
    print("kjører process_processed_data fra processsor2nd.py")
    
    # ==========================================================
    #  regne ratioer. Oppdatere mdf
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
        
    
    """VIKTIG - denne oppdaterer .JSON-filen"""
    update_processed_metadata(mdf)
    return mdf