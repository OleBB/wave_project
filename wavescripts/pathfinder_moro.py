#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 20:35:57 2025

@author: ole
"""


from pathlib import Path
import os
from datetime import datetime
import json
import re
import pandas as pd

#FOLDER1 = Path("/Users/ole/Kodevik/wave_project/wavedata/20251110-tett6roof-lowM-ekte580/")
#fil1 = Path("/Users/ole/Kodevik/wave_project/wavedata/20251110-tett6roof-lowM-ekte580/fullpanel-fullwind-amp0100-freq0650-per15-depth580-mstop30-run1.csv")

FOLDER1 = Path("/Users/ole/Kodevik/wave_project/wavezarchive")
fil1 = Path("/Users/ole/Kodevik/wave_project/wavezarchive/sixttry6roof-fullpanel-fullwind-amp0100-freq1300-per120-depth580-mstop30-run1-COPY.csv")

print(fil1.name)
#%% - string manipulation. Beholder alt før første dash og alt etter siste dash
#datafoldername1 = re.sub(r'-.*-', '-', fil1)
#datafoldername1 = re.sub(r'-', '-', fil1)
#%%

def extract_wavemetadata(filename):
    # Default values
    metadata = {
        "WindCondition": "",
        "TunnelCondition": "",       # Default value
        "PanelCondition": "",    # Default value
        "Mooring": "",          # high was default BEFORE 20251106-fullwind.
        "WaveAmplitudeInput [Volt]": "",       
        "WaveFrequencyInput [Hz]": "",          
        "WavePeriodInput": "",         
        "WaterDepth [mm]": "",         
        "Extra seconds": "",         
        "Run number": ""
    }
    
    wind_match = re.search(r'-([A-Za-z]+)wind-', filename)
    if wind_match:
        metadata["WindCondition"] = wind_match.group(1)

    tunnel_match = re.search(r'([0-9])roof', filename)
    if tunnel_match:
        metadata["TunnelCondition"] = tunnel_match.group(1) + " roof plates"
    
    #### LEGG TIL noe som leser mappenavnet om det er 6roof eller ikke.
    
    panel_match = re.search(r'([A-Za-z]+)panel', filename)
    if panel_match:
        metadata["PanelCondition"] = panel_match.group(1)
    
    modtime = os.path.getmtime(filename)
    date = datetime.fromtimestamp(modtime)
    date_match = re.search(r'(\d{8})', filename)
    if date_match:
            print('first if')
            metadata["Date"] = date_match.group(1)
            if metadata["Date"] < "20251106":
                metadata["Mooring"] = "high"
            else:
                metadata["Mooring"] = "low"
    mooringdate = '2025-11-6'
    mooringdatetime = datetime.strptime(mooringdate, '%Y-%m-%d')
    if date < mooringdatetime:
        metadata["Mooring"] = "high"
    else:
        metadata["Mooring"] = "low"
            
    amplitude_match = re.search(r'-amp([A-Za-z0-9]+)-', filename)
    if amplitude_match:
        metadata["WaveAmplitudeInput [Volt]"] = amplitude_match.group(1)  

    frequency_match = re.search(r'-freq([A-Za-z0-9]+)', filename)
    if frequency_match:
        metadata["WaveFrequencyInput [Hz]"] =  frequency_match.group(1)  
    
    period_match = re.search(r'-per([A-Za-z0-9]+)', filename)
    if period_match:
        metadata["WavePeriodInput"] =  period_match.group(1) 
        
    depth_match = re.search(r'-depth([A-Za-z0-9]+)', filename)
    if depth_match:
        metadata["WaterDepth [mm]"] =  depth_match.group(1) 
        
    mstop_match = re.search(r'-mstop([A-Za-z0-9]+)', filename)
    if mstop_match:
        metadata["Extra seconds"] =  mstop_match.group(1) 
    
    run_match = re.search(r'-run([0-9])', filename, re.IGNORECASE)  # case-insensitive match
    if run_match:
        metadata["Run number"] = run_match.group(1).lower()  # Store as lower case for consistency
    
    return metadata


#  ---- hardkodet, bytt ut senere --- 
#files = [fil1]
folder_path = FOLDER1
files = os.listdir(folder_path)

fil2 = "sixttry6roof-fullpanel-fullwind-amp0100-freq1300-per120-depth580-mstop30-run1-COPY.csv"


metas = extract_wavemetadata(fil2)
print(metas)



#%%

# --- MAIN PROCESSING ---
for file in files:
    if file.endswith('.csv'):
        # Extract metadata from the filename
        metadata = extract_wavemetadata(file)
        
        # Create a JSON structure with the current timestamp
        json_data = {
            "json data": metadata
        }

# Create a JSON filename based on the text file name
json_filename = os.path.join(folder_path, file.replace('.csv', '.json'))


# Write the JSON data to a file
with open(json_filename, 'w') as json_file:
    json.dump(json_data, json_file, indent=4)

print(f"Created JSON file for: {file}")


f1 = extract_wavemetadata(fil1)
print(f1)






