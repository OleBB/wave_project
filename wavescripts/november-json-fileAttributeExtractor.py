#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 12:46:16 2025

@author: ole+gpt uio
"""

import os
import json
import re
from datetime import datetime  # Importing datetime

# Define the folder containing your data files
#Done. folder_path = '/Users/ole/Kodevik/wave_project/pressuredata/20251106-lowestwindUtenProbe2-fullpanel-amp0100-freq1300'
folder_path = '/Users/ole/Kodevik/wave_project/pressuredata/20251105-lowestwindUtenProbe2-fullpanel'


# List all files in the folder
files = os.listdir(folder_path)

# Define a function to extract metadata from filenames
def extract_metadata(filename):
    # Default values
    metadata = {
        "Date": "",
        "WindCondition": "Lowest",
        "TunnelCondition": "UP2",  # Default value
        "PanelCondition": "full",    # Default value
        "Mooring": "high",          #default BEFORE 20251106-fullwind.
        "WaveAmplitude": "",       # Default value
        "WaveFrequency": "",          # Default value
        "PitotPlacement": "",         # Default value
        "FileContents": "",
        "Real Pitot Height(mm)": 0,              # Default as zero
    }
    
    
    pitot_match = re.search(r'-pitot(\d+)', filename)
    if pitot_match:
        metadata["PitotPlacement"] = pitot_match.group(1)  

    tunnel_match = re.search(r'-lowestwind([A-Za-z0-9]+)', filename)
    if tunnel_match:
        metadata["TunnelCondition"] = tunnel_match.group(1)

    amplitude_match = re.search(r'-amp([A-Za-z0-9]+)-', filename)
    if amplitude_match:
        metadata["WaveAmplitude"] = 'A' + amplitude_match.group(1)  

    wave_match = re.search(r'-freq([A-Za-z0-9]+)', filename)
    if wave_match:
        metadata["WaveFrequency"] = 'F' + wave_match.group(1)  
    
    contents_match = re.search(r'(speed|stats)', filename, re.IGNORECASE)  # case-insensitive match
    if contents_match:
        metadata["FileContents"] = contents_match.group(1).lower()  # Store as lower case for consistency
        
    height_match = re.search(r'moh(\d+)', filename)  
    if height_match:
        metadata["Height(mm)"] = int(height_match.group(1))
    
    return metadata

# --- MAIN PROCESSING ---
for file in files:
    if file.endswith('.txt'):
        # Extract metadata from the filename
        metadata = extract_metadata(file)
        
        # Create a JSON structure with the current timestamp
        json_data = {
            "created_at": datetime.now().isoformat(),  # Set to current timestamp
            "parent": folder_path,
            "filename": file,
            "metadata": metadata,
            "notes": ""
        }

        # Create a JSON filename based on the text file name
        json_filename = os.path.join(folder_path, file.replace('.txt', '.json'))

        # Write the JSON data to a file
        with open(json_filename, 'w') as json_file:
            json.dump(json_data, json_file, indent=4)

        print(f"Created JSON file for: {file}")
