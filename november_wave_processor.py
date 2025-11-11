#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 16:06:42 2025

@author: ole
"""

#from wavescripts.wave_processor import WaveProcessor
from pathlib import Path
import numpy as np
import pandas as pd
#import os

"""1 . Lese inn signalet"""

project_root = Path(__file__).parent
data_folder = project_root / "wavedata/20251110-tett6roof-lowM-ekte580"
results_folder = project_root / "waveresults"

results_folder.mkdir(exist_ok=True) # Ensure results folder exists

print('root is =  ', project_root)
print('datafolder = ', data_folder)
print('resultatene ligger i =', results_folder)

