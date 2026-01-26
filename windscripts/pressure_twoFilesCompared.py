#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 15:58:45 2025

@author: ole
"""

# compare_files.py

import numpy as np
from scipy import stats

def read_values(filename):
    values = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().strip('"').replace(',', '.')
            if line:
                try:
                    values.append(float(line))
                except ValueError:
                    pass
    return np.array(values)

def compare_files(file1, file2):
    """Return a dictionary with summary statistics and t-test result."""
    data1 = read_values(file1)
    data2 = read_values(file2)

    def stats_summary(data):
        return dict(
            mean=np.mean(data),
            min=np.min(data),
            max=np.max(data),
            std=np.std(data, ddof=1),
            count=len(data)
        )

    s1, s2 = stats_summary(data1), stats_summary(data2)
    t_stat, p_val = stats.ttest_ind(data1, data2, equal_var=False)

    return {
        "file1": s1,
        "file2": s2,
        "mean_diff": s1["mean"] - s2["mean"],
        "t_stat": t_stat,
        "p_val": p_val
    }
