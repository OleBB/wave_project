#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 10:10:11 2026

@author: ole
"""
import numpy as np
import pandas as pd
import pytest



#ny fil for å teste grener i git

#kan også prøve meg med pytest. 

def funksjon(x): 
    return x+1

def test_svaret():
    assert funksjon(3) == 5
    
    
# from wavescripts.processor import compute_fft


# compute_fft(processed_dfs, meta_row)
    
    
# def test_fft():
    
#     assert
    

"""
Purpose of pytest in a data science project
Prevent regressions automatically: What you validated today can silently break next week. Tests re-run the same checks on every change without you having to remember them.
Codify assumptions as executable documentation: Tests make your data/algorithm contracts explicit (e.g., timestamps are monotonic, sampling is 1 Hz, peak distance ≥ N).
Enable safe refactoring: When you optimize or restructure code, tests tell you instantly if behavior changed.
Catch edge cases you didn’t exercise manually: Parametrized tests and property-based tests try lots of inputs quickly.
Support collaboration and CI: Teammates and CI run tests on their machines/PRs so issues don’t slip into main.
Reproducibility over time: With fixed seeds and “golden” baselines (metrics or small images), tests ensure future runs match expectations within tolerances.
Faster feedback loop: A test suite runs in seconds. Manual plots and eyeballing take minutes and are easy to skip.
How tests differ from runtime error handling
Error handling deals with “what to do when something goes wrong at runtime” (e.g., raise or fallback). Tests check that “things still behave as intended” during development.
Example:
Error handling: If data has duplicates, raise ValueError.
Tests: Ensure the function raises ValueError given a tiny synthetic DataFrame with duplicates. That guards the contract and ensures your checks remain in place.
Concrete scenarios where pytest adds value beyond manual checks
Visual vs. metric regression
You eyeball plots now, but a future change subtly shifts scaling or filters. A test that asserts summary metrics (mean, RMS, peak counts, dominant frequency) or image similarity within tolerance will catch it immediately.
Pipeline drift
You change a resampling method or filter order. A test that runs a tiny synthetic signal through the pipeline and asserts the expected spectrum/peaks prevents unnoticed drift.
Data quality gates
Tests on tiny sample inputs ensure: monotonic timestamps, max gap ≤ X seconds, NaN fraction ≤ Y%, amplitude range within bounds. These run automatically in CI, not just when you remember to check.
Refactoring safety
You move logic from notebooks to modules, rename functions, or vectorize code. Tests verify outputs still match known baselines.
"""