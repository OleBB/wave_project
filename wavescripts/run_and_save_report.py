 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 13:56:37 2025

@author: ole
"""


# --------------------------------------------------------------
# Imports & helper for Markdown formatting
# --------------------------------------------------------------
import io
import contextlib
from pathlib import Path
import matplotlib.pyplot as plt

# Optional – a tiny helper to make headings look nice in Markdown
def md_heading(text, level=2):
    return f"{'#' * level} {text}\n\n"

# --------------------------------------------------------------
#  Your original data‑processing functions (place‑holders)
# --------------------------------------------------------------
# NOTE: Replace these with the real implementations you already have.
def wind_damping_analysis(processed_dfs, meta_sel, window_ms):
    # … your code …
    return "summary_df result"

def full_tank_diagnostics(processed_dfs, window_ms):
    # … your code …
    return "summary result"

def amplitude_overview(processed_dfs, window_ms):
    # … your code …
    return "overview result"

def compare_probe_amplitudes_and_lag(df, start_ms, end_ms):
    # … your code …
    # Example return value (the real function will return a dict)
    return {"lag_ms": 12.3}

# --------------------------------------------------------------
#  Load / create the data you already have
# --------------------------------------------------------------
# `processed_dfs` must be a dict of DataFrames, e.g.:
# processed_dfs = {"tank1": pd.read_csv("tank1.csv", index_col=0)}
# For this example we just create a dummy DataFrame:

import pandas as pd
import numpy as np

np.random.seed(0)
time = np.arange(0, 20000)               # ms
df_dummy = pd.DataFrame({
    "eta_1": np.random.normal(0, 10, size=len(time)),
    "eta_2": np.random.normal(0, 10, size=len(time)),
    "eta_3": np.random.normal(0, 10, size=len(time)),
    "eta_4": np.random.normal(0, 10, size=len(time)),
}, index=time)

processed_dfs = {"example": df_dummy}
meta_sel = None   # whatever you need for theBelow is a **self‑contained** way to run the snippet you posted **and** capture everything that normally goes to the console into a Markdown file (e.g. `report.md`).  
#You can drop the whole block into a new `.py` file and execute it – the script will create the markdown file next to it.


# --------------------------------------------------------------
#  Imports & helper for Markdown formatting
# --------------------------------------------------------------
import io
import contextlib
from pathlib import Path
import matplotlib.pyplot as plt

# Optional – a tiny helper to make headings look nice in Markdown
def md_heading(text, level=2):
    return f"{'#' * level} {text}\n\n"

# --------------------------------------------------------------
#  Your original data‑processing functions (place‑holders)
# --------------------------------------------------------------
# NOTE: Replace these with the real implementations you already have.
def wind_damping_analysis(processed_dfs, meta_sel, window_ms):
    # … your code …
    return "summary_df result"

def full_tank_diagnostics(processed_dfs, window_ms):
    # … your code …
    return "summary result"

def amplitude_overview(processed_dfs, window_ms):
    # … your code …
    return "overview result"

def compare_probe_amplitudes_and_lag(df, start_ms, end_ms):
    # … your code …
    # Example return value (the real function will return a dict)
    return {"lag_ms": 12.3}

# --------------------------------------------------------------
#  Load / create the data you already have
# --------------------------------------------------------------
# `processed_dfs` must be a dict of DataFrames, e.g.:
# processed_dfs = {"tank1": pd.read_csv("tank1.csv", index_col=0)}
# For this example we just create a dummy DataFrame:

import pandas as pd
import numpy as np

np.random.seed(0)
time = np.arange(0, 20000)               # ms
df_dummy = pd.DataFrame({
    "eta_1": np.random.normal(0, 10, size=len(time)),
    "eta_2": np.random.normal(0, 10, size=len(time)),
    "eta_3": np.random.normal(0, 10, size=len(time)),
    "eta_4": np.random.normal(0, 10, size=len(time)),
}, index=time)

processed_dfs = {"example": df_dummy}
meta_sel = None   # whatever you need for the analysis functions

# --------------------------------------------------------------
#  Capture everything that would be printed
# --------------------------------------------------------------
output_path = Path("report.md")
with output_path.open("w", encoding="utf-8") as f, contextlib.redirect_stdout(f):

    # ---- write a title -------------------------------------------------
    print(md_heading(" Wind‑Damping / Tank Diagnostics Report", level=1))

    # ---- run the analyses (you can keep the return values if you need them) ----
    summary_df = wind_damping_analysis(processed_dfs, meta_sel,
                                       window_ms=(6000, 14000))
    summary = full_tank_diagnostics(processed_dfs, window_ms=(8000, 8100))
    overview = amplitude_overview(processed_dfs, window_ms=(5000, 15000))

    # ---- pick a DataFrame -------------------------------------------------
    df = list(processed_dfs.values())[0]

    # ---- 1 Print the first few rows ------------------------------------
    print(md_heading("Data preview (first 5 rows)", level=2))
    print('name:', df.head(), "\n")               # prints a nicely formatted DataFrame

    # ---- 2 Raw amplitudes ------------------------------------------------
    print(md_heading("Raw amplitudes (before any fix)", level=2))
    for i in range(1, 5):
        col = f"eta_{i}"
        if col in df.columns:
            amp = (df[col].quantile(0.99) - df[col].quantile(0.01)) / 2
            status = 'PROBABLY BAD' if amp > 50 else 'OK'
            print(f"  Probe {i}: {amp:.1f} mm  →  {status}")

    # ---- 3 Compare probe amplitudes & lag --------------------------------
    print("\n" + md_heading("Probe‑amplitude & lag comparison", level=2))
    result = compare_probe_amplitudes_and_lag(df, start_ms=6000, end_ms=7000)
    res = compare_probe_amplitudes_and_lag(df, start_ms=5000, end_ms=15000)
    print("Result (6000‑7000 ms):", result)
    print("Result (5000‑15000 ms):", res)

    # ---- 4  Plot (saved as an image) ------------------------------------
    # We still create the plot, but we **save** it to a file instead of showing it.
    window = df.loc[5000:15000]
    t = (window.index - window.index[0]) / 1000  # seconds

    plt.figure(figsize=(10, 4))
    plt.plot(t, window["eta_2"], label="Probe 2", alpha=0.8)
    plt.plot(t,
             window["eta_3"] - res["lag_ms"],
             label=f"Probe 3 (shifted -{res['lag_ms']:.0f}ms)",
             alpha=0.8)

    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Elevation [mm]")
    plt.title("Perfect alignment after time‑shift correction")
    plt.grid(alpha=0.3)

    img_path = Path("alignment_plot.png")
    plt.tight_layout()
    plt.savefig(img_path, dpi=150)
    plt.close()

    # ---- 5  Embed the plot in the markdown --------------------------------
    print("\n" + md_heading("Alignment plot", level=2))
    print(f"![Alignment plot]({img_path.name})\n")

# --------------------------------------------------------------
# 6 Done – the file `report.md` now contains everything
# --------------------------------------------------------------
print(f"✅  Report written to: {output_path.resolve()}")
print(f"🖼️  Plot saved as: {img_path.resolve()}")
