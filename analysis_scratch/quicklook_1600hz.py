"""
Quick visual inspection of 1.6 Hz full-panel runs.

Focus: the two dropout_critical/suspicious no-wind 0.3V runs from 20260327,
compared to normal references from the same folder and from 20260323.

Run:
    conda run -n draumkvedet python analysis_scratch/quicklook_1600hz.py
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).parent.parent))
from wavescripts.improved_data_loader import load_analysis_data, load_processed_dfs

BASE = Path(__file__).parent.parent
WDIR = BASE / "wavedata"

IN_POS  = "9373/170"
OUT_POS = "12400/250"
FS = 250  # Hz

# ---- Runs to inspect -------------------------------------------------------
# Each entry: (short_label, csv_relative_path, annotation)
RUNS_OF_INTEREST = [
    # Problem runs from 20260327
    (
        "20260327 run1 ⚠ dropout_critical\nnowind 0.3V per40",
        "20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange/"
        "fullpanel-nowind-amp0300-freq1600-per40-depth580-mstop30-run1.csv",
        "OUT/IN=1.268 (dropout: 170 cut)",
    ),
    (
        "20260327 run2 ? anomalous high\nnowind 0.3V per40",
        "20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange/"
        "fullpanel-nowind-amp0300-freq1600-per40-depth580-mstop30-run2.csv",
        "OUT/IN=0.764 (ok flag but high)",
    ),
    # Normal references from 20260327
    (
        "20260327 run3 ✓ normal\nnowind 0.3V per40",
        "20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange/"
        "fullpanel-nowind-amp0300-freq1600-per40-depth580-mstop30-run3.csv",
        "OUT/IN=0.553",
    ),
    (
        "20260327 run4 ✓ normal\nnowind 0.3V per40",
        "20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange/"
        "fullpanel-nowind-amp0300-freq1600-per40-depth580-mstop30-run4.csv",
        "OUT/IN=0.540",
    ),
    # Clean reference from 20260323
    (
        "20260323 run1 ✓ clean ref\nnowind 0.3V per40",
        "20260323-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100/"
        "fullpanel-nowind-amp0300-freq1600-per40-depth580-mstop30-run1.csv",
        "OUT/IN=0.686",
    ),
    # Full-wind reference (same folder)
    (
        "20260327 fullwind run1 ✓ ref\nfullwind 0.3V per40",
        "20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange/"
        "fullpanel-fullwind-amp0300-freq1600-per40-depth580-mstop30-run1.csv",
        "OUT/IN=0.634",
    ),
]

csv_paths = [BASE / "wavedata" / rel for _, rel, _ in RUNS_OF_INTEREST]

# ---- Load processed time-series -------------------------------------------
PROCESSED_DIRS = [
    Path("waveprocessed/PROCESSED-20260323-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100"),
    Path("waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"),
]
print("Loading meta...")
combined_meta, _, _, _ = load_analysis_data(*PROCESSED_DIRS, load_processed=False)

print("Loading processed time-series (may take ~20 s)...")
processed_dfs = load_processed_dfs(*PROCESSED_DIRS)
print(f"Loaded {len(processed_dfs)} processed runs")

# ---- Get analysis window info from meta ------------------------------------
meta_idx = {str(r): r for r in combined_meta["path"]}
in_col  = f"Probe {IN_POS}"
out_col = f"Probe {OUT_POS}"

# ---- Plot ------------------------------------------------------------------
n = len(RUNS_OF_INTEREST)
fig, axes = plt.subplots(n, 1, figsize=(14, 2.8 * n), sharex=False)
if n == 1:
    axes = [axes]

for ax, (label, rel, annot), csv_path in zip(axes, RUNS_OF_INTEREST, csv_paths):
    path_str = str(csv_path)
    df = processed_dfs.get(path_str)
    meta_row = combined_meta[combined_meta["path"] == path_str]

    if df is None:
        ax.text(0.5, 0.5, f"NOT FOUND in cache:\n{rel}", transform=ax.transAxes,
                ha="center", va="center", color="red")
        ax.set_title(label)
        continue

    t = np.arange(len(df)) / FS

    # Plot cleaned signals (eta = zeroed + fault-NaN'd; faults show as gaps in line)
    in_eta        = df.get(f"eta_{IN_POS}")
    out_eta       = df.get(f"eta_{OUT_POS}")
    in_interp     = df.get(f"eta_{IN_POS}_interp")   # interpolated fill (red overlay)
    out_interp    = df.get(f"eta_{OUT_POS}_interp")

    if in_eta is not None:
        ax.plot(t, in_eta.values,  color="steelblue",  lw=0.6, label=f"IN  {IN_POS}",  zorder=3)
    if out_eta is not None:
        ax.plot(t, out_eta.values, color="darkorange", lw=0.6, label=f"OUT {OUT_POS}", zorder=3)

    # Overlay interpolated sections in red — visible where eta is NaN but interp is not
    if in_eta is not None and in_interp is not None:
        interp_mask_in = in_eta.isna() & in_interp.notna()
        if interp_mask_in.any():
            # Draw a red line only at the interpolated positions
            interp_sig = in_interp.copy()
            interp_sig[~interp_mask_in] = np.nan
            ax.plot(t, interp_sig.values, color="red", lw=1.5,
                    label="IN interp-filled", zorder=4)

    # Mark analysis window from meta
    if not meta_row.empty:
        row = meta_row.iloc[0]
        qflag      = row.get("quality_flag", "?")
        cut_in     = row.get(f"cut_samples_{IN_POS}", 0)
        interp_in  = row.get(f"interp_samples_{IN_POS}", 0)

        start_samp = row.get(f"Computed Probe {IN_POS} start")
        end_samp   = row.get(f"Computed Probe {IN_POS} end")
        if start_samp is not None and not np.isnan(start_samp):
            mstart = start_samp / FS
            ax.axvline(mstart, color="green", lw=1.2, ls="--", label=f"mstart={mstart:.1f}s")
        if end_samp is not None and not np.isnan(end_samp):
            mstop_val = end_samp / FS
            ax.axvline(mstop_val, color="red", lw=1.2, ls="--", label=f"mstop={mstop_val:.1f}s")
            if start_samp is not None and not np.isnan(start_samp):
                ax.axvspan(mstart, mstop_val, alpha=0.07, color="green")

        fft_in = row.get(f"Probe {IN_POS} Amplitude (FFT)", float("nan"))
        amp_in = row.get(f"Probe {IN_POS} Amplitude",       float("nan"))
        stab_in = fft_in / amp_in if amp_in > 0 else float("nan")
        info = (f"flag={qflag}  cut_IN={int(cut_in)}  interp_IN={int(interp_in)}"
                f"  IN_FFT={fft_in:.1f}mm  IN_ws={stab_in:.2f}  {annot}")
    else:
        info = f"(no meta found in loaded dirs)  {annot}"

    ax.set_title(f"{label}  |  {info}", fontsize=8)
    ax.set_ylabel("η (mm)", fontsize=8)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, t[-1])

axes[-1].set_xlabel("Time (s)")
fig.suptitle("1.6 Hz full-panel signal inspection — IN and OUT probes", fontsize=11, y=1.002)
plt.tight_layout()

out_path = BASE / "analysis_scratch" / "quicklook_1600hz.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out_path}")
plt.show()
