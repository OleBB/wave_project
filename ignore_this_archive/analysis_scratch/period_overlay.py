"""
Period overlay plot for run2 at 1.6 Hz nowind.

Stacks every individual upcrossing-to-upcrossing period on top of each other
so the phase-locked probe artifact is clearly visible.

Shows:
  - All periods overlaid (thin gray lines)
  - Median period (blue) — robust central estimate
  - Periods with a deep trough anomaly highlighted (red)
  - Phase of the known anomaly region marked

Run:
    conda run -n draumkvedet python analysis_scratch/period_overlay.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from wavescripts.improved_data_loader import load_analysis_data, load_processed_dfs

BASE = Path(__file__).parent.parent
FS   = 250.0
POS  = "9373/170"
FREQ = 1.6

PROC_DIR = BASE / "waveprocessed" / (
    "PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"
)

combined_meta, _, _, _ = load_analysis_data(PROC_DIR, load_processed=False)
processed_dfs = load_processed_dfs(PROC_DIR)

# ── Load run2 and clean references ────────────────────────────────────────────
def get_run(name):
    key = next(k for k in processed_dfs if name in k)
    df  = processed_dfs[key]
    row = combined_meta[combined_meta["path"] == key].iloc[0]
    start = int(row[f"Computed Probe {POS} start"])
    end   = int(row[f"Computed Probe {POS} end"])
    sig   = df[f"eta_{POS}_interp"].values[start:end]
    return sig, start, end

sig2, s2, e2 = get_run("fullpanel-nowind-amp0300-freq1600-per40-depth580-mstop30-run2.csv")
sig3, s3, e3 = get_run("fullpanel-nowind-amp0300-freq1600-per40-depth580-mstop30-run3.csv")
sig4, s4, e4 = get_run("fullpanel-nowind-amp0300-freq1600-per40-depth580-mstop30-run4.csv")

# ── Find upcrossings (zero upcrossings of the zeroed signal) ──────────────────
def find_upcrossings(sig):
    mean = np.nanmean(sig)
    s = sig - mean
    crossings = []
    for i in range(len(s) - 1):
        if np.isnan(s[i]) or np.isnan(s[i+1]):
            continue
        if s[i] < 0 and s[i+1] >= 0:
            # linear interpolation for sub-sample accuracy
            frac = -s[i] / (s[i+1] - s[i])
            crossings.append(i + frac)
    return np.array(crossings)

uc2 = find_upcrossings(sig2)
uc3 = find_upcrossings(sig3)
uc4 = find_upcrossings(sig4)

# ── Extract periods and resample to fixed grid ────────────────────────────────
N_PHASE = 200   # phase grid points per period (0 to 2π)
phase_grid = np.linspace(0, 1, N_PHASE, endpoint=False)

def extract_periods(sig, upcrossings):
    """Return list of (period_signal resampled to N_PHASE points)."""
    periods = []
    for i in range(len(upcrossings) - 1):
        i0 = int(np.floor(upcrossings[i]))
        i1 = int(np.ceil(upcrossings[i + 1]))
        if i1 >= len(sig):
            break
        chunk = sig[i0:i1]
        if len(chunk) < 10:
            continue
        t_chunk = np.linspace(0, 1, len(chunk), endpoint=False)
        resampled = np.interp(phase_grid, t_chunk, chunk)
        periods.append(resampled)
    return np.array(periods)

periods2 = extract_periods(sig2, uc2)
periods3 = extract_periods(sig3, uc3)
periods4 = extract_periods(sig4, uc4)

# ── Classify run2 periods by anomaly severity ─────────────────────────────────
# "False trough": min value below -40mm (severe probe fault)
# "Indent": min in the post-trough rising phase (phase 0.6–0.85) below -12mm
#           but above -40mm (subtle artifact)
# "Clean": everything else

SEVERE_THRESH = -40.0    # below this → false trough
INDENT_PHASE  = (0.55, 0.85)   # post-trough rising phase window (fraction of period)
INDENT_THRESH = -12.0    # indent if min in that window < this (but > SEVERE_THRESH)

idx_phase = np.where(
    (phase_grid >= INDENT_PHASE[0]) & (phase_grid <= INDENT_PHASE[1])
)[0]

severe_idx  = []
indent_idx  = []
clean_idx   = []

for i, p in enumerate(periods2):
    min_val = np.nanmin(p)
    min_post = np.nanmin(p[idx_phase])
    if min_val < SEVERE_THRESH:
        severe_idx.append(i)
    elif min_post < INDENT_THRESH:
        indent_idx.append(i)
    else:
        clean_idx.append(i)

print(f"Run2: {len(periods2)} periods total")
print(f"  Clean:        {len(clean_idx)}")
print(f"  Indent only:  {len(indent_idx)}  (post-trough dip < {INDENT_THRESH} mm)")
print(f"  Severe fault: {len(severe_idx)}  (min < {SEVERE_THRESH} mm)")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
fig.suptitle(
    "Period overlay — individual upcrossing periods stacked\n"
    "Left: run2 (anomalous)   Middle: run3 (clean ref)   Right: run4 (clean ref)",
    fontsize=10,
)

def plot_overlay(ax, periods, title, severe=None, indent=None, clean=None,
                 show_classification=True):
    if show_classification and severe is not None:
        for i in clean:
            ax.plot(phase_grid, periods[i], color="steelblue", lw=0.6, alpha=0.5)
        for i in indent:
            ax.plot(phase_grid, periods[i], color="orange", lw=0.8, alpha=0.8)
        for i in severe:
            ax.plot(phase_grid, periods[i], color="red", lw=0.9, alpha=0.9)
        # legend proxies
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], color="steelblue", lw=1.5, label=f"clean ({len(clean)})"),
            Line2D([0], [0], color="orange",    lw=1.5, label=f"indent ({len(indent)})"),
            Line2D([0], [0], color="red",       lw=1.5, label=f"severe ({len(severe)})"),
        ]
        ax.legend(handles=handles, fontsize=8, loc="upper right")
    else:
        for p in periods:
            ax.plot(phase_grid, p, color="steelblue", lw=0.5, alpha=0.4)

    # Median
    med = np.nanmedian(periods, axis=0)
    ax.plot(phase_grid, med, color="black", lw=1.8, label="median", zorder=5)

    # Mark the post-trough anomaly window
    ax.axvspan(INDENT_PHASE[0], INDENT_PHASE[1],
               alpha=0.08, color="red", label="post-trough window")
    ax.axhline(INDENT_THRESH, color="orange", lw=0.8, ls=":",
               label=f"indent thresh ({INDENT_THRESH} mm)")
    ax.axhline(SEVERE_THRESH, color="red", lw=0.8, ls=":",
               label=f"severe thresh ({SEVERE_THRESH} mm)")

    ax.set_xlabel("Phase (fraction of period)")
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

plot_overlay(axes[0], periods2, f"run2  (n={len(periods2)} periods)",
             severe=severe_idx, indent=indent_idx, clean=clean_idx)
def classify(periods):
    sev, ind, cln = [], [], []
    for i, p in enumerate(periods):
        min_val  = np.nanmin(p)
        min_post = np.nanmin(p[idx_phase])
        if min_val < SEVERE_THRESH:
            sev.append(i)
        elif min_post < INDENT_THRESH:
            ind.append(i)
        else:
            cln.append(i)
    return sev, ind, cln

sev3, ind3, cln3 = classify(periods3)
sev4, ind4, cln4 = classify(periods4)
print(f"Run3: clean={len(cln3)}, indent={len(ind3)}, severe={len(sev3)}")
print(f"Run4: clean={len(cln4)}, indent={len(ind4)}, severe={len(sev4)}")

plot_overlay(axes[1], periods3, f"run3  (n={len(periods3)} periods)",
             severe=sev3, indent=ind3, clean=cln3)
plot_overlay(axes[2], periods4, f"run4  (n={len(periods4)} periods)",
             severe=sev4, indent=ind4, clean=cln4)

axes[0].set_ylabel("η (mm)")

# Add phase labels on bottom axis
for ax in axes:
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0\n(upcross)", "¼", "½\n(trough)", "¾", "1\n(upcross)"])

plt.tight_layout()
out_path = Path(__file__).parent / "period_overlay.png"
plt.savefig(out_path, dpi=150)
print(f"Saved: {out_path}")
