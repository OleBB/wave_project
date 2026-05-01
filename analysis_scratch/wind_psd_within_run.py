"""
Step 0 of the wind-snippet validation ladder: within-run stationarity.

Take ONE long nowave+fullwind run, slice into non-overlapping snippets of
length SNIPPET_S, compute one periodogram per snippet, and overlay them
per probe. Median + 16/84 percentile envelope show the within-run scatter.

Question this answers in one figure:
    "Is the wind statistically stationary within a single ~30 s recording?"

Loads ONE processed dataset only — no all-folder load.

Output:
    analysis_scratch/wind_psd_within_run.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import periodogram

from wavescripts.improved_data_loader import load_analysis_data, load_processed_dfs

FS = 250.0
BASE = Path("/Users/ole/Kodevik/wave_project")

# Single dataset & single run — ONE folder loaded total.
TARGET_DIR = BASE / "waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"
RUN_CSV    = str(BASE / "wavedata/20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange/fullpanel-fullwind-nowave-depth580-mstop30-run4.csv")

# march2026_better_rearranging: IN=9373/170, OUT=12400/250, parallel=9373/340, upstream=8804/250
PROBES = ["8804/250", "9373/170", "9373/340", "12400/250"]
LABELS = {
    "8804/250":  "8804/250 (upstream)",
    "9373/170":  "9373/170 (IN, wall)",
    "9373/340":  "9373/340 (IN, far)",
    "12400/250": "12400/250 (OUT)",
}
COLORS = {
    "8804/250":  "#888888",
    "9373/170":  "#FEA11B",
    "9373/340":  "#2ca02c",
    "12400/250": "#1E9C68",
}

SNIPPET_S = 2.0          # snippet length in seconds
WINDOW    = "hann"       # spectral window — Hann reduces leakage vs default boxcar
DETREND   = "linear"     # remove per-snippet linear trend (kills slow seiche / setup drift)

# ── Load minimal data ─────────────────────────────────────────────────────
print(f"Loading meta + processed_dfs for ONE dataset …")
meta, _, _, _ = load_analysis_data(str(TARGET_DIR), load_processed=False)
proc = load_processed_dfs(str(TARGET_DIR))

if RUN_CSV not in proc:
    print(f"\nAvailable nowave+fullwind paths in this folder:")
    for p in sorted(proc):
        if "fullwind-nowave" in p:
            print(f"  {p}")
    raise SystemExit(f"\nRun not in cache: {RUN_CSV}")

df = proc[RUN_CSV]
T  = len(df) / FS
print(f"  run length: {T:.1f} s  ({len(df)} samples)")

# ── Slice & compute periodogram per snippet ──────────────────────────────
snippet_n  = int(SNIPPET_S * FS)
n_snippets = len(df) // snippet_n
print(f"  {n_snippets} non-overlapping snippets × {SNIPPET_S} s "
      f"(Δf = {1/SNIPPET_S:.2f} Hz, samples/snippet = {snippet_n})")

per_probe_psd = {}   # probe → (freqs, psd_matrix [n_snippets × n_freqs])
for probe in PROBES:
    col = f"eta_{probe}_interp" if f"eta_{probe}_interp" in df.columns else f"eta_{probe}"
    if col not in df.columns:
        print(f"  skip {probe}: no eta column")
        continue
    eta = df[col].to_numpy(dtype=float)

    psds = []
    freqs = None
    for i in range(n_snippets):
        seg = eta[i * snippet_n:(i + 1) * snippet_n]
        if not np.all(np.isfinite(seg)):
            psds.append(None)
            continue
        f, P = periodogram(seg, fs=FS, window=WINDOW,
                           scaling="density", detrend=DETREND)
        if freqs is None:
            freqs = f
        psds.append(P)
    # Drop dropouts; stack survivors
    valid = [P for P in psds if P is not None]
    if not valid:
        continue
    per_probe_psd[probe] = (freqs, np.vstack(valid), len(valid))
    if len(valid) < n_snippets:
        print(f"  {probe}: {n_snippets - len(valid)} snippet(s) dropped (NaN)")

# ── Plot: 2×2 grid, log-y, shared axes ───────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 8.5),
                          sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.25, wspace=0.10)

for ax, probe in zip(axes.flat, PROBES):
    if probe not in per_probe_psd:
        ax.text(0.5, 0.5, f"{probe}\nno data",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color="#888")
        ax.set_title(LABELS[probe], fontsize=10)
        continue

    freqs, psd_mat, n_used = per_probe_psd[probe]
    color = COLORS[probe]

    # Faint per-snippet lines
    for P in psd_mat:
        ax.semilogy(freqs, P, color=color, lw=0.4, alpha=0.18)

    # Percentile envelope (16-84) — non-parametric "1σ" equivalent
    p16 = np.nanpercentile(psd_mat, 16, axis=0)
    p50 = np.nanpercentile(psd_mat, 50, axis=0)
    p84 = np.nanpercentile(psd_mat, 84, axis=0)
    ax.fill_between(freqs, p16, p84, color=color, alpha=0.30,
                    label="16–84 %")
    ax.semilogy(freqs, p50, color=color, lw=2.0, label="median")

    ax.set_title(f"{LABELS[probe]}   (n={n_used})", fontsize=10)
    ax.grid(True, which="major", alpha=0.35)
    ax.grid(True, which="minor", alpha=0.15)
    ax.legend(fontsize=8, loc="lower left")

axes[0, 0].set_ylabel("PSD [mm²/Hz]", fontsize=10)
axes[1, 0].set_ylabel("PSD [mm²/Hz]", fontsize=10)
axes[1, 0].set_xlabel("Frequency [Hz]", fontsize=10)
axes[1, 1].set_xlabel("Frequency [Hz]", fontsize=10)
axes[0, 0].set_xlim(0, 16)

fig.suptitle(
    f"Within-run wind PSD scatter — {Path(RUN_CSV).name}\n"
    f"{n_snippets} non-overlapping {SNIPPET_S:g}s snippets, "
    f"per-snippet linear detrend, Hann window  |  Δf={1/SNIPPET_S:.2f} Hz  |  "
    f"run length = {T:.1f} s",
    fontsize=11,
)

out = Path(__file__).parent / "wind_psd_within_run.png"
fig.savefig(out, dpi=160, bbox_inches="tight")
print(f"\n  → {out.relative_to(BASE)}")
print("Done.")
