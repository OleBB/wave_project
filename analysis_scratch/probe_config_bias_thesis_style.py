"""
Probe-config bias — thesis-style K_t vs k figure
==================================================

Mirrors the visual language of `plot_damping_freq` (CH05 §1):
  - 3 subfigures, one per amplitude (A1 / A2 / A3)
  - x = k, top axis = paddle frequency (full dispersion via freq_to_k)
  - y = K_t = OUT/IN (FFT)
  - marker shape per amp tier (○ A1, □ A2, △ A3) — thesis convention
  - horizontal $K_t$ y-label above leftmost tick

Probe config is the primary colour. Palette deliberately avoids red/blue
(thesis-wide WIND_COLOR_MAP: red=med vind, blue=uten vind) so wind can be
read separately via linestyle: solid = uten vind, dashed = med vind.

  100/high → green
  100/low  → purple   (canon — height100 + lowrange)
  272/high → gold-brown

136/high is dropped — only 3 runs in this slice, too sparse to be a usable line.

All frequencies kept (no thesis-scope restriction). All `below_90_loose230 /
full panel` data, including cells where only one config sampled (no n filter).

Outputs (one PDF per amplitude tier):
    analysis_scratch/probe_config_bias_thesis_A{1,2,3}.pdf
    output/FIGURES/ch05_probe_config_bias_A{1,2,3}.pdf
"""

import sys
import warnings
from pathlib import Path
import glob

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import os
os.chdir(BASE)

from wavescripts.improved_data_loader import load_analysis_data
from wavescripts.plot_utils import (
    apply_thesis_style, freq_to_k, add_freq_axis, apply_horizontal_ylabel,
)

apply_thesis_style()

# ── 1. Load & filter ───────────────────────────────────────────────────────────
print("1. Loading processed folders …")
all_dirs = sorted(glob.glob(str(BASE / "waveprocessed" / "PROCESSED-*")))
meta, _, _, _ = load_analysis_data(*all_dirs, load_processed=False)

sel = meta[
    meta["WaveFrequencyInput [Hz]"].notna()
    & (meta["WaveFrequencyInput [Hz]"] > 0)
    & (meta["quality_flag"] == "ok")
    & meta["OUT/IN (FFT)"].notna()
    & meta["WindCondition"].isin(["no", "full"])
    & (meta["WaveFrequencyInput [Hz]"] < 2.0)
    & (meta["OUT/IN (FFT)"].between(0.1, 2.0))
    & (meta["Mooring"] == "below_90_loose230")
    & (meta["PanelCondition"] == "full")
].copy()

sel["cfg"] = (sel["probe_height_mm"].astype("Int64").astype(str)
              + "/" + sel["probe_range_mode"].astype(str))
sel["amp_v"] = sel["WaveAmplitudeInput [Volt]"].apply(lambda v: round(float(v), 2))
sel = sel.rename(columns={"WaveFrequencyInput [Hz]": "freq",
                           "WindCondition": "wind"})
print(f"   {len(sel)} rows in below_90_loose230 / full panel")

# ── 2. Aggregate per (cfg, freq, wind, amp) ────────────────────────────────────
agg = (sel.groupby(["cfg", "freq", "wind", "amp_v"])
          ["OUT/IN (FFT)"]
          .agg(["mean", "std", "count"])
          .reset_index())

# ── 3. Plot — one figure per amplitude tier (matches damping_freq layout) ─────
AMP_TIERS = [(0.10, "A1", "o"),
             (0.20, "A2", "s"),
             (0.30, "A3", "^")]

# Drop 136/high (only 3 rows in below_90_loose230/full panel).
CFG_ORDER = ["100/high", "100/low", "272/high"]
CFG_COLOR = {
    "100/high": "#2CA02C",   # green
    "100/low":  "#7B3F99",   # purple (canon)
    "272/high": "#A0721B",   # gold-brown
}
# Filter the data accordingly so the agg / freqs_present / y-range all
# reflect only the 3 plotted configs.
agg = agg[agg["cfg"].isin(CFG_ORDER)].copy()
WIND_LS = {"no": "-", "full": "--"}
WIND_LABEL = {"no": "uten vind", "full": "med vind"}

# y-limits — pick a single shared range so the 3 subfigures stack visually.
y_min = float(agg["mean"].min()) - 0.05
y_max = float(agg["mean"].max()) + 0.05
print(f"   y-range across all subfigs: [{y_min:.2f}, {y_max:.2f}]")

# x-limits in k from the freq range present.
freqs_present = sorted(agg["freq"].unique())
print(f"   freqs present: {freqs_present}")
ks_present = freq_to_k(np.array(freqs_present))
k_min = float(ks_present.min()) - 0.3
k_max = float(ks_present.max()) + 0.3

for amp_v, amp_tag, marker in AMP_TIERS:
    sub = agg[agg["amp_v"] == amp_v]
    if sub.empty:
        print(f"   [{amp_tag}] no rows — skipping")
        continue

    fig, ax = plt.subplots(figsize=(7, 3))

    for cfg in CFG_ORDER:
        for wind in ["no", "full"]:
            grp = sub[(sub["cfg"] == cfg) & (sub["wind"] == wind)]
            if grp.empty:
                continue
            grp = grp.sort_values("freq")
            ax.errorbar(
                freq_to_k(grp["freq"].values), grp["mean"],
                yerr=grp["std"].fillna(0),
                color=CFG_COLOR[cfg], marker=marker,
                markersize=6, linewidth=1.4, ls=WIND_LS[wind],
                capsize=3, mec="black", mew=0.4, alpha=0.95,
            )

    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.4)
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(k_min, k_max)
    ax.grid(True, alpha=0.3)

    # Top axis: paddle frequency. Show every freq present in the data.
    secax = add_freq_axis(ax)
    secax.set_xticks(freqs_present)
    secax.set_xticklabels([f"{f:.2f}" for f in freqs_present], fontsize=7)
    ax.set_xlabel("$k$", fontsize=11)
    ax.xaxis.set_label_coords(1.02, -0.025)
    secax.set_xlabel("$f$", fontsize=11)
    secax.xaxis.set_label_coords(1.02, 1.025)

    # Legend: probe config (colour) + wind (linestyle) — split into two
    # compact mini-legends so neither axis is cluttered.
    cfg_handles = [
        mlines.Line2D([], [], color=CFG_COLOR[c], marker=marker,
                      ms=6, lw=1.4, mec="black", mew=0.4,
                      label=c + (" (canon)" if c == "100/low" else ""))
        for c in CFG_ORDER if (sub["cfg"] == c).any()
    ]
    wind_handles = [
        mlines.Line2D([], [], color="black", lw=1.4, ls=WIND_LS[w],
                      label=WIND_LABEL[w])
        for w in ["no", "full"]
    ]
    leg_cfg = ax.legend(handles=cfg_handles, loc="lower left",
                         fontsize=7.5, framealpha=0.92,
                         title="probe cfg", title_fontsize=7.5)
    ax.add_artist(leg_cfg)
    ax.legend(handles=wind_handles, loc="lower right",
               fontsize=7.5, framealpha=0.92,
               title="vind", title_fontsize=7.5)

    fig.subplots_adjust(left=0.07, right=0.97, top=0.85, bottom=0.18)
    apply_horizontal_ylabel(ax, r"$K_t$", fontsize=12)

    # Subfig title — small, top-left, names the amp tier.
    ax.text(0.005, 0.985, amp_tag, transform=ax.transAxes,
             ha="left", va="top", fontsize=10, color="#444",
             bbox=dict(boxstyle="round,pad=0.18",
                       facecolor="white", alpha=0.85, edgecolor="#bbb"))

    scratch = (Path(__file__).parent
                / f"probe_config_bias_thesis_{amp_tag}.pdf")
    out_pdf = (BASE / "output" / "FIGURES"
                / f"ch05_probe_config_bias_{amp_tag}.pdf")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(scratch, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    print(f"   [{amp_tag}] Saved → {scratch.relative_to(BASE)}")
    print(f"   [{amp_tag}] Saved → {out_pdf.relative_to(BASE)}")
    plt.close(fig)

print("\nDone.")
