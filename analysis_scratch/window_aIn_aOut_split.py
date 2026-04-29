"""
A_IN / A_OUT split — does adding periods shrink the ratio because of FFT
artifact, or because the underlying signals diverge?

Re-uses window_squeeze_summary.csv (per40 at 1.6 Hz, N_offset=10, N_length
swept 5..30T) — the worst-case regime.

Top panel    — A_IN(N) and A_OUT(N) in mm, separately, normalized to N=10
                (each curve = 1.0 at N=10).
Bottom panel — OUT/IN(N), the ratio.

If the artifact theory is right: A_IN and A_OUT should fall together
(same Δf, same sinc characteristic), the ratio should stay flat.
If it's physics: A_IN and A_OUT diverge — typically A_IN grows
(reflections, wind broadband) while A_OUT shrinks (ringdown).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from wavescripts.plot_utils import apply_thesis_style, WIND_COLOR_MAP

apply_thesis_style()

CSV = Path(__file__).parent / "window_squeeze_summary.csv"
OUT_PNG = Path(__file__).parent / "window_aIn_aOut_split.png"
OUT_PDF = Path(__file__).parent / "window_aIn_aOut_split.pdf"

df = pd.read_csv(CSV)
df = df[(df["step"] == 2) & (df["N_offset_T"] == 10)].copy()
print(f"per40 1.6 Hz, N_offset=10: {len(df)} rows · "
      f"{df['path'].nunique()} runs · winds {sorted(df['wind'].unique())}")

# Normalize each (run) curve so A(N=10) = 1.0
# Normalize each (path) curve so A(N=10) = 1.0
df = df.sort_values(["path", "N_length_T"]).reset_index(drop=True)
df["A_in_norm"]  = np.nan
df["A_out_norm"] = np.nan
for path in df["path"].unique():
    mask = df["path"] == path
    sub = df[mask]
    ref_in  = sub.loc[sub["N_length_T"] == 10, "A_in"].iloc[0]
    ref_out = sub.loc[sub["N_length_T"] == 10, "A_out"].iloc[0]
    df.loc[mask, "A_in_norm"]  = df.loc[mask, "A_in"]  / ref_in
    df.loc[mask, "A_out_norm"] = df.loc[mask, "A_out"] / ref_out

fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

ax = axes[0]
for path, g in df.groupby("path"):
    g = g.sort_values("N_length_T")
    wind = g["wind"].iloc[0]
    color = WIND_COLOR_MAP[wind]
    ax.plot(g["N_length_T"], g["A_in_norm"],
            color=color, ls="-",  marker="o", ms=4, lw=1.4,
            label=f"A_IN  ({wind})")
    ax.plot(g["N_length_T"], g["A_out_norm"],
            color=color, ls="--", marker="s", ms=4, lw=1.4, mfc="white",
            label=f"A_OUT ({wind})")

ax.axhline(1.0, color="black", lw=0.5, alpha=0.4)
ax.axvline(10, color="#2ECC71", lw=1.0, ls=":", alpha=0.7,
           label="reference  N=10")
ax.axvline(15, color="#1A6E2A", lw=1.0, ls="-", alpha=0.4,
           label="chosen N(1.6)=15")
ax.set_ylabel("A(N) / A(N=10)   [unitless]", fontsize=10)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8, loc="lower left", ncol=2)
ax.set_title("per40, f = 1.6 Hz, 0.2 V, full panel — N_offset=10",
             fontsize=10)

ax = axes[1]
for path, g in df.groupby("path"):
    g = g.sort_values("N_length_T")
    wind = g["wind"].iloc[0]
    color = WIND_COLOR_MAP[wind]
    ax.plot(g["N_length_T"], g["OUT_IN"],
            color=color, ls="-", marker="o", ms=4, lw=1.4,
            label=f"OUT/IN  ({wind})")

ax.axvline(10, color="#2ECC71", lw=1.0, ls=":", alpha=0.7)
ax.axvline(15, color="#1A6E2A", lw=1.0, ls="-", alpha=0.4)
ax.set_xlabel("N_length [periods]", fontsize=10)
ax.set_ylabel("OUT/IN (FFT)", fontsize=10)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8, loc="lower left")

fig.tight_layout()
fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"saved → {OUT_PDF.relative_to(BASE)}")
print(f"        {OUT_PNG.relative_to(BASE)}")

# Numerical summary
print("\nNumerical summary — drift from N=10 reference:")
print(f"{'wind':<6} {'metric':<10} {'N=5':>8} {'N=10':>8} {'N=15':>8} {'N=20':>8} {'N=25':>8} {'N=30':>8}")
for path, g in df.groupby("path"):
    g = g.sort_values("N_length_T")
    wind = g["wind"].iloc[0]
    for col, label in [("A_in_norm", "A_IN  "),
                       ("A_out_norm", "A_OUT "),
                       ("OUT_IN",    "OUT/IN")]:
        ref10 = g.loc[g["N_length_T"] == 10, col].iloc[0] if label != "OUT/IN" else 1.0
        if label == "OUT/IN":
            vals = {N: g.loc[g["N_length_T"] == N, col].iloc[0]
                    for N in (5, 10, 15, 20, 25, 30)}
            cells = [f"{vals[N]:>8.4f}" for N in (5, 10, 15, 20, 25, 30)]
        else:
            vals = {N: (g.loc[g["N_length_T"] == N, col].iloc[0] - 1) * 100
                    for N in (5, 10, 15, 20, 25, 30)}
            cells = [f"{vals[N]:>+7.2f}%" for N in (5, 10, 15, 20, 25, 30)]
        print(f"{wind:<6} {label:<10} {' '.join(cells)}")
