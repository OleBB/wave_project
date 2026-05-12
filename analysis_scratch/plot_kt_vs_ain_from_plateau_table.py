"""Quick scatter of K_t vs A_in using the values printed in
output/TABLES/ch04_plateau_values.tex (table A.1).

Goal: see whether K_t depends on A_in alone, or whether wind/frequency
shift the K_t(A_in) trend.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# (amp_label, freq_Hz, wind, n, A_in_mm, A_out_mm, K_t, sigma_Kt)
ROWS = [
    ("A1", 1.3, "full",  10, 7.49,  5.85, 0.778, 0.046),
    ("A1", 1.3, "uten",   9, 7.50,  4.96, 0.662, 0.028),
    ("A1", 1.4, "full",   3, 8.01,  5.69, 0.710, 0.080),
    ("A1", 1.4, "uten",   2, 8.01,  4.06, 0.507, 0.036),
    ("A1", 1.5, "full",   3, 7.99,  5.67, 0.710, 0.076),
    ("A1", 1.5, "uten",   2, 7.67,  3.46, 0.452, 0.016),
    ("A1", 1.6, "full",   3, 8.29,  5.09, 0.597, 0.030),
    ("A1", 1.6, "uten",   2, 7.12,  2.61, 0.367, 0.005),
    ("A2", 1.3, "full",   3, 14.73, 12.75, 0.844, 0.019),
    ("A2", 1.3, "uten",   2, 14.76, 11.93, 0.809, 0.007),
    ("A2", 1.4, "full",   3, 15.56, 11.89, 0.744, 0.054),
    ("A2", 1.4, "uten",   2, 15.27, 10.48, 0.687, 0.003),
    ("A2", 1.5, "full",   3, 15.36, 11.35, 0.714, 0.045),
    ("A2", 1.5, "uten",   2, 14.98,  8.98, 0.599, 0.019),
    ("A2", 1.6, "full",   3, 15.28, 10.72, 0.689, 0.021),
    ("A2", 1.6, "uten",   2, 14.69,  7.13, 0.486, 0.000),
    ("A3", 1.3, "full",   4, 22.25, 18.51, 0.827, 0.016),
    ("A3", 1.3, "uten",   2, 21.52, 17.88, 0.831, 0.003),
    ("A3", 1.4, "full",   4, 23.11, 17.64, 0.768, 0.009),
    ("A3", 1.4, "uten",   2, 22.41, 15.03, 0.671, 0.005),
    ("A3", 1.5, "full",   4, 23.12, 17.05, 0.737, 0.010),
    ("A3", 1.5, "uten",   2, 21.48, 13.09, 0.609, 0.002),
    ("A3", 1.6, "full",   5, 23.08, 16.23, 0.702, 0.017),
    ("A3", 1.6, "uten",   3, 20.68, 11.01, 0.535, 0.025),
]

df = pd.DataFrame(
    ROWS,
    columns=["amp", "f_Hz", "wind", "n", "A_in_mm", "A_out_mm", "K_t", "sigma_Kt"],
)

FREQ_COLOR = {
    1.3: "#1f77b4",  # blue
    1.4: "#2ca02c",  # green
    1.5: "#ff7f0e",  # orange
    1.6: "#d62728",  # red
}
WIND_MARKER = {"full": "o", "uten": "s"}   # filled circle / square
WIND_FACE   = {"full": True, "uten": False}  # full=filled, uten=hollow

fig, ax = plt.subplots(figsize=(7.5, 5.0))

for _, r in df.iterrows():
    color = FREQ_COLOR[r["f_Hz"]]
    marker = WIND_MARKER[r["wind"]]
    facecolor = color if WIND_FACE[r["wind"]] else "white"
    ax.errorbar(
        r["A_in_mm"], r["K_t"],
        yerr=r["sigma_Kt"],
        fmt=marker,
        markerfacecolor=facecolor,
        markeredgecolor=color,
        ecolor=color,
        markersize=8,
        capsize=3,
        linewidth=1.0,
        zorder=3,
    )

# Connect points at same (freq, wind) across the three amplitudes
for f in sorted(df["f_Hz"].unique()):
    for wind in ("full", "uten"):
        sub = df[(df["f_Hz"] == f) & (df["wind"] == wind)].sort_values("A_in_mm")
        ls = "-" if wind == "full" else "--"
        ax.plot(
            sub["A_in_mm"], sub["K_t"],
            color=FREQ_COLOR[f],
            linestyle=ls,
            linewidth=1.0,
            alpha=0.55,
            zorder=2,
        )

# Legend: two groups
from matplotlib.lines import Line2D
freq_handles = [
    Line2D([0], [0], color=FREQ_COLOR[f], marker="o", linestyle="-",
           markersize=7, label=f"{f:.1f} Hz")
    for f in sorted(FREQ_COLOR)
]
wind_handles = [
    Line2D([0], [0], color="black", marker="o", linestyle="-",
           markerfacecolor="black", markersize=7, label="Full vind"),
    Line2D([0], [0], color="black", marker="s", linestyle="--",
           markerfacecolor="white", markersize=7, label="Uten vind"),
]
leg1 = ax.legend(handles=freq_handles, title="Frekvens",
                 loc="upper left", bbox_to_anchor=(1.01, 1.00), frameon=False)
ax.add_artist(leg1)
ax.legend(handles=wind_handles, title="Vind",
          loc="upper left", bbox_to_anchor=(1.01, 0.55), frameon=False)

# Annotate amplitude bands lightly
for amp, x_center in [("A1", 7.7), ("A2", 15.1), ("A3", 22.3)]:
    ax.text(x_center, 0.30, amp, ha="center", va="bottom",
            fontsize=10, color="0.4", zorder=1)

ax.set_xlabel(r"$A_\mathrm{Inn}$ [mm]  (paddle-frequency FFT amplitude on IN side)")
ax.set_ylabel(r"$K_t = A_\mathrm{Ut}/A_\mathrm{Inn}$")
ax.set_title("Transmisjonskoeffisient mot innkommende amplitude\n"
             "(canon Mars-2026 fullpanel, kilde: Table A.1 platåverdier)")
ax.set_ylim(0.30, 0.90)
ax.grid(True, alpha=0.3)

plt.tight_layout()

OUT = Path(__file__).resolve().parent / "plot_kt_vs_ain_from_plateau_table.pdf"
plt.savefig(OUT, bbox_inches="tight")
plt.savefig(OUT.with_suffix(".png"), dpi=150, bbox_inches="tight")
print(f"saved: {OUT}")
print(f"saved: {OUT.with_suffix('.png')}")
