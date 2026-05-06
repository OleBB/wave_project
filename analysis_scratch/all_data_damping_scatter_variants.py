"""
All-data damping scatter — exploratory variants
================================================

Two cuts of `all_data_damping_scatter.py`, parameterised by panel filter and
whether the canonical cond4_h100_low cluster is excluded:

  V1: PanelCondition ∈ {full, reverse}, all hardware
  V2: PanelCondition ∈ {full, reverse}, exclude canon (cond4_h100_low)

Same visual language as the published `ch05_damping_all_data_scatter`:
  wind → colour, amplitude → marker shape, hardware → fill.

Outputs (scratch — not wired into thesis):
  analysis_scratch/all_data_damping_scatter_v1_full_reverse.pdf
  analysis_scratch/all_data_damping_scatter_v2_full_reverse_no_canon.pdf
  output/FIGURES/ch05_damping_all_data_scatter_v1_full_reverse.pdf
  output/FIGURES/ch05_damping_all_data_scatter_v2_full_reverse_no_canon.pdf
  analysis_scratch/all_data_damping_scatter_variants_summary.csv

Run:
    conda run -n draumkvedet python analysis_scratch/all_data_damping_scatter_variants.py
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
from matplotlib.ticker import MultipleLocator

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import os
os.chdir(BASE)

from wavescripts.improved_data_loader import load_analysis_data
from wavescripts.constants import PROBE_HEIGHT_DEFAULT_MM
from wavescripts.plot_utils import (
    freq_to_k, add_freq_axis, WIND_COLOR_MAP, amp_to_label,
    apply_thesis_style,
)

apply_thesis_style()

SCRATCH_DIR = Path(__file__).parent
THESIS_DIR  = BASE / "output" / "FIGURES"
THESIS_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Load everything once ────────────────────────────────────────────────────
print("1. Loading all processed folders …")
all_dirs = sorted(glob.glob(str(BASE / "waveprocessed" / "PROCESSED-*")))
print(f"   {len(all_dirs)} folders")
meta, _, _, _ = load_analysis_data(*all_dirs, load_processed=False)
print(f"   {len(meta)} total rows\n")

def assign_condition(row):
    in_pos = row.get("in_position", None)
    if in_pos == "9373/250":
        return "legacy_nov2025"
    h = row.get("probe_height_mm", PROBE_HEIGHT_DEFAULT_MM)
    r = row.get("probe_range_mode", "high")
    if pd.isna(h):
        h = PROBE_HEIGHT_DEFAULT_MM
    h = int(h)
    if h == 272 and r == "high":
        return "cond1_h272_high"
    if h == 136 and r == "high":
        return "cond2_h136_high"
    if h == 100 and r == "high":
        return "cond3_h100_high_WRONG"
    if h == 100 and r == "low":
        return "cond4_h100_low"
    return "other"

meta["condition"] = meta.apply(assign_condition, axis=1)
FINAL_CONDITION = "cond4_h100_low"

# ── 2. Common filter helper ────────────────────────────────────────────────────
def filter_data(panels, exclude_canon: bool):
    sel = meta[
        meta["WaveFrequencyInput [Hz]"].notna()
        & (meta["WaveFrequencyInput [Hz]"] > 0)
        & meta["PanelCondition"].isin(panels)
        & (meta["quality_flag"] == "ok")
        & meta["OUT/IN (FFT)"].notna()
    ].copy()
    n_extreme = ((sel["OUT/IN (FFT)"] > 2.0) | (sel["OUT/IN (FFT)"] < 0.1)).sum()
    sel = sel[(sel["OUT/IN (FFT)"] <= 2.0) & (sel["OUT/IN (FFT)"] >= 0.1)]
    sel = sel[sel["WindCondition"].isin(["no", "full"])]
    n_drop_2hz = int((sel["WaveFrequencyInput [Hz]"] >= 2.0).sum())
    sel = sel[sel["WaveFrequencyInput [Hz]"] < 2.0].copy()
    sel["k"] = freq_to_k(sel["WaveFrequencyInput [Hz]"].values)
    sel["is_final"] = sel["condition"] == FINAL_CONDITION
    if exclude_canon:
        sel = sel[~sel["is_final"]].copy()
    return sel, int(n_extreme), n_drop_2hz

# ── 3. Plot helper ─────────────────────────────────────────────────────────────
WIND_LABEL = {"no": "uten vind", "full": "med vind"}
AMP_MARKER = {0.10: "o", 0.20: "s", 0.30: "^"}
AMP_MARKER_DEFAULT = "X"
MARKER_SIZE = 55
ALPHA_FILLED   = 0.65
ALPHA_HOLLOW   = 0.85
EDGE_LW_FILLED = 0.3
EDGE_LW_HOLLOW = 1.4

def _round_amp(v):
    return round(float(v), 2)

def make_scatter(wave_clip, panel_label: str, title_extra: str, out_pdfs: list[Path]):
    fig, ax = plt.subplots(figsize=(6.27, 9.5))

    for is_final in [False, True]:
        sub_h = wave_clip[wave_clip["is_final"] == is_final]
        if sub_h.empty:
            continue
        for wind, color in [("no", WIND_COLOR_MAP["no"]),
                            ("full", WIND_COLOR_MAP["full"])]:
            for amp_v, marker in AMP_MARKER.items():
                s = sub_h[(sub_h["WindCondition"] == wind)
                          & (sub_h["WaveAmplitudeInput [Volt]"]
                             .apply(_round_amp) == amp_v)]
                if s.empty:
                    continue
                if is_final:
                    fc, ec, lw, a = color, "black", EDGE_LW_FILLED, ALPHA_FILLED
                else:
                    fc, ec, lw, a = "none", color, EDGE_LW_HOLLOW, ALPHA_HOLLOW
                ax.scatter(
                    s["k"], s["OUT/IN (FFT)"],
                    facecolors=fc, edgecolors=ec, marker=marker,
                    s=MARKER_SIZE, linewidths=lw, alpha=a,
                    zorder=3 if is_final else 2,
                )

    # catch-all for unusual amps (rarely needed)
    _recognised_amps = set(AMP_MARKER.keys())
    unknown = wave_clip[~wave_clip["WaveAmplitudeInput [Volt]"]
                        .apply(_round_amp).isin(_recognised_amps)]
    if not unknown.empty:
        for is_final in [False, True]:
            u = unknown[unknown["is_final"] == is_final]
            if u.empty:
                continue
            for wind, color in [("no", WIND_COLOR_MAP["no"]),
                                ("full", WIND_COLOR_MAP["full"])]:
                uw = u[u["WindCondition"] == wind]
                if uw.empty:
                    continue
                ax.scatter(
                    uw["k"], uw["OUT/IN (FFT)"],
                    facecolors=(color if is_final else "none"),
                    edgecolors=("black" if is_final else color),
                    marker=AMP_MARKER_DEFAULT, s=MARKER_SIZE,
                    linewidths=(EDGE_LW_FILLED if is_final
                                else EDGE_LW_HOLLOW),
                    alpha=(ALPHA_FILLED if is_final else ALPHA_HOLLOW),
                    zorder=3 if is_final else 2,
                )

    thesis_k_lo = float(freq_to_k(np.array([1.3]))[0])
    thesis_k_hi = float(freq_to_k(np.array([1.6]))[0])
    ax.axvspan(thesis_k_lo, thesis_k_hi,
               color=WIND_COLOR_MAP["no"], alpha=0.07, lw=0, zorder=1)
    ax.text(thesis_k_hi - 0.1, 1.16,
            "Hovedfokus\n1,3–1,6 Hz", ha="right", va="top",
            fontsize=8, color="#1F618D", alpha=0.85,
            bbox=dict(boxstyle="round,pad=0.2",
                      facecolor="white", alpha=0.75, edgecolor="none"))

    ax.axhline(1.0, color="black", lw=0.6, ls="--", alpha=0.5)
    ax.set_xlabel("$k$", fontsize=11)
    ax.set_ylabel(r"$K_t$", fontsize=12, rotation=0, ha="left", va="bottom")
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.xaxis.set_minor_locator(MultipleLocator(1.0))
    ax.grid(which="major", alpha=0.30, lw=0.6)
    ax.grid(which="minor", alpha=0.15, lw=0.4)
    ax.set_ylim(0.1, 1.18)

    secax = add_freq_axis(ax)
    secax.set_xlabel(r"Frekvens (Hz)", fontsize=9)
    _used_freqs = sorted(wave_clip["WaveFrequencyInput [Hz]"].unique())
    secax.set_xticks(_used_freqs)
    secax.set_xticklabels([f"{f:.1f}" for f in _used_freqs])
    secax.tick_params(labelsize=7)

    wind_handles = [
        mlines.Line2D([], [], color=WIND_COLOR_MAP[w],
                      linestyle="-", linewidth=5, label=WIND_LABEL[w])
        for w in ["no", "full"]
    ]
    amp_handles = [
        mlines.Line2D([], [], color="black",
                      marker=AMP_MARKER[v], linestyle="None", markersize=8,
                      markerfacecolor="lightgray", markeredgecolor="black",
                      markeredgewidth=0.3, label=amp_to_label(v))
        for v in (0.10, 0.20, 0.30)
    ]
    hardware_handles = [
        mlines.Line2D([], [], color="black",
                      marker="o", linestyle="None", markersize=8,
                      markerfacecolor="black", markeredgecolor="black",
                      markeredgewidth=0.3,
                      label="endelig oppsett (h100/low)"),
        mlines.Line2D([], [], color="black",
                      marker="o", linestyle="None", markersize=8,
                      markerfacecolor="none", markeredgecolor="black",
                      markeredgewidth=1.4, label="tidligere oppsett"),
    ]
    leg_hw = ax.legend(handles=hardware_handles, loc="upper right",
                       bbox_to_anchor=(0.995, 0.995), fontsize=8,
                       framealpha=0.92, title="Eksperiment", title_fontsize=8)
    ax.add_artist(leg_hw)
    leg_w = ax.legend(handles=wind_handles, loc="upper right",
                      bbox_to_anchor=(0.995, 0.86), fontsize=8,
                      framealpha=0.92, title="Vind", title_fontsize=8)
    ax.add_artist(leg_w)
    ax.legend(handles=amp_handles, loc="upper right",
              bbox_to_anchor=(0.995, 0.74), fontsize=8,
              framealpha=0.92, title="Amplitude", title_fontsize=8)

    # Variant tag in upper-left so the variants don't get confused at a glance.
    ax.text(0.012, 0.995,
            f"Paneler: {panel_label}\n{title_extra}",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=8, color="#444",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="white", alpha=0.85, edgecolor="#bbb"))

    fig.subplots_adjust(left=0.10, right=0.98, top=0.95, bottom=0.06)
    fig.canvas.draw()
    _renderer = fig.canvas.get_renderer()
    _ticks = [t for t in ax.yaxis.get_ticklabels()
              if t.get_visible() and t.get_text().strip()]
    if _ticks:
        _left_disp = min(t.get_window_extent(renderer=_renderer).x0
                         for t in _ticks)
        _x_axes = ax.transAxes.inverted().transform((_left_disp, 0))[0]
        ax.yaxis.set_label_coords(_x_axes, 1.02)

    for p in out_pdfs:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight", pad_inches=0.02)
        print(f"   Saved → {p.relative_to(BASE)}")
    plt.close(fig)

# ── 4. Build the variants ──────────────────────────────────────────────────────
VARIANTS = [
    dict(
        tag="v1_full_reverse",
        panels=["full", "reverse"],
        exclude_canon=False,
        panel_label="full + reverse",
        title_extra="alle oppsett",
    ),
    dict(
        tag="v2_full_reverse_no_canon",
        panels=["full", "reverse"],
        exclude_canon=True,
        panel_label="full + reverse",
        title_extra="uten canon (cond4_h100_low)",
    ),
]

summary_rows = []
for v in VARIANTS:
    print(f"\n── Variant {v['tag']} ─────────────────────────")
    print(f"   panels={v['panels']}  exclude_canon={v['exclude_canon']}")
    sel, n_extreme, n_drop_2hz = filter_data(v["panels"], v["exclude_canon"])
    print(f"   {len(sel)} runs after filter "
          f"(clipped {n_extreme} extreme, dropped {n_drop_2hz} ≥2 Hz)")
    if len(sel) == 0:
        print("   ⚠ no rows — skipping plot")
        continue
    pivot = (sel.groupby(["PanelCondition", "WindCondition"])
                .size().unstack(fill_value=0))
    print("   counts (PanelCondition × WindCondition):")
    for line in pivot.to_string().splitlines():
        print(f"     {line}")
    # 0.65 Hz quick-check
    n_065 = int((sel["WaveFrequencyInput [Hz]"] == 0.65).sum())
    print(f"   rows at 0.65 Hz: {n_065}")
    out_pdfs = [
        SCRATCH_DIR / f"all_data_damping_scatter_{v['tag']}.pdf",
        THESIS_DIR  / f"ch05_damping_all_data_scatter_{v['tag']}.pdf",
    ]
    make_scatter(sel, v["panel_label"], v["title_extra"], out_pdfs)
    summary_rows.append(dict(
        tag=v["tag"],
        n=len(sel),
        n_final_cond4=int(sel["is_final"].sum()),
        n_earlier_hw=len(sel) - int(sel["is_final"].sum()),
        n_065hz=n_065,
        freq_min=float(sel["WaveFrequencyInput [Hz]"].min()),
        freq_max=float(sel["WaveFrequencyInput [Hz]"].max()),
        n_extreme_clipped=n_extreme,
        n_drop_2hz=n_drop_2hz,
    ))

if summary_rows:
    summary_df = pd.DataFrame(summary_rows)
    csv_path = SCRATCH_DIR / "all_data_damping_scatter_variants_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"\nSummary → {csv_path.relative_to(BASE)}")
    print(summary_df.to_string(index=False))

print("\nDone.")
