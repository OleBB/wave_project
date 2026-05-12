"""
Wind-Doppler arrival-shift study (2026-05-05)
=============================================

Question: under fullwind, the paddle wave arrives ~0.15-0.25 T earlier at
the IN probe than under nowind. Is this consistent with wind-driven
surface-current Doppler (H1)? Does the IN-OUT leg show a different shift
(suggesting a lee-side counter-current behind the panel, H1a)?

Method
------
For each (freq, amp, wind) cell on canon March-2026 lowrange + full panel:
  1. Compute mean snap-shift per probe in samples and seconds.
  2. Δt(probe) = mean_snap_fw[probe] - mean_snap_nw[probe], in seconds.
     (Theoretical hg_expected_start is identical fw vs nw — verified.)
  3. Plot Δt vs probe distance r per cell. Under uniform-wind H1:
        Δt(r) = -r·U/c_g²   (linear through origin, slope = -U/c_g²)
     Under H1a (lee-side counter-current):
        slope changes between r=9.373 (IN) and r=12.4 (OUT).
  4. Solve for U on each leg (paddle->IN, IN->OUT) per cell.
  5. Visual proof: overlay η(t) at IN for one cell, fw vs nw, with
     detected upcrossings marked.

Filters
-------
Canon: PROCESSED-20260326/27 lowrange.
Panel: full only. Quality: ok. Mooring: pooled.
Frequencies: 1.3, 1.4, 1.5, 1.6 Hz. Amplitudes: 0.1, 0.2, 0.3 V.

Outputs
-------
analysis_scratch/wind_doppler_arrival_shift.csv         per-cell numbers
analysis_scratch/wind_doppler_arrival_shift_legs.csv    per-leg current
analysis_scratch/wind_doppler_arrival_shift_dr.png      Δt vs r grid
analysis_scratch/wind_doppler_arrival_shift_eta.png     η(t) overlay
analysis_scratch/wind_doppler_arrival_shift.md          findings
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from wavescripts.constants import HG, c_group  # noqa: E402
from wavescripts.plot_utils import apply_thesis_style  # noqa: E402
apply_thesis_style()

OUT_DIR = BASE / "analysis_scratch"
FS = 250.0  # sampling rate (Hz)

# Canon dirs (CH05 results)
CANON = [
    BASE / "waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange",
    BASE / "waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange",
]

PROBES = [
    ("8804/250",  8.804),
    ("9373/170",  9.373),
    ("9373/340",  9.373),
    ("12400/250", 12.400),
]
PROBE_R = {p: r for p, r in PROBES}
PROBE_COLOR = {
    "8804/250":  "#888888",   # upstream
    "9373/170":  "#1f77b4",   # IN wall
    "9373/340":  "#aec7e8",   # IN far
    "12400/250": "#d62728",   # OUT
}

FREQS = [1.3, 1.4, 1.5, 1.6]
AMPS  = [0.1, 0.2, 0.3]
WINDS = ["no", "full"]

WIND_COLOR_OVERLAY = {"no": "#1f77b4", "full": "#d62728"}
WIND_LABEL = {"no": "Uten vind", "full": "Full vind"}

PROBE_LABEL_NO = {
    "8804/250":  "Foran",
    "9373/170":  "Innkommende",
    "9373/340":  "Innkommende",
    "12400/250": "Utgående",
}
AMP_LABEL = {0.1: r"$A_1$", 0.2: r"$A_2$", 0.3: r"$A_3$"}

# A4 width minus 1 inch (narrow margins) → matches \linewidth in thesis.
A4_W_IN = 8.27
FIG_W   = A4_W_IN - 1.0  # 7.27"


def _info_box_text(fw_runs, nw_runs, f_hz, amp, probe):
    """Info-box string: probe label, amp/freq, Δt in the 7T–17T window.

    Δt = (start_fw − start_nw)/FS read from `Computed Probe {pos} start`
    in each run's meta — the snap-aligned anchor of the 10-period
    analysis window. Negative Δt = fullwind window starts earlier.
    """
    start_col = f"Computed Probe {probe} start"
    fw_val = fw_runs[start_col].iloc[0] if len(fw_runs) and start_col in fw_runs.columns else np.nan
    nw_val = nw_runs[start_col].iloc[0] if len(nw_runs) and start_col in nw_runs.columns else np.nan
    if not np.isfinite(fw_val) or not np.isfinite(nw_val):
        dt_str = r"$\Delta t$ = n/a"
    else:
        dt_ms = (float(fw_val) - float(nw_val)) / FS * 1000.0
        tag = "fullvind tidligere" if dt_ms < 0 else "fullvind senere"
        dt_str = rf"$\Delta t$ = {dt_ms:+.0f} ms ({tag})"
    return (
        f"{PROBE_LABEL_NO[probe]} ({probe})\n"
        f"{AMP_LABEL[amp]}, {f_hz:.1f} Hz\n"
        f"{dt_str}"
    )


# ---------------------------------------------------------------------------
# 1. Load meta (canon only, fullpanel + ok + thesis cells)
# ---------------------------------------------------------------------------
def load_canon_meta() -> pd.DataFrame:
    rows = []
    for d in CANON:
        with open(d / "meta.json") as f:
            data = json.load(f)
        for r in data:
            r["_folder"] = d.name
            rows.append(r)
    df = pd.DataFrame(rows)

    df = df[df["WaveFrequencyInput [Hz]"].notna()]
    df = df[df["PanelCondition"] == "full"]
    df = df[df["quality_flag"] == "ok"]
    df = df[df["WaveFrequencyInput [Hz]"].isin(FREQS)]
    df = df[df["WaveAmplitudeInput [Volt]"].isin(AMPS)]
    df = df[df["WindCondition"].isin(WINDS)]
    return df.copy()


# ---------------------------------------------------------------------------
# 2. Per-cell, per-probe snap shift table
# ---------------------------------------------------------------------------
def per_cell_table(meta: pd.DataFrame) -> pd.DataFrame:
    """Returns long-format: one row per (freq, amp, probe), with mean snap
    shift in seconds for fw and nw, and Δt = fw − nw."""
    rows = []
    for f_hz in FREQS:
        cg = c_group(f_hz, HG.TANK_DEPTH_M)
        for amp in AMPS:
            cell = meta[
                (meta["WaveFrequencyInput [Hz]"] == f_hz)
                & (meta["WaveAmplitudeInput [Volt]"] == amp)
            ]
            if cell.empty:
                continue
            for probe, r in PROBES:
                col = f"Probe {probe} hg_snap_shift"
                if col not in cell.columns:
                    continue
                # mean snap in samples per wind condition
                snaps = {}
                ns = {}
                for w in WINDS:
                    s = cell[cell["WindCondition"] == w][col].dropna()
                    snaps[w] = s.mean() if len(s) else np.nan
                    ns[w] = len(s)
                # convert to seconds
                snap_fw_s = snaps["full"] / FS
                snap_nw_s = snaps["no"] / FS
                dt_s = snap_fw_s - snap_nw_s
                # apparent uniform-current solution: Δt = -r·U/c_g²
                # (only valid if uniform from paddle to probe)
                u_apparent = -dt_s * cg ** 2 / r if np.isfinite(dt_s) else np.nan
                rows.append({
                    "f_hz": f_hz,
                    "amp_V": amp,
                    "probe": probe,
                    "r_m": r,
                    "c_g": cg,
                    "snap_fw_samples": snaps["full"],
                    "snap_nw_samples": snaps["no"],
                    "snap_fw_s": snap_fw_s,
                    "snap_nw_s": snap_nw_s,
                    "dt_s": dt_s,
                    "dt_T": dt_s * f_hz,
                    "u_apparent_mm_s": u_apparent * 1000,
                    "n_fw": ns["full"],
                    "n_nw": ns["no"],
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Per-leg apparent current
# ---------------------------------------------------------------------------
def per_leg_table(cells: pd.DataFrame) -> pd.DataFrame:
    """Decompose Δt into per-leg apparent current.

    Three legs:
        L1: paddle (r=0)  -> 8804/250  (Δr = 8.804)
        L2: 8804/250      -> 9373/170  (Δr = 0.569)
        L3: 9373/170      -> 12400/250 (Δr = 3.027)

    Δt(r) = sum over legs i of -Δr_i · U_i / c_g²  for legs upstream of r.

    Solve for U_1, U_2, U_3 from Δt at the three reference probes
    (8804, 9373, 12400). Use 9373/170 as the IN reference (parallel
    9373/340 is just a redundancy check, not a fourth equation).
    """
    rows = []
    for f_hz in FREQS:
        cg = c_group(f_hz, HG.TANK_DEPTH_M)
        for amp in AMPS:
            sub = cells[(cells["f_hz"] == f_hz) & (cells["amp_V"] == amp)]
            if sub.empty:
                continue

            def dt_at(probe):
                row = sub[sub["probe"] == probe]
                return float(row["dt_s"].iloc[0]) if len(row) else np.nan

            dt_8804  = dt_at("8804/250")
            dt_9373  = dt_at("9373/170")  # IN reference (wall)
            dt_9373f = dt_at("9373/340")  # IN parallel (far)
            dt_12400 = dt_at("12400/250")

            # Leg lengths
            L1, L2, L3 = 8.804, 9.373 - 8.804, 12.400 - 9.373

            # Legs assume cumulative effect:
            #   Δt(8804)  = -L1·U1/c²
            #   Δt(9373)  = -(L1·U1 + L2·U2)/c²
            #   Δt(12400) = -(L1·U1 + L2·U2 + L3·U3)/c²
            # Solve sequentially.
            U1 = -dt_8804 * cg ** 2 / L1
            U2 = -(dt_9373 - dt_8804) * cg ** 2 / L2
            U3 = -(dt_12400 - dt_9373) * cg ** 2 / L3

            rows.append({
                "f_hz": f_hz,
                "amp_V": amp,
                "c_g": cg,
                "dt_8804_s":  dt_8804,
                "dt_9373_s":  dt_9373,
                "dt_9373f_s": dt_9373f,
                "dt_12400_s": dt_12400,
                "U_paddle_to_8804_mm_s": U1 * 1000,
                "U_8804_to_9373_mm_s":   U2 * 1000,
                "U_9373_to_12400_mm_s":  U3 * 1000,
                # parallel-probe sanity
                "dt_9373_parallel_diff_s": dt_9373f - dt_9373,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Plot Δt vs r per cell
# ---------------------------------------------------------------------------
def plot_dt_vs_r(cells: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(len(FREQS), len(AMPS), figsize=(12, 10),
                             sharex=True, sharey=False)

    for i, f_hz in enumerate(FREQS):
        cg = c_group(f_hz, HG.TANK_DEPTH_M)
        for j, amp in enumerate(AMPS):
            ax = axes[i, j]
            sub = cells[(cells["f_hz"] == f_hz) & (cells["amp_V"] == amp)]
            if sub.empty:
                ax.set_visible(False)
                continue

            for _, row in sub.iterrows():
                ax.scatter(row["r_m"], row["dt_s"] * 1000,
                           color=PROBE_COLOR[row["probe"]], s=60,
                           edgecolor="k", lw=0.6, zorder=3,
                           label=row["probe"])

            # Two reference fits:
            #  H1 (uniform Doppler): linear through origin, slope from IN
            #  H_uniform (timing offset): horizontal at mean Δt
            in_row = sub[sub["probe"] == "9373/170"]
            if len(in_row):
                u_in = -float(in_row["dt_s"].iloc[0]) * cg ** 2 / 9.373
                rs = np.linspace(0, 13, 50)
                dt_pred = -rs * u_in / cg ** 2 * 1000
                ax.plot(rs, dt_pred, "--", color="grey", lw=1,
                        label=f"H1 Doppler  U={u_in*1000:.1f} mm/s")

                mean_dt = sub["dt_s"].mean() * 1000
                ax.axhline(mean_dt, color="orange", lw=1, ls=":",
                           label=f"uniform shift {mean_dt:.0f} ms")

                # Compute regression slope through origin and R² vs both models
                rs_d = sub["r_m"].values
                ys = sub["dt_s"].values * 1000
                # H1 (slope through origin, free slope)
                slope_h1 = np.sum(rs_d * ys) / np.sum(rs_d ** 2)
                pred_h1 = slope_h1 * rs_d
                # H_uniform (free constant)
                pred_un = np.mean(ys) * np.ones_like(ys)
                ss_tot = np.sum((ys - np.mean(ys)) ** 2)
                ss_h1  = np.sum((ys - pred_h1) ** 2)
                ss_un  = np.sum((ys - pred_un) ** 2)
                # Lower SSres = better fit
                better = "uniform" if ss_un < ss_h1 else "Doppler"
                ax.text(0.02, 0.05, f"better: {better}\n"
                        f"SSres uni={ss_un:.0f}\n"
                        f"SSres H1={ss_h1:.0f}",
                        transform=ax.transAxes, fontsize=6,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="grey", alpha=0.8))

            ax.axhline(0, color="k", lw=0.3)
            ax.axvline(9.373, color="grey", lw=0.3, ls=":")
            ax.set_title(f"{f_hz} Hz, {amp} V", fontsize=10)
            if i == len(FREQS) - 1:
                ax.set_xlabel("probe distance r [m]")
            if j == 0:
                ax.set_ylabel("Δt = fw − nw [ms]")
            ax.grid(alpha=0.3)

            if i == 0 and j == len(AMPS) - 1:
                ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("Wind-induced arrival shift vs probe distance "
                 "(canon, full panel, ok)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Visual: η(t) overlay for one cell
# ---------------------------------------------------------------------------
def plot_eta_overlay(meta: pd.DataFrame, out_path: Path,
                     f_pick: float = 1.4, amp_pick: float = 0.2) -> None:
    cell = meta[
        (meta["WaveFrequencyInput [Hz]"] == f_pick)
        & (meta["WaveAmplitudeInput [Volt]"] == amp_pick)
    ]
    fw_runs = cell[cell["WindCondition"] == "full"]
    nw_runs = cell[cell["WindCondition"] == "no"]
    if fw_runs.empty or nw_runs.empty:
        print(f"  no data for {f_pick} Hz / {amp_pick} V")
        return

    # Pick first run of each
    fw_path = fw_runs["path"].iloc[0]
    nw_path = nw_runs["path"].iloc[0]

    # Load processed_dfs from the corresponding canon folder
    eta_fw = _load_eta_for_run(fw_path)
    eta_nw = _load_eta_for_run(nw_path)
    if eta_fw is None or eta_nw is None:
        print("  missing eta in processed_dfs")
        return

    # 4 panels: 8804, 9373/170, 12400/250 — 3 probes, with zoom
    probes = ["8804/250", "9373/170", "12400/250"]
    fig, axes = plt.subplots(len(probes), 2, figsize=(15, 9),
                             sharex="col", gridspec_kw={"width_ratios": [3, 2]})
    for row, probe in enumerate(probes):
        eta_col = f"eta_{probe}"
        if eta_col not in eta_fw.columns or eta_col not in eta_nw.columns:
            for ax in axes[row]:
                ax.set_visible(False)
            continue
        t_fw = np.arange(len(eta_fw)) / FS
        t_nw = np.arange(len(eta_nw)) / FS
        eta_nw_arr = eta_nw[eta_col].values
        eta_fw_arr = eta_fw[eta_col].values

        r = PROBE_R[probe]
        T_start = r * f_pick / c_group(f_pick) + HG.N_OFFSET
        t_exp = T_start / f_pick

        # Get post-snap starts
        col_start = f"Computed Probe {probe} start"
        t_snap_nw = nw_runs[col_start].iloc[0] / FS if col_start in nw_runs.columns else None
        t_snap_fw = fw_runs[col_start].iloc[0] / FS if col_start in fw_runs.columns else None

        for ax, xlim in zip(axes[row], [(t_exp - 4, t_exp + 8),
                                         (t_exp - 1.0, t_exp + 1.0)]):
            ax.plot(t_nw, eta_nw_arr, color="#1f77b4", lw=0.9, label="nowind")
            ax.plot(t_fw, eta_fw_arr, color="#d62728", lw=0.9, alpha=0.85,
                    label="fullwind")
            ax.axhline(0, color="k", lw=0.3)
            ax.axvline(t_exp, color="k", ls=":", lw=0.7, label="H&G expected")
            if t_snap_nw is not None:
                ax.axvline(t_snap_nw, color="#1f77b4", lw=1.2, alpha=0.7,
                           label=f"nw snap ({t_snap_nw:.3f}s)")
            if t_snap_fw is not None:
                ax.axvline(t_snap_fw, color="#d62728", lw=1.2, alpha=0.7,
                           label=f"fw snap ({t_snap_fw:.3f}s)")
                if t_snap_nw is not None:
                    dt_ms = (t_snap_fw - t_snap_nw) * 1000
                    ax.text(0.5, 0.95, f"Δt = {dt_ms:.0f} ms",
                            transform=ax.transAxes, ha="center", va="top",
                            fontsize=10,
                            bbox=dict(boxstyle="round,pad=0.3",
                                      fc="lightyellow", ec="grey"))
            ax.set_xlim(xlim)
            ax.grid(alpha=0.3)
            if row == 0:
                ax.legend(fontsize=7, loc="upper right")
        axes[row, 0].set_ylabel(f"η [mm]\n{probe} (r={r}m)")

    axes[-1, 0].set_xlabel("time [s] — wide view")
    axes[-1, 1].set_xlabel("time [s] — zoomed at H&G expected")
    fig.suptitle(f"η(t) overlay — f={f_pick} Hz, A={amp_pick} V, full panel\n"
                 f"vertical lines: black=theory (identical fw/nw), "
                 f"blue=nw snap, red=fw snap", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_pre_paddle(meta: pd.DataFrame, out_path: Path,
                    f_pick: float = 1.3, amp_pick: float = 0.2,
                    thesis_pdf: Path | None = None,
                    thesis_name: str | None = None) -> None:
    """Show η in the FIRST 25 s — covers the pre-paddle stillwater +
    paddle ramp + first wave arrival. Compares fw vs nw at IN.

    The whole question: is the paddle START different between fw and nw,
    or is the wave-train arrival different given identical paddle starts?

    If paddle motion shows up at the same recorded time in both runs →
    the apparent Δt is propagation/detection. If paddle motion itself is
    shifted → it's a hardware/recording timing offset, not physics.
    """
    cell = meta[
        (meta["WaveFrequencyInput [Hz]"] == f_pick)
        & (meta["WaveAmplitudeInput [Volt]"] == amp_pick)
    ]
    fw_runs = cell[cell["WindCondition"] == "full"]
    nw_runs = cell[cell["WindCondition"] == "no"]
    if fw_runs.empty or nw_runs.empty:
        return

    fw_path = fw_runs["path"].iloc[0]
    nw_path = nw_runs["path"].iloc[0]
    eta_fw = _load_eta_for_run(fw_path)
    eta_nw = _load_eta_for_run(nw_path)
    if eta_fw is None or eta_nw is None:
        return

    fig, axes = plt.subplots(3, 1, figsize=(FIG_W, 5.5), sharex=True)
    fig.subplots_adjust(left=0.065, right=0.995, top=0.965,
                        bottom=0.10, hspace=0.18)

    probes_stacked = ["8804/250", "9373/170", "12400/250"]
    for ax, probe in zip(axes, probes_stacked):
        eta_col = f"eta_{probe}"
        if eta_col not in eta_fw.columns:
            continue
        t_fw = np.arange(len(eta_fw)) / FS
        t_nw = np.arange(len(eta_nw)) / FS
        ax.plot(t_nw, eta_nw[eta_col].values,
                color=WIND_COLOR_OVERLAY["no"], lw=0.6,
                label=WIND_LABEL["no"])
        ax.plot(t_fw, eta_fw[eta_col].values,
                color=WIND_COLOR_OVERLAY["full"], lw=0.6, alpha=0.85,
                label=WIND_LABEL["full"])

        # Theoretical first-arrival time at this probe (faint guide)
        r = PROBE_R[probe]
        t_arr = r / c_group(f_pick)
        ax.axvline(t_arr, color="k", ls=":", lw=0.5)

        ax.grid(alpha=0.3)
        ax.axhline(0, color="k", lw=0.3)

        # Info box per probe (probe label / amp / freq / Δt 7T–17T window)
        ax.text(0.012, 0.96,
                _info_box_text(fw_runs, nw_runs, f_pick, amp_pick, probe),
                transform=ax.transAxes, fontsize=8, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="grey", alpha=0.9))

        if probe == probes_stacked[0]:
            ax.legend(fontsize=8, loc="upper right")

    axes[-1].set_xlabel("Tid [s]")
    axes[-1].set_xlim(0, 25)

    # y-axis label lifted to figure top-left
    fig.text(0.006, 0.985, r"$\eta$ [mm]", fontsize=10,
             va="top", ha="left")

    fig.savefig(out_path, dpi=130)
    if thesis_pdf is not None:
        thesis_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(thesis_pdf)
    plt.close(fig)

    if thesis_name is not None:
        _write_pre_paddle_stub(fw_runs, nw_runs, f_pick, amp_pick, thesis_name)


def _write_pre_paddle_stub(fw_runs, nw_runs, f_hz, amp, figure_name):
    import wavescripts.plot_utils as pu
    pu.TEXFIGU_DIR = BASE / "output" / "TEXFIGU"
    pu.FIGURES_DIR = BASE / "output" / "FIGURES"

    fw_path = fw_runs["path"].iloc[0] if len(fw_runs) else None
    nw_path = nw_runs["path"].iloc[0] if len(nw_runs) else None
    _meta = pu.build_fig_meta(
        {
            "filters": {
                "PanelCondition":            "full",
                "WaveFrequencyInput [Hz]":   f_hz,
                "WaveAmplitudeInput [Volt]": amp,
                "WindCondition":             ["no", "full"],
                "quality_flag":              "ok",
                "probes":                    "8804/250+9373/170+12400/250",
            },
            "plotting": {"figure_name": figure_name},
        },
        chapter="04",
        extra={"script": "analysis_scratch/wind_doppler_arrival_shift.py"},
        computed_in=("analysis_scratch/wind_doppler_arrival_shift.py "
                     "(pre-paddle window 0–25 s, eta vs t at all 3 probes; "
                     "compares wind-wave background at upstream IN probes vs "
                     "panel-shadowed OUT probe)"),
        data_class="DELEG",
        findings_doc="memory/methodology_wind_enhances_A_in.md",
        extra_params=(
            f"runs: nw={nw_path}, fw={fw_path}. "
            f"plot covers t ∈ (0, 25 s) — entire pre-paddle + chirp + first arrival. "
            f"vertical dotted line per panel = r/c_g(f) for that probe."
        ),
        max_run_paths=4,
    )
    pu.write_figure_stub(_meta, plot_type="pre_paddle_overlay",
                         subfig_filenames=[figure_name], force=True)
    print(f"   stub → output/TEXFIGU/{figure_name}.tex")


def rank_linear_vs_flat(cells: pd.DataFrame) -> None:
    """For each (freq, amp) cell, decide which model fits better:
    - Doppler (linear through origin):  Δt = -r·U/c²
    - Uniform (flat):                  Δt = const
    Print the count and the cell-level breakdown."""
    print(f"  {'cell':>10}  {'SSres uni':>10}  {'SSres dop':>10}  {'better':>9}")
    n_uni = n_dop = 0
    for f_hz in FREQS:
        for amp in AMPS:
            sub = cells[(cells["f_hz"] == f_hz) & (cells["amp_V"] == amp)]
            if sub.empty or sub["dt_s"].isna().all():
                continue
            # Skip cells where any probe is period-aliased (|Δt_T| > 0.5)
            if (sub["dt_T"].abs() > 0.5).any():
                continue
            rs = sub["r_m"].values
            ys = sub["dt_s"].values * 1000  # ms
            slope = np.sum(rs * ys) / np.sum(rs ** 2)
            mean = np.mean(ys)
            ss_dop = np.sum((ys - slope * rs) ** 2)
            ss_uni = np.sum((ys - mean) ** 2)
            better = "uniform" if ss_uni < ss_dop else "Doppler"
            n_uni += better == "uniform"
            n_dop += better == "Doppler"
            print(f"  {f_hz:.1f}Hz/{amp:.1f}V  {ss_uni:>10.0f}  "
                  f"{ss_dop:>10.0f}  {better:>9}")
    print(f"\n  Total clean cells: uniform-fit better in "
          f"{n_uni}, Doppler-fit better in {n_dop}.")


def _load_eta_for_run(run_path: str) -> pd.DataFrame | None:
    """Load just the rows matching `run_path` from the corresponding
    PROCESSED-* processed_dfs.parquet."""
    rp = Path(run_path)
    # Determine which canon folder this run is in
    for d in CANON:
        wave_name = d.name.removeprefix("PROCESSED-")
        if wave_name in str(rp):
            df = pd.read_parquet(d / "processed_dfs.parquet")
            sub = df[df["_path"] == run_path]
            if len(sub):
                return sub.reset_index(drop=True)
    return None


# ---------------------------------------------------------------------------
# 6. Driver
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"Loading canon meta from {len(CANON)} folders...")
    meta = load_canon_meta()
    print(f"  {len(meta)} runs after filters")
    print(f"  freqs: {sorted(meta['WaveFrequencyInput [Hz]'].unique())}")
    print(f"  amps : {sorted(meta['WaveAmplitudeInput [Volt]'].unique())}")
    print(f"  winds: {sorted(meta['WindCondition'].unique())}")

    print("\nBuilding per-cell, per-probe snap shift table...")
    cells = per_cell_table(meta)
    csv_cells = OUT_DIR / "wind_doppler_arrival_shift.csv"
    cells.to_csv(csv_cells, index=False)
    print(f"  → {csv_cells.relative_to(BASE)} ({len(cells)} rows)")

    # Print summary table: dt in ms, per probe, by cell
    print("\n=== Δt = fw − nw [ms] (negative = fullwind earlier) ===")
    pivot = cells.pivot_table(
        index=["f_hz", "amp_V"], columns="probe", values="dt_s") * 1000
    print(pivot.round(1).to_string())

    print("\n=== Δt = fw − nw [periods] ===")
    pivot_T = cells.pivot_table(
        index=["f_hz", "amp_V"], columns="probe", values="dt_T")
    print(pivot_T.round(3).to_string())

    print("\nBuilding per-leg apparent current...")
    legs = per_leg_table(cells)
    csv_legs = OUT_DIR / "wind_doppler_arrival_shift_legs.csv"
    legs.to_csv(csv_legs, index=False)
    print(f"  → {csv_legs.relative_to(BASE)} ({len(legs)} rows)")

    print("\n=== Apparent surface current per leg [mm/s] ===")
    print(legs[["f_hz", "amp_V",
                "U_paddle_to_8804_mm_s",
                "U_8804_to_9373_mm_s",
                "U_9373_to_12400_mm_s"]].round(1).to_string(index=False))

    print("\n=== Parallel-probe sanity (9373/170 vs 9373/340), Δt diff [ms] ===")
    print((legs[["f_hz", "amp_V", "dt_9373_parallel_diff_s"]]
           .assign(dt_diff_ms=lambda d: d["dt_9373_parallel_diff_s"] * 1000)
           .drop(columns="dt_9373_parallel_diff_s")
           .round(1).to_string(index=False)))

    print("\nPlotting Δt vs r grid...")
    plot_dt_vs_r(cells, OUT_DIR / "wind_doppler_arrival_shift_dr.png")
    print(f"  → analysis_scratch/wind_doppler_arrival_shift_dr.png")

    # Use the cleanest cell for visual: 1.3 Hz, 0.2 V (uniform Δt = -134 ms)
    print("\nPlotting η(t) overlay (1.3 Hz, 0.2 V — cleanest cell)...")
    plot_eta_overlay(meta, OUT_DIR / "wind_doppler_arrival_shift_eta.png",
                     f_pick=1.3, amp_pick=0.2)
    print(f"  → analysis_scratch/wind_doppler_arrival_shift_eta.png")

    print("\nPlotting pre-paddle window (paddle-trigger sanity check)...")
    plot_pre_paddle(meta, OUT_DIR / "wind_doppler_arrival_shift_paddle_start.png",
                    f_pick=1.3, amp_pick=0.2,
                    thesis_pdf=BASE / "output/FIGURES/ch04_wind_pre_paddle_overlay.pdf",
                    thesis_name="ch04_wind_pre_paddle_overlay")
    print(f"  → analysis_scratch/wind_doppler_arrival_shift_paddle_start.png "
          f"+ output/FIGURES/ch04_wind_pre_paddle_overlay.pdf")

    print("\nLinear vs flat fit ranking across all cells:")
    rank_linear_vs_flat(cells)

    # Quick markdown summary
    md = OUT_DIR / "wind_doppler_arrival_shift.md"
    with open(md, "w") as f:
        f.write(_make_findings_md(cells, legs))
    print(f"\nFindings → {md.relative_to(BASE)}")


def _make_findings_md(cells: pd.DataFrame, legs: pd.DataFrame) -> str:
    in_only = cells[cells["probe"] == "9373/170"]
    median_dt_in = in_only["dt_s"].median() * 1000
    median_dt_T = in_only["dt_T"].median()

    # Cleanest cell: 1.3 Hz, 0.2 V
    clean = cells[(cells["f_hz"] == 1.3) & (cells["amp_V"] == 0.2)].sort_values("r_m")

    lines = [
        "# Wind arrival-shift study (2026-05-05)",
        "",
        "**Headline observation**: under fullwind, the paddle wave H&G",
        f"window snaps **{abs(median_dt_in):.0f} ms (≈{abs(median_dt_T):.2f} T) earlier at IN**",
        "than under nowind — and the shift is **approximately the same size**",
        "**at every probe**, regardless of distance from the paddle.",
        "",
        "## What I expected vs what I found",
        "",
        "Initial hypothesis (H1): wind-driven surface current adds to c_g.",
        "Prediction: Δt(r) = -r·U/c_g², linear through origin.",
        "",
        "**Result**: Δt(r) is essentially **flat** in r in the cleanest cells,",
        "not linear. Cleanest example — 1.3 Hz, A=0.2 V (canon, full panel):",
        "",
        clean[["probe", "r_m", "dt_s", "dt_T", "n_fw", "n_nw"]]
            .assign(dt_ms=lambda d: d["dt_s"] * 1000)
            .drop(columns="dt_s")
            .round(3).to_string(index=False),
        "",
        "All four probes shift by ~134 ms, irrespective of distance",
        "(r ranges from 8.8 m to 12.4 m, a 41 % spread). A Doppler model",
        "would predict the OUT shift to be 12.4/8.8 = 1.41× the 8804 shift.",
        "",
        "Across all 9 'clean' cells (no period-aliasing on any probe), a",
        "uniform-shift model fits better than a linear-Doppler model in 7/9.",
        "",
        "## What this rules out",
        "",
        "- **Simple uniform-current Doppler.** Killed by the flat Δt(r).",
        "- **Detection-threshold bias from envelope amplitude (H3 from",
        "  earlier discussion).** Would scale with envelope slope at each",
        "  probe; OUT envelope is ~5× smaller than IN, so OUT shift should",
        "  be much larger if H3 dominated. Observed: OUT shift ≈ IN shift.",
        "",
        "## What survives — candidate explanations",
        "",
        "*Candidate H4* — **Wave-source timing shift.** Under fullwind there",
        "are pre-existing wind-waves at the paddle. The first detectable",
        "'paddle-frequency' upcrossing forms slightly earlier because the",
        "wind-wave carrier and the paddle wave constructively combine. The",
        "wave field then propagates at normal c_g and arrives uniformly",
        "earlier at every probe. Predicts a flat Δt(r). **Consistent.**",
        "",
        "*Candidate H7* — **Paddle hardware response under wind load.** Air",
        "drag on the paddle face could shift its actual motion onset by",
        "~100 ms relative to the command signal. A trigger-time shift at",
        "the source predicts a flat Δt(r) at all probes. **Consistent.**",
        "Test: inspect paddle command/feedback channel if logged.",
        "",
        "*Candidate H8* — **Snap-anchoring on wind-wave upcrossings.** The",
        "H&G snap finds the nearest zero-upcrossing within ±T of the",
        "theoretical start. Under fullwind at IN/8804 there are wind-wave",
        "upcrossings every ~0.2-0.3 s — the snap may bias toward an earlier",
        "one. **Inconsistent with the uniform shift**: OUT (no wind waves —",
        "verified in the pre-paddle plot) should not have this bias, yet",
        "OUT also shifts by the same amount.",
        "",
        "## Pre-paddle sanity check (paddle_start.png)",
        "",
        "Pre-paddle η at OUT shows essentially no wind-wave activity",
        "(panel does shadow effectively). Pre-paddle η at 8804 and IN",
        "shows clear wind-wave noise (~5-10 mm). Both fw and nw recordings",
        "appear to share the same recorded t=0; no obvious paddle-trigger",
        "offset is visible. **H7 hardware shift cannot be confirmed from",
        "η alone — would need the paddle command channel.**",
        "",
        "## Per-leg 'apparent current' if you insist on Doppler",
        "",
        legs[["f_hz", "amp_V",
              "U_paddle_to_8804_mm_s",
              "U_8804_to_9373_mm_s",
              "U_9373_to_12400_mm_s"]].round(1).to_string(index=False),
        "",
        "These numbers are physically suspect: the paddle→8804 leg shows",
        "a tiny consistent ~3-5 mm/s, the 8804→9373 leg shows much larger",
        "20-50 mm/s, and the 9373→12400 (panel-shadow) leg fluctuates",
        "wildly including negative values. None of this is consistent with",
        "a real surface current. The decomposition is the math forcing a",
        "flat Δt(r) into a linear-Doppler frame.",
        "",
        "## Interpretation",
        "",
        "The wave field arrives uniformly earlier at every probe under",
        "fullwind. This is a **source-side or detection-side** effect, not",
        "a propagation effect. Most likely candidates are H4 (wave-",
        "generation timing shifted by pre-existing wind-wave field) or",
        "H7 (paddle hardware response under wind load).",
        "",
        "**This does not refute the wind→IN coupling story** documented in",
        "`methodology_wind_enhances_A_in.md` (10-17% A_in enhancement). The",
        "amplitude enhancement and the timing shift are two distinct",
        "observations that may share a single source (wind interacts with",
        "wave generation or first cycles) but are independent measurements.",
        "",
        "## Files",
        "- wind_doppler_arrival_shift.csv — per-cell, per-probe table",
        "- wind_doppler_arrival_shift_legs.csv — per-leg apparent current",
        "- wind_doppler_arrival_shift_dr.png — Δt vs r scatter (key plot)",
        "- wind_doppler_arrival_shift_eta.png — η(t) overlay 1.3 Hz 0.2 V",
        "- wind_doppler_arrival_shift_paddle_start.png — pre-paddle sanity",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
