"""CLAUDE - The key addition is the propagated uncertainty — since v = sqrt(2*ΔP/ρ) is nonlinear, the Stan values in mA need to be scaled by dv/d(mA) evaluated at each second's operating point. This gives you honest error bars in m/s rather than mA. The rest of the plotting code from before stays the same."""
# %%
import re
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from datetime import datetime
from collections import defaultdict

# match the thesis font / sizes used by the wind-profile plot
sys.path.insert(0, os.path.expanduser("~/Kodevik/wave_project"))
from wavescripts.plot_utils import apply_thesis_style
apply_thesis_style()

# --- USER SETTINGS ---
ein_folder = r"/Users/ole/Kodevik/wave_project/pressuredata/20251107-fullwindUP2-allpanel-angleTest"

stats_folder = ein_folder + "/20251107-fullwindUP2-allpanel-angleTest-STATS"

angle_pattern = re.compile(r"ang(\d+)", re.IGNORECASE)
# sub_pattern = re.compile(r"ang(\d+)([A-Za-z]{2})", re.IGNORECASE)
sub_pattern = re.compile(r"ang(\d+)mm", re.IGNORECASE)

def parse_stats_file(file_path):
    """Read LabView stats .txt, return dict of column arrays."""
    cols = None
    data = {k: [] for k in ["Arit", "RMS", "Stan", "Vari", "Kurt", "Medi", "Mode", "Summ", "Skew"]}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # parse quoted, tab-separated fields
            fields = [f.strip().strip('"').replace(',', '.') for f in line.split('\t')]
            # header row detection
            if fields[0].startswith("Arit") or fields[0].startswith('"Arit'):
                cols = [f[:4] for f in fields]  # use first 4 chars as key
                continue
            if cols is None:
                # try to infer from position if no header found
                cols = list(data.keys())
            try:
                vals = [float(f) for f in fields]
                for k, v in zip(cols, vals):
                    short = k[:4]
                    if short in data:
                        data[short].append(v)
            except ValueError:
                continue
    return {k: np.array(v) for k, v in data.items()}


AIR_DENSITY = 1.225  # kg/m³, standard air at sea level

"""hugs at dette er basert på 4-20mA kalibrert til 0-100 Pa"""
def ma_to_windspeed(ma):
    dp = (ma - 4) / 16 * 100   # mA → Pa
    dp = np.maximum(dp, 0)      # clip negatives just in case
    return np.sqrt(2 * dp / AIR_DENSITY)

def process_stats_folder(folder_path):
    results = []
    for fname in os.listdir(folder_path):
        if not fname.lower().endswith(".txt"):
            continue
        match = angle_pattern.search(fname)
        if not match:
            continue
        submatch = sub_pattern.search(fname)
        prelim = not bool(submatch)

        angle = int(match.group(1))
        is_run2 = "run2" in fname.lower()
        file_path = os.path.join(folder_path, fname)
        d = parse_stats_file(file_path)

        # drop dead/zero rows
        mask = d["Arit"] > 0.1
        d = {k: v[mask] for k, v in d.items()}
        if len(d["Arit"]) == 0:
            continue

        n_seconds  = len(d["Arit"])

        # --- physics helpers ---
        def ma_to_pa(ma):
            return np.maximum((ma - 4.0) / 16.0 * 100.0, 0.0)

        def pa_to_v(pa):
            return np.sqrt(2.0 * np.maximum(pa, 0.001) / AIR_DENSITY)

        # per-second velocity (for drift)
        v_per_second = pa_to_v(ma_to_pa(d["Arit"]))
        mean_speed   = np.mean(v_per_second)

        # --- corrected uncertainty propagation ---
        # evaluate sensitivity at the mean operating point, not per-second
        ma_mean  = np.mean(d["Arit"])
        pa_mean  = float(ma_to_pa(ma_mean))
        pa_mean  = max(pa_mean, 0.001)

        dv_dpa   = 1.0 / np.sqrt(2.0 * pa_mean * AIR_DENSITY)   # (m/s)/Pa
        dv_dma   = dv_dpa * (100.0 / 16.0)                               # (m/s)/mA

        # standard error of the mean in mA → convert to m/s
        sem_ma   = np.mean(d["Stan"]) / np.sqrt(n_seconds)
        sem_v    = dv_dma * sem_ma                  # sensor noise contribution

        # temporal instability: std of per-second velocity estimates
        drift_v  = np.std(v_per_second, ddof=1)

        # combine in quadrature (GUM standard)
        total_unc = np.sqrt(drift_v**2 + sem_v**2)

        mean_kurt = np.mean(d["Kurt"])
        mean_skew = np.mean(d["Skew"])

        # --- turbulence intensity ---
        std_v_inst_per_s = dv_dma * d["Stan"]
        mean_std_v_inst  = np.mean(std_v_inst_per_s)
        TI = mean_std_v_inst / mean_speed if mean_speed > 0 else np.nan

        # --- percentiles of 1 s means ---
        v_min, v_max          = np.min(v_per_second), np.max(v_per_second)
        v_p05, v_p50, v_p95   = np.percentile(v_per_second, [5, 50, 95])

        results.append({
            "angle":      angle,
            "fname":      fname,
            "prelim_run": prelim,
            "is_run2":    is_run2,
            "mean_speed": mean_speed,
            "drift_std":  drift_v,
            "noise_mean": sem_v,
            "total_unc":  total_unc,
            "mean_kurt":  mean_kurt,
            "mean_skew":  mean_skew,
            "n_seconds":  n_seconds,
            "TI":         TI,
            "v_min":      v_min,
            "v_max":      v_max,
            "v_p05":      v_p05,
            "v_p50":      v_p50,
            "v_p95":      v_p95,
        })

    results.sort(key=lambda x: x["angle"])
    return results


def summarize_results(results):
    """Print 3–5 representative rows for pasting into another LLM."""
    if not results:
        print("No results to summarize.")
        return

    # Pick representative rows: lowest angle, highest angle, middle angle,
    # first run2 found, first prelim found (deduplicated by index)
    candidates = {}
    candidates["lowest_angle"]  = results[0]
    candidates["highest_angle"] = results[-1]
    candidates["middle"]        = results[len(results) // 2]
    for r in results:
        if r["is_run2"] and "run2" not in candidates:
            candidates["run2"] = r
        if r["prelim_run"] and "prelim" not in candidates:
            candidates["prelim"] = r

    seen = set()
    selected = []
    for r in candidates.values():
        key = r["fname"]
        if key not in seen:
            seen.add(key)
            selected.append(r)

    run_label = lambda r: "run2" if r["is_run2"] else ("prelim" if r["prelim_run"] else "run1")

    print("\n--- summarize_results (paste into LLM) ---")
    for r in selected:
        print(
            f"angle={r['angle']}, run={run_label(r)}, "
            f"mean_speed={r['mean_speed']:.2f}, TI={r['TI']:.3f}, "
            f"drift_std={r['drift_std']:.3f}, "
            f"v_p05={r['v_p05']:.2f}, v_p50={r['v_p50']:.2f}, v_p95={r['v_p95']:.2f}, "
            f"v_min={r['v_min']:.2f}, v_max={r['v_max']:.2f}, "
            f"total_unc={r['total_unc']:.3f}"
        )
    print("---")


# --- PLOT ---
results = process_stats_folder(stats_folder)
summarize_results(results)

# --- PLOT SETTINGS ---
SPLIT_PLOTS = True   # True = 3 separate figures for LaTeX, False = combined
SAVE_PLOTS = True

def make_plots(results, split=False, save=False):
    colors = [
    "tab:orange" if r["is_run2"]
    else "tab:green" if r["prelim_run"]
    else "tab:blue"
    for r in results
    ]

    # angles = [r["angle"] for r in results]

    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines
    p0 = mpatches.Patch(color="tab:green",   label="Kjøring 0")
    p1 = mpatches.Patch(color="tab:blue",   label="Kjøring 1")
    p2 = mpatches.Patch(color="tab:orange", label="Kjøring 2")
    run_legend = [p0,p1, p2]

    def style_ax(ax):
        ax.grid(True, linestyle='--', linewidth=0.5)
        ax.set_xlabel("Vinkel [grader]")
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    if split:
        figs = []
        fig1, ax1 = plt.subplots(figsize=(6.27, 3.0))
        fig2, ax2 = plt.subplots(figsize=(6.27, 3.9))
        fig3, ax3 = plt.subplots(figsize=(6.27, 3.9))
        figs = [(fig1, ax1), (fig2, ax2), (fig3, ax3)]
    else:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
        fig.suptitle("Vindfart per grad (LabView)", fontsize=13)

    # --- Panel 1: pooled angle response, normalised, cos-theta reference ---
    by_angle = defaultdict(list)
    for r in results:
        by_angle[r["angle"]].append((r["mean_speed"], r["total_unc"]))
    angs    = sorted(by_angle.keys())
    U_0     = float(np.mean([v for v, _ in by_angle[min(angs)]]))
    means   = np.array([np.mean([v for v, _ in by_angle[a]]) for a in angs]) / U_0
    spread  = np.array([
        np.std([v for v, _ in by_angle[a]], ddof=1) if len(by_angle[a]) > 1
        else by_angle[a][0][1]
        for a in angs
    ]) / U_0
    ang_max = int(max(angs))

    # operational range used in main experiment (edit if your alignment differs)
    ax1.axvspan(0, 2, alpha=0.15, color='tab:green',
                label="Brukt i hovedforsøk")

    # cos(theta) geometric reference
    th = np.linspace(0, ang_max, 200)
    ax1.plot(th, np.cos(np.deg2rad(th)),
             linestyle='--', color='dimgray', linewidth=1.0,
             label=r"$\cos\theta$ (geometrisk forventning)")

    # measured response (runs pooled at each angle)
    ax1.errorbar(angs, means, yerr=spread,
                 fmt='o-', color='tab:blue', markersize=3,
                 linewidth=0.8, capsize=2, elinewidth=0.6, capthick=0.6,
                 label="Måling")

    ax1.axhline(1.0, color='dimgray', linewidth=0.5, alpha=0.4)
    ax1.set_xlabel("Vinkel mellom probe og strømning [grader]")
    ax1.set_ylabel(r"$U(\theta)\,/\,U_0$")
    ax1.set_xlim(-0.5, ang_max + 0.5)
    ax1.set_ylim(0.7, 1.1)
    ax1.set_xticks(range(0, ang_max + 1, 5))
    ax1.set_xticks(range(0, ang_max + 1), minor=True)
    ax1.grid(True, which='major', linestyle='--', linewidth=0.5)
    ax1.grid(True, which='minor', linestyle='--', linewidth=0.3, alpha=0.4)
    ax1.legend(fontsize=10, loc='lower left')

    # --- Panel 2: uncertainty components ---
    tri = mlines.Line2D([], [], color='gray', marker='^', linestyle='None',
                        label='Temporal variabilitet (standardavvik av sekundverdier)')
    sq  = mlines.Line2D([], [], color='gray', marker='s', linestyle='None', alpha=0.6,
                        label='Sensorusikkerhet (propagert standardavvik)')
    tot = mlines.Line2D([], [], color='gray', marker='o', linestyle='None', alpha=0.4,
                        label='Samlet usikkerhet (√(drift² + støy²))')
    for r, c in zip(results, colors):
        ax2.scatter(r["angle"], r["drift_std"],  marker='^', color=c, zorder=3)
        ax2.scatter(r["angle"], r["noise_mean"], marker='s', color=c, alpha=0.6, zorder=3)
        ax2.scatter(r["angle"], r["total_unc"],  marker='o', color=c, alpha=0.4, zorder=2)
    ax2.set_ylabel("Vindvariasjon [m/s]")
    ax2.set_title("Målingsusikkerhet")
    ax2.legend(handles=[tri, sq, tot], fontsize=8)
    style_ax(ax2)

    # --- Panel 3: skewness + kurtosis (LabVIEW 2016 returns excess kurtosis; normal = 0) ---
    ax3b = ax3.twinx()

    for r, c in zip(results, colors):
        ax3.scatter(r["angle"], r["mean_skew"], marker='D', color=c, s=30)
        ax3b.scatter(r["angle"], r["mean_kurt"], marker='*', color=c, s=40, alpha=0.7)

    ax3.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax3b.axhline(0, color='gray', linewidth=0.8, linestyle=':', label='Normalfordeling (eksess-kurtose=0)')

    ax3.set_ylabel("Gjennomsnittlig skjevhet [-]", color='black')
    ax3b.set_ylabel("Gjennomsnittlig kurtose [-]", color='gray')
    ax3b.tick_params(axis='y', labelcolor='gray')
    ax3.set_title("Strømningssymmetri (skjevhet) og haletykkelse (eksess-kurtose)")

    skew_handle = mlines.Line2D(
        [], [], color='black', marker='D', linestyle='None', markersize=6,
        label='Skjevhet'
    )
    kurt_handle = mlines.Line2D(
        [], [], color='gray', marker='*', linestyle='None', markersize=8,
        label='Kurtose'
    )
    line_handles, _ = ax3b.get_legend_handles_labels()
    ax3.legend(
        handles=run_legend + [skew_handle, kurt_handle] + line_handles,
        fontsize=9, loc='lower right'
    )
    if ax3b.get_legend():
        ax3b.get_legend().remove()
    style_ax(ax3)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = os.path.expanduser("~/Kodevik/wave_project/windresults")
    os.makedirs(fig_path, exist_ok=True)
    if split and save:
        for (fig, _), name in zip(figs, ["windspeed_angles", "uncertainty_angles", "skewness_angles"]):
            print(f"saving to {fig_path}/{name}_{ts}.pdf")
            fig.savefig(f"{fig_path}/{name}_{ts}.pdf", bbox_inches='tight')
    elif save and not split:
        name = "wind_angles_stats"
        print(f"saving to {fig_path}/{name}_{ts}.pdf")
        fig.savefig(f"{fig_path}/{name}_{ts}.pdf", bbox_inches='tight')


    if split:
        for fig, _ in figs:
            fig.tight_layout()
        plt.show()
    else:
        fig.tight_layout()
        plt.show()

def print_summary(results):
    run_label = {(False, False): "run1", (False, True): "run2", (True, False): "prelim"}
    header = f"{'Angle':>6}  {'Run':<6}  {'Speed':>7}  {'p50':>7}  {'±Unc':>6}  {'Drift':>6}  {'TI':>6}  {'Skew':>6}  {'Kurt(ex)':>8}  {'n [s]':>6}"
    print(header)
    print("-" * len(header))
    for r in results:
        label = run_label.get((r["prelim_run"], r["is_run2"]), "?")
        print(
            f"{r['angle']:>6}  {label:<6}  "
            f"{r['mean_speed']:>7.3f}  "
            f"{r['v_p50']:>7.3f}  "
            f"{r['total_unc']:>6.3f}  "
            f"{r['drift_std']:>6.3f}  "
            f"{r['TI']:>6.3f}  "
            f"{r['mean_skew']:>6.3f}  "
            f"{r['mean_kurt']:>8.3f}  "
            f"{r['n_seconds']:>6}"
        )
    speeds = [r["mean_speed"] for r in results]
    print(f"\nSpeed range: {min(speeds):.2f} – {max(speeds):.2f} m/s  |  mean: {np.mean(speeds):.2f} m/s")

print_summary(results)
make_plots(results, split=SPLIT_PLOTS, save=SAVE_PLOTS)
# fig_path = "windresults/wind_angles_stats.pdf"   # or .png, .svg, .pgf, ...
# plt.savefig(fig_path)
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# %%
#TODO - bruke på ekte fart, per høyde? denne gir et avansert plot
# ... de
"""Important caveat for your thesis: label this as "approximated Gaussian distribution"
— you're reconstructing it from summary statistics, not raw samples.
The shape is assumed Gaussian, which your near-zero skewness actually supports as a reasonable assumption.
The width of each curve represents drift_std
— the temporal wind speed variation during the run, which is your dominant uncertainty source."""
def plot_distributions(results):
    from scipy.stats import norm
    vinkellisten = []
    fig, ax = plt.subplots(figsize=(10, 5))

    for r in results:
        if not r["angle"] < 8:
            continue
        vinkellisten.append(r["angle"])
        color = (
            "tab:orange" if r["is_run2"]
            else "tab:green" if r["prelim_run"]
            else "tab:blue"
        )
        mu    = r["mean_speed"]
        sigma = r["drift_std"]   # temporal spread in m/s — the dominant uncertainty

        x = np.linspace(mu - 4*sigma, mu + 4*sigma, 300)
        y = norm.pdf(x, mu, sigma)

        ax.plot(x, y, color=color, alpha=0.6, linewidth=1.2)
        ax.axvline(mu, color=color, linewidth=0.5, alpha=0.3)

    vinkelmax = max(vinkellisten)
    import matplotlib.patches as mpatches
    p0 = mpatches.Patch(color="tab:green",  label="Kjøring 0")
    p1 = mpatches.Patch(color="tab:blue",   label="Kjøring 1")
    p2 = mpatches.Patch(color="tab:orange", label="Kjøring 2")
    ax.legend(handles=[p0, p1, p2], fontsize=9)

    ax.set_xlabel("Vindfart [m/s]")
    ax.set_ylabel("Sannsynlighetstetthet [-]")
    ax.set_title(f"Approksimert fordeling av vindfart per kjøring, vinkler opptil {vinkelmax} grader")
    ax.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    figname = "winddistribution_reconstructed"
    fig_path = "windresults"
    print(f"saving to {fig_path}/{figname}")
    fig.savefig(f"{fig_path}/{figname}.pdf", bbox_inches='tight')
    plt.show()

plot_distributions(results)


# %%
# --- Visual: pitot at each tested angle, vertical stack ---
# Reader-friendly geometric view of the cosine-law angle response.
# Each row: probe rotated to the tested angle, horizontal wind arrow,
# green arrow showing the wind component along the probe axis,
# and the measured U(theta)/U_0 ratio.

def plot_angle_visual(results):
    from matplotlib.patches import Rectangle
    import matplotlib.transforms as mtr

    by_angle = defaultdict(list)
    for r in results:
        by_angle[r["angle"]].append(r["mean_speed"])
    angs   = sorted(by_angle.keys())
    U_0    = float(np.mean(by_angle[min(angs)]))
    ratios = {a: float(np.mean(by_angle[a])) / U_0 for a in angs}

    PROBE_LEN = 1.4
    PROBE_W   = 0.10
    WIND_LEN  = 1.2
    PIVOT_X   = 0.0
    SPACING   = 1.5
    n_rows    = len(angs)

    fig, ax = plt.subplots(figsize=(4.8, 1.0 * n_rows + 1.2))

    for i, ang in enumerate(angs):
        y     = -i * SPACING
        theta = np.deg2rad(ang)
        ratio = ratios[ang]

        # incoming wind (horizontal, fixed length)
        ax.annotate(
            "", xy=(PIVOT_X - 0.05, y),
            xytext=(PIVOT_X - 0.05 - WIND_LEN, y),
            arrowprops=dict(arrowstyle="-|>", color='steelblue', lw=1.3),
            zorder=3,
        )

        # probe rectangle, rotated about its tip at (PIVOT_X, y)
        rect = Rectangle(
            (PIVOT_X, y - PROBE_W / 2),
            PROBE_LEN, PROBE_W,
            facecolor='lightgray', edgecolor='black', linewidth=0.8,
            zorder=4,
        )
        rect.set_transform(
            mtr.Affine2D().rotate_deg_around(PIVOT_X, y, ang) + ax.transData
        )
        ax.add_patch(rect)

        # captured component along probe axis (length = WIND_LEN * cos(theta))
        proj_len = WIND_LEN * np.cos(theta)
        proj_end = (PIVOT_X + proj_len * np.cos(theta),
                    y       + proj_len * np.sin(theta))
        ax.annotate(
            "", xy=proj_end, xytext=(PIVOT_X, y),
            arrowprops=dict(arrowstyle="-|>", color='tab:green', lw=1.5),
            zorder=5,
        )

        # right-side label
        ax.text(PIVOT_X + PROBE_LEN + 0.4, y,
                f"{ang}°,  $U(\\theta)/U_0 = {ratio:.3f}$",
                va='center', ha='left', fontsize=10)

    # legend at the top
    legend_elements = [
        mlines.Line2D([0], [0], color='steelblue', lw=1.3,
                      label="Vind (horisontal)"),
        mlines.Line2D([0], [0], color='tab:green', lw=1.5,
                      label=r"Fanget komponent ($\propto \cos\theta$)"),
        mlines.Line2D([0], [0], color='black', lw=0,
                      marker='s', markerfacecolor='lightgray', markersize=10,
                      label="Pitotrør"),
    ]
    ax.legend(handles=legend_elements, loc='upper left',
              fontsize=9, frameon=True, bbox_to_anchor=(0.0, 1.0))

    ax.set_xlim(PIVOT_X - WIND_LEN - 0.4, PIVOT_X + PROBE_LEN + 3.0)
    ax.set_ylim(-(n_rows - 1) * SPACING - 0.8, 1.4)
    ax.set_aspect('equal')
    ax.axis('off')

    fig.tight_layout()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = os.path.expanduser("~/Kodevik/wave_project/windresults")
    out = os.path.join(fig_path, f"angle_visual_{ts}.pdf")
    fig.savefig(out, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.show()


plot_angle_visual(results)


# %%
# --- Visual: loss in measured wind speed vs angular misalignment ---
# Plots 100 * (1 - U(θ)/U_0) — the percentage of wind speed lost to
# pitot-axis misalignment θ. The reader reads the methodology answer
# directly: "if our alignment uncertainty is ±X°, our error is ≤ Y%".
#
# The cos-curve becomes a prediction we're testing; the threshold line
# is a numeric commitment we're meeting.

def plot_angle_loss(results, theta_max_deg=30,
                    alignment_unc_deg=2.0, threshold_pct=1.0):
    by_angle = defaultdict(list)
    for r in results:
        by_angle[r["angle"]].append(r["mean_speed"])
    angs   = sorted(by_angle.keys())
    U_0    = float(np.mean(by_angle[min(angs)]))
    losses = {a: 100.0 * (1 - float(np.mean(by_angle[a])) / U_0) for a in angs}

    angs_use   = [a for a in angs if a <= theta_max_deg]
    losses_use = [losses[a] for a in angs_use]

    fig, ax = plt.subplots(figsize=(6.27, 3.5))

    # cosine prediction: 100 * (1 - cos θ)
    th = np.linspace(0, theta_max_deg, 200)
    cos_loss = 100.0 * (1 - np.cos(np.deg2rad(th)))
    ax.plot(th, cos_loss,
            linestyle='-', color='dimgray', linewidth=1.0,
            label=r"$100\,(1-\cos\theta)$ (forventet)")

    # alignment-uncertainty band — what we claim our alignment was within
    ax.axvspan(0, alignment_unc_deg, color='tab:green', alpha=0.15,
               label=f"Justeringsusikkerhet (±{alignment_unc_deg:g}°)")

    # threshold line — the loss the reader agrees is acceptable
    ax.axhline(threshold_pct, color='tab:red', linestyle='--',
               linewidth=0.8, alpha=0.8,
               label=f"{threshold_pct:g}% terskel")

    # angle at which cos crosses the threshold
    theta_cross = float(np.degrees(np.arccos(1 - threshold_pct / 100.0)))
    if theta_cross < theta_max_deg:
        ax.axvline(theta_cross, color='tab:red', linestyle=':',
                   linewidth=0.6, alpha=0.7)
        ax.text(theta_cross + 0.4, threshold_pct + 0.4,
                f"{threshold_pct:g}% nås ved {theta_cross:.1f}°",
                fontsize=8, color='tab:red',
                va='bottom', ha='left')

    # measurements
    ax.scatter(angs_use, losses_use, s=26, color='tab:blue', zorder=5,
               label="Måling")

    # report the worst-case loss within the alignment uncertainty band
    worst_loss = 100.0 * (1 - np.cos(np.deg2rad(alignment_unc_deg)))
    ax.text(0.02, 0.95,
            (f"Innenfor ±{alignment_unc_deg:g}°: "
             f"tap $\\leq$ {worst_loss:.2f}%"),
            transform=ax.transAxes, fontsize=9,
            va='top', ha='left',
            bbox=dict(facecolor='white', edgecolor='tab:green',
                      boxstyle='round,pad=0.3', alpha=0.9))

    ax.set_xlabel(r"Avvik fra strømlinje, $\theta$ [grader]")
    ax.set_ylabel(r"Tap i målt vindfart $\;100\,(1-U/U_0)$ [%]")
    ax.set_xlim(0, theta_max_deg)
    ymax = max(losses_use + [cos_loss.max()]) * 1.12
    ax.set_ylim(0, ymax)
    ax.set_xticks(range(0, theta_max_deg + 1, 5))
    ax.set_xticks(range(0, theta_max_deg + 1), minor=True)
    ax.grid(True, which='major', linestyle='--', linewidth=0.5)
    ax.grid(True, which='minor', linestyle='--', linewidth=0.3, alpha=0.4)
    ax.legend(fontsize=9, loc='upper left',
              bbox_to_anchor=(0.02, 0.85))

    fig.tight_layout()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = os.path.expanduser("~/Kodevik/wave_project/windresults")
    out = os.path.join(fig_path, f"angle_loss_{ts}.pdf")
    fig.savefig(out, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.show()


plot_angle_loss(results, theta_max_deg=30,
                alignment_unc_deg=2.0, threshold_pct=1.0)


# %%
# --- Variant: loss plot with errorbars + ±1σ noise band ---
# Same x/y axes as plot_angle_loss, but each dot now carries its own
# yerr (propagated from total_unc on numerator + denominator), and the
# cosine prediction is wrapped in a shaded ±1σ band representing the
# typical measurement-floor on the ratio. Dots whose errorbars overlap
# the band are consistent with cos law, regardless of where the central
# value lands. The white box now reports both the predicted loss and
# the measurement floor so the reader can compare them.

def plot_angle_loss_with_uncertainty(results, theta_max_deg=30,
                                     alignment_unc_deg=2.0,
                                     threshold_pct=1.0):
    by_angle_speed = defaultdict(list)
    by_angle_unc   = defaultdict(list)
    for r in results:
        by_angle_speed[r["angle"]].append(r["mean_speed"])
        by_angle_unc[r["angle"]].append(r["total_unc"])

    angs    = sorted(by_angle_speed.keys())
    a_min   = min(angs)
    U_0     = float(np.mean(by_angle_speed[a_min]))
    # uncertainty on the denominator: pool runs at the smallest angle.
    # Use run-to-run std if multiple runs, else mean of total_unc values.
    if len(by_angle_speed[a_min]) > 1:
        sigma_U_0 = float(np.std(by_angle_speed[a_min], ddof=1))
    else:
        sigma_U_0 = float(np.mean(by_angle_unc[a_min]))

    # per-angle measured speed, total_unc, and propagated ratio uncertainty
    angs_use, losses_use, loss_err_use = [], [], []
    for a in angs:
        if a > theta_max_deg:
            continue
        U_t   = float(np.mean(by_angle_speed[a]))
        if len(by_angle_speed[a]) > 1:
            sigma_U_t = float(np.std(by_angle_speed[a], ddof=1))
        else:
            sigma_U_t = float(np.mean(by_angle_unc[a]))
        ratio = U_t / U_0
        # gaussian error propagation:
        # sigma_ratio = ratio * sqrt((sigma_U_t/U_t)^2 + (sigma_U_0/U_0)^2)
        sigma_ratio = ratio * np.sqrt((sigma_U_t / U_t) ** 2
                                      + (sigma_U_0 / U_0) ** 2)
        angs_use.append(a)
        losses_use.append(100.0 * (1 - ratio))
        loss_err_use.append(100.0 * sigma_ratio)

    # representative noise band: median errorbar across measurements
    sigma_band_pp = float(np.median(loss_err_use)) if loss_err_use else 0.0

    fig, ax = plt.subplots(figsize=(6.27, 3.5))

    th = np.linspace(0, theta_max_deg, 200)
    cos_loss = 100.0 * (1 - np.cos(np.deg2rad(th)))

    # ±1σ noise band around the cosine prediction
    ax.fill_between(th,
                    cos_loss - sigma_band_pp,
                    cos_loss + sigma_band_pp,
                    color='dimgray', alpha=0.15,
                    label=f"±1σ målestøy ($\\approx${sigma_band_pp:.1f} %-poeng)")

    # cosine prediction line
    ax.plot(th, cos_loss,
            linestyle='-', color='dimgray', linewidth=1.0,
            label=r"$100\,(1-\cos\theta)$ (forventet)")

    # alignment-uncertainty band
    ax.axvspan(0, alignment_unc_deg, color='tab:green', alpha=0.15,
               label=f"Justeringsusikkerhet (±{alignment_unc_deg:g}°)")

    # threshold line
    ax.axhline(threshold_pct, color='tab:red', linestyle='--',
               linewidth=0.8, alpha=0.8,
               label=f"{threshold_pct:g}% terskel")

    theta_cross = float(np.degrees(np.arccos(1 - threshold_pct / 100.0)))
    if theta_cross < theta_max_deg:
        ax.axvline(theta_cross, color='tab:red', linestyle=':',
                   linewidth=0.6, alpha=0.7)

    # measurements with errorbars
    ax.errorbar(angs_use, losses_use, yerr=loss_err_use,
                fmt='o', color='tab:blue', markersize=4,
                capsize=2, elinewidth=0.8, capthick=0.8,
                zorder=5, label="Måling")

    # info box: predicted loss within alignment band vs measurement floor
    pred_loss_band = 100.0 * (1 - np.cos(np.deg2rad(alignment_unc_deg)))
    info_lines = [
        f"Innenfor ±{alignment_unc_deg:g}°:  forventet tap $\\leq$ {pred_loss_band:.2f}%",
        f"Måleoppløsning på forholdet:  $\\sim${sigma_band_pp:.1f} %-poeng",
    ]
    ax.text(0.98, 0.95,
            "\n".join(info_lines),
            transform=ax.transAxes, fontsize=8,
            va='top', ha='right',
            bbox=dict(facecolor='white', edgecolor='gray',
                      boxstyle='round,pad=0.3', alpha=0.9))

    ax.set_xlabel(r"Avvik fra strømlinje, $\theta$ [grader]")
    ax.set_ylabel(r"Tap i målt vindfart $\;100\,(1-U/U_0)$ [%]")
    ax.set_xlim(0, theta_max_deg)
    ymax = max([l + e for l, e in zip(losses_use, loss_err_use)]
               + [cos_loss.max() + sigma_band_pp]) * 1.10
    ymin = min([l - e for l, e in zip(losses_use, loss_err_use)]
               + [-sigma_band_pp]) * 1.10
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(range(0, theta_max_deg + 1, 5))
    ax.set_xticks(range(0, theta_max_deg + 1), minor=True)
    ax.grid(True, which='major', linestyle='--', linewidth=0.5)
    ax.grid(True, which='minor', linestyle='--', linewidth=0.3, alpha=0.4)
    ax.legend(fontsize=8, loc='upper left',
              bbox_to_anchor=(0.02, 0.78))

    fig.tight_layout()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = os.path.expanduser("~/Kodevik/wave_project/windresults")
    out = os.path.join(fig_path, f"angle_loss_with_uncertainty_{ts}.pdf")
    fig.savefig(out, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.show()


plot_angle_loss_with_uncertainty(results, theta_max_deg=30,
                                 alignment_unc_deg=2.0, threshold_pct=1.0)
