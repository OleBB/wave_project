"""CLAUDE - The key addition is the propagated uncertainty — since v = sqrt(2*ΔP/ρ) is nonlinear, the Stan values in mA need to be scaled by dv/d(mA) evaluated at each second's operating point. This gives you honest error bars in m/s rather than mA. The rest of the plotting code from before stays the same."""
# %%
import re
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from datetime import datetime

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
        fig1, ax1 = plt.subplots(figsize=(7, 4))
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        fig3, ax3 = plt.subplots(figsize=(7, 4))
        figs = [(fig1, ax1), (fig2, ax2), (fig3, ax3)]
    else:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
        fig.suptitle("Vindfart per grad (LabView)", fontsize=13)

    # --- Panel 1: mean wind speed + error bars ---
    for r, c in zip(results, colors):
        ax1.errorbar(r["angle"], r["mean_speed"], yerr=r["total_unc"],
                     fmt='o', color=c, capsize=4, markersize=5)
    ax1.set_ylabel("Gjennomsnittlig vindfart [m/s]")
    ax1.set_title("Estimert vindfart med samlet usikkerhet")
    ax1.legend(handles=run_legend, fontsize=9)
    style_ax(ax1)

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
