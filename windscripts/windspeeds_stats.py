# %%
import os
import re
import numpy as np

AIR_DENSITY = 1.225  # kg/m^3, adjust if you used another value

height_pattern = re.compile(r"moh(\d+)", re.IGNORECASE)

def parse_stats_file(file_path):
    """
    Parse a LabVIEW mAstats file into a dict of numpy arrays.
    Assumes a header with quoted column names and comma as decimal separator.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    # First line: header
    header = lines[0]
    # e.g. ""Arit"  ""RMS""  ""Stan" ..."
    cols = [c.strip().replace('"', '') for c in header.split('\t')]
    # Some may be empty because of double quotes; keep only non-empty
    cols = [c for c in cols if c]

    data = {c: [] for c in cols}

    # Remaining lines: numeric rows with comma decimal separator
    for line in lines[1:]:
        parts = [p.strip().replace('"', '').replace(',', '.') for p in line.split('\t')]
        # Some trailing empty columns; keep only as many as we have cols
        parts = parts[:len(cols)]
        if len(parts) != len(cols):
            continue
        try:
            values = [float(p) for p in parts]
        except ValueError:
            continue
        for c, v in zip(cols, values):
            data[c].append(v)

    # Convert lists to numpy arrays
    for c in data:
        data[c] = np.array(data[c], dtype=float)

    return data
def process_stats_folder_by_height(folder_path, fname_filter=None):
    """
    Process all *-mAstats-mohXXX.txt files in a STATS folder.
    fname_filter: optional substring — only process files containing this string.
    Returns a list of dicts, one per height, with:
      height_mm, mean_speed, drift_std, TI, total_unc, etc.
    """
    results = []
    for fname in os.listdir(folder_path):
        if not fname.lower().endswith(".txt"):
            continue

        # Only look at mAstats files with a height tag
        if "mAstats" not in fname:
            continue

        if fname_filter is not None and fname_filter not in fname:
            continue

        hmatch = height_pattern.search(fname)
        if not hmatch:
            continue

        height_mm = int(hmatch.group(1))
        file_path = os.path.join(folder_path, fname)
        d = parse_stats_file(file_path)

        if "Arit" not in d or "Stan" not in d:
            continue

        # --- drop dead/zero rows and sub-zero-wind rows ---
        # Arit < 4.0 mA means pressure is below sensor zero → no meaningful wind.
        # Arit > 0.1 filters disconnected/dead channels.
        n_raw = d["Arit"].size
        mask = (d["Arit"] > 0.1) & (d["Arit"] >= 4.0)
        for k in d.keys():
            d[k] = d[k][mask]
        if d["Arit"].size == 0:
            print(f"  [{fname}] skipped — all rows below sensor zero (no wind at this height)")
            continue
        n_subzero = n_raw - d["Arit"].size
        if n_subzero > n_raw * 0.3:   # >30% of seconds lost → unreliable
            print(f"  [{fname}] SKIPPED: {n_subzero}/{n_raw} s below sensor zero — likely hose/connector fault mid-run")
            continue

        # --- drop spike rows: wave splash / droplet on pitot ---
        # A single contaminated second produces Kurt~455, Skew~-17.
        # Normal wind: |Skew| < 2, Kurt < 20. Threshold is conservative.
        n_dropped = 0
        if "Kurt" in d and "Skew" in d:
            spike_mask = (np.abs(d["Skew"]) < 5) & (d["Kurt"] < 50)
            n_dropped = int(np.sum(~spike_mask))
            if n_dropped > 0:
                print(f"  [{fname}] dropped {n_dropped} spike row(s) (Kurt/Skew outlier)")
            for k in d.keys():
                d[k] = d[k][spike_mask]
        if d["Arit"].size == 0:
            continue

        n_seconds = d["Arit"].size

        # --- helpers ---
        def ma_to_pa(ma):
            return np.maximum((ma - 4.0) / 16.0 * 100.0, 0.0)

        def pa_to_v(pa):
            return np.sqrt(2.0 * np.maximum(pa, 0.001) / AIR_DENSITY)

        # per-second velocity (1 s means)
        v_per_second = pa_to_v(ma_to_pa(d["Arit"]))
        mean_speed   = np.mean(v_per_second)

        # sensitivity at mean operating point (as before)
        ma_mean = np.mean(d["Arit"])
        pa_mean = float(ma_to_pa(ma_mean))
        pa_mean = max(pa_mean, 0.001)

        dv_dpa = 1.0 / np.sqrt(2.0 * pa_mean * AIR_DENSITY)     # (m/s)/Pa
        dv_dma = dv_dpa * (100.0 / 16.0)                        # (m/s)/mA

        # sensor noise contribution to mean (SEM in mA → m/s)
        sem_ma = np.mean(d["Stan"]) / np.sqrt(n_seconds)
        sem_v  = dv_dma * sem_ma

        # temporal variability of 1 s means (drift / unsteadiness)
        drift_v = np.std(v_per_second, ddof=1)

        # turbulence intensity from instantaneous std (per second)
        std_v_inst_per_s = dv_dma * d["Stan"]
        mean_std_v_inst  = np.mean(std_v_inst_per_s)
        TI = mean_std_v_inst / mean_speed if mean_speed >= 0.5 else np.nan  # NaN if near-zero wind

        # combined uncertainty of mean
        total_unc = np.sqrt(drift_v**2 + sem_v**2)

        # shape of distribution of 1 s means
        v_p05, v_p50, v_p95 = np.percentile(v_per_second, [5, 50, 95])
        v_min, v_max        = np.min(v_per_second), np.max(v_per_second)

        mean_kurt = np.mean(d.get("Kurt", np.nan))
        mean_skew = np.mean(d.get("Skew", np.nan))

        results.append({
            "height_mm":  height_mm,
            "fname":      fname,
            "n_dropped":  n_dropped,
            "mean_speed": mean_speed,
            "drift_std":  drift_v,
            "noise_mean": sem_v,
            "total_unc":  total_unc,
            "TI":         TI,
            "v_min":      v_min,
            "v_max":      v_max,
            "v_p05":      v_p05,
            "v_p50":      v_p50,
            "v_p95":      v_p95,
            "mean_kurt":  mean_kurt,
            "mean_skew":  mean_skew,
            "n_seconds":  n_seconds,
        })

    results.sort(key=lambda x: x["height_mm"])
    return results

def summarize_height_stats(results):
    print("--- height_stats (paste into LLM) ---")
    for r in results:
        print(
            f"z={r['height_mm']} mm, "
            f"mean_speed={r['mean_speed']:.3f} m/s, "
            f"TI={r['TI']:.3f}, "
            f"drift_std={r['drift_std']:.3f} m/s, "
            f"v_p05={r['v_p05']:.3f}, v_p50={r['v_p50']:.3f}, v_p95={r['v_p95']:.3f}, "
            f"v_min={r['v_min']:.3f}, v_max={r['v_max']:.3f}, "
            f"total_unc={r['total_unc']:.3f} m/s, "
            f"n={r['n_seconds']}"
        )
    print("--- end ---")

# --- USER SETTINGS ---
# Each entry: (stats_folder_path, legend_label, fname_filter_or_None, linestyle)
# linestyle: '-' = sealed/closed roof, '--' = slisse (open slit)
# Comment out any datasets you want to exclude.
DATASETS = [
    (
        "/Users/ole/Kodevik/wave_project/pressuredata/20251106-fullwindUtenProbe2-fullpanel/20251106-fullwindUtenProbe2-fullpanel-STATS",
        "Full vind, uten bølger (06.11)",
        None, '-',
    ),
    (
        "/Users/ole/Kodevik/wave_project/pressuredata/20251106-fullwindUtenProbe2-fullpanel-amp0100-freq1300/20251106-fullwindUtenProbe2-fullpanel-amp0100-freq1300-STATS",
        "Full vind, med bølger (06.11)",
        None, '-',
    ),
    (
        "/Users/ole/Kodevik/wave_project/pressuredata/20251106-lowestwindUtenProbe2-fullpanel-amp0100-freq1300/20251106-lowestwindUtenProbe2-fullpanel-amp0100-freq1300-STATS",
        "Laveste vind, med bølger (06.11)",
        None, '-',
    ),
    # tunnelTest (07.11) excluded — different experiment (roof config comparison),
    # see windscripts/tunneltest.py
]

SPLIT_PLOTS = True
SAVE_PLOTS  = True

# %%
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from datetime import datetime

def make_height_stats_plots(datasets, split=False, save=False):
    """
    datasets: list of (results, label) tuples.
    Panels (height on y-axis throughout):
      1. Mean wind speed + percentile band + error bars
      2. Uncertainty components (drift, noise, total)
      3. TI (turbulence intensity)
      4. Skewness + kurtosis (twinx)
    """
    colors   = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    markers  = ['o', 's', '^', 'D', '*']

    def style_ax(ax):
        ax.grid(True, linestyle='--', linewidth=0.5)
        ax.set_ylabel("Høyde over vannet [mm]")

    panel_titles = [
        "Vindprofil med usikkerhet",
        "Usikkerhetsbidrag",
        "Turbulensintensitet (TI)",
        "Skjevhet og kurtose",
    ]

    if split:
        figs = [plt.subplots(figsize=(7, 5)) for _ in range(4)]
        ax1, ax2, ax3, ax4 = [f[1] for f in figs]
    else:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 6))
        fig.suptitle("Statistiske egenskaper for vindprofil", fontsize=13)

    legend_handles = []

    for i, entry in enumerate(datasets):
        results, label = entry[0], entry[1]
        ls = entry[2] if len(entry) > 2 else '-'
        c = colors[i % len(colors)]
        m = markers[i % len(markers)]
        z   = np.array([r["height_mm"]  for r in results])

        # --- Panel 1: mean speed + percentile band + errorbars ---
        spd  = np.array([r["mean_speed"] for r in results])
        unc  = np.array([r["total_unc"]  for r in results])
        p05  = np.array([r["v_p05"]      for r in results])
        p95  = np.array([r["v_p95"]      for r in results])
        ax1.errorbar(spd, z, xerr=unc, fmt=m+ls, color=c, capsize=4, markersize=5, label=label)
        ax1.fill_betweenx(z, p05, p95, alpha=0.15, color=c, linestyle=ls)

        # --- Panel 2: uncertainty components ---
        drift = np.array([r["drift_std"]  for r in results])
        noise = np.array([r["noise_mean"] for r in results])
        ax2.plot(drift, z, marker='^', linestyle=ls,  color=c, label=f"{label} – temporal")
        ax2.plot(noise, z, marker='s', linestyle=ls,  color=c, alpha=0.6, label=f"{label} – sensor")
        ax2.plot(unc,   z, marker='o', linestyle=ls,  color=c, alpha=0.4, label=f"{label} – samlet")

        # --- Panel 3: TI ---
        TI = np.array([r["TI"] for r in results]) * 100
        ax3.plot(TI, z, marker=m, linestyle=ls, color=c, label=label)

        # --- Panel 4: skewness + kurtosis (twinx) ---
        skew = np.array([r["mean_skew"] for r in results])
        kurt = np.array([r["mean_kurt"] for r in results])
        ax4b = ax4.twiny() if i == 0 else ax4._aux
        if i == 0:
            ax4._aux = ax4b
        ax4.plot(skew, z,  marker='D', linestyle=ls, color=c, markersize=5)
        ax4b.plot(kurt, z, marker='*', linestyle=ls, color=c, markersize=7, alpha=0.7)

        legend_handles.append(mlines.Line2D([], [], color=c, marker=m, linestyle=ls, label=label))

        # --- mark heights where spike samples were dropped ---
        for r in results:
            if r["n_dropped"] > 0:
                n_tot = r["n_seconds"] + r["n_dropped"]
                for ax in (ax1, ax2, ax3, ax4):
                    ax.axhline(r["height_mm"], color=c, linewidth=0.8,
                               linestyle=':', alpha=0.6)
                ax1.annotate(f"{r['n_dropped']}/{n_tot} s spikes",
                             xy=(spd[list(z).index(r["height_mm"])], r["height_mm"]),
                             xytext=(8, 0), textcoords='offset points',
                             fontsize=7, color=c, va='center')

    # --- Panel 1 styling ---
    ax1.set_xlabel("Vindfart [m/s]")
    ax1.set_title(panel_titles[0])
    ax1.legend(fontsize=8)
    style_ax(ax1)

    # --- Panel 2 styling ---
    ax2.set_xlabel("Vindvariasjon [m/s]")
    ax2.set_title(panel_titles[1])
    tri = mlines.Line2D([], [], color='gray', marker='^', linestyle='-',   label='Temporal variabilitet')
    sq  = mlines.Line2D([], [], color='gray', marker='s', linestyle='--',  label='Sensorusikkerhet', alpha=0.6)
    tot = mlines.Line2D([], [], color='gray', marker='o', linestyle=':',   label='Samlet usikkerhet', alpha=0.4)
    ax2.legend(handles=legend_handles + [tri, sq, tot], fontsize=7)
    style_ax(ax2)

    # --- Panel 3 styling ---
    ax3.set_xlabel("TI [%]")
    ax3.set_title(panel_titles[2])
    ax3.legend(handles=legend_handles, fontsize=8)
    style_ax(ax3)

    # --- Panel 4 styling ---
    ax4.axvline(0, color='gray', linewidth=0.8, linestyle='--')
    ax4b = ax4._aux
    ax4b.axvline(0, color='gray', linewidth=0.8, linestyle=':')
    ax4.set_xlabel("Skjevhet [-]", color='black')
    ax4b.set_xlabel("Eksess-kurtose [-]", color='gray')
    ax4b.tick_params(axis='x', labelcolor='gray')
    ax4.set_title(panel_titles[3])
    skew_h = mlines.Line2D([], [], color='black', marker='D', linestyle='-',  markersize=5, label='Skjevhet')
    kurt_h = mlines.Line2D([], [], color='gray',  marker='*', linestyle='--', markersize=7, label='Eksess-kurtose')
    ax4.legend(handles=legend_handles + [skew_h, kurt_h], fontsize=7)
    style_ax(ax4)

    # --- save ---
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = os.path.expanduser("~/Kodevik/wave_project/windresults")
    os.makedirs(fig_path, exist_ok=True)

    panel_names = ["windspeed_profile", "uncertainty_profile", "TI_profile", "skewness_profile"]

    if split and save:
        for (fig, _), pname in zip(figs, panel_names):
            fig.tight_layout()
            out = os.path.join(fig_path, f"{pname}_{ts}.pdf")
            fig.savefig(out, bbox_inches='tight')
            print(f"Saved: {out}")
        for fig, _ in figs:
            plt.figure(fig.number)
            plt.show()
    else:
        fig.tight_layout()
        if save:
            out = os.path.join(fig_path, f"windprofile_stats_{ts}.pdf")
            fig.savefig(out, bbox_inches='tight')
            print(f"Saved: {out}")
        plt.show()


# %%
all_results = []
for entry in DATASETS:
    stats_folder, label = entry[0], entry[1]
    fname_filter = entry[2] if len(entry) > 2 else None
    linestyle    = entry[3] if len(entry) > 3 else '-'
    res = process_stats_folder_by_height(stats_folder, fname_filter=fname_filter)
    summarize_height_stats(res)
    all_results.append((res, label, linestyle))

make_height_stats_plots(all_results, split=SPLIT_PLOTS, save=SAVE_PLOTS)
 # %%
