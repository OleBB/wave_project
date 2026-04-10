# %%
"""
Tunnel roof configuration comparison — 2025-11-07

Three roof configs tested at the same heights, same wind condition (lowestwind):
  slisse        : 6-roof with open slit   → dashed line
  7roof-pappTett: 7-roof sealed cardboard → solid line
  7roof-slisse  : 7-roof with open slit   → dashed line

Plus a NOWIND baseline (not plotted as a profile — used as zero reference only).

Question: which roof config gives the most uniform, low-TI wind profile?
"""
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from datetime import datetime

# --- reuse processing logic from windspeeds_stats ---
# (copy rather than import to keep scripts self-contained)

AIR_DENSITY     = 1.225  # kg/m³
height_pattern  = re.compile(r"moh(\d+)", re.IGNORECASE)
STATS_FOLDER    = (
    "/Users/ole/Kodevik/wave_project/pressuredata"
    "/20251107-lowestwindUP2-fullpanel-tunnelTest"
    "/20251107-lowestwindUP2-fullpanel-tunnelTest-STATS"
)

SAVE = True

# --- roof configs to compare ---
# (fname_filter, label, linestyle)
CONFIGS = [
    ("slisse",        "6-tak slisse",    "--"),
    ("7roof-pappTett","7-tak tett",       "-"),
    ("7roof-slisse",  "7-tak slisse",    "--"),
]


def parse_stats_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    cols = [c.strip().replace('"', '') for c in lines[0].split('\t')]
    cols = [c for c in cols if c]
    data = {c: [] for c in cols}
    for line in lines[1:]:
        parts = [p.strip().replace('"', '').replace(',', '.') for p in line.split('\t')]
        parts = parts[:len(cols)]
        if len(parts) != len(cols):
            continue
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            continue
        for c, v in zip(cols, vals):
            data[c].append(v)
    return {c: np.array(v) for c, v in data.items()}


def process_config(folder, fname_filter):
    results = []
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith(".txt"):
            continue
        if "mAstats" not in fname:
            continue
        if fname_filter not in fname:
            continue
        # exclude 7roof files when looking for plain slisse
        if fname_filter == "slisse" and "7roof" in fname:
            continue
        hmatch = height_pattern.search(fname)
        if not hmatch:
            continue
        height_mm = int(hmatch.group(1))
        d = parse_stats_file(os.path.join(folder, fname))
        if "Arit" not in d or "Stan" not in d:
            continue

        mask = (d["Arit"] > 0.1) & (d["Arit"] >= 4.0)
        for k in d:
            d[k] = d[k][mask]
        if d["Arit"].size == 0:
            print(f"  [{fname}] skipped — all rows at/below sensor zero (no wind)")
            continue

        n_dropped = 0
        if "Kurt" in d and "Skew" in d:
            spike = (np.abs(d["Skew"]) < 5) & (d["Kurt"] < 50)
            n_dropped = int(np.sum(~spike))
            if n_dropped:
                print(f"  [{fname}] dropped {n_dropped} spike row(s)")
            for k in d:
                d[k] = d[k][spike]
        if d["Arit"].size == 0:
            continue

        def ma_to_pa(ma):
            return np.maximum((ma - 4.0) / 16.0 * 100.0, 0.0)
        def pa_to_v(pa):
            return np.sqrt(2.0 * np.maximum(pa, 0.001) / AIR_DENSITY)

        v_per_s    = pa_to_v(ma_to_pa(d["Arit"]))
        mean_speed = np.mean(v_per_s)
        ma_mean    = np.mean(d["Arit"])
        pa_mean    = max(float(ma_to_pa(ma_mean)), 0.001)
        dv_dpa     = 1.0 / np.sqrt(2.0 * pa_mean * AIR_DENSITY)
        dv_dma     = dv_dpa * (100.0 / 16.0)
        sem_v      = dv_dma * np.mean(d["Stan"]) / np.sqrt(d["Arit"].size)
        drift_v    = np.std(v_per_s, ddof=1)
        total_unc  = np.sqrt(drift_v**2 + sem_v**2)
        TI         = (dv_dma * np.mean(d["Stan"])) / mean_speed if mean_speed >= 0.5 else np.nan
        v_p05, v_p50, v_p95 = np.percentile(v_per_s, [5, 50, 95])

        results.append({
            "height_mm":  height_mm,
            "mean_speed": mean_speed,
            "total_unc":  total_unc,
            "drift_std":  drift_v,
            "TI":         TI,
            "v_p05":      v_p05,
            "v_p50":      v_p50,
            "v_p95":      v_p95,
            "n_dropped":  n_dropped,
            "n_seconds":  d["Arit"].size,
            "mean_skew":  np.mean(d.get("Skew", np.array([np.nan]))),
            "mean_kurt":  np.mean(d.get("Kurt", np.array([np.nan]))),
        })

    results.sort(key=lambda x: x["height_mm"])
    return results


# %%
colors  = ["tab:blue", "tab:orange", "tab:green"]
markers = ['o', 's', '^']

all_data = []
for (filt, label, ls), c, m in zip(CONFIGS, colors, markers):
    res = process_config(STATS_FOLDER, filt)
    print(f"\n{label} ({len(res)} heights):")
    for r in res:
        ti_str = f"{r['TI']*100:.1f}%" if not np.isnan(r["TI"]) else "n/a"
        print(f"  z={r['height_mm']:3d} mm  v={r['mean_speed']:.2f} m/s  TI={ti_str}")
    all_data.append((res, label, ls, c, m))

# %%
ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
fig_path = os.path.expanduser("~/Kodevik/wave_project/windresults")
os.makedirs(fig_path, exist_ok=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7))
fig.suptitle("Takonfigurasjonstest — vindprofil og turbulensintensitet (07.11)", fontsize=12)

legend_handles = []
for res, label, ls, c, m in all_data:
    if not res:
        continue
    z    = np.array([r["height_mm"]  for r in res])
    spd  = np.array([r["mean_speed"] for r in res])
    unc  = np.array([r["total_unc"]  for r in res])
    p05  = np.array([r["v_p05"]      for r in res])
    p95  = np.array([r["v_p95"]      for r in res])
    TI   = np.array([r["TI"]         for r in res]) * 100

    ax1.errorbar(spd, z, xerr=unc, fmt=m+ls, color=c, capsize=4, markersize=5, label=label)
    ax1.fill_betweenx(z, p05, p95, alpha=0.12, color=c)
    ax2.plot(TI, z, marker=m, linestyle=ls, color=c, label=label)

    # mark spike-dropped heights
    for r in res:
        if r["n_dropped"] > 0:
            n_tot = r["n_seconds"] + r["n_dropped"]
            ax1.axhline(r["height_mm"], color=c, linewidth=0.6, linestyle=':', alpha=0.5)
            ax2.axhline(r["height_mm"], color=c, linewidth=0.6, linestyle=':', alpha=0.5)
            ax1.annotate(f"{r['n_dropped']}/{n_tot} s utelatt",
                         xy=(spd[list(z).index(r["height_mm"])], r["height_mm"]),
                         xytext=(6, 0), textcoords='offset points',
                         fontsize=7, color=c, va='center')

    legend_handles.append(mlines.Line2D([], [], color=c, marker=m, linestyle=ls, label=label))

# linestyle legend entries
solid_h  = mlines.Line2D([], [], color='gray', linestyle='-',  label='Tett tak')
dashed_h = mlines.Line2D([], [], color='gray', linestyle='--', label='Slisse (spalte i tak)')

for ax, xlabel, title in [
    (ax1, "Vindfart [m/s]", "Vindprofil med usikkerhet"),
    (ax2, "TI [%]",         "Turbulensintensitet"),
]:
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Høyde over vannet [mm]")
    ax.set_title(title)
    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.legend(handles=legend_handles + [solid_h, dashed_h], fontsize=8)

fig.tight_layout()
if SAVE:
    out = os.path.join(fig_path, f"tunneltest_{ts}.pdf")
    fig.savefig(out, bbox_inches='tight')
    print(f"\nSaved: {out}")
plt.show()
# %%
