# %%
"""
Combined wind profile — all available fullwind runs.

Each dataset is shown as a faint individual line.
A combined profile (mean across runs at each height) is shown bold.
Heights measured in only one run get a different marker to flag lower confidence.
"""
import sys, os
sys.path.insert(0, os.path.expanduser("~/Kodevik/wave_project"))   # only if needed
from wavescripts.plot_utils import apply_thesis_style

apply_thesis_style()              # draft — fast, NCM via OTF, no LaTeX round-trip
# apply_thesis_style(usetex=True)   # final — true LaTeX/Computer Modern, slower

import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from datetime import datetime
from collections import defaultdict
from scipy.stats import linregress

AIR_DENSITY    = 1.225
height_pattern = re.compile(r"moh(\d+)", re.IGNORECASE)

SAVE      = True
SHOW_ROOF = True
ROOF_MM   = 380

ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
fig_path = os.path.expanduser("~/Kodevik/wave_project/windresults")
os.makedirs(fig_path, exist_ok=True)

# --- datasets to combine ---
# (stats_folder, short_label)
# Set INCLUDE_04NOV = True to include the 04.11 run (wave probe in flow path).
INCLUDE_04NOV = True

FULLWIND_DATASETS = [
    *(
        [(
            "/Users/ole/Kodevik/wave_project/pressuredata"
            "/20251104-fullwind/20251104-fullwind-stats",
            "04.11 med probe",
        )]
        if INCLUDE_04NOV else []
    ),
    (
        # Bonus folder alongside the 04.11 stats folder. Contains ONE mAstats file:
        # moh031 — long run (n=180) at actual 30 mm (+1 mm convention for long runs).
        # NOT present in the main 04.11-fullwind-stats folder, so this is new data.
        # 5.357 m/s, 6.813 mA — consistent with 04.11 fullwind speeds.
        "/Users/ole/Kodevik/wave_project/pressuredata"
        "/20251104-fullwind/20251104-readme-bonus",
        "04.11 med probe bonus (moh031=30mm)",
    ),
    (
        # NOTE: this stats folder contains moh040 AND moh041. Per the +1 mm convention
        # (same as the lowestwind bonus: long runs get height label +1 mm so they can be
        # identified), moh041 is a long-duration run at actual 40 mm. Both are loaded
        # by process_folder and appear as separate heights (40 mm and 41 mm) in the plot.
        # File: .../20251105-fullwindUtenProbe2-fullpanel-pitot10075-mAstats-moh041.txt
        #
        # OFFSET: 05.11 fullwind reads ~5-8% faster than 04.11 and 06.11 at all heights.
        # The ratio is flat (1.04-1.08) across the entire profile — identical profile shape,
        # just uniformly scaled up. Hypothesis: the mooring was in its original (higher)
        # position on Nov 4 and Nov 5, blocking some of the wind channel cross-section.
        # A narrower effective cross-section → higher velocity by continuity (flat ratio).
        # On Nov 4, the wave probe was also in the flow path — an extra obstruction that
        # partially cancels the mooring effect, putting 04.11 back in line with 06.11.
        # On Nov 5 (no probe, mooring still up), nothing compensates → faster.
        # On Nov 6, mooring moved down → more open cross-section → baseline speed.
        # Status: plausible hypothesis, not confirmed.
        "/Users/ole/Kodevik/wave_project/pressuredata"
        "/20251105-fullwindUtenProbe2-fullpanel"
        "/20251105-fullwindUtenProbe2-fullpanel-stats",
        "05.11 med panel",
    ),
    (
        "/Users/ole/Kodevik/wave_project/pressuredata"
        "/20251106-fullwindUtenProbe2-fullpanel"
        "/20251106-fullwindUtenProbe2-fullpanel-STATS",
        "06.11 uten bølger",
    ),
    (
        # NOTE: a BONUS subfolder exists alongside this stats folder:
        #   .../20251106-fullwindUP2-fullpanel-pitot10075-amp0100-freq1300-BONUS/
        # README says (verbatim): "uten amp og freq så menes jo selvfølgelig at det er
        #   en referansemåling uten bølge" (no-wave reference runs mixed in here).
        # Files present: moh191 (no-wave), moh230 (no-wave), moh230 (with waves).
        # HOWEVER: all three files show Arit ≈ 12–14 mA → implied speeds 9–10 m/s.
        # Tunnel max is well below 6 m/s — these values are physically impossible.
        # The LabVIEW VI was clearly misconfigured for these bonus runs. DO NOT USE.
        "/Users/ole/Kodevik/wave_project/pressuredata"
        "/20251106-fullwindUtenProbe2-fullpanel-amp0100-freq1300"
        "/20251106-fullwindUtenProbe2-fullpanel-amp0100-freq1300-STATS",
        "06.11 med bølger",
    ),
]

LOWESTWIND_DATASETS = [
    (
        "/Users/ole/Kodevik/wave_project/pressuredata"
        "/20251105-lowestwind"
        "/20251105-lowestwind-stats",
        "05.11 uten panel",
    ),
    (
        "/Users/ole/Kodevik/wave_project/pressuredata"
        "/20251105-lowestwind-fullpanel"
        "/20251105-lowestwind-fullpanel-stats",
        "05.11 med panel",
    ),
    (
        # NOTE: bonus folder has a mislabelled date in the folder name (20251104) but the
        # files inside are dated 20251105 — it is a Nov 5 session, just a typo in folder name.
        #
        # README inside says (verbatim):
        #   "forsøk på å se på en laaang tidsserie. normalt har jeg alle serier på 30 sek,
        #    men de lengre feilmerker jeg med vilje +1 slik at høgden blir 1 mm for mye"
        #
        # Translation: deliberately long run (~305 rows vs normal ~30). Height label is
        # moh016 but ACTUAL height is 15 mm — the +1 mm offset is intentional so long
        # runs can be identified by the odd height number.
        #
        # Single point at h=16 mm (true 15 mm), 3.58 m/s, 305 clean rows — valid.
        # This is the closest-to-surface lowestwind measurement we have.
        "/Users/ole/Kodevik/wave_project/pressuredata"
        "/20251105-lowestwind-fullpanel"
        "/20251104-lowestwind-fullpanel-bonus",  # folder name date is a typo, content is 05.11
        "05.11 med panel bonus (moh016=15mm)",
    ),
    (
        # NOTE: a bonus subfolder exists alongside the stats folder:
        #   .../20251105-lowestwindUtenProbe-fullpanel-pitot10075-bonus/
        # It contains ONE file: moh160-pitot15grader (pitot tube tilted 15° from vertical).
        #   → 30 rows, mean 5.284 mA, 3.62 m/s at 160 mm
        # This is NOT a valid wind profile point (angled sensor reads lower than true speed).
        # Value: angle sensitivity / correction research only. NOT loaded here.
        # File: .../20251105-lowestwindUtenProbe-fullpanel-pitot10075-mAstats-moh160-pitot15grader.txt
        # Also in the root folder: readme-240erLangKjøringOgProbe2blefjerna.rft.rtf —
        # "240 er lang kjøring og probe 2 ble fjerna" (the moh240 run is long, and probe 2 was removed).
        "/Users/ole/Kodevik/wave_project/pressuredata"
        "/20251105-lowestwindUtenProbe2-fullpanel"
        "/20251105-lowestwindUtenProbe-fullpanel-pitot10075-stats",
        "05.11 UP2 med panel",
    ),
    (
        # 06.11 lowestwind angle test folder — contains ONLY 0° measurements (no angle suffix
        # in mAstats filenames). Two files at moh050:
        #   pitot10075-mAstats-moh050.txt      n=30,  3.656 m/s  (standard run)
        #   extratid-pitot10075-mAstats-moh050.txt  n=360, 3.723 m/s  (extra long run;
        #     corresponding speed file is moh051 per the +1 convention)
        # The angled runs (ang0xtratid etc.) exist only as speed files — not loaded here.
        # "hullitanken" variants = hole in tank (compromised tunnel) — not present as mAstats.
        "/Users/ole/Kodevik/wave_project/pressuredata"
        "/20251106-lowestwindUtenProbe2-fullpanel-amp0100-freq1300"
        "/20251106-lowestwindUP2-angletest",
        "06.11 med bølger angletest 0°",
    ),
    (
        # Bonus folder alongside the 06.11 lowestwind-med-bølger stats folder.
        # Contains moh050 (n=120, 3.716 m/s) and a no-wind moh59 (all below zero, useless).
        # README says (verbatim): "i denne første kjøringen på 50moh så har startet jeg
        #   120 pittotmålinger i det jeg hørte padla starte sine 120 perioder (på 1.3hz).
        #   Derfor begynte målingene litt før bølgene var nådd frem og målingene sluttet
        #   etter at bølgene hadde gitt seg. Derfor må dataene croppes hvis man ser på et snitt."
        # Translation: measurements started before waves arrived and ended after they died —
        # includes some no-wave periods at start/end. For wind speed this is fine (fan
        # setting unchanged by waves). moh050 is not in the main stats folder. Include it.
        # The nowind file (moh59, all below zero) is automatically skipped by process_folder.
        "/Users/ole/Kodevik/wave_project/pressuredata"
        "/20251106-lowestwindUtenProbe2-fullpanel-amp0100-freq1300"
        "/20251106-lowestwindUtenProbe2-fullpanel-amp0100-freq1300-BONUS",
        "06.11 med bølger bonus (moh050)",
    ),
    (
        "/Users/ole/Kodevik/wave_project/pressuredata"
        "/20251106-lowestwindUtenProbe2-fullpanel-amp0100-freq1300"
        "/20251106-lowestwindUtenProbe2-fullpanel-amp0100-freq1300-STATS",
        "06.11 med bølger",
    ),
]


# ---- shared processing (same logic as windspeeds_stats.py) ----

def parse_stats_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
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


def process_folder(folder_path, fname_filter=None, exclude_filter=None):
    results = []
    for fname in sorted(os.listdir(folder_path)):
        if not fname.lower().endswith(".txt"):
            continue
        if "mAstats" not in fname:
            continue
        if fname_filter and fname_filter not in fname:
            continue
        if exclude_filter and exclude_filter in fname:
            continue
        hmatch = height_pattern.search(fname)
        if not hmatch:
            continue
        height_mm = int(hmatch.group(1))
        d = parse_stats_file(os.path.join(folder_path, fname))
        if "Arit" not in d or "Stan" not in d:
            continue

        n_raw = d["Arit"].size
        mask  = (d["Arit"] > 0.1) & (d["Arit"] >= 4.0)
        for k in d:
            d[k] = d[k][mask]
        if d["Arit"].size == 0:
            continue
        if (n_raw - d["Arit"].size) > n_raw * 0.3:
            print(f"  [{fname}] SKIPPED: majority below sensor zero (hose/connector fault)")
            continue

        if "Kurt" in d and "Skew" in d:
            spike = (np.abs(d["Skew"]) < 5) & (d["Kurt"] < 50)
            if np.sum(~spike):
                print(f"  [{fname}] dropped {np.sum(~spike)} spike row(s)")
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
        drift_std  = float(np.std(v_per_s, ddof=1)) if v_per_s.size > 1 else 0.0
        pa_mean    = max(float(np.mean(ma_to_pa(d["Arit"]))), 0.001)
        dv_dma     = (1 / np.sqrt(2 * pa_mean * AIR_DENSITY)) * (100 / 16)
        sem_v      = dv_dma * np.mean(d["Stan"]) / np.sqrt(d["Arit"].size)
        total_unc  = np.sqrt(drift_std**2 + sem_v**2)

        # Within-block fluctuation stats (averaged across blocks):
        # each block ≈ 1 s at 100 Hz → block skew/kurt characterise the PDF of the
        # fast fluctuations; their mean across blocks is the run-level descriptor.
        #
        # LabVIEW's Basic / 1D Sample Statistics VIs (2016) return *excess*
        # kurtosis — i.e. Gaussian = 0. See NI reference: "kurtosis" is
        # defined as μ4/σ⁴ − 3.  So the `Kurt` column in the mAstats files
        # is already the excess kurtosis we want to plot; do NOT subtract 3.
        TI          = (dv_dma * np.mean(d["Stan"])) / mean_speed if mean_speed >= 0.5 else np.nan
        skew_mean   = float(np.mean(d["Skew"])) if "Skew" in d and d["Skew"].size else np.nan
        kurt_mean   = float(np.mean(d["Kurt"])) if "Kurt" in d and d["Kurt"].size else np.nan
        median_v    = float(np.median(v_per_s))
        mm_gap_norm = (mean_speed - median_v) / drift_std if drift_std > 1e-6 else 0.0

        results.append({
            "height_mm":   height_mm,
            "mean_speed":  mean_speed,
            "median_speed": median_v,
            "total_unc":   total_unc,
            "drift_std":   drift_std,
            "TI":          TI,
            "skew":        skew_mean,
            "excess_kurt": kurt_mean,   # LabVIEW already returns excess kurt
            "mm_gap_norm": mm_gap_norm,
        })

    results.sort(key=lambda x: x["height_mm"])
    return results


def read_nowind_ref(filepath):
    """
    Read a nowind mAstats file WITHOUT the below-zero filter.
    Returns {"height_mm", "mean_mA", "std_mA", "noise_speed_ms"} or None.

    The sensor reads ~3.96 mA (just below 4.0 mA zero) with no wind →
    direct speed formula gives imaginary values, so speed is 0 m/s.
    noise_speed_ms = sqrt(2 * std_mA / 16 * 100 / rho) quantifies the
    equivalent speed-domain noise floor (typically ~0.06 m/s).
    """
    fname = os.path.basename(filepath)
    hmatch = height_pattern.search(fname)
    if not hmatch:
        return None
    height_mm = int(hmatch.group(1))
    d = parse_stats_file(filepath)
    if "Arit" not in d or "Stan" not in d or d["Arit"].size == 0:
        return None
    mean_mA = float(np.mean(d["Arit"]))
    std_mA  = float(np.mean(d["Stan"]))          # mean within-run std across blocks
    # noise floor in speed units: linearise around zero
    noise_speed_ms = float(np.sqrt(2.0 * std_mA / 16.0 * 100.0 / AIR_DENSITY))
    return {
        "height_mm":      height_mm,
        "mean_mA":        mean_mA,
        "std_mA":         std_mA,
        "noise_speed_ms": noise_speed_ms,
    }


# nowind reference — 06.11, moh59, no wind, fan off
# Sensor reads ~3.962 mA (below 4.0 mA zero → speed = 0 m/s by definition).
# Used as a noise-floor reference marker on the lowestwind individual-run plot.
NOWIND_MOH59_FILE = (
    "/Users/ole/Kodevik/wave_project/pressuredata"
    "/20251106-lowestwindUtenProbe2-fullpanel-amp0100-freq1300"
    "/20251106-lowestwindUtenProbe2-fullpanel-amp0100-freq1300-BONUS"
    "/20251106-nowind-pitot10075-mAstats-moh59.txt"
)
NOWIND_MOH59 = read_nowind_ref(NOWIND_MOH59_FILE)
if NOWIND_MOH59:
    print(f"Nowind ref @ {NOWIND_MOH59['height_mm']} mm: "
          f"Arit={NOWIND_MOH59['mean_mA']:.3f} mA, "
          f"Stan={NOWIND_MOH59['std_mA']:.3f} mA, "
          f"noise_floor≈{NOWIND_MOH59['noise_speed_ms']:.3f} m/s")


# %%
# --- load all datasets ---
all_datasets = []
for folder, label in FULLWIND_DATASETS:
    res = process_folder(folder)
    print(f"{label}: {len(res)} heights")
    all_datasets.append((res, label))

# --- combine: mean speed at each height across runs ---
# height → list of (mean_speed, total_unc) from each run that measured it
by_height = defaultdict(list)
for res, label in all_datasets:
    for r in res:
        by_height[r["height_mm"]].append((r["mean_speed"], r["total_unc"]))

combined_heights = sorted(by_height.keys())
combined_speed   = np.array([np.mean([v for v, _ in by_height[z]]) for z in combined_heights])
combined_spread  = np.array([np.std( [v for v, _ in by_height[z]], ddof=1)
                             if len(by_height[z]) > 1 else
                             by_height[z][0][1]           # fall back to single-run uncertainty
                             for z in combined_heights])
n_runs_per_height = np.array([len(by_height[z]) for z in combined_heights])

# %%
colors  = ["tab:blue", "tab:orange", "tab:green"]
markers = ['o', 's', '^']

# tick layout: label every other, minor gridline for the rest
# 380 (roof) is always forced into the labeled set
all_tick_heights = sorted(set(combined_heights + ([ROOF_MM] if SHOW_ROOF else [])))
labeled   = [h for i, h in enumerate(all_tick_heights) if i % 2 == 0]
unlabeled = [h for i, h in enumerate(all_tick_heights) if i % 2 == 1]
if SHOW_ROOF and ROOF_MM in unlabeled:
    unlabeled.remove(ROOF_MM)
    labeled   = sorted(labeled + [ROOF_MM])

def style_ax(ax):
    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.set_ylabel("Høyde over vannet [mm]")
    ax.set_yticks(labeled)
    ax.set_yticks(unlabeled, minor=True)
    ax.tick_params(axis='y', which='major', labelsize=7)
    ax.tick_params(axis='y', which='minor', length=3, labelsize=0)
    ax.yaxis.grid(True, which='minor', linestyle='--', linewidth=0.3, alpha=0.5)
    if SHOW_ROOF:
        ax.set_ylim(bottom=0, top=ROOF_MM)
        ax.axhline(ROOF_MM, color='brown', linewidth=1.0, linestyle='-', alpha=0.6)

# --- figure 1: individual runs + combined profile ---
fig1, ax1 = plt.subplots(figsize=(7, 7))
fig1.suptitle("Full vind — kombinert vindprofil", fontsize=12)

for (res, label), c, m in zip(all_datasets, colors, markers):
    z   = np.array([r["height_mm"]  for r in res])
    spd = np.array([r["mean_speed"] for r in res])
    ax1.plot(spd, z, marker=m, linestyle='--', color=c, alpha=0.35,
             linewidth=1, markersize=4, label=label)

ax1.plot(combined_speed, combined_heights, marker='D', linestyle='-',
         color='black', linewidth=2, markersize=6, label='Kombinert gjennomsnitt', zorder=5)
ax1.fill_betweenx(combined_heights,
                  combined_speed - combined_spread,
                  combined_speed + combined_spread,
                  alpha=0.15, color='black', label='±spredning mellom kjøringer')

single_mask = n_runs_per_height == 1
if single_mask.any():
    ax1.scatter(combined_speed[single_mask], np.array(combined_heights)[single_mask],
                marker='x', color='black', s=40, zorder=6, label='Kun én kjøring')

ax1.set_xlabel("Vindfart [m/s]")
ax1.set_title("Vindprofil")
ax1.legend(fontsize=8)
style_ax(ax1)
fig1.tight_layout()
if SAVE:
    out = os.path.join(fig_path, f"windprofile_combined_{ts}.pdf")
    fig1.savefig(out, bbox_inches='tight')
    print(f"Saved: {out}")
plt.show()

# --- figure 2: run-to-run spread per height ---
fig2, ax2 = plt.subplots(figsize=(7, 7))
fig2.suptitle("Full vind — variasjon mellom kjøringer", fontsize=12)

ax2.barh(combined_heights, combined_spread * 2, height=4,
         color='steelblue', alpha=0.7, label='Spredning (2×std)')
ax2.set_xlabel("Spredning mellom kjøringer [m/s]")
ax2.set_title("Variasjon mellom kjøringer per høyde")
ax2.legend(fontsize=8)
style_ax(ax2)
fig2.tight_layout()
if SAVE:
    out = os.path.join(fig_path, f"windprofile_spread_{ts}.pdf")
    fig2.savefig(out, bbox_inches='tight')
    print(f"Saved: {out}")
plt.show()


# %%
# --- figure 3: fullwind combined vs lowestwind combined on same axes ---

def _combine(datasets):
    """Return (combined_heights, combined_speed, combined_spread, n_per_height, all_datasets_loaded)."""
    by_h = defaultdict(list)
    loaded = []
    for entry in datasets:
        folder, label = entry[0], entry[1]
        kwargs = entry[2] if len(entry) > 2 else {}
        res = process_folder(folder, **kwargs)
        print(f"  {label}: {len(res)} heights")
        loaded.append((res, label))
        for r in res:
            by_h[r["height_mm"]].append((r["mean_speed"], r["total_unc"]))
    heights = sorted(by_h.keys())
    speed   = np.array([np.mean([v for v, _ in by_h[z]]) for z in heights])
    spread  = np.array([
        np.std([v for v, _ in by_h[z]], ddof=1) if len(by_h[z]) > 1
        else by_h[z][0][1]
        for z in heights
    ])
    n_per   = np.array([len(by_h[z]) for z in heights])
    return heights, speed, spread, n_per, loaded

print("--- fullwind ---")
fw_heights, fw_speed, fw_spread, fw_n, fw_loaded = _combine(FULLWIND_DATASETS)
print("--- lowestwind ---")
lw_heights, lw_speed, lw_spread, lw_n, lw_loaded = _combine(LOWESTWIND_DATASETS)

# y-tick layout for combined figure
all_tick_h3 = sorted(set(fw_heights + lw_heights + ([ROOF_MM] if SHOW_ROOF else [])))
labeled3   = [h for i, h in enumerate(all_tick_h3) if i % 2 == 0]
unlabeled3 = [h for i, h in enumerate(all_tick_h3) if i % 2 == 1]
if SHOW_ROOF and ROOF_MM in unlabeled3:
    unlabeled3.remove(ROOF_MM)
    labeled3 = sorted(labeled3 + [ROOF_MM])

def style_ax3(ax):
    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.set_ylabel("Høyde over vannet [mm]")
    ax.set_yticks(labeled3)
    ax.set_yticks(unlabeled3, minor=True)
    ax.tick_params(axis='y', which='major', labelsize=7)
    ax.tick_params(axis='y', which='minor', length=3, labelsize=0)
    ax.yaxis.grid(True, which='minor', linestyle='--', linewidth=0.3, alpha=0.5)
    if SHOW_ROOF:
        ax.set_ylim(bottom=0, top=ROOF_MM)
        ax.axhline(ROOF_MM, color='brown', linewidth=0.2, linestyle='-', alpha=0.6)

fw_color = "tab:red"
lw_color = "tab:green"
fw_markers_ind = ['o', 's', '^', 'D', 'v']
lw_markers_ind = ['o', 's', '^', 'D', 'v']

# heights in cm just for this plot
fw_heights_cm = np.array(fw_heights) / 10.0
lw_heights_cm = np.array(lw_heights) / 10.0
ROOF_CM       = ROOF_MM / 10.0

fig3, ax3 = plt.subplots(figsize=(9, 7))

ax3.errorbar(fw_speed, fw_heights_cm, xerr=fw_spread,
             fmt='D-', color=fw_color, linewidth=0.1, markersize=3,
             capsize=4, label="Full vind", zorder=5)
ax3.fill_betweenx(fw_heights_cm,
                  fw_speed - fw_spread, fw_speed + fw_spread,
                  alpha=0.15, color=fw_color)

ax3.errorbar(lw_speed, lw_heights_cm, xerr=lw_spread,
             fmt='o-', color=lw_color, linewidth=0.1, markersize=3,
             capsize=4, label="Laveste vind", zorder=5)
ax3.fill_betweenx(lw_heights_cm,
                  lw_speed - lw_spread, lw_speed + lw_spread,
                  alpha=0.15, color=lw_color)

ax3.set_xlabel("Vindfart [m/s]")
ax3.set_xlim(left=0)
ax3.set_ylabel("Høyde over vannet [cm]")
ax3.set_ylim(0, ROOF_CM)
ax3.set_yticks(range(0, int(ROOF_CM) + 1,2))           # 0, 1, 2, ..., 38
ax3.set_yticks(range(0, int(ROOF_CM) + 1), minor=True) #grid only           # 0, 1, 2, ..., 38
ax3.grid(True, which='minor', linestyle='--', linewidth=0.5, alpha=0.4)
# ax3.axhline(ROOF_CM, color='brown', linewidth=1.0, linestyle='-',
#             alpha=0.6, label="Tak")
ax3.legend(fontsize=12, loc='best')


from matplotlib.patches import Rectangle
import matplotlib.transforms as mtransforms

# Pitot probe: hangs from roof (38 cm) down to 27.2 cm — 10.8 cm long
PROBE_TIP_CM    = 27.2
PROBE_X_FRAC    = 0.93          # 93% from the left of the axes
PROBE_WIDTH_FRAC = 0.025        # 2.5% of axes width — skinny

trans = mtransforms.blended_transform_factory(ax3.transAxes, ax3.transData)

probe = Rectangle(
    (PROBE_X_FRAC, PROBE_TIP_CM),                # (x_left, y_bottom)
    PROBE_WIDTH_FRAC,                            # width (axes-fraction)
    ROOF_CM - PROBE_TIP_CM,                      # height (data, cm)
    facecolor='lightgray', edgecolor='black', linewidth=0.6,
    alpha=0.85, zorder=4, transform=trans,
)
ax3.add_patch(probe)

# optional small label next to the tip
ax3.text(PROBE_X_FRAC + PROBE_WIDTH_FRAC + 0.005, PROBE_TIP_CM,
         "pitotrør\n(rekkevidde)", transform=trans,
         ha='left', va='center', fontsize=7, color='dimgray')

fig3.tight_layout()
if SAVE:
    out = os.path.join(fig_path, f"windprofile_combined_conditions_{ts}.pdf")
    fig3.savefig(out, bbox_inches='tight')
    print(f"Saved: {out}")
plt.show()
# %%
# --- figure 4: same as figure 3 but log y-axis in metres ---

def mm_to_m(arr):
    return np.array(arr) / 1000.0

fig4, ax4 = plt.subplots(figsize=(7, 7))
fig4.suptitle("Vindprofil — full og laveste vind kombinert (log høyde)", fontsize=12)

for (res, label), m in zip(fw_loaded, fw_markers_ind):
    z   = mm_to_m([r["height_mm"]  for r in res])
    spd = np.array([r["mean_speed"] for r in res])
    ax4.plot(spd, z, marker=m, linestyle='--', color=fw_color, alpha=0.30,
             linewidth=0.1, markersize=4, label=f"Full vind – {label}")

for (res, label), m in zip(lw_loaded, lw_markers_ind):
    z   = mm_to_m([r["height_mm"]  for r in res])
    spd = np.array([r["mean_speed"] for r in res])
    ax4.plot(spd, z, marker=m, linestyle='--', color=lw_color, alpha=0.30,
             linewidth=0.1, markersize=4, label=f"Laveste vind – {label}")

fw_heights_m = mm_to_m(fw_heights)
lw_heights_m = mm_to_m(lw_heights)

ax4.errorbar(fw_speed, fw_heights_m, xerr=fw_spread,
             fmt='D-', color=fw_color, linewidth=0.5, markersize=6,
             capsize=4, label=f"Full vind kombinert (n={len(FULLWIND_DATASETS)})", zorder=5)
ax4.fill_betweenx(fw_heights_m,
                  fw_speed - fw_spread, fw_speed + fw_spread,
                  alpha=0.15, color=fw_color)

ax4.errorbar(lw_speed, lw_heights_m, xerr=lw_spread,
             fmt='o-', color=lw_color, linewidth=0.5, markersize=6,
             capsize=4, label=f"Laveste vind kombinert (n={len(LOWESTWIND_DATASETS)})", zorder=5)
ax4.fill_betweenx(lw_heights_m,
                  lw_speed - lw_spread, lw_speed + lw_spread,
                  alpha=0.15, color=lw_color)

for heights_m, speed_arr, n_arr, color in [
    (fw_heights_m, fw_speed, fw_n, fw_color),
    (lw_heights_m, lw_speed, lw_n, lw_color),
]:
    single = n_arr == 1
    if single.any():
        ax4.scatter(speed_arr[single], heights_m[single],
                    marker='x', color=color, s=50, zorder=6, label='Kun én kjøring')

ax4.set_xlabel("Vindfart [m/s]")
ax4.set_title("Kombinert vindprofil per vindkondisjon (log høyde)")
ax4.set_xlim(left=0)
ax4.set_yscale('log')
ax4.set_ylim(0.01, 1.0)
ax4.set_ylabel("Høyde over vannet [m]")
ax4.grid(True, which='major', linestyle='--', linewidth=0.5)
ax4.grid(True, which='minor', linestyle='--', linewidth=0.3, alpha=0.4)
if SHOW_ROOF:
    ax4.axhline(ROOF_MM / 1000, color='brown', linewidth=1.0, linestyle='-', alpha=0.6,
                label=f"Tak ({ROOF_MM} mm)")
ax4.legend(fontsize=8)
fig4.tight_layout()
if SAVE:
    out = os.path.join(fig_path, f"windprofile_combined_conditions_log_{ts}.pdf")
    fig4.savefig(out, bbox_inches='tight')
    print(f"Saved: {out}")
plt.show()

# --- figure 5: log plot with negated x-axis (−7 to −2) to flip the profiles ---
# Aspect ratio (6.78, 6.15) mirrors the Vollestad & Jensen (2021) wind-profile figure
# we reference for thesis comparison.
fig5, ax5 = plt.subplots(figsize=(6.78, 6.15))

# individual runs: plotted faintly for context, NOT shown in legend
for (res, label), m in zip(fw_loaded, fw_markers_ind):
    z   = mm_to_m([r["height_mm"]  for r in res])
    spd = -np.array([r["mean_speed"] for r in res])
    ax5.plot(spd, z, marker=m, linestyle='--', color=fw_color, alpha=0.30,
             linewidth=0.5, markersize=4, label="_nolegend_")

for (res, label), m in zip(lw_loaded, lw_markers_ind):
    z   = mm_to_m([r["height_mm"]  for r in res])
    spd = -np.array([r["mean_speed"] for r in res])
    ax5.plot(spd, z, marker=m, linestyle='--', color=lw_color, alpha=0.30,
             linewidth=0.5, markersize=4, label="_nolegend_")

ax5.errorbar(-fw_speed, fw_heights_m, xerr=fw_spread,
             fmt='D-', color=fw_color, linewidth=0.5, markersize=6,
             capsize=4, label="Full vind", zorder=5)
ax5.fill_betweenx(fw_heights_m,
                  -fw_speed - fw_spread, -fw_speed + fw_spread,
                  alpha=0.15, color=fw_color)

ax5.errorbar(-lw_speed, lw_heights_m, xerr=lw_spread,
             fmt='o-', color=lw_color, linewidth=0.5, markersize=6,
             capsize=4, label="Laveste vind", zorder=5)
ax5.fill_betweenx(lw_heights_m,
                  -lw_speed - lw_spread, -lw_speed + lw_spread,
                  alpha=0.15, color=lw_color)

# single-run markers kept on plot but not in legend
for heights_m, speed_arr, n_arr, color in [
    (fw_heights_m, fw_speed, fw_n, fw_color),
    (lw_heights_m, lw_speed, lw_n, lw_color),
]:
    single = n_arr == 1
    if single.any():
        ax5.scatter(-speed_arr[single], heights_m[single],
                    marker='x', color=color, s=50, zorder=6, label="_nolegend_")

ax5.set_xlabel("Vindfart [m/s]")
ax5.set_xlim(-7, -2)
ax5.set_yscale('log')
ax5.set_ylim(0.001, 1.0)
ax5.set_ylabel("Høyde over vannet [m]")
ax5.grid(True, which='major', linestyle='--', linewidth=0.5)
ax5.grid(True, which='minor', linestyle='--', linewidth=0.3, alpha=0.4)
if SHOW_ROOF:
    ax5.axhline(ROOF_MM / 1000, color='brown', linewidth=1.0, linestyle='-', alpha=0.6,
                label="Tak")
ax5.legend(fontsize=9, loc='best')
# force axes rectangle to match Vollestad & Jensen (2021) axes aspect.
# Aspect measured by pixel-scanning the reference PDF axes spines (ignoring a
# 1-pixel black crop-edge artifact at row 0): h/w ≈ 0.9064, consistent with the
# hand-measured 615/678 ratio from the cropped PDF.
ax5.set_box_aspect(0.9064)
fig5.tight_layout()
if SAVE:
    out = os.path.join(fig_path, f"windprofile_combined_conditions_log_flipped_{ts}.pdf")
    fig5.savefig(out, bbox_inches='tight')
    print(f"Saved: {out}")
plt.show()

# TODO  - need a variant of a windprofile plot which is perfectly square. Im trying to copy the profile in VOLLESTAD: '/Users/ole/Kodevik/wave_litterature_backup/Vollestad and Jensen - 2021 - Modification of Airflow Structure Due to Wave Breaking on a Submerged Topography.pdf'

# %%
# --- figures 6 & 7: individual runs per condition (for outlier inspection) ---
# Each dataset shown as its own clearly labelled line; no combining.
# Figure 6 = fullwind, Figure 7 = lowestwind.

_ind_colors = plt.cm.tab10.colors
_ind_markers = ['o', 's', '^', 'D', 'v', 'p', 'h', 'P', '*', 'X']


def _plot_individual_runs(loaded, color_list, marker_list, cond_label, file_tag,
                          labels=None, nowind_ref=None):
    """Two panels side by side: linear (mm) and log (m). X-axis auto-fitted.
    labels: optional list of strings to override the labels from loaded.
    nowind_ref: optional dict from read_nowind_ref() — drawn as a grey × marker at v=0
                with a rightward errorbar showing the noise floor in m/s."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 7), sharey=False)
    fig.suptitle(f"Vindprofil — {cond_label} — enkeltmålinger", fontsize=12)

    for i, (res, label) in enumerate(loaded):
        if labels is not None and i < len(labels):
            label = labels[i]
        color  = color_list[i % len(color_list)]
        marker = marker_list[i % len(marker_list)]
        z_mm   = np.array([r["height_mm"]  for r in res])
        spd    = np.array([r["mean_speed"] for r in res])
        unc    = np.array([r["total_unc"]  for r in res])

        axL.errorbar(spd, z_mm, xerr=unc, fmt=f"{marker}-", capsize=3,
                     linewidth=1.5, markersize=5, color=color, label=label)
        axR.errorbar(spd, z_mm / 1000, xerr=unc, fmt=f"{marker}-", capsize=3,
                     linewidth=1.5, markersize=5, color=color, label=label)

    # linear panel
    axL.set_xlabel("Vindfart [m/s]")
    axL.set_ylabel("Høyde over vannet [mm]")
    if SHOW_ROOF:
        axL.set_ylim(bottom=0, top=ROOF_MM)
        axL.axhline(ROOF_MM, color='brown', linewidth=1, linestyle='-', alpha=0.6, label="Tak")
    axL.grid(True, linestyle='--', linewidth=0.5)
    axL.legend(fontsize=8)

    # log panel
    axR.set_xlabel("Vindfart [m/s]")
    axR.set_ylabel("Høyde over vannet [m]")
    axR.set_yscale('log')
    axR.set_ylim(0.01, 1.0)
    if SHOW_ROOF:
        axR.axhline(ROOF_MM / 1000, color='brown', linewidth=1, linestyle='-', alpha=0.6,
                    label="Tak")
    axR.grid(True, which='major', linestyle='--', linewidth=0.5)
    axR.grid(True, which='minor', linestyle='--', linewidth=0.3, alpha=0.4)
    axR.legend(fontsize=8)

    if nowind_ref is not None:
        z_mm  = nowind_ref["height_mm"]
        noise = nowind_ref["noise_speed_ms"]
        label = (f"nowind referanse @ {z_mm} mm\n"
                 f"(Arit≈{nowind_ref['mean_mA']:.3f} mA, "
                 f"Stan≈{nowind_ref['std_mA']:.3f} mA, "
                 f"støynivå≈{noise:.3f} m/s)")
        _nw_kw = dict(fmt='x', color='dimgray', markersize=10, markeredgewidth=2,
                      capsize=4, linewidth=1.5, zorder=6, label=label)
        # errorbar extends rightward only (xerr = [[left], [right]] per point)
        axL.errorbar([0], [z_mm],        xerr=[[0], [noise]], **_nw_kw)
        axR.errorbar([0], [z_mm / 1000], xerr=[[0], [noise]], **_nw_kw)
        for ax in (axL, axR):
            ax.legend(fontsize=8)

    fig.tight_layout()
    if SAVE:
        out = os.path.join(fig_path, f"windprofile_individual_{file_tag}_{ts}.pdf")
        fig.savefig(out, bbox_inches='tight')
        print(f"Saved: {out}")
    plt.show()


FW_FULL_LABELS = [
    *(["04. november 2025 – med bølgeprobe i strømmen"] if INCLUDE_04NOV else []),
    *(["04. november 2025 – med probe, bonus langtidsserie (moh031 = 30 mm)"] if INCLUDE_04NOV else []),
    "05. november 2025 – UP2 med panel, uten bølger",
    "06. november 2025 – UP2 med panel, uten bølger",
    "06. november 2025 – UP2 med panel, med bølger (0.1V, 1.3Hz)",
]
LW_FULL_LABELS = [
    "05. november 2025 – uten panel",
    "05. november 2025 – med panel",
    # bonus: folder mislabelled 20251104 but content is 05.11; moh016 = actual 15 mm (readme: +1 mm convention for long runs)
    "05. november 2025 – med panel, bonus langtidsserie (moh016 = 15 mm)",
    "05. november 2025 – UP2 med panel",
    "06. november 2025 – angletest 0° (moh050, n=30 + n=360 extratid)",
    "06. november 2025 – med bølger bonus (moh050, mixed wave/no-wave periods)",
    "06. november 2025 – med bølger (0.1V, 1.3Hz)",
]

_plot_individual_runs(fw_loaded, _ind_colors, _ind_markers, "Full vind", "fullwind",
                      labels=FW_FULL_LABELS)
_plot_individual_runs(lw_loaded, _ind_colors, _ind_markers, "Laveste vind", "lowestwind",
                      labels=LW_FULL_LABELS, nowind_ref=NOWIND_MOH59)
# %%

# --- figure 8: tunnelTest 7-roof — pappTett vs slisse (standalone) ---
#
# Dataset: 20251107-lowestwindUP2-fullpanel-tunnelTest
#   pappTett = properly sealed 7-plate roof  → same sealed-tunnel condition as all main experiments
#   slisse   = longitudinal slit along the roof length → physically distinct "vacuum cleaner"
#              flow (suction through slot); NOT representative of the main tunnel config.
#              Included here for comparison only — do NOT mix into the main analysis.
#
# NOTE: both conditions live in the same STATS subfolder; distinguished by fname_filter.
# NOTE: pappTett moh049 has two runs — n=120 (3.631 m/s) and n=120 ekstratid (3.973 m/s).
#       process_folder returns both as separate height-49 entries and _combine merges them.
#       The spread between those two short runs is the uncertainty shown.
#
TUNNELTEST_STATS = (
    "/Users/ole/Kodevik/wave_project/pressuredata"
    "/20251107-lowestwindUP2-fullpanel-tunnelTest"
    "/20251107-lowestwindUP2-fullpanel-tunnelTest-STATS"
)

tt_papptett = process_folder(TUNNELTEST_STATS, fname_filter="pappTett")
tt_slisse   = process_folder(TUNNELTEST_STATS, fname_filter="slisse")

fig8, (ax8L, ax8R) = plt.subplots(1, 2, figsize=(12, 7), sharey=False)
fig8.suptitle(
    "TunnelTest 7-tak — 07. november 2025, laveste vind\n"
    "pappTett (tett tak, gyldig) vs slisse (langsgående sprekk, annen fysikk)",
    fontsize=11,
)

_tt_datasets = [
    (tt_papptett, "pappTett (tett tak)", "tab:blue",  "o"),
    (tt_slisse,   "slisse (sprekk)",     "tab:red",   "s"),
]

for res, label, color, marker in _tt_datasets:
    if not res:
        continue
    z_mm = np.array([r["height_mm"]  for r in res])
    spd  = np.array([r["mean_speed"] for r in res])
    unc  = np.array([r["total_unc"]  for r in res])
    ax8L.errorbar(spd, z_mm, xerr=unc, fmt=f"{marker}-", capsize=3,
                  linewidth=1.5, markersize=5, color=color, label=label)
    ax8R.errorbar(spd, z_mm / 1000, xerr=unc, fmt=f"{marker}-", capsize=3,
                  linewidth=1.5, markersize=5, color=color, label=label)

# linear panel
ax8L.set_xlabel("Vindfart [m/s]")
ax8L.set_ylabel("Høyde over vannet [mm]")
if SHOW_ROOF:
    ax8L.set_ylim(bottom=0, top=ROOF_MM)
    ax8L.axhline(ROOF_MM, color='brown', linewidth=1, linestyle='-', alpha=0.6, label="Tak")
ax8L.grid(True, linestyle='--', linewidth=0.5)
ax8L.legend(fontsize=9)

# log panel
ax8R.set_xlabel("Vindfart [m/s]")
ax8R.set_ylabel("Høyde over vannet [m]")
ax8R.set_yscale('log')
ax8R.set_ylim(0.01, 1.0)
if SHOW_ROOF:
    ax8R.axhline(ROOF_MM / 1000, color='brown', linewidth=1, linestyle='-', alpha=0.6,
                 label="Tak")
ax8R.grid(True, which='major', linestyle='--', linewidth=0.5)
ax8R.grid(True, which='minor', linestyle='--', linewidth=0.3, alpha=0.4)
ax8R.legend(fontsize=9)

fig8.tight_layout()
if SAVE:
    out = os.path.join(fig_path, f"windprofile_tunneltest_7roof_{ts}.pdf")
    fig8.savefig(out, bbox_inches='tight')
    print(f"Saved: {out}")
plt.show()

# %%
# --- figure 9: 6-tak vs 7-tak lowestwind wind profiles ---
#
# The main LOWESTWIND_DATASETS (05.11 / 06.11) were all recorded with the 6-plate
# roof (normal setup). The 7-plate roof was only used during the 07.11 tunnelTest,
# and only under lowestwind. The slisse variants (open slit) are NOT a representative
# sealed-tunnel condition and are excluded here. The 6-roof single-point from
# ekstratid-pappTett (moh049) is also excluded — it would add one dot on top of
# the well-sampled 6-roof combined profile and clutter the figure.
#
# The profile shapes are what matter — exact height matching between the two
# configurations is not required.

# 7-tak tett profile: select only 7roof-pappTett files (exclude slisse variants)
tt_7roof_tett = process_folder(TUNNELTEST_STATS,
                               fname_filter="7roof-pappTett",
                               exclude_filter="slisse")

# Combine the 4 main 6-roof lowestwind runs (lw_loaded is already loaded above).
# Reuse the existing lw_heights_m / lw_speed / lw_spread arrays — these are the
# combined 6-roof lowestwind profile.

# Build combined 7-roof profile (no across-run averaging needed — single folder,
# but _combine handles the repeated moh049 entry cleanly).
def _combine_single(res_list):
    by_h = defaultdict(list)
    for r in res_list:
        by_h[r["height_mm"]].append((r["mean_speed"], r["total_unc"]))
    heights = sorted(by_h.keys())
    speed   = np.array([np.mean([v for v, _ in by_h[z]]) for z in heights])
    spread  = np.array([
        np.std([v for v, _ in by_h[z]], ddof=1) if len(by_h[z]) > 1
        else by_h[z][0][1]
        for z in heights
    ])
    return np.array(heights), speed, spread

tt7_heights_mm, tt7_speed, tt7_spread = _combine_single(tt_7roof_tett)
tt7_heights_m = tt7_heights_mm / 1000.0

roof6_color = "tab:purple"
roof7_color = "tab:olive"

fig9, ax9 = plt.subplots(figsize=(6.78, 6.15))

# individual 6-roof runs: faint context lines, hidden from legend
_faint_markers = ['o', 's', '^', 'D', 'v', 'p', 'h']
for (res, _label), m in zip(lw_loaded, _faint_markers):
    z   = mm_to_m([r["height_mm"]  for r in res])
    spd = np.array([r["mean_speed"] for r in res])
    ax9.plot(spd, z, marker=m, linestyle='--', color=roof6_color, alpha=0.25,
             linewidth=1, markersize=4, label="_nolegend_")

# combined 6-roof lowestwind profile (4 main runs, already loaded as lw_*)
ax9.errorbar(lw_speed, lw_heights_m, xerr=lw_spread,
             fmt='D-', color=roof6_color, linewidth=2, markersize=6,
             capsize=4, label="6 tak (normalt oppsett)", zorder=5)
ax9.fill_betweenx(lw_heights_m,
                  lw_speed - lw_spread, lw_speed + lw_spread,
                  alpha=0.15, color=roof6_color)

# combined 7-roof lowestwind profile (07.11 tunnelTest pappTett only)
ax9.errorbar(tt7_speed, tt7_heights_m, xerr=tt7_spread,
             fmt='o-', color=roof7_color, linewidth=2, markersize=6,
             capsize=4, label="7 tak (tunneltest 07.11)", zorder=5)
ax9.fill_betweenx(tt7_heights_m,
                  tt7_speed - tt7_spread, tt7_speed + tt7_spread,
                  alpha=0.15, color=roof7_color)

ax9.set_xlabel("Vindfart [m/s]")
ax9.set_ylabel("Høyde over vannet [m]")
ax9.set_yscale('log')
ax9.set_ylim(0.001, 1.0)
# auto-fit x to both profiles with a small margin
_all_speeds = np.concatenate([lw_speed, tt7_speed])
_xpad = 0.2
ax9.set_xlim(max(0, _all_speeds.min() - _xpad), _all_speeds.max() + _xpad)
ax9.grid(True, which='major', linestyle='--', linewidth=0.5)
ax9.grid(True, which='minor', linestyle='--', linewidth=0.3, alpha=0.4)
if SHOW_ROOF:
    ax9.axhline(ROOF_MM / 1000, color='brown', linewidth=1.0, linestyle='-', alpha=0.6,
                label="Tak")
ax9.legend(fontsize=9, loc='best')
fig9.tight_layout()
if SAVE:
    out = os.path.join(fig_path, f"windprofile_6vs7_roof_lowestwind_{ts}.pdf")
    fig9.savefig(out, bbox_inches='tight')
    print(f"Saved: {out}")
plt.show()

# %%
# --- figure 10: log-law fit to the mean profiles ---
#
# Standard boundary-layer characterisation:
#     U(z) = (u*/κ) · ln(z/z₀),    κ = 0.41 (von Kármán)
#
# Fitting U against ln(z) gives:
#     slope     = u* / κ              → u* = slope · κ   [m/s]    (friction velocity)
#     intercept = -(u*/κ) · ln(z₀)    → z₀ = exp(-intercept/slope) [m]  (roughness length)
#
# Full measurement range is used (tallest point 245 mm, still 135 mm below roof).

KAPPA = 0.41

def fit_loglaw(heights_m, speeds):
    """Linear regression of U vs ln(z). Returns (u_star, z0, r2, predict_fn)."""
    heights_m = np.asarray(heights_m)
    speeds    = np.asarray(speeds)
    mask = (heights_m > 0) & np.isfinite(speeds)
    lnz  = np.log(heights_m[mask])
    U    = speeds[mask]
    reg  = linregress(lnz, U)
    u_star = reg.slope * KAPPA
    z0     = float(np.exp(-reg.intercept / reg.slope))
    predict = lambda z: reg.slope * np.log(z) + reg.intercept
    return u_star, z0, reg.rvalue ** 2, predict

fw_ustar, fw_z0, fw_r2, fw_fit = fit_loglaw(fw_heights_m, fw_speed)
lw_ustar, lw_z0, lw_r2, lw_fit = fit_loglaw(lw_heights_m, lw_speed)

print(f"\nLog-law fit  (κ={KAPPA})")
print(f"  Full vind:     u* = {fw_ustar:.3f} m/s,  z₀ = {fw_z0*1000:.3f} mm,  R² = {fw_r2:.4f}")
print(f"  Laveste vind:  u* = {lw_ustar:.3f} m/s,  z₀ = {lw_z0*1000:.3f} mm,  R² = {lw_r2:.4f}")

fig10, ax10 = plt.subplots(figsize=(6.78, 6.15))

# measured profiles with errorbars
def _fmt_z0(z0_m):
    """Format z₀ so very small values are readable (mm + scientific in m)."""
    z0_mm = z0_m * 1000
    if z0_mm >= 0.01:
        return f"{z0_mm:.3f} mm"
    return f"{z0_m:.2e} m"

ax10.errorbar(fw_speed, fw_heights_m, xerr=fw_spread,
              fmt='D', color=fw_color, markersize=6, capsize=4,
              linestyle='none',
              label=f"Full vind  (u*={fw_ustar:.3f} m/s, z₀={_fmt_z0(fw_z0)})",
              zorder=5)
ax10.errorbar(lw_speed, lw_heights_m, xerr=lw_spread,
              fmt='o', color=lw_color, markersize=6, capsize=4,
              linestyle='none',
              label=f"Laveste vind  (u*={lw_ustar:.3f} m/s, z₀={_fmt_z0(lw_z0)})",
              zorder=5)

# fitted log-law lines: sample z across the data range, extrapolate slightly
z_fit = np.logspace(np.log10(0.005), np.log10(0.4), 200)
ax10.plot(fw_fit(z_fit), z_fit, '--', color=fw_color, linewidth=1.4, alpha=0.8,
          label="_nolegend_")
ax10.plot(lw_fit(z_fit), z_fit, '--', color=lw_color, linewidth=1.4, alpha=0.8,
          label="_nolegend_")

ax10.set_xlabel("Vindfart [m/s]")
ax10.set_ylabel("Høyde over vannet [m]")
ax10.set_yscale('log')
ax10.set_ylim(0.001, 1.0)
ax10.set_xlim(left=0)
ax10.grid(True, which='major', linestyle='--', linewidth=0.5)
ax10.grid(True, which='minor', linestyle='--', linewidth=0.3, alpha=0.4)
if SHOW_ROOF:
    ax10.axhline(ROOF_MM / 1000, color='brown', linewidth=1.0, linestyle='-',
                 alpha=0.6, label="Tak")

# R² annotation box
ann = (f"$U(z) = \\frac{{u_*}}{{\\kappa}}\\,\\ln(z/z_0)$,  $\\kappa={KAPPA}$\n"
       f"Full vind:     R² = {fw_r2:.3f}\n"
       f"Laveste vind:  R² = {lw_r2:.3f}")
ax10.text(0.98, 0.02, ann, transform=ax10.transAxes, ha='right', va='bottom',
          fontsize=8, family='serif',
          bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray', boxstyle='round,pad=0.4'))

ax10.legend(fontsize=8, loc='upper left')
ax10.set_box_aspect(0.9064)
fig10.tight_layout()
if SAVE:
    out = os.path.join(fig_path, f"windprofile_loglaw_fit_{ts}.pdf")
    fig10.savefig(out, bbox_inches='tight')
    print(f"Saved: {out}")
plt.show()

# %%
# --- figure 11: turbulence character (TI, skew, excess kurtosis vs height) ---
#
# Wind-tunnel quality / turbulence-PDF characterisation. Each quantity is the
# mean across all runs at each height, so it mirrors the combined profile.
#
# Reference lines mark Gaussian turbulence: skew = 0, excess kurt = 0.
# Shaded "quiet" bands show typical wind-tunnel acceptance ranges:
#   - TI:         <5% is considered low-turbulence
#   - |skew|:     <0.3 passes a symmetry check
#   - |exc kurt|: <1.0 passes a Gaussian-tail check

def combine_stat(loaded, key):
    """For each height, return mean value of `key` across runs that measured it."""
    by_h = defaultdict(list)
    for res, _ in loaded:
        for r in res:
            if key in r and not (isinstance(r[key], float) and np.isnan(r[key])):
                by_h[r["height_mm"]].append(r[key])
    heights = sorted(by_h.keys())
    means   = np.array([np.mean(by_h[z]) for z in heights])
    return np.array(heights) / 1000.0, means

fw_z_TI,   fw_TI   = combine_stat(fw_loaded, "TI")
fw_z_sk,   fw_sk   = combine_stat(fw_loaded, "skew")
fw_z_ek,   fw_ek   = combine_stat(fw_loaded, "excess_kurt")
lw_z_TI,   lw_TI   = combine_stat(lw_loaded, "TI")
lw_z_sk,   lw_sk   = combine_stat(lw_loaded, "skew")
lw_z_ek,   lw_ek   = combine_stat(lw_loaded, "excess_kurt")

# 3 panels stacked horizontally, shared y-axis (log)
fig11, axes11 = plt.subplots(1, 3, figsize=(11.5, 6.15), sharey=True)
axT, axS, axK = axes11

def _plot_metric(ax, xlabel, fw_z, fw_v, lw_z, lw_v):
    """Scatter (no connecting lines — height points aren't monotonic enough)."""
    ax.scatter(fw_v, fw_z, marker='D', color=fw_color, s=35,
               edgecolor='white', linewidth=0.5, zorder=4, label="Full vind")
    ax.scatter(lw_v, lw_z, marker='o', color=lw_color, s=35,
               edgecolor='white', linewidth=0.5, zorder=4, label="Laveste vind")
    ax.set_xlabel(xlabel)
    ax.grid(True, which='major', linestyle='--', linewidth=0.5)
    ax.grid(True, which='minor', linestyle='--', linewidth=0.3, alpha=0.4)
    if SHOW_ROOF:
        ax.axhline(ROOF_MM / 1000, color='brown', linewidth=1.0, alpha=0.6)

# TI panel (as %)
_plot_metric(axT, "TI [%]", fw_z_TI, fw_TI * 100, lw_z_TI, lw_TI * 100)
axT.axvspan(0, 5, color='green', alpha=0.07, label='_nolegend_')
axT.axvline(5, color='green', linestyle=':', linewidth=1, alpha=0.7, label="TI = 5 %")
_TI_max = np.nanmax(np.concatenate([fw_TI, lw_TI])) * 100
axT.set_xlim(0, max(8, _TI_max * 1.1))

# Skew panel — Gaussian at 0
_plot_metric(axS, "Skjevhet", fw_z_sk, fw_sk, lw_z_sk, lw_sk)
axS.axvspan(-0.3, 0.3, color='green', alpha=0.07, label='_nolegend_')
axS.axvline(0, color='black', linestyle=':', linewidth=1, alpha=0.7, label="Gauss (skjevhet = 0)")
_sk_absmax = np.nanmax(np.abs(np.concatenate([fw_sk, lw_sk])))
axS.set_xlim(-max(0.5, _sk_absmax * 1.1), max(0.5, _sk_absmax * 1.1))

# Excess kurtosis panel — Gaussian at 0. Source column is LabVIEW `Kurt`
# from the mAstats files, which is *already* excess kurt (NI convention,
# μ4/σ⁴ − 3). We do not subtract anything further.
_plot_metric(axK, "Ekstra Kurtose (Gauss = 0)", fw_z_ek, fw_ek, lw_z_ek, lw_ek)
axK.axvspan(-1, 1, color='green', alpha=0.07, label='_nolegend_')
axK.axvline(0, color='black', linestyle=':', linewidth=1, alpha=0.7, label="Gauss (exc.kurt = 0)")
_ek_max = np.nanmax(np.concatenate([fw_ek, lw_ek]))
_ek_min = np.nanmin(np.concatenate([fw_ek, lw_ek]))
axK.set_xlim(min(-1.5, _ek_min * 1.1), max(2.0, _ek_max * 1.1))

axT.set_yscale('log')
axT.set_ylim(0.001, 1.0)
axT.set_ylabel("Høyde over vannet [m]")

# legends: show each axis's Gaussian reference plus data lines; compact
for ax in (axT, axS, axK):
    ax.legend(fontsize=8, loc='best')

fig11.suptitle("Turbulenskarakter — TI, skevhet og ekstra kurtose per høyde", fontsize=12)
fig11.tight_layout()
if SAVE:
    out = os.path.join(fig_path, f"windprofile_turbulence_character_{ts}.pdf")
    fig11.savefig(out, bbox_inches='tight')
    print(f"Saved: {out}")
plt.show()

# %%
# --- final step: regenerate the Vollestad comparison overlay ---
# Side-effect script: rasterises the fig5 PDF we just wrote and aligns it
# pixel-wise with Vollestad & Jensen (2021)'s published profile.
if SAVE:
    try:
        from vollestad_overlay import main as _overlay_main
        _overlay_main()
    except Exception as e:
        print(f"Overlay step skipped ({type(e).__name__}: {e})")
# %%
