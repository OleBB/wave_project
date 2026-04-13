# %%
"""
Oct 28–29 2025 — sealing day pitot data, full and lowest wind.

Height encoding: _NN = NN×10 mm above water (confirmed by comment in _08:
  "nå er 80 = fil_08, som så må regnes mot vannoverflaten. (dybden er 58.)")

Sensor calibrated 15:05 Oct 28 → zero = 4.0 mA, 100 Pa, Nov formula.

Units:
  20251028-pitot-fullwind:    m/s  (VI outputting pre-computed wind speed)
  20251028-pitot-lowestwind:  m/s  (same VI)
  20251029-pitot-lowestwind:  unknown — 0.37–0.44, too low for m/s (Oct30 gives 3.7 m/s
                               at 70mm), too low for mA. Possibly Volts from brief DAQ
                               reconfiguration, or fan not running properly. Treat as suspect.

Cross-reference (Oct 30): speed_070.txt gives 3.66–3.73 m/s while stats_070.txt shows
Arit ~5.3 mA → sqrt(2*1.3/16*100/1.225) = 3.64 m/s. Confirms Oct 28 values are m/s.

Oct 28 lowestwind notes from file comments:
  _01–07: 3.26 mA (below zero, fan off or positioning)
  _07:    "falskfil" — discard
  _08:    "ny posisjon. nå er 80 = fil_08" → 80 mm
  _09:    "30 sek på 90"  → 90 mm
  _10:    "30 sek på 100" → 100 mm
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import os, re

SAVE_DIR = Path("/Users/ole/Kodevik/wave_project/windresults")
SAVE_DIR.mkdir(exist_ok=True)
SAVE = True
AIR_DENSITY = 1.225
ROOF_MM = 380
SHUNT_OHM = 100.0   # Ω — Oct 29 DAQ shunt resistor


def ma_to_speed(ma):
    return np.sqrt(np.maximum(2.0 * (ma - 4.0) / 16.0 * 100.0 / AIR_DENSITY, 0.0))


def read_value(filepath):
    """Return first numeric non-NaN value after X_Value header, or None."""
    with open(filepath, encoding="latin-1") as f:
        lines = f.readlines()
    start = None
    for i, l in enumerate(lines):
        if l.strip().startswith("X_Value"):
            start = i + 1
            break
    if start is None:
        return None
    for l in lines[start:]:
        for p in l.strip().split("\t"):
            p = p.replace(",", ".").strip()
            try:
                v = float(p)
                return None if np.isnan(v) else v
            except ValueError:
                continue
    return None


def load_panelposisjon_folder(folder, unit="ms", exclude_fnames=None):
    """
    Load _NN.lvm files. unit: "ms" = already m/s (Oct 28), "mA", or "V" (Volts/shunt).
    Returns sorted list of {height_mm, mean_v, above_zero}.
    """
    exclude = set(exclude_fnames or [])
    rows = []
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".lvm") or fname in exclude:
            continue
        m = re.search(r"_(\d{2})\.lvm", fname)
        if not m:
            continue
        h = int(m.group(1)) * 10
        val = read_value(os.path.join(folder, fname))
        if val is None:
            continue
        if unit == "ms":
            v = val
        elif unit == "V":
            v = ma_to_speed(val / SHUNT_OHM * 1000.0)
        else:  # mA
            v = ma_to_speed(val)
        rows.append({"height_mm": h, "fname": fname, "raw": val,
                     "mean_v": v, "above_zero": v > 0.1})
    rows.sort(key=lambda r: r["height_mm"])
    return rows


def load_oct29_lowestwind(folder):
    """
    Oct 29 uses non-standard filenames and Volt units.
    pitot-lowestwind-panelposisjon9960mm-h_070.lvm → 70 mm, Volts.
    pitot-lowestwind-panelposisjon9960mm-h_.lvm    → unknown height, skip.
    """
    rows = []
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".lvm"):
            continue
        m = re.search(r"-h_(\d+)\.lvm", fname)
        if not m:
            continue
        h = int(m.group(1))
        val = read_value(os.path.join(folder, fname))
        if val is None:
            continue
        ma = val / SHUNT_OHM * 1000.0
        rows.append({"height_mm": h, "fname": fname, "raw": val,
                     "mean_v": ma_to_speed(ma), "above_zero": ma >= 4.0})
    rows.sort(key=lambda r: r["height_mm"])
    return rows


# ── load ──────────────────────────────────────────────────────────────────────

fw28  = load_panelposisjon_folder(
    "/Users/ole/Kodevik/wave_project/pressuredata/20251028-pitot-fullwind",
    unit="ms",
)
lw28  = load_panelposisjon_folder(
    "/Users/ole/Kodevik/wave_project/pressuredata/20251028-pitot-lowestwind",
    unit="ms",
    exclude_fnames=["pitot-lowestwind-panelposisjon_07.lvm"],  # "falskfil"
)
lw29  = load_oct29_lowestwind(
    "/Users/ole/Kodevik/wave_project/pressuredata/20251029-pitot-lowestwind",
)

# ── printout ──────────────────────────────────────────────────────────────────

def print_table(label, rows):
    print(f"\n{label}")
    print(f"  {'h_mm':>6}  {'raw':>10}  {'v [m/s]':>8}  note")
    print("  " + "-"*45)
    for r in rows:
        note = "near zero" if not r["above_zero"] else ""
        print(f"  {r['height_mm']:>6}  {r['raw']:>10.4f}  {r['mean_v']:>8.3f}  {note}")

print_table("Oct 28 fullwind (mA, post-calibration)", fw28)
print_table("Oct 28 lowestwind (mA, post-calibration)", lw28)
print_table("Oct 29 lowestwind (Volts/100Ω → mA, fixed pos 9960mm)", lw29)


# ── plot ──────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 7))
fig.suptitle(
    "28. oktober 2025 — sealing day, post-kalibrering (15:05)\nVerdier er pre-beregnet m/s fra LabVIEW-VI",
    fontsize=11
)
ax.set_xlabel("Vindfart [m/s]")
ax.set_ylabel("Høyde over vannet [mm]")
ax.set_ylim(0, ROOF_MM)
ax.axhline(ROOF_MM, color="brown", linewidth=1, linestyle="-", alpha=0.5, label="Tak")
ax.grid(True, linestyle="--", linewidth=0.5)

DATASETS = [
    (fw28,  "Full vind 28. okt",    "tab:red",   "o"),
    (lw28,  "Laveste vind 28. okt", "tab:green", "s"),
]

for rows, label, color, marker in DATASETS:
    if not rows:
        continue
    valid   = [r for r in rows if r["above_zero"]]
    invalid = [r for r in rows if not r["above_zero"]]
    if valid:
        z = np.array([r["height_mm"] for r in valid])
        v = np.array([r["mean_v"]    for r in valid])
        ax.plot(v, z, marker=marker, linestyle="-", color=color,
                markersize=6, linewidth=1.5, label=label)
    if invalid:
        z_i = np.array([r["height_mm"] for r in invalid])
        v_i = np.array([r["mean_v"]    for r in invalid])
        ax.scatter(v_i, z_i, marker="x", s=50, color=color, alpha=0.5, zorder=4)

ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
ax.set_xlim(left=0)
ax.legend(fontsize=9)

# note about Oct 29
fig.text(0.5, 0.01,
         "Okt 29-data utelatt — ukjent enhet (0.37–0.44), mulig Volt fra annen DAQ-konfig",
         ha="center", fontsize=8, style="italic")

fig.tight_layout(rect=[0, 0.04, 1, 1])
ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
if SAVE:
    out = SAVE_DIR / f"oct28_sealing_day_{ts_str}.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"\nSaved: {out}")
plt.show()
# %%
