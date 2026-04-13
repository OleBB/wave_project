# %%
"""
NOTE: The Oct 28 sealing-day data (panelposisjon_00–14) is handled separately
in windscripts/oct28_sealing_day.py — those files capture the calibration
and sealing transition in real time, not a height profile.
"""

# %%
"""
Exploration of early pitot .lvm datasets (Oct 2025).

Height convention (from Oct 17 readme):
  _01fem  → 15 mm above water
  _02     → 20 mm  (i.e. file number × 10 mm)
  _03     → 30 mm
  ...
  _30     → 300 mm

Formula used in Oct 2025 (as found in pressure_pro.py):
  v = sqrt((mA - 3.8) * 18.51)

  The 3.8 mA zero is CORRECT for this sensor — it was an old, previously uncalibrated
  unit whose physical zero had drifted to 3.8 mA (vs the standard 4.0 mA of the Nov sensor).
  The error is only in the pressure range: 150 Pa was assumed but the sensor is 100 Pa range.
  Correction: multiply all Oct speeds by 1/sqrt(1.5) ≈ 0.816.

Corrected Oct formula (right zero, right range):
  v = sqrt(2 * (mA - 3.8) / 16 * 100 / 1.225)

Nov formula (different sensor, calibrated to 4.0 mA):
  v = sqrt(2 * (mA - 4.0) / 16 * 100 / 1.225)

Oct 17 files: values in Amps → multiply by 1000 to get mA.
Oct 24–25 files: values already in mA.
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from io import StringIO

AIR_DENSITY = 1.225

BASE = Path("/Users/ole/Kodevik/wave_project/pressuredata")
SAVE_DIR = Path("/Users/ole/Kodevik/wave_project/windresults")
SAVE_DIR.mkdir(exist_ok=True)
SAVE = True


# ── helpers ──────────────────────────────────────────────────────────────────

def read_lvm(filepath):
    """
    Read a LabVIEW .lvm file. Returns numpy array of the data column (mA or A).
    Returns None if the file has no X_Value header (broken/empty).
    """
    with open(filepath, encoding="latin-1") as f:
        lines = f.readlines()

    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("X_Value"):
            start = i + 1
            break
    if start is None:
        return None

    values = []
    for line in lines[start:]:
        parts = line.strip().split("\t")
        # data column is index 1 (index 0 is X/time, often empty)
        col = parts[1] if len(parts) > 1 else parts[0]
        col = col.strip().replace('"', '').replace(",", ".")
        try:
            values.append(float(col))
        except ValueError:
            pass
    return np.array(values) if values else None


def height_from_filename(fname):
    """
    Map filename to height_mm using Oct 2025 convention:
      *_01fem* → 15 mm
      *_NN*    → NN * 10 mm  (NN = 02..30)
    Returns None if pattern not matched.
    """
    name = Path(fname).stem
    if "01fem" in name:
        return 15
    m = re.search(r"_(\d{2})$", name)
    if m:
        return int(m.group(1)) * 10
    return None


def ma_to_speed_oct_raw(ma):
    """Oct formula as written in pressure_pro.py: 150 Pa range, ρ≈1.0, zero=3.8 mA."""
    return np.sqrt(np.maximum((ma - 3.8) * 18.51, 0.0))


def ma_to_speed_oct(ma):
    """Oct corrected: right zero (3.8 mA, sensor-specific), right range (100 Pa), ρ=1.225."""
    return np.sqrt(np.maximum(2.0 * (ma - 3.8) / 16.0 * 100.0 / AIR_DENSITY, 0.0))


def ma_to_speed_nov(ma):
    """Nov 2025+ formula: 100 Pa range, ρ=1.225, zero=4.0 mA (calibrated sensor)."""
    return np.sqrt(np.maximum(2.0 * (ma - 4.0) / 16.0 * 100.0 / AIR_DENSITY, 0.0))


def load_profile_folder(folder, unit="mA", skip_below_zero=True):
    """
    Load all .lvm files from a folder, assign heights, return sorted list of
    {"height_mm", "mean_mA", "std_mA", "n_samples", "mean_speed_oct", "mean_speed_nov"} dicts.

    unit: "mA" (Oct 24+) or "A" (Oct 17, values in Amps → ×1000).
    """
    results = []
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith(".lvm"):
            continue
        h = height_from_filename(fname)
        if h is None:
            continue
        data = read_lvm(os.path.join(folder, fname))
        if data is None or data.size == 0:
            continue
        if unit == "A":
            data = data * 1000.0   # Amps → mA
        if skip_below_zero:
            data = data[data >= 4.0]
        if data.size == 0:
            print(f"  [{fname}] all values below sensor zero — skipped")
            continue
        mean_ma = float(np.mean(data))
        std_ma  = float(np.std(data, ddof=1)) if data.size > 1 else 0.0
        results.append({
            "height_mm":      h,
            "fname":          fname,
            "n_samples":      data.size,
            "mean_mA":        mean_ma,
            "std_mA":         std_ma,
            "mean_speed_oct": float(ma_to_speed_oct(mean_ma)),
            "mean_speed_nov": float(ma_to_speed_nov(mean_ma)),
        })
    results.sort(key=lambda x: x["height_mm"])
    return results


# ── load datasets ─────────────────────────────────────────────────────────────

print("Loading Oct 17 fullwind (100 Hz raw, values in Amps)...")
oct17_fw = load_profile_folder(
    BASE / "20251017-pitot-fullwind", unit="A"
)

print("Loading Oct 24 fullwind (single means, mA)...")
oct24_fw = load_profile_folder(
    BASE / "20251024-pitot-mean-fullwind", unit="mA"
)

print("Loading Oct 24 nowind (noise floor at 15 mm)...")
oct24_nw = load_profile_folder(
    BASE / "20251024-pitot-mean-nowind", unit="mA"
)

print("Loading Oct 25 lowestwind (single means, mA)...")
oct25_lw = load_profile_folder(
    BASE / "20251025-pitot-mean-lowestwind", unit="mA"
)

# ── printout ──────────────────────────────────────────────────────────────────

def print_profile(label, results):
    print(f"\n{'='*60}")
    print(f"  {label}  ({len(results)} heights)")
    print(f"{'='*60}")
    print(f"  {'z_mm':>6}  {'n':>6}  {'mean_mA':>8}  {'std_mA':>7}  {'v_oct':>6}  {'v_nov':>6}")
    for r in results:
        print(f"  {r['height_mm']:>6}  {r['n_samples']:>6}  "
              f"{r['mean_mA']:>8.3f}  {r['std_mA']:>7.4f}  "
              f"{r['mean_speed_oct']:>6.3f}  {r['mean_speed_nov']:>6.3f}")
    if results:
        speeds_oct = [r["mean_speed_oct"] for r in results]
        speeds_nov = [r["mean_speed_nov"] for r in results]
        print(f"  Range oct: {min(speeds_oct):.2f}–{max(speeds_oct):.2f} m/s")
        print(f"  Range nov: {min(speeds_nov):.2f}–{max(speeds_nov):.2f} m/s")

print_profile("Oct 17 — fullwind (100 Hz raw)", oct17_fw)
print_profile("Oct 24 — fullwind (single means)", oct24_fw)
print_profile("Oct 24 — nowind (noise floor, 15 mm)", oct24_nw)
print_profile("Oct 25 — lowestwind (single means)", oct25_lw)


# ── plot ──────────────────────────────────────────────────────────────────────

from datetime import datetime
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

ROOF_MM = 380

def _z(results):
    return np.array([r["height_mm"] for r in results])

def _v_oct(results):
    return np.array([r["mean_speed_oct"] for r in results])

def _v_nov(results):
    return np.array([r["mean_speed_nov"] for r in results])


# --- Figure 1: linear y-axis, both speed formulas shown ---
fig1, axes = plt.subplots(1, 3, figsize=(15, 7), sharey=True)
fig1.suptitle("Oct 2025 .lvm profiles — linear height", fontsize=12)

datasets = [
    (oct17_fw, "Oct 17 fullwind (100 Hz)", "tab:red",    "o", "-"),
    (oct24_fw, "Oct 24 fullwind (means)",  "tab:orange", "s", "-"),
    (oct25_lw, "Oct 25 lowestwind (means)","tab:green",  "^", "-"),
]

for ax, (title, formula) in zip(axes, [
    ("Oct korrigert (100 Pa, zero=3.8)", "oct"),
    ("Nov formel (100 Pa, zero=4.0)",    "nov"),
    ("Sammenligning",                    "both"),
]):
    ax.set_title(title, fontsize=9)
    ax.set_ylabel("Høyde over vannet [mm]")
    ax.set_xlabel("Vindfart [m/s]")
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.set_ylim(0, ROOF_MM)
    ax.axhline(ROOF_MM, color="brown", linewidth=1, linestyle="-", alpha=0.6, label="Tak")

    for results, label, color, marker, ls in datasets:
        if not results:
            continue
        z = _z(results)
        if formula == "oct":
            v = _v_oct(results)
            ax.plot(v, z, marker=marker, linestyle=ls, color=color, label=label, markersize=5)
        elif formula == "nov":
            v = _v_nov(results)
            ax.plot(v, z, marker=marker, linestyle=ls, color=color, label=label, markersize=5)
        else:
            v_o = _v_oct(results)
            v_n = _v_nov(results)
            ax.plot(v_o, z, marker=marker, linestyle="--", color=color, alpha=0.5,
                    label=f"{label} (oct)", markersize=4)
            ax.plot(v_n, z, marker=marker, linestyle="-", color=color,
                    label=f"{label} (nov)", markersize=5)

    ax.set_xlim(left=0)
    ax.legend(fontsize=7)

fig1.tight_layout()
if SAVE:
    out = SAVE_DIR / f"lvm_profiles_linear_{ts}.pdf"
    fig1.savefig(out, bbox_inches="tight")
    print(f"\nSaved: {out}")
plt.show()


# --- Figure 2: log height, Nov formula, compared with Nov mAstats style ---
fig2, ax2 = plt.subplots(figsize=(7, 7))
fig2.suptitle("Oct 2025 .lvm profiles — log høyde (nov-formel)", fontsize=12)

for results, label, color, marker, ls in datasets:
    if not results:
        continue
    z = _z(results) / 1000.0   # mm → m
    v = _v_nov(results)
    ax2.plot(v, z, marker=marker, linestyle=ls, color=color, label=label,
             markersize=5, linewidth=1.5)

# nowind: show as scatter at 15 mm
if oct24_nw:
    z_nw = np.array([r["height_mm"] for r in oct24_nw]) / 1000.0
    v_nw = np.array([r["mean_speed_nov"] for r in oct24_nw])
    ax2.scatter(v_nw, z_nw, marker="x", color="black", s=60, zorder=5,
                label=f"Oct 24 nowind @ 15 mm (n={len(oct24_nw)})")

ax2.set_xlabel("Vindfart [m/s]")
ax2.set_ylabel("Høyde over vannet [m]")
ax2.set_title("Log høyde — nov-formel")
ax2.set_yscale("log")
ax2.set_ylim(0.001, 1.0)
ax2.set_xlim(left=0)
ax2.grid(True, which="major", linestyle="--", linewidth=0.5)
ax2.grid(True, which="minor", linestyle="--", linewidth=0.3, alpha=0.4)
ax2.axhline(ROOF_MM / 1000, color="brown", linewidth=1, linestyle="-", alpha=0.6,
            label=f"Tak ({ROOF_MM} mm)")
ax2.legend(fontsize=8)
fig2.tight_layout()
if SAVE:
    out = SAVE_DIR / f"lvm_profiles_log_{ts}.pdf"
    fig2.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
plt.show()


# --- Figure 3: Oct 17 time-series variability — std per height ---
if oct17_fw:
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(10, 6), sharey=True)
    fig3.suptitle("Oct 17 fullwind — 100 Hz time-series statistics per height", fontsize=11)

    z   = _z(oct17_fw)
    v   = _v_oct(oct17_fw)
    std = np.array([ma_to_speed_oct(r["mean_mA"] + r["std_mA"]) -
                    ma_to_speed_oct(r["mean_mA"])   # linearised std in m/s
                    for r in oct17_fw])
    n   = np.array([r["n_samples"] for r in oct17_fw])

    ax3a.errorbar(v, z, xerr=std, fmt="o-", color="tab:red", capsize=4,
                  markersize=5, label="mean ± std (100 Hz, 10 s)")
    ax3a.set_xlabel("Vindfart [m/s] (oct formula)")
    ax3a.set_ylabel("Høyde over vannet [mm]")
    ax3a.set_title("Profil med std")
    ax3a.set_xlim(left=0)
    ax3a.set_ylim(0, ROOF_MM)
    ax3a.axhline(ROOF_MM, color="brown", linewidth=1, alpha=0.6)
    ax3a.grid(True, linestyle="--", linewidth=0.5)
    ax3a.legend(fontsize=8)

    ti = std / np.where(v > 0.1, v, np.nan) * 100
    ax3b.plot(ti, z, "o-", color="tab:red", markersize=5, label="TI = std/mean [%]")
    ax3b.set_xlabel("TI [%]")
    ax3b.set_title("Turbulensintensitet (grov, fra 10 s std)")
    ax3b.grid(True, linestyle="--", linewidth=0.5)
    ax3b.axhline(ROOF_MM, color="brown", linewidth=1, alpha=0.6)
    ax3b.legend(fontsize=8)

    fig3.tight_layout()
    if SAVE:
        out = SAVE_DIR / f"lvm_oct17_variability_{ts}.pdf"
        fig3.savefig(out, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.show()

# %%
