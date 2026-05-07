"""
Surgical in-place patch — recompute ka/Ursell columns from FFT amplitude.
========================================================================

Companion to the 2026-05-07 pipeline fix at
[`wavescripts/processor.py`](../wavescripts/processor.py) lines 1163 + 1215.
Avoids the full ~20 min `python main.py --force-recompute` by patching the
already-cached `meta.json` files in place. Only touches the columns whose
values change with the fix:

  - `Probe {pos} ka (FFT)`            (per-probe; uses `Wavenumber (FFT)` × `Amplitude (FFT)` / 1000)
  - `Probe {pos} Ursell (FFT)`        (per-probe; 2a × λ² / d³)
  - `IN ka (FFT)`     / `OUT ka (FFT)`
  - `IN Ursell (FFT)` / `OUT Ursell (FFT)`
        (canonical means across same-distance probes — same averaging
         logic as processor2nd.py)
  - `Expected ka`                     (global; uses input `Wavenumber` × IN `Amplitude (FFT)` / 1000)
  - `Ursell` (no suffix)              (global; 2 × IN a × `Expected Wavelength`² / d³)

Everything else in `meta.json` is left untouched. K_t (`OUT/IN (FFT)`) is
already FFT/FFT and unchanged. Pre-existing `_disagree_frac` columns also
unchanged (those use `Amplitude (FFT)`).

Usage:
    conda run -n draumkvedet python analysis_scratch/repair_ka_in_meta.py
        # dry-run by default; edit DRY_RUN at top to apply.

A timestamped backup of every patched `meta.json` is written next to the
file as `meta.json.bak.<timestamp>` before any change.
"""

import sys
import json
import shutil
from datetime import datetime as _dt
from pathlib import Path
import glob

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from wavescripts.improved_data_loader import get_configuration_for_date

# ── 1. Toggles ────────────────────────────────────────────────────────────────
DRY_RUN = False  # set False to actually overwrite meta.json files
MM_TO_M = 1e-3


def _probes_at_same_distance(cfg, ref_probe_idx: int) -> list[str]:
    """Mirror of wavescripts.processor2nd._probes_at_same_distance.

    Returns position strings (e.g. "9373/170") of all probes whose
    longitudinal distance matches `ref_probe_idx`'s longitudinal distance.
    """
    col_names = cfg.probe_col_names()
    ref_pos   = col_names[ref_probe_idx]
    ref_dist  = ref_pos.split("/")[0]
    return [pos for pos in col_names.values()
            if pos.split("/")[0] == ref_dist]


def _patch_row(row: dict, cfg) -> tuple[dict, dict]:
    """Patch one meta row in place. Returns (row, change_log).

    change_log: dict of {col: (old, new)} for every column actually modified.
    """
    log: dict = {}
    col_names = cfg.probe_col_names()
    all_positions = list(col_names.values())

    # Water depth in metres (column "WaterDepth [mm]" stores mm).
    H_mm = row.get("WaterDepth [mm]", None)
    H_m  = float(H_mm) * MM_TO_M if H_mm not in (None, "") else None

    # ── A. Per-probe ka and Ursell ────────────────────────────────────────
    for pos in all_positions:
        a_col      = f"Probe {pos} Amplitude (FFT)"
        k_col      = f"Probe {pos} Wavenumber (FFT)"
        ka_col     = f"Probe {pos} ka (FFT)"
        lam_col    = f"Probe {pos} Wavelength (FFT)"
        ursell_col = f"Probe {pos} Ursell (FFT)"

        a_fft = row.get(a_col)
        k_val = row.get(k_col)
        if a_fft in (None, "") or k_val in (None, ""):
            continue
        try:
            a_m = float(a_fft) * MM_TO_M
            k   = float(k_val)
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(a_m) and np.isfinite(k) and k > 0):
            continue

        new_ka = a_m * k
        old_ka = row.get(ka_col)
        if old_ka is None or not np.isclose(float(old_ka), new_ka,
                                             rtol=1e-9, atol=1e-12,
                                             equal_nan=True):
            log[ka_col] = (old_ka, new_ka)
        row[ka_col] = new_ka

        # Ursell = 2a × λ² / d³.
        lam   = row.get(lam_col)
        if lam in (None, "") or H_m is None or H_m <= 0:
            continue
        try:
            lam_m = float(lam)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(lam_m):
            continue
        new_ursell = (2.0 * a_m * lam_m ** 2) / (H_m ** 3)
        old_ursell = row.get(ursell_col)
        if old_ursell is None or not np.isclose(float(old_ursell), new_ursell,
                                                  rtol=1e-9, atol=1e-12,
                                                  equal_nan=True):
            log[ursell_col] = (old_ursell, new_ursell)
        row[ursell_col] = new_ursell

    # ── B. Canonical IN/OUT means (mirror processor2nd.py averaging) ──────
    in_probe  = row.get("in_probe")
    out_probe = row.get("out_probe")
    if in_probe is not None and out_probe is not None:
        try:
            in_p, out_p = int(in_probe), int(out_probe)
        except (TypeError, ValueError):
            in_p = out_p = None
        if in_p is not None and out_p is not None:
            in_positions  = _probes_at_same_distance(cfg, in_p)
            out_positions = _probes_at_same_distance(cfg, out_p)

            for suffix, target in [("ka (FFT)",     ("IN ka (FFT)", "OUT ka (FFT)")),
                                    ("Ursell (FFT)", ("IN Ursell (FFT)", "OUT Ursell (FFT)"))]:
                for positions, tgt_col in zip([in_positions, out_positions], target):
                    vals = []
                    for p in positions:
                        v = row.get(f"Probe {p} {suffix}")
                        if v is None or v == "":
                            continue
                        try:
                            f = float(v)
                        except (TypeError, ValueError):
                            continue
                        if np.isfinite(f):
                            vals.append(f)
                    new_v = float(np.mean(vals)) if vals else float("nan")
                    old_v = row.get(tgt_col)
                    if (old_v is None
                            or (isinstance(old_v, float) and np.isnan(old_v) and not np.isnan(new_v))
                            or (isinstance(old_v, (int, float)) and not np.isnan(new_v)
                                and not np.isclose(float(old_v), new_v,
                                                    rtol=1e-9, atol=1e-12))):
                        log[tgt_col] = (old_v, new_v)
                    row[tgt_col] = new_v

    # ── C. Global "Expected ka" + "Ursell" (mirror processor.py:1211-1218) ─
    # Global "Expected ka" and "Expected Ursell" use the input-frequency-derived
    # wavenumber (GC.WAVENUMBER → "Expected Wavenumber") and the IN reference
    # probe's FFT amplitude.
    in_pos_ref = row.get("in_position")
    in_a_fft   = row.get(f"Probe {in_pos_ref} Amplitude (FFT)") if in_pos_ref else None
    k_global   = row.get("Expected Wavenumber")
    lam_global = row.get("Expected Wavelength")
    if (in_a_fft not in (None, "")
            and k_global not in (None, "")):
        try:
            a_m = float(in_a_fft) * MM_TO_M
            k_g = float(k_global)
        except (TypeError, ValueError):
            a_m = k_g = None
        if a_m is not None and k_g is not None and np.isfinite(a_m) and np.isfinite(k_g):
            new_eka = a_m * k_g
            old_eka = row.get("Expected ka")
            if old_eka is None or not np.isclose(float(old_eka), new_eka,
                                                   rtol=1e-9, atol=1e-12,
                                                   equal_nan=True):
                log["Expected ka"] = (old_eka, new_eka)
            row["Expected ka"] = new_eka

            if lam_global not in (None, "") and H_m not in (None, 0) and H_m > 0:
                try:
                    lam_g = float(lam_global)
                except (TypeError, ValueError):
                    lam_g = None
                if lam_g is not None and np.isfinite(lam_g):
                    new_urs = (2.0 * a_m * lam_g ** 2) / (H_m ** 3)
                    # Global Ursell is named "Expected Ursell" in meta.json
                    # (no plain "Ursell" column exists at the global level).
                    old_urs = row.get("Expected Ursell")
                    if old_urs is None or not np.isclose(float(old_urs), new_urs,
                                                          rtol=1e-9, atol=1e-12,
                                                          equal_nan=True):
                        log["Expected Ursell"] = (old_urs, new_urs)
                    row["Expected Ursell"] = new_urs

    return row, log


def _patch_folder(folder: Path) -> dict:
    """Patch every row in folder/meta.json. Returns {col: change_count}."""
    meta_path = folder / "meta.json"
    if not meta_path.exists():
        return {}
    with meta_path.open() as f:
        meta = json.load(f)
    if not isinstance(meta, list) or not meta:
        return {}

    # Date determines probe configuration.
    file_date = _dt.fromisoformat(str(meta[0].get("file_date")))
    cfg = get_configuration_for_date(file_date)

    col_changes: dict = {}
    for row in meta:
        _, log = _patch_row(row, cfg)
        for col, (_old, _new) in log.items():
            col_changes[col] = col_changes.get(col, 0) + 1

    if not DRY_RUN and col_changes:
        backup = meta_path.with_suffix(
            f".json.bak.{_dt.now().strftime('%Y%m%dT%H%M%S')}"
        )
        shutil.copy2(meta_path, backup)
        with meta_path.open("w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False, allow_nan=True)

    return col_changes


def main():
    print(f"Repair mode: {'DRY-RUN' if DRY_RUN else 'WRITE'}")
    folders = sorted(BASE.glob("waveprocessed/PROCESSED-*"))
    print(f"  {len(folders)} folder(s) found\n")

    grand: dict = {}
    for folder in folders:
        changes = _patch_folder(folder)
        n_total = sum(changes.values())
        if n_total == 0:
            print(f"  ✓ {folder.name}: no changes (already FFT-correct)")
            continue
        print(f"  ⟳ {folder.name}: {n_total} cells across {len(changes)} columns")
        for col, n in sorted(changes.items()):
            print(f"      {col:38s} : {n:4d} cells")
            grand[col] = grand.get(col, 0) + n

    print("\n=== TOTALS ===")
    if not grand:
        print("  no changes anywhere")
    else:
        total_cells = sum(grand.values())
        print(f"  {total_cells} cells across {len(grand)} columns:")
        for col, n in sorted(grand.items()):
            print(f"    {col:38s} : {n:6d} cells")
    print(f"\nMode: {'DRY-RUN — set DRY_RUN=False to apply' if DRY_RUN else 'WRITE — meta.json files updated; .bak.* timestamped backups beside each'}")


if __name__ == "__main__":
    main()
