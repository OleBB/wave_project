#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 17:18:03 2025

@author: ole
"""
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

from typing import Dict, List, Tuple

from wavescripts.improved_data_loader import update_processed_metadata, get_configuration_for_date
from wavescripts.wave_detection import find_wave_range
from wavescripts.signal_processing import compute_psd_with_amplitudes, compute_fft_with_amplitudes, compute_amplitudes, compute_nowave_psd
from wavescripts.wave_physics import calculate_wavenumbers_vectorized, calculate_wavedimensions, calculate_windspeed

from scipy.interpolate import PchipInterpolator
from wavescripts.constants import SIGNAL, RAMP, MEASUREMENT, CLIP, STILLWATER, STILLWATER_EXCLUDE, get_smoothing_window, PROBE_RANGE_MODES, PROBE_HEIGHT_DEFAULT_MM
from wavescripts.constants import (
    ProbeColumns as PC,
    GlobalColumns as GC,
    ColumnGroups as CG,
    CalculationResultColumns as RC
)


# ========================================================== #
# === Make sure stillwater levels are computed and valid === #
# ========================================================== #
def _probe_median_from_run(df: pd.DataFrame, probe_col: str, n_samples: int | None = None) -> float | None:
    """Return median of a probe column from a single run's dataframe.

    Args:
        n_samples: if given, only the first n_samples rows are used (e.g. first 1 second).
    """
    if probe_col not in df.columns:
        return None
    src = df[probe_col].iloc[:n_samples] if n_samples is not None else df[probe_col]
    vals = pd.to_numeric(src, errors='coerce').dropna()
    return float(vals.median()) if len(vals) > 0 else None


def _probe_noise_amplitude_from_run(
    df: pd.DataFrame,
    probe_col: str,
    n_samples: int | None = None,
    start_sample: int = 0,
) -> float | None:
    """Return (P97.5 - P2.5) / 2 of a probe column — the pipeline amplitude definition.

    This is the noise floor for that probe in that run: the smallest wave amplitude
    indistinguishable from background fluctuations.  Consistent with how
    'Probe {pos} Amplitude' is computed everywhere else in the pipeline.

    Args:
        start_sample: First sample to include.  Used to skip the initial settling
                      period of a run that started too soon after a wave/wind run,
                      or simply to focus on the calmer latter portion of the run.
    """
    if probe_col not in df.columns:
        return None
    src = df[probe_col].iloc[start_sample:n_samples]
    vals = pd.to_numeric(src, errors='coerce').dropna().values
    if len(vals) < 2:
        return None
    return float((np.nanpercentile(vals, 97.5) - np.nanpercentile(vals, 2.5)) / 2.0)


def ensure_stillwater_columns(
    dfs: dict[str, pd.DataFrame],
    meta: pd.DataFrame,
    cfg,
) -> pd.DataFrame:
    """
    Computes per-run stillwater levels using each run's own pre-wave signal.

    Strategy (per run, per probe):
      - Nowave runs:  full-run median  (no wave arriving, all samples valid)
      - Wave runs:    mean of first PRE_WAVE_S seconds  (before wave front arrives)

    This is self-calibrating: each run references its own water level, so
    wind-setup shifts (which vary run-to-run and take ~10 min to equalise
    after wind-off) are automatically captured without any cross-run fitting.

    Diagnostic prints compare wave-run pre-wave means against nowave means
    grouped by wind condition — use these to validate that PRE_WAVE_S is
    short enough to avoid wave contamination.

    Safe to call multiple times — skips if all Stillwater columns are already
    populated with per-run (varying) values.
    """
    _PRE_WAVE_N = int(STILLWATER.PRE_WAVE_S * MEASUREMENT.SAMPLING_RATE)

    if not dfs:
        raise ValueError(
            "processed_dfs is empty — reload with load_processed=True before calling ensure_stillwater_columns."
        )

    # When called with combined_meta (all dates), restrict to rows that belong
    # to this cfg's date range so probe column names match.
    meta = meta.copy()
    # format='ISO8601' tolerates both "2025-11-12T15:36:14" and
    # "2026-02-05T10:52:00.195183" (with microseconds) — the latter appears
    # when metadata extraction falls back to file-mtime for e.g. .stats.csv
    # files accidentally caught by the *.csv glob.
    meta_t = pd.to_datetime(meta["file_date"], format='ISO8601')
    in_range = meta_t >= pd.Timestamp(cfg.valid_from)
    if cfg.valid_until is not None:
        in_range &= meta_t < pd.Timestamp(cfg.valid_until)
    meta = meta[in_range].reset_index(drop=True)
    if meta.empty:
        raise ValueError(f"No rows in meta match cfg '{cfg.name}' date range.")

    col_names = cfg.probe_col_names()
    probe_cols = [f"Stillwater Probe {pos}" for pos in col_names.values()]

    # Skip if already computed with per-run (varying) values
    if all(col in meta.columns for col in probe_cols):
        if meta[probe_cols].notna().all().all():
            if meta[probe_cols[0]].std() > 0.001:
                print("Stillwater levels already computed per-run → skipping")
                return meta

    print(f"Computing per-run stillwater levels "
          f"(nowave: full-run median; wave: first {STILLWATER.PRE_WAVE_S:.1f} s mean)...")

    nowave_mask = (
        meta[GC.WAVE_FREQUENCY_INPUT].isna()
        | meta["path"].astype(str).str.lower().str.contains("nowave")
    )
    wind_str = meta[GC.WIND_CONDITION].astype(str).str.strip().str.lower()
    nowind_mask = wind_str.isin(["no", "", "nan", "none"])

    new_cols = {}    # probe col → per-row values
    noise_updates = {}

    for i, pos in col_names.items():
        probe_col = f"Probe {pos}"
        per_run_values = {}    # path → stillwater value
        nowave_nowind_vals = []  # for noise floor

        for _, row in meta.iterrows():
            path = row["path"]
            fname = Path(path).name

            if any(kw in fname for kw in STILLWATER_EXCLUDE):
                print(f"  ⚠ Skipping excluded run: {fname}")
                per_run_values[path] = np.nan
                continue

            df_run = dfs.get(path)
            if df_run is None:
                per_run_values[path] = np.nan
                continue

            is_nowave = nowave_mask.loc[row.name]
            # nowave: full run median; wave: first PRE_WAVE_S seconds mean
            n_samp = None if is_nowave else _PRE_WAVE_N
            v = _probe_median_from_run(df_run, probe_col, n_samp)
            per_run_values[path] = v if v is not None else np.nan

        stillwater_series = meta["path"].map(per_run_values)

        # ── fallback for NaN stillwater (e.g. probe malfunction) ─────────────
        # Use session median of valid values so downstream doesn't crash.
        # The probe's signal is unusable anyway — this only prevents a hard error.
        nan_paths = [p for p, v in per_run_values.items() if np.isnan(v)]
        if nan_paths:
            valid_vals = [v for v in per_run_values.values() if not np.isnan(v)]
            fallback = float(np.median(valid_vals)) if valid_vals else 0.0
            for p in nan_paths:
                per_run_values[p] = fallback
                print(f"  ⚠ Probe {pos}: NaN stillwater for {Path(p).name} "
                      f"— using session median {fallback:.3f} mm as fallback")
            stillwater_series = meta["path"].map(per_run_values)

        new_cols[f"Stillwater Probe {pos}"] = stillwater_series.values

        # ── diagnostics: compare pre-wave means vs nowave reference ──────────
        # Collect nowave+nowind mean (the stable reference)
        nowave_nowind_rows = meta[nowave_mask & nowind_mask]
        ref_vals = []
        for _, row in nowave_nowind_rows.iterrows():
            df_run = dfs.get(row["path"])
            if df_run is None:
                continue
            v = _probe_median_from_run(df_run, probe_col, None)
            if v is not None:
                ref_vals.append(v)
                nowave_nowind_vals.append(v)

        ref_mean = float(np.mean(ref_vals)) if ref_vals else None

        # Print per-run values grouped by nowave vs wave
        for _, row in meta.iterrows():
            path = row["path"]
            fname = Path(path).name
            v = per_run_values.get(path, np.nan)
            if np.isnan(v):
                continue
            is_nowave = nowave_mask.loc[row.name]
            tag = "[nowave]" if is_nowave else f"[first {STILLWATER.PRE_WAVE_S:.1f}s]"
            delta = f"  Δ={v - ref_mean:+.2f} mm vs nowave ref" if ref_mean is not None else ""
            print(f"  Probe {pos}  {fname}: {v:.3f} mm  {tag}{delta}")

        # Noise std for nowave+nowind runs (probe noise floor)
        for _, row in nowave_nowind_rows.iterrows():
            df_run = dfs.get(row["path"])
            if df_run is None:
                continue
            vals = pd.to_numeric(df_run.get(probe_col, pd.Series()), errors='coerce').dropna()
            if len(vals) > 0:
                noise_updates.setdefault(row["path"], {})[f"Probe {pos} Stillwater Std"] = float(vals.std())

    meta = meta.assign(**new_cols)

    # Write noise std into metadata (nowave+nowind rows only)
    std_col_names = list(next(iter(noise_updates.values()), {}).keys())
    new_std_cols = {c: np.nan for c in std_col_names if c not in meta.columns}
    if new_std_cols:
        meta = meta.assign(**new_std_cols)
    for path, std_dict in noise_updates.items():
        idx = meta.index[meta["path"] == path]
        for col_name, val in std_dict.items():
            meta.loc[idx, col_name] = val

    # ── Per-hardware-config noise floor ──────────────────────────────────────
    #
    # Probe noise depends on three independent hardware factors, ALL of which can
    # change between experiment folders.  A single scalar broadcast blends configs
    # with very different noise levels and gives wrong thresholds for both ends.
    #
    # Factor 1 — physical probe unit identity (probe number 1–4 in the config).
    #   Each probe box has different internal electronics and calibration history.
    #   Swapping probes between positions (e.g. March 2026 rearrangement) changes
    #   the noise at each position even if height and range-mode are unchanged.
    #   Probe numbers ARE the hardware IDs — they do not follow position strings.
    #
    # Factor 2 — probe_height_mm (distance from sensor face to still-water surface).
    #   Longer path → more air → weaker ultrasonic return → higher noise.
    #   Old default h272 (pre-2026-03-23, no keyword in folder name) had ~2–3×
    #   higher noise than h100 (post-2026-03-23).  Folders without a height keyword
    #   are assigned PROBE_HEIGHT_DEFAULT_MM = 272 by the data loader.
    #
    # Factor 3 — probe_range_mode ("high" / "low", hardware switch on probe box).
    #   "high" range (130–350 mm window) has coarser quantisation.
    #   "low"  range (30–150 mm window) has finer quantisation and lower noise.
    #   High-range is required at h272 (sensor face is already outside the low-range
    #   window); low-range is preferred — and required — at h100.
    #
    # Second-order factors tracked in combined_meta but NOT used here:
    #   - WaterDepth [mm]  : changes the water-surface reflection geometry slightly.
    #   - Room temperature : affects ultrasonic speed-of-sound → mm/sample conversion.
    #     (TODO: plot speed-of-sound vs temperature across all folders to quantify
    #      this systematic.  Air at 20 °C ≈ 343 m/s; at 10 °C ≈ 337 m/s — ~1.7 %
    #      difference, translating to ~1.7 % amplitude scale error if uncorrected.
    #      Relevant especially when comparing November 2025 and March 2026 data.)
    #
    # Strategy:
    #   Accumulate stillwater amplitude measurements keyed by the tuple
    #   (probe_i, probe_height_mm, probe_range_mode).
    #   Each run then receives the mean noise of its OWN hardware config, not a
    #   blended average that mixes noisy h272/high with quiet h100/low data.
    #
    # Column written: "Probe {i} at {pos} Uncertainty"  — per-row float [mm]
    #   Value = mean (P97.5−P2.5)/2 across all nowave+nowind runs that share the
    #   same (probe_i, height_mm, range_mode) as this row.
    #   NaN if no matching stillwater data exists for that hardware config — this
    #   flags the row as having no trustworthy threshold rather than silently
    #   inheriting a wrong one from a different physical setup.
    #
    # Downstream use (filters.py damping_grouper):
    #   reliable = (Amplitude(FFT) >= k * Uncertainty)   (k ≈ 2)
    #   The check is now per-run and hardware-config-aware.

    def _hw_key_from_row(row_: pd.Series, probe_i_: int) -> tuple:
        """Return (probe_i, height_mm, range_mode) — a hashable hardware config key.

        Missing or NaN height falls back to PROBE_HEIGHT_DEFAULT_MM (272 mm) — the
        historical default before the probe-height keyword was introduced in folder
        names (2026-03-23).  Missing range_mode defaults to 'high'.
        """
        h = row_.get("probe_height_mm")
        if h is None or (isinstance(h, float) and np.isnan(h)):
            h = PROBE_HEIGHT_DEFAULT_MM
        else:
            h = int(float(h))
        r = str(row_.get("probe_range_mode") or "high").strip().lower()
        return (probe_i_, h, r)

    # ── Settling-time thresholds for noise floor quality gate ────────────────
    # Required inter_run_gap_s before a nowave+nowind run is trustworthy as a
    # stillwater noise floor reference.  Keyed by (prev_had_wind, sub1hz, long).
    #
    # Physics rationale (from tank observations and fromMaxToZeroWin runs):
    #
    #   Wind (any):
    #     Fan-off decay takes ~12 min (fromMaxToZeroWin characterisation).
    #     Wind generates broadband surface roughness that outlasts the fan.
    #
    #   Sub-1 Hz waves (long wavelength):
    #     Orbital motion reaches the tank floor at d=580 mm even at 0.65 Hz.
    #     Reflections from walls and panel geometry create a complicated standing
    #     field that decays slowly.  Per240 = long steady state = more energy.
    #
    #   High-freq waves (≥1 Hz):
    #     Shorter wavelength, orbitals confined near surface, faster viscous decay.
    #     per40 = short burst, little residual.  per240 = more, but still manageable.
    #
    #   Nowave preceding run:
    #     No wave energy added — water already at rest, no gap needed.
    #
    # Thresholds (seconds) — conservative but not excessive:
    _SETTLE_GAP_S: dict[tuple[bool, bool, bool], float] = {
        # (had_wind, sub_1hz,  long_run)  → gap_s
        (True,  True,  True):  720.0,   # wind + sub-1Hz long  → 12 min
        (True,  True,  False): 720.0,   # wind + sub-1Hz short → 12 min (wind dominates)
        (True,  False, True):  720.0,   # wind + high-f long   → 12 min
        (True,  False, False): 720.0,   # wind + high-f short  → 12 min
        (False, True,  True):  600.0,   # no-wind + sub-1Hz + per240 → 10 min (worst non-wind)
        (False, True,  False): 300.0,   # no-wind + sub-1Hz + per40  →  5 min
        (False, False, True):  300.0,   # no-wind + high-f  + per240 →  5 min
        (False, False, False): 120.0,   # no-wind + high-f  + per40  →  2 min
    }

    # ── Noise floor accumulation: use all nowave+nowind rows ────────────────
    # Rather than excluding under-settled runs outright, we compute a per-run
    # start_sample that skips the unsettled beginning.  Two rules apply:
    #
    #   Rule A — "latter-third" (always):
    #     The last two-thirds of any stillwater run are calmer than the first.
    #     The first third can still be recovering from the previous activity.
    #     → start_s ≥ run_duration_s / 3
    #
    #   Rule B — "settling deficit" (when gap < required):
    #     If the gap since the previous wave/wind run is shorter than the
    #     empirical threshold from _SETTLE_GAP_S, part of the run is still
    #     settling.  Skip the estimated unsettled prefix:
    #     → start_s ≥ (required_gap_s − actual_gap_s)
    #
    # The effective start_sample = max(Rule A, Rule B) * Fs.
    #
    # Runs where the remaining signal (after start_sample) is shorter than
    # MIN_REMAINING_S are skipped entirely — too little settled data.
    #
    # This means we NEVER discard a run just for being "too close" to a wave run;
    # we simply measure it from its calmer end.

    _fs = float(MEASUREMENT.SAMPLING_RATE)

    # Exclude wind-transition runs (fromMax*/fromZero*) even when WindCondition=="no":
    # during most of the run wind was active — the latter portion is not clean stillwater.
    _run_cat = meta.get("run_category", pd.Series("", index=meta.index)).astype(str).str.strip()
    _wind_decay_mask = _run_cat == "wind_decay"

    nowave_nowind_rows = meta[nowave_mask & nowind_mask & ~_wind_decay_mask]
    noise_floor_accum: dict[tuple, list[float]] = {}
    n_partial = 0   # runs where start_sample > 0
    n_too_short = 0  # runs skipped because remaining settled signal is too short

    for _, row in nowave_nowind_rows.iterrows():
        df_run = dfs.get(row["path"])
        if df_run is None:
            continue

        run_n = len(df_run)
        run_duration_s = run_n / _fs

        # Rule B: settling deficit from previous run
        prev_cat  = str(row.get("prev_run_category") or "").strip().lower()
        prev_wind = str(row.get("prev_run_wind")     or "").strip().lower()
        gap = row.get("inter_run_gap_s")
        gap = float(gap) if (
            gap is not None and not (isinstance(gap, float) and np.isnan(gap))
        ) else 0.0

        # Short-circuit only when the previous run was also nowave AND no-wind.
        # A nowave_control run with wind (e.g. ULSonly+full) still needs wind
        # settling time — check prev_wind explicitly.
        if prev_cat in {"", "nowave_control"} and prev_wind not in {"full", "lowest"}:
            required_s = 0.0
        else:
            _had_wind  = prev_wind in {"full", "lowest"}
            _prev_freq = row.get("prev_run_freq_hz")
            _sub_1hz   = (
                (_prev_freq is not None)
                and not (isinstance(_prev_freq, float) and np.isnan(_prev_freq))
                and float(_prev_freq) < 1.0
            )
            _prev_nper = row.get("prev_run_nperiods")
            _long_run  = (
                (_prev_nper is not None)
                and not (isinstance(_prev_nper, float) and np.isnan(_prev_nper))
                and float(_prev_nper) >= 120
            )
            required_s = _SETTLE_GAP_S.get((_had_wind, _sub_1hz, _long_run), 300.0)

        settling_deficit_s = max(0.0, required_s - gap)

        # Effective skip: whichever is larger — settling deficit or first-third
        start_s = max(settling_deficit_s, run_duration_s / 3.0)
        start_sample = int(start_s * _fs)

        # Minimum remaining signal: proportional to run length, floored at 15 s.
        # max(15, 40% of run) keeps short mstop30 runs usable (21 s remaining)
        # while still demanding a meaningful window on longer runs.
        min_remaining_s = max(15.0, run_duration_s * 0.4)
        if (run_n - start_sample) < int(min_remaining_s * _fs):
            n_too_short += 1
            continue
        if start_sample > 0:
            n_partial += 1

        for i, pos in col_names.items():
            v = _probe_noise_amplitude_from_run(df_run, f"Probe {pos}", None, start_sample)
            if v is not None:
                key = _hw_key_from_row(row, i)
                noise_floor_accum.setdefault(key, []).append(v)

    if n_partial > 0:
        print(f"  Noise floor: {n_partial} run(s) used with partial offset "
              f"(first-third skip or settling-deficit rule)")
    if n_too_short > 0:
        print(f"  Noise floor: {n_too_short} run(s) skipped — too short after offset")

    # Mean per hardware-config key — printed grouped for easy verification
    noise_floor_by_key: dict[tuple, float] = {}
    for key in sorted(noise_floor_accum):
        vals = noise_floor_accum[key]
        probe_i, height, rmode = key
        pos = col_names[probe_i]
        mean_val = float(np.mean(vals))
        noise_floor_by_key[key] = mean_val
        print(f"  Noise floor  Probe {probe_i} at {pos} [{height}mm/{rmode}]: "
              f"{mean_val:.4f} mm  (n={len(vals)}, range {min(vals):.4f}–{max(vals):.4f})")

    # Assign per-row Uncertainty: each row gets the mean for its own hardware config.
    # Rows whose (probe_i, height_mm, range_mode) combination has no matching
    # stillwater data receive NaN — no threshold is better than a wrong threshold.
    if noise_floor_by_key:
        unc_cols = {}
        for i, pos in col_names.items():
            col = f"Probe {i} at {pos} Uncertainty"
            unc_cols[col] = meta.apply(
                lambda row_, _i=i: noise_floor_by_key.get(_hw_key_from_row(row_, _i), np.nan),
                axis=1,
            ).values
        meta = meta.assign(**unc_cols)

    # Ensure PROCESSED_folder is set so meta can be saved
    if "PROCESSED_folder" not in meta.columns:
        if "experiment_folder" in meta.columns:
            meta["PROCESSED_folder"] = "PROCESSED-" + meta["experiment_folder"].iloc[0]
        else:
            raw_folder = Path(meta["path"].iloc[0]).parent.name
            meta["PROCESSED_folder"] = f"PROCESSED-{raw_folder}"

    update_processed_metadata(meta, force_recompute=False)
    print("\nStillwater levels saved to meta.json")

    return meta


def remove_outliers():
    #lag noe basert på steepness, kanskje tilogmed ak. Hvis ak er for bratt
    # og datapunktet for høyt, så må den markeres, og så fjernes.
    #se Karens script
    return

def _extract_stillwater_levels(meta_full: pd.DataFrame, cfg, debug: bool) -> dict:
    """Extract per-path stillwater levels from metadata.
    Returns {path: {probe_i: value}} with one entry per run.
    Probe column names in meta are position-based (e.g. 'Stillwater Probe 9373/170').
    """
    col_names = cfg.probe_col_names()  # {1: "9373/170", ...}
    stillwater = {}
    for _, row in meta_full.iterrows():
        path = row["path"]
        stillwater[path] = {}
        for i, pos in col_names.items():
            col = f"Stillwater Probe {pos}"
            val = row[col]
            if pd.isna(val):
                raise ValueError(f"{col} is NaN for {Path(path).name}!")
            stillwater[path][i] = float(val)
    if debug:
        first_path = meta_full["path"].iloc[0]
        for i, pos in col_names.items():
            print(f"  Stillwater Probe {pos} @ first run = {stillwater[first_path][i]:.3f} mm")
    return stillwater


def _remask_long_gaps(filled: np.ndarray, nan_mask: np.ndarray, max_gap: int) -> np.ndarray:
    """Re-apply NaN to runs that were longer than max_gap before interpolation."""
    run_start = None
    for j in range(len(nan_mask) + 1):
        if j < len(nan_mask) and nan_mask[j]:
            if run_start is None:
                run_start = j
        else:
            if run_start is not None:
                if j - run_start > max_gap:
                    filled[run_start:j] = np.nan
                run_start = None
    return filled


def _detect_stuck_segments(
    raw_signal: np.ndarray,
    fs: float,
    std_thresh: float = None,
    min_duration_s: float = None,
    level_thresh_mm: float = 5.0,
) -> list[tuple[int, int]]:
    """Detect flat/stuck segments in a raw probe signal (before zeroing).

    Two conditions must both be true for a segment to be flagged:
      1. Rolling std < std_thresh  (probe is flat)
      2. Segment mean differs from the run's overall median by > level_thresh_mm
         (probe is flat at the WRONG level — not just genuinely still water)

    Condition 2 prevents false positives from the pre-wave stillwater period in
    nowind runs, where water is legitimately very flat at the correct level.
    The 198 mm hardware fault satisfies both: flat AND offset by ~100 mm from
    the true water level.

    Returns list of (start_sample, end_sample) tuples (inclusive endpoints).
    """
    if std_thresh is None:
        std_thresh = CLIP.STUCK_STD_MM
    if min_duration_s is None:
        min_duration_s = CLIP.STUCK_MIN_S

    min_samples = int(min_duration_s * fs)
    win_samples = max(int(0.2 * fs), 10)  # 0.2 s rolling window (shorter than any wave period)

    vals = pd.to_numeric(pd.Series(raw_signal), errors='coerce')
    run_median = float(vals.median())
    rolling_std = vals.rolling(window=win_samples, center=True, min_periods=win_samples // 2).std().to_numpy()

    # NaN regions count as "not stuck"
    is_flat = (rolling_std < std_thresh) & ~np.isnan(raw_signal)

    # Collect candidate flat segments, then filter by wrong-level criterion
    segments = []
    in_seg = False
    seg_start = 0
    for k in range(len(is_flat) + 1):
        ended = (k == len(is_flat)) or not is_flat[k]
        if is_flat[k] if k < len(is_flat) else False:
            if not in_seg:
                in_seg = True
                seg_start = k
        if ended and in_seg:
            in_seg = False
            seg_end = k - 1
            if seg_end - seg_start + 1 >= min_samples:
                seg_mean = float(np.nanmean(raw_signal[seg_start:seg_end + 1]))
                if abs(seg_mean - run_median) > level_thresh_mm:
                    segments.append((seg_start, seg_end))

    return segments


def _detect_dc_step(
    raw_signal: np.ndarray,
    fs: float,
    step_mm: float = None,
    window_s: float = None,
) -> int | None:
    """Detect a sudden one-way DC level shift (probe repositioning / hardware fault).

    Physical motivation:
      A probe can shift its DC output mid-run: the mean level jumps from the normal
      operating range (~100 mm tip-to-water) to a fault level (~190 mm in observed
      cases). The wave signal may still be superimposed on the new DC level, but
      the stillwater reference captured in the first 2 s is no longer valid — all
      post-step data must be discarded.

      This is distinct from a spike (which the velocity filter handles): a spike
      returns to the previous level immediately. A DC step stays at the new level
      for the remainder of the recording.

    Algorithm:
      1. Compute a causal rolling median (window = DC_STEP_WINDOW_S, default 1 s).
         A 1 s window covers ~0.65 wave periods at the lowest wave frequency
         (0.65 Hz), so the median tracks the DC level rather than wave oscillations.
         Maximum median drift from a wave alone: ~25 mm at 0.65 Hz, 0.3 V amplitude.
      2. Compute the absolute element-wise diff of the rolling median — this peaks
         at the midpoint of the causal window's transition through the step.
         For a causal window of length win, the transition spans [s, s+win] where
         s is the actual step onset. The diff peak occurs at approximately s+win//2.
      3. Subtract win//2 from the peak index to recover the estimated step onset s.
      4. Sanity check: compare pre-step and post-step medians. If the actual level
         change is below DC_STEP_MM, reject (no real step — just noise or a ramp).

    Returns:
      Index of the first invalid post-step sample (= estimated step onset, clamped
      to [0, n-1]), or None if no step is found.

    Note on precision:
      Step onset is accurate to ±(DC_STEP_WINDOW_S / 2) seconds. An additional
      DC_STEP_BUFFER_S margin is applied by the caller, so ±0.5 s imprecision in
      step detection is fully covered.
    """
    step_mm  = CLIP.DC_STEP_MM       if step_mm  is None else step_mm
    window_s = CLIP.DC_STEP_WINDOW_S if window_s is None else window_s

    win = max(int(window_s * fs), 10)
    n   = len(raw_signal)

    vals = pd.to_numeric(pd.Series(raw_signal), errors='coerce')
    # Causal (left-sided) rolling median: at sample i, covers [i-win+1, i].
    # min_periods=win//4 avoids NaN at the very start of the signal.
    rolling_med = vals.rolling(window=win, center=False,
                               min_periods=win // 4).median().to_numpy()

    # Quick reject: if the total range of the rolling median is below threshold,
    # there is definitely no step large enough to concern us.
    valid_med = rolling_med[~np.isnan(rolling_med)]
    if len(valid_med) < 2 or (np.nanmax(valid_med) - np.nanmin(valid_med)) < step_mm:
        return None

    # Find the sharpest transition in the rolling median.
    # The causal window creates a ramp at the step: the diff peaks at s + win//2.
    # Subtracting win//2 recovers the estimated onset sample s.
    diff_abs = np.abs(np.diff(rolling_med))
    peak_idx = int(np.nanargmax(diff_abs))
    step_idx = max(0, peak_idx - win // 2)

    # Sanity check: is the actual signal level genuinely different before vs after?
    min_flank = max(10, win // 4)
    if step_idx < min_flank or n - step_idx < min_flank:
        return None  # step too close to edge — unreliable estimate
    pre_level  = float(np.nanmedian(raw_signal[:step_idx]))
    post_level = float(np.nanmedian(raw_signal[step_idx:]))
    if abs(post_level - pre_level) < step_mm:
        return None  # level difference too small — likely just a drift, not a step

    return step_idx


def _zero_and_smooth_signals(
    dfs: dict,
    meta_sel: pd.DataFrame,
    stillwater: dict,
    cfg,
    win: int,
    debug: bool
) -> tuple[dict, dict]:
    """Zero signals using stillwater, clean outliers, interpolate small gaps, add moving averages.

    Returns (processed_dfs, clip_stats) where
    clip_stats = {path: {pos: {"samples_clipped": int, "max_gap": int, "step_at_s": float|None}}}
    """
    fs = MEASUREMENT.SAMPLING_RATE
    col_names = cfg.probe_col_names()
    processed_dfs = {}
    clip_stats = {}

    for _, row in meta_sel.iterrows():
        path = row["path"]
        if path not in dfs:
            print(f"Warning: File not loaded: {path}")
            continue

        df = dfs[path].copy()
        clip_stats[path] = {}

        freq = row.get("WaveFrequencyInput [Hz]")
        wind = row.get("WindCondition", "no")
        is_wave_run = freq is not None and not pd.isna(freq)
        is_stillwater = not is_wave_run and wind == "no"

        # Max gap to interpolate: 1/4 wavelength for wave runs, fixed fallback otherwise
        if is_wave_run:
            max_interp_gap = int(fs / (4.0 * float(freq)))
        else:
            max_interp_gap = CLIP.INTERP_MAX_GAP

        if is_stillwater:
            clip_mm = None  # no hard cap — stuck/velocity detection handles faults
        elif is_wave_run:
            volt = row.get("WaveAmplitudeInput [Volt]")
            if volt is not None and not pd.isna(float(volt)):
                wind_extra = CLIP.WIND_BASE_VOLT if wind != "no" else 0.0
                clip_mm = CLIP.WAVE_CLIP_FACTOR * (float(volt) + wind_extra)
            else:
                clip_mm = CLIP.WAVE_MM  # fallback: no voltage info
        else:
            clip_mm = CLIP.WAVE_MM  # nowave + wind runs

        for i, pos in col_names.items():
            probe_col = f"Probe {pos}"
            if probe_col not in df.columns:
                print(f"  Missing column {probe_col} in {Path(path).name}")
                continue

            # ── Layer 0a: Ceiling detection (raw signal, before zeroing) ─────────
            # Probe hardware saturates at ~198.70 mm when the tip is out of range.
            # Three observed variants: exact freeze, near-ceiling noise, and DC
            # shift with wave on top. All share the symptom that raw ≈ PROBE_CEILING_MM.
            # See CLIP.PROBE_CEILING_MM / CEILING_BAND_MM in constants.py.
            raw_vals = pd.to_numeric(df[probe_col], errors='coerce').to_numpy(dtype=float).copy()
            ceil_mask = np.abs(raw_vals - CLIP.PROBE_CEILING_MM) < CLIP.CEILING_BAND_MM
            if ceil_mask.any():
                n_ceil = int(ceil_mask.sum())
                raw_vals[ceil_mask] = np.nan
                print(f"  CEILCLIP [{Path(path).name}] {pos}: "
                      f"{n_ceil} samples within {CLIP.CEILING_BAND_MM:.1f} mm of "
                      f"ceiling ({CLIP.PROBE_CEILING_MM:.2f} mm) → NaN")

            # ── Layer 0b: DC step detection (raw signal, before zeroing) ─────────
            # Probe repositioning or hardware fault shifts the DC level mid-run
            # (e.g. from ~100 mm to ~190 mm). After the step, the stillwater
            # reference (first 2 s) is no longer valid — all post-step data is NaN.
            # A half-window safety buffer (DC_STEP_BUFFER_S) is prepended to the
            # NaN region to guard against the rolling-median lag underestimating
            # how early the step actually started.
            step_idx = _detect_dc_step(raw_vals, fs)
            step_at_s = None
            if step_idx is not None:
                buf_samples = int(CLIP.DC_STEP_BUFFER_S * fs)
                step_start  = max(0, step_idx - buf_samples)
                raw_vals[step_start:] = np.nan
                step_at_s = step_start / fs
                print(f"  STEPCLIP [{Path(path).name}] {pos}: "
                      f"DC step at ~{step_idx / fs:.1f} s "
                      f"→ NaN from {step_at_s:.1f} s onward "
                      f"({len(raw_vals) - step_start} samples)")

            # ── Layer 0c: Stuck probe detection (raw signal, before zeroing) ──────
            # Redundant safety net for ceiling-freeze variants (already caught by
            # 0a) and any other flat-at-wrong-level fault. Two conditions required:
            # rolling std < STUCK_STD_MM AND segment mean differs from run median
            # by > level_thresh_mm, to avoid false positives in genuinely still
            # pre-wave periods.
            stuck_segs = _detect_stuck_segments(raw_vals, fs)
            if stuck_segs:
                stuck_mask = np.zeros(len(raw_vals), dtype=bool)
                for s, e in stuck_segs:
                    stuck_mask[s:e + 1] = True
                n_stuck = int(stuck_mask.sum())
                raw_vals[stuck_mask] = np.nan
                print(f"  STUCK [{Path(path).name}] {pos}: "
                      f"{len(stuck_segs)} segment(s), {n_stuck} samples "
                      f"({n_stuck / fs:.1f} s) → NaN")
            else:
                stuck_mask = None

            eta_col = f"eta_{pos}"
            df[eta_col] = -(raw_vals - stillwater[path][i])

            # ── Layer 0d: Physical range clip (eta_ signal, after zeroing) ──────
            # Samples outside the probe's valid measurement window are physically
            # impossible and indicate sensor dropout (surface moved out of range).
            # Valid eta_ range is derived from probe_height_mm and probe_range_mode:
            #   eta_floor   = -(range_max_mm - probe_height_mm)  [deepest valid trough]
            #   eta_ceiling =   probe_height_mm - range_min_mm   [tallest valid crest]
            # Guard: skip if probe_height_mm is outside the window (eta_floor >= 0),
            # which happens for the old default height272 + high-range setup where
            # the stated max_mm (250) is likely a nominal accuracy limit rather than
            # a hard cutoff — those runs have no reliable range-based floor.
            _probe_h = row.get("probe_height_mm", PROBE_HEIGHT_DEFAULT_MM)
            _range_mode = row.get("probe_range_mode", "high")
            _range_limits = PROBE_RANGE_MODES.get(str(_range_mode))
            if _range_limits is not None and _probe_h is not None and not pd.isna(float(_probe_h)):
                _h = float(_probe_h)
                _eta_floor   = -(_range_limits["max_mm"] - _h)
                _eta_ceiling =   _h - _range_limits["min_mm"]
                # Floor clip only — ceiling clip deliberately omitted.
                # The high-range mode's stated min_mm (130 mm) is a nominal
                # accuracy limit, not a hard cutoff. Applying it as a ceiling
                # (e.g. +6 mm at height136) would clip real wave crests.
                if _eta_floor < 0:
                    _floor_mask = df[eta_col] < _eta_floor
                    if _floor_mask.any():
                        df.loc[_floor_mask, eta_col] = np.nan
                        print(f"  RANGECLIP [{Path(path).name}] {pos}: "
                              f"{int(_floor_mask.sum())} samples below "
                              f"floor {_eta_floor:.0f} mm → NaN")

            # ── Layer 1: Hard cap (wave/wind runs only; stillwater uses detection layers) ─
            if clip_mm is not None:
                outlier_mask = df[eta_col].abs() > clip_mm
                if outlier_mask.any():
                    df.loc[outlier_mask, eta_col] = np.nan
                    print(f"  CLIP [{Path(path).name}] {pos}: {int(outlier_mask.sum())} samples |η| > {clip_mm} mm → NaN")

            # ── Layer 2: Velocity filter + ±VEL_BUFFER shoulder removal ─────
            _eta = df[eta_col].to_numpy(dtype=float).copy()
            _d = np.diff(_eta)
            _rise = _d[:-1]
            _fall = _d[1:]
            _spike_core = (
                (np.abs(_rise) > CLIP.DIFF_MM) &
                (np.abs(_fall) > CLIP.DIFF_MM) &
                (_rise * _fall < 0)
            )
            spike_indices = np.where(_spike_core)[0] + 1
            if spike_indices.size > 0:
                buf_mask = np.zeros(len(_eta), dtype=bool)
                for offset in range(-CLIP.VEL_BUFFER, CLIP.VEL_BUFFER + 1):
                    shifted = spike_indices + offset
                    valid = (shifted >= 0) & (shifted < len(_eta))
                    buf_mask[shifted[valid]] = True
                _eta[buf_mask] = np.nan
                df[eta_col] = _eta
                print(f"  VELCLIP [{Path(path).name}] {pos}: {spike_indices.size} spike(s), {int(buf_mask.sum())} samples → NaN")

            # ── Layer 3: Isolated sample check ───────────────────────────────
            _nan_mask = np.isnan(_eta)
            _isolated = ~_nan_mask & np.roll(_nan_mask, 1) & np.roll(_nan_mask, -1)
            _isolated[[0, -1]] = False
            if _isolated.any():
                _eta[_isolated] = np.nan
                df[eta_col] = _eta
                print(f"  ISOCLIP [{Path(path).name}] {pos}: {int(_isolated.sum())} isolated sample(s) → NaN")

            # ── Clip stats ───────────────────────────────────────────────────
            _final_nan = np.isnan(_eta)
            n_clipped_total = int(_final_nan.sum())
            if _final_nan.any():
                runs = np.diff(np.concatenate([[0], _final_nan.astype(int), [0]]))
                starts = np.where(runs == 1)[0]
                ends   = np.where(runs == -1)[0]
                max_gap_val = int((ends - starts).max())
            else:
                max_gap_val = 0
            clip_stats[path][pos] = {
                "samples_clipped": n_clipped_total,
                "max_gap": max_gap_val,
                "stuck_segments": stuck_segs if stuck_segs else [],
                "step_at_s": step_at_s,  # None if no DC step detected; float (seconds) if detected
            }

            # ── eta_interp: pchip fill of small gaps (display layer) ─────────
            interp_col = f"eta_{pos}_interp"
            _valid = ~_final_nan
            if _final_nan.any() and _valid.sum() > 3:
                _idx = np.arange(len(_eta))
                _pchip = PchipInterpolator(_idx[_valid], _eta[_valid], extrapolate=False)
                _filled_interp = _pchip(_idx)
                _filled_interp = _remask_long_gaps(_filled_interp, _final_nan, max_interp_gap)
                df[interp_col] = _filled_interp
            else:
                df[interp_col] = _eta.copy()

            # ── _ma: linear fill of small gaps then smooth ───────────────────
            ma_col = f"{probe_col}_ma"
            if _final_nan.any() and _valid.sum() > 2:
                _idx = np.arange(len(_eta))
                _filled_ma = np.interp(_idx, _idx[_valid], _eta[_valid])
                _filled_ma = _remask_long_gaps(_filled_ma, _final_nan, max_interp_gap)
            else:
                _filled_ma = _eta.copy()
            df[ma_col] = pd.Series(_filled_ma, index=df.index).rolling(window=win, center=False).mean()

            if debug:
                print(f"  {Path(path).name:35} → {eta_col} mean = {df[eta_col].mean():.4f} mm")

        processed_dfs[path] = df

    return processed_dfs, clip_stats


def run_find_wave_ranges(
    processed_dfs: dict,
    meta_sel: pd.DataFrame,
    cfg,
    win: int,
    range_plot: bool,
    debug: bool
) -> pd.DataFrame:
    """Find wave ranges for all probes."""
    col_names = cfg.probe_col_names()  # {1: "9373/170", ...}
    for idx, row in meta_sel.iterrows():
        path = row["path"]
        df = processed_dfs[path]

        freq = row.get("WaveFrequencyInput [Hz]")
        freq = float(freq) if freq is not None and not pd.isna(freq) else None

        for i, pos in col_names.items():
            probe_col = f"Probe {pos}"
            start, end, debug_info = find_wave_range(
                df, row, data_col=probe_col, probe_num=i, detect_win=win, range_plot=range_plot, debug=debug
            )
            meta_sel.loc[idx, f"Computed Probe {pos} start"] = start
            meta_sel.loc[idx, f"Computed Probe {pos} end"] = end

            # Wave quality metrics from upcrossings
            if (
                debug_info is not None
                and freq is not None
                and start is not None
                and end is not None
            ):
                upcrossings = debug_info.get("wave_upcrossings")
                samples_per_period = int(round(MEASUREMENT.SAMPLING_RATE / freq))
                sig = df[probe_col].values[start:end]

                # 1. wave_stability: autocorrelation at lag = 1 period (FFT-based, O(n log n))
                sig_centered = sig - np.mean(sig)
                n = len(sig_centered)
                if n > 2 * samples_per_period:
                    fft_sig = np.fft.rfft(sig_centered, n=2 * n)
                    ac = np.fft.irfft(fft_sig * np.conj(fft_sig))[:n].real
                    ac_norm = ac / ac[0] if ac[0] != 0 else ac
                    wave_stability = float(ac_norm[samples_per_period])
                else:
                    wave_stability = np.nan

                # 2. period_amplitude_cv: coefficient of variation of per-period amplitudes
                if upcrossings is not None and len(upcrossings) >= 2:
                    period_amps = [
                        float(np.ptp(sig[upcrossings[j] - start : upcrossings[j + 1] - start]))
                        for j in range(len(upcrossings) - 1)
                        if upcrossings[j + 1] - start <= n and upcrossings[j] - start >= 0
                    ]
                    if len(period_amps) >= 2:
                        mean_amp = np.mean(period_amps)
                        period_cv = float(np.std(period_amps) / mean_amp) if mean_amp > 0 else np.nan
                    else:
                        period_cv = np.nan
                else:
                    period_cv = np.nan

                # 3. Hm0: spectral significant wave height = 4 × std(eta)
                hm0 = float(4.0 * np.std(sig_centered)) if n > 0 else np.nan

                # 4. Hs: zero-crossing significant wave height = mean of top 1/3 wave heights
                hs = np.nan
                if upcrossings is not None and len(upcrossings) >= 2:
                    wave_heights = [
                        float(np.ptp(sig[upcrossings[j] - start : upcrossings[j + 1] - start]))
                        for j in range(len(upcrossings) - 1)
                        if upcrossings[j + 1] - start <= n and upcrossings[j] - start >= 0
                    ]
                    if wave_heights:
                        n_top = max(1, len(wave_heights) // 3)
                        hs = float(np.mean(sorted(wave_heights, reverse=True)[:n_top]))

                meta_sel.loc[idx, f"Probe {pos} wave_stability"] = wave_stability
                meta_sel.loc[idx, f"Probe {pos} period_amplitude_cv"] = period_cv
                meta_sel.loc[idx, f"Probe {pos} Significant Wave Height Hm0"] = hm0
                meta_sel.loc[idx, f"Probe {pos} Significant Wave Height Hs"]  = hs

        if debug and start:
            print(f'start: {start}, end: {end}, debug: {debug_info}')

    return meta_sel


def _update_all_metrics(
    processed_dfs: dict,
    meta_sel: pd.DataFrame,
    stillwater: dict,
    amplitudes_psd_df: pd.DataFrame,
    amplitudes_fft_df: pd.DataFrame,
    cfg,
) -> pd.DataFrame:
    """Update metadata with all computed metrics, using position-based probe names."""
    meta_indexed = meta_sel.set_index(GC.PATH).copy()
    col_names = cfg.probe_col_names()  # {1: "9373/170", 2: "12545", ...}

    # ============================================================================
    # SECTION 1: DIRECT ASSIGNMENT of pre-computed values
    # ============================================================================

    # Amplitudes from np.percentile — columns already position-based
    amplitudes = compute_amplitudes(processed_dfs, meta_sel, cfg)
    amp_cols = [c for c in amplitudes.columns if c != GC.PATH]
    meta_indexed[amp_cols] = amplitudes.set_index(GC.PATH)[amp_cols]

    # Amplitudes from PSD
    psd_cols = [c for c in amplitudes_psd_df.columns if c != GC.PATH]
    meta_indexed[psd_cols] = amplitudes_psd_df.set_index(GC.PATH)[psd_cols]

    # FFT amplitudes, frequencies, periods
    fft_df_indexed = amplitudes_fft_df.set_index(GC.PATH)
    fft_cols = [c for c in amplitudes_fft_df.columns if c != GC.PATH]
    meta_indexed[fft_cols] = fft_df_indexed[fft_cols]

    # ============================================================================
    # SECTION 2: DERIVED CALCULATIONS using the assigned values
    # ============================================================================

    # Windspeed needed by calculate_wavedimensions (Wind/Celerity, f/f_PM)
    meta_indexed[GC.WINDSPEED] = calculate_windspeed(meta_indexed[GC.WIND_CONDITION])

    for i, pos in col_names.items():
        freq_col = f"Probe {pos} Frequency (FFT)"
        k_col    = f"Probe {pos} Wavenumber (FFT)"
        amp_col  = f"Probe {pos} Amplitude"

        meta_indexed[k_col] = calculate_wavenumbers_vectorized(
            frequencies=meta_indexed[freq_col],
            heights=meta_indexed[GC.WATER_DEPTH]
        )

        res = calculate_wavedimensions(
            k=meta_indexed[k_col],
            H=meta_indexed[GC.WATER_DEPTH],
            PC=meta_indexed[GC.PANEL_CONDITION],
            amp=meta_indexed[amp_col],
            windspeed=meta_indexed[GC.WINDSPEED],
            freq=meta_indexed[freq_col],
        )

        wave_dim_cols = [
            f"Probe {pos} Wavelength (FFT)",
            f"Probe {pos} kL (FFT)",
            f"Probe {pos} ka (FFT)",
            f"Probe {pos} tanh(kH) (FFT)",
            f"Probe {pos} Celerity (FFT)",
        ]
        meta_indexed[wave_dim_cols] = res[RC.WAVE_DIMENSION_COLS]

        physics_cols = [
            f"Probe {pos} Froude (FFT)",
            f"Probe {pos} Wind/Celerity (FFT)",
            f"Probe {pos} f/f_PM (FFT)",
            f"Probe {pos} Ursell (FFT)",
        ]
        meta_indexed[physics_cols] = res[RC.PHYSICS_COLS]

    # Global wave dimensions — "input-based ka"
    # k is solved from WaveFrequencyInput [Hz] (wavemaker setting, not measured).
    # Amplitude is the measured IN probe amplitude.
    # This gives a hybrid ka: intended frequency × actual wave height at IN probe.
    # Thesis writer: distinguish from per-probe "Probe {pos} ka (FFT)" which uses
    # the FFT-measured frequency at each probe (closer to the actual incoming wave).
    in_pos = col_names[cfg.in_probe]
    meta_indexed[GC.WAVENUMBER] = calculate_wavenumbers_vectorized(
        frequencies=meta_indexed[GC.WAVE_FREQUENCY_INPUT],
        heights=meta_indexed[GC.WATER_DEPTH]
    )
    global_res = calculate_wavedimensions(
        k=meta_indexed[GC.WAVENUMBER],
        H=meta_indexed[GC.WATER_DEPTH],
        PC=meta_indexed[GC.PANEL_CONDITION],
        amp=meta_indexed[f"Probe {in_pos} Amplitude"],
        windspeed=meta_indexed[GC.WINDSPEED],
        freq=meta_indexed[GC.WAVE_FREQUENCY_INPUT],
    )
    meta_indexed[CG.GLOBAL_WAVE_DIMENSION_COLS] = global_res[RC.WAVE_DIMENSION_COLS_WITH_KH]
    meta_indexed[[GC.FROUDE, GC.WIND_CELERITY, GC.F_PM_RATIO, GC.URSELL]] = global_res[RC.PHYSICS_COLS]

    # Stillwater columns are already per-row values set by ensure_stillwater_columns

    return meta_indexed.reset_index()


def _set_output_folder(
    meta_sel: pd.DataFrame,
    meta_full: pd.DataFrame,
    debug: bool
) -> pd.DataFrame:
    """Velg output folder for processed data."""
    if "PROCESSED_folder" not in meta_sel.columns:
        if "PROCESSED_folder" in meta_full.columns:
            folder = meta_full["PROCESSED_folder"].iloc[0]
        elif "experiment_folder" in meta_full.columns:
            folder = f"PROCESSED-{meta_full['experiment_folder'].iloc[0]}"
        else:
            raw_folder = Path(meta_full["path"].iloc[0]).parent.name
            folder = f"PROCESSED-{raw_folder}"

        meta_sel["PROCESSED_folder"] = folder
        if debug:
            print(f"Set PROCESSED_folder = {folder}")

    return meta_sel


def _write_quality_flags(
    meta_sel: pd.DataFrame,
    clip_stats: dict,
    cfg,
    flags_file: Path,
) -> pd.DataFrame:
    """Check for probe malfunctions overlapping the stable analysis window.

    Adds `probe_{pos}_malfunction` (bool) and `quality_flag` ('ok' / description)
    columns to meta_sel.  Appends flagged runs to flags_file (one line per run).

    A run is flagged if a stuck segment overlaps [good_start_idx, good_end_idx]
    for the IN or OUT probe.  Other-probe malfunctions are noted but don't flag
    the run as unusable for damping.
    """
    col_names = cfg.probe_col_names()
    in_pos  = cfg.probe_col_names()[cfg.in_probe]
    out_pos = cfg.probe_col_names()[cfg.out_probe]

    # Initialise columns
    for pos in col_names.values():
        meta_sel[f"probe_{pos}_malfunction"] = False
    if "quality_flag" not in meta_sel.columns:
        meta_sel["quality_flag"] = "ok"

    flagged_lines = []

    for _, row in meta_sel.iterrows():
        path = row["path"]
        fname = Path(path).name
        stats = clip_stats.get(path, {})

        # good_start / good_end for the IN probe (all probes share the same window)
        start_col = f"Computed Probe {in_pos} start"
        end_col   = f"Computed Probe {in_pos} end"
        good_start = row.get(start_col)
        good_end   = row.get(end_col)
        has_window = (
            good_start is not None and good_end is not None
            and not pd.isna(good_start) and not pd.isna(good_end)
        )

        affected_in_window = []
        for pos, pstats in stats.items():
            segs      = pstats.get("stuck_segments", [])
            step_at_s = pstats.get("step_at_s")      # None or float (seconds from run start)

            # Stuck probe segments (Layer 0c) ─────────────────────────────────
            if segs:
                idx = meta_sel.index[meta_sel["path"] == path][0]
                meta_sel.at[idx, f"probe_{pos}_malfunction"] = True
                if has_window:
                    for s, e in segs:
                        if s <= int(good_end) and e >= int(good_start):
                            affected_in_window.append(pos)
                            break

            # DC step (Layer 0b) ───────────────────────────────────────────────
            # A step means all samples from step_at_s onward are NaN. If the
            # step occurred before the analysis window ends, the FFT will fail
            # (or yield a severely degraded result) and the run should be flagged.
            if step_at_s is not None:
                idx = meta_sel.index[meta_sel["path"] == path][0]
                meta_sel.at[idx, f"probe_{pos}_malfunction"] = True
                if has_window:
                    step_sample = int(step_at_s * MEASUREMENT.SAMPLING_RATE)
                    if step_sample < int(good_end) and pos not in affected_in_window:
                        affected_in_window.append(pos)

        if affected_in_window:
            idx = meta_sel.index[meta_sel["path"] == path][0]
            is_critical = in_pos in affected_in_window or out_pos in affected_in_window
            flag = (
                "probe_malfunction_critical"   # IN or OUT broken in analysis window
                if is_critical else
                "probe_malfunction_secondary"  # only auxiliary probe broken
            )
            meta_sel.at[idx, "quality_flag"] = flag
            critical_tag = " [CRITICAL — IN/OUT affected]" if is_critical else ""
            line = (
                f"{flag} | {fname} | probes: {', '.join(affected_in_window)}{critical_tag}"
            )
            print(f"  ⚠ QUALITY FLAG: {line}")
            flagged_lines.append(line)

    if flagged_lines:
        flags_file.parent.mkdir(parents=True, exist_ok=True)
        with open(flags_file, "a") as fh:
            for line in flagged_lines:
                fh.write(line + "\n")
        print(f"  Quality flags written → {flags_file}")

    return meta_sel


def _tail_upcross_periods(
    sig: np.ndarray, fs: float
) -> tuple[np.ndarray, np.ndarray]:
    """Zero-upcrossing period series for the tail signal.

    eta_ is already zeroed to the stillwater level, so zero-upcrossings of
    eta_ define wave cycle boundaries.  Period of cycle i is simply
        T[i] = t_upcross[i+1] − t_upcross[i]
    One value per wave cycle — no windowing, no averaging.

    Returns
    -------
    t_uc : 1-D array, time [s] of each upcrossing (0 = start of sig)
    T_uc : 1-D array, period of that cycle [s]; len(T_uc) == len(t_uc).
           The last upcrossing has no following pair and is excluded so the
           arrays have equal length.

    Crossings with T < CLIP.TAIL_PERIOD_MIN_S or T > CLIP.TAIL_PERIOD_MAX_S
    are dropped to filter noise spikes and slow DC-drift crossings.
    """
    s     = np.where(np.isnan(sig), 0.0, np.asarray(sig, dtype=float))
    cross = np.where((s[:-1] < 0) & (s[1:] >= 0))[0]
    if len(cross) < 2:
        return np.array([]), np.array([])
    t_uc = cross / fs
    T_uc = np.diff(t_uc)
    keep = (T_uc >= CLIP.TAIL_PERIOD_MIN_S) & (T_uc <= CLIP.TAIL_PERIOD_MAX_S)
    return t_uc[:-1][keep], T_uc[keep]


def _compute_tail_amplitudes(
    processed_dfs: dict,
    meta_sel: pd.DataFrame,
    cfg,
) -> pd.DataFrame:
    """Measure residual wave energy in the tail of each wave run.

    After the mstop sequence the wavemaker decelerates and the tank rings down.
    This function captures the amplitude of the signal in TAIL_WINDOW_S seconds
    immediately AFTER the stable wave range ends (i.e., just after the last clean
    wave period used for FFT). That amplitude is the starting residual energy that
    the NEXT run will inherit if the inter-run wait is short.

    Uses the same percentile amplitude definition as the main analysis:
        amp = (P97.5 − P2.5) / 2

    Stores `mstop_tail_mm_{pos}` per probe row in meta_sel.
    NaN for nowave/stillwater runs (no wave range → no tail defined).

    Physical note:
        The free long-wave (piston transient) arrives at each probe well before
        the tail window starts, so it does not contaminate this measurement.
        Wind-wave energy will appear here for fullwind runs — the percentile
        amplitude includes all frequencies, not just the paddle frequency.
        Compare `mstop_tail_mm` across wind conditions with this in mind.
    """
    fs        = MEASUREMENT.SAMPLING_RATE
    tail_n    = int(CLIP.TAIL_WINDOW_S * fs)
    col_names = cfg.probe_col_names()
    in_pos    = col_names[cfg.in_probe]
    end_col   = f"Computed Probe {in_pos} end"

    for idx, row in meta_sel.iterrows():
        path     = row["path"]
        good_end = row.get(end_col)
        if path not in processed_dfs or good_end is None or pd.isna(good_end):
            continue  # nowave run or wave range not found — skip

        good_end  = int(good_end)
        df        = processed_dfs[path]
        n_signal  = len(df)

        # Paddle period for this run (used to define "cleared" threshold)
        freq_hz    = row.get("WaveFrequencyInput [Hz]")
        paddle_T   = (1.0 / float(freq_hz)) if freq_hz and not pd.isna(freq_hz) else None

        for pos in col_names.values():
            eta_col = f"eta_{pos}"
            if eta_col not in df.columns:
                continue

            sig_full = df[eta_col].values  # full run signal

            # ── mstop_tail_mm: amplitude in first TAIL_WINDOW_S of tail ──────
            tail_start = good_end
            tail_end   = min(n_signal, good_end + tail_n)
            if tail_end > tail_start:
                seg = sig_full[tail_start:tail_end]
                seg = seg[~np.isnan(seg)]
                meta_sel.at[idx, f"mstop_tail_mm_{pos}"] = float(
                    (np.percentile(seg, 97.5) - np.percentile(seg, 2.5)) / 2.0
                ) if len(seg) >= 10 else np.nan
            else:
                meta_sel.at[idx, f"mstop_tail_mm_{pos}"] = np.nan

            # ── tail period diagnostics (zero-upcrossing, per cycle) ─────────
            # Compute the per-cycle period series for the entire tail, then
            # read off the period at each configured sample offset (nearest
            # upcrossing to that time).  No windowing — every dot is one cycle.
            tail_all = sig_full[good_end:]
            t_uc, T_uc = _tail_upcross_periods(tail_all, fs)

            if len(t_uc) == 0:
                # No valid crossings — tail too short or below noise
                for t_s in CLIP.TAIL_PERIOD_SAMPLE_OFFSETS_S:
                    meta_sel.at[idx, f"tail_uc_period_at_{t_s:.0f}s_{pos}"] = np.nan
                meta_sel.at[idx, f"tail_clear_s_{pos}"] = np.nan
                continue

            for t_s in CLIP.TAIL_PERIOD_SAMPLE_OFFSETS_S:
                # Find the upcrossing nearest to t_s seconds into the tail
                diffs = np.abs(t_uc - t_s)
                nearest = int(np.argmin(diffs))
                # Only accept if the nearest upcrossing is within half a paddle
                # period of the requested time (avoids reading a crossing from
                # the wrong part of the tail when the tail is short)
                if paddle_T is not None and diffs[nearest] > 0.5 * paddle_T:
                    meta_sel.at[idx, f"tail_uc_period_at_{t_s:.0f}s_{pos}"] = np.nan
                elif diffs[nearest] > 5.0:   # fallback: >5 s away → skip
                    meta_sel.at[idx, f"tail_uc_period_at_{t_s:.0f}s_{pos}"] = np.nan
                else:
                    meta_sel.at[idx, f"tail_uc_period_at_{t_s:.0f}s_{pos}"] = float(T_uc[nearest])

            # tail_clear_s: time of the first upcrossing cycle whose period
            # exceeds TAIL_CLEAR_FACTOR × paddle_T.  Indicates when the
            # paddle-frequency energy has dissipated (longer period = seiche).
            if paddle_T is None:
                meta_sel.at[idx, f"tail_clear_s_{pos}"] = np.nan
                continue

            clear_threshold = CLIP.TAIL_CLEAR_FACTOR * paddle_T
            cleared = np.where(T_uc > clear_threshold)[0]
            meta_sel.at[idx, f"tail_clear_s_{pos}"] = (
                float(t_uc[cleared[0]]) if cleared.size > 0 else np.nan
            )

    return meta_sel


def process_selected_data(
    dfs: dict[str, pd.DataFrame],
    meta_sel: pd.DataFrame,
    meta_full: pd.DataFrame,
    processvariables: dict,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict]:
    """
    1. Zeroes all selected runs using the shared stillwater levels.
    2. Adds eta_1..eta_4 (zeroed signal) and moving average.
    3. Find wave range (optional)
    4. Regner PSD og FFT med tilhørende
    5. Oppdaterer meta
    """
    fs = MEASUREMENT.SAMPLING_RATE
    # 0. unpack processvariables
    prosessering = processvariables.get("prosessering", {})

    force_recompute = prosessering.get("force_recompute", False)
    debug = prosessering.get("debug", False)
    win = prosessering.get("smoothing_window", 1)
    find_range = prosessering.get("find_range", False)
    range_plot = prosessering.get("range_plot", False)

    # Derive cfg once for the whole folder (all files share the same configuration)
    file_date = datetime.fromisoformat(str(meta_sel["file_date"].iloc[0]))
    cfg = get_configuration_for_date(file_date)

    # 0b. Capture hardware speed-of-sound from the raw "Mach" column.
    #
    # Every raw CSV row contains a 6th value: the probe hardware's own real-time
    # measurement of the speed of sound in air [m/s].  The data loader reads this
    # as column "Mach" but discards it after loading — it is not in processed_dfs
    # or meta.json.  We extract two scalars per run here, before the raw DataFrames
    # are processed and "Mach" is lost.
    #
    # Physical meaning:
    #   c_air ≈ 331 + 0.606 × T_Celsius  [m/s]   (dry air approximation)
    #   Humidity adds a small positive correction (~0.1–0.3 % at typical indoor RH).
    #
    # The hardware almost certainly APPLIES this measurement for its own time-of-
    # flight → mm conversion (that is why it measures it).  The reported probe values
    # are therefore already compensated; the "Mach" column is diagnostic logging of
    # the compensation that was applied.
    #
    # Even in the worst case (no hardware compensation, fixed internal c_ref):
    #   Total observed spread across all sessions: 342.20–343.44 m/s = 0.36 %
    #   Scale error on a 10 mm wave: ~0.036 mm — well below 0.25 mm target.
    #   For OUT/IN ratios: ZERO — both probes share the same air column at the
    #   same moment, so any scale error cancels in the ratio.
    #
    # We store mean ± std per run so that:
    #   (a) seasonal/daily temperature trends can be plotted from combined_meta
    #       (sound_speed_mean_ms is a temperature proxy without a direct sensor)
    #   (b) the November 2025 vs March 2026 comparison can quote the actual
    #       lab-condition difference rather than a theoretical worst-case bound
    #   (c) the late-March high-humidity sessions are identifiable by their
    #       distinctly lower mean c_air (~342.2 m/s vs ~343.4 m/s in November)
    for _, _row in meta_sel.iterrows():
        _path = _row["path"]
        _df_raw = dfs.get(_path)
        if _df_raw is None or "Mach" not in _df_raw.columns:
            continue
        _mach = pd.to_numeric(_df_raw["Mach"], errors="coerce").dropna()
        if len(_mach) == 0:
            continue
        _mask = meta_sel["path"] == _path
        meta_sel.loc[_mask, "sound_speed_mean_ms"] = float(_mach.mean())
        meta_sel.loc[_mask, "sound_speed_std_ms"]  = float(_mach.std())

    # 1. Ensure stillwater levels are computed
    _meta_full_cols_before = set(meta_full.columns)
    meta_full = ensure_stillwater_columns(dfs, meta_full, cfg)
    stillwater = _extract_stillwater_levels(meta_full, cfg, debug)

    # Propagate any columns that ensure_stillwater_columns added to meta_full → meta_sel.
    # meta_sel is what gets saved to meta.json (here at line ~1172 and in processor2nd).
    # Without this step, all stillwater columns are lost when force-recompute=True because
    # the later save overwrites meta.json entirely with meta_sel.
    _sw_new_cols = [c for c in meta_full.columns if c not in _meta_full_cols_before]
    if _sw_new_cols:
        _mf_sw = meta_full.set_index("path")[_sw_new_cols]
        for c in _sw_new_cols:
            meta_sel[c] = meta_sel["path"].map(_mf_sw[c])

    # 2. Process dataframes: zero, clean, interpolate, add moving averages
    processed_dfs, clip_stats = _zero_and_smooth_signals(dfs, meta_sel, stillwater, cfg, win, debug)

    # Merge clip quality stats into meta_sel
    col_names = cfg.probe_col_names()
    for path, probe_stats in clip_stats.items():
        mask = meta_sel["path"] == path
        for pos, stats in probe_stats.items():
            meta_sel.loc[mask, f"samples_clipped_{pos}"] = stats["samples_clipped"]
            meta_sel.loc[mask, f"max_gap_{pos}"]         = stats["max_gap"]
            # NaN means no step detected (normal run); a float value = step onset in seconds
            meta_sel.loc[mask, f"step_at_s_{pos}"] = stats.get("step_at_s", None)

    # 3. Optional: find wave ranges
    if find_range:
        meta_sel = run_find_wave_ranges(processed_dfs, meta_sel, cfg, win, range_plot, debug)
        # 3a. Tail amplitudes — residual energy after the wave train, per probe
        meta_sel = _compute_tail_amplitudes(processed_dfs, meta_sel, cfg)

    # 3b. Quality flags: probe malfunction vs stable analysis window
    _flags_file = Path(__file__).parents[1] / "waveprocessed" / "quality_flags.txt"
    meta_sel = _write_quality_flags(meta_sel, clip_stats, cfg, _flags_file)
    # Merge per-probe malfunction bools into meta_sel (already done inside _write_quality_flags)

    # 4. a - Compute PSDs and amplitudes from PSD (wave runs only)
    psd_dict, amplitudes_psd_df = compute_psd_with_amplitudes(processed_dfs, meta_sel, cfg, fs=fs, debug=debug)

    # 4. a2 - Broadband PSD for nowave runs (wind-only + stillwater), merged into psd_dict
    _meta_nowave = meta_sel[meta_sel["WaveFrequencyInput [Hz]"].isna()]
    if not _meta_nowave.empty:
        psd_dict.update(compute_nowave_psd(processed_dfs, _meta_nowave, cfg, fs=fs))

    # 4. b - compute FFT and amplitudes from FFT
    fft_dict, amplitudes_fft_df = compute_fft_with_amplitudes(processed_dfs, meta_sel, cfg, fs=fs, debug=debug)

    # 5. Compute and update all metrics (amplitudes, wavenumbers, dimensions, windspeed)
    meta_sel = _update_all_metrics(processed_dfs, meta_sel, stillwater, amplitudes_psd_df, amplitudes_fft_df, cfg)

    # 6. Set output folder and save metadata
    meta_sel = _set_output_folder(meta_sel, meta_full, debug)

    update_processed_metadata(meta_sel, force_recompute=force_recompute)

    if debug:
        print(f"\nProcessing complete! {len(processed_dfs)} files zeroed and ready.")

    return processed_dfs, meta_sel, psd_dict, fft_dict
