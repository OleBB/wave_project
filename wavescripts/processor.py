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
from wavescripts.constants import SIGNAL, RAMP, MEASUREMENT, CLIP, get_smoothing_window
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


def _probe_noise_amplitude_from_run(df: pd.DataFrame, probe_col: str, n_samples: int | None = None) -> float | None:
    """Return (P97.5 - P2.5) / 2 of a probe column — the pipeline amplitude definition.

    This is the noise floor for that probe in that run: the smallest wave amplitude
    indistinguishable from background fluctuations.  Consistent with how
    'Probe {pos} Amplitude' is computed everywhere else in the pipeline.
    """
    if probe_col not in df.columns:
        return None
    src = df[probe_col].iloc[:n_samples] if n_samples is not None else df[probe_col]
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
    Computes per-run stillwater levels by weighted linear regression over ALL
    no-wind runs in the folder, accounting for slow water-level drift (evaporation,
    wind setup) across a session.

    Data sources (per probe):
      - No-wind + no-wave runs: full-run median, weight=5  (high confidence)
      - No-wind + wave runs:    first PRE_WAVE_S seconds,  weight=1  (pre-wave window only)

    Outlier rejection: two-pass polyfit. Any point with residual > OUTLIER_MM is
    removed and the fit is recomputed.  Outliers are printed but not silently dropped.

    Safe to call multiple times — skips if per-row values already exist.
    """
    _PRE_WAVE_S  = 2.0   # seconds of pre-wave signal to use from wave runs
    _PRE_WAVE_N  = int(_PRE_WAVE_S * MEASUREMENT.SAMPLING_RATE)
    _W_NOWAVE    = 5     # weight for full no-wave runs
    _W_WAVE      = 1     # weight for 2-s pre-wave snippets
    _OUTLIER_MM  = 1.0   # residual threshold for outlier rejection [mm]

    meta = meta.copy()  # defragment before any column additions

    probe_positions = list(cfg.probe_col_names().values())
    probe_cols = [f"Stillwater Probe {pos}" for pos in probe_positions]

    # Skip if already computed with time-varying (interpolated) values
    if all(col in meta.columns for col in probe_cols):
        if meta[probe_cols].notna().all().all():
            if meta[probe_cols[0]].std() > 0.001:
                print("Stillwater levels already time-interpolated → skipping")
                return meta

    print("Computing stillwater levels from weighted linear fit over all no-wind runs...")

    col_names = cfg.probe_col_names()

    wind_str = meta[GC.WIND_CONDITION].astype(str).str.strip().str.lower()
    nowind   = wind_str.isin(["no", "", "nan", "none"])
    nowave   = meta[GC.WAVE_FREQUENCY_INPUT].isna() | meta["path"].astype(str).str.lower().str.contains("nowave")

    nowind_rows = meta[nowind].copy()
    if nowind_rows.empty:
        raise ValueError("No no-wind runs found — cannot compute stillwater!")

    nowind_rows = nowind_rows.assign(
        _t       = pd.to_datetime(nowind_rows["file_date"]),
        _is_nowave = nowave.reindex(nowind_rows.index).fillna(False),
    ).sort_values("_t").reset_index(drop=True)

    # ── per-probe weighted linear fit ────────────────────────────────────────
    new_cols   = {}   # probe col → Series of fitted values for all meta rows
    noise_updates = {}  # path → {col: std}

    for i, pos in col_names.items():
        probe_col = f"Probe {pos}"
        pts_t, pts_v, pts_w, pts_label = [], [], [], []

        for _, row in nowind_rows.iterrows():
            df_run = dfs.get(row["path"])
            if df_run is None:
                continue
            n_samp = None if row["_is_nowave"] else _PRE_WAVE_N
            v = _probe_median_from_run(df_run, probe_col, n_samp)
            if v is None:
                continue
            pts_t.append(row["_t"].timestamp())
            pts_v.append(v)
            pts_w.append(_W_NOWAVE if row["_is_nowave"] else _W_WAVE)
            pts_label.append(Path(row["path"]).name)

        if not pts_t:
            raise ValueError(f"No data points for Probe {pos}")

        ts = np.array(pts_t)
        vs = np.array(pts_v)
        ws = np.array(pts_w, dtype=float)
        t_origin = ts[0]
        ts_rel = ts - t_origin

        if len(pts_t) == 1:
            coeffs = np.array([0.0, vs[0]])  # flat
        else:
            coeffs = np.polyfit(ts_rel, vs, deg=1, w=ws)
            residuals = np.abs(vs - np.polyval(coeffs, ts_rel))
            outlier_mask = residuals > _OUTLIER_MM
            if outlier_mask.any() and (~outlier_mask).sum() >= 2:
                for lbl, res in zip(pts_label, residuals):
                    if res > _OUTLIER_MM:
                        print(f"  ⚠ Probe {pos}: outlier rejected — {lbl}  (residual {res:+.3f} mm)")
                coeffs = np.polyfit(ts_rel[~outlier_mask], vs[~outlier_mask], deg=1,
                                    w=ws[~outlier_mask])

        # Apply fit to every run in meta
        meta_t = pd.to_datetime(meta["file_date"])
        if meta_t.dt.tz is not None:
            meta_t = meta_t.dt.tz_localize(None)
        t_arr = meta_t.map(lambda t: t.timestamp()) - t_origin
        new_cols[f"Stillwater Probe {pos}"] = np.polyval(coeffs, t_arr).values

        # Print fit summary
        fit_at_start = np.polyval(coeffs, 0)
        fit_at_end   = np.polyval(coeffs, ts_rel[-1])
        print(f"  Probe {pos}: {len(pts_t)} points → "
              f"start {fit_at_start:.3f} mm, end {fit_at_end:.3f} mm "
              f"(drift {fit_at_end - fit_at_start:+.3f} mm over session)")

        # Control: print residuals for all data points
        fit_vals = np.polyval(coeffs, ts_rel)
        for lbl, actual, fitted, w in zip(pts_label, vs, fit_vals, ws):
            tag = " [nowave]" if w == _W_NOWAVE else " [pre-wave 2s]"
            print(f"    {lbl}: actual={actual:.3f}  fit={fitted:.3f}  Δ={actual-fitted:+.3f} mm{tag}")

        # Noise std for no-wave runs
        for _, row in nowind_rows[nowind_rows["_is_nowave"]].iterrows():
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

    # Compute (P97.5-P2.5)/2 noise floor per probe from all no-wind/no-wave runs,
    # broadcast to EVERY row as "Probe {n} at {pos} Uncertainty".
    nowave_nowind_rows = nowind_rows[nowind_rows["_is_nowave"]]
    noise_floor_accum: dict[int, list[float]] = {i: [] for i in col_names}
    for _, row in nowave_nowind_rows.iterrows():
        df_run = dfs.get(row["path"])
        if df_run is None:
            continue
        for i, pos in col_names.items():
            v = _probe_noise_amplitude_from_run(df_run, f"Probe {pos}", None)
            if v is not None:
                noise_floor_accum[i].append(v)

    unc_cols = {}
    for i, pos in col_names.items():
        vals = noise_floor_accum[i]
        if vals:
            unc_cols[f"Probe {i} at {pos} Uncertainty"] = float(np.mean(vals))
            print(f"  Noise floor  Probe {i} at {pos}: {np.mean(vals):.4f} mm"
                  f"  (n={len(vals)}, range {min(vals):.4f}–{max(vals):.4f})")
    if unc_cols:
        meta = meta.assign(**{c: v for c, v in unc_cols.items()})

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
    clip_stats = {path: {pos: {"samples_clipped": int, "max_gap": int}}}
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
            clip_mm = CLIP.NOWIND_MM
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

            eta_col = f"eta_{pos}"
            df[eta_col] = -(df[probe_col] - stillwater[path][i])

            # ── Layer 1: Hard cap ────────────────────────────────────────────
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
            clip_stats[path][pos] = {"samples_clipped": n_clipped_total, "max_gap": max_gap_val}

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

    # 1. Ensure stillwater levels are computed
    meta_full = ensure_stillwater_columns(dfs, meta_full, cfg)
    stillwater = _extract_stillwater_levels(meta_full, cfg, debug)

    # 2. Process dataframes: zero, clean, interpolate, add moving averages
    processed_dfs, clip_stats = _zero_and_smooth_signals(dfs, meta_sel, stillwater, cfg, win, debug)

    # Merge clip quality stats into meta_sel
    col_names = cfg.probe_col_names()
    for path, probe_stats in clip_stats.items():
        mask = meta_sel["path"] == path
        for pos, stats in probe_stats.items():
            meta_sel.loc[mask, f"samples_clipped_{pos}"] = stats["samples_clipped"]
            meta_sel.loc[mask, f"max_gap_{pos}"] = stats["max_gap"]
    
    # 3. Optional: find wave ranges
    if find_range:
        meta_sel = run_find_wave_ranges(processed_dfs, meta_sel, cfg, win, range_plot, debug)
    
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
