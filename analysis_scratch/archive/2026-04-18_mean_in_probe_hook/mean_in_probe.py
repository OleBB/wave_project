"""
Mean IN probe reference
========================

The default IN probe in `march2026_better_rearranging` is 9373/170. Its
partner 9373/340 sits at the **same longitudinal distance** from the
paddle but a different lateral position. In principle the two probes
see the same incident wave, so single-probe glitches (e.g. the 0.3 V
nowind transient dip at 9373/170 caught by the Huseby & Grue window
check, see ``analysis_scratch/huseby_grue_window.pdf``) can be
detected by comparing them.

This module promotes that comparison into the canonical IN reference
used by CH05 figures. For every row where both probes have an FFT
amplitude:

  A_in_mean  = mean(A_9373/170, A_9373/340)
  ka_in_mean = mean(ka_9373/170, ka_9373/340)      (re-derived)

and three consistency columns:

  ain_disagree_mm   = |A_9373/170 − A_9373/340|
  ain_disagree_frac = ain_disagree_mm / A_in_mean
  ain_probe_consistent = ain_disagree_frac < threshold   (default 10%)

The transformation stores the derived values in a pseudo-probe column
``Probe 9373_mean Amplitude (FFT)`` and sets ``in_position`` to the
pseudo value ``"9373_mean"``. The existing plotter functions look up
``Probe {in_position} Amplitude (FFT)`` dynamically, so no plotter
code change is required.

Usage — inline in main_save_figures.py, after the Mooring merge::

    from wavescripts.mean_in_probe import apply_mean_in_reference
    apply_mean_in_reference(meta_results)

This is a post-load transformation. It does not touch the pipeline or
the cached meta.json. The originals (``Probe 9373/170 Amplitude (FFT)``
and ``Probe 9373/340 Amplitude (FFT)``) are preserved unchanged.

Per-row behaviour:
  - Where both probes have valid FFT amplitudes: uses the mean, sets
    in_position = "9373_mean".
  - Where only one probe is valid: uses the single valid value and
    still sets in_position = "9373_mean" (documented fallback).
  - Where neither probe is valid: leaves row untouched (in_position
    keeps its original value, mean columns stay NaN).

The ``IN ka (FFT)`` column is overwritten in place with the mean of
``Probe 9373/170 ka (FFT)`` and ``Probe 9373/340 ka (FFT)`` — these
are independent per-probe measurements (same k, but measured ``a``
differs), so the mean is the natural derived quantity.
"""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd

IN1_POS = "9373/170"
IN2_POS = "9373/340"
MEAN_POS = "9373_mean"

_AIN_FFT_1 = f"Probe {IN1_POS} Amplitude (FFT)"
_AIN_FFT_2 = f"Probe {IN2_POS} Amplitude (FFT)"
_AIN_TD_1  = f"Probe {IN1_POS} Amplitude"
_AIN_TD_2  = f"Probe {IN2_POS} Amplitude"
_KA_1      = f"Probe {IN1_POS} ka (FFT)"
_KA_2      = f"Probe {IN2_POS} ka (FFT)"

_MEAN_FFT  = f"Probe {MEAN_POS} Amplitude (FFT)"
_MEAN_TD   = f"Probe {MEAN_POS} Amplitude"
_MEAN_KA   = f"Probe {MEAN_POS} ka (FFT)"


def apply_mean_in_reference(meta_df: pd.DataFrame,
                             disagreement_threshold_frac: float = 0.10) -> pd.DataFrame:
    """
    Replace the IN probe reference with the mean of 9373/170 and 9373/340.

    Mutates ``meta_df`` in place (also returns it for chaining). Adds the
    pseudo-probe columns, consistency flags, and updates ``in_position``
    / ``IN ka (FFT)``.

    Parameters
    ----------
    meta_df : pd.DataFrame
        Metadata frame (typically ``meta_results``) with
        ``Probe 9373/170 Amplitude (FFT)`` and ``Probe 9373/340
        Amplitude (FFT)`` columns.
    disagreement_threshold_frac : float
        Threshold for the ``ain_probe_consistent`` flag. Default 10%.

    Returns
    -------
    meta_df : pd.DataFrame
        The same frame, mutated.
    """
    # Only rows with the primary probe config (9373/170 as IN) qualify
    # for the transformation. Rows with the old config (9373/250 IN)
    # don't have a 9373/340 partner — leave them untouched.
    if "in_position" not in meta_df.columns:
        print("[mean_in_probe] in_position column missing — nothing to do")
        return meta_df

    qualifies = meta_df["in_position"].eq(IN1_POS)
    n_qual = int(qualifies.sum())
    print(f"[mean_in_probe] {n_qual}/{len(meta_df)} rows qualify "
          f"(in_position == '{IN1_POS}')")
    if n_qual == 0:
        return meta_df

    # Silence "Mean of empty slice" warnings — nowave runs have NaN FFT
    # amplitudes in both probes (no paddle wave to measure), so nanmean
    # over an all-NaN column pair is expected. NaN propagates correctly.
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning, message="Mean of empty slice"
        )
        warnings.filterwarnings("ignore", category=RuntimeWarning,
                                message="invalid value encountered")

        # ── FFT amplitude mean ─────────────────────────────────────────────────
        a1 = meta_df[_AIN_FFT_1].to_numpy(dtype=float, copy=True)
        a2 = meta_df[_AIN_FFT_2].to_numpy(dtype=float, copy=True)
        a_mean_fft = np.nanmean(np.vstack([a1, a2]), axis=0)
        meta_df[_MEAN_FFT] = a_mean_fft

        # ── Time-domain amplitude mean (damping_grouper fallback) ──────────────
        if _AIN_TD_1 in meta_df.columns and _AIN_TD_2 in meta_df.columns:
            t1 = meta_df[_AIN_TD_1].to_numpy(dtype=float, copy=True)
            t2 = meta_df[_AIN_TD_2].to_numpy(dtype=float, copy=True)
            meta_df[_MEAN_TD] = np.nanmean(np.vstack([t1, t2]), axis=0)

        # ── ka mean (k shared across parallel probes; mean(ka) == k × mean A) ─
        if _KA_1 in meta_df.columns and _KA_2 in meta_df.columns:
            k1 = meta_df[_KA_1].to_numpy(dtype=float, copy=True)
            k2 = meta_df[_KA_2].to_numpy(dtype=float, copy=True)
            ka_mean = np.nanmean(np.vstack([k1, k2]), axis=0)
            meta_df[_MEAN_KA] = ka_mean
            # Plotters read "IN ka (FFT)" directly — overwrite in place.
            if "IN ka (FFT)" in meta_df.columns:
                meta_df.loc[qualifies, "IN ka (FFT)"] = ka_mean[qualifies.to_numpy()]

        # ── Disagreement diagnostics ───────────────────────────────────────────
        disagree_mm = np.abs(a1 - a2)
        disagree_frac = disagree_mm / np.where(a_mean_fft > 0, a_mean_fft, np.nan)
    meta_df["ain_disagree_mm"]   = disagree_mm
    meta_df["ain_disagree_frac"] = disagree_frac
    meta_df["ain_probe_consistent"] = disagree_frac < disagreement_threshold_frac

    # ── Redirect in_position to the pseudo-probe ──────────────────────────────
    meta_df.loc[qualifies, "in_position"] = MEAN_POS

    # Diagnostic print
    n_consistent = int(meta_df.loc[qualifies, "ain_probe_consistent"].sum())
    worst = meta_df.loc[qualifies, "ain_disagree_frac"].max()
    print(f"[mean_in_probe] added Probe {MEAN_POS} columns; "
          f"{n_consistent}/{n_qual} runs consistent at "
          f"{disagreement_threshold_frac*100:.0f}% threshold "
          f"(worst disagreement = {worst*100:.1f}%)")

    return meta_df


def probe_agreement_summary(meta_df: pd.DataFrame) -> pd.DataFrame:
    """Return a small summary of the agreement stats by (amp, wind, freq)."""
    if "ain_disagree_frac" not in meta_df.columns:
        raise ValueError("Run apply_mean_in_reference first.")
    wave = meta_df[meta_df["WaveFrequencyInput [Hz]"].notna()].copy()
    wave["freq_r"] = wave["WaveFrequencyInput [Hz]"].round(2)
    wave["amp_r"]  = wave["WaveAmplitudeInput [Volt]"].round(2)
    grp = wave.groupby(["amp_r", "freq_r", "WindCondition"]).agg(
        n=("ain_disagree_frac", "size"),
        disagree_frac_mean=("ain_disagree_frac", "mean"),
        disagree_frac_max=("ain_disagree_frac", "max"),
        disagree_mm_mean=("ain_disagree_mm", "mean"),
        n_inconsistent=("ain_probe_consistent",
                        lambda s: int((~s).sum())),
    ).reset_index()
    return grp
