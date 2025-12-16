#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 22:01:11 2025

@author: ole
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, correlate
from pathlib import Path


def compare_probe_amplitudes_and_lag(df, 
                                   col1, 
                                   col2,
                                   start_ms,   # where the nice waves start
                                   end_ms):   # where they end
    """
    Returns amplitude and time lag between two probes in a clean interval.
    """
    # 1. Cut the good part
    window = df.loc[start_ms:end_ms]
    print('nu printes df.loc[start_ms]')
    print(df.loc[start_ms])
    
    s1 = window[col1].values
    s2 = window[col2].values
    time_ms = window.index.values  # assuming index is milliseconds

    # 2. Find amplitude (half of peak-to-peak over several waves)
    amp1 = (np.percentile(s1, 99) - np.percentile(s1, 1)) / 2
    amp2 = (np.percentile(s2, 99) - np.percentile(s2, 1)) / 2

    # Or even better: mean of peak-to-peak over individual waves
    # Find upward zero crossings for clean phase reference
    zero_cross1 = np.where(np.diff(np.sign(s1)) > 0)[0]
    zero_cross2 = np.where(np.diff(np.sign(s2)) > 0)[0]

    # Take only crossings inside the clean window
    peaks1, _ = find_peaks(s1, distance=50)  # adjust distance based on your sampling rate
    peaks2, _ = find_peaks(s2, distance=50)

    # Amplitude as mean crest-to-trough
    amp1_better = np.mean(s1[peaks1] - np.interp(peaks1, np.arange(len(s1)), s1) + s1[peaks1])
    # Simpler and robust:
    amp1_final = np.std(s1) * 2.0   # for pure sinusoid: H = 2√2 σ ≈ 2.8 σ, but we use 2σ as good approx
    amp2_final = np.std(s2) * 2.0

    # Better yet — use actual peak-to-peak of several waves
    if len(peaks1) > 3:
        amp1_final = np.mean(s1[peaks1] - np.minimum(s1[peaks1-1], s1[peaks1+1]))  # rough crest-to-trough
    if len(peaks2) > 3:
        amp2_final = np.mean(s2[peaks2] - np.minimum(s2[peaks2-1], s2[peaks2+1]))

    # 3. Time lag via cross-correlation (MOST ACCURATE!)
    from scipy.signal import correlate
    corr = correlate(s1 - s1.mean(), s2 - s2.mean(), mode='full')
    lags = np.arange(-len(s1)+1, len(s1))
    lag_ms = lags[np.argmax(corr)] * (time_ms[1] - time_ms[0])  # convert samples → ms

    # Convert to distance (if you know probe spacing!)
    probe_distance_mm = 500  # TK TODO CHANGE THIS: actual distance between Probe 1 and 2 in mm
    celerity_m_s = probe_distance_mm / 1000 / (abs(lag_ms)/1000) if lag_ms != 0 else np.inf

    # Print beautiful result
    print(f"\nWave Comparison in {start_ms}–{end_ms} ms window:")
    print(f"   Amplitude Probe 1 : {amp1_final:.2f} mm")
    print(f"   Amplitude Probe 2 : {amp2_final:.2f} mm")
    print(f"   Attenuation       : {100*(amp2_final/amp1_final-1):+.1f}%")
    print(f"   Time lag (P1→P2)  : {lag_ms:+.1f} ms")
    print(f"   Celerity          : {celerity_m_s:.2f} m/s" if np.isfinite(celerity_m_s) else "   Celerity          : infinite (no lag)")

    return {
        "amp1": amp1_final,
        "amp2": amp2_final,
        "lag_ms": lag_ms,
        "celerity_m_s": celerity_m_s
    }



def amplitude_overview(processed_dfs, window_ms):
    start_ms, end_ms = window_ms
    results = []

    print(f"\nWAVE TANK AMPLITUDE ANALYSIS — Window: {start_ms}–{end_ms} ms")
    print("="*110)
    print(f"{'File':<35} {'P1':>6} {'P2':>6} {'P3':>6} {'P4':>6}  {'P2/P1':>7} {'P4/P3':>7}  {'Lag12':>6}  Verdict")
    print("-"*110)

    for path, df in processed_dfs.items():
        try:
            window = df.loc[start_ms:end_ms]
        except:
            window = df

        amps = {}
        for i in range(1,5):
            col = f"eta_{i}"
            if col not in window.columns:
                amps[i] = np.nan
                continue
            s = window[col].dropna()
            if len(s) < 50:
                amps[i] = np.nan
                continue
            amp = (np.percentile(s, 99) - np.percentile(s, 1)) / 2
            amps[i] = round(amp, 2)

        # Ratios
        r21 = round(amps.get(2,0) / amps.get(1,1), 3) if amps.get(1) else np.nan
        r43 = round(amps.get(4,0) / amps.get(3,1), 3) if amps.get(3) else np.nan

        # Time lag P1→P2 via cross-correlation
        from scipy.signal import correlate
        s1 = window["eta_1"].dropna().values if "eta_1" in window.columns else np.array([])
        s2 = window["eta_2"].dropna().values if "eta_2" in window.columns else np.array([])
        lag_ms = 0
        if len(s1)>100 and len(s2)>100:
            corr = correlate(s1 - s1.mean(), s2 - s2.mean(), mode='full')
            lag_samples = np.argmax(corr) - (len(s1) - 1)
            dt_ms = np.median(np.diff(window.index))  # ms per sample
            lag_ms = round(lag_samples * dt_ms, 0)

        # Physical verdict
        if 0.8 <= r21 <= 1.25 and abs(lag_ms) > 50:
            verdict = "Normal propagation"
        elif abs(r21 - 1) > 0.4:
            verdict = "P2 CALIBRATION ERROR"
        elif abs(lag_ms) < 30:
            verdict = "P1/P2 too close?"
        else:
            verdict = "OK"

        if abs(r43 - 1) > 0.2:
            verdict += " | P3/P4 MISMATCH!"

        filename = Path(path).name[:34]
        print(f"{filename:<35} "
              f"{amps.get(1,'—'):>6} "
              f"{amps.get(2,'—'):>6} "
              f"{amps.get(3,'—'):>6} "
              f"{amps.get(4,'—'):>6}  "
              f"{r21:>7} "
              f"{r43:>7}  "
              f"{lag_ms:>+6.0f}ms  {verdict}")

        results.append({**amps, "r21": r21, "r43": r43, "lag_ms": lag_ms, "verdict": verdict})

    print("="*110)
    return pd.DataFrame(results)



def full_tank_diagnostics(processed_dfs,window_ms):
    start_ms, end_ms = window_ms
    results = []

    print(f"\nWAVE TANK DIAGNOSTIC — Window: {start_ms}–{end_ms} ms")
    print("="*120)
    print(f"{'File':<32} {'P1':>6} {'P2':>6} {'P3':>6} {'P4':>6}  "
          f"{'P2/P1':>6} {'P3/P2':>7} {'Lag12':>6} {'Lag23':>6}  Verdict")
    print("-"*120)

    for path, df in processed_dfs.items():
        try:
            w = df.loc[start_ms:end_ms]
        except:
            w = df

        amps = {}
        for i in range(1,5):
            col = f"eta_{i}"
            if col not in w.columns: 
                amps[i] = np.nan
                continue
            s = w[col].dropna()
            if len(s) < 100:
                amps[i] = np.nan
                continue
            amp = (np.percentile(s, 99) - np.percentile(s, 1)) / 2
            amps[i] = round(amp, 2)

        # Ratios
        r21 = round(amps.get(2,0) / amps.get(1,1), 3) if amps.get(1) else np.nan
        r32 = round(amps.get(3,0) / amps.get(2,1), 3) if amps.get(2) else np.nan

        # Time lags via cross-correlation
        def get_lag(col_a, col_b):
            if col_a not in w or col_b not in w: return np.nan
            a = w[col_a].interpolate().values
            b = w[col_b].interpolate().values
            if len(a) < 100: return np.nan
            corr = correlate(a - a.mean(), b - b.mean(), mode='full')
            lag = np.argmax(corr) - (len(a) - 1)
            dt = np.median(np.diff(w.index))
            return round(lag * dt, 0)

        lag12 = get_lag("eta_1", "eta_2")   # P1 → P2: 30 cm
        lag23 = get_lag("eta_2", "eta_3")   # P2 → P3: 3.0 m

        # Physical checks
        verdict = []

        # P1 → P2 (30 cm)
        if abs(lag12) > 50 and 0.7 <= r21 <= 1.3:
            celerity = 0.30 / (abs(lag12)/1000) if lag12 != 0 else np.inf
            verdict.append(f"Normal (c={celerity:.1f}m/s)")
        elif abs(r21 - 1) > 0.5:
            verdict.append("P2 CALIBRATION ERROR")

        # P2 → P3/P4 (3 m) — expect damping
        if r32 < 0.85:
            verdict.append(f"Strong damping ({r32:.2f})")
        elif r32 > 1.1:
            verdict.append("Amplification? Check setup")

        # P3 vs P4 should be almost identical
        r43 = amps.get(4,0)/amps.get(3,1) if amps.get(3) else np.nan
        if abs(r43 - 1) > 0.15:
            verdict.append("P3≠P4 → sensor drift")

        verdict_str = " | ".join(verdict) if verdict else "OK"

        filename = Path(path).name[:31]
        print(f"{filename:<32} "
              f"{amps.get(1,'—'):>6} "
              f"{amps.get(2,'—'):>6} "
              f"{amps.get(3,'—'):>6} "
              f"{amps.get(4,'—'):>6}  "
              f"{r21:>6} "
              f"{r32:>7} "
              f"{lag12:>+5.0f}ms "
              f"{lag23:>+5.0f}ms  {verdict_str}")

        results.append({
            "file": Path(path).name,
            "amp1": amps.get(1), "amp2": amps.get(2), "amp3": amps.get(3), "amp4": amps.get(4),
            "r21": r21, "r32": r32, "lag12_ms": lag12, "lag23_ms": lag23,
            "verdict": verdict_str
        })

    print("="*120)
    return pd.DataFrame(results)




def _to_scalar_numeric(v):
    """Normalize a metadata cell to a single float or np.nan.

    - If v is a pd.Series/list/ndarray, take the first element.
    - Coerce to numeric, return float or np.nan.
    """
    # If it's a pandas Series or list-like, take first element
    if isinstance(v, (pd.Series, list, tuple, np.ndarray)):
        # if empty -> NaN
        if len(v) == 0:
            return np.nan
        v = v[0]
    # coerce to numeric
    try:
        return float(pd.to_numeric(v, errors="coerce"))
    except Exception:
        return np.nan


def _safe_round_ratio(a, b):
    """Return  a/b or np.nan if invalid."""
    try:
        a = float(a)
        b = float(b)
    except Exception:
        return np.nan
    if np.isnan(a) or np.isnan(b) or b == 0:
        return np.nan
    return a / b

def newtons_metode():
    
    return

from scipy.optimize import brentq
def k_from_omega(omega, g=9.81, H=0.580):
    f = lambda k: g*k*np.tanh(k*H) - omega**2
    k0 = omega**2/g                 # deep‑water guess
    k1 = omega/np.sqrt(g*H)         # shallow‑water guess
    a, b = min(k0, k1)*0.1, max(k0, k1)*10
    while f(a)*f(b) > 0:
        a, b = a/2, b*2
    return brentq(f, a, b)


from scipy.optimize import brentq
def calculate_wavenumber(freq, H):
    """Tar inn frekvens og høyde
    bruker BRENTQ fra scipy
    """
    g = 9.81
    period = 1/freq
    omega = 2*np.pi/period
    f = lambda k: g*k*np.tanh(k*H) - omega**2
    k0 = omega**2/g #deep water guess
    k1 = omega/np.sqrt(g*H) #shallow water guess
    a, b = min(k0, k1)*0.1, max(k0, k1)*10
    while f(a)*f(b) >0:
        a, b = a/2, b*2
    
    return brentq(f, a, b)
    

def wind_damping_analysis(meta_df):
    """
    Full analysis of wave damping (P3/P2) vs wind condition.
    Returns a DataFrame of results.
    """
    results = []
    meta_sel = meta_df.copy()

    print("WAVE DAMPING vs WIND CONDITION")
    wind_groups = {"full": [], "no": [], "lowest": [], "other": []}

    for idx, row in meta_sel.iterrows():
        #metarows = meta_sel[meta_sel["path"] == path]


        wind = str(row.get("WindCondition", "")).lower().strip()
        wind = wind if wind in ["full", "no", "lowest"] else "other"

        # extract amplitudes as scalars
        P1 = _to_scalar_numeric(row.get("Probe 1 Amplitude"))
        P2 = _to_scalar_numeric(row.get("Probe 2 Amplitude"))
        P3 = _to_scalar_numeric(row.get("Probe 3 Amplitude"))
        P4 = _to_scalar_numeric(row.get("Probe 4 Amplitude"))

        # compute ratios 
        P2toP1 = _safe_round_ratio(P2, P1)
        P3toP2 = _safe_round_ratio(P3, P2)
        P4toP3 = _safe_round_ratio(P4, P3)
        
        noWaveRun =  _to_scalar_numeric(row.get("WaveAmplitudeInput [Volt]"))
        if pd.isna(noWaveRun): 
            #print("NO WAVEINPUT")
            continue
        
        # verdict rules (guard against NaN comparisons)
        verdict = []
        if P2toP1 is np.nan or P2toP1 is None:
            verdict.append("P2/P1?")   # cannot evaluate
        else:
            if not (0.8 <= P2toP1 <= 1.3):
                verdict.append("?")

        if P3toP2 is not np.nan and P3toP2 is not None:
            if P3toP2 > 1.1:
                verdict.append("Amplification!")
            wind_groups[wind].append(P3toP2)
        else:
            # keep consistent grouping even if NaN: do not append
            pass

        if P4toP3 is not np.nan and P4toP3 is not None:
            if abs(P4toP3 - 1) > 0.15:
                verdict.append("P3≠P4")

        wind_label = {"full":"full", "no":"no", "lowest":"lowest", "other":"??"}.get(wind, wind.upper())

        results.append({
            "path": row["path"],
            "WindCondition": wind_label,
            "Probe 1 Amplitude": P1, "Probe 2 Amplitude": P2, "Probe 3 Amplitude": P3, "Probe 4 Amplitude": P4,
            "P2/P1": P2toP1, "P3/P2": P3toP2, "P4/P3": P4toP3
        })

    # SUMMARY BY WIND CONDITION
    print("\nDAMPING SUMMARY :")
    print("-" * 50)
    for w in ["no", "lowest", "full"]:
        ratios = [r for r in wind_groups[w] if not (r is np.nan)]
        if ratios:
            mean_ratio = np.mean(ratios)
            std_ratio = np.std(ratios)
            print(f"{w.upper():<8} → P3/P2 = {mean_ratio:.3f} ± {std_ratio:.3f}  (n={len(ratios)} runs)")
        else:
            print(f"{w.upper():<8} → no valid runs")

    return pd.DataFrame(results)


def probe_comparisor(meta_df):
    dataf = meta_df.copy()

    for idx, row, in dataf.iterrows():
        P1 = row["Probe 1 Amplitude"]
        P2 = row["Probe 2 Amplitude"]
        P3 = row["Probe 3 Amplitude"]
        P4 = row["Probe 4 Amplitude"]
        
        if P1 != 0:
            dataf.at[idx, "P2/P1"] = P2/P1
            
        if P2 != 0:
            dataf.at[idx, "P3/P2"] = P3/P2
        
        if P3 != 0:
            dataf.at[idx, "P4/P3"] = P4/P3
          

    return dataf
    
































