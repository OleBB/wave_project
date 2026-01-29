#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 17:18:03 2025

@author: ole
"""
from pathlib import Path
import pandas as pd
import numpy as np
from wavescripts.data_loader import update_processed_metadata
from scipy.signal import find_peaks
from scipy.signal import welch
from scipy.optimize import brentq

PROBES = ["Probe 1", "Probe 2", "Probe 3", "Probe 4"]

def find_wave_range(
    df,
    meta_row,  # metadata for selected files
    data_col,
    detect_win=1,
    range_plot: bool = False,
):
    
    if (meta_row["WindCondition"]) == "full":
        detect_win = 15
    if (meta_row["WindCondition"]) == "low":
        detect_win = 10
    if (meta_row["WindCondition"]) == "low":
        detect_win = 1

    # ==========================================================
    # 1. smoothe signalet med moving average vindu: detect_win
    # ==========================================================
    signal_smooth = (
        df[data_col]
        .rolling(window=detect_win, center=True, min_periods=1)
        .mean()
        .bfill().ffill()
        .values
    )

    debug_info = {
        "baseline_mean": None,
        "baseline_std": None,
        "first_motion_idx": None,
        "samples_per_period": None,
        "detected_peaks": None,
        "ramp_found": None,
        "ramp_length_peaks": None,
        "keep_periods_used": None,
    }

    # ─────── finne tidsstegene ─────── 
    dt = (df["Date"].iloc[1] - df["Date"].iloc[0]).total_seconds()
    Fs = 1.0 / dt

    # ─────── hente ut input-frekvens ─────── TK bytte ut med ekte frekven senere?
    input_freq = meta_row["WaveFrequencyInput [Hz]"] if isinstance(meta_row, pd.Series) else meta_row["WaveFrequencyInput [Hz]"].iloc[0]
    importertfrekvens = float(input_freq)
    if pd.isna(input_freq):
        print("no freq found, assuming no wave")
        good_start_idx = 0
        good_end_idx = len(df)
        debug_info = None
        return good_start_idx, good_end_idx, debug_info
        
    samples_per_period = int(round(Fs / importertfrekvens))
    
    # ─────── velge antall perioder ─────── 
    input_periods = (meta_row["WavePeriodInput"])
    keep_periods= round((input_periods-13)*1.0) #trekke fra perioder, -per15- er det bare 4 gode, mens på -per40- per er ish 30 gode. TK todo velge en bedre skalering
    keep_seconds= keep_periods/input_freq
    keep_idx = keep_seconds*250 # 1 sek = 250 målinger
    good_range = keep_idx

    # MANUELL CALCULERING for 1.3 Hz
    P1amp01frwq13eyeball = 4500
    P2handcalc = P1amp01frwq13eyeball+100 #tidligere: 62.5 fra 250målinger på ett sekund, ganget et kvart sekund, estimert reisetid for bølgen på 1.3hz  
    P3handcalc = P2handcalc+1700 #tidligere: en 1.3hz gir periode på 700idx? 250 målinger per sek
    
    P1amp01f065eyeball = 3950
    P2_f065_handcalc = P1amp01f065eyeball+50 #
    P3_f065_handcalc = P2_f065_handcalc+500 #



    # ==========================================================
    #  1.b Start og slutt på signalet tilpasset innkommende bølge og vindforhold
    # ==========================================================
    """ELIF RETURN SNARVEI"""
    if input_freq == 1.3:
        if data_col == "Probe 1":
            good_start_idx = P1amp01frwq13eyeball 
            good_end_idx = good_start_idx+keep_idx
                #return good_start_idx, good_end_idx, debug_info
        elif data_col == "Probe 2" : 
            good_start_idx = P2handcalc
            good_end_idx = P2handcalc + keep_idx
                #return good_start_idx, good_end_idx, debug_info
        elif data_col == "Probe 3" : 
            good_start_idx = P3handcalc
            good_end_idx = good_start_idx + keep_idx
        elif data_col == "Probe 4" : 
            good_start_idx = P3handcalc
            good_end_idx = good_start_idx + keep_idx
                # return good_start_idx, good_end_idx, debug_info
    if input_freq == 0.65:
        if data_col == "Probe 1": 
            good_start_idx = P1amp01f065eyeball 
            good_end_idx = good_start_idx+keep_idx
                #return good_start_idx, good_end_idx, debug_info
        elif data_col == "Probe 2" : 
             good_start_idx = P2_f065_handcalc
             good_end_idx = good_start_idx + keep_idx
                #return good_start_idx, good_end_idx, debug_info
        elif data_col == "Probe 3" : 
            good_start_idx = P3_f065_handcalc
            good_end_idx = good_start_idx + keep_idx
        elif data_col == "Probe 4" : 
            good_start_idx = P3_f065_handcalc
            good_end_idx = good_start_idx + keep_idx
                # return good_start_idx, good_end_idx, debug_info
    #import sys; print('exit'); sys.exit()
     
    
    #fullpanel-fullwind-amp02-freq13- correct @5780
    # no panel, amp03, freq0650: 2300? probe=??
    #fullpanel-fullwind-amp01-freq0650-per15-probe3: 4000 korrekt
    
    """
    Først: sjekke paneltilstand:
        hvis ingen panel
    
    Neste: sjekke vindforhold: 
        hvis sterk vind
        
    Hvis lav frekvens: da er det kortere (nesten ingen) ramp.
    
    Ramp må tape for høyeste peaks, i hvertfall når panel
        
    Så, enkelt basere probe 2 på 1 , og 34 på 2?
    """
    
    """elif (meta_row["WindCondition"]) == "lowest" and input_freq == 1.3:
        print('lowestwind og 1.3')
        if data_col == "Probe 1": 
                good_start_idx = P1amp01frwq13eyeball 
                good_end_idx = good_start_idx+keep_idx
                #return good_start_idx, good_end_idx, debug_info
        elif data_col == "Probe 2" : 
                good_start_idx = P2handcalc
                good_end_idx = P2handcalc + keep_idx
                #return good_start_idx, good_end_idx, debug_info
        elif data_col == "Probe 3" : 
                good_start_idx = P3handcalc
                good_end_idx = P3handcalc + keep_idx
                #return good_start_idx, good_end_idx, debug_info
        elif data_col == "Probe 4" : 
                good_start_idx = P3handcalc
                good_end_idx = P3handcalc + keep_idx
                #return good_start_idx, good_end_idx, debug_info"""

    
    baseline_seconds = 2
    sigma_factor=1.0
    skip_periods=None

    min_ramp_peaks=5
    max_ramp_peaks=15
    max_dips_allowed=2
    min_growth_factor = 1.015
    #import sys; print('exit'); sys.exit()

    # ==========================================================
    # 2. Baseline & first motion (still useful for rough start)
    # ==========================================================
    baseline_samples = int(baseline_seconds * Fs)
    baseline = signal_smooth[:baseline_samples]
    baseline_mean = np.mean(baseline)
    baseline_std = np.std(baseline)
    threshold = baseline_mean + sigma_factor*baseline_std
    
    print('baselines:')
    print(f'_samples: {baseline_samples}, _mean: {baseline_mean}, _seconds {baseline_seconds}, _std {baseline_std}')
    #import sys; print('exit'); sys.exit()
    
    """ærbe
    #print('threshold verdi: ', threshold)
    #print('eexit')
    #print('='*99)
    
    #voltgrense  = meta_row["WaveAmplitudeInput [Volt]"]
    #grense = baseline_mean + (voltgrense*100)/3
    #input volt på 0.1 gir omtrentelig <10mm amplitude.
    #import sys; sys.exit()
    """
    
    above_noise = signal_smooth > threshold
    
    first_motion_idx = np.argmax(above_noise) if np.any(above_noise) else 0
    
    # ==========================================================
    # 3. Peak detection on absolute signal (handles both positive/negative swings)
    # ==========================================================
    # Use prominence and distance tuned to your frequency
    min_distance = max(3, input_periods *0.9 )  # at least 0.9 period apart
    peaks, properties = find_peaks(
        np.abs(signal_smooth),
        distance=min_distance,
        prominence=3 * baseline_std,  # ignore noise peaks
        height=threshold
    )
    
    #TODO
    # ==========================================================
    # ikke i bruk, TK , endret sigmafaktor i staden.
    # ==========================================================
    """
    TODO: LAGE noe som fanger opp de med 15 perioder.
    for der er signalet veldig kort.
    lettest: FANGE OPP DE 10 største bølgene. starte fra den første.

    """
    """
    # ta 10 største, så ta den første, så ta 10 perioder
    numbaofpeaks = len(peaks)
    if meta_row["WavePeriodInput"] <16 and numbaofpeaks >3 :
        
        largest_ampl = np.abs(signal_smooth[peaks])
        kth = 3
        largest_peaks = np.argpartition(peaks, kth)
        good_start_idx = largest_peaks[0]
        good_end_idx = largest_peaks[-1]
        debug_info = None
        print("="*99)
        print(f'LESS THAN 16 periods, choosing largest peaks')
        print("="*99)
        return good_start_idx, good_end_idx, debug_info
    
    """
    #
    # ==========================================================
    #  
    # ==========================================================
    """
    if len(peaks) < min_ramp_peaks + 3:
        print("Not enough peaks detected – falling back to legacy method")
        skip_periods = skip_periods or (5 + 12)
        keep_periods = keep_periods or 5
        good_start_idx = first_motion_idx + int(skip_periods * samples_per_period)
        good_range = int(keep_periods * samples_per_period)
        good_range = min(good_range, len(df) - good_start_idx)
        good_end_idx = good_start_idx + good_range
        return good_start_idx, good_end_idx, {}

    peak_amplitudes = np.abs(signal_smooth[peaks])

    # ==========================================================
    # 4. Ramp-up detection: nearly monotonic increase with dips
    # ==========================================================
    def find_best_ramp(seq, min_len=5, max_len=15, max_dips=2, min_growth=2.0):
        n = len(seq)
        best_start = best_end = -1
        best_len = 0
        best_dips = 99

        for length in range(min_len, min(max_len + 1, n)):
            for start in range(n - length + 1):
                end = start + length - 1
                sub = seq[start:end + 1]

                dips = sum(1 for i in range(1, len(sub)) if sub[i] <= sub[i - 1])
                growth_ok = sub[-1] >= sub[0] * min_growth

                if dips <= max_dips and growth_ok:
                    if length > best_len or (length == best_len and dips < best_dips):
                        best_start, best_end = start, end
                        best_len = length
                        best_dips = dips

        if best_start == -1:
            return None
        return best_start, best_end, seq[best_start:best_end + 1]
    # ==========================================================
    # Kjører "find_best_ramp(...)"
    # ==========================================================
    ramp_result = find_best_ramp(
        peak_amplitudes,
        min_len=min_ramp_peaks,
        max_len=max_ramp_peaks,
        max_dips=max_dips_allowed,
        min_growth=min_growth_factor
    )
    
    
    if ramp_result is None:
        print("No clear ramp-up found – using legacy timing")
        skip_periods = skip_periods or (5 + 12)
        keep_periods = keep_periods or 8
        good_start_idx = first_motion_idx + int(skip_periods * samples_per_period)
    else:
        ramp_start_peak_idx, ramp_end_peak_idx, ramp_seq = ramp_result
        print(f"RAMP-UP DETECTED: {len(ramp_seq)} peaks, "
              f"from {ramp_seq[0]:.2f} → {ramp_seq[-1]:.2f} (x{ramp_seq[-1]/ramp_seq[0]:.1f})")
        # Convert last ramp peak → sample index
        last_ramp_sample_idx = peaks[ramp_end_peak_idx]
        # Stable phase starts right after ramp-up
        good_start_idx = last_ramp_sample_idx + samples_per_period // 4  # small safety margin
        keep_periods = keep_periods or 10

    # Final stable window - nå tar den start+range=end
    good_range     = int(keep_periods * samples_per_period)
    good_start_idx = min(good_start_idx, len(df) - good_range - 1)
    good_range     = min(good_range, len(df) - good_start_idx)
    good_end_idx   = good_start_idx + good_range
    """
    # ==========================================================
    # 5.a Få hjelp av grok
    # ==========================================================
    # for å printe verdiene slik at grok kunne forstå signalet
    # if 'dumped' not in locals():   # only once
        # print("\n=== SIGNAL FOR GROK (downsampled 200:1) ===")
        # print("value")
        # print("\n".join(f"{x:.5f}" for x in df[data_col].values[::200]))  # every 200th point
        # print("=== FULL STATS ===")
        # print(f"total_points: {len(df)}")
        # print(f"dt_sec: {dt:.6f}")
        # print(f"frequency_hz: {importertfrekvens}")
        # print(f"samples_per_period: {samples_per_period}")
        # print(f"baseline_mean: {baseline_mean:.4f}")
        # print(f"baseline_std: {baseline_std:.4f}")
        # print(f"first_motion_idx: {first_motion_idx}")
        # print("=== PEAKS (index, value) ===")
        # peaks_abs = np.abs(signal_smooth[peaks]) if 'peaks' in locals() and len(peaks)>0 else []
        # for i, (pidx, pval) in enumerate(zip(peaks[:30], peaks_abs[:30])):  # max 30 peaks
            # print(f"{pidx:5d} -> {pval:.5f}")
        # if len(peaks) > 30:
            # print("... (more peaks exist)")
        # print("=== END – COPY ALL ABOVE AND SEND TO GROK ===")
        # dumped = True
        #import sys; sys.exit(0)
        
    # ==========================================================
    # 5.b) Plotting – safe version that works with your current plot_ramp_detection
    # ==========================================================
    if range_plot:
        try:
            from wavescripts.plotter import plot_ramp_detection

            # Build kwargs only with arguments your current function actually accepts
            plot_kwargs = {
                "df": df,
                "meta_sel": meta_row,
                "data_col": data_col,
                "signal": signal_smooth,
                "baseline_mean": baseline_mean,
                "threshold": threshold,
                "first_motion_idx": first_motion_idx,
                "good_start_idx": good_start_idx,
                "good_range": good_range,
                "title": f"Smart Ramp Detection – {data_col}"
            }

            # Only add new arguments if we have them and ramp was found
            """
            if 'peaks' in locals() and ramp_result is not None:
                plot_kwargs["peaks"] = peaks
                plot_kwargs["peak_amplitudes"] = peak_amplitudes
                ramp_peak_samples = peaks[ramp_result[0]:ramp_result[1]+1]
                plot_kwargs["ramp_peak_indices"] = ramp_peak_samples
            """
            plot_ramp_detection(**plot_kwargs)

        except Exception as e:
            print(f"Plot failed (will work after you update plotter): {e}")

    debug_info = {
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "first_motion_idx": first_motion_idx,
        "samples_per_period": samples_per_period,
        "detected_peaks": len(peaks),
        #"ramp_found": ramp_result is not None,
        #"ramp_length_peaks": len(ramp_result[2]) if ramp_result else None,
        "keep_periods_used": keep_periods,
    }

    return good_start_idx, good_end_idx, debug_info


# ========================================================== #
# === Make sure stillwater levels are computed and valid === #
# ========================================================== #
def ensure_stillwater_columns(
    dfs: dict[str, pd.DataFrame],
    meta: pd.DataFrame,
) -> pd.DataFrame:
    """
    Computes the true still-water level for each probe using ALL "no wind" runs,
    then copies that value into EVERY row of the metadata (including windy runs).
    Safe to call multiple times.
    """
    probe_cols = [f"Stillwater Probe {i}" for i in range(1, 5)]

    # If all columns exist AND all values are real numbers → we're done
    if all(col in meta.columns for col in probe_cols):
        if meta[probe_cols].notna().all().all():  # every cell has a real number
            print("Stillwater levels already computed and valid → skipping")
            return meta

    print("Computing still-water levels from all 'WindCondition == no' runs...")

    # Find all no-wind runs
    mask = meta["WindCondition"].astype(str).str.strip().str.lower() == "no"
    nowind_paths = meta.loc[mask, "path"].tolist()

    if not nowind_paths:
        raise ValueError("No runs with WindCondition == 'no' found! Cannot compute still water.")

    # Compute median from ALL calm data combined
    stillwater_values = {}
    for i in range(1, 5):
        probe_name = f"probe {i}"  # adjust if your columns are named differently
        all_values = []

        for path in nowind_paths:
            if path in dfs:
                df = dfs[path]
                if probe_name in df.columns:
                    clean = pd.to_numeric(df[probe_name], errors='coerce').dropna()
                    all_values.extend(clean.tolist())
                # Also try other common names just in case
                elif f"Probe {i}" in df.columns:
                    clean = pd.to_numeric(df[f"Probe {i}"], errors='coerce').dropna()
                    all_values.extend(clean.tolist())

        if len(all_values) == 0:
            raise ValueError(f"No valid data found for {probe_name} in any no-wind run!")

        level = np.median(all_values)
        stillwater_values[f"Stillwater Probe {i}"] = float(level)
        print(f"  Stillwater Probe {i}: {level:.3f} mm  (from {len(all_values):,} samples)")

    # Write the same value into EVERY row (this is correct!)
    for col, value in stillwater_values.items():
        meta[col] = value

    # Make sure we can save correctly
    if "PROCESSED_folder" not in meta.columns:
        if "experiment_folder" in meta.columns:
            meta["PROCESSED_folder"] = "PROCESSED-" + meta["experiment_folder"].iloc[0]
        else:
            raw_folder = Path(meta["path"].iloc[0]).parent.name
            meta["PROCESSED_folder"] = f"PROCESSED-{raw_folder}"

    # Save to disk
    update_processed_metadata(meta, force_recompute=False)
    print("Stillwater levels successfully saved to meta.json for ALL runs")

    return meta



def compute_amplitudes(processed_dfs: dict, meta_row: pd.DataFrame) -> pd.DataFrame:
    """Compute wave amplitudes from np.percentile"""
    records = []
    for path, df in processed_dfs.items():
        subset_meta = meta_row[meta_row["path"] == path]
        for _, row in subset_meta.iterrows():
            row_out = {"path": path}
            for i in range(1, 5):
                amplitude = _extract_probe_amplitude(df, row, i)
                if amplitude is not None:
                    row_out[f"Probe {i} Amplitude"] = amplitude
            records.append(row_out)
    return pd.DataFrame.from_records(records)

def compute_amplitudes_from_psd(f, pxx, target_freq, window=0.5):
    """Hent amplituden fra PSD ved gitt frekvens"""
    mask = (f >= target_freq - window) & (f <= target_freq + window)
    # psd_at_freq = pxx[mask].max()
    deltaf = f[1]-f[0] #frequency resolution
    # amplitude = np.sqrt(2 * psd_at_freq * deltaf)
    var = pxx[mask].sum() * deltaf
    sigma = np.sqrt(var)
    amplitude = np.sqrt(2) * sigma
    return amplitude

def compute_amplitudes_from_fft(fft_freqs, fft_magnitude, target_freq, window=0.5):
    """
    Extract amplitude from FFT at a given frequency.
    
    Args:
        fft_freqs: Frequency array from FFT
        fft_magnitude: Magnitude of FFT (already normalized to amplitude)
        target_freq: Target frequency (Hz)
        window: Frequency window around target (Hz). Default 0.5 Hz.
    
    Returns:
        Amplitude at target frequency
    """
    mask = (fft_freqs >= target_freq - window) & (fft_freqs <= target_freq + window)
    
    if not mask.any():
        # No frequencies in range - fallback to closest
        closest_idx = np.argmin(np.abs(fft_freqs - target_freq))
        return fft_magnitude[closest_idx]
    
    # Return the maximum amplitude in the window (peak)
    amplitude = fft_magnitude[mask].max()
    
    return amplitude


def compute_psd_with_amplitudes(processed_dfs: dict, meta_row: pd.DataFrame, fs: float = 250, debug:bool=False) -> Tuple[dict, list]:
    """Compute Power Spectral Density for each probe."""
    psd_dict = {}
    amplitude_records = []
    for path, df in processed_dfs.items():
        subset_meta = meta_row[meta_row["path"] == path ]
        for _, row in subset_meta.iterrows():
            row_out = {"path": path}
            
            freq = row["WaveFrequencyInput [Hz]"]
            # Validate frequency
            if pd.isna(freq) or freq <= 0:
                if debug:
                    print(f"Advarsel: Ingen Input frequency {freq} for {path}, skipping")
                continue
            
            psd_df = None
            desired_resolution = 0.125 #Hz per steg i Psd'en.
            npersegment = int(fs/desired_resolution)
            for i in range(1, 5):
                signal = _extract_probe_signal(df, row, i)
                nperseg = max(1, min(npersegment, len(signal)))
                    
                if signal is not None:

                    # nperseg = npersegment #har noen mye kortere signaler
                    noverlap = nperseg // 2  # or int(0.75 * nperseg)
                    
                    f, pxx = welch(
                        signal, fs=fs,
                        window='hann',
                        nperseg=nperseg,
                        noverlap=noverlap,
                        detrend='constant',
                        scaling='density'
                    )
                    if psd_df is None:
                        psd_df = pd.DataFrame(index=f)
                        psd_df.index.name = "Frequencies"  
                    psd_df[f"Pxx {i}"] = pxx
                    # - - - amplitude
                    amplitude = compute_amplitudes_from_psd(f, pxx, freq)
                    if debug:
                        print("amplitden inni PSD loopen er ", amplitude)
                    row_out[f"Probe {i} Amplitude (PSD)"] = amplitude
            if psd_df is not None:
                psd_dict[path] = psd_df
            amplitude_records.append(row_out)

    if debug:
        print(f"=== PSD Complete: {len(amplitude_records)} records ===\n")
    return psd_dict, amplitude_records

def compute_fft_with_amplitudes(processed_dfs: dict, meta_row: pd.DataFrame, fs: float = 250, debug:bool=False) -> Tuple[dict, list]:
    """Compute FFT for each probe. and calculate amplitude"""
    fft_dict = {}
    amplitude_records = []
    for path, df in processed_dfs.items():
        subset_meta = meta_row[meta_row["path"] == path]
        for _, row in subset_meta.iterrows():
            row_out = {"path": path}
            
            freq = row["WaveFrequencyInput [Hz]"]
            # Validate frequency
            if pd.isna(freq) or freq <= 0:
                if debug:
                    print(f"advarsel: No input frequency {freq} for {path}, skipping")
                continue
            
            fft_df = None
            series_list = []
            for i in range(1, 5):
                
                if (signal := _extract_probe_signal(df, row, i) )is not None: #Walrus operator := sjekker om den ikke er None
                    N = len(signal)
                    fft_vals = np.fft.rfft(signal)  # rfft only returns positive frequencies
                    fft_freqs = np.fft.rfftfreq(N, d=1/fs)
                    
                    amplitudar = np.abs(fft_vals) * 2 / N
                    amplitudar[0] = amplitudar[0] / 2 #0hz should not be doubled
                    
                    if N % 2 == 0:
                        amplitudar[-1] = amplitudar[-1] / 2 #if N is even, Nyquist freq should not be doubled
                    
                    series_list.append(pd.Series(amplitudar, index=fft_freqs, name=f"FFT {i}"))
                    # amplitude entall
                    amplitude = compute_amplitudes_from_fft(fft_freqs, amplitudar, freq)
                    row_out[f"Probe {i} Amplitude (FFT)"] = amplitude
                    
                    if debug:
                        print(f"  Probe {i}: Amplitude = {amplitude:.3f} mm at {freq:.3f} Hz")
            
            if series_list:  # Only create DataFrame if we have data
                fft_df = pd.concat(series_list, axis=1)
                fft_df = fft_df.sort_index()  # sorterer
                fft_df.index.name = "Frequencies"
                fft_dict[path] = fft_df
            amplitude_records.append(row_out)
    if debug:
        print(f"=== FFT Complete: {len(amplitude_records)} records ===\n")
    return fft_dict, amplitude_records


    
    
def _extract_probe_signal(df: pd.DataFrame, row: pd.Series, probe_num: int) -> np.ndarray | None:
    """Extract and validate signal data for a specific probe."""
    col = f"eta_{probe_num}"
    start_val = row.get(f"Computed Probe {probe_num} start")
    end_val = row.get(f"Computed Probe {probe_num} end")
    
    if pd.isna(start_val) or pd.isna(end_val) or col not in df.columns:
        return None
    
    try:
        s_idx = max(0, int(start_val))
        e_idx = min(len(df) - 1, int(end_val))
    except (TypeError, ValueError):
        return None
    
    if s_idx >= e_idx:
        return None
    
    signal = df[col].iloc[s_idx:e_idx+1].dropna().to_numpy()
    return signal if signal.size > 0 else None

def _extract_probe_amplitude(df: pd.DataFrame, row: pd.Series, probe_num: int) -> float | None:
    """Extract amplitude for a specific probe."""
    signal = _extract_probe_signal(df, row, probe_num)
    if signal is None:
        return None
    return float((np.percentile(signal, 99.5) - np.percentile(signal, 0.5)) / 2.0)


def calculate_wavenumbers(frequencies, heights):
    """Tar inn frekvens og høyde
    bruker BRENTQ fra scipy
    """
    freq = np.asarray(frequencies)
    H = np.broadcast_to(np.asarray(heights), freq.shape)
    k = np.zeros_like(freq, dtype=float)
    
    valid = freq>0
    i_valid = np.flatnonzero(valid)
    if i_valid.size ==0:
        return k
    g = 9.81
    for idx in i_valid:
        fr = freq.flat[idx]
        h = H.flat[idx]/1000.0 #konverter fra millimeter
        omega = 2 * np.pi * fr
        
        def disp(k_wave):
            return g*k_wave* np.tanh(k_wave * h) - omega**2
        
        k_deep = omega**2 / g
        k_shallow = omega / np.sqrt(g * h) if h >0 else k_deep
        
        a = min(k_deep, k_shallow) *0.1
        b = max(k_deep, k_shallow) *10
        
        fa = disp(a)
        fb = disp(b)
        while fa * fb > 0:
            a /= 2
            b *= 2
            fa = disp(a)
            fb = disp(b)
        k.flat[idx] = brentq(disp, a, b)
    return k

def calculate_celerity(wavenumbers,heights):
    k = np.asarray(wavenumbers)
    H = np.broadcast_to(np.asarray(heights), k.shape)
    c = np.zeros_like(k,dtype=float)
    
    g = 9.81
    sigma = 0.074 #ved 20celcius
    rho = 1000 #10^3 kg/m^3
    
    c = np.sqrt( g/ k * np.tanh(k*H))
    return c


def calculate_windspeed(windcond: pd.Series) -> pd.Series:
    # normalize labels to avoid casing/whitespace issues
    wc = windcond.astype(str).str.lower().str.strip()
    speed_map = {
        "no": 0.0,
        "lowest": 3.8,
        "low": 3.8,   # if you also use "low"
        "full": 5.8,
    }
    return wc.map(speed_map)


def calculate_wavedimensions(k: pd.Series, 
                             H: pd.Series, 
                             PC: pd.Series,
                             P2A: pd.Series,
                             ) -> pd.DataFrame:
    panel_length_map = {
            "purple": 1.048,
            "yellow": 1.572,
            "full": 2.62,
            "reverse": 2.62,
        }
    
    # Align (inner) on indices to keep only rows present in both k and H
    k_aligned, H_aligned = k.align(H, join="inner")
    idx = k_aligned.index
    P2A_aligned = None if P2A is None else P2A.reindex(idx)
    PC_aligned = None if PC is None else PC.reindex(idx)

    k_arr = k_aligned.to_numpy(dtype=float)
    Hm = H_aligned.to_numpy(dtype=float)/1000.0 #fra millimeter
     
    valid_k = k_arr > 0.0 # Mask for valid k values
    g = 9.81
    
    kH = np.full_like(k_arr, np.nan, dtype=float)
    kH[valid_k] = k_arr[valid_k] * Hm[valid_k]
     
    tanhkh = np.full_like(k_arr, np.nan, dtype=float)
    tanhkh[valid_k] = np.tanh(kH[valid_k])
    
    wavelength = np.full_like(k_arr, np.nan, dtype=float)
    wavelength[valid_k] = 2.0 * np.pi / k_arr[valid_k]
    
    L_arr = np.full_like(k_arr, np.nan, dtype=float)
    if PC_aligned is not None:
        pc_norm = PC_aligned.astype(str).str.strip().str.lower()
        L_char = pc_norm.map(panel_length_map)
        L_arr = L_char.to_numpy(dtype=float)
    else:
        print('PanelCondition missing - no kL to calculate')
    kL = np.full_like(k_arr, np.nan, dtype=float)
    mask_kL = valid_k & np.isfinite(L_arr)
    kL[mask_kL] = k_arr[mask_kL] * L_arr[mask_kL]
    
    ak = np.full_like(k_arr, np.nan, dtype=float)
    if P2A_aligned is not None:
        a_arr = P2A_aligned.to_numpy(dtype=float) / 1000.0  #fra millimeter
        ak[valid_k] = a_arr[valid_k] * k_arr[valid_k]
    else:
        print('No probe 2 amplitude - no ak to calculate')
    
    c = np.full_like(k_arr, np.nan, dtype=float)
    c[valid_k] = np.sqrt((g / k_arr[valid_k]) * tanhkh[valid_k])
    
    out = pd.DataFrame(
        {"Wavelength": wavelength, 
         "kL": kL, 
         "ak": ak, 
         "kH": kH, 
         "tanh(kH)": tanhkh, 
         "Celerity": c,
             }, 
            index=idx
        )
    return out


def remove_outliers():
    #lag noe basert på steepness, kanskje tilogmed ak. Hvis ak er for bratt
    # og datapunktet for høyt, så må den markeres, og så fjernes.
    #se Karens script
    return


"claude=============="
def _extract_stillwater_levels(meta_full: pd.DataFrame, debug: bool) -> dict:
    """Extract stillwater levels from metadata."""
    stillwater = {}
    for i in range(1, 5):
        val = meta_full[f"Stillwater Probe {i}"].iloc[0]
        if pd.isna(val):
            raise ValueError(f"Stillwater Probe {i} is NaN!")
        stillwater[i] = float(val)
        if debug:
            print(f"  Stillwater Probe {i} = {val:.3f} mm")
    return stillwater


def _zero_and_smooth_signals(
    dfs: dict, 
    meta_sel: pd.DataFrame, 
    stillwater: dict, 
    win: int,
    debug: bool
) -> dict[str, pd.DataFrame]:
    """Zero signals using stillwater and add moving averages."""
    processed_dfs = {}
    for _, row in meta_sel.iterrows():
        path = row["path"]
        if path not in dfs:
            print(f"Warning: File not loaded: {path}")
            continue

        df = dfs[path].copy()
        for i in range(1, 5):
            probe_col = f"Probe {i}"
            if probe_col not in df.columns:
                print(f"  Missing column {probe_col} in {Path(path).name}")
                continue

            eta_col = f"eta_{i}"
            df[eta_col] = -(df[probe_col] - stillwater[i])
            df[f"{probe_col}_ma"] = df[eta_col].rolling(window=win, center=False).mean()
            
            if debug:
                print(f"  {Path(path).name:35} → eta_{i} mean = {df[eta_col].mean():.4f} mm")
        
        processed_dfs[path] = df
    
    return processed_dfs


def _find_wave_ranges(
    processed_dfs: dict,
    meta_sel: pd.DataFrame,
    win: int,
    range_plot: bool,
    debug: bool
) -> pd.DataFrame:
    """Find wave ranges for all probes."""
    for idx, row in meta_sel.iterrows():
        path = row["path"]
        df = processed_dfs[path]
      
        for i in range(1, 5):
            probe = f"Probe {i}"
            start, end, debug_info = find_wave_range(
                df, row, data_col=probe, detect_win=win, range_plot=range_plot
            )
            meta_sel.loc[idx, f'Computed Probe {i} start'] = start
            meta_sel.loc[idx, f'Computed Probe {i} end'] = end
        
        if debug and start:
            print(f'start: {start}, end: {end}, debug: {debug_info}')
    
    return meta_sel


def _update_all_metrics(
    processed_dfs: dict,
    meta_sel: pd.DataFrame,
    stillwater: dict,
    amplitudes_psd_df: pd.DataFrame,
    amplitudes_fft_df: pd.DataFrame,
) -> pd.DataFrame:
    """Kalkuler og oppdater all computed metrics in metadata."""
    meta_indexed = meta_sel.set_index("path")
    
    # Amplitudes from np.percentile
    amplitudes = compute_amplitudes(processed_dfs, meta_sel)
    amp_cols = [f"Probe {i} Amplitude" for i in range(1, 5)]
    meta_indexed.update(amplitudes.set_index("path")[amp_cols])
    
    #Amplitudes from psd
    amp_psd_cols = [f"Probe {i} Amplitude (PSD)" for i in range(1, 5)]
    meta_indexed[amp_psd_cols] = amplitudes_psd_df.set_index("path")[amp_psd_cols]  # Direct assignment
    
    #Amplitudes from FFT
    amp_fft_cols = [f"Probe {i} Amplitude (FFT)" for i in range(1, 5)]
    meta_indexed[amp_fft_cols] = amplitudes_fft_df.set_index("path")[amp_fft_cols]  # Direct assignment

    
    # Wavenumbers
    meta_indexed["Wavenumber"] = calculate_wavenumbers(
        meta_indexed["WaveFrequencyInput [Hz]"], 
        meta_indexed["WaterDepth [mm]"]
    )
    
    # Wave dimensions
    wave_dims = calculate_wavedimensions(
        k=meta_indexed["Wavenumber"],
        H=meta_indexed["WaterDepth [mm]"],
        PC=meta_indexed["PanelCondition"],
        P2A=meta_indexed["Probe 2 Amplitude"],
    )
    meta_indexed[["Wavelength", "kL", "ak", "kH", "tanh(kH)", "Celerity"]] = wave_dims
    
    # Windspeed
    meta_indexed["Windspeed"] = calculate_windspeed(meta_indexed["WindCondition"])
    
    # Add stillwater columns if missing
    for i in range(1, 5):
        col = f"Stillwater Probe {i}"
        if col not in meta_indexed.columns:
            meta_indexed[col] = stillwater[i]
    
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
    fs = 250 #samplerate -foreløpig hardkodet mange steder.
    # 0. unpack processvariables
    # overordnet = processvariables.get("overordnet", {})
    prosessering = processvariables.get("prosessering", {})
    
    debug = prosessering.get("debug", False)
    win = prosessering.get("smoothing window", 1)
    find_range =prosessering.get("find_range", False)
    range_plot =prosessering.get("range_plot", False)
    force_recompute =prosessering.get("force_recompute", False)
    
    # 1. Ensure stillwater levels are computed
    meta_full = ensure_stillwater_columns(dfs, meta_full)
    stillwater = _extract_stillwater_levels(meta_full, debug)

    # 2. Process dataframes: zero and add moving averages
    processed_dfs = _zero_and_smooth_signals(dfs, meta_sel, stillwater, win, debug)
    
    # 3. Optional: find wave ranges
    if find_range:
        meta_sel = _find_wave_ranges(processed_dfs, meta_sel, win, range_plot, debug)
    
    # 4. a - Compute PSDs and amplitudes from PSD
    psd_dict, amplitudes_from_psd  = compute_psd_with_amplitudes(processed_dfs, meta_sel, fs=fs,debug=debug)
    amplitudes_psd_df = pd.DataFrame(amplitudes_from_psd)
   
    # 4. b - compute FFT and amplitudes from FFT
    fft_dict, amplitudes_from_fft = compute_fft_with_amplitudes(processed_dfs, meta_sel, fs=fs, debug=debug)
    amplitudes_fft_df = pd.DataFrame(amplitudes_from_fft)
    # print("FFT columns:", amplitudes_fft_df.columns.tolist())
    # print("FFT shape:", amplitudes_fft_df.shape)
    # print("FFT head:", amplitudes_fft_df.head())

    # 5. Compute and update all metrics (amplitudes, wavenumbers, dimensions, windspeed)
    meta_sel = _update_all_metrics(processed_dfs, meta_sel, stillwater, amplitudes_psd_df, amplitudes_fft_df)

    # 6. Set output folder and save metadata
    meta_sel = _set_output_folder(meta_sel, meta_full, debug)

    update_processed_metadata(meta_sel, force_recompute=force_recompute)

    if debug:
        print(f"\nProcessing complete! {len(processed_dfs)} files zeroed and ready.")
    
    return processed_dfs, meta_sel, psd_dict, fft_dict

