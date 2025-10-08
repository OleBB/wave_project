#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 12:49:18 2025

@author: ole
"""

"""
Convert time series acceleration data to a PSD for use in Random Vibration analysis.
"""

import numpy as np
from scipy.signal import welch, find_peaks
from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt
from scipy import fft
import pandas as pd


def psd_estimator_adaptive(signal, sampling_rate=1000, min_seg=64, max_seg=512):
    # Adaptive segment length based on signal length
    segment_length = min(max(min_seg, len(signal)//8), max_seg)
    overlap_step = segment_length // 2
    results = []
    for i in range(0, len(signal) - segment_length + 1, overlap_step):
        segment = signal[i:i + segment_length]
        f, psd_values = welch(segment, fs=sampling_rate, nperseg=segment_length)
        results.append(psd_values)
    return f, np.mean(results, axis=0)

def psd_estimator_hamming(signal, sampling_rate=1000, segment_length=256, window='hamming'):
    overlap_step = segment_length // 2
    results = []
    for i in range(0, len(signal) - segment_length + 1, overlap_step):
        segment = signal[i:i + segment_length]
        f, psd_values = welch(segment, fs=sampling_rate, window=window, nperseg=segment_length)
        results.append(psd_values)
    return f, np.mean(results, axis=0)

def annot_max(x,y, ax=None):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text= "x={:.3f}, y={:.3f}".format(xmax, ymax)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94,0.96), **kw)
    
def annot_peaks(x:np.array,y:np.array, ax=None, peak_distance=30, y_position_modifier=1):
    yindices, _ = find_peaks(y, distance=peak_distance)
    xmax = x[yindices]
    ymax = y[yindices]
    ymodifier = {k:v for k,v in zip(y, y_position_modifier-minmax_scale(y, feature_range=(0,y_position_modifier)))}
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->", color="k", connectionstyle="arc3,rad=0")
    kw = dict(xycoords='data',textcoords="data",arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    for xmx, ymx in zip(xmax,ymax):
        text= "x={:.3f},\ny={:.3f}".format(xmx, ymx)
        ax.annotate(text, xy=(xmx, ymx), xytext=(xmx*1.1, ymx*(1.5+ymodifier[ymx])), **kw)

"""
# Read in time series csv file
TIME_SERIES_FILE = 'time_series.csv'
"""
#her må jeg putte inn mitt..



dat = pd.read_csv(TIME_SERIES_FILE, skiprows=[1])  # time in secs, accel in g's
sig = np.float64(dat['Accel'])

# Determine sampling frequency
dt = dat['Time'][1] - dat['Time'][0]
fs = 1./dt
N = len(sig)

# Compute the PSD using the hamming window approach
psd_hamming = psd_estimator_hamming(sig, sampling_rate=fs)

# Compute the PSD using the adaptive approach
psd_adaptive = psd_estimator_adaptive(sig, sampling_rate=fs)

# Compute the maximum of the hamming PSD to add to plot
psd_max = np.max(psd_hamming[1])

# Manually calculate the PSD using FFT
xdft = fft.fft(sig)
xdft = xdft[0:int(N/2)+1]
psdx = 1/(fs*N) * np.abs(xdft)**2
psdx[1:-1] = 2*psdx[1:-1]
freq_axis = np.linspace(0, fs/2, num=int(N/2)+1)     # PSD frequency axis

# Plot the time series data
plt.clf()
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 8))
plt.subplots_adjust(hspace=0.25)
ax0.plot(dat['Time'], sig)
ax0.set_xlabel('Time (s)')
ax0.set_ylabel('Accel (g)')
ax0.grid(True)

# Plot both PSD estimations
ax1.semilogy(psd_hamming[0], psd_hamming[1], label='Hamming Window')
ax1.semilogy(psd_adaptive[0], psd_adaptive[1], label='Adaptive Estimation')
ax1.set_xlabel('Frequency (Hz)')
ax1.set_ylabel('PSD (G^2/Hz)')
ax1.grid(True)
ax1.semilogy(freq_axis[:-2], psdx[:-2], label='Manual Using FFT')  # Ignore zero values at highest frequencies

# Add maximum to PSD plot
xmin, xmax = ax1.get_xlim()
ax1.hlines(y=psd_max, xmin=xmin, xmax=xmax, colors='red', linestyles='--', lw=0.5)  # add horiz line at max force
ax1.set_yticks(list(ax1.get_yticks()) + [psd_max])
ticks = ax1.get_yticklabels()
ticks[-1].set_fontweight('bold')
ticks[-1].set_color('red')
ticks[-1].set_fontsize(15)
x = psd_hamming[0]
y = psd_hamming[1]
annot_max(x, y, ax1)      # Add flag to annotate the location of the maximum
#annot_peaks(x, y, ax1, peak_distance=1000, y_position_modifier=100)

#xmax = x[np.argmax(y)]
#ax1.annotate("Frequency={:.4f},\nPSD={:.4f}".format(xmax, psd_max), xy=(xmax, psd_max), xytext=(xmax, psd_max+5),
#             arrowprops=dict(arrowstyle="->", color="k", connectionstyle="arc3,rad=0"))

ax1.legend()
plt.show()

# Write the PSD (Hamming) to csv file
df_psd = pd.DataFrame(data={'Frequency [Hz]': psd_hamming[0], 'PSD [g^2/Hz]': psd_hamming[1]})
df_psd.to_csv(TIME_SERIES_FILE.split('.')[0] + '_psd.csv', index=False)
