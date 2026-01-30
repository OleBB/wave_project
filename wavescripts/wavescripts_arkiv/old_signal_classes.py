#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 28 14:53:40 2025

@author: gpt
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from pathlib import Path

class Signal:
    """Single signal object with data and processing methods."""

    def __init__(self, file_path=None, data=None):
        if file_path:
            self.file_path = Path(file_path)
            self.data = self.load_csv(file_path)
        elif data is not None:
            self.file_path = None
            self.data = np.array(data)
        else:
            raise ValueError("Provide file_path or data")
        self.fs = None

    def load_csv(self, file_path):
        df = pd.read_csv(file_path)
        return df.iloc[:, 0].to_numpy()

    def bandpass_filter(self, lowcut, highcut, fs, order=4):
        self.fs = fs
        b, a = butter(order, [lowcut/(fs/2), highcut/(fs/2)], btype='band')
        self.data = filtfilt(b, a, self.data)

    def normalize(self):
        self.data = self.data / np.max(np.abs(self.data))

    def plot(self, title=None, time=None):
        if time is None:
            time = np.arange(len(self.data))
        plt.figure(figsize=(8,4))
        plt.plot(time, self.data)
        plt.title(title if title else "Signal")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.show()

    def compute_fft(self):
        n = len(self.data)
        fft_vals = np.fft.fft(self.data)
        freqs = np.fft.fftfreq(n, d=1/self.fs if self.fs else 1)
        return freqs[:n//2], np.abs(fft_vals[:n//2])


class SignalBatch:
    """Batch processor for multiple signals in a folder."""

    def __init__(self, folder_path):
        self.folder_path = Path(folder_path)
        self.signals = []
        self.load_all()

    def load_all(self):
        csv_files = list(self.folder_path.glob("*.csv"))
        for f in csv_files:
            sig = Signal(f)
            self.signals.append(sig)

    def apply_filter(self, lowcut, highcut, fs, order=4):
        for sig in self.signals:
            sig.bandpass_filter(lowcut, highcut, fs, order=order)

    def normalize_all(self):
        for sig in self.signals:
            sig.normalize()

    def plot_all(self, overlay=False):
        if overlay:
            plt.figure(figsize=(10,5))
            for sig in self.signals:
                plt.plot(sig.data, label=sig.file_path.name if sig.file_path else "signal")
            plt.legend()
            plt.xlabel("Sample")
            plt.ylabel("Amplitude")
            plt.title("Overlayed Signals")
            plt.grid(True)
            plt.show()
        else:
            for sig in self.signals:
                sig.plot(title=sig.file_path.name if sig.file_path else "Signal")

    def compute_fft_all(self):
        results = {}
        for sig in self.signals:
            freqs, mag = sig.compute_fft()
            results[sig.file_path.name if sig.file_path else f"signal_{id(sig)}"] = (freqs, mag)
        return results
