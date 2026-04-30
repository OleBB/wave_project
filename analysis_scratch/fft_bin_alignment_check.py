"""
fft_bin_alignment_check.py
==========================
Show — concretely — that the H&G 10-period window aligns the paddle
frequency with FFT bin 10, and that sinc leakage is therefore tiny.

Outputs (to analysis_scratch/):
  fft_bin_alignment_table.csv     — per-run measurement
  fft_bin_alignment.png           — visual: spectrum zoom + sinc curve
  fft_bin_alignment_findings.md   — short, citable summary (written separately)

Math, in 4 lines:
  df            = fs / N                       # FFT bin spacing
  bin10         = 10 * df                      # 10th bin centre (where the
                                               #   paddle should land if the
                                               #   window is 10 periods long)
  offset_bins   = (f_target - bin10) / df      # how far in bin widths
  sinc_atten    = 1 - |sin(pi·offset)/(pi·offset)|   # nearest-bin reading

Run from repo root:  python analysis_scratch/fft_bin_alignment_check.py
"""

from pathlib import Path
import sys, os
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
os.chdir(BASE)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from wavescripts.improved_data_loader import load_analysis_data
from wavescripts.filters import apply_experimental_filters

OUT_DIR = Path("analysis_scratch")
PROCESSED = [
    Path("waveprocessed/PROCESSED-20260326-ProbePos4_31_FPV_2-tett6roof-under9Mooring-height100-lowrange"),
    Path("waveprocessed/PROCESSED-20260327-ProbePos4_31_FPV_2-tett6roof-under9Mooring30-height100-lowrange"),
]
FREQS  = [1.3, 1.4, 1.5, 1.6]
WINDS  = ["no", "full"]
PROBES = ["9373/170", "9373/340", "12400/250"]


def measure_alignment(meta, fft_dict):
    rows = []
    for f_target in FREQS:
        for wind in WINDS:
            sel = apply_experimental_filters(meta, {"filters": {
                "WaveAmplitudeInput [Volt]": 0.2,
                "WaveFrequencyInput [Hz]":   f_target,
                "WindCondition":             wind,
                "PanelCondition":            "full",
            }})
            for _, row in sel.iterrows():
                path = row["path"]
                if path not in fft_dict:
                    continue
                df_fft = fft_dict[path]
                for probe in PROBES:
                    col = f"FFT {probe} complex"
                    if col not in df_fft:
                        continue
                    ser = df_fft[col].dropna()
                    if ser.empty:
                        continue
                    freq_bins = ser.index.values
                    N         = len(freq_bins)
                    dfreq     = abs(freq_bins[1] - freq_bins[0])      # bin width = fs/N
                    bin10     = 10.0 * dfreq                          # 10th bin centre
                    offset    = (f_target - bin10) / dfreq            # offset in bin widths
                    sinc_at   = 1.0 - abs(np.sinc(offset))            # nearest-bin loss
                    rows.append({
                        "f_target":      f_target,
                        "wind":          wind,
                        "probe":         probe,
                        "N":             N,
                        "df_Hz":         dfreq,
                        "bin10_Hz":      bin10,
                        "offset_bins":   offset,
                        "sinc_atten_pct": sinc_at * 100,
                    })
    return pd.DataFrame(rows)


def make_figure(df_meas, fft_dict, meta, out_path: Path):
    """Two panels:
      (left)  zoom of |FFT| near paddle for the 1.4 Hz canon nowind run.
              Vertical lines = bin centres. Red marker = f_paddle.
      (right) theoretical sinc curve vs measured (offset, attenuation) points.
    """
    sel = apply_experimental_filters(meta, {"filters": {
        "WaveAmplitudeInput [Volt]": 0.2,
        "WaveFrequencyInput [Hz]":   1.4,
        "WindCondition":             "no",
        "PanelCondition":            "full",
    }})
    path = sel.iloc[0]["path"]
    df_fft = fft_dict[path]
    probe = "9373/170"
    mag = df_fft[f"FFT {probe}"].dropna()
    freqs = mag.index.values
    pos = freqs > 0
    f, m = freqs[pos], mag.values[pos] * 2.0    # *2 for one-sided amplitude

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=120)

    # Panel 1: spectrum zoom with bin grid
    ax = axes[0]
    win = (f > 1.0) & (f < 1.8)
    ax.stem(f[win], m[win], basefmt=" ", linefmt="C0-", markerfmt="C0o")
    for fc in f[win]:
        ax.axvline(fc, color="lightgray", lw=0.4, zorder=0)
    ax.axvline(1.4, color="red", lw=1.5, linestyle="--",
               label=r"$f_\mathrm{paddle}=1.4$ Hz")
    df0 = abs(f[1] - f[0])
    ax.set_xlim(1.0, 1.8)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("FFT amplitude [mm]")
    ax.set_title(f"(a) Spectrum near paddle  —  $\\Delta f={df0:.4f}$ Hz, "
                 f"bin 10 at {10*df0:.4f} Hz")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: sinc curve + measured points
    ax = axes[1]
    xs = np.linspace(-0.6, 0.6, 401)
    ax.plot(xs, (1 - np.abs(np.sinc(xs))) * 100,
            color="black", lw=1.2, label="theoretical:  $1-|\\mathrm{sinc}(\\delta)|$")
    ax.scatter(df_meas["offset_bins"], df_meas["sinc_atten_pct"],
               s=20, color="C3", alpha=0.7, label=f"measured ({len(df_meas)} probe-runs)")
    ax.set_xlabel("Offset from bin 10  [bin widths]")
    ax.set_ylabel("Sinc attenuation at nearest bin  [%]")
    ax.set_title("(b) Single-bin sinc loss vs alignment")
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.5, 40)
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.legend(loc="upper center", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path)
    print(f"  saved {out_path}")


def main():
    meta, _, fft_dict, _ = load_analysis_data(*PROCESSED, load_processed=False)
    df_meas = measure_alignment(meta, fft_dict)

    csv_path = OUT_DIR / "fft_bin_alignment_table.csv"
    df_meas.to_csv(csv_path, index=False)
    print(f"  saved {csv_path}  ({len(df_meas)} rows)")

    print("\n── Summary ──")
    print(f"measurements:               {len(df_meas)}")
    print(f"|offset|  median:           {df_meas['offset_bins'].abs().median():.4f} bins")
    print(f"|offset|  max:              {df_meas['offset_bins'].abs().max():.4f} bins")
    print(f"sinc attenuation median:    {df_meas['sinc_atten_pct'].median():.4f} %")
    print(f"sinc attenuation max:       {df_meas['sinc_atten_pct'].max():.4f} %")
    print(f"bin width range:            "
          f"{df_meas['df_Hz'].min():.3f} – {df_meas['df_Hz'].max():.3f} Hz")

    make_figure(df_meas, fft_dict, meta, OUT_DIR / "fft_bin_alignment.png")


if __name__ == "__main__":
    main()
