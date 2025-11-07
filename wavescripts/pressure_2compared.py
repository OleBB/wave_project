import numpy as np
from scipy import stats  # needs: pip install scipy

# --- hardcoded file paths ---
#file1 = "/Users/ole/Kodevik/wave_project/pressuredata/20251104-fullwind/pitot10075-fullwind-speed-moh030.txt"
#file2 = "/Users/ole/Kodevik/wave_project/pressuredata/20251104-fullwind/20251104-readme-bonus/pitot10075-fullwind-speed-moh31.txt"

#file1 = "/Users/ole/Kodevik/wave_project/pressuredata/20251105-lowestwind-fullpanel/20251105-lowestwind-fullpanel-pitot10075-speed-moh015.txt"
#file2 = "/Users/ole/Kodevik/wave_project/pressuredata/20251105-lowestwind-fullpanel/20251104-lowestwind-fullpanel-bonus/20251105-lowestwind-fullpanel-pitot10075-speed-moh016.txt"

file1 = "/Users/ole/Kodevik/wave_project/pressuredata/20251105-lowestwindUtenProbe-fullpanel/20251105-lowestwindUtenProbe-fullpanel-pitot10075-speed-moh160.txt"
file2 = "/Users/ole/Kodevik/wave_project/pressuredata/20251105-lowestwindUtenProbe-fullpanel/20251105-lowestwindUtenProbe-fullpanel-pitot10075-bonus/20251105-lowestwindUtenProbe-fullpanel-pitot10075-speed-moh160-pitot15grader.txt"

file1 = "/Users/ole/Kodevik/wave_project/pressuredata/20251106-lowestwindUtenProbe2-fullpanel-amp0100-freq1300/20251106-lowestwindUP2-angletest/20251106-lowestwindUP2-allpanel-angleTest-pitot10075-speed-moh051.txt"
file2 = "/Users/ole/Kodevik/wave_project/pressuredata/20251106-lowestwindUtenProbe2-fullpanel-amp0100-freq1300/20251106-lowestwindUP2-angletest/20251106-lowestwindUP2-allpanel-angleTest-pitot10075ang0xtratid-speed-moh051.txt"

def read_values(filename):
    """Reads quoted numbers with commas as decimals and returns a NumPy array."""
    values = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().strip('"').replace(',', '.')
            if line:
                try:
                    values.append(float(line))
                except ValueError:
                    pass
    return np.array(values)

# --- read both files ---
data1 = read_values(file1)
data2 = read_values(file2)

# --- compute basic stats ---
def stats_summary(data):
    return {
        "mean": np.mean(data),
        "min": np.min(data),
        "max": np.max(data),
        "std": np.std(data, ddof=1),
        "count": len(data)
    }

stats1 = stats_summary(data1)
stats2 = stats_summary(data2)

# --- print summaries ---
print(f"File 1: {file1}")
for k, v in stats1.items():
    print(f"  {k:>5}: {v:.5f}")

print(f"\nFile 2: {file2}")
for k, v in stats2.items():
    print(f"  {k:>5}: {v:.5f}")

# --- mean difference ---
mean_diff = stats1["mean"] - stats2["mean"]
print(f"\nMean difference (File1 - File2): {mean_diff:.5f}")

# --- two-sample t-test (unequal lengths/variances) ---
t_stat, p_val = stats.ttest_ind(data1, data2, equal_var=False)

print(f"\nT-statistic: {t_stat:.3f}")
print(f"P-value: {p_val:.5f}")

# --- interpret significance ---
alpha = 0.05  # 95% confidence
if p_val < alpha:
    print("❗ The difference in means is statistically significant (p < 0.05).")
else:
    print("✅ No statistically significant difference (p ≥ 0.05).")
