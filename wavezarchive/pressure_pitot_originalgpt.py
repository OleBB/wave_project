
import matplotlib.pyplot as plt
import glob
import re
import pandas as pd

def load_data(folder):
    files = glob.glob(f"{folder}/*.txt")
    data = {}
    for filename in files:
        match = re.search(r"moh(\d+)", filename)
        if not match:
            continue
        key = int(match.group(1))
        with open(filename, "r", encoding="utf-8") as f:
            lines = [line.strip().strip('"').replace(',', '.') for line in f if line.strip()]
            values = [float(x) for x in lines]
        data[key] = values

    # ✅ FIX: allow unequal-length columns
    df = pd.DataFrame({k: pd.Series(v) for k, v in sorted(data.items())})
    return df

    return pd.DataFrame(dict(sorted(data.items())))

# --- load both folders ---
df_full = load_data("../pressuredata/20251104-fullwind/")
df_low = load_data("../pressuredata/20251104-lowestwind/")

# --- compute means ---
mean_full = df_full.mean()
mean_low = df_low.mean()


#%%
# --- plot both on same figure ---
plt.figure(figsize=(7,5))
plt.plot(mean_full.values, mean_full.index, '-o', label="Full Wind")
plt.plot(mean_low.values, mean_low.index, '-s', label="Lowest Wind")
plt.title("Windspeeds over water 10075mm from padle")
plt.ylabel("Height (mm)")
plt.xlabel("Mean Value")
plt.legend()
plt.ylim(bottom=0) 
plt.grid(True)
plt.tight_layout()
plt.show()

