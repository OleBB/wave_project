import re
import numpy as np
import os
import matplotlib.pyplot as plt

# --- USER SETTINGS ---
ein_folder = r"/Users/ole/Kodevik/wave_project/pressuredata/20251107-lowestwindUP2-allpanel-angleTest"

angle_pattern = re.compile(r"ang(\d+)", re.IGNORECASE)

def read_mean_from_file(file_path):
    values = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().replace('"', '').replace(',', '.')
            if line:
                try:
                    values.append(float(line))
                except ValueError:
                    pass
    return np.mean(values) if values else np.nan


def process_files(folder_path):
    angles = []
    means = []
    labels = []

    for fname in os.listdir(folder_path):
        if not fname.lower().endswith(".txt"):
            continue

        match = angle_pattern.search(fname)
        if not match:
            continue

        angle = int(match.group(1))
        file_path = os.path.join(folder_path, fname)
        mean_val = read_mean_from_file(file_path)

        angles.append(angle)
        means.append(mean_val)
        labels.append(fname)

    # sort by angle
    angles, means, labels = zip(*sorted(zip(angles, means, labels)))
    return angles, means, labels


# --- MAIN PLOTTING ---
fig, ax = plt.subplots(figsize=(8, 6))

angles, means, labels = process_files(ein_folder)

ax.scatter(angles, means)

# annotate each point
for a, m, lab in zip(angles, means, labels):
    short = lab[:12]   # or any substring you want
    ax.annotate(short, (a, m), xytext=(0, 5), textcoords="offset points",
                ha='center', fontsize=8)

ax.set_xlabel("Angle")
ax.set_ylabel("Mean pressure")
plt.show()


ax.set_ylabel("Angle in degrees")
ax.set_xlabel("Wind speed [m/s]")
ax.set_title("Wind speed accuracy based on angle ")

ax.set_ylim(ymin=min(angles), ymax=30)

ax.grid(True, which='both', linestyle='--', linewidth=0.5)
#ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))  # Ensure major ticks are integer values
#ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=10))     # Set number of bins/ticks on the y-axis
#ax.legend(title="Folder")
#plt.tight_layout()

