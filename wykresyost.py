import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================
INPUT_FILE = "najlepsze_pary_dylacja1.txt"  # Your text file
HISTOGRAM_OUTPUT = "img/hd_histogram.png"
BOXPLOT_OUTPUT = "img/hd_boxplot.png"
SCATTER_OUTPUT = "img/hd_scatter.png"
CDF_OUTPUT = "img/hd_cdf.png"

# ==========================================
# 2. READ DATA FROM FILE
# ==========================================
baseline_hd = []
dilated_hd = []

if not os.path.exists(INPUT_FILE):
    print(f"❌ Error: Could not find '{INPUT_FILE}'.")
    exit()

print(f"Reading data from {INPUT_FILE}...")

with open(INPUT_FILE, "r") as f:
    for line in f:
        line = line.strip()
        if not line: continue
        
        parts = line.split(";")
        
        # Structure: PairID; Name1; Name2; HD_STD; HD_DIL
        if len(parts) >= 5:
            try:
                val_std = float(parts[3])
                val_dil = float(parts[4])
                baseline_hd.append(val_std)
                dilated_hd.append(val_dil)
            except ValueError:
                print(f"⚠️ Skipping invalid line: {line}")
        elif len(parts) == 4: # Fallback for older format
            try:
                val_std = float(parts[2])
                val_dil = float(parts[3])
                baseline_hd.append(val_std)
                dilated_hd.append(val_dil)
            except ValueError: pass
            
base = np.array(baseline_hd)
dil = np.array(dilated_hd)

print(f"✅ Successfully loaded {len(baseline_hd)} pairs.")

# ==========================================
# 3. GENERATE HISTOGRAM WITH MEANS
# ==========================================
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

# Calculate Means
mean_base = np.mean(baseline_hd)
mean_dil = np.mean(dilated_hd)

# Plot Distributions
sns.histplot(baseline_hd, color="skyblue", label="Baseline (Standard)", kde=True, element="step", alpha=0.5)
sns.histplot(dilated_hd, color="red", label="Dilated (Accordion Effect)", kde=True, element="step", alpha=0.5)

# Add Threshold Line (Black Dashed)
plt.axvline(x=0.32, color='black', linestyle='--', linewidth=2, label="Threshold (0.32)")

# Add Mean Lines (Dotted)
plt.axvline(mean_base, color='blue', linestyle=':', linewidth=2, label=f'Mean Baseline ({mean_base:.3f})')
plt.axvline(mean_dil, color='darkred', linestyle=':', linewidth=2, label=f'Mean Dilated ({mean_dil:.3f})')

# Add Text Annotations for Means (Optional, for better visibility)
plt.text(mean_base, plt.ylim()[1]*0.9, f'{mean_base:.2f}', color='blue', ha='right', fontweight='bold')
plt.text(mean_dil, plt.ylim()[1]*0.9, f'{mean_dil:.2f}', color='darkred', ha='left', fontweight='bold')

plt.title("Shift in Hamming Distance Distribution", fontsize=14, fontweight='bold')
plt.xlabel("Hamming Distance (HD)", fontsize=12)
plt.ylabel("Number of Pairs", fontsize=12)
plt.legend(loc='upper right')
plt.xlim(0, 0.6)

os.makedirs(os.path.dirname(HISTOGRAM_OUTPUT), exist_ok=True)
plt.savefig(HISTOGRAM_OUTPUT, dpi=300)
print(f"Histogram saved to {HISTOGRAM_OUTPUT}")
plt.close()

# ==========================================
# 4. GENERATE BOXPLOT
# ==========================================
plt.figure(figsize=(8, 6))

data = [baseline_hd, dilated_hd]
labels = ['Baseline', 'Dilated']

bplot = plt.boxplot(data, vert=True, patch_artist=True, labels=labels, widths=0.5)

colors = ['skyblue', 'lightcoral']
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)

plt.axhline(y=0.32, color='black', linestyle='--', linewidth=1.5, label="Rejection Threshold")

plt.title("Statistical Degradation of Matching Score", fontsize=14, fontweight='bold')
plt.ylabel("Hamming Distance", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()

plt.savefig(BOXPLOT_OUTPUT, dpi=300)
print(f"Boxplot saved to {BOXPLOT_OUTPUT}")
plt.close()
plt.figure(figsize=(8, 8))
sns.set_style("whitegrid")

# Define Categories for Coloring
# 1. Safe: Accepted Before -> Accepted After
safe_mask = (base <= 0.32) & (dil <= 0.32)
# 2. False Rejection: Accepted Before -> Rejected After (CRITICAL GROUP)
frr_mask = (base <= 0.32) & (dil > 0.32)
# 3. Always Bad: Rejected Before (Shouldn't happen in your "Best Pairs", but good to handle)
bad_mask = (base > 0.32)

plt.scatter(base[safe_mask], dil[safe_mask], c='green', alpha=0.5, label='Stable Match (Safe)', s=20)
plt.scatter(base[frr_mask], dil[frr_mask], c='red', alpha=0.6, label='False Rejection (Risk)', s=20, marker='x')

# Add Reference Lines
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', alpha=0.5, label="No Change (y=x)")
plt.axvline(0.32, color='black', linestyle=':', alpha=0.5)
plt.axhline(0.32, color='black', linestyle=':', alpha=0.5)

# Styling
plt.title("Individual Score Degradation", fontsize=14, fontweight='bold')
plt.xlabel("Baseline HD (Standard)", fontsize=12)
plt.ylabel("Dilated HD (Simulated)", fontsize=12)
plt.xlim(0, 0.5)
plt.ylim(0, 0.6)
plt.legend(loc='upper right')

# Annotate the "Danger Zone"
count_fail = np.sum(frr_mask)
total = len(base)
percent_fail = (count_fail / total) * 100
plt.text(0.02, 0.55, f"FRR Increase: {percent_fail:.1f}%", fontsize=12, color='red', fontweight='bold', 
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'))

plt.savefig(SCATTER_OUTPUT, dpi=300)
print(f"Scatter plot saved to {SCATTER_OUTPUT}")
plt.close()

# ==========================================
# PLOT 2: EMPIRICAL CDF (ACCEPTANCE RATES)
# ==========================================
plt.figure(figsize=(10, 6))

# Sort data
sorted_base = np.sort(base)
sorted_dil = np.sort(dil)
y_vals = np.arange(1, len(base) + 1) / len(base)

# Plot Curves
plt.step(sorted_base, y_vals, label='Baseline Acceptance', color='blue', where='post')
plt.step(sorted_dil, y_vals, label='Dilated Acceptance', color='red', where='post')

# Threshold Marker
plt.axvline(0.32, color='black', linestyle='--', label='Threshold (0.32)')

# Calculate "Drop" at 0.32
# Find the Y-value (Acceptance Rate) at X=0.32
acc_base = np.sum(base <= 0.32) / total
acc_dil = np.sum(dil <= 0.32) / total
drop = acc_base - acc_dil

# Draw Arrow showing the drop
plt.annotate('', xy=(0.32, acc_dil), xytext=(0.32, acc_base),
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.text(0.33, (acc_base + acc_dil)/2, f"-{drop*100:.1f}% Acceptance", color='black', fontweight='bold')

plt.title("Cumulative Match Probability (CDF)", fontsize=14, fontweight='bold')
plt.xlabel("Hamming Distance Threshold", fontsize=12)
plt.ylabel("Proportion of Pairs Accepted", fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(loc='lower right')
plt.xlim(0, 0.6)

plt.savefig(CDF_OUTPUT, dpi=300)
print(f"CDF plot saved to {CDF_OUTPUT}")
plt.close()
