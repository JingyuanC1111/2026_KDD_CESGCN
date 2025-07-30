import matplotlib.pyplot as plt
import pandas as pd

# Data from the table
dates = [
    "2024-11-23", "2024-11-30", "2024-12-07", "2024-12-14", "2024-12-21",
    "2024-12-28", "2025-01-04", "2025-01-11", "2025-01-18"
]

data = {
    "CESGCN": [43.5, 41.4, 92.34, 185.01, 179.67, 168.9, 198.2, 155.08, 163.8],
    #"CESGCN-UNIFIED": [69.0, 77.4, 157.89, 263.5, 260.5, 310.3, 290.16, 265.07, 349.98],
    #"CESGCN-WO-MTE": [46.07, 71.18, 162.18, 202.1, 203.4, 265.0, 298.7, 228.82, 172.98],
    #"CESGCN-WO-ADAPTIVE": [51.86, 65.9, 156.5, 199.6, 223.5, 222.1, 277.3, 224.7, 229.1],
    #"CESGCN-WO-DC": [52.5, 97.1, 187.95, 298.6, 339.66, 384.8, 408.78, 309.6, 300.99],
    "Flusight_ensemble": [32.89, 63.9, 125.76, 217.83, 234.19, 236.38, 153.68, 104.5, 101.0],
    "Flusight_baseline": [45.3, 80.5, 147.4, 258.23, 299.35, 302.93, 196.86, 136.8, 121.2],
}

# Convert to DataFrame
df = pd.DataFrame(data, index=pd.to_datetime(dates))

# Define key models to emphasize
bold_lines = ["CESGCN", "Flusight_ensemble", "Flusight_baseline"]

# Create the plot
plt.figure(figsize=(14, 7))
for column in df.columns:
    linestyle = '-' if column in bold_lines else '--'
    linewidth = 3.5 if column in bold_lines else 2
    markersize = 8 if column in bold_lines else 6
    alpha = 0.9 if column in bold_lines else 0.7
    plt.plot(df.index, df[column], marker='o', linestyle=linestyle, linewidth=linewidth, markersize=markersize, label=column, alpha=alpha)

# Enhancing visualization
plt.xlabel("Date", fontsize=14)
plt.ylabel("Mean WIS Across Region", fontsize=14)
plt.title("Performance Comparison of CESGCN and Flusight Models in 2024 Season", fontsize=16, fontweight='bold')
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=10, loc="upper left", bbox_to_anchor=(1, 1))
plt.grid(True, linestyle="--", alpha=0.5)

# Display the plot
plt.tight_layout()
plt.show()
print('True')
