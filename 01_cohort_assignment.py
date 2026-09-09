"""
Phase 1 -- K-Means Cohort Assignment
=====================================
Derives spatial demand cohorts for Seoul bike-sharing stations using
K-Means clustering on Latitude, Longitude, and Bike_Demand.

Outputs:
  * MGWR_with_Cohorts.csv  -- original data + 'Cohort' column (0-indexed label)
  * cohort_elbow.png       -- inertia elbow plot (k = 2..10)
  * cohort_profiles.png    -- per-cohort boxplots of key features
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

FIG_DIR = "outputs/figures"
TBL_DIR = "outputs/tables"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TBL_DIR, exist_ok=True)

#  0. Style 
plt.rcParams.update({
    "figure.facecolor": "#ffffff",
    "axes.facecolor":   "#ffffff",
    "axes.edgecolor":   "#333333",
    "axes.labelcolor":  "#222222",
    "xtick.color":      "#222222",
    "ytick.color":      "#222222",
    "text.color":       "#111111",
    "grid.color":       "#e5e7eb",
    "grid.linewidth":   0.6,
    "font.family":      "DejaVu Sans",
    "font.size":        11,
})

COHORT_COLORS = ["#1976d2", "#388e3c", "#f57c00", "#d32f2f"]
COHORT_NAMES  = ["Cohort A", "Cohort B", "Cohort C", "Cohort D"]

#  1. Load data 
print("Loading data...")
df = pd.read_csv("MGWR Analysis Data.csv", encoding="latin1")
df.columns = df.columns.str.strip()

# Rename the temperature column (has encoding artefact)
df.rename(columns={c: "Temperature_C" for c in df.columns if "Temp" in c}, inplace=True)

print(f"  {len(df):,} stations loaded, {df.shape[1]} features")

#  2. Build clustering feature matrix 
CLUSTER_FEATURES = ["Latitude", "Longitude", "Bike_Demand"]

X_raw = df[CLUSTER_FEATURES].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

#  3. Elbow + Silhouette analysis (k = 2  10) 
print("Running elbow analysis...")
k_range   = range(2, 11)
inertias  = []
sil_scores = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))
    print(f"  k={k}  inertia={km.inertia_:,.1f}  silhouette={sil_scores[-1]:.4f}")

#  4. Plot elbow 
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor("#ffffff")
fig.suptitle("K-Means Cohort Selection — Seoul Bike Stations", fontsize=14, weight="bold", color="#111111", y=1.02)

# Inertia
ax1 = axes[0]
ax1.plot(list(k_range), inertias, "o-", color="#1976d2", linewidth=2.5, markersize=7)
ax1.axvline(4, color="#d32f2f", linewidth=1.5, linestyle="--", label="Chosen k = 4")
ax1.set_xlabel("Number of Clusters (k)")
ax1.set_ylabel("Inertia (WCSS)")
ax1.set_title("Elbow Plot", weight="bold")
ax1.legend(facecolor="#ffffff", edgecolor="#cccccc")
ax1.grid(True)

# Silhouette
ax2 = axes[1]
ax2.plot(list(k_range), sil_scores, "s-", color="#388e3c", linewidth=2.5, markersize=7)
ax2.axvline(4, color="#d32f2f", linewidth=1.5, linestyle="--", label="Chosen k = 4")
ax2.set_xlabel("Number of Clusters (k)")
ax2.set_ylabel("Silhouette Score")
ax2.set_title("Silhouette Score", weight="bold")
ax2.legend(facecolor="#ffffff", edgecolor="#cccccc")
ax2.grid(True)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/cohort_elbow.png", dpi=300, bbox_inches="tight", facecolor="#ffffff")
plt.close()
print("  -> cohort_elbow.png saved")

#  5. Final K-Means with k = 4 
print("\nFitting final K-Means (k=4)...")
K = 4
km_final = KMeans(n_clusters=K, random_state=42, n_init=10)
df["Cohort"] = km_final.fit_predict(X_scaled)

# Sort cohort labels by ascending mean Bike_Demand so labels are interpretable
cohort_means = df.groupby("Cohort")["Bike_Demand"].mean().sort_values()
label_map = {old: new for new, old in enumerate(cohort_means.index)}
df["Cohort"] = df["Cohort"].map(label_map)

#  6. Cohort summary 
print("\nCohort Summary:")
summary = df.groupby("Cohort").agg(
    Count=("ID", "count"),
    Demand_mean=("Bike_Demand", "mean"),
    Demand_std=("Bike_Demand", "std"),
    Lat_mean=("Latitude", "mean"),
    Lon_mean=("Longitude", "mean"),
    Time_mean=("Time", "mean"),
    BusBoarding_mean=("Bus_boarding", "mean"),
    SubBoarding_mean=("Sub_boarding", "mean"),
).round(3)
print(summary.to_string())

#  7. Profile plot 
PROFILE_FEATURES = ["Bike_Demand", "Bus_boarding", "Sub_boarding",
                    "Pop_perc", "Slope", "Temperature_C"]

fig = plt.figure(figsize=(16, 9))
fig.patch.set_facecolor("#ffffff")
gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

for idx, feat in enumerate(PROFILE_FEATURES):
    ax = fig.add_subplot(gs[idx // 3, idx % 3])
    data_by_cohort = [df.loc[df["Cohort"] == c, feat].values for c in range(K)]
    bp = ax.boxplot(
        data_by_cohort,
        patch_artist=True,
        widths=0.5,
        medianprops=dict(color="#111111", linewidth=2),
        whiskerprops=dict(color="#555555"),
        capprops=dict(color="#555555"),
        flierprops=dict(marker="o", markersize=2, color="#777777", alpha=0.4),
    )
    for patch, color in zip(bp["boxes"], COHORT_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    ax.set_title(feat.replace("_", " "), fontsize=10, weight="bold")
    ax.set_xticks(range(1, K + 1))
    ax.set_xticklabels(COHORT_NAMES, fontsize=8)
    ax.grid(True, axis="y")

fig.suptitle("Feature Profiles by Cohort — Seoul Bike Stations",
             fontsize=14, weight="bold", color="#111111")

# Add legend strip
for i, (name, color) in enumerate(zip(COHORT_NAMES, COHORT_COLORS)):
    fig.text(0.15 + i * 0.18, 0.01, f"■ {name}",
             color=color, fontsize=10, ha="center", weight="bold")

plt.savefig(f"{FIG_DIR}/cohort_profiles.png", dpi=300, bbox_inches="tight", facecolor="#ffffff")
plt.close()
print("  -> cohort_profiles.png saved")

#  8. Save enriched CSV 
out_path = f"{TBL_DIR}/MGWR_with_Cohorts.csv"
df.to_csv(out_path, index=False)
print(f"\n  -> {out_path} saved  ({len(df):,} rows, columns: {df.columns.tolist()[-4:]} )")

print("\n Phase 1 complete.")
