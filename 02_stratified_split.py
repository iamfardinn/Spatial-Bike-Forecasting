"""
Phase 2 -- Stratified Train / Test Split
==========================================
Splits MGWR_with_Cohorts.csv into train (80 %) and test (20 %) sets,
stratified on the 'Cohort' column so each cohort's demand distribution
is faithfully represented in both partitions.

Outputs:
  * train.csv              -- 80 % stratified sample
  * test.csv               -- 20 % stratified sample
  * split_validation.png   -- cohort proportion & demand distribution check
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.model_selection import train_test_split

# ── 0. Style (matches Phase 1) ────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#1a1d27",
    "axes.edgecolor":   "#3a3f55",
    "axes.labelcolor":  "#c8cfe8",
    "xtick.color":      "#c8cfe8",
    "ytick.color":      "#c8cfe8",
    "text.color":       "#e8ecf4",
    "grid.color":       "#2e3248",
    "grid.linewidth":   0.6,
    "font.family":      "DejaVu Sans",
    "font.size":        11,
})

COHORT_COLORS = ["#4fc3f7", "#81c784", "#ffb74d", "#f48fb1"]
COHORT_NAMES  = ["Cohort A", "Cohort B", "Cohort C", "Cohort D"]
RANDOM_STATE  = 42
TEST_SIZE     = 0.20

# ── 1. Load enriched data ──────────────────────────────────────────────────────
print("Loading MGWR_with_Cohorts.csv ...")
df = pd.read_csv("MGWR_with_Cohorts.csv")
print(f"  {len(df):,} stations, {df.shape[1]} columns")

# Feature groups (excluding target and cohort label)
FEATURES = [
    # Spatial
    "Latitude", "Longitude", "Slope",
    # Land-use
    "Resident", "Commercial", "Industry", "Green", "Fp_perc", "Pop_perc",
    # Transit
    "Bus_boarding", "Bus_alighting", "Sub_boarding", "Sub_alighting",
    "Bike_on_count", "Bike_off_count", "NUM_BUS", "NUM_SUB",
    # Weather
    "Temperature_C", "Precipitation_mm", "Humidity", "TinyDust",
    # Temporal
    "Time",
]
TARGET = "Bike_Demand"

# Keep only columns that actually exist in the CSV
FEATURES = [f for f in FEATURES if f in df.columns]
print(f"  Features used: {len(FEATURES)}")

# ── 2. Stratified split ────────────────────────────────────────────────────────
print(f"\nPerforming stratified {int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)} split ...")

train_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df["Cohort"],
)

print(f"  Train: {len(train_df):,} stations")
print(f"  Test : {len(test_df):,} stations")

# ── 3. Per-cohort summary ──────────────────────────────────────────────────────
print("\nCohort distribution check:")
print(f"  {'Cohort':<10} {'Total':>7} {'Train':>7} {'Test':>7} {'Train%':>8} {'Test%':>7}")
print("  " + "-" * 52)
for c, name in enumerate(COHORT_NAMES):
    total = len(df[df["Cohort"] == c])
    tr    = len(train_df[train_df["Cohort"] == c])
    te    = len(test_df[test_df["Cohort"] == c])
    print(f"  {name:<10} {total:>7,} {tr:>7,} {te:>7,} {tr/total*100:>7.1f}% {te/total*100:>7.1f}%")

# ── 4. Validation plot ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 9))
fig.patch.set_facecolor("#0f1117")
gs = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

# --- Panel A: Cohort counts (Train vs Test side-by-side bars) -----------------
ax1 = fig.add_subplot(gs[0, 0])
k         = len(COHORT_NAMES)
x         = np.arange(k)
bar_w     = 0.35
train_counts = [len(train_df[train_df["Cohort"] == c]) for c in range(k)]
test_counts  = [len(test_df[test_df["Cohort"] == c])  for c in range(k)]

bars_tr = ax1.bar(x - bar_w/2, train_counts, bar_w,
                  color=COHORT_COLORS, alpha=0.85, label="Train")
bars_te = ax1.bar(x + bar_w/2, test_counts,  bar_w,
                  color=COHORT_COLORS, alpha=0.45, label="Test",
                  edgecolor=COHORT_COLORS, linewidth=1.5)

for bar in bars_tr:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             f"{int(bar.get_height())}", ha="center", va="bottom", fontsize=8)
for bar in bars_te:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             f"{int(bar.get_height())}", ha="center", va="bottom", fontsize=8)

ax1.set_xticks(x)
ax1.set_xticklabels(COHORT_NAMES, fontsize=9)
ax1.set_ylabel("Station Count")
ax1.set_title("Cohort Counts — Train vs Test", weight="bold")
ax1.legend(facecolor="#1a1d27", edgecolor="#3a3f55")
ax1.grid(True, axis="y")

# --- Panel B: Cohort proportion stacked bars ----------------------------------
ax2 = fig.add_subplot(gs[0, 1])
splits = {"Train": train_df, "Test": test_df, "Full": df}
bottom_vals = {s: 0 for s in splits}

for c, (name, color) in enumerate(zip(COHORT_NAMES, COHORT_COLORS)):
    proportions = [len(splits[s][splits[s]["Cohort"] == c]) / len(splits[s]) * 100
                   for s in splits]
    bars = ax2.bar(list(splits.keys()), proportions, bottom=list(bottom_vals.values()),
                   color=color, alpha=0.85, width=0.5)
    for bar, prop in zip(bars, proportions):
        if prop > 4:
            ax2.text(bar.get_x() + bar.get_width()/2,
                     bar.get_y() + bar.get_height()/2,
                     f"{prop:.1f}%", ha="center", va="center",
                     fontsize=8, color="#0f1117", weight="bold")
    for s, p in zip(splits, proportions):
        bottom_vals[s] += p

ax2.set_ylabel("Proportion (%)")
ax2.set_title("Cohort Proportions — Stratification Check", weight="bold")
ax2.set_ylim(0, 105)
ax2.grid(True, axis="y")
legend_patches = [mpatches.Patch(color=c, label=n)
                  for c, n in zip(COHORT_COLORS, COHORT_NAMES)]
ax2.legend(handles=legend_patches, facecolor="#1a1d27", edgecolor="#3a3f55",
           fontsize=8, loc="upper right")

# --- Panel C: Bike_Demand distribution histogram Train vs Test ----------------
ax3 = fig.add_subplot(gs[1, 0])
bins = np.linspace(df[TARGET].min(), df[TARGET].quantile(0.99), 40)
ax3.hist(train_df[TARGET], bins=bins, color="#4fc3f7", alpha=0.6,
         density=True, label=f"Train (n={len(train_df):,})")
ax3.hist(test_df[TARGET],  bins=bins, color="#f48fb1", alpha=0.6,
         density=True, label=f"Test  (n={len(test_df):,})")
ax3.set_xlabel("Bike Demand (avg daily trips)")
ax3.set_ylabel("Density")
ax3.set_title("Demand Distribution — Train vs Test", weight="bold")
ax3.legend(facecolor="#1a1d27", edgecolor="#3a3f55")
ax3.grid(True)

# --- Panel D: Per-cohort mean demand (Train vs Test) --------------------------
ax4 = fig.add_subplot(gs[1, 1])
train_means = [train_df[train_df["Cohort"] == c][TARGET].mean() for c in range(k)]
test_means  = [test_df[test_df["Cohort"] == c][TARGET].mean()  for c in range(k)]

ax4.plot(COHORT_NAMES, train_means, "o-", color="#4fc3f7",
         linewidth=2.5, markersize=8, label="Train mean")
ax4.plot(COHORT_NAMES, test_means,  "s--", color="#f48fb1",
         linewidth=2.5, markersize=8, label="Test mean")

for i, (tr, te) in enumerate(zip(train_means, test_means)):
    ax4.text(i, tr + 0.2, f"{tr:.2f}", ha="center", fontsize=8, color="#4fc3f7")
    ax4.text(i, te - 0.6, f"{te:.2f}", ha="center", fontsize=8, color="#f48fb1")

ax4.set_ylabel("Mean Bike Demand")
ax4.set_title("Per-Cohort Mean Demand — Train vs Test", weight="bold")
ax4.legend(facecolor="#1a1d27", edgecolor="#3a3f55")
ax4.grid(True)

fig.suptitle("Phase 2 — Stratified Split Validation  |  Seoul Bike Stations",
             fontsize=14, weight="bold", color="#e8ecf4", y=1.01)

plt.savefig("split_validation.png", dpi=150, bbox_inches="tight",
            facecolor="#0f1117")
plt.close()
print("\n  -> split_validation.png saved")

# ── 5. Save splits ─────────────────────────────────────────────────────────────
train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv",   index=False)
print("  -> train.csv saved")
print("  -> test.csv  saved")

print(f"""
==========================================
  Phase 2 complete
==========================================
  Total stations : {len(df):,}
  Train          : {len(train_df):,}  ({(1-TEST_SIZE)*100:.0f} %)
  Test           : {len(test_df):,}  ({TEST_SIZE*100:.0f} %)
  Stratified on  : Cohort (k=4)
  Random seed    : {RANDOM_STATE}
==========================================
""")
