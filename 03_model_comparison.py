"""
Phase 3 -- Random Forest vs XGBoost Model Comparison
======================================================
Trains and evaluates two regressors on the stratified split from Phase 2.
Both models are also broken down per-cohort to show where each excels.

Outputs:
  * model_comparison.png   -- 4-panel performance dashboard
  * metrics_summary.csv    -- RMSE / MAE / R2 for both models (overall + per-cohort)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# ── 0. Style (matches Phase 1 / 2) ───────────────────────────────────────────
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
RF_COLOR      = "#7c83fd"
XGB_COLOR     = "#fd7c7c"
TARGET        = "Bike_Demand"

# ── 1. Load splits ────────────────────────────────────────────────────────────
print("Loading train / test splits ...")
train_df = pd.read_csv("train.csv")
test_df  = pd.read_csv("test.csv")
print(f"  Train: {len(train_df):,}  |  Test: {len(test_df):,}")

FEATURES = [c for c in train_df.columns if c not in [TARGET, "Cohort", "ID"]]
print(f"  Features: {len(FEATURES)}")

X_train, y_train = train_df[FEATURES], train_df[TARGET]
X_test,  y_test  = test_df[FEATURES],  test_df[TARGET]

# ── 2. Train models ───────────────────────────────────────────────────────────
print("\nTraining Random Forest ...")
rf = RandomForestRegressor(n_estimators=300, max_features="sqrt",
                           min_samples_leaf=3, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print("  done.")

print("Training XGBoost ...")
xgb_model = xgb.XGBRegressor(
    n_estimators=500, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    min_child_weight=3, random_state=42,
    verbosity=0, n_jobs=-1,
)
xgb_model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)
xgb_pred = xgb_model.predict(X_test)
print("  done.")

# ── 3. Metrics helper ─────────────────────────────────────────────────────────
def metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return rmse, mae, r2

# Overall
rf_rmse,  rf_mae,  rf_r2  = metrics(y_test, rf_pred)
xgb_rmse, xgb_mae, xgb_r2 = metrics(y_test, xgb_pred)

print(f"\nOverall Performance:")
print(f"  {'Model':<12} {'RMSE':>8} {'MAE':>8} {'R2':>8}")
print(f"  {'-'*38}")
print(f"  {'RF':<12} {rf_rmse:>8.4f} {rf_mae:>8.4f} {rf_r2:>8.4f}")
print(f"  {'XGBoost':<12} {xgb_rmse:>8.4f} {xgb_mae:>8.4f} {xgb_r2:>8.4f}")

# Per-cohort
rows = []
K = len(COHORT_NAMES)
cohort_col = test_df["Cohort"].values

print(f"\nPer-Cohort Performance:")
print(f"  {'Cohort':<12} {'RF RMSE':>9} {'XGB RMSE':>10} {'RF R2':>8} {'XGB R2':>8}")
print(f"  {'-'*52}")

for c, name in enumerate(COHORT_NAMES):
    mask = cohort_col == c
    if mask.sum() == 0:
        continue
    r_rmse, r_mae, r_r2  = metrics(y_test[mask], rf_pred[mask])
    x_rmse, x_mae, x_r2  = metrics(y_test[mask], xgb_pred[mask])
    rows.append({"Cohort": name,
                 "RF_RMSE": r_rmse, "RF_MAE": r_mae,  "RF_R2":  r_r2,
                 "XGB_RMSE": x_rmse, "XGB_MAE": x_mae, "XGB_R2": x_r2})
    print(f"  {name:<12} {r_rmse:>9.4f} {x_rmse:>10.4f} {r_r2:>8.4f} {x_r2:>8.4f}")

# Overall row
rows.insert(0, {"Cohort": "Overall",
                "RF_RMSE": rf_rmse,  "RF_MAE": rf_mae,   "RF_R2":  rf_r2,
                "XGB_RMSE": xgb_rmse, "XGB_MAE": xgb_mae, "XGB_R2": xgb_r2})

metrics_df = pd.DataFrame(rows)
metrics_df.to_csv("metrics_summary.csv", index=False)
print("\n  -> metrics_summary.csv saved")

# ── 4. Plots ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 10))
fig.patch.set_facecolor("#0f1117")
gs = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

# --- Panel A: Actual vs Predicted scatter (both models) ----------------------
ax1 = fig.add_subplot(gs[0, 0])
lim = (y_test.min() * 0.95, y_test.max() * 1.05)
ax1.scatter(y_test, rf_pred,  alpha=0.35, s=18, color=RF_COLOR,  label=f"RF    R²={rf_r2:.3f}")
ax1.scatter(y_test, xgb_pred, alpha=0.35, s=18, color=XGB_COLOR, label=f"XGB  R²={xgb_r2:.3f}")
ax1.plot(lim, lim, "--", color="#8892b0", linewidth=1.2, label="Perfect fit")
ax1.set_xlim(lim); ax1.set_ylim(lim)
ax1.set_xlabel("Actual Bike Demand")
ax1.set_ylabel("Predicted Bike Demand")
ax1.set_title("Actual vs Predicted", weight="bold")
ax1.legend(facecolor="#1a1d27", edgecolor="#3a3f55", fontsize=9)
ax1.grid(True)

# --- Panel B: Per-cohort RMSE bar chart --------------------------------------
ax2 = fig.add_subplot(gs[0, 1])
cohort_rows = metrics_df[metrics_df["Cohort"] != "Overall"]
x   = np.arange(len(cohort_rows))
w   = 0.35
b1  = ax2.bar(x - w/2, cohort_rows["RF_RMSE"],  w, color=RF_COLOR,  alpha=0.85, label="Random Forest")
b2  = ax2.bar(x + w/2, cohort_rows["XGB_RMSE"], w, color=XGB_COLOR, alpha=0.85, label="XGBoost")
for bar in list(b1) + list(b2):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)
ax2.set_xticks(x)
ax2.set_xticklabels(cohort_rows["Cohort"], fontsize=9)
ax2.set_ylabel("RMSE")
ax2.set_title("Per-Cohort RMSE — RF vs XGBoost", weight="bold")
ax2.legend(facecolor="#1a1d27", edgecolor="#3a3f55")
ax2.grid(True, axis="y")

# --- Panel C: Feature importances (top 15, both models) ---------------------
ax3 = fig.add_subplot(gs[1, 0])
top_n = 15
rf_imp  = pd.Series(rf.feature_importances_,      index=FEATURES).nlargest(top_n).sort_values()
xgb_imp = pd.Series(xgb_model.feature_importances_, index=FEATURES)
xgb_imp = xgb_imp[rf_imp.index]   # align to same features

y_pos = np.arange(top_n)
ax3.barh(y_pos - 0.2, rf_imp.values,  0.4, color=RF_COLOR,  alpha=0.85, label="Random Forest")
ax3.barh(y_pos + 0.2, xgb_imp.values, 0.4, color=XGB_COLOR, alpha=0.85, label="XGBoost")
ax3.set_yticks(y_pos)
ax3.set_yticklabels([f.replace("_", " ") for f in rf_imp.index], fontsize=8)
ax3.set_xlabel("Feature Importance")
ax3.set_title(f"Top {top_n} Feature Importances", weight="bold")
ax3.legend(facecolor="#1a1d27", edgecolor="#3a3f55", fontsize=9)
ax3.grid(True, axis="x")

# --- Panel D: Per-cohort R² comparison ---------------------------------------
ax4 = fig.add_subplot(gs[1, 1])
rf_r2s  = cohort_rows["RF_R2"].values
xgb_r2s = cohort_rows["XGB_R2"].values
names   = cohort_rows["Cohort"].values

ax4.plot(names, rf_r2s,  "o-", color=RF_COLOR,  linewidth=2.5, markersize=9, label="Random Forest")
ax4.plot(names, xgb_r2s, "s--", color=XGB_COLOR, linewidth=2.5, markersize=9, label="XGBoost")
for i, (r, x) in enumerate(zip(rf_r2s, xgb_r2s)):
    ax4.text(i, r  + 0.008, f"{r:.3f}",  ha="center", fontsize=8, color=RF_COLOR)
    ax4.text(i, x  - 0.018, f"{x:.3f}",  ha="center", fontsize=8, color=XGB_COLOR)
ax4.set_ylabel("R2 Score")
ax4.set_title("Per-Cohort R2 — RF vs XGBoost", weight="bold")
ax4.legend(facecolor="#1a1d27", edgecolor="#3a3f55")
ax4.grid(True)

fig.suptitle(
    f"Phase 3 — Model Comparison  |  RF (R2={rf_r2:.3f}) vs XGBoost (R2={xgb_r2:.3f})",
    fontsize=14, weight="bold", color="#e8ecf4", y=1.01
)

plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight", facecolor="#0f1117")
plt.close()
print("  -> model_comparison.png saved")

# ── 5. Summary ────────────────────────────────────────────────────────────────
winner = "XGBoost" if xgb_r2 > rf_r2 else "Random Forest"
print(f"""
==========================================
  Phase 3 complete
==========================================
  Random Forest   RMSE={rf_rmse:.4f}  R2={rf_r2:.4f}
  XGBoost         RMSE={xgb_rmse:.4f}  R2={xgb_r2:.4f}
  Winner          {winner}
==========================================
""")
