"""
Pipeline Orchestrator -- Seoul Bike Demand Forecasting
=======================================================
Runs all 4 phases in sequence and prints a consolidated summary:

  Phase 1 --> K-Means cohort assignment
  Phase 2 --> Stratified 80/20 train/test split
  Phase 3 --> RF vs XGBoost model comparison
  Phase 4 --> Cohort map + Moran's I spatial autocorrelation

Usage:
  python main.py
"""

import subprocess
import sys
import json
import time
import os
import pandas as pd

FIG_DIR = "outputs/figures"
TBL_DIR = "outputs/tables"

# ── Helpers ───────────────────────────────────────────────────────────────────
def banner(text):
    line = "=" * 52
    print(f"\n{line}")
    print(f"  {text}")
    print(line)

def run_phase(script, label):
    banner(f"Running {label} --> {script}")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, script],
        capture_output=False,   # stream output directly to console
        text=True,
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n  [ERROR] {script} exited with code {result.returncode}")
        sys.exit(result.returncode)
    print(f"\n  [{label} done in {elapsed:.1f}s]")

# ── Run all phases ─────────────────────────────────────────────────────────────
PHASES = [
    ("01_cohort_assignment.py", "Phase 1 -- Cohort Assignment"),
    ("02_stratified_split.py",  "Phase 2 -- Stratified Split"),
    ("03_model_comparison.py",  "Phase 3 -- Model Comparison"),
    ("04_cohort_map.py",        "Phase 4 -- Cohort Map & Moran's I"),
]

total_start = time.time()
for script, label in PHASES:
    run_phase(script, label)

total_elapsed = time.time() - total_start

# ── Consolidated summary ───────────────────────────────────────────────────────
banner("PIPELINE COMPLETE  --  Final Summary")

# Phase 1: cohort counts from MGWR_with_Cohorts.csv
try:
    df = pd.read_csv(f"{TBL_DIR}/MGWR_with_Cohorts.csv")
    cohort_names = ["Cohort A", "Cohort B", "Cohort C", "Cohort D"]
    print("\n  Cohort Assignment (Phase 1)")
    print(f"  {'Cohort':<12} {'Count':>7} {'Mean Demand':>13}")
    print("  " + "-" * 34)
    for c, name in enumerate(cohort_names):
        sub = df[df["Cohort"] == c]
        print(f"  {name:<12} {len(sub):>7,} {sub['Bike_Demand'].mean():>13.3f}")
except Exception as e:
    print(f"  (cohort summary unavailable: {e})")

# Phase 2: split sizes
try:
    train_n = len(pd.read_csv(f"{TBL_DIR}/train.csv"))
    test_n  = len(pd.read_csv(f"{TBL_DIR}/test.csv"))
    print(f"\n  Train/Test Split (Phase 2)")
    print(f"  Train : {train_n:,} stations  ({train_n/(train_n+test_n)*100:.0f}%)")
    print(f"  Test  : {test_n:,} stations  ({test_n/(train_n+test_n)*100:.0f}%)")
except Exception as e:
    print(f"  (split summary unavailable: {e})")

# Phase 3: model metrics
try:
    metrics = pd.read_csv(f"{TBL_DIR}/metrics_summary.csv")
    overall = metrics[metrics["Cohort"] == "Overall"].iloc[0]
    rf_r2   = overall["RF_R2"]
    xgb_r2  = overall["XGB_R2"]
    winner  = "XGBoost" if xgb_r2 > rf_r2 else "Random Forest"
    print(f"\n  Model Performance (Phase 3)")
    print(f"  {'Model':<16} {'RMSE':>8} {'MAE':>8} {'R2':>8}")
    print("  " + "-" * 42)
    print(f"  {'Random Forest':<16} {overall['RF_RMSE']:>8.4f} {overall['RF_MAE']:>8.4f} {rf_r2:>8.4f}")
    print(f"  {'XGBoost':<16} {overall['XGB_RMSE']:>8.4f} {overall['XGB_MAE']:>8.4f} {xgb_r2:>8.4f}")
    print(f"  --> Winner: {winner}")
except Exception as e:
    print(f"  (model summary unavailable: {e})")

# Phase 4: Moran's I
try:
    with open(f"{TBL_DIR}/morans_i_results.json") as f:
        mi = json.load(f)
    print(f"\n  Spatial Autocorrelation (Phase 4)")
    print(f"  Moran's I : {mi['morans_I']:.4f}")
    print(f"  Z-score   : {mi['z_score']:.2f}")
    print(f"  p-value   : {mi['p_value']:.6f}")
    sig = "Significant" if mi["p_value"] < 0.05 else "Not significant"
    print(f"  Result    : {sig} positive spatial autocorrelation")
except Exception as e:
    print(f"  (Moran's I summary unavailable: {e})")

# Output artefacts
print(f"""
  Output Files
  ------------
  MGWR_with_Cohorts.csv    outputs/tables/
  train.csv / test.csv     outputs/tables/
  metrics_summary.csv      outputs/tables/
  morans_i_results.json    outputs/tables/
  cohort_centroids.csv     outputs/tables/
  cohort_elbow.png         outputs/figures/
  cohort_profiles.png      outputs/figures/
  split_validation.png     outputs/figures/
  model_comparison.png     outputs/figures/
  cohort_map.png           outputs/figures/
  cohort_map.html          outputs/figures/

  Total pipeline time: {total_elapsed:.1f}s
""")
