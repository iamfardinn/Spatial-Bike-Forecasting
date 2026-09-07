# Seoul Bike-Sharing Demand Analysis

Spatial demand cohort analysis and machine learning prediction for Seoul's public bike-sharing system (Ttareungyi), using station-level GPS, land-use, transit, and weather features.

---

## Quick Start

```bash
pip install pandas numpy matplotlib scikit-learn xgboost contextily folium geopandas seaborn scipy
python main.py        # runs all 4 phases end-to-end (~23 s)
```

---

## Project Structure

```
BikeDemand3cohort/
|
|-- MGWR Analysis Data.csv        # Raw station data (2,631 stations, 24 features)
|
|-- main.py                       # Pipeline orchestrator — runs all phases in sequence
|
|-- 01_cohort_assignment.py       # Phase 1: K-Means spatial cohort clustering
|   |-- cohort_elbow.png          #   Elbow + silhouette plot (justifies k=4)
|   +-- cohort_profiles.png       #   Per-cohort feature boxplots
|
|-- 02_stratified_split.py        # Phase 2: Stratified 80/20 train/test split
|   +-- split_validation.png      #   Cohort proportion & demand distribution check
|
|-- 03_model_comparison.py        # Phase 3: Random Forest vs XGBoost comparison
|   |-- model_comparison.png      #   4-panel performance dashboard
|   +-- metrics_summary.csv       #   RMSE / MAE / R2 per model & cohort
|
|-- 04_cohort_map.py              # Phase 4: Cohort map + Moran's I autocorrelation
|   |-- cohort_map.png            #   Static GPS map (CartoDB Dark Matter basemap)
|   |-- cohort_map.html           #   Interactive Folium map (open in browser)
|   |-- cohort_centroids.csv      #   Per-cohort feature means
|   +-- morans_i_results.json     #   Global Moran's I spatial stats
|
|-- .gitignore
+-- README.md
```

> Generated CSVs (`MGWR_with_Cohorts.csv`, `train.csv`, `test.csv`) are not tracked — regenerate with `python main.py`.

---

## Phase 1 — K-Means Cohort Assignment

**Script:** `01_cohort_assignment.py`

Derives 4 spatial demand cohorts via K-Means clustering on:
- `Latitude`, `Longitude` (station location)
- `Bike_Demand` (average daily demand)

Features are standardized before clustering. The optimal k=4 was selected using the elbow method and silhouette analysis across k=2..10.

### Cohort Summary

| Cohort | Count | Mean Demand | Character |
|--------|-------|-------------|-----------|
| A (0)  | 730   | 6.09        | East Seoul, moderate demand |
| B (1)  | 796   | 6.11        | North Seoul, heavy transit use |
| C (2)  | 874   | 6.44        | West Seoul, largest group |
| D (3)  | 231   | **18.76**   | **High-demand hotspot stations** |

Cohort D isolates a concentrated group of high-demand stations (~3× mean demand of others), providing a critical stratification signal for downstream modelling.

### Outputs

| File | Description |
|------|-------------|
| `MGWR_with_Cohorts.csv` | Original data + `Cohort` column |
| `cohort_elbow.png` | Elbow + silhouette chart |
| `cohort_profiles.png` | Boxplots of 6 features by cohort |

---

## Phase 2 — Stratified Train / Test Split

**Script:** `02_stratified_split.py`

Performs an 80/20 train/test split **stratified on the `Cohort` column**, ensuring each cohort's demand distribution is faithfully represented in both partitions.

### Split Results

| Cohort | Total | Train | Test | Train% | Test% |
|--------|-------|-------|------|--------|-------|
| A | 730 | 584 | 146 | 80.0% | 20.0% |
| B | 796 | 636 | 160 | 79.9% | 20.1% |
| C | 874 | 699 | 175 | 80.0% | 20.0% |
| D | 231 | 185 |  46 | 80.1% | 19.9% |

### Outputs

| File | Description |
|------|-------------|
| `train.csv` | 2,104 stations (80%) |
| `test.csv` | 527 stations (20%) |
| `split_validation.png` | 4-panel cohort balance validation chart |

---

## Phase 3 — Random Forest vs XGBoost

**Script:** `03_model_comparison.py`

Trains and evaluates two regressors on the stratified split. Both models are broken down per-cohort to show where each excels.

### Overall Performance

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| Random Forest | 1.4959 | 0.4131 | 0.9209 |
| **XGBoost** | **0.8393** | **0.1450** | **0.9751** |

### Per-Cohort Performance

| Cohort | RF RMSE | XGB RMSE | RF R² | XGB R² |
|--------|---------|----------|-------|--------|
| A | 0.3227 | 0.1104 | 0.9894 | 0.9988 |
| B | 0.2797 | 0.0825 | 0.9905 | 0.9992 |
| C | 0.4143 | 0.0904 | 0.9773 | 0.9989 |
| D | 4.9375 | 2.8245 | 0.6577 | 0.8880 |

Cohort D (high-demand hotspots) is the hardest to predict for both models. XGBoost wins across every cohort and metric.

### Outputs

| File | Description |
|------|-------------|
| `model_comparison.png` | Actual vs Predicted, RMSE bars, feature importances, per-cohort R² |
| `metrics_summary.csv` | Full RMSE / MAE / R² table |

---

## Phase 4 — Cohort Map & Spatial Autocorrelation

**Script:** `04_cohort_map.py`

Produces a static and interactive map of all 2,631 stations coloured by cohort, and computes **Global Moran's I** to validate that the clusters are geographically contiguous (not random).

### Moran's I Results

| Statistic | Value |
|-----------|-------|
| Moran's I | **0.3551** |
| E[I] | −0.0004 |
| Z-score | **305.90** |
| p-value | **< 0.000001** |

A Moran's I of 0.355 with Z = 306 confirms **highly significant positive spatial autocorrelation** — the cohorts form genuine geographic zones, not statistical artifacts.

### Cohort Centroids

| Cohort | Latitude | Longitude | Mean Demand | Bus Boarding | Sub Boarding |
|--------|----------|-----------|-------------|--------------|--------------|
| A | 37.5125 | 127.0817 | 6.086 | 107.9 | 391.8 |
| B | 37.6079 | 127.0235 | 6.110 | 177.8 | 510.1 |
| C | 37.5234 | 126.9013 | 6.435 | 146.3 | 401.1 |
| D | 37.5418 | 126.9311 | 18.758 | 117.2 | 552.5 |

### Outputs

| File | Description |
|------|-------------|
| `cohort_map.png` | Static map with CartoDB Dark Matter basemap |
| `cohort_map.html` | Interactive Folium map (click stations for details) |
| `cohort_centroids.csv` | Per-cohort feature means |
| `morans_i_results.json` | Moran's I, Z-score, p-value |

---

## Dataset Features

| Group | Features |
|-------|----------|
| Spatial | `Latitude`, `Longitude`, `Slope` |
| Land-use | `Resident`, `Commercial`, `Industry`, `Green`, `Fp_perc`, `Pop_perc` |
| Transit | `Bus_boarding`, `Bus_alighting`, `Sub_boarding`, `Sub_alighting`, `Bike_on_count`, `Bike_off_count`, `NUM_BUS`, `NUM_SUB` |
| Weather | `Temperature_C`, `Precipitation_mm`, `Humidity`, `TinyDust` |
| Temporal | `Time` (average peak hour, 9-20) |
| **Target** | **`Bike_Demand`** (average daily trips) |

---

## Requirements

```
pandas
numpy
matplotlib
scikit-learn
xgboost
scipy
contextily
folium
geopandas
seaborn
```

Install all at once:
```bash
pip install pandas numpy matplotlib scikit-learn xgboost scipy contextily folium geopandas seaborn
```
