# Seoul Bike-Sharing Demand Analysis

Spatial demand cohort analysis and machine learning prediction for Seoul's public bike-sharing system (Ttareungyi), using station-level GPS, land-use, transit, and weather features.

---

## Project Structure

```
BikeDemand3cohort/
|
|-- MGWR Analysis Data.csv        # Raw station data (2,631 stations, 24 features)
|
|-- 01_cohort_assignment.py       # Phase 1: K-Means spatial cohort clustering
|   |-- cohort_elbow.png          #   Elbow + silhouette plot (justifies k=4)
|   +-- cohort_profiles.png       #   Per-cohort feature boxplots
|
|-- .gitignore
+-- README.md
```

> Subsequent phases (stratified split, RF vs XGBoost, cohort map) will be added as separate scripts.

---

## Phase 1 -- K-Means Cohort Assignment

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

Cohort D isolates a concentrated group of high-demand stations (~3x mean demand of others), providing a critical stratification signal for downstream modelling.

### Outputs

| File | Description |
|------|-------------|
| `MGWR_with_Cohorts.csv` | Original data + `Cohort` column (generated, not tracked) |
| `cohort_elbow.png` | Elbow + silhouette chart |
| `cohort_profiles.png` | Boxplots of 6 features by cohort |

### How to run

```bash
pip install pandas numpy matplotlib scikit-learn
python 01_cohort_assignment.py
```

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
xgboost          # Phase 3
contextily       # Phase 4
folium           # Phase 4
geopandas        # Phase 4
seaborn          # Phase 3
```

Install all at once:
```bash
pip install pandas numpy matplotlib scikit-learn xgboost contextily folium geopandas seaborn
```
