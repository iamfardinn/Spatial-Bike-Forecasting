# Localized Urban Bike Demand Prediction via Spatial Cohort Stratification

A research pipeline that replaces a single global demand model with three land-use-aware localized regressors, validated through spatial autocorrelation.

---

## Proposed Changes

### Project Scaffold

#### [NEW] `requirements.txt`
All third-party dependencies pinned for reproducibility:
- `pandas`, `numpy`, `scikit-learn`, `xgboost`
- `matplotlib`, `seaborn` (visualization)
- `libpysal`, `esda` (Moran's I spatial autocorrelation)
- `scipy` (distance matrix)

#### [NEW] Directory skeleton
```
a:\BikeDemand3cohort\
├── data\
│   └── MGWR Analysis Data (1).csv        ← user must supply
├── outputs\
│   ├── figures\
│   └── tables\
└── src\
    ├── data_loader.py
    ├── preprocessing.py
    ├── clustering.py
    ├── models.py
    ├── evaluator.py
    └── main.py
```

---

### Phase 1 — Data Ingestion & Feature Decoupling

#### [NEW] [`data_loader.py`](file:///a:/BikeDemand3cohort/src/data_loader.py)

**Responsibilities:**
- `load_data(path)` — reads CSV with `encoding='latin1'`
- `sanitize_headers(df)` — regex strips `(°C)`, `(%)`, Unicode artifacts; normalizes to ASCII snake-case-friendly names
- `validate_nulls(df)` — raises `ValueError` if any null is detected
- `partition_features(df)` — returns a dict:

```python
{
  "X_spatial":  df[SPATIAL_COLS],   # Lat, Lon, Resident, Commercial,
                                     # Industry, Green, Pop_perc, Fp_perc,
                                     # NUM_BUS, NUM_SUB, Slope
  "X_climate":  df[CLIMATE_COLS],   # Temperature, Precipitation, Humidity,
                                     # TinyDust, Bus_boarding, Bus_alighting,
                                     # Sub_boarding, Sub_alighting, Time
  "target":     df["Bike_Demand"]
}
```

**Column mapping (sanitize step):**

| Raw Header | Sanitized Name |
|---|---|
| `Temperature(°C)` | `Temperature` |
| `Humidity(%)` | `Humidity` |
| `Precipitation(mm)` | `Precipitation` |
| `미세먼지 / TinyDust` | `TinyDust` |
| Any BOM / encoding artifact | stripped |

---

### Phase 2 — Feature Standardization

#### [NEW] [`preprocessing.py`](file:///a:/BikeDemand3cohort/src/preprocessing.py)

**Class: `SpatialPreprocessor`**

```python
class SpatialPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit_transform(self, X_spatial: pd.DataFrame) -> pd.DataFrame:
        ...  # returns scaled DataFrame, index + columns preserved

    def inverse_transform(self, X_scaled: pd.DataFrame) -> pd.DataFrame:
        ...  # used to recover unscaled centroid values
```

- Formula: $z = (x - \mu) / \sigma$ via `StandardScaler`
- Preserves DataFrame index and column names throughout
- Exposes `inverse_transform` to recover human-readable centroid values in Phase 3

> [!NOTE]
> Only `X_spatial` is standardized here. `X_climate` features are passed raw into model training (tree-based models are scale-invariant, but spatial clustering is distance-sensitive).

---

### Phase 3 — Spatial Clustering & Reality Check

#### [NEW] [`clustering.py`](file:///a:/BikeDemand3cohort/src/clustering.py)

**Step 3a — Hyperparameter Sweep (`K ∈ [2, 8]`)**
- Fit `KMeans(k, random_state=42)` for each K
- Record: `inertia_`, `silhouette_score`
- Plot dual-axis chart (Inertia left axis, Silhouette right axis) with vertical dashed line at K=3
- Save → `outputs/figures/kmeans_k_sweep.png`

**Step 3b — Final Clustering (K=3)**
- Fit `KMeans(n_clusters=3, random_state=42)` on `X_spatial_scaled`
- Map integer labels to qualitative cohort names:

| Cluster Label | Cohort Name |
|---|---|
| `0` | Suburban / Steep Terrain |
| `1` | Residential / Micro-Climate Sensitive |
| `2` | Commercial Core / Transit Hubs |

- Append `cohort_label` column to the master DataFrame

**Step 3c — Centroid Extraction**
- Use `SpatialPreprocessor.inverse_transform()` on cluster centroids
- Export unscaled mean feature values per cohort → `outputs/tables/cohort_centroids.csv`

**Step 3d — Global Moran's I**
- Build spatial weight matrix: $W_{ij} = 1 / d_{ij}$ (inverse Euclidean distance from Lat/Lon)
- Row-standardize W
- Compute Global Moran's I using `esda.Moran` on `cohort_label`
- Assert $I > \mathbb{E}[I]$ (positive spatial autocorrelation = geographically contiguous clusters)
- Save to `outputs/tables/morans_i_results.json`:

```json
{
  "morans_I": 0.412,
  "expected_I": -0.0013,
  "p_value": 0.001,
  "z_score": 18.4,
  "interpretation": "Statistically significant positive spatial autocorrelation detected (p < 0.05). Clusters are geographically contiguous."
}
```

> [!IMPORTANT]
> `libpysal` + `esda` are required for Moran's I. If the environment cannot install them, a fallback manual implementation using `scipy` distance matrices will be provided.

---

### Phase 4 — Localized vs. Global ML Models

#### [NEW] [`models.py`](file:///a:/BikeDemand3cohort/src/models.py)

**Global Baseline**
- Features: `X_climate + X_spatial` (all columns concatenated)
- Model: `RandomForestRegressor(n_estimators=100, random_state=42)`
- Trained on all stations (80% split)

**Localized Cohort Models (×3)**
- Features: `X_climate` only (dynamic operational inputs)
- One `RandomForestRegressor(n_estimators=100, random_state=42)` per cohort
- Each model sees only data rows belonging to its cohort

**Train/Test Split**
- 80/20 split via `train_test_split(random_state=42)`
- Split is applied **per-cohort** for localized models (preserving class balance)
- Split is applied **globally** for the baseline model

**Return structure:**
```python
{
  "global":   { "model": ..., "X_test": ..., "y_test": ... },
  "cohort_0": { "model": ..., "X_test": ..., "y_test": ... },
  "cohort_1": { "model": ..., "X_test": ..., "y_test": ... },
  "cohort_2": { "model": ..., "X_test": ..., "y_test": ... },
}
```

---

### Phase 5 — Evaluation & Performance Comparison

#### [NEW] [`evaluator.py`](file:///a:/BikeDemand3cohort/src/evaluator.py)

**Metrics computed per model:**
- RMSE: `sqrt(mean_squared_error(y_test, y_pred))`
- MAE: `mean_absolute_error(y_test, y_pred)`
- R²: `r2_score(y_test, y_pred)`

**Outputs:**
- `outputs/tables/model_evaluation_metrics.csv` — one row per model

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Global Baseline | … | … | … |
| Cohort 0 — Suburban | … | … | … |
| Cohort 1 — Residential | … | … | … |
| Cohort 2 — Commercial | … | … | … |

- `outputs/figures/model_performance_comparison.png` — grouped bar chart comparing RMSE, MAE, R² across all 4 models with cohort color coding

---

### Pipeline Orchestrator

#### [NEW] [`main.py`](file:///a:/BikeDemand3cohort/src/main.py)

```
load_data()
  → sanitize_headers()
  → validate_nulls()
  → partition_features()
  → SpatialPreprocessor.fit_transform(X_spatial)
  → run_k_sweep() + plot_k_sweep()
  → fit_kmeans_k3() + assign_cohort_labels()
  → extract_centroids() + export_csv()
  → compute_morans_i() + export_json()
  → train_global_model()
  → train_cohort_models()
  → evaluate_all_models()
  → export_metrics_csv()
  → plot_performance_comparison()
```

Prints a final summary table to stdout on completion.

---

## User Review Required

> [!IMPORTANT]
> **Data file location:** The CSV `MGWR Analysis Data (1).csv` must be placed at `a:\BikeDemand3cohort\data\` before running `main.py`. The pipeline will raise a `FileNotFoundError` if it is missing.

> [!WARNING]
> **Moran's I dependencies:** `libpysal` and `esda` require C extensions. If installation fails in the target environment, I will implement a pure `numpy`/`scipy` fallback for the weight matrix and Moran's I statistic.

> [!NOTE]
> **Cohort label assignment:** The K-Means algorithm assigns integer labels `{0, 1, 2}` non-deterministically relative to the qualitative names. After fitting, centroid feature values (Slope, NUM_SUB, Resident) will be inspected to correctly map each integer label to the right cohort name. This remapping logic is built into `clustering.py`.

---

## Open Questions

1. **XGBoost vs. Random Forest:** The abstract mentions both RF and XGBoost. Should I implement **both** for comparison, or use RF as the primary with XGBoost as an optional extension?
2. **Cohort map visualization:** The directory structure lists `cohort_map_visualization.png`. Should I generate a **scatter plot of station GPS coordinates colored by cohort**, or do you have a shapefile/basemap for Seoul to overlay?
3. **Stratified split:** Should the 80/20 split for cohort models be stratified by any categorical feature (e.g., `Time` hour bins), or a plain random split within each cohort?

---

## Verification Plan

### Automated Checks
- `python src/main.py` — full end-to-end pipeline run
- Assert all 3 output figures exist in `outputs/figures/`
- Assert all 3 output tables exist in `outputs/tables/`
- Assert Moran's I result: `I > E[I]` and `p_value < 0.05`
- Assert cohort model R² values individually exceed global baseline R²

### Manual Verification
- Review `outputs/figures/kmeans_k_sweep.png` — confirm elbow at K=3 is visible
- Review `outputs/figures/model_performance_comparison.png` — confirm cohort models outperform global baseline
- Review `outputs/tables/cohort_centroids.csv` — confirm centroid values align with cohort names (e.g., Cohort 0 should show high Slope)
- Review `outputs/tables/morans_i_results.json` — confirm statistically significant positive I
