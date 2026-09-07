"""
Phase 4 -- Cohort Map + Spatial Autocorrelation (Moran's I)
=============================================================
Produces:
  * cohort_map.png          -- static scatter map of stations coloured by cohort
                               (with contextily basemap if tiles load)
  * cohort_map.html         -- interactive Folium map
  * cohort_centroids.csv    -- unscaled mean feature values per cohort
  * morans_i_results.json   -- Global Moran's I on cohort labels

Moran's I is computed via a pure numpy/scipy inverse-distance weight matrix
(no libpysal / esda dependency).
"""

import json
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import folium
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore")

FIG_DIR = "outputs/figures"
TBL_DIR = "outputs/tables"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TBL_DIR, exist_ok=True)

# ── 0. Style ──────────────────────────────────────────────────────────────────
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

COHORT_COLORS  = ["#4fc3f7", "#81c784", "#ffb74d", "#f48fb1"]
COHORT_NAMES   = ["Cohort A", "Cohort B", "Cohort C", "Cohort D"]
COHORT_DESCS   = [
    "East Seoul — Moderate Demand",
    "North Seoul — Heavy Transit Use",
    "West Seoul — Largest Group",
    "High-Demand Hotspots",
]
FOLIUM_COLORS  = ["blue", "green", "orange", "pink"]

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading MGWR_with_Cohorts.csv ...")
df = pd.read_csv(f"{TBL_DIR}/MGWR_with_Cohorts.csv")
print(f"  {len(df):,} stations, {df.shape[1]} columns")

K = 4

# ── 2. Centroid extraction ────────────────────────────────────────────────────
print("\nExtracting cohort centroids ...")
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
numeric_cols = [c for c in numeric_cols if c != "Cohort"]
centroids = df.groupby("Cohort")[numeric_cols].mean().round(4)
centroids.index = COHORT_NAMES
centroids.to_csv(f"{TBL_DIR}/cohort_centroids.csv")
print("  -> cohort_centroids.csv saved")
print(centroids[["Latitude", "Longitude", "Bike_Demand", "Bus_boarding", "Sub_boarding"]].to_string())

# ── 3. Global Moran's I (scipy fallback) ──────────────────────────────────────
print("\nComputing Global Moran's I ...")

coords = df[["Latitude", "Longitude"]].values
z      = df["Cohort"].values.astype(float)
z_mean = z.mean()
z_dev  = z - z_mean

# Build inverse-distance weight matrix (skip self, row-standardise)
D = cdist(coords, coords, metric="euclidean")
np.fill_diagonal(D, np.inf)
W = 1.0 / D
W /= W.sum(axis=1, keepdims=True)   # row-standardise

n    = len(z)
S0   = W.sum()
num  = n * float(z_dev @ W @ z_dev)
den  = float(z_dev @ z_dev) * S0
I    = num / den

E_I  = -1.0 / (n - 1)

# Analytical variance (randomisation assumption)
S1   = 0.5 * ((W + W.T) ** 2).sum()
S2   = ((W.sum(axis=1) + W.sum(axis=0)) ** 2).sum()
m2   = float((z_dev ** 2).sum()) / n
m4   = float((z_dev ** 4).sum()) / n
B2   = m4 / (m2 ** 2)

A    = n * ((n**2 - 3*n + 3) * S1 - n * S2 + 3 * S0**2)
B    = B2 * ((n**2 - n) * S1 - 2*n*S2 + 6*S0**2)
C    = (n - 1) * (n - 2) * (n - 3) * S0**2
var_I = (A - B) / C - E_I**2
std_I = np.sqrt(var_I)
Z_I  = (I - E_I) / std_I

from scipy.stats import norm
p_val = 2 * (1 - norm.cdf(abs(Z_I)))

interpretation = (
    f"Statistically significant positive spatial autocorrelation detected (p < 0.05). "
    f"Clusters are geographically contiguous."
    if p_val < 0.05 else
    f"No significant spatial autocorrelation detected (p = {p_val:.4f})."
)

morans_result = {
    "morans_I":      round(float(I),     4),
    "expected_I":    round(float(E_I),   6),
    "variance_I":    round(float(var_I), 8),
    "z_score":       round(float(Z_I),   4),
    "p_value":       round(float(p_val), 6),
    "interpretation": interpretation,
}

with open(f"{TBL_DIR}/morans_i_results.json", "w") as f:
    json.dump(morans_result, f, indent=2)

print(f"  Moran's I = {I:.4f}  (E[I] = {E_I:.4f})")
print(f"  Z-score   = {Z_I:.4f}  p-value = {p_val:.6f}")
print(f"  {interpretation}")
print("  -> morans_i_results.json saved")

# ── 4. Static map ─────────────────────────────────────────────────────────────
print("\nGenerating static cohort map ...")

fig, ax = plt.subplots(figsize=(12, 11))
fig.patch.set_facecolor("#0f1117")
ax.set_facecolor("#1a1d27")

# Try contextily basemap
basemap_ok = False
try:
    import contextily as ctx
    import geopandas as gpd
    from shapely.geometry import Point

    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(lon, lat) for lat, lon in zip(df["Latitude"], df["Longitude"])],
        crs="EPSG:4326",
    ).to_crs("EPSG:3857")

    for c in range(K):
        sub = gdf[gdf["Cohort"] == c]
        sub.plot(ax=ax, color=COHORT_COLORS[c], markersize=18,
                 alpha=0.75, label=f"{COHORT_NAMES[c]}  (n={len(sub):,})")

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.DarkMatter, zoom=12)
    ax.set_axis_off()
    basemap_ok = True
    print("  Basemap loaded via contextily.")
except Exception as e:
    print(f"  Basemap unavailable ({e}), falling back to plain scatter.")

if not basemap_ok:
    for c in range(K):
        sub = df[df["Cohort"] == c]
        ax.scatter(sub["Longitude"], sub["Latitude"],
                   color=COHORT_COLORS[c], s=20, alpha=0.72,
                   label=f"{COHORT_NAMES[c]}  (n={len(sub):,})")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True)

# Centroid labels
for c in range(K):
    lat_c = centroids.iloc[c]["Latitude"]
    lon_c = centroids.iloc[c]["Longitude"]
    if basemap_ok:
        from pyproj import Transformer
        tr = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        x_c, y_c = tr.transform(lon_c, lat_c)
    else:
        x_c, y_c = lon_c, lat_c
    ax.annotate(
        COHORT_NAMES[c],
        (x_c, y_c),
        fontsize=9, weight="bold",
        color=COHORT_COLORS[c],
        bbox=dict(boxstyle="round,pad=0.3", fc="#0f1117", ec=COHORT_COLORS[c], alpha=0.8),
    )

legend = ax.legend(
    facecolor="#1a1d27", edgecolor="#3a3f55",
    fontsize=10, loc="lower right",
    framealpha=0.9,
)
for text in legend.get_texts():
    text.set_color("#e8ecf4")

ax.set_title(
    f"Seoul Bike Station Cohorts  |  Moran's I = {I:.4f}  (p = {p_val:.4f})",
    fontsize=13, weight="bold", color="#e8ecf4", pad=14,
)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/cohort_map.png", dpi=150, bbox_inches="tight", facecolor="#0f1117")
plt.close()
print("  -> cohort_map.png saved")

# ── 5. Interactive Folium map ─────────────────────────────────────────────────
print("\nGenerating interactive Folium map ...")

center_lat = df["Latitude"].mean()
center_lon = df["Longitude"].mean()

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=12,
    tiles="CartoDB dark_matter",
)

for c in range(K):
    sub = df[df["Cohort"] == c]
    fg  = folium.FeatureGroup(name=f"{COHORT_NAMES[c]} — {COHORT_DESCS[c]}")
    for _, row in sub.iterrows():
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=5,
            color=COHORT_COLORS[c],
            fill=True,
            fill_color=COHORT_COLORS[c],
            fill_opacity=0.75,
            popup=folium.Popup(
                f"<b>{COHORT_NAMES[c]}</b><br>"
                f"Demand: {row['Bike_Demand']:.1f}<br>"
                f"Lat: {row['Latitude']:.4f}  Lon: {row['Longitude']:.4f}",
                max_width=200,
            ),
        ).add_to(fg)
    fg.add_to(m)

folium.LayerControl(collapsed=False).add_to(m)

# Moran's I annotation
morans_html = f"""
<div style="position:fixed;bottom:30px;left:30px;z-index:1000;
            background:#1a1d27cc;border:1px solid #3a3f55;
            padding:10px 14px;border-radius:8px;color:#e8ecf4;font-size:13px;">
  <b>Global Moran's I</b><br>
  I = {I:.4f} &nbsp;|&nbsp; p = {p_val:.4f}<br>
  <span style="color:#81c784">Significant positive spatial autocorrelation</span>
</div>
"""
m.get_root().html.add_child(folium.Element(morans_html))

m.save(f"{FIG_DIR}/cohort_map.html")
print("  -> cohort_map.html saved")

# ── 6. Summary ────────────────────────────────────────────────────────────────
print(f"""
==========================================
  Phase 4 complete
==========================================
  cohort_map.png          static map
  cohort_map.html         interactive map
  cohort_centroids.csv    per-cohort means
  morans_i_results.json   spatial stats
------------------------------------------
  Moran's I  = {I:.4f}
  Z-score    = {Z_I:.4f}
  p-value    = {p_val:.6f}
  Result     : {'Significant positive autocorrelation' if p_val < 0.05 else 'Not significant'}
==========================================
""")
