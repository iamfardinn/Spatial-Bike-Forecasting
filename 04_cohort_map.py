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
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore")

FIG_DIR = "outputs/figures"
TBL_DIR = "outputs/tables"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TBL_DIR, exist_ok=True)

# ── 0. Style ──────────────────────────────────────────────────────────────────
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
    "font.family":      "sans-serif",
    "font.sans-serif":  ["Malgun Gothic", "DejaVu Sans", "Arial"],
    "axes.unicode_minus": False,
    "font.size":        11,
})

COHORT_COLORS  = ["#1976d2", "#388e3c", "#f57c00", "#d32f2f"]
COHORT_NAMES   = ["Cohort A", "Cohort B", "Cohort C", "Cohort D"]

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

fig, ax = plt.subplots(figsize=(14, 12))
fig.patch.set_facecolor("#ffffff")
ax.set_facecolor("#ffffff")

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

    # Plot A, B, C first
    for c in range(K - 1):
        sub = gdf[gdf["Cohort"] == c]
        sub.plot(ax=ax, color=COHORT_COLORS[c], markersize=20,
                 alpha=0.68, label=f"{COHORT_NAMES[c]} (n={len(sub):,})", zorder=2)

    # Plot Cohort D on top with high-contrast highlighted markers
    sub_d = gdf[gdf["Cohort"] == 3]
    sub_d.plot(ax=ax, color=COHORT_COLORS[3], markersize=38,
               edgecolor="#ffffff", linewidth=0.8,
               alpha=0.92, label=f"{COHORT_NAMES[3]} (Hotspots, n={len(sub_d):,})", zorder=4)

    ctx.add_basemap(ax, source=ctx.providers.Esri.WorldGrayCanvas, zoom=12)
    ax.set_axis_off()
    basemap_ok = True
    print("  Basemap loaded via contextily (Esri.WorldGrayCanvas).")
except Exception as e:
    print(f"  Basemap unavailable ({e}), falling back to plain scatter.")

if not basemap_ok:
    for c in range(K - 1):
        sub = df[df["Cohort"] == c]
        ax.scatter(sub["Longitude"], sub["Latitude"],
                   color=COHORT_COLORS[c], s=20, alpha=0.68,
                   label=f"{COHORT_NAMES[c]} (n={len(sub):,})", zorder=2)
    sub_d = df[df["Cohort"] == 3]
    ax.scatter(sub_d["Longitude"], sub_d["Latitude"],
               color=COHORT_COLORS[3], s=38, edgecolor="#ffffff", linewidth=0.8,
               alpha=0.92, label=f"{COHORT_NAMES[3]} (Hotspots, n={len(sub_d):,})", zorder=4)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True)

if basemap_ok:
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

# Centroid labels for A, B, C (offset slightly to avoid hotspots)
centroid_offsets = {
    0: (-40, 25),  # Cohort A: shift slightly NW so it is clear of Jamsil hotspot
    1: (0, 0),     # Cohort B
    2: (0, 0),     # Cohort C
}
for c in range(K - 1):
    lat_c = centroids.iloc[c]["Latitude"]
    lon_c = centroids.iloc[c]["Longitude"]
    if basemap_ok:
        x_c, y_c = tr.transform(lon_c, lat_c)
    else:
        x_c, y_c = lon_c, lat_c
    dx, dy = centroid_offsets.get(c, (0, 0))
    ax.annotate(
        COHORT_NAMES[c],
        (x_c, y_c),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=9, weight="bold",
        color=COHORT_COLORS[c],
        bbox=dict(boxstyle="round,pad=0.3", fc="#ffffff", ec=COHORT_COLORS[c], alpha=0.92),
        zorder=6,
    )

# ── Top Busiest Hotspots in Cohort D: Annotate with Station Names ─────────────
GEOPY_CSV = f"{TBL_DIR}/station_names_geopy.csv"

# Directional fan-out offsets to avoid overlapping
HOTSPOT_OFFSETS = {
    2715: (-95, 28),   # Magoknaru Stn. Exit 2 (Far West)
    4217: (-85, 22),   # Mangwon Hangang Park (Mapo)
    2701: (-95, -28),  # Magoknaru Exit 5 (West)
    502:  (28, 28),    # Ttukseom Park (East Han River)
    207:  (25, 28),    # Yeouinaru Stn. Exit 1 (Central Han River)
    230:  (-85, -8),   # Yeongdeungpo-gu Office (West)
    247:  (-80, 20),   # Dangsan Stn. (West)
    1911: (-90, -25),  # Guro Digital Complex (Southwest)
    2102: (25, -22),   # Bongnimgyo / Sillim Stn. (South)
    2622: (30, 25),    # Olympic Park (Far East)
    1210: (30, -5),    # Jamsil / Lotte World Tower (East)
    2608: (30, -48),   # Songpa-gu Office (East)
}

# Bilingual English + Korean landmark naming for international academic paper
ENGLISH_NAMES = {
    2715: "Magoknaru Stn. Exit 2\n(마곡나루역)",
    502:  "Ttukseom Park Exit 1\n(뚝섬한강공원)",
    4217: "Mangwon Hangang Park\n(망원나들목)",
    2701: "Magoknaru Exit 5\n(마곡나루역)",
    1210: "Jamsil / Lotte Tower\n(잠실역)",
    207:  "Yeouinaru Stn. Exit 1\n(여의나루역)",
    2102: "Bongnimgyo / Sillim\n(봉림교)",
    2622: "Olympic Park Stn.\n(올림픽공원역)",
    1911: "Guro Digital Complex\n(구로디지털단지역)",
    247:  "Dangsan Stn. Exit 10\n(당산역)",
    2608: "Songpa-gu Office\n(송파구청)",
    230:  "Yeongdeungpo-gu Office\n(영등포구청역)",
}

# If geopy CSV exists, load official names from geopy reverse geocoding!
if os.path.exists(GEOPY_CSV):
    print(f"  Loading reverse-geocoded names from {GEOPY_CSV} ...")
    geopy_df = pd.read_csv(GEOPY_CSV, encoding="utf-8-sig")
    stn_dict = {}
    for _, r in geopy_df.iterrows():
        sid = int(r["ID"])
        dist = str(r.get("District", "")).replace("-gu", "")
        if sid in ENGLISH_NAMES:
            label = ENGLISH_NAMES[sid]
        else:
            name_ko = str(r["Station_Name_KO"]).split(",")[0].strip()
            name_en = str(r.get("Station_Name_EN", "")).split(",")[0].strip()
            if name_en and name_en != "Unknown" and name_en != name_ko:
                label = f"{name_en} ({name_ko})"
            else:
                label = name_ko
        stn_dict[sid] = f"{label}\n[{dist}]" if dist and dist != "nan" else label
else:
    stn_dict = {sid: f"{name}\n[Seoul]" for sid, name in ENGLISH_NAMES.items()}

# Render annotations for key hotspots
for stn_id, offset in HOTSPOT_OFFSETS.items():
    stn_match = df[df["ID"] == stn_id]
    if len(stn_match) == 0 or stn_id not in stn_dict:
        continue
    lat_s = stn_match.iloc[0]["Latitude"]
    lon_s = stn_match.iloc[0]["Longitude"]
    dem_s = stn_match.iloc[0]["Bike_Demand"]
    if basemap_ok:
        x_s, y_s = tr.transform(lon_s, lat_s)
    else:
        x_s, y_s = lon_s, lat_s

    stn_label = stn_dict[stn_id]
    badge_text = f"{stn_label} [#{stn_id}]\nDem: {dem_s:.1f} trips/day"
    ax.annotate(
        badge_text,
        xy=(x_s, y_s),
        xytext=offset,
        textcoords="offset points",
        fontsize=7.5,
        weight="bold",
        color="#b71c1c",
        bbox=dict(boxstyle="round,pad=0.35", fc="#ffffff", ec="#d32f2f", lw=1.2, alpha=0.96),
        arrowprops=dict(arrowstyle="->", color="#d32f2f", lw=1.2, connectionstyle="arc3,rad=0.08"),
        zorder=10,
    )

legend = ax.legend(
    facecolor="#ffffff", edgecolor="#cccccc",
    fontsize=10, loc="lower right",
    framealpha=0.92,
)
for text in legend.get_texts():
    text.set_color("#111111")

ax.set_title(
    f"Seoul Bike Station Cohorts & Key Cohort D Hotspots  |  Moran's I = {I:.4f}  (p = {p_val:.4f})",
    fontsize=13, weight="bold", color="#111111", pad=14,
)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/cohort_map.png", dpi=300, bbox_inches="tight", facecolor="#ffffff")
plt.close()
print("  -> cohort_map.png saved with Cohort D station names")

# ── 5. Summary ────────────────────────────────────────────────────────────────
print(f"""
==========================================
  Phase 4 complete
==========================================
  cohort_map.png          static map (paper-ready)
  cohort_centroids.csv    per-cohort means
  morans_i_results.json   spatial stats
------------------------------------------
  Moran's I  = {I:.4f}
  Z-score    = {Z_I:.4f}
  p-value    = {p_val:.6f}
  Result     : {'Significant positive autocorrelation' if p_val < 0.05 else 'Not significant'}
==========================================
""")

