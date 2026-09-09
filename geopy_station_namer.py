"""
Geopy Station Namer for Seoul Bike-Sharing Network
==================================================
Uses `geopy.geocoders.Nominatim` to reverse-geocode station GPS coordinates
(Latitude, Longitude) to real-world station names, road names, and district names.

Features:
  * Persistent caching in outputs/tables/station_names_geopy.csv so you can
    run/stop/resume anytime without losing progress or repeating API calls.
  * Rate-limited to 1 request/sec per OpenStreetMap Nominatim usage policy.
  * Captures Korean name (amenity/POI), English road & borough/district name.
  * Command-line options:
      python geopy_station_namer.py --cohort D    (names all 231 Cohort D stations)
      python geopy_station_namer.py --top 15      (names top 15 busiest stations)
      python geopy_station_namer.py --all         (names all 2,631 stations across Seoul)
"""

import os
import sys
import time
import argparse
import pandas as pd

# Fix Windows console encoding for Korean characters
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

TBL_DIR = "outputs/tables"
CACHE_FILE = f"{TBL_DIR}/station_names_geopy.csv"
os.makedirs(TBL_DIR, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Reverse geocode Seoul bike stations using geopy")
    parser.add_argument("--cohort", type=str, default=None, choices=["A", "B", "C", "D"],
                        help="Geocode all stations in a specific cohort (e.g. 'D')")
    parser.add_argument("--top", type=int, default=None,
                        help="Geocode top N highest demand stations")
    parser.add_argument("--all", action="store_true",
                        help="Geocode all 2,631 stations in the dataset")
    return parser.parse_args()

def extract_names(loc_ko, loc_en):
    """Extract clean Korean and English names from geopy location objects."""
    name_ko = "Unknown"
    name_en = "Unknown"
    district = ""
    road = ""

    if loc_ko:
        addr_ko = loc_ko.raw.get("address", {})
        # Check amenity or POI tags
        for k in ["amenity", "subway", "railway", "station", "leisure", "tourism"]:
            if k in addr_ko:
                name_ko = addr_ko[k]
                break
        if name_ko == "Unknown":
            r = addr_ko.get("road", "")
            d = addr_ko.get("borough") or addr_ko.get("suburb", "")
            name_ko = f"{r} ({d})" if r and d else loc_ko.address.split(",")[0]

    if loc_en:
        addr_en = loc_en.raw.get("address", {})
        district = addr_en.get("borough") or addr_en.get("suburb") or ""
        road = addr_en.get("road") or ""
        for k in ["amenity", "subway", "railway", "station", "leisure", "tourism"]:
            if k in addr_en:
                name_en = addr_en[k]
                break
        if name_en == "Unknown":
            if road and district:
                name_en = f"{road}, {district}"
            elif road:
                name_en = road
            else:
                name_en = loc_en.address.split(",")[0]

    full_addr = loc_en.address if loc_en else (loc_ko.address if loc_ko else "")
    return name_ko, name_en, district, road, full_addr

def main():
    args = parse_args()
    input_csv = f"{TBL_DIR}/MGWR_with_Cohorts.csv"
    if not os.path.exists(input_csv):
        print(f"[ERROR] {input_csv} not found. Run 01_cohort_assignment.py first.")
        sys.exit(1)

    df = pd.read_csv(input_csv)
    cohort_map = {0: "A", 1: "B", 2: "C", 3: "D"}
    df["Cohort_Letter"] = df["Cohort"].map(cohort_map)

    # Filter targets
    if args.cohort:
        target_df = df[df["Cohort_Letter"] == args.cohort.upper()].copy()
        print(f"Targeting Cohort {args.cohort.upper()} ({len(target_df):,} stations)")
    elif args.top:
        target_df = df.sort_values("Bike_Demand", ascending=False).head(args.top).copy()
        print(f"Targeting Top {args.top} highest-demand stations")
    elif args.all:
        target_df = df.copy()
        print(f"Targeting all {len(target_df):,} stations")
    else:
        # Default: Cohort D (the busiest one)
        target_df = df[df["Cohort_Letter"] == "D"].copy()
        print("No filter specified. Defaulting to Cohort D (231 busiest stations).")
        print("Tip: Use --top 15 for a quick run, or --all for everything.")

    # Load cache if available
    cache = {}
    if os.path.exists(CACHE_FILE):
        cached_df = pd.read_csv(CACHE_FILE)
        for _, row in cached_df.iterrows():
            cache[int(row["ID"])] = row.to_dict()
        print(f"Loaded {len(cache):,} existing station names from cache: {CACHE_FILE}")

    # Initialize geolocator
    geolocator = Nominatim(user_agent="seoul_bike_cohort_study_v2")
    reverse_geocode = RateLimiter(geolocator.reverse, min_delay_seconds=1.0)

    results = []
    total = len(target_df)
    processed = 0

    print(f"\nStarting geopy reverse geocoding ({total} stations to process) ...")
    print("-" * 65)

    for idx, (_, row) in enumerate(target_df.iterrows(), 1):
        stn_id = int(row["ID"])
        lat = row["Latitude"]
        lon = row["Longitude"]
        dem = row["Bike_Demand"]
        cohort = row["Cohort_Letter"]

        # If already in cache, use cached version
        if stn_id in cache:
            results.append(cache[stn_id])
            continue

        processed += 1
        try:
            # Query Korean and English addresses
            loc_ko = reverse_geocode((lat, lon), language="ko")
            time.sleep(0.5)
            loc_en = reverse_geocode((lat, lon), language="en")

            name_ko, name_en, district, road, full_addr = extract_names(loc_ko, loc_en)

            entry = {
                "ID": stn_id,
                "Cohort": cohort,
                "Bike_Demand": round(dem, 3),
                "Latitude": lat,
                "Longitude": lon,
                "Station_Name_KO": name_ko,
                "Station_Name_EN": name_en,
                "District": district,
                "Road": road,
                "Full_Address": full_addr,
            }
            cache[stn_id] = entry
            results.append(entry)

            print(f"[{idx}/{total}] ID {stn_id:<5} | Dem: {dem:>5.1f} | {name_ko} ({district})")

            # Save periodically every 10 stations so progress is never lost
            if processed % 10 == 0:
                pd.DataFrame(list(cache.values())).to_csv(CACHE_FILE, index=False, encoding="utf-8-sig")

        except Exception as e:
            print(f"[{idx}/{total}] ID {stn_id:<5} | Error: {e}")

    # Final save of cache
    final_df = pd.DataFrame(list(cache.values()))
    final_df.sort_values("Bike_Demand", ascending=False, inplace=True)
    final_df.to_csv(CACHE_FILE, index=False, encoding="utf-8-sig")
    print("-" * 65)
    print(f"[DONE] Saved {len(final_df):,} station names to {CACHE_FILE}")

if __name__ == "__main__":
    main()
