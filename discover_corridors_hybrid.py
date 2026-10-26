
from pathlib import Path
import warnings

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hdbscan
from shapely.geometry import box

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# 1. CONFIGURATION
# ============================================================
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

PLACES_FILE = DATA / "places_filtered.geojson"
ROADS_FILE = DATA / "roads_in_study_area.geojson"

WEST, SOUTH, EAST, NORTH = -87.683, 41.903, -87.659, 41.913
BOUNDARY = box(WEST, SOUTH, EAST, NORTH)

ROAD_BUFFER_M = 50
CLUSTER_SIZE = 10
CLUSTER_MIN_SAMPLES = 5
BIN_LENGTH_M = 100

# ============================================================
# 2. LOAD AND PREPARE DATA
# ============================================================
print("Loading Chicago places and roads...")
DATA.mkdir(parents=True, exist_ok=True)

if not PLACES_FILE.exists():
    raise FileNotFoundError(f"Places file not found: {PLACES_FILE}")

if not ROADS_FILE.exists():
    raise FileNotFoundError(f"Roads file not found: {ROADS_FILE}")

places = gpd.read_file(PLACES_FILE)
roads = gpd.read_file(ROADS_FILE)

if places.empty:
    raise ValueError("The places file contains no records.")

if roads.empty:
    raise ValueError("The roads file contains no records.")

if places.crs is None:
    places = places.set_crs("EPSG:4326")

if roads.crs is None:
    roads = roads.set_crs("EPSG:4326")

places = places.to_crs("EPSG:4326")
roads = roads.to_crs("EPSG:4326")

places = places[
    places.geometry.notna() & ~places.geometry.is_empty
].copy()

roads = roads[
    roads.geometry.notna() & ~roads.geometry.is_empty
].copy()

places = places[places.geometry.intersects(BOUNDARY)].copy()
roads = roads[roads.geometry.intersects(BOUNDARY)].copy()

if places.empty:
    raise ValueError("No places found inside the study boundary.")

if roads.empty:
    raise ValueError("No roads found inside the study boundary.")

print(f"Places inside boundary: {len(places)}")
print(f"Road records inside boundary: {len(roads)}")

# Convert to metres for buffers and distances.
places = places.to_crs("EPSG:32616").reset_index(drop=True)
roads = roads.to_crs("EPSG:32616").reset_index(drop=True)

boundary_gdf = gpd.GeoDataFrame(
    {"name": ["Selected study area"]},
    geometry=[BOUNDARY],
    crs="EPSG:4326"
).to_crs("EPSG:32616")

boundary_geom = boundary_gdf.geometry.iloc[0]

roads["geometry"] = roads.geometry.intersection(boundary_geom)

roads = roads[
    roads.geometry.notna()
    & ~roads.geometry.is_empty
    & (roads.geometry.length > 0)
].copy().reset_index(drop=True)

if roads.empty:
    raise ValueError("No usable road geometry remains after clipping.")

# ============================================================
# 3. CLUSTER PLACES USING HDBSCAN
# ============================================================
print("Clustering places with HDBSCAN...")

places["cluster_point"] = places.geometry.representative_point()
places["x"] = places["cluster_point"].x
places["y"] = places["cluster_point"].y
places["place_id"] = np.arange(len(places), dtype=int)

coords = places[["x", "y"]].to_numpy()

if len(coords) < CLUSTER_SIZE:
    places["cluster_id"] = -1
else:
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=CLUSTER_SIZE,
        min_samples=CLUSTER_MIN_SAMPLES,
        metric="euclidean",
        cluster_selection_method="eom"
    )
    places["cluster_id"] = clusterer.fit_predict(coords)

cluster_count = len(set(places["cluster_id"]) - {-1})
clustered_count = int((places["cluster_id"] >= 0).sum())

print(f"Place clusters found: {cluster_count}")
print(f"Places assigned to clusters: {clustered_count}")

place_points = gpd.GeoDataFrame(
    {
        "place_id": places["place_id"].to_numpy(),
        "cluster_id": places["cluster_id"].to_numpy()
    },
    geometry=places["cluster_point"].to_numpy(),
    crs=places.crs
).reset_index(drop=True)

places_save = places.drop(
    columns=["cluster_point", "x", "y"]
).to_crs("EPSG:4326")

places_save.to_file(
    DATA / "hdbscan_places.geojson",
    driver="GeoJSON"
)

# ============================================================
# 4. EXTRACT STREET NAMES
# ============================================================
def extract_street_name(row):
    for field in ("name", "street_name", "road_name", "names"):
        if field not in row.index:
            continue

        value = row[field]

        if isinstance(value, str) and value.strip():
            return value.strip()

        if isinstance(value, dict):
            primary = value.get("primary")

            if isinstance(primary, str) and primary.strip():
                return primary.strip()

            for item in value.values():
                if isinstance(item, str) and item.strip():
                    return item.strip()

                if isinstance(item, dict):
                    candidate = item.get("value") or item.get("name")
                    if isinstance(candidate, str) and candidate.strip():
                        return candidate.strip()

                if isinstance(item, list):
                    for subitem in item:
                        if isinstance(subitem, str) and subitem.strip():
                            return subitem.strip()

                        if isinstance(subitem, dict):
                            candidate = (
                                subitem.get("value")
                                or subitem.get("name")
                            )
                            if isinstance(candidate, str) and candidate.strip():
                                return candidate.strip()

    return "Unnamed road"


print("Extracting street names...")

roads["street_name"] = roads.apply(extract_street_name, axis=1)
roads["road_id"] = np.arange(len(roads), dtype=int)

# ============================================================
# 5. GROUP ROADS AND COUNT UNIQUE NEARBY PLACES
# ============================================================
print("Associating places with nearby roads...")

street_geometries = roads.dissolve(
    by="street_name",
    as_index=False
)

street_geometries["street_length_m"] = (
    street_geometries.geometry.length
)

place_sindex = place_points.sindex
place_geometries = place_points.geometry.to_list()
place_ids = place_points["place_id"].to_numpy()
place_clusters = place_points["cluster_id"].to_numpy()

count_rows = []

for _, street in street_geometries.iterrows():
    buffer_geom = street.geometry.buffer(ROAD_BUFFER_M)

    possible = place_sindex.query(buffer_geom)
    matched_positions = []

    for pos in possible:
        pos = int(pos)

        if buffer_geom.intersects(place_geometries[pos]):
            matched_positions.append(pos)

    unique_ids = {
        int(place_ids[pos]) for pos in matched_positions
    }

    cluster_ids = {
        int(place_clusters[pos])
        for pos in matched_positions
        if int(place_clusters[pos]) >= 0
    }

    count_rows.append({
        "street_name": street["street_name"],
        "unique_nearby_places": len(unique_ids),
        "nearby_place_clusters": len(cluster_ids)
    })

place_counts = pd.DataFrame(count_rows)

# IMPORTANT: Create scores before using scores["..."].
scores = street_geometries[
    ["street_name", "street_length_m", "geometry"]
].merge(
    place_counts,
    on="street_name",
    how="left"
)

scores["unique_nearby_places"] = (
    scores["unique_nearby_places"].fillna(0).astype(int)
)

scores["nearby_place_clusters"] = (
    scores["nearby_place_clusters"].fillna(0).astype(int)
)

scores["places_per_100m"] = (
    scores["unique_nearby_places"]
    / scores["street_length_m"].clip(lower=1)
    * 100
)

# ============================================================
# 6. MEASURE ACTIVITY CONTINUITY
# ============================================================
print("Measuring activity continuity along streets...")

continuity_rows = []

for _, road in roads.iterrows():
    geom = road.geometry
    length = float(geom.length)

    if length <= 0:
        continue

    number_of_bins = max(
        1, int(np.ceil(length / BIN_LENGTH_M))
    )

    active_bins = 0

    for i in range(number_of_bins):
        start = i * BIN_LENGTH_M
        end = min((i + 1) * BIN_LENGTH_M, length)

        midpoint = geom.interpolate((start + end) / 2)
        bin_buffer = midpoint.buffer(ROAD_BUFFER_M)

        possible = place_sindex.query(bin_buffer)
        found_place = False

        for pos in possible:
            if bin_buffer.intersects(place_geometries[int(pos)]):
                found_place = True
                break

        if found_place:
            active_bins += 1

    continuity_rows.append({
        "street_name": road["street_name"],
        "active_bins": active_bins,
        "total_bins": number_of_bins
    })

if continuity_rows:
    continuity_by_road = pd.DataFrame(continuity_rows)

    continuity_by_street = (
        continuity_by_road
        .groupby("street_name", as_index=False)
        .agg(
            active_bins=("active_bins", "sum"),
            total_bins=("total_bins", "sum")
        )
    )

    continuity_by_street["activity_continuity"] = (
        continuity_by_street["active_bins"]
        / continuity_by_street["total_bins"].clip(lower=1)
    )

    scores = scores.merge(
        continuity_by_street[
            ["street_name", "activity_continuity"]
        ],
        on="street_name",
        how="left"
    )
else:
    scores["activity_continuity"] = 0.0

scores["activity_continuity"] = (
    scores["activity_continuity"].fillna(0.0)
)

# ============================================================
# 7. NORMALISE AND CALCULATE CORRIDOR SCORE
# ============================================================
def minmax(series):
    low = series.min()
    high = series.max()

    if pd.isna(low) or pd.isna(high) or high == low:
        return pd.Series(0.0, index=series.index)

    return (series - low) / (high - low)


scores["place_score"] = minmax(
    scores["unique_nearby_places"]
)

scores["density_score"] = minmax(
    scores["places_per_100m"]
)

scores["continuity_score"] = minmax(
    scores["activity_continuity"]
)

scores["cluster_score"] = minmax(
    scores["nearby_place_clusters"]
)

scores["corridor_score"] = 100 * (
    0.40 * scores["place_score"]
    + 0.25 * scores["density_score"]
    + 0.25 * scores["continuity_score"]
    + 0.10 * scores["cluster_score"]
)

ranked = scores[
    (scores["street_name"] != "Unnamed road")
    & (scores["street_length_m"] >= 100)
    & (scores["unique_nearby_places"] >= 5)
].copy()

ranked = ranked.sort_values(
    "corridor_score",
    ascending=False
).reset_index(drop=True)

csv_path = DATA / "corridor_scores.csv"

ranked.drop(
    columns="geometry"
).to_csv(csv_path, index=False)

print("\n" + "=" * 72)
print("TOP CORRIDOR CANDIDATES")
print("=" * 72)

if ranked.empty:
    print("No named streets met the current filters.")
    print("Check street-name extraction or relax the filters.")
else:
    show_cols = [
        "street_name",
        "street_length_m",
        "unique_nearby_places",
        "places_per_100m",
        "nearby_place_clusters",
        "activity_continuity",
        "corridor_score"
    ]

    print(
        ranked[show_cols]
        .head(15)
        .round(2)
        .to_string(index=False)
    )

# ============================================================
# 8. DRAW FINAL MAP
# ============================================================
print("\nCreating final corridor map...")

fig, ax = plt.subplots(figsize=(13, 10))

boundary_gdf.boundary.plot(
    ax=ax, color="black", linewidth=2
)

roads.plot(
    ax=ax, color="lightgray", linewidth=0.7
)

noise = places[places["cluster_id"] == -1]
clustered = places[places["cluster_id"] >= 0]

if not noise.empty:
    gpd.GeoDataFrame(
        noise,
        geometry="cluster_point",
        crs=places.crs
    ).plot(
        ax=ax,
        color="gray",
        markersize=4,
        alpha=0.35,
        label="Unclustered places"
    )

if not clustered.empty:
    gpd.GeoDataFrame(
        clustered,
        geometry="cluster_point",
        crs=places.crs
    ).plot(
        ax=ax,
        column="cluster_id",
        categorical=True,
        cmap="tab20",
        markersize=8,
        alpha=0.75,
        legend=False
    )

top = ranked.head(3)

if not top.empty:
    top.plot(
        ax=ax,
        color="red",
        linewidth=3,
        label="Top corridor candidates"
    )

    for _, row in top.iterrows():
        label_point = row.geometry.representative_point()

        ax.annotate(
            str(row["street_name"]),
            (label_point.x, label_point.y),
            fontsize=8,
            fontweight="bold",
            xytext=(4, 4),
            textcoords="offset points"
        )

ax.set_title(
    "Chicago Corridor Discovery\n"
    "HDBSCAN Place Clusters + Road Activity Scoring",
    fontsize=14
)

ax.set_xlabel("Easting (metres)")
ax.set_ylabel("Northing (metres)")
ax.set_aspect("equal", adjustable="datalim")

handles, labels = ax.get_legend_handles_labels()

if handles:
    ax.legend(loc="best")

plt.tight_layout()

map_path = DATA / "final_corridor_map.png"

plt.savefig(
    map_path,
    dpi=220,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)

# ============================================================
# 9. SUMMARY
# ============================================================
print("\n" + "=" * 72)
print("ANALYSIS COMPLETE")
print("=" * 72)

print(f"Place records analysed: {len(places)}")
print(f"Road segments analysed: {len(roads)}")
print(f"HDBSCAN clusters found: {cluster_count}")
print(f"Places assigned to clusters: {clustered_count}")
print(f"Ranked street candidates: {len(ranked)}")
print(f"Clustered places file: {DATA / 'hdbscan_places.geojson'}")
print(f"Corridor scores CSV: {csv_path}")
print(f"Final map: {map_path}")

print(
    "Note: scores are heuristic candidate rankings. "
    "Inspect the map before treating them as validated corridors."
)
