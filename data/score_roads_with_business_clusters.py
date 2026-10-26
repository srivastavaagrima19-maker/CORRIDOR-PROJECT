
from pathlib import Path

import geopandas as gpd
import pandas as pd

DATA = Path(__file__).resolve().parent

PLACES_FILE = DATA / "business_clusters.geojson"
ROADS_FILE = DATA / "roads_in_study_area.geojson"

OUTPUT_GEOJSON = DATA / "roads_with_business_scores.geojson"
OUTPUT_CSV = DATA / "roads_with_business_scores.csv"

# Distance from a road within which businesses are considered nearby.
NEARBY_RADIUS_METRES = 50

print("Loading business places and roads...")

places = gpd.read_file(PLACES_FILE)
roads = gpd.read_file(ROADS_FILE)

# Work in a projected CRS so distances and buffers use metres.
places = places.to_crs("EPSG:32616")
roads = roads.to_crs("EPSG:32616")

# Remove records without usable geometry.
places = places[
    places.geometry.notna() & ~places.geometry.is_empty
].copy()

roads = roads[
    roads.geometry.notna() & ~roads.geometry.is_empty
].copy()

places["analysis_category"] = (
    places["analysis_category"].fillna("Unknown").astype(str)
)

# Create a 50-metre buffer around every road segment.
road_buffers = roads[["geometry"]].copy()
road_buffers["road_index"] = roads.index
road_buffers["geometry"] = road_buffers.geometry.buffer(
    NEARBY_RADIUS_METRES
)

# Match business locations to road buffers.
matches = gpd.sjoin(
    places[["analysis_category", "category_cluster", "geometry"]],
    road_buffers[["road_index", "geometry"]],
    how="inner",
    predicate="within",
)

# Count businesses and category variety near each road segment.
summary = (
    matches.groupby("road_index")
    .agg(
        nearby_businesses=("analysis_category", "size"),
        category_variety=("analysis_category", "nunique"),
        category_clusters=("category_cluster", "nunique"),
    )
)

# Find the share of nearby businesses belonging to the most common category.
category_counts = (
    matches.groupby(["road_index", "analysis_category"])
    .size()
    .rename("category_count")
    .reset_index()
)

dominant = (
    category_counts.groupby("road_index")["category_count"]
    .max()
    .rename("dominant_category_count")
)

summary = summary.join(dominant)
summary["dominant_category_share"] = (
    summary["dominant_category_count"]
    / summary["nearby_businesses"]
)

# Attach results to the road segments.
roads["road_index"] = roads.index
roads = roads.join(summary, on="road_index")

count_columns = [
    "nearby_businesses",
    "category_variety",
    "category_clusters",
]

for column in count_columns:
    roads[column] = roads[column].fillna(0).astype(int)

roads["dominant_category_share"] = (
    roads["dominant_category_share"].fillna(0)
)

# Normalize measures using percentile ranks.
# This reduces the effect of a few exceptionally busy road segments.
roads["business_density_score"] = (
    roads["nearby_businesses"].rank(pct=True)
)

roads["category_variety_score"] = (
    roads["category_variety"].rank(pct=True)
)

roads["category_cluster_score"] = (
    roads["category_clusters"].rank(pct=True)
)

roads["specialization_score"] = (
    roads["dominant_category_share"].rank(pct=True)
)

# Combined score: supports both mixed-use and specialized commercial roads.
roads["commercial_activity_score"] = (
    0.40 * roads["business_density_score"]
    + 0.25 * roads["category_variety_score"]
    + 0.20 * roads["category_cluster_score"]
    + 0.15 * roads["specialization_score"]
)

roads = roads.sort_values(
    "commercial_activity_score", ascending=False
)

# Save a CSV for inspection and GeoJSON for mapping.
roads.to_csv(OUTPUT_CSV, index=False)

roads.to_crs("EPSG:4326").to_file(
    OUTPUT_GEOJSON, driver="GeoJSON"
)

print("\nRoad scoring completed.")
print("Road segments:", len(roads))
print("Segments with nearby businesses:",
      int((roads["nearby_businesses"] > 0).sum()))

print("\nTop 20 road segments:")
print(
    roads[
        [
            "street_name",
            "class",
            "nearby_businesses",
            "category_variety",
            "category_clusters",
            "dominant_category_share",
            "commercial_activity_score",
        ]
    ]
    .head(20)
    .round(3)
    .to_string(index=False)
)

print("\nSaved files:")
print(OUTPUT_CSV)
print(OUTPUT_GEOJSON)
