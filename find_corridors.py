
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from shapely.geometry import box

# Project folders
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

# Exact study-area boundary
WEST, SOUTH, EAST, NORTH = -87.683, 41.903, -87.659, 41.913
study_area = gpd.GeoDataFrame(
    geometry=[box(WEST, SOUTH, EAST, NORTH)],
    crs="EPSG:4326"
)

# Locate the existing files
place_file = DATA / "places_filtered.geojson"
road_files = list(DATA.glob("*segment*.geojson"))

if not place_file.exists():
    raise FileNotFoundError(
        f"Cannot find {place_file}. Check that your filtered places file exists."
    )

if not road_files:
    raise FileNotFoundError(
        "Cannot find a road-segment GeoJSON file in the data folder."
    )

road_file = road_files[0]
print("Places file:", place_file.name)
print("Road file:", road_file.name)

# Load data
places = gpd.read_file(place_file)
roads = gpd.read_file(road_file)

print("\nOriginal place records:", len(places))
print("Original road records:", len(roads))

# Ensure both datasets use longitude/latitude
if places.crs is None:
    places = places.set_crs("EPSG:4326")
else:
    places = places.to_crs("EPSG:4326")

if roads.crs is None:
    roads = roads.set_crs("EPSG:4326")
else:
    roads = roads.to_crs("EPSG:4326")

# Clip long geometries to the selected study area
roads = gpd.clip(roads, study_area)
places = gpd.clip(places, study_area)

roads = roads[roads.geometry.notna() & ~roads.geometry.is_empty].copy()
places = places[places.geometry.notna() & ~places.geometry.is_empty].copy()

print("\nRecords inside study area")
print("Places:", len(places))
print("Road segments:", len(roads))

# Keep road features; remove rail and pedestrian-only infrastructure
if "subtype" in roads.columns:
    roads = roads[roads["subtype"].astype(str).str.lower() == "road"].copy()

excluded_classes = {
    "footway", "steps", "cycleway", "path", "bridleway"
}
if "class" in roads.columns:
    roads = roads[
        ~roads["class"].astype(str).str.lower().isin(excluded_classes)
    ].copy()

if roads.empty:
    raise ValueError(
        "No eligible road segments remain after filtering. "
        "Check the segment file's subtype and class fields."
    )

# Use a local projected coordinate system for distance calculations
roads = roads.to_crs("EPSG:32616")
places = places.to_crs("EPSG:32616")

# Get place points, converting other geometries to representative points
places["geometry"] = places.geometry.representative_point()

# Create a 50-metre buffer around each road segment and count nearby places
road_buffers = roads[["geometry"]].copy()
road_buffers["geometry"] = road_buffers.geometry.buffer(50)

joined = gpd.sjoin(
    places[["geometry"]],
    road_buffers,
    how="left",
    predicate="within"
)

counts = joined.groupby("index_right").size()
roads["nearby_places_50m"] = (
    roads.index.to_series().map(counts).fillna(0).astype(int)
)

# Calculate road-segment length
roads["length_m"] = roads.geometry.length

# Extract readable street names where available
def get_street_name(value):
    if isinstance(value, dict):
        return value.get("primary")
    if isinstance(value, str):
        return value
    return None

if "names" in roads.columns:
    roads["street_name"] = roads["names"].apply(get_street_name)
else:
    roads["street_name"] = None

# Aggregate named streets; keep unnamed segments separate
roads["candidate"] = roads.apply(
    lambda row: row["street_name"]
    if pd.notna(row["street_name"]) and str(row["street_name"]).strip()
    else f"Unnamed segment {row.name}",
    axis=1
)

summary = (
    roads.groupby("candidate")
    .agg(
        road_segments=("length_m", "size"),
        total_length_m=("length_m", "sum"),
        nearby_places=("nearby_places_50m", "sum")
    )
    .reset_index()
)

summary["places_per_100m"] = (
    summary["nearby_places"] / summary["total_length_m"] * 100
).fillna(0)

summary = summary.sort_values(
    ["nearby_places", "places_per_100m"],
    ascending=False
)

# Save results
output_csv = DATA / "corridor_candidates.csv"
output_roads = DATA / "roads_in_study_area.geojson"
output_map = DATA / "corridor_candidates.png"

summary.to_csv(output_csv, index=False)
roads.to_crs("EPSG:4326").to_file(output_roads, driver="GeoJSON")

# Plot road segments, coloured by nearby-place counts
roads_plot = roads.to_crs("EPSG:4326")

fig, ax = plt.subplots(figsize=(12, 9))
study_area.boundary.plot(ax=ax, color="black", linewidth=1)
roads_plot.plot(
    ax=ax,
    column="nearby_places_50m",
    legend=True,
    cmap="viridis",
    linewidth=1.5
)
ax.set_title("Road segments and nearby places — selected Chicago study area")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.tight_layout()
plt.savefig(output_map, dpi=200)
plt.close()

print("\nTop 15 preliminary corridor candidates:")
print(summary.head(15).to_string(index=False))

print("\nSaved files:")
print(output_csv)
print(output_roads)
print(output_map)
