
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

ROAD_FILE = DATA / "roads_in_study_area.geojson"
PLACE_FILE = DATA / "places_filtered.geojson"

TARGET_STREETS = [
    "North Milwaukee Avenue",
    "West North Avenue",
    "North Damen Avenue",
]

roads = gpd.read_file(ROAD_FILE)
places = gpd.read_file(PLACE_FILE)

# Standardize coordinate systems
if roads.crs is None:
    roads = roads.set_crs("EPSG:4326")
if places.crs is None:
    places = places.set_crs("EPSG:4326")

roads = roads.to_crs("EPSG:4326")
places = places.to_crs("EPSG:4326")

def extract_name(value):
    if isinstance(value, dict):
        return value.get("primary")
    if isinstance(value, str):
        return value
    return None

# Recover street names from the road data
if "names" not in roads.columns:
    raise ValueError("The road file does not contain a 'names' column.")

roads["street_name"] = roads["names"].apply(extract_name)

selected = roads[roads["street_name"].isin(TARGET_STREETS)].copy()

if selected.empty:
    raise ValueError(
        "No target streets were found. Check the names field in the road file."
    )

# Project to metres for distance calculations
selected = selected.to_crs("EPSG:32616")
places = places.to_crs("EPSG:32616")

# Use representative points for place polygons
places["geometry"] = places.geometry.representative_point()

# Count nearby place records within 50 metres of each road segment
buffers = selected[["street_name", "geometry"]].copy()
buffers["geometry"] = buffers.geometry.buffer(50)

joined = gpd.sjoin(
    places[["geometry"]],
    buffers,
    how="inner",
    predicate="within"
)

counts = joined.groupby("index_right").size()
selected["nearby_places_50m"] = (
    selected.index.to_series().map(counts).fillna(0).astype(int)
)

# Summarize each street
summary = (
    selected.groupby("street_name")
    .agg(
        road_segments=("geometry", "size"),
        total_road_length_m=("geometry", lambda s: s.length.sum()),
        nearby_place_segment_matches=("nearby_places_50m", "sum")
    )
    .reset_index()
)

summary.to_csv(DATA / "corridor_validation_summary.csv", index=False)

# Draw the selected streets and place locations
fig, ax = plt.subplots(figsize=(12, 9))

places.plot(
    ax=ax,
    color="lightgray",
    markersize=3,
    alpha=0.45
)

selected.plot(
    ax=ax,
    column="street_name",
    categorical=True,
    legend=True,
    linewidth=3
)

ax.set_title("Candidate corridor validation — Chicago study area")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.tight_layout()

plt.savefig(DATA / "corridor_validation_map.png", dpi=200)
plt.close()

print("\nVALIDATION SUMMARY")
print(summary.to_string(index=False))

print("\nSaved:")
print(DATA / "corridor_validation_summary.csv")
print(DATA / "corridor_validation_map.png")
