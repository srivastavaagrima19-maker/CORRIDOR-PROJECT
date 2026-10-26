
from pathlib import Path
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

DATA = Path(__file__).parent

places_file = DATA / "places_filtered.geojson"

# Select the existing road dataset.
road_files = [
    f for f in DATA.glob("*segment*.geojson")
    if f.is_file()
]

if not places_file.exists():
    raise FileNotFoundError(
        f"Could not find {places_file.name} in the data folder."
    )

if not road_files:
    raise FileNotFoundError(
        "Could not find the original Overture road-segment GeoJSON."
    )

roads_file = road_files[0]

places = gpd.read_file(places_file)
roads = gpd.read_file(roads_file)

# Use a projected coordinate system so distances are in metres.
if places.crs is None or roads.crs is None:
    raise ValueError("One of the datasets has no CRS defined.")

places = places.to_crs(epsg=32616)
roads = roads.to_crs(epsg=32616)

# Keep roads that can form useful public street corridors.
if "subtype" in roads.columns:
    roads = roads[roads["subtype"] == "road"].copy()

if "class" in roads.columns:
    excluded = ["footway", "steps", "cycleway"]
    roads = roads[~roads["class"].isin(excluded)].copy()

# Count nearby businesses around each road segment.
# This is an initial screening score, not the final corridor result.
places["geometry"] = places.geometry.representative_point()

buffer_distance = 40  # metres on either side of each road segment
roads["nearby_place_count"] = 0

for index, road in roads.iterrows():
    nearby = places.geometry.distance(road.geometry).le(
        buffer_distance
    )
    roads.loc[index, "nearby_place_count"] = int(nearby.sum())

# Give each segment a length and a density measure.
roads["length_m"] = roads.geometry.length
roads["places_per_100m"] = (
    roads["nearby_place_count"] /
    roads["length_m"].clip(lower=1) * 100
)

# Save the highest-scoring individual street segments for review.
roads = roads.sort_values(
    ["places_per_100m", "nearby_place_count"],
    ascending=False
)

output_csv = DATA / "corridor_segment_candidates.csv"

roads.drop(columns="geometry").to_csv(output_csv, index=False)

# Make a map showing the street segments and nearby place activity.
fig, ax = plt.subplots(figsize=(12, 10))

roads.plot(
    ax=ax,
    column="places_per_100m",
    cmap="YlOrRd",
    linewidth=1.5,
    legend=True,
    missing_kwds={"color": "lightgrey"}
)

places.plot(
    ax=ax,
    color="navy",
    markersize=2,
    alpha=0.35
)

ax.set_title("Downtown Corridor Screening: Nearby Places per 100 m")
ax.set_axis_off()
plt.tight_layout()

map_file = DATA / "corridor_segment_candidates.png"
plt.savefig(map_file, dpi=200, bbox_inches="tight")
plt.close()

print("Road segments analysed:", len(roads))
print("Filtered places used:", len(places))
print("Saved candidate table:", output_csv)
print("Saved candidate map:", map_file)
print("\nTop 15 individual road segments:")
print(
    roads.drop(columns="geometry")[
        [
            c for c in [
                "names", "class", "length_m",
                "nearby_place_count", "places_per_100m"
            ]
            if c in roads.columns
        ]
    ].head(15).to_string(index=False)
)
