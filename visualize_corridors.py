
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Project folders
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# Existing files
roads_file = DATA_DIR / "roads_in_study_area.geojson"
places_file = DATA_DIR / "places_filtered.geojson"

# Candidate corridors found in our earlier analysis
candidate_streets = {
    "North Milwaukee Avenue": "red",
    "West North Avenue": "blue",
    "North Damen Avenue": "green"
}

# Load data
roads = gpd.read_file(roads_file)
places = gpd.read_file(places_file)

# Use a projected CRS so distances are measured in metres
if roads.crs is None:
    roads = roads.set_crs("EPSG:4326")

if places.crs is None:
    places = places.set_crs("EPSG:4326")

roads = roads.to_crs("EPSG:32616")
places = places.to_crs("EPSG:32616")

# Extract a street name from Overture's names field
def get_street_name(value):
    if isinstance(value, dict):
        # Try common Overture name fields
        for key in ("primary", "common", "rules"):
            item = value.get(key)
            if isinstance(item, str):
                return item
            if isinstance(item, dict):
                for subvalue in item.values():
                    if isinstance(subvalue, str):
                        return subvalue
        return ""
    if isinstance(value, str):
        return value
    return ""

# Roads may store names in a "names" field
if "names" in roads.columns:
    roads["street_name"] = roads["names"].apply(get_street_name)
else:
    roads["street_name"] = ""

# Keep only the selected candidate streets
roads["street_name"] = roads["street_name"].fillna("")
selected_roads = roads[
    roads["street_name"].isin(candidate_streets.keys())
].copy()

if selected_roads.empty:
    print("No candidate street names matched.")
    print("Available road columns:", list(roads.columns))
    print("Example names:", roads["street_name"].dropna().head(10).tolist())
    raise SystemExit(
        "Check the names field format before plotting."
    )

# Plot
fig, ax = plt.subplots(figsize=(13, 10))

# Background road network
roads.plot(
    ax=ax,
    color="lightgrey",
    linewidth=0.45,
    alpha=0.8,
    label="Other roads"
)

# Background places
places.plot(
    ax=ax,
    color="black",
    markersize=3,
    alpha=0.35,
    label="Places"
)

# Highlight each candidate corridor
for street, color in candidate_streets.items():
    part = selected_roads[selected_roads["street_name"] == street]
    if not part.empty:
        part.plot(
            ax=ax,
            color=color,
            linewidth=3,
            label=street
        )

ax.set_title(
    "Preliminary Corridor Candidates — Downtown Chicago",
    fontsize=16,
    fontweight="bold"
)
ax.set_xlabel("Easting (metres)")
ax.set_ylabel("Northing (metres)")
ax.legend(loc="best")
ax.set_aspect("equal")
ax.grid(alpha=0.2)

plt.tight_layout()

# Save the image
output_file = DATA_DIR / "corridors_visualization.png"
plt.savefig(output_file, dpi=300, bbox_inches="tight")
plt.show()

print(f"Visualization saved to: {output_file}")
print("Candidate road segments plotted:", len(selected_roads))
