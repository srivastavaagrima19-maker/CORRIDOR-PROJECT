
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# ============================================================
# 1. FIND THE DATA FILES
# ============================================================

data_folder = Path(__file__).parent


def find_file(pattern):
    matches = list(data_folder.glob(pattern))

    if not matches:
        raise FileNotFoundError(
            f"Could not find a file matching: {pattern}"
        )

    return matches[0]


building_file = find_file("*building*.geojson")
place_file = find_file("*place*.geojson")
segment_file = find_file("*segment*.geojson")


# ============================================================
# 2. LOAD THE DATA
# ============================================================

print("Loading Chicago geographic data...")

buildings = gpd.read_file(building_file)
places = gpd.read_file(place_file)
roads = gpd.read_file(segment_file)

print("\nDATA LOADED SUCCESSFULLY")
print("----------------------------------")
print("Buildings:", len(buildings))
print("Places:", len(places))
print("Road segments:", len(roads))


# ============================================================
# 3. REMOVE EMPTY GEOMETRIES
# ============================================================

buildings = buildings[
    buildings.geometry.notna()
    & ~buildings.geometry.is_empty
].copy()

places = places[
    places.geometry.notna()
    & ~places.geometry.is_empty
].copy()

roads = roads[
    roads.geometry.notna()
    & ~roads.geometry.is_empty
].copy()


# ============================================================
# 4. CONVERT TO A PROJECTED COORDINATE SYSTEM
# ============================================================

# UTM Zone 16N is suitable for this Chicago study area.
buildings = buildings.to_crs(epsg=32616)
places = places.to_crs(epsg=32616)
roads = roads.to_crs(epsg=32616)


# ============================================================
# 5. DEFINE THE STUDY AREA
# ============================================================

# Use buildings and places to define the main study area.
building_bounds = buildings.total_bounds
place_bounds = places.total_bounds

min_x = min(building_bounds[0], place_bounds[0])
min_y = min(building_bounds[1], place_bounds[1])
max_x = max(building_bounds[2], place_bounds[2])
max_y = max(building_bounds[3], place_bounds[3])

# Add 10% margin on each side.
margin_x = (max_x - min_x) * 0.10
margin_y = (max_y - min_y) * 0.10

min_x -= margin_x
max_x += margin_x
min_y -= margin_y
max_y += margin_y

study_area = gpd.GeoSeries(
    [buildings.geometry.unary_union.convex_hull],
    crs=buildings.crs
).iloc[0].envelope

# Use the rectangular map bounds for consistent map limits.
from shapely.geometry import box

study_area = box(min_x, min_y, max_x, max_y)

print("\nFOCUSED STUDY AREA")
print("----------------------------------")
print("Width:", round(max_x - min_x, 1), "metres")
print("Height:", round(max_y - min_y, 1), "metres")


# ============================================================
# 6. SELECT ROADS WITHIN THE STUDY AREA
# ============================================================

roads_focus = roads[
    roads.geometry.intersects(study_area)
].copy()

# Clip lines at the map boundary.
roads_focus["geometry"] = roads_focus.geometry.intersection(
    study_area
)

roads_focus = roads_focus[
    roads_focus.geometry.notna()
    & ~roads_focus.geometry.is_empty
].copy()

print("Road segments in focused area:", len(roads_focus))


# ============================================================
# 7. SHARED MAP SETTINGS
# ============================================================

def prepare_map(ax, title):
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=15, pad=12)
    ax.set_axis_off()


# ============================================================
# 8. MAP 1: ROAD NETWORK ONLY
# ============================================================

print("\nCreating road network map...")

fig, ax = plt.subplots(figsize=(12, 10))

roads_focus.plot(
    ax=ax,
    color="black",
    linewidth=0.65
)

prepare_map(ax, "Chicago Study Area — Road Network")

plt.tight_layout()

road_map_path = data_folder / "map1_roads.png"

plt.savefig(
    road_map_path,
    dpi=250,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()


# ============================================================
# 9. MAP 2: BUILDING FOOTPRINTS ONLY
# ============================================================

print("Creating building footprint map...")

fig, ax = plt.subplots(figsize=(12, 10))

buildings.plot(
    ax=ax,
    color="steelblue",
    edgecolor="white",
    linewidth=0.15
)

prepare_map(ax, "Chicago Study Area — Building Footprints")

plt.tight_layout()

building_map_path = data_folder / "map2_buildings.png"

plt.savefig(
    building_map_path,
    dpi=250,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()


# ============================================================
# 10. MAP 3: PLACE DENSITY USING A 100-METRE GRID
# ============================================================

print("Creating place density map...")

cell_size = 100

# Remove places without usable point coordinates.
place_points = places[
    places.geometry.geom_type == "Point"
].copy()

place_points = place_points[
    place_points.geometry.x.between(min_x, max_x)
    & place_points.geometry.y.between(min_y, max_y)
].copy()

# Assign every place to a 100 m grid cell.
place_points["grid_x"] = np.floor(
    (place_points.geometry.x - min_x) / cell_size
).astype(int)

place_points["grid_y"] = np.floor(
    (place_points.geometry.y - min_y) / cell_size
).astype(int)

# Count places in each grid cell.
density = (
    place_points.groupby(["grid_x", "grid_y"])
    .size()
    .reset_index(name="place_count")
)

print("Places inside study area:", len(place_points))
print("Grid cells containing places:", len(density))

fig, ax = plt.subplots(figsize=(12, 10))

# Draw roads as a light background.
roads_focus.plot(
    ax=ax,
    color="lightgrey",
    linewidth=0.4,
    zorder=1
)

# Draw cells containing places.
max_count = max(1, int(density["place_count"].max()))

for _, row in density.iterrows():

    x = min_x + row["grid_x"] * cell_size
    y = min_y + row["grid_y"] * cell_size

    # Darker red means more places in the cell.
    intensity = row["place_count"] / max_count

    rectangle = Rectangle(
        (x, y),
        cell_size,
        cell_size,
        facecolor=plt.cm.YlOrRd(intensity),
        edgecolor="none",
        alpha=0.85,
        zorder=2
    )

    ax.add_patch(rectangle)

prepare_map(
    ax,
    "Chicago Study Area — Place Density (100 m Grid)"
)

plt.tight_layout()

density_map_path = data_folder / "map3_place_density.png"

plt.savefig(
    density_map_path,
    dpi=250,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()


# ============================================================
# 11. SAVE PLACE DENSITY DATA
# ============================================================

density_csv_path = data_folder / "place_density_100m.csv"

density.to_csv(density_csv_path, index=False)


# ============================================================
# 12. FINISH
# ============================================================

print("\nALL MAPS CREATED SUCCESSFULLY")
print("----------------------------------")
print("Road map:", road_map_path)
print("Building map:", building_map_path)
print("Place density map:", density_map_path)
print("Place density data:", density_csv_path)

print("\nCompleted successfully!")






