
from pathlib import Path
import warnings

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

from shapely.geometry import box, Point
from shapely.ops import linemerge, substring

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. FILES AND STUDY AREA
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

PLACES_FILE = DATA / "places_filtered.geojson"
ROADS_FILE = DATA / "roads_in_study_area.geojson"

# Same Chicago study area as your previous script
WEST, SOUTH, EAST, NORTH = -87.683, 41.903, -87.659, 41.913
STUDY_AREA = box(WEST, SOUTH, EAST, NORTH)

# Analysis settings, in metres
CRS_METRES = "EPSG:32616"
SECTION_LENGTH = 100
NEARBY_RADIUS = 50
MIN_PLACES = 8
MIN_CORRIDOR_LENGTH = 200
CORRIDOR_HALF_WIDTH = 50

TARGET_STREET = "North Milwaukee Avenue"


# --------------------------------------------------
# 2. HELPER FUNCTIONS
# --------------------------------------------------

def get_street_name(row):
    for column in ["street_name", "name", "road_name", "street"]:
        if column in row.index:
            value = row[column]

            if isinstance(value, str) and value.strip():
                return value.strip()

    return None


def get_line_parts(geometry):
    if geometry is None or geometry.is_empty:
        return []

    if geometry.geom_type == "LineString":
        return [geometry]

    if geometry.geom_type == "MultiLineString":
        return list(geometry.geoms)

    if geometry.geom_type == "GeometryCollection":
        parts = []
        for item in geometry.geoms:
            parts.extend(get_line_parts(item))
        return parts

    return []


# --------------------------------------------------
# 3. LOAD AND FILTER DATA
# --------------------------------------------------

if not PLACES_FILE.exists():
    raise FileNotFoundError(f"File not found: {PLACES_FILE}")

if not ROADS_FILE.exists():
    raise FileNotFoundError(f"File not found: {ROADS_FILE}")

places = gpd.read_file(PLACES_FILE)
roads = gpd.read_file(ROADS_FILE)

if places.crs is None:
    places = places.set_crs("EPSG:4326")

if roads.crs is None:
    roads = roads.set_crs("EPSG:4326")

places = places.to_crs("EPSG:4326")
roads = roads.to_crs("EPSG:4326")

places = places[
    places.geometry.intersects(STUDY_AREA)
].copy()

roads = roads[
    roads.geometry.intersects(STUDY_AREA)
].copy()

# Clip geometries to the specified study area
roads["geometry"] = roads.geometry.intersection(STUDY_AREA)

roads = roads[
    ~roads.geometry.is_empty
].copy()

boundary_gdf = gpd.GeoDataFrame(
    {"name": ["Study area"]},
    geometry=[STUDY_AREA],
    crs="EPSG:4326"
)

# Convert to metres for distance and length calculations
places = places.to_crs(CRS_METRES)
roads = roads.to_crs(CRS_METRES)
boundary = boundary_gdf.to_crs(CRS_METRES).geometry.iloc[0]

# Use representative points for places represented by polygons
places = places[
    places.geometry.notna() & ~places.geometry.is_empty
].copy()

places["geometry"] = places.geometry.representative_point()
places = places.reset_index(drop=True)

print("Places inside study area:", len(places))
print("Road records inside study area:", len(roads))


# --------------------------------------------------
# 4. FIND NORTH MILWAUKEE AVENUE
# --------------------------------------------------

roads["street_name"] = roads.apply(
    get_street_name, axis=1
)

named_roads = roads[
    roads["street_name"].notna()
].copy()

def normalize_name(name):
    return " ".join(str(name).lower().split())


target_normalized = normalize_name(TARGET_STREET)

matching_roads = named_roads[
    named_roads["street_name"].map(normalize_name)
    == target_normalized
].copy()

# Allow a name variation if exact matching fails
if matching_roads.empty:
    matching_roads = named_roads[
        named_roads["street_name"].map(
            lambda name: (
                "milwaukee" in normalize_name(name)
                and (
                    "avenue" in normalize_name(name)
                    or "ave" in normalize_name(name)
                )
            )
        )
    ].copy()

if matching_roads.empty:
    available_names = sorted(
        named_roads["street_name"].unique()
    )

    print("North Milwaukee Avenue was not matched.")
    print("Some street names available in your data:")
    print(available_names[:40])

    raise ValueError(
        "Check the street-name column and TARGET_STREET."
    )

street_name = matching_roads["street_name"].value_counts().index[0]

matching_roads = matching_roads[
    matching_roads["street_name"] == street_name
].copy()

print("Street being analyzed:", street_name)


# --------------------------------------------------
# 5. COMBINE ROAD GEOMETRIES
# --------------------------------------------------

combined_geometry = linemerge(
    matching_roads.geometry.union_all()
)

road_parts = get_line_parts(combined_geometry)

if not road_parts:
    raise ValueError("No usable road line was found.")


# --------------------------------------------------
# 6. DIVIDE THE ROAD INTO 100-METRE SECTIONS
# --------------------------------------------------

section_records = []
section_geometries = []

place_index = places.sindex

for part_number, road_line in enumerate(road_parts):

    distance = 0.0

    while distance < road_line.length:

        end_distance = min(
            distance + SECTION_LENGTH,
            road_line.length
        )

        section = substring(
            road_line,
            distance,
            end_distance
        )

        if section.is_empty or section.length == 0:
            distance = end_distance
            continue

        search_area = section.buffer(NEARBY_RADIUS)

        candidate_indices = place_index.query(search_area)

        candidate_places = places.iloc[
            candidate_indices
        ]

        # Count places within 50 metres of this road section
        nearby_count = int(
            candidate_places.geometry.distance(section)
            .le(NEARBY_RADIUS)
            .sum()
        )

        section_records.append({
            "part_number": part_number,
            "start_m": distance,
            "end_m": end_distance,
            "length_m": section.length,
            "nearby_places": nearby_count,
            "active": nearby_count >= MIN_PLACES
        })

        section_geometries.append(section)

        distance = end_distance


sections = gpd.GeoDataFrame(
    section_records,
    geometry=section_geometries,
    crs=CRS_METRES
)

if sections.empty:
    raise ValueError("No road sections were generated.")

print("Road sections analyzed:", len(sections))
print("Active sections:", int(sections["active"].sum()))


# --------------------------------------------------
# 7. FIND CONSECUTIVE ACTIVE ROAD SECTIONS
# --------------------------------------------------

active_runs = []

for part_number, group in sections.groupby("part_number"):

    group = group.sort_values("start_m")

    current_run = []

    for index, row in group.iterrows():

        if row["active"]:
            current_run.append(index)

        else:
            if current_run:
                active_runs.append(current_run)
                current_run = []

    if current_run:
        active_runs.append(current_run)


# Keep runs that meet the minimum length
qualifying_runs = []

for run_indices in active_runs:

    run = sections.loc[run_indices]

    run_length = run["length_m"].sum()

    if run_length >= MIN_CORRIDOR_LENGTH:
        qualifying_runs.append(run)


# Save the measured sections for inspection
sections.drop(columns="geometry").to_csv(
    DATA / "one_corridor_sections.csv",
    index=False
)


# --------------------------------------------------
# 8. SELECT ONE CORRIDOR AND GENERATE ITS BOUNDARY
# --------------------------------------------------

if not qualifying_runs:

    print("\nNo qualifying corridor section was found.")
    print("Section counts saved to:")
    print(DATA / "one_corridor_sections.csv")

    print("\nSections with the most nearby places:")

    print(
        sections.sort_values(
            "nearby_places",
            ascending=False
        )[
            ["start_m", "end_m", "nearby_places", "active"]
        ].head(10).to_string(index=False)
    )

    raise SystemExit(
        "No boundary generated. Inspect the section counts "
        "and adjust the activity threshold if justified."
    )


# Choose the longest continuous qualifying active run
chosen_run = max(
    qualifying_runs,
    key=lambda run: run["length_m"].sum()
)

part_number = int(chosen_run["part_number"].iloc[0])
source_line = road_parts[part_number]

start_distance = float(chosen_run["start_m"].min())
end_distance = float(chosen_run["end_m"].max())

corridor_line = substring(
    source_line,
    start_distance,
    end_distance
)

if corridor_line.is_empty:
    raise ValueError("Could not create the corridor centerline.")

# The polygon is an envelope around the detected road section.
# Its width is a parameter, not a naturally discovered boundary.
corridor_polygon = corridor_line.buffer(
    CORRIDOR_HALF_WIDTH
).intersection(boundary)

if corridor_polygon.is_empty:
    raise ValueError("Corridor polygon is empty after clipping.")

start_point = Point(corridor_line.coords[0])
end_point = Point(corridor_line.coords[-1])

start_wgs84 = gpd.GeoSeries(
    [start_point], crs=CRS_METRES
).to_crs("EPSG:4326").iloc[0]

end_wgs84 = gpd.GeoSeries(
    [end_point], crs=CRS_METRES
).to_crs("EPSG:4326").iloc[0]

boundary_output = gpd.GeoDataFrame(
    [{
        "street_name": street_name,
        "length_m": corridor_line.length,
        "active_sections": len(chosen_run),
        "mean_places_per_section": chosen_run[
            "nearby_places"
        ].mean(),
        "start_lon": start_wgs84.x,
        "start_lat": start_wgs84.y,
        "end_lon": end_wgs84.x,
        "end_lat": end_wgs84.y,
        "half_width_m": CORRIDOR_HALF_WIDTH
    }],
    geometry=[corridor_polygon],
    crs=CRS_METRES
)

centerline_output = gpd.GeoDataFrame(
    [{
        "street_name": street_name,
        "length_m": corridor_line.length
    }],
    geometry=[corridor_line],
    crs=CRS_METRES
)

boundary_output.to_crs("EPSG:4326").to_file(
    DATA / "one_corridor_boundary.geojson",
    driver="GeoJSON"
)

centerline_output.to_crs("EPSG:4326").to_file(
    DATA / "one_corridor_centerline.geojson",
    driver="GeoJSON"
)


# --------------------------------------------------
# 9. DRAW THE FINAL MAP
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 10))

# Study-area outline
gpd.GeoSeries(
    [boundary], crs=CRS_METRES
).boundary.plot(
    ax=ax,
    color="black",
    linewidth=2,
    label="Study-area boundary"
)

# Roads and places for context
roads.plot(
    ax=ax,
    color="lightgray",
    linewidth=0.6
)

places.plot(
    ax=ax,
    color="gray",
    markersize=3,
    alpha=0.5
)

# Active sections
sections[sections["active"]].plot(
    ax=ax,
    color="darkorange",
    linewidth=3,
    label="Active road sections"
)

# Corridor polygon and centerline
boundary_output.plot(
    ax=ax,
    color="deepskyblue",
    edgecolor="blue",
    alpha=0.35,
    linewidth=2,
    label="Estimated corridor boundary"
)

centerline_output.plot(
    ax=ax,
    color="blue",
    linewidth=2,
    label="Corridor centerline"
)

gpd.GeoSeries(
    [start_point], crs=CRS_METRES
).plot(
    ax=ax,
    color="green",
    markersize=70,
    label="Start"
)

gpd.GeoSeries(
    [end_point], crs=CRS_METRES
).plot(
    ax=ax,
    color="red",
    markersize=70,
    label="End"
)

ax.set_title(
    f"Detected Corridor: {street_name}\n"
    f"Length: {corridor_line.length:.0f} metres",
    fontsize=14
)

ax.set_axis_off()
ax.legend(loc="best")

plt.tight_layout()

map_file = DATA / "one_corridor_boundary_map.png"

plt.savefig(
    map_file,
    dpi=220,
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------
# 10. PRINT RESULTS
# --------------------------------------------------

print("\nCORRIDOR CANDIDATE GENERATED")
print("----------------------------")
print("Street:", street_name)
print(f"Active length: {corridor_line.length:.1f} metres")
print("Active sections:", len(chosen_run))

print(
    f"Start longitude, latitude: "
    f"{start_wgs84.x:.6f}, {start_wgs84.y:.6f}"
)

print(
    f"End longitude, latitude: "
    f"{end_wgs84.x:.6f}, {end_wgs84.y:.6f}"
)

print(
    f"Polygon half-width parameter: "
    f"{CORRIDOR_HALF_WIDTH} metres"
)

print("\nFiles created:")
print(DATA / "one_corridor_boundary.geojson")
print(DATA / "one_corridor_centerline.geojson")
print(DATA / "one_corridor_sections.csv")
print(map_file)

print(
    "\nImportant: this is a data-derived candidate boundary. "
    "Review the map to verify that the selected section is "
    "a meaningful corridor."
)
