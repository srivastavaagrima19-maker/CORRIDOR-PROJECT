
from pathlib import Path
import warnings
import math

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from shapely.geometry import box, LineString
from shapely.ops import substring

warnings.filterwarnings("ignore")

# ==================================================
# 1. SETTINGS
# ==================================================

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

PLACES_FILE = DATA / "places_filtered.geojson"
ROADS_FILE = DATA / "roads_in_study_area.geojson"

WEST, SOUTH, EAST, NORTH = -87.683, 41.903, -87.659, 41.913
STUDY_AREA = box(WEST, SOUTH, EAST, NORTH)

CRS_METRES = "EPSG:32616"
CRS_GEO = "EPSG:4326"

NEARBY_RADIUS = 50
MIN_SEGMENT_LENGTH = 20
MIN_CORRIDOR_LENGTH = 200
MAX_CORRIDOR_LENGTH = 1500
MIN_CORRIDOR_SEGMENTS = 3
BOUNDARY_WIDTH = 35
MAX_TURN_DEGREES = 45
MAX_JOIN_GAP_METRES = 5

# ==================================================
# 2. LOAD DATA
# ==================================================

if not PLACES_FILE.exists() or not ROADS_FILE.exists():
    raise FileNotFoundError("Required input files are missing.")

places = gpd.read_file(PLACES_FILE)
roads = gpd.read_file(ROADS_FILE)

if places.crs is None:
    places = places.set_crs(CRS_GEO)
if roads.crs is None:
    roads = roads.set_crs(CRS_GEO)

places = places.to_crs(CRS_GEO)
roads = roads.to_crs(CRS_GEO)

places = places[
    places.geometry.notna() & ~places.geometry.is_empty
].copy()

roads = roads[
    roads.geometry.notna()
    & ~roads.geometry.is_empty
    & roads.geometry.geom_type.isin(["LineString", "MultiLineString"])
].copy()

places = places[places.geometry.intersects(STUDY_AREA)].copy()
roads = roads[roads.geometry.intersects(STUDY_AREA)].copy()

roads["geometry"] = roads.geometry.intersection(STUDY_AREA)
roads = roads[
    roads.geometry.notna()
    & ~roads.geometry.is_empty
    & roads.geometry.geom_type.isin(["LineString", "MultiLineString"])
].copy()

places = places.to_crs(CRS_METRES)
roads = roads.to_crs(CRS_METRES)

boundary = gpd.GeoSeries(
    [STUDY_AREA], crs=CRS_GEO
).to_crs(CRS_METRES).iloc[0]

places = places.copy()
places["geometry"] = places.geometry.representative_point()

places = places.reset_index(drop=True)
roads = roads.reset_index(drop=True)

print("Places:", len(places))
print("Road records:", len(roads))

# ==================================================
# 3. MEASURE PLACE ACTIVITY AROUND ROADS
# ==================================================

place_index = places.sindex
counts = []

for _, road in roads.iterrows():
    line = road.geometry

    if line.length < MIN_SEGMENT_LENGTH:
        counts.append(0)
        continue

    nearby_ids = place_index.query(line.buffer(NEARBY_RADIUS))
    nearby = places.iloc[nearby_ids]

    count = int(
        nearby.geometry.distance(line).le(NEARBY_RADIUS).sum()
    )
    counts.append(count)

roads["calculated_places_50m"] = counts
roads["length_m_calculated"] = roads.geometry.length
roads["places_per_100m"] = (
    roads["calculated_places_50m"] * 100
    / roads["length_m_calculated"].clip(lower=1)
)

valid = roads["length_m_calculated"] >= MIN_SEGMENT_LENGTH

if not valid.any():
    raise SystemExit("No road segments meet the minimum length.")

counts_valid = roads.loc[valid, "calculated_places_50m"]
density_valid = roads.loc[valid, "places_per_100m"]

count_scale = max(float(counts_valid.quantile(0.95)), 1.0)
density_scale = max(float(density_valid.quantile(0.95)), 1.0)

roads["activity_score"] = 0.0
roads.loc[valid, "activity_score"] = (
    0.5 * counts_valid.clip(upper=count_scale) / count_scale
    + 0.5 * density_valid.clip(upper=density_scale) / density_scale
)

threshold = float(
    roads.loc[valid, "activity_score"].quantile(0.70)
)

roads["high_activity"] = (
    valid & (roads["activity_score"] >= threshold)
)

print("Activity threshold:", round(threshold, 3))
print("High-activity roads:", int(roads["high_activity"].sum()))

# ==================================================
# 4. SPLIT HIGH-ACTIVITY ROADS AT CONNECTORS
# ==================================================

sections = []

for road_idx, road in roads.iterrows():
    if not bool(road["high_activity"]):
        continue

    line = road.geometry

    if line.geom_type != "LineString":
        continue

    length = line.length
    connectors = road.get("connectors")

    if not isinstance(connectors, list):
        continue

    ordered = []

    for item in connectors:
        if not isinstance(item, dict):
            continue

        cid = item.get("connector_id")
        position = item.get("at")

        if cid is None or position is None:
            continue

        try:
            position = float(position)
        except (TypeError, ValueError):
            continue

        if 0 <= position <= 1:
            ordered.append((position, str(cid)))

    ordered.sort(key=lambda item: item[0])

    # Keep the same connector only once when it occurs consecutively.
    cleaned = []
    for item in ordered:
        if not cleaned or item[1] != cleaned[-1][1]:
            cleaned.append(item)

    for i in range(len(cleaned) - 1):
        start_at, u = cleaned[i]
        end_at, v = cleaned[i + 1]

        if u == v or end_at <= start_at:
            continue

        section_line = substring(
            line, start_at * length, end_at * length
        )

        if section_line.geom_type != "LineString":
            continue

        section_length = section_line.length

        if section_length < 1:
            continue

        sections.append({
            "section_id": len(sections),
            "road_idx": road_idx,
            "u": u,
            "v": v,
            "length_m": section_length,
            "activity_score": float(road["activity_score"]),
            "places_50m": int(road["calculated_places_50m"]),
            "density": float(road["places_per_100m"]),
            "geometry": section_line,
        })

if not sections:
    raise SystemExit("No connector sections were created.")

section_by_id = {
    section["section_id"]: section for section in sections
}

print("High-activity connector sections:", len(sections))

# ==================================================
# 5. BUILD THE CONNECTOR GRAPH
# ==================================================

graph = nx.MultiGraph()

for section in sections:
    graph.add_edge(
        section["u"],
        section["v"],
        key=section["section_id"],
        section_id=section["section_id"],
    )

def endpoint_angle(line, at_start):
    """Direction pointing away from the selected endpoint."""
    coords = list(line.coords)

    if len(coords) < 2:
        return None

    if at_start:
        a, b = coords[0], coords[1]
    else:
        a, b = coords[-1], coords[-2]

    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))

def straightness_turn(outgoing_a, outgoing_b):
    """
    At a junction, straight continuation has opposite outward
    directions, so its turn deviation is close to zero.
    """
    difference = abs(
        (outgoing_a - outgoing_b + 180) % 360 - 180
    )
    return abs(180 - difference)

def incident_sections(node):
    result = []

    for neighbour, keyed in graph.adj[node].items():
        for _, data in keyed.items():
            sid = data["section_id"]
            section = section_by_id[sid]

            if section["u"] == node:
                angle = endpoint_angle(section["geometry"], True)
            else:
                angle = endpoint_angle(section["geometry"], False)

            result.append((sid, angle))

    return result

# ==================================================
# 6. TRACE INDEPENDENT CONTINUOUS PATHS
# ==================================================

def trace_path(start_node, first_sid):
    """
    Trace one path. A local set prevents repeated sections inside
    this path, but sections remain available to other paths.
    """
    path_ids = []
    local_used = set()
    current_node = start_node
    sid = first_sid
    total_length = 0.0

    while sid not in local_used:
        section = section_by_id[sid]

        if total_length + section["length_m"] > MAX_CORRIDOR_LENGTH:
            break

        local_used.add(sid)
        path_ids.append(sid)
        total_length += section["length_m"]

        if section["u"] == current_node:
            next_node = section["v"]
            incoming_outward = endpoint_angle(
                section["geometry"], False
            )
        else:
            next_node = section["u"]
            incoming_outward = endpoint_angle(
                section["geometry"], True
            )

        if incoming_outward is None:
            break

        options = []

        for next_sid, angle in incident_sections(next_node):
            if next_sid in local_used or angle is None:
                continue

            turn = straightness_turn(incoming_outward, angle)

            if turn <= MAX_TURN_DEGREES:
                options.append((turn, next_sid))

        if not options:
            break

        options.sort(key=lambda item: item[0])
        sid = options[0][1]
        current_node = next_node

    return path_ids

# Try all section starts independently, including sections at
# intersections. Duplicate paths are removed afterward.
raw_paths = []
seen_paths = set()

for section in sections:
    for start_node in (section["u"], section["v"]):
        path = trace_path(start_node, section["section_id"])

        if not path:
            continue

        # A sorted signature identifies the same group of sections
        # even when the path was traced in the reverse direction.
        signature = tuple(sorted(path))

        if signature not in seen_paths:
            seen_paths.add(signature)
            raw_paths.append(path)

print("\nPATH DIAGNOSTICS")
print("----------------")
print("Unique traced paths:", len(raw_paths))

raw_lengths = [
    sum(section_by_id[sid]["length_m"] for sid in path)
    for path in raw_paths
]

print(
    "Paths in requested length range:",
    sum(
        MIN_CORRIDOR_LENGTH <= length <= MAX_CORRIDOR_LENGTH
        for length in raw_lengths
    )
)

print(
    "Shortest traced path (m):",
    round(min(raw_lengths), 2) if raw_lengths else "None"
)
print(
    "Longest traced path (m):",
    round(max(raw_lengths), 2) if raw_lengths else "None"
)

# ==================================================
# 7. BUILD AND CHECK PATH GEOMETRIES
# ==================================================

candidate_records = []
candidate_lines = []
candidate_boundaries = []

rejected_short = 0
rejected_length = 0
rejected_geometry = 0

for path in raw_paths:
    group = [section_by_id[sid] for sid in path]

    if len(group) < MIN_CORRIDOR_SEGMENTS:
        rejected_short += 1
        continue

    estimated_length = sum(
        section["length_m"] for section in group
    )

    if not (
        MIN_CORRIDOR_LENGTH
        <= estimated_length
        <= MAX_CORRIDOR_LENGTH
    ):
        rejected_length += 1
        continue

    # Reconstruct a line in traversal order. Permit a small geometric
    # gap caused by rounding, but reject genuinely disconnected paths.
    coordinates = []
    failed = False

    for section in group:
        coords = list(section["geometry"].coords)

        if not coordinates:
            coordinates.extend(coords)
            continue

        last_x, last_y = coordinates[-1]
        first_gap = math.hypot(
            last_x - coords[0][0],
            last_y - coords[0][1],
        )
        last_gap = math.hypot(
            last_x - coords[-1][0],
            last_y - coords[-1][1],
        )

        if first_gap <= MAX_JOIN_GAP_METRES:
            # Snap a small gap to the previous endpoint.
            coordinates.extend(coords[1:])
        elif last_gap <= MAX_JOIN_GAP_METRES:
            coords.reverse()
            coordinates.extend(coords[1:])
        else:
            failed = True
            break

    if failed or len(coordinates) < 2:
        rejected_geometry += 1
        continue

    centerline = LineString(coordinates)

    if centerline.length < MIN_CORRIDOR_LENGTH:
        rejected_geometry += 1
        continue

    mean_score = sum(
        item["activity_score"] for item in group
    ) / len(group)

    mean_places = sum(
        item["places_50m"] for item in group
    ) / len(group)

    mean_density = sum(
        item["density"] for item in group
    ) / len(group)

    score = (
        0.55 * mean_score
        + 0.25 * min(centerline.length / 1000, 1.0)
        + 0.20 * min(len(group) / 10, 1.0)
    )

    candidate_id = len(candidate_records) + 1

    candidate_records.append({
        "candidate_id": candidate_id,
        "segment_count": len(group),
        "length_m": centerline.length,
        "mean_places_50m": mean_places,
        "mean_places_per_100m": mean_density,
        "mean_activity_score": mean_score,
        "candidate_score": score,
    })

    candidate_lines.append(centerline)
    candidate_boundaries.append(
        centerline.buffer(BOUNDARY_WIDTH)
    )

print("\nFILTER DIAGNOSTICS")
print("------------------")
print("Rejected: too few sections:", rejected_short)
print("Rejected: outside length range:", rejected_length)
print("Rejected: disconnected geometry:", rejected_geometry)
print("Accepted candidates:", len(candidate_records))

if not candidate_records:
    raise SystemExit(
        "No candidates survived. Use the PATH DIAGNOSTICS and "
        "FILTER DIAGNOSTICS above to identify the limiting stage."
    )

# ==================================================
# 8. RANK AND SAVE CANDIDATES
# ==================================================

candidates = gpd.GeoDataFrame(
    candidate_records,
    geometry=candidate_lines,
    crs=CRS_METRES,
).sort_values(
    "candidate_score", ascending=False
).reset_index(drop=True)

candidates["rank"] = candidates.index + 1

boundaries = gpd.GeoDataFrame(
    {
        "candidate_id": candidates["candidate_id"],
        "rank": candidates["rank"],
        "boundary_width_m": BOUNDARY_WIDTH,
    },
    geometry=[
        line.buffer(BOUNDARY_WIDTH)
        for line in candidates.geometry
    ],
    crs=CRS_METRES,
)

candidates.drop(columns="geometry").to_csv(
    DATA / "network_corridor_candidates.csv",
    index=False,
)

candidates.to_crs(CRS_GEO).to_file(
    DATA / "network_corridor_candidates.geojson",
    driver="GeoJSON",
)

boundaries.to_crs(CRS_GEO).to_file(
    DATA / "network_corridor_boundaries.geojson",
    driver="GeoJSON",
)

# ==================================================
# 9. CREATE REVIEW MAP
# ==================================================

fig, ax = plt.subplots(figsize=(13, 10))

roads.plot(ax=ax, color="lightgray", linewidth=0.5)
places.plot(ax=ax, color="gray", markersize=2, alpha=0.35)

roads[roads["high_activity"]].plot(
    ax=ax, color="orange", linewidth=1.2
)

candidates.head(5).plot(
    ax=ax, column="rank", cmap="viridis",
    linewidth=3, legend=True
)

gpd.GeoSeries([boundary], crs=CRS_METRES).boundary.plot(
    ax=ax, color="black", linewidth=1.5
)

ax.set_title(
    "Preliminary Data-Driven Corridor Candidates\n"
    "Orange = high activity; colored lines = candidate paths"
)
ax.set_axis_off()
plt.tight_layout()

map_path = DATA / "network_corridor_candidates.png"
plt.savefig(map_path, dpi=220, bbox_inches="tight")
plt.close()

# ==================================================
# 10. REPORT RESULTS
# ==================================================

print("\nTOP CORRIDOR CANDIDATES")
print("-----------------------")
print(
    candidates.drop(columns="geometry")
    .head(10)
    .to_string(index=False)
)

print("\nFiles created:")
print(DATA / "network_corridor_candidates.csv")
print(DATA / "network_corridor_candidates.geojson")
print(DATA / "network_corridor_boundaries.geojson")
print(map_path)

print(
    "\nThese are preliminary candidates, not yet validated "
    "commercial corridors. Review their geometry and compare "
    "the leading candidate against an actual street map."
)

