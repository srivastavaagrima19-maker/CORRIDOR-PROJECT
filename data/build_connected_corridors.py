

from pathlib import Path
import ast
from collections import defaultdict

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, MultiLineString
from shapely.ops import unary_union

DATA = Path(__file__).resolve().parent

INPUT_FILE = DATA / "roads_with_business_scores.geojson"
OUTPUT_CSV = DATA / "connected_corridor_candidates.csv"
OUTPUT_GEOJSON = DATA / "connected_corridor_candidates.geojson"
OUTPUT_BOUNDARIES = DATA / "connected_corridor_boundaries.geojson"

# Initial corridor settings
MIN_SEGMENTS = 3
MIN_LENGTH_M = 200
MAX_LENGTH_M = 1500
BOUNDARY_WIDTH_M = 35
MIN_SEGMENT_SCORE = 0.70
MAX_TURN_DEGREES = 45

print("Loading scored roads...")

roads = gpd.read_file(INPUT_FILE).to_crs("EPSG:32616")
roads = roads.reset_index(drop=True)

def parse_connectors(value):
    if isinstance(value, list):
        return value
    if value is None:
        return []
    try:
        parsed = ast.literal_eval(value)
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError, TypeError):
        return []

roads["parsed_connectors"] = roads["connectors"].apply(parse_connectors)

# Map each connector to the roads containing it.
connector_to_roads = defaultdict(set)

for road_idx, connector_list in enumerate(roads["parsed_connectors"]):
    for connector in connector_list:
        if isinstance(connector, dict):
            connector_id = connector.get("connector_id")
            if connector_id:
                connector_to_roads[connector_id].add(road_idx)

# Construct road-to-road adjacency.
graph = {idx: set() for idx in roads.index}

for road_indices in connector_to_roads.values():
    indices = list(road_indices)
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            a, b = indices[i], indices[j]
            graph[a].add(b)
            graph[b].add(a)

# Keep only commercially active road segments as path seeds.
scores = roads["commercial_activity_score"].fillna(0)
active_roads = set(roads.index[scores >= MIN_SEGMENT_SCORE])

print("Total road segments:", len(roads))
print("Active road segments:", len(active_roads))

# Get the approximate angle of a line at its ends.
def endpoint_heading(line, at_start):
    coords = list(line.coords)

    if len(coords) < 2:
        return None

    if at_start:
        p1, p2 = coords[0], coords[1]
    else:
        p1, p2 = coords[-1], coords[-2]

    import math
    return math.degrees(math.atan2(p1[1] - p2[1], p1[0] - p2[0])) % 360

def angle_difference(a, b):
    if a is None or b is None:
        return 0
    diff = abs(a - b) % 360
    return min(diff, 360 - diff)

def line_parts(geom):
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, MultiLineString):
        return list(geom.geoms)
    return []

# Pick the longest line part for simple path measurements.
def main_line(geom):
    parts = line_parts(geom)
    if not parts:
        return None
    return max(parts, key=lambda line: line.length)

# Trace short paths from each active road.
# Prevent immediate cycles and cap path length.
candidates = []
seen_paths = set()

for start in sorted(active_roads, key=lambda i: scores.iloc[i], reverse=True):
    stack = [([start], set([start]))]

    while stack:
        path, used = stack.pop()

        path_key = tuple(sorted(path))
        if path_key in seen_paths:
            continue
        seen_paths.add(path_key)

        path_roads = roads.loc[path]
        path_geom = unary_union(
            [
                part
                for geom in path_roads.geometry
                for part in line_parts(geom)
            ]
        )

        path_length = path_geom.length

        if len(path) >= MIN_SEGMENTS and MIN_LENGTH_M <= path_length <= MAX_LENGTH_M:
            avg_score = float(
                path_roads["commercial_activity_score"].mean()
            )

            candidates.append({
                "road_indices": tuple(path),
                "segment_count": len(path),
                "length_m": path_length,
                "average_activity_score": avg_score,
                "max_activity_score": float(
                    path_roads["commercial_activity_score"].max()
                ),
                "nearby_businesses_sum": int(
                    path_roads["nearby_businesses"].sum()
                ),
                "category_variety_max": int(
                    path_roads["category_variety"].max()
                ),
                "geometry": path_geom,
            })

        if path_length >= MAX_LENGTH_M:
            continue

        current = path[-1]
        current_line = main_line(roads.loc[current].geometry)

        for neighbour in graph[current]:
            if neighbour in used:
                continue

            # Permit active roads and moderately active connecting roads.
            if scores.iloc[neighbour] < 0.50:
                continue

            next_line = main_line(roads.loc[neighbour].geometry)
            if current_line is None or next_line is None:
                continue

            # Reject sharp turns using the closest endpoint headings.
            h1 = endpoint_heading(current_line, at_start=False)
            h2 = endpoint_heading(next_line, at_start=True)

            if angle_difference(h1, h2) > MAX_TURN_DEGREES:
                continue

            stack.append((path + [neighbour], used | {neighbour}))

# Remove duplicate paths and keep candidates in a useful length range.
if not candidates:
    print("\nNo paths were generated.")
    print("We will inspect connector topology and score thresholds next.")

else:
    candidates_df = pd.DataFrame(candidates)

    candidates_df = candidates_df.sort_values(
        ["average_activity_score", "length_m"],
        ascending=[False, False],
    )

    # Greedy deduplication: discard paths sharing most of their road segments.
    selected = []
    selected_sets = []

    for _, row in candidates_df.iterrows():
        current_set = set(row["road_indices"])

        duplicate = False
        for previous_set in selected_sets:
            overlap = len(current_set & previous_set) / max(
                1, min(len(current_set), len(previous_set))
            )

            if overlap >= 0.60:
                duplicate = True
                break

        if not duplicate:
            selected.append(row)
            selected_sets.append(current_set)

    output_records = []

    for candidate_id, row in enumerate(selected, start=1):
        geometry = row["geometry"]

        output_records.append({
            "candidate_id": candidate_id,
            "segment_count": row["segment_count"],
            "length_m": round(row["length_m"], 1),
            "average_activity_score": round(
                row["average_activity_score"], 3
            ),
            "max_activity_score": round(row["max_activity_score"], 3),
            "nearby_businesses_sum": row["nearby_businesses_sum"],
            "category_variety_max": row["category_variety_max"],
            "geometry": geometry,
            "boundary": geometry.buffer(BOUNDARY_WIDTH_M),
        })

    candidate_gdf = gpd.GeoDataFrame(
        output_records,
        geometry="geometry",
        crs=roads.crs,
    )

    candidate_gdf.to_crs("EPSG:4326").to_file(
        OUTPUT_GEOJSON,
        driver="GeoJSON",
    )

    candidate_gdf.drop(
        columns=["geometry", "boundary"]
    ).to_csv(OUTPUT_CSV, index=False)

    boundary_gdf = gpd.GeoDataFrame(
        candidate_gdf.drop(columns=["geometry", "boundary"]),
        geometry=candidate_gdf["boundary"],
        crs=roads.crs,
    )

    boundary_gdf.to_crs("EPSG:4326").to_file(
        OUTPUT_BOUNDARIES,
        driver="GeoJSON",
    )

    print("\nPath candidates after deduplication:", len(candidate_gdf))
    print("\nTop candidates:")
    print(
        candidate_gdf[
            [
                "candidate_id",
                "segment_count",
                "length_m",
                "average_activity_score",
                "nearby_businesses_sum",
                "category_variety_max",
            ]
        ].head(20).to_string(index=False)
    )

    print("\nSaved files:")
    print(OUTPUT_CSV)
    print(OUTPUT_GEOJSON)
    print(OUTPUT_BOUNDARIES)
