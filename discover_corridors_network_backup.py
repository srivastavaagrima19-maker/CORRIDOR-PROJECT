
from pathlib import Path
import warnings

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx

from shapely.geometry import box, Point, LineString
from shapely.ops import unary_union

warnings.filterwarnings("ignore")

# ==================================================
# 1. FILES AND SETTINGS
# ==================================================

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

PLACES_FILE = DATA / "places_filtered.geojson"
ROADS_FILE = DATA / "roads_in_study_area.geojson"

WEST, SOUTH, EAST, NORTH = -87.683, 41.903, -87.659, 41.913
STUDY_AREA = box(WEST, SOUTH, EAST, NORTH)

CRS_METRES = "EPSG:32616"
NEARBY_RADIUS = 50
MIN_SEGMENT_LENGTH = 20
MIN_CORRIDOR_LENGTH = 200
MIN_CORRIDOR_SEGMENTS = 3

# ==================================================
# 2. LOAD DATA
# ==================================================

if not PLACES_FILE.exists() or not ROADS_FILE.exists():
    raise FileNotFoundError(
        "Required input files were not found in the data folder."
    )

places = gpd.read_file(PLACES_FILE)
roads = gpd.read_file(ROADS_FILE)

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
    roads.geometry.notna()
    & ~roads.geometry.is_empty
    & roads.geometry.geom_type.isin(["LineString", "MultiLineString"])
].copy()

places = places[places.geometry.intersects(STUDY_AREA)].copy()
roads = roads[roads.geometry.intersects(STUDY_AREA)].copy()

roads["geometry"] = roads.geometry.intersection(STUDY_AREA)
roads = roads[
    roads.geometry.notna() & ~roads.geometry.is_empty
].copy()

places = places.to_crs(CRS_METRES)
roads = roads.to_crs(CRS_METRES)

boundary = gpd.GeoSeries(
    [STUDY_AREA], crs="EPSG:4326"
).to_crs(CRS_METRES).iloc[0]

places["geometry"] = places.geometry.representative_point()
places = places.reset_index(drop=True)
roads = roads.reset_index(drop=True)

print("Places:", len(places))
print("Road segments:", len(roads))

# ==================================================
# 3. CALCULATE PLACE ACTIVITY FOR EVERY ROAD
# ==================================================

place_index = places.sindex
activity = []

for idx, row in roads.iterrows():
    line = row.geometry
    length = line.length

    if length < MIN_SEGMENT_LENGTH:
        activity.append(0)
        continue

    search_area = line.buffer(NEARBY_RADIUS)
    candidate_ids = place_index.query(search_area)
    nearby = places.iloc[candidate_ids]

    count = int(
        nearby.geometry.distance(line).le(NEARBY_RADIUS).sum()
    )
    activity.append(count)

roads["calculated_places_50m"] = activity
roads["length_m_calculated"] = roads.geometry.length
roads["places_per_100m"] = (
    roads["calculated_places_50m"] * 100
    / roads["length_m_calculated"].clip(lower=1)
)

# ==================================================
# 4. BUILD A CONNECTED ROAD GRAPH
# ==================================================

# A connector represents a point along a road segment.
# Road sections sharing a connector can be joined in the graph.
graph = nx.Graph()
connector_to_roads = {}

for idx, row in roads.iterrows():
    connectors = row.get("connectors")
    valid_connectors = []

    if isinstance(connectors, list):
        for item in connectors:
            if isinstance(item, dict):
                connector_id = item.get("connector_id")
                if connector_id:
                    valid_connectors.append(str(connector_id))

    # If connector data is absent, use a unique node.
    if len(valid_connectors) < 2:
        valid_connectors = [f"isolated_start_{idx}", f"isolated_end_{idx}"]

    start_node = valid_connectors[0]
    end_node = valid_connectors[-1]

    graph.add_edge(
        start_node,
        end_node,
        road_index=idx,
        length=float(row["length_m_calculated"]),
        activity=int(row["calculated_places_50m"]),
        density=float(row["places_per_100m"]),
    )

    for connector_id in valid_connectors:
        connector_to_roads.setdefault(connector_id, set()).add(idx)

# ==================================================
# 5. SCORE ROADS WITHOUT SELECTING A STREET
# ==================================================

# Use activity counts and length-normalized density together.
# Very short roads can otherwise receive artificially high scores.
valid = roads["length_m_calculated"] >= MIN_SEGMENT_LENGTH

roads["activity_score"] = 0.0

if valid.any():
    counts = roads.loc[valid, "calculated_places_50m"]
    density = roads.loc[valid, "places_per_100m"]

    count_scale = max(float(counts.quantile(0.95)), 1.0)
    density_scale = max(float(density.quantile(0.95)), 1.0)

    roads.loc[valid, "activity_score"] = (
        0.5 * counts.clip(upper=count_scale) / count_scale
        + 0.5 * density.clip(upper=density_scale) / density_scale
    )

# Identify high-activity roads relative to this study area's distribution.
threshold = roads.loc[valid, "activity_score"].quantile(0.70)

roads["high_activity"] = (
    valid & (roads["activity_score"] >= threshold)
)

print("High-activity threshold:", round(float(threshold), 3))
print("High-activity segments:", int(roads["high_activity"].sum()))

# ==================================================
# 6. GROUP HIGH-ACTIVITY ROADS INTO CANDIDATES
# ==================================================

# Candidate roads are connected when they share connector IDs.
# Because connector locations may represent internal road points,
# candidates are preliminary and require visual validation.

active_indices = set(
    roads.index[roads["high_activity"]].tolist()
)

active_graph = nx.Graph()
for idx in active_indices:
    active_graph.add_node(idx)

for connector_id, road_indices in connector_to_roads.items():
    active_here = sorted(active_indices.intersection(road_indices))

    for i in range(len(active_here)):
        for j in range(i + 1, len(active_here)):
            active_graph.add_edge(active_here[i], active_here[j])

components = [
    component for component in nx.connected_components(active_graph)
    if len(component) >= MIN_CORRIDOR_SEGMENTS
]

# ==================================================
# 7. RANK CANDIDATES
# ==================================================

candidate_records = []
candidate_geometries = []

for candidate_id, component in enumerate(components, start=1):
    group = roads.loc[sorted(component)].copy()

    total_length = float(group["length_m_calculated"].sum())
    mean_activity = float(group["calculated_places_50m"].mean())
    mean_density = float(group["places_per_100m"].mean())
    mean_score = float(group["activity_score"].mean())

    if total_length < MIN_CORRIDOR_LENGTH:
        continue

    # Keep the score interpretable and reproducible.
    # It ranks candidates; it does not prove a real-world corridor.
    score = (
        0.40 * mean_score
        + 0.30 * min(total_length / 1000, 1.0)
        + 0.30 * min(len(group) / 10, 1.0)
    )

    geometry = unary_union(list(group.geometry))

    candidate_records.append({
        "candidate_id": candidate_id,
        "segment_count": len(group),
        "length_m": total_length,
        "mean_places_50m": mean_activity,
        "mean_places_per_100m": mean_density,
        "mean_activity_score": mean_score,
        "candidate_score": score,
    })
    candidate_geometries.append(geometry)

if not candidate_records:
    raise SystemExit(
        "No connected candidate passed the current filters. "
        "We should inspect connector structure and thresholds "
        "before changing parameters."
    )

candidates = gpd.GeoDataFrame(
    candidate_records,
    geometry=candidate_geometries,
    crs=CRS_METRES
).sort_values("candidate_score", ascending=False)

candidates = candidates.reset_index(drop=True)
candidates["rank"] = candidates.index + 1

candidates.drop(columns="geometry").to_csv(
    DATA / "network_corridor_candidates.csv", index=False
)

candidates.to_crs("EPSG:4326").to_file(
    DATA / "network_corridor_candidates.geojson",
    driver="GeoJSON"
)

# ==================================================
# 8. CREATE A MAP FOR VISUAL REVIEW
# ==================================================

fig, ax = plt.subplots(figsize=(13, 10))

roads.plot(ax=ax, color="lightgray", linewidth=0.5)

places.plot(
    ax=ax, color="gray", markersize=2, alpha=0.35
)

roads[roads["high_activity"]].plot(
    ax=ax, color="orange", linewidth=1.5
)

candidates.head(5).plot(
    ax=ax, column="rank", cmap="viridis",
    linewidth=3, legend=True
)

gpd.GeoSeries([boundary], crs=CRS_METRES).boundary.plot(
    ax=ax, color="black", linewidth=1.5
)

ax.set_title(
    "Preliminary Network-Based Corridor Candidates\n"
    "Orange = high-activity road segments; colored lines = top candidates"
)
ax.set_axis_off()
plt.tight_layout()

map_path = DATA / "network_corridor_candidates.png"
plt.savefig(map_path, dpi=220, bbox_inches="tight")
plt.close()

# ==================================================
# 9. REPORT RESULTS
# ==================================================

print("\nTOP NETWORK CORRIDOR CANDIDATES")
print("--------------------------------")
print(
    candidates.drop(columns="geometry").head(10).to_string(index=False)
)

print("\nFiles created:")
print(DATA / "network_corridor_candidates.csv")
print(DATA / "network_corridor_candidates.geojson")
print(map_path)

print(
    "\nIMPORTANT: These are preliminary candidates. "
    "Connector topology and the map must be validated before "
    "calling any candidate a real commercial corridor."
)
