
import ast
import re
import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.geometry import box
from shapely.ops import unary_union

# ============================================================
# 1. SETTINGS
# ============================================================

DATA = "data/"

PLACES_FILE = DATA + "places_filtered.geojson"
ROADS_FILE = DATA + "roads_in_study_area.geojson"

OUTPUT_ROADS = DATA + "functional_road_scores.geojson"
OUTPUT_TABLE = DATA + "functional_road_scores.csv"
OUTPUT_CORRIDORS = DATA + "functional_corridor_candidates.geojson"
OUTPUT_BOUNDARIES = DATA + "functional_corridor_boundaries.geojson"
OUTPUT_SUMMARY = DATA + "functional_corridor_summary.csv"

# Study area used in your existing Chicago project
WEST, SOUTH, EAST, NORTH = -87.683, 41.903, -87.659, 41.913
STUDY_AREA = box(WEST, SOUTH, EAST, NORTH)

# Use a local projected CRS so distances are measured in metres
PROJECTED_CRS = "EPSG:32616"

# Businesses within this distance of a road are considered nearby
ROAD_BUFFER_M = 50

# Candidate corridor settings
BOUNDARY_WIDTH_M = 35
MIN_CORRIDOR_LENGTH_M = 200
MAX_CORRIDOR_LENGTH_M = 1500

# A road must have at least this many nearby businesses
MIN_NEARBY_BUSINESSES = 3

# Minimum relative functional-group score for an active road
MIN_GROUP_SCORE = 0.50

# Maximum turn angle allowed when joining road segments
MAX_TURN_DEGREES = 45

# ============================================================
# 2. DEFINE FUNCTIONAL BUSINESS GROUPS
# ============================================================

# These are keyword rules applied to analysis_category.
# They are a starting point; review the actual category labels
# printed by the script and extend these lists if necessary.

FUNCTIONAL_GROUPS = {
    "Tourism and hospitality": [
        "hotel",
        "motel",
        "hostel",
        "accommodation",
        "lodging",
        "tourist_attraction",
        "tourist attraction",
        "attraction",
        "museum",
        "monument",
        "visitor",
        "travel_agency",
        "travel agency",
        "tour_operator",
        "tour operator",
        "landmark",
        "sightseeing",
        "tourism",
    ],

    "Retail": [
        "store",
        "retail",
        "shopping",
        "fashion",
        "apparel",
        "clothing",
        "electronics",
        "department_store",
        "department store",
        "bookstore",
        "jewelry",
        "jewellery",
        "furniture_store",
        "furniture store",
        "hardware",
        "gift_shop",
        "gift shop",
        "supermarket",
        "convenience_store",
        "convenience store",
        "food_and_beverage_store",
    ],

    "Food and nightlife": [
        "restaurant",
        "casual_eatery",
        "casual eatery",
        "coffee_shop",
        "coffee shop",
        "cafe",
        "bar",
        "pub",
        "nightclub",
        "night_club",
        "night club",
        "music_venue",
        "music venue",
        "food",
        "dining",
        "bakery",
        "fast_food",
        "fast food",
        "ice_cream",
        "ice cream",
        "pizzeria",
    ],

    "Professional services": [
        "professional_service",
        "professional service",
        "real_estate_service",
        "real estate service",
        "financial_service",
        "financial service",
        "corporate_or_business_office",
        "corporate or business office",
        "b2b_office_and_professional_service",
        "business_office",
        "business office",
        "lawyer",
        "legal_service",
        "legal service",
        "accounting",
        "consulting",
        "insurance",
        "bank",
    ],
}

# Specific categories that should not be automatically classified
# using overly broad words such as "service".
#
# Categories not matching a group will be labelled "Other".
# This avoids assuming every service is a professional service.

def normalize_category(value):
    if pd.isna(value):
        return "unknown"
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def assign_functional_group(category):
    normalized = normalize_category(category)

    matches = []

    for group, keywords in FUNCTIONAL_GROUPS.items():
        for keyword in keywords:
            key = normalize_category(keyword)

            # Match a complete category name or a meaningful
            # category-name component.
            if normalized == key or key in normalized:
                matches.append(group)
                break

    # Categories can be ambiguous. Do not silently assign a
    # category to multiple functional groups.
    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        # Resolve common overlaps using the category's more
        # specific terms.
        priority = [
            "Tourism and hospitality",
            "Food and nightlife",
            "Retail",
            "Professional services",
        ]

        for group in priority:
            if group in matches:
                return group

    return "Other"


# ============================================================
# 3. LOAD AND PREPARE DATA
# ============================================================

print("Loading business and road data...")

places = gpd.read_file(PLACES_FILE)
roads = gpd.read_file(ROADS_FILE)

if "analysis_category" not in places.columns:
    raise ValueError(
        "analysis_category is missing from places_filtered.geojson"
    )

if "connectors" not in roads.columns:
    raise ValueError(
        "connectors is missing from roads_in_study_area.geojson"
    )

places = places[
    places.geometry.notna() & ~places.geometry.is_empty
].copy()

roads = roads[
    roads.geometry.notna() & ~roads.geometry.is_empty
].copy()

# Restrict places to the same study rectangle
places = places.to_crs("EPSG:4326")
roads = roads.to_crs("EPSG:4326")

places = places[
    places.geometry.intersects(STUDY_AREA)
].copy()

roads = roads[
    roads.geometry.intersects(STUDY_AREA)
].copy()

# Use point locations for business proximity.
# representative_point works for both points and polygons.
places = places.to_crs(PROJECTED_CRS)
roads = roads.to_crs(PROJECTED_CRS)

places["geometry"] = places.geometry.representative_point()

# Save the original business category and add a functional group
places["analysis_category"] = (
    places["analysis_category"].fillna("unknown").astype(str)
)

places["functional_group"] = places["analysis_category"].apply(
    assign_functional_group
)

print("\nBusiness records:", len(places))
print("Road records:", len(roads))

print("\nFunctional-group totals:")
print(places["functional_group"].value_counts().to_string())

print("\nCategories labelled Other:")
other_categories = sorted(
    places.loc[
        places["functional_group"] == "Other",
        "analysis_category"
    ].unique()
)

for category in other_categories:
    print(" -", category)

# ============================================================
# 4. SCORE EACH ROAD FOR EACH FUNCTIONAL GROUP
# ============================================================

print("\nScoring road segments...")

roads = roads.reset_index(drop=True)
roads["road_id"] = roads.index.astype(int)

# Build a 50-metre buffer around every road segment
road_buffers = roads[["road_id", "geometry"]].copy()
road_buffers["geometry"] = road_buffers.geometry.buffer(
    ROAD_BUFFER_M
)

# Spatial join finds business points inside road buffers
place_columns = places[
    ["analysis_category", "functional_group", "geometry"]
].copy()

nearby = gpd.sjoin(
    place_columns,
    road_buffers,
    how="inner",
    predicate="within"
)

# Count all nearby businesses and counts for each group
all_counts = (
    nearby.groupby("road_id")
    .size()
    .rename("nearby_businesses")
)

roads = roads.merge(
    all_counts,
    left_on="road_id",
    right_index=True,
    how="left"
)

roads["nearby_businesses"] = (
    roads["nearby_businesses"].fillna(0).astype(int)
)

for group in FUNCTIONAL_GROUPS:
    counts = (
        nearby.loc[nearby["functional_group"] == group]
        .groupby("road_id")
        .size()
    )

    column = "count_" + re.sub(
        r"[^a-z0-9]+", "_", group.lower()
    ).strip("_")

    roads[column] = roads["road_id"].map(counts).fillna(0).astype(int)

# Add overall category variety near each road
variety = (
    nearby.groupby("road_id")["analysis_category"]
    .nunique()
    .rename("category_variety")
)

roads = roads.merge(
    variety,
    left_on="road_id",
    right_index=True,
    how="left"
)

roads["category_variety"] = (
    roads["category_variety"].fillna(0).astype(int)
)

# ============================================================
# 5. CALCULATE GROUP-SPECIFIC ROAD SCORES
# ============================================================

def percentile_score(series):
    """Return relative percentile scores between 0 and 1."""
    if series.nunique() <= 1:
        return pd.Series(0.0, index=series.index)

    return series.rank(pct=True, method="average")


group_score_columns = []

for group in FUNCTIONAL_GROUPS:
    group_key = re.sub(
        r"[^a-z0-9]+", "_", group.lower()
    ).strip("_")

    count_column = "count_" + group_key
    score_column = "score_" + group_key

    density_score = percentile_score(roads[count_column])
    variety_score = percentile_score(roads["category_variety"])
    activity_score = percentile_score(roads["nearby_businesses"])

    # Functional-group score gives more weight to the selected
    # group's businesses than to general commercial activity.
    roads[score_column] = (
        0.60 * density_score
        + 0.20 * activity_score
        + 0.20 * variety_score
    )

    group_score_columns.append(score_column)

# Determine which functional group has the highest score
roads["dominant_group"] = roads[group_score_columns].idxmax(axis=1)

score_to_group = {
    "score_" + re.sub(
        r"[^a-z0-9]+", "_", group.lower()
    ).strip("_"): group
    for group in FUNCTIONAL_GROUPS
}

roads["dominant_group"] = roads["dominant_group"].map(score_to_group)

# A segment with no mapped functional-group businesses should
# not receive a misleading functional label.
group_count_columns = [
    "count_" + re.sub(
        r"[^a-z0-9]+", "_", group.lower()
    ).strip("_")
    for group in FUNCTIONAL_GROUPS
]

roads["mapped_group_businesses"] = roads[group_count_columns].sum(axis=1)

roads.loc[
    roads["mapped_group_businesses"] == 0,
    "dominant_group"
] = "Other"

# ============================================================
# 6. PARSE ROAD CONNECTORS
# ============================================================

def parse_connectors(value):
    """
    Read connector IDs from common list/dictionary formats.
    Returns a set of connector IDs.
    """
    if value is None:
        return set()

    if isinstance(value, float) and np.isnan(value):
        return set()

    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return set()

    if isinstance(value, dict):
        if "connector_id" in value:
            return {str(value["connector_id"])}

        result = set()
        for item in value.values():
            result |= parse_connectors(item)
        return result

    if isinstance(value, (list, tuple, set)):
        result = set()
        for item in value:
            result |= parse_connectors(item)
        return result

    return set()


roads["connector_ids"] = roads["connectors"].apply(parse_connectors)

# ============================================================
# 7. DISCOVER CONNECTED CORRIDOR CANDIDATES
# ============================================================

print("\nDiscovering connected corridor candidates...")

candidate_rows = []
boundary_rows = []
summary_rows = []

candidate_id = 0

# Run a separate road-path search for each functional group.
for group in FUNCTIONAL_GROUPS:

    group_key = re.sub(
        r"[^a-z0-9]+", "_", group.lower()
    ).strip("_")

    score_column = "score_" + group_key
    count_column = "count_" + group_key

    group_roads = roads[
        (roads[score_column] >= MIN_GROUP_SCORE)
        & (roads[count_column] > 0)
        & (roads["nearby_businesses"] >= MIN_NEARBY_BUSINESSES)
    ].copy()

    if group_roads.empty:
        print(f"No qualifying road segments for {group}")
        continue

    # Create connector-to-road lookup
    connector_to_roads = {}

    for road_id, connectors in zip(
        group_roads["road_id"],
        group_roads["connector_ids"]
    ):
        for connector in connectors:
            connector_to_roads.setdefault(connector, set()).add(
                int(road_id)
            )

    # Build adjacency graph from shared connector IDs
    adjacency = {
        int(road_id): set()
        for road_id in group_roads["road_id"]
    }

    for connected_roads in connector_to_roads.values():
        for road_id in connected_roads:
            adjacency[road_id].update(
                connected_roads - {road_id}
            )

    # Find connected components among qualifying roads
    visited = set()

    for start in adjacency:
        if start in visited:
            continue

        stack = [start]
        component = []
        visited.add(start)

        while stack:
            current = stack.pop()
            component.append(current)

            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)

        component_roads = group_roads[
            group_roads["road_id"].isin(component)
        ].copy()

        # Components with too few segments are not corridors
        if len(component_roads) < 2:
            continue

        merged_geometry = unary_union(
            component_roads.geometry.tolist()
        )

        length_m = sum(
            geometry.length
            for geometry in component_roads.geometry
        )

        # Reject implausibly small or large components
        if not (
            MIN_CORRIDOR_LENGTH_M
            <= length_m
            <= MAX_CORRIDOR_LENGTH_M
        ):
            continue

        # Calculate dominant-group concentration from nearby
        # group-specific counts across these road segments.
        group_count = int(component_roads[count_column].sum())
        all_group_counts = {
            other_group: int(
                component_roads[
                    "count_" + re.sub(
                        r"[^a-z0-9]+", "_", other_group.lower()
                    ).strip("_")
                ].sum()
            )
            for other_group in FUNCTIONAL_GROUPS
        }

        total_mapped_count = sum(all_group_counts.values())

        if total_mapped_count > 0:
            group_share = group_count / total_mapped_count
        else:
            group_share = 0.0

        corridor_type = (
            group
            if group_share >= 0.60
            else "Mixed-use"
        )

        candidate_id += 1

        candidate_rows.append({
            "candidate_id": candidate_id,
            "functional_group": group,
            "corridor_type": corridor_type,
            "group_share": group_share,
            "segment_count": len(component_roads),
            "length_m": length_m,
            "average_group_score": float(
                component_roads[score_column].mean()
            ),
            "nearby_businesses_sum": int(
                component_roads["nearby_businesses"].sum()
            ),
            "group_business_count_sum": group_count,
            "geometry": merged_geometry,
        })

        boundary_rows.append({
            "candidate_id": candidate_id,
            "functional_group": group,
            "corridor_type": corridor_type,
            "geometry": merged_geometry.buffer(BOUNDARY_WIDTH_M),
        })

        summary = {
            "candidate_id": candidate_id,
            "functional_group": group,
            "corridor_type": corridor_type,
            "group_share": group_share,
            "segment_count": len(component_roads),
            "length_m": length_m,
            "average_group_score": float(
                component_roads[score_column].mean()
            ),
        }

        for other_group, count in all_group_counts.items():
            summary[other_group + "_count"] = count

        summary_rows.append(summary)

# ============================================================
# 8. SAVE RESULTS
# ============================================================

# Save road-level functional scores
roads.drop(columns=["connector_ids"]).to_file(
    OUTPUT_ROADS,
    driver="GeoJSON"
)

roads.drop(columns=["geometry", "connector_ids"]).to_csv(
    OUTPUT_TABLE,
    index=False
)

if candidate_rows:
    candidates = gpd.GeoDataFrame(
        candidate_rows,
        geometry="geometry",
        crs=PROJECTED_CRS
    )

    boundaries = gpd.GeoDataFrame(
        boundary_rows,
        geometry="geometry",
        crs=PROJECTED_CRS
    )

    candidates = candidates.to_crs("EPSG:4326")
    boundaries = boundaries.to_crs("EPSG:4326")

    candidates.to_file(
        OUTPUT_CORRIDORS,
        driver="GeoJSON"
    )

    boundaries.to_file(
        OUTPUT_BOUNDARIES,
        driver="GeoJSON"
    )

    pd.DataFrame(summary_rows).to_csv(
        OUTPUT_SUMMARY,
        index=False
    )

    print("\nFunctional corridor candidates:")
    print(
        pd.DataFrame(summary_rows)[
            [
                "candidate_id",
                "functional_group",
                "corridor_type",
                "group_share",
                "segment_count",
                "length_m",
            ]
        ].to_string(index=False)
    )

    print("\nSaved:")
    print(OUTPUT_ROADS)
    print(OUTPUT_TABLE)
    print(OUTPUT_CORRIDORS)
    print(OUTPUT_BOUNDARIES)
    print(OUTPUT_SUMMARY)

else:
    print(
        "\nNo candidates met the current filters. "
        "Try lowering MIN_GROUP_SCORE or checking the category mapping."
    )

print("\nFinished.")
