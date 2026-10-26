from pathlib import Path
import re
from collections import defaultdict

import geopandas as gpd
import pandas as pd
import streamlit as st
import folium

from streamlit_folium import st_folium
from shapely.geometry import box, Point
from shapely.ops import unary_union


# ============================================================
# 1. PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Chicago Corridor Explorer",
    page_icon="🗺️",
    layout="wide",
)

st.title("🗺️ Chicago Commercial Corridor Explorer")
st.write(
    "Discover potential commercial corridors using business activity, "
    "road connectivity, and mapped place categories."
)
st.info(
    "This is a research prototype. Detected corridors are candidates "
    "for review, not official commercial-district boundaries."
)


# ============================================================
# 2. STUDY AREAS
# ============================================================


ORIGINAL_AREA = "Wicker Park–Bucktown (original study area)"
LOGAN_AREA = "Logan Square"
PILSEN_AREA = "Pilsen"
CHINATOWN_AREA = "Chinatown"

AREAS = [
    ORIGINAL_AREA,
    LOGAN_AREA,
    PILSEN_AREA,
    CHINATOWN_AREA,
]


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
PILSEN_DATA_DIR = DATA_DIR / "pilsen"
CHINATOWN_DATA_DIR = DATA_DIR / "chinatown"
CACHE_DIR = DATA_DIR / "osm_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

LOGAN_BBOX = (-87.717, 41.921, -87.692, 41.931)

LOGAN_BBOX = (-87.717, 41.921, -87.692, 41.931)
PILSEN_BBOX = (-87.665, 41.854, -87.649, 41.860)


LOGAN_PLACE_FILE = (
    "overture-2026-09-23.1-place--87.717,41.921,-87.692,41.931.geojson"
)
LOGAN_SEGMENT_FILE = (
    "overture-2026-09-23.1-segment--87.717,41.921,-87.692,41.931.geojson"
)
LOGAN_BUILDING_FILE = (
    "overture-2026-09-23.1-building--87.717,41.921,-87.692,41.931.geojson"
)
PILSEN_PLACE_FILE = (
    "overture-2026-10-21.0-place--87.665,41.854,-87.649,41.860.geojson"
)
PILSEN_SEGMENT_FILE = (
    "overture-2026-10-21.0-segment--87.665,41.854,-87.649,41.860.geojson"
)
PILSEN_BUILDING_FILE = (
    "overture-2026-10-21.0-building--87.665,41.854,-87.649,41.860.geojson"
)

# Chinatown study area: bounds from the downloaded Overture files
CHINATOWN_BBOX = (-87.639, 41.848, -87.623, 41.854)

CHINATOWN_PLACE_FILE = (
    "overture-2026-10-21.0-place--87.639,41.848,-87.623,41.854.geojson"
)

CHINATOWN_SEGMENT_FILE = (
    "overture-2026-10-21.0-segment--87.639,41.848,-87.623,41.854.geojson"
)

CHINATOWN_BUILDING_FILE = (
    "overture-2026-10-21.0-building--87.639,41.848,-87.623,41.854.geojson"
)



# ============================================================
# 3. BUSINESS GROUPS AND COLORS
# ============================================================

GROUPS = {
    "Tourism and hospitality": [
        "hotel", "motel", "hostel", "accommodation", "lodging",
        "tourism", "museum", "attraction", "gallery", "visitor",
        "guest_house",
    ],
    "Retail": [
        "shop", "retail", "store", "shopping", "clothing", "fashion",
        "books", "jewelry", "supermarket", "marketplace", "florist",
    ],
    "Food and nightlife": [
        "restaurant", "cafe", "coffee", "fast_food", "bar", "pub",
        "nightclub", "food", "bakery", "ice_cream", "biergarten",
        "pizza", "deli", "brewery", "food_and_drink",
    ],
    "Professional services": [
        "office", "lawyer", "accountant", "financial", "bank",
        "insurance", "consulting", "real_estate", "design_service",
        "architect", "dentist", "doctor", "beauty", "hairdresser",
        "repair", "professional_service",
    ],
}

GROUP_COLORS = {
    "Tourism and hospitality": "#8e44ad",
    "Retail": "#e67e22",
    "Food and nightlife": "#2980b9",
    "Professional services": "#27ae60",
    "Other": "#7f8c8d",
}

CORRIDOR_COLORS = {
    ORIGINAL_AREA: "#1769aa",
    LOGAN_AREA: "#d62728",
    PILSEN_AREA: "#2E8B57",
    CHINATOWN_AREA: "#e67e22",
}


# ============================================================
# 4. COMMON HELPERS
# ============================================================

def read_geojson(path):
    """Read GeoJSON and assign WGS84 if CRS is missing."""
    frame = gpd.read_file(path)
    if frame.crs is None:
        frame = frame.set_crs("EPSG:4326")
    return frame



def safe_to_wgs84(frame):
    """Convert a GeoDataFrame to WGS84 and make its data JSON-safe."""
    import numpy as np
    import pandas as pd

    if frame is None:
        return None

    frame = frame.copy()

    if frame.crs is None:
        frame = frame.set_crs("EPSG:4326")
    elif frame.crs.to_string() != "EPSG:4326":
        frame = frame.to_crs("EPSG:4326")

    def make_json_safe(value):
        if isinstance(value, np.ndarray):
            return [make_json_safe(item) for item in value.tolist()]

        if isinstance(value, np.generic):
            return make_json_safe(value.item())

        if isinstance(value, dict):
            return {
                key: make_json_safe(item)
                for key, item in value.items()
            }

        if isinstance(value, (list, tuple)):
            return [make_json_safe(item) for item in value]

        if value is pd.NA or value is pd.NaT:
            return None

        return value

    for column in frame.columns:
        if column != frame.geometry.name:
            frame[column] = frame[column].map(make_json_safe)

    return frame


def places_to_geojson(frame):
    """Convert place data into JSON-safe GeoJSON."""
    import json
    import numpy as np

    frame = safe_to_wgs84(frame)

    if frame is None:
        return '{"type":"FeatureCollection","features":[]}'

    geojson = json.loads(frame.to_json(na="null", drop_id=True))
    
    
    def clean(value):
        if isinstance(value, np.ndarray):
            return [clean(item) for item in value.tolist()]

        if isinstance(value, np.generic):
            return clean(value.item())

        if isinstance(value, dict):
            return {str(key): clean(item) for key, item in value.items()}

        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]

        if value is None:
            return None

        if isinstance(value, float):
            return value if np.isfinite(value) else None

        return value

    return json.dumps(clean(geojson), allow_nan=False)


    def clean(value):
        if isinstance(value, np.ndarray):
            return [clean(item) for item in value.tolist()]

        if isinstance(value, np.generic):
            return clean(value.item())

        if isinstance(value, dict):
            return {str(key): clean(item) for key, item in value.items()}

        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]

        if value is None:
            return None

        if isinstance(value, float):
            return value if np.isfinite(value) else None

        return value

    return json.dumps(clean(geojson), allow_nan=False)



def clean_geometry(frame):
    """Remove missing and empty geometries."""
    if frame is None:
        return None
    if frame.empty:
        return frame.copy()
    return frame[
        frame.geometry.notna() & ~frame.geometry.is_empty
    ].copy()


def normalize_text(value):
    """Convert nested values and tags into searchable text."""
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(
            f"{key} {normalize_text(item)}"
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set)):
        return " ".join(normalize_text(item) for item in value)
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).lower().replace("_", " ").replace("-", " ")



def classify(row):
    """Classify an OSM or Overture place into a business group."""

    keys = [
        "amenity", "shop", "office", "tourism", "leisure", "craft",
        "name", "cuisine", "functional_group", "business_category",
        "category", "category_name", "primary_category",
        "analysis_category", "type", "subtype", "class",
        "basic_category", "taxonomy",
    ]

    searchable = " ".join(
        normalize_text(row.get(key, "")) for key in keys
    )

    # Keep your existing business groups.
    group_order = [
        "Food and nightlife",
        "Tourism and hospitality",
        "Retail",
        "Professional services",
    ]

    for group in group_order:
        for keyword in GROUPS[group]:
            normalized_keyword = normalize_text(keyword)
            if normalized_keyword and normalized_keyword in searchable:
                return group

    # Break places not matched above into more specific groups.
    extra_groups = {
        "Healthcare": [
            "hospital", "clinic", "doctors", "dentist",
            "pharmacy", "medical", "healthcare",
        ],
        "Education": [
            "school", "college", "university", "kindergarten",
            "childcare", "training",
        ],
        "Personal services": [
            "hairdresser", "beauty", "salon", "barber",
            "laundry", "dry cleaning", "tailor",
        ],
        "Transport and automotive": [
            "parking", "fuel", "charging station", "car repair",
            "car wash", "automotive", "bicycle repair",
            "bus station", "taxi",
        ],
        "Entertainment and recreation": [
            "cinema", "theatre", "museum", "arts centre",
            "sports centre", "stadium", "bowling",
            "nightclub", "park", "fitness centre",
        ],
        "Community and public services": [
            "library", "community centre", "townhall",
            "government", "police", "fire station",
            "post office", "social facility",
        ],
        "Financial services": [
            "bank", "atm", "insurance", "accountant",
            "financial",
        ],
        "Religious places": [
            "place of worship", "church", "mosque",
            "temple", "synagogue",
        ],
        "Industrial and utilities": [
            "warehouse", "factory", "industrial",
            "electricity", "water works", "waste management",
        ],
    }

    for group, keywords in extra_groups.items():
        for keyword in keywords:
            if normalize_text(keyword) in searchable:
                return group

    return "Other / Unclassified"



def estimate_metric_crs(frame):
    """Choose a projected CRS for distance calculations."""
    try:
        crs = frame.estimate_utm_crs()
        if crs is not None:
            return crs
    except Exception:
        pass
    return "EPSG:32616"  # Chicago, UTM zone 16N


def percentile(series):
    """Convert numeric values into relative percentile scores."""
    numeric = pd.to_numeric(series, errors="coerce").fillna(0)
    if numeric.nunique() <= 1:
        return pd.Series(0.0, index=numeric.index)
    return numeric.rank(pct=True, method="average")


def safe_number(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


# ============================================================
# 5. FIND ORIGINAL WICKER PARK DATA
# ============================================================

def find_original_data_dir():
    """Find the existing original project data directory locally."""
    home = Path.home()
    candidates = [
        DATA_DIR,
        APP_DIR.parent / "data",
        home / "OneDrive" / "Desktop" / "chicago_corridor_project" / "data",
        home / "Desktop" / "chicago_corridor_project" / "data",
    ]
    for folder in candidates:
        if (folder / "places_filtered.geojson").exists():
            return folder
    return None


# ============================================================
# 6. LOAD ORIGINAL WICKER PARK RESULTS
# ============================================================

def load_precomputed_wicker_results():
    """Load saved Wicker Park outputs without recalculating them."""
    data_dir = find_original_data_dir()
    if data_dir is None:
        raise FileNotFoundError(
            "Could not find the original data folder containing "
            "places_filtered.geojson. Check the original project path."
        )

    places_path = data_dir / "places_filtered.geojson"
    road_options = [
        "functional_road_scores.geojson",
        "roads_with_business_scores.geojson",
        "roads_in_study_area.geojson",
    ]
    roads_path = next(
        (
            data_dir / filename
            for filename in road_options
            if (data_dir / filename).exists()
        ),
        None,
    )
    corridors_path = data_dir / "functional_corridor_candidates.geojson"
    boundaries_path = data_dir / "functional_corridor_boundaries.geojson"

    required = {
        "places": places_path,
        "roads": roads_path,
        "corridors": corridors_path,
        "boundaries": boundaries_path,
    }
    missing = [
        name for name, path in required.items()
        if path is None or not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing original result files: " + ", ".join(missing)
        )

    places = safe_to_wgs84(clean_geometry(read_geojson(places_path)))
    roads = safe_to_wgs84(clean_geometry(read_geojson(roads_path)))
    corridors = safe_to_wgs84(clean_geometry(read_geojson(corridors_path)))
    boundaries = safe_to_wgs84(clean_geometry(read_geojson(boundaries_path)))

    if "functional_group" not in places.columns:
        places["functional_group"] = places.apply(classify, axis=1)
    else:
        valid_groups = set(GROUPS) | {"Other"}
        invalid = ~places["functional_group"].astype(str).isin(valid_groups)
        places.loc[invalid, "functional_group"] = places.loc[invalid].apply(
            classify, axis=1
        )

    if "name" not in places.columns:
        places["name"] = "Place"
    if "candidate_id" not in corridors.columns:
        corridors["candidate_id"] = [f"W{i + 1}" for i in range(len(corridors))]
    if "candidate_id" not in boundaries.columns:
        boundaries["candidate_id"] = [f"W{i + 1}" for i in range(len(boundaries))]
    if "corridor_type" not in corridors.columns:
        corridors["corridor_type"] = "Candidate"
    if "functional_group" not in corridors.columns:
        corridors["functional_group"] = "Mixed-use"
    if "corridor_type" not in boundaries.columns:
        boundaries["corridor_type"] = "Candidate"
    if "functional_group" not in boundaries.columns:
        boundaries["functional_group"] = "Mixed-use"

    boundary = gpd.GeoDataFrame(
        {"name": ["Wicker Park–Bucktown study area"]},
        geometry=[box(-87.683, 41.903, -87.659, 41.913)],
        crs="EPSG:4326",
    )

    st.success(
        "Loaded the existing Wicker Park–Bucktown results. "
        "Original files were not recalculated or overwritten."
    )
    return {
        "area": ORIGINAL_AREA,
        "boundary": boundary,
        "places": places,
        "roads": roads,
        "corridors": corridors,
        "boundaries": boundaries,
        "buildings": None,
    }


# ============================================================
# 7. LOAD LOGAN SQUARE OVERTURE DATA
# ============================================================

def extract_overture_name(value):
    if isinstance(value, dict):
        return value.get("primary", "Unnamed place")
    if value is None:
        return "Unnamed place"
    try:
        if pd.isna(value):
            return "Unnamed place"
    except (TypeError, ValueError):
        pass
    return str(value)


def get_segment_endpoints(connectors):
    """Extract first and last connector IDs from a segment connectors list."""
    if not isinstance(connectors, (list, tuple)):
        return None, None
    valid = []
    for item in connectors:
        if not isinstance(item, dict):
            continue
        connector_id = item.get("connector_id")
        position = item.get("at")
        if connector_id is None:
            continue
        try:
            position = float(position)
        except (TypeError, ValueError):
            continue
        valid.append((position, str(connector_id)))
    if len(valid) < 2:
        return None, None
    valid.sort(key=lambda item: item[0])
    return valid[0][1], valid[-1][1]


def get_overture_logan_data(area=LOGAN_AREA):
    """Load Logan Square's local Overture GeoJSON files."""
    
    
    if area == PILSEN_AREA:
        places_path = PILSEN_DATA_DIR / PILSEN_PLACE_FILE
        segments_path = PILSEN_DATA_DIR / PILSEN_SEGMENT_FILE
        buildings_path = PILSEN_DATA_DIR / PILSEN_BUILDING_FILE

    elif area == CHINATOWN_AREA:
        places_path = CHINATOWN_DATA_DIR / CHINATOWN_PLACE_FILE
        segments_path = CHINATOWN_DATA_DIR / CHINATOWN_SEGMENT_FILE
        buildings_path = CHINATOWN_DATA_DIR / CHINATOWN_BUILDING_FILE

    else:
        places_path = DATA_DIR / LOGAN_PLACE_FILE
        segments_path = DATA_DIR / LOGAN_SEGMENT_FILE
        buildings_path = DATA_DIR / LOGAN_BUILDING_FILE


    missing = [
        path.name for path in [places_path, segments_path]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing Logan Square data files in:\n{DATA_DIR}\n\n"
            + "\n".join(missing)
        )

    places = safe_to_wgs84(read_geojson(places_path))
    segments = safe_to_wgs84(read_geojson(segments_path))
    buildings = None

    if buildings_path.exists():
        try:
            buildings = clean_geometry(
                safe_to_wgs84(read_geojson(buildings_path))
            )
        except Exception as exc:
            st.warning(f"Building footprints could not be loaded: {exc}")

    places = clean_geometry(places)
    segments = clean_geometry(segments)

    if "names" in places.columns:
        places["name"] = places["names"].apply(extract_overture_name)
    elif "name" not in places.columns:
        places["name"] = "Unnamed place"

    if "taxonomy" in places.columns:
        places["primary_category"] = places["taxonomy"].apply(
            lambda value: (
                value.get("primary") if isinstance(value, dict) else None
            )
        )

    if "basic_category" in places.columns:
        places["category"] = places["basic_category"].astype(str)
    elif "category" not in places.columns:
        places["category"] = "unknown"

    places["analysis_category"] = places.apply(
        lambda row: (
            normalize_text(row.get("primary_category", ""))
            or normalize_text(row.get("category", ""))
            or "unknown"
        ),
        axis=1,
    )
    places["functional_group"] = places.apply(classify, axis=1)

    if "subtype" in segments.columns:
        subtype = segments["subtype"].astype(str).str.lower()
        road_mask = subtype.isin(["road", "street", "highway"])
        if road_mask.any():
            segments = segments.loc[road_mask].copy()

    if "class" in segments.columns:
        excluded_classes = {
            "footway", "pedestrian", "steps", "path", "cycleway", "bridleway"
        }
        mask = ~segments["class"].astype(str).str.lower().isin(excluded_classes)
        segments = segments.loc[mask].copy()

    if "connectors" not in segments.columns:
        raise ValueError(
            "The Overture Segments file does not contain the required connectors column."
        )

    endpoints = segments["connectors"].apply(get_segment_endpoints)
    segments["u"] = endpoints.apply(lambda pair: pair[0])
    segments["v"] = endpoints.apply(lambda pair: pair[1])
    segments = segments[
        segments["u"].notna() & segments["v"].notna()
    ].copy()

    places = places.reset_index(drop=True)
    segments = segments.reset_index(drop=True)

    if places.empty:
        raise ValueError("The Logan Square Places file has no usable features.")
    if segments.empty:
        raise ValueError(
            "No usable connected road segments were found. "
            "Check the Overture connector data and filters."
        )

    
  
    
    if area == PILSEN_AREA:
        study_bbox = PILSEN_BBOX
        area_name = "Pilsen"

    elif area == CHINATOWN_AREA:
        study_bbox = CHINATOWN_BBOX
        area_name = "Chinatown"

    else:
        study_bbox = LOGAN_BBOX
        area_name = "Logan Square"


    boundary = gpd.GeoDataFrame(
        {"name": [f"{area_name} study area"]},
        geometry=[box(*study_bbox)],
        crs="EPSG:4326",
    )

    st.success(
        f"Loaded {area_name}: {len(places):,} places and "
        f"{len(segments):,} road segments."
    )

    return boundary, places, None, segments, buildings


# ============================================================
# 8. CONNECTIVITY: CONNECT ROAD SEGMENTS
# ============================================================

def build_road_components(active_roads, snap_tolerance_m=12):
    """
    Find connected road components using connector IDs and nearby
    road endpoints. Returns lists of active_roads row indices.
    """
    if active_roads.empty:
        return []

    roads = active_roads.copy()
    adjacency = {int(idx): set() for idx in roads.index}
    node_to_roads = defaultdict(set)

    for idx, row in roads.iterrows():
        u, v = row.get("u"), row.get("v")
        if pd.notna(u):
            node_to_roads[str(u)].add(int(idx))
        if pd.notna(v):
            node_to_roads[str(v)].add(int(idx))

    for road_ids in node_to_roads.values():
        for road_id in road_ids:
            adjacency[road_id].update(road_ids - {road_id})

    endpoint_records = []
    for idx, row in roads.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "LineString":
            coords = list(geom.coords)
            if len(coords) >= 2:
                endpoint_records.extend([
                    (int(idx), Point(coords[0])),
                    (int(idx), Point(coords[-1])),
                ])
        elif geom.geom_type == "MultiLineString":
            parts = list(geom.geoms)
            if parts:
                first_coords = list(parts[0].coords)
                last_coords = list(parts[-1].coords)
                endpoint_records.extend([
                    (int(idx), Point(first_coords[0])),
                    (int(idx), Point(last_coords[-1])),
                ])

    if endpoint_records and snap_tolerance_m > 0:
        points = gpd.GeoDataFrame(
            {"road_id": [r[0] for r in endpoint_records]},
            geometry=[r[1] for r in endpoint_records],
            crs=roads.crs,
        )
        try:
            spatial_index = points.sindex
            for i, row in points.iterrows():
                point = row.geometry
                road_a = int(row["road_id"])
                candidates = spatial_index.intersection(
                    point.buffer(snap_tolerance_m).bounds
                )
                for j in candidates:
                    if j <= i:
                        continue
                    other = points.iloc[j]
                    road_b = int(other["road_id"])
                    if road_a == road_b:
                        continue
                    if point.distance(other.geometry) <= snap_tolerance_m:
                        adjacency[road_a].add(road_b)
                        adjacency[road_b].add(road_a)
        except Exception:
            # Connector-based grouping still works if the spatial index fails.
            pass

    components = []
    visited = set()
    for start in adjacency:
        if start in visited:
            continue
        stack = [start]
        visited.add(start)
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbour in adjacency[current]:
                if neighbour not in visited:
                    visited.add(neighbour)
                    stack.append(neighbour)
        components.append(component)
    return components


# ============================================================
# 9. CREATE ONE TIGHT CORRIDOR BOUNDARY
# ============================================================


def make_corridor_boundary(
    road_geometries,
    nearby_business_points=None,
    boundary_width=20,
    business_buffer=12,
    smoothing_distance=5,
):
    """
    Create a corridor polygon around connected roads and nearby
    business points. All distances are in metres.
    """
    road_geometries = [
        geom for geom in road_geometries
        if geom is not None and not geom.is_empty
    ]

    if not road_geometries:
        return None

    roads_union = unary_union(road_geometries)

    # Initial boundary around the connected road network.
    corridor = roads_union.buffer(
        max(float(boundary_width), 1.0),
        cap_style="round",
        join_style="round",
    )

    if corridor.is_empty:
        return None

    # Include business points only when they are reasonably close
    # to this road component. This avoids pulling in distant businesses.
    if nearby_business_points is not None:
        points = [
            geom for geom in nearby_business_points
            if geom is not None and not geom.is_empty
        ]

        if points:
            business_area = unary_union(points).buffer(
                max(float(business_buffer), 1.0)
            )

            # Keep only the business area connected to the road buffer.
            connected_business_area = business_area.intersection(
                roads_union.buffer(
                    max(float(boundary_width), 1.0)
                    + max(float(business_buffer), 1.0)
                )
            )

            corridor = unary_union([
                corridor,
                connected_business_area,
            ])

    # Smooth small gaps and sharp edges without forcing separate
    # commercial pockets into one enormous polygon.
    if smoothing_distance > 0:
        distance = min(
            float(smoothing_distance),
            max(float(boundary_width) * 0.5, 1.0),
        )
        smoothed = corridor.buffer(distance).buffer(-distance)

        if not smoothed.is_empty:
            corridor = smoothed

    if not corridor.is_valid:
        corridor = corridor.buffer(0)

    if corridor.is_empty:
        return None

    # Preserve separate polygons rather than discarding all but
    # the largest one.
    if corridor.geom_type == "GeometryCollection":
        polygons = [
            geom for geom in corridor.geoms
            if geom.geom_type == "Polygon" and not geom.is_empty
        ]
        if not polygons:
            return None
        corridor = unary_union(polygons)

    if corridor.geom_type not in ("Polygon", "MultiPolygon"):
        return None

    return corridor



# ============================================================
# 10. SCORE ROADS AND DETECT CORRIDORS
# ============================================================

def analyze(
    boundary,
    places,
    graph,
    edges,
    buffer_m,
    min_businesses,
    min_length,
    max_length,
    min_score,
    boundary_width,
    snap_tolerance_m=12,
):
    """Score active roads, group connected segments, and create corridors."""
    if edges.empty:
        raise ValueError("There are no road segments to analyze.")
    if "u" not in edges.columns or "v" not in edges.columns:
        raise ValueError("Road segments need u and v endpoint IDs.")

    metric_crs = estimate_metric_crs(boundary)
    places_m = safe_to_wgs84(places).to_crs(metric_crs).copy()
    roads_m = safe_to_wgs84(edges).to_crs(metric_crs).copy()
    places_m = clean_geometry(places_m)
    roads_m = clean_geometry(roads_m)
    places_m["geometry"] = places_m.geometry.representative_point()
    roads_m = roads_m.reset_index(drop=True)
    roads_m["u"] = roads_m["u"].astype(str)
    roads_m["v"] = roads_m["v"].astype(str)
    roads_m["road_id"] = roads_m.index
    roads_m["length_m"] = roads_m.geometry.length
    

    # Count places within the buffer of each road.
    road_buffers = roads_m[["road_id", "geometry"]].copy()
    road_buffers["geometry"] = road_buffers.geometry.buffer(buffer_m)

    place_columns = ["functional_group", "analysis_category", "geometry"]
    for column in place_columns:
        if column not in places_m.columns:
            places_m[column] = "unknown"

    nearby = gpd.sjoin(
        places_m[place_columns],
        road_buffers,
        how="inner",
        predicate="within",
    )
    total_counts = nearby.groupby("road_id").size()
    roads_m["nearby_businesses"] = (
        roads_m["road_id"].map(total_counts).fillna(0).astype(int)
    )
           # Business density per 100 metres of road
    roads_m["business_density"] = (
        roads_m["nearby_businesses"]
        / roads_m["length_m"].clip(lower=1)
    ) * 100
    group_count_columns = {}
    for group in GROUPS:
        column = "count_" + re.sub(
            r"[^a-z0-9]+", "_", group.lower()
        ).strip("_")
        group_count_columns[group] = column
        counts = (
            nearby.loc[nearby["functional_group"] == group]
            .groupby("road_id").size()
        )
        roads_m[column] = (
            roads_m["road_id"].map(counts).fillna(0).astype(int)
        )

    category_variety = nearby.groupby("road_id")["analysis_category"].nunique()
    roads_m["category_variety"] = (
        roads_m["road_id"].map(category_variety).fillna(0).astype(int)
    )

    activity_score = percentile(roads_m["nearby_businesses"])
    variety_score = percentile(roads_m["category_variety"])
    score_columns = {}

    for group, count_column in group_count_columns.items():
        density_score = percentile(roads_m[count_column])
        score_column = "score_" + re.sub(
            r"[^a-z0-9]+", "_", group.lower()
        ).strip("_")
        roads_m[score_column] = (
            0.60 * density_score
            + 0.20 * activity_score
            + 0.20 * variety_score
        )
        score_columns[group] = score_column

    candidate_rows = []
    boundary_rows = []
    next_candidate_id = 0

    
   
    # Select active road segments using activity and business counts
    any_active_mask = pd.Series(
        False,
        index=roads_m.index,
    )

    for group, count_column in group_count_columns.items():
        score_column = score_columns[group]

        active_mask = (
            (roads_m[score_column] >= min_score)
            & (roads_m[count_column] > 0)
            & (roads_m["nearby_businesses"] >= min_businesses)
        )

        any_active_mask |= active_mask

    active_roads = roads_m.loc[any_active_mask].copy()


    if active_roads.empty:
        roads_out = roads_m.drop(columns=["u", "v"], errors="ignore")
        empty = gpd.GeoDataFrame(
            columns=[
                "candidate_id", "functional_group",
                "corridor_type", "geometry",
            ],
            geometry="geometry",
            crs=metric_crs,
        ).to_crs("EPSG:4326")
        return (
            safe_to_wgs84(places),
            safe_to_wgs84(roads_out),
            empty.copy(),
            empty.copy(),
        )

    components = build_road_components(
        active_roads,
        snap_tolerance_m=snap_tolerance_m,
    )
    for component_ids in components:
        component = active_roads.loc[
            active_roads.index.isin(component_ids)
        ].copy()
        if component.empty or len(component) < 2:
            continue

        total_length = float(component["length_m"].sum())
        if not (min_length <= total_length <= max_length):
            continue

        group_counts = {
            group: int(component[count_column].sum())
            for group, count_column in group_count_columns.items()
        }
        total_group_businesses = sum(group_counts.values())
        if total_group_businesses == 0:
            continue

        dominant_group = max(group_counts, key=group_counts.get)
        dominant_share = group_counts[dominant_group] / total_group_businesses
        corridor_type = (
            dominant_group if dominant_share >= 0.60 else "Mixed-use"
        )
        merged_geometry = unary_union(list(component.geometry))
        dominant_score_column = score_columns[dominant_group]
        average_score = float(component[dominant_score_column].mean())
        nearby_businesses_sum = int(component["nearby_businesses"].sum())

        next_candidate_id += 1
        candidate_id = f"L{next_candidate_id}"

        candidate_rows.append({
            "candidate_id": candidate_id,
            "functional_group": dominant_group,
            "corridor_type": corridor_type,
            "group_share": dominant_share,
            "segment_count": len(component),
            "length_m": total_length,
            "average_group_score": average_score,
            "nearby_businesses_sum": nearby_businesses_sum,
            "group_business_count_sum": group_counts[dominant_group],
            "geometry": merged_geometry,
        })

        
        # Create a boundary using connected roads and nearby businesses.
        component_roads = list(component.geometry)

        component_roads_buffer = unary_union(
            component_roads
        ).buffer(max(float(buffer_m), 1.0))

        component_businesses = places_m.loc[
            places_m.geometry.intersects(component_roads_buffer)
        ].geometry.tolist()

        boundary_geometry = make_corridor_boundary(
            road_geometries=component_roads,
            nearby_business_points=component_businesses,
            boundary_width=boundary_width,
            business_buffer=12,
            smoothing_distance=5,
        )

        
        if boundary_geometry is not None and not boundary_geometry.is_empty:
            boundary_rows.append({
                "candidate_id": candidate_id,
                "functional_group": dominant_group,
                "corridor_type": corridor_type,
                "geometry": boundary_geometry,
            })

    roads_out = roads_m.drop(columns=["u", "v"], errors="ignore")
    corridors = gpd.GeoDataFrame(
        candidate_rows, geometry="geometry", crs=metric_crs
    )
    boundaries = gpd.GeoDataFrame(
        boundary_rows, geometry="geometry", crs=metric_crs
    )

    if corridors.empty:
        corridors = gpd.GeoDataFrame(
            columns=[
                "candidate_id", "functional_group",
                "corridor_type", "geometry",
            ],
            geometry="geometry",
            crs=metric_crs,
        )
    if boundaries.empty:
        boundaries = gpd.GeoDataFrame(
            columns=[
                "candidate_id", "functional_group",
                "corridor_type", "geometry",
            ],
            geometry="geometry",
            crs=metric_crs,
        )

    return (
        safe_to_wgs84(places),
        safe_to_wgs84(roads_out),
        safe_to_wgs84(corridors),
        safe_to_wgs84(boundaries),
    )


# ============================================================
# 11. CLASSIFY CORRIDORS USING NEARBY BUSINESSES
# ============================================================

def enrich_corridor_classification(corridors, places, buffer_m=45):
    """Count businesses near corridor centerlines and assign a group."""
    corridors = safe_to_wgs84(corridors).copy()
    places = safe_to_wgs84(places).copy()
    if corridors.empty:
        return corridors

    if "functional_group" not in places.columns:
        places["functional_group"] = places.apply(classify, axis=1)

    metric_crs = estimate_metric_crs(corridors)
    corridors_m = corridors.to_crs(metric_crs).copy()
    places_m = places.to_crs(metric_crs).copy()
    places_m["geometry"] = places_m.geometry.representative_point()

    corridor_buffers = corridors_m[["geometry"]].copy()
    corridor_buffers["corridor_row"] = corridor_buffers.index
    corridor_buffers["geometry"] = corridor_buffers.geometry.buffer(buffer_m)

    joined = gpd.sjoin(
        places_m[["functional_group", "geometry"]],
        corridor_buffers[["corridor_row", "geometry"]],
        how="inner",
        predicate="within",
    )

    counts, groups, types, shares = [], [], [], []
    valid_groups = set(GROUPS)

    for idx, row in corridors_m.iterrows():
        nearby_rows = joined[joined["corridor_row"] == idx]
        counts.append(int(len(nearby_rows)))
        group_counts = (
            nearby_rows.loc[
                nearby_rows["functional_group"].isin(valid_groups),
                "functional_group",
            ].value_counts()
        )
        if group_counts.empty:
            existing_group = str(row.get("functional_group", ""))
            if existing_group not in valid_groups:
                existing_group = "Other"
            existing_type = str(row.get("corridor_type", "")) or existing_group
            groups.append(existing_group)
            types.append(existing_type)
            shares.append(safe_number(row.get("group_share", 0)))
        else:
            dominant = str(group_counts.index[0])
            share = float(group_counts.iloc[0] / group_counts.sum())
            groups.append(dominant)
            types.append(dominant if share >= 0.60 else "Mixed-use")
            shares.append(share)

    corridors["business_count"] = counts
    corridors["functional_group"] = groups
    corridors["corridor_type"] = types
    corridors["classification_share"] = shares

    if "average_group_score" in corridors.columns:
        corridors["corridor_score"] = pd.to_numeric(
            corridors["average_group_score"], errors="coerce"
        )
    elif "score" in corridors.columns:
        corridors["corridor_score"] = pd.to_numeric(
            corridors["score"], errors="coerce"
        )
    else:
        corridors["corridor_score"] = pd.NA

    return corridors


def enrich_boundaries(boundaries, corridors):
    """Copy corridor classification fields onto matching polygons."""
    boundaries = safe_to_wgs84(boundaries).copy()
    if boundaries.empty or corridors.empty:
        return boundaries
    if (
        "candidate_id" not in boundaries.columns
        or "candidate_id" not in corridors.columns
    ):
        return boundaries

    fields = [
        "candidate_id", "corridor_type", "functional_group",
        "business_count", "corridor_score", "classification_share",
        "length_m",
    ]
    fields = [column for column in fields if column in corridors.columns]
    lookup = (
        corridors[fields].drop_duplicates("candidate_id")
        .set_index("candidate_id")
    )
    for column in fields:
        if column != "candidate_id":
            boundaries[column] = boundaries["candidate_id"].map(lookup[column])
    return boundaries


# ============================================================
# 12. MAP HELPERS
# ============================================================

# ============================================================
# BUSINESS CATEGORY CHARTS
# ============================================================

def render_business_charts(area_data):
    """Display business category counts for the selected neighbourhood."""

    places = area_data["places"].copy()
    
    category_columns = [
        "basic_category",
        "taxonomy",
        "primary_category",
        "category",
        "analysis_category",
        "functional_group",
    ]


    if places.empty:
        st.warning("No business data is available for this neighbourhood.")
        return

    # Make sure every place has a business category.
    if "functional_group" not in places.columns:
        places["functional_group"] = places.apply(classify, axis=1)

    places["functional_group"] = (
        places["functional_group"]
        .fillna("Other")
        .replace("", "Other")
    )

  
    # Count businesses in each category, excluding unclassified places.
    places = places[
        ~places["functional_group"].isin([
            "Other",
            "Other / Unclassified",
        ])
    ]

    if places.empty:
        st.warning("No classified business data is available.")
        return

    category_counts = (
        places["functional_group"]
        .value_counts()
        .rename_axis("Business category")
        .reset_index(name="Number of businesses")
    )


    total_businesses = len(places)

    top_category = (
        category_counts.iloc[0]["Business category"]
        if not category_counts.empty
        else "No categories found"
    )

    st.subheader("Business composition")

    st.caption(
        f"Based on {total_businesses:,} classified places in "
        f"{area_data['area']}. Unclassified places are excluded."
    )

    st.write("**Most common business category:**", top_category)
    # Summary metrics.
    col1, col2, col3 = st.columns(3)

    col1.metric("Total mapped places", f"{total_businesses:,}")
    col2.metric("Business categories", category_counts.shape[0])
    col3.metric("Largest category", top_category)

     # Bar chart.
    st.markdown("#### Businesses by category")
    st.bar_chart(
        category_counts.set_index("Business category")[
            "Number of businesses"
        ]
    )

    # Exact counts in a table.
    st.markdown("#### Category-wise counts")
    category_counts["Percentage"] = (
        category_counts["Number of businesses"]
        / total_businesses * 100
    ).round(1)

    st.dataframe(
    category_counts,
    hide_index=True,
    width="stretch",
)

def get_map_center(frame, fallback=(41.926, -87.7045)):
    if frame is None or frame.empty:
        return fallback
    points = frame.geometry.representative_point()
    if points.empty:
        return fallback
    return float(points.y.mean()), float(points.x.mean())


def fit_map_to_frame(map_object, frame):
    if frame is None or frame.empty:
        return
    minx, miny, maxx, maxy = frame.total_bounds
    map_object.fit_bounds([[miny, minx], [maxy, maxx]], padding=(15, 15))





def add_business_points(
    map_object,
    places,
    limit=5000,
    corridor_boundary=None,
    outside_limit=300,
):
    """Show category-coloured dots inside and fewer grey dots outside."""

    if places is None or places.empty:
        return

    places = places.copy()

    if "functional_group" not in places.columns:
        places["functional_group"] = places.apply(classify, axis=1)

    inside_places = []
    outside_places = []

    for _, row in places.iterrows():
        geom = row.geometry

        if geom is None or geom.is_empty:
            continue

        point = geom.representative_point()

        inside = (
            corridor_boundary.covers(point)
            if corridor_boundary is not None
            else True
        )

        if inside:
            inside_places.append((row, point))
        else:
            outside_places.append((row, point))

    if outside_limit is not None and len(outside_places) > outside_limit:
        step = len(outside_places) / outside_limit
        outside_places = [
            outside_places[int(i * step)]
            for i in range(outside_limit)
        ]

    for row, point in inside_places + outside_places:
        group = str(row.get("functional_group", "Other"))
        name = row.get("name", "Unnamed place")

        inside = (
            corridor_boundary.covers(point)
            if corridor_boundary is not None
            else True
        )

        if inside:
            color = GROUP_COLORS.get(group, GROUP_COLORS["Other"])
            opacity, radius = 0.9, 4
            label = f"INSIDE corridor — {name} — {group}"
        else:
            color = "#a6a6a6"
            opacity, radius = 0.45, 3
            label = f"OUTSIDE corridor — {name} — {group}"

        folium.CircleMarker(
            location=[point.y, point.x],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=opacity,
            weight=0.5,
            tooltip=label,
        ).add_to(map_object)


def add_corridor_layers(map_object, corridors, boundaries, line_color):
    """Add corridor boundaries and centerlines to a corridor-only map."""
    if boundaries is not None and not boundaries.empty:
        fields = [
            field for field in [
                "candidate_id", "corridor_type", "functional_group",
                "business_count", "corridor_score", "length_m",
            ] if field in boundaries.columns
        ]
        aliases = {
            "candidate_id": "Candidate",
            "corridor_type": "Corridor classification",
            "functional_group": "Dominant business group",
            "business_count": "Nearby businesses",
            "corridor_score": "Corridor score",
            "length_m": "Length (m)",
        }
        folium.GeoJson(
            boundaries,
            name="Corridor boundaries",
            style_function=lambda feature: {
                "color": "#ff3b30",
                "weight": 3,
                "fillColor": "#ff6b6b",
                "fillOpacity": 0.18,
            },
            tooltip=(
                folium.GeoJsonTooltip(
                    fields=fields,
                    aliases=[aliases[field] for field in fields],
                ) if fields else None
            ),
        ).add_to(map_object)

    if corridors is not None and not corridors.empty:
        fields = [
            field for field in [
                "candidate_id", "corridor_type", "functional_group",
                "business_count", "corridor_score", "length_m",
                "classification_share",
            ] if field in corridors.columns
        ]
        aliases = {
            "candidate_id": "Candidate",
            "corridor_type": "Corridor classification",
            "functional_group": "Dominant business group",
            "business_count": "Nearby businesses",
            "corridor_score": "Corridor score",
            "length_m": "Length (m)",
            "classification_share": "Dominant-group share",
        }
        folium.GeoJson(
            corridors,
            name="Corridor centerlines",
            style_function=lambda feature: {
                "color": line_color, "weight": 5, "opacity": 0.95
            },
            tooltip=(
                folium.GeoJsonTooltip(
                    fields=fields,
                    aliases=[aliases[field] for field in fields],
                ) if fields else None
            ),
        ).add_to(map_object)

    folium.LayerControl(collapsed=False).add_to(map_object)


def render_original_map(data, places, roads, buildings):
    """Render raw geography without corridor overlays."""
    center = get_map_center(places)
    map_object = folium.Map(
        location=center, zoom_start=15, tiles="OpenStreetMap",
        control_scale=True,
    )
    fit_map_to_frame(
        map_object,
        roads if roads is not None and not roads.empty else places,
    )
    if roads is not None and not roads.empty:
        folium.GeoJson(
            roads,
            name="Road segments",
            style_function=lambda feature: {
                "color": "#aeb6bf", "weight": 1, "opacity": 0.45
            },
        ).add_to(map_object)
    if buildings is not None and not buildings.empty:
        folium.GeoJson(
            buildings.head(12000),
            name="Building footprints",
            style_function=lambda feature: {
                "color": "#616a6b", "weight": 0.4,
                "fillColor": "#b2bec3", "fillOpacity": 0.22,
            },
        ).add_to(map_object)
    add_business_points(map_object, places)
    folium.LayerControl(collapsed=False).add_to(map_object)
    st_folium(
        map_object,
        key=("original_map_logan" if data["area"] == LOGAN_AREA else "original_map_wicker"),
        use_container_width=True,
        height=600,
    )


def render_corridor_map(area, corridors, boundaries):
    """Render only detected corridors and boundaries."""
    source = (
        corridors if corridors is not None and not corridors.empty
        else boundaries
    )
    if source is None or source.empty:
        st.warning(
            "No corridor geometries are available. "
            "Try changing the analysis thresholds."
        )
        return
    map_object = folium.Map(
        location=get_map_center(source), zoom_start=14,
        tiles="OpenStreetMap", control_scale=True,
    )
    fit_map_to_frame(map_object, source)
    add_corridor_layers(
        map_object, corridors, boundaries,
        CORRIDOR_COLORS.get(area, "#1769aa"),
    )
    st_folium(
        map_object,
        key=("corridor_map_logan" if area == LOGAN_AREA else "corridor_map_wicker"),
        use_container_width=True,
        height=600,
    )


# ============================================================
# 13. SIDEBAR
# ============================================================


with st.sidebar:
    st.header("Study settings")

    area_label = st.selectbox(
        "Chicago neighborhood",
        AREAS,
    )

    # Fixed settings (hidden from users)
    buffer_m = 50
    min_businesses = 3
    min_length = 200
    max_length = 1500
    min_score = 0.50
    boundary_width = 30
    snap_tolerance_m = 12

    run = st.button(
        "Load / analyze selected neighborhood",
        type="primary",
        use_container_width=True,
    )


if max_length <= min_length:
    st.error("Maximum corridor length must be greater than minimum length.")
    st.stop()


# ============================================================
# 14. LOAD / ANALYZE SELECTED AREA
# ============================================================

if "wicker_analysis" not in st.session_state and area_label == ORIGINAL_AREA:
    try:
        st.session_state["wicker_analysis"] = load_precomputed_wicker_results()
    except Exception as exc:
        st.error("Could not load the saved Wicker Park results.")
        st.exception(exc)

if run:
    try:
        if area_label == ORIGINAL_AREA:
            with st.spinner("Loading saved Wicker Park–Bucktown results..."):
                st.session_state["wicker_analysis"] = load_precomputed_wicker_results()
            st.session_state["analysis"] = st.session_state["wicker_analysis"]

        elif area_label == LOGAN_AREA:
            with st.spinner("Reading Logan Square data and detecting corridors..."):
                boundary, places, graph, edges, buildings = get_overture_logan_data()
                places_out, roads_out, corridors, boundaries = analyze(
                    boundary=boundary,
                    places=places,
                    graph=graph,
                    edges=edges,
                    buffer_m=buffer_m,
                    min_businesses=min_businesses,
                    min_length=min_length,
                    max_length=max_length,
                    min_score=min_score,
                    boundary_width=boundary_width,
                    snap_tolerance_m=snap_tolerance_m,
                )
                st.session_state["logan_analysis"] = {
                    "area": LOGAN_AREA,
                    "boundary": boundary,
                    "places": places_out,
                    "roads": roads_out,
                    "corridors": corridors,
                    "boundaries": boundaries,
                    "buildings": buildings,
                }
            st.session_state["analysis"] = st.session_state["logan_analysis"]
            
        elif area_label == PILSEN_AREA:
            with st.spinner("Reading Pilsen data and detecting corridors..."):
                boundary, places, graph, edges, buildings = (
                    get_overture_logan_data(PILSEN_AREA)
                )
                places_out, roads_out, corridors, boundaries = analyze(
                    boundary=boundary,
                    places=places,
                    graph=graph,
                    edges=edges,
                    buffer_m=buffer_m,
                    min_businesses=min_businesses,
                    min_length=min_length,
                    max_length=max_length,
                    min_score=min_score,
                    boundary_width=boundary_width,
                    snap_tolerance_m=snap_tolerance_m,
                )
                st.session_state["pilsen_analysis"] = {
                    "area": PILSEN_AREA,
                    "boundary": boundary,
                    "places": places_out,
                    "roads": roads_out,
                    "corridors": corridors,
                    "boundaries": boundaries,
                    "buildings": buildings,
                }
            st.session_state["analysis"] = st.session_state["pilsen_analysis"]
            
        elif area_label == CHINATOWN_AREA:
            with st.spinner("Reading Chinatown data and detecting corridors..."):
                boundary, places, graph, edges, buildings = (
                    get_overture_logan_data(CHINATOWN_AREA)
                )

                places_out, roads_out, corridors, boundaries = analyze(
                    boundary=boundary,
                    places=places,
                    graph=graph,
                    edges=edges,
                    buffer_m=buffer_m,
                    min_businesses=min_businesses,
                    min_length=min_length,
                    max_length=max_length,
                    min_score=min_score,
                    boundary_width=boundary_width,
                    snap_tolerance_m=snap_tolerance_m,
                )

                st.session_state["chinatown_analysis"] = {
                    "area": CHINATOWN_AREA,
                    "boundary": boundary,
                    "places": places_out,
                    "roads": roads_out,
                    "corridors": corridors,
                    "boundaries": boundaries,
                    "buildings": buildings,
                }

            st.session_state["analysis"] = (
                st.session_state["chinatown_analysis"]
            )



    except Exception as exc:
        st.error(
            "Loading or analysis failed. Your original saved files "
            "have not been overwritten."
        )
        st.exception(exc)


# ============================================================
# 15. SELECT ONE PRINCIPAL CORRIDOR
# ============================================================

def choose_principal_corridor(corridors, boundaries):
    """Choose the strongest candidate and retain its matching polygon."""
    corridors = safe_to_wgs84(corridors).copy()
    boundaries = safe_to_wgs84(boundaries).copy()

    if corridors is None or corridors.empty:
        if boundaries is None:
            return corridors, boundaries
        return corridors, boundaries.iloc[0:0].copy()

    for col in [
        "business_count", "corridor_score",
        "average_group_score", "length_m",
    ]:
        if col not in corridors.columns:
            corridors[col] = 0
        corridors[col] = pd.to_numeric(
            corridors[col], errors="coerce"
        ).fillna(0)

    ranked = corridors.sort_values(
        ["business_count", "corridor_score", "average_group_score", "length_m"],
        ascending=False,
    )
    chosen = ranked.head(1).copy()

    if (
        boundaries is None or boundaries.empty
        or "candidate_id" not in boundaries.columns
        or "candidate_id" not in chosen.columns
    ):
        selected_boundaries = (
            boundaries.iloc[0:0].copy()
            if boundaries is not None
            else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        )
    else:
        selected_id = chosen.iloc[0]["candidate_id"]
        selected_boundaries = boundaries[
            boundaries["candidate_id"] == selected_id
        ].copy()

        if not selected_boundaries.empty:
            merged = unary_union(list(selected_boundaries.geometry))
            # Reduce a MultiPolygon to one polygon, if needed.
            if merged.geom_type == "MultiPolygon":
                merged = max(merged.geoms, key=lambda polygon: polygon.area)
            selected_boundaries = gpd.GeoDataFrame(
                [{"candidate_id": selected_id, "geometry": merged}],
                geometry="geometry",
                crs="EPSG:4326",
            )

    return chosen, selected_boundaries


# ============================================================
# 16. RENDER BOTH NEIGHBORHOOD MAPS
# ============================================================

def render_neighborhood_map(area_data, map_key, line_color):
    """
    Show muted roads, category-coloured places, one selected corridor,
    and a closed boundary. Dots outside the boundary are grey.
    """
    if area_data is None:
        st.info(
            "Load this neighbourhood using the sidebar, then both maps "
            "will appear here."
        )
        return

    places = safe_to_wgs84(area_data["places"])
    roads = safe_to_wgs84(area_data["roads"])
    all_corridors = enrich_corridor_classification(
        area_data["corridors"], places
    )
    all_boundaries = enrich_boundaries(
        area_data["boundaries"], all_corridors
    )
    corridor, boundary = choose_principal_corridor(
        all_corridors, all_boundaries
    )

    map_source = (
        corridor
        if corridor is not None and not corridor.empty
        else places
    )
    m = folium.Map(
        location=get_map_center(map_source),
        zoom_start=15,
        tiles="OpenStreetMap",
        control_scale=True,
    )
    fit_map_to_frame(m, map_source)

    if roads is not None and not roads.empty:
        folium.GeoJson(
            roads,
            name="Road network",
            style_function=lambda feature: {
                "color": "#aeb6bf", "weight": 1, "opacity": 0.45
            },
        ).add_to(m)

    selected_boundary_geometry = None
    if boundary is not None and not boundary.empty:
        valid_geometries = [
            geom for geom in boundary.geometry
            if geom is not None and not geom.is_empty
        ]
        if valid_geometries:
            selected_boundary_geometry = unary_union(valid_geometries)

            if selected_boundary_geometry.geom_type == "MultiPolygon":
                selected_boundary_geometry = max(
                    selected_boundary_geometry.geoms,
                    key=lambda polygon: polygon.area,
                )

    # Place dots inside the boundary retain category colours;
    # dots outside the boundary are shown in grey.
    add_business_points(
        m,
        places,
        limit=None,
        corridor_boundary=selected_boundary_geometry,
         outside_limit=300,
    )

    if boundary is not None and not boundary.empty:
        boundary_fields = [
            c for c in [
                "candidate_id", "corridor_type", "functional_group"
            ] if c in boundary.columns
        ]
        folium.GeoJson(
            boundary,
            name="Selected corridor boundary",
            style_function=lambda feature: {
                "color": line_color,
                "weight": 3,
                "fillColor": line_color,
                "fillOpacity": 0.12,
            },
            tooltip=(
                folium.GeoJsonTooltip(fields=boundary_fields)
                if boundary_fields else None
            ),
        ).add_to(m)

    if corridor is not None and not corridor.empty:
        corridor_fields = [
            c for c in [
                "candidate_id", "corridor_type", "functional_group",
                "business_count", "length_m",
            ] if c in corridor.columns
        ]
        folium.GeoJson(
            corridor,
            name="Selected corridor centreline",
            style_function=lambda feature: {
                "color": line_color, "weight": 6, "opacity": 1
            },
            tooltip=(
                folium.GeoJsonTooltip(fields=corridor_fields)
                if corridor_fields else None
            ),
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    st_folium(
    m,
    key=map_key,
    use_container_width=True,
    height=500,
    returned_objects=[],
)

    if corridor is None or corridor.empty:
        st.warning(
            "No corridor candidates were found. Try lowering the score "
            "or nearby-business threshold in the sidebar."
        )
    else:
        row = corridor.iloc[0]
        st.caption(
            f"Selected candidate: {row.get('candidate_id', 'N/A')} · "
            f"{row.get('functional_group', 'Unclassified')} · "
            f"length {float(row.get('length_m', 0)):.0f} m. "
            "The polygon is an inferred study boundary, not an official district boundary."
        )


st.divider()
st.header("Three Chicago neighbourhood corridor maps")
st.write(
    "Each map displays one principal connected corridor, its closed "
    "boundary, and category-coloured business locations. Places outside "
    "the selected boundary are shown in grey."
)


wicker_data = st.session_state.get("wicker_analysis")
logan_data = st.session_state.get("logan_analysis")
pilsen_data = st.session_state.get("pilsen_analysis")
chinatown_data = st.session_state.get("chinatown_analysis")




if wicker_data is None:
    try:
        wicker_data = load_precomputed_wicker_results()
        st.session_state["wicker_analysis"] = wicker_data
    except Exception as exc:
        st.warning(
            "Wicker Park–Bucktown saved results could not be loaded. "
            "Check that the original data folder is available."
        )
        st.caption(str(exc))


# ============================================================
# 16. DISPLAY THE SELECTED NEIGHBORHOOD MAP
# ============================================================

st.divider()
st.header("Chicago Commercial Corridor Explorer")



# ============================================================
# NEIGHBOURHOOD COMPARISON TABLE
# ============================================================

st.divider()
st.header("Neighbourhood Comparison")

comparison_data = [
    (ORIGINAL_AREA, wicker_data),
    (LOGAN_AREA, logan_data),
    (PILSEN_AREA, pilsen_data),
    (CHINATOWN_AREA, chinatown_data),
]

comparison_rows = []

for area_name, data in comparison_data:
    if data is None:
        comparison_rows.append({
            "Neighbourhood": area_name,
            "Road network length (km)": "Not loaded",
            "Business categories": "Not loaded",
        })
        continue

    # Calculate total road length in kilometres.
    roads = data.get("roads")
    if roads is not None and not roads.empty:
        roads_m = roads.to_crs(estimate_metric_crs(roads))
        road_length_km = round(
            roads_m.geometry.length.sum() / 1000, 2
        )
    else:
        road_length_km = None

    # Count businesses in each category.
    places = data["places"].copy()

    if "functional_group" not in places.columns:
        places["functional_group"] = places.apply(classify, axis=1)

    places["functional_group"] = (
        places["functional_group"].fillna("Other / Unclassified")
    )

    categories = sorted(
        places.loc[
            ~places["functional_group"].isin([
                "Other",
                "Other / Unclassified",
                "",
            ]),
            "functional_group",
        ].astype(str).unique()
    )

    comparison_rows.append({
        "Neighbourhood": area_name,
        "Road network length (km)": (
            road_length_km if road_length_km is not None else "Unavailable"
        ),
        "Business categories": (
            ", ".join(categories) if categories else "None classified"
        ),
    })


# ============================================================
# NEIGHBOURHOOD COMPARISON TABLE
# ============================================================

st.divider()
st.header("Neighbourhood Business Comparison")

comparison_data = [
    (ORIGINAL_AREA, wicker_data),
    (LOGAN_AREA, logan_data),
    (PILSEN_AREA, pilsen_data),
    (CHINATOWN_AREA, chinatown_data),
]

comparison_rows = []

for area_name, data in comparison_data:

    # Handle neighbourhoods whose data has not been loaded.
    if data is None:
        comparison_rows.append({
            "Neighbourhood": area_name,
            "Total businesses": "Not loaded",
            "Highest business category": "Not loaded",
            "Businesses in highest category": "Not loaded",
        })
        continue

    # Get the business dataset.
    places = data.get("places")

    if places is None or places.empty:
        comparison_rows.append({
            "Neighbourhood": area_name,
            "Total businesses": 0,
            "Highest business category": "No data",
            "Businesses in highest category": 0,
        })
        continue

    places = places.copy()

    # Classify businesses if the category column does not exist.
    if "functional_group" not in places.columns:
        places["functional_group"] = places.apply(classify, axis=1)

    places["functional_group"] = (
        places["functional_group"]
        .fillna("Other / Unclassified")
        .astype(str)
        .str.strip()
    )

    # Exclude unclassified businesses when finding the top category.
    valid_places = places[
        ~places["functional_group"].isin([
            "Other",
            "Other / Unclassified",
            "",
        ])
    ]

    # Count businesses in each category.
    category_counts = valid_places["functional_group"].value_counts()

    total_businesses = len(places)

    # Find the category with the highest number of businesses.
    if not category_counts.empty:
        highest_category = category_counts.index[0]
        highest_count = int(category_counts.iloc[0])
    else:
        highest_category = "No categories found"
        highest_count = 0

    # Add the neighbourhood's results to the table.
    comparison_rows.append({
        "Neighbourhood": area_name,
        "Total businesses": total_businesses,
        "Highest business category": highest_category,
        "Businesses in highest category": highest_count,
    })

# Convert results into a DataFrame.
comparison_table = pd.DataFrame(comparison_rows)

# Display the comparison table.
st.dataframe(
    comparison_table,
    hide_index=True,
    use_container_width=True,
)

st.caption(
    "Total businesses counts all records in each loaded place dataset. "
    "The highest business category is the most frequent classified "
    "functional group. Unloaded neighbourhoods will show 'Not loaded'."
)
selected_area_data = {
    ORIGINAL_AREA: wicker_data,
    LOGAN_AREA: logan_data,
    PILSEN_AREA: pilsen_data,
    CHINATOWN_AREA: chinatown_data,
}.get(area_label)


if selected_area_data is None:
    st.info(
        f"{area_label} has not been loaded yet. "
        "Choose the neighborhood in the sidebar and click "
        "'Load / analyze selected neighborhood'."
    )
else:
    st.subheader(area_label)

    map_key = {
        ORIGINAL_AREA: "principal_wicker_map",
        LOGAN_AREA: "principal_logan_map",
        PILSEN_AREA: "principal_pilsen_map",
         CHINATOWN_AREA: "principal_chinatown_map",
    }[area_label]
    render_neighborhood_map(
        selected_area_data,
        map_key,
        CORRIDOR_COLORS[area_label],
    )

    st.divider()
    render_business_charts(selected_area_data)





# ============================================================
# 17. DOWNLOADS
# ============================================================


download_areas = [
    (wicker_data, "wicker_park_bucktown"),
    (logan_data, "logan_square"),
    (pilsen_data, "pilsen"),
    (chinatown_data, "chinatown"),
]


for area_data, slug in download_areas:
    if area_data is None:
        continue

    with st.expander(f"Downloads — {area_data['area']}"):
        # Download all business/place data for this neighbourhood.
        st.download_button(
            "Download business/place data (GeoJSON)",
            data=places_to_geojson(area_data["places"]),
            file_name=f"{slug}_places.geojson",
            mime="application/geo+json",
            key=f"{slug}_places_download",
        )

        # Prepare corridor classifications for downloading.
        all_corridors = enrich_corridor_classification(
            area_data["corridors"],
            area_data["places"],
        )

        all_boundaries = enrich_boundaries(
            area_data["boundaries"],
            all_corridors,
        )

        selected, selected_boundary = choose_principal_corridor(
            all_corridors,
            all_boundaries,
        )

        if selected is not None and not selected.empty:
            st.download_button(
                "Download selected corridor (GeoJSON)",
                data=selected.to_json(),
                file_name=f"{slug}_selected_corridor.geojson",
                mime="application/geo+json",
                key=f"{slug}_corridor_download",
            )

        if (
            selected_boundary is not None
            and not selected_boundary.empty
        ):
            st.download_button(
                "Download selected corridor boundary (GeoJSON)",
                data=selected_boundary.to_json(),
                file_name=f"{slug}_corridor_boundary.geojson",
                mime="application/geo+json",
                key=f"{slug}_boundary_download",
            )
