
import geopandas as gpd
import folium
from folium.plugins import MousePosition

# Select the corridor to display
SELECTED_CANDIDATE_ID = 1

# File paths
DATA = "data/"

roads = gpd.read_file(DATA + "roads_with_business_scores.geojson")
places = gpd.read_file(DATA + "business_clusters.geojson")
corridors = gpd.read_file(DATA + "connected_corridor_candidates.geojson")
boundaries = gpd.read_file(DATA + "connected_corridor_boundaries.geojson")

# Convert all layers to latitude/longitude coordinates
roads = roads.to_crs("EPSG:4326")
places = places.to_crs("EPSG:4326")
corridors = corridors.to_crs("EPSG:4326")
boundaries = boundaries.to_crs("EPSG:4326")

# Select Candidate 1
corridor = corridors[
    corridors["candidate_id"] == SELECTED_CANDIDATE_ID
]

boundary = boundaries[
    boundaries["candidate_id"] == SELECTED_CANDIDATE_ID
]

if corridor.empty or boundary.empty:
    raise ValueError("Selected corridor or its boundary was not found.")

corridor_geometry = corridor.geometry.iloc[0]
boundary_geometry = boundary.geometry.iloc[0]

# Center the map on the selected corridor
center = corridor_geometry.centroid

m = folium.Map(
    location=[center.y, center.x],
    zoom_start=16,
    tiles="OpenStreetMap"
)

# Show the road network
folium.GeoJson(
    roads,
    name="Road network",
    style_function=lambda feature: {
        "color": "#999999",
        "weight": 1,
        "opacity": 0.5
    }
).add_to(m)

# Show the corridor boundary
folium.GeoJson(
    boundary,
    name="Corridor boundary",
    style_function=lambda feature: {
        "color": "orange",
        "weight": 3,
        "fillColor": "orange",
        "fillOpacity": 0.18
    },
    tooltip="Candidate 1 boundary"
).add_to(m)

# Show the corridor centerline
folium.GeoJson(
    corridor,
    name="Corridor centerline",
    style_function=lambda feature: {
        "color": "red",
        "weight": 6,
        "opacity": 0.9
    },
    tooltip="Candidate 1 centerline"
).add_to(m)

# Find businesses located inside the corridor boundary
inside = places[
    places.geometry.within(boundary_geometry)
].copy()

# Display business points and their coordinates
for _, row in inside.iterrows():
    point = row.geometry

    if point is None or point.is_empty:
        continue

    latitude = point.y
    longitude = point.x

    category = row.get("analysis_category", "Unknown category")
    cluster = row.get("category_cluster", "Unassigned")

    popup_text = f"""
    <b>Business category:</b> {category}<br>
    <b>Category cluster:</b> {cluster}<br>
    <b>Latitude:</b> {latitude:.6f}<br>
    <b>Longitude:</b> {longitude:.6f}
    """

    folium.CircleMarker(
        location=[latitude, longitude],
        radius=4,
        color="blue",
        fill=True,
        fill_opacity=0.8,
        popup=folium.Popup(popup_text, max_width=300),
        tooltip=f"{category} | {latitude:.5f}, {longitude:.5f}"
    ).add_to(m)

# Show the corridor's approximate center coordinates
folium.Marker(
    location=[center.y, center.x],
    tooltip="Approximate corridor center",
    popup=(
        f"<b>Corridor center</b><br>"
        f"Latitude: {center.y:.6f}<br>"
        f"Longitude: {center.x:.6f}"
    ),
    icon=folium.Icon(color="red", icon="info-sign")
).add_to(m)

# Show live coordinates when moving the mouse over the map
MousePosition(
    position="bottomright",
    separator=" | ",
    prefix="Latitude / Longitude:",
    num_digits=6
).add_to(m)

folium.LayerControl().add_to(m)

# Save the interactive map
output_file = DATA + "interactive_corridor_map.html"
m.save(output_file)

print("Interactive map saved to:", output_file)
print("Businesses inside corridor boundary:", len(inside))
print(f"Corridor center latitude: {center.y:.6f}")
print(f"Corridor center longitude: {center.x:.6f}")
