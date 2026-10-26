
import geopandas as gpd
import folium
from pathlib import Path

DATA = Path(__file__).resolve().parent

# Load the real street network and discovered corridor boundary
roads = gpd.read_file(DATA / "roads_in_study_area.geojson")
boundaries = gpd.read_file(DATA / "functional_corridor_boundaries.geojson")
candidates = gpd.read_file(DATA / "functional_corridor_candidates.geojson")

# Convert all layers to latitude/longitude for the map
roads = roads.to_crs(epsg=4326)
boundaries = boundaries.to_crs(epsg=4326)
candidates = candidates.to_crs(epsg=4326)

# Center the map on the study area
center = [
    41.908,
    -87.671
]

m = folium.Map(
    location=center,
    zoom_start=15,
    tiles="OpenStreetMap"
)

# Real streets
folium.GeoJson(
    roads,
    name="Real street network",
    style_function=lambda feature: {
        "color": "#555555",
        "weight": 2,
        "opacity": 0.7
    }
).add_to(m)

# Detected corridor boundary
folium.GeoJson(
    boundaries,
    name="Detected corridor boundaries",
    style_function=lambda feature: {
        "color": "#e63946",
        "weight": 3,
        "fillColor": "#ff6b6b",
        "fillOpacity": 0.25
    },
    tooltip=folium.GeoJsonTooltip(
        fields=[
            field for field in
            ["candidate_id", "functional_group", "corridor_type"]
            if field in boundaries.columns
        ]
    )
).add_to(m)

# Detected corridor centerlines
folium.GeoJson(
    candidates,
    name="Detected corridor centerlines",
    style_function=lambda feature: {
        "color": "#1565c0",
        "weight": 5,
        "opacity": 0.9
    },
    tooltip=folium.GeoJsonTooltip(
        fields=[
            field for field in
            ["candidate_id", "functional_group", "length_m"]
            if field in candidates.columns
        ]
    )
).add_to(m)

folium.LayerControl().add_to(m)

output = DATA / "corridor_real_map_comparison.html"
m.save(str(output))

print("Map saved to:", output)
