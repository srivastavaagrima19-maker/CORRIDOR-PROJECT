
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box, Point
from pathlib import Path

# Project data folder
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# Your selected Chicago study-area boundary
west = -87.683
south = 41.903
east = -87.659
north = 41.913

# Create the rectangular boundary
boundary = gpd.GeoDataFrame(
    {"name": ["Selected study area"]},
    geometry=[box(west, south, east, north)],
    crs="EPSG:4326"
)

# Load existing road data, if available
roads_file = DATA_DIR / "roads_in_study_area.geojson"
roads = gpd.read_file(roads_file) if roads_file.exists() else None

# Project to metres for plotting
boundary = boundary.to_crs("EPSG:32616")

if roads is not None:
    if roads.crs is None:
        roads = roads.set_crs("EPSG:4326")
    roads = roads.to_crs("EPSG:32616")
    roads = gpd.clip(roads, boundary)

# Get the four corners
corners = {
    "SW": (west, south),
    "SE": (east, south),
    "NW": (west, north),
    "NE": (east, north)
}

fig, ax = plt.subplots(figsize=(12, 9))

# Plot roads within the boundary
if roads is not None and not roads.empty:
    roads.plot(
        ax=ax,
        color="gray",
        linewidth=0.7,
        label="Road network"
    )

# Plot the boundary
boundary.boundary.plot(
    ax=ax,
    color="red",
    linewidth=3,
    label="Selected boundary"
)

# Mark and label all four corners
for label, (lon, lat) in corners.items():
    point = gpd.GeoSeries(
        [Point(lon, lat)], crs="EPSG:4326"
    ).to_crs("EPSG:32616")

    x, y = point.iloc[0].x, point.iloc[0].y

    ax.scatter(x, y, color="blue", s=45, zorder=5)
    ax.annotate(
        f"{label}\n({lon}, {lat})",
        (x, y),
        xytext=(6, 6),
        textcoords="offset points",
        fontsize=8
    )

ax.set_title(
    "Selected Study Area Boundary — Downtown Chicago",
    fontsize=15,
    fontweight="bold"
)
ax.set_xlabel("Easting (metres)")
ax.set_ylabel("Northing (metres)")
ax.set_aspect("equal")
ax.grid(alpha=0.25)
ax.legend()
plt.tight_layout()

# Save visualization
output = DATA_DIR / "selected_area_boundary.png"
plt.savefig(output, dpi=300, bbox_inches="tight")
plt.show()

print(f"Boundary visualization saved to: {output}")
