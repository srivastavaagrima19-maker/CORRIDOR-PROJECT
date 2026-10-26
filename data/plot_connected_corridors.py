
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D

# --------------------------------------------------
# 1. Settings and input files
# --------------------------------------------------
DATA = "data/"
SELECTED_CANDIDATE_ID = 1

roads = gpd.read_file(DATA + "roads_with_business_scores.geojson")
places = gpd.read_file(DATA + "business_clusters.geojson")
corridors = gpd.read_file(DATA + "connected_corridor_candidates.geojson")
boundaries = gpd.read_file(DATA + "connected_corridor_boundaries.geojson")

# Convert all layers to latitude/longitude
roads = roads.to_crs("EPSG:4326")
places = places.to_crs("EPSG:4326")
corridors = corridors.to_crs("EPSG:4326")
boundaries = boundaries.to_crs("EPSG:4326")

# --------------------------------------------------
# 2. Select corridor and boundary
# --------------------------------------------------
selected_corridor = corridors[
    corridors["candidate_id"] == SELECTED_CANDIDATE_ID
]

selected_boundary = boundaries[
    boundaries["candidate_id"] == SELECTED_CANDIDATE_ID
]

if selected_corridor.empty or selected_boundary.empty:
    raise ValueError("Selected corridor or boundary was not found.")

corridor_geometry = selected_corridor.geometry.iloc[0]
boundary_geometry = selected_boundary.geometry.iloc[0]

# Include all business points inside this corridor boundary
inside = places[
    places.geometry.within(boundary_geometry)
].copy()

if inside.empty:
    raise ValueError("No business points found inside the corridor boundary.")

# --------------------------------------------------
# 3. Prepare category-specific cluster labels
# --------------------------------------------------
inside["analysis_category"] = (
    inside["analysis_category"].fillna("Unknown category").astype(str)
)

inside["category_cluster"] = (
    inside["category_cluster"].fillna("noise").astype(str)
)

# Keep noise separate for display; DBSCAN noise points are not a cluster.
inside["plot_cluster"] = inside["category_cluster"].apply(
    lambda label: "Unclustered / noise"
    if label.lower() == "noise"
    else label
)

cluster_names = sorted(inside["plot_cluster"].unique())

# Use a distinct colour for each displayed cluster label
cmap = plt.get_cmap("gist_ncar", max(len(cluster_names), 1))
cluster_color = {
    name: cmap(i) for i, name in enumerate(cluster_names)
}

# --------------------------------------------------
# 4. Draw the map
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(17, 11))

# All roads
roads.plot(
    ax=ax,
    color="lightgray",
    linewidth=0.7,
    alpha=0.8,
    zorder=1
)

# Corridor boundary
selected_boundary.plot(
    ax=ax,
    facecolor="orange",
    edgecolor="darkorange",
    alpha=0.18,
    linewidth=2.5,
    zorder=2
)

# Corridor centerline
selected_corridor.plot(
    ax=ax,
    color="red",
    linewidth=3,
    zorder=3
)

# Plot every business point, coloured by category-specific cluster
for cluster_name in cluster_names:
    subset = inside[inside["plot_cluster"] == cluster_name]

    subset.plot(
        ax=ax,
        color=cluster_color[cluster_name],
        markersize=30,
        alpha=0.9,
        edgecolor="black",
        linewidth=0.35,
        zorder=4
    )

# --------------------------------------------------
# 5. Geographic axes
# --------------------------------------------------
ax.set_xlabel("Longitude (degrees)", fontsize=12)
ax.set_ylabel("Latitude (degrees)", fontsize=12)

ax.xaxis.set_major_locator(MultipleLocator(0.002))
ax.yaxis.set_major_locator(MultipleLocator(0.002))

ax.tick_params(axis="x", labelrotation=35)
ax.grid(True, linestyle="--", alpha=0.4)

ax.set_title(
    f"Chicago Commercial Corridor — Candidate {SELECTED_CANDIDATE_ID}\n"
    "All Businesses Coloured by Category-Specific Cluster",
    fontsize=14,
    pad=15
)

# --------------------------------------------------
# 6. Legend: clusters and corridor features
# --------------------------------------------------
handles = []

for cluster_name in cluster_names:
    handles.append(
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markerfacecolor=cluster_color[cluster_name],
            markeredgecolor="black",
            markersize=7,
            label=cluster_name
        )
    )

handles.extend([
    Line2D([0], [0], color="darkorange", linewidth=2.5,
           label="Corridor boundary"),
    Line2D([0], [0], color="red", linewidth=3,
           label="Corridor centerline"),
    Line2D([0], [0], color="lightgray", linewidth=2,
           label="Road network")
])

ax.legend(
    handles=handles,
    title="Business clusters",
    loc="upper left",
    bbox_to_anchor=(1.02, 1),
    fontsize=7,
    title_fontsize=9
)

# --------------------------------------------------
# 7. Corridor center coordinates
# --------------------------------------------------
center = corridor_geometry.centroid

info = (
    f"Approximate corridor center\n"
    f"Latitude: {center.y:.6f}\n"
    f"Longitude: {center.x:.6f}\n"
    f"Businesses inside boundary: {len(inside)}\n"
    f"Displayed cluster labels: {len(cluster_names)}"
)

ax.text(
    0.02, 0.02,
    info,
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment="bottom",
    bbox=dict(
        boxstyle="round,pad=0.5",
        facecolor="white",
        edgecolor="gray",
        alpha=0.95
    )
)

ax.set_aspect("equal", adjustable="datalim")
plt.tight_layout()

# --------------------------------------------------
# 8. Save and display
# --------------------------------------------------
output_file = DATA + "selected_corridor_with_coordinates.png"

plt.savefig(output_file, dpi=300, bbox_inches="tight")

print("Map saved to:", output_file)
print("Total business points inside corridor:", len(inside))
print("\nBusinesses per category:")
print(inside["analysis_category"].value_counts().to_string())

print("\nBusiness points per displayed cluster:")
print(inside["plot_cluster"].value_counts().to_string())

print(f"\nCorridor center latitude: {center.y:.6f}")
print(f"Corridor center longitude: {center.x:.6f}")

plt.show()

