
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

DATA = "data/"

CORRIDORS_FILE = DATA + "functional_corridor_candidates.geojson"
BOUNDARIES_FILE = DATA + "functional_corridor_boundaries.geojson"
ROADS_FILE = DATA + "roads_in_study_area.geojson"
PLACES_FILE = DATA + "places_filtered.geojson"

# Set to a candidate ID printed by the discovery script.
# Change this value to inspect another candidate.
SELECTED_CANDIDATE_ID = 1

corridors = gpd.read_file(CORRIDORS_FILE).to_crs("EPSG:4326")
boundaries = gpd.read_file(BOUNDARIES_FILE).to_crs("EPSG:4326")
roads = gpd.read_file(ROADS_FILE).to_crs("EPSG:4326")
places = gpd.read_file(PLACES_FILE).to_crs("EPSG:4326")

corridor = corridors[
    corridors["candidate_id"] == SELECTED_CANDIDATE_ID
]

boundary = boundaries[
    boundaries["candidate_id"] == SELECTED_CANDIDATE_ID
]

if corridor.empty or boundary.empty:
    raise ValueError(
        "Candidate ID not found. Check functional_corridor_summary.csv."
    )

boundary_geometry = boundary.geometry.iloc[0]
corridor_geometry = corridor.geometry.iloc[0]

# Use point locations for businesses
places = places[
    places.geometry.notna() & ~places.geometry.is_empty
].copy()

places["geometry"] = places.geometry.representative_point()

# Display all businesses inside the selected boundary
inside = places[
    places.geometry.within(boundary_geometry)
].copy()

# Apply the same functional-group mapping as the discovery script
def normalize_category(value):
    import re
    if value is None:
        return "unknown"
    return re.sub(
        r"[^a-z0-9]+", "_", str(value).lower()
    ).strip("_")

FUNCTIONAL_GROUPS = {
    "Tourism and hospitality": [
        "hotel", "motel", "hostel", "accommodation", "lodging",
        "tourist_attraction", "tourist attraction", "attraction",
        "museum", "monument", "visitor", "travel_agency",
        "travel agency", "tour_operator", "tour operator",
        "landmark", "sightseeing", "tourism",
    ],
    "Retail": [
        "store", "retail", "shopping", "fashion", "apparel",
        "clothing", "electronics", "department_store",
        "department store", "bookstore", "jewelry", "jewellery",
        "furniture_store", "furniture store", "hardware",
        "gift_shop", "gift shop", "supermarket",
        "convenience_store", "convenience store",
        "food_and_beverage_store",
    ],
    "Food and nightlife": [
        "restaurant", "casual_eatery", "casual eatery",
        "coffee_shop", "coffee shop", "cafe", "bar", "pub",
        "nightclub", "night_club", "night club", "music_venue",
        "music venue", "food", "dining", "bakery", "fast_food",
        "fast food", "ice_cream", "ice cream", "pizzeria",
    ],
    "Professional services": [
        "professional_service", "professional service",
        "real_estate_service", "real estate service",
        "financial_service", "financial service",
        "corporate_or_business_office",
        "corporate or business office",
        "b2b_office_and_professional_service",
        "business_office", "business office",
        "lawyer", "legal_service", "legal service",
        "accounting", "consulting", "insurance", "bank",
    ],
}

def assign_functional_group(category):
    normalized = normalize_category(category)
    matches = []

    for group, keywords in FUNCTIONAL_GROUPS.items():
        if any(
            normalized == normalize_category(keyword)
            or normalize_category(keyword) in normalized
            for keyword in keywords
        ):
            matches.append(group)

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        for group in [
            "Tourism and hospitality",
            "Food and nightlife",
            "Retail",
            "Professional services",
        ]:
            if group in matches:
                return group

    return "Other"

inside["functional_group"] = inside["analysis_category"].apply(
    assign_functional_group
)

# Functional-group colours
group_colors = {
    "Tourism and hospitality": "#8E44AD",
    "Retail": "#E67E22",
    "Food and nightlife": "#2980B9",
    "Professional services": "#27AE60",
    "Other": "#7F8C8D",
}

fig, ax = plt.subplots(figsize=(15, 10))

roads.plot(
    ax=ax,
    color="lightgray",
    linewidth=0.7,
    alpha=0.8,
    zorder=1
)

boundary.plot(
    ax=ax,
    facecolor="orange",
    edgecolor="darkorange",
    alpha=0.18,
    linewidth=2.5,
    zorder=2
)

corridor.plot(
    ax=ax,
    color="red",
    linewidth=3,
    zorder=3
)

# Plot businesses using the broader functional group
for group, color in group_colors.items():
    subset = inside[inside["functional_group"] == group]

    if not subset.empty:
        subset.plot(
            ax=ax,
            color=color,
            markersize=28,
            alpha=0.9,
            edgecolor="white",
            linewidth=0.4,
            zorder=4
        )

# Legend for functional groups
handles = [
    Line2D(
        [0], [0],
        marker="o",
        linestyle="None",
        markerfacecolor=color,
        markeredgecolor="white",
        markersize=8,
        label=group
    )
    for group, color in group_colors.items()
]

handles.extend([
    Line2D([0], [0], color="darkorange", linewidth=2.5,
           label="Corridor boundary"),
    Line2D([0], [0], color="red", linewidth=3,
           label="Corridor centerline"),
    Line2D([0], [0], color="lightgray", linewidth=2,
           label="Road network"),
])

ax.legend(
    handles=handles,
    title="Functional groups",
    loc="upper left",
    bbox_to_anchor=(1.02, 1),
    fontsize=9
)

# Coordinates
center = corridor_geometry.centroid

ax.text(
    0.02, 0.02,
    f"Candidate: {SELECTED_CANDIDATE_ID}\n"
    f"Functional group: {corridor.iloc[0]['functional_group']}\n"
    f"Corridor type: {corridor.iloc[0]['corridor_type']}\n"
    f"Latitude: {center.y:.6f}\n"
    f"Longitude: {center.x:.6f}\n"
    f"Businesses inside boundary: {len(inside)}",
    transform=ax.transAxes,
    va="bottom",
    fontsize=10,
    bbox=dict(
        boxstyle="round,pad=0.5",
        facecolor="white",
        edgecolor="gray",
        alpha=0.95
    )
)

ax.set_xlabel("Longitude (degrees)")
ax.set_ylabel("Latitude (degrees)")
ax.xaxis.set_major_locator(MultipleLocator(0.002))
ax.yaxis.set_major_locator(MultipleLocator(0.002))
ax.tick_params(axis="x", labelrotation=35)
ax.grid(True, linestyle="--", alpha=0.4)

ax.set_title(
    "Chicago Functional Commercial Corridor\n"
    "Businesses, Boundary, and Geographic Coordinates",
    fontsize=14
)

ax.set_aspect("equal", adjustable="datalim")
plt.tight_layout()

output = DATA + "selected_functional_corridor.png"
plt.savefig(output, dpi=300, bbox_inches="tight")

print("Saved map:", output)
print("\nBusiness counts by functional group:")
print(inside["functional_group"].value_counts().to_string())

plt.show()
