
from pathlib import Path

import geopandas as gpd
import pandas as pd
from sklearn.cluster import DBSCAN


# --------------------------------------------------
# 1. File paths and settings
# --------------------------------------------------

DATA = Path(__file__).resolve().parent
CRS = "EPSG:32616"

SPATIAL_CLUSTER_DISTANCE = 100  # metres
MIN_SPATIAL_PLACES = 4

CATEGORY_CLUSTER_DISTANCE = 100  # metres
MIN_CATEGORY_PLACES = 3

SPECIALIZATION_THRESHOLD = 0.60


# --------------------------------------------------
# 2. Load the existing place dataset
# --------------------------------------------------

places = gpd.read_file(
    DATA / "places_filtered.geojson"
)

print("Loaded places:", len(places))

if places.crs is None:
    raise ValueError(
        "The places dataset has no CRS. "
        "Check the original GeoJSON before continuing."
    )

places = places.to_crs(CRS)

# Remove missing or empty geometries
places = places[
    places.geometry.notna()
    & ~places.geometry.is_empty
].copy()

if "analysis_category" not in places.columns:
    raise ValueError(
        "Column 'analysis_category' was not found."
    )

# Keep missing categories identifiable
places["analysis_category"] = (
    places["analysis_category"]
    .fillna("Unknown")
    .astype(str)
)

# Use representative points for distance calculations.
# The active geometry will become these points.
places["cluster_point"] = (
    places.geometry.representative_point()
)

places = places.set_geometry("cluster_point")

# Remove the extra geometry column to avoid the GeoJSON
# multiple-geometry-column export error.
if "geometry" in places.columns:
    places = places.drop(columns=["geometry"])


# --------------------------------------------------
# 3. Prepare coordinates in metres
# --------------------------------------------------

coordinates = pd.DataFrame({
    "x": places.geometry.x,
    "y": places.geometry.y
})

if len(places) == 0:
    raise ValueError(
        "No valid place geometries remain after cleaning."
    )


# --------------------------------------------------
# 4. Find general spatial clusters
# --------------------------------------------------

spatial_model = DBSCAN(
    eps=SPATIAL_CLUSTER_DISTANCE,
    min_samples=MIN_SPATIAL_PLACES
)

places["spatial_cluster"] = spatial_model.fit_predict(
    coordinates[["x", "y"]]
)

# DBSCAN label -1 means noise: not assigned to a cluster.


# --------------------------------------------------
# 5. Find nearby clusters within each business category
# --------------------------------------------------

places["category_cluster"] = "noise"

for category, index in places.groupby(
    "analysis_category",
    sort=False
).groups.items():

    category_places = places.loc[index]

    category_coordinates = pd.DataFrame({
        "x": category_places.geometry.x,
        "y": category_places.geometry.y
    })

    if len(category_places) < MIN_CATEGORY_PLACES:
        continue

    category_model = DBSCAN(
        eps=CATEGORY_CLUSTER_DISTANCE,
        min_samples=MIN_CATEGORY_PLACES
    )

    labels = category_model.fit_predict(
        category_coordinates[["x", "y"]]
    )

    # Prefix each cluster label with its category name
    category_labels = [
        f"{category}_{label}" if label >= 0 else "noise"
        for label in labels
    ]

    places.loc[index, "category_cluster"] = category_labels


# --------------------------------------------------
# 6. Summarize general spatial clusters
# --------------------------------------------------

clustered = places[
    places["spatial_cluster"] >= 0
].copy()

if clustered.empty:
    summary = pd.DataFrame(columns=[
        "spatial_cluster",
        "place_count",
        "category_count",
        "dominant_category",
        "dominant_category_count",
        "dominant_share",
        "cluster_type"
    ])

else:
    summary = (
        clustered.groupby("spatial_cluster")
        .agg(
            place_count=("analysis_category", "size"),
            category_count=("analysis_category", "nunique")
        )
        .reset_index()
    )

    # Most common category within each spatial cluster
    category_counts = (
        clustered.groupby(
            ["spatial_cluster", "analysis_category"]
        )
        .size()
        .reset_index(name="count")
        .sort_values(
            ["spatial_cluster", "count", "analysis_category"],
            ascending=[True, False, True]
        )
    )

    dominant = (
        category_counts
        .drop_duplicates("spatial_cluster")
        .rename(columns={
            "analysis_category": "dominant_category",
            "count": "dominant_category_count"
        })
    )

    summary = summary.merge(
        dominant[
            [
                "spatial_cluster",
                "dominant_category",
                "dominant_category_count"
            ]
        ],
        on="spatial_cluster",
        how="left"
    )

    summary["dominant_share"] = (
        summary["dominant_category_count"]
        / summary["place_count"]
    )

    # Initial classification, based on category share
    summary["cluster_type"] = summary[
        "dominant_share"
    ].apply(
        lambda share: (
            "specialized"
            if share >= SPECIALIZATION_THRESHOLD
            else "mixed-use"
        )
    )


# --------------------------------------------------
# 7. Save cluster summary and place-level results
# --------------------------------------------------

summary_path = DATA / "business_spatial_cluster_summary.csv"
places_path = DATA / "business_clusters.geojson"

summary.to_csv(summary_path, index=False)

# Save one geometry column only: representative-point geometry
places.to_file(
    places_path,
    driver="GeoJSON"
)


# --------------------------------------------------
# 8. Print results
# --------------------------------------------------

print("\nBusiness clustering completed.")
print("Valid places processed:", len(places))

print(
    "Places assigned to spatial clusters:",
    int((places["spatial_cluster"] >= 0).sum())
)

print(
    "Places marked as spatial noise:",
    int((places["spatial_cluster"] == -1).sum())
)

print("Spatial clusters found:", len(summary))

if not summary.empty:
    print("\nLargest clusters:")

    print(
        summary.sort_values(
            "place_count",
            ascending=False
        )
        .head(20)
        .to_string(index=False)
    )

    print("\nCluster type counts:")

    print(summary["cluster_type"].value_counts().to_string())

print("\nCategory-specific clusters found:",
      places.loc[
          places["category_cluster"] != "noise",
          "category_cluster"
      ].nunique())

print("\nSaved files:")
print(places_path)
print(summary_path)

