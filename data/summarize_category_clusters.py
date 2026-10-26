
from pathlib import Path
import geopandas as gpd
import pandas as pd

DATA = Path(__file__).resolve().parent

places = gpd.read_file(
    DATA / "business_clusters.geojson"
)

# Keep places that belong to a category-specific cluster
clustered = places[
    places["category_cluster"].notna()
    & (places["category_cluster"] != "noise")
].copy()

# Count places in each category-specific cluster
summary = (
    clustered.groupby(
        ["analysis_category", "category_cluster"]
    )
    .agg(
        place_count=("category_cluster", "size")
    )
    .reset_index()
    .sort_values("place_count", ascending=False)
)

output = DATA / "category_cluster_details.csv"
summary.to_csv(output, index=False)

print("Category-specific clusters:", len(summary))
print("\nLargest category-specific clusters:")
print(summary.head(30).to_string(index=False))
print("\nSaved to:", output)
