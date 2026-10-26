
from pathlib import Path
import geopandas as gpd
import pandas as pd

data_folder = Path(__file__).parent
input_file = data_folder / "chicago_places_new.geojson"

places = gpd.read_file(input_file)

def get_primary(value):
    if isinstance(value, dict):
        return str(value.get("primary") or "").strip()
    return ""

places["place_name"] = places["names"].apply(get_primary)
places["taxonomy_primary"] = places["taxonomy"].apply(get_primary)

places["analysis_category"] = places["basic_category"].fillna(
    places["taxonomy_primary"]
)

# Provisional filter for exploring likely commercial places.
# This does not prove that a place is genuine.
filtered = places[
    places["analysis_category"].notna()
    & (places["analysis_category"].astype(str).str.strip() != "")
    & (places["confidence"] >= 0.65)
    & places.geometry.notna()
    & ~places.geometry.is_empty
].copy()

print("All place records:", len(places))
print("Records with a category:", (
    places["analysis_category"].notna()
    & (places["analysis_category"].astype(str).str.strip() != "")
).sum())
print("Records passing provisional filter:", len(filtered))

print("\nCategories in filtered dataset:")
print(filtered["analysis_category"].value_counts().head(25))

output_file = data_folder / "places_filtered.geojson"
filtered.to_file(output_file, driver="GeoJSON")

print("\nSaved filtered dataset:", output_file)

