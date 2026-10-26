
from pathlib import Path
import geopandas as gpd
import pandas as pd

data_folder = Path(__file__).parent

files = list(data_folder.glob("*place*.geojson"))

if not files:
    raise FileNotFoundError("No place GeoJSON file found.")

places = gpd.read_file(files[0])

print("\nTOTAL PLACE RECORDS:", len(places))
print("\nALL AVAILABLE COLUMNS:")
print(places.columns.tolist())

# Display category-related columns
category_columns = [
    col for col in places.columns
    if any(word in col.lower()
           for word in ["category", "taxonomy", "name", "confidence"])
]

print("\nCATEGORY AND NAME COLUMNS:")
print(category_columns)

for col in category_columns:
    print(f"\n--- {col} ---")

    if places[col].apply(lambda x: isinstance(x, (dict, list))).any():
        print(places[col].dropna().head(5).to_string(index=False))
    else:
        print(places[col].value_counts(dropna=False).head(20))

# Save a small sample for inspection
places.head(20).drop(
    columns="geometry", errors="ignore"
).to_csv(data_folder / "places_sample.csv", index=False)

print("\nSaved sample to data/places_sample.csv")
