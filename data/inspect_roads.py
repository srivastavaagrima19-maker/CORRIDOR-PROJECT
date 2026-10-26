
from pathlib import Path
import geopandas as gpd

data_folder = Path(__file__).parent

road_file = next(data_folder.glob("*segment*.geojson"))
roads = gpd.read_file(road_file)

print("Road file:", road_file.name)
print("Total road segments:", len(roads))
print("\nColumns:")
print(roads.columns.tolist())

print("\nCoordinate system:", roads.crs)
print("\nGeometry types:")
print(roads.geometry.geom_type.value_counts())

print("\nMissing geometries:", roads.geometry.isna().sum())
print("Empty geometries:", roads.geometry.is_empty.sum())

print("\nSample road records:")
print(roads.head(5).to_string())

for column in roads.columns:
    if any(word in column.lower() for word in
           ["class", "subtype", "highway", "road", "name"]):
        print(f"\nValues in {column}:")
        print(roads[column].value_counts(dropna=False).head(15))
