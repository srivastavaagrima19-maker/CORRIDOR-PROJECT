
from pathlib import Path
import geopandas as gpd
import pandas as pd

data_folder = Path(__file__).parent

original_file = next(
    data_folder.glob("*2026-09-23*place*.geojson")
)
new_file = data_folder / "chicago_places_new.geojson"

old = gpd.read_file(original_file)
new = gpd.read_file(new_file)


def extract_name(value):
    if isinstance(value, dict):
        return str(value.get("primary") or "").strip()
    return str(value or "").strip()


def report_quality(label, df):
    names = df["names"].apply(extract_name)
    categories = df["basic_category"]

    print(f"\n========== {label} ==========")
    print("Total records:", len(df))
    print("Missing names:", (names == "").sum())
    print("Missing categories:", categories.isna().sum())
    print("Unique IDs:", df["id"].nunique())
    print("Unique names:", names[names != ""].nunique())

    print("\nSample names:")
    for name in names.head(20):
        print("-", name)


report_quality("ORIGINAL DATASET", old)
report_quality("NEW DATASET", new)

print("\n========== CATEGORY COMPARISON ==========")
print("\nOriginal top categories:")
print(
    old["basic_category"]
    .fillna("uncategorized")
    .value_counts()
    .head(15)
)

print("\nNew top categories:")
print(
    new["basic_category"]
    .fillna("uncategorized")
    .value_counts()
    .head(15)
)


