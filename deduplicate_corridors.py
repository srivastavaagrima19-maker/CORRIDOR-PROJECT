
from pathlib import Path
import pandas as pd
import geopandas as gpd

# Project data folder
DATA = Path(__file__).resolve().parent / "data"

# Input files
csv_path = DATA / "network_corridor_candidates.csv"
geo_path = DATA / "network_corridor_candidates.geojson"

# Output files
out_csv = DATA / "distinct_corridor_candidates.csv"
out_geo = DATA / "distinct_corridor_candidates.geojson"
out_boundary = DATA / "distinct_corridor_boundaries.geojson"

# Load candidates and scores
scores = pd.read_csv(csv_path)
candidates = gpd.read_file(geo_path).to_crs("EPSG:32616")

# Match the score records to the candidate geometries
scores["candidate_id"] = scores["candidate_id"].astype(str)
candidates["candidate_id"] = candidates["candidate_id"].astype(str)

gdf = candidates.merge(
    scores,
    on="candidate_id",
    how="inner",
    suffixes=("", "_csv")
)

# Prefer higher-scoring candidates
gdf = gdf.sort_values("candidate_score", ascending=False).reset_index(drop=True)

# Keep only geometrically distinct paths
kept = []
overlap_threshold = 0.60

for _, candidate in gdf.iterrows():
    line = candidate.geometry

    if line is None or line.is_empty or line.length == 0:
        continue

    duplicate = False

    for accepted in kept:
        other_line = accepted.geometry

        overlap_length = line.intersection(other_line).length
        shorter_length = min(line.length, other_line.length)

        overlap_ratio = (
            overlap_length / shorter_length
            if shorter_length > 0
            else 0
        )

        if overlap_ratio >= overlap_threshold:
            duplicate = True
            break

    if not duplicate:
        kept.append(candidate)

# Save distinct candidates
distinct = gpd.GeoDataFrame(kept, geometry="geometry", crs="EPSG:32616")
distinct = distinct.reset_index(drop=True)
distinct["distinct_rank"] = range(1, len(distinct) + 1)

distinct.to_file(out_geo, driver="GeoJSON")

# Save the matching table without geometry
distinct.drop(columns="geometry").to_csv(out_csv, index=False)

# Create a 35-metre buffer boundary around each candidate
boundaries = distinct[["candidate_id", "distinct_rank", "geometry"]].copy()
boundaries["geometry"] = boundaries.geometry.buffer(35)
boundaries = boundaries.set_geometry("geometry")
boundaries.to_file(out_boundary, driver="GeoJSON")

print("Original candidates:", len(gdf))
print("Distinct candidates retained:", len(distinct))
print("Candidates removed as duplicates:", len(gdf) - len(distinct))

print("\nTOP DISTINCT CORRIDORS")
columns = [
    c for c in [
        "distinct_rank",
        "candidate_id",
        "segment_count",
        "length_m",
        "mean_places_50m",
        "mean_activity_score",
        "candidate_score",
    ]
    if c in distinct.columns
]

print(distinct[columns].head(15).to_string(index=False))

print("\nFiles saved:")
print(out_csv)
print(out_geo)
print(out_boundary)
