
from pathlib import Path
import geopandas as gpd
import pandas as pd

DATA = Path(__file__).resolve().parent
CRS = "EPSG:32616"

candidates = gpd.read_file(
    DATA / "distinct_corridor_candidates.geojson"
).to_crs(CRS)

roads = gpd.read_file(
    DATA / "roads_in_study_area.geojson"
).to_crs(CRS)

# Inspect the 10 highest-ranked distinct candidates
candidates = candidates.sort_values(
    "candidate_score", ascending=False
).head(10)

print("TOP 10 CANDIDATES: STREET-NAME CHECK")
print("=" * 65)

for _, candidate in candidates.iterrows():
    line = candidate.geometry
    matches = []

    for _, road in roads.iterrows():
        road_line = road.geometry

        if road_line is None or road_line.is_empty:
            continue

        # Measure how much of this road overlaps the candidate line
        overlap = line.intersection(road_line).length

        if overlap >= 10:
            name = road.get("street_name")
            road_class = road.get("class")

            matches.append({
                "street": str(name) if pd.notna(name) else "(unnamed)",
                "class": str(road_class) if pd.notna(road_class) else "(unknown)",
                "overlap_m": overlap,
            })

    if matches:
        street_table = pd.DataFrame(matches)
        summary = (
            street_table.groupby(["street", "class"], as_index=False)
            ["overlap_m"].sum()
            .sort_values("overlap_m", ascending=False)
        )
    else:
        summary = pd.DataFrame()

    print(
        f"\nRank {candidate.get('distinct_rank', '?')} | "
        f"Candidate ID: {candidate['candidate_id']} | "
        f"Length: {line.length:.1f} m | "
        f"Score: {candidate['candidate_score']:.3f}"
    )

    if summary.empty:
        print("  No matching road segments found.")
    else:
        for _, row in summary.head(8).iterrows():
            print(
                f"  {row['street']} "
                f"({row['class']}): "
                f"{row['overlap_m']:.1f} m"
            )

print("\nStreet-name check completed.")
