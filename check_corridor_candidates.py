
from pathlib import Path
import geopandas as gpd

DATA = Path(__file__).resolve().parent / "data"

candidates = gpd.read_file(
    DATA / "network_corridor_candidates.geojson"
).to_crs("EPSG:32616")

print("Total candidates:", len(candidates))
print("\nTop candidate geometry checks:")

for i in range(min(10, len(candidates))):
    a = candidates.iloc[i].geometry

    print(
        f"\nRank {i + 1}:",
        "| Length:", round(a.length, 1), "m",
        "| Geometry type:", a.geom_type,
        "| Bounds:", tuple(round(x, 1) for x in a.bounds),
    )

    for j in range(i):
        b = candidates.iloc[j].geometry

        overlap = a.intersection(b).length
        shorter = min(a.length, b.length)
        overlap_pct = 100 * overlap / shorter if shorter else 0

        if overlap_pct >= 50:
            print(
                f"  Overlaps rank {j + 1}: "
                f"{overlap_pct:.1f}% of shorter path"
            )
