
from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt

DATA = Path(__file__).resolve().parent
CRS = "EPSG:32616"

candidates = gpd.read_file(
    DATA / "distinct_corridor_candidates.geojson"
).to_crs(CRS)

roads = gpd.read_file(
    DATA / "roads_in_study_area.geojson"
).to_crs(CRS)

boundaries = gpd.read_file(
    DATA / "distinct_corridor_boundaries.geojson"
).to_crs(CRS)

# Select the six highest-scoring distinct candidates
top = candidates.sort_values(
    "candidate_score", ascending=False
).head(6)

fig, ax = plt.subplots(figsize=(12, 10))

# Background road network
roads.plot(ax=ax, color="lightgray", linewidth=0.5)

# Candidate boundaries
top_ids = set(top["candidate_id"])
selected_boundaries = boundaries[
    boundaries["candidate_id"].isin(top_ids)
]

selected_boundaries.plot(
    ax=ax, alpha=0.18, edgecolor="black", linewidth=0.8
)

# Candidate corridor lines and rank labels
top.plot(
    ax=ax, column="candidate_score",
    cmap="tab10", linewidth=2.5, legend=True
)

for _, row in top.iterrows():
    point = row.geometry.interpolate(0.5, normalized=True)
    rank = row.get("distinct_rank", "?")
    ax.annotate(
        f"Rank {rank} (ID {row['candidate_id']})",
        (point.x, point.y),
        fontsize=8,
        xytext=(4, 4),
        textcoords="offset points"
    )

ax.set_title("Top 6 Data-Discovered Chicago Corridor Candidates")
ax.set_axis_off()
plt.tight_layout()

output = DATA / "top_corridors_map.png"
plt.savefig(output, dpi=200, bbox_inches="tight")
plt.close()

print(f"Map saved to: {output}")
print("Candidates shown:")
print(top[["distinct_rank", "candidate_id", "length_m",
           "candidate_score"]].to_string(index=False))
