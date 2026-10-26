
from pathlib import Path
import osmnx as ox

data_folder = Path(__file__).parent

# Same Chicago study area
north = 41.913
south = 41.903
east = -87.659
west = -87.683

tags = {
    "shop": True,
    "office": True,
    "craft": True,
    "amenity": [
        "restaurant", "cafe", "fast_food",
        "bar", "pub", "bank", "pharmacy",
        "clinic", "dentist", "marketplace",
        "theatre", "cinema"
    ],
    "tourism": ["hotel", "museum", "gallery"],
    "leisure": ["fitness_centre", "sports_centre"]
}

# Try alternate public Overpass endpoints
endpoints  = [
    "https://overpass.kumi.systems/api",
    "https://overpass-api.de/api",
]
    


ox.settings.timeout = 60
ox.settings.use_cache = True

businesses = None

for endpoint in endpoints:
    print(f"\nTrying server: {endpoint}")

    ox.settings.overpass_url = endpoint

    try:
        businesses = ox.features.features_from_bbox(
            (west, south, east, north),
            tags
        )

        print("Download successful!")
        break

    except Exception as error:
        print("This server failed:", error)
        businesses = None

if businesses is None:
    print("\nBoth servers failed.")
    print("Do not keep retrying rapidly.")
    print("We will use another download method if needed.")

elif businesses.empty:
    print("The server returned no matching features.")

else:
    output_file = data_folder / "osm_businesses.geojson"

    businesses.to_file(output_file, driver="GeoJSON")

    print("\nDOWNLOAD SUCCESSFUL")
    print("Records downloaded:", len(businesses))
    print("Saved to:", output_file)

