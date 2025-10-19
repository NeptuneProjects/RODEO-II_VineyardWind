"""
Compute distances between equipment locations (3DVHA, VLA1, VLA2).
"""

from pathlib import Path

import polars as pl
from geopy.distance import geodesic

from vineyard.config import get_path


def compute_distances(csv_path: Path, output_path: Path):
    """
    Compute pairwise distances between 3DVHA, VLA1, and VLA2.

    Args:
        csv_path: Path to the equipment CSV file
        output_path: Path to save the distance lookup table

    Returns:
        Polars DataFrame with distance lookup table
    """
    # Read equipment data
    df = pl.read_csv(csv_path)

    # Filter for the three equipment of interest
    equipment_names = ["3DVHA", "VLA1", "VLA2"]
    df_filtered = df.filter(pl.col("equipment").is_in(equipment_names))

    # Create coordinate pairs (latitude, longitude)
    coords = {}
    for row in df_filtered.iter_rows(named=True):
        coords[row["equipment"]] = (row["latitude"], row["longitude"])

    # Compute pairwise distances and store in lists
    from_equipment = []
    to_equipment = []
    distance_meters = []

    for i, name1 in enumerate(equipment_names):
        for name2 in equipment_names[i + 1 :]:
            if name1 in coords and name2 in coords:
                distance = geodesic(coords[name1], coords[name2])

                # Store both directions for easy lookup
                for src, dst in [(name1, name2), (name2, name1)]:
                    from_equipment.append(src)
                    to_equipment.append(dst)
                    distance_meters.append(distance.meters)

    distance_df = pl.DataFrame(
        {
            "from_equipment": from_equipment,
            "to_equipment": to_equipment,
            "distance_meters": distance_meters,
        }
    )

    distance_df.write_csv(output_path)
    return distance_df


if __name__ == "__main__":
    compute_distances(get_path("equipment_config"), get_path("distance_lut"))
