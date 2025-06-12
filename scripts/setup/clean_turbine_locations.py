#!/usr/bin/env python3
"""This script cleans the turbine location data and saves it to the
configuration directory.
"""
import argparse
from pathlib import Path

import polars as pl

from vineyard.config import get_path


def main(source_file: Path, dest_file: Path) -> None:
    (
        pl.read_csv(source_file)
        .select(["Name", "Latitude dd", "longitude dd"])
        .rename({"Name": "Turbine", "Latitude dd": "lat", "longitude dd": "lon"})
        .write_csv(dest_file)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean turbine locations data.")
    parser.add_argument(
        "--input",
        type=Path,
        default=get_path("raw_turbine_data"),
        help="Path to the input CSV file containing turbine locations.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=get_path("turbine_config"),
        help="Path to save the cleaned CSV file.",
    )
    args = parser.parse_args()
    main(args.input, args.output)
