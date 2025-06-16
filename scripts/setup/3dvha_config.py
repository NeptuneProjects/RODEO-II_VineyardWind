#!/usr/bin/env python3
"""3DVHA configuration for plotting and analysis."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from polars import DataFrame
import polars as pl
from tritonoa.data.time import TIME_PRECISION

from vineyard.config import get_path


def define_array_positions():
    h0 = np.array([0.0, 0.0, -np.sqrt(3 / 8)])  # top
    h1 = np.array([1 / np.sqrt(12), 1 / 2, 1 / np.sqrt(24)])  # right
    h2 = np.array([1 / np.sqrt(12), -1 / 2, 1 / np.sqrt(24)])  # left
    h3 = np.array([-1 / np.sqrt(3), 0.0, 1 / np.sqrt(24)])  # back
    middle = np.array([0.0, 0.0, 0.0])
    return np.row_stack([h0, h1, h2, h3, middle])


def get_vector_sensor_motion(path: Path) -> DataFrame:
    glob_pattern = "m*.log"
    regex_pattern = r"(-?\d+\.?\d*)"
    return (
        pl.read_csv(f"{str(path)}/{glob_pattern}")
        .with_columns(
            pl.col("dateTimeUTC").str.strptime(pl.Datetime, "%Y/%m/%d-%H:%M:%S")
        )
        .with_columns(pl.col("roll").str.extract(regex_pattern).cast(pl.Float64))
        .with_columns(pl.col("pitch").str.extract(regex_pattern).cast(pl.Float64))
        .with_columns(pl.col("yaw").str.extract(regex_pattern).cast(pl.Float64))
    )


def main(vector_sensor_imu_raw: Path, vector_sensor_imu_processed: Path):
    df = get_vector_sensor_motion(vector_sensor_imu_raw)
    df.write_csv(vector_sensor_imu_processed)

    # array_positions = define_array_positions()
    # hydrophone_positions = array_positions.copy()
    # vector_sensor_positions = array_positions.copy()[4:]
    # print(array_positions)
    return


if __name__ == "__main__":
    parser = ArgumentParser(
        description="3DVHA configuration for plotting and analysis."
    )
    parser.add_argument(
        "--vecsens_motion_src",
        type=Path,
        default=get_path("vector_sensor_imu_raw"),
        help="Path to the 3DVHA data directory.",
    )
    parser.add_argument(
        "--vecsens_motion_dest",
        type=Path,
        default=get_path("vector_sensor_imu_processed"),
        help="Path to the 3DVHA data directory.",
    )
    args = parser.parse_args()
    main(args.vecsens_motion_src, args.vecsens_motion_dest)
