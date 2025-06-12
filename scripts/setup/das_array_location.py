#!/usr/bin/env python3
"""Read DAS array location data from HDF5 and interpolate bathymetry data
at cable locations."""

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl
from scipy.interpolate import griddata

from vineyard.config import get_path
from vineyard.readers import read_bathymetry, read_das_locations


def interpolate_bathymetry(
    lon: Sequence[float],
    lat: Sequence[float],
    bathy: npt.NDArray[np.float64],
    lon_p: Sequence[float],
    lat_p: Sequence[float],
) -> npt.NDArray[np.float64]:
    grid_x, grid_y = np.meshgrid(lon, lat)
    return griddata(
        (grid_x.flatten(), grid_y.flatten()),
        np.flipud(bathy).flatten(),
        (lon_p, lat_p),
        method="cubic",
        fill_value=-9999.0,
    )


def main(das_location: Path, bathy_data: Path, savepath: Path) -> None:
    bathy, lonvec, latvec = read_bathymetry(bathy_data)
    df = read_das_locations(das_location)
    bathy_p = interpolate_bathymetry(
        lonvec, latvec, bathy, df["longitude"], df["latitude"]
    )
    df.with_columns(pl.lit(pl.Series(bathy_p)).alias("bathymetry")).write_csv(savepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process DAS array location data and save to CSV."
    )
    parser.add_argument(
        "--das_location",
        type=Path,
        default=get_path("das_location_raw"),
        help="Path to the DAS location data file.",
    )
    parser.add_argument(
        "--bathy_data",
        type=Path,
        default=get_path("bathy_data"),
        help="Path to the bathymetry data file (HDF5 format).",
    )
    parser.add_argument(
        "--savepath",
        type=Path,
        default=get_path("das_location"),
        help="Path to save the processed DAS array data.",
    )
    args = parser.parse_args()
    main(args.das_location, args.bathy_data, args.savepath)
