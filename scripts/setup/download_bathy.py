#!/usr/bin/env python3
"""Script to download bathymetry data from NCEI and save it to a local file."""

import argparse
import logging
from pathlib import Path

import bathyreq
import h5py
import numpy as np
import numpy.typing as npt

from vineyard.config import get_path
from vineyard.readers import read_bbox


def download_bathy(
    bounds: list[float], xres: int = 400, yres: int = 400
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Download bathymetry data from NCEI.

    Args:
        bounds: List of bounding box coordinates [lon_min, lat_min, lon_max, lat_max].
        xres: X resolution of the bathymetry data.
        yres: Y resolution of the bathymetry data.
    Returns:
        Tuple of numpy arrays containing bathymetry data, longitude vector,
        and latitude vector.
    """
    req = bathyreq.BathyRequest()
    return req.get_area(longitude=bounds[0], latitude=bounds[1], size=[xres, yres])


def save_bathy_data(
    data: npt.NDArray[np.float64],
    lonvec: npt.NDArray[np.float64],
    latvec: npt.NDArray[np.float64],
    fname: Path = "bathy.hdf5",
) -> None:
    """Save bathymetry data to a local file."""
    with h5py.File(fname, "w") as f:
        f.create_dataset("data", data=data)
        f.create_dataset("lonvec", data=lonvec)
        f.create_dataset("latvec", data=latvec)


def main(bounds_file: Path, fname: Path, xres: int, yres: int) -> None:
    bounds = read_bbox(bounds_file, "bounds")

    logging.info(f"Bounding box read from {bounds_file}.")

    logging.info(
        f"Downloading bathymetry data for bounding box: {bounds}"
        f" with resolution {xres}x{yres}"
    )
    data, lonvec, latvec = download_bathy(bounds, xres, yres)
    logging.info("Bathymetry data downloaded successfully.")

    save_bathy_data(data, lonvec, latvec, fname=fname)
    logging.info(f"Saving bathymetry data to {fname}")


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bounds_file",
        type=Path,
        default=get_path("bounds_file"),
        help="Set the filename of the bounding box coordinates.",
    )
    parser.add_argument(
        "--fname",
        type=Path,
        default=get_path("bathy_data"),
        help="Set the filename of the bathymetry data.",
    )
    parser.add_argument(
        "--xres",
        type=int,
        default=800,
        help="Set the x resolution of the bathymetry data.",
    )
    parser.add_argument(
        "--yres",
        type=int,
        default=800,
        help="Set the y resolution of the bathymetry data.",
    )
    args = parser.parse_args()
    main(args.bounds_file, args.fname, args.xres, args.yres)
