#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script to download bathymetry data from NCEI and save it to a local file."""

import argparse
import logging

import bathyreq
import h5py
import numpy as np
import numpy.typing as npt

import vwdas.paths as paths
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
    fname: str = "bathy.hdf5",
) -> None:
    """Save bathymetry data to a local file."""
    with h5py.File(paths.data.bathy / fname, "w") as f:
        f.create_dataset("data", data=data)
        f.create_dataset("lonvec", data=lonvec)
        f.create_dataset("latvec", data=latvec)


def main(args: argparse.Namespace) -> None:
    bounds = read_bbox(paths.data.bathy_bounds, "bounds")

    logging.info(f"Bounding box read from {paths.data.bathy_bounds}.")

    logging.info(
        f"Downloading bathymetry data for bounding box: {bounds}"
        f" with resolution {args.xres}x{args.yres}"
    )
    data, lonvec, latvec = download_bathy(bounds, args.xres, args.yres)
    logging.info("Bathymetry data downloaded successfully.")

    save_bathy_data(data, lonvec, latvec, fname=args.fname)
    logging.info(f"Saving bathymetry data to {paths.data.bathy / args.fname}")


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--xres",
        type=int,
        default=400,
        help="Set the x resolution of the bathymetry data.",
    )
    parser.add_argument(
        "--yres",
        type=int,
        default=400,
        help="Set the y resolution of the bathymetry data.",
    )
    parser.add_argument(
        "--fname",
        type=str,
        default="bathy.hdf5",
        help="Set the filename of the bathymetry data.",
    )
    args = parser.parse_args()
    main(args)
