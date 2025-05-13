#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from vwdas import paths
from vwdas import readers

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata


def main():
    bathy, lonvec, latvec = readers.read_bathymetry(paths.data.bathy / "bathy.hdf5")
    das_locations = readers.read_das_locations(paths.data.das_location)

    # Interpolate the bathymetry data to the DAS locations
    grid_x, grid_y = np.meshgrid(lonvec, latvec)
    grid_z = griddata(
        (grid_x.flatten(), grid_y.flatten()),
        np.flipud(bathy).flatten(),
        (das_locations["longitude"], das_locations["latitude"]),
        method="cubic",
        fill_value=-9999.0,
    )
    das_locations["bathymetry"] = grid_z
    das_locations.to_csv(paths.data.das / "das_array.csv")


if __name__ == "__main__":
    main()
