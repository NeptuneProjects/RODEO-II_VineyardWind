#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt

import vwdas.paths as paths
import vwdas.plotting as plotting
import vwdas.readers as readers

map_kwargs = {
    "bounds": {
        "type": "bounds",
        "meridians": 0.2,
        "parallels": 0.2,
        "meridian_labels": [0, 0, 1, 0],
        "parallel_labels": [1, 0, 0, 0],
        "legend_loc": None,
        "scale_bar": 10,
        "shallowest_contour_depth": -10.0,
    },
    "inset": {
        "type": "inset",
        "meridians": 0.01,
        "parallels": 0.01,
        "meridian_labels": [0, 0, 0, 1],
        "parallel_labels": [0, 1, 0, 0],
        "legend_loc": "upper left",
        "scale_bar": 1,
        "shallowest_contour_depth": -4.0,
    },
}


def main(args: argparse.Namespace) -> None:
    maps = ["bounds", "inset"]

    logging.info("Loading data and plotting map.")
    bathy, lonvec, latvec = readers.read_bathymetry(paths.data.bathy / "bathy.hdf5")
    das_locations = readers.read_das_locations(paths.data.das_location)
    equip_locations = readers.read_sensor_locations(paths.data.equipment)
    turbine_locations = readers.read_turbine_locations(paths.data.turbines)

    fig, axs = plt.subplots(ncols=2, figsize=(8, 4.5), gridspec_kw={"wspace": 0.01})

    bboxes = {
        "bounds": readers.read_bbox(paths.data.bathy_bounds, "bounds"),
        "inset": readers.read_bbox(paths.data.bathy_bounds, "inset"),
    }

    active_turbine = {
        "label": "Turbine AN36",
        "longitude": -70.50724613582075,
        "latitude": 41.103092699720655,
    }
    sound_trap = {
        "label": "Monitor Hydrophone",
        "longitude": -70.5606,
        "latitude": 41.3321,
    }
    
    for map_type, ax in zip(maps, axs):
        ax = plotting.plot_study_area(
            bathy,
            lonvec,
            latvec,
            das_locations,
            equip_locations,
            turbine_locations,
            active_turbine=active_turbine,
            sound_trap=None if map_type == "bounds" else sound_trap,
            bounds=bboxes[map_type],
            ax=ax,
            inset=bboxes["inset"] if map_type == "bounds" else None,
            **map_kwargs[map_type],
        )
        logging.info(f"Map `{map_type}` plotted.")

    fname = args.savepath / f"map.png"
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Map saved to {fname.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot map of the study area with bathymetry and sensor locations."
    )
    parser.add_argument(
        "-s", "--savepath", type=Path, default=paths.reports.figures, help="Path to save the figure."
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging."
    )
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO if args.verbose else logging.ERROR)
    main(args)
