#!/usr/bin/env python3

import argparse
from collections.abc import Sequence
import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import polars as pl
from polars import DataFrame

from vineyard.config import get_path
import vineyard.plotting as plotting
import vineyard.readers as readers

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


def create_map(
    maps: list[str],
    bathy: dict,
    lonvec: Sequence[float],
    latvec: Sequence[float],
    das_locations: DataFrame,
    equip_locations: DataFrame,
    turbine_locations: DataFrame,
    active_turbine: dict,
    bboxes: dict[str, tuple[float, float, float, float]],
    sound_trap: dict | None = None,
) -> Figure:
    fig, axs = plt.subplots(ncols=2, figsize=(8, 4.5), gridspec_kw={"wspace": 0.01})
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
    return fig


def main(
    bathy_data: Path,
    das_location: Path,
    equipment: Path,
    turbines: Path,
    bounds_file: Path,
    active_turbine_name: str,
    savepath: Path,
    dpi: int = 300,
) -> None:
    maps = ["bounds", "inset"]

    logging.info("Loading data and plotting map.")
    bathy, lonvec, latvec = readers.read_bathymetry(bathy_data)
    das_locations = pl.read_csv(das_location)
    equip_locations = pl.read_csv(equipment)
    turbine_locations = pl.read_csv(turbines)

    bboxes = {
        "bounds": readers.read_bbox(bounds_file, "bounds"),
        "inset": readers.read_bbox(bounds_file, "inset"),
    }

    active_turbine = {
        "label": f"Turbine {active_turbine_name}",
        "longitude": turbine_locations.filter(pl.col("Turbine") == active_turbine_name)
        .select("lon")
        .item(),
        "latitude": turbine_locations.filter(pl.col("Turbine") == active_turbine_name)
        .select("lat")
        .item(),
    }
    sound_trap = {
        "label": "Monitor Hydrophone",
        "longitude": equip_locations.filter(pl.col("name") == "Monitor")
        .select("longitude")
        .item(),
        "latitude": equip_locations.filter(pl.col("name") == "Monitor")
        .select("latitude")
        .item(),
    }

    logging.info("Creating map figure.")
    fig = create_map(
        maps,
        bathy,
        lonvec,
        latvec,
        das_locations,
        equip_locations,
        turbine_locations,
        active_turbine,
        bboxes,
        sound_trap=sound_trap,
    )

    fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Map saved to {savepath.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot map of the study area with bathymetry and sensor locations."
    )
    parser.add_argument(
        "--bathy_data",
        type=Path,
        default=get_path("bathy_data"),
        help="Path to the bathymetry data file (HDF5 format).",
    )
    parser.add_argument(
        "--das_location",
        type=Path,
        default=get_path("das_location"),
        help="Path to the file containing DAS locations.",
    )
    parser.add_argument(
        "--equipment",
        type=Path,
        default=get_path("equipment_config"),
        help="Path to the file containing equipment locations.",
    )
    parser.add_argument(
        "--turbines",
        type=Path,
        default=get_path("turbine_config"),
        help="Path to the file containing turbine locations.",
    )
    parser.add_argument(
        "--bounds_file",
        type=Path,
        default=get_path("bounds_file"),
        help="Path to the file containing bounding box coordinates.",
    )
    parser.add_argument(
        "--active_turbine",
        type=str,
        default="AN36",
        help="Active turbine identifier (default: AN36).",
    )
    parser.add_argument(
        "--savepath",
        type=Path,
        default=get_path("figures") / "map.png",
        help="Path to save the figure.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for the saved figure (default: 300).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging."
    )
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO if args.verbose else logging.ERROR)
    main(
        args.bathy_data,
        args.das_location,
        args.equipment,
        args.turbines,
        args.bounds_file,
        args.active_turbine,
        args.savepath,
        args.dpi,
    )
