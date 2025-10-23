#!/usr/bin/env python3

import argparse
from collections.abc import Sequence
import logging
from pathlib import Path

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.basemap import Basemap
import numpy as np
import polars as pl
from polars import DataFrame

from vineyard.config import get_path
import vineyard.plotting as plotting
import vineyard.readers as readers

BBOX_INSET = [[-70.65, -70.249], [40.9, 41.201]]
BBOX_OUTER = [[-71.0, -68.999], [39.5, 41.601]]

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
    bathy: dict,
    lonvec: Sequence[float],
    latvec: Sequence[float],
    equip_locations: DataFrame,
    turbine_locations: DataFrame,
    active_turbine: dict,
    bbox_inset: list[list[float, float]],
    bbox_outer: list[list[float, float]],
    bearings: Sequence[float],
    times: Sequence[np.datetime64],
) -> Figure:
    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(7, 4),
        gridspec_kw={"wspace": 0.05},
        constrained_layout=True,
    )

    ax = axes[0]
    ax, _ = plotting.plot_study_area(
        bathy,
        lonvec,
        latvec,
        equipment_df=equip_locations,
        turbines_df=turbine_locations,
        active_turbine=active_turbine,
        bounds=bbox_inset,
        ax=ax,
        scale_bar=10,
        levelsf=np.arange(-100, 10, 5),
        levelsc=np.arange(-100, 1, 5),
    )

    ax = axes[1]
    ax, m = plotting.plot_study_area(
        bathy,
        lonvec,
        latvec,
        equipment_df=equip_locations,
        active_turbine=active_turbine,
        bounds=bbox_outer,
        ax=ax,
        scale_bar=50,
        meridians=0.5,
        parallels=0.5,
        parallel_labels=[0, 1, 0, 0],
        shallowest_contour_depth=-1.0,
        levelsf=np.arange(-2500, 100, 50),
        levelsc=np.arange(-2500, 40, 50),
        show_legend=False,
    )

    xy = m(bbox_inset[0][0], bbox_inset[1][0])
    xwidth = m(bbox_inset[0][1], bbox_inset[1][0])[0] - xy[0]
    yheight = m(bbox_inset[0][0], bbox_inset[1][1])[1] - xy[1]

    box = Rectangle(
        xy,
        xwidth,
        yheight,
        linewidth=1,
        edgecolor="red",
        facecolor="none",
    )
    ax.add_patch(box)

    colors = plt.cm.RdPu(np.linspace(0.25, 1, len(bearings)))
    for brg, color in zip(bearings, colors):
        x_start, y_start = m(
            equip_locations.filter(pl.col("equipment") == "VLA1")["longitude"],
            equip_locations.filter(pl.col("equipment") == "VLA1")["latitude"],
        )
        # Convert bearing to mathematical angle (bearing: 0=N, 90=E; math: 0=E, 90=N)
        math_angle = 90.0 - brg
        x_end = x_start + 200e3 * np.cos(np.deg2rad(math_angle))
        y_end = y_start + 200e3 * np.sin(np.deg2rad(math_angle))
        ax.plot(
            [x_start, x_end],
            [y_start, y_end],
            color=color,
            linestyle="-",
            linewidth=1,
        )

    # Add inset map showing New England context
    ax_inset = inset_axes(ax, width="25%", height="25%", loc="upper right", borderpad=0.5)
    ne_bounds = [[-77.5, -62.5], [32.5, 47.5]]
    m_inset = Basemap(
        projection="merc",
        llcrnrlon=ne_bounds[0][0],
        llcrnrlat=ne_bounds[1][0],
        urcrnrlon=ne_bounds[0][1],
        urcrnrlat=ne_bounds[1][1],
        resolution="i",
        ax=ax_inset,
    )
    m_inset.drawcoastlines(linewidth=0.5, color="black")
    m_inset.fillcontinents(color="lightgray", lake_color="white")

    # Draw box showing main plot location
    xy_box = m_inset(bbox_outer[0][0], bbox_outer[1][0])
    xwidth_box = m_inset(bbox_outer[0][1], bbox_outer[1][0])[0] - xy_box[0]
    yheight_box = m_inset(bbox_outer[0][0], bbox_outer[1][1])[1] - xy_box[1]

    context_box = Rectangle(
        xy_box,
        xwidth_box,
        yheight_box,
        linewidth=1.0,
        edgecolor="blue",
        facecolor="none",
    )
    ax_inset.add_patch(context_box)

    # Add colorbar for bearing lines
    norm = Normalize(vmin=times.min(), vmax=times.max())
    sm = ScalarMappable(cmap=plt.cm.RdPu, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", shrink=0.7, pad=-0.05)
    cbar.set_label("Time of bearing estimate to whale (HH:MM)")

    # Format colorbar ticks as datetime strings (HH:MM)
    tick_times = np.linspace(times.min(), times.max(), 5)
    cbar.set_ticks(tick_times)
    cbar.set_ticklabels(
        [str(np.datetime64(int(t), "ns")).split("T")[1][:5] for t in tick_times]
    )

    return fig


def main(
    bathy_data: Path,
    equipment: Path,
    turbines: Path,
    active_turbine_name: str,
    savepath: Path,
    dpi: int = 300,
) -> None:
    logging.info("Loading data and plotting map.")
    bathy, lonvec, latvec = readers.read_bathymetry(bathy_data)
    equip_locations = pl.read_csv(equipment)
    turbine_locations = pl.read_csv(turbines)

    active_turbine = {
        "label": f"Turbine {active_turbine_name}",
        "longitude": turbine_locations.filter(pl.col("Turbine") == active_turbine_name)
        .select("lon")
        .item(),
        "latitude": turbine_locations.filter(pl.col("Turbine") == active_turbine_name)
        .select("lat")
        .item(),
    }

    bearings = np.linspace(160.0, 170.0, 6)
    start_time = np.datetime64("2023-12-01T22:15:00", "ns")
    end_time = np.datetime64("2023-12-01T22:25:00", "ns")
    times = np.linspace(
        start_time.astype("int64"),
        end_time.astype("int64"),
        len(bearings),
        dtype=np.int64,
    )

    logging.info("Creating map figure.")
    fig = create_map(
        bathy,
        lonvec,
        latvec,
        equip_locations,
        turbine_locations,
        active_turbine,
        BBOX_INSET,
        BBOX_OUTER,
        bearings,
        times,
    )

    savepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(savepath, dpi=dpi)
    # fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
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
        "--active_turbine",
        type=str,
        default="AN36",
        help="Active turbine identifier (default: AN36).",
    )
    parser.add_argument(
        "--savepath",
        type=Path,
        default=get_path("figures") / "maps" / "map.png",
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
        args.equipment,
        args.turbines,
        args.active_turbine,
        args.savepath,
        args.dpi,
    )
