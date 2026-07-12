"""Create a two-panel map figure showing the experimental setup for the
Vineyard Wind project.
"""

from collections.abc import Sequence
from pathlib import Path

import cmocean as cmo
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from polars import DataFrame

from vineyard.figures.common import add_panel_label
from vineyard.figures.maps import (
    _create_context_inset,
    get_active_turbine_info,
    load_map_data,
    plot_study_area,
)

BBOX_INSET = [[-70.55, -70.249], [41.0, 41.2]]


def plot_experiment_setup(
    bathy_data: Path,
    sensor_data: Path,
    turbine_data: Path,
    active_turbine_name: str,
    image_file: Path,
) -> Figure:
    """Create a two-panel map figure from data files.

    This is the top-level function that orchestrates data loading, processing,
    and figure creation.

    Args:
        bathy_data: Path to bathymetry data file
        sensor_data: Path to sensor locations CSV
        turbine_data: Path to turbine locations CSV
        active_turbine_name: Name of the active turbine to highlight
        whale_bearings: Path to whale bearing data CSV
    """
    (
        (bathy, lonvec, latvec),
        equip_locations,
        turbine_locations,
        _,
        _,
    ) = load_map_data(bathy_data, sensor_data, turbine_data)
    active_turbine = get_active_turbine_info(turbine_locations, active_turbine_name)

    fig, axes = plt.subplots(
        ncols=2,
        figsize=(12, 6),
        gridspec_kw={"wspace": -0.15, "width_ratios": [1, 0.5]},
    )
    ax = axes[0]
    ax = plot_image(image_file, ax=ax)
    ax.text(
        0.88,
        0.33,
        "3DVHA\n(Site A)",
        transform=ax.transAxes,
        ha="center",
        bbox=dict(facecolor="white", edgecolor="none", pad=1.0),
        zorder=40,
    )
    ax.text(
        0.63,
        0.4,
        "VLA\n(Sites B & C)",
        transform=ax.transAxes,
        ha="center",
        bbox=dict(facecolor="white", edgecolor="none", pad=1.0),
        zorder=40,
    )

    ax = axes[1]
    ax = create_map(
        bathy,
        lonvec,
        latvec,
        equip_locations,
        turbine_locations,
        active_turbine,
        BBOX_INSET,
    )
    legend_bbox = (0.25, 0.9)
    legend_ncol = 1
    ax.legend(
        facecolor="white",
        edgecolor="black",
        bbox_to_anchor=legend_bbox,
        loc="center",
        ncol=legend_ncol,
        framealpha=1.0,
        markerscale=0.8,
    )
    ax.text(
        0.59,
        0.4,
        "Site A",
        transform=ax.transAxes,
        ha="center",
        bbox=dict(facecolor="white", edgecolor="none", pad=1.0),
        zorder=40,
    )
    ax.text(
        0.73,
        0.3,
        "Site B",
        transform=ax.transAxes,
        ha="center",
        bbox=dict(facecolor="white", edgecolor="none", pad=1.0),
        zorder=40,
    )
    ax.text(
        0.9,
        0.4,
        "Site C",
        transform=ax.transAxes,
        ha="center",
        bbox=dict(facecolor="white", edgecolor="none", pad=1.0),
        zorder=40,
    )

    fig.canvas.draw()
    map_pos = axes[1].get_position()
    img_pos = axes[0].get_position()
    axes[0].set_position([img_pos.x0, map_pos.y0, img_pos.width, map_pos.height])

    return fig


def plot_image(image_file: Path, ax: Axes | None = None) -> Axes:
    """Plot an image on a Matplotlib Axes.

    Args:
        image_file: Path to the image file to plot
        ax: Matplotlib Axes to plot on

    Returns:
        Matplotlib Axes with the image.
    """
    if ax is None:
        ax = plt.gca()

    image = plt.imread(image_file)
    ax.imshow(image)
    ax.axis("off")
    add_panel_label(ax, "a")
    return ax


def create_map(
    bathy: dict,
    lonvec: Sequence[float],
    latvec: Sequence[float],
    equip_locations: DataFrame,
    turbine_locations: DataFrame,
    active_turbine: dict,
    bbox_inset: list[list[float, float]],
    ax: Axes | None = None,
) -> Axes:
    """Create a two-panel map figure showing regional context and detailed study area.

    Args:
        bathy: Bathymetry data dictionary
        lonvec: Longitude vector for bathymetry
        latvec: Latitude vector for bathymetry
        equip_locations: DataFrame of equipment locations
        turbine_locations: DataFrame of turbine locations
        active_turbine: Dict with active turbine info (label, longitude, latitude)
        bbox_inset: Bounding box for inset (detailed) map [[lon_min, lon_max], [lat_min, lat_max]]

    Returns:
        Matplotlib Axes with the map.
    """
    ax, _ = plot_study_area(
        bathy,
        lonvec,
        latvec,
        equipment_df=equip_locations,
        turbines_df=turbine_locations,
        active_turbine=active_turbine,
        bounds=bbox_inset,
        ax=ax,
        scale_bar=10,
        levelsc=np.arange(-100, 1, 5),
        meridians=0.2,
        parallels=0.1,
        parallel_labels=[0, 1, 0, 0],
        marker_size=100,
        show_legend=False,
    )
    _create_context_inset(ax, bbox_inset, bounds=[[-72.5, -69.5], [40.5, 42.5]])
    add_panel_label(ax, "b")
    return ax
