"""Map figure creation and related utilities."""

from collections.abc import Sequence
from pathlib import Path

import cmocean as cmo
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.basemap import Basemap
from polars import DataFrame

from vineyard.figures.common import add_panel_label
from vineyard.figures.maps import (
    _create_context_inset,
    _get_active_turbine_info,
    _load_map_data,
    _plot_study_area,
)

BBOX_OUTER = [[-71.0, -68.999], [39.5, 41.601]]


def _add_bearing_colorbar(fig: Figure, times: Sequence[np.datetime64]) -> None:
    """Add a colorbar showing the time mapping for bearing lines."""
    norm = Normalize(vmin=times.min(), vmax=times.max())
    sm = ScalarMappable(cmap=plt.cm.RdPu, norm=norm)
    sm.set_array([])
    cax = fig.add_axes([0.25, 0.06, 0.5, 0.03])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("Time on 1 Dec 2023 (HH:MM UTC)")

    # Format colorbar ticks as datetime strings (HH:MM)
    tick_times = np.linspace(times.min(), times.max(), 5)
    cbar.set_ticks(tick_times)
    cbar.set_ticklabels(
        [str(np.datetime64(int(t), "us")).split("T")[1][:5] for t in tick_times]
    )


def plot_whale_tracking(
    bathy_data: Path,
    sensor_data: Path,
    turbine_data: Path,
    active_turbine_name: str,
    whale_bearings: Path,
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
        whale_df,
    ) = _load_map_data(bathy_data, sensor_data, turbine_data, whale_bearings)
    active_turbine = _get_active_turbine_info(turbine_locations, active_turbine_name)
    bearings, times = _process_whale_bearings(whale_df)

    fig = plt.figure(figsize=(10, 5))
    subfigs = fig.subfigures(nrows=1, ncols=2, wspace=-0.12, width_ratios=[0.4, 0.6])

    create_map(
        bathy,
        lonvec,
        latvec,
        equip_locations,
        active_turbine,
        BBOX_OUTER,
        bearings,
        times,
        subfigs[0],
    )

    plot_whale_data(whale_df, subfig=subfigs[1])
    return fig


def create_map(
    bathy: dict,
    lonvec: Sequence[float],
    latvec: Sequence[float],
    equip_locations: DataFrame,
    active_turbine: dict,
    bbox_outer: list[list[float, float]],
    bearings: Sequence[float],
    times: Sequence[np.datetime64],
    subfig: plt.Figure | None = None,
) -> Figure:
    """Create a two-panel map figure showing regional context and detailed study area.

    Args:
        bathy: Bathymetry data dictionary
        lonvec: Longitude vector for bathymetry
        latvec: Latitude vector for bathymetry
        equip_locations: DataFrame of equipment locations
        active_turbine: Dict with active turbine info (label, longitude, latitude)
        bbox_inset: Bounding box for inset (detailed) map [[lon_min, lon_max], [lat_min, lat_max]]
        bbox_outer: Bounding box for outer (regional) map
        bearings: Array of bearing angles
        times: Array of timestamps for bearing colormap

    Returns:
        Matplotlib figure with two map panels
    """
    if subfig is not None:
        ax = subfig.add_subplot(111)
        fig = subfig.figure
    else:
        fig, ax = plt.subplots(figsize=(10, 6))

    ax, m = _plot_study_area(
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
        shallowest_contour_depth=-1.0,
        levelsf=np.arange(-2500, 100, 50),
        levelsc=np.arange(-2500, 40, 100),
        show_legend=False,
    )

    # Plot bearing lines with color mapping
    colors = plt.cm.RdPu(np.linspace(0.25, 1, len(bearings)))
    _plot_bearing_lines(ax, m, equip_locations, bearings, colors)

    # Add New England context inset
    _create_context_inset(ax, bbox_outer, bounds=[[-73.0, -67.0], [37.0, 43.0]])

    # Add panel label
    add_panel_label(ax, "a")

    # Add colorbar for bearing times
    _add_bearing_colorbar(subfig, times)

    return fig


def _plot_bearing_lines(
    ax,
    m: Basemap,
    equip_locations: DataFrame,
    bearings: Sequence[float],
    colors: np.ndarray,
) -> None:
    """Plot bearing lines from VLA1 with specified colors."""
    for brg, color in zip(bearings, colors):
        x_start, y_start = m(
            equip_locations.filter(pl.col("mooring_name") == "VLA1")["longitude"],
            equip_locations.filter(pl.col("mooring_name") == "VLA1")["latitude"],
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
            zorder=15,
        )


def _process_whale_bearings(whale_df: DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Process whale bearings by correcting ambiguities and filtering.

    Returns:
        Tuple of (bearings, times) as numpy arrays
    """
    # Correct bearing ambiguities
    unamb_bearings = whale_df.filter(pl.col("vla1_brg") > 90)
    amb_bearings = whale_df.filter(pl.col("vla1_brg") < 90)
    amb_bearings = amb_bearings.with_columns(
        pl.col("vla1_brg").add(2 * (90 - pl.col("vla1_brg")))
    )

    # Concatenate, filter, and sort
    corr_whale_df = (
        pl.concat([unamb_bearings, amb_bearings])
        .filter(pl.col("vla1_brg") < 175, pl.col("vla1_brg") > 155)
        .sort("timestamp")
    )

    bearings = corr_whale_df["vla1_brg"].to_numpy()
    times = np.array(
        [np.datetime64(i, "us").astype("int64") for i in corr_whale_df["timestamp"]]
    )

    return bearings, times


def plot_whale_data(df: pl.DataFrame, subfig: plt.Figure | None = None) -> Figure:
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator, offset_formats=["%Y-%b-%d"] * 6)

    host = subfig if subfig is not None else plt.figure(figsize=(10, 6))
    fig = subfig.figure if subfig is not None else host

    gs = host.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.3)
    gs_upper = gs[0].subgridspec(3, 1, hspace=0.05)
    axes = [host.add_subplot(gs_upper[i]) for i in range(3)]
    ax4 = host.add_subplot(gs[1])

    for ax in axes[1:]:
        ax.sharex(axes[0])

    ax = axes[0]
    ax.scatter(
        df["timestamp"],
        df["vla1_brg"],
        color="k",
        s=10,
        zorder=10,
    )
    ax.set_ylim(160, 200)
    ax.grid()
    ax.tick_params(labelbottom=False)
    ax.set_ylabel("Fin whale DOA (°)")
    add_panel_label(ax, "b")

    ax = axes[1]
    ax.scatter(
        df["timestamp"],
        df["angular_velocity_smoothed"],
        color="k",
        s=10,
        zorder=10,
    )
    add_panel_label(ax, "c")

    ax.grid()
    ax.tick_params(labelbottom=False)
    ax.set_ylabel("Angular velocity (°/s)")

    ax = axes[2]
    ax.scatter(
        df["timestamp"],
        df["whale_range_km"],
        color="k",
        s=10,
        zorder=10,
    )
    ax.set_ylim(0, 1000)
    ax.grid()
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Max. range (km)")
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    add_panel_label(ax, "d")

    ax4 = _plot_range_est_distribution(df, ax4)
    add_panel_label(ax4, "e")

    return fig


def _plot_range_est_distribution(df: pl.DataFrame, ax: Axes) -> Axes:
    """Plot distribution of whale range estimates."""
    data = df["whale_range_km"].filter(df["whale_range_km"] < 500).to_numpy()
    ax.hist(
        data,
        bins=50,
        color="grey",
        zorder=10,
        label=None,
    )
    ax.axvline(
        np.min(data),
        color="tab:blue",
        linestyle="--",
        linewidth=3,
        label=f"Minimum estimate: {np.min(data):.1f} km",
        zorder=15,
    )
    ax.axvline(
        np.median(data),
        color="tab:red",
        linestyle="--",
        linewidth=3,
        label=f"Median estimate: {np.median(data):.1f} km",
        zorder=15,
    )
    ax.legend(loc="upper right")
    ax.set_xlabel("Maximum range estimates (km)")
    ax.set_ylabel("Count")
    ax.grid()
    return ax
