"""Map figure creation and related utilities."""

from collections.abc import Sequence
from pathlib import Path

import cmocean as cmo
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
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

BBOX_OUTER = [[-70.57, -70.09], [40.75, 41.201]]
INNER_BBOX = [[-74.0, -69.0], [38.5, 43.5]]
RLIM = (0, 40)
BEARING_COLORBAR_CMAP = cmo.cm.thermal


def _add_bearing_colorbar(fig: Figure, times: np.ndarray) -> None:
    """Add a colorbar showing the time mapping for bearing lines."""
    norm = Normalize(vmin=times.min(), vmax=times.max())
    sm = ScalarMappable(cmap=BEARING_COLORBAR_CMAP, norm=norm)
    sm.set_array([])
    cax = fig.add_axes([0.25, 0.06, 0.5, 0.03])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("Time on 1 Dec 2023 (HH:MM UTC) of\nmax. range estimates")

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
    whale_ranges: Path,
    time_ranges: Sequence[tuple[np.datetime64, np.datetime64]],
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
        time_ranges: List of time ranges to highlight on the map [(start, end), ...]

    Returns:
        Matplotlib figure with the whale tracking maps and data plots
    """
    (
        (bathy, lonvec, latvec),
        equip_locations,
        turbine_locations,
        whale_df,
        range_df,
    ) = _load_map_data(
        bathy_data, sensor_data, turbine_data, whale_bearings, whale_ranges
    )
    active_turbine = _get_active_turbine_info(turbine_locations, active_turbine_name)
    # bearings, times, ranges = _process_whale_bearings(whale_df, range_df)
    bearings = range_df["mean_bearing"].to_numpy()
    times = np.array(
        [np.datetime64(t, "us").astype("int64") for t in range_df["timestamp"]]
    )
    ranges = range_df["whale_range_km"].to_numpy()

    fig = plt.figure(figsize=(8, 4))
    subfigs = fig.subfigures(nrows=1, ncols=2, wspace=-0.2, width_ratios=[0.6, 0.4])
    plot_whale_data(whale_df, range_df, time_ranges, subfig=subfigs[0])

    create_map(
        bathy,
        lonvec,
        latvec,
        equip_locations,
        range_df,
        active_turbine,
        BBOX_OUTER,
        bearings,
        times,
        subfigs[1],
        ranges,
    )

    return fig


def create_map(
    bathy: dict,
    lonvec: Sequence[float],
    latvec: Sequence[float],
    equip_locations: DataFrame,
    whale_data: DataFrame,
    active_turbine: dict,
    bbox_outer: list[list[float, float]],
    bearings: Sequence[float],
    times: np.ndarray,
    subfig: plt.Figure | None = None,
    ranges: Sequence[float] | None = None,
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
        scale_bar=10,
        meridians=0.25,
        parallels=0.25,
        shallowest_contour_depth=-1.0,
        levelsc=np.arange(-100, 1, 5),
        show_legend=True,
        legend_bbox=(0.5, 0.934),
        legend_ncol=1,
        parallel_labels=[0, 1, 0, 0],
    )

    # Plot estimated whale positions
    # _plot_whale_positions(ax, m, equip_locations, whale_data)
    _plot_whale_position_brackets(ax, m, equip_locations, whale_data)

    # Add New England context inset
    _create_context_inset(ax, bbox_outer, bounds=INNER_BBOX)

    # Add panel label
    add_panel_label(ax, "d")

    # Add colorbar for bearing times
    _add_bearing_colorbar(subfig, times)

    return fig


def _plot_whale_position_brackets(
    ax: Axes,
    m: Basemap,
    equip_locations: DataFrame,
    whale_data: DataFrame,
    range_min_km: float = 5.0,
    cap_km: float = 0.75,
) -> None:
    """Plot whale position uncertainty as radial brackets along DOA from VLA1.

    Each bracket spans from min to max range along the bearing direction,
    with perpendicular caps at each end, colored by time.
    """
    x_center, y_center = m(
        equip_locations.filter(pl.col("mooring_name") == "VLA1")["longitude"],
        equip_locations.filter(pl.col("mooring_name") == "VLA1")["latitude"],
    )
    valid = whale_data.filter(pl.col("whale_range_km") < RLIM[1]).sort("timestamp")
    bearings_rad = np.deg2rad(valid["mean_bearing"].to_numpy())
    range_min_m = range_min_km * 1000
    ranges_max_m = valid["whale_range_km"].to_numpy() * 1000
    cap_m = cap_km * 1000

    times = np.array(
        [np.datetime64(i, "us").astype("int64") for i in valid["timestamp"]]
    )
    norm = Normalize(vmin=times.min(), vmax=times.max())
    colors = BEARING_COLORBAR_CMAP(norm(times))

    # Radial unit vector (bearing: sin=E, cos=N in map coords)
    radial_x = np.sin(bearings_rad)
    radial_y = np.cos(bearings_rad)
    # Perpendicular unit vector (90° CCW from radial)
    perp_x = np.cos(bearings_rad)
    perp_y = -np.sin(bearings_rad)

    x0, y0 = x_center[0], y_center[0]
    for i, color in enumerate(colors):
        x_near = x0 + range_min_m * radial_x[i]
        y_near = y0 + range_min_m * radial_y[i]
        x_far = x0 + ranges_max_m[i] * radial_x[i]
        y_far = y0 + ranges_max_m[i] * radial_y[i]

        # Radial shaft
        ax.plot([x_near, x_far], [y_near, y_far], color=color, linewidth=1.5, zorder=15)
        # Near cap
        ax.plot(
            [x_near - cap_m * perp_x[i], x_near + cap_m * perp_x[i]],
            [y_near - cap_m * perp_y[i], y_near + cap_m * perp_y[i]],
            color=color,
            linewidth=1.5,
            zorder=15,
        )
        # Far cap
        ax.plot(
            [x_far - cap_m * perp_x[i], x_far + cap_m * perp_x[i]],
            [y_far - cap_m * perp_y[i], y_far + cap_m * perp_y[i]],
            color=color,
            linewidth=1.5,
            zorder=15,
        )


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


def plot_whale_data(
    whale_df: pl.DataFrame,
    range_df: pl.DataFrame,
    time_ranges: Sequence[tuple[np.datetime64, np.datetime64]],
    subfig: plt.Figure | None = None,
) -> Figure:
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator, offset_formats=["%Y-%b-%d"] * 6)

    host = subfig if subfig is not None else plt.figure(figsize=(10, 6))
    fig = subfig.figure if subfig is not None else host

    axes = host.subplots(
        nrows=3,
        ncols=1,
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 1], "hspace": 0.12},
    )

    for ax in axes[1:]:
        ax.sharex(axes[0])

    ax = axes[0]
    ax.scatter(
        whale_df["timestamp"],
        whale_df["vla1_brg"],
        color="k",
        s=10,
        zorder=10,
    )
    for row in range_df.iter_rows(named=True):
        ax.plot(
            [np.datetime64(row["start_time"]), np.datetime64(row["stop_time"])],
            [row["y0"], row["y1"]],
            color="red",
            zorder=15,
        )
    [
        ax.axvspan(
            start,
            end,
            facecolor="gray",
            edgecolor="none",
            alpha=0.3,
            zorder=5,
        )
        for start, end in time_ranges
    ]
    ax.set_ylim(160, 200)
    ax.grid()
    ax.tick_params(labelbottom=False)
    ax.set_ylabel("Fin whale DOA (°)")
    ax.legend(["Estimates", "Segmented linear fits"], loc="upper right", framealpha=1.0)
    add_panel_label(ax, "a")

    ax = axes[1]
    ax.scatter(
        range_df["timestamp"],
        range_df["slope_deg_s"],
        color="k",
        s=10,
        zorder=10,
    )
    [
        ax.axvspan(
            start,
            end,
            facecolor="gray",
            edgecolor="none",
            alpha=0.3,
            zorder=5,
        )
        for start, end in time_ranges
    ]
    add_panel_label(ax, "b")

    ax.grid()
    ax.tick_params(labelbottom=False)
    ax.set_ylabel("Angular velocity (°/s)")

    ax = axes[2]
    ax.scatter(
        range_df["timestamp"],
        range_df["whale_range_km"],
        color="k",
        s=10,
        zorder=15,
    )
    [
        ax.axvspan(
            start,
            end,
            facecolor="gray",
            edgecolor="none",
            alpha=0.3,
            zorder=5,
        )
        for start, end in time_ranges
    ]
    ax.set_ylim(RLIM)
    ax.grid()
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Max. range (km)")
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    add_panel_label(ax, "c")

    host.align_ylabels(axes)

    return fig
