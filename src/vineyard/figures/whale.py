"""Plot whale tracking results."""

from collections.abc import Sequence
from pathlib import Path

import matplotlib
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
    get_active_turbine_info,
    load_map_data,
    plot_study_area,
)

BBOX_OUTER = [[-70.4, -70.3], [40.83, 41.151]]
INNER_BBOX = [[-74.0, -69.0], [38.5, 43.5]]
RLIM = (0, 25)

_cmap = matplotlib.cm.get_cmap("turbo")
colorbar_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "custom", _cmap(np.linspace(0.0, 0.9, 256))
)


def _add_bearing_colorbar(fig: Figure, times: np.ndarray) -> None:
    """Add a colorbar showing the time mapping for bearing lines."""
    norm = Normalize(vmin=times.min(), vmax=times.max())
    sm = ScalarMappable(cmap=colorbar_cmap, norm=norm)
    sm.set_array([])
    cax = fig.add_axes([0.23, 0.84, 0.5, 0.03])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("Time on 1 Dec 2023 (HH:MM UTC) of\nmax. range estimates")

    # Format colorbar ticks as datetime strings (HH:MM)
    tick_times = np.linspace(times.min(), times.max(), 5)
    cbar.set_ticks(tick_times)
    cbar.set_ticklabels(
        [str(np.datetime64(int(t), "us")).split("T")[1][:5] for t in tick_times]
    )


def plot_trajectory_cartesian(
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

    ax, m = plot_study_area(
        bathy,
        lonvec,
        latvec,
        equipment_df=equip_locations,
        active_turbine=active_turbine,
        bounds=bbox_outer,
        ax=ax,
        meridians=0.25,
        parallels=0.25,
        shallowest_contour_depth=-1.0,
        levelsc=np.arange(-100, 1, 5),
        show_legend=True,
        legend_bbox=(0.5, 0.934),
        legend_ncol=1,
        parallel_labels=[0, 1, 0, 0],
        aspect=0.25,
        projected_ticks=True,
        tick_spacing_km=5,
    )

    # Plot estimated whale positions
    _whale_position_brackets(ax, m, equip_locations, whale_data)

    # Add panel label
    add_panel_label(ax, "d")

    # Add colorbar for bearing times
    _add_bearing_colorbar(subfig, times)

    return fig


def plot_trajectory_phase(
    whale_data: DataFrame,
    subfig: plt.Figure | None = None,
) -> Figure:
    """Bearing vs. range phase-space trajectory plot.

    Points connected chronologically with a time-colored line; uncertainty
    brackets show range bounds at each bearing estimate.
    """
    if subfig is not None:
        ax = subfig.add_subplot(111)
        fig = subfig.figure
    else:
        fig, ax = plt.subplots(figsize=(5, 4))

    valid = whale_data.filter(pl.col("whale_range_km") < RLIM[1]).sort("timestamp")
    bearings_deg = valid["mean_bearing"].to_numpy()
    range_min_km = valid["whale_range_km_25pct"].to_numpy()
    ranges_max_km = valid["whale_range_km"].to_numpy()
    ranges_mean = np.mean([range_min_km, ranges_max_km], axis=0)

    t_arr = np.array(
        [np.datetime64(i, "us").astype("int64") for i in valid["timestamp"]]
    )
    norm = Normalize(vmin=t_arr.min(), vmax=t_arr.max())

    # Uncertainty brackets: vertical shaft + horizontal caps, one per estimate
    cap_deg = 1.0
    plot_colors = colorbar_cmap(norm(t_arr))
    for i, color in enumerate(plot_colors):
        ax.vlines(
            bearings_deg[i],
            range_min_km[i],
            ranges_max_km[i],
            color=color,
            linewidth=2,
            zorder=15,
        )
        ax.hlines(
            range_min_km[i],
            bearings_deg[i] - cap_deg,
            bearings_deg[i] + cap_deg,
            color=color,
            linewidth=2,
            zorder=15,
        )
        ax.hlines(
            ranges_max_km[i],
            bearings_deg[i] - cap_deg,
            bearings_deg[i] + cap_deg,
            color=color,
            linewidth=2,
            zorder=15,
        )

    ax.set_xlabel("DOA (°T)")
    ax.set_ylabel("Range (km)")
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_ylim(RLIM)
    ax.grid(True, alpha=0.3)

    if subfig is not None:
        _add_bearing_colorbar(subfig, t_arr)

    add_panel_label(ax, "d")

    return fig


def plot_trajectory_polar(
    whale_data: DataFrame,
    subfig: plt.Figure | None = None,
    cap_km: float = 0.75,
) -> Figure:
    """Alternative to create_map: whale position brackets on a polar (range/bearing) plot.

    Displays the southern angular hemisphere only (bearings 90°–270°) using
    compass arithmetic (0° = N, increasing clockwise).
    """
    if subfig is not None:
        ax = subfig.add_subplot(111, projection="polar")
        fig = subfig.figure
    else:
        fig, ax = plt.subplots(figsize=(5, 4), subplot_kw={"projection": "polar"})

    # Compass convention: 0° at North, clockwise
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    # Southern hemisphere: East (90°) → South (180°) → West (270°)
    ax.set_thetalim(np.deg2rad(150), np.deg2rad(210))
    ax.set_ylim(RLIM)

    valid = whale_data.filter(pl.col("whale_range_km") < RLIM[1]).sort("timestamp")
    bearings_rad = np.deg2rad(valid["mean_bearing"].to_numpy())
    range_min_km = valid["whale_range_km_25pct"].to_numpy()
    ranges_max_km = valid["whale_range_km"].to_numpy()
    ranges_mean = np.mean([range_min_km, ranges_max_km], axis=0)

    t_arr = np.array(
        [np.datetime64(i, "us").astype("int64") for i in valid["timestamp"]]
    )
    norm = Normalize(vmin=t_arr.min(), vmax=t_arr.max())
    plot_colors = colorbar_cmap(norm(t_arr))

    # for i, color in enumerate(plot_colors):

    for i, color in enumerate(plot_colors):
        brg = bearings_rad[i]
        ax.plot(
            brg,
            ranges_mean[i],
            marker="o",
            color=color,
            markersize=5,
            zorder=50,
        )
        r_near, r_far = range_min_km[i], ranges_max_km[i]
        # Cap half-angle: arc of length cap_km at the mean radius
        cap_angle = cap_km / ((r_near + r_far) / 2)

        # Radial shaft
        ax.plot(
            [brg, brg], [r_near, r_far], color="k", alpha=0.25, linewidth=1.5, zorder=15
        )
        # Near and far caps as arcs at constant radius
        for r_cap in (r_near, r_far):
            cap_thetas = np.linspace(brg - cap_angle, brg + cap_angle, 10)
            ax.plot(
                cap_thetas,
                np.full_like(cap_thetas, r_cap),
                color="k",
                alpha=0.25,
                linewidth=1.5,
                zorder=15,
            )

    ax.set_thetagrids([150, 165, 180, 195, 210])
    ax.set_rticks([5, 10, 15, 20])
    ax.set_rlabel_position(150)
    ax.text(
        np.deg2rad(135),
        (RLIM[0] + RLIM[1]) / 2,
        "Range (km)",
        rotation=-60,
        ha="center",
        va="center",
    )

    if subfig is not None:
        _add_bearing_colorbar(subfig, t_arr)

    add_panel_label(ax, "d")

    return fig


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
    ax.axhline(180, color="k", linestyle="--", linewidth=0.75, zorder=5)
    ax.tick_params(labelbottom=False)
    ax.set_ylabel("DOA (°T)")
    ax.legend(["Estimates", "Segmented linear fits"], loc="upper right", framealpha=1.0)
    add_panel_label(ax, "a")

    ax = axes[1]
    ax.scatter(
        range_df["timestamp"],
        range_df["slope_deg_s"] * 3600,
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
    ax.axhline(0, color="k", linestyle="--", linewidth=0.75, zorder=5)
    add_panel_label(ax, "b")

    ax.grid()
    ax.tick_params(labelbottom=False)
    ax.set_ylabel("Angular velocity (°/h)")

    ax = axes[2]
    range_df = range_df.filter(pl.col("whale_range_km") < RLIM[1]).sort("timestamp")
    max_range = range_df["whale_range_km"].to_numpy()
    min_range = range_df["whale_range_km_25pct"].to_numpy()
    mean_range = np.mean([max_range, min_range], axis=0)
    vert_err = np.array([max_range - mean_range, mean_range - min_range])
    ax.scatter(
        range_df["timestamp"],
        range_df["whale_range_km"],
        marker="v",
        color="tab:blue",
        s=20,
        zorder=15,
        label="10 km/h",
    )
    ax.scatter(
        range_df["timestamp"],
        range_df["whale_range_km_25pct"],
        marker="^",
        color="tab:red",
        s=20,
        zorder=15,
        label="2.5 km/h",
    )
    ax.errorbar(
        range_df["timestamp"],
        mean_range,
        yerr=vert_err,
        fmt="none",
        color="k",
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
    ax.set_ylim(RLIM)
    ax.grid()
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Max. range (km)")
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.legend(loc="upper right", framealpha=1.0)
    add_panel_label(ax, "c")

    host.align_ylabels(axes)

    return fig


def plot_whale_map(
    bathy_data: Path,
    sensor_data: Path,
    turbine_data: Path,
    active_turbine_name: str,
    whale_bearings: Path,
    whale_ranges: Path,
) -> Figure:
    """Create a map figure showing whale tracking results."""
    (
        (bathy, lonvec, latvec),
        equip_locations,
        turbine_locations,
        _,
        range_df,
    ) = load_map_data(
        bathy_data, sensor_data, turbine_data, whale_bearings, whale_ranges
    )
    active_turbine = get_active_turbine_info(turbine_locations, active_turbine_name)
    # bearings, times, ranges = _process_whale_bearings(whale_df, range_df)
    bearings = range_df["mean_bearing"].to_numpy()
    times = np.array(
        [np.datetime64(t, "us").astype("int64") for t in range_df["timestamp"]]
    )
    return plot_trajectory_cartesian(
        bathy,
        lonvec,
        latvec,
        equip_locations,
        range_df,
        active_turbine,
        BBOX_OUTER,
        bearings,
        times,
    )


def plot_whale_tracking(
    bathy_data: Path,
    sensor_data: Path,
    turbine_data: Path,
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
        whale_bearings: Path to whale bearing data CSV
        whale_ranges: Path to whale range data CSV
        time_ranges: List of (start, end) time tuples for highlighting on plots

    Returns:
        Figure with the whale tracking maps and data plots
    """
    _, _, _, whale_df, range_df = load_map_data(
        bathy_data, sensor_data, turbine_data, whale_bearings, whale_ranges
    )

    fig = plt.figure(figsize=(8, 4))
    subfigs = fig.subfigures(nrows=1, ncols=2, wspace=-0.15, width_ratios=[0.6, 0.4])
    plot_whale_data(whale_df, range_df, time_ranges, subfig=subfigs[0])

    plot_trajectory_phase(
        range_df,
        subfig=subfigs[1],
    )

    return fig


def _whale_position_brackets(
    ax: Axes,
    m: Basemap,
    equip_locations: DataFrame,
    whale_data: DataFrame,
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
    range_min_m = valid["whale_range_km_25pct"].to_numpy() * 1000
    ranges_max_m = valid["whale_range_km"].to_numpy() * 1000
    cap_m = cap_km * 1000

    times = np.array(
        [np.datetime64(i, "us").astype("int64") for i in valid["timestamp"]]
    )
    norm = Normalize(vmin=times.min(), vmax=times.max())
    colors = colorbar_cmap(norm(times))

    # Radial unit vector (bearing: sin=E, cos=N in map coords)
    radial_x = np.sin(bearings_rad)
    radial_y = np.cos(bearings_rad)
    # Perpendicular unit vector (90° CCW from radial)
    perp_x = np.cos(bearings_rad)
    perp_y = -np.sin(bearings_rad)

    x0, y0 = x_center[0], y_center[0]
    for i, color in enumerate(colors):
        x_near = x0 + range_min_m[i] * radial_x[i]
        y_near = y0 + range_min_m[i] * radial_y[i]
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
