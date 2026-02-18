"""Map figure creation and related utilities."""

import logging
from collections.abc import Sequence
from pathlib import Path

import cmasher as cmr
import cmocean as cmo
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.basemap import Basemap
from numpy.typing import NDArray
from polars import DataFrame

import vineyard.readers as readers
from vineyard.figures.common import add_panel_label, save_and_show_figure

# Bounding box constants for map figures
BBOX_INSET = [[-70.65, -70.249], [40.9, 41.201]]
BBOX_OUTER = [[-71.0, -68.999], [39.5, 41.601]]


def _add_bearing_colorbar(fig: Figure, ax, times: Sequence[np.datetime64]) -> None:
    """Add a colorbar showing the time mapping for bearing lines."""
    norm = Normalize(vmin=times.min(), vmax=times.max())
    sm = ScalarMappable(cmap=plt.cm.RdPu, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", shrink=0.7, pad=-0.05)
    cbar.set_label("Time on 1 Dec 2023 (HH:MM UTC)")

    # Format colorbar ticks as datetime strings (HH:MM)
    tick_times = np.linspace(times.min(), times.max(), 5)
    cbar.set_ticks(tick_times)
    cbar.set_ticklabels(
        [str(np.datetime64(int(t), "us")).split("T")[1][:5] for t in tick_times]
    )


def _add_bounding_box(
    ax, m: Basemap, bbox: list[list[float, float]], color: str = "red"
) -> None:
    """Add a bounding box rectangle to a map."""
    xy = m(bbox[0][0], bbox[1][0])
    xwidth = m(bbox[0][1], bbox[1][0])[0] - xy[0]
    yheight = m(bbox[0][0], bbox[1][1])[1] - xy[1]

    box = Rectangle(
        xy,
        xwidth,
        yheight,
        linewidth=1,
        edgecolor=color,
        facecolor="none",
    )
    ax.add_patch(box)


def create_map_panels(
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
    return create_maps(
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


def _create_context_inset(
    ax,
    bbox_outer: list[list[float, float]],
    bounds: list[list[float, float]] = [[-77.5, -62.5], [32.5, 47.5]],
) -> None:
    """Create a New England context inset map showing the study area location."""
    ax_inset = inset_axes(
        ax, width="25%", height="25%", loc="upper right", borderpad=0.5
    )
    m_inset = Basemap(
        projection="merc",
        llcrnrlon=bounds[0][0],
        llcrnrlat=bounds[1][0],
        urcrnrlon=bounds[0][1],
        urcrnrlat=bounds[1][1],
        resolution="i",
        ax=ax_inset,
    )
    m_inset.drawcoastlines(linewidth=0.5, color="black")
    m_inset.fillcontinents(color="lightgray", lake_color="white")

    # Draw box showing main plot location
    _add_bounding_box(ax_inset, m_inset, bbox_outer, color="blue")


def _create_inner_map_panel(
    ax,
    bathy: dict,
    lonvec: Sequence[float],
    latvec: Sequence[float],
    equip_locations: DataFrame,
    turbine_locations: DataFrame,
    active_turbine: dict,
    bbox_inset: list[list[float, float]],
) -> None:
    """Create the inner (detailed) map panel showing turbines and equipment."""
    ax, _ = _plot_study_area(
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
        parallel_labels=[0, 1, 0, 0],
    )

    # Add panel label
    add_panel_label(ax, "b")


def _create_outer_map_panel(
    ax,
    bathy: dict,
    lonvec: Sequence[float],
    latvec: Sequence[float],
    equip_locations: DataFrame,
    active_turbine: dict,
    bbox_outer: list[list[float, float]],
    bbox_inset: list[list[float, float]],
    bearings: Sequence[float],
) -> None:
    """Create the outer (regional) map panel with bearing lines and context."""
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

    # Add bounding box showing inset region
    _add_bounding_box(ax, m, bbox_inset, color="red")

    # Plot bearing lines with color mapping
    colors = plt.cm.RdPu(np.linspace(0.25, 1, len(bearings)))
    _plot_bearing_lines(ax, m, equip_locations, bearings, colors)

    # Add New England context inset
    _create_context_inset(ax, bbox_outer)

    # Add panel label
    add_panel_label(ax, "a")


def create_maps(
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
    """Create a two-panel map figure showing regional context and detailed study area.

    Args:
        bathy: Bathymetry data dictionary
        lonvec: Longitude vector for bathymetry
        latvec: Latitude vector for bathymetry
        equip_locations: DataFrame of equipment locations
        turbine_locations: DataFrame of turbine locations
        active_turbine: Dict with active turbine info (label, longitude, latitude)
        bbox_inset: Bounding box for inset (detailed) map [[lon_min, lon_max], [lat_min, lat_max]]
        bbox_outer: Bounding box for outer (regional) map
        bearings: Array of bearing angles
        times: Array of timestamps for bearing colormap

    Returns:
        Matplotlib figure with two map panels
    """
    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(7, 4),
        gridspec_kw={"wspace": 0.05},
        constrained_layout=True,
    )

    # Create outer (regional) map panel
    _create_outer_map_panel(
        axes[0],
        bathy,
        lonvec,
        latvec,
        equip_locations,
        active_turbine,
        bbox_outer,
        bbox_inset,
        bearings,
    )

    # Add colorbar for bearing times
    _add_bearing_colorbar(fig, axes[0], times)

    # Create inner (detailed) map panel
    _create_inner_map_panel(
        axes[1],
        bathy,
        lonvec,
        latvec,
        equip_locations,
        turbine_locations,
        active_turbine,
        bbox_inset,
    )

    return fig


def _get_active_turbine_info(turbine_locations: DataFrame, turbine_name: str) -> dict:
    """Extract active turbine information as a dict for plotting."""
    return {
        "label": f"Turbine {turbine_name}",
        "longitude": turbine_locations.filter(pl.col("Turbine") == turbine_name)
        .select("lon")
        .item(),
        "latitude": turbine_locations.filter(pl.col("Turbine") == turbine_name)
        .select("lat")
        .item(),
    }


def _load_map_data(
    bathy_data: Path,
    sensor_data: Path,
    turbine_data: Path,
    whale_bearings: Path | None = None,
) -> tuple[
    tuple[dict, Sequence[float], Sequence[float]],
    DataFrame,
    DataFrame,
    DataFrame | None,
]:
    """Load all data files needed for map creation.

    Returns:
        Tuple of (bathy_data, equip_locations, turbine_locations, whale_df)
        where bathy_data is (bathy, lonvec, latvec)
    """
    logging.info("Loading data files.")
    bathy, lonvec, latvec = readers.read_bathymetry(bathy_data)
    equip_locations = pl.read_csv(sensor_data)
    turbine_locations = pl.read_csv(turbine_data)
    whale_df = (
        pl.read_csv(whale_bearings).cast({"timestamp": pl.Datetime})
        if whale_bearings
        else None
    )

    return (bathy, lonvec, latvec), equip_locations, turbine_locations, whale_df


def _plot_bathy(
    data: NDArray[np.float64],
    lonvec: NDArray[np.float64],
    latvec: NDArray[np.float64],
    m: Basemap,
    ax: Axes | None = None,
    shallowest_contour_depth: float = 0.0,
    levelsf=np.arange(-100, 10, 5),
    levelsc=np.arange(-100, 1, 5),
) -> tuple[plt.contourf, Axes]:

    data[data > 0] = 0.1

    if ax is None:
        ax = plt.gca()

    # Create a modified colormap truncated for shallow water and gray for
    # positive values
    n_bins = 256
    colors_array = cmr.get_sub_cmap("cmo.deep_r", 0.5, 1.0)(np.linspace(0, 1, n_bins))
    colors_list = np.vstack((colors_array, np.array([0.8, 0.8, 0.8, 0.8])))
    custom_cmap = colors.ListedColormap(colors_list)

    vmin = data.min()
    vmax = max(data.max(), 0.1)  # Ensure positive range exists

    # Create boundaries with n_bins below zero, 1 above zero
    boundaries = np.linspace(vmin, 0, n_bins)
    boundaries = np.append(boundaries, vmax)

    # Create the BoundaryNorm
    norm = colors.BoundaryNorm(boundaries, custom_cmap.N)

    lonlon, latlat = np.meshgrid(lonvec, latvec)
    im = m.contourf(
        lonlon,
        latlat,
        np.flipud(data),
        cmap=custom_cmap,
        norm=norm,
        levels=levelsf,
        latlon=True,
        ax=ax,
    )
    idx = np.argmin(np.abs(levelsc - shallowest_contour_depth))
    CS_water = m.contour(
        lonlon,
        latlat,
        np.flipud(data),
        colors="k",
        levels=levelsc[0 : idx + 1],
        linewidths=0.5,
        latlon=True,
        ax=ax,
    )
    m.contour(
        lonlon,
        latlat,
        np.flipud(data),
        colors="k",
        levels=levelsc[idx:],
        linewidths=0.5,
        latlon=True,
        ax=ax,
    )
    ax.clabel(
        CS_water,
        inline=True,
        fmt=lambda x: f"{abs(x):.0f}",
        fontsize=plt.rcParams["font.size"] - 2,
    )
    return im, ax


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


def _plot_study_area(
    bathy_data: NDArray[np.float64],
    lonvec: NDArray[np.float64],
    latvec: NDArray[np.float64],
    equipment_df: pl.DataFrame | None = None,
    turbines_df: pl.DataFrame | None = None,
    active_turbine: dict | None = None,
    bounds: list[list[float]] | None = None,
    ax: Axes | None = None,
    scale_bar: int = 1,
    shallowest_contour_depth: float = 0.0,
    levelsf=np.arange(-100, 10, 5),
    levelsc=np.arange(-100, 1, 5),
    meridians: float = 0.2,
    parallels: float = 0.2,
    meridian_labels: list[int] = [0, 0, 1, 0],
    parallel_labels: list[int] = [1, 0, 0, 0],
    marker_size: int = 50,
    show_legend: bool = True,
) -> tuple[Axes, Basemap]:
    if bounds is None:
        llcrnrlat = np.min(latvec)
        urcrnrlat = np.max(latvec)
        llcrnrlon = np.min(lonvec)
        urcrnrlon = np.max(lonvec)
        bounds = np.array([[llcrnrlon, urcrnrlon], [llcrnrlat, urcrnrlat]])
    else:
        llcrnrlon = bounds[0][0]
        urcrnrlon = bounds[0][1]
        llcrnrlat = bounds[1][0]
        urcrnrlat = bounds[1][1]

    if ax is None:
        ax = plt.gca()

    m = Basemap(
        projection="tmerc",
        llcrnrlat=llcrnrlat,
        urcrnrlat=urcrnrlat,
        llcrnrlon=llcrnrlon,
        urcrnrlon=urcrnrlon,
        resolution="f",
        lon_0=np.mean(lonvec),
        lat_0=np.mean(latvec),
    )
    m.drawmeridians(
        np.arange(llcrnrlon, urcrnrlon, meridians), labels=meridian_labels, ax=ax
    )
    m.drawparallels(
        np.arange(llcrnrlat, urcrnrlat, parallels), labels=parallel_labels, ax=ax
    )
    xlim = m(np.array(bounds[0]), np.ones_like(bounds[0]) * np.mean(bounds[1]))[0]
    ylim = m(np.ones_like(bounds[0]) * np.mean(bounds[0]), np.array(bounds[1]))[1]

    _, ax = _plot_bathy(
        bathy_data,
        lonvec=lonvec,
        latvec=latvec,
        m=m,
        ax=ax,
        shallowest_contour_depth=shallowest_contour_depth,
        levelsf=levelsf,
        levelsc=levelsc,
    )

    if turbines_df is not None:
        ax.scatter(
            *m(turbines_df["lon"], turbines_df["lat"]),
            marker="h",
            c="none",
            edgecolors="k",
            linewidth=0.5,
            # s=marker_size,
            zorder=20,
            label="Turbines",
        )

    if active_turbine is not None:
        ax.scatter(
            *m(active_turbine["longitude"], active_turbine["latitude"]),
            marker="h",
            c="tab:orange",
            edgecolors="k",
            # linewidth=1,
            s=marker_size,
            zorder=30,
            label=active_turbine["label"],
        )

    if equipment_df is not None:
        ax.scatter(
            *m(equipment_df["longitude"], equipment_df["latitude"]),
            marker="v",
            c="tab:green",
            edgecolors="k",
            # linewidth=1,
            s=marker_size,
            zorder=20,
            label="Hydrophone Array",
        )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    if show_legend:
        leg = ax.legend(
            facecolor="white",
            edgecolor="black",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=3,
        )
        leg.get_frame().set_alpha(None)

    scalebar = AnchoredSizeBar(
        ax.transData,
        scale_bar * 1e3,
        f"{scale_bar:d} km",
        "lower right",
        pad=0.1,
        color="k",
        frameon=True,
        size_vertical=20 * scale_bar,
        zorder=50,
    )
    ax.add_artist(scalebar)

    return ax, m


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
