from pathlib import Path

import cmasher as cmr
import cmocean as cmo
import matplotlib.colors as colors
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from mpl_toolkits.basemap import Basemap
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.signal as signal
from tritonoa.data.stream import DataStream

from vineyard import config

JASA_STYLE = Path(config.get_path("jasa_style"))

plt.style.use(JASA_STYLE)

savefig_kwargs = {
    "bbox_inches": "tight",
    "dpi": 300,
    "facecolor": "white",
}


def draw_polygon(
    m: Basemap,
    longitudes: list[float],
    latitudes: list[float],
    fill: bool = False,
    alpha: float = 1.0,
    edgecolor: str = "red",
    linewidth: float = 1.0,
    zorder: int = 10,
):
    x, y = m(longitudes, latitudes)
    polygon = Polygon(
        xy=list(zip(x, y)),
        closed=True,
        fill=fill,
        alpha=alpha,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
    )
    return polygon


def find_closest_contour_index(levels: npt.NDArray[np.float64], value: float) -> int:
    """
    Find the index of the contour level that is closest to a given value.

    Args
    levels: The contour levels.
    value: The value to find the closest contour level for.

    Returns
        The index of the closest contour level.
    """
    return np.argmin(np.abs(levels - value))


def plot_bathy(
    data: npt.NDArray[np.float64],
    lonvec: npt.NDArray[np.float64],
    latvec: npt.NDArray[np.float64],
    m: Basemap,
    ax: plt.Axes | None = None,
    shallowest_contour_depth: float = 0.0,
) -> tuple[plt.contourf, plt.Axes]:

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

    levelsf = np.arange(-100, 10, 5)
    levelsc = np.arange(-100, 1, 5)
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
    idx = find_closest_contour_index(levelsc, shallowest_contour_depth)
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
        CS_water, inline=True, fmt="%1.0f", fontsize=plt.rcParams["font.size"] - 2
    )
    return im, ax


def plot_shru_pectrograms(
    ds: DataStream,
    nperseg: int = 128,
    noverlap: float = 64,
    nfft: int | None = 2**12,
    fmin: float | None = None,
    fmax: float | None = None,
    vmin: float = 70.0,
    vmax: float = 130.0,
    figsize: tuple[float] = (8, 6),
    xlabel: str = "Time (s)",
    ylabel: str = "Frequency (Hz)",
    title: str = None,
) -> plt.Figure:
    fs = ds.stats.sampling_rate

    if nfft is None:
        nfft = nperseg

    fig, axs = plt.subplots(
        ds.num_channels, 1, figsize=figsize, gridspec_kw={"hspace": 0.3}
    )
    fig.suptitle(title)

    channels = np.arange(ds.num_channels)

    for i, channel in enumerate(channels):
        f, t, Sxx = signal.spectrogram(
            ds.data[i],
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
        )

        if fmin and fmax:
            Sxx = Sxx[(f >= fmin) & (f <= fmax), :]
            f = f[(f >= fmin) & (f <= fmax)]
        elif fmin:
            Sxx = Sxx[f >= fmin, :]
            f = f[f >= fmin]
        elif fmax:
            Sxx = Sxx[f <= fmax, :]
            f = f[f <= fmax]

        if vmin is None:
            vmin = Sxx.min()
        if vmax is None:
            vmax = Sxx.max()

        ax = axs[i]
        im = plot_spectrogram(f, t, Sxx, ax=ax, vmin=vmin, vmax=vmax)

        ax.set_title(f"Channel {channel}", fontsize=10, ha="left", x=0)
        if channel == channels[-1]:
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
        else:
            ax.set_xticklabels([])
            ax.set_xlabel("")

        cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(f"PSD ($\\mathrm{{{ds.stats.units}}}^2 / \\mathrm{{Hz}}$)")

    if title:
        fig.suptitle(title, fontsize=12, y=0.95)
    return fig


def plot_spectrogram(f, t, Sxx, ax=None, vmin=None, vmax=None) -> plt.Axes:
    if ax is None:
        ax = plt.gca()
    # return ax.pcolormesh(t, f, 10 * np.log10(Sxx), cmap="inferno", vmin=vmin, vmax=vmax)
    extent = (t[0], t[-1], f[0], f[-1])
    return ax.imshow(
        10 * np.log10(Sxx),
        extent=extent,
        cmap="inferno",
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        origin="lower",
        interpolation="none",
    )


# def plot_spectrogram(
#     ds: DataStream,
#     channel=0,
#     nfft: int = 2**15,
#     nperseg: int = 128,
#     noverlap: int = 64,
#     xlabel: str = "Time (s)",
#     ylabel: str = "Frequency (Hz)",
#     title: str = None,
#     vmin: float = 70,
#     vmax: float = 130,
#     ax=None,
# ) -> plt.Axes:
#     fs = ds.stats.sampling_rate

#     f, t, Sxx = signal.spectrogram(
#         ds.data[channel],
#         fs=fs,
#         noverlap=noverlap,
#         nperseg=nperseg,
#         nfft=nfft,
#     )

#     fig, ax = plt.subplots(figsize=(10, 5))
#     im = ax.pcolormesh(
#         t,
#         f,
#         10 * np.log10(Sxx),
#         shading="auto",
#         cmap="magma",
#         vmin=vmin,
#         vmax=vmax,
#     )
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)
#     ax.set_title(title)
#     cbar = fig.colorbar(im, ax=ax)
#     cbar.set_label("Power Spectral Density (dB re 1 $\\mu$Pa/Hz$^2$)")

#     return fig


def plot_study_area(
    bathy_data: npt.NDArray[np.float64],
    lonvec: npt.NDArray[np.float64],
    latvec: npt.NDArray[np.float64],
    das_df: pd.DataFrame,
    equipment_df: pd.DataFrame,
    turbines_df: pd.DataFrame,
    active_turbine: dict | None = None,
    sound_trap: dict | None = None,
    bounds: list[list[float]] | None = None,
    ax: plt.Axes | None = None,
    scale_bar: float = 1.0,
    shallowest_contour_depth: float = 0.0,
    legend_loc: str | None = None,
    meridians: float = 0.2,
    parallels: float = 0.2,
    meridian_labels: list[int] = [0, 0, 1, 0],
    parallel_labels: list[int] = [1, 0, 0, 0],
    inset: list[list[float]] | None = None,
    *args,
    **kwargs,
) -> plt.Axes:
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

    # fig = plt.figure(figsize=figsize)
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

    _, ax = plot_bathy(
        bathy_data,
        lonvec=lonvec,
        latvec=latvec,
        m=m,
        ax=ax,
        shallowest_contour_depth=shallowest_contour_depth,
    )

    ax.scatter(
        *m(turbines_df["longitude"], turbines_df["latitude"]),
        marker="h",
        c="yellow",
        edgecolors="k",
        # linewidth=1,
        # s=150,
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
            # s=150,
            zorder=30,
            label=active_turbine["label"],
        )

    ax.plot(
        *m(das_df["longitude"], das_df["latitude"]),
        c="tab:red",
        linewidth=2,
        label="MVCO DAS Array",
        zorder=20,
    )
    ax.scatter(
        *m(-70.566595, 41.324978),
        marker="d",
        c="tab:red",
        edgecolors="k",
        zorder=20,
        label="WHOI Air-Sea Interation Tower",
    )
    ax.scatter(
        *m(equipment_df["longitude"], equipment_df["latitude"]),
        marker="v",
        c="tab:green",
        edgecolors="k",
        # linewidth=1,
        zorder=20,
        label="VLA",
    )

    if inset:
        polygon = draw_polygon(
            m,
            longitudes=[
                inset[0][0] - 0.01,
                inset[0][1],
                inset[0][1],
                inset[0][0] - 0.01,
            ],
            latitudes=[
                inset[1][0] - 0.01,
                inset[1][0] - 0.01,
                inset[1][1],
                inset[1][1],
            ],
            fill=False,
            edgecolor="b",
            linewidth=1,
        )
        ax.add_patch(polygon)
    if sound_trap is not None:
        ax.scatter(
            *m(sound_trap["longitude"], sound_trap["latitude"]),
            marker="s",
            c="tab:blue",
            edgecolors="k",
            # linewidth=1,
            # s=150,
            zorder=30,
            label=sound_trap["label"],
        )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    if legend_loc:
        leg = ax.legend(
            facecolor="white",
            edgecolor="black",
            loc="upper left",
            bbox_to_anchor=(-0.7, -0.05),
            ncol=3,
        )
        leg.get_frame().set_alpha(None)

    if kwargs.get("type") == "inset":
        ax.text(
            0.02,
            0.982,
            "Inset",
            ha="left",
            va="top",
            transform=ax.transAxes,
            bbox=dict(facecolor="white", edgecolor="black"),
        )

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

    return ax
