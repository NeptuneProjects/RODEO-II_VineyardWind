import logging
import tomllib
from pathlib import Path

import bathyreq
import h5py
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class BathymetryConfig(BaseModel):
    """Configuration for bathymetry data."""

    bounds: list[list[float]] = Field(
        default=[[-71.0, -68.999], [39.5, 41.601]],
        description="Bounding box as [[lon_min, lon_max], [lat_min, lat_max]].",
    )
    output_path: Path = "data/bathy/bathy.hdf5"
    xres: int = 400
    yres: int = 400


def bathy_etl(config: BathymetryConfig) -> None:
    bounds = config.bounds

    logger.info(f"Bounding box read from configuration.")

    logger.info(
        f"Downloading bathymetry data for bounding box: {bounds}"
        f" with resolution {config.xres}x{config.yres}"
    )
    data, lonvec, latvec = download_bathy(bounds, config.xres, config.yres)
    logger.info("Bathymetry data downloaded successfully.")

    save_bathy_data(data, lonvec, latvec, fname=config.output_path)
    logger.info(f"Saving bathymetry data to {config.output_path}")


def download_bathy(
    bounds: list[list[float]], xres: int = 400, yres: int = 400
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Download bathymetry data from NCEI.

    Args:
        bounds: Bounding box as [[lon_min, lon_max], [lat_min, lat_max]].
        xres: X resolution of the bathymetry data.
        yres: Y resolution of the bathymetry data.
    Returns:
        Tuple of numpy arrays containing bathymetry data, longitude vector,
        and latitude vector.
    """
    req = bathyreq.BathyRequest()
    return req.get_area(longitude=bounds[0], latitude=bounds[1], size=[xres, yres])


def save_bathy_data(
    data: npt.NDArray[np.float64],
    lonvec: npt.NDArray[np.float64],
    latvec: npt.NDArray[np.float64],
    fname: Path = Path.cwd() / "data" / "bathy.hdf5",
) -> None:
    """Save bathymetry data to a local file."""
    with h5py.File(fname, "w") as f:
        f.create_dataset("data", data=data)
        f.create_dataset("lonvec", data=lonvec)
        f.create_dataset("latvec", data=latvec)
