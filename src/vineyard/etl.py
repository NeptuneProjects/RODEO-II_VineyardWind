import logging
import tomllib
from pathlib import Path

import bathyreq
import h5py
import numpy as np
import numpy.typing as npt
import polars as pl
import pymap3d as pm
from geopy.distance import geodesic
from pydantic import BaseModel, Field
from tritonoa.data.inventory import Inventory
from tritonoa.data.signal import SignalParams
from tritonoa.data.time import ClockParameters


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


class ETLConfig(BaseModel):
    """Configuration for the ETL process."""

    bathymetry: BathymetryConfig = BathymetryConfig()
    distances: Path = "data/distances.csv"
    sensor_data: Path | None = None
    turbine_data: Path | None = None
    source_pile: str | None = None
    inventory_config: Path = "config/inventory.toml"
    inventory_dir: Path = "data/acoustic"
    ref_mooring: str = "VLA1"


def modify_sensor_table(
    sensor_data_path: Path,
    turbine_path: Path,
    source_pile: str,
    ref_mooring: str = "VLA1",
) -> None:
    """Load sensor positions from equipment config and compute ENU coordinates."""
    df = pl.read_csv(sensor_data_path)

    if not (
        "easting" in df.columns
        and "northing" in df.columns
        and "ref_lat" in df.columns
        and "ref_lon" in df.columns
    ):
        logger.info("Computing ENU coordinates for sensor positions...")
        lat0, lon0 = (
            df.filter(pl.col("mooring_name") == ref_mooring)
            .select(["latitude", "longitude"])
            .row(0)
        )

        df = df.with_columns(
            pl.struct("latitude", "longitude")
            .map_elements(
                lambda cols: dict(
                    zip(
                        ("easting", "northing"),
                        compute_northing_easting(
                            cols["latitude"], cols["longitude"], lat0, lon0
                        ),
                    )
                ),
                return_dtype=pl.Struct(
                    [
                        pl.Field("easting", pl.Float64),
                        pl.Field("northing", pl.Float64),
                    ]
                ),
            )
            .alias("result"),
            pl.lit(lat0, dtype=pl.Float64).alias("ref_lat"),
            pl.lit(lon0, dtype=pl.Float64).alias("ref_lon"),
        ).unnest("result")

    if not "dist_to_pile_m" in df.columns:
        logger.info("Computing distance to pile for sensor positions...")
        turbine_df = pl.read_csv(turbine_path)
        pile_location = (
            turbine_df.filter(pl.col("Turbine") == source_pile)
            .select(["lat", "lon"])
            .row(0)
        )

        df = df.with_columns(
            pl.struct("latitude", "longitude")
            .map_elements(
                lambda cols: geodesic(
                    (cols["latitude"], cols["longitude"]), pile_location
                ).meters
            )
            .alias("dist_to_pile_m")
        )

    df.write_csv(sensor_data_path)


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


def compute_distances(sensor_data_path: Path, output_path: Path) -> pl.DataFrame:
    """
    Compute pairwise distances between 3DVHA, VLA1, and VLA2.

    Args:
        sensor_data_path: Path to the equipment CSV file
        output_path: Path to save the distance lookup table

    Returns:
        Polars DataFrame with distance lookup table
    """
    # Read equipment data
    df = pl.read_csv(sensor_data_path)

    # Filter for the three equipment of interest
    equipment_names = ["3DVHA", "VLA1", "VLA2"]
    df_filtered = df.filter(pl.col("mooring_name").is_in(equipment_names))

    # Create coordinate pairs (latitude, longitude)
    coords = {}
    for row in df_filtered.iter_rows(named=True):
        coords[row["mooring_name"]] = (row["latitude"], row["longitude"])

    # Compute pairwise distances and store in lists
    from_equipment = []
    to_equipment = []
    distance_meters = []

    for i, name1 in enumerate(equipment_names):
        for name2 in equipment_names[i + 1 :]:
            if name1 in coords and name2 in coords:
                distance = geodesic(coords[name1], coords[name2])

                for src, dst in [(name1, name2), (name2, name1)]:
                    from_equipment.append(src)
                    to_equipment.append(dst)
                    distance_meters.append(distance.meters)

    distance_df = pl.DataFrame(
        {
            "from_equipment": from_equipment,
            "to_equipment": to_equipment,
            "distance_meters": distance_meters,
        }
    )

    distance_df.write_csv(output_path)
    return distance_df


def compute_northing_easting(lat, lon, lat0, lon0):
    """Convert latitude/longitude to local ENU coordinates."""
    easting, northing, _ = pm.geodetic2enu(lat, lon, 0, lat0, lon0, 0)
    return easting, northing


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


def inventory_acoustic_data(config_file: Path) -> None:
    """Build inventories for raw acoustic data."""
    with open(config_file, "rb") as f:
        config = tomllib.load(f)

    for key in config:
        logging.info(f"Building inventory for {key}...")

        data_cfg = config[key]["data"]
        clock_cfg = config[key].get("clock", None)
        hyd_cfg = config[key].get("hydrophone", None)

        if clock_cfg is None:
            clock_params = ClockParameters()
        else:
            time_check_0 = np.datetime64(clock_cfg["time_check_0"])
            time_check_1 = np.datetime64(clock_cfg["time_check_1"])
            clock_params = ClockParameters(
                time_check_0=time_check_0,
                time_check_1=time_check_1,
                offset_0=clock_cfg["offset_0"],
                offset_1=clock_cfg["offset_1"],
            )

        if hyd_cfg is None:
            signal_params = SignalParams()
        else:
            signal_params = SignalParams(
                gain=hyd_cfg["fixed_gain"],
                sensitivity=hyd_cfg["sensitivity"],
            )

        inv = Inventory()
        inv.build(
            dataset_path=data_cfg["directory"],
            glob_pattern=data_cfg["glob_pattern"],
            clock_params=clock_params,
            conditioner=signal_params,
            file_format=data_cfg.get("file_format", None),
        )
        savepath = Path(data_cfg["destination"])
        inv.save(savepath)

        logging.info(f"Inventory saved to {savepath.resolve()}")


def run_etl(config: ETLConfig) -> None:
    bathy_etl(config.bathymetry)
    compute_distances(config.sensor_data, config.distances)
    inventory_acoustic_data(config.inventory_config)
    modify_sensor_table(
        config.sensor_data, config.turbine_data, config.source_pile, config.ref_mooring
    )


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
