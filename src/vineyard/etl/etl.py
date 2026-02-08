import logging
from pathlib import Path

from pydantic import BaseModel

from vineyard.etl.bathy import BathymetryConfig, bathy_etl
from vineyard.etl.distances import compute_distances

logger = logging.getLogger(__name__)


class ETLConfig(BaseModel):
    """Configuration for the ETL process."""

    bathymetry: BathymetryConfig = BathymetryConfig()
    distances: Path = "data/distances.csv"
    sensor_data: Path | None = None


def run_etl(config: ETLConfig) -> None:
    bathy_etl(config.bathymetry)
    compute_distances(config.sensor_data, config.distances)
