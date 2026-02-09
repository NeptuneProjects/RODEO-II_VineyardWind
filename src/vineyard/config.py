from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vineyard.etl import ETLConfig
from vineyard.process import ProcessConfig

SENSORS = {
    "3dvha": {
        "metadata": {
            "channel_names": [
                "3DVHA Front Hydrophone",
                "3DVHA Right Hydrophone",
                "3DVHA Left Hydrophone",
                "3DVHA Back Hydrophone",
                "3DVHA Particle Motion X",
                "3DVHA Particle Motion Y",
                "3DVHA Particle Motion Z",
                "3DVHA Omni Hydrophone",
            ]
        },
    },
    "vla1": {
        "metadata": {
            "channel_names": [
                "VLA1 Channel 1",
                "VLA1 Channel 2",
                "VLA1 Channel 3",
                "VLA1 Channel 4",
            ]
        },
    },
    "vla2": {
        "metadata": {
            "channel_names": [
                "VLA2 Channel 1",
                "VLA2 Channel 2",
                "VLA2 Channel 3",
                "VLA2 Channel 4",
            ]
        },
    },
}


class MetadataConfig(BaseModel):
    """Configuration for sensor metadata."""

    sensor_data: Path = "data/sensors.csv"
    turbine_data: Path = "data/turbines.csv"


class ConfigModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    metadata_config: MetadataConfig = Field(alias="metadata")
    etl_config: ETLConfig = Field(alias="etl")
    process_config: ProcessConfig = Field(alias="process")

    @model_validator(mode="after")
    def sync_attributes(self) -> "ConfigModel":
        """Set etl_config.sensor_data from metadata_config.sensor_data."""
        if self.etl_config.sensor_data is None:
            self.etl_config.sensor_data = self.metadata_config.sensor_data
        if self.process_config.inventory_path is None:
            self.process_config.inventory_path = self.etl_config.inventory_dir
        return self
