#! /usr/bin/env python3
"""Run workflows with a given configuration."""

import tomllib
from argparse import ArgumentParser
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vineyard.etl import ETLConfig, run_etl
from vineyard.figures import PlottingConfig, make_figures
from vineyard.process import ProcessConfig, process_data
from vineyard.tdoa import LocalizationConfig, localize


class MetadataConfig(BaseModel):
    """Configuration for sensor metadata."""

    sensor_data: Path = "data/sensors.csv"
    turbine_data: Path = "data/turbines.csv"
    source_pile: str = "AN36"
    calibration_dir: Path = "data/acoustic/calibration/"


class Config(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    metadata_config: MetadataConfig = Field(alias="metadata")
    etl_config: ETLConfig = Field(alias="etl")
    process_config: ProcessConfig = Field(alias="process")
    tdoa_config: LocalizationConfig = Field(alias="tdoa")
    plotting_config: PlottingConfig = Field(alias="plotting")

    @model_validator(mode="after")
    def sync_attributes(self) -> "Config":
        """Ensure that attributes across different config sections are consistent."""
        # ETL sync
        if self.etl_config.sensor_data is None:
            self.etl_config.sensor_data = self.metadata_config.sensor_data
        if self.etl_config.turbine_data is None:
            self.etl_config.turbine_data = self.metadata_config.turbine_data
        if self.etl_config.source_pile is None:
            self.etl_config.source_pile = self.metadata_config.source_pile

        # Processing sync
        if self.process_config.inventory_path is None:
            self.process_config.inventory_path = self.etl_config.inventory_dir
        if self.process_config.calibration_dir is None:
            self.process_config.calibration_dir = self.metadata_config.calibration_dir

        # Plotting sync
        if self.plotting_config.calibration_dir is None:
            self.plotting_config.calibration_dir = self.metadata_config.calibration_dir

        if self.plotting_config.map is not None:
            if self.plotting_config.map.bathy_data is None:
                self.plotting_config.map.bathy_data = (
                    self.etl_config.bathymetry.output_path
                )
            if self.plotting_config.map.sensor_data is None:
                self.plotting_config.map.sensor_data = self.metadata_config.sensor_data
            if self.plotting_config.map.turbine_data is None:
                self.plotting_config.map.turbine_data = (
                    self.metadata_config.turbine_data
                )
            if self.plotting_config.map.active_turbine_name is None:
                self.plotting_config.map.active_turbine_name = (
                    self.metadata_config.source_pile
                )
            if self.plotting_config.map.whale_bearings is None:
                self.plotting_config.map.whale_bearings = (
                    self.tdoa_config.localization_file
                )

        if self.plotting_config.signal_template is not None:
            if self.plotting_config.signal_template.inventory_dir is None:
                self.plotting_config.signal_template.inventory_dir = (
                    self.etl_config.inventory_dir
                )

        return self


def run_workflow(
    command: Literal["run", "etl", "process", "localize", "plot"],
    config_path: Path,
    show: bool = False,
) -> Config:
    """Run the ETL workflow with the given config file.

    Args:
        command: The workflow command to execute. 'run' executes both ETL and
            processing steps, 'etl' executes only the ETL step, 'process'
            executes only the processing step, 'localize' executes only
            the TDOA estimation step, and 'plot' generates the figures.
        config_path: Path to the TOML configuration file.

    Returns:
        The loaded ConfigModel instance.
    """

    COMMAND_REGISTRY = {
        "run": lambda config: (
            run_etl(config.etl_config),
            process_data(config.process_config),
            localize(config.tdoa_config),
        ),
        "etl": lambda config: run_etl(config.etl_config),
        "process": lambda config: process_data(config.process_config),
        "localize": lambda config: localize(config.tdoa_config),
        "plot": lambda config: make_figures(config.plotting_config, show=show),
    }

    with open(config_path, "rb") as f:
        config = Config(**tomllib.load(f))

    if command not in COMMAND_REGISTRY:
        raise ValueError(f"Invalid command: {command}")

    COMMAND_REGISTRY[command](config)

    return config


def main() -> None:
    """Entry point for the workflow command-line tool."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["run", "etl", "process", "localize", "plot"],
        help=(
            "The workflow command to execute. 'run' executes both ETL and "
            "processing steps, 'etl' executes only the ETL step, 'process' "
            "executes only the processing step, 'localize' executes only "
            "the TDOA estimation step, and 'plot' generates the figures."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path.cwd() / "config" / "config.toml",
        help=(
            "Path to the configuration file (TOML format). Default is "
            "'./config/config.toml'."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Whether to display the generated figures after saving.",
    )
    args = parser.parse_args()
    run_workflow(args.command, args.config, show=args.show)


if __name__ == "__main__":
    main()
