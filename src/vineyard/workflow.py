#! /usr/bin/env python3
"""Run workflows with a given configuration."""

import tomllib
from argparse import ArgumentParser
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vineyard.etl import ETLConfig, run_etl
from vineyard.evaluate import (
    EvaluationConfig,
    run_evaluation,
    run_pr_sweep,
)
from vineyard.figures import PlottingConfig, make_figures
from vineyard.process import ProcessConfig, detect_whale_calls, process_data
from vineyard.doa import localize_farfield
from vineyard.tdoa import LocalizationConfig


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
    evaluation_config: EvaluationConfig | None = Field(default=None, alias="evaluation")

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

        if self.plotting_config.experiment is not None:
            if self.plotting_config.experiment.bathy_data is None:
                self.plotting_config.experiment.bathy_data = (
                    self.etl_config.bathymetry.output_path
                )
            if self.plotting_config.experiment.sensor_data is None:
                self.plotting_config.experiment.sensor_data = (
                    self.metadata_config.sensor_data
                )
            if self.plotting_config.experiment.turbine_data is None:
                self.plotting_config.experiment.turbine_data = (
                    self.metadata_config.turbine_data
                )
            if self.plotting_config.experiment.active_turbine_name is None:
                self.plotting_config.experiment.active_turbine_name = (
                    self.metadata_config.source_pile
                )

        if self.plotting_config.whale_tracking is not None:
            if self.plotting_config.whale_tracking.whale_data is None:
                self.plotting_config.whale_tracking.whale_data = (
                    self.tdoa_config.localization_file
                )
            if self.plotting_config.whale_tracking.time_ranges is None:
                self.plotting_config.whale_tracking.time_ranges = (
                    self.process_config.time_ranges
                )

        if self.plotting_config.signal_template is not None:
            if self.plotting_config.signal_template.inventory_dir is None:
                self.plotting_config.signal_template.inventory_dir = (
                    self.etl_config.inventory_dir
                )

        if self.plotting_config.template_construction is not None:
            tc = self.plotting_config.template_construction
            # Sync paths
            if tc.strike_index_path is None:
                tc.strike_index_path = self.process_config.strike_config.strike_index
            if tc.strike_corr_path is None:
                tc.strike_corr_path = self.process_config.strike_config.strike_corr
            if tc.inventory_path is None:
                tc.inventory_path = (
                    self.process_config.inventory_path
                    / f"inventory_{tc.sensor_name}.csv"
                )
            # Sync timing parameters
            if tc.start_time is None:
                tc.start_time = self.process_config.start_time
            if tc.end_time is None:
                tc.end_time = self.process_config.end_time
            # Sync template parameters from process.template config
            if tc.buffer_start is None:
                tc.buffer_start = self.process_config.template_config.buffer_start
            if tc.buffer_end is None:
                tc.buffer_end = self.process_config.template_config.buffer_end
            if tc.window_size is None:
                tc.window_size = self.process_config.template_config.window_size
            if tc.taper_pc is None:
                tc.taper_pc = self.process_config.template_config.taper_pc
            if tc.dec_factor is None:
                tc.dec_factor = self.process_config.template_config.dec_factor
            if tc.filt_type is None:
                tc.filt_type = self.process_config.template_config.filt_type
            if tc.filt_freq is None:
                tc.filt_freq = self.process_config.template_config.filt_freq
            # Sync sensor-specific parameters (channel, ylim)
            for sensor in self.process_config.template_config.sensors:
                if sensor["name"] == tc.sensor_name:
                    if tc.channel is None:
                        tc.channel = sensor["channel"]
                    if tc.ylim is None:
                        tc.ylim = sensor.get("ylim")
                    break

        if self.plotting_config.denoising is not None:
            dn = self.plotting_config.denoising
            if dn.data_dir is None:
                dn.data_dir = self.process_config.denoise_config.denoised_data
            if dn.template_data is None:
                dn.template_data = (
                    self.process_config.whale_template_config.template_data
                )

        if self.plotting_config.correlation is not None:
            corr = self.plotting_config.correlation
            if corr.strike_window_size is None:
                corr.strike_window_size = (
                    self.process_config.template_config.window_size
                )

        if (
            self.evaluation_config is not None
            and self.evaluation_config.pr_sweep is not None
        ):
            ps = self.evaluation_config.pr_sweep
            wd = self.process_config.whale_detection_config
            if wd is not None:
                if ps.sensors is None:
                    ps.sensors = wd.sensors
                # Always align these with the detector so the sweep is comparable
                ps.denoised_channel = wd.channel
                if wd.filt_type is not None:
                    ps.filt_type = wd.filt_type
                if wd.filt_freq is not None:
                    ps.filt_freq = wd.filt_freq
            if ps.pc_data_dir is None:
                ps.pc_data_dir = self.evaluation_config.pc_data_dir
            ps.match_window_s = self.evaluation_config.time_window_s

        return self


def run_workflow(
    command: Literal["run", "etl", "process", "localize", "plot", "detect", "evaluate"],
    config_path: Path,
    show: bool = False,
    prsweep: bool = False,
) -> Config:
    """Run the ETL workflow with the given config file.

    Args:
        command: The workflow command to execute. 'run' executes both ETL and
            processing steps, 'etl' executes only the ETL step, 'process'
            executes only the processing step, 'localize' executes only
            the TDOA estimation step, 'plot' generates the figures, 'detect'
            runs only the whale call detection step, and 'evaluate' runs
            detection performance evaluation.
        config_path: Path to the TOML configuration file.

    Returns:
        The loaded ConfigModel instance.
    """

    COMMAND_REGISTRY = {
        "run": lambda config: (
            run_etl(config.etl_config),
            process_data(config.process_config),
            localize_farfield(config.tdoa_config),
        ),
        "etl": lambda config: run_etl(config.etl_config),
        "process": lambda config: process_data(config.process_config),
        "localize": lambda config: localize_farfield(config.tdoa_config),
        "plot": lambda config: make_figures(config.plotting_config, show=show),
        "detect": lambda config: detect_whale_calls(
            config.process_config.whale_detection_config,
            config.process_config.denoise_config.denoised_data,
        ),
        "evaluate": lambda config: run_evaluation(
            config.evaluation_config,
            config.process_config.time_ranges,
        ),
    }

    with open(config_path, "rb") as f:
        config = Config(**tomllib.load(f))

    if command not in COMMAND_REGISTRY:
        raise ValueError(f"Invalid command: {command}")

    COMMAND_REGISTRY[command](config)

    if prsweep:
        if command != "evaluate":
            raise ValueError("--prsweep is only valid with the 'evaluate' command")
        if (
            config.evaluation_config is None
            or config.evaluation_config.pr_sweep is None
        ):
            raise ValueError(
                "--prsweep requires [evaluation.pr_sweep] to be configured in config.toml"
            )
        run_pr_sweep(
            config.evaluation_config.pr_sweep,
            config.evaluation_config.annotations_file,
            config.process_config.time_ranges,
        )

    return config


def main() -> None:
    """Entry point for the workflow command-line tool."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["run", "etl", "process", "localize", "plot", "detect", "evaluate"],
        help=(
            "The workflow command to execute. 'run' executes both ETL and "
            "processing steps, 'etl' executes only the ETL step, 'process' "
            "executes only the processing step, 'localize' executes only "
            "the TDOA estimation step, 'plot' generates the figures, "
            "'detect' runs only the whale call detection step (denoised and "
            "raw channels if configured), and 'evaluate' runs detection "
            "performance evaluation."
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
    parser.add_argument(
        "--prsweep",
        action="store_true",
        help=(
            "When used with 'evaluate', run a full detection threshold sweep "
            "and save PR curve data to the path configured in [evaluation.pr_sweep]."
        ),
    )
    args = parser.parse_args()
    run_workflow(args.command, args.config, show=args.show, prsweep=args.prsweep)


if __name__ == "__main__":
    main()
