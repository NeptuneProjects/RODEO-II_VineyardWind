#! /usr/bin/env python3
"""Run workflows with a given configuration."""

import tomllib
from argparse import ArgumentParser
from pathlib import Path
from typing import Literal

from vineyard.config import ConfigModel
from vineyard.etl import run_etl
from vineyard.process import process_data
from vineyard.tdoa import localize


def run_workflow(
    command: Literal["run", "etl", "process", "localize"], config_path: Path
) -> ConfigModel:
    """Run the ETL workflow with the given config file.

    Args:
        command: The workflow command to execute. 'run' executes both ETL and
            processing steps, 'etl' executes only the ETL step, 'process'
            executes only the processing step, and 'localize' executes only
            the TDOA estimation step.
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
    }

    with open(config_path, "rb") as f:
        config = ConfigModel(**tomllib.load(f))

    if command not in COMMAND_REGISTRY:
        raise ValueError(f"Invalid command: {command}")

    COMMAND_REGISTRY[command](config)

    return config


def main() -> None:
    """Entry point for the workflow command-line tool."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["run", "etl", "process", "localize"],
        help=(
            "The workflow command to execute. 'run' executes both ETL and "
            "processing steps, 'etl' executes only the ETL step, 'process' "
            "executes only the processing step, and 'localize' executes only "
            "the TDOA estimation step."
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
    args = parser.parse_args()
    run_workflow(args.command, args.config)


if __name__ == "__main__":
    main()
