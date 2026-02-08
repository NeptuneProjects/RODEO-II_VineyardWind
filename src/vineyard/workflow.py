#! /usr/bin/env python3
import tomllib
from argparse import ArgumentParser
from pathlib import Path

from vineyard.config import ConfigModel
from vineyard.etl import run_etl
from vineyard.process import process_data


def run_workflow(config_path: Path) -> ConfigModel:
    """Run the ETL workflow with the given config file.

    Args:
        config_path: Path to the TOML configuration file.

    Returns:
        The loaded ConfigModel instance.
    """
    with open(config_path, "rb") as f:
        config = ConfigModel(**tomllib.load(f))

    run_etl(config.etl_config)
    process_data(config.process_config)

    return config


def main() -> None:
    """Entry point for the workflow command-line tool."""
    parser = ArgumentParser(
        description="Run the ETL process with a given configuration."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path.cwd() / "config" / "config.toml",
        help="Path to the configuration file (TOML format).",
    )
    args = parser.parse_args()
    run_workflow(args.config)


if __name__ == "__main__":
    main()
