# -*- coding: utf-8 -*-

import tomllib
from pathlib import Path


class _PathsConfig:

    class Config:
        jasa_style: Path

    class Data:
        bathy: Path
        bathy_bounds: Path
        das: Path
        das_location: Path
        equipment: Path

    class Reports:
        figures: Path
        results: Path

    config: Config
    data: Data

    def __init__(self):
        self.config = self.Config()
        self.data = self.Data()
        self.reports = self.Reports()

        # Load configuration
        config_path = Path.cwd() / "config.toml"
        try:
            with open(config_path, "rb") as f:
                config = tomllib.load(f)

            # Populate top-level attributes
            for key, value in config.items():
                if isinstance(value, dict):
                    # Handle nested configs
                    if hasattr(self, key):
                        section = getattr(self, key)
                        for subkey, subvalue in value.items():
                            setattr(section, subkey, Path(subvalue).resolve())
                else:
                    setattr(self, key, value)
        except Exception as e:
            import sys

            sys.stderr.write(f"Error loading paths configuration: {str(e)}\n")


paths = _PathsConfig()
