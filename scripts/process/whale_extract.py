#!/usr/bin/env python3
"""Script to extract whale call templates using times defined in a configuration file."""

from argparse import ArgumentParser
from pathlib import Path
import tomllib

import h5py

from vineyard.config import get_path
from vineyard.readers import read_acoustic_data

SENSORS = [
    {
        "name": "3dvha",
        "channel": 7,
    },
    {
        "name": "vla1",
        "channel": 3,
    },
    {
        "name": "vla2",
        "channel": 0,
    },
]


def main(file: Path, output_file: Path):
    with open(file, "rb") as f:
        config = tomllib.load(f)

    with h5py.File(output_file, "w") as f:
        for sensor in SENSORS:
            sensor_name = sensor["name"]
            channel = sensor["channel"]

            time_start = config[sensor_name]["fin_whale"]["start"]
            time_end = config[sensor_name]["fin_whale"]["end"]

            template = read_acoustic_data(
                get_path(f"{sensor_name}_inventory"),
                time_start,
                time_end,
                channels=channel,
                taper_pc=0.25,
                filt_type="bandpass",
                filt_freq=[19.0, 25.0],
            ).taper(max_percentage=0.25)

            g = f.create_group(f"{sensor_name}_fin_whale")
            template.create_hdf5_dataset(g)
            print(f"Saved template for {sensor_name} to {output_file}")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--whale_config",
        type=Path,
        default=get_path("whale_config"),
        help="Path to whale configuration file.",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        default=get_path("whale_templates"),
        help="Path to output HDF5 file.",
    )
    args = parser.parse_args()
    main(args.whale_config, args.output_file)
