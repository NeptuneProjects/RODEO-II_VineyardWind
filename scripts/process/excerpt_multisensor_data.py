#!/usr/bin/env python3

from argparse import ArgumentParser
from pathlib import Path

import h5py
import numpy as np
from tritonoa.data.reader import read_inventory
from tritonoa.data.time import TIME_PRECISION, convert_datetime64_to_string

from vineyard.config import SENSORS, get_path


def main(time_start: str, time_end: str, output_dir: Path):
    t0 = np.datetime64(time_start, TIME_PRECISION)
    t1 = np.datetime64(time_end, TIME_PRECISION)

    file = output_dir / (
        f"acoust_data_{convert_datetime64_to_string(t0)}_"
        f"{convert_datetime64_to_string(t1)}.h5"
    )

    with h5py.File(file, "w") as f:
        for sensor in SENSORS:
            inventory = get_path(f"{sensor}_inventory")
            ds = read_inventory(
                inventory,
                time_start=np.datetime64(time_start, TIME_PRECISION),
                time_end=np.datetime64(time_end, TIME_PRECISION),
                metadata=SENSORS[sensor].get("metadata", None),
            )
            g = f.create_group(sensor)
            ds.create_hdf5_dataset(g)


if __name__ == "__main__":
    parser = ArgumentParser(description="Process multisensor data")
    parser.add_argument(
        "--time_start",
        type=str,
        default="2023-12-01T22:25:00",
        help="Start time in ISO format",
    )
    parser.add_argument(
        "--time_end",
        type=str,
        default="2023-12-01T22:26:00",
        help="End time in ISO format",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=get_path("acoustic_data"),
        help="Output HDF5 directory",
    )
    args = parser.parse_args()
    main(args.time_start, args.time_end, args.output_dir)
