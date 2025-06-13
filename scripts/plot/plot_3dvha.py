import argparse
import logging
from pathlib import Path

import numpy as np
from tritonoa.data.reader import read_inventory
from tritonoa.data.time import TIME_PRECISION

from vineyard.config import get_path

def main(inv_file: Path):
    time_start = np.datetime64("2023-12-01T21:06:00", TIME_PRECISION)
    time_end = np.datetime64("2023-12-01T21:07:00", TIME_PRECISION)
    ds = read_inventory(inv_file, time_start=time_start, time_end=time_end)
    print(ds.num_channels, ds.num_samples)


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    parser = argparse.ArgumentParser(description="Plot 3D VHA data.")
    parser.add_argument(
        "--inv_file", type=Path, default=get_path("3dvha_inventory"), help="Path to the inventory file."
    )
    args = parser.parse_args()

    main(args.inv_file)
