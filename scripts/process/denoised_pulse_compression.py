#!/usr/bin/env python

import matplotlib.pyplot as plt
from tritonoa.data.reader import read_hdf5
from tritonoa.data.stream import DataStream

from vineyard.config import get_path


def main():
    sensors = ["3dvha", "vla1", "vla2"]
    savepath = get_path("figures") / "denoised"
    savepath.mkdir(parents=True, exist_ok=True)

    # TODO: Load whale templates
    template = None

    for sensor in sensors:
        ds = read_hdf5(get_path("denoised_data") / f"{sensor}.h5")
        ds.data = ds.data[1]

        ds_pc = ds.pulse_compression(template)


if __name__ == "__main__":
    main()
