#!/usr/bin/env python

from pathlib import Path

import h5py
import numpy as np
from tritonoa.data.reader import read_hdf5, read_hdf5_group

from vineyard.config import get_path


def main(denoised_data_path: Path, whale_templates_path: Path, output_path: Path):
    sensors = ["3dvha", "vla1", "vla2"]
    output_path.mkdir(parents=True, exist_ok=True)

    with h5py.File(whale_templates_path, "r") as f_whale:
        for sensor in sensors:
            ds = read_hdf5(denoised_data_path / f"{sensor}.h5")
            ds.data = ds.data[1] / np.max(np.abs(ds.data[1]))

            template = read_hdf5_group(f_whale[f"{sensor}_fin_whale"])
            ds_pc = ds.copy().pulse_compression(template.data.squeeze())
            ds_pc.data = ds_pc.data / np.max(np.abs(ds_pc.data))

            ds_pc.write_hdf5(output_path / f"{sensor}_pc.h5")


if __name__ == "__main__":
    denoised_data_path = get_path("denoised_data")
    whale_templates_path = get_path("whale_templates")
    output_path = get_path("pulse_comp_data")
    main(denoised_data_path, whale_templates_path, output_path)
