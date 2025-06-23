#!/usr/bin/env python

from adaptfilt import lms, nlms
import matplotlib.pyplot as plt
import numpy as np
from tritonoa.data.reader import read_inventory
from tritonoa.data.time import TIME_PRECISION

from vineyard.config import SENSORS, get_path


def load_data():
    sensor = "3dvha"
    frequencies = [15.0, 35.0]
    channels = 7
    time_start = np.datetime64("2023-12-01 22:25:00", TIME_PRECISION)
    time_end = np.datetime64("2023-12-01 22:26:00", TIME_PRECISION)

    inventory = get_path(f"{sensor}_inventory")
    ds = read_inventory(
        inventory,
        time_start=np.datetime64(time_start, TIME_PRECISION),
        time_end=np.datetime64(time_end, TIME_PRECISION),
        channels=channels,
    ).filter(filt_type="bandpass", freq=frequencies)
    return ds.seconds, ds.data[0]


def main():
    t, x = load_data()
    
    plt.figure()
    plt.plot(t, x, label='Original Signal')
    plt.show()


if __name__ == "__main__":
    main()
