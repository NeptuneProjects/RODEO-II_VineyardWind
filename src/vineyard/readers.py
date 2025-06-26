"""Functions to read data from various file formats used in the project."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import logging
import os
from pathlib import Path
import struct
import tomllib

import dascore as dc
from dascore.core.spool import DataFrameSpool
import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl
from tritonoa.data.reader import read_inventory
from tritonoa.data.stream import DataStream
from tritonoa.data.time import TIME_PRECISION

from vineyard.signal import convert_to_strain_rate, subtract_median


@dataclass(frozen=True)
class DASArrayProperties:
    name: str
    sampling_rate_hz: float
    spatial_resolution_m: float
    fiber_type: str
    zone_type: str
    start_distance_m: float
    stop_distance_m: float
    start_positioning_m: float
    measure_length_m: float
    orig_sampling_rate_hz: float
    time_decimation_factor: int
    calibration_factor_nm: float
    gauge_length_m: float
    start_time: datetime
    scale_factor: float
    calibration_factor: float

    @staticmethod
    def _format_time(time: str) -> None:
        clean_time = time.replace(" (UTC)", "")
        new_time = datetime.strptime(clean_time, "%d-%b-%Y %H:%M:%S.%f")
        return new_time.replace(tzinfo=timezone.utc)

    @classmethod
    def from_dict(cls, properties: dict) -> "DASArrayProperties":
        """Create a FileProperties instance from a dictionary of properties."""
        return cls(
            name=properties["name"],
            sampling_rate_hz=properties["SamplingFrequency[Hz]"],
            spatial_resolution_m=properties["SpatialResolution[m]"],
            fiber_type=properties["Fibre Type"],
            zone_type=properties["Zone Type"],
            start_distance_m=properties["Start Distance (m)"],
            stop_distance_m=properties["Stop Distance (m)"],
            start_positioning_m=properties["StartPosition[m]"],
            measure_length_m=properties["MeasureLength[m]"],
            orig_sampling_rate_hz=properties["Precise Sampling Frequency (Hz)"],
            time_decimation_factor=properties["Time Decimation"],
            calibration_factor_nm=properties["Unit Calibration (nm)"],
            gauge_length_m=properties["GaugeLength"],
            start_time=cls._format_time(properties["GPSTimeStamp"]),
            scale_factor=1 / 8192,
            calibration_factor=properties["Unit Calibration (nm)"],
        )


@dataclass
class TDMSHeader:
    """Class representing TDMS file header information"""

    decimated: bool = False
    next_segment_offset: int = 0
    data_offset: int = 0
    file_size: int = 0
    n_ch: int = 0
    properties: list[tuple[str, any, int]] = field(default_factory=list)
    data_type: int = 0
    chunk_size: int = 0
    channel_length: int = 0
    data_type_str: str = ""

    def get_property_dict(self) -> dict[str, any]:
        """Returns properties as a dictionary of name:value pairs"""
        return {name: value for name, value, _ in self.properties}


def read_acoustic_data(
    inventory: Path,
    start: str | np.datetime64,
    end: str | np.datetime64,
    channels: int | Sequence[int] | None = None,
    taper_pc: float | None = None,
    dec_factor: int | None = None,
    filt_type: str | None = None,
    filt_freq: float | Sequence[float] | None = None,
    metadata: dict | None = None,
) -> DataStream:
    if isinstance(start, str) | isinstance(start, datetime):
        start = np.datetime64(start, TIME_PRECISION)
    if isinstance(end, str) | isinstance(end, datetime):
        end = np.datetime64(end, TIME_PRECISION)

    ds = read_inventory(
        file_path=inventory,
        time_start=start,
        time_end=end,
        channels=channels,
        metadata=metadata,
    )
    if taper_pc is not None:
        ds.taper(max_percentage=taper_pc)
    if dec_factor is not None:
        ds.decimate(dec_factor)
    if filt_type is not None and filt_freq is not None:
        ds.filter(filt_type, filt_freq)
    return ds


def read_bathymetry(
    file: Path,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Read bathymetry data from an HDF5 file.

    Args:
        file: Path to the HDF5 file.
    Returns:
        Tuple of numpy arrays containing bathymetry data, longitude vector, and latitude vector.
    """
    logging.info(f"Reading file: {file}")
    with h5py.File(file, "r") as f:
        data = f.get("data")[:]
        lonvec = f.get("lonvec")[:]
        latvec = f.get("latvec")[:]
    return data, lonvec, latvec


def read_bbox(fname: Path, key: str) -> list[list[float]]:
    with open(fname, "rb") as f:
        return tomllib.load(f)[key]


def read_das_array_properties(file: Path) -> DASArrayProperties:
    tdms_properties = read_tdms_properties(file)
    return DASArrayProperties.from_dict(tdms_properties)


def read_das_locations(file: Path) -> pl.DataFrame:
    """Read the DAS location from a HDF5 file.

    Args:
        file (Path): Path to the HDF5 file.
    Returns:
        Dataframe with DAS location in latitudes/longitudes.
    """
    logging.info(f"Reading file: {file}")
    with h5py.File(file, "r") as f:
        das_location = {
            "latitude": f.get("Lat")[:],
            "longitude": f.get("Lon")[:],
            "error_m": f.get("Location_err[m]")[:],
        }
    return pl.DataFrame(das_location)


def read_sensor_locations(file: Path) -> pd.DataFrame:
    """Read the sensor locations from a CSV file.

    Args:
        file (Path): Path to the CSV file.
    Returns:
        Dataframe with sensor locations in latitudes/longitudes.
    """
    logging.info(f"Reading file: {file}")
    df = pd.read_csv(file)
    return df.drop(index=[1, 2, 3, 4]).rename(
        columns={
            "Latitude [dd]": "latitude",
            "Longitude [dd]": "longitude",
            "Depth [m]": "depth",
        }
    )


def read_tdms(
    file: Path,
    time: tuple[str | None] | None = None,
    channel: tuple[int | None] | None = None,
) -> tuple[dc.Patch, DASArrayProperties]:
    def add_channel_numbers(patch: DataFrameSpool) -> DataFrameSpool:
        """
        Add channel number to the patch.
        """
        dist_len = patch.coord_shapes["distance"][0]
        channel_number = np.arange(dist_len)
        return patch.update_coords(channel=("distance", channel_number))
        # return patch.update_coords(
        #     channel_number=("distance", channel_number)
        # ).set_dims(distance="channel_number")

    # 1. Get attributes from the directory
    properties = read_das_array_properties(list(file.glob("*.tdms"))[0])

    # 2. Get the multi-file spool from the directory
    spool = dc.spool(file)

    # 2.5 Select by time
    if time:
        patch = spool.select(time=time).chunk(time=None)[0]

    logging.info(f"Original patch shape: {patch.shape}")

    # 6. Add the channel coordinates to the patch
    patch = add_channel_numbers(patch)

    # 7. Select the first and last channel
    out = patch.select(channel=channel, relative=True)

    # Convert to strain rate

    new_data = convert_to_strain_rate(
        out.data,
        properties.scale_factor,
        properties.calibration_factor,
        properties.sampling_rate_hz,
        properties.gauge_length_m,
    )
    new_data = subtract_median(new_data)

    new_patch = out.update(data=new_data)

    return new_patch, properties


def read_tdms_header(file_path: Path) -> TDMSHeader:
    """Read TDMS file header information

    Args:
        file_path: Path to the TDMS file

    Returns:
        A TDMSFileInfo object containing the header information
    """
    fileinfo = TDMSHeader()
    fileinfo.file_size = os.path.getsize(file_path)

    with open(file_path, "rb") as fid:
        # Read lead in
        fid.seek(4)  # Jump the "TDSm" tag
        decimated_byte = int.from_bytes(fid.read(1), byteorder="little")
        decimated_bin = format(decimated_byte, "08b")
        fileinfo.decimated = (
            decimated_bin[2] == "0"
        )  # Get data format, decimated or not

        fid.seek(12)  # Jump to next Segment offset
        fileinfo.next_segment_offset = (
            int.from_bytes(fid.read(8), byteorder="little") + 28
        )
        fileinfo.data_offset = int.from_bytes(fid.read(8), byteorder="little") + 28

        if fileinfo.next_segment_offset == -1:
            fileinfo.next_segment_offset = fileinfo.file_size

        # Read properties
        fid.seek(28)
        fileinfo.n_ch = (
            int.from_bytes(fid.read(4), byteorder="little") - 2
        )  # Total objects - file objects - group objects
        n = int.from_bytes(fid.read(4), byteorder="little")
        object_name = fid.read(n).decode("utf-8", errors="replace")

        fid.seek(4, 1)  # Relative seek from current position
        n = int.from_bytes(fid.read(4), byteorder="little")

        for i in range(n):
            l = int.from_bytes(fid.read(4), byteorder="little")
            prop_name = fid.read(l).decode("utf-8", errors="replace")
            property_type = int.from_bytes(fid.read(4), byteorder="little")

            prop_value = None
            if property_type == 32:  # String
                l = int.from_bytes(fid.read(4), byteorder="little")
                prop_value = fid.read(l).decode("utf-8", errors="replace")
            elif property_type == 9:  # Single
                prop_value = struct.unpack("<f", fid.read(4))[0]
            elif property_type == 5:  # UInt8
                prop_value = int.from_bytes(fid.read(1), byteorder="little")
            elif property_type == 10:  # Double
                prop_value = struct.unpack("<d", fid.read(8))[0]
            elif property_type == 33:  # Boolean
                prop_value = bool(int.from_bytes(fid.read(1), byteorder="little"))
            elif property_type == 3:  # Int32
                prop_value = int.from_bytes(
                    fid.read(4), byteorder="little", signed=True
                )
            elif property_type == 2:  # Int16
                prop_value = int.from_bytes(
                    fid.read(2), byteorder="little", signed=True
                )
            elif property_type == 7:  # UInt32
                prop_value = int.from_bytes(fid.read(4), byteorder="little")
            elif property_type == 6:  # UInt16
                prop_value = int.from_bytes(fid.read(2), byteorder="little")
            elif property_type == 68:  # Timestamp
                fract = struct.unpack("<Q", fid.read(8))[0] * 2**-64
                seconds = struct.unpack("<q", fid.read(8))[0]

                # Calculate date from seconds since Jan 1, 1904
                base_date = datetime(1904, 1, 1)
                if seconds == 0 and fract == 0:
                    prop_value = "N/A"
                else:
                    date = base_date + timedelta(seconds=seconds)
                    prop_value = date.strftime("%d-%b-%Y %H:%M:%S.%f")[:-3] + " (UTC)"
            else:
                raise ValueError(f"Error: Property type not defined: {property_type}")

            fileinfo.properties.append((prop_name, prop_value, property_type))

        # Read group information and channel path
        group_name_len = int.from_bytes(fid.read(4), byteorder="little")
        fid.seek(8 + group_name_len, 1)  # Jump Group Information

        channel_path_len = int.from_bytes(fid.read(4), byteorder="little")
        fid.seek(
            4 + channel_path_len, 1
        )  # Jump first channel path and length of index information

        # Read data type and chunk size
        fileinfo.data_type = int.from_bytes(fid.read(4), byteorder="little")
        fid.seek(4, 1)  # Jump Dimension of the raw data array
        fileinfo.chunk_size = int.from_bytes(fid.read(4), byteorder="little")

        # Determine data type string and calculate channel length
        if fileinfo.file_size == fileinfo.next_segment_offset:
            fileinfo.channel_length = 0
        else:
            # Try to read the next segment for more precise channel length
            try:
                fid.seek(fileinfo.next_segment_offset + 12)
                offset2 = int.from_bytes(fid.read(8), byteorder="little")
                offset1 = int.from_bytes(fid.read(8), byteorder="little")
                length_diff = offset2 - offset1

                if fileinfo.data_type == 2:  # int16
                    fileinfo.data_type_str = "int16"
                    fileinfo.channel_length = int(
                        (
                            length_diff
                            + fileinfo.next_segment_offset
                            - fileinfo.data_offset
                        )
                        / fileinfo.n_ch
                        / 2
                    )
                elif fileinfo.data_type == 9:  # single
                    fileinfo.data_type_str = "single"
                    fileinfo.channel_length = int(
                        (
                            length_diff
                            + fileinfo.next_segment_offset
                            - fileinfo.data_offset
                        )
                        / fileinfo.n_ch
                        / 4
                    )
                else:
                    fileinfo.data_type_str = f"unknown_{fileinfo.data_type}"
            except:
                # Fallback: estimate channel length from file size
                if fileinfo.data_type == 2:  # int16
                    fileinfo.data_type_str = "int16"
                    fileinfo.channel_length = int(
                        (fileinfo.file_size - fileinfo.data_offset) / fileinfo.n_ch / 2
                    )
                elif fileinfo.data_type == 9:  # single
                    fileinfo.data_type_str = "single"
                    fileinfo.channel_length = int(
                        (fileinfo.file_size - fileinfo.data_offset) / fileinfo.n_ch / 4
                    )
                else:
                    fileinfo.data_type_str = f"unknown_{fileinfo.data_type}"

    return fileinfo


def read_tdms_properties(file_path: Path) -> dict[str, any]:
    """Convenience function to read TDMS properties as a dictionary

    Args:
        file_path: Path to the TDMS file

    Returns:
        dictionary containing the file properties
    """
    fileinfo = read_tdms_header(file_path)
    return fileinfo.get_property_dict()


def read_turbine_locations(file: Path) -> pd.DataFrame:
    """Read the turbine locations from a CSV file.

    Args:
        file (Path): Path to the CSV file.
    Returns:
        Dataframe with turbine locations in latitudes/longitudes.
    """
    logging.info(f"Reading file: {file}")
    df = pd.read_csv(file)
    return df.drop(index=[63, 64])[["Latitude dd", "longitude dd"]].rename(
        columns={
            "Latitude dd": "latitude",
            "longitude dd": "longitude",
        }
    )


# @dataclass(frozen=True)
# class tsdmFileProperties:
#     scale_factor: float = 1.0 / 8192.0
#     calibration_factor: float = 116.0
# file_name: str
# sampling_rate_hz: float
# spatial_resolution_m: float
# fiber_type: str
# zone_type: str
# start_distance_m: float
# stop_distance_m: float
# start_positioning_m: float
# measure_length_m: float
# orig_sampling_rate_hz: float
# time_decimation_factor: int
# calibration_factor_nm: float
# gauge_length_m: float
# start_time: datetime

# def __post_init__(self):
#     self._add_utc()

# def _add_utc(self) -> None:
#     """Ensure that the start_time is in UTC if no time zone is given."""
#     if self.start_time.tzinfo is None:
#         self.start_time = self.start_time.replace(tzinfo=timezone.utc)

# @classmethod
# def from_dict(cls, properties: dict) -> "tsdmFileProperties":
#     """Create a FileProperties instance from a dictionary of properties."""
#     return cls(
#         file_name=properties["name"],
#         sampling_rate_hz=properties["SamplingFrequency[Hz]"],
#         spatial_resolution_m=properties["SpatialResolution[m]"],
#         fiber_type=properties["Fibre Type"],
#         zone_type=properties["Zone Type"],
#         start_distance_m=properties["Start Distance (m)"],
#         stop_distance_m=properties["Stop Distance (m)"],
#         start_positioning_m=properties["StartPosition[m]"],
#         measure_length_m=properties["MeasureLength[m]"],
#         orig_sampling_rate_hz=properties["Precise Sampling Frequency (Hz)"],
#         time_decimation_factor=properties["Time Decimation"],
#         calibration_factor_nm=properties["Unit Calibration (nm)"],
#         gauge_length_m=properties["GaugeLength"],
#         start_time=properties["GPSTimeStamp"],
#     )
