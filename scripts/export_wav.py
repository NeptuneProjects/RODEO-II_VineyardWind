"""Export a trimmed segment of acoustic data as a WAV file."""

import argparse
import tomllib
from pathlib import Path

import numpy as np
import scipy.io.wavfile
from tritonoa.data.reader import read_and_process


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sensor",
        default="vla1",
        choices=["3dvha", "vla1", "vla2"],
        help="Sensor name",
    )
    parser.add_argument(
        "--time-start",
        default="2023-12-01T05:22:59",
        help="Start time (ISO 8601, e.g. 2023-12-01T05:22:59)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5244.0,
        help="Duration in seconds",
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=3,
        help="Channel index to export (default: all channels)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output WAV file path (default: <sensor>_<time-start>_<duration>s.wav)",
    )
    args = parser.parse_args()

    root = Path(__file__).parent.parent

    with open(root / "config" / "config.toml", "rb") as f:
        config = tomllib.load(f)

    inventory_dir = root / config["etl"]["inventory_dir"]
    inventory_path = inventory_dir / f"inventory_{args.sensor}.csv"

    time_start = np.datetime64(args.time_start)
    time_end = time_start + np.timedelta64(int(args.duration * 1e6), "us")

    ds = read_and_process(
        inventory_path,
        time_start,
        time_end,
        channels=args.channel,
        detrend=False,
    )

    fs = int(ds.stats.sampling_rate)
    data = ds.data.squeeze()

    if args.output is None:
        safe_start = args.time_start.replace(":", "-")
        args.output = Path(f"{args.sensor}_{safe_start}_{args.duration}s.wav")

    # Normalize to float32 for WAV (scipy accepts float32 in [-1, 1])
    max_val = np.abs(data).max()
    if max_val > 0:
        data = (data / max_val).astype(np.float32)
    else:
        data = data.astype(np.float32)

    if data.ndim > 1:
        # scipy expects (samples, channels) for multi-channel
        data = data.T

    scipy.io.wavfile.write(args.output, fs, data)
    print(f"Wrote {args.output}  ({fs} Hz, {data.shape})")


if __name__ == "__main__":
    main()
