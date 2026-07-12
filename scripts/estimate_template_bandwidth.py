"""Estimate bandwidth and center frequency of whale call templates.

Loads templates from whale_templates.h5 and compares two methods:
  1. Instantaneous frequency (restricted to envelope > 30% of peak)
  2. PSD at multiple dB thresholds (-3, -6, -10 dB)

Results for 3dvha sensor (2023-12-01 deployment):
  type1: fc ≈ 24 Hz, BW ≈ 9.5 Hz  (duration 1.5 s)  — validated against manual 10 Hz
  type2: fc ≈ 20 Hz, BW ≈ 4–6 Hz  (duration 1.0 s)  — narrower than manual estimate

Usage:
    python scripts/estimate_template_bandwidth.py
"""

from pathlib import Path

import numpy as np
from scipy.signal import butter, hilbert, periodogram, sosfiltfilt

TEMPLATE_FILE = Path("data/acoustic/templates/whale_templates.h5")
SENSORS = ["3dvha", "vla1", "vla2"]
CALL_TYPES = ["type1", "type2"]
PREFILTER_LO = 10.0  # Hz — bandpass before instantaneous frequency
PREFILTER_HI = 45.0  # Hz
ENV_THRESHOLD = 0.30  # fraction of peak envelope; samples below are excluded
PSD_THRESHOLDS = [-3, -6, -10]  # dB


def bandpass(
    x: np.ndarray, fs: float, lo: float, hi: float, order: int = 4
) -> np.ndarray:
    sos = butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def instantaneous_bandwidth(
    template: np.ndarray,
    fs: float,
    lo: float = PREFILTER_LO,
    hi: float = PREFILTER_HI,
    env_threshold_frac: float = ENV_THRESHOLD,
) -> dict[str, float]:
    filtered = bandpass(template, fs, lo, hi)
    analytic = hilbert(filtered)
    env = np.abs(analytic)
    inst_freq = np.diff(np.unwrap(np.angle(analytic))) / (2 * np.pi) * fs

    valid = env[:-1] > env_threshold_frac * env.max()
    if valid.sum() < 10:
        return {
            "fc": float("nan"),
            "bw": float("nan"),
            "f_lo": float("nan"),
            "f_hi": float("nan"),
        }

    if_valid = inst_freq[valid]
    f_lo = float(if_valid.min())
    f_hi = float(if_valid.max())
    return {"fc": (f_hi + f_lo) / 2, "bw": f_hi - f_lo, "f_lo": f_lo, "f_hi": f_hi}


def psd_bandwidth(
    template: np.ndarray,
    fs: float,
    lo: float = PREFILTER_LO,
    hi: float = PREFILTER_HI,
    threshold_db: float = -10.0,
) -> dict[str, float]:
    filtered = bandpass(template, fs, lo, hi)
    f, Pxx = periodogram(filtered, fs=fs)
    Pxx_db = 10 * np.log10(Pxx / Pxx.max() + 1e-30)
    mask = (f > lo * 0.5) & (f < hi * 2)
    sig = f[mask][Pxx_db[mask] > threshold_db]
    if len(sig) < 2:
        return {
            "fc": float("nan"),
            "bw": float("nan"),
            "f_lo": float("nan"),
            "f_hi": float("nan"),
        }
    f_lo, f_hi = float(sig.min()), float(sig.max())
    return {"fc": (f_hi + f_lo) / 2, "bw": f_hi - f_lo, "f_lo": f_lo, "f_hi": f_hi}


def main() -> None:
    import h5py

    with h5py.File(TEMPLATE_FILE, "r") as f:
        for sensor in SENSORS:
            if sensor not in f:
                continue
            print(f"\n{'=' * 60}")
            print(f"Sensor: {sensor}")
            print(f"{'=' * 60}")

            for call_type in CALL_TYPES:
                path = f"{sensor}/{call_type}"
                if path not in f:
                    continue

                group = f[path]
                fs = float(group.attrs["sampling_rate"])
                template = group["data"][0]
                duration_s = len(template) / fs

                print(f"\n  {call_type}  (fs={fs:.1f} Hz, duration={duration_s:.2f} s)")

                inst = instantaneous_bandwidth(template, fs)
                print(
                    f"  Inst. freq (env>{ENV_THRESHOLD * 100:.0f}%): "
                    f"fc={inst['fc']:.1f} Hz  BW={inst['bw']:.1f} Hz  "
                    f"({inst['f_lo']:.1f}–{inst['f_hi']:.1f} Hz)"
                )

                for thr in PSD_THRESHOLDS:
                    p = psd_bandwidth(template, fs, threshold_db=thr)
                    print(
                        f"  PSD {thr:+d} dB:              "
                        f"fc={p['fc']:.1f} Hz  BW={p['bw']:.1f} Hz  "
                        f"({p['f_lo']:.1f}–{p['f_hi']:.1f} Hz)"
                    )

                print(
                    "  Manual reference:        fc=24.0 Hz  BW=10.0 Hz  (19.0–29.0 Hz)"
                )


if __name__ == "__main__":
    main()
