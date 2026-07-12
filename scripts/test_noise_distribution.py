"""Test whether noise in the matched-filter envelope follows a Rayleigh distribution.

For each sampled detection, extracts the noise region of the pulse-compressed
envelope (everything outside ±template_duration_s of the peak) from all three
sensors.  Normalizes each detection's noise samples to unit mean before pooling,
which isolates the shape test from varying noise levels across space and time.

Under the Rayleigh assumption:
  - Bandpass Gaussian noise → analytic signal envelope e ~ Rayleigh(σ)
  - Equivalently, e² ~ Exponential(mean=σ²)
  - Therefore: median(e²) / mean(e²) = ln(2) ≈ 0.693

Two analyses are run side-by-side:
  Full:    all noise samples (includes pile driving transients in upper tail)
  Trimmed: samples ≤ P95 per detection (isolates background, removes transients)

The trimmed result is the primary basis for the recommendation, because the full
distribution is expected to fail even under correct Rayleigh assumptions whenever
pile driving transients contaminate the noise window.

Output supports the choice of noise_correction in LocalizationConfig:
  "rayleigh"  → ln(2) ≈ 0.693  (unbiased if background is Rayleigh)
  "none"      → 1.0             (raw median; biased low ~30% if Rayleigh holds)
  float       → empirical value estimated from trimmed analysis

Usage:
    python scripts/test_noise_distribution.py
"""

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.signal import hilbert
from scipy.stats import anderson, expon, gamma, kurtosis
from tqdm import tqdm
from tritonoa.data.reader import read_hdf5

import vineyard.readers as readers

# --- Configuration ---
PC_DATA_PATH = Path("data/acoustic/denoised")
TDOA_FILE = Path("data/acoustic/tdoa/localization_raw.csv")
OUTPUT_FILE = Path("reports/figures/noise_distribution_test.png")
SENSORS = ["3dvha", "vla1", "vla2"]
CHANNEL = 5
TEMPLATE_DURATION_S = 1.2
WINDOW_S = 5.0
FILT_TYPE = "bandpass"
FILT_FREQ = [15.0, 50.0]
N_DETECTIONS = 60  # uniformly sampled across the observation period; None = all
RAYLEIGH_CORRECTION = np.log(2)  # ln(2) ≈ 0.693
TRIM_PERCENTILE = 95.0  # upper percentile to exclude for background-only test


def extract_noise_samples(
    pc_path: Path,
    channel: int,
    detection_time: np.datetime64,
    template_duration_s: float,
    window_s: float = WINDOW_S,
    filt_type: str | None = FILT_TYPE,
    filt_freq: float | list[float] | None = FILT_FREQ,
) -> np.ndarray | None:
    """Return squared envelope samples from the noise region around a detection."""
    try:
        meta = read_hdf5(pc_path, start=0, stop=1)
        fs = meta.stats.sampling_rate
        t_file_start = meta.stats.time_init

        dt_s = (detection_time - t_file_start) / np.timedelta64(1, "s")
        peak_sample = int(round(float(dt_s) * fs))

        half_win = int(window_s * fs)
        start_idx = max(0, peak_sample - half_win)
        stop_idx = peak_sample + half_win

        ds = readers.process_datastream(
            read_hdf5(pc_path, start=start_idx, stop=stop_idx),
            filt_type=filt_type,
            filt_freq=filt_freq,
        )

        env_sq = np.abs(hilbert(ds.data[channel])) ** 2
        peak_local = peak_sample - start_idx
        n = len(env_sq)
        sig_half = int(template_duration_s * fs)
        sig_start = max(0, peak_local - sig_half)
        sig_stop = min(n, peak_local + sig_half)

        noise_mask = np.ones(n, dtype=bool)
        noise_mask[sig_start:sig_stop] = False
        return env_sq[noise_mask]
    except Exception:
        return None


def fit_gamma(samples: np.ndarray) -> tuple[float, float]:
    """MLE fit of Gamma(shape, scale) with location fixed at 0."""
    a, _loc, scale = gamma.fit(samples, floc=0)
    return float(a), float(scale)


def log_ccdf(samples: np.ndarray, n_points: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """Empirical complementary CDF subsampled to n_points for plotting."""
    x = np.sort(samples)
    idx = np.unique(np.round(np.linspace(0, len(x) - 1, n_points)).astype(int))
    x = x[idx]
    sf = 1.0 - np.arange(1, len(x) + 1) / (len(x) + 1)
    return x, sf


def ad_verdict(result) -> str:
    """Return a human-readable Anderson-Darling verdict string."""
    rejected_at = [
        sl for sl, cv in zip(result.significance_level, result.critical_values)
        if result.statistic > cv
    ]
    if rejected_at:
        return f"rejected at {min(rejected_at):.0f}% level"
    return f"not rejected at {max(result.significance_level):.0f}% level"


def main() -> None:
    df = pl.read_csv(TDOA_FILE, try_parse_dates=True)

    if N_DETECTIONS is not None and N_DETECTIONS < len(df):
        indices = np.round(np.linspace(0, len(df) - 1, N_DETECTIONS)).astype(int)
        df = df[indices]

    print(f"Sampling {len(df)} detections across {len(SENSORS)} sensors...")

    correction_times: list[np.datetime64] = []
    correction_full: list[float] = []   # median(e²)/mean(e²) — full window
    correction_trim: list[float] = []   # median(e²)/mean(e²) — trimmed at P95
    sensor_labels: list[str] = []
    all_normalized: list[np.ndarray] = []  # per-detection×sensor, normalized to unit mean

    for row in tqdm(df.iter_rows(named=True), total=len(df)):
        ref_time = np.datetime64(row["timestamp"])

        for sensor in SENSORS:
            pc_path = PC_DATA_PATH / f"{sensor}_pc.h5"
            if not pc_path.exists():
                continue

            tdoa_s = row[sensor]
            t_sensor = ref_time + np.timedelta64(int(tdoa_s * 1e6), "us")

            noise_sq = extract_noise_samples(pc_path, CHANNEL, t_sensor, TEMPLATE_DURATION_S)
            if noise_sq is None or len(noise_sq) < 20:
                continue

            mean_val = float(np.mean(noise_sq))
            if mean_val <= 0:
                continue

            norm = noise_sq / mean_val
            thresh = float(np.percentile(norm, TRIM_PERCENTILE))
            norm_trim = norm[norm <= thresh]

            correction_times.append(ref_time)
            correction_full.append(float(np.median(norm)))           # = median(e²)/mean(e²)
            sensor_labels.append(sensor)
            all_normalized.append(norm)

            if len(norm_trim) >= 5:
                correction_trim.append(
                    float(np.median(norm_trim) / np.mean(norm_trim))
                )
            else:
                correction_trim.append(float("nan"))

    if not all_normalized:
        print("No noise samples extracted — check PC data path and channel.")
        return

    pooled = np.concatenate(all_normalized)
    corr_full_arr = np.array(correction_full)
    corr_trim_arr = np.array(correction_trim)
    corr_trim_valid = corr_trim_arr[np.isfinite(corr_trim_arr)]
    correction_times_arr = np.array(correction_times, dtype="datetime64[us]")

    # Pooled trimmed: rebuild from per-detection arrays after trimming each
    pooled_trim = np.concatenate([
        n[n <= np.percentile(n, TRIM_PERCENTILE)] for n in all_normalized
    ])

    # --- Statistical tests ---
    ad_full = anderson(pooled, dist="expon")
    kurt_full = float(kurtosis(pooled, fisher=True))
    gamma_shape_full, gamma_scale_full = fit_gamma(pooled)

    ad_trim = anderson(pooled_trim, dist="expon")
    kurt_trim = float(kurtosis(pooled_trim, fisher=True))
    gamma_shape_trim, gamma_scale_trim = fit_gamma(pooled_trim)

    mean_corr_full = float(np.mean(corr_full_arr))
    mean_corr_trim = float(np.mean(corr_trim_valid))
    std_corr_trim = float(np.std(corr_trim_valid))

    # Recommendation: AD test is not useful at these sample sizes (millions of
    # samples → any deviation is detected). Use kurtosis and Gamma shape instead.
    # Excess kurtosis ≈ 6 → exponential → Rayleigh background.
    # Gamma shape ≈ 1   → exponential.
    # Note: empirical correction factor may be pulled below ln(2) by residual
    # transient contamination surviving P95 trimming; kurtosis is more reliable.
    rayleigh_shape = abs(kurt_trim - 6.0) < 1.5 or gamma_shape_trim > 0.75
    correction_stable = std_corr_trim < 0.08
    if rayleigh_shape:
        recommendation = '"rayleigh"   — background kurtosis ≈ 6; Rayleigh holds'
    elif correction_stable:
        recommendation = f'float: {mean_corr_trim:.4f}   — stable non-Rayleigh background'
    else:
        recommendation = '"none"   — background non-Rayleigh and factor unstable'

    # --- Figure ---
    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32)

    ax_qq = fig.add_subplot(gs[0, 0])
    ax_ccdf = fig.add_subplot(gs[0, 1])
    ax_hist = fig.add_subplot(gs[0, 2])
    ax_time = fig.add_subplot(gs[1, 0])
    ax_chist = fig.add_subplot(gs[1, 1])
    ax_summary = fig.add_subplot(gs[1, 2])

    # ── Panel 1: Q-Q (trimmed, to show background fit) ──────────────────────
    n_qq = min(5000, len(pooled_trim))
    rng = np.random.default_rng(0)
    qq_samp = np.sort(rng.choice(pooled_trim, n_qq, replace=False))
    qq_th = expon.ppf(np.linspace(1 / (n_qq + 1), n_qq / (n_qq + 1), n_qq))
    lim = max(qq_th.max(), qq_samp.max()) * 1.05

    ax_qq.scatter(qq_th, qq_samp, s=2, alpha=0.25, color="steelblue", rasterized=True,
                  label=f"Background (≤P{TRIM_PERCENTILE:.0f})")
    ax_qq.plot([0, lim], [0, lim], "r-", lw=1.5, label="y = x  [Rayleigh]")
    ax_qq.set_xlim(0, lim)
    ax_qq.set_ylim(0, lim)
    ax_qq.set_xlabel("Exponential quantile (theoretical)")
    ax_qq.set_ylabel("Empirical e² (normalized, trimmed)")
    ax_qq.set_title("Q-Q: background vs Exponential")
    ax_qq.legend(fontsize=8)

    # ── Panel 2: Log-survival — full vs trimmed vs Exp(1) ───────────────────
    x_emp, sf_emp = log_ccdf(pooled)
    x_tr, sf_tr = log_ccdf(pooled_trim)
    x_th = np.linspace(0, x_emp.max(), 400)

    ax_ccdf.semilogy(x_emp, sf_emp, color="steelblue", lw=1.5, label="Full (incl. transients)")
    ax_ccdf.semilogy(x_tr, sf_tr, color="darkorange", lw=1.5, label=f"Background (≤P{TRIM_PERCENTILE:.0f})")
    ax_ccdf.semilogy(x_th, expon.sf(x_th), "r--", lw=1.5, label="Exp(1)  [Rayleigh]")
    ax_ccdf.semilogy(
        x_th, gamma.sf(x_th, a=gamma_shape_trim, scale=gamma_scale_trim),
        "g:", lw=1.5, label=f"Gamma(a={gamma_shape_trim:.2f})  [trimmed fit]"
    )
    ax_ccdf.set_xlim(0, np.percentile(pooled, 99.5))
    ax_ccdf.set_xlabel("Normalized e²")
    ax_ccdf.set_ylabel("P(E² > x)")
    ax_ccdf.set_title("Log-survival: full vs background")
    ax_ccdf.legend(fontsize=7.5)

    # ── Panel 3: PDF histogram (trimmed bulk only) ───────────────────────────
    x_max_h = float(np.percentile(pooled_trim, 99))
    bins = np.linspace(0, x_max_h, 60)
    ax_hist.hist(pooled_trim, bins=bins, density=True,
                 color="darkorange", alpha=0.6, label=f"Background (≤P{TRIM_PERCENTILE:.0f})")
    x_pdf = np.linspace(0, x_max_h, 300)
    ax_hist.plot(x_pdf, expon.pdf(x_pdf), "r--", lw=1.5, label="Exp(1)  [Rayleigh]")
    ax_hist.plot(
        x_pdf, gamma.pdf(x_pdf, a=gamma_shape_trim, scale=gamma_scale_trim),
        "g:", lw=1.5, label=f"Gamma(a={gamma_shape_trim:.2f})"
    )
    ax_hist.set_xlim(0, x_max_h)
    ax_hist.set_xlabel("Normalized e²")
    ax_hist.set_ylabel("Probability density")
    ax_hist.set_title("Background PDF with fits")
    ax_hist.legend(fontsize=8)

    # ── Panel 4: Correction factor vs time ──────────────────────────────────
    sensor_colors = {"3dvha": "steelblue", "vla1": "darkorange", "vla2": "seagreen"}
    t_epoch = correction_times_arr.astype("datetime64[ms]").astype(float) / 1e3
    t0 = t_epoch.min()
    t_min = (t_epoch - t0) / 60.0

    for sensor in SENSORS:
        mask = np.array(sensor_labels) == sensor
        if mask.any():
            ax_time.scatter(t_min[mask], corr_full_arr[mask], s=8, alpha=0.4,
                            color=sensor_colors[sensor], label=f"{sensor} (full)")
            ax_time.scatter(t_min[mask], corr_trim_arr[mask], s=8, alpha=0.8,
                            color=sensor_colors[sensor], marker="^",
                            label=f"{sensor} (trim)")

    ax_time.axhline(RAYLEIGH_CORRECTION, color="r", lw=1.5, ls="--",
                    label=f"ln(2) = {RAYLEIGH_CORRECTION:.3f}")
    ax_time.axhline(1.0, color="k", lw=1, ls=":", label="1.0 (no corr.)")
    ax_time.set_ylabel("median(e²) / mean(e²)")
    ax_time.set_xlabel("Time from first detection (min)")
    ax_time.set_title("Correction factor vs time\n(circles=full, triangles=trimmed)")
    ax_time.legend(fontsize=6, ncol=2)

    # ── Panel 5: Distribution of correction factors ──────────────────────────
    bins_c = np.linspace(0, 1.2, 40)
    ax_chist.hist(corr_full_arr, bins=bins_c, density=True,
                  color="steelblue", alpha=0.5, label="Full")
    ax_chist.hist(corr_trim_valid, bins=bins_c, density=True,
                  color="darkorange", alpha=0.6, label=f"Trimmed (≤P{TRIM_PERCENTILE:.0f})")
    ax_chist.axvline(RAYLEIGH_CORRECTION, color="r", lw=1.5, ls="--",
                     label=f"ln(2)={RAYLEIGH_CORRECTION:.3f}")
    ax_chist.axvline(1.0, color="k", lw=1, ls=":", label="1.0")
    ax_chist.axvline(mean_corr_trim, color="darkorange", lw=2, ls="-",
                     label=f"trim mean={mean_corr_trim:.3f}")
    ax_chist.set_xlabel("median(e²) / mean(e²)")
    ax_chist.set_ylabel("Density")
    ax_chist.set_title("Correction factor distributions")
    ax_chist.legend(fontsize=7.5)

    # ── Panel 6: Summary ─────────────────────────────────────────────────────
    ax_summary.axis("off")

    lines = [
        "─── Full distribution (all samples) ───────────────",
        f"  AD statistic:   {ad_full.statistic:.2f}",
        f"  AD verdict:     {ad_verdict(ad_full)}",
        f"  Excess kurtosis:{kurt_full:.1f}   (Exp = 6.0)",
        f"  Gamma shape:    {gamma_shape_full:.3f}  (1.0 = Exp)",
        f"  Mean corr.:     {mean_corr_full:.4f}",
        "",
        f"─── Background only (≤P{TRIM_PERCENTILE:.0f}, transients removed) ──",
        f"  AD statistic:   {ad_trim.statistic:.2f}",
        f"  AD verdict:     {ad_verdict(ad_trim)}",
        f"  Excess kurtosis:{kurt_trim:.1f}   (Exp = 6.0)",
        f"  Gamma shape:    {gamma_shape_trim:.3f}  (1.0 = Exp)",
        f"  Mean corr.:     {mean_corr_trim:.4f}  ± {std_corr_trim:.4f}",
        f"  Rayleigh ln(2): {RAYLEIGH_CORRECTION:.4f}",
        f"  Δ from ln(2):   {mean_corr_trim - RAYLEIGH_CORRECTION:+.4f}",
        "",
        "─── Interpretation ─────────────────────────────────",
        "  AD test is unreliable at these sample sizes",
        "  (millions of samples; any deviation detected).",
        "  Use kurtosis and Gamma shape as primary",
        "  diagnostics. Full rejection is expected from",
        "  pile driving transients in the noise window.",
        "  Trimmed kurtosis ≈ 6 → Rayleigh background.",
        "  Trimmed factor < ln(2) likely reflects residual",
        "  contamination not removed by P95 trimming.",
        "",
        "─── Recommendation ─────────────────────────────────",
        "  noise_correction =",
        f"    {recommendation}",
    ]

    ax_summary.text(
        0.03, 0.98, "\n".join(lines),
        transform=ax_summary.transAxes,
        fontsize=7.5, va="top", ha="left",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    fig.suptitle(
        f"Noise distribution test — {len(pooled):,} pooled samples "
        f"({len(all_normalized)} detection×sensor windows from {len(df)} detections)",
        fontsize=11,
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FILE, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"\nSaved: {OUTPUT_FILE}")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
