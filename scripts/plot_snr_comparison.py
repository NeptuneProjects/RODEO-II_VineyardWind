"""SNR distributions at annotated whale calls: piling vs. quiet (denoised channel).

Compares the pulse-compressed SNR (dB) and single-channel timing uncertainty
(σ_t, ms) between piling and quiet conditions on the denoised channel.

Why piling vs. quiet (not raw vs. denoised):
  - The matched filter template is sensitive to both whale calls AND pile
    strikes (similar duration and frequency band), so raw PC SNR during
    piling is contaminated by pile correlation — not a valid comparison.
  - Denoising is only applied during active pile driving.  During quiet
    periods the raw and denoised signals are identical, so quiet-period SNR
    is the unmodified baseline.
  - If SNR_piling (denoised) ≈ SNR_quiet, denoising effectively recovered
    the baseline and self-nulling is minimal.  Any gap is a joint upper
    bound on residual interference + self-nulling.

Data source: reports/evaluation/snr_comparison.csv
  Supports both old schema (with 'data' column: raw/denoised rows) and new
  schema (denoised-only, no 'data' column).

Usage:
    python scripts/plot_snr_comparison.py
"""
