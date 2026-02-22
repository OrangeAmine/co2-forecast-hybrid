# Noise-Derived Features Design

**Date:** 2026-02-22
**Goal:** Add noise-derived features for both CO2 forecasting and occupancy inference

## Motivation

Currently, raw Noise (dB) passes through the pipeline with zero feature engineering,
while CO2 gets dCO2, lag features, rolling statistics, and deviation features. Noise
is a valuable occupancy signal (human activity generates acoustic events) but is
underutilized.

## Feature Tiers

### Tier 1 (Approach A) — Always computed

| Feature | Formula | Rationale |
|---------|---------|-----------|
| dNoise | diff(Noise) / hours_per_step | Activity onset/offset detection |
| Noise_lag_1 | shift(1) | 1-step autoregressive context |
| Noise_lag_6 | shift(6) | 6-hour context |
| Noise_lag_24 | shift(24) | Diurnal pattern |
| Noise_rolling_mean_3 | rolling(3).mean() | Smoothed trend |
| Noise_rolling_std_3 | rolling(3).std() | Short-term variability |
| Noise_rolling_mean_6 | rolling(6).mean() | Medium-term trend |
| Noise_rolling_std_6 | rolling(6).std() | Medium-term variability |

### Tier 2 (Approach B extras) — Optional via config flag

| Feature | Formula | Rationale |
|---------|---------|-----------|
| Noise_energy | 10^(Noise/10) | Linear energy (additive, not log) |
| Noise_deviation_from_baseline | Noise - hourly_p75 | Excess above per-hour baseline |
| CO2_Noise_corr_6 | rolling_corr(CO2, Noise, 6) | Cross-sensor correlation |

## Files Modified

1. `src/data/preprocessing.py` — new `compute_noise_features()` function
2. `src/data/pipeline.py` — call in `after_split_enhanced()`
3. `configs/experiments/preproc_D_enhanced_1h.yaml` — add Tier 1 features
4. `configs/experiments/preproc_E_occupancy_1h.yaml` — add Tier 1 features
5. New: `configs/experiments/preproc_F_noise_tier2_1h.yaml` — Tier 1 + Tier 2

## Non-breaking

Existing configs without noise features in feature_columns are unaffected.
Benchmark uses synthetic data and is unaffected.
