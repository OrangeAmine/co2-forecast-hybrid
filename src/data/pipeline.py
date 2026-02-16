"""Split-aware preprocessing pipeline for CO2 forecasting experiments.

Orchestrates preprocessing with a strict before/after split boundary to
prevent data leakage. Supports two pipeline variants:
    - "simple":   Interpolation + dCO2 only (variants A, B)
    - "enhanced": + denoising, outlier clipping, lags, rolling stats (variants C, D)

Public API:
    ``run_preprocessing_pipeline()`` — full pipeline from raw XLS to split-aware DataFrames
    ``before_split()``              — steps safe to run on the full dataset
    ``after_split_simple()``        — post-split processing for simple variants
    ``after_split_enhanced()``      — post-split processing for enhanced variants
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from src.data.preprocessing import chronological_split, compute_dco2
from src.data.raw_processing import run_pipeline as run_raw_pipeline

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────────────────────────────

def run_preprocessing_pipeline(
    raw_dir: Path,
    variant_config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Full pipeline: raw XLS -> split-aware (train, val, test) DataFrames.

    Returns 3 DataFrames with all features computed, NaN-free, ready for
    scaling and sequencing by the DataModule.

    Args:
        raw_dir: Path to raw XLS data directory.
        variant_config: Merged config dict containing 'data' and optionally
            'preprocessing' keys.

    Returns:
        Tuple of (train_df, val_df, test_df) with datetime column and all
        feature + target columns, NaN-free.
    """
    data_cfg = variant_config["data"]
    interval_minutes = data_cfg["interval_minutes"]
    max_interp_gap_minutes = data_cfg.get("max_interp_gap_minutes", 60)
    pipeline_variant = data_cfg.get("pipeline_variant", "simple")

    # Steps 1-6: Load raw data, resample, add deterministic temporal features
    df = before_split(
        raw_dir=raw_dir,
        interval_minutes=interval_minutes,
        max_ffill_minutes=data_cfg.get("max_ffill_minutes", 30 if interval_minutes <= 5 else 120),
    )

    # Chronological split (on data WITH NaN gaps)
    train_ratio = data_cfg.get("train_ratio", 0.70)
    val_ratio = data_cfg.get("val_ratio", 0.15)
    test_ratio = data_cfg.get("test_ratio", 0.15)

    # Reset index to column for chronological_split (expects a column-based DF)
    df_reset = df.reset_index().rename(columns={df.index.name or "index": "datetime"})

    train_df, val_df, test_df = chronological_split(
        df_reset, train_ratio, val_ratio, test_ratio,
    )

    logger.info(f"\n  Split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # Post-split processing
    if pipeline_variant == "simple":
        train_df, val_df, test_df = after_split_simple(
            train_df, val_df, test_df,
            interval_minutes=interval_minutes,
            max_interp_gap_minutes=max_interp_gap_minutes,
        )
    elif pipeline_variant == "enhanced":
        preproc_cfg = variant_config.get("preprocessing", {})
        train_df, val_df, test_df = after_split_enhanced(
            train_df, val_df, test_df,
            interval_minutes=interval_minutes,
            config=preproc_cfg,
            max_interp_gap_minutes=max_interp_gap_minutes,
        )
    else:
        raise ValueError(f"Unknown pipeline_variant: '{pipeline_variant}'. Use 'simple' or 'enhanced'.")

    logger.info(f"\n  Final sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    logger.info(f"  Columns: {train_df.columns.tolist()}")

    return train_df, val_df, test_df


def before_split(
    raw_dir: Path,
    interval_minutes: int,
    max_ffill_minutes: int,
) -> pd.DataFrame:
    """Steps safe to run on the full dataset before splitting.

    1. Load raw XLS files           (from raw_processing)
    2. Remove duplicates            (from raw_processing)
    3. Sort by time                 (from raw_processing)
    4. Remove impossible values     (from raw_processing)
    5. Resample to regular grid     (from raw_processing, mean aggregation)
    6. Forward-fill ONLY very short gaps (<=max_ffill_minutes)
    7. Rename columns (Temperature->TemperatureExt, etc.)
    8. Add DETERMINISTIC temporal features (pure functions of timestamp):
       - Day_sin, Day_cos  (hour-of-day cycle)
       - Year_sin, Year_cos (day-of-year cycle)
       NOTE: dCO2 is NOT added here (depends on CO2 values -> post-split).

    Args:
        raw_dir: Path to raw XLS data directory.
        interval_minutes: Resampling interval in minutes.
        max_ffill_minutes: Maximum gap duration to forward-fill.

    Returns:
        DataFrame WITH NaN gaps preserved (no dropna), datetime index.
    """
    # Steps 1-5 via raw_processing.run_pipeline
    df = run_raw_pipeline(
        raw_dir=raw_dir,
        interval_minutes=interval_minutes,
        max_ffill_minutes=max_ffill_minutes,
    )

    # Step 7: Rename columns to match expected feature names
    rename_map = {
        "Temperature": "TemperatureExt",
        "Humidity": "Hrext",
        "Pressure": "Pression",
    }
    df = df.rename(columns=rename_map)

    # Step 8: Deterministic temporal features (pure functions of timestamp)
    dt_idx = pd.DatetimeIndex(df.index)
    dt_naive = dt_idx.tz_localize(None) if dt_idx.tz is not None else dt_idx
    hour = dt_naive.hour + dt_naive.minute / 60.0  # type: ignore[operator]
    day_of_year = dt_naive.dayofyear + hour / 24.0  # type: ignore[operator]

    # 24h daily cycle
    df["Day_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["Day_cos"] = np.cos(2 * np.pi * hour / 24.0)
    # Annual cycle
    df["Year_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df["Year_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)

    logger.info(f"  Added deterministic temporal features: Day_sin/cos, Year_sin/cos")
    logger.info(f"  NaN rows preserved: {int(df.isna().any(axis=1).sum())}")  # type: ignore[union-attr]

    return df


def after_split_simple(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    interval_minutes: int,
    max_interp_gap_minutes: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Post-split processing for simple variants (A, B).

    Applied independently within each split:
    9.  Interpolate NaN gaps (linear, up to max_interp_gap_minutes)
    10. Drop rows with remaining NaN (gaps > max_interp_gap_minutes)
    11. Compute dCO2 from raw CO2 within each split
    12. Drop NaN from dCO2 (first row of each split)

    Args:
        train_df: Training DataFrame (may contain NaN).
        val_df: Validation DataFrame.
        test_df: Test DataFrame.
        interval_minutes: Sampling interval in minutes.
        max_interp_gap_minutes: Maximum gap to interpolate (default 60 min).

    Returns:
        Tuple of (train_df, val_df, test_df), NaN-free.
    """
    logger.info("\n  Post-split processing (simple variant)...")

    results = []
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        df = df.copy()

        # Step 9: Interpolate NaN gaps
        df = _interpolate_gaps(df, max_interp_gap_minutes, interval_minutes)
        _log_nan_impact(name, df, "interpolation")

        # Step 10: Drop remaining NaN rows
        df = df.dropna().reset_index(drop=True)

        # Step 11: Compute dCO2
        df = compute_dco2(df, interval_minutes)

        # Step 12: Drop NaN from dCO2 (first row)
        _log_nan_impact(name, df, "dCO2")
        df = df.dropna().reset_index(drop=True)

        logger.info(f"  [{name}] Final: {len(df)} rows")
        results.append(df)

    return results[0], results[1], results[2]


def after_split_enhanced(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    interval_minutes: int,
    config: dict,
    max_interp_gap_minutes: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Post-split processing for enhanced variants (C, D).

    Applied independently within each split:
    9.  Interpolate NaN gaps (linear, up to max_interp_gap_minutes)
    10. Drop rows with remaining NaN (gaps > max_interp_gap_minutes)
    11. Denoise CO2 (Savitzky-Golay, mode='nearest', per split independently)
    12. Outlier detection [train-fit]:
        - Compute IQR bounds from TRAIN split only
        - Clip values in ALL splits to those bounds
    13. Compute dCO2 from DENOISED CO2 within each split
    14. Feature engineering within each split:
        - Lag features (CO2 at configurable offsets)
        - Rolling statistics (mean, std over configurable windows)
        - Weekday sin/cos encoding
    15. Drop NaN rows from lags/rolling/dCO2

    Args:
        train_df: Training DataFrame (may contain NaN).
        val_df: Validation DataFrame.
        test_df: Test DataFrame.
        interval_minutes: Sampling interval in minutes.
        config: Preprocessing config dict with denoising, outlier_detection,
            lag_steps, rolling_windows keys.
        max_interp_gap_minutes: Maximum gap to interpolate (default 60 min).

    Returns:
        Tuple of (train_df, val_df, test_df), NaN-free.
    """
    logger.info("\n  Post-split processing (enhanced variant)...")

    denoise_cfg = config.get("denoising", {})
    outlier_cfg = config.get("outlier_detection", {})
    lag_steps = config.get("lag_steps", [12, 72, 288])
    rolling_windows = config.get("rolling_windows", [12, 72])

    # Steps 9-10: Interpolate and drop NaN for all splits
    splits = {}
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        df = df.copy()
        df = _interpolate_gaps(df, max_interp_gap_minutes, interval_minutes)
        _log_nan_impact(name, df, "interpolation")
        df = df.dropna().reset_index(drop=True)
        splits[name] = df

    # Step 11: Denoise CO2 per split
    window_length = denoise_cfg.get("window_length", 11 if interval_minutes <= 5 else 5)
    polyorder = denoise_cfg.get("polyorder", 3 if interval_minutes <= 5 else 2)
    mode = denoise_cfg.get("mode", "nearest")

    for name in splits:
        splits[name] = _denoise_co2(
            splits[name], window_length=window_length,
            polyorder=polyorder, mode=mode,
        )

    # Step 12: Outlier detection — fit on train, clip all
    outlier_columns = outlier_cfg.get(
        "columns", ["CO2", "TemperatureExt", "Hrext", "Noise", "Pression"]
    )
    multiplier = outlier_cfg.get("multiplier", 3.0)
    bounds = _compute_outlier_bounds(splits["train"], outlier_columns, multiplier)
    for name in splits:
        splits[name] = _clip_outliers(splits[name], bounds)

    # Steps 13-15: dCO2, lags, rolling, weekday per split, then drop NaN
    results = []
    for name in ["train", "val", "test"]:
        df = splits[name]

        # Step 13: dCO2 from denoised CO2
        df = compute_dco2(df, interval_minutes)

        # Step 14a: Lag features
        for lag in lag_steps:
            col_name = f"CO2_lag_{lag}"
            df[col_name] = df["CO2"].shift(lag)

        # Step 14b: Rolling statistics
        # Note: rolling_std with min_periods=1 returns NaN for the first row
        # (std of a single value is undefined). These NaNs are dropped in step 15.
        for w in rolling_windows:
            df[f"CO2_rolling_mean_{w}"] = df["CO2"].rolling(window=w, min_periods=1).mean()
            df[f"CO2_rolling_std_{w}"] = df["CO2"].rolling(window=w, min_periods=1).std()

        # Step 14c: Weekday sin/cos
        if "datetime" in df.columns:
            dt = pd.to_datetime(df["datetime"])
            day_of_week = dt.dt.dayofweek
        else:
            # Fallback if datetime is index
            day_of_week = pd.DatetimeIndex(df.index).dayofweek  # type: ignore[attr-defined]
        df["Weekday_sin"] = np.sin(2 * np.pi * day_of_week / 7.0)
        df["Weekday_cos"] = np.cos(2 * np.pi * day_of_week / 7.0)

        # Step 15: Drop NaN rows from lags/rolling/dCO2
        _log_nan_impact(name, df, "lags/rolling/dCO2")
        df = df.dropna().reset_index(drop=True)
        logger.info(f"  [{name}] Final: {len(df)} rows")
        results.append(df)

    # Remove CO2_raw helper column if present
    for i in range(3):
        if "CO2_raw" in results[i].columns:
            results[i] = results[i].drop(columns=["CO2_raw"])

    return results[0], results[1], results[2]


# ──────────────────────────────────────────────────────────────────────
#  Internal helpers
# ──────────────────────────────────────────────────────────────────────

def _interpolate_gaps(
    df: pd.DataFrame,
    max_gap_minutes: int,
    interval_minutes: int,
) -> pd.DataFrame:
    """Linearly interpolate NaN gaps up to max_gap_minutes within a single split.

    Args:
        df: DataFrame for a single split.
        max_gap_minutes: Maximum gap duration to interpolate.
        interval_minutes: Sampling interval in minutes.

    Returns:
        DataFrame with short gaps interpolated, long gaps still NaN.
    """
    max_consecutive = max_gap_minutes // interval_minutes
    # Only interpolate sensor columns (temporal sin/cos are already complete)
    sensor_cols = ["CO2", "TemperatureExt", "Hrext", "Noise", "Pression"]
    for col in sensor_cols:
        if col in df.columns:
            df[col] = df[col].interpolate(method="linear", limit=max_consecutive)
    return df



def _denoise_co2(
    df: pd.DataFrame,
    window_length: int = 11,
    polyorder: int = 3,
    mode: str = "nearest",
) -> pd.DataFrame:
    """Apply Savitzky-Golay denoising to the CO2 column.

    Args:
        df: DataFrame with CO2 column.
        window_length: Filter window size (must be odd).
        polyorder: Polynomial order.
        mode: Edge handling mode (default 'nearest').

    Returns:
        DataFrame with denoised CO2. Original saved as CO2_raw.
    """
    # Ensure window_length is odd
    if window_length % 2 == 0:
        window_length += 1

    if len(df) < window_length:
        logger.warning(f"  DataFrame has {len(df)} rows < window_length={window_length}. Skipping denoising.")
        return df

    df["CO2_raw"] = df["CO2"].copy()
    df["CO2"] = savgol_filter(df["CO2"].values, window_length, polyorder, mode=mode)

    noise_reduction = np.std(df["CO2_raw"] - df["CO2"])
    logger.info(f"  Denoised CO2 (SavGol window={window_length}, poly={polyorder}). "
                f"Noise std removed: {noise_reduction:.2f} ppm")

    return df


def _compute_outlier_bounds(
    train_df: pd.DataFrame,
    columns: list[str],
    multiplier: float = 3.0,
) -> dict[str, tuple[float, float]]:
    """Compute IQR-based clip bounds from training data only.

    Args:
        train_df: Training split DataFrame.
        columns: Columns to compute bounds for.
        multiplier: IQR multiplier (default 3.0 = conservative).

    Returns:
        Dict mapping column name to (lower_bound, upper_bound).
    """
    bounds = {}
    for col in columns:
        if col not in train_df.columns:
            continue
        q1 = train_df[col].quantile(0.25)
        q3 = train_df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        bounds[col] = (lower, upper)
        logger.info(f"  Outlier bounds [{col}]: [{lower:.1f}, {upper:.1f}] (IQR*{multiplier})")
    return bounds


def _clip_outliers(
    df: pd.DataFrame,
    bounds: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    """Clip values to bounds (computed from train). Applied to any split.

    Args:
        df: DataFrame for a single split.
        bounds: Dict mapping column -> (lower, upper) bounds.

    Returns:
        DataFrame with clipped values.
    """
    for col, (lower, upper) in bounds.items():
        if col in df.columns:
            n_clipped = ((df[col] < lower) | (df[col] > upper)).sum()
            df[col] = df[col].clip(lower=lower, upper=upper)
            if n_clipped > 0:
                logger.info(f"  Clipped {n_clipped} outliers in {col}")
    return df


def _log_nan_impact(split_name: str, df: pd.DataFrame, step_name: str) -> None:
    """Log how many rows have NaN after a processing step.

    Args:
        split_name: Name of the split (train/val/test).
        df: DataFrame to check.
        step_name: Name of the step for logging.
    """
    n_nan_rows = int(df.isna().any(axis=1).sum())  # type: ignore[union-attr]
    n_clean = len(df) - n_nan_rows
    logger.info(f"  [{split_name}] After {step_name}: {n_nan_rows} rows with NaN, "
                f"{n_clean} clean rows")
