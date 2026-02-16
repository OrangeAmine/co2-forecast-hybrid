"""Raw data preprocessing pipeline.

Transforms raw XLS sensor files into a clean, regularly-sampled DataFrame.

Pipeline: Raw XLS -> Remove duplicates -> Sort by time ->
          Remove impossible values -> Resample to regular grid.

Returns a DataFrame with NaN gaps preserved (no dropna). Post-split
interpolation and feature engineering are handled by ``pipeline.py``.

Supports configurable resampling intervals (default: 5-min, also 1h for
hourly variants) via the ``interval_minutes`` parameter in ``run_pipeline()``.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Valid sensor ranges for impossible value detection
VALID_RANGES: dict[str, tuple[float, float]] = {
    "CO2": (300, 5000),
    "Temperature": (-20, 50),
    "Humidity": (0, 100),
    "Noise": (30, 120),
    "Pressure": (900, 1100),
}


def load_raw_xls_files(raw_dir: Path) -> pd.DataFrame:
    """Load and concatenate all raw XLS sensor files.

    Each XLS file has 2 metadata rows followed by headers at row 3.
    Actual columns: Timestamp (unix), DatetimeStr, Temperature, Humidity,
    CO2, Noise, Pressure. Uses xlrd directly with ignore_workbook_corruption
    to handle non-standard XLS formatting from the Netatmo export.

    Args:
        raw_dir: Path to the raw data directory containing BM_2021/ and BM_2022/ subdirs.

    Returns:
        Concatenated DataFrame with datetime column parsed from unix timestamps.
    """
    import xlrd

    xls_files = sorted(raw_dir.glob("**/*.xls"))
    if not xls_files:
        raise FileNotFoundError(f"No .xls files found in {raw_dir}")

    logger.info(f"Found {len(xls_files)} XLS files to process")

    col_names = ["Timestamp", "DatetimeStr", "Temperature", "Humidity",
                 "CO2", "Noise", "Pressure"]

    frames: list[pd.DataFrame] = []
    for filepath in xls_files:
        logger.info(f"  Loading {filepath.name}...")
        wb = xlrd.open_workbook(str(filepath), ignore_workbook_corruption=True)
        sheet = wb.sheet_by_index(0)
        # Data starts at row 3 (rows 0-1 are metadata, row 2 is headers)
        rows = []
        for row_idx in range(3, sheet.nrows):
            rows.append(sheet.row_values(row_idx)[:7])
        df = pd.DataFrame(rows, columns=col_names)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    # Drop DatetimeStr column (we derive datetime from unix Timestamp)
    combined = combined.drop(columns=["DatetimeStr"])

    # Drop rows where Timestamp is not numeric (header row contamination)
    combined["Timestamp"] = pd.to_numeric(combined["Timestamp"], errors="coerce")
    n_before = len(combined)
    combined = combined.dropna(subset=["Timestamp"])
    n_dropped = n_before - len(combined)
    if n_dropped > 0:
        logger.info(f"  Dropped {n_dropped} rows with invalid timestamps")

    # Convert unix timestamp to datetime with timezone
    combined["datetime"] = pd.to_datetime(
        combined["Timestamp"], unit="s", utc=True
    ).dt.tz_convert("Europe/Paris")

    # Ensure numeric types for sensor columns
    for col in ["Temperature", "Humidity", "CO2", "Noise", "Pressure"]:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    combined = combined.drop(columns=["Timestamp"])

    logger.info(f"  Total records loaded: {len(combined)}")
    logger.info(f"  Date range: {combined['datetime'].min()} to {combined['datetime'].max()}")

    return combined


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate timestamps, keeping the first occurrence.

    Args:
        df: DataFrame with 'datetime' column.

    Returns:
        DataFrame with duplicates removed.
    """
    n_before = len(df)
    df = df.drop_duplicates(subset=["datetime"], keep="first")
    n_removed = n_before - len(df)
    logger.info(f"  Removed {n_removed} duplicate timestamps")
    return df


def sort_by_time(df: pd.DataFrame) -> pd.DataFrame:
    """Sort DataFrame by datetime column in ascending order.

    Args:
        df: DataFrame with 'datetime' column.

    Returns:
        Sorted DataFrame with reset index.
    """
    df = df.sort_values("datetime").reset_index(drop=True)
    logger.info("  Sorted by datetime")
    return df


def remove_impossible_values(
    df: pd.DataFrame,
    valid_ranges: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Replace out-of-range sensor values with NaN.

    Does NOT drop entire rows -- only individual values outside valid ranges
    are replaced with NaN, preserving other columns.

    Args:
        df: DataFrame with sensor columns.
        valid_ranges: Dict mapping column names to (min, max) valid ranges.
            Defaults to module-level VALID_RANGES.

    Returns:
        DataFrame with impossible values replaced by NaN.
    """
    if valid_ranges is None:
        valid_ranges = VALID_RANGES

    for col, (vmin, vmax) in valid_ranges.items():
        if col not in df.columns:
            continue
        mask = (df[col] < vmin) | (df[col] > vmax)
        n_invalid = mask.sum()
        if n_invalid > 0:
            logger.info(f"  {col}: {n_invalid} impossible values replaced with NaN "
                        f"(valid range: [{vmin}, {vmax}])")
            df.loc[mask, col] = np.nan

    return df


def resample_to_interval(
    df: pd.DataFrame,
    interval_minutes: int = 5,
    max_ffill_minutes: int = 30,
) -> pd.DataFrame:
    """Resample irregular timestamps to a regular time grid.

    Uses mean aggregation within each time bin. Short gaps are
    forward-filled; longer gaps remain as NaN.

    Args:
        df: DataFrame with 'datetime' column and sensor readings.
        interval_minutes: Resampling interval in minutes (e.g., 5 or 60).
        max_ffill_minutes: Maximum gap duration (in minutes) to forward-fill.

    Returns:
        Regularly-sampled DataFrame with datetime index.
        NaN rows from gaps longer than max_ffill_minutes are preserved.

    Raises:
        ValueError: If interval_minutes is not positive.
    """
    if interval_minutes <= 0:
        raise ValueError(f"interval_minutes must be positive, got {interval_minutes}")

    df = df.set_index("datetime")

    # Convert to UTC for resampling to avoid DST ambiguity issues
    df.index = df.index.tz_convert("UTC")

    resample_rule = f"{interval_minutes}min"
    resampled = df.resample(resample_rule).mean()

    # Forward-fill short gaps
    max_periods = max(max_ffill_minutes // interval_minutes, 1)
    resampled = resampled.ffill(limit=max_periods)

    n_nan = resampled.isna().sum()
    logger.info(f"  After resampling to {interval_minutes}-min grid: {len(resampled)} records")
    logger.info(f"  NaN counts after forward-fill (limit={max_ffill_minutes}min):\n{n_nan.to_string()}")

    # Convert back to Europe/Paris
    resampled.index = resampled.index.tz_convert("Europe/Paris")

    return resampled


def resample_to_5min(df: pd.DataFrame, max_ffill_minutes: int = 30) -> pd.DataFrame:
    """Resample to 5-minute grid (backward-compatible wrapper).

    Args:
        df: DataFrame with 'datetime' column and sensor readings.
        max_ffill_minutes: Maximum gap duration (in minutes) to forward-fill.

    Returns:
        Regularly-sampled DataFrame at 5-min resolution.
    """
    return resample_to_interval(df, interval_minutes=5, max_ffill_minutes=max_ffill_minutes)


def run_pipeline(
    raw_dir: Path,
    interval_minutes: int = 5,
    max_ffill_minutes: int | None = None,
) -> pd.DataFrame:
    """Execute the raw data preprocessing pipeline (steps 1-5 only).

    Pipeline: Load XLS -> Remove duplicates -> Sort -> Remove impossible
    values -> Resample to grid. Returns a DataFrame with NaN gaps
    preserved for post-split interpolation.

    Args:
        raw_dir: Path to directory containing raw XLS files.
        interval_minutes: Resampling interval in minutes (5 for 5-min, 60 for 1h).
        max_ffill_minutes: Maximum gap to forward-fill. Defaults to
            30 for 5-min data, 120 for 1h data.

    Returns:
        Resampled DataFrame with datetime index and NaN gaps preserved.
    """
    if max_ffill_minutes is None:
        # Sensible defaults: 30min for 5-min grid, 120min for 1h grid
        max_ffill_minutes = 30 if interval_minutes <= 5 else 120

    logger.info("=" * 60)
    logger.info("RAW DATA PREPROCESSING PIPELINE")
    logger.info("=" * 60)

    logger.info("\n[1/5] Loading raw XLS files...")
    df = load_raw_xls_files(raw_dir)

    logger.info("\n[2/5] Removing duplicates...")
    df = remove_duplicates(df)

    logger.info("\n[3/5] Sorting by time...")
    df = sort_by_time(df)

    logger.info("\n[4/5] Removing impossible values...")
    df = remove_impossible_values(df)

    logger.info(f"\n[5/5] Resampling to {interval_minutes}-minute grid...")
    df = resample_to_interval(df, interval_minutes=interval_minutes,
                              max_ffill_minutes=max_ffill_minutes)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("RAW PIPELINE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Records: {len(df)}")
    logger.info(f"  Date range: {df.index.min()} to {df.index.max()}")
    logger.info(f"  Columns: {df.columns.tolist()}")
    logger.info(f"  NaN counts:\n{df.isna().sum().to_string()}")

    return df
