"""Data preprocessing utilities for model training.

Handles loading the processed CSV, chronological splitting, scaling,
and sliding window sequence creation.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

# Type alias for all supported scaler types
ScalerType = StandardScaler | MinMaxScaler | RobustScaler

SCALER_MAP: dict[str, type[ScalerType]] = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}


def load_and_parse_data(
    csv_path: Path,
    datetime_column: str = "datetime",
) -> pd.DataFrame:
    """Load processed CSV and parse datetime column.

    Args:
        csv_path: Path to the processed CSV file.
        datetime_column: Name of the datetime column.

    Returns:
        DataFrame sorted by datetime with the datetime column parsed.
    """
    df = pd.read_csv(csv_path, parse_dates=[datetime_column])
    df = df.sort_values(datetime_column).reset_index(drop=True)
    return df


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data chronologically into train/val/test sets.

    Args:
        df: Full DataFrame, assumed sorted by time.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for testing.

    Returns:
        Tuple of (train_df, val_df, test_df).

    Raises:
        ValueError: If ratios don't sum to approximately 1.0.
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)

    return train_df, val_df, test_df


def fit_scalers(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    scaler_type: str = "standard",
) -> tuple[ScalerType, ScalerType]:
    """Fit separate scalers for features and target on training data.

    Args:
        train_df: Training DataFrame.
        feature_columns: List of feature column names.
        target_column: Name of the target column.
        scaler_type: "standard", "minmax", or "robust".

    Returns:
        Tuple of (feature_scaler, target_scaler), both fitted on training data.

    Raises:
        ValueError: If scaler_type is not recognized.
    """
    scaler_cls = SCALER_MAP.get(scaler_type)
    if scaler_cls is None:
        raise ValueError(
            f"Unknown scaler_type '{scaler_type}'. "
            f"Supported: {list(SCALER_MAP.keys())}"
        )

    feature_scaler = scaler_cls()
    feature_scaler.fit(train_df[feature_columns].values)

    target_scaler = scaler_cls()
    target_scaler.fit(train_df[[target_column]].values)

    return feature_scaler, target_scaler


def apply_scalers(
    df: pd.DataFrame,
    feature_scaler: ScalerType,
    target_scaler: ScalerType,
    feature_columns: list[str],
    target_column: str,
) -> np.ndarray:
    """Apply fitted scalers to a DataFrame.

    Args:
        df: DataFrame to scale.
        feature_scaler: Fitted scaler for features.
        target_scaler: Fitted scaler for target.
        feature_columns: List of feature column names.
        target_column: Name of the target column.

    Returns:
        Array of shape (n_samples, n_features + 1) where the first columns
        are scaled features and the last column is the scaled target.
    """
    scaled_features = feature_scaler.transform(df[feature_columns].values)
    scaled_target = target_scaler.transform(df[[target_column]].values)

    return np.hstack([scaled_features, scaled_target])


def inverse_scale_target(
    values: np.ndarray,
    target_scaler: ScalerType,
) -> np.ndarray:
    """Inverse-transform scaled target values back to original scale.

    Args:
        values: Scaled target values of shape (n,) or (n, horizon).
        target_scaler: Fitted target scaler.

    Returns:
        Values in original scale, same shape as input.
    """
    original_shape = values.shape
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    elif values.ndim == 2 and values.shape[1] > 1:
        # For multi-step: inverse transform each step independently
        # The scaler was fit on single-column data, so process column by column
        result = np.zeros_like(values)
        for i in range(values.shape[1]):
            result[:, i] = target_scaler.inverse_transform(
                values[:, i].reshape(-1, 1)
            ).ravel()
        return result

    result = target_scaler.inverse_transform(values)
    return result.reshape(original_shape)


def compute_dco2(
    df: pd.DataFrame,
    interval_minutes: int,
    co2_column: str = "CO2",
) -> pd.DataFrame:
    """Compute CO2 rate of change (ppm/hour) within a single split.

    First row of the split gets NaN (no previous value to diff against).
    This avoids cross-split leakage when called independently per split.

    Args:
        df: DataFrame containing the CO2 column.
        interval_minutes: Sampling interval in minutes.
        co2_column: Name of the CO2 column.

    Returns:
        DataFrame with 'dCO2' column added.
    """
    hours_per_step = interval_minutes / 60.0
    df["dCO2"] = df[co2_column].diff() / hours_per_step
    return df


def create_sequences(
    data: np.ndarray,
    lookback: int,
    horizon: int,
    stride: int = 1,
    target_idx: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """Create sliding window sequences for time series forecasting.

    Args:
        data: Scaled numpy array of shape (n_samples, n_cols).
        lookback: Number of past timesteps in each input window.
        horizon: Number of future timesteps to predict.
        stride: Step between consecutive windows.
        target_idx: Column index of the target variable (default: last column).

    Returns:
        X: Input windows of shape (n_sequences, lookback, n_cols).
        y: Target values of shape (n_sequences, horizon).
    """
    # Defensive check: NaN/Inf in input would silently propagate into
    # training tensors and produce NaN losses with no clear error message.
    if np.isnan(data).any():
        nan_count = np.isnan(data).sum()
        raise ValueError(
            f"Input data contains {nan_count} NaN values. "
            f"Clean the data before creating sequences."
        )
    if np.isinf(data).any():
        inf_count = np.isinf(data).sum()
        raise ValueError(
            f"Input data contains {inf_count} Inf values. "
            f"Clean the data before creating sequences."
        )

    n_samples = data.shape[0]
    n_sequences = (n_samples - lookback - horizon) // stride + 1

    if n_sequences < 1:
        raise ValueError(
            f"Not enough data for windowing: n_samples={n_samples}, "
            f"lookback={lookback}, horizon={horizon}, stride={stride}. "
            f"Need at least {lookback + horizon} samples."
        )

    X = np.zeros((n_sequences, lookback, data.shape[1]), dtype=np.float32)
    y = np.zeros((n_sequences, horizon), dtype=np.float32)

    for i in range(n_sequences):
        start = i * stride
        X[i] = data[start : start + lookback]
        y[i] = data[start + lookback : start + lookback + horizon, target_idx]

    return X, y
