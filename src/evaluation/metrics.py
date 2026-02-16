"""Evaluation metrics for time series forecasting."""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute regression metrics on original-scale values.

    Args:
        y_true: Ground truth values (original scale).
        y_pred: Predicted values (original scale).

    Returns:
        Dictionary with keys: "mse", "rmse", "mae", "mape", "r2".

    Raises:
        ValueError: If inputs contain NaN or Inf values.
    """
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        raise ValueError(
            "NaN detected in metric inputs. "
            f"y_true NaNs: {np.isnan(y_true).sum()}, "
            f"y_pred NaNs: {np.isnan(y_pred).sum()}"
        )
    if np.isinf(y_true).any() or np.isinf(y_pred).any():
        raise ValueError(
            "Inf detected in metric inputs. "
            f"y_true Infs: {np.isinf(y_true).sum()}, "
            f"y_pred Infs: {np.isinf(y_pred).sum()}"
        )
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    # MAPE: filter out near-zero values to avoid division by zero.
    # Use a small epsilon instead of an arbitrary threshold so the metric
    # generalizes correctly to any target scale (e.g., normalized values).
    _MAPE_EPS = 1e-8
    mask = np.abs(y_true) > _MAPE_EPS
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = float("nan")

    r2 = r2_score(y_true, y_pred)

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "mape": float(mape),
        "r2": float(r2),
    }


def save_metrics(
    metrics: dict[str, float],
    model_name: str,
    output_path: Path,
    experiment_info: dict | None = None,
) -> None:
    """Save metrics to JSON file and print summary.

    Args:
        metrics: Dictionary of metric name -> value.
        model_name: Name of the model.
        output_path: Path to save the JSON file.
        experiment_info: Optional experiment metadata dict (name, label, description).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict = {"model": model_name, "metrics": metrics}
    if experiment_info is not None:
        payload["experiment"] = experiment_info

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\n{'='*50}")
    print(f"  {model_name} - Test Metrics")
    print(f"{'='*50}")
    for name, value in metrics.items():
        print(f"  {name.upper():>6s}: {value:.4f}")
    print(f"{'='*50}")
    print(f"  Saved to: {output_path}")
