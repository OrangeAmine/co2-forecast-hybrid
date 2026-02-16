"""Train the SARIMA model for CO2 forecasting.

SARIMA is a univariate statistical baseline — it uses only the target
variable (CO2) and captures temporal structure (trend, seasonality,
autocorrelation) without exogenous features.

Usage:
    python scripts/train_sarima.py
    python scripts/train_sarima.py --horizon 1
    python scripts/train_sarima.py --horizon 24
"""

import argparse
import copy
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.datamodule import CO2DataModule
from src.data.preprocessing import inverse_scale_target
from src.evaluation.metrics import compute_metrics, save_metrics
from src.evaluation.visualization import (
    plot_predictions_vs_actual,
    plot_residuals,
    plot_scatter,
)
from src.models.sarima import SARIMAForecaster
from src.utils.config import load_config
from src.utils.seed import seed_everything


def train_single_horizon(config: dict) -> None:
    """Train SARIMA for a single forecast horizon.

    Args:
        config: Merged configuration dictionary.
    """
    horizon = config["data"]["forecast_horizon_hours"]
    exp_name = config.get("experiment", {}).get("name", "")
    base_name = f"SARIMA_h{horizon}"
    model_name = f"{exp_name}_{base_name}" if exp_name else base_name

    seed_everything(config["training"]["seed"])

    # Data — use the standard pipeline for consistent splits and scaling
    datamodule = CO2DataModule(config)
    datamodule.setup()

    print(f"\n{'='*60}")
    print(f"  Training {model_name}")
    print(f"  Lookback: {config['data']['lookback_hours']}h "
          f"({datamodule.lookback_steps} steps)")
    print(f"  Horizon: {horizon}h ({datamodule.horizon_steps} steps)")
    print(f"  Train samples: {len(datamodule.train_dataset)}")
    print(f"  Val samples: {len(datamodule.val_dataset)}")
    print(f"  Test samples: {len(datamodule.test_dataset)}")
    print(f"{'='*60}\n")

    # Create run directory
    results_dir = Path(config["training"]["results_dir"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = results_dir / f"{model_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # SARIMA uses scaled data from the sliding windows (target is last col)
    # Extract training target series from the raw training split
    # to fit the SARIMA parameters, then predict on test windows.
    X_train = datamodule.train_dataset.X.numpy()
    y_train = datamodule.train_dataset.y.numpy()
    X_test = datamodule.test_dataset.X.numpy()
    y_test_scaled = datamodule.test_dataset.y.numpy()

    # For SARIMA we extract the target column (last) from the full training
    # sequence windows and reconstruct a continuous series for fitting.
    # Use the first window's full lookback + each subsequent window's last step
    # This approximates the original scaled training series.
    target_idx = -1  # target is last column in scaled data
    train_target_first_window = X_train[0, :, target_idx]
    train_target_remaining = y_train[:, 0]  # first forecast step of each window
    train_series = np.concatenate([train_target_first_window, train_target_remaining])

    # Fit model
    t0 = time.time()
    model = SARIMAForecaster(config)

    # Use a simpler order for the benchmark (large seasonal periods are slow)
    # The config specifies the seasonal period; for 1h horizon the
    # default (1,1,1)x(1,1,1,288) is used but can be costly.
    print(f"  SARIMA order: {model.order}")
    print(f"  SARIMA seasonal order: {model.seasonal_order}")
    print(f"  Fitting on {len(train_series)} scaled target values...")

    model.fit(train_series)
    fit_time = time.time() - t0
    print(f"  Fitting completed in {fit_time:.1f}s")

    # Predict on test windows
    print(f"  Generating forecasts for {len(X_test)} test windows...")
    t0 = time.time()
    y_pred_scaled = model.predict_batch(X_test, target_idx=target_idx)
    pred_time = time.time() - t0
    print(f"  Prediction completed in {pred_time:.1f}s")

    # Inverse scale
    y_pred = inverse_scale_target(y_pred_scaled, datamodule.target_scaler)
    y_true = inverse_scale_target(y_test_scaled, datamodule.target_scaler)

    # Metrics
    metrics = compute_metrics(y_true, y_pred)
    save_metrics(metrics, model_name, run_dir / "metrics.json",
                 experiment_info=config.get("experiment"))

    # Save predictions
    np.savez(run_dir / "predictions.npz", y_true=y_true, y_pred=y_pred)

    # Save scalers
    datamodule.save_scalers(run_dir)

    # Plots
    plots_dir = run_dir / "plots"
    plot_predictions_vs_actual(y_true, y_pred, model_name, plots_dir / "predictions.png")
    plot_scatter(y_true, y_pred, model_name, plots_dir / "scatter.png")
    plot_residuals(y_true, y_pred, model_name, plots_dir / "residuals.png")

    print(f"\nAll outputs saved to: {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SARIMA model")
    parser.add_argument("--horizon", type=int, nargs="+", default=[1, 24],
                        help="Forecast horizon(s) in hours (default: 1 24)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override max epochs (not used by SARIMA)")
    parser.add_argument("--lookback", type=int, default=None,
                        help="Override lookback hours")
    parser.add_argument("--experiment", type=str, default=None,
                        help="Path to experiment config YAML")
    args = parser.parse_args()

    config_files = [
        str(PROJECT_ROOT / "configs" / "training.yaml"),
        str(PROJECT_ROOT / "configs" / "data.yaml"),
        str(PROJECT_ROOT / "configs" / "sarima.yaml"),
    ]
    if args.experiment:
        config_files.append(args.experiment)

    base_config = load_config(config_files)

    for horizon in args.horizon:
        config = copy.deepcopy(base_config)
        config["data"]["forecast_horizon_hours"] = horizon

        if args.lookback:
            config["data"]["lookback_hours"] = args.lookback

        train_single_horizon(config)


if __name__ == "__main__":
    main()
