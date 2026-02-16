"""Train the XGBoost model for CO2 forecasting.

Usage:
    python scripts/train_xgboost.py
    python scripts/train_xgboost.py --horizon 1
    python scripts/train_xgboost.py --horizon 24
"""

import argparse
import copy
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
from src.models.xgboost_model import XGBoostForecaster
from src.utils.config import load_config
from src.utils.seed import seed_everything


def train_single_horizon(config: dict) -> None:
    """Train XGBoost for a single forecast horizon.

    Args:
        config: Merged configuration dictionary.
    """
    horizon = config["data"]["forecast_horizon_hours"]
    exp_name = config.get("experiment", {}).get("name", "")
    base_name = f"XGBoost_h{horizon}"
    model_name = f"{exp_name}_{base_name}" if exp_name else base_name

    seed_everything(config["training"]["seed"])

    # Data — same pipeline as neural models
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

    # Extract numpy arrays from datasets
    X_train = datamodule.train_dataset.X.numpy()
    y_train = datamodule.train_dataset.y.numpy()
    X_val = datamodule.val_dataset.X.numpy()
    y_val = datamodule.val_dataset.y.numpy()
    X_test = datamodule.test_dataset.X.numpy()
    y_test_scaled = datamodule.test_dataset.y.numpy()

    # Fit model
    model = XGBoostForecaster(config)

    t0 = time.time()
    print(f"  Fitting XGBoost with {model.n_estimators} estimators, "
          f"max_depth={model.max_depth}...")
    model.fit(X_train, y_train, X_val, y_val)
    elapsed = time.time() - t0
    print(f"  Fitting completed in {elapsed:.1f}s")

    # Predict
    t0 = time.time()
    y_pred_scaled = model.predict(X_test)
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
    parser = argparse.ArgumentParser(description="Train XGBoost model")
    parser.add_argument("--horizon", type=int, nargs="+", default=[1, 24],
                        help="Forecast horizon(s) in hours (default: 1 24)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override max epochs (controls n_estimators)")
    parser.add_argument("--lookback", type=int, default=None,
                        help="Override lookback hours")
    parser.add_argument("--experiment", type=str, default=None,
                        help="Path to experiment config YAML")
    args = parser.parse_args()

    config_files = [
        str(PROJECT_ROOT / "configs" / "training.yaml"),
        str(PROJECT_ROOT / "configs" / "data.yaml"),
        str(PROJECT_ROOT / "configs" / "xgboost.yaml"),
    ]
    if args.experiment:
        config_files.append(args.experiment)

    base_config = load_config(config_files)

    for horizon in args.horizon:
        config = copy.deepcopy(base_config)
        config["data"]["forecast_horizon_hours"] = horizon

        if args.epochs:
            config["model"]["n_estimators"] = args.epochs
        if args.lookback:
            config["data"]["lookback_hours"] = args.lookback

        train_single_horizon(config)


if __name__ == "__main__":
    main()
