"""Train the CNN-LSTM hybrid model for CO2 forecasting.

Usage:
    python scripts/train_cnn_lstm.py
    python scripts/train_cnn_lstm.py --horizon 1
    python scripts/train_cnn_lstm.py --horizon 24
"""

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.datamodule import CO2DataModule
from src.data.preprocessing import inverse_scale_target
from src.evaluation.metrics import compute_metrics, save_metrics
from src.evaluation.visualization import (
    plot_predictions_vs_actual,
    plot_residuals,
    plot_scatter,
    plot_training_curves,
)
from src.models.cnn_lstm import CNNLSTMForecaster
from src.training.trainer import create_trainer
from src.utils.config import load_config
from src.utils.seed import seed_everything


def train_single_horizon(config: dict) -> None:
    """Train CNN-LSTM for a single forecast horizon."""
    horizon = config["data"]["forecast_horizon_hours"]
    exp_name = config.get("experiment", {}).get("name", "")
    base_name = f"CNN-LSTM_h{horizon}"
    model_name = f"{exp_name}_{base_name}" if exp_name else base_name

    seed_everything(config["training"]["seed"])

    # Data
    datamodule = CO2DataModule(config)
    datamodule.setup()

    print(f"\n{'='*60}")
    print(f"  Training {model_name}")
    print(f"  Lookback: {config['data']['lookback_hours']}h "
          f"({datamodule.lookback_steps} steps)")
    print(f"  Horizon: {horizon}h ({datamodule.horizon_steps} steps)")
    assert datamodule.train_dataset is not None
    assert datamodule.val_dataset is not None
    assert datamodule.test_dataset is not None
    print(f"  Train samples: {len(datamodule.train_dataset)}")
    print(f"  Val samples: {len(datamodule.val_dataset)}")
    print(f"  Test samples: {len(datamodule.test_dataset)}")
    print(f"{'='*60}\n")

    # Model
    model = CNNLSTMForecaster(config)

    # Trainer
    trainer, run_dir = create_trainer(config, model_name=model_name)

    # Train
    trainer.fit(model, datamodule=datamodule)

    # Persist scalers for reproducible inference
    datamodule.save_scalers(run_dir)

    # Test
    trainer.test(model, datamodule=datamodule, ckpt_path="best")

    # Predictions
    predictions = trainer.predict(model, datamodule.test_dataloader(), ckpt_path="best")
    assert predictions is not None
    y_pred_scaled = torch.cat(predictions, dim=0).numpy()  # type: ignore[arg-type]
    assert datamodule.test_dataset is not None
    y_true_scaled = datamodule.test_dataset.y.numpy()

    assert datamodule.target_scaler is not None
    y_pred = inverse_scale_target(y_pred_scaled, datamodule.target_scaler)
    y_true = inverse_scale_target(y_true_scaled, datamodule.target_scaler)

    # Metrics
    metrics = compute_metrics(y_true, y_pred)
    save_metrics(metrics, model_name, run_dir / "metrics.json",
                 experiment_info=config.get("experiment"))

    np.savez(run_dir / "predictions.npz", y_true=y_true, y_pred=y_pred)

    # Plots
    plots_dir = run_dir / "plots"
    plot_predictions_vs_actual(y_true, y_pred, model_name, plots_dir / "predictions.png")
    plot_scatter(y_true, y_pred, model_name, plots_dir / "scatter.png")
    plot_residuals(y_true, y_pred, model_name, plots_dir / "residuals.png")
    plot_training_curves(run_dir / "tb_logs", model_name, plots_dir / "training_curves.png")

    print(f"\nAll outputs saved to: {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CNN-LSTM model")
    parser.add_argument("--horizon", type=int, nargs="+", default=[1, 24])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lookback", type=int, default=None)
    parser.add_argument("--experiment", type=str, default=None,
                        help="Path to experiment config YAML")
    args = parser.parse_args()

    config_files = [
        str(PROJECT_ROOT / "configs" / "training.yaml"),
        str(PROJECT_ROOT / "configs" / "data.yaml"),
        str(PROJECT_ROOT / "configs" / "cnn_lstm.yaml"),
    ]
    if args.experiment:
        config_files.append(args.experiment)

    base_config = load_config(config_files)

    for horizon in args.horizon:
        # Full deep copy prevents cross-horizon mutation of nested dicts
        config = copy.deepcopy(base_config)
        config["data"]["forecast_horizon_hours"] = horizon

        if args.epochs:
            config["training"]["max_epochs"] = args.epochs
        if args.lookback:
            config["data"]["lookback_hours"] = args.lookback

        train_single_horizon(config)


if __name__ == "__main__":
    main()
