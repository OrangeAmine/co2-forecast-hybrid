"""Train the HMM-LSTM hybrid model for CO2 forecasting.

Two-stage pipeline:
1. Fit HMM on training data to detect regime states
2. Augment features with HMM posterior probabilities
3. Train LSTM on augmented features

Usage:
    python scripts/train_hmm_lstm.py
    python scripts/train_hmm_lstm.py --horizon 1
    python scripts/train_hmm_lstm.py --horizon 24
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
from src.data.preprocessing import (
    chronological_split,
    inverse_scale_target,
    load_and_parse_data,
)
from src.evaluation.metrics import compute_metrics, save_metrics
from src.evaluation.visualization import (
    plot_predictions_vs_actual,
    plot_residuals,
    plot_scatter,
    plot_training_curves,
)
from src.models.hmm_lstm import HMMLSTMForecaster, HMMRegimeDetector
from src.training.trainer import create_trainer
from src.utils.config import load_config
from src.utils.seed import seed_everything


def train_single_horizon(config: dict) -> None:
    """Train HMM-LSTM for a single forecast horizon."""
    horizon = config["data"]["forecast_horizon_hours"]
    exp_name = config.get("experiment", {}).get("name", "")
    base_name = f"HMM-LSTM_h{horizon}"
    model_name = f"{exp_name}_{base_name}" if exp_name else base_name
    model_cfg = config["model"]

    seed_everything(config["training"]["seed"])

    # Step 1: Load and split raw (unscaled) data
    csv_path = Path(config["data"]["processed_csv"])
    df = load_and_parse_data(csv_path, config["data"]["datetime_column"])

    train_df, val_df, test_df = chronological_split(
        df,
        config["data"]["train_ratio"],
        config["data"]["val_ratio"],
        config["data"]["test_ratio"],
    )

    # Step 2: Fit HMM on training data (unscaled)
    print(f"\n[HMM] Fitting HMM with {model_cfg['hmm_n_states']} states "
          f"on features: {model_cfg['hmm_features']}...")

    hmm_detector = HMMRegimeDetector(
        n_states=model_cfg["hmm_n_states"],
        covariance_type=model_cfg["hmm_covariance_type"],
        n_iter=model_cfg["hmm_n_iter"],
        hmm_features=model_cfg["hmm_features"],
    )
    hmm_detector.fit(train_df)

    # Step 3: Get regime probabilities for all splits
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        probs = hmm_detector.predict_proba(split_df)
        for i in range(model_cfg["hmm_n_states"]):
            split_df[f"hmm_state_{i}"] = probs[:, i]
        print(f"  {split_name}: added {model_cfg['hmm_n_states']} HMM state features")

    # Step 4: Update feature columns to include HMM features
    hmm_feature_names = [f"hmm_state_{i}" for i in range(model_cfg["hmm_n_states"])]
    config["data"]["feature_columns"] = config["data"]["feature_columns"] + hmm_feature_names

    print(f"\n{'='*60}")
    print(f"  Training {model_name}")
    print(f"  Lookback: {config['data']['lookback_hours']}h")
    print(f"  Horizon: {horizon}h")
    print(f"  Features: {len(config['data']['feature_columns'])} "
          f"(+{model_cfg['hmm_n_states']} HMM states)")
    print(f"  Train samples: {len(train_df)}")
    print(f"{'='*60}\n")

    # Step 5: Create DataModule from augmented DataFrames
    datamodule = CO2DataModule.from_dataframes(train_df, val_df, test_df, config)

    # Step 6: Train LSTM with augmented features
    # input_size = original features + target + HMM states
    n_input = len(config["data"]["feature_columns"]) + 1
    model = HMMLSTMForecaster(config, input_size=n_input)

    trainer, run_dir = create_trainer(config, model_name=model_name)

    trainer.fit(model, datamodule=datamodule)

    # Persist scalers for reproducible inference
    datamodule.save_scalers(run_dir)

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
    parser = argparse.ArgumentParser(description="Train HMM-LSTM model")
    parser.add_argument("--horizon", type=int, nargs="+", default=[1, 24])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lookback", type=int, default=None)
    parser.add_argument("--experiment", type=str, default=None,
                        help="Path to experiment config YAML")
    args = parser.parse_args()

    config_files = [
        str(PROJECT_ROOT / "configs" / "training.yaml"),
        str(PROJECT_ROOT / "configs" / "data.yaml"),
        str(PROJECT_ROOT / "configs" / "hmm_lstm.yaml"),
    ]
    if args.experiment:
        config_files.append(args.experiment)

    base_config = load_config(config_files)

    for horizon in args.horizon:
        # Full deep copy prevents any cross-horizon mutation. The previous
        # shallow copy left config["model"] and config["training"] as shared
        # references, which would break if train_single_horizon() mutated them.
        config = copy.deepcopy(base_config)
        config["data"]["forecast_horizon_hours"] = horizon

        if args.epochs:
            config["training"]["max_epochs"] = args.epochs
        if args.lookback:
            config["data"]["lookback_hours"] = args.lookback

        train_single_horizon(config)


if __name__ == "__main__":
    main()
