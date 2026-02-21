"""Recursive multi-step forecasting using a pre-trained h=1 model.

Instead of training a direct h=24 model (which struggles because the linear
output layer must predict all 24 steps at once), this script:
  1. Loads a pre-trained h=1 model checkpoint
  2. For each test sample, iteratively predicts 1 step ahead
  3. Shifts the lookback window forward, inserting the predicted CO2 value
     while using actual known features (temperature, time encodings, etc.)
  4. Repeats for `target_horizon` steps (e.g., 24)

This autoregressive approach lets the model accumulate context step-by-step
rather than making a single long-range jump. Error accumulation is the main
risk — prediction errors compound at each step.

Key design choice:
  - Exogenous features (temperature, pressure, time encodings, etc.) use
    actual future values since these are either measured or deterministic.
  - Only the CO2 target column is replaced with the model's prediction,
    because CO2 is the only unknown quantity at inference time.
  - Lag features (CO2_lag_1, CO2_lag_6, etc.) and rolling stats are
    NOT updated during recursive steps since they are pre-computed from
    raw data. This is a simplification; a production system would recompute
    them from the predicted CO2 trajectory.

Usage:
    python scripts/run_recursive_forecast.py --checkpoint results/preproc_E_LSTM_h1_*/checkpoints/best-*.ckpt --target-horizon 24
    python scripts/run_recursive_forecast.py --checkpoint results/preproc_E_Seq2Seq_h1_*/checkpoints/best-*.ckpt --target-horizon 24
"""

import argparse
import copy
import glob
import sys
import time
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
)
from src.models.lstm import LSTMForecaster
from src.models.seq2seq import Seq2SeqForecaster
from src.utils.config import load_config
from src.utils.seed import seed_everything


# Map model class names to their constructors
MODEL_CLASSES = {
    "LSTMForecaster": LSTMForecaster,
    "Seq2SeqForecaster": Seq2SeqForecaster,
}


def recursive_predict(
    model: torch.nn.Module,
    X_test_full: np.ndarray,
    y_test_full: np.ndarray,
    lookback: int,
    target_horizon: int,
    target_col_idx: int = -1,
    device: str = "cuda",
) -> np.ndarray:
    """Perform recursive multi-step forecasting.

    For each test sample i:
      - Start with the lookback window X_test_full[i]  (lookback, n_features)
      - Predict 1 step ahead using the h=1 model
      - Shift window forward: drop oldest row, append new row where
        CO2 = predicted value, other features = actual values from data
      - Repeat for `target_horizon` steps

    To get the actual future features for each step, we use the fact that
    the test data windows are created with stride=1, so X_test_full[i+k]
    contains the actual features at time t+k (offset by lookback).

    Args:
        model: Pre-trained h=1 model on GPU.
        X_test_full: All test input windows (n_windows, lookback, n_features).
        y_test_full: Ground truth h=24 targets (n_windows, target_horizon).
        lookback: Number of lookback steps.
        target_horizon: How many steps to predict recursively.
        target_col_idx: Column index of CO2 target in feature array.
        device: Torch device.

    Returns:
        Predictions array of shape (n_valid, target_horizon) in scaled space.
    """
    model.eval()
    n_windows = X_test_full.shape[0]
    n_features = X_test_full.shape[2]

    # We need target_horizon future windows after each starting point
    # to get the actual exogenous features for each recursive step.
    # The last (target_horizon - 1) windows don't have enough future data.
    n_valid = n_windows - target_horizon + 1
    if n_valid <= 0:
        raise ValueError(
            f"Not enough test windows ({n_windows}) for "
            f"target_horizon={target_horizon}. Need at least {target_horizon}."
        )

    predictions = np.zeros((n_valid, target_horizon), dtype=np.float32)

    with torch.no_grad():
        for i in range(n_valid):
            if i % 200 == 0:
                print(f"  Recursive prediction: {i}/{n_valid} samples...")

            # Start with the actual lookback window for sample i
            window = X_test_full[i].copy()  # (lookback, n_features)

            for step in range(target_horizon):
                # Predict 1 step ahead
                x_tensor = torch.from_numpy(window).float().unsqueeze(0).to(device)
                pred = model(x_tensor)  # (1, 1) for h=1 model
                pred_val = pred.cpu().numpy().flatten()[0]
                predictions[i, step] = pred_val

                # If not the last step, construct the next window
                if step < target_horizon - 1:
                    # Get actual features for the next timestep from
                    # the (i + step + 1)-th window's last row.
                    # X_test_full[i + step + 1] has lookback rows;
                    # the last row (index -1) corresponds to time t + step + 1
                    future_features = X_test_full[i + step + 1, -1, :].copy()

                    # Replace CO2 (target) with our prediction
                    future_features[target_col_idx] = pred_val

                    # Shift window: drop first row, append new row
                    window = np.concatenate(
                        [window[1:], future_features.reshape(1, -1)],
                        axis=0,
                    )

    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recursive multi-step forecasting with h=1 model"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path (or glob pattern) to model checkpoint file"
    )
    parser.add_argument(
        "--target-horizon", type=int, default=24,
        help="Number of steps to predict recursively (default: 24)"
    )
    parser.add_argument(
        "--experiment", type=str, default=None,
        help="Path to experiment config YAML"
    )
    parser.add_argument(
        "--model-type", type=str, default="lstm",
        choices=["lstm", "seq2seq"],
        help="Model architecture (default: lstm)"
    )
    parser.add_argument(
        "--lookback", type=int, default=None,
        help="Override lookback hours (must match training config)"
    )
    args = parser.parse_args()

    # Resolve checkpoint glob
    matches = glob.glob(args.checkpoint)
    if not matches:
        print(f"ERROR: No checkpoint found matching '{args.checkpoint}'")
        sys.exit(1)
    ckpt_path = Path(matches[0])
    print(f"  Checkpoint: {ckpt_path}")

    # Load config — must match what the h=1 model was trained with
    model_config_name = "lstm.yaml" if args.model_type == "lstm" else "seq2seq.yaml"
    config_files = [
        str(PROJECT_ROOT / "configs" / "training.yaml"),
        str(PROJECT_ROOT / "configs" / "data.yaml"),
        str(PROJECT_ROOT / "configs" / model_config_name),
    ]
    if args.experiment:
        config_files.append(args.experiment)

    config = load_config(config_files)

    # Override lookback if specified (must match what the model was trained with)
    if args.lookback:
        config["data"]["lookback_hours"] = args.lookback

    # Force h=1 for the data module (we create h=1 windows for the model)
    config_h1 = copy.deepcopy(config)
    config_h1["data"]["forecast_horizon_hours"] = 1

    # Also create h=target_horizon windows to get ground truth
    config_h24 = copy.deepcopy(config)
    config_h24["data"]["forecast_horizon_hours"] = args.target_horizon

    seed_everything(config["training"]["seed"])

    # Setup h=1 data to get windows
    dm_h1 = CO2DataModule(config_h1)
    dm_h1.setup()

    # Setup h=target_horizon data to get ground truth
    dm_h24 = CO2DataModule(config_h24)
    dm_h24.setup()

    assert dm_h1.test_dataset is not None
    assert dm_h24.test_dataset is not None

    X_test_h1 = dm_h1.test_dataset.X.numpy()
    y_test_h24 = dm_h24.test_dataset.y.numpy()

    lookback = dm_h1.lookback_steps
    n_features = X_test_h1.shape[2]

    print(f"\n{'='*60}")
    print(f"  Recursive Forecasting")
    print(f"  Model: {args.model_type}")
    print(f"  Lookback: {lookback} steps")
    print(f"  Target horizon: {args.target_horizon} steps")
    print(f"  Test windows (h=1): {X_test_h1.shape[0]}")
    print(f"  Test windows (h={args.target_horizon}): {y_test_h24.shape[0]}")
    print(f"{'='*60}\n")

    # Load model from checkpoint
    if args.model_type == "lstm":
        model = LSTMForecaster.load_from_checkpoint(
            ckpt_path, config=config_h1
        )
    else:
        model = Seq2SeqForecaster.load_from_checkpoint(
            ckpt_path, config=config_h1
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Run recursive prediction
    t0 = time.time()
    y_pred_scaled = recursive_predict(
        model=model,
        X_test_full=X_test_h1,
        y_test_full=y_test_h24,
        lookback=lookback,
        target_horizon=args.target_horizon,
        target_col_idx=-1,  # CO2 is the last column
        device=device,
    )
    elapsed = time.time() - t0
    print(f"\n  Recursive prediction completed in {elapsed:.1f}s")

    # The recursive predictions cover samples 0..n_valid-1
    # The h=24 ground truth also starts from a slightly different offset
    # since h=24 windows need more data at the end.
    # Align them: both should correspond to the same starting timesteps.
    n_valid = y_pred_scaled.shape[0]

    # h=1 windows: each window i starts at index i in the scaled data
    # h=24 windows: each window j also starts at index j
    # But h=1 has more windows (since horizon is shorter).
    # The first n_valid recursive predictions correspond to h=1 windows 0..n_valid-1
    # which should align with h=24 windows 0..n_valid-1 (same starting points)
    y_true_scaled = y_test_h24[:n_valid]

    # Inverse scale
    assert dm_h1.target_scaler is not None
    y_pred = inverse_scale_target(y_pred_scaled, dm_h1.target_scaler)
    y_true = inverse_scale_target(y_true_scaled, dm_h1.target_scaler)

    # Compute metrics
    metrics = compute_metrics(y_true, y_pred)

    # Determine model name and output directory
    exp_name = config.get("experiment", {}).get("name", "")
    model_label = f"{args.model_type.upper()}_recursive"
    model_name = f"{exp_name}_{model_label}_h{args.target_horizon}" if exp_name else f"{model_label}_h{args.target_horizon}"

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config["training"]["results_dir"]) / f"{model_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    save_metrics(metrics, model_name, run_dir / "metrics.json",
                 experiment_info=config.get("experiment"))

    # Save predictions
    np.savez(run_dir / "predictions.npz", y_true=y_true, y_pred=y_pred)

    # Plots
    plots_dir = run_dir / "plots"
    plot_predictions_vs_actual(y_true, y_pred, model_name, plots_dir / "predictions.png")
    plot_scatter(y_true, y_pred, model_name, plots_dir / "scatter.png")
    plot_residuals(y_true, y_pred, model_name, plots_dir / "residuals.png")

    print(f"\nAll outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()
